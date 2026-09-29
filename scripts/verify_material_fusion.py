"""Bounded verification harness for a material-fusion checkpoint.

Runs the checks the experiment requires beyond unit tests, on the actual
checkpoint (optionally on GPU):

1. data admission: the evaluated dataset is re-verified read-only against
   the audit and the fingerprints recorded in the run config/checkpoint.
2. pretrained SHA256: the pretrained file's digest must match the recorded
   value before anything else runs.
3. stage evidence: the real training-path snapshots (init / post-warmup /
   post-finetune, written by ``train_material_fusion.py
   --stage-snapshot-dir``) prove what actually trained — the pretrained
   backbone stays frozen during warmup, the FRM/FFM layers and the head
   train in warmup, and the backbone unfreezes during finetune. A
   checkpoint without this evidence is reported as unprovable and the
   verification fails; comparing against a different random
   initialization is never used as evidence.
4. gradient finiteness: recorded per epoch by the production training
   path (``metrics_history.jsonl``).
5. reload: save/reload raw logits on the same device, dtype and eval
   state (FP32, atol=1e-5, rtol=0).
6. exact fallback routing: forced explicit polar-invalid flags on real
   val batches; invalid rows must equal the fallback's raw logits
   bit-for-bit and valid rows the model's, order preserved, no tolerance.

Any failed check makes the process exit non-zero and is listed in the
report's ``failed_items``.

Usage:
    python scripts/verify_material_fusion.py \
        --checkpoint runs/train/<id>/material_fusion/last.pt \
        --stage-snapshots analysis/runs/cmx_<id>/stage_snapshots \
        --device 0 --limit-batches 2 \
        --output analysis/runs/cmx_<id>/material_fusion_verify.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import QUALITY_VECTOR_KEYS
from core.material_fusion_dataset import MaterialFusionDataset, admit_material_dataset
from models.material_fusion import (
    STAGE_SNAPSHOT_FORMAT,
    build_material_model,
    check_run_config_checkpoint_consistency,
    expected_stage_for_epoch,
    load_material_checkpoint,
    load_pretrained_into_model,
    read_material_checkpoint_metadata,
    save_material_checkpoint,
    sha256_of_file,
    validate_stage_snapshot_metadata,
    validate_variant,
)
from models.polar_fusion import read_manifest_split, torch_device_name
from scripts.eval_material_fusion import load_fallback_gray, resolve_eval_config

VALID_RATIO_INDEX = QUALITY_VECTOR_KEYS.index("valid_ratio")

DEFAULT_RELOAD_ATOL = 1e-5

PARAM_GROUPS = ("backbone", "fusion", "head")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-config", default=None,
                        help="Resolved run configuration (default: run_config.json "
                             "next to the checkpoint, when present).")
    parser.add_argument("--config", default="configs/experiments/material_fusion_v1.yaml",
                        help="Experiment template used only when no run config exists.")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--pretrained", default=None,
                        help="Pretrained MiT-B0 path override (defaults to the run config).")
    parser.add_argument("--fallback-weights", default=None,
                        help="Historical gray fallback override (defaults to the run config).")
    parser.add_argument("--stage-snapshots", default=None,
                        help="Stage snapshot directory (default: <checkpoint dir>/stage_snapshots).")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="Cap the routing-check val batches (0 = all).")
    parser.add_argument("--reload-atol", type=float, default=DEFAULT_RELOAD_ATOL,
                        help="Absolute tolerance for the reload comparison "
                             "(rtol is fixed at 0; do not relax without investigating).")
    parser.add_argument("--output", default=None,
                        help="JSON report path (default: <checkpoint dir>/verify_report.json).")
    return parser.parse_args(argv)


def _group_maps(model) -> tuple[dict[str, str], dict[str, str]]:
    """name -> protocol group for parameters and for buffers."""
    groups = model.parameter_groups()
    by_id: dict[int, str] = {}
    for group, params in groups.items():
        for param in params:
            by_id[id(param)] = group
    param_map = {name: by_id[id(param)] for name, param in model.named_parameters()}

    module_groups: dict[str, str] = {}
    for module_name, _ in model.encoder.named_children():
        top = module_name.split(".")[0]
        module_groups[module_name] = (
            "fusion" if top.startswith(("FRMs", "FFMs")) else "backbone"
        )
    buffer_map: dict[str, str] = {}
    for name, _ in model.named_buffers():
        parts = name.split(".")
        if parts[0] == "encoder" and len(parts) > 1:
            buffer_map[name] = module_groups.get(parts[1], "backbone")
        else:
            buffer_map[name] = "head"
    return param_map, buffer_map


def compare_states(model, before: dict, after: dict) -> dict:
    """Strict per-group parameter/buffer update summary between two states.

    Structural problems raise ``ValueError`` instead of being skipped:
    different key sets (missing or extra tensors), shape or dtype
    mismatches, non-finite floating values, and states that do not cover
    the model's parameters/buffers exactly (so an empty or partial group
    can never produce a vacuous "unchanged" verdict). The changed/unchanged
    decision uses ``torch.equal`` (exact, no float32 downcast); deltas are
    reported in float64.
    """
    param_map, buffer_map = _group_maps(model)
    if set(before) != set(after):
        missing = sorted(set(before) - set(after))
        extra = sorted(set(after) - set(before))
        raise ValueError(
            f"state key sets differ (missing={missing[:5]}, extra={extra[:5]})"
        )
    expected_keys = set(param_map) | set(buffer_map)
    missing = sorted(expected_keys - set(before))
    extra = sorted(set(before) - expected_keys)
    if missing:
        raise ValueError(f"state is missing model tensors: {missing[:5]}")
    if extra:
        raise ValueError(f"state has tensors unknown to the model: {extra[:5]}")

    for name, value in before.items():
        other = after[name]
        if value.shape != other.shape:
            raise ValueError(
                f"{name}: shape mismatch {tuple(value.shape)} vs {tuple(other.shape)}"
            )
        if value.dtype != other.dtype:
            raise ValueError(
                f"{name}: dtype mismatch {value.dtype} vs {other.dtype}"
            )
        if (value.is_floating_point() or value.is_complex()) and (
            not torch.isfinite(value).all() or not torch.isfinite(other).all()
        ):
            raise ValueError(f"{name}: non-finite values in compared state")

    def summarize(name_map: dict[str, str]) -> dict[str, dict]:
        summary = {
            group: {"changed": 0, "total": 0, "max_delta": 0.0} for group in PARAM_GROUPS
        }
        for name, group in name_map.items():
            summary[group]["total"] += 1
            if not torch.equal(before[name], after[name]):
                summary[group]["changed"] += 1
            delta = float(
                (after[name].double() - before[name].double()).abs().max().item()
            )
            summary[group]["max_delta"] = max(summary[group]["max_delta"], delta)
        return summary

    return {"params": summarize(param_map), "buffers": summarize(buffer_map)}


def _require_groups_present(model, variant: str, param_map: dict, buffer_map: dict) -> None:
    """No vacuous verdicts: the groups a variant must have are non-empty."""
    from core.material_fusion_dataset import variant_is_dual

    for group in ("backbone", "head"):
        if not any(g == group for g in param_map.values()):
            raise ValueError(f"model has no {group} parameters; refusing to compare")
    if variant_is_dual(variant) and not any(g == "fusion" for g in param_map.values()):
        raise ValueError(
            f"variant {variant!r} must have fusion (FRM/FFM) parameters; "
            "an empty fusion group cannot pass a freeze check"
        )


def _load_snapshot(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" not in payload:
        raise ValueError(f"stage snapshot {path} has no state_dict")
    return payload


def _legacy_stage_check(snapshot_dir: Path, checkpoint_path: Path) -> dict:
    """Read-only, clearly-labeled checks for pre-v2 evidence directories.

    Reports what the snapshot files alone can show (internal state
    progression) and what is missing (run identity, digests, checkpoint
    binding). It never counts as strict acceptance.
    """
    findings: list[dict] = []
    available = []
    for role, filename in (("init", "init.pt"), ("post_warmup", "post_warmup.pt"),
                           ("post_finetune", "post_finetune.pt")):
        if (snapshot_dir / filename).is_file():
            available.append(role)
    findings.append({
        "check": "snapshot_files_present",
        "passed": bool(available),
        "detail": f"present roles: {available}",
    })
    missing_evidence = [
        "run identity (run_id/variant/seed/data fingerprint/pretrained SHA)",
        "per-file sha256 digests",
        "checkpoint digest binding",
        "run protocol (epochs/warmup_epochs/smoke/limit_batches)",
    ]
    return {
        "findings": findings,
        "present_roles": available,
        "missing_evidence": missing_evidence,
        "checkpoint": str(checkpoint_path),
    }


def verify_stage_updates(
    snapshot_dir: str | Path,
    checkpoint_path: str | Path,
    model=None,
    checkpoint_metadata=None,
) -> dict:
    """Prove what trained, bound to this run and this checkpoint.

    Requires v2 evidence metadata: identity fields compared against the
    checkpoint metadata, per-file digests recomputed, and the verified
    checkpoint bound by digest to a recorded role. ``last`` must equal the
    post-finetune snapshot bit-for-bit; an early ``best`` must match its
    recorded epoch/stage and must not be a copy of the final state. If a
    ``model`` is passed, its actual state must match the checkpoint. A
    snapshot directory without v2 metadata is reported as legacy and can
    never pass strict verification.
    """
    snapshot_dir = Path(snapshot_dir)
    checkpoint_path = Path(checkpoint_path)
    checks: list[dict] = []
    failed: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            failed.append(f"{name}: {detail}")

    meta_path = snapshot_dir / "stage_snapshots.json"
    if not meta_path.is_file():
        legacy = _legacy_stage_check(snapshot_dir, checkpoint_path)
        return {
            "available": False,
            "passed": False,
            "format": "legacy",
            "format_version": 0,
            "checkpoint_role": "",
            "reason": (
                "stage evidence metadata (stage_snapshots.json) is missing; "
                "strict identity/digest verification is unavailable"
            ),
            "legacy_findings": legacy,
            "checks": checks,
            "failed_items": [
                "legacy stage evidence: run identity, digests and checkpoint "
                "binding are unprovable (missing v2 metadata)"
            ],
        }

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "passed": False,
            "format": "invalid",
            "format_version": 0,
            "checkpoint_role": "",
            "reason": f"stage evidence metadata is not valid JSON: {exc}",
            "checks": checks,
            "failed_items": [f"stage evidence metadata is not valid JSON: {exc}"],
        }
    try:
        evidence = validate_stage_snapshot_metadata(payload)
    except ValueError as exc:
        # A metadata file from the pre-v2 format is legacy evidence: report
        # what is missing instead of presenting it as a strict v2 failure.
        if isinstance(payload, dict) and "format" not in payload:
            legacy = _legacy_stage_check(snapshot_dir, checkpoint_path)
            legacy["missing_fields"] = str(exc)
            return {
                "available": False,
                "passed": False,
                "format": "legacy",
                "format_version": 0,
                "checkpoint_role": "",
                "reason": (
                    "stage evidence uses the pre-v2 metadata format; strict "
                    "identity/digest verification is unavailable"
                ),
                "legacy_findings": legacy,
                "checks": checks,
                "failed_items": [
                    "legacy stage evidence: run identity, digests and checkpoint "
                    f"binding are unprovable (pre-v2 metadata: {exc})"
                ],
            }
        return {
            "available": False,
            "passed": False,
            "format": "invalid",
            "format_version": 0,
            "checkpoint_role": "",
            "reason": f"invalid stage evidence metadata: {exc}",
            "checks": checks,
            "failed_items": [f"stage evidence metadata invalid: {exc}"],
        }

    if checkpoint_metadata is None:
        try:
            checkpoint_metadata = read_material_checkpoint_metadata(checkpoint_path)
        except (ValueError, RuntimeError, OSError) as exc:
            return {
                "available": False,
                "passed": False,
                "format": STAGE_SNAPSHOT_FORMAT,
                "format_version": evidence["format_version"],
                "checkpoint_role": "",
                "reason": f"the verified checkpoint is not a valid material-fusion checkpoint: {exc}",
                "checks": checks,
                "failed_items": [
                    f"verified checkpoint invalid or not the recorded file: {exc}"
                ],
            }
    meta = checkpoint_metadata.__dict__ if hasattr(checkpoint_metadata, "__dict__") else dict(checkpoint_metadata)

    # --- identity: evidence must belong to this checkpoint's run ---------
    for field in ("variant", "imgsz", "preprocessing_version", "upstream_commit",
                  "pretrained_sha256", "data_fingerprint", "seed", "smoke",
                  "limit_batches", "physical_batch", "accum_steps", "effective_batch"):
        record(
            f"identity_{field}",
            evidence[field] == meta[field],
            f"evidence {field}={evidence[field]!r} vs checkpoint {meta[field]!r}",
        )
    # run id is derivable when the checkpoint sits in the standard layout.
    try:
        expected_run_id = checkpoint_path.resolve().parents[1].name
        if checkpoint_path.resolve().parent.name == "material_fusion":
            record(
                "identity_run_id",
                evidence["run_id"] == expected_run_id,
                f"evidence run_id={evidence['run_id']!r} vs checkpoint run "
                f"{expected_run_id!r}",
            )
    except IndexError:  # pragma: no cover - defensive
        pass

    # --- checkpoint digest binding ---------------------------------------
    checkpoint_digest = sha256_of_file(checkpoint_path)
    role = ""
    for candidate in ("last", "best"):
        if evidence["checkpoints"][candidate]["sha256"] == checkpoint_digest:
            role = candidate
    record(
        "checkpoint_digest_recorded",
        bool(role),
        f"checkpoint digest {checkpoint_digest[:12]}… matches role {role or '(none)'}",
    )
    if not role:
        return {
            "available": True,
            "passed": False,
            "format": STAGE_SNAPSHOT_FORMAT,
            "format_version": evidence["format_version"],
            "checkpoint_role": "",
            "reason": "the verified checkpoint is not the file this evidence records",
            "checks": checks,
            "failed_items": failed,
        }
    record_entry = evidence["checkpoints"][role]
    record(
        "checkpoint_role_epoch_stage",
        record_entry["epoch"] == meta["epoch"] and record_entry["stage"] == meta["stage"],
        f"evidence records {role} epoch/stage "
        f"{record_entry['epoch']}/{record_entry['stage']}, checkpoint has "
        f"{meta['epoch']}/{meta['stage']}",
    )
    if role == "last":
        record(
            "last_checkpoint_is_final_epoch",
            meta["epoch"] == evidence["epochs"]
            and meta["stage"] == expected_stage_for_epoch(
                evidence["epochs"], evidence["warmup_epochs"]
            ),
            f"last checkpoint epoch/stage {meta['epoch']}/{meta['stage']} vs run "
            f"epochs={evidence['epochs']} warmup_epochs={evidence['warmup_epochs']}",
        )

    # --- snapshot files: existence and digests ---------------------------
    snapshot_states: dict[str, dict] = {}
    for role_name, entry in evidence["snapshots"].items():
        path = snapshot_dir / entry["file"]
        if not path.is_file():
            record(f"snapshot_{role_name}_present", False, f"{entry['file']} missing")
            continue
        digest = sha256_of_file(path)
        if digest != entry["sha256"]:
            record(
                f"snapshot_{role_name}_digest", False,
                f"{entry['file']} digest {digest[:12]}… does not match the "
                f"recorded {entry['sha256'][:12]}…",
            )
            continue
        record(f"snapshot_{role_name}_digest", True, f"{entry['file']} digest matches")
        snapshot = _load_snapshot(path)
        record(
            f"snapshot_{role_name}_epoch_stage",
            snapshot.get("epoch") == entry["epoch"] and snapshot.get("stage") == entry["stage"],
            f"{entry['file']} records epoch/stage "
            f"{snapshot.get('epoch')}/{snapshot.get('stage')} vs evidence "
            f"{entry['epoch']}/{entry['stage']}",
        )
        snapshot_states[role_name] = snapshot["state_dict"]

    if failed:
        return {
            "available": True,
            "passed": False,
            "format": STAGE_SNAPSHOT_FORMAT,
            "format_version": evidence["format_version"],
            "checkpoint_role": role,
            "reason": "; ".join(failed),
            "checks": checks,
            "failed_items": failed,
        }

    # --- model binding ----------------------------------------------------
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_state = checkpoint_payload["state_dict"]
    if model is not None:
        try:
            model_state = model.state_dict()
            if set(model_state) != set(checkpoint_state):
                raise ValueError(
                    "in-memory model has different tensor names than the checkpoint"
                )
            problems: list[str] = []
            for name in checkpoint_state:
                model_tensor = model_state[name]
                checkpoint_tensor = checkpoint_state[name]
                if model_tensor.shape != checkpoint_tensor.shape:
                    problems.append(
                        f"{name}: shape {tuple(model_tensor.shape)} vs "
                        f"{tuple(checkpoint_tensor.shape)}"
                    )
                    continue
                if model_tensor.dtype != checkpoint_tensor.dtype:
                    problems.append(
                        f"{name}: dtype {model_tensor.dtype} vs "
                        f"{checkpoint_tensor.dtype}"
                    )
                    continue
                if (model_tensor.is_floating_point() or model_tensor.is_complex()) and not (
                    torch.isfinite(model_tensor).all()
                ):
                    problems.append(f"{name}: non-finite values in the in-memory model")
                    continue
                # Compare detached CPU copies: the checkpoint was loaded on
                # CPU and the model may live on CUDA. Neither the model's
                # device/dtype/mode nor its tensors are modified.
                if not torch.equal(
                    model_tensor.detach().cpu(), checkpoint_tensor.detach().cpu()
                ):
                    problems.append(f"{name}: values differ")
            record(
                "in_memory_model_matches_checkpoint",
                not problems,
                "; ".join(problems[:3]) if problems
                else "model state equals the checkpoint (bit-exact)",
            )
        except ValueError as exc:
            record("in_memory_model_matches_checkpoint", False, str(exc))
        if failed:
            return {
                "available": True,
                "passed": False,
                "format": STAGE_SNAPSHOT_FORMAT,
                "format_version": evidence["format_version"],
                "checkpoint_role": role,
                "reason": "; ".join(failed),
                "checks": checks,
                "failed_items": failed,
            }

    # --- checkpoint <-> final snapshot state binding ----------------------
    grouping_model = model
    if grouping_model is None:
        grouping_model = build_material_model(
            evidence["variant"], imgsz=evidence["imgsz"],
            drop_path_rate=meta.get("drop_path_rate", 0.0)
            if isinstance(meta, dict) else 0.0,
        )
    try:
        _require_groups_present(
            grouping_model, evidence["variant"],
            *_group_maps(grouping_model),
        )
        final_state = snapshot_states["post_finetune"]
        if role == "last":
            diff = compare_states(grouping_model, checkpoint_state, final_state)
            unchanged = all(
                diff[section][group]["changed"] == 0
                for section in ("params", "buffers") for group in PARAM_GROUPS
            )
            record(
                "last_checkpoint_matches_final_snapshot",
                unchanged,
                "checkpoint state differs from post_finetune"
                if not unchanged else "bit-identical to post_finetune",
            )
        else:
            best_epoch = evidence["checkpoints"]["best"]["epoch"]
            if best_epoch == evidence["epochs"]:
                diff = compare_states(grouping_model, checkpoint_state, final_state)
                unchanged = all(
                    diff[section][group]["changed"] == 0
                    for section in ("params", "buffers") for group in PARAM_GROUPS
                )
                record(
                    "best_checkpoint_matches_final_snapshot",
                    unchanged,
                    "best is the final epoch but differs from post_finetune"
                    if not unchanged else "bit-identical to post_finetune",
                )
            else:
                identical = all(
                    torch.equal(checkpoint_state[name], final_state[name])
                    for name in checkpoint_state
                ) and set(checkpoint_state) == set(final_state)
                record(
                    "early_best_differs_from_final",
                    not identical,
                    "early best checkpoint is bit-identical to the final "
                    "snapshot; the final state cannot stand in for an earlier "
                    "best epoch",
                )

        # --- stage progression -------------------------------------------
        # Branch on the validated run protocol: only warmup_epochs > 0 has a
        # warmup stage. Missing warmup evidence is never read as "no warmup".
        warmup_applicable = evidence["warmup_epochs"] > 0
        if warmup_applicable:
            warmup = compare_states(grouping_model, snapshot_states["init"],
                                    snapshot_states["post_warmup"])
            record(
                "warmup_backbone_frozen",
                warmup["params"]["backbone"]["changed"] == 0,
                f"backbone params changed {warmup['params']['backbone']['changed']}/"
                f"{warmup['params']['backbone']['total']} during warmup (must be 0)",
            )
            record(
                "warmup_backbone_buffers_unchanged",
                warmup["buffers"]["backbone"]["changed"] == 0,
                f"backbone buffers changed {warmup['buffers']['backbone']['changed']}/"
                f"{warmup['buffers']['backbone']['total']} during warmup (must be 0)",
            )
            if warmup["params"]["fusion"]["total"]:
                record(
                    "warmup_fusion_trained",
                    warmup["params"]["fusion"]["changed"] > 0,
                    f"fusion params changed {warmup['params']['fusion']['changed']}/"
                    f"{warmup['params']['fusion']['total']} during warmup (must be > 0)",
                )
                record(
                    "warmup_fusion_buffers_updated",
                    warmup["buffers"]["fusion"]["changed"] > 0,
                    f"fusion buffers changed {warmup['buffers']['fusion']['changed']}/"
                    f"{warmup['buffers']['fusion']['total']} during warmup "
                    "(BatchNorm statistics must update)",
                )
            record(
                "warmup_head_trained",
                warmup["params"]["head"]["changed"] > 0,
                f"head params changed {warmup['params']['head']['changed']}/"
                f"{warmup['params']['head']['total']} during warmup (must be > 0)",
            )
            finetune_from = snapshot_states["post_warmup"]
            finetune_detail = "post_warmup -> post_finetune"
        else:
            record(
                "warmup_not_applicable",
                True,
                "warmup_epochs=0: the run protocol has no warmup stage; no "
                "warmup checks are claimed",
            )
            finetune_from = snapshot_states["init"]
            finetune_detail = "init -> post_finetune (warmup_epochs=0)"

        finetune = compare_states(grouping_model, finetune_from,
                                  snapshot_states["post_finetune"])
        record(
            "finetune_backbone_unfroze_and_trained",
            finetune["params"]["backbone"]["changed"] > 0,
            f"backbone params changed {finetune['params']['backbone']['changed']}/"
            f"{finetune['params']['backbone']['total']} during finetune "
            f"({finetune_detail}; must be > 0)",
        )
        if finetune["params"]["fusion"]["total"]:
            record(
                "finetune_fusion_trained",
                finetune["params"]["fusion"]["changed"] > 0,
                f"fusion params changed {finetune['params']['fusion']['changed']}/"
                f"{finetune['params']['fusion']['total']} during finetune (must be > 0)",
            )
        record(
            "finetune_head_trained",
            finetune["params"]["head"]["changed"] > 0,
            f"head params changed {finetune['params']['head']['changed']}/"
            f"{finetune['params']['head']['total']} during finetune (must be > 0)",
        )
    except ValueError as exc:
        record("state_comparison", False, str(exc))

    return {
        "available": True,
        "passed": not failed,
        "format": STAGE_SNAPSHOT_FORMAT,
        "format_version": evidence["format_version"],
        "checkpoint_role": role,
        "run_id": evidence["run_id"],
        "warmup_applicable": bool(evidence["warmup_epochs"] > 0),
        "reason": "" if not failed else "; ".join(failed),
        "checks": checks,
        "failed_items": failed,
        "snapshots": {
            role_name: str(snapshot_dir / entry["file"])
            for role_name, entry in evidence["snapshots"].items()
        },
    }


def verify_gradient_finiteness(
    history_path: str | Path,
    epochs: int,
    warmup_epochs: int,
) -> dict:
    """The training history must match the run protocol exactly.

    One row per epoch (1..``epochs``, no missing/duplicate rows), the
    protocol stage for each epoch number, and ``gradients_finite`` as a
    strict boolean True in every row. A missing file, missing field,
    non-bool or False value all fail.
    """
    path = Path(history_path)
    failed: list[str] = []
    if not path.is_file():
        return {
            "available": False,
            "passed": False,
            "rows": 0,
            "failed_items": [f"training history {path} is missing"],
        }
    entries = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            failed.append(f"history line {line_number} is not valid JSON: {exc}")
    if not failed:
        seen_epochs: list[int] = []
        for entry in entries:
            epoch = entry.get("epoch")
            stage = entry.get("stage")
            finite = entry.get("gradients_finite")
            if type(epoch) is not int:
                failed.append(f"history row has non-int epoch: {epoch!r}")
                continue
            if type(finite) is not bool:
                failed.append(
                    f"epoch {epoch}: gradients_finite must be a bool, got {finite!r}"
                )
            elif finite is not True:
                failed.append(f"epoch {epoch}: gradients_finite is False")
            if stage != expected_stage_for_epoch(epoch, warmup_epochs):
                failed.append(
                    f"epoch {epoch}: stage {stage!r} does not match the protocol "
                    f"stage {expected_stage_for_epoch(epoch, warmup_epochs)!r}"
                )
            seen_epochs.append(epoch)
        expected_epochs = list(range(1, epochs + 1))
        if sorted(seen_epochs) != expected_epochs:
            failed.append(
                f"history epochs {sorted(seen_epochs)} do not match the run's "
                f"epochs {expected_epochs}"
            )
    return {
        "available": True,
        "passed": not failed,
        "rows": len(entries),
        "failed_items": failed,
    }


def verify_pretrained_sha(path: str | Path, expected_sha256: str) -> dict:
    """The pretrained file's SHA256 against the recorded value."""
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    return {
        "path": str(path),
        "expected": expected_sha256,
        "actual": actual,
        "matches": bool(expected_sha256) and actual == expected_sha256,
    }


@torch.no_grad()
def route_batch(model, fallback, rgb, aux, invalid):
    """True-bypass routing for one batch, preserving sample order.

    Invalid samples are forwarded through the fallback only, valid samples
    through the model only; results are scattered back by absolute index.
    Returns ``(predictions, routed_logits)``.
    """
    invalid = invalid.bool()
    valid = ~invalid
    valid_logits = None
    invalid_logits = None
    if valid.any():
        sub = rgb[valid]
        valid_logits = model(sub, aux[valid]) if aux is not None else model(sub)
    if invalid.any():
        invalid_logits = fallback(rgb[invalid])
    reference = valid_logits if valid_logits is not None else invalid_logits
    routed_logits = torch.empty(
        rgb.shape[0], reference.shape[1], dtype=reference.dtype, device=reference.device
    )
    if valid_logits is not None:
        routed_logits[valid] = valid_logits
    if invalid_logits is not None:
        routed_logits[invalid] = invalid_logits
    return routed_logits.argmax(dim=1), routed_logits


@torch.no_grad()
def verify_reload(
    checkpoint: str | Path,
    variant: str,
    imgsz: int,
    device: str,
    atol: float = DEFAULT_RELOAD_ATOL,
    reference_state: dict | None = None,
    seed: int = 2026,
) -> dict:
    """Save/reload raw-logits comparison on the same device/dtype/eval state."""
    validate_variant(variant)
    model, _, _ = load_material_checkpoint(checkpoint, variant=variant, imgsz=imgsz)
    model = model.to(device).eval()

    generator = torch.Generator().manual_seed(seed)
    rgb = torch.rand(2, 3, imgsz, imgsz, generator=generator)
    aux = torch.rand(2, 3, imgsz, imgsz, generator=generator)

    def forward(model_to_run):
        model_to_run = model_to_run.to(device).eval()
        with torch.no_grad():
            return model_to_run(rgb.to(device), aux.to(device)) if model_to_run.is_dual \
                else model_to_run(rgb.to(device))

    logits = forward(model)
    with tempfile.TemporaryDirectory(prefix="material_fusion_reload_") as tmp:
        metadata = read_material_checkpoint_metadata(checkpoint)
        roundtrip = save_material_checkpoint(Path(tmp) / "reload.pt", model, metadata)
        reloaded, _, _ = load_material_checkpoint(roundtrip, variant=variant, imgsz=imgsz)
        reloaded_logits = forward(reloaded)

    max_abs_diff = float((logits - reloaded_logits).abs().max().item())
    reference_diff = None
    if reference_state is not None:
        reference_model = build_material_model(variant, imgsz=imgsz)
        reference_model.load_state_dict(reference_state)
        reference_logits = forward(reference_model)
        reference_diff = float((logits - reference_logits).abs().max().item())

    within = max_abs_diff <= atol
    if reference_diff is not None:
        within = within and reference_diff <= atol
    return {
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_vs_reference": reference_diff,
        "within_tolerance": within,
        "atol": atol,
        "rtol": 0.0,
        "device": device,
        "dtype": str(logits.dtype).replace("torch.", ""),
        "eval_mode": True,
    }


@torch.no_grad()
def verify_fallback_routing(
    model,
    fallback,
    loader,
    device: str,
    forced_invalid_indices: list[int] | None = None,
    limit_batches: int = 0,
) -> dict:
    """Exact routing verification with explicit polar-invalid flags.

    Invalid rows must equal the fallback's raw logits bit-for-bit and valid
    rows the model's; no numeric tolerance is used anywhere in this check.
    """
    forced = set(forced_invalid_indices or [])
    model = model.to(device).eval()
    fallback = fallback.to(device).eval()
    offset = 0
    invalid_samples = 0
    valid_samples = 0
    invalid_exact = True
    valid_exact = True
    for batch_index, batch in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        if model.is_dual:
            rgb, aux, quality, labels = batch
        else:
            rgb, quality, labels = batch
            aux = None
        rgb = rgb.to(device)
        aux = aux.to(device) if aux is not None else None
        indices = torch.arange(offset, offset + rgb.shape[0])
        forced_mask = torch.tensor([int(i) in forced for i in indices])
        invalid = (quality[:, VALID_RATIO_INDEX] <= 0) | forced_mask
        offset += rgb.shape[0]

        preds, routed_logits = route_batch(model, fallback, rgb, aux, invalid)
        invalid_samples += int(invalid.sum())
        valid_samples += int((~invalid).sum())
        if invalid.any():
            expected = fallback(rgb[invalid])
            invalid_exact = invalid_exact and bool(torch.equal(routed_logits[invalid], expected))
            invalid_exact = invalid_exact and bool(
                torch.equal(preds[invalid], expected.argmax(dim=1))
            )
        if (~invalid).any():
            sub = rgb[~invalid]
            expected = model(sub, aux[~invalid]) if aux is not None else model(sub)
            valid_exact = valid_exact and bool(torch.equal(routed_logits[~invalid], expected))
            valid_exact = valid_exact and bool(
                torch.equal(preds[~invalid], expected.argmax(dim=1))
            )
    return {
        "invalid_samples": invalid_samples,
        "valid_samples": valid_samples,
        "invalid_rows_exact": invalid_exact,
        "valid_rows_exact": valid_exact,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    output = (
        Path(args.output) if args.output else checkpoint.parent / "verify_report.json"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        print(
            f"verification refused: report {output} already exists; use a new "
            "--output path instead of overwriting a previous report",
            file=sys.stderr,
        )
        return 2

    try:
        device = torch_device_name(args.device)
    except ValueError as exc:
        print(f"unsupported --device {args.device!r}: {exc}", file=sys.stderr)
        return 2

    try:
        metadata = read_material_checkpoint_metadata(checkpoint)
        config, config_provenance, assert_epoch_stage = resolve_eval_config(
            args, checkpoint, metadata
        )
        variant = config["variant"]
        # The same strict config <-> checkpoint contract evaluation uses.
        consistency = check_run_config_checkpoint_consistency(
            config, metadata.__dict__, assert_epoch_stage=assert_epoch_stage
        )

        data_root = config["data"]
        if not Path(data_root).is_absolute():
            data_root = PROJECT_ROOT / data_root
        expected_fingerprint = config["data_fingerprint"] or metadata.data_fingerprint
        data_report = admit_material_dataset(data_root, expected_fingerprint)
        if metadata.data_fingerprint and data_report["audit_fingerprint"] != metadata.data_fingerprint:
            raise ValueError(
                f"checkpoint records dataset fingerprint {metadata.data_fingerprint} "
                f"but the verified data is {data_report['audit_fingerprint']}"
            )
        if config["preprocessing_version"] != metadata.preprocessing_version:
            raise ValueError(
                f"config preprocessing_version {config['preprocessing_version']!r} "
                f"does not match the checkpoint's {metadata.preprocessing_version!r}"
            )

        pretrained = Path(args.pretrained or config["pretrained"])
        if not pretrained.is_absolute():
            pretrained = PROJECT_ROOT / pretrained
        expected_sha = metadata.pretrained_sha256 or config["pretrained_sha256"]
        sha_report = verify_pretrained_sha(pretrained, expected_sha)
        if not sha_report["matches"]:
            raise ValueError(
                f"pretrained file {pretrained} sha256 {sha_report['actual'] or 'missing'} "
                f"does not match the recorded {expected_sha or '(none)'}"
            )

        fallback_path = Path(args.fallback_weights or config["fallback_weights"])
        if not fallback_path.is_absolute():
            fallback_path = PROJECT_ROOT / fallback_path
    except (ValueError, FileNotFoundError) as exc:
        print(f"verification refused: {exc}", file=sys.stderr)
        return 2

    model, _, _ = load_material_checkpoint(checkpoint, variant=variant, imgsz=metadata.imgsz)
    model = model.to(device)

    snapshot_dir = (
        Path(args.stage_snapshots)
        if args.stage_snapshots
        else checkpoint.parent / "stage_snapshots"
    )
    if not snapshot_dir.is_absolute():
        snapshot_dir = PROJECT_ROOT / snapshot_dir

    report: dict = {
        "checkpoint": str(checkpoint),
        "variant": variant,
        "imgsz": metadata.imgsz,
        "device": device,
        "smoke": metadata.smoke,
        "limit_batches": metadata.limit_batches,
        "epoch": metadata.epoch,
        "stage": metadata.stage,
        "config_source": config_provenance,
        "config_consistency": consistency,
        "data": {
            "root": data_report["data_root"],
            "audit_fingerprint": data_report["audit_fingerprint"],
            "verified_files": data_report["verified_files"],
        },
        "pretrained_sha256": sha_report,
    }

    pretrained_report = load_pretrained_into_model(
        build_material_model(variant, imgsz=metadata.imgsz), pretrained
    )
    report["pretrained_load"] = pretrained_report
    report["reload"] = verify_reload(
        checkpoint, variant=variant, imgsz=metadata.imgsz, device=device,
        atol=args.reload_atol,
    )
    report["stage_updates"] = verify_stage_updates(
        snapshot_dir, checkpoint, model=model, checkpoint_metadata=metadata
    )
    report["gradient_finiteness"] = verify_gradient_finiteness(
        checkpoint.parent / "metrics_history.jsonl",
        epochs=int(config["epochs"]),
        warmup_epochs=int(config["warmup_epochs"]),
    )

    paths, _, _ = read_manifest_split(data_root, args.split)
    loader = torch.utils.data.DataLoader(
        MaterialFusionDataset(paths, variant=variant, imgsz=metadata.imgsz),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 8,
    )
    fallback, fallback_identity = load_fallback_gray(
        fallback_path, device, metadata.imgsz
    )
    forced = [0, 3] if len(paths) > 3 else [0]
    report["fallback_weights"] = str(fallback_path)
    report["fallback_identity"] = fallback_identity
    report["fallback_routing"] = verify_fallback_routing(
        model, fallback, loader, device, forced_invalid_indices=forced,
        limit_batches=args.limit_batches,
    )

    failed_items: list[str] = []
    if config_provenance["source"] != "run_config":
        failed_items.append(
            f"config source is {config_provenance['source']!r}: strict "
            "config/checkpoint consistency is unprovable without the run's "
            "resolved run_config.json"
        )
    if not report["reload"]["within_tolerance"]:
        failed_items.append(
            f"reload: max_abs_diff {report['reload']['max_abs_diff']:.3e} > "
            f"atol {report['reload']['atol']:.1e} (rtol 0)"
        )
    if not (report["fallback_routing"]["invalid_rows_exact"]
            and report["fallback_routing"]["valid_rows_exact"]):
        failed_items.append("fallback routing: rows are not bit-exact")
    failed_items.extend(report["stage_updates"]["failed_items"])
    failed_items.extend(
        f"gradient finiteness: {item}"
        for item in report["gradient_finiteness"]["failed_items"]
    )
    report["failed_items"] = failed_items
    report["passed"] = not failed_items

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verify report: {output}")
    print(
        f"reload max_abs_diff={report['reload']['max_abs_diff']:.3e} "
        f"(within_tolerance={report['reload']['within_tolerance']}) | "
        f"stage_updates={report['stage_updates']['passed']} | "
        f"grads_finite={report['gradient_finiteness']['passed']} | "
        f"fallback exact="
        f"{report['fallback_routing']['invalid_rows_exact'] and report['fallback_routing']['valid_rows_exact']}"
    )
    if failed_items:
        print("FAILED:", file=sys.stderr)
        for item in failed_items:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
