#!/usr/bin/env python
"""material_upsample_impact_v1 — the switchable disparity-upsample fix and
its classification-impact verification (batch 2).

Two explicit upsample versions of the SAME production matcher:

- ``legacy_zero_linear_v1``  — the original path, bitwise unchanged, the
  default everywhere;
- ``valid_weighted_legacy_support_v1`` — the only approved fix: the
  low-resolution SGBM support is unchanged (nearest-valid), but the kept
  positions carry the valid-weight-normalized disparity
  (Dstar = N / W / scale where W > 0).

The evaluation re-uses the SAVED D0 predictions on all 180 original val
frames (never Model A — a recording probe must see zero calls), the fixed
matching from the frozen runtime evidence, and the complete frozen
nine-candidate matrix. The legacy arm must first REPRODUCE the saved
180-frame results (per-prediction classes exactly, logits within the
frozen tolerance) before the fix arm's impact comparison is allowed. No
training, no re-selection, no working-point change; a flat or worse
metric is a valid outcome and is reported as measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.existing_data_protocol import (  # noqa: E402
    BudgetExceeded,
    BudgetWatch,
    sha256_of_file,
)
from core.measurement_diagnostic import (  # noqa: E402
    pipeline_gt_prediction_join,
)
from core.stereo_matching import (  # noqa: E402
    UPSAMPLE_LEGACY_ZERO_LINEAR_V1,
    UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1,
    StereoMatcher,
    StereoMatcherConfig,
    build_horizontal_bands,
    to_gray_u8,
)
from gui.inference_engine import DualStageInferenceEngine  # noqa: E402
from scripts.eval_material_pipeline import (  # noqa: E402
    OPERATING_POINT,
    MATCH_RULE,
    _build_record,
    aggregate_metrics,
    paired_changes,
    write_json_bytes,
    write_jsonl_bytes,
)
from scripts.eval_material_residual_pipeline import (  # noqa: E402
    _SharedGrayExtractor,
    _group_frames,
    _pick,
    build_classifiers,
    load_protocol,
)
from models.material_residual_classifier import (  # noqa: E402
    CLASS_NAMES,
    build_frame_fold_map,
    load_residual_runtime_config,
)

MODES = ("preflight", "evaluate")
ARMS = ("legacy", "fix")
EXIT_OK = 0
EXIT_PARTIAL = 2
EXIT_FAILED = 3
EXIT_USAGE = 4
_REPARSE_FLAG = getattr(__import__("stat"), "FILE_ATTRIBUTE_REPARSE_POINT",
                        0x400)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def is_link_or_reparse(path: Path) -> bool:
    stat = __import__("stat")
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_FLAG)


def dump_json(path: Path, payload: Any) -> None:
    def check(node: Any, where: str) -> None:
        if isinstance(node, float):
            if not np.isfinite(node):
                raise ValueError(f"refusing non-finite float at {where}")
        elif isinstance(node, dict):
            for key, value in node.items():
                check(value, f"{where}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                check(value, f"{where}[{index}]")

    check(payload, path.name)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    lines = []
    for index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(
                    f"refusing non-finite float in row {index} field {key!r}")
        lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Source registry: every consumed input gets an identity at registration
# (before its consuming read wherever this entry controls the order).


class IdentityError(ValueError):
    pass


class SourceRegistry:
    """First-read identities for every consumed input.

    Registration kinds, all resolved from VERIFIED anchors (never from a
    current re-hash posing as history):

    - ``anchor``: the four frozen anchors pinned by this batch's config
      (verified against the pinned digest, then registered with that pin
      as their binding);
    - ``frozen_binding``: the expectation comes from a verified anchor's
      recorded entries (artifact manifest, batch-1 source manifest, the
      frozen runtime config's own file pins, the experiment config's
      reference-run identities);
    - ``experiment_config``: this batch's own experiment config, read
      once to build the run and registered immediately after that first
      read;
    - ``batch2_snapshot``: a file no verified manifest covers (most of
      the 180 val frames' images / D0 masks); a NEW identity is
      snapshotted here, before this batch's consuming reads. This is
      this batch's protection and is never backfilled into any earlier
      run's history.

    Registrations dedupe by path. A later registration of an
    already-registered path is checked against the recorded identity: a
    DIFFERENT frozen expectation is a binding conflict and is rejected.
    """

    def __init__(self, known_bindings: dict[str, dict]) -> None:
        self.bindings = dict(known_bindings)
        self.records: dict[str, dict] = {}

    @staticmethod
    def _check_conflict(rel: str, record: dict,
                        binding: dict | None) -> None:
        if binding is None:
            return
        expected = str(binding.get("expected_sha256", ""))
        if expected and expected != record["sha256"]:
            raise IdentityError(
                f"{rel}: binding conflict - registered sha256 "
                f"{record['sha256']} != the later binding's expectation "
                f"{expected} ({binding.get('source', 'anchor')})")
        expected_bytes = binding.get("expected_bytes")
        if expected_bytes is not None and \
                int(expected_bytes) != record["bytes"]:
            raise IdentityError(
                f"{rel}: binding conflict - registered {record['bytes']} "
                f"bytes != the later binding's expectation "
                f"{int(expected_bytes)}")

    def register(self, path: Path, role: str, *, kind: str | None = None,
                 binding: dict | None = None) -> dict:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        binding = binding if binding is not None else self.bindings.get(rel)
        existing = self.records.get(rel)
        if existing is not None:
            self._check_conflict(rel, existing, binding)
            return existing
        if is_link_or_reparse(path) or is_link_or_reparse(path.parent):
            raise IdentityError(f"{rel}: link or reparse point")
        resolved = path.resolve()
        root = PROJECT_ROOT.resolve()
        if root != resolved and root not in resolved.parents:
            raise IdentityError(f"{rel}: escapes the project root")
        if not path.is_file():
            raise IdentityError(f"{rel}: missing at registration")
        digest = sha256_of_file(path)
        size = path.stat().st_size
        if binding is not None:
            if digest != binding["expected_sha256"]:
                raise IdentityError(
                    f"{rel}: sha256 {digest} != frozen expectation "
                    f"{binding['expected_sha256']} "
                    f"({binding.get('source', 'anchor')}); refusing")
            if binding.get("expected_bytes") is not None and \
                    size != int(binding["expected_bytes"]):
                raise IdentityError(
                    f"{rel}: {size} bytes != frozen expectation")
        if kind is not None:
            registration = kind
        elif role.startswith("anchor:"):
            registration = "anchor"
        elif binding is not None:
            registration = "frozen_binding"
        else:
            registration = "batch2_snapshot"
        record = {
            "path": rel, "role": role, "bytes": size, "sha256": digest,
            "binding": binding, "registration": registration,
        }
        self.records[rel] = record
        return record

    def end_snapshot(self) -> dict[str, dict]:
        """End-of-run identities, tolerant: a missing or unreadable file is
        recorded as a problem status instead of raising, so the manifest
        survives to show the failure."""
        snapshot: dict[str, dict] = {}
        for rel in sorted(self.records):
            path = PROJECT_ROOT / rel
            if not path.is_file():
                snapshot[rel] = {"status": "missing_at_end", "bytes": None,
                                 "sha256": None}
                continue
            try:
                snapshot[rel] = {
                    "status": "ok", "bytes": path.stat().st_size,
                    "sha256": sha256_of_file(path)}
            except OSError as exc:
                snapshot[rel] = {
                    "status": f"read_failed:{type(exc).__name__}",
                    "bytes": None, "sha256": None, "detail": str(exc)}
        return snapshot

    def compare(self) -> dict:
        problems = []
        after = self.end_snapshot()
        rows = []
        by_registration: dict[str, int] = {}
        for rel, before in sorted(self.records.items()):
            post = after[rel]
            status = post["status"]
            if status == "ok":
                unchanged = (before["sha256"] == post["sha256"]
                             and before["bytes"] == post["bytes"])
                if not unchanged:
                    problems.append(f"{rel}: CHANGED after registration")
            else:
                unchanged = False
                problems.append(f"{rel}: {status}")
            binding = before.get("binding")
            registration = before["registration"]
            by_registration[registration] = \
                by_registration.get(registration, 0) + 1
            rows.append({
                "path": rel,
                "role": before["role"],
                "bytes": before["bytes"],
                "sha256_at_registration": before["sha256"],
                "bytes_at_end": post["bytes"],
                "sha256_at_end": post["sha256"],
                "unchanged": unchanged,
                "status": status,
                "registration": registration,
                "binding_kind": ("anchor_or_frozen"
                                 if binding is not None or
                                 registration in ("anchor", "frozen_binding",
                                                  "experiment_config")
                                 else "batch2_snapshot"),
                "binding_source": (binding or {}).get("source"),
            })
        rows_meta = {
            "registered_sources": len(self.records),
            "registered_by_kind": by_registration,
            "identical": not problems,
            "problems": problems,
            "rows": rows,
            "window": "identity recorded at first read: the experiment "
                      "config is read once to build the run and registered "
                      "immediately after; the four anchors are verified "
                      "against their pinned digests and registered as "
                      "anchors; every pinned model/scaler/cache/identity "
                      "file (including every candidate's fold_files), the "
                      "fit summary/admission, the protocol, the split CSV "
                      "and both reference formal_v2 files are registered "
                      "with their verified pins BEFORE their consuming "
                      "reads; end-of-run re-comparison detects any later "
                      "change",
            "scope_note": "the protected set is this run's registered "
                          "inputs only - experiment config, split CSV, "
                          "anchors, protocol, fit summary/admission, "
                          "historical evidence, reference formal_v2 "
                          "predictions and metrics, images, D0 masks, "
                          "encoder/classifier/scaler checkpoints and "
                          "cache bindings; no whole-project protection is "
                          "claimed",
        }
        return rows_meta

    def write_manifest(self, path: Path, protection: dict) -> None:
        dump_json(path, {
            "registered_sources": protection["registered_sources"],
            "registered_by_kind": protection["registered_by_kind"],
            "identical": protection["identical"],
            "problems": protection["problems"],
            "window": protection["window"],
            "scope_note": protection["scope_note"],
            "items": protection["rows"],
        })


@dataclass
class RunContext:
    config: dict
    output_dir: Path
    mode: str
    device: str
    budget_minutes: float
    max_frames: int | None
    run_id: str
    watch: BudgetWatch
    registry: SourceRegistry
    config_path: Path | None = None
    stages: dict = field(default_factory=dict)

    def stage(self, name: str, status: str, **details: Any) -> None:
        self.stages[name] = {"status": status, **details}
        log(f"stage {name}: {status}")


# ---------------------------------------------------------------------------
# Admission: anchors, frozen bindings, authoritative support


def verify_anchor(path: Path, expected_sha: str, registry: SourceRegistry,
                  role: str, expected_bytes: int | None = None) -> dict:
    if is_link_or_reparse(path):
        raise IdentityError(f"anchor {path} is a link/reparse point")
    digest = sha256_of_file(path)
    if digest != expected_sha:
        raise IdentityError(
            f"anchor {path}: bytes sha256 {digest} != pinned {expected_sha}")
    # the verified pin becomes the anchor's binding so the registry
    # classifies it as an anchor, not as this batch's snapshot
    rel = (path if path.is_absolute() else PROJECT_ROOT / path)
    registry.bindings[rel.relative_to(PROJECT_ROOT).as_posix()] = {
        "source": f"config.anchors[{role.split(':', 1)[1]}]",
        "expected_sha256": str(expected_sha),
        **({"expected_bytes": int(expected_bytes)}
           if expected_bytes is not None else {}),
    }
    record = registry.register(path, role, kind="anchor")
    return {"path": path, "sha256": digest, "record": record}


def build_bindings(cfg: dict, artifact_manifest: dict,
                   batch1_manifest: dict) -> dict[str, dict]:
    """Frozen expectations for every input this batch consumes, resolved
    from verified anchors only."""
    bindings: dict[str, dict] = {}
    manifest_dir = Path(cfg["anchors"]["artifact_manifest"]["path"]).parent
    for key, rel in (("runtime_frames", "evaluate/frames_complete.jsonl"),
                     ("runtime_predictions",
                      "evaluate/predictions_complete.jsonl"),
                     ("runtime_layer_b_summary",
                      "evaluate/layer_b_summary.json")):
        entry = (artifact_manifest.get("files") or {}).get(rel)
        if not isinstance(entry, dict) or "sha256" not in entry:
            raise IdentityError(
                f"artifact_manifest has no usable entry for {rel!r}")
        bound = (manifest_dir / rel).resolve()
        configured = (PROJECT_ROOT / cfg["sources"][key]).resolve()
        if bound != configured:
            raise IdentityError(
                f"sources.{key} does not resolve to the artifact manifest "
                f"entry {manifest_rel_err(rel, bound, configured)}")
        bindings[cfg["sources"][key]] = {
            "source": f"artifact_manifest.files[{rel!r}]",
            "expected_sha256": str(entry["sha256"]),
            "expected_bytes": int(entry["bytes"]),
        }
    # batch-1 manifest-covered inputs: re-verify current bytes against the
    # end-of-run identities batch 1 recorded and verified (the manifest is
    # itself an anchor of this batch)
    for entry in batch1_manifest.get("consumed_inputs", []):
        expected = entry.get("after_sha256") or entry.get("first_read_sha256")
        if not expected:
            raise IdentityError(
                f"batch1 manifest entry {entry.get('path')!r} carries no "
                "verified sha256; cannot bind it")
        bindings[entry["path"]] = {
            "source": "batch1_source_manifest",
            "expected_sha256": str(expected),
        }
    return bindings


def manifest_rel_err(rel: str, bound: Path, configured: Path) -> str:
    return f"{rel} -> {bound} != configured {configured}"


def load_admission(cfg: dict, registry: SourceRegistry,
                   config_path: Path | None = None) -> dict:
    """Verify every anchor and frozen binding; register every consumed
    input BEFORE its consuming read; return the authoritative inputs
    (runtime config object, protocol, frames, matches, D0 records).
    No model is constructed and no SGBM runs here."""
    if config_path is not None:
        # the experiment config is read once in main() before the run
        # context exists; its identity is registered here, before every
        # later read inside this entry
        registry.register(config_path, "experiment_config",
                          kind="experiment_config")
    for name, anchor in cfg["anchors"].items():
        verify_anchor(PROJECT_ROOT / anchor["path"], anchor["sha256"],
                      registry, f"anchor:{name}",
                      anchor.get("bytes"))
    artifact_manifest = json.loads(
        (PROJECT_ROOT / cfg["anchors"]["artifact_manifest"]["path"])
        .read_text(encoding="utf-8"))
    batch1_manifest = json.loads(
        (PROJECT_ROOT / cfg["anchors"]["batch1_source_manifest"]["path"])
        .read_text(encoding="utf-8"))
    bindings = build_bindings(cfg, artifact_manifest, batch1_manifest)
    registry.bindings.update(bindings)

    runtime_cfg = load_residual_runtime_config(
        PROJECT_ROOT / cfg["anchors"]["runtime_config"]["path"])

    # register every model/scaler/cache/identity file the frozen loader
    # will read (or digest-verify) during classifier construction, with
    # the runtime config's own pins as the binding — never a current
    # re-hash posing as history. This enumerates EVERY candidate's
    # fold_files (not just the legacy residual_models registry), the
    # protocol file, and the fit summary/admission the loader consumes.
    def frozen(rel: str, sha: str) -> dict:
        return {"source": "frozen_runtime_config",
                "expected_sha256": str(sha)}

    def register_frozen(rel: str, role: str, sha: str) -> None:
        registry.bindings[rel] = frozen(rel, sha)
        registry.register(PROJECT_ROOT / rel, role)

    register_frozen(cfg["anchors"]["runtime_config"]["path"],
                    "runtime_config_file",
                    cfg["anchors"]["runtime_config"]["sha256"])
    register_frozen(runtime_cfg.encoder_path, "encoder",
                    runtime_cfg.encoder_sha256)
    for cache_ref in tuple(runtime_cfg.global_cache) + \
            tuple(runtime_cfg.local_cache):
        register_frozen(cache_ref.path,
                        f"cache:{Path(cache_ref.path).name}",
                        cache_ref.sha256)
    for fold, refs in runtime_cfg.p0_models.items():
        for name, ref in refs.items():
            register_frozen(ref.path, f"p0_model:{fold}:{name}",
                            ref.sha256)
    for seed_map in runtime_cfg.residual_models.values():
        for fold, variants in seed_map.items():
            for variant, files in variants.items():
                for name, ref in files.items():
                    register_frozen(
                        ref.path,
                        f"residual_model:{fold}:{variant}:{name}",
                        ref.sha256)
    for fold, ref in runtime_cfg.r0_records.items():
        register_frozen(ref.path, f"r0_records:{fold}", ref.sha256)
    for cand in runtime_cfg.candidates:
        for fold, files in cand.fold_files.items():
            for name, ref in files.items():
                register_frozen(ref.path,
                                f"candidate:{cand.candidate_id}:"
                                f"{fold}:{name}", ref.sha256)
    register_frozen(runtime_cfg.protocol_path, "protocol",
                    runtime_cfg.protocol_sha256)
    fit_summary_rel = (Path(runtime_cfg.fit_dir) / "summary.json").as_posix()
    fit_admission_rel = (Path(runtime_cfg.fit_dir)
                         / "admission.json").as_posix()
    register_frozen(fit_summary_rel, "fit_summary",
                    runtime_cfg.summary_sha256)
    register_frozen(fit_admission_rel, "fit_admission",
                    runtime_cfg.admission_sha256)

    # the retained formal_v2 run is consumed late (regression reference);
    # both files are registered here with the experiment config's
    # retention identities as their bindings
    reference = cfg.get("reference_formal_v2") or {}
    if reference:
        for key, role in (("predictions_arms",
                           "reference_predictions_arms"),
                          ("metrics_summary",
                           "reference_metrics_summary")):
            rel = str(reference[key])
            registry.bindings[rel] = {
                "source": "reference_formal_v2_config",
                "expected_sha256": str(reference[f"{key}_sha256"])}
            registry.register(PROJECT_ROOT / rel, role)

    protocol = load_protocol(runtime_cfg)

    # the split CSV is consumed here for the first time: register (with
    # the batch-1 manifest binding when it covers it) BEFORE the read
    registry.register(PROJECT_ROOT / cfg["sources"]["split_csv"],
                      "split_csv")
    with (PROJECT_ROOT / cfg["sources"]["split_csv"]).open(
            encoding="utf-8") as handle:
        val_rows = [row for row in csv.DictReader(handle)
                    if row["split"] == "val"]
    expected_frames = int(cfg["support_expectations"]["frames"])
    if len(val_rows) != expected_frames:
        raise IdentityError(
            f"val rows {len(val_rows)} != pinned {expected_frames}")
    frame_folds = build_frame_fold_map(val_rows, protocol)

    registry.register(PROJECT_ROOT / cfg["sources"]["runtime_frames"],
                      "runtime_frames_complete")
    registry.register(PROJECT_ROOT / cfg["sources"]["runtime_predictions"],
                      "runtime_predictions_complete")
    registry.register(
        PROJECT_ROOT / cfg["sources"]["runtime_layer_b_summary"],
        "runtime_layer_b_summary")
    frames = read_jsonl(PROJECT_ROOT / cfg["sources"]["runtime_frames"])
    predictions = read_jsonl(
        PROJECT_ROOT / cfg["sources"]["runtime_predictions"])
    join = pipeline_gt_prediction_join(frames, [r for r in predictions
                                                if r.get("candidate_id") == "R0"])
    totals = join["totals"]
    expectations = cfg["support_expectations"]
    support = {
        "frames": totals["frames"],
        "gt_instances": totals["gt_instances"],
        "predictions": totals["predictions"],
        "matches": totals["matches"],
        "missed_gt": totals["gt_instances"] - totals["matches"],
        "unmatched_predictions": totals["predictions"] - totals["matches"],
    }
    for key, mine in support.items():
        theirs = int(expectations[key])
        if mine != theirs:
            raise IdentityError(
                f"support {key}: {mine} != pinned {theirs}")

    d0_dir = (PROJECT_ROOT / cfg["sources"]["detection_dir"] /
              "conditions" / "D0")
    registry.register(PROJECT_ROOT / cfg["sources"]["detection_dir"] /
                      "summary.json", "detection_summary")
    registry.register(d0_dir / "predictions.jsonl", "d0_predictions")
    d0_predictions = read_jsonl(d0_dir / "predictions.jsonl")
    by_frame: dict[str, dict[int, dict]] = {}
    for row in d0_predictions:
        frame_map = by_frame.setdefault(row["frame_id"], {})
        index = int(row["prediction_index"])
        if index in frame_map:
            raise IdentityError(
                f"duplicate D0 prediction index {index} in {row['frame_id']}")
        frame_map[index] = row
    unmatched_expect = {
        "d0_rows": len(d0_predictions),
    }
    if unmatched_expect["d0_rows"] != totals["predictions"]:
        raise IdentityError(
            f"D0 prediction rows {unmatched_expect['d0_rows']} != "
            f"historical predictions {totals['predictions']}")

    # register every frame's per-frame inputs (images + D0 masks)
    image_roles = {}
    for row in val_rows:
        stem = row["output_stem"]
        image_roles[stem] = row
        registry.register(PROJECT_ROOT / row["left_path"],
                          f"image_left:{stem}")
        registry.register(PROJECT_ROOT / row["right_path"],
                          f"image_right:{stem}")
        registry.register(d0_dir / "masks" / f"{stem}.npz",
                          f"d0_masks:{stem}")
    new_snapshot = sorted(
        r["path"] for r in registry.records.values()
        if r["registration"] == "batch2_snapshot")
    return {
        "runtime_cfg": runtime_cfg,
        "protocol": protocol,
        "val_rows": val_rows,
        "frame_folds": frame_folds,
        "frames": frames,
        "frame_by_id": {f["frame_id"]: f for f in frames},
        "predictions": predictions,
        "d0_by_frame": by_frame,
        "new_snapshot_paths": new_snapshot,
        "batch1_covered": len(batch1_manifest.get("consumed_inputs", [])),
    }


def load_frame_instances(admission: dict, cfg: dict, stem: str,
                         registry: SourceRegistry) -> tuple[list, list]:
    """Saved-D0 instances for one frame, identity-checked against the D0
    prediction records and the frozen frames_complete entry."""
    from models.segmentation import Instance

    frame = admission["frame_by_id"].get(stem)
    if frame is None:
        raise IdentityError(f"{stem}: absent from frames_complete")
    if frame.get("frame_error") is not None:
        raise IdentityError(f"{stem}: historical frame_error is set")
    payload_file = (PROJECT_ROOT / cfg["sources"]["detection_dir"] /
                    "conditions" / "D0" / "masks" / f"{stem}.npz")
    d0_rows = admission["d0_by_frame"].get(stem, {})
    frame_pred_ids = {int(p["pred_instance_id"])
                      for p in frame["predictions"]}
    if not d0_rows:
        if frame["predictions"]:
            raise IdentityError(
                f"{stem}: expected {len(frame['predictions'])} predictions "
                "but D0 has no prediction records; a missing mask file is "
                "never treated as an empty frame")
        return [], []
    registry.register(payload_file, f"d0_masks:{stem}")
    with np.load(payload_file) as payload:
        shape = tuple(int(v) for v in payload["shape"])
        packed = payload["masks"]
        confidences = payload["confidences"]
        bboxes = payload["bboxes"]
    count = int(confidences.shape[0])
    if count != len(d0_rows):
        raise IdentityError(
            f"{stem}: mask count {count} != D0 rows {len(d0_rows)}")
    if set(d0_rows) != set(range(count)):
        raise IdentityError(f"{stem}: prediction ids are not 0..{count - 1}")
    if frame_pred_ids != set(range(count)):
        raise IdentityError(
            f"{stem}: frames_complete prediction ids {sorted(frame_pred_ids)}"
            f" != 0..{count - 1}")
    bits = np.unpackbits(packed, axis=1)[:, :shape[0] * shape[1]]
    instances = []
    for index in range(count):
        row = d0_rows[index]
        bbox = tuple(int(v) for v in bboxes[index])
        if tuple(int(v) for v in row["bbox"]) != bbox:
            raise IdentityError(
                f"{stem}/pred{index}: npz bbox {bbox} != D0 record "
                f"{tuple(int(v) for v in row['bbox'])}")
        mask = bits[index].reshape(shape).astype(np.uint8)
        instances.append(Instance(
            id=index, bbox=bbox, mask=mask, confidence=float(confidences[index]),
            class_id=0, class_name="d0_prediction"))
    return instances, frame


# ---------------------------------------------------------------------------
# Evaluation core


class ReproductionError(RuntimeError):
    """The legacy arm failed to reproduce the saved 180-frame results."""


class ModelAProbe:
    """Records calls and then raises — the evaluation must never run
    detection, and a swallowed exception must not fake zero calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, image: np.ndarray, **kwargs: Any) -> list:
        self.calls.append({"shape": list(np.asarray(image).shape),
                           "kwargs": sorted(kwargs)})
        raise AssertionError(
            "Model A probe invoked: the upsample-impact evaluation must "
            "reuse the saved D0 predictions instead of running detection")


def build_arm_engine(cfg: dict, runtime_cfg, version: str, device: str,
                     probe: ModelAProbe) -> tuple[StereoMatcher,
                                                  DualStageInferenceEngine]:
    stereo = runtime_cfg.stereo
    calibration = runtime_cfg.calibration
    stereo_cfg = StereoMatcherConfig(
        matcher=str(stereo["matcher"]),
        mode=str(stereo["mode"]),
        max_disparity=int(stereo["max_disp"]),
        scale=float(stereo["scale"]),
        block_size=int(stereo["block_size"]),
        uniqueness_ratio=int(stereo["uniqueness_ratio"]),
        speckle_window=int(stereo["speckle_window"]),
        speckle_range=int(stereo["speckle_range"]),
        texture_threshold=float(stereo["texture_threshold"]),
        lr_check_threshold_px=float(stereo["lr_check_threshold_px"]),
        min_valid_ratio=float(stereo["min_valid_ratio"]),
        upsample_version=version)
    expected = cfg["arms"][("legacy" if version ==
                            UPSAMPLE_LEGACY_ZERO_LINEAR_V1 else "fix")]
    if stereo_cfg.upsample_version != expected["upsample_version"]:
        raise IdentityError(
            f"constructed matcher upsample_version "
            f"{stereo_cfg.upsample_version!r} != configured "
            f"{expected['upsample_version']!r}; refusing to run a path the "
            "config did not select")
    matcher = StereoMatcher(stereo_cfg)
    engine = DualStageInferenceEngine(
        model_a=probe, model_b=None,
        matcher=matcher, rectifier=None, rectify_enabled=False,
        device="cpu",
        crop_padding=int(runtime_cfg.crop_padding),
        band_vertical_margin=int(stereo.get("band_margin", 20)),
        baseline=float(calibration["baseline"]),
        focal_length=float(calibration["focal_length"]),
        model_b_input_mode="material_fusion")
    return matcher, engine


class _ReplayMatcher(StereoMatcher):
    """Returns one precomputed :class:`StereoMatchResult` from the arm's
    own production matcher. The engine's orchestration (instance stats,
    polar, inputs) then runs on exactly the maps the arm measured, with no
    second SGBM pass and no second stereo implementation."""

    def __init__(self, config: StereoMatcherConfig,
                 preset) -> None:
        super().__init__(config)
        self._preset = preset

    def compute_bands(self, left, right, bands, full_image_threshold=0.9):
        return self._preset

    def compute(self, left, right):
        return self._preset


def run_frame_arms(matcher_legacy: StereoMatcher, matcher_fix: StereoMatcher,
                   engine_legacy: DualStageInferenceEngine,
                   engine_fix: DualStageInferenceEngine,
                   left: np.ndarray, right: np.ndarray, instances: list,
                   stem: str, classifiers: dict,
                   extractor: "_SharedGrayExtractor", s_classifier,
                   band_margin: int) -> dict:
    """Both arms of one frame over ONE low-resolution matching pass.

    The legacy matcher runs ``match_cores`` exactly once (both SGBM passes
    per band plus consistency/texture checks — the complete low-resolution
    core); the SAME cores are assembled by each matcher through its own
    approved upsample version and replayed into the production post-
    processing. The fix matcher runs NO matching at all, which the call
    counters prove."""
    height = left.shape[0]
    bands = build_horizontal_bands(
        (inst.bbox for inst in instances), image_height=height,
        vertical_margin=band_margin)
    cores = matcher_legacy.match_cores(left, right, bands)
    result_legacy = matcher_legacy.assemble_from_cores(cores, left.shape)
    result_fix = matcher_fix.assemble_from_cores(cores, left.shape)

    def core_digest(cap) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(cap.disp_small).tobytes())
        digest.update(np.ascontiguousarray(cap.valid_small).tobytes())
        return digest.hexdigest()

    core_evidence = {
        "core_count": len(cores),
        "legacy_sgbm_calls": matcher_legacy.lowres_match_calls,
        "fix_sgbm_calls": matcher_fix.lowres_match_calls,
        "core_sha256": [core_digest(c.capture) for c in cores],
        "core_shapes": [list(c.capture.disp_small.shape) for c in cores],
        "note": "legacy_sgbm_calls counts low-resolution match invocations; "
                "each invocation covers BOTH underlying SGBM passes. The "
                "fix matcher must show zero calls: it assembles the shared "
                "cores through its own upsample version.",
    }
    if matcher_fix.lowres_match_calls != 0 or \
            matcher_legacy.lowres_match_calls != len(cores):
        raise IdentityError(
            f"{stem}: low-resolution match call evidence broken "
            f"(legacy {matcher_legacy.lowres_match_calls}, fix "
            f"{matcher_fix.lowres_match_calls}, cores {len(cores)}); the "
            "single-core contract requires exactly one matching pass on "
            "the legacy matcher and none on the fix matcher")

    arms: dict[str, dict] = {}
    for arm, stereo, engine in (("legacy", result_legacy, engine_legacy),
                                ("fix", result_fix, engine_fix)):
        engine.matcher = _ReplayMatcher(matcher_legacy.config, stereo)
        result = engine.process_instances_detailed(
            left, right, instances, already_rectified=True, frame_id=stem)
        inputs = list(result.material_inputs)
        gray = extractor.extract(inputs)
        outcomes = {cid: clf.predict_batch(inputs, gray=gray)
                    for cid, clf in classifiers.items()}
        cs = [s_classifier.local_features(inp) for inp in inputs]
        arms[arm] = {
            "result": result, "inputs": inputs, "gray": gray,
            "outcomes": outcomes, "cs": cs,
            "lowres_cores": list(matcher_legacy.last_lowres_cores),
            "stereo_disparity": stereo.disparity,
            "stereo_valid": stereo.valid,
        }
    return {"arms": arms, "core_evidence": core_evidence}


def frame_arm_comparison(frame: dict, instances: list,
                         legacy: dict, fix: dict) -> dict:
    """Instance-masked paired disparity/q comparison between the two arms
    (fix minus legacy), matched and unmatched predictions separately.

    When either arm's polar_result is None that arm's q support is treated
    as all-invalid: the other arm's in-instance valid pixels become lost or
    gained and the q common region is empty. The disparity statistics are
    computed INDEPENDENTLY of the q maps (they never get skipped because q
    is missing). Empty common regions still contribute lost/gained; error
    statistics need a non-empty common region (n=0 -> nulls). These are
    PATH DIFFERENCES on real images without ground truth, never accuracy."""
    matched_pred_ids = {int(pi) for _gi, pi, _ in frame["matches"]}
    acc = {
        "q_common_pixels": 0, "q_absdq_sum": 0.0, "q_lost": 0, "q_gained": 0,
        "d_common_pixels": 0, "d_absdiff_sum": 0.0,
        "d_lost": 0, "d_gained": 0,
        "abs_dq_values": [], "matched_q_common_pixels": 0,
        "matched_q_absdq_sum": 0.0, "matched_q_lost": 0, "matched_q_gained": 0,
        "unmatched_q_common_pixels": 0, "unmatched_q_absdq_sum": 0.0,
        "unmatched_q_lost": 0, "unmatched_q_gained": 0,
    }
    polar_legacy = legacy["result"].polar_result
    polar_fix = fix["result"].polar_result
    b_qv_all = (polar_legacy.valid_mask > 0) if polar_legacy is not None         else np.zeros(legacy["stereo_valid"].shape, bool)
    a_qv_all = (polar_fix.valid_mask > 0) if polar_fix is not None \
        else np.zeros(fix["stereo_valid"].shape, bool)
    q_legacy = (polar_legacy.signed_q.astype(np.float64)
                if polar_legacy is not None else None)
    q_fix = (polar_fix.signed_q.astype(np.float64)
             if polar_fix is not None else None)
    for inst in instances:
        arm_key = ("matched" if int(inst.id) in matched_pred_ids
                   else "unmatched")
        mask = inst.mask > 0
        b_qv = b_qv_all
        a_qv = a_qv_all
        common = mask & b_qv & a_qv
        if q_legacy is not None and q_fix is not None:
            dq = (q_fix - q_legacy)[common]
        else:
            dq = np.zeros(0, np.float64)
        lost = int((mask & b_qv & ~a_qv).sum())
        gained = int((mask & a_qv & ~b_qv).sum())
        acc["q_common_pixels"] += int(common.sum())
        acc["q_absdq_sum"] += float(np.abs(dq).sum())
        acc["q_lost"] += lost
        acc["q_gained"] += gained
        acc[f"{arm_key}_q_common_pixels"] += int(common.sum())
        acc[f"{arm_key}_q_absdq_sum"] += float(np.abs(dq).sum())
        acc[f"{arm_key}_q_lost"] += lost
        acc[f"{arm_key}_q_gained"] += gained
        if dq.size:
            acc["abs_dq_values"].append(dq)
        d_common = mask & (legacy["stereo_valid"] > 0) & \
            (fix["stereo_valid"] > 0)
        dd = (fix["stereo_disparity"].astype(np.float64)
              - legacy["stereo_disparity"].astype(np.float64))[d_common]
        acc["d_common_pixels"] += int(d_common.sum())
        acc["d_absdiff_sum"] += float(np.abs(dd).sum())
        acc["d_lost"] += int((mask & (legacy["stereo_valid"] > 0)
                              & ~(fix["stereo_valid"] > 0)).sum())
        acc["d_gained"] += int((mask & (fix["stereo_valid"] > 0)
                                & ~(legacy["stereo_valid"] > 0)).sum())
    stats = {
        "q_common_pixels": acc["q_common_pixels"],
        "q_absdq_mean_on_common": (
            acc["q_absdq_sum"] / acc["q_common_pixels"]
            if acc["q_common_pixels"] else None),
        "q_lost": acc["q_lost"], "q_gained": acc["q_gained"],
        "disparity_common_pixels": acc["d_common_pixels"],
        "disparity_absdiff_mean_on_common": (
            acc["d_absdiff_sum"] / acc["d_common_pixels"]
            if acc["d_common_pixels"] else None),
        "disparity_lost": acc["d_lost"], "disparity_gained": acc["d_gained"],
        "abs_dq_pooled_p95": "NOT_COMPUTED",
        "matched": {
            "q_common_pixels": acc["matched_q_common_pixels"],
            "q_absdq_mean_on_common": (
                acc["matched_q_absdq_sum"] / acc["matched_q_common_pixels"]
                if acc["matched_q_common_pixels"] else None),
            "q_lost": acc["matched_q_lost"],
            "q_gained": acc["matched_q_gained"],
        },
        "unmatched": {
            "q_common_pixels": acc["unmatched_q_common_pixels"],
            "q_absdq_mean_on_common": (
                acc["unmatched_q_absdq_sum"] / acc["unmatched_q_common_pixels"]
                if acc["unmatched_q_common_pixels"] else None),
            "q_lost": acc["unmatched_q_lost"],
            "q_gained": acc["unmatched_q_gained"],
        },
    }
    if acc["abs_dq_values"]:
        pooled = np.concatenate(acc["abs_dq_values"])
        stats["abs_dq_instancewise_note"] = (
            "per-instance common-region differences pooled here for the "
            "MAE only; percentiles are NOT_COMPUTED at pooled level")
        stats["abs_dq_mean"] = float(np.mean(pooled))
    else:
        stats["abs_dq_mean"] = None
    return stats



# ---------------------------------------------------------------------------
# Unmatched-prediction class changes: per-identity evidence (matched
# predictions are covered by paired_fix_minus_legacy's detail)


def unmatched_class_changes(rows: list[dict],
                            candidate_ids: list[str]) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for row in rows:
        by_key[(row["arm"], row["candidate_id"], row["frame_id"],
                int(row["pred_instance_id"]))] = row
    changes = []
    for cid in candidate_ids:
        for key, legacy_row in by_key.items():
            arm, _cid, frame_id, pred_id = key
            if arm != "legacy" or _cid != cid:
                continue
            if legacy_row.get("matched_gt_id") is not None:
                continue
            fix_row = by_key.get(("fix", cid, frame_id, pred_id))
            if fix_row is None:
                raise IdentityError(
                    f"missing fix-arm row for unmatched {cid}/{frame_id}/"
                    f"{pred_id}")
            if legacy_row.get("predicted_class") != \
                    fix_row.get("predicted_class") or \
                    legacy_row.get("status") != fix_row.get("status"):
                changes.append({
                    "candidate_id": cid, "frame_id": frame_id,
                    "pred_instance_id": pred_id,
                    "matched": False,
                    "legacy_class": legacy_row.get("predicted_class"),
                    "fix_class": fix_row.get("predicted_class"),
                    "legacy_status": legacy_row.get("status"),
                    "fix_status": fix_row.get("status"),
                    "legacy_logits": legacy_row.get("final_logits"),
                    "fix_logits": fix_row.get("final_logits"),
                })
    return changes


def verify_reference_run(rows: list[dict], cfg: dict,
                         project_root: Path) -> dict:
    """The retained formal_v2 run is the repair's regression reference: the
    new run must reproduce its per-prediction classes, fallbacks and
    logits (frozen tolerance) for BOTH arms, or the repair changed a
    classification result and the run stops. BOTH retained files
    (predictions and metrics) are checked against the pinned retention
    identities; a logits length mismatch and a valid row whose logits are
    not four finite values are explicit failures, never skipped."""
    reference = cfg.get("reference_formal_v2") or {}
    ref_path = reference.get("predictions_arms")
    metrics_path = reference.get("metrics_summary")
    if not ref_path:
        return {"pairs": 0, "problem_count": 0, "problems": [],
                "note": "no reference run configured"}
    retained_identities: dict[str, str] = {}
    for label, rel, sha_key in (
            ("predictions_arms", ref_path, "predictions_arms_sha256"),
            ("metrics_summary", metrics_path, "metrics_summary_sha256")):
        if not rel:
            raise IdentityError(
                f"reference formal_v2 {label} path is not configured")
        pinned = reference.get(sha_key)
        if not pinned:
            raise IdentityError(
                f"reference formal_v2 {label} carries no pinned sha256")
        actual = sha256_of_file(project_root / rel)
        if actual != str(pinned):
            raise IdentityError(
                f"reference formal_v2 {label} changed since the retention "
                f"check: {actual} != {pinned}; retained evidence may have "
                "been tampered with")
        retained_identities[label] = actual

    def invalid_logits(status: Any, values: Any) -> str | None:
        if status != "valid":
            return None
        if not isinstance(values, list) or len(values) != 4:
            return "valid row logits are not 4 values"
        if not all(isinstance(v, (int, float)) and np.isfinite(v)
                   for v in values):
            return "valid row logits contain non-finite values"
        return None

    historical = {}
    for line in (project_root / ref_path).read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            historical[(row["arm"], row["candidate_id"], row["frame_id"],
                        int(row["pred_instance_id"]))] = row
    atol = float(cfg["tolerances"]["logit_atol"])
    problems = []
    checked = 0
    for row in rows:
        key = (row["arm"], row["candidate_id"], row["frame_id"],
               int(row["pred_instance_id"]))
        ref = historical.get(key)
        if ref is None:
            problems.append(f"{key}: absent from the reference run")
            continue
        checked += 1
        if row["status"] != ref.get("status") or \
                row["predicted_class"] != ref.get("predicted_class"):
            problems.append(
                f"{key}: class/status {row['predicted_class']!r}/"
                f"{row['status']!r} != reference "
                f"{ref.get('predicted_class')!r}/{ref.get('status')!r}")
            continue
        if bool(row["used_fallback"]) != bool(ref.get("used_fallback")):
            problems.append(f"{key}: used_fallback changed")
        mine = row.get("final_logits") or []
        theirs = ref.get("final_logits") or []
        for label, status, values in (("new", row.get("status"), mine),
                                      ("reference", ref.get("status"),
                                       theirs)):
            flaw = invalid_logits(status, values)
            if flaw:
                problems.append(f"{key}: {label} run {flaw}")
        if len(mine) != len(theirs):
            problems.append(f"{key}: logits length {len(mine)} != "
                            f"{len(theirs)}")
        elif mine:
            diff = float(np.max(np.abs(
                np.asarray(mine, dtype=np.float64)
                - np.asarray(theirs, dtype=float))))
            if diff > atol:
                problems.append(f"{key}: logits diff {diff} > {atol}")
    return {"pairs": checked, "problem_count": len(problems),
            "problems": problems[:20], "atol": atol,
            "reference": ref_path,
            "retained_identities": retained_identities}



def independent_recompute(frames: list[dict], rows: list[dict],
                          candidate_ids: list[str], arms: tuple) -> dict:
    """Rebuild every reported metric from the raw GT, the FIXED matches and
    the flat per-prediction rows — plain counting loops, no production
    summarizer. Semantics mirror the frozen evaluation contract: a matched
    instance whose prediction failed is an unclassified e2e failure; a
    failed unmatched prediction contributes no FP (class -1 is dropped);
    classification errors are FN for the true class and FP for the
    predicted class."""
    num_c = len(CLASS_NAMES)
    result: dict[str, Any] = {}
    matched_keys = {}
    matched_gt_by_frame: dict[str, set] = {}
    gt_total = sum(len(frame["gt"]) for frame in frames)
    predictions_total = sum(len(frame["predictions"]) for frame in frames)
    for frame in frames:
        gt_classes = {int(g["gt_id"]): int(g["class_id"]) for g in frame["gt"]}
        matched_gt_by_frame[frame["frame_id"]] = {
            int(gt_id) for gt_id, _p, _iou in frame["matches"]}
        for gt_id, pred_id, _iou in frame["matches"]:
            matched_keys[(frame["frame_id"], int(pred_id))] = (
                int(gt_id), gt_classes[int(gt_id)])
    for arm in arms:
        result[arm] = {}
        for cid in candidate_ids:
            cond_tp = [0] * num_c
            cond_fn = [0] * num_c
            cond_fp = [0] * num_c
            e2e_tp = [0] * num_c
            e2e_fp = [0] * num_c
            e2e_fn = [0] * num_c
            confusion = [[0] * num_c for _ in range(num_c)]
            correct = samples = failed = fallback = rows_seen = 0
            for row in rows:
                if row["arm"] != arm or row["candidate_id"] != cid:
                    continue
                rows_seen += 1
                key = (row["frame_id"], int(row["pred_instance_id"]))
                matched = key in matched_keys
                if row.get("used_fallback"):
                    fallback += 1
                valid = row.get("status") == "valid"
                pred_class = row.get("predicted_class")
                if not matched:
                    if valid and isinstance(pred_class, int) and \
                            0 <= pred_class < num_c:
                        e2e_fp[pred_class] += 1
                    continue
                gt_id, true_class = matched_keys[key]
                if not valid or pred_class is None:
                    # unclassified matched: an e2e failure, excluded from
                    # the conditional denominator (production semantics)
                    failed += 1
                    e2e_fn[true_class] += 1
                    continue
                samples += 1
                if pred_class == true_class:
                    correct += 1
                    cond_tp[true_class] += 1
                    e2e_tp[true_class] += 1
                else:
                    cond_fn[true_class] += 1
                    cond_fp[pred_class] += 1
                    e2e_fn[true_class] += 1
                    e2e_fp[pred_class] += 1
                confusion[true_class][pred_class] += 1
            # missed GT: every authoritative GT with no match in its frame
            # contributes an e2e FN (arm-independent, but each arm/candidate
            # rebuilds it from the raw GT and the FIXED matches)
            for frame in frames:
                gt_classes = {int(g["gt_id"]): int(g["class_id"])
                              for g in frame["gt"]}
                for gt_id, gt_class in gt_classes.items():
                    if gt_id not in matched_gt_by_frame[frame["frame_id"]]:
                        e2e_fn[gt_class] += 1

            def prf(tp: int, fp: int, fn: int) -> dict:
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = (2 * prec * rec / (prec + rec)
                      if prec + rec else 0.0)
                return {"precision": prec, "recall": rec, "f1": f1}

            macro_f1 = float(np.mean([
                prf(cond_tp[i], cond_fp[i], cond_fn[i])["f1"]
                for i in range(num_c)]))
            e2e_macro_f1 = float(np.mean([
                prf(e2e_tp[i], e2e_fp[i], e2e_fn[i])["f1"]
                for i in range(num_c)]))
            result[arm][cid] = {
                "samples": samples,
                "correct": correct,
                "accuracy": correct / samples if samples else 0.0,
                "macro_f1": macro_f1,
                "confusion": confusion,
                "conditional_tp": cond_tp, "conditional_fp": cond_fp,
                "conditional_fn": cond_fn,
                "conditional_per_class": [
                    prf(cond_tp[i], cond_fp[i], cond_fn[i])
                    for i in range(num_c)],
                "end_to_end_macro_f1": e2e_macro_f1,
                "end_to_end_per_class": [
                    prf(e2e_tp[i], e2e_fp[i], e2e_fn[i])
                    for i in range(num_c)],
                "e2e_tp": e2e_tp, "e2e_fp": e2e_fp, "e2e_fn": e2e_fn,
                "failed_matched": failed,
                "used_fallback": fallback,
                "predictions_total": rows_seen,
                "support": {
                    "evaluated_frames": len(frames),
                    "gt_instances": gt_total,
                    "predictions": predictions_total,
                    "matched": len(matched_keys),
                    "missed_gt": gt_total - len(matched_keys),
                    "unmatched_predictions": predictions_total
                    - len(matched_keys),
                    "unclassified_matched": failed,
                },
            }
    return result


def paired_fix_minus_legacy(rows: list[dict], candidate_ids: list[str]) -> dict:
    """Per candidate: fix-vs-legacy prediction changes over matched
    instances. A failed prediction counts as incorrect; quality/input
    changes without a prediction flip are reported as covariates, never as
    accuracy gains."""
    by_key: dict[tuple, dict] = {}
    for row in rows:
        by_key[(row["arm"], row["candidate_id"], row["frame_id"],
                int(row["pred_instance_id"]))] = row
    out: dict[str, dict] = {}
    for cid in candidate_ids:
        improved = broken = both_correct = both_wrong = unchanged = 0
        pairs_total = logits_changed = logits_changed_matched = 0
        logits_max_diff: float | None = None
        detail = []
        for key, legacy_row in by_key.items():
            arm, row_cid, frame_id, pred_id = key
            if arm != "legacy" or row_cid != cid:
                continue
            fix_row = by_key.get(("fix", cid, frame_id, pred_id))
            if fix_row is None:
                raise IdentityError(
                    f"missing fix-arm row for {cid}/{frame_id}/{pred_id}")
            # class-unchanged and logits-bitwise-unchanged are DIFFERENT
            # claims: count every pair whose final_logits lists differ at
            # all, over ALL of the candidate's pairs (not only matched)
            pairs_total += 1
            mine = list(legacy_row.get("final_logits") or [])
            theirs = list(fix_row.get("final_logits") or [])
            if mine != theirs:
                logits_changed += 1
                if legacy_row.get("matched_gt_id") is not None:
                    logits_changed_matched += 1
                if len(mine) == len(theirs) and mine:
                    diff = float(np.max(np.abs(
                        np.asarray(mine, dtype=np.float64)
                        - np.asarray(theirs, dtype=np.float64))))
                    logits_max_diff = (diff if logits_max_diff is None
                                       else max(logits_max_diff, diff))
            legacy_ok = (legacy_row.get("status") == "valid"
                         and legacy_row.get("matched_gt_id") is not None
                         and legacy_row.get("predicted_class")
                         == legacy_row.get("true_class"))
            fix_ok = (fix_row.get("status") == "valid"
                      and fix_row.get("matched_gt_id") is not None
                      and fix_row.get("predicted_class")
                      == fix_row.get("true_class"))
            if not (legacy_row.get("matched_gt_id") is not None
                    and fix_row.get("matched_gt_id") is not None):
                continue
            if fix_ok and not legacy_ok:
                improved += 1
            elif legacy_ok and not fix_ok:
                broken += 1
            elif fix_ok and legacy_ok:
                both_correct += 1
            else:
                both_wrong += 1
            if (fix_row.get("predicted_class")
                    != legacy_row.get("predicted_class")) or \
                    (fix_row.get("status") != legacy_row.get("status")):
                detail.append({
                    "frame_id": frame_id, "pred_instance_id": pred_id,
                    "matched_gt_id": legacy_row.get("matched_gt_id"),
                    "true_class": legacy_row.get("true_class"),
                    "legacy_class": legacy_row.get("predicted_class"),
                    "fix_class": fix_row.get("predicted_class"),
                    "legacy_status": legacy_row.get("status"),
                    "fix_status": fix_row.get("status"),
                    "legacy_logits": legacy_row.get("final_logits"),
                    "fix_logits": fix_row.get("final_logits"),
                    "legacy_fallback": legacy_row.get("used_fallback"),
                    "fix_fallback": fix_row.get("used_fallback"),
                })
        matched_total = improved + broken + both_correct + both_wrong
        out[cid] = {
            "matched_instances": matched_total,
            "improved": improved, "broken": broken,
            "both_correct": both_correct, "both_wrong": both_wrong,
            "changed_predictions": len(detail),
            "pairs_total": pairs_total,
            "logits_bitwise_changed_pairs": logits_changed,
            "logits_bitwise_changed_matched_pairs": logits_changed_matched,
            "logits_max_abs_diff_when_changed": logits_max_diff,
            "class_unchanged_note": "class-unchanged and "
                                    "logits-bitwise-unchanged are separate "
                                    "claims; the changed-pair counts above "
                                    "cover ALL of this candidate's "
                                    "fix-vs-legacy pairs",
            "detail": detail,
        }
    return out


# ---------------------------------------------------------------------------
# Modes


def run_preflight(ctx: RunContext) -> None:
    """Identity/config admission only: anchors, frozen bindings, the new
    snapshot for files the first batch's manifest does not cover, and the
    authoritative support counts. No model, no SGBM, no image decoding."""
    cfg = ctx.config
    ctx.watch.check("preflight start")
    admission = load_admission(cfg, ctx.registry, ctx.config_path)
    candidate_ids = sorted(s.candidate_id
                           for s in admission["runtime_cfg"].candidates)
    expected = cfg["candidates"]["expected_ids"]
    if candidate_ids != sorted(expected):
        raise IdentityError(
            f"candidate registry {candidate_ids} != pinned {sorted(expected)}")
    ctx.stage("preflight", "COMPLETED",
              val_frames=len(admission["val_rows"]),
              candidates=candidate_ids,
              batch1_covered_inputs=admission["batch1_covered"],
              batch2_new_snapshot_inputs=len(admission["new_snapshot_paths"]),
              support=cfg["support_expectations"])
    dump_json(ctx.output_dir / "preflight_plan.json", {
        "run_id": ctx.run_id,
        "anchors_verified": sorted(cfg["anchors"]),
        "candidates": candidate_ids,
        "arms": {"legacy": cfg["arms"]["legacy"]["upsample_version"],
                 "fix": cfg["arms"]["fix"]["upsample_version"]},
        "support_expectations": cfg["support_expectations"],
        "frames": [row["output_stem"] for row in admission["val_rows"]],
        "batch2_new_snapshot_inputs": admission["new_snapshot_paths"],
        "batch1_covered_inputs": admission["batch1_covered"],
        "registered_total": len(ctx.registry.records),
        "registered_by_kind": sorted(
            {kind: sum(1 for r in ctx.registry.records.values()
                       if r["registration"] == kind)
             for kind in {r["registration"]
                          for r in ctx.registry.records.values()}}.items()),
        "reference_formal_v2_configured":
            bool(cfg.get("reference_formal_v2")),
        "note": "preflight ran no model and no SGBM",
    })


def run_evaluate(ctx: RunContext) -> None:
    import cv2

    cfg = ctx.config
    ctx.watch.check("admission start")
    admission = load_admission(cfg, ctx.registry, ctx.config_path)
    runtime_cfg = admission["runtime_cfg"]
    ctx.stage("admission", "COMPLETED",
              val_frames=len(admission["val_rows"]),
              batch2_new_snapshot_inputs=len(admission["new_snapshot_paths"]))
    ctx.watch.check("before classifier load")
    classifiers = build_classifiers(
        runtime_cfg, admission["protocol"], device=ctx.device,
        frame_folds=admission["frame_folds"])
    candidate_ids = sorted(classifiers)
    expected = sorted(s.candidate_id for s in runtime_cfg.candidates)
    if candidate_ids != expected:
        raise IdentityError(
            f"loaded candidates {candidate_ids} != registry {expected}")
    r0_id = _pick(classifiers, "p0", "R0", None)
    s_classifier = classifiers[r0_id]
    extractor = _SharedGrayExtractor(classifiers)
    ctx.stage("classifiers", "COMPLETED", candidates=candidate_ids,
              device=ctx.device)

    probe = ModelAProbe()
    matcher_legacy, engine_legacy = build_arm_engine(
        cfg, runtime_cfg, UPSAMPLE_LEGACY_ZERO_LINEAR_V1, ctx.device, probe)
    matcher_fix, engine_fix = build_arm_engine(
        cfg, runtime_cfg, UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1,
        ctx.device, probe)
    ctx.stage("arms", "COMPLETED",
              legacy=UPSAMPLE_LEGACY_ZERO_LINEAR_V1,
              fix=UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)

    val_rows = admission["val_rows"]
    if ctx.max_frames is not None:
        val_rows = val_rows[:ctx.max_frames]
    band_margin = int(runtime_cfg.stereo.get("band_margin", 20))
    historical = {(row["candidate_id"], row["frame_id"],
                   int(row["pred_instance_id"])): row
                  for row in admission["predictions"]}
    frame_by_id = admission["frame_by_id"]
    fold_of = admission["frame_folds"]

    all_rows: list[dict] = []
    measurement_rows: list[dict] = []
    frames_out: list[dict] = []
    prod_records: dict[str, dict[str, list]] = {
        arm: {cid: [] for cid in candidate_ids} for arm in ARMS}
    # fallback tallied DIRECTLY from the classifier outcomes on the
    # production side; compared against the prediction-row tally of the
    # independent recomputation over the same all-predictions denominator
    production_fallback: dict[str, dict[str, dict[str, int]]] = {
        arm: {cid: {"used_fallback": 0, "predictions": 0}
              for cid in candidate_ids} for arm in ARMS}
    frame_loop_budget: str | None = None
    legacy_problem_frames: list[str] = []
    engine_check_failures: list[str] = []
    rows_path = ctx.output_dir / "predictions_arms.jsonl"
    handle = rows_path.open("w", encoding="utf-8")

    def emit(row: dict) -> None:
        handle.write(json.dumps(row, ensure_ascii=False,
                                allow_nan=False) + "\n")

    try:
        for index, row in enumerate(val_rows):
            ctx.watch.check(f"frame {row['output_stem']}")
            stem = row["output_stem"]
            frame = frame_by_id[stem]
            instances, _ = load_frame_instances(admission, cfg, stem,
                                                ctx.registry)
            left = to_gray_u8(cv2.imread(
                str(PROJECT_ROOT / row["left_path"]), cv2.IMREAD_UNCHANGED))
            right = to_gray_u8(cv2.imread(
                str(PROJECT_ROOT / row["right_path"]), cv2.IMREAD_UNCHANGED))
            if left.shape != right.shape:
                raise IdentityError(f"{stem}: left/right shape mismatch")
            gt_classes = [int(g["class_id"]) for g in frame["gt"]]
            matches = [(int(g), int(p), float(i)) for g, p, i
                       in frame["matches"]]
            matched_by_pred = {pi: gi for gi, pi, _ in matches}
            fold = fold_of[stem]
            group = row["group_name"]

            ctx.watch.check(f"frame arms {stem}")
            arm_runs = run_frame_arms(
                matcher_legacy, matcher_fix, engine_legacy, engine_fix,
                left, right, instances, stem, classifiers, extractor,
                s_classifier, band_margin)
            if probe.calls:
                raise IdentityError(
                    f"Model A probe recorded {len(probe.calls)} calls; "
                    "the evaluation must never run detection")
            # identical gray crops -> identical G
            for pos, g_legacy in enumerate(
                    arm_runs["arms"]["legacy"]["gray"]):
                g_fix = arm_runs["arms"]["fix"]["gray"][pos]
                if (g_legacy is None) != (g_fix is None):
                    raise IdentityError(
                        f"{stem}/pred{pos}: G presence differs across arms")
                if g_legacy is not None and not np.array_equal(
                        g_legacy, g_fix):
                    raise IdentityError(
                        f"{stem}/pred{pos}: G changed across arms although "
                        "the gray crop and mask are fixed")

            frame_row: dict[str, Any] = {
                "frame_id": stem, "group": group, "fold": fold,
                "gt_instances": len(gt_classes),
                "predictions": len(instances),
                "matches": len(matches),
                "comparison": frame_arm_comparison(
                    frame, instances, arm_runs["arms"]["legacy"],
                    arm_runs["arms"]["fix"]),
                "core_evidence": arm_runs["core_evidence"],
            }
            for arm in ARMS:
                run = arm_runs["arms"][arm]
                matched_quality = {}
                for gi, pi, _ in matches:
                    inputs = run["inputs"]
                    matched_quality[str(gi)] = (
                        float(inputs[pi].quality[0])
                        if pi < len(inputs) and inputs[pi] is not None
                        else None)
                for cid in candidate_ids:
                    results = run["outcomes"][cid]
                    fallback_entry = production_fallback[arm][cid]
                    for outcome in results:
                        fallback_entry["predictions"] += 1
                        fallback_entry["used_fallback"] += int(
                            bool(outcome.used_fallback))
                    unclassified = {
                        gi for gi, pi, _ in matches
                        if not results[pi].valid}
                    prod_records[arm][cid].append(_build_record(
                        frame_id=stem, group=group, gt_classes=gt_classes,
                        matches=matches, results=results,
                        unclassified=unclassified,
                        matched_quality=matched_quality))
                    spec = classifiers[cid].candidate_spec
                    for pi, result in enumerate(results):
                        matched_gt = matched_by_pred.get(pi)
                        true_class = None
                        if matched_gt is not None:
                            true_class = next(
                                int(g["class_id"]) for g in frame["gt"]
                                if int(g["gt_id"]) == matched_gt)
                        out_row = {
                            "arm": arm, "candidate_id": cid,
                            "family": spec.family, "seed": spec.seed,
                            "fold": fold, "frame_id": stem, "group": group,
                            "pred_instance_id": pi,
                            "matched_gt_id": matched_gt,
                            "true_class": true_class,
                            "predicted_class": result.class_id
                            if result.valid else None,
                            "final_logits": list(result.raw_logits or []),
                            "status": "valid" if result.valid else "failed",
                            "used_fallback": bool(result.used_fallback),
                            "fallback_reason": result.fallback_reason,
                            "valid_ratio": float(run["inputs"][pi].quality[0])
                            if pi < len(run["inputs"]) else None,
                            "model_identity": result.model_identity,
                            "error": None if result.valid
                            else result.invalid_reason,
                        }
                        all_rows.append(out_row)
                        emit(out_row)
                for pi, inp in enumerate(run["inputs"]):
                    local = run["cs"][pi]
                    measurement_rows.append({
                        "arm": arm, "frame_id": stem,
                        "instance_id": pi,
                        "matched": pi in matched_by_pred,
                        "quality": [float(v) for v in inp.quality],
                        "polar_invalid": bool(inp.polar_invalid),
                        "invalid_reason": inp.invalid_reason,
                        "s_valid": bool(local["s_valid"]),
                        "s_missing_reason": local.get("reason", ""),
                        "c14": [float(v) for v in local["c_features"]],
                        "s16": [float(v) if np.isfinite(v) else None
                                for v in local["s_features"]],
                    })
            frames_out.append(frame_row)
            handle.flush()
            log(f"frames {index + 1}/{len(val_rows)}: {stem}")
    except BudgetExceeded as exc:
        frame_loop_budget = str(exc)
        log(f"budget exhausted inside the frame loop: {exc}")
    finally:
        handle.close()

    # per-frame evidence is dumped even when the budget stopped the loop;
    # the completed predictions_arms.jsonl rows are already on disk
    # (streamed and flushed per frame)
    dump_jsonl(ctx.output_dir / "frames_arms.jsonl", frames_out)
    dump_jsonl(ctx.output_dir / "measurement_inputs.jsonl", measurement_rows)
    if frame_loop_budget is not None:
        raise BudgetExceeded(frame_loop_budget)
    formal = ctx.max_frames is None
    # complete expected KEY SET, not just a row count
    expected_keys = set()
    for row in val_rows:
        frame = admission["frame_by_id"][row["output_stem"]]
        for pred in frame["predictions"]:
            for cid in candidate_ids:
                for arm in ARMS:
                    expected_keys.add(
                        (arm, cid, frame["frame_id"],
                         int(pred["pred_instance_id"])))
    if len(all_rows) != len(expected_keys):
        raise IdentityError(
            f"prediction row count {len(all_rows)} != expected unique key "
            f"count {len(expected_keys)} (the formal run pins "
            f"{len(expected_keys)} rows)")
    problems = prediction_key_problems(expected_keys, all_rows)
    if problems:
        raise IdentityError(
            f"prediction key set incomplete: {problems[:3]}")
    if legacy_problem_frames:
        raise IdentityError(
            f"low-resolution core mismatch on: {legacy_problem_frames}")

    ctx.watch.check("before legacy reproduction check")
    reproduction = verify_legacy_reproduction(
        [r for r in all_rows if r["arm"] == "legacy"], historical,
        candidate_ids, float(cfg["tolerances"]["logit_atol"]),
        {row["output_stem"] for row in val_rows})
    dump_json(ctx.output_dir / "legacy_reproduction.json", reproduction)
    if reproduction["problem_count"]:
        raise ReproductionError(
            f"legacy arm did not reproduce the saved results "
            f"({reproduction['problem_count']} problems, first: "
            f"{reproduction['problems'][0]}); the formal impact comparison "
            "is withheld")
    ctx.stage("legacy_reproduction", "COMPLETED",
              checked=reproduction["checked"])

    r0_check = verify_r0_invariance(all_rows, r0_id)
    dump_json(ctx.output_dir / "r0_invariance.json", r0_check)
    if r0_check["problem_count"]:
        raise IdentityError(
            f"R0 changed across arms: {r0_check['problems']}")
    ctx.stage("r0_invariance", "COMPLETED", pairs=r0_check["pairs"])

    ctx.watch.check("before metrics")
    metrics = {arm: {cid: aggregate_metrics(prod_records[arm][cid],
                                            CLASS_NAMES)
                     for cid in candidate_ids} for arm in ARMS}
    per_group = {
        arm: {cid: {group: aggregate_metrics(group_records, CLASS_NAMES)
                    for group, group_records in sorted(
                        _group_frames(prod_records[arm][cid]).items())}
              for cid in candidate_ids} for arm in ARMS}
    dump_json(ctx.output_dir / "group_metrics.json", {
        "note": "per capture-group metrics for BOTH arms and all nine "
                "candidates; four val groups exist and val participated in "
                "historical development - descriptive only",
        "by_arm": per_group,
    })
    independent = independent_recompute(
        [admission["frame_by_id"][row["output_stem"]] for row in val_rows],
        all_rows, candidate_ids, ARMS)
    comparison = compare_production_independent(
        metrics, independent, production_fallback, candidate_ids)
    dump_json(ctx.output_dir / "metrics_summary.json", {
        "production": metrics,
        "independent": independent,
        "production_vs_independent": comparison,
        "fallback_production_classifier_outcomes": production_fallback,
        "compared_fields": comparison["compared_fields"],
    })
    if comparison["mismatches"]:
        raise IdentityError(
            f"independent recomputation disagrees: "
            f"{comparison['mismatches'][:5]}")
    ctx.stage("metrics", "COMPLETED",
              production_vs_independent_mismatches=0)

    ctx.watch.check("before impact summary")
    paired = paired_fix_minus_legacy(all_rows, candidate_ids)
    within_arm: dict[str, dict] = {}
    for arm in ARMS:
        for family in ("linear_local", "conditional"):
            seeds = (2026,) if family == "linear_local" \
                else (2026, 2027, 2028)
            for seed in seeds:
                rcs = _pick(classifiers, family, "RCS", seed)
                rc = _pick(classifiers, family, "RC", seed)
                within_arm[f"{arm}:{rcs}_minus_RC"] = paired_changes(
                    prod_records[arm][rc], prod_records[arm][rcs])
                within_arm[f"{arm}:{rcs}_minus_R0"] = paired_changes(
                    prod_records[arm][r0_id], prod_records[arm][rcs])
    cs_summary = summarize_cs_changes(measurement_rows)
    ctx.watch.check("before impact publish")
    dump_json(ctx.output_dir / "impact_summary.json", {
        "paired_fix_minus_legacy": {
            cid: {k: v for k, v in out.items() if k != "detail"}
            for cid, out in paired.items()},
        "paired_detail": {cid: out["detail"]
                          for cid, out in paired.items()},
        "paired_semantics": "fix-vs-legacy pairs cover the 346 MATCHED "
                            "predictions only; the production within-arm "
                            "paired_changes cover all 368 GT including the "
                            "22 missed (a missed GT is a pair member with "
                            "no prediction) - the two counts must never be "
                            "mixed",
        "unmatched_class_changes": unmatched_class_changes(
            all_rows, candidate_ids),
        "within_arm_same_seed": {
            k: {key: value for key, value in outcome.items()
                if key != "changed"}
            for k, outcome in within_arm.items()},
        "cs_input_changes": cs_summary,
        "pooled_pixel_p95": "NOT_COMPUTED",
        "attribution_note": "the fix changes the q/polar inputs; RC/RCS "
                            "input changes are covariates of that fix, and "
                            "only prediction flips count as improved or "
                            "broken",
    })
    ctx.stage("impact", "COMPLETED",
              rows=len(all_rows),
              improved_broken={cid: (out["improved"], out["broken"])
                               for cid, out in paired.items()})

    ctx.watch.check("before reference-run verification")
    reference = verify_reference_run(all_rows, cfg, PROJECT_ROOT)
    dump_json(ctx.output_dir / "reference_run_check.json", reference)
    if reference["problem_count"]:
        raise IdentityError(
            f"the new run diverges from the retained formal_v2 run: "
            f"{reference['problems'][:3]}; per the frozen contract the "
            "repair must not change any classification result - stopping")
    ctx.stage("reference_check", "COMPLETED", pairs=reference["pairs"])


def compare_production_independent(metrics: dict, independent: dict,
                                   production_fallback: dict,
                                   candidate_ids: list[str]) -> dict:
    """Field-by-field equality between the production aggregate and the
    independent pure-counting recomputation — per-class tp/fp/fn, P/R/F1,
    summary metrics, confusion matrix, support and classification-failure
    counts, and the fallback tally (classifier outcomes on the production
    side vs prediction rows on the independent side, same all-predictions
    denominator). ``compared_fields`` lists exactly the comparisons that
    were executed."""
    names = list(CLASS_NAMES)
    mismatches: list[str] = []
    checked = 0

    def num(where: str, mine, theirs) -> None:
        nonlocal checked
        checked += 1
        if abs(float(mine) - float(theirs)) > 1e-9:
            mismatches.append(f"{where}: {mine} != {theirs}")

    def ints(where: str, mine, theirs) -> None:
        nonlocal checked
        checked += 1
        if int(mine) != int(theirs):
            mismatches.append(f"{where}: {mine} != {theirs}")

    for arm in ARMS:
        for cid in candidate_ids:
            prod = metrics[arm][cid]
            ind = independent[arm][cid]
            cond = prod["conditional_classification"]
            e2e = prod["end_to_end"]
            tag = f"{arm}/{cid}"
            ints(f"{tag}: conditional.samples", cond["samples"],
                 ind["samples"])
            num(f"{tag}: conditional.accuracy", cond["accuracy"],
                ind["accuracy"])
            num(f"{tag}: conditional.macro_f1", cond["macro_f1"],
                ind["macro_f1"])
            num(f"{tag}: end_to_end.macro_f1", e2e["macro_f1"],
                ind["end_to_end_macro_f1"])
            for i, name in enumerate(names):
                pc = cond["per_class"][name]
                ic = ind["conditional_per_class"][i]
                for key in ("precision", "recall", "f1"):
                    num(f"{tag}: conditional.per_class[{name}].{key}",
                        pc[key], ic[key])
                for key, vector in (("tp", ind["conditional_tp"]),
                                    ("fp", ind["conditional_fp"]),
                                    ("fn", ind["conditional_fn"])):
                    ints(f"{tag}: conditional.per_class[{name}].{key}",
                         pc[key], vector[i])
                pe = e2e["per_class"][name]
                ie = ind["end_to_end_per_class"][i]
                for key in ("precision", "recall", "f1"):
                    num(f"{tag}: end_to_end.per_class[{name}].{key}",
                        pe[key], ie[key])
                for key, vector in (("tp", ind["e2e_tp"]),
                                    ("fp", ind["e2e_fp"]),
                                    ("fn", ind["e2e_fn"])):
                    ints(f"{tag}: end_to_end.per_class[{name}].{key}",
                         pe[key], vector[i])
            if cond["confusion_matrix"] != ind["confusion"]:
                mismatches.append(f"{tag}: conditional confusion differs")
            checked += 1
            counts = prod["counts"]
            support = ind["support"]
            for prod_key, ind_key in (
                    ("frames", "evaluated_frames"),
                    ("gt_instances", "gt_instances"),
                    ("predictions", "predictions"),
                    ("matched", "matched"),
                    ("missed_gt", "missed_gt"),
                    ("unmatched_predictions", "unmatched_predictions"),
                    ("unclassified_matched", "unclassified_matched")):
                ints(f"{tag}: support.{prod_key}", counts[prod_key],
                     support[ind_key])
            prod_fb = production_fallback[arm][cid]
            ints(f"{tag}: fallback.used_fallback (classifier outcomes vs "
                 f"prediction rows)", prod_fb["used_fallback"],
                 ind["used_fallback"])
            ints(f"{tag}: fallback.predictions (all-predictions denominator)",
                 prod_fb["predictions"], ind["predictions_total"])
    compared_fields = [
        "conditional samples/accuracy/macro_f1",
        "conditional per-class precision/recall/f1",
        "conditional per-class tp/fp/fn",
        "conditional confusion matrix",
        "end_to_end macro_f1 and per-class precision/recall/f1",
        "end_to_end per-class tp/fp/fn",
        "support counts (evaluated frames, gt_instances, predictions, "
        "matched, missed_gt, unmatched_predictions, unclassified_matched)",
        "used_fallback: classifier-outcome tally (production side) vs "
        "prediction-row tally (independent side) over the same "
        "all-predictions denominator",
    ]
    return {"checked": checked, "mismatches": mismatches,
            "compared_fields": compared_fields}


# ---------------------------------------------------------------------------
# Reproduction, R0 invariance, summaries


def prediction_key_problems(expected_keys: set, rows: list[dict]) -> list[str]:
    """The complete expected key set (arm x candidate x frame x prediction
    from the authoritative frames) must match the emitted rows exactly;
    duplicates, missing keys and extra keys are all rejected — a row
    count alone can hide a substitution."""
    seen: set = set()
    duplicate_keys: list = []
    for row in rows:
        key = (row["arm"], row["candidate_id"], row["frame_id"],
               int(row["pred_instance_id"]))
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
    actual_keys = seen
    problems = []
    if duplicate_keys:
        problems.append(f"{len(duplicate_keys)} duplicate keys, first "
                        f"{sorted(duplicate_keys)[:5]}")
    missing_keys = sorted(expected_keys - actual_keys)[:10]
    extra_keys = sorted(actual_keys - expected_keys)[:10]
    if missing_keys:
        problems.append(f"{len(expected_keys - actual_keys)} missing keys, "
                        f"first {missing_keys}")
    if extra_keys:
        problems.append(f"{len(actual_keys - expected_keys)} unexpected "
                        f"keys, first {extra_keys}")
    return problems



def verify_legacy_reproduction(legacy_rows: list[dict], historical: dict,
                               candidate_ids: list[str], atol: float,
                               evaluated_frames: set) -> dict:
    """The legacy arm must reproduce the saved 180-frame results before any
    fix-vs-legacy comparison is published: per-prediction classes exactly,
    logits within the frozen tolerance, fallback flags equal."""
    problems = []
    checked = 0
    for row in legacy_rows:
        key = (row["candidate_id"], row["frame_id"],
               row["pred_instance_id"])
        hist = historical.get(key)
        if hist is None:
            problems.append(f"{key}: no historical row")
            continue
        checked += 1
        if row["status"] != hist.get("status"):
            problems.append(
                f"{key}: status {row['status']!r} != "
                f"{hist.get('status')!r}")
            continue
        if row["predicted_class"] != hist.get("predicted_class"):
            problems.append(
                f"{key}: class {row['predicted_class']!r} != "
                f"{hist.get('predicted_class')!r}")
            continue
        if bool(row["used_fallback"]) != bool(hist.get("used_fallback")):
            problems.append(
                f"{key}: used_fallback {row['used_fallback']} != "
                f"{hist.get('used_fallback')}")
        mine = row.get("final_logits") or []
        theirs = hist.get("final_logits") or []
        if len(mine) != len(theirs):
            problems.append(f"{key}: logits length {len(mine)} != "
                            f"{len(theirs)}")
        elif mine:
            diff = float(np.max(np.abs(
                np.asarray(mine, dtype=np.float64)
                - np.asarray(theirs, dtype=np.float64))))
            if diff > atol:
                problems.append(f"{key}: logits max abs diff {diff} > "
                                f"{atol}")
    return {"checked": checked, "problems": problems[:20],
            "problem_count": len(problems),
            "atol": atol,
            "frames": sorted(evaluated_frames)}


def verify_r0_invariance(rows: list[dict], r0_id: str) -> dict:
    """R0 reads only the gray crop and the mask: its predictions and logits
    must be IDENTICAL across arms. A difference is an implementation
    failure, never a measurement effect."""
    by_key: dict[tuple, dict] = {}
    problems = []
    for row in rows:
        if row["candidate_id"] != r0_id:
            continue
        key = (row["frame_id"], row["pred_instance_id"])
        if key not in by_key:
            by_key[key] = {"legacy": row, "fix": None}
        else:
            by_key[key]["fix"] = row
    for key, pair in by_key.items():
        legacy_row, fix_row = pair["legacy"], pair["fix"]
        if fix_row is None:
            problems.append(f"{key}: missing fix-arm R0 row")
            continue
        if legacy_row["predicted_class"] != fix_row["predicted_class"]:
            problems.append(f"{key}: R0 class changed across arms")
        l_logits = np.asarray(legacy_row["final_logits"] or [], dtype=np.float64)
        f_logits = np.asarray(fix_row["final_logits"] or [], dtype=np.float64)
        if l_logits.shape != f_logits.shape or \
                not np.array_equal(l_logits, f_logits):
            problems.append(f"{key}: R0 logits changed across arms")
    return {"pairs": len(by_key), "problems": problems[:20],
            "problem_count": len(problems)}


def summarize_cs_changes(measurement_rows: list[dict]) -> dict:
    """C/S input-distribution changes between arms. These are INPUT
    changes recorded as covariates: they must never be reported as a
    polarization content gain by themselves."""
    by_key: dict[tuple, dict] = {}
    for row in measurement_rows:
        by_key[(row["arm"], row["frame_id"], row["instance_id"])] = row
    s_flips = {"legacy_valid_fix_invalid": 0, "legacy_invalid_fix_valid": 0,
               "both_valid": 0, "both_invalid": 0}
    c_deltas = []
    s_deltas_on_common_valid = []
    for key, legacy_row in by_key.items():
        if key[0] != "legacy":
            continue
        fix_row = by_key.get(("fix", key[1], key[2]))
        if fix_row is None:
            continue
        lv, fv = bool(legacy_row["s_valid"]), bool(fix_row["s_valid"])
        if lv and fv:
            s_flips["both_valid"] += 1
            c_deltas.append(float(np.max(np.abs(
                np.asarray(legacy_row["c14"], dtype=np.float64)
                - np.asarray(fix_row["c14"], dtype=np.float64)))))
            s_deltas_on_common_valid.append(float(np.max(np.abs(
                np.asarray(legacy_row["s16"], dtype=np.float64)
                - np.asarray(fix_row["s16"], dtype=np.float64)))))
        elif lv:
            s_flips["legacy_valid_fix_invalid"] += 1
        elif fv:
            s_flips["legacy_invalid_fix_valid"] += 1
        else:
            s_flips["both_invalid"] += 1
    return {
        "instances": len(by_key) // 2,
        "s_valid_flips": s_flips,
        "c14_max_abs_delta_mean_over_instances": (
            float(np.mean(c_deltas)) if c_deltas else None),
        "s16_max_abs_delta_mean_over_instances_both_valid": (
            float(np.mean(s_deltas_on_common_valid))
            if s_deltas_on_common_valid else None),
        "note": "input-distribution covariates only; the classifier was "
                "frozen and no gain is claimed from them",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="material_upsample_impact_v1 evaluation entry")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output-dir", required=True,
                        help="final output directory; must NOT exist")
    parser.add_argument("--device", default="0")
    parser.add_argument("--budget-minutes", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="bounded smoke only; the formal run passes "
                             "nothing and uses all 180 val frames")
    args = parser.parse_args(argv)
    if not (args.budget_minutes > 0 and np.isfinite(args.budget_minutes)):
        log("invalid --budget-minutes")
        return EXIT_USAGE
    if args.max_frames is not None and (
            type(args.max_frames) is not int or args.max_frames <= 0):
        log("--max-frames must be a positive int")
        return EXIT_USAGE

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        log(f"config not found: {config_path}")
        return EXIT_USAGE
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"config parse failed: {exc}")
        return EXIT_USAGE
    if not isinstance(config, dict) or \
            config.get("experiment") != "material_upsample_impact_v1":
        log("config is not material_upsample_impact_v1")
        return EXIT_USAGE

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists():
        log(f"refusing: output directory already exists: {output_dir}")
        return EXIT_USAGE
    if is_link_or_reparse(output_dir) or is_link_or_reparse(output_dir.parent):
        log("refusing: output directory or parent is a link/reparse point")
        return EXIT_USAGE
    rel_output = output_dir.relative_to(PROJECT_ROOT).as_posix()
    probe = subprocess.run(["git", "check-ignore", "-q", "--", rel_output],
                           cwd=str(PROJECT_ROOT))
    if probe.returncode != 0:
        log(f"refusing: output directory {rel_output!r} is not gitignored")
        return EXIT_USAGE
    output_dir.mkdir(parents=True)

    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_" \
             f"{uuid.uuid4().hex[:8]}"
    ctx = RunContext(
        config=config, output_dir=output_dir, mode=args.mode,
        device=str(args.device), budget_minutes=float(args.budget_minutes),
        max_frames=args.max_frames, run_id=run_id,
        watch=BudgetWatch(float(args.budget_minutes)),
        registry=SourceRegistry({}), config_path=config_path)

    failure = None
    interrupted = False
    try:
        try:
            if args.mode == "preflight":
                run_preflight(ctx)
            else:
                run_evaluate(ctx)
        except BudgetExceeded as exc:
            interrupted = True
            # the overrun has judged the run PARTIAL; the remaining
            # closeout runs budget-exempt so the evidence is published
            ctx.watch = None
            log(f"budget exhausted: {exc}")
        except (IdentityError, ReproductionError, FileNotFoundError,
                ValueError, OSError, KeyError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            log(f"failure: {failure}")
    except Exception as exc:  # unexpected: still write a status file
        failure = f"UNEXPECTED {type(exc).__name__}: {exc}"
        log(f"unexpected failure: {failure}")
    return closeout_and_finalize(ctx, interrupted=interrupted,
                                 failure=failure)


def closeout_and_finalize(ctx: RunContext, *, interrupted: bool,
                          failure: str | None) -> int:
    """Publish phase: protection snapshot, manifest and status. Once the
    budget is exhausted the remaining closeout runs budget-exempt, but an
    overrun DURING the publish phase itself forces PARTIAL — a run whose
    budget dies before its status has been judged never reports SUCCESS.
    The manifest is written even when the snapshot reports problems, so
    missing/changed inputs stay visible on disk."""
    protection = None

    def watch_check(at: str) -> None:
        if ctx.watch is not None:
            ctx.watch.check(at)

    def publish_protection() -> dict:
        published = ctx.registry.compare()
        ctx.registry.write_manifest(ctx.output_dir / "source_manifest.json",
                                    published)
        ctx.stage("protection",
                  "COMPLETED" if published["identical"] else "FAILED",
                  registered=published["registered_sources"],
                  problems=published["problems"])
        return published

    try:
        watch_check("before protection snapshot")
        protection = publish_protection()
        watch_check("after protection snapshot")
        watch_check("before finalize")
        if not protection["identical"] and failure is None:
            failure = "protection check found changed inputs"
    except BudgetExceeded as exc:
        interrupted = True
        ctx.watch = None
        log(f"budget exceeded during publish: {exc}")
        try:
            protection = publish_protection()
            if not protection["identical"] and failure is None:
                failure = "protection check found changed inputs"
        except Exception as exc2:
            log(f"closeout after publish overrun failed: {exc2}")
            if failure is None:
                failure = f"CLOSEOUT {type(exc2).__name__}: {exc2}"
    except Exception as exc:
        log(f"closeout failed: {exc}")
        if failure is None:
            failure = f"CLOSEOUT {type(exc).__name__}: {exc}"
    return finalize(ctx, interrupted=interrupted, failure=failure,
                    protection=protection)


def finalize(ctx: RunContext, *, interrupted: bool, failure: str | None,
             protection: dict | None) -> int:
    failed = [s for s, rec in ctx.stages.items() if rec["status"] == "FAILED"]
    partial = [s for s, rec in ctx.stages.items() if rec["status"] == "PARTIAL"]
    if failure:
        status, exit_code = "FAILED", EXIT_FAILED
    elif interrupted or partial:
        status, exit_code = "PARTIAL", EXIT_PARTIAL
    elif failed:
        status, exit_code = "FAILED", EXIT_FAILED
    else:
        status, exit_code = "COMPLETED", EXIT_OK
    dump_json(ctx.output_dir / "run_status.json", {
        "run_id": ctx.run_id, "mode": ctx.mode, "device": ctx.device,
        "budget_minutes": ctx.budget_minutes,
        "max_frames": ctx.max_frames,
        "smoke": ctx.max_frames is not None,
        "stages": ctx.stages,
        "interrupted_by_budget": interrupted,
        "failure": failure,
        "overall_status": status,
        "exit_code": exit_code,
        "finished_at": utc_now(),
        "protection": protection,
    })
    log(f"overall: {status} (exit {exit_code})")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
