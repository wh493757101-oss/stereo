"""Evaluate a trained material-fusion checkpoint with optional fallback routing.

Loads a checkpoint written by ``scripts/train_material_fusion.py``,
resolves the *actual run configuration* (the run's ``run_config.json``
next to the checkpoint, or an explicit ``--run-config``; only when neither
exists does it fall back to the experiment template plus explicit CLI
overrides), admits the actual dataset read-only, and reports two separate
metric sets:

- ``pure_classifier``: the raw CMX/material-fusion classifier on every
  sample. Always available; never loads or calls the fallback.
- ``routed`` (dual variants only, unless ``--no-fallback``): samples whose
  polarization measurement is invalid (``valid_ratio == 0`` in the V4
  quality vector) bypass the model and are classified by the independent
  historical gray Model B; valid samples use the material-fusion model.
  Mixed batches keep the exact original sample order. When routing is off
  the report says so explicitly and carries no routed metrics.

The gray-only variant (E1) has no polarization input, so routing is not
applicable and the historical gray weights are never loaded for it.

Usage (per variant; ``<run-dir>`` = runs/train/<run-id>/material_fusion):

    # E1 gray (pure, no fallback needed)
    python scripts/eval_material_fusion.py \
        --checkpoint <run-dir>/best.pt --split val --device 0 \
        --output analysis/runs/cmx_<run-id>/eval_gray_val.json

    # E2 gray_gray_valid / E3 gray_valid / E4 gray_polar (routed)
    python scripts/eval_material_fusion.py \
        --checkpoint <run-dir>/best.pt --split val --device 0 \
        --output analysis/runs/cmx_<run-id>/eval_<variant>_val.json

    # E2/E3/E4 pure-only
    python scripts/eval_material_fusion.py \
        --checkpoint <run-dir>/best.pt --split val --device 0 --no-fallback \
        --output analysis/runs/cmx_<run-id>/eval_<variant>_pure_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import QUALITY_VECTOR_KEYS
from core.material_fusion_dataset import (
    CLASS_NAMES,
    MaterialFusionDataset,
)
from core.material_mask_dataset import (
    POOLING_INSTANCE_MASK,
    admit_dataset_for_protocol,
    require_fingerprint,
)
from models.material_fusion import (
    MATERIAL_FUSION_MASK_VERSION,
    check_run_config_checkpoint_consistency,
    load_material_checkpoint,
    read_material_checkpoint_metadata,
    validate_material_config,
    variant_is_dual,
)
from models.polar_fusion import (
    GrayBackboneAdapter,
    architecture_name,
    build_gray_class_perm,
    read_manifest_split,
    torch_device_name,
)
from scripts.train_polar_fusion import classification_metrics

VALID_RATIO_INDEX = QUALITY_VECTOR_KEYS.index("valid_ratio")

# Experiment templates. The v1 template is the default for a v1/GAP
# checkpoint; a v2 checkpoint resolves to the v2 template instead, so the
# protocol metadata and the dataset path are never taken from the wrong
# protocol. An explicit --config still wins.
V1_TEMPLATE = "configs/experiments/material_fusion_v1.yaml"
V2_TEMPLATE = "configs/experiments/material_fusion_v2_mask.yaml"
DEFAULT_TEMPLATE = V1_TEMPLATE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-config", default=None,
                        help="Resolved run configuration (default: run_config.json "
                             "next to the checkpoint, when present).")
    parser.add_argument("--config", default=None,
                        help="Experiment template used only when no run config "
                             "exists (default: the template matching the "
                             "checkpoint's recorded protocol version).")
    parser.add_argument("--variant", default=None,
                        help="Explicit variant; must match the checkpoint.")
    parser.add_argument("--data", default=None, help="Dataset root override.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Split to evaluate (default: val; test is for final "
                             "reporting only, never for model selection).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fallback-weights", default=None,
                        help="Historical gray Model B checkpoint override.")
    parser.add_argument("--limit-batches", type=int, default=0,
                        help="Cap evaluation batches (0 = all); bounded smoke "
                             "evaluation uses a small cap.")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Pure classifier only; never loads the fallback.")
    parser.add_argument("--output", default=None,
                        help="JSON report path (default: <checkpoint dir>/eval_<split>.json).")
    return parser.parse_args(argv)


def _default_template_for(version: str) -> Path:
    """The experiment template matching a checkpoint's recorded protocol.

    A v2 checkpoint was trained on the V5 dataset under the mask-pooling
    protocol; using the v1 template for it would silently supply the wrong
    dataset, pooling mode and mask metadata.
    """
    if version == MATERIAL_FUSION_MASK_VERSION:
        return PROJECT_ROOT / V2_TEMPLATE
    return PROJECT_ROOT / V1_TEMPLATE


def _template_path(args: argparse.Namespace, checkpoint_metadata=None) -> Path:
    if args.config:
        template_path = Path(args.config)
        if not template_path.is_absolute():
            template_path = PROJECT_ROOT / template_path
        if not template_path.is_file():
            raise FileNotFoundError(f"experiment template not found: {template_path}")
        return template_path
    if checkpoint_metadata is not None:
        return _default_template_for(checkpoint_metadata.version)
    return PROJECT_ROOT / V1_TEMPLATE


def _derive_config_from_checkpoint(
    args: argparse.Namespace,
    checkpoint_metadata,
    template_path: Path,
) -> tuple[dict, list[str], list[str]]:
    """Config for a run without a v2 run config.

    Template values are used only for what the checkpoint cannot record
    (epochs, warmup schedule, learning rates, data location, fallback
    weights); every value the checkpoint does record (variant, imgsz,
    batches, seed, smoke markers, fingerprints, pretrained SHA,
    preprocessing/upstream pin, and the whole pooling/mask protocol) replaces
    the template default. The template itself is selected from the
    checkpoint's recorded protocol version, so a v2 checkpoint never inherits
    v1 pooling or V4 dataset defaults. Nothing is presented as a confirmed
    historical fact beyond the checkpoint fields, and a field the checkpoint
    does not record is never invented.
    """
    import yaml

    template = validate_material_config(
        yaml.safe_load(template_path.read_text(encoding="utf-8"))
    )
    derived = {
        "experiment": checkpoint_metadata.version,
        "variant": checkpoint_metadata.variant,
        "imgsz": checkpoint_metadata.imgsz,
        "batch": checkpoint_metadata.physical_batch,
        "accum_steps": checkpoint_metadata.accum_steps,
        "effective_batch": checkpoint_metadata.effective_batch,
        "seed": checkpoint_metadata.seed,
        "smoke": checkpoint_metadata.smoke,
        "limit_batches": checkpoint_metadata.limit_batches,
        "data_fingerprint": checkpoint_metadata.data_fingerprint,
        "pretrained_sha256": checkpoint_metadata.pretrained_sha256,
        "preprocessing_version": checkpoint_metadata.preprocessing_version,
        "upstream_commit": checkpoint_metadata.upstream_commit,
        "pooling_mode": checkpoint_metadata.pooling_mode,
        "mask_schema": checkpoint_metadata.mask_schema,
        "mask_resize_version": checkpoint_metadata.mask_resize_version,
        "parent_data_fingerprint": checkpoint_metadata.parent_data_fingerprint,
    }
    config = {**template, **derived}
    if args.data is not None:
        config["data"] = str(args.data)
    if args.variant is not None:
        config["variant"] = args.variant
    config = validate_material_config(config)
    unasserted = [
        "epochs", "warmup_epochs", "lr", "encoder_lr", "weight_decay",
        "data", "fallback_weights",
    ]
    return config, sorted(derived), unasserted


def resolve_eval_config(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_metadata,
) -> tuple[dict, dict, bool]:
    """The run configuration for this checkpoint, plus its provenance.

    Priority: explicit ``--run-config`` > ``run_config.json`` next to the
    checkpoint > experiment template. A v2 run config (with a resolved
    ``config`` section) is the actual run configuration and everything is
    asserted against it; an explicit ``--variant`` that contradicts it is
    refused. Legacy run configs and the template only supply the values
    the checkpoint cannot record, and the provenance says so. The
    checkpoint's recorded variant must match in every case. Returns
    ``(config, provenance, assert_epoch_stage)``.
    """
    checkpoint_variant = checkpoint_metadata.variant
    if args.run_config:
        run_config_path = Path(args.run_config)
        if not run_config_path.is_absolute():
            run_config_path = PROJECT_ROOT / run_config_path
        if not run_config_path.is_file():
            raise FileNotFoundError(f"--run-config not found: {run_config_path}")
    else:
        run_config_path = checkpoint_path.parent / "run_config.json"

    if run_config_path.is_file():
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        if "config" in payload:
            config = validate_material_config(payload["config"])
            if args.variant is not None and args.variant != config["variant"]:
                raise ValueError(
                    f"--variant {args.variant!r} conflicts with the actual run "
                    f"configuration variant {config['variant']!r}; refusing"
                )
            if args.data is not None:
                config = validate_material_config({**config, "data": str(args.data)})
            provenance = {
                "source": "run_config",
                "path": str(run_config_path),
                "run_id": str(payload.get("run_id", "")),
                "derived_fields": [],
                "unasserted_fields": [],
            }
            assert_epoch_stage = True
        else:
            config, derived, unasserted = _derive_config_from_checkpoint(
                args, checkpoint_metadata,
                _template_path(args, checkpoint_metadata),
            )
            provenance = {
                "source": "run_config_legacy",
                "path": str(run_config_path),
                "run_id": str(payload.get("run_id", "")),
                "derived_fields": derived,
                "unasserted_fields": unasserted,
            }
            assert_epoch_stage = False
    else:
        config, derived, unasserted = _derive_config_from_checkpoint(
            args, checkpoint_metadata, _template_path(args, checkpoint_metadata)
        )
        provenance = {
            "source": "template",
            "path": str(_template_path(args, checkpoint_metadata)),
            "run_id": "",
            "derived_fields": derived,
            "unasserted_fields": unasserted,
        }
        assert_epoch_stage = False

    if config["variant"] != checkpoint_variant:
        raise ValueError(
            f"resolved variant {config['variant']!r} does not match the "
            f"checkpoint variant {checkpoint_variant!r}"
        )
    return config, provenance, assert_epoch_stage


def load_fallback_gray(
    path: str | Path,
    device: str,
    imgsz: int,
    num_classes: int = len(CLASS_NAMES),
) -> tuple[GrayBackboneAdapter, str]:
    """Load and identity-check the historical gray Model B fallback.

    Returns the adapter (raw logits in canonical class order) and a short
    identity string for the report. A missing file raises
    ``FileNotFoundError``; a class-order or head-width mismatch raises
    ``ValueError`` — never a silent substitution.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"fallback gray checkpoint {path} is missing; the fallback "
            "routing validation cannot run without it (no automatic "
            "substitution)"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    names = getattr(module, "names", None)
    if not names:
        raise ValueError(
            f"fallback checkpoint {path} has no class names; cannot verify its "
            "class order"
        )
    perm = build_gray_class_perm(names, CLASS_NAMES)
    module = module.to(device).eval()
    probe = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        raw = GrayBackboneAdapter._unwrap(module(probe))
    if tuple(raw.shape) != (1, len(names)) or len(names) != num_classes:
        raise ValueError(
            f"fallback checkpoint {path} produced {tuple(raw.shape)} raw logits "
            f"for {len(names)} class names; expected (1, {num_classes}) — wrong "
            "classification head or class list"
        )
    adapter = GrayBackboneAdapter(module, perm=perm).to(device).eval()
    identity = (
        f"{type(module).__name__} architecture={architecture_name(module) or 'unknown'} "
        f"classes={list(CLASS_NAMES)}"
    )
    return adapter, identity


def _unpack_eval_batch(batch, dual: bool, device: str, with_instance_mask: bool = False):
    """Move one batch to the device; labels stay on the CPU.

    Returns ``(rgb, aux, quality, labels, instance_mask)``; the mask is
    ``None`` unless the loader carries one, so the model never receives a
    mask it did not ask for.
    """
    from core.material_fusion_dataset import unpack_batch

    rgb, aux, quality, labels, mask = unpack_batch(batch, dual, with_instance_mask)
    if rgb is not None:
        rgb = rgb.to(device)
    if aux is not None:
        aux = aux.to(device)
    if quality is not None:
        quality = quality.to(device)
    if mask is not None:
        mask = mask.to(device)
    return rgb, aux, quality, labels, mask


@torch.no_grad()
def route_batch(model, fallback, rgb, aux, invalid, instance_mask=None):
    """True-bypass polar-invalid routing for one batch, preserving order.

    Invalid-polar samples are forwarded through the independent historical
    gray fallback *only*; valid samples through the material-fusion model
    *only*. Results are scattered back by absolute index so mixed batches
    keep the original sample order. ``instance_mask`` is forwarded only when
    the model pools with one.

    This is the production routing used by :func:`evaluate_with_fallback`,
    and it is also what the bounded smoke verification checks directly.
    Returns ``(predictions, routed_logits, used_fallback)`` where
    ``used_fallback`` marks exactly the samples that went through the
    fallback.
    """
    invalid = invalid.bool()
    valid = ~invalid
    valid_logits = None
    invalid_logits = None
    if valid.any():
        sub = rgb[valid]
        sub_aux = aux[valid] if aux is not None else None
        sub_mask = instance_mask[valid] if instance_mask is not None else None
        if sub_mask is None:
            valid_logits = model(sub, sub_aux) if model.is_dual else model(sub)
        else:
            valid_logits = model(sub, sub_aux, sub_mask)
    if invalid.any():
        invalid_logits = fallback(rgb[invalid])
    reference = valid_logits if valid_logits is not None else invalid_logits
    if reference is None:
        raise ValueError("route_batch received an empty batch")
    routed_logits = torch.empty(
        rgb.shape[0], reference.shape[1], dtype=reference.dtype, device=reference.device
    )
    if valid_logits is not None:
        routed_logits[valid] = valid_logits
    if invalid_logits is not None:
        routed_logits[invalid] = invalid_logits
    return routed_logits.argmax(dim=1), routed_logits, invalid


@torch.no_grad()
def evaluate_with_fallback(
    model,
    fallback,
    loader,
    device: str,
    num_classes: int = len(CLASS_NAMES),
    limit_batches: int = 0,
    routing: bool = True,
) -> dict:
    """Pure and (optionally) routed metrics over one loader.

    ``routing=False`` is pure-only: the fallback is never called (it may be
    ``None``) and the report marks routing as not applied with ``routed``
    set to ``None`` — no fabricated routed metrics. With ``routing=True``
    invalid-polar samples bypass the model and go through the fallback
    only, valid samples through the model only, scattered back by absolute
    index so mixed batches keep the original order.
    """
    if routing and fallback is None:
        raise ValueError("routing requires a fallback model")
    model.eval()
    if fallback is not None:
        fallback.eval()
    labels_all: list[int] = []
    pure_preds: list[int] = []
    routed_preds: list[int] = []
    invalid_flags: list[bool] = []
    evaluated_batches = 0
    for batch_index, batch in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        evaluated_batches += 1
        # A model without the property (a stub, or a legacy wrapper) pools
        # with GAP and therefore receives no mask.
        with_mask = bool(getattr(model, "uses_instance_mask", False))
        rgb, aux, quality, labels, instance_mask = _unpack_eval_batch(
            batch, model.is_dual, device, with_mask
        )
        labels_all.extend(labels.tolist())
        pure_logits = (
            model(rgb, aux, instance_mask) if instance_mask is not None
            else (model(rgb, aux) if model.is_dual else model(rgb))
        )
        pure_preds.extend(pure_logits.argmax(dim=1).cpu().tolist())

        invalid = quality[:, VALID_RATIO_INDEX] <= 0
        invalid_flags.extend(bool(flag) for flag in invalid.tolist())
        if routing:
            routed, _, _ = route_batch(
                model, fallback, rgb, aux, invalid, instance_mask
            )
            routed_preds.extend(routed.tolist())

    invalid_count = sum(invalid_flags)
    result: dict = {
        "samples": len(labels_all),
        "evaluated_batches": evaluated_batches,
        "invalid_samples": invalid_count,
        "invalid_ratio": invalid_count / max(len(labels_all), 1),
        "pure_classifier": classification_metrics(labels_all, pure_preds, num_classes),
        "routing": {
            "applied": bool(routing),
            "invalid_samples": invalid_count,
            "invalid_ratio": invalid_count / max(len(labels_all), 1),
        },
        "routed": None,
    }
    if routing:
        routed_metrics = classification_metrics(labels_all, routed_preds, num_classes)
        changed = sum(1 for pure, routed_p in zip(pure_preds, routed_preds) if pure != routed_p)
        result["routing"]["changed_predictions"] = changed
        result["routing"]["changed_ratio"] = changed / max(len(labels_all), 1)
        result["routed"] = routed_metrics
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    output = (
        Path(args.output) if args.output else checkpoint.parent / f"eval_{args.split}.json"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        print(
            f"evaluation refused: report {output} already exists; use a new "
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
        # Strict config <-> checkpoint contract (the same one verify uses).
        consistency = check_run_config_checkpoint_consistency(
            config, metadata.__dict__, assert_epoch_stage=assert_epoch_stage
        )

        # Actual-data admission before any model evaluation or report. The
        # gate follows the *validated* pooling protocol: a v2/instance_mask
        # run is admitted through the V5 gate (mask content, annotation
        # source, sample mapping, V4 parent, class order), a v1/GAP run
        # through the original V4 gate. The expected fingerprint is never
        # cleared, skipped or replaced with an older-format value.
        data_root = config["data"]
        if not Path(data_root).is_absolute():
            data_root = PROJECT_ROOT / data_root
        if config["pooling_mode"] != metadata.pooling_mode:
            raise ValueError(
                f"resolved config pooling_mode {config['pooling_mode']!r} does not "
                f"match the checkpoint's {metadata.pooling_mode!r}"
            )
        expected_fingerprint = config["data_fingerprint"] or metadata.data_fingerprint
        if metadata.pooling_mode == POOLING_INSTANCE_MASK:
            # A v2 evaluation must be bound to the exact V5 dataset the
            # checkpoint records; an empty expectation is refused rather than
            # read as "any V5 dataset".
            require_fingerprint(expected_fingerprint, "run data_fingerprint")
        data_report = admit_dataset_for_protocol(
            data_root,
            pooling_mode=metadata.pooling_mode,
            expected_fingerprint=expected_fingerprint,
            expected_parent_fingerprint=metadata.parent_data_fingerprint,
            expected_mask_schema=metadata.mask_schema,
            expected_mask_resize_version=metadata.mask_resize_version,
        )
        if metadata.data_fingerprint and data_report["audit_fingerprint"] != metadata.data_fingerprint:
            raise ValueError(
                f"checkpoint records dataset fingerprint {metadata.data_fingerprint} "
                f"but the evaluated data is {data_report['audit_fingerprint']}"
            )
        if config["preprocessing_version"] != metadata.preprocessing_version:
            raise ValueError(
                f"config preprocessing_version {config['preprocessing_version']!r} "
                f"does not match the checkpoint's {metadata.preprocessing_version!r}"
            )
    except (ValueError, FileNotFoundError) as exc:
        print(f"evaluation refused: {exc}", file=sys.stderr)
        return 2

    dual = variant_is_dual(variant)
    routing = dual and not args.no_fallback
    routing_reason = ""
    if not dual:
        routing_reason = (
            "gray-only variant has no polarization input; fallback routing is "
            "not applicable"
        )
    elif args.no_fallback:
        routing_reason = "disabled by --no-fallback"

    fallback = None
    fallback_identity = ""
    fallback_path = ""
    if routing:
        fallback_path = args.fallback_weights or config.get("fallback_weights", "")
        fallback_file = Path(fallback_path)
        if not fallback_file.is_absolute():
            fallback_file = PROJECT_ROOT / fallback_file
        try:
            fallback, fallback_identity = load_fallback_gray(
                fallback_file, device, metadata.imgsz
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"evaluation refused: {exc}", file=sys.stderr)
            return 2

    try:
        model, metadata, _ = load_material_checkpoint(
            checkpoint, variant=variant, imgsz=metadata.imgsz
        )
    except ValueError as exc:
        print(f"evaluation refused: {exc}", file=sys.stderr)
        return 2
    model = model.to(device)

    paths, _, _ = read_manifest_split(data_root, args.split)
    loader = torch.utils.data.DataLoader(
        MaterialFusionDataset(
            paths, variant=variant, imgsz=metadata.imgsz,
            with_instance_mask=bool(getattr(model, "uses_instance_mask", False)),
        ),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 8,
    )

    result = evaluate_with_fallback(
        model, fallback, loader, device, num_classes=4,
        limit_batches=args.limit_batches, routing=routing,
    )
    result["routing"]["reason"] = routing_reason
    result.update(
        {
            "checkpoint": str(checkpoint),
            "variant": variant,
            "split": args.split,
            "imgsz": metadata.imgsz,
            "version": metadata.version,
            "class_names": list(metadata.class_names),
            "data_fingerprint": data_report["audit_fingerprint"],
            "config_source": config_provenance,
            "config_consistency": consistency,
            "training": {
                "smoke": metadata.smoke,
                "limit_batches": metadata.limit_batches,
                "epoch": metadata.epoch,
                "stage": metadata.stage,
            },
            "evaluation": {
                "limit_batches": int(args.limit_batches),
                "samples": result["samples"],
                "batches": result["evaluated_batches"],
                "split_size": len(paths),
                "truncated": result["samples"] < len(paths),
            },
            "data": {
                "root": data_report["data_root"],
                "audit_fingerprint": data_report["audit_fingerprint"],
                "verified_files": data_report["verified_files"],
            },
            "fallback_weights": fallback_path,
            "fallback_identity": fallback_identity,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Eval report: {output}")
    print(
        f"pure macro_f1={result['pure_classifier']['macro_f1']:.4f} "
        f"acc={result['pure_classifier']['accuracy']:.4f} | "
        f"routed={'n/a' if result['routed'] is None else format(result['routed']['macro_f1'], '.4f')} "
        f"(invalid {result['invalid_samples']}/{result['samples']}, "
        f"routing={'on' if routing else 'off'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
