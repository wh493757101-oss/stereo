"""Three-stage YOLO26 training pipeline (current mainline).

Fixed serial order, each stage in its own Python process, one run id:

    1. model_a       scripts/train_model_a.py           -> runs/train/<id>/model_a/
    2. gray_fusion   scripts/train_gray_fusion.py       -> runs/train/<id>/gray_fusion/
    3. fusion_freeze scripts/train_polar_fusion.py --phase freeze
                     -> runs/train/<id>/polar_fusion/freeze/

The fusion stage always uses this run's own gray checkpoint
(``runs/train/<id>/gray_fusion/best.pt``); falling back to historical or
smoke gray weights is never allowed.

Not included by design: the four-class segmentation baseline, the old
standalone polar classification, and Fusion joint finetuning. The old
four-stage code is archived under ``scripts/legacy_four_stage/``.

Before any child process starts the pipeline checks: run id validity,
per-stage output conflicts, base checkpoints (task and architecture),
segmentation data yaml, the strict V4 dataset gate (audit + digest
fingerprint), GPU availability and parameter consistency. After each
stage it verifies the produced artifacts (weights load, architecture,
class order, imgsz, dataset sources, model-selection evidence) instead of
trusting the exit code alone.

``--smoke`` runs a bounded real-GPU chain (Model A: 1 epoch on a data
fraction; Gray/Fusion: 2 epochs x <= 3 train/val batches) and marks the
run as smoke in every config and in ``pipeline_report.json``. A smoke
gray checkpoint is never accepted as a formal fusion input.

Technical pipeline checks passing means the chain executed correctly,
never that the models are good enough: model quality and deployment
remain a separate review. ``--stages`` continues remaining stages of an
unfinished run; this is not full checkpoint resume.

Usage:
    python scripts/train_pipeline.py --run-id <run-id> --device 0
    python scripts/train_pipeline.py --run-id <run-id> --device 0 --preflight-only
    python scripts/train_pipeline.py --run-id <run-id> --device 0 --smoke
    python scripts/train_pipeline.py --run-id <run-id> --device 0 --stages gray_fusion fusion_freeze
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.polar_fusion import (
    architecture_name,
    load_fusion_checkpoint,
    read_fusion_metadata,
    rebuild_gray_backbone,
)
from scripts.train_model_a import BINARY_CLASS_NAME, _validate_binary_dataset
from scripts.train_polar_fusion import validate_dataset
from scripts.training_common import (
    CLS_BASE,
    SEG_BASE,
    DeviceUnavailableError,
    InvalidRunIdError,
    resolve_device,
    validate_run_id,
)

REPORT_VERSION = 2

STAGES = ("model_a", "gray_fusion", "fusion_freeze")
STAGE_SCRIPTS = {
    "model_a": "train_model_a.py",
    "gray_fusion": "train_gray_fusion.py",
    "fusion_freeze": "train_polar_fusion.py",
}
STAGE_SUBDIRS = {
    "model_a": Path("model_a"),
    "gray_fusion": Path("gray_fusion"),
    "fusion_freeze": Path("polar_fusion") / "freeze",
}
# Fixed class-id order (CLAUDE.md invariant 3), cross-checked against the
# V4 manifest during preflight.
CLASS_NAMES = [
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
]
DEFAULT_DATA_V4 = "datasets/underwater_cls_fusion_v4_band"
DEFAULT_SEG_DATA = "datasets/underwater_seg_binary_v2/data.yaml"

STAGE_DEFAULTS: dict[str, dict] = {
    "model_a": {"epochs": 100, "batch": 8, "imgsz": 640},
    "gray_fusion": {"epochs": 30, "batch": 32, "imgsz": 224},
    "fusion_freeze": {"epochs": 30, "batch": 32, "imgsz": 224},
}
SMOKE_LIMITS: dict[str, dict] = {
    "model_a": {"epochs": 1, "batch": 2, "imgsz": 640, "fraction": 0.01},
    "gray_fusion": {"epochs": 2, "batch": 8, "imgsz": 224, "limit_batches": 3},
    "fusion_freeze": {"epochs": 2, "batch": 8, "imgsz": 224, "limit_batches": 3},
}
PARAM_PREFIX = {
    "model_a": "model_a",
    "gray_fusion": "gray",
    "fusion_freeze": "fusion",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id shared by all three stages (runs/train/<run-id>/...).",
    )
    parser.add_argument("--device", default="0", help="Training device.")
    parser.add_argument("--seed", type=int, default=2026)
    for stage in STAGES:
        prefix = PARAM_PREFIX[stage]
        defaults = STAGE_DEFAULTS[stage]
        parser.add_argument(
            f"--{prefix}-epochs", type=int, default=None,
            help=f"{stage} epochs (default: {defaults['epochs']}).",
        )
        parser.add_argument(
            f"--{prefix}-batch", type=int, default=None,
            help=f"{stage} batch size (default: {defaults['batch']}).",
        )
        parser.add_argument(
            f"--{prefix}-imgsz", type=int, default=None,
            help=f"{stage} input size (default: {defaults['imgsz']}).",
        )
    parser.add_argument(
        "--preflight",
        "--preflight-only",
        dest="preflight_only",
        action="store_true",
        help="Run the read-only preflight checks and print the plan; no "
        "training directories or reports are created and nothing is trained.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded all-chain verification: Model A 1 epoch on a data "
        "fraction, Gray/Fusion 2 epochs x <= 3 train/val batches; the run "
        "is marked as smoke everywhere.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="Stages to run in the fixed mainline order; use to continue "
        "the remaining stages of an unfinished run (not full resume).",
    )
    args = parser.parse_args(argv)
    for stage in STAGES:
        prefix = PARAM_PREFIX[stage]
        for key in ("epochs", "batch", "imgsz"):
            value = getattr(args, f"{prefix}_{key}")
            if value is not None and value < 1:
                parser.error(f"--{prefix}-{key} must be a positive integer")
    return args


def _validate_stages(selected: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(selected)
    if not selected:
        raise ValueError("at least one training stage is required")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate training stages are not allowed")
    unknown = [stage for stage in selected if stage not in STAGES]
    if unknown:
        raise ValueError(f"unknown training stages: {unknown}")
    indices = [STAGES.index(stage) for stage in selected]
    if indices != sorted(indices):
        raise ValueError(
            f"stage order must follow the fixed mainline order {STAGES}; "
            f"got {selected}"
        )
    return selected


def resolve_params(args: argparse.Namespace) -> dict[str, dict]:
    """Resolved per-stage parameters (CLI overrides > smoke limits > defaults)."""
    params: dict[str, dict] = {}
    for stage in STAGES:
        base = dict(SMOKE_LIMITS[stage] if args.smoke else STAGE_DEFAULTS[stage])
        prefix = PARAM_PREFIX[stage]
        for key in ("epochs", "batch", "imgsz"):
            value = getattr(args, f"{prefix}_{key}")
            if value is not None:
                base[key] = value
        params[stage] = base
    return params


def stage_dir(run_id: str, stage: str) -> Path:
    return PROJECT_ROOT / "runs" / "train" / run_id / STAGE_SUBDIRS[stage]


def gray_best_path(run_id: str) -> Path:
    return stage_dir(run_id, "gray_fusion") / "best.pt"


def _probe_checkpoint(path: Path) -> dict:
    """Load a checkpoint and describe its task/architecture/output."""
    import torch
    from ultralytics import YOLO

    module = YOLO(str(path)).model
    info: dict[str, Any] = {
        "task": getattr(module, "task", None),
        "architecture": architecture_name(module),
        "names": [str(module.names[key]) for key in sorted(module.names)],
    }
    head = module.model[-1]
    if hasattr(head, "linear"):
        info["head_out_features"] = int(head.linear.out_features)
    if info["task"] == "classify":
        module.eval()
        with torch.no_grad():
            out = module(torch.zeros(2, 3, 224, 224))
        logits = out[1] if isinstance(out, tuple) else out
        info["logits_shape"] = tuple(logits.shape)
        info["logits_finite"] = bool(torch.isfinite(logits).all())
    return info


def _collect_tensors(obj) -> list:
    """Recursively collect tensors from nested tuples/lists/dicts."""
    import torch

    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (tuple, list)):
        tensors: list = []
        for item in obj:
            tensors.extend(_collect_tensors(item))
        return tensors
    if isinstance(obj, dict):
        tensors = []
        for item in obj.values():
            tensors.extend(_collect_tensors(item))
        return tensors
    return []


def _probe_seg_inference(best_path: Path, image_path: Path) -> dict:
    """Real binary-mask inference plus a finite forward probe."""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(best_path))
    results = model.predict(str(image_path), conf=0.001, verbose=False)
    result = results[0]
    detections = 0 if result.boxes is None else int(len(result.boxes))
    masks_finite = True
    masks_present = False
    if getattr(result, "masks", None) is not None and result.masks.data is not None:
        masks_present = bool(len(result.masks.data) > 0)
        if masks_present:
            masks_finite = bool(torch.isfinite(result.masks.data).all())
    module = model.model
    module.eval()
    device = next(module.parameters()).device
    with torch.no_grad():
        out = module(torch.zeros(1, 3, 640, 640, device=device))
    tensors = _collect_tensors(out)
    if not tensors or not all(bool(torch.isfinite(t).all()) for t in tensors):
        raise RuntimeError("segmentation forward output is missing or not finite")
    return {
        "detections": detections,
        "masks_present": masks_present,
        "masks_finite": masks_finite,
        "forward_shapes": [tuple(t.shape) for t in tensors],
    }


def _probe_fusion_checkpoint(path: Path, device: str = "cpu") -> dict:
    """Reload a fusion checkpoint through the production rebuild path."""
    import torch

    metadata = read_fusion_metadata(path)
    backbone = rebuild_gray_backbone(metadata, device)
    model, metadata, payload = load_fusion_checkpoint(path, backbone)
    model = model.to(device).eval()
    size = int(metadata.imgsz)
    gray = torch.zeros(2, 3, size, size, device=device)
    polar = torch.zeros(2, 3, size, size, device=device)
    quality = torch.tensor([[0.5, 0.5, 0.5, 0.1]] * 2, device=device)
    with torch.no_grad():
        out = model(gray, polar, quality)
    logits = out["final_logits"]
    state_finite = all(
        bool(torch.isfinite(value).all())
        for value in payload["state_dict"].values()
    )
    return {
        "class_names": list(metadata.class_names),
        "imgsz": int(metadata.imgsz),
        "gray_weights": metadata.gray_weights,
        "architecture": metadata.architecture,
        "state_finite": state_finite,
        "logits_shape": tuple(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "limit_batches": metadata.limit_batches,
        "smoke": metadata.smoke,
        "dataset_fingerprint": payload.get("dataset_fingerprint", ""),
    }


def _check_seg_base(base_path: Path) -> dict:
    if not base_path.is_file():
        raise FileNotFoundError(
            f"base checkpoint {base_path} is missing; no automatic download"
        )
    info = _probe_checkpoint(base_path)
    if info["task"] != "segment":
        raise ValueError(
            f"base checkpoint {base_path} task {info['task']!r} is not "
            "segmentation; refusing (yolo26n.pt is a detection model)"
        )
    if not str(info["architecture"]).startswith("yolo26"):
        raise ValueError(
            f"base checkpoint architecture {info['architecture']!r} is not "
            "YOLO26; refusing"
        )
    return info


def _check_cls_base(base_path: Path) -> dict:
    if not base_path.is_file():
        raise FileNotFoundError(
            f"base checkpoint {base_path} is missing; no automatic download"
        )
    info = _probe_checkpoint(base_path)
    if info["task"] != "classify":
        raise ValueError(
            f"base checkpoint {base_path} task {info['task']!r} is not "
            "classification; refusing"
        )
    if not str(info["architecture"]).startswith("yolo26"):
        raise ValueError(
            f"base checkpoint architecture {info['architecture']!r} is not "
            "YOLO26; refusing"
        )
    return info


def _check_seg_data(data_path: Path) -> dict:
    """Same single-class validation the Model A entry uses (no bypass)."""
    return _validate_binary_dataset(data_path)


def _check_smoke_marker(config: dict, require_formal: bool, owner: str) -> None:
    """Strict smoke/limit_batches admission: missing, null, string or
    boolean values are never silently treated as a formal full run."""
    smoke_flag = config.get("smoke")
    limit_batches = config.get("limit_batches")
    if require_formal:
        if smoke_flag is not False:
            raise RuntimeError(
                f"{owner} smoke field must be exactly false for formal use; "
                f"got {smoke_flag!r} (missing or non-boolean fields are not "
                "formal full training)"
            )
        if not (type(limit_batches) is int and limit_batches == 0):
            raise RuntimeError(
                f"{owner} limit_batches must be the integer 0 for formal use; "
                f"got {limit_batches!r}"
            )
    else:
        if smoke_flag is not True:
            raise RuntimeError(
                f"{owner} smoke field must be exactly true for a smoke "
                f"pipeline; got {smoke_flag!r}"
            )
        if not (type(limit_batches) is int and limit_batches > 0):
            raise RuntimeError(
                f"{owner} limit_batches must be a positive integer for a "
                f"smoke run; got {limit_batches!r}"
            )


def _check_fingerprint(recorded, expected: str, owner: str) -> str:
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError(
            f"{owner} dataset_fingerprint is missing/empty; it cannot be "
            "bound to the verified V4 data"
        )
    if recorded != expected:
        raise RuntimeError(
            f"{owner} dataset_fingerprint {recorded!r} != verified V4 "
            f"fingerprint {expected!r}"
        )
    return recorded


def _validate_gray_config(
    config: dict,
    run_id: str,
    imgsz: int,
    require_formal: bool,
    expected_fingerprint: str,
) -> None:
    if config.get("run_id") != run_id:
        raise RuntimeError(
            f"gray train_config run_id {config.get('run_id')!r} != {run_id!r}"
        )
    if not str(config.get("architecture", "")).startswith("yolo26"):
        raise RuntimeError(
            f"gray checkpoint architecture {config.get('architecture')!r} is "
            "not YOLO26"
        )
    if list(config.get("class_names", [])) != CLASS_NAMES:
        raise RuntimeError(
            f"gray class order {config.get('class_names')} != {CLASS_NAMES}"
        )
    if int(config.get("imgsz", -1)) != int(imgsz):
        raise RuntimeError(
            f"gray imgsz {config.get('imgsz')} != requested fusion imgsz {imgsz}"
        )
    _check_smoke_marker(config, require_formal, "gray")
    _check_fingerprint(
        config.get("dataset_fingerprint"), expected_fingerprint,
        "gray train_config",
    )
    if "macro-f1" not in str(config.get("selection", "")).lower():
        raise RuntimeError(
            "gray train_config does not record macro-F1-priority selection"
        )


def _check_gray_selection_evidence(run_dir: Path, config: dict) -> int:
    """best_epoch must be the argmax of (macro_f1, accuracy) in metrics.json."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError(f"gray artifact missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics:
        raise RuntimeError(f"gray metrics.json is empty: {metrics_path}")
    best_epoch = config.get("best_epoch")
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= len(metrics):
        raise RuntimeError(
            f"gray train_config best_epoch {best_epoch!r} is invalid"
        )
    keys = [(float(e["val_macro_f1"]), float(e["val_acc"])) for e in metrics]
    if keys[best_epoch - 1] != max(keys):
        raise RuntimeError(
            "gray model selection does not follow macro-F1 priority: "
            f"best_epoch={best_epoch} but the argmax of (macro_f1, accuracy) "
            f"is epoch {keys.index(max(keys)) + 1}"
        )
    return best_epoch


def _check_gray_artifacts(
    run_id: str, imgsz: int, require_formal: bool, expected_fingerprint: str
) -> dict:
    run_dir = stage_dir(run_id, "gray_fusion")
    for name in ("best.pt", "last.pt", "train_config.json", "metrics.json"):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"gray artifact missing: {run_dir / name}")
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    _validate_gray_config(config, run_id, imgsz, require_formal, expected_fingerprint)
    best_epoch = _check_gray_selection_evidence(run_dir, config)
    probe = _probe_checkpoint(run_dir / "best.pt")
    if probe["task"] != "classify":
        raise RuntimeError(f"gray checkpoint task {probe['task']!r} is not classify")
    if not str(probe["architecture"]).startswith("yolo26"):
        raise RuntimeError(
            f"gray checkpoint architecture {probe['architecture']!r} is not YOLO26"
        )
    if probe["names"] != CLASS_NAMES:
        raise RuntimeError(f"gray checkpoint names {probe['names']} != {CLASS_NAMES}")
    if probe.get("head_out_features") != len(CLASS_NAMES):
        raise RuntimeError(
            f"gray head outputs {probe.get('head_out_features')} classes"
        )
    if probe.get("logits_shape") != (2, len(CLASS_NAMES)) or not probe.get(
        "logits_finite"
    ):
        raise RuntimeError("gray checkpoint logits probe failed")
    return {
        "class_names": list(config["class_names"]),
        "best_epoch": best_epoch,
        "smoke": bool(config.get("smoke")),
        "imgsz": int(config["imgsz"]),
        "dataset_fingerprint": config["dataset_fingerprint"],
    }


def _check_gray_ready(run_id: str, imgsz: int, expected_fingerprint: str) -> dict:
    """Fusion-only preflight: this run's gray artifacts must exist, be
    compatible with the verified V4 data, and not be smoke output."""
    return _check_gray_artifacts(run_id, imgsz, True, expected_fingerprint)


def _check_model_a_artifacts(run_id: str, params: dict, require_formal: bool) -> dict:
    run_dir = stage_dir(run_id, "model_a")
    for name in ("weights/best.pt", "weights/last.pt", "train_config.json"):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"model_a artifact missing: {run_dir / name}")
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    if config.get("run_id") != run_id:
        raise RuntimeError(
            f"model_a train_config run_id {config.get('run_id')!r} != {run_id!r}"
        )
    if not str(config.get("architecture", "")).startswith("yolo26"):
        raise RuntimeError(
            f"model_a architecture {config.get('architecture')!r} is not YOLO26"
        )
    if config.get("task") != "segment":
        raise RuntimeError(f"model_a task {config.get('task')!r} is not segment")
    if list(config.get("data_names", [])) != [BINARY_CLASS_NAME]:
        raise RuntimeError(
            f"model_a data_names {config.get('data_names')!r} != "
            f"['{BINARY_CLASS_NAME}']"
        )
    if int(config.get("imgsz", -1)) != int(params["imgsz"]):
        raise RuntimeError(
            f"model_a imgsz {config.get('imgsz')} != requested {params['imgsz']}"
        )
    # Model A records only the smoke marker (its truncation is the data
    # fraction); require an explicit boolean matching the pipeline mode.
    smoke_flag = config.get("smoke")
    if require_formal:
        if smoke_flag is not False:
            raise RuntimeError(
                f"model_a smoke field must be exactly false for formal use; "
                f"got {smoke_flag!r}"
            )
    else:
        if smoke_flag is not True:
            raise RuntimeError(
                f"model_a smoke field must be exactly true for a smoke "
                f"pipeline; got {smoke_flag!r}"
            )
    probes = {}
    for name in ("best.pt", "last.pt"):
        probe = _probe_checkpoint(run_dir / "weights" / name)
        if probe["task"] != "segment":
            raise RuntimeError(
                f"model_a {name} task {probe['task']!r} is not segment"
            )
        if not str(probe["architecture"]).startswith("yolo26"):
            raise RuntimeError(
                f"model_a {name} architecture {probe['architecture']!r} is not YOLO26"
            )
        if probe["names"] != [BINARY_CLASS_NAME]:
            raise RuntimeError(
                f"model_a {name} names {probe['names']} != ['{BINARY_CLASS_NAME}']"
            )
        probes[name] = probe
    inference = None
    data_path = Path(config.get("data", ""))
    if data_path.is_file():
        data_info = _validate_binary_dataset(data_path)
        sample = data_info.get("sample_val_image")
        if sample:
            inference = _probe_seg_inference(
                run_dir / "weights" / "best.pt", Path(sample)
            )
    return {
        "imgsz": int(config["imgsz"]),
        "smoke": bool(config.get("smoke")),
        "data_names": config.get("data_names"),
        "train_images_used": config.get("train_images_used"),
        "inference": inference,
    }


def _check_fusion_artifacts(
    run_id: str, imgsz: int, require_formal: bool, expected_fingerprint: str
) -> dict:
    run_dir = stage_dir(run_id, "fusion_freeze")
    for name in ("best.pt", "last.pt", "train_config.json"):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"fusion artifact missing: {run_dir / name}")
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    if config.get("run_id") != run_id:
        raise RuntimeError(
            f"fusion train_config run_id {config.get('run_id')!r} != {run_id!r}"
        )
    if config.get("phase") != "freeze":
        raise RuntimeError(f"fusion phase {config.get('phase')!r} is not freeze")
    if int(config.get("imgsz", -1)) != int(imgsz):
        raise RuntimeError(
            f"fusion imgsz {config.get('imgsz')} != requested {imgsz}"
        )
    recorded = Path(str(config.get("gray_weights", "")))
    if not recorded.is_absolute():
        recorded = PROJECT_ROOT / recorded
    expected = gray_best_path(run_id)
    if recorded.resolve() != expected.resolve():
        raise RuntimeError(
            f"fusion gray source {recorded} does not point at this run's gray "
            f"checkpoint {expected}"
        )
    _check_smoke_marker(config, require_formal, "fusion train_config")
    config_fingerprint = _check_fingerprint(
        config.get("dataset_fingerprint"), expected_fingerprint,
        "fusion train_config",
    )
    probe = _probe_fusion_checkpoint(run_dir / "last.pt")
    if not probe["state_finite"] or not probe["logits_finite"]:
        raise RuntimeError("fusion checkpoint contains non-finite tensors")
    if probe["logits_shape"] != (2, len(CLASS_NAMES)):
        raise RuntimeError(f"fusion logits shape {probe['logits_shape']}")
    if probe["class_names"] != CLASS_NAMES:
        raise RuntimeError(f"fusion class names {probe['class_names']} != {CLASS_NAMES}")
    # Independent strict type/mode validation of the checkpoint metadata:
    # numeric equality with the train_config is not enough (3.0 == 3 and
    # False == 0 would otherwise let type-invalid fields pass).
    _check_smoke_marker(probe, require_formal, "fusion checkpoint metadata")
    # Config and checkpoint metadata must not contradict each other.
    if (
        probe.get("limit_batches") != config.get("limit_batches")
        or probe.get("smoke") is not config.get("smoke")
    ):
        raise RuntimeError(
            "fusion checkpoint metadata (limit_batches="
            f"{probe.get('limit_batches')!r}, smoke={probe.get('smoke')!r}) "
            "contradicts train_config (limit_batches="
            f"{config.get('limit_batches')!r}, smoke={config.get('smoke')!r})"
        )
    _check_fingerprint(
        probe.get("dataset_fingerprint"), expected_fingerprint,
        "fusion checkpoint",
    )
    return {
        "gray_weights": str(expected),
        "imgsz": probe["imgsz"],
        "logits_shape": probe["logits_shape"],
        "smoke": config.get("smoke"),
        "limit_batches": config.get("limit_batches"),
        "dataset_fingerprint": config_fingerprint,
    }


def _check_stage_artifacts(
    stage: str, run_id: str, params: dict, require_formal: bool,
    expected_fingerprint: str,
) -> dict:
    if stage == "model_a":
        return _check_model_a_artifacts(run_id, params, require_formal)
    if stage == "gray_fusion":
        return _check_gray_artifacts(
            run_id, params["imgsz"], require_formal, expected_fingerprint
        )
    return _check_fusion_artifacts(
        run_id, params["imgsz"], require_formal, expected_fingerprint
    )


def _build_command(
    stage: str, params: dict, run_id: str, device: str, seed: int
) -> list[str]:
    entry = params[stage]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / STAGE_SCRIPTS[stage]),
        "--run-id", run_id,
        "--device", device,
        "--seed", str(seed),
        "--epochs", str(entry["epochs"]),
        "--batch", str(entry["batch"]),
        "--imgsz", str(entry["imgsz"]),
    ]
    if stage == "model_a":
        command += ["--base", SEG_BASE, "--data", DEFAULT_SEG_DATA]
        if entry.get("fraction") is not None:
            command += ["--fraction", str(entry["fraction"])]
    elif stage == "gray_fusion":
        if entry.get("limit_batches"):
            command += ["--limit-batches", str(entry["limit_batches"])]
    else:
        command += [
            "--phase", "freeze",
            "--gray-weights", str(gray_best_path(run_id)),
        ]
        if entry.get("limit_batches"):
            command += ["--limit-batches", str(entry["limit_batches"])]
    return command


def _stage_sources(stage: str, run_id: str, plan: dict) -> dict:
    if stage == "model_a":
        return {"base": SEG_BASE, "data": DEFAULT_SEG_DATA}
    if stage == "gray_fusion":
        return {
            "data": DEFAULT_DATA_V4,
            "dataset_fingerprint": plan.get("v4_fingerprint", ""),
        }
    return {
        "data": DEFAULT_DATA_V4,
        "dataset_fingerprint": plan.get("v4_fingerprint", ""),
        "gray_weights": str(gray_best_path(run_id)),
    }


def _trim_probe(info: dict) -> dict:
    """Bound probe output size in plans/reports (keep a name sample)."""
    trimmed = dict(info)
    names = trimmed.get("names")
    if isinstance(names, list) and len(names) > 5:
        trimmed["names"] = names[:5]
        trimmed["names_count"] = len(names)
    return trimmed


def preflight(args: argparse.Namespace, params: dict, selected: Sequence[str]) -> dict:
    """All read-only checks; runs before any child process starts."""
    validate_run_id(args.run_id)
    resolve_device(args.device)

    conflicts = [
        stage_dir(args.run_id, stage)
        for stage in selected
        if stage_dir(args.run_id, stage).exists()
    ]
    if conflicts:
        paths = ", ".join(str(path) for path in conflicts)
        raise ValueError(
            "selected model output already exists; choose only unfinished "
            f"stages or use a new run id: {paths}"
        )

    if (
        "gray_fusion" in selected
        and "fusion_freeze" in selected
        and int(params["gray_fusion"]["imgsz"]) != int(params["fusion_freeze"]["imgsz"])
    ):
        raise ValueError(
            f"gray and fusion imgsz must match: "
            f"{params['gray_fusion']['imgsz']} != {params['fusion_freeze']['imgsz']}"
        )

    plan: dict = {
        "run_id": args.run_id,
        "device": args.device,
        "seed": args.seed,
        "smoke": args.smoke,
        "stages": list(selected),
        "params": params,
    }
    if "model_a" in selected:
        plan["seg_base"] = _trim_probe(_check_seg_base(PROJECT_ROOT / SEG_BASE))
        plan["seg_data"] = _check_seg_data(PROJECT_ROOT / DEFAULT_SEG_DATA)
    if "gray_fusion" in selected or "fusion_freeze" in selected:
        plan["cls_base"] = _trim_probe(_check_cls_base(PROJECT_ROOT / CLS_BASE))
        dataset_report = validate_dataset(PROJECT_ROOT / DEFAULT_DATA_V4)
        fingerprint = dataset_report.get("audit_fingerprint", "")
        if not fingerprint:
            raise ValueError(
                "V4 dataset fingerprint is empty; refusing (the strict "
                "dataset gate must return a verified fingerprint)"
            )
        plan["v4_fingerprint"] = fingerprint
        manifest_names = dataset_report.get("class_names")
        if manifest_names and list(manifest_names) != CLASS_NAMES:
            raise ValueError(
                f"V4 manifest class order {manifest_names} != {CLASS_NAMES}"
            )
    if "fusion_freeze" in selected and "gray_fusion" not in selected:
        plan["gray_ready"] = _check_gray_ready(
            args.run_id, int(params["fusion_freeze"]["imgsz"]), plan["v4_fingerprint"]
        )
    return plan


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_report(path: Path, report: dict) -> None:
    """Atomic safe write (temp file + os.replace): a plain write error can
    never truncate the existing report. A power loss or kill -9 can still
    lose the very last update; that limitation is accepted and documented."""
    report["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _load_report(path: Path, run_id: str) -> dict:
    """Load the cumulative report; migrate v1 losslessly.

    A corrupted report, or one belonging to a different run id, is refused
    rather than silently overwritten.
    """
    if not path.is_file():
        return {
            "report_version": REPORT_VERSION,
            "run_id": run_id,
            "created_at": _now(),
            "attempts": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"existing pipeline report is unreadable and will not be "
            f"overwritten: {path}: {exc}"
        )
    if not isinstance(data, dict):
        raise RuntimeError(
            f"existing pipeline report is malformed and will not be "
            f"overwritten: {path}"
        )
    if data.get("run_id") != run_id:
        raise RuntimeError(
            f"existing pipeline report belongs to run id "
            f"{data.get('run_id')!r}, not {run_id!r}; refusing to overwrite"
        )
    if data.get("report_version") == REPORT_VERSION:
        if not isinstance(data.get("attempts"), list):
            raise RuntimeError(
                f"existing pipeline report is malformed (attempts missing); "
                f"refusing to overwrite: {path}"
            )
        return data
    # v1 flat report -> lossless migration into one legacy attempt entry.
    legacy = {"attempt_id": "attempt_001_legacy", "migrated_from": "v1", **data}
    return {
        "report_version": REPORT_VERSION,
        "run_id": run_id,
        "created_at": data.get("created_at", _now()),
        "attempts": [legacy],
    }


def _refresh_summary(report: dict) -> None:
    status: dict[str, str] = {}
    for stage in STAGES:
        latest = None
        for attempt in report["attempts"]:
            entry = attempt.get("stages", {}).get(stage)
            if entry is not None:
                latest = entry.get("status")
        status[stage] = latest or "PENDING"
    report["stage_status"] = status
    report["remaining_stages"] = [
        stage for stage in STAGES if status[stage] != "PASSED"
    ]
    report["pipeline_complete"] = not report["remaining_stages"]


def run_pipeline(args: argparse.Namespace, runner: Callable[..., Any] = subprocess.run) -> int:
    try:
        selected = _validate_stages(args.stages)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    params = resolve_params(args)

    if args.preflight_only:
        try:
            plan = preflight(args, params, selected)
        except (InvalidRunIdError, DeviceUnavailableError, FileNotFoundError,
                ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"preflight": True, **plan}, ensure_ascii=False, indent=2))
        return 0

    try:
        plan = preflight(args, params, selected)
    except (InvalidRunIdError, DeviceUnavailableError, FileNotFoundError,
            ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report_path = PROJECT_ROOT / "runs" / "train" / args.run_id / "pipeline_report.json"
    try:
        report = _load_report(report_path, args.run_id)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    attempt: dict = {
        "attempt_id": f"attempt_{len(report['attempts']) + 1:03d}",
        "started_at": _now(),
        "finished_at": None,
        "device": args.device,
        "seed": args.seed,
        "smoke": args.smoke,
        "planned_stages": list(selected),
        "params": params,
        "preflight": {
            key: value for key, value in plan.items()
            if key not in ("params", "stages")
        },
        "stages": {},
        "status": "RUNNING",
    }
    report["attempts"].append(attempt)
    _refresh_summary(report)
    _write_report(report_path, report)

    expected_fingerprint = plan.get("v4_fingerprint", "")
    require_formal = not args.smoke

    def _finish_attempt(status: str) -> None:
        attempt["status"] = status
        attempt["finished_at"] = _now()
        _refresh_summary(report)
        _write_report(report_path, report)

    for stage in selected:
        command = _build_command(stage, params, args.run_id, args.device, args.seed)
        print(f"=== stage: {stage} ===")
        print(" ".join(command))
        entry: dict = {
            "command": command,
            "sources": _stage_sources(stage, args.run_id, plan),
            "smoke": args.smoke,
            "started_at": _now(),
            "status": "RUNNING",
        }
        attempt["stages"][stage] = entry
        # Persist this stage's command and start state before launching.
        _write_report(report_path, report)
        try:
            result = runner(command, cwd=PROJECT_ROOT, check=False)
        except KeyboardInterrupt:
            entry["finished_at"] = _now()
            entry["status"] = "INTERRUPTED"
            entry["failure"] = "interrupted by user"
            _finish_attempt("INTERRUPTED")
            print(
                f"ERROR: interrupted during stage {stage}; remaining stages "
                "were not started.",
                file=sys.stderr,
            )
            return 130
        except OSError as exc:
            entry["finished_at"] = _now()
            entry["status"] = "LAUNCH_FAILED"
            entry["failure"] = f"failed to launch: {exc}"
            _finish_attempt("FAILED")
            print(
                f"ERROR: failed to launch stage {stage}: {exc}; remaining "
                "stages were not started.",
                file=sys.stderr,
            )
            return 1
        entry["finished_at"] = _now()
        entry["exit_code"] = result.returncode
        if result.returncode != 0:
            entry["status"] = "FAILED"
            entry["failure"] = f"exit code {result.returncode}"
            _finish_attempt("FAILED")
            print(
                f"ERROR: stage {stage} failed with exit code {result.returncode}; "
                "remaining stages were not started.",
                file=sys.stderr,
            )
            return result.returncode if result.returncode > 0 else 1
        try:
            entry["artifacts"] = _check_stage_artifacts(
                stage, args.run_id, params[stage], require_formal,
                expected_fingerprint,
            )
        except KeyboardInterrupt:
            entry["status"] = "INTERRUPTED"
            entry["failure"] = "interrupted during artifact verification"
            _finish_attempt("INTERRUPTED")
            print(
                f"ERROR: interrupted while verifying stage {stage}; remaining "
                "stages were not started.",
                file=sys.stderr,
            )
            return 130
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            entry["status"] = "FAILED"
            entry["failure"] = f"artifact check failed: {exc}"
            _finish_attempt("FAILED")
            print(
                f"ERROR: stage {stage} exited 0 but its artifacts failed "
                f"verification: {exc}; remaining stages were not started.",
                file=sys.stderr,
            )
            return 1
        entry["status"] = "PASSED"
        _write_report(report_path, report)
        print(f"[{stage}] completed and verified")

    _finish_attempt("PASSED")
    if report["pipeline_complete"]:
        print(f"Full three-stage pipeline complete under run id: {args.run_id}")
    else:
        print(f"Selected stages completed under run id: {args.run_id}")
        print(
            "Pipeline not complete; remaining stages: "
            f"{', '.join(report['remaining_stages'])}"
        )
    print(f"Pipeline report: {report_path}")
    return 0


def main(argv: list[str] | None = None, runner: Callable[..., Any] = subprocess.run) -> int:
    args = parse_args(argv)
    return run_pipeline(args, runner=runner)


if __name__ == "__main__":
    sys.exit(main())
