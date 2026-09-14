"""Reproducible training CLI for the four model stages.

Stages:
    baseline : 4-class YOLO-seg on grayscale copies (reference model).
    a        : binary YOLO-seg, single class ``target`` (Model A).
    b-gray   : YOLO-cls on [gray, gray, gray] crops (gray-only Model B).
    b-polar  : YOLO-cls on [gray, polar, gray] crops (polar-aided Model B).

All stages run with seed 2026, deterministic mode, AMP, early stopping,
and explicit project/name so best.pt/last.pt land in a predictable run
directory. Every invocation groups its outputs under a validated run id:
``<resolved project>/<run-id>/model_<stage>``. Relative project paths are
anchored at the repository root so runs land in ``runs/train/<run-id>/<name>``
instead of Ultralytics' task-prefixed nested path. Dataloader workers
default to 0 on Windows (spawned
data-loader processes re-import torch and fail to load the CUDA DLLs)
and to the Ultralytics default of 8 on other platforms. Stages b-gray
and b-polar share every training argument except the data root and run
name. Color auto-augmentation, HSV jitter, and random erasing are disabled
for both because the input channels are physical features rather than RGB.

Usage:
    python scripts/train_models.py --stage baseline --run-id run_20260914_manual_labels
    python scripts/train_models.py --stage a --run-id run_20260914_manual_labels
    python scripts/train_models.py --stage b-gray --run-id smoke_20260914 --device cpu
    python scripts/train_models.py --stage b-polar --run-id run_20260914_manual_labels
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SEED = 2026
DEFAULT_PATIENCE = 20
DEFAULT_PROJECT = "runs/train"
# Ultralytics defaults to 8 dataloader workers; on Windows, spawned
# data-loader processes re-import torch and fail to load the CUDA DLLs,
# so default to single-process data loading there.
DEFAULT_WORKERS = 0 if sys.platform == "win32" else 8

# Development training bases: YOLO26nano as of 2026-09. The local
# yolo26n.pt is a *detection* checkpoint and must not be used for Model A
# (segmentation) or Model B (classification). yolo26n-seg.pt /
# yolo26n-cls.pt are Ultralytics COCO-pretrained bases; when absent locally
# they must be reported, not silently substituted.
SEG_BASE = "yolo26n-seg.pt"
CLS_BASE = "yolo26n-cls.pt"
# Historical YOLOv8 bases kept for reference/regression comparisons only.
LEGACY_SEG_BASE = "yolov8n-seg.pt"
LEGACY_CLS_BASE = "yolov8n-cls.pt"

STAGE_DEFAULTS: dict[str, dict] = {
    "baseline": {
        "base": SEG_BASE,
        "data": "datasets/underwater_seg_v2/data.yaml",
        "imgsz": 640,
        "batch": 8,
        "epochs": 100,
    },
    "a": {
        "base": SEG_BASE,
        "data": "datasets/underwater_seg_binary_v2/data.yaml",
        "imgsz": 640,
        "batch": 8,
        "epochs": 100,
    },
    "b-gray": {
        "base": CLS_BASE,
        "data": "datasets/underwater_cls_gray_v2",
        "imgsz": 224,
        "batch": 64,
        "epochs": 80,
    },
    "b-polar": {
        "base": CLS_BASE,
        "data": "datasets/underwater_cls_polar_v2",
        "imgsz": 224,
        "batch": 64,
        "epochs": 80,
    },
}

CLASSES_STAGES = ("b-gray", "b-polar")


class DeviceUnavailableError(RuntimeError):
    """Raised when a CUDA device is requested but CUDA is not usable."""


class InvalidRunIdError(ValueError):
    """Raised when a run id is not a safe single path component."""


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+\Z")


def validate_run_id(run_id: str) -> str:
    """Validate a run id as a single safe path component.

    Accepts ASCII letters, digits, ``.``, ``_`` and ``-`` only; rejects
    empty strings, ``.``/``..``, path separators, absolute paths, spaces,
    other punctuation and non-ASCII characters.
    """
    if (
        not isinstance(run_id, str)
        or run_id in (".", "..")
        or _RUN_ID_PATTERN.match(run_id) is None
    ):
        raise InvalidRunIdError(
            f"Invalid run id {run_id!r}: must be a single path component "
            "of ASCII letters, digits, '.', '_' or '-'."
        )
    return run_id


def resolve_device(device: str, cuda_available: bool | None = None) -> str:
    """Fail fast when a CUDA device is requested but CUDA is unavailable."""
    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available()
    if str(device).strip().lower() != "cpu" and not cuda_available:
        raise DeviceUnavailableError(
            f"Device '{device}' was requested but torch.cuda.is_available() is "
            "False. Re-run with --device cpu for a (slow, non-production) "
            "smoke test, or on a machine with CUDA."
        )
    return device


def resolve_project(project: str) -> str:
    """Anchor a relative project path at the repository root.

    Ultralytics prefixes relative project paths with its task name (for
    example ``segmentation/runs/train/...``); resolving against the repo
    root keeps runs under ``<repo>/runs/train``. Absolute paths are
    returned unchanged.
    """
    path = Path(project)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _load_yolo_model(weight_path: str):
    """Load an Ultralytics model; kept separate so tests cannot start training."""
    from ultralytics import YOLO

    return YOLO(weight_path)


def build_train_kwargs(
    stage: str,
    *,
    seed: int = DEFAULT_SEED,
    patience: int = DEFAULT_PATIENCE,
    project: str = DEFAULT_PROJECT,
    name: str | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Build the Ultralytics ``model.train(**kwargs)`` argument dict.

    b-gray and b-polar must differ only in ``data`` and ``name``: identical
    augmentations (HSV disabled), identical seed and initialization path.
    """
    if stage not in STAGE_DEFAULTS:
        raise ValueError(f"Unknown stage '{stage}'. Choose from {sorted(STAGE_DEFAULTS)}")
    defaults = STAGE_DEFAULTS[stage]

    resolved = {
        "epochs": defaults["epochs"] if epochs is None else epochs,
        "imgsz": defaults["imgsz"] if imgsz is None else imgsz,
        "batch": defaults["batch"] if batch is None else batch,
    }

    kwargs = {
        "data": defaults["data"],
        "seed": seed,
        "deterministic": True,
        "amp": True,
        "patience": patience,
        "project": project,
        "name": name or f"model_{stage}",
        "save": True,
        "exist_ok": False,
        "epochs": resolved["epochs"],
        "imgsz": resolved["imgsz"],
        "batch": resolved["batch"],
        "workers": workers,
        "val": True,
    }
    if stage in CLASSES_STAGES:
        # Channel semantics are [gray, polar, gray] / [gray, gray, gray],
        # not RGB, so color policies would corrupt the physical channels.
        kwargs["auto_augment"] = None
        kwargs["erasing"] = 0.0
        kwargs["hsv_h"] = 0.0
        kwargs["hsv_s"] = 0.0
        kwargs["hsv_v"] = 0.0

    return kwargs


def train_stage(
    stage: str,
    *,
    device: str = "0",
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    data: str | None = None,
    name: str | None = None,
    project: str = DEFAULT_PROJECT,
    seed: int = DEFAULT_SEED,
    patience: int = DEFAULT_PATIENCE,
    workers: int = DEFAULT_WORKERS,
    run_id: str | None = None,
    model_factory: Callable[[str], Any] | None = None,
) -> Path:
    """Run one training stage and return the run directory.

    When ``run_id`` is given, outputs are grouped under
    ``<resolved project>/<run-id>`` (name stays ``model_<stage>``); without
    it, the resolved project is used as-is for backward compatibility.
    """
    if run_id is not None:
        validate_run_id(run_id)

    resolve_device(device)

    defaults = STAGE_DEFAULTS[stage]
    resolved_project = resolve_project(project)
    if run_id is not None:
        resolved_project = str(Path(resolved_project) / run_id)
    kwargs = build_train_kwargs(
        stage,
        seed=seed,
        patience=patience,
        project=resolved_project,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
    )
    if data is not None:
        kwargs["data"] = data
    kwargs["device"] = device

    weight_path = defaults["base"]
    print(f"[{stage}] loading base weights: {weight_path}")
    model = (model_factory or _load_yolo_model)(weight_path)
    print(f"[{stage}] training with: {kwargs}")
    model.train(**kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", resolved_project))
    best = save_dir / "weights" / "best.pt"
    print(f"[{stage}] done. best={best if best.exists() else '<missing>'}, "
          f"last={save_dir / 'weights' / 'last.pt'}")
    return save_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", required=True, choices=sorted(STAGE_DEFAULTS),
        help="Which model stage to train.",
    )
    parser.add_argument(
        "--run-id", required=True,
        help="Run id grouping this training's outputs under "
             "<project>/<run-id>/ (single path component: letters, digits, "
             "'.', '_', '-').",
    )
    parser.add_argument("--device", default="0", help="'0'/'cuda' or 'cpu' (smoke test only).")
    parser.add_argument("--epochs", type=int, default=None, help="Override stage default.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override stage default.")
    parser.add_argument("--batch", type=int, default=None, help="Override stage default.")
    parser.add_argument("--data", default=None, help="Override default dataset path.")
    parser.add_argument("--name", default=None, help="Override run name.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Ultralytics project dir.")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Dataloader workers (default: 0 on Windows to avoid CUDA DLL "
        "load failures in spawned workers, 8 elsewhere).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_run_id(args.run_id)
        train_stage(
            args.stage,
            device=args.device,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            data=args.data,
            name=args.name,
            project=args.project,
            seed=args.seed,
            patience=args.patience,
            workers=args.workers,
            run_id=args.run_id,
        )
    except InvalidRunIdError as exc:
        print(f"ERROR: invalid run id: {exc}", file=sys.stderr)
        return 2
    except DeviceUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
