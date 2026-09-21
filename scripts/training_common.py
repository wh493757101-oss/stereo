"""Shared training plumbing for every training entry in this repository.

Extracted from the former four-stage ``scripts/train_models.py`` so the
active three-stage pipeline (``scripts/train_pipeline.py``), the
per-stage entries and the archived legacy code all share one definition
of run-id validation, device resolution, project anchoring and the
training defaults.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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

# Development training bases: YOLO26nano. The local yolo26n.pt is a
# *detection* checkpoint and must not be used for Model A (segmentation)
# or Model B (classification). yolo26n-seg.pt / yolo26n-cls.pt are
# Ultralytics COCO-pretrained bases; when absent locally they must be
# reported, not silently substituted.
SEG_BASE = "yolo26n-seg.pt"
CLS_BASE = "yolo26n-cls.pt"
# Historical YOLOv8 bases kept for reference/regression comparisons only.
LEGACY_SEG_BASE = "yolov8n-seg.pt"
LEGACY_CLS_BASE = "yolov8n-cls.pt"


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
