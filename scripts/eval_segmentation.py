"""Evaluate YOLO segmentation checkpoints on a shared test split.

Compares the baseline 4-class model and the binary Model A on their test
sets and writes a single stable JSON report. Ultralytics is imported lazily
inside :func:`_load_yolo` so importing this module (e.g. for tests or CLI
help) never loads torch/ultralytics.

用法:
    python scripts/eval_segmentation.py --output analysis/runs/<run-id>/segmentation_test.json
    python scripts/eval_segmentation.py --split test --device 0 --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CHECKPOINT = Path("runs/train/run_20260913_initial/model_baseline/weights/best.pt")
DEFAULT_BASELINE_DATA = Path("datasets/underwater_seg_v2/data.yaml")
DEFAULT_MODEL_A_CHECKPOINT = Path("runs/train/run_20260913_initial/model_a/weights/best.pt")
DEFAULT_MODEL_A_DATA = Path("datasets/underwater_seg_binary_v2/data.yaml")
DEFAULT_EVAL_PROJECT = Path("runs/eval")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# results_dict key -> report key, per metric component.
_OVERALL_KEYS: dict[str, dict[str, str]] = {
    "box": {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    },
    "mask": {
        "precision": "metrics/precision(M)",
        "recall": "metrics/recall(M)",
        "map50": "metrics/mAP50(M)",
        "map50_95": "metrics/mAP50-95(M)",
    },
}


@dataclass(frozen=True)
class EvalTask:
    """One checkpoint + dataset pair to evaluate."""

    name: str
    checkpoint: Path
    data: Path


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_overall(results_dict: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Extract overall box/mask metrics from an Ultralytics results_dict."""
    overall: dict[str, dict[str, float]] = {}
    for component, keys in _OVERALL_KEYS.items():
        values = {
            out: float(results_dict[src])
            for out, src in keys.items()
            if src in results_dict
        }
        if values:
            overall[component] = values
    return overall


def _per_class_ap(component: Any, class_names: Sequence[str]) -> dict[str, dict[str, float]]:
    """Map a Metric component (all_ap + ap_class_index) to per-class stats."""
    all_ap = getattr(component, "all_ap", None)
    index = getattr(component, "ap_class_index", None)
    if all_ap is None or index is None:
        return {}
    all_ap = np.asarray(all_ap)
    stats: dict[str, dict[str, float]] = {}
    for row, raw_cls in enumerate(index):
        cls = int(raw_cls)
        if 0 <= cls < len(class_names) and row < len(all_ap):
            stats[class_names[cls]] = {
                "map50": float(all_ap[row][0]),
                "map50_95": float(np.mean(all_ap[row])),
            }
    return stats


def extract_per_class(metrics: Any, class_names: Sequence[str]) -> dict[str, dict[str, dict[str, float]]]:
    """Extract per-class box/mask mAP from SegmentMetrics-like object.

    Uses ``metrics.seg`` (current Ultralytics) with fallback to the legacy
    ``metrics.mask`` attribute; components without AP data are skipped.
    """
    per_class: dict[str, dict[str, dict[str, float]]] = {}
    box = getattr(metrics, "box", None)
    seg = getattr(metrics, "seg", None)
    if seg is None:
        seg = getattr(metrics, "mask", None)
    for name, component in (("box", box), ("mask", seg)):
        if component is None:
            continue
        for class_name, values in _per_class_ap(component, class_names).items():
            per_class.setdefault(class_name, {})[name] = values
    return per_class


def extract_counts(metrics: Any, class_names: Sequence[str]) -> dict[str, dict[str, int]]:
    """Extract ground-truth instance counts per class when exposed."""
    nt_per_class = getattr(metrics, "nt_per_class", None)
    if nt_per_class is None:
        return {}
    counts = {
        name: int(nt_per_class[i])
        for i, name in enumerate(class_names)
        if i < len(nt_per_class)
    }
    return {"instances_per_class": counts}


# ---------------------------------------------------------------------------
# data.yaml helpers
# ---------------------------------------------------------------------------

def _load_data_yaml(data_yaml: Path) -> dict:
    with open(data_yaml, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_class_names(data_yaml: Path) -> list[str]:
    """Read class names from a YOLO data.yaml (mapping or list form)."""
    names = _load_data_yaml(data_yaml).get("names")
    if isinstance(names, Mapping):
        return [str(names[key]) for key in sorted(names, key=int)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"{data_yaml}: 'names' must be a mapping or a list")


def count_split_images(data_yaml: Path, split: str) -> int | None:
    """Count image files of one split, or None when the directory is absent."""
    cfg = _load_data_yaml(data_yaml)
    root = Path(cfg.get("path") or ".")
    if not root.is_absolute():
        root = data_yaml.parent / root
    split_value = cfg.get(split)
    if split_value is None:
        return None
    split_dir = root / str(split_value)
    if not split_dir.is_dir():
        return None
    return sum(
        1
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def validate_task(task: EvalTask) -> None:
    """Raise FileNotFoundError naming the missing input, if any."""
    if not task.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {task.checkpoint}")
    if not task.data.is_file():
        raise FileNotFoundError(f"Dataset data.yaml not found: {task.data}")


def _load_yolo(checkpoint: Path) -> Any:
    """Load a Ultralytics YOLO model (lazy import, call-time resolution)."""
    from ultralytics import YOLO

    return YOLO(str(checkpoint))


def evaluate_task(
    task: EvalTask,
    split: str = "test",
    device: str = "cpu",
    yolo_factory: Callable[[Path], Any] | None = None,
    *,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 0,
    project: Path = DEFAULT_EVAL_PROJECT,
) -> dict[str, Any]:
    """Run YOLO ``val`` for one task and build its report section."""
    validate_task(task)
    factory = yolo_factory if yolo_factory is not None else _load_yolo
    model = factory(task.checkpoint)
    resolved_project = project if project.is_absolute() else PROJECT_ROOT / project
    metrics = model.val(
        data=str(task.data),
        split=split,
        device=device,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        plots=False,
        project=str(resolved_project),
        name=f"{task.name}-{split}",
        exist_ok=True,
    )

    class_names = load_class_names(task.data)
    return {
        "checkpoint": task.checkpoint.as_posix(),
        "data": task.data.as_posix(),
        "split": split,
        "device": device,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "class_names": class_names,
        "n_split_images": count_split_images(task.data, split),
        "overall": extract_overall(getattr(metrics, "results_dict", {})),
        "per_class": extract_per_class(metrics, class_names),
        "counts": extract_counts(metrics, class_names),
    }


# ---------------------------------------------------------------------------
# Report building and writing
# ---------------------------------------------------------------------------

def build_report(
    tasks: Sequence[EvalTask],
    sections: Sequence[dict[str, Any]],
    split: str,
    device: str,
) -> dict[str, Any]:
    """Assemble the top-level report keyed by task name."""
    return {
        "split": split,
        "device": device,
        "models": {task.name: section for task, section in zip(tasks, sections)},
    }


def write_report(report: dict[str, Any], output: Path) -> Path:
    """Write the report as stable JSON (sorted keys, trailing newline)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation checkpoints and write a JSON report."
    )
    parser.add_argument(
        "--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT,
        help="Baseline 4-class checkpoint.",
    )
    parser.add_argument(
        "--baseline-data", type=Path, default=DEFAULT_BASELINE_DATA,
        help="Baseline dataset data.yaml.",
    )
    parser.add_argument(
        "--model-a-checkpoint", type=Path, default=DEFAULT_MODEL_A_CHECKPOINT,
        help="Binary Model A checkpoint.",
    )
    parser.add_argument(
        "--model-a-data", type=Path, default=DEFAULT_MODEL_A_DATA,
        help="Binary Model A dataset data.yaml.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--device", default="cpu", help="Ultralytics device argument.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Validation data-loader workers (0 is reliable on Windows).",
    )
    parser.add_argument(
        "--project", type=Path, default=DEFAULT_EVAL_PROJECT,
        help="Ultralytics validation artifact directory.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSON report path."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tasks = [
        EvalTask("model_baseline", args.baseline_checkpoint, args.baseline_data),
        EvalTask("model_a", args.model_a_checkpoint, args.model_a_data),
    ]

    try:
        for task in tasks:
            validate_task(task)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sections = [
        evaluate_task(
            task,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=args.project,
        )
        for task in tasks
    ]
    report = build_report(tasks, sections, split=args.split, device=args.device)
    output = write_report(report, args.output)
    print(f"Wrote report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
