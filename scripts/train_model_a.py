"""Train Model A: YOLO26 binary instance segmentation.

Stage 1 of the three-stage mainline (``scripts/train_pipeline.py``). This is
the extracted stage-``a`` logic of the retired four-stage trainer: binary
YOLO-seg with a single class ``target`` on
``datasets/underwater_seg_binary_v2``, initialized from the YOLO26nano
segmentation base (``yolo26n-seg.pt``; the local ``yolo26n.pt`` detection
checkpoint is refused). Training keeps the historical parameters: seed
2026, deterministic mode, AMP, early stopping, Windows workers=0.

``--fraction`` limits the training data ratio (Ultralytics ``fraction``
argument) for bounded smoke runs; it is recorded in ``train_config.json``
together with the ``smoke`` marker and the actual image counts.

Outputs land in ``runs/train/<run-id>/model_a/`` (Ultralytics layout:
``weights/best.pt``, ``weights/last.pt``, ``args.yaml``, ``results.csv``)
plus this entry's ``train_config.json``. An existing target directory is
refused.

Usage:
    python scripts/train_model_a.py --run-id run_x --device 0
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import file_digest
from models.polar_fusion import architecture_name
from scripts.training_common import (
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WORKERS,
    SEG_BASE,
    DeviceUnavailableError,
    InvalidRunIdError,
    resolve_device,
    validate_run_id,
)

DEFAULT_BASE = SEG_BASE
DEFAULT_DATA = "datasets/underwater_seg_binary_v2/data.yaml"
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_EPOCHS = 100
MODEL_A_VERSION = "model_a_v1"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id grouping outputs under runs/train/<run-id>/model_a.",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"YOLO26 segmentation base checkpoint (default: {DEFAULT_BASE}); "
        "must exist locally (no automatic download).",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Binary segmentation data.yaml (default: %(default)s).",
    )
    parser.add_argument("--device", default="0", help="Training device.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Dataloader workers (default: 0 on Windows, 8 elsewhere).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Training data fraction in (0, 1]; smoke/verification only "
        "(recorded together with the smoke marker).",
    )
    args = parser.parse_args(argv)
    if args.imgsz < 1:
        parser.error("--imgsz must be a positive integer")
    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    if args.fraction is not None and not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be in (0, 1]")
    return args


def _load_yolo_model(weight_path: str):
    """Load an Ultralytics model; kept separate so tests cannot train."""
    from ultralytics import YOLO

    return YOLO(weight_path)


BINARY_CLASS_NAME = "target"


def _normalized_names(config: dict, data_yaml: Path) -> list[str]:
    """Class names in id order from either YAML representation."""
    names = config.get("names")
    if isinstance(names, dict):
        entries: dict[int, str] = {}
        for key, value in names.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                raise ValueError(f"invalid class id {key!r} in {data_yaml}")
            entries[index] = str(value)
        if sorted(entries) != [0]:
            raise ValueError(
                f"binary segmentation must define exactly class id 0 "
                f"('{BINARY_CLASS_NAME}') in {data_yaml}; got ids {sorted(entries)}"
            )
        return [entries[0]]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise ValueError(f"missing class names in {data_yaml}")


def _split_entries(data_yaml: Path, config: dict, split: str) -> tuple[Path, list[Path]]:
    """Image list for one split, resolved exactly as training would use it."""
    entry = config.get(split)
    if entry is None:
        raise ValueError(f"{data_yaml} is missing the {split!r} split")
    root = Path(str(config.get("path", data_yaml.parent)))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    target = root / str(entry)
    if not target.exists():
        raise ValueError(f"{split} split path does not exist: {target}")
    if target.is_file():  # Ultralytics .txt image-list form
        images = []
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            path = Path(line)
            images.append(path if path.is_absolute() else (target.parent / path))
        return target, sorted(images)
    images = sorted(
        path for path in target.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    return target, images


def _label_path_for(image_path: Path) -> Path:
    """Ultralytics convention: sibling ``images`` -> ``labels``, .txt."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path has no 'images' component: {image_path}")


def _validate_seg_label(label_path: Path) -> int:
    """Validate one segmentation label file (read-only); returns polygon count."""
    polygons = 0
    for line_number, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue  # empty file / blank lines = background convention
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2 != 0:
            raise ValueError(
                f"{label_path}:{line_number}: not a segmentation polygon "
                f"({len(fields)} fields; detection-box labels are not accepted)"
            )
        class_token = fields[0]
        if not re.fullmatch(r"[0-9]+", class_token):
            raise ValueError(
                f"{label_path}:{line_number}: class id {class_token!r} is not "
                "an integer"
            )
        if int(class_token) != 0:
            raise ValueError(
                f"{label_path}:{line_number}: class id {class_token} != 0 "
                f"(single class {BINARY_CLASS_NAME!r} only)"
            )
        try:
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            raise ValueError(
                f"{label_path}:{line_number}: non-numeric polygon coordinate"
            )
        if not all(math.isfinite(value) for value in coords):
            raise ValueError(
                f"{label_path}:{line_number}: polygon coordinates must be finite"
            )
        polygons += 1
    return polygons


def _validate_binary_dataset(data_path: Path) -> dict:
    """Read-only single-class binary segmentation validation.

    Shared by the standalone Model A entry and the pipeline preflight so the
    two paths can never diverge. Checks the data yaml (exactly one class
    ``0: target``, matching ``nc``), resolves the train/val splits as
    training uses them, and validates every referenced segmentation label
    (integer class 0, polygon structure, finite coordinates). Missing or
    empty label files are legal background images, but each split must
    contain at least one valid polygon. No caches are written and no labels
    are fixed or rewritten.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"missing segmentation data yaml: {data_path}")
    import yaml

    with data_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"invalid data yaml: {data_path}")
    names = _normalized_names(config, data_path)
    if names != [BINARY_CLASS_NAME]:
        raise ValueError(
            f"binary segmentation data.yaml must define exactly one class "
            f"'0: {BINARY_CLASS_NAME}'; got names={names} in {data_path}"
        )
    nc = config.get("nc")
    if nc is not None and (isinstance(nc, bool) or not isinstance(nc, int) or nc != 1):
        raise ValueError(
            f"nc must be 1 for the single-class dataset; got {nc!r} in {data_path}"
        )
    result: dict = {"names": names, "nc": 1, "backgrounds": {}}
    sample_val_image: str | None = None
    for split in ("train", "val"):
        _, images = _split_entries(data_path, config, split)
        if not images:
            raise ValueError(f"no {split} images referenced by {data_path}")
        polygons = 0
        labelled = 0
        backgrounds = 0
        for image in images:
            label = _label_path_for(image)
            if not label.is_file():
                backgrounds += 1
                continue
            count = _validate_seg_label(label)
            if count == 0:
                backgrounds += 1
            else:
                labelled += 1
                polygons += count
            if split == "val" and sample_val_image is None:
                sample_val_image = str(image)
        if polygons < 1:
            raise ValueError(
                f"{split} split has no valid target annotation; a dataset "
                f"without valid polygons must not pass ({data_path})"
            )
        result[f"{split}_images"] = len(images)
        result[f"{split}_labelled_images"] = labelled
        result[f"{split}_polygons"] = polygons
        result["backgrounds"][split] = backgrounds
    result["sample_val_image"] = sample_val_image
    return result


def run_training(args: argparse.Namespace, model_factory: Callable[[str], Any] | None = None) -> Path:
    """Run the binary segmentation training; returns the Ultralytics save dir."""
    validate_run_id(args.run_id)
    resolve_device(args.device)

    base_path = Path(args.base)
    base_path = base_path if base_path.is_absolute() else PROJECT_ROOT / base_path
    if not base_path.is_file():
        raise FileNotFoundError(
            f"base checkpoint {base_path} is missing; no automatic download"
        )
    data_path = Path(args.data)
    data_path = data_path if data_path.is_absolute() else PROJECT_ROOT / data_path
    # Shared single-class validation (read-only; also used by the pipeline
    # preflight so the entry can never be bypassed).
    data_info = _validate_binary_dataset(data_path)

    run_dir = PROJECT_ROOT / "runs" / "train" / args.run_id / "model_a"
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use a new --run-id "
            "instead of overwriting a previous run"
        )

    factory = model_factory or _load_yolo_model
    model = factory(str(base_path))
    module = model.model
    task = getattr(module, "task", None)
    if task != "segment":
        raise ValueError(
            f"base checkpoint {base_path} task {task!r} is not segmentation; "
            "refusing (yolo26n.pt is a detection model, not a Model A base)"
        )
    architecture = architecture_name(module)
    if not architecture.startswith("yolo26"):
        raise ValueError(
            f"base checkpoint architecture {architecture!r} is not YOLO26; "
            "refusing (legacy YOLOv8 weights are comparison-only)"
        )

    train_images = data_info["train_images"]
    val_images = data_info["val_images"]
    smoke = args.fraction is not None and args.fraction < 1.0
    kwargs = {
        "data": str(data_path),
        "seed": args.seed,
        "deterministic": True,
        "amp": True,
        "patience": args.patience,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "save": True,
        "exist_ok": False,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "val": True,
        "device": args.device,
    }
    if args.fraction is not None:
        kwargs["fraction"] = args.fraction
    print(f"[model_a] base={base_path} data={data_path}")
    print(f"[model_a] training with: {kwargs}")
    model.train(**kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", run_dir))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise RuntimeError(
            f"training finished but best/last weights are missing under {save_dir}"
        )

    train_images_used = train_images
    if args.fraction is not None and train_images is not None:
        train_images_used = max(round(train_images * args.fraction), 1)
    config = {
        "entry": "train_model_a",
        "version": MODEL_A_VERSION,
        "run_id": args.run_id,
        "base": str(base_path),
        "base_sha256": file_digest(base_path),
        "architecture": architecture,
        "task": task,
        "data": str(data_path),
        "data_names": data_info["names"],
        "train_images": train_images,
        "val_images": val_images,
        "train_images_used": train_images_used,
        "train_polygons": data_info["train_polygons"],
        "val_polygons": data_info["val_polygons"],
        "backgrounds": data_info["backgrounds"],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs": args.epochs,
        "seed": args.seed,
        "patience": args.patience,
        "workers": args.workers,
        "device": args.device,
        "fraction": args.fraction,
        "smoke": smoke,
        "selection": "Ultralytics internal best (fitness)",
    }
    (save_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[model_a] done. best={best}, last={last}")
    return save_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = run_training(args)
    except InvalidRunIdError as exc:
        print(f"invalid run id: {exc}", file=sys.stderr)
        return 2
    except DeviceUnavailableError as exc:
        print(f"device unavailable: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"model A training refused: {exc}", file=sys.stderr)
        return 2
    print(f"Model A run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
