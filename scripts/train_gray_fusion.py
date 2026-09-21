"""Train the YOLO26 gray-only classifier on the fusion npz dataset (V4).

Fair-comparison counterpart of the Polar Fusion gray branch: same samples,
splits, class order, direct-resize preprocessing and imgsz as
``FusionClsDataset``, but the whole YOLO26 classification model trains on
the gray channel only (polar/quality inputs are not used to predict).
Outputs are standard Ultralytics classification checkpoints (4-class head
in dataset class order) that the existing
``prepare_gray_backbone(..., gray_weights=...)`` can load as the frozen
gray branch of polar fusion training.

Model selection uses the full val split with val macro-F1 primary and
accuracy as tiebreak (identical to fusion). The test split is never used
for training or selection.

Formal runs use ``--limit-batches 0`` (all batches). A positive
``--limit-batches`` is a smoke/verification truncation: it is recorded as
``smoke: true`` in ``train_config.json`` and its metrics must not be used
as model acceptance evidence.

Usage:
    python scripts/train_gray_fusion.py --run-id run_x --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import file_digest
from models.polar_fusion import (
    GrayBackboneAdapter,
    FusionClsDataset,
    architecture_name,
    class_names_from_manifest,
    prepare_gray_backbone,
    read_manifest_split,
)
from scripts.training_common import (
    DeviceUnavailableError,
    InvalidRunIdError,
    resolve_device,
    validate_run_id,
)
from scripts.train_polar_fusion import (
    DEFAULT_IMGSZ,
    DEFAULT_SEED,
    _set_seed,
    _verify_backbone_output,
    classification_metrics,
    torch_device_name,
    validate_dataset,
)

DEFAULT_BASE = "yolo26n-cls.pt"
DEFAULT_DATA = "datasets/underwater_cls_fusion_v4_band"
GRAY_VERSION = "gray_fusion_v1"
PREPROCESSING = (
    "FusionClsDataset gray channel: left crop resized to (imgsz, imgsz) with "
    "cv2.INTER_AREA, /255, replicated to 3 channels; no augmentation"
)


class GrayFusionDataset(torch.utils.data.Dataset):
    """Gray-only view of FusionClsDataset: yields ``(gray, label)``.

    Returns exactly the tensors ``FusionClsDataset`` produces for the same
    sample index, so gray-only training sees the same input distribution
    as the fusion gray branch.
    """

    def __init__(self, sample_paths, imgsz: int = DEFAULT_IMGSZ):
        self._fusion = FusionClsDataset(sample_paths, imgsz=imgsz)

    def __len__(self) -> int:
        return len(self._fusion)

    def __getitem__(self, index: int):
        gray, _polar, _quality, label = self._fusion[index]
        return gray, label


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id grouping outputs under runs/train/<run-id>/gray_fusion.",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"YOLO26 classification base checkpoint (default: {DEFAULT_BASE}); "
        "must exist locally (no automatic download).",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Fusion npz dataset root (default: %(default)s).",
    )
    parser.add_argument("--device", default="0", help="Training device.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="AdamW learning rate."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Smoke/verification truncation: cap train/val batches per epoch "
        "(0 = all batches; positive values mark the run as smoke).",
    )
    args = parser.parse_args(argv)
    if args.imgsz < 1:
        parser.error("--imgsz must be a positive integer")
    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.limit_batches < 0:
        parser.error("--limit-batches must be non-negative")
    return args


def _selection_key(metrics: dict) -> tuple[float, float]:
    """Model-selection key: val macro-F1 primary, accuracy tiebreak."""
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]))


def _unfreeze_gray_model(model: nn.Module) -> int:
    """Enable gradients on every parameter of the gray model.

    Ultralytics classification checkpoints load with all parameters frozen
    (``requires_grad=False``); gray-only training is full-model training, so
    the entry explicitly unfreezes everything. Weights are never
    re-initialized or reset. Returns the trainable parameter tensor count.
    """
    for param in model.parameters():
        param.requires_grad_(True)
    return sum(1 for p in model.parameters() if p.requires_grad)


def _save_gray_checkpoint(module: nn.Module, path: Path) -> None:
    """Save an Ultralytics-loadable classification checkpoint.

    The payload matches the Ultralytics checkpoint format (``{"model": ...}``)
    so ``YOLO(path)`` / ``prepare_gray_backbone(..., gray_weights=...)`` load
    it unchanged. The live module is pickled as-is: device, dtype and
    train/eval mode are not modified.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": module}, path)


@torch.no_grad()
def _evaluate_gray(
    model: GrayBackboneAdapter, loader, device: str, num_classes: int, limit_batches: int = 0
) -> tuple[dict, int, int]:
    """Val metrics over a loader; returns (metrics, batches_used, samples)."""
    model.eval()
    labels_all: list[int] = []
    preds_all: list[int] = []
    batches = 0
    for batch_index, (gray, labels) in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        logits = model(gray.to(device))
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.tolist())
        batches += 1
    return classification_metrics(labels_all, preds_all, num_classes), batches, len(labels_all)


def run_training(
    args: argparse.Namespace,
    save_hook: Callable[[nn.Module, Path, dict], None] | None = None,
    init_hook: Callable[[nn.Module], None] | None = None,
) -> Path:
    """Full gray-only training loop.

    ``save_hook`` is an optional verification observer called after each
    checkpoint write as ``save_hook(module, path, info)``; ``init_hook`` is
    called once with the raw model right after construction, unfreeze and
    optimizer setup, before any optimizer update. Neither hook affects
    training behavior.
    """
    _set_seed(args.seed)
    data_root = PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    # Strict dataset admission: audit report, per-file digests, fingerprint.
    dataset_report = validate_dataset(data_root)

    device = torch_device_name(resolve_device(args.device))
    class_names = class_names_from_manifest(data_root)
    num_classes = len(class_names)

    base_path = Path(args.base)
    base_path = base_path if base_path.is_absolute() else PROJECT_ROOT / base_path
    if not base_path.is_file():
        raise FileNotFoundError(
            f"base checkpoint {base_path} is missing; no automatic download"
        )

    # Identify task and architecture from the actually loaded checkpoint:
    # yolo26n.pt is a detection model and must never pass as a gray base.
    from ultralytics import YOLO

    probe = YOLO(str(base_path)).model
    task = getattr(probe, "task", None)
    if task != "classify":
        raise ValueError(
            f"base checkpoint {base_path} task {task!r} is not classification; "
            "refusing (yolo26n.pt is a detection model, not a Model B base)"
        )
    architecture = architecture_name(probe)
    if not architecture.startswith("yolo26"):
        raise ValueError(
            f"base checkpoint architecture {architecture!r} is not YOLO26; "
            "refusing (legacy YOLOv8 weights are comparison-only and not a "
            "valid gray base for this entry)"
        )

    # Single construction path shared with fusion: fresh 4-class head from
    # the base, dataset class order, moved to the training device.
    backbone, info = prepare_gray_backbone(base_path, "", class_names, device)
    model = backbone

    # Gray-only trains the whole model: Ultralytics checkpoints load with
    # every parameter frozen, so explicitly unfreeze all parameters (no
    # weight re-initialization) before the optimizer is created.
    trainable = _unfreeze_gray_model(model)
    parameters_total = sum(1 for _ in model.parameters())
    if trainable != parameters_total:
        raise RuntimeError(
            f"gray model unfreeze incomplete: only {trainable}/{parameters_total} "
            "parameters are trainable (all parameters must be unfrozen for "
            "gray-only full-model training)"
        )
    _verify_backbone_output(backbone, args.imgsz, num_classes, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    model_ids = {id(param) for param in model.parameters()}
    if optimizer_ids != model_ids:
        raise RuntimeError(
            "optimizer does not cover exactly the model parameters; refusing "
            "to train a partially covered model"
        )
    criterion = nn.CrossEntropyLoss()
    if init_hook is not None:
        init_hook(model.module)

    train_paths, _, _ = read_manifest_split(data_root, "train")
    val_paths, _, _ = read_manifest_split(data_root, "val")
    train_loader = torch.utils.data.DataLoader(
        GrayFusionDataset(train_paths, imgsz=args.imgsz),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0 if sys.platform == "win32" else 8,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = torch.utils.data.DataLoader(
        GrayFusionDataset(val_paths, imgsz=args.imgsz),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 8,
    )

    run_dir = PROJECT_ROOT / "runs" / "train" / args.run_id / "gray_fusion"
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use a new --run-id "
            "instead of overwriting a previous run"
        )
    run_dir.mkdir(parents=True)

    metrics_history: list[dict] = []
    best_key = (-1.0, -1.0)
    best_epoch: int | None = None
    best_path = run_dir / "best.pt"
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for batch_index, (gray, labels) in enumerate(train_loader):
            if args.limit_batches and batch_index >= args.limit_batches:
                break
            gray = gray.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(gray)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach())
            train_batches += 1
        train_loss = train_loss_sum / max(train_batches, 1)

        val_metrics, val_batches, val_samples = _evaluate_gray(
            model, val_loader, device, num_classes, args.limit_batches
        )
        recalls = " ".join(
            f"{name}={val_metrics['per_class_recall'].get(str(i), 0.0):.3f}"
            for i, name in enumerate(class_names)
        )
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} {recalls}"
        )
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_batches": train_batches,
            "val_batches": val_batches,
            "val_samples": val_samples,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_acc": val_metrics["accuracy"],
            "val_per_class_recall": val_metrics["per_class_recall"],
        }
        metrics_history.append(entry)

        selection_key = _selection_key(val_metrics)
        if selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch + 1
            _save_gray_checkpoint(model.module, best_path)
            if save_hook is not None:
                save_hook(model.module, best_path, {"phase": "best", "epoch": epoch + 1, "metrics": entry})

    last_path = run_dir / "last.pt"
    _save_gray_checkpoint(model.module, last_path)
    if save_hook is not None:
        save_hook(model.module, last_path, {"phase": "last", "epoch": args.epochs, "metrics": metrics_history[-1]})

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = {
        "entry": "train_gray_fusion",
        "version": GRAY_VERSION,
        "run_id": args.run_id,
        "base": str(base_path),
        "base_sha256": file_digest(base_path),
        "architecture": architecture,
        "data": str(data_root),
        "dataset_fingerprint": dataset_report.get("audit_fingerprint", ""),
        "class_names": class_names,
        "num_classes": num_classes,
        "parameters_total": parameters_total,
        "parameters_trainable": trainable,
        "imgsz": args.imgsz,
        "preprocessing": PREPROCESSING,
        "optimizer": "AdamW",
        "lr": args.lr,
        "batch": args.batch,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": device,
        "limit_batches": args.limit_batches,
        "smoke": args.limit_batches > 0,
        "selection": "val macro-F1 primary, accuracy tiebreak (full val split)",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_key[0],
        "best_val_acc": best_key[1],
        "train_samples": len(train_paths),
        "val_samples": len(val_paths),
    }
    (run_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_run_id(args.run_id)
    except InvalidRunIdError as exc:
        print(f"invalid run id: {exc}", file=sys.stderr)
        return 2
    try:
        run_dir = run_training(args)
    except DeviceUnavailableError as exc:
        print(f"device unavailable: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"gray training refused: {exc}", file=sys.stderr)
        return 2
    print(f"Gray-only run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
