"""Train the Polar Fusion classification model on the V3 fusion dataset.

Phases:
    freeze : train only the polar delta head and the quality gate on top of
             a frozen gray backbone (default, first stage).
    joint  : optional follow-up finetune with the gray backbone unfrozen at
             a lower learning rate.

The gray backbone is the Ultralytics classification model (default base
``yolo26n-cls.pt``) loaded from a local checkpoint. If the base checkpoint
is missing the run is refused with a clear message (no network download).

``--dry-run`` validates the dataset (manifest counts, sample decoding,
quality vector shape) and the fusion structure (delta/gate forward shapes,
forced gate behavior) without training and without creating run outputs.

Usage:
    python scripts/train_polar_fusion.py --run-id run_x --dry-run
    python scripts/train_polar_fusion.py --run-id run_x --phase freeze
    python scripts/train_polar_fusion.py --run-id run_x --phase joint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import (
    QUALITY_VECTOR_LENGTH,
    load_fusion_sample,
)
from models.polar_fusion import (
    DEFAULT_BASE_MODEL,
    FUSION_VERSION,
    FusionCheckpointMetadata,
    FusionClsDataset,
    PolarFusionModel,
    class_names_from_manifest,
    fusion_metadata,
    read_manifest_split,
    save_fusion_checkpoint,
)
from scripts.train_models import InvalidRunIdError, validate_run_id

DEFAULT_SEED = 2026
DEFAULT_DATA = "datasets/underwater_cls_fusion_v3"
DEFAULT_PHASE = "freeze"
PHASES = ("freeze", "joint")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id grouping outputs under runs/train/<run-id>/polar_fusion.",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE_MODEL,
        help=f"Gray backbone base checkpoint (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="V3 fusion dataset root (default: %(default)s).",
    )
    parser.add_argument("--device", default="0", help="Training device.")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate.")
    parser.add_argument(
        "--gray-lr", type=float, default=1e-5, help="Gray backbone lr (joint phase)."
    )
    parser.add_argument("--phase", choices=PHASES, default=DEFAULT_PHASE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and structure only; no training, no run outputs.",
    )
    return parser.parse_args(argv)


def validate_dataset(data_root: Path) -> dict:
    """Dataset gate for dry-run and training: counts, decoding, quality shape."""
    manifest = data_root / "dataset_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest}")

    class_names = class_names_from_manifest(data_root)
    if len(class_names) < 2:
        raise ValueError(f"need >= 2 classes, found {class_names}")

    report: dict = {"class_names": class_names, "splits": {}}
    for split in ("train", "val", "test"):
        paths, class_ids, _ = read_manifest_split(data_root, split)
        report["splits"][split] = len(paths)
        # Decode the first two samples of every split.
        for path in paths[:2]:
            sample = load_fusion_sample(path)
            if sample.quality.shape != (QUALITY_VECTOR_LENGTH,):
                raise ValueError(f"bad quality vector in {path}")
            if int(sample.class_id) not in set(class_ids):
                raise ValueError(f"class id {sample.class_id} out of range in {path}")
    if report["splits"].get("train", 0) == 0 or report["splits"].get("val", 0) == 0:
        raise ValueError("train and val splits must both be non-empty")
    return report


class _StructureStubBackbone(nn.Module):
    """Minimal stand-in backbone for structure validation without weights."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes)
        )

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        return self.net(gray)


def structure_check(num_classes: int, imgsz: int) -> dict:
    """Validate delta/gate/fusion shapes and the forced-gate behavior."""
    torch.manual_seed(0)
    model = PolarFusionModel(_StructureStubBackbone(num_classes), num_classes)
    model.eval()
    batch = 3
    gray = torch.rand(batch, 3, imgsz, imgsz)
    polar = torch.rand(batch, 3, imgsz, imgsz)
    quality = torch.rand(batch, QUALITY_VECTOR_LENGTH)
    with torch.no_grad():
        out = model(gray, polar, quality)
    assert out["final_logits"].shape == (batch, num_classes)
    assert out["gray_logits"].shape == (batch, num_classes)
    assert out["polar_delta"].shape == (batch, num_classes)
    assert out["gate"].shape == (batch, 1)
    assert torch.all((out["gate"] >= 0) & (out["gate"] <= 1))

    # valid_ratio == 0 forces the gate to exactly 0 -> pure gray logits.
    zero_quality = quality.clone()
    zero_quality[:, 0] = 0.0
    with torch.no_grad():
        gated = model(gray, polar, zero_quality)
    assert torch.all(gated["gate"] == 0.0)
    assert torch.equal(gated["final_logits"], gated["gray_logits"])

    # Explicit invalid flag (sync/matching failure) also forces gate 0.
    with torch.no_grad():
        flagged = model(gray, polar, quality, polar_invalid=torch.ones(batch))
    assert torch.all(flagged["gate"] == 0.0)
    return {"num_classes": num_classes, "imgsz": imgsz, "forced_gate_verified": True}


def dry_run(args: argparse.Namespace) -> int:
    data_root = PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    dataset_report = validate_dataset(data_root)
    structure_report = structure_check(len(dataset_report["class_names"]), args.imgsz)
    base_path = PROJECT_ROOT / args.base
    base_present = base_path.is_file()
    report = {
        "dry_run": True,
        "version": FUSION_VERSION,
        "data_root": str(data_root),
        "dataset": dataset_report,
        "structure": structure_report,
        "base_model": args.base,
        "base_checkpoint_present": base_present,
        "base_checkpoint_path": str(base_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not base_present:
        print(
            f"NOTE: base checkpoint {args.base} is not present locally; "
            "formal training requires it (no automatic download).",
            file=sys.stderr,
        )
    return 0


def load_gray_backbone(base_path: Path, device: str) -> nn.Module:
    """Load the Ultralytics classification model's inner nn.Module."""
    from ultralytics import YOLO

    yolo = YOLO(str(base_path))
    backbone = yolo.model
    backbone.to(device)
    return backbone


def run_training(args: argparse.Namespace) -> Path:
    """Full training loop (freeze or joint). Not executed by --dry-run."""
    data_root = PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    validate_dataset(data_root)
    base_path = PROJECT_ROOT / args.base if not Path(args.base).is_absolute() else Path(args.base)
    if not base_path.is_file():
        raise FileNotFoundError(
            f"base checkpoint {base_path} is missing; no automatic download"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    backbone = load_gray_backbone(base_path, device)
    class_names = class_names_from_manifest(data_root)
    model = PolarFusionModel(backbone, num_classes=len(class_names)).to(device)

    # Phase freeze: gray weights are fixed; joint: everything trains, with
    # the backbone at the lower --gray-lr rate.
    joint = args.phase == "joint"
    model.set_gray_frozen(not joint)
    head_params = [
        {"params": model.delta_net.parameters(), "lr": args.lr},
        {"params": model.gate_net.parameters(), "lr": args.lr},
    ]
    if joint:
        head_params.append({"params": model.gray_backbone.parameters(), "lr": args.gray_lr})
    optimizer = torch.optim.AdamW(head_params)
    criterion = nn.CrossEntropyLoss()

    train_paths, _, _ = read_manifest_split(data_root, "train")
    val_paths, _, _ = read_manifest_split(data_root, "val")
    train_loader = torch.utils.data.DataLoader(
        FusionClsDataset(train_paths, imgsz=args.imgsz),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0 if sys.platform == "win32" else 8,
    )
    val_loader = torch.utils.data.DataLoader(
        FusionClsDataset(val_paths, imgsz=args.imgsz),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 8,
    )

    run_dir = PROJECT_ROOT / "runs" / "train" / args.run_id / "polar_fusion"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata: FusionCheckpointMetadata = fusion_metadata(
        class_names, base_model=args.base, imgsz=args.imgsz
    )
    best_val_acc = -1.0
    best_path = run_dir / "best.pt"
    for epoch in range(args.epochs):
        model.train()
        for gray, polar, quality, labels in train_loader:
            gray = gray.to(device)
            polar = polar.to(device)
            quality = quality.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(gray, polar, quality)
            loss = criterion(out["final_logits"], labels)
            loss.backward()
            optimizer.step()

        val_acc = _evaluate_accuracy(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_fusion_checkpoint(
                best_path,
                model,
                metadata,
                extra={"phase": args.phase, "epoch": epoch + 1, "val_acc": val_acc},
            )
    save_fusion_checkpoint(
        run_dir / "last.pt",
        model,
        metadata,
        extra={"phase": args.phase, "epochs": args.epochs, "best_val_acc": best_val_acc},
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "phase": args.phase,
                "base": args.base,
                "data": str(data_root),
                "imgsz": args.imgsz,
                "batch": args.batch,
                "epochs": args.epochs,
                "lr": args.lr,
                "gray_lr": args.gray_lr,
                "seed": args.seed,
                "version": FUSION_VERSION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


@torch.no_grad()
def _evaluate_accuracy(model: PolarFusionModel, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for gray, polar, quality, labels in loader:
        out = model(
            gray.to(device), polar.to(device), quality.to(device)
        )
        predictions = out["final_logits"].argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_run_id(args.run_id)
    except InvalidRunIdError as exc:
        print(f"invalid run id: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(args)
    run_dir = run_training(args)
    print(f"Polar fusion run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
