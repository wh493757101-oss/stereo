"""Evaluate a trained Polar Fusion checkpoint on the V3 test split.

Loads a checkpoint written by ``scripts/train_polar_fusion.py``, rebuilds
the fusion model around a freshly loaded gray backbone (the architecture
named in the checkpoint metadata), and reports paired metrics:

- gray-only accuracy / macro-F1 (gate effectively 0 everywhere)
- fusion accuracy / macro-F1 (learned gate)
- gate statistics and the fraction of samples where the fusion prediction
  differs from the gray-only prediction

The gray backbone checkpoint named by ``--base`` must be available locally;
evaluation never downloads weights and never trains.

Usage:
    python scripts/eval_polar_fusion.py --checkpoint runs/train/<run-id>/polar_fusion/best.pt \
        --output analysis/runs/<run-id>/polar_fusion_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.polar_fusion import (
    FusionClsDataset,
    load_fusion_checkpoint,
    read_fusion_metadata,
    read_manifest_split,
)
from scripts.train_polar_fusion import load_gray_backbone

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--base",
        default=None,
        help="Gray backbone checkpoint; defaults to the one recorded in the "
        "fusion checkpoint metadata.",
    )
    parser.add_argument(
        "--data",
        default="datasets/underwater_cls_fusion_v3",
        help="V3 fusion dataset root (default: %(default)s).",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report path (default: <checkpoint dir>/eval_<split>.json).",
    )
    return parser.parse_args(argv)


def _macro_f1(labels: list[int], preds: list[int], num_classes: int) -> float:
    f1s = []
    for cls in range(num_classes):
        tp = sum(1 for l, p in zip(labels, preds) if l == cls and p == cls)
        fp = sum(1 for l, p in zip(labels, preds) if l != cls and p == cls)
        fn = sum(1 for l, p in zip(labels, preds) if l == cls and p != cls)
        if tp == 0:
            f1s.append(0.0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str,
    num_classes: int,
) -> dict:
    model.eval()
    gray_preds: list[int] = []
    fusion_preds: list[int] = []
    labels_all: list[int] = []
    gates: list[float] = []
    for gray, polar, quality, labels in loader:
        out = model(gray.to(device), polar.to(device), quality.to(device))
        gray_preds.extend(out["gray_logits"].argmax(dim=1).cpu().tolist())
        fusion_preds.extend(out["final_logits"].argmax(dim=1).cpu().tolist())
        gates.extend(out["gate"].squeeze(1).cpu().tolist())
        labels_all.extend(labels.tolist())

    num = len(labels_all)
    gray_acc = sum(1 for l, p in zip(labels_all, gray_preds) if l == p) / max(num, 1)
    fusion_acc = sum(1 for l, p in zip(labels_all, fusion_preds) if l == p) / max(num, 1)
    changed = sum(1 for g, f in zip(gray_preds, fusion_preds) if g != f)
    return {
        "samples": num,
        "gray": {
            "accuracy": gray_acc,
            "macro_f1": _macro_f1(labels_all, gray_preds, num_classes),
        },
        "fusion": {
            "accuracy": fusion_acc,
            "macro_f1": _macro_f1(labels_all, fusion_preds, num_classes),
        },
        "prediction_changes": changed,
        "prediction_change_ratio": changed / max(num, 1),
        "gate": {
            "mean": float(np.mean(gates)) if gates else 0.0,
            "min": float(np.min(gates)) if gates else 0.0,
            "max": float(np.max(gates)) if gates else 0.0,
            "zero_fraction": float(np.mean([g == 0.0 for g in gates])) if gates else 0.0,
        },
        "per_class_counts": dict(Counter(labels_all)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    device = args.device
    metadata = read_fusion_metadata(checkpoint)

    base_name = args.base or metadata.base_model
    base_path = Path(base_name)
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_name
    if not base_path.is_file():
        print(
            f"gray backbone checkpoint {base_path} is missing; evaluation "
            "requires it locally (no automatic download).",
            file=sys.stderr,
        )
        return 2

    backbone = load_gray_backbone(base_path, device)
    model, metadata, _ = load_fusion_checkpoint(checkpoint, backbone)
    model = model.to(device)

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    imgsz = args.imgsz or metadata.imgsz
    paths, _, _ = read_manifest_split(data_root, args.split)
    loader = torch.utils.data.DataLoader(
        FusionClsDataset(paths, imgsz=imgsz),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 8,
    )

    result = evaluate(model, loader, device, model.num_classes)
    result.update(
        {
            "checkpoint": str(checkpoint),
            "base_model": str(base_path),
            "split": args.split,
            "version": metadata.version,
            "class_names": list(metadata.class_names),
            "quality_vector_keys": list(metadata.quality_vector_keys),
        }
    )

    output = (
        Path(args.output)
        if args.output
        else checkpoint.parent / f"eval_{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Eval report: {output}")
    print(
        f"gray acc={result['gray']['accuracy']:.4f} "
        f"fusion acc={result['fusion']['accuracy']:.4f} "
        f"(gate mean={result['gate']['mean']:.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
