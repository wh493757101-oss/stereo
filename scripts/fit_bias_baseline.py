"""Optional low-cost bias baselines for the material-fusion comparison.

Diagnostic-only helpers (never a substitute for the main matrix):

- ``fixed_bias``   — four class biases added to the *frozen* historical
  gray Model B logits. Fitted on the train split by bounded gradient
  descent on the cached logits; no image is read during the fit.
- ``quality_only`` — a linear correction reading only the V4 quality
  vector (never the polarization images), added to the same frozen gray
  logits. Fitted on the train split with the same bounded budget.

Fitting uses train only; evaluation uses val only. The test split is
refused. Nothing here updates the frozen gray model, and ``collect`` never
invokes this script — both tasks are started explicitly by the user.

Usage:
    python scripts/fit_bias_baseline.py --mode both --device 0 \
        --output analysis/runs/material_bias_baseline/report.json
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

from core.material_fusion_dataset import CLASS_NAMES, MaterialFusionDataset
from models.polar_fusion import read_manifest_split, torch_device_name
from scripts.run_material_experiments import detailed_metrics

GRAY_VARIANT = "gray"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="both",
                        choices=["fixed_bias", "quality_only", "both"])
    parser.add_argument("--gray-weights", default=None,
                        help="Frozen gray Model B checkpoint "
                             "(default: the experiment config's fallback_weights).")
    parser.add_argument("--steps", type=int, default=500,
                        help="Bounded fit budget in optimizer steps (default: %(default)s).")
    parser.add_argument("--fit-lr", type=float, default=0.01)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--data", default="datasets/underwater_cls_fusion_v4_band",
                        help="Dataset root (default: the V4 fusion dataset).")
    parser.add_argument("--output", required=True,
                        help="Report JSON path (must not exist).")
    return parser.parse_args(argv)


@torch.no_grad()
def collect_gray_logits(model, loader, device: str) -> dict:
    """Cached frozen-gray logits/quality/labels (also the eval baseline)."""
    model.eval()
    logits_all: list[list[float]] = []
    quality_all: list[list[float]] = []
    labels_all: list[int] = []
    for rgb, quality, labels in loader:
        logits = model(rgb.to(device))
        logits_all.extend(logits.cpu().tolist())
        quality_all.extend(quality.tolist())
        labels_all.extend(labels.tolist())
    return {
        "logits": torch.tensor(logits_all),
        "quality": torch.tensor(quality_all),
        "labels": torch.tensor(labels_all),
    }


def fit_fixed_bias(data: dict, steps: int, fit_lr: float) -> torch.nn.Parameter:
    """Four class biases fitted on cached train logits (bounded budget)."""
    bias = torch.nn.Parameter(torch.zeros(len(CLASS_NAMES)))
    optimizer = torch.optim.Adam([bias], lr=fit_lr)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = criterion(data["logits"] + bias, data["labels"])
        loss.backward()
        optimizer.step()
    return bias.detach()


def fit_quality_correction(data: dict, steps: int, fit_lr: float) -> torch.nn.Linear:
    """Quality-vector-only linear correction on cached train logits."""
    correction = torch.nn.Linear(data["quality"].shape[1], len(CLASS_NAMES))
    optimizer = torch.optim.Adam(correction.parameters(), lr=fit_lr)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = criterion(data["logits"] + correction(data["quality"]), data["labels"])
        loss.backward()
        optimizer.step()
    return correction.eval()


def run_baseline(args: argparse.Namespace) -> dict:
    from scripts.eval_material_fusion import load_fallback_gray

    if args.steps < 1:
        raise ValueError("--steps must be >= 1 (bounded fit budget)")
    device = torch_device_name(args.device)
    import yaml

    template_path = PROJECT_ROOT / "configs/experiments/material_fusion_v1.yaml"
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    gray_path = args.gray_weights or template["fallback_weights"]
    gray_file = Path(gray_path)
    if not gray_file.is_absolute():
        gray_file = PROJECT_ROOT / gray_file
    gray_model, gray_identity = load_fallback_gray(gray_file, device, 224)

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    fit_paths, _, _ = read_manifest_split(data_root, "train")
    eval_paths, _, _ = read_manifest_split(data_root, "val")

    fit_data = collect_gray_logits(
        gray_model,
        torch.utils.data.DataLoader(
            MaterialFusionDataset(fit_paths, variant=GRAY_VARIANT, imgsz=224),
            batch_size=args.batch, shuffle=False, num_workers=0,
        ),
        device,
    )
    eval_data = collect_gray_logits(
        gray_model,
        torch.utils.data.DataLoader(
            MaterialFusionDataset(eval_paths, variant=GRAY_VARIANT, imgsz=224),
            batch_size=args.batch, shuffle=False, num_workers=0,
        ),
        device,
    )

    baseline = detailed_metrics(
        eval_data["labels"].tolist(),
        eval_data["logits"].argmax(dim=1).tolist(),
    )
    modes: dict = {}
    if args.mode in ("fixed_bias", "both"):
        bias = fit_fixed_bias(fit_data, args.steps, args.fit_lr)
        corrected = detailed_metrics(
            eval_data["labels"].tolist(),
            (eval_data["logits"] + bias).argmax(dim=1).tolist(),
        )
        modes["fixed_bias"] = {
            "params": len(CLASS_NAMES),
            "reads_polar_images": False,
            "val": corrected,
            "val_macro_f1_delta_vs_gray": corrected["macro_f1"] - baseline["macro_f1"],
        }
    if args.mode in ("quality_only", "both"):
        correction = fit_quality_correction(fit_data, args.steps, args.fit_lr)
        corrected = detailed_metrics(
            eval_data["labels"].tolist(),
            (eval_data["logits"] + correction(eval_data["quality"])).argmax(dim=1).tolist(),
        )
        modes["quality_only"] = {
            "params": sum(p.numel() for p in correction.parameters()),
            "reads_polar_images": False,
            "reads_quality_vector_only": True,
            "val": corrected,
            "val_macro_f1_delta_vs_gray": corrected["macro_f1"] - baseline["macro_f1"],
        }

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"report {output} already exists; refusing to overwrite")
    report = {
        "gray_model_identity": gray_identity,
        "gray_weights": str(gray_file),
        "fit_split": "train",
        "eval_split": "val",
        "test_split_used": False,
        "fit_steps": int(args.steps),
        "fit_lr": float(args.fit_lr),
        "frozen_gray_baseline_val": baseline,
        "modes": modes,
        "note": "Diagnostic only; does not replace the main comparison matrix.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_baseline(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"bias baseline refused: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    print(f"bias baseline report: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
