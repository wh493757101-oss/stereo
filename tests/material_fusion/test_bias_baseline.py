"""Tests for the optional bias/quality-only baseline (fit on train, eval on val)."""

from __future__ import annotations

import pytest
import torch

from .conftest import build_v4_dataset

from scripts.fit_bias_baseline import (
    collect_gray_logits,
    fit_fixed_bias,
    fit_quality_correction,
    parse_args,
    run_baseline,
)


class _StubGray(torch.nn.Module):
    """Deterministic frozen-gray stand-in: logits from the mean color."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, rgb):
        return self.linear(rgb.mean(dim=(2, 3)))


def _make_data():
    torch.manual_seed(0)
    logits = torch.randn(12, 4)
    quality = torch.rand(12, 4)
    labels = torch.randint(0, 4, (12,))
    return {"logits": logits, "quality": quality, "labels": labels}


def test_fit_fixed_bias_does_not_touch_cached_logits_and_reduces_loss():
    data = _make_data()
    frozen_logits = data["logits"].clone()
    criterion = torch.nn.CrossEntropyLoss()
    loss_before = criterion(data["logits"], data["labels"]).item()
    bias = fit_fixed_bias(data, steps=50, fit_lr=0.05)
    assert torch.equal(data["logits"], frozen_logits)
    loss_after = criterion(data["logits"] + bias, data["labels"]).item()
    assert loss_after < loss_before
    assert bias.shape == (4,)


def test_fit_quality_correction_uses_only_quality_vector():
    data = _make_data()
    correction = fit_quality_correction(data, steps=10, fit_lr=0.01)
    assert sum(p.numel() for p in correction.parameters()) == 4 * 4 + 4
    assert correction.training is False


def test_run_baseline_fits_on_train_and_evaluates_on_val_only(
    scratch_dir, monkeypatch
):
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(8, 4, 4), size=16)
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "report.json"
    args = parse_args([
        "--mode", "both", "--device", "cpu", "--steps", "20", "--batch", "4",
        "--data", str(data_root), "--output", str(output),
    ])
    report = run_baseline(args)
    assert report["fit_split"] == "train"
    assert report["eval_split"] == "val"
    assert report["test_split_used"] is False
    assert report["fit_steps"] == 20
    assert report["gray_model_identity"] == "stub gray identity"
    assert report["frozen_gray_baseline_val"]["samples"] == 4
    assert set(report["modes"]) == {"fixed_bias", "quality_only"}
    for mode in report["modes"].values():
        assert mode["reads_polar_images"] is False
        assert mode["val"]["samples"] == 4
        assert isinstance(mode["val_macro_f1_delta_vs_gray"], float)
    assert report["modes"]["quality_only"]["reads_quality_vector_only"] is True
    assert output.is_file()


def test_run_baseline_refuses_existing_report(scratch_dir, monkeypatch):
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(8, 4, 4), size=16)
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    output = scratch_dir / "report.json"
    output.write_text("{}", encoding="utf-8")
    args = parse_args([
        "--mode", "fixed_bias", "--device", "cpu", "--data", str(data_root),
        "--output", str(output),
    ])
    with pytest.raises(FileExistsError):
        run_baseline(args)


def test_run_baseline_requires_bounded_positive_steps(scratch_dir, monkeypatch):
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(8, 4, 4), size=16)
    monkeypatch.setattr(
        "scripts.eval_material_fusion.load_fallback_gray",
        lambda path, device, imgsz: (_StubGray(), "stub gray identity"),
    )
    args = parse_args([
        "--mode", "fixed_bias", "--device", "cpu", "--steps", "0",
        "--data", str(data_root), "--output", str(scratch_dir / "report.json"),
    ])
    with pytest.raises(ValueError, match="bounded"):
        run_baseline(args)


def test_collect_gray_logits_shapes(scratch_dir):
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4), size=16)
    from core.material_fusion_dataset import MaterialFusionDataset
    from models.polar_fusion import read_manifest_split

    paths, _, _ = read_manifest_split(data_root, "val")
    data = collect_gray_logits(
        _StubGray(),
        torch.utils.data.DataLoader(
            MaterialFusionDataset(paths, variant="gray", imgsz=16),
            batch_size=4, shuffle=False, num_workers=0,
        ),
        "cpu",
    )
    assert data["logits"].shape == (4, 4)
    assert data["quality"].shape == (4, 4)
    assert data["labels"].shape == (4,)
