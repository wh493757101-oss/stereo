"""Tests for the shuffled-polarization inference ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from .conftest import build_v4_dataset

from scripts.ablate_material_shuffle import (
    DonorShuffleDataset,
    derangement,
    parse_args,
    run_ablation,
    routed_predictions,
)


def _model_fingerprint(model) -> dict:
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def _assert_models_equal(before: dict, after: dict) -> None:
    assert set(before) == set(after)
    for key in before:
        assert before[key].dtype == after[key].dtype, key
        assert torch.equal(before[key], after[key]), key


# ---------------------------------------------------------------------------
# donor mapping


def test_derangement_is_reproducible_and_fixed_point_free():
    rng_a = np.random.default_rng(2026)
    rng_b = np.random.default_rng(2026)
    perm_a = derangement(64, rng_a)
    perm_b = derangement(64, rng_b)
    assert np.array_equal(perm_a, perm_b)
    assert not np.any(perm_a == np.arange(64))
    assert sorted(perm_a.tolist()) == list(range(64))


def test_derangement_differs_across_protocol_seeds():
    perms = [derangement(64, np.random.default_rng(seed)) for seed in (2026, 2027, 2028)]
    assert any(not np.array_equal(perms[0], other) for other in perms[1:])


def test_derangement_requires_two_samples():
    with pytest.raises(ValueError, match="at least 2"):
        derangement(1, np.random.default_rng(0))


def test_routed_predictions_replace_only_invalid_samples():
    base = [0, 1, 2, 3]
    fallback = [3, 2, 1, 0]
    # valid, invalid, valid, invalid — invalid samples bypass the model
    valid_flags = [True, False, True, False]
    routed = routed_predictions(base, fallback, valid_flags)
    assert routed == [0, 2, 2, 0]
    with pytest.raises(ValueError, match="lengths"):
        routed_predictions(base, fallback[:-1], valid_flags)


# ---------------------------------------------------------------------------
# donor dataset semantics


def test_donor_dataset_full_mode_swaps_all_polar_channels(scratch_dir):
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4), size=16)
    from models.polar_fusion import FusionClsDataset, read_manifest_split

    paths, _, _ = read_manifest_split(data_root, "val")
    donor = derangement(len(paths), np.random.default_rng(2026))
    dataset = DonorShuffleDataset(paths, "gray_polar", 64, donor, content_only=False)
    assert len(dataset) == len(paths)
    base = FusionClsDataset(paths, imgsz=64)
    for index in range(len(paths)):
        _, own_polar, _, _ = base[index]
        _, donor_polar, _, _ = base[int(donor[index])]
        rgb, aux, quality, class_id = dataset[index]
        assert torch.equal(aux, donor_polar)
        assert torch.equal(rgb, base[index][0])
        assert quality.shape == (4,)


def test_donor_dataset_full_mode_follows_the_donor_quality_vector(scratch_dir):
    """The valid channel and the quality vector describe the same measurement."""
    data_root = build_v4_dataset(
        scratch_dir / "v4", counts=(4, 4, 4), size=16, invalid_indices={"val": {1}},
    )
    from models.polar_fusion import FusionClsDataset, read_manifest_split

    paths, _, _ = read_manifest_split(data_root, "val")
    base = FusionClsDataset(paths, imgsz=64)
    # each sample reads the next sample's measurement, so sample 0 reads the
    # all-invalid sample 1
    donor = (np.arange(len(paths)) + 1) % len(paths)
    dataset = DonorShuffleDataset(paths, "gray_polar", 64, donor, content_only=False)
    for index in range(len(paths)):
        _, _, donor_quality, _ = base[int(donor[index])]
        _, _, quality, _ = dataset[index]
        assert torch.equal(quality, donor_quality)
    _, aux0, quality0, _ = dataset[0]
    assert quality0[0].item() == 0.0
    assert float(aux0[2].max()) == 0.0


def test_donor_dataset_content_only_keeps_own_valid(scratch_dir):
    data_root = build_v4_dataset(scratch_dir / "v4", counts=(4, 4, 4), size=16)
    from models.polar_fusion import FusionClsDataset, read_manifest_split

    paths, _, _ = read_manifest_split(data_root, "val")
    donor = derangement(len(paths), np.random.default_rng(2027))
    dataset = DonorShuffleDataset(paths, "gray_polar", 64, donor, content_only=True)
    base = FusionClsDataset(paths, imgsz=64)
    for index in range(len(paths)):
        _, own_polar, own_quality, _ = base[index]
        _, donor_polar, _, _ = base[int(donor[index])]
        _, aux, quality, _ = dataset[index]
        assert torch.equal(aux[:2], donor_polar[:2])
        assert torch.equal(aux[2], own_polar[2])
        assert torch.equal(quality, own_quality)


# ---------------------------------------------------------------------------
# end-to-end inference-only ablation


def _val_valid_flags(data_root) -> list[bool]:
    """Per-sample validity read independently of the donor view."""
    from models.polar_fusion import FusionClsDataset, read_manifest_split

    paths, _, _ = read_manifest_split(data_root, "val")
    base = FusionClsDataset(paths, imgsz=64)
    return [bool(base[index][2][0].item() > 0) for index in range(len(paths))]


def _fabricate_checkpoint(scratch_dir, variant="gray_polar"):
    data_root = build_v4_dataset(
        scratch_dir / "v4", counts=(8, 4, 4), size=16, invalid_indices={"val": {1}},
    )
    from models.material_fusion import (
        build_material_model,
        material_fusion_metadata,
        save_material_checkpoint,
    )
    from scripts.train_polar_fusion import validate_dataset

    fingerprint = validate_dataset(data_root)["audit_fingerprint"]
    torch.manual_seed(3)
    model = build_material_model(variant, imgsz=64)
    checkpoint = scratch_dir / f"ckpt_{variant}.pt"
    save_material_checkpoint(
        checkpoint, model,
        material_fusion_metadata(
            variant=variant, imgsz=64, data_fingerprint=fingerprint,
            pretrained_sha256="a" * 64, seed=2026, epoch=2, stage="finetune",
            smoke=False, limit_batches=0, val_macro_f1=0.5, val_accuracy=0.6,
            physical_batch=1, accum_steps=2, effective_batch=2, drop_path_rate=0.0,
        ),
    )
    return checkpoint, data_root, model


def test_ablation_reports_logits_and_predictions_without_training(
    scratch_dir, monkeypatch
):
    checkpoint, data_root, model = _fabricate_checkpoint(scratch_dir)
    before = _model_fingerprint(model)
    output = scratch_dir / "report.json"
    args = parse_args([
        "--checkpoint", str(checkpoint), "--split", "val", "--device", "cpu",
        "--seeds", "2026,2027", "--mode", "both", "--batch", "4",
        "--data", str(data_root), "--output", str(output),
    ])
    report = run_ablation(args)
    _assert_models_equal(before, _model_fingerprint(model))
    assert report["split"] == "val" and report["test_split_used"] is False
    assert report["n_samples"] == 4
    assert report["baseline"]["invalid_samples"] == 1
    assert [entry["shuffle_seed"] for entry in report["per_seed"]] == [2026, 2027]
    valid_flags = _val_valid_flags(data_root)
    for entry in report["per_seed"]:
        for mode in ("full", "content_only"):
            block = entry[mode]
            assert block["raw_logit_delta"]["max_abs"] >= 0.0
            assert 0.0 <= block["changed_predictions"]["ratio"] <= 1.0
            assert block["changed_predictions"]["count"] == sum(
                count
                for row in block["changed_predictions"]["transitions"].values()
                for count in row.values()
            )
            assert "baseline_accuracy" in block and "accuracy" in block
        # full mode moves the valid channel *and* the quality vector that
        # routing reads, so the validity-change count must follow the donor
        # mapping instead of being structurally zero
        donor = derangement(len(valid_flags), np.random.default_rng(entry["shuffle_seed"]))
        expected_changed = sum(
            valid_flags[index] != valid_flags[int(donor[index])]
            for index in range(len(valid_flags))
        )
        assert expected_changed >= 1
        assert entry["full"]["valid_mask_changed"] == expected_changed
        assert "valid_mask_changed" not in entry["content_only"]
    donor_payload = json.loads(
        (scratch_dir / "donors_seed2026.json").read_text(encoding="utf-8")
    )
    assert len(donor_payload["donor_index"]) == 4
    assert sorted(donor_payload["donor_index"]) == [0, 1, 2, 3]
    assert output.is_file()


def test_ablation_refuses_single_stream_checkpoint(scratch_dir):
    checkpoint, data_root, _ = _fabricate_checkpoint(scratch_dir, variant="gray")
    args = parse_args([
        "--checkpoint", str(checkpoint), "--split", "val", "--device", "cpu",
        "--data", str(data_root), "--output", str(scratch_dir / "report.json"),
    ])
    with pytest.raises(ValueError, match="not applicable"):
        run_ablation(args)


def test_ablation_refuses_existing_report_and_test_split(scratch_dir):
    checkpoint, data_root, _ = _fabricate_checkpoint(scratch_dir)
    existing = scratch_dir / "report.json"
    existing.write_text("{}", encoding="utf-8")
    args = parse_args([
        "--checkpoint", str(checkpoint), "--split", "val", "--device", "cpu",
        "--data", str(data_root), "--output", str(existing),
    ])
    with pytest.raises(FileExistsError):
        run_ablation(args)
    with pytest.raises(SystemExit):
        parse_args([
            "--checkpoint", str(checkpoint), "--split", "test", "--device", "cpu",
            "--data", str(data_root), "--output", str(scratch_dir / "other.json"),
        ])
