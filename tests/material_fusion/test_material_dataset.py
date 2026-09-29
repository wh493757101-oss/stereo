"""Dataset adapter tests: variant mapping, V4 tensor consistency, admission."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.material_fusion_dataset import (
    PREPROCESSING_VERSION,
    MaterialFusionDataset,
    variant_inputs,
)
from core.fusion_dataset import dataset_fingerprint, verify_dataset_integrity
from models.polar_fusion import FusionClsDataset, read_manifest_split

from .conftest import build_v4_dataset


class TestVariantInputs:
    def test_gray_variant_returns_no_aux(self):
        gray = torch.rand(3, 8, 8)
        polar = torch.rand(3, 8, 8)
        rgb, aux = variant_inputs("gray", gray, polar)
        assert torch.equal(rgb, gray)
        assert aux is None

    def test_gray_gray_valid_aux_is_gray_gray_valid(self):
        gray = torch.rand(3, 8, 8)
        polar = torch.rand(3, 8, 8)
        _, aux = variant_inputs("gray_gray_valid", gray, polar)
        assert torch.equal(aux[0], gray[0])
        assert torch.equal(aux[1], gray[0])
        assert torch.equal(aux[2], polar[2])

    def test_gray_valid_aux_is_zero_zero_valid(self):
        gray = torch.rand(3, 8, 8)
        polar = torch.rand(3, 8, 8)
        _, aux = variant_inputs("gray_valid", gray, polar)
        assert torch.equal(aux[0], torch.zeros_like(gray[0]))
        assert torch.equal(aux[1], torch.zeros_like(gray[0]))
        assert torch.equal(aux[2], polar[2])

    def test_gray_polar_aux_is_the_v4_polar_tensor(self):
        gray = torch.rand(3, 8, 8)
        polar = torch.rand(3, 8, 8)
        _, aux = variant_inputs("gray_polar", gray, polar)
        assert torch.equal(aux, polar)

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError):
            variant_inputs("rgb_polar", torch.rand(3, 8, 8), torch.rand(3, 8, 8))


class TestDatasetAdapter:
    def test_preprocessing_version_is_pinned(self):
        assert PREPROCESSING_VERSION == "v4_band_224_area_linear_v1"

    def test_adapter_reproduces_v4_tensors_exactly(self, scratch_dir):
        root = build_v4_dataset(scratch_dir / "v4")
        paths, _, _ = read_manifest_split(root, "val")
        base = FusionClsDataset(paths, imgsz=64)
        adapted = MaterialFusionDataset(paths, variant="gray_polar", imgsz=64)
        for index in range(len(paths)):
            gray_t, polar_t, quality_t, class_id = base[index]
            rgb, aux, quality_a, class_a = adapted[index]
            assert torch.equal(rgb, gray_t)
            assert torch.equal(aux, polar_t)
            assert torch.equal(quality_a, quality_t)
            assert class_a == class_id

    def test_gray_variant_yields_three_tuple(self, scratch_dir):
        root = build_v4_dataset(scratch_dir / "v4")
        paths, _, _ = read_manifest_split(root, "val")
        dataset = MaterialFusionDataset(paths, variant="gray", imgsz=64)
        rgb, quality, class_id = dataset[0]
        assert rgb.shape == (3, 64, 64)
        assert quality.shape == (4,)
        assert isinstance(class_id, int)

    def test_dual_variants_yield_four_tuple(self, scratch_dir):
        root = build_v4_dataset(scratch_dir / "v4")
        paths, _, _ = read_manifest_split(root, "val")
        for variant in ("gray_gray_valid", "gray_valid", "gray_polar"):
            dataset = MaterialFusionDataset(paths, variant=variant, imgsz=64)
            rgb, aux, quality, class_id = dataset[0]
            assert rgb.shape == (3, 64, 64)
            assert aux.shape == (3, 64, 64)
            assert quality.shape == (4,)

    def test_imgsz_mismatch_with_v4_resize_rejected(self, scratch_dir):
        root = build_v4_dataset(scratch_dir / "v4")
        paths, _, _ = read_manifest_split(root, "val")
        with pytest.raises(ValueError):
            MaterialFusionDataset(paths, variant="gray", imgsz=0)


class TestDatasetAdmission:
    def test_valid_dataset_passes_and_fingerprints(self, scratch_dir):
        from scripts.train_polar_fusion import validate_dataset

        root = build_v4_dataset(scratch_dir / "v4")
        report = validate_dataset(root)
        assert report["splits"] == {"train": 8, "val": 4, "test": 4}
        assert report["audit_passed"] is True
        assert report["audit_fingerprint"] == dataset_fingerprint(root)
        assert verify_dataset_integrity(root)["verified_files"] == 16

    def test_tampered_sample_is_rejected(self, scratch_dir):
        from scripts.train_polar_fusion import validate_dataset

        root = build_v4_dataset(scratch_dir / "v4")
        sample = next(root.rglob("*.npz"))
        with np.load(sample) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["gray"] = (arrays["gray"].astype(np.uint16) + 1).astype(np.uint8)
        np.savez_compressed(sample, **arrays)
        with pytest.raises(ValueError, match="changed since the audit"):
            validate_dataset(root)

    def test_added_sample_is_rejected(self, scratch_dir):
        from scripts.train_polar_fusion import validate_dataset

        root = build_v4_dataset(scratch_dir / "v4")
        source = next(root.rglob("*.npz"))
        target = root / "train" / "metal_submarine" / "extra_sample.npz"
        target.write_bytes(source.read_bytes())
        with pytest.raises(ValueError, match="do not match the manifest"):
            validate_dataset(root)

    def test_edited_manifest_is_rejected(self, scratch_dir):
        from scripts.train_polar_fusion import validate_dataset

        root = build_v4_dataset(scratch_dir / "v4")
        manifest = root / "dataset_manifest.csv"
        text = manifest.read_text(encoding="utf-8")
        manifest.write_text(text.replace("train_sample_000", "train_sample_999", 1), encoding="utf-8")
        with pytest.raises(ValueError, match="digest mismatch"):
            validate_dataset(root)
