"""Tests for core.fusion_dataset and scripts.prepare_cls_fusion_dataset."""

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.fusion_dataset import (
    NPZ_FIELDS,
    QUALITY_VECTOR_KEYS,
    QUALITY_VECTOR_LENGTH,
    build_quality_vector,
    load_fusion_sample,
    quality_vector_from_result,
    save_fusion_sample,
)
from core.polar_compute import compute_polar_features
from core.stereo_matching import StereoMatcher, StereoMatcherConfig
from scripts.prepare_cls_fusion_dataset import (
    FusionSampleRecord,
    audit_fusion_dataset,
    build_fusion_dataset,
)


@pytest.fixture
def tmp_path() -> Path:
    """Isolated work dir under the system temp (pytest basetemp under tests/
    can be locked by another process on Windows)."""
    workdir = Path(tempfile.mkdtemp(prefix="fusion_dataset_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class TestQualityVector:
    def test_key_order_and_length(self):
        assert QUALITY_VECTOR_KEYS == (
            "valid_ratio",
            "in_bounds_ratio",
            "brightness_valid_ratio",
            "mean_abs_q",
        )
        assert QUALITY_VECTOR_LENGTH == 4

    def test_build_quality_vector_order(self):
        vector = build_quality_vector(0.5, 0.25, 0.75, 0.1)
        assert vector.dtype == np.float32
        np.testing.assert_allclose(vector, [0.5, 0.25, 0.75, 0.1])

    def test_quality_from_result_full_mask(self):
        left = np.full((4, 8), 100, dtype=np.uint8)
        right = np.full((4, 8), 60, dtype=np.uint8)
        disparity = np.full((4, 8), 2.0, dtype=np.float32)
        mask = np.zeros((4, 8), dtype=np.uint8)
        mask[1:3, 3:6] = 1
        result = compute_polar_features(left, right, disparity, object_mask=mask)

        vector, components = quality_vector_from_result(result, mask)

        # Every mask pixel is valid, in bounds and bright; q = 40/160
        assert components["valid_ratio"] == pytest.approx(1.0)
        assert components["in_bounds_ratio"] == pytest.approx(1.0)
        assert components["brightness_valid_ratio"] == pytest.approx(1.0)
        assert components["mean_abs_q"] == pytest.approx(0.25, abs=1e-5)
        np.testing.assert_allclose(
            vector,
            [1.0, 1.0, 1.0, 0.25],
            rtol=1e-5,
        )

    def test_quality_from_result_empty_mask(self):
        left = np.full((4, 8), 100, dtype=np.uint8)
        right = np.full((4, 8), 100, dtype=np.uint8)
        disparity = np.full((4, 8), 2.0, dtype=np.float32)
        result = compute_polar_features(left, right, disparity)
        empty = np.zeros((4, 8), dtype=np.uint8)

        vector, components = quality_vector_from_result(result, empty)

        assert all(value == 0.0 for value in components.values())
        np.testing.assert_array_equal(vector, np.zeros(QUALITY_VECTOR_LENGTH))


class TestSaveLoadRoundtrip:
    def _sample_arrays(self, shape=(6, 8)):
        rng = np.random.default_rng(0)
        gray = rng.integers(0, 256, size=shape, dtype=np.uint8)
        signed = rng.uniform(-1, 1, size=shape).astype(np.float32)
        abs_q = np.abs(signed)
        valid = (rng.uniform(size=shape) > 0.5).astype(np.uint8)
        quality = build_quality_vector(0.5, 0.6, 0.7, 0.2)
        return gray, signed, abs_q, valid, quality

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        gray, signed, abs_q, valid, quality = self._sample_arrays()
        path = tmp_path / "train" / "cls" / "sample.npz"
        save_fusion_sample(path, gray, signed, abs_q, valid, quality, class_id=2)

        sample = load_fusion_sample(path)
        np.testing.assert_array_equal(sample.gray, gray)
        np.testing.assert_array_equal(sample.signed_q, signed)
        np.testing.assert_array_equal(sample.abs_q, abs_q)
        np.testing.assert_array_equal(sample.valid, valid)
        np.testing.assert_array_equal(sample.quality, quality)
        assert sample.class_id == 2

    def test_saved_fields_are_exactly_the_contract(self, tmp_path):
        gray, signed, abs_q, valid, quality = self._sample_arrays()
        path = save_fusion_sample(
            tmp_path / "s.npz", gray, signed, abs_q, valid, quality, class_id=0
        )
        with np.load(path) as data:
            assert set(data.files) == set(NPZ_FIELDS)
            assert data["gray"].dtype == np.uint8
            assert data["signed_q"].dtype == np.float32
            assert data["abs_q"].dtype == np.float32
            assert data["valid"].dtype == np.uint8
            assert data["quality"].dtype == np.float32
            assert data["class_id"].dtype == np.int64
            assert data["class_id"].shape == ()

    def test_shape_mismatch_rejected(self, tmp_path):
        gray, signed, abs_q, valid, quality = self._sample_arrays()
        with pytest.raises(ValueError, match="signed_q shape"):
            save_fusion_sample(
                tmp_path / "s.npz",
                gray,
                signed[:, :4],
                abs_q,
                valid,
                quality,
                class_id=0,
            )

    def test_bad_quality_shape_rejected(self, tmp_path):
        gray, signed, abs_q, valid, _ = self._sample_arrays()
        with pytest.raises(ValueError, match="quality"):
            save_fusion_sample(
                tmp_path / "s.npz",
                gray,
                signed,
                abs_q,
                valid,
                np.zeros(3, dtype=np.float32),
                class_id=0,
            )

    def test_load_rejects_missing_field(self, tmp_path):
        np.savez(tmp_path / "partial.npz", gray=np.zeros((2, 2), np.uint8))
        with pytest.raises(ValueError, match="missing fields"):
            load_fusion_sample(tmp_path / "partial.npz")

    def test_load_rejects_nan(self, tmp_path):
        gray = np.zeros((2, 2), np.uint8)
        nan_signed = np.full((2, 2), np.nan, np.float32)
        save = tmp_path / "nan.npz"
        np.savez_compressed(
            save,
            gray=gray,
            signed_q=nan_signed,
            abs_q=np.zeros((2, 2), np.float32),
            valid=np.zeros((2, 2), np.uint8),
            quality=np.zeros(QUALITY_VECTOR_LENGTH, np.float32),
            class_id=np.asarray(0, np.int64),
        )
        with pytest.raises(ValueError, match="NaN"):
            load_fusion_sample(save)


def make_textured_pair(width=256, height=128, disparity=32, seed=0):
    """Small synthetic rectified pair: right(x) = left(x + disparity)."""
    rng = np.random.default_rng(seed)
    left = rng.integers(30, 226, size=(height, width), dtype=np.uint8)
    right = rng.integers(30, 226, size=(height, width), dtype=np.uint8)
    if disparity > 0:
        right[:, : width - disparity] = left[:, disparity:]
    return left, right


def polygon_label(class_id, x1, y1, x2, y2, width, height):
    """YOLO polygon string for an axis-aligned box (normalized coords)."""
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in pts)
    return f"{class_id} {coords}"


@pytest.fixture
def synthetic_source(tmp_path):
    """Minimal v2-style source root: 2 pairs (train/val), 1 group each."""
    import cv2

    source = tmp_path / "seg_v2"
    images_root = tmp_path / "Rectified_v2"
    (source / "labels" / "train").mkdir(parents=True)
    (source / "labels" / "val").mkdir(parents=True)
    (source / "labels" / "test").mkdir(parents=True)

    pairs = []
    for stem, split, group, seed in (
        ("groupA_000", "train", "groupA", 0),
        ("groupB_000", "val", "groupB", 1),
    ):
        left, right = make_textured_pair(seed=seed)
        pair_dir = images_root / group
        (pair_dir / "left").mkdir(parents=True, exist_ok=True)
        (pair_dir / "right").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pair_dir / "left" / "000.png"), left)
        cv2.imwrite(str(pair_dir / "right" / "000.png"), right)
        # One object per pair; straddles x < disparity so out-of-bounds
        # right-view samples occur inside the crop (occlusion margin).
        (source / "labels" / split / f"{stem}.txt").write_text(
            polygon_label(1, 16, 40, 80, 80, 256, 128), encoding="utf-8"
        )
        pairs.append((stem, split, group, pair_dir))

    with (source / "pair_manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["output_stem", "split", "group_name", "left_path", "right_path"])
        for stem, split, group, pair_dir in pairs:
            writer.writerow(
                [
                    stem,
                    split,
                    group,
                    (pair_dir / "left" / "000.png").as_posix(),
                    (pair_dir / "right" / "000.png").as_posix(),
                ]
            )
    (source / "data.yaml").write_text(
        "path: unused\nnames:\n  0: metal_submarine\n  1: plastic_fish\n",
        encoding="utf-8",
    )
    return source


def small_matcher():
    return StereoMatcher(
        StereoMatcherConfig(
            max_disparity=64,
            scale=0.25,
            block_size=7,
            speckle_window=50,
            speckle_range=16,
        )
    )


class TestBuildFusionDataset:
    def test_generates_npz_manifest_and_expected_counts(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source,
            output_root=output,
            matcher=small_matcher(),
            crop_pad=5,
        )

        assert len(records) == 2
        assert (output / "dataset_manifest.csv").is_file()
        assert (output / "dataset_summary.json").is_file()
        sample = load_fusion_sample(output / records[0].npz_path)
        assert sample.class_id == 1
        assert sample.gray.shape == sample.abs_q.shape
        # Crop window: box x[16,80] y[40,80] + pad 5 on each side
        assert sample.gray.shape == (50, 74)

    def test_manifest_carries_all_provenance(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source,
            output_root=output,
            matcher=small_matcher(),
        )

        with (output / "dataset_manifest.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        row = rows[0]
        for field in (
            "split",
            "class_name",
            "group_name",
            "source_frame",
            "object_index",
            "polar_valid_ratio",
            "stereo_reason",
        ):
            assert field in row
        assert row["class_name"] == "plastic_fish"
        assert row["source_frame"] == records[0].source_frame

    def test_polar_uses_per_pixel_disparity_with_valid_map(self, tmp_path, synthetic_source):
        import cv2

        output = tmp_path / "fusion_v3"
        build_fusion_dataset(
            source_root=synthetic_source,
            output_root=output,
            matcher=small_matcher(),
        )

        # Rebuild the expected polar from the matcher's own dense result.
        left = cv2.imread(
            str(tmp_path / "Rectified_v2" / "groupA" / "left" / "000.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        right = cv2.imread(
            str(tmp_path / "Rectified_v2" / "groupA" / "right" / "000.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        matcher = small_matcher()
        result = matcher.compute(left, right)
        # fillPoly semantics: polygon boundary pixels are inside the mask.
        mask = np.zeros(left.shape, dtype=np.uint8)
        mask[40:81, 16:81] = 1
        expected = compute_polar_features(
            left, right, result.disparity, object_mask=mask, disparity_valid=result.valid
        )

        with (output / "dataset_manifest.csv").open(encoding="utf-8") as fh:
            first_row = next(csv.DictReader(fh))
        sample = load_fusion_sample(output / first_row["npz_path"])

        # Crop window with default pad 10: box x[16,80] y[40,80] -> x[6,90), y[30,90)
        y1, y2, x1, x2 = 30, 90, 6, 90
        np.testing.assert_allclose(
            sample.abs_q, expected.abs_q[y1:y2, x1:x2], rtol=1e-5
        )
        np.testing.assert_array_equal(
            sample.valid, expected.valid_mask[y1:y2, x1:x2].astype(np.uint8)
        )
        # Columns with x < disparity 32 have no right-view correspondence:
        # they must be invalid with zero differential, never saturated at 1.
        assert np.all(sample.valid[:, : 32 - x1] == 0)
        assert np.all(sample.abs_q[:, : 32 - x1] == 0.0)
        assert np.any(sample.valid[:, 32 - x1 :] > 0)

    def test_invalid_stereo_writes_zeroed_sample_with_reason(self, tmp_path, synthetic_source):
        class AlwaysInvalidMatcher:
            def compute(self, left, right):
                from core.stereo_matching import StereoMatchResult

                return StereoMatchResult(
                    disparity=np.zeros(left.shape, np.float32),
                    valid=np.zeros(left.shape, np.uint8),
                    elapsed_s=0.0,
                    valid_ratio=0.0,
                )

            def instance_stats(self, result, mask, instance_id=0):
                from core.stereo_matching import InstanceDepth

                return InstanceDepth(
                    instance_id, 0.0, 0.0, None, False, "no_valid_disparity"
                )

        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source,
            output_root=output,
            matcher=AlwaysInvalidMatcher(),
        )

        assert all(not r.stereo_valid for r in records)
        assert all(r.stereo_reason == "no_valid_disparity" for r in records)
        sample = load_fusion_sample(output / records[0].npz_path)
        assert np.all(sample.signed_q == 0.0)
        assert np.all(sample.abs_q == 0.0)
        assert np.all(sample.valid == 0)
        np.testing.assert_array_equal(sample.quality, np.zeros(QUALITY_VECTOR_LENGTH))
        assert np.any(sample.gray > 0)  # gray crop still written

    def test_clean_required_for_existing_output(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v3"
        build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        with pytest.raises(RuntimeError, match="not empty"):
            build_fusion_dataset(
                source_root=synthetic_source,
                output_root=output,
                matcher=small_matcher(),
            )
        # --clean replaces it
        records = build_fusion_dataset(
            source_root=synthetic_source,
            output_root=output,
            matcher=small_matcher(),
            clean=True,
        )
        assert len(records) == 2

    def test_output_inside_source_rejected(self, tmp_path, synthetic_source):
        with pytest.raises(ValueError, match="outside the source"):
            build_fusion_dataset(
                source_root=synthetic_source,
                output_root=synthetic_source / "fusion_v3",
                matcher=small_matcher(),
            )


class TestBandMatchingMode:
    def test_bands_mode_matches_inference_pipeline(self, tmp_path, synthetic_source):
        """--matching bands must produce the same sample set as the full
        mode (same labels), using build_horizontal_bands + compute_bands."""
        from core.stereo_matching import build_horizontal_bands

        output = tmp_path / "fusion_v4_band"
        records = build_fusion_dataset(
            source_root=synthetic_source,
            output_root=output,
            matcher=small_matcher(),
            clean=True,
            matching="bands",
            band_margin=20,
            full_image_threshold=0.9,
        )
        assert len(records) == 2
        assert records[0].npz_path == "train/plastic_fish/groupA_000_obj000.npz"
        sample = load_fusion_sample(output / records[0].npz_path)
        assert sample.gray.shape == sample.abs_q.shape

        # The band result must come from compute_bands, not a dense pass:
        # recompute with the same band pipeline and compare the disparity
        # restricted to the object band.
        import cv2

        left = cv2.imread(
            str(tmp_path / "Rectified_v2" / "groupA" / "left" / "000.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        right = cv2.imread(
            str(tmp_path / "Rectified_v2" / "groupA" / "right" / "000.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        matcher = small_matcher()
        bands = build_horizontal_bands([(16, 40, 80, 80)], left.shape[0], 20)
        band_result = matcher.compute_bands(left, right, bands, full_image_threshold=0.9)
        mask = np.zeros(left.shape, dtype=np.uint8)
        mask[40:81, 16:81] = 1
        expected = compute_polar_features(
            left, right, band_result.disparity, object_mask=mask, disparity_valid=band_result.valid
        )
        y1, y2, x1, x2 = 30, 90, 6, 90
        np.testing.assert_allclose(
            sample.abs_q, expected.abs_q[y1:y2, x1:x2], rtol=1e-5
        )

    def test_band_recorded_in_generation_params(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v4_band"
        build_fusion_dataset(
            source_root=synthetic_source,
            output_root=output,
            matcher=small_matcher(),
            clean=True,
            matching="bands",
            band_margin=20,
            full_image_threshold=0.9,
        )
        summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
        assert summary["generation"]["matching"] == "bands"
        assert summary["generation"]["band_margin"] == 20
        assert summary["generation"]["full_image_threshold"] == 0.9

    def test_unknown_matching_mode_rejected(self, tmp_path, synthetic_source):
        with pytest.raises(ValueError, match="matching"):
            build_fusion_dataset(
                source_root=synthetic_source,
                output_root=tmp_path / "x",
                matcher=small_matcher(),
                matching="dense",
            )


class TestAudit:
    def make_records(self, records_spec, quality=(0.8, 0.9, 0.9, 0.1), polar_valid_ratio=0.8):
        return [
            FusionSampleRecord(
                sample_name=f"s{i}",
                split=split,
                group_name=group,
                class_id=class_id,
                class_name=f"class_{class_id}",
                source_frame=f"frame_{i}",
                object_index=i,
                stereo_valid=True,
                stereo_reason="ok",
                stereo_valid_ratio=0.9,
                disparity=12.0,
                polar_valid_ratio=polar_valid_ratio,
                quality=quality,
                npz_path=npz_path,
            )
            for i, (split, group, class_id, npz_path) in enumerate(records_spec)
        ]

    def test_audit_passes_on_consistent_dataset(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        audit = audit_fusion_dataset(
            output_root=output,
            records=records,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits={"train": 1, "val": 1, "test": 0},
        )

        assert audit["audit_passed"] is True
        assert audit["decoded_samples"] == 2
        assert audit["decode_failures"] == []
        assert audit["group_split_leakage"] == {}
        assert audit["counts"] == {"total": 2, "train": 1, "val": 1, "test": 0}
        audit_file = tmp_path / "audit" / "fusion_v3_audit.json"
        assert audit_file.is_file()
        assert json.loads(audit_file.read_text(encoding="utf-8"))["audit_passed"] is True
        previews = list((tmp_path / "audit" / "fusion_v3_preview").glob("*_preview.png"))
        assert len(previews) == 2  # PREVIEW_PER_CLASS=2 per class, 1 sample here

    def test_audit_flags_group_leakage(self, tmp_path):
        records = self.make_records(
            [
                ("train", "groupA", 0, "train/class_0/s0.npz"),
                ("val", "groupA", 0, "val/class_0/s1.npz"),
            ]
        )
        # npz files do not exist -> decode failures; leakage still reported
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["group_split_leakage"] == {"groupA": ["train", "val"]}
        assert audit["audit_passed"] is False

    def test_audit_flags_count_mismatch(self, tmp_path):
        records = self.make_records([("train", "g", 0, "x.npz")])
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits={"train": 5, "val": 0, "test": 0},
        )
        assert audit["count_match_expected"] is False
        assert audit["audit_passed"] is False

    def test_audit_flags_saturated_invalid_pixels(self, tmp_path):
        # Hand-craft a sample whose invalid pixels carry nonzero abs_q.
        gray = np.full((4, 4), 100, np.uint8)
        bad_abs = np.full((4, 4), 1.0, np.float32)  # saturated everywhere
        save_fusion_sample(
            tmp_path / "bad.npz",
            gray=gray,
            signed_q=bad_abs.copy(),
            abs_q=bad_abs,
            valid=np.zeros((4, 4), np.uint8),
            quality=build_quality_vector(0, 0, 0, 0),
            class_id=0,
        )
        records = self.make_records(
            [("train", "g", 0, "bad.npz")],
            quality=(0.0, 0.0, 0.0, 0.0),
            polar_valid_ratio=0.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["invalid_pixel_nonzero_failures"] == ["bad.npz"]
        assert audit["audit_passed"] is False

    def test_audit_flags_saturated_left_band(self, tmp_path):
        gray = np.full((4, 8), 100, np.uint8)
        abs_q = np.zeros((4, 8), np.float32)
        abs_q[:, 0] = 1.0  # full saturated column at the left edge
        save_fusion_sample(
            tmp_path / "bad.npz",
            gray=gray,
            signed_q=abs_q.copy(),
            abs_q=abs_q,
            valid=np.ones((4, 8), np.uint8),
            # 4 of 32 valid pixels are saturated -> crop mean 0.125.
            quality=build_quality_vector(1, 1, 1, 0.125),
            class_id=0,
        )
        records = self.make_records(
            [("train", "g", 0, "bad.npz")],
            quality=(1.0, 1.0, 1.0, 0.125),
            polar_valid_ratio=1.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["saturated_left_band_failures"] == ["bad.npz"]
        assert audit["audit_passed"] is False

    def test_audit_tolerates_scattered_saturated_pixels(self, tmp_path):
        # Physically real extreme ratios (single pixels) are not the
        # out-of-bounds band signature and must not fail the audit.
        gray = np.full((4, 8), 100, np.uint8)
        abs_q = np.zeros((4, 8), np.float32)
        abs_q[1, 5] = 1.0  # isolated saturated valid pixel, not a column band
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=gray,
            signed_q=abs_q.copy(),
            abs_q=abs_q,
            valid=np.ones((4, 8), np.uint8),
            # 1 of 32 valid pixels is saturated -> crop mean 0.03125.
            quality=build_quality_vector(1, 1, 1, 0.03125),
            class_id=0,
        )
        records = self.make_records(
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(1.0, 1.0, 1.0, 0.03125),
            polar_valid_ratio=1.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["saturated_left_band_failures"] == []
        assert audit["saturated_valid_pixels"] == 1
        assert audit["audit_passed"] is True

    def test_audit_flags_npz_quality_mismatch(self, tmp_path):
        """The npz quality vector is the training gate input; a divergence
        from the manifest quality must fail the audit."""
        gray = np.full((4, 4), 100, np.uint8)
        signed = np.full((4, 4), 0.5, np.float32)
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=gray,
            signed_q=signed,
            abs_q=np.abs(signed),
            valid=np.ones((4, 4), np.uint8),
            quality=build_quality_vector(0.9, 1.0, 1.0, 0.5),  # npz says 0.9
            class_id=0,
        )
        records = self.make_records(
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(0.1, 1.0, 1.0, 0.5),  # manifest says 0.1
            polar_valid_ratio=0.1,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["npz_quality_failures"]
        assert any("manifest" in f for f in audit["npz_quality_failures"])
        assert audit["audit_passed"] is False

    def test_audit_flags_nonzero_quality_without_valid_pixels(self, tmp_path):
        gray = np.full((4, 4), 100, np.uint8)
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=gray,
            signed_q=np.zeros((4, 4), np.float32),
            abs_q=np.zeros((4, 4), np.float32),
            valid=np.zeros((4, 4), np.uint8),
            quality=build_quality_vector(0.0, 0.0, 0.0, 0.7),  # impossible
            class_id=0,
        )
        records = self.make_records(
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(0.0, 0.0, 0.0, 0.7),
            polar_valid_ratio=0.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert any(
            "zero valid pixels" in f for f in audit["npz_quality_failures"]
        )
        assert audit["audit_passed"] is False

    def test_audit_flags_mean_abs_q_mismatch(self, tmp_path):
        """All valid abs_q = 0.5 but both the npz and the manifest claim
        mean 0.1: the substantive recomputation must fail (the old
        mean<=max bound could not catch this)."""
        gray = np.full((4, 4), 100, np.uint8)
        signed = np.full((4, 4), 0.5, np.float32)
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=gray,
            signed_q=signed,
            abs_q=np.abs(signed),
            valid=np.ones((4, 4), np.uint8),
            quality=build_quality_vector(1.0, 1.0, 1.0, 0.1),
            class_id=0,
        )
        records = self.make_records(
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(1.0, 1.0, 1.0, 0.1),
            polar_valid_ratio=1.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert any("recomputed crop mean" in f for f in audit["npz_quality_failures"])
        assert audit["audit_passed"] is False

    def test_audit_writes_report_into_dataset_root(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        audit_fusion_dataset(
            output_root=output,
            records=records,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert (output / "dataset_audit.json").is_file()
        in_root = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
        assert in_root["audit_passed"] is True

    def test_audit_flags_value_range_violations(self, tmp_path):
        gray = np.full((4, 4), 100, np.uint8)
        # abs_q inconsistent with |signed_q| on valid pixels.
        signed = np.full((4, 4), 0.5, np.float32)
        abs_q = np.full((4, 4), 0.2, np.float32)
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "bad_abs.npz",
            gray=gray,
            signed_q=signed,
            abs_q=abs_q,
            valid=np.ones((4, 4), np.uint8),
            quality=build_quality_vector(1, 1, 1, 0.2),
            class_id=0,
        )
        # valid mask holding a value that is neither 0 nor 1.
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "bad_valid.npz",
            gray=gray,
            signed_q=np.zeros((4, 4), np.float32),
            abs_q=np.zeros((4, 4), np.float32),
            valid=np.full((4, 4), 2, np.uint8),
            quality=build_quality_vector(1, 1, 1, 0),
            class_id=0,
        )
        records = self.make_records(
            [
                ("train", "g", 0, "train/class_0/bad_abs.npz"),
                ("train", "g", 0, "train/class_0/bad_valid.npz"),
            ]
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert len(audit["value_range_failures"]) == 2
        assert any("abs_q" in failure for failure in audit["value_range_failures"])
        assert any("0/1" in failure for failure in audit["value_range_failures"])
        assert audit["audit_passed"] is False

    def test_audit_flags_duplicate_rows_and_orphans(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        # An npz on disk that the manifest does not reference.
        (output / "val" / "plastic_fish" / "orphan.npz").write_bytes(b"junk")
        audit = audit_fusion_dataset(
            output_root=output,
            records=records,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["orphan_npz_files"] == ["val/plastic_fish/orphan.npz"]
        assert audit["audit_passed"] is False

        duplicated = list(records) + [records[0]]
        audit = audit_fusion_dataset(
            output_root=output,
            records=duplicated,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["duplicate_manifest_rows"]
        assert audit["audit_passed"] is False

    def test_audit_flags_class_order_and_quality_mismatch(self, tmp_path, synthetic_source):
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        # class_names passed in an order that contradicts the manifest ids.
        audit = audit_fusion_dataset(
            output_root=output,
            records=records,
            class_names=["plastic_fish", "metal_submarine"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["class_order_mismatches"]
        assert audit["audit_passed"] is False

        # quality[0] (valid_ratio) must equal the manifest polar_valid_ratio.
        swapped = [
            FusionSampleRecord(
                sample_name=r.sample_name,
                split=r.split,
                group_name=r.group_name,
                class_id=r.class_id,
                class_name=r.class_name,
                source_frame=r.source_frame,
                object_index=r.object_index,
                stereo_valid=r.stereo_valid,
                stereo_reason=r.stereo_reason,
                stereo_valid_ratio=r.stereo_valid_ratio,
                disparity=r.disparity,
                polar_valid_ratio=0.123456,
                quality=r.quality,
                npz_path=r.npz_path,
            )
            for r in records
        ]
        audit = audit_fusion_dataset(
            output_root=output,
            records=swapped,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["quality_consistency_failures"]
        assert audit["audit_passed"] is False

    def test_records_roundtrip_through_manifest(self, tmp_path, synthetic_source):
        from scripts.prepare_cls_fusion_dataset import records_from_manifest

        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        rebuilt = records_from_manifest(output)
        assert len(rebuilt) == len(records)
        for original, again in zip(records, rebuilt):
            # The manifest stores 6-decimal text, so compare with tolerance.
            assert original.sample_name == again.sample_name
            assert original.split == again.split
            assert original.npz_path == again.npz_path
            assert original.stereo_valid == again.stereo_valid
            assert original.stereo_reason == again.stereo_reason
            assert original.disparity == pytest.approx(again.disparity, abs=1e-4)
            assert original.polar_valid_ratio == pytest.approx(
                again.polar_valid_ratio, abs=1e-6
            )
            for a, b in zip(original.quality, again.quality):
                assert a == pytest.approx(b, abs=1e-6)


class TestAuditBinding:
    """The audit report must record the manifest/summary digests so the
    training gate can bind the verdict to the current data (issue 1)."""

    def test_audit_records_manifest_and_summary_digests(
        self, tmp_path, synthetic_source
    ):
        import hashlib

        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        audit = audit_fusion_dataset(
            output_root=output,
            records=records,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits={"train": 1, "val": 1, "test": 0},
        )
        assert audit["audit_passed"] is True
        assert audit["manifest_sha256"] == hashlib.sha256(
            (output / "dataset_manifest.csv").read_bytes()
        ).hexdigest()
        assert audit["summary_sha256"] == hashlib.sha256(
            (output / "dataset_summary.json").read_bytes()
        ).hexdigest()

    def test_formal_audit_fails_without_summary(self, tmp_path, synthetic_source):
        """A formal dataset (expected counts given) must keep its generation
        summary; the audit verdict fails when it is missing."""
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        (output / "dataset_summary.json").unlink()
        audit = audit_fusion_dataset(
            output_root=output,
            records=records,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits={"train": 1, "val": 1, "test": 0},
        )
        assert audit["summary_sha256"] == ""
        assert audit["audit_passed"] is False


class TestQualitySemantics:
    """Zero-valid-pixel quality semantics and substantive mean_abs_q
    recomputation (review issues 3/4)."""

    def _dark_pair_sample(self):
        """Real compute_polar_features + quality_vector_from_result on an
        all-black pair: positive disparity, the object maps in bounds, no
        pixel is bright enough to measure."""
        left = np.zeros((4, 8), dtype=np.uint8)
        right = np.zeros((4, 8), dtype=np.uint8)
        disparity = np.full((4, 8), 2.0, dtype=np.float32)
        mask = np.zeros((4, 8), dtype=np.uint8)
        mask[1:3, 3:6] = 1
        result = compute_polar_features(left, right, disparity, object_mask=mask)
        vector, components = quality_vector_from_result(result, mask)
        return result, vector, components

    def test_dark_pair_quality_is_zero_one_zero_zero(self):
        _, vector, components = self._dark_pair_sample()
        assert components["valid_ratio"] == 0.0
        assert components["in_bounds_ratio"] == 1.0
        assert components["brightness_valid_ratio"] == 0.0
        assert components["mean_abs_q"] == 0.0
        np.testing.assert_allclose(vector, [0.0, 1.0, 0.0, 0.0])

    def test_zero_valid_quality_accepts_in_bounds_dark_pair(self, tmp_path):
        """[0, 1, 0, 0] is legitimate for a dark pair and must pass; the
        in_bounds/brightness ratios are not forced to zero."""
        result, vector, _ = self._dark_pair_sample()
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=np.zeros((4, 8), np.uint8),
            signed_q=result.signed_q,
            abs_q=result.abs_q,
            valid=result.valid_mask.astype(np.uint8),
            quality=vector,
            class_id=0,
        )
        records = TestAudit.make_records(
            None,
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(0.0, 1.0, 0.0, 0.0),
            polar_valid_ratio=0.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["npz_quality_failures"] == []
        assert audit["audit_passed"] is True

    def test_zero_valid_quality_rejects_nonzero_valid_ratio(self, tmp_path):
        result, _, _ = self._dark_pair_sample()
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=np.zeros((4, 8), np.uint8),
            signed_q=result.signed_q,
            abs_q=result.abs_q,
            valid=result.valid_mask.astype(np.uint8),
            quality=build_quality_vector(0.5, 1.0, 0.0, 0.0),
            class_id=0,
        )
        records = TestAudit.make_records(
            None,
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(0.5, 1.0, 0.0, 0.0),
            polar_valid_ratio=0.5,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert any("zero valid pixels" in f for f in audit["npz_quality_failures"])
        assert audit["audit_passed"] is False

    def test_zero_valid_quality_rejects_nonzero_mean_abs_q(self, tmp_path):
        result, _, _ = self._dark_pair_sample()
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=np.zeros((4, 8), np.uint8),
            signed_q=result.signed_q,
            abs_q=result.abs_q,
            valid=result.valid_mask.astype(np.uint8),
            quality=build_quality_vector(0.0, 1.0, 0.0, 0.3),
            class_id=0,
        )
        records = TestAudit.make_records(
            None,
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(0.0, 1.0, 0.0, 0.3),
            polar_valid_ratio=0.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert any("zero valid pixels" in f for f in audit["npz_quality_failures"])
        assert audit["audit_passed"] is False

    def test_negative_signed_q_with_matching_mean_passes(self, tmp_path):
        """Legal negative signed_q values: abs_q = |signed_q| and the
        recorded mean must match the crop recomputation."""
        signed = np.full((4, 4), -0.4, np.float32)
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=np.full((4, 4), 100, np.uint8),
            signed_q=signed,
            abs_q=np.abs(signed),
            valid=np.ones((4, 4), np.uint8),
            quality=build_quality_vector(1.0, 1.0, 1.0, 0.4),
            class_id=0,
        )
        records = TestAudit.make_records(
            None,
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(1.0, 1.0, 1.0, 0.4),
            polar_valid_ratio=1.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["npz_quality_failures"] == []
        assert audit["audit_passed"] is True

    def test_mean_abs_q_rounding_boundary_passes(self, tmp_path):
        """float32 mean vs the manifest's 6-decimal text: a value at the
        rounding boundary must still pass (rtol=0, atol=1.5e-6)."""
        value = np.float32(1.0 / 3.0)  # 0.33333334...
        abs_q = np.full((4, 4), value, np.float32)
        mean = float(abs_q.mean())
        quality = build_quality_vector(1.0, 1.0, 1.0, mean)
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=np.full((4, 4), 100, np.uint8),
            signed_q=abs_q.copy(),
            abs_q=abs_q,
            valid=np.ones((4, 4), np.uint8),
            quality=quality,
            class_id=0,
        )
        # The manifest stores 6-decimal text.
        records = TestAudit.make_records(
            None,
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=(1.0, 1.0, 1.0, round(mean, 6)),
            polar_valid_ratio=1.0,
        )
        audit = audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )
        assert audit["npz_quality_failures"] == []
        assert audit["audit_passed"] is True

    def test_generated_dataset_passes_mean_recomputation(
        self, tmp_path, synthetic_source
    ):
        """End-to-end: samples built by the real generator (mask-restricted
        valid, crop window covering the full mask) pass the recomputation."""
        output = tmp_path / "fusion_v3"
        records = build_fusion_dataset(
            source_root=synthetic_source, output_root=output, matcher=small_matcher()
        )
        audit = audit_fusion_dataset(
            output_root=output,
            records=records,
            class_names=["metal_submarine", "plastic_fish"],
            audit_root=tmp_path / "audit",
            expected_splits={"train": 1, "val": 1, "test": 0},
        )
        assert audit["npz_quality_failures"] == []
        assert audit["audit_passed"] is True

    @staticmethod
    def _uniform_valid_quality_audit(tmp_path, quality, mean=0.5):
        """Audit one sample whose valid pixels all carry abs_q == mean, so
        the mean_abs_q recomputation matches and only the ratio invariants
        can fail."""
        signed = np.full((4, 4), mean, np.float32)
        save_fusion_sample(
            tmp_path / "train" / "class_0" / "s0.npz",
            gray=np.full((4, 4), 100, np.uint8),
            signed_q=signed,
            abs_q=np.abs(signed),
            valid=np.ones((4, 4), np.uint8),
            quality=build_quality_vector(*quality),
            class_id=0,
        )
        records = TestAudit.make_records(
            None,
            [("train", "g", 0, "train/class_0/s0.npz")],
            quality=tuple(float(v) for v in quality),
            polar_valid_ratio=float(quality[0]),
        )
        return audit_fusion_dataset(
            output_root=tmp_path,
            records=records,
            class_names=["class_0"],
            audit_root=tmp_path / "audit",
            expected_splits=None,
        )

    def test_nonempty_valid_requires_positive_valid_ratio(self, tmp_path):
        """[0, 1, 1, 0.5] with a non-empty valid mask and a matching mean:
        valid_ratio must be positive whenever the crop has valid pixels."""
        audit = self._uniform_valid_quality_audit(tmp_path, (0.0, 1.0, 1.0, 0.5))
        assert audit["npz_quality_failures"]
        assert any("must be positive" in f for f in audit["npz_quality_failures"])
        assert audit["audit_passed"] is False

    def test_valid_ratio_exceeding_in_bounds_rejected(self, tmp_path):
        """[0.8, 0.2, 0.3, 0.5] with a matching mean: valid pixels are a
        subset of the in-bounds pixels, so valid_ratio can never exceed
        in_bounds_ratio."""
        audit = self._uniform_valid_quality_audit(tmp_path, (0.8, 0.2, 0.3, 0.5))
        assert any("in_bounds_ratio" in f for f in audit["npz_quality_failures"])
        assert audit["audit_passed"] is False

    def test_valid_ratio_exceeding_brightness_rejected(self, tmp_path):
        """[0.5, 0.6, 0.3, 0.5]: in-bounds containment holds, only the
        brightness containment is violated, so this must be caught by the
        dedicated brightness branch."""
        audit = self._uniform_valid_quality_audit(tmp_path, (0.5, 0.6, 0.3, 0.5))
        assert any(
            "brightness_valid_ratio" in f for f in audit["npz_quality_failures"]
        )
        assert audit["audit_passed"] is False

    def test_legal_positive_quality_passes(self, tmp_path):
        audit = self._uniform_valid_quality_audit(tmp_path, (0.5, 0.6, 0.7, 0.5))
        assert audit["npz_quality_failures"] == []
        assert audit["audit_passed"] is True

    def test_valid_ratio_equal_to_upper_bounds_passes(self, tmp_path):
        """Equality with either containment bound is legal."""
        audit = self._uniform_valid_quality_audit(tmp_path, (0.5, 0.5, 0.5, 0.5))
        assert audit["npz_quality_failures"] == []
        assert audit["audit_passed"] is True


class TestCropCutRefusal:
    """A crop window that cuts valid pixels cannot be verified by the
    mean_abs_q audit; generation must refuse it explicitly (issue 4)."""

    def test_crop_window_cutting_valid_pixels_refused(
        self, tmp_path, synthetic_source, monkeypatch
    ):
        from scripts.make_polar_dataset import CropWindow
        import scripts.prepare_cls_fusion_dataset as pcf

        def tiny_window(polygon, width, height, padding):
            return CropWindow(x1=0, y1=0, x2=2, y2=2)

        monkeypatch.setattr(pcf, "make_crop_window", tiny_window)
        with pytest.raises(ValueError, match="cuts valid polar pixels"):
            build_fusion_dataset(
                source_root=synthetic_source,
                output_root=tmp_path / "fusion_v3",
                matcher=small_matcher(),
            )
