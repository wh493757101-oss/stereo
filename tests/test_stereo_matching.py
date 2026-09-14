import os
import unittest.mock

import cv2
import numpy as np
import pytest

from core.stereo_matching import (
    HorizontalBand,
    StereoMatchResult,
    StereoMatcher,
    StereoMatcherConfig,
    build_horizontal_bands,
    compute_disparity_ncc,
    compute_instance_depths,
    disparity_to_depth,
    merge_horizontal_bands,
    robust_disparity,
    to_gray_u8,
)

# Documented tolerance: SGBM runs at scale 0.25 with 1/16-pixel subpixel
# resolution, so on synthetic textured rectified pairs the recovered
# full-resolution disparity is expected to match the ground truth to well
# under one pixel; +/-2 full-resolution pixels is the accepted bound.
DISPARITY_TOLERANCE_PX = 2.0


def make_textured_pair(
    width: int = 1024,
    height: int = 512,
    disparity: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic rectified pair with texture at the matching scale.

    right(x, y) = left(x + disparity, y); columns beyond the shifted content
    are filled with independent texture. Texture is generated at low
    resolution and upscaled so it survives the matcher's INTER_AREA
    downscaling.
    """
    rng = np.random.default_rng(seed)
    base = cv2.resize(
        rng.integers(0, 256, size=(height // 4, width // 4), dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    left = np.clip(base, 0, 255).astype(np.uint8)
    right = np.clip(
        cv2.resize(
            rng.integers(0, 256, size=(height // 4, width // 4), dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        ),
        0,
        255,
    ).astype(np.uint8)
    if disparity > 0:
        right[:, : width - disparity] = left[:, disparity:]
    return left, right


def default_matcher() -> StereoMatcher:
    return StereoMatcher(StereoMatcherConfig())


class TestConfigValidation:
    def test_rejects_non_sgbm_matcher(self):
        with pytest.raises(ValueError, match="matcher"):
            StereoMatcherConfig(matcher="ncc")

    def test_rejects_nonpositive_max_disparity(self):
        with pytest.raises(ValueError, match="max_disparity"):
            StereoMatcherConfig(max_disparity=0)

    def test_rejects_bad_scale(self):
        with pytest.raises(ValueError, match="scale"):
            StereoMatcherConfig(scale=0.0)
        with pytest.raises(ValueError, match="scale"):
            StereoMatcherConfig(scale=1.5)

    def test_rejects_unknown_sgbm_mode(self):
        with pytest.raises(ValueError, match="SGBM mode"):
            StereoMatcherConfig(mode="unknown")

    def test_rejects_even_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            StereoMatcherConfig(block_size=8)

    def test_rejects_bad_min_valid_ratio(self):
        with pytest.raises(ValueError, match="min_valid_ratio"):
            StereoMatcherConfig(min_valid_ratio=1.5)

    def test_num_disparities_rounds_up_to_multiple_of_16(self):
        matcher = StereoMatcher(StereoMatcherConfig(max_disparity=768, scale=0.25))
        assert matcher.num_disparities == 192
        matcher = StereoMatcher(StereoMatcherConfig(max_disparity=700, scale=0.25))
        assert matcher.num_disparities == 176
        assert matcher.num_disparities % 16 == 0
        assert matcher.num_disparities * 4 >= 700


class TestImageHandling:
    def test_three_channel_input_matches_grayscale(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        result_gray = default_matcher().compute(left, right)
        result_bgr = default_matcher().compute(left_bgr, right_bgr)

        assert result_gray.disparity.shape == left.shape
        gray_valid = result_gray.disparity[result_gray.valid > 0]
        bgr_valid = result_bgr.disparity[result_bgr.valid > 0]
        assert np.median(gray_valid) == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)
        assert np.median(bgr_valid) == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)

    def test_float_input_is_accepted(self):
        left, right = make_textured_pair(512, 256, disparity=64)
        result = default_matcher().compute(left.astype(np.float32), right.astype(np.float32))
        valid = result.disparity[result.valid > 0]
        assert np.median(valid) == pytest.approx(64, abs=DISPARITY_TOLERANCE_PX)

    def test_single_channel_hw1_input_matches_grayscale(self):
        # Ultralytics patches cv2.imread so grayscale reads return (H, W, 1)
        left, right = make_textured_pair(512, 256, disparity=128)
        result = default_matcher().compute(left[..., None], right[..., None])
        valid = result.disparity[result.valid > 0]
        assert result.disparity.shape == left.shape
        assert np.median(valid) == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)

    def test_unsupported_channel_count_raises(self):
        left = np.zeros((64, 64, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="channel"):
            default_matcher().compute(left, left)

    def test_shape_mismatch_raises(self):
        left, _ = make_textured_pair(256, 128)
        _, right = make_textured_pair(256, 64)
        with pytest.raises(ValueError, match="mismatch"):
            default_matcher().compute(left, right)


class TestToGrayU8:
    def test_2d_uint8_passthrough_is_contiguous(self):
        img = np.asfortranarray(np.full((8, 6), 7, dtype=np.uint8))
        out = to_gray_u8(img)
        assert out.shape == (8, 6)
        assert out.dtype == np.uint8
        assert out.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(out, img)

    def test_hw1_single_channel_is_squeezed(self):
        img = np.arange(48, dtype=np.uint8).reshape(8, 6, 1)
        out = to_gray_u8(img)
        assert out.shape == (8, 6)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, img[:, :, 0])

    def test_hw1_float_single_channel_is_scaled_to_uint8_range(self):
        img = np.full((4, 4, 1), 300.0, dtype=np.float32)
        out = to_gray_u8(img)
        assert out.shape == (4, 4)
        assert out.dtype == np.uint8
        assert np.all(out == 255)

    def test_three_channel_uses_luma_weights(self):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # blue plane
        out = to_gray_u8(bgr)
        assert out.shape == (4, 4)
        # BGR2GRAY weights blue at ~0.114
        assert np.all(out == np.rint(255 * 0.114).astype(np.uint8))

    def test_four_channel_bgra_accepted(self):
        bgra = np.zeros((4, 4, 4), dtype=np.uint8)
        bgra[:, :, 2] = 255  # red plane
        out = to_gray_u8(bgra)
        assert out.shape == (4, 4)
        assert out.dtype == np.uint8
        assert np.all(out == np.rint(255 * 0.299).astype(np.uint8))

    def test_unsupported_shapes_raise(self):
        with pytest.raises(ValueError, match="channel"):
            to_gray_u8(np.zeros((4, 4, 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="channel"):
            to_gray_u8(np.zeros((4, 4, 5), dtype=np.uint8))
        with pytest.raises(ValueError, match="channel"):
            to_gray_u8(np.zeros((4,), dtype=np.uint8))


class TestComputeOutputs:
    def test_output_shapes_and_dtypes(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        result = default_matcher().compute(left, right)

        assert isinstance(result, StereoMatchResult)
        assert result.disparity.shape == left.shape
        assert result.disparity.dtype == np.float32
        assert result.valid.shape == left.shape
        assert result.valid.dtype == np.uint8
        assert result.elapsed_s >= 0.0
        assert 0.0 <= result.valid_ratio <= 1.0

    def test_invalid_pixels_are_zero(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        result = default_matcher().compute(left, right)

        invalid = result.valid == 0
        assert np.all(result.disparity[invalid] == 0.0)
        if np.any(result.valid > 0):
            assert np.all(result.disparity[result.valid > 0] > 0.0)

    def test_zero_disparity_pair_yields_no_valid_depth_evidence(self):
        left, right = make_textured_pair(512, 256, disparity=0)
        result = default_matcher().compute(left, right)

        # Zero disparity is on the SGBM search boundary and is treated as
        # invalid: no positive disparities survive, so the depth evidence is
        # "no detectable depth".
        assert result.valid_ratio == pytest.approx(0.0, abs=0.01)
        assert np.all(result.disparity == 0.0)

    @pytest.mark.parametrize("disparity", [64, 384, 720])
    def test_positive_disparity_recovered_in_visible_roi(self, disparity):
        left, right = make_textured_pair(1024, 512, disparity=disparity)
        result = default_matcher().compute(left, right)

        # Physically visible ROI: left pixels whose right correspondence
        # exists (x >= disparity), excluding the outer margin.
        x0, x1 = disparity + 16, left.shape[1] - 16
        roi_valid = result.valid[:, x0:x1] > 0
        roi_values = result.disparity[:, x0:x1][roi_valid]

        assert roi_valid.mean() > 0.5
        assert np.median(roi_values) == pytest.approx(
            disparity, abs=DISPARITY_TOLERANCE_PX
        )
        assert np.abs(roi_values - disparity).mean() <= DISPARITY_TOLERANCE_PX

    def test_too_small_image_returns_all_invalid(self):
        tiny = np.full((16, 20), 128, dtype=np.uint8)
        result = default_matcher().compute(tiny, tiny)

        assert result.disparity.shape == tiny.shape
        assert np.all(result.valid == 0)
        assert result.valid_ratio == 0.0


class TestDepthConversion:
    def test_depth_units_meters(self):
        depth = disparity_to_depth(disparity=100.0, baseline=0.1, focal_length=1000.0)
        assert depth == pytest.approx(1.0)

    def test_invalid_disparity_returns_sentinel(self):
        assert disparity_to_depth(0.0, 0.1, 1000.0) == -1.0
        assert disparity_to_depth(-5.0, 0.1, 1000.0) == -1.0
        assert disparity_to_depth(float("nan"), 0.1, 1000.0) == -1.0
        assert disparity_to_depth(float("inf"), 0.1, 1000.0) == -1.0

    def test_no_rounding_of_core_value(self):
        depth = disparity_to_depth(3.0, 0.09890970798524992, 3643.5231322766995)
        assert depth == pytest.approx(0.09890970798524992 * 3643.5231322766995 / 3.0)


def craft_result(
    disparity_values: dict[tuple[int, int], float],
    shape: tuple[int, int],
) -> StereoMatchResult:
    disparity = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=np.uint8)
    for (y, x), value in disparity_values.items():
        disparity[y, x] = value
        valid[y, x] = 1
    return StereoMatchResult(
        disparity=disparity, valid=valid, elapsed_s=0.0, valid_ratio=0.5
    )


class TestInstanceStatistics:
    def make_mask(self, shape=(8, 8)) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        mask[2:6, 2:6] = 255
        return mask

    def test_median_with_mad_rejection(self):
        mask = self.make_mask()
        values = {}
        for i, y in enumerate(range(2, 6)):
            for j, x in enumerate(range(2, 6)):
                v = 40.0 if (i * 4 + j) < 15 else 400.0
                values[(y, x)] = v
        result = craft_result(values, mask.shape)

        stats = default_matcher().instance_stats(result, mask, instance_id=3)

        assert stats.instance_id == 3
        assert stats.disparity == pytest.approx(40.0)
        assert stats.valid is True
        assert stats.reason == "ok"

    def test_valid_ratio_within_mask(self):
        mask = self.make_mask()
        values = {}
        for i, y in enumerate(range(2, 6)):
            for j, x in enumerate(range(2, 6)):
                if i * 4 + j < 8:  # half the mask pixels valid
                    values[(y, x)] = 20.0
        result = craft_result(values, mask.shape)

        strict_matcher = StereoMatcher(StereoMatcherConfig(min_valid_ratio=0.6))
        stats = strict_matcher.instance_stats(result, mask)

        assert stats.valid_ratio == pytest.approx(0.5)
        assert stats.disparity == pytest.approx(20.0)
        assert stats.valid is False
        assert stats.reason == "low_valid_ratio"

    def test_empty_mask(self):
        result = craft_result({}, (4, 4))
        stats = default_matcher().instance_stats(result, np.zeros((4, 4), np.uint8))

        assert stats.reason == "empty_mask"
        assert stats.valid is False
        assert stats.depth_m is None

    def test_mask_without_valid_pixels(self):
        mask = self.make_mask((8, 8))
        result = craft_result({}, (8, 8))
        stats = default_matcher().instance_stats(result, mask)

        assert stats.reason == "no_valid_disparity"
        assert stats.valid_ratio == 0.0

    def test_instance_depths_computes_depth_once_for_all_masks(self):
        matcher = default_matcher()
        left, right = make_textured_pair(512, 256, disparity=128)
        result = matcher.compute(left, right)

        masks = [
            make_object_mask(512, 256, 250, 128, 60),
            make_object_mask(512, 256, 400, 100, 80),
            make_object_mask(512, 256, 300, 200, 50),
        ]
        depths = matcher.instance_depths(result, masks, baseline=0.1, focal_length=1000.0)

        assert len(depths) == 3
        for i, depth in enumerate(depths):
            assert depth.instance_id == i
            assert depth.valid
            assert depth.reason == "ok"
            expected = 0.1 * 1000.0 / depth.disparity
            assert depth.depth_m == pytest.approx(expected)
        # All three use the same precomputed dense result: matching ran once.
        assert result.elapsed_s >= 0.0

    def test_object_disparity_map_is_constant_inside_mask(self):
        matcher = default_matcher()
        left, right = make_textured_pair(512, 256, disparity=128)
        result = matcher.compute(left, right)
        mask = make_object_mask(512, 256, 256, 128, 80)

        filled = matcher.object_disparity_map(result, mask)

        assert filled.shape == mask.shape
        assert filled.dtype == np.float32
        assert np.all(filled[mask == 0] == 0.0)
        inside = filled[mask > 0]
        stats = matcher.instance_stats(result, mask)
        assert np.all(inside == stats.disparity)


def make_object_mask(
    width: int, height: int, cx: int, cy: int, radius: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


class TestCompatibilityWrappers:
    def test_compute_disparity_ncc_delegates_to_sgbm(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        mask = make_object_mask(512, 256, 300, 128, 60)

        disparity, confidence, disp_map = compute_disparity_ncc(
            left, right, mask, max_disp=768, window_size=7
        )

        assert disparity == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)
        assert 0.0 <= confidence <= 1.0
        assert disp_map is not None
        assert disp_map.shape == left.shape

    def test_compute_disparity_ncc_empty_mask_returns_none_map(self):
        left, right = make_textured_pair(128, 64, disparity=32)
        disparity, confidence, disp_map = compute_disparity_ncc(
            left, right, np.zeros((64, 128), np.uint8)
        )
        assert disparity == 0.0
        assert confidence == 0.0
        assert disp_map is None

    def test_compute_instance_depths_one_dense_compute_for_many_masks(self):
        calls = []
        original = StereoMatcher.compute

        def spy(self, left, right):
            calls.append(1)
            return original(self, left, right)

        left, right = make_textured_pair(512, 256, disparity=128)
        left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        masks = [
            make_object_mask(512, 256, 300, 128, 60),
            make_object_mask(512, 256, 430, 200, 50),
        ]

        import core.stereo_matching as sm

        with unittest.mock.patch.object(sm.StereoMatcher, "compute", spy):
            results = compute_instance_depths(
                left_bgr,
                right_bgr,
                masks,
                baseline=0.1,
                focal_length=1000.0,
                max_disp=768,
                window_size=7,
            )

        assert len(calls) == 1
        assert len(results) == 2
        for i, entry in enumerate(results):
            assert entry["instance_id"] == i
            assert entry["disparity"] == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)
            assert entry["depth"] == pytest.approx(0.1 * 1000.0 / 128.0, rel=0.05)
            assert entry["confidence"] > 0.0


@pytest.mark.skipif(
    os.environ.get("STEREO_BENCHMARK") != "1",
    reason="manual benchmark: run with STEREO_BENCHMARK=1; no wall-clock assertion in CI",
)
def test_benchmark_stereo_1280x1024(capsys: pytest.CaptureFixture) -> None:
    left, right = make_textured_pair(1280, 1024, disparity=384, seed=7)
    matcher = StereoMatcher(StereoMatcherConfig())

    result = matcher.compute(left, right)

    roi_valid = result.valid[:, 400:1264] > 0
    median = float(np.median(result.disparity[:, 400:1264][roi_valid]))
    capsys.readouterr()
    print(
        f"1280x1024 SGBM: elapsed={result.elapsed_s:.3f}s, "
        f"valid_ratio={result.valid_ratio:.3f}, roi_median={median:.1f}px"
    )
    assert result.elapsed_s > 0.0
    assert median == pytest.approx(384, abs=DISPARITY_TOLERANCE_PX)


def test_robust_disparity_uses_median_not_mean():
    disparities = np.array([8, 8, 9, 9, 40], dtype=np.float32)

    disparity = robust_disparity(disparities)

    assert disparity == 8.5
    assert disparity != float(disparities.mean())


def test_robust_disparity_returns_zero_for_empty_values():
    assert robust_disparity(np.array([], dtype=np.float32)) == 0.0


class TestHorizontalBands:
    def test_band_rejects_invalid_rows(self):
        with pytest.raises(ValueError, match="invalid band rows"):
            HorizontalBand(5, 5)
        with pytest.raises(ValueError, match="invalid band rows"):
            HorizontalBand(-1, 4)

    def test_band_height(self):
        assert HorizontalBand(3, 10).height == 7

    def test_merge_overlapping_bands(self):
        merged = merge_horizontal_bands(
            [HorizontalBand(0, 50), HorizontalBand(40, 100)]
        )
        assert merged == [HorizontalBand(0, 100)]

    def test_merge_touching_bands(self):
        # Adjacent bands (no gap) are merged to avoid double edge effects.
        merged = merge_horizontal_bands([HorizontalBand(10, 50), HorizontalBand(50, 90)])
        assert merged == [HorizontalBand(10, 90)]

    def test_merge_disjoint_bands_stay_separate_and_sorted(self):
        merged = merge_horizontal_bands(
            [HorizontalBand(60, 100), HorizontalBand(0, 20), HorizontalBand(30, 50)]
        )
        assert merged == [
            HorizontalBand(0, 20),
            HorizontalBand(30, 50),
            HorizontalBand(60, 100),
        ]

    def test_merge_contained_band_is_absorbed(self):
        merged = merge_horizontal_bands(
            [HorizontalBand(0, 100), HorizontalBand(10, 20)]
        )
        assert merged == [HorizontalBand(0, 100)]

    def test_merge_empty_input(self):
        assert merge_horizontal_bands([]) == []

    def test_build_bands_expands_bboxes_with_margin(self):
        bands = build_horizontal_bands(
            [(100, 100, 200, 180)], image_height=512, vertical_margin=10
        )
        assert bands == [HorizontalBand(90, 190)]

    def test_build_bands_clips_to_image(self):
        bands = build_horizontal_bands(
            [(0, 0, 64, 100), (0, 500, 64, 512)], image_height=512, vertical_margin=20
        )
        assert bands[0].y1 == 0
        assert bands[-1].y2 == 512

    def test_build_bands_merges_vertical_overlap(self):
        bands = build_horizontal_bands(
            [(0, 100, 200, 180), (50, 150, 250, 300)],
            image_height=512,
            vertical_margin=10,
        )
        assert bands == [HorizontalBand(90, 310)]

    def test_build_bands_ignores_x_extent(self):
        # Bands always span the full width; only rows matter.
        left = build_horizontal_bands([(0, 100, 10, 180)], 512, 10)
        right = build_horizontal_bands([(400, 100, 500, 180)], 512, 10)
        assert left == right

    def test_build_bands_no_bboxes_gives_no_bands(self):
        assert build_horizontal_bands([], image_height=512) == []

    def test_build_bands_rejects_malformed_bbox(self):
        with pytest.raises(ValueError, match="bbox"):
            build_horizontal_bands([(1, 2, 3)], image_height=512)


class TestComputeBands:
    def test_empty_bands_skip_sgbm(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        result = default_matcher().compute_bands(left, right, [])

        assert result.disparity.shape == left.shape
        assert np.all(result.disparity == 0.0)
        assert np.all(result.valid == 0)
        assert result.elapsed_s == 0.0
        assert result.valid_ratio == 0.0

    def test_rows_outside_bands_are_invalid(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        bands = [HorizontalBand(100, 180)]
        result = default_matcher().compute_bands(left, right, bands)

        assert np.all(result.valid[:100] == 0)
        assert np.all(result.disparity[:100] == 0.0)
        assert np.all(result.valid[180:] == 0)
        assert np.all(result.disparity[180:] == 0.0)
        inside = result.valid[100:180] > 0
        assert inside.mean() > 0.3
        values = result.disparity[100:180][inside]
        assert np.median(values) == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)

    def test_valid_ratio_is_over_full_image(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        bands = [HorizontalBand(0, 64)]  # quarter of the rows
        result = default_matcher().compute_bands(left, right, bands)

        expected = float(result.valid.sum()) / (256 * 512)
        assert result.valid_ratio == pytest.approx(expected)
        assert result.valid_ratio < 0.25

    def test_bands_are_effectively_disjoint_no_double_coverage(self):
        # Overlapping input bands must not run SGBM twice on shared rows.
        left, right = make_textured_pair(512, 256, disparity=128)
        calls = []
        original = StereoMatcher._match_pair

        def spy(self, left_img, right_img):
            calls.append(left_img.shape)
            return original(self, left_img, right_img)

        import core.stereo_matching as sm

        with unittest.mock.patch.object(sm.StereoMatcher, "_match_pair", spy):
            result = default_matcher().compute_bands(
                left, right, [HorizontalBand(0, 100), HorizontalBand(50, 160)]
            )

        assert len(calls) == 1  # merged into one band before matching
        assert result.valid_ratio > 0.0

    def test_near_full_coverage_falls_back_to_single_full_pass(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        calls = []
        original = StereoMatcher.compute

        def spy(self, left_img, right_img):
            calls.append(1)
            return original(self, left_img, right_img)

        import core.stereo_matching as sm

        with unittest.mock.patch.object(sm.StereoMatcher, "compute", spy):
            result = default_matcher().compute_bands(
                left, right, [HorizontalBand(0, 240)]  # 240/256 = 0.94 >= 0.9
            )

        assert len(calls) == 1
        valid = result.disparity[result.valid > 0]
        assert np.median(valid) == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)

    def test_multiple_disjoint_bands_each_matched(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        bands = [HorizontalBand(0, 64), HorizontalBand(192, 256)]
        result = default_matcher().compute_bands(left, right, bands)

        for y1, y2 in ((0, 64), (192, 256)):
            inside = result.valid[y1:y2] > 0
            assert inside.mean() > 0.3
            values = result.disparity[y1:y2][inside]
            assert np.median(values) == pytest.approx(
                128, abs=DISPARITY_TOLERANCE_PX
            )
        assert np.all(result.valid[64:192] == 0)

    def test_bands_clipped_to_image_rows(self):
        left, right = make_textured_pair(512, 256, disparity=128)
        result = default_matcher().compute_bands(
            left, right, [HorizontalBand(0, 300), HorizontalBand(100, 999)]
        )
        # Merged and clipped: one band covering the whole image -> full pass.
        valid = result.disparity[result.valid > 0]
        assert np.median(valid) == pytest.approx(128, abs=DISPARITY_TOLERANCE_PX)
