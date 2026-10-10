"""Deterministic reliability diagnostics for the polarimetric S region (2A).

Everything here runs on synthetic arrays and synthetic V5-style npz
samples in the per-test scratch directory: no real dataset, no model
forward, no stereo matching. The tests pin the frozen definitions
(erode3x3 interior, the eight S statistics) and the contract that the
proposed region rule can never change the missing state.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.material_mask_dataset import (
    MaskFusionSampleArrays,
    load_mask_sample,
    save_mask_sample,
)
from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    build_cache_binding,
    build_polar_stats,
    authoritative_sample_index,
    verify_cache_arrays,
    verify_cache_binding,
    verify_cache_manifest,
)
from core.polar_region_quality import (
    BAND_DEFINITION,
    DENOM_LEVELS,
    INTERIOR_DEFINITION,
    PERTURBATION_DELTAS,
    Q_HEAVY,
    REGION_RULE_ID,
    RELIABLE_REGION_DEFINITION_VERSION,
    build_reliable_region_cache,
    evaluate_region_rule,
    interior_and_band_masks,
    recover_denominator,
    region_polar_stats,
    sample_region_diagnostics,
    shift_content,
    verify_sample_correspondence,
)

from .test_cache_admission import make_protocol


# ---------------------------------------------------------------------------
# Synthetic sample helpers


def make_sample(gray, signed_q, abs_q, valid, instance_mask):
    """One synthetic V5-style sample (quality/class_id are unused here)."""
    gray = np.asarray(gray, dtype=np.uint8)
    h, w = gray.shape
    return MaskFusionSampleArrays(
        gray=gray,
        signed_q=np.asarray(signed_q, dtype=np.float32),
        abs_q=np.asarray(abs_q, dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
        quality=np.zeros(4, dtype=np.float32),
        class_id=0,
        instance_mask=np.asarray(instance_mask, dtype=np.uint8),
    )


def ramp_sample():
    """5x5 sample: full mask, q increasing to the right, all valid."""
    gray = np.full((5, 5), 128, dtype=np.uint8)
    signed = np.tile(np.linspace(-0.4, 0.4, 5, dtype=np.float32), (5, 1))
    valid = np.ones((5, 5), dtype=bool)
    mask = np.ones((5, 5), dtype=np.uint8)
    return make_sample(gray, signed, np.abs(signed), valid, mask)


# ---------------------------------------------------------------------------
# Interior / band: the frozen erode3x3 split


class TestInteriorAndBand:
    def test_interior_is_single_3x3_erosion(self):
        # A 5x5 foreground block centered in a 7x7 image: the background
        # ring lets erosion behave morphologically (the frozen cv2 call
        # never erodes at the IMAGE border — see the next test).
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1
        interior, band = interior_and_band_masks(mask)
        assert interior.sum() == 9  # 3x3 center survives one erosion
        assert band.sum() == 16  # the ring
        assert (interior.astype(np.uint8) | band.astype(np.uint8)).sum() == 25
        assert not (interior & band).any()

    def test_image_border_pixels_are_not_eroded(self):
        # The frozen cv2.erode call uses the +infinity default border
        # value, so a mask touching the image border keeps its outermost
        # ring in the "interior". This quirk is part of the frozen 2B
        # definition and is pinned here — not silently changed.
        mask = np.ones((5, 5), dtype=np.uint8)
        interior, band = interior_and_band_masks(mask)
        assert interior.sum() == 25
        assert band.sum() == 0

    def test_single_pixel_mask_has_empty_interior(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 2] = 1
        interior, band = interior_and_band_masks(mask)
        assert not interior.any()
        assert np.array_equal(band, mask > 0)

    def test_band_plus_interior_is_mask_and_disjoint(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            mask = (rng.random((9, 7)) > 0.5).astype(np.uint8)
            interior, band = interior_and_band_masks(mask)
            union = interior.astype(np.uint8) | band.astype(np.uint8)
            assert np.array_equal(union > 0, mask > 0)
            assert not (interior & band).any()

    def test_matches_the_frozen_part2b_definition(self):
        from scripts.prepare_material_probe import (
            interior_and_band_masks as part2b_masks,
        )

        rng = np.random.default_rng(4)
        for _ in range(5):
            mask = (rng.random((11, 6)) > 0.6).astype(np.uint8)
            mine = interior_and_band_masks(mask)
            frozen = part2b_masks(mask)
            assert np.array_equal(mine[0], frozen[0] > 0)
            assert np.array_equal(mine[1], frozen[1] > 0)

    def test_definitions_are_the_frozen_strings(self):
        assert INTERIOR_DEFINITION == "erode3x3_instance_mask_v1"
        assert BAND_DEFINITION == "instance_mask_minus_interior_v1"


# ---------------------------------------------------------------------------
# region_polar_stats: the frozen eight statistics on an explicit region


class TestRegionPolarStats:
    def test_matches_build_polar_stats_on_the_original_region(self):
        rng = np.random.default_rng(5)
        for index in range(6):
            signed = rng.uniform(-1, 1, size=(7, 6)).astype(np.float32)
            valid = rng.random((7, 6)) > 0.3
            mask = (rng.random((7, 6)) > 0.3).astype(np.uint8)
            sample = make_sample(
                np.full((7, 6), 100, np.uint8), signed, np.abs(signed),
                valid, mask)
            region = (valid & (mask > 0))
            mine, nonempty = region_polar_stats(signed, np.abs(signed),
                                                region)
            reference, reference_has = build_polar_stats(
                signed_q=signed, abs_q=np.abs(signed), valid=valid,
                instance_mask=mask)
            assert nonempty == reference_has
            assert np.array_equal(mine, reference)

    def test_hand_computed_values(self):
        signed = np.zeros((3, 3), dtype=np.float32)
        signed[0, 0] = 0.1
        signed[0, 1] = 0.3
        region = np.zeros((3, 3), dtype=bool)
        region[0, 0] = region[0, 1] = True
        values, nonempty = region_polar_stats(
            signed, np.abs(signed), region)
        assert nonempty
        assert values[0] == pytest.approx(0.2, abs=1e-6)  # mean
        assert values[1] == pytest.approx(0.1, abs=1e-6)  # std ddof0
        assert values[4] == pytest.approx(0.2, abs=1e-6)  # p50
        assert values[7] == pytest.approx(0.2, abs=1e-6)  # abs mean

    def test_empty_region_is_explicit_nan_miss(self):
        values, nonempty = region_polar_stats(
            np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32),
            np.zeros((2, 2), dtype=bool))
        assert not nonempty
        assert values.shape == (len(POLAR_STAT_KEYS),)
        assert np.isnan(values).all()

    def test_non_finite_inside_region_refused(self):
        signed = np.ones((2, 2), dtype=np.float32)
        signed[0, 0] = np.nan
        region = np.ones((2, 2), dtype=bool)
        with pytest.raises(ValueError, match="non-finite"):
            region_polar_stats(signed, np.abs(signed), region)

    def test_non_finite_outside_region_tolerated(self):
        signed = np.ones((2, 2), dtype=np.float32)
        signed[0, 0] = np.nan  # outside the region: not measured
        region = np.zeros((2, 2), dtype=bool)
        region[1, 1] = True
        values, nonempty = region_polar_stats(signed, np.abs(signed), region)
        assert nonempty
        assert np.isfinite(values).all()


# ---------------------------------------------------------------------------
# Denominator recovery: algebra on the saved arrays, not re-measurement


class TestRecoverDenominator:
    @staticmethod
    def _saved_q(left, right_warped):
        """signed_q exactly as core.polar_compute saved it (eps=1e-6)."""
        left_f = np.float64(left)
        right_f = np.float64(right_warped)
        return np.float32(
            (left_f - right_f) / (left_f + right_f + 1e-6))

    def test_recovers_the_generation_time_denominator(self):
        left = np.array([[200, 80]], dtype=np.uint8)
        right_warped = np.array([[150, 70]], dtype=np.float64)
        q = self._saved_q(left, right_warped)
        valid = np.array([[True, True]])
        denom, known = recover_denominator(left, q, valid)
        assert known.all()
        assert denom[0, 0] == pytest.approx(350.0, abs=1e-2)
        assert denom[0, 1] == pytest.approx(150.0, abs=1e-2)

    def test_left_black_pixel_sum_is_only_bounded(self):
        # L=0 with a bright right sample is valid and q == -1 exactly;
        # the sum is then not recoverable from the saved arrays.
        left = np.array([[0, 40]], dtype=np.uint8)
        right_warped = np.array([[50, 40]], dtype=np.float64)
        q = self._saved_q(left, right_warped)
        valid = np.array([[True, True]])
        assert q[0, 0] == np.float32(-1.0)
        denom, known = recover_denominator(left, q, valid)
        assert not known[0, 0]
        assert np.isnan(denom[0, 0])
        assert known[0, 1]
        assert denom[0, 1] == pytest.approx(80.0, abs=1e-2)

    def test_invalid_pixels_are_unknown(self):
        left = np.array([[200, 80]], dtype=np.uint8)
        q = self._saved_q(left, np.array([[150.0, 70.0]]))
        valid = np.array([[False, True]])
        denom, known = recover_denominator(left, q, valid)
        assert not known[0, 0]
        assert np.isnan(denom[0, 0])
        assert known[0, 1]

    def test_right_black_sample_sums_to_left(self):
        # R' = 0 -> q = +1 and the sum equals L exactly.
        left = np.array([[90]], dtype=np.uint8)
        q = self._saved_q(left, np.array([[0.0]]))
        denom, known = recover_denominator(left, q, np.ones((1, 1), bool))
        assert known[0, 0]
        assert denom[0, 0] == pytest.approx(90.0, abs=1e-2)


# ---------------------------------------------------------------------------
# shift_content: the +-1 px sampling perturbation


class TestShiftContent:
    def test_zero_shift_is_identity(self):
        sample = ramp_sample()
        out = shift_content(sample.gray, sample.signed_q, sample.abs_q,
                            sample.valid, (0, 0))
        for a, b in zip((sample.gray, sample.signed_q, sample.abs_q,
                         sample.valid), out):
            assert np.array_equal(a, b)

    def test_shift_moves_content_and_vacated_edge_is_invalid(self):
        sample = ramp_sample()
        gray2, signed2, abs2, valid2 = shift_content(
            sample.gray, sample.signed_q, sample.abs_q, sample.valid,
            (1, 0))
        assert np.array_equal(gray2[1:], sample.gray[:-1])
        assert not valid2[0].any()  # vacated row cannot be measured
        assert valid2[1:].all()
        assert np.array_equal(signed2[1:], sample.signed_q[:-1])

    def test_shift_is_deterministic(self):
        sample = ramp_sample()
        first = shift_content(sample.gray, sample.signed_q, sample.abs_q,
                              sample.valid, (0, -1))
        second = shift_content(sample.gray, sample.signed_q, sample.abs_q,
                               sample.valid, (0, -1))
        for a, b in zip(first, second):
            assert np.array_equal(a, b)

    def test_uniform_content_stats_survive_a_shift(self):
        # Uniform q: a 1-px shift only removes edge pixels from the
        # region; the surviving values are unchanged, so the statistics
        # must be exactly equal for the interior region.
        gray = np.full((6, 6), 100, np.uint8)
        signed = np.full((6, 6), 0.25, np.float32)
        valid = np.ones((6, 6), dtype=bool)
        mask = np.ones((6, 6), dtype=np.uint8)
        base, _ = region_polar_stats(signed, np.abs(signed),
                                     np.ones((6, 6), dtype=bool))
        _, signed2, _, valid2 = shift_content(
            gray, signed, np.abs(signed), valid, (0, 1))
        shifted, nonempty = region_polar_stats(
            signed2, np.abs(signed2), (mask > 0) & valid2)
        assert nonempty
        assert np.array_equal(base, shifted)


# ---------------------------------------------------------------------------
# Per-sample diagnostics


class TestSampleRegionDiagnostics:
    def test_counts_and_valid_ratios(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1  # away from the image border
        valid = np.ones((7, 7), dtype=bool)
        valid[1, 1:6] = False  # the top band ring is not measurable
        sample = make_sample(np.full((7, 7), 100, np.uint8),
                             np.full((7, 7), 0.2, np.float32),
                             np.full((7, 7), 0.2, np.float32),
                             valid, mask)
        report = sample_region_diagnostics(sample)
        assert report["counts"]["mask_pixels"] == 25
        assert report["counts"]["interior_pixels"] == 9
        assert report["counts"]["band_pixels"] == 16
        assert report["counts"]["region_pixels"] == 20
        assert report["counts"]["interior_valid_pixels"] == 9
        assert report["counts"]["band_valid_pixels"] == 11
        assert report["ratios"]["interior_valid_ratio"] == pytest.approx(1.0)
        assert report["ratios"]["band_valid_ratio"] == \
            pytest.approx(11 / 16)

    def test_valid_and_q_outside_the_gt_mask_are_ignored(self):
        # polar_valid (or a nonzero q) outside the annotated instance is
        # never a measurement region: the GT mask gates everything.
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        valid = np.ones((4, 4), dtype=bool)  # "valid" everywhere
        signed = np.full((4, 4), 0.9, np.float32)  # nonzero everywhere
        signed[1:3, 1:3] = 0.1
        sample = make_sample(np.full((4, 4), 100, np.uint8), signed,
                             np.abs(signed), valid, mask)
        report = sample_region_diagnostics(sample)
        assert report["counts"]["region_pixels"] == 4
        assert report["original_stats"][0] == pytest.approx(0.1, abs=1e-6)

    def test_degenerate_fractions_reported(self):
        mask = np.ones((1, 5), dtype=np.uint8)
        gray = np.array([[255, 250, 100, 0, 50]], dtype=np.uint8)
        right = np.array([100.0, 100.0, 100.0, 50.0, 50.0])
        signed = np.zeros((1, 5), dtype=np.float32)
        for col in range(5):
            left_f = np.float64(gray[0, col])
            signed[0, col] = np.float32(
                (left_f - right[col]) / (left_f + right[col] + 1e-6))
        valid = np.ones((1, 5), dtype=bool)
        sample = make_sample(gray, signed, np.abs(signed), valid, mask)
        report = sample_region_diagnostics(sample)
        deg = report["degenerate"]["region"]
        assert deg["gray_eq_255"] == pytest.approx(0.2)
        assert deg["gray_ge_250"] == pytest.approx(0.4)
        assert deg["gray_eq_0"] == pytest.approx(0.2)
        assert deg["abs_q_eq_1"] == pytest.approx(0.2)  # the L=0 pixel

    def test_low_denominator_fraction_uses_recovered_sum(self):
        # L + R' = 18 < 20 for the first pixel, 40 for the second.
        gray = np.array([[10, 20]], dtype=np.uint8)
        right = np.array([8.0, 20.0])
        signed = np.zeros((1, 2), dtype=np.float32)
        for col in range(2):
            left_f = np.float64(gray[0, col])
            signed[0, col] = np.float32(
                (left_f - right[col]) / (left_f + right[col] + 1e-6))
        mask = np.ones((1, 2), dtype=np.uint8)
        valid = np.ones((1, 2), dtype=bool)
        sample = make_sample(gray, signed, np.abs(signed), valid, mask)
        report = sample_region_diagnostics(sample)
        low = report["degenerate"]["region"]["denom_below"]
        assert low["20.0"] == pytest.approx(0.5)
        assert low["30.0"] == pytest.approx(0.5)
        assert low["50.0"] == pytest.approx(1.0)

    def test_band_heavy_q_share_exceeds_pixel_share(self):
        # Extreme |q| lives only in the band ring: its share of |q|>=0.5
        # pixels must exceed the band's pixel share.
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1
        signed = np.full((7, 7), 0.1, np.float32)
        signed[1, 1:6] = 0.9  # only the top band row is extreme
        valid = np.ones((7, 7), dtype=bool)
        sample = make_sample(np.full((7, 7), 100, np.uint8), signed,
                             np.abs(signed), valid, mask)
        report = sample_region_diagnostics(sample)
        assert report["band_influence"]["region_pixel_share_band"] == \
            pytest.approx(16 / 25)
        assert report["band_influence"]["heavy_q_share_band"] == \
            pytest.approx(1.0)

    def test_truncation_flag(self):
        mask = np.ones((4, 4), dtype=np.uint8)
        touching = make_sample(np.full((4, 4), 100, np.uint8),
                               np.zeros((4, 4), np.float32),
                               np.zeros((4, 4), np.float32),
                               np.ones((4, 4), bool), mask)
        assert sample_region_diagnostics(touching)["truncation"][
            "mask_touches_crop_border"] is True
        inner = mask.copy()
        inner[0, :] = 0
        inner[-1, :] = 0
        inner[:, 0] = 0
        inner[:, -1] = 0
        inside = make_sample(np.full((4, 4), 100, np.uint8),
                             np.zeros((4, 4), np.float32),
                             np.zeros((4, 4), np.float32),
                             np.ones((4, 4), bool), inner)
        assert sample_region_diagnostics(inside)["truncation"][
            "mask_touches_crop_border"] is False

    def test_perturbation_deltas_reported_per_key(self):
        sample = ramp_sample()
        report = sample_region_diagnostics(sample)
        pert = report["perturbation"]
        assert set(pert["per_delta"].keys()) == {
            "(1, 0)", "(-1, 0)", "(0, 1)", "(0, -1)"}
        for delta_key, entry in pert["per_delta"].items():
            assert set(entry["abs_delta"].keys()) == set(POLAR_STAT_KEYS)
        # A right-moving ramp sampled through a fixed mask: horizontal
        # shifts move values, vertical shifts only drop edge columns.
        horizontal = max(
            pert["per_delta"][key]["abs_delta"]["signed_q_mean"]
            for key in ("(0, 1)", "(0, -1)"))
        assert horizontal > 0.0

    def test_perturbation_of_uniform_content_is_exactly_zero(self):
        gray = np.full((6, 6), 100, np.uint8)
        signed = np.full((6, 6), 0.3, np.float32)
        sample = make_sample(gray, signed, np.abs(signed),
                             np.ones((6, 6), bool), np.ones((6, 6), np.uint8))
        report = sample_region_diagnostics(sample)
        for entry in report["perturbation"]["per_delta"].values():
            for key, value in entry["abs_delta"].items():
                assert value == pytest.approx(0.0, abs=1e-7), key

    def test_diagnostics_are_deterministic(self):
        sample = ramp_sample()
        first = sample_region_diagnostics(sample)
        second = sample_region_diagnostics(sample)
        assert json.dumps(first, sort_keys=True, allow_nan=False) == \
            json.dumps(second, sort_keys=True, allow_nan=False)


# ---------------------------------------------------------------------------
# The single rule: interior(erode3x3) with original-region fallback


class TestEvaluateRegionRule:
    def test_nonempty_interior_uses_interior_stats(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        valid = np.ones((5, 5), dtype=bool)
        sample = make_sample(np.full((5, 5), 100, np.uint8),
                             ramp_sample().signed_q,
                             np.abs(ramp_sample().signed_q), valid, mask)
        decision = evaluate_region_rule(sample)
        assert decision["rule_id"] == REGION_RULE_ID
        assert decision["fallback_used"] is False
        assert decision["chosen_region"] == "interior_erode3x3"
        interior, _ = interior_and_band_masks(mask)
        expected, _ = region_polar_stats(
            sample.signed_q, sample.abs_q, interior & valid)
        assert np.array_equal(decision["stats"], expected)

    def test_empty_interior_falls_back_to_the_original_region(self):
        # A one-pixel-wide mask has no interior: the rule must fall back
        # to the ORIGINAL region and produce exactly the original stats.
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = 1
        signed = np.full((4, 4), 0.4, np.float32)
        valid = np.zeros((4, 4), dtype=bool)
        valid[1, 1] = True
        sample = make_sample(np.full((4, 4), 100, np.uint8), signed,
                             np.abs(signed), valid, mask)
        decision = evaluate_region_rule(sample)
        assert decision["fallback_used"] is True
        assert decision["chosen_region"] == "fallback_original"
        original, original_has = build_polar_stats(
            signed_q=sample.signed_q, abs_q=sample.abs_q,
            valid=sample.valid, instance_mask=mask)
        assert decision["has_valid_s"] == original_has is True
        assert np.array_equal(decision["stats"], original)

    def test_missing_state_cannot_change_under_the_rule(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
            mask = (rng.random((h, w)) > 0.5).astype(np.uint8)
            if not mask.any():
                mask[0, 0] = 1
            valid = rng.random((h, w)) > 0.4
            signed = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
            sample = make_sample(np.full((h, w), 100, np.uint8), signed,
                                 np.abs(signed), valid, mask)
            decision = evaluate_region_rule(sample)
            _, original_has = build_polar_stats(
                signed_q=sample.signed_q, abs_q=sample.abs_q,
                valid=sample.valid, instance_mask=mask)
            assert decision["has_valid_s"] == original_has

    def test_all_invalid_sample_stays_invalid(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        sample = make_sample(np.full((5, 5), 100, np.uint8),
                             np.zeros((5, 5), np.float32),
                             np.zeros((5, 5), np.float32),
                             np.zeros((5, 5), bool), mask)
        decision = evaluate_region_rule(sample)
        assert decision["has_valid_s"] is False
        assert decision["stats"] is None
        assert decision["fallback_used"] is False


# ---------------------------------------------------------------------------
# Correspondence: saved crop vs the rectified left frame


class TestSampleCorrespondence:
    @staticmethod
    def _write_left_png(path: Path, left_gray: np.ndarray) -> None:
        assert cv2.imwrite(str(path), left_gray)

    def test_matching_crop_and_mask_pass(self, scratch_dir):
        left_gray = np.arange(40, dtype=np.uint8).reshape(5, 8) + 100
        png = scratch_dir / "left.png"
        self._write_left_png(png, left_gray)
        y1, y2, x1, x2 = 1, 4, 2, 6
        crop = left_gray[y1:y2, x1:x2]
        mask = np.ones(crop.shape, dtype=np.uint8)
        mask[0, 0] = 0
        sample = make_sample(crop, np.zeros(crop.shape, np.float32),
                             np.zeros(crop.shape, np.float32),
                             np.ones(crop.shape, bool), mask)
        reloaded = load_mask_sample(save_mask_sample(
            scratch_dir / "s.npz", gray=sample.gray,
            signed_q=sample.signed_q, abs_q=sample.abs_q,
            valid=(mask > 0).astype(np.uint8), quality=sample.quality,
            class_id=0, instance_mask=mask))
        report = verify_sample_correspondence(
            reloaded, left_gray, (y1, x1, y2, x2),
            expected_mask_pixels=int(mask.sum()))
        assert report["gray_identical"] is True
        assert report["mask_pixels_match"] is True
        assert report["crop_within_image"] is True

    def test_gray_mismatch_is_refused(self, scratch_dir):
        left_gray = np.full((5, 8), 7, dtype=np.uint8)
        png = scratch_dir / "left.png"
        self._write_left_png(png, left_gray)
        crop = left_gray[1:4, 2:6].copy()
        crop[0, 0] = 200
        sample = make_sample(crop, np.zeros(crop.shape, np.float32),
                             np.zeros(crop.shape, np.float32),
                             np.ones(crop.shape, bool),
                             np.ones(crop.shape, np.uint8))
        with pytest.raises(ValueError, match="gray"):
            verify_sample_correspondence(
                sample, left_gray, (1, 2, 4, 6), expected_mask_pixels=12)

    def test_mask_pixel_count_mismatch_is_refused(self, scratch_dir):
        left_gray = np.full((5, 8), 7, dtype=np.uint8)
        sample = make_sample(left_gray[1:4, 2:6],
                             np.zeros((3, 4), np.float32),
                             np.zeros((3, 4), np.float32),
                             np.ones((3, 4), bool),
                             np.ones((3, 4), np.uint8))
        with pytest.raises(ValueError, match="mask"):
            verify_sample_correspondence(
                sample, left_gray, (1, 2, 4, 6), expected_mask_pixels=99)

    def test_crop_out_of_image_is_refused(self, scratch_dir):
        left_gray = np.full((5, 8), 7, dtype=np.uint8)
        sample = make_sample(np.zeros((3, 4), np.uint8),
                             np.zeros((3, 4), np.float32),
                             np.zeros((3, 4), np.float32),
                             np.ones((3, 4), bool),
                             np.ones((3, 4), np.uint8))
        with pytest.raises(ValueError, match="crop"):
            verify_sample_correspondence(
                sample, left_gray, (3, 6, 9, 12), expected_mask_pixels=12)


# ---------------------------------------------------------------------------
# Reliable-region cache machinery (synthetic chain; the real build waits
# for the user's approval of the rule)


def build_region_chain(scratch_dir: Path) -> dict:
    """A self-consistent synthetic chain with REAL npz samples.

    The nine synthetic samples carry controlled regions: interior-only
    validity, band-only validity and fully invalid measurements. The
    source cache stores the ORIGINAL-region S statistics (the frozen
    definition), so the reliable-region builder can be compared against
    it.
    """
    from .test_cache_admission import G_DIM

    protocol = make_protocol()
    authority = authoritative_sample_index(protocol)
    names = sorted(authority)

    data_root = scratch_dir / "v5"
    rng = np.random.default_rng(11)
    samples = {}
    for index, name in enumerate(names):
        h = w = 8
        mask = np.ones((h, w), dtype=np.uint8)
        if index % 3 == 1:
            mask[:2, :] = 0  # interior exists (6x8 -> eroded core)
        if index % 3 == 2:
            mask[:] = 0
            mask[3, 3] = 1  # single pixel: no interior at all
        valid = np.zeros((h, w), dtype=bool)
        if index % 3 == 0:
            valid[:] = True
        elif index % 3 == 1:
            valid[2:, :] = True  # interior valid, band partially
        else:
            valid[3, 3] = True  # band-only validity
        signed = rng.uniform(-0.8, 0.8, size=(h, w)).astype(np.float32)
        gray = rng.integers(40, 220, size=(h, w)).astype(np.uint8)
        path = save_mask_sample(
            data_root / authority[name]["identity"]["npz_path"],
            gray=gray, signed_q=signed,
            abs_q=np.abs(signed), valid=valid.astype(np.uint8),
            quality=np.zeros(4, dtype=np.float32),
            class_id=authority[name]["identity"]["class_id"],
            instance_mask=mask)
        samples[name] = load_mask_sample(path)

    feature_dir = scratch_dir / "source_features"
    feature_dir.mkdir(parents=True)
    n = len(names)
    s_rows = np.full((n, len(POLAR_STAT_KEYS)), np.nan, dtype=np.float32)
    s_valid = np.zeros(n, dtype=bool)
    for row, name in enumerate(names):
        sample = samples[name]
        values, has = build_polar_stats(
            signed_q=sample.signed_q, abs_q=sample.abs_q,
            valid=sample.valid, instance_mask=sample.instance_mask)
        s_rows[row] = values
        s_valid[row] = has
    g = rng.normal(size=(n, G_DIM)).astype(np.float32)
    c = rng.normal(size=(n, len(CONTROL_FEATURE_KEYS))).astype(np.float32)
    c[:, CONTROL_FEATURE_KEYS.index("s_region_nonempty_flag")] = \
        s_valid.astype(np.float32)
    class_ids = np.asarray([authority[name]["identity"]["class_id"]
                            for name in names], dtype=np.int64)
    manifest = {
        "samples": [
            {
                "sample_name": name,
                "npz_path": authority[name]["identity"]["npz_path"],
                "split": authority[name]["identity"]["split"],
                "group": authority[name]["identity"]["group"],
                "source_frame": authority[name]["identity"]["source_frame"],
                "object_index": authority[name]["identity"]["object_index"],
                "class_id": authority[name]["identity"]["class_id"],
                "class_name": authority[name]["identity"]["class_name"],
                "physical_object_id": "unknown",
                "has_valid_s": bool(s_valid[row]),
                "fold_roles": dict(authority[name]["roles"]),
            }
            for row, name in enumerate(names)
        ],
        "count": n,
    }
    np.savez_compressed(feature_dir / "feature_cache.npz",
                        g_features=g, c_features=c, s_features=s_rows,
                        s_valid=s_valid, class_id=class_ids)
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    definition = {
        "feature_definition_version": "existing_data_probe_v1",
        "data_fingerprint": protocol["data"]["fingerprint"],
        "protocol_sha256": protocol["protocol_sha256"],
        "official_mit_b0": dict(
            protocol["config"]["baselines"]["official_mit_b0"]),
        "imgsz": protocol["config"]["features"]["imgsz"],
        "preprocessing_version":
            protocol["config"]["features"]["preprocessing_version"],
        "slot_G": {"dim": G_DIM, "definition": "test G"},
        "slot_C": {"keys": list(CONTROL_FEATURE_KEYS)},
        "slot_S": {"keys": list(POLAR_STAT_KEYS)},
    }
    (feature_dir / "feature_definition.json").write_text(
        json.dumps(definition, indent=2), encoding="utf-8")
    build_cache_binding(feature_dir, protocol)
    return {
        "protocol": protocol,
        "authority": authority,
        "names": names,
        "data_root": data_root,
        "feature_dir": feature_dir,
        "samples": samples,
        "arrays": {"g_features": g, "c_features": c, "s_features": s_rows,
                   "s_valid": s_valid, "class_id": class_ids},
        "manifest": manifest,
    }


class TestReliableRegionCache:
    def test_build_produces_a_verifiable_new_cache(self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        output = scratch_dir / "reliable_features"
        report = build_reliable_region_cache(
            source_feature_dir=chain["feature_dir"],
            data_root=chain["data_root"],
            protocol=chain["protocol"],
            output_dir=output)
        assert report["sample_count"] == len(chain["names"])
        # the frozen verification chain accepts the new directory's
        # manifest and arrays against the same protocol
        new_cache = dict(np.load(output / "feature_cache.npz"))
        new_manifest = json.loads(
            (output / "feature_manifest.json").read_text(encoding="utf-8"))
        new_definition = json.loads(
            (output / "feature_definition.json").read_text(encoding="utf-8"))
        samples = verify_cache_manifest(new_manifest, chain["authority"],
                                        chain["protocol"])
        verify_cache_arrays(new_cache, samples, new_definition,
                            chain["protocol"])
        verify_cache_binding(output, chain["protocol"])

    def test_g_and_c_are_unchanged_and_missing_state_is_identical(
            self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        output = scratch_dir / "reliable_features"
        build_reliable_region_cache(
            source_feature_dir=chain["feature_dir"],
            data_root=chain["data_root"],
            protocol=chain["protocol"],
            output_dir=output)
        new_cache = dict(np.load(output / "feature_cache.npz"))
        assert np.array_equal(new_cache["g_features"],
                              chain["arrays"]["g_features"])
        assert np.array_equal(new_cache["c_features"],
                              chain["arrays"]["c_features"])
        assert np.array_equal(new_cache["s_valid"],
                              chain["arrays"]["s_valid"])
        assert np.array_equal(new_cache["class_id"],
                              chain["arrays"]["class_id"])
        invalid = ~chain["arrays"]["s_valid"]
        assert np.isnan(new_cache["s_features"][invalid]).all()
        assert np.isfinite(new_cache["s_features"][~invalid]).all()

    def test_s_values_recomputed_on_the_rule_region(self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        output = scratch_dir / "reliable_features"
        build_reliable_region_cache(
            source_feature_dir=chain["feature_dir"],
            data_root=chain["data_root"],
            protocol=chain["protocol"],
            output_dir=output)
        new_cache = dict(np.load(output / "feature_cache.npz"))
        for row, name in enumerate(chain["names"]):
            decision = evaluate_region_rule(chain["samples"][name])
            if decision["has_valid_s"]:
                assert np.array_equal(new_cache["s_features"][row],
                                      decision["stats"])
            else:
                assert np.isnan(new_cache["s_features"][row]).all()

    def test_manifest_keeps_identity_and_adds_region_records(
            self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        output = scratch_dir / "reliable_features"
        build_reliable_region_cache(
            source_feature_dir=chain["feature_dir"],
            data_root=chain["data_root"],
            protocol=chain["protocol"],
            output_dir=output)
        new_manifest = json.loads(
            (output / "feature_manifest.json").read_text(encoding="utf-8"))
        old_rows = {row["sample_name"]: row
                    for row in chain["manifest"]["samples"]}
        assert [row["sample_name"] for row in new_manifest["samples"]] == \
            [row["sample_name"] for row in chain["manifest"]["samples"]]
        for row in new_manifest["samples"]:
            old = old_rows[row["sample_name"]]
            for key in ("npz_path", "split", "group", "class_id",
                        "fold_roles", "has_valid_s"):
                assert row[key] == old[key], key
            assert row["s_region_rule_id"] == REGION_RULE_ID
            assert "s_region_fallback" in row
            assert "s_region_pixels" in row

    def test_new_definition_declares_the_rule_and_new_version(
            self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        output = scratch_dir / "reliable_features"
        build_reliable_region_cache(
            source_feature_dir=chain["feature_dir"],
            data_root=chain["data_root"],
            protocol=chain["protocol"],
            output_dir=output)
        definition = json.loads(
            (output / "feature_definition.json").read_text(encoding="utf-8"))
        assert definition["feature_definition_version"] == \
            RELIABLE_REGION_DEFINITION_VERSION
        assert definition["region"]["rule_id"] == REGION_RULE_ID
        assert definition["region"]["interior_definition"] == \
            INTERIOR_DEFINITION
        assert definition["region"]["fallback"] == "original_region"
        assert list(definition["slot_S"]["keys"]) == list(POLAR_STAT_KEYS)
        assert list(definition["slot_C"]["keys"]) == list(CONTROL_FEATURE_KEYS)

    def test_source_cache_stays_byte_identical(self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        source = chain["feature_dir"]
        before = {
            name: (source / name).read_bytes()
            for name in ("feature_cache.npz", "feature_manifest.json",
                         "feature_definition.json", "cache_binding.json")
        }
        build_reliable_region_cache(
            source_feature_dir=source,
            data_root=chain["data_root"],
            protocol=chain["protocol"],
            output_dir=scratch_dir / "reliable_features")
        for name, payload in before.items():
            assert (source / name).read_bytes() == payload

    def test_existing_output_dir_is_refused(self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        output = scratch_dir / "reliable_features"
        output.mkdir()
        with pytest.raises(ValueError, match="already exists"):
            build_reliable_region_cache(
                source_feature_dir=chain["feature_dir"],
                data_root=chain["data_root"],
                protocol=chain["protocol"],
                output_dir=output)

    def test_tampered_source_cache_is_refused(self, scratch_dir):
        chain = build_region_chain(scratch_dir)
        tampered = scratch_dir / "tampered_features"
        tampered.mkdir()
        for name in ("feature_cache.npz", "feature_manifest.json",
                     "feature_definition.json", "cache_binding.json"):
            (tampered / name).write_bytes(
                (chain["feature_dir"] / name).read_bytes())
        cache = dict(np.load(tampered / "feature_cache.npz"))
        cache["s_features"][0, 0] += 1.0
        np.savez_compressed(tampered / "feature_cache.npz", **cache)
        with pytest.raises(ValueError, match="binding"):
            build_reliable_region_cache(
                source_feature_dir=tampered,
                data_root=chain["data_root"],
                protocol=chain["protocol"],
                output_dir=scratch_dir / "reliable_from_tampered")

    def test_denom_levels_and_rule_id_are_frozen(self):
        assert REGION_RULE_ID == "interior_erode3x3_fallback_original_v1"
        assert PERTURBATION_DELTAS == ((1, 0), (-1, 0), (0, 1), (0, -1))
        assert Q_HEAVY == 0.5
        assert DENOM_LEVELS == (20.0, 30.0, 50.0, 100.0)
