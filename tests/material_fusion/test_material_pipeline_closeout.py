"""4B closeout-round tests: fallback summary, strict alignment, zero overlap.

Covers the remaining review gaps:

- the fallback summary must count the real ``used_fallback`` field over
  exactly ``layer=B_model_a_predictions, condition=routed``, per
  variant/seed, with the reason counts summing to the numerator and no
  pure/A2 records leaking in;
- the alignment verdict must require full coverage of the expected sample
  set with every required check present, must reject duplicates, extras,
  unknown statuses, inner failures and non-exact float arrays, and must
  distinguish FAILED from PARTIAL;
- the unmatched-GT record must not invent a "best prediction" when every
  overlap is zero, and must distinguish "no predictions" from "no positive
  overlap";
- the corrected performance stage name describes what was actually timed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.material_fusion_dataset import CLASS_NAMES


# ---------------------------------------------------------------------------
# Fallback summary


def _instance_row(variant="gray_polar", seed=2026, frame="frame_a",
                  pred_index=0, condition="routed", layer="B_model_a_predictions",
                  used_fallback=False, reason="", valid=True):
    return {
        "layer": layer,
        "condition": condition,
        "variant": variant,
        "seed": seed,
        "frame_id": frame,
        "pred_instance_id": pred_index,
        "used_fallback": used_fallback,
        "fallback_reason": reason,
        "pred_class": 0 if valid else None,
        "error": None if valid else "invalid_instance_structure",
    }


class TestFallbackSummary:
    def _mixed_rows(self):
        rows = [
            _instance_row(variant="gray_polar", seed=2026, frame="f1", pred_index=0),
            _instance_row(variant="gray_polar", seed=2026, frame="f1", pred_index=1,
                          used_fallback=True, reason="no_valid_disparity"),
            _instance_row(variant="gray_polar", seed=2026, frame="f2", pred_index=0,
                          used_fallback=True, reason="low_valid_ratio"),
            _instance_row(variant="gray_polar", seed=2026, frame="f2", pred_index=1,
                          used_fallback=True, reason="no_valid_disparity"),
            _instance_row(variant="gray_polar", seed=2026, frame="f3", pred_index=0,
                          valid=False),
            _instance_row(variant="gray", seed=2026, frame="f1", pred_index=0),
            _instance_row(variant="gray", seed=2026, frame="f2", pred_index=0),
            # must be excluded: A2 layer, pure condition, another seed
            _instance_row(variant="gray_polar", seed=2026, frame="f1", pred_index=0,
                          layer="A2_gt_controlled", used_fallback=True,
                          reason="no_valid_disparity"),
            _instance_row(variant="gray_polar", seed=2026, frame="f1", pred_index=0,
                          condition="pure", used_fallback=True,
                          reason="no_valid_disparity"),
            _instance_row(variant="gray_polar", seed=2027, frame="f1", pred_index=0,
                          used_fallback=True, reason="no_valid_disparity"),
        ]
        return rows

    def test_filters_exactly_and_counts_reasons(self):
        from scripts.eval_material_pipeline import summarise_fallback

        summary = summarise_fallback(
            self._mixed_rows(),
            candidates=[("gray", 2026), ("gray_polar", 2026), ("gray_polar", 2027)],
        )
        entry = summary["gray_polar_s2026"]
        assert entry["instances"] == 5
        assert entry["used_fallback_instances"] == 3
        assert entry["unclassified_instances"] == 1
        assert entry["reasons"] == {"no_valid_disparity": 2, "low_valid_ratio": 1}
        assert sum(entry["reasons"].values()) == entry["used_fallback_instances"]
        assert entry["fallback_ratio"] == pytest.approx(3 / 5)
        assert summary["gray_s2026"]["used_fallback_instances"] == 0
        assert summary["gray_polar_s2027"]["instances"] == 1

    def test_e1_must_not_use_the_fallback(self):
        from scripts.eval_material_pipeline import summarise_fallback

        rows = [
            _instance_row(variant="gray", seed=2026, frame="f1", pred_index=0,
                          used_fallback=True, reason="no_valid_disparity"),
        ]
        with pytest.raises(ValueError, match="E1|gray"):
            summarise_fallback(rows, candidates=[("gray", 2026)])

    def test_duplicate_keys_are_rejected(self):
        from scripts.eval_material_pipeline import summarise_fallback

        row = _instance_row(frame="f1", pred_index=0)
        with pytest.raises(ValueError, match="duplicate"):
            summarise_fallback([row, dict(row)],
                               candidates=[("gray_polar", 2026)])

    def test_missing_fields_and_unknown_identity_are_rejected(self):
        from scripts.eval_material_pipeline import summarise_fallback

        broken = _instance_row()
        broken.pop("used_fallback")
        with pytest.raises(ValueError, match="used_fallback"):
            summarise_fallback([broken], candidates=[("gray_polar", 2026)])
        with pytest.raises(ValueError, match="not among the candidates|identity"):
            summarise_fallback([_instance_row(variant="gray_valid", seed=2028)],
                               candidates=[("gray_polar", 2026)])

    def test_reason_mismatch_is_rejected(self):
        from scripts.eval_material_pipeline import summarise_fallback

        rows = [
            _instance_row(used_fallback=True, reason="no_valid_disparity"),
            _instance_row(frame="f2", pred_index=0, used_fallback=False,
                          reason="no_valid_disparity"),
        ]
        with pytest.raises(ValueError, match="reason"):
            summarise_fallback(rows, candidates=[("gray_polar", 2026)])

    def test_historical_records_recompute_all_twelve_candidates(self):
        source = Path(
            "analysis/runs/material_runtime_v2_mask/4b_fix_20260928T114135Z/"
            "quality/instances.jsonl"
        )
        if not source.is_file():
            pytest.skip("historical evidence is not present")
        from scripts.eval_material_pipeline import summarise_fallback

        rows = [json.loads(line) for line in source.read_bytes()
                .decode("utf-8").splitlines() if line.strip()]
        candidates = [
            (variant, seed)
            for variant in ("gray", "gray_gray_valid", "gray_valid", "gray_polar")
            for seed in (2026, 2027, 2028)
        ]
        summary = summarise_fallback(rows, candidates=candidates)
        assert set(summary) == {
            f"{variant}_s{seed}" for variant, seed in candidates
        }
        for key, entry in summary.items():
            assert entry["instances"] == 380, key
            assert sum(entry["reasons"].values()) == entry["used_fallback_instances"]
            if key.startswith("gray_s"):
                assert entry["used_fallback_instances"] == 0, key
        assert summary["gray_polar_s2026"]["used_fallback_instances"] == 3
        assert summary["gray_polar_s2026"]["reasons"] == {
            "no_valid_disparity": 2, "low_valid_ratio": 1
        }


# ---------------------------------------------------------------------------
# Strict alignment verdict


def _alignment_row(frame="frame_a", object_index=0, status="PASSED",
                   checks=None, sample=True):
    row = {
        "frame_id": frame,
        "object_index": object_index,
        "status": status,
        "npz_path": "x.npz" if sample else None,
        "checks": checks if checks is not None else _good_checks(),
    }
    return row


def _good_checks(exact=True):
    check = {"status": "PASSED" if exact else "FAILED", "exact_equal": exact,
             "max_abs_err": 0.0}
    checks = {name: dict(check) for name in (
        "gray", "instance_mask", "polar_valid", "signed_q", "abs_q", "quality",
        "gray_rgb", "gray_mask", "gray_logits",
        "gray_gray_valid_rgb", "gray_gray_valid_aux", "gray_gray_valid_mask",
        "gray_gray_valid_logits",
        "gray_valid_rgb", "gray_valid_aux", "gray_valid_mask",
        "gray_valid_logits",
        "gray_polar_rgb", "gray_polar_aux", "gray_polar_mask",
        "gray_polar_logits",
    )}
    return checks


class TestStrictAlignmentVerdict:
    def test_empty_rows_are_not_a_pass(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        verdict = evaluate_alignment([], expected_keys=[("frame_a", 0)])
        assert verdict["status"] == "PARTIAL"
        assert verdict["coverage"]["missing"] == ["frame_a|0"]

    def test_missing_and_extra_and_duplicate_samples(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        expected = [("frame_a", 0), ("frame_b", 0)]
        rows = [_alignment_row("frame_a", 0)]
        verdict = evaluate_alignment(rows, expected_keys=expected)
        assert verdict["status"] == "PARTIAL"
        assert verdict["coverage"]["missing"] == ["frame_b|0"]

        extra = [_alignment_row("frame_a", 0), _alignment_row("frame_b", 0),
                 _alignment_row("frame_c", 0)]
        verdict = evaluate_alignment(extra, expected_keys=expected)
        assert verdict["status"] == "FAILED"
        assert verdict["coverage"]["extra"] == ["frame_c|0"]

        duplicated = [_alignment_row("frame_a", 0), _alignment_row("frame_a", 0),
                      _alignment_row("frame_b", 0)]
        verdict = evaluate_alignment(duplicated, expected_keys=expected)
        assert verdict["status"] == "FAILED"
        assert verdict["coverage"]["duplicates"] == ["frame_a|0"]

    def test_no_npz_rows_are_partial_and_listed(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        rows = [_alignment_row("frame_a", 0, status="NO_NPZ", sample=False,
                               checks={}),
                _alignment_row("frame_b", 0)]
        verdict = evaluate_alignment(
            rows, expected_keys=[("frame_a", 0), ("frame_b", 0)]
        )
        assert verdict["status"] == "PARTIAL"
        assert verdict["coverage"]["no_npz"] == ["frame_a|0"]

    def test_unavailable_required_check_is_partial(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        checks = _good_checks()
        checks["gray"] = {"status": "UNAVAILABLE", "exact_equal": False,
                          "max_abs_err": None}
        rows = [_alignment_row(checks=checks)]
        verdict = evaluate_alignment(rows, expected_keys=[("frame_a", 0)])
        assert verdict["status"] == "PARTIAL"
        assert verdict["unfinished"]

    def test_unknown_status_and_empty_checks_are_not_a_pass(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        rows = [_alignment_row(status="WHATEVER")]
        verdict = evaluate_alignment(rows, expected_keys=[("frame_a", 0)])
        assert verdict["status"] == "FAILED"
        rows = [_alignment_row(checks={})]
        verdict = evaluate_alignment(rows, expected_keys=[("frame_a", 0)])
        assert verdict["status"] == "FAILED"

    def test_outer_pass_with_inner_failure_is_failed(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        checks = _good_checks()
        checks["signed_q"] = {"status": "FAILED", "exact_equal": False,
                              "max_abs_err": 5e-7}
        rows = [_alignment_row(status="PASSED", checks=checks)]
        verdict = evaluate_alignment(rows, expected_keys=[("frame_a", 0)])
        assert verdict["status"] == "FAILED"
        assert verdict["mismatches"][0]["field"] == "signed_q"

    def test_signed_q_tiny_difference_is_rejected(self):
        from scripts.eval_material_pipeline import _compare_arrays

        actual = np.zeros((4, 4), dtype=np.float32)
        expected = actual.copy()
        expected[0, 0] = 5e-7
        result = _compare_arrays(actual, expected, kind="exact")
        assert result["status"] == "FAILED"
        assert result["exact_equal"] is False

    def test_quality_tolerance_boundary(self):
        from scripts.eval_material_pipeline import _compare_arrays

        actual = np.zeros(4, dtype=np.float32)
        within = actual.copy()
        within[0] = 1e-6
        outside = actual.copy()
        outside[0] = 2e-6
        assert _compare_arrays(actual, within, kind="quality")["status"] == "PASSED"
        assert _compare_arrays(actual, outside, kind="quality")["status"] == "FAILED"

    def test_dtype_shape_and_non_finite_conflicts(self):
        from scripts.eval_material_pipeline import _compare_arrays

        expected = np.zeros((4, 4), dtype=np.uint8)
        assert _compare_arrays(np.zeros((4, 5), dtype=np.uint8), expected,
                               kind="exact")["status"] == "FAILED"
        assert _compare_arrays(np.zeros((4, 4), dtype=np.float32), expected,
                               kind="exact")["status"] == "FAILED"
        bad = np.zeros((4, 4), dtype=np.float32)
        bad[0, 0] = np.nan
        assert _compare_arrays(bad, bad.copy(), kind="exact")["status"] == "FAILED"

    def test_full_expected_set_with_all_checks_passes(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        rows = [_alignment_row("frame_a", 0), _alignment_row("frame_b", 0)]
        verdict = evaluate_alignment(
            rows, expected_keys=[("frame_a", 0), ("frame_b", 0)]
        )
        assert verdict["status"] == "PASSED"
        assert verdict["coverage"]["missing"] == []


# ---------------------------------------------------------------------------
# Zero-overlap unmatched records


class TestZeroOverlapRecords:
    def _mask(self, x0=10, x1=40):
        m = np.zeros((100, 100), dtype=np.uint8)
        m[10:40, x0:x1] = 1
        return m

    def test_zero_overlap_keeps_null_best_prediction(self):
        from scripts.eval_material_pipeline import unmatched_gt_records

        rows = unmatched_gt_records(
            frame_id="f", group="g", gt_classes=[3],
            gt_masks=[self._mask(x0=60, x1=90)],
            pred_masks=[self._mask(x0=10, x1=40)],
            matches=[],
        )
        record = rows[0]
        assert record["max_mask_iou"] == 0.0
        assert record["best_prediction_index"] is None
        assert record["best_prediction_matched_elsewhere"] is None
        assert record["no_positive_overlap"] is True
        assert record["predictions_in_frame"] == 1

    def test_no_predictions_is_distinct_from_zero_overlap(self):
        from scripts.eval_material_pipeline import unmatched_gt_records

        rows = unmatched_gt_records(
            frame_id="f", group="g", gt_classes=[0], gt_masks=[self._mask()],
            pred_masks=[], matches=[],
        )
        record = rows[0]
        assert record["predictions_in_frame"] == 0
        assert record["max_mask_iou"] is None
        assert record["best_prediction_index"] is None
        assert record["no_positive_overlap"] is None

    def test_positive_overlap_selects_the_best_and_reports_the_flag(self):
        from scripts.eval_material_pipeline import unmatched_gt_records

        matched_gt = self._mask(x0=60, x1=90)      # gt 0, taken by pred 0
        unmatched_gt = self._mask(x0=10, x1=40)    # gt 1, unmatched
        pred_matched = self._mask(x0=62, x1=92)    # pred 0
        pred_free = self._mask(x0=12, x1=42)       # pred 1, overlaps gt 1
        rows = unmatched_gt_records(
            frame_id="f", group="g", gt_classes=[2, 3],
            gt_masks=[matched_gt, unmatched_gt],
            pred_masks=[pred_matched, pred_free], matches=[(0, 0, 0.9)],
        )
        assert len(rows) == 1
        record = rows[0]
        assert record["gt_id"] == 1
        assert record["best_prediction_index"] == 1
        assert record["max_mask_iou"] > 0
        assert record["best_prediction_matched_elsewhere"] is False
        assert record["no_positive_overlap"] is False

    def test_positive_overlap_on_an_already_matched_prediction(self):
        from scripts.eval_material_pipeline import unmatched_gt_records

        matched_gt = self._mask(x0=10, x1=40)
        unmatched_gt = self._mask(x0=12, x1=42)
        shared_pred = self._mask(x0=11, x1=41)
        rows = unmatched_gt_records(
            frame_id="f", group="g", gt_classes=[0, 0],
            gt_masks=[matched_gt, unmatched_gt],
            pred_masks=[shared_pred], matches=[(0, 0, 0.9)],
        )
        record = rows[0]
        assert record["best_prediction_index"] == 0
        assert record["best_prediction_matched_elsewhere"] is True
        assert record["no_positive_overlap"] is False

    def test_positive_tie_uses_a_stable_tie_break(self):
        from scripts.eval_material_pipeline import unmatched_gt_records

        gt = self._mask(x0=10, x1=40)
        left = self._mask(x0=8, x1=38)
        right = self._mask(x0=12, x1=42)
        first = unmatched_gt_records(
            frame_id="f", group="g", gt_classes=[0], gt_masks=[gt],
            pred_masks=[left, right], matches=[],
        )[0]
        second = unmatched_gt_records(
            frame_id="f", group="g", gt_classes=[0], gt_masks=[gt],
            pred_masks=[left, right], matches=[],
        )[0]
        assert first["best_prediction_index"] == second["best_prediction_index"]
        assert first["max_mask_iou"] == second["max_mask_iou"]
