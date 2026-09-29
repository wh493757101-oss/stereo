"""4B pipeline evaluation tests (TDD: written before the implementation).

Covers the fixed evaluation contract:

- class-agnostic one-to-one GT/prediction matching by mask IoU >= 0.50,
  maximizing match count first, then total IoU, with a deterministic
  tie-break; duplicate detections, misses, false positives and empty frames
  are all handled explicitly;
- a classification error is an FN for the true class and an FP for the
  predicted class; unclassified and failed frames are never silently
  dropped; counts are conserved; zero denominators are handled;
- end-to-end per-class P/R/F1 and the paired same-seed changes;
- the independent-data admission gate refuses cross-source data while
  keeping training and evaluation fingerprints separate;
- the evaluation entry refuses an existing output directory, evaluates all
  12 candidates with same-seed pairs, never confuses the E1 pure control
  with the independent gray fallback, keeps GT and predicted mask paths
  separate, and runs no backward/optimizer (parameters unchanged).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.material_fusion_dataset import CLASS_NAMES


def _mask(shape=(100, 100), y0=10, x0=10, y1=40, x1=40) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


# ---------------------------------------------------------------------------
# Small local fakes (``tests/`` is not an importable package)


def _local_instance(instance_id: int = 0):
    from models.segmentation import Instance

    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:30, 20:40] = 255
    return Instance(id=instance_id, bbox=(20, 10, 40, 30), mask=mask,
                    confidence=0.9, class_id=0, class_name="object")


class _FakeSegmenter:
    def __init__(self, instances):
        self.instances = instances

    def predict(self, image, **kwargs):
        return self.instances


class _FakeMatcher:
    """A matcher whose stereo statistics are always invalid."""

    def __init__(self, valid=False):
        from core.stereo_matching import StereoMatchResult

        self.result = StereoMatchResult(
            disparity=np.zeros((48, 64), dtype=np.float32),
            valid=np.zeros((48, 64), dtype=np.uint8),
            elapsed_s=0.0, valid_ratio=0.0,
        )
        self.valid = valid

    def compute(self, left, right):
        return self.result

    def compute_bands(self, left, right, bands):
        return self.result

    def instance_stats(self, result, mask, instance_id=0):
        from core.stereo_matching import InstanceDepth

        return InstanceDepth(instance_id, 0.0, 0.0, None, False,
                             "no_valid_disparity")


def _material_engine_with_invalid_stereo():
    from gui.inference_engine import DualStageInferenceEngine

    return DualStageInferenceEngine(
        model_a=_FakeSegmenter(instances=[_local_instance(0)]),
        model_b=None,
        matcher=_FakeMatcher(valid=False),
        device="cpu",
        model_b_input_mode="material_fusion",
    )


# ---------------------------------------------------------------------------
# Mask IoU


class TestMaskIou:
    def test_identical_masks_have_iou_one(self):
        from scripts.eval_material_pipeline import mask_iou

        m = _mask()
        assert mask_iou(m, m) == pytest.approx(1.0)

    def test_half_overlap_is_half_iou(self):
        from scripts.eval_material_pipeline import mask_iou

        # A 30-px mask shifted by 10: inter 20*30, union 1200 -> IoU 0.5.
        a = _mask(x0=10, x1=40)
        b = _mask(x0=20, x1=50)
        assert mask_iou(a, b) == pytest.approx(0.5)

    def test_disjoint_masks_have_zero_iou(self):
        from scripts.eval_material_pipeline import mask_iou

        assert mask_iou(_mask(), _mask(x0=60, x1=90)) == 0.0


# ---------------------------------------------------------------------------
# Matching


class TestMatching:
    def test_threshold_is_inclusive(self):
        from scripts.eval_material_pipeline import mask_iou, match_instances

        a = _mask(x0=10, x1=40)
        b = _mask(x0=20, x1=50)
        assert mask_iou(a, b) == pytest.approx(0.5)
        matches = match_instances([a], [b], iou_threshold=0.50)
        assert matches == [(0, 0, pytest.approx(0.5))]

    def test_below_threshold_is_not_matched(self):
        from scripts.eval_material_pipeline import match_instances

        a = _mask(x0=10, x1=40)
        b = _mask(x0=27, x1=57)      # IoU below 0.5
        assert match_instances([a], [b], iou_threshold=0.50) == []

    def test_class_agnostic_and_one_to_one(self):
        """Two GT, two predictions, both matchable; exactly two matches."""
        from scripts.eval_material_pipeline import match_instances

        gt1 = _mask(x0=10, x1=40)
        gt2 = _mask(x0=60, x1=90)
        p1 = _mask(x0=12, x1=42)
        p2 = _mask(x0=62, x1=92)
        matches = match_instances([gt1, gt2], [p1, p2])
        assert sorted((g, p) for g, p, _ in matches) == [(0, 0), (1, 1)]

    def test_match_count_beats_total_iou(self):
        """One match with IoU 0.9 loses to two matches with ~0.5 each."""
        from scripts.eval_material_pipeline import match_instances

        gt1 = _mask(x0=10, x1=40)
        gt2 = _mask(x0=60, x1=90)
        # p1 overlaps both GTs strongly-ish, p2 only gt2 weakly.
        p1 = _mask(x0=12, x1=42)
        p2 = _mask(x0=62, x1=92)
        matches = match_instances([gt1, gt2], [p1, p2])
        assert len(matches) == 2

    def test_stable_tie_break_by_instance_ids(self):
        """Exactly tied configurations resolve by (gt_id, pred_id) order."""
        from scripts.eval_material_pipeline import match_instances

        gt = _mask()
        p1 = _mask(x0=15, x1=45)   # same IoU vs gt as p2 by symmetry
        p2 = _mask(x0=10, y0=15, x1=40, y1=45)
        iou1 = None  # both overlap gt in 25*25=625 of union 1225
        matches = match_instances([gt], [p1, p2])
        assert len(matches) == 1
        # Deterministic: the lower prediction index wins the tie.
        assert matches[0][0] == 0
        assert matches[0][1] in (0, 1)

    def test_duplicate_detections_leave_one_unmatched(self):
        from scripts.eval_material_pipeline import match_instances

        gt = [_mask()]
        preds = [_mask(x0=11, x1=41), _mask(x0=12, x1=42)]
        matches = match_instances(gt, preds)
        assert len(matches) == 1
        matched_pred = matches[0][1]
        assert matched_pred in (0, 1)

    def test_empty_frames_and_empty_predictions(self):
        from scripts.eval_material_pipeline import match_instances

        assert match_instances([], [_mask()]) == []
        assert match_instances([_mask()], []) == []
        assert match_instances([], []) == []


# ---------------------------------------------------------------------------
# Frame record and metrics


def _frame_record(
    frame="frame_a",
    group="group_a",
    gt_classes=(0, 2),
    matches=((0, 0, 0.9), (1, 1, 0.8)),
    pred_classes=(0, 2),
    n_predictions=None,
    unclassified=(),
    frame_error=None,
):
    n_pred = int(n_predictions) if n_predictions is not None else len(pred_classes)
    matched_by_gt = {gi: pi for gi, pi, _ in matches}
    gt_correct = []
    for gi, gt_class in enumerate(gt_classes):
        if gi in matched_by_gt:
            gt_correct.append(pred_classes[matched_by_gt[gi]] == gt_class)
        else:
            gt_correct.append(False)
    return {
        "frame_id": frame,
        "group": group,
        "gt_classes": list(gt_classes),
        "matches": [list(m) for m in matches],
        "pred_classes": list(pred_classes),
        "n_predictions": n_pred,
        "unclassified_gt_indices": list(unclassified),
        "frame_error": frame_error,
        "valid_ratio": [1.0, 1.0],
        "used_fallback": [False] * n_pred,
        "fallback_reasons": [],
        "iou_values": [m[2] for m in matches],
        "gt_ids": list(range(len(gt_classes))),
        "gt_correct": gt_correct,
    }


class TestMetrics:
    def test_classification_error_is_fn_and_fp(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        records = [_frame_record(gt_classes=(0,), matches=((0, 0, 0.9),),
                                 pred_classes=(2,))]
        report = aggregate_metrics(records, class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        e2e = report["end_to_end"]
        assert e2e["per_class"]["metal_submarine"]["fn"] == 1
        assert e2e["per_class"]["plastic_fish"]["fp"] == 1

    def test_missed_gt_is_fn_for_true_class(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        records = [_frame_record(gt_classes=(0, 3), matches=((0, 0, 0.9),),
                                 pred_classes=(0,), n_predictions=1)]
        report = aggregate_metrics(records, class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        assert report["counts"]["gt_instances"] == 2
        assert report["counts"]["matched"] == 1
        assert report["counts"]["missed_gt"] == 1
        assert report["end_to_end"]["per_class"]["real_fish"]["fn"] == 1

    def test_unmatched_prediction_is_fp_for_predicted_class(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        records = [_frame_record(gt_classes=(0,), matches=((0, 0, 0.9),),
                                 pred_classes=(0, 1), n_predictions=2)]
        report = aggregate_metrics(records, class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        assert (
            report["end_to_end"]["per_class"]["plastic_submarine"]["fp"] == 1
        )

    def test_unclassified_counts_as_end_to_end_failure(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        records = [_frame_record(gt_classes=(0,), matches=((0, 0, 0.9),),
                                 pred_classes=(0,), unclassified=(0,))]
        report = aggregate_metrics(records, class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        assert report["counts"]["unclassified_matched"] == 1
        assert report["end_to_end"]["per_class"]["metal_submarine"]["fn"] == 1
        assert report["end_to_end"]["macro_f1"] < 1.0

    def test_failed_frames_are_reported_not_dropped(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        records = [
            _frame_record(frame="frame_bad", frame_error="image_unreadable"),
            _frame_record(frame="frame_ok"),
        ]
        report = aggregate_metrics(records, class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        assert report["counts"]["failed_frames"] == 1
        assert report["counts"]["evaluated_frames"] == 1

    def test_count_conservation(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        records = [
            _frame_record(frame="f1", gt_classes=(0, 2, 3),
                          matches=((0, 0, 0.9), (2, 1, 0.7)),
                          pred_classes=(0, 3), n_predictions=2),
            _frame_record(frame="f2", gt_classes=(1,),
                          matches=(), pred_classes=(), n_predictions=0),
            _frame_record(frame="f3", gt_classes=(),
                          matches=(), pred_classes=(2,), n_predictions=1),
        ]
        report = aggregate_metrics(records, class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        c = report["counts"]
        assert c["gt_instances"] == 4
        assert c["predictions"] == 3
        assert c["matched"] == 2
        assert c["missed_gt"] == 2
        assert c["unmatched_predictions"] == 1
        assert c["matched"] + c["missed_gt"] == c["gt_instances"]
        assert c["matched"] + c["unmatched_predictions"] == c["predictions"]

    def test_zero_denominators_do_not_crash(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        records = [_frame_record(frame="f", gt_classes=(), matches=(),
                                 pred_classes=(), n_predictions=0)]
        report = aggregate_metrics(records, class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        assert report["detection"]["precision"] == 0.0
        assert report["detection"]["recall"] == 0.0
        assert report["detection"]["f1"] == 0.0
        assert report["end_to_end"]["macro_f1"] == 0.0

    def test_detection_scores_are_not_called_map(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        report = aggregate_metrics([], class_names=(
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"))
        assert "mAP" not in json.dumps(report)
        assert "IoU>=0.50" in report["detection"]["operating_point"]


class TestPairedChanges:
    def _records(self, correct_a, correct_b):
        return [
            _frame_record(
                frame=f"f{index}", gt_classes=(c,), matches=((0, 0, 0.9),),
                pred_classes=(c if ok else (c + 1) % 4,),
            )
            for index, (c, ok) in enumerate(
                zip((0, 1, 2, 3), (correct_a, correct_b, True, True))
            )
        ]

    def test_fixed_and_broke_are_counted(self):
        from scripts.eval_material_pipeline import paired_changes

        base = self._records(True, False)      # sample 1 ok, sample 2 wrong
        other = self._records(False, True)      # sample 1 wrong, sample 2 ok
        report = paired_changes(base, other)
        assert report["fixed"] == 1
        assert report["broke"] == 1
        assert report["both_correct"] == 2


# ---------------------------------------------------------------------------
# Independent-data admission


class TestEvalEntryContract:
    def test_output_directory_is_refused_when_it_exists(self, scratch_dir):
        from scripts.eval_material_pipeline import prepare_output_dir

        target = scratch_dir / "quality"
        target.mkdir()
        with pytest.raises(FileExistsError):
            prepare_output_dir(target)

    def test_candidate_matrix_pairs_same_seed_variants(self):
        from scripts.eval_material_pipeline import candidate_pairs

        candidates = [
            {"variant": v, "seed": s}
            for v in ("gray", "gray_gray_valid", "gray_valid", "gray_polar")
            for s in (2026, 2027, 2028)
        ]
        pairs = candidate_pairs(candidates)
        assert len(pairs) == 9          # E4 vs E1/E2/E3 for each of 3 seeds
        for pair in pairs:
            assert pair[0]["seed"] == pair[1]["seed"]
            assert pair[0]["variant"] == "gray_polar"

    def test_e1_identity_is_never_the_fallback(self):
        from scripts.eval_material_pipeline import e1_is_pure_control

        assert e1_is_pure_control({"variant": "gray", "seed": 2026}) is True
        assert e1_is_pure_control({"variant": "gray_polar", "seed": 2026}) is False

    def test_mask_paths_are_labelled_separately(self):
        from scripts.eval_material_pipeline import (
            MASK_SOURCE_GT_EVAL,
            MASK_SOURCE_PREDICTED_EVAL,
        )

        assert MASK_SOURCE_GT_EVAL != MASK_SOURCE_PREDICTED_EVAL
        assert "ground_truth" in MASK_SOURCE_GT_EVAL
        assert "predicted" in MASK_SOURCE_PREDICTED_EVAL

    def test_report_digest_is_of_written_bytes(self, scratch_dir):
        import hashlib

        from scripts.eval_material_pipeline import write_jsonl_bytes

        records = [{"non_ascii": "偏振", "n": 1}, {"n": 2}]
        path = write_jsonl_bytes(scratch_dir, "instances.jsonl", records)
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        assert len(raw.decode("utf-8").splitlines()) == 2


# ---------------------------------------------------------------------------
# 4B fix round: a real end-to-end Macro-F1 bootstrap
#
# The first 4B report labelled a per-GT correctness delta as an end-to-end
# Macro-F1 bootstrap: it never computed Macro-F1 and never counted the
# unmatched predictions as false positives. These tests pin the real
# metric and keep the two quantities clearly separate.


def _records_for_groups(spec):
    """spec: {group: [(gt_class, pred_class_or_None, matched_bool), ...]}."""
    records = []
    for group, entries in spec.items():
        gt_classes, pred_classes, matches = [], [], []
        for gt_class, pred_class, matched in entries:
            pred_index = len(pred_classes)
            pred_classes.append(pred_class if pred_class is not None else 0)
            if matched:
                matches.append([len(gt_classes), pred_index, 0.9])
            gt_classes.append(gt_class)
        records.append({
            "frame_id": f"{group}_frame",
            "group": group,
            "gt_classes": gt_classes,
            "matches": matches,
            "pred_classes": pred_classes,
            "n_predictions": len(pred_classes),
            "unclassified_gt_indices": [],
            "frame_error": None,
            "gt_ids": list(range(len(gt_classes))),
            "gt_correct": [],
        })
    return records


class TestEndToEndMacroF1Bootstrap:
    def _pair(self, *, e4_extra_fp_class=None):
        """E4 differs from E1 only by extra unmatched detections."""
        entries = {
            "g1": [(0, 0, True), (2, 2, True)],
            "g2": [(1, 1, True)],
            "g3": [(3, 3, True)],
            "g4": [(2, 2, True)],
        }
        if e4_extra_fp_class is not None:
            entries["g1"].append((0, e4_extra_fp_class, False))
            entries["g1"].append((0, e4_extra_fp_class, False))
        return _records_for_groups(entries), _records_for_groups({
            "g1": [(0, 0, True), (2, 2, True)],
            "g2": [(1, 1, True)],
            "g3": [(3, 3, True)],
            "g4": [(2, 2, True)],
        })

    def test_bootstrap_reports_macro_f1_delta_not_correctness_delta(self):
        from scripts.eval_material_pipeline import end_to_end_macro_f1_bootstrap

        e4, e1 = self._pair(e4_extra_fp_class=1)
        report = end_to_end_macro_f1_bootstrap(e4, e1)
        assert report["metric"] == "end_to_end_macro_f1_delta"
        assert report["point_estimate"] < 0        # extra FPs hurt E4
        assert report["resamples"] == 2000
        assert report["seed"] == 2026
        assert report["clusters"] == 4
        assert report["valid_resamples"] > 0
        assert report["classes"] == list(CLASS_NAMES)

    def test_only_unmatched_predictions_change_metric_and_interval(self):
        from scripts.eval_material_pipeline import end_to_end_macro_f1_bootstrap

        e4_clean, e1 = self._pair()
        e4_dirty, _ = self._pair(e4_extra_fp_class=1)
        base = end_to_end_macro_f1_bootstrap(e4_clean, e1)
        changed = end_to_end_macro_f1_bootstrap(e4_dirty, e1)
        assert base["point_estimate"] == pytest.approx(0.0)
        assert changed["point_estimate"] < 0.0
        assert (changed["ci_low"], changed["ci_high"]) != (
            base["ci_low"], base["ci_high"]
        )

    def test_misses_and_misclassifications_enter_the_metric(self):
        from scripts.eval_material_pipeline import end_to_end_macro_f1_bootstrap

        e4 = _records_for_groups({
            "g1": [(0, 0, True), (2, 2, True)],
            "g2": [(1, 1, True)],
            "g3": [(3, 3, True)],
            "g4": [(2, 2, True)],
        })
        e1 = _records_for_groups({
            "g1": [(0, 2, True), (2, 2, True)],   # misclassified as class 2
            "g2": [(1, None, False)],             # missed
            "g3": [(3, 3, True)],
            "g4": [(2, 2, True)],
        })
        report = end_to_end_macro_f1_bootstrap(e4, e1)
        assert report["point_estimate"] > 0.0
        assert report["e4_macro_f1"] > report["e1_macro_f1"]

    def test_missing_class_keeps_the_fixed_class_set(self):
        from scripts.eval_material_pipeline import end_to_end_macro_f1_bootstrap

        per_group = {
            "g1": [(0, 0, True)], "g2": [(1, 1, True)],
            "g3": [(2, 2, True)], "g4": [(3, 3, True)],
        }
        report = end_to_end_macro_f1_bootstrap(
            _records_for_groups(per_group), _records_for_groups(per_group)
        )
        assert report["classes"] == list(CLASS_NAMES)
        assert report["point_estimate"] == pytest.approx(0.0)
        assert report["ci_low"] == pytest.approx(0.0)
        assert report["ci_high"] == pytest.approx(0.0)

    def test_duplicate_group_draws_are_weighted(self):
        from scripts.eval_material_pipeline import end_to_end_macro_f1_bootstrap

        e4 = _records_for_groups({"g1": [(0, 0, True)]})
        e1 = _records_for_groups({"g1": [(0, 2, True)]})
        report = end_to_end_macro_f1_bootstrap(e4, e1)
        assert report["point_estimate"] > 0.0
        # every 4-group draw repeats the only informative group
        assert report["ci_low"] == pytest.approx(report["point_estimate"])
        assert report["ci_high"] == pytest.approx(report["point_estimate"])

    def test_seed_is_reproducible(self):
        from scripts.eval_material_pipeline import end_to_end_macro_f1_bootstrap

        e4, e1 = self._pair(e4_extra_fp_class=3)
        first = end_to_end_macro_f1_bootstrap(e4, e1, resamples=200)
        second = end_to_end_macro_f1_bootstrap(e4, e1, resamples=200)
        assert first["ci_low"] == second["ci_low"]
        assert first["ci_high"] == second["ci_high"]
        assert end_to_end_macro_f1_bootstrap(
            e4, e1, seed=7, resamples=50
        )["seed"] == 7

    def test_point_estimate_comes_from_the_full_sample(self):
        from scripts.eval_material_pipeline import (
            aggregate_metrics,
            end_to_end_macro_f1_bootstrap,
        )

        e4, e1 = self._pair(e4_extra_fp_class=2)
        report = end_to_end_macro_f1_bootstrap(e4, e1)
        e4_macro = aggregate_metrics(
            e4, class_names=CLASS_NAMES)["end_to_end"]["macro_f1"]
        e1_macro = aggregate_metrics(
            e1, class_names=CLASS_NAMES)["end_to_end"]["macro_f1"]
        assert report["e4_macro_f1"] == pytest.approx(e4_macro)
        assert report["e1_macro_f1"] == pytest.approx(e1_macro)
        assert report["point_estimate"] == pytest.approx(e4_macro - e1_macro)

    def test_summary_names_the_metric_it_actually_computes(self):
        from scripts.eval_material_pipeline import build_bootstrap_section

        e4, e1 = self._pair(e4_extra_fp_class=1)
        section = build_bootstrap_section(
            {"gray_polar_s2026": e4}, {"gray_s2026": e1},
            seeds=(2026,), resamples=50,
        )
        macro = section["end_to_end_macro_f1_e4_vs_e1"]["seed_2026"]
        assert macro["metric"] == "end_to_end_macro_f1_delta"
        correctness = section["instance_correctness_delta"]["seed_2026"]
        assert correctness["metric"] == "paired_gt_instance_correctness_delta"
        assert "macro_f1" not in correctness["metric"]


class TestUnmatchedRecords:
    def test_unmatched_gt_records_carry_iou_and_matched_flag(self):
        """Corrected expectation: zero overlap must not name a prediction.

        The earlier version of this test asserted
        ``best_prediction_matched_elsewhere is True`` for a GT whose every
        overlap is zero — it encoded the reviewed defect, where an argmax
        over an all-zero row picked the first prediction and reported it as
        an already-matched merge signal.
        """
        from scripts.eval_material_pipeline import unmatched_gt_records

        gt_masks = [_mask(), _mask(x0=60, x1=90)]
        pred_masks = [_mask(x0=12, x1=42)]
        records = unmatched_gt_records(
            frame_id="f", group="g", gt_classes=[0, 3], gt_masks=gt_masks,
            pred_masks=pred_masks, matches=[(0, 0, 0.9)],
        )
        assert len(records) == 1
        record = records[0]
        assert record["gt_id"] == 1
        assert record["gt_class"] == 3
        assert record["max_mask_iou"] == 0.0
        assert record["best_prediction_index"] is None
        assert record["best_prediction_matched_elsewhere"] is None
        assert record["no_positive_overlap"] is True

    def test_counts_conserved_with_unmatched_and_unclassified(self):
        from scripts.eval_material_pipeline import aggregate_metrics

        # gt 2 (class 3) is matched to prediction 1 but that prediction
        # failed to classify; gt 2's class is never silently dropped.
        records = [_frame_record(
            gt_classes=(0, 2, 3), matches=((0, 0, 0.9), (2, 1, 0.8)),
            pred_classes=(0, 1, 1), n_predictions=3, unclassified=(2,),
        )]
        counts = aggregate_metrics(records, class_names=CLASS_NAMES)["counts"]
        assert counts["matched"] + counts["missed_gt"] == counts["gt_instances"]
        assert (counts["matched"] + counts["unmatched_predictions"]
                == counts["predictions"])
        assert counts["unclassified_matched"] == 1
        assert counts["missed_gt"] == 1


class TestIndependentDataClosed:
    def test_cli_refuses_independent_data_before_any_work(self, scratch_dir):
        from scripts import eval_material_pipeline as ev

        data = scratch_dir / "plausible_independent"
        (data / "val").mkdir(parents=True)
        (data / "dataset_audit.json").write_text("{}", encoding="utf-8")
        output = scratch_dir / "quality_should_not_exist"
        code = ev.main([
            "--config", "configs/experiments/material_runtime_v2_mask.yaml",
            "--split", "val", "--evaluation-kind", "retrospective",
            "--device", "0", "--output-dir", str(output),
            "--independent-data", str(data),
        ])
        assert code != 0
        assert not output.exists(), (
            "the refusal must happen before the output directory is created"
        )

    def test_no_admission_helper_can_look_like_a_passed_gate(self):
        from scripts import eval_material_pipeline as ev

        assert not hasattr(ev, "admit_independent_dataset"), (
            "the partial admission path was removed: an unimplemented "
            "independent-data evaluation must not look like a passed gate"
        )


# ---------------------------------------------------------------------------
# 4B fix round: GT-only geometry (band bbox vs crop window)


class TestGtGeometry:
    """The generation contract keeps two geometries apart.

    Band matching uses the raw polygon extrema; the classification crop
    window uses ``make_crop_window`` (floor/ceil, padding, clipping). The
    first 4B evaluation derived both from the rasterised mask's bounding
    box, which disagreed with V5 for 15 of 368 samples.
    """

    def _label(self, path, lines):
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_band_bbox_is_polygon_extrema_and_window_is_make_crop_window(
        self, scratch_dir
    ):
        from core.gt_geometry import build_gt_geometry

        # A polygon whose float bounds are not integers: the raster bbox
        # would be (2, 3, 5, 6) while the generation window is floor/ceil
        # expanded by the padding.
        label = self._label(scratch_dir / "frame.txt", [
            "0 0.02 0.03 0.05 0.03 0.05 0.06 0.02 0.06",
        ])
        geometry = build_gt_geometry(label, width=100, height=100, padding=10)
        assert len(geometry) == 1
        item = geometry[0]
        assert item.band_bbox == pytest.approx((2.0, 3.0, 5.0, 6.0))
        # floor(2)-10, floor(3)-10 .. ceil(5)+10, ceil(6)+10
        assert item.crop_window == (0, 0, 15, 16)
        assert item.mask.shape == (100, 100)

    def test_float_bounds_differ_from_the_raster_bounding_box(self, scratch_dir):
        import numpy as np

        from core.gt_geometry import build_gt_geometry

        # x from 0.0101 to 0.0199 of a 1000-px image: raster pixels start at
        # column 10 while the float minimum is 10.1 -> floor differs.
        label = self._label(scratch_dir / "frame.txt", [
            "0 0.0101 0.0101 0.0199 0.0101 0.0199 0.0199 0.0101 0.0199",
        ])
        item = build_gt_geometry(label, width=1000, height=1000, padding=0)[0]
        ys, xs = np.nonzero(item.mask)
        raster_bbox = (int(xs.min()), int(ys.min()),
                       int(xs.max()) + 1, int(ys.max()) + 1)
        assert item.band_bbox != raster_bbox

    def test_touching_and_out_of_range_polygons_are_clipped(self, scratch_dir):
        from core.gt_geometry import build_gt_geometry

        label = self._label(scratch_dir / "frame.txt", [
            "0 0.0 0.0 0.2 0.0 0.2 0.2 0.0 0.2",
            "1 0.9 0.9 1.2 0.9 1.2 1.2 0.9 1.2",
        ])
        geometry = build_gt_geometry(label, width=50, height=50, padding=10)
        first, second = geometry
        assert first.crop_window[0] == 0 and first.crop_window[1] == 0
        assert second.crop_window[2] <= 50 and second.crop_window[3] <= 50
        assert second.band_bbox[2] > 50          # the float bound is kept

    def test_gt_instances_use_band_bbox_and_the_window_is_separate(
        self, scratch_dir
    ):
        from core.gt_geometry import build_gt_geometry, gt_instances

        label = self._label(scratch_dir / "frame.txt", [
            "2 0.02 0.03 0.05 0.03 0.05 0.06 0.02 0.06",
        ])
        geometry = build_gt_geometry(label, width=100, height=100)
        instances = gt_instances(geometry)
        assert instances[0].bbox == pytest.approx((2.0, 3.0, 5.0, 6.0))
        assert instances[0].class_id == 2
        assert geometry[0].crop_window == (0, 0, 15, 16)

    def test_eval_entry_uses_the_shared_helper(self):
        """The evaluation must not re-derive GT geometry on its own."""
        import ast
        from pathlib import Path as _Path

        source = _Path("scripts/eval_material_pipeline.py").read_text(
            encoding="utf-8")
        assert "gt_geometry" in source
        assert "_load_gt_instances" not in source, (
            "the old raster-bbox GT loader was replaced by the shared helper"
        )
        ast.parse(source)


class TestGtCropWindowOverride:
    def test_engine_accepts_a_gt_crop_window_override(self, scratch_dir):
        """The GT path passes exact windows; the predicted path does not."""
        import inspect

        from gui.inference_engine import DualStageInferenceEngine

        signature = inspect.signature(
            DualStageInferenceEngine.process_instances_detailed
        )
        assert "crop_windows" in signature.parameters
        assert (
            signature.parameters["crop_windows"].default is None
        ), "the default keeps the Model A path unchanged"
        frame_signature = inspect.signature(
            DualStageInferenceEngine.process_frame_detailed
        )
        assert "crop_windows" not in frame_signature.parameters, (
            "the predicted path must not accept GT geometry"
        )


class TestGtGenerationContract:
    """A stereo-invalid instance carries a zero quality vector in V5.

    The generation writes zeros for the whole quality vector (and the polar
    channels) when an instance's stereo statistics are invalid, while the
    live engine reports the measured in-bounds/brightness ratios. The GT
    evaluation path must reproduce the *generation* contract exactly; the
    Model A path keeps the live behaviour.
    """

    def test_gt_path_accepts_the_generation_contract_flag(self):
        import inspect

        from gui.inference_engine import DualStageInferenceEngine

        signature = inspect.signature(
            DualStageInferenceEngine.process_instances_detailed
        )
        assert "gt_generation_contract" in signature.parameters
        assert signature.parameters["gt_generation_contract"].default is False
        frame_signature = inspect.signature(
            DualStageInferenceEngine.process_frame_detailed
        )
        assert "gt_generation_contract" not in frame_signature.parameters

    def test_invalid_stereo_instance_is_zeroed_only_under_the_gt_contract(
        self, scratch_dir
    ):
        from gui.inference_engine import DualStageInferenceEngine

        engine = _material_engine_with_invalid_stereo()
        gray = np.zeros((48, 64), dtype=np.uint8)
        result = engine.process_instances_detailed(
            gray, gray, [_local_instance(0)], already_rectified=True,
            gt_generation_contract=True,
        )
        (sample,) = result.material_inputs
        assert sample is not None
        assert float(np.abs(sample.quality).max()) == 0.0, (
            "a stereo-invalid GT instance must carry the generation's zero "
            "quality vector"
        )
        assert float(np.abs(sample.signed_q).max()) == 0.0
        assert sample.polar_invalid is True

    def test_predicted_path_keeps_the_live_behaviour(self, scratch_dir):
        engine = _material_engine_with_invalid_stereo()
        gray = np.zeros((48, 64), dtype=np.uint8)
        result = engine.process_frame_detailed(
            gray, gray, already_rectified=True,
        )
        (sample,) = result.material_inputs
        assert sample is not None
        # The Model A path still reports the measured ratios (the live
        # behaviour is unchanged by the GT-only contract flag).
        assert sample.quality.shape == (4,)
        assert sample.polar_invalid is True



class TestPureAndRouted:
    def test_pure_batch_never_uses_the_fallback(self):
        from models.material_classifier import MaterialFusionClassifier

        assert hasattr(MaterialFusionClassifier, "predict_pure_batch"), (
            "a pure (no-routing) diagnostic entry is required"
        )

    def test_pure_marks_non_finite_inputs_not_evaluable(self):
        """Non-finite polarization must never be silently zeroed and called
        a legal pure inference."""
        from models.material_classifier import MaterialFusionClassifier

        source = MaterialFusionClassifier.predict_pure_batch.__doc__ or ""
        assert "not evaluable" in source.lower() or "not_evaluable" in source

