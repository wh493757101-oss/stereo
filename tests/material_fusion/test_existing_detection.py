"""Synthetic tests for the fixed-condition detection diagnosis.

Model A itself is replaced by stub predictors (tiny torch modules with
deterministic outputs); the tests pin condition passing, the frozen
matching rule, the D0 stop rule, identity/correspondence handling and
the attribution rules. No real weights, frames or datasets are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from core.material_probe import compare_module_states, snapshot_module_state

from scripts.diagnose_existing_detection import (
    ATTRIBUTION_SIGN_NOTE,
    attribution_for_unmatched_gt,
    build_condition_plan,
    d0_consistency_report,
    load_historical_d0,
    run_detection_diagnosis,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures


def _frame_record(frame_id: str, group: str, gt_classes: list[int],
                  predictions: int, matched: int) -> dict:
    return {
        "frame_id": frame_id,
        "group": group,
        "gt_instances": len(gt_classes),
        "gt_classes": gt_classes,
        "predictions": predictions,
        "matched": matched,
        "missed_gt": len(gt_classes) - matched,
        "unmatched_predictions": predictions - matched,
    }


class _Instance:
    """The subset of models.segmentation.Instance the runner consumes."""

    def __init__(self, index, bbox, mask, confidence, class_id=0):
        self.id = index
        self.bbox = bbox
        self.mask = mask
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = "target"


class _StubModel:
    """A tiny torch-module-backed stub predictor recording its kwargs."""

    def __init__(self, per_frame_masks, image_shape=(64, 80), shift=0.0):
        self.model = nn.Linear(2, 2)  # named_parameters snapshot target
        self.per_frame_masks = per_frame_masks
        self.image_shape = image_shape
        self.calls = []  # (conf, imgsz) per predict call
        self.shift = shift

    def predict(self, image_bgr, *, conf, imgsz):
        self.calls.append((conf, imgsz))
        frame_index = len(self.calls) - 1
        masks = self.per_frame_masks[frame_index % len(self.per_frame_masks)]
        instances = []
        for index, mask in enumerate(masks):
            full = np.zeros(self.image_shape, dtype=np.uint8)
            h, w = mask.shape
            full[:h, :w] = mask
            instances.append(_Instance(
                index, (0, 0, w, h), full,
                round(0.9 - index * 0.1 - self.shift, 3)))
        return instances


def _box_mask(shape, y1, x1, y2, x2):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


CONFIG = {
    "experiment": "material_existing_data_v1",
    "data": {"root": "v5", "dev_pool_splits": ["train", "val"]},
    "baselines": {
        "model_a": {"path": "weights/model_a.pt",
                    "sha256": "0" * 64},
        "historical_d0_evidence": {
            "frames_jsonl": "evidence/frames.jsonl",
            "summary_json": "evidence/summary.json",
            "reference_candidate": "gray_polar_s2026",
            "reference_layer": "layer_b.routed",
        },
    },
    "detection": {
        "split": "val",
        "conditions": {
            "D0": {"imgsz": 640, "conf": 0.25},
            "D1": {"imgsz": 640, "conf": 0.10},
            "D2": {"imgsz": 640, "conf": 0.05},
            "D3": {"imgsz": 960, "conf": 0.25},
        },
        "nms_iou": 0.45,
        "match_iou_threshold": 0.50,
        "order": ["D0", "D1", "D2", "D3"],
        "d0_reproduction": "require",
        "timing": {"enabled": False, "warmup": 1, "repeats": 2},
        "atlas": {"matched_controls_per_class": 1, "margin": 8},
        "max_runtime_minutes": 30,
    },
}


def _frames_source():
    """Two val frames with fixed GT polygons (class 0 and class 3)."""
    frames = []
    for index, (stem, classes) in enumerate([
            ("val_frame_000", [0, 3]),
            ("val_frame_001", [2])]):
        gt = []
        for object_index, class_id in enumerate(classes):
            mask = _box_mask((64, 80), 4 + 14 * object_index,
                             4 + 30 * object_index,
                             18 + 14 * object_index,
                             20 + 30 * object_index)
            ys, xs = np.nonzero(mask)
            gt.append({
                "object_index": object_index,
                "class_id": class_id,
                "class_name": ["metal_submarine", "plastic_submarine",
                               "plastic_fish", "real_fish"][class_id],
                "mask": mask,
                "bbox": (int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1),
            })
        frames.append({
            "frame_id": stem, "group": f"g{index}", "split": "val",
            "image": np.zeros((64, 80, 3), dtype=np.uint8),
            "gt": gt,
        })
    return frames


def _perfect_model(frames):
    """Predict exactly the GT masks with conf 0.9 (D0-reproducible)."""
    per_frame = [[gt["mask"] for gt in frame["gt"]] for frame in frames]
    return _StubModel(per_frame)


def _write_historical_evidence(scratch_dir: Path, frames) -> Path:
    evidence = scratch_dir / "evidence"
    evidence.mkdir(parents=True)
    records = []
    for frame in frames:
        records.append(_frame_record(
            frame["frame_id"], frame["group"],
            [gt["class_id"] for gt in frame["gt"]],
            predictions=len(frame["gt"]), matched=len(frame["gt"])))
    (evidence / "frames.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (evidence / "summary.json").write_text(json.dumps({
        "per_candidate": {"gray_polar_s2026": {
            "layer_b": {"routed": {"counts": {
                "frames": len(records),
                "gt_instances": sum(r["gt_instances"] for r in records),
                "predictions": sum(r["predictions"] for r in records),
                "matched": sum(r["matched"] for r in records),
            }}}}}}), encoding="utf-8")
    return evidence


# ---------------------------------------------------------------------------
# Condition plan


class TestConditionPlan:
    def test_conditions_read_from_config_in_order(self):
        plan = build_condition_plan(CONFIG)
        assert [c["name"] for c in plan] == ["D0", "D1", "D2", "D3"]
        assert plan[0]["conf"] == 0.25 and plan[0]["imgsz"] == 640
        assert plan[2]["conf"] == 0.05
        assert plan[3]["imgsz"] == 960 and plan[3]["conf"] == 0.25

    def test_match_threshold_is_frozen(self):
        plan = build_condition_plan(CONFIG)
        assert all(c["match_iou"] == 0.50 for c in plan)
        assert all(c["nms_iou"] == 0.45 for c in plan)


# ---------------------------------------------------------------------------
# Historical baseline binding


class TestHistoricalD0:
    def test_loads_aggregates_and_per_frame(self, tmp_path):
        frames = _frames_source()
        evidence = _write_historical_evidence(tmp_path, frames)
        historical = load_historical_d0(
            CONFIG, tmp_path, lambda relative: tmp_path / relative)
        assert historical["frames"] == 2
        assert historical["gt_instances"] == 3
        assert historical["predictions"] == 3
        assert historical["matched"] == 3
        assert historical["per_frame"]["val_frame_000"]["predictions"] == 2

    def test_missing_evidence_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_historical_d0(
                CONFIG, tmp_path, lambda relative: tmp_path / relative)


# ---------------------------------------------------------------------------
# D0 consistency gate


class TestD0Consistency:
    def test_consistent_passes(self):
        observed = {
            "frames": 2, "gt_instances": 3, "predictions": 3, "matched": 3,
            "per_frame": {"val_frame_000": {"predictions": 2, "matched": 2},
                          "val_frame_001": {"predictions": 1, "matched": 1}},
        }
        historical = {
            "frames": 2, "gt_instances": 3, "predictions": 3, "matched": 3,
            "per_frame": {"val_frame_000": {"predictions": 2, "matched": 2},
                          "val_frame_001": {"predictions": 1, "matched": 1}},
        }
        report = d0_consistency_report(observed, historical)
        assert report["consistent"] is True
        assert report["mismatches"] == []

    def test_prediction_count_mismatch_detected(self):
        observed = {"frames": 2, "gt_instances": 3, "predictions": 4,
                    "matched": 3,
                    "per_frame": {"val_frame_000": {"predictions": 3,
                                                    "matched": 2},
                                  "val_frame_001": {"predictions": 1,
                                                    "matched": 1}}}
        historical = {"frames": 2, "gt_instances": 3, "predictions": 3,
                      "matched": 3,
                      "per_frame": {"val_frame_000": {"predictions": 2,
                                                      "matched": 2},
                                    "val_frame_001": {"predictions": 1,
                                                      "matched": 1}}}
        report = d0_consistency_report(observed, historical)
        assert report["consistent"] is False
        assert any("predictions" in m for m in report["mismatches"])
        assert any("val_frame_000" in m for m in report["mismatches"])


# ---------------------------------------------------------------------------
# Attribution rules (signs only, never claimed as human-reviewed)


class TestAttributionRules:
    def test_lower_confidence_rescues(self):
        result = attribution_for_unmatched_gt({
            "matched_D1": True, "matched_D2": False, "matched_D3": False,
            "d0_max_iou": 0.0, "d0_best_matched_elsewhere": None,
            "d0_predictions_in_frame": 3,
        })
        assert result == "confidence_related"

    def test_scale_rescues(self):
        result = attribution_for_unmatched_gt({
            "matched_D1": False, "matched_D2": False, "matched_D3": True,
            "d0_max_iou": 0.1, "d0_best_matched_elsewhere": False,
            "d0_predictions_in_frame": 2,
        })
        assert result == "input_scale_related"

    def test_duplicate_or_merge_sign(self):
        result = attribution_for_unmatched_gt({
            "matched_D1": False, "matched_D2": False, "matched_D3": False,
            "d0_max_iou": 0.3, "d0_best_matched_elsewhere": True,
            "d0_predictions_in_frame": 2,
        })
        assert result == "duplicate_or_merge_sign"

    def test_insufficient_mask_overlap_sign(self):
        result = attribution_for_unmatched_gt({
            "matched_D1": False, "matched_D2": False, "matched_D3": False,
            "d0_max_iou": 0.3, "d0_best_matched_elsewhere": False,
            "d0_predictions_in_frame": 2,
        })
        assert result == "insufficient_mask_overlap_sign"

    def test_no_corresponding_prediction(self):
        result = attribution_for_unmatched_gt({
            "matched_D1": False, "matched_D2": False, "matched_D3": False,
            "d0_max_iou": 0.0, "d0_best_matched_elsewhere": None,
            "d0_predictions_in_frame": 2,
        })
        assert result == "no_corresponding_prediction_sign"

    def test_no_prediction_in_frame(self):
        result = attribution_for_unmatched_gt({
            "matched_D1": False, "matched_D2": False, "matched_D3": False,
            "d0_max_iou": None, "d0_best_matched_elsewhere": None,
            "d0_predictions_in_frame": 0,
        })
        assert result == "no_prediction_in_frame"

    def test_sign_note_is_explicit(self):
        assert "no human review" in ATTRIBUTION_SIGN_NOTE
        assert "sign" in ATTRIBUTION_SIGN_NOTE


# ---------------------------------------------------------------------------
# Full diagnosis over stub models


def _write_split_assignment(scratch_dir: Path, frames) -> Path:
    path = scratch_dir / "split_assignment.csv"
    lines = ["output_stem,split,group_name,block_id,left_path"]
    for frame in frames:
        lines.append(f"{frame['frame_id']},val,{frame['group']},0,left.png")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestRunDetectionDiagnosis:
    def test_perfect_model_reproduces_and_matches(self, scratch_dir):
        frames = _frames_source()
        model = _perfect_model(frames)

        def factory(path, conf, iou, imgsz, device):
            return model

        summary = run_detection_diagnosis(
            config=CONFIG,
            output_dir=scratch_dir / "detection",
            frames_provider=lambda: frames,
            model_factory=factory,
            historical=None,  # no baseline binding in this synthetic run
            device="cpu",
        )
        assert summary["status"] == "COMPLETED"
        d0 = summary["conditions"]["D0"]
        assert d0["counts"]["gt_instances"] == 3
        assert d0["counts"]["predictions"] == 3
        assert d0["counts"]["matched"] == 3
        assert d0["counts"]["unmatched_gt"] == 0
        # condition parameters reached the stub, in order
        assert model.calls[0] == (0.25, 640)
        assert model.calls[-1] == (0.25, 960)
        # artifacts exist
        assert (scratch_dir / "detection" / "conditions" / "D0"
                / "frames.jsonl").is_file()
        assert (scratch_dir / "detection" / "conditions" / "D3"
                / "per_gt.jsonl").is_file()
        assert (scratch_dir / "detection" / "summary.json").is_file()

    def test_d0_mismatch_stops_later_conditions(self, scratch_dir):
        frames = _frames_source()
        # one extra false prediction in the first frame -> D0 differs from
        # the historical evidence
        model = _StubModel([
            [gt["mask"] for gt in frames[0]["gt"]]
            + [_box_mask((32, 40), 0, 0, 10, 10)],
            [gt["mask"] for gt in frames[1]["gt"]],
        ])
        evidence = _write_historical_evidence(scratch_dir, frames)

        def factory(path, conf, iou, imgsz, device):
            return model

        summary = run_detection_diagnosis(
            config=CONFIG,
            output_dir=scratch_dir / "detection",
            frames_provider=lambda: frames,
            model_factory=factory,
            historical=load_historical_d0(
                CONFIG, scratch_dir,
                lambda relative: scratch_dir / relative),
            device="cpu",
        )
        assert summary["status"] == "FAILED_D0_REPRODUCTION"
        assert summary["d0_consistency"]["consistent"] is False
        assert "D1" not in summary["conditions"]
        assert not (scratch_dir / "detection" / "conditions" / "D1").exists()

    def test_mask_size_mismatch_refused(self, scratch_dir):
        frames = _frames_source()

        class _BadModel(_StubModel):
            def predict(self, image_bgr, *, conf, imgsz):
                instances = super().predict(image_bgr, conf=conf, imgsz=imgsz)
                first = instances[0]
                instances[0] = _Instance(
                    first.id, first.bbox, first.mask[:20, :20].copy(),
                    first.confidence)
                return instances

        model = _BadModel([[gt["mask"] for gt in f["gt"]]
                           for f in frames])

        def factory(path, conf, iou, imgsz, device):
            return model

        with pytest.raises(ValueError, match="mask"):
            run_detection_diagnosis(
                config=CONFIG,
                output_dir=scratch_dir / "detection",
                frames_provider=lambda: frames,
                model_factory=factory,
                historical=None,
                device="cpu",
            )

    def test_empty_prediction_frame_is_recorded(self, scratch_dir):
        frames = _frames_source()
        model = _StubModel([[]])  # every frame predicts nothing

        def factory(path, conf, iou, imgsz, device):
            return model

        summary = run_detection_diagnosis(
            config=CONFIG,
            output_dir=scratch_dir / "detection",
            frames_provider=lambda: frames,
            model_factory=factory,
            historical=None,
            device="cpu",
        )
        d0 = summary["conditions"]["D0"]
        assert d0["counts"]["predictions"] == 0
        assert d0["counts"]["unmatched_gt"] == 3
        records = [
            json.loads(line)
            for line in (scratch_dir / "detection" / "conditions" / "D0"
                         / "unmatched_gt.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 3
        assert all(r["no_positive_overlap"] or
                   r["predictions_in_frame"] == 0 for r in records)

    def test_gt_identity_stable_across_conditions(self, scratch_dir):
        frames = _frames_source()
        # D0 misses the second GT of frame 0; D1 (lower conf) recovers it
        gt0_masks = [gt["mask"] for gt in frames[0]["gt"]]
        gt1_masks = [gt["mask"] for gt in frames[1]["gt"]]

        class _CondModel(_StubModel):
            def __init__(self):
                super().__init__([gt0_masks, gt1_masks])
                self.frame_visits = {}

            def predict(self, image_bgr, *, conf, imgsz):
                call = len(self.calls)
                instances = super().predict(
                    image_bgr, conf=conf, imgsz=imgsz)
                # frames alternate per call; drop the last GT at conf 0.25
                if conf == 0.25 and imgsz == 640 and instances:
                    instances = instances[:-1]
                return instances

        model = _CondModel()

        def factory(path, conf, iou, imgsz, device):
            return model

        summary = run_detection_diagnosis(
            config=CONFIG,
            output_dir=scratch_dir / "detection",
            frames_provider=lambda: frames,
            model_factory=factory,
            historical=None,
            device="cpu",
        )
        per_gt_d0 = [
            json.loads(line)
            for line in (scratch_dir / "detection" / "conditions" / "D0"
                         / "per_gt.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        per_gt_d1 = [
            json.loads(line)
            for line in (scratch_dir / "detection" / "conditions" / "D1"
                         / "per_gt.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        # (frame_id, gt_id) identities match across conditions
        d0_ids = {(r["frame_id"], r["gt_id"]) for r in per_gt_d0}
        d1_ids = {(r["frame_id"], r["gt_id"]) for r in per_gt_d1}
        assert d0_ids == d1_ids == {("val_frame_000", 0),
                                    ("val_frame_000", 1),
                                    ("val_frame_001", 0)}
        # the GT that D0 missed is matched at the lower confidence, and the
        # attribution rule names it
        unmatched_d0 = [
            json.loads(line)
            for line in (scratch_dir / "detection" / "conditions" / "D0"
                         / "unmatched_gt.jsonl")
            .read_text(encoding="utf-8").splitlines()
        ]
        assert len(unmatched_d0) == 2  # frame0 GT1 + frame1 GT0 (0 preds)
        by_identity = {(r["frame_id"], r["gt_id"]): r
                       for r in unmatched_d0}
        # both are recovered by the lower-confidence conditions (D1/D2 run
        # the same scale with more predictions), so the sign is
        # confidence_related for both
        assert (by_identity[("val_frame_000", 1)]["attribution"]
                == "confidence_related")
        assert (by_identity[("val_frame_001", 0)]["attribution"]
                == "confidence_related")
        assert by_identity[("val_frame_000", 1)]["note"] == \
            ATTRIBUTION_SIGN_NOTE

    def test_module_state_unchanged_across_all_conditions(self, scratch_dir):
        frames = _frames_source()
        model = _perfect_model(frames)
        before = snapshot_module_state(model.model)

        def factory(path, conf, iou, imgsz, device):
            return model

        run_detection_diagnosis(
            config=CONFIG,
            output_dir=scratch_dir / "detection",
            frames_provider=lambda: frames,
            model_factory=factory,
            historical=None,
            device="cpu",
        )
        after = snapshot_module_state(model.model)
        assert compare_module_states(before, after)["unchanged"]
        frames = _frames_source()
        model = _perfect_model(frames)

        def factory(path, conf, iou, imgsz, device):
            return model

    def test_saved_masks_are_rebuildable(self, scratch_dir):
        frames = _frames_source()
        model = _perfect_model(frames)

        def factory(path, conf, iou, imgsz, device):
            return model

        run_detection_diagnosis(
            config=CONFIG,
            output_dir=scratch_dir / "detection",
            frames_provider=lambda: frames,
            model_factory=factory,
            historical=None,
            device="cpu",
        )
        mask_files = list((scratch_dir / "detection" / "conditions" / "D0"
                           / "masks").glob("*.npz"))
        assert len(mask_files) == 2
        with np.load(mask_files[0]) as payload:
            shape = tuple(int(v) for v in payload["shape"])
            packed = payload["masks"]
            confidences = payload["confidences"]
        reconstructed = np.unpackbits(packed, axis=1)[:, :shape[0] * shape[1]] \
            .reshape(-1, shape[0], shape[1])
        assert reconstructed.shape[1:] == (64, 80)
        assert reconstructed.shape[0] == confidences.shape[0]
