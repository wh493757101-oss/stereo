"""Entry-level tests for the detection diagnosis fixes.

All inputs are synthetic; the real main() is exercised with monkeypatched
model/frames/baseline providers so no real weight, frame or forward is
involved. The timing tests verify synchronization calls with recording
stand-ins and never initialize CUDA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.diagnose_existing_detection import (
    measure_timing,
    rebuild_atlas_from_evidence,
    render_event_panel,
    run_detection_diagnosis,
)

from .test_existing_detection import (
    CONFIG,
    _Instance,
    _StubModel,
    _box_mask,
    _frames_source,
    _write_historical_evidence,
)


# ---------------------------------------------------------------------------
# Panel rendering: predictions must actually be drawn


class TestRenderEventPanel:
    def _image(self):
        return np.zeros((64, 80, 3), dtype=np.uint8)

    def test_predictions_drawn_in_red(self, scratch_dir):
        gt = _box_mask((64, 80), 4, 4, 20, 24)
        pred = _box_mask((64, 80), 8, 10, 26, 34)
        path = render_event_panel(
            self._image(), [{"mask": gt}],
            [{"mask": pred, "confidence": 0.42, "label": "p0"}],
            scratch_dir / "panel.png", "title", "note", margin=4)
        panel = np.imread(path) if hasattr(np, "imread") else None
        import cv2

        drawn = cv2.imread(str(path))
        assert drawn is not None
        # pure red (BGR 0,0,255) pixels exist -> the prediction contour
        assert int(((drawn[:, :, 2] == 255) & (drawn[:, :, 0] == 0)
                    & (drawn[:, :, 1] == 0)).sum()) > 0
        # pure green (BGR 0,200,0 or 0,255,0) pixels exist -> GT contour
        assert int(((drawn[:, :, 1] >= 200) & (drawn[:, :, 2] == 0)
                    & (drawn[:, :, 0] == 0)).sum()) > 0

    def test_no_prediction_no_red(self, scratch_dir):
        import cv2

        gt = _box_mask((64, 80), 4, 4, 20, 24)
        path = render_event_panel(
            self._image(), [{"mask": gt}], [],
            scratch_dir / "panel_no_pred.png", "title", "note", margin=4)
        drawn = cv2.imread(str(path))
        assert int(((drawn[:, :, 2] == 255) & (drawn[:, :, 0] == 0)).sum()) == 0


# ---------------------------------------------------------------------------
# Atlas rebuild from saved evidence


def _write_detection_evidence(evidence_dir: Path, frames, model_masks) -> None:
    """A minimal saved-condition tree (masks npz + the four jsonl files)."""
    import json as json_module

    condition_dir = evidence_dir / "conditions" / "D0"
    (condition_dir / "masks").mkdir(parents=True)
    prediction_rows = []
    per_gt_rows = []
    unmatched_gt_rows = []
    unmatched_pred_rows = []
    for frame_index, frame in enumerate(frames):
        masks = model_masks[frame_index]
        np.savez_compressed(
            condition_dir / "masks" / f"{frame['frame_id']}.npz",
            masks=np.packbits(
                np.stack([m > 0 for m in masks]).reshape(len(masks), -1),
                axis=1) if masks else np.zeros((0, 640), dtype=np.uint8),
            shape=np.asarray([64, 80]),
            confidences=np.asarray(
                [0.9 - 0.1 * i for i in range(len(masks))],
                dtype=np.float32),
            bboxes=np.asarray([[0, 0, 30, 30]] * len(masks), dtype=np.int32),
        )
        for gt in frame["gt"]:
            ious = [
                float((np.logical_and(m > 0, gt["mask"] > 0)).sum()
                      / (np.logical_or(m > 0, gt["mask"] > 0)).sum())
                for m in masks
            ] if masks else []
            matched = bool(ious) and max(ious) >= 0.5
            matched_index = ious.index(max(ious)) if matched else None
            per_gt_rows.append({
                "frame_id": frame["frame_id"],
                "gt_id": gt["object_index"],
                "class_name": gt["class_name"],
                "matched": matched,
                "match_iou": max(ious) if matched else None,
                "max_iou_any_prediction": max(ious) if ious else None,
            })
            if not matched:
                unmatched_gt_rows.append({
                    "frame_id": frame["frame_id"],
                    "group": frame["group"],
                    "gt_id": gt["object_index"],
                    "class_id": gt["class_id"],
                    "class_name": gt["class_name"],
                    "max_iou_any_prediction": max(ious) if ious else 0.0,
                    "predictions_in_frame": len(masks),
                    "attribution": "confidence_related" if matched is False
                    and ious and max(ious) == 0 else
                    "no_corresponding_prediction_sign",
                })
        for index, mask in enumerate(masks):
            record = {
                "frame_id": frame["frame_id"],
                "prediction_index": index,
                "confidence": 0.9 - 0.1 * index,
                "matched_gt_id": None,
                "max_iou_any_gt": None,
            }
            prediction_rows.append(record)
            if not any(
                    (np.logical_and(mask > 0, gt["mask"] > 0)).sum()
                    / (np.logical_or(mask > 0, gt["mask"] > 0)).sum() >= 0.5
                    for gt in frame["gt"]):
                unmatched_pred_rows.append({
                    "frame_id": frame["frame_id"],
                    "group": frame["group"],
                    "prediction_index": index,
                    "confidence": record["confidence"],
                    "max_iou_any_gt": 0.0,
                })
    for name, rows in (
            ("predictions.jsonl", prediction_rows),
            ("per_gt.jsonl", per_gt_rows),
            ("unmatched_gt.jsonl", unmatched_gt_rows),
            ("unmatched_predictions.jsonl", unmatched_pred_rows)):
        (condition_dir / name).write_text(
            "".join(json_module.dumps(r) + "\n" for r in rows),
            encoding="utf-8")


class TestRebuildAtlas:
    def test_rebuild_draws_predictions(self, scratch_dir):
        import cv2

        frames = _frames_source()
        # frame0: 2 predictions matching its 2 GT; frame1: 1 spurious
        # prediction overlapping nothing -> unmatched prediction
        model_masks = [
            [gt["mask"] for gt in frames[0]["gt"]],
            [_box_mask((64, 80), 40, 50, 60, 75)],
        ]
        evidence = scratch_dir / "evidence"
        _write_detection_evidence(evidence, frames, model_masks)
        report = rebuild_atlas_from_evidence(
            evidence, frames, scratch_dir / "atlas_out",
            controls_per_class=1, margin=4)
        assert report["status"] == "COMPLETED"
        assert report["panels"] > 0
        assert set(report["inputs"]) == {
            "predictions.jsonl", "per_gt.jsonl", "unmatched_gt.jsonl",
            "unmatched_predictions.jsonl"}
        assert all(len(d) == 64 for d in report["inputs"].values())
        atlas = scratch_dir / "atlas_out" / "atlas"
        unmatched_pred_panels = list(atlas.glob("unmatched_pred_*.png"))
        assert unmatched_pred_panels, "unmatched-prediction panel missing"
        drawn = cv2.imread(str(unmatched_pred_panels[0]))
        # the panel must show the prediction (red) AND the GT (green)
        assert int(((drawn[:, :, 2] == 255) & (drawn[:, :, 0] == 0)).sum()) > 0
        assert int(((drawn[:, :, 1] >= 200) & (drawn[:, :, 0] == 0)).sum()) > 0
        unmatched_gt_panels = list(atlas.glob("unmatched_gt_*.png"))
        assert unmatched_gt_panels

    def test_missing_masks_fail(self, scratch_dir):
        frames = _frames_source()
        evidence = scratch_dir / "evidence"
        _write_detection_evidence(evidence, frames, [[], []])
        import shutil as shutil_module

        shutil_module.rmtree(evidence / "conditions" / "D0" / "masks")
        report = rebuild_atlas_from_evidence(
            evidence, frames, scratch_dir / "atlas_out2")
        assert report["status"] in ("FAILED", "PARTIAL")
        assert report["problems"]


# ---------------------------------------------------------------------------
# Real main(): D0 failure stops everything; budget exhaustion is PARTIAL


class _CountingModel(_StubModel):
    def __init__(self, per_frame_masks):
        super().__init__(per_frame_masks)
        self.predict_calls = 0

    def predict(self, image_bgr, *, conf, imgsz):
        self.predict_calls += 1
        return super().predict(image_bgr, conf=conf, imgsz=imgsz)


@pytest.fixture()
def synthetic_main(monkeypatch, scratch_dir):
    """Patch the real main()'s providers to fully synthetic ones."""
    import scripts.diagnose_existing_detection as ded

    calls = {"models": 0, "timing": 0}

    class _Adapter:
        def __init__(self, per_frame_masks):
            self._stub = _CountingModel(per_frame_masks)
            # the torch module the runner snapshots for the no-change proof
            self.model = self._stub.model

        def predict(self, image_bgr, *, conf, imgsz):
            return self._stub.predict(image_bgr, conf=conf, imgsz=imgsz)

    def _factory(path, iou, device):
        calls["models"] += 1
        frames = _frames_source()
        perfect = [[gt["mask"] for gt in f["gt"]] for f in frames]
        if calls["models"] == 1:
            # the first construction serves the D0 condition; it injects one
            # extra false prediction in frame0 so the D0 counts differ from
            # the historical baseline
            perfect[0] = (perfect[0]
                          + [_box_mask((64, 80), 40, 50, 55, 70)])
        return _Adapter(perfect)

    monkeypatch.setattr(ded, "_ConditionedModelA", _factory)
    monkeypatch.setattr(ded, "load_experiment_config", lambda path: CONFIG)
    monkeypatch.setattr(ded, "verify_baseline_identities",
                        lambda config, root: [])
    monkeypatch.setattr(
        ded, "load_historical_d0",
        lambda config, root: {
            "source": "synthetic", "reference": None, "frames": 2,
            "gt_instances": 3, "predictions": 3, "matched": 3,
            "per_frame": {
                "val_frame_000": {"predictions": 2, "matched": 2,
                                  "gt_instances": 2},
                "val_frame_001": {"predictions": 1, "matched": 1,
                                  "gt_instances": 1},
            }})
    monkeypatch.setattr(ded, "real_frames_provider",
                        lambda config, root: _frames_source())

    def _timing_bomb(*args, **kwargs):
        calls["timing"] += 1
        raise AssertionError("timing must not run after a failed D0")

    monkeypatch.setattr(ded, "measure_timing", _timing_bomb)
    return ded, calls


class TestMainStopRules:
    def test_d0_failure_skips_timing_and_later_conditions(
            self, synthetic_main, scratch_dir):
        ded, calls = synthetic_main
        code = ded.main([
            "--config", "unused.yaml",
            "--output-dir", str(scratch_dir / "det_out"),
            "--device", "cpu",
        ])
        assert code == 2
        summary = json.loads(
            (scratch_dir / "det_out" / "summary.json")
            .read_text(encoding="utf-8"))
        assert summary["status"] == "FAILED_D0_REPRODUCTION"
        assert "D1" not in summary["conditions"]
        timing = json.loads(
            (scratch_dir / "det_out" / "timing.json").read_text(
                encoding="utf-8"))
        assert timing["enabled"] is False
        assert "not run" in timing["note"]
        # only the D0-condition model and its frame forwards happened
        assert calls["models"] == 1
        assert calls["timing"] == 0
        # atlas still produced from the SAVED evidence (no new forwards)
        assert (scratch_dir / "det_out" / "atlas").is_dir()

    def test_budget_exhaustion_is_partial(self, synthetic_main, scratch_dir,
                                          monkeypatch):
        ded, calls = synthetic_main
        from core.existing_data_protocol import BudgetWatch

        clock = {"now": 0.0}
        exhausted = BudgetWatch(1.0, clock=lambda: clock["now"])
        clock["now"] = 7200.0  # exhausted right after construction
        monkeypatch.setattr(ded, "BudgetWatch", lambda minutes: exhausted)
        code = ded.main([
            "--config", "unused.yaml",
            "--output-dir", str(scratch_dir / "det_out_budget"),
            "--device", "cpu",
        ])
        assert code == 2
        summary = json.loads(
            (scratch_dir / "det_out_budget" / "summary.json")
            .read_text(encoding="utf-8"))
        assert summary["status"] == "PARTIAL_BUDGET_EXCEEDED"
        assert summary["conditions"] == {}  # nothing ran
        timing = json.loads(
            (scratch_dir / "det_out_budget" / "timing.json")
            .read_text(encoding="utf-8"))
        assert timing["enabled"] is False
        assert calls["timing"] == 0


class TestTimingDevice:
    def _model(self):
        frames = _frames_source()
        return _StubModel([[gt["mask"] for gt in frames[0]["gt"]]])

    def _timing_config(self):
        config = json.loads(json.dumps(CONFIG))
        config["detection"]["timing"] = {"enabled": True, "warmup": 1,
                                         "repeats": 2}
        return config

    def test_cuda_device_is_normalized_and_synced(self, monkeypatch):
        import scripts.diagnose_existing_detection as ded

        calls = []
        monkeypatch.setattr("torch.cuda.synchronize",
                            lambda *a, **k: calls.append("sync"))
        report = measure_timing(
            self._model(), _frames_source(),
            __import__("scripts.diagnose_existing_detection",
                       fromlist=["build_condition_plan"]).build_condition_plan(
                self._timing_config()),
            self._timing_config(), "0")
        assert report["device_normalized"] == "cuda:0"
        assert report["synchronized"] is True
        assert calls  # the sync actually ran around the measured calls

    def test_cpu_never_touches_cuda(self, monkeypatch):
        import torch

        def _bomb(*args, **kwargs):
            raise AssertionError("the CPU path must not initialize CUDA")

        monkeypatch.setattr(torch.cuda, "synchronize", _bomb)
        monkeypatch.setattr(torch.cuda, "is_available", _bomb)
        report = measure_timing(
            self._model(), _frames_source(),
            __import__("scripts.diagnose_existing_detection",
                       fromlist=["build_condition_plan"]).build_condition_plan(
                self._timing_config()),
            self._timing_config(), "cpu")
        assert report["device_normalized"] == "cpu"
        assert report["synchronized"] is False

    def test_budget_exhausted_before_repeat(self, monkeypatch):
        import pytest as _pytest

        from core.existing_data_protocol import BudgetExceeded, BudgetWatch

        clock = {"now": 0.0}
        budget = BudgetWatch(1.0, clock=lambda: clock["now"])
        clock["now"] = 7200.0
        plan = __import__("scripts.diagnose_existing_detection",
                          fromlist=["build_condition_plan"]) \
            .build_condition_plan(self._timing_config())
        with _pytest.raises(BudgetExceeded):
            measure_timing(self._model(), _frames_source(), plan,
                           self._timing_config(), "cpu", budget=budget)
