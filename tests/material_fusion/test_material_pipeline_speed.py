"""4B pipeline performance benchmark tests (TDD: written first).

Covers the performance contract:

- warmup iterations are excluded from the reported samples;
- the measurement count is enforced and reported;
- GPU stages are synchronized (a stub CUDA event/clock proves the sync
  call actually happens between the stage and the timestamp);
- per-stage diagnostics and the end-to-end throughput are computed
  separately (stage times are never summed into the total);
- target-count buckets come from the *actual* Model A predictions, never
  from the GT count; a missing natural scenario is reported as not covered
  instead of being synthesized silently;
- scenario selection is fixed before timing (never "whichever was fastest");
- the output directory refuses to overwrite; the report carries the
  required fields (p50/p95, throughput, peak VRAM, hardware, scope notes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class FakeClock:
    """A monotonic fake clock with an explicit sync marker."""

    def __init__(self):
        self.now = 0.0
        self.syncs = 0

    def perf_counter(self):
        return self.now

    def cuda_synchronize(self):
        self.syncs += 1
        self.now += 0.001        # synchronization itself costs time

    def advance(self, seconds):
        self.now += seconds


class TestStageTiming:
    def test_gpu_stage_times_are_taken_after_synchronization(self):
        from scripts.benchmark_material_pipeline import timed_stage

        clock = FakeClock()

        def stage():
            clock.advance(0.05)   # "GPU work"

        elapsed = timed_stage(
            stage, clock.perf_counter, clock.cuda_synchronize, device="cuda"
        )
        assert clock.syncs >= 1
        assert elapsed >= 0.05

    def test_cpu_stage_does_not_invoke_cuda_sync(self):
        from scripts.benchmark_material_pipeline import timed_stage

        clock = FakeClock()

        def stage():
            clock.advance(0.02)

        timed_stage(stage, clock.perf_counter, clock.cuda_synchronize, device="cpu")
        assert clock.syncs == 0

    def test_total_is_measured_separately_from_stage_sum(self):
        from scripts.benchmark_material_pipeline import summarise_repeats

        # Per-stage diagnostics carry synchronization overhead and the
        # stages overlap in the end-to-end pass: the stage sum is reported
        # for diagnosis but is never the total.
        stage_samples = {"model_a": [0.030, 0.031], "sgbm": [0.080, 0.081]}
        total_samples = [0.100, 0.102]
        report = summarise_repeats(stage_samples, total_samples)
        assert report["total"]["p50"] == pytest.approx(0.100)
        assert report["stage_sum_p50"] == pytest.approx(0.110)
        assert report["total"]["n"] == 2


class TestWarmup:
    def test_warmup_runs_are_excluded(self):
        from scripts.benchmark_material_pipeline import run_repeats

        calls = []

        def once():
            calls.append(len(calls))
            return len(calls)

        samples = run_repeats(once, warmup=3, repeats=5)
        assert len(samples) == 5
        # once() returns len(calls) after appending: the three warmup calls
        # return 1, 2, 3 and are excluded from the measured samples.
        assert samples == [4, 5, 6, 7, 8]
        assert len(calls) == 8

    def test_measurement_budget_stops_and_reports_partial(self):
        from scripts.benchmark_material_pipeline import run_repeats

        def slow():
            import time as _time

            _time.sleep(0.05)
            return 0.05

        report = run_repeats(
            slow, warmup=0, repeats=1000, max_seconds=0.2
        )
        assert report["completed"] < 1000
        assert report["status"] == "PARTIAL"


class TestScenarioBuckets:
    def test_scenario_survey_feeds_model_a_the_production_input(self):
        """Model A gets the 3-channel grayscale copy, never a 1-channel array.

        The first benchmark attempt crashed on exactly this contract
        (``expected ... 3 channels, but got 1``); this test pins it.
        """
        import numpy as np

        from scripts.benchmark_material_pipeline import _survey_frames

        class RecordingModelA:
            def __init__(self):
                self.shapes = []
                self.kwargs = []

            def predict(self, image, **kwargs):
                self.shapes.append(np.asarray(image).shape)
                self.kwargs.append(kwargs)
                return []

        model_a = RecordingModelA()
        engine = type("Engine", (), {"model_a": model_a})()
        gray = np.zeros((512, 640), dtype=np.uint8)
        rows = [{"output_stem": "frame_a", "group_name": "g"}]
        frames = _survey_frames(engine, rows, {"frame_a": (gray, gray)}, 640)
        assert frames[0]["predicted_instances"] == 0
        assert len(model_a.shapes) == 1
        assert model_a.shapes[0][2] == 3, (
            "the survey must pass the 3-channel grayscale copy"
        )
        assert model_a.kwargs[0] == {"imgsz": 640}, (
            "the survey must use the configured Model A imgsz"
        )

    def test_buckets_come_from_actual_predictions_not_gt(self):
        from scripts.benchmark_material_pipeline import bucket_by_predictions

        frames = [
            {"frame_id": "a", "predicted_instances": 0},
            {"frame_id": "b", "predicted_instances": 1},
            {"frame_id": "c", "predicted_instances": 1},
            {"frame_id": "d", "predicted_instances": 7},
        ]
        buckets = bucket_by_predictions(frames, wanted=(0, 1, 2, 5))
        assert buckets["0"] == ["a"]
        assert sorted(buckets["1"]) == ["b", "c"]
        assert buckets["2"] is None            # not covered, not synthesized
        assert buckets["5"] is None
        assert buckets["max_available"] == ["d"]

    def test_scenario_selection_is_fixed_before_timing(self):
        from scripts.benchmark_material_pipeline import bucket_by_predictions

        frames = [
            {"frame_id": "fast", "predicted_instances": 1},
            {"frame_id": "slow", "predicted_instances": 1},
        ]
        buckets = bucket_by_predictions(frames, wanted=(1,))
        assert buckets["1"] == ["fast", "slow"]   # both kept, no cherry-pick


class TestBenchmarkReport:
    def test_output_directory_refuses_to_overwrite(self, scratch_dir):
        from scripts.benchmark_material_pipeline import prepare_output_dir

        target = scratch_dir / "performance"
        target.mkdir()
        with pytest.raises(FileExistsError):
            prepare_output_dir(target)

    def test_percentiles_and_fields_are_present(self):
        from scripts.benchmark_material_pipeline import percentile_summary

        samples = [float(i) for i in range(1, 101)]
        summary = percentile_summary(samples)
        assert summary["n"] == 100
        assert summary["p50"] == pytest.approx(50.0)
        assert summary["p95"] == pytest.approx(95.0)
        assert summary["min"] == pytest.approx(1.0)
        assert summary["max"] == pytest.approx(100.0)

    def test_scope_notes_record_not_measured_items(self):
        from scripts.benchmark_material_pipeline import performance_scope

        scope = performance_scope(scope="offline-replay")
        assert scope["queue_latency"] == "NOT_MEASURED"
        assert scope["drop_rate"] == "NOT_MEASURED"
        assert "camera" in scope["excludes"]
        assert "GUI" in scope["excludes"]


# ---------------------------------------------------------------------------
# 4B fix round: measurement bookkeeping and honest stage naming


class TestMeasurementBookkeeping:
    def test_gpu_stage_syncs_before_and_after(self):
        from scripts.benchmark_material_pipeline import timed_stage

        clock = FakeClock()

        def stage():
            clock.advance(0.03)

        timed_stage(stage, clock.perf_counter, clock.cuda_synchronize,
                    device="cuda")
        assert clock.syncs >= 2, (
            "a device stage must be synchronized before the start timestamp "
            "and after the stage completes"
        )

    def test_stage_names_describe_what_was_measured(self):
        from scripts.benchmark_material_pipeline import stage_definitions

        names = stage_definitions()
        assert "classification_wrapper" in names
        assert names["classification_wrapper"]["includes"]
        assert "pure CMX forward" in names["classification_wrapper"]["excludes"]
        # stages the round did not isolate are declared, not implied
        for stage in ("crop_only", "host_to_device_only", "pure_forward_only"):
            assert names[stage]["status"] == "NOT_MEASURED"

    def test_selection_is_persisted_before_timing(self, scratch_dir):
        import json as _json

        from scripts.benchmark_material_pipeline import persist_scenario_selection

        selection = {
            "1": {"frame_id": "frame_a", "sha256": "a" * 64},
            "2": None,
        }
        path = persist_scenario_selection(scratch_dir, selection)
        assert path.name == "scenario_selection.json"
        payload = _json.loads(path.read_bytes().decode("utf-8"))
        assert payload["selection"]["1"]["frame_id"] == "frame_a"
        assert payload["selection"]["2"] is None
        assert "written_before_measurement" in payload

    def test_available_and_measured_frames_are_distinct(self):
        from scripts.benchmark_material_pipeline import bucket_coverage

        buckets = {"1": ["a", "b", "c"], "2": ["d"], "5": None}
        coverage = bucket_coverage(buckets, measured={"1": ["a"], "2": ["d"]})
        assert coverage["1"]["available_frames"] == 3
        assert coverage["1"]["measured_unique_frames"] == 1
        assert coverage["2"]["measured_unique_frames"] == 1
        assert coverage["5"]["status"] == "NOT_COVERED"

    def test_raw_repeat_arrays_are_saved(self, scratch_dir):
        import json as _json

        from scripts.benchmark_material_pipeline import save_raw_timings

        raw = {
            "1": {"e1": {"total": [0.10, 0.11], "stages": {"model_a": [0.02]}}},
        }
        path = save_raw_timings(scratch_dir, raw)
        payload = _json.loads(path.read_bytes().decode("utf-8"))
        assert payload["raw"]["1"]["e1"]["total"] == [0.10, 0.11]

    def test_partial_propagates_to_the_overall_status(self):
        from scripts.benchmark_material_pipeline import overall_status

        assert overall_status([{"status": "COMPLETED"},
                               {"status": "COMPLETED"}]) == "COMPLETED"
        assert overall_status([{"status": "COMPLETED"},
                               {"status": "PARTIAL"}]) == "PARTIAL"
        assert overall_status([{"status": "FAILED"},
                               {"status": "COMPLETED"}]) == "PARTIAL"
        assert overall_status([{"status": "NOT_COVERED"},
                               {"status": "COMPLETED"}]) == "PARTIAL"

    def test_mixed_fallback_is_not_applicable_for_a_single_instance(self):
        from scripts.benchmark_material_pipeline import mixed_fallback_status

        assert mixed_fallback_status(instance_count=1) == "NOT_APPLICABLE"
        assert mixed_fallback_status(instance_count=4) == "MEASURED"

    def test_model_a_receives_the_configured_imgsz(self):
        from scripts.benchmark_material_pipeline import model_a_predict_kwargs

        assert model_a_predict_kwargs(640) == {"imgsz": 640}
        assert model_a_predict_kwargs(None) == {}

    def test_residency_is_sequential_with_peak_and_reserved(self, scratch_dir):
        from scripts.benchmark_material_pipeline import residency_record

        record = residency_record(
            variant="e1", seed=2026,
            components=["model_a", "material_fusion_v2_mask"],
            loaded_vram_mb=180.0, peak_allocated_mb=210.0,
            peak_reserved_mb=260.0, released=True,
        )
        assert record["variant"] == "e1"
        assert record["peak_allocated_mb"] == 210.0
        assert record["peak_reserved_mb"] == 260.0
        assert record["released_before_next_engine"] is True
        assert record["components"] == ["model_a", "material_fusion_v2_mask"]

