"""Acceptance-round tests: the closeout must reject failure inputs.

Every case here is a counterexample that the previous implementation let
through: a dtype-mismatched or non-finite tensor comparison, a historical
row whose recorded fields contradict each other, an admission step that
certified itself from the outputs under review, and a final status/exit
code that ignored the verification it had just run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Strict tensor comparison on the real call path


class TestSummarySupportCount:
    """Direct coverage checks for the not-yet-wired summary support input."""

    @staticmethod
    def _check(predictions, expected_frames, supported, *, extra_id=None):
        from scripts.closeout_material_pipeline import verify_instance_coverage

        records = []
        for condition in ("pure", "routed"):
            records.append({
                "variant": "gray", "seed": 2026,
                "layer": "A2_gt_controlled", "condition": condition,
                "frame_id": "frame_a", "gt_instance_id": 0,
            })
            for frame_id, count in predictions.items():
                for index in range(count):
                    records.append({
                        "variant": "gray", "seed": 2026,
                        "layer": "B_model_a_predictions",
                        "condition": condition, "frame_id": frame_id,
                        "pred_instance_id": index,
                    })
            if extra_id is not None:
                records.append({
                    "variant": "gray", "seed": 2026,
                    "layer": "B_model_a_predictions", "condition": condition,
                    "frame_id": "frame_a", "pred_instance_id": extra_id,
                })
        return verify_instance_coverage(
            records, candidates=[("gray", 2026)],
            expected_a2_keys=[("frame_a", 0)],
            frame_predictions=predictions,
            expected_frame_ids=expected_frames,
            summary_supports={"gray_s2026": supported},
        )

    @pytest.mark.parametrize("count,supported,status", [
        (1, 1, "PASSED"), (1, 2, "FAILED"),
        (0, 0, "PASSED"), (0, 1, "FAILED"),
    ])
    def test_complete_evidence_compares_total(self, count, supported, status):
        result = self._check({"frame_a": count}, ["frame_a"], supported)
        assert result["status"] == status
        if status == "FAILED":
            assert any("summary reports" in reason
                       for reason in result["failures"])
        else:
            assert result["failures"] == []

    @pytest.mark.parametrize("predictions,expected_frames,supported", [
        ({"frame_a": 1}, ["frame_a", "frame_b"], 1),
        ({"frame_a": 1}, ["frame_a", "frame_b"], 2),
        ({}, ["frame_a"], 0),
        ({"frame_a": 1}, None, 1),
    ])
    def test_incomplete_evidence_cannot_compare_total(
        self, predictions, expected_frames, supported
    ):
        result = self._check(predictions, expected_frames, supported)
        assert result["status"] == "PARTIAL"
        assert result["failures"] == []
        assert any("summary" in reason and "cannot be verified" in reason
                   for reason in result["partials"])

    def test_incomplete_total_does_not_hide_known_frame_contradiction(self):
        result = self._check(
            {"frame_a": 1}, ["frame_a", "frame_b"], 2, extra_id=99,
        )
        assert result["status"] == "FAILED"
        assert any("out-of-range" in reason for reason in result["failures"])


class TestStrictTensorComparison:
    def test_identical_float32_tensors_pass(self):
        from scripts.eval_material_pipeline import compare_tensors

        tensor = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        result = compare_tensors(tensor, tensor.clone())
        assert result["status"] == "PASSED"
        assert result["exact_equal"] is True

    def test_dtype_mismatch_is_rejected_even_when_values_match(self):
        from scripts.eval_material_pipeline import compare_tensors

        result = compare_tensors(torch.tensor([1.0], dtype=torch.float32),
                                 torch.tensor([1.0], dtype=torch.float64))
        assert result["status"] == "FAILED"
        assert result["reason"] == "dtype mismatch"

    def test_non_finite_tensors_are_rejected(self):
        from scripts.eval_material_pipeline import compare_tensors

        for value in (float("inf"), float("-inf"), float("nan")):
            tensor = torch.tensor([value], dtype=torch.float32)
            result = compare_tensors(tensor, tensor.clone())
            assert result["status"] == "FAILED", value
            assert result["reason"] == "non-finite values"

    def test_shape_mismatch_and_empty_tensors_are_rejected(self):
        from scripts.eval_material_pipeline import compare_tensors

        assert compare_tensors(torch.zeros(2, 2), torch.zeros(2, 3))["status"] \
            == "FAILED"
        empty = torch.zeros(0, dtype=torch.float32)
        result = compare_tensors(empty, empty.clone())
        assert result["status"] == "FAILED"
        assert result["reason"] == "empty required input"

    def test_differing_values_report_a_finite_error(self):
        from scripts.eval_material_pipeline import compare_tensors

        result = compare_tensors(torch.tensor([1.0, 2.0]),
                                 torch.tensor([1.0, 3.0]))
        assert result["status"] == "FAILED"
        assert result["max_abs_err"] == pytest.approx(1.0)

    def test_missing_tensor_stays_unavailable(self):
        from scripts.eval_material_pipeline import compare_tensors

        result = compare_tensors(None, torch.zeros(1))
        assert result["status"] == "UNAVAILABLE"


class TestStrictArrayComparison:
    def test_non_finite_arrays_fail_without_a_nan_error(self):
        from scripts.eval_material_pipeline import _compare_arrays

        array = np.array([np.inf], dtype=np.float32)
        result = _compare_arrays(array, array.copy(), kind="exact")
        assert result["status"] == "FAILED"
        assert result["max_abs_err"] is None, "no NaN may reach the report"
        assert json.dumps(result)  # serialisable without NaN

    def test_empty_required_arrays_are_rejected(self):
        from scripts.eval_material_pipeline import _compare_arrays

        empty = np.zeros((0, 0), dtype=np.float32)
        assert _compare_arrays(empty, empty.copy(), kind="exact")["status"] \
            == "FAILED"


# ---------------------------------------------------------------------------
# Historical-row consistency


class TestHistoricalRowConsistency:
    def _checks(self, **overrides):
        check = {"status": "PASSED", "exact_equal": True, "max_abs_err": 0.0}
        names = (
            "gray", "instance_mask", "polar_valid", "signed_q", "abs_q",
            "quality", "gray_rgb", "gray_mask", "gray_logits",
            "gray_gray_valid_rgb", "gray_gray_valid_aux", "gray_gray_valid_mask",
            "gray_gray_valid_logits", "gray_valid_rgb", "gray_valid_aux",
            "gray_valid_mask", "gray_valid_logits", "gray_polar_rgb",
            "gray_polar_aux", "gray_polar_mask", "gray_polar_logits",
        )
        checks = {name: dict(check) for name in names}
        checks.update(overrides)
        return checks

    def _row(self, checks, status="PASSED"):
        return {"frame_id": "f", "object_index": 0, "status": status,
                "npz_path": "x.npz", "checks": checks}

    def test_exact_check_with_a_recorded_error_is_a_contradiction(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        checks = self._checks(signed_q={"status": "PASSED", "exact_equal": False,
                                        "max_abs_err": 5e-7})
        verdict = evaluate_alignment([self._row(checks)],
                                     expected_keys=[("f", 0)])
        assert verdict["status"] == "FAILED"
        assert any("signed_q" in str(m.get("field")) for m in verdict["mismatches"])

    def test_exact_check_with_a_nonzero_error_is_a_contradiction(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        checks = self._checks(abs_q={"status": "PASSED", "exact_equal": True,
                                     "max_abs_err": 1e-9})
        verdict = evaluate_alignment([self._row(checks)],
                                     expected_keys=[("f", 0)])
        assert verdict["status"] == "FAILED"

    def test_tolerated_kinds_may_differ_within_their_tolerance(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        checks = self._checks(
            quality={"status": "PASSED", "exact_equal": False,
                     "max_abs_err": 1e-6, "kind": "quality"},
            gray_polar_logits={"status": "PASSED", "exact_equal": False,
                               "max_abs_err": 1e-5, "kind": "logits"},
        )
        verdict = evaluate_alignment([self._row(checks)],
                                     expected_keys=[("f", 0)])
        assert verdict["status"] == "PASSED"

    def test_row_level_unavailable_is_not_a_pass(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        verdict = evaluate_alignment([self._row(self._checks(), status="UNAVAILABLE")],
                                     expected_keys=[("f", 0)])
        assert verdict["status"] == "PARTIAL"

    def test_unknown_check_status_is_rejected(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        checks = self._checks(gray={"status": "MAYBE", "exact_equal": True,
                                    "max_abs_err": 0.0})
        verdict = evaluate_alignment([self._row(checks)],
                                     expected_keys=[("f", 0)])
        assert verdict["status"] in ("FAILED", "PARTIAL")
        assert verdict["status"] != "PASSED"
        assert verdict["unknown_check_status"]


# ---------------------------------------------------------------------------
# Admission cannot be self-certified


class TestAdmissionIsNotSelfCertified:
    def test_empty_manifest_mapping_is_refused(self, scratch_dir):
        from scripts.closeout_material_pipeline import verify_source

        source = scratch_dir / "source"
        (source / "quality").mkdir(parents=True)
        (source / "performance").mkdir(parents=True)
        (source / "quality" / "artifact_manifest.json").write_text(
            json.dumps({"files": {}}), encoding="utf-8")
        (source / "performance" / "artifact_manifest.json").write_text(
            json.dumps({"files": {}}), encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError)):
            verify_source(source)

    def test_consumed_file_without_a_digest_is_refused(self, scratch_dir):
        from scripts.closeout_material_pipeline import verify_source

        source = scratch_dir / "source"
        for sub in ("quality", "performance"):
            (source / sub).mkdir(parents=True, exist_ok=True)
        summary = source / "quality" / "summary.json"
        summary.write_bytes(b"{}\n")
        (source / "quality" / "artifact_manifest.json").write_text(
            json.dumps({"files": {"other.json": "0" * 64}}), encoding="utf-8")
        (source / "performance" / "artifact_manifest.json").write_text(
            json.dumps({"files": {}}), encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError)):
            verify_source(source)

    def test_escaping_manifest_path_is_refused(self, scratch_dir):
        from scripts.closeout_material_pipeline import verify_source

        source = scratch_dir / "source"
        for sub in ("quality", "performance"):
            (source / sub).mkdir(parents=True, exist_ok=True)
        (source / "quality" / "artifact_manifest.json").write_text(
            json.dumps({"files": {"../outside.json": "0" * 64}}),
            encoding="utf-8")
        (source / "performance" / "artifact_manifest.json").write_text(
            json.dumps({"files": {}}), encoding="utf-8")
        with pytest.raises((ValueError, FileNotFoundError)):
            verify_source(source)


class TestIndependentExpectedSet:
    def test_expected_keys_come_from_the_gt_index_not_the_outputs(self):
        from scripts.closeout_material_pipeline import expected_keys_from_sources

        gt_index = [
            {"frame_id": "frame_a", "instances": [{"object_index": 0},
                                                  {"object_index": 1}]},
            {"frame_id": "frame_b", "instances": [{"object_index": 0}]},
        ]
        keys, without_npz = expected_keys_from_sources(
            gt_index, npz_keys={("frame_a", 0), ("frame_a", 1), ("frame_b", 0)})
        assert keys == [("frame_a", 0), ("frame_a", 1), ("frame_b", 0)]
        assert without_npz == []

    def test_gt_without_an_npz_is_kept_and_marked(self):
        from scripts.closeout_material_pipeline import expected_keys_from_sources

        gt_index = [{"frame_id": "frame_a",
                     "instances": [{"object_index": 0}, {"object_index": 7}]}]
        keys, without_npz = expected_keys_from_sources(
            gt_index, npz_keys={("frame_a", 0)})
        assert without_npz == [("frame_a", 7)]
        assert ("frame_a", 7) in keys, (
            "a GT without an NPZ must stay in the expected set"
        )

    def test_dropping_the_same_gt_from_both_outputs_is_still_detected(self):
        from scripts.closeout_material_pipeline import evaluate_alignment

        # Both the A2 records and the alignment rows omit ("frame_b", 0):
        # the independent expected set still catches the missing sample.
        rows = [{"frame_id": "frame_a", "object_index": 0, "status": "PASSED",
                 "checks": {"gray": {"status": "PASSED", "exact_equal": True,
                                     "max_abs_err": 0.0}}}]
        verdict = evaluate_alignment(
            rows, expected_keys=[("frame_a", 0), ("frame_b", 0)]
        )
        assert verdict["status"] != "PASSED"
        assert verdict["coverage"]["missing"] == ["frame_b|0"]

    def test_candidate_identity_conflicts_are_refused(self):
        from scripts.closeout_material_pipeline import verify_candidates

        good = {"candidates": [
            {"variant": variant, "seed": seed, "sha256": "a" * 64}
            for variant in ("gray", "gray_gray_valid", "gray_valid", "gray_polar")
            for seed in (2026, 2027, 2028)
        ]}
        shas = {(c["variant"], c["seed"]): "a" * 64
                for c in good["candidates"]}
        verify_candidates(good["candidates"], expected_shas=shas)
        with pytest.raises(ValueError, match="missing"):
            verify_candidates(good["candidates"][:-1], expected_shas=shas)
        with pytest.raises(ValueError, match="duplicate"):
            verify_candidates(good["candidates"] + [good["candidates"][0]],
                              expected_shas={
                (c["variant"], c["seed"]): "a" * 64
                for c in good["candidates"]})
        with pytest.raises(ValueError, match="sha256"):
            verify_candidates(good["candidates"], expected_shas={
                (c["variant"], c["seed"]): "b" * 64
                for c in good["candidates"]})


# ---------------------------------------------------------------------------
# Consistency checks compare source against corrected output


class TestConsistencyChecksAreReal:
    def _fixture(self, scratch_dir):
        source = scratch_dir / "source"
        (source / "quality").mkdir(parents=True)
        (source / "performance").mkdir(parents=True)
        metrics = {"macro_f1": 0.5, "counts": {"predictions": 2, "matched": 1}}
        blocks = {"layer_a2": {"pure": dict(metrics), "routed": dict(metrics)},
                  "layer_b": {"pure": dict(metrics), "routed": dict(metrics)}}
        summary = {
            "status": "COMPLETED",
            "per_candidate": {
                "gray_polar_s2026": dict(blocks),
                "gray_s2026": dict(blocks),
            },
            "bootstrap": {"end_to_end_macro_f1_e4_vs_e1": {
                "seed_2026": {"point_estimate": -0.0035, "ci_low": -0.02,
                              "ci_high": 0.08}}},
            "paired_changes": [{"pair": "a_vs_b", "fixed": 1, "broke": 0}],
        }
        (source / "quality" / "summary.json").write_bytes(
            (json.dumps(summary) + "\n").encode("utf-8"))
        (source / "performance" / "performance_report.json").write_bytes(
            (json.dumps({"status": "PARTIAL", "scenarios": {}}) + "\n").encode())
        (source / "performance" / "raw_timings.json").write_bytes(
            (json.dumps({"raw": {"1": {"gray": {"total": [0.1, 0.11]}}}})
             + "\n").encode("utf-8"))
        return source

    def test_a_changed_metric_is_detected(self, scratch_dir):
        """Uses the complete shared fixture; production structure is enforced."""
        from scripts.closeout_material_pipeline import (
            compare_source_and_corrected,
        )
        from .pipeline_synthetic import build_synthetic_source

        fixture = build_synthetic_source(scratch_dir / "synthetic")
        summary = json.loads((fixture["source"] / "quality"
                              / "summary.json").read_bytes().decode())
        timings = json.loads((fixture["source"] / "performance"
                              / "raw_timings.json").read_bytes().decode())
        renamed = json.loads(json.dumps(timings))
        stages = renamed["raw"]["1"]["gray"]["stages"]
        stages["cpu_tensor_preparation"] = stages.pop("crop_preprocess_cpu")
        required = [f"{c['variant']}_s{c['seed']}"
                    for c in fixture["candidates"]]
        corrected = json.loads(json.dumps(summary))
        corrected["per_candidate"][required[0]]["layer_b"]["routed"][
            "end_to_end"]["macro_f1"] = 0.25
        result = compare_source_and_corrected(
            summary, corrected, source_timings=timings,
            corrected_timings=renamed, required_candidates=required,
        )
        assert result["status"] == "FAILED"
        assert any("macro_f1" in entry for entry in result["mismatches"])
        assert result["compared_metric_blocks"] == 48

    def test_a_changed_bootstrap_or_timing_is_detected(self, scratch_dir):
        from scripts.closeout_material_pipeline import compare_source_and_corrected

        source = self._fixture(scratch_dir)
        source_summary = json.loads(
            (source / "quality" / "summary.json").read_bytes())
        corrected = json.loads(json.dumps(source_summary))
        corrected["bootstrap"]["end_to_end_macro_f1_e4_vs_e1"]["seed_2026"][
            "ci_high"] = 0.5
        timings = json.loads(
            (source / "performance" / "raw_timings.json").read_bytes())
        corrected_timings = json.loads(json.dumps(timings))
        corrected_timings["raw"]["1"]["gray"]["total"] = [0.2, 0.22]
        result = compare_source_and_corrected(
            source_summary, corrected, source_timings=timings,
            corrected_timings=corrected_timings,
        )
        assert len(result["mismatches"]) >= 2

    def test_a_dropped_candidate_is_detected(self, scratch_dir):
        from scripts.closeout_material_pipeline import compare_source_and_corrected

        source = self._fixture(scratch_dir)
        source_summary = json.loads(
            (source / "quality" / "summary.json").read_bytes())
        corrected = json.loads(json.dumps(source_summary))
        corrected["per_candidate"].pop("gray_s2026")
        result = compare_source_and_corrected(
            source_summary, corrected, source_timings={}, corrected_timings={})
        assert any("gray_s2026" in entry for entry in result["mismatches"])

    def test_an_unchanged_pair_reports_no_mismatch(self, scratch_dir):
        """Uses the complete legal shared fixture, not a two-candidate stub."""
        from scripts.closeout_material_pipeline import (
            compare_source_and_corrected,
        )
        from .pipeline_synthetic import build_synthetic_source

        fixture = build_synthetic_source(scratch_dir / "synthetic")
        summary = json.loads((fixture["source"] / "quality"
                              / "summary.json").read_bytes().decode())
        timings = json.loads((fixture["source"] / "performance"
                              / "raw_timings.json").read_bytes().decode())
        required = [f"{c['variant']}_s{c['seed']}"
                    for c in fixture["candidates"]]
        # The corrected timings carry the approved stage rename; nothing else
        # may differ.
        corrected_timings = json.loads(json.dumps(timings))
        stages = corrected_timings["raw"]["1"]["gray"]["stages"]
        stages["cpu_tensor_preparation"] = stages.pop("crop_preprocess_cpu")
        result = compare_source_and_corrected(
            summary, json.loads(json.dumps(summary)),
            source_timings=timings, corrected_timings=corrected_timings,
            required_candidates=required,
        )
        assert result["mismatches"] == []
        assert result["compared_metric_blocks"] == 48
        assert result["compared_candidates"] == 12
        assert result["bootstrap_seeds"] == ["seed_2026", "seed_2027",
                                             "seed_2028"]

    def test_missing_candidate_and_missing_block_are_refused(self, scratch_dir):
        from scripts.closeout_material_pipeline import (
            compare_source_and_corrected,
        )
        from .pipeline_synthetic import build_synthetic_source

        fixture = build_synthetic_source(scratch_dir / "synthetic")
        summary = json.loads((fixture["source"] / "quality"
                              / "summary.json").read_bytes().decode())
        timings = json.loads((fixture["source"] / "performance"
                              / "raw_timings.json").read_bytes().decode())
        required = [f"{c['variant']}_s{c['seed']}"
                    for c in fixture["candidates"]]
        renamed = json.loads(json.dumps(timings))
        renamed["raw"]["1"]["gray"]["stages"]["cpu_tensor_preparation"] = (
            renamed["raw"]["1"]["gray"]["stages"].pop("crop_preprocess_cpu"))
        dropped = json.loads(json.dumps(summary))
        dropped["per_candidate"].pop(required[0])
        result = compare_source_and_corrected(
            dropped, json.loads(json.dumps(dropped)),
            source_timings=timings, corrected_timings=renamed,
            required_candidates=required,
        )
        assert any("missing candidates" in entry
                   for entry in result["mismatches"])

        hollow = json.loads(json.dumps(summary))
        hollow["per_candidate"][required[0]]["layer_b"]["routed"] = {}
        result = compare_source_and_corrected(
            hollow, json.loads(json.dumps(hollow)),
            source_timings=timings, corrected_timings=renamed,
            required_candidates=required,
        )
        assert any("missing or empty" in entry for entry in result["mismatches"])
        assert result["status"] == "FAILED"

    def test_recursive_diff_is_covered_by_its_own_unit_test(self):
        from scripts.closeout_material_pipeline import _diff_paths

        assert _diff_paths({"a": 1}, {"a": 1}) == []
        assert _diff_paths({"a": 1}, {"a": 2}) == ["a: 1 -> 2"]
        assert _diff_paths({"a": 1}, {}) == ["a: missing from the corrected report"]
