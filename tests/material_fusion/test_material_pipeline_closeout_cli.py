"""CLI recovery tests: a complete legal synthetic source must pass through the
real ``main()``, and each single-factor corruption must be refused with the
right reason, status and exit code.

No model, no engine and no forward runs: the model factories are replaced by
"fail if called" probes to prove the entry never builds one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .pipeline_synthetic import (
    build_synthetic_source,
    perturb_alignment_row,
)


@pytest.fixture()
def legal(scratch_dir, monkeypatch):
    """A complete legal synthetic source plus fail-if-called model probes."""
    fixture = build_synthetic_source(scratch_dir / "synthetic")

    import gui.inference_engine as engine_module
    import models.material_classifier as classifier_module

    def _forbidden(*args, **kwargs):
        raise AssertionError("the closeout must not build or load a model")

    monkeypatch.setattr(engine_module.DualStageInferenceEngine,
                        "from_material_config",
                        classmethod(lambda cls, *a, **k: _forbidden()))
    monkeypatch.setattr(classifier_module,
                        "build_material_classifier_for_candidate", _forbidden)
    return fixture


def _run(fixture, output: Path):
    from scripts import closeout_material_pipeline as closeout

    code = closeout.main([
        "--source-dir", str(fixture["source"]),
        "--config", str(fixture["config"]),
        "--gt-index", str(fixture["gt_index"]),
        "--output-dir", str(output),
    ])
    summary = json.loads(
        (output / "quality_summary_corrected.json").read_bytes().decode())
    markdown = (output / "correction_summary.md").read_bytes().decode()
    return code, summary, markdown


class TestLegalSource:
    def test_complete_legal_evidence_completes_with_exit_zero(
        self, legal, scratch_dir
    ):
        code, summary, markdown = _run(legal, scratch_dir / "out")
        assert code == 0, summary.get("consistency_checks", {}).get("mismatches")
        assert summary["status"] == "COMPLETED"
        assert "COMPLETED" in markdown
        assert summary["source_status"] == "COMPLETED"
        assert summary["alignment"]["status"] == "PASSED"
        assert summary["alignment"]["covered_samples"] == 4
        assert summary["admission"]["candidate_check"]["candidates"] == 12
        assert summary["consistency_checks"]["compared_metric_blocks"] == 48
        assert summary["consistency_checks"]["mismatches"] == []
        assert summary["consistency_checks"]["performance_status_preserved"] \
            == "PARTIAL"
        assert summary["failed_attempt_alignment"]["status"] == "FAILED"

    def test_written_report_matches_its_manifest(self, legal, scratch_dir):
        import hashlib

        output = scratch_dir / "out"
        _run(legal, output)
        manifest = json.loads(
            (output / "artifact_manifest.json").read_bytes().decode())
        for name, recorded in manifest["files"].items():
            actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
            assert actual == recorded, name

    def test_performance_stage_rename_keeps_every_value(
        self, legal, scratch_dir
    ):
        output = scratch_dir / "out"
        _run(legal, output)
        corrected = json.loads(
            (output / "performance_report_corrected.json").read_bytes().decode())
        stages = corrected["scenarios"]["1"]["variants"]["gray"]["stages"]
        assert "cpu_tensor_preparation" in stages
        assert stages["cpu_tensor_preparation"]["p50"] == 0.002
        definitions = corrected["conditions"]["stage_definitions"][
            "cpu_tensor_preparation"]
        assert "ALREADY CROPPED" in definitions["includes"]
        assert "crop-only" in definitions["excludes"]
        assert definitions["renamed_from"] == "crop_preprocess_cpu"
        assert corrected["status"] == "PARTIAL"

    def test_output_refusals(self, legal, scratch_dir):
        from scripts import closeout_material_pipeline as closeout

        existing = scratch_dir / "existing"
        existing.mkdir()
        with pytest.raises(FileExistsError):
            closeout.main(["--source-dir", str(legal["source"]),
                           "--config", str(legal["config"]),
                           "--gt-index", str(legal["gt_index"]),
                           "--output-dir", str(existing)])
        with pytest.raises(ValueError):
            closeout.main(["--source-dir", str(legal["source"]),
                           "--config", str(legal["config"]),
                           "--gt-index", str(legal["gt_index"]),
                           "--output-dir", str(legal["source"] / "nested")])


class TestSingleFactorCorruptions:
    def test_alignment_failure_is_refused(self, legal, scratch_dir):
        perturb_alignment_row(legal["source"], index=0, status="FAILED")
        code, summary, markdown = _run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert summary["alignment"]["status"] == "FAILED"
        assert "FAILED" in markdown

    def test_incomplete_alignment_is_partial(self, legal, scratch_dir):
        perturb_alignment_row(legal["source"], index=0, status="NO_NPZ",
                              drop_checks=True)
        code, summary, markdown = _run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "PARTIAL"
        assert "PARTIAL" in markdown

    def test_empty_manifest_is_refused_for_its_content(self, legal):
        from scripts.closeout_material_pipeline import verify_source

        (legal["source"] / "quality" / "artifact_manifest.json").write_bytes(
            (json.dumps({"files": {}}) + "\n").encode("utf-8"))
        with pytest.raises(ValueError, match="non-empty mapping"):
            verify_source(legal["source"])

    def test_missing_digest_for_a_consumed_file_is_refused(self, legal):
        from scripts.closeout_material_pipeline import verify_source

        manifest_path = legal["source"] / "quality" / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_bytes().decode())
        manifest["files"].pop("instances.jsonl")
        manifest_path.write_bytes((json.dumps(manifest) + "\n").encode("utf-8"))
        with pytest.raises(ValueError, match="not covered by any digest"):
            verify_source(legal["source"])

    def test_wrong_digest_is_refused(self, legal):
        from scripts.closeout_material_pipeline import verify_source

        manifest_path = legal["source"] / "quality" / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_bytes().decode())
        manifest["files"]["summary.json"] = "0" * 64
        manifest_path.write_bytes((json.dumps(manifest) + "\n").encode("utf-8"))
        with pytest.raises(ValueError, match="digest mismatch"):
            verify_source(legal["source"])

    def test_escaping_manifest_path_is_refused(self, legal):
        from scripts.closeout_material_pipeline import verify_source

        manifest_path = legal["source"] / "quality" / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_bytes().decode())
        manifest["files"]["../outside.json"] = "a" * 64
        manifest_path.write_bytes((json.dumps(manifest) + "\n").encode("utf-8"))
        with pytest.raises(ValueError, match="escaping|outside"):
            verify_source(legal["source"])

    def test_corrupted_failed_attempt_is_refused(self, legal):
        from scripts.closeout_material_pipeline import verify_source

        (legal["source"] / "quality_attempt1_alignment_failed"
         / "a2_alignment.jsonl").write_bytes(b"{not json\n")
        with pytest.raises(ValueError, match="digest mismatch"):
            verify_source(legal["source"])

    def test_candidate_identity_conflicts_are_refused(self, legal):
        from models.material_classifier import load_material_runtime_config
        from scripts.closeout_material_pipeline import verify_candidates

        snapshot = json.loads((legal["source"] / "quality"
                               / "protocol_snapshot.json").read_bytes().decode())
        shas = {(entry["variant"], entry["seed"]): entry["sha256"]
                for entry in snapshot["model_identities"]}
        config = load_material_runtime_config(legal["config"])
        records = [{"variant": c.variant, "seed": c.seed, "sha256": c.sha256}
                   for c in config.candidates]
        verify_candidates(records, expected_shas=shas)
        with pytest.raises(ValueError, match="missing candidates"):
            verify_candidates(records[:-1], expected_shas=shas)
        with pytest.raises(ValueError, match="duplicate"):
            verify_candidates(records + [records[0]], expected_shas=shas)
        with pytest.raises(ValueError, match="sha256"):
            verify_candidates(records,
                              expected_shas={key: "b" * 64 for key in shas})

    def test_gt_without_npz_is_kept_and_listed(self):
        from scripts.closeout_material_pipeline import (
            expected_keys_from_sources,
        )

        gt = [{"frame_id": "frame_a",
               "instances": [{"object_index": 0}, {"object_index": 9}]}]
        keys, without_npz = expected_keys_from_sources(
            gt, npz_keys={("frame_a", 0)})
        assert keys == [("frame_a", 0), ("frame_a", 9)]
        assert without_npz == [("frame_a", 9)]

    def test_gt_missing_from_both_outputs_is_still_detected(self):
        from scripts.eval_material_pipeline import evaluate_alignment

        rows = [{"frame_id": "frame_a", "object_index": 0, "status": "PASSED",
                 "checks": {}}]
        verdict = evaluate_alignment(
            rows, expected_keys=[("frame_a", 0), ("frame_b", 0)])
        assert verdict["coverage"]["missing"] == ["frame_b|0"]
        assert verdict["status"] != "PASSED"


class TestBIdentityCoverage:
    """Equal-count identity substitution must be refused.

    The B records are replaced one at a time with an unknown frame or an
    out-of-range instance index; the record count and every other field stay
    legal, so only an identity-set comparison can catch it.
    """

    def _rewrite(self, fixture, mutate):
        import json as _json
        from .pipeline_synthetic import refresh_manifest

        path = fixture["source"] / "quality" / "instances.jsonl"
        rows = [_json.loads(line) for line in path.read_bytes().decode()
                .splitlines()]
        mutate(rows)
        path.write_bytes("".join(_json.dumps(row, sort_keys=True) + "\n"
                                 for row in rows).encode("utf-8"))
        refresh_manifest(fixture["source"] / "quality", [
            "summary.json", "protocol_snapshot.json", "instances.jsonl",
            "a2_alignment_summary.json", "a2_alignment.jsonl",
            "unmatched_gt.jsonl", "frames.jsonl"])

    @pytest.mark.parametrize("condition", ["pure", "routed"])
    @pytest.mark.parametrize("mutation", ["unknown_frame", "out_of_range",
                                          "duplicate_identity"])
    def test_equal_count_identity_errors_are_refused(
        self, legal, scratch_dir, condition, mutation
    ):
        def mutate(rows):
            target = next(
                index for index, row in enumerate(rows)
                if row["layer"] == "B_model_a_predictions"
                and row["condition"] == condition
            )
            if mutation == "unknown_frame":
                rows[target]["frame_id"] = "nonexistent_frame"
            elif mutation == "out_of_range":
                rows[target]["pred_instance_id"] = 999
            else:
                # A real duplicate: the second record of the same candidate
                # and condition copies the first one's (frame, index).
                sibling = next(
                    index for index in range(target + 1, len(rows))
                    if rows[index]["layer"] == "B_model_a_predictions"
                    and rows[index]["condition"] == condition
                    and rows[index]["variant"] == rows[target]["variant"]
                    and rows[index]["seed"] == rows[target]["seed"]
                )
                rows[sibling]["pred_instance_id"] = rows[target][
                    "pred_instance_id"]

        self._rewrite(legal, mutate)
        code, summary, _markdown = _run(legal, scratch_dir / "out")
        assert code != 0, f"{condition}/{mutation} must not complete"
        assert summary["status"] == "FAILED"
        coverage = summary["instance_coverage"]
        assert coverage["failures"], "the identity error must be reported"


class TestIllegalBInstanceId:
    """An illegal B instance id must be a controlled refusal, not a crash."""

    @pytest.mark.parametrize("condition", ["pure", "routed"])
    @pytest.mark.parametrize("bad_value,label", [
        (None, "none"), (True, "bool"), (0.0, "float"), ("0", "string"),
        ([0], "list"), ({"id": 0}, "dict"), ("MISSING", "missing_field"),
    ])
    def test_illegal_pred_instance_id_is_refused(
        self, legal, scratch_dir, condition, bad_value, label
    ):
        import json as _json
        from .pipeline_synthetic import refresh_manifest

        path = legal["source"] / "quality" / "instances.jsonl"
        rows = [_json.loads(line) for line in path.read_bytes().decode()
                .splitlines()]
        target = next(index for index, row in enumerate(rows)
                      if row["layer"] == "B_model_a_predictions"
                      and row["condition"] == condition)
        if bad_value == "MISSING":
            rows[target].pop("pred_instance_id")
        else:
            rows[target]["pred_instance_id"] = bad_value
        path.write_bytes("".join(_json.dumps(row, sort_keys=True) + "\n"
                                 for row in rows).encode("utf-8"))
        refresh_manifest(legal["source"] / "quality", [
            "summary.json", "protocol_snapshot.json", "instances.jsonl",
            "a2_alignment_summary.json", "a2_alignment.jsonl",
            "unmatched_gt.jsonl", "frames.jsonl"])
        code, summary, markdown = _run(legal, scratch_dir / "out")
        assert code != 0, f"{condition}/{label} must be refused"
        assert summary["status"] == "FAILED"
        reason = summary.get("instance_record_problem") or ""
        assert "pred_instance_id" in reason, reason
        assert "FAILED" in markdown


class TestFrameEvidenceBinding:
    """The four frame-evidence binding refusals through the real entry.

    Each scenario changes exactly one factor of the complete legal fixture
    and calls ``scripts.closeout_material_pipeline.main()``; the refusal is
    asserted by exception type and by the message naming ``frames.jsonl``.
    Every scenario uses its own output directory so a "directory already
    exists" refusal can never mask the target error.
    """

    def _run(self, fixture, output):
        from scripts import closeout_material_pipeline as closeout

        return closeout.main([
            "--source-dir", str(fixture["source"]),
            "--config", str(fixture["config"]),
            "--gt-index", str(fixture["gt_index"]),
            "--output-dir", str(output),
        ])

    def _manifest(self, fixture):
        return json.loads((fixture["source"] / "quality"
                           / "artifact_manifest.json").read_bytes().decode())

    def _write_manifest(self, fixture, manifest):
        (fixture["source"] / "quality" / "artifact_manifest.json").write_bytes(
            (json.dumps(manifest) + "\n").encode("utf-8"))

    def test_manifest_missing_the_frame_entry_is_refused(self, legal,
                                                         scratch_dir):
        """The file exists, the manifest simply does not list it."""
        manifest = self._manifest(legal)
        manifest["files"].pop("frames.jsonl")
        self._write_manifest(legal, manifest)
        with pytest.raises(ValueError) as exc_info:
            self._run(legal, scratch_dir / "out_uncovered")
        message = str(exc_info.value)
        assert "frames.jsonl" in message
        assert "not covered" in message

    def test_missing_frame_file_is_refused(self, legal, scratch_dir):
        """The file is gone while the manifest still records it."""
        (legal["source"] / "quality" / "frames.jsonl").unlink()
        with pytest.raises(FileNotFoundError) as exc_info:
            self._run(legal, scratch_dir / "out_missing")
        message = str(exc_info.value)
        assert "frames.jsonl" in message
        assert "missing" in message

    def test_file_and_entry_both_missing_is_refused(self, legal, scratch_dir):
        """Neither the file nor its manifest entry exists."""
        (legal["source"] / "quality" / "frames.jsonl").unlink()
        manifest = self._manifest(legal)
        manifest["files"].pop("frames.jsonl")
        self._write_manifest(legal, manifest)
        with pytest.raises(FileNotFoundError) as exc_info:
            self._run(legal, scratch_dir / "out_both_missing")
        message = str(exc_info.value)
        assert "frames.jsonl" in message
        assert "missing" in message

    def test_wrong_frame_digest_is_refused(self, legal, scratch_dir):
        """A well-formed but wrong digest must not be accepted."""
        manifest = self._manifest(legal)
        manifest["files"]["frames.jsonl"] = "0" * 64
        self._write_manifest(legal, manifest)
        with pytest.raises(ValueError) as exc_info:
            self._run(legal, scratch_dir / "out_wrong_digest")
        message = str(exc_info.value)
        assert "frames.jsonl" in message
        assert "digest mismatch" in message


class TestFrameEvidenceSchema:
    """frames.jsonl structure, identity, duplicates and frame coverage.

    Every scenario runs the real ``main()`` on the complete legal fixture and
    changes exactly one factor; the frame evidence is validated before it is
    turned into a prediction map, so an illegal count, a repeated frame, a
    frame outside the frozen scope or a missing frame cannot be laundered
    into a "complete" B coverage.
    """

    QUALITY_FILES = [
        "summary.json", "protocol_snapshot.json", "instances.jsonl",
        "a2_alignment_summary.json", "a2_alignment.jsonl", "unmatched_gt.jsonl",
        "frames.jsonl",
    ]

    def _rows(self, fixture):
        return [json.loads(line) for line in (
            fixture["source"] / "quality" / "frames.jsonl").read_bytes()
            .decode().splitlines()]

    def _write(self, fixture, rows):
        from .pipeline_synthetic import refresh_manifest

        path = fixture["source"] / "quality" / "frames.jsonl"
        text = "".join(
            (row if isinstance(row, str) else json.dumps(row, sort_keys=True))
            + "\n" for row in rows
        )
        path.write_bytes(text.encode("utf-8"))
        refresh_manifest(fixture["source"] / "quality", self.QUALITY_FILES)

    def _run(self, fixture, output):
        from scripts import closeout_material_pipeline as closeout

        code = closeout.main([
            "--source-dir", str(fixture["source"]),
            "--config", str(fixture["config"]),
            "--gt-index", str(fixture["gt_index"]),
            "--output-dir", str(output),
        ])
        summary = json.loads((output / "quality_summary_corrected.json")
                             .read_bytes().decode())
        markdown = (output / "correction_summary.md").read_bytes().decode()
        return code, summary, markdown

    def test_legal_zero_prediction_frame_is_kept(self, legal, scratch_dir):
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code == 0
        assert summary["status"] == "COMPLETED"
        verdict = summary["frame_evidence"]
        assert verdict["status"] == "PASSED"
        assert verdict["expected_frames"] == 3
        assert verdict["frame_predictions"]["frame_c"] == 0
        assert verdict["missing_frames"] == []

    def test_zero_prediction_frame_with_b_records_fails(self, legal,
                                                        scratch_dir):
        from .pipeline_synthetic import refresh_manifest

        path = legal["source"] / "quality" / "instances.jsonl"
        records = [json.loads(line) for line in path.read_bytes().decode()
                   .splitlines()]
        template = dict(records[0])
        template.update({
            "layer": "B_model_a_predictions", "condition": "routed",
            "frame_id": "frame_c", "pred_instance_id": 0,
            "gt_instance_id": None,
        })
        records.append(template)
        path.write_bytes("".join(json.dumps(r, sort_keys=True) + "\n"
                                 for r in records).encode("utf-8"))
        refresh_manifest(legal["source"] / "quality", self.QUALITY_FILES)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert any("zero predictions" in entry
                   for entry in summary["instance_coverage"]["failures"])

    def test_missing_frame_record_is_partial(self, legal, scratch_dir):
        rows = [row for row in self._rows(legal)
                if row["frame_id"] != "frame_c"]
        self._write(legal, rows)
        code, summary, markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "PARTIAL"
        assert summary["frame_evidence"]["missing_frames"] == ["frame_c"]
        assert "PARTIAL" in markdown

    def test_empty_frame_evidence_is_partial(self, legal, scratch_dir):
        self._write(legal, [])
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "PARTIAL"
        assert summary["frame_evidence"]["missing_frames"]

    def test_unknown_frame_replacing_a_legal_one_fails(self, legal,
                                                       scratch_dir):
        rows = self._rows(legal)
        rows[0]["frame_id"] = "nonexistent_frame"
        self._write(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert summary["frame_evidence"]["extra_frames"] == [
            "nonexistent_frame"]

    def test_extra_unknown_frame_fails(self, legal, scratch_dir):
        rows = self._rows(legal)
        extra = dict(rows[0])
        extra["frame_id"] = "extra_frame"
        rows.append(extra)
        self._write(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"

    def test_exact_duplicate_frame_fails(self, legal, scratch_dir):
        rows = self._rows(legal)
        rows.append(dict(rows[0]))
        self._write(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert any("duplicate frame record" in entry
                   for entry in summary["frame_evidence"]["failures"])

    def test_duplicate_frame_with_other_count_fails(self, legal, scratch_dir):
        rows = self._rows(legal)
        duplicate = dict(rows[0])
        duplicate["predictions"] = rows[0]["predictions"] + 1
        rows.append(duplicate)
        self._write(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert any("duplicate frame record" in entry
                   for entry in summary["frame_evidence"]["failures"])

    @pytest.mark.parametrize("value,label", [
        ("MISSING", "missing"), (None, "none"), (True, "bool_true"),
        (False, "bool_false"), (0.0, "float_zero"), (1.5, "float"),
        ("0", "string"), (-1, "negative"), ([0], "list"), ({"n": 0}, "dict"),
    ])
    def test_illegal_predictions_are_refused(self, legal, scratch_dir,
                                             value, label):
        rows = self._rows(legal)
        if value == "MISSING":
            rows[0].pop("predictions")
        else:
            rows[0]["predictions"] = value
        self._write(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0, label
        assert summary["status"] == "FAILED"
        assert any("predictions" in entry
                   for entry in summary["frame_evidence"]["failures"])

    @pytest.mark.parametrize("value,label", [
        ("MISSING", "missing"), (None, "none"), ("", "empty_string"),
        (123, "number"), ([0], "list"), ({"f": 0}, "dict"),
    ])
    def test_illegal_frame_id_is_refused(self, legal, scratch_dir, value,
                                         label):
        rows = self._rows(legal)
        if value == "MISSING":
            rows[0].pop("frame_id")
        else:
            rows[0]["frame_id"] = value
        self._write(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0, label
        assert summary["status"] == "FAILED"
        assert any("frame_id" in entry
                   for entry in summary["frame_evidence"]["failures"])

    def test_non_object_frame_record_is_refused(self, legal, scratch_dir):
        rows = self._rows(legal)
        rows.append("[1, 2, 3]")          # a bare JSON array, written verbatim
        self._write(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert any("not a JSON object" in entry
                   for entry in summary["frame_evidence"]["failures"])


class TestMissingFrameEvidence:
    """A frame whose prediction evidence is missing is unverifiable, not wrong.

    The expected frame set comes from the bound GT index. When a frame's
    record is absent, its prediction count is unknown, so its B instances
    cannot be called unknown or out-of-range; the coverage is unverifiable
    (PARTIAL). Count-independent contradictions (illegal or negative ids,
    duplicate identities, records on frames outside the frozen scope) remain
    FAILED, and a contradiction outranks a gap.
    """

    QUALITY_FILES = [
        "summary.json", "protocol_snapshot.json", "instances.jsonl",
        "a2_alignment_summary.json", "a2_alignment.jsonl", "unmatched_gt.jsonl",
        "frames.jsonl",
    ]

    def _frames(self, fixture):
        return [json.loads(line) for line in (
            fixture["source"] / "quality" / "frames.jsonl").read_bytes()
            .decode().splitlines()]

    def _instances(self, fixture):
        return [json.loads(line) for line in (
            fixture["source"] / "quality" / "instances.jsonl").read_bytes()
            .decode().splitlines()]

    def _refresh(self, fixture):
        from .pipeline_synthetic import refresh_manifest

        refresh_manifest(fixture["source"] / "quality", self.QUALITY_FILES)

    def _write_frames(self, fixture, rows):
        (fixture["source"] / "quality" / "frames.jsonl").write_bytes(
            "".join(json.dumps(row, sort_keys=True) + "\n"
                    for row in rows).encode("utf-8"))
        self._refresh(fixture)

    def _write_instances(self, fixture, rows):
        (fixture["source"] / "quality" / "instances.jsonl").write_bytes(
            "".join(json.dumps(row, sort_keys=True) + "\n"
                    for row in rows).encode("utf-8"))
        self._refresh(fixture)

    def _run(self, fixture, output):
        from scripts import closeout_material_pipeline as closeout

        code = closeout.main([
            "--source-dir", str(fixture["source"]),
            "--config", str(fixture["config"]),
            "--gt-index", str(fixture["gt_index"]),
            "--output-dir", str(output),
        ])
        summary = json.loads((output / "quality_summary_corrected.json")
                             .read_bytes().decode())
        markdown = (output / "correction_material_summary.md"
                    if False else output / "correction_summary.md")
        return code, summary, markdown.read_bytes().decode()

    def _drop_frame(self, fixture, frame_id):
        self._write_frames(fixture, [row for row in self._frames(fixture)
                                     if row["frame_id"] != frame_id])

    def test_missing_frame_with_predictions_is_partial_not_unknown(
        self, legal, scratch_dir
    ):
        self._drop_frame(legal, "frame_a")
        code, summary, markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "PARTIAL"
        assert summary["frame_evidence"]["status"] == "PARTIAL"
        coverage = summary["instance_coverage"]
        assert coverage["status"] == "PARTIAL"
        joined = " ".join(coverage["failures"])
        assert "unknown or out-of-range" not in joined, (
            "a missing count must not be reported as a wrong identity"
        )
        assert "PARTIAL" in markdown

    def test_missing_zero_prediction_frame_is_partial(self, legal, scratch_dir):
        self._drop_frame(legal, "frame_c")
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "PARTIAL"
        assert summary["instance_coverage"]["status"] != "PASSED"

    def test_missing_frame_with_all_records_deleted_is_partial(
        self, legal, scratch_dir
    ):
        self._drop_frame(legal, "frame_a")
        rows = [row for row in self._instances(legal)
                if row["frame_id"] != "frame_a"]
        self._write_instances(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "PARTIAL"

    def test_missing_frame_with_out_of_range_on_another_frame_fails(
        self, legal, scratch_dir
    ):
        self._drop_frame(legal, "frame_a")
        rows = self._instances(legal)
        target = next(index for index, row in enumerate(rows)
                      if row["layer"] == "B_model_a_predictions"
                      and row["condition"] == "routed"
                      and row["frame_id"] == "frame_b")
        rows[target]["pred_instance_id"] = 999
        self._write_instances(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert any("frame_b" in entry and "out-of-range" in entry
                   for entry in summary["instance_coverage"]["failures"])

    def test_missing_frame_with_duplicate_identity_still_fails(
        self, legal, scratch_dir
    ):
        self._drop_frame(legal, "frame_a")
        rows = self._instances(legal)
        indices = [index for index, row in enumerate(rows)
                   if row["layer"] == "B_model_a_predictions"
                   and row["condition"] == "routed"
                   and row["frame_id"] == "frame_a"]
        rows[indices[1]]["pred_instance_id"] = rows[indices[0]][
            "pred_instance_id"]
        self._write_instances(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert any("duplicate instance identities" in entry
                   for entry in summary["instance_coverage"]["failures"])

    @pytest.mark.parametrize("condition", ["pure", "routed"])
    @pytest.mark.parametrize("value,label", [(-1, "negative"), ([0], "list")])
    def test_missing_frame_with_illegal_id_still_fails(
        self, legal, scratch_dir, condition, value, label
    ):
        self._drop_frame(legal, "frame_a")
        rows = self._instances(legal)
        target = next(index for index, row in enumerate(rows)
                      if row["layer"] == "B_model_a_predictions"
                      and row["condition"] == condition
                      and row["frame_id"] == "frame_a")
        rows[target]["pred_instance_id"] = value
        self._write_instances(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0, label
        assert summary["status"] == "FAILED"
        assert any("pred_instance_id" in entry
                   for entry in summary["instance_coverage"]["failures"])

    def test_missing_frame_with_records_on_an_unknown_frame_fails(
        self, legal, scratch_dir
    ):
        self._drop_frame(legal, "frame_a")
        rows = self._instances(legal)
        template = next(row for row in rows
                        if row["layer"] == "B_model_a_predictions"
                        and row["condition"] == "routed")
        extra = dict(template)
        extra.update({"frame_id": "outside_frame", "pred_instance_id": 0})
        rows.append(extra)
        self._write_instances(legal, rows)
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "FAILED"
        assert any("outside the frozen evaluation scope" in entry
                   for entry in summary["instance_coverage"]["failures"])

    def test_empty_frame_evidence_with_legal_records_is_partial(
        self, legal, scratch_dir
    ):
        self._write_frames(legal, [])
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code != 0
        assert summary["status"] == "PARTIAL"
        assert summary["instance_coverage"]["status"] != "PASSED"

    def test_complete_legal_fixture_still_completes(self, legal, scratch_dir):
        code, summary, _markdown = self._run(legal, scratch_dir / "out")
        assert code == 0
        assert summary["status"] == "COMPLETED"
        assert summary["instance_coverage"]["status"] == "PASSED"
