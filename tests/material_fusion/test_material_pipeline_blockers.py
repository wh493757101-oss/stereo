"""Real-entry regressions for the four remaining closeout blockers."""

from __future__ import annotations

import hashlib
import json

import pytest

from .pipeline_synthetic import refresh_manifest
from .test_material_pipeline_closeout_cli import legal, _run  # noqa: F401


QUALITY_FILES = (
    "summary.json", "protocol_snapshot.json", "instances.jsonl",
    "a2_alignment_summary.json", "a2_alignment.jsonl",
    "unmatched_gt.jsonl", "frames.jsonl",
)


def change_summary(fixture, mutate):
    path = fixture["source"] / "quality" / "summary.json"
    payload = json.loads(path.read_bytes())
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest(path.parent, QUALITY_FILES)


class TestSummarySupportAdmission:
    @pytest.mark.parametrize("view", ["pure", "routed"])
    @pytest.mark.parametrize("tag", ["gray_s2026", "gray_polar_s2028"])
    def test_self_consistent_summary_count_must_match_evidence(
        self, legal, scratch_dir, view, tag
    ):
        def mutate(payload):
            block = payload["per_candidate"][tag]["layer_b"][view]
            block["counts"]["predictions"] = 5
            block["counts"]["unmatched_predictions"] = 1
            block["detection"]["fp"] = 1

        change_summary(legal, mutate)
        code, report, markdown = _run(legal, scratch_dir / "out")
        assert code != 0
        assert report["status"] == "FAILED"
        assert any("summary reports 5" in reason and view in reason
                   for reason in report["instance_coverage"]["failures"])
        assert "status: **FAILED**" in markdown

    @pytest.mark.parametrize("bad", [False, 4.0, "4", None])
    def test_illegal_summary_support_is_rejected(self, legal, scratch_dir, bad):
        def mutate(payload):
            payload["per_candidate"]["gray_s2026"]["layer_b"]["pure"][
                "counts"]["predictions"] = bad

        change_summary(legal, mutate)
        code, report, _ = _run(legal, scratch_dir / "out")
        assert code != 0
        assert report["status"] == "FAILED"

    def test_missing_frame_is_not_compared_as_a_full_total(self, legal, scratch_dir):
        path = legal["source"] / "quality" / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows
                                if row["frame_id"] != "frame_a"), encoding="utf-8")
        refresh_manifest(path.parent, QUALITY_FILES)
        code, report, _ = _run(legal, scratch_dir / "out")
        assert code != 0
        assert report["status"] == "PARTIAL"
        assert report["instance_coverage"]["failures"] == []
        assert any("summary" in reason and "cannot be verified" in reason
                   for reason in report["instance_coverage"]["partials"])


class TestBootstrapAdmission:
    @pytest.mark.parametrize("status", ["UNVERIFIED", "NOT_MEASURED", "NOT_ESTIMABLE"])
    @pytest.mark.parametrize("seed", [2026, 2027, 2028])
    def test_unverifiable_seed_propagates_partial(self, legal, scratch_dir, status, seed):
        def mutate(payload):
            payload["bootstrap"]["end_to_end_macro_f1_e4_vs_e1"][
                f"seed_{seed}"] = {"status": status, "reason": "no independent estimate"}

        change_summary(legal, mutate)
        code, report, markdown = _run(legal, scratch_dir / "out")
        assert code != 0
        assert report["status"] == "PARTIAL"
        assert report["consistency_checks"]["status"] == "PARTIAL"
        assert any(f"seed_{seed}" in reason and status in reason
                   for reason in report["consistency_checks"]["partials"])
        assert "status: **PARTIAL**" in markdown

    def test_contradiction_outranks_unverifiable_bootstrap(self, legal, scratch_dir):
        def mutate(payload):
            payload["bootstrap"]["end_to_end_macro_f1_e4_vs_e1"]["seed_2026"] = {
                "status": "UNVERIFIED"}
            payload.pop("conclusions")

        change_summary(legal, mutate)
        code, report, _ = _run(legal, scratch_dir / "out")
        assert code != 0
        assert report["status"] == "FAILED"


class TestRequiredResultSections:
    @pytest.mark.parametrize("section", ["paired_changes", "conclusions"])
    @pytest.mark.parametrize("mode", ["missing", "null", "empty", "wrong_type"])
    def test_missing_both_sides_is_not_agreement(self, legal, scratch_dir, section, mode):
        def mutate(payload):
            if mode == "missing":
                payload.pop(section)
            else:
                payload[section] = {"null": None, "empty": [], "wrong_type": "ok"}[mode]

        change_summary(legal, mutate)
        code, report, markdown = _run(legal, scratch_dir / "out")
        assert code != 0
        assert report["status"] == "FAILED"
        assert any(section in reason for reason in report["consistency_checks"]["mismatches"])
        assert "status: **FAILED**" in markdown


class TestFinalPublication:
    @pytest.mark.parametrize("target", ["metric", "bootstrap", "timing", "performance",
                                        "invalid_json", "manifest", "manifest_payload"])
    def test_serialization_fault_never_announces_success(
        self, legal, scratch_dir, monkeypatch, capsys, target
    ):
        from scripts import closeout_material_pipeline as closeout

        original = closeout.write_json_bytes
        fired = False

        def inject(directory, name, payload):
            nonlocal fired
            if not fired and target == "manifest_payload" and name == "artifact_manifest.json":
                fired = True
                payload["files"].pop("raw_timings_corrected.json")
            path = original(directory, name, payload)
            names = {"metric": "quality_summary_corrected.json",
                     "bootstrap": "quality_summary_corrected.json",
                     "timing": "raw_timings_corrected.json",
                     "performance": "performance_report_corrected.json",
                     "invalid_json": "quality_summary_corrected.json",
                     "manifest": "artifact_manifest.json",
                     "manifest_payload": "artifact_manifest.json"}
            if not fired and name == names[target]:
                fired = True
                if target == "invalid_json":
                    path.write_bytes(b"{broken")
                    return path
                altered = json.loads(path.read_bytes())
                if target == "metric":
                    altered["per_candidate"]["gray_s2026"]["layer_b"]["pure"][
                        "conditional_classification"]["accuracy"] = 0.25
                elif target == "bootstrap":
                    altered["bootstrap"]["end_to_end_macro_f1_e4_vs_e1"][
                        "seed_2026"]["ci_low"] = -0.5
                elif target == "timing":
                    altered["raw"]["1"]["gray"]["total"][0] = 99.0
                elif target == "performance":
                    altered["scenarios"]["1"]["variants"]["gray"]["throughput_fps"] = 99.0
                else:
                    altered["files"].pop("raw_timings_corrected.json")
                path.write_text(json.dumps(altered), encoding="utf-8")
            return path

        monkeypatch.setattr(closeout, "write_json_bytes", inject)
        output = scratch_dir / "out"
        code, report, markdown = _run(legal, output)
        captured = capsys.readouterr()
        assert fired
        assert code != 0
        assert report["status"] == "FAILED"
        assert "status: **FAILED**" in markdown
        assert "status: COMPLETED" not in captured.out + captured.err
        manifest = json.loads((output / "artifact_manifest.json").read_bytes())
        assert "raw_timings_corrected.json" in manifest["files"]
        for name, digest in manifest["files"].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest

    def test_failure_rewrite_is_verified_again(self, legal, scratch_dir, monkeypatch, capsys):
        from scripts import closeout_material_pipeline as closeout

        original = closeout.write_json_bytes
        writes = 0

        def persistent_fault(directory, name, payload):
            nonlocal writes
            path = original(directory, name, payload)
            if name == "quality_summary_corrected.json":
                writes += 1
                altered = json.loads(path.read_bytes())
                altered["status"] = "COMPLETED"
                altered["conclusions"] = {"tampered": "yes"}
                path.write_text(json.dumps(altered), encoding="utf-8")
            return path

        monkeypatch.setattr(closeout, "write_json_bytes", persistent_fault)
        output = scratch_dir / "out"
        code = closeout.main(["--source-dir", str(legal["source"]),
                              "--config", str(legal["config"]),
                              "--gt-index", str(legal["gt_index"]),
                              "--output-dir", str(output)])
        captured = capsys.readouterr()
        assert code != 0
        assert writes == 2, "bounded failure rewrite must be rechecked"
        assert "status: COMPLETED" not in captured.out + captured.err
        assert not (output / "quality_summary_corrected.json").exists()
        assert not (output / "artifact_manifest.json").exists()
        marker = json.loads((output / "publication_failed.json").read_bytes())
        assert marker["status"] == "FAILED"

    def test_write_exception_is_a_failed_publication(self, legal, scratch_dir, monkeypatch, capsys):
        from scripts import closeout_material_pipeline as closeout

        original = closeout.write_json_bytes
        fired = False

        def fail_once(directory, name, payload):
            nonlocal fired
            if name == "raw_timings_corrected.json" and not fired:
                fired = True
                raise OSError("injected disk failure")
            return original(directory, name, payload)

        monkeypatch.setattr(closeout, "write_json_bytes", fail_once)
        code, report, markdown = _run(legal, scratch_dir / "out")
        captured = capsys.readouterr()
        assert code != 0
        assert report["status"] == "FAILED"
        assert "status: **FAILED**" in markdown
        assert "status: COMPLETED" not in captured.out + captured.err

    def test_markdown_corruption_is_verified(self, legal, scratch_dir, monkeypatch, capsys):
        from pathlib import Path

        original = Path.write_bytes
        fired = False

        def corrupt_markdown(path, data):
            nonlocal fired
            if path.name == "correction_summary.md" and not fired:
                fired = True
                data = b"status: COMPLETED -- damaged Markdown"
            return original(path, data)

        monkeypatch.setattr(Path, "write_bytes", corrupt_markdown)
        code, report, markdown = _run(legal, scratch_dir / "out")
        captured = capsys.readouterr()
        assert fired
        assert code != 0
        assert report["status"] == "FAILED"
        assert "status: **FAILED**" in markdown
        assert "status: COMPLETED" not in captured.out + captured.err
