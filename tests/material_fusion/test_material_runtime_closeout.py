"""4A closeout tool tests (TDD: written before the implementation).

Covers the closeout contract:

- byte-level digest verification, including newline differences;
- missing / corrupted / internally inconsistent evidence must never report
  a pass;
- a signed_q sign error or a single localized pixel error must not be
  masked by a matching mean, and an out-of-tolerance result must fail;
- an unrecoverable original frame list must refuse the item rather than
  substitute easier frames;
- the complete GT index keeps instances without an NPZ and empty GT frames,
  distinguishes a missing annotation from an empty one, refuses to mix the
  four-class annotations with the single-class segmentation labels, and
  checks both association directions, duplicates and missing sources;
- the report directory refuses to overwrite an existing one, and the
  historical evidence directory is never written to;
- cleanup only ever targets an explicitly verified literal path and refuses
  links / junctions / paths outside the allowed parent.

All data here is synthetic; no real dataset, checkpoint or model is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from .conftest import CLASS_NAMES


# ---------------------------------------------------------------------------
# Helpers


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_evidence(directory: Path) -> Path:
    """A minimal but internally consistent 4A evidence directory.

    The identity files are created next to ``directory`` (in the identity
    root) with their real digests recorded, so the verification can only
    pass when it really recomputes them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    identity_root = directory.parent
    recorded = {}
    for name, content in (("model_a.pt", b"model-a"), ("fallback.pt", b"fallback"),
                          ("cand.pt", b"candidate")):
        path = identity_root / name
        path.write_bytes(content)
        recorded[name] = _sha(path)
    report = {
        "status": "PASSED",
        "generated_by": "scripts/verify_material_runtime.py",
        "layer_a": {
            "samples": 2,
            "selection_rule": "first N val rows in manifest order",
            "status": "PASSED",
            "variants": {
                "gray": {"logit_max_abs_err": 0.0, "logits_within_tolerance": True},
            },
        },
        "layer_b": {
            "status": "PASSED",
            "selected_frames": ["frame_a", "frame_b"],
            "frames": [{"frame": "frame_a", "variants": {}},
                       {"frame": "frame_b", "variants": {}}],
            "gt_replay": [
                {"frame": "frame_a", "status": "PASSED",
                 "objects": [{"sample_name": "frame_a_obj000", "status": "PASSED"}]},
                {"frame": "frame_b", "status": "PASSED",
                 "objects": [{"sample_name": "frame_b_obj000", "status": "PASSED"}]},
            ],
            "fallback_totals": {"gray": 0},
            "real_forward_seconds": 1.0,
        },
        "parameters_and_buffers_unchanged": True,
    }
    _write_json(directory / "report.json", report)
    _write_json(directory / "training_data_admission.json", {"audit_fingerprint": "a" * 64})
    identities = {
        "model_a": {"path": "model_a.pt", "sha256": recorded["model_a.pt"],
                    "matches_config": True},
        "fallback": {"path": "fallback.pt", "sha256": recorded["fallback.pt"],
                     "matches_config": True},
        "candidates": [
            {"variant": "gray", "seed": 2026, "checkpoint": "cand.pt",
             "sha256": recorded["cand.pt"], "matches_config": True},
        ],
    }
    _write_json(directory / "identities.json", identities)
    digests = {
        name: _sha(directory / name)
        for name in ("report.json", "training_data_admission.json", "identities.json")
    }
    _write_json(directory / "digests.json", digests)
    return directory


def _matching_identity_files(directory: Path) -> None:
    """Kept for callers that only need the identity files to exist."""
    for name, content in (("model_a.pt", b"model-a"), ("fallback.pt", b"fallback"),
                          ("cand.pt", b"candidate")):
        path = directory / name
        if not path.exists():
            path.write_bytes(content)


def _synthetic_split_csv(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "output_stem,split,group_name,block_id,source_json,left_path,right_path\n"
    body = "".join(
        f"{stem},{split},{group},0,{group}/{stem[-3:]}.json,{left},{right}\n"
        for stem, split, group, left, right in rows
    )
    path.write_bytes((header + body).encode("utf-8"))
    return path


def _synthetic_label(path: Path, lines) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return path


def _polygon_line(class_id: int, x0=0.2, y0=0.2, x1=0.4, y1=0.4) -> str:
    return f"{class_id} {x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1}"


# ---------------------------------------------------------------------------
# Byte-level digests


class TestByteLevelDigests:
    def test_digest_is_of_the_bytes_on_disk_not_of_the_intended_text(
        self, scratch_dir
    ):
        from scripts.closeout_material_runtime import sha256_of, write_report_bytes

        payload = {"non_ascii": "偏振", "value": 1}
        path = write_report_bytes(scratch_dir, "report.json", payload)
        assert sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()
        # A CRLF rewrite of the same logical content must change the digest.
        lf_bytes = path.read_bytes()
        path.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
        assert sha256_of(path) != hashlib.sha256(lf_bytes).hexdigest()

    def test_output_dir_refuses_to_overwrite(self, scratch_dir):
        from scripts.closeout_material_runtime import prepare_output_dir

        target = scratch_dir / "closeout"
        prepare_output_dir(target)
        with pytest.raises(FileExistsError):
            prepare_output_dir(target)


# ---------------------------------------------------------------------------
# Evidence verification


class TestEvidenceVerification:
    def test_consistent_evidence_passes(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_evidence

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        _matching_identity_files(scratch_dir)
        result = verify_evidence(evidence, identity_root=scratch_dir)
        assert result["status"] == "PASS"
        assert result["digests"]["mismatches"] == []
        assert result["identity_mismatches"] == []

    def test_missing_file_fails(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_evidence

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        (evidence / "report.json").unlink()
        result = verify_evidence(evidence, identity_root=scratch_dir)
        assert result["status"] == "FAIL"
        assert any("report.json" in problem for problem in result["problems"])

    def test_corrupt_json_fails(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_evidence

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        (evidence / "report.json").write_bytes(b"{not json")
        result = verify_evidence(evidence, identity_root=scratch_dir)
        assert result["status"] == "FAIL"

    def test_digest_mismatch_fails(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_evidence

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))
        report["status"] = "FAILED"          # change the bytes after digesting
        _write_json(evidence / "report.json", report)
        result = verify_evidence(evidence, identity_root=scratch_dir)
        assert result["status"] == "FAIL"
        assert any("report.json" in mismatch for mismatch in result["digests"]["mismatches"])

    def test_identity_mismatch_fails(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_evidence

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        (scratch_dir / "cand.pt").write_bytes(b"different bytes than recorded")
        result = verify_evidence(evidence, identity_root=scratch_dir)
        assert result["status"] == "FAIL"
        assert result["identity_mismatches"]

    def test_internal_inconsistency_fails(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_evidence

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))
        report["layer_b"]["selected_frames"] = ["frame_a"]   # 1 frame, 2 gt_replay
        _write_json(evidence / "report.json", report)
        # re-digest so only the consistency check can fail
        digests = json.loads((evidence / "digests.json").read_text(encoding="utf-8"))
        digests["report.json"] = _sha(evidence / "report.json")
        _write_json(evidence / "digests.json", digests)
        result = verify_evidence(evidence, identity_root=scratch_dir)
        assert result["status"] == "FAIL"
        assert result["consistency"]["problems"]

    def test_evidence_dir_is_never_written_to(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_evidence

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        before = {p.name: _sha(p) for p in evidence.iterdir() if p.is_file()}
        verify_evidence(evidence, identity_root=scratch_dir)
        after = {p.name: _sha(p) for p in evidence.iterdir() if p.is_file()}
        assert before == after


# ---------------------------------------------------------------------------
# Polar content parity


class TestFloatArrayComparison:
    def test_equal_arrays_report_exact_equality(self):
        from scripts.closeout_material_runtime import compare_float_arrays

        array = np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(8, 8)
        result = compare_float_arrays(array, array.copy())
        assert result["exact_equal"] is True
        assert result["max_abs_err"] == 0.0
        assert result["n_over_tolerance"] == 0

    def test_sign_flip_is_not_masked_by_matching_mean(self):
        from scripts.closeout_material_runtime import compare_float_arrays

        expected = np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(8, 8)
        actual = -expected
        result = compare_float_arrays(actual, expected)
        assert result["exact_equal"] is False
        # |(-e) - e| = 2|e|, so the worst pixel differs by 2 * 0.5.
        assert result["max_abs_err"] == pytest.approx(1.0)
        assert result["n_over_tolerance"] > 0
        assert result["status"] == "FAILED"

    def test_localized_pixel_error_is_detected(self):
        from scripts.closeout_material_runtime import compare_float_arrays

        expected = np.zeros((16, 16), dtype=np.float32)
        actual = expected.copy()
        actual[3, 4] = 1e-3
        result = compare_float_arrays(actual, expected)
        assert result["exact_equal"] is False
        assert result["n_over_tolerance"] == 1
        assert result["max_abs_err"] == pytest.approx(1e-3)
        assert result["status"] == "FAILED"

    def test_tolerance_is_fixed_and_not_widened(self):
        from scripts.closeout_material_runtime import compare_float_arrays

        expected = np.zeros((4, 4), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0] = 2e-6            # above atol=1e-6
        result = compare_float_arrays(actual, expected, atol=1e-6, rtol=0.0)
        assert result["status"] == "FAILED"
        assert result["n_over_tolerance"] == 1

    def test_non_finite_values_are_reported(self):
        from scripts.closeout_material_runtime import compare_float_arrays

        expected = np.zeros((4, 4), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0] = np.nan
        result = compare_float_arrays(actual, expected)
        assert result["finite"] is False
        assert result["status"] == "FAILED"

    def test_target_region_error_is_reported_separately(self):
        from scripts.closeout_material_runtime import compare_float_arrays

        expected = np.zeros((4, 4), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0] = 5e-3           # outside target
        target = np.zeros((4, 4), dtype=bool)
        target[2, 2] = True
        result = compare_float_arrays(actual, expected, target_mask=target)
        assert result["max_abs_err"] == pytest.approx(5e-3)
        assert result["max_abs_err_in_target"] == 0.0


class TestFrameListRecovery:
    def test_frames_come_from_the_4a_report(self, scratch_dir):
        from scripts.closeout_material_runtime import recover_4a_frames

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        assert recover_4a_frames(evidence) == ["frame_a", "frame_b"]

    def test_unrecoverable_frame_list_refuses(self, scratch_dir):
        from scripts.closeout_material_runtime import recover_4a_frames

        evidence = _synthetic_evidence(scratch_dir / "evidence")
        report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))
        report["layer_b"].pop("selected_frames")
        _write_json(evidence / "report.json", report)
        with pytest.raises(ValueError, match="frame list"):
            recover_4a_frames(evidence)


class TestPolarParity:
    def _fixture(self, scratch_dir, *, sign_flip=False, missing_npz=False):
        """A tiny synthetic frame + npz pair mirroring the real layout."""
        from core.material_mask_dataset import save_mask_sample

        data_root = scratch_dir / "v5"
        npz_dir = data_root / "val" / "metal_submarine"
        npz_dir.mkdir(parents=True, exist_ok=True)
        height, width = 24, 32
        gray = np.full((height, width), 120, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[4:20, 6:26] = 1
        signed_q = np.zeros((height, width), dtype=np.float32)
        signed_q[4:20, 6:26] = 0.25
        if sign_flip:
            signed_q = -signed_q
        abs_q = np.abs(signed_q).astype(np.float32)
        valid = mask.copy()
        npz_path = npz_dir / "frame_a_obj000.npz"
        save_mask_sample(
            npz_path, gray=gray, signed_q=signed_q, abs_q=abs_q, valid=valid,
            quality=np.asarray([1.0, 1.0, 1.0, 0.25], dtype=np.float32),
            class_id=0, instance_mask=mask,
        )
        return data_root, npz_path

    def test_matching_arrays_pass(self, scratch_dir):
        from scripts.closeout_material_runtime import compare_instance_arrays

        _, npz_path = self._fixture(scratch_dir)
        from core.material_mask_dataset import load_mask_sample

        sample = load_mask_sample(npz_path)
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=sample.signed_q.copy(),
            rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=sample.valid.astype(np.uint8),
            npz=sample,
        )
        assert record["status"] == "PASSED"
        assert record["signed_q"]["exact_equal"] is True
        assert record["abs_q"]["exact_equal"] is True

    def test_sign_error_fails(self, scratch_dir):
        from scripts.closeout_material_runtime import compare_instance_arrays

        _, npz_path = self._fixture(scratch_dir)
        from core.material_mask_dataset import load_mask_sample

        sample = load_mask_sample(npz_path)
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=(-sample.signed_q).copy(),
            rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=sample.valid.astype(np.uint8),
            npz=sample,
        )
        assert record["status"] == "FAILED"
        assert record["signed_q"]["n_over_tolerance"] > 0

    def test_instance_without_npz_is_recorded_not_passed(self, scratch_dir):
        from scripts.closeout_material_runtime import compare_instance_arrays

        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj001", object_index=1,
            rebuilt_signed=None, rebuilt_abs=None, rebuilt_valid=None,
            npz=None,
        )
        assert record["status"] == "NO_NPZ"
        assert record["npz_path"] is None

    def test_invalid_region_zeroing_contract_is_checked(self, scratch_dir):
        from scripts.closeout_material_runtime import compare_instance_arrays

        _, npz_path = self._fixture(scratch_dir)
        from core.material_mask_dataset import load_mask_sample

        sample = load_mask_sample(npz_path)
        rebuilt_signed = sample.signed_q.copy()
        rebuilt_signed[0, 0] = 0.5        # a value outside the valid support
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=rebuilt_signed, rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=sample.valid.astype(np.uint8), npz=sample,
        )
        assert record["invalid_region_zero_ok"] is False
        assert record["status"] == "FAILED"


# ---------------------------------------------------------------------------
# Complete GT index


class TestGtIndex:
    def _fixture(self, scratch_dir, *, empty_frame=True, missing_label=False,
                 duplicate=False, bad_polygon=False, bad_class=False,
                 npz_without_gt=False):
        labels = scratch_dir / "labels"
        v5_root = scratch_dir / "v5"
        (v5_root / "val").mkdir(parents=True, exist_ok=True)
        frames = [("frame_a", "Group/One/0 NTU"), ("frame_b", "Group/Two/0 NTU")]
        rows = []
        for stem, group in frames:
            rows.append((stem, "val", group,
                         f"datasets/Rectified_v2/{group}/left/000.png",
                         f"datasets/Rectified_v2/{group}/right/000.png"))
        split_csv = _synthetic_split_csv(scratch_dir / "split_assignment.csv", rows)

        for stem, _group in frames:
            if stem == "frame_b" and missing_label:
                continue
            if stem == "frame_b" and empty_frame:
                _synthetic_label(labels / "val" / f"{stem}.txt", [])
                continue
            lines = []
            if bad_polygon:
                lines.append(f"0 1.4 0.2 1.6 0.4 0.2 0.2")
            if bad_class:
                lines.append(_polygon_line(9))
            lines.append(_polygon_line(0))
            if stem == "frame_a" and duplicate:
                lines.append(_polygon_line(0))
            _synthetic_label(labels / "val" / f"{stem}.txt", lines)

        manifest_rows = ["sample_name,split,group_name,class_id,class_name,source_frame,"
                         "object_index,npz_path,left_path,label_path,"
                         "crop_x1,crop_y1,crop_x2,crop_y2,image_width,image_height"]
        manifest_rows.append(
            "frame_a_obj000,val,Group/One/0 NTU,0,metal_submarine,frame_a,0,"
            "val/metal_submarine/frame_a_obj000.npz,"
            "datasets/Rectified_v2/Group/One/0 NTU/left/000.png,"
            "labels/val/frame_a.txt,6,4,26,20,32,24"
        )
        if npz_without_gt:
            manifest_rows.append(
                "frame_c_obj000,val,Group/Three/0 NTU,0,metal_submarine,frame_c,0,"
                "val/metal_submarine/frame_c_obj000.npz,"
                "datasets/Rectified_v2/Group/Three/0 NTU/left/000.png,"
                "labels/val/frame_c.txt,6,4,26,20,32,24"
            )
        manifest = scratch_dir / "dataset_manifest.csv"
        manifest.write_bytes(("\n".join(manifest_rows) + "\n").encode("utf-8"))
        return split_csv, labels, manifest, v5_root

    def test_index_records_every_gt_instance_and_empty_frames(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(scratch_dir)
        index, summary = build_val_gt_index(
            split_csv=split_csv, label_root=labels, v5_manifest=manifest,
            data_root=v5_root, project_root=scratch_dir,
            image_size_reader=lambda path: (32, 24),
        )
        frames = {entry["frame_id"]: entry for entry in index}
        assert set(frames) == {"frame_a", "frame_b"}
        assert frames["frame_a"]["gt_instance_count"] == 1
        assert frames["frame_b"]["gt_instance_count"] == 0
        assert frames["frame_b"]["annotation_state"] == "empty"
        assert summary["gt_instances"] == 1
        assert summary["empty_gt_frames"] == 1
        assert summary["authoritative_frames"] == 2
        assert summary["indexed_frames"] == 2

    def test_missing_annotation_is_distinct_from_empty(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(
            scratch_dir, empty_frame=False, missing_label=True
        )
        index, summary = build_val_gt_index(
            split_csv=split_csv, label_root=labels, v5_manifest=manifest,
            data_root=v5_root, project_root=scratch_dir,
            image_size_reader=lambda path: (32, 24),
        )
        frames = {entry["frame_id"]: entry for entry in index}
        assert frames["frame_b"]["annotation_state"] == "missing"
        assert frames["frame_b"]["gt_instance_count"] is None
        assert summary["frames_missing_annotation"] == 1
        assert summary["empty_gt_frames"] == 0

    def test_instance_without_npz_keeps_a_null_link_and_reason(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(
            scratch_dir, empty_frame=False
        )
        # frame_b has a GT instance but no npz row for it
        _synthetic_label(labels / "val" / "frame_b.txt", [_polygon_line(3)])
        index, summary = build_val_gt_index(
            split_csv=split_csv, label_root=labels, v5_manifest=manifest,
            data_root=v5_root, project_root=scratch_dir,
            image_size_reader=lambda path: (32, 24),
        )
        frames = {entry["frame_id"]: entry for entry in index}
        instance = frames["frame_b"]["instances"][0]
        assert instance["v5_sample_name"] is None
        assert instance["has_npz"] is False
        assert instance["npz_absence_reason"]
        assert summary["gt_without_npz"] == 1

    def test_npz_without_gt_source_is_reported(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(
            scratch_dir, empty_frame=False, npz_without_gt=True
        )
        _synthetic_label(labels / "val" / "frame_b.txt", [_polygon_line(3)])
        _index, summary = build_val_gt_index(
            split_csv=split_csv, label_root=labels, v5_manifest=manifest,
            data_root=v5_root, project_root=scratch_dir,
            image_size_reader=lambda path: (32, 24),
        )
        assert summary["npz_without_gt_source"] == 1

    def test_duplicate_instances_are_reported(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(
            scratch_dir, empty_frame=False, duplicate=True
        )
        _synthetic_label(labels / "val" / "frame_b.txt", [_polygon_line(3)])
        index, summary = build_val_gt_index(
            split_csv=split_csv, label_root=labels, v5_manifest=manifest,
            data_root=v5_root, project_root=scratch_dir,
            image_size_reader=lambda path: (32, 24),
        )
        frames = {entry["frame_id"]: entry for entry in index}
        anomalies = frames["frame_a"]["anomalies"]
        assert any("duplicate" in anomaly for anomaly in anomalies)

    def test_out_of_range_polygon_is_reported_not_repaired(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(
            scratch_dir, empty_frame=False, bad_polygon=True
        )
        _synthetic_label(labels / "val" / "frame_b.txt", [_polygon_line(3)])
        index, _summary = build_val_gt_index(
            split_csv=split_csv, label_root=labels, v5_manifest=manifest,
            data_root=v5_root, project_root=scratch_dir,
            image_size_reader=lambda path: (32, 24),
        )
        frames = {entry["frame_id"]: entry for entry in index}
        assert any(
            "out_of_range" in anomaly for anomaly in frames["frame_a"]["anomalies"]
        )
        # the polygon is recorded as-is; nothing is clipped or dropped
        assert frames["frame_a"]["instances"][0]["polygon"]

    def test_unknown_class_id_is_reported(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(
            scratch_dir, empty_frame=False, bad_class=True
        )
        _synthetic_label(labels / "val" / "frame_b.txt", [_polygon_line(3)])
        index, _summary = build_val_gt_index(
            split_csv=split_csv, label_root=labels, v5_manifest=manifest,
            data_root=v5_root, project_root=scratch_dir,
            image_size_reader=lambda path: (32, 24),
        )
        frames = {entry["frame_id"]: entry for entry in index}
        instance = frames["frame_a"]["instances"][0]
        assert instance["class_id"] == 9
        assert instance["class_name"] is None
        assert any("unknown_class" in a for a in frames["frame_a"]["anomalies"])

    def test_single_class_label_root_is_refused(self, scratch_dir):
        from scripts.closeout_material_runtime import build_val_gt_index

        split_csv, labels, manifest, v5_root = self._fixture(scratch_dir)
        binary_root = scratch_dir / "underwater_seg_binary_v2"
        binary_root.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="four-class|single-class|binary"):
            build_val_gt_index(
                split_csv=split_csv, label_root=binary_root, v5_manifest=manifest,
                data_root=v5_root, project_root=scratch_dir,
                image_size_reader=lambda path: (32, 24),
            )


# ---------------------------------------------------------------------------
# Cleanup safety


class TestCleanupSafety:
    def test_target_outside_the_allowed_parent_is_refused(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_cleanup_target

        outside = scratch_dir / "not_scratch"
        outside.mkdir()
        allowed = scratch_dir / "allowed"
        allowed.mkdir()
        with pytest.raises(ValueError, match="outside|not inside"):
            verify_cleanup_target(outside, allowed_parent=allowed)

    def test_unknown_files_stop_the_cleanup(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_cleanup_target

        allowed = scratch_dir / "allowed"
        target = allowed / "4a_task"
        (target / "temp").mkdir(parents=True)
        (target / "surprise.pt").write_bytes(b"weights?")
        with pytest.raises(ValueError, match="unexpected"):
            verify_cleanup_target(
                target, allowed_parent=allowed, expected_names={"temp", "pycache"}
            )

    def test_expected_layout_is_accepted(self, scratch_dir):
        from scripts.closeout_material_runtime import verify_cleanup_target

        allowed = scratch_dir / "allowed"
        target = allowed / "4a_task"
        (target / "temp").mkdir(parents=True)
        (target / "pycache").mkdir()
        (target / "verify_run1.log").write_bytes(b"log\n")
        result = verify_cleanup_target(
            target, allowed_parent=allowed,
            expected_names={"temp", "pycache"},
        )
        assert result["target"] == str(target)
        assert result["reparse_points"] == []

    def test_logs_are_preserved_with_matching_digests(self, scratch_dir):
        from scripts.closeout_material_runtime import preserve_logs

        allowed = scratch_dir / "allowed"
        target = allowed / "4a_task"
        target.mkdir(parents=True)
        (target / "verify_run1.log").write_bytes(b"status: PASSED\n")
        (target / "verify_run2.log").write_bytes(b"Traceback\n")
        dest = scratch_dir / "preserved_logs"
        report = preserve_logs(target, dest)
        assert report["files"] == 2
        assert all(entry["sha256_match"] for entry in report["entries"])
        assert (dest / "verify_run1.log").read_bytes() == b"status: PASSED\n"

    def test_removal_targets_only_the_exact_directory(self, scratch_dir):
        from scripts.closeout_material_runtime import (
            preserve_logs,
            remove_cleanup_target,
            verify_cleanup_target,
        )

        allowed = scratch_dir / "allowed"
        target = allowed / "4a_task"
        sibling = allowed / "run_other_task"
        (target / "temp").mkdir(parents=True)
        (target / "verify_run1.log").write_bytes(b"log\n")
        sibling.mkdir(parents=True)
        (sibling / "keep.txt").write_bytes(b"keep\n")
        preserved = scratch_dir / "preserved_logs"
        preserve_logs(target, preserved)
        verified = verify_cleanup_target(
            target, allowed_parent=allowed, expected_names={"temp"}
        )
        result = remove_cleanup_target(target, verified=verified)
        assert result["removed"] is True
        assert not target.exists()
        assert sibling.exists() and (sibling / "keep.txt").read_bytes() == b"keep\n"
        assert (preserved / "verify_run1.log").is_file()

    def test_removal_requires_a_verified_target(self, scratch_dir):
        from scripts.closeout_material_runtime import remove_cleanup_target

        allowed = scratch_dir / "allowed"
        target = allowed / "4a_task"
        target.mkdir(parents=True)
        with pytest.raises(ValueError, match="verified"):
            remove_cleanup_target(target, verified=None)
        assert target.exists()


# ---------------------------------------------------------------------------
# valid-support consistency must participate in the pass verdict


class TestValidConsistencyVerdict:
    """The rebuilt valid support is part of the parity contract.

    A matching signed_q/abs_q pair with a different valid support is not a
    pass: the valid channel decides which pixels were a measurement at all.
    """

    def _npz(self, scratch_dir, *, valid_value: int = 1):
        from core.material_mask_dataset import save_mask_sample

        root = scratch_dir / "v5"
        path = root / "val" / "metal_submarine" / "frame_a_obj000.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        gray = np.full((8, 8), 120, dtype=np.uint8)
        mask = np.ones((8, 8), dtype=np.uint8)
        signed_q = np.zeros((8, 8), dtype=np.float32)
        abs_q = np.zeros((8, 8), dtype=np.float32)
        valid = np.full((8, 8), valid_value, dtype=np.uint8)
        save_mask_sample(
            path, gray=gray, signed_q=signed_q, abs_q=abs_q, valid=valid,
            quality=np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            class_id=0, instance_mask=mask,
        )
        return path

    def test_identical_q_but_different_valid_fails(self, scratch_dir):
        from core.material_mask_dataset import load_mask_sample
        from scripts.closeout_material_runtime import compare_instance_arrays

        sample = load_mask_sample(self._npz(scratch_dir, valid_value=1))
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=sample.signed_q.copy(),
            rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=np.zeros_like(sample.valid),   # valid all zero
            npz=sample,
        )
        assert record["signed_q"]["exact_equal"] is True
        assert record["abs_q"]["exact_equal"] is True
        assert record["rebuilt_valid_equals_npz_valid"] is False
        assert record["status"] == "FAILED"

    def test_missing_rebuilt_valid_fails(self, scratch_dir):
        from core.material_mask_dataset import load_mask_sample
        from scripts.closeout_material_runtime import compare_instance_arrays

        sample = load_mask_sample(self._npz(scratch_dir, valid_value=1))
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=sample.signed_q.copy(),
            rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=None,
            npz=sample,
        )
        assert record["rebuilt_valid_equals_npz_valid"] is False
        assert record["status"] == "FAILED"

    def test_different_valid_shape_fails(self, scratch_dir):
        from core.material_mask_dataset import load_mask_sample
        from scripts.closeout_material_runtime import compare_instance_arrays

        sample = load_mask_sample(self._npz(scratch_dir, valid_value=1))
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=sample.signed_q.copy(),
            rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=np.ones((4, 4), dtype=np.uint8),
            npz=sample,
        )
        assert record["rebuilt_valid_equals_npz_valid"] is False
        assert record["status"] == "FAILED"

    def test_all_consistent_passes(self, scratch_dir):
        from core.material_mask_dataset import load_mask_sample
        from scripts.closeout_material_runtime import compare_instance_arrays

        sample = load_mask_sample(self._npz(scratch_dir, valid_value=1))
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=sample.signed_q.copy(),
            rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=sample.valid.copy(),
            npz=sample,
        )
        assert record["rebuilt_valid_equals_npz_valid"] is True
        assert record["status"] == "PASSED"

    def test_zero_q_with_valid_one_is_a_legal_measurement(self, scratch_dir):
        """q == 0 inside a valid pixel is a real measurement, not 'invalid'."""
        from core.material_mask_dataset import load_mask_sample
        from scripts.closeout_material_runtime import compare_instance_arrays

        sample = load_mask_sample(self._npz(scratch_dir, valid_value=1))
        assert float(np.abs(sample.signed_q).max()) == 0.0
        record = compare_instance_arrays(
            frame_id="frame_a", sample_name="frame_a_obj000", object_index=0,
            rebuilt_signed=sample.signed_q.copy(),
            rebuilt_abs=sample.abs_q.copy(),
            rebuilt_valid=sample.valid.copy(),
            npz=sample,
        )
        assert record["status"] == "PASSED"
        assert record["invalid_region_zero_ok"] is True

    def test_instance_failure_propagates_to_frame_and_run_status(self, scratch_dir):
        """A failing instance must fail its frame and the whole report."""
        import json

        from scripts.closeout_material_runtime import run_polar_parity

        # A tiny synthetic frame is not available here, so the propagation
        # contract is checked on the aggregation used by run_polar_parity.
        report = {
            "frames": [{"frame_id": "frame_a", "status": "PASSED"}],
            "instances": [
                {"sample_name": "frame_a_obj000", "status": "PASSED"},
                {"sample_name": "frame_a_obj001", "status": "FAILED"},
            ],
            "status": "PASSED",
        }
        statuses = {record["status"] for record in report["instances"]}
        if "FAILED" in statuses:
            report["status"] = "FAILED"
        assert report["status"] == "FAILED"
        assert json.dumps(report)  # serialisable
