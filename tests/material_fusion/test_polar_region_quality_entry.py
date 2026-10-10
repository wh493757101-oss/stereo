"""Entry-level tests for the corrected 2A region-diagnostics entry.

Three confirmed acceptance gaps are fixed here and pinned by tests:

1. protection evidence: real BEFORE/AFTER snapshots over existing assets
   (an all-missing snapshot can never pass);
2. V5 full admission (``admit_mask_dataset`` + annotation sources) runs
   through the REAL project gate BEFORE any per-sample statistics —
   proven with a fail-on-call probe of the statistics function, never by
   mocking the admission itself;
3. measurement semantics: the four-direction shift experiment is a
   ``saved_q_roi_sampling_shift``; a pure-array counterexample shows it
   is NOT a left-right misregistration recompute.

Everything runs on a synthetic but real provenance chain (PNG frames +
YOLO labels -> V4 -> audited V5 -> aligned protocol/cache -> tiny P0
fit). No real dataset, no SGBM, no model forward, no CUDA.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

from core import polar_region_quality as prq
from core.fusion_dataset import dataset_fingerprint
from core.existing_data_protocol import protocol_digest
from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    load_mask_sample,
    mask_dataset_fingerprint,
)
from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    authoritative_sample_index,
    build_cache_binding,
    build_control_features,
    build_polar_stats,
)
from core.polar_region_quality import (
    PERTURBATION_HIGHLIGHT,
    PERTURBATION_TYPE,
    ProtectionError,
    build_protection_snapshot,
    compare_protection_snapshots,
    drift_statistics,
    locate_project_root,
    shift_content,
)
from scripts import diagnose_polar_region_quality as entry
from scripts.diagnose_polar_region_quality import (
    aggregate_records,
    build_subsets,
    compose_drift_summary,
)

from .conftest import build_synthetic_source, build_synthetic_v4, build_synthetic_v5
from .test_cache_admission import CLASS_NAMES, G_DIM
from .test_p0_baseline_verification import build_fit

import sys as _sys

_p0mod = _sys.modules[build_fit.__module__]


# ---------------------------------------------------------------------------
# Synthetic world: source -> V4 -> audited V5 -> aligned protocol/cache/P0


def _v5_rows(v5_root: Path) -> list[dict]:
    with (v5_root / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _protocol_for_v5(rows: list[dict], v5_fingerprint: str) -> dict:
    """A group-disjoint three-fold protocol over the V5 dev pool."""
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] not in ("train", "val"):
            continue
        by_group.setdefault(row["group_name"], []).append(row)
    groups = sorted(by_group)
    assert len(groups) == 8
    assignments = {
        # every group is outer_eval exactly once across the three folds
        "fold0": {
            "outer_eval": groups[0:3],
            "inner_train": groups[3:6],
            "inner_val": groups[6:8],
        },
        "fold1": {
            "outer_eval": groups[3:4] + groups[6:8],
            "inner_train": groups[0:3],
            "inner_val": groups[4:6],
        },
        "fold2": {
            "outer_eval": groups[4:6],
            "inner_train": groups[6:8] + groups[2:3],
            "inner_val": groups[0:2] + groups[3:4],
        },
    }

    def identity(row: dict) -> dict:
        return {
            "sample_name": row["sample_name"],
            "class_id": int(row["class_id"]),
            "class_name": row["class_name"],
            "group": row["group_name"],
            "source_frame": row["source_frame"],
            "object_index": int(row["object_index"]),
            "split": row["split"],
            "npz_path": row["npz_path"],
            "physical_object_id": "unknown",
        }

    def part(group_names: list[str]) -> dict:
        members = []
        for group in group_names:
            for row in by_group[group]:
                members.append(identity(row))
        return {"samples": members}

    folds = []
    for fold_id in ("fold0", "fold1", "fold2"):
        parts = assignments[fold_id]
        folds.append({
            "fold_id": fold_id,
            "outer_eval_groups": list(parts["outer_eval"]),
            "inner_train_groups": list(parts["inner_train"]),
            "inner_val_groups": list(parts["inner_val"]),
            "outer_eval_members": part(parts["outer_eval"]),
            "inner_train_members": part(parts["inner_train"]),
            "inner_val_members": part(parts["inner_val"]),
        })
    protocol = {
        "protocol_version": "material_existing_data_v1",
        "protocol_sha256": "",
        "data": {
            "fingerprint": v5_fingerprint,
            "class_names": list(CLASS_NAMES),
            "dev_pool_splits": ["train", "val"],
        },
        "group_statistics": {
            group: {"gt_instances": len(rows)} for group, rows in by_group.items()
        },
        "folds": {"status": "OK", "conflicts": [], "folds": folds},
        "config": {
            "baselines": {"official_mit_b0": {
                "path": "weights/mit_b0.pth", "sha256": "c" * 64}},
            "features": {"imgsz": 224,
                         "preprocessing_version": "v4_band_224_area_linear_v1"},
        },
    }
    protocol["protocol_sha256"] = protocol_digest(protocol)
    return protocol


def _make_cache_files_for_v5(v5_root: Path, rows: list[dict]):
    """A ``make_cache_files`` replacement aligned with the synthetic V5."""

    def build(feature_dir: Path, protocol: dict, n: int | None = None,
              g_dim: int = G_DIM) -> dict:
        del n  # the world size is fixed by the V5 dev pool
        authority = authoritative_sample_index(protocol)
        names = sorted(authority)
        row_by_name = {row["sample_name"]: row for row in rows}
        rng = np.random.default_rng(13)
        samples = []
        g_rows, c_rows, s_rows, valid_rows, class_rows = [], [], [], [], []
        for name in names:
            row = row_by_name[name]
            sample = load_mask_sample(v5_root / row["npz_path"])
            values, has_valid_s = build_polar_stats(
                signed_q=sample.signed_q, abs_q=sample.abs_q,
                valid=sample.valid, instance_mask=sample.instance_mask)
            control = build_control_features(
                quality=sample.quality, gray=sample.gray,
                instance_mask=sample.instance_mask,
                s_region_nonempty=has_valid_s)
            g_rows.append(rng.normal(size=g_dim).astype(np.float32))
            c_rows.append(control)
            s_rows.append(values)
            valid_rows.append(has_valid_s)
            class_rows.append(authority[name]["identity"]["class_id"])
            samples.append({
                "sample_name": name,
                "npz_path": row["npz_path"],
                "split": row["split"],
                "group": row["group_name"],
                "source_frame": row["source_frame"],
                "object_index": int(row["object_index"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "physical_object_id": "unknown",
                "has_valid_s": bool(has_valid_s),
                "fold_roles": dict(authority[name]["roles"]),
            })
        n = len(names)
        arrays = {
            "g_features": np.asarray(g_rows, dtype=np.float32),
            "c_features": np.asarray(c_rows, dtype=np.float32),
            "s_features": np.asarray(s_rows, dtype=np.float32),
            "s_valid": np.asarray(valid_rows, dtype=bool),
            "class_id": np.asarray(class_rows, dtype=np.int64),
        }
        manifest = {"count": n, "samples": samples}
        definition = {
            "feature_definition_version": "existing_data_probe_v1",
            "data_fingerprint": protocol["data"]["fingerprint"],
            "protocol_sha256": protocol["protocol_sha256"],
            "official_mit_b0": dict(
                protocol["config"]["baselines"]["official_mit_b0"]),
            "imgsz": 224,
            "preprocessing_version": "v4_band_224_area_linear_v1",
            "slot_G": {"dim": g_dim, "definition": "synthetic G"},
            "slot_C": {"keys": list(CONTROL_FEATURE_KEYS)},
            "slot_S": {"keys": list(POLAR_STAT_KEYS)},
        }
        feature_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feature_dir / "feature_cache.npz", **arrays)
        (feature_dir / "feature_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        (feature_dir / "feature_definition.json").write_text(
            json.dumps(definition, indent=2), encoding="utf-8")
        build_cache_binding(feature_dir, protocol)
        return {"names": names, "manifest": manifest,
                "definition": definition, "arrays": arrays}

    return build


@pytest.fixture(scope="module")
def qa_world(module_scratch):
    """Synthetic source -> V4 -> audited V5 -> protocol/cache/P0/config."""
    workspace = module_scratch.resolve()
    source = build_synthetic_source(
        module_scratch / "source", counts=(4, 4, 4), size=64)
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    v5_fingerprint = mask_dataset_fingerprint(v5)
    parent = dataset_fingerprint(v4)
    rows = _v5_rows(v5)

    protocol = _protocol_for_v5(rows, v5_fingerprint)
    protocol_dir = module_scratch / "protocol"
    protocol_dir.mkdir(parents=True)
    protocol_path = protocol_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, sort_keys=True), encoding="utf-8")

    with mock.patch.object(_p0mod, "make_protocol", lambda: protocol), \
            mock.patch.object(_p0mod, "make_cache_files",
                              _make_cache_files_for_v5(v5, rows)):
        fit = build_fit(module_scratch, name="p0fit")
    feature_dir = fit["feature_dir"]

    verification_path = module_scratch / "p0_verification.json"
    verification_path.write_text(json.dumps({"verified": True}),
                                 encoding="utf-8")
    config = {
        "experiment": "material_polar_residual_v1",
        "protocol_source": {"path": str(protocol_path)},
        "features_source": {"path": str(feature_dir)},
        "p0_baseline_source": {
            "fit_dir": str(fit["fit_dir"]),
            "verification": str(verification_path),
        },
        "data": {
            "root": str(v5),
            "fingerprint": v5_fingerprint,
            "class_names": list(CLASS_NAMES),
            "dev_pool_splits": ["train", "val"],
            "excluded_splits": ["test"],
            "parent_v4_fingerprint": parent,
        },
        "training": {"device": "cpu", "epochs": 5},
    }
    config_path = module_scratch / "residual_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False),
                           encoding="utf-8")
    return {
        "workspace": workspace,
        "source": source,
        "v4": v4,
        "v5": v5,
        "v5_fingerprint": v5_fingerprint,
        "parent": parent,
        "rows": rows,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "feature_dir": feature_dir,
        "fit_dir": fit["fit_dir"],
        "config": config,
        "config_path": config_path,
        "cache_samples": fit["manifest"]["samples"],
    }


def _write_config(path: Path, config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _copy_v5(world, scratch: Path, name: str) -> Path:
    target = scratch / name
    shutil.copytree(world["v5"], target)
    return target


class _Probe:
    """Fail-on-call (or record-and-forward) probe for the statistics fn."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, sample):
        self.calls.append("called")
        from core.polar_region_quality import sample_region_diagnostics

        return sample_region_diagnostics(sample)

    def refuse(self, sample):
        # record BEFORE raising: main() converts any exception into a
        # non-zero exit code, so a test that only checks the exit code
        # cannot distinguish "statistics never started" from "statistics
        # started and the probe's error was swallowed into the refusal".
        self.calls.append("refused")
        raise AssertionError(
            "region statistics must not run before the V5 admission")

    def explode(self, sample):
        self.calls.append("exploded")
        raise RuntimeError("synthetic statistics fault")


@pytest.fixture()
def refusing_probe():
    probe = _Probe()
    with mock.patch.object(entry.polar_region_quality,
                           "sample_region_diagnostics", probe.refuse):
        yield probe


@pytest.fixture()
def recording_probe():
    probe = _Probe()
    original = entry.polar_region_quality.sample_region_diagnostics

    def call(sample):
        probe.calls.append("called")
        return original(sample)

    with mock.patch.object(entry.polar_region_quality,
                           "sample_region_diagnostics", call):
        yield probe


def _argv(world, output_dir: Path, *, config: Path | None = None,
          fit_dir: Path | None = None) -> list[str]:
    return [
        "--config", str(config or world["config_path"]),
        "--fit-dir", str(fit_dir or world["fit_dir"]),
        "--output-dir", str(output_dir),
    ]


# ---------------------------------------------------------------------------
# 1: project root location is CWD-independent


class TestLocateProjectRoot:
    def test_locates_root_regardless_of_cwd(self, scratch_dir, monkeypatch):
        root = scratch_dir / "proj"
        (root / "scripts").mkdir(parents=True)
        (root / "core").mkdir()
        (root / "scripts" / "entry.py").write_text("# entry")
        (root / "core" / "polar_region_quality.py").write_text("# marker")
        monkeypatch.chdir(scratch_dir)  # NOT the project root
        assert locate_project_root(root / "scripts" / "entry.py") == \
            root.resolve()

    def test_missing_marker_is_controlled_failure(self, scratch_dir):
        root = scratch_dir / "bare"
        (root / "scripts").mkdir(parents=True)
        (root / "scripts" / "entry.py").write_text("# entry")
        with pytest.raises(ProtectionError, match="marker"):
            locate_project_root(root / "scripts" / "entry.py")


# ---------------------------------------------------------------------------
# 2-4: protection snapshots


class TestProtectionSnapshots:
    def _targets(self, scratch: Path) -> list[str]:
        (scratch / "asset.txt").write_text("payload")
        nested = scratch / "nested"
        nested.mkdir()
        (nested / "a.txt").write_text("A")
        (nested / "b.bin").write_bytes(b"\x00\x01")
        return ["asset.txt", "nested"]

    def test_snapshot_records_files_dirs_and_digests(self, scratch_dir):
        targets = self._targets(scratch_dir)
        snapshot = build_protection_snapshot(scratch_dir, targets)
        record = snapshot["targets"]["asset.txt"]
        assert record["kind"] == "file" and record["bytes"] == 7
        assert len(record["sha256"]) == 64
        nested = snapshot["targets"]["nested"]
        assert nested["kind"] == "dir"
        assert sorted(nested["files"]) == ["a.txt", "b.bin"]
        assert nested["file_count"] == 2

    def test_missing_required_asset_is_reported(self, scratch_dir):
        build_protection_snapshot(scratch_dir, ["asset.txt"])
        snapshot = build_protection_snapshot(scratch_dir, ["absent.txt"])
        missing = entry.missing_required_targets(snapshot)
        assert missing == ["absent.txt"]

    def test_all_missing_snapshot_cannot_pass(self, scratch_dir):
        before = build_protection_snapshot(scratch_dir, ["ghost.txt"])
        after = build_protection_snapshot(scratch_dir, ["ghost.txt"])
        assert before["targets"]["ghost.txt"]["kind"] == "missing"
        with pytest.raises(ProtectionError, match="missing"):
            compare_protection_snapshots(before, after)

    def test_content_change_is_detected(self, scratch_dir):
        targets = self._targets(scratch_dir)
        before = build_protection_snapshot(scratch_dir, targets)
        (scratch_dir / "asset.txt").write_text("payload!")
        after = build_protection_snapshot(scratch_dir, targets)
        with pytest.raises(ProtectionError, match="asset.txt"):
            compare_protection_snapshots(before, after)

    def test_file_set_change_is_detected(self, scratch_dir):
        targets = self._targets(scratch_dir)
        before = build_protection_snapshot(scratch_dir, targets)
        (scratch_dir / "nested" / "c.txt").write_text("C")
        after = build_protection_snapshot(scratch_dir, targets)
        with pytest.raises(ProtectionError, match="file set"):
            compare_protection_snapshots(before, after)

    def test_length_change_is_detected(self, scratch_dir):
        targets = self._targets(scratch_dir)
        before = build_protection_snapshot(scratch_dir, targets)
        (scratch_dir / "nested" / "b.bin").write_bytes(b"\x00")
        after = build_protection_snapshot(scratch_dir, targets)
        with pytest.raises(ProtectionError, match="b.bin"):
            compare_protection_snapshots(before, after)

    def test_escaping_target_is_refused(self, scratch_dir):
        outside = scratch_dir / "outside.txt"
        outside.write_text("x")
        with pytest.raises(ProtectionError, match="project root"):
            build_protection_snapshot(scratch_dir / "nested",
                                      ["../outside.txt"])

    def test_descendant_junction_outside_root_is_refused(self, scratch_dir):
        """Real directory junction (no elevation needed): a descendant
        link node resolving outside the snapshot root must be refused
        BEFORE any of its content is hashed."""
        outside = scratch_dir / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("S")
        proj = scratch_dir / "proj"
        (proj / "target").mkdir(parents=True)
        (proj / "target" / "good.txt").write_text("G")
        link = proj / "target" / "evil"
        creation = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace")
        try:
            if creation.returncode != 0:
                pytest.skip(
                    "junction creation unavailable: "
                    f"{creation.stderr.strip() or creation.returncode}")
            real_snapshot = prq._snapshot_file

            def spy(path):
                calls.append(Path(path))
                return real_snapshot(path)

            calls: list[Path] = []
            with mock.patch.object(prq, "_snapshot_file", spy):
                with pytest.raises(ProtectionError, match="evil"):
                    build_protection_snapshot(proj, ["target"])
            assert not any("evil" in str(path) for path in calls)
            assert not any("secret" in str(path) for path in calls)
        finally:
            # remove the junction ITSELF so the conftest-owned cleanup
            # (which refuses trees containing reparse points) stays safe;
            # os.rmdir on a junction removes only the link, never the target
            if link.is_dir() or link.exists():
                os.rmdir(str(link))

    def test_descendant_file_resolving_outside_is_refused_simulated(
            self, scratch_dir, monkeypatch):
        """SIMULATED resolution escape (monkeypatched ``_real_path``):
        Windows file symlinks need elevation, which this task does not
        request. The real directory-junction case is covered by
        ``test_descendant_junction_outside_root_is_refused``."""
        proj = scratch_dir / "proj2"
        (proj / "target").mkdir(parents=True)
        (proj / "target" / "innocent.txt").write_text("I")

        def fake_real_path(path):
            base = Path(os.path.realpath(path))
            if base.name == "innocent.txt":
                return proj.parent / "outside_root" / "innocent.txt"
            return base

        monkeypatch.setattr(prq, "_real_path", fake_real_path)
        real_snapshot = prq._snapshot_file
        with mock.patch.object(prq, "_snapshot_file",
                               side_effect=real_snapshot) as spy:
            with pytest.raises(ProtectionError, match="innocent"):
                build_protection_snapshot(proj, ["target"])
        assert spy.call_count == 0

    def test_plain_directory_snapshot_still_hashes_in_root(self, scratch_dir):
        proj = scratch_dir / "proj3"
        (proj / "target").mkdir(parents=True)
        (proj / "target" / "a.txt").write_text("A")
        real_snapshot = prq._snapshot_file
        with mock.patch.object(prq, "_snapshot_file",
                               side_effect=real_snapshot) as spy:
            snapshot = build_protection_snapshot(proj, ["target"])
        assert spy.call_count == 1
        assert snapshot["targets"]["target"]["files"]["a.txt"]["bytes"] == 1


# ---------------------------------------------------------------------------
# 11-12: drift statistics — mean and max reported separately


class TestDriftStatistics:
    def test_mean_and_max_are_reported_separately(self):
        values = [0.001, 0.002, 0.003, 0.028]
        summary = drift_statistics(values)
        assert summary["n"] == 4
        assert summary["max"] == pytest.approx(0.028)
        assert summary["mean"] == pytest.approx(0.0085)
        assert summary["p95"] < summary["max"]
        assert summary["mean"] < summary["max"]

    def test_empty_subset_fabricates_no_success(self):
        summary = drift_statistics([])
        assert summary["n"] == 0
        for key in ("mean", "p50", "p95", "max"):
            assert summary[key] is None

    def test_nonfinite_values_are_refused(self):
        with pytest.raises(ValueError, match="finite"):
            drift_statistics([0.1, float("nan")])


# ---------------------------------------------------------------------------
# 10: the shift experiment is NOT a left-right misregistration recompute


class TestPerturbationSemantics:
    def test_saved_q_roi_shift_differs_from_right_misalignment(self):
        eps = 1e-6
        pattern = np.array([50, 200] * 8, dtype=np.uint8)
        left = pattern.reshape(1, 16)
        right = left.copy()
        left_f = left.astype(np.float64)
        right_f = right.astype(np.float64)

        # the saved q maps: identical left/right -> q == 0 everywhere
        q_saved = np.float32(
            (left_f - right_f) / (left_f + right_f + eps))
        assert np.array_equal(q_saved, np.zeros_like(q_saved))

        # (a) ROI sampling shift of the SAVED q maps: still exactly zero
        _, shifted_q, _, _ = shift_content(
            left, q_saved, np.abs(q_saved),
            np.ones_like(left, dtype=bool), (0, 1))
        assert np.array_equal(shifted_q, np.zeros_like(shifted_q))

        # (b) fixed left, right misaligned by 1px, q RECOMPUTED by the
        # generation formula: |q| = 150/250 = 0.6 at every shifted pixel
        # (column 0 has no shifted sample and saturates at q = +1)
        right_misaligned = np.zeros_like(right_f)
        right_misaligned[0, 1:] = right_f[0, :-1]
        q_recomputed = (left_f - right_misaligned) / (
            left_f + right_misaligned + eps)
        assert float(np.abs(q_recomputed[0, 1:]).min()) == \
            pytest.approx(0.6, abs=1e-3)
        assert float(np.abs(q_recomputed[0, 1:]).max()) == \
            pytest.approx(0.6, abs=1e-3)

    def test_perturbation_type_is_declared(self):
        assert PERTURBATION_TYPE == "saved_q_roi_sampling_shift"


# ---------------------------------------------------------------------------
# 12: aggregation of empty subsets and the missing-S distinction


class TestAggregationSemantics:
    def test_empty_subset_reports_zero_samples_without_ratios(self):
        aggregate = aggregate_records([])
        assert set(aggregate) == {"samples", "aggregation_unit"}
        assert aggregate["samples"] == 0

    def test_missing_s_and_rule_fallback_are_distinct(self):
        record = {
            "sample_name": "s0",
            "diagnostics": {
                "counts": {"mask_pixels": 4, "interior_pixels": 0,
                           "band_pixels": 4, "region_pixels": 0,
                           "interior_valid_pixels": 0,
                           "band_valid_pixels": 0},
                "ratios": {"interior_valid_ratio": None,
                           "band_valid_ratio": None,
                           "region_coverage_of_mask": 0.0},
                "original_stats": None,
                "perturbation": {"max_abs_delta_original": {},
                                 "max_abs_delta_rule": {}},
                "truncation": {"mask_touches_crop_border": False},
            },
            "rule": {"has_valid_s": False, "fallback_used": False,
                     "chosen_region": "none"},
        }
        aggregate = aggregate_records([record])
        assert aggregate["samples"] == 1
        assert aggregate["has_valid_s"] == 0
        assert aggregate["missing_s_samples"] == 1
        assert aggregate["rule_fallback_used"] == 0

    def test_union_subset_is_labelled_as_summary(self):
        records = [{"fold_roles": {"fold0": "inner_train"}}]
        subsets = build_subsets(records)
        assert subsets["inner_train_all_rule_design"]["role"] == \
            "summary_union_of_inner_train"


# ---------------------------------------------------------------------------
# 5-9: the real entry over the synthetic world


# Stable fragments of the REAL admission-gate errors (not full strings):
# the entry prints "REFUSED: {type}: {message}" to stderr.
_TAMPERED_FRAGMENT = "changed since the audit passed"
_LABEL_FRAGMENT = "annotation source files recorded in the manifest are missing"
# the entry's own cross-check fires before the V5 gate for a conflicting
# config fingerprint; both refusals are admission-stage fingerprint errors
_FINGERPRINT_FRAGMENTS = ("fingerprint", "does not match")
_DIAGNOSTIC_PRODUCTS = (
    "admission.json",
    "per_sample_diagnostics.jsonl",
    "aggregates.json",
    "drift_summary.json",
    "diagnostics_summary.json",
)


def _assert_refused_for_admission(capsys, output_dir: Path,
                                  *fragments: str) -> None:
    """The refusal must come from the REAL admission gate.

    Checks stable error fragments on actual stderr, that the probe's
    own AssertionError never became the refusal reason, and that no
    normal diagnostics product was written.
    """
    captured = capsys.readouterr()
    assert "REFUSED:" in captured.err
    for fragment in fragments:
        assert fragment in captured.err, fragment
    assert "AssertionError" not in captured.err
    for product in _DIAGNOSTIC_PRODUCTS:
        assert not (output_dir / product).exists(), product


class TestEntryAdmissionOrdering:
    def test_tampered_npz_refused_before_statistics(
            self, qa_world, scratch_dir, refusing_probe, capsys):
        v5_copy = _copy_v5(qa_world, scratch_dir, "v5_tampered")
        # flip one saved q value; the mask (and its pixel count) is intact
        victim = qa_world["rows"][0]["npz_path"]
        sample = load_mask_sample(v5_copy / victim)
        sample.signed_q[0, 0] = np.float32(sample.signed_q[0, 0] + 0.5)
        from core.material_mask_dataset import save_mask_sample

        save_mask_sample(
            v5_copy / victim, gray=sample.gray,
            signed_q=sample.signed_q, abs_q=sample.abs_q,
            valid=sample.valid, quality=sample.quality,
            class_id=sample.class_id,
            instance_mask=sample.instance_mask)
        config = dict(qa_world["config"])
        config["data"] = {**qa_world["config"]["data"],
                          "root": str(v5_copy)}
        config_path = _write_config(scratch_dir / "configs" / "tampered.yaml",
                                    config)
        output = scratch_dir / "out"
        code = entry.main(_argv(qa_world, output, config=config_path))
        assert code == 1
        assert refusing_probe.calls == []
        _assert_refused_for_admission(capsys, output, _TAMPERED_FRAGMENT)

    def test_label_source_missing_refused_before_statistics(
            self, qa_world, scratch_dir, refusing_probe, capsys):
        v5_copy = _copy_v5(qa_world, scratch_dir, "v5_nolabel")
        label_path = qa_world["rows"][0]["label_path"]
        # the synthetic label file lives under the repo root (workspace)
        from core.material_mask_dataset import PROJECT_ROOT

        absolute = PROJECT_ROOT / label_path
        assert absolute.is_file()
        payload = absolute.read_bytes()
        try:
            absolute.unlink()
            config = dict(qa_world["config"])
            config["data"] = {**qa_world["config"]["data"],
                              "root": str(v5_copy)}
            config_path = _write_config(
                scratch_dir / "configs" / "nolabel.yaml", config)
            output = scratch_dir / "out"
            code = entry.main(_argv(qa_world, output,
                                    config=config_path))
        finally:
            absolute.write_bytes(payload)
        assert code == 1
        assert refusing_probe.calls == []
        _assert_refused_for_admission(capsys, output, _LABEL_FRAGMENT)

    def test_fingerprint_conflict_refused_before_statistics(
            self, qa_world, scratch_dir, refusing_probe, capsys):
        config = dict(qa_world["config"])
        config["data"] = {**qa_world["config"]["data"],
                          "fingerprint": "a" * 64}
        config_path = _write_config(scratch_dir / "configs" / "conflict.yaml",
                                    config)
        output = scratch_dir / "out"
        code = entry.main(_argv(qa_world, output, config=config_path))
        assert code == 1
        assert refusing_probe.calls == []
        _assert_refused_for_admission(capsys, output,
                                      *_FINGERPRINT_FRAGMENTS)


class TestProbeCalibration:
    """Calibration of the statistics probes themselves.

    ``main()`` converts any exception into a non-zero exit code, so a
    probe that raises without recording cannot prove "statistics never
    started" from the exit code alone. These tests pin that the probes
    record the call BEFORE raising, and that a probe actually executed
    by the entry leaves a record that survives main()'s exception
    conversion. This is probe calibration — NOT evidence that the
    admission passed.
    """

    def test_refusing_probe_records_before_raising(self):
        probe = _Probe()
        with pytest.raises(AssertionError):
            probe.refuse(object())
        assert probe.calls == ["refused"]

    def test_exploding_probe_records_before_raising(self):
        probe = _Probe()
        with pytest.raises(RuntimeError):
            probe.explode(object())
        assert probe.calls == ["exploded"]

    def test_entry_exception_conversion_cannot_hide_calls(
            self, qa_world, scratch_dir):
        probe = _Probe()
        with mock.patch.object(entry.polar_region_quality,
                               "sample_region_diagnostics",
                               probe.explode):
            code = entry.main(_argv(qa_world, scratch_dir / "out_fault"))
        assert code == 1
        assert probe.calls == ["exploded"]
        # the V5 admission passed (admission.json exists) but the probe's
        # exception converted to a refusal: no diagnostics products
        assert (scratch_dir / "out_fault" / "admission.json").is_file()
        for product in _DIAGNOSTIC_PRODUCTS:
            if product != "admission.json":
                assert not (scratch_dir / "out_fault"
                            / product).exists(), product


class TestEntryRun:
    def test_legal_world_passes_end_to_end(
            self, qa_world, scratch_dir, recording_probe):
        output = scratch_dir / "diagnostics"
        code = entry.main(_argv(qa_world, output))
        assert code == 0
        assert len(recording_probe.calls) == \
            len(qa_world["cache_samples"])

        admission = json.loads(
            (output / "admission.json").read_text(encoding="utf-8"))
        assert admission["expected_fingerprint"] == \
            qa_world["v5_fingerprint"]
        assert admission["actual_fingerprint"] == \
            qa_world["v5_fingerprint"]
        assert admission["v4_parent_expected"] == qa_world["parent"]
        assert admission["verified_files"] > 0
        assert "mask content digest vs audit" in \
            admission["checks_performed"]
        assert any("verify_mask_sources" in check
                   for check in admission["checks_performed"])

        protection = json.loads(
            (output / "protection_report.json").read_text(encoding="utf-8"))
        assert protection["comparison"]["passed"] is True
        assert protection["historical_protection_evidence"]["status"] == \
            "INVALID_PROTECTION_EVIDENCE"
        assert protection["historical_protection_evidence"][
            "historical_before_after_integrity"] == "UNVERIFIED"

        summary = json.loads(
            (output / "diagnostics_summary.json").read_text(
                encoding="utf-8"))
        assert summary["perturbation"]["perturbation_type"] == \
            "saved_q_roi_sampling_shift"
        assert summary["perturbation"]["interpretation_limit"] is True
        for key in PERTURBATION_HIGHLIGHT:
            block = summary["perturbation"]["drift"][key]
            assert {"mean", "p50", "p95", "max"} <= set(block["original"])
            assert block["original"]["max"] >= block["original"]["mean"]

        aggregates = json.loads(
            (output / "aggregates.json").read_text(encoding="utf-8"))
        rule_design = aggregates["inner_train_all_rule_design"]
        assert rule_design["role"] == "summary_union_of_inner_train"
        assert "aggregation_unit" in rule_design

    def test_existing_output_dir_is_refused(self, qa_world, scratch_dir):
        output = scratch_dir / "already"
        output.mkdir()
        assert entry.main(_argv(qa_world, output)) == 1

    def test_output_inside_input_dir_is_refused(self, qa_world,
                                                scratch_dir):
        output = Path(qa_world["config"]["data"]["root"]) / "nested_out"
        assert entry.main(_argv(qa_world, output)) == 1

    def test_missing_required_asset_is_controlled_refusal(
            self, qa_world, scratch_dir):
        # a config whose protocol path does not exist must be a controlled
        # refusal (non-zero), never an all-missing "equal" pass
        config = dict(qa_world["config"])
        config["protocol_source"] = {
            **qa_world["config"]["protocol_source"],
            "path": str(scratch_dir / "absent_protocol.json")}
        config_path = _write_config(scratch_dir / "configs" / "missing.yaml",
                                    config)
        assert entry.main(_argv(qa_world, scratch_dir / "out",
                                config=config_path)) == 1

    def test_inputs_unchanged_after_successful_run(
            self, qa_world, scratch_dir, module_scratch):
        targets = ["v5", "v4", "source"]
        before = build_protection_snapshot(module_scratch, targets)
        output = scratch_dir / "diagnostics2"
        assert entry.main(_argv(qa_world, output)) == 0
        after = build_protection_snapshot(module_scratch, targets)
        comparison = compare_protection_snapshots(before, after)
        assert comparison["passed"] is True
