"""Cache binding, per-fold authority and array admission tests.

All fixtures are synthetic (tiny arrays, tiny protocol dicts); no real
cache, protocol or dataset is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.existing_data_protocol import BudgetExceeded, BudgetWatch
from core.material_probe import (
    BOUND_CACHE_FILES,
    CACHE_BINDING_VERSION,
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    authoritative_sample_index,
    build_cache_binding,
    verify_cache_arrays,
    verify_cache_binding,
    verify_cache_manifest,
)

G_DIM = 4
CLASS_NAMES = ("metal_submarine", "plastic_submarine",
               "plastic_fish", "real_fish")


def _sample(name: str, group: str, split: str, class_id: int,
            object_index: int) -> dict:
    frame = f"frame_{group}_{object_index:03d}"
    return {
        "sample_name": name,
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "group": group,
        "source_frame": frame,
        "object_index": object_index,
        "split": split,
        "npz_path": f"{split}/{CLASS_NAMES[class_id]}/{name}.npz",
        "physical_object_id": "unknown",
    }


def _members(*names_samples):
    return {"samples": [dict(sample) for sample, _role in names_samples]}


def make_protocol() -> dict:
    """Three groups x three samples, two folds, whole groups per part."""
    groups = {
        "g0": ("train", [0, 1, 2]),
        "g1": ("val", [1, 2, 3]),
        "g2": ("train", [2, 3, 0]),
    }
    samples = {}
    for group, (split, classes) in groups.items():
        for index, class_id in enumerate(classes):
            name = f"{group}_obj{index:03d}"
            samples[name] = _sample(name, group, split, class_id, index)

    def part(sample_names):
        return {"samples": [dict(samples[name]) for name in sample_names]}

    folds = [
        {
            "fold_id": "fold0",
            "outer_eval_groups": ["g0"],
            "inner_train_groups": ["g1"],
            "inner_val_groups": ["g2"],
            "outer_eval_members": part(
                ["g0_obj000", "g0_obj001", "g0_obj002"]),
            "inner_train_members": part(
                ["g1_obj000", "g1_obj001", "g1_obj002"]),
            "inner_val_members": part(
                ["g2_obj000", "g2_obj001", "g2_obj002"]),
        },
        {
            "fold_id": "fold1",
            "outer_eval_groups": ["g1"],
            "inner_train_groups": ["g2"],
            "inner_val_groups": ["g0"],
            "outer_eval_members": part(
                ["g1_obj000", "g1_obj001", "g1_obj002"]),
            "inner_train_members": part(
                ["g2_obj000", "g2_obj001", "g2_obj002"]),
            "inner_val_members": part(
                ["g0_obj000", "g0_obj001", "g0_obj002"]),
        },
        {
            "fold_id": "fold2",
            "outer_eval_groups": ["g2"],
            "inner_train_groups": ["g0"],
            "inner_val_groups": ["g1"],
            "outer_eval_members": part(
                ["g2_obj000", "g2_obj001", "g2_obj002"]),
            "inner_train_members": part(
                ["g0_obj000", "g0_obj001", "g0_obj002"]),
            "inner_val_members": part(
                ["g1_obj000", "g1_obj001", "g1_obj002"]),
        },
    ]
    from core.existing_data_protocol import protocol_digest

    protocol = {
        "protocol_version": "material_existing_data_v1",
        "protocol_sha256": "",
        "data": {
            "fingerprint": "b" * 64,
            "class_names": list(CLASS_NAMES),
            "dev_pool_splits": ["train", "val"],
        },
        "group_statistics": {
            name: {"gt_instances": 3} for name in groups
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


def make_cache_files(feature_dir: Path, protocol: dict,
                     n: int = 9, g_dim: int = G_DIM) -> dict:
    """A synthetic but fully self-consistent feature cache."""
    authority = authoritative_sample_index(protocol)
    names = sorted(authority)
    rng = np.random.default_rng(0)
    g = rng.normal(size=(n, g_dim)).astype(np.float32)
    c = rng.normal(size=(n, len(CONTROL_FEATURE_KEYS))).astype(np.float32)
    c[:, CONTROL_FEATURE_KEYS.index("s_region_nonempty_flag")] = 1.0
    s = rng.normal(size=(n, len(POLAR_STAT_KEYS))).astype(np.float32)
    s_valid = np.ones(n, dtype=bool)
    class_ids = np.asarray([authority[name]["identity"]["class_id"]
                            for name in names], dtype=np.int64)
    manifest = {
        "samples": [
            {
                "sample_name": name,
                "npz_path": authority[name]["identity"]["npz_path"],
                "split": authority[name]["identity"]["split"],
                "group": authority[name]["identity"]["group"],
                "source_frame": authority[name]["identity"]["source_frame"],
                "object_index": authority[name]["identity"]["object_index"],
                "class_id": authority[name]["identity"]["class_id"],
                "class_name": authority[name]["identity"]["class_name"],
                "physical_object_id": "unknown",
                "has_valid_s": True,
                "fold_roles": dict(authority[name]["roles"]),
            }
            for name in names
        ],
        "count": n,
    }
    definition = {
        "feature_definition_version": "existing_data_probe_v1",
        "data_fingerprint": protocol["data"]["fingerprint"],
        "protocol_sha256": protocol["protocol_sha256"],
        "official_mit_b0": dict(
            protocol["config"]["baselines"]["official_mit_b0"]),
        "preprocessing_version":
            protocol["config"]["features"]["preprocessing_version"],
        "imgsz": protocol["config"]["features"]["imgsz"],
        "class_names": list(CLASS_NAMES),
        "slot_G": {"dim": g_dim, "definition": "test G"},
        "slot_C": {"keys": list(CONTROL_FEATURE_KEYS)},
        "slot_S": {"keys": list(POLAR_STAT_KEYS)},
    }
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_dir / "feature_cache.npz",
                        g_features=g, c_features=c, s_features=s,
                        s_valid=s_valid, class_id=class_ids)
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (feature_dir / "feature_definition.json").write_text(
        json.dumps(definition, indent=2), encoding="utf-8")
    return {
        "names": names,
        "manifest": manifest,
        "definition": definition,
        "arrays": {"g_features": g, "c_features": c, "s_features": s,
                   "s_valid": s_valid, "class_id": class_ids},
    }


class TestCacheBinding:
    def test_build_and_verify_round_trip(self, scratch_dir):
        protocol = make_protocol()
        files = make_cache_files(scratch_dir / "cache", protocol)
        binding = build_cache_binding(scratch_dir / "cache", protocol)
        assert binding["binding_version"] == CACHE_BINDING_VERSION
        assert binding["protocol_sha256"] == protocol["protocol_sha256"]
        assert binding["data_fingerprint"] == protocol["data"]["fingerprint"]
        verified = verify_cache_binding(scratch_dir / "cache", protocol)
        assert verified["files"]["feature_cache.npz"]["bytes"] > 0
        assert files["names"]

    def test_tampered_cache_bytes_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        build_cache_binding(scratch_dir / "cache", protocol)
        cache = dict(np.load(scratch_dir / "cache" / "feature_cache.npz"))
        cache["g_features"][0, 0] += 1.0  # silent content change
        np.savez_compressed(scratch_dir / "cache" / "feature_cache.npz",
                            **cache)
        with pytest.raises(ValueError, match="does not match the binding"):
            verify_cache_binding(scratch_dir / "cache", protocol)

    def test_tampered_manifest_bytes_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        build_cache_binding(scratch_dir / "cache", protocol)
        manifest = json.loads(
            (scratch_dir / "cache" / "feature_manifest.json")
            .read_text(encoding="utf-8"))
        manifest["count"] = 99
        (scratch_dir / "cache" / "feature_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match the binding"):
            verify_cache_binding(scratch_dir / "cache", protocol)

    def test_tampered_definition_bytes_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        build_cache_binding(scratch_dir / "cache", protocol)
        definition = json.loads(
            (scratch_dir / "cache" / "feature_definition.json")
            .read_text(encoding="utf-8"))
        definition["imgsz"] = 320
        (scratch_dir / "cache" / "feature_definition.json").write_text(
            json.dumps(definition), encoding="utf-8")
        with pytest.raises(ValueError, match="does not match the binding"):
            verify_cache_binding(scratch_dir / "cache", protocol)

    def test_missing_binding_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        with pytest.raises(FileNotFoundError, match="binding record missing"):
            verify_cache_binding(scratch_dir / "cache", protocol)

    def test_wrong_protocol_digest_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        build_cache_binding(scratch_dir / "cache", protocol)
        other = make_protocol()
        other["protocol_sha256"] = "e" * 64
        with pytest.raises(ValueError, match="protocol digest"):
            verify_cache_binding(scratch_dir / "cache", other)

    def test_wrong_binding_version_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        binding = build_cache_binding(scratch_dir / "cache", protocol)
        binding["binding_version"] = "v0"
        (scratch_dir / "cache" / "cache_binding.json").write_text(
            json.dumps(binding), encoding="utf-8")
        with pytest.raises(ValueError, match="binding version"):
            verify_cache_binding(scratch_dir / "cache", protocol)

    def test_bad_digest_format_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        binding = build_cache_binding(scratch_dir / "cache", protocol)
        binding["files"]["feature_cache.npz"]["sha256"] = "xyz"
        (scratch_dir / "cache" / "cache_binding.json").write_text(
            json.dumps(binding), encoding="utf-8")
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            verify_cache_binding(scratch_dir / "cache", protocol)

    def test_missing_bound_file_refused(self, scratch_dir):
        protocol = make_protocol()
        make_cache_files(scratch_dir / "cache", protocol)
        build_cache_binding(scratch_dir / "cache", protocol)
        (scratch_dir / "cache" / "feature_definition.json").unlink()
        with pytest.raises(FileNotFoundError, match="bound cache file"):
            verify_cache_binding(scratch_dir / "cache", protocol)


class TestAuthoritativeSampleIndex:
    def test_valid_protocol_passes(self):
        authority = authoritative_sample_index(make_protocol())
        assert len(authority) == 9
        for name, entry in authority.items():
            assert set(entry["roles"]) == {"fold0", "fold1", "fold2"}
            assert sorted(entry["roles"].values()).count("outer_eval") == 1

    def test_duplicate_in_one_fold_and_missing_in_other_refused(self):
        # swap: g0_obj000 gets fold1's inner_train role; g0_obj001 loses
        # fold1 entirely -> total counts unchanged, per-fold broken
        protocol = make_protocol()
        fold1 = protocol["folds"]["folds"][1]
        moved = fold1["inner_val_members"]["samples"].pop(0)
        fold1["inner_train_members"]["samples"].append(moved)
        with pytest.raises(ValueError, match="per-fold validation"):
            authoritative_sample_index(protocol)

    def test_outer_eval_twice_refused(self):
        protocol = make_protocol()
        fold1 = protocol["folds"]["folds"][1]
        # move one inner_val sample of fold1 into its outer_eval
        moved = fold1["inner_val_members"]["samples"].pop(0)
        fold1["outer_eval_members"]["samples"].append(moved)
        with pytest.raises(ValueError, match="outer_eval held 2 times"):
            authoritative_sample_index(protocol)

    def test_group_crossing_refused(self):
        protocol = make_protocol()
        fold0 = protocol["folds"]["folds"][0]
        # put a g1 sample into fold0's inner_val (whose group set is g2)
        stranger = sample_copy(protocol, "g1_obj000")
        fold0["inner_val_members"]["samples"].append(stranger)
        with pytest.raises(ValueError, match="is not among"):
            authoritative_sample_index(protocol)

    def test_test_split_member_refused(self):
        protocol = make_protocol()
        sample = protocol["folds"]["folds"][0]["inner_val_members"][
            "samples"][0]
        sample["split"] = "test"
        with pytest.raises(ValueError, match="test member"):
            authoritative_sample_index(protocol)


def sample_copy(protocol: dict, name: str) -> dict:
    """A copy of a member record wherever it appears in the protocol."""
    for fold in protocol["folds"]["folds"]:
        for part in ("outer_eval", "inner_train", "inner_val"):
            for sample in fold[f"{part}_members"]["samples"]:
                if sample["sample_name"] == name:
                    return dict(sample)
    raise AssertionError(name)


class TestVerifyCacheManifest:
    def test_self_consistent_manifest_passes(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        verified = verify_cache_manifest(
            files["manifest"], authority, protocol)
        assert len(verified) == 9

    def test_row_order_violation_refused(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        files["manifest"]["samples"].reverse()
        with pytest.raises(ValueError, match="row order"):
            verify_cache_manifest(files["manifest"], authority, protocol)

    def test_role_swap_refused(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        # equal-count swap of two samples' fold1 roles (g0_obj000 holds
        # inner_val, g1_obj000 holds outer_eval): totals unchanged,
        # per-sample roles broken
        samples = files["manifest"]["samples"]
        a = next(s for s in samples if s["sample_name"] == "g0_obj000")
        b = next(s for s in samples if s["sample_name"] == "g1_obj000")
        a["fold_roles"]["fold1"], b["fold_roles"]["fold1"] = (
            b["fold_roles"]["fold1"], a["fold_roles"]["fold1"])
        with pytest.raises(ValueError, match="role"):
            verify_cache_manifest(files["manifest"], authority, protocol)

    def test_class_change_refused(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        files["manifest"]["samples"][0]["class_id"] = 3
        with pytest.raises(ValueError, match="class_id"):
            verify_cache_manifest(files["manifest"], authority, protocol)

    def test_group_change_refused(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        files["manifest"]["samples"][0]["group"] = "g9"
        with pytest.raises(ValueError, match="group"):
            verify_cache_manifest(files["manifest"], authority, protocol)

    def test_npz_path_change_refused(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        files["manifest"]["samples"][0]["npz_path"] = "other.npz"
        with pytest.raises(ValueError, match="npz_path"):
            verify_cache_manifest(files["manifest"], authority, protocol)

    def test_non_boolean_has_valid_s_refused(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        files["manifest"]["samples"][0]["has_valid_s"] = 1
        with pytest.raises(ValueError, match="strict boolean"):
            verify_cache_manifest(files["manifest"], authority, protocol)


def scratch_tmp() -> str:
    import tempfile

    return tempfile.mkdtemp(dir="D:/Python/Project/my_data/.test_artifacts")


class TestVerifyCacheArrays:
    def _cache(self):
        protocol = make_protocol()
        authority = authoritative_sample_index(protocol)
        files = make_cache_files(Path(scratch_tmp()) / "cache", protocol)
        return protocol, files

    def test_valid_arrays_pass(self):
        protocol, files = self._cache()
        verify_cache_arrays(files["arrays"], files["manifest"]["samples"],
                            files["definition"], protocol)

    def test_s_valid_uint8_refused(self):
        protocol, files = self._cache()
        arrays = dict(files["arrays"])
        arrays["s_valid"] = arrays["s_valid"].astype(np.uint8)
        with pytest.raises(ValueError, match="strict boolean"):
            verify_cache_arrays(arrays, files["manifest"]["samples"],
                                files["definition"], protocol)

    def test_class_id_mismatch_refused(self):
        protocol, files = self._cache()
        arrays = dict(files["arrays"])
        arrays["class_id"] = arrays["class_id"].copy()
        arrays["class_id"][0] = (arrays["class_id"][0] + 1) % 4
        with pytest.raises(ValueError, match="class_id"):
            verify_cache_arrays(arrays, files["manifest"]["samples"],
                                files["definition"], protocol)

    def test_class_id_float_refused(self):
        protocol, files = self._cache()
        arrays = dict(files["arrays"])
        arrays["class_id"] = arrays["class_id"].astype(np.float32)
        with pytest.raises(ValueError, match="integer"):
            verify_cache_arrays(arrays, files["manifest"]["samples"],
                                files["definition"], protocol)

    def test_non_finite_g_refused(self):
        protocol, files = self._cache()
        arrays = dict(files["arrays"])
        arrays["g_features"] = arrays["g_features"].copy()
        arrays["g_features"][0, 0] = np.nan
        with pytest.raises(ValueError, match="g_features contains NaN/Inf"):
            verify_cache_arrays(arrays, files["manifest"]["samples"],
                                files["definition"], protocol)

    def test_nan_in_valid_s_refused(self):
        protocol, files = self._cache()
        arrays = dict(files["arrays"])
        arrays["s_features"] = arrays["s_features"].copy()
        arrays["s_features"][0, 0] = np.nan
        with pytest.raises(ValueError, match="rows marked valid"):
            verify_cache_arrays(arrays, files["manifest"]["samples"],
                                files["definition"], protocol)

    def test_missing_s_rows_must_be_nan(self):
        protocol, files = self._cache()
        authority = authoritative_sample_index(protocol)
        arrays = dict(files["arrays"])
        manifest = json.loads(json.dumps(files["manifest"]))
        manifest["samples"][0]["has_valid_s"] = False
        verified = verify_cache_manifest(manifest, authority, protocol)
        arrays["s_valid"] = arrays["s_valid"].copy()
        arrays["s_valid"][0] = False
        # still a real measurement value -> violates the explicit-miss rule
        with pytest.raises(ValueError, match="explicit-miss"):
            verify_cache_arrays(arrays, verified, files["definition"],
                                protocol)

    def test_flag_column_mismatch_refused(self):
        protocol, files = self._cache()
        arrays = dict(files["arrays"])
        arrays["c_features"] = arrays["c_features"].copy()
        flag = CONTROL_FEATURE_KEYS.index("s_region_nonempty_flag")
        arrays["c_features"][0, flag] = 0.0
        with pytest.raises(ValueError, match="s_region_nonempty_flag"):
            verify_cache_arrays(arrays, files["manifest"]["samples"],
                                files["definition"], protocol)

    def test_g_dim_mismatch_refused(self):
        protocol, files = self._cache()
        arrays = dict(files["arrays"])
        arrays["g_features"] = arrays["g_features"][:, :2]
        with pytest.raises(ValueError, match="slot_G.dim"):
            verify_cache_arrays(arrays, files["manifest"]["samples"],
                                files["definition"], protocol)

    def test_preprocessing_version_mismatch_refused(self):
        protocol, files = self._cache()
        definition = json.loads(json.dumps(files["definition"]))
        definition["preprocessing_version"] = "other_v9"
        with pytest.raises(ValueError, match="preprocessing_version"):
            verify_cache_arrays(files["arrays"],
                                files["manifest"]["samples"], definition,
                                protocol)

    def test_slot_key_order_mismatch_refused(self):
        protocol, files = self._cache()
        definition = json.loads(json.dumps(files["definition"]))
        definition["slot_C"]["keys"] = list(reversed(definition["slot_C"]["keys"]))
        with pytest.raises(ValueError, match="slot_C keys"):
            verify_cache_arrays(files["arrays"],
                                files["manifest"]["samples"], definition,
                                protocol)

    def test_definition_class_order_mismatch_refused(self):
        protocol, files = self._cache()
        definition = json.loads(json.dumps(files["definition"]))
        definition["class_names"] = list(reversed(definition["class_names"]))
        with pytest.raises(ValueError, match="class order"):
            verify_cache_arrays(files["arrays"],
                                files["manifest"]["samples"], definition,
                                protocol)


class TestBudgetWatch:
    def test_within_budget_passes(self):
        clock = {"now": 0.0}
        budget = BudgetWatch(10.0, clock=lambda: clock["now"])
        clock["now"] = 500.0
        budget.check("frame boundary")
        assert budget.remaining_s() == pytest.approx(100.0)

    def test_exceeded_budget_raises(self):
        clock = {"now": 0.0}
        budget = BudgetWatch(10.0, clock=lambda: clock["now"])
        clock["now"] = 601.0
        with pytest.raises(BudgetExceeded, match="safe boundary"):
            budget.check("condition D1")

    def test_shared_start_not_per_step(self):
        clock = {"now": 0.0}
        budget = BudgetWatch(10.0, clock=lambda: clock["now"])
        clock["now"] = 599.0
        budget.check("step A")
        clock["now"] = 601.0
        with pytest.raises(BudgetExceeded):
            budget.check("step B")  # the deadline did not move

    def test_invalid_minutes_refused(self):
        with pytest.raises(ValueError, match="positive"):
            BudgetWatch(0, clock=lambda: 0.0)


# ---------------------------------------------------------------------------
# Cache upgrade entry (synthetic old evidence)


class TestUpgradeProbeCache:
    def _build_old_evidence(self, scratch_dir):
        """An old-run directory: protocol + features + artifact manifest."""
        import hashlib
        import sys as sys_module

        protocol = make_protocol()
        feature_dir = scratch_dir / "old" / "features"
        make_cache_files(feature_dir, protocol)
        (scratch_dir / "old" / "protocol").mkdir(parents=True, exist_ok=True)
        (scratch_dir / "old" / "protocol" / "protocol.json").write_text(
            json.dumps(protocol, sort_keys=True), encoding="utf-8")
        files = {}
        for name in BOUND_CACHE_FILES:
            path = feature_dir / name
            files[f"features/{name}"] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        (scratch_dir / "old" / "artifact_manifest.json").write_text(
            json.dumps({"files": files}, indent=2), encoding="utf-8")
        # a minimal experiment config bound to the synthetic chain
        config = {
            "experiment": "material_existing_data_v1",
            "data": {
                "root": "unused",
                "fingerprint": protocol["data"]["fingerprint"],
                "parent_v4_fingerprint": "0" * 64,
            },
        }
        config_path = scratch_dir / "config.yaml"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return protocol, config_path

    def _patch_admission(self, monkeypatch):
        import scripts.upgrade_probe_cache as upgrade_module

        monkeypatch.setattr(
            upgrade_module, "admit_mask_dataset",
            lambda root, expected_fingerprint, expected_parent_fingerprint: {
                "verified_files": 1,
                "audit_fingerprint": expected_fingerprint,
                "v4_parent_fingerprint": expected_parent_fingerprint,
            })

    def test_upgrade_copies_bytes_and_binds(self, scratch_dir, monkeypatch):
        import scripts.upgrade_probe_cache as upgrade_module

        self._patch_admission(monkeypatch)
        protocol, config_path = self._build_old_evidence(scratch_dir)
        output = scratch_dir / "upgraded"
        code = upgrade_module.main([
            "--old-run-dir", str(scratch_dir / "old"),
            "--config", str(config_path),
            "--output-dir", str(output),
        ])
        assert code == 0
        report = json.loads(
            (output / "upgrade_report.json").read_text(encoding="utf-8"))
        assert all(r["identical"]
                   for r in report["byte_equality"].values())
        # the corrected cache passes the repaired unified admission
        from core.material_probe import verify_cache_binding

        binding = verify_cache_binding(output / "corrected_features",
                                       protocol)
        assert binding["protocol_sha256"] == protocol["protocol_sha256"]
        assert (output / "admission.json").is_file()

    def test_tampered_old_feature_refused(self, scratch_dir, monkeypatch):
        import scripts.upgrade_probe_cache as upgrade_module

        self._patch_admission(monkeypatch)
        protocol, config_path = self._build_old_evidence(scratch_dir)
        cache = dict(np.load(scratch_dir / "old" / "features"
                             / "feature_cache.npz"))
        cache["g_features"][0, 0] += 5.0
        np.savez_compressed(
            scratch_dir / "old" / "features" / "feature_cache.npz", **cache)
        code = upgrade_module.main([
            "--old-run-dir", str(scratch_dir / "old"),
            "--config", str(config_path),
            "--output-dir", str(scratch_dir / "upgraded_bad"),
        ])
        assert code == 1
        assert not (scratch_dir / "upgraded_bad").exists()

    def test_uncovered_old_manifest_refused(self, scratch_dir, monkeypatch):
        import scripts.upgrade_probe_cache as upgrade_module

        self._patch_admission(monkeypatch)
        protocol, config_path = self._build_old_evidence(scratch_dir)
        manifest_path = scratch_dir / "old" / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["files"]["features/feature_cache.npz"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        code = upgrade_module.main([
            "--old-run-dir", str(scratch_dir / "old"),
            "--config", str(config_path),
            "--output-dir", str(scratch_dir / "upgraded_bad2"),
        ])
        assert code == 1

    def _scratch(self):
        import tempfile

        return tempfile.mkdtemp(
            dir="D:/Python/Project/my_data/.test_artifacts")
