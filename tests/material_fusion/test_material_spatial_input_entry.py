"""material_spatial_input_v1 entry tests on a synthetic stereo world:
preflight without any rebuild, the smoke build+verify chain, drift
refusals before any consumer, the full V5 admission, the real
PowerShell wrapper against FakePython and the zero-training probes on
real symbols. No model training anywhere; the fix cache is built by the
production recipe."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

from scripts import prepare_material_spatial_input as prep
from scripts import verify_material_spatial_input as verify
from scripts.prepare_material_residual_scale_features import (
    CACHE_KIND_FULL,
)
from .conftest import build_synthetic_v4, build_synthetic_v5
from .test_polar_region_quality_entry import (
    _make_cache_files_for_v5,
    _protocol_for_v5,
    _v5_rows,
)
from .test_material_residual_scale_features import (
    SYNTH_MATCHER,
    _build_stereo_source,
)

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")
_ROOT = prep._PROJECT_ROOT


def _dump_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True),
                    encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def spatial_world(module_scratch):
    """Synthetic stereo world + a production-built fix cache + the batch
    config. Dataset builds only; no model training anywhere."""
    from core.material_probe import (
        authoritative_sample_index,
        build_cache_binding,
    )
    from core.material_mask_dataset import mask_dataset_fingerprint
    from core.fusion_dataset import dataset_fingerprint
    from core.existing_data_protocol import protocol_digest

    workspace = module_scratch.resolve()
    source = _build_stereo_source(module_scratch / "source")
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    fingerprint = mask_dataset_fingerprint(v5)
    rows = _v5_rows(v5)
    dev_rows = [row for row in rows
                if row["split"] in ("train", "val")]
    dev_frames = sorted({row["source_frame"] for row in dev_rows})
    dev_groups = sorted({row["group_name"] for row in dev_rows})
    protocol = _protocol_for_v5(rows, fingerprint)
    # the V5 gate reads the V4 parent fingerprint from the protocol's
    # admission block (the real frozen protocol carries it too)
    protocol["admission"] = {
        "v4_parent_fingerprint": dataset_fingerprint(v4)}
    protocol["protocol_sha256"] = protocol_digest(protocol)
    protocol_dir = module_scratch / "protocol"
    protocol_dir.mkdir(parents=True)
    protocol_path = protocol_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True),
                             encoding="utf-8")
    authority = authoritative_sample_index(protocol)

    # the synthetic "old experiment config" carrying the SYNTH recipe
    old_config = {
        "fix_inputs": {**SYNTH_MATCHER,
                       "upsample_version": "valid_weighted_legacy_"
                                           "support_v1",
                       "band_margin": 20,
                       "full_image_threshold": 0.9, "crop_pad": 10},
        "local_features": {
            "region_rule": "local_partition_interior_band_v1",
            "feature_definition_version":
                "existing_data_probe_v1_local_interior_band_v1"},
        # verify_scale_cache_markers reads the dev-pool expectations
        # from the old config (the real frozen config carries them too)
        "support_expectations": {
            "frames": len(dev_frames), "instances": len(dev_rows),
            "groups": len(dev_groups), "empty_frames": 0},
    }
    old_config_path = _dump_yaml(module_scratch / "old_config.yaml",
                                 old_config)

    # the synthetic frozen fix cache (production build; no training)
    global_dir = module_scratch / "global_cache"
    global_dir.mkdir(parents=True)
    _make_cache_files_for_v5(v5, rows)(global_dir, protocol)
    (global_dir / "cache_binding.json").write_text(
        json.dumps(build_cache_binding(global_dir, protocol),
                   sort_keys=True), encoding="utf-8")

    cfg = {
        "experiment": "material_spatial_input_v1",
        "frozen_sources": {
            "old_experiment_config": {
                "path": str(old_config_path),
                "sha256": hashlib.sha256(
                    old_config_path.read_bytes()).hexdigest()},
            "fix_cache_dir": str(module_scratch / "unused_features"),
            "fix_cache_files": {},
            "build_source_manifest": {
                "path": str(module_scratch / "unused_manifest.json"),
                "sha256": "0" * 64},
            "protocol": {
                "path": str(protocol_path),
                "file_sha256": hashlib.sha256(
                    protocol_path.read_bytes()).hexdigest(),
                "internal_digest": protocol["protocol_sha256"]},
            "v5_data": {
                "root": str(v5),
                "fingerprint": fingerprint},
            "pair_manifest": {"path": str(source / "pair_manifest.csv")},
        },
        "dev_pool": {
            "splits": ["train", "val"], "frames": 0, "nonempty_frames": 0,
            "empty_frames": 0, "instances": 0, "groups": 0,
            "s_valid_true": 0, "s_valid_false": 0,
            "empty_frame_names": []},
        "verify": {"grid_atol": 1.0e-6, "grid_rtol": 0.0},
        "budgets": {"min_free_gib_before_build": 10},
    }

    # the production fix-cache build over the synthetic world
    from core.material_residual_scale import DeadlineWatch

    dev_rows = [row for row in rows if row["split"] in ("train", "val")]
    dev_frames = sorted({row["source_frame"] for row in dev_rows})
    groups = sorted({row["group_name"] for row in dev_rows})
    ctx = {"registry": prep.SourceRegistry({}),
           "watch": DeadlineWatch(budget_minutes=30),
           "stage": lambda *a, **k: None, "log": lambda message: None}
    # the fix cache build borrows the third-batch recipe helpers via the
    # established fixture path (production functions; no training)
    from scripts.prepare_material_residual_scale_features import (
        run_preflight as old_run_preflight,
        run_feature_build as old_run_feature_build,
        write_scale_cache as old_write_scale_cache,
    )

    old_cfg = {
        "experiment": "material_residual_scale_v1",
        "anchors": {
            "runtime_config": {
                "path": str(module_scratch / "placeholder.yaml"),
                "sha256": "0" * 64},
            "protocol_file": {"path": str(protocol_path),
                              "sha256": "0" * 64},
            "artifact_manifest": {"path": str(protocol_path),
                                  "sha256": "0" * 64},
            "upsample_impact_config": {"path": str(protocol_path),
                                       "sha256": "0" * 64},
            "batch1_source_manifest": {"path": str(protocol_path),
                                       "sha256": "0" * 64}},
        "protocol_source": {"path": str(protocol_path)},
        "features_source": {"path": str(global_dir)},
        "p0_baseline_source": {
            "fit_dir": str(module_scratch / "unused_fit"),
            "verification": str(module_scratch / "unused_ver.json")},
        "data": {"root": str(v5), "source_root": str(source),
                 "fingerprint": fingerprint,
                 "parent_v4_fingerprint": dataset_fingerprint(v4),
                 "class_names": list(CLASS_NAMES),
                 "dev_pool_splits": ["train", "val"],
                 "excluded_splits": ["test"]},
        "fix_inputs": old_config["fix_inputs"],
        "local_features": old_config["local_features"],
        "support_expectations": {
            "frames": len(dev_frames), "instances": len(dev_rows),
            "groups": len(groups), "empty_frames": 0},
        "scale_rule": {"source": "p0_logits_inner_train",
                       "statistic": "median", "dtype": "float64",
                       "min_scale": 1.0},
        "arms": {"fixed": {"magnitude_factor": 0.5},
                 "scaled": {"magnitude_factor": 0.5}},
        "seeds": [2026, 2027, 2028],
        "shuffle": {"content_seeds": [2026, 2027, 2028, 2029, 2030]},
        "training": {"device": "cpu", "full_batch": True, "epochs": 1,
                     "optimizer": "adamw", "lr": 0.01,
                     "weight_decay": 0.001, "aux_loss_weight": 0.01},
        "head": {"hidden": 8},
    }
    p0_stub = module_scratch / "unused_fit"
    p0_stub.mkdir(parents=True, exist_ok=True)
    (p0_stub / "summary.json").write_text("{}", encoding="utf-8")
    build_ctx = {"registry": prep.SourceRegistry({}),
                 "watch": DeadlineWatch(budget_minutes=30),
                 "stage": lambda *a, **k: None,
                 "log": lambda message: None}
    pre = old_run_preflight(build_ctx, old_cfg)
    build = old_run_feature_build(build_ctx, old_cfg, pre["census"],
                                  pre["global_info"], frame_limit=None)
    fix_dir = module_scratch / "fix_cache" / "features"
    old_write_scale_cache(
        fix_dir, old_cfg, pre["protocol"], build, kind=CACHE_KIND_FULL,
        complete=True, interrupted=False,
        data_fingerprint=old_cfg["data"]["fingerprint"],
        protocol_sha256=pre["protocol"]["protocol_sha256"],
        empty_frames=pre["census"]["empty_frames"])
    cache_arrays = dict(np.load(fix_dir / "feature_cache.npz"))
    manifest = json.loads((fix_dir / "feature_manifest.json").read_text(
        encoding="utf-8"))

    # the synthetic build source manifest (historical expectations): the
    # files the fix build consumed, with their actual digests. The pair
    # manifest is deliberately LEFT OUT so the spatial run's own
    # registration of it is a genuine new_snapshot.
    import csv as _csv

    consumed = [str(protocol_path), str(v5 / "dataset_manifest.csv"),
                str(v5 / "dataset_audit.json"),
                str(v5 / "dataset_summary.json"),
                str(fix_dir / "cache_binding.json"),
                str(fix_dir / "feature_cache.npz"),
                str(fix_dir / "feature_manifest.json"),
                str(fix_dir / "feature_definition.json")]
    consumed.append(str(global_dir / "feature_cache.npz"))
    for sample in manifest["samples"]:
        consumed.append(str(v5 / sample["npz_path"]))
    pair_rows = {}
    with (source / "pair_manifest.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        for row in _csv.DictReader(handle):
            pair_rows[str(row["output_stem"]).strip()] = row
    v5_csv_rows = {}
    with (v5 / "dataset_manifest.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        for row in _csv.DictReader(handle):
            v5_csv_rows[row["sample_name"]] = row
    # the V5 gate registers every npz and every annotation source
    # (test included, integrity reads only); the manifest/summary are
    # registered too. Images (left/right) are consumed per dev frame.
    for row in rows:
        consumed.append(str(v5 / row["npz_path"]))
        consumed.append(str(_ROOT / row["label_path"]))
    for stem in dev_frames:
        row = v5_csv_rows[[n for n, r in v5_csv_rows.items()
                           if r["source_frame"] == stem][0]]
        consumed.append(str(_ROOT / row["left_path"]))
        consumed.append(str(_ROOT / pair_rows[stem]["right_path"]))
    items = []
    for path in consumed:
        p = Path(path)
        if not p.is_file():
            continue
        items.append({
            "path": p.resolve().relative_to(_ROOT).as_posix(),
            "role": "synthetic_build_consumed",
            "sha256_at_registration": hashlib.sha256(
                p.read_bytes()).hexdigest(),
            "sha256_at_end": hashlib.sha256(
                p.read_bytes()).hexdigest(),
            "unchanged": True, "status": "ok"})
    manifest_path = module_scratch / "build_source_manifest.json"
    manifest_path.write_text(json.dumps({"items": items}, indent=1),
                             encoding="utf-8")

    # the batch config now pins the REAL synthetic artifacts
    cfg["frozen_sources"]["fix_cache_dir"] = str(fix_dir)
    for name in ("cache_binding.json", "feature_cache.npz",
                 "feature_manifest.json", "feature_definition.json"):
        cfg["frozen_sources"]["fix_cache_files"][name] = {
            "sha256": hashlib.sha256(
                (fix_dir / name).read_bytes()).hexdigest()}
    cfg["frozen_sources"]["build_source_manifest"] = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    old_config["support_expectations"] = {
        "frames": len(dev_frames), "instances": len(dev_rows),
        "groups": len(groups), "empty_frames": 0}
    # the dev-pool expectations from the synthetic world
    s_valid = cache_arrays["s_valid"]
    cfg["dev_pool"] = {
        "splits": ["train", "val"], "frames": len(dev_frames),
        "nonempty_frames": len(dev_frames), "empty_frames": 0,
        "instances": len(dev_rows), "groups": len(groups),
        "s_valid_true": int(np.asarray(s_valid).sum()),
        "s_valid_false": int(len(s_valid) - np.asarray(s_valid).sum()),
        "empty_frame_names": []}
    cfg_path = _dump_yaml(module_scratch / "configs" /
                          "material_spatial_input_v1.yaml", cfg)
    return {
        "workspace": workspace, "source": source, "v5": v5,
        "protocol": protocol, "authority": authority, "cfg": cfg,
        "cfg_path": cfg_path, "fix_dir": fix_dir,
        "cache_arrays": cache_arrays, "manifest": manifest["samples"],
        "dev_frames": dev_frames, "verification": None,
    }


@pytest.fixture(scope="module")
def spatial_patches(spatial_world, module_scratch):
    """Point the entry's pinned-config constants at the synthetic world."""
    config_rel = spatial_world["cfg_path"].relative_to(_ROOT).as_posix()
    config_sha = hashlib.sha256(
        spatial_world["cfg_path"].read_bytes()).hexdigest()
    patches = [
        mock.patch.object(prep, "PINNED_CONFIG_REL", config_rel),
        mock.patch.object(verify, "PINNED_CONFIG_REL", config_rel),
        mock.patch.object(prep, "PINNED_CONFIG_SHA256", config_sha),
        mock.patch.object(verify, "PINNED_CONFIG_SHA256", config_sha),
        mock.patch.object(prep, "OUTPUT_ROOT",
                          (module_scratch / "spatial_root").relative_to(
                              _ROOT).as_posix()),
        mock.patch.object(verify, "OUTPUT_ROOT",
                          (module_scratch / "spatial_root").relative_to(
                              _ROOT).as_posix()),
    ]
    for context in patches:
        context.start()
    try:
        yield {"config_rel": config_rel, "config_path":
               spatial_world["cfg_path"]}
    finally:
        for context in patches:
            context.stop()


def _run_prepare(spatial_world, output_dir, mode, budget_deadline,
                 max_frames=None, config_path=None):
    argv = ["--config", str(config_path or spatial_world["cfg_path"]),
            "--mode", mode, "--output-dir", str(output_dir),
            "--deadline-utc", budget_deadline]
    if max_frames is not None:
        argv += ["--max-frames", str(max_frames)]
    return prep.main(argv)


def _local_smoke_cache(spatial_world, module_scratch, name):
    """A fresh smoke cache built inside the entry module (the verifier
    module's fixture is not importable here)."""
    out = module_scratch / "spatial_root" / name
    status = _run_prepare(spatial_world, out, "smoke",
                          "2030-01-01T00:00:00+00:00", max_frames=4)
    assert status == prep.EXIT_OK
    return out


class TestPreflight:
    def test_preflight_completes_without_any_rebuild(
            self, spatial_world, spatial_patches, module_scratch):
        """Every actual rebuild entry the prepare module binds is bombed
        on the module's own namespace (prep.StereoMatcher, the module's
        compute_polar_features reference, cv2.remap, the native/grid/
        permutation boundaries) — preflight must complete with zero
        hits. The full V5 admission (integrity reads only) DOES run."""
        calls = {"matcher": 0, "polar": 0, "remap": 0, "native": 0,
                 "grid": 0, "perm": 0}

        def counting_bomb(key):
            def _bomb(*a, **k):
                calls[key] += 1
                raise AssertionError(f"preflight must not run {key}")
            return _bomb

        with mock.patch.object(prep, "StereoMatcher",
                               counting_bomb("matcher")), \
                mock.patch.object(prep, "compute_polar_features",
                                  counting_bomb("polar")), \
                mock.patch.object(prep.cv2, "remap",
                                  counting_bomb("remap")), \
                mock.patch.object(prep, "_native_rebuild_for_instance",
                                  counting_bomb("native")), \
                mock.patch.object(prep, "_build_grid_planes",
                                  counting_bomb("grid")), \
                mock.patch.object(prep, "plan_q_permutation",
                                  counting_bomb("perm")):
            code = _run_prepare(spatial_world,
                                module_scratch / "spatial_root" /
                                "preflight", "preflight",
                                "2030-01-01T00:00:00+00:00")
        assert code == prep.EXIT_OK
        assert calls == {"matcher": 0, "polar": 0, "remap": 0,
                         "native": 0, "grid": 0, "perm": 0}
        out = module_scratch / "spatial_root" / "preflight"
        status = json.loads((out / "builder" / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["overall_status"] == "COMPLETED"
        assert status["input_admitted"] is False
        assert (out / "builder" / "preflight_plan.json").is_file()
        plan = json.loads((out / "builder" / "preflight_plan.json")
                          .read_text(encoding="utf-8"))
        # the full V5 fingerprint admission really executed in preflight
        assert plan["v5_gate"]["audit_fingerprint"] == \
            spatial_world["cfg"]["frozen_sources"]["v5_data"][
                "fingerprint"]
        selection = json.loads((out / "builder" / "smoke_selection.json")
                               .read_text(encoding="utf-8"))
        assert selection["mode"] == "preflight"


class TestBuildDiskAndSmokeBounds:
    @pytest.mark.parametrize("extra_bytes", [-1, 0, 1])
    def test_full_build_checks_actual_target_free_space(
            self, spatial_world, spatial_patches, module_scratch,
            extra_bytes):
        from types import SimpleNamespace

        required = 10 * 1024 ** 3
        out = module_scratch / "spatial_root" / f"disk_{extra_bytes}"
        with mock.patch("shutil.disk_usage", return_value=SimpleNamespace(
                free=required + extra_bytes)) as query, mock.patch.object(
                    prep, "_build_frames", wraps=prep._build_frames) as build:
            code = _run_prepare(spatial_world, out, "build",
                                "2030-01-01T00:00:00+00:00")
        status = json.loads((out / "builder" / "run_status.json")
                            .read_text(encoding="utf-8"))
        query.assert_called_once_with(out.resolve())
        record = status["stages"]["disk_space"]
        assert record["path"] == str(out.resolve())
        assert record["free_bytes"] == required + extra_bytes
        assert record["required_bytes"] == required
        if extra_bytes < 0:
            assert code == prep.EXIT_REFUSED
            assert record["status"] == "FAILED"
            assert "free disk space" in status["failure"]
            build.assert_not_called()
            assert not (out / "cache").exists()
            assert status["input_admitted"] is False
        else:
            assert code == prep.EXIT_OK
            assert record["status"] == "COMPLETED"
            assert build.call_count == 1

    def test_disk_query_error_refuses_before_pixels(
            self, spatial_world, spatial_patches, module_scratch):
        out = module_scratch / "spatial_root" / "disk_query_error"
        with mock.patch("shutil.disk_usage", side_effect=OSError(
                "target volume unavailable")), mock.patch.object(
                    prep, "_build_frames") as build:
            code = _run_prepare(spatial_world, out, "build",
                                "2030-01-01T00:00:00+00:00")
        assert code == prep.EXIT_REFUSED
        build.assert_not_called()
        status = json.loads((out / "builder" / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert "target volume unavailable" in status["failure"]
        assert status["stages"]["disk_space"]["status"] == "FAILED"
        assert not (out / "cache").exists()

    @pytest.mark.parametrize("maximum", [1, 2, 3])
    def test_small_smoke_cap_refused_before_admission(
            self, spatial_world, spatial_patches, module_scratch, maximum):
        cfg = copy.deepcopy(spatial_world["cfg"])
        cfg["dev_pool"]["empty_frame_names"] = [f"empty_{i}" for i in range(4)]
        cfg["dev_pool"]["empty_frames"] = 4
        path = _dump_yaml(module_scratch / "configs" / f"cap_{maximum}.yaml", cfg)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out = module_scratch / "spatial_root" / f"cap_{maximum}"
        with mock.patch.object(prep, "PINNED_CONFIG_REL",
                               path.relative_to(_ROOT).as_posix()), \
                mock.patch.object(prep, "PINNED_CONFIG_SHA256", digest), \
                mock.patch.object(prep, "run_admission") as admission:
            code = _run_prepare(spatial_world, out, "smoke",
                                "2030-01-01T00:00:00+00:00",
                                max_frames=maximum, config_path=path)
        assert code == prep.EXIT_REFUSED
        admission.assert_not_called()
        assert not out.exists()

    @pytest.mark.parametrize("maximum", [3, 4, 12])
    def test_selector_preserves_required_empties_within_cap(
            self, spatial_world, maximum):
        census = {"empty_frames": [f"empty_{i}" for i in range(4)]}
        args = ({"max_frames": maximum}, spatial_world["cfg"], census,
                spatial_world["cache_arrays"], spatial_world["manifest"])
        if maximum < 4:
            with pytest.raises(prep.ScaleRefused, match="empty frames"):
                prep._smoke_selection(*args)
        else:
            plan = prep._smoke_selection(*args)
            assert plan["empty_frames"] == census["empty_frames"]
            assert len(plan["empty_frames"]) + len(plan["picked"]) <= maximum


class TestZeroTrainingProbes:
    def test_probes_fire_on_real_symbols(self):
        """Every patched symbol is REAL and the bomb provably fires on
        it before the entry run relies on the zero counts — without
        calling the original training/forward implementations and
        without create=True forged symbols."""
        import core.material_probe as probe_mod
        import core.material_conditional_residual as cond_mod

        def scaler_bomb(*a, **k):
            raise AssertionError("Standardizer.fit fired")

        def head_bomb(*a, **k):
            raise AssertionError("build_conditional_head fired")

        def logits_bomb(*a, **k):
            raise AssertionError("conditional_logits fired")

        def module_call_bomb(self, *a, **k):
            raise AssertionError("torch.nn.Module.__call__ fired")

        def optimizer_step_bomb(self, *a, **k):
            raise AssertionError("Optimizer.step fired")

        class OptimBomb:
            def __init__(self, *a, **k):
                raise AssertionError("an optimizer was constructed")

        with mock.patch.object(probe_mod.Standardizer, "fit",
                               scaler_bomb):
            with pytest.raises(AssertionError):
                probe_mod.Standardizer.fit(None, None)
        with mock.patch.object(cond_mod, "build_conditional_head",
                               head_bomb):
            with pytest.raises(AssertionError):
                cond_mod.build_conditional_head(seed=1, input_dim=4,
                                                hidden=2)
        with mock.patch.object(cond_mod, "conditional_logits",
                               logits_bomb):
            with pytest.raises(AssertionError):
                cond_mod.conditional_logits(None, None, None, None, 0.5)
        with mock.patch.object(torch.optim, "AdamW", OptimBomb):
            with pytest.raises(AssertionError):
                torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
        # torch.nn.Module.__call__: the bomb fires on a real scratch
        # module call (the original forward never runs)
        class _Tiny(torch.nn.Module):
            def forward(self, value):
                return value + 1.0

        with mock.patch.object(torch.nn.Module, "__call__",
                               module_call_bomb):
            with pytest.raises(AssertionError):
                _Tiny()(torch.zeros(1))
        # Optimizer.step (AdamW inherits Adam.step): the bomb fires
        # instead of the real step
        with mock.patch.object(torch.optim.Adam, "step",
                               optimizer_step_bomb):
            optimizer = torch.optim.AdamW(
                [torch.nn.Parameter(torch.zeros(1))], lr=0.01)
            with pytest.raises(AssertionError):
                optimizer.step()

    def test_smoke_chain_has_zero_forbidden_calls(
            self, spatial_world, spatial_patches, module_scratch):
        import core.material_probe as probe_mod
        import core.material_conditional_residual as cond_mod

        calls = {"scaler_fit": 0, "head_build": 0, "cond_logits": 0,
                 "optimizer": 0, "module_call": 0, "adamw_step": 0}

        def counter(key, label):
            def _bomb(*a, **k):
                calls[key] += 1
                raise AssertionError(f"{label} must never run")
            return _bomb

        out = module_scratch / "spatial_root" / "smoke"
        with mock.patch.object(probe_mod.Standardizer, "fit",
                               counter("scaler_fit",
                                       "Standardizer.fit")), \
                mock.patch.object(cond_mod, "build_conditional_head",
                                  counter("head_build",
                                          "build_conditional_head")), \
                mock.patch.object(cond_mod, "conditional_logits",
                                  counter("cond_logits",
                                          "conditional_logits")), \
                mock.patch.object(torch.optim, "AdamW",
                                  counter("optimizer", "AdamW")), \
                mock.patch.object(torch.nn.Module, "__call__",
                                  counter("module_call",
                                          "nn.Module.__call__")), \
                mock.patch.object(torch.optim.Adam, "step",
                                  counter("adamw_step",
                                          "Optimizer.step")):
            code = _run_prepare(spatial_world, out, "smoke",
                                "2030-01-01T00:00:00+00:00",
                                max_frames=6)
        assert code == prep.EXIT_OK
        assert calls == {"scaler_fit": 0, "head_build": 0,
                         "cond_logits": 0, "optimizer": 0,
                         "module_call": 0, "adamw_step": 0}
        status = json.loads((out / "builder" / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["overall_status"] == "BUILT_UNVERIFIED"
        assert status["input_admitted"] is False
        assert status["cache_schema"] == \
            "material_spatial_input_cache_v1"
        assert status["implementation_revision"] == "review_fix_v1"
        assert status["built_frames"] >= 1
        assert status["built_instances"] == 2 * status["built_frames"]
        definition = json.loads((out / "cache" / "definition.json")
                                .read_text(encoding="utf-8"))
        assert definition["mode"] == "smoke"
        assert definition["kind"] == "material_spatial_input_smoke_v1"
        assert definition["complete"] is False
        # the binding covers the metadata too
        binding = json.loads((out / "cache" / "binding.json")
                             .read_text(encoding="utf-8"))
        bound_paths = {record["path"] for record in binding["artifacts"]}
        assert {"cache/definition.json", "cache/samples.jsonl",
                "cache/frames.jsonl"} <= bound_paths
        ledger = [json.loads(line) for line in
                  (out / "cache" / "samples.jsonl").read_text(
                      encoding="utf-8").splitlines() if line.strip()]
        for row in ledger:
            assert row["fold_roles"] == \
                spatial_world["authority"][row["sample_name"]]["roles"]
            assert row["cache_row"] == next(
                index for index, sample
                in enumerate(spatial_world["manifest"])
                if sample["sample_name"] == row["sample_name"])


class TestVerifyChainZeroTraining:
    def test_verify_chain_has_zero_forbidden_calls(
            self, spatial_world, spatial_patches, module_scratch):
        """The verifier chain is probed like the builder chain: the
        real training/forward entries are bombed and the full synthetic
        smoke verify runs with zero hits."""
        import shutil
        from .test_material_spatial_input_verifier import _run_verify

        import core.material_probe as probe_mod
        import core.material_conditional_residual as cond_mod

        smoke_dir = _local_smoke_cache(spatial_world, module_scratch,
                                       "verify_probe_src")
        cache_dir = module_scratch / "copies" / "verify_probe_src"
        shutil.copytree(smoke_dir, cache_dir)

        calls = {"scaler_fit": 0, "head_build": 0, "cond_logits": 0,
                 "optimizer": 0, "module_call": 0, "adamw_step": 0}

        def counter(key, label):
            def _bomb(*a, **k):
                calls[key] += 1
                raise AssertionError(f"{label} must never run")
            return _bomb

        with mock.patch.object(probe_mod.Standardizer, "fit",
                               counter("scaler_fit",
                                       "Standardizer.fit")),                 mock.patch.object(cond_mod, "build_conditional_head",
                                  counter("head_build",
                                          "build_conditional_head")),                 mock.patch.object(cond_mod, "conditional_logits",
                                  counter("cond_logits",
                                          "conditional_logits")),                 mock.patch.object(torch.optim, "AdamW",
                                  counter("optimizer", "AdamW")),                 mock.patch.object(torch.nn.Module, "__call__",
                                  counter("module_call",
                                          "nn.Module.__call__")),                 mock.patch.object(torch.optim.Adam, "step",
                                  counter("adamw_step",
                                          "Optimizer.step")):
            code = _run_verify(spatial_world, cache_dir,
                               cache_dir.parent.parent)
        assert code == verify.EXIT_OK
        assert calls == {"scaler_fit": 0, "head_build": 0,
                         "cond_logits": 0, "optimizer": 0,
                         "module_call": 0, "adamw_step": 0}


class TestDriftAndDeadline:
    def test_config_drift_never_reaches_the_parser(
            self, spatial_world, spatial_patches, module_scratch):
        """A VALID-content config drift (still parseable yaml) is
        refused at registration: the yaml parser is never called, for
        the prepare and the verifier alike."""
        cfg_path = spatial_world["cfg_path"]
        # a small cache so the verifier call reaches the config gate;
        # it is built BEFORE the drift
        cache_dir = module_scratch / "spatial_root" / "drift_config_cache"
        assert _run_prepare(spatial_world, cache_dir, "smoke",
                            "2030-01-01T00:00:00+00:00",
                            max_frames=4) == prep.EXIT_OK
        original = cfg_path.read_bytes()
        drifted = yaml.safe_load(original)
        drifted["dev_pool"]["instances"] = \
            int(drifted["dev_pool"]["instances"]) + 1
        cfg_path.write_text(yaml.safe_dump(drifted, sort_keys=True),
                            encoding="utf-8")
        out = module_scratch / "spatial_root" / "drift_config"
        try:
            parser_calls = {"n": 0}

            def counting_load(*a, **k):
                parser_calls["n"] += 1
                return yaml.safe_load(*a, **k)

            with mock.patch.object(prep.yaml, "safe_load",
                                   counting_load):
                code = _run_prepare(spatial_world, out, "build",
                                    "2030-01-01T00:00:00+00:00")
            assert code == prep.EXIT_REFUSED
            assert parser_calls["n"] == 0, \
                "a drifted config reached the parser"
            verify_calls = {"n": 0}

            def counting_load_verify(*a, **k):
                verify_calls["n"] += 1
                return yaml.safe_load(*a, **k)

            with mock.patch.object(verify.yaml, "safe_load",
                                   counting_load_verify):
                code = verify.main([
                    "--config", str(cfg_path), "--mode", "smoke",
                    "--cache-dir", str(cache_dir),
                    "--output-dir", str(cache_dir / "verification"),
                    "--deadline-utc", "2030-01-01T00:00:00+00:00",
                    "--expected-binding-sha256", "0" * 64])
            assert code == verify.EXIT_REFUSED
            assert verify_calls["n"] == 0
        finally:
            cfg_path.write_bytes(original)

    def test_historical_json_drift_refused_before_consumer(
            self, spatial_world, spatial_patches, module_scratch):
        """A drifted (still valid) protocol JSON is refused at
        registration — the protocol consumer never runs."""
        protocol_path = None
        for item in (module_scratch / "protocol").iterdir():
            if item.suffix == ".json":
                protocol_path = item
        original = protocol_path.read_bytes()
        protocol = json.loads(original)
        first_group = sorted(protocol["group_statistics"])[0]
        protocol["group_statistics"][first_group]["gt_instances"] += 1
        protocol_path.write_text(json.dumps(protocol, sort_keys=True),
                                 encoding="utf-8")
        out = module_scratch / "spatial_root" / "drift_protocol"
        try:
            consumer_calls = {"n": 0}

            def consumer_bomb(*a, **k):
                consumer_calls["n"] += 1
                raise AssertionError("the drifted protocol was parsed")

            with mock.patch.object(prep, "authoritative_sample_index",
                                   consumer_bomb):
                code = _run_prepare(spatial_world, out, "build",
                                    "2030-01-01T00:00:00+00:00")
            assert code == prep.EXIT_REFUSED
            assert consumer_calls["n"] == 0
            status = json.loads((out / "builder" / "run_status.json")
                                .read_text(encoding="utf-8"))
            assert "sha256" in status["failure"]
        finally:
            protocol_path.write_bytes(original)

    def test_historical_csv_drift_refused_before_consumer(
            self, spatial_world, spatial_patches, module_scratch):
        """A drifted (still parseable) V5 manifest CSV is refused at
        registration — the npz consumer never runs."""
        manifest_path = spatial_world["v5"] / "dataset_manifest.csv"
        original = manifest_path.read_bytes()
        text = original.decode("utf-8-sig")
        lines = text.splitlines()
        drifted = lines + [lines[-1].replace(
            lines[-1].split(",")[0],
            lines[-1].split(",")[0] + "_drifted")]
        manifest_path.write_text("\r\n".join(drifted) + "\r\n",
                                 encoding="utf-8-sig")
        out = module_scratch / "spatial_root" / "drift_csv"
        try:
            consumer_calls = {"n": 0}

            def consumer_bomb(*a, **k):
                consumer_calls["n"] += 1
                raise AssertionError("the drifted CSV was consumed")

            with mock.patch.object(prep, "load_mask_sample",
                                   consumer_bomb):
                code = _run_prepare(spatial_world, out, "build",
                                    "2030-01-01T00:00:00+00:00")
            assert code == prep.EXIT_REFUSED
            assert consumer_calls["n"] == 0
        finally:
            manifest_path.write_bytes(original)

    def test_verifier_refuses_drifted_sources_before_its_rebuild(
            self, spatial_world, spatial_patches, module_scratch):
        """The verifier re-registers the historical sources under the
        same expectations: a drifted V5 manifest CSV (covered by the
        build manifest) is refused before its own matcher ever runs."""
        import shutil
        from .test_material_spatial_input_verifier import _run_verify

        smoke_dir = _local_smoke_cache(spatial_world, module_scratch,
                                       "verifier_drift_src")
        cache_dir = module_scratch / "copies" / "verifier_drift_src"
        shutil.copytree(smoke_dir, cache_dir)
        manifest_path = spatial_world["v5"] / "dataset_manifest.csv"
        original = manifest_path.read_bytes()
        text = original.decode("utf-8-sig")
        manifest_path.write_text(text + text.splitlines()[0] + "\r\n",
                                 encoding="utf-8-sig")
        try:
            consumer_calls = {"n": 0}

            def consumer_bomb(*a, **k):
                consumer_calls["n"] += 1
                raise AssertionError("the verifier consumed drifted "
                                     "sources")

            with mock.patch.object(verify, "StereoMatcher",
                                   consumer_bomb):
                code = _run_verify(spatial_world, cache_dir,
                                   cache_dir.parent.parent)
            assert code == verify.EXIT_REFUSED
            assert consumer_calls["n"] == 0
        finally:
            manifest_path.write_bytes(original)

    def test_drifted_test_split_asset_fails_the_v5_gate(
            self, spatial_world, spatial_patches, module_scratch):
        """A test-split npz with drifted (still loadable) content fails
        the FULL V5 admission; the test split contributes no cache row
        (the census counts train/val only)."""
        import csv as _csv

        data_root = spatial_world["v5"]
        with (data_root / "dataset_manifest.csv").open(
                encoding="utf-8-sig", newline="") as handle:
            rows = list(_csv.DictReader(handle))
        test_rows = [row for row in rows if row["split"] == "test"]
        assert test_rows, "the synthetic world has no test split"
        target = data_root / test_rows[0]["npz_path"]
        original = target.read_bytes()
        with np.load(target) as payload:
            arrays = {key: payload[key] for key in payload.files}
        arrays["gray"] = arrays["gray"] + np.uint8(1)
        np.savez_compressed(target, **arrays)
        out = module_scratch / "spatial_root" / "drift_test"
        try:
            code = _run_prepare(spatial_world, out, "build",
                                "2030-01-01T00:00:00+00:00")
            assert code == prep.EXIT_REFUSED
            status = json.loads((out / "builder" / "run_status.json")
                                .read_text(encoding="utf-8"))
            assert "digest" in status["failure"] or \
                "sha256" in status["failure"] or \
                "changed" in status["failure"] or \
                "fingerprint" in status["failure"]
            assert not (out / "cache" / "samples.jsonl").exists(), \
                "a run that failed the V5 gate must not write cache rows"
        finally:
            target.write_bytes(original)

    def test_verifier_does_not_overwrite_builder_evidence(
            self, spatial_world, spatial_patches, module_scratch):
        import shutil
        from .test_material_spatial_input_verifier import _run_verify

        smoke_dir = _local_smoke_cache(spatial_world, module_scratch,
                                       "not_overwrite_src")
        # the copy keeps the run-root NAME: the verifier refuses
        # cross-run evidence stitching on a renamed run
        cache_dir = module_scratch / "copies" / "not_overwrite_src"
        shutil.copytree(smoke_dir, cache_dir)
        builder_manifest = cache_dir / "builder" / "source_manifest.json"
        before = builder_manifest.read_bytes()
        code = _run_verify(spatial_world, cache_dir,
                           cache_dir.parent.parent)
        assert code == verify.EXIT_OK
        assert builder_manifest.read_bytes() == before, \
            "the verifier overwrote the builder source manifest"
        out = cache_dir / "verification"
        assert (out / "source_manifest.json").is_file()
        assert (out / "source_manifest.json").read_bytes() != before

    def test_new_snapshot_only_for_uncovered_sources(
            self, spatial_world, spatial_patches, module_scratch):
        """A normal run records historical binding_source kinds for
        covered files and marks ONLY the truly uncovered pair manifest
        as its own new_snapshot."""
        out = module_scratch / "spatial_root" / "snapshot"
        code = _run_prepare(spatial_world, out, "smoke",
                            "2030-01-01T00:00:00+00:00", max_frames=4)
        assert code == prep.EXIT_OK
        manifest = json.loads((out / "builder" / "source_manifest.json")
                              .read_text(encoding="utf-8"))
        kinds = {}
        for item in manifest["items"]:
            kinds.setdefault(item["registration"], []).append(item["path"])
        assert kinds.get("frozen_binding"), \
            "covered sources must register as frozen_binding"
        snapshots = kinds.get("batch2_snapshot", [])
        pair_rel = spatial_world["source"].joinpath(
            "pair_manifest.csv").resolve().relative_to(
            _ROOT).as_posix()
        assert [p.lower() for p in snapshots] == [pair_rel.lower()], \
            f"only the pair manifest may be a new_snapshot: {snapshots}"

    def test_conflicting_path_alias_expectations_are_refused(
            self, spatial_world, spatial_patches, module_scratch):
        from core.material_spatial_input_admission import (
            AdmissionRefused,
            ExpectationBook,
        )

        book = ExpectationBook(_ROOT)
        book.add(spatial_world["cfg_path"], "a", "1" * 64)
        # the same file under a different-case alias string is the same
        # canonical path: a different expectation is a conflict
        alias = Path(str(spatial_world["cfg_path"]).upper())
        with pytest.raises(AdmissionRefused):
            book.add(alias, "b", "2" * 64)

    def test_ancestor_reparse_point_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        from core.material_spatial_input_admission import (
            AdmissionRefused,
            check_consumed_path,
        )

        real = module_scratch / "reparse_real"
        real.mkdir()
        (real / "leaf.txt").write_text("x", encoding="utf-8")
        link = module_scratch / "reparse_link"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link),
                        str(real)], check=True,
                       capture_output=True)
        try:
            with pytest.raises(AdmissionRefused):
                check_consumed_path(link / "leaf.txt", _ROOT)
        finally:
            # remove the link itself first, never recurse through it
            os.rmdir(link)

    def test_late_deadline_publishes_partial(
            self, spatial_world, spatial_patches, module_scratch):
        """A fake watch lets every stage complete and expires exactly at
        the real success-publish boundary: PARTIAL, non-zero, evidence
        kept — not a tiny-budget instant timeout."""
        from core.material_residual_scale import DeadlineExceeded

        class BoundaryWatch:
            def __init__(self, *, deadline_utc=None, budget_minutes=None):
                pass

            def check(self, label):
                if label == "before success status publish":
                    raise DeadlineExceeded(
                        "expired at the success-publish boundary")

            def remaining_s(self):
                return 1.0

        out = module_scratch / "spatial_root" / "deadline"
        with mock.patch.object(prep, "DeadlineWatch", BoundaryWatch):
            code = _run_prepare(spatial_world, out, "smoke",
                                "2030-01-01T00:00:00+00:00",
                                max_frames=4)
        assert code == prep.EXIT_PARTIAL
        status = json.loads((out / "builder" / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["input_admitted"] is False
        assert status["interrupted_by_deadline"] is True
        assert status["unfinished"]
        completed = {name for name, entry in status["stages"].items()
                     if entry["status"] == "COMPLETED"}
        assert {"admission", "admission_done", "build",
                "protection"} <= completed
        assert (out / "cache" / "definition.json").is_file()

    def test_unknown_cache_version_is_refused(
            self, spatial_world, spatial_patches, module_scratch):
        definition_path = spatial_world["fix_dir"] / \
            "feature_definition.json"
        original = definition_path.read_bytes()
        out = module_scratch / "spatial_root" / "badversion"
        try:
            definition = json.loads(original)
            definition["feature_definition_version"] = "unknown_v9"
            definition_path.write_text(json.dumps(definition),
                                       encoding="utf-8")
            # the batch config pins the tampered bytes so the cache
            # VERSION check (not the hash gate) is what refuses
            new_digest = hashlib.sha256(
                definition_path.read_bytes()).hexdigest()
            cfg = copy.deepcopy(spatial_world["cfg"])
            cfg["frozen_sources"]["fix_cache_files"][
                "feature_definition.json"] = {"sha256": new_digest}
            # the fix cache's own binding covers the same file: sync
            # its record, then cascade its own config pin and build
            # manifest item, so the cache VERSION check (not a binding
            # conflict) is what refuses
            fix_binding_path = spatial_world["fix_dir"] / \
                "cache_binding.json"
            fix_binding = json.loads(fix_binding_path.read_text(
                encoding="utf-8"))
            if isinstance(fix_binding.get("files"), dict) and \
                    "feature_definition.json" in fix_binding["files"]:
                fix_binding["files"]["feature_definition.json"][
                    "sha256"] = new_digest
                fix_binding_path.write_text(
                    json.dumps(fix_binding, sort_keys=True),
                    encoding="utf-8")
                cfg["frozen_sources"]["fix_cache_files"][
                    "cache_binding.json"] = {
                        "sha256": hashlib.sha256(
                            fix_binding_path.read_bytes()).hexdigest()}
            # the historical build manifest covers the same file: sync
            # its item digests too so the cache VERSION check (not a
            # binding conflict) is what refuses
            build_manifest_path = Path(cfg["frozen_sources"][
                "build_source_manifest"]["path"])
            build_manifest = json.loads(build_manifest_path.read_text(
                encoding="utf-8"))
            target_rel = definition_path.resolve().relative_to(
                _ROOT).as_posix()
            binding_rel = fix_binding_path.resolve().relative_to(
                _ROOT).as_posix()
            for item in build_manifest["items"]:
                if item["path"] == target_rel:
                    item["sha256_at_registration"] = new_digest
                    item["sha256_at_end"] = new_digest
                if item["path"] == binding_rel:
                    synced_binding = hashlib.sha256(
                        fix_binding_path.read_bytes()).hexdigest()
                    item["sha256_at_registration"] = synced_binding
                    item["sha256_at_end"] = synced_binding
            build_manifest_path.write_text(
                json.dumps(build_manifest, indent=1), encoding="utf-8")
            cfg["frozen_sources"]["build_source_manifest"] = {
                "path": str(build_manifest_path),
                "sha256": hashlib.sha256(
                    build_manifest_path.read_bytes()).hexdigest()}
            cfg_path = _dump_yaml(
                module_scratch / "configs" / "spatial_badver.yaml", cfg)
            bad_sha = hashlib.sha256(
                cfg_path.read_bytes()).hexdigest()
            with mock.patch.object(prep, "PINNED_CONFIG_SHA256", bad_sha), \
                    mock.patch.object(
                        prep, "PINNED_CONFIG_REL",
                        cfg_path.relative_to(_ROOT).as_posix()):
                code = _run_prepare(
                    spatial_world, out, "build",
                    "2030-01-01T00:00:00+00:00", config_path=cfg_path)
            assert code == prep.EXIT_REFUSED
            status = json.loads((out / "builder" / "run_status.json")
                                .read_text(encoding="utf-8"))
            assert "version" in status["failure"]
        finally:
            definition_path.write_bytes(original)


# ---------------------------------------------------------------------------
# the real PowerShell wrapper against FakePython


FAKE_PYTHON_SOURCE = '''
"""Test double: records argv, writes small synthetic stage evidence."""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
log_path = os.environ.get("FAKE_PY_LOG")
if log_path:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"argv": argv}) + chr(10))
scenario_path = os.environ.get("FAKE_PY_SCENARIO")
scenario = {}
if scenario_path and Path(scenario_path).is_file():
    scenario = json.loads(Path(scenario_path).read_text(encoding="utf-8"))

def arg_value(name):
    if name in argv:
        return argv[argv.index(name) + 1]
    return None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

out_dir = Path(arg_value("--output-dir"))
mode = arg_value("--mode") or "build"
is_prepare = "prepare_material_spatial_input" in (argv[1] if len(argv) > 1
                                                  else "")
which = "prepare" if is_prepare else "verify"

if scenario.get("sleep_s"):
    time.sleep(float(scenario["sleep_s"]))

if scenario.get("create_status_dir") and which == "verify":
    sabotage = out_dir.parent / "run_status.json"
    sabotage.mkdir(parents=True, exist_ok=True)
    (sabotage / "sentinel.txt").write_text("occupied", encoding="utf-8")
if scenario.get("journal_dir") and which == "verify":
    wrapper_dir = out_dir.parent / "wrapper"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    journal = wrapper_dir / "publish_history.jsonl"
    if journal.is_file():
        journal.unlink()
    journal.mkdir(parents=True, exist_ok=True)
if scenario.get("tmp_dir") and which == "verify":
    (out_dir.parent / "run_status.json.tmp").mkdir(parents=True,
                                                   exist_ok=True)

exit_key = f"{which}_exit"
writes = scenario.get(f"{which}_writes", True)
if writes:
    if which == "prepare":
        builder = out_dir / "builder"
        builder.mkdir(parents=True, exist_ok=True)
        counts = scenario.get("builder_counts",
                              [941, 2446] if mode == "build" else [4, 8])
        definition_counts = scenario.get("definition_counts", counts)
        stages = {"admission": {"status": "COMPLETED"},
                  "protection": {"status": "COMPLETED"}}
        if mode == "build":
            stages["disk_space"] = {"status": "COMPLETED",
                                   "path": str(out_dir),
                                   "free_bytes": 20 * 1024 ** 3,
                                   "required_bytes": 10 * 1024 ** 3}
            stages["disk_space"].update(scenario.get("disk_space_override", {}))
        if mode != "preflight":
            stages["build"] = {"status": "COMPLETED"}
            cache = out_dir / "cache"
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "binding.json").write_text(json.dumps(
                {"experiment": "material_spatial_input_v1",
                 "artifacts": []}), encoding="utf-8")
            (cache / "definition.json").write_text(json.dumps(
                {"experiment": "material_spatial_input_v1",
                 "dev_pool": {"frames": 945, "nonempty_frames": 941,
                              "empty_frames": 4, "instances": 2446,
                              "groups": 21, "s_valid_true": 2341,
                              "s_valid_false": 105,
                              "empty_frame_names": []},
                 "built": {"frames": int(definition_counts[0]),
                           "instances": int(definition_counts[1]),
                           "empty_frames": []}}), encoding="utf-8")
        for missing in scenario.get("stages_missing", []):
            stages.pop(missing, None)
        for stage_name, status in scenario.get("stages_bad", {}).items():
            stages[stage_name] = {"status": status}
        registered = 3
        protection_ok = bool(scenario.get("protection_ok", True))
        items = [{"path": f"src/{i}"} for i in range(registered)]
        (builder / "source_manifest.json").write_text(json.dumps(
            {"identical": protection_ok, "items": items,
             "problems": []}), encoding="utf-8")
        if scenario.get("protection_missing"):
            protection = None
        else:
            protection = {"identical": protection_ok, "problems": [],
                          "registered_sources": registered}
        if scenario.get("protection_problems"):
            protection["problems"] = ["src/0: CHANGED"]
        binding_sha = None
        binding_path = out_dir / "cache" / "binding.json"
        if binding_path.is_file():
            binding_sha = sha(binding_path)
        run_mode = scenario.get("builder_run_mode", mode)
        status = {
            "run_mode": run_mode,
            "experiment": "material_spatial_input_v1",
            "run_id": scenario.get("builder_run_id", out_dir.name),
            "config_sha256": sha(arg_value("--config")),
            "deadline_utc": arg_value("--deadline-utc"),
            "cache_schema": "material_spatial_input_cache_v1",
            "implementation_revision": "review_fix_v1",
            "cache_kind": ("material_spatial_input_full_v1"
                           if mode == "build" else
                           ("material_spatial_input_smoke_v1"
                            if mode == "smoke" else None)),
            "cache_complete": mode == "build",
            "cache_binding_sha256": binding_sha,
            "overall_status": ("COMPLETED" if mode == "preflight"
                               else "BUILT_UNVERIFIED"),
            "exit_code": int(scenario.get("builder_status_exit", 0)),
            "failure": scenario.get("builder_failure"),
            "unfinished": scenario.get("builder_unfinished", []),
            "input_admitted": False,
            "interrupted_by_deadline": False,
            "built_frames": int(counts[0]),
            "built_instances": int(counts[1]),
            "stages": stages,
            "protection": protection,
        }
        if scenario.get("builder_count_float"):
            status["built_frames"] = float(status["built_frames"])
        (builder / "run_status.json").write_text(json.dumps(status),
                                                 encoding="utf-8")
        if scenario.get("drift_binding_after_prepare"):
            # a genuine drift AFTER the recorded digest: the wrapper's
            # builder-gate binding check must refuse
            binding_path.write_text(json.dumps(
                {"experiment": "material_spatial_input_v1",
                 "drifted": True}), encoding="utf-8")
    else:
        # --output-dir IS the verification dir for the verify call
        verification = out_dir
        verification.mkdir(parents=True, exist_ok=True)
        binding_sha = None
        binding_path = out_dir.parent / "cache" / "binding.json"
        if binding_path.is_file():
            binding_sha = sha(binding_path)
        if scenario.get("drift_binding_during_verify"):
            binding_path.write_text(json.dumps(
                {"experiment": "material_spatial_input_v1",
                 "drifted_during_verify": True}), encoding="utf-8")
            binding_sha = sha(binding_path)
        is_full = mode == "full"
        counts = scenario.get("verifier_counts",
                              scenario.get(
                                  "builder_counts",
                                  [941, 2446] if is_full else [4, 8]))
        stages = {"inventory": {"status": "COMPLETED"},
                  "cache_identity": {"status": "COMPLETED"},
                  "independent_rebuild": {"status": "COMPLETED"},
                  "protection": {"status": "COMPLETED"},
                  "final_binding": {"status": "COMPLETED",
                                    "sha256": binding_sha}}
        protection_ok = bool(scenario.get("protection_ok", True))
        status = {
            "mode": scenario.get("verify_mode",
                                 "full" if is_full else "smoke"),
            "experiment": "material_spatial_input_v1",
            "run_id": Path(arg_value("--cache-dir")
                           or out_dir.parent).name,
            "config_sha256": sha(arg_value("--config")),
            "deadline_utc": arg_value("--deadline-utc"),
            "cache_binding_sha256": binding_sha,
            "cache_schema": "material_spatial_input_cache_v1",
            "implementation_revision": "review_fix_v1",
            "overall_status": "VERIFIED", "exit_code": 0,
            "failure": scenario.get("verify_failure"),
            "unfinished": scenario.get("verify_unfinished", []),
            "input_admitted": False,
            "protection": {"identical": protection_ok, "problems": []},
            "stages": stages,
        }
        report_counts = scenario.get("report_counts", counts)
        outcome = {"rebuilt_frames": int(counts[0]),
                   "rebuilt_instances": int(counts[1]),
                   "ledger_rows": int(counts[1])}
        status["outcome"] = outcome
        stages["independent_rebuild"].update(outcome)
        report = {
            "mode": scenario.get("report_mode", status["mode"]),
            "run_id": scenario.get("report_run_id", status["run_id"]),
            "experiment": status["experiment"],
            "config_sha256": status["config_sha256"],
            "deadline_utc": status["deadline_utc"],
            "cache_binding_sha256": status["cache_binding_sha256"],
            "cache_schema": status["cache_schema"],
            "implementation_revision": status["implementation_revision"],
            "counts": {"rebuilt_frames": int(report_counts[0]),
                       "rebuilt_instances": int(report_counts[1]),
                       "ledger_rows": int(report_counts[1])},
            "outcome": outcome,
            "failure": scenario.get("report_failure",
                                    scenario.get("verify_failure")),
            "problems": scenario.get("report_problems", []),
        }
        report_path = verification / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        status["report_sha256"] = sha(report_path)
        if scenario.get("status_count_float"):
            status["outcome"] = dict(outcome, rebuilt_frames=float(counts[0]))
        if scenario.get("stage_count_float"):
            status["stages"]["independent_rebuild"]["rebuilt_frames"] = float(counts[0])
        if scenario.get("drift_report_after_status"):
            report["run_id"] = "drifted_report"
            report_path.write_text(json.dumps(report), encoding="utf-8")
        (verification / "status.json").write_text(
            json.dumps(status), encoding="utf-8")

sys.exit(int(scenario.get(exit_key, 0)))
'''


@pytest.fixture(scope="module")
def wrapper_sandbox(spatial_world, module_scratch, artifacts):
    """A byte-identical wrapper copy in the sandbox; the wrapper's
    PSScriptRoot parent is the sandbox root and its experiment root is
    <sandbox>/analysis/runs/material_spatial_input_v1."""
    # PS5/.NET still has MAX_PATH-sensitive IO even when Python accepts
    # long paths. Keep this owned sandbox shallow, inside session scratch.
    sandbox = artifacts / ("w_" + uuid.uuid4().hex[:6])
    experiment_root = sandbox / "analysis" / "runs" / \
        "material_spatial_input_v1"
    (sandbox / "scripts").mkdir(parents=True)
    experiment_root.mkdir(parents=True)
    (sandbox / "configs" / "experiments").mkdir(parents=True)
    wrapper_src = _ROOT / "scripts" / "manual_material_spatial_input.ps1"
    (sandbox / "scripts" / "manual_material_spatial_input.ps1") \
        .write_bytes(wrapper_src.read_bytes())
    (sandbox / "configs" / "experiments" /
     "material_spatial_input_v1.yaml").write_text(
        "experiment: material_spatial_input_v1\n", encoding="utf-8")
    (sandbox / "FakePython.py").write_text(FAKE_PYTHON_SOURCE,
                                           encoding="utf-8")
    (sandbox / "FakePython.cmd").write_text(
        "@echo off" + chr(13) + chr(10)
        + f'"{sys.executable}" "{sandbox / "FakePython.py"}" %*'
        + chr(13) + chr(10), encoding="utf-8")
    return {"sandbox": sandbox, "experiment_root": experiment_root}


def _clock_file(sandbox, token, base_iso, step_s):
    path = sandbox / f"clock_{token}.json"
    path.write_text(json.dumps({"base": base_iso, "step_s": step_s}),
                    encoding="utf-8")
    return str(path)


class TestRealPowerShellWrapper:
    def _run(self, wrapper_sandbox, out_dir, mode="build", budget="5",
             scenario=None, max_frames=None, clock=None,
             python_exe="fake", rollback_io_fault=False):
        sandbox = wrapper_sandbox["sandbox"]
        token = uuid.uuid4().hex[:8]
        scenario_path = sandbox / f"scenario_{token}.json"
        scenario_path.write_text(json.dumps(scenario or {}),
                                 encoding="utf-8")
        log_path = sandbox / f"calls_{token}.jsonl"
        env = dict(os.environ)
        env["FAKE_PY_SCENARIO"] = str(scenario_path)
        env["FAKE_PY_LOG"] = str(log_path)
        env.pop("MATERIAL_SPATIAL_WRAPPER_CLOCK_FILE", None)
        if clock is not None:
            env["MATERIAL_SPATIAL_WRAPPER_CLOCK_FILE"] = clock
        script = sandbox / "scripts" / "manual_material_spatial_input.ps1"
        if rollback_io_fault:
            # Only the clock/IO seam changes in this owned copy. The
            # real gate and publisher run; the fault starts AFTER true
            # has been committed, at clock call four.
            source = script.read_text(encoding="utf-8-sig")
            needle = "$script:ClockCalls += 1"
            assert source.count(needle) == 1
            source = source.replace(needle, needle + "\n" +
                "        if ($k -eq 3) {\n"
                "            New-Item -ItemType Directory -Path "
                "(Join-Path $script:RunRoot 'run_status.json.tmp') | Out-Null\n"
                "        }")
            script = sandbox / "scripts" / f"io_fault_{token}.ps1"
            script.write_text(source, encoding="utf-8-sig")
        args = ["powershell.exe", "-NoProfile", "-NonInteractive",
                "-File",
                str(script),
                "-Mode", mode, "-OutDir", str(out_dir),
                "-BudgetMinutes", budget]
        if python_exe is not None:
            args += ["-PythonExe", str(sandbox / "FakePython.cmd")
                     if python_exe == "fake" else str(python_exe)]
        if max_frames is not None:
            args += ["-MaxFrames", str(max_frames)]
        proc = subprocess.run(args, capture_output=True, text=True,
                              env=env, timeout=180)
        if proc.stderr:
            print("wrapper stderr:", proc.stderr)
        calls = []
        if log_path.exists():
            calls = [json.loads(line) for line in
                     log_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        return proc, calls

    @staticmethod
    def _status(out_dir):
        return json.loads((Path(out_dir) / "run_status.json")
                          .read_text(encoding="utf-8"))

    @staticmethod
    def _journal(out_dir):
        path = Path(out_dir) / "wrapper" / "publish_history.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]

    def test_build_success_is_admitted_with_matching_evidence(
            self, wrapper_sandbox, module_scratch):
        out = wrapper_sandbox["experiment_root"] / "runs" / "ok_build"
        proc, calls = self._run(wrapper_sandbox, out)
        assert proc.returncode == 0, proc.stderr
        status = self._status(out)
        assert status["overall_status"] == "COMPLETED"
        assert status["input_admitted"] is True
        assert len(calls) == 2
        deadlines = {entry["argv"][entry["argv"].index("--deadline-utc")
                                   + 1] for entry in calls}
        assert len(deadlines) == 1
        (deadline_value,) = deadlines
        assert deadline_value.endswith("Z")
        assert status["deadline_utc"] == deadline_value
        journal = self._journal(out)
        assert [event["published"] for event in journal] == ["COMPLETED"]

    def test_spaces_path_stays_one_argv_element(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "spa ced dir"
        proc, calls = self._run(wrapper_sandbox, out)
        assert proc.returncode == 0, proc.stderr
        prep_argv = calls[0]["argv"]
        index = prep_argv.index("--output-dir")
        assert prep_argv[index + 1] == str(out)
        assert (out / "run_status.json").is_file()

    def test_builder_failure_skips_the_verifier(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "builder_fail"
        proc, calls = self._run(wrapper_sandbox, out,
                                scenario={"prepare_exit": 3})
        assert proc.returncode == 3
        assert len(calls) == 1, "the verifier ran after a builder failure"
        status = self._status(out)
        assert status["input_admitted"] is False
        assert status["overall_status"] == "FAILED"

    def test_builder_exit2_keeps_partial_and_skips_the_verifier(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "builder_e2"
        proc, calls = self._run(wrapper_sandbox, out,
                                scenario={"prepare_exit": 2})
        assert proc.returncode == 2
        assert len(calls) == 1, "the verifier ran after a PARTIAL builder"
        status = self._status(out)
        assert status["overall_status"] == "PARTIAL"
        assert status["input_admitted"] is False

    def test_verifier_partial_is_kept_not_failed(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "v_partial"
        proc, _calls = self._run(wrapper_sandbox, out,
                                 scenario={"verify_exit": 2})
        assert proc.returncode == 2
        status = self._status(out)
        assert status["overall_status"] == "PARTIAL"
        assert status["input_admitted"] is False

    def test_exit0_without_verifier_evidence_is_not_admitted(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "no_ev"
        proc, _calls = self._run(wrapper_sandbox, out,
                                 scenario={"verify_writes": False})
        assert proc.returncode == 1
        status = self._status(out)
        assert status["input_admitted"] is False
        assert status["overall_status"] == "FAILED"

    @pytest.mark.parametrize("scenario,fragment", [
        ({"protection_missing": True}, "protection"),
        ({"protection_ok": False}, "protection"),
        ({"protection_problems": True}, "protection"),
        ({"stages_missing": ["build"]}, "stage"),
        ({"stages_bad": {"build": "FAILED"}}, "stage"),
        ({"builder_run_mode": "smoke"}, "identity"),
        ({"builder_status_exit": 4}, "failure"),
        ({"builder_failure": "boom"}, "failure"),
        ({"builder_unfinished": ["stage x"]}, "unfinished"),
    ])
    def test_builder_gate_refusals(self, wrapper_sandbox, scenario,
                                   fragment):
        out = wrapper_sandbox["experiment_root"] / "runs" / \
            ("gate_" + fragment + "_" + str(abs(hash(fragment)) % 97))
        proc, _calls = self._run(wrapper_sandbox, out, scenario=scenario)
        assert proc.returncode == 1, (fragment, proc.stdout, proc.stderr)
        status = self._status(out)
        assert status["input_admitted"] is False
        assert status["overall_status"] == "FAILED"
        assert fragment in json.dumps(status).lower() or \
            fragment in proc.stderr.lower(), fragment

    @pytest.mark.parametrize("scenario,fragment", [
        ({"verifier_counts": [940, 2400]}, "approved"),
        ({"builder_counts": [940, 2400],
          "verifier_counts": [941, 2446]}, "disagree"),
    ])
    def test_count_gates(self, wrapper_sandbox, scenario, fragment):
        out = wrapper_sandbox["experiment_root"] / "runs" / \
            ("counts_" + fragment)
        proc, _calls = self._run(wrapper_sandbox, out, scenario=scenario)
        assert proc.returncode == 1, (fragment, proc.stdout)
        status = self._status(out)
        assert status["input_admitted"] is False
        assert status["overall_status"] == "FAILED"

    def test_report_cross_run_is_refused(self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "xrun"
        proc, _calls = self._run(
            wrapper_sandbox, out,
            scenario={"report_run_id": "another_run"})
        assert proc.returncode == 1
        status = self._status(out)
        assert status["overall_status"] == "FAILED"

    def test_report_wrong_mode_is_refused(self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "bad_mode"
        proc, _calls = self._run(
            wrapper_sandbox, out, scenario={"report_mode": "smoke"})
        assert proc.returncode == 1
        status = self._status(out)
        assert status["overall_status"] == "FAILED"

    def test_report_failure_is_refused(self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "v_fail"
        proc, _calls = self._run(
            wrapper_sandbox, out, scenario={"report_failure": "boom"})
        assert proc.returncode == 1
        status = self._status(out)
        assert status["overall_status"] == "FAILED"

    @pytest.mark.parametrize("scenario", [
        {"report_problems": ["bad source"]},
        {"report_counts": [940, 2446]},
        {"drift_report_after_status": True},
        {"builder_counts": [940, 2445],
         "verifier_counts": [940, 2445], "definition_counts": [941, 2446]},
    ])
    def test_report_is_independently_bound_and_counts_are_approved(
            self, wrapper_sandbox, scenario):
        out = wrapper_sandbox["experiment_root"] / "runs" / uuid.uuid4().hex
        proc, _ = self._run(wrapper_sandbox, out, scenario=scenario)
        assert proc.returncode == 1, proc.stdout
        assert self._status(out)["input_admitted"] is False

    @pytest.mark.parametrize("field", ["builder_count_float",
                                       "status_count_float", "stage_count_float"])
    def test_equal_valued_certificate_counts_still_require_integers(
            self, wrapper_sandbox, field):
        out = wrapper_sandbox["experiment_root"] / "runs" / field
        proc, _ = self._run(wrapper_sandbox, out, scenario={field: True})
        assert proc.returncode == 1, proc.stdout
        assert self._status(out)["input_admitted"] is False

    @pytest.mark.parametrize("mode", ["preflight", "smoke", "build"])
    @pytest.mark.parametrize("step,events", [
        (30, ["PARTIAL"]), (20, ["COMPLETED", "PARTIAL"]),
    ])
    def test_publish_effective_result_controls_exit_for_every_mode(
            self, wrapper_sandbox, mode, step, events):
        token = uuid.uuid4().hex
        out = wrapper_sandbox["experiment_root"] / "runs" / token
        clock = _clock_file(wrapper_sandbox["sandbox"], token,
                            "2026-01-01T00:00:00Z", step)
        proc, _ = self._run(wrapper_sandbox, out, mode=mode, budget="1",
                            max_frames=6 if mode == "smoke" else None,
                            clock=clock)
        assert proc.returncode == 2, proc.stdout
        assert "overall: COMPLETED" not in proc.stdout
        status = self._status(out)
        assert status["overall_status"] == "PARTIAL"
        assert status["input_admitted"] is False
        assert status["exit_code"] == proc.returncode
        assert [item["published"] for item in self._journal(out)] == events

    def test_default_interpreter_refuses_the_test_clock_without_writes(
            self, wrapper_sandbox):
        token = uuid.uuid4().hex
        out = wrapper_sandbox["experiment_root"] / "runs" / token
        clock = _clock_file(wrapper_sandbox["sandbox"], token,
                            "2035-01-01T00:00:00Z", 0)
        proc, calls = self._run(wrapper_sandbox, out, clock=clock,
                                python_exe=None)
        assert proc.returncode == 1
        assert "clock" in proc.stderr.lower()
        assert not out.exists()
        assert calls == []

    def test_binding_drift_before_the_verifier_is_refused(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "drift_pre"
        proc, calls = self._run(
            wrapper_sandbox, out,
            scenario={"drift_binding_after_prepare": True})
        assert proc.returncode == 1
        assert len(calls) == 1, "the verifier ran despite the drift"
        status = self._status(out)
        assert status["overall_status"] == "FAILED"
        assert "bind" in json.dumps(status).lower()

    def test_binding_drift_during_verify_is_refused(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "drift_mid"
        proc, calls = self._run(
            wrapper_sandbox, out,
            scenario={"drift_binding_during_verify": True})
        assert proc.returncode == 1
        assert len(calls) == 2
        status = self._status(out)
        assert status["overall_status"] == "FAILED"
        assert "drift" in json.dumps(status).lower()

    def test_publish_expired_before_the_publish_is_partial(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "expired"
        # the sequence clock: call 1 computes the deadline (base + 1
        # minute), call 2 (the pre-publish check) lands exactly on the
        # deadline -> expired before any COMPLETED write
        clock = _clock_file(wrapper_sandbox["sandbox"], "expired",
                            "2026-01-01T00:00:00Z", 60)
        proc, _calls = self._run(wrapper_sandbox, out, budget="1",
                                 clock=clock)
        assert proc.returncode == 2
        status = self._status(out)
        assert status["overall_status"] == "PARTIAL"
        assert status["input_admitted"] is False
        journal = self._journal(out)
        assert [event["published"] for event in journal] == ["PARTIAL"]

    def test_publish_rolling_over_the_deadline_rolls_back(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "rollback"
        # calls: 1 deadline, 2 pre-publish (1/3), 3 pre-replace (2/3),
        # 4 post-commit (3/3 = the deadline) -> the committed COMPLETED
        # is rolled back to PARTIAL and both events stay in the journal
        clock = _clock_file(wrapper_sandbox["sandbox"], "rollback",
                            "2026-01-01T00:00:00Z", 20)
        proc, _calls = self._run(wrapper_sandbox, out, budget="1",
                                 clock=clock)
        assert proc.returncode == 2
        status = self._status(out)
        assert status["overall_status"] == "PARTIAL"
        assert status["input_admitted"] is False
        journal = self._journal(out)
        assert [event["published"] for event in journal] == \
            ["COMPLETED", "PARTIAL"], \
            "the journal must keep the success attempt and the rollback"

    def test_journal_failure_after_a_true_publish_closes_failed(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "jfail"
        proc, _calls = self._run(wrapper_sandbox, out,
                                 scenario={"journal_dir": True})
        assert proc.returncode != 0
        assert "overall: COMPLETED" not in proc.stdout
        # the committed true status must not survive the failure closure
        assert not (out / "run_status.json").exists(), \
            "the success status survived a post-publish journal failure"

    def test_rollback_write_failure_still_closes_failed(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "rwfail"
        clock = _clock_file(wrapper_sandbox["sandbox"], "rwfail",
                            "2026-01-01T00:00:00Z", 20)
        proc, _calls = self._run(wrapper_sandbox, out, budget="1",
                                 clock=clock, rollback_io_fault=True)
        assert proc.returncode != 0
        assert "overall: COMPLETED" not in proc.stdout
        assert not (out / "run_status.json").exists()
        assert [item["published"] for item in self._journal(out)] == [
            "COMPLETED"]

    def test_status_write_failure_exits_nonzero_without_success(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "write_fail"
        proc, _calls = self._run(
            wrapper_sandbox, out, scenario={"create_status_dir": True})
        assert proc.returncode != 0
        assert "overall: COMPLETED" not in proc.stdout
        assert (out / "run_status.json").is_dir()

    def test_fake_preflight_status_and_wording_are_accurate(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "pf"
        proc, calls = self._run(wrapper_sandbox, out, mode="preflight")
        assert proc.returncode == 0, proc.stderr
        assert len(calls) == 1, "preflight must not call the verifier"
        status = self._status(out)
        assert status["overall_status"] == "COMPLETED"
        assert status["input_admitted"] is False
        assert status["builder_status"] == "COMPLETED"
        assert status["verifier_status"] == "NOT_RUN"
        assert "not run" in status["detail"].lower()
        assert "binding" in status["detail"].lower()

    def test_smoke_success_is_never_admitted(
            self, wrapper_sandbox):
        out = wrapper_sandbox["experiment_root"] / "runs" / "smoke_ok"
        proc, _calls = self._run(wrapper_sandbox, out, mode="smoke",
                                 max_frames=6)
        assert proc.returncode == 0
        status = self._status(out)
        assert status["overall_status"] == "COMPLETED"
        assert status["input_admitted"] is False

    def test_budget_and_param_gates_are_refused(self, wrapper_sandbox):
        root = wrapper_sandbox["experiment_root"] / "runs"
        cases = [
            ("build", "0", None, None),        # zero budget
            ("build", "25", None, None),       # over the build cap
            ("smoke", "6", None, 6),           # over the smoke cap
            ("smoke", "5", None, 0),           # MaxFrames 0
            ("smoke", "5", None, 1),           # fewer than mandatory empties
            ("smoke", "5", None, 2),
            ("smoke", "5", None, 3),
            ("smoke", "5", None, 13),          # MaxFrames too big
            ("build", "5", None, 3),           # MaxFrames outside smoke
        ]
        for mode, budget, _x, max_frames in cases:
            target = root / ("gate_" + mode + "_" + budget + "_" +
                             str(max_frames))
            proc, _calls = self._run(wrapper_sandbox, target, mode=mode,
                                     budget=budget,
                                     max_frames=max_frames)
            assert proc.returncode == 1, (mode, budget, max_frames)
            assert "refused" in proc.stderr

    @pytest.mark.parametrize("scenario", [
        {"stages_missing": ["disk_space"]},
        {"disk_space_override": {"free_bytes": 10 * 1024 ** 3 - 1}},
        {"disk_space_override": {"required_bytes": 1}},
        {"disk_space_override": {"free_bytes": float(20 * 1024 ** 3)}},
        {"disk_space_override": {"path": "D:\\other_volume\\wrong"}},
    ])
    def test_full_wrapper_requires_disk_certificate(self, wrapper_sandbox, scenario):
        out = wrapper_sandbox["experiment_root"] / "runs" / uuid.uuid4().hex[:8]
        proc, calls = self._run(wrapper_sandbox, out, scenario=scenario)
        assert proc.returncode != 0
        assert len(calls) == 1, "unqualified disk evidence must not reach verifier"
        status = self._status(out)
        assert status["input_admitted"] is False
        assert status["overall_status"] == "FAILED"
        assert "disk" in status["detail"].lower()

    def test_refused_targets_leave_the_filesystem_untouched(
            self, wrapper_sandbox):
        sandbox = wrapper_sandbox["sandbox"]
        experiment_root = wrapper_sandbox["experiment_root"]
        existing_dir = experiment_root / "runs" / "exists_dir"
        existing_dir.mkdir(parents=True)
        (existing_dir / "sentinel.txt").write_text("keep",
                                                   encoding="utf-8")
        existing_file = experiment_root / "runs" / "exists_file"
        existing_file.write_text("keep", encoding="utf-8")
        # a similar-prefix sibling of the experiment root
        sibling = sandbox / "analysis" / "runs" / \
            "material_spatial_input_v1_evil"
        sibling.mkdir(parents=True)
        (sibling / "sentinel.txt").write_text("keep", encoding="utf-8")
        # inside the project but outside the experiment root
        other_tree = sandbox / "other_tree"
        other_tree.mkdir()
        before_other = sorted(p.name for p in other_tree.iterdir())
        before_sibling = sorted(p.name for p in sibling.iterdir())
        cases = [
            ("existing dir", existing_dir),
            ("existing file", existing_file),
            ("traversal", experiment_root / "runs" / "sub" / ".." /
             "evil"),
            ("similar-prefix sibling", sibling / "out"),
            ("experiment root itself", experiment_root),
            ("in-project outside the root", other_tree / "out"),
        ]
        for label, target in cases:
            proc, calls = self._run(wrapper_sandbox, target)
            assert proc.returncode == 1, label
            assert "refused" in proc.stderr, label
            assert calls == [], f"{label}: a child was invoked"
        assert (existing_dir / "sentinel.txt").read_text() == "keep"
        assert existing_file.read_text() == "keep"
        assert sorted(p.name for p in sibling.iterdir()) == \
            before_sibling
        assert sorted(p.name for p in other_tree.iterdir()) == \
            before_other
        assert not (experiment_root / "runs" / "evil").exists()
        assert not (experiment_root / "runs" / "sub").exists()
        assert not (other_tree / "out").exists()
        assert not (sibling / "out").exists()

    def test_junction_ancestor_is_refused(
            self, wrapper_sandbox):
        real = wrapper_sandbox["sandbox"] / "junction_real"
        real.mkdir(exist_ok=True)
        link = wrapper_sandbox["experiment_root"] / "junction_link"
        if not link.exists():
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link),
                            str(real)], check=True,
                           capture_output=True)
        try:
            proc, calls = self._run(wrapper_sandbox, link / "out")
            assert proc.returncode == 1
            assert "refused" in proc.stderr
            assert calls == []
            assert not (real / "out").exists()
        finally:
            # remove the link itself first, never recurse through it
            os.rmdir(link)


_REVIEW_ENV_NAMES = ("TEMP", "TMP", "MATERIAL_FUSION_TEST_ARTIFACTS")


@contextmanager
def _review_environment_scope(scratch, record, saved=None):
    """The review runner and its fault tests use this same finally path."""
    original = dict(saved) if saved is not None else {
        name: os.environ.get(name) for name in _REVIEW_ENV_NAMES}
    record.update(saved=original, set_to=str(scratch))
    try:
        for name in _REVIEW_ENV_NAMES:
            os.environ[name] = str(scratch)
        yield record
    finally:
        errors, restored = [], {}
        for name in _REVIEW_ENV_NAMES:
            try:
                if original[name] is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = original[name]
                restored[name] = os.environ.get(name)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        record.update(
            restored=restored,
            restored_values_match_saved={
                name: name in restored and restored[name] == original[name]
                for name in _REVIEW_ENV_NAMES},
            errors=errors)
        if errors:
            raise RuntimeError("environment restoration failed: " +
                               "; ".join(errors))


def _review_remove_scratch(scratch, project_root):
    """No recursive delete: validate the WHOLE tree, then remove leaves."""
    from core.material_spatial_input_admission import (
        validated_scratch_cleanup_paths,
    )

    scratch = Path(scratch)
    paths = validated_scratch_cleanup_paths(scratch, project_root)
    for path in paths:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    if scratch.exists():
        scratch.rmdir()
    return {"outcome": "REMOVED", "scratch_root": str(scratch),
            "remaining_paths": [], "removed_children": len(paths)}


class TestReviewRunnerSafety:
    @pytest.mark.parametrize("fail", [False, True])
    def test_normal_and_exceptional_environment_restore_with_real_cleanup(
            self, module_scratch, fail):
        root = module_scratch / f"runner_scope_{fail}"
        (root / "child").mkdir(parents=True)
        (root / "child" / "payload.txt").write_text("owned")
        before = {name: os.environ.get(name) for name in _REVIEW_ENV_NAMES}
        record = {}
        try:
            try:
                with _review_environment_scope(root, record):
                    assert all(os.environ[name] == str(root)
                               for name in _REVIEW_ENV_NAMES)
                    assert (root / "child" / "payload.txt").exists()
                    if fail:
                        raise RuntimeError("injected main failure")
            except RuntimeError as exc:
                assert fail and str(exc) == "injected main failure"
        finally:
            result = _review_remove_scratch(root, _ROOT)
        assert result["removed_children"] == 2
        assert not root.exists()
        assert all(record["restored_values_match_saved"].values())
        assert {name: os.environ.get(name) for name in _REVIEW_ENV_NAMES} == before

    @pytest.mark.parametrize("location", ["root", "nested"])
    def test_cleanup_refuses_root_and_nested_junction_before_any_deletion(
            self, module_scratch, location):
        from core.material_spatial_input_admission import AdmissionRefused

        sentinel = module_scratch / f"outside_sentinel_{location}"
        sentinel.mkdir()
        (sentinel / "sentinel.txt").write_bytes(b"must survive")
        root = module_scratch / f"cleanup_link_{location}"
        if location == "nested":
            (root / "child").mkdir(parents=True)
            (root / "owned.txt").write_bytes(b"also survives refusal")
            link = root / "child" / "junction"
        else:
            link = root
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link),
                        str(sentinel)], check=True, capture_output=True)
        try:
            with pytest.raises(AdmissionRefused, match="reparse|link|junction"):
                _review_remove_scratch(root, _ROOT)
            assert (sentinel / "sentinel.txt").read_bytes() == b"must survive"
            if location == "nested":
                assert (root / "owned.txt").read_bytes() == b"also survives refusal"
        finally:
            os.rmdir(link)

    def test_cleanup_rejects_tree_root_and_project_tree(self, module_scratch):
        from core.material_spatial_input_admission import AdmissionRefused

        for target in (_ROOT, _ROOT / ".test_artifacts", _ROOT / "datasets"):
            with pytest.raises(AdmissionRefused):
                _review_remove_scratch(target, _ROOT)
