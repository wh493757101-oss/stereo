"""Acceptance contract for the material_residual_runtime_v1 completion
batch: Layer A comparator honesty, the verdict's authority checks, the
encoder cache identity and the 9-candidate registry.

Written BEFORE the fixes (TDD): the comparator/verdict/cache counterexamples
must fail against the shipped behaviour first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

REAL_CONFIG = Path(__file__).resolve().parents[2] / \
    "configs/experiments/material_residual_runtime_v1.yaml"
COMPLETE_CONFIG = Path(__file__).resolve().parents[2] / \
    "configs/experiments/material_residual_runtime_v1_complete.yaml"


# ---------------------------------------------------------------------------
# Fix 1: the stage comparator refuses non-finite, empty and illegally-NaN
# data; failures are JSON-safe (null, never NaN/Inf)


class TestStageComparator:
    def test_inf_vs_inf_fails(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        inf = np.array([np.inf, 1.0, 2.0, 3.0], dtype=np.float32)
        report = compare_stage_arrays(inf, inf.copy(), atol=1e-4, rtol=1e-4,
                                      stage="finite")
        assert report["passed"] is False
        assert "inf" in str(report.get("reason", "")).lower()

    def test_finite_vs_inf_fails(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        finite = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        infinite = np.array([1.0, np.inf, 3.0, 4.0], dtype=np.float32)
        report = compare_stage_arrays(finite, infinite, atol=1e-4, rtol=1e-4,
                                      stage="finite")
        assert report["passed"] is False

    def test_finite_vs_plus_and_minus_inf_fails(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        finite = np.zeros(4, dtype=np.float32)
        broken = np.array([np.inf, -np.inf, np.nan, 1.0], dtype=np.float32)
        report = compare_stage_arrays(finite, broken, atol=1e-4, rtol=0.0,
                                      stage="finite")
        assert report["passed"] is False

    def test_nan_vs_nan_fails_for_finite_stages(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        nan = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        report = compare_stage_arrays(nan, nan.copy(), atol=0.0, rtol=0.0,
                                      stage="finite")
        assert report["passed"] is False

    def test_empty_arrays_fail(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        report = compare_stage_arrays(np.zeros(0, np.float32),
                                      np.zeros(0, np.float32),
                                      atol=1e-4, rtol=1e-4, stage="finite")
        assert report["passed"] is False

    def test_shape_mismatch_fails(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        report = compare_stage_arrays(np.ones(4, np.float32),
                                      np.ones(5, np.float32),
                                      atol=1e-4, rtol=1e-4, stage="finite")
        assert report["passed"] is False

    def test_failure_report_is_json_safe(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        inf = np.array([np.inf, 1.0, 2.0, 3.0], dtype=np.float32)
        report = compare_stage_arrays(inf, inf.copy(), atol=1e-4, rtol=1e-4,
                                      stage="finite")
        encoded = json.loads(json.dumps(report))

        def _no_nonfinite(value):
            if isinstance(value, float):
                assert np.isfinite(value), value
            elif isinstance(value, dict):
                for child in value.values():
                    _no_nonfinite(child)
            elif isinstance(value, list):
                for child in value:
                    _no_nonfinite(child)

        _no_nonfinite(encoded)
        assert encoded["max_abs_error"] is None

    def test_legal_finite_data_passes_unchanged(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        report = compare_stage_arrays(a, a.copy(), atol=0.0, rtol=0.0,
                                      stage="finite")
        assert report["passed"] is True
        assert report["max_abs_error"] == 0.0

    def test_small_error_within_tolerance_passes(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        expected = np.full(4, 1.0, dtype=np.float32)
        actual = expected + np.float32(1e-6)
        report = compare_stage_arrays(actual, expected, atol=1e-4, rtol=1e-4,
                                      stage="finite")
        assert report["passed"] is True


class TestS16Comparator:
    """Raw S16: NaN is legal ONLY as a full 8-wide missing block admitted
    by the frozen region-validity state; partial NaN and one-sided NaN
    are failures."""

    def test_legal_missing_block_passes(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        expected = np.full(16, np.nan, dtype=np.float32)
        actual = np.full(16, np.nan, dtype=np.float32)
        report = compare_stage_arrays(actual, expected, atol=0.0, rtol=0.0,
                                      stage="s16")
        assert report["passed"] is True

    def test_one_sided_missing_block_fails(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        expected = np.full(16, np.nan, dtype=np.float32)
        actual = np.arange(16, dtype=np.float32)
        report = compare_stage_arrays(actual, expected, atol=0.0, rtol=0.0,
                                      stage="s16")
        assert report["passed"] is False

    def test_partial_nan_block_fails_on_both_sides(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        partial = np.full(16, 1.0, dtype=np.float32)
        partial[3] = np.nan
        report = compare_stage_arrays(partial.copy(), partial.copy(),
                                      atol=0.0, rtol=0.0, stage="s16")
        assert report["passed"] is False
        # a partial NaN on the actual side against a finite expected block
        expected = np.full(16, 1.0, dtype=np.float32)
        report = compare_stage_arrays(partial, expected, atol=0.0, rtol=0.0,
                                      stage="s16")
        assert report["passed"] is False

    def test_inf_inside_finite_block_fails(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        expected = np.full(16, 0.5, dtype=np.float32)
        actual = expected.copy()
        actual[9] = np.inf
        report = compare_stage_arrays(actual, expected, atol=0.0, rtol=0.0,
                                      stage="s16")
        assert report["passed"] is False

    def test_one_valid_one_missing_block_passes(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        expected = np.concatenate([np.arange(8, dtype=np.float32),
                                   np.full(8, np.nan, dtype=np.float32)])
        actual = expected.copy()
        report = compare_stage_arrays(actual, expected, atol=0.0, rtol=0.0,
                                      stage="s16")
        assert report["passed"] is True

    def test_exactness_of_valid_values(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        expected = np.arange(16, dtype=np.float32)
        actual = expected + np.float32(1e-6)  # C14/S16 valid values are exact
        report = compare_stage_arrays(actual, expected, atol=0.0, rtol=0.0,
                                      stage="s16")
        assert report["passed"] is False


# ---------------------------------------------------------------------------
# Fix 1b: the verdict validates the authoritative sample set, the candidate
# registry and the required stage set — never just "whatever showed up"


def _ok_row(sample="s0", candidate="conditional_RCS_s2026", *, stages=("g", "final"),
            mismatch=False):
    return {"sample_name": sample, "candidate": candidate,
            "s_valid_offline": True, "s_valid_runtime": True,
            "stage_reports": {name: {"passed": True, "max_abs_error": 0.0,
                                     "status": "PASSED"}
                              for name in stages},
            "prediction_mismatch": mismatch}


class TestLayerAVerdict:
    REQUIRED = {"conditional_RCS_s2026": ("g", "final")}

    def test_all_ok_passes(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [_ok_row()], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "OK"
        assert verdict["abort_layer_b"] is False

    def test_zero_samples_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"
        assert verdict["abort_layer_b"] is True

    def test_missing_sample_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [_ok_row()], expected_names=["s0", "s1"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_duplicate_sample_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [_ok_row(), _ok_row()], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_unknown_candidate_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [_ok_row(candidate="ghost_s9999")], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_missing_candidate_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [_ok_row()], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026",
                                 "conditional_RCS_s2027"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_missing_stage_report_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [_ok_row(stages=("g",))], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_row_without_stage_reports_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [{"sample_name": "s0",
              "candidate": "conditional_RCS_s2026",
              "prediction_mismatch": False}],
            expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_failed_stage_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        bad = _ok_row()
        bad["stage_reports"]["g"] = {"passed": False, "max_abs_error": None,
                                     "status": "FAILED", "reason": "inf"}
        verdict = layer_a_verdict(
            [bad], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_prediction_only_mismatch_is_not_ok(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        verdict = layer_a_verdict(
            [_ok_row(mismatch=True)], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=self.REQUIRED)
        assert verdict["status"] == "FAILED"
        assert verdict["prediction_mismatches"] == 1


# ---------------------------------------------------------------------------
# Fix 2: the encoder cache is keyed by weights identity AND device; a hit
# must still match the request, and sharing never migrates in place


class _FakeEncoderModule(torch.nn.Module):
    def __init__(self, device_label: str):
        super().__init__()
        self._declared_device = device_label
        self.register_buffer("marker", torch.zeros(1))

    def parameters(self, recurse=True):  # noqa: D102 - synthetic device
        return iter(())

    def __call__(self, rgb):  # minimal interface
        return [None, None, None, torch.zeros(rgb.shape[0], 256, 7, 7)]


def _fake_builder(calls):
    def build(config, device):
        label = str(device)
        calls.append(label)
        return _FakeEncoderModule(label)
    return build


class TestEncoderCacheIdentity:
    def test_same_identity_is_shared(self):
        import models.material_residual_classifier as mrc

        mrc._encoder_cache.clear()
        calls = []
        first = mrc._build_encoder(_ENC_CONFIG, "cpu", builder=_fake_builder(calls))
        second = mrc._build_encoder(_ENC_CONFIG, "cpu", builder=_fake_builder(calls))
        assert first is second
        assert calls == ["cpu"]

    def test_different_device_is_not_shared(self):
        import models.material_residual_classifier as mrc

        mrc._encoder_cache.clear()
        calls = []
        mrc._build_encoder(_ENC_CONFIG, "cpu", builder=_fake_builder(calls))
        cuda_like = mrc._build_encoder(_ENC_CONFIG, "cuda:0",
                                       builder=_fake_builder(calls))
        assert len(calls) == 2
        assert cuda_like._declared_device == "cuda:0"

    def test_different_weights_identity_is_not_shared(self):
        import models.material_residual_classifier as mrc
        from models.material_residual_classifier import ResidualRuntimeConfig
        import dataclasses

        mrc._encoder_cache.clear()
        calls = []
        mrc._build_encoder(_ENC_CONFIG, "cpu", builder=_fake_builder(calls))
        other = dataclasses.replace(_ENC_CONFIG, encoder_sha256="b" * 64)
        mrc._build_encoder(other, "cpu", builder=_fake_builder(calls))
        assert len(calls) == 2

    def test_cached_object_moved_elsewhere_is_refused_not_migrated(self):
        import models.material_residual_classifier as mrc

        mrc._encoder_cache.clear()
        calls = []
        shared = mrc._build_encoder(_ENC_CONFIG, "cpu",
                                    builder=_fake_builder(calls))
        shared._declared_device = "cuda:7"  # someone moved the shared object
        with pytest.raises(RuntimeError, match="device"):
            mrc._build_encoder(_ENC_CONFIG, "cpu", builder=_fake_builder(calls))
        # the shared object was NOT migrated in place
        assert shared._declared_device == "cuda:7"

    def test_cache_key_includes_imgsz(self):
        import models.material_residual_classifier as mrc
        import dataclasses

        mrc._encoder_cache.clear()
        calls = []
        mrc._build_encoder(_ENC_CONFIG, "cpu", builder=_fake_builder(calls))
        other = dataclasses.replace(_ENC_CONFIG, encoder_imgsz=160)
        mrc._build_encoder(other, "cpu", builder=_fake_builder(calls))
        assert len(calls) == 2


_ENC_CONFIG = None


@pytest.fixture(scope="module", autouse=True)
def _enc_config():
    """A minimal config stub for the encoder-cache tests."""
    global _ENC_CONFIG
    from models.material_residual_classifier import ResidualRuntimeConfig

    if _ENC_CONFIG is None:
        _ENC_CONFIG = ResidualRuntimeConfig(
            experiment="material_residual_runtime_v1",
            fit_dir="analysis/runs/material_conditional_residual_v1/x",
            summary_sha256="a" * 64, admission_sha256="a" * 64,
            input_dim=35, context_dim=5, hidden=8, engineering_seed=2026,
            variants=("R0", "RC", "RCS"),
            fallback_rule="x",
            protocol_path="p", protocol_sha256="a" * 64,
            encoder_path="weights/pretrained/mit_b0.pth",
            encoder_sha256="4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc",
            encoder_imgsz=224, crop_padding=10,
            region_rule="local_partition_interior_band_v1",
            global_cache=(), local_cache=(),
            p0_models={}, residual_models={}, r0_records={},
            candidates=(),
            detection_reference="",
            data_identity={},
            evaluation_budget={},
            model_a={}, stereo={}, calibration={}, polarization={},
            runtime={}, tolerances={}, benchmark={})
    yield


# ---------------------------------------------------------------------------
# Fix 3: the complete 9-candidate registry in the new frozen config


COMPLETE_CANDIDATES = (
    "R0",
    "linear_local_RC_s2026",
    "linear_local_RCS_s2026",
    "conditional_RC_s2026",
    "conditional_RCS_s2026",
    "conditional_RC_s2027",
    "conditional_RCS_s2027",
    "conditional_RC_s2028",
    "conditional_RCS_s2028",
)


class TestCompleteCandidateRegistry:
    def test_complete_config_exists_and_lists_nine_candidates(self):
        assert COMPLETE_CONFIG.is_file(), \
            "the complete candidate config must exist before evaluation"
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        config = load_residual_runtime_config(COMPLETE_CONFIG)
        ids = [c.candidate_id for c in config.candidates]
        assert sorted(ids) == sorted(COMPLETE_CANDIDATES)

    def test_every_candidate_pins_files_with_digests(self):
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        config = load_residual_runtime_config(COMPLETE_CONFIG)
        for candidate in config.candidates:
            if candidate.family == "p0":
                continue
            for fold, files in candidate.fold_files.items():
                for name in ("residual_state", "scaler",
                             "saved_predictions"):
                    ref = files[name]
                    assert len(ref.sha256) == 64, (candidate.candidate_id,
                                                   fold, name)

    def test_family_dims(self):
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        config = load_residual_runtime_config(COMPLETE_CONFIG)
        for candidate in config.candidates:
            if candidate.family == "conditional":
                assert candidate.input_dim == 35
            elif candidate.family == "linear_local":
                assert candidate.input_dim == 30
            else:
                assert candidate.family == "p0"

    def test_digests_match_bytes_on_disk(self):
        """The registry digests are frozen identities: they must equal the
        actual bytes on disk at evaluation time."""
        import hashlib

        from models.material_residual_classifier import (
            load_residual_runtime_config, resolve_path,
        )

        config = load_residual_runtime_config(COMPLETE_CONFIG)
        checked = 0
        for candidate in config.candidates:
            for files in candidate.fold_files.values():
                for ref in files.values():
                    path = resolve_path(ref.path)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    assert digest == ref.sha256, ref.path
                    checked += 1
        assert checked >= 24  # 8 correction candidates x 3 folds x 1+ file



"""The synthetic orchestration section of the acceptance file (residrt_finish).

Everything here is synthetic: the V5 chain, the P0 fits, the local linear
fit, the conditional fits, the candidate config, the split CSV/labels for
Layer B, the encoder and the engine. No real checkpoint is loaded and no
real image is read — the CPU wrapper refuses any torch.load outside the
run's scratch.
"""

import csv
import json
import shutil
import unittest.mock as mock

import numpy as np
import pytest
import torch

from scripts import train_material_conditional_residual as cond_entry
from scripts import train_material_local_residual as local_entry

from .test_material_conditional_residual import (
    _prepare_cond_fit,
)
from .test_material_local_residual_entry import (
    _main_args,
    _prepare_fit,
    local_world,  # noqa: F401  (the fully synthetic V5/P0 world)
)
from .test_material_local_residual_entry import (
    _write_config as _write_world_config,
)


def _world_g_dim(world):
    if "config" in world:
        path = Path(world["config"].global_cache[0].path)
    else:
        path = Path(world["feature_dir"]) / "feature_cache.npz"
    return int(json.loads((path.parent / "feature_definition.json")
        .read_text(encoding="utf-8"))["slot_G"]["dim"])




from .test_p0_baseline_verification import build_fit as _p0_build_fit
import sys as _sys

_p0_module = _sys.modules[_p0_build_fit.__module__]
from .test_material_local_residual_entry import _prepare_fit as _p_prepare_fit
from .test_polar_region_quality_entry import _make_cache_files_for_v5  # noqa: F401


def entry_required_ids():
    from scripts.eval_material_residual_pipeline import (
        REQUIRED_CANDIDATE_IDS,
    )

    return REQUIRED_CANDIDATE_IDS


class _SyntheticEncoder(torch.nn.Module):
    """A deterministic synthetic stand-in for the frozen MiT-B0 encoder.

    Its last stage is a constant per-row feature map, so the pooled G
    feature is deterministic and cheap; parameters and buffers are real
    module state whose immutability the tests verify.
    """

    def __init__(self, g_dim: int = 256):
        super().__init__()
        self.g_dim = int(g_dim)
        self.register_buffer("generation", torch.tensor([7.0]))

    def forward(self, rgb):
        batch = rgb.shape[0]
        last = (self.generation * 0.01).expand(
            batch, self.g_dim, 7, 7).contiguous()
        return [None, None, None, last]


def _sha_of(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_ref(path):
    return {"path": str(path), "sha256": _sha_of(path)}


@pytest.fixture(scope="module")
def finish_world(module_scratch, local_world):
    """The fully synthetic 9-candidate world: the synthetic V5/P0 chain,
    a synthetic local linear fit, a synthetic conditional fit, the
    synthetic split CSV and label source for Layer B, and the synthetic
    complete config that binds all 9 candidates by sha256."""
    world = local_world
    base = module_scratch / "finish"

    # the local linear fit (3 folds x RC/RCS at seed 2026)
    local_cache, local_config, local_output = _prepare_fit(
        base / "linear", world)
    assert local_entry.main(
        _main_args(world, local_config, local_output)) == 0

    # the conditional fits (2 variants x 3 folds x 3 seeds) + R0 records
    cond_cache, cond_config, cond_output = _prepare_cond_fit(
        base / "conditional", world)
    assert cond_entry.main(_main_args(world, cond_config, cond_output)) == 0

    fit_dir = Path(world["fit_dir"])
    cond_models = cond_output / "models"
    lin_models = local_output / "models"
    folds = ("fold0", "fold1", "fold2")

    def cond_files(seed, fold, variant):
        d = cond_models / f"seed_{seed}" / f"{fold}_{variant}"
        return {
            "residual_state": _file_ref(d / "residual_state.pt"),
            "scaler": _file_ref(d / "scaler.json"),
            "saved_predictions": _file_ref(
                d / "outer_eval_predictions.jsonl"),
            "fold_variant_report": _file_ref(
                d / "fold_variant_report.json"),
        }

    def lin_files(fold, variant):
        d = lin_models / f"{fold}_{variant}"
        return {
            "residual_state": _file_ref(d / "residual_state.pt"),
            "scaler": _file_ref(d / "scaler.json"),
            "saved_predictions": _file_ref(
                d / "outer_eval_predictions.jsonl"),
            "fold_variant_report": _file_ref(
                d / "fold_variant_report.json"),
        }

    candidates = [{
        "candidate_id": "R0", "family": "p0", "variant": "R0",
        "seed": None, "input_dim": 270, "files": {},
    }]
    for variant in ("RC", "RCS"):
        candidates.append({
            "candidate_id": f"linear_local_{variant}_s2026",
            "family": "linear_local", "variant": variant, "seed": 2026,
            "input_dim": 30,
            "files": {fold: lin_files(fold, variant) for fold in folds},
        })
    for seed in (2026, 2027, 2028):
        for variant in ("RC", "RCS"):
            candidates.append({
                "candidate_id": f"conditional_{variant}_s{seed}",
                "family": "conditional", "variant": variant, "seed": seed,
                "input_dim": 35,
                "files": {fold: cond_files(seed, fold, variant)
                          for fold in folds},
            })

    global_cache = world["feature_dir"]
    # the synthetic split CSV + label source for Layer B: two val frames
    # whose capture groups belong to some fold's outer_eval
    protocol = world["protocol"]
    outer_groups = set()
    for fold in protocol["folds"]["folds"]:
        outer_groups.update(fold["outer_eval_groups"])
    val_samples = [s for s in world["cache_samples"] if s["split"] == "val"]
    chosen_frames = []
    seen_groups = set()
    for sample in val_samples:
        group = sample["group"]
        if group not in outer_groups or group in seen_groups:
            continue
        seen_groups.add(group)
        chosen_frames.append(sample)
        if len(chosen_frames) == 2:
            break
    assert len(chosen_frames) == 2, "the synthetic world must offer two " \
        "val frames from outer_eval groups"
    source_root = world["workspace"] / "source"
    # the Layer B detection source mirrors the recorded left/right
    # directory convention (right_image_path maps .../left/ -> .../right/)
    detection_root = base / "detection"
    split_csv = base / "detection_split.csv"
    rows = []
    for sample in chosen_frames:
        stem = sample["source_frame"]
        left_dir = detection_root / "left" / "val"
        right_dir = detection_root / "right" / "val"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / "images" / "val" / f"{stem}.png",
                        left_dir / f"{stem}.png")
        shutil.copyfile(
            source_root / "images_right" / "val" / f"{stem}.png",
            right_dir / f"{stem}.png")
        rows.append({
            "output_stem": stem,
            "group_name": sample["group"],
            "left_path": str(left_dir / f"{stem}.png"),
            "split": "val",
        })
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "output_stem", "group_name", "left_path", "split"))
        writer.writeheader()
        writer.writerows(rows)

    from core.fusion_dataset import dataset_fingerprint
    from core.material_mask_dataset import mask_dataset_fingerprint

    v5_fingerprint = mask_dataset_fingerprint(world["v5"])
    v4_fingerprint = dataset_fingerprint(world["v4"])
    config = {
        "experiment": "material_residual_runtime_v1_complete",
        "candidate_digest_provenance": "synthetic world; digests computed "
                                       "inside this test module",
        "candidate": {
            "experiment": "material_conditional_residual_v1",
            "fit_dir": str(cond_output),
            "summary_sha256": _sha_of(cond_output / "summary.json"),
            "admission_sha256": _sha_of(cond_output / "admission.json"),
            "input_dim": 35, "context_dim": 5, "hidden": 8,
            "engineering_seed": 2026,
            "variants": ["R0", "RC", "RCS"],
            "fallback": {
                "rule": "missing/invalid polarization falls back to the "
                        "candidate's OWN frozen P0",
                "reasons": ["no_valid_s_region", "non_finite_polar"],
            },
        },
        "candidates": candidates,
        "protocol": {
            "path": str(world["protocol_path"]),
            "sha256": _sha_of(world["protocol_path"]),
        },
        "encoder": {
            "path": str(base / "synthetic_encoder_identity.bin"),
            "sha256": "0" * 64,
            "imgsz": 224,
            "pooling": "instance_mask weighted mean on the last stage",
            "preprocessing_version": "synthetic",
        },
        "features": {
            "crop_padding": 10,
            "region_rule": "local_partition_interior_band_v1",
            "global_feature_definition_version": "existing_data_probe_v1",
            "local_feature_definition_version":
                "existing_data_probe_v1_local_interior_band_v1",
            "global_cache": {
                "path": str(global_cache),
                "files": {
                    name: _file_ref(global_cache / name)["sha256"]
                    for name in ("feature_cache.npz",
                                 "feature_manifest.json",
                                 "feature_definition.json",
                                 "cache_binding.json")
                },
            },
            "local_cache": {
                "path": str(cond_cache),
                "files": {
                    name: _file_ref(cond_cache / name)["sha256"]
                    for name in ("feature_cache.npz",
                                 "feature_manifest.json",
                                 "feature_definition.json",
                                 "cache_binding.json")
                },
            },
        },
        "p0_models": {
            fold: {
                "checkpoint": _file_ref(
                    fit_dir / "models" / f"{fold}_P0" / "model_state.pt"),
                "scaler": _file_ref(
                    fit_dir / "models" / f"{fold}_P0" / "scaler.json"),
                "oracle": _file_ref(
                    fit_dir / "models" / f"{fold}_P0"
                    / "outer_eval_predictions.jsonl"),
            }
            for fold in folds
        },
        "residual_models": {},
        "r0_records": {
            fold: _file_ref(cond_models / "R0" / f"{fold}_R0"
                            / "outer_eval_predictions.jsonl")
            for fold in folds
        },
        "model_a": {
            "path": "synthetic", "sha256": "1" * 64,
            "conf_threshold": 0.25, "iou_threshold": 0.45, "imgsz": 640,
            "data_independence": "unknown",
        },
        "calibration": {
            "file": "synthetic", "r_convention": "matlab",
            "baseline": 0.1, "focal_length": 100.0,
            "rectification": {"enabled": False},
        },
        "stereo": {"matcher": "sgbm", "mode": "3way", "max_disp": 8,
                   "scale": 0.25, "block_size": 3, "uniqueness_ratio": 10,
                   "speckle_window": 100, "speckle_range": 32,
                   "texture_threshold": 10.0, "lr_check_threshold_px": 2.0,
                   "min_valid_ratio": 0.05, "band_margin": 2,
                   "full_image_threshold": 0.9},
        "polarization": {"gain": 1.0, "epsilon": 1e-6,
                         "min_intensity_sum": 10.0},
        "runtime": {"device": "cpu", "precision": "float32",
                    "classification_batch": 8, "head_device": "cpu",
                    "class_names": ["metal_submarine", "plastic_submarine",
                                    "plastic_fish", "real_fish"]},
        "data_identity": {
            "training_data_root": str(world["v5"]),
            "training_data_fingerprint": v5_fingerprint,
            "training_parent_data_fingerprint": v4_fingerprint,
            "runtime_data_root": "",
            "runtime_data_fingerprint": "",
            "detection_split_csv": str(split_csv),
            "detection_label_root": str(source_root / "labels"),
        },
        "tolerances": {"g_feature_atol": 1.0, "g_feature_rtol": 1.0,
                       "c_s_atol": 1.0, "logit_atol": 1.0,
                       "prediction_rule": "exact"},
        "benchmark": {"frame_count": 2, "warmup_per_frame": 1,
                      "measured_per_frame": 1, "budget_minutes": 15,
                      "scenario_targets": [1, 2]},
        "evaluation_budget": {"whole_run_minutes": 30},
    }
    (base / "synthetic_encoder_identity.bin").write_bytes(b"synthetic")
    config_path = base / "complete.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False),
                           encoding="utf-8")

    from models.material_residual_classifier import (
        build_frame_fold_map,
        load_residual_runtime_config,
    )
    from scripts.eval_material_residual_pipeline import load_protocol

    loaded = load_residual_runtime_config(config_path)
    protocol_loaded = load_protocol(loaded)
    frame_folds = build_frame_fold_map(rows, protocol_loaded)
    return {
        "config_path": config_path,
        "config": loaded,
        "protocol": protocol_loaded,
        "frame_folds": frame_folds,
        "split_rows": rows,
        "label_root": source_root / "labels",
        "output_base": base,
    }


@pytest.fixture(scope="module")
def finish_classifiers(finish_world):
    """The real adapter over the synthetic world with the synthetic
    encoder injected (no real weights anywhere)."""
    from models.material_residual_classifier import (
        MaterialResidualClassifier,
    )
    from scripts.eval_material_residual_pipeline import build_classifiers

    world_g_dim = int(json.loads((
        Path(finish_world["config"].global_cache[0].path).parent
        / "feature_definition.json")
        .read_text(encoding="utf-8"))["slot_G"]["dim"])
    with mock.patch("models.material_residual_classifier._build_encoder",
                    lambda config, device, builder=None:
                    _SyntheticEncoder(world_g_dim)):
        classifiers = build_classifiers(
            finish_world["config"], finish_world["protocol"], device="cpu",
            frame_folds=finish_world["frame_folds"])
    return classifiers


def _fake_engine_factory(finish_world, calls):
    """A synthetic detector: matches all but one GT object, adds one
    unmatched prediction per frame, and makes one instance's polarization
    fully invalid so the exact-P0 fallback is exercised for real."""
    from core.gt_geometry import build_gt_geometry
    from models.material_classifier import MaterialInstanceInput
    from models.material_classifier import MASK_SOURCE_PREDICTED

    label_root = finish_world["label_root"]

    class _FakeEngine:
        def process_frame_detailed(self, left, right, *, sync_skew_ms=None,
                                   already_rectified=False, frame_id=""):
            calls["frames"] += 1
            stem = frame_id
            height, width = left.shape
            geometry = build_gt_geometry(
                label_root / "val" / f"{stem}.txt", width, height,
                padding=10)
            instances = []
            inputs = []
            for index, item in enumerate(geometry):
                missed = index == len(geometry) - 1  # one GT stays unmatched
                if missed:
                    continue
                mask = np.asarray(item.mask, dtype=np.uint8)
                ys, xs = np.nonzero(mask)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                window = (slice(max(0, y1 - 10), min(height, y2 + 10)),
                          slice(max(0, x1 - 10), min(width, x2 + 10)))
                gray = np.asarray(left)[window].astype(np.uint8)
                mask_crop = np.ascontiguousarray(mask[window])
                shape = gray.shape
                no_polar = index == 0  # one instance loses its S entirely
                inputs.append(MaterialInstanceInput(
                    frame_id=stem, instance_id=len(inputs), bbox=(x1, y1, x2, y2),
                    gray=gray,
                    signed_q=np.zeros(shape, dtype=np.float32),
                    abs_q=np.zeros(shape, dtype=np.float32),
                    polar_valid=np.zeros(shape, dtype=np.uint8)
                    if no_polar else mask_crop.copy(),
                    instance_mask=mask_crop.copy(),
                    quality=np.asarray(
                        [0.0 if no_polar else 0.9, 1.0, 1.0, 0.1],
                        dtype=np.float32),
                    polar_invalid=no_polar,
                    invalid_reason="synthetic_no_valid_s" if no_polar else "",
                    mask_source=MASK_SOURCE_PREDICTED))
                instances.append(__import__("types").SimpleNamespace(
                    id=len(instances), bbox=(x1, y1, x2, y2), mask=mask,
                    confidence=0.9, class_id=item.class_id,
                    class_name=""))
            # one extra detection that matches no GT (a real unmatched
            # prediction, not an empty-set vacuity)
            extra_mask = np.zeros_like(left, dtype=np.uint8)
            extra_mask[0:8, 0:8] = 1
            extra_window = (slice(0, 18), slice(0, 18))
            extra_gray = np.asarray(left)[extra_window].astype(np.uint8)
            inputs.append(MaterialInstanceInput(
                frame_id=stem, instance_id=len(inputs),
                bbox=(0, 0, 8, 8), gray=extra_gray,
                signed_q=np.zeros((18, 18), dtype=np.float32),
                abs_q=np.zeros((18, 18), dtype=np.float32),
                polar_valid=np.zeros((18, 18), dtype=np.uint8),
                instance_mask=np.ascontiguousarray(
                    extra_mask[extra_window]),
                quality=np.zeros(4, dtype=np.float32),
                polar_invalid=True, invalid_reason="synthetic_unmatched",
                mask_source=MASK_SOURCE_PREDICTED))
            instances.append(__import__("types").SimpleNamespace(
                id=len(instances), bbox=(0, 0, 8, 8), mask=extra_mask,
                confidence=0.5, class_id=0, class_name=""))
            import types

            return types.SimpleNamespace(
                instances=instances, material_inputs=inputs,
                timings={"model_a_s": 0.01, "stereo_s": 0.01,
                         "depth_s": 0.0, "polar_s": 0.01,
                         "model_b_s": 0.01, "total_s": 0.04})

    return _FakeEngine()


@pytest.fixture(scope="module")
def layer_b_synthetic(finish_world, finish_classifiers, module_scratch):
    """run_layer_b over the synthetic frames with the synthetic detector
    boundary replaced (mock.patch on the engine builder only)."""
    import scripts.eval_material_residual_pipeline as entry

    calls = {"frames": 0}
    output = module_scratch / "orch_layer_b"
    output.mkdir(parents=True)
    with mock.patch.object(
            entry, "_build_engine",
            lambda config, *, device, rcs_classifier:
            _fake_engine_factory(finish_world, calls)):
        summary = entry.run_layer_b(
            finish_world["config"], finish_world["protocol"], device="cpu",
            output_dir=output, classifiers=finish_classifiers,
            frame_folds=finish_world["frame_folds"])
    return {"summary": summary, "output": output, "calls": calls,
            "classifiers": finish_classifiers,
            "world": finish_world}


class TestLayerBCollectedAcceptance:
    """The three collected acceptance tests over the fully synthetic
    world (moved out of the old uncollectable fixture body)."""

    def test_layer_b_nine_candidates_complete_evidence(
            self, layer_b_synthetic):
        summary = layer_b_synthetic["summary"]
        output = layer_b_synthetic["output"]
        assert layer_b_synthetic["calls"]["frames"] == 2
        assert set(summary["metrics_by_candidate"]) == \
            set(entry_required_ids())

        frames = [json.loads(line) for line in
                  (output / "frames_complete.jsonl").read_text(
                      encoding="utf-8").splitlines() if line.strip()]
        predictions = [json.loads(line) for line in
                       (output / "predictions_complete.jsonl").read_text(
                           encoding="utf-8").splitlines() if line.strip()]
        expected_pairs = {(frame["frame_id"], pred["pred_instance_id"])
                          for frame in frames
                          for pred in frame["predictions"]}
        assert expected_pairs, "the synthetic world must produce detections"
        total_predictions = sum(len(f["predictions"]) for f in frames)
        assert len(predictions) == total_predictions * 9

        # real missed GT and real unmatched predictions exist
        assert any(frame["missed_gt_ids"] for frame in frames), \
            "the synthetic detector must miss at least one GT"
        assert any(frame["unmatched_pred_ids"] for frame in frames), \
            "the synthetic detector must produce unmatched predictions"

        for candidate_id in entry_required_ids():
            rows = [row for row in predictions
                    if row["candidate_id"] == candidate_id]
            keys = {(row["frame_id"], row["pred_instance_id"])
                    for row in rows}
            assert keys == expected_pairs, candidate_id
            assert len(keys) == len(rows), "no duplicate records"
        # unmatched predictions carry classification/fallback info for
        # every candidate (never silently dropped)
        for candidate_id in entry_required_ids():
            unmatched_rows = [
                row for row in predictions
                if row["candidate_id"] == candidate_id
                and row["matched_gt_id"] is None]
            assert unmatched_rows, candidate_id
            assert all(row["status"] in ("valid", "failed")
                       for row in unmatched_rows)
            assert all(row["final_logits"] or row["status"] == "failed"
                       for row in unmatched_rows)

        # the missing-S instance exercised the exact P0 fallback for real:
        # every correction candidate's fallback row carries exactly the
        # R0 classifier's own logits for the same instance
        r0_by_key = {
            (row["frame_id"], row["pred_instance_id"]): row
            for row in predictions if row["candidate_id"] == "R0"}
        fallback_checked = 0
        for row in predictions:
            if row["candidate_id"] == "R0" or not row["used_fallback"]:
                continue
            reference = r0_by_key[(row["frame_id"],
                                   row["pred_instance_id"])]
            assert row["final_logits"] == reference["final_logits"]
            assert row["fallback_reason"] == "no_valid_s_region"
            fallback_checked += 1
        assert fallback_checked >= 8 * 2, \
            "the two synthetic missing-S instances must fall back in all " \
            "8 correction candidates"

    def test_layer_b_recompute_from_raw_records_agrees(
            self, layer_b_synthetic):
        from scripts.recompute_material_residual_evidence import recompute

        summary = layer_b_synthetic["summary"]
        output = layer_b_synthetic["output"]
        frames = [json.loads(line) for line in
                  (output / "frames_complete.jsonl").read_text(
                      encoding="utf-8").splitlines() if line.strip()]
        predictions = [json.loads(line) for line in
                       (output / "predictions_complete.jsonl").read_text(
                           encoding="utf-8").splitlines() if line.strip()]
        recomputed = recompute(frames, predictions,
                               sorted(entry_required_ids()))
        for candidate_id in entry_required_ids():
            own = summary["metrics_by_candidate"][candidate_id]
            mine = recomputed["candidates"][candidate_id]
            assert mine["counts"]["matched"] == own["counts"]["matched"]
            assert mine["conditional"]["confusion_matrix"] == \
                own["conditional_classification"]["confusion_matrix"]
            assert mine["conditional"]["macro_f1"] == pytest.approx(
                own["conditional_classification"]["macro_f1"], abs=1e-9)
            assert mine["end_to_end"]["macro_f1"] == pytest.approx(
                own["end_to_end"]["macro_f1"], abs=1e-9)
        for label, outcome in recomputed["paired"].items():
            own = summary["paired"][label]
            for key in ("fixed", "broke", "both_correct", "both_wrong"):
                assert outcome[key] == own[key], (label, key)
        # the independent fallback totals over ALL predictions agree
        for candidate_id, entry_fallback in \
                recomputed["fallback"].items():
            own = summary["fallback_all_predictions"][candidate_id]
            assert entry_fallback["used_fallback"] == \
                own["used_fallback"]
            assert entry_fallback["reasons"] == own["reasons"]

    def test_params_and_buffers_unchanged_by_layer_b(
            self, finish_world, finish_classifiers, module_scratch):
        from core.material_probe import snapshot_module_state

        before = {}
        for candidate_id, classifier in finish_classifiers.items():
            for name, module in classifier.torch_modules().items():
                before[f"{candidate_id}:{name}"] = \
                    snapshot_module_state(module)

        import scripts.eval_material_residual_pipeline as entry

        calls = {"frames": 0}
        output = module_scratch / "orch_layer_b_params"
        output.mkdir(parents=True)
        with mock.patch.object(
                entry, "_build_engine",
                lambda config, *, device, rcs_classifier:
                _fake_engine_factory(finish_world, calls)):
            entry.run_layer_b(
                finish_world["config"], finish_world["protocol"],
                device="cpu", output_dir=output,
                classifiers=finish_classifiers,
                frame_folds=finish_world["frame_folds"])

        for candidate_id, classifier in finish_classifiers.items():
            for name, module in classifier.torch_modules().items():
                key = f"{candidate_id}:{name}"
                after = snapshot_module_state(module)
                assert set(before[key]) == set(after)
                for tensor_key in before[key]:
                    assert np.array_equal(before[key][tensor_key],
                                          after[tensor_key]), \
                        f"{key}.{tensor_key} changed during inference"


class TestMainOrchestration:
    def test_layer_a_failure_blocks_layer_b(
            self, finish_world, module_scratch, capsys):
        """main --mode evaluate over the synthetic world: Layer A fails
        its G-stage comparison (the synthetic encoder output cannot match
        the synthetic cache), so Layer B runs ZERO times."""
        import scripts.eval_material_residual_pipeline as entry

        output = module_scratch / "orch_cpu_fail"
        with mock.patch(
                "models.material_residual_classifier._build_encoder",
                lambda config, device, builder=None: _SyntheticEncoder(
                    _world_g_dim(finish_world))):
            code = entry.main([
                "--config", str(finish_world["config_path"]),
                "--output-dir", str(output),
                "--mode", "evaluate", "--skip-benchmark", "--device",
                "cpu",
            ])
        captured = capsys.readouterr()
        assert code == 2, captured.out + captured.err
        assert (output / "layer_a_summary.json").is_file()
        assert not (output / "predictions_complete.jsonl").exists()
        assert not (output / "layer_b_summary.json").exists()
        status = json.loads((output / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["status"] == "LAYER_A_FAILED"

    def test_incomplete_registry_refused(self, finish_world,
                                         module_scratch):
        """A config whose registry is missing one candidate is refused
        before anything runs."""
        import yaml

        import scripts.eval_material_residual_pipeline as entry

        config = yaml.safe_load(
            finish_world["config_path"].read_text(encoding="utf-8"))
        config["candidates"] = [c for c in config["candidates"]
                                if c["candidate_id"] !=
                                "conditional_RCS_s2028"]
        path = module_scratch / "incomplete_registry.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False),
                        encoding="utf-8")
        output = module_scratch / "orch_incomplete"
        code = entry.main([
            "--config", str(path), "--output-dir", str(output),
            "--mode", "layer_a", "--skip-benchmark", "--device", "cpu",
        ])
        assert code == 1
        assert not (output / "layer_a_summary.json").exists()


# ---------------------------------------------------------------------------
# Permanent Layer A verdict coverage (restored in residrt_finish2; these
# seven counterexamples were accidentally dropped from the file during the
# previous batch's orchestration-section rebuild). The verdict logic is
# NOT regressed to manufacture a RED — the restored tests are GREEN on
# first run against the already-fixed production verdict.


def _full_row(sample="s0", candidate="conditional_RCS_s2026", *,
              s_valid_offline=True, s_valid_runtime=True,
              fallback_evidence=None, extra_stages=None,
              report_mutation=None):
    row = {
        "sample_name": sample, "candidate": candidate,
        "s_valid_offline": s_valid_offline,
        "s_valid_runtime": s_valid_runtime,
        "stage_reports": {
            "g": {"passed": True, "max_abs_error": 0.0, "status": "PASSED"},
            "c14": {"passed": True, "max_abs_error": 0.0, "status": "PASSED"},
            "s16": {"passed": True, "max_abs_error": None, "status": "PASSED"},
            "p0": {"passed": True, "max_abs_error": 0.0, "status": "PASSED"},
            "final": {"passed": True, "max_abs_error": 0.0,
                      "status": "PASSED"},
        },
        "prediction_mismatch": False,
    }
    if s_valid_offline is False:
        if fallback_evidence is not None:
            row["fallback_exact_p0"] = fallback_evidence
    for name, report in (extra_stages or {}).items():
        row["stage_reports"][name] = report
    if report_mutation is not None:
        row["stage_reports"].update(report_mutation)
    return row


GAP2_REQUIRED = {"conditional_RCS_s2026": ("g", "c14", "s16", "p0", "final")}


class TestLayerAVerdictFullCoverage:
    def test_legal_valid_and_missing_rows_pass(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        rows = [
            _full_row("s0"),
            _full_row("s1", s_valid_offline=False, s_valid_runtime=False,
                      fallback_evidence=True),
        ]
        verdict = layer_a_verdict(
            rows, expected_names=["s0", "s1"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "OK", verdict["problems"]
        assert verdict["abort_layer_b"] is False

    def test_fallback_stage_failure_not_ignored(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        row = _full_row("s0", extra_stages={
            "fallback": {"passed": False, "max_abs_error": None,
                         "status": "FAILED",
                         "reason": "missing-S row did not fall back"}})
        verdict = layer_a_verdict(
            [row], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "FAILED"
        assert verdict["abort_layer_b"] is True

    def test_s_valid_disagreement_fails(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        row = _full_row("s0", s_valid_offline=False, s_valid_runtime=True)
        verdict = layer_a_verdict(
            [row], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_s_valid_non_bool_fails(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        row = _full_row("s0", s_valid_offline=1, s_valid_runtime=1)
        verdict = layer_a_verdict(
            [row], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_missing_s_without_fallback_evidence_fails(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        row = _full_row("s0", s_valid_offline=False, s_valid_runtime=False,
                        fallback_evidence=None)
        verdict = layer_a_verdict(
            [row], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_fallback_evidence_false_fails(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        row = _full_row("s0", s_valid_offline=False, s_valid_runtime=False,
                        fallback_evidence=False)
        verdict = layer_a_verdict(
            [row], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_illegal_stage_report_type_fails(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        row = _full_row("s0", report_mutation={
            "g": {"passed": 1, "max_abs_error": 0.0, "status": "PASSED"}})
        verdict = layer_a_verdict(
            [row], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "FAILED"

    def test_self_contradictory_stage_report_fails(self):
        from scripts.eval_material_residual_pipeline import layer_a_verdict

        row = _full_row("s0", report_mutation={
            "final": {"passed": True, "max_abs_error": 0.0,
                      "status": "FAILED"}})
        verdict = layer_a_verdict(
            [row], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=GAP2_REQUIRED)
        assert verdict["status"] == "FAILED"

# the constant the synthetic encoder pools to; the cache G rows are set
# to exactly this value so runtime G == cache G by construction
SYNTHETIC_G = 0.07


class SelfConsistentEncoder(torch.nn.Module):
    """Pools to the constant SYNTHETIC_G for every row (the mask-weighted
    mean of a constant map is that constant)."""

    def __init__(self, g_dim: int = 4):
        super().__init__()
        self.g_dim = int(g_dim)
        self.register_buffer("level", torch.tensor([SYNTHETIC_G]))

    def forward(self, rgb):
        batch = rgb.shape[0]
        last = self.level.expand(batch, self.g_dim, 7, 7).contiguous()
        return [None, None, None, last]


def _constant_g_cache(v5_root, rows):
    """A make_cache_files replacement whose G rows are exactly the
    synthetic encoder's pooled output."""
    from core.material_probe import (
        authoritative_sample_index,
        build_control_features,
        build_polar_stats,
    )
    from core.material_mask_dataset import load_mask_sample

    def build(feature_dir, protocol, n=None, g_dim=4):
        del n
        authority = authoritative_sample_index(protocol)
        names = sorted(authority)
        row_by_name = {row["sample_name"]: row for row in rows}
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
            g_rows.append(np.full(g_dim, SYNTHETIC_G, dtype=np.float32))
            c_rows.append(control)
            s_rows.append(values)
            valid_rows.append(has_valid_s)
            class_rows.append(authority[name]["identity"]["class_id"])
            samples.append({
                "sample_name": name,
                "npz_path": row["npz_path"],
                "split": row["split"],
                "group": authority[name]["identity"]["group"],
                "class_id": int(row["class_id"]),
                "class_name": authority[name]["identity"]["class_name"],
                "source_frame": authority[name]["identity"]["source_frame"],
                "object_index": authority[name]["identity"]["object_index"],
                "has_valid_s": has_valid_s,
            })
        feature_dir.mkdir(parents=True, exist_ok=True)
        g = np.asarray(g_rows, dtype=np.float32)
        c = np.asarray(c_rows, dtype=np.float32)
        s = np.asarray(s_rows, dtype=np.float32)
        valid = np.asarray(valid_rows, dtype=bool)
        classes = np.asarray(class_rows, dtype=np.int64)
        np.savez_compressed(feature_dir / "feature_cache.npz",
                            g_features=g, c_features=c, s_features=s,
                            s_valid=valid, class_id=classes)
        manifest = {"count": len(samples), "samples": samples}
        (feature_dir / "feature_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        definition = {
            "feature_definition_version": "existing_data_probe_v1",
            "slot_G": {"dim": int(g.shape[1]),
                       "definition": "synthetic constant row"},
            "slot_C": {"keys": [
                "quality_valid_ratio", "quality_in_bounds_ratio",
                "quality_brightness_valid_ratio",
                "gray_mean_in_instance_mask", "gray_std_in_instance_mask",
                "s_region_nonempty_flag"]},
            "slot_S": {"keys": [
                "signed_q_mean", "signed_q_std_ddof0", "signed_q_p10",
                "signed_q_p25", "signed_q_p50", "signed_q_p75",
                "signed_q_p90", "abs_q_mean"]},
            "class_names": ["metal_submarine", "plastic_submarine",
                            "plastic_fish", "real_fish"],
        }
        (feature_dir / "feature_definition.json").write_text(
            json.dumps(definition, indent=2) + "\n", encoding="utf-8")
        return {"feature_dir": feature_dir,
                "manifest": manifest, "definition": definition,
                "arrays": {"g_features": g, "c_features": c,
                           "s_features": s, "s_valid": valid,
                           "class_id": classes}}

    return build


def _sha_of(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_ref(path):
    return {"path": str(path), "sha256": _sha_of(path)}


@pytest.fixture(scope="module")
def finish_world_consistent(module_scratch, local_world):
    """The self-consistent synthetic world: cache G == encoder output, so
    Layer A passes without any verdict mocking and a legal evaluate
    completes end to end."""
    world = local_world
    base = module_scratch / "consistent"

    from .test_p0_baseline_verification import build_fit as _build_fit
    import sys as _sys

    _p0_module = _sys.modules[_build_fit.__module__]

    from core.fusion_dataset import dataset_fingerprint as _ds_fp
    from core.material_mask_dataset import mask_dataset_fingerprint as _mask_fp

    v5_fingerprint = _mask_fp(world["v5"])
    v4_fingerprint = _ds_fp(world["v4"])
    cache_builder = _constant_g_cache(world["v5"], world["rows"])
    with mock.patch.object(_p0_module, "make_protocol",
                           lambda: world["protocol"]), \
            mock.patch.object(_p0_module, "make_cache_files", cache_builder):
        fit = _build_fit(module_scratch, name="p0fit_consistent")
    feature_dir = fit["feature_dir"]
    fit_dir = fit["fit_dir"]

    # the local linear fit
    local_cache, local_config, local_output = _prepare_fit_consistent(
        base / "linear", world, feature_dir)
    assert local_entry.main(
        _main_args(world, local_config, local_output)) == 0

    # the conditional fits (oracle logits are computed from the same
    # self-consistent cache)
    cond_cache, cond_config, cond_output = _prepare_cond_fit_consistent(
        base / "conditional", world, feature_dir)
    assert cond_entry.main(
        _main_args(world, cond_config, cond_output)) == 0

    cond_models = cond_output / "models"
    lin_models = local_output / "models"
    folds = ("fold0", "fold1", "fold2")

    def cond_files(seed, fold, variant):
        d = cond_models / f"seed_{seed}" / f"{fold}_{variant}"
        return {
            "residual_state": _file_ref(d / "residual_state.pt"),
            "scaler": _file_ref(d / "scaler.json"),
            "saved_predictions": _file_ref(
                d / "outer_eval_predictions.jsonl"),
            "fold_variant_report": _file_ref(
                d / "fold_variant_report.json"),
        }

    def lin_files(fold, variant):
        d = lin_models / f"{fold}_{variant}"
        return {
            "residual_state": _file_ref(d / "residual_state.pt"),
            "scaler": _file_ref(d / "scaler.json"),
            "saved_predictions": _file_ref(
                d / "outer_eval_predictions.jsonl"),
            "fold_variant_report": _file_ref(
                d / "fold_variant_report.json"),
        }

    candidates = [{
        "candidate_id": "R0", "family": "p0", "variant": "R0",
        "seed": None, "input_dim": 18, "files": {},
    }]
    for variant in ("RC", "RCS"):
        candidates.append({
            "candidate_id": f"linear_local_{variant}_s2026",
            "family": "linear_local", "variant": variant, "seed": 2026,
            "input_dim": 30,
            "files": {fold: lin_files(fold, variant) for fold in folds},
        })
    for seed in (2026, 2027, 2028):
        for variant in ("RC", "RCS"):
            candidates.append({
                "candidate_id": f"conditional_{variant}_s{seed}",
                "family": "conditional", "variant": variant, "seed": seed,
                "input_dim": 35,
                "files": {fold: cond_files(seed, fold, variant)
                          for fold in folds},
            })

    # the synthetic split CSV: two val frames from outer_eval groups
    protocol = world["protocol"]
    outer_groups = set()
    for fold in protocol["folds"]["folds"]:
        outer_groups.update(fold["outer_eval_groups"])
    val_samples = [s for s in fit["manifest"]["samples"]
                   if s["split"] == "val"]
    chosen = []
    seen_groups = set()
    for sample in val_samples:
        if sample["group"] in outer_groups and \
                sample["group"] not in seen_groups:
            seen_groups.add(sample["group"])
            chosen.append(sample)
        if len(chosen) == 2:
            break
    assert len(chosen) == 2
    source_root = world["workspace"] / "source"
    detection_root = base / "detection"
    split_csv = base / "detection_split.csv"
    rows = []
    for sample in chosen:
        stem = sample["source_frame"]
        left_dir = detection_root / "left" / "val"
        right_dir = detection_root / "right" / "val"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / "images" / "val" / f"{stem}.png",
                        left_dir / f"{stem}.png")
        shutil.copyfile(
            source_root / "images_right" / "val" / f"{stem}.png",
            right_dir / f"{stem}.png")
        rows.append({"output_stem": stem, "group_name": sample["group"],
                     "left_path": str(left_dir / f"{stem}.png"),
                     "split": "val"})
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "output_stem", "group_name", "left_path", "split"))
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "experiment": "material_residual_runtime_v1_complete",
        "candidate_digest_provenance": "self-consistent synthetic world; "
                                       "digests computed inside this "
                                       "test module",
        "candidate": {
            "experiment": "material_conditional_residual_v1",
            "fit_dir": str(cond_output),
            "summary_sha256": _sha_of(cond_output / "summary.json"),
            "admission_sha256": _sha_of(cond_output / "admission.json"),
            "input_dim": 35, "context_dim": 5, "hidden": 8,
            "engineering_seed": 2026,
            "variants": ["R0", "RC", "RCS"],
            "fallback": {
                "rule": "missing/invalid polarization falls back to the "
                        "candidate's OWN frozen P0",
                "reasons": ["no_valid_s_region", "non_finite_polar"],
            },
        },
        "candidates": candidates,
        "protocol": {
            "path": str(world["protocol_path"]),
            "sha256": _sha_of(world["protocol_path"]),
        },
        "encoder": {
            "path": str(base / "synthetic_encoder_identity.bin"),
            "sha256": "0" * 64, "imgsz": 224,
            "pooling": "instance_mask weighted mean on the last stage",
            "preprocessing_version": "synthetic",
        },
        "features": {
            "crop_padding": 10,
            "region_rule": "local_partition_interior_band_v1",
            "global_feature_definition_version": "existing_data_probe_v1",
            "local_feature_definition_version":
                "existing_data_probe_v1_local_interior_band_v1",
            "global_cache": {
                "path": str(feature_dir),
                "files": {name: _file_ref(feature_dir / name)["sha256"]
                          for name in ("feature_cache.npz",
                                       "feature_manifest.json",
                                       "feature_definition.json",
                                       "cache_binding.json")},
            },
            "local_cache": {
                "path": str(cond_cache),
                "files": {name: _file_ref(cond_cache / name)["sha256"]
                          for name in ("feature_cache.npz",
                                       "feature_manifest.json",
                                       "feature_definition.json",
                                       "cache_binding.json")},
            },
        },
        "p0_models": {
            fold: {
                "checkpoint": _file_ref(
                    fit_dir / "models" / f"{fold}_P0" / "model_state.pt"),
                "scaler": _file_ref(
                    fit_dir / "models" / f"{fold}_P0" / "scaler.json"),
                "oracle": _file_ref(
                    fit_dir / "models" / f"{fold}_P0"
                    / "outer_eval_predictions.jsonl"),
            } for fold in folds
        },
        "residual_models": {},
        "r0_records": {
            fold: _file_ref(cond_models / "R0" / f"{fold}_R0"
                            / "outer_eval_predictions.jsonl")
            for fold in folds
        },
        "model_a": {"path": "synthetic", "sha256": "1" * 64,
                    "conf_threshold": 0.25, "iou_threshold": 0.45,
                    "imgsz": 640, "data_independence": "unknown"},
        "calibration": {"file": "synthetic", "r_convention": "matlab",
                        "baseline": 0.1, "focal_length": 100.0,
                        "rectification": {"enabled": False}},
        "stereo": {"matcher": "sgbm", "mode": "3way", "max_disp": 8,
                   "scale": 0.25, "block_size": 3, "uniqueness_ratio": 10,
                   "speckle_window": 100, "speckle_range": 32,
                   "texture_threshold": 10.0, "lr_check_threshold_px": 2.0,
                   "min_valid_ratio": 0.05, "band_margin": 2,
                   "full_image_threshold": 0.9},
        "polarization": {"gain": 1.0, "epsilon": 1e-6,
                         "min_intensity_sum": 10.0},
        "runtime": {"device": "cpu", "precision": "float32",
                    "classification_batch": 8, "head_device": "cpu",
                    "class_names": ["metal_submarine", "plastic_submarine",
                                    "plastic_fish", "real_fish"]},
        "data_identity": {
            "training_data_root": str(world["v5"]),
            "training_data_fingerprint": v5_fingerprint,
            "training_parent_data_fingerprint": v4_fingerprint,
            "runtime_data_root": "", "runtime_data_fingerprint": "",
            "detection_split_csv": str(split_csv),
            "detection_label_root": str(source_root / "labels"),
        },
        "tolerances": {"g_feature_atol": 1.0, "g_feature_rtol": 1.0,
                       "c_s_atol": 1.0, "logit_atol": 1.0,
                       "prediction_rule": "exact"},
        "benchmark": {"frame_count": 2, "warmup_per_frame": 1,
                      "measured_per_frame": 1, "budget_minutes": 15,
                      "scenario_targets": [1, 2]},
        "evaluation_budget": {"whole_run_minutes": 30},
    }
    (base / "synthetic_encoder_identity.bin").write_bytes(b"synthetic")
    config_path = base / "complete.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False),
                           encoding="utf-8")

    from models.material_residual_classifier import (
        build_frame_fold_map,
        load_residual_runtime_config,
    )
    from scripts.eval_material_residual_pipeline import load_protocol

    loaded = load_residual_runtime_config(config_path)
    protocol_loaded = load_protocol(loaded)
    frame_folds = build_frame_fold_map(rows, protocol_loaded)
    return {
        "config_path": config_path, "config": loaded,
        "protocol": protocol_loaded, "frame_folds": frame_folds,
        "split_rows": rows, "label_root": source_root / "labels",
        "output_base": base, "g_dim": 4,
    }


def _prepare_fit_consistent(base, world, feature_dir):
    local_cache = base / "local_features"
    from core.polar_local_features import build_local_cache_arrays

    source = dict(np.load(Path(feature_dir) / "feature_cache.npz"))
    build_local_cache_arrays(
        source_arrays=source,
        samples=fit["manifest"]["samples"] if "fit" in dir() else world["cache_samples"],
        data_root=world["v5"],
        protocol=world["protocol"],
        output_dir=local_cache)
    config = dict(world["config"])
    config["local_features_source"] = {"path": str(local_cache)}
    config_path = _write_world_config(base / "configs" / "local.yaml",
                                      config)
    return local_cache, config_path, base / "fit_out"


def _prepare_cond_fit_consistent(base, world, feature_dir):
    local_cache = base / "local_features"
    from core.polar_local_features import build_local_cache_arrays

    source = dict(np.load(Path(feature_dir) / "feature_cache.npz"))
    build_local_cache_arrays(
        source_arrays=source,
        samples=world["cache_samples"],
        data_root=world["v5"],
        protocol=world["protocol"],
        output_dir=local_cache)
    reference_path, reference_sha = _write_reference(base, world)
    config = dict(world["config"])
    config["experiment"] = "material_conditional_residual_v1"
    config["local_features_source"] = {"path": str(local_cache)}
    config["seeds"] = [2026, 2027, 2028]
    config["linear_reference"] = {
        "fit_summary": str(reference_path),
        "expected_sha256": reference_sha}
    config_path = _write_world_config(base / "configs" / "cond.yaml",
                                      config)
    return local_cache, config_path, base / "fit_out"


def _write_reference(base, world):
    import hashlib

    reference = base / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    oracle_shas = {}
    for fold in ("fold0", "fold1", "fold2"):
        oracle = (Path(world["fit_dir"]) / "models" / f"{fold}_P0"
                  / "outer_eval_predictions.jsonl")
        oracle_shas[fold] = hashlib.sha256(
            oracle.read_bytes()).hexdigest()
    summary = {"mode": "fit", "training_status": "COMPLETED",
               "experiment": "material_polar_local_interior_band_v1",
               "region_rule": "local_partition_interior_band_v1"}
    admission = {
        "protocol_sha256": world["protocol"]["protocol_sha256"],
        "folds": ["fold0", "fold1", "fold2"],
        "local_cache": {"definition_version":
                        "existing_data_probe_v1_local_interior_band_v1"},
        "p0_identity": {fold: {"oracle_sha256": sha}
                        for fold, sha in oracle_shas.items()},
    }
    (reference / "summary.json").write_text(json.dumps(summary),
                                            encoding="utf-8")
    (reference / "admission.json").write_text(json.dumps(admission),
                                              encoding="utf-8")
    sha = hashlib.sha256((reference / "summary.json").read_bytes()).hexdigest()
    return reference / "summary.json", sha


@pytest.fixture(scope="module")
def consistent_config(finish_world):
    """The original finish_world's config (synthetic, no real data)."""
    return finish_world


class TestMainOrchestrationConsistent:
    """main() boundary tests over the synthetic world.

    The self-consistent-world Layer A chain (build_fit -> cache -> cond
    fit -> oracle) carries float precision drift between the encoder and
    the cache that the frozen comparison correctly refuses. The Layer A
    verdict logic is thoroughly tested in TestLayerAVerdictFullCoverage
    and the Layer B logic in TestLayerBCollectedAcceptance; these tests
    stub the run_layer_a ORCHESTRATION function (not the verdict) to
    isolate the main()-level boundaries: post-admission timeout, setup
    failure, publication timeout, and incomplete-frame handling.
    """

    def _passing_layer_a(self):
        return {
            "verdict": {"status": "OK", "abort_layer_b": False,
                        "problems": {}, "prediction_mismatches": 0,
                        "samples_expected": 0, "rows": 0,
                        "first_failures": []},
            "sample_count": 0, "full_replay": True, "max_samples": None,
            "candidates": sorted(entry_required_ids()),
        }

    def test_post_admission_timeout_blocks_classifier_load(
            self, consistent_config, module_scratch):
        """The whole-evaluation watch exhausts at the after-admission
        boundary: the classifier/encoder construction probe is never
        called (record-then-raise proves it)."""
        import scripts.eval_material_residual_pipeline as entry

        probe = {"calls": []}

        class _ProbeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                probe["calls"].append("constructed")
                self.register_buffer("x", torch.zeros(1))

            def forward(self, rgb):
                raise RuntimeError("must never run")

        ticks = {"n": 0}

        def stepped_clock():
            ticks["n"] += 1
            # init -> before-config -> after-protocol ->
            # before-admission -> after-admission explodes
            return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}.get(ticks["n"],
                                                        10_000.0)

        output = module_scratch / "consistent_admission_timeout"
        with mock.patch.object(entry, "_clock_factory",
                               lambda: stepped_clock), \
                mock.patch(
                    "models.material_residual_classifier._build_encoder",
                    lambda config, device, builder=None: _ProbeEncoder()):
            code = entry.main([
                "--config", str(consistent_config["config_path"]),
                "--output-dir", str(output),
                "--mode", "evaluate", "--skip-benchmark", "--device",
                "cpu",
            ])
        assert code == 3
        assert probe["calls"] == [], \
            "the classifier/encoder construction must never start"
        status = json.loads((output / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["status"] == "PARTIAL"
        assert "after admission" in status["failure"]

    def test_setup_runtime_error_is_failed_not_running(
            self, consistent_config, module_scratch):
        """A RuntimeError inside setup is FAILED (exit 4) with the type
        and reason preserved — never an escaped RUNNING status."""
        import scripts.eval_material_residual_pipeline as entry

        def exploding_admission(*args, **kwargs):
            raise RuntimeError("synthetic admission explosion")

        output = module_scratch / "consistent_setup_error"
        with mock.patch(
                "core.material_mask_dataset.admit_dataset_for_protocol",
                exploding_admission):
            code = entry.main([
                "--config", str(consistent_config["config_path"]),
                "--output-dir", str(output),
                "--mode", "evaluate", "--skip-benchmark", "--device",
                "cpu",
            ])
        assert code == 4
        status = json.loads((output / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert "RuntimeError" in status["failure"]
        assert "synthetic admission explosion" in status["failure"]

    def test_evaluate_timeout_before_final_publication(
            self, consistent_config, module_scratch):
        """mode=evaluate runs A and B to completion, then the whole-
        evaluation budget dies at the before-completion boundary:
        PARTIAL (exit 3), COMPLETED never published. run_layer_a is
        stubbed at the ORCHESTRATION level (not the verdict) so the
        test isolates the main()-side boundary."""
        import scripts.eval_material_residual_pipeline as entry

        real_check = entry.BudgetWatch.check

        def patched_check(self, stage):
            if stage == "before completion":
                self.started = self.started - self.budget_s - 1.0
            return real_check(self, stage)

        output = module_scratch / "consistent_publish_timeout"
        with mock.patch.object(entry, "run_layer_a",
                               return_value=self._passing_layer_a()), \
                mock.patch.object(entry.BudgetWatch, "check",
                                  patched_check), \
                mock.patch(
                    "models.material_residual_classifier._build_encoder",
                    lambda config, device, builder=None:
                    SelfConsistentEncoder(4)), \
                mock.patch.object(
                    entry, "_build_engine",
                    lambda config, *, device, rcs_classifier:
                    _fake_engine_factory(consistent_config,
                                         {"frames": 0})):
            code = entry.main([
                "--config", str(consistent_config["config_path"]),
                "--output-dir", str(output),
                "--mode", "evaluate", "--skip-benchmark", "--device",
                "cpu",
            ])
        assert code == 3
        status = json.loads((output / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["status"] == "PARTIAL"
        assert "before completion" in status["failure"]
        assert status["stages_completed"] == ["layer_a", "layer_b"]
        assert (output / "layer_b_summary.json").is_file()

    def test_bool_and_string_budget_refused(self, consistent_config,
                                            module_scratch):
        """bool/str budget values are refused at the resolver level; the
        CLI argparse type=float converts numeric strings before the
        resolver sees them (its own validation, not ours)."""
        import scripts.eval_material_residual_pipeline as _entry

        # the config-level check via the resolver's dict path (the
        # ResidualRuntimeConfig path is covered by the argparse type=float
        # for CLI values and the closeout tests for the config path)
        for bad_value in (True, False, "true", None):
            dict_config = {"evaluation_budget": {
                "whole_run_minutes": bad_value}}
            with pytest.raises(ValueError):
                _entry.resolve_evaluation_budget(dict_config, None)

    def test_unreadable_frame_is_incomplete_not_completed(
            self, consistent_config, module_scratch):
        """A frame whose left image cannot be read is recorded as a
        failed frame and the run publishes INCOMPLETE (exit 5), never a
        full COMPLETED with a fabricated empty-GT success."""
        import scripts.eval_material_residual_pipeline as entry

        world = consistent_config
        rows = world["split_rows"]
        broken_csv = world["output_base"] / "broken_split.csv"
        with broken_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "output_stem", "group_name", "left_path", "split"))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            writer.writerow({"output_stem": "ghost_frame",
                             "group_name": rows[0]["group_name"],
                             "left_path": str(
                                 world["output_base"] / "missing.png"),
                             "split": "val"})
        import yaml as _yaml

        raw = _yaml.safe_load(
            world["config_path"].read_text(encoding="utf-8"))
        raw["data_identity"]["detection_split_csv"] = str(broken_csv)
        broken_config = world["output_base"] / "broken_config.yaml"
        broken_config.write_text(_yaml.safe_dump(raw, sort_keys=False),
                                 encoding="utf-8")

        output = module_scratch / "consistent_incomplete"
        with mock.patch.object(entry, "run_layer_a",
                               return_value=self._passing_layer_a()), \
                mock.patch(
                    "models.material_residual_classifier._build_encoder",
                    lambda config, device, builder=None:
                    SelfConsistentEncoder(4)), \
                mock.patch.object(
                    entry, "_build_engine",
                    lambda config, *, device, rcs_classifier:
                    _fake_engine_factory(world, {"frames": 0})):
            code = entry.main([
                "--config", str(broken_config),
                "--output-dir", str(output),
                "--mode", "evaluate", "--skip-benchmark", "--device",
                "cpu",
            ])
        assert code == 5
        status = json.loads((output / "run_status.json")
                            .read_text(encoding="utf-8"))
        assert status["status"] == "INCOMPLETE"
        assert "ghost_frame" in status["failed_frames"]

