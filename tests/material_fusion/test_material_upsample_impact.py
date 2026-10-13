"""material_upsample_impact_v1 tests.

Core contract: the two upsample versions behave exactly as specified, the
legacy path is bitwise unchanged, and the evaluation entry's identity,
reproduction and reporting machinery fails loudly instead of silently
degrading. Everything is synthetic and CPU-only unless stated.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.eval_material_upsample_impact as impact
from scripts.eval_material_pipeline import _build_record, aggregate_metrics
from core.measurement_diagnostic import (
    LowResStereoCore,
    run_sgbm_lowres,
    upsample_lowres,
)
from core.stereo_matching import (
    UPSAMPLE_LEGACY_ZERO_LINEAR_V1,
    UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1,
    StereoMatcher,
    StereoMatcherConfig,
    build_horizontal_bands,
    upsample_disparity_lowres,
)

PROJECT_ROOT = Path(impact.__file__).resolve().parents[1]


def _scene(seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.integers(40, 220, (48, 96)).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, 10:] = left[:, :-10]
    return left, right


def _legacy_reference(disp_small, valid_small, out_w, out_h, scale):
    """The pre-batch final step, re-stated here from the original code so
    the legacy path is compared against an independent copy of itself."""
    disp_full = (
        __import__("cv2").resize(disp_small, (out_w, out_h),
                                 interpolation=__import__("cv2").INTER_LINEAR)
        / scale)
    valid_full = __import__("cv2").resize(
        valid_small.astype(np.uint8), (out_w, out_h),
        interpolation=__import__("cv2").INTER_NEAREST)
    disp_full = np.where(valid_full > 0, disp_full, 0.0).astype(np.float32)
    return disp_full, valid_full


def test_legacy_version_bitwise_matches_the_original_final_step():
    left, right = _scene()
    matcher = StereoMatcher(StereoMatcherConfig(
        max_disparity=32, scale=0.25, block_size=5))
    result = matcher.compute(left, right)
    core = matcher.last_lowres_cores[-1]
    ref_disp, ref_valid = _legacy_reference(
        core.disp_small, core.valid_small, left.shape[1], left.shape[0],
        matcher.config.scale)
    np.testing.assert_array_equal(result.disparity, ref_disp)
    np.testing.assert_array_equal(result.valid, ref_valid)
    # and the independent batch-1 production copy agrees bitwise too
    diag_disp, diag_valid = upsample_lowres(
        LowResStereoCore(core.disp_small, core.valid_small.astype(np.uint8),
                         float(core.valid_small.mean()), 0.0,
                         core.disp_small.shape),
        left.shape[1], left.shape[0], matcher.config.scale,
        method="production")
    np.testing.assert_array_equal(diag_disp, result.disparity)
    np.testing.assert_array_equal(diag_valid, result.valid)


def test_legacy_unchanged_at_scale_one_and_non_integer_sizes():
    for scale, (h, w) in ((1.0, (40, 60)), (0.3, (37, 53)), (0.5, (23, 41))):
        cfg = StereoMatcherConfig(max_disparity=24, scale=scale,
                                  block_size=5)
        matcher = StereoMatcher(cfg)
        full_left, full_right = _scene()
        left = np.ascontiguousarray(full_left[:h, :w])
        right = np.ascontiguousarray(full_right[:h, :w])
        result = matcher.compute(left, right)
        core = matcher.last_lowres_cores[-1]
        ref_disp, ref_valid = _legacy_reference(
            core.disp_small, core.valid_small, w, h, scale)
        np.testing.assert_array_equal(result.disparity, ref_disp)
        np.testing.assert_array_equal(result.valid, ref_valid)


def test_fix_restores_constant_plane_per_hand_computation():
    """Low-res 7x7 plane (d0/scale) with low-res (3,2) invalid; upscaled
    to 28x28 at scale 0.25. For every dst pixel with W>0 and a valid
    nearest tap the fix returns exactly d0; the pixel at (0,0)-corner taps
    touching the invalid block return d0 where the legacy path returned a
    diluted value."""
    sh = sw = 7
    out_h = out_w = 28
    scale = 0.25
    d0 = 8.0
    disp_small = np.full((sh, sw), d0 * scale, np.float32)
    valid_small = np.ones((sh, sw), np.uint8)
    valid_small[3, 2] = 0
    disp_small[valid_small == 0] = 0.0
    fix_d, fix_v = upsample_disparity_lowres(
        disp_small, valid_small, out_w, out_w, scale,
        UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
    legacy_d, legacy_v = upsample_disparity_lowres(
        disp_small, valid_small, out_w, out_h, scale,
        UPSAMPLE_LEGACY_ZERO_LINEAR_V1)
    # nearest support is identical between versions
    np.testing.assert_array_equal(fix_v > 0, legacy_v > 0)
    # every kept pixel is restored to the exact plane value
    kept = fix_v > 0
    assert kept.sum() > 0
    np.testing.assert_allclose(fix_d[kept], d0, atol=1e-4)
    # the legacy path demonstrably diluted some of those same pixels
    both = kept & (legacy_v > 0)
    assert both.sum() > 0
    dilution = np.abs(legacy_d[both] - d0)
    assert dilution.max() > 0.05
    assert np.abs(fix_d[both] - d0).max() < 1e-4


def test_single_valid_tap_nearest_valid_w_0390625_is_kept():
    """7x7 -> 4x4 (per-axis ratio 1.75): dst (0,0) has a single valid tap
    with weight 0.625 per axis, so W = 0.390625 — below the diagnostic
    control's 0.5 w_min, but the fix keeps it with the correctly
    normalized disparity (no threshold misuse)."""
    src = np.zeros((7, 7), np.float32)
    valid = np.zeros((7, 7), np.uint8)
    valid[0, 0] = 1
    d0 = 12.0
    scale = 0.25
    src[0, 0] = d0 * scale
    fix_d, fix_v = upsample_disparity_lowres(
        src, valid, 4, 4, scale, UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
    assert fix_v[0, 0] == 1
    assert fix_d[0, 0] == pytest.approx(d0, abs=1e-4)
    legacy_d, legacy_v = upsample_disparity_lowres(
        src, valid, 4, 4, scale, UPSAMPLE_LEGACY_ZERO_LINEAR_V1)
    assert legacy_v[0, 0] == 1
    assert legacy_d[0, 0] == pytest.approx(0.390625 * d0, abs=1e-4)
    # every other dst pixel has no valid nearest tap in both versions
    assert int(fix_v.sum()) == 1 and int(legacy_v.sum()) == 1


def test_nearest_invalid_stays_invalid_even_with_valid_other_taps():
    src = np.zeros((7, 7), np.float32)
    valid = np.zeros((7, 7), np.uint8)
    valid[0, 0] = 1  # only tap (0,0) valid
    d0 = 12.0
    scale = 0.25
    src[0, 0] = d0 * scale
    fix_d, fix_v = upsample_disparity_lowres(
        src, valid, 4, 4, scale, UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
    # the only valid pixel is the one whose nearest low-res tap is (0,0)
    assert int(fix_v.sum()) == 1 and fix_v[0, 0] == 1
    # no position outside it gains support although bilinear taps of some
    # neighbours reach (0,0)
    assert int((fix_v > 0).sum()) == int(
        (upsample_disparity_lowres(
            src, valid, 4, 4, scale,
            UPSAMPLE_LEGACY_ZERO_LINEAR_V1)[1] > 0).sum())


def test_fix_ignores_the_fill_values_of_invalid_taps():
    """The fix masks invalid taps explicitly: whatever values sit at
    invalid low-resolution positions must not change its output."""
    rng = np.random.default_rng(6)
    disp = rng.uniform(1.0, 3.0, (7, 7)).astype(np.float32)
    valid = np.ones((7, 7), np.uint8)
    valid[2:4, 3:5] = 0
    masked = np.where(valid > 0, disp, 0.0).astype(np.float32)
    garbage = disp.copy()
    garbage[valid == 0] = 999.0
    fix_masked, fix_valid = upsample_disparity_lowres(
        masked, valid, 28, 28, 0.25,
        UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
    fix_garbage, _ = upsample_disparity_lowres(
        garbage, valid, 28, 28, 0.25,
        UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
    np.testing.assert_array_equal(fix_masked, fix_garbage)
    # and on this scene the fix genuinely differs from the legacy path
    legacy_d, _ = upsample_disparity_lowres(
        masked, valid, 28, 28, 0.25, UPSAMPLE_LEGACY_ZERO_LINEAR_V1)
    assert not np.array_equal(fix_masked, legacy_d)


def test_all_valid_all_invalid_small_and_border_sizes():
    rng = np.random.default_rng(11)
    plane = rng.uniform(1.0, 3.0, (6, 6)).astype(np.float32)
    ones = np.ones((6, 6), np.uint8)
    zeros = np.zeros((6, 6), np.uint8)
    for out_w, out_h in ((24, 24), (17, 19), (5, 9), (1, 1)):
        fix_d, fix_v = upsample_disparity_lowres(
            plane, ones, out_w, out_h, 0.25,
            UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
        leg_d, leg_v = upsample_disparity_lowres(
            plane, ones, out_w, out_h, 0.25,
            UPSAMPLE_LEGACY_ZERO_LINEAR_V1)
        # all-valid: no invalid taps -> both versions identical
        np.testing.assert_array_equal(fix_d, leg_d)
        np.testing.assert_array_equal(fix_v, leg_v)
        fix_d0, fix_v0 = upsample_disparity_lowres(
            zeros.astype(np.float32), zeros, out_w, out_h, 0.25,
            UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
        assert int(fix_v0.sum()) == 0
        assert float(np.abs(fix_d0).max()) == 0.0


def test_non_finite_or_non_positive_valid_tap_rejected():
    base = np.full((4, 4), 2.0, np.float32)
    valid = np.ones((4, 4), np.uint8)
    for bad in (np.nan, np.inf, -1.0, 0.0):
        disp = base.copy()
        disp[1, 1] = bad
        with pytest.raises(ValueError):
            upsample_disparity_lowres(
                disp, valid, 16, 16, 0.25,
                UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)


def test_unknown_version_rejected_everywhere():
    with pytest.raises(ValueError):
        StereoMatcherConfig(upsample_version="w_min_0_5_v1")
    with pytest.raises(ValueError):
        upsample_disparity_lowres(
            np.ones((4, 4), np.float32), np.ones((4, 4), np.uint8),
            8, 8, 0.25, "valid_weighted_w05")


def test_valid_valid_discontinuity_keeps_original_interpolation():
    disp = np.full((6, 6), 2.0, np.float32)
    disp[:, 3:] = 5.0
    valid = np.ones((6, 6), np.uint8)
    fix_d, fix_v = upsample_disparity_lowres(
        disp, valid, 24, 24, 0.25,
        UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
    leg_d, leg_v = upsample_disparity_lowres(
        disp, valid, 24, 24, 0.25, UPSAMPLE_LEGACY_ZERO_LINEAR_V1)
    np.testing.assert_array_equal(fix_d, leg_d)
    np.testing.assert_array_equal(fix_v, leg_v)
    # the boundary still mixes the two planes in BOTH versions: not fixed
    assert np.abs(fix_d.astype(np.float64) - 8.0).max() > 1e-6


def test_nearest_valid_but_all_bilinear_taps_invalid_is_excluded():
    """A dst pixel whose NEAREST low-res tap is valid can still have every
    bilinear tap invalid (the nearest and bilinear index conventions
    diverge on downsampling): the fix must exclude it (W=0) while the
    legacy path marks it valid with a zero-valued disparity."""
    valid = np.zeros((12, 12), np.uint8)
    valid[3, 3] = 1
    disp = np.zeros((12, 12), np.float32)
    disp[3, 3] = 0.25
    # 12 -> 4 downsample (ratio 3.0): dst(1,1) has nearest (3,3) valid but
    # bilinear taps (4,5) invalid in both axes
    fix_d, fix_v = upsample_disparity_lowres(
        disp, valid, 4, 4, 1.0, UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1)
    leg_d, leg_v = upsample_disparity_lowres(
        disp, valid, 4, 4, 1.0, UPSAMPLE_LEGACY_ZERO_LINEAR_V1)
    assert leg_v[1, 1] == 1 and leg_d[1, 1] == 0.0, \
        "the legacy path fabricates a valid zero disparity here"
    assert fix_v[1, 1] == 0, \
        "the fix must refuse a position whose bilinear weight sum is 0"
    assert int(fix_v.sum()) == 0
    assert float(np.abs(fix_d).max()) == 0.0


def test_real_production_matcher_reaches_the_fix_version():
    """The explicit version reaches the production matcher and actually
    changes the output (a config entry that silently ran the legacy path
    would fail here)."""
    left, right = _scene(21)
    matcher_l = StereoMatcher(StereoMatcherConfig(
        max_disparity=32, scale=0.25, block_size=5))
    matcher_f = StereoMatcher(StereoMatcherConfig(
        max_disparity=32, scale=0.25, block_size=5,
        upsample_version=UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1))
    assert matcher_f.config.upsample_version == \
        UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1
    r_l = matcher_l.compute(left, right)
    r_f = matcher_f.compute(left, right)
    common = (r_l.valid > 0) & (r_f.valid > 0)
    assert common.sum() > 0
    diff = np.abs(r_f.disparity[common] - r_l.disparity[common])
    assert int((diff > 1e-6).sum()) > 0, (
        "the fix version must change diluted disparities on a scene with "
        "invalid taps; identical output means the version was ignored")


def test_compute_bands_versions_and_policies_unchanged():
    # a blocky (4-px) texture matches reliably at 0.25 downscale; the band
    # is tall enough for a real in-band match and rows 46+ stay genuinely
    # out-of-band
    rng = np.random.default_rng(31)
    low = rng.integers(60, 200, (16, 24)).astype(np.uint8)
    full = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)
    left = np.ascontiguousarray(full[:64, :96])
    right = np.zeros_like(left)
    right[:, 10:] = left[:, :-10]
    bands = build_horizontal_bands([(0, 0, 95, 36)], image_height=64,
                                   vertical_margin=5)
    for version in (UPSAMPLE_LEGACY_ZERO_LINEAR_V1,
                    UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1):
        matcher = StereoMatcher(StereoMatcherConfig(
            max_disparity=32, scale=0.25, block_size=5,
            upsample_version=version))
        result = matcher.compute_bands(left, right, bands)
        # the band covers rows 0..40 (+margin); rows 46.. are genuinely
        # out-of-band and must stay entirely invalid, while the band
        # interior carries valid disparity
        assert int(result.valid[:41].sum()) > 0
        assert int(result.valid[46:].sum()) == 0, \
            "rows outside every band must stay invalid"
        # low-coverage rejection: a band whose matching fails entirely
        empty = matcher.compute_bands(
            np.full((10, 40), 128, np.uint8),
            np.full((10, 40), 128, np.uint8),
            build_horizontal_bands([(0, 0, 39, 10)], image_height=10,
                                   vertical_margin=0))
        assert int(empty.valid.sum()) == 0
        # full-image fallback: bands covering >=90% behave like compute
        full = matcher.compute_bands(
            left, right, build_horizontal_bands(
                [(0, 0, 95, 59)], image_height=64, vertical_margin=5))
        single = matcher.compute(left, right)
        np.testing.assert_array_equal(full.disparity, single.disparity)
        np.testing.assert_array_equal(full.valid, single.valid)


# ---------------------------------------------------------------------------
# Evaluation entry machinery (stub classifiers over the real frame data)


HISTORICAL_PREDICTIONS = (PROJECT_ROOT /
                          "analysis/runs/material_residual_runtime_v1/"
                          "residrt_c_20261009T125242Z_6c0ed9/evaluate/"
                          "predictions_complete.jsonl")


class _StubSpec:
    def __init__(self, candidate_id: str) -> None:
        self.candidate_id = candidate_id
        self.family = "p0"
        self.variant = "R0"
        self.seed = None
        if candidate_id != "R0":
            self.family = ("linear_local"
                           if candidate_id.startswith("linear_local_")
                           else "conditional")
            tail = candidate_id.split("_")
            self.variant = tail[-2]
            self.seed = int(tail[-1].lstrip("s"))


class _StubResult:
    def __init__(self, class_id, logits, valid=True, used_fallback=False,
                 fallback_reason=""):
        self.class_id = class_id
        self.valid = valid
        self.raw_logits = logits
        self.used_fallback = used_fallback
        self.fallback_reason = fallback_reason
        self.model_identity = "stub"
        self.invalid_reason = None if valid else "stub_failure"
        self.confidence = 0.5


class _StubClassifier:
    def __init__(self, candidate_id: str, historical: dict) -> None:
        self.candidate_spec = _StubSpec(candidate_id)
        self.candidate_id = candidate_id
        self._historical = historical

    def predict_batch(self, inputs, gray=None):
        outcomes = []
        for inp in inputs:
            row = self._historical.get(
                (self.candidate_id, inp.frame_id, int(inp.instance_id)))
            if row is None or row.get("status") != "valid":
                outcomes.append(_StubResult(-1, [], valid=False))
            else:
                outcomes.append(_StubResult(
                    int(row["predicted_class"]),
                    [float(v) for v in row["final_logits"]],
                    used_fallback=bool(row.get("used_fallback")),
                    fallback_reason=row.get("fallback_reason", "")))
        return outcomes

    def gray_features(self, samples):
        return np.zeros((len(samples), 256), dtype=np.float32)

    def local_features(self, sample):
        c14 = np.zeros(14, dtype=np.float32)
        c14[0] = float(sample.quality[0])
        s16 = np.full(16, np.nan, dtype=np.float32)
        return {"c_features": c14, "s_features": s16, "s_valid": False,
                "reason": "stub"}


class _StubRuntimeConfig:
    def __init__(self) -> None:
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )
        real = load_residual_runtime_config(
            PROJECT_ROOT / "configs/experiments/"
            "material_residual_runtime_v1_complete.yaml")
        self.stereo = real.stereo
        self.calibration = real.calibration
        self.crop_padding = real.crop_padding
        self.candidates = real.candidates
        self.engineering_seed = real.engineering_seed
        # identity fields the admission registry binds and the frozen
        # loader digest-verifies (real pinned files, never fabricated)
        self.encoder_path = real.encoder_path
        self.encoder_sha256 = real.encoder_sha256
        self.global_cache = real.global_cache
        self.local_cache = real.local_cache
        self.p0_models = real.p0_models
        self.residual_models = real.residual_models
        self.r0_records = real.r0_records
        # loader-consumed identity files the admission registry now
        # registers (fit summary/admission + protocol pins, candidates'
        # fold_files come through real.candidates above)
        self.fit_dir = real.fit_dir
        self.summary_sha256 = real.summary_sha256
        self.admission_sha256 = real.admission_sha256
        self.protocol_path = real.protocol_path
        self.protocol_sha256 = real.protocol_sha256


def _historical_rows() -> dict:
    rows = {}
    for line in HISTORICAL_PREDICTIONS.read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["candidate_id"], row["frame_id"],
                  int(row["pred_instance_id"]))] = row
    return rows


@pytest.fixture()
def entry_world(monkeypatch):
    """Stub classifiers that REPRODUCE the historical predictions, so the
    entry's frame loop, identity checks, reproduction gate and reporting
    run end-to-end without loading any real model."""
    historical = _historical_rows()

    def fake_build_classifiers(config, protocol, *, device, frame_folds,
                               encoder=None):
        return {cid: _StubClassifier(cid, historical)
                for cid in sorted({cid for cid, _, _ in historical})}

    monkeypatch.setattr(impact, "build_classifiers", fake_build_classifiers)
    monkeypatch.setattr(impact, "load_residual_runtime_config",
                        lambda path: _StubRuntimeConfig())

    def fake_load_protocol(config):
        import csv as csv_mod

        with (PROJECT_ROOT / "datasets/underwater_seg_binary_v2/"
              "split_assignment.csv").open(encoding="utf-8") as handle:
            groups = sorted({row["group_name"] for row in
                             csv_mod.DictReader(handle)
                             if row["split"] == "val"})
        return {"folds": {"folds": [
            {"fold_id": "fold0", "outer_eval_groups": groups}]}}

    monkeypatch.setattr(impact, "load_protocol", fake_load_protocol)
    return historical


def _entry_config(tmp_path: Path, candidates: list[str]) -> Path:
    import yaml

    base = yaml.safe_load((PROJECT_ROOT / "configs/experiments/"
                           "material_upsample_impact_v1.yaml")
                          .read_text(encoding="utf-8"))
    base["candidates"]["expected_ids"] = candidates
    # the stub classifiers parrot the HISTORICAL runtime rows, not the
    # retained formal_v2 run, so the reference-run check (which requires
    # an exact reproduction of formal_v2) is not configured here; it is
    # covered by unit tests against synthetic rows and by the real formal
    # run with real checkpoints
    base.pop("reference_formal_v2", None)
    path = tmp_path / "entry_config.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def test_entry_evaluate_smoke_runs_and_verifies_identity(
        entry_world, tmp_path: Path) -> None:
    config_path = _entry_config(
        tmp_path, sorted({k[0] for k in entry_world}))
    out = tmp_path / "run_out"
    code = impact.main([
        "--config", str(config_path), "--mode", "evaluate",
        "--output-dir", str(out), "--device", "cpu",
        "--budget-minutes", "10", "--max-frames", "2",
    ])
    assert code == impact.EXIT_OK, \
        (out / "run_status.json").read_text(encoding="utf-8")
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "COMPLETED"
    assert status["smoke"] is True
    rows = [json.loads(line) for line in
            (out / "predictions_arms.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    # rows == predictions of the first 2 val frames x 9 candidates x 2 arms
    frames = [json.loads(line) for line in
              (out / "frames_arms.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    expected_rows = sum(f["predictions"] for f in frames) * 9 * 2
    assert len(rows) == expected_rows > 0
    keys = {(r["arm"], r["candidate_id"], r["frame_id"],
             r["pred_instance_id"]) for r in rows}
    assert len(keys) == len(rows), "duplicate prediction rows"
    reproduction = json.loads((out / "legacy_reproduction.json")
                              .read_text(encoding="utf-8"))
    assert reproduction["problem_count"] == 0
    r0 = json.loads((out / "r0_invariance.json").read_text(
        encoding="utf-8"))
    assert r0["problem_count"] == 0
    protection = json.loads((out / "run_status.json").read_text(
        encoding="utf-8"))["protection"]
    assert protection["identical"] is True
    metrics = json.loads((out / "metrics_summary.json").read_text(
        encoding="utf-8"))
    assert metrics["production_vs_independent"]["mismatches"] == []


def test_entry_reproduction_failure_withholds_impact(
        entry_world, tmp_path: Path) -> None:
    """A biased legacy arm must fail the reproduction gate and withhold the
    impact comparison entirely."""
    config_path = _entry_config(
        tmp_path, sorted({k[0] for k in entry_world}))
    # bias: every legacy prediction shifted by one class
    original = _StubClassifier.predict_batch

    def biased(self, inputs, gray=None):
        outcomes = original(self, inputs, gray=gray)
        for outcome in outcomes:
            if outcome.valid and outcome.class_id is not None:
                outcome.class_id = (outcome.class_id + 1) % 4
                outcome.raw_logits = [
                    v + 1.0 for v in outcome.raw_logits]
        return outcomes

    monkeypatch_setattr = pytest.MonkeyPatch()
    monkeypatch_setattr.setattr(_StubClassifier, "predict_batch",
                                biased)
    try:
        out = tmp_path / "run_out"
        code = impact.main([
            "--config", str(config_path), "--mode", "evaluate",
            "--output-dir", str(out), "--device", "cpu",
            "--budget-minutes", "10", "--max-frames", "1",
        ])
        assert code == impact.EXIT_FAILED
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert "did not reproduce" in (status["failure"] or "")
        assert not (out / "impact_summary.json").exists(), (
            "the impact comparison must be withheld when legacy fails "
            "to reproduce history")
    finally:
        monkeypatch_setattr.undo()


def test_entry_budget_exhaustion_keeps_partial_evidence(
        entry_world, tmp_path: Path, monkeypatch) -> None:
    config_path = _entry_config(
        tmp_path, sorted({k[0] for k in entry_world}))
    checks = {"n": 0}

    class ExplodingWatch:
        def __init__(self, minutes: float, *, clock=None) -> None:
            pass

        def check(self, at: str = "") -> None:
            checks["n"] += 1
            if checks["n"] > 6:  # let admission + a frame or two through
                raise impact.BudgetExceeded(f"fake budget gone at {at}")

        def remaining_s(self) -> float:
            return 0.0

    monkeypatch.setattr(impact, "BudgetWatch", ExplodingWatch)
    out = tmp_path / "run_out"
    code = impact.main([
        "--config", str(config_path), "--mode", "evaluate",
        "--output-dir", str(out), "--device", "cpu",
        "--budget-minutes", "10", "--max-frames", "3",
    ])
    assert code == impact.EXIT_PARTIAL
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "PARTIAL"
    assert status["interrupted_by_budget"] is True
    rows = [line for line in (out / "predictions_arms.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) > 0, "finished frames' evidence must be kept"
    assert not (out / "impact_summary.json").exists()


def test_model_a_probe_records_then_raises():
    probe = impact.ModelAProbe()
    with pytest.raises(AssertionError):
        probe.predict(np.zeros((4, 4), np.uint8), imgsz=640)
    assert len(probe.calls) == 1, \
        "the probe must RECORD the call before raising, so a swallowed " \
        "exception cannot fake zero calls"


def test_load_frame_instances_identity_failures(tmp_path: Path) -> None:
    import numpy as np

    from models.segmentation import Instance  # noqa: F401

    det_dir = tmp_path / "detection" / "conditions" / "D0"
    (det_dir / "masks").mkdir(parents=True)
    frame_id = "F1"
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 1
    mask2 = np.zeros((8, 8), np.uint8)
    mask2[1:5, 3:7] = 1
    np.savez(det_dir / "masks" / f"{frame_id}.npz",
             masks=np.packbits(np.stack([mask, mask2]).reshape(2, -1), axis=1),
             shape=np.array([8, 8]),
             confidences=np.array([0.9, 0.8], dtype=np.float32),
             bboxes=np.array([[2, 2, 6, 6], [2, 2, 6, 6]], dtype=np.int32))
    d0_rows = {0: {"bbox": [2, 2, 6, 6]}, 1: {"bbox": [2, 2, 6, 6]}}
    frame = {"frame_id": frame_id, "frame_error": None,
             "predictions": [{"pred_instance_id": 0},
                             {"pred_instance_id": 1}],
             "gt": [{"gt_id": 0, "class_id": 1}], "matches": [[0, 0, 0.9]]}
    cfg = {"sources": {"detection_dir": str(tmp_path / "detection")}}
    admission = {"frame_by_id": {frame_id: frame}, "d0_by_frame":
                 {frame_id: d0_rows}}
    registry = impact.SourceRegistry({})
    instances, _ = impact.load_frame_instances(admission, cfg, frame_id,
                                               registry)
    assert [i.id for i in instances] == [0, 1]
    # wrong bbox in the D0 record
    bad = {0: {"bbox": [0, 0, 1, 1]}, 1: {"bbox": [2, 2, 6, 6]}}
    with pytest.raises(impact.IdentityError, match="bbox"):
        impact.load_frame_instances(
            {**admission, "d0_by_frame": {frame_id: bad}}, cfg, frame_id,
            registry)
    # missing mask file for an expected non-empty frame
    empty_dir = tmp_path / "no_masks"
    cfg_missing = {"sources": {"detection_dir": str(empty_dir.parent)}}
    admission_missing = {
        "frame_by_id": {frame_id: frame},
        "d0_by_frame": {frame_id: {0: {"bbox": [2, 2, 6, 6]}}}}
    with pytest.raises(Exception):
        impact.load_frame_instances(admission_missing, cfg_missing,
                                    frame_id, impact.SourceRegistry({}))
    # duplicate prediction ids in the D0 record
    dup = {0: {"bbox": [2, 2, 6, 6]}}
    with pytest.raises(impact.IdentityError):
        impact.load_frame_instances(
            {**admission, "d0_by_frame": {frame_id: dup}}, cfg, frame_id,
            registry)


def test_paired_fix_minus_legacy_counts_and_detail():
    def row(arm, cid, fid, pid, gt, true, pred, status="valid",
            fallback=False):
        return {"arm": arm, "candidate_id": cid, "frame_id": fid,
                "pred_instance_id": pid, "matched_gt_id": gt,
                "true_class": true, "predicted_class": pred,
                "status": status, "used_fallback": fallback,
                "final_logits": [1.0, 2.0, 3.0, 4.0],
                "fallback_reason": "x" if fallback else ""}

    rows = [
        # improved: legacy wrong, fix right
        row("legacy", "c", "F1", 0, 0, 1, 2),
        row("fix", "c", "F1", 0, 0, 1, 1),
        # broken: legacy right, fix wrong
        row("legacy", "c", "F1", 1, 1, 2, 2),
        row("fix", "c", "F1", 1, 1, 2, 0),
        # both wrong with different wrong classes
        row("legacy", "c", "F1", 2, 2, 0, 1),
        row("fix", "c", "F1", 2, 2, 0, 3),
        # unmatched prediction: excluded from the paired counts
        row("legacy", "c", "F1", 3, None, None, 0),
        row("fix", "c", "F1", 3, None, None, 0),
    ]
    out = impact.paired_fix_minus_legacy(rows, ["c"])["c"]
    assert out["matched_instances"] == 3
    assert out["improved"] == 1 and out["broken"] == 1
    assert out["both_correct"] == 0 and out["both_wrong"] == 1
    assert out["changed_predictions"] == 3
    assert all(d["true_class"] is not None for d in out["detail"])


def test_cs_change_summary_records_covariates_not_gains():
    legacy_cs = {"c_features": np.zeros(14, np.float32),
                 "s_features": np.full(16, 0.5, np.float32),
                 "s_valid": True, "reason": ""}
    fix_cs = {"c_features": np.ones(14, np.float32) * 0.5,
              "s_features": np.full(16, 0.8, np.float32),
              "s_valid": True, "reason": ""}
    rows = []
    for arm, cs in (("legacy", legacy_cs), ("fix", fix_cs)):
        rows.append({
            "arm": arm, "frame_id": "F1", "instance_id": 0,
            "matched": True, "quality": [0.5, 0.6, 0.7, 0.0],
            "polar_invalid": False, "invalid_reason": "",
            "s_valid": cs["s_valid"], "s_missing_reason": cs["reason"],
            "c14": [float(v) for v in cs["c_features"]],
            "s16": [float(v) for v in cs["s_features"]],
        })
    out = impact.summarize_cs_changes(rows)
    assert out["s_valid_flips"]["both_valid"] == 1
    assert out["c14_max_abs_delta_mean_over_instances"] == pytest.approx(0.5)
    assert out["s16_max_abs_delta_mean_over_instances_both_valid"] == \
        pytest.approx(0.3)
    assert "no gain is claimed" in out["note"]


def test_independent_recompute_hand_computed_full_semantics():
    """Hand-computed world: a correct match, a wrong match, a FAILED match,
    a missed GT, a valid unmatched prediction and a failed unmatched
    prediction. The independent recomputer must reproduce every count
    exactly (including the missed-GT e2e FN that flat prediction rows
    never mention)."""
    frames = [
        {"frame_id": "FA", "gt": [{"gt_id": 0, "class_id": 0},
                                  {"gt_id": 1, "class_id": 1},
                                  {"gt_id": 2, "class_id": 2}],
         "predictions": [{"pred_instance_id": 0},
                         {"pred_instance_id": 1},
                         {"pred_instance_id": 2}],
         "matches": [[0, 0, 0.9], [1, 1, 0.8], [2, 2, 0.7]]},
        {"frame_id": "FB", "gt": [{"gt_id": 0, "class_id": 3}],
         "predictions": [{"pred_instance_id": 0},
                         {"pred_instance_id": 1}],
         "matches": []},
    ]
    # FA: pred0 correct (class0 for gt0); pred1 wrong (class2 for gt1);
    #     pred2 FAILED on gt2. FB has no predictions: gt3 missed.
    # FB also carries a valid unmatched prediction of class 1 and a failed
    # unmatched one (the failed one contributes no FP).
    rows = []
    for cid in ("R0",):
        rows += [
            {"arm": "legacy", "candidate_id": cid, "frame_id": "FA",
             "pred_instance_id": 0, "matched_gt_id": 0, "true_class": 0,
             "predicted_class": 0, "status": "valid",
             "used_fallback": False, "final_logits": [2, 0, 0, 0]},
            {"arm": "legacy", "candidate_id": cid, "frame_id": "FA",
             "pred_instance_id": 1, "matched_gt_id": 1, "true_class": 1,
             "predicted_class": 2, "status": "valid",
             "used_fallback": False, "final_logits": [0, 0, 2, 0]},
            {"arm": "legacy", "candidate_id": cid, "frame_id": "FA",
             "pred_instance_id": 2, "matched_gt_id": 2, "true_class": 2,
             "predicted_class": None, "status": "failed",
             "used_fallback": False, "final_logits": []},
            {"arm": "legacy", "candidate_id": cid, "frame_id": "FB",
             "pred_instance_id": 0, "matched_gt_id": None,
             "true_class": None, "predicted_class": 1, "status": "valid",
             "used_fallback": False, "final_logits": [0, 2, 0, 0]},
            {"arm": "legacy", "candidate_id": cid, "frame_id": "FB",
             "pred_instance_id": 1, "matched_gt_id": None,
             "true_class": None, "predicted_class": None,
             "status": "failed", "used_fallback": False,
             "final_logits": []},
        ]
    out = impact.independent_recompute(frames, rows, ["R0"],
                                       ("legacy",))["legacy"]["R0"]
    # conditional denominator counts CLASSIFIED matches only (production
    # semantics): 2 samples (failed match excluded), 1 correct
    assert out["samples"] == 2 and out["correct"] == 1
    assert out["accuracy"] == pytest.approx(1 / 2)
    assert out["failed_matched"] == 1
    assert out["confusion"] == [[1, 0, 0, 0], [0, 0, 1, 0],
                                [0, 0, 0, 0], [0, 0, 0, 0]]
    # e2e: tp class0=1; fn class1=1 (wrong match), class2=1 (failed
    # match), class3=1 (missed GT in FB); fp class2=1 (wrong pred),
    # class1=1 (valid unmatched)
    assert out["e2e_tp"] == [1, 0, 0, 0]
    assert out["e2e_fn"] == [0, 1, 1, 1]
    assert out["e2e_fp"] == [0, 1, 1, 0]
    assert out["failed_matched"] == 1
    # the failed unmatched prediction contributes NO fp
    assert out["e2e_fp"][1] == 1
    # cross-check the SAME world through the production record path
    results_stub = [
        _StubResult(0, [2, 0, 0, 0]),
        _StubResult(2, [0, 0, 2, 0]),
        _StubResult(-1, [], valid=False),
    ]
    record_a = _build_record(frame_id="FA", group="G",
                             gt_classes=[0, 1, 2],
                             matches=[(0, 0, 0.9), (1, 1, 0.8),
                                      (2, 2, 0.7)],
                             results=results_stub, unclassified={2},
                             matched_quality={})
    record_b = _build_record(frame_id="FB", group="G", gt_classes=[3],
                             matches=[], results=[_StubResult(1, [0, 2, 0, 0]),
                                                  _StubResult(-1, [],
                                                              valid=False)],
                             unclassified=set(), matched_quality={})
    prod = aggregate_metrics([record_a, record_b], list(impact.CLASS_NAMES))
    ind_cond = out
    prod_cond = prod["conditional_classification"]
    assert prod_cond["samples"] == ind_cond["samples"]
    assert prod_cond["accuracy"] == pytest.approx(ind_cond["accuracy"])
    assert prod_cond["macro_f1"] == pytest.approx(ind_cond["macro_f1"])
    assert prod_cond["confusion_matrix"] == ind_cond["confusion"]
    assert prod["end_to_end"]["macro_f1"] == \
        pytest.approx(ind_cond["end_to_end_macro_f1"])


def test_verify_reference_run_detects_divergence_and_tampering(
        tmp_path: Path) -> None:
    """The retained formal_v2 run is the repair's regression reference:
    identical rows pass, a changed class fails, BOTH retained files are
    identity-checked, a logits length mismatch and a valid row whose
    logits are not four finite values fail explicitly, and a tampered
    reference file is refused before any comparison."""
    metrics_path = tmp_path / "reference_metrics_summary.json"
    metrics_path.write_text(json.dumps({"macro_f1": 0.5}), encoding="utf-8")

    def write_ref(rows) -> None:
        ref_path.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    ref_rows = [
        {"arm": "legacy", "candidate_id": "R0", "frame_id": "F1",
         "pred_instance_id": 0, "matched_gt_id": 0, "true_class": 1,
         "predicted_class": 1, "status": "valid", "used_fallback": False,
         "fallback_reason": "", "final_logits": [0.1, 2.0, 0.0, 0.0]},
        {"arm": "fix", "candidate_id": "R0", "frame_id": "F1",
         "pred_instance_id": 0, "matched_gt_id": 0, "true_class": 1,
         "predicted_class": 1, "status": "valid", "used_fallback": False,
         "fallback_reason": "", "final_logits": [0.1, 2.0, 0.0, 0.0]},
    ]
    ref_path = tmp_path / "reference_predictions.jsonl"
    write_ref(ref_rows)
    cfg = {
        "tolerances": {"logit_atol": 1e-4},
        "reference_formal_v2": {
            "predictions_arms": str(ref_path),
            "predictions_arms_sha256": impact.sha256_of_file(ref_path),
            "metrics_summary": str(metrics_path),
            "metrics_summary_sha256": impact.sha256_of_file(metrics_path),
        },
    }
    identical = [dict(r) for r in ref_rows]
    out = impact.verify_reference_run(identical, cfg, PROJECT_ROOT)
    assert out["problem_count"] == 0 and out["pairs"] == 2
    assert set(out["retained_identities"]) == {"predictions_arms",
                                               "metrics_summary"}
    divergent = [dict(ref_rows[0]),
                 {**ref_rows[1], "predicted_class": 2}]
    out = impact.verify_reference_run(divergent, cfg, PROJECT_ROOT)
    assert out["problem_count"] == 1
    assert "class/status" in out["problems"][0]
    # a logits LENGTH mismatch between the new and the reference row is an
    # explicit problem, never silently skipped
    short = [dict(ref_rows[0]), {**ref_rows[1],
                                 "final_logits": [0.1, 2.0, 0.0]}]
    out = impact.verify_reference_run(short, cfg, PROJECT_ROOT)
    assert any("logits length" in p for p in out["problems"])
    assert any("not 4 values" in p for p in out["problems"])
    # a valid row with non-finite logits fails explicitly
    nan = [dict(ref_rows[0]), {**ref_rows[1],
                               "final_logits": [0.1, float("nan"),
                                                0.0, 0.0]}]
    out = impact.verify_reference_run(nan, cfg, PROJECT_ROOT)
    assert any("non-finite" in p for p in out["problems"])
    # a tampered reference predictions file is refused before comparison
    write_ref([{**ref_rows[0], "predicted_class": 3}, ref_rows[1]])
    with pytest.raises(impact.IdentityError, match="tampered"):
        impact.verify_reference_run(identical, cfg, PROJECT_ROOT)
    # a tampered reference METRICS file is refused as well
    write_ref(ref_rows)
    metrics_path.write_text(json.dumps({"macro_f1": 0.9}), encoding="utf-8")
    with pytest.raises(impact.IdentityError, match="metrics_summary"):
        impact.verify_reference_run(identical, cfg, PROJECT_ROOT)


def test_frame_arm_comparison_single_arm_missing_and_all_invalid():
    """Either arm missing its polar map means that arm's q support is
    all-invalid: the other arm's pixels become lost or gained, the common
    region is empty, and the disparity statistics are computed
    independently of the q maps."""
    h, w = 8, 8
    mask = np.zeros((h, w), bool)
    mask[2:6, 2:6] = True
    stereo_l = np.zeros((h, w), np.float32)
    stereo_f = np.zeros((h, w), np.float32)
    valid_l = np.zeros((h, w), np.uint8)
    valid_f = np.zeros((h, w), np.uint8)
    valid_l[2:6, 2:6] = 1
    valid_f[2:6, 2:6] = 1
    zeros = np.zeros((h, w), np.float32)
    frame = {"matches": [[0, 0, 0.9]]}

    from core.polar_compute import PolarFeatureResult
    polar_l = PolarFeatureResult(
        signed_q=np.full((h, w), 0.5, np.float32),
        abs_q=np.full((h, w), 0.5, np.float32),
        valid_mask=valid_l.astype(bool),
        in_bounds_mask=np.ones((h, w), bool),
        brightness_valid_mask=np.ones((h, w), bool),
        valid_ratio=1.0, in_bounds_ratio=1.0, brightness_valid_ratio=1.0)
    inst = SimpleNamespace(id=0, mask=mask.astype(np.uint8))
    # arm records are DICTS (as run_frame_arms returns them); legacy has q,
    # fix does not -> every legacy q pixel is lost
    legacy = {"result": SimpleNamespace(polar_result=polar_l),
              "stereo_disparity": zeros, "stereo_valid": valid_l}
    fix = {"result": SimpleNamespace(polar_result=None),
           "stereo_disparity": zeros, "stereo_valid": valid_f}
    out = impact.frame_arm_comparison(frame, [inst], legacy, fix)
    assert out["q_common_pixels"] == 0
    assert out["q_lost"] == 16 and out["q_gained"] == 0
    assert out["q_absdq_mean_on_common"] is None
    # disparity stats are computed independently (both zero here)
    assert out["disparity_common_pixels"] == 16
    assert out["disparity_absdiff_mean_on_common"] == 0.0
    # both arms invalid: nothing lost or gained, everything empty
    legacy_none = {"result": SimpleNamespace(polar_result=None),
                   "stereo_disparity": zeros, "stereo_valid": valid_l}
    fix_none = {"result": SimpleNamespace(polar_result=None),
                "stereo_disparity": zeros, "stereo_valid": valid_f}
    out = impact.frame_arm_comparison(frame, [inst], legacy_none, fix_none)
    assert out["q_common_pixels"] == 0
    assert out["q_lost"] == 0 and out["q_gained"] == 0
    assert out["disparity_common_pixels"] == 16


def test_lowres_core_single_pass_evidence():
    """The shared-core contract is proven by CALL COUNTS, not by comparing
    two results: the legacy matcher performs exactly one low-resolution
    match per band and the fix matcher performs none."""
    left, right = _scene(41)
    matcher_l = StereoMatcher(StereoMatcherConfig(
        max_disparity=32, scale=0.25, block_size=5))
    matcher_f = StereoMatcher(StereoMatcherConfig(
        max_disparity=32, scale=0.25, block_size=5,
        upsample_version=UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1))
    bands = build_horizontal_bands([(0, 0, 95, 20), (30, 0, 95, 44)],
                                   image_height=48, vertical_margin=5)
    cores = matcher_l.match_cores(left, right, bands)
    assert matcher_l.lowres_match_calls == len(cores) >= 1
    assert matcher_f.lowres_match_calls == 0
    result_l = matcher_l.assemble_from_cores(cores, left.shape)
    result_f = matcher_f.assemble_from_cores(cores, left.shape)
    assert matcher_f.lowres_match_calls == 0, \
        "assembly must not run any matching on the fix matcher"
    assert int(result_f.valid.sum()) > 0
    # a second matching pass on either matcher would break the contract
    matcher_l.match_cores(left, right, bands)
    assert matcher_l.lowres_match_calls == len(cores), \
        "each top-level call resets and re-counts its own matches"


def test_from_config_and_from_material_config_pass_upsample_version(
        tmp_path: Path) -> None:
    """Both production config entries must forward stereo.upsample_version:
    absent -> legacy default; an explicit fix in a config COPY reaches the
    real matcher; an unknown version is refused through the REAL config
    entries, not only at dataclass construction. Config copies live in
    tmp_path, never in a fixed probe path."""
    import gui.inference_engine as engine_mod
    from gui.inference_engine import DualStageInferenceEngine
    import yaml

    class _StubModelA:
        def predict(self, image, **kwargs):
            raise AssertionError("Model A must never run in this test")

    assert engine_mod.UPSAMPLE_LEGACY_ZERO_LINEAR_V1 == "legacy_zero_linear_v1"
    # from_config: default YAML has no upsample_version key -> legacy; the
    # model_a stub is injected via overrides so no real Model A loads
    engine = DualStageInferenceEngine.from_config(
        model_a=_StubModelA(), model_b=None, device="cpu")
    assert engine.matcher.config.upsample_version == "legacy_zero_linear_v1"
    # an explicit fix version in a COPY of the config reaches the matcher
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/default.yaml")
                         .read_text(encoding="utf-8"))
    cfg.setdefault("stereo", {})["upsample_version"] = \
        "valid_weighted_legacy_support_v1"
    cfg["model_a"] = {"path": "configs/default.yaml"}
    cfg.setdefault("model_b", {"path": "configs/default.yaml"})
    cfg_path = tmp_path / "default_fix_copy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    engine = DualStageInferenceEngine.from_config(
        config_path=cfg_path,
        model_a=_StubModelA(), model_b=None, device="cpu")
    assert engine.matcher.config.upsample_version == \
        "valid_weighted_legacy_support_v1"
    # an unknown version is refused THROUGH the real from_config entry
    cfg["stereo"]["upsample_version"] = "no_such_version"
    bad_path = tmp_path / "default_unknown_copy.yaml"
    bad_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="upsample_version"):
        DualStageInferenceEngine.from_config(
            config_path=bad_path,
            model_a=_StubModelA(), model_b=None, device="cpu")


def test_entry_late_budget_exhaustion_publishes_partial(
        entry_world, tmp_path: Path, monkeypatch) -> None:
    """A budget that runs out in the LATE phase (after metrics) must still
    publish the completed evidence as PARTIAL with a nonzero exit - never
    SUCCESS."""
    config_path = _entry_config(
        tmp_path, sorted({k[0] for k in entry_world}))

    class LateExplodingWatch:
        def __init__(self, minutes: float, *, clock=None) -> None:
            pass

        def check(self, at: str = "") -> None:
            if at.startswith("before impact"):
                raise impact.BudgetExceeded(f"late budget gone at {at}")

        def remaining_s(self) -> float:
            return 0.0

    monkeypatch.setattr(impact, "BudgetWatch", LateExplodingWatch)
    out = tmp_path / "run_out"
    code = impact.main([
        "--config", str(config_path), "--mode", "evaluate",
        "--output-dir", str(out), "--device", "cpu",
        "--budget-minutes", "10", "--max-frames", "1",
    ])
    assert code == impact.EXIT_PARTIAL
    status = json.loads((out / "run_status.json").read_text(
        encoding="utf-8"))
    assert status["overall_status"] == "PARTIAL"
    assert status["interrupted_by_budget"] is True
    # metrics completed before the exhaustion and are kept
    assert (out / "metrics_summary.json").exists()
    assert not (out / "impact_summary.json").exists()


def test_prediction_key_problems_flags_missing_extra_and_duplicate_rows():
    expected = {("legacy", "R0", "F1", 0), ("legacy", "R0", "F1", 1),
                ("fix", "R0", "F1", 0), ("fix", "R0", "F1", 1)}
    rows = [{"arm": a, "candidate_id": "R0", "frame_id": "F1",
             "pred_instance_id": p} for a in ("legacy", "fix")
            for p in (0, 1)]
    assert impact.prediction_key_problems(expected, rows) == []
    rows_missing = [r for r in rows
                    if not (r["arm"] == "fix" and r["pred_instance_id"] == 1)]
    problems = impact.prediction_key_problems(expected, rows_missing)
    assert any("missing" in p for p in problems)
    extra = rows + [{"arm": "fix", "candidate_id": "R0", "frame_id": "F1",
                     "pred_instance_id": 9}]
    problems = impact.prediction_key_problems(expected, extra)
    assert any("unexpected" in p for p in problems)
    # a duplicated row is rejected even though the key SET is complete
    duplicated = rows + [dict(rows[0])]
    problems = impact.prediction_key_problems(expected, duplicated)
    assert any("duplicate" in p for p in problems), problems


def test_from_material_config_passes_upsample_version(tmp_path: Path):
    """from_material_config forwards stereo.upsample_version from a config
    COPY (the existing material runtime config stays byte-identical):
    default legacy, an explicit fix version reaches the matcher, and
    Model A is a controlled stub via model_a_factory (never loaded or
    run)."""
    from gui.inference_engine import DualStageInferenceEngine

    calls = {"model_a_factory": 0}

    def stub_model_a_factory(**kwargs):
        calls["model_a_factory"] += 1
        return SimpleNamespace(predict=lambda image, **kw: [])

    original_path = (PROJECT_ROOT / "configs/experiments/"
                     "material_runtime_v2_mask.yaml")
    original_bytes = original_path.read_text(encoding="utf-8")
    assert "upsample_version" not in original_bytes,         "the existing material runtime config must stay byte-identical"

    import yaml
    cfg = yaml.safe_load(original_bytes)
    cfg.setdefault("stereo", {})["upsample_version"] =         "valid_weighted_legacy_support_v1"
    copy_path = tmp_path / "material_runtime_fix_copy.yaml"
    copy_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    engine = DualStageInferenceEngine.from_material_config(
        copy_path, variant="gray_polar", seed=2026, device="cpu",
        model_a_factory=stub_model_a_factory)
    assert calls["model_a_factory"] == 1,         "Model A must come from the stub factory, never the real loader"
    assert engine.matcher.config.upsample_version ==         "valid_weighted_legacy_support_v1"

    # the unmodified config defaults to legacy
    engine = DualStageInferenceEngine.from_material_config(
        original_path, variant="gray_polar", seed=2026, device="cpu",
        model_a_factory=stub_model_a_factory)
    assert engine.matcher.config.upsample_version == "legacy_zero_linear_v1"


def test_from_material_config_rejects_unknown_upsample_version(
        tmp_path: Path):
    """The second REAL config entry must refuse an unknown
    stereo.upsample_version through from_material_config itself — not by
    constructing StereoMatcherConfig directly and not via from_config.
    The existing material runtime config is COPIED to tmp_path with ONLY
    stereo.upsample_version set to a clearly nonexistent value; Model A
    stays the controlled stub factory (never really loaded or run); the
    original config file must remain byte-identical."""
    from gui.inference_engine import DualStageInferenceEngine
    import yaml

    calls = {"model_a_factory": 0}

    def stub_model_a_factory(**kwargs):
        calls["model_a_factory"] += 1
        return SimpleNamespace(predict=lambda image, **kw: [])

    original_path = (PROJECT_ROOT / "configs/experiments/"
                     "material_runtime_v2_mask.yaml")
    original_bytes = original_path.read_bytes()

    cfg = yaml.safe_load(original_path.read_text(encoding="utf-8"))
    cfg.setdefault("stereo", {})["upsample_version"] = \
        "definitely_unknown_upsample_version"
    copy_path = tmp_path / "material_runtime_unknown_version_copy.yaml"
    copy_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown upsample_version"):
        DualStageInferenceEngine.from_material_config(
            copy_path, variant="gray_polar", seed=2026, device="cpu",
            model_a_factory=stub_model_a_factory)
    # Model A only ever came from the controlled stub (at most one call,
    # before the matcher config is validated); no engine was built and no
    # real Model A was loaded or run
    assert calls["model_a_factory"] <= 1
    # the existing runtime config on disk is untouched
    assert original_path.read_bytes() == original_bytes




def _project_scratch(prefix: str) -> Path:
    """A UNIQUE scratch subdir under the repo's ignored .test_artifacts/
    (the registry only registers paths inside the project root); the
    caller removes it in a finally block."""
    base = PROJECT_ROOT / ".test_artifacts"
    base.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=base))


def test_source_registry_conflict_anchor_kind_and_missing_at_end():
    """Registrations dedupe by path; a conflicting frozen expectation on an
    already-registered path is rejected; anchors register as anchors
    (never mislabeled as this batch's snapshot); a file deleted after
    registration becomes a recorded manifest problem instead of an
    exception that would hide the evidence."""
    scratch = _project_scratch("registry_guard")
    try:
        anchor_file = scratch / "anchor_like.bin"
        anchor_file.write_bytes(b"anchor-bytes")
        snapshot_file = scratch / "snapshot.bin"
        snapshot_file.write_bytes(b"snapshot-bytes")
        rel_anchor = anchor_file.relative_to(PROJECT_ROOT).as_posix()
        rel_snapshot = snapshot_file.relative_to(PROJECT_ROOT).as_posix()
        registry_cls = getattr(impact, "SourceRegistry")
        registry = registry_cls({
        })
        registry.bindings[rel_anchor] = {
            "source": "config.anchors[test_anchor]",
            "expected_sha256": getattr(impact, "sha256_of_file")(anchor_file)}
        anchor_record = registry.register(
            anchor_file, "anchor:test_anchor", kind="anchor")
        assert anchor_record["registration"] == "anchor"
        snapshot_record = registry.register(snapshot_file, "snapshot:test")
        assert snapshot_record["registration"] == "batch2_snapshot"
        # dedupe by path returns the SAME record (no second identity)
        assert registry.register(
            anchor_file, "anchor:test_anchor") is anchor_record
        # a conflicting expectation for the same path is refused
        registry.bindings[rel_anchor] = {
            "source": "later_pin", "expected_sha256": "0" * 64}
        with pytest.raises(getattr(impact, "IdentityError"),
                           match="binding conflict"):
            registry.register(anchor_file, "anchor:test_anchor")
        protection = registry.compare()
        kinds = {r["path"]: r["binding_kind"]
                  for r in protection["rows"]}
        assert kinds[rel_anchor] == "anchor_or_frozen"
        assert kinds[rel_snapshot] == "batch2_snapshot"
        assert protection["identical"] is True
        # a file deleted after registration is RECORDED, never hidden
        snapshot_file.unlink()
        protection = registry.compare()
        assert protection["identical"] is False
        assert any("missing_at_end" in p
                   for p in protection["problems"])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _mutation_world():
    """A small two-arm world computed through the REAL production
    aggregate and the REAL independent recomputer, so the comparator is
    exercised against genuine structures."""
    frames = [
        {"frame_id": "FA", "gt": [{"gt_id": 0,
                                         "class_id": 0},
                                       {"gt_id": 1,
                                         "class_id": 1}],
          "predictions": [{"pred_instance_id": 0},
                              {"pred_instance_id": 1}],
          "matches": [[0, 0, 0.9], [1, 1, 0.8]]},
        {"frame_id": "FB", "gt": [{"gt_id": 0,
                                         "class_id": 3}],
          "predictions": [{"pred_instance_id": 0}],
          "matches": []},
    ]
    rows = []
    for arm in ("legacy", "fix"):
        rows += [
            {"arm": arm, "candidate_id": "R0", "frame_id": "FA",
              "pred_instance_id": 0, "matched_gt_id": 0, "true_class": 0,
              "predicted_class": 0, "status": "valid",
              "used_fallback": False, "final_logits":
                  [2.0, 0.0, 0.0, 0.0]},
            {"arm": arm, "candidate_id": "R0", "frame_id": "FA",
              "pred_instance_id": 1, "matched_gt_id": 1, "true_class": 1,
              "predicted_class": 2, "status": "valid",
              "used_fallback": False, "final_logits":
                  [0.0, 0.0, 2.0, 0.0]},
            {"arm": arm, "candidate_id": "R0", "frame_id": "FB",
              "pred_instance_id": 0, "matched_gt_id": None,
              "true_class": None, "predicted_class": 1,
              "status": "valid", "used_fallback": False,
              "final_logits": [0.0, 2.0, 0.0, 0.0]}]
    records = [
        _build_record(frame_id="FA", group="G", gt_classes=[0, 1],
                      matches=[(0, 0, 0.9), (1, 1, 0.8)],
                      results=[_StubResult(0, [2.0, 0.0, 0.0, 0.0]),
                               _StubResult(2, [0.0, 0.0, 2.0, 0.0])],
                      unclassified=set(), matched_quality={}),
        _build_record(frame_id="FB", group="G", gt_classes=[3],
                      matches=[],
                      results=[_StubResult(1, [0.0, 2.0, 0.0, 0.0])],
                      unclassified=set(), matched_quality={}),
    ]
    metrics = {arm: {"R0": aggregate_metrics(
        records, list(getattr(impact, "CLASS_NAMES")))}
        for arm in ("legacy", "fix")}
    recompute = getattr(impact, "independent_recompute")
    independent = recompute(frames, rows, ["R0"], ("legacy", "fix"))
    production_fallback = {
        arm: {"R0": {"used_fallback": 0, "predictions": 3}}
        for arm in ("legacy", "fix")}
    return metrics, independent, production_fallback


def test_comparator_flags_mutations_even_when_macro_f1_survives():
    """Only the mutated field may differ: swapping whole per-class e2e
    vectors keeps the macro F1 (same multiset) yet must be flagged, and so
    must support, per-class tp/fp/fn, confusion and fallback changes."""
    metrics, independent, production_fallback = _mutation_world()
    names = list(getattr(impact, "CLASS_NAMES"))
    compare = getattr(impact, "compare_production_independent")
    base = compare(metrics, independent, production_fallback, ["R0"])
    assert base["mismatches"] == []
    assert base["checked"] > 0
    # (1) per-class e2e vectors swapped: macro F1 UNCHANGED, comparator
    # flags the per-class tp/fp/fn vectors
    swapped = copy.deepcopy(metrics)
    per_class = swapped["legacy"]["R0"]["end_to_end"]["per_class"]
    per_class[names[0]], per_class[names[1]] = (
        per_class[names[1]], per_class[names[0]])
    out = compare(swapped, independent, production_fallback, ["R0"])
    assert any("end_to_end.per_class" in m for m in out["mismatches"]), \
        out["mismatches"]
    # (2) a support count silently bumped
    bumped = copy.deepcopy(metrics)
    bumped["fix"]["R0"]["counts"]["predictions"] += 1
    out = compare(bumped, independent, production_fallback, ["R0"])
    assert any("support.predictions" in m for m in out["mismatches"])
    # (3) a conditional per-class fn bumped
    fn = copy.deepcopy(metrics)
    fn["legacy"]["R0"]["conditional_classification"]["per_class"][
        names[3]]["fn"] += 1
    out = compare(fn, independent, production_fallback, ["R0"])
    assert any("conditional.per_class" in m for m in out["mismatches"])
    # (4) the confusion matrix silently edited
    conf = copy.deepcopy(metrics)
    conf["fix"]["R0"]["conditional_classification"]["confusion_matrix"][0][1] += 1
    out = compare(conf, independent, production_fallback, ["R0"])
    assert any("confusion" in m for m in out["mismatches"])
    # (5) the fallback tally changed although every F1 is untouched
    fb = copy.deepcopy(production_fallback)
    fb["fix"]["R0"]["used_fallback"] += 1
    out = compare(metrics, independent, fb, ["R0"])
    assert any("fallback.used_fallback" in m for m in out["mismatches"])
    # (6) the fallback denominator changed
    fb2 = copy.deepcopy(production_fallback)
    fb2["legacy"]["R0"]["predictions"] += 1
    out = compare(metrics, independent, fb2, ["R0"])
    assert any("fallback.predictions" in m for m in out["mismatches"])


class _ExplodingAtWatch:
    """A BudgetWatch stand-in that raises exactly at one checkpoint."""

    def __init__(self, explode_at: str) -> None:
        self._explode_at = explode_at

    def check(self, at: str = "") -> None:
        if at == self._explode_at:
            raise getattr(impact, "BudgetExceeded")(
                f"fake budget exhausted at {at}")

    def remaining_s(self) -> float:
        return 0.0


@pytest.mark.parametrize("explode_at", [
    "before protection snapshot",
    "after protection snapshot",
    "before finalize",
])
def test_publish_phase_budget_overrun_stays_partial_and_keeps_evidence(
        tmp_path: Path, explode_at: str) -> None:
    """A budget overrun at any publish checkpoint (before the protection
    snapshot, after it, before finalize) must force PARTIAL - a run whose
    budget dies during publish is never allowed to report SUCCESS - and
    the manifest, the status and the earlier prediction evidence must all
    be on disk afterwards."""
    out = tmp_path / "run_out"
    out.mkdir(parents=True)
    (out / "predictions_arms.jsonl").write_text(
        '{"arm": "legacy"}\n', encoding="utf-8")
    (out / "frames_arms.jsonl").write_text(
        '{"frame_id": "F1"}\n', encoding="utf-8")
    scratch = _project_scratch("publish_guard")
    try:
        registry_cls = getattr(impact, "SourceRegistry")
        registry = registry_cls({})
        for name in ("pinned_a.bin", "pinned_b.bin"):
            source = scratch / name
            source.write_bytes(name.encode())
            registry.register(source, f"source:{name}")
        ctx = getattr(impact, "RunContext")(
            config={}, output_dir=out, mode="evaluate", device="cpu",
            budget_minutes=1.0, max_frames=None, run_id="publish_test",
            watch=_ExplodingAtWatch(explode_at), registry=registry)
        closeout = getattr(impact, "closeout_and_finalize")
        code = closeout(ctx, interrupted=False, failure=None)
        assert code == getattr(impact, "EXIT_PARTIAL")
        status = json.loads((out / "run_status.json").read_text(
            encoding="utf-8"))
        assert status["overall_status"] == "PARTIAL"
        assert status["interrupted_by_budget"] is True
        manifest = json.loads((out / "source_manifest.json").read_text(
            encoding="utf-8"))
        assert manifest["registered_sources"] == 2
        assert manifest["identical"] is True
        assert (out / "predictions_arms.jsonl").read_text(
            encoding="utf-8").startswith('{"arm"')
        assert (out / "frames_arms.jsonl").exists()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
