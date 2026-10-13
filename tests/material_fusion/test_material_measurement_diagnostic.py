"""material_measurement_diagnostic_v1 tests.

Everything here is synthetic and CPU-only: no model weights, no real
datasets, no network. The synthetic entry world (``build_world``) lives in
this test's scratch under ``.test_artifacts``; the real frozen evidence is
never touched. Analytic expectations are computed inside the tests from
first principles, never from the code under test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.diagnose_material_measurement as diag_mod
from core.measurement_diagnostic import (
    IdentityError,
    PRODUCTION_UPSAMPLE,
    SelectionConflict,
    build_correspondence_pair,
    edge_unmeasurable_mask,
    instance_boundary_split,
    instance_paired_comparison,
    load_and_merge_p0_oracles,
    pipeline_gt_prediction_join,
    residual_capacity_stats,
    run_sgbm_lowres,
    select_diagnostic_frames,
    shift_right_image,
    upsample_lowres,
    LowResStereoCore,
)
from core.stereo_matching import StereoMatcher, StereoMatcherConfig

PROJECT_ROOT = Path(diag_mod.__file__).resolve().parents[1]

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")


def _logits(true: int, pred: int, m: float) -> list[float]:
    """4-class logits with argmax=pred and logit[pred]-logit[true]=m."""
    values = [-9.0, -9.0, -9.0, -9.0]
    values[true] = -1.0
    values[pred] = -1.0 + m
    if pred == true:
        values[true] = 5.0
    return [float(v) for v in values]


def _oracle_rows(samples: list[dict]) -> list[dict]:
    rows = []
    for sample in samples:
        correct = sample["pred"] == sample["true"]
        logits = ([5.0 if i == sample["pred"] else -5.0 for i in range(4)]
                  if correct else _logits(sample["true"], sample["pred"],
                                          sample["m"]))
        rows.append({
            "class_id": sample["true"],
            "group": sample["group"],
            "logits": logits,
            "prediction": sample["pred"],
            "sample_name": sample["sample_name"],
        })
    return rows


# ---------------------------------------------------------------------------
# Residual capacity statistics (task A core)


def _capacity_fixture():
    samples = [
        # fold0: 4 rows, 2 errors (one m<1, one m>=1 and missing S)
        {"fold": "fold0", "sample_name": "s0", "group": "g", "true": 0,
         "pred": 0},
        {"fold": "fold0", "sample_name": "s1", "group": "g", "true": 2,
         "pred": 3, "m": 0.5},
        {"fold": "fold0", "sample_name": "s2", "group": "g", "true": 1,
         "pred": 3, "m": 2.0},
        {"fold": "fold0", "sample_name": "s3", "group": "g", "true": 3,
         "pred": 3},
        # fold1: 4 rows, 1 error m<1
        {"fold": "fold1", "sample_name": "s4", "group": "g", "true": 1,
         "pred": 1},
        {"fold": "fold1", "sample_name": "s5", "group": "g", "true": 0,
         "pred": 2, "m": 0.25},
        {"fold": "fold1", "sample_name": "s6", "group": "g", "true": 2,
         "pred": 2},
        {"fold": "fold1", "sample_name": "s7", "group": "g", "true": 3,
         "pred": 0, "m": 0.75},
        # fold2: 4 rows, all correct
        {"fold": "fold2", "sample_name": "s8", "group": "g", "true": 0,
         "pred": 0},
        {"fold": "fold2", "sample_name": "s9", "group": "g", "true": 1,
         "pred": 1},
        {"fold": "fold2", "sample_name": "sa", "group": "g", "true": 2,
         "pred": 2},
        {"fold": "fold2", "sample_name": "sb", "group": "g", "true": 3,
         "pred": 3},
    ]
    manifest = []
    for sample in samples:
        manifest.append({
            "sample_name": sample["sample_name"],
            "class_id": sample["true"],
            "class_name": CLASS_NAMES[sample["true"]],
            "group": sample["group"],
            "fold_roles": {f: "outer_eval" if f == sample["fold"]
                           else "inner_train" for f in
                           ("fold0", "fold1", "fold2")},
            "has_valid_s": not (sample["sample_name"] == "s2"),
            "source_frame": "frame_" + sample["sample_name"],
        })
    fold_files = {}
    for fold in ("fold0", "fold1", "fold2"):
        fold_files[fold] = _oracle_rows(
            [s for s in samples if s["fold"] == fold])
    return samples, manifest, fold_files


def test_residual_capacity_counts_match_hand_computed():
    samples, manifest, fold_files = _capacity_fixture()
    rows = load_and_merge_p0_oracles(fold_files, manifest)
    for row in rows:
        row["has_valid_s"] = next(
            s["has_valid_s"] for s in manifest
            if s["sample_name"] == row["sample_name"])
    stats = residual_capacity_stats(rows)
    assert stats["totals"]["samples"] == 12
    assert stats["totals"]["errors"] == 4
    assert stats["totals"]["m_ge_1"] == 1
    assert stats["totals"]["m_lt_1"] == 3
    assert stats["totals"]["errors_with_missing_polar"] == 1
    assert stats["totals"]["m_lt_1_with_valid_polar"] == 3
    assert stats["per_fold"]["fold0"] == {
        "samples": 4, "errors": 2, "m_ge_1": 1}
    assert stats["per_fold"]["fold1"] == {
        "samples": 4, "errors": 2, "m_ge_1": 0}
    # every error record carries fold/group/class identity
    for record in stats["per_error_records"]:
        assert record["fold"].startswith("fold")
        assert record["group"] == "g"
        assert record["true_class_name"] == CLASS_NAMES[record["true_class"]]


def test_capacity_m_is_exact_logit_gap():
    samples, manifest, fold_files = _capacity_fixture()
    rows = load_and_merge_p0_oracles(fold_files, manifest)
    by_name = {s["sample_name"]: s for s in manifest}
    for row in rows:
        row["has_valid_s"] = bool(by_name[row["sample_name"]]["has_valid_s"])
    stats = residual_capacity_stats(rows)
    records = {e["sample_name"]: e for e in stats["per_error_records"]}
    assert set(records) == {"s1", "s2", "s5", "s7"}
    for name, sample in (("s1", 0.5), ("s2", 2.0), ("s5", 0.25), ("s7", 0.75)):
        assert records[name]["m"] == pytest.approx(sample)
        assert records[name]["m_ge_1"] is (sample >= 1.0)
    assert records["s2"]["has_valid_s"] is False
    assert records["s1"]["has_valid_s"] is True
    assert stats["m_statistics_on_errors"]["n"] == 4
    assert stats["m_statistics_on_errors"]["max_abs"] == pytest.approx(2.0)
    assert stats["m_statistics_on_errors"]["p50"] == pytest.approx(0.625)


def test_merge_refuses_duplicate_sample_names_across_folds():
    _, manifest, fold_files = _capacity_fixture()
    fold_files["fold1"] = list(fold_files["fold1"]) + [dict(fold_files["fold0"][0])]
    with pytest.raises(IdentityError, match="duplicate sample_name"):
        load_and_merge_p0_oracles(fold_files, manifest)


def test_merge_refuses_argmax_prediction_mismatch():
    _, manifest, fold_files = _capacity_fixture()
    row = dict(fold_files["fold0"][0])
    row["prediction"] = (row["prediction"] + 1) % 4
    fold_files["fold0"] = [row]
    with pytest.raises(IdentityError, match="argmax"):
        load_and_merge_p0_oracles(fold_files, manifest)


def test_merge_refuses_nonfinite_logits():
    _, manifest, fold_files = _capacity_fixture()
    row = dict(fold_files["fold0"][0])
    row["logits"] = [float("nan"), 0.0, 0.0, 0.0]
    fold_files["fold0"] = [row]
    with pytest.raises(IdentityError, match="non-finite"):
        load_and_merge_p0_oracles(fold_files, manifest)


def test_merge_refuses_wrong_fold_role():
    _, manifest, _ = _capacity_fixture()
    samples, _, fold_files = _capacity_fixture()
    # move s0 (a fold0 outer_eval row) into the fold1 file
    fold_files["fold1"] = [dict(fold_files["fold0"][0])] + fold_files["fold1"]
    fold_files["fold0"] = fold_files["fold0"][1:]
    with pytest.raises(IdentityError, match="outer_eval role"):
        load_and_merge_p0_oracles(fold_files, manifest)


def test_merge_refuses_class_out_of_range():
    _, manifest, fold_files = _capacity_fixture()
    row = dict(fold_files["fold0"][0])
    row["class_id"] = 4
    fold_files["fold0"] = [row]
    with pytest.raises(IdentityError, match="class_id"):
        load_and_merge_p0_oracles(fold_files, manifest)


def test_merge_refuses_row_absent_from_manifest():
    _, manifest, fold_files = _capacity_fixture()
    extra = dict(fold_files["fold0"][0])
    extra["sample_name"] = "ghost"
    fold_files["fold0"] = [extra]
    with pytest.raises(IdentityError, match="feature manifest"):
        load_and_merge_p0_oracles(fold_files, manifest)


# ---------------------------------------------------------------------------
# Pipeline GT/R0 join (task A, production side)


def _join_fixture():
    frames = [
        {"frame_id": "F1", "group": "GB", "fold": "fold0", "frame_error": None,
         "gt": [{"gt_id": 0, "class_id": 2}, {"gt_id": 1, "class_id": 1}],
         "predictions": [{"pred_instance_id": 0}, {"pred_instance_id": 1}],
         "matches": [[0, 0, 0.8], [1, 1, 0.9]],
         "missed_gt_ids": [], "unmatched_pred_ids": []},
        {"frame_id": "F2", "group": "GB", "fold": "fold0", "frame_error": None,
         "gt": [{"gt_id": 0, "class_id": 3}],
         "predictions": [{"pred_instance_id": 0}],
         "matches": [], "missed_gt_ids": [0], "unmatched_pred_ids": [0]},
        {"frame_id": "F3", "group": "GA", "fold": "fold1", "frame_error": None,
         "gt": [{"gt_id": 0, "class_id": 0}],
         "predictions": [{"pred_instance_id": 0}],
         "matches": [[0, 0, 0.7]],
         "missed_gt_ids": [], "unmatched_pred_ids": []},
    ]
    r0 = [
        {"candidate_id": "R0", "frame_id": "F1", "pred_instance_id": 0,
         "fold": "fold0", "group": "GB", "matched_gt_id": 0,
         "final_logits": _logits(2, 2, 0.0), "predicted_class": 2,
         "used_fallback": False, "fallback_reason": ""},
        {"candidate_id": "R0", "frame_id": "F1", "pred_instance_id": 1,
         "fold": "fold0", "group": "GB", "matched_gt_id": 1,
         "final_logits": _logits(1, 2, 0.5), "predicted_class": 2,
         "used_fallback": False, "fallback_reason": ""},
        {"candidate_id": "R0", "frame_id": "F2", "pred_instance_id": 0,
         "fold": "fold0", "group": "GB", "matched_gt_id": None,
         "final_logits": _logits(3, 3, 0.0), "predicted_class": 3,
         "used_fallback": False, "fallback_reason": ""},
        {"candidate_id": "R0", "frame_id": "F3", "pred_instance_id": 0,
         "fold": "fold1", "group": "GA", "matched_gt_id": 0,
         "final_logits": _logits(0, 1, 2.0), "predicted_class": 1,
         "used_fallback": False, "fallback_reason": ""},
    ]
    return frames, r0


def test_join_counts_and_error_m_values():
    frames, r0 = _join_fixture()
    join = pipeline_gt_prediction_join(frames, r0)
    t = join["totals"]
    assert t["frames"] == 3
    assert t["gt_instances"] == 4
    assert t["predictions"] == 4
    assert t["matches"] == 3
    assert t["matched_r0_joined"] == 3
    assert t["errors"] == 2
    assert t["m_ge_1"] == 1
    assert t["m_lt_1"] == 1
    errors = {(e["frame_id"], e["gt_id"]): e for e in join["errors"]}
    assert errors[("F1", 1)]["m"] == pytest.approx(0.5)
    assert errors[("F1", 1)]["m_ge_1"] is False
    assert errors[("F3", 0)]["m"] == pytest.approx(2.0)
    assert errors[("F3", 0)]["m_ge_1"] is True


def test_join_refuses_matched_gt_id_inconsistent_with_saved_match():
    frames, r0 = _join_fixture()
    r0[1] = {**r0[1], "matched_gt_id": 0}
    with pytest.raises(IdentityError, match="saved match"):
        pipeline_gt_prediction_join(frames, r0)


def test_join_refuses_matched_pred_without_match_record():
    frames, r0 = _join_fixture()
    # make F2 self-consistently contain a second prediction whose saved
    # match takes gt0, then let the R0 row of pred 0 claim gt0 as well
    frames[1]["predictions"].append({"pred_instance_id": 1})
    frames[1]["matches"] = [[0, 1, 0.5]]
    frames[1]["missed_gt_ids"] = []
    frames[1]["unmatched_pred_ids"] = [0]
    r0.append({"candidate_id": "R0", "frame_id": "F2", "pred_instance_id": 1,
               "fold": "fold0", "group": "GB", "matched_gt_id": 0,
               "final_logits": _logits(3, 3, 0.0), "predicted_class": 3,
               "used_fallback": False, "fallback_reason": ""})
    r0[2] = {**r0[2], "matched_gt_id": 0}
    with pytest.raises(IdentityError, match="no such match"):
        pipeline_gt_prediction_join(frames, r0)


def test_join_refuses_duplicate_r0_row():
    frames, r0 = _join_fixture()
    with pytest.raises(IdentityError, match="duplicate R0"):
        pipeline_gt_prediction_join(frames, r0 + [r0[0]])


def test_join_refuses_non_one_to_one_matches():
    frames, r0 = _join_fixture()
    frames[0]["matches"] = [[0, 0, 0.8], [0, 1, 0.5]]
    with pytest.raises(IdentityError, match="one-to-one"):
        pipeline_gt_prediction_join(frames, r0)


def test_join_refuses_argmax_mismatch():
    frames, r0 = _join_fixture()
    row = dict(r0[0])
    row["predicted_class"] = 1
    with pytest.raises(IdentityError, match="argmax"):
        pipeline_gt_prediction_join(frames, [row])


# ---------------------------------------------------------------------------
# Analytic upsample counterexamples (task B) — independent expectations


def _bilinear_weights(src_len: int, dst_len: int) -> list[list[tuple[int, float]]]:
    """Test-local INTER_LINEAR reference (OpenCV convention:
    src = (dst + 0.5) * src/dst - 0.5, border-clamped)."""
    taps = []
    ratio = src_len / dst_len
    for dst in range(dst_len):
        src = (dst + 0.5) * ratio - 0.5
        src = min(max(src, 0.0), src_len - 1.0)
        i0 = int(math.floor(src))
        frac = src - i0
        i1 = min(i0 + 1, src_len - 1)
        if frac == 0.0:
            taps.append([(i0, 1.0)])
        else:
            taps.append([(i0, 1.0 - frac), (i1, frac)])
    return taps


def test_analytic_taps_match_hand_computed_values():
    # upscale 2 -> 4 with src = (dst+0.5)/2 - 0.5, border-clamped to
    # [0, src_len-1]:
    # dst 0 -> src -0.25 (clamped to 0) -> single tap (0, 1.0);
    # dst 1 -> src 0.25 -> (0, 0.75), (1, 0.25);
    # dst 2 -> src 0.75 -> (0, 0.25), (1, 0.75);
    # dst 3 -> src 1.25 (clamped to 1) -> single tap (1, 1.0)
    taps = _bilinear_weights(2, 4)
    assert taps[0] == [(0, 1.0)]
    assert taps[1] == [(0, 0.75), (1, 0.25)]
    assert taps[2] == [(0, 0.25), (1, 0.75)]
    assert taps[3] == [(1, 1.0)]


def test_production_upsample_biases_valid_pixels_exactly_as_predicted():
    sh, sw, scale = 6, 24, 0.25
    out_h, out_w = sh * 4, sw * 4
    d0 = 6.0
    disp_small = np.full((sh, sw), d0 * scale, dtype=np.float32)
    valid_small = np.ones((sh, sw), dtype=np.uint8)
    valid_small[2:4, 12:14] = 0
    disp_small[valid_small == 0] = 0.0
    core = LowResStereoCore(disp_small, valid_small, 1.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, out_w, out_h, scale,
                                     method="production")
    # independent prediction: disp_full = d0 * (valid weight mass) where the
    # nearest low-res pixel (cv2 INTER_NEAREST convention j = x*sw//out_w)
    # is valid
    wx = _bilinear_weights(sw, out_w)
    wy = _bilinear_weights(sh, out_h)
    expected_disp = np.zeros((out_h, out_w))
    expected_valid = np.zeros((out_h, out_w), dtype=bool)
    for y in range(out_h):
        jy = y * sh // out_h
        for x in range(out_w):
            jx = x * sw // out_w
            mass = 0.0
            for iy, wy_ in wy[y]:
                for ix, wx_ in wx[x]:
                    mass += wy_ * wx_ * float(valid_small[iy, ix])
            expected_disp[y, x] = d0 * mass
            expected_valid[y, x] = bool(valid_small[jy, jx])
    marked = expected_valid
    biased = marked & (np.abs(expected_disp - d0) > 1e-9)
    assert biased.sum() > 0, "the scenario must contain tainted valid pixels"
    measured_err = np.abs(prod_d.astype(np.float64)[biased] -
                          expected_disp[biased])
    assert measured_err.max() < 2e-3  # float32 resize precision
    np.testing.assert_array_equal(prod_v > 0, expected_valid)
    max_bias = float(np.max(d0 - expected_disp[biased]))
    assert prod_d[biased].min() == pytest.approx(d0 - max_bias, abs=2e-3)
    # with a 2-column invalid block the worst TAINTED-MARKED stencil keeps
    # 5/8 of its weight on valid taps -> bias = d0 * 3/8 (a full-weight zero
    # tap would make the nearest neighbour invalid, so it is not marked)
    assert max_bias == pytest.approx(d0 * 0.375, abs=2e-3)


def test_control_restores_plane_at_tainted_valid_pixels():
    sh, sw, scale = 6, 24, 0.25
    out_h, out_w = sh * 4, sw * 4
    d0 = 6.0
    disp_small = np.full((sh, sw), d0 * scale, dtype=np.float32)
    valid_small = np.ones((sh, sw), dtype=np.uint8)
    valid_small[2:4, 12:14] = 0
    disp_small[valid_small == 0] = 0.0
    core = LowResStereoCore(disp_small, valid_small, 1.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, out_w, out_h, scale,
                                     method="production")
    ctrl_d, ctrl_v = upsample_lowres(core, out_w, out_h, scale,
                                     method="valid_weighted", w_min=0.5)
    wx = _bilinear_weights(sw, out_w)
    wy = _bilinear_weights(sh, out_h)
    tainted = np.zeros((out_h, out_w), dtype=bool)
    for y in range(out_h):
        for x in range(out_w):
            if prod_v[y, x] == 0:
                continue
            touches = any(valid_small[iy, ix] == 0
                          for iy, _ in wy[y] for ix, _ in wx[x])
            mass = sum(wy_ * wx_ * float(valid_small[iy, ix])
                       for iy, wy_ in wy[y] for ix, wx_ in wx[x])
            tainted[y, x] = touches and mass < 1.0 - 1e-9
    assert tainted.sum() > 0
    # control restores the exact plane where its own weight suffices
    ctrl_mass = np.zeros((out_h, out_w))
    for y in range(out_h):
        for x in range(out_w):
            ctrl_mass[y, x] = sum(
                wy_ * wx_ * float(valid_small[iy, ix])
                for iy, wy_ in wy[y] for ix, wx_ in wx[x])
    measurable = tainted & (ctrl_v > 0)
    assert measurable.sum() > 0
    np.testing.assert_allclose(ctrl_d[measurable], d0, atol=1e-4)
    # where the valid weight is below w_min the control refuses to measure
    ctrl_mass = np.zeros((out_h, out_w))
    for y in range(out_h):
        for x in range(out_w):
            ctrl_mass[y, x] = sum(
                wy_ * wx_ * float(valid_small[iy, ix])
                for iy, wy_ in wy[y] for ix, wx_ in wx[x])
    refused = tainted & (ctrl_v == 0)
    if refused.any():
        assert np.all(ctrl_mass[refused] < 0.5)


def test_fully_valid_plane_is_exact_for_both_methods():
    sh, sw, scale = 4, 16, 0.25
    d0 = 6.0
    core = LowResStereoCore(
        np.full((sh, sw), d0 * scale, np.float32),
        np.ones((sh, sw), np.uint8), 1.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, sw * 4, sh * 4, scale,
                                     method="production")
    ctrl_d, ctrl_v = upsample_lowres(core, sw * 4, sh * 4, scale,
                                     method="valid_weighted", w_min=0.5)
    assert int(prod_v.sum()) == sh * 4 * sw * 4
    np.testing.assert_allclose(prod_d, d0, atol=1e-4)
    np.testing.assert_allclose(ctrl_d, d0, atol=1e-4)
    assert int(ctrl_v.sum()) == sh * 4 * sw * 4


def test_valid_valid_discontinuity_is_identical_for_both_methods():
    sh, sw, scale = 4, 16, 0.25
    disp_small = np.full((sh, sw), 1.5, np.float32)
    disp_small[:, sw // 2:] = 3.0
    core = LowResStereoCore(disp_small, np.ones((sh, sw), np.uint8),
                            1.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, sw * 4, sh * 4, scale,
                                     method="production")
    ctrl_d, ctrl_v = upsample_lowres(core, sw * 4, sh * 4, scale,
                                     method="valid_weighted", w_min=0.5)
    np.testing.assert_array_equal(prod_d, ctrl_d)
    np.testing.assert_array_equal(prod_v, ctrl_v)
    # both mix the two planes near the boundary: the control does NOT
    # repair valid-valid boundary mixing
    boundary = np.abs(ctrl_d.astype(np.float64) - 6.0) > 1e-6
    assert boundary.sum() > 0


def test_all_invalid_region_fabricates_no_pixels():
    sh, sw, scale = 4, 16, 0.25
    core = LowResStereoCore(np.zeros((sh, sw), np.float32),
                            np.zeros((sh, sw), np.uint8), 0.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, sw * 4, sh * 4, scale,
                                     method="production")
    ctrl_d, ctrl_v = upsample_lowres(core, sw * 4, sh * 4, scale,
                                     method="valid_weighted", w_min=0.5)
    assert int(prod_v.sum()) == 0
    assert int(ctrl_v.sum()) == 0
    assert float(np.abs(prod_d).max()) == 0.0
    assert float(np.abs(ctrl_d).max()) == 0.0


def test_copied_core_matches_production_match_pair_bitwise():
    cfg = StereoMatcherConfig(max_disparity=32, scale=1.0, block_size=5,
                              texture_threshold=10.0)
    rng = np.random.default_rng(7)
    left = rng.integers(40, 220, size=(40, 80)).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, 10:] = left[:, :-10]
    matcher = StereoMatcher(cfg)
    prod_disp, prod_valid, _elapsed, prod_ratio = matcher._match_pair(
        left, right)
    core = run_sgbm_lowres(left, right, cfg)
    diag_disp, diag_valid = upsample_lowres(core, 80, 40, cfg.scale,
                                            method="production")
    np.testing.assert_array_equal(prod_disp, diag_disp)
    np.testing.assert_array_equal(prod_valid, diag_valid)
    assert prod_ratio == core.valid_ratio


def test_shift_zero_fills_and_marks_fill_pixels():
    right = np.arange(20, dtype=np.uint8).reshape(4, 5)
    shifted, fill = shift_right_image(right, 1, 0)
    assert shifted[0, 0] == 0 and shifted[0, 1] == right[0, 0]
    assert fill[:, 0].all() and not fill[:, 1:].any()
    shifted, fill = shift_right_image(right, -1, 0)
    assert shifted[0, -1] == 0 and shifted[0, -2] == right[0, -1]
    assert fill[:, -1].all()
    shifted, fill = shift_right_image(right, 0, 1)
    assert shifted[0, 0] == 0 and shifted[1, 0] == right[0, 0]
    assert fill[0, :].all()
    with pytest.raises(ValueError):
        shift_right_image(right, 1, 1)


def test_edge_unmeasurable_mask_marks_sample_landing_pixels():
    h, w = 4, 10
    disp = np.full((h, w), 3.0, np.float32)
    # dx=+1: fill column 0; in-bounds samples touch it when 0 <= x-d < 1,
    # i.e. exactly x = 3 for d = 3 (x < 3 is already out of bounds)
    mask = edge_unmeasurable_mask((h, w), disp, 1, 0)
    assert mask[:, 3].all() and not mask[:, :3].any() \
        and not mask[:, 4:].any()
    # dx=-1: fill column w-1; in-bounds samples touch it when x-d > w-2,
    # i.e. never for d=3 (max sx = w-1-d = 6 <= w-2)
    mask = edge_unmeasurable_mask((h, w), disp, -1, 0)
    assert not mask.any()
    disp_zero = np.zeros((h, w), np.float32)
    mask = edge_unmeasurable_mask((h, w), disp_zero, -1, 0)
    assert mask[:, w - 1].all() and not mask[:, :w - 1].any()
    # dy=+1 forbids row 0; dy=-1 forbids row h-1
    assert edge_unmeasurable_mask((h, w), disp, 0, 1)[0, :].all()
    assert not edge_unmeasurable_mask((h, w), disp, 0, 1)[1:, :].any()
    assert edge_unmeasurable_mask((h, w), disp, 0, -1)[h - 1, :].all()
    assert not edge_unmeasurable_mask((h, w), disp, 0, -1)[:h - 1, :].any()


def test_build_correspondence_pair_q_truth_is_exact_at_true_disparity():
    rng = np.random.default_rng(11)
    pattern = rng.integers(60, 200, size=(8, 40)).astype(np.float64)
    d0, c = 6.0, 0.6
    left, right, q_true = build_correspondence_pair(pattern, d0, c)
    from core.polar_compute import compute_polar_features
    disp = np.full(left.shape, d0, np.float32)
    valid = np.zeros(left.shape, np.uint8)
    valid[:, int(d0):] = 1
    result = compute_polar_features(left, right, disp, disparity_valid=valid)
    measured = result.signed_q.astype(np.float64)
    truth = np.nan_to_num(q_true, nan=0.0)
    np.testing.assert_allclose(measured, truth, atol=1e-6)


# ---------------------------------------------------------------------------
# Frozen selection (task C core)


def _selection_world():
    frames = []
    for i in range(6):
        frames.append({"frame_id": f"A_{i:03d}", "group": "GA",
                       "split": "train", "gt_instances": 3,
                       "gt_class_counts": {"0": 1, "1": 1, "2": 1}})
    for i in range(6):
        counts = {"3": 1, "1": 1} if i < 2 else {"1": 2}
        frames.append({"frame_id": f"B_{i:03d}", "group": "GB", "split": "val",
                       "gt_instances": sum(counts.values()),
                       "gt_class_counts": counts})
    for i in range(4):
        counts = {} if i == 3 else {"1": 1}
        frames.append({"frame_id": f"C_{i:03d}", "group": "GC", "split": "val",
                       "gt_instances": sum(counts.values()),
                       "gt_class_counts": counts})
    controls = ["B_000", "B_002"]
    missing = [
        {"sample_name": "mA", "class_name": "real_fish", "has_valid_s": False,
         "source_frame": "A_000"},
        {"sample_name": "mB", "class_name": "metal_submarine",
         "has_valid_s": False, "source_frame": "A_001"},
        {"sample_name": "mC", "class_name": "plastic_fish",
         "has_valid_s": False, "source_frame": "A_002"},
    ]
    return frames, controls, missing


def test_selection_covers_all_required_rules_deterministically():
    frames, controls, missing = _selection_world()
    sel = select_diagnostic_frames(frames, controls, missing,
                                   pipeline_m_lt1_frames=["B_001"],
                                   max_frames=64)
    by_id = {f["frame_id"]: f for f in sel["frames"]}
    # val real-fish frames
    for fid in ("B_000", "B_001"):
        assert fid in by_id
        assert "val_real_fish_gt" in by_id[fid]["reasons"]
    # no-target dev frame
    assert "no_target_dev" in by_id["C_003"]["reasons"]
    # control panels
    for fid in controls:
        assert "normal_control_panel" in by_id[fid]["reasons"]
    # missing-polar classes: one frame per class, first by sample_name
    assert "missing_polar_class_real_fish" in by_id["A_000"]["reasons"][0]
    assert "missing_polar_class_metal_submarine" in \
        by_id["A_001"]["reasons"][0]
    # pipeline m<1 frame
    assert "pipeline_m_below_1_error" in by_id["B_001"]["reasons"]
    # every group covered
    assert sel["groups_covered"] == 3
    # determinism
    again = select_diagnostic_frames(frames, controls, missing,
                                     pipeline_m_lt1_frames=["B_001"],
                                     max_frames=64)
    assert [f["frame_id"] for f in sel["frames"]] == \
        [f["frame_id"] for f in again["frames"]]


def test_selection_fills_to_cap_by_cycling_groups():
    frames, controls, missing = _selection_world()
    required = select_diagnostic_frames(frames, controls, missing,
                                        pipeline_m_lt1_frames=[],
                                        max_frames=64)
    assert required["required_count"] == 7  # C_003 already covers GC
    sel = select_diagnostic_frames(frames, controls, missing,
                                   pipeline_m_lt1_frames=[], max_frames=12)
    assert sel["selected_count"] == 12
    assert sel["required_count"] == required["required_count"]
    first_required = {
        f["frame_id"] for f in required["frames"]
        if any(not r.startswith("fill_group_coverage")
               for r in f["reasons"])}
    selected_ids = [f["frame_id"] for f in sel["frames"]]
    assert first_required <= set(selected_ids)
    # fills preserve group coverage balance: A_003 (next unselected of GA)
    assert "A_003" in selected_ids


def test_selection_conflict_when_required_exceeds_cap():
    frames, controls, missing = _selection_world()
    with pytest.raises(SelectionConflict, match="refusing to drop"):
        select_diagnostic_frames(frames, controls, missing,
                                 pipeline_m_lt1_frames=[], max_frames=3)


def test_selection_rejects_unknown_frames_and_bad_caps():
    frames, controls, missing = _selection_world()
    with pytest.raises(IdentityError):
        select_diagnostic_frames(frames, ["ghost_frame"], missing,
                                 pipeline_m_lt1_frames=[], max_frames=64)
    with pytest.raises(ValueError):
        select_diagnostic_frames(frames, controls, missing,
                                 pipeline_m_lt1_frames=[], max_frames=0)


# ---------------------------------------------------------------------------
# measure_condition: arms are actually invoked, Model A probe silent


def _textured_arm_world(seed: int):
    """A 32x64 blocky stereo pair whose instance has valid stereo stats.

    Precondition-guarded: the tests below are only meaningful when the
    engine reference actually measures something."""
    from models.segmentation import Instance

    cfg = StereoMatcherConfig(max_disparity=32, scale=1.0, block_size=5,
                              texture_threshold=10.0)
    rng = np.random.default_rng(seed)
    low = rng.integers(60, 200, (8, 20)).astype(np.uint8)
    left = np.repeat(np.repeat(low, 4, 0), 4, 1)[:, :64]
    right = np.zeros_like(left)
    right[:, 6:] = left[:, :-6]
    mask = np.zeros((32, 64), np.uint8)
    mask[8:24, 10:50] = 1
    instances = [Instance(id=0, bbox=(10, 8, 50, 24), mask=mask,
                          confidence=1.0, class_id=2, class_name="x")]
    stereo = StereoMatcher(cfg)
    result = stereo.compute(left, right)
    assert int(result.valid.sum()) > 0, "test pair must yield valid stereo"
    stats = stereo.instance_stats(result, mask, 0)
    assert stats.valid, f"test pair instance must be measurable: {stats.reason}"
    return cfg, left, right, instances


class _CountingMatcher(StereoMatcher):
    def __init__(self, config):
        super().__init__(config)
        self.compute_bands_calls = 0

    def compute_bands(self, left, right, bands, full_image_threshold=0.9):
        self.compute_bands_calls += 1
        return super().compute_bands(left, right, bands,
                                     full_image_threshold)


def test_measure_condition_runs_all_arms_with_engine_crosscheck():
    from gui.inference_engine import DualStageInferenceEngine

    import scripts.diagnose_material_measurement as dm
    cfg, left, right, instances = _textured_arm_world(3)
    probe = dm._ModelAProbe()
    engine = DualStageInferenceEngine(
        model_a=probe, model_b=None, matcher=_CountingMatcher(cfg),
        rectifier=None, rectify_enabled=False, device="cpu")
    counting = {"diag_production": 0, "diag_weighted": 0}
    real_diag = dm.DiagnosticUpsampleMatcher

    class CountingDiag(real_diag):
        def __init__(self, config, *, method=PRODUCTION_UPSAMPLE, w_min=0.5):
            super().__init__(config, method=method, w_min=w_min)
            counting["diag_production" if method == PRODUCTION_UPSAMPLE
                     else "diag_weighted"] += 1

    prod_calls = {"compute_bands": 0}
    real_prod = dm.StereoMatcher

    class CountingProd(real_prod):
        def compute_bands(self, left, right, bands, full_image_threshold=0.9):
            prod_calls["compute_bands"] += 1
            return super().compute_bands(left, right, bands,
                                         full_image_threshold)

    orig = (dm.StereoMatcher, dm.DiagnosticUpsampleMatcher)
    dm.StereoMatcher = CountingProd
    dm.DiagnosticUpsampleMatcher = CountingDiag
    try:
        result = dm._measure_condition(
            None, stereo_cfg=cfg, engine=engine, probe=probe,
            left=left, right=right, instances=instances, crop_windows=None,
            w_min=0.5, shifts=[(1, 0), (0, 1)], engine_check=True)
    finally:
        dm.StereoMatcher, dm.DiagnosticUpsampleMatcher = orig

    # probe: Model A was never called (record-then-raise would have thrown)
    assert probe.calls == []
    # arms: baseline diag-production 1, weighted 1; the production matcher
    # runs the engine reference once plus one full re-match per shift
    assert counting == {"diag_production": 1, "diag_weighted": 1}
    assert prod_calls["compute_bands"] == 3
    # the engine cross-check passed: identical measurement chain
    assert result["engine_reference"]["crosscheck"]["identical"] is True
    assert result["engine_reference"]["model_a_probe_calls"] == 0
    # the shift arms produced frame-context records plus per-instance
    # paired statistics
    assert set(result["shifts"]) == {"shift_dxp1_dyp0", "shift_dxp0_dyp1"}
    for record in result["shifts"].values():
        diff = (record["full_recompute"]["vs_baseline_frame_context"]
                ["q_common_valid"])
        assert diff["pixels"] >= 0
        assert "signed_q_diff" in diff
        assert record["fixed_baseline_warp"]["note"]
        # the edge masks must come from each arm's own disparity
        assert record["full_recompute"]["edge_mask_source"] == \
            "stereo3.disparity (re-matched)"
        assert record["fixed_baseline_warp"]["edge_mask_source"] == \
            "stereo0.disparity (baseline)"
        assert "edge_excluded_pixels" in record["full_recompute"]
        assert "edge_excluded_pixels" in record["fixed_baseline_warp"]
        for paired in record["full_recompute"]["paired"]:
            assert set(paired) >= {"q_valid_counts", "signed_dq", "dabs_q",
                                   "disparity_valid_counts",
                                   "disparity_diff", "interior", "boundary"}
    # the upsample control arm has frame context and paired statistics
    assert "vs_baseline_frame_context" in result["arms"]["upsample_control"]
    assert result["arms"]["upsample_control"]["paired"]


def test_engine_and_orchestration_diverge_if_upsample_changes():
    """A mutated control MUST break the bitwise cross-check, proving the
    cross-check actually compares the measurement chain (not a tautology)."""
    from gui.inference_engine import DualStageInferenceEngine

    import scripts.diagnose_material_measurement as dm
    cfg, left, right, instances = _textured_arm_world(5)
    probe = dm._ModelAProbe()
    engine = DualStageInferenceEngine(
        model_a=probe, model_b=None, matcher=dm.StereoMatcher(cfg),
        rectifier=None, rectify_enabled=False, device="cpu")

    class LyingBaseline(dm.DiagnosticUpsampleMatcher):
        def _match_pair_like(self, left_gray, right_gray):
            disp, valid, elapsed, ratio = super()._match_pair_like(
                left_gray, right_gray)
            # half a pixel is large enough to survive the uint8 remap
            # quantization and the 2-decimal depth record rounding
            return disp + np.float32(0.5), valid, elapsed, ratio

    orig = dm.DiagnosticUpsampleMatcher
    dm.DiagnosticUpsampleMatcher = LyingBaseline
    try:
        result = dm._measure_condition(
            None, stereo_cfg=cfg, engine=engine, probe=probe,
            left=left, right=right, instances=instances, crop_windows=None,
            w_min=0.5, shifts=[(1, 0)], engine_check=True)
    finally:
        dm.DiagnosticUpsampleMatcher = orig
    assert result["engine_reference"]["crosscheck"]["identical"] is False
    assert result["engine_reference"]["crosscheck"]["checks"][
        "signed_q_bitwise"] is False


# ---------------------------------------------------------------------------
# Synthetic entry world (no real evidence is ever touched)


class World:
    """A self-consistent miniature of every input the entry consumes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        rng = np.random.default_rng(2026)
        self.groups = {
            "GA": ("train", [f"A_{i:03d}" for i in range(6)]),
            "GB": ("val", [f"B_{i:03d}" for i in range(6)]),
            "GC": ("val", [f"C_{i:03d}" for i in range(4)]),
        }
        self.frames = {}
        for group, (split, stems) in self.groups.items():
            for stem in stems:
                if split == "train":
                    counts = {"0": 1, "1": 1, "2": 1}
                elif stem == "C_003":
                    counts = {}
                elif stem in ("B_000", "B_001"):
                    counts = {"3": 1, "1": 1}
                else:
                    counts = {"1": 2}
                self.frames[stem] = {"group": group, "split": split,
                                     "counts": counts}
        self._write_images(rng)
        self._write_labels()
        self._write_split_csv()
        self._write_protocol()
        self._write_manifest()
        self._write_runtime_evidence()
        self._write_detection()
        self._write_atlas()
        self._write_configs()

    # -- images / labels ----------------------------------------------------
    def _write_images(self, rng) -> None:
        import cv2

        for stem, info in self.frames.items():
            d = self.root / "images" / info["group"]
            d.mkdir(parents=True, exist_ok=True)
            base = rng.integers(40, 220, size=(64, 72)).astype(np.uint8)
            left = np.ascontiguousarray(base)
            right = np.zeros_like(left)
            right[:, 6:] = left[:, :-6]
            cv2.imwrite(str(d / f"{stem}_left.png"), left)
            cv2.imwrite(str(d / f"{stem}_right.png"), right)

    def _label_lines(self, stem: str) -> list[str]:
        counts = self.frames[stem]["counts"]
        lines = []
        slot = 0
        for cls in sorted(counts):
            for _ in range(counts[cls]):
                x0, y0 = 0.15 + 0.06 * slot, 0.25
                x1, y1 = x0 + 0.18, 0.75
                slot += 1
                lines.append(
                    f"{cls} {x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f} "
                    f"{x1:.3f} {y0:.3f} {x0:.3f} {y1:.3f}")
        return lines

    def _write_labels(self) -> None:
        for stem, info in self.frames.items():
            d = self.root / "labels" / info["split"]
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{stem}.txt").write_text(
                "\n".join(self._label_lines(stem)) + "\n", encoding="utf-8")

    def _image_paths(self, stem: str) -> tuple[str, str]:
        info = self.frames[stem]
        # project-root-relative, exactly like the real split CSV
        base = (self.root / "images" / info["group"] / stem).relative_to(
            PROJECT_ROOT).as_posix()
        return (base + "_left.png", base + "_right.png")

    def _write_split_csv(self) -> None:
        lines = ["output_stem,split,group_name,block_id,source_json,"
                 "left_path,right_path"]
        for stem, info in self.frames.items():
            left, right = self._image_paths(stem)
            lines.append(f"{stem},{info['split']},{info['group']},0,src.json,"
                         f"{left},{right}")
        (self.root / "split_assignment.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    # -- protocol / manifest -------------------------------------------------
    def _write_protocol(self) -> None:
        frames = []
        for stem, info in self.frames.items():
            frames.append({
                "frame_id": stem, "group": info["group"],
                "split": info["split"], "block_id": "0",
                "gt_instances": sum(info["counts"].values()),
                "gt_class_counts": dict(info["counts"]),
                "label_exists": True, "label_error": None,
            })
        protocol = {
            "protocol_sha256": "0" * 64,
            "data": {"class_names": list(CLASS_NAMES),
                     "dev_pool_splits": ["train", "val"]},
            "inventory": {"frames": frames},
        }
        d = self.root / "protocol"
        d.mkdir(parents=True, exist_ok=True)
        (d / "protocol.json").write_text(
            json.dumps(protocol, indent=1), encoding="utf-8")

    def _sample_rows(self) -> list[dict]:
        rows = []
        layout = {
            "fold0": ["sA_0", "sA_1", "sA_2", "sA_3"],
            "fold1": ["sB_0", "sB_1", "sB_2", "sB_3"],
            "fold2": ["sC_0", "sC_1", "sC_2", "sC_3"],
        }
        true_of = [0, 2, 1, 3, 1, 0, 2, 3, 0, 1, 2, 3]
        pred_of = {1: 3, 2: 3, 5: 2, 7: 0}
        m_of = {1: 0.5, 2: 2.0, 5: 0.25, 7: 0.75}
        index = 0
        for fold, names in layout.items():
            for name in names:
                true = true_of[index]
                pred = pred_of.get(index, true)
                rows.append({
                    "sample_name": name,
                    "class_id": true,
                    "class_name": CLASS_NAMES[true],
                    "group": "GA",
                    "fold_roles": {f: ("outer_eval" if f == fold else
                                       "inner_train")
                                   for f in ("fold0", "fold1", "fold2")},
                    "has_valid_s": index != 2,
                    "source_frame": "A_000",
                    "object_index": index % 3,
                    "split": "train",
                    "npz_path": f"train/{CLASS_NAMES[true]}/x.npz",
                })
                index += 1
        self._pred_of = pred_of
        self._m_of = m_of
        self._layout = layout
        return rows

    def _write_manifest(self) -> None:
        rows = self._sample_rows()
        d = self.root / "local_features"
        d.mkdir(parents=True, exist_ok=True)
        (d / "feature_manifest.json").write_text(
            json.dumps({"count": len(rows), "samples": rows}, indent=1),
            encoding="utf-8")

    def _oracle_row(self, index: int, row: dict) -> dict:
        pred = self._pred_of.get(index, row["class_id"])
        if pred == row["class_id"]:
            logits = [5.0 if i == pred else -5.0 for i in range(4)]
        else:
            logits = _logits(row["class_id"], pred, self._m_of[index])
        return {"class_id": row["class_id"], "group": row["group"],
                "logits": logits, "prediction": pred,
                "sample_name": row["sample_name"]}

    # -- runtime evidence -----------------------------------------------------
    def _write_runtime_evidence(self) -> None:
        import hashlib

        d = self.root / "evaluate"
        d.mkdir(parents=True, exist_ok=True)
        (d / "layer_b_summary.json").write_text("{}", encoding="utf-8")
        rows = self._sample_rows()
        oracles = {}
        index = 0
        for fold, names in self._layout.items():
            lines = []
            for _ in names:
                lines.append(json.dumps(self._oracle_row(index, rows[index])))
                index += 1
            p = self.root / "fit" / "models" / f"{fold}_P0" / \
                "outer_eval_predictions.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            oracles[fold] = p

        # pipeline evidence: F1=B_000 (m<1 error), F2=B_001 (missed real
        # fish), F3=A_000 (m>=1 error, train)
        frames = [
            {"frame_id": "B_000", "group": "GB", "fold": "fold0",
             "frame_error": None,
             "gt": [{"gt_id": 0, "class_id": 3}, {"gt_id": 1, "class_id": 1}],
             "predictions": [{"pred_instance_id": 0},
                             {"pred_instance_id": 1}],
             "matches": [[0, 0, 0.8], [1, 1, 0.9]],
             "missed_gt_ids": [], "unmatched_pred_ids": []},
            {"frame_id": "B_001", "group": "GB", "fold": "fold0",
             "frame_error": None,
             "gt": [{"gt_id": 0, "class_id": 3}, {"gt_id": 1, "class_id": 1}],
             "predictions": [{"pred_instance_id": 0}],
             "matches": [[1, 0, 0.7]],
             "missed_gt_ids": [0], "unmatched_pred_ids": []},
            {"frame_id": "A_000", "group": "GA", "fold": "fold1",
             "frame_error": None,
             "gt": [{"gt_id": 0, "class_id": 0}],
             "predictions": [{"pred_instance_id": 0}],
             "matches": [[0, 0, 0.75]],
             "missed_gt_ids": [], "unmatched_pred_ids": []},
        ]
        r0 = [
            {"candidate_id": "R0", "frame_id": "B_000", "fold": "fold0",
             "group": "GB", "pred_instance_id": 0, "matched_gt_id": 0,
             "final_logits": _logits(3, 3, 0.0), "predicted_class": 3,
             "used_fallback": False, "fallback_reason": "",
             "status": "valid", "mask_source": "predicted_instance_mask"},
            {"candidate_id": "R0", "frame_id": "B_000", "fold": "fold0",
             "group": "GB", "pred_instance_id": 1, "matched_gt_id": 1,
             "final_logits": _logits(1, 2, 0.5), "predicted_class": 2,
             "used_fallback": False, "fallback_reason": "",
             "status": "valid", "mask_source": "predicted_instance_mask"},
            {"candidate_id": "R0", "frame_id": "B_001", "fold": "fold0",
             "group": "GB", "pred_instance_id": 0, "matched_gt_id": 1,
             "final_logits": _logits(1, 1, 0.0), "predicted_class": 1,
             "used_fallback": False, "fallback_reason": "",
             "status": "valid", "mask_source": "predicted_instance_mask"},
            {"candidate_id": "R0", "frame_id": "A_000", "fold": "fold1",
             "group": "GA", "pred_instance_id": 0, "matched_gt_id": 0,
             "final_logits": _logits(0, 1, 2.0), "predicted_class": 1,
             "used_fallback": False, "fallback_reason": "",
             "status": "valid", "mask_source": "predicted_instance_mask"},
        ]
        (d / "frames_complete.jsonl").write_text(
            "\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")
        (d / "predictions_complete.jsonl").write_text(
            "\n".join(json.dumps(r) for r in r0) + "\n", encoding="utf-8")
        self._oracles = oracles
        # frozen identities the entry binds from the artifact manifest
        self._runtime_files = {}
        for rel in ("evaluate/frames_complete.jsonl",
                    "evaluate/predictions_complete.jsonl",
                    "evaluate/layer_b_summary.json"):
            payload = (self.root / rel).read_bytes()
            self._runtime_files[rel] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}

    # -- detection evidence ----------------------------------------------------
    def _write_detection(self) -> None:
        import cv2

        for cond in ("D0", "D1", "D2", "D3"):
            d = self.root / "detection" / "conditions" / cond
            (d / "masks").mkdir(parents=True, exist_ok=True)
            conf = {"D0": 0.25, "D1": 0.10, "D2": 0.05, "D3": 0.25}[cond]
            # one real-fish unmatched GT case on B_001 gt0
            unmatched = [{
                "frame_id": "B_001", "gt_id": 0, "class_id": 3,
                "class_name": "real_fish", "group": "GB",
                "attribution": "no_corresponding_prediction_sign",
                "max_iou_any_prediction": 0.0, "no_positive_overlap": True,
                "best_prediction_index": None,
                "best_prediction_matched_elsewhere": False,
                "predictions_in_frame": 1,
                "note": "synthetic sign",
                "signs": {"d0_max_iou": 0.0, "d0_predictions_in_frame": 1,
                          "d0_best_matched_elsewhere": False,
                          "matched_D1": False, "matched_D2": False,
                          "matched_D3": False},
            }] if cond == "D0" else [{
                "frame_id": "B_001", "gt_id": 0, "class_id": 3,
                "class_name": "real_fish", "group": "GB",
                "attribution": "no_corresponding_prediction_sign",
                "max_iou_any_prediction": 0.0, "no_positive_overlap": True,
                "best_prediction_index": None,
                "best_prediction_matched_elsewhere": False,
                "predictions_in_frame": 1, "note": "synthetic sign",
                "signs": {"matched_D1": False, "matched_D2": False,
                          "matched_D3": False},
            }]
            (d / "unmatched_gt.jsonl").write_text(
                "\n".join(json.dumps(r) for r in unmatched) + "\n",
                encoding="utf-8")
            (d / "unmatched_predictions.jsonl").write_text(
                "", encoding="utf-8")
            (d / "frames.jsonl").write_text(
                "\n".join(json.dumps({
                    "frame_id": s, "group": self.frames[s]["group"],
                    "gt_instances": sum(self.frames[s]["counts"].values()),
                    "matched": 0, "missed_gt":
                        sum(self.frames[s]["counts"].values()),
                    "predictions": 0, "unmatched_predictions": 0})
                    for s in sorted(self.frames)) + "\n", encoding="utf-8")
            per_gt, preds = [], []
            for stem in sorted(self.frames):
                for gt_id in range(sum(self.frames[stem]["counts"].values())):
                    per_gt.append({
                        "frame_id": stem, "gt_id": gt_id, "class_id": 3,
                        "class_name": "real_fish", "group": "GB",
                        "matched": False, "match_iou": 0.0,
                        "max_iou_any_prediction": 0.0,
                        "best_prediction_index": None,
                        "best_prediction_matched_elsewhere": False,
                        "predictions_in_frame": 1})
            (d / "per_gt.jsonl").write_text(
                "\n".join(json.dumps(r) for r in per_gt) + "\n",
                encoding="utf-8")
            (d / "predictions.jsonl").write_text(
                "\n".join(json.dumps(r) for r in preds) + "\n",
                encoding="utf-8")
            if cond == "D0":
                for stem in sorted(self.frames):
                    if self.frames[stem]["split"] != "val":
                        continue
                    mask = np.zeros((64, 72), np.uint8)
                    mask[10:30, 14:38] = 1
                    packed = np.packbits(
                        mask.reshape(1, -1), axis=1)
                    np.savez(
                        d / "masks" / f"{stem}.npz",
                        masks=packed, shape=np.array([64, 72]),
                        confidences=np.array([0.9], dtype=np.float32),
                        bboxes=np.array([[14, 10, 38, 30]], dtype=np.int32))
        (self.root / "detection" / "summary.json").write_text(
            json.dumps({"status": "COMPLETED"}), encoding="utf-8")

    # -- atlas ---------------------------------------------------------------
    def _write_atlas(self) -> None:
        import cv2

        d = self.root / "atlas_rebuild" / "atlas"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "unmatched_gt_B_001_gt0.png"),
                    np.full((16, 16), 128, np.uint8))
        cv2.imwrite(str(d / "control_real_fish_B_000_gt0.png"),
                    np.full((16, 16), 128, np.uint8))
        cv2.imwrite(str(d / "control_metal_submarine_B_002_gt0.png"),
                    np.full((16, 16), 128, np.uint8))
        (self.root / "atlas_rebuild" / "atlas_manifest.json").write_text(
            json.dumps({"panels": 3, "status": "COMPLETED",
                        "problems": [], "note": "synthetic",
                        "unmatched_gt_panels": 1,
                        "unmatched_prediction_panels": 0}),
            encoding="utf-8")

    # -- configs ---------------------------------------------------------------
    def _stereo_pins(self) -> dict:
        return {"matcher": "sgbm", "mode": "3way", "scale": 1.0,
                "max_disp": 16, "block_size": 5, "uniqueness_ratio": 10,
                "speckle_window": 100, "speckle_range": 32,
                "texture_threshold": 10.0, "lr_check_threshold_px": 2.0,
                "min_valid_ratio": 0.05, "band_margin": 20,
                "full_image_threshold": 0.9}

    def _write_configs(self) -> None:
        def rel(path: Path) -> str:
            return path.relative_to(PROJECT_ROOT).as_posix()

        frozen = {
            "candidate": {"class_names": list(CLASS_NAMES)},
            "stereo": self._stereo_pins(),
            "polarization": {"gain": 1.0, "epsilon": 1.0e-06,
                             "min_intensity_sum": 10.0},
            "p0_models": {},
        }
        oracles_rel = {}
        for fold, path in self._oracles.items():
            oracles_rel[fold] = rel(path)
            frozen["p0_models"][fold] = {
                "oracle": {"path": rel(path),
                           "sha256": diag_mod.sha256_of_file(path)}}
        local_manifest = self.root / "local_features" / "feature_manifest.json"
        frozen["features"] = {"local_cache": {
            "path": rel(self.root / "local_features"),
            "files": {"feature_manifest.json":
                      diag_mod.sha256_of_file(local_manifest)}}}
        frozen_path = self.root / "fake_frozen_runtime.yaml"
        frozen_path.write_text(yaml.safe_dump(frozen), encoding="utf-8")
        manifest_path = self.root / "artifact_manifest.json"
        manifest_path.write_text(json.dumps({
            "note": "synthetic",
            "files": self._runtime_files,
        }), encoding="utf-8")

        config = {
            "experiment": "material_measurement_diagnostic_v1_synthetic",
            "purpose": "synthetic test world",
            "anchors": {
                "frozen_runtime_config": {
                    "path": rel(frozen_path),
                    "sha256": diag_mod.sha256_of_file(frozen_path)},
                "protocol": {
                    "path": rel(self.root / "protocol" / "protocol.json"),
                    "sha256": diag_mod.sha256_of_file(
                        self.root / "protocol" / "protocol.json")},
                "artifact_manifest": {
                    "path": rel(self.root / "artifact_manifest.json"),
                    "sha256": diag_mod.sha256_of_file(
                        self.root / "artifact_manifest.json")},
            },
            "protocol_expectations": {"val_frames": 10,
                                      "dev_splits": ["train", "val"]},
            "sources": {
                "p0_oracles": oracles_rel,
                "local_feature_manifest":
                    rel(self.root / "local_features" / "feature_manifest.json"),
                "runtime_evidence_dir": rel(self.root / "evaluate"),
                "runtime_frames":
                    rel(self.root / "evaluate" /
                        "frames_complete.jsonl"),
                "runtime_predictions":
                    rel(self.root / "evaluate" /
                        "predictions_complete.jsonl"),
                "runtime_layer_b_summary":
                    rel(self.root / "evaluate" /
                        "layer_b_summary.json"),
                "detection_dir": rel(self.root / "detection"),
                "atlas_dir": rel(self.root / "atlas_rebuild" / "atlas"),
                "atlas_manifest":
                    rel(self.root / "atlas_rebuild" / "atlas_manifest.json"),
                "split_csv": rel(self.root / "split_assignment.csv"),
                "label_root": rel(self.root / "labels"),
            },
            "class_names": list(CLASS_NAMES),
            "stereo": self._stereo_pins(),
            "polarization": {"gain": 1.0, "epsilon": 1.0e-06,
                             "min_intensity_sum": 10.0},
            "crop_padding": 10,
            "calibration": {"baseline": 0.0989,
                            "focal_length": 3643.52},
            "image_source": {"already_rectified": True,
                             "exposure_gain_gamma": "NOT_AVAILABLE",
                             "sync_status": "unknown"},
            "selection": {
                "max_frames_cap": 8,
                "required_rules": ["val_real_fish_gt", "no_target_dev",
                                   "normal_control_panels",
                                   "missing_polar_per_class",
                                   "pipeline_m_below_1_frames",
                                   "group_coverage_first_frame"],
                "fill_rule": "cycle groups",
            },
            "interventions": {
                "upsample_control": {"method": "valid_weighted",
                                     "w_min": 0.5},
                "right_shifts_px": [[1, 0], [-1, 0], [0, 1], [0, -1]],
            },
            "synthetic": {"seed": 2026},
            "budget": {"whole_run_minutes": 30},
            "output": {"evidence_root": "unused", "require_gitignored": True,
                       "final_leaf_must_not_exist": True},
            "review": {"human_reviewed_default": False,
                       "visual_review_source": "claude_visual_review",
                       "record_based_source": "record_based"},
            "provenance_disclosure": "synthetic world",
        }
        self.config_path = self.root / "diagnostic_config.yaml"
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")


def build_world(scratch: Path) -> World:
    world_root = scratch / "world"
    world_root.mkdir(parents=True, exist_ok=True)
    return World(world_root)


def _run_entry(world: World, output_dir: Path, extra: list[str]) -> int:
    return diag_mod.main([
        "--config", str(world.config_path),
        "--output-dir", str(output_dir),
        "--mode", "all",
        "--budget-minutes", "10",
        "--max-frames", "8",
        *extra,
    ])


# ---------------------------------------------------------------------------
# Entry-level tests


def test_dump_json_refuses_non_finite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        diag_mod.dump_json(tmp_path / "x.json", {"a": float("nan")})
    assert not (tmp_path / "x.json").exists()


def test_entry_refuses_existing_output_dir(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)
    out = scratch_dir / "run_out"
    out.mkdir()
    code = _run_entry(world, out, [])
    assert code == diag_mod.EXIT_USAGE
    assert not (out / "run_status.json").exists()


def test_entry_refuses_output_outside_gitignore(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)
    probe_dir = PROJECT_ROOT / f"tests/_diag_refusal_probe_{scratch_dir.name}"
    try:
        code = diag_mod.main([
            "--config", str(world.config_path),
            "--output-dir", str(probe_dir),
            "--mode", "all",
        ])
        assert code == diag_mod.EXIT_USAGE
        assert not probe_dir.exists()
    finally:
        if probe_dir.exists():  # never leave a probe directory behind
            probe_dir.rmdir()


def test_entry_refuses_max_frames_over_cap(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)
    out = scratch_dir / "run_out"
    code = diag_mod.main([
        "--config", str(world.config_path),
        "--output-dir", str(out),
        "--mode", "all",
        "--max-frames", "99",
    ])
    assert code == diag_mod.EXIT_USAGE
    assert not out.exists()


def test_entry_budget_exhaustion_reports_partial(scratch_dir: Path,
                                                 monkeypatch) -> None:
    world = build_world(scratch_dir)
    out = scratch_dir / "run_out"

    class ExplodingWatch:
        def __init__(self, minutes: float, *, clock=None) -> None:
            self.calls = 0

        def check(self, at: str = "") -> None:
            self.calls += 1
            if self.calls > 1:
                raise diag_mod.BudgetExceeded(
                    f"fake budget exhausted at {at}")

        def remaining_s(self) -> float:
            return 0.0

    monkeypatch.setattr(diag_mod, "BudgetWatch", ExplodingWatch)
    code = _run_entry(world, out, [])
    assert code == diag_mod.EXIT_PARTIAL
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "PARTIAL"
    assert status["interrupted_by_budget"] is True
    assert status["stages"]["verify"]["status"] == "COMPLETED"
    assert "selection" in status["stages_not_run"]


def test_entry_saved_mode_scopes_its_stages(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)
    out = scratch_dir / "run_out"
    code = diag_mod.main([
        "--config", str(world.config_path),
        "--output-dir", str(out),
        "--mode", "saved",
        "--budget-minutes", "10",
        "--max-frames", "8",
    ])
    assert code == diag_mod.EXIT_OK
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "COMPLETED"
    assert set(status["stages"]) == {"verify", "selection", "saved",
                                     "protection"}
    for stage in ("synthetic", "recompute", "review"):
        assert stage in status["stages_not_in_mode"]
    assert (out / "residual_capacity_summary.json").exists()
    assert not (out / "synthetic_report.json").exists()
    assert not (out / "recompute_frames.jsonl").exists()
    assert not (out / "real_fish_review.csv").exists()


def test_entry_all_mode_completes_with_full_evidence(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)
    out = scratch_dir / "run_out"
    code = _run_entry(world, out, [])
    assert code == diag_mod.EXIT_OK, \
        (out / "run_status.json").read_text(encoding="utf-8")
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["overall_status"] == "COMPLETED"
    assert status["stages_not_run"] == []
    for name in ("run_status.json", "selection_manifest.json",
                 "residual_capacity_summary.json", "synthetic_report.json",
                 "recompute_frames.jsonl", "recompute_instances.jsonl",
                 "real_fish_review.csv", "protected_before.json",
                 "protected_after.json", "protection_check.json",
                 "source_manifest.json", "diagnostic_report.md"):
        assert (out / name).exists(), name

    # task A: the world's hand-computed numbers survive the entry
    summary = json.loads(
        (out / "residual_capacity_summary.json").read_text(encoding="utf-8"))
    assert summary["outer_eval"]["totals"]["samples"] == 12
    assert summary["outer_eval"]["totals"]["errors"] == 4
    assert summary["outer_eval"]["totals"]["m_ge_1"] == 1
    assert summary["outer_eval"]["totals"]["m_lt_1"] == 3
    assert summary["pipeline_join"]["totals"]["errors"] == 2
    assert summary["pipeline_join"]["totals"]["m_ge_1"] == 1
    assert summary["pipeline_join"]["totals"]["m_lt_1"] == 1

    # task C: frozen selection
    selection = json.loads(
        (out / "selection_manifest.json").read_text(encoding="utf-8"))
    assert selection["selected_count"] == 8
    assert selection["groups_covered"] == 3
    ids = {f["frame_id"] for f in selection["frames"]}
    assert {"B_000", "B_001", "C_003", "B_002", "A_000", "C_000"} <= ids

    # task D: every selected frame recorded, engine cross-check identical
    frames = [json.loads(line) for line in
              (out / "recompute_frames.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    assert len(frames) == 8
    for row in frames:
        gt = row["gt_conditional"]
        if gt.get("n_instances", 0):
            assert gt["engine_reference"]["crosscheck"]["identical"] is True

    # task E: exactly one real-fish case, record-based, human review false
    import csv as csv_mod
    with (out / "real_fish_review.csv").open(encoding="utf-8") as handle:
        rows = list(csv_mod.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["frame_id"] == "B_001" and rows[0]["gt_id"] == "0"
    assert rows[0]["human_reviewed"] == "false"
    assert rows[0]["review_source"] == "record_based"

    protection = json.loads(
        (out / "protection_check.json").read_text(encoding="utf-8"))
    assert protection["identical"] is True
    assert protection["files_compared"] > 10


def test_entry_records_claude_visual_observations(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)
    out = scratch_dir / "run_out"
    obs = {
        "observations": [{
            "frame_id": "B_001", "gt_id": 0,
            "observations": "synthetic observation for the test world",
            "evidence_strength": "test",
            "items_to_confirm": "synthetic",
            "human_reviewed": False,
        }]
    }
    obs_path = scratch_dir / "observations.json"
    obs_path.write_text(json.dumps(obs), encoding="utf-8")
    code = _run_entry(world, out,
                      ["--record-visual-observations", str(obs_path)])
    assert code == diag_mod.EXIT_OK
    import csv as csv_mod
    with (out / "real_fish_review.csv").open(encoding="utf-8") as handle:
        rows = list(csv_mod.DictReader(handle))
    assert rows[0]["review_source"] == "claude_visual_review"
    assert rows[0]["human_reviewed"] == "false"
    assert rows[0]["visual_observation"] == \
        "synthetic observation for the test world"


def test_entry_rejects_observations_claiming_human_review(
        scratch_dir: Path) -> None:
    world = build_world(scratch_dir)
    out = scratch_dir / "run_out"
    obs_path = scratch_dir / "observations.json"
    obs_path.write_text(json.dumps({"observations": [{
        "frame_id": "B_001", "gt_id": 0, "observations": "x",
        "evidence_strength": "y", "items_to_confirm": "z",
        "human_reviewed": True}]}), encoding="utf-8")
    code = _run_entry(world, out,
                      ["--record-visual-observations", str(obs_path)])
    assert code == diag_mod.EXIT_FAILED


# ---------------------------------------------------------------------------
# Per-instance paired statistics (repair 1)


def _paired_world():
    """One 4x8 instance; baseline measures the left half, the arm measures
    the whole mask with shifted q values, exercising cancellation, valid
    set changes, empty regions and the abs_q definition."""
    from core.measurement_diagnostic import instance_paired_comparison

    mask = np.zeros((4, 8), bool)
    mask[:, 2:6] = True
    interior, boundary = instance_boundary_split(mask.astype(np.uint8))
    base_q = np.zeros((4, 8), np.float32)
    base_abs = np.zeros((4, 8), np.float32)
    base_qvalid = np.zeros((4, 8), bool)
    base_qvalid[:, 2:4] = True          # baseline only measured columns 2-3
    base_q[:, 2:4] = 0.5
    base_abs[:, 2:4] = 0.5
    disp = np.full((4, 8), 6.0, np.float32)
    valid = np.zeros((4, 8), bool)
    valid[:, 1:7] = True
    return (mask, interior, boundary, disp, valid, base_q, base_abs,
            base_qvalid)


def test_paired_stats_cancellation_abs_definition_and_flips():
    (mask, interior, boundary, disp, valid, base_q, base_abs,
     base_qvalid) = _paired_world()
    # arm: column 2 flips sign (+0.5 -> -0.5) => dq=-1, dabs=0;
    # column 3 rises +0.5 -> +0.7 => dq=+0.2, dabs=+0.2;
    # columns 4-5 are newly measured at +0.1 (gained validity)
    arm_q = np.zeros((4, 8), np.float32)
    arm_abs = np.zeros((4, 8), np.float32)
    arm_qvalid = np.zeros((4, 8), bool)
    arm_qvalid[:, 2:6] = True
    arm_q[:, 2] = -0.5
    arm_abs[:, 2] = 0.5
    arm_q[:, 3] = 0.7
    arm_abs[:, 3] = 0.7
    arm_q[:, 4:6] = 0.1
    arm_abs[:, 4:6] = 0.1
    out = instance_paired_comparison(
        mask, interior, boundary, disp, valid, base_q, base_abs,
        base_qvalid, disp, valid, arm_q, arm_abs, arm_qvalid)
    counts = out["q_valid_counts"]
    assert counts == {"baseline": 8, "arm": 16, "common": 8, "lost": 0,
                      "gained": 8}
    dq = out["signed_dq"]
    # per pixel on common (columns 2-3, 4 rows): [-1, +0.2] per row
    assert dq["n"] == 8
    assert dq["signed_mean"] == pytest.approx((-1.0 + 0.2) / 2)
    # the mean difference cancels most of the change: it must NOT be
    # mistaken for the pixel error
    assert dq["abs_mean"] == pytest.approx((1.0 + 0.2) / 2)
    assert dq["abs_max"] == pytest.approx(1.0)
    # dabs_q: sign flip contributes 0, rise contributes +0.2
    assert out["dabs_q"]["signed_mean"] == pytest.approx(0.2 / 2)
    assert out["dabs_q"]["abs_mean"] == pytest.approx(0.2 / 2)
    # disparity valid sets identical -> no disparity flips
    assert out["disparity_valid_counts"]["lost"] == 0
    assert out["disparity_valid_counts"]["gained"] == 0
    dd = out["disparity_diff"]
    assert dd["n"] == 16 and dd["abs_max"] == 0.0


def test_paired_stats_empty_common_reports_nulls():
    (mask, interior, boundary, disp, valid, base_q, base_abs,
     base_qvalid) = _paired_world()
    arm_qvalid = np.zeros(mask.shape, bool)  # arm measures nothing
    arm_q = np.zeros(mask.shape, np.float32)
    arm_abs = np.zeros(mask.shape, np.float32)
    out = instance_paired_comparison(
        mask, interior, boundary, disp, valid, base_q, base_abs,
        base_qvalid, disp, valid, arm_q, arm_abs, arm_qvalid)
    assert out["signed_dq"] == {"n": 0, "signed_mean": None, "abs_mean": None,
                                "abs_p50": None, "abs_p95": None,
                                "abs_max": None}
    assert out["q_valid_counts"]["common"] == 0
    assert out["q_valid_counts"]["lost"] == 8
    assert out["q_valid_counts"]["gained"] == 0
    assert out["boundary"]["signed_dq"]["n"] == 0
    # interior of this 4x8 mask (3x3 erosion of columns 2:6) still has pixels
    assert out["interior"]["pixel_count"] > 0


def test_paired_stats_fixed_regions_are_mask_only():
    """Interior/boundary regions are fixed by the instance mask; changing
    the arm's validity only in the boundary must leave the interior block
    untouched."""
    (mask, interior, boundary, disp, valid, base_q, base_abs,
     base_qvalid) = _paired_world()
    arm_qvalid = base_qvalid.copy()
    arm_q = base_q.copy()
    arm_abs = base_abs.copy()
    arm_qvalid[:, 2] = False  # drop one boundary column
    arm_qvalid[:, 5] = True   # gain one boundary column
    arm_q[:, 5] = 0.3
    arm_abs[:, 5] = 0.3
    out = instance_paired_comparison(
        mask, interior, boundary, disp, valid, base_q, base_abs,
        base_qvalid, disp, valid, arm_q, arm_abs, arm_qvalid)
    assert out["interior"]["q_valid_counts"]["baseline"] == \
        out["interior"]["q_valid_counts"]["arm"]
    assert out["interior"]["q_valid_counts"]["common"] == \
        out["interior"]["q_valid_counts"]["baseline"]
    assert out["boundary"]["q_valid_counts"]["baseline"] == \
        out["q_valid_counts"]["baseline"] - \
        out["interior"]["q_valid_counts"]["baseline"]
    assert out["boundary"]["q_valid_counts"]["gained"] > 0


# ---------------------------------------------------------------------------
# Repair 2: edge masks must come from each arm's own disparity


def test_measure_condition_edge_masks_use_their_own_disparity():
    from gui.inference_engine import DualStageInferenceEngine

    import scripts.diagnose_material_measurement as dm
    cfg, left, right, instances = _textured_arm_world(3)
    probe = dm._ModelAProbe()
    engine = DualStageInferenceEngine(
        model_a=probe, model_b=None, matcher=dm.StereoMatcher(cfg),
        rectifier=None, rectify_enabled=False, device="cpu")
    calls = []
    real_edge = dm.edge_unmeasurable_mask

    def recording_edge(shape, disparity, dx, dy):
        calls.append(np.asarray(disparity).copy())
        return real_edge(shape, disparity, dx, dy)

    orig = dm.edge_unmeasurable_mask
    dm.edge_unmeasurable_mask = recording_edge
    try:
        result = dm._measure_condition(
            None, stereo_cfg=cfg, engine=engine, probe=probe,
            left=left, right=right, instances=instances, crop_windows=None,
            w_min=0.5, shifts=[(1, 0)], engine_check=False)
    finally:
        dm.edge_unmeasurable_mask = orig
    assert len(calls) == 2, "one edge call per shift arm"
    full = result["shifts"]["shift_dxp1_dyp0"]["full_recompute"]
    fixed = result["shifts"]["shift_dxp1_dyp0"]["fixed_baseline_warp"]
    assert full["edge_mask_source"] == "stereo3.disparity (re-matched)"
    assert fixed["edge_mask_source"] == "stereo0.disparity (baseline)"
    assert "edge_excluded_pixels" in full and "edge_excluded_pixels" in fixed
    # math-level counterexamples: two disparity maps whose sample positions
    # fall on opposite sides of the fill-touch zone (dx=+1 marks where
    # 0 <= x-d < 1: d=3.0 -> only x=3; d=0.5 -> only x=1)
    shape = (4, 10)
    disp_far = np.full(shape, 3.0, np.float32)
    disp_near = np.full(shape, 0.5, np.float32)
    edge_far = real_edge(shape, disp_far, 1, 0)
    edge_near = real_edge(shape, disp_near, 1, 0)
    assert edge_far[:, 3].all() and not edge_far[:, 2].any()
    assert edge_near[:, 1].all() and not edge_near[:, 2].any()
    assert not np.array_equal(edge_far, edge_near)


# ---------------------------------------------------------------------------
# Repair 3: source binding and first-read protection


def _rewrite_anchor(world: Path, anchor: str, mutate) -> Path:
    """Mutate a world anchor file and re-pin its sha256 in the config so
    the anchor check itself passes and the mutation reaches the binding
    logic."""
    import yaml as yaml_mod

    config_path = world / "diagnostic_config.yaml"
    config = yaml_mod.safe_load(config_path.read_text(encoding="utf-8"))
    target = PROJECT_ROOT / config["anchors"][anchor]["path"]
    mutate(target)
    config["anchors"][anchor]["sha256"] = diag_mod.sha256_of_file(target)
    config_path.write_text(yaml_mod.safe_dump(config), encoding="utf-8")
    return config_path


def test_entry_refuses_missing_artifact_manifest_entry(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)

    def mutate(path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["files"]["evaluate/frames_complete.jsonl"]
        path.write_text(json.dumps(data), encoding="utf-8")

    config_path = _rewrite_anchor(world.root, "artifact_manifest", mutate)
    out = scratch_dir / "run_out"
    code = diag_mod.main([
        "--config", str(config_path), "--output-dir", str(out),
        "--mode", "saved", "--budget-minutes", "5", "--max-frames", "8",
    ])
    assert code == diag_mod.EXIT_FAILED
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert "artifact_manifest has no usable entry" in status["failure"]


def test_entry_refuses_binding_content_mismatch(scratch_dir: Path) -> None:
    world = build_world(scratch_dir)

    def mutate(path: Path) -> None:
        # the manifest stays frozen; the BOUND FILE's content is tampered
        frames = path.parent / "evaluate" / "frames_complete.jsonl"
        frames.write_text(
            frames.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    config_path = _rewrite_anchor(world.root, "artifact_manifest", mutate)
    out = scratch_dir / "run_out"
    code = diag_mod.main([
        "--config", str(config_path), "--output-dir", str(out),
        "--mode", "saved", "--budget-minutes", "5", "--max-frames", "8",
    ])
    assert code == diag_mod.EXIT_FAILED
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert "frozen expectation" in status["failure"]


def test_protection_rejects_change_after_first_read(tmp_path: Path) -> None:
    """End-of-run comparison must flag any input that changed after its
    first-read identity was recorded."""
    consumed = tmp_path / "input.txt"
    consumed.write_text("v1", encoding="utf-8")
    ctx = diag_mod.RunContext(
        config={"anchors": {}}, config_path=tmp_path / "c.yaml",
        output_dir=tmp_path / "out", mode="saved", budget_minutes=1.0,
        max_frames=1, run_id="t", watch=diag_mod.BudgetWatch(1.0))
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.consume(consumed, "role")
    consumed.write_text("v2", encoding="utf-8")  # change AFTER first read
    diag_mod.run_protection(ctx, "before")
    diag_mod.run_protection(ctx, "after")
    result = diag_mod.compare_protection(ctx)
    assert result["identical"] is False
    assert any("CHANGED after its first read" in p
               for p in result["problems"])


def test_protection_accepts_unchanged_inputs(tmp_path: Path) -> None:
    consumed = tmp_path / "input.txt"
    consumed.write_text("v1", encoding="utf-8")
    ctx = diag_mod.RunContext(
        config={"anchors": {}}, config_path=tmp_path / "c.yaml",
        output_dir=tmp_path / "out", mode="saved", budget_minutes=1.0,
        max_frames=1, run_id="t", watch=diag_mod.BudgetWatch(1.0))
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.consume(consumed, "role")
    diag_mod.run_protection(ctx, "before")
    diag_mod.run_protection(ctx, "after")
    result = diag_mod.compare_protection(ctx)
    assert result["identical"] is True
    assert result["files_compared"] == 1


def test_consume_refuses_bytes_diverging_from_frozen_binding(
        tmp_path: Path) -> None:
    consumed = tmp_path / "input.txt"
    consumed.write_text("actual", encoding="utf-8")
    ctx = diag_mod.RunContext(
        config={"anchors": {}}, config_path=tmp_path / "c.yaml",
        output_dir=tmp_path / "out", mode="saved", budget_minutes=1.0,
        max_frames=1, run_id="t", watch=diag_mod.BudgetWatch(1.0))
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(IdentityError, match="frozen expectation"):
        ctx.consume(consumed, "role", binding={
            "source": "unit test", "expected_sha256": "0" * 64,
            "expected_bytes": 3})


# ---------------------------------------------------------------------------
# Repair round 3: aggregation semantics, p95 wording, peak restriction


def _agg_row(conditioning, arm_name, group, class_name, paired_records):
    """A minimal frame row carrying one arm's paired records."""
    per_inst = [{"class_name": class_name} for _ in paired_records]
    other = {"n_instances": 0, "mask_source": "test", "arms": {},
             "shifts": {}}
    block = {"n_instances": len(paired_records), "mask_source": "test",
             "arms": {arm_name: {"paired": paired_records,
                                 "per_instance": per_inst}},
             "shifts": {}}
    row = {"frame_id": "F", "group": group}
    if conditioning == "gt_conditional":
        row["gt_conditional"] = block
        row["pred_conditional"] = other
    else:
        row["pred_conditional"] = block
        row["gt_conditional"] = other
    return row


def _null_stats(n=0):
    return {"n": n, "signed_mean": None, "abs_mean": None, "abs_p50": None,
            "abs_p95": None, "abs_max": None}


def test_aggregate_lost_gained_sum_over_all_records_including_empty_common():
    """lost/gained must cumulate over every instance record: an instance
    whose common region is EMPTY can still lose all its baseline-valid
    pixels (baseline valid, arm all invalid)."""
    empty_common = {
        "instance_id": 0,
        "q_valid_counts": {"baseline": 10, "arm": 0, "common": 0,
                           "lost": 10, "gained": 0},
        "signed_dq": _null_stats(0),
        "dabs_q": _null_stats(0),
        "disparity_valid_counts": {"baseline": 0, "arm": 0, "common": 0,
                                   "lost": 0, "gained": 0},
        "disparity_diff": _null_stats(0),
    }
    measured = {
        "instance_id": 1,
        "q_valid_counts": {"baseline": 4, "arm": 4, "common": 4,
                           "lost": 2, "gained": 3},
        "signed_dq": {"n": 4, "signed_mean": 0.1, "abs_mean": 0.2,
                      "abs_p50": 0.2, "abs_p95": 0.3, "abs_max": 0.4},
        "dabs_q": {"n": 4, "signed_mean": 0.0, "abs_mean": 0.05,
                   "abs_p50": 0.05, "abs_p95": 0.05, "abs_max": 0.05},
        "disparity_valid_counts": {"baseline": 4, "arm": 4, "common": 4,
                                   "lost": 0, "gained": 0},
        "disparity_diff": {"n": 4, "signed_mean": 0.0, "abs_mean": 0.0,
                           "abs_p50": 0.0, "abs_p95": 0.0, "abs_max": 0.0},
    }
    summary = diag_mod.aggregate_paired_stats([
        _agg_row("gt_conditional", "a", "G", "C",
                 [empty_common, measured])])
    overall = summary["overall"]
    assert overall["q_lost_total"] == 12  # 10 from the EMPTY-common record
    assert overall["q_gained_total"] == 3
    assert overall["instances_total"] == 2
    assert overall["instances_with_common_region"] == 1
    assert overall["common_pixels_total"] == 4
    # means use only the with-common record
    assert overall["abs_dq_mean_pixel_weighted"] == pytest.approx(0.2)


def test_aggregate_all_common_empty_stats_null_counts_correct():
    """When every common region is empty the mean/percentile fields are
    null, but the validity counts and flip totals stay correct."""
    empty1 = {
        "q_valid_counts": {"baseline": 5, "arm": 0, "common": 0,
                           "lost": 5, "gained": 0},
        "signed_dq": _null_stats(0), "dabs_q": _null_stats(0),
        "disparity_valid_counts": {"baseline": 0, "arm": 0, "common": 0,
                                   "lost": 0, "gained": 0},
        "disparity_diff": _null_stats(0),
    }
    empty2 = {
        "q_valid_counts": {"baseline": 0, "arm": 7, "common": 0,
                           "lost": 0, "gained": 7},
        "signed_dq": _null_stats(0), "dabs_q": _null_stats(0),
        "disparity_valid_counts": {"baseline": 0, "arm": 0, "common": 0,
                                   "lost": 0, "gained": 0},
        "disparity_diff": _null_stats(0),
    }
    summary = diag_mod.aggregate_paired_stats([
        _agg_row("pred_conditional", "a", "G", "C", [empty1, empty2])])
    overall = summary["overall"]
    assert overall["common_pixels_total"] == 0
    assert overall["instances_with_common_region"] == 0
    assert overall["signed_dq_mean_pixel_weighted"] is None
    assert overall["abs_dq_mean_pixel_weighted"] is None
    assert overall["abs_dq_instance_p95_pixelcount_weighted"] is None
    assert overall["q_lost_total"] == 5
    assert overall["q_gained_total"] == 7


def test_aggregate_p95_field_is_weighted_instance_p95_not_pooled():
    """The reported p95 field is the pixel-count-weighted mean of
    PER-INSTANCE p95 values. A pooled pixel p95 would give a different
    number here; the aggregate must equal the former and must never be
    readable as the latter."""
    # instance A: 100 common pixels, |dq| 0.2 except one 0.9 -> p95 = 0.2
    inst_a = {
        "q_valid_counts": {"baseline": 100, "arm": 100, "common": 100,
                           "lost": 0, "gained": 0},
        "signed_dq": {"n": 100, "signed_mean": 0.2, "abs_mean": 0.2,
                      "abs_p50": 0.2, "abs_p95": 0.2, "abs_max": 0.9},
        "dabs_q": {"n": 100, "signed_mean": 0.0, "abs_mean": 0.0,
                   "abs_p50": 0.0, "abs_p95": 0.0, "abs_max": 0.0},
        "disparity_valid_counts": {"baseline": 0, "arm": 0, "common": 0,
                                   "lost": 0, "gained": 0},
        "disparity_diff": _null_stats(0),
    }
    # instance B: 10 common pixels, all |dq| = 0.9 -> p95 = 0.9
    inst_b = {
        "q_valid_counts": {"baseline": 10, "arm": 10, "common": 10,
                           "lost": 0, "gained": 0},
        "signed_dq": {"n": 10, "signed_mean": 0.9, "abs_mean": 0.9,
                      "abs_p50": 0.9, "abs_p95": 0.9, "abs_max": 0.9},
        "dabs_q": {"n": 10, "signed_mean": 0.0, "abs_mean": 0.0,
                   "abs_p50": 0.0, "abs_p95": 0.0, "abs_max": 0.0},
        "disparity_valid_counts": {"baseline": 0, "arm": 0, "common": 0,
                                   "lost": 0, "gained": 0},
        "disparity_diff": _null_stats(0),
    }
    summary = diag_mod.aggregate_paired_stats([
        _agg_row("gt_conditional", "a", "G", "C", [inst_a, inst_b])])
    overall = summary["overall"]
    weighted_instance_p95 = (0.2 * 100 + 0.9 * 10) / 110
    assert overall["abs_dq_instance_p95_pixelcount_weighted"] == \
        pytest.approx(weighted_instance_p95)
    # a pooled pixel p95 over the same 110 pixels would be 0.9 (the large
    # instance's pixels occupy the top decile) — demonstrably a DIFFERENT
    # number, so the field must not be labeled pooled
    pooled_pixel_p95 = 0.9
    assert pooled_pixel_p95 != pytest.approx(weighted_instance_p95)
    assert "NOT_COMPUTED" in overall["pooled_pixel_p95"]
    assert "NOT_COMPUTED" in summary["definition"]["p95_field"]


def test_aggregate_separates_arms_per_conditioning():
    """Arm summaries must be provided separately per conditioning; the
    mixed block is overview-only."""
    r_gt = {
        "q_valid_counts": {"baseline": 4, "arm": 4, "common": 4,
                           "lost": 1, "gained": 0},
        "signed_dq": {"n": 4, "signed_mean": 0.5, "abs_mean": 0.5,
                      "abs_p50": 0.5, "abs_p95": 0.5, "abs_max": 0.5},
        "dabs_q": {"n": 4, "signed_mean": 0.0, "abs_mean": 0.0,
                   "abs_p50": 0.0, "abs_p95": 0.0, "abs_max": 0.0},
        "disparity_valid_counts": {"baseline": 0, "arm": 0, "common": 0,
                                   "lost": 0, "gained": 0},
        "disparity_diff": _null_stats(0), "instance_id": 0}
    r_pred = {
        "q_valid_counts": {"baseline": 4, "arm": 4, "common": 4,
                           "lost": 0, "gained": 2},
        "signed_dq": {"n": 4, "signed_mean": -0.5, "abs_mean": 0.5,
                      "abs_p50": 0.5, "abs_p95": 0.5, "abs_max": 0.5},
        "dabs_q": {"n": 4, "signed_mean": 0.0, "abs_mean": 0.0,
                   "abs_p50": 0.0, "abs_p95": 0.0, "abs_max": 0.0},
        "disparity_valid_counts": {"baseline": 0, "arm": 0, "common": 0,
                                   "lost": 0, "gained": 0},
        "disparity_diff": _null_stats(0), "instance_id": 0}
    summary = diag_mod.aggregate_paired_stats([
        _agg_row("gt_conditional", "armA", "G", "C", [r_gt]),
        _agg_row("pred_conditional", "armA", "G", "C", [r_pred])])
    per_cond = summary["by_arm_per_conditioning"]
    assert set(per_cond) == {"gt_conditional", "pred_conditional"}
    assert per_cond["gt_conditional"]["armA"]["q_lost_total"] == 1
    assert per_cond["gt_conditional"]["armA"]["q_gained_total"] == 0
    assert per_cond["pred_conditional"]["armA"]["q_lost_total"] == 0
    assert per_cond["pred_conditional"]["armA"]["q_gained_total"] == 2
    mixed = summary["overview_mixed_conditionings"]["by_arm"]["armA"]
    assert mixed["q_lost_total"] == 1 and mixed["q_gained_total"] == 2
    assert summary["overview_mixed_conditionings"]["note"]


# ---------------------------------------------------------------------------
# Repair round 3: worst-case peak restricted to the instance common region


class _FakePolar:
    def __init__(self, signed_q, valid_mask):
        self.signed_q = signed_q
        self.abs_q = np.abs(signed_q)
        self.valid_mask = valid_mask


def test_worst_payload_peak_is_restricted_to_instance_common_region():
    """A larger |dq| inside the crop but OUTSIDE the common region
    (baseline-invalid there) must never be selected; the recorded peak
    must equal the instance's paired common-region abs_max."""
    h, w = 12, 12
    mask = np.zeros((h, w), bool)
    mask[4:8, 4:8] = True
    base_q = np.zeros((h, w), np.float32)
    arm_q = np.zeros((h, w), np.float32)
    base_qvalid = np.zeros((h, w), bool)
    arm_qvalid = np.zeros((h, w), bool)
    # common region: left half of the mask (columns 4-5)
    base_qvalid[4:8, 4:6] = True
    arm_qvalid[4:8, 4:8] = True
    # inside common: |dq| = 0.3 at (5,4)
    base_q[5, 4] = 0.0
    arm_q[5, 4] = 0.3
    # outside common but inside the mask (baseline invalid at (5,7)):
    # a much larger |dq| = 0.9 that must NOT be selected
    base_q[5, 7] = 0.0
    arm_q[5, 7] = 0.9
    base_qvalid[5, 7] = False
    # outside the mask entirely (baseline valid, huge diff): also excluded
    base_q[1, 1] = 0.0
    arm_q[1, 1] = 5.0
    base_qvalid[1, 1] = True
    arm_qvalid[1, 1] = True
    gray = np.zeros((h, w), np.uint8)
    payload = diag_mod._crop_worst_payload(
        gray, gray, mask, None, _FakePolar(base_q, base_qvalid),
        None, _FakePolar(arm_q, arm_qvalid))
    assert payload["peak_pixel_global"] == [4, 5]
    assert payload["peak_abs_dq"] == pytest.approx(0.3)
    # the stored |dq| map is masked to the common region
    assert float(payload["abs_dq"].max()) == pytest.approx(0.3)
    # consistency with the paired statistics of the same instance
    interior, boundary = instance_boundary_split(mask.astype(np.uint8))
    paired = instance_paired_comparison(
        mask, interior, boundary,
        np.zeros((h, w), np.float32), np.ones((h, w), bool),
        base_q, np.abs(base_q), base_qvalid,
        np.zeros((h, w), np.float32), np.ones((h, w), bool),
        arm_q, np.abs(arm_q), arm_qvalid)
    assert paired["signed_dq"]["abs_max"] == pytest.approx(
        payload["peak_abs_dq"])
    assert payload["peak_abs_dq"] != pytest.approx(0.9)


def test_worst_payload_empty_common_has_no_peak():
    h, w = 8, 8
    mask = np.zeros((h, w), bool)
    mask[2:6, 2:6] = True
    base_q = np.zeros((h, w), np.float32)
    arm_q = np.full((h, w), 0.7, np.float32)
    base_qvalid = np.zeros((h, w), bool)   # baseline measures nothing
    arm_qvalid = mask.copy()
    payload = diag_mod._crop_worst_payload(
        np.zeros((h, w), np.uint8), None, mask, None,
        _FakePolar(base_q, base_qvalid), None,
        _FakePolar(arm_q, arm_qvalid))
    assert payload["peak_pixel_global"] is None
    assert payload["peak_abs_dq"] is None
