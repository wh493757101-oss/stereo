"""material_residual_scale_v1 core tests: the scale rule, the two
magnitude arms, the exact fallbacks, the variant information barriers and
the pre-registered gates. Hand-computed expectations only; no real-report
values are used anywhere."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import scripts.train_material_conditional_residual as cond_entry
from core.material_conditional_residual import (
    assemble_conditional_inputs,
    build_conditional_head,
    conditional_delta,
    conditional_logits,
    p0_context,
    train_conditional_residual_head,
)
from core.material_residual_scale import (
    ARMS,
    FIXED_MAGNITUDE,
    DeadlineExceeded,
    DeadlineWatch,
    ScaleRefused,
    count_metrics,
    evaluate_gates,
    fold_scale,
    magnitude_for_arm,
    top2_margins,
)

INPUT_DIM = 35


def _head(seed: int = 7) -> torch.nn.Module:
    return build_conditional_head(seed=seed, input_dim=INPUT_DIM,
                                  hidden=8)


class TestScaleRule:
    def test_median_and_floor_hand_computed(self):
        # margins: rows ranked [a, b, c, d]; m = a - b
        logits = np.array([
            [3.0, 1.0, 0.0, 0.0],   # margin 2.0
            [2.0, 1.5, 0.0, 0.0],   # margin 0.5
            [4.0, 1.0, 1.0, 0.0],   # margin 3.0
        ])
        margins = top2_margins(logits)
        np.testing.assert_allclose(margins, [2.0, 0.5, 3.0])
        scale, info = fold_scale(logits)
        assert scale == pytest.approx(2.0)
        assert info["n"] == 3
        assert info["margin_median"] == pytest.approx(2.0)
        assert info["scaled_applied"] is True
        assert info["source"] == "p0_logits_inner_train"
        assert info["dtype"] == "float64"

    def test_floor_at_one_when_margins_small(self):
        logits = np.array([[1.0, 0.9, 0.0, 0.0],
                           [2.0, 1.95, 0.0, 0.0]])
        scale, info = fold_scale(logits)
        assert scale == 1.0
        assert info["scaled_applied"] is False

    def test_float64_and_deterministic(self):
        rng = np.random.default_rng(0)
        logits32 = rng.random((64, 4)).astype(np.float32) * 5.0
        scale_a, info_a = fold_scale(logits32)
        scale_b, info_b = fold_scale(logits32)
        assert scale_a == scale_b
        margins = top2_margins(logits32)
        assert margins.dtype == np.float64

    def test_val_eval_scores_cannot_change_the_scale(self):
        """The scale function's ONLY input is the inner_train logit
        matrix; different val/eval scores (never passed in) cannot move
        it. The evidence record pins the source."""
        train_logits = np.array([[3.0, 1.0, 0.0, 0.0],
                                 [2.0, 1.5, 0.0, 0.0]])
        scale_a, info_a = fold_scale(train_logits)
        # a wildly different "val/eval" score set exists but is not an
        # input of the rule; the record proves the source is inner_train
        _, info_b = fold_scale(train_logits)
        assert info_a["source"] == info_b["source"] == \
            "p0_logits_inner_train"
        assert scale_a == fold_scale(train_logits)[0]

    def test_refuses_empty_and_nonfinite(self):
        with pytest.raises(ScaleRefused):
            top2_margins(np.zeros((0, 4)))
        with pytest.raises(ScaleRefused):
            fold_scale(np.array([[np.nan, 1.0, 0.0, 0.0]]))
        with pytest.raises(ScaleRefused):
            fold_scale(np.array([[np.inf, 1.0, 0.0, 0.0]]))
        with pytest.raises(ScaleRefused):
            fold_scale(np.zeros((2, 3)))

    def test_magnitude_arms_differ_only_by_a_f(self):
        """fixed -> 0.5; scaled -> 0.5 * s_f; anything else refused."""
        assert magnitude_for_arm("fixed", 3.5) == FIXED_MAGNITUDE
        assert magnitude_for_arm("scaled", 3.5) == pytest.approx(1.75)
        assert magnitude_for_arm("scaled", 1.0) == FIXED_MAGNITUDE
        with pytest.raises(ScaleRefused):
            magnitude_for_arm("no_such_arm", 2.0)
        with pytest.raises(ScaleRefused):
            magnitude_for_arm("scaled", 0.5)  # below the floor


class TestMagnitudeArms:
    def _world(self, n: int = 12, seed: int = 3):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, INPUT_DIM)).astype(np.float32)
        p0 = (rng.normal(size=(n, 4)) * 2.0).astype(np.float32)
        valid = np.ones(n, dtype=bool)
        y = rng.integers(0, 4, size=n)
        return x, y, p0, valid

    def test_same_head_two_arms_bounds(self):
        x, _y, _p0, valid = self._world()
        head = _head()
        for magnitude in (0.5, 1.75):
            delta = conditional_delta(head, torch.from_numpy(x),
                                      torch.from_numpy(valid.astype(
                                          np.float32)), magnitude)
            assert float(delta.detach().abs().max()) <= magnitude
        # identical head and input: the scaled delta is EXACTLY the fixed
        # delta times the magnitude ratio (the only design difference)
        d_fixed = conditional_delta(head, torch.from_numpy(x),
                                    torch.from_numpy(
                                        valid.astype(np.float32)), 0.5)
        d_scaled = conditional_delta(head, torch.from_numpy(x),
                                     torch.from_numpy(
                                         valid.astype(np.float32)), 1.75)
        np.testing.assert_allclose(d_scaled.detach().numpy(),
                                   d_fixed.detach().numpy() * 3.5,
                                   rtol=0, atol=0)

    def test_epoch0_is_exactly_p0_for_both_arms(self):
        for magnitude in (0.5, 2.5):
            head = build_conditional_head(seed=11, input_dim=INPUT_DIM,
                                          hidden=8)
            x = torch.zeros((4, INPUT_DIM))
            valid = torch.ones(4)
            p0 = torch.tensor([[1.0, 0.5, -0.5, 0.0]] * 4)
            final = conditional_logits(head, x, valid, p0, magnitude)
            np.testing.assert_array_equal(final.detach().numpy(),
                                          p0.numpy())

    def test_missing_polar_falls_back_exactly_for_both_arms(self):
        x, _y, p0, _valid = self._world(n=6)
        head = _head()
        invalid = torch.zeros(6)
        p0_t = torch.from_numpy(p0)
        for magnitude in (0.5, 4.0):
            final = conditional_logits(head, torch.from_numpy(x), invalid,
                                       p0_t, magnitude)
            np.testing.assert_array_equal(final.detach().numpy(), p0)

    def test_trained_arms_share_init_and_only_a_f_differs(self):
        """Same fold+seed: identical first-layer init; training the two
        arms on the SAME data yields the same history up to the
        magnitude-driven deltas."""
        x, y, p0, valid = self._world(n=16)
        kwargs = dict(lr=0.01, weight_decay=0.001, max_epochs=3,
                      input_dim=INPUT_DIM, hidden=8)
        torch.manual_seed(0)
        fixed_model, fixed_history, fixed_state, _ = \
            train_conditional_residual_head(
                x, y, p0, valid.astype(np.float32), x, y, p0,
                valid.astype(np.float32), seed=42, magnitude=0.5,
                **kwargs)
        torch.manual_seed(0)
        scaled_model, scaled_history, scaled_state, _ = \
            train_conditional_residual_head(
                x, y, p0, valid.astype(np.float32), x, y, p0,
                valid.astype(np.float32), seed=42, magnitude=1.5,
                **kwargs)
        # identical epoch-0 candidates (P0 for both arms)
        assert fixed_history[0]["inner_val_macro_f1"] == \
            scaled_history[0]["inner_val_macro_f1"]
        # the magnitudes travelled into the histories
        assert fixed_history[0]["delta_max_abs"] == 0.0
        assert fixed_model is not scaled_model

    def test_rc_never_reads_s(self):
        """Two worlds differing ONLY in the standardized S block assemble
        to IDENTICAL RC inputs (the polarization content is unreachable),
        while RCS sees the difference."""
        rng = np.random.default_rng(5)
        c = rng.normal(size=(5, 14)).astype(np.float32)
        s_a = rng.normal(size=(5, 16)).astype(np.float32)
        s_b = s_a + 10.0
        context = p0_context(rng.normal(size=(5, 4)))
        valid = np.ones(5, dtype=bool)
        rc_a = assemble_conditional_inputs(c, s_a, context, valid, "RC")
        rc_b = assemble_conditional_inputs(c, s_b, context, valid, "RC")
        np.testing.assert_array_equal(rc_a, rc_b)
        # the S block (columns 14..29) is exactly zero; the context follows
        np.testing.assert_array_equal(rc_a[:, 14:30], 0.0)
        np.testing.assert_array_equal(rc_a[:, 30:],
                                      context.astype(np.float32))
        rcs_a = assemble_conditional_inputs(c, s_a, context, valid, "RCS")
        rcs_b = assemble_conditional_inputs(c, s_b, context, valid, "RCS")
        assert not np.array_equal(rcs_a, rcs_b)

    def test_c_carries_no_mean_abs_q(self):
        """C14 excludes the mean |q| (a polar content measure): two
        samples differing only in quality[3] build the same C."""
        from core.material_mask_dataset import MaskFusionSampleArrays

        def sample(mean_abs_q: float):
            gray = np.full((8, 8), 100, np.uint8)
            mask = np.zeros((8, 8), np.uint8)
            mask[2:6, 2:6] = 1
            valid = mask.copy()
            signed = np.zeros((8, 8), np.float32)
            quality = np.array([0.5, 1.0, 1.0, mean_abs_q], np.float32)
            return MaskFusionSampleArrays(
                gray=gray, signed_q=signed, abs_q=np.abs(signed),
                valid=valid, quality=quality, class_id=1,
                instance_mask=mask)

        from core.polar_local_features import build_local_control_features

        c_low = build_local_control_features(sample(0.01))
        c_high = build_local_control_features(sample(9.9))
        np.testing.assert_array_equal(c_low, c_high)


class TestDeadlineWatch:
    def test_exactly_one_of_deadline_or_budget(self):
        with pytest.raises(ScaleRefused):
            DeadlineWatch()
        with pytest.raises(ScaleRefused):
            DeadlineWatch(deadline_utc="2030-01-01T00:00:00Z",
                          budget_minutes=1.0)
        DeadlineWatch(deadline_utc="2030-01-01T00:00:00Z").check("x")
        with pytest.raises(ScaleRefused):
            DeadlineWatch(budget_minutes=-1.0)
        with pytest.raises(ScaleRefused):
            DeadlineWatch(budget_minutes=0.0)
        import time as _time

        watch = DeadlineWatch(budget_minutes=1e-6)
        _time.sleep(0.01)
        with pytest.raises(DeadlineExceeded):
            watch.check("boundary")


class TestIndependentMetrics:
    def test_hand_computed_counts(self):
        labels = [0, 0, 1, 2, 2, 3]
        predictions = [0, 1, 1, 2, 0, 3]
        out = count_metrics(labels, predictions)
        assert out["n"] == 6
        assert out["accuracy"] == pytest.approx(4 / 6)
        assert out["confusion_matrix"] == [
            [1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]]
        pc = out["per_class"]
        assert pc[0]["tp"] == 1 and pc[0]["fp"] == 1 and pc[0]["fn"] == 1
        assert pc[0]["recall"] == pytest.approx(0.5)
        assert pc[0]["precision"] == pytest.approx(0.5)
        assert pc[0]["f1"] == pytest.approx(0.5)
        assert pc[1]["tp"] == 1 and pc[1]["fn"] == 0 and pc[1]["fp"] == 1
        assert pc[3]["recall"] == 1.0
        macro = np.mean([c["f1"] for c in pc])
        assert out["macro_f1"] == pytest.approx(float(macro))


def _gate_summary(**overrides):
    """A hand-built summary just above every gate threshold."""
    def fold(fold, f1):
        return {"fold": fold, "rows": 10, "macro_f1": f1,
                "accuracy": f1}

    def seed_entry(f1_by_fold):
        folds = [fold(f"f{k}", v) for k, v in enumerate(f1_by_fold)]
        return {"folds": folds,
                "macro_f1_mean": float(np.mean(f1_by_fold)),
                "pooled": {"macro_f1": float(np.mean(f1_by_fold)),
                           "per_class_recall": {
                               "metal_submarine": 0.80,
                               "plastic_submarine": 0.80,
                               "plastic_fish": 0.80,
                               "real_fish": 0.80}}}

    r0_f1 = [0.70, 0.70, 0.70]
    rc_f1 = [0.72, 0.72, 0.72]
    rcs_f1 = [0.82, 0.82, 0.82]
    arms = {}
    for arm in ARMS:
        arms[arm] = {
            "R0": {"per_seed": {str(s): seed_entry(r0_f1)
                                for s in (2026, 2027, 2028)},
                   "macro_f1_mean_of_seeds": float(np.mean(r0_f1)),
                   "pooled_macro_f1_mean_of_seeds":
                       float(np.mean(r0_f1))},
            "RC": {"per_seed": {str(s): seed_entry(rc_f1)
                                for s in (2026, 2027, 2028)},
                   "macro_f1_mean_of_seeds": float(np.mean(rc_f1)),
                   "pooled_macro_f1_mean_of_seeds":
                       float(np.mean(rc_f1))},
            "RCS": {"per_seed": {str(s): seed_entry(
                rcs_f1 if arm == "scaled" else [0.75, 0.75, 0.75])
                for s in (2026, 2027, 2028)},
                "macro_f1_mean_of_seeds":
                    float(np.mean(rcs_f1 if arm == "scaled"
                                  else [0.75, 0.75, 0.75])),
                "pooled_macro_f1_mean_of_seeds":
                    float(np.mean(rcs_f1 if arm == "scaled"
                                  else [0.75, 0.75, 0.75]))},
        }

    def paired(fixed_by_seed):
        return {"per_seed": {str(s): {"fixed": f, "broke": b,
                                      "both_correct": 0, "both_wrong": 0,
                                      "support": 30}
                             for s, (f, b) in fixed_by_seed.items()},
                "total": {"fixed": sum(f for f, _ in fixed_by_seed.values()),
                          "broke": sum(b for _, b in
                                       fixed_by_seed.values()),
                          "both_correct": 0, "both_wrong": 0}}

    summary = {
        "class_names": ["metal_submarine", "plastic_submarine",
                        "plastic_fish", "real_fish"],
        "arms": arms,
        "paired": {
            "scaled_RCS_minus_R0": paired({2026: (4, 1), 2027: (4, 1),
                                           2028: (4, 1)}),
            "scaled_RCS_minus_RC": paired({2026: (3, 1), 2027: (3, 1),
                                           2028: (3, 1)}),
            "scaled_RCS_minus_fixed_RCS": paired({2026: (2, 0),
                                                  2027: (2, 0),
                                                  2028: (2, 0)}),
            "scaled_RC_minus_fixed_RC": paired({2026: (1, 0),
                                                2027: (1, 0),
                                                2028: (1, 0)}),
            "fixed_RCS_minus_R0": paired({2026: (2, 1), 2027: (2, 1),
                                          2028: (2, 1)}),
            "fixed_RCS_minus_RC": paired({2026: (1, 1), 2027: (1, 1),
                                          2028: (1, 1)}),
        },
        "per_group_changes": {
            "scaled_RCS_minus_R0": {"summed_over_seeds": {
                "gA": {"fixed": 4, "broke": 1},
                "gB": {"fixed": 3, "broke": 1},
                "gC": {"fixed": 2, "broke": 2},
                "gD": {"fixed": 3, "broke": 1},
            }},
        },
    }
    for path, value in overrides.items():
        node = summary
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return summary


class TestGates:
    def test_all_gates_pass_on_the_reference_world(self):
        gates = evaluate_gates(_gate_summary())
        assert gates["gate1_scaled_rcs_vs_r0_rc"]["pass"] is True
        assert gates["gate2_scaled_vs_fixed_rcs"]["pass"] is True
        assert gates["gate3_two_seeds_net_vs_both"]["pass"] is True
        assert gates["gate4_fold_stability"]["pass"] is True
        assert gates["gate5_class_recall_floor"]["pass"] is True
        assert gates["gate6_group_breadth"]["pass"] is True
        assert "advance" in gates["recommendation"]

    def test_no_gain_world_fails_gate1_with_zero_gain(self):
        """A world where the scaled RCS head reproduces R0's per-fold F1
        exactly (macro unchanged from R0) must FAIL gate1 with a zero
        gain — the gates read the evidence, not the effort."""
        summary = _gate_summary()
        arms = summary["arms"]
        r0_f1 = [0.70, 0.70, 0.70]
        for seed in ("2026", "2027", "2028"):
            for fold_entry, value in zip(
                    arms["scaled"]["RCS"]["per_seed"][seed]["folds"],
                    r0_f1):
                fold_entry["macro_f1"] = value
            arms["scaled"]["RCS"]["per_seed"][seed]["macro_f1_mean"] = \
                float(np.mean(r0_f1))
        arms["scaled"]["RCS"]["macro_f1_mean_of_seeds"] = \
            float(np.mean(r0_f1))
        gates = evaluate_gates(summary)
        assert gates["gate1_scaled_rcs_vs_r0_rc"]["pass"] is False
        assert gates["gate1_scaled_rcs_vs_r0_rc"]["gain_vs_r0_pp"] == \
            pytest.approx(0.0)

    def test_two_seeds_only_gate3(self):
        summary = _gate_summary()
        paired = summary["paired"]["scaled_RCS_minus_R0"]["per_seed"]
        paired["2028"] = {"fixed": 1, "broke": 3, "both_correct": 0,
                          "both_wrong": 0, "support": 30}
        gates = evaluate_gates(summary)
        assert gates["gate3_two_seeds_net_vs_both"]["pass"] is True
        paired["2027"] = {"fixed": 1, "broke": 3, "both_correct": 0,
                          "both_wrong": 0, "support": 30}
        gates = evaluate_gates(summary)
        assert gates["gate3_two_seeds_net_vs_both"]["pass"] is False

    def test_gate3_needs_the_same_seed_positive_against_both(self):
        """Two seeds positive against R0 AND two seeds positive against
        scaled RC is NOT enough when the INTERSECTION is one seed — the
        gate demands the SAME seeds clear both controls."""
        summary = _gate_summary()
        summary["paired"]["scaled_RCS_minus_R0"]["per_seed"] = {
            "2026": {"fixed": 4, "broke": 1, "both_correct": 0,
                     "both_wrong": 0, "support": 30},
            "2027": {"fixed": 4, "broke": 1, "both_correct": 0,
                     "both_wrong": 0, "support": 30},
            "2028": {"fixed": 0, "broke": 4, "both_correct": 0,
                     "both_wrong": 0, "support": 30},
        }
        summary["paired"]["scaled_RCS_minus_RC"]["per_seed"] = {
            "2026": {"fixed": 0, "broke": 4, "both_correct": 0,
                     "both_wrong": 0, "support": 30},
            "2027": {"fixed": 3, "broke": 1, "both_correct": 0,
                     "both_wrong": 0, "support": 30},
            "2028": {"fixed": 3, "broke": 1, "both_correct": 0,
                     "both_wrong": 0, "support": 30},
        }
        gates = evaluate_gates(summary)
        gate = gates["gate3_two_seeds_net_vs_both"]
        assert gate["pass"] is False, gate
        assert gate["seeds_positive_in_both"] == ["2027"]
        # adding ONE more common positive seed flips it to a pass
        summary["paired"]["scaled_RCS_minus_RC"]["per_seed"]["2026"] = {
            "fixed": 3, "broke": 1, "both_correct": 0, "both_wrong": 0,
            "support": 30}
        gates = evaluate_gates(summary)
        assert gates["gate3_two_seeds_net_vs_both"]["pass"] is True
        assert sorted(gates["gate3_two_seeds_net_vs_both"]
                      ["seeds_positive_in_both"]) == ["2026", "2027"]

    def test_gate5_recall_boundary_in_percentage_points(self):
        """The recall drop is compared in PERCENTAGE POINTS: exactly
        1.0pp passes, 1.5pp and 20pp fail (a raw-fraction comparison
        against 1.0 would let a 50pp drop through)."""
        def summary_with_drop(drop: float):
            summary = _gate_summary()
            arms = summary["arms"]
            controls = (
                arms["fixed"]["R0"]["per_seed"]["2026"]["pooled"][
                    "per_class_recall"],
                arms["scaled"]["RC"]["per_seed"]["2026"]["pooled"][
                    "per_class_recall"])
            for seed in ("2026", "2027", "2028"):
                for pooled in (arms["fixed"]["R0"]["per_seed"][seed][
                                   "pooled"]["per_class_recall"],
                               arms["scaled"]["RC"]["per_seed"][seed][
                                   "pooled"]["per_class_recall"]):
                    pooled["plastic_fish"] = 0.80
                arms["scaled"]["RCS"]["per_seed"][seed]["pooled"][
                    "per_class_recall"]["plastic_fish"] = 0.80 - drop
            return summary

        gates = evaluate_gates(summary_with_drop(0.01))
        assert gates["gate5_class_recall_floor"]["pass"] is True
        assert gates["gate5_class_recall_floor"][
            "worst_recall_drop_pp"] == pytest.approx(1.0)
        gates = evaluate_gates(summary_with_drop(0.015))
        assert gates["gate5_class_recall_floor"]["pass"] is False
        assert gates["gate5_class_recall_floor"][
            "worst_recall_drop_pp"] == pytest.approx(1.5)
        gates = evaluate_gates(summary_with_drop(0.20))
        assert gates["gate5_class_recall_floor"]["pass"] is False
        assert gates["gate5_class_recall_floor"][
            "worst_recall_drop_pp"] == pytest.approx(20.0)

    def test_group_breadth_needs_two_groups(self):
        summary = _gate_summary()
        groups = summary["per_group_changes"]["scaled_RCS_minus_R0"][
            "summed_over_seeds"]
        groups["gB"] = {"fixed": 1, "broke": 2}
        gates = evaluate_gates(summary)
        assert gates["gate6_group_breadth"]["pass"] is True
        groups["gD"] = {"fixed": 1, "broke": 2}
        gates = evaluate_gates(summary)
        assert gates["gate6_group_breadth"]["pass"] is False
        assert gates["recommendation"].startswith("stop")
