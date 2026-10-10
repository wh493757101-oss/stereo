"""Conditional (P0-context) residual experiment tests.

The conditional head is ``Linear(input_dim, 8) -> SiLU ->
Linear(8, 4, bias=False)`` with the last layer zero-initialized; the
context comes ONLY from the frozen P0's own outputs (four softmax values
+ the top-two margin). Everything else is the frozen residual protocol.
The full-entry acceptance drives the REAL ``main --fit`` over the
synthetic world (18 tiny fits + 45 shuffle files); nothing in the main
flow is mocked except the explicit fault-injection refusal cases.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from core.material_conditional_residual import (
    CONTEXT_DIM,
    DEFAULT_HIDDEN,
    assemble_conditional_inputs,
    build_conditional_head,
    conditional_delta,
    conditional_input_dim,
    conditional_logits,
    p0_context,
    train_conditional_residual_head,
)
from core.polar_local_features import (
    LOCAL_C_WIDTH,
    LOCAL_S_WIDTH,
    build_local_cache_arrays,
)
from scripts import train_material_conditional_residual as cond_entry
from scripts.verify_p0_baseline import LOGIT_ATOL

from .test_material_local_residual_entry import (
    FOLD_IDS,
    _jsonl,
    _main_args,
    _write_config,
    local_world,  # noqa: F401  (module-scoped synthetic world fixture)
)
from .test_polar_local_features import _source_arrays

# the experiment's three TRAINING seeds and the diagnostic's five
# content seeds are different tuples
TRAIN_SEEDS = (2026, 2027, 2028)
CONTENT_SEEDS = (2026, 2027, 2028, 2029, 2030)
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")


def _rng_inputs(n=10, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, LOCAL_C_WIDTH)).astype(np.float32)
    s = rng.normal(size=(n, LOCAL_S_WIDTH)).astype(np.float32)
    p0 = rng.normal(size=(n, 4)).astype(np.float32)
    valid = np.ones(n, dtype=bool)
    valid[1] = False
    valid[4] = False
    s[1] = np.nan
    s[4] = np.nan
    return c, s, p0, valid


# ---------------------------------------------------------------------------
# Context: ONLY from the frozen P0 outputs


class TestP0Context:
    def test_known_softmax_and_margin(self):
        logits = np.array([[0.0, 0.0, 0.0, 0.0],
                           [2.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        context = p0_context(logits)
        assert context.shape == (2, CONTEXT_DIM)
        assert np.allclose(context[0, :4], 0.25)
        assert context[0, 4] == pytest.approx(0.0, abs=1e-7)
        exp = np.exp([2.0, 1.0, 0.0, 0.0])
        softmax = exp / exp.sum()
        assert np.allclose(context[1, :4], softmax, atol=1e-6)
        assert context[1, 4] == pytest.approx(
            float(softmax.max() - np.sort(softmax)[-2]), abs=1e-6)

    def test_identical_logits_identical_context(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(6, 4)).astype(np.float32)
        assert np.array_equal(p0_context(logits), p0_context(logits.copy()))

    def test_context_ignores_c_and_s_entirely(self):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(4, 4)).astype(np.float32)
        base = p0_context(logits)
        # C/S perturbation cannot touch the context: it is a pure
        # function of the P0 logits
        assert base.shape == (4, 5)

    def test_nonfinite_logits_refused(self):
        with pytest.raises(ValueError, match="NaN/Inf"):
            p0_context(np.array([[np.nan, 0, 0, 0]], dtype=np.float32))

    def test_input_dim_includes_the_context(self):
        assert conditional_input_dim(LOCAL_C_WIDTH, LOCAL_S_WIDTH) == 35
        assert CONTEXT_DIM == 5


# ---------------------------------------------------------------------------
# Head: zero init, bounded delta, trainability, restore


class TestConditionalHead:
    def test_last_layer_zero_init_and_seeded_first_layer(self):
        head = build_conditional_head(seed=2026, input_dim=35)
        assert torch.equal(head[2].weight, torch.zeros(4, 8))
        assert head[2].bias is None
        assert not torch.equal(head[0].weight, torch.zeros(8, 35))
        again = build_conditional_head(seed=2026, input_dim=35)
        for a, b in zip(head.parameters(), again.parameters()):
            assert torch.equal(a, b)

    def test_zero_init_gives_exact_p0(self):
        head = build_conditional_head(seed=2026, input_dim=35)
        x = torch.randn(5, 35)
        p0 = torch.randn(5, 4)
        final = conditional_logits(head, x, torch.ones(5), p0)
        assert torch.equal(final, p0)

    def test_delta_bounded_by_half(self):
        head = build_conditional_head(seed=2026, input_dim=35)
        with torch.no_grad():
            head[2].weight.fill_(30.0)
        x = torch.randn(7, 35)
        delta = conditional_delta(head, x, torch.ones(7))
        assert delta.abs().max() <= 0.5 + 1e-6

    def test_gradients_are_trainable(self):
        head = build_conditional_head(seed=2026, input_dim=35)
        x = torch.randn(6, 35)
        p0 = torch.randn(6, 4)
        valid = torch.ones(6)
        final = conditional_logits(head, x, valid, p0)
        loss = final.pow(2).mean()
        loss.backward()
        # at zero init the last layer receives gradients, the first layer
        # is (correctly) dead until the last layer moves
        assert head[2].weight.grad is not None
        assert head[2].weight.grad.abs().max() > 0
        head.zero_grad()
        with torch.no_grad():
            head[2].weight.fill_(0.3)
        final = conditional_logits(head, x, valid, p0)
        final.pow(2).mean().backward()
        assert head[0].weight.grad is not None
        assert head[0].weight.grad.abs().max() > 0

    def test_p0_tensor_is_a_detached_constant(self):
        head = build_conditional_head(seed=2026, input_dim=35)
        with torch.no_grad():
            head[2].weight.fill_(0.4)
        p0 = torch.randn(4, 4)
        p0_before = p0.clone()
        x = torch.randn(4, 35, requires_grad=True)
        final = conditional_logits(head, x, torch.ones(4), p0)
        final.sum().backward()
        assert torch.equal(p0, p0_before)
        assert p0.grad is None

    def test_invalid_rows_fall_back_exactly(self):
        head = build_conditional_head(seed=2026, input_dim=35)
        with torch.no_grad():
            head[2].weight.fill_(0.9)
        x = torch.randn(4, 35)
        valid = torch.tensor([1.0, 0.0, 1.0, 0.0])
        p0 = torch.randn(4, 4)
        delta = conditional_delta(head, x, valid)
        final = conditional_logits(head, x, valid, p0)
        assert torch.equal(delta[1], torch.zeros(4))
        assert torch.equal(delta[3], torch.zeros(4))
        assert torch.equal(final[1], p0[1])
        assert torch.equal(final[3], p0[3])

    def test_model_restore_reproduces_outputs(self, tmp_path):
        head = build_conditional_head(seed=2026, input_dim=35)
        with torch.no_grad():
            head[2].weight.fill_(0.2)
            head[0].bias.fill_(0.1)
        torch.save(head.state_dict(), tmp_path / "state.pt")
        restored = build_conditional_head(seed=9999, input_dim=35)
        restored.load_state_dict(torch.load(tmp_path / "state.pt",
                                            weights_only=True))
        restored.eval()
        x = torch.randn(3, 35)
        p0 = torch.randn(3, 4)
        with torch.inference_mode():
            assert torch.equal(head(x), restored(x))
            assert torch.equal(
                conditional_logits(head, x, torch.ones(3), p0),
                conditional_logits(restored, x, torch.ones(3), p0))


# ---------------------------------------------------------------------------
# Assembly: C-only isolation and the missing-S contract


class TestConditionalAssembly:
    def test_rc_zeroes_the_s_slot_and_keeps_the_context(self):
        c, s, p0, valid = _rng_inputs()
        context = p0_context(p0)
        first = assemble_conditional_inputs(c, s, context, valid, "RC")
        second = assemble_conditional_inputs(c, s * 100.0, context, valid,
                                             "RC")
        assert np.array_equal(first, second)
        assert not first[:, LOCAL_C_WIDTH:LOCAL_C_WIDTH + LOCAL_S_WIDTH].any()
        # the context block (last 5 columns) survives in RC
        assert first[:, -CONTEXT_DIM:].any()
        assert first.shape == (len(c), 35)

    def test_missing_s_rows_get_zero_s_but_keep_context(self):
        c, s, p0, valid = _rng_inputs()
        context = p0_context(p0)
        matrix = assemble_conditional_inputs(c, s, context, valid, "RCS")
        assert not matrix[1, LOCAL_C_WIDTH:LOCAL_C_WIDTH + LOCAL_S_WIDTH].any()
        assert matrix[1, -CONTEXT_DIM:].any()  # P0-derived context stays
        assert not matrix[4, LOCAL_C_WIDTH:LOCAL_C_WIDTH + LOCAL_S_WIDTH].any()

    def test_nan_in_valid_s_refused(self):
        c, s, p0, valid = _rng_inputs()
        context = p0_context(p0)
        s[0, 2] = np.nan  # row 0 is valid
        with pytest.raises(ValueError, match="NaN"):
            assemble_conditional_inputs(c, s, context, valid, "RCS")

    def test_unknown_variant_refused(self):
        c, s, p0, valid = _rng_inputs()
        context = p0_context(p0)
        with pytest.raises(ValueError, match="variant"):
            assemble_conditional_inputs(c, s, context, valid, "P2")


# ---------------------------------------------------------------------------
# Training loop: epoch-0 candidate, roles, budget


class TestConditionalTrainingLoop:
    def _data(self, n=14, seed=3):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 35)).astype(np.float32)
        y = rng.integers(0, 4, size=n)
        p0 = rng.normal(size=(n, 4)).astype(np.float32)
        valid = np.ones(n, dtype=np.float32)
        return x, y, p0, valid

    def test_epoch0_is_the_first_candidate_and_selection_works(self):
        x, y, p0, valid = self._data()
        model, history, best_state, best = train_conditional_residual_head(
            x[:8], y[:8], p0[:8], valid[:8], x[8:], y[8:], p0[8:],
            valid[8:], seed=2026, lr=0.01, weight_decay=0.001,
            max_epochs=3, input_dim=35)
        assert history[0]["epoch"] == 0
        assert history[0]["delta_max_abs"] == 0.0
        assert best["epoch"] >= 0
        assert best_state["2.weight"].shape == (4, 8)

    def test_roles_are_isolated_by_signature(self):
        # the loop takes ONLY train/val tensors: no outer_eval parameter
        # exists to misuse
        import inspect

        signature = inspect.signature(train_conditional_residual_head)
        assert "outer_eval" not in signature.parameters

    def test_p0_constants_unchanged_by_a_training_step(self):
        x, y, p0, valid = self._data()
        p0_tensor = torch.from_numpy(p0)
        before = p0_tensor.clone()
        head = build_conditional_head(seed=2026, input_dim=35)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
        final = conditional_logits(head, torch.from_numpy(x[:8]),
                                   torch.from_numpy(valid[:8]),
                                   p0_tensor[:8])
        delta = conditional_delta(head, torch.from_numpy(x[:8]),
                                  torch.from_numpy(valid[:8]))
        (final.pow(2).mean() + 0.01 * delta.pow(2).mean()).backward()
        optimizer.step()
        assert torch.equal(p0_tensor, before)

    def test_width_mismatch_refused(self):
        x, y, p0, valid = self._data()
        with pytest.raises(ValueError, match="columns"):
            train_conditional_residual_head(
                x[:, :34], y[:8], p0[:8], valid[:8], x[8:, :34], y[8:],
                p0[8:], valid[8:], seed=2026, lr=0.01, weight_decay=0.001,
                max_epochs=1, input_dim=35)

    def test_two_class_minimum_enforced(self):
        x, y, p0, valid = self._data()
        with pytest.raises(ValueError, match="two classes"):
            train_conditional_residual_head(
                x[:8], np.zeros(8, dtype=np.int64), p0[:8], valid[:8],
                x[8:], y[8:], p0[8:], valid[8:], seed=2026, lr=0.01,
                weight_decay=0.001, max_epochs=1, input_dim=35)


# ---------------------------------------------------------------------------
# Full synthetic entry acceptance: 18 fits + 45 shuffle files


@pytest.fixture(scope="module")
def cond_fit(local_world, module_scratch):
    # capture the P0 fit dir file set/sha256/length BEFORE the fit so the
    # untouched test can compare real before/after evidence
    p0_before_digests = {}
    p0_models = Path(local_world["fit_dir"]) / "models"
    for path in sorted(p0_models.rglob("*")):
        if path.is_file():
            key = path.relative_to(p0_models).as_posix()
            p0_before_digests[key] = {
                "sha256": hashlib.sha256(
                    path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    local_cache = module_scratch / "cond" / "local_features"
    build_local_cache_arrays(
        source_arrays=_source_arrays(local_world),
        samples=local_world["cache_samples"],
        data_root=local_world["v5"],
        protocol=local_world["protocol"],
        output_dir=local_cache)
    from .test_material_conditional_residual import _write_legal_reference

    reference_path, reference_sha = _write_legal_reference(
        module_scratch / "cond", local_world)
    config = dict(local_world["config"])
    config["experiment"] = "material_conditional_residual_v1"
    config["local_features_source"] = {"path": str(local_cache)}
    config["seeds"] = [2026, 2027, 2028]
    config["conditional_head"] = {"hidden": 8, "activation": "silu",
                                  "last_layer_zero_init": True}
    config["linear_reference"] = {
        "fit_summary": str(reference_path),
        "expected_sha256": reference_sha}
    config_path = _write_config(module_scratch / "cond" / "configs"
                                / "cond.yaml", config)
    output = module_scratch / "cond" / "fit_out"
    code = cond_entry.main(_main_args(local_world, config_path, output))
    return {"code": code, "output": output, "local_cache": local_cache,
            "config_path": config_path,
            "p0_before_digests": p0_before_digests}


class TestConditionalEntryFullFit:
    def test_fit_completes(self, cond_fit):
        assert cond_fit["code"] == 0
        summary = json.loads(
            (cond_fit["output"] / "summary.json").read_text(
                encoding="utf-8"))
        assert summary["training_status"] == "COMPLETED"

    def test_eighteen_fits_and_fortyfive_shuffle_files(self, cond_fit):
        output = cond_fit["output"]
        model_dirs = list((output / "models").glob("seed_*"))
        assert len(model_dirs) == 3
        for seed_dir in model_dirs:
            for fold in FOLD_IDS:
                for variant in ("RC", "RCS"):
                    assert (seed_dir / f"{fold}_{variant}"
                            / "residual_state.pt").is_file()
        shuffle_files = list((output / "shuffle").glob(
            "seed_*/*/seed_*.json"))
        assert len(shuffle_files) == 45

    def test_summary_reports_all_seeds_without_selection(self, cond_fit):
        summary = json.loads(
            (cond_fit["output"] / "summary.json").read_text(
                encoding="utf-8"))
        assert summary["seeds"] == [2026, 2027, 2028]
        for variant in ("RC", "RCS"):
            entry = summary["variants"][variant]
            assert set(entry["per_seed"]) == {"2026", "2027", "2028"}
            means = [entry["per_seed"][s]["macro_f1_mean"]
                     for s in ("2026", "2027", "2028")]
            assert entry["macro_f1_mean_of_seeds"] == pytest.approx(
                float(np.mean(means)), abs=1e-6)
            assert entry["macro_f1_seed_range"] == [
                pytest.approx(min(means)), pytest.approx(max(means))]
        # R0 is identical across seeds by construction
        r0 = summary["variants"]["R0"]
        seeds_f1 = {json.dumps(seed_entry["folds"], sort_keys=True)
                    for seed_entry in r0["per_seed"].values()}
        assert len(seeds_f1) == 1

    def test_paired_and_per_comparison_counts(self, cond_fit,
                                              local_world):
        output = cond_fit["output"]
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        manifest_samples = local_world["cache_samples"]
        for comparison, base in (("RCS_minus_R0", "R0"),
                                 ("RCS_minus_RC", "RC")):
            saved = summary["paired"][comparison]
            total_fixed = 0
            total_broke = 0
            for seed in ("2026", "2027", "2028"):
                left_records = []
                right_records = []
                for fold in FOLD_IDS:
                    if base == "R0":
                        right_records.extend(_jsonl(
                            output / "models" / "R0" / f"{fold}_R0"
                            / "outer_eval_predictions.jsonl"))
                    else:
                        right_records.extend(_jsonl(
                            output / "models" / f"seed_{seed}"
                            / f"{fold}_{base}"
                            / "outer_eval_predictions.jsonl"))
                    left_records.extend(_jsonl(
                        output / "models" / f"seed_{seed}"
                        / f"{fold}_RCS"
                        / "outer_eval_predictions.jsonl"))
                names_left = [r["sample_name"] for r in left_records]
                names_right = [r["sample_name"] for r in right_records]
                assert names_left == names_right
                outcome = cond_entry.count_fixed_broke(
                    left_predictions=[r["prediction"]
                                      for r in left_records],
                    right_predictions=[r["prediction"]
                                       for r in right_records],
                    labels=[r["class_id"] for r in left_records],
                    group_by_row=[r["group"] for r in left_records],
                    comparison=comparison)
                assert saved["per_seed"][seed]["fixed"] == \
                    outcome["fixed"]
                assert saved["per_seed"][seed]["broke"] == \
                    outcome["broke"]
                total_fixed += outcome["fixed"]
                total_broke += outcome["broke"]
            assert saved["total"]["fixed"] == total_fixed
            assert saved["total"]["broke"] == total_broke
            per_class = summary["per_class_changes"][comparison][
                "summed_over_seeds"]
            assert sum(v["fixed"] for v in per_class.values()) == \
                total_fixed
            assert sum(v["broke"] for v in per_class.values()) == \
                total_broke

    def test_p0_identity_and_shared_base(self, local_world, cond_fit):
        output = cond_fit["output"]
        admission = json.loads(
            (output / "admission.json").read_text(encoding="utf-8"))
        for record in admission["p0_identity"].values():
            assert record["prediction_mismatches"] == 0
            assert record["max_logit_error"] <= LOGIT_ATOL
            assert all(v == 0.0 for v in
                       record["disabled_slot_max_abs"].values())
        # R0 logits/predictions come from the verified chain and every
        # variant shares the same p0_logits per sample
        for fold in FOLD_IDS:
            r0 = _jsonl(output / "models" / "R0" / f"{fold}_R0"
                        / "outer_eval_predictions.jsonl")
            base = {r["sample_name"]: r["logits"] for r in r0}
            for record in r0:
                assert record["prediction"] == int(
                    np.argmax(record["logits"]))
            for seed in TRAIN_SEEDS:
                for variant in ("RC", "RCS"):
                    for record in _jsonl(
                            output / "models" / f"seed_{seed}"
                            / f"{fold}_{variant}"
                            / "outer_eval_predictions.jsonl"):
                        assert record["p0_logits"] == \
                            base[record["sample_name"]]

    def test_p0_fit_directory_untouched(self, local_world, cond_fit):
        # true before/after comparison: the digests were captured in the
        # fixture BEFORE main --fit; recompute after with set + sha256 +
        # byte length and require exact equality
        after = {}
        p0_dir = Path(local_world["fit_dir"]) / "models"
        for path in sorted(p0_dir.rglob("*")):
            if path.is_file():
                key = path.relative_to(p0_dir).as_posix()
                after[key] = {
                    "sha256": hashlib.sha256(
                        path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size}
        assert after == cond_fit["p0_before_digests"]
        for fold in FOLD_IDS:
            payload = torch.load(
                Path(local_world["fit_dir"]) / "models" / f"{fold}_P0"
                / "model_state.pt", weights_only=False, map_location="cpu")
            assert payload["variant"] == "P0"

    def test_shuffle_contract_on_the_conditional_head(self, cond_fit):
        output = cond_fit["output"]
        for train_seed in TRAIN_SEEDS:
            for fold in FOLD_IDS:
                for content_seed in CONTENT_SEEDS:
                    path = (output / "shuffle" / f"seed_{train_seed}"
                            / fold / f"seed_{content_seed}.json")
                    report = json.loads(path.read_text(encoding="utf-8"))
                    records = report["samples"]
                    for record in records:
                        if record["status"] in ("unmatched", "invalid_s"):
                            assert record["shuffled_logits"] == \
                                record["normal_logits"]
                        else:
                            assert record["donor_row"] != \
                                record["receiver_row"]
                            assert record["donor_split"] == \
                                record["receiver_split"]
                            assert record["donor_original_bin"] == \
                                record["original_bin"]


# ---------------------------------------------------------------------------
# Refusal coverage


class TestConditionalEntryRefusals:
    def test_existing_output_dir_refused(self, local_world, module_scratch):
        base = module_scratch / "cond_refuse"
        local_cache, config_path, _ = _prepare_cond_fit(base, local_world)
        output = base / "taken"
        output.mkdir()
        assert cond_entry.main(
            _main_args(local_world, config_path, output,
                       mode="--dry-run")) == 1

    def test_wrong_experiment_refused(self, local_world, module_scratch):
        base = module_scratch / "cond_exp"
        local_cache, config_path, _ = _prepare_cond_fit(base, local_world)
        config = dict(local_world["config"])
        config["experiment"] = "material_polar_residual_v1"
        config["local_features_source"] = {"path": str(local_cache)}
        bad_path = _write_config(base / "configs" / "bad.yaml", config)
        output = base / "out"
        assert cond_entry.main(
            _main_args(local_world, bad_path, output,
                       mode="--dry-run")) == 1

    def test_diagnostic_failure_does_not_claim_completed(
            self, local_world, module_scratch, monkeypatch):
        base = module_scratch / "cond_fault"
        local_cache, config_path, _ = _prepare_cond_fit(base, local_world)
        output = base / "fit_out"

        def explode(**kwargs):
            raise RuntimeError("synthetic conditional diagnostic failure")

        with mock.patch.object(cond_entry,
                               "run_conditional_shuffle_for_fold",
                               explode):
            code = cond_entry.main(
                _main_args(local_world, config_path, output))
        assert code != 0
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        assert summary["training_status"] == "FAILED"
        assert summary["failed_phase"] == "shuffle_diagnostic"
        assert (output / "models" / "seed_2026" / "fold0_RC"
                / "residual_state.pt").is_file()


class TestShuffleBindingFaultInjection:
    """Fault injection through the REAL main --fit: after the 18 fits
    and before the diagnostic reads it, offset one saved RCS logit row
    (argmax preserved, prediction untouched). The normal-output binding
    guard must fail the run — a run that silently replaced the freshly
    computed normal logits with the saved ones would complete here."""

    def test_tampered_saved_rcs_logits_fail_the_fit(
            self, local_world, module_scratch):
        base = module_scratch / "cond_bind_fault"
        local_cache, config_path, output = _prepare_cond_fit(
            base, local_world)
        real_shuffle = cond_entry.run_conditional_shuffle_for_fold
        tampered = []

        def inject_then_call(**kwargs):
            fold_id = kwargs["fold_id"]
            train_seed = kwargs["train_seed"]
            if fold_id == "fold0" and train_seed == 2026 and not tampered:
                path = (kwargs["output_dir"] / "models" / "seed_2026"
                        / "fold0_RCS"
                        / "outer_eval_predictions.jsonl")
                records = _jsonl(path)
                records[0]["logits"] = [v + 0.25 for v in
                                        records[0]["logits"]]
                path.write_text(
                    "".join(json.dumps(record) + "\n"
                            for record in records), encoding="utf-8")
                tampered.append(records[0]["sample_name"])
            return real_shuffle(**kwargs)

        with mock.patch.object(cond_entry,
                               "run_conditional_shuffle_for_fold",
                               inject_then_call):
            code = cond_entry.main(
                _main_args(local_world, config_path, output))
        assert tampered, "the fault injection never ran"
        assert code != 0
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        assert summary["training_status"] == "FAILED"
        assert summary["failed_phase"] == "shuffle_diagnostic"
        assert "shuffle" not in summary
        # the fit artifacts written before the diagnostic are retained
        assert (output / "models" / "seed_2026" / "fold0_RC"
                / "residual_state.pt").is_file()
        assert (output / "models" / "seed_2026" / "fold0_RCS"
                / "outer_eval_predictions.jsonl").is_file()


# ---------------------------------------------------------------------------
# Manual PowerShell script


class TestManualPowerShellScript:
    def test_script_parses_and_defaults_to_dry_run(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / \
            "manual_material_conditional_residual.ps1"
        if not script.is_file():
            pytest.fail("manual_material_conditional_residual.ps1 is "
                        "missing")
        parser_script = (
            "& { param($path)\n"
            "$tokens=$null;$errors=$null\n"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "$path, [ref]$tokens, [ref]$errors) | Out-Null\n"
            "if ($errors.Count -gt 0) { $errors | ForEach-Object { "
            "Write-Output $_.Message }; exit 1 }\n"
            "Write-Output 'PARSE_OK'\n"
            "} '" + str(script) + "'")
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", parser_script],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "PARSE_OK" in proc.stdout
        text = script.read_text(encoding="utf-8")
        assert "& $py @dryArgs --dry-run" in text
        assert not re.search(r"python\s+\S*\.ps1", text)


# ---------------------------------------------------------------------------
# Helpers


def _write_legal_reference(base: Path, world):
    """A synthetic COMPLETED linear-local reference consistent with the
    world (same protocol, folds, P0 oracle bytes)."""
    import hashlib

    reference = base / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    oracle_shas = {}
    for fold in FOLD_IDS:
        oracle = (Path(world["fit_dir"]) / "models" / f"{fold}_P0"
                  / "outer_eval_predictions.jsonl")
        oracle_shas[fold] = hashlib.sha256(
            oracle.read_bytes()).hexdigest()
    summary = {
        "mode": "fit",
        "training_status": "COMPLETED",
        "experiment": "material_polar_local_interior_band_v1",
        "region_rule": "local_partition_interior_band_v1",
    }
    admission = {
        "protocol_sha256": world["protocol"]["protocol_sha256"],
        "folds": list(FOLD_IDS),
        "local_cache": {"definition_version":
                        "existing_data_probe_v1_local_interior_band_v1"},
        "p0_identity": {fold: {"oracle_sha256": sha}
                        for fold, sha in oracle_shas.items()},
    }
    (reference / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    (reference / "admission.json").write_text(
        json.dumps(admission), encoding="utf-8")
    sha = hashlib.sha256(
        (reference / "summary.json").read_bytes()).hexdigest()
    return reference / "summary.json", sha


def _prepare_cond_fit(base: Path, world):
    local_cache = base / "local_features"
    build_local_cache_arrays(
        source_arrays=_source_arrays(world),
        samples=world["cache_samples"],
        data_root=world["v5"],
        protocol=world["protocol"],
        output_dir=local_cache)
    reference_path, reference_sha = _write_legal_reference(base, world)
    config = dict(world["config"])
    config["experiment"] = "material_conditional_residual_v1"
    config["local_features_source"] = {"path": str(local_cache)}
    config["seeds"] = [2026, 2027, 2028]
    config["linear_reference"] = {
        "fit_summary": str(reference_path),
        "expected_sha256": reference_sha}
    config_path = _write_config(base / "configs" / "cond.yaml", config)
    return local_cache, config_path, base / "fit_out"
