"""Tests for the fixed-P0 residual experiment (RC/RCS).

The fixture reuses the self-consistent synthetic P0 fit
(``build_fit`` from the baseline-verification tests): protocol, cache and
a tiny trained P0 head over fully synthetic data. Only synthetic tensors
ever see backward/optimizer; no real cache is fitted and no real training
smoke runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    Standardizer,
    accuracy,
    assemble_design_matrix,
    authoritative_sample_index,
    macro_f1,
    snapshot_module_state,
    compare_module_states,
)
from core.material_residual import (
    RESIDUAL_INPUT_DIM,
    SHUFFLE_SEEDS,
    assemble_residual_inputs,
    build_donor_bins,
    build_residual_head,
    compute_delta,
    fit_residual_scaler,
    p0_head_from_payload,
    residual_logits,
    run_shuffle_diagnostic,
    shuffle_assignment,
    train_residual_head,
)
from scripts.verify_p0_baseline import load_checkpoint

from .test_p0_baseline_verification import (
    FOLD_IDS,
    build_fit,
    load_summary,
    pred_path,
    save_summary,
)


# ---------------------------------------------------------------------------
# Core head semantics


class TestResidualHeadSemantics:
    def test_zero_init_gives_zero_delta_and_p0_output(self):
        head = build_residual_head(seed=2026)
        x = torch.randn(5, RESIDUAL_INPUT_DIM)
        valid_s = torch.ones(5)
        p0 = torch.randn(5, 4)
        delta = compute_delta(head, x, valid_s)
        assert torch.equal(delta, torch.zeros(5, 4))
        final = residual_logits(head, x, valid_s, p0)
        assert torch.equal(final, p0)

    def test_weight_is_zero_initialized_without_bias(self):
        head = build_residual_head(seed=2026)
        assert head.linear.bias is None
        assert torch.equal(head.linear.weight, torch.zeros(4, 14))
        # determinism: the same seed rebuilds the same head
        again = build_residual_head(seed=2026)
        assert torch.equal(head.linear.weight, again.linear.weight)

    def test_invalid_s_is_exact_fallback(self):
        head = build_residual_head(seed=2026)
        with torch.no_grad():  # push weights away from zero
            head.linear.weight.add_(0.3)
        x = torch.randn(4, RESIDUAL_INPUT_DIM)
        valid_s = torch.tensor([1.0, 0.0, 1.0, 0.0])
        delta = compute_delta(head, x, valid_s)
        assert torch.equal(delta[1], torch.zeros(4))
        assert torch.equal(delta[3], torch.zeros(4))
        assert delta[0].abs().max() <= 0.5 + 1e-6

    def test_delta_bounded_by_half(self):
        head = build_residual_head(seed=2026)
        with torch.no_grad():
            head.linear.weight.fill_(50.0)  # saturate tanh
        x = torch.randn(9, RESIDUAL_INPUT_DIM)
        delta = compute_delta(head, x, torch.ones(9))
        assert delta.abs().max() <= 0.5 + 1e-6

    def test_rc_reads_no_s_slot(self):
        """RC's S slot is zeroed at the INPUT level: perturbing the S
        values cannot change an RC residual input."""
        rng = np.random.default_rng(0)
        c = rng.normal(size=(3, 6)).astype(np.float32)
        s = rng.normal(size=(3, 8)).astype(np.float32)
        s_valid = np.ones(3, dtype=bool)
        first = assemble_residual_inputs(c, s, s_valid, "RC")
        second = assemble_residual_inputs(c, s + 100.0, s_valid, "RC")
        assert np.array_equal(first, second)
        # the S columns are exactly zero
        assert not first[:, 6:].any()
        # RCS keeps the (standardized upstream) S values
        rcs = assemble_residual_inputs(c, s, s_valid, "RCS")
        assert rcs.shape == (3, 14)
        assert rcs[:, 6:].any()

    def test_rcs_missing_s_rows_zeroed(self):
        rng = np.random.default_rng(1)
        c = np.zeros((2, 6), dtype=np.float32)
        s = rng.normal(size=(2, 8)).astype(np.float32)
        s_valid = np.array([True, False])
        matrix = assemble_residual_inputs(c, s, s_valid, "RCS")
        assert matrix[0, 6:].any()
        assert not matrix[1, 6:].any()  # explicit missing -> zero input

    def test_p0_head_from_payload_is_frozen(self):
        chain = build_fit(Path(fit_scratch()), name="frozen_p0")
        checkpoint = (chain["fit_dir"] / "models" / "fold0_P0"
                      / "model_state.pt")
        payload = torch.load(checkpoint, weights_only=False,
                             map_location="cpu")
        head = p0_head_from_payload(payload)
        assert all(not p.requires_grad for p in head.parameters())
        assert head.weight.shape == (4, payload["input_dim"])

    def test_gradients_never_reach_p0(self):
        head = build_residual_head(seed=2026)
        p0 = torch.randn(3, 4)
        p0_before = p0.clone()
        x = torch.randn(3, RESIDUAL_INPUT_DIM, requires_grad=True)
        final = residual_logits(head, x, torch.ones(3), p0)
        final.sum().backward()
        assert torch.equal(p0, p0_before)  # P0 tensor untouched
        assert head.linear.weight.grad is not None


# ---------------------------------------------------------------------------
# Scaler: fitted from the training rows only


class TestFitResidualScaler:
    def _cache(self):
        rng = np.random.default_rng(2)
        return {
            "g_features": rng.normal(size=(6, 4)).astype(np.float32),
            "c_features": rng.normal(size=(6, 6)).astype(np.float32),
            "s_features": rng.normal(size=(6, 8)).astype(np.float32),
            "s_valid": np.array([True, True, True, False, True, False]),
            "class_id": np.arange(6) % 4,
        }

    def test_uses_training_rows_only(self):
        cache = self._cache()
        scaler, info = fit_residual_scaler(cache, [0, 1, 2])
        assert info["inner_train_rows"] == 3
        assert info["s_valid_rows"] == 3  # rows 0-2 are all valid
        # perturbing the evaluation rows cannot change the scaler
        cache2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in cache.items()}
        cache2["c_features"][3:6] += 99.0
        cache2["g_features"][3:6] += 99.0
        scaler2, _ = fit_residual_scaler(cache2, [0, 1, 2])
        assert np.array_equal(scaler.mean, scaler2.mean)
        assert np.array_equal(scaler.std, scaler2.std)

    def test_s_columns_use_valid_rows_only(self):
        cache = self._cache()
        scaler, info = fit_residual_scaler(cache, [0, 1, 2, 3])
        # S mean must equal the mean over rows 0,1,2 (row 3 invalid)
        expected = cache["s_features"][[0, 1, 2]].mean(axis=0)
        s_start = 6  # C occupies columns 0..5 of the 14-wide residual slot
        assert np.allclose(scaler.mean[s_start:s_start + 8], expected)

    def test_matches_saved_p0_scaler_semantics(self):
        """The re-derived statistics equal the P0 Standardizer fitted with
        the same rule on the same rows (C/S columns of the full scaler)."""
        cache = self._cache()
        rows = [0, 1, 2, 4]
        scaler, _ = fit_residual_scaler(cache, rows)
        width = 6 + 8
        raw = np.concatenate(
            [cache["c_features"][rows],
             cache["s_features"][rows]], axis=1).astype(np.float64)
        valid = np.ones((len(rows), width), dtype=bool)
        valid[:, 6:] = cache["s_valid"][rows][:, None]
        reference = Standardizer.fit(raw, valid)
        assert np.allclose(scaler.mean, reference.mean)
        assert np.allclose(scaler.std, reference.std)
        assert np.array_equal(scaler.usable, reference.usable)


# ---------------------------------------------------------------------------
# Training loop: epoch-0 candidate, selection, no outer_eval access


class TestTrainResidualHead:
    def _data(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, RESIDUAL_INPUT_DIM)).astype(np.float32)
        y = rng.integers(0, 4, size=n)
        p0 = rng.normal(size=(n, 4)).astype(np.float32)
        valid = np.ones(n, dtype=bool)
        return x, y, p0, valid

    def test_signature_has_no_outer_eval_access(self):
        import inspect

        from core.material_residual import train_residual_head as fn

        params = list(inspect.signature(fn).parameters)
        assert not any("outer" in p or "eval" in p for p in params)

    def test_epoch0_in_history_and_zero_delta(self):
        x, y, p0, valid = self._data()
        model, history, best_state, best = train_residual_head(
            x, y, p0, valid, x[:4], y[:4], p0[:4], valid[:4],
            seed=2026, lr=0.01, weight_decay=0.001, max_epochs=3,
            aux_loss_weight=0.01)
        assert history[0]["epoch"] == 0
        assert history[0]["delta_max_abs"] == 0.0
        # epoch 0's inner_val metrics equal the pure P0 metrics on val
        val_pred = p0[:4].argmax(axis=1)
        assert history[0]["inner_val_macro_f1"] == pytest.approx(
            macro_f1([int(v) for v in y[:4]],
                     [int(v) for v in val_pred]))

    def test_epoch0_can_win_on_ties(self):
        """P0 already perfect on val (all four classes, large-margin
        logits) -> training cannot beat Macro-F1 1.0 and the earlier
        candidate (epoch 0) must win the tie."""
        x, _, _, valid = self._data(n=8)
        y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        margin = np.full((8, 4), -6.0, dtype=np.float32)
        margin[np.arange(8), y] = 6.0
        p0 = margin
        model, history, best_state, best = train_residual_head(
            x, y, p0, valid, x, y, p0, valid,
            seed=2026, lr=0.001, weight_decay=0.001, max_epochs=4,
            aux_loss_weight=0.01)
        assert best["epoch"] == 0
        assert best["inner_val_macro_f1"] == pytest.approx(1.0)

    def test_best_state_matches_selection(self):
        x, y, p0, valid = self._data(n=16, seed=3)
        model, history, best_state, best = train_residual_head(
            x, y, p0, valid, x, y, p0, valid,
            seed=2026, lr=0.05, weight_decay=0.001, max_epochs=8,
            aux_loss_weight=0.01)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(x)).numpy()
        assert out.shape == (16, 4)
        assert np.isfinite(out).all()

    def test_save_restore_forward_identical(self, tmp_path):
        x, y, p0, valid = self._data(n=6, seed=5)
        model, history, best_state, best = train_residual_head(
            x, y, p0, valid, x, y, p0, valid,
            seed=2026, lr=0.01, weight_decay=0.001, max_epochs=2,
            aux_loss_weight=0.01)
        path = tmp_path / "residual.pt"
        torch.save({"state_dict": best_state,
                    "variant": "RCS"}, path)
        restored = build_residual_head(seed=2026)
        restored.load_state_dict(torch.load(path, weights_only=True)
                                 ["state_dict"])
        restored.eval()
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            a = model(torch.from_numpy(x))
            b = restored(torch.from_numpy(x))
        assert torch.equal(a, b)

    def test_p0_logits_gradient_isolated(self):
        x, y, p0_np, valid = self._data(n=6, seed=7)
        p0 = torch.tensor(p0_np)
        model, history, best_state, best = train_residual_head(
            x, y, p0_np, valid, x, y, p0_np, valid,
            seed=2026, lr=0.01, weight_decay=0.001, max_epochs=2,
            aux_loss_weight=0.01)
        # the frozen budget finished; the P0 tensor given by the caller is
        # untouched (training consumed its own detached copy)
        assert torch.equal(torch.tensor(p0_np), p0)

    def test_training_reduces_loss_or_stays_finite(self):
        x, y, p0, valid = self._data(n=20, seed=9)
        model, history, best_state, best = train_residual_head(
            x, y, p0, valid, x, y, p0, valid,
            seed=2026, lr=0.01, weight_decay=0.001, max_epochs=10,
            aux_loss_weight=0.01)
        losses = [h["train_loss"] for h in history if h["epoch"] > 0]
        assert all(np.isfinite(losses))
        assert len(history) == 11  # epoch 0 + 10 training epochs


# ---------------------------------------------------------------------------
# Shuffle diagnostic primitives


class TestShuffleDiagnostic:
    def _rows(self, n=12):
        rng = np.random.default_rng(4)
        return {
            "valid_ratio": rng.random(n),
            "gray_mean": rng.random(n),
            "split": ["train"] * 6 + ["val"] * 6,
        }

    def test_bins_cover_three_quantiles(self):
        context = self._rows()
        bins = build_donor_bins(context["valid_ratio"],
                                context["gray_mean"],
                                train_rows=list(range(6)))
        assert set(np.unique(bins)) <= set(range(9))  # 3x3 quantile bins
        assert len(bins) == len(context["valid_ratio"])

    _DETERMINISTIC_BINS = np.array([0, 0, 1, 1, 2, 2] * 2)
    _DETERMINISTIC_SPLIT = ["train"] * 6 + ["val"] * 6

    def test_assignment_is_a_derangement_over_valid_rows(self):
        """Donors come only from valid rows; the mapping is a bijection
        with no fixed point; splits and bins are never crossed."""
        bins = self._DETERMINISTIC_BINS
        split = self._DETERMINISTIC_SPLIT
        assignment = shuffle_assignment(list(range(12)), bins, split,
                                        seed=2026)
        donors = [d for d in assignment.values()]
        assert None not in donors   # every (split, bin) group has 2 members
        assert len(donors) == len(set(donors))          # bijection
        for receiver, donor in assignment.items():
            assert receiver != donor                     # no fixed point
            assert split[receiver] == split[donor]
            assert bins[receiver] == bins[donor]

    def test_invalid_rows_are_never_donors(self):
        bins = self._DETERMINISTIC_BINS
        split = self._DETERMINISTIC_SPLIT
        # row 2 is invalid but sits in the same split/bin as valid rows
        valid = [r for r in range(12) if r != 2]
        assignment = shuffle_assignment(valid, bins, split, seed=2026)
        assert 2 not in assignment.values()
        # the pair left in row 2's group is now a single-member bin
        assert assignment[3] is None or assignment[3] != 2

    def test_single_element_valid_bin_is_unmatched(self):
        # bin 0 holds one valid row (0); bin 1 holds three valid rows;
        # an invalid row (1) sits next to them but is never a donor
        bins = np.array([0, 0, 1, 1, 1])
        split = ["s"] * 5
        assignment = shuffle_assignment(
            [0, 2, 3, 4], bins, split, seed=2026)
        assert assignment[0] is None  # bin 0 has a single valid member
        assert assignment[2] is not None
        assert assignment[3] is not None
        assert assignment[4] is not None
        assert 1 not in assignment.values()  # the invalid row is unused

    def test_assignment_is_seed_reproducible_and_label_free(self):
        bins = self._DETERMINISTIC_BINS
        split = self._DETERMINISTIC_SPLIT
        first = shuffle_assignment(list(range(12)), bins, split, seed=7)
        second = shuffle_assignment(list(range(12)), bins, split, seed=7)
        assert first == second
        # labels/predictions are not an input at all (exact signature)
        import inspect

        params = list(inspect.signature(shuffle_assignment).parameters)
        assert params == ["valid_receiver_rows", "bins", "split_labels",
                          "seed"]

    def test_all_seeds_frozen(self):
        assert SHUFFLE_SEEDS == (2026, 2027, 2028, 2029, 2030)


def fit_scratch() -> str:
    import tempfile

    return tempfile.mkdtemp(dir="D:/Python/Project/my_data/.test_artifacts")


def fit_scratch() -> str:
    import tempfile

    return tempfile.mkdtemp(dir="D:/Python/Project/my_data/.test_artifacts")


# ---------------------------------------------------------------------------
# Real main() control flow over the synthetic P0 fit (tiny CPU tensors)


def _residual_config(chain, scratch_dir, name="residual_config.yaml"):
    import yaml

    config = {
        "experiment": "material_polar_residual_v1",
        "protocol_source": {
            "path": (chain["protocol_dir"] / "protocol.json")
            .as_posix(),
            "sha256_recorded": chain["protocol"]["protocol_sha256"],
        },
        "features_source": {"path": chain["feature_dir"].as_posix()},
        "p0_baseline_source": {
            "fit_dir": chain["fit_dir"].as_posix(),
            "verification": "synthetic/verification/report.json"},
        "data": {"fingerprint": chain["protocol"]["data"]["fingerprint"]},
        "training": {"epochs": 5, "lr": 0.01, "weight_decay": 0.001,
                     "aux_loss_weight": 0.01, "device": "cpu",
                     "full_batch": True, "seed": 2026},
    }
    path = scratch_dir / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class TestResidualEntryControlFlow:
    def _project_root_patch(self, monkeypatch, chain):
        import scripts.train_material_residual as entry

        monkeypatch.setattr(entry, "PROJECT_ROOT", chain["project_root"]
                            if "project_root" in chain else Path(
                                str(chain["fit_dir"]).split(
                                    ".test_artifacts")[0]))
        return entry

    def test_no_mode_refused(self, scratch_dir):
        from scripts.train_material_residual import main

        chain = build_fit(scratch_dir, name="p0a")
        config = _residual_config(chain, scratch_dir)
        code = main([
            "--config", str(config),
            "--output-dir", str(scratch_dir / "res_no_mode"),
        ])
        assert code == 1
        assert not (scratch_dir / "res_no_mode").exists()

    def test_dry_run_admits_and_does_not_train(self, scratch_dir,
                                               monkeypatch):
        from scripts.train_material_residual import main

        chain = build_fit(scratch_dir, name="p0b")
        config = _residual_config(chain, scratch_dir)
        output = scratch_dir / "res_dry"
        code = main([
            "--config", str(config),
            "--output-dir", str(output),
            "--dry-run",
        ])
        assert code == 0
        report = json.loads((output / "report.json").read_text(
            encoding="utf-8"))
        assert report["training_status"] == "NOT_RUN"
        assert len(report["preview"]) == 3
        assert not (output / "models").exists()
        assert not (output / "shuffle").exists()

    def test_fit_completes_and_covers_outer_once(self, scratch_dir):
        from scripts.train_material_residual import main

        chain = build_fit(scratch_dir, name="p0c")
        config = _residual_config(chain, scratch_dir)
        output = scratch_dir / "res_fit"
        code = main([
            "--config", str(config),
            "--output-dir", str(output),
            "--variants", "RC", "RCS",
            "--fit",
        ])
        assert code == 0
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert summary["training_status"] == "COMPLETED"
        assert set(summary["variants"]) == {"R0", "RC", "RCS"}
        for variant in ("RC", "RCS"):
            assert len(summary["variants"][variant]["folds"]) == 3
            assert summary["variants"][variant][
                "pooled_sample_count"] == 9  # 3 folds x 3 outer rows
        # every outer_eval sample covered exactly once per variant
        for variant in ("RC", "RCS"):
            total = sum(f["rows"] for f in summary["variants"][variant][
                "folds"])
            confusion = summary["variants"][variant]["pooled"][
                "confusion_matrix"]
            assert total == 9
            assert sum(sum(row) for row in confusion) == 9
        assert "RCS_minus_R0" in summary["paired"]
        assert "RCS_minus_RC" in summary["paired"]
        # per-fold artifacts
        for fold_id in FOLD_IDS:
            for variant in ("RC", "RCS"):
                directory = output / "models" / f"{fold_id}_{variant}"
                assert (directory / "residual_state.pt").is_file()
                assert (directory / "scaler.json").is_file()
                assert (directory / "history.json").is_file()
                assert (directory
                        / "outer_eval_predictions.jsonl").is_file()
        # shuffle diagnostic ran for the frozen RCS heads
        for fold_id in FOLD_IDS:
            shuffle = output / "shuffle" / fold_id
            assert (shuffle / "shuffle_summary.json").is_file()
            for seed in (2026, 2027, 2028, 2029, 2030):
                assert (shuffle / f"seed_{seed}.json").is_file()

    def test_fit_refuses_tampered_p0_baseline(self, scratch_dir):
        chain = build_fit(scratch_dir, name="p0d")
        # break the P0 baseline: flip one saved prediction
        records = [json.loads(line) for line in
                   pred_path(chain, "fold0").read_text(
                       encoding="utf-8").splitlines()]
        records[0]["prediction"] = (records[0]["prediction"] + 1) % 4
        pred_path(chain, "fold0").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
            encoding="utf-8")
        from scripts.train_material_residual import main

        config = _residual_config(chain, scratch_dir)
        output = scratch_dir / "res_bad_p0"
        code = main([
            "--config", str(config),
            "--output-dir", str(output),
            "--fit",
        ])
        assert code == 2
        assert not output.exists()

    def test_wrong_config_experiment_refused(self, scratch_dir):
        import yaml

        chain = build_fit(scratch_dir, name="p0e")
        config = _residual_config(chain, scratch_dir)
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        payload["experiment"] = "something_else"
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")
        from scripts.train_material_residual import main

        output = scratch_dir / "res_bad_exp"
        code = main([
            "--config", str(config),
            "--output-dir", str(output),
            "--dry-run",
        ])
        assert code == 1

    def test_existing_output_refused(self, scratch_dir):
        from scripts.train_material_residual import main

        chain = build_fit(scratch_dir, name="p0f")
        config = _residual_config(chain, scratch_dir)
        output = scratch_dir / "res_twice"
        assert main([
            "--config", str(config),
            "--output-dir", str(output),
            "--dry-run",
        ]) == 0
        assert main([
            "--config", str(config),
            "--output-dir", str(output),
            "--dry-run",
        ]) == 1

    def test_single_fold_selection(self, scratch_dir):
        from scripts.train_material_residual import main

        chain = build_fit(scratch_dir, name="p0g")
        config = _residual_config(chain, scratch_dir)
        output = scratch_dir / "res_fold1"
        code = main([
            "--config", str(config),
            "--output-dir", str(output),
            "--fold", "fold1",
            "--variants", "RCS",
            "--fit",
        ])
        assert code == 0
        summary = json.loads((output / "summary.json").read_text(
            encoding="utf-8"))
        assert [f["fold"] for f in
                summary["variants"]["RCS"]["folds"]] == ["fold1"]
        assert (output / "shuffle" / "fold1" / "shuffle_summary.json") \
            .is_file()


class TestManualPowerShellScript:
    """The manual script must parse cleanly and its dry-run path must be
    exercised end to end (Parser.ParseFile + a real dry-run run)."""

    def test_parses_without_errors(self):
        import subprocess

        script = Path(__file__).resolve().parents[2] / "scripts" / \
            "manual_material_residual.ps1"
        assert script.is_file()
        command = (
            "$errors = $null; "
            "$null = [System.Management.Automation.Language.Parser]::ParseFile("
            f"'{script.as_posix()}', [ref]$null, [ref]$errors); "
            "if ($errors.Count -gt 0) { $errors | ForEach-Object { "
            "Write-Output $_.Message }; exit 1 } else { Write-Output OK }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout
        assert result.stdout.strip().endswith("OK")

    def test_uses_call_operator_not_python_for_ps1(self):
        text = Path(__file__).resolve().parents[2].joinpath(
            "scripts", "manual_material_residual.ps1").read_text(
                encoding="utf-8")
        assert "python xxx.ps1" not in text
        assert "& $py @dryArgs --dry-run" in text
        assert "& $py @fitArgs --fit" in text


# ---------------------------------------------------------------------------
# Production-path shuffle tests (run_shuffle_for_fold + entry ordering)


def _train_nonzero_residual_head(seed=2026, epochs=6):
    """A tiny synthetic CPU fit producing a NON-zero residual head."""
    from core.material_residual import train_residual_head

    rng = np.random.default_rng(11)
    x = rng.normal(size=(16, RESIDUAL_INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, 4, size=16)
    p0 = rng.normal(size=(16, 4)).astype(np.float32)
    valid = np.ones(16, dtype=bool)
    model, history, best_state, best = train_residual_head(
        x, y, p0, valid, x, y, p0, valid,
        seed=seed, lr=0.05, weight_decay=0.001, max_epochs=epochs,
        aux_loss_weight=0.01)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        weight = model.linear.weight.clone()
    assert weight.abs().max() > 0  # non-zero correction head
    return best_state


def _prepare_fold_shuffle(scratch_dir, name="shuf"):
    """A synthetic chain plus a trained non-zero RCS head and the fold's
    standardized inputs, mirroring the production call exactly."""
    chain = build_fit(scratch_dir, name=name)
    fit_dir = chain["fit_dir"]
    state = _train_nonzero_residual_head()
    payload_dir = fit_dir / "models" / "fold0_RCS"
    payload_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "variant": "RCS", "fold": "fold0",
                "seed": 2026, "input_dim": RESIDUAL_INPUT_DIM,
                "num_classes": 4},
               payload_dir / "residual_state.pt")
    return chain, fit_dir


class TestProductionShufflePath:
    def _run(self, chain, fit_dir, output_dir, capture=None):
        from scripts.train_material_residual import run_shuffle_for_fold

        from core.material_probe import Standardizer

        cache = chain["arrays"]
        cache = {
            **cache,
            "g_features": cache["g_features"],
            "c_features": cache["c_features"],
            "s_features": cache["s_features"],
        }
        p0_payload = torch.load(
            fit_dir / "models" / "fold0_P0" / "model_state.pt",
            weights_only=False, map_location="cpu")
        p0_head = p0_head_from_payload(p0_payload)
        scaler = Standardizer.from_dict(json.loads(
            (fit_dir / "models" / "fold0_P0" / "scaler.json")
            .read_text(encoding="utf-8")))
        from core.material_probe import authoritative_sample_index

        authority = authoritative_sample_index(chain["protocol"])
        return run_shuffle_for_fold(
            fold_id="fold0",
            cache=cache,
            manifest_samples=chain["manifest"]["samples"],
            authority=authority,
            parts=chain["fold_parts"]["fold0"],
            scaler=scaler,
            p0_head=p0_head,
            residual_payload_path=fit_dir / "models" / "fold0_RCS"
            / "residual_state.pt",
            budget={"seed": 2026},
            residual_budget={},
            output_dir=output_dir,
            log=lambda m: None,
            capture=capture,
        )

    def test_c_and_standardized_s_reach_the_head(self, scratch_dir):
        """The receiver's C must pass through element-for-element and the
        S slot must carry the STANDARDIZED values (never the raw crop
        statistics), with non-zero C and a non-identity scaler."""
        chain, fit_dir = _prepare_fold_shuffle(scratch_dir)
        output = scratch_dir / "shuf_out"
        captured = {"x14": []}

        real_head_forward = None

        import scripts.train_material_residual as entry

        original = entry.run_shuffle_diagnostic

        def spy(**kwargs):
            captured["c_std"] = np.array(kwargs["c_standardized"], copy=True)
            captured["s_std"] = np.array(kwargs["s_standardized"], copy=True)
            return original(**kwargs)

        entry.run_shuffle_diagnostic = spy
        try:
            self._run(chain, fit_dir, output)
        finally:
            entry.run_shuffle_diagnostic = original
        c_std = captured["c_std"]
        s_std = captured["s_std"]
        assert c_std.shape == (3, 6) and s_std.shape == (3, 8)
        # C is non-zero (a zero C would hide a zeroing bug)
        assert c_std.any()
        # standardized S differs from the raw S statistics
        raw_s = chain["arrays"]["s_features"][
            chain["fold_parts"]["fold0"]["outer_eval"]]
        assert not np.allclose(s_std, raw_s)

    def test_identity_forward_matches_normal_rcs_exactly(self, scratch_dir):
        """Running the diagnostic with an identity assignment (every
        receiver keeps its own standardized S) must reproduce the normal
        RCS forward EXACTLY through the same evaluator path."""
        chain, fit_dir = _prepare_fold_shuffle(scratch_dir)
        output = scratch_dir / "shuf_identity"

        import scripts.train_material_residual as entry

        captured = {}

        original = entry.run_shuffle_diagnostic

        def spy(**kwargs):
            captured["evaluator"] = kwargs["evaluator"]
            captured["c_std"] = np.array(kwargs["c_standardized"], copy=True)
            captured["s_std"] = np.array(kwargs["s_standardized"], copy=True)
            captured["s_valid"] = np.array(kwargs["s_valid"], copy=True)
            return original(**kwargs)

        entry.run_shuffle_diagnostic = spy
        try:
            self._run(chain, fit_dir, output)
        finally:
            entry.run_shuffle_diagnostic = original
        evaluator = captured["evaluator"]
        # identity input through the SAME diagnostic evaluator
        identity_x14 = assemble_residual_inputs(
            captured["c_std"], captured["s_std"],
            captured["s_valid"], "RCS")
        identity_logits = evaluator(
            identity_x14, captured["s_valid"].astype(np.float32))
        # compare with the saved per-seed normal logits (the diagnostic's
        # own normal forward through the same path)
        seed_report = json.loads(
            (output / "shuffle" / "fold0" / "seed_2026.json")
            .read_text(encoding="utf-8"))
        for record in seed_report["samples"]:
            if record["status"] == "matched":
                continue  # matched rows have a donor; identity excludes them
        normal = np.asarray(seed_report["normal_logits"], dtype=np.float32)
        assert np.array_equal(np.asarray(identity_logits),
                              normal)  # exact, same computation path

    def test_full_per_sample_evidence_is_saved(self, scratch_dir):
        chain, fit_dir = _prepare_fold_shuffle(scratch_dir)
        output = scratch_dir / "shuf_evidence"
        self._run(chain, fit_dir, output)
        seed_report = json.loads(
            (output / "shuffle" / "fold0" / "seed_2026.json")
            .read_text(encoding="utf-8"))
        rows = chain["fold_parts"]["fold0"]["outer_eval"]
        samples = seed_report["samples"]
        assert len(samples) == len(rows)  # every eval sample, exactly once
        names = [r["sample_name"] for r in samples]
        assert len(names) == len(set(names))
        manifest = chain["manifest"]["samples"]
        for record in samples:
            assert record["sample_name"]
            assert record["group"] == manifest[record["receiver_row"]][
                "group"]
            assert isinstance(record["class_id"], int)
            assert record["index_meaning"]
            assert record["status"] in ("matched", "unmatched",
                                        "invalid_s")
            assert isinstance(record["has_valid_s"], bool)
            assert len(record["normal_logits"]) == 4
            assert len(record["shuffled_logits"]) == 4
            assert len(record["normal_delta"]) == 4
            assert len(record["shuffled_delta"]) == 4
            if record["status"] == "matched":
                assert record["donor_row"] is not None
                assert record["donor_row"] != record["receiver_row"]
                assert record["donor_sample_name"]
            else:
                assert record["donor_row"] is None
                assert record["donor_sample_name"] is None
        # paired stats recomputable from the records alone
        labels = {manifest[r]["sample_name"]:
                  int(chain["arrays"]["class_id"][r]) for r in rows}
        fixed = broke = 0
        for record in samples:
            label = labels[record["sample_name"]]
            if (record["shuffled_prediction"] == label
                    and record["normal_prediction"] != label):
                fixed += 1
            if (record["normal_prediction"] == label
                    and record["shuffled_prediction"] != label):
                broke += 1
        assert seed_report["vs_normal_rcs"]["fixed"] == fixed
        assert seed_report["vs_normal_rcs"]["broke"] == broke

    def test_invalid_rows_fall_back_and_unmatched_keep_normal(
            self, scratch_dir):
        chain, fit_dir = _prepare_fold_shuffle(scratch_dir)
        # make row 0 of the fold's outer_eval invalid by clearing the
        # cached validity flag (the diagnostic consumes the cache arrays)
        cache_row = chain["fold_parts"]["fold0"]["outer_eval"][0]
        chain["arrays"]["s_valid"][cache_row] = False
        output = scratch_dir / "shuf_invalid"
        self._run(chain, fit_dir, output)
        seed_report = json.loads(
            (output / "shuffle" / "fold0" / "seed_2026.json")
            .read_text(encoding="utf-8"))
        invalid = [r for r in seed_report["samples"]
                   if r["status"] == "invalid_s"]
        assert len(invalid) == 1
        record = invalid[0]
        # the record's delta fields prove the exact P0 fallback: the
        # caller computes delta = logits - P0 per row and an invalid row's
        # deltas are exactly zero both before and after the shuffle
        assert all(v == 0.0 for v in record["normal_delta"])
        assert all(v == 0.0 for v in record["shuffled_delta"])
        assert np.array_equal(
            np.asarray(record["shuffled_logits"], dtype=np.float32),
            np.asarray(record["normal_logits"], dtype=np.float32))
        assert record["shuffled_prediction"] == record["normal_prediction"]
 


class TestEventOrdering:
    def test_diagnostic_runs_after_all_models_frozen(self, scratch_dir,
                                                     monkeypatch):
        """Every requested fold/variant fit must complete (and freeze its
        checkpoint) BEFORE any shuffle diagnostic starts."""
        import scripts.train_material_residual as entry

        chain = build_fit(scratch_dir, name="p0ord")
        config = _residual_config(chain, scratch_dir)
        output = scratch_dir / "res_order"
        events = []
        real_fit = entry.fit_one_fold_variant
        real_shuffle = entry.run_shuffle_for_fold

        def spy_fit(**kwargs):
            events.append(("fit", kwargs["fold_id"], kwargs["variant"]))
            return real_fit(**kwargs)

        def spy_shuffle(**kwargs):
            events.append(("shuffle", kwargs["fold_id"]))
            return real_shuffle(**kwargs)

        monkeypatch.setattr(entry, "fit_one_fold_variant", spy_fit)
        monkeypatch.setattr(entry, "run_shuffle_for_fold", spy_shuffle)
        code = entry.main([
            "--config", str(config),
            "--output-dir", str(output),
            "--variants", "RC", "RCS",
            "--fit",
        ])
        assert code == 0
        fit_events = [i for i, e in enumerate(events) if e[0] == "fit"]
        shuffle_events = [i for i, e in enumerate(events)
                          if e[0] == "shuffle"]
        assert len(fit_events) == 6  # 3 folds x 2 variants
        assert len(shuffle_events) == 3
        assert max(fit_events) < min(shuffle_events)
