"""P0 input identity for the local residual entry.

The frozen P0 uses ONLY the standardized G columns: after the scaler
transform, ``assemble_design_matrix(..., "P0")`` must zero the C/S
slots. The pre-fix local entry fed the full standardized G+C+S into the
frozen head — the weights never changed, but the forward input did.

The independent oracle here is ALWAYS built from the correct assembly
(the same construction ``scripts/verify_p0_baseline.py`` uses) or from a
frozen prediction file — never from the entry's own helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    Standardizer,
    assemble_design_matrix,
)
from core.material_residual import (
    compute_delta,
    residual_logits,
)
from scripts.verify_p0_baseline import LOGIT_ATOL
from scripts import train_material_local_residual as local_entry

from .test_polar_local_features import (
    C_LOCAL_KEYS,
)
from .test_material_local_residual_entry import (
    _main_args,
    _prepare_fit,
    local_world,  # noqa: F401  (module-scoped synthetic world fixture)
)


# ---------------------------------------------------------------------------
# Synthetic P0 world: nonzero C/S content and a head with nonzero
# disabled-slot weights (build_probe_head's random init guarantees this
# for the fixture worlds; here it is explicit)


def _synthetic_p0_world(n=8, g_dim=256, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, g_dim)).astype(np.float32)
    c = rng.normal(size=(n, len(CONTROL_FEATURE_KEYS))).astype(np.float32)
    s = rng.normal(size=(n, len(POLAR_STAT_KEYS))).astype(np.float32)
    s_valid = np.ones(n, dtype=bool)
    cache = {
        "g_features": g,
        "c_features": c,
        "s_features": s,
        "s_valid": s_valid,
        "class_id": rng.integers(0, 4, size=n).astype(np.int64),
    }
    rows = list(range(n))
    width = g_dim + len(CONTROL_FEATURE_KEYS) + len(POLAR_STAT_KEYS)
    column_valid = np.ones((n, width), dtype=bool)
    scaler = Standardizer.fit(
        np.concatenate([g, c, s], axis=1).astype(np.float64), column_valid)
    payload = {
        "state_dict": {
            "weight": torch.randn(4, width, generator=None) * 0.5,
            "bias": torch.randn(4) * 0.5,
        },
        "input_dim": width,
        "num_classes": 4,
        "variant": "P0",
        "fold": "foldX",
        "seed": 2026,
    }
    torch.manual_seed(seed)
    payload["state_dict"]["weight"] = torch.rand(4, width) * 2 - 1.0
    payload["state_dict"]["bias"] = torch.rand(4) * 2 - 1.0
    head = local_entry.p0_head_from_payload(payload)
    assert float(head.weight[:, g_dim:].abs().max()) > 0.0, (
        "the synthetic head must have nonzero disabled-slot weights")
    return cache, rows, scaler, head, g_dim


def _oracle_design(cache, rows, scaler, g_dim):
    """The correct P0 assembly, built HERE exactly like
    verify_p0_baseline does — never via the entry's helper."""
    g = cache["g_features"][rows]
    c = cache["c_features"][rows]
    s = cache["s_features"][rows]
    s_valid = cache["s_valid"][rows].astype(bool)
    width = g.shape[1] + c.shape[1] + s.shape[1]
    column_valid = np.ones((len(rows), width), dtype=bool)
    column_valid[:, g.shape[1] + c.shape[1]:] = s_valid[:, None]
    raw = np.concatenate([g, c, s], axis=1)
    standardized = scaler.transform(raw, column_valid)
    return assemble_design_matrix(
        standardized[:, :g_dim],
        standardized[:, g_dim:g_dim + len(CONTROL_FEATURE_KEYS)],
        standardized[:, g_dim + len(CONTROL_FEATURE_KEYS):],
        "P0").astype(np.float32)


# ---------------------------------------------------------------------------
# A. the design helper must zero the disabled slots and match the oracle


class TestP0DesignIdentity:
    def test_design_zeroes_disabled_slots_and_matches_oracle(self):
        cache, rows, scaler, head, g_dim = _synthetic_p0_world()
        # the synthetic cache really has nonzero C/S content
        assert float(np.abs(cache["c_features"]).max()) > 0.0
        assert float(np.abs(cache["s_features"]).max()) > 0.0
        design = local_entry._p0_design_for_rows(rows, cache, scaler)
        assert design.shape == (len(rows), 270)
        assert design.dtype == np.float32
        assert np.isfinite(design).all()
        disabled_max = float(np.abs(design[:, g_dim:]).max())
        assert disabled_max == 0.0, (
            "the P0 design must zero the C/S slots exactly")
        oracle_design = _oracle_design(cache, rows, scaler, g_dim)
        assert np.array_equal(design, oracle_design)
        oracle_logits = local_entry.p0_logits_of(head, oracle_design)
        runtime_logits = local_entry.p0_logits_of(head, design)
        assert float(np.max(np.abs(runtime_logits - oracle_logits))) == 0.0
        assert [int(v) for v in runtime_logits.argmax(axis=1)] == [
            int(v) for v in oracle_logits.argmax(axis=1)]

    def test_perturbing_cs_cannot_change_the_p0_base(self):
        cache, rows, scaler, head, g_dim = _synthetic_p0_world()
        design = local_entry._p0_design_for_rows(rows, cache, scaler)
        logits = local_entry.p0_logits_of(head, design)
        cache2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in cache.items()}
        cache2["c_features"] += 100.0
        cache2["s_features"] *= -3.0
        design2 = local_entry._p0_design_for_rows(rows, cache2, scaler)
        assert np.array_equal(design2, design)
        assert np.array_equal(
            local_entry.p0_logits_of(head, design2), logits)

    def test_runtime_logits_match_the_logit_atol_contract(self):
        cache, rows, scaler, head, g_dim = _synthetic_p0_world()
        design = local_entry._p0_design_for_rows(rows, cache, scaler)
        runtime = local_entry.p0_logits_of(head, design)
        oracle = local_entry.p0_logits_of(
            head, _oracle_design(cache, rows, scaler, g_dim))
        # the historical-oracle comparison: absolute tolerance 1e-4,
        # rtol 0 (never np.allclose's default relative tolerance)
        assert float(np.max(np.abs(runtime - oracle))) <= LOGIT_ATOL


# ---------------------------------------------------------------------------
# C. epoch0 and the invalid-S fallback against the verified runtime P0


class TestEpoch0AndInvalidFallback:
    def _runtime_p0(self, n=6):
        cache, rows, scaler, head, g_dim = _synthetic_p0_world(n=n, seed=3)
        design = local_entry._p0_design_for_rows(rows, cache, scaler)
        p0 = local_entry.p0_logits_of(head, design)
        oracle = local_entry.p0_logits_of(
            head, _oracle_design(cache, rows, scaler, g_dim))
        assert float(np.max(np.abs(p0 - oracle))) <= LOGIT_ATOL
        return p0, oracle

    def test_epoch0_output_is_exactly_the_runtime_p0(self):
        p0, _oracle = self._runtime_p0()
        head = local_entry.build_residual_head(seed=2026,
                                               input_dim=30)  # W = 0
        x = torch.randn(len(p0), 30)
        valid = torch.ones(len(p0))
        with torch.inference_mode():
            final = residual_logits(head, x, valid, torch.from_numpy(p0))
        assert final.detach().numpy().dtype == np.float32
        assert np.array_equal(final.numpy(), p0)

    def test_invalid_rows_fall_back_exactly_with_a_nonzero_head(self):
        p0, _oracle = self._runtime_p0()
        head = local_entry.build_residual_head(seed=2026, input_dim=30)
        with torch.no_grad():
            head.linear.weight.fill_(0.7)  # a deliberately nonzero head
        x = torch.randn(len(p0), 30)
        valid = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        with torch.inference_mode():
            final = residual_logits(head, x, valid, torch.from_numpy(p0))
            delta = compute_delta(head, x, valid)
        invalid = valid == 0.0
        assert np.array_equal(delta.numpy()[invalid],
                              np.zeros((int(invalid.sum()), 4)))
        assert np.array_equal(final.numpy()[invalid], p0[invalid])
        assert not np.array_equal(final.numpy()[~invalid], p0[~invalid])


# ---------------------------------------------------------------------------
# B. the real entry refuses a wrong P0 base BEFORE any training


class _RecordingProbe:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append("called")
        raise AssertionError("training must not start on a wrong P0 base")


class TestEntryRefusesWrongP0Base:
    @staticmethod
    def _legacy_unmasked_design(rows, cache, scaler):
        """The pre-fix implementation: standardized G+C+S, no masking."""
        g = cache["g_features"][rows]
        c = cache["c_features"][rows]
        s = cache["s_features"][rows]
        s_valid = cache["s_valid"][rows].astype(bool)
        width = g.shape[1] + c.shape[1] + s.shape[1]
        column_valid = np.ones((len(rows), width), dtype=bool)
        column_valid[:, g.shape[1] + c.shape[1]:] = s_valid[:, None]
        raw = np.concatenate([g, c, s], axis=1)
        return scaler.transform(raw, column_valid).astype(np.float32)

    def test_dry_run_refuses_the_unmasked_base(
            self, local_world, module_scratch, monkeypatch, capsys):
        local_cache, config_path, output = _prepare_fit(
            module_scratch / "p0fault_dry", local_world)
        monkeypatch.setattr(local_entry, "_p0_design_for_rows",
                            self._legacy_unmasked_design)
        code = local_entry.main(
            _main_args(local_world, config_path, output, mode="--dry-run"))
        assert code != 0
        captured = capsys.readouterr()
        assert "P0" in captured.err
        assert ("identity" in captured.err or "MISMATCH" in captured.err
                or "disabled" in captured.err or "slot" in captured.err)
        assert not (output / "models").exists()
        if (output / "report.json").exists():
            report = json.loads(
                (output / "report.json").read_text(encoding="utf-8"))
            assert report.get("training_status") != "COMPLETED"

    def test_fit_refuses_before_training(
            self, local_world, module_scratch, monkeypatch, capsys):
        local_cache, config_path, output = _prepare_fit(
            module_scratch / "p0fault_fit", local_world)
        probe = _RecordingProbe()
        monkeypatch.setattr(local_entry, "_p0_design_for_rows",
                            self._legacy_unmasked_design)
        monkeypatch.setattr(local_entry, "train_residual_head", probe)
        code = local_entry.main(
            _main_args(local_world, config_path, output, mode="--fit"))
        assert code != 0
        assert probe.calls == []
        captured = capsys.readouterr()
        assert "P0" in captured.err
        assert not (output / "models").exists()
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert summary.get("training_status") != "COMPLETED"

    def test_correct_base_passes_dry_run(
            self, local_world, module_scratch, capsys):
        local_cache, config_path, output = _prepare_fit(
            module_scratch / "p0ok_dry", local_world)
        code = local_entry.main(
            _main_args(local_world, config_path, output, mode="--dry-run"))
        assert code == 0
        admission = json.loads(
            (output / "admission.json").read_text(encoding="utf-8"))
        identity = admission["p0_identity"]
        for fold in ("fold0", "fold1", "fold2"):
            record = identity[fold]
            assert record["prediction_mismatches"] == 0
            assert record["max_logit_error"] <= LOGIT_ATOL
            for split in ("train", "inner_val", "outer_eval"):
                assert record["disabled_slot_max_abs"][split] == 0.0
