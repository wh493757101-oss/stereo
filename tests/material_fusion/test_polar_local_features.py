"""Local polarization statistics (interior/band) under the frozen
measurement region and the frozen residual correction model.

Scheme A pre-registered in the experiment protocol: the ORIGINAL
measurement region (instance_mask & polar_valid) is partitioned — never
filtered — into the frozen interior (erode3x3) and contour band, each
carrying the existing eight S statistics plus explicit per-region C
control information. Everything runs on synthetic chains built by the
production generators; no real fit and no real-data training here.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

from core.material_mask_dataset import MaskFusionSampleArrays
from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    Standardizer,
    build_cache_binding,
    build_control_features,
    build_polar_stats,
    verify_cache_binding,
)
from core.material_residual import (
    RESIDUAL_INPUT_DIM,
    build_residual_head,
    compute_delta,
    train_residual_head,
)
from core.polar_local_features import (
    BAND_BLOCK,
    C_LOCAL_KEYS,
    INTERIOR_BLOCK,
    LOCAL_C_WIDTH,
    LOCAL_S_WIDTH,
    LOCAL_TOTAL_DIM,
    assemble_local_residual_inputs,
    build_local_cache_arrays,
    build_local_control_features,
    build_local_polar_stats,
    fit_local_scaler,
    verify_local_cache,
)
from core.polar_region_quality import (
    BAND_DEFINITION,
    INTERIOR_DEFINITION,
    interior_and_band_masks,
)

from .test_polar_region_quality_entry import (
    _argv,
    _write_config,
    qa_world,  # noqa: F401  (module-scoped synthetic world fixture)
)


def _source_arrays(qa_world) -> dict:
    """The verified global cache arrays of the synthetic world (the local
    builder copies G/s_valid/class_id from them byte-identically)."""
    return dict(np.load(qa_world["feature_dir"] / "feature_cache.npz"))


# ---------------------------------------------------------------------------
# Synthetic samples (direct arrays; nothing touches the disk)


def make_sample(gray, signed_q, abs_q, valid, instance_mask):
    gray = np.asarray(gray, dtype=np.uint8)
    return MaskFusionSampleArrays(
        gray=gray,
        signed_q=np.asarray(signed_q, dtype=np.float32),
        abs_q=np.asarray(abs_q, dtype=np.float32),
        valid=np.asarray(valid, dtype=bool),
        quality=np.zeros(4, dtype=np.float32),
        class_id=0,
        instance_mask=np.asarray(instance_mask, dtype=np.uint8),
    )


def ring_sample():
    """7x7 mask with a measurable interior and band."""
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 1:6] = 1
    signed = np.linspace(-0.3, 0.3, 49, dtype=np.float32).reshape(7, 7)
    return make_sample(np.full((7, 7), 120, np.uint8), signed,
                       np.abs(signed), np.ones((7, 7), bool), mask)


# ---------------------------------------------------------------------------
# Region partition: mutual exclusion and the frozen definitions


class TestRegionPartition:
    def test_interior_and_band_are_a_partition(self):
        rng = np.random.default_rng(0)
        for _ in range(6):
            mask = (rng.random((9, 8)) > 0.4).astype(np.uint8)
            if not mask.any():
                mask[2, 2] = 1
            interior, band = interior_and_band_masks(mask)
            assert not (interior & band).any()
            assert np.array_equal(interior | band, mask > 0)

    def test_definitions_are_the_frozen_strings(self):
        assert INTERIOR_DEFINITION == "erode3x3_instance_mask_v1"
        assert BAND_DEFINITION == "instance_mask_minus_interior_v1"

    def test_single_pixel_mask_has_empty_interior(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = 1
        interior, band = interior_and_band_masks(mask)
        assert not interior.any()
        assert np.array_equal(band, mask > 0)


# ---------------------------------------------------------------------------
# Local S statistics and C control information


class TestLocalStats:
    def test_s_local_is_two_frozen_blocks_of_eight(self):
        sample = ring_sample()
        values, blocks = build_local_polar_stats(sample)
        assert values.shape == (LOCAL_S_WIDTH,)
        assert blocks["interior_valid"] and blocks["band_valid"]
        interior, band = interior_and_band_masks(sample.instance_mask)
        expected_in, _ = build_polar_stats(
            signed_q=sample.signed_q, abs_q=sample.abs_q,
            valid=sample.valid, instance_mask=interior.astype(np.uint8))
        expected_bd, _ = build_polar_stats(
            signed_q=sample.signed_q, abs_q=sample.abs_q,
            valid=sample.valid, instance_mask=band.astype(np.uint8))
        assert np.array_equal(values[:8], expected_in)
        assert np.array_equal(values[8:], expected_bd)

    def test_empty_region_is_explicit_nan_not_zero(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1  # single pixel: no interior
        sample = make_sample(np.full((5, 5), 100, np.uint8),
                             np.full((5, 5), 0.4, np.float32),
                             np.full((5, 5), 0.4, np.float32),
                             np.ones((5, 5), bool), mask)
        values, blocks = build_local_polar_stats(sample)
        assert not blocks["interior_valid"]
        assert np.isnan(values[:8]).all()
        assert blocks["band_valid"]
        assert np.isfinite(values[8:]).all()

    def test_c_local_keys_and_widths(self):
        assert len(C_LOCAL_KEYS) == LOCAL_C_WIDTH == 14
        assert LOCAL_S_WIDTH == 16
        assert LOCAL_TOTAL_DIM == 30
        assert list(C_LOCAL_KEYS[:6]) == list(CONTROL_FEATURE_KEYS)
        assert list(C_LOCAL_KEYS[6:]) == [
            "interior_coverage", "interior_empty_flag",
            "interior_gray_mean", "interior_gray_std",
            "band_coverage", "band_empty_flag",
            "band_gray_mean", "band_gray_std",
        ]
        assert INTERIOR_BLOCK == (0, 8) and BAND_BLOCK == (8, 16)

    def test_c_local_control_values(self):
        sample = ring_sample()
        c = build_local_control_features(sample)
        assert c.shape == (LOCAL_C_WIDTH,)
        assert np.isfinite(c).all()
        interior, band = interior_and_band_masks(sample.instance_mask)
        assert c[C_LOCAL_KEYS.index("interior_coverage")] == pytest.approx(
            float((interior & sample.valid).sum() / interior.sum()))
        assert c[C_LOCAL_KEYS.index("band_empty_flag")] == 0.0
        gray_in = sample.gray[(interior & sample.valid)].astype(np.float64)
        assert c[C_LOCAL_KEYS.index("interior_gray_mean")] == pytest.approx(
            float(gray_in.mean() / 255.0), abs=1e-6)

    def test_missing_region_control_is_explicit(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        sample = make_sample(np.full((5, 5), 100, np.uint8),
                             np.full((5, 5), 0.4, np.float32),
                             np.full((5, 5), 0.4, np.float32),
                             np.ones((5, 5), bool), mask)
        c = build_local_control_features(sample)
        assert c[C_LOCAL_KEYS.index("interior_coverage")] == 0.0
        assert c[C_LOCAL_KEYS.index("interior_empty_flag")] == 1.0
        assert c[C_LOCAL_KEYS.index("interior_gray_mean")] == 0.0
        assert c[C_LOCAL_KEYS.index("interior_gray_std")] == 0.0
        assert c[C_LOCAL_KEYS.index("band_empty_flag")] == 0.0


# ---------------------------------------------------------------------------
# Missing state: the partition cannot change sample-level validity


class TestMissingState:
    def test_s_valid_matches_the_original_definition(self):
        rng = np.random.default_rng(3)
        for _ in range(30):
            h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
            mask = (rng.random((h, w)) > 0.5).astype(np.uint8)
            if not mask.any():
                mask[0, 0] = 1
            valid = rng.random((h, w)) > 0.4
            signed = rng.uniform(-1, 1, size=(h, w)).astype(np.float32)
            sample = make_sample(np.full((h, w), 100, np.uint8), signed,
                                 np.abs(signed), valid, mask)
            values, blocks = build_local_polar_stats(sample)
            _, original_has = build_polar_stats(
                signed_q=sample.signed_q, abs_q=sample.abs_q,
                valid=sample.valid, instance_mask=sample.instance_mask)
            sample_valid = blocks["interior_valid"] or blocks["band_valid"]
            assert sample_valid == original_has
            if not sample_valid:
                assert np.isnan(values).all()
            elif not blocks["interior_valid"]:
                assert np.isnan(values[:8]).all()
                assert np.isfinite(values[8:]).all()
            elif not blocks["band_valid"]:
                assert np.isnan(values[8:]).all()
                assert np.isfinite(values[:8]).all()

    def test_all_invalid_sample_is_all_nan(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        sample = make_sample(np.full((5, 5), 100, np.uint8),
                             np.zeros((5, 5), np.float32),
                             np.zeros((5, 5), np.float32),
                             np.zeros((5, 5), bool), mask)
        values, blocks = build_local_polar_stats(sample)
        assert not blocks["interior_valid"] and not blocks["band_valid"]
        assert np.isnan(values).all()


# ---------------------------------------------------------------------------
# Residual assembly: RC never reads polarization content


class TestLocalAssembly:
    def _std(self, n=4, seed=5):
        rng = np.random.default_rng(seed)
        c = rng.normal(size=(n, LOCAL_C_WIDTH)).astype(np.float32)
        s = rng.normal(size=(n, LOCAL_S_WIDTH)).astype(np.float32)
        s_valid = np.array([True, True, False, True])
        s[2, :] = np.nan  # no valid S for row 2 (both blocks)
        return c, s, s_valid

    def test_rc_zeroes_the_polarization_slots(self):
        c, s, s_valid = self._std()
        first = assemble_local_residual_inputs(c, s, s_valid, "RC")
        second = assemble_local_residual_inputs(c, s * 100.0, s_valid, "RC")
        assert np.array_equal(first, second)
        assert not first[:, LOCAL_C_WIDTH:].any()
        assert first.shape == (4, LOCAL_TOTAL_DIM)

    def test_rcs_keeps_standardized_values_and_zeroes_missing(self):
        c, s, s_valid = self._std()
        matrix = assemble_local_residual_inputs(c, s, s_valid, "RCS")
        assert matrix[0, LOCAL_C_WIDTH:].any()
        assert not matrix[2, LOCAL_C_WIDTH:].any()  # missing -> exact zeros
        assert np.array_equal(matrix[:, :LOCAL_C_WIDTH], c)

    def test_unknown_variant_refused(self):
        c, s, s_valid = self._std()
        with pytest.raises(ValueError, match="variant"):
            assemble_local_residual_inputs(c, s, s_valid, "P2")

    def test_nan_in_valid_rows_refused(self):
        c, s, s_valid = self._std()
        s[0, 3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            assemble_local_residual_inputs(c, s, s_valid, "RCS")


# ---------------------------------------------------------------------------
# Standardization: inner_train only, per-cell region validity


class TestLocalScaler:
    def _cache(self, n=8):
        rng = np.random.default_rng(7)
        c = rng.normal(size=(n, LOCAL_C_WIDTH)).astype(np.float32)
        s = rng.normal(size=(n, LOCAL_S_WIDTH)).astype(np.float32)
        s_valid = np.ones(n, dtype=bool)
        interior_ok = np.array([True, True, False, True, True, False, True,
                                True])
        s[~interior_ok, :8] = np.nan
        s[3, 8:] = np.nan  # band missing too -> row fully invalid
        s_valid[3] = False
        return {"c_features": c, "s_features": s, "s_valid": s_valid}

    def test_fit_uses_inner_train_rows_and_valid_cells_only(self):
        cache = self._cache()
        rows = [0, 1, 2]
        scaler, info = fit_local_scaler(cache, rows)
        assert info["inner_train_rows"] == 3
        # interior block excluded for row 2 (NaN): the interior S mean must
        # come from rows 0 and 1 only
        expected = cache["s_features"][[0, 1], :8].mean(axis=0)
        assert np.allclose(scaler.mean[LOCAL_C_WIDTH:LOCAL_C_WIDTH + 8],
                           expected)
        # band block uses rows 0, 1, 2
        expected_band = cache["s_features"][[0, 1, 2], 8:].mean(axis=0)
        assert np.allclose(scaler.mean[LOCAL_C_WIDTH + 8:], expected_band)

    def test_eval_rows_cannot_change_the_scaler(self):
        cache = self._cache()
        rows = [0, 1, 2]
        scaler, _ = fit_local_scaler(cache, rows)
        cache2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in cache.items()}
        cache2["c_features"][5:] += 99.0
        cache2["s_features"][5:] += 99.0
        scaler2, _ = fit_local_scaler(cache2, rows)
        assert np.array_equal(scaler.mean, scaler2.mean)
        assert np.array_equal(scaler.usable, scaler2.usable)

    def test_nan_cells_transform_to_zero(self):
        cache = self._cache()
        rows = list(range(8))
        scaler, _ = fit_local_scaler(cache, rows)
        n = 8
        width = LOCAL_C_WIDTH + LOCAL_S_WIDTH
        raw = np.concatenate([cache["c_features"], cache["s_features"]],
                             axis=1).astype(np.float64)
        column_valid = ~np.isnan(raw)
        standardized = scaler.transform(raw, column_valid)
        # row 2 interior block: NaN cells -> exactly zero
        assert np.array_equal(
            standardized[2, LOCAL_C_WIDTH:LOCAL_C_WIDTH + 8],
            np.zeros(8, dtype=np.float32))
        assert standardized[2, LOCAL_C_WIDTH + 8:].any()


# ---------------------------------------------------------------------------
# Local head + training loop (parameterized, frozen defaults intact)


class TestLocalHead:
    def test_parameterized_head_is_zero_initialized(self):
        head = build_residual_head(seed=2026, input_dim=LOCAL_TOTAL_DIM)
        assert head.linear.weight.shape == (4, LOCAL_TOTAL_DIM)
        assert head.linear.bias is None
        assert torch.equal(head.linear.weight,
                           torch.zeros(4, LOCAL_TOTAL_DIM))
        again = build_residual_head(seed=2026, input_dim=LOCAL_TOTAL_DIM)
        assert torch.equal(head.linear.weight, again.linear.weight)

    def test_frozen_default_width_is_unchanged(self):
        head = build_residual_head(seed=2026)
        assert head.linear.weight.shape == (4, RESIDUAL_INPUT_DIM)

    def test_delta_bounded_and_gated_wide(self):
        head = build_residual_head(seed=2026, input_dim=LOCAL_TOTAL_DIM)
        with torch.no_grad():
            head.linear.weight.fill_(50.0)
        x = torch.randn(6, LOCAL_TOTAL_DIM)
        valid = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        delta = compute_delta(head, x, valid)
        assert delta.abs().max() <= 0.5 + 1e-6
        assert torch.equal(delta[1], torch.zeros(4))

    def test_tiny_training_run_accepts_the_local_width(self):
        rng = np.random.default_rng(11)
        n = 12
        x = rng.normal(size=(n, LOCAL_TOTAL_DIM)).astype(np.float32)
        y = rng.integers(0, 4, size=n)
        p0 = rng.normal(size=(n, 4)).astype(np.float32)
        valid = np.ones(n, dtype=np.float32)
        model, history, best_state, best = train_residual_head(
            x[:8], y[:8], p0[:8], valid[:8], x[8:], y[8:], p0[8:],
            valid[8:], seed=2026, lr=0.01, weight_decay=0.001,
            max_epochs=3, input_dim=LOCAL_TOTAL_DIM)
        assert history[0]["epoch"] == 0
        assert best["epoch"] >= 0
        assert best_state["linear.weight"].shape == (4, LOCAL_TOTAL_DIM)

    def test_frozen_width_still_enforced_by_default(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(6, LOCAL_TOTAL_DIM)).astype(np.float32)
        with pytest.raises(ValueError, match="14 columns"):
            train_residual_head(
                x[:3], np.zeros(3, dtype=np.int64),
                np.zeros((3, 4), np.float32), np.ones(3),
                x[3:], np.zeros(3, dtype=np.int64),
                np.zeros((3, 4), np.float32), np.ones(3),
                seed=2026, lr=0.01, weight_decay=0.001, max_epochs=1)


# ---------------------------------------------------------------------------
# Local cache: identity, integrity, G untouched, old cache protected


class TestLocalCache:
    def _build(self, qa_world, scratch_dir, name):
        return build_local_cache_arrays(
            source_arrays=_source_arrays(qa_world),
            samples=qa_world["cache_samples"],
            data_root=qa_world["v5"],
            protocol=qa_world["protocol"],
            output_dir=scratch_dir / name)

    def test_build_verify_and_old_cache_protection(self, qa_world,
                                                   scratch_dir):
        self._build(qa_world, scratch_dir, "local_features")
        output = scratch_dir / "local_features"
        verify_local_cache(output, qa_world["protocol"])
        verify_cache_binding(output, qa_world["protocol"])
        source = _source_arrays(qa_world)
        new_cache = dict(np.load(output / "feature_cache.npz"))
        assert np.array_equal(new_cache["g_features"],
                              source["g_features"])
        assert np.array_equal(new_cache["class_id"], source["class_id"])
        assert np.array_equal(new_cache["s_valid"], source["s_valid"])
        invalid = ~source["s_valid"]
        assert np.isnan(new_cache["s_features"][invalid]).all()
        # valid rows need at least one finite part (the synthetic V5 leaves
        # the band unmeasured everywhere, so NaN blocks in valid rows are
        # exactly the explicit-miss contract under test)
        assert np.isfinite(new_cache["s_features"][~invalid]).any(axis=1).all()
        manifest = json.loads(
            (output / "feature_manifest.json").read_text(encoding="utf-8"))
        old_rows = {r["sample_name"]: r
                    for r in qa_world["cache_samples"]}
        assert [r["sample_name"] for r in manifest["samples"]] == \
            [r["sample_name"] for r in qa_world["cache_samples"]]
        for row in manifest["samples"]:
            old = old_rows[row["sample_name"]]
            assert row["npz_path"] == old["npz_path"]
            assert row["fold_roles"] == old["fold_roles"]
            assert row["region_rule"] == "local_partition_interior_band_v1"
            assert "interior_pixels" in row and "band_pixels" in row

    def test_definition_declares_the_local_version(self, qa_world,
                                                   scratch_dir):
        self._build(qa_world, scratch_dir, "local_def")
        definition = json.loads(
            (scratch_dir / "local_def" / "feature_definition.json")
            .read_text(encoding="utf-8"))
        assert definition["feature_definition_version"] == \
            "existing_data_probe_v1_local_interior_band_v1"
        assert definition["region"]["interior_definition"] == \
            INTERIOR_DEFINITION
        assert definition["region"]["band_definition"] == BAND_DEFINITION
        assert definition["region"]["rule"] == \
            "local_partition_interior_band_v1"
        assert definition["slot_S"]["width"] == LOCAL_S_WIDTH
        assert definition["slot_C"]["width"] == LOCAL_C_WIDTH
        assert list(definition["slot_C"]["keys"]) == list(C_LOCAL_KEYS)

    def test_existing_output_refused_and_source_untouched(
            self, qa_world, scratch_dir):
        source_files = {
            name: (qa_world["feature_dir"] / name).read_bytes()
            for name in ("feature_cache.npz", "feature_manifest.json",
                         "feature_definition.json", "cache_binding.json")
        }
        self._build(qa_world, scratch_dir, "local_once")
        with pytest.raises(ValueError, match="already exists"):
            self._build(qa_world, scratch_dir, "local_once")
        for name, payload in source_files.items():
            assert (qa_world["feature_dir"] / name).read_bytes() == payload

    def test_row_integrity_region_sums(self, qa_world, scratch_dir):
        self._build(qa_world, scratch_dir, "local_rows")
        manifest = json.loads(
            (scratch_dir / "local_rows" / "feature_manifest.json")
            .read_text(encoding="utf-8"))
        for row in manifest["samples"]:
            assert row["interior_pixels"] + row["band_pixels"] == \
                row["mask_pixels"]
            assert row["interior_valid_pixels"] + row["band_valid_pixels"] \
                == row["region_pixels"]


# ---------------------------------------------------------------------------
# Entry (synthetic dry-run)


class TestLocalEntry:
    def test_dry_run_on_the_synthetic_world(self, qa_world, scratch_dir):
        from scripts import train_material_local_residual as local_entry

        local_cache = scratch_dir / "local_entry_features"
        build_local_cache_arrays(
            source_arrays=_source_arrays(qa_world),
            samples=qa_world["cache_samples"],
            data_root=qa_world["v5"],
            protocol=qa_world["protocol"],
            output_dir=local_cache)
        config = dict(qa_world["config"])
        config["experiment"] = "material_polar_local_interior_band_v1"
        config["local_features_source"] = {"path": str(local_cache)}
        config["variants"] = ["RC", "RCS"]
        config["old_global_reference"] = {"fit_summary": str(
            qa_world["fit_dir"] / "summary.json")}
        config_path = _write_config(scratch_dir / "configs" / "local.yaml",
                                    config)
        output = scratch_dir / "local_dry_run"
        code = local_entry.main(_argv(qa_world, output,
                                      config=config_path) + ["--dry-run"])
        assert code == 0
        report = json.loads(
            (output / "report.json").read_text(encoding="utf-8"))
        assert report["mode"] == "dry-run"
        assert report["local_cache"]["sample_count"] == \
            len(qa_world["cache_samples"])
        assert report["input_dim"] == LOCAL_TOTAL_DIM

    def test_existing_output_refused(self, qa_world, scratch_dir):
        from scripts import train_material_local_residual as local_entry

        output = scratch_dir / "taken"
        output.mkdir()
        config_path = _write_config(scratch_dir / "configs" / "c.yaml",
                                    qa_world["config"])
        assert local_entry.main(_argv(qa_world, output,
                                      config=config_path)
                                + ["--dry-run"]) == 1


# ---------------------------------------------------------------------------
# Manual PowerShell script


class TestManualPowerShellScript:
    def test_script_parses_and_defaults_to_dry_run(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / \
            "manual_material_local_residual.ps1"
        if not script.is_file():
            pytest.fail("manual_material_local_residual.ps1 is missing")
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
        # the ps1 must never be invoked THROUGH python (it is a PowerShell
        # entry; python only appears as the interpreter path in $py)
        assert not re.search(r"python\s+\S*\.ps1", text)


# ---------------------------------------------------------------------------
# Local cache ARRAY admission: Inf / partial-NaN / both-NaN valid rows are
# refused on array SEMANTICS (the binding is re-signed the production way
# on the synthetic cache so the refusal cannot come from a digest mismatch)


class TestLocalCacheArrayAdmission:
    def _built(self, qa_world, scratch_dir, name):
        output = scratch_dir / name
        build_local_cache_arrays(
            source_arrays=_source_arrays(qa_world),
            samples=qa_world["cache_samples"],
            data_root=qa_world["v5"],
            protocol=qa_world["protocol"],
            output_dir=output)
        return output

    def _tamper_and_rebind(self, output, protocol, mutate):
        cache = dict(np.load(output / "feature_cache.npz"))
        mutate(cache)
        np.savez_compressed(output / "feature_cache.npz", **cache)
        # re-sign the binding the production way: the refusal below must
        # come from the array semantics, never from a digest mismatch
        build_cache_binding(output, protocol)
        return dict(np.load(output / "feature_cache.npz"))

    def test_inf_in_valid_block_refused(self, qa_world, scratch_dir):
        output = self._built(qa_world, scratch_dir, "adm_inf")

        def mutate(cache):
            cache["s_features"][0, 0] = np.inf

        self._tamper_and_rebind(output, qa_world["protocol"], mutate)
        with pytest.raises(ValueError, match="entirely finite or entirely"):
            verify_local_cache(output, qa_world["protocol"])

    def test_partial_nan_block_refused(self, qa_world, scratch_dir):
        output = self._built(qa_world, scratch_dir, "adm_partial")

        def mutate(cache):
            cache["s_features"][0, :3] = np.nan  # rest of the block finite

        self._tamper_and_rebind(output, qa_world["protocol"], mutate)
        with pytest.raises(ValueError, match="entirely finite or entirely"):
            verify_local_cache(output, qa_world["protocol"])

    def test_valid_row_with_both_blocks_nan_refused(
            self, qa_world, scratch_dir):
        output = self._built(qa_world, scratch_dir, "adm_bothnan")

        def mutate(cache):
            cache["s_features"][0, :] = np.nan  # flags left stale on purpose

        self._tamper_and_rebind(output, qa_world["protocol"], mutate)
        with pytest.raises(ValueError,
                           match="at least one entirely finite part"):
            verify_local_cache(output, qa_world["protocol"])

    def test_invalid_row_with_finite_block_refused(
            self, qa_world, scratch_dir):
        output = self._built(qa_world, scratch_dir, "adm_invalidthat")

        def mutate(cache):
            cache["s_valid"][0] = False  # row 0 keeps finite values

        self._tamper_and_rebind(output, qa_world["protocol"], mutate)
        with pytest.raises(ValueError, match="all-NaN across BOTH parts"):
            verify_local_cache(output, qa_world["protocol"])

    def test_fully_missing_row_with_consistent_flags_passes(
            self, qa_world, scratch_dir):
        output = self._built(qa_world, scratch_dir, "adm_legal_missing")

        def mutate(cache):
            cache["s_features"][0, :] = np.nan
            cache["s_valid"][0] = False
            # make the C controls consistent the way the builder would
            cache["c_features"][
                0, C_LOCAL_KEYS.index("s_region_nonempty_flag")] = 0.0
            cache["c_features"][
                0, C_LOCAL_KEYS.index("interior_empty_flag")] = 1.0
            cache["c_features"][
                0, C_LOCAL_KEYS.index("band_empty_flag")] = 1.0
            cache["c_features"][
                0, C_LOCAL_KEYS.index("interior_coverage")] = 0.0
            cache["c_features"][
                0, C_LOCAL_KEYS.index("band_coverage")] = 0.0

        self._tamper_and_rebind(output, qa_world["protocol"], mutate)
        report = verify_local_cache(output, qa_world["protocol"])
        assert report["sample_count"] == len(qa_world["cache_samples"])

    def test_stale_coverage_flag_refused(self, qa_world, scratch_dir):
        output = self._built(qa_world, scratch_dir, "adm_stalecov")

        def mutate(cache):
            # the band block is all-NaN but its coverage claims measured
            # pixels and its flag claims non-empty
            cache["c_features"][
                0, C_LOCAL_KEYS.index("band_coverage")] = 0.5

        self._tamper_and_rebind(output, qa_world["protocol"], mutate)
        with pytest.raises(ValueError, match="band_coverage"):
            verify_local_cache(output, qa_world["protocol"])
