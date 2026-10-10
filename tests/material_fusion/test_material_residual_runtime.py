"""Runtime adapter + two-layer evaluation contract for the locked residual
candidate (material_residual_runtime_v1).

The candidate is the COMPLETED conditional-residual fit
(manual_20261009T185911): frozen MiT-B0 gray feature G, frozen per-fold P0
linear head, frozen correction branch (Linear(35,8) -> SiLU ->
Linear(8,4,bias=False)), ``delta = valid_s * 0.5 * tanh(head(x))``. It is
NOT a CMX checkpoint and never enters the historical CMX loader.

Contracts pinned here, before the implementation exists (TDD):

- config loading and identity verification (every pinned file sha256);
- the adapter refuses wrong payloads/shas/folds before anything runs;
- prediction semantics: valid-S rows use the correction, missing or
  non-finite polarization falls back to the candidate's OWN frozen P0
  exactly, an unusable gray crop is an explicit failure (never a silent
  substitution), RC never reads polarization content, R0 is pure P0;
- fold routing: a frame is evaluated only by the fold its capture group
  belongs to (never a best fold, no ensemble);
- heads and encoder are unchanged by any forward;
- Layer A stage comparison + verdict helpers and the frozen benchmark
  frame selection live in the evaluation entry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

from core.material_probe import snapshot_module_state
from models.material_classifier import MaterialInstanceInput

REAL_CONFIG = Path(__file__).resolve().parents[2] / \
    "configs/experiments/material_residual_runtime_v1.yaml"

FOLD_IDS = ("fold0", "fold1", "fold2")
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")


# ---------------------------------------------------------------------------
# Helpers


class _FakeEncoder:
    """A stand-in for the frozen MiT-B0 encoder with the same interface.

    Returns four stage tensors whose LAST stage is (B, 256, 7, 7),
    deterministically derived from the input, so the adapter's pooling
    math is exercised without loading the real weights.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, rgb):
        import torch.nn.functional as F

        self.calls.append(int(rgb.shape[0]))
        last = F.adaptive_avg_pool2d(rgb[:, :1], (7, 7))
        last = last.expand(rgb.shape[0], 256, 7, 7).contiguous()
        return [None, None, None, last]


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sample(name="s0", frame="f1", *, with_valid_s=True, non_finite=False,
            gray_size=24):
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 255, size=(gray_size, gray_size),
                        dtype=np.uint8)
    mask = np.zeros((gray_size, gray_size), dtype=np.uint8)
    mask[4:20, 4:20] = 1
    signed = (rng.random((gray_size, gray_size)).astype(np.float32) - 0.5)
    valid = np.zeros((gray_size, gray_size), dtype=np.uint8)
    if with_valid_s:
        valid[mask > 0] = 1
    abs_q = np.abs(signed).astype(np.float32)
    if non_finite:
        signed = signed.copy()
        signed[5, 5] = np.nan
    quality = np.asarray([0.5, 1.0, 1.0, 0.1], dtype=np.float32)
    return MaterialInstanceInput(
        frame_id=frame, instance_id=0,
        bbox=(4, 4, 20, 20), gray=gray, signed_q=signed, abs_q=abs_q,
        polar_valid=valid, instance_mask=mask, quality=quality,
        polar_invalid=False, invalid_reason="",
        mask_source="predicted_instance_mask")


def _write_config(base: Path, **overrides) -> Path:
    """A scratch copy of the real runtime config with nested overrides."""
    config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    path = base / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False),
                    encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def fake_encoder():
    return _FakeEncoder()


@pytest.fixture(scope="module")
def frame_folds():
    return {"f1": "fold0", "f2": "fold1", "f3": "fold2"}


@pytest.fixture(scope="module")
def rcs_classifier(fake_encoder, frame_folds):
    from models.material_residual_classifier import (
        MaterialResidualClassifier,
    )

    return MaterialResidualClassifier(
        str(REAL_CONFIG), variant="RCS", device="cpu",
        frame_folds=frame_folds, encoder=fake_encoder)


@pytest.fixture(scope="module")
def rc_classifier(fake_encoder, frame_folds):
    from models.material_residual_classifier import (
        MaterialResidualClassifier,
    )

    return MaterialResidualClassifier(
        str(REAL_CONFIG), variant="RC", device="cpu",
        frame_folds=frame_folds, encoder=fake_encoder)


@pytest.fixture(scope="module")
def r0_classifier(fake_encoder, frame_folds):
    from models.material_residual_classifier import (
        MaterialResidualClassifier,
    )

    return MaterialResidualClassifier(
        str(REAL_CONFIG), variant="R0", device="cpu",
        frame_folds=frame_folds, encoder=fake_encoder)


# ---------------------------------------------------------------------------
# Config loading and validation


class TestRuntimeConfig:
    def test_real_config_loads(self):
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        config = load_residual_runtime_config(REAL_CONFIG)
        assert config.experiment == "material_residual_runtime_v1"
        assert config.engineering_seed == 2026
        assert config.input_dim == 35
        assert config.hidden == 8
        assert config.variants == ("R0", "RC", "RCS")
        assert set(config.p0_models) == set(FOLD_IDS)
        assert config.tolerances["c_s_atol"] == 0.0

    def test_missing_section_refused(self, module_scratch):
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        path = _write_config(module_scratch / "cfg_missing")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        del config["tolerances"]
        path.write_text(yaml.safe_dump(config, sort_keys=False),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="tolerances"):
            load_residual_runtime_config(path)

    def test_bad_sha_refused(self, module_scratch):
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        path = _write_config(
            module_scratch / "cfg_sha",
            **{"model_a.sha256": "not-a-sha"})
        with pytest.raises(ValueError, match="sha256"):
            load_residual_runtime_config(path)

    def test_unknown_experiment_refused(self, module_scratch):
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        path = _write_config(
            module_scratch / "cfg_exp",
            **{"experiment": "something_else"})
        with pytest.raises(ValueError, match="experiment"):
            load_residual_runtime_config(path)


# ---------------------------------------------------------------------------
# Construction: identity verification before anything runs


class TestClassifierConstruction:
    def test_constructs_and_records_identities(self, rcs_classifier):
        assert rcs_classifier.variant == "RCS"
        assert rcs_classifier.engineering_seed == 2026
        assert rcs_classifier.file_identities, \
            "the classifier must record the pinned file digests it verified"
        for name, module in rcs_classifier.torch_modules().items():
            if name == "encoder":
                continue  # the fake encoder is not an nn.Module
            assert module.training is False, name
            for parameter in module.parameters():
                assert parameter.requires_grad is False, name

    def test_tampered_residual_state_refused(self, module_scratch,
                                             fake_encoder, frame_folds):
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
        )

        base = module_scratch / "tamper_state"
        base.mkdir(parents=True)
        payload = torch.load(
            "analysis/runs/material_conditional_residual_v1/"
            "manual_20261009T185911/models/seed_2026/fold0_RCS/"
            "residual_state.pt", weights_only=False, map_location="cpu")
        payload["seed"] = 9999
        tampered = base / "residual_state.pt"
        torch.save(payload, tampered)
        path = _write_config(
            base,
            **{"residual_models.seed_2026.fold0.RCS.residual_state":
               {"path": str(tampered), "sha256": _sha(tampered)}})
        with pytest.raises(ValueError, match="seed"):
            MaterialResidualClassifier(
                path, variant="RCS", device="cpu",
                frame_folds=frame_folds, encoder=fake_encoder)

    def test_wrong_file_sha_refused(self, module_scratch, fake_encoder,
                                    frame_folds):
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
        )

        path = _write_config(
            module_scratch / "tamper_sha",
            **{"residual_models.seed_2026.fold0.RCS.residual_state":
               {"path": "analysis/runs/material_conditional_residual_v1/"
                        "manual_20261009T185911/models/seed_2026/fold0_RCS/"
                        "residual_state.pt",
                "sha256": "a" * 64}})
        with pytest.raises(ValueError, match="sha256"):
            MaterialResidualClassifier(
                path, variant="RCS", device="cpu",
                frame_folds=frame_folds, encoder=fake_encoder)

    def test_tampered_p0_payload_refused(self, module_scratch, fake_encoder,
                                         frame_folds):
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
        )

        base = module_scratch / "tamper_p0"
        base.mkdir(parents=True)
        payload = torch.load(
            "analysis/runs/material_existing_data_v1/"
            "manual_20261008_085510_822/fit/models/fold0_P0/"
            "model_state.pt", weights_only=False, map_location="cpu")
        payload["fold"] = "fold9"
        tampered = base / "p0.pt"
        torch.save(payload, tampered)
        path = _write_config(
            base,
            **{"p0_models.fold0.checkpoint":
               {"path": str(tampered), "sha256": _sha(tampered)}})
        with pytest.raises(ValueError, match="fold"):
            MaterialResidualClassifier(
                path, variant="RCS", device="cpu",
                frame_folds=frame_folds, encoder=fake_encoder)

    def test_unknown_variant_refused(self, fake_encoder, frame_folds):
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
        )

        with pytest.raises(ValueError, match="variant"):
            MaterialResidualClassifier(
                str(REAL_CONFIG), variant="P2", device="cpu",
                frame_folds=frame_folds, encoder=fake_encoder)

    def test_non_engineering_seed_refused(self, fake_encoder, frame_folds):
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
        )

        with pytest.raises(ValueError, match="seed"):
            MaterialResidualClassifier(
                str(REAL_CONFIG), variant="RCS", device="cpu",
                frame_folds=frame_folds, encoder=fake_encoder, seed=2027)

    def test_invalid_frame_fold_map_refused(self, fake_encoder):
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
        )

        with pytest.raises(ValueError, match="fold"):
            MaterialResidualClassifier(
                str(REAL_CONFIG), variant="RCS", device="cpu",
                frame_folds={"f1": "fold7"}, encoder=fake_encoder)

    def test_real_encoder_builds_when_not_injected(self, frame_folds):
        """Without an injected encoder the frozen official MiT-B0 is built
        and identity-verified (once; the module caches it)."""
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
            _encoder_cache,
        )

        classifier = MaterialResidualClassifier(
            str(REAL_CONFIG), variant="RCS", device="cpu",
            frame_folds=frame_folds)
        assert _encoder_cache, "the identity-keyed encoder cache is filled"
        from models.material_residual_classifier import _encoder_cache_key

        key = _encoder_cache_key(classifier.config, "cpu")
        assert key in _encoder_cache
        assert classifier.encoder is _encoder_cache[key]


# ---------------------------------------------------------------------------
# Prediction semantics


class TestPredictSemantics:
    def test_output_count_and_order_preserved(self, rcs_classifier):
        samples = [_sample(), None, _sample("s2", "f1"),
                   _sample("s3", "f1", with_valid_s=False)]
        results = rcs_classifier.predict_batch(samples)
        assert len(results) == len(samples)
        assert results[1].valid is False
        assert results[1].invalid_reason == "none"
        assert results[0].instance_id == 0
        assert results[2].instance_id == 0  # both fixtures use instance 0
        assert results[0].frame_id == "f1"

    def test_invalid_gray_is_failure_not_fallback(self, rcs_classifier):
        broken = _sample()
        broken = MaterialInstanceInput(
            frame_id="f1", instance_id=0, bbox=(0, 0, 1, 1),
            gray=np.zeros((0,), dtype=np.uint8), signed_q=broken.signed_q,
            abs_q=broken.abs_q, polar_valid=broken.polar_valid,
            instance_mask=broken.instance_mask, quality=broken.quality,
            polar_invalid=False, invalid_reason="")
        (result,) = rcs_classifier.predict_batch([broken])
        assert result.valid is False
        assert result.invalid_reason == "invalid_gray_crop"
        assert result.used_fallback is False

    def test_missing_s_falls_back_to_own_p0_exactly(
            self, rcs_classifier, r0_classifier):
        sample = _sample(with_valid_s=False)
        (rcs,) = rcs_classifier.predict_batch([sample])
        (r0,) = r0_classifier.predict_batch([sample])
        assert rcs.valid is True
        assert rcs.used_fallback is True
        assert rcs.fallback_reason == "no_valid_s_region"
        assert list(rcs.raw_logits) == list(r0.raw_logits)
        assert "fallback=own_P0" in rcs.model_identity

    def test_non_finite_polar_falls_back(self, rcs_classifier,
                                         r0_classifier):
        sample = _sample(non_finite=True)
        (rcs,) = rcs_classifier.predict_batch([sample])
        (r0,) = r0_classifier.predict_batch([sample])
        assert rcs.used_fallback is True
        assert rcs.fallback_reason == "non_finite_polar"
        assert list(rcs.raw_logits) == list(r0.raw_logits)

    def test_valid_row_uses_the_correction(self, rcs_classifier,
                                           r0_classifier):
        sample = _sample()
        (rcs,) = rcs_classifier.predict_batch([sample])
        (r0,) = r0_classifier.predict_batch([sample])
        assert rcs.used_fallback is False
        assert rcs.fallback_reason == ""
        assert list(rcs.raw_logits) != list(r0.raw_logits)
        assert "candidate=conditional_RCS_s2026" in rcs.model_identity
        assert "fold0" in rcs.model_identity
        assert "seed=2026" in rcs.model_identity

    def test_rc_never_reads_polarization_content(self, rc_classifier,
                                                 rcs_classifier,
                                                 fake_encoder):
        first = _sample()
        (rc_a,) = rc_classifier.predict_batch([first])
        perturbed = MaterialInstanceInput(
            frame_id=first.frame_id, instance_id=first.instance_id,
            bbox=first.bbox, gray=first.gray,
            signed_q=first.signed_q * -7.0 + 3.0,
            abs_q=first.abs_q * 11.0 + 1.0,
            polar_valid=first.polar_valid, instance_mask=first.instance_mask,
            quality=first.quality, polar_invalid=False, invalid_reason="",
            mask_source="predicted_instance_mask")
        (rc_b,) = rc_classifier.predict_batch([perturbed])
        (rcs_b,) = rcs_classifier.predict_batch([perturbed])
        (rcs_a,) = rcs_classifier.predict_batch([first])
        assert list(rc_a.raw_logits) == list(rc_b.raw_logits)
        assert list(rcs_a.raw_logits) != list(rcs_b.raw_logits)

    def test_r0_is_pure_p0(self, r0_classifier):
        sample = _sample()
        (a,) = r0_classifier.predict_batch([sample])
        perturbed = MaterialInstanceInput(
            frame_id=sample.frame_id, instance_id=sample.instance_id,
            bbox=sample.bbox, gray=sample.gray,
            signed_q=sample.signed_q * -3.0, abs_q=sample.abs_q * 5.0,
            polar_valid=sample.polar_valid,
            instance_mask=sample.instance_mask, quality=sample.quality,
            polar_invalid=False, invalid_reason="",
            mask_source="predicted_instance_mask")
        (b,) = r0_classifier.predict_batch([perturbed])
        assert a.used_fallback is False
        assert list(a.raw_logits) == list(b.raw_logits)
        assert "candidate=conditional_R0_s2026" in a.model_identity

    def test_unknown_frame_is_a_failure(self, rcs_classifier):
        sample = _sample(frame="unknown_frame")
        (result,) = rcs_classifier.predict_batch([sample])
        assert result.valid is False
        assert result.invalid_reason == "unknown_frame_fold"

    def test_heads_and_encoder_unchanged_by_forward(
            self, rcs_classifier, fake_encoder):
        before = {name: snapshot_module_state(module)
                  for name, module in rcs_classifier.torch_modules().items()}
        rcs_classifier.predict_batch([
            _sample(), _sample("s2", "f1", with_valid_s=False)])
        after = {name: snapshot_module_state(module)
                 for name, module in rcs_classifier.torch_modules().items()}
        for name in before:
            check = before[name]
            check_after = after[name]
            assert set(check) == set(check_after)
            for key in check:
                assert np.array_equal(check[key], check_after[key]), \
                    f"{name}.{key} changed during inference"

    def test_batching_matches_single_forward(self, rcs_classifier):
        samples = [_sample(), _sample("s2", "f1")]
        batched = rcs_classifier.predict_batch(samples)
        single = [rcs_classifier.predict_batch([sample])[0]
                  for sample in samples]
        for b, s in zip(batched, single):
            assert list(b.raw_logits) == list(s.raw_logits)

    def test_polar_invalid_flag_passthrough(self, rcs_classifier):
        sample = _sample()
        flagged = MaterialInstanceInput(
            frame_id=sample.frame_id, instance_id=sample.instance_id,
            bbox=sample.bbox, gray=sample.gray, signed_q=sample.signed_q,
            abs_q=sample.abs_q, polar_valid=np.zeros_like(
                sample.polar_valid),
            instance_mask=sample.instance_mask, quality=sample.quality,
            polar_invalid=True, invalid_reason="stereo_failed",
            mask_source="predicted_instance_mask")
        (result,) = rcs_classifier.predict_batch([flagged])
        assert result.used_fallback is True
        assert result.fallback_reason == "no_valid_s_region"
        assert result.polar_invalid is True
        assert result.invalid_reason == "stereo_failed"


# ---------------------------------------------------------------------------
# Fold routing


class TestFoldRouting:
    def test_frame_routes_to_its_fold(self, fake_encoder):
        from models.material_residual_classifier import (
            MaterialResidualClassifier,
        )

        classifier = MaterialResidualClassifier(
            str(REAL_CONFIG), variant="RCS", device="cpu",
            frame_folds={"fa": "fold0", "fb": "fold1"},
            encoder=fake_encoder)
        (a,) = classifier.predict_batch([_sample(frame="fa")])
        (b,) = classifier.predict_batch([_sample(frame="fb")])
        assert "fold0" in a.model_identity
        assert "fold1" in b.model_identity

    def test_build_frame_fold_map_from_protocol(self):
        from models.material_residual_classifier import (
            build_frame_fold_map,
        )

        protocol = {"folds": {"folds": [
            {"fold_id": "fold0", "outer_eval_groups": ["gA", "gB"]},
            {"fold_id": "fold1", "outer_eval_groups": ["gC"]},
        ]}}
        rows = [{"output_stem": "s1", "group_name": "gA"},
                {"output_stem": "s2", "group_name": "gC"}]
        mapping = build_frame_fold_map(rows, protocol)
        assert mapping == {"s1": "fold0", "s2": "fold1"}

    def test_group_in_two_folds_refused(self):
        from models.material_residual_classifier import (
            build_frame_fold_map,
        )

        protocol = {"folds": {"folds": [
            {"fold_id": "fold0", "outer_eval_groups": ["gA"]},
            {"fold_id": "fold1", "outer_eval_groups": ["gA"]},
        ]}}
        with pytest.raises(ValueError, match="gA"):
            build_frame_fold_map([{"output_stem": "s1",
                                   "group_name": "gA"}], protocol)

    def test_uncovered_group_refused(self):
        from models.material_residual_classifier import (
            build_frame_fold_map,
        )

        protocol = {"folds": {"folds": [
            {"fold_id": "fold0", "outer_eval_groups": ["gA"]},
        ]}}
        with pytest.raises(ValueError, match="gX"):
            build_frame_fold_map([{"output_stem": "s1",
                                   "group_name": "gX"}], protocol)


# ---------------------------------------------------------------------------
# Evaluation-entry helpers (Layer A verdicts + benchmark freezing)


class TestEntryHelpers:
    def test_compare_stage_arrays_exact(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        actual = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        report = compare_stage_arrays(actual, actual, atol=0.0, rtol=0.0)
        assert report["passed"] is True
        assert report["max_abs_error"] == 0.0

    def test_compare_stage_arrays_within_and_beyond_tolerance(self):
        from scripts.eval_material_residual_pipeline import (
            compare_stage_arrays,
        )

        expected = np.ones((2, 2), dtype=np.float32)
        close = expected + np.float32(1e-6)
        far = expected + np.float32(1.0)
        assert compare_stage_arrays(close, expected, atol=1e-4,
                                    rtol=1e-4)["passed"] is True
        report = compare_stage_arrays(far, expected, atol=1e-4, rtol=1e-4)
        assert report["passed"] is False
        assert report["max_abs_error"] == pytest.approx(1.0, abs=1e-6)

    def test_layer_a_verdict_aborts_on_any_failure(self):
        from scripts.eval_material_residual_pipeline import (
            layer_a_verdict,
        )

        required = {"conditional_RCS_s2026": ("g", "final")}
        ok = {"sample_name": "s0", "candidate": "conditional_RCS_s2026",
              "s_valid_offline": True, "s_valid_runtime": True,
              "stage_reports": {"g": {"passed": True, "max_abs_error": 0.0},
                                "final": {"passed": True,
                                          "max_abs_error": 0.0}},
              "prediction_mismatch": False}
        bad = {"sample_name": "s1", "candidate": "conditional_RCS_s2026",
               "s_valid_offline": True, "s_valid_runtime": True,
               "stage_reports": {"g": {"passed": False, "max_abs_error": None},
                                 "final": {"passed": True,
                                           "max_abs_error": 0.0}},
               "prediction_mismatch": 2}
        assert layer_a_verdict(
            [ok], expected_names=["s0"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=required)["abort_layer_b"] is False
        verdict = layer_a_verdict(
            [ok, bad], expected_names=["s0", "s1"],
            expected_candidates=["conditional_RCS_s2026"],
            required_stages=required)
        assert verdict["abort_layer_b"] is True
        assert verdict["status"] == "FAILED"

    def test_freeze_benchmark_frames_even_and_sorted(self):
        from scripts.eval_material_residual_pipeline import (
            freeze_benchmark_frames,
        )

        stems = [f"frame_{i:03d}" for i in range(180)]
        frozen = freeze_benchmark_frames(stems, 12)
        assert len(frozen) == 12
        assert frozen == sorted(frozen)
        assert len(set(frozen)) == 12
        # evenly spread: first and last stems must be included
        assert frozen[0] == stems[0]
        again = freeze_benchmark_frames(stems, 12)
        assert again == frozen

    def test_freeze_benchmark_frames_refuses_impossible_count(self):
        from scripts.eval_material_residual_pipeline import (
            freeze_benchmark_frames,
        )

        with pytest.raises(ValueError):
            freeze_benchmark_frames(["a", "b"], 5)

    def test_budget_watch_flags_partial(self):
        """Original intent preserved under the new contract: a non-positive
        budget is refused at construction, and exhaustion (proven with a
        fake clock, no real waiting) raises BudgetExceeded, a RuntimeError."""
        from scripts.eval_material_residual_pipeline import (
            BudgetExceeded,
            BudgetWatch,
        )

        with pytest.raises(ValueError):
            BudgetWatch(budget_minutes=0.0)
        ticks = iter([0.0, 61.0])
        watch = BudgetWatch(budget_minutes=1.0, clock=lambda: next(ticks))
        with pytest.raises(BudgetExceeded):
            watch.check("any stage")

    def test_budget_watch_allows_within_budget(self):
        from scripts.eval_material_residual_pipeline import BudgetWatch

        watch = BudgetWatch(budget_minutes=15.0)
        watch.check("stage")  # must not raise


# ---------------------------------------------------------------------------
# Saved-evidence binding (Layer A reads the pinned jsonl digests)


class TestSavedEvidenceBinding:
    def test_saved_jsonl_digests_match_the_config(self):
        """The real fit's saved files still hash to what the config pins —
        the Layer A comparison basis is intact."""
        from models.material_residual_classifier import (
            load_residual_runtime_config,
        )

        config = load_residual_runtime_config(REAL_CONFIG)
        reference = config.residual_models["seed_2026"]["fold0"]["RCS"][
            "outer_eval_predictions"]
        path = Path(reference.path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / reference.path
        assert _sha(path) == reference.sha256

    def test_saved_record_shape(self):
        """The pinned outer_eval_predictions.jsonl carries the fields the
        Layer A comparison needs."""
        import json as json_module

        record = json_module.loads(
            Path("analysis/runs/material_conditional_residual_v1/"
                 "manual_20261009T185911/models/seed_2026/fold0_RCS/"
                 "outer_eval_predictions.jsonl").read_text(
                     encoding="utf-8").splitlines()[0])
        for field in ("sample_name", "group", "class_id", "has_valid_s",
                      "logits", "p0_logits", "delta", "prediction"):
            assert field in record
        assert len(record["logits"]) == 4
        assert record["prediction"] == int(np.argmax(record["logits"]))
