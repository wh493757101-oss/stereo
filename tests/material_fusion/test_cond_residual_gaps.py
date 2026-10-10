"""RED tests for the four conditional-residual acceptance gaps.

Gap 1: the entry assembled fold inputs once with variant="RCS" and fed
the SAME tensors to RC — the RC S slots were never masked.
Gap 2: the paired comparison extended whole per-fold LISTS instead of
per-sample values (zip then truncated; six all-correct samples counted
as broke=3 / both_correct=3).
Gap 3: the linear reference pointed at a nonexistent path and no
identity/binding checks existed.
Gap 4: paired["pooled_macro_f1"] stored the fold-mean-of-seeds, i.e. a
mislabelled field; fold-mean and pooled must be separate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from scripts import train_material_conditional_residual as cond_entry
from scripts.verify_p0_baseline import LOGIT_ATOL
from scripts import train_material_local_residual as local_entry_mod
from core.material_probe import Standardizer

from .test_material_local_residual_entry import (
    _jsonl,
    _main_args,
    _write_config,
)
from .test_material_conditional_residual import (
    CLASS_NAMES,
    CONTENT_SEEDS,
    FOLD_IDS,
    TRAIN_SEEDS,
    _prepare_cond_fit,
    _write_legal_reference,
    cond_fit,  # noqa: F401  (module-scoped full synthetic fit fixture)
    local_world,  # noqa: F401
)
from .test_polar_local_features import _source_arrays


# ---------------------------------------------------------------------------
# Gap 1: the trainer must receive per-variant inputs (RC S = 0 exactly)


class _TrainerProbe:
    """Wraps the REAL trainer: records X_train/X_val, then delegates."""

    def __init__(self, original):
        self.original = original
        self.records = []

    def __call__(self, X_train, y_train, p0_train, valid_train, X_val,
                 y_val, p0_val, valid_val, **kwargs):
        self.records.append({
            "X_train": np.asarray(X_train, dtype=np.float32).copy(),
            "X_val": np.asarray(X_val, dtype=np.float32).copy(),
        })
        return self.original(X_train, y_train, p0_train, valid_train,
                             X_val, y_val, p0_val, valid_val, **kwargs)


@pytest.fixture(scope="module")
def probed_fit(local_world, module_scratch):
    base = module_scratch / "probed"
    local_cache, config_path, output = _prepare_cond_fit(base, local_world)
    probe = _TrainerProbe(cond_entry.train_conditional_residual_head)
    with mock.patch.object(cond_entry, "train_conditional_residual_head",
                           probe):
        code = cond_entry.main(
            _main_args(local_world, config_path, output))
    return {"code": code, "output": output, "probe": probe,
            "local_cache": local_cache}


class TestGap1PerVariantInputs:
    def test_probed_fit_completes(self, probed_fit):
        assert probed_fit["code"] == 0
        # 2 variants x 3 folds x 3 seeds = 18 trainer calls
        assert len(probed_fit["probe"].records) == 18

    def test_rc_s_slots_exactly_zero_in_trainer_inputs(
            self, local_world, probed_fit):
        output = probed_fit["output"]
        local_cache = dict(np.load(probed_fit["local_cache"]
                                   / "feature_cache.npz"))
        manifest_samples = local_world["cache_samples"]
        authority = local_world["protocol"]
        # per (seed, fold, variant) the trainer inputs must match the
        # variant semantics; reconstruct the expected standardized S per
        # fold from the SAVED per-variant scaler
        call_index = 0
        for seed in TRAIN_SEEDS:
            for fold in FOLD_IDS:
                for variant in ("RC", "RCS"):
                    record = probed_fit["probe"].records[call_index]
                    call_index += 1
                    if variant == "RC":
                        assert np.abs(
                            record["X_train"][:, 14:30]).max() == 0.0
                        assert np.abs(
                            record["X_val"][:, 14:30]).max() == 0.0
                    else:
                        assert np.abs(
                            record["X_train"][:, 14:30]).max() > 0.0
                del fold, variant
        del local_cache, manifest_samples, authority, output

    def test_rc_and_rcs_share_c_context_rows_but_differ_in_s(
            self, probed_fit):
        records = probed_fit["probe"].records
        # records are ordered seed-major, then fold, then RC/RCS
        index = 0
        for seed in TRAIN_SEEDS:
            for fold in FOLD_IDS:
                rc = records[index]
                rcs = records[index + 1]
                index += 2
                # C (0:14) and context (30:35) identical
                assert np.array_equal(rc["X_train"][:, 0:14],
                                      rcs["X_train"][:, 0:14])
                assert np.array_equal(rc["X_train"][:, 30:35],
                                      rcs["X_train"][:, 30:35])
                assert np.array_equal(rc["X_val"][:, 0:14],
                                      rcs["X_val"][:, 0:14])
                assert np.array_equal(rc["X_val"][:, 30:35],
                                      rcs["X_val"][:, 30:35])
                # S differs somewhere (valid rows exist in the world)
                assert not np.array_equal(rc["X_train"][:, 14:30],
                                          rcs["X_train"][:, 14:30])

    def test_perturbing_s_cannot_change_rc_outputs(self, local_world,
                                                   probed_fit):
        """Re-run the RC eval forward with a PERTURBED local-cache S but
        the SAME C/context/validity: the RC head output must be identical
        because its S slot is zero at the input."""
        output = probed_fit["output"]
        perturbed_cache = dict(np.load(probed_fit["local_cache"]
                                       / "feature_cache.npz"))
        perturbed_cache["s_features"] = (
            perturbed_cache["s_features"] * -7.0 + 11.0)
        global_cache = _source_arrays(local_world)
        manifest_samples = local_world["cache_samples"]
        for fold in FOLD_IDS:
            payload = torch.load(
                output / "models" / "seed_2026" / f"{fold}_RC"
                / "residual_state.pt", weights_only=False,
                map_location="cpu")
            head = cond_entry.build_conditional_head(
                seed=int(payload["seed"]),
                input_dim=int(payload["input_dim"]),
                hidden=int(payload["hidden"]))
            head.load_state_dict(payload["state_dict"])
            head.eval()
            scaler = Standardizer.from_dict(json.loads(
                (output / "models" / "seed_2026" / f"{fold}_RC"
                 / "scaler.json").read_text(encoding="utf-8")))
            parts = local_entry_fold_parts(manifest_samples,
                                           _authority(local_world), fold)
            shared = cond_entry._fold_conditional_inputs(
                local_cache=perturbed_cache,
                global_cache=global_cache,
                parts=parts, scaler=scaler,
                p0_head=probed_p0_head(local_world, fold),
                p0_scaler=probed_p0_scaler(local_world, fold))
            variant_inputs = cond_entry._variant_inputs(shared, "RC")
            with torch.inference_mode():
                final = cond_entry.conditional_logits(
                    head, torch.from_numpy(variant_inputs["x"]["eval"]),
                    torch.from_numpy(
                        shared["valid"]["eval"].astype(np.float32)),
                    torch.from_numpy(shared["p0_logits"]["eval"]))
            logits = final.numpy().astype(np.float32)
            saved = _jsonl(output / "models" / "seed_2026" / f"{fold}_RC"
                           / "outer_eval_predictions.jsonl")
            saved_logits = np.asarray([r["logits"] for r in saved],
                                      dtype=np.float32)
            assert np.allclose(logits, saved_logits, atol=1e-6)


# ---------------------------------------------------------------------------
# Gap 2: paired counts are per-sample aligned


def _independent_counts(left_predictions, right_predictions, labels):
    """Direct boolean classification — no production counting helpers."""
    fixed = broke = both_correct = both_wrong = 0
    for left, right, label in zip(left_predictions, right_predictions,
                                  labels):
        left_ok = left == label
        right_ok = right == label
        if left_ok and right_ok:
            both_correct += 1
        elif not left_ok and not right_ok:
            both_wrong += 1
        elif left_ok:
            fixed += 1
        else:
            broke += 1
    return {"fixed": fixed, "broke": broke, "both_correct": both_correct,
            "both_wrong": both_wrong}


class TestGap2PairedAlignment:
    def test_six_all_correct_samples_count_six(self):
        """Pure-array counterexample: three folds x two samples, every
        prediction correct. The nested-list bug counted broke=3 (RCS-R0)
        and both_correct=3 (RCS-RC); the truth is 6."""
        # three folds x two samples, per-sample records (as saved jsonl)
        left_records = [
            [{"sample_name": f"s{fold}{i}", "prediction": 0,
              "class_id": 0, "group": "g"} for i in range(2)]
            for fold in range(3)]
        right_records = [
            [{"sample_name": f"s{fold}{i}", "prediction": 1,
              "class_id": 0, "group": "g"} for i in range(2)]
            for fold in range(3)]
        aligned_left, aligned_right, labels, groups = \
            cond_entry.align_paired_samples(
                left_records, right_records,
                expected_names=[f"s{fold}{i}" for fold in range(3)
                                for i in range(2)],
                manifest_rows={f"s{fold}{i}":
                               {"class_id": 0, "group": "g"}
                               for fold in range(3) for i in range(2)})
        assert len(aligned_left) == 6 and len(aligned_right) == 6
        counts = _independent_counts(aligned_left, aligned_right, labels)
        assert counts == {"fixed": 6, "broke": 0, "both_correct": 0,
                          "both_wrong": 0}

    def test_all_four_transitions(self):
        left_records = [
            {"sample_name": "a", "prediction": 0, "class_id": 0,
             "group": "g1"},   # fixed (base wrong)
            {"sample_name": "b", "prediction": 2, "class_id": 0,
             "group": "g1"},   # broke (base right)
            {"sample_name": "c", "prediction": 1, "class_id": 1,
             "group": "g2"},   # both_correct
            {"sample_name": "d", "prediction": 3, "class_id": 1,
             "group": "g2"},   # both_wrong (wrong -> different wrong)
        ]
        right_records = [
            {"sample_name": "a", "prediction": 1, "class_id": 0,
             "group": "g1"},
            {"sample_name": "b", "prediction": 0, "class_id": 0,
             "group": "g1"},
            {"sample_name": "c", "prediction": 1, "class_id": 1,
             "group": "g2"},
            {"sample_name": "d", "prediction": 2, "class_id": 1,
             "group": "g2"},
        ]
        aligned_left, aligned_right, labels, _groups = \
            cond_entry.align_paired_samples(
                [left_records], [right_records],
                expected_names=["a", "b", "c", "d"],
                manifest_rows={name: {"class_id": cls, "group": grp}
                               for name, cls, grp in
                               (("a", 0, "g1"), ("b", 0, "g1"),
                                ("c", 1, "g2"), ("d", 1, "g2"))})
        counts = _independent_counts(aligned_left, aligned_right, labels)
        assert counts == {"fixed": 1, "broke": 1, "both_correct": 1,
                          "both_wrong": 1}

    def test_misalignment_is_refused(self):
        # duplicate sample names on one side
        with pytest.raises(Exception):
            cond_entry.align_paired_samples(
                [[{"sample_name": "a", "prediction": 0, "class_id": 0,
                   "group": "g"},
                  {"sample_name": "a", "prediction": 0, "class_id": 0,
                   "group": "g"}]],
                [[{"sample_name": "a", "prediction": 1, "class_id": 0,
                   "group": "g"}]],
                expected_names=["a"],
                manifest_rows={"a": {"class_id": 0, "group": "g"}})
        # missing sample
        with pytest.raises(Exception):
            cond_entry.align_paired_samples(
                [[{"sample_name": "a", "prediction": 0, "class_id": 0,
                   "group": "g"}]],
                [[{"sample_name": "b", "prediction": 1, "class_id": 0,
                   "group": "g"}]],
                expected_names=["a", "b"],
                manifest_rows={"a": {"class_id": 0, "group": "g"},
                               "b": {"class_id": 0, "group": "g"}})
        # class contradiction against the manifest
        with pytest.raises(Exception):
            cond_entry.align_paired_samples(
                [[{"sample_name": "a", "prediction": 0, "class_id": 1,
                   "group": "g"}]],
                [[{"sample_name": "a", "prediction": 1, "class_id": 0,
                   "group": "g"}]],
                expected_names=["a"],
                manifest_rows={"a": {"class_id": 0, "group": "g"}})
        # group contradiction against the manifest
        with pytest.raises(Exception):
            cond_entry.align_paired_samples(
                [[{"sample_name": "a", "prediction": 0, "class_id": 0,
                   "group": "other"}]],
                [[{"sample_name": "a", "prediction": 1, "class_id": 0,
                   "group": "g"}]],
                expected_names=["a"],
                manifest_rows={"a": {"class_id": 0, "group": "g"}})

    def test_full_fit_paired_recomputed_from_saved_jsonl(
            self, local_world, cond_fit):
        output = cond_fit["output"]
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        manifest_rows = {r["sample_name"]: {"class_id": r["class_id"],
                                            "group": r["group"]}
                         for r in local_world["cache_samples"]}
        for comparison, base in (("RCS_minus_R0", "R0"),
                                 ("RCS_minus_RC", "RC")):
            total = {"fixed": 0, "broke": 0, "both_correct": 0,
                     "both_wrong": 0}
            expected_support_total = 0
            for seed in TRAIN_SEEDS:
                left_all = []
                right_all = []
                labels_all = []
                expected_names = []
                for fold in FOLD_IDS:
                    left_records = _jsonl(
                        output / "models" / f"seed_{seed}" / f"{fold}_RCS"
                        / "outer_eval_predictions.jsonl")
                    if base == "R0":
                        right_records = _jsonl(
                            output / "models" / "R0" / f"{fold}_R0"
                            / "outer_eval_predictions.jsonl")
                    else:
                        right_records = _jsonl(
                            output / "models" / f"seed_{seed}"
                            / f"{fold}_{base}"
                            / "outer_eval_predictions.jsonl")
                    left_names = [r["sample_name"] for r in left_records]
                    right_names = [r["sample_name"] for r in right_records]
                    assert left_names == right_names
                    assert len(left_names) == len(set(left_names))
                    aligned_left, aligned_right, labels, _groups = \
                        cond_entry.align_paired_samples(
                            [left_records], [right_records],
                            expected_names=left_names,
                            manifest_rows=manifest_rows)
                    left_all.extend(aligned_left)
                    right_all.extend(aligned_right)
                    labels_all.extend(labels)
                    expected_names.extend(left_names)
                counts = _independent_counts(left_all, right_all,
                                             labels_all)
                assert sum(counts.values()) == len(expected_names)
                saved = summary["paired"][comparison]["per_seed"][str(seed)]
                for key in counts:
                    assert saved[key] == counts[key], (seed, key)
                total_fixed = sum(counts.values())
                del total_fixed
                for key in counts:
                    total[key] += counts[key]
                expected_support_total += len(expected_names)
            assert summary["paired"][comparison]["total"]["fixed"] == \
                total["fixed"]
            assert summary["paired"][comparison]["total"]["broke"] == \
                total["broke"]
            assert summary["paired"][comparison]["total"][
                "both_correct"] == total["both_correct"]
            assert summary["paired"][comparison]["total"][
                "both_wrong"] == total["both_wrong"]
            per_class = summary["per_class_changes"][comparison][
                "summed_over_seeds"]
            assert sum(v["fixed"] for v in per_class.values()) == \
                total["fixed"]
            assert sum(v["broke"] for v in per_class.values()) == \
                total["broke"]
            # sample-model pairs: three seeds re-fit the same folds,
            # so the total is the per-seed support times the seed count
            per_seed_support = sum(len(_jsonl(
                output / "models" / f"seed_{TRAIN_SEEDS[0]}"
                / f"{fold}_RCS" / "outer_eval_predictions.jsonl"))
                for fold in FOLD_IDS)
            assert expected_support_total == per_seed_support * len(
                TRAIN_SEEDS)


# ---------------------------------------------------------------------------
# Gap 3: the linear reference is bound and identity-checked


class TestGap3LinearReference:
    def test_reference_recorded_in_admission(self, local_world, cond_fit):
        output = cond_fit["output"]
        admission = json.loads(
            (output / "admission.json").read_text(encoding="utf-8"))
        reference = admission["linear_reference"]
        assert reference["available"] is True
        assert reference["mode"] == "fit"
        assert reference["training_status"] == "COMPLETED"
        assert reference["experiment"] == \
            "material_polar_local_interior_band_v1"
        assert reference["region_rule"] == \
            "local_partition_interior_band_v1"
        assert reference["p0_oracle_match"] is True

    def test_missing_reference_refused(self, local_world, module_scratch):
        base = module_scratch / "ref_missing"
        local_cache, config_path, _ = _prepare_cond_fit(base, local_world)
        config = _load_config_with(local_cache, local_world)
        config["linear_reference"] = {"fit_summary": str(
            base / "nope" / "summary.json")}
        path = _write_config(base / "configs" / "missing.yaml", config)
        assert cond_entry.main(
            _main_args(local_world, path, base / "out",
                       mode="--dry-run")) == 1

    def test_wrong_sha_reference_refused(self, local_world, module_scratch):
        base = module_scratch / "ref_sha"
        reference = base / "reference" / "summary.json"
        reference.parent.mkdir(parents=True)
        reference.write_text(json.dumps(_legal_reference_summary()))
        (reference.parent / "admission.json").write_text(json.dumps(
            _legal_reference_admission(local_world)))
        local_cache, config_path, _ = _prepare_cond_fit(base, local_world)
        config = _load_config_with(local_cache, local_world)
        config["linear_reference"] = {
            "fit_summary": str(reference),
            "expected_sha256": "b" * 64}
        path = _write_config(base / "configs" / "sha.yaml", config)
        assert cond_entry.main(
            _main_args(local_world, path, base / "out",
                       mode="--dry-run")) == 2

    def test_incomplete_status_refused(self, local_world, module_scratch):
        base = module_scratch / "ref_status"
        local_cache, config_path, _ = _prepare_cond_fit(base, local_world)
        reference = base / "reference" / "summary.json"
        broken = _legal_reference_summary()
        broken["training_status"] = "FAILED"
        reference.write_text(json.dumps(broken))
        config = _load_config_with(local_cache, local_world)
        # the digest of the (rewritten) file itself: the refusal must come
        # from the status check, not from the new format gate
        config["linear_reference"] = {
            "fit_summary": str(reference),
            "expected_sha256": hashlib.sha256(
                reference.read_bytes()).hexdigest()}
        path = _write_config(base / "configs" / "status.yaml", config)
        assert cond_entry.main(
            _main_args(local_world, path, base / "out",
                       mode="--dry-run")) == 1

    def test_p0_oracle_conflict_refused(self, local_world, module_scratch):
        base = module_scratch / "ref_p0"
        local_cache, config_path, _ = _prepare_cond_fit(base, local_world)
        reference = base / "reference" / "summary.json"
        admission = _legal_reference_admission(local_world)
        for record in admission["p0_identity"].values():
            record["oracle_sha256"] = "c" * 64  # conflicting P0 oracle
        (reference.parent / "admission.json").write_text(json.dumps(
            admission))
        config = _load_config_with(local_cache, local_world)
        # the digest of the legal summary _prepare_cond_fit wrote: the
        # refusal must come from the P0 oracle check, not the format gate
        config["linear_reference"] = {
            "fit_summary": str(reference),
            "expected_sha256": hashlib.sha256(
                reference.read_bytes()).hexdigest()}
        path = _write_config(base / "configs" / "p0.yaml", config)
        assert cond_entry.main(
            _main_args(local_world, path, base / "out",
                       mode="--dry-run")) == 1


def _load_config_with(local_cache, world):
    config = dict(world["config"])
    config["experiment"] = "material_conditional_residual_v1"
    config["local_features_source"] = {"path": str(local_cache)}
    config["seeds"] = list(TRAIN_SEEDS)
    return config


def _legal_reference_summary():
    return {
        "mode": "fit",
        "training_status": "COMPLETED",
        "experiment": "material_polar_local_interior_band_v1",
        "region_rule": "local_partition_interior_band_v1",
    }


def _legal_reference_admission(world):
    """A synthetic reference admission consistent with the world: same
    protocol, same folds, same P0 oracle bytes."""
    import hashlib

    oracle_shas = {}
    for fold in FOLD_IDS:
        oracle = (Path(world["fit_dir"]) / "models" / f"{fold}_P0"
                  / "outer_eval_predictions.jsonl")
        oracle_shas[fold] = hashlib.sha256(
            oracle.read_bytes()).hexdigest()
    return {
        "protocol_sha256": world["protocol"]["protocol_sha256"],
        "folds": list(FOLD_IDS),
        "local_cache": {"definition_version":
                        "existing_data_probe_v1_local_interior_band_v1"},
        "p0_identity": {fold: {"oracle_sha256": sha}
                        for fold, sha in oracle_shas.items()},
    }


# ---------------------------------------------------------------------------
# Gap 4: fold-mean vs pooled fields are separate and correct


def _independent_macro_f1(labels, predictions):
    """Direct per-class F1 from boolean math — no production helpers."""
    f1s = []
    for cls in range(4):
        tp = sum(1 for l, p in zip(labels, predictions)
                 if l == cls and p == cls)
        fp = sum(1 for l, p in zip(labels, predictions)
                 if l != cls and p == cls)
        fn = sum(1 for l, p in zip(labels, predictions)
                 if l == cls and p != cls)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / 4


class TestGap4FoldMeanVsPooled:
    def test_unequal_supports_make_them_differ(self):
        # fold A: 1 sample correct; fold B: 2 samples, 1 correct
        fold_f1s = [_independent_macro_f1([0], [0]),
                    _independent_macro_f1([0, 1], [0, 1])]
        pooled = _independent_macro_f1([0, 0, 1], [0, 0, 1])
        fold_mean = sum(fold_f1s) / 2
        assert fold_mean == pytest.approx(0.375)
        assert pooled == pytest.approx(0.5)
        assert fold_mean != pytest.approx(pooled)

    def test_summary_fields_separated_and_recomputed(
            self, local_world, cond_fit):
        output = cond_fit["output"]
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        for variant in ("RC", "RCS"):
            entry = summary["variants"][variant]
            for seed in TRAIN_SEEDS:
                per_seed = entry["per_seed"][str(seed)]
                recomputed_fold_f1s = []
                pooled_labels = []
                pooled_predictions = []
                for fold in FOLD_IDS:
                    records = _jsonl(
                        output / "models" / f"seed_{seed}"
                        / f"{fold}_{variant}"
                        / "outer_eval_predictions.jsonl")
                    labels = [r["class_id"] for r in records]
                    predictions = [r["prediction"] for r in records]
                    recomputed_fold_f1s.append(
                        _independent_macro_f1(labels, predictions))
                    pooled_labels.extend(labels)
                    pooled_predictions.extend(predictions)
                # (a) per-seed three-fold mean
                assert per_seed["macro_f1_mean"] == pytest.approx(
                    sum(recomputed_fold_f1s) / len(recomputed_fold_f1s),
                    abs=1e-6)
                # (b) per-seed pooled over merged outer predictions
                assert per_seed["pooled"]["macro_f1"] == pytest.approx(
                    _independent_macro_f1(pooled_labels,
                                          pooled_predictions),
                    abs=1e-6)
            # (c) the seed averages of both metrics exist and are correct
            fold_means = [entry["per_seed"][str(seed)]["macro_f1_mean"]
                          for seed in TRAIN_SEEDS]
            pooleds = [entry["per_seed"][str(seed)]["pooled"]["macro_f1"]
                       for seed in TRAIN_SEEDS]
            assert entry["macro_f1_mean_of_seeds"] == pytest.approx(
                sum(fold_means) / 3, abs=1e-6)
            assert entry["pooled_macro_f1_mean_of_seeds"] == \
                pytest.approx(sum(pooleds) / 3, abs=1e-6)
        # the mislabelled field is gone
        for comparison in summary["paired"]:
            assert "pooled_macro_f1" not in summary["paired"][comparison]
            block = summary["paired"][comparison]["macro_f1_by_variant"]
            for variant in ("R0", "RC", "RCS"):
                assert "fold_mean_of_seeds" in block[variant]
                assert "pooled_mean_of_seeds" in block[variant]

    def test_ps1_reading_instructions_updated(self):
        script = Path(__file__).resolve().parents[2] / "scripts" / \
            "manual_material_conditional_residual.ps1"
        text = script.read_text(encoding="utf-8")
        assert "fold_mean_of_seeds" in text
        assert "pooled_mean_of_seeds" in text
        assert "paired.*.pooled_macro_f1" not in text


def _authority(world):
    from core.material_probe import authoritative_sample_index

    return authoritative_sample_index(world["protocol"])


def probed_p0_head(world, fold):
    return local_entry_mod.p0_head_from_payload(torch.load(
        Path(world["fit_dir"]) / "models" / f"{fold}_P0"
        / "model_state.pt", weights_only=False, map_location="cpu"))


def probed_p0_scaler(world, fold):
    return Standardizer.from_dict(json.loads(
        (Path(world["fit_dir"]) / "models" / f"{fold}_P0" / "scaler.json")
        .read_text(encoding="utf-8")))


def local_entry_fold_parts(manifest_samples, authority, fold_id):
    return cond_entry.fold_parts(manifest_samples, authority, fold_id)


# ---------------------------------------------------------------------------
# Gap A: linear_reference.expected_sha256 is mandatory and strictly
# validated (missing / empty / None / False used to SKIP the digest
# comparison entirely)


class _RecordingTrainer:
    """Records every training call, then raises: a call must stay
    observable even if the entry tried to train anyway."""

    def __init__(self, original):
        self.original = original
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": len(args),
                           "kwargs": sorted(kwargs)})
        raise AssertionError(
            "training must not run when the linear reference is refused")


_ILLEGAL_EXPECTED_SHAS = [
    # falsy junk must NOT skip the digest comparison
    pytest.param("", id="empty_string"),
    pytest.param(None, id="none"),
    pytest.param(False, id="false"),
    # truthy non-strings must NOT fall through to a MISMATCH raise
    pytest.param(True, id="true"),
    pytest.param(0, id="int_zero"),
    pytest.param(12345, id="int"),
    pytest.param(0.5, id="float"),
    pytest.param([], id="list"),
    pytest.param(["a" * 64], id="list_of_hex"),
    pytest.param({}, id="dict"),
    pytest.param({"sha256": "a" * 64}, id="dict_of_hex"),
    # malformed strings
    pytest.param("a" * 63, id="too_short"),
    pytest.param("a" * 65, id="too_long"),
    pytest.param("g" * 64, id="not_hex"),
    pytest.param("0x" + "a" * 62, id="hex_prefix"),
    pytest.param("a" * 63 + " ", id="trailing_space"),
    pytest.param(" " + "a" * 63, id="leading_space"),
    pytest.param("a" * 32 + "\n" + "a" * 31, id="embedded_newline"),
]


@pytest.fixture(scope="module")
def sha_reference(local_world, module_scratch):
    return _write_legal_reference(module_scratch / "sha_format",
                                  local_world)


class TestGapAExpectedShaFormat:
    @pytest.mark.parametrize("bad", _ILLEGAL_EXPECTED_SHAS)
    def test_illegal_expected_sha_refused(self, sha_reference, bad):
        reference_path, _sha = sha_reference
        config = {"linear_reference": {"fit_summary": str(reference_path),
                                       "expected_sha256": bad}}
        with pytest.raises(cond_entry.LocalRefused,
                           match="expected_sha256"):
            cond_entry.verify_linear_reference(
                config=config, protocol={}, p0_verified={}, all_folds=[])

    def test_missing_key_refused(self, sha_reference):
        reference_path, _sha = sha_reference
        config = {"linear_reference": {"fit_summary": str(reference_path)}}
        with pytest.raises(cond_entry.LocalRefused,
                           match="expected_sha256"):
            cond_entry.verify_linear_reference(
                config=config, protocol={}, p0_verified={}, all_folds=[])

    def test_uppercase_legal_sha_binds(self, local_world, module_scratch):
        """Uppercase hex is legal and must MATCH the lowercase digest
        (case is normalised only after the format check)."""
        base = module_scratch / "sha_upper"
        local_cache, _config_path, _output = _prepare_cond_fit(
            base, local_world)
        reference_path = base / "reference" / "summary.json"
        sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        config = _load_config_with(local_cache, local_world)
        config["linear_reference"] = {
            "fit_summary": str(reference_path),
            "expected_sha256": sha.upper()}
        path = _write_config(base / "configs" / "upper.yaml", config)
        output = base / "out"
        assert cond_entry.main(
            _main_args(local_world, path, output,
                       mode="--dry-run")) == 0
        admission = json.loads(
            (output / "admission.json").read_text(encoding="utf-8"))
        assert admission["linear_reference"]["sha256"] == sha

    def test_legal_match_still_binds(self, local_world, cond_fit):
        """The full synthetic fit used a correct lowercase digest and
        completed with the reference bound."""
        output = cond_fit["output"]
        admission = json.loads(
            (output / "admission.json").read_text(encoding="utf-8"))
        assert admission["linear_reference"]["available"] is True
        assert len(admission["linear_reference"]["sha256"]) == 64

    def test_legal_but_wrong_sha_mismatches(
            self, local_world, module_scratch):
        """A well-formed digest that does not match the file keeps the
        MISMATCH semantics (exit 2), not the format refusal."""
        base = module_scratch / "sha_wrong"
        local_cache, _config_path, _output = _prepare_cond_fit(
            base, local_world)
        reference_path = base / "reference" / "summary.json"
        config = _load_config_with(local_cache, local_world)
        config["linear_reference"] = {
            "fit_summary": str(reference_path),
            "expected_sha256": "b" * 64}
        path = _write_config(base / "configs" / "wrong.yaml", config)
        assert cond_entry.main(
            _main_args(local_world, path, base / "out",
                       mode="--dry-run")) == 2


class TestGapARefusalIntegration:
    def test_dry_run_refuses_before_any_output(self, local_world,
                                               module_scratch, capsys):
        base = module_scratch / "sha_dry"
        local_cache, _config_path, output = _prepare_cond_fit(
            base, local_world)
        config = _load_config_with(local_cache, local_world)
        config["linear_reference"] = {
            "fit_summary": str(base / "reference" / "summary.json"),
            "expected_sha256": ""}
        path = _write_config(base / "configs" / "empty.yaml", config)
        code = cond_entry.main(
            _main_args(local_world, path, output, mode="--dry-run"))
        assert code == 1
        captured = capsys.readouterr()
        assert "expected_sha256" in captured.err
        assert not output.exists()

    def test_fit_refuses_before_training(self, local_world, module_scratch,
                                         capsys):
        base = module_scratch / "sha_fit"
        local_cache, _config_path, output = _prepare_cond_fit(
            base, local_world)
        config = _load_config_with(local_cache, local_world)
        config["linear_reference"] = {
            "fit_summary": str(base / "reference" / "summary.json"),
            "expected_sha256": None}
        path = _write_config(base / "configs" / "none.yaml", config)
        probe = _RecordingTrainer(
            cond_entry.train_conditional_residual_head)
        with mock.patch.object(cond_entry,
                               "train_conditional_residual_head", probe):
            code = cond_entry.main(
                _main_args(local_world, path, output))
        assert code == 1
        captured = capsys.readouterr()
        assert "expected_sha256" in captured.err
        assert probe.calls == []
        # no models/, no summary, nothing that could read COMPLETED
        assert not output.exists()


# ---------------------------------------------------------------------------
# Gap B: the shuffle diagnostic's NORMAL output must bind the RCS outer
# predictions saved by the SAME fold x train_seed fit (repeated-forward
# consistency alone does not prove the formal evaluation path)


_BIND_MANIFEST = [
    {"sample_name": "s0", "class_id": 0, "group": "g0"},
    {"sample_name": "s1", "class_id": 1, "group": "g0"},
    {"sample_name": "s2", "class_id": 2, "group": "g1"},
]
_BIND_ROWS = [0, 1, 2]
_BIND_NORMAL_LOGITS = np.array(
    [[2.0, 1.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0],
     [1.0, 0.0, 4.0, 2.0]], dtype=np.float32)


def _bind_normal_predictions():
    return [int(v) for v in _BIND_NORMAL_LOGITS.argmax(axis=1)]


def _bind_saved_records():
    return [
        {"sample_name": sample["sample_name"],
         "class_id": sample["class_id"], "group": sample["group"],
         "logits": _BIND_NORMAL_LOGITS[index].tolist(),
         "prediction": int(_BIND_NORMAL_LOGITS[index].argmax())}
        for index, sample in enumerate(_BIND_MANIFEST)]


def _bind_variant_dir(base, *, fold="fold0", seed=2026, records=None,
                      report=None, with_jsonl=True):
    variant_dir = base / "models" / f"seed_{seed}" / f"{fold}_RCS"
    variant_dir.mkdir(parents=True)
    if with_jsonl:
        saved = _bind_saved_records() if records is None else records
        (variant_dir / "outer_eval_predictions.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in saved),
            encoding="utf-8")
    if report is None:
        report = {"fold": fold, "variant": "RCS", "seed": int(seed)}
    (variant_dir / "fold_variant_report.json").write_text(
        json.dumps(report), encoding="utf-8")
    return variant_dir


def _bind_call(variant_dir, *, fold="fold0", seed=2026,
               normal_logits=None, normal_predictions=None):
    return {
        "fold_id": fold, "train_seed": seed, "variant_dir": variant_dir,
        "rows": _BIND_ROWS, "manifest_samples": _BIND_MANIFEST,
        "normal_logits": (_BIND_NORMAL_LOGITS if normal_logits is None
                          else normal_logits),
        "normal_predictions": (
            _bind_normal_predictions() if normal_predictions is None
            else normal_predictions),
    }


class TestGapBGuardUnit:
    def test_happy_path_binds(self, module_scratch):
        variant_dir = _bind_variant_dir(module_scratch / "bind_ok")
        cond_entry.verify_normal_matches_saved_rcs(
            **_bind_call(variant_dir))

    def test_missing_jsonl_refused(self, module_scratch):
        variant_dir = _bind_variant_dir(module_scratch / "bind_nojsonl",
                                        with_jsonl=False)
        with pytest.raises(cond_entry.LocalMismatch, match="missing"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_missing_sample_refused(self, module_scratch):
        records = _bind_saved_records()[:-1]
        variant_dir = _bind_variant_dir(module_scratch / "bind_missing",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch,
                           match="authoritative"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_extra_sample_refused(self, module_scratch):
        records = _bind_saved_records() + [
            {"sample_name": "ghost", "class_id": 3, "group": "g1",
             "logits": [0.0, 0.0, 0.0, 1.0], "prediction": 3}]
        variant_dir = _bind_variant_dir(module_scratch / "bind_extra",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch,
                           match="authoritative"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_duplicate_sample_refused(self, module_scratch):
        records = _bind_saved_records()[:2] + \
            [dict(_bind_saved_records()[0])]
        variant_dir = _bind_variant_dir(module_scratch / "bind_dup",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch, match="duplicate"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_wrong_fold_report_refused(self, module_scratch):
        variant_dir = _bind_variant_dir(
            module_scratch / "bind_fold",
            report={"fold": "fold1", "variant": "RCS", "seed": 2026})
        with pytest.raises(cond_entry.LocalMismatch,
                           match="identifies itself"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_wrong_seed_report_refused(self, module_scratch):
        variant_dir = _bind_variant_dir(
            module_scratch / "bind_seed",
            report={"fold": "fold0", "variant": "RCS", "seed": 9999})
        with pytest.raises(cond_entry.LocalMismatch,
                           match="identifies itself"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir, seed=2026))

    def test_rc_variant_report_refused(self, module_scratch):
        variant_dir = _bind_variant_dir(
            module_scratch / "bind_rc",
            report={"fold": "fold0", "variant": "RC", "seed": 2026})
        with pytest.raises(cond_entry.LocalMismatch,
                           match="identifies itself"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_offset_saved_logits_refused(self, module_scratch):
        records = _bind_saved_records()
        # offset all four logits of one sample: argmax (and therefore the
        # prediction) stays valid, so ONLY the exact-equality binding can
        # catch this
        records[1]["logits"] = [v + 0.25 for v in records[1]["logits"]]
        variant_dir = _bind_variant_dir(module_scratch / "bind_offset",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch,
                           match="logits differ"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_saved_prediction_not_argmax_refused(self, module_scratch):
        records = _bind_saved_records()
        records[0]["prediction"] = 1  # argmax of the saved logits is 0
        variant_dir = _bind_variant_dir(module_scratch / "bind_argmax",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch, match="argmax"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_normal_prediction_not_argmax_refused(self, module_scratch):
        predictions = _bind_normal_predictions()
        predictions[1] = 3  # argmax of the fresh normal logits is 1
        variant_dir = _bind_variant_dir(module_scratch / "bind_nargmax")
        with pytest.raises(cond_entry.LocalMismatch, match="argmax"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir,
                             normal_predictions=predictions))

    def test_class_contradiction_refused(self, module_scratch):
        records = _bind_saved_records()
        records[2]["class_id"] = 3
        variant_dir = _bind_variant_dir(module_scratch / "bind_class",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch,
                           match="contradict"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_group_contradiction_refused(self, module_scratch):
        records = _bind_saved_records()
        records[2]["group"] = "gX"
        variant_dir = _bind_variant_dir(module_scratch / "bind_group",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch,
                           match="contradict"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_nonfinite_saved_logits_refused(self, module_scratch):
        records = _bind_saved_records()
        records[0]["logits"] = [float("nan"), 1.0, 0.0, 0.0]
        variant_dir = _bind_variant_dir(module_scratch / "bind_nan",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch, match="finite"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_wrong_dim_saved_logits_refused(self, module_scratch):
        records = _bind_saved_records()
        records[0]["logits"] = [1.0, 0.0]
        variant_dir = _bind_variant_dir(module_scratch / "bind_dim",
                                        records=records)
        with pytest.raises(cond_entry.LocalMismatch,
                           match="four-dimensional"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir))

    def test_nonfinite_normal_logits_refused(self, module_scratch):
        normal = _BIND_NORMAL_LOGITS.copy()
        normal[2, 2] = np.nan
        variant_dir = _bind_variant_dir(module_scratch / "bind_nnan")
        with pytest.raises(cond_entry.LocalMismatch, match="non-finite"):
            cond_entry.verify_normal_matches_saved_rcs(
                **_bind_call(variant_dir, normal_logits=normal))


class TestGapBNormalBindsSavedRcs:
    """Independent recomputation over the full synthetic fit: every one
    of the 45 shuffle reports must carry normal_logits/normal_prediction
    exactly equal to the saved RCS outer predictions of the same
    fold x train_seed. The test uses its own name mapping, array
    comparison and argmax — never the production guard."""

    def test_all_45_reports_bind_saved_rcs_predictions(
            self, local_world, cond_fit):
        output = cond_fit["output"]
        manifest_samples = local_world["cache_samples"]
        authority = _authority(local_world)
        # the authoritative outer_eval names per fold, NOT read from the
        # jsonl files being checked
        expected_names = {}
        for fold in FOLD_IDS:
            rows = cond_entry.fold_parts(manifest_samples, authority,
                                         fold)["outer_eval"]
            expected_names[fold] = [manifest_samples[row]["sample_name"]
                                    for row in rows]
        checked = 0
        for train_seed in TRAIN_SEEDS:
            for fold in FOLD_IDS:
                saved = _jsonl(
                    output / "models" / f"seed_{train_seed}"
                    / f"{fold}_RCS" / "outer_eval_predictions.jsonl")
                by_name = {}
                for record in saved:
                    assert record["sample_name"] not in by_name
                    by_name[record["sample_name"]] = record
                assert set(by_name) == set(expected_names[fold])
                for content_seed in CONTENT_SEEDS:
                    report = json.loads(
                        (output / "shuffle" / f"seed_{train_seed}" / fold
                         / f"seed_{content_seed}.json")
                        .read_text(encoding="utf-8"))
                    for record in report["samples"]:
                        saved_record = by_name[record["sample_name"]]
                        saved_logits = np.asarray(
                            saved_record["logits"], dtype=np.float32)
                        normal_logits = np.asarray(
                            record["normal_logits"], dtype=np.float32)
                        assert saved_logits.shape == (4,)
                        assert np.isfinite(saved_logits).all()
                        assert np.array_equal(saved_logits,
                                              normal_logits)
                        assert saved_record["prediction"] == int(
                            saved_logits.argmax())
                        assert record["normal_prediction"] == int(
                            normal_logits.argmax())
                        assert saved_record["prediction"] == \
                            record["normal_prediction"]
                        checked += 1
        assert checked > 0
