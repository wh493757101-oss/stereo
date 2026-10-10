"""Full synthetic CPU acceptance for the local residual training entry.

One complete legal case drives the REAL ``main([...,"--fit","--fold",
"all"])`` path: real admission (frozen chain + local cache), six tiny
fits (3 folds x RC/RCS), save/reload, outer predictions, summary, and
fifteen shuffle results (3 folds x 5 frozen seeds). Nothing in the main
flow, the training core, the metrics or the shuffle core is mocked.

The synthetic world controls per-object validity patterns so the local
cache covers: both blocks valid, interior-only, band-only, fully
missing, and (by construction of the composite grouping) at least one
single-member shuffle group. 2-5 epoch budgets here are synthetic only;
the formal experiment budget is untouched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from core.material_probe import (
    Standardizer,
    accuracy,
    authoritative_sample_index,
    macro_f1,
)
from core.polar_local_features import (
    build_local_cache_arrays,
    fit_local_scaler,
)
from scripts import train_material_local_residual as local_entry

from .conftest import build_synthetic_source, build_synthetic_v5
from .test_polar_region_quality_entry import (
    _make_cache_files_for_v5,
    _protocol_for_v5,
    _v5_rows,
    _write_config,
)
from .test_p0_baseline_verification import build_fit
import sys as _sys

_p0mod = _sys.modules[build_fit.__module__]

FOLD_IDS = ("fold0", "fold1", "fold2")
SEEDS = (2026, 2027, 2028, 2029, 2030)
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")

# the controlled validity patterns, cycled per source object
PATTERNS = ("both", "interior", "band", "none")


# ---------------------------------------------------------------------------
# A V4 builder with per-object validity patterns (production helpers)


def build_patterned_v4(source_root: Path, v4_root: Path, workspace: Path,
                       crop_pad: int = 10) -> Path:
    """The production V4 generation with a controlled per-object validity
    pattern (both / interior / band / none), cycled in manifest order."""
    import cv2

    from core.fusion_dataset import (
        build_quality_vector,
        save_fusion_sample,
    )
    from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons
    from scripts.prepare_cls_fusion_dataset import (
        FusionSampleRecord,
        audit_fusion_dataset,
        build_object_mask,
        load_gray,
        read_class_names,
        read_source_manifest,
        write_manifest,
    )

    rng = np.random.default_rng(0)
    source_root = Path(source_root)
    v4_root = Path(v4_root)
    v4_root.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(source_root)
    records = []
    pattern_counter = 0
    for row in read_source_manifest(source_root, workspace):
        label_path = source_root / "labels" / row.split / \
            f"{row.output_stem}.txt"
        left_gray = load_gray(row.left_path)
        height, width = left_gray.shape[:2]
        polygons = read_yolo_polygons(label_path, width, height)
        for object_index, annotation in enumerate(polygons):
            mask = build_object_mask(annotation.polygon, width, height)
            window = make_crop_window(annotation.polygon, width, height,
                                      crop_pad)
            crop_gray = left_gray[window.y1:window.y2, window.x1:window.x2]
            crop_mask = mask[window.y1:window.y2, window.x1:window.x2]
            interior = cv2.erode(crop_mask, np.ones((3, 3), np.uint8),
                                 iterations=1)
            band = crop_mask & (1 - interior)
            pattern = PATTERNS[pattern_counter % len(PATTERNS)]
            pattern_counter += 1
            if pattern == "both":
                valid = crop_mask.copy()
            elif pattern == "interior":
                valid = interior.copy()
            elif pattern == "band":
                valid = band.copy()
            else:
                valid = np.zeros_like(crop_mask)
            signed = np.zeros(crop_gray.shape, dtype=np.float32)
            values = (rng.random(crop_gray.shape).astype(np.float32) - 0.5)
            signed[valid > 0] = values[valid > 0]
            abs_q = np.abs(signed).astype(np.float32)
            total = int(crop_mask.sum())
            valid_count = int(valid.sum())
            valid_ratio = valid_count / total if total else 0.0
            mean_abs_q = (float(abs_q[valid > 0].mean())
                          if valid_count else 0.0)
            quality = build_quality_vector(
                valid_ratio=valid_ratio, in_bounds_ratio=1.0,
                brightness_valid_ratio=1.0, mean_abs_q=mean_abs_q)
            class_name = class_names[annotation.class_id]
            sample_name = f"{row.output_stem}_obj{object_index:03d}"
            relative = Path(row.split) / class_name / f"{sample_name}.npz"
            save_fusion_sample(
                v4_root / relative, gray=crop_gray, signed_q=signed,
                abs_q=abs_q, valid=valid, quality=quality,
                class_id=annotation.class_id)
            records.append(FusionSampleRecord(
                sample_name=sample_name, split=row.split,
                group_name=row.group_name, class_id=annotation.class_id,
                class_name=class_name, source_frame=row.output_stem,
                object_index=object_index, stereo_valid=True,
                stereo_reason="ok", stereo_valid_ratio=1.0,
                disparity=100.0, polar_valid_ratio=valid_ratio,
                quality=tuple(float(v) for v in quality),
                npz_path=relative.as_posix()))
    write_manifest(
        v4_root, class_names, records,
        generation={"matching": "bands", "band_margin": 20,
                    "full_image_threshold": 0.9, "crop_pad": crop_pad})
    expected = {split: sum(1 for r in records if r.split == split)
                for split in ("train", "val", "test")}
    audit_fusion_dataset(
        output_root=v4_root, records=records, class_names=class_names,
        audit_root=v4_root / "audit", expected_splits=expected)
    return v4_root


@pytest.fixture(scope="module")
def local_world(module_scratch):
    """Synthetic chain with all four validity patterns + protocol/cache/
    P0 fit/local config (no local cache yet)."""
    workspace = module_scratch.resolve()
    source = build_synthetic_source(
        module_scratch / "source", counts=(4, 4, 4), size=64)
    v4 = build_patterned_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)
    from core.fusion_dataset import dataset_fingerprint
    from core.material_mask_dataset import mask_dataset_fingerprint

    v5_fingerprint = mask_dataset_fingerprint(v5)
    rows = _v5_rows(v5)
    protocol = _protocol_for_v5(rows, v5_fingerprint)
    protocol_dir = module_scratch / "protocol"
    protocol_dir.mkdir(parents=True)
    protocol_path = protocol_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True),
                             encoding="utf-8")

    with mock.patch.object(_p0mod, "make_protocol", lambda: protocol), \
            mock.patch.object(_p0mod, "make_cache_files",
                              _make_cache_files_for_v5(v5, rows)):
        fit = build_fit(module_scratch, name="p0fit")

    (module_scratch / "p0_verification.json").write_text(
        json.dumps({"verified": True}), encoding="utf-8")
    config = {
        "experiment": "material_polar_local_interior_band_v1",
        "protocol_source": {"path": str(protocol_path)},
        "features_source": {"path": str(fit["feature_dir"])},
        "p0_baseline_source": {
            "fit_dir": str(fit["fit_dir"]),
            "verification": str(module_scratch / "p0_verification.json"),
        },
        "local_features_source": {"path": ""},
        "data": {
            "root": str(v5),
            "fingerprint": v5_fingerprint,
            "class_names": list(protocol["data"]["class_names"]),
            "dev_pool_splits": ["train", "val"],
            "excluded_splits": ["test"],
            "parent_v4_fingerprint": dataset_fingerprint(v4),
        },
        "old_global_reference": {"fit_summary": str(
            fit["fit_dir"] / "summary.json")},
        "training": {
            "device": "cpu", "full_batch": True, "epochs": 3,
            "optimizer": "adamw", "lr": 0.01, "weight_decay": 0.001,
            "aux_loss_weight": 0.01, "seed": 2026,
        },
        "variants": {"RC": "C-only", "RCS": "C+S"},
    }
    return {
        "workspace": workspace,
        "source": source,
        "v4": v4,
        "v5": v5,
        "rows": rows,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "feature_dir": fit["feature_dir"],
        "fit_dir": fit["fit_dir"],
        "config": config,
        "cache_samples": fit["manifest"]["samples"],
    }


def _source_arrays(world) -> dict:
    return dict(np.load(world["feature_dir"] / "feature_cache.npz"))


def _p0_dir_digests(world) -> dict:
    digests = {}
    p0_dir = Path(world["fit_dir"]) / "models"
    for path in sorted(p0_dir.rglob("*")):
        if path.is_file():
            key = path.relative_to(p0_dir).as_posix()
            digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


@pytest.fixture(scope="module")
def fit_result(local_world, module_scratch):
    """One complete real main --fit over the synthetic world."""
    local_cache = module_scratch / "acceptance" / "local_features"
    build_local_cache_arrays(
        source_arrays=_source_arrays(local_world),
        samples=local_world["cache_samples"],
        data_root=local_world["v5"],
        protocol=local_world["protocol"],
        output_dir=local_cache)
    config = dict(local_world["config"])
    config["local_features_source"] = {"path": str(local_cache)}
    config_path = _write_config(
        module_scratch / "acceptance" / "configs" / "local.yaml", config)
    output = module_scratch / "acceptance" / "fit_out"
    p0_before = _p0_dir_digests(local_world)
    code = local_entry.main([
        "--config", str(config_path),
        "--fit-dir", str(local_world["fit_dir"]),
        "--output-dir", str(output),
        "--fold", "all",
        "--fit",
    ])
    return {
        "code": code,
        "output": output,
        "local_cache": local_cache,
        "config_path": config_path,
        "p0_before": p0_before,
    }


# ---------------------------------------------------------------------------
# The complete legal synthetic fit


class TestLocalEntryFullFit:
    def test_fit_completes_with_completed_status(self, fit_result):
        assert fit_result["code"] == 0, (
            fit_result["output"] / "summary.json")
        summary = json.loads(
            (fit_result["output"] / "summary.json").read_text(
                encoding="utf-8"))
        assert summary["training_status"] == "COMPLETED"
        assert summary["mode"] == "fit"

    def test_six_fits_and_fifteen_shuffle_files(self, fit_result):
        output = fit_result["output"]
        for fold in FOLD_IDS:
            for variant in ("RC", "RCS"):
                assert (output / "models" / f"{fold}_{variant}"
                        / "residual_state.pt").is_file()
                assert (output / "models" / f"{fold}_{variant}"
                        / "outer_eval_predictions.jsonl").is_file()
            for seed in SEEDS:
                assert (output / "shuffle" / fold / f"seed_{seed}.json"
                        ).is_file()
        assert len(list((output / "models").glob("*_RC"))) == 3
        assert len(list((output / "models").glob("*_RCS"))) == 3
        assert len(list((output / "shuffle").glob("*/seed_*.json"))) == 15

    def test_p0_checkpoints_untouched(self, local_world, fit_result):
        after = _p0_dir_digests(local_world)
        assert after == fit_result["p0_before"]

    def test_outer_samples_aligned_by_sample_name(self, local_world,
                                                  fit_result):
        authority = authoritative_sample_index(local_world["protocol"])
        manifest_samples = local_world["cache_samples"]
        for fold in FOLD_IDS:
            outer_rows = [index for index, sample
                          in enumerate(manifest_samples)
                          if authority[sample["sample_name"]]["roles"][fold]
                          == "outer_eval"]
            outer_names = sorted(manifest_samples[row]["sample_name"]
                                 for row in outer_rows)
            for variant in ("RC", "RCS"):
                records = _jsonl(fit_result["output"] / "models"
                                 / f"{fold}_{variant}"
                                 / "outer_eval_predictions.jsonl")
                names = [r["sample_name"] for r in records]
                assert sorted(names) == outer_names
                assert len(names) == len(set(names))

    def test_independent_metric_recomputation(self, local_world,
                                              fit_result):
        output = fit_result["output"]
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        for variant in ("R0", "RC", "RCS"):
            fold_f1s = []
            pooled_labels = []
            pooled_predictions = []
            for fold in FOLD_IDS:
                records = _jsonl(output / "models" / f"{fold}_{variant}"
                                 / "outer_eval_predictions.jsonl")
                labels = [r["class_id"] for r in records]
                predictions = [r["prediction"] for r in records]
                recomputed_f1 = macro_f1(labels, predictions)
                recomputed_acc = accuracy(labels, predictions)
                if variant == "R0":
                    saved = next(f for f in summary["variants"]["R0"]
                                 ["folds"] if f["fold"] == fold)
                    assert saved["macro_f1"] == pytest.approx(
                        recomputed_f1, abs=1e-6)
                    assert saved["accuracy"] == pytest.approx(
                        recomputed_acc, abs=1e-6)
                else:
                    saved = json.loads(
                        (output / "models" / f"{fold}_{variant}"
                         / "fold_variant_report.json").read_text(
                             encoding="utf-8"))
                    assert saved["outer_eval_macro_f1"] == pytest.approx(
                        recomputed_f1, abs=1e-6)
                    assert saved["outer_eval_accuracy"] == pytest.approx(
                        recomputed_acc, abs=1e-6)
                fold_f1s.append(recomputed_f1)
                pooled_labels.extend(labels)
                pooled_predictions.extend(predictions)
            entry = summary["variants"][variant]
            assert entry["macro_f1_mean"] == pytest.approx(
                float(np.mean(fold_f1s)), abs=1e-6)
            recomputed_pooled = local_entry._metrics(
                pooled_labels, pooled_predictions)
            assert entry["pooled"]["macro_f1"] == pytest.approx(
                recomputed_pooled["macro_f1"], abs=1e-6)
            assert entry["pooled_sample_count"] == len(pooled_labels)
            # pooled support must equal the summed per-fold support
            saved_support = sum(
                f["support"] if variant != "R0" else f["rows"]
                for f in entry["folds"])
            assert entry["pooled_sample_count"] == saved_support

    def test_paired_and_per_comparison_class_counts(self, local_world,
                                                    fit_result):
        output = fit_result["output"]
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        authority = authoritative_sample_index(local_world["protocol"])
        manifest_samples = local_world["cache_samples"]
        evidence = {}
        for variant in ("R0", "RC", "RCS"):
            per_fold = {}
            for fold in FOLD_IDS:
                records = _jsonl(output / "models" / f"{fold}_{variant}"
                                 / "outer_eval_predictions.jsonl")
                per_fold[fold] = {
                    "labels": [r["class_id"] for r in records],
                    "predictions": [r["prediction"] for r in records],
                    "groups": [r["group"] for r in records],
                }
            evidence[variant] = per_fold
        for comparison, base in (("RCS_minus_R0", "R0"),
                                 ("RCS_minus_RC", "RC")):
            left_predictions = []
            right_predictions = []
            labels_all = []
            groups_all = []
            for fold in FOLD_IDS:
                left = evidence["RCS"][fold]
                right = evidence[base][fold]
                left_predictions.extend(left["predictions"])
                right_predictions.extend(right["predictions"])
                labels_all.extend(left["labels"])
                groups_all.extend(left["groups"])
            recomputed = local_entry.count_fixed_broke(
                left_predictions=left_predictions,
                right_predictions=right_predictions,
                labels=labels_all,
                group_by_row=groups_all,
                comparison=comparison)
            saved_paired = summary["paired"][comparison]
            assert saved_paired["fixed"] == recomputed["fixed"]
            assert saved_paired["broke"] == recomputed["broke"]
            assert saved_paired["both_correct"] == \
                recomputed["both_correct"]
            assert saved_paired["both_wrong"] == recomputed["both_wrong"]
            class_fixed = sum(v["fixed"] for v in
                              summary["per_class_changes"][comparison]
                              .values())
            class_broke = sum(v["broke"] for v in
                              summary["per_class_changes"][comparison]
                              .values())
            assert class_fixed == recomputed["fixed"]
            assert class_broke == recomputed["broke"]
            assert set(summary["per_class_changes"]) == {
                "RCS_minus_R0", "RCS_minus_RC"}
            assert set(summary["per_group_changes"]) == {
                "RCS_minus_R0", "RCS_minus_RC"}

    def test_local_scaler_uses_inner_train_only(self, local_world,
                                                fit_result):
        local_cache = dict(np.load(fit_result["local_cache"]
                                   / "feature_cache.npz"))
        manifest_samples = local_world["cache_samples"]
        for fold in FOLD_IDS:
            for variant in ("RC", "RCS"):
                saved = Standardizer.from_dict(json.loads(
                    (fit_result["output"] / "models" / f"{fold}_{variant}"
                     / "scaler.json").read_text(encoding="utf-8")))
                inner_rows = [
                    index for index, sample in enumerate(manifest_samples)
                    if sample["fold_roles"][fold] == "inner_train"]
                expected, _info = fit_local_scaler(local_cache, inner_rows)
                assert np.allclose(saved.mean, expected.mean)
                assert np.allclose(saved.std, expected.std)

    def test_saved_models_reload_consistently(self, local_world,
                                              fit_result):
        output = fit_result["output"]
        local_cache = dict(np.load(fit_result["local_cache"]
                                   / "feature_cache.npz"))
        manifest_samples = local_world["cache_samples"]
        authority = authoritative_sample_index(local_world["protocol"])
        p0_fit_dir = Path(local_world["fit_dir"])
        from scripts.verify_p0_baseline import LOGIT_ATOL
        for fold in FOLD_IDS:
            # the independent ORACLE: the frozen P0 fit's saved prediction
            # file (its logits are the correct-P0 forward) — the anchor
            # for the base identity, NOT the entry helper's self-recompute
            oracle = _jsonl(p0_fit_dir / "models" / f"{fold}_P0"
                            / "outer_eval_predictions.jsonl")
            oracle_logits = np.asarray([r["logits"] for r in oracle],
                                       dtype=np.float32)
            oracle_predictions = [r["prediction"] for r in oracle]
            rows_eval = local_entry.fold_parts(manifest_samples,
                                               authority,
                                               fold)["outer_eval"]
            r0_records = _jsonl(output / "models" / f"{fold}_R0"
                                / "outer_eval_predictions.jsonl")
            r0_logits = np.asarray([r["logits"] for r in r0_records],
                                   dtype=np.float32)
            assert float(np.max(np.abs(r0_logits - oracle_logits))) <= \
                LOGIT_ATOL
            assert [r["prediction"] for r in r0_records] == \
                oracle_predictions
            # every R0 record is self-consistent
            for record in r0_records:
                assert record["prediction"] == int(
                    np.argmax(record["logits"]))
            p0_logits = r0_logits
            for variant in ("RC", "RCS"):
                payload = torch.load(
                    output / "models" / f"{fold}_{variant}"
                    / "residual_state.pt", weights_only=False,
                    map_location="cpu")
                head = local_entry.build_residual_head(
                    seed=int(payload["seed"]),
                    input_dim=int(payload["input_dim"]))
                head.load_state_dict(payload["state_dict"])
                head.eval()
                scaler = Standardizer.from_dict(json.loads(
                    (output / "models" / f"{fold}_{variant}"
                     / "scaler.json").read_text(encoding="utf-8")))
                standardized = local_entry._standardize_local(
                    local_cache, scaler, rows_eval)
                valid = local_cache["s_valid"][rows_eval].astype(bool)
                x_eval = local_entry._local_design(standardized, valid,
                                                   variant)
                with torch.inference_mode():
                    delta = local_entry.compute_delta(
                        head, torch.from_numpy(x_eval),
                        torch.from_numpy(valid.astype(np.float32))
                    ).numpy().astype(np.float32)
                logits = p0_logits + delta
                saved_records = _jsonl(
                    output / "models" / f"{fold}_{variant}"
                    / "outer_eval_predictions.jsonl")
                saved_logits = np.asarray(
                    [r["logits"] for r in saved_records], dtype=np.float32)
                assert np.allclose(logits, saved_logits, atol=1e-6)
                assert [r["prediction"] for r in saved_records] == [
                    int(v) for v in logits.argmax(axis=1)]
                assert np.allclose(
                    np.asarray([r["p0_logits"] for r in saved_records]),
                    p0_logits, atol=1e-6)

    def test_shuffle_contract(self, local_world, fit_result):
        output = fit_result["output"]
        local_cache = dict(np.load(fit_result["local_cache"]
                                   / "feature_cache.npz"))
        manifest_samples = local_world["cache_samples"]
        total_unmatched = 0
        for fold in FOLD_IDS:
            saved_outer = {r["sample_name"]: r for r in
                           _jsonl(output / "models" / f"{fold}_RCS"
                                  / "outer_eval_predictions.jsonl")}
            for seed in SEEDS:
                seed_report = json.loads(
                    (output / "shuffle" / fold / f"seed_{seed}.json")
                    .read_text(encoding="utf-8"))
                records = seed_report["samples"]
                outer_names = sorted(saved_outer)
                assert sorted(r["sample_name"] for r in records) == \
                    outer_names
                assert seed_report["receivers"] == sum(
                    1 for r in records if r["has_valid_s"])
                matched = [r for r in records if r["status"] == "matched"]
                assert seed_report["matched"] == len(matched)
                total_unmatched += sum(1 for r in records
                                       if r["status"] == "unmatched")
                grouped_donors: dict = {}
                for record in records:
                    row_index = _row_index(manifest_samples,
                                           record["sample_name"])
                    if record["status"] in ("unmatched", "invalid_s"):
                        assert record["shuffled_logits"] == \
                            record["normal_logits"]
                        assert record["shuffled_prediction"] == \
                            record["normal_prediction"]
                        assert record["donor_row"] is None
                        continue
                    assert record["donor_row"] != record["receiver_row"]
                    assert record["donor_split"] == record["receiver_split"]
                    assert record["donor_original_bin"] == \
                        record["original_bin"]
                    assert record["donor_interior_valid"] == \
                        record["interior_valid"]
                    assert record["donor_band_valid"] == \
                        record["band_valid"]
                    assert record["interior_block_swapped"] == \
                        record["interior_valid"]
                    assert record["band_block_swapped"] == \
                        record["band_valid"]
                    assert record["has_valid_s"] == \
                        bool(local_cache["s_valid"][row_index])
                    group_key = (record["receiver_split"],
                                 record["composite_bin"])
                    grouped_donors.setdefault(group_key, set()).add(
                        record["donor_row"])
                for group_key, donors in grouped_donors.items():
                    group_receivers = {
                        r["receiver_row"] for r in matched
                        if (r["receiver_split"], r["composite_bin"])
                        == group_key}
                    assert donors == group_receivers
                for record in records:
                    assert record["normal_logits"] == \
                        saved_outer[record["sample_name"]]["logits"]
                for region in ("interior", "band"):
                    denominator = seed_report[f"{region}_valid_receivers"]
                    swapped = seed_report[f"{region}_swapped"]
                    assert 0 <= swapped <= denominator
        assert total_unmatched >= 1

    def test_p0_identity_recorded_and_shared_base(self, fit_result):
        from scripts.verify_p0_baseline import LOGIT_ATOL

        output = fit_result["output"]
        admission = json.loads(
            (output / "admission.json").read_text(encoding="utf-8"))
        assert set(admission["p0_identity"]) == set(FOLD_IDS)
        for record in admission["p0_identity"].values():
            assert record["prediction_mismatches"] == 0
            assert record["max_logit_error"] <= LOGIT_ATOL
            assert all(v == 0.0 for v in
                       record["disabled_slot_max_abs"].values())
            assert record["missing_s_rows"] >= 0
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        assert set(summary["p0_identity"]) == set(FOLD_IDS)
        total_missing = 0
        for fold in FOLD_IDS:
            r0 = _jsonl(output / "models" / f"{fold}_R0"
                        / "outer_eval_predictions.jsonl")
            base = {r["sample_name"]: r["logits"] for r in r0}
            total_missing += sum(1 for r in r0 if not r["has_valid_s"])
            for record in r0:
                assert record["prediction"] == int(
                    np.argmax(record["logits"]))
            for variant in ("RC", "RCS"):
                for record in _jsonl(
                        output / "models" / f"{fold}_{variant}"
                        / "outer_eval_predictions.jsonl"):
                    assert record["p0_logits"] ==                         base[record["sample_name"]]
        assert total_missing > 0

    def test_world_covers_all_validity_patterns(self, fit_result):
        manifest = json.loads(
            (fit_result["local_cache"] / "feature_manifest.json")
            .read_text(encoding="utf-8"))
        patterns = {(bool(r["interior_valid_pixels"]),
                     bool(r["band_valid_pixels"])) for r in
                    manifest["samples"]}
        assert patterns == {(True, True), (True, False), (False, True),
                            (False, False)}


# ---------------------------------------------------------------------------
# Failure paths


class TestLocalEntryFailurePaths:
    def test_diagnostic_failure_does_not_claim_completed(
            self, local_world, module_scratch, monkeypatch):
        base = module_scratch / "fault"
        local_cache, config_path, _ = _prepare_fit(base, local_world)
        output = base / "fit_out"

        def explode(**kwargs):
            raise RuntimeError("synthetic diagnostic failure")

        with mock.patch.object(local_entry, "run_local_shuffle_for_fold",
                               explode):
            code = local_entry.main(
                _main_args(local_world, config_path, output))
        assert code != 0
        summary = json.loads(
            (output / "summary.json").read_text(encoding="utf-8"))
        assert summary["training_status"] == "FAILED"
        assert summary["failed_phase"] == "shuffle_diagnostic"
        assert "synthetic diagnostic failure" in summary["failure"]
        assert "shuffle" not in summary
        assert (output / "models" / "fold0_RC" / "residual_state.pt"
                ).is_file()

    def test_existing_output_dir_refused(self, local_world, module_scratch):
        base = module_scratch / "refuse"
        local_cache, config_path, _ = _prepare_fit(base, local_world)
        output = base / "taken"
        output.mkdir()
        assert local_entry.main(
            _main_args(local_world, config_path, output,
                       mode="--dry-run")) == 1


# ---------------------------------------------------------------------------
# Pure counting helper: per-comparison attribution


class TestCountFixedBroke:
    def test_same_sample_fixed_counts_once_per_comparison(self):
        r0 = local_entry.count_fixed_broke(
            left_predictions=[0], right_predictions=[1], labels=[0],
            group_by_row=["g"], comparison="RCS_minus_R0")
        rc = local_entry.count_fixed_broke(
            left_predictions=[0], right_predictions=[2], labels=[0],
            group_by_row=["g"], comparison="RCS_minus_RC")
        assert r0["fixed"] == 1 and r0["broke"] == 0
        assert rc["fixed"] == 1 and rc["broke"] == 0
        assert r0["comparison"] != rc["comparison"]
        assert r0["per_class"][CLASS_NAMES[0]]["fixed"] == 1
        assert rc["per_class"][CLASS_NAMES[0]]["fixed"] == 1
        assert r0["per_group"]["g"]["fixed"] == 1
        assert rc["per_group"]["g"]["fixed"] == 1

    def test_broke_and_both_cases(self):
        result = local_entry.count_fixed_broke(
            left_predictions=[1, 0, 2, 3],
            right_predictions=[0, 1, 2, 0],
            labels=[0, 0, 2, 1],
            group_by_row=["g1", "g1", "g2", "g2"],
            comparison="RCS_minus_R0")
        assert result["fixed"] == 1
        assert result["broke"] == 1
        assert result["both_correct"] == 1
        assert result["both_wrong"] == 1


# ---------------------------------------------------------------------------
# Helpers


def _prepare_fit(base: Path, world):
    local_cache = base / "local_features"
    build_local_cache_arrays(
        source_arrays=_source_arrays(world),
        samples=world["cache_samples"],
        data_root=world["v5"],
        protocol=world["protocol"],
        output_dir=local_cache)
    config = dict(world["config"])
    config["local_features_source"] = {"path": str(local_cache)}
    config_path = _write_config(base / "configs" / "local.yaml", config)
    return local_cache, config_path, base / "fit_out"


def _main_args(world, config_path: Path, output: Path,
               mode: str = "--fit") -> list:
    return [
        "--config", str(config_path),
        "--fit-dir", str(world["fit_dir"]),
        "--output-dir", str(output),
        "--fold", "all",
        mode,
    ]


def _jsonl(path: Path) -> list:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines()]


def _row_index(manifest_samples, sample_name):
    for index, sample in enumerate(manifest_samples):
        if sample["sample_name"] == sample_name:
            return index
    raise KeyError(sample_name)
