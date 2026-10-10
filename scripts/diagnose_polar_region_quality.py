"""Corrected READ-ONLY region diagnostics for the polarimetric S slot.

2A corrective batch. This entry migrates the necessary logic of the
historical (untracked) runner in
``analysis/runs/material_polar_residual_v1/20261008T144500Z_e7f8a9/`` —
same algorithms, same frozen statistical definitions — and closes three
confirmed acceptance gaps:

1. Real protection evidence: a BEFORE snapshot over existing protected
   inputs (config/protocol/cache/P0 fit/user fit/V5/historical 2A
   evidence), an independent AFTER re-computation, and a comparison that
   checks file sets plus per-file digests and byte lengths. Missing
   required assets are a controlled refusal; an all-missing snapshot can
   never pass. The historical 26-item snapshots are recorded as
   ``INVALID_PROTECTION_EVIDENCE`` (their project root resolved to
   ``analysis/runs``; every record was ``{"missing": true}``) with
   ``historical_before_after_integrity = UNVERIFIED``.
2. V5 full admission THROUGH THE PROJECT'S OWN GATE
   (``admit_dataset_for_protocol`` -> ``admit_mask_dataset`` ->
   ``verify_mask_sources``) with the fingerprint taken from the frozen
   config/protocol/feature-definition — executed BEFORE any per-sample
   region statistics. The original cache/P0 ``run_admission`` chain is
   retained.
3. Measurement semantics: the four-direction shift experiment is
   declared as ``perturbation_type = saved_q_roi_sampling_shift``
   (resampling of the ALREADY-SAVED q maps). It is NOT a left-right
   misregistration recompute, and drift mean/p95 relative to the
   between-sample spread are distribution summaries, NOT error bounds —
   the per-sample maximum is reported separately.

Authorization boundary: no training, no feature-cache build, no SGBM,
no model improvement, no deployment. ``--output-dir`` must be a fresh
directory outside every input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from core import polar_region_quality
from core.material_mask_dataset import (
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    V4_PARENT_FINGERPRINT,
    admit_dataset_for_protocol,
    load_mask_sample,
)
from core.material_probe import POLAR_STAT_KEYS
from core.polar_region_quality import (
    BAND_DEFINITION,
    INTERIOR_DEFINITION,
    PERTURBATION_HIGHLIGHT,
    PERTURBATION_TYPE,
    REGION_RULE_ID,
    ProtectionError,
    build_protection_snapshot,
    compare_protection_snapshots,
    drift_statistics,
    evaluate_region_rule,
    locate_project_root,
    missing_required_targets,
    verify_sample_correspondence,
)

EXIT_OK = 0
EXIT_REFUSED = 1
QUANTILE_POINTS = (0.05, 0.25, 0.5, 0.75, 0.95)
FOLD_IDS = ("fold0", "fold1", "fold2")

# Fixed extra protection targets (project-relative), beyond the inputs
# named by the config: the historical evidence, frozen weights and the
# unchanged sources this round must not touch (the historical 2A
# protection list as the starting point). Sources this batch IS
# authorized to change are excluded here and listed in
# INTENTIONALLY_MODIFIED instead.
PROTECTED_EXTRA_TARGETS = (
    "analysis/runs/material_polar_residual_v1/20261008T144500Z_e7f8a9",
    "analysis/runs/material_polar_residual_v1/20261008T160000Z_b3c4d5",
    "analysis/runs/material_polar_residual_v1/20261008T180000Z_e4f5a6",
    "analysis/runs/material_existing_data_v1/20260930T021500Z_a3f7c1",
    "analysis/runs/material_existing_data_v1/20260930T045500Z_c5d9e8",
    "analysis/runs/material_existing_data_v1/20261008T140000Z_a9b8c7",
    "configs/default.yaml",
    "weights/pretrained/mit_b0.pth",
    "core/polar_compute.py",
    "core/material_mask_dataset.py",
    "core/material_probe.py",
    "core/material_residual.py",
    "core/existing_data_protocol.py",
    "scripts/train_material_residual.py",
    "scripts/verify_p0_baseline.py",
)

# Sources this batch is authorized to change; they are NOT protected
# assets and are listed in the protection report instead.
INTENTIONALLY_MODIFIED = (
    "core/polar_region_quality.py",
    "scripts/diagnose_polar_region_quality.py",
    "tests/material_fusion/test_polar_region_quality.py",
    "tests/material_fusion/test_polar_region_quality_entry.py",
    "docs/experiments/material_polar_residual_v1.md",
    "HANDOVER.md",
)

# The historical 2A protection snapshots compared two all-missing
# objects; they prove nothing and are marked accordingly.
HISTORICAL_PROTECTION_EVIDENCE = {
    "files": [
        "analysis/runs/material_polar_residual_v1/20261008T144500Z_e7f8a9"
        "/region_diagnostics/protected_before.json",
        "analysis/runs/material_polar_residual_v1/20261008T144500Z_e7f8a9"
        "/region_diagnostics/protected_after.json",
    ],
    "status": "INVALID_PROTECTION_EVIDENCE",
    "historical_before_after_integrity": "UNVERIFIED",
    "reason": (
        "the historical snapshot script derived its project root as "
        "parents[3] of the evidence file, which resolved to "
        "analysis/runs; all 26 records were {\"missing\": true} and the "
        "two all-missing objects compared equal. The historical "
        "pre-execution state is not reconstructed here."),
}


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True, allow_nan=False) + "\n")


def _clean(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def quantiles_of(values, points=QUANTILE_POINTS):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return {str(p): None for p in points}
    array = np.asarray(finite, dtype=np.float64)
    return {str(p): _clean(np.quantile(array, p)) for p in points}


def mean_of(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return _clean(np.mean(finite)) if finite else None


# ---------------------------------------------------------------------------
# Inputs and correspondence


def load_v5_rows(manifest_path: Path) -> dict:
    with manifest_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty V5 manifest: {manifest_path}")
    by_name = {}
    for row in rows:
        name = row["sample_name"]
        if name in by_name:
            raise ValueError(f"duplicate V5 manifest row: {name}")
        by_name[name] = row
    return by_name


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return image


_RIGHT_CONVENTIONS = (("\\left\\", "\\right\\"), ("/left/", "/right/"),
                      ("\\images\\", "\\images_right\\"),
                      ("/images/", "/images_right/"))


def right_frame_path(left_path: str):
    """Derive the rectified right frame; ``None`` if no convention fits."""
    for old, new in _RIGHT_CONVENTIONS:
        if old in left_path:
            return left_path.replace(old, new, 1)
    return None


def load_variant_predictions(fit_dir: Path, fold_id: str,
                             variant: str) -> dict:
    path = fit_dir / "models" / f"{fold_id}_{variant}" / \
        "outer_eval_predictions.jsonl"
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            predictions[record["sample_name"]] = int(record["prediction"])
    return predictions


# ---------------------------------------------------------------------------
# Subset construction and aggregation (same definitions as the historical
# run; clarified labelling and denominators)


def build_subsets(records: list[dict]) -> dict:
    """Labelled row subsets: rule design = per-fold inner_train only.

    The union of the three folds' inner_train rows is a SUMMARY view and
    is labelled as such — it is not any single fold's training set.
    inner_val / outer_eval / full-pool subsets stay descriptive.
    """
    subsets: dict[str, dict] = {}
    for fold_id in FOLD_IDS:
        subsets[f"{fold_id}_inner_train"] = {
            "role": "rule_design_inner_train",
            "records": [r for r in records
                        if r["fold_roles"].get(fold_id) == "inner_train"],
        }
    subsets["inner_train_all_rule_design"] = {
        "role": "summary_union_of_inner_train",
        "records": [r for r in records
                    if any(r["fold_roles"].get(fold_id) == "inner_train"
                           for fold_id in FOLD_IDS)],
    }
    for fold_id in FOLD_IDS:
        subsets[f"{fold_id}_inner_val_descriptive"] = {
            "role": "descriptive_inner_val",
            "records": [r for r in records
                        if r["fold_roles"].get(fold_id) == "inner_val"],
        }
        subsets[f"{fold_id}_outer_eval_descriptive"] = {
            "role": "descriptive_outer_eval",
            "records": [r for r in records
                        if r["fold_roles"].get(fold_id) == "outer_eval"],
        }
    subsets["full_pool_descriptive"] = {
        "role": "descriptive_full_pool", "records": list(records),
    }
    return subsets


_NUMERIC_KEYS = (
    ("ratios", "interior_valid_ratio"),
    ("ratios", "band_valid_ratio"),
    ("ratios", "region_coverage_of_mask"),
    ("counts", "region_pixels"),
    ("counts", "interior_pixels"),
    ("counts", "interior_valid_pixels"),
)
_DEGENERATE_KEYS = (
    "gray_eq_255", "gray_ge_250", "gray_eq_0", "gray_le_10", "abs_q_eq_1",
    "denom_known_share",
)
_DENOM_LEVELS = ("20.0", "30.0", "50.0", "100.0")

AGGREGATION_UNIT_NOTE = (
    "every ratio/fraction is aggregated as the MEAN of per-sample values; "
    "per-sample denominators: interior_valid_ratio over that sample's "
    "interior_pixels, band_valid_ratio over band_pixels, degenerate "
    "fractions over the region's pixel count, denom_below over the "
    "sample's denom-known pixels. These are NOT merged pixel ratios.")


def aggregate_records(records: list[dict]) -> dict:
    """Aggregate per-sample diagnostics for one row subset."""
    count = len(records)
    if count == 0:
        return {"samples": 0,
                "aggregation_unit": AGGREGATION_UNIT_NOTE}
    fallback = sum(1 for r in records if r["rule"]["fallback_used"])
    valid_s = sum(1 for r in records if r["rule"]["has_valid_s"])
    interior_region = sum(1 for r in records
                          if r["rule"]["chosen_region"]
                          == "interior_erode3x3")
    aggregate = {
        "samples": count,
        "aggregation_unit": AGGREGATION_UNIT_NOTE,
        "has_valid_s": valid_s,
        "has_valid_s_share": _clean(valid_s / count),
        "missing_s_samples": count - valid_s,
        "missing_s_note": (
            "samples with NO valid S at all (original definition); "
            "distinct from rule_fallback_used, which counts only "
            "interior-empty-but-original-valid samples"),
        "rule_fallback_used": fallback,
        "rule_fallback_share": _clean(fallback / count),
        "rule_interior_chosen": interior_region,
        "region_pixels_quantiles": quantiles_of(
            [r["diagnostics"]["counts"]["region_pixels"] for r in records]),
        "interior_share_of_region_quantiles": quantiles_of([
            _clean(r["diagnostics"]["counts"]["interior_valid_pixels"]
                   / r["diagnostics"]["counts"]["region_pixels"])
            if r["diagnostics"]["counts"]["region_pixels"] else None
            for r in records]),
    }
    for section, key in _NUMERIC_KEYS:
        aggregate[f"{section}_{key}"] = {
            "mean": mean_of([r["diagnostics"].get(section, {}).get(key)
                             for r in records]),
            "quantiles": quantiles_of(
                [r["diagnostics"].get(section, {}).get(key)
                 for r in records]),
        }
    for block in ("region", "interior", "band"):
        block_stats = {}
        for key in _DEGENERATE_KEYS:
            block_stats[key] = mean_of([
                r["diagnostics"].get("degenerate", {}).get(block, {})
                .get(key) for r in records])
        for level in _DENOM_LEVELS:
            block_stats[f"denom_below_{level}"] = mean_of([
                r["diagnostics"].get("degenerate", {}).get(block, {})
                .get("denom_below", {}).get(level) for r in records])
        aggregate[f"degenerate_{block}_mean"] = block_stats
    for key in ("region_pixel_share_band", "heavy_q_share_band",
                "mean_abs_q_interior", "mean_abs_q_band"):
        aggregate[f"band_influence_{key}"] = {
            "mean": mean_of([r["diagnostics"].get("band_influence", {})
                             .get(key) for r in records]),
            "quantiles": quantiles_of(
                [r["diagnostics"].get("band_influence", {}).get(key)
                 for r in records]),
        }
    for label, field in (("original", "max_abs_delta_original"),
                         ("rule", "max_abs_delta_rule")):
        for stat_key in PERTURBATION_HIGHLIGHT:
            values = [r["diagnostics"].get("perturbation", {}).get(field,
                      {}).get(stat_key) for r in records]
            aggregate[f"perturbation_{label}_{stat_key}"] = {
                "mean": mean_of(values),
                "quantiles": quantiles_of(values),
            }
    win_rates = {}
    for stat_key in PERTURBATION_HIGHLIGHT:
        compared = 0
        wins = 0
        for record in records:
            original = record["diagnostics"].get("perturbation", {}) \
                .get("max_abs_delta_original", {}).get(stat_key)
            rule = record["diagnostics"].get("perturbation", {}) \
                .get("max_abs_delta_rule", {}).get(stat_key)
            if original is None or rule is None:
                continue
            compared += 1
            wins += int(rule < original)
        win_rates[stat_key] = {
            "compared": compared,
            "rule_strictly_more_stable": wins,
            "share": _clean(wins / compared) if compared else None,
        }
    aggregate["perturbation_rule_win_rate"] = win_rates
    aggregate["truncation_touch_share"] = _clean(
        sum(1 for r in records
            if r["diagnostics"].get("truncation", {})
            .get("mask_touches_crop_border")) / count)
    return aggregate


def compose_drift_summary(records: list[dict], *,
                          subset_label: str) -> dict:
    """Drift mean/p50/p95/max per highlighted statistic, plus the
    between-sample spread reference and the derived ratios.

    mean/std and p95/std are distribution summaries over THIS pool; they
    are not error bounds. The per-sample worst case (max) is reported
    alongside and is the observed worst case in this pool only.
    """
    def collect(field: str, key: str) -> list[float]:
        values = []
        for record in records:
            value = record["diagnostics"].get("perturbation", {}) \
                .get(field, {}).get(key)
            if value is not None:
                values.append(value)
        return values

    drift = {}
    for key in PERTURBATION_HIGHLIGHT:
        drift[key] = {
            "original": drift_statistics(collect("max_abs_delta_original",
                                                 key)),
            "rule": drift_statistics(collect("max_abs_delta_rule", key)),
        }
    between_sample = {}
    for key in PERTURBATION_HIGHLIGHT:
        stat_index = POLAR_STAT_KEYS.index(key)
        values = [record["diagnostics"]["original_stats"][stat_index]
                  for record in records
                  if record.get("rule", {}).get("has_valid_s")
                  and record["diagnostics"].get("original_stats")
                  is not None]
        value = _clean(np.std(values, ddof=0)) if len(values) > 1 else None
        between_sample[key] = {
            "value": value, "n": len(values), "subset": subset_label,
        }
    ratios = {}
    for kind, stats in (("mean", "mean"), ("p95", "p95")):
        block = {}
        for key in PERTURBATION_HIGHLIGHT:
            spread = between_sample[key]["value"]
            entry = {}
            for region in ("original", "rule"):
                statistic = drift[key][region][stats]
                entry[region] = _clean(statistic / spread) \
                    if spread else None
            block[key] = entry
        ratios[f"{kind}_over_std"] = block
    return {
        "perturbation_type": PERTURBATION_TYPE,
        "note": (
            "per-sample worst-case drift over the four +-1px shifts of "
            "the SAVED q maps (saved_q_roi_sampling_shift). mean/std and "
            "p95/std are distribution summaries over this pool, NOT "
            "error bounds; the per-sample maximum is reported alongside "
            "and is only the observed worst case of this pool. This "
            "experiment does not displace the right image and cannot "
            "validate left-right registration, exposure/gain/sync "
            "consistency, or disparity reliability."),
        "drift": drift,
        "between_sample_std": between_sample,
        **ratios,
    }


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only region diagnostics for the S slot with "
                    "full V5 admission and real protection evidence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fit-dir", required=True,
                        help="the completed residual fit directory "
                             "(read-only; RC/RCS saved predictions are "
                             "read if present)")
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; must not exist and must "
                             "not lie inside any input")
    args = parser.parse_args(argv)

    started = time.time()
    output_dir = Path(args.output_dir)
    try:
        project_root = locate_project_root(__file__)
        if output_dir.exists():
            raise ValueError(
                f"output directory {output_dir} already exists; refuse to "
                "overwrite any prior evidence")

        config = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8"))
        if config.get("experiment") != "material_polar_residual_v1":
            raise ValueError(
                f"config experiment {config.get('experiment')!r} is not "
                "material_polar_residual_v1")

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else project_root / path

        protocol_path = resolve(config["protocol_source"]["path"])
        features_dir = resolve(config["features_source"]["path"])
        p0_fit_dir = resolve(config["p0_baseline_source"]["fit_dir"])
        fit_dir = resolve(args.fit_dir)
        data_root = resolve(config["data"]["root"])
        extra_targets = [project_root / target
                         for target in PROTECTED_EXTRA_TARGETS]
        anchors = [Path(args.config), protocol_path, features_dir,
                   p0_fit_dir, fit_dir, data_root] + extra_targets
        output_resolved = output_dir.resolve()
        for anchor in anchors:
            anchor_dir = anchor if anchor.is_dir() else anchor.parent
            anchor_real = anchor_dir.resolve()
            if output_resolved == anchor_real or \
                    anchor_real in output_resolved.parents:
                raise ValueError(
                    f"output directory {output_dir} lies inside input "
                    f"{anchor_dir}")
            if output_resolved in anchor_real.parents:
                raise ValueError(
                    f"input {anchor_dir} lies inside the output directory")

        target_names = [str(args.config),
                        config["protocol_source"]["path"],
                        config["features_source"]["path"],
                        config["p0_baseline_source"]["fit_dir"],
                        str(args.fit_dir), config["data"]["root"],
                        *PROTECTED_EXTRA_TARGETS]
        before = build_protection_snapshot(project_root, target_names)
        missing = missing_required_targets(before)
        if missing:
            raise ProtectionError(
                f"required protected inputs are missing: {missing}")

        fit_summary = load_json(fit_dir / "summary.json")
        if fit_summary.get("mode") != "fit" or \
                fit_summary.get("training_status") != "COMPLETED":
            raise ValueError(
                f"{fit_dir} is not a completed residual fit "
                f"(mode={fit_summary.get('mode')!r}, training_status="
                f"{fit_summary.get('training_status')!r})")

        # ---- the original cache/P0 admission chain (retained) ----
        import scripts.train_material_residual as residual_entry

        admission = residual_entry.run_admission(config, p0_fit_dir)
        protocol = admission["protocol"]
        authority = admission["authority"]
        manifest_samples = admission["manifest"]["samples"]
        if fit_summary["admission"]["protocol_sha256"] != \
                protocol["protocol_sha256"]:
            raise ValueError(
                "the fit's protocol digest does not match the frozen "
                "protocol")

        # ---- the expected fingerprint comes from the FROZEN sources ----
        expected_fingerprint = config["data"]["fingerprint"]
        if protocol["data"]["fingerprint"] != expected_fingerprint:
            raise ValueError(
                "protocol fingerprint does not match the config's data "
                "fingerprint")
        definition = load_json(features_dir / "feature_definition.json")
        if definition.get("data_fingerprint") != expected_fingerprint:
            raise ValueError(
                "feature definition fingerprint does not match the "
                "config's data fingerprint")

        # ---- V5 FULL ADMISSION before ANY per-sample statistics ----
        expected_parent = config["data"].get("parent_v4_fingerprint") \
            or V4_PARENT_FINGERPRINT
        v5_report = admit_dataset_for_protocol(
            data_root,
            pooling_mode="instance_mask",
            expected_fingerprint=expected_fingerprint,
            expected_parent_fingerprint=expected_parent,
            expected_mask_schema=MASK_SCHEMA_VERSION,
            expected_mask_resize_version=MASK_RESIZE_VERSION)
        if v5_report["audit_fingerprint"] != expected_fingerprint:
            raise ValueError(
                "V5 admission fingerprint does not match the expected "
                "frozen fingerprint")
        admission_record = {
            "gate": ("core.material_mask_dataset."
                     "admit_dataset_for_protocol (pooling="
                     "instance_mask; runs admit_mask_dataset including "
                     "verify_dataset_integrity, the mask content digest "
                     "and verify_mask_sources)"),
            "data_root": str(data_root.resolve()),
            "expected_fingerprint": expected_fingerprint,
            "actual_fingerprint": v5_report["audit_fingerprint"],
            "fingerprint_sources": {
                "config_data_fingerprint": expected_fingerprint,
                "protocol_data_fingerprint":
                    protocol["data"]["fingerprint"],
                "feature_definition_data_fingerprint":
                    definition.get("data_fingerprint"),
            },
            "v4_parent_expected": expected_parent,
            "v4_parent_actual": v5_report["v4_parent_fingerprint"],
            "mask_schema_version": v5_report["mask_schema_version"],
            "mask_resize_version": v5_report["mask_resize_version"],
            "verified_files": v5_report["verified_files"],
            "mask_content_sha256": v5_report["mask_content_sha256"],
            "label_source_sha256": v5_report["label_source_sha256"],
            "sample_mapping_sha256": v5_report["sample_mapping_sha256"],
            "source_label_files": v5_report["source_label_files"],
            "class_names": v5_report["class_names"],
            "splits": v5_report["splits"],
            "checks_performed": [
                "manifest/summary/file digest integrity "
                "(verify_dataset_integrity)",
                "mask invariants per sample (load + validate)",
                "mask content digest vs audit",
                "annotation source digest re-verification "
                "(verify_mask_sources)",
                "sample->frame/object/crop mapping digest",
                "V5 fingerprint vs frozen expected",
                "V4 parent fingerprint vs frozen expected",
                "mask schema/resize versions",
                "class order",
            ],
            "checks_not_performed": {
                "per_pixel_disparity_decomposition": "NOT_AVAILABLE "
                    "(disparity/consistency maps are not stored; "
                    "recomputation needs SGBM and is out of scope)",
                "right_frame_pixel_correspondence": "NOT_AVAILABLE "
                    "(needs the disparity warp)",
                "exposure_gain_gamma_sync": "NOT_AVAILABLE (not "
                    "recorded in the dataset)",
            },
        }
        output_dir.mkdir(parents=True)
        write_json(output_dir / "admission.json", admission_record)

        v5_rows = load_v5_rows(data_root / "dataset_manifest.csv")

        # ---- batch-1 changed samples (descriptive; saved files only) --
        p0_predictions: dict = {}
        variant_predictions: dict = {}
        have_variants = True
        for variant in ("RC", "RCS"):
            try:
                merged: dict = {}
                for fold_id in FOLD_IDS:
                    merged.update(load_variant_predictions(
                        fit_dir, fold_id, variant))
                variant_predictions[variant] = merged
            except FileNotFoundError:
                have_variants = False
        changed = None
        if have_variants:
            for fold_id in FOLD_IDS:
                outer_names = [
                    s["sample_name"] for s in manifest_samples
                    if authority[s["sample_name"]]["roles"][fold_id]
                    == "outer_eval"]
                values = admission["fold_results"][fold_id]["predictions"]
                if len(values) != len(outer_names):
                    raise ValueError(
                        f"fold {fold_id}: P0 prediction count does not "
                        "match the outer_eval rows")
                p0_predictions.update(zip(outer_names, values))
            changed = {
                "RCS_vs_R0": sorted(
                    name for name, prediction
                    in variant_predictions["RCS"].items()
                    if name in p0_predictions
                    and prediction != p0_predictions[name]),
                "RCS_vs_RC": sorted(
                    name for name, prediction
                    in variant_predictions["RCS"].items()
                    if name in variant_predictions["RC"]
                    and prediction != variant_predictions["RC"][name]),
            }

        # ---- per-sample pass (statistics probe-patchable) ----
        left_gray_cache: dict = {}
        records = []
        correspondence = {
            "checked_samples": 0,
            "gray_mismatches": [],
            "mask_count_mismatches": [],
            "missing_left_frames": [],
            "missing_right_frames": [],
            "right_path_unresolvable": [],
        }
        started_loop = time.time()
        for manifest_row in manifest_samples:
            name = manifest_row["sample_name"]
            v5_row = v5_rows.get(name)
            if v5_row is None:
                raise ValueError(
                    f"cache sample {name} is absent from the V5 manifest")
            sample = load_mask_sample(data_root / manifest_row["npz_path"])
            diagnostics = polar_region_quality.sample_region_diagnostics(
                sample)
            decision = polar_region_quality.evaluate_region_rule(sample)

            correspondence_record = None
            left_path = v5_row["left_path"]
            absolute_left = project_root / left_path
            if not absolute_left.is_file():
                correspondence["missing_left_frames"].append(name)
            else:
                if left_path not in left_gray_cache:
                    left_gray_cache[left_path] = read_gray(absolute_left)
                left_gray = left_gray_cache[left_path]
                crop_box = (int(v5_row["crop_y1"]), int(v5_row["crop_x1"]),
                            int(v5_row["crop_y2"]), int(v5_row["crop_x2"]))
                try:
                    correspondence_record = verify_sample_correspondence(
                        sample, left_gray, crop_box,
                        expected_mask_pixels=int(v5_row["mask_pixels"]))
                    correspondence["checked_samples"] += 1
                except ValueError as error:
                    if "gray" in str(error):
                        correspondence["gray_mismatches"].append(
                            f"{name}: {error}")
                    else:
                        correspondence["mask_count_mismatches"].append(
                            f"{name}: {error}")
                right_path = right_frame_path(left_path)
                if right_path is None:
                    correspondence["right_path_unresolvable"].append(name)
                elif not (project_root / right_path).is_file():
                    correspondence["missing_right_frames"].append(name)

            records.append({
                "sample_name": name,
                "group": manifest_row["group"],
                "split": manifest_row["split"],
                "class_id": manifest_row["class_id"],
                "fold_roles": dict(authority[name]["roles"]),
                "v5_stereo": {
                    "stereo_valid": v5_row["stereo_valid"],
                    "stereo_reason": v5_row["stereo_reason"],
                    "stereo_valid_ratio":
                        _clean(v5_row["stereo_valid_ratio"]),
                    "disparity": _clean(v5_row["disparity"]),
                },
                "diagnostics": diagnostics,
                "rule": decision,
                "correspondence": correspondence_record,
            })
        elapsed = time.time() - started_loop

        # ---- subsets, aggregates, drift summary ----
        subsets = build_subsets(records)
        aggregates = {
            label: {"role": spec["role"],
                    **aggregate_records(spec["records"])}
            for label, spec in subsets.items()
        }
        union_records = subsets["inner_train_all_rule_design"]["records"]
        drift_summary = compose_drift_summary(
            union_records,
            subset_label=("valid-S rows of the summary union of the "
                          "three folds' inner_train rows (not a "
                          "single-fold training set)"))
        missing_s_total = sum(1 for r in records
                              if not r["rule"]["has_valid_s"])
        fallback_total = sum(1 for r in records
                             if r["rule"]["fallback_used"])

        changed_context = None
        if changed is not None:
            by_name = {record["sample_name"]: record
                       for record in records}
            changed_context = {
                key: [{
                    "sample_name": name,
                    "group": by_name[name]["group"],
                    "region_pixels": by_name[name]["diagnostics"]
                    ["counts"]["region_pixels"],
                    "interior_valid_pixels": by_name[name]["diagnostics"]
                    ["counts"]["interior_valid_pixels"],
                    "rule_fallback": by_name[name]["rule"]
                    ["fallback_used"],
                    "interior_valid_ratio": by_name[name]["diagnostics"]
                    ["ratios"]["interior_valid_ratio"],
                    "band_valid_ratio": by_name[name]["diagnostics"]
                    ["ratios"]["band_valid_ratio"],
                } for name in names if name in by_name]
                for key, names in changed.items()}

        summary = {
            "batch": "2A corrective region diagnostics (read-only)",
            "generated_by": "scripts/diagnose_polar_region_quality.py",
            "inputs": {
                "config": str(args.config),
                "fit_dir": str(fit_dir),
                "protocol_sha256": protocol["protocol_sha256"],
                "data_root": str(data_root.resolve()),
                "features_path": config["features_source"]["path"],
            },
            "frozen_definitions": {
                "interior_definition": INTERIOR_DEFINITION,
                "band_definition": BAND_DEFINITION,
                "region_rule_id": REGION_RULE_ID,
                "s_stat_keys": list(POLAR_STAT_KEYS),
            },
            "missing_state": {
                "samples": len(records),
                "missing_s_samples": missing_s_total,
                "rule_fallback_used": fallback_total,
                "note": (
                    "rule_fallback_used=0 does NOT mean every sample has "
                    "a valid S; missing_s_samples counts dev-pool samples "
                    "with no valid S at all under the original definition "
                    "(they keep has_valid_s=false and are exact P0 "
                    "fallbacks)"),
            },
            "perturbation": {**drift_summary, "interpretation_limit": True},
            "unverified_or_not_available": {
                "left_right_pixel_correspondence_error": "UNVERIFIED",
                "disparity_and_consistency_failure_decomposition":
                    "NOT_AVAILABLE",
                "q_change_under_real_left_right_misalignment":
                    "NOT_AVAILABLE (no SGBM, no recompute)",
                "exposure_gain_gamma_sync": "NOT_AVAILABLE",
                "erosion_rule_effect_on_classification_metrics":
                    "UNVERIFIED (never measured)",
                "left_image_saturation_and_dark_pixel_statistics": (
                    "descriptive of the LEFT crop only; the existing "
                    "valid gate does not prove registration, exposure or "
                    "sync errors are absent, and the between-sample "
                    "spread is not itself a material-discriminability "
                    "measure"),
            },
            "changed_samples_descriptive": {
                "available": changed is not None,
                "counts": ({key: len(value) for key, value
                            in changed.items()} if changed else None),
                "note": ("outer_eval changes of the reviewed fit; "
                         "descriptive reference only, never used to fit "
                         "rules"),
                "samples": changed_context,
            },
            "correspondence": correspondence,
            "elapsed_sample_loop_s": _clean(elapsed),
        }
        write_jsonl(output_dir / "per_sample_diagnostics.jsonl", records)
        write_json(output_dir / "aggregates.json", aggregates)
        write_json(output_dir / "drift_summary.json", drift_summary)
        write_json(output_dir / "diagnostics_summary.json", summary)

        # ---- AFTER protection: independent re-computation and compare --
        after = build_protection_snapshot(project_root, target_names)
        comparison = compare_protection_snapshots(before, after)
        write_json(output_dir / "protection_report.json", {
            "target_names": target_names,
            "before_files": before,
            "after_files": after,
            "comparison": comparison,
            "intentionally_modified": list(INTENTIONALLY_MODIFIED),
            "historical_protection_evidence":
                HISTORICAL_PROTECTION_EVIDENCE,
            "scope_note": (
                "this report proves protection only for the window "
                "between the two snapshots of THIS run; it does not "
                "reconstruct the historical pre-execution state"),
        })

        problems = (correspondence["gray_mismatches"]
                    + correspondence["mask_count_mismatches"]
                    + correspondence["missing_left_frames"]
                    + correspondence["missing_right_frames"])
        if problems:
            print(f"REFUSED: {len(problems)} correspondence problems; "
                  "report written", file=sys.stderr)
            return EXIT_REFUSED
        print(f"OK: {len(records)} samples diagnosed in "
              f"{time.time() - started:.1f}s; protection passed "
              f"({comparison['targets']} targets, "
              f"{comparison['files']} files)", file=sys.stderr)
        return EXIT_OK
    except ProtectionError as error:
        print(f"REFUSED: ProtectionError: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
