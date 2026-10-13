"""material_polar_complementarity_v1 — the zero-training diagnostic entry.

Consumes ONLY the third batch's saved real fit (outer_eval prediction
rows), its independent verification report and the verified fix-version
feature cache. Never fits a model or a scaler, never runs a classifier
forward; the only pixel computation is the production stereo/q recipe
re-run for the few frames a panel needs (selected-instance crops only,
bands always from the frame's FULL GT instance set).

Exit codes: 0 COMPLETED, 1 REFUSED (identity/contract/count conflicts),
2 PARTIAL (deadline crossed; evidence kept, never a success).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.material_polar_complementarity import (  # noqa: E402
    ARMS,
    EXPERIMENT,
    EXPECTED_COUNTS,
    FOLD_IDS,
    build_conditional_analysis,
    MAX_CONTROLS,
    MAX_PANELS,
    PREDICTION_FILES,
    PREDICTION_SET_SHA256,
    TRAIN_SEEDS,
    VARIANTS,
    ComplementarityRefused,
    assign_bins,
    build_ledger,
    conditional_tables,
    derive_counts,
    fold_bin_edges,
    prediction_set_digest,
    q1_fish_pair_common_support,
    select_controls,
    sha256_of_file,
    subset_layered_analysis,
    verify_expected_counts,
    verify_panel_coverage,
    verify_saved_binning,
    verify_saved_ledger,
)
from core.material_residual_scale import (  # noqa: E402
    DeadlineExceeded,
    DeadlineWatch,
    ScaleRefused,
    UPSAMPLE_FIX_VERSION,
    load_scale_config,
    matcher_config_of,
    verify_scale_cache_markers,
)
from core.material_mask_dataset import (  # noqa: E402
    MaskFusionSampleArrays,
    load_mask_sample,
)
from core.polar_local_features import (  # noqa: E402
    build_local_control_features,
    build_local_polar_stats,
    region_records,
)
from core.gt_geometry import build_gt_geometry  # noqa: E402
from core.stereo_matching import (  # noqa: E402
    StereoMatcher,
    StereoMatcherConfig,
    build_horizontal_bands,
)
from core.polar_compute import compute_polar_features  # noqa: E402
from core.fusion_dataset import (  # noqa: E402
    QUALITY_VECTOR_KEYS,
    build_quality_vector,
    quality_vector_from_result,
)
from core.material_probe import authoritative_sample_index  # noqa: E402
from scripts.eval_material_upsample_impact import (  # noqa: E402
    IdentityError,
    SourceRegistry,
    dump_json,
    dump_jsonl,
    is_link_or_reparse,
    verify_anchor,
)
from scripts.prepare_cls_fusion_dataset import load_gray  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_PARTIAL = 2

DIAGNOSTIC_ROOT = "analysis/runs/material_polar_complementarity_diagnostic_v1"
FIT_DIR = ("analysis/runs/material_residual_scale_v1/"
           "fit_cdd17d10ea5b49f4b99c06841ec10320")
FIT_DIR_VERIF_DIR = FIT_DIR + "_verification"
CACHE_DIR = ("analysis/runs/material_residual_scale_v1/"
             "20261011T074823Z_e36bf75e/cache_build/features")
PINNED_CONFIG_REL = ("configs/experiments/"
                     "material_residual_scale_v1.yaml")
# The ONLY prior diagnostic directory the postprocess mode may consume.
PRIOR_DIAGNOSTIC_REL = (DIAGNOSTIC_ROOT + "/"
                        "20261011T160738Z_fd30770f/diagnostic")

# The frozen admission table: digests recorded by the third-batch review
# for the files this diagnostic consumes. They are HISTORICAL
# expectations — a mismatch refuses the run, it is never updated.
PINNED_INPUTS = {
    "configs/experiments/material_residual_scale_v1.yaml":
        "697bf9f14b5dcd7475ee777bf7b543ce3c11ff6e41adb00fd716919aedcf9a09",
    f"{FIT_DIR}/summary.json":
        "43828eca275ca817252ec56ebed954731355994aa73a148953cd4a262ddc3291",
    f"{FIT_DIR}/admission.json":
        "6301293343d87d0a45ba9dd60227c4578aa89e2f2198185b2ccfaf21e3e7ff42",
    f"{FIT_DIR}/run_status.json":
        "daeeb8e6372b8fe07de94df9ee195b38be49c68c96dd82ff0308662ea44d8809",
    f"{FIT_DIR}_verification/verification_report.json":
        "e8c135ea0915822e4b7da8f03f63902defa72302000869aff74e3c1ab785f5e8",
    "analysis/runs/material_residual_scale_v1/20261011T074823Z_e36bf75e/"
    "cache_build/features/cache_binding.json":
        "b009717913c5fa23a1a5294bf7fa66fda3aa13ce9764dcaf35f19dbd1f432872",
    "analysis/runs/material_residual_scale_v1/20261011T074823Z_e36bf75e/"
    "cache_build/features/feature_cache.npz":
        "7e9c68ea8c9c57d2bd391a0465cbd8ead32450cfa7f8a731b7160e35615f7f15",
    # the cache-build run's own source manifest: the verified historical
    # record of every frame image / label / V5 npz the fix rebuild
    # consumed. Its digest was recorded once at diagnostic implementation
    # time from that run's published evidence (the same anchor pattern as
    # the seven review-pinned files above) and is never re-derived.
    "analysis/runs/material_residual_scale_v1/20261011T074823Z_e36bf75e/"
    "cache_build/source_manifest.json":
        "c7a4c7dd6cba25a3aad1a6e0e46d10232eb1f8baaf2abf40e7407a00da1ae499",
}

# The third-batch facts archived here (checked against the fit summary;
# from the pre-registered fold/seed aggregation, never pooled metrics).
THIRD_BATCH_FACTS = {
    "macro_f1_mean_of_seeds": {
        ("fixed", "R0"): 0.9327834914858463,
        ("fixed", "RC"): 0.931763316947873,
        ("fixed", "RCS"): 0.9332926504493860,
        ("scaled", "RC"): 0.9294368431690824,
        ("scaled", "RCS"): 0.9316354979669029,
    },
    "gate_pass": {"gate1_scaled_rcs_vs_r0_rc": False,
                  "gate2_scaled_vs_fixed_rcs": False,
                  "gate3_two_seeds_net_vs_both": False,
                  "gate4_fold_stability": False,
                  "gate5_class_recall_floor": True,
                  "gate6_group_breadth": True},
}

# Zero-training call counters: the paths this entry must NEVER take.
ZERO_CALL_KEYS = ("new_model_fits", "new_scaler_fits",
                  "classifier_forwards", "optimizer_steps")

# The postprocess mode consumes ONLY the prior diagnostic's saved
# artifacts. These digests come from the review of that run — a mismatch
# refuses the postprocess, never updates an expectation.
PINNED_PRIOR_ARTIFACTS = {
    "instance_ledger.json":
        "c2f44ae84c37d0e8e476c449b27ec56ff8b98f2d50747689322974576267c668",
    "binning.json":
        "4762423acd9c4e4b14cc3c075b168148450f1720be376cdd9a9fb5aa86d32ff0",
    "counts.json":
        "87e2c420df02ff2c9cce7f6c6f0e49aa6064bd5abf8484922475d5d332789f5f",
    "selection_manifest.json":
        "acac1bac4505b84453e7d4204715f181febeebac42c838a0f7002c58520ea248",
    "panels_manifest.json":
        "e2fcefff2df81ef25bb7c1a6cf6c4b7c00214c9400221895b0f2acf053ede551",
    "source_manifest.json":
        "312c6ee2f5e0753a6c2f0e816a78a114e736edeed852c2425116f4d26ab3471b",
    "run_status.json":
        "829c1f180a8b2bcfafa64ee89635d74f258550e4bcbb829c7503efeab224f32f",
}

# The postprocess re-derives the counts and cross-checks them against
# both the saved counts.json and these review-confirmed facts.
POSTPROCESS_EXPECTED = {
    "unique_instances": 2446,
    "r0_errors": 150,
    "r0_errors_with_s": 145,
    "corrected_union": 12,
    "broken_union": 17,
    "changed_union": 29,
    "corrected_pairs": 32,
    "broken_pairs": 34,
    "instance_model_pairs_evaluated": 29352,
    "rcs_vs_r0_comparison_pairs": 14676,
}


def _zero_calls() -> dict:
    return {key: 0 for key in ZERO_CALL_KEYS}


# ---------------------------------------------------------------------------
# Admission


def _validate_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise ScaleRefused(
            f"output directory already exists: {output_dir}")
    root = (_PROJECT_ROOT / DIAGNOSTIC_ROOT).resolve()
    resolved = output_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise ScaleRefused(
            f"output directory {resolved} is not inside {root}")
    probe = resolved
    while probe != _PROJECT_ROOT.resolve():
        if is_link_or_reparse(probe):
            raise ScaleRefused(
                f"link or reparse point on the output path: {probe}")
        probe = probe.parent
    if is_link_or_reparse(output_dir.parent):
        raise ScaleRefused(
            "the output parent is a link/reparse point")


def _register_pinned(registry: SourceRegistry, ctx: dict) -> dict:
    """Hash-verify every pinned input and register it BEFORE any other
    read. The binding is the review-recorded digest itself."""
    for rel, expected in PINNED_INPUTS.items():
        path = _PROJECT_ROOT / rel
        if not path.is_file():
            raise ScaleRefused(f"pinned input missing: {path}")
        actual = sha256_of_file(path)
        if actual != expected:
            raise ScaleRefused(
                f"{rel}: sha256 {actual} != the review-pinned {expected}; "
                "refusing (expectations are never updated)")
    registered = {}
    for rel, expected in PINNED_INPUTS.items():
        verify_anchor(_PROJECT_ROOT / rel, expected, registry,
                      f"anchor:diagnostic_input")
        registered[rel] = expected
    ctx["log"](f"pinned inputs verified and registered: "
               f"{len(registered)}")
    return registered


def _expand_manifest_bindings(registry: SourceRegistry, manifest_path: Path,
                              source: str) -> int:
    """A run's own source manifest is a NEW source for this diagnostic:
    it is snapshotted before its first read, then every row's recorded
    at-registration digest becomes a historical expectation for that
    run's consumed files. Never backfilled."""
    registry.register(manifest_path, source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    added = 0
    for item in manifest.get("items", []):
        digest = item.get("sha256_at_registration")
        rel = item.get("path")
        if digest and rel:
            binding = {"source": source,
                       "expected_sha256": str(digest)}
            existing = registry.bindings.get(rel)
            if existing is None:
                registry.bindings[rel] = binding
                added += 1
    return added


def run_admission(ctx: dict, args) -> dict:
    registry = ctx["registry"]
    ctx["stage"]("admission", "STARTED")
    # identity FIRST: every pinned input (the config among them) is
    # hash-verified and registered BEFORE the config parser runs — a
    # substitute or drifted file never reaches load_scale_config
    pinned = _register_pinned(registry, ctx)
    cfg = load_scale_config(Path(args.config))

    # the frozen global cache: binding first (already verified above),
    # then its internal file bindings become historical expectations
    feature_dir = _PROJECT_ROOT / CACHE_DIR
    binding = json.loads((feature_dir / "cache_binding.json").read_text(
        encoding="utf-8"))
    files_binding = binding.get("files") or {}
    for name in ("feature_manifest.json", "feature_definition.json"):
        rel = (feature_dir / name).relative_to(_PROJECT_ROOT).as_posix()
        record = files_binding.get(name)
        if not record:
            raise ScaleRefused(
                f"cache binding does not cover {name}")
        actual = sha256_of_file(feature_dir / name)
        if actual != record["sha256"]:
            raise ScaleRefused(
                f"{name}: sha256 {actual} != the cache binding's "
                f"{record['sha256']}")
        registry.bindings.setdefault(rel, {
            "source": "global_cache_binding",
            "expected_sha256": record["sha256"]})
        registry.register(feature_dir / name, f"global_cache:{name}")
    for name, record in files_binding.items():
        rel = (feature_dir / name).relative_to(_PROJECT_ROOT).as_posix()
        if rel not in registry.bindings:
            registry.bindings[rel] = {
                "source": "global_cache_binding",
                "expected_sha256": record["sha256"]}

    # the fit's own source manifest expands into historical expectations
    # for everything the fit consumed; the cache-build run's manifest
    # (when pinned) covers the frame images / labels / V5 npz the fix
    # rebuild consumed
    expanded = _expand_manifest_bindings(
        registry, _PROJECT_ROOT / FIT_DIR / "source_manifest.json",
        "fit_source_manifest")
    cache_build_rel = ("analysis/runs/material_residual_scale_v1/"
                       "20261011T074823Z_e36bf75e/cache_build/"
                       "source_manifest.json")
    cache_build_expanded = 0
    if cache_build_rel in pinned:
        cache_build_expanded = _expand_manifest_bindings(
            registry, _PROJECT_ROOT / cache_build_rel,
            "cache_build_source_manifest")
    ctx["log"](f"historical expectations: fit_source_manifest +{expanded}, "
               f"cache_build_source_manifest +{cache_build_expanded}, "
               "global_cache_binding from the frozen cache binding")

    # the 39 prediction files, registered before the set-digest read
    for rel in PREDICTION_FILES:
        registry.register(_PROJECT_ROOT / rel, f"fit_predictions:{rel}")
    digest = prediction_set_digest(_PROJECT_ROOT, PREDICTION_FILES)
    if digest != PREDICTION_SET_SHA256:
        raise ScaleRefused(
            f"the 39-file prediction set digest {digest} != the "
            f"review-pinned {PREDICTION_SET_SHA256}")
    ctx["stage"]("prediction_set_digest", "COMPLETED",
                 files=len(PREDICTION_FILES))

    # third-batch facts: the fit must be the verified COMPLETED run
    fit_status = json.loads((_PROJECT_ROOT / FIT_DIR /
                             "run_status.json").read_text(
        encoding="utf-8"))
    if fit_status.get("overall_status") != "COMPLETED":
        raise ScaleRefused(
            "the pinned fit run_status is not COMPLETED")
    verification = json.loads(
        (_PROJECT_ROOT / FIT_DIR_VERIF_DIR /
         "verification_report.json").read_text(encoding="utf-8"))
    if verification.get("problem_count") != 0:
        raise ScaleRefused(
            "the independent verification report does not have "
            f"problem_count == 0 ({verification.get('problem_count')})")
    fit_summary = json.loads((_PROJECT_ROOT / FIT_DIR /
                              "summary.json").read_text(encoding="utf-8"))
    for gate, expected_pass in THIRD_BATCH_FACTS["gate_pass"].items():
        actual = bool((fit_summary.get("gates") or {}).get(
            gate, {}).get("pass"))
        if actual != expected_pass:
            raise ScaleRefused(
                f"gate {gate}: pass={actual} != the archived fact "
                f"{expected_pass}")
    for (arm, variant), expected in THIRD_BATCH_FACTS[
            "macro_f1_mean_of_seeds"].items():
        actual = fit_summary["arms"][arm][variant][
            "macro_f1_mean_of_seeds"]
        if float(actual) != expected:
            raise ScaleRefused(
                f"{arm}/{variant} macro_f1_mean_of_seeds {actual!r} != "
                f"the archived fact {expected!r}")

    # protocol authority + cache contract
    protocol_path = _PROJECT_ROOT / cfg["protocol_source"]["path"]
    registry.register(protocol_path, "protocol_file")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    authority = authoritative_sample_index(protocol)
    markers = verify_scale_cache_markers(feature_dir, cfg)
    registry.register(feature_dir / "feature_definition.json",
                      "cache_definition_marker")
    npz = np.load(feature_dir / "feature_cache.npz")
    ctx["stage"]("admission", "COMPLETED",
                 markers=markers.get("cache_kind"),
                 protocol="verified")

    manifest = json.loads((feature_dir / "feature_manifest.json")
                          .read_text(encoding="utf-8"))
    return {"cfg": cfg, "protocol": protocol, "authority": authority,
            "npz": npz, "manifest": manifest["samples"],
            "feature_dir": feature_dir}


# ---------------------------------------------------------------------------
# Selected-frame rebuild (the production fix recipe, selected crops only)


def rebuild_frame(ctx, cfg: dict, matcher, stem: str, rows: list,
                  pair_manifest: dict, selected_names: set) -> dict:
    """One frame's production fix recipe. Bands come from the frame's
    FULL GT instance set; only selected instances keep diagnostics."""
    registry = ctx["registry"]
    first = rows[0]
    left_path = _PROJECT_ROOT / first["left_path"]
    right_path = _PROJECT_ROOT / pair_manifest[stem]["right_path"]
    label_path = _PROJECT_ROOT / first["label_path"]
    registry.register(left_path, f"diag_image_left:{stem}")
    registry.register(right_path, f"diag_image_right:{stem}")
    registry.register(label_path, f"diag_label:{stem}")
    left = load_gray(left_path)
    right = load_gray(right_path)
    height, width = left.shape[:2]
    geometry = build_gt_geometry(label_path, width, height,
                                 padding=int(cfg["fix_inputs"]["crop_pad"]))
    if len(geometry) != len(rows):
        raise ScaleRefused(
            f"frame {stem}: {len(geometry)} GT instances != {len(rows)} "
            "V5 rows")
    bands = build_horizontal_bands(
        [item.band_bbox for item in geometry], image_height=height,
        vertical_margin=int(cfg["fix_inputs"]["band_margin"]))
    result = matcher.compute_bands(
        left, right, bands,
        full_image_threshold=float(cfg["fix_inputs"]["full_image_threshold"]))
    by_name = {f"{stem}_obj{item.object_index:03d}": item
               for item in geometry}
    v5_rows = {row["sample_name"]: row for row in rows}
    diagnostics = {}
    for name in sorted(selected_names & set(by_name)):
        item = by_name[name]
        v5_row = v5_rows[name]
        v5_path = _PROJECT_ROOT / cfg["data"]["root"] / v5_row["npz_path"]
        registry.register(v5_path, f"diag_v5_sample:{name}")
        v5_sample = load_mask_sample(v5_path)
        if v5_sample.class_id != item.class_id:
            raise ScaleRefused(f"{name}: V5 class != GT class")
        window = (int(v5_row["crop_x1"]), int(v5_row["crop_y1"]),
                  int(v5_row["crop_x2"]), int(v5_row["crop_y2"]))
        if window != item.crop_window:
            raise ScaleRefused(
                f"{name}: V5 crop window != GT geometry")
        x1, y1, x2, y2 = window
        if not np.array_equal(v5_sample.gray, left[y1:y2, x1:x2]):
            raise ScaleRefused(
                f"{name}: the V5 gray crop is not the rectified left crop")
        if not np.array_equal(v5_sample.instance_mask,
                              item.mask[y1:y2, x1:x2]):
            raise ScaleRefused(
                f"{name}: the V5 instance mask is not the GT raster")
        mask = item.mask
        stats = matcher.instance_stats(result, mask, item.object_index)
        if stats.valid:
            polar_result = compute_polar_features(
                left, right, result.disparity, object_mask=mask,
                disparity_valid=result.valid)
            quality, components = quality_vector_from_result(
                polar_result, mask)
            crop_signed = polar_result.signed_q[y1:y2, x1:x2]
            crop_abs = polar_result.abs_q[y1:y2, x1:x2]
            crop_valid = polar_result.valid_mask[y1:y2, x1:x2].astype(
                np.uint8)
            if int(crop_valid.sum()) != \
                    int(polar_result.valid_mask.sum()):
                raise ScaleRefused(
                    f"{name}: the crop window cuts valid polar pixels")
            stereo_reason = "ok"
        else:
            components = {key: 0.0 for key in QUALITY_VECTOR_KEYS}
            quality = build_quality_vector(**components)
            crop_signed = np.zeros((y2 - y1, x2 - x1), np.float32)
            crop_abs = np.zeros((y2 - y1, x2 - x1), np.float32)
            crop_valid = np.zeros((y2 - y1, x2 - x1), np.uint8)
            stereo_reason = str(stats.reason)
        fix_arrays = MaskFusionSampleArrays(
            gray=v5_sample.gray, signed_q=crop_signed, abs_q=crop_abs,
            valid=crop_valid,
            quality=np.asarray(quality, dtype=np.float32),
            class_id=int(item.class_id),
            instance_mask=v5_sample.instance_mask)
        values, blocks = build_local_polar_stats(fix_arrays)
        diagnostics[name] = {
            "sample_name": name, "frame": stem,
            "crop_window": list(window),
            "gray_crop": v5_sample.gray,
            "signed_q_crop": crop_signed, "abs_q_crop": crop_abs,
            "valid_crop": crop_valid.astype(bool),
            "instance_mask_crop": v5_sample.instance_mask,
            "s_features": values, "blocks": blocks,
            "c_features": build_local_control_features(fix_arrays),
            "region": {**region_records(fix_arrays),
                       "blocks": {k: bool(v)
                                  for k, v in blocks.items()}},
            "fix_provenance": {
                "upsample_version": UPSAMPLE_FIX_VERSION,
                "stereo_valid": bool(stats.valid),
                "stereo_reason": stereo_reason,
                "quality_components": {k: float(v) for k, v
                                       in components.items()},
            },
        }
    return {"diagnostics": diagnostics, "geometry_count": len(geometry),
            "band_count": len(bands)}


def compare_rebuilt_to_cache(name: str, index: int, rebuilt: dict,
                             cache_arrays: dict, manifest_samples: list) \
        -> dict:
    """Bitwise comparison of the rebuilt C14/S16/region against the
    verified cache row; NaN positions must coincide exactly."""
    problems = []
    cached_c = np.asarray(cache_arrays["c_features"][index],
                          dtype=np.float32)
    cached_s = np.asarray(cache_arrays["s_features"][index],
                          dtype=np.float32)
    rebuilt_c = np.asarray(rebuilt["c_features"], dtype=np.float32)
    rebuilt_s = np.asarray(rebuilt["s_features"], dtype=np.float32)
    if not np.array_equal(rebuilt_c.view(np.uint32),
                          cached_c.view(np.uint32)):
        differing = [i for i in range(14)
                     if rebuilt_c[i].tobytes() != cached_c[i].tobytes()]
        problems.append(f"{name}: C14 bitwise mismatch at {differing}")
    nan_positions_differ = int(
        (np.isnan(rebuilt_s) != np.isnan(cached_s)).sum())
    if nan_positions_differ:
        problems.append(
            f"{name}: S16 NaN positions differ at {nan_positions_differ} "
            "cells")
    finite = ~np.isnan(rebuilt_s) & ~np.isnan(cached_s)
    if not np.array_equal(rebuilt_s[finite].view(np.uint32),
                          cached_s[finite].view(np.uint32)):
        problems.append(f"{name}: S16 finite values differ bitwise")
    sample = manifest_samples[index]
    numeric_region_keys = ("mask_pixels", "interior_pixels", "band_pixels",
                           "interior_valid_pixels", "band_valid_pixels",
                           "region_pixels")
    for key in numeric_region_keys:
        if int(sample[key]) != int(rebuilt["region"][key]):
            problems.append(
                f"{name}: region.{key} rebuilt "
                f"{rebuilt['region'][key]} != cache {sample[key]}")
    if rebuilt["region"].get("region_rule") != sample.get("region_rule"):
        problems.append(f"{name}: the region rule differs from the cache")
    if bool(rebuilt["s_valid"] if "s_valid" in rebuilt
            else not np.isnan(rebuilt_s).all()) != \
            bool(cache_arrays["s_valid"][index]):
        problems.append(f"{name}: s_valid differs from the cache")
    return {"sample_name": name, "problems": problems,
            "bitwise_c14": not problems}


# ---------------------------------------------------------------------------
# Panels


def _masked_image(array: np.ndarray, valid: np.ndarray):
    import numpy.ma as ma

    return ma.masked_array(np.asarray(array, dtype=np.float64),
                           mask=~np.asarray(valid, dtype=bool))


def render_panel(panel_dir: Path, panel: dict, diag: dict, ledger_row: dict,
                 class_names) -> dict:
    """One PNG: fixed display ranges everywhere, invalid areas visibly
    masked (stored zeros are never shown as real low q)."""
    gray = diag["gray_crop"]
    from core.polar_region_quality import interior_and_band_masks

    interior, band = interior_and_band_masks(diag["instance_mask_crop"])
    fig = plt.figure(figsize=(13.5, 7.2))
    axes = fig.subplots(2, 4)
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("gray crop [0,255]")
    axes[0, 1].imshow(_masked_image(diag["signed_q_crop"],
                                    diag["valid_crop"]),
                      cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 1].set_title("fix signed_q [-1,1]; gray=invalid")
    axes[0, 2].imshow(_masked_image(diag["abs_q_crop"],
                                    diag["valid_crop"]),
                      cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title("fix abs_q [0,1]; gray=invalid")
    axes[0, 3].imshow(diag["valid_crop"], cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title("polar_valid (fix)")
    axes[1, 0].imshow(diag["instance_mask_crop"], cmap="gray",
                      vmin=0, vmax=1)
    axes[1, 0].set_title("GT instance mask")
    overlay = np.zeros((*interior.shape, 3), dtype=np.float32)
    overlay[interior] = (0.1, 0.65, 0.15)
    overlay[band] = (0.85, 0.55, 0.1)
    axes[1, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].imshow(overlay, alpha=0.45)
    axes[1, 1].set_title("interior(green)/band(orange)")
    s_text = "\n".join(
        f"S[{i:02d}] {('nan' if v is None else format(v, '+.4f'))}"
        for i, v in enumerate(ledger_row["s_features"]))
    axes[1, 2].axis(False)
    axes[1, 2].text(0.0, 1.0,
                    f"S16 (interior 0-7 / band 8-15)\n{s_text}",
                    fontsize=6, family="monospace",
                    verticalalignment="top")
    axes[1, 3].axis(False)
    lines = [f"R0 pred={ledger_row['r0']['prediction']} "
             f"{'OK' if ledger_row['r0']['correct'] else 'ERR'} "
             f"margin={ledger_row['r0']['raw_logit_margin']:.3f}"]
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            for variant in VARIANTS:
                path = ledger_row["paths"][f"{arm}|{seed}|{variant}"]
                mark = "OK" if path["correct"] else "ERR"
                tag = ""
                if variant == "RCS":
                    vs_r0 = ledger_row["rcs_vs_r0"][f"{arm}|{seed}"]
                    tag = (" FIX" if vs_r0["fixed"]
                           else " BRK" if vs_r0["broke"] else "")
                lines.append(f"{arm[:3]}/{seed}/{variant}: "
                             f"p={path['prediction']} {mark}{tag}")
    axes[1, 3].text(0.0, 1.0, "\n".join(lines), fontsize=6,
                    family="monospace", verticalalignment="top")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    kind = panel["kind"]
    fig.suptitle(
        f"[{kind}] {panel['panel_id']} | {panel['sample_name']} | "
        f"{ledger_row['class_name']} | group {ledger_row['group']} | "
        f"{ledger_row['outer_fold']} | "
        f"upsample={UPSAMPLE_FIX_VERSION}", fontsize=9)
    out = panel_dir / f"{panel['panel_id']}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"file": out.name, "bytes": out.stat().st_size}


# ---------------------------------------------------------------------------
# Report


def suggest_route(tables: dict, stats: dict) -> dict:
    """The honest route statement. There is NO automatic route threshold:
    neither pooled IQR separations nor a corrected/broken imbalance can
    promote a route. The reviewed conclusion for this data is C — the
    saved outcomes do not establish polarization benefit, spatial
    complementarity or a measurement-side dominant explanation. Route B
    is listed only as a to-be-verified re-collection/calibration
    suggestion and route A only as a possible future hypothesis."""
    return {
        "route": "C",
        "reason": (
            "现有小样本条件比较不足以确认跨组稳定线索。尚未建立偏振净收"
            "益，也未确认空间互补性或测量侧主导解释。保存结果的既定事实"
            "是：RCS 曾纠正（12 个唯一实例）不多于曾破坏（17）。首选路"
            "线 C 为当前审查结论——停止当前数据上的模型扩张，保留灰度基"
            "线；保留逐格观察，不自动升级路线。B（补采集/标定）只是待验"
            "证的建议，A 只作为将来可能讨论的假设。"),
    }


def write_report(report_path: Path, facts: dict, tables: dict,
                 route: dict, notes: list[str],
                 counts: dict, rebuild_summary: dict) -> None:
    lines = [
        "# material_polar_complementarity_v1 — 一页路线建议", "",
        "科学状态：**尚未建立偏振净收益**。本诊断不训练、不拟合 scaler、"
        "不做分类前向；全部数字来自已核验的第三批保存结果与冻结修复缓存。",
        "",
        "## 已归档的第三批事实",
        f"- 正式拟合 COMPLETED；独立核验 problem_count=0；Gate1-4 未通过，"
        f"Gate5/6 通过；按预注册决定停止本轮模型扩展。",
        f"- 原口径 Macro-F1 均值：R0 "
        f"{facts['macro_f1']['R0']:.6f}、fixed RC "
        f"{facts['macro_f1']['fixed RC']:.6f}、fixed RCS "
        f"{facts['macro_f1']['fixed RCS']:.6f}、scaled RC "
        f"{facts['macro_f1']['scaled RC']:.6f}、scaled RCS "
        f"{facts['macro_f1']['scaled RCS']:.6f}（折/种子汇总口径）。",
        f"- scaled RCS 相对 R0 −0.114799pp、相对 scaled RC +0.219865pp、"
        "相对 fixed RCS −0.165715pp。",
        "",
        "## 本次唯一实例台账（全部由数据推导并与预注册期望核对）",
        f"- 唯一实例 {counts['unique_instances']}；R0 错误 "
        f"{counts['r0_errors']}（其中 {counts['r0_errors_with_s']} 有 "
        "S）；RCS 曾纠正（并集）"
        f"{counts['corrected_union']}、曾破坏 {counts['broken_union']}、"
        f"正确性变化并集 {counts['changed_union']}。",
        f"- 三组主要混淆（唯一 R0 错误）：54（42+12）/ 52（24+28）/"
        f"28（3+25）；其余错误 {counts['other_errors']}。",
        f"- 曾纠正实例的原混淆：real_fish→plastic_fish 8、"
        "plastic_fish→real_fish 3、plastic_fish→plastic_submarine 1；"
        "两种潜艇 0。",
        f"- 实例—模型对（12 路径 × 实例）：评估 "
        f"{counts['instance_model_pairs_evaluated']}，纠正对 "
        f"{counts['corrected_pairs']}，破坏对 {counts['broken_pairs']}"
        "（跨种子累计，不当作独立样本）。",
        "",
        "## 条件分布要点（唯一实例、冻结分箱）",
        "- 见 conditional_tables.json：子集概览 + 真实分层单元"
        "（真实类别 / ×折 / ×组 / ×折×联合分箱 / ×组×折×联合分箱）与同组"
        "比较；每折 C0/C3 内层训练三分位边界及来源在 binning.json。",
        "- low_support 类标记只是实现层展示提示，不是预注册门槛，也不是"
        "路线选择器。",
        "",
        "## 分层观察与同组比较",
        "- 逐组、逐折、逐联合分箱的支持数、有效 n、中位数与分位差全部落"
        "盘于 conditional_tables.json 与 same_group_comparisons.json；"
        "不可比较的单元显式标注 support=0 与统计 null（不制造零测量）。",
        "- 分箱比较保留折语境：不同折的同号分箱不代表相同绝对亮度。",
        "",
        "## 首选路线",
        f"**路线 {route['route']}** — {route['reason']}",
        ROUTE_DETAIL[route["route"]],
        "",
        "## 物理与统计限制",
        "- 当前 q 是双视点 0°/90° 配准差分代理，不是完整 DoLP/AoLP；"
        "曝光、gain/gamma、同步历史元数据不足。",
        "- physical_object_id 不可确认；采集组不等同于独立实物。",
        "- 内容依赖、类差异或可视模式不证明材质因果。",
        "- 现有外层结果已用于设计本诊断；同池后续实验属探索性开发，"
        "不构成新的盲测确认。",
        "",
        "## 证据索引",
        f"- 台账：instance_ledger.json（{counts['unique_instances']} 行）"
        "与 instance_ledger_brief.jsonl；计数 counts.json；选例 "
        "selection_manifest.json；面板 panels/ 与 panels_manifest.json、"
        "index.html；条件表 conditional_tables.json 与 "
        "same_group_comparisons.json；分箱来源 binning.json；来源清单 "
        "source_manifest.json。",
        f"- 本地重建：{rebuild_summary['frames']} 帧、"
        f"{rebuild_summary['instances']} 个选中实例，与缓存逐项核对 "
        f"{'一致' if not rebuild_summary['problems'] else '不一致'}。",
    ]
    if notes:
        lines += ["", "## 选例备注"] + [f"- {n}" for n in notes]
    lines += [
        "",
        "## 面板观察状态",
        "- 每张面板的自动观察（重建来源、块有效性）已记录于 "
        "panels_manifest.json（记录者：自动规则）；**运行时没有任何"
        "人工看图复核**，人工观察字段保留为未知，等待审查者填写。",
        "- 疑似标签问题只登记、不修改标签、不删除实例、不重算"
        "“清洗后收益”。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ROUTE_DETAIL = {
    "A": ("后续（仅假设，不实现、不训练、不宣布第四批）：提出一个空间表征"
          "对照假设——例如把 interior/band 分区统计替换为保留空间排列的"
          "低维描述（分块 q 直方图/方向梯度），与同折同种子 RC 对照；"
          "需要相同的灰度/质量控制与组隔离验证，由未来授权的新实验执行。"),
    "B": ("具体补采集/标定需求（本次不采集、不改测量算法）："
          "1) 对已知的金属/塑料/真鱼样本，在记录曝光与 gain 的情况下采集"
          "配准良好的 0°/90° 与完整偏振序列（DoLP/AoLP 可计算），"
          "替代当前的双视点差分代理；"
          "2) 补齐同步时间戳与增益/gamma 元数据，使跨组比较可控；"
          "3) 对边界带的有效性定义做一次标定复核（当前 band 上配准误差"
          "直接进入 q）。"),
    "C": ("当前证据不足：停止在该数据上的模型扩张，保留灰度基线；"
          "缺少的证据是跨组一致、与错误方向绑定的偏振空间/统计线索。"
          "不继续追加诊断循环，除非出现新的采集或标定证据。"),
}


# ---------------------------------------------------------------------------
# Main


def _full_mode_phases(ctx: dict, args, output_dir: Path) -> None:
    """The original full diagnostic flow: admission over the pinned
    frozen inputs, the 39-file ledger, control selection, the per-frame
    fix-q rebuild for the selected panels and the corrected report.
    Failures are stashed into ctx for the shared closeout."""
    registry = ctx["registry"]
    watch = ctx["watch"]
    stage = ctx["stage"]
    log = ctx["log"]
    try:
        try:
            watch.check("before admission")
            admission = run_admission(ctx, args)
            stage("admission_done", "COMPLETED")
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the ledger")
            ledger, stats = build_ledger(
                _PROJECT_ROOT / FIT_DIR, admission["npz"],
                admission["manifest"], admission["authority"])
            problems = verify_expected_counts(stats)
            if problems:
                raise ScaleRefused(
                    "the derived counts disagree with the "
                    f"pre-registered expectations: {problems}")
            dump_json(output_dir / "instance_ledger.json",
                      {"experiment": EXPERIMENT,
                       "note": ("one row per unique instance; S16 null "
                                "means a missing block; margins are raw "
                                "logit top1-top2 (no softmax)"),
                       "rows": ledger})
            brief = [
                {"sample_name": r["sample_name"], "group": r["group"],
                 "outer_fold": r["outer_fold"], "class_name": r["class_name"],
                 "r0_prediction": r["r0"]["prediction"],
                 "r0_correct": r["r0"]["correct"],
                 "raw_logit_margin": r["r0"]["raw_logit_margin"],
                 "s_valid": r["s_valid"],
                 "c0_valid_ratio": r["c_features"][0],
                 "c3_gray_mean": r["c_features"][3],
                 "corrected_paths": sum(
                     1 for arm in ARMS for seed in TRAIN_SEEDS
                     if r["rcs_vs_r0"][f"{arm}|{seed}"]["fixed"]),
                 "broken_paths": sum(
                     1 for arm in ARMS for seed in TRAIN_SEEDS
                     if r["rcs_vs_r0"][f"{arm}|{seed}"]["broke"]),
                 "changed": r["sample_name"] in set(stats["changed_names"])}
                for r in ledger]
            dump_jsonl(output_dir / "instance_ledger_brief.jsonl", brief)
            dump_json(output_dir / "counts.json",
                      {k: v for k, v in stats.items()})
            stage("ledger", "COMPLETED", rows=len(ledger))
        except DeadlineExceeded:
            raise
        try:
            watch.check("before binning and controls")
            edges = fold_bin_edges(admission["npz"], admission["manifest"],
                                   admission["authority"])
            row_values = {
                row["sample_name"]: {
                    "valid_ratio": row["c_features"][0],
                    "gray_mean": row["c_features"][3],
                    "outer_fold": row["outer_fold"]}
                for row in ledger}
            bins = assign_bins(row_values, edges)
            dump_json(output_dir / "binning.json",
                      {"edges_by_fold": edges, "bins": bins,
                       "rule": ("C0/C3 terciles per outer fold from "
                                "inner_train only; interior/band validity "
                                "recorded separately in the ledger")})
            control_panels, control_notes = select_controls(ledger, stats)
            dump_json(output_dir / "selection_manifest.json", {
                "changed_panels": [
                    {"panel_id": f"changed_{index:02d}",
                     "sample_name": name, "kind": "changed",
                     "reason": ("the instance's correctness CHANGED for "
                                "at least one of the 12 RCS paths "
                                "(fixed or broke versus R0)")}
                    for index, name in
                    enumerate(stats["changed_names"])],
                "control_panels": control_panels,
                "control_notes": control_notes,
                "max_controls": MAX_CONTROLS,
                "max_panels": MAX_PANELS})
            stage("controls", "COMPLETED",
                  controls=len(control_panels),
                  notes=len(control_notes))
        except DeadlineExceeded:
            raise
        panels = [
            {"panel_id": f"changed_{index:02d}", "sample_name": name,
             "kind": "changed"}
            for index, name in enumerate(stats["changed_names"])]
        panels += control_panels
        if len(panels) > MAX_PANELS:
            raise ScaleRefused(
                f"{len(panels)} panels exceeds the {MAX_PANELS} cap")
        try:
            watch.check("before the local rebuild")
            import csv as _csv

            source_root = _PROJECT_ROOT / admission["cfg"]["data"][
                "source_root"]
            pair_path = source_root / "pair_manifest.csv"
            registry.register(pair_path, "source_pair_manifest")
            pair_manifest = {}
            with pair_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in _csv.DictReader(handle):
                    pair_manifest[str(row["output_stem"]).strip()] = row
            dev_rows = admission["manifest"]
            manifest_by_name = {s["sample_name"]: s for s in dev_rows}
            v5_manifest_path = (
                _PROJECT_ROOT / admission["cfg"]["data"]["root"] /
                "dataset_manifest.csv")
            registry.register(v5_manifest_path,
                              "v5_record:dataset_manifest.csv")
            with v5_manifest_path.open(encoding="utf-8-sig",
                                       newline="") as handle:
                v5_rows = {row["sample_name"]: row
                           for row in _csv.DictReader(handle)}
            matcher_params = matcher_config_of(admission["cfg"])
            stereo_cfg = StereoMatcherConfig(
                matcher=str(matcher_params["matcher"]),
                mode=str(matcher_params["mode"]),
                max_disparity=int(matcher_params["max_disp"]),
                scale=float(matcher_params["scale"]),
                block_size=int(matcher_params["block_size"]),
                uniqueness_ratio=int(matcher_params["uniqueness_ratio"]),
                speckle_window=int(matcher_params["speckle_window"]),
                speckle_range=int(matcher_params["speckle_range"]),
                texture_threshold=float(matcher_params["texture_threshold"]),
                lr_check_threshold_px=float(
                    matcher_params["lr_check_threshold_px"]),
                min_valid_ratio=float(matcher_params["min_valid_ratio"]),
                upsample_version=UPSAMPLE_FIX_VERSION)
            if stereo_cfg.upsample_version != UPSAMPLE_FIX_VERSION:
                raise ScaleRefused("matcher did not receive the fix version")
            matcher = StereoMatcher(stereo_cfg)
            frames = sorted({manifest_by_name[p["sample_name"]][
                "source_frame"] for p in panels})
            frame_instances: dict[str, list] = {}
            for sample in dev_rows:
                frame_instances.setdefault(sample["source_frame"],
                                           []).append(sample)
            rebuild_problems: list[str] = []
            rebuild_records = []
            selected = {p["sample_name"] for p in panels}
            panel_dir = output_dir / "panels"
            panel_dir.mkdir(parents=True)
            panels_manifest = []
            for frame_index, stem in enumerate(frames):
                watch.check(f"rebuild frame {stem}")
                rows = frame_instances[stem]
                v5_by_name = {}
                for row in rows:
                    row = {**row, **{k: v for k, v in
                                     v5_rows[row["sample_name"]].items()
                                     if k in ("crop_x1", "crop_y1",
                                              "crop_x2", "crop_y2",
                                              "left_path", "label_path")}}
                    v5_by_name[row["sample_name"]] = row
                for name, vrow in v5_by_name.items():
                    if "left_path" not in vrow:
                        raise ScaleRefused(
                            f"{name}: the V5 manifest row lacks paths")
                rebuilt = rebuild_frame(ctx, admission["cfg"], matcher,
                                        stem, [v5_by_name[name] for name
                                               in sorted(v5_by_name)],
                                        pair_manifest,
                                        selected & set(v5_by_name))
                for name in sorted(selected & set(v5_by_name)):
                    diag = rebuilt["diagnostics"][name]
                    row_index = [s["sample_name"] for s in dev_rows].index(
                        name)
                    comparison = compare_rebuilt_to_cache(
                        name, row_index, diag, admission["npz"],
                        dev_rows)
                    rebuild_problems.extend(comparison["problems"])
                    rebuild_records.append(comparison)
                    ledger_row = next(r for r in ledger
                                      if r["sample_name"] == name)
                    panel = next(p for p in panels
                                 if p["sample_name"] == name)
                    rendered = render_panel(panel_dir, panel, diag,
                                            ledger_row,
                                            admission["cfg"]["data"][
                                                "class_names"])
                    stereo_reason = diag["fix_provenance"]["stereo_reason"]
                    panels_manifest.append({
                        **panel, **rendered,
                        "source_frame": stem,
                        "rebuild_check": comparison,
                        "observation_automatic": (
                            f"stereo={stereo_reason}; interior_valid="
                            f"{diag['blocks']['interior_valid']}; "
                            f"band_valid={diag['blocks']['band_valid']}; "
                            "recorded by the automatic rebuild rules"),
                        "observation_recorder": "自动规则",
                        "observation_human": None,
                        "note": ("no human panel review has happened at "
                                 "run time; human observations stay "
                                 "unknown until a reviewer records them")})
                ctx["log"](f"rebuilt {frame_index + 1}/{len(frames)}: "
                           f"{stem}")
            if rebuild_problems:
                raise ScaleRefused(
                    "the rebuilt fix statistics disagree with the "
                    f"verified cache: {rebuild_problems[:5]}")
            dump_json(output_dir / "panels_manifest.json",
                      {"panels": panels_manifest,
                       "display_rules": {
                           "gray": "vmin=0, vmax=255 (fixed)",
                           "signed_q": "vmin=-1, vmax=1 (fixed)",
                           "abs_q": "vmin=0, vmax=1 (fixed)",
                           "invalid": "masked light-gray; stored zeros "
                                      "are never shown as real low q"},
                       "panel_count": len(panels_manifest),
                       "cap": MAX_PANELS})
            index_rows = []
            for p in panels_manifest:
                state = "OK" if not p["rebuild_check"]["problems"] \
                    else "FAIL"
                index_rows.append(
                    f"<tr><td>{p['panel_id']}</td>"
                    f"<td>{p['sample_name']}</td><td>{p['kind']}</td>"
                    f"<td>{state}</td>"
                    f"<td><a href='panels/{p['file']}'>{p['file']}</a>"
                    "</td></tr>")
            (output_dir / "index.html").write_text(
                "\n".join(
                    ["<!doctype html><meta charset='utf-8'>",
                     "<title>polar complementarity panels</title>",
                     "<table border=1><tr><th>panel</th><th>sample</th>"
                     "<th>kind</th><th>rebuild</th><th>image</th></tr>"]
                    + index_rows + ["</table>"]),
                encoding="utf-8")
            stage("rebuild_and_panels", "COMPLETED",
                  frames=len(frames), instances=len(panels_manifest))
            rebuild_summary = {"frames": len(frames),
                               "instances": len(panels_manifest),
                               "problems": rebuild_problems}
        except DeadlineExceeded:
            raise
        try:
            watch.check("before conditional tables")
            tables = conditional_tables(ledger, stats, bins)
            dump_json(output_dir / "conditional_tables.json",
                      {"tables": tables})
            dump_json(output_dir / "same_group_comparisons.json",
                      {"comparisons": tables.get("same_group_comparisons",
                                                 [])})
            stage("conditional_tables", "COMPLETED")
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the report")
            facts = {
                "macro_f1": {
                    "R0": THIRD_BATCH_FACTS["macro_f1_mean_of_seeds"][
                        ("fixed", "R0")],
                    "fixed RC": THIRD_BATCH_FACTS[
                        "macro_f1_mean_of_seeds"][("fixed", "RC")],
                    "fixed RCS": THIRD_BATCH_FACTS[
                        "macro_f1_mean_of_seeds"][("fixed", "RCS")],
                    "scaled RC": THIRD_BATCH_FACTS[
                        "macro_f1_mean_of_seeds"][("scaled", "RC")],
                    "scaled RCS": THIRD_BATCH_FACTS[
                        "macro_f1_mean_of_seeds"][("scaled", "RCS")],
                }}
            route = suggest_route(tables, stats)
            dump_json(output_dir / "route_suggestion.json", route)
            write_report(output_dir / "report.md", facts, tables,
                         route, control_notes, stats,
                         rebuild_summary)
            stage("report", "COMPLETED", route=route["route"])
        except DeadlineExceeded:
            raise
    except DeadlineExceeded as error:
        ctx["interrupted"] = True
        ctx["unfinished"].append(str(error))
        log(f"deadline exceeded: {error}")
    except (ScaleRefused, IdentityError, ComplementarityRefused,
            FileNotFoundError, ValueError, OSError, KeyError) as exc:
        ctx["failure"] = f"{type(exc).__name__}: {exc}"
        log(f"failure: {ctx['failure']}")
    except Exception as exc:  # unexpected: record and refuse success
        import traceback

        ctx["failure"] = f"UNEXPECTED {type(exc).__name__}: {exc}"
        ctx["failure_traceback"] = traceback.format_exc()
        log(f"unexpected failure: {ctx['failure']}\n"
            f"{ctx['failure_traceback']}")


# ---------------------------------------------------------------------------
# Postprocess mode: a pure offline re-analysis of the ONE prior
# diagnostic directory. It never rebuilds pixels, never renders panels
# and never runs a model — it re-derives the counts from the saved
# ledger, re-checks the saved binning and panel coverage and produces
# the corrected layered statistics and report.


def _postprocess_phases(ctx: dict, args, output_dir: Path) -> None:
    registry = ctx["registry"]
    watch = ctx["watch"]
    stage = ctx["stage"]
    log = ctx["log"]
    prior_dir = _PROJECT_ROOT / PRIOR_DIAGNOSTIC_REL
    try:
        try:
            watch.check("before postprocess admission")
            # identity FIRST: every pinned prior artifact (and the pinned
            # config) is hash-verified and registered BEFORE any of them
            # is read; a mismatch refuses and never updates an expectation
            for name, expected in PINNED_PRIOR_ARTIFACTS.items():
                path = prior_dir / name
                if not path.is_file():
                    raise ScaleRefused(f"prior artifact missing: {path}")
                actual = sha256_of_file(path)
                if actual != expected:
                    raise ScaleRefused(
                        f"{name}: sha256 {actual} != the review-pinned "
                        f"{expected}; refusing (expectations are never "
                        "updated)")
            registered = {}
            for name, expected in PINNED_PRIOR_ARTIFACTS.items():
                verify_anchor(prior_dir / name, expected, registry,
                              "anchor:prior_artifact")
                registered[name] = expected
            config_path = _PROJECT_ROOT / PINNED_CONFIG_REL
            verify_anchor(config_path, PINNED_INPUTS[PINNED_CONFIG_REL],
                          registry, "anchor:postprocess_config")
            registered[PINNED_CONFIG_REL] = \
                PINNED_INPUTS[PINNED_CONFIG_REL]
            # the postprocess never parses the config: nothing in the
            # offline re-analysis consumes it; its identity is checked so
            # the run is bound to the SAME frozen file the diagnostic used
            dump_json(output_dir / "postprocess_admission.json", {
                "mode": "postprocess", "experiment": EXPERIMENT,
                "prior_diagnostic_dir": str(prior_dir),
                "pinned_artifacts": registered,
                "config_parsed": False,
                "config_note": ("the config is identity-checked and "
                                "registered but never parsed: no "
                                "postprocess computation reads it"),
                "zero_rebuild_note": ("this mode never calls "
                                      "rebuild_frame, StereoMatcher, "
                                      "compute_polar_features or "
                                      "render_panel and never renders a "
                                      "new PNG"),
            })
            stage("postprocess_admission", "COMPLETED",
                  artifacts=len(registered))
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the saved ledger")
            ledger_doc = json.loads(
                (prior_dir / "instance_ledger.json").read_text(
                    encoding="utf-8"))
            rows = ledger_doc.get("rows") or []
            problems = verify_saved_ledger(rows)
            if problems:
                raise ScaleRefused(
                    "the saved ledger fails structural validation: "
                    f"{problems[:5]}")
            stats = derive_counts(rows)
            saved_counts = json.loads(
                (prior_dir / "counts.json").read_text(encoding="utf-8"))
            mismatches = [key for key in
                          ("unique_instances", "r0_errors",
                           "r0_errors_with_s", "corrected_union",
                           "broken_union", "changed_union",
                           "corrected_pairs", "broken_pairs",
                           "instance_model_pairs_evaluated")
                          if stats.get(key) != saved_counts.get(key)]
            if mismatches:
                raise ScaleRefused(
                    "the re-derived counts disagree with the saved "
                    f"counts.json on {mismatches}")
            count_problems = []
            for key, expected in POSTPROCESS_EXPECTED.items():
                actual = (6 * stats["unique_instances"]
                          if key == "rcs_vs_r0_comparison_pairs"
                          else stats.get(key))
                if actual != expected:
                    count_problems.append(
                        f"{key}: re-derived {actual} != the "
                        f"review-confirmed {expected}")
            if count_problems:
                raise ScaleRefused(
                    "the re-derived counts disagree with the "
                    f"pre-registered expectations: {count_problems}")
            dump_json(output_dir / "counts.json", {
                **{k: v for k, v in stats.items()},
                "denominators": {
                    "rc_rcs_saved_prediction_pairs":
                        stats["instance_model_pairs_evaluated"],
                    "rcs_vs_r0_comparison_pairs":
                        6 * stats["unique_instances"],
                    "note": ("both are repeated model evaluations, not "
                             "independent sample counts; the first is "
                             "12 x 2446 saved RC+RCS prediction pairs, "
                             "the second is 6 x 2446 RCS-versus-R0 "
                             "comparisons of which 32 fixed and 34 "
                             "broke"),
                },
                "rederivation": {
                    "source": "instance_ledger.json (saved rows)",
                    "counts_json_agreement": True,
                    "expected_counts_agreement": True,
                }})
            stage("ledger_rederivation", "COMPLETED", rows=len(rows))
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the saved binning")
            binning_doc = json.loads(
                (prior_dir / "binning.json").read_text(encoding="utf-8"))
            problems = verify_saved_binning(rows, binning_doc)
            if problems:
                raise ScaleRefused(
                    "the saved binning does not re-derive: "
                    f"{problems[:5]}")
            bin_of = binning_doc["bins"]
            stage("binning_recheck", "COMPLETED",
                 instances=len(bin_of))
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the saved panel coverage")
            panels_manifest = json.loads(
                (prior_dir / "panels_manifest.json").read_text(
                    encoding="utf-8"))
            selection = json.loads(
                (prior_dir / "selection_manifest.json").read_text(
                    encoding="utf-8"))
            panels_dir = prior_dir / "panels"
            problems = verify_panel_coverage(
                panels_manifest, selection, stats["changed_names"],
                panels_dir)
            if problems:
                raise ScaleRefused(
                    "the saved panel coverage fails: "
                    f"{problems[:5]}")
            stage("panel_coverage", "COMPLETED",
                  panels=len(panels_manifest.get("panels", [])),
                  rebuild_problems=0)
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the layered statistics")
            analysis = build_conditional_analysis(rows, stats, bin_of)
            check = analysis.get("comparison_count_check") or {}
            per_group = check.get("per_confusion_group") or {}
            if check.get("global") != sum(per_group.values()):
                raise ScaleRefused(
                    "the comparison count is not conserved across the "
                    f"confusion groups: {check}")
            dump_json(output_dir / "conditional_tables.json",
                      {"tables": analysis})
            dump_json(output_dir / "same_group_comparisons.json",
                      {"comparisons":
                       analysis.get("same_group_comparisons", []),
                       "comparisons_by_group":
                       analysis.get("comparisons_by_group", {}),
                       "count_check": check})
            stage("layered_statistics", "COMPLETED",
                  comparisons=len(analysis.get("same_group_comparisons",
                                               [])),
                  per_confusion_group=per_group)
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the five-layer subset statistics")
            layered = subset_layered_analysis(rows, stats, bin_of)
            conservation = layered.get("conservation_checks") or {}
            if conservation.get("problems"):
                raise ScaleRefused(
                    "the five-layer subset statistics fail the "
                    f"conservation checks: {conservation['problems'][:5]}")
            dump_json(output_dir / "layered_subsets.json",
                      {"layered_subsets": layered})
            q1 = q1_fish_pair_common_support(analysis)
            dump_json(output_dir / "q1_fish_pair_common_support.json", q1)
            stage("five_layer_statistics", "COMPLETED",
                  conservation=conservation.get("verdict"),
                  q1_cells_with_both_sides=q1["cells_with_both_sides"],
                  q1_cells_with_common_valid_component=q1[
                      "cells_with_common_valid_component"])
        except DeadlineExceeded:
            raise
        try:
            watch.check("before the corrected report")
            route = suggest_route(analysis, stats)
            dump_json(output_dir / "route_suggestion.json", route)
            _write_postprocess_report(
                output_dir / "report.md", stats, analysis, route,
                prior_dir, panels_manifest, output_dir, layered, q1)
            _write_postprocess_index(output_dir / "index.html",
                                     panels_manifest, panels_dir,
                                     output_dir)
            stage("report", "COMPLETED", route=route["route"])
        except DeadlineExceeded:
            raise
    except DeadlineExceeded as error:
        ctx["interrupted"] = True
        ctx["unfinished"].append(str(error))
        log(f"deadline exceeded: {error}")
    except (ScaleRefused, IdentityError, ComplementarityRefused,
            FileNotFoundError, ValueError, OSError, KeyError) as exc:
        ctx["failure"] = f"{type(exc).__name__}: {exc}"
        log(f"failure: {ctx['failure']}")
    except Exception as exc:
        import traceback

        ctx["failure"] = f"UNEXPECTED {type(exc).__name__}: {exc}"
        ctx["failure_traceback"] = traceback.format_exc()
        log(f"unexpected failure: {ctx['failure']}\n"
            f"{ctx['failure_traceback']}")


def _question_answers(analysis: dict, q1: dict) -> dict:
    """Mechanical, descriptive answers to the four review questions.
    Q1 uses ONLY confusion_group=plastic_fish_vs_real_fish,
    comparison_type=corrected_vs_always_wrong (never a global count);
    Q3 reports the broken-vs-stable-correct comparable cells PER
    confusion group with the overlap note; Q4 names the mirror
    relation between the two directions of one pair. Every number
    traces to an explicit comparison set."""
    comparisons = analysis.get("same_group_comparisons", [])

    def comparable_of(comparison_type, group_name=None):
        cells = [c for c in comparisons
                 if c.get("comparison_type") == comparison_type
                 and (group_name is None
                      or c.get("confusion_group") == group_name)]
        return [c for c in cells if c.get("comparable")]

    # Q3: the broken-vs-stable-correct comparable cells PER confusion
    # group; the three sets overlap (an instance can belong to several
    # pairs) — per-group counts are reported, and any pooled number must
    # deduplicate by (true class, group, fold, bin)
    q3_per_group = {}
    q3_dedup_cells = {}
    for group in ("plastic_submarine_vs_plastic_fish",
                  "plastic_fish_vs_real_fish",
                  "metal_submarine_vs_plastic_submarine"):
        cells = comparable_of("broken_vs_stable_correct", group)
        q3_per_group[group] = len(cells)
        for cell in cells:
            key = (cell["true_class"], cell["group"], cell["outer_fold"],
                   cell["bin"])
            q3_dedup_cells[key] = cell
    dark_observations = []
    for cell in comparable_of("broken_vs_stable_correct"):
        for key in ("c0_valid_ratio", "c3_gray_mean"):
            gap = (cell.get("median_gaps") or {}).get(key)
            if gap is not None:
                dark_observations.append({
                    "confusion_group": cell.get("confusion_group"),
                    "true_class": cell.get("true_class"),
                    "group": cell.get("group"),
                    "outer_fold": cell.get("outer_fold"),
                    "bin": cell.get("bin"), "component": key,
                    **gap})
    class_cells = [c for c in comparisons
                   if c.get("comparison_type") == "class_vs_class"
                   and c.get("confusion_group")
                   == "metal_submarine_vs_plastic_submarine"
                   and c.get("comparable")]
    return {
        "q1_fish_pair": q1,
        "q1_verdict": (
            f"observed (small-sample description): the fish-pair "
            f"corrected-vs-always-wrong comparisons have "
            f"{q1['cells_with_both_sides']} cells with BOTH sides present "
            f"across {q1['n_groups_spanned']} collection groups; "
            f"{q1['cells_with_common_valid_component']} of them share at "
            "least one valid S16 component (the rest are both-sides-"
            "present but polarization-incomparable). Supports of 1-3 per "
            "side are small samples; the per-cell median differences and "
            "directions are registered in same_group_comparisons.json, "
            "the supports and common-valid-component counts in "
            "q1_fish_pair_common_support.json — they "
            "are NOT a cross-group stable cue, NOT material causality "
            "and NOT a net-benefit claim"),
        "q2_s16_spatial_information": {
            "verdict": "not verifiable with this diagnostic",
            "reason": ("S16 is a fixed statistical summary; whether it "
                       "discards useful spatial arrangement can only be "
                       "answered with pixel-level representations, which "
                       "this zero-rebuild closeout does not read"),
        },
        "q3_broken_vs_dark_low_coverage": {
            "comparable_cells_per_confusion_group": q3_per_group,
            "dedup_note": ("the three per-group sets overlap; the "
                           "deduplicated cell count (by true class, "
                           "group, fold, bin) is "
                           f"{len(q3_dedup_cells)}"),
            "cells": dark_observations[:48],
            "verdict": (
                "observed: broken-vs-stable-correct C0/C3 median "
                "directions in the comparable cells are listed below; "
                "they are descriptive associations with support counts, "
                "NOT a co-occurrence with edge/registration phenomena "
                "(no pixel was read) and NOT a cause claim — dark-area "
                "or coverage associations do not establish a boundary "
                "or registration reason"),
        },
        "q4_metal_vs_plastic_submarine": {
            "comparable_cells": len(class_cells),
            "mirror_note": ("the two directions of the pair are the same "
                            "cells read in opposite order — never two "
                            "independent supports"),
            "cells": class_cells[:24],
            "verdict": (
                "observed: the comparable class-pair cells are listed "
                "with supports and median directions; no cross-group "
                "consistent distinction is claimed"
                if class_cells else
                "insufficient data: no same-cell class comparison has "
                "support on both sides"),
        },
    }


def _write_postprocess_report(report_path: Path, stats: dict,
                              analysis: dict, route: dict,
                              prior_dir: Path, panels_manifest: dict,
                              output_dir: Path, layered: dict,
                              q1: dict) -> None:
    panels = panels_manifest.get("panels") or []
    comparisons = analysis.get("same_group_comparisons", [])
    comparable_cells = [c for c in comparisons if c.get("comparable")]
    answers = _question_answers(analysis, q1)
    check = analysis.get("comparison_count_check") or {}
    conservation = layered.get("conservation_checks") or {}
    import os

    old_report_href = Path(os.path.relpath(
        prior_dir / "report.md", start=output_dir)).as_posix()
    q1_table = ["| 真实类别 | 组 | 折 | bin | 纠正侧 | 始终错误侧 | 共同有效分量 |",
                "|---|---|---|---|---|---|---|"]
    for row in q1["cell_table"]:
        q1_table.append(
            f"| {row['true_class']} | {row['group']} | "
            f"{row['outer_fold']} | {row['bin']} | "
            f"{row['corrected_support']} | {row['always_wrong_support']} | "
            f"{row['n_common_valid_components']} |")
    q3 = answers["q3_broken_vs_dark_low_coverage"]
    q3_per = q3["comparable_cells_per_confusion_group"]
    lines = [
        "# material_polar_complementarity_v1 — 收尾补正报告（postprocess，第二次）",
        "",
        "科学状态：**尚未建立偏振净收益**。本报告是纯离线补正：只消费已",
        "保存的诊断产物（instance_ledger / binning / counts / selection /",
        "panels_manifest / run_status），重新推导计数、补齐三类比较与五层",
        "×子集统计；**本次重建 0 帧、新渲染面板 0 张、新模型拟合 / 新 ",
        "scaler 拟合 / 分类前向 0 次**。",
        "",
        "## 对旧报告与上一轮补正的订正",
        f"- 旧报告（{old_report_href}）的首选路线 B 依据不足，维持订正为",
        "  **C**；路线 C 是当前审查结论，不是 IQR 或纠错数量的自动分类结",
        "  果；不得为维持 C 抹去实际存在的线索。",
        "- 上一轮“真假鱼 0 个共同支持格”的表述不准确：本轮由台账推导，",
        f"  真假鱼 corrected-vs-always-wrong 有 {q1['cells_with_both_sides']}",
        f"  个两侧有实例支持的格（跨 {q1['n_groups_spanned']} 个采集组），",
        f"  其中 {q1['cells_with_common_valid_component']} 个有共同有效 ",
        "  S16 分量。",
        "- 上一轮“完整分层已经验证未见跨组一致可分离模式”的表述撤回：分",
        "  层统计本轮才完整生成（此前只落盘最细层联合格），尚未据此做出",
        "  “已验证未见”的结论；只有下述逐格观察。",
        "- 旧“与缓存逐项核对不一致”是实现层列表判断错误；41 项 ",
        "  rebuild_check 全部为空问题列表。",
        "- 比较汇总缺陷订正：上一轮 conditional_tables 内每对比较被后一对",
        "  覆盖（只余金属/塑料潜艇 412 条）；本轮三对全部落盘并通过 ",
        "  global=Σper-group 守恒校验。",
        "",
        "## 计数口径（全部由保存台账重新推导，不复制旧 counts）",
        f"- 唯一实例 {stats['unique_instances']}；R0 错误 "
        f"{stats['r0_errors']}（有 S 的 {stats['r0_errors_with_s']}）；",
        f"- RCS 曾纠正 {stats['corrected_union']}、曾破坏 "
        f"{stats['broken_union']}、正确性变化并集 {stats['changed_union']}；",
        f"- RC+RCS 保存预测实例—模型对：12×2446＝"
        f"{stats['instance_model_pairs_evaluated']}；",
        f"- RCS 相对 R0 比较：6×2446＝{6 * stats['unique_instances']}，",
        f"  其中纠正 {stats['corrected_pairs']}、破坏 {stats['broken_pairs']}。",
        "  两者都是重复模型评估，不是独立样本数。",
        "",
        "## 三类比较（每个混淆对完整保存；全局=三对连接）",
        f"- 全局比较 {check.get('global')} 条 = Σper-confusion-group "
        + " + ".join(
            f"{v} ({k})" for k, v in
            (check.get("per_confusion_group") or {}).items())
        + "；守恒校验通过。每条比较带 confusion_group、comparison_type、",
        "  comparison_id 与完整条件键。",
        f"- 两侧均有实例支持的比较 {len(comparable_cells)} 条。",
        "",
        "## Q1：真假鱼 corrected-vs-always-wrong（只引用该对）",
        f"- 两侧有实例支持：{q1['cells_with_both_sides']} 格，涉及 "
        f"{q1['n_groups_spanned']} 个采集组；",
        f"- 有至少一个共同有效 S16 分量："
        f"{q1['cells_with_common_valid_component']} 格，涉及 "
        f"{len(q1['groups_with_common_valid_component'])} 个采集组；",
        "- 两侧都有实例但无共同有效分量的格："
        f"{len(q1['cells_both_sides_but_no_common_valid_component'])} 个",
        "（不能称为偏振数值可比较）；",
        "- 明确区分：两侧有样本 ≠ 两侧有共同有效分量 ≠ 充分科学证据；共",
        "  同支持不自动成为跨组稳定线索。逐格中位数差与方向见 ",
        "  same_group_comparisons.json；支持数与共同有效分量数见 ",
        "  q1_fish_pair_common_support.json。",
        "",
        *q1_table,
        "",
        "## Q3：曾破坏 vs 稳定正确（按混淆对）",
        f"- 可比较格：塑料潜艇/塑料鱼 {q3_per['plastic_submarine_vs_plastic_fish']}",
        f"，真假鱼 {q3_per['plastic_fish_vs_real_fish']}，金属/塑料潜艇 ",
        f"{q3_per['metal_submarine_vs_plastic_submarine']}；三个集合存在",
        f"重合，按 (真实类别,组,折,bin) 去重后 {q3['dedup_note'].split('is ')[-1]} 个。",
        "- 暗区/覆盖率关联是描述性统计，不等于边界或配准原因。",
        "",
        "## Q4：金属/塑料潜艇（镜像说明）",
        f"- 可比较格 {answers['q4_metal_vs_plastic_submarine']['comparable_cells']}",
        "  个；正反方向是同一组数据的镜像，不计为独立复现；不宣称跨组一",
        "  致区别。",
        "",
        "## 五层 × 各子集统计",
        "- layered_subsets.json 为每个混淆对 × 7 个子集 × 5 层（L1 真实类",
        "  别；L2 类别×折；L3 类别×组保留折；L4 类别×折×联合 bin；L5 类",
        "  别×组×折×bin）的实际统计：support、缺测数、内/边带有效数、各 ",
        "  S16 分量有效 n/缺测 n/中位数/Q25/Q75、C0/C3 分位；",
        f"  守恒校验 {conservation.get('verdict')}（分区 support 之和=唯一",
        "  成员数；n_valid+n_missing=support）。",
        "- 粗层由该子集自身成员重算，不由细格合并；同号 bin 不跨折混合；",
        "  不同子集/层级重叠，不得相加当独立样本；空 S/空单元为 null，不",
        "  是零测量。",
        "",
        "## 重建复用声明",
        "- 复用旧证据：39 帧、41 个实例的重建核对一致（旧 panels_manifest",
        "  的 rebuild_check 全部为空问题列表，本次逐项复核）；**本次重建 ",
        "0 帧、新渲染面板 0 张**，不把旧重建算作本次执行。",
        f"- 旧面板 {len(panels)} 张（29 变化 + 12 控制）原样保留；本报告",
        "  与 index.html 只引用旧面板文件，不覆盖旧 panels_manifest。",
        "",
        "## 分层观察的支持与限制",
        f"- 同组比较单元共 {len(comparisons)} 条（三对合计），两侧均有支",
        f"持的 {len(comparable_cells)} 条；全部单元的支持数、有效 n、中位",
        "  数与分位方向见 same_group_comparisons.json。观察只是统计关联，",
        "  不是原因；没有实际看图记录，不写成空间观察或人工确认。旧面板可",
        "  人工查看，但任何新观察必须另行记录观察者与样本，不得回写旧 ",
        "  panels_manifest。没有观察就保留未知。",
        "- 分箱比较保留折语境：不同折的同号分箱不代表相同绝对亮度。",
        "- low_support 类标记只是实现层展示提示，不是预注册门槛，也不是路",
        "  线选择器。",
        "",
        "## 首选路线",
        f"**路线 {route['route']}** — {route['reason']}",
        "",
        "## 物理与统计限制",
        "- 当前 q 是双视点 0°/90° 配准差分代理，不是完整 DoLP/AoLP。",
        "- 曝光、gain/gamma、同步历史元数据不足；physical_object_id 不可",
        "  确认；采集组不等同于独立实物。",
        "- 现有外层结果已用于设计诊断；同池后续实验属探索性开发。",
        "",
        "## 证据索引（本次输出均引用旧面板，不重写旧证据）",
        "- postprocess_admission.json / source_manifest.json / counts.json",
        "  / conditional_tables.json / same_group_comparisons.json /",
        "  layered_subsets.json / q1_fish_pair_common_support.json /",
        "  route_suggestion.json / report.md / index.html / run_status.json",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_postprocess_index(index_path: Path, panels_manifest: dict,
                             panels_dir: Path, output_dir: Path) -> None:
    import os

    rows = []
    for p in panels_manifest.get("panels", []):
        href = Path(os.path.relpath(panels_dir / p["file"],
                                    start=output_dir)).as_posix()
        target = (output_dir / href).resolve()
        if not target.is_file():
            raise ScaleRefused(
                f"the panel link target does not exist: {target}")
        rows.append(
            f"<tr><td>{p['panel_id']}</td><td>{p['sample_name']}</td>"
            f"<td>{p['kind']}</td><td>OK</td>"
            f"<td><a href='{href}'>{p['file']}</a></td></tr>")
    index_path.write_text(
        "\n".join(
            ["<!doctype html><meta charset='utf-8'>",
             "<title>polar complementarity panels (prior run, referenced)",
             "</title>",
             "<p>the panels below are the PRIOR diagnostic's files, "
             "referenced in place — nothing was re-rendered</p>",
             "<table border=1><tr><th>panel</th><th>sample</th>"
             "<th>kind</th><th>rebuild</th><th>image</th></tr>"]
            + rows + ["</table>"]),
        encoding="utf-8")


def _closeout_and_status(ctx: dict, args, output_dir: Path,
                         started: float) -> int:
    registry = ctx["registry"]
    watch = ctx["watch"]
    stage = ctx["stage"]
    log = ctx["log"]
    stages = ctx["stages"]
    failure = ctx["failure"]
    interrupted = ctx["interrupted"]
    unfinished = ctx["unfinished"]
    try:
        protection = registry.compare()
        registry.write_manifest(output_dir / "source_manifest.json",
                                protection)
        stage("protection",
              "COMPLETED" if protection["identical"] else "FAILED",
              registered=protection["registered_sources"],
              problems=protection["problems"])
        if not protection["identical"] and failure is None:
            failure = "protection check found changed inputs"
    except Exception as exc:
        log(f"closeout failed: {exc}")
        if failure is None:
            failure = f"CLOSEOUT {type(exc).__name__}: {exc}"
    if not interrupted:
        try:
            watch.check("before success status publish")
        except DeadlineExceeded as exc:
            interrupted = True
            unfinished.append(str(exc))
            log(f"deadline crossed before the success status: {exc}")
    if failure:
        status, code = "FAILED", EXIT_REFUSED
    elif interrupted:
        status, code = "PARTIAL", EXIT_PARTIAL
    else:
        status, code = "COMPLETED", EXIT_OK
    dump_json(output_dir / "run_status.json", {
        "run_mode": args.mode, "experiment": EXPERIMENT,
        "stages": stages, "interrupted_by_deadline": interrupted,
        "unfinished": unfinished, "failure": failure,
        "failure_traceback": ctx["failure_traceback"],
        "overall_status": status, "exit_code": code,
        "zero_training_calls": {**_zero_calls(),
                                "note": ("this entry contains no model "
                                         "fit, scaler fit or classifier "
                                         "forward call path; entry tests "
                                         "prove the shared paths cannot "
                                         "fire")},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime()),
        "elapsed_s": time.time() - started,
        "protection": protection,
    })
    log(f"overall: {status} (exit {code})")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="material_polar_complementarity_v1 diagnostic "
                    "(zero training; saved-fit consumers only). Modes: "
                    "full (the original diagnostic) and postprocess "
                    "(pure offline re-analysis of the ONE prior "
                    "diagnostic directory).")
    parser.add_argument("--mode", choices=("full", "postprocess"),
                        default="full")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fit-dir", default=None)
    parser.add_argument("--verification-dir", default=None)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--diagnostic-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-minutes", type=float, required=True)
    args = parser.parse_args(argv)

    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
              f"{message}", file=sys.stderr, flush=True)

    try:
        if args.mode == "full":
            missing = [name for name, value in (
                ("--fit-dir", args.fit_dir),
                ("--verification-dir", args.verification_dir),
                ("--feature-dir", args.feature_dir)) if not value]
            disallowed = [name for name, value in
                          (("--diagnostic-dir", args.diagnostic_dir),)
                          if value]
            pinned_checks = (
                (args.fit_dir, _PROJECT_ROOT / FIT_DIR),
                (args.verification_dir,
                 _PROJECT_ROOT / FIT_DIR_VERIF_DIR),
                (args.feature_dir, _PROJECT_ROOT / CACHE_DIR))
        else:
            missing = [name for name, value in
                       (("--diagnostic-dir", args.diagnostic_dir),)
                       if not value]
            disallowed = [name for name, value in (
                ("--fit-dir", args.fit_dir),
                ("--verification-dir", args.verification_dir),
                ("--feature-dir", args.feature_dir)) if value]
            pinned_checks = (
                (args.diagnostic_dir,
                 _PROJECT_ROOT / PRIOR_DIAGNOSTIC_REL),)
        if missing:
            raise ScaleRefused(
                f"{args.mode} mode is missing {missing}")
        if disallowed:
            raise ScaleRefused(
                f"{args.mode} mode must not receive {disallowed}")
        # the config is a PINNED input: a substitute path is refused even
        # if its content were identical
        if Path(args.config).resolve() != \
                (_PROJECT_ROOT / PINNED_CONFIG_REL).resolve():
            raise ScaleRefused(
                f"{args.config} is not the pinned config "
                f"{_PROJECT_ROOT / PINNED_CONFIG_REL}")
        for given, expected in pinned_checks:
            if Path(given).resolve() != Path(expected).resolve():
                raise ScaleRefused(
                    f"{given} is not the pinned diagnostic input "
                    f"{expected}")
        if float(args.budget_minutes) <= 0:
            raise ScaleRefused("--budget-minutes must be positive")
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = _PROJECT_ROOT / output_dir
        _validate_output_dir(output_dir)
    except (ScaleRefused, IdentityError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    registry = SourceRegistry({})
    watch = DeadlineWatch(budget_minutes=float(args.budget_minutes))
    ctx = {
        "registry": registry, "watch": watch,
        "stage": (lambda name, status, **details: None), "log": log,
        "stages": {}, "failure": None, "interrupted": False,
        "unfinished": [], "failure_traceback": None,
    }

    def stage(name: str, status: str, **details) -> None:
        ctx["stages"][name] = {"status": status, **details}
        log(f"stage {name}: {status}")

    ctx["stage"] = stage
    output_dir.mkdir(parents=True)
    if args.mode == "full":
        _full_mode_phases(ctx, args, output_dir)
    else:
        _postprocess_phases(ctx, args, output_dir)
    return _closeout_and_status(ctx, args, output_dir, started)


if __name__ == "__main__":
    raise SystemExit(main())
