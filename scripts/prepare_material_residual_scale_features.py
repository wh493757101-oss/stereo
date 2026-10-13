"""material_residual_scale_v1 — build the fix-version feature cache.

The dev pool (train+val: 21 groups, 945 frames, 2446 instances, 4 frames
without targets) gets its polarization inputs REGENERATED through the
PRODUCTION ``StereoMatcher`` with the approved fix upsample version
(``valid_weighted_legacy_support_v1``): the same band contract, the same
polar differential and the same C14/S16 statistics as the original
pipeline — never a second stereo/upsample/polar implementation. The gray
crops, instance masks, classes, groups and the G256 features are the
FROZEN V5 / global-cache identities, checked per sample. The old test
split never contributes a sample, a statistic, a training row or a cache
row.

Modes:
- ``preflight`` — anchors, V5 admission, protocol, global cache and the
  dev-pool census; no feature computation;
- ``smoke``    — the first ``--max-frames`` dev frames into a cache that
  is MARKED INCOMPLETE (the training entry refuses it);
- ``build``    — the full dev pool into the complete experiment cache.

A shared ``--deadline-utc`` bounds the whole non-training flow: an
overrun stops NEW computation at a safe frame boundary, publishes the
PARTIAL evidence and exits non-zero — it never manufactures a completed
cache. Every consumed input is registered BEFORE its consuming read;
registration conflicts, missing files and end-of-run changes are
recorded and fail the acceptance.
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

from core.material_residual_scale import (  # noqa: E402
    CACHE_KIND_FULL,
    CACHE_KIND_SMOKE,
    EXPERIMENT,
    DeadlineExceeded,
    DeadlineWatch,
    ScaleRefused,
    UPSAMPLE_FIX_VERSION,
    expand_historical_bindings,
    load_scale_config,
    matcher_config_of,
    scale_cache_definition,
)
from core.material_mask_dataset import (  # noqa: E402
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    MaskFusionSampleArrays,
    admit_dataset_for_protocol,
    load_mask_sample,
)
from core.polar_local_features import (  # noqa: E402
    LOCAL_FEATURE_DEFINITION_VERSION,
    LOCAL_REGION_RULE,
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
from scripts.eval_material_upsample_impact import (  # noqa: E402
    IdentityError,
    SourceRegistry,
    dump_json,
    dump_jsonl,
    is_link_or_reparse,
    verify_anchor,
)
from scripts.prepare_cls_fusion_dataset import load_gray  # noqa: E402
from scripts.prepare_material_mask_dataset import read_pair_manifest  # noqa: E402

MODES = ("preflight", "smoke", "build")
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_PARTIAL = 2


def census_dev_pool(cfg: dict, registry=None) -> dict:
    """The authoritative dev-pool census from the frozen records.

    Dev rows come from the V5 dataset manifest (split in the configured
    dev pool); the frame list comes from the source pair manifest. The
    counts are checked against the pinned support expectations — a
    disagreement refuses the run before any feature computation.
    """
    import csv

    data_root = _PROJECT_ROOT / cfg["data"]["root"]
    splits = set(cfg["data"]["dev_pool_splits"])
    with (data_root / "dataset_manifest.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    dev_rows = [row for row in all_rows if row["split"] in splits]
    excluded = sorted({row["split"] for row in all_rows} - splits)
    if excluded != sorted(cfg["data"]["excluded_splits"]):
        raise ScaleRefused(
            f"dataset splits {excluded} != excluded_splits "
            f"{sorted(cfg['data']['excluded_splits'])}")
    source_root = _PROJECT_ROOT / cfg["data"]["source_root"]
    pair_path = source_root / "pair_manifest.csv"
    if not pair_path.is_file():
        raise ScaleRefused(f"source pair manifest missing: {pair_path}")
    # the pair manifest (and every record steered by it) is registered
    # before its consuming read; no historical record covers it, so the
    # registration is THIS batch's first-read snapshot
    if registry is not None:
        registry.register(pair_path, "source_pair_manifest")
    import csv as _csv

    pairs: dict[str, dict] = {}
    with pair_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in _csv.DictReader(handle):
            stem = str(row["output_stem"]).strip()
            if stem in pairs:
                raise ScaleRefused(
                    f"duplicate output_stem {stem!r} in {pair_path}")
            if not row.get("left_path") or not row.get("right_path"):
                raise ScaleRefused(
                    f"{stem}: the pair manifest row lacks a path")
            pairs[stem] = {
                "split": str(row["split"]).strip(),
                "group_name": str(row["group_name"]).strip(),
                "left_path": str(row["left_path"]).strip(),
                "right_path": str(row["right_path"]).strip(),
            }
    if not pairs:
        raise ScaleRefused(f"the source pair manifest is empty: {pair_path}")
    dev_frames = sorted(stem for stem, pair in pairs.items()
                        if pair["split"] in splits)
    frames_with_samples = sorted({row["source_frame"] for row in dev_rows})
    empty_frames = [stem for stem in dev_frames
                    if stem not in set(frames_with_samples)]
    groups = sorted({row["group_name"] for row in dev_rows})
    expectations = cfg["support_expectations"]
    checks = {
        "instances": (len(dev_rows), int(expectations["instances"])),
        "frames": (len(dev_frames), int(expectations["frames"])),
        "groups": (len(groups), int(expectations["groups"])),
        "empty_frames": (len(empty_frames),
                         int(expectations["empty_frames"])),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ScaleRefused(
                f"dev-pool census {name}: {actual} != pinned {expected}")
    return {
        "dev_rows": dev_rows,
        "dev_frames": dev_frames,
        "frames_with_samples": frames_with_samples,
        "empty_frames": empty_frames,
        "groups": groups,
        "pair_manifest": pairs,
        "counts": {name: actual for name, (actual, _e) in checks.items()},
    }


def run_v5_admission(cfg: dict) -> dict:
    """The project's own V5 admission gate (fingerprint integrity).

    The gate re-digests the WHOLE dataset including test-split assets;
    that is an integrity read only — the old test contributes no sample,
    statistic, training row or cache row. Disclosed in every report.
    """
    report = admit_dataset_for_protocol(
        _PROJECT_ROOT / cfg["data"]["root"],
        pooling_mode="instance_mask",
        expected_fingerprint=cfg["data"]["fingerprint"],
        expected_parent_fingerprint=cfg["data"]["parent_v4_fingerprint"],
        expected_mask_schema=MASK_SCHEMA_VERSION,
        expected_mask_resize_version=MASK_RESIZE_VERSION)
    if report["audit_fingerprint"] != cfg["data"]["fingerprint"]:
        raise ScaleRefused(
            "V5 admission fingerprint does not match the pinned "
            "fingerprint")
    return report


def register_global_cache(registry: SourceRegistry, cfg: dict,
                          protocol: dict) -> dict:
    """Verify the frozen global cache and register every consumed file
    (the binding record is the frozen expectation)."""
    import scripts.verify_p0_baseline as baseline

    global_dir = _PROJECT_ROOT / cfg["features_source"]["path"]
    # registration (each file hashed against the expanded historical
    # expectations) happens BEFORE the consumer reads the cache: a
    # drifted file is refused here and never reaches verify_feature_cache
    for name in ("cache_binding.json", "feature_cache.npz",
                 "feature_manifest.json", "feature_definition.json"):
        registry.register(global_dir / name, f"global_cache:{name}")
    binding, definition, manifest, arrays, authority = \
        baseline.verify_feature_cache(global_dir, protocol)
    if definition.get("data_fingerprint") != cfg["data"]["fingerprint"]:
        raise ScaleRefused(
            "global cache data fingerprint does not match the config")
    return {"binding": binding, "definition": definition,
            "manifest": manifest, "arrays": arrays,
            "authority": authority, "dir": global_dir}


def register_p0_fit(registry: SourceRegistry, cfg: dict) -> Path:
    """Register the frozen P0 fit's summary/admission and per-fold P0
    model files."""
    fit_dir = _PROJECT_ROOT / cfg["p0_baseline_source"]["fit_dir"]
    if not fit_dir.is_dir():
        raise ScaleRefused(f"P0 fit directory missing: {fit_dir}")
    if not (fit_dir / "summary.json").is_file():
        raise ScaleRefused(
            f"P0 fit summary missing: {fit_dir / 'summary.json'}")
    for name in ("summary.json", "admission.json"):
        path = fit_dir / name
        if path.is_file():
            registry.register(path, f"p0_fit:{name}")
    models = fit_dir / "models"
    for fold_dir in sorted(models.glob("*_P0")):
        for name in ("model_state.pt", "scaler.json",
                     "outer_eval_predictions.jsonl"):
            path = fold_dir / name
            if path.is_file():
                registry.register(path,
                                  f"p0_model:{fold_dir.name}:{name}")
    return fit_dir


def register_v5_records(registry: SourceRegistry, cfg: dict) -> Path:
    """Register the V5 dataset bookkeeping records."""
    data_root = _PROJECT_ROOT / cfg["data"]["root"]
    for name in ("dataset_manifest.csv", "dataset_summary.json",
                 "dataset_audit.json"):
        path = data_root / name
        if not path.is_file():
            raise ScaleRefused(f"V5 record missing: {path}")
        registry.register(path, f"v5_record:{name}")
    return data_root


def run_preflight(ctx, cfg: dict) -> dict:
    """Identity/config admission and the dev-pool census; no feature
    computation, no per-sample reads, no model loads."""
    import scripts.verify_p0_baseline as baseline

    registry = ctx["registry"]
    protocol_path = _PROJECT_ROOT / cfg["protocol_source"]["path"]
    registry.register(protocol_path, "protocol_file")
    protocol = baseline.load_protocol(protocol_path)
    if protocol["data"]["fingerprint"] != cfg["data"]["fingerprint"]:
        raise ScaleRefused(
            "protocol data fingerprint does not match the config")
    # registration (with hash verification against the expanded
    # historical expectations) happens BEFORE the consumers
    global_info = register_global_cache(registry, cfg, protocol)
    register_v5_records(registry, cfg)
    v5_report = run_v5_admission(cfg)
    register_p0_fit(registry, cfg)
    census = census_dev_pool(cfg, registry)
    authority = global_info["authority"]
    if set(authority) != {row["sample_name"]
                          for row in census["dev_rows"]}:
        raise ScaleRefused(
            "the protocol authority does not cover exactly the dev pool")
    ctx["stage"]("v5_admission", "COMPLETED",
                 fingerprint=v5_report["audit_fingerprint"],
                 disclosure=("the gate re-digests the whole V5 dataset "
                             "including test assets; integrity read only"))
    ctx["stage"]("preflight", "COMPLETED", **census["counts"])
    return {"protocol": protocol, "global_info": global_info,
            "census": census, "v5_report": v5_report}


def run_feature_build(ctx, cfg: dict, census: dict, global_info: dict,
                      frame_limit: int | None) -> dict:
    """The fix-version feature rebuild over the dev pool (shared by smoke
    and build). Gray crops, instance masks, classes and G256 are the
    frozen identities; signed_q/abs_q/polar_valid/quality and the C14/S16
    statistics are regenerated through the production fix path."""
    registry = ctx["registry"]
    watch = ctx["watch"]
    matcher_params = matcher_config_of(cfg)
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
        lr_check_threshold_px=float(matcher_params["lr_check_threshold_px"]),
        min_valid_ratio=float(matcher_params["min_valid_ratio"]),
        upsample_version=UPSAMPLE_FIX_VERSION,
    )
    if stereo_cfg.upsample_version != UPSAMPLE_FIX_VERSION:
        raise ScaleRefused("the fix matcher did not receive the fix version")
    matcher = StereoMatcher(stereo_cfg)
    band_margin = int(cfg["fix_inputs"]["band_margin"])
    full_threshold = float(cfg["fix_inputs"]["full_image_threshold"])
    crop_pad = int(cfg["fix_inputs"]["crop_pad"])

    dev_rows = census["dev_rows"]
    by_sample = {row["sample_name"]: row for row in dev_rows}
    manifest_samples = global_info["manifest"]["samples"]
    names = [row["sample_name"] for row in manifest_samples]
    if names != sorted(names):
        raise ScaleRefused(
            "the global cache manifest is not sorted by sample_name")
    if set(names) != set(by_sample):
        raise ScaleRefused(
            "the global cache manifest sample set differs from the V5 "
            "dev pool")
    g_source = np.asarray(global_info["arrays"]["g_features"])
    old_s_valid = np.asarray(global_info["arrays"]["s_valid"]).astype(bool)
    n = len(names)
    row_of = {name: index for index, name in enumerate(names)}

    frames_to_build = census["frames_with_samples"]
    if frame_limit is not None:
        frames_to_build = frames_to_build[:frame_limit]
    frame_rows: dict[str, list[dict]] = {}
    for row in dev_rows:
        frame_rows.setdefault(row["source_frame"], []).append(row)

    s_fix = np.full((n, 16), np.nan, dtype=np.float32)
    c_fix = np.zeros((n, 14), dtype=np.float32)
    region_by_name: dict[str, dict] = {}
    fix_provenance: dict[str, dict] = {}
    frame_ledger = []
    missing_changed = []
    built_instances = 0

    for frame_index, stem in enumerate(frames_to_build):
        watch.check(f"frame {stem}")
        rows = frame_rows[stem]
        first = rows[0]
        left_path = _PROJECT_ROOT / first["left_path"]
        right_rel = census["pair_manifest"][stem]["right_path"]
        right_path = _PROJECT_ROOT / right_rel
        label_path = _PROJECT_ROOT / first["label_path"]
        registry.register(left_path, f"image_left:{stem}")
        registry.register(right_path, f"image_right:{stem}")
        registry.register(label_path, f"label:{stem}")
        left = load_gray(left_path)
        right = load_gray(right_path)
        height, width = left.shape[:2]
        geometry = build_gt_geometry(label_path, width, height,
                                     padding=crop_pad)
        if len(geometry) != len(rows):
            raise ScaleRefused(
                f"frame {stem}: {len(geometry)} GT instances != "
                f"{len(rows)} V5 rows")
        bands = build_horizontal_bands(
            [item.band_bbox for item in geometry], image_height=height,
            vertical_margin=band_margin)
        result = matcher.compute_bands(left, right, bands,
                                       full_image_threshold=full_threshold)
        for item in geometry:
            name = f"{stem}_obj{item.object_index:03d}"
            v5_row = by_sample[name]
            if v5_row["split"] not in cfg["data"]["dev_pool_splits"]:
                raise ScaleRefused(f"{name}: not a dev-pool sample")
            v5_path = _PROJECT_ROOT / cfg["data"]["root"] / \
                v5_row["npz_path"]
            registry.register(v5_path, f"v5_sample:{name}")
            v5_sample = load_mask_sample(v5_path)
            if v5_sample.class_id != item.class_id:
                raise ScaleRefused(
                    f"{name}: class {v5_sample.class_id} != GT "
                    f"{item.class_id}")
            window = (int(v5_row["crop_x1"]), int(v5_row["crop_y1"]),
                      int(v5_row["crop_x2"]), int(v5_row["crop_y2"]))
            if window != item.crop_window:
                raise ScaleRefused(
                    f"{name}: V5 crop window {window} != GT geometry "
                    f"{item.crop_window}")
            x1, y1, x2, y2 = window
            if not np.array_equal(v5_sample.gray, left[y1:y2, x1:x2]):
                raise ScaleRefused(
                    f"{name}: the V5 gray crop is not the rectified left "
                    "crop; the rebuild contract is violated")
            if not np.array_equal(v5_sample.instance_mask,
                                  item.mask[y1:y2, x1:x2]):
                raise ScaleRefused(
                    f"{name}: the V5 instance mask is not the GT "
                    "rasterisation on the crop window")
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
                crop_valid = polar_result.valid_mask[
                    y1:y2, x1:x2].astype(np.uint8)
                if int(crop_valid.sum()) != int(polar_result.valid_mask.sum()):
                    raise ScaleRefused(
                        f"{name}: the crop window cuts valid polar "
                        "pixels; the mean_abs_q audit cannot verify such "
                        "samples")
                stereo_valid, stereo_reason = True, "ok"
            else:
                components = {key: 0.0 for key in QUALITY_VECTOR_KEYS}
                quality = build_quality_vector(**components)
                crop_signed = np.zeros((y2 - y1, x2 - x1), np.float32)
                crop_abs = np.zeros((y2 - y1, x2 - x1), np.float32)
                crop_valid = np.zeros((y2 - y1, x2 - x1), np.uint8)
                stereo_valid, stereo_reason = False, str(stats.reason)
            fix_arrays = MaskFusionSampleArrays(
                gray=v5_sample.gray,
                signed_q=crop_signed,
                abs_q=crop_abs,
                valid=crop_valid,
                quality=np.asarray(quality, dtype=np.float32),
                class_id=int(item.class_id),
                instance_mask=v5_sample.instance_mask)
            values, blocks = build_local_polar_stats(fix_arrays)
            s_fix[row_of[name]] = values
            c_fix[row_of[name]] = build_local_control_features(fix_arrays)
            region_by_name[name] = {
                **region_records(fix_arrays),
                "blocks": {key: bool(value)
                           for key, value in blocks.items()},
            }
            fix_provenance[name] = {
                "sample_name": name,
                "frame": stem,
                "upsample_version": UPSAMPLE_FIX_VERSION,
                "stereo_valid": stereo_valid,
                "stereo_reason": stereo_reason,
                "quality_components": {key: float(value) for key, value
                                       in components.items()},
                "crop_window": list(window),
                "interior_valid": bool(blocks["interior_valid"]),
                "band_valid": bool(blocks["band_valid"]),
            }
            built_instances += 1
        frame_ledger.append({
            "frame": stem, "split": first["split"],
            "group": first["group_name"], "instances": len(rows),
            "note": "band-restricted production SGBM pass (fix version)",
        })
        ctx["log"](f"frames {frame_index + 1}/{len(frames_to_build)}: "
                   f"{stem} ({len(rows)} instances)")

    if frame_limit is None:
        for name in names:
            if name not in fix_provenance:
                raise ScaleRefused(
                    f"{name}: the frame was not built (frames built "
                    f"{len(frames_to_build)})")
        kept_names = names
        kept_rows = list(range(n))
    else:
        # a bounded (smoke) build keeps ONLY the built frames' samples;
        # the cache is marked incomplete and refused by training anyway
        kept_names = [name for name in names
                      if name in fix_provenance]
        kept_rows = [row_of[name] for name in kept_names]
    for name in kept_names:
        row = row_of[name]
        new_valid = not bool(np.isnan(s_fix[row]).all())
        if new_valid != bool(old_s_valid[row]):
            missing_changed.append({
                "sample_name": name,
                "old_s_valid": bool(old_s_valid[row]),
                "fix_s_valid": new_valid,
            })

    # the row order IS the global order, so G is carried over untouched
    class_id = np.asarray([row["class_id"] for name in kept_names
                           for row in [by_sample[name]]],
                          dtype=np.int64)
    if not np.array_equal(
            class_id,
            np.asarray([int(manifest_samples[row_of[name]]["class_id"])
                        for name in kept_names], dtype=np.int64)):
        raise ScaleRefused("class rows disagree with the V5 dev pool")
    s_valid = ~np.isnan(s_fix[kept_rows]).all(axis=1)
    cache_arrays = {
        "g_features": np.asarray(g_source[kept_rows], dtype=np.float32),
        "c_features": np.ascontiguousarray(c_fix[kept_rows]),
        "s_features": np.ascontiguousarray(s_fix[kept_rows]),
        "s_valid": s_valid,
        "class_id": class_id,
    }
    kept_manifest = [manifest_samples[row] for row in kept_rows]
    kept_region = {name: region_by_name[name] for name in kept_names}
    kept_provenance = {name: fix_provenance[name]
                       for name in kept_names}
    return {
        "cache_arrays": cache_arrays,
        "manifest_samples": kept_manifest,
        "region_by_name": kept_region,
        "fix_provenance": kept_provenance,
        "frame_ledger": frame_ledger,
        "missing_changed": missing_changed,
        "built_frames": len(frames_to_build),
        "built_instances": built_instances,
    }


def write_scale_cache(feature_dir: Path, cfg: dict, protocol: dict,
                      build: dict, *, kind: str, complete: bool,
                      interrupted: bool, data_fingerprint: str,
                      protocol_sha256: str, empty_frames: list[str]) -> None:
    """Write the cache (npz + manifest + definition + binding) into a
    fresh directory; the complete full cache self-verifies the generic
    contract (binding bytes, manifest identity, arrays)."""
    from core.material_probe import build_cache_binding
    from core.polar_local_features import verify_local_cache

    feature_dir.mkdir(parents=True)
    np.savez_compressed(feature_dir / "feature_cache.npz",
                        **build["cache_arrays"])
    samples = []
    for row in build["manifest_samples"]:
        name = row["sample_name"]
        samples.append({**dict(row),
                        **build["region_by_name"][name],
                        "fix_polar": build["fix_provenance"][name]})
    _write_json(feature_dir / "feature_manifest.json",
                {"count": len(samples), "samples": samples})
    definition = scale_cache_definition(
        cfg, kind=kind, complete=complete,
        frames_built=build["built_frames"],
        instances_built=build["built_instances"],
        empty_frames=empty_frames,
        data_fingerprint=data_fingerprint,
        protocol_sha256=protocol_sha256,
        g_dim=int(build["cache_arrays"]["g_features"].shape[1]))
    definition["interrupted"] = bool(interrupted)
    _write_json(feature_dir / "feature_definition.json", definition)
    build_cache_binding(feature_dir, protocol)
    if kind == CACHE_KIND_FULL and complete and not interrupted:
        verify_local_cache(feature_dir, protocol)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                               sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="material_residual_scale_v1 fix-version feature "
                    "preparation (preflight / smoke / build).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; must not exist")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="smoke only: cap the dev frames")
    parser.add_argument("--deadline-utc", required=True,
                        help="ISO-8601 instant bounding the whole flow")
    args = parser.parse_args(argv)
    if args.max_frames is not None and (
            type(args.max_frames) is not int or args.max_frames <= 0):
        print("refused: --max-frames must be a positive int (smoke only)",
              file=sys.stderr)
        return EXIT_REFUSED
    if args.mode != "smoke" and args.max_frames is not None:
        print("refused: --max-frames is only valid for --mode smoke",
              file=sys.stderr)
        return EXIT_REFUSED

    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
              f"{message}", file=sys.stderr, flush=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    try:
        cfg = load_scale_config(config_path)
    except ScaleRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    if output_dir.exists():
        print(f"refused: output directory already exists: {output_dir}",
              file=sys.stderr)
        return EXIT_REFUSED
    if is_link_or_reparse(output_dir) or \
            is_link_or_reparse(output_dir.parent):
        print("refused: output directory or parent is a link/reparse point",
              file=sys.stderr)
        return EXIT_REFUSED

    registry = SourceRegistry({})
    watch = DeadlineWatch(deadline_utc=args.deadline_utc)
    stages: dict = {}

    def stage(name: str, status: str, **details) -> None:
        stages[name] = {"status": status, **details}
        log(f"stage {name}: {status}")

    ctx = {"registry": registry, "watch": watch, "stage": stage,
           "log": log}
    failure = None
    interrupted = False
    protection = None
    output_dir.mkdir(parents=True)
    try:
        try:
            registry.register(config_path, "experiment_config",
                              kind="experiment_config")
            for name, anchor in cfg["anchors"].items():
                verify_anchor(_PROJECT_ROOT / anchor["path"],
                              anchor["sha256"], registry, f"anchor:{name}")
            stage("anchors", "COMPLETED", names=sorted(cfg["anchors"]))
            # historical expectations are expanded BEFORE any consumer
            # runs; the expansion itself registers the binding records it
            # reads (batch-1 manifest is already an anchor)
            ctx["watch"].check("before historical binding expansion")
            expansion = expand_historical_bindings(registry, cfg,
                                                   _PROJECT_ROOT)
            stage("historical_bindings", "COMPLETED",
                  **expansion["added"])
            ctx["watch"].check("before the dev-pool census")
            pre = run_preflight(ctx, cfg)
            protocol = pre["protocol"]
            if args.mode == "preflight":
                ctx["watch"].check("before the preflight plan publish")
                dump_json(output_dir / "preflight_plan.json", {
                    "run_mode": "preflight",
                    "experiment": EXPERIMENT,
                    "anchors": sorted(cfg["anchors"]),
                    "dev_pool": pre["census"]["counts"],
                    "empty_frames": pre["census"]["empty_frames"],
                    "upsample_version": UPSAMPLE_FIX_VERSION,
                    "v5_disclosure": (
                        "the admission gate re-digests the whole V5 "
                        "dataset including test assets; integrity only"),
                    "note": "preflight ran no feature computation",
                })
            else:
                if args.mode == "smoke":
                    kind, complete, frame_limit = \
                        CACHE_KIND_SMOKE, False, int(args.max_frames)
                else:
                    kind, complete, frame_limit = \
                        CACHE_KIND_FULL, True, None
                ctx["watch"].check("before the fix rebuild")
                build = run_feature_build(
                    ctx, cfg, pre["census"], pre["global_info"],
                    frame_limit)
                feature_dir = output_dir / "features"
                try:
                    write_scale_cache(
                        feature_dir, cfg, protocol, build, kind=kind,
                        complete=complete, interrupted=False,
                        data_fingerprint=cfg["data"]["fingerprint"],
                        protocol_sha256=protocol["protocol_sha256"],
                        empty_frames=pre["census"]["empty_frames"])
                except DeadlineExceeded:
                    raise
                dump_json(output_dir / "build_report.json", {
                    "run_mode": args.mode,
                    "experiment": EXPERIMENT,
                    "cache_kind": kind,
                    "complete": complete,
                    "feature_dir": str(feature_dir),
                    "upsample_version": UPSAMPLE_FIX_VERSION,
                    "matcher": matcher_config_of(cfg),
                    "built_frames": build["built_frames"],
                    "built_instances": build["built_instances"],
                    "empty_frames": pre["census"]["empty_frames"],
                    "s_valid_changes_vs_old": {
                        "count": len(build["missing_changed"]),
                        "rows": build["missing_changed"][:200],
                        "note": ("the fix inputs may change the "
                                 "polarization missing state; the change "
                                 "is reported, never forced back to the "
                                 "old state"),
                    },
                    "g_identity": ("copied row-for-row from the verified "
                                   "frozen global cache by sample_name; "
                                   "bytes identical"),
                    "v5_admission_disclosure": (
                        "admit_dataset_for_protocol re-digests the WHOLE "
                        "V5 dataset including test-split assets; the old "
                        "test contributes NO sample, statistic, training "
                        "row or cache row"),
                    "region_rule": LOCAL_REGION_RULE,
                    "feature_definition_version":
                        LOCAL_FEATURE_DEFINITION_VERSION,
                })
                dump_jsonl(output_dir / "frame_ledger.jsonl",
                           build["frame_ledger"])
                ctx["watch"].check("before the cache publish")
                stage("features", "COMPLETED",
                      frames=build["built_frames"],
                      instances=build["built_instances"])
        except DeadlineExceeded as exc:
            interrupted = True
            ctx["watch"] = None  # closeout runs deadline-exempt
            log(f"deadline exceeded: {exc}")
            if args.mode in ("smoke", "build") and \
                    (output_dir / "features").is_dir() and \
                    not (output_dir / "features" /
                         "feature_definition.json").exists():
                # keep the deadline-interrupted build's partial evidence:
                # no definition was written yet, so write an explicit
                # interrupted marker definition (still refused by the
                # training entry)
                try:
                    _write_json(output_dir / "features" /
                                "feature_definition.json", {
                                    "experiment": EXPERIMENT,
                                    "cache_kind": CACHE_KIND_SMOKE,
                                    "complete": False,
                                    "interrupted": True,
                                    "note": ("the build was interrupted "
                                             "by the shared deadline; "
                                             "this cache is never "
                                             "consumed for training")})
                except OSError:
                    pass
        except (ScaleRefused, IdentityError, FileNotFoundError,
                ValueError, OSError, KeyError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            log(f"failure: {failure}")
    except Exception as exc:  # unexpected: still write a status file
        failure = f"UNEXPECTED {type(exc).__name__}: {exc}"
        log(f"unexpected failure: {failure}")

    # closeout: the protection manifest is evidence (kept after an
    # expiry), but the SUCCESS STATUS needs its own deadline check —
    # the protection verification itself crossing the deadline must
    # never yield a COMPLETED run
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
    if not interrupted and ctx["watch"] is not None:
        try:
            ctx["watch"].check("before success status publish")
        except DeadlineExceeded as exc:
            interrupted = True
            ctx["watch"] = None  # only the status write remains
            log(f"deadline crossed before the success status: {exc}")
            if failure is None and args.mode == "build" and                     (output_dir / "build_report.json").is_file():
                log("the completed cache stays on disk but the run is "
                    "PARTIAL, never a success")
    if failure:
        status, code = "FAILED", EXIT_REFUSED
    elif interrupted:
        status, code = "PARTIAL", EXIT_PARTIAL
    else:
        status, code = "COMPLETED", EXIT_OK
    dump_json(output_dir / "run_status.json", {
        "run_mode": args.mode, "experiment": EXPERIMENT,
        "stages": stages, "interrupted_by_deadline": interrupted,
        "failure": failure, "overall_status": status, "exit_code": code,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started,
        "protection": protection,
    })
    log(f"overall: {status} (exit {code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
