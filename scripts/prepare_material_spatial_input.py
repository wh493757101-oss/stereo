"""material_spatial_input_v1 — the native-pixel/grid cache builder.

Zero training: no model fit, no scaler fit, no classifier or encoder
forward, no optimizer step. The only pixel work is the production CPU
stereo recipe (valid_weighted_legacy_support_v1) and the q rebuild.

Modes:
- preflight: admission + census + smoke selection plan; no SGBM/warp.
- smoke: the same build over at most 12 deterministic frames
  (input_admitted=false, smoke kind).
- build: the full 941-frame dev pool; completes as BUILT_UNVERIFIED —
  only the independent verifier + wrapper publish input_admitted=true.

Exit codes: 0 COMPLETED/BUILT_UNVERIFIED, 1 REFUSED/FAILED, 2 PARTIAL.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from core.material_spatial_input import (  # noqa: E402
    EXPERIMENT,
    GRID_PLANES,
    area_average_sat,
    GRID_SCHEMA,
    GRID_SIZE,
    PERMUTATION_SCHEMA,
    NativeInstanceArrays,
    SpatialRefused,
    apply_q_permutation,
    canvas_offset,
    coverage_bits_hex8,
    coverage_from_overlap,
    exact_cell_overlap,
    plan_q_permutation,
    scaled_size,
    validate_grid_planes,
)
from core.material_spatial_input_admission import (  # noqa: E402
    CACHE_SCHEMA,
    IMPLEMENTATION_REVISION,
    KIND_FULL,
    KIND_SMOKE,
    NATIVE_ARRAY_KEYS,
    AdmissionRefused,
    ExpectationBook,
    build_cache_binding,
    canonical_rel,
    check_consumed_path,
    npz_key_records,
    register_consumed,
    run_v5_admission,
    dev_census,
    validate_new_run_root,
)
from core.material_residual_scale import (  # noqa: E402
    DeadlineExceeded,
    DeadlineWatch,
    ScaleRefused,
    UPSAMPLE_FIX_VERSION,
    matcher_config_of,
    verify_scale_cache_markers,
)
from core.material_mask_dataset import (  # noqa: E402
    MASK_RESIZE_VERSION,
    MASK_SCHEMA_VERSION,
    MaskFusionSampleArrays,
    admit_dataset_for_protocol,
    load_mask_sample,
)
from core.material_probe import (  # noqa: E402
    authoritative_sample_index,
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
    to_gray_u8,
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
    is_link_or_reparse,
    sha256_of_file,
    verify_anchor,
)
from scripts.prepare_cls_fusion_dataset import load_gray  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_PARTIAL = 2

# the batch config's frozen bytes; registration verifies the CURRENT
# file against this pin BEFORE the config is parsed (parser-call-count
# zero on any content drift)
PINNED_CONFIG_SHA256 = "0761eb9f11b484f68d85df3c444a4c203f8de0f877dc72d5a57473e727b6314e"


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1)
                    + chr(10), encoding="utf-8")


def _write_jsonl(path: Path, rows: list) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text(chr(10).join(lines)
                    + (chr(10) if lines else ""), encoding="utf-8")

OUTPUT_ROOT = "analysis/runs/material_spatial_input_v1"
PINNED_CONFIG_REL = "configs/experiments/material_spatial_input_v1.yaml"
OLD_CONFIG_REL = "configs/experiments/material_residual_scale_v1.yaml"

# matcher/geometry keys that must match the frozen old config item by item
FROZEN_RECIPE_KEYS = (
    "matcher", "mode", "max_disp", "scale", "block_size",
    "uniqueness_ratio", "speckle_window", "speckle_range",
    "texture_threshold", "lr_check_threshold_px", "min_valid_ratio",
    "band_margin", "full_image_threshold", "crop_pad",
)


def registered_right_crop(right_u8: np.ndarray, disparity: np.ndarray,
                          window: tuple[int, int, int, int]) -> np.ndarray:
    """The registered right crop, mirroring compute_polar_features'
    exact sampling: float32 right image, map_x = x - d (float32, NO
    disparity clipping), INTER_LINEAR, BORDER_CONSTANT (default border
    value 0). The invalid positions are zeroed by the caller through V.
    This deliberately does NOT use warp_with_disparity, which clips the
    disparity first."""
    right_f = to_gray_u8(right_u8).astype(np.float32)
    h, w = right_f.shape
    disp = np.asarray(disparity, dtype=np.float32)
    if disp.shape != right_f.shape:
        raise ScaleRefused("disparity shape != right image shape")
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    map_x = xx - disp
    warped = cv2.remap(right_f, map_x, yy,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    x1, y1, x2, y2 = window
    return warped[y1:y2, x1:x2]


def _build_grid_planes(L: np.ndarray, R: np.ndarray, q: np.ndarray,
                       M: np.ndarray, V: np.ndarray,
                       interior: np.ndarray, band: np.ndarray) -> dict:
    """The builder-side grid: nine float32 planes. Responses use the SAT
    area form; support and coverage come from the exact integer overlap
    geometry. R is normalized to the gray scale /255 like L; the native
    arrays are never modified in place (every conversion copies)."""
    valid = V > 0
    if not np.all(valid <= (M > 0)):
        raise ScaleRefused("V must be a subset of M")
    w, h = L.shape[1], L.shape[0]
    w2, h2 = scaled_size(w, h)
    offset = canvas_offset(w2, h2)
    a_interior = (interior > 0) & valid
    a_band = (band > 0) & valid
    sources = {"L": L.astype(np.float32) / np.float32(255.0),
               "R": R.astype(np.float32) / np.float32(255.0),
               "q": np.asarray(q, dtype=np.float32)}
    planes: dict[str, np.ndarray] = {}
    for region_name, a_mask in (("interior", a_interior),
                                ("band", a_band)):
        overlap = exact_cell_overlap(a_mask.astype(np.uint8), w, h,
                                     w2, h2, offset)
        planes[f"coverage_{region_name}"] = coverage_from_overlap(
            overlap, w, h)
        for source_name, src in sources.items():
            bounds = (-1.0, 1.0) if source_name == "q" else (0.0, 1.0)
            response, _denom = area_average_sat(src, a_mask, w2, h2,
                                                offset, bounds=bounds)
            planes[f"{source_name}_{region_name}"] = response
    overlap_object = exact_cell_overlap((M > 0).astype(np.uint8), w, h,
                                        w2, h2, offset)
    planes["coverage_object"] = coverage_from_overlap(overlap_object,
                                                      w, h)
    if set(planes) != set(GRID_PLANES):
        raise ScaleRefused("the grid plane set differs from the contract")
    validate_grid_planes(planes, w2, h2, offset)
    return {"planes": planes, "offset": list(offset), "w2": w2, "h2": h2}


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = validate_new_run_root(output_dir, _PROJECT_ROOT)
    root = (_PROJECT_ROOT / OUTPUT_ROOT).resolve()
    # a STRICT subdirectory of the experiment root: the root itself,
    # sibling trees and similar-prefix paths are refused
    if resolved == root or root not in resolved.parents:
        raise ScaleRefused(
            f"output directory {resolved} is not a strict subdirectory "
            f"of the experiment root {root}")
    return resolved


def _verify_and_register_pinned(ctx: dict, cfg: dict) -> dict:
    registry = ctx["registry"]
    book: ExpectationBook = ctx["book"]
    pinned_paths = {}
    for name, record in cfg["frozen_sources"]["fix_cache_files"].items():
        rel = f"{cfg['frozen_sources']['fix_cache_dir']}/{name}"
        pinned_paths[rel] = record["sha256"]
    pinned_paths[cfg["frozen_sources"]["old_experiment_config"]["path"]] =         cfg["frozen_sources"]["old_experiment_config"]["sha256"]
    pinned_paths[cfg["frozen_sources"]["build_source_manifest"]["path"]] =         cfg["frozen_sources"]["build_source_manifest"]["sha256"]
    pinned_paths[cfg["frozen_sources"]["protocol"]["path"]] =         cfg["frozen_sources"]["protocol"]["file_sha256"]
    for rel, expected in pinned_paths.items():
        path = _PROJECT_ROOT / rel
        if not path.is_file():
            raise ScaleRefused(f"pinned input missing: {path}")
        actual = sha256_of_file(path)
        if actual != expected:
            raise ScaleRefused(
                f"{rel}: sha256 {actual} != the pinned {expected}; "
                "refusing (expectations are never updated)")
    registered = {}
    for rel, expected in pinned_paths.items():
        book.add(_PROJECT_ROOT / rel, "config.pin", expected)
        verify_anchor(_PROJECT_ROOT / rel, expected, registry,
                      "anchor:spatial_input")
        registered[rel] = expected
    ctx["log"](f"pinned inputs verified and registered: {len(registered)}")
    return registered


def _expand_historical_expectations(ctx: dict, cfg: dict) -> dict:
    """Historical expectations from the verified build source manifest
    and the frozen fix cache binding, merged through the canonical
    expectation book (a conflicting sha for the same canonical path is
    refused). Only rows the manifest itself proves unchanged with
    matching registration/end digests and a valid status become
    historical expectations; anything else this run consumes is a
    new_snapshot of its own."""
    registry = ctx["registry"]
    book: ExpectationBook = ctx["book"]
    manifest_rel = cfg["frozen_sources"]["build_source_manifest"]["path"]
    manifest_path = _PROJECT_ROOT / manifest_rel
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expanded = {"build_source_manifest": 0, "fix_cache_binding": 0}
    for item in manifest.get("items", []):
        if item.get("status") != "ok" or item.get("unchanged") is not True:
            continue
        digest = item.get("sha256_at_registration")
        end_digest = item.get("sha256_at_end")
        rel = item.get("path")
        if not digest or not rel or digest != end_digest:
            continue
        if book.add(_PROJECT_ROOT / rel, "build_source_manifest",
                    digest)["added"]:
            expanded["build_source_manifest"] += 1
    # the frozen fix cache's own binding record covers its files
    feature_dir = _PROJECT_ROOT / cfg["frozen_sources"]["fix_cache_dir"]
    binding = json.loads((feature_dir / "cache_binding.json").read_text(
        encoding="utf-8"))
    for name, record in (binding.get("files") or {}).items():
        if book.add(feature_dir / name, "fix_cache_binding",
                    record["sha256"])["added"]:
            expanded["fix_cache_binding"] += 1
    # every book expectation becomes a registry binding so a later
    # registration verifies the CURRENT bytes against history; the
    # registry keys by its own raw rel string, so seed under that
    for rel, expected in book.expected.items():
        raw_rel = expected.get("rel_raw", rel)
        existing = registry.bindings.get(raw_rel)
        if existing is not None and                 existing.get("expected_sha256") !=                 expected["expected_sha256"]:
            raise ScaleRefused(
                f"{raw_rel}: conflicting historical expectations "
                f"({existing.get('source')} vs "
                f"{expected.get('source')})")
        if existing is None:
            registry.bindings[raw_rel] = dict(expected)
    ctx["log"](f"historical expectations expanded: {expanded}")
    return expanded


def _census(ctx: dict, cfg: dict, registry: SourceRegistry) -> dict:
    """Delegate to the shared dev-pool census (identical rules)."""
    return dev_census(cfg, registry, ctx["book"], _PROJECT_ROOT)


def _run_v5_gate(ctx: dict, cfg: dict) -> dict:
    """Delegate to the shared full V5 admission (identical rules for the
    builder and the verifier)."""
    return run_v5_admission(ctx["registry"], ctx["book"], ctx["protocol"],
                            cfg, _PROJECT_ROOT, log=ctx["log"])


def _smoke_selection(ctx: dict, cfg: dict, census: dict,
                     cache_arrays: dict, manifest_samples: list) -> dict:
    """Deterministic smoke selection from the frozen metadata only: the
    four empty frames plus metadata extremes (aspect ratio, largest
    crop, empty S block, partial block validity), ties by sample_name,
    deduplicated by frame, at most --max-frames."""
    limit = int(ctx["max_frames"] or 12)
    if limit < len(census["empty_frames"]):
        raise ScaleRefused(
            "--max-frames cannot be smaller than the required empty frames")
    row_of = {sample["sample_name"]: index
              for index, sample in enumerate(manifest_samples)}
    s_features = cache_arrays["s_features"]
    candidates = []
    for sample in manifest_samples:
        name = sample["sample_name"]
        window = sample.get("fix_polar", {}).get("crop_window") or [0, 0, 0, 0]
        w = max(1, int(window[2]) - int(window[0]))
        h = max(1, int(window[3]) - int(window[1]))
        candidates.append({
            "sample_name": name, "frame": sample["source_frame"],
            "w": w, "h": h, "area": w * h,
            "aspect_wh": w / h, "aspect_hw": h / w,
            "s_valid": bool(sample["has_valid_s"]),
            "blocks": sample.get("blocks", {}),
        })
        candidates[-1].update({"row": row_of[name]})
    picked = []       # (reason, sample_name, frame)
    picked_names = set()
    picked_frames = set()

    def pick(reason: str, name: str) -> None:
        if name in picked_names:
            return
        sample = next(c for c in candidates
                      if c["sample_name"] == name)
        if sample["frame"] in picked_frames:
            return
        picked.append((reason, name, sample["frame"]))
        picked_names.add(name)
        picked_frames.add(sample["frame"])

    selection = {"picked": picked, "empty_frames": list(
        census["empty_frames"]), "missing_conditions": []}
    remaining = limit - len(census["empty_frames"])
    conditions = [
        ("max_aspect_wh", lambda c: (-c["aspect_wh"], c["sample_name"])),
        ("max_aspect_hw", lambda c: (-c["aspect_hw"], c["sample_name"])),
        ("largest_crop", lambda c: (-c["area"], c["sample_name"])),
        ("s_valid_false", lambda c: (c["sample_name"],)
         if not c["s_valid"] else None),
        ("partial_block", lambda c: (c["sample_name"],)
         if c["blocks"] and (c["blocks"].get("interior_valid")
                             != c["blocks"].get("band_valid")) else None),
    ]
    for reason, key in conditions:
        if len(picked) >= remaining:
            selection["missing_conditions"].append(
                f"{reason}: the frame budget is exhausted")
            continue
        pool = sorted((c for c in candidates), key=lambda c: c["sample_name"])
        if reason in ("s_valid_false", "partial_block"):
            pool = [c for c in pool
                    if (reason == "s_valid_false"
                        and not c["s_valid"])
                    or (reason == "partial_block" and c["blocks"]
                        and (c["blocks"].get("interior_valid")
                             != c["blocks"].get("band_valid")))]
            if not pool:
                selection["missing_conditions"].append(
                    f"{reason}: no matching sample in the frozen metadata")
                continue
            chosen = pool[0]
        else:
            ranked = sorted(pool, key=key)
            chosen = ranked[0]
        pick(reason, chosen["sample_name"])
    if len(picked) < remaining:
        selection["missing_conditions"].append(
            "more non-empty slots remained than distinct conditions")
    if len(picked_frames) + len(selection["empty_frames"]) > limit:
        raise ScaleRefused("the smoke selection exceeds --max-frames")
    return selection


def _check_build_disk_space(ctx: dict) -> None:
    """Check the actual output volume immediately before full pixel build."""
    path = ctx["output_dir"].resolve()
    required_gib = ctx["cfg"]["budgets"]["min_free_gib_before_build"]
    if type(required_gib) is not int or required_gib <= 0:
        raise ScaleRefused("min_free_gib_before_build must be a positive integer")
    record = {"path": str(path), "required_bytes": required_gib * 1024 ** 3}
    ctx["stage"]("disk_space", "STARTED", **record)
    ctx["watch"].check("before the build disk-space query")
    try:
        record["free_bytes"] = shutil.disk_usage(path).free
    except OSError as exc:
        ctx["stage"]("disk_space", "FAILED", **record, error=str(exc))
        raise
    if type(record["free_bytes"]) is not int or record["free_bytes"] < 0:
        ctx["stage"]("disk_space", "FAILED", **record)
        raise ScaleRefused("the disk-space query returned an invalid byte count")
    if record["free_bytes"] < record["required_bytes"]:
        ctx["stage"]("disk_space", "FAILED", **record)
        raise ScaleRefused(
            f"free disk space on {path}: {record['free_bytes']} bytes; "
            f"at least {record['required_bytes']} bytes required")
    ctx["watch"].check("after the build disk-space query")
    ctx["stage"]("disk_space", "COMPLETED", **record)


def run_admission(ctx: dict, args) -> dict:
    registry = ctx["registry"]
    book: ExpectationBook = ctx["book"]
    ctx["stage"]("admission", "STARTED")
    _verify_and_register_pinned(ctx, ctx["cfg"])
    # historical expectations expand BEFORE any consumer read of the
    # files they cover
    expanded = _expand_historical_expectations(ctx, ctx["cfg"])
    cfg = ctx["cfg"]
    old_config_path = _PROJECT_ROOT /         cfg["frozen_sources"]["old_experiment_config"]["path"]
    register_consumed(registry, old_config_path, "old_experiment_config",
                      _PROJECT_ROOT, book.binding_for(old_config_path))
    old_config = yaml.safe_load(
        old_config_path.read_text(encoding="utf-8"))
    # the recipe comes solely from the frozen old config (hash-pinned
    # above); the batch config deliberately does not duplicate it
    old_fix = old_config["fix_inputs"]
    for key in FROZEN_RECIPE_KEYS:
        if key not in old_fix:
            raise ScaleRefused(
                f"the frozen old config is missing fix_inputs.{key}")
    # protocol internal digest
    protocol_path = _PROJECT_ROOT / cfg["frozen_sources"]["protocol"][
        "path"]
    register_consumed(registry, protocol_path, "frozen_protocol",
                      _PROJECT_ROOT, book.binding_for(protocol_path))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ctx["protocol"] = protocol
    if protocol.get("protocol_sha256") !=             cfg["frozen_sources"]["protocol"]["internal_digest"]:
        raise ScaleRefused("the protocol internal digest does not match")
    authority = authoritative_sample_index(protocol)
    # the frozen fix cache: markers + binding + arrays identity, each
    # file registered against the book before its consuming read
    feature_dir = _PROJECT_ROOT / cfg["frozen_sources"]["fix_cache_dir"]
    for name in ("feature_definition.json", "feature_cache.npz",
                 "feature_manifest.json"):
        path = feature_dir / name
        register_consumed(registry, path, f"fix_cache:{name}",
                          _PROJECT_ROOT, book.binding_for(path))
    markers = verify_scale_cache_markers(feature_dir, old_config)
    definition = json.loads((feature_dir / "feature_definition.json")
                            .read_text(encoding="utf-8"))
    if definition.get("feature_definition_version") !=             old_config["local_features"]["feature_definition_version"]:
        raise ScaleRefused(
            "the fix cache local feature version does not match the "
            "frozen config")
    cache_arrays = dict(np.load(feature_dir / "feature_cache.npz"))
    manifest = json.loads((feature_dir / "feature_manifest.json")
                          .read_text(encoding="utf-8"))
    names = [sample["sample_name"] for sample in manifest["samples"]]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ScaleRefused("the fix cache manifest is not uniquely sorted")
    v5_report = _run_v5_gate(ctx, cfg)
    census = _census(ctx, cfg, registry)
    row_of = {sample["sample_name"]: index
              for index, sample in enumerate(manifest["samples"])}
    # the census identities must match the frozen cache manifest exactly
    census_names = sorted(row["sample_name"] for row in census["dev_rows"])
    if census_names != names:
        raise ScaleRefused("the dev pool differs from the frozen cache "
                           "manifest sample set")
    s_valid = np.asarray(cache_arrays["s_valid"], dtype=bool)
    true_count = int(s_valid.sum())
    if true_count != int(cfg["dev_pool"]["s_valid_true"]) or \
            len(s_valid) - true_count != int(cfg["dev_pool"]["s_valid_false"]):
        raise ScaleRefused("the frozen cache s_valid counts do not match "
                           "the pinned expectations")
    selection = _smoke_selection(ctx, cfg, census, cache_arrays,
                                 manifest["samples"])
    _write_json(ctx["output_dir"] / "builder" / "smoke_selection.json", {
        "mode": ctx["mode"], "selection": {
            "picked": [list(item) for item in selection["picked"]],
            "empty_frames": selection["empty_frames"],
            "missing_conditions": selection["missing_conditions"]}})
    stage = ctx["stage"]
    stage("admission", "COMPLETED", markers=markers.get("cache_kind"),
          historical_bindings=expanded, v5_gate=v5_report)
    return {"cfg": cfg, "old_config": old_config, "protocol": protocol,
            "authority": authority, "cache_arrays": cache_arrays,
            "manifest": manifest["samples"], "census": census,
            "feature_dir": feature_dir, "selection": selection}


def _native_rebuild_for_instance(ctx: dict, item, v5_row: dict,
                                 v5_path: Path, result, left: np.ndarray,
                                 right: np.ndarray) -> dict:
    """One instance's production-recipe native rebuild (no model calls)."""
    cfg = ctx["cfg"]
    v5_sample = load_mask_sample(v5_path)
    if v5_sample.class_id != item.class_id:
        raise ScaleRefused(
            f"{v5_row['sample_name']}: V5 class != GT class")
    window = (int(v5_row["crop_x1"]), int(v5_row["crop_y1"]),
              int(v5_row["crop_x2"]), int(v5_row["crop_y2"]))
    if window != item.crop_window:
        raise ScaleRefused(
            f"{v5_row['sample_name']}: V5 crop window != GT geometry")
    x1, y1, x2, y2 = window
    if not np.array_equal(v5_sample.gray, left[y1:y2, x1:x2]):
        raise ScaleRefused(
            f"{v5_row['sample_name']}: the V5 gray crop is not the "
            "rectified left crop")
    if not np.array_equal(v5_sample.instance_mask,
                          item.mask[y1:y2, x1:x2]):
        raise ScaleRefused(
            f"{v5_row['sample_name']}: the V5 instance mask is not the "
            "GT raster")
    mask = item.mask
    stats = ctx["matcher"].instance_stats(result, mask, item.object_index)
    if stats.valid:
        polar_result = compute_polar_features(
            left, right, result.disparity, object_mask=mask,
            disparity_valid=result.valid)
        quality, components = quality_vector_from_result(polar_result, mask)
        crop_q = polar_result.signed_q[y1:y2, x1:x2]
        crop_v = polar_result.valid_mask[y1:y2, x1:x2].astype(np.uint8)
        if int(crop_v.sum()) != int(polar_result.valid_mask.sum()):
            raise ScaleRefused(
                f"{v5_row['sample_name']}: the crop window cuts valid "
                "polar pixels")
        warped_right = ctx["warped_right"]
        crop_r = warped_right[y1:y2, x1:x2].astype(np.float32)
        stereo_reason = "ok"
    else:
        components = {key: 0.0 for key in QUALITY_VECTOR_KEYS}
        quality = build_quality_vector(**components)
        crop_q = np.zeros((y2 - y1, x2 - x1), np.float32)
        crop_v = np.zeros((y2 - y1, x2 - x1), np.uint8)
        crop_r = np.zeros((y2 - y1, x2 - x1), np.float32)
        stereo_reason = str(stats.reason)
    crop_r = np.where(crop_v > 0, crop_r, np.float32(0.0)).astype(
        np.float32)
    native = NativeInstanceArrays(
        sample_name=v5_row["sample_name"], frame=v5_row["source_frame"],
        L=v5_sample.gray, R=crop_r, q=crop_q,
        M=v5_sample.instance_mask.astype(np.uint8), V=crop_v,
        region=np.zeros((y2 - y1, x2 - x1), np.uint8),
        quality=np.asarray(quality, dtype=np.float32))
    native.validate()
    from core.polar_region_quality import interior_and_band_masks

    interior, band = interior_and_band_masks(native.M)
    native.region[(interior > 0)] = 1
    native.region[(band > 0)] = 2
    # the frozen C14/S16/region statistics recomputed from the NATIVE
    # pixels: bitwise identity with the frozen cache is the correctness
    # proof of the rebuild
    fix_arrays = MaskFusionSampleArrays(
        gray=native.L, signed_q=native.q, abs_q=np.abs(native.q),
        valid=native.V,
        quality=np.asarray(native.quality, dtype=np.float32),
        class_id=int(item.class_id),
        instance_mask=native.M)
    values, blocks = build_local_polar_stats(fix_arrays)
    c_features = build_local_control_features(fix_arrays)
    region = {**region_records(fix_arrays),
              "blocks": {k: bool(v) for k, v in blocks.items()}}
    return {"native": native, "interior": interior, "band": band,
            "s16": values, "c14": c_features, "region": region,
            "blocks": blocks, "window": list(window),
            "stereo_valid": bool(stats.valid),
            "stereo_reason": stereo_reason,
            "quality_components": {k: float(v) for k, v
                                   in components.items()}}


def _compare_to_frozen(name: str, row: int, rebuilt: dict,
                       cache_arrays: dict, manifest_samples: list) -> list:
    problems = []
    cached_c = np.asarray(cache_arrays["c_features"][row], dtype=np.float32)
    cached_s = np.asarray(cache_arrays["s_features"][row], dtype=np.float32)
    rebuilt_c = np.asarray(rebuilt["c14"], dtype=np.float32)
    rebuilt_s = np.asarray(rebuilt["s16"], dtype=np.float32)
    if not np.array_equal(rebuilt_c.view(np.uint32),
                          cached_c.view(np.uint32)):
        problems.append(f"{name}: C14 bitwise mismatch")
    if (np.isnan(rebuilt_s) != np.isnan(cached_s)).any():
        problems.append(f"{name}: S16 NaN positions differ")
    finite = ~np.isnan(rebuilt_s) & ~np.isnan(cached_s)
    if not np.array_equal(rebuilt_s[finite].view(np.uint32),
                          cached_s[finite].view(np.uint32)):
        problems.append(f"{name}: S16 finite values differ bitwise")
    sample = manifest_samples[row]
    numeric = ("mask_pixels", "interior_pixels", "band_pixels",
               "interior_valid_pixels", "band_valid_pixels",
               "region_pixels")
    for key in numeric:
        if int(sample[key]) != int(rebuilt["region"][key]):
            problems.append(f"{name}: region.{key} differs from the cache")
    rebuilt_valid = not np.isnan(rebuilt_s).all()
    if rebuilt_valid != bool(cache_arrays["s_valid"][row]):
        problems.append(f"{name}: s_valid differs from the cache")
    return problems


def _build_frames(ctx: dict, admission: dict, frames: list,
                  empty_frames: list) -> dict:
    registry = ctx["registry"]
    watch = ctx["watch"]
    log = ctx["log"]
    output_dir = ctx["output_dir"]
    cfg = admission["cfg"]
    matcher_params = matcher_config_of(admission["old_config"])
    # band_margin / full_image_threshold / crop_pad live in fix_inputs
    # but outside matcher_config_of's matcher-key subset
    fix_inputs = admission["old_config"]["fix_inputs"]
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
        raise ScaleRefused("the matcher did not receive the fix version")
    ctx["matcher"] = StereoMatcher(stereo_cfg)
    cache_native = output_dir / "cache" / "native"
    cache_grid = output_dir / "cache" / "grid"
    cache_perm = output_dir / "cache" / "permutation"
    for directory in (cache_native, cache_grid, cache_perm):
        directory.mkdir(parents=True, exist_ok=True)
    cache_arrays = admission["cache_arrays"]
    manifest_samples = admission["manifest"]
    row_of = {sample["sample_name"]: index
              for index, sample in enumerate(manifest_samples)}
    frame_rows = admission["census"]["frame_rows"]
    pairs = admission["census"]["pairs"]
    data_root = _PROJECT_ROOT / cfg["frozen_sources"]["v5_data"]["root"]
    frames_ledger = []
    samples_ledger = []
    inventory = []
    built_instances = 0
    for frame_index, stem in enumerate(frames):
        watch.check(f"frame {stem}")
        rows = frame_rows[stem]
        first = rows[0]
        left_path = _PROJECT_ROOT / first["left_path"]
        right_path = _PROJECT_ROOT / pairs[stem]["right_path"]
        label_path = _PROJECT_ROOT / first["label_path"]
        book: ExpectationBook = ctx["book"]
        for path, role in ((left_path, f"image_left:{stem}"),
                           (right_path, f"image_right:{stem}"),
                           (label_path, f"label:{stem}")):
            register_consumed(registry, path, role, _PROJECT_ROOT,
                              book.binding_for(path))
        left = load_gray(left_path)
        right = load_gray(right_path)
        height, width = left.shape[:2]
        geometry = build_gt_geometry(
            label_path, width, height,
            padding=int(fix_inputs["crop_pad"]))
        if len(geometry) != len(rows):
            raise ScaleRefused(
                f"frame {stem}: {len(geometry)} GT instances != "
                f"{len(rows)} V5 rows")
        bands = build_horizontal_bands(
            [item.band_bbox for item in geometry], image_height=height,
            vertical_margin=int(fix_inputs["band_margin"]))
        result = ctx["matcher"].compute_bands(
            left, right, bands,
            full_image_threshold=float(
                fix_inputs["full_image_threshold"]))
        right_f = to_gray_u8(right).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(height, dtype=np.float32),
                             np.arange(width, dtype=np.float32),
                             indexing="ij")
        ctx["warped_right"] = cv2.remap(
            right_f, xx - result.disparity.astype(np.float32), yy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        by_name = {f"{stem}_obj{item.object_index:03d}": item
                   for item in geometry}
        v5_rows_by_name = {}
        for row in rows:
            v5_rows_by_name[row["sample_name"]] = row
        native_payload = {}
        grid_payload = {}
        perm_records = {}
        for name in sorted(by_name):
            watch.check(f"instance {name}")
            item = by_name[name]
            v5_row = v5_rows_by_name[name]
            v5_path = data_root / v5_row["npz_path"]
            register_consumed(registry, v5_path, f"v5_sample:{name}",
                              _PROJECT_ROOT, book.binding_for(v5_path))
            rebuilt = _native_rebuild_for_instance(
                ctx, item, v5_row, v5_path, result, left, right)
            native = rebuilt["native"]
            row = row_of[name]
            problems = _compare_to_frozen(name, row, rebuilt,
                                          cache_arrays, manifest_samples)
            if problems:
                raise ScaleRefused(
                    "the native rebuild disagrees with the frozen cache: "
                    f"{problems}")
            grid = _build_grid_planes(native.L, native.R, native.q,
                                      native.M, native.V, rebuilt[
                                          "interior"], rebuilt["band"])
            planes = grid["planes"]
            plan = plan_q_permutation(planes, name)
            native_payload[name] = {
                "L": native.L, "R": native.R, "q": native.q,
                "M": native.M, "V": native.V, "region": native.region,
                "quality": native.quality}
            grid_payload[name] = np.stack(
                [planes[key] for key in GRID_PLANES], axis=0)
            perm_records[name] = plan
            samples_ledger.append({
                "sample_name": name, "frame": stem,
                "cache_row": row,
                "class_id": int(v5_row["class_id"]),
                "class_name": manifest_samples[row]["class_name"],
                "group": v5_row["group_name"],
                "split": v5_row["split"],
                "crop_window": rebuilt["window"],
                "stereo_valid": rebuilt["stereo_valid"],
                "stereo_reason": rebuilt["stereo_reason"],
                "has_valid_s": bool(cache_arrays["s_valid"][row]),
                "blocks": rebuilt["blocks"],
                "w2": grid["w2"], "h2": grid["h2"],
                "offset": grid["offset"],
                "fold_roles": dict(
                    admission["authority"][name]["roles"])})
            built_instances += 1
        native_path = cache_native / f"frame_{stem}.npz"
        np.savez_compressed(
            native_path,
            **{f"{name}|{key}": arrays[key]
               for name, arrays in native_payload.items()
               for key in ("L", "R", "q", "M", "V", "region", "quality")})
        grid_path = cache_grid / f"frame_{stem}.npz"
        np.savez_compressed(
            grid_path,
            **{f"{name}|planes": stacked
               for name, stacked in grid_payload.items()})
        perm_path = cache_perm / f"frame_{stem}.json"
        perm_path.write_text(
            json.dumps(perm_records, ensure_ascii=False) + "\n",
            encoding="utf-8")
        for path, kind in ((native_path, "native"), (grid_path, "grid"),
                           (perm_path, "permutation")):
            inventory.append({
                "path": path.relative_to(output_dir).as_posix(),
                "kind": kind, "frame": stem,
                "bytes": path.stat().st_size,
                "sha256": sha256_of_file(path),
                "npz_keys": (npz_key_records(path)
                             if kind in ("native", "grid") else None)})
        frames_ledger.append({
            "frame": stem, "split": first["split"],
            "group": first["group_name"], "instances": len(rows),
            "note": "band-restricted production SGBM pass (fix version)"})
        ctx["warped_right"] = None
        log(f"frames {frame_index + 1}/{len(frames)}: {stem} "
            f"({len(rows)} instances)")
    for stem in empty_frames:
        watch.check(f"empty frame {stem}")
        row = admission["census"]["pairs"][stem]
        left_path = _PROJECT_ROOT / row["left_path"]
        right_path = _PROJECT_ROOT / row["right_path"]
        for path, role in ((left_path, f"image_left:{stem}"),
                           (right_path, f"image_right:{stem}")):
            register_consumed(registry, path, role, _PROJECT_ROOT,
                              book.binding_for(path))
        frames_ledger.append({
            "frame": stem, "split": None, "group": None, "instances": 0,
            "note": "empty frame: frame ledger only, no instance records"})
    old_config = admission["old_config"]
    recipe = {**matcher_config_of(old_config),
              "band_margin": int(old_config["fix_inputs"]["band_margin"]),
              "full_image_threshold": float(
                  old_config["fix_inputs"]["full_image_threshold"]),
              "crop_pad": int(old_config["fix_inputs"]["crop_pad"]),
              "upsample_version": UPSAMPLE_FIX_VERSION}
    definition = {
        "experiment": EXPERIMENT,
        "cache_schema": CACHE_SCHEMA,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "kind": KIND_FULL if ctx["mode"] == "build" else KIND_SMOKE,
        "mode": ctx["mode"],
        "complete": ctx["mode"] == "build",
        "grid_schema": GRID_SCHEMA,
        "grid_size": GRID_SIZE,
        "permutation_schema": PERMUTATION_SCHEMA,
        "upsample_version": UPSAMPLE_FIX_VERSION,
        "content_seed": 2026,
        "planes": list(GRID_PLANES),
        "recipe": {key: recipe[key] for key in sorted(recipe)},
        "recipe_source": cfg["frozen_sources"]["old_experiment_config"][
            "path"],
        "recipe_source_sha256":
            cfg["frozen_sources"]["old_experiment_config"]["sha256"],
        "source_identity": {
            "config_sha256": ctx["config_sha256"],
            "protocol_sha256":
                cfg["frozen_sources"]["protocol"]["file_sha256"],
            "v5_fingerprint":
                cfg["frozen_sources"]["v5_data"]["fingerprint"]},
        "native_arrays": {key: {"L": "uint8", "R": "float32",
                                "q": "float32", "M": "uint8",
                                "V": "uint8", "region": "uint8",
                                "quality": "float32(4)"}[key]
                          for key in NATIVE_ARRAY_KEYS},
        "dev_pool": {key: cfg["dev_pool"][key] for key in
                     ("frames", "nonempty_frames", "empty_frames",
                      "instances", "groups", "s_valid_true",
                      "s_valid_false", "empty_frame_names")},
        "built": {"frames": len(frames), "instances": built_instances,
                  "empty_frames": list(empty_frames)},
        "run_id": output_dir.name,
    }
    _write_json(output_dir / "cache" / "definition.json", definition)
    _write_jsonl(output_dir / "cache" / "samples.jsonl", samples_ledger)
    _write_jsonl(output_dir / "cache" / "frames.jsonl", frames_ledger)
    inventory.append({"path": "cache/definition.json", "kind": "metadata",
                      "bytes": (output_dir / "cache" /
                                "definition.json").stat().st_size,
                      "sha256": sha256_of_file(output_dir / "cache" /
                                               "definition.json"),
                      "npz_keys": None})
    inventory.append({"path": "cache/samples.jsonl", "kind": "metadata",
                      "bytes": (output_dir / "cache" /
                                "samples.jsonl").stat().st_size,
                      "sha256": sha256_of_file(output_dir / "cache" /
                                               "samples.jsonl"),
                      "npz_keys": None})
    inventory.append({"path": "cache/frames.jsonl", "kind": "metadata",
                      "bytes": (output_dir / "cache" /
                                "frames.jsonl").stat().st_size,
                      "sha256": sha256_of_file(output_dir / "cache" /
                                               "frames.jsonl"),
                      "npz_keys": None})
    return {"inventory": inventory, "definition": definition,
            "built_frames": len(frames), "built_instances":
            built_instances}


def _write_binding(output_dir: Path, inventory: list) -> dict:
    binding = build_cache_binding(output_dir, inventory)
    _write_json(output_dir / "cache" / "binding.json", binding)
    return binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "smoke", "build"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
              f"{message}", file=sys.stderr, flush=True)

    registry = SourceRegistry({})
    try:
        if args.mode != "smoke" and args.max_frames is not None:
            raise ScaleRefused("--max-frames is only valid for smoke")
        if args.max_frames is not None and (
                args.max_frames <= 0 or args.max_frames > 12):
            raise ScaleRefused("--max-frames must be within 1..12")
        config_path = Path(args.config)
        if config_path.resolve() != \
                (_PROJECT_ROOT / PINNED_CONFIG_REL).resolve():
            raise ScaleRefused(
                f"{args.config} is not the pinned batch config "
                f"{_PROJECT_ROOT / PINNED_CONFIG_REL}")
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = _PROJECT_ROOT / output_dir
        _validate_output_dir(output_dir)
        # the config registers (hash-verified against the pinned bytes)
        # BEFORE it is parsed: a drifted config never reaches the parser
        config_record = registry.register(
            config_path, "experiment_config",
            binding={"source": "pinned_config",
                     "expected_sha256": PINNED_CONFIG_SHA256},
            kind="experiment_config")
        cfg = yaml.safe_load(
            config_path.read_text(encoding="utf-8"))
        if cfg.get("experiment") != EXPERIMENT:
            raise ScaleRefused("the config experiment does not match")
        if args.mode == "smoke" and args.max_frames is not None and \
                args.max_frames < len(cfg["dev_pool"]["empty_frame_names"]):
            raise ScaleRefused(
                "--max-frames cannot be smaller than the required empty frames")
    except (ScaleRefused, IdentityError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    watch = DeadlineWatch(deadline_utc=args.deadline_utc)
    stages: dict = {}

    def stage(name: str, status: str, **details) -> None:
        stages[name] = {"status": status, **details}
        log(f"stage {name}: {status}")

    config_sha = config_record["sha256"]
    ctx = {"registry": registry, "watch": watch, "stage": stage,
           "stages": stages,
           "log": log, "output_dir": output_dir, "mode": args.mode,
           "max_frames": args.max_frames, "config_rel": PINNED_CONFIG_REL,
           "config_sha256": config_sha,
           "book": ExpectationBook(_PROJECT_ROOT),
           "cfg": cfg, "matcher": None, "warped_right": None,
           "protocol": None}
    failure = None
    interrupted = False
    unfinished: list[str] = []
    failure_traceback = None
    output_dir.mkdir(parents=True)
    (output_dir / "builder").mkdir(parents=True, exist_ok=True)
    try:
        try:
            watch.check("before admission")
            admission = run_admission(ctx, args)
            stage("admission_done", "COMPLETED")
            if args.mode == "preflight":
                _write_json(output_dir / "builder" / "preflight_plan.json", {
                    "run_mode": "preflight",
                    "note": ("preflight ran no SGBM, warp or pixel "
                             "rebuild"),
                    "smoke_selection": admission["selection"],
                    "v5_gate": ctx["stages"].get("admission", {}).get(
                        "v5_gate"),
                })
                stage("preflight", "COMPLETED")
                ctx["preflight_status"] = "COMPLETED"
                return _closeout(ctx, started, args)
        except DeadlineExceeded as error:
            ctx["interrupted"] = True
            ctx["unfinished"] = unfinished
            ctx["unfinished"].append(str(error))
            log(f"deadline exceeded: {error}")
            return _closeout(ctx, started, args)
        except (ScaleRefused, IdentityError, SpatialRefused,
                FileNotFoundError, ValueError, OSError, KeyError) as exc:
            ctx["failure"] = f"{type(exc).__name__}: {exc}"
            log(f"failure: {ctx['failure']}")
            return _closeout(ctx, started, args)
        try:
            watch.check("before the native/grid build")
            if args.mode == "smoke":
                picked = admission["selection"]["picked"]
                frames = sorted({frame for _reason, _name, frame
                                 in picked})
                # the empty frames only contribute frame-ledger rows
                empty_frames = sorted(
                    admission["selection"]["empty_frames"])
            else:
                frames = admission["census"]["frames"]
                empty_frames = admission["census"]["empty_frames"]
                _check_build_disk_space(ctx)
            build = _build_frames(ctx, admission, frames, empty_frames)
            binding = _write_binding(output_dir, build["inventory"])
            stage("build", "COMPLETED",
                  frames=build["built_frames"],
                  instances=build["built_instances"])
            ctx["build_outcome"] = {
                "built_frames": build["built_frames"],
                "built_instances": build["built_instances"],
                "artifacts": binding["artifact_count"]}
            ctx["empty_frames"] = sorted(empty_frames)
            return _closeout(ctx, started, args)
        except DeadlineExceeded as error:
            ctx["interrupted"] = True
            ctx["unfinished"] = unfinished
            ctx["unfinished"].append(str(error))
            log(f"deadline exceeded: {error}")
            return _closeout(ctx, started, args)
        except (ScaleRefused, IdentityError, SpatialRefused,
                FileNotFoundError, ValueError, OSError, KeyError) as exc:
            ctx["failure"] = f"{type(exc).__name__}: {exc}"
            log(f"failure: {ctx['failure']}")
            return _closeout(ctx, started, args)
        except Exception as exc:
            import traceback

            ctx["failure"] = f"UNEXPECTED {type(exc).__name__}: {exc}"
            ctx["failure_traceback"] = traceback.format_exc()
            log(f"unexpected failure: {ctx['failure']}")
            return _closeout(ctx, started, args)
    except Exception as exc:  # absolute safety net
        import traceback

        ctx["failure"] = f"UNEXPECTED {type(exc).__name__}: {exc}"
        ctx["failure_traceback"] = traceback.format_exc()
        log(f"unexpected failure: {ctx['failure']}")
        return _closeout(ctx, started, args)


def _closeout(ctx: dict, started: float, args) -> int:
    registry = ctx["registry"]
    watch = ctx["watch"]
    log = ctx["log"]
    output_dir = ctx["output_dir"]
    builder_dir = output_dir / "builder"
    failure = ctx.get("failure")
    interrupted = ctx.get("interrupted", False)
    unfinished = ctx.get("unfinished", [])
    try:
        protection = registry.compare()
        registry.write_manifest(builder_dir / "source_manifest.json",
                                protection)
        ctx["stage"]("protection",
                     "COMPLETED" if protection["identical"] else "FAILED",
                     registered=protection["registered_sources"],
                     problems=protection["problems"])
        ctx["protection"] = protection
        if not protection["identical"] and failure is None:
            failure = "protection check found changed inputs"
    except Exception as exc:
        log(f"closeout failed: {exc}")
        if failure is None:
            failure = f"CLOSEOUT {type(exc).__name__}: {exc}"
    if not interrupted and ctx["watch"] is not None:
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
    build_outcome = ctx.get("build_outcome") or {}
    overall = status
    if status == "COMPLETED" and "build_outcome" in ctx:
        overall = "BUILT_UNVERIFIED"   # the wrapper publishes COMPLETED
    binding_path = output_dir / "cache" / "binding.json"
    _write_json(builder_dir / "run_status.json", {
        "run_mode": ctx["mode"], "experiment": EXPERIMENT,
        "run_id": output_dir.name,
        "config_sha256": ctx.get("config_sha256"),
        "deadline_utc": args.deadline_utc,
        "cache_schema": CACHE_SCHEMA,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "cache_kind": KIND_FULL if ctx["mode"] == "build" else (
            KIND_SMOKE if ctx["mode"] == "smoke" else None),
        "cache_complete": ctx["mode"] == "build",
        "cache_binding_sha256": (sha256_of_file(binding_path)
                                 if binding_path.is_file() else None),
        "cache_binding_artifacts": (build_outcome.get("artifacts")
                                    if build_outcome else None),
        "stages": ctx["stages"], "interrupted_by_deadline": interrupted,
        "unfinished": unfinished, "failure": failure,
        "failure_traceback": ctx.get("failure_traceback"),
        "overall_status": overall, "exit_code": code,
        "input_admitted": False,
        "zero_training_calls": {"model_fits": 0, "scaler_fits": 0,
                                "classifier_forwards": 0,
                                "encoder_forwards": 0,
                                "optimizer_steps": 0},
        "built_frames": build_outcome.get("built_frames"),
        "built_instances": build_outcome.get("built_instances"),
        "empty_frames": ctx.get("empty_frames"),
        "elapsed_s": time.time() - started,
        "protection": ctx.get("protection"),
        "source_manifest": "builder/source_manifest.json",
        "note": ("builder completion is BUILT_UNVERIFIED; only the "
                 "wrapper publishes COMPLETED/input_admitted=true after "
                 "the independent verifier, source protection and "
                 "binding all pass within the shared deadline"),
    })
    log(f"overall: {overall} (exit {code})")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
