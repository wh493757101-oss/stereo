"""material_spatial_input_v1 — the independent spatial-input verifier.

Re-reads the registered sources, rebuilds every frame with the
production recipe through its OWN code path, re-derives the grid with an
independent exact-integer weight-matrix implementation (its own
letterbox, its own overlap matrices — it never imports the builder's or
core's grid/area functions) and re-derives the deterministic
q-permutation plans from the saved float32 coverage bit patterns.

Contract (review_fix_v1): the verifier consumes the run root's cache
binding only after hash-verifying it against --expected-binding-sha256,
registers every binding artifact before loading it, derives the
authoritative instance sets from the frozen manifest/V5/protocol (never
from the saved shard keys) and requires the shard key sets to equal
those authoritative sets exactly.

Grid tolerance: atol=1e-6, rtol=0 (never relaxed). Native arrays are
compared bitwise. Exit codes: 0 verified, 1 refused/failed, 2 partial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    GRID_PLANES,
    GRID_SIZE,
    PERMUTATION_SCHEMA,
    SpatialRefused,
    apply_q_permutation,
    validate_grid_planes,
)
from core.material_spatial_input_admission import (  # noqa: E402
    CACHE_SCHEMA,
    IMPLEMENTATION_REVISION,
    META_RELS,
    AdmissionRefused,
    ExpectationBook,
    authoritative_frame_instances,
    check_definition,
    check_frames_ledger,
    check_native_arrays,
    check_permutation_plan,
    check_samples_ledger,
    dev_census,
    expected_artifact_paths,
    expected_shard_keys,
    register_consumed,
    run_v5_admission,
    verify_inventory,
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
    to_gray_u8,
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
    is_link_or_reparse,
    sha256_of_file,
    verify_anchor,
)
from scripts.prepare_cls_fusion_dataset import load_gray  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_PARTIAL = 2

GRID_ATOL = 1e-6
GRID_RTOL = 0.0
PINNED_CONFIG_REL = "configs/experiments/material_spatial_input_v1.yaml"
# the batch config's frozen bytes; verified before the config is parsed
PINNED_CONFIG_SHA256 = "0761eb9f11b484f68d85df3c444a4c203f8de0f877dc72d5a57473e727b6314e"
OUTPUT_ROOT = "analysis/runs/material_spatial_input_v1"


class _VerifierRefused(ScaleRefused):
    pass


# ---------------------------------------------------------------------------
# independent geometry: the verifier's OWN letterbox and exact integer
# overlap weight matrices — deliberately not imported from
# core.material_spatial_input and never delegated to the builder


def _independent_letterbox(w: int, h: int,
                           grid: int = GRID_SIZE) -> tuple[int, int,
                                                           tuple[int, int]]:
    s = 1.0 if max(w, h) <= grid else grid / max(w, h)
    w2 = max(1, int(np.floor(w * s + 0.5)))
    h2 = max(1, int(np.floor(h * s + 0.5)))
    x0 = (grid - w2) // 2
    y0 = (grid - h2) // 2
    return w2, h2, (x0, y0)


def _independent_overlap(mask: np.ndarray, w: int, h: int, w2: int,
                         h2: int) -> np.ndarray:
    """T[i, j] = sum(mask[y, x] * lx[j, x] * ly[i, y]): the exact integer
    overlap of the mask with each cell, built from explicit per-axis
    weight matrices (all int64, so T == 0 and T > 0 are exact)."""
    m = (np.asarray(mask) != 0).astype(np.int64)
    if m.shape != (h, w):
        raise _VerifierRefused("mask shape mismatch")
    x = np.arange(w, dtype=np.int64)
    j = np.arange(w2, dtype=np.int64)
    u0 = (j * w)[:, None]
    u1 = u0 + w
    lx = np.minimum(u1, (x + 1)[None, :] * w2) - \
        np.maximum(u0, (x * w2)[None, :])
    lx = np.maximum(lx, 0)
    y = np.arange(h, dtype=np.int64)
    i = np.arange(h2, dtype=np.int64)
    v0 = (i * h)[:, None]
    v1 = v0 + h
    ly = np.minimum(v1, (y + 1)[None, :] * h2) - \
        np.maximum(v0, (y * h2)[None, :])
    ly = np.maximum(ly, 0)
    return ly @ m @ lx.T


def _build_grid_planes(L: np.ndarray, R: np.ndarray, q: np.ndarray,
                       M: np.ndarray, V: np.ndarray, interior: np.ndarray,
                       band: np.ndarray) -> dict:
    """The verifier-side grid through the independent weight-matrix
    reference. R is normalized to the gray scale /255 like L."""
    valid = V > 0
    if not np.all(valid <= (M > 0)):
        raise _VerifierRefused("V is not a subset of M")
    w, h = L.shape[1], L.shape[0]
    w2, h2, (x0, y0) = _independent_letterbox(w, h)
    a_interior = (interior > 0) & valid
    a_band = (band > 0) & valid
    sources = {"L": L.astype(np.float32) / np.float32(255.0),
               "R": R.astype(np.float32) / np.float32(255.0),
               "q": np.asarray(q, dtype=np.float32)}
    planes: dict[str, np.ndarray] = {}
    supports: dict[str, np.ndarray] = {}
    for region_name, a_mask in (("interior", a_interior),
                                ("band", a_band)):
        overlap = _independent_overlap(a_mask.astype(np.uint8), w, h,
                                       w2, h2)
        supports[region_name] = overlap
        coverage = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        coverage[y0:y0 + h2, x0:x0 + w2] = (
            overlap.astype(np.float64) / float(w * h)).astype(np.float32)
        planes[f"coverage_{region_name}"] = coverage
        supported = overlap > 0
        for source_name, src in sources.items():
            weighted = _weighted_source_sum(a_mask, src, w, h, w2, h2)
            response = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            window = np.zeros((h2, w2), dtype=np.float64)
            np.divide(weighted, overlap.astype(np.float64), out=window,
                      where=supported)
            response[y0:y0 + h2, x0:x0 + w2] = window.astype(np.float32)
            planes[f"{source_name}_{region_name}"] = response
    overlap_object = _independent_overlap((M > 0).astype(np.uint8), w, h,
                                          w2, h2)
    supports["object"] = overlap_object
    coverage_object = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    coverage_object[y0:y0 + h2, x0:x0 + w2] = (
        overlap_object.astype(np.float64) / float(w * h)).astype(
        np.float32)
    planes["coverage_object"] = coverage_object
    validate_grid_planes(planes, w2, h2, (x0, y0))
    return {"planes": planes, "offset": [x0, y0], "w2": w2, "h2": h2,
            "supports": supports}


def _weighted_source_sum(a_mask: np.ndarray, src: np.ndarray, w: int,
                         h: int, w2: int, h2: int) -> np.ndarray:
    """T_X[i, j] = sum(A[y, x] * X[y, x] * lx[j, x] * ly[i, y]) — the
    coverage-weighted source sum over each cell, in the same exact
    integer weight-matrix form (float64 only because X is float)."""
    m = (np.asarray(a_mask) != 0).astype(np.float64)
    x = np.arange(w, dtype=np.int64)
    j = np.arange(w2, dtype=np.int64)
    u0 = (j * w)[:, None]
    u1 = u0 + w
    lx = np.minimum(u1, (x + 1)[None, :] * w2) - \
        np.maximum(u0, (x * w2)[None, :])
    lx = np.maximum(lx, 0).astype(np.float64)
    y = np.arange(h, dtype=np.int64)
    i = np.arange(h2, dtype=np.int64)
    v0 = (i * h)[:, None]
    v1 = v0 + h
    ly = np.minimum(v1, (y + 1)[None, :] * h2) - \
        np.maximum(v0, (y * h2)[None, :])
    ly = np.maximum(ly, 0).astype(np.float64)
    values = m * np.asarray(src, dtype=np.float64)
    return ly @ values @ lx.T


def _check_saved_planes_against_independent(
        saved_planes: dict, supports: dict, w2: int, h2: int,
        offset: tuple[int, int]) -> None:
    """The saved planes are checked against the verifier's OWN exact
    integer support sets — never against the saved coverage's own claim:
      - independent T == 0  ->  the saved coverage and all three
        responses must be positive-zero bit patterns;
      - independent T > 0   ->  the saved coverage strictly positive and
        finite;
      - the saved positive-support set equals the independent T > 0 set
        exactly (per region and for the object coverage);
      - padding: all nine planes positive zero.
    Other numeric properties (ranges, finiteness) are checked by
    validate_grid_planes on top."""
    x0, y0 = offset
    grid = GRID_SIZE
    window = np.zeros((grid, grid), dtype=bool)
    window[y0:y0 + h2, x0:x0 + w2] = True
    padding = ~window
    for plane_name in GRID_PLANES:
        plane = saved_planes[plane_name]
        bits = plane.view(np.uint32)
        if np.any(bits[padding] != 0):
            raise ScaleRefused(
                f"{plane_name}: padding is not positive zero")

    def _placed(overlap: np.ndarray) -> np.ndarray:
        placed = np.zeros((grid, grid), dtype=np.int64)
        placed[y0:y0 + h2, x0:x0 + w2] = overlap
        return placed

    independent_support = {
        "interior": _placed(supports["interior"]) > 0,
        "band": _placed(supports["band"]) > 0,
    }
    independent_object = _placed(supports["object"]) > 0
    for region in ("interior", "band"):
        independent = independent_support[region]
        coverage = saved_planes[f"coverage_{region}"]
        coverage_bits = coverage.view(np.uint32)
        # saved positive-support set == the independent T > 0 set
        saved_support = coverage_bits != 0
        if not np.array_equal(saved_support, independent):
            extra = int(np.count_nonzero(saved_support & ~independent))
            missing = int(np.count_nonzero(independent & ~saved_support))
            raise ScaleRefused(
                f"coverage_{region}: the saved support set differs from "
                f"the independent exact support (extra {extra}, missing "
                f"{missing})")
        # independent T == 0 -> coverage and responses positive zero
        if np.any(coverage_bits[~independent] != 0):
            raise ScaleRefused(
                f"coverage_{region}: non-zero coverage where the "
                "independent overlap is exactly zero")
        for source in ("L", "R", "q"):
            bits = saved_planes[f"{source}_{region}"].view(np.uint32)
            if np.any(bits[~independent] != 0):
                raise ScaleRefused(
                    f"{source}_{region}: non-zero response where the "
                    "independent overlap is exactly zero")
        # independent T > 0 -> coverage strictly positive and finite
        if independent.any():
            values = coverage[independent]
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ScaleRefused(
                    f"coverage_{region}: the saved coverage is not "
                    "strictly positive where the independent overlap "
                    "is positive")
    object_bits = saved_planes["coverage_object"].view(np.uint32)
    if not np.array_equal(object_bits != 0, independent_object):
        raise ScaleRefused(
            "coverage_object: the saved support set differs from the "
            "independent exact M overlap")
    if np.any(object_bits[~independent_object] != 0):
        raise ScaleRefused(
            "coverage_object: non-zero coverage where the independent "
            "M overlap is exactly zero")


def _plan_q_permutation_independent(planes: dict, sample_name: str,
                                    content_seed: int = 2026) -> dict:
    """The verifier's independent re-derivation of
    coverage_matched_parts_pcg64_v1 from the saved float32 coverage bit
    patterns (never from the verifier's own rebuild, whose float values
    may differ by ~1 ULP)."""
    plan = {"sample_name": sample_name, "schema": PERMUTATION_SCHEMA,
            "content_seed": int(content_seed), "regions": {}}
    for region in ("interior", "band"):
        response = np.asarray(planes[f"q_{region}"], dtype=np.float32)
        coverage = np.asarray(planes[f"coverage_{region}"],
                              dtype=np.float32)
        flat_r = response.ravel()
        flat_c = coverage.ravel()
        movable = [index for index in range(flat_c.size)
                   if flat_c[index].view(np.uint32) != 0]
        groups: dict[str, dict] = {}
        by_bits: dict[str, list[int]] = {}
        for index in movable:
            bits = f"{int(flat_c[index].view(np.uint32)):08x}"
            by_bits.setdefault(bits, []).append(index)
        for bits, cells in sorted(by_bits.items()):
            domain = (f"material_spatial_input_v1|{sample_name}|{region}|"
                      f"{bits}|{content_seed}")
            seed64 = int.from_bytes(
                hashlib.sha256(domain.encode("utf-8")).digest()[:8],
                "little")
            record = {"coverage_bits_hex8": bits, "cells": len(cells)}
            if len(cells) > 1:
                perm = list(np.random.Generator(
                    np.random.PCG64(seed64)).permutation(len(cells)))
                old = [flat_r[index].view(np.uint32) for index in cells]
                new = [old[position] for position in perm]
                moved = sum(1 for a, b in zip(perm, range(len(perm)))
                            if a != b)
                changed = sum(1 for a, b in zip(old, new) if a != b)
                record.update({"seed64": seed64, "permuted": True,
                               "moved_indices": moved,
                               "value_changed_cells": changed,
                               "cell_indices": [int(v) for v in cells],
                               "permutation": [int(v) for v in perm]})
            else:
                record.update({"permuted": False, "moved_indices": 0,
                               "value_changed_cells": 0})
            groups[bits] = record
        plan["regions"][region] = {
            "movable_cells": len(movable), "groups": groups,
            "moved_indices": sum(g.get("moved_indices", 0)
                                 for g in groups.values()),
            "value_changed_cells": sum(
                g.get("value_changed_cells", 0)
                for g in groups.values())}
    return plan


def _registered_right_crop(right_u8: np.ndarray, disparity: np.ndarray,
                           window: tuple[int, int, int, int]) -> np.ndarray:
    """The verifier's own mirror of compute_polar_features' sampling
    (float32 right, map_x = x - d without clipping, INTER_LINEAR,
    BORDER_CONSTANT)."""
    right_f = to_gray_u8(right_u8).astype(np.float32)
    h, w = right_f.shape
    disp = np.asarray(disparity, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    warped = cv2.remap(right_f, xx - disp, yy,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    x1, y1, x2, y2 = window
    return warped[y1:y2, x1:x2]


def _admission(ctx: dict, cfg: dict) -> dict:
    """The same admission rules as the builder: pinned anchors first,
    historical expansion before any consumer read, register-before-read
    for the old config/protocol/fix cache, the full V5 gate and the
    dev-pool census."""
    registry = ctx["registry"]
    book: ExpectationBook = ctx["book"]
    ctx["stage"]("admission", "STARTED")
    pinned = {}
    for name, record in cfg["frozen_sources"]["fix_cache_files"].items():
        rel = f"{cfg['frozen_sources']['fix_cache_dir']}/{name}"
        pinned[rel] = record["sha256"]
    pinned[cfg["frozen_sources"]["old_experiment_config"]["path"]] = \
        cfg["frozen_sources"]["old_experiment_config"]["sha256"]
    pinned[cfg["frozen_sources"]["build_source_manifest"]["path"]] = \
        cfg["frozen_sources"]["build_source_manifest"]["sha256"]
    pinned[cfg["frozen_sources"]["protocol"]["path"]] = \
        cfg["frozen_sources"]["protocol"]["file_sha256"]
    for rel, expected in pinned.items():
        path = _PROJECT_ROOT / rel
        if not path.is_file():
            raise ScaleRefused(f"pinned input missing: {path}")
        actual = sha256_of_file(path)
        if actual != expected:
            raise ScaleRefused(
                f"{rel}: sha256 {actual} != the pinned {expected}")
        book.add(path, "config.pin", expected)
        verify_anchor(path, expected, registry, "anchor:spatial_verify")
    manifest_path = _PROJECT_ROOT / cfg["frozen_sources"][
        "build_source_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("items", []):
        if item.get("status") != "ok" or item.get("unchanged") is not True:
            continue
        digest = item.get("sha256_at_registration")
        end_digest = item.get("sha256_at_end")
        rel = item.get("path")
        if not digest or not rel or digest != end_digest:
            continue
        book.add(_PROJECT_ROOT / rel, "build_source_manifest", digest)
    feature_dir = _PROJECT_ROOT / cfg["frozen_sources"]["fix_cache_dir"]
    binding = json.loads((feature_dir / "cache_binding.json").read_text(
        encoding="utf-8"))
    for name, record in (binding.get("files") or {}).items():
        book.add(feature_dir / name, "fix_cache_binding", record["sha256"])
    for rel, expected in book.expected.items():
        raw_rel = expected.get("rel_raw", rel)
        existing = registry.bindings.get(raw_rel)
        if existing is not None and \
                existing.get("expected_sha256") != \
                expected["expected_sha256"]:
            raise ScaleRefused(f"{raw_rel}: conflicting historical "
                               "expectations")
        if existing is None:
            registry.bindings[raw_rel] = dict(expected)
    old_config_path = _PROJECT_ROOT / \
        cfg["frozen_sources"]["old_experiment_config"]["path"]
    register_consumed(registry, old_config_path, "old_experiment_config",
                      _PROJECT_ROOT, book.binding_for(old_config_path))
    old_config = yaml.safe_load(old_config_path.read_text(
        encoding="utf-8"))
    for key in ("matcher", "mode", "max_disp", "scale", "block_size",
                "uniqueness_ratio", "speckle_window", "speckle_range",
                "texture_threshold", "lr_check_threshold_px",
                "min_valid_ratio", "band_margin", "full_image_threshold",
                "crop_pad"):
        if key not in old_config["fix_inputs"]:
            raise ScaleRefused(
                f"the frozen old config is missing fix_inputs.{key}")
    protocol_path = _PROJECT_ROOT / cfg["frozen_sources"]["protocol"][
        "path"]
    register_consumed(registry, protocol_path, "frozen_protocol",
                      _PROJECT_ROOT, book.binding_for(protocol_path))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ctx["protocol"] = protocol
    if protocol.get("protocol_sha256") != \
            cfg["frozen_sources"]["protocol"]["internal_digest"]:
        raise ScaleRefused("the protocol internal digest does not match")
    authority = authoritative_sample_index(protocol)
    for name in ("feature_definition.json", "feature_cache.npz",
                 "feature_manifest.json"):
        path = feature_dir / name
        register_consumed(registry, path, f"fix_cache:{name}",
                          _PROJECT_ROOT, book.binding_for(path))
    markers = verify_scale_cache_markers(feature_dir, old_config)
    cache_definition = json.loads(
        (feature_dir / "feature_definition.json").read_text(
            encoding="utf-8"))
    if cache_definition.get("feature_definition_version") != \
            old_config["local_features"]["feature_definition_version"]:
        raise ScaleRefused("the fix cache local feature version does not "
                           "match the frozen config")
    cache_arrays = dict(np.load(feature_dir / "feature_cache.npz"))
    frozen_manifest = json.loads(
        (feature_dir / "feature_manifest.json").read_text(
            encoding="utf-8"))
    names = [sample["sample_name"] for sample in frozen_manifest["samples"]]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ScaleRefused("the fix cache manifest is not uniquely sorted")
    v5_report = run_v5_admission(registry, book, protocol, cfg,
                                 _PROJECT_ROOT, log=ctx["log"])
    census = dev_census(cfg, registry, book, _PROJECT_ROOT)
    census_names = sorted(row["sample_name"] for row in census["dev_rows"])
    if census_names != names:
        raise ScaleRefused("the dev pool differs from the frozen cache "
                           "manifest sample set")
    ctx["stage"]("admission", "COMPLETED", markers=markers.get(
        "cache_kind"), v5_gate=v5_report)
    return {"cfg": cfg, "old_config": old_config, "protocol": protocol,
            "authority": authority, "cache_arrays": cache_arrays,
            "manifest": frozen_manifest["samples"],
            "feature_dir": feature_dir, "census": census,
            "v5_report": v5_report}


def _load_jsonl(path: Path) -> list:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_cache_identity(ctx: dict, admission: dict, cache_dir: Path,
                           binding: dict) -> tuple[dict, dict, dict]:
    """Definition contract, ledger identity and the authoritative shard
    key sets. Returns (definition, samples_ledger, frames_ledger)."""
    cfg = admission["cfg"]
    mode = ctx["mode"]
    run_root = cache_dir
    # the builder's own evidence binds this run (no cross-run stitching;
    # the status file was registered before the binding was parsed)
    builder_status_path = run_root / "builder" / "run_status.json"
    builder_status = json.loads(builder_status_path.read_text(
        encoding="utf-8"))
    for field in ("run_id", "config_sha256", "deadline_utc"):
        if str(builder_status.get(field)) != str(ctx[field]):
            raise ScaleRefused(
                f"the builder evidence {field} "
                f"{builder_status.get(field)!r} does not match this "
                f"run ({ctx[field]!r}); refusing cross-run stitching")
    if builder_status.get("cache_binding_sha256") != \
            ctx["expected_binding_sha256"]:
        raise ScaleRefused("the builder status does not bind the "
                           "expected cache binding digest")
    # the definition contract
    definition = json.loads((run_root / "cache" / "definition.json")
                            .read_text(encoding="utf-8"))
    old_config = admission["old_config"]
    matcher_params = matcher_config_of(old_config)
    recipe = {**matcher_params,
              "band_margin": int(old_config["fix_inputs"]["band_margin"]),
              "full_image_threshold": float(
                  old_config["fix_inputs"]["full_image_threshold"]),
              "crop_pad": int(old_config["fix_inputs"]["crop_pad"]),
              "upsample_version": UPSAMPLE_FIX_VERSION}
    expected_frames = int(cfg["dev_pool"]["nonempty_frames"])
    expected_instances = int(cfg["dev_pool"]["instances"])
    expected_empty = sorted(cfg["dev_pool"]["empty_frame_names"])
    check_definition(definition,
                     mode=mode,
                     config=cfg, recipe=recipe,
                     recipe_source_sha=cfg["frozen_sources"][
                         "old_experiment_config"]["sha256"],
                     planes=GRID_PLANES,
                     built_frames=expected_frames,
                     built_instances=expected_instances,
                     empty_frames=expected_empty)
    if definition.get("source_identity", {}).get("config_sha256") != \
            ctx["config_sha256"]:
        raise ScaleRefused("definition.config_sha256 does not match this "
                           "run's config")
    if definition.get("run_id") != ctx["run_id"]:
        raise ScaleRefused("definition.run_id does not match this run")
    # the ledgers
    samples_ledger = _load_jsonl(run_root / "cache" / "samples.jsonl")
    frames_ledger = _load_jsonl(run_root / "cache" / "frames.jsonl")
    check_samples_ledger(samples_ledger, admission["manifest"],
                         admission["authority"], "samples.jsonl",
                         require_complete=(mode == "full"))
    if mode == "full":
        if sorted(row["sample_name"] for row in samples_ledger) != \
                sorted(sample["sample_name"]
                       for sample in admission["manifest"]):
            raise ScaleRefused("the sample ledger differs from the "
                               "frozen manifest instance set")
    authoritative = authoritative_frame_instances(admission["manifest"])
    per_frame_ledger: dict[str, set[str]] = {}
    for row in samples_ledger:
        per_frame_ledger.setdefault(row["frame"], set()).add(
            row["sample_name"])
    artifact_map = {record["path"]: record for record in
                    binding.get("artifacts", [])}
    for stem, ledger_names in sorted(per_frame_ledger.items()):
        expected = expected_shard_keys(
            authoritative[stem] if mode == "full" else ledger_names)
        native_record = artifact_map.get(f"cache/native/frame_{stem}.npz")
        grid_record = artifact_map.get(f"cache/grid/frame_{stem}.npz")
        if native_record is None or grid_record is None:
            raise ScaleRefused(f"frame {stem}: shards missing from the "
                               "binding")
        native_keys = {entry["key"]
                       for entry in native_record["npz_keys"]}
        grid_keys = {entry["key"] for entry in grid_record["npz_keys"]}
        if native_keys != expected["native"]:
            raise ScaleRefused(
                f"frame {stem}: native shard keys differ from the "
                "authoritative instance set")
        if grid_keys != expected["grid"]:
            raise ScaleRefused(
                f"frame {stem}: grid shard keys differ from the "
                "authoritative instance set")
    if mode == "full":
        expected_frame_map = {
            stem: {"count": len(rows), "split": rows[0]["split"],
                   "group": rows[0]["group_name"]}
            for stem, rows in admission["census"]["frame_rows"].items()}
        for stem in admission["census"]["empty_frames"]:
            expected_frame_map[stem] = {"count": 0, "split": None,
                                        "group": None}
        check_frames_ledger(frames_ledger, expected_frame_map,
                            admission["census"]["empty_frames"])
    else:
        # Selection comes from authority-checked samples. Frame metadata
        # and complete per-frame instance counts come from the census,
        # never from the frame ledger that is being checked.
        empty_names = admission["census"]["empty_frames"]
        expected = {}
        for stem in per_frame_ledger:
            source_rows = admission["census"]["frame_rows"][stem]
            expected[stem] = {
                "count": len(source_rows), "split": source_rows[0]["split"],
                "group": source_rows[0]["group_name"]}
        for stem in empty_names:
            expected[stem] = {"count": 0, "split": None, "group": None}
        check_frames_ledger(frames_ledger, expected, empty_names)
    ctx["stage"]("cache_identity", "COMPLETED",
                 sample_rows=len(samples_ledger),
                 frames=len(frames_ledger))
    return definition, samples_ledger, frames_ledger


def _verify_native_rebuild(ctx: dict, admission: dict, cache_dir: Path,
                           frames: list, samples_ledger: list) -> dict:
    """The independent per-frame rebuild: native bitwise, C14/S16, the
    saved-plane strict contract, the independent grid reference and the
    permutation re-derivation with conservation."""
    cfg = admission["cfg"]
    registry = ctx["registry"]
    watch = ctx["watch"]
    log = ctx["log"]
    matcher_params = matcher_config_of(admission["old_config"])
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
    matcher = StereoMatcher(stereo_cfg)
    data_root = _PROJECT_ROOT / cfg["frozen_sources"]["v5_data"]["root"]
    cache_arrays = admission["cache_arrays"]
    manifest_samples = admission["manifest"]
    row_of = {sample["sample_name"]: index
              for index, sample in enumerate(manifest_samples)}
    book: ExpectationBook = ctx["book"]
    fix_inputs = admission["old_config"]["fix_inputs"]
    grid_atol = float(cfg["verify"]["grid_atol"])
    problems = []
    rebuilt_instances = 0
    perm_stats = {"plans": 0, "support_cells": 0, "permutable_cells": 0,
                  "singleton_cells": 0, "moved_cells": 0,
                  "changed_cells": 0}
    grid_stats = {"total_elements": 0, "nonzero_elements": 0,
                  "mismatches": []}
    native_compare = {"instances": 0, "mismatches": []}
    ledger_by_name = {row["sample_name"]: row for row in samples_ledger}
    frames_ledger_by_frame = {row["frame"]: row
                              for row in ctx["frames_ledger"]}
    authority = admission["authority"]
    for stem in frames:
        watch.check(f"verify frame {stem}")
        frame_rows = ctx["frame_rows"][stem]
        first = frame_rows[0]
        ledger_frame = frames_ledger_by_frame.get(stem)
        if ledger_frame is None:
            problems.append(f"{stem}: missing from the frames ledger")
            continue
        if ledger_frame["instances"] != len(frame_rows):
            problems.append(f"{stem}: frames-ledger instance count "
                            f"{ledger_frame['instances']} != census "
                            f"{len(frame_rows)}")
        if ledger_frame["group"] != first["group_name"] or \
                ledger_frame["split"] != first["split"]:
            problems.append(f"{stem}: frames-ledger group/split differ "
                            "from the census")
        left_path = _PROJECT_ROOT / first["left_path"]
        right_path = _PROJECT_ROOT / ctx["pairs"][stem]["right_path"]
        label_path = _PROJECT_ROOT / first["label_path"]
        for path, role in ((left_path, f"verify_image_left:{stem}"),
                           (right_path, f"verify_image_right:{stem}"),
                           (label_path, f"verify_label:{stem}")):
            register_consumed(registry, path, role, _PROJECT_ROOT,
                              book.binding_for(path))
        left = load_gray(left_path)
        right = load_gray(right_path)
        height, width = left.shape[:2]
        geometry = build_gt_geometry(
            label_path, width, height,
            padding=int(fix_inputs["crop_pad"]))
        gt_names = sorted(f"{stem}_obj{item.object_index:03d}"
                          for item in geometry)
        ledger_names = sorted(row["sample_name"] for row in samples_ledger
                              if row["frame"] == stem)
        if ledger_names != gt_names:
            problems.append(f"{stem}: the sample ledger differs from the "
                            "frame's GT instances")
            continue
        native_path = cache_dir / "cache" / "native" / f"frame_{stem}.npz"
        grid_path = cache_dir / "cache" / "grid" / f"frame_{stem}.npz"
        perm_path = cache_dir / "cache" / "permutation" / \
            f"frame_{stem}.json"
        if not native_path.is_file() or not grid_path.is_file() or \
                not perm_path.is_file():
            problems.append(f"{stem}: shard missing")
            continue
        saved_native = dict(np.load(native_path))
        saved_grid = dict(np.load(grid_path))
        saved_perm = json.loads(perm_path.read_text(encoding="utf-8"))
        if set(saved_perm) != set(gt_names):
            problems.append(
                f"{stem}: the permutation shard's instance set differs "
                "from the authoritative set")
            continue
        shard_names = sorted({key.split("|")[0] for key in saved_native})
        if shard_names != gt_names:
            problems.append(f"{stem}: the shard samples differ from the "
                            "authoritative GT instances")
            continue
        bands = build_horizontal_bands(
            [item.band_bbox for item in geometry], image_height=height,
            vertical_margin=int(fix_inputs["band_margin"]))
        result = matcher.compute_bands(
            left, right, bands,
            full_image_threshold=float(fix_inputs["full_image_threshold"]))
        right_f = to_gray_u8(right).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(height, dtype=np.float32),
                             np.arange(width, dtype=np.float32),
                             indexing="ij")
        warped_right = cv2.remap(
            right_f, xx - result.disparity.astype(np.float32), yy,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        by_name = {f"{stem}_obj{item.object_index:03d}": item
                   for item in geometry}
        v5_by_name = {row["sample_name"]: row for row in frame_rows}
        for name in gt_names:
            watch.check(f"verify instance {name}")
            item = by_name[name]
            v5_row = v5_by_name[name]
            v5_path = data_root / v5_row["npz_path"]
            register_consumed(registry, v5_path, f"verify_v5_sample:{name}",
                              _PROJECT_ROOT, book.binding_for(v5_path))
            v5_sample = load_mask_sample(v5_path)
            window = (int(v5_row["crop_x1"]), int(v5_row["crop_y1"]),
                      int(v5_row["crop_x2"]), int(v5_row["crop_y2"]))
            # the rebuild-chain identity: V5 class/crop/gray/mask against
            # the GT geometry, the protocol authority and the frozen
            # records — the same checks the builder ran on its own
            # rebuild
            if v5_sample.class_id != item.class_id:
                problems.append(f"{name}: V5 class != GT class")
            authority_identity = authority[name]["identity"]
            if int(v5_row["class_id"]) != \
                    int(authority_identity["class_id"]):
                problems.append(f"{name}: V5 class != protocol authority")
            if window != item.crop_window:
                problems.append(f"{name}: V5 crop window != GT geometry")
            x1, y1, x2, y2 = window
            if not np.array_equal(v5_sample.gray, left[y1:y2, x1:x2]):
                problems.append(
                    f"{name}: the V5 gray crop is not the rectified "
                    "left crop")
            if not np.array_equal(v5_sample.instance_mask,
                                  item.mask[y1:y2, x1:x2]):
                problems.append(
                    f"{name}: the V5 instance mask is not the GT raster")
            stats = matcher.instance_stats(result, item.mask,
                                           item.object_index)
            if stats.valid:
                polar_result = compute_polar_features(
                    left, right, result.disparity, object_mask=item.mask,
                    disparity_valid=result.valid)
                quality, _components = quality_vector_from_result(
                    polar_result, item.mask)
                crop_q = polar_result.signed_q[window[1]:window[3],
                                               window[0]:window[2]]
                crop_v = polar_result.valid_mask[
                    window[1]:window[3],
                    window[0]:window[2]].astype(np.uint8)
                if int(crop_v.sum()) != int(polar_result.valid_mask.sum()):
                    problems.append(
                        f"{name}: the crop window cuts valid polar "
                        "pixels")
                crop_r = np.where(
                    crop_v > 0,
                    warped_right[window[1]:window[3],
                                 window[0]:window[2]].astype(np.float32),
                    np.float32(0.0)).astype(np.float32)
            else:
                quality = build_quality_vector(
                    **{key: 0.0 for key in QUALITY_VECTOR_KEYS})
                crop_q = np.zeros((window[3] - window[1],
                                   window[2] - window[0]), np.float32)
                crop_v = np.zeros((window[3] - window[1],
                                   window[2] - window[0]), np.uint8)
                crop_r = np.zeros((window[3] - window[1],
                                   window[2] - window[0]), np.float32)
            # the native contract, checked on the ACTUAL loaded arrays —
            # independent of the binding's dtype/shape records
            try:
                check_native_arrays(saved_native, name, window[2] -
                                    window[0], window[3] - window[1])
            except (ScaleRefused, SpatialRefused) as exc:
                problems.append(f"{name}: native contract: {exc}")
            expected_native = {
                "L": v5_sample.gray,
                "M": v5_sample.instance_mask.astype(np.uint8),
                "q": crop_q, "V": crop_v, "R": crop_r,
                "quality": np.asarray(quality, dtype=np.float32)}
            for key, expected in expected_native.items():
                saved = saved_native.get(f"{name}|{key}")
                if saved is None:
                    problems.append(f"{name}|{key}: missing from shard")
                    continue
                if saved.dtype != expected.dtype:
                    problems.append(f"{name}|{key}: dtype changed")
                    continue
                if expected.dtype == np.uint8:
                    same = np.array_equal(saved, expected)
                else:
                    same = np.array_equal(saved.view(np.uint32),
                                          expected.view(np.uint32))
                if not same:
                    native_compare["mismatches"].append(f"{name}|{key}")
            from core.polar_region_quality import interior_and_band_masks

            interior, band = interior_and_band_masks(
                v5_sample.instance_mask.astype(np.uint8))
            region_plane = np.zeros_like(interior, dtype=np.uint8)
            region_plane[interior > 0] = 1
            region_plane[band > 0] = 2
            saved_region = saved_native.get(f"{name}|region")
            if saved_region is None or not np.array_equal(saved_region,
                                                          region_plane):
                native_compare["mismatches"].append(f"{name}|region")
            native_compare["instances"] += 1
            # C14/S16 from the verifier's rebuild vs the frozen cache
            fix_arrays = MaskFusionSampleArrays(
                gray=v5_sample.gray, signed_q=crop_q,
                abs_q=np.abs(crop_q), valid=crop_v,
                quality=np.asarray(quality, dtype=np.float32),
                class_id=int(item.class_id),
                instance_mask=v5_sample.instance_mask)
            values, blocks = build_local_polar_stats(fix_arrays)
            c_features = build_local_control_features(fix_arrays)
            row = row_of[name]
            cached_s = np.asarray(cache_arrays["s_features"][row],
                                  dtype=np.float32)
            cached_c = np.asarray(cache_arrays["c_features"][row],
                                  dtype=np.float32)
            if (np.isnan(values) != np.isnan(cached_s)).any() or \
                    not np.array_equal(
                        values[~np.isnan(values)].view(np.uint32),
                        cached_s[~np.isnan(cached_s)].view(np.uint32)):
                problems.append(f"{name}: rebuilt S16 != frozen cache")
            if not np.array_equal(c_features.view(np.uint32),
                                  cached_c.view(np.uint32)):
                problems.append(f"{name}: rebuilt C14 != frozen cache")
            rebuilt_region = region_records(fix_arrays)
            sample_manifest = manifest_samples[row]
            for key in ("mask_pixels", "interior_pixels", "band_pixels",
                        "interior_valid_pixels", "band_valid_pixels",
                        "region_pixels"):
                if int(rebuilt_region[key]) != int(sample_manifest[key]):
                    problems.append(f"{name}: rebuilt region.{key} != "
                                    "frozen manifest")
            # blocks per item against the new ledger AND the frozen
            # manifest; s_valid recomputed by the frozen native-stats
            # definition (not-all-NaN S16) against both sources
            rebuilt_s_valid = not np.isnan(values).all()
            ledger_row = ledger_by_name[name]
            if {k: bool(v) for k, v in blocks.items()} != \
                    {k: bool(v) for k, v in ledger_row["blocks"].items()}:
                problems.append(f"{name}: rebuilt blocks != ledger")
            if {k: bool(v) for k, v in blocks.items()} != \
                    {k: bool(v) for k, v in sample_manifest["blocks"]
                     .items()}:
                problems.append(f"{name}: rebuilt blocks != frozen "
                                "manifest")
            if rebuilt_s_valid != bool(cache_arrays["s_valid"][row]):
                problems.append(f"{name}: rebuilt s_valid != frozen "
                                "cache")
            if rebuilt_s_valid != bool(ledger_row["has_valid_s"]):
                problems.append(f"{name}: rebuilt s_valid != ledger "
                                "has_valid_s")
            # the grid through the independent weight-matrix reference
            grid = _build_grid_planes(
                v5_sample.gray, crop_r, crop_q,
                v5_sample.instance_mask.astype(np.uint8), crop_v,
                interior, band)
            saved_planes = saved_grid.get(f"{name}|planes")
            if saved_planes is None:
                problems.append(f"{name}|planes: missing from shard")
                continue
            if saved_planes.dtype != np.float32 or saved_planes.shape != \
                    (9, GRID_SIZE, GRID_SIZE):
                problems.append(f"{name}|planes: wrong shape/dtype")
                continue
            saved_dict = {plane_name: saved_planes[plane_index]
                          for plane_index, plane_name
                          in enumerate(GRID_PLANES)}
            w2, h2, offset = _independent_letterbox(
                window[2] - window[0], window[3] - window[1])
            try:
                # the strict support cross-check runs against the
                # verifier's OWN exact integer support sets first
                _check_saved_planes_against_independent(
                    saved_dict, grid["supports"], w2, h2, offset)
                # then the numeric contract (ranges, finiteness)
                validate_grid_planes(saved_dict, w2, h2, offset)
            except (ScaleRefused, SpatialRefused) as exc:
                problems.append(f"{name}|planes: {exc}")
                continue
            for plane_index, plane_name in enumerate(GRID_PLANES):
                saved_plane = saved_planes[plane_index]
                rebuilt_plane = grid["planes"][plane_name]
                grid_stats["total_elements"] += int(saved_plane.size)
                grid_stats["nonzero_elements"] += int(
                    np.count_nonzero(saved_plane))
                if not np.allclose(saved_plane, rebuilt_plane,
                                   atol=grid_atol, rtol=GRID_RTOL):
                    grid_stats["mismatches"].append(
                        f"{name}|{plane_name}")
            # the permutation plan: closed-set/type/bijection validation
            # plus the independent re-derivation from the SAVED float32
            # coverage bit patterns
            saved_plan = saved_perm.get(name)
            if saved_plan is None:
                problems.append(f"{name}: permutation plan missing")
                continue
            try:
                check_permutation_plan(saved_plan, name, gt_names)
            except (ScaleRefused, SpatialRefused) as exc:
                problems.append(f"{name}: permutation plan contract: "
                                f"{exc}")
                continue
            rebuilt_plan = _plan_q_permutation_independent(saved_dict,
                                                           name)
            for region in ("interior", "band"):
                if region not in saved_plan.get("regions", {}):
                    continue
                a = rebuilt_plan["regions"][region]
                b = saved_plan["regions"][region]
                if a["movable_cells"] != b["movable_cells"] or \
                        a["moved_indices"] != b["moved_indices"] or \
                        a["value_changed_cells"] != \
                        b["value_changed_cells"] or a["groups"] != \
                        b["groups"]:
                    problems.append(f"{name}|{region}: permutation plan "
                                    "differs from the independent "
                                    "re-derivation")
                perm_stats["plans"] += 1
                perm_stats["support_cells"] += int(a["movable_cells"])
                perm_stats["moved_cells"] += int(a["moved_indices"])
                perm_stats["changed_cells"] += int(
                    a["value_changed_cells"])
                for group in a["groups"].values():
                    if group.get("permuted"):
                        perm_stats["permutable_cells"] += int(
                            group["cells"])
                    else:
                        perm_stats["singleton_cells"] += int(
                            group["cells"])
            # conservation on the SAVED grid planes
            apply_q_permutation({k: np.array(v, dtype=np.float32)
                                 for k, v in saved_dict.items()},
                                saved_plan)
            rebuilt_instances += 1
        log(f"verified {stem}")
    if problems:
        raise ScaleRefused(
            f"the independent verification found {len(problems)} "
            f"problems, first: {problems[:5]}")
    if native_compare["mismatches"]:
        raise ScaleRefused("native mismatches: "
                           f"{native_compare['mismatches'][:5]}")
    if grid_stats["mismatches"]:
        raise ScaleRefused("grid mismatches: "
                           f"{grid_stats['mismatches'][:5]}")
    return {"rebuilt_frames": len(frames),
            "rebuilt_instances": rebuilt_instances,
            "native_instances": native_compare["instances"],
            "grid": grid_stats, "permutation": perm_stats,
            "ledger_rows": len(samples_ledger)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True,
                        choices=("smoke", "full"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    args = parser.parse_args(argv)

    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
              f"{message}", file=sys.stderr, flush=True)

    registry = SourceRegistry({})
    try:
        config_path = Path(args.config)
        if config_path.resolve() != \
                (_PROJECT_ROOT / PINNED_CONFIG_REL).resolve():
            raise ScaleRefused(
                f"{args.config} is not the pinned batch config")
        if not args.expected_binding_sha256 or \
                len(args.expected_binding_sha256) != 64:
            raise ScaleRefused("--expected-binding-sha256 must be a "
                               "64-hex digest from the builder evidence")
        cache_dir = Path(args.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = _PROJECT_ROOT / cache_dir
        resolved_cache = cache_dir.resolve()
        root = (_PROJECT_ROOT / OUTPUT_ROOT).resolve()
        if resolved_cache != root and root not in resolved_cache.parents:
            raise ScaleRefused(
                f"cache dir {resolved_cache} is not inside {root}")
        if not (cache_dir / "cache" / "binding.json").is_file():
            raise ScaleRefused(f"no binding.json under {cache_dir}")
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = _PROJECT_ROOT / output_dir
        expected_output = cache_dir / "verification"
        if output_dir.resolve() != expected_output.resolve():
            raise ScaleRefused(
                f"--output-dir must be the run root's verification "
                f"subdirectory {expected_output}")
        if output_dir.exists():
            raise ScaleRefused(f"verification directory already exists: "
                               f"{output_dir}")
        probe = resolved_cache
        project = _PROJECT_ROOT.resolve()
        while probe != project:
            if is_link_or_reparse(probe):
                raise ScaleRefused(
                    f"link or reparse point on the run path: {probe}")
            probe = probe.parent
        # the binding hash-verifies against the expected digest BEFORE it
        # is parsed
        registry.register(cache_dir / "cache" / "binding.json",
                          "cache_binding",
                          binding={"source": "expected_binding_sha256",
                                   "expected_sha256":
                                       args.expected_binding_sha256})
        registry.register(cache_dir / "builder" / "run_status.json",
                          "builder:run_status", kind="builder_evidence")
        config_record = registry.register(
            config_path, "experiment_config",
            binding={"source": "pinned_config",
                     "expected_sha256": PINNED_CONFIG_SHA256},
            kind="experiment_config")
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if cfg.get("experiment") != "material_spatial_input_v1":
            raise ScaleRefused("the config experiment does not match")
        config_sha = config_record["sha256"]
    except (ScaleRefused, IdentityError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    output_dir.mkdir(parents=True)
    watch = DeadlineWatch(deadline_utc=args.deadline_utc)
    stages: dict = {}
    ctx = {"registry": registry, "watch": watch,
           "stage": (lambda name, status, **details:
                     stages.setdefault(name, {"status": status,
                                              **details})),
           "log": log, "output_dir": output_dir, "mode": args.mode,
           "cache_dir": cache_dir,
           "run_id": cache_dir.resolve().name,
           "config_sha256": config_sha,
           "_cfg": cfg,
           "expected_binding_sha256": args.expected_binding_sha256,
           "deadline_utc": args.deadline_utc,
           "book": ExpectationBook(_PROJECT_ROOT),
           "protocol": None, "pairs": {}, "frame_rows": {}}
    failure = None
    failure_traceback = None
    interrupted = False
    unfinished: list[str] = []
    outcome = {}
    try:
        try:
            watch.check("before verifier admission")
            admission = _admission(ctx, ctx["_cfg"])
            ctx["stage"]("admission_done", "COMPLETED")
            ctx["frame_rows"] = admission["census"]["frame_rows"]
            ctx["pairs"] = admission["census"]["pairs"]
        except DeadlineExceeded as error:
            interrupted = True
            unfinished.append(str(error))
            log(f"deadline exceeded: {error}")
        except (ScaleRefused, IdentityError, SpatialRefused,
                FileNotFoundError, ValueError, OSError, KeyError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            log(f"failure: {failure}")
        if failure is None:
            try:
                watch.check("before the cache identity check")
                binding = json.loads(
                    (cache_dir / "cache" / "binding.json").read_text(
                        encoding="utf-8"))
                artifact_map = {record["path"]: record for record in
                                binding.get("artifacts", [])}
                # register+verify the fixed metadata trio, then parse the
                # samples ledger so the smoke frame set derives from the
                # ledger — never from the binding's own shard report
                for rel in META_RELS:
                    record = artifact_map.get(rel)
                    if record is None:
                        raise ScaleRefused(
                            f"metadata missing from the binding: {rel}")
                    register_consumed(
                        registry, cache_dir / rel, f"cache:metadata:{rel}",
                        _PROJECT_ROOT,
                        {"source": "cache_binding",
                         "expected_sha256": str(record["sha256"]),
                         "expected_bytes": int(record["bytes"])})
                samples_pre = _load_jsonl(cache_dir / "cache" /
                                          "samples.jsonl")
                # the expected artifact set derives from the
                # authoritative identity: three metadata files plus the
                # shards of every non-empty frame the mode must cover
                authoritative = authoritative_frame_instances(
                    admission["manifest"])
                if args.mode == "full":
                    built_frames = sorted(admission["census"][
                        "frame_rows"])
                else:
                    built_frames = sorted({row["frame"]
                                           for row in samples_pre})
                expected_rels = expected_artifact_paths(
                    authoritative, built_frames)
                inventory = verify_inventory(cache_dir, binding, registry,
                                             _PROJECT_ROOT, expected_rels)
                stages["inventory"] = {"status": "COMPLETED",
                                       **inventory}
                definition, samples_ledger, frames_ledger = \
                    _verify_cache_identity(ctx, admission, cache_dir,
                                           binding)
                ctx["samples_ledger"] = samples_ledger
                ctx["frames_ledger"] = frames_ledger
                ctx["definition"] = definition
            except DeadlineExceeded as error:
                interrupted = True
                unfinished.append(str(error))
                log(f"deadline exceeded: {error}")
            except (ScaleRefused, IdentityError, SpatialRefused,
                    FileNotFoundError, ValueError, OSError,
                    KeyError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                log(f"failure: {failure}")
        if failure is None:
            try:
                watch.check("before the independent rebuild")
                if args.mode == "smoke":
                    frames = sorted({row["frame"]
                                     for row in ctx["samples_ledger"]})
                else:
                    frames = sorted(ctx["frame_rows"])
                outcome = _verify_native_rebuild(
                    ctx, admission, cache_dir, frames,
                    ctx["samples_ledger"])
                if args.mode == "full":
                    pool = admission["cfg"]["dev_pool"]
                    if outcome["rebuilt_frames"] != \
                            int(pool["nonempty_frames"]) or \
                            outcome["rebuilt_instances"] != \
                            int(pool["instances"]) or \
                            outcome["ledger_rows"] != \
                            int(pool["instances"]):
                        raise ScaleRefused(
                            "the verified counts do not meet the "
                            "approved dev pool "
                            f"({pool['nonempty_frames']} frames / "
                            f"{pool['instances']} instances)")
                stages["independent_rebuild"] = {"status": "COMPLETED",
                                                 **outcome}
            except DeadlineExceeded as error:
                interrupted = True
                unfinished.append(str(error))
                log(f"deadline exceeded: {error}")
            except (ScaleRefused, IdentityError, SpatialRefused,
                    FileNotFoundError, ValueError, OSError,
                    KeyError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                log(f"failure: {failure}")
            except Exception as exc:
                import traceback

                failure = f"UNEXPECTED {type(exc).__name__}: {exc}"
                failure_traceback = traceback.format_exc()
                log(f"unexpected failure: {failure}")
    except Exception as exc:  # safety net
        failure = f"UNEXPECTED {type(exc).__name__}: {exc}"
    try:
        protection = registry.compare()
        registry.write_manifest(output_dir / "source_manifest.json",
                                protection)
        stages["protection"] = {
            "status": "COMPLETED" if protection["identical"] else "FAILED",
            "registered": protection["registered_sources"],
            "problems": protection["problems"]}
        ctx["protection"] = protection
        if not protection["identical"] and failure is None:
            failure = "protection check found changed inputs"
        # the binding itself must still match the expected digest after
        # everything was consumed (consumption-period drift)
        binding_now = sha256_of_file(cache_dir / "cache" / "binding.json")
        stages["final_binding"] = {
            "status": "COMPLETED"
            if binding_now == args.expected_binding_sha256 else "FAILED",
            "sha256": binding_now}
        if binding_now != args.expected_binding_sha256 and failure is None:
            failure = "the cache binding changed during consumption"
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
        status, code = "VERIFIED", EXIT_OK
    _write_status(output_dir, ctx, args, stages, status, code, outcome,
                  failure, failure_traceback, interrupted, unfinished,
                  started)
    log(f"overall: {status} (exit {code})")
    return code


def _write_status(output_dir: Path, ctx: dict, args, stages: dict,
                  status: str, code: int, outcome: dict, failure,
                  failure_traceback, interrupted: bool, unfinished: list,
                  started: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode, "experiment": "material_spatial_input_v1",
        "run_id": ctx["run_id"],
        "config_sha256": ctx["config_sha256"],
        "deadline_utc": ctx["deadline_utc"],
        "cache_binding_sha256": ctx["expected_binding_sha256"],
        "cache_schema": CACHE_SCHEMA,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "stages": stages, "interrupted_by_deadline": interrupted,
        "unfinished": unfinished, "failure": failure,
        "failure_traceback": failure_traceback,
        "overall_status": status, "exit_code": code,
        "input_admitted": False,
        "outcome": outcome, "elapsed_s": time.time() - started,
        "protection": ctx.get("protection"),
        "zero_training_calls": {"model_fits": 0, "scaler_fits": 0,
                                "classifier_forwards": 0,
                                "encoder_forwards": 0,
                                "optimizer_steps": 0},
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps({
            "mode": args.mode, "run_id": ctx["run_id"],
            "experiment": "material_spatial_input_v1",
            "config_sha256": ctx["config_sha256"],
            "deadline_utc": ctx["deadline_utc"],
            "cache_binding_sha256": ctx["expected_binding_sha256"],
            "cache_schema": CACHE_SCHEMA,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "outcome": outcome, "failure": failure,
            "problems": ctx.get("protection", {}).get("problems", []),
            "grid_tolerance": {"atol": GRID_ATOL, "rtol": GRID_RTOL},
            "counts": {"rebuilt_frames": outcome.get("rebuilt_frames"),
                       "rebuilt_instances":
                           outcome.get("rebuilt_instances"),
                       "ledger_rows": outcome.get("ledger_rows")},
            "note": ("all grid comparisons used the verifier's own exact "
                     "integer weight-matrix reference at atol=1e-6, "
                     "rtol=0; native arrays were compared bitwise; the "
                     "authoritative instance sets came from the frozen "
                     "manifest/V5/protocol, never from the saved shard "
                     "keys"),
        }, ensure_ascii=False, indent=1) + chr(10), encoding="utf-8")
    payload["report_sha256"] = sha256_of_file(report_path)
    (output_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + chr(10),
        encoding="utf-8")


def canonical_key(path: Path) -> str:
    from core.material_spatial_input_admission import canonical_rel

    return canonical_rel(path, _PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
