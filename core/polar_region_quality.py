"""Deterministic reliability diagnostics for the polarimetric S region.

Batch 2A context: the frozen residual protocol's first real fit found no
polarization benefit (RCS within noise of R0/RC), so the measurement
region of the S slot is now under investigation. Everything in this
module is deterministic algebra and aggregation over SAVED V5 arrays:

- no model forward, no encoder, no P0 head;
- no stereo (re-)matching, no SGBM, no disparity re-estimation;
- no polarization re-measurement from raw frames.

The eight S statistics keep their frozen definitions
(``core.material_probe.POLAR_STAT_KEYS``); the proposed single rule may
only change the pixel set they are computed on. The rule keeps the
missing-state contract intact by construction: when the reliable region
is empty it falls back to the ORIGINAL region, so ``has_valid_s``, the
NaN-row rule and the C ``s_region_nonempty_flag`` column cannot change.

Measurement-semantics boundary (2A corrective batch): the four-direction
shift experiment has ``perturbation_type = saved_q_roi_sampling_shift``.
It resamples the ALREADY-COMPUTED ``signed_q``/``abs_q``/``valid`` maps
against the fixed annotation; it does NOT fix the left image, displace
the right image and recompute q. It therefore says nothing about
left-right registration, exposure/gain/sync consistency, or disparity
failure modes, and its distribution summaries (mean/p95 relative to the
between-sample spread) are NOT error bounds — the observed per-sample
maximum is reported separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.material_mask_dataset import load_mask_sample
from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    QUANTILE_METHOD,
    QUANTILE_PERCENTILES,
    build_cache_binding,
    authoritative_sample_index,
    verify_cache_arrays,
    verify_cache_binding,
    verify_cache_manifest,
)

PARENT_FEATURE_DEFINITION_VERSION = "existing_data_probe_v1"
RELIABLE_REGION_DEFINITION_VERSION = (
    "existing_data_probe_v1_reliable_region_interior_v1")

# Frozen in the part-2B feature preparation, before any residual result
# was seen; never redefined afterwards.
INTERIOR_DEFINITION = "erode3x3_instance_mask_v1"
BAND_DEFINITION = "instance_mask_minus_interior_v1"

# The single proposed rule (batch 2A): S statistics on the eroded
# interior; when it holds no valid pixel, fall back to the original
# region so the missing state is preserved exactly.
REGION_RULE_ID = "interior_erode3x3_fallback_original_v1"
REGION_RULE_NOTE = (
    "S statistics are computed on interior(erode3x3 instance_mask) & "
    "polar_valid; when that region is empty the ORIGINAL region "
    "(instance_mask & polar_valid) is used and the sample is flagged as "
    "a fallback. has_valid_s is therefore identical to the original "
    "definition for every sample, and G/C/P0 are untouched.")

# The sampling perturbation: content shifts by one pixel relative to the
# fixed annotation (a 1-px misregistration model). Vacated edge pixels
# become unmeasurable.
PERTURBATION_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# What the shift experiment actually measures: a resampling of the
# ALREADY-SAVED q maps against the fixed annotation. It is NOT a
# left-right misregistration recompute experiment.
PERTURBATION_TYPE = "saved_q_roi_sampling_shift"

# The statistics whose drift is highlighted in reports.
PERTURBATION_HIGHLIGHT = ("signed_q_mean", "signed_q_p50", "abs_q_mean")

# Descriptive diagnostic levels (NOT rule thresholds; no threshold is
# fitted in this batch).
SATURATION_GRAY = 250
DARK_GRAY = 10
DENOM_LEVELS = (20.0, 30.0, 50.0, 100.0)
Q_HEAVY = 0.5

# Generation-time constants of core.polar_compute.compute_polar_features
# as V4 called it (right_gain=1.0), used by the denominator recovery.
_MIN_INTENSITY_SUM_BOUND = 10.0


# ---------------------------------------------------------------------------
# Region primitives


def interior_and_band_masks(instance_mask: np.ndarray):
    """The frozen interior / contour-band split (single 3x3 erosion)."""
    mask = (np.asarray(instance_mask) > 0).astype(np.uint8)
    interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    band = mask & (1 - interior)
    return interior.astype(bool), band.astype(bool)


def _frac(numerator: float, denominator: float):
    """A JSON-safe ratio; ``None`` rather than 0/0 or NaN."""
    if denominator == 0:
        return None
    return float(numerator / denominator)


def region_polar_stats(signed_q: np.ndarray, abs_q: np.ndarray,
                       region: np.ndarray):
    """The frozen eight S statistics on an explicit pixel region.

    Identical to :func:`core.material_probe.build_polar_stats` when the
    region is ``polar_valid & instance_mask``. An empty region is an
    explicit miss (NaN values, ``nonempty=False``); non-finite values
    INSIDE the region mean the sample is broken and are refused.
    """
    signed_q = np.asarray(signed_q, dtype=np.float32)
    abs_q = np.asarray(abs_q, dtype=np.float32)
    region = np.asarray(region, dtype=bool)
    if region.shape != signed_q.shape or region.shape != abs_q.shape:
        raise ValueError(
            f"region shape {region.shape} does not match the q maps "
            f"{signed_q.shape}")
    count = int(region.sum())
    if count == 0:
        return (np.full(len(POLAR_STAT_KEYS), np.nan, dtype=np.float32),
                False)
    signed_values = signed_q[region]
    abs_values = abs_q[region]
    if not np.isfinite(signed_values).all() or \
            not np.isfinite(abs_values).all():
        raise ValueError(
            "non-finite polarization value inside the measurement region; "
            "the sample cannot be measured")
    quantiles = np.percentile(signed_values, QUANTILE_PERCENTILES,
                              method="linear")
    values = (
        float(signed_values.mean()),
        float(signed_values.std(ddof=0)),
        *(float(q) for q in quantiles),
        float(abs_values.mean()),
    )
    return np.asarray(values, dtype=np.float32), True


def shift_content(gray: np.ndarray, signed_q: np.ndarray, abs_q: np.ndarray,
                  valid: np.ndarray, delta: tuple[int, int]):
    """Shift the measurement content by integer (dy, dx).

    The annotation (instance mask) stays fixed; vacated edge pixels
    become 0 / unmeasurable. This models a 1-px misregistration between
    the annotated region and the ALREADY-COMPUTED content
    (``perturbation_type = saved_q_roi_sampling_shift``): it resamples
    the saved q maps and does NOT displace the right image and recompute
    q, so it cannot validate left-right registration.
    """
    dy, dx = int(delta[0]), int(delta[1])
    h, w = np.asarray(gray).shape[:2]

    def _shift(source, fill):
        out = np.full_like(source, fill)
        ys_dst = slice(max(dy, 0), min(h + dy, h))
        ys_src = slice(ys_dst.start - dy, ys_dst.stop - dy)
        xs_dst = slice(max(dx, 0), min(w + dx, w))
        xs_src = slice(xs_dst.start - dx, xs_dst.stop - dx)
        out[ys_dst, xs_dst] = source[ys_src, xs_src]
        return out

    return (
        _shift(np.asarray(gray), 0),
        _shift(np.asarray(signed_q, dtype=np.float32), 0),
        _shift(np.asarray(abs_q, dtype=np.float32), 0),
        _shift(np.asarray(valid, dtype=bool), False),
    )


def recover_denominator(gray: np.ndarray, signed_q: np.ndarray,
                        valid: np.ndarray):
    """Recover the generation-time denominator ``L + warp(R)`` per pixel.

    At a valid pixel the saved ``signed_q`` is
    ``(L - R') / (L + R' + eps)`` with ``eps = 1e-6`` and gain 1.0 (the
    V4 generation call), so ``L + R' = 2L / (1 + q)`` for ``q > -1`` —
    plain algebra on the saved arrays, not a re-measurement. ``q == -1``
    means ``L == 0`` exactly; the sum is then only known to be at least
    the generation-time ``min_intensity_sum`` (10), so the pixel is
    reported as unknown and excluded from below-threshold fractions.
    """
    left = np.asarray(gray).astype(np.float64)
    q = np.asarray(signed_q).astype(np.float64)
    valid = np.asarray(valid, dtype=bool)
    known = valid & (q > -1.0)
    denom = np.full(left.shape, np.nan, dtype=np.float64)
    denom[known] = 2.0 * left[known] / (1.0 + q[known])
    return denom, known


# ---------------------------------------------------------------------------
# Per-sample diagnostics


def _degenerate_block(region: np.ndarray, gray: np.ndarray,
                      signed_q: np.ndarray, denom: np.ndarray,
                      known: np.ndarray) -> dict:
    region = np.asarray(region, dtype=bool)
    count = int(region.sum())
    gray_region = gray[region]
    abs_q_region = np.abs(signed_q[region].astype(np.float64))
    denom_region = denom[region]
    known_region = known[region]
    below = {}
    for level in DENOM_LEVELS:
        key = str(float(level))
        below[key] = _frac(
            int((known_region & (denom_region < level)).sum()),
            int(known_region.sum()))
    return {
        "pixels": count,
        "gray_eq_255": _frac(int((gray_region == 255).sum()), count),
        "gray_ge_250": _frac(
            int((gray_region >= SATURATION_GRAY).sum()), count),
        "gray_eq_0": _frac(int((gray_region == 0).sum()), count),
        "gray_le_10": _frac(int((gray_region <= DARK_GRAY).sum()), count),
        "abs_q_eq_1": _frac(
            int((abs_q_region >= 1.0 - 1e-7).sum()), count),
        "denom_known_share": _frac(int(known_region.sum()), count),
        "denom_below": below,
    }


def sample_region_diagnostics(sample, *,
                              deltas: tuple = PERTURBATION_DELTAS) -> dict:
    """All deterministic per-sample region diagnostics in one pass.

    ``sample`` is a V5-style sample with ``gray``/``signed_q``/``abs_q``/
    ``valid``/``instance_mask``. Nothing is fitted; every number is a
    count, ratio or frozen statistic over the saved arrays.
    """
    gray = np.asarray(sample.gray)
    signed_q = np.asarray(sample.signed_q, dtype=np.float32)
    abs_q = np.asarray(sample.abs_q, dtype=np.float32)
    valid = np.asarray(sample.valid, dtype=bool)
    mask = np.asarray(sample.instance_mask) > 0
    interior, band = interior_and_band_masks(sample.instance_mask)

    region = mask & valid
    region_interior = interior & valid
    region_band = band & valid
    denom, known = recover_denominator(gray, signed_q, valid)

    original_stats, original_has = region_polar_stats(signed_q, abs_q,
                                                      region)
    interior_stats, interior_has = region_polar_stats(
        signed_q, abs_q, region_interior)
    band_stats, band_has = region_polar_stats(signed_q, abs_q, region_band)

    counts = {
        "mask_pixels": int(mask.sum()),
        "interior_pixels": int(interior.sum()),
        "band_pixels": int(band.sum()),
        "region_pixels": int(region.sum()),
        "interior_valid_pixels": int(region_interior.sum()),
        "band_valid_pixels": int(region_band.sum()),
    }
    ratios = {
        "interior_valid_ratio": _frac(counts["interior_valid_pixels"],
                                      counts["interior_pixels"]),
        "band_valid_ratio": _frac(counts["band_valid_pixels"],
                                  counts["band_pixels"]),
        "region_coverage_of_mask": _frac(counts["region_pixels"],
                                         counts["mask_pixels"]),
    }

    # Where do the extreme |q| pixels sit? The band's pixel share versus
    # its share among heavy-|q| pixels shows whether the contour ring
    # contributes disproportionate content to the pooled statistics.
    heavy = region & (np.abs(signed_q) >= Q_HEAVY)
    band_influence = {
        "heavy_pixels": int(heavy.sum()),
        "region_pixel_share_band": _frac(counts["band_valid_pixels"],
                                         counts["region_pixels"]),
        "heavy_q_share_band": _frac(int((heavy & band).sum()),
                                    int(heavy.sum())),
        "mean_abs_q_interior": (float(interior_stats[7])
                                if interior_has else None),
        "mean_abs_q_band": float(band_stats[7]) if band_has else None,
    }

    truncation = {
        "mask_touches_crop_border": bool(
            mask[0].any() or mask[-1].any() or mask[:, 0].any()
            or mask[:, -1].any()),
        "border_mask_pixels": int(
            mask[0].sum() + mask[-1].sum() + mask[:, 0].sum()
            + mask[:, -1].sum()),
    }

    degenerate = {
        "region": _degenerate_block(region, gray, signed_q, denom, known),
        "interior": _degenerate_block(region_interior, gray, signed_q,
                                      denom, known),
        "band": _degenerate_block(region_band, gray, signed_q, denom,
                                  known),
    }

    # The +-1 px sampling perturbation, for the original region and for
    # the proposed rule region (including its fallback), against their
    # own unshifted statistics.
    rule_base_stats, rule_base_has = _rule_region_parts(
        mask, interior, signed_q, abs_q, valid)
    per_delta = {}
    max_original: dict[str, float] = {}
    max_rule: dict[str, float] = {}
    for delta in deltas:
        _, signed2, abs2, valid2 = shift_content(
            gray, signed_q, abs_q, valid, delta)
        shifted_region = mask & valid2
        shifted_stats, shifted_has = region_polar_stats(
            signed2, abs2, shifted_region)
        rule_stats, rule_has = _rule_stats_from_parts(
            mask, interior, signed2, abs2, valid2)
        entry = {
            "shifted_nonempty": shifted_has,
            "abs_delta": _abs_delta_per_key(original_stats, original_has,
                                            shifted_stats, shifted_has),
            "rule_shifted_nonempty": rule_has,
            "abs_delta_rule": _abs_delta_per_key(
                rule_base_stats, rule_base_has, rule_stats, rule_has),
        }
        for key, value in (entry["abs_delta"] or {}).items():
            if value is not None:
                max_original[key] = max(max_original.get(key, 0.0), value)
        for key, value in (entry["abs_delta_rule"] or {}).items():
            if value is not None:
                max_rule[key] = max(max_rule.get(key, 0.0), value)
        per_delta[str((int(delta[0]), int(delta[1])))] = entry

    return {
        "counts": counts,
        "ratios": ratios,
        "original_stats": (original_stats.tolist() if original_has
                           else None),
        "interior_stats": (interior_stats.tolist() if interior_has
                           else None),
        "band_stats": (band_stats.tolist() if band_has else None),
        "band_influence": band_influence,
        "truncation": truncation,
        "degenerate": degenerate,
        "perturbation": {
            "perturbation_type": PERTURBATION_TYPE,
            "deltas": [str((int(d[0]), int(d[1]))) for d in deltas],
            "per_delta": per_delta,
            "max_abs_delta_original": max_original,
            "max_abs_delta_rule": max_rule,
            "interpretation_limit": (
                "per-sample worst-case resampling drift of the SAVED q "
                "maps under +-1px ROI shifts; not a left-right "
                "registration check and not an error bound"),
        },
    }


def _abs_delta_per_key(base_stats, base_has, shifted_stats, shifted_has):
    """Per-key |delta| between two statistic vectors (JSON-safe)."""
    if not base_has or not shifted_has:
        return None
    keys = list(POLAR_STAT_KEYS)
    deltas = np.abs(
        np.asarray(shifted_stats, dtype=np.float64)
        - np.asarray(base_stats, dtype=np.float64))
    return {key: float(value) for key, value in zip(keys, deltas)}


def _rule_region(mask: np.ndarray, interior: np.ndarray,
                 valid: np.ndarray):
    """The proposed rule's pixel region (with the original fallback).

    The fallback flag is only set when the fallback region is actually
    USED (nonempty); a sample measurable nowhere has no fallback and no
    valid S — exactly the original missing state.
    """
    region_interior = interior & valid
    if region_interior.any():
        return region_interior, False, "interior_erode3x3"
    region_original = mask & valid
    if region_original.any():
        return region_original, True, "fallback_original"
    return region_original, False, "none"


def _rule_stats_from_parts(mask, interior, signed_q, abs_q, valid):
    region, _, _ = _rule_region(mask, interior, valid)
    return region_polar_stats(signed_q, abs_q, region)


def _rule_region_parts(mask, interior, signed_q, abs_q, valid):
    """(base_stats, base_has) of the rule on the UNSHIFTED content."""
    region, _, _ = _rule_region(mask, interior, valid)
    stats, has = region_polar_stats(signed_q, abs_q, region)
    return stats, has


def evaluate_region_rule(sample) -> dict:
    """Apply the single proposed rule to one sample.

    Returns the chosen region, the fallback flag, ``has_valid_s`` and
    the eight statistics on the chosen region. By construction
    ``has_valid_s`` equals the ORIGINAL definition for every sample: an
    empty reliable region falls back to the original region.
    """
    gray = np.asarray(sample.gray)
    signed_q = np.asarray(sample.signed_q, dtype=np.float32)
    abs_q = np.asarray(sample.abs_q, dtype=np.float32)
    valid = np.asarray(sample.valid, dtype=bool)
    mask = np.asarray(sample.instance_mask) > 0
    interior, _ = interior_and_band_masks(sample.instance_mask)

    region, fallback, chosen = _rule_region(mask, interior, valid)
    stats, has = region_polar_stats(signed_q, abs_q, region)
    return {
        "rule_id": REGION_RULE_ID,
        "chosen_region": chosen,
        "fallback_used": bool(fallback),
        "region_pixels": int(region.sum()),
        "interior_valid_pixels": int((interior & valid).sum()),
        "has_valid_s": bool(has),
        "stats": stats.tolist() if has else None,
    }


# ---------------------------------------------------------------------------
# Correspondence: saved crop vs the rectified left frame


def verify_sample_correspondence(sample, left_gray_full: np.ndarray,
                                 crop_box: tuple[int, int, int, int],
                                 *, expected_mask_pixels: int) -> dict:
    """Deterministic crop/mask/left-frame correspondence check.

    ``crop_box`` is ``(y1, x1, y2, x2)`` — the crop window the V4
    generator used on the rectified left frame. The saved ``gray`` must
    equal that window exactly, the crop must lie inside the frame, and
    the instance-mask pixel count must match the manifest record. Any
    mismatch raises ``ValueError``.
    """
    y1, x1, y2, x2 = (int(v) for v in crop_box)
    left = np.asarray(left_gray_full)
    h, w = left.shape[:2]
    if not (0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w):
        raise ValueError(
            f"crop box {(y1, x1, y2, x2)} lies outside the "
            f"{h}x{w} left frame")
    crop = left[y1:y2, x1:x2]
    if crop.shape != np.asarray(sample.gray).shape or \
            not np.array_equal(crop, np.asarray(sample.gray)):
        raise ValueError(
            "saved gray does not exactly match the rectified left frame "
            f"crop {crop_box}")
    mask_pixels = int(np.asarray(sample.instance_mask).sum())
    if mask_pixels != int(expected_mask_pixels):
        raise ValueError(
            f"mask pixel count {mask_pixels} != manifest record "
            f"{expected_mask_pixels}")
    return {
        "gray_identical": True,
        "mask_pixels_match": True,
        "crop_within_image": True,
        "crop_box": [y1, x1, y2, x2],
    }


# ---------------------------------------------------------------------------
# Reliable-region cache machinery (the REAL build waits for the user's
# approval of the rule; this builder is exercised on synthetic chains)


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def build_reliable_region_cache(*, source_feature_dir, data_root, protocol,
                                output_dir) -> dict:
    """Build an independent reliable-region feature cache.

    Row order, G, C, ``s_valid`` and ``class_id`` are carried over
    unchanged; only the eight S values are recomputed on the rule
    region. The source directory is strictly read-only and the output
    directory must not exist. The new cache declares its own feature
    definition version, so the FROZEN residual entry (which pins
    ``existing_data_probe_v1``) refuses it until that entry is
    explicitly extended — a documented follow-up decision, not a silent
    swap.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(
            f"output directory {output_dir} already exists; reliable-"
            "region caches are written to fresh directories only")
    source_feature_dir = Path(source_feature_dir)
    try:
        verify_cache_binding(source_feature_dir, protocol)
    except Exception as error:
        raise ValueError(
            "source cache binding verification failed: "
            f"{type(error).__name__}: {error}") from error
    source_definition = _load_json(
        source_feature_dir / "feature_definition.json")
    if source_definition.get("feature_definition_version") != \
            PARENT_FEATURE_DEFINITION_VERSION:
        raise ValueError(
            "source feature definition version "
            f"{source_definition.get('feature_definition_version')!r} is "
            f"not {PARENT_FEATURE_DEFINITION_VERSION!r}")
    source_manifest = _load_json(source_feature_dir
                                 / "feature_manifest.json")
    authority = authoritative_sample_index(protocol)
    samples = verify_cache_manifest(source_manifest, authority, protocol)
    source_cache = dict(np.load(source_feature_dir / "feature_cache.npz"))
    verify_cache_arrays(source_cache, samples, source_definition, protocol)

    data_root = Path(data_root)
    n = len(samples)
    s_features = np.full((n, len(POLAR_STAT_KEYS)), np.nan,
                         dtype=np.float32)
    region_records = []
    fallback_count = 0
    for row, manifest_row in enumerate(samples):
        sample = load_mask_sample(data_root / manifest_row["npz_path"])
        if sample.class_id != int(manifest_row["class_id"]):
            raise ValueError(
                f"{manifest_row['npz_path']}: class_id {sample.class_id} "
                f"!= manifest {manifest_row['class_id']}")
        decision = evaluate_region_rule(sample)
        if decision["has_valid_s"] != bool(manifest_row["has_valid_s"]):
            raise ValueError(
                f"{manifest_row['sample_name']}: the rule changed the "
                "missing state; the fallback contract is broken")
        if decision["has_valid_s"]:
            s_features[row] = decision["stats"]
        fallback_count += int(decision["fallback_used"])
        region_records.append({
            "s_region_rule_id": REGION_RULE_ID,
            "s_region_chosen": decision["chosen_region"],
            "s_region_fallback": bool(decision["fallback_used"]),
            "s_region_pixels": decision["region_pixels"],
            "s_region_interior_valid_pixels":
                decision["interior_valid_pixels"],
        })

    s_valid = np.asarray(source_cache["s_valid"]).copy()
    if not np.array_equal(s_valid, np.asarray(
            [bool(row["has_valid_s"]) for row in samples])):
        raise ValueError(
            "source s_valid disagrees with the manifest has_valid_s; "
            "refusing to propagate an inconsistent missing state")

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "feature_cache.npz",
        g_features=source_cache["g_features"],
        c_features=source_cache["c_features"],
        s_features=s_features,
        s_valid=s_valid,
        class_id=source_cache["class_id"])
    new_manifest = {
        "count": n,
        "samples": [
            {**dict(row), **record}
            for row, record in zip(samples, region_records)
        ],
    }
    _write_json(output_dir / "feature_manifest.json", new_manifest)
    definition = {
        "feature_definition_version": RELIABLE_REGION_DEFINITION_VERSION,
        "parent_feature_definition_version":
            PARENT_FEATURE_DEFINITION_VERSION,
        "data_fingerprint": protocol["data"]["fingerprint"],
        "protocol_sha256": protocol["protocol_sha256"],
        "official_mit_b0": dict(source_definition["official_mit_b0"]),
        "class_names": list(protocol["data"]["class_names"]),
        "imgsz": source_definition.get("imgsz"),
        "preprocessing_version":
            source_definition.get("preprocessing_version"),
        "region": {
            "rule_id": REGION_RULE_ID,
            "rule_note": REGION_RULE_NOTE,
            "interior_definition": INTERIOR_DEFINITION,
            "band_definition": BAND_DEFINITION,
            "fallback": "original_region",
        },
        "slot_G": dict(source_definition["slot_G"]),
        "slot_C": dict(source_definition["slot_C"]),
        "slot_S": {
            "keys": list(POLAR_STAT_KEYS),
            "missing_rule": (
                "has_valid_s=false, values NaN in npz / null in json "
                "(unchanged; the fallback keeps has_valid_s identical to "
                "the original definition)"),
            "quantile_method": QUANTILE_METHOD,
            "region": REGION_RULE_ID,
        },
    }
    _write_json(output_dir / "feature_definition.json", definition)
    build_cache_binding(output_dir, protocol)

    # the frozen verification chain must accept the new directory
    verify_cache_binding(output_dir, protocol)
    verify_cache_manifest(_load_json(output_dir / "feature_manifest.json"),
                          authority, protocol)
    verify_cache_arrays(dict(np.load(output_dir / "feature_cache.npz")),
                        samples, definition, protocol)
    return {
        "rule_id": REGION_RULE_ID,
        "definition_version": RELIABLE_REGION_DEFINITION_VERSION,
        "sample_count": n,
        "fallback_count": fallback_count,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# Protection evidence (2A corrective batch)
#
# The historical 2A snapshot used a wrong project-root derivation and
# recorded every target as {"missing": true}; two all-missing objects
# compared equal and proved nothing. These helpers make the required
# assets explicit, refuse missing ones, and compare file SETS plus
# per-file digests and byte lengths.


class ProtectionError(RuntimeError):
    """Raised when protection evidence cannot be established or compared."""


def locate_project_root(entry_file) -> Path:
    """The project root of a formal entry living in ``<root>/scripts/``.

    Derived from the entry file's own location (CWD-independent), then
    validated against a marker module so a moved or copied file fails
    instead of silently protecting the wrong tree.
    """
    root = Path(entry_file).resolve().parents[1]
    marker = root / "core" / "polar_region_quality.py"
    if not marker.is_file():
        raise ProtectionError(
            f"project root marker missing: {marker}; the entry does not "
            "sit in a recognised <root>/scripts/ layout")
    return root


def _real_path(path: Path) -> Path:
    """The strictly resolved location of ``path`` (junction/symlink
    aware). A separate helper so tests can simulate resolutions that
    Windows cannot create without elevation."""
    import os

    return Path(os.path.realpath(path))


def _snapshot_file(path: Path) -> dict:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"kind": "file", "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def build_protection_snapshot(project_root, targets) -> dict:
    """Digest every target (file or directory tree) under one root.

    Directories record their file set with per-file sha256 and byte
    length. A target that resolves outside the project root is refused
    (path/link escape); a missing target is recorded as
    ``{"kind": "missing"}`` and must be checked by the caller via
    :func:`missing_required_targets` — two all-missing snapshots never
    compare equal (see :func:`compare_protection_snapshots`).
    """
    import os

    project_root = Path(project_root).resolve()
    root_real = Path(os.path.realpath(project_root))
    records: dict[str, dict] = {}
    for target in targets:
        path = Path(target)
        absolute = path if path.is_absolute() else project_root / path
        if not absolute.exists():
            records[str(target)] = {"kind": "missing"}
            continue
        real = _real_path(absolute)
        if real != root_real and root_real not in real.parents:
            raise ProtectionError(
                f"protection target {target!r} resolves to {real}, outside "
                f"the project root {root_real}")
        if absolute.is_dir():
            files = {}
            # every descendant — including junction/symlink nodes, which
            # rglob may not descend into — is resolution-checked BEFORE
            # its content would be read or hashed, so an escaping link
            # can never be silently treated as an empty subtree
            for entry_path in sorted(absolute.rglob("*")):
                entry_real = _real_path(entry_path)
                if entry_real != root_real                         and root_real not in entry_real.parents:
                    raise ProtectionError(
                        f"protection target {str(target)!r}: descendant "
                        f"{entry_path.relative_to(absolute).as_posix()!r} "
                        f"resolves to {entry_real}, outside the project "
                        f"root {root_real}")
                if entry_path.is_file():
                    relative = entry_path.relative_to(absolute).as_posix()
                    files[relative] = _snapshot_file(entry_path)
            records[str(target)] = {
                "kind": "dir", "file_count": len(files), "files": files,
            }
        else:
            records[str(target)] = _snapshot_file(absolute)
    return {"project_root": str(root_real), "targets": records}


def missing_required_targets(snapshot: dict) -> list[str]:
    """The snapshot targets that do not exist on disk."""
    return sorted(name for name, record in snapshot["targets"].items()
                  if record.get("kind") == "missing")


def compare_protection_snapshots(before: dict, after: dict) -> dict:
    """Independent re-computation comparison of two protection snapshots.

    Checks the target set, the per-target kind, directory file SETS and
    every file's sha256 and byte length. Any missing target, any set
    mismatch and any digest/length difference raises ``ProtectionError``;
    record counts alone never prove equality.
    """
    mismatches: list[str] = []
    before_targets = before.get("targets", {})
    after_targets = after.get("targets", {})
    if set(before_targets) != set(after_targets):
        raise ProtectionError(
            "protection target sets differ: "
            f"{sorted(set(before_targets) ^ set(after_targets))}")
    total_files = 0
    for name in sorted(before_targets):
        old = before_targets[name]
        new = after_targets[name]
        if old.get("kind") == "missing" or new.get("kind") == "missing":
            mismatches.append(f"{name}: missing (before kind "
                              f"{old.get('kind')!r}, after kind "
                              f"{new.get('kind')!r})")
            continue
        if old["kind"] != new["kind"]:
            mismatches.append(f"{name}: kind changed {old['kind']!r} -> "
                              f"{new['kind']!r}")
            continue
        if old["kind"] == "file":
            total_files += 1
            if (old["sha256"], old["bytes"]) != (new["sha256"],
                                                 new["bytes"]):
                mismatches.append(
                    f"{name}: file digest/length changed")
            continue
        old_files = old.get("files", {})
        new_files = new.get("files", {})
        if set(old_files) != set(new_files):
            difference = sorted(set(old_files) ^ set(new_files))[:5]
            mismatches.append(f"{name}: directory file set changed "
                              f"(first: {difference})")
            continue
        total_files += len(old_files)
        for relative in sorted(old_files):
            a = old_files[relative]
            b = new_files[relative]
            if (a["sha256"], a["bytes"]) != (b["sha256"], b["bytes"]):
                mismatches.append(f"{name}/{relative}: digest/length "
                                  "changed")
    if mismatches:
        raise ProtectionError(
            f"protection comparison failed ({len(mismatches)} mismatches, "
            f"first: {mismatches[:3]})")
    return {"passed": True, "targets": len(before_targets),
            "files": total_files, "mismatches": []}


# ---------------------------------------------------------------------------
# Drift summaries (2A corrective batch): mean/p50/p95/max are reported
# separately; only ratios against the between-sample spread are derived,
# and they are distribution summaries, NOT error bounds.


def drift_statistics(values) -> dict:
    """Mean/p50/p95/max of per-sample worst-case drift values.

    An empty subset returns ``n=0`` with ``None`` statistics — never a
    fabricated zero "success". Non-finite inputs are refused (the JSON
    contract forbids NaN/Inf).
    """
    finite = [float(v) for v in values]
    if finite and not np.isfinite(finite).all():
        raise ValueError("drift values must all be finite")
    if not finite:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "n": len(finite),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }
