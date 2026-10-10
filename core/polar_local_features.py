"""Local polarization statistics: interior/band partition (scheme A).

The frozen measurement region (``instance_mask & polar_valid``) is
PARTITIONED — never filtered — into the frozen interior
(``erode3x3_instance_mask_v1``) and contour band
(``instance_mask_minus_interior_v1``). Each part carries the EXISTING
eight S statistics, plus explicit per-region C control information
(coverage, empty flag, gray mean/std). The scheme pre-registered in
``docs/experiments/material_polar_local_interior_band_v1.md``:

- no region-reliability rule changes: the union of the two parts is
  exactly the original region, so ``s_valid`` is identical to the global
  cache by construction (pinned by tests);
- an empty part is an explicit NaN miss in its S block plus an explicit
  empty flag in C — never a fabricated zero measurement;
- G, the per-fold P0 heads/scalers, the fold membership, the training
  budget, the correction magnitude and the loss are unchanged; only the
  residual input width grows (14 -> 30) with the same zero-initialized
  ``delta = valid_s * 0.5 * tanh(W x)`` form.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.material_mask_dataset import load_mask_sample
from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    Standardizer,
    build_cache_binding,
    build_control_features,
    authoritative_sample_index,
    verify_cache_binding,
    verify_cache_manifest,
)
from core.polar_region_quality import (
    BAND_DEFINITION,
    INTERIOR_DEFINITION,
    interior_and_band_masks,
    region_polar_stats,
)

LOCAL_REGION_RULE = "local_partition_interior_band_v1"
LOCAL_FEATURE_DEFINITION_VERSION = (
    "existing_data_probe_v1_local_interior_band_v1")

REGION_CONTROL_KEYS = (
    "interior_coverage",
    "interior_empty_flag",
    "interior_gray_mean",
    "interior_gray_std",
    "band_coverage",
    "band_empty_flag",
    "band_gray_mean",
    "band_gray_std",
)
C_LOCAL_KEYS = (*CONTROL_FEATURE_KEYS, *REGION_CONTROL_KEYS)
INTERIOR_BLOCK = (0, len(POLAR_STAT_KEYS))
BAND_BLOCK = (len(POLAR_STAT_KEYS), 2 * len(POLAR_STAT_KEYS))
LOCAL_C_WIDTH = len(C_LOCAL_KEYS)
LOCAL_S_WIDTH = 2 * len(POLAR_STAT_KEYS)
LOCAL_TOTAL_DIM = LOCAL_C_WIDTH + LOCAL_S_WIDTH

LOCAL_RULE_NOTE = (
    "the ORIGINAL region instance_mask & polar_valid is partitioned into "
    "interior (erode3x3) and band (mask minus interior); each part holds "
    "the existing 8 S statistics on part & polar_valid; an empty part is "
    "an explicit NaN block plus an empty flag in C. No pixel is excluded "
    "and s_valid is identical to the global definition by construction.")


def local_region_masks(sample):
    """The two PART masks (interior/band) of a sample's measurement region."""
    interior, band = interior_and_band_masks(sample.instance_mask)
    valid = np.asarray(sample.valid, dtype=bool)
    return interior & valid, band & valid, interior, band


def build_local_polar_stats(sample):
    """The 16-wide S slot: the frozen 8 statistics per part.

    Returns ``(values, blocks)`` where ``blocks`` records per-part
    validity. An empty part yields an all-NaN block (the explicit miss).
    """
    interior_valid, band_valid, _, _ = local_region_masks(sample)
    interior_values, interior_has = region_polar_stats(
        sample.signed_q, sample.abs_q, interior_valid)
    band_values, band_has = region_polar_stats(
        sample.signed_q, sample.abs_q, band_valid)
    values = np.concatenate([interior_values, band_values]).astype(
        np.float32)
    return values, {"interior_valid": bool(interior_has),
                    "band_valid": bool(band_has)}


def _region_control(region: np.ndarray, region_valid: np.ndarray,
                    gray: np.ndarray) -> tuple[float, float, float, float]:
    region_pixels = int(region.sum())
    valid_pixels = int(region_valid.sum())
    coverage = float(valid_pixels / region_pixels) if region_pixels else 0.0
    empty = 1.0 if valid_pixels == 0 else 0.0
    if valid_pixels:
        gray_values = gray[region_valid].astype(np.float64) / 255.0
        gray_mean = float(gray_values.mean())
        gray_std = float(gray_values.std(ddof=0))
    else:
        # explicit control convention (flagged by empty_flag), never a
        # fabricated measurement of a region that has no valid pixels
        gray_mean = 0.0
        gray_std = 0.0
    return coverage, empty, gray_mean, gray_std


def build_local_control_features(sample) -> np.ndarray:
    """The 14-wide C slot: the original six controls plus the per-part
    coverage / empty flag / gray mean+std (gray over part & polar_valid,
    scaled to [0, 1])."""
    base = build_control_features(
        quality=sample.quality, gray=sample.gray,
        instance_mask=sample.instance_mask,
        s_region_nonempty=bool(
            (np.asarray(sample.valid, dtype=bool)
             & (np.asarray(sample.instance_mask) > 0)).any()))
    interior, band = interior_and_band_masks(sample.instance_mask)
    valid = np.asarray(sample.valid, dtype=bool)
    gray = np.asarray(sample.gray)
    interior_ctrl = _region_control(interior, interior & valid, gray)
    band_ctrl = _region_control(band, band & valid, gray)
    return np.asarray([*base, *interior_ctrl, *band_ctrl], dtype=np.float32)


def region_records(sample) -> dict:
    """Per-sample region bookkeeping for the cache manifest."""
    interior_valid, band_valid, interior, band = local_region_masks(sample)
    mask = np.asarray(sample.instance_mask) > 0
    return {
        "region_rule": LOCAL_REGION_RULE,
        "mask_pixels": int(mask.sum()),
        "interior_pixels": int(interior.sum()),
        "band_pixels": int(band.sum()),
        "interior_valid_pixels": int(interior_valid.sum()),
        "band_valid_pixels": int(band_valid.sum()),
        "region_pixels": int((mask & np.asarray(sample.valid, bool)).sum()),
    }


def fit_local_scaler(cache: dict, inner_rows: list):
    """The local standardizer, fitted on inner_train rows only.

    C columns use every training row; each S cell uses the rows where
    that cell is finite (per-part validity: an empty part is NaN and
    excluded). Zero-variance or fully-invalid columns transform to
    exactly zero (Standardizer's deterministic rule).
    """
    c = np.asarray(cache["c_features"], dtype=np.float64)[list(inner_rows)]
    s = np.asarray(cache["s_features"], dtype=np.float64)[list(inner_rows)]
    if not np.isfinite(c).all():
        raise ValueError("C features contain NaN/Inf")
    if not ((np.isnan(s)) | (np.isfinite(s))).all():
        raise ValueError("S features contain non-finite non-NaN values")
    width = LOCAL_C_WIDTH + LOCAL_S_WIDTH
    matrix = np.concatenate([c, s], axis=1)
    column_valid = np.ones(matrix.shape, dtype=bool)
    column_valid[:, LOCAL_C_WIDTH:] = ~np.isnan(s)
    scaler = Standardizer.fit(matrix, column_valid)
    info = {
        "inner_train_rows": len(list(inner_rows)),
        "width": width,
        "rule": ("inner_train only; C all rows; S cells only where the "
                 "part has valid pixels (NaN cells excluded)"),
    }
    return scaler, info


def assemble_local_residual_inputs(c_features: np.ndarray,
                                   s_features: np.ndarray,
                                   s_valid: np.ndarray,
                                   variant: str) -> np.ndarray:
    """The 30-wide residual input from the standardized local slots.

    ``RC`` zeroes both polarization blocks unconditionally (it never
    reads polarization content). Rows with ``s_valid=False`` get an
    exactly-zero S part under BOTH variants (the explicit missing rule).
    NaN must not appear in valid rows: the standardizer has already
    turned empty-part cells into exact zeros.
    """
    from core.material_residual import validate_variant

    variant = validate_variant(variant)
    c = np.asarray(c_features, dtype=np.float32)
    s = np.asarray(s_features, dtype=np.float32)
    valid = np.asarray(s_valid).astype(bool)
    if c.ndim != 2 or c.shape[1] != LOCAL_C_WIDTH:
        raise ValueError(
            f"C features must be (n, {LOCAL_C_WIDTH}), got {c.shape}")
    if s.ndim != 2 or s.shape[1] != LOCAL_S_WIDTH:
        raise ValueError(
            f"S features must be (n, {LOCAL_S_WIDTH}), got {s.shape}")
    if len(c) != len(s) or len(valid) != len(c):
        raise ValueError("C/S/s_valid row counts disagree")
    if not np.isfinite(c).all():
        raise ValueError("C features contain NaN/Inf")
    if valid.any() and not np.isfinite(s[valid]).all():
        raise ValueError(
            "S features contain NaN/Inf in rows marked valid; the "
            "standardizer must zero empty-part cells before assembly")
    s_input = np.zeros_like(s, dtype=np.float32)
    s_input[valid] = s[valid]
    if variant == "RC":
        s_input = np.zeros_like(s_input)
    return np.concatenate([c, s_input], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Local feature cache


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def verify_local_cache(feature_dir, protocol: dict) -> dict:
    """The local cache admission: binding bytes, manifest row identity and
    the local array contract (widths, per-part NaN rules, C flag
    consistency, class order)."""
    feature_dir = Path(feature_dir)
    verify_cache_binding(feature_dir, protocol)
    definition = _load_json(feature_dir / "feature_definition.json")
    if definition.get("feature_definition_version") != \
            LOCAL_FEATURE_DEFINITION_VERSION:
        raise ValueError(
            "local feature definition version mismatch: "
            f"{definition.get('feature_definition_version')!r}")
    manifest = _load_json(feature_dir / "feature_manifest.json")
    authority = authoritative_sample_index(protocol)
    samples = verify_cache_manifest(manifest, authority, protocol)
    cache = dict(np.load(feature_dir / "feature_cache.npz"))
    expected_keys = {"g_features", "c_features", "s_features", "s_valid",
                     "class_id"}
    if set(cache) != expected_keys:
        raise ValueError(
            f"local cache npz keys {sorted(cache)} != {sorted(expected_keys)}")
    n = len(samples)
    g = cache["g_features"]
    c = cache["c_features"]
    s = cache["s_features"]
    s_valid = cache["s_valid"]
    class_ids = cache["class_id"]
    g_dim = int(definition.get("slot_G", {}).get("dim", g.shape[1]))
    if g.dtype != np.float32 or g.shape != (n, g_dim):
        raise ValueError(f"g_features shape/dtype {g.shape}/{g.dtype} "
                         f"!= ({n}, {g_dim}) float32")
    if c.dtype != np.float32 or c.shape != (n, LOCAL_C_WIDTH):
        raise ValueError(f"c_features shape {c.shape} != "
                         f"({n}, {LOCAL_C_WIDTH})")
    if s.dtype != np.float32 or s.shape != (n, LOCAL_S_WIDTH):
        raise ValueError(f"s_features shape {s.shape} != "
                         f"({n}, {LOCAL_S_WIDTH})")
    if s_valid.dtype != np.bool_ or s_valid.shape != (n,):
        raise ValueError("s_valid must be a strict boolean (n,) array")
    if not np.isfinite(g).all():
        raise ValueError("g_features contains NaN/Inf")
    if not np.isfinite(c).all():
        raise ValueError("c_features contains NaN/Inf")
    def _block_state(block: np.ndarray, name: str):
        """(all_nan, all_finite) per row; Inf and partial-NaN blocks are
        refused — a part is either entirely finite or entirely NaN."""
        all_nan = np.isnan(block).all(axis=1)
        all_finite = np.isfinite(block).all(axis=1)
        if not (all_nan | all_finite).all():
            bad = int((~(all_nan | all_finite)).sum())
            raise ValueError(
                f"the {name} S block must be entirely finite or entirely "
                f"NaN per row; {bad} row(s) carry Inf or partial-NaN "
                "values")
        return all_nan, all_finite

    invalid = ~s_valid
    interior_all_nan, interior_all_finite = _block_state(
        s[:, INTERIOR_BLOCK[0]:INTERIOR_BLOCK[1]], "interior")
    band_all_nan, band_all_finite = _block_state(
        s[:, BAND_BLOCK[0]:BAND_BLOCK[1]], "band")
    # rows without a valid S: BOTH parts must be all-NaN
    if invalid.any() and not (interior_all_nan[invalid].all()
                              and band_all_nan[invalid].all()):
        raise ValueError(
            "rows without a valid S must be all-NaN across BOTH parts "
            "(the explicit-miss rule)")
    # rows marked valid: at least one ENTIRELY FINITE part; a valid row
    # with both parts NaN is refused
    if s_valid.any() and not (interior_all_finite[s_valid]
                              | band_all_finite[s_valid]).all():
        raise ValueError(
            "rows marked valid must have at least one entirely finite "
            "part; a valid row with both parts all-NaN is refused")
    # C flag consistency over ALL rows (empty parts and the sample-level
    # missing state are explicit controls, never silent zeros)
    flag_col = c[:, C_LOCAL_KEYS.index("s_region_nonempty_flag")]
    if not np.array_equal(flag_col > 0.5, s_valid):
        raise ValueError(
            "c_features s_region_nonempty_flag does not match s_valid")
    interior_flag = c[:, C_LOCAL_KEYS.index("interior_empty_flag")] > 0.5
    band_flag = c[:, C_LOCAL_KEYS.index("band_empty_flag")] > 0.5
    if not np.array_equal(interior_flag, interior_all_nan):
        raise ValueError(
            "interior_empty_flag does not match the interior NaN block")
    if not np.array_equal(band_flag, band_all_nan):
        raise ValueError(
            "band_empty_flag does not match the band NaN block")
    interior_coverage = c[:, C_LOCAL_KEYS.index("interior_coverage")]
    if not np.array_equal(interior_coverage <= 0.0, interior_all_nan):
        raise ValueError(
            "interior_coverage must be zero exactly when the interior "
            "block is all-NaN")
    band_coverage = c[:, C_LOCAL_KEYS.index("band_coverage")]
    if not np.array_equal(band_coverage <= 0.0, band_all_nan):
        raise ValueError(
            "band_coverage must be zero exactly when the band block is "
            "all-NaN")
    manifest_ids = np.asarray([row["class_id"] for row in samples],
                              dtype=np.int64)
    if not np.array_equal(class_ids.astype(np.int64), manifest_ids):
        raise ValueError("cache class_id rows do not match the manifest")
    return {"sample_count": n, "definition_version":
            definition["feature_definition_version"]}


def build_local_cache_arrays(*, source_arrays: dict, samples: list,
                             data_root, protocol: dict, output_dir) -> dict:
    """Build the local interior/band feature cache.

    Row order, ``g_features``, ``s_valid`` and ``class_id`` are carried
    over from the verified global cache unchanged (G stays
    byte-identical); only ``c_features``/``s_features`` are recomputed
    from the V5 samples. The output directory must not exist and the
    source cache is never written.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(
            f"output directory {output_dir} already exists; local feature "
            "caches are written to fresh directories only")
    n = len(samples)
    g_source = np.asarray(source_arrays["g_features"])
    if g_source.shape[0] != n:
        raise ValueError("source G row count does not match the manifest")
    s_local = np.full((n, LOCAL_S_WIDTH), np.nan, dtype=np.float32)
    c_local = np.zeros((n, LOCAL_C_WIDTH), dtype=np.float32)
    records = []
    data_root = Path(data_root)
    for row, manifest_row in enumerate(samples):
        sample = load_mask_sample(data_root / manifest_row["npz_path"])
        if sample.class_id != int(manifest_row["class_id"]):
            raise ValueError(
                f"{manifest_row['npz_path']}: class_id {sample.class_id} "
                f"!= manifest {manifest_row['class_id']}")
        values, _blocks = build_local_polar_stats(sample)
        c_local[row] = build_local_control_features(sample)
        s_local[row] = values
        records.append(region_records(sample))
    s_valid = np.asarray(source_arrays["s_valid"]).copy()
    computed_valid = ~np.isnan(s_local).all(axis=1)
    if not np.array_equal(s_valid, computed_valid):
        raise ValueError(
            "the local partition changed the sample-level missing state; "
            "refusing an inconsistent cache")
    class_id = np.asarray(source_arrays["class_id"]).copy()
    if not np.array_equal(
            class_id,
            np.asarray([row["class_id"] for row in samples], dtype=np.int64)):
        raise ValueError("source class_id rows do not match the manifest")

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "feature_cache.npz",
        g_features=g_source,
        c_features=c_local,
        s_features=s_local,
        s_valid=s_valid,
        class_id=class_id)
    manifest = {
        "count": n,
        "samples": [{**dict(row), **record}
                    for row, record in zip(samples, records)],
    }
    _write_json(output_dir / "feature_manifest.json", manifest)
    definition = {
        "feature_definition_version": LOCAL_FEATURE_DEFINITION_VERSION,
        "parent_feature_definition_version": "existing_data_probe_v1",
        "data_fingerprint": protocol["data"]["fingerprint"],
        "protocol_sha256": protocol["protocol_sha256"],
        "official_mit_b0": dict(
            protocol["config"]["baselines"]["official_mit_b0"]),
        "class_names": list(protocol["data"]["class_names"]),
        "region": {
            "rule": LOCAL_REGION_RULE,
            "rule_note": LOCAL_RULE_NOTE,
            "interior_definition": INTERIOR_DEFINITION,
            "band_definition": BAND_DEFINITION,
            "partition_of": "instance_mask & polar_valid",
        },
        "slot_G": {"dim": int(g_source.shape[1]),
                   "definition": "copied byte-identical from the frozen "
                                 "global cache"},
        "slot_C": {"keys": list(C_LOCAL_KEYS), "width": LOCAL_C_WIDTH},
        "slot_S": {"keys": list(POLAR_STAT_KEYS), "width": LOCAL_S_WIDTH,
                   "blocks": {"interior": list(INTERIOR_BLOCK),
                              "band": list(BAND_BLOCK)},
                   "quantile_method": "numpy.percentile (linear "
                                      "interpolation)",
                   "missing_rule": (
                       "an empty part is an all-NaN block plus an "
                       "explicit empty flag in C; a sample without any "
                       "valid S is all-NaN (has_valid_s=false, identical "
                       "to the global definition)")},
    }
    _write_json(output_dir / "feature_definition.json", definition)
    build_cache_binding(output_dir, protocol)
    report = verify_local_cache(output_dir, protocol)
    return {
        "sample_count": report["sample_count"],
        "definition_version": LOCAL_FEATURE_DEFINITION_VERSION,
        "region_rule": LOCAL_REGION_RULE,
        "input_dim": LOCAL_TOTAL_DIM,
        "output_dir": str(output_dir),
    }
