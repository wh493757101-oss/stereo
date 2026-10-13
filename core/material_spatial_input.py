"""material_spatial_input_v1 — zero-training spatial pixel input contracts.

Frozen building blocks for the native-pixel cache, the unified 128x128
area-averaged grid (schema ``area_parts_letterbox128_v1``) and the
deterministic q-permutation definition (schema
``coverage_matched_parts_pcg64_v1``). Pure functions only: no model, no
scaler, no classifier or encoder forward, no optimizer.

Grid layout: the scaled crop (w' x h', never upscaled) sits at offset
(y0, x0) inside the 128x128 canvas with
    x0 = (128 - w') // 2 , remaining padding on the right,
    y0 = (128 - h') // 2 , remaining padding on the bottom,
all padding filled with positive zeros.

area(): the source rectangle is partitioned by the ACTUAL w'/h' ratios
and every grid cell is the source-overlap-area-weighted mean of its
source pixels (the rounded w'/h' are used as the real geometry; the
scale factor s is never used as an approximation of the sampling
bounds). Support and coverage come from EXACT integer overlap geometry
(``exact_cell_overlap``): T == 0 cells are positive zero everywhere and
no threshold may ever reinterpret them; the response itself may use the
float64 SAT form.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

EXPERIMENT = "material_spatial_input_v1"
GRID_SCHEMA = "area_parts_letterbox128_v1"
PERMUTATION_SCHEMA = "coverage_matched_parts_pcg64_v1"
GRID_SIZE = 128
CONTENT_SEED = 2026
PERMUTATION_DOMAIN = "material_spatial_input_v1"
REGIONS = ("interior", "band")
RESPONSE_SOURCES = ("L", "R", "q")
FROZEN_RIGHT_GAIN = 1.0
FROZEN_MIN_INTENSITY_SUM = 10.0
FROZEN_EPSILON = 1e-6

# the nine float32 planes: three sources x two region responses plus
# three shared coverage planes
GRID_PLANES = (
    "L_interior", "L_band", "R_interior", "R_band", "q_interior", "q_band",
    "coverage_interior", "coverage_band", "coverage_object",
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_PARTIAL = 2


class SpatialRefused(Exception):
    """An identity, contract or conservation refusal."""


# ---------------------------------------------------------------------------
# Geometry


def scaled_size(w: int, h: int, grid: int = GRID_SIZE) -> tuple[int, int]:
    """The letterbox geometry: never upscale, only shrink over the grid."""
    s = min(1.0, grid / max(w, h))
    w2 = max(1, int(np.floor(w * s + 0.5)))
    h2 = max(1, int(np.floor(h * s + 0.5)))
    return w2, h2


def canvas_offset(w2: int, h2: int, grid: int = GRID_SIZE) -> tuple[int, int]:
    """Top-left offset; the remainder goes to the right/bottom."""
    return (grid - w2) // 2, (grid - h2) // 2


def exact_cell_overlap(mask: np.ndarray, w: int, h: int, w2: int,
                       h2: int, offset: tuple[int, int],
                       grid: int = GRID_SIZE) -> np.ndarray:
    """T[iy, ix] = sum(mask[y, x] * lx[x] * ly[y]) — the exact integer
    overlap area of a 0/1 mask with each grid cell, in scaled units
    where source pixel x spans [x*w2, (x+1)*w2) and grid cell j spans
    [j*w, (j+1)*w). T / (w2*h2) is the overlap area in source pixels;
    T / (w*h) is the [0,1] coverage fraction. Pure int64 boundary
    arithmetic (prefix sums over the exact piecewise-linear overlaps):
    T == 0 and T > 0 are exact support decisions with no float noise, so
    no threshold may ever be applied on top of them."""
    if w2 > w or h2 > h:
        raise SpatialRefused("the letterbox geometry must never upscale")
    m = (np.asarray(mask) != 0).astype(np.int64)
    if m.shape != (h, w):
        raise SpatialRefused("mask shape does not match the crop size")

    # --- x direction, per source row: RowScaled(y, j) = sum_x m*lx ---
    # lx(j, x) = overlap of [x*w2, (x+1)*w2) with [j*w, (j+1)*w), which
    # is w2 - clipR - clipL on the intersecting pixel range and 0 outside.
    p1 = np.zeros((h, w + 1), dtype=np.int64)
    p1[:, 1:] = m.cumsum(axis=1)
    px = np.zeros((h, w + 1), dtype=np.int64)
    px[:, 1:] = (m * np.arange(w, dtype=np.int64)).cumsum(axis=1)
    j = np.arange(w2, dtype=np.int64)
    u0 = j * w
    u1 = u0 + w
    xA = u0 // w2                       # first intersecting pixel
    xB = -(-u1 // w2)                   # ceil(u1/w2), exclusive end
    xR = u1 // w2                       # first pixel with a right clip
    xL = -(-u0 // w2) - 1               # last pixel with a left clip

    def _rng(prefix: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # prefix is (h, w+1); a/b are (w2,) index arrays -> (h, w2)
        return prefix[:, b] - prefix[:, a]

    s1 = _rng(p1, xA, xB)
    sx = _rng(px, xA, xB)
    sr1 = _rng(p1, xR, xB)
    srx = _rng(px, xR, xB)
    sr = w2 * (srx + sr1) - u1 * sr1
    sl1 = _rng(p1, xA, np.minimum(xL + 1, xB))
    slx = _rng(px, xA, np.minimum(xL + 1, xB))
    sl = u0 * sl1 - w2 * slx
    row_scaled = w2 * s1 - sr - sl      # (h, w2), all int64 >= 0

    # --- y direction, per grid column: T(i, j) = sum_y ly * RowScaled ---
    q1 = np.zeros((h + 1, w2), dtype=np.int64)
    q1[1:] = row_scaled.cumsum(axis=0)
    qy = np.zeros((h + 1, w2), dtype=np.int64)
    qy[1:] = (row_scaled
              * np.arange(h, dtype=np.int64)[:, None]).cumsum(axis=0)
    i = np.arange(h2, dtype=np.int64)
    v0 = (i * h)[:, None]
    v1 = v0 + h
    yA = (v0.ravel() // h2)
    yB = -(-v1.ravel() // h2)
    yR = (v1.ravel() // h2)
    yL = (-(-v0.ravel() // h2)) - 1

    def _rng_y(prefix: np.ndarray, a: np.ndarray,
               b: np.ndarray) -> np.ndarray:
        return prefix[b] - prefix[a]

    c1 = _rng_y(q1, yA, yB)
    cy = _rng_y(qy, yA, yB)
    cr1 = _rng_y(q1, yR, yB)
    cry = _rng_y(qy, yR, yB)
    sb = h2 * (cry + cr1) - v1 * cr1
    ct1 = _rng_y(q1, yA, np.minimum(yL + 1, yB))
    cty = _rng_y(qy, yA, np.minimum(yL + 1, yB))
    st = v0 * ct1 - h2 * cty
    overlap = h2 * c1 - sb - st         # (h2, w2), all int64 >= 0
    if overlap.min() < 0:
        raise SpatialRefused("the exact overlap computation went negative")
    placed = np.zeros((grid, grid), dtype=np.int64)
    x_off, y_off = offset
    placed[y_off:y_off + h2, x_off:x_off + w2] = overlap
    return placed


def coverage_from_overlap(overlap: np.ndarray, w: int, h: int) -> np.ndarray:
    """coverage = T / (w*h) as float32; T == 0 gives positive zero."""
    cov = np.asarray(overlap).astype(np.float64) / float(w * h)
    out = cov.astype(np.float32)
    out[out == np.float32(0.0)] = np.float32(0.0)
    return out


def exact_weighted_sum(mask: np.ndarray, src: np.ndarray, w: int, h: int,
                       w2: int, h2: int) -> np.ndarray:
    """T_X[iy, ix] = sum(A[y,x] * X[y,x] * lx[x] * ly[y]) — the exact
    integer-weight coverage-weighted source sum per grid cell (float64
    only because X is float). Used by the builder for the targeted
    numeric-stability recompute of SAT responses that left the physical
    range; the verifier keeps its own independent implementation."""
    m = (np.asarray(mask) != 0).astype(np.float64)
    if m.shape != (h, w):
        raise SpatialRefused("mask shape does not match the crop size")
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


def _per_cell_supported_hull(values: np.ndarray, mask: np.ndarray,
                             w: int, h: int, w2: int, h2: int,
                             find_max: bool) -> np.ndarray:
    """Per grid cell the min (or max) of the supported source values —
    the convex-hull bounds a weighted mean must stay within. The pixels
    intersecting one cell form the exact contiguous range
    [floor(i*h/h2), ceil((i+1)*h/h2)) x [floor(j*w/w2), ceil((j+1)*w/w2)),
    so the hull is a min/max reduce over a few gathered shifts with
    per-cell bounds (all cells at once)."""
    pad = -np.inf if find_max else np.inf
    AX = np.where(mask, np.asarray(values, dtype=np.float64), pad)
    i = np.arange(h2, dtype=np.int64)
    j = np.arange(w2, dtype=np.int64)
    y_a = (i * h) // h2
    y_b = -(((i + 1) * -h) // h2)          # ceil((i+1)*h/h2)
    x_a = (j * w) // w2
    x_b = -(((j + 1) * -w) // w2)          # ceil((j+1)*w/w2)
    h_lim, w_lim = AX.shape
    reduce_fn = np.maximum if find_max else np.minimum
    hull = np.full((h2, w2), pad, dtype=np.float64)
    for dy in range(int(np.ceil(h / h2)) + 1):
        row_idx = np.clip(y_a + dy, 0, h_lim - 1)
        valid_y = ((y_a + dy) < y_b) & ((y_a + dy) < h_lim)
        for dx in range(int(np.ceil(w / w2)) + 1):
            col_idx = np.clip(x_a + dx, 0, w_lim - 1)
            gathered = AX[np.ix_(row_idx, col_idx)]
            gathered = np.where(valid_y[:, None] &
                                ((x_a + dx) < x_b)[None, :],
                                gathered, pad)
            hull = reduce_fn(hull, gathered)
    return hull


def area_average_sat(src: np.ndarray, region_mask: np.ndarray,
                     w2: int, h2: int, offset: tuple[int, int],
                     grid: int = GRID_SIZE,
                     bounds: tuple[float, float] = (float("-inf"),
                                                    float("inf"))
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The builder's area(): the response is the source-overlap-area
    weighted mean over A (float64 summed-area table with fractional
    endpoints), while the SUPPORT and the denominator come from the
    exact integer geometry (exact_cell_overlap):
        T == 0  ->  response and denominator are positive zero,
        T > 0   ->  response = N / (T / (w2*h2)).
    Non-finite values at supported positions are refused, never clamped.
    There is deliberately no "tiny value means no support" threshold.

    Numeric stability: for supported cells whose SAT response left the
    physical range (lo, hi) — the four-corner cancellation can leave
    ~1e-13 noise that turns a true zero negative — only those cells are
    recomputed exactly as T_X / T from the integer intersection
    weights, then cast to float32. A recomputed value that is still
    non-finite or out of range is refused, never clamped."""
    h, w = src.shape
    lo, hi = float(bounds[0]), float(bounds[1])
    if not lo <= hi:
        raise SpatialRefused("the physical bounds are inverted")
    mask = np.asarray(region_mask) != 0
    src_f = np.asarray(src, dtype=np.float64)
    values = np.where(mask, src_f, 0.0)
    if mask.any():
        supported_values = values[mask]
        if supported_values.size and \
                not np.all(np.isfinite(supported_values)):
            raise SpatialRefused(
                "non-finite source values at supported positions")
        src_min = float(np.min(supported_values))
        src_max = float(np.max(supported_values))
        if src_min < lo or src_max > hi:
            raise SpatialRefused(
                "supported source values outside the physical range "
                f"[{lo}, {hi}]")
    overlap = exact_cell_overlap(mask.astype(np.uint8), w, h, w2, h2,
                                 offset, grid)
    supported = overlap > 0

    def tables(f: np.ndarray):
        sat = np.zeros((h + 1, w + 1), dtype=np.float64)
        sat[1:, 1:] = f.cumsum(axis=0).cumsum(axis=1)
        rp = np.zeros((h + 1, w + 1), dtype=np.float64)
        rp[:h, 1:] = f.cumsum(axis=1)
        cp = np.zeros((h + 1, w + 1), dtype=np.float64)
        cp[1:, :w] = f.cumsum(axis=0)
        return sat, rp, cp

    sat, rp, cp = tables(values)
    corner_values = values

    def integral(sat: np.ndarray, rp: np.ndarray, cp: np.ndarray,
                 corners: np.ndarray, X: np.ndarray, Y: np.ndarray):
        # X indexes COLUMNS (j) and Y indexes ROWS (i); the returned
        # integral is indexed [y, x]
        i = np.clip(np.floor(Y).astype(np.int64), 0, h)
        j = np.clip(np.floor(X).astype(np.int64), 0, w)
        fx = X - j
        fy = Y - i
        i0 = np.minimum(i, h - 1)
        j0 = np.minimum(j, w - 1)
        return (sat[i, j] + fx * cp[i, j] + fy * rp[i, j]
                + fx * fy * corners[i0, j0])

    def rectangle(f: np.ndarray):
        s, r, c = tables(f)
        xs = np.arange(w2 + 1, dtype=np.float64) * (w / w2)
        ys = np.arange(h2 + 1, dtype=np.float64) * (h / h2)
        total = (integral(s, r, c, f, xs[None, :], ys[:, None])[1:, 1:]
                 - integral(s, r, c, f, xs[None, :], ys[:, None])[:-1, 1:]
                 - integral(s, r, c, f, xs[None, :], ys[:, None])[1:, :-1]
                 + integral(s, r, c, f, xs[None, :], ys[:, None])[:-1, :-1])
        return total

    numer_full = rectangle(values)
    x0, y0 = offset
    numer = np.zeros((grid, grid), dtype=np.float64)
    numer[y0:y0 + h2, x0:x0 + w2] = numer_full
    # the exact denominator in source-pixel units: T / (w2*h2)
    denom = np.zeros((grid, grid), dtype=np.float64)
    denom[y0:y0 + h2, x0:x0 + w2] = (overlap /
                                     float(w2 * h2))[y0:y0 + h2,
                                                     x0:x0 + w2]
    response = np.zeros_like(denom)
    np.divide(numer, denom, out=response, where=supported)
    # numeric-stability patch: a supported cell's response must lie in
    # the convex hull of that cell's own supported source values (the
    # four-corner SAT cancellation can leave ~1e-13 noise that lands
    # outside a degenerate hull, e.g. a true-zero cell). Only those
    # cells are recomputed exactly as T_X / T from the integer
    # intersection weights; a recomputed value that is still non-finite
    # or outside the hull is refused, never clamped.
    win = np.s_[y0:y0 + h2, x0:x0 + w2]
    hull_lo = _per_cell_supported_hull(src_f, mask, w, h, w2, h2,
                                       find_max=False)
    hull_hi = _per_cell_supported_hull(src_f, mask, w, h, w2, h2,
                                       find_max=True)
    resp_win = response[win]
    supported_win = supported[win]
    finite = np.isfinite(resp_win)
    unstable = supported_win & (~finite | (resp_win < hull_lo) |
                                (resp_win > hull_hi))
    if unstable.any():
        overlap_window = overlap[win]
        exact = np.zeros((h2, w2), dtype=np.float64)
        np.divide(exact_weighted_sum(mask, src_f, w, h, w2, h2),
                  overlap_window.astype(np.float64), out=exact,
                  where=overlap_window > 0)
        patched = exact[unstable]
        if not np.all(np.isfinite(patched)) or \
                np.any(patched < hull_lo[unstable] - 0.0) or \
                np.any(patched > hull_hi[unstable] + 0.0):
            raise SpatialRefused(
                "the exact recomputed response is still non-finite or "
                "outside its cell's supported-value hull")
        resp_win[unstable] = patched
        response[win] = resp_win
    # positive-zero contract wherever the value is zero
    response[response == 0.0] = 0.0
    denom[denom == 0.0] = 0.0
    return response.astype(np.float32), denom.astype(np.float32)


def validate_grid_planes(planes: dict, w2: int, h2: int,
                         offset: tuple[int, int],
                         grid: int = GRID_SIZE) -> None:
    """The strict float32 contract for the nine saved planes: dtype and
    shape, finiteness, exact positive-zero padding on every plane, exact
    positive zero for responses and coverage wherever there is no
    support (coverage bit pattern == 0), coverage in [0,1] without
    negative zero, L/R responses in [0,1] and q responses in [-1,1]."""
    if set(planes) != set(GRID_PLANES):
        raise SpatialRefused("the plane set differs from the contract")
    x_off, y_off = offset
    window = np.zeros((grid, grid), dtype=bool)
    window[y_off:y_off + h2, x_off:x_off + w2] = True
    padding = ~window
    for name, plane in planes.items():
        if plane.dtype != np.float32 or plane.shape != (grid, grid):
            raise SpatialRefused(f"plane {name} has the wrong shape/dtype")
        if not np.all(np.isfinite(plane)):
            raise SpatialRefused(f"plane {name} has non-finite values")
        bits = plane.view(np.uint32)
        if np.any(bits[padding] != 0):
            raise SpatialRefused(f"plane {name}: padding is not +0")
    for name in ("coverage_interior", "coverage_band", "coverage_object"):
        plane = planes[name]
        bits = plane.view(np.uint32)
        if np.any((plane < 0) | (plane > 1)):
            raise SpatialRefused(f"{name} outside [0,1]")
        if np.any((plane == 0) & (bits != 0)):
            raise SpatialRefused(f"{name}: negative zero where zero")
    for region in REGIONS:
        coverage = planes[f"coverage_{region}"]
        supported = coverage.view(np.uint32) != 0
        for source in RESPONSE_SOURCES:
            response = planes[f"{source}_{region}"]
            bits = response.view(np.uint32)
            if np.any(bits[~supported] != 0):
                raise SpatialRefused(
                    f"{source}_{region}: non-zero response without "
                    "support")
            low, high = (-1.0, 1.0) if source == "q" else (0.0, 1.0)
            if np.any((response < low) | (response > high)):
                raise SpatialRefused(
                    f"{source}_{region} outside [{low}, {high}]")




# ---------------------------------------------------------------------------
# Deterministic q permutation (coverage_matched_parts_pcg64_v1)


def coverage_bits_hex8(value: np.float32) -> str:
    """The float32 bit pattern of one coverage value as 8 hex chars."""
    return f"{np.float32(value).view(np.uint32):08x}"


def group_seed64(sample_name: str, region: str, bits_hex8: str,
                 content_seed: int = CONTENT_SEED) -> int:
    """seed64 = the first 8 bytes of
    SHA256('material_spatial_input_v1|sample_name|region|'
           'coverage_bits_hex8|2026')
    interpreted little-endian as an unsigned integer."""
    domain = (f"{PERMUTATION_DOMAIN}|{sample_name}|{region}|{bits_hex8}|"
              f"{content_seed}")
    return int.from_bytes(
        hashlib.sha256(domain.encode("utf-8")).digest()[:8], "little")


def plan_q_permutation(planes: dict, sample_name: str,
                       content_seed: int = CONTENT_SEED) -> dict:
    """The deterministic within-coverage-group q permutation for one
    instance's grid. Interior and band are handled separately; only
    cells with D > 0 move, grouped by the region coverage's float32 bit
    pattern; coverage and object coverage never move; single-cell groups
    keep their value; no redraws and no forced success.

    Returns the plan with per-region records and applies NOTHING — the
    caller applies/derives the permuted values from the recorded index
    permutations."""
    plan = {"sample_name": sample_name,
            "schema": PERMUTATION_SCHEMA,
            "content_seed": int(content_seed), "regions": {}}
    for region in REGIONS:
        response = np.asarray(planes[f"q_{region}"], dtype=np.float32)
        coverage = np.asarray(planes[f"coverage_{region}"],
                              dtype=np.float32)
        flat_r = response.ravel()
        flat_c = coverage.ravel()
        movable = np.flatnonzero(flat_c.view(np.uint32) != 0)
        groups: dict[str, dict] = {}
        if movable.size:
            keys = flat_c[movable].view(np.uint32)
            uniq, inverse = np.unique(keys, return_inverse=True)
            for g_index, bits in enumerate(uniq):
                members = movable[inverse == g_index]
                bits_hex8 = f"{int(bits):08x}"
                record = {
                    "coverage_bits_hex8": bits_hex8,
                    "cells": int(members.size),
                }
                if members.size > 1:
                    seed64 = group_seed64(sample_name, region, bits_hex8,
                                          content_seed)
                    perm = np.random.Generator(
                        np.random.PCG64(seed64)).permutation(members.size)
                    old_values = flat_r[members]
                    new_values = old_values[perm]
                    moved = int((perm != np.arange(members.size)).sum())
                    changed = int((new_values.view(np.uint32) !=
                                   old_values.view(np.uint32)).sum())
                    record.update({
                        "seed64": seed64,
                        "permuted": True,
                        "moved_indices": moved,
                        "value_changed_cells": changed,
                        "cell_indices": [int(v) for v in members],
                        "permutation": [int(v) for v in perm],
                    })
                else:
                    record.update({"permuted": False, "moved_indices": 0,
                                   "value_changed_cells": 0})
                groups[bits_hex8] = record
        moved = sum(g.get("moved_indices", 0) for g in groups.values())
        changed = sum(g.get("value_changed_cells", 0)
                      for g in groups.values())
        plan["regions"][region] = {
            "movable_cells": int(movable.size),
            "groups": groups,
            "moved_indices": moved,
            "value_changed_cells": changed,
        }
    return plan


def apply_q_permutation(planes: dict, plan: dict) -> dict:
    """Apply a recorded plan to a copy of the grid planes and verify the
    conservation identities. Returns the permuted planes copy."""
    permuted = {k: np.array(v, dtype=np.float32)
                for k, v in planes.items()}
    for region in REGIONS:
        response = permuted[f"q_{region}"]
        coverage = permuted[f"coverage_{region}"]
        before_pairs = np.stack([
            response.ravel().view(np.uint32),
            coverage.ravel().view(np.uint32)], axis=1)
        record = plan["regions"][region]
        for bits_hex8, group in record["groups"].items():
            if not group.get("permuted"):
                continue
            cells = np.asarray(group["cell_indices"], dtype=np.int64)
            perm = np.asarray(group["permutation"], dtype=np.int64)
            values = response.ravel()[cells]
            response.ravel()[cells] = values[perm]
        after_pairs = np.stack([
            response.ravel().view(np.uint32),
            coverage.ravel().view(np.uint32)], axis=1)
        if np.array_equal(coverage.ravel().view(np.uint32),
                          before_pairs[:, 1]) is False:
            raise SpatialRefused(f"{region}: coverage moved")
        before_sorted = before_pairs[np.lexsort(before_pairs.T[::-1])]
        after_sorted = after_pairs[np.lexsort(after_pairs.T[::-1])]
        if not np.array_equal(before_sorted, after_sorted):
            raise SpatialRefused(
                f"{region}: the (q, coverage) bit multiset changed")
    return permuted


# ---------------------------------------------------------------------------
# Native record contracts


@dataclass(frozen=True)
class NativeInstanceArrays:
    """One instance's native pixel record (all crops at the V5 crop
    window). L is the untouched V5 left gray crop; R is the registered
    right crop (float32, interpolation kept at valid positions, +0 at
    invalid); q is the production signed_q crop (+0 at invalid); M the
    V5 instance mask; V the fix valid mask (a subset of M); region 0/1/2
    = outside/interior/band at native resolution; quality the production
    4-vector."""
    sample_name: str
    frame: str
    L: np.ndarray
    R: np.ndarray
    q: np.ndarray
    M: np.ndarray
    V: np.ndarray
    region: np.ndarray
    quality: np.ndarray

    @staticmethod
    def validate_geometry(w: int, h: int, w2: int, h2: int,
                          offset: tuple[int, int]) -> None:
        """The declared letterbox geometry must be the frozen one for the
        crop size and must fit the canvas."""
        if (w2, h2) != scaled_size(w, h):
            raise SpatialRefused(
                f"letterbox size ({w2}, {h2}) != the frozen geometry "
                f"{scaled_size(w, h)} for crop {w}x{h}")
        if tuple(offset) != canvas_offset(w2, h2):
            raise SpatialRefused(
                f"letterbox offset {tuple(offset)} != the frozen "
                f"{canvas_offset(w2, h2)}")

    def validate(self) -> None:
        h, w = self.L.shape
        for name, array in (("R", self.R), ("q", self.q),
                            ("M", self.M), ("V", self.V),
                            ("region", self.region)):
            if array.shape != (h, w):
                raise SpatialRefused(
                    f"{self.sample_name}: {name} shape {array.shape} != "
                    f"L shape {(h, w)}")
        if self.L.dtype != np.uint8 or self.M.dtype != np.uint8 or \
                self.V.dtype != np.uint8 or self.region.dtype != np.uint8:
            raise SpatialRefused(f"{self.sample_name}: L/M/V/region must "
                                 "be uint8")
        if self.R.dtype != np.float32 or self.q.dtype != np.float32:
            raise SpatialRefused(f"{self.sample_name}: R/q must be float32")
        if self.quality.shape != (4,) or self.quality.dtype != np.float32:
            raise SpatialRefused(f"{self.sample_name}: quality must be "
                                 "float32(4)")
        if not np.all((self.region == 0) | (self.region == 1)
                      | (self.region == 2)):
            raise SpatialRefused(f"{self.sample_name}: region values "
                                 "outside {0,1,2}")
        valid = self.V > 0
        if not np.all(valid <= (self.M > 0)):
            raise SpatialRefused(f"{self.sample_name}: V is not a subset "
                                 "of M")
        if not np.all(self.q[~valid] == np.float32(0.0)):
            raise SpatialRefused(f"{self.sample_name}: q is not +0 at "
                                 "invalid positions")
        if not np.all(self.R[~valid] == np.float32(0.0)):
            raise SpatialRefused(f"{self.sample_name}: R is not +0 at "
                                 "invalid positions")
        if not np.all(np.isfinite(self.R)) or not np.all(np.isfinite(self.q)):
            raise SpatialRefused(f"{self.sample_name}: non-finite R/q")
