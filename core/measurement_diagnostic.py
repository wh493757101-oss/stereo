"""Diagnostic-only library for material_measurement_diagnostic_v1.

This module NEVER changes production behaviour. It exists so the batch-1
measurement diagnostics can:

1. re-run the production SGBM core with a swappable final upsampling step
   (the production path vs. a valid-weight-normalized control) while the
   low-resolution core is computed exactly once per comparison;
2. build synthetic counterexamples with independent analytic ground truth
   for the production upsample chain (zero-filled invalid low-resolution
   disparity -> INTER_LINEAR upsample, INTER_NEAREST valid mask, keep at
   valid positions);
3. compute the residual-capacity statistics from saved P0 logits with
   strict sample-identity checks;
4. join the runtime pipeline evidence by (frame_id, matched_gt_id) with
   the saved match relations verified, never by zip or row counts.

The copied SGBM core (``run_sgbm_lowres``) is byte-for-byte the production
``StereoMatcher._match_pair`` body up to (but excluding) the final
upsampling. It is guarded by an equivalence test that fails if production
and the copy ever diverge; the diagnostic entry additionally re-checks the
equivalence on every real frame it processes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.stereo_matching import StereoMatcherConfig

PRODUCTION_UPSAMPLE = "production"
WEIGHTED_UPSAMPLE = "valid_weighted"
UPSAMPLE_METHODS = (PRODUCTION_UPSAMPLE, WEIGHTED_UPSAMPLE)

# Reuse the production feature computation verbatim for every q comparison.
from core.polar_compute import compute_polar_features  # noqa: E402,F401

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish", "real_fish")


# ---------------------------------------------------------------------------
# The copied SGBM core and the swappable final upsample


@dataclass(frozen=True)
class LowResStereoCore:
    """The low-resolution intermediates the production matcher computes
    internally, exposed so both upsampling variants can share one SGBM run."""

    disp_small: np.ndarray  # float32, invalid entries are exactly 0.0
    valid_small: np.ndarray  # uint8 0/1
    valid_ratio: float
    elapsed_s: float
    low_res_shape: tuple[int, int]


def run_sgbm_lowres(
    left_gray: np.ndarray, right_gray: np.ndarray, cfg: StereoMatcherConfig
) -> LowResStereoCore:
    """The production ``StereoMatcher._match_pair`` core, up to the final
    upsample. Verbatim copy (see module docstring); guarded by tests."""

    h, w = left_gray.shape
    sw = max(1, int(round(w * cfg.scale)))
    sh = max(1, int(round(h * cfg.scale)))
    left_small = cv2.resize(left_gray, (sw, sh), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right_gray, (sw, sh), interpolation=cv2.INTER_AREA)

    num_disp = max(16, int(np.ceil((cfg.max_disparity * cfg.scale) / 16.0)) * 16)
    if min(sw, sh) < cfg.block_size:
        return LowResStereoCore(
            disp_small=np.zeros((sh, sw), np.float32),
            valid_small=np.zeros((sh, sw), np.uint8),
            valid_ratio=0.0,
            elapsed_s=0.0,
            low_res_shape=(sh, sw),
        )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    left_match = clahe.apply(left_small)
    right_match = clahe.apply(right_small)

    p1 = 8 * cfg.block_size * cfg.block_size
    p2 = 32 * cfg.block_size * cfg.block_size
    common = dict(
        blockSize=cfg.block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=-1,
        preFilterCap=63,
        uniquenessRatio=cfg.uniqueness_ratio,
        speckleWindowSize=cfg.speckle_window,
        speckleRange=cfg.speckle_range,
    )
    modes = {
        "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
        "hh": cv2.STEREO_SGBM_MODE_HH,
        "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }
    sgbm_mode = modes[cfg.mode]
    sgbm_left = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, mode=sgbm_mode, **common
    )
    sgbm_right = cv2.StereoSGBM_create(
        minDisparity=-num_disp, numDisparities=num_disp, mode=sgbm_mode, **common
    )

    pad = num_disp
    left_pad = cv2.copyMakeBorder(left_match, 0, 0, pad, pad, cv2.BORDER_REPLICATE)
    right_pad = cv2.copyMakeBorder(right_match, 0, 0, pad, pad, cv2.BORDER_REPLICATE)

    started = time.perf_counter()
    disp_l = (
        sgbm_left.compute(left_pad, right_pad).astype(np.float32)[:, pad:-pad] / 16.0
    )
    disp_r_neg = (
        sgbm_right.compute(right_pad, left_pad).astype(np.float32)[:, pad:-pad] / 16.0
    )
    disp_r = -disp_r_neg
    elapsed = time.perf_counter() - started

    tol = cfg.lr_check_threshold_px * cfg.scale
    xs = np.arange(sw, dtype=np.float32)
    sample_x = np.rint(xs[None, :] - disp_l).astype(np.int32)
    in_range = (sample_x >= 0) & (sample_x < sw)
    sx = np.clip(sample_x, 0, sw - 1)
    right_d = disp_r[np.arange(sh)[:, None], sx]
    consistent = (disp_l > 0) & in_range & (np.abs(right_d - disp_l) <= tol + 1e-6)

    if cfg.texture_threshold > 0:
        dx = cv2.Sobel(left_match, cv2.CV_32F, 1, 0, ksize=3)
        texture = cv2.boxFilter(
            np.abs(dx), -1, (cfg.block_size, cfg.block_size), normalize=False
        )
        consistent &= texture > cfg.texture_threshold

    valid_small = consistent
    disp_small = np.where(valid_small, disp_l, 0.0).astype(np.float32)
    valid_ratio = float(valid_small.mean())
    return LowResStereoCore(
        disp_small=disp_small,
        valid_small=valid_small.astype(np.uint8),
        valid_ratio=valid_ratio,
        elapsed_s=elapsed,
        low_res_shape=(sh, sw),
    )


def upsample_lowres(
    core: LowResStereoCore,
    out_w: int,
    out_h: int,
    scale: float,
    method: str = PRODUCTION_UPSAMPLE,
    w_min: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one final upsampling variant to the shared low-res core.

    ``production`` reproduces the production final step exactly:
    the zero-filled low-res disparity is INTER_LINEAR-upsampled, the valid
    mask is INTER_NEAREST-upsampled, and the disparity is kept only where
    the upsampled mask is valid.

    ``valid_weighted`` is the diagnostic control: the weighted sum of the
    VALID low-res disparities and the summed valid weight are interpolated
    separately (a plain bilinear resize of ``disp_small`` is exactly the
    weighted sum because invalid entries are zero), then divided where the
    accumulated valid weight reaches ``w_min``. Positions below ``w_min``
    are marked unmeasurable. This control is NOT a default fix and does
    not repair valid-valid depth discontinuities.
    """
    if method not in UPSAMPLE_METHODS:
        raise ValueError(f"unknown upsample method: {method!r}")
    if not 0.0 < w_min <= 1.0:
        raise ValueError(f"w_min must be in (0, 1], got {w_min!r}")
    sh, sw = core.low_res_shape
    if method == PRODUCTION_UPSAMPLE:
        disp_full = (
            cv2.resize(core.disp_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            / scale
        )
        valid_full = cv2.resize(
            core.valid_small, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )
        disp_full = np.where(valid_full > 0, disp_full, 0.0).astype(np.float32)
        return disp_full, valid_full

    weights = cv2.resize(
        core.valid_small.astype(np.float32), (out_w, out_h),
        interpolation=cv2.INTER_LINEAR,
    )
    weighted_sum = cv2.resize(
        core.disp_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR
    )
    valid_full = (weights >= np.float32(w_min)).astype(np.uint8)
    denom = np.maximum(weights, np.float32(1e-12))
    disp_full = np.where(
        valid_full > 0, weighted_sum / denom / scale, 0.0
    ).astype(np.float32)
    return disp_full, valid_full


class DiagnosticUpsampleMatcher:
    """A matcher exposing the production ``compute``/``compute_bands``
    interface with the final upsample swapped for one diagnostic variant.

    It subclasses :class:`core.stereo_matching.StereoMatcher` and overrides
    only ``_match_pair``; band handling, ``min_valid_ratio`` policy and
    ``instance_stats`` are inherited unchanged from production.
    """

    def __init__(
        self,
        config: StereoMatcherConfig,
        *,
        method: str = PRODUCTION_UPSAMPLE,
        w_min: float = 0.5,
    ) -> None:
        from core.stereo_matching import StereoMatcher

        if method not in UPSAMPLE_METHODS:
            raise ValueError(f"unknown upsample method: {method!r}")
        self._inner = StereoMatcher(config)
        self.config = self._inner.config
        self.method = method
        self.w_min = float(w_min)

    @property
    def num_disparities(self) -> int:
        return self._inner.num_disparities

    def compute(self, left: np.ndarray, right: np.ndarray):
        return self._compute_like_production(left, right, bands=None)

    def compute_bands(self, left: np.ndarray, right: np.ndarray, bands, full_image_threshold: float = 0.9):
        return self._compute_like_production(left, right, bands=bands,
                                             full_image_threshold=full_image_threshold)

    def instance_stats(self, result, mask, instance_id: int = 0):
        return self._inner.instance_stats(result, mask, instance_id=instance_id)

    def _match_pair_like(self, left_gray: np.ndarray, right_gray: np.ndarray):
        h, w = left_gray.shape
        core = run_sgbm_lowres(left_gray, right_gray, self.config)
        disp_full, valid_full = upsample_lowres(
            core, w, h, self.config.scale, method=self.method, w_min=self.w_min
        )
        return disp_full, valid_full, core.elapsed_s, core.valid_ratio

    def _compute_like_production(self, left, right, bands=None, full_image_threshold: float = 0.9):
        """The production ``compute``/``compute_bands`` bodies, re-expressed
        on top of ``_match_pair_like``. Verbatim semantics: same
        normalization, same band clipping/merging, same min_valid_ratio
        policy, same full-image fallback."""
        from core.stereo_matching import (
            HorizontalBand,
            StereoMatchResult,
            merge_horizontal_bands,
            to_gray_u8,
        )

        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        if left_gray.shape != right_gray.shape:
            raise ValueError(
                f"left/right shape mismatch: {left_gray.shape} vs {right_gray.shape}"
            )
        if left_gray.ndim != 2:
            raise ValueError(f"expected single-channel image, got shape {left_gray.shape}")
        h, w = left_gray.shape
        if bands is None:
            disp, valid, elapsed, valid_ratio = self._match_pair_like(left_gray, right_gray)
            if valid_ratio < self.config.min_valid_ratio:
                disp = np.zeros((h, w), dtype=np.float32)
                valid = np.zeros((h, w), dtype=np.uint8)
            return StereoMatchResult(disparity=disp, valid=valid, elapsed_s=elapsed,
                                     valid_ratio=valid_ratio)

        if not 0.0 < full_image_threshold <= 1.0:
            raise ValueError("full_image_threshold must be in (0, 1]")
        clipped_bands = []
        for band in bands:
            y1 = max(0, int(band.y1))
            y2 = min(h, int(band.y2))
            if y2 > y1:
                clipped_bands.append(HorizontalBand(y1, y2))
        clipped = merge_horizontal_bands(clipped_bands)
        if not clipped:
            return StereoMatchResult(
                disparity=np.zeros((h, w), dtype=np.float32),
                valid=np.zeros((h, w), dtype=np.uint8),
                elapsed_s=0.0, valid_ratio=0.0)
        coverage = sum(band.height for band in clipped)
        if coverage >= full_image_threshold * h:
            return self.compute(left_gray, right_gray)
        disp = np.zeros((h, w), dtype=np.float32)
        valid = np.zeros((h, w), dtype=np.uint8)
        elapsed = 0.0
        valid_pixels = 0
        for band in clipped:
            band_disp, band_valid, band_elapsed, band_ratio = self._match_pair_like(
                left_gray[band.y1:band.y2], right_gray[band.y1:band.y2])
            elapsed += band_elapsed
            if band_ratio < self.config.min_valid_ratio:
                continue
            disp[band.y1:band.y2] = band_disp
            valid[band.y1:band.y2] = band_valid
            valid_pixels += int(band_valid.sum())
        return StereoMatchResult(disparity=disp, valid=valid, elapsed_s=elapsed,
                                 valid_ratio=valid_pixels / float(h * w))


# ---------------------------------------------------------------------------
# Right-image shift interventions


def shift_right_image(
    right_gray: np.ndarray, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray]:
    """Shift the right image by (dx, dy) with ZERO fill (never wrap).

    Returns ``(shifted, fill_mask)`` where ``fill_mask`` is True exactly on
    the newly introduced border pixels. (dx, dy) must be a single-axis
    shift with |dx| + |dy| == 1, matching the frozen intervention set.
    """
    if (abs(dx) + abs(dy)) != 1:
        raise ValueError(f"expected a single-axis +-1px shift, got dx={dx}, dy={dy}")
    h, w = right_gray.shape[:2]
    shifted = np.zeros_like(right_gray)
    fill = np.zeros((h, w), dtype=bool)
    if dx == 1:
        shifted[:, 1:] = right_gray[:, :-1]
        fill[:, 0] = True
    elif dx == -1:
        shifted[:, :-1] = right_gray[:, 1:]
        fill[:, w - 1] = True
    elif dy == 1:
        shifted[1:, :] = right_gray[:-1, :]
        fill[0, :] = True
    else:
        shifted[:-1, :] = right_gray[1:, :]
        fill[h - 1, :] = True
    return shifted, fill


def edge_unmeasurable_mask(
    shape: tuple[int, int],
    disparity: np.ndarray,
    dx: int,
    dy: int,
) -> np.ndarray:
    """Pixels whose RIGHT-IMAGE SAMPLE lands on (or touches) a filled edge.

    The warp samples the right image at column ``x - d`` (bilinearly), row
    ``y``. A filled border column of the shifted right image contributes
    fabricated content, so any measurement sampling it is marked
    unmeasurable: for dx=+1 the bilinear kernel touches fill column 0 when
    ``x - d < 1``; for dx=-1 when ``x - d > w - 2``. Vertical shifts sample
    integer rows, so dy=+1 forbids row 0 and dy=-1 row h-1.
    """
    h, w = shape
    disp = np.asarray(disparity, dtype=np.float32)
    sx = np.arange(w, dtype=np.float32)[None, :] - disp
    mask = np.zeros((h, w), dtype=bool)
    if dx == 1:
        mask |= (sx >= 0.0) & (sx < 1.0)
    elif dx == -1:
        mask |= sx > float(w - 2)
    if dy == 1:
        mask |= np.arange(h)[:, None] == 0
    elif dy == -1:
        mask |= np.arange(h)[:, None] == (h - 1)
    return mask


# ---------------------------------------------------------------------------
# Comparison statistics


def _finite_percentiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max_abs": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def compare_measurements(
    baseline_disp: np.ndarray,
    baseline_valid: np.ndarray,
    baseline_q: np.ndarray,
    baseline_qvalid: np.ndarray,
    arm_disp: np.ndarray,
    arm_valid: np.ndarray,
    arm_q: np.ndarray,
    arm_qvalid: np.ndarray,
) -> dict[str, Any]:
    """Path-difference statistics between two measurement arms.

    Reports only path differences: valid-pixel counts and flips, and the
    signed_q / abs_q / disparity differences over the COMMON valid region.
    No arm is ever called more accurate; real images have no disparity
    ground truth here.
    """
    base_v = np.asarray(baseline_valid) > 0
    arm_v = np.asarray(arm_valid) > 0
    common = base_v & arm_v
    result: dict[str, Any] = {
        "valid_pixels": {"baseline": int(base_v.sum()), "arm": int(arm_v.sum()),
                         "common": int(common.sum())},
        "validity_flips": {
            "baseline_valid_arm_invalid": int((base_v & ~arm_v).sum()),
            "baseline_invalid_arm_valid": int((~base_v & arm_v).sum()),
        },
        "disparity_diff_common_valid": _finite_percentiles(
            (np.asarray(arm_disp, dtype=np.float64)
             - np.asarray(baseline_disp, dtype=np.float64))[common]),
    }
    base_qv = np.asarray(baseline_qvalid) > 0
    arm_qv = np.asarray(arm_qvalid) > 0
    q_common = base_qv & arm_qv
    q_base = np.asarray(baseline_q, dtype=np.float64)
    q_arm = np.asarray(arm_q, dtype=np.float64)
    result["q_common_valid"] = {
        "pixels": int(q_common.sum()),
        "signed_q_diff": _finite_percentiles((q_arm - q_base)[q_common]),
        "abs_q_diff": _finite_percentiles(np.abs(q_arm - q_base)[q_common]),
        "q_validity_flips": {
            "baseline_valid_arm_invalid": int((base_qv & ~arm_qv).sum()),
            "baseline_invalid_arm_valid": int((~base_qv & arm_qv).sum()),
        },
    }
    return result


def instance_boundary_split(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(interior, boundary) pixel masks: interior = 3x3 erosion of the mask,
    boundary = mask minus interior (the project's interior/band rule)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    interior = cv2.erode(m, np.ones((3, 3), np.uint8)).astype(bool)
    boundary = m.astype(bool) & ~interior
    return interior, boundary


def _paired_diff_stats(
    diff: np.ndarray, region: np.ndarray
) -> dict[str, Any]:
    """Signed mean plus absolute-value statistics of one difference map
    restricted to ``region``. Empty region -> n=0 with null statistics."""
    values = diff[region]
    if values.size == 0:
        return {"n": 0, "signed_mean": None, "abs_mean": None,
                "abs_p50": None, "abs_p95": None, "abs_max": None}
    a = np.abs(values)
    return {
        "n": int(values.size),
        "signed_mean": float(np.mean(values)),
        "abs_mean": float(np.mean(a)),
        "abs_p50": float(np.percentile(a, 50)),
        "abs_p95": float(np.percentile(a, 95)),
        "abs_max": float(np.max(a)),
    }


def instance_paired_comparison(
    instance_mask: np.ndarray,
    interior: np.ndarray,
    boundary: np.ndarray,
    base_disp: np.ndarray,
    base_valid: np.ndarray,
    base_q: np.ndarray,
    base_abs_q: np.ndarray,
    base_qvalid: np.ndarray,
    arm_disp: np.ndarray,
    arm_valid: np.ndarray,
    arm_q: np.ndarray,
    arm_abs_q: np.ndarray,
    arm_qvalid: np.ndarray,
) -> dict[str, Any]:
    """Paired per-instance statistics on FIXED regions.

    The comparison regions are fixed by construction: the instance mask and
    its interior/boundary split come from the mask alone and are identical
    for both sides. The measured common region is
    ``instance_mask & baseline_qvalid & arm_qvalid`` (disparity:
    ``instance_mask & baseline_valid & arm_valid``). All differences are
    per-pixel on those common regions:

    - ``signed_dq_*``: arm_q - baseline_q; ``signed_mean`` is a mean
      DIFFERENCE (positive and negative changes cancel) and is explicitly
      NOT a pixel error metric; ``abs_mean`` of the same difference IS the
      per-pixel MAE on the common region.
    - ``dabs_q_*``: arm_abs_q - baseline_abs_q; ``signed_mean`` can be
      nonzero through cancellation and is a feature-magnitude change, not
      an error; its absolute-value statistics are reported alongside.
    - validity: baseline/arm/common counts inside the instance plus
      lost/gained flips, reported for q and for disparity separately.

    Empty common regions are reported with n=0 and null statistics.
    """
    inst = np.asarray(instance_mask) > 0
    b_qv = np.asarray(base_qvalid) > 0
    a_qv = np.asarray(arm_qvalid) > 0
    b_v = np.asarray(base_valid) > 0
    a_v = np.asarray(arm_valid) > 0

    def block(region: np.ndarray) -> dict[str, Any]:
        common_q = region & b_qv & a_qv
        common_v = region & b_v & a_v
        out: dict[str, Any] = {
            "pixel_count": int(region.sum()),
            "q_valid_counts": {
                "baseline": int((region & b_qv).sum()),
                "arm": int((region & a_qv).sum()),
                "common": int(common_q.sum()),
                "lost": int((region & b_qv & ~a_qv).sum()),
                "gained": int((region & a_qv & ~b_qv).sum()),
            },
            "signed_dq": _paired_diff_stats(
                (np.asarray(arm_q, dtype=np.float64)
                 - np.asarray(base_q, dtype=np.float64)), common_q),
            "dabs_q": _paired_diff_stats(
                (np.asarray(arm_abs_q, dtype=np.float64)
                 - np.asarray(base_abs_q, dtype=np.float64)), common_q),
            "disparity_valid_counts": {
                "baseline": int((region & b_v).sum()),
                "arm": int((region & a_v).sum()),
                "common": int(common_v.sum()),
                "lost": int((region & b_v & ~a_v).sum()),
                "gained": int((region & a_v & ~b_v).sum()),
            },
            "disparity_diff": _paired_diff_stats(
                (np.asarray(arm_disp, dtype=np.float64)
                 - np.asarray(base_disp, dtype=np.float64)), common_v),
        }
        return out

    result = {"region": "instance", "pixel_count": int(inst.sum())}
    result.update(block(inst))
    interior_block = block(inst & interior)
    interior_block["region"] = "interior_fixed"
    boundary_block = block(inst & boundary)
    boundary_block["region"] = "boundary_fixed"
    result["interior"] = interior_block
    result["boundary"] = boundary_block
    return result


# ---------------------------------------------------------------------------
# Residual capacity statistics (saved logits only; no model is ever run)


class IdentityError(ValueError):
    """A saved record violates the identity/shape/finiteness contract."""


def _validate_logit_row(row: dict, *, source: str) -> None:
    for key in ("sample_name", "class_id", "prediction", "logits"):
        if key not in row:
            raise IdentityError(f"{source}: record missing {key!r}")
    name = row["sample_name"]
    if not isinstance(name, str) or not name:
        raise IdentityError(f"{source}: bad sample_name {name!r}")
    class_id = row["class_id"]
    if type(class_id) is not int or not 0 <= class_id < len(CLASS_NAMES):
        raise IdentityError(f"{source} {name!r}: class_id out of range: {class_id!r}")
    prediction = row["prediction"]
    if type(prediction) is not int or not 0 <= prediction < len(CLASS_NAMES):
        raise IdentityError(f"{source} {name!r}: prediction out of range: {prediction!r}")
    logits = row["logits"]
    if not isinstance(logits, list) or len(logits) != len(CLASS_NAMES):
        raise IdentityError(
            f"{source} {name!r}: logits must be a {len(CLASS_NAMES)}-list, got {logits!r}")
    values = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise IdentityError(f"{source} {name!r}: non-finite logits")
    if int(np.argmax(values)) != prediction:
        raise IdentityError(
            f"{source} {name!r}: prediction {prediction} != argmax {int(np.argmax(values))}")


def load_and_merge_p0_oracles(
    fold_files: dict[str, Any],
    manifest_samples: list[dict],
) -> list[dict]:
    """Load the three frozen P0 outer-eval oracle files, validate every
    row's identity and merge them by sample_name (never by zip or count).

    Each ``fold_files`` value is either a JSONL path or an in-memory list
    of rows (tests/synthetic worlds). Cross-checks each row against the
    local feature manifest: the row's fold must be exactly the fold whose
    role for that sample is ``outer_eval``, and class_id/group must agree.
    """
    by_name: dict[str, dict] = {}
    role_of = {}
    class_of = {}
    group_of = {}
    for sample in manifest_samples:
        name = sample.get("sample_name")
        if not isinstance(name, str) or not name:
            raise IdentityError("manifest row without sample_name")
        if name in role_of:
            raise IdentityError(f"manifest duplicate sample_name {name!r}")
        outer = [fold for fold, role in sample.get("fold_roles", {}).items()
                 if role == "outer_eval"]
        if len(outer) != 1:
            raise IdentityError(
                f"manifest sample {name!r}: expected exactly one outer_eval "
                f"role, got {outer!r}")
        role_of[name] = outer[0]
        class_of[name] = sample.get("class_id")
        group_of[name] = sample.get("group")
    for fold, source in sorted(fold_files.items()):
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8") as handle:
                raw_rows = []
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    label = f"{source}:{line_no}"
                    try:
                        raw_rows.append(
                            (label, json.loads(line)))
                    except ValueError as exc:
                        raise IdentityError(
                            f"{label}: invalid JSON ({exc})") from exc
        else:
            raw_rows = [(f"{fold}[{i}]", row)
                        for i, row in enumerate(source)]
        for source_label, row in raw_rows:
            _validate_logit_row(row, source=source_label)
            name = row["sample_name"]
            if name in by_name:
                raise IdentityError(
                    f"duplicate sample_name across folds: {name!r} in "
                    f"{fold} and {by_name[name]['fold']}")
            if name not in role_of:
                raise IdentityError(
                    f"{source_label}: sample {name!r} is not in the frozen "
                    f"feature manifest")
            if role_of[name] != fold:
                raise IdentityError(
                    f"{source_label}: sample {name!r} appears in {fold} but "
                    f"its outer_eval role is {role_of[name]!r}")
            if class_of[name] is not None and int(class_of[name]) != row["class_id"]:
                raise IdentityError(
                    f"{source_label}: class_id {row['class_id']} != manifest "
                    f"{class_of[name]!r} for {name!r}")
            if group_of[name] is not None and group_of[name] != row.get("group"):
                raise IdentityError(
                    f"{source_label}: group mismatch for {name!r}")
            by_name[name] = {**row, "fold": fold}
    return [by_name[name] for name in sorted(by_name)]


def residual_capacity_stats(rows: list[dict]) -> dict[str, Any]:
    """Per-error capacity statistics: m = logit[pred] - logit[true].

    m >= 1 means even an ideal per-class correction bounded by the frozen
    rule (|delta| <= 0.5 per class, so any score gap changes by < 1) cannot
    flip this sample. This is a capacity upper bound only; m < 1 does NOT
    mean the correction succeeds.
    """
    per_fold: dict[str, dict[str, int]] = {}
    per_class: dict[str, int] = {}
    errors: list[dict] = []
    m_values: list[float] = []
    for row in rows:
        per_fold.setdefault(row["fold"], {"samples": 0, "errors": 0,
                                          "m_ge_1": 0})
    for row in rows:
        logits = np.asarray(row["logits"], dtype=np.float64)
        true_class = int(row["class_id"])
        prediction = int(row["prediction"])
        if prediction == true_class:
            continue
        m = float(logits[prediction] - logits[true_class])
        m_values.append(m)
        fold = row["fold"]
        bucket = per_fold.setdefault(
            fold, {"samples": 0, "errors": 0, "m_ge_1": 0})
        bucket["errors"] += 1
        if m >= 1.0:
            bucket["m_ge_1"] += 1
        per_class[CLASS_NAMES[true_class]] = per_class.get(CLASS_NAMES[true_class], 0) + 1
        errors.append({
            "sample_name": row["sample_name"],
            "fold": fold,
            "group": row.get("group"),
            "true_class": true_class,
            "true_class_name": CLASS_NAMES[true_class],
            "predicted_class": prediction,
            "predicted_class_name": CLASS_NAMES[prediction],
            "m": m,
            "m_ge_1": bool(m >= 1.0),
            "has_valid_s": row.get("has_valid_s"),
        })
    for row in rows:
        per_fold[row["fold"]]["samples"] += 1
    total = len(rows)
    total_errors = len(errors)
    m_ge_1 = sum(1 for e in errors if e["m_ge_1"])
    missing_s_errors = [e for e in errors if e["has_valid_s"] is False]
    return {
        "definition": {
            "m": "logit[prediction] - logit[true_class] of the frozen P0 logits",
            "m_ge_1": "even an ideal per-class residual cannot flip the sample "
                      "because the frozen rule changes any two-class score gap "
                      "by less than 1 (delta = valid_s * 0.5 * tanh(head)); "
                      "capacity upper bound only, not realized gain",
            "accuracy_not_macro_f1": "the upper bound applies to per-sample "
                                     "accuracy; no Macro-F1 upper bound is claimed",
        },
        "totals": {
            "samples": total,
            "errors": total_errors,
            "m_ge_1": m_ge_1,
            "m_lt_1": total_errors - m_ge_1,
            "errors_with_missing_polar": len(missing_s_errors),
            "m_lt_1_with_valid_polar": sum(
                1 for e in errors if not e["m_ge_1"] and e["has_valid_s"] is not False),
        },
        "per_fold": per_fold,
        "errors_per_true_class": per_class,
        "per_error_records": errors,
        "m_statistics_on_errors": _finite_percentiles(
            np.asarray(m_values, dtype=np.float64)) if m_values else {
            "n": 0, "mean": None, "p50": None, "p95": None, "max_abs": None},
        "note": "m < 1 is necessary but NOT sufficient for a correction; "
                "missing-S rows fall back to the exact frozen P0 regardless of m",
    }


def pipeline_gt_prediction_join(
    frames: list[dict], r0_rows: list[dict]
) -> dict[str, Any]:
    """Join runtime GT with R0 predictions by (frame_id, matched_gt_id),
    verifying the saved match relations instead of trusting row counts."""
    errors: list[dict] = []
    total_gt = total_pred = total_matches = 0
    fallback_matched = 0
    frame_ids = set()
    for frame in frames:
        frame_id = frame["frame_id"]
        if frame_id in frame_ids:
            raise IdentityError(f"duplicate frame_id in frames: {frame_id!r}")
        frame_ids.add(frame_id)
        if frame.get("frame_error") is not None:
            raise IdentityError(f"frame {frame_id!r} has frame_error; the "
                                "join refuses incomplete frames")
        gt_classes = {int(g["gt_id"]): int(g["class_id"]) for g in frame["gt"]}
        pred_ids = {int(p["pred_instance_id"]) for p in frame["predictions"]}
        matched_gt = set()
        matched_pred = set()
        for gt_id, pred_id, _iou in frame["matches"]:
            gt_id, pred_id = int(gt_id), int(pred_id)
            if gt_id not in gt_classes:
                raise IdentityError(
                    f"{frame_id}: match references unknown gt_id {gt_id}")
            if pred_id not in pred_ids:
                raise IdentityError(
                    f"{frame_id}: match references unknown pred {pred_id}")
            if gt_id in matched_gt or pred_id in matched_pred:
                raise IdentityError(
                    f"{frame_id}: matches are not one-to-one")
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
        declared_missed = {int(g) for g in frame["missed_gt_ids"]}
        if declared_missed != set(gt_classes) - matched_gt:
            raise IdentityError(f"{frame_id}: missed_gt_ids inconsistent")
        declared_unmatched = {int(p) for p in frame["unmatched_pred_ids"]}
        if declared_unmatched != pred_ids - matched_pred:
            raise IdentityError(f"{frame_id}: unmatched_pred_ids inconsistent")
        total_gt += len(gt_classes)
        total_pred += len(pred_ids)
        total_matches += len(frame["matches"])
    r0_by_key = {}
    for row in r0_rows:
        key = (row["frame_id"], int(row["pred_instance_id"]))
        if key in r0_by_key:
            raise IdentityError(f"duplicate R0 row for {key!r}")
        if row.get("candidate_id") != "R0":
            raise IdentityError(f"non-R0 row passed to the join: {key!r}")
        logits = row.get("final_logits")
        if not isinstance(logits, list) or len(logits) != len(CLASS_NAMES) \
                or not all(np.isfinite(v) for v in logits):
            raise IdentityError(f"R0 row {key!r}: bad final_logits")
        if int(np.argmax(logits)) != int(row["predicted_class"]):
            raise IdentityError(f"R0 row {key!r}: predicted_class != argmax")
        r0_by_key[key] = row
    matches_by_pred = {}
    gt_class_by_frame_pred = {}
    frame_meta = {}
    for frame in frames:
        for g in frame["gt"]:
            gt_class_by_frame_pred[(frame["frame_id"], int(g["gt_id"]))] = \
                int(g["class_id"])
        frame_meta[frame["frame_id"]] = (frame.get("fold"), frame.get("group"))
        for gt_id, pred_id, iou in frame["matches"]:
            matches_by_pred[(frame["frame_id"], int(pred_id))] = (
                int(gt_id), float(iou))
    joined = 0
    for key, row in r0_by_key.items():
        if key[0] not in frame_ids:
            raise IdentityError(f"R0 row for unknown frame {key!r}")
        row_fold = row.get("fold")
        row_group = row.get("group")
        frame_fold, frame_group = frame_meta[key[0]]
        if row_fold is not None and frame_fold is not None and row_fold != frame_fold:
            raise IdentityError(
                f"R0 row {key!r}: fold {row_fold!r} != frame fold {frame_fold!r}")
        if row_group is not None and frame_group is not None \
                and row_group != frame_group:
            raise IdentityError(
                f"R0 row {key!r}: group mismatch with the frame record")
        saved = matches_by_pred.get(key)
        declared = row.get("matched_gt_id")
        if saved is None:
            if declared is not None:
                raise IdentityError(
                    f"R0 row {key!r} claims matched_gt_id {declared} but the "
                    "frame records no such match")
            continue
        if declared is None or int(declared) != saved[0]:
            raise IdentityError(
                f"R0 row {key!r}: matched_gt_id {declared!r} != saved match "
                f"{saved[0]}")
        joined += 1
        true_class = gt_class_by_frame_pred[(key[0], saved[0])]
        logits = np.asarray(row["final_logits"], dtype=np.float64)
        predicted = int(row["predicted_class"])
        record = {
            "frame_id": key[0],
            "gt_id": saved[0],
            "pred_instance_id": key[1],
            "iou": saved[1],
            "true_class": true_class,
            "true_class_name": CLASS_NAMES[true_class],
            "predicted_class": predicted,
            "predicted_class_name": CLASS_NAMES[predicted],
            "used_fallback": bool(row.get("used_fallback")),
            "fallback_reason": row.get("fallback_reason"),
        }
        if row.get("used_fallback"):
            fallback_matched += 1
        if predicted != true_class:
            m = float(logits[predicted] - logits[true_class])
            record.update({"classification": "error", "m": m,
                           "m_ge_1": bool(m >= 1.0)})
            errors.append(record)
        else:
            record["classification"] = "correct"
    return {
        "totals": {
            "frames": len(frames),
            "gt_instances": total_gt,
            "predictions": total_pred,
            "matches": total_matches,
            "r0_rows": len(r0_by_key),
            "matched_r0_joined": joined,
            "errors": len(errors),
            "m_ge_1": sum(1 for e in errors if e["m_ge_1"]),
            "m_lt_1": sum(1 for e in errors if not e["m_ge_1"]),
            "errors_with_missing_polar_fallback": sum(
                1 for e in errors if e["used_fallback"]),
        },
        "note": "R0 final_logits are the runtime P0 logits of the predicted "
                "instance (different inputs from the outer-eval oracles); "
                "m thresholds have the same frozen-rule meaning",
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Frozen frame selection (task C)


class SelectionConflict(RuntimeError):
    """The required selection does not fit the frozen frame cap."""


def select_diagnostic_frames(
    protocol_frames: list[dict],
    control_panel_frame_ids: list[str],
    missing_s_rows: list[dict],
    pipeline_m_lt1_frames: list[str],
    max_frames: int,
    real_fish_class_id: int = 3,
) -> dict[str, Any]:
    """Deterministic diagnostic frame selection from stable identities only.

    Required coverage (every rule contributes frames; duplicates accumulate
    reasons, never drop coverage):
    - every original-val frame containing real_fish GT;
    - the dev-pool no-target frames;
    - the frames of the existing normal-control atlas panels;
    - at least one frame per capture group (first frame by sorted frame_id);
    - one frame per class that has polar-missing samples in the dev pool
      (first such sample by sorted sample_name; class/error information is
      used for this retrospective selection ONLY, never for online routing
      or model selection);
    - the frames of the pipeline join's m < 1 errors.

    The remaining slots (up to ``max_frames``) are filled by cycling the
    groups in sorted order and taking each group's next unselected frame in
    sorted frame_id order. A required set larger than ``max_frames`` is a
    :class:`SelectionConflict`, never a silent drop.
    """
    if type(max_frames) is not int or max_frames <= 0:
        raise ValueError(f"max_frames must be a positive int, got {max_frames!r}")
    by_id: dict[str, dict] = {}
    for frame in protocol_frames:
        fid = frame["frame_id"]
        if fid in by_id:
            raise IdentityError(f"duplicate frame_id in protocol: {fid!r}")
        by_id[fid] = frame
    groups = sorted({frame["group"] for frame in protocol_frames})
    frames_of_group = {
        group: sorted(f["frame_id"] for f in protocol_frames if f["group"] == group)
        for group in groups
    }

    selected: dict[str, list[str]] = {}

    def add(frame_id: str, reason: str) -> None:
        if frame_id not in by_id:
            raise IdentityError(
                f"selection rule {reason!r} produced unknown frame {frame_id!r}")
        selected.setdefault(frame_id, []).append(reason)

    for fid in sorted(by_id):
        f = by_id[fid]
        if f["split"] == "val" and int(f.get("gt_class_counts", {}).get(
                str(real_fish_class_id), 0)) > 0:
            add(fid, "val_real_fish_gt")
        if int(f.get("gt_instances", 0)) == 0:
            add(fid, "no_target_dev")
    for fid in sorted(set(control_panel_frame_ids)):
        add(fid, "normal_control_panel")
    per_class_frame: dict[str, tuple[str, str]] = {}
    for row in sorted(missing_s_rows, key=lambda r: r["sample_name"]):
        if row.get("has_valid_s") is not False:
            continue
        cls = row.get("class_name")
        if cls and cls not in per_class_frame:
            per_class_frame[cls] = (row["source_frame"], row["sample_name"])
    for cls in sorted(per_class_frame):
        fid, sample = per_class_frame[cls]
        add(fid, f"missing_polar_class_{cls}(first_by_sample_name:{sample})")
    for fid in sorted(set(pipeline_m_lt1_frames)):
        add(fid, "pipeline_m_below_1_error")
    required_ids = sorted(selected)
    for group in groups:
        if not any(by_id[fid]["group"] == group for fid in required_ids):
            add(frames_of_group[group][0], f"group_coverage:{group}")
    required_ids = sorted(selected)
    if len(required_ids) > max_frames:
        raise SelectionConflict(
            f"required selection has {len(required_ids)} frames > cap "
            f"{max_frames}; refusing to drop required coverage or to "
            "silently widen the budget")

    group_cursor = 0
    while len(selected) < max_frames:
        group = groups[group_cursor % len(groups)]
        group_cursor += 1
        progressed = False
        for fid in frames_of_group[group]:
            if fid not in selected:
                add(fid, f"fill_group_coverage:{group}")
                progressed = True
                break
        if not progressed and all(
            all(f in selected for f in frames_of_group[g]) for g in groups
        ):
            break
    order = sorted(selected)
    frames_out = []
    for fid in order:
        f = by_id[fid]
        frames_out.append({
            "frame_id": fid,
            "group": f["group"],
            "split": f["split"],
            "gt_instances": int(f.get("gt_instances", 0)),
            "reasons": selected[fid],
        })
    return {
        "max_frames": max_frames,
        "required_count": len(required_ids),
        "selected_count": len(frames_out),
        "groups_total": len(groups),
        "groups_covered": len({f["group"] for f in frames_out}),
        "frames": frames_out,
    }


# ---------------------------------------------------------------------------
# Synthetic counterexamples with independent analytic ground truth (task B)


def _analytic_bilinear_taps(src_len: int, dst_len: int) -> list[list[tuple[int, float]]]:
    """OpenCV INTER_LINEAR 1-D tap positions/weights from first principles.

    Independent reference arithmetic (``src = (dst + 0.5) * src/dst - 0.5``,
    clamped at the borders); the tests verify it against hand-computed
    values. Used only to PREDICT what the production chain should do, never
    to produce its output. For upscaling (``dst_len >= src_len``).
    """
    taps: list[list[tuple[int, float]]] = []
    ratio = src_len / dst_len
    for dst in range(dst_len):
        src = (dst + 0.5) * ratio - 0.5
        src = max(0.0, min(src, src_len - 1.0))
        i0 = int(np.floor(src))
        frac = src - i0
        i1 = min(i0 + 1, src_len - 1)
        if frac == 0.0 or i0 == i1:
            taps.append([(i0, 1.0)])
        else:
            taps.append([(i0, 1.0 - frac), (i1, frac)])
    return taps


def build_correspondence_pair(
    pattern: np.ndarray, disparity: float, right_coefficient: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic rectified pair with analytically known correspondence and q.

    ``pattern`` (H, W + disparity) holds the integer-valued scene.
    left(x, y) = pattern(x, y); right(x, y) = rint(c * pattern(x + d, y)).
    Both images are exact uint8, so the production ``to_gray_u8`` boundary
    is lossless. The TRUE signed q of left pixel x (what the production
    chain would measure at the true disparity d) is computable from the
    two constructed arrays alone:
        q_true(x) = (left(x) - right(x-d)) / (left(x) + right(x-d))   [x >= d]
    because an integer disparity makes the bilinear warp sample exactly
    one pixel. Pixels with x < d (the warp would leave the image) or a
    denominator below the production brightness gate are truth-invalid
    (NaN). Nothing here reads the code under test.
    Returns (left_u8, right_u8, q_true).
    """
    pattern = np.asarray(pattern)
    if not np.all(np.equal(np.rint(pattern), pattern)):
        raise ValueError("pattern must be integer-valued for an exact uint8 pair")
    h, wtot = pattern.shape
    d = float(disparity)
    if not d.is_integer() or d < 1 or wtot <= d:
        raise ValueError("pattern width must exceed an integer disparity >= 1")
    di = int(d)
    left = np.clip(pattern[:, :wtot - di], 0, 255).astype(np.uint8)
    right = np.clip(
        np.rint(right_coefficient * pattern[:, di:]), 0, 255
    ).astype(np.uint8)
    lf = left.astype(np.float64)
    rf = right.astype(np.float64)
    q_true = np.full(left.shape, np.nan)
    n_truth = max(0, left.shape[1] - di)  # left pixels x = di .. W-1
    if n_truth:
        l_part = lf[:, di:di + n_truth]
        r_part = rf[:, :n_truth]
        denom = l_part + r_part
        with np.errstate(invalid="ignore", divide="ignore"):
            q = (l_part - r_part) / denom
        measured = denom >= 10.0
        q_true[:, di:di + n_truth][measured] = q[measured]
    return left, right, q_true


def evaluate_q_chain(
    left_u8: np.ndarray,
    right_u8: np.ndarray,
    disp_full: np.ndarray,
    valid_full: np.ndarray,
    q_true: np.ndarray,
    disparity_true: np.ndarray,
) -> dict[str, Any]:
    """Run the production polar chain on one disparity variant and compare
    against the analytic ground truth. Ground truth comes from
    ``build_correspondence_pair``; nothing here generates expectations from
    the code under test."""
    result = compute_polar_features(
        left_u8, right_u8, disp_full, disparity_valid=valid_full,
    )
    q_meas = result.signed_q.astype(np.float64)
    valid = result.valid_mask
    truth_valid = np.isfinite(q_true)
    q_err = np.abs(q_meas - q_true)
    d_err = np.abs(disp_full.astype(np.float64) - disparity_true.astype(np.float64))
    marked_valid = valid & truth_valid
    return {
        "production_valid_mask": valid,
        "production_signed_q": result.signed_q,
        "production_abs_q": result.abs_q,
        "valid_pixels": int(valid.sum()),
        "valid_marked_with_disparity_error": int(
            (marked_valid & (d_err > 1e-4)).sum()),
        "valid_marked_with_q_error": int((marked_valid & (q_err > 1e-4)).sum()),
        "disparity_error_on_valid_marked": _finite_percentiles(
            d_err[marked_valid]),
        "q_error_on_valid_marked": _finite_percentiles(q_err[marked_valid]),
        "validity_flips_vs_truth": {
            "measured_valid_truth_invalid": int((valid & ~truth_valid).sum()),
            "truth_valid_measured_invalid": int((~valid & truth_valid).sum()),
        },
    }


def run_synthetic_scenarios(
    stereo_cfg: StereoMatcherConfig,
    *,
    w_min: float = 0.5,
    seed: int = 2026,
) -> dict[str, dict[str, Any]]:
    """The four frozen counterexamples plus the in-situ production check.

    Every scenario has independent analytic truth; the function reports
    measurements only. The tests assert the analytic expectations.
    """
    rng = np.random.default_rng(seed)
    scale = stereo_cfg.scale
    results: dict[str, dict[str, Any]] = {}

    def _make_world(sh: int, sw: int, d0: float, c: float):
        out_h, out_w = sh * 4, sw * 4
        pattern_w = out_w + int(d0)
        pattern = rng.integers(60, 200, size=(out_h, pattern_w))
        left, right, q_true = build_correspondence_pair(pattern, d0, c)
        disparity_true = np.full((out_h, out_w), d0, dtype=np.float64)
        return left, right, q_true, disparity_true, out_h, out_w

    # --- S1: constant plane with an adjacent invalid low-res block -------
    sh, sw = 6, 24
    d0 = 6.0
    left_u8, right_u8, q_true, d_true, out_h, out_w = _make_world(sh, sw, d0, 0.6)
    disp_small = np.full((sh, sw), d0 * scale, dtype=np.float32)
    valid_small = np.ones((sh, sw), dtype=np.uint8)
    valid_small[2:4, 12:14] = 0
    disp_small[valid_small == 0] = 0.0
    core = LowResStereoCore(disp_small, valid_small, float(valid_small.mean()),
                            0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=PRODUCTION_UPSAMPLE)
    ctrl_d, ctrl_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=WEIGHTED_UPSAMPLE, w_min=w_min)
    prod = evaluate_q_chain(left_u8, right_u8, prod_d, prod_v, q_true, d_true)
    ctrl = evaluate_q_chain(left_u8, right_u8, ctrl_d, ctrl_v, q_true, d_true)
    results["constant_plane_adjacent_invalid"] = {
        "production": prod,
        "control_valid_weighted": ctrl,
        "expected_verdict": "production marks valid pixels whose disparity "
                            "is pulled toward zero by the zero-filled "
                            "invalid block; the control returns the exact "
                            "plane where the valid weight is sufficient",
    }

    # --- S2: fully valid plane -------------------------------------------
    disp_small = np.full((sh, sw), d0 * scale, dtype=np.float32)
    valid_small = np.ones((sh, sw), dtype=np.uint8)
    core = LowResStereoCore(disp_small, valid_small, 1.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=PRODUCTION_UPSAMPLE)
    ctrl_d, ctrl_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=WEIGHTED_UPSAMPLE, w_min=w_min)
    prod = evaluate_q_chain(left_u8, right_u8, prod_d, prod_v, q_true, d_true)
    ctrl = evaluate_q_chain(left_u8, right_u8, ctrl_d, ctrl_v, q_true, d_true)
    results["fully_valid_plane"] = {
        "production": prod,
        "control_valid_weighted": ctrl,
        "expected_verdict": "both methods return the exact plane; no "
                            "zero-fill risk exists without invalid regions",
    }

    # --- S3: valid-valid depth discontinuity ------------------------------
    d1, d2 = 6.0, 12.0
    left_u8, right_u8, q_true, d_true, out_h, out_w = _make_world(sh, sw, d1, 0.6)
    del d_true  # the pair is built with a single plane; q truth presumes d1
    disp_small = np.full((sh, sw), d1 * scale, dtype=np.float32)
    disp_small[:, sw // 2:] = d2 * scale
    valid_small = np.ones((sh, sw), dtype=np.uint8)
    core = LowResStereoCore(disp_small, valid_small, 1.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=PRODUCTION_UPSAMPLE)
    ctrl_d, ctrl_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=WEIGHTED_UPSAMPLE, w_min=w_min)
    prod = evaluate_q_chain(left_u8, right_u8, prod_d, prod_v, q_true,
                            np.full((out_h, out_w), d1, dtype=np.float64))
    ctrl = evaluate_q_chain(left_u8, right_u8, ctrl_d, ctrl_v, q_true,
                            np.full((out_h, out_w), d1, dtype=np.float64))
    results["depth_discontinuity_both_valid"] = {
        "production": prod,
        "control_valid_weighted": ctrl,
        "upsampled_maps_bitwise_identical": bool(
            np.array_equal(prod_d, ctrl_d) and np.array_equal(prod_v, ctrl_v)),
        "expected_verdict": "with no invalid low-res pixels both methods are "
                            "identical: the control does NOT repair "
                            "valid-valid boundary mixing (disclosed "
                            "limitation of the control, not a production "
                            "defect claim)",
    }

    # --- S4: all-invalid low-res region -----------------------------------
    disp_small = np.zeros((sh, sw), dtype=np.float32)
    valid_small = np.zeros((sh, sw), dtype=np.uint8)
    core = LowResStereoCore(disp_small, valid_small, 0.0, 0.0, (sh, sw))
    prod_d, prod_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=PRODUCTION_UPSAMPLE)
    ctrl_d, ctrl_v = upsample_lowres(core, out_w, out_h, scale,
                                     method=WEIGHTED_UPSAMPLE, w_min=w_min)
    results["all_invalid_region"] = {
        "production_valid_pixels": int(prod_v.sum()),
        "control_valid_pixels": int(ctrl_v.sum()),
        "expected_verdict": "no valid pixel is fabricated by either method",
    }

    # --- S5: in-situ production matcher zero-fill risk --------------------
    results["production_matcher_zero_fill_risk"] = _in_situ_zero_fill_risk(
        stereo_cfg, w_min=w_min, seed=seed)
    return results


def _in_situ_zero_fill_risk(
    stereo_cfg: StereoMatcherConfig, *, w_min: float, seed: int
) -> dict[str, Any]:
    """Run the REAL production matcher on a textured planar scene with an
    engineered textureless block, then compare the production upsample
    against the valid-weighted control on the same low-res core.

    The scene is built at the SGBM working resolution (constant 4x4 blocks)
    so INTER_AREA downscaling is lossless and SGBM matches the true
    disparity; an identity brightness transfer keeps matching quality out
    of the picture. The zero-fill risk under test lives in the final
    upsample step, which only mixes taps when scale < 1."""
    import dataclasses

    from core.stereo_matching import StereoMatcher

    cfg = dataclasses.replace(
        stereo_cfg, scale=0.25, max_disparity=768, block_size=7)
    rng = np.random.default_rng(seed + 1)
    h, w = 64, 256
    d0 = 10.0
    low = rng.integers(60, 200, size=(h // 4, (w + int(d0) + 3) // 4))
    pattern = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)[:, : w + int(d0)]
    left, right, _ = build_correspondence_pair(pattern, d0, 1.0)
    # textureless block: identical constant columns in BOTH views
    left[:, 56:64] = 128
    right[:, 56:64] = 128
    matcher = StereoMatcher(cfg)
    result = matcher.compute(left, right)
    core = run_sgbm_lowres(left, right, cfg)
    prod_d, prod_v = upsample_lowres(core, w, h, cfg.scale,
                                     method=PRODUCTION_UPSAMPLE)
    ctrl_d, ctrl_v = upsample_lowres(core, w, h, cfg.scale,
                                     method=WEIGHTED_UPSAMPLE, w_min=w_min)
    prod_matches_engine = bool(
        np.array_equal(result.disparity, prod_d)
        and np.array_equal(result.valid, prod_v))
    # plane reference: median of the low-res disparities far from the block
    plane = core.disp_small[:, :10][core.valid_small[:, :10] > 0]
    plane_d0 = float(np.median(plane)) / cfg.scale if plane.size else None
    # Attribution at production-valid pixels whose bilinear stencil touches
    # the zero-filled invalid block: the CONTROL (weighted mean of valid
    # taps only) separates "pulled down by the zero-filled taps" (control
    # restores a higher value) from "the low-res valid disparities
    # themselves are already biased near the block" (control agrees with
    # production) — the latter is SGBM edge response, not an upsample
    # defect.
    sh_low, sw_low = core.low_res_shape
    wx = _analytic_bilinear_taps(sw_low, w)
    wy = _analytic_bilinear_taps(sh_low, h)
    valid_low = core.valid_small > 0
    invalid_mass = np.zeros((h, w), dtype=np.float64)
    for y, taps_y in enumerate(wy):
        for x, taps_x in enumerate(wx):
            mass = 0.0
            for iy, wy_ in taps_y:
                for ix, wx_ in taps_x:
                    if not valid_low[iy, ix]:
                        mass += wy_ * wx_
            invalid_mass[y, x] = mass
    touches = invalid_mass > 1e-9
    prod_valid = prod_v > 0
    touch_valid = touches & prod_valid
    zero_fill_pull = np.zeros((h, w), dtype=bool)
    pull_max = None
    if touch_valid.any():
        diff = ctrl_d.astype(np.float64) - prod_d.astype(np.float64)
        zero_fill_pull = touch_valid & (diff > 0.05)
        if zero_fill_pull.any():
            pull_max = float(np.max(diff[zero_fill_pull]))
    biased_below_plane = prod_valid & (prod_d < plane_d0 * 0.9) \
        if plane_d0 else None
    n_biased = int(biased_below_plane.sum()) if biased_below_plane is not None else 0
    return {
        "production_matcher_valid_pixels": int(result.valid.sum()),
        "lowres_core_matches_production_bitwise": prod_matches_engine,
        "plane_disparity_fullres": plane_d0,
        "valid_marked_pixels_with_invalid_tap_in_stencil": int(touch_valid.sum()),
        "zero_fill_pull_pixels_control_restores": int(zero_fill_pull.sum()),
        "max_control_minus_production_at_pull_pixels_px": pull_max,
        "valid_marked_pixels_below_90pct_plane": n_biased,
        "expected_verdict": "the production upsample can mark valid pixels "
                            "whose disparity was pulled toward zero by the "
                            "zero-filled textureless block; pixels whose "
                            "bias the control does NOT restore come from "
                            "the low-res disparities themselves (SGBM edge "
                            "response), not from the zero-fill upsample",
    }
