"""OpenCV StereoSGBM stereo matching for the underwater polar stereo rig.

Replaces the legacy per-pixel Python NCC implementation. The matcher runs
once per frame on downscaled rectified images, enforces left-right
consistency, and upsamples disparity to full resolution. Per-instance depth
statistics (median disparity after MAD rejection, valid ratio, depth) are
derived from that single dense disparity map.

Assumes rectified inputs (row-aligned epipolar geometry).
"""

import collections.abc
import dataclasses
import time

import cv2
import numpy as np


def robust_disparity(disparities: np.ndarray) -> float:
    """Return a robust object-level disparity using median + MAD rejection."""
    values = np.asarray(disparities, dtype=np.float32)
    if values.size == 0:
        return 0.0

    if values.size > 3:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        inliers = values[np.abs(values - median) <= 2.0 * mad + 1e-6]
        if inliers.size > 0:
            values = inliers

    return float(np.median(values))


@dataclasses.dataclass(frozen=True)
class StereoMatcherConfig:
    """Configuration for :class:`StereoMatcher`.

    ``max_disparity``, ``block_size`` and ``lr_check_threshold_px`` are
    expressed in full-resolution pixels; matching internally runs at
    ``scale`` of the input size.
    """

    matcher: str = "sgbm"
    mode: str = "3way"
    max_disparity: int = 768
    scale: float = 0.25
    block_size: int = 7
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 32
    texture_threshold: float = 10.0
    lr_check_threshold_px: float = 2.0
    min_valid_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.matcher != "sgbm":
            raise ValueError(f"unsupported matcher: {self.matcher!r}")
        if self.mode not in {"sgbm", "hh", "3way"}:
            raise ValueError(f"unsupported SGBM mode: {self.mode!r}")
        if self.max_disparity <= 0:
            raise ValueError("max_disparity must be positive")
        if not 0.0 < self.scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        if self.block_size < 1 or self.block_size % 2 == 0:
            raise ValueError("block_size must be a positive odd integer")
        if self.lr_check_threshold_px < 0:
            raise ValueError("lr_check_threshold_px must be non-negative")
        if not 0.0 <= self.min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class StereoMatchResult:
    """Dense disparity result at full input resolution."""

    disparity: np.ndarray  # float32 (H, W), 0 where invalid
    valid: np.ndarray  # uint8 (H, W), 1 where the disparity is valid
    elapsed_s: float
    valid_ratio: float  # global ratio of valid pixels


@dataclasses.dataclass(frozen=True)
class HorizontalBand:
    """Full-width horizontal image strip used for band-restricted matching.

    ``y1`` is inclusive, ``y2`` exclusive. Bands always keep the full image
    width and the full configured disparity range; only rows are restricted.
    """

    y1: int
    y2: int

    def __post_init__(self) -> None:
        if self.y1 < 0 or self.y2 <= self.y1:
            raise ValueError(f"invalid band rows: y1={self.y1}, y2={self.y2}")

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def merge_horizontal_bands(
    bands: "collections.abc.Iterable[HorizontalBand]",
) -> list[HorizontalBand]:
    """Merge vertically overlapping (or touching) bands; drop empties.

    Returns bands sorted by ``y1`` that are pairwise disjoint, so no row is
    ever covered by two bands (which would double-count timing and valid
    pixels).
    """
    cleaned = sorted(
        (band for band in bands if band.height > 0), key=lambda b: (b.y1, b.y2)
    )
    merged: list[HorizontalBand] = []
    for band in cleaned:
        if merged and band.y1 <= merged[-1].y2:
            previous = merged[-1]
            merged[-1] = HorizontalBand(previous.y1, max(previous.y2, band.y2))
        else:
            merged.append(band)
    return merged


def build_horizontal_bands(
    bboxes,
    image_height: int,
    vertical_margin: int = 10,
) -> list[HorizontalBand]:
    """Build merged full-width bands from Model A bounding boxes.

    Each bbox (x1, y1, x2, y2) is expanded by ``vertical_margin`` rows, its
    x extent ignored (bands span the full width), clipped to the image and
    merged with any band it overlaps vertically.
    """
    bands: list[HorizontalBand] = []
    for bbox in bboxes:
        coords = [float(v) for v in bbox]
        if len(coords) != 4:
            raise ValueError(f"bbox must be (x1, y1, x2, y2), got {bbox!r}")
        y1 = max(0, int(np.floor(coords[1])) - int(vertical_margin))
        y2 = min(int(image_height), int(np.ceil(coords[3])) + int(vertical_margin))
        if y2 > y1:
            bands.append(HorizontalBand(y1, y2))
    return merge_horizontal_bands(bands)


@dataclasses.dataclass(frozen=True)
class InstanceDepth:
    """Per-instance disparity/depth statistics with an explicit validity field."""

    instance_id: int
    disparity: float
    valid_ratio: float
    depth_m: float | None
    valid: bool
    reason: str


def to_gray_u8(image: np.ndarray) -> np.ndarray:
    """Normalize an image to contiguous single-channel uint8 ``(H, W)``.

    Accepts ``(H, W)``, ``(H, W, 1)`` (e.g. Ultralytics-patched
    ``cv2.imread`` grayscale output), and 3/4-channel color arrays. Other
    dimensionality or channel counts raise ``ValueError``.
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        code = cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        arr = cv2.cvtColor(arr, code)
    else:
        raise ValueError(
            f"expected (H, W), (H, W, 1), 3- or 4-channel image, "
            f"got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def _num_disparities(max_disparity: int, scale: float) -> int:
    """Low-resolution numDisparities, rounded up to a positive multiple of 16."""
    raw = max_disparity * scale
    return max(16, int(np.ceil(raw / 16.0)) * 16)


class StereoMatcher:
    """Dense StereoSGBM matcher with left-right consistency checking.

    The dense disparity map is computed once per frame on downscaled images;
    all instance statistics reuse that single computation. Matching uses
    CLAHE-normalized contrast, but returned disparity is expressed in
    full-resolution pixels.
    """

    def __init__(self, config: StereoMatcherConfig | None = None) -> None:
        self.config = config or StereoMatcherConfig()

    @property
    def num_disparities(self) -> int:
        return _num_disparities(self.config.max_disparity, self.config.scale)

    def compute(self, left: np.ndarray, right: np.ndarray) -> StereoMatchResult:
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        if left_gray.shape != right_gray.shape:
            raise ValueError(
                f"left/right shape mismatch: {left_gray.shape} vs {right_gray.shape}"
            )
        if left_gray.ndim != 2:
            raise ValueError(f"expected single-channel image, got shape {left_gray.shape}")

        h, w = left_gray.shape
        disparity, valid, elapsed, valid_ratio = self._match_pair(left_gray, right_gray)
        if valid_ratio < self.config.min_valid_ratio:
            disparity = np.zeros((h, w), dtype=np.float32)
            valid = np.zeros((h, w), dtype=np.uint8)

        return StereoMatchResult(
            disparity=disparity,
            valid=valid,
            elapsed_s=elapsed,
            valid_ratio=valid_ratio,
        )

    def compute_bands(
        self,
        left: np.ndarray,
        right: np.ndarray,
        bands: list[HorizontalBand],
        full_image_threshold: float = 0.9,
    ) -> StereoMatchResult:
        """Band-restricted matching; full-size result, rows outside bands invalid.

        Only the merged band rows are matched (full width and the full
        configured disparity range inside each band). When the merged bands
        cover at least ``full_image_threshold`` of the image height, a single
        full-image :meth:`compute` pass is cheaper and is used instead.
        """
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        if left_gray.shape != right_gray.shape:
            raise ValueError(
                f"left/right shape mismatch: {left_gray.shape} vs {right_gray.shape}"
            )
        if left_gray.ndim != 2:
            raise ValueError(f"expected single-channel image, got shape {left_gray.shape}")
        if not 0.0 < full_image_threshold <= 1.0:
            raise ValueError("full_image_threshold must be in (0, 1]")

        h, w = left_gray.shape
        clipped_bands: list[HorizontalBand] = []
        for band in bands:
            y1 = max(0, int(band.y1))
            y2 = min(h, int(band.y2))
            if y2 > y1:
                clipped_bands.append(HorizontalBand(y1, y2))
        clipped = merge_horizontal_bands(clipped_bands)
        if not clipped:
            # No target rows: skip SGBM entirely.
            return StereoMatchResult(
                disparity=np.zeros((h, w), dtype=np.float32),
                valid=np.zeros((h, w), dtype=np.uint8),
                elapsed_s=0.0,
                valid_ratio=0.0,
            )

        coverage = sum(band.height for band in clipped)
        if coverage >= full_image_threshold * h:
            return self.compute(left_gray, right_gray)

        disparity = np.zeros((h, w), dtype=np.float32)
        valid = np.zeros((h, w), dtype=np.uint8)
        elapsed = 0.0
        valid_pixels = 0
        for band in clipped:
            band_disp, band_valid, band_elapsed, band_ratio = self._match_pair(
                left_gray[band.y1 : band.y2], right_gray[band.y1 : band.y2]
            )
            elapsed += band_elapsed
            if band_ratio < self.config.min_valid_ratio:
                continue
            disparity[band.y1 : band.y2] = band_disp
            valid[band.y1 : band.y2] = band_valid
            valid_pixels += int(band_valid.sum())

        return StereoMatchResult(
            disparity=disparity,
            valid=valid,
            elapsed_s=elapsed,
            valid_ratio=valid_pixels / float(h * w),
        )

    def _match_pair(
        self, left_gray: np.ndarray, right_gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Core SGBM pipeline on same-shape grayscale images.

        Returns ``(disparity, valid, elapsed_s, valid_ratio)`` where the maps
        are at the input resolution and ``valid_ratio`` is measured over the
        whole (cropped) input. The caller applies ``min_valid_ratio`` policy.
        """
        cfg = self.config
        h, w = left_gray.shape
        sw = max(1, int(round(w * cfg.scale)))
        sh = max(1, int(round(h * cfg.scale)))
        left_small = cv2.resize(left_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_gray, (sw, sh), interpolation=cv2.INTER_AREA)

        num_disp = self.num_disparities
        if min(sw, sh) < cfg.block_size:
            # Scene too small for the configured block size; report no data
            # instead of crashing (matches legacy behaviour on tiny inputs).
            return np.zeros((h, w), np.float32), np.zeros((h, w), np.uint8), 0.0, 0.0

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        left_match = clahe.apply(left_small)
        right_match = clahe.apply(right_small)

        p1 = 8 * cfg.block_size * cfg.block_size
        p2 = 32 * cfg.block_size * cfg.block_size
        common = dict(
            blockSize=cfg.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=-1,  # internal check disabled; external check below
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
            minDisparity=0,
            numDisparities=num_disp,
            mode=sgbm_mode,
            **common,
        )
        # Right matcher: minDisparity=-num_disp makes the raw output v live in
        # [-num_disp, 0) with right(x, y) <-> left(x - v, y), so the positive
        # right-view disparity is -v.
        sgbm_right = cv2.StereoSGBM_create(
            minDisparity=-num_disp,
            numDisparities=num_disp,
            mode=sgbm_mode,
            **common,
        )

        # OpenCV SGBM invalidates the outermost numDisparities columns of the
        # reference image (leftmost for the left pass, rightmost for the right
        # pass). Pad both edges with replicated columns and crop afterwards so
        # the full view is matchable in both passes.
        pad = num_disp
        left_pad = cv2.copyMakeBorder(
            left_match, 0, 0, pad, pad, cv2.BORDER_REPLICATE
        )
        right_pad = cv2.copyMakeBorder(
            right_match, 0, 0, pad, pad, cv2.BORDER_REPLICATE
        )

        started = time.perf_counter()
        disp_l = (
            sgbm_left.compute(left_pad, right_pad).astype(np.float32)[:, pad:-pad]
            / 16.0
        )
        disp_r_neg = (
            sgbm_right.compute(right_pad, left_pad).astype(np.float32)[:, pad:-pad]
            / 16.0
        )
        disp_r = -disp_r_neg
        elapsed = time.perf_counter() - started

        # Left-right consistency: for left pixel (y, x) with disparity d, the
        # right pixel is x - d and its right-view disparity must equal d.
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

        disp_full = (
            cv2.resize(disp_small, (w, h), interpolation=cv2.INTER_LINEAR) / cfg.scale
        )
        valid_full = cv2.resize(
            valid_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        disp_full = np.where(valid_full > 0, disp_full, 0.0).astype(np.float32)

        return disp_full, valid_full, elapsed, valid_ratio

    def instance_stats(
        self,
        result: StereoMatchResult,
        mask: np.ndarray,
        instance_id: int = 0,
    ) -> InstanceDepth:
        """Disparity-level statistics for one instance mask.

        ``depth_m`` is always None here; use :meth:`instance_depths` (or
        :func:`disparity_to_depth`) to convert to meters.
        """
        mask_bool = np.asarray(mask > 0)
        if mask_bool.shape != result.disparity.shape:
            raise ValueError(
                f"mask shape {mask_bool.shape} != disparity shape {result.disparity.shape}"
            )
        if not mask_bool.any():
            return InstanceDepth(instance_id, 0.0, 0.0, None, False, "empty_mask")

        valid_in = mask_bool & (result.valid > 0)
        ratio = float(valid_in.sum()) / float(mask_bool.sum())
        if not valid_in.any():
            return InstanceDepth(instance_id, 0.0, ratio, None, False, "no_valid_disparity")

        disparity = robust_disparity(result.disparity[valid_in])
        if disparity <= 0:
            return InstanceDepth(instance_id, disparity, ratio, None, False, "nonpositive_disparity")
        if ratio < self.config.min_valid_ratio:
            return InstanceDepth(instance_id, disparity, ratio, None, False, "low_valid_ratio")
        return InstanceDepth(instance_id, disparity, ratio, None, True, "ok")

    def instance_depths(
        self,
        result: StereoMatchResult,
        masks: list[np.ndarray],
        baseline: float,
        focal_length: float,
    ) -> list[InstanceDepth]:
        """Per-instance depth statistics reusing one precomputed dense result."""
        depths: list[InstanceDepth] = []
        for i, mask in enumerate(masks):
            stats = self.instance_stats(result, mask, instance_id=i)
            if stats.reason in {"ok", "low_valid_ratio"}:
                depth = disparity_to_depth(stats.disparity, baseline, focal_length)
                depth_m = depth if depth > 0 else None
            else:
                depth_m = None
            depths.append(
                dataclasses.replace(stats, depth_m=depth_m, valid=stats.valid and depth_m is not None)
            )
        return depths

    def object_disparity_map(
        self,
        result: StereoMatchResult,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Constant object-level disparity inside ``mask`` for polar warping.

        Uses the same MAD-rejected median as :meth:`instance_stats`, so the
        warp and the reported depth are always consistent.
        """
        filled = np.zeros(result.disparity.shape, dtype=np.float32)
        mask_bool = np.asarray(mask > 0)
        if not mask_bool.any():
            return filled
        valid_in = mask_bool & (result.valid > 0)
        if not valid_in.any():
            return filled
        disparity = robust_disparity(result.disparity[valid_in])
        if disparity <= 0:
            return filled
        filled[mask_bool] = disparity
        return filled


def disparity_to_depth(
    disparity: float,
    baseline: float,
    focal_length: float,
) -> float:
    """Disparity (full-resolution pixels) to depth in meters.

    ``baseline`` is in meters and ``focal_length`` in full-resolution pixels.
    Returns -1.0 (invalid) for nonpositive or non-finite disparity.
    """
    if not np.isfinite(disparity) or disparity <= 0:
        return -1.0
    return float((baseline * focal_length) / disparity)


def compute_disparity_ncc(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    mask: np.ndarray,
    max_disp: int = 128,
    window_size: int = 9,
    min_ncc: float = 0.5,
) -> tuple[float, float, np.ndarray | None]:
    """Deprecated compatibility wrapper around :class:`StereoMatcher`.

    ``min_ncc`` is accepted for signature compatibility but ignored: SGBM has
    no per-window NCC threshold. Runs one dense SGBM computation and returns
    ``(robust_disparity, valid_ratio_in_mask, full_disparity_map)``.
    """
    del min_ncc
    matcher = StereoMatcher(
        StereoMatcherConfig(max_disparity=max_disp, block_size=window_size)
    )
    result = matcher.compute(left_gray, right_gray)
    if not np.any(np.asarray(mask) > 0):
        return 0.0, 0.0, None
    stats = matcher.instance_stats(result, mask)
    return stats.disparity, stats.valid_ratio, result.disparity


def compute_instance_depths(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    masks: list[np.ndarray],
    baseline: float,
    focal_length: float,
    max_disp: int = 128,
    window_size: int = 9,
    min_ncc: float = 0.5,
) -> list[dict]:
    """Deprecated compatibility wrapper; one dense SGBM pass for all masks."""
    del min_ncc
    matcher = StereoMatcher(
        StereoMatcherConfig(max_disparity=max_disp, block_size=window_size)
    )
    result = matcher.compute(left_bgr, right_bgr)
    depths = matcher.instance_depths(result, masks, baseline, focal_length)
    return [
        {
            "instance_id": d.instance_id,
            "disparity": round(d.disparity, 2),
            "depth": round(d.depth_m, 3) if d.depth_m is not None else -1.0,
            "confidence": round(d.valid_ratio, 3),
        }
        for d in depths
    ]
