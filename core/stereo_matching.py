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


UPSAMPLE_LEGACY_ZERO_LINEAR_V1 = "legacy_zero_linear_v1"
UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1 = "valid_weighted_legacy_support_v1"
UPSAMPLE_VERSIONS = (
    UPSAMPLE_LEGACY_ZERO_LINEAR_V1,
    UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1,
)


def upsample_disparity_lowres(disp_small: np.ndarray,
                              valid_small: np.ndarray,
                              out_w: int, out_h: int, scale: float,
                              version: str) -> tuple[np.ndarray, np.ndarray]:
    """The final low-resolution -> full-resolution disparity upsampling.

    Two explicit versions; anything else is refused:

    - ``legacy_zero_linear_v1``: the original production path, kept bitwise
      identical — the zero-filled low-res disparity is INTER_LINEAR-
      upsampled, the valid mask is INTER_NEAREST-upsampled, and the
      disparity is kept only where the upsampled mask is valid.

    - ``valid_weighted_legacy_support_v1``: the low-resolution SUPPORT is
      unchanged (``Vn = resize_NEAREST(V) > 0``; the valid domain never
      extends past the nearest-valid positions), but the disparity at the
      kept positions is the valid-weight-normalized value::

          N = resize_LINEAR(where(V, d, 0));  W = resize_LINEAR(V.float32)
          Dstar = N / W / scale   (only where W > 0)
          Vnew = Vn AND (W > 0) AND isfinite(Dstar) AND (Dstar > 0)

      The numerator masks invalid taps explicitly (it never relies on the
      caller having already zeroed them), ``W = 0`` positions are never
      divided, and a non-finite or non-positive value AT a valid
      low-resolution tap is rejected outright. Invalid output positions
      are exactly 0. Valid-valid depth discontinuities keep the original
      interpolation behaviour (this version does not repair them).
    """
    if version == UPSAMPLE_LEGACY_ZERO_LINEAR_V1:
        disp_full = cv2.resize(
            disp_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR
        ) / scale
        valid_full = cv2.resize(
            valid_small.astype(np.uint8), (out_w, out_h),
            interpolation=cv2.INTER_NEAREST)
        disp_full = np.where(valid_full > 0, disp_full, 0.0).astype(np.float32)
        return disp_full, valid_full
    if version == UPSAMPLE_VALID_WEIGHTED_LEGACY_SUPPORT_V1:
        valid = valid_small > 0
        if valid.any():
            valid_values = np.asarray(disp_small)[valid]
            if not np.isfinite(valid_values).all():
                raise ValueError(
                    "non-finite value at a valid low-resolution disparity "
                    "tap; the weighted upsample refuses to propagate it")
            if (valid_values <= 0).any():
                raise ValueError(
                    "non-positive value at a valid low-resolution "
                    "disparity tap")
        numerator = cv2.resize(
            np.where(valid, disp_small, 0.0).astype(np.float32),
            (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        weights = cv2.resize(
            valid.astype(np.float32), (out_w, out_h),
            interpolation=cv2.INTER_LINEAR)
        nearest_valid = cv2.resize(
            valid_small.astype(np.uint8), (out_w, out_h),
            interpolation=cv2.INTER_NEAREST) > 0
        has_weight = weights > 0
        dstar = np.divide(
            numerator, weights, out=np.zeros_like(numerator),
            where=has_weight) / scale
        new_valid = (
            nearest_valid & has_weight & np.isfinite(dstar) & (dstar > 0))
        disp_full = np.where(new_valid, dstar, 0.0).astype(np.float32)
        return disp_full, new_valid.astype(np.uint8)
    raise ValueError(
        f"unknown disparity upsample version: {version!r}; expected one "
        f"of {list(UPSAMPLE_VERSIONS)}")


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
    upsample_version: str = UPSAMPLE_LEGACY_ZERO_LINEAR_V1

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
        if self.upsample_version not in UPSAMPLE_VERSIONS:
            raise ValueError(
                f"unknown upsample_version: {self.upsample_version!r}; "
                f"expected one of {list(UPSAMPLE_VERSIONS)}")


@dataclasses.dataclass(frozen=True)
class LowResCapture:
    """The low-resolution intermediates of one ``_match_pair`` call.

    A small internal capture interface so the two-arm upsample comparison
    can prove both arms matched the identical low-resolution core without
    running SGBM twice. Not a public result type."""

    disp_small: np.ndarray
    valid_small: np.ndarray


@dataclasses.dataclass(frozen=True)
class LowResCoreRecord:
    """One low-resolution match core plus the extent it was matched for.

    ``band is None`` marks a full-frame match; otherwise ``(y1, y2)`` is
    the inclusive-exclusive row range the core was matched on. The capture
    holds BOTH SGBM passes (left and right view) and the consistency and
    texture checks — it is the complete low-resolution matching result,
    before any upsampling."""

    capture: LowResCapture
    band: tuple[int, int] | None
    valid_ratio: float
    elapsed_s: float


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
        self.last_lowres_cores: list[LowResCapture] = []
        self.lowres_match_calls = 0

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

        records = self._match_cores_normalized(left_gray, right_gray, None)
        return self.assemble_from_cores(records, left_gray.shape)

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
            self.last_lowres_cores = []
            self.lowres_match_calls = 0
            return StereoMatchResult(
                disparity=np.zeros((h, w), dtype=np.float32),
                valid=np.zeros((h, w), dtype=np.uint8),
                elapsed_s=0.0,
                valid_ratio=0.0,
            )

        coverage = sum(band.height for band in clipped)
        if coverage >= full_image_threshold * h:
            return self.compute(left_gray, right_gray)

        records = self._match_cores_normalized(left_gray, right_gray, clipped)
        return self.assemble_from_cores(records, left_gray.shape)

    def _match_pair(
        self, left_gray: np.ndarray, right_gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compatibility wrapper: one core plus assembly, returning the
        full-resolution ``(disparity, valid, elapsed_s, valid_ratio)`` the
        pre-refactor ``_match_pair`` returned (bitwise identical)."""
        records = self._match_cores_normalized(left_gray, right_gray, None)
        result = self.assemble_from_cores(records, left_gray.shape)
        return result.disparity, result.valid, result.elapsed_s, \
            result.valid_ratio

    def match_cores(self, left: np.ndarray, right: np.ndarray,
                    bands: list[HorizontalBand] | None = None,
                    full_image_threshold: float = 0.9) -> list[LowResCoreRecord]:
        """Run the low-resolution match ONCE per band (or once for the full
        frame) and return the cores WITHOUT upsampling.

        Each core contains both SGBM passes (left and right view), the
        left-right consistency check and the texture filter — the complete
        low-resolution matching result. Feeding the same cores through
        :meth:`assemble_from_cores` on two matchers whose configs differ
        only in ``upsample_version`` guarantees the two arms share one
        matching pass."""
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        if left_gray.shape != right_gray.shape:
            raise ValueError(
                f"left/right shape mismatch: {left_gray.shape} vs {right_gray.shape}"
            )
        if left_gray.ndim != 2:
            raise ValueError(f"expected single-channel image, got shape {left_gray.shape}")
        if bands is not None and not 0.0 < full_image_threshold <= 1.0:
            raise ValueError("full_image_threshold must be in (0, 1]")
        return self._match_cores_normalized(left_gray, right_gray, bands,
                                            full_image_threshold)

    def assemble_from_cores(self, records: list[LowResCoreRecord],
                            out_shape: tuple[int, int]) -> StereoMatchResult:
        """Assemble a full-resolution :class:`StereoMatchResult` from core
        records, applying the production upsampling version and the
        unchanged min_valid_ratio policies (per band for band cores, whole
        frame for a full-frame core). No matching happens here."""
        h, w = out_shape
        cfg = self.config
        if not records:
            return StereoMatchResult(
                disparity=np.zeros((h, w), dtype=np.float32),
                valid=np.zeros((h, w), dtype=np.uint8),
                elapsed_s=0.0,
                valid_ratio=0.0,
            )
        elapsed = sum(record.elapsed_s for record in records)
        if records[0].band is None:
            record = records[0]
            disp_full, valid_full = upsample_disparity_lowres(
                record.capture.disp_small, record.capture.valid_small,
                w, h, cfg.scale, cfg.upsample_version)
            valid_ratio = record.valid_ratio
            if valid_ratio < cfg.min_valid_ratio:
                disp_full = np.zeros((h, w), dtype=np.float32)
                valid_full = np.zeros((h, w), dtype=np.uint8)
            return StereoMatchResult(disparity=disp_full, valid=valid_full,
                                     elapsed_s=elapsed,
                                     valid_ratio=valid_ratio)
        disparity = np.zeros((h, w), dtype=np.float32)
        valid = np.zeros((h, w), dtype=np.uint8)
        valid_pixels = 0
        for record in records:
            y1, y2 = record.band
            band_disp, band_valid = upsample_disparity_lowres(
                record.capture.disp_small, record.capture.valid_small,
                w, y2 - y1, cfg.scale, cfg.upsample_version)
            if record.valid_ratio < cfg.min_valid_ratio:
                continue
            disparity[y1:y2] = band_disp
            valid[y1:y2] = band_valid
            valid_pixels += int(band_valid.sum())
        return StereoMatchResult(
            disparity=disparity,
            valid=valid,
            elapsed_s=elapsed,
            valid_ratio=valid_pixels / float(h * w),
        )

    def _match_cores_normalized(self, left_gray: np.ndarray,
                                right_gray: np.ndarray,
                                bands: list[HorizontalBand] | None,
                                full_image_threshold: float = 0.9
                                ) -> list[LowResCoreRecord]:
        self.last_lowres_cores = []
        self.lowres_match_calls = 0
        if bands is None:
            capture, ratio, elapsed = self._match_core(left_gray, right_gray)
            return [LowResCoreRecord(capture=capture, band=None,
                                     valid_ratio=ratio, elapsed_s=elapsed)]
        # production coverage policy: near-full bands fall back to ONE
        # full-frame core (identical to compute_bands' historical branch)
        coverage = sum(band.height for band in bands)
        if coverage >= full_image_threshold * left_gray.shape[0]:
            capture, ratio, elapsed = self._match_core(left_gray, right_gray)
            return [LowResCoreRecord(capture=capture, band=None,
                                     valid_ratio=ratio, elapsed_s=elapsed)]
        records = []
        for band in bands:
            capture, ratio, elapsed = self._match_core(
                left_gray[band.y1:band.y2], right_gray[band.y1:band.y2])
            records.append(LowResCoreRecord(
                capture=capture, band=(int(band.y1), int(band.y2)),
                valid_ratio=ratio, elapsed_s=elapsed))
        return records

    def _match_core(
        self, left_gray: np.ndarray, right_gray: np.ndarray
    ) -> tuple[LowResCapture, float, float]:
        """One low-resolution match: downscale, CLAHE, both SGBM passes
        (left and right view), left-right consistency and texture checks.

        Returns ``(capture, valid_ratio, elapsed_s)`` at the low
        resolution, before any upsampling. ``lowres_match_calls`` counts
        these invocations; one call covers both underlying SGBM passes."""
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
            empty = LowResCapture(
                disp_small=np.zeros((sh, sw), np.float32),
                valid_small=np.zeros((sh, sw), np.uint8))
            self.last_lowres_cores.append(empty)
            self.lowres_match_calls += 1
            return empty, 0.0, 0.0

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
        # small internal capture interface: the two-arm upsample comparison
        # re-uses one SGBM result and must prove the low-resolution core is
        # shared; no behaviour change
        capture = LowResCapture(
            disp_small=disp_small.copy(), valid_small=valid_small.copy())
        self.last_lowres_cores.append(capture)
        self.lowres_match_calls += 1
        return capture, valid_ratio, elapsed

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
