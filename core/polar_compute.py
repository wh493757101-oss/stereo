"""Polarization feature computation and image warping.

Computes the approximate polarization feature:
    P = |L - warp(R)| / (L + warp(R))
where L is the left image (0-degree polarization) and R is the right image
(90-degree polarization), warped to the left view using a disparity map.

The warp depends on a disparity map: right pixel at column (x - d) maps to
left pixel at column x on the same row y.
"""

import dataclasses

import cv2
import numpy as np

from core.stereo_matching import StereoMatcher, StereoMatcherConfig, to_gray_u8

POLAR_EPSILON = 1e-6


@dataclasses.dataclass(frozen=True)
class PolarFeatureResult:
    """Structured polarization differential result with explicit validity.

    ``signed_q``/``abs_q`` are 0 wherever ``valid_mask`` is False, so an
    invalid pixel is distinguishable from a true zero-polarization pixel
    (which has ``valid_mask`` True and ``abs_q`` 0).

    The three ratio fields are computed over ``object_mask`` pixels when a
    mask is supplied, otherwise over the whole image.
    """

    signed_q: np.ndarray  # float32 (H, W), in [-1, 1], 0 where invalid
    abs_q: np.ndarray  # float32 (H, W), in [0, 1], 0 where invalid
    valid_mask: np.ndarray  # bool (H, W)
    in_bounds_mask: np.ndarray  # bool (H, W)
    valid_ratio: float
    in_bounds_ratio: float
    brightness_valid_ratio: float


def warp_with_disparity(
    image: np.ndarray,
    disparity: np.ndarray,
) -> np.ndarray:
    """Warp an image using a per-pixel disparity map.

    For left-view pixel at row y, column x, the corresponding right pixel
    is at row y, column (x - d). So we sample the right image at
    (x - d, y) to produce the warped image aligned to the left view.

    Args:
        image: input image, grayscale or multi-channel, uint8 or float32.
        disparity: per-pixel disparity, same H/W as image, float32.

    Returns:
        warped image, same shape and dtype as input.
    """
    h, w = image.shape[:2]
    disp = np.clip(disparity, 0, w - 1).astype(np.float32)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    map_x = xx - disp
    map_y = yy

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def compute_polar_feature(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    disparity_map: np.ndarray,
    mask: np.ndarray | None = None,
    min_valid_sum: float = 1e-6,
) -> np.ndarray:
    """Compute the polarization feature map.

    P = |L - warp(R)| / (L + warp(R))

    This intentionally does not perform per-frame brightness normalization:
    the feature is a brightness-ratio measurement. ``min_valid_sum`` only
    suppresses pixels where the denominator is too dark to be numerically
    meaningful.

    Args:
        left_gray: left grayscale image (0-degree polarization).
        right_gray: right grayscale image (90-degree polarization).
        disparity_map: per-pixel disparity, float32.
        mask: optional binary mask, uint8. Non-mask regions are zeroed.
        min_valid_sum: minimum L + warped(R) needed for a valid ratio.

    Returns:
        polar: float32 array in [0, 1], same H/W as input.
    """
    left_f = to_gray_u8(left_gray).astype(np.float32)
    right_f = to_gray_u8(right_gray).astype(np.float32)

    warped_right = warp_with_disparity(right_f, disparity_map)

    denom = left_f + warped_right
    valid = denom >= min_valid_sum
    polar = np.zeros_like(left_f, dtype=np.float32)
    np.divide(
        np.abs(left_f - warped_right),
        denom,
        out=polar,
        where=valid,
    )

    if mask is not None:
        polar = polar * (mask > 0).astype(np.float32)

    return np.clip(polar, 0.0, 1.0)


def compute_polar_features(
    left: np.ndarray,
    right: np.ndarray,
    disparity: np.ndarray,
    object_mask: np.ndarray | None = None,
    disparity_valid: np.ndarray | None = None,
    right_gain: float = 1.0,
    min_intensity_sum: float = 10.0,
    epsilon: float = POLAR_EPSILON,
) -> PolarFeatureResult:
    """Compute the signed polarization differential with explicit validity.

    signed_q = (L - gain * warp(R)) / (L + gain * warp(R) + epsilon)
    abs_q    = |signed_q|

    A pixel is valid only when ALL of the following hold:
    - inside ``object_mask`` (when a mask is supplied);
    - the disparity is finite and strictly positive;
    - the right-view sample position (x - d) lies inside the image;
    - the pixel passes ``disparity_valid`` (e.g. the stereo matcher's
      left-right-consistency valid map), when supplied;
    - L + gain*warp(R) >= ``min_intensity_sum`` (bright enough to measure).

    Invalid pixels get signed_q == 0 and abs_q == 0; consumers must check
    ``valid_mask`` to distinguish them from genuine zero polarization.

    Args:
        left: left image (0-degree polarization channel).
        right: right image (90-degree polarization channel).
        disparity: per-pixel disparity, float32, same H/W as the images.
        object_mask: optional uint8/bool mask restricting the measurement.
        disparity_valid: optional per-pixel validity mask for the disparity
            (e.g. ``StereoMatchResult.valid``).
        right_gain: multiplicative gain applied to the warped right image
            before differencing (compensates camera exposure differences).
        min_intensity_sum: minimum L + gain*warp(R) for a usable ratio.
        epsilon: denominator guard (irrelevant above min_intensity_sum).

    Returns:
        PolarFeatureResult with maps at the input resolution.
    """
    left_f = to_gray_u8(left).astype(np.float32)
    right_f = to_gray_u8(right).astype(np.float32)
    if left_f.shape != right_f.shape:
        raise ValueError(
            f"left/right shape mismatch: {left_f.shape} vs {right_f.shape}"
        )
    disp = np.asarray(disparity, dtype=np.float32)
    if disp.shape != left_f.shape:
        raise ValueError(
            f"disparity shape {disp.shape} != image shape {left_f.shape}"
        )

    h, w = left_f.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    map_x = xx - disp
    # BORDER_CONSTANT yields 0 outside the image; those samples are excluded
    # by in_bounds below instead of saturating the ratio towards 1.
    warped_right = cv2.remap(
        right_f,
        map_x,
        yy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if right_gain != 1.0:
        warped_right = warped_right * np.float32(right_gain)

    denom = left_f + warped_right
    brightness_valid = denom >= float(min_intensity_sum)
    in_bounds = (map_x >= 0.0) & (map_x <= w - 1)
    disparity_ok = np.isfinite(disp) & (disp > 0)
    consistency = (
        np.ones((h, w), dtype=bool)
        if disparity_valid is None
        else np.asarray(disparity_valid) > 0
    )
    if consistency.shape != (h, w):
        raise ValueError(
            f"disparity_valid shape {consistency.shape} != image shape {(h, w)}"
        )
    target = (
        np.ones((h, w), dtype=bool)
        if object_mask is None
        else np.asarray(object_mask) > 0
    )
    if target.shape != (h, w):
        raise ValueError(
            f"object_mask shape {target.shape} != image shape {(h, w)}"
        )

    valid = target & disparity_ok & in_bounds & consistency & brightness_valid

    signed_q = np.zeros((h, w), dtype=np.float32)
    np.divide(
        left_f - warped_right,
        denom + np.float32(epsilon),
        out=signed_q,
        where=valid,
    )
    signed_q = np.clip(signed_q, -1.0, 1.0).astype(np.float32)
    abs_q = np.abs(signed_q)

    total = int(target.sum())
    if total == 0:
        valid_ratio = in_bounds_ratio = brightness_valid_ratio = 0.0
    else:
        valid_ratio = float((valid & target).sum()) / total
        in_bounds_ratio = float((in_bounds & target).sum()) / total
        brightness_valid_ratio = float((brightness_valid & target).sum()) / total

    return PolarFeatureResult(
        signed_q=signed_q,
        abs_q=abs_q,
        valid_mask=valid,
        in_bounds_mask=in_bounds,
        valid_ratio=valid_ratio,
        in_bounds_ratio=in_bounds_ratio,
        brightness_valid_ratio=brightness_valid_ratio,
    )


def fill_disparity_in_mask(
    disparity_map: np.ndarray,
    mask: np.ndarray,
    fill_value: float | None = None,
) -> np.ndarray:
    """Fill a mask region with a robust object-level disparity.

    For warping the whole instance, a sparse disparity map would leave most
    mask pixels at zero disparity. This helper fills the mask with the median
    valid disparity. StereoMatcher.object_disparity_map provides the same
    fill derived from the matcher's own robust statistic.
    """
    mask_bool = mask > 0
    filled = np.zeros_like(disparity_map, dtype=np.float32)
    if not np.any(mask_bool):
        return filled

    if fill_value is None:
        valid_values = disparity_map[mask_bool & (disparity_map > 0)]
        if valid_values.size == 0:
            return filled
        fill_value = float(np.median(valid_values))

    filled[mask_bool] = max(float(fill_value), 0.0)
    return filled


def build_polar_yolo_image(
    gray: np.ndarray,
    polar: np.ndarray,
    third_channel: str = "gray",
) -> np.ndarray:
    """Build a standard 3-channel YOLO image from gray + polarization.

    Channel layout defaults to [gray, polar, gray]. This keeps Ultralytics'
    normal image dataloader/export path while still exposing the polarization
    cue to the network.
    """
    gray_u8 = to_gray_u8(gray)
    polar_u8 = np.rint(np.clip(polar, 0.0, 1.0) * 255).astype(np.uint8)

    if third_channel == "gray":
        ch2 = gray_u8
    elif third_channel == "zero":
        ch2 = np.zeros_like(gray_u8)
    elif third_channel == "polar":
        ch2 = polar_u8
    else:
        raise ValueError(f"Unsupported third_channel: {third_channel}")

    return np.stack([gray_u8, polar_u8, ch2], axis=-1)


def compute_full_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disp: int = 128,
    window_size: int = 9,
) -> np.ndarray:
    """Full-image stereo matching (compatibility wrapper around StereoMatcher).

    Delegates to :class:`core.stereo_matching.StereoMatcher` (OpenCV SGBM with
    left-right consistency); no Python NCC search loops are executed.
    Invalid pixels (including zero disparity) are zero in the returned map.

    Args:
        left_gray: left grayscale image.
        right_gray: right grayscale image.
        max_disp: full-resolution maximum disparity in pixels.
        window_size: SGBM block size, must be odd.

    Returns:
        disparity_map: float32 array, same H/W as input, 0 where invalid.
    """
    matcher = StereoMatcher(
        StereoMatcherConfig(max_disparity=max_disp, block_size=window_size)
    )
    result = matcher.compute(left_gray, right_gray)
    return result.disparity
