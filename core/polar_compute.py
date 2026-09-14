"""Polarization feature computation and image warping.

Computes the approximate polarization feature:
    P = |L - warp(R)| / (L + warp(R))
where L is the left image (0-degree polarization) and R is the right image
(90-degree polarization), warped to the left view using a disparity map.

The warp depends on a disparity map: right pixel at column (x - d) maps to
left pixel at column x on the same row y.
"""

import cv2
import numpy as np

from core.stereo_matching import StereoMatcher, StereoMatcherConfig, to_gray_u8


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
