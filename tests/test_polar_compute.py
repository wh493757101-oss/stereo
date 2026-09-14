import numpy as np
import pytest

from core.polar_compute import (
    build_polar_yolo_image,
    compute_full_disparity,
    compute_polar_feature,
    fill_disparity_in_mask,
)


def test_build_polar_yolo_image_keeps_gray_and_polar_channels():
    gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    polar = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)

    image = build_polar_yolo_image(gray, polar)

    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[..., 0], gray)
    np.testing.assert_array_equal(image[..., 1], np.array([[0, 128], [255, 64]], dtype=np.uint8))
    np.testing.assert_array_equal(image[..., 2], gray)


def test_fill_disparity_in_mask_uses_median_valid_disparity():
    sparse = np.zeros((4, 4), dtype=np.float32)
    sparse[1, 1] = 4.0
    sparse[1, 2] = 6.0
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    filled = fill_disparity_in_mask(sparse, mask)

    assert filled[0, 0] == 0.0
    assert np.all(filled[mask > 0] == 5.0)


def test_compute_polar_feature_uses_ratio_without_brightness_normalization():
    left = np.array([[100, 200]], dtype=np.uint8)
    right = np.array([[50, 100]], dtype=np.uint8)
    disparity = np.zeros_like(left, dtype=np.float32)

    polar = compute_polar_feature(left, right, disparity)

    np.testing.assert_allclose(polar, np.array([[1 / 3, 1 / 3]], dtype=np.float32), rtol=1e-5)


def test_compute_polar_feature_accepts_hw1_single_channel_inputs():
    # Ultralytics patches cv2.imread so grayscale reads return (H, W, 1)
    left = np.array([[[100], [200]]], dtype=np.uint8)
    right = np.array([[[50], [100]]], dtype=np.uint8)
    disparity = np.zeros((1, 2), dtype=np.float32)

    polar = compute_polar_feature(left, right, disparity)

    assert polar.shape == (1, 2)
    np.testing.assert_allclose(polar, np.array([[1 / 3, 1 / 3]], dtype=np.float32), rtol=1e-5)


def test_compute_polar_feature_suppresses_too_dark_pixels_only():
    left = np.array([[0, 50]], dtype=np.uint8)
    right = np.array([[0, 25]], dtype=np.uint8)
    disparity = np.zeros_like(left, dtype=np.float32)

    polar = compute_polar_feature(left, right, disparity, min_valid_sum=10.0)

    assert polar[0, 0] == 0.0
    np.testing.assert_allclose(polar[0, 1], 1 / 3, rtol=1e-5)


def _textured_pair(width=1024, height=512, disparity=128, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    base = cv2.resize(
        rng.integers(0, 256, size=(height // 4, width // 4), dtype=np.uint8),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    left = np.clip(base, 0, 255).astype(np.uint8)
    right = np.clip(
        cv2.resize(
            rng.integers(0, 256, size=(height // 4, width // 4), dtype=np.uint8),
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        ),
        0,
        255,
    ).astype(np.uint8)
    if disparity > 0:
        right[:, : width - disparity] = left[:, disparity:]
    return left, right


def test_compute_full_disparity_delegates_to_sgbm_matcher():
    left, right = _textured_pair(disparity=128)

    disp = compute_full_disparity(left, right, max_disp=768, window_size=7)

    assert disp.shape == left.shape
    assert disp.dtype == np.float32
    roi = disp[:, 144:1008]
    valid = roi[roi > 0]
    assert valid.size > 0
    assert np.median(valid) == pytest.approx(128, abs=2.0)


def test_compute_full_disparity_zero_disparity_pair_is_all_invalid():
    left, right = _textured_pair(disparity=0)

    disp = compute_full_disparity(left, right, max_disp=768, window_size=7)

    assert np.all(disp == 0.0)


def test_compute_full_disparity_keeps_window_size_signature():
    left, right = _textured_pair(disparity=64, seed=1)

    disp = compute_full_disparity(left, right, max_disp=384, window_size=5)

    valid = disp[disp > 0]
    assert np.median(valid) == pytest.approx(64, abs=2.0)
