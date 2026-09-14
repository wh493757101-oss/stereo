import numpy as np
import pytest

from core.polar_compute import (
    PolarFeatureResult,
    build_polar_yolo_image,
    compute_full_disparity,
    compute_polar_feature,
    compute_polar_features,
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


def test_compute_polar_features_signed_and_abs_differential():
    right_full = np.zeros((1, 4), dtype=np.uint8)
    right_full[0, 0] = 60
    right_full[0, 1] = 100
    left_full = np.zeros((1, 4), dtype=np.uint8)
    left_full[0, 2] = 100
    left_full[0, 3] = 60
    disp = np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32)

    result = compute_polar_features(left_full, right_full, disp)

    assert isinstance(result, PolarFeatureResult)
    assert result.signed_q.shape == (1, 4)
    # Pixel 2: L=100, R_warp=60 -> positive; pixel 3: L=60, R_warp=100 -> negative
    assert result.signed_q[0, 2] == pytest.approx(40.0 / 160.0, abs=1e-5)
    assert result.signed_q[0, 3] == pytest.approx(-40.0 / 160.0, abs=1e-5)
    assert result.abs_q[0, 2] == pytest.approx(0.25, abs=1e-5)
    assert result.abs_q[0, 3] == pytest.approx(0.25, abs=1e-5)
    # Pixels 0/1 have zero intensity: brightness-invalid, differential must be 0
    assert result.signed_q[0, 0] == 0.0 and result.signed_q[0, 1] == 0.0
    assert not result.valid_mask[0, 0]
    assert not result.valid_mask[0, 1]


def test_compute_polar_features_out_of_bounds_is_invalid_not_saturated():
    # Left pixel at column 0 with disparity 2 samples right column -2: the
    # legacy warp produced 0 there, saturating |L-0|/(L+0) to 1. The new API
    # must mark the pixel invalid with a zero differential instead.
    left = np.full((1, 8), 100, dtype=np.uint8)
    right = np.full((1, 8), 100, dtype=np.uint8)
    disparity = np.full((1, 8), 2.0, dtype=np.float32)

    result = compute_polar_features(left, right, disparity)

    assert result.in_bounds_mask[0, 0] == False
    assert result.in_bounds_mask[0, 1] == False
    assert result.in_bounds_mask[0, 2] == True
    assert result.valid_mask[0, 0] == False
    assert result.signed_q[0, 0] == 0.0
    assert result.abs_q[0, 0] == 0.0  # not 1.0
    assert result.valid_mask[0, 2] == True
    assert result.signed_q[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_compute_polar_features_dark_pixels_invalid():
    left = np.array([[5, 200]], dtype=np.uint8)
    right = np.array([[5, 200]], dtype=np.uint8)
    disparity = np.array([[1.0, 1.0]], dtype=np.float32)

    result = compute_polar_features(
        left, right, disparity, min_intensity_sum=10.0
    )

    assert result.valid_mask[0, 0] == False
    assert result.abs_q[0, 0] == 0.0
    assert result.brightness_valid_ratio == pytest.approx(0.5)


def test_compute_polar_features_right_gain_compensates_exposure():
    left_img = np.array([[0, 100]], dtype=np.uint8)
    right_img = np.array([[50, 0]], dtype=np.uint8)
    disp = np.array([[1.0, 1.0]], dtype=np.float32)

    no_gain = compute_polar_features(left_img, right_img, disp)
    with_gain = compute_polar_features(left_img, right_img, disp, right_gain=2.0)

    assert no_gain.signed_q[0, 1] == pytest.approx(50.0 / 150.0, abs=1e-5)
    # gain=2 makes warped right equal 100 = left -> zero differential
    assert with_gain.signed_q[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_compute_polar_features_requires_positive_disparity():
    left = np.full((1, 4), 100, dtype=np.uint8)
    right = np.full((1, 4), 100, dtype=np.uint8)
    disparity = np.array([[0.0, -2.0, 2.0, np.nan]], dtype=np.float32)

    result = compute_polar_features(left, right, disparity)

    assert result.valid_mask[0, 0] == False  # zero disparity
    assert result.valid_mask[0, 1] == False  # negative disparity
    assert result.valid_mask[0, 2] == True
    assert result.valid_mask[0, 3] == False  # NaN disparity
    assert result.valid_ratio == pytest.approx(0.25)


def test_compute_polar_features_respects_disparity_validity_mask():
    left = np.full((1, 4), 100, dtype=np.uint8)
    right = np.full((1, 4), 100, dtype=np.uint8)
    disparity = np.full((1, 4), 2.0, dtype=np.float32)
    disparity_valid = np.array([[1, 0, 1, 1]], dtype=np.uint8)

    result = compute_polar_features(
        left, right, disparity, disparity_valid=disparity_valid
    )

    assert result.valid_mask[0, 1] == False
    assert result.valid_mask[0, 2] == True
    # Columns 0/1 are out of bounds (x-2 < 0) and column 1 fails the
    # disparity validity mask; only columns 2/3 remain valid.
    assert result.valid_ratio == pytest.approx(0.5)


def test_compute_polar_features_object_mask_scopes_ratios_and_pixels():
    left = np.full((2, 4), 100, dtype=np.uint8)
    right = np.full((2, 4), 60, dtype=np.uint8)
    disparity = np.full((2, 4), 2.0, dtype=np.float32)
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[1, 3] = 1  # single target pixel; columns 0/1 out of bounds anyway

    result = compute_polar_features(left, right, disparity, object_mask=mask)

    # Outside the mask everything is invalid even where the geometry is fine
    assert result.valid_mask[0, 2] == False
    assert result.valid_mask[1, 3] == True
    assert result.signed_q[1, 3] == pytest.approx(40.0 / 160.0, abs=1e-5)
    # Ratios are relative to the 1-pixel mask, not the 8-pixel image
    assert result.valid_ratio == pytest.approx(1.0)
    assert result.in_bounds_ratio == pytest.approx(1.0)
    assert result.brightness_valid_ratio == pytest.approx(1.0)


def test_compute_polar_features_empty_mask_gives_zero_ratios():
    left = np.full((2, 2), 100, dtype=np.uint8)
    right = np.full((2, 2), 100, dtype=np.uint8)
    disparity = np.full((2, 2), 1.0, dtype=np.float32)
    mask = np.zeros((2, 2), dtype=np.uint8)

    result = compute_polar_features(left, right, disparity, object_mask=mask)

    assert result.valid_ratio == 0.0
    assert result.in_bounds_ratio == 0.0
    assert result.brightness_valid_ratio == 0.0
    assert not result.valid_mask.any()


def test_compute_polar_features_true_zero_polarization_is_valid():
    left = np.full((1, 4), 100, dtype=np.uint8)
    right = np.full((1, 4), 100, dtype=np.uint8)
    disparity = np.full((1, 4), 2.0, dtype=np.float32)

    result = compute_polar_features(left, right, disparity)

    assert result.valid_mask[0, 2] == True
    assert result.signed_q[0, 2] == 0.0
    assert result.abs_q[0, 2] == 0.0  # valid zero, not the invalid sentinel


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
