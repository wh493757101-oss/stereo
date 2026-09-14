"""Tests for core.rectification (StereoRectifier)."""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.rectification import StereoRectifier


@pytest.fixture
def tmp_path() -> Path:
    """Isolated work dir under the system temp (pytest basetemp under tests/
    can be locked by another process on Windows)."""
    workdir = Path(tempfile.mkdtemp(prefix="rectification_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

WIDTH, HEIGHT = 64, 48
IMAGE_SIZE = (WIDTH, HEIGHT)

# Near-identity calibration: zero distortion, axis-aligned cameras, small
# horizontal baseline. stereoRectify then yields (approximately) identity
# rectification, so rectified output should match the input closely.
K = np.array(
    [[50.0, 0.0, WIDTH / 2.0], [0.0, 50.0, HEIGHT / 2.0], [0.0, 0.0, 1.0]]
)
DIST = np.zeros(5, dtype=np.float64)
T_X = np.array([0.05, 0.0, 0.0])


def write_calib(path, r=None, t=T_X, image_size=None, k1=K, k2=K, d1=DIST, d2=DIST):
    np.savez(
        str(path),
        K1=k1,
        D1=d1,
        K2=k2,
        D2=d2,
        R=np.eye(3) if r is None else r,
        T=np.asarray(t, dtype=np.float64).reshape(-1, 1),
        image_size=np.array(IMAGE_SIZE if image_size is None else image_size),
    )
    return path


def make_image(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8)


@pytest.fixture
def calib_path(tmp_path):
    return write_calib(tmp_path / "calib.npz")


class TestConstruction:
    def test_missing_keys_raise(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(str(path), K1=K, D1=DIST, K2=K, D2=DIST, R=np.eye(3), T=T_X)
        with pytest.raises(ValueError, match="missing keys"):
            StereoRectifier(path)

    def test_unknown_convention_raises(self, calib_path):
        with pytest.raises(ValueError, match="convention"):
            StereoRectifier(calib_path, r_convention="cuda")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            StereoRectifier(tmp_path / "nope.npz")

    def test_zero_translation_raises(self, tmp_path):
        path = write_calib(tmp_path / "zero_t.npz", t=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="T"):
            StereoRectifier(path)

    def test_wrong_length_translation_raises(self, tmp_path):
        path = write_calib(tmp_path / "short_t.npz", t=(0.05, 0.0))
        with pytest.raises(ValueError, match="T must contain exactly 3"):
            StereoRectifier(path)

    def test_nonfinite_distortion_raises(self, tmp_path):
        path = write_calib(tmp_path / "nan_d.npz", d1=np.array([0.0, np.nan, 0, 0, 0]))
        with pytest.raises(ValueError, match="D1.*non-finite"):
            StereoRectifier(path)

    def test_nonfinite_translation_raises(self, tmp_path):
        path = write_calib(tmp_path / "inf_t.npz", t=(np.inf, 0.0, 0.0))
        with pytest.raises(ValueError, match="T"):
            StereoRectifier(path)

    def test_nonpositive_image_size_raises(self, tmp_path):
        path = write_calib(tmp_path / "bad_size.npz", image_size=[0, 48])
        with pytest.raises(ValueError, match="image_size"):
            StereoRectifier(path)

    def test_stereo_rectify_called_exactly_once(self, calib_path, monkeypatch):
        calls = []
        original = cv2.stereoRectify

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(cv2, "stereoRectify", counting)
        StereoRectifier(calib_path)
        assert len(calls) == 1

    def test_matlab_transposes_r_into_opencv_convention(self, tmp_path):
        angle = np.deg2rad(7.0)
        r_opencv = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        matlab_path = write_calib(tmp_path / "matlab.npz", r=r_opencv.T)
        opencv_path = write_calib(tmp_path / "opencv.npz", r=r_opencv)

        matlab_rectifier = StereoRectifier(matlab_path, r_convention="matlab")
        opencv_rectifier = StereoRectifier(opencv_path, r_convention="opencv")

        np.testing.assert_allclose(matlab_rectifier.r_cv, r_opencv, atol=1e-12)
        np.testing.assert_allclose(
            matlab_rectifier._map1x, opencv_rectifier._map1x, atol=1e-9
        )
        np.testing.assert_allclose(
            matlab_rectifier._map2y, opencv_rectifier._map2y, atol=1e-9
        )

    def test_opencv_convention_uses_r_as_is(self, tmp_path):
        angle = np.deg2rad(7.0)
        r_opencv = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        path = write_calib(tmp_path / "opencv.npz", r=r_opencv)
        rectifier = StereoRectifier(path, r_convention="opencv")
        np.testing.assert_allclose(rectifier.r_cv, r_opencv, atol=1e-12)


class TestRectify:
    def test_returns_contiguous_2d_uint8(self, calib_path):
        rectifier = StereoRectifier(calib_path)
        left, right = rectifier.rectify(make_image(1), make_image(2))
        assert left.dtype == np.uint8 and right.dtype == np.uint8
        assert left.ndim == 2 and right.ndim == 2
        assert left.shape == (HEIGHT, WIDTH) and right.shape == (HEIGHT, WIDTH)
        assert left.flags["C_CONTIGUOUS"] and right.flags["C_CONTIGUOUS"]

    def test_near_identity_calibration_preserves_image(self, calib_path):
        rectifier = StereoRectifier(calib_path, alpha=-1.0)
        left = make_image(3)
        left_out, _ = rectifier.rectify(left, make_image(4))
        np.testing.assert_allclose(left_out.astype(np.float32), left, atol=2)

    def test_accepts_3ch_and_bgra_inputs(self, calib_path):
        rectifier = StereoRectifier(calib_path)
        gray = make_image(5)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        left_from_gray, _ = rectifier.rectify(gray, gray)
        left_from_bgr, _ = rectifier.rectify(bgr, bgr)
        np.testing.assert_array_equal(left_from_gray, left_from_bgr)

    def test_shape_mismatch_raises(self, calib_path):
        rectifier = StereoRectifier(calib_path)
        wrong = np.zeros((HEIGHT + 2, WIDTH), dtype=np.uint8)
        with pytest.raises(ValueError, match="image_size"):
            rectifier.rectify(wrong, make_image(6))

    def test_invalid_dimensions_raise(self, calib_path):
        rectifier = StereoRectifier(calib_path)
        with pytest.raises(ValueError, match="expected"):
            rectifier.rectify(
                np.zeros((2, 2, 2), dtype=np.uint8), make_image(7)
            )
