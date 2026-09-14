"""Runtime online stereo rectification.

Loads a stereo calibration NPZ (keys ``K1, D1, K2, D2, R, T, image_size``),
computes ``cv2.stereoRectify`` and both undistort/rectify remap pairs exactly
once during construction, and rectifies synchronized frame pairs on demand.

Rotation-matrix convention
--------------------------
``cv2.stereoRectify`` expects ``R`` to rotate points from the left camera
frame into the right camera frame (OpenCV convention). Calibration exported
from MATLAB stores the transpose of that matrix, so ``r_convention="matlab"``
(default) applies ``R_cv = R.T``; ``"opencv"`` uses ``R`` as-is.

This module intentionally does not import from ``scripts/``: the runtime
engine must not depend on dataset-batch tooling.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.stereo_matching import to_gray_u8

R_CONVENTIONS = ("matlab", "opencv")
REQUIRED_KEYS = ("K1", "D1", "K2", "D2", "R", "T", "image_size")


def _matrix3(name: str, value: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(
            f"{path.name}: {name} must be a 3x3 matrix, got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name}: {name} contains non-finite values")
    return array


class StereoRectifier:
    """Precomputed stereo rectifier for online frame pairs.

    Args:
        calibration_path: path to a ``.npz`` with ``K1, D1, K2, D2, R, T,
            image_size``. ``image_size`` must be ``[width, height]``.
        r_convention: convention of the stored ``R`` matrix; ``"matlab"``
            applies ``R_cv = R.T``, ``"opencv"`` uses ``R`` as-is.
        alpha: ``cv2.stereoRectify`` free-scaling parameter.

    Raises:
        FileNotFoundError: if the calibration file does not exist.
        ValueError: on missing keys, invalid shapes, an unknown convention,
            or a degenerate translation vector.
    """

    def __init__(
        self,
        calibration_path: str | Path,
        r_convention: str = "matlab",
        alpha: float = 0.0,
    ) -> None:
        path = Path(calibration_path)
        if r_convention not in R_CONVENTIONS:
            raise ValueError(
                f"Unknown R convention {r_convention!r}; expected one of {R_CONVENTIONS}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        data = np.load(str(path))
        missing = sorted(set(REQUIRED_KEYS) - set(data.files))
        if missing:
            raise ValueError(
                f"Calibration file missing keys: {', '.join(missing)}"
            )

        k1 = _matrix3("K1", data["K1"], path)
        k2 = _matrix3("K2", data["K2"], path)
        r_raw = _matrix3("R", data["R"], path)
        d1 = np.asarray(data["D1"], dtype=np.float64).reshape(-1)
        d2 = np.asarray(data["D2"], dtype=np.float64).reshape(-1)
        if d1.size < 4 or d2.size < 4:
            raise ValueError(
                f"{path.name}: D1/D2 must contain at least 4 distortion coefficients"
            )
        for name, dist in (("D1", d1), ("D2", d2)):
            if not np.isfinite(dist).all():
                raise ValueError(
                    f"{path.name}: {name} contains non-finite distortion coefficients"
                )
        t_arr = np.asarray(data["T"], dtype=np.float64).reshape(-1)
        if t_arr.size != 3:
            raise ValueError(
                f"{path.name}: T must contain exactly 3 translation values, "
                f"got {t_arr.size}"
            )
        t = t_arr.reshape(3, 1)
        if not np.isfinite(t).all() or np.allclose(t, 0.0):
            raise ValueError(
                f"{path.name}: T must be a non-zero translation vector"
            )

        image_size_arr = np.asarray(data["image_size"]).reshape(-1)
        if image_size_arr.size != 2:
            raise ValueError(
                f"{path.name}: image_size must contain [width, height]"
            )
        width, height = int(image_size_arr[0]), int(image_size_arr[1])
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{path.name}: image_size must be positive, got [{width}, {height}]"
            )

        self.calibration_path = path
        self.r_convention = r_convention
        self.alpha = float(alpha)
        self.image_size = (width, height)
        self.r_cv = np.ascontiguousarray(
            r_raw.T if r_convention == "matlab" else r_raw
        )

        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            k1,
            d1,
            k2,
            d2,
            self.image_size,
            self.r_cv,
            t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=self.alpha,
        )
        self.r1, self.r2, self.p1, self.p2, self.q = r1, r2, p1, p2, q
        self._map1x, self._map1y = cv2.initUndistortRectifyMap(
            k1, d1, r1, p1, self.image_size, cv2.CV_32FC1
        )
        self._map2x, self._map2y = cv2.initUndistortRectifyMap(
            k2, d2, r2, p2, self.image_size, cv2.CV_32FC1
        )

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rectify one synchronized pair into aligned grayscale images.

        Accepts ``(H, W)``, ``(H, W, 1)``, 3- or 4-channel uint8 inputs via
        the shared :func:`to_gray_u8` boundary. Both images must match the
        calibration ``image_size``.

        Returns:
            ``(left_rectified, right_rectified)`` — contiguous 2-D uint8
            arrays of shape ``(height, width)``.
        """
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        expected = (self.image_size[1], self.image_size[0])
        for name, image in (("left", left_gray), ("right", right_gray)):
            if image.shape != expected:
                raise ValueError(
                    f"{name} image shape {image.shape} does not match "
                    f"calibration image_size (width, height)={self.image_size}"
                )
        left_out = cv2.remap(
            left_gray, self._map1x, self._map1y, cv2.INTER_LINEAR
        )
        right_out = cv2.remap(
            right_gray, self._map2x, self._map2y, cv2.INTER_LINEAR
        )
        return np.ascontiguousarray(left_out), np.ascontiguousarray(right_out)
