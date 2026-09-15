"""Shared I/O and definitions for the V3 polar fusion classification dataset.

Each sample is one ``.npz`` file holding only numeric arrays:

    gray      uint8    (h, w)     left grayscale crop
    signed_q  float32  (h, w)     signed polarization differential, 0 where invalid
    abs_q     float32  (h, w)     |signed_q|, 0 where invalid
    valid     uint8    (h, w)     1 where the polarization measurement is valid
    quality   float32  (4,)       per-instance quality vector (see
                                  QUALITY_VECTOR_KEYS)
    class_id  int64    ()         class index consistent with the source
                                  segmentation dataset

All string provenance (split, class name, capture group, source frame,
object index, validity ratios, failure reasons) lives in
``dataset_manifest.csv`` next to the samples, never inside the ``.npz``.

The quality vector is the contract between dataset generation, the fusion
model's gate input and inference-time quality measurement; the same
definition must be recorded in fusion checkpoints.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

NPZ_FIELDS = ("gray", "signed_q", "abs_q", "valid", "quality", "class_id")

# Order and meaning of the per-sample quality vector. ``mean_abs_q`` is the
# mean of abs_q over valid pixels inside the instance mask (0.0 when no
# pixel is valid).
QUALITY_VECTOR_KEYS = (
    "valid_ratio",
    "in_bounds_ratio",
    "brightness_valid_ratio",
    "mean_abs_q",
)
QUALITY_VECTOR_LENGTH = len(QUALITY_VECTOR_KEYS)


def dataset_fingerprint(data_root: str | Path) -> str:
    """Stable sha256 over the manifest and the audit report.

    Training runs record this so a checkpoint can be traced back to the
    exact dataset state it was trained on.
    """
    import hashlib

    data_root = Path(data_root)
    digest = hashlib.sha256()
    for name in ("dataset_manifest.csv", "dataset_audit.json"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((data_root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class FusionSampleArrays:
    """Validated in-memory form of one fusion ``.npz`` sample."""

    gray: np.ndarray
    signed_q: np.ndarray
    abs_q: np.ndarray
    valid: np.ndarray
    quality: np.ndarray
    class_id: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.gray.shape


def build_quality_vector(
    valid_ratio: float,
    in_bounds_ratio: float,
    brightness_valid_ratio: float,
    mean_abs_q: float,
) -> np.ndarray:
    """Assemble the (4,) float32 quality vector in canonical key order."""
    return np.asarray(
        [valid_ratio, in_bounds_ratio, brightness_valid_ratio, mean_abs_q],
        dtype=np.float32,
    )


def quality_vector_from_result(
    polar_result,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Quality vector for one instance mask from a PolarFeatureResult.

    ``polar_result`` must have been computed with ``object_mask`` covering
    this mask (ratios are measured inside ``mask``). Returns the vector and
    the component dict for manifest/audit reporting.
    """
    mask_bool = np.asarray(mask) > 0
    total = int(mask_bool.sum())
    if total == 0:
        components = {key: 0.0 for key in QUALITY_VECTOR_KEYS}
        return build_quality_vector(**components), components

    valid = polar_result.valid_mask & mask_bool
    valid_count = int(valid.sum())
    valid_ratio = valid_count / total
    if valid_count:
        mean_abs_q = float(polar_result.abs_q[valid].mean())
    else:
        mean_abs_q = 0.0
    components = {
        "valid_ratio": valid_ratio,
        "in_bounds_ratio": float((polar_result.in_bounds_mask & mask_bool).sum()) / total,
        "brightness_valid_ratio": float(
            (polar_result.brightness_valid_mask & mask_bool).sum()
        )
        / total,
        "mean_abs_q": mean_abs_q,
    }
    return build_quality_vector(**components), components


def save_fusion_sample(
    path: str | Path,
    gray: np.ndarray,
    signed_q: np.ndarray,
    abs_q: np.ndarray,
    valid: np.ndarray,
    quality: np.ndarray,
    class_id: int,
) -> Path:
    """Write one validated fusion ``.npz`` sample; returns the path."""
    path = Path(path)
    gray = np.asarray(gray, dtype=np.uint8)
    signed_q = np.asarray(signed_q, dtype=np.float32)
    abs_q = np.asarray(abs_q, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.uint8)
    quality = np.asarray(quality, dtype=np.float32)
    shape = gray.shape
    for name, arr in (
        ("signed_q", signed_q),
        ("abs_q", abs_q),
        ("valid", valid),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} != gray shape {shape}")
    if quality.shape != (QUALITY_VECTOR_LENGTH,):
        raise ValueError(
            f"quality must have shape ({QUALITY_VECTOR_LENGTH},), got {quality.shape}"
        )
    if not np.all(np.isfinite(quality)):
        raise ValueError("quality vector contains NaN/Inf")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gray=gray,
        signed_q=signed_q,
        abs_q=abs_q,
        valid=valid,
        quality=quality,
        class_id=np.asarray(int(class_id), dtype=np.int64),
    )
    return path


def load_fusion_sample(path: str | Path) -> FusionSampleArrays:
    """Load and validate one fusion ``.npz`` sample."""
    path = Path(path)
    with np.load(path) as data:
        missing = [field for field in NPZ_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
        gray = data["gray"]
        signed_q = data["signed_q"]
        abs_q = data["abs_q"]
        valid = data["valid"]
        quality = data["quality"]
        class_id = data["class_id"]

    if gray.dtype != np.uint8:
        raise ValueError(f"{path}: gray dtype {gray.dtype} != uint8")
    for name, arr, dtype in (
        ("signed_q", signed_q, np.float32),
        ("abs_q", abs_q, np.float32),
        ("quality", quality, np.float32),
    ):
        if arr.dtype != dtype:
            raise ValueError(f"{path}: {name} dtype {arr.dtype} != {dtype}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{path}: {name} contains NaN/Inf")
    if valid.dtype != np.uint8:
        raise ValueError(f"{path}: valid dtype {valid.dtype} != uint8")
    if class_id.dtype != np.int64 or class_id.shape != ():
        raise ValueError(f"{path}: class_id must be a scalar int64")
    if quality.shape != (QUALITY_VECTOR_LENGTH,):
        raise ValueError(
            f"{path}: quality shape {quality.shape} != ({QUALITY_VECTOR_LENGTH},)"
        )
    shape = gray.shape
    for name, arr in (("signed_q", signed_q), ("abs_q", abs_q), ("valid", valid)):
        if arr.shape != shape:
            raise ValueError(f"{path}: {name} shape {arr.shape} != gray shape {shape}")

    return FusionSampleArrays(
        gray=gray,
        signed_q=signed_q,
        abs_q=abs_q,
        valid=valid,
        quality=quality,
        class_id=int(class_id),
    )
