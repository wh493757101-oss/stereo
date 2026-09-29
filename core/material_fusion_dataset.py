"""V4 fusion-dataset adapter for the CMX material-fusion experiment.

The adapter reuses the V4 preprocessing verbatim (``FusionClsDataset`` from
``models.polar_fusion``: INTER_AREA gray resize to ``imgsz``, ``/255``
scaling, INTER_LINEAR channel resize for ``[signed_q, abs_q, valid]``) and
only rearranges the tensors per experiment variant:

    gray             single-stream MiT-B0 on [G, G, G]
    gray_gray_valid  CMX, appearance [G, G, G], auxiliary [G, G, V]
    gray_valid       CMX, appearance [G, G, G], auxiliary [0, 0, V]
    gray_polar       CMX, appearance [G, G, G], auxiliary [signed_q, abs_q, V]

``G`` is the scaled [0, 1] grayscale of the same sample and ``V`` the V4
valid channel. No ImageNet mean/std normalization is applied anywhere: the
appearance branch and every polarization component keep the native V4
[0, 1]-scale values, and the quality vector is exposed for availability
routing/diagnostics only (never concatenated into classification features).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

PREPROCESSING_VERSION = "v4_band_224_area_linear_v1"

VARIANTS = ("gray", "gray_gray_valid", "gray_valid", "gray_polar")

# Fixed project class order (CLAUDE.md invariant 3).
CLASS_NAMES = (
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
)

AUX_CHANNEL_MEANING = {
    "gray": None,
    "gray_gray_valid": ("gray", "gray", "valid"),
    "gray_valid": ("zero", "zero", "valid"),
    "gray_polar": ("signed_q", "abs_q", "valid"),
}


def validate_variant(variant: str) -> str:
    """Return the variant name after checking it is one of the four."""
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown material-fusion variant {variant!r}; expected one of {VARIANTS}"
        )
    return variant


def variant_is_dual(variant: str) -> bool:
    """True for the CMX dual-stream variants (all except ``gray``)."""
    return validate_variant(variant) != "gray"


def variant_inputs(
    variant: str,
    gray_t: torch.Tensor,
    polar_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map V4-preprocessed tensors to ``(appearance, auxiliary)`` inputs.

    ``gray_t`` is the (3, H, W) replicated [0, 1] gray crop and ``polar_t``
    the (3, H, W) ``[signed_q, abs_q, valid]`` tensor, exactly as produced
    by the shared V4 preprocessing. The single-stream variant returns
    ``auxiliary=None``.
    """
    validate_variant(variant)
    if variant == "gray":
        return gray_t, None
    if variant == "gray_gray_valid":
        return gray_t, torch.stack((gray_t[0], gray_t[0], polar_t[2]), dim=0)
    if variant == "gray_valid":
        zeros = torch.zeros_like(gray_t[0])
        return gray_t, torch.stack((zeros, zeros, polar_t[2]), dim=0)
    # gray_polar: the V4 polar tensor is already [signed_q, abs_q, valid].
    return gray_t, polar_t


def instance_mask_tensor(
    instance_mask, imgsz: int, cv2
) -> torch.Tensor:
    """The offline instance-mask input tensor, shared by dataset and runtime.

    The annotated (or predicted) binary mask is resampled into the model
    input frame with the same INTER_AREA call the V4 channels get, so mask
    and inputs share one coordinate frame. cv2's area resampling is a
    float32 average and can overshoot 1.0 by one float32 ulp (1.0000002);
    the mask is a coverage fraction, so the rounding is clipped back into
    [0, 1] rather than being treated as a different value.
    """
    import numpy as np

    mask = cv2.resize(
        np.ascontiguousarray(instance_mask, dtype=np.float32),
        (imgsz, imgsz),
        interpolation=cv2.INTER_AREA,
    )
    mask = np.clip(mask, 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(mask)).float()


class MaterialFusionDataset(torch.utils.data.Dataset):
    """Torch dataset yielding variant inputs from the fusion npz samples.

    Dual variants yield ``(appearance, auxiliary, quality, class_id)``;
    the ``gray`` variant yields ``(appearance, quality, class_id)``.

    With ``with_instance_mask=True`` (V5 samples only) the instance mask is
    appended as a ``(imgsz, imgsz)`` float32 tensor in [0, 1]: the annotated
    region resampled into the model input frame with the same INTER_AREA
    call the V4 channels get, so mask and inputs share one coordinate frame.
    The six V4 tensors are produced by exactly the same code path as before,
    element for element.
    """

    def __init__(
        self,
        sample_paths: Sequence[Path],
        variant: str,
        imgsz: int = 224,
        with_instance_mask: bool = False,
    ):
        from models.polar_fusion import FusionClsDataset

        if int(imgsz) <= 0:
            raise ValueError(f"imgsz must be a positive integer, got {imgsz!r}")
        self.variant = validate_variant(variant)
        self.imgsz = int(imgsz)
        self.with_instance_mask = bool(with_instance_mask)
        self._base = FusionClsDataset(sample_paths, imgsz=self.imgsz)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int):
        if self.with_instance_mask:
            return self._getitem_mask(index)
        gray_t, polar_t, quality_t, class_id = self._base[index]
        rgb, aux = variant_inputs(self.variant, gray_t, polar_t)
        if aux is None:
            return rgb, quality_t, class_id
        return rgb, aux, quality_t, class_id

    def _getitem_mask(self, index: int):
        import cv2
        import numpy as np

        from core.material_mask_dataset import load_mask_sample
        from models.polar_fusion import _fusion_tensors

        sample = load_mask_sample(self._base.sample_paths[index])
        gray_t, polar_t, quality_t = _fusion_tensors(sample, self.imgsz, cv2)
        mask_t = instance_mask_tensor(sample.instance_mask, self.imgsz, cv2)
        rgb, aux = variant_inputs(self.variant, gray_t, polar_t)
        if aux is None:
            return rgb, quality_t, int(sample.class_id), mask_t
        return rgb, aux, quality_t, int(sample.class_id), mask_t


def unpack_batch(
    batch,
    is_dual: bool,
    with_instance_mask: bool,
) -> tuple:
    """Split one collated batch into ``(rgb, aux, quality, class_id, mask)``.

    ``mask`` is ``None`` when the loader does not carry one, so a call site
    never has to know which dataset variant produced the batch.
    """
    batch = list(batch)
    mask = batch.pop() if with_instance_mask else None
    if is_dual:
        rgb, aux, quality, class_id = batch
    else:
        rgb, quality, class_id = batch
        aux = None
    return rgb, aux, quality, class_id, mask


def admit_material_dataset(
    data_root,
    expected_fingerprint: str = "",
) -> dict:
    """Read-only admission gate for evaluation/verification.

    Reuses the existing dataset audit gate (``validate_dataset``: passed
    audit, manifest/summary/per-file digest re-verification, fingerprint)
    and additionally pins the canonical class order. Nothing is written:
    no audit file is regenerated or modified. Raises on any mismatch.
    """
    from pathlib import Path

    from models.polar_fusion import class_names_from_manifest
    from scripts.train_polar_fusion import validate_dataset

    data_root = Path(data_root)
    report = validate_dataset(data_root)  # audit_passed + digest re-verification
    fingerprint = report.get("audit_fingerprint", "")
    names = class_names_from_manifest(data_root)
    if list(names) != list(CLASS_NAMES):
        raise ValueError(
            f"dataset {data_root} class order {list(names)} does not match the "
            f"fixed project order {list(CLASS_NAMES)}"
        )
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError(
            f"dataset {data_root} fingerprint {fingerprint} does not match the "
            f"expected {expected_fingerprint}; refusing to evaluate on a "
            "different dataset state"
        )
    return {
        "data_root": str(data_root),
        "audit_fingerprint": fingerprint,
        "verified_files": report.get("integrity_verified_files"),
        "class_names": list(names),
        "splits": report["splits"],
    }
