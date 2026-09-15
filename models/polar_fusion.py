"""Polar Fusion classification model.

Structure (fixed):

    gray crop -> YOLO26n-cls backbone -> gray_logits
    [signed_q, abs_q, valid] -> small CNN -> polar_delta
    quality vector -> MLP + sigmoid -> gate
    final_logits = gray_logits + gate * polar_delta

The gate is forced to 0 whenever the polarization measurement is
untrustworthy: valid_ratio == 0 in the quality vector, or an explicit
per-sample invalid flag (sync skew exceeded / stereo matching failed).
With gate == 0 the fusion output is exactly the gray-only logits, so the
model can never do worse than its gray backbone on invalid polar inputs.

Checkpoints carry the class names, input format, base model, quality
vector definition and a version tag so a saved model can always be
reconstructed and audited (see :func:`save_fusion_checkpoint`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from core.fusion_dataset import (
    QUALITY_VECTOR_KEYS,
    QUALITY_VECTOR_LENGTH,
    load_fusion_sample,
)

FUSION_VERSION = "polar_fusion_v1"
INPUT_FORMAT = (
    "gray: (3, imgsz, imgsz) uint8-replicated left gray crop / 255; "
    "polar: (3, imgsz, imgsz) [signed_q, abs_q, valid]; "
    "quality: (4,) float32 vector per core.fusion_dataset.QUALITY_VECTOR_KEYS"
)
DEFAULT_BASE_MODEL = "yolo26n-cls.pt"
VALID_RATIO_INDEX = QUALITY_VECTOR_KEYS.index("valid_ratio")


class PolarDeltaNet(nn.Module):
    """Small CNN mapping [signed_q, abs_q, valid] to per-class logit deltas."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, polar: torch.Tensor) -> torch.Tensor:
        return self.net(polar)


class PolarGateNet(nn.Module):
    """MLP + sigmoid over the quality vector producing one scalar gate."""

    def __init__(self, quality_dim: int = QUALITY_VECTOR_LENGTH, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(quality_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, quality: torch.Tensor) -> torch.Tensor:
        return self.net(quality)


class PolarFusionModel(nn.Module):
    """Gated fusion of a frozen-style gray classifier and a polar delta head.

    ``gray_backbone`` is any ``nn.Module`` mapping a normalized gray batch
    ``(B, 3, H, W)`` to class logits ``(B, num_classes)`` (the Ultralytics
    classification model's inner module in production).
    """

    def __init__(
        self,
        gray_backbone: nn.Module,
        num_classes: int,
        quality_dim: int = QUALITY_VECTOR_LENGTH,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.gray_backbone = gray_backbone
        self.num_classes = num_classes
        self.delta_net = PolarDeltaNet(num_classes)
        self.gate_net = PolarGateNet(quality_dim)

    def forward(
        self,
        gray: torch.Tensor,
        polar: torch.Tensor,
        quality: torch.Tensor,
        polar_invalid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute fused logits.

        Args:
            gray: (B, 3, H, W) normalized gray input for the backbone.
            polar: (B, 3, H, W) [signed_q, abs_q, valid] channels.
            quality: (B, quality_dim) float quality vector.
            polar_invalid: optional (B,) bool tensor forcing gate = 0
                (e.g. sync skew exceeded, stereo matching failed).

        Returns:
            dict with ``final_logits``, ``gray_logits``, ``polar_delta``,
            ``gate`` (all with batch dimension).
        """
        gray_logits = self.gray_backbone(gray)
        if gray_logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"gray backbone produced {gray_logits.shape[-1]} logits, "
                f"expected {self.num_classes}"
            )
        polar_delta = self.delta_net(polar)
        gate = self.gate_net(quality)  # (B, 1) in [0, 1]

        forced_zero = quality[:, VALID_RATIO_INDEX] <= 0
        if polar_invalid is not None:
            forced_zero = forced_zero | polar_invalid.bool()
        gate = gate * (~forced_zero).unsqueeze(1).to(gate.dtype)

        final_logits = gray_logits + gate * polar_delta
        return {
            "final_logits": final_logits,
            "gray_logits": gray_logits,
            "polar_delta": polar_delta,
            "gate": gate,
        }

    def trainable_parameters(self, include_gray: bool) -> list[nn.Parameter]:
        """Parameters for the requested training phase.

        ``include_gray=False``: freeze phase, only delta/gate heads train.
        ``include_gray=True``: joint finetune of everything.
        """
        head_params = list(self.delta_net.parameters()) + list(self.gate_net.parameters())
        if include_gray:
            return head_params + list(self.gray_backbone.parameters())
        return head_params

    def set_gray_frozen(self, frozen: bool) -> None:
        for param in self.gray_backbone.parameters():
            param.requires_grad = not frozen


class GrayBackboneAdapter(nn.Module):
    """Normalizes a real Ultralytics classification model for fusion use.

    Handles two real-world behaviors of the Ultralytics ClassificationModel:
    eval-mode ``forward`` returns a ``(softmax_probs, raw_logits)`` tuple
    while train mode returns a tensor; and the checkpoint's class order may
    differ from the V3 dataset order (the accepted gray weights use
    ``plastic_fish=1, plastic_submarine=2`` while V3 uses the seg dataset's
    ``plastic_submarine=1, plastic_fish=2``).

    ``perm`` maps output columns to the target order: column ``i`` of the
    adapter output is the backbone's column ``perm[i]``. ``None`` keeps the
    backbone order unchanged.
    """

    def __init__(self, module: nn.Module, perm: Sequence[int] | None = None):
        super().__init__()
        self.module = module
        self.register_buffer(
            "perm",
            torch.as_tensor(list(perm), dtype=torch.long) if perm is not None else None,
        )

    @staticmethod
    def _unwrap(output: Any) -> torch.Tensor:
        """Extract the raw logits tensor from an Ultralytics output."""
        if isinstance(output, (tuple, list)):
            if len(output) == 2:
                first, second = output
                # The row-stochastic element is the softmaxed probability;
                # the other one is the raw logits we must add deltas to.
                first_is_prob = (
                    isinstance(first, torch.Tensor)
                    and first.dim() == 2
                    and bool(
                        torch.allclose(
                            first.sum(-1),
                            torch.ones(first.shape[0], device=first.device),
                            atol=1e-3,
                        )
                    )
                )
                return second if first_is_prob else first
            return output[0]
        return output

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        logits = self._unwrap(self.module(gray))
        if self.perm is not None:
            logits = logits[:, self.perm]
        return logits


def build_gray_class_perm(
    gray_class_names: dict[int, str] | list[str],
    target_class_names: Sequence[str],
) -> list[int]:
    """Column permutation aligning gray backbone logits to the target order.

    ``perm[i]`` is the gray backbone column index whose class name equals
    ``target_class_names[i]``. Raises when any target class is missing or
    gray names are ambiguous.
    """
    names = (
        list(gray_class_names)
        if isinstance(gray_class_names, (list, tuple))
        else [str(gray_class_names[key]) for key in sorted(gray_class_names)]
    )
    if len(set(names)) != len(names):
        raise ValueError(f"gray class names are not unique: {names}")
    perm: list[int] = []
    for target in target_class_names:
        if target not in names:
            raise ValueError(
                f"target class {target!r} not present in gray backbone names {names}; "
                "class order mismatch must be resolved before training"
            )
        perm.append(names.index(target))
    return perm


@dataclasses.dataclass(frozen=True)
class FusionCheckpointMetadata:
    """Audit metadata for a fusion checkpoint.

    ``base_model`` records the architecture the gray branch was actually
    built from (derived from the loaded checkpoint's yaml, e.g.
    ``yolov8n-cls`` for the legacy weights), never the requested default.
    ``gray_weights`` is the path of the pretrained 4-class gray checkpoint
    loaded into the branch (empty string for a fresh-head build).
    """

    version: str
    class_names: tuple[str, ...]
    input_format: str
    base_model: str
    quality_vector_keys: tuple[str, ...]
    imgsz: int
    gray_weights: str = ""
    gray_class_names: tuple[str, ...] = ()
    head_replaced: bool = False
    class_permutation: tuple[int, ...] = ()


def fusion_metadata(
    class_names: Sequence[str],
    base_model: str = DEFAULT_BASE_MODEL,
    imgsz: int = 224,
    gray_weights: str = "",
    gray_class_names: Sequence[str] = (),
    head_replaced: bool = False,
    class_permutation: Sequence[int] = (),
) -> FusionCheckpointMetadata:
    return FusionCheckpointMetadata(
        version=FUSION_VERSION,
        class_names=tuple(str(name) for name in class_names),
        input_format=INPUT_FORMAT,
        base_model=str(base_model),
        quality_vector_keys=tuple(QUALITY_VECTOR_KEYS),
        imgsz=int(imgsz),
        gray_weights=str(gray_weights),
        gray_class_names=tuple(str(name) for name in gray_class_names),
        head_replaced=bool(head_replaced),
        class_permutation=tuple(int(i) for i in class_permutation),
    )


def save_fusion_checkpoint(
    path: str | Path,
    model: PolarFusionModel,
    metadata: FusionCheckpointMetadata,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save model weights plus the mandatory audit metadata."""
    path = Path(path)
    if len(metadata.class_names) != model.num_classes:
        raise ValueError(
            f"class_names ({len(metadata.class_names)}) != num_classes "
            f"({model.num_classes})"
        )
    payload = {
        "version": metadata.version,
        "class_names": list(metadata.class_names),
        "input_format": metadata.input_format,
        "base_model": metadata.base_model,
        "quality_vector_keys": list(metadata.quality_vector_keys),
        "imgsz": metadata.imgsz,
        "gray_weights": metadata.gray_weights,
        "gray_class_names": list(metadata.gray_class_names),
        "head_replaced": metadata.head_replaced,
        "class_permutation": list(metadata.class_permutation),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _metadata_from_payload(path: str | Path, payload: dict[str, Any]) -> FusionCheckpointMetadata:
    for field in ("version", "class_names", "input_format", "base_model",
                  "quality_vector_keys"):
        if field not in payload:
            raise ValueError(f"checkpoint {path} is missing field {field!r}")
    return FusionCheckpointMetadata(
        version=str(payload["version"]),
        class_names=tuple(str(n) for n in payload["class_names"]),
        input_format=str(payload["input_format"]),
        base_model=str(payload["base_model"]),
        quality_vector_keys=tuple(payload["quality_vector_keys"]),
        imgsz=int(payload.get("imgsz", 224)),
        gray_weights=str(payload.get("gray_weights", "")),
        gray_class_names=tuple(str(n) for n in payload.get("gray_class_names", ())),
        head_replaced=bool(payload.get("head_replaced", False)),
        class_permutation=tuple(int(i) for i in payload.get("class_permutation", ())),
    )


def read_fusion_metadata(path: str | Path) -> FusionCheckpointMetadata:
    """Read only the audit metadata from a checkpoint (no model rebuild)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _metadata_from_payload(path, payload)


def load_fusion_checkpoint(
    path: str | Path,
    gray_backbone: nn.Module,
) -> tuple[PolarFusionModel, FusionCheckpointMetadata, dict[str, Any]]:
    """Rebuild the fusion model from a checkpoint.

    ``gray_backbone`` must be a freshly initialized backbone matching the
    checkpoint architecture (its weights are overwritten by the checkpoint).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        raise ValueError(f"checkpoint {path} is missing field 'state_dict'")
    metadata = _metadata_from_payload(path, payload)
    model = PolarFusionModel(
        gray_backbone, num_classes=len(metadata.class_names),
        quality_dim=len(metadata.quality_vector_keys),
    )
    model.load_state_dict(payload["state_dict"])
    return model, metadata, payload


class FusionClsDataset(torch.utils.data.Dataset):
    """Torch dataset over the V3 fusion npz samples for one split.

    Yields ``(gray, polar, quality, class_id)`` where gray is the replicated
    3-channel crop resized to ``imgsz`` and normalized to [0, 1], polar is
    the stacked [signed_q, abs_q, valid] crop resized to ``imgsz``, and
    quality is the per-sample float32 vector.
    """

    def __init__(self, sample_paths: Sequence[Path], imgsz: int = 224):
        import cv2

        self.sample_paths = [Path(p) for p in sample_paths]
        self.imgsz = int(imgsz)
        self._cv2 = cv2

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int):
        sample = load_fusion_sample(self.sample_paths[index])
        size = (self.imgsz, self.imgsz)

        gray = self._cv2.resize(sample.gray, size, interpolation=self._cv2.INTER_AREA)
        gray_t = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)
        gray_t = gray_t.unsqueeze(0).expand(3, -1, -1).contiguous()

        polar = np.stack(
            [sample.signed_q, sample.abs_q, sample.valid.astype(np.float32)], axis=0
        )
        polar_resized = np.stack(
            [
                self._cv2.resize(polar[c], size, interpolation=self._cv2.INTER_LINEAR)
                for c in range(3)
            ],
            axis=0,
        )
        polar_t = torch.from_numpy(np.ascontiguousarray(polar_resized)).float()

        quality_t = torch.from_numpy(np.asarray(sample.quality, dtype=np.float32))
        return gray_t, polar_t, quality_t, int(sample.class_id)


def read_manifest_split(
    dataset_root: str | Path,
    split: str,
) -> tuple[list[Path], list[int], list[str]]:
    """Sample paths, class ids and class names for one split from the manifest."""
    import csv

    dataset_root = Path(dataset_root)
    with (dataset_root / "dataset_manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    split_rows = [row for row in rows if row["split"] == split]
    if not split_rows:
        raise ValueError(f"no {split!r} samples in {dataset_root / 'dataset_manifest.csv'}")
    paths = [dataset_root / row["npz_path"] for row in split_rows]
    class_ids = [int(row["class_id"]) for row in split_rows]
    class_names = [row["class_name"] for row in split_rows]
    return paths, class_ids, class_names


def class_names_from_manifest(dataset_root: str | Path) -> list[str]:
    """Ordered class names by first appearance of class_id in the manifest."""
    import csv

    with (Path(dataset_root) / "dataset_manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_id: dict[int, str] = {}
    for row in rows:
        by_id.setdefault(int(row["class_id"]), row["class_name"])
    return [by_id[key] for key in sorted(by_id)]
