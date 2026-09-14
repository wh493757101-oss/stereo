"""YOLO instance segmentation model wrapper.

The production path uses standard 3-channel YOLO inputs. Polar-aided material
classification is encoded as a pseudo-RGB image: [gray, polar, gray].
Custom 1-channel/2-channel first-conv surgery remains available for later
experiments, but is no longer the default pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from core.polar_compute import build_polar_yolo_image


@dataclass
class Instance:
    """Single instance segmentation result."""
    id: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    confidence: float
    class_id: int
    class_name: str


def _modify_first_conv(model, in_channels: int) -> None:
    """Replace the first conv layer to accept a different number of input channels."""
    first_layer = model.model.model[0]
    old_conv = first_layer.conv
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
        elif in_channels == 2:
            new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
            new_conv.weight[:, 1] = 0.0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    first_layer.conv = new_conv


class SegmentationModel:
    """YOLO segmentation wrapper.

    Args:
        model_path: path to .pt or .engine file.
        in_channels: 3 for standard YOLO, or experimental custom channels.
        device: inference device.
        conf_threshold: confidence threshold.
        iou_threshold: NMS IoU threshold.
    """

    def __init__(
        self,
        model_path: str | Path,
        in_channels: int = 3,
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.in_channels = in_channels
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.model = YOLO(str(self.model_path))

        if in_channels != 3 and self.model_path.suffix == ".pt":
            _modify_first_conv(self.model, in_channels)

        self._class_names: dict[int, str] = {}

    @property
    def class_names(self) -> dict[int, str]:
        if not self._class_names and self.model is not None:
            self._class_names = self.model.names or {}
        return self._class_names

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Convert image to the expected number of channels."""
        if self.in_channels == 1:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
        elif self.in_channels == 2:
            raise ValueError("2-channel input requires gray + polar arrays; use predict_2ch()")
        return image

    def predict(self, image: np.ndarray, **kwargs) -> list[Instance]:
        """Run instance segmentation on a single image.

        Args:
            image: BGR image (H, W, 3) for 3ch, or grayscale (H, W) for 1ch.
        """
        image = self._prepare_input(image)

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            **kwargs,
        )

        return self._parse_results(results, image.shape[:2])

    def predict_2ch(self, gray: np.ndarray, polar: np.ndarray, **kwargs) -> list[Instance]:
        """Run inference with experimental 2-channel input.

        Args:
            gray: grayscale image (H, W), uint8.
            polar: polarization feature map (H, W), float32 in [0, 1].
        """
        polar_u8 = (np.clip(polar, 0, 1) * 255).astype(np.uint8)
        stacked = np.stack([gray, polar_u8], axis=-1)

        results = self.model.predict(
            source=stacked,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            **kwargs,
        )

        return self._parse_results(results, gray.shape[:2])

    def predict_polar(self, gray: np.ndarray, polar: np.ndarray, **kwargs) -> list[Instance]:
        """Run standard 3-channel polar-aided YOLO inference."""
        image = build_polar_yolo_image(gray, polar)

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            **kwargs,
        )

        return self._parse_results(results, gray.shape[:2])

    def _parse_results(self, results, orig_shape) -> list[Instance]:
        instances = []
        if not results or results[0].masks is None:
            return instances

        result = results[0]
        orig_h, orig_w = orig_shape

        for i, (mask_tensor, box_tensor, cls_tensor, conf_tensor) in enumerate(
            zip(
                result.masks.data,
                result.boxes.data,
                result.boxes.cls,
                result.boxes.conf,
            )
        ):
            mask_np = mask_tensor.cpu().numpy().astype(np.float32)
            if mask_np.shape != (orig_h, orig_w):
                mask_np = cv2.resize(mask_np, (orig_w, orig_h))
            mask_bin = (mask_np * 255).astype(np.uint8)

            x1, y1, x2, y2 = box_tensor[:4].cpu().numpy().astype(int).tolist()
            class_id = int(cls_tensor.item())
            class_name = self.class_names.get(class_id, f"class_{class_id}")

            instances.append(Instance(
                id=i,
                bbox=(x1, y1, x2, y2),
                mask=mask_bin,
                confidence=round(float(conf_tensor.item()), 3),
                class_id=class_id,
                class_name=class_name,
            ))

        instances.sort(key=lambda x: x.confidence, reverse=True)
        return instances

    def export_tensorrt(
        self,
        output_path: str | Path | None = None,
        imgsz: int = 640,
        half: bool = True,
    ) -> Path:
        output_path = Path(output_path) if output_path else self.model_path.with_suffix(".engine")
        self.model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=self.device,
        )
        return output_path
