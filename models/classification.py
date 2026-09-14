"""Ultralytics classification model wrapper (Model B stages).

Consumes a pseudo-RGB crop (H, W, 3 uint8 NumPy array) where channel
semantics are decided upstream: [gray, gray, gray] for B-gray and
[gray, polar, gray] for B-polar. The wrapper never touches segmentation
masks.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClassificationResult:
    """Immutable single-crop classification outcome.

    ``valid`` is False when the underlying model produced no probability
    vector (empty/failed result); ``top1_id``, ``top1_name`` and
    ``top1_conf`` are None in that case.
    """

    top1_id: int | None
    top1_name: str | None
    top1_conf: float | None
    probs_available: bool

    @property
    def valid(self) -> bool:
        return self.probs_available

    @classmethod
    def empty(cls) -> "ClassificationResult":
        return cls(top1_id=None, top1_name=None, top1_conf=None, probs_available=False)


def _resolve_class_name(names, class_id: int) -> str | None:
    """Map a class id through dict- or list-shaped Ultralytics names."""
    if names is None:
        return None
    if isinstance(names, dict):
        if class_id in names:
            return str(names[class_id])
        return names.get(str(class_id))
    if isinstance(names, (list, tuple)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return None


class ClassificationModel:
    """Wrapper around an Ultralytics classification checkpoint.

    Args:
        model_path: path to a .pt classification checkpoint.
        device: inference device, e.g. "cpu", "0", or "cuda".
        imgsz: optional inference image size forwarded to
            ``model.predict(imgsz=...)``; ``None`` (default) keeps the
            Ultralytics default and is compatible with existing callers.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        imgsz: int | None = None,
    ):
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.device = device
        self.imgsz = int(imgsz) if imgsz is not None else None
        self.model = YOLO(str(self.model_path))
        self._names = self.model.names

    def predict(self, image: np.ndarray) -> ClassificationResult:
        """Classify a single pseudo-RGB crop (H, W, 3) uint8."""
        if image is None or getattr(image, "ndim", 0) != 3:
            return ClassificationResult.empty()

        kwargs: dict = {}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        results = self.model.predict(
            source=image,
            device=self.device,
            verbose=False,
            **kwargs,
        )
        if not results:
            return ClassificationResult.empty()
        return self._parse_result(results[0])

    def predict_batch(self, images: list[np.ndarray]) -> list[ClassificationResult]:
        """Classify a batch of pseudo-RGB crops in one model call.

        Invalid entries (None or not (H, W, 3)) yield ``ClassificationResult.empty()``
        placeholders so output order always matches input order.
        """
        outputs: list[ClassificationResult | None] = []
        batch_indices: list[int] = []
        for index, image in enumerate(images):
            if image is None or getattr(image, "ndim", 0) != 3:
                outputs.append(ClassificationResult.empty())
            else:
                outputs.append(None)
                batch_indices.append(index)

        if batch_indices:
            kwargs: dict = {}
            if self.imgsz is not None:
                kwargs["imgsz"] = self.imgsz
            results = self.model.predict(
                source=[images[i] for i in batch_indices],
                device=self.device,
                verbose=False,
                **kwargs,
            )
            for slot, index in enumerate(batch_indices):
                if slot < len(results):
                    outputs[index] = self._parse_result(results[slot])
                else:
                    outputs[index] = ClassificationResult.empty()

        return [result if result is not None else ClassificationResult.empty() for result in outputs]

    def _parse_result(self, result) -> ClassificationResult:
        """Extract the top-1 outcome from one Ultralytics result object."""
        probs = getattr(result, "probs", None)
        if probs is None:
            return ClassificationResult.empty()

        top1 = getattr(probs, "top1", None)
        top1conf = getattr(probs, "top1conf", None)
        if top1 is None or top1conf is None:
            return ClassificationResult.empty()

        top1 = int(top1)
        conf = float(top1conf.item() if hasattr(top1conf, "item") else top1conf)
        # Stable fallback so downstream Instance.class_name is never None for
        # a valid result (e.g. unmapped class ids in the checkpoint names).
        name = _resolve_class_name(self._names, top1) or f"class_{top1}"

        return ClassificationResult(
            top1_id=top1,
            top1_name=name,
            top1_conf=conf,
            probs_available=True,
        )
