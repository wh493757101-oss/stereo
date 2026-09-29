"""GT-only geometry for the evaluation paths.

The V5 generation keeps two geometries strictly apart, and so must any
evaluation that wants to reproduce it:

- **band bbox** — the raw polygon extrema (``min``/``max`` of the float
  pixel coordinates), used for the band-restricted SGBM pass;
- **crop window** — ``make_crop_window``'s floor/ceil expansion by the
  padding, clipped to the image, used for the classification crop.

Deriving both from the rasterised mask's bounding box (what the first 4B
evaluation did) disagrees with V5 for 15 of 368 samples, because the
raster bounding box is an integer hull of the drawn polygon while the crop
window floors the float bounds.

This module is shared by the evaluation entry; it never touches the
predicted path, the Model A detections or the online pipeline behaviour.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from core.material_fusion_dataset import CLASS_NAMES


@dataclasses.dataclass(frozen=True)
class GtInstanceGeometry:
    """One annotated instance's geometry, in the generation's terms."""

    object_index: int
    class_id: int
    class_name: str | None
    polygon: tuple[tuple[float, float], ...]
    band_bbox: tuple[float, float, float, float]
    crop_window: tuple[int, int, int, int]
    mask: np.ndarray

    @property
    def polygon_extrema(self) -> tuple[float, float, float, float]:
        return self.band_bbox


def build_gt_geometry(
    label_path: str | Path,
    width: int,
    height: int,
    *,
    padding: int = 10,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> list[GtInstanceGeometry]:
    """Every annotated instance of one frame, with both geometries.

    Reuses the generator's own helpers (``read_yolo_polygons``,
    ``build_object_mask``, ``make_crop_window``) so the evaluation never
    re-implements the geometry contract.
    """
    from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons
    from scripts.prepare_cls_fusion_dataset import build_object_mask

    polygons = read_yolo_polygons(Path(label_path), width, height)
    geometry: list[GtInstanceGeometry] = []
    for index, annotation in enumerate(polygons):
        polygon = annotation.polygon
        xs = [x for x, _ in polygon]
        ys = [y for _, y in polygon]
        window = make_crop_window(polygon, width, height, padding)
        geometry.append(
            GtInstanceGeometry(
                object_index=index,
                class_id=int(annotation.class_id),
                class_name=(
                    class_names[annotation.class_id]
                    if 0 <= annotation.class_id < len(class_names) else None
                ),
                polygon=tuple((float(x), float(y)) for x, y in polygon),
                band_bbox=(min(xs), min(ys), max(xs), max(ys)),
                crop_window=(window.x1, window.y1, window.x2, window.y2),
                mask=build_object_mask(polygon, width, height),
            )
        )
    return geometry


def gt_instances(geometry: list[GtInstanceGeometry]):
    """``Instance`` objects carrying the *band* bbox (generation contract).

    The crop window is deliberately not encoded in the bbox: the caller
    passes it separately as the GT-only crop override, so the predicted
    path keeps deriving its window from the detection bbox exactly as the
    online pipeline does.
    """
    from models.segmentation import Instance

    return [
        Instance(
            id=item.object_index,
            bbox=item.band_bbox,
            mask=item.mask,
            confidence=1.0,
            class_id=item.class_id,
            class_name=item.class_name,
        )
        for item in geometry
    ]


def gt_crop_windows(
    geometry: list[GtInstanceGeometry],
) -> dict[int, tuple[int, int, int, int]]:
    """The per-instance crop override handed to the engine's GT path."""
    return {item.object_index: item.crop_window for item in geometry}


__all__ = [
    "GtInstanceGeometry",
    "build_gt_geometry",
    "gt_crop_windows",
    "gt_instances",
]
