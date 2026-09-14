import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


CLASS_MAP = {
    "metal_submarine": 0,
    "plastic_submarine": 1,
    "plastic_fish": 2,
    "real_fish": 3,
}
BACKGROUND_ID = 255

def polygon_from_shape(shape):
    shape_type = shape.get("shape_type", "polygon")
    points = np.array(shape["points"], dtype=np.float32)

    if shape_type in ("polygon", "linestrip"):
        return np.round(points).astype(np.int32)

    if shape_type == "rectangle":
        (x1, y1), (x2, y2) = points[:2]
        return np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32,
        )

    if shape_type == "circle":
        (cx, cy), (px, py) = points[:2]
        radius = int(round(np.hypot(px - cx, py - cy)))
        return ("circle", (int(round(cx)), int(round(cy))), radius)

    raise ValueError(f"Unsupported shape_type: {shape_type}")


def draw_shape(mask, shape, class_id):
    parsed = polygon_from_shape(shape)
    if isinstance(parsed, tuple) and parsed[0] == "circle":
        _, center, radius = parsed
        cv2.circle(mask, center, radius, int(class_id), thickness=-1)
        return

    cv2.fillPoly(mask, [parsed], int(class_id))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_one_json(json_path: Path, output_dir: Path):
    data = load_json(json_path)
    image_height = int(data["imageHeight"])
    image_width = int(data["imageWidth"])
    mask = np.full((image_height, image_width), BACKGROUND_ID, dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = str(shape["label"]).strip()
        if label not in CLASS_MAP:
            supported = ", ".join(CLASS_MAP.keys())
            raise KeyError(
                f"Unknown label '{label}' in {json_path.name}. "
                f"Supported labels: {supported}"
            )
        draw_shape(mask, shape, CLASS_MAP[label])

    output_path = output_dir / f"{json_path.stem}.png"
    cv2.imwrite(str(output_path), mask)
    return output_path


def collect_json_files(json_dir: Path):
    return sorted(
        path
        for path in json_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )


def collect_image_stems(image_dir: Path):
    return {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    }


def convert_directory(
    name: str, json_dir: Path, output_dir: Path, image_dir: Optional[Path] = None
):
    if not json_dir.exists():
        raise RuntimeError(f"{name}: json directory not found: {json_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = collect_json_files(json_dir)
    if not json_files:
        print(f"{name}: no JSON files found in {json_dir}")
        return

    if image_dir is not None:
        if not image_dir.exists():
            raise RuntimeError(f"{name}: image directory not found: {image_dir}")
        image_stems = collect_image_stems(image_dir)
        missing_images = [
            path.stem for path in json_files if path.stem not in image_stems
        ]
        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise RuntimeError(
                f"{name}: JSON stems do not match images in {image_dir}. Missing images for: {preview}"
            )

    print(f"{name}: converting {len(json_files)} JSON files")
    for json_path in json_files:
        output_path = convert_one_json(json_path, output_dir)
        print(f"  Saved mask: {output_path}")

    print(f"{name}: completed -> {output_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations into single-channel class-id masks."
    )
    parser.add_argument(
        "--json-dir",
        "--json_dir",
        required=True,
        dest="json_dir",
        help="Directory containing LabelMe JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        required=True,
        dest="output_dir",
        help="Directory in which class-id PNG masks are written.",
    )
    parser.add_argument(
        "--image-dir",
        "--image_dir",
        dest="image_dir",
        default=None,
        help="Optional image directory used to verify matching filename stems.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    convert_directory(
        name="LabelMe",
        json_dir=Path(args.json_dir),
        output_dir=Path(args.output_dir),
        image_dir=Path(args.image_dir) if args.image_dir else None,
    )


if __name__ == "__main__":
    main()
