"""Prepare a Labelme dataset for Ultralytics YOLO-seg training.

The input layout mirrors the rectified stereo dataset:

    annotations_root/<scene>/<objects>/<turbidity>/<stem>.json
    image_root/<scene>/<objects>/<turbidity>/left/<stem>.png
    image_root/<scene>/<objects>/<turbidity>/right/<stem>.png

The output is a complete YOLO-seg dataset with normalized polygon labels and
a train/val/test split that never places frames from the same capture group
in more than one split. Exact duplicate annotations (same class id and
converted polygon within one JSON file) are dropped; source JSON files are
never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

DEFAULT_CLASS_NAMES = (
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
)
BACKGROUND_ID = 255
SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLIT_NAMES = ("train", "val", "test")
TURBIDITY_PATTERN = re.compile(r"(\d+)\s*ntu", re.IGNORECASE)


@dataclass(frozen=True)
class PolygonAnnotation:
    """A single Labelme shape normalized to YOLO-seg coordinates."""

    class_id: int
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class LabeledImage:
    """One annotated left image and its matching stereo right image."""

    output_stem: str
    relative_json: Path
    group_name: str
    json_path: Path
    left_path: Path
    right_path: Path
    width: int
    height: int
    annotations: tuple[PolygonAnnotation, ...]
    duplicates_removed: int = 0


@dataclass(frozen=True)
class SplitAssignment:
    """A prepared image and the block used for split isolation."""

    item: LabeledImage
    split: str
    block_id: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert nested Labelme polygons into a YOLO-seg dataset.",
    )
    parser.add_argument(
        "--annotations",
        default="datasets/bz_JSON_v2",
        help="Root directory containing nested Labelme JSON files.",
    )
    parser.add_argument(
        "--image-root",
        default="datasets/Rectified_v2",
        help="Rectified stereo root containing left/ and right/ directories.",
    )
    parser.add_argument(
        "--output",
        default="datasets/underwater_seg_v2",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Ordered comma-separated class names; Labelme labels must match exactly.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of groups (group split) or blocks (block split) held out for test.",
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-by",
        choices=["block", "group"],
        default="group",
        help="Whole capture groups go to exactly one split by default; "
        "'block' keeps every scene class in train/val for small datasets.",
    )
    parser.add_argument(
        "--skip-copy-images",
        action="store_true",
        help="Write labels and manifests only; useful for an existing image tree.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save class-id masks with background=255 for visual inspection.",
    )
    parser.add_argument("--min-polygon-area", type=float, default=1.0)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove an existing non-empty output directory before writing.",
    )
    return parser.parse_args(argv)


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().lower())


def sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "item"


def make_output_stem(relative_json: Path) -> str:
    prefix = "_".join(sanitize_token(part) for part in relative_json.parts[:-1])
    stem = sanitize_token(relative_json.stem)
    return f"{prefix}_{stem}"


def finite_points(raw_points: list[list[float]]) -> np.ndarray:
    points = np.asarray(raw_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must be an array of [x, y] values")
    if not np.isfinite(points).all():
        raise ValueError("points contain non-finite coordinates")
    return points


def polygon_for_shape(
    shape: dict,
    width: int,
    height: int,
    min_area: float,
) -> tuple[tuple[float, float], ...]:
    shape_type = str(shape.get("shape_type", "polygon")).lower()
    points = finite_points(shape.get("points", []))

    if shape_type in {"polygon", "linestrip"}:
        polygon = points
    elif shape_type == "rectangle":
        if len(points) < 2:
            raise ValueError("rectangle requires two corner points")
        (x1, y1), (x2, y2) = points[:2]
        polygon = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    elif shape_type == "circle":
        if len(points) < 2:
            raise ValueError("circle requires center and edge points")
        (cx, cy), (px, py) = points[:2]
        radius = float(np.hypot(px - cx, py - cy))
        if radius <= 0:
            raise ValueError("circle radius must be positive")
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        polygon = np.stack(
            [cx + radius * np.cos(angles), cy + radius * np.sin(angles)],
            axis=1,
        ).astype(np.float32)
    else:
        raise ValueError(f"unsupported Labelme shape_type: {shape_type}")

    if len(polygon) < 3:
        raise ValueError("polygon requires at least three points")
    polygon[:, 0] = np.clip(polygon[:, 0], 0.0, float(width))
    polygon[:, 1] = np.clip(polygon[:, 1], 0.0, float(height))
    contour = polygon.astype(np.float32)
    if float(cv2.contourArea(contour)) < min_area:
        raise ValueError(f"polygon area is below {min_area} pixel(s)")
    return tuple((float(x), float(y)) for x, y in polygon)


def parse_labelme_shapes(
    data: dict,
    json_path: Path,
    class_to_id: dict[str, int],
    min_area: float,
) -> tuple[PolygonAnnotation, ...]:
    annotations: list[PolygonAnnotation] = []
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])

    for shape_index, shape in enumerate(data.get("shapes", [])):
        label = normalize_label(str(shape.get("label", "")))
        if label not in class_to_id:
            supported = ", ".join(sorted(class_to_id))
            raise ValueError(
                f"unknown label {label!r} in {json_path.name}; supported: {supported}"
            )
        try:
            polygon = polygon_for_shape(shape, width, height, min_area)
        except ValueError as exc:
            raise ValueError(f"{json_path.name}, shape {shape_index}: {exc}") from exc
        annotations.append(PolygonAnnotation(class_to_id[label], polygon))

    return tuple(annotations)


def remove_duplicate_annotations(
    annotations: Iterable[PolygonAnnotation],
) -> tuple[tuple[PolygonAnnotation, ...], int]:
    """Drop exact duplicates (same class id and converted polygon).

    Only annotations that compare fully equal after label normalization
    and polygon conversion are removed; the first occurrence keeps its
    position. Returns the deduplicated annotations and the number removed.
    """
    ordered = list(annotations)
    seen: set[PolygonAnnotation] = set()
    kept: list[PolygonAnnotation] = []
    for annotation in ordered:
        if annotation not in seen:
            seen.add(annotation)
            kept.append(annotation)
    return tuple(kept), len(ordered) - len(kept)


def find_stereo_images(group_dir: Path, stem: str) -> tuple[Path, Path]:
    left_dir = group_dir / "left"
    right_dir = group_dir / "right"
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise FileNotFoundError(f"expected stereo directories under {group_dir}")

    left_matches = [
        path
        for path in left_dir.iterdir()
        if path.is_file()
        and path.stem == stem
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    if len(left_matches) != 1:
        raise FileNotFoundError(f"expected one left image for {group_dir / stem}")

    left_path = left_matches[0]
    right_path = right_dir / left_path.name
    if not right_path.is_file():
        raise FileNotFoundError(f"missing right image: {right_path}")
    return left_path, right_path


def verify_image_size(path: Path, width: int, height: int) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    actual_height, actual_width = image.shape[:2]
    if actual_width != width or actual_height != height:
        raise ValueError(
            f"image size mismatch for {path}: JSON={width}x{height}, "
            f"actual={actual_width}x{actual_height}"
        )


def load_labeled_images(
    annotation_root: Path,
    image_root: Path,
    class_to_id: dict[str, int],
    min_area: float,
) -> list[LabeledImage]:
    json_paths = sorted(
        path for path in annotation_root.rglob("*.json") if path.is_file()
    )
    if not json_paths:
        raise FileNotFoundError(f"no Labelme JSON files under {annotation_root}")

    items: list[LabeledImage] = []
    used_stems: dict[str, Path] = {}

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        relative_json = json_path.relative_to(annotation_root)
        group_dir = image_root.joinpath(*relative_json.parts[:-1])
        left_path, right_path = find_stereo_images(group_dir, json_path.stem)

        output_stem = make_output_stem(relative_json)
        collision_key = output_stem.lower()
        if collision_key in used_stems:
            raise ValueError(
                f"duplicate output stem {output_stem!r} for "
                f"{relative_json.as_posix()} and "
                f"{used_stems[collision_key].relative_to(annotation_root).as_posix()}"
            )
        used_stems[collision_key] = json_path

        width = int(data["imageWidth"])
        height = int(data["imageHeight"])
        verify_image_size(left_path, width, height)
        annotations, duplicates_removed = remove_duplicate_annotations(
            parse_labelme_shapes(
                data=data,
                json_path=json_path,
                class_to_id=class_to_id,
                min_area=min_area,
            )
        )
        items.append(
            LabeledImage(
                output_stem=output_stem,
                relative_json=relative_json,
                group_name=relative_json.parent.as_posix(),
                json_path=json_path,
                left_path=left_path,
                right_path=right_path,
                width=width,
                height=height,
                annotations=annotations,
                duplicates_removed=duplicates_removed,
            )
        )

    return items


def turbidity_token(group_name: str) -> str:
    """Extract a turbidity bucket token (e.g. 'turbidity_20') from a group name."""
    match = TURBIDITY_PATTERN.search(group_name)
    return f"turbidity_{match.group(1)}" if match else "turbidity_unknown"


def group_feature_vector(
    group_name: str, annotations: Iterable[PolygonAnnotation]
) -> dict[str, float]:
    """Class counts plus a turbidity one-hot, used for split stratification."""
    features = {f"class_{class_id}": 0.0 for class_id in range(4)}
    for annotation in annotations:
        features[f"class_{annotation.class_id}"] = features.get(
            f"class_{annotation.class_id}", 0.0
        ) + 1.0
    features[turbidity_token(group_name)] = 1.0
    return features


def group_split_counts(
    n_groups: int, val_ratio: float, test_ratio: float
) -> dict[str, int]:
    """Deterministic whole-group counts (round half up); train takes the remainder."""
    n_val = int(math.floor(n_groups * val_ratio + 0.5)) if val_ratio > 0 else 0
    n_test = int(math.floor(n_groups * test_ratio + 0.5)) if test_ratio > 0 else 0
    n_val = min(n_val, n_groups)
    n_test = min(n_test, n_groups - n_val)
    return {"train": n_groups - n_val - n_test, "val": n_val, "test": n_test}


def split_by_group(
    items: list[LabeledImage],
    val_ratio: float,
    test_ratio: float,
) -> list[SplitAssignment]:
    """Assign whole capture groups to train/val/test with greedy stratification.

    Groups are visited in sorted-name order and each is placed in the split
    whose remaining feature deficit (class counts + turbidity) best matches
    the group's profile. No group ever spans two splits.
    """
    grouped: dict[str, list[LabeledImage]] = defaultdict(list)
    for item in items:
        grouped[item.group_name].append(item)

    group_names = sorted(grouped)
    counts = group_split_counts(len(group_names), val_ratio, test_ratio)
    capacities = {split: counts[split] for split in SPLIT_NAMES}

    features = {
        name: group_feature_vector(
            name,
            [
                annotation
                for item in grouped[name]
                for annotation in item.annotations
            ],
        )
        for name in group_names
    }
    totals: dict[str, float] = defaultdict(float)
    for vector in features.values():
        for key, value in vector.items():
            totals[key] += value

    targets = {
        split: {
            key: totals[key] * capacities[split] / max(1, len(group_names))
            for key in totals
        }
        for split in SPLIT_NAMES
    }
    current: dict[str, dict[str, float]] = {split: defaultdict(float) for split in SPLIT_NAMES}
    filled = {split: 0 for split in SPLIT_NAMES}
    split_of: dict[str, str] = {}

    for name in group_names:
        candidates = [split for split in SPLIT_NAMES if filled[split] < capacities[split]]
        if not candidates:
            raise RuntimeError(
                f"split capacity exhausted for group {name!r}; "
                f"capacities={capacities}, groups={len(group_names)}"
            )

        def deficit(split: str) -> tuple[float, float]:
            vector = features[name]
            score = sum(
                value * (targets[split].get(key, 0.0) - current[split][key])
                for key, value in vector.items()
            )
            capacity = capacities[split]
            fill_fraction = filled[split] / capacity if capacity else 1.0
            return (score, -fill_fraction)

        chosen = max(candidates, key=deficit)
        split_of[name] = chosen
        filled[chosen] += 1
        for key, value in features[name].items():
            current[chosen][key] += value

    assignments: list[SplitAssignment] = []
    for item in items:
        assignments.append(
            SplitAssignment(item=item, split=split_of[item.group_name], block_id=0)
        )
    return assignments


def split_by_block(
    items: list[LabeledImage],
    val_ratio: float,
    block_size: int,
    seed: int,
    test_ratio: float = 0.0,
) -> list[SplitAssignment]:
    grouped: dict[str, list[LabeledImage]] = defaultdict(list)
    for item in items:
        grouped[item.group_name].append(item)

    assignments: list[SplitAssignment] = []
    for group_name in sorted(grouped):
        ordered = sorted(grouped[group_name], key=lambda item: item.output_stem)
        blocks = [
            ordered[start : start + block_size]
            for start in range(0, len(ordered), block_size)
        ]
        block_ids = list(range(len(blocks)))
        rng = random.Random(f"{seed}:{group_name}")
        rng.shuffle(block_ids)
        val_block_count = max(1, int(round(len(blocks) * val_ratio)))
        remaining = len(blocks) - val_block_count
        test_block_count = 0
        if test_ratio > 0 and remaining > 1:
            test_block_count = max(1, int(round(len(blocks) * test_ratio)))
            test_block_count = min(test_block_count, remaining - 1)
        val_blocks = set(block_ids[:val_block_count])
        test_blocks = set(
            block_ids[val_block_count : val_block_count + test_block_count]
        )

        for block_id, block in enumerate(blocks):
            if block_id in val_blocks:
                split = "val"
            elif block_id in test_blocks:
                split = "test"
            else:
                split = "train"
            assignments.extend(
                SplitAssignment(item=item, split=split, block_id=block_id)
                for item in block
            )
    return assignments


def prepare_output_root(output_root: Path, clean: bool) -> None:
    resolved = output_root.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not clean:
            raise RuntimeError(
                f"output directory is not empty: {resolved}; pass --clean to replace it"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def yolo_label_lines(
    annotations: Iterable[PolygonAnnotation],
    width: int,
    height: int,
) -> list[str]:
    lines: list[str] = []
    for annotation in annotations:
        coordinates = [str(annotation.class_id)]
        for x, y in annotation.polygon:
            coordinates.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
        lines.append(" ".join(coordinates))
    return lines


def write_labels(
    output_root: Path,
    assignments: list[SplitAssignment],
) -> None:
    grouped: dict[str, dict[str, LabeledImage]] = defaultdict(dict)
    for assignment in assignments:
        grouped[assignment.split][assignment.item.output_stem] = assignment.item

    for split, images in grouped.items():
        label_dir = output_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        for output_stem, item in images.items():
            lines = yolo_label_lines(item.annotations, item.width, item.height)
            (label_dir / f"{output_stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )


def write_masks(
    output_root: Path,
    assignments: list[SplitAssignment],
) -> None:
    grouped: dict[str, dict[str, LabeledImage]] = defaultdict(dict)
    for assignment in assignments:
        grouped[assignment.split][assignment.item.output_stem] = assignment.item

    for split, images in grouped.items():
        mask_dir = output_root / "masks" / split
        mask_dir.mkdir(parents=True, exist_ok=True)
        for output_stem, item in images.items():
            mask = np.full((item.height, item.width), BACKGROUND_ID, dtype=np.uint8)
            for annotation in item.annotations:
                points = np.asarray(annotation.polygon, dtype=np.int32)
                cv2.fillPoly(mask, [points], annotation.class_id)
            suffix = item.left_path.suffix.lower()
            cv2.imwrite(str(mask_dir / f"{output_stem}{suffix}"), mask)


def copy_images(
    output_root: Path,
    assignments: list[SplitAssignment],
) -> None:
    for assignment in assignments:
        split = assignment.split
        item = assignment.item
        image_dir = output_root / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            item.left_path,
            image_dir / f"{item.output_stem}{item.left_path.suffix.lower()}",
        )


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_split_assignment(
    output_root: Path,
    assignments: list[SplitAssignment],
) -> None:
    output_path = output_root / "split_assignment.csv"
    fieldnames = [
        "output_stem",
        "split",
        "group_name",
        "block_id",
        "source_json",
        "left_path",
        "right_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for assignment in assignments:
            item = assignment.item
            writer.writerow(
                {
                    "output_stem": item.output_stem,
                    "split": assignment.split,
                    "group_name": item.group_name,
                    "block_id": assignment.block_id,
                    "source_json": item.relative_json.as_posix(),
                    "left_path": relative_or_absolute(item.left_path),
                    "right_path": relative_or_absolute(item.right_path),
                }
            )


def write_pair_manifest(
    output_root: Path,
    assignments: list[SplitAssignment],
) -> None:
    output_path = output_root / "pair_manifest.csv"
    fieldnames = [
        "output_stem",
        "split",
        "group_name",
        "left_path",
        "right_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for assignment in assignments:
            item = assignment.item
            writer.writerow(
                {
                    "output_stem": item.output_stem,
                    "split": assignment.split,
                    "group_name": item.group_name,
                    "left_path": relative_or_absolute(item.left_path),
                    "right_path": relative_or_absolute(item.right_path),
                }
            )


def write_data_yaml(output_root: Path, class_names: list[str], splits: list[str]) -> None:
    names_block = "\n".join(
        f"  {class_id}: {name}" for class_id, name in enumerate(class_names)
    )
    split_lines = [f"{split}: images/{split}" for split in splits]
    content = (
        f"path: {output_root.resolve().as_posix()}\n"
        + "\n".join(split_lines)
        + f"\nnames:\n{names_block}\n"
    )
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def write_summary(
    output_root: Path,
    class_names: list[str],
    assignments: list[SplitAssignment],
    split_by: str,
    val_ratio: float,
    block_size: int,
    seed: int,
    test_ratio: float,
) -> None:
    present_splits = [split for split in SPLIT_NAMES if split in {
        assignment.split for assignment in assignments
    }]
    split_counts = Counter(assignment.split for assignment in assignments)
    class_counts: dict[str, Counter[str]] = {
        split: Counter(
            annotation.class_id
            for assignment in assignments
            if assignment.split == split
            for annotation in assignment.item.annotations
        )
        for split in present_splits
    }
    groups_per_split = {
        split: sorted({assignment.item.group_name for assignment in assignments
                       if assignment.split == split})
        for split in present_splits
    }
    if split_by == "group":
        assigned_groups = [group for groups in groups_per_split.values() for group in groups]
        if len(assigned_groups) != len(set(assigned_groups)):
            raise RuntimeError("split isolation violated: a group appears in multiple splits")
    removed_duplicates = sum(
        assignment.item.duplicates_removed for assignment in assignments
    )
    summary = {
        "class_names": class_names,
        "duplicate_annotations_removed": removed_duplicates,
        "splits": {
            split: {
                "images": split_counts[split],
                "groups": groups_per_split[split],
                "annotations": {
                    name: class_counts[split][class_id]
                    for class_id, name in enumerate(class_names)
                },
            }
            for split in present_splits
        },
        "split": {
            "strategy": split_by,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "block_size": block_size,
            "seed": seed,
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_ratio <= 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if not 0.0 <= args.test_ratio <= 1.0:
        raise ValueError("--test-ratio must be between 0 and 1")
    if args.val_ratio + args.test_ratio > 1.0:
        raise ValueError("--val-ratio + --test-ratio must not exceed 1")
    if args.block_size <= 0:
        raise ValueError("--block-size must be greater than zero")

    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("--classes must contain unique, non-empty names")
    class_to_id = {name: class_id for class_id, name in enumerate(class_names)}

    annotation_root = Path(args.annotations)
    image_root = Path(args.image_root)
    output_root = Path(args.output)
    items = load_labeled_images(
        annotation_root=annotation_root,
        image_root=image_root,
        class_to_id=class_to_id,
        min_area=args.min_polygon_area,
    )
    if args.split_by == "group":
        assignments = split_by_group(items, args.val_ratio, args.test_ratio)
    else:
        assignments = split_by_block(
            items, args.val_ratio, args.block_size, args.seed, args.test_ratio
        )

    present_splits = sorted({assignment.split for assignment in assignments})
    prepare_output_root(output_root, args.clean)
    for split in present_splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
    if not args.skip_copy_images:
        copy_images(output_root, assignments)
    write_labels(output_root, assignments)
    if args.save_masks:
        write_masks(output_root, assignments)
    write_split_assignment(output_root, assignments)
    write_pair_manifest(output_root, assignments)
    write_data_yaml(output_root, class_names, present_splits)
    write_summary(
        output_root=output_root,
        class_names=class_names,
        assignments=assignments,
        split_by=args.split_by,
        val_ratio=args.val_ratio,
        block_size=args.block_size,
        seed=args.seed,
        test_ratio=args.test_ratio,
    )

    split_counts = Counter(assignment.split for assignment in assignments)
    annotation_count = sum(
        len(assignment.item.annotations) for assignment in assignments
    )
    removed_duplicates = sum(item.duplicates_removed for item in items)
    print(f"Converted {len(items)} Labelme files ({annotation_count} polygons).")
    print(f"Removed {removed_duplicates} exact duplicate annotations.")
    print(
        "Split images: "
        + "/".join(f"{split}={split_counts[split]}" for split in present_splits)
    )
    print(f"YOLO dataset: {output_root.resolve()}")


if __name__ == "__main__":
    main()
