"""Build paired gray-only and polar-aided classification datasets (Model B).

Consumes the leakage-free four-class v2 segmentation dataset (pair manifest
plus YOLO-seg labels) and emits two Ultralytics classification roots in one
run:

    datasets/underwater_cls_gray_v2   -> [gray, gray, gray] crops
    datasets/underwater_cls_polar_v2  -> [gray, polar, gray] crops

Both roots share identical relative class paths, so every gray crop has a
polar twin. Per stereo pair exactly one dense ``StereoMatcher.compute`` is
executed; each polygon becomes a full-image mask whose robust object
disparity (``instance_stats`` / ``object_disparity_map``) is used as the
constant disparity for the polar warp computed from the original left/right
intensities. Invalid stereo never drops a sample: both crops are still
written and the CSV manifest records validity, reason, valid ratio and
disparity so downstream training can filter without losing difficult
examples.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from core.polar_compute import build_polar_yolo_image, compute_polar_feature
from core.stereo_matching import StereoMatcher, StereoMatcherConfig, to_gray_u8
from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRow:
    output_stem: str
    split: str
    group_name: str
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class SampleRecord:
    sample_name: str
    split: str
    group_name: str
    class_id: int
    class_name: str
    source_frame: str
    object_index: int
    stereo_valid: bool
    stereo_reason: str
    valid_ratio: float
    disparity: float
    gray_path: str
    polar_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired gray/polar Model-B classification datasets.",
    )
    parser.add_argument(
        "--source",
        default="datasets/underwater_seg_v2",
        help="Source four-class YOLO-seg dataset root (pair manifest + labels).",
    )
    parser.add_argument(
        "--gray-output",
        default="datasets/underwater_cls_gray_v2",
        help="Output root for the [gray, gray, gray] classification dataset.",
    )
    parser.add_argument(
        "--polar-output",
        default="datasets/underwater_cls_polar_v2",
        help="Output root for the [gray, polar, gray] classification dataset.",
    )
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=10,
        help="Padding in pixels around each instance polygon.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace existing non-empty output directories.",
    )
    parser.add_argument(
        "--sgbm-mode",
        choices=["sgbm", "hh", "3way"],
        default="3way",
        help="OpenCV StereoSGBM dynamic-programming mode",
    )
    parser.add_argument(
        "--max-disp",
        type=int,
        default=768,
        help="Full-resolution maximum disparity in pixels",
    )
    parser.add_argument(
        "--block-size",
        "--window",
        dest="block_size",
        type=int,
        default=7,
        help="SGBM block size in full-resolution pixels (odd)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Matching scale relative to full resolution",
    )
    parser.add_argument(
        "--lr-check-threshold",
        type=float,
        default=2.0,
        help="Left-right consistency threshold in full-resolution pixels",
    )
    parser.add_argument(
        "--uniqueness-ratio",
        type=int,
        default=10,
        help="SGBM uniqueness ratio (percent)",
    )
    parser.add_argument(
        "--speckle-window",
        type=int,
        default=100,
        help="SGBM speckle window size",
    )
    parser.add_argument(
        "--speckle-range",
        type=int,
        default=32,
        help="SGBM speckle range",
    )
    parser.add_argument(
        "--texture-threshold",
        type=float,
        default=10.0,
        help="Minimum local horizontal-gradient texture for valid disparity",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.05,
        help="Minimum global valid-pixel ratio; below it the disparity map is discarded",
    )
    args = parser.parse_args()
    if args.crop_pad < 0:
        parser.error("--crop-pad must be non-negative")
    return args


def build_matcher(args: argparse.Namespace) -> StereoMatcher:
    return StereoMatcher(
        StereoMatcherConfig(
            matcher="sgbm",
            mode=args.sgbm_mode,
            max_disparity=args.max_disp,
            scale=args.scale,
            block_size=args.block_size,
            uniqueness_ratio=args.uniqueness_ratio,
            speckle_window=args.speckle_window,
            speckle_range=args.speckle_range,
            texture_threshold=args.texture_threshold,
            lr_check_threshold_px=args.lr_check_threshold,
            min_valid_ratio=args.min_valid_ratio,
        )
    )


def ensure_disjoint_roots(a: Path, b: Path, a_label: str, b_label: str) -> None:
    ra, rb = a.resolve(), b.resolve()
    if ra == rb or rb.is_relative_to(ra) or ra.is_relative_to(rb):
        raise ValueError(
            f"{a_label} ({ra}) and {b_label} ({rb}) overlap; "
            "output roots must be outside each other and outside the source"
        )


def prepare_output_root(output_root: Path, clean: bool) -> None:
    resolved = output_root.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not clean:
            raise RuntimeError(
                f"output directory is not empty: {resolved}; pass --clean to replace it"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def read_source_manifest(source_root: Path, workspace: Path) -> list[ManifestRow]:
    manifest_path = source_root / "pair_manifest.csv"
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"output_stem", "split", "group_name", "left_path", "right_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                f"{manifest_path} is missing required columns: {', '.join(missing)}"
            )
        for line_index, row in enumerate(reader, start=2):
            output_stem = str(row["output_stem"]).strip()
            split = str(row["split"]).strip()
            if not output_stem or split not in SPLIT_NAMES:
                raise ValueError(
                    f"invalid output_stem/split at {manifest_path}:{line_index}"
                )
            if output_stem in seen:
                raise ValueError(
                    f"duplicate output_stem {output_stem!r} in {manifest_path}"
                )
            seen.add(output_stem)
            rows.append(
                ManifestRow(
                    output_stem=output_stem,
                    split=split,
                    group_name=str(row["group_name"]).strip(),
                    left_path=_resolve_manifest_path(row["left_path"], workspace),
                    right_path=_resolve_manifest_path(row["right_path"], workspace),
                )
            )
    if not rows:
        raise ValueError(f"pair manifest is empty: {manifest_path}")
    return rows


def _resolve_manifest_path(value: str, workspace: Path) -> Path:
    path = Path(str(value).strip())
    return path if path.is_absolute() else workspace / path


def read_class_names(source_root: Path) -> list[str]:
    with (source_root / "data.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid class names in {source_root / 'data.yaml'}")
    return [str(name) for name in names]


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return to_gray_u8(image)


def build_object_mask(
    polygon: tuple[tuple[float, float], ...],
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def write_classification_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def build_cls_datasets(
    source_root: Path,
    gray_root: Path,
    polar_root: Path,
    matcher: StereoMatcher,
    crop_pad: int = 10,
    clean: bool = False,
    workspace: Path | None = None,
) -> list[SampleRecord]:
    """Build both classification roots; returns one record per written crop."""
    workspace = workspace if workspace is not None else Path.cwd()

    ensure_disjoint_roots(gray_root, polar_root, "gray output", "polar output")
    ensure_disjoint_roots(source_root, gray_root, "source root", "gray output")
    ensure_disjoint_roots(source_root, polar_root, "source root", "polar output")

    class_names = read_class_names(source_root)
    rows = read_source_manifest(source_root, workspace)
    prepare_output_root(gray_root, clean)
    prepare_output_root(polar_root, clean)
    for split in SPLIT_NAMES:
        for class_name in class_names:
            (gray_root / split / class_name).mkdir(parents=True, exist_ok=True)
            (polar_root / split / class_name).mkdir(parents=True, exist_ok=True)

    records: list[SampleRecord] = []
    for row in rows:
        label_path = source_root / "labels" / row.split / f"{row.output_stem}.txt"
        if not label_path.is_file():
            continue
        left_gray = load_gray(row.left_path)
        height, width = left_gray.shape[:2]
        polygons = read_yolo_polygons(label_path, width, height)
        if not polygons:
            continue
        right_gray = load_gray(row.right_path)
        # Exactly one dense matching pass per stereo pair; every object
        # below reuses this single disparity map.
        result = matcher.compute(left_gray, right_gray)

        for object_index, annotation in enumerate(polygons):
            if not 0 <= annotation.class_id < len(class_names):
                raise ValueError(
                    f"unknown class id {annotation.class_id} in {label_path}; "
                    f"supported ids: 0..{len(class_names) - 1}"
                )
            class_name = class_names[annotation.class_id]
            mask = build_object_mask(annotation.polygon, width, height)
            stats = matcher.instance_stats(result, mask, object_index)
            if stats.valid:
                object_disparity = matcher.object_disparity_map(result, mask)
                polar = compute_polar_feature(
                    left_gray, right_gray, object_disparity, mask=mask
                )
            else:
                # Without a valid correspondence, same-column L/R intensity
                # differences are not a physical polarization measurement.
                polar = np.zeros_like(left_gray, dtype=np.float32)

            window = make_crop_window(
                annotation.polygon, width, height, crop_pad
            )
            crop_gray = left_gray[window.y1 : window.y2, window.x1 : window.x2]
            crop_polar = polar[window.y1 : window.y2, window.x1 : window.x2]

            sample_name = f"{row.output_stem}_obj{object_index:03d}"
            relative = Path(row.split) / class_name / f"{sample_name}.png"
            write_classification_image(
                gray_root / relative,
                np.stack([crop_gray, crop_gray, crop_gray], axis=-1),
            )
            write_classification_image(
                polar_root / relative,
                build_polar_yolo_image(crop_gray, crop_polar),
            )
            records.append(
                SampleRecord(
                    sample_name=sample_name,
                    split=row.split,
                    group_name=row.group_name,
                    class_id=annotation.class_id,
                    class_name=class_name,
                    source_frame=row.output_stem,
                    object_index=object_index,
                    stereo_valid=stats.valid,
                    stereo_reason=stats.reason,
                    valid_ratio=round(stats.valid_ratio, 6),
                    disparity=round(stats.disparity, 4),
                    gray_path=relative.as_posix(),
                    polar_path=relative.as_posix(),
                )
            )

    write_manifest_and_summary(
        gray_root=gray_root,
        polar_root=polar_root,
        class_names=class_names,
        records=records,
        source_root=source_root,
        crop_pad=crop_pad,
    )
    return records


def main() -> None:
    args = parse_args()
    matcher = build_matcher(args)
    records = build_cls_datasets(
        source_root=Path(args.source),
        gray_root=Path(args.gray_output),
        polar_root=Path(args.polar_output),
        matcher=matcher,
        crop_pad=args.crop_pad,
        clean=args.clean,
    )
    print(f"Gray classification dataset: {Path(args.gray_output).resolve()}")
    print(f"Polar classification dataset: {Path(args.polar_output).resolve()}")
    print(f"Samples: {len(records)}")


MANIFEST_FIELDS = (
    "sample_name",
    "split",
    "group_name",
    "class_id",
    "class_name",
    "source_frame",
    "object_index",
    "stereo_valid",
    "stereo_reason",
    "valid_ratio",
    "disparity",
    "gray_path",
    "polar_path",
)


def record_to_row(record: SampleRecord) -> dict[str, str]:
    return {
        "sample_name": record.sample_name,
        "split": record.split,
        "group_name": record.group_name,
        "class_id": str(record.class_id),
        "class_name": record.class_name,
        "source_frame": record.source_frame,
        "object_index": str(record.object_index),
        "stereo_valid": "true" if record.stereo_valid else "false",
        "stereo_reason": record.stereo_reason,
        "valid_ratio": f"{record.valid_ratio:.6f}",
        "disparity": f"{record.disparity:.4f}",
        "gray_path": record.gray_path,
        "polar_path": record.polar_path,
    }


def write_manifest_and_summary(
    gray_root: Path,
    polar_root: Path,
    class_names: list[str],
    records: list[SampleRecord],
    source_root: Path,
    crop_pad: int,
) -> None:
    csv_text = _manifest_csv_text(records)
    summary_text = _summary_json_text(
        gray_root=gray_root,
        polar_root=polar_root,
        class_names=class_names,
        records=records,
        source_root=source_root,
        crop_pad=crop_pad,
    )
    for root in (gray_root, polar_root):
        (root / "dataset_manifest.csv").write_text(csv_text, encoding="utf-8")
        (root / "dataset_summary.json").write_text(summary_text, encoding="utf-8")


def _manifest_csv_text(records: list[SampleRecord]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_FIELDS))
    writer.writeheader()
    for record in records:
        writer.writerow(record_to_row(record))
    return buffer.getvalue()


def _summary_json_text(
    gray_root: Path,
    polar_root: Path,
    class_names: list[str],
    records: list[SampleRecord],
    source_root: Path,
    crop_pad: int,
) -> str:
    by_split: dict[str, list[SampleRecord]] = {
        split: [] for split in SPLIT_NAMES
    }
    for record in records:
        by_split[record.split].append(record)

    group_to_splits: dict[str, set[str]] = {}
    for record in records:
        group_to_splits.setdefault(record.group_name, set()).add(record.split)
    for group, splits in group_to_splits.items():
        if len(splits) > 1:
            raise RuntimeError(
                f"split isolation violated: group {group!r} spans {sorted(splits)}"
            )

    summary = {
        "class_names": class_names,
        "source_root": source_root.resolve().as_posix(),
        "crop_padding": crop_pad,
        "gray_root": gray_root.resolve().as_posix(),
        "polar_root": polar_root.resolve().as_posix(),
        "splits": {
            split: {
                "samples": len(rows),
                "per_class": {
                    name: sum(1 for row in rows if row.class_name == name)
                    for name in class_names
                },
                "stereo_valid": sum(1 for row in rows if row.stereo_valid),
                "stereo_invalid": sum(1 for row in rows if not row.stereo_valid),
            }
            for split, rows in by_split.items()
            if split in {record.split for record in records}
        },
    }
    return json.dumps(summary, ensure_ascii=False, indent=2) + "\n"


if __name__ == "__main__":
    main()
