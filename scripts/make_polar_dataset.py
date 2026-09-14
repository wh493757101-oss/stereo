"""Offline polar-aided YOLO training data preparation.

For each left/right image pair (already rectified):
  1. Full-image StereoSGBM matching -> disparity map (one dense pass per pair)
  2. Warp right to left view
  3. Polar feature P = |L - warp(R)| / (L + warp(R)) from original intensities
  4. Save [gray, polar, gray] as a standard 3-channel PNG for YOLO
  5. Optionally also save the legacy [gray, polar] 2-channel .npy file

The normal path consumes pair_manifest.csv emitted by
scripts/prepare_yolo_dataset.py so both train and val preserve their split. Use --crop
to emit one padded polar crop and localized label per instance for Stage-3 training.
"""

import argparse
import csv
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.polar_compute import (
    build_polar_yolo_image,
    compute_polar_feature,
)
from core.stereo_matching import StereoMatcher, StereoMatcherConfig, to_gray_u8

DEFAULT_CLASS_NAMES = (
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
)


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return to_gray_u8(img)


def process_pair(
    left_path: Path,
    right_path: Path,
    image_dir: Path,
    matcher: StereoMatcher,
    output_format: str,
    output_stem: str | None = None,
) -> list[Path]:
    left_gray = load_gray(left_path)
    right_gray = load_gray(right_path)

    result = matcher.compute(left_gray, right_gray)
    disp = result.disparity
    polar = compute_polar_feature(left_gray, right_gray, disp)

    output_paths = []
    stem = output_stem or left_path.stem
    if output_format in {"png3", "both"}:
        yolo_image = build_polar_yolo_image(left_gray, polar)
        png_path = image_dir / f"{stem}.png"
        cv2.imwrite(str(png_path), yolo_image)
        output_paths.append(png_path)

    if output_format in {"npy2", "both"}:
        two_ch = np.stack([left_gray.astype(np.float32), polar], axis=0)
        npy_path = image_dir / f"{stem}.npy"
        np.save(str(npy_path), two_ch)
        output_paths.append(npy_path)

    return output_paths


def process_pair_crops(
    left_path: Path,
    right_path: Path,
    label_path: Path,
    image_dir: Path,
    matcher: StereoMatcher,
    padding: int,
    output_stem: str,
) -> int:
    left_gray = load_gray(left_path)
    right_gray = load_gray(right_path)
    result = matcher.compute(left_gray, right_gray)
    disp = result.disparity
    polar = compute_polar_feature(left_gray, right_gray, disp)

    annotations = read_yolo_polygons(label_path, left_gray.shape[1], left_gray.shape[0])
    if not annotations:
        raise ValueError(f"no YOLO polygons in {label_path}")

    for index, annotation in enumerate(annotations):
        crop_stem = f"{output_stem}_obj{index:03d}"
        crop_window = make_crop_window(
            annotation.polygon,
            left_gray.shape[1],
            left_gray.shape[0],
            padding,
        )
        write_polar_crop(
            gray=left_gray,
            polar=polar,
            window=crop_window,
            annotation=annotation,
            image_dir=image_dir,
            output_stem=crop_stem,
        )
    return len(annotations)


def write_data_yaml(output_dir: Path, names: list[str]) -> None:
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(names))
    content = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names_block}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def resolve_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


@dataclass(frozen=True)
class YoloPolygon:
    class_id: int
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CropWindow:
    x1: int
    y1: int
    x2: int
    y2: int


def read_yolo_polygons(
    label_path: Path,
    width: int,
    height: int,
) -> list[YoloPolygon]:
    polygons: list[YoloPolygon] = []
    for line_index, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or len(values) % 2 == 0:
            raise ValueError(f"invalid YOLO polygon at {label_path}:{line_index}")
        points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygons.append(
            YoloPolygon(
                class_id=int(values[0]),
                polygon=tuple((float(x), float(y)) for x, y in points),
            )
        )
    return polygons


def make_crop_window(
    polygon: tuple[tuple[float, float], ...],
    width: int,
    height: int,
    padding: int,
) -> CropWindow:
    points = np.asarray(polygon, dtype=np.float32)
    x1 = max(0, math.floor(float(points[:, 0].min())) - padding)
    y1 = max(0, math.floor(float(points[:, 1].min())) - padding)
    x2 = min(width, math.ceil(float(points[:, 0].max())) + padding)
    y2 = min(height, math.ceil(float(points[:, 1].max())) + padding)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("annotation is too small for a Model-B crop")
    return CropWindow(x1=x1, y1=y1, x2=x2, y2=y2)


def crop_polygon(
    polygon: tuple[tuple[float, float], ...],
    window: CropWindow,
) -> tuple[tuple[float, float], ...]:
    return tuple((float(x - window.x1), float(y - window.y1)) for x, y in polygon)


def write_polar_crop(
    gray: np.ndarray,
    polar: np.ndarray,
    window: CropWindow,
    annotation: YoloPolygon,
    image_dir: Path,
    output_stem: str,
) -> Path:
    local_polygon = crop_polygon(annotation.polygon, window)
    points = np.asarray(local_polygon, dtype=np.float32)
    crop_h, crop_w = window.y2 - window.y1, window.x2 - window.x1
    if (
        points[:, 0].min() < 0.0
        or points[:, 0].max() > crop_w
        or points[:, 1].min() < 0.0
        or points[:, 1].max() > crop_h
    ):
        raise ValueError(f"Model-B polygon exceeds crop for {output_stem}")

    crop_gray = gray[window.y1 : window.y2, window.x1 : window.x2]
    crop_polar = polar[window.y1 : window.y2, window.x1 : window.x2]
    # Match inference: the polar channel is valid only inside the target mask.
    polar_mask = np.zeros(crop_gray.shape, dtype=np.uint8)
    cv2.fillPoly(polar_mask, [points.astype(np.int32)], 255)
    crop_polar = crop_polar * (polar_mask > 0)
    image_path = image_dir / f"{output_stem}.png"
    cv2.imwrite(str(image_path), build_polar_yolo_image(crop_gray, crop_polar))

    coordinates = [str(annotation.class_id)]
    coordinates.extend(
        f"{x / crop_w:.6f}" + " " + f"{y / crop_h:.6f}" for x, y in local_polygon
    )
    label_dir = image_dir.parent.parent.joinpath("labels", image_dir.name)
    label_dir.mkdir(parents=True, exist_ok=True)
    label_dir.joinpath(f"{output_stem}.txt").write_text(
        " ".join(coordinates) + "\n", encoding="utf-8"
    )
    return image_path


def read_pair_manifest(
    manifest_path: Path,
    workspace: Path,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_stems: set[str] = set()

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"output_stem", "split", "left_path", "right_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(
                f"{manifest_path} is missing required columns: {', '.join(missing)}"
            )

        for row_index, row in enumerate(reader, start=2):
            output_stem = str(row["output_stem"]).strip()
            split = str(row["split"]).strip()
            if not output_stem or split not in {"train", "val", "test"}:
                raise ValueError(
                    f"invalid output_stem/split at {manifest_path}:{row_index}"
                )
            if output_stem in seen_stems:
                raise ValueError(
                    f"duplicate output_stem {output_stem!r} in {manifest_path}"
                )
            seen_stems.add(output_stem)

            grouped[split].append(
                {
                    "output_stem": output_stem,
                    "left_path": str(
                        resolve_path(str(row["left_path"]).strip(), workspace)
                    ),
                    "right_path": str(
                        resolve_path(str(row["right_path"]).strip(), workspace)
                    ),
                }
            )

    if not grouped:
        raise ValueError(f"pair manifest is empty: {manifest_path}")
    return dict(grouped)


def prepare_manifest_split(
    split: str,
    rows: list[dict[str, str]],
    out_dir: Path,
    labels_root: Path | None,
    args: argparse.Namespace,
) -> tuple[int, int]:
    image_dir = out_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dst = out_dir / "labels" / split
    if labels_root is not None:
        label_dst.mkdir(parents=True, exist_ok=True)

    image_count = 0
    copied_labels = 0
    for index, row in enumerate(rows, start=1):
        left_path = Path(row["left_path"])
        right_path = Path(row["right_path"])
        output_stem = row["output_stem"]

        if args.crop:
            if labels_root is None:
                raise RuntimeError("--crop requires --labels-root")
            label_path = labels_root / split / f"{output_stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"missing Model-B label: {label_path}")
            crop_count = process_pair_crops(
                left_path=left_path,
                right_path=right_path,
                label_path=label_path,
                image_dir=image_dir,
                matcher=args.matcher,
                padding=args.crop_pad,
                output_stem=output_stem,
            )
            image_count += crop_count
            copied_labels += crop_count
        else:
            process_pair(
                left_path=left_path,
                right_path=right_path,
                image_dir=image_dir,
                matcher=args.matcher,
                output_format=args.format,
                output_stem=output_stem,
            )
            image_count += 1

            if labels_root is not None:
                label_path = labels_root / split / f"{output_stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, label_dst / label_path.name)
                    copied_labels += 1
                else:
                    print(f"  warning: missing YOLO label {label_path}")

        if index % 50 == 0 or index == len(rows):
            print(f"  {split}: processed {index}/{len(rows)}")

    return image_count, copied_labels


def prepare_legacy_input(
    in_dir: Path,
    out_dir: Path,
    split: str,
    labels_dir: Path | None,
    args: argparse.Namespace,
) -> tuple[int, int]:
    left_dir = in_dir / "left"
    right_dir = in_dir / "right"
    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Need {left_dir} and {right_dir}")

    image_dir = out_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dst: Path | None = None
    if labels_dir is not None:
        label_dst = out_dir / "labels" / split
        label_dst.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob("*.png")) + sorted(left_dir.glob("*.jpg"))
    copied_labels = 0
    for index, left_path in enumerate(left_files, start=1):
        right_path = right_dir / left_path.name
        if not right_path.exists():
            print(f"  skip {left_path.name}: no right image")
            continue

        process_pair(
            left_path=left_path,
            right_path=right_path,
            image_dir=image_dir,
            matcher=args.matcher,
            output_format=args.format,
        )

        if label_dst is not None and labels_dir is not None:
            label_path = labels_dir / f"{left_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, label_dst / label_path.name)
                copied_labels += 1

        if index % 50 == 0 or index == len(left_files):
            print(f"  {split}: processed {index}/{len(left_files)}")

    return len(left_files), copied_labels


def prepare_output_root(output_root: Path, clean: bool) -> None:
    resolved = output_root.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not clean:
            raise RuntimeError(
                f"output directory is not empty: {resolved}; pass --clean to replace it"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare polar-aided YOLO training data",
    )
    parser.add_argument(
        "--manifest",
        help="pair_manifest.csv written by scripts/prepare_yolo_dataset.py",
    )
    parser.add_argument(
        "--input",
        help="Legacy mode: root directory with left/ and right/ subdirectories",
    )
    parser.add_argument("--output", required=True, help="Output dataset root")
    parser.add_argument(
        "--clean", action="store_true", help="Replace an existing output dataset"
    )
    parser.add_argument(
        "--labels",
        help="Legacy mode: flat YOLO label directory matching image stems",
    )
    parser.add_argument(
        "--labels-root",
        help="Manifest mode: root containing labels/train and labels/val",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Split for legacy input mode",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Manifest splits to process",
    )
    parser.add_argument(
        "--workspace", default=".", help="Base for relative manifest paths"
    )
    parser.add_argument(
        "--format",
        default="png3",
        choices=["png3", "npy2", "both"],
    )
    parser.add_argument(
        "--names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names for data.yaml generation",
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
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Write one polar-aided crop per instance for Stage-3 Model B",
    )
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=10,
        help="Padding around each instance when --crop is enabled",
    )
    args = parser.parse_args()

    if bool(args.manifest) == bool(args.input):
        parser.error("pass exactly one of --manifest or --input")
    if args.manifest and args.labels:
        parser.error("--labels is legacy-only; use --labels-root with --manifest")
    if not args.manifest and args.labels_root:
        parser.error("--labels-root requires --manifest")
    if args.crop:
        if not args.manifest or not args.labels_root:
            parser.error("--crop requires --manifest and --labels-root")
        if args.format == "npy2":
            parser.error("--crop requires png3 or both; YOLO inference uses png3")
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


def main() -> None:
    args = parse_args()
    # One matcher instance is shared by every pair in both full-frame and crop
    # mode, so offline preparation and the online helper path use identical
    # settings.
    args.matcher = build_matcher(args)
    out_dir = Path(args.output)
    labels_root = Path(args.labels_root) if args.labels_root else None
    class_names = [name.strip() for name in args.names.split(",") if name.strip()]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("--names must contain unique, non-empty class names")

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Pair manifest not found: {manifest_path}")
        grouped = read_pair_manifest(manifest_path, Path(args.workspace).resolve())
        selected_splits = [split for split in args.splits if split in grouped]
        missing_splits = set(args.splits) - set(selected_splits)
        if missing_splits:
            raise ValueError(
                f"pair manifest has no rows for splits: {sorted(missing_splits)}"
            )
        prepare_output_root(out_dir, args.clean)
        summary = {
            split: prepare_manifest_split(
                split=split,
                rows=grouped[split],
                out_dir=out_dir,
                labels_root=labels_root,
                args=args,
            )
            for split in selected_splits
        }
    else:
        prepare_output_root(out_dir, args.clean)
        summary = {
            args.split: prepare_legacy_input(
                in_dir=Path(args.input),
                out_dir=out_dir,
                split=args.split,
                labels_dir=Path(args.labels) if args.labels else None,
                args=args,
            )
        }

    write_data_yaml(out_dir, class_names)
    print(f"Done. Output: {out_dir.resolve()}")
    for split, (image_count, label_count) in summary.items():
        print(f"  {split}: images={image_count}, labels={label_count}")


if __name__ == "__main__":
    main()
