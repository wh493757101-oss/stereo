"""Migrate Labelme polygons from old to new rectified-left coordinates.

The old rectified dataset was produced with the raw MATLAB-exported rotation
matrix handed to ``cv2.stereoRectify`` directly, while the v2 rectification
uses ``R_cv = R_matlab.T``. Both rectified-left views observe the same
undistorted left-camera rays, so the mapping between their pixel grids is an
exact homography::

    H = P1_new @ R1_new @ inv(R1_old) @ inv(P1_old)

applied to homogeneous old-rectified pixels. This command rewrites every
Labelme JSON under ``--annotations`` into that new coordinate system and
writes only to a distinct ``--output`` root.

For every new rectified left image an output JSON is produced: either the
migrated annotation, or an empty Labelme JSON when no old annotation exists,
so known negative frames are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.rectify_stereo_dataset import (
        IMAGE_EXTENSIONS,
        R_CONVENTIONS,
        RectificationMaps,
        build_rectify_maps,
        load_calibration_npz,
    )
except ModuleNotFoundError:  # Support direct `python scripts/...py` execution.
    from rectify_stereo_dataset import (
        IMAGE_EXTENSIONS,
        R_CONVENTIONS,
        RectificationMaps,
        build_rectify_maps,
        load_calibration_npz,
    )


LABELME_VERSION = "5.2.1"


def left_rectified_homography(
    old_maps: RectificationMaps, new_maps: RectificationMaps
) -> np.ndarray:
    """Homography mapping old rectified-left pixels to new rectified-left pixels.

    Both rectified views are pure rotations of the same left-camera ray fan,
    so ``x_new ~ P1n @ R1n @ inv(R1o) @ inv(P1o) @ x_old``.
    """
    homography = (
        new_maps.p1[:, :3]
        @ new_maps.r1
        @ np.linalg.inv(old_maps.r1)
        @ np.linalg.inv(old_maps.p1[:, :3])
    )
    if not np.isfinite(homography).all():
        raise ValueError("rectified-left homography contains non-finite values")
    return homography / homography[2, 2]


def validate_output_root(output_root: Path, protected_roots: list[Path]) -> None:
    """Refuse an output root equal to, inside, or containing a protected root."""
    output = os.path.normcase(str(output_root.resolve()))
    for protected in protected_roots:
        candidate = os.path.normcase(str(protected.resolve()))
        if output == candidate:
            raise ValueError(f"output root aliases protected root: {protected}")
        if output.startswith(candidate + os.sep) or candidate.startswith(output + os.sep):
            raise ValueError(
                f"output root {output_root} overlaps protected root: {protected}"
            )


def collect_new_left_images(new_image_root: Path) -> list[tuple[Path, Path]]:
    """Return sorted (relative group dir, left image path) pairs for every image."""
    if not new_image_root.is_dir():
        raise FileNotFoundError(f"new rectified image root not found: {new_image_root}")

    entries: list[tuple[Path, Path]] = []
    for left_dir in sorted(new_image_root.rglob("left")):
        if not left_dir.is_dir():
            continue
        relative_group = left_dir.parent.relative_to(new_image_root)
        for image_path in sorted(left_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                entries.append((relative_group, image_path))
    if not entries:
        raise FileNotFoundError(f"no left images found under {new_image_root}")
    return entries


def transform_points(
    points: list[list[float]],
    homography: np.ndarray,
    width: int,
    height: int,
) -> list[list[float]]:
    if not points:
        return []
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(array).all():
        raise ValueError("shape points contain non-finite coordinates")
    transformed = cv2.perspectiveTransform(
        array.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    transformed[:, 0] = np.clip(transformed[:, 0], 0.0, float(width))
    transformed[:, 1] = np.clip(transformed[:, 1], 0.0, float(height))
    return [[float(x), float(y)] for x, y in transformed]


def empty_labelme_json(
    image_path: str, width: int, height: int
) -> dict:
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": [],
        "imagePath": image_path,
        "imageData": None,
        "imageWidth": width,
        "imageHeight": height,
    }


def migrate_annotation(
    data: dict,
    homography: np.ndarray,
    width: int,
    height: int,
    image_path: str,
    keep_image_data: bool = False,
) -> dict:
    migrated = dict(data)
    shapes: list[dict] = []
    for shape in data.get("shapes", []):
        new_shape = dict(shape)
        new_shape["points"] = transform_points(
            shape.get("points", []), homography, width, height
        )
        shapes.append(new_shape)

    migrated["shapes"] = shapes
    migrated["imagePath"] = image_path
    migrated["imageWidth"] = width
    migrated["imageHeight"] = height
    if not keep_image_data:
        migrated["imageData"] = None
    return migrated


def migrate_annotations(
    annotation_root: Path,
    new_image_root: Path,
    output_root: Path,
    homography: np.ndarray,
    width: int,
    height: int,
    keep_image_data: bool = False,
    dry_run: bool = False,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for relative_group, image_path in collect_new_left_images(new_image_root):
        stem = image_path.stem
        output_dir = output_root / relative_group
        output_json = output_dir / f"{stem}.json"
        source_json = annotation_root / relative_group / f"{stem}.json"
        # LabelMe resolves imagePath relative to the JSON file's directory, so
        # build a cross-root relative path even though output and image trees
        # are separate.
        relative_image = Path(
            os.path.relpath(image_path.resolve(), start=output_dir.resolve())
        ).as_posix()

        if source_json.is_file():
            with source_json.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            migrated = migrate_annotation(
                data=data,
                homography=homography,
                width=width,
                height=height,
                image_path=relative_image,
                keep_image_data=keep_image_data,
            )
            status = "migrated"
            shape_count = len(migrated["shapes"])
        else:
            migrated = empty_labelme_json(relative_image, width, height)
            status = "empty"
            shape_count = 0

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        rows.append(
            {
                "relative_group": relative_group.as_posix(),
                "stem": stem,
                "source_json": (
                    source_json.relative_to(annotation_root).as_posix()
                    if source_json.is_file()
                    else ""
                ),
                "output_json": output_json.relative_to(output_root).as_posix(),
                "status": status,
                "shape_count": shape_count,
            }
        )

    return rows


def write_migration_manifest(output_root: Path, rows: list[dict[str, str | int]]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "migration_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Labelme polygons into the new rectified-left coordinate system.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("datasets/bz_JSON"),
        help="Root containing old Labelme JSON files.",
    )
    parser.add_argument(
        "--new-image-root",
        type=Path,
        required=True,
        help="Root of the new rectified dataset (searched recursively for left/ dirs).",
    )
    parser.add_argument(
        "--old-image-root",
        type=Path,
        default=Path("datasets/Rectified"),
        help="Root of the old rectified dataset, used only for output-root safety checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Distinct output root for migrated JSON files.",
    )
    parser.add_argument(
        "--old-calib",
        type=Path,
        default=Path("datasets/bd_image/stereo_calib.npz"),
        help="Calibration used to produce the old rectified dataset.",
    )
    parser.add_argument(
        "--old-r-convention",
        choices=R_CONVENTIONS,
        default="opencv",
        help="Convention of R in --old-calib; the old dataset used the stored R as-is.",
    )
    parser.add_argument(
        "--new-calib",
        type=Path,
        default=Path("datasets/bd_image/stereo_calib.npz"),
        help="Calibration used to produce the new rectified dataset.",
    )
    parser.add_argument(
        "--new-r-convention",
        choices=R_CONVENTIONS,
        default="matlab",
        help="Convention of R in --new-calib; 'matlab' applies R_cv = R.T.",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="cv2.stereoRectify alpha")
    parser.add_argument(
        "--keep-image-data",
        action="store_true",
        help="Keep the embedded base64 imageData instead of removing it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned migration without writing any files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    validate_output_root(
        args.output,
        [args.annotations, args.new_image_root, args.old_image_root],
    )

    old_calibration = load_calibration_npz(args.old_calib, r_convention=args.old_r_convention)
    new_calibration = load_calibration_npz(args.new_calib, r_convention=args.new_r_convention)
    if old_calibration.image_size != new_calibration.image_size:
        raise ValueError(
            "old and new calibration image sizes differ: "
            f"{old_calibration.image_size} vs {new_calibration.image_size}"
        )

    old_maps = build_rectify_maps(old_calibration, alpha=args.alpha)
    new_maps = build_rectify_maps(new_calibration, alpha=args.alpha)
    homography = left_rectified_homography(old_maps, new_maps)

    rows = migrate_annotations(
        annotation_root=args.annotations,
        new_image_root=args.new_image_root,
        output_root=args.output,
        homography=homography,
        width=new_calibration.image_size[0],
        height=new_calibration.image_size[1],
        keep_image_data=args.keep_image_data,
        dry_run=args.dry_run,
    )

    migrated = sum(1 for row in rows if row["status"] == "migrated")
    empty = sum(1 for row in rows if row["status"] == "empty")
    if args.dry_run:
        print(f"Dry run: {migrated} annotations to migrate, {empty} empty frames")
    else:
        manifest_path = write_migration_manifest(args.output, rows)
        print(f"Migrated {migrated} annotations, wrote {empty} empty frames")
        print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
