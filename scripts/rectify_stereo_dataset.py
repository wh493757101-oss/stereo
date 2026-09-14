"""Batch undistort + stereo-rectify nested left/right datasets.

Expected input examples:
    datasets/Single/Metal submarine/20 NTU/left/000.png
    datasets/Single/Metal submarine/20 NTU/right/000.png
    datasets/Mixed/All mixed/10 NTU/left/000.png
    datasets/Mixed/All mixed/10 NTU/right/000.png

Calibration .npz keys:
    K1, D1, K2, D2, R, T, image_size

``image_size`` must be [width, height], matching OpenCV's convention.

Rotation matrix convention
--------------------------
``cv2.stereoRectify`` expects ``R`` to rotate points from the left camera
frame into the right camera frame (OpenCV convention). Calibration exported
from MATLAB uses the transpose of that matrix, so by default this module
applies ``R_cv = R_matlab.T``. Pass ``r_convention="opencv"`` (CLI:
``--r-convention opencv``) for calibration files whose ``R`` is already in
OpenCV convention.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
R_CONVENTIONS = ("matlab", "opencv")


@dataclass(frozen=True)
class ImagePair:
    stem: str
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class RectifyGroup:
    input_root: Path
    group_dir: Path
    relative_group: Path
    pairs: list[ImagePair]


@dataclass(frozen=True)
class StereoCalibration:
    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    r: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int]
    r_raw: np.ndarray
    r_convention: str


@dataclass(frozen=True)
class RectificationMaps:
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q: np.ndarray


def _image_files_by_stem(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def output_group_root(output_root: Path, input_root: Path, group_dir: Path) -> Path:
    """Return output directory preserving input-root name and nested group path."""
    return output_root / input_root.name / group_dir.relative_to(input_root)


def collect_rectify_groups(input_roots: list[Path]) -> list[RectifyGroup]:
    """Find all nested directories that contain matched left/right image pairs."""
    groups: list[RectifyGroup] = []

    for input_root in input_roots:
        if not input_root.exists():
            raise FileNotFoundError(f"Input root not found: {input_root}")

        for left_dir in sorted(input_root.rglob("left")):
            if not left_dir.is_dir():
                continue
            group_dir = left_dir.parent
            right_dir = group_dir / "right"
            if not right_dir.is_dir():
                continue

            left_files = _image_files_by_stem(left_dir)
            right_files = _image_files_by_stem(right_dir)
            stems = sorted(set(left_files) & set(right_files))
            if not stems:
                continue

            groups.append(
                RectifyGroup(
                    input_root=input_root,
                    group_dir=group_dir,
                    relative_group=input_root.name / group_dir.relative_to(input_root),
                    pairs=[
                        ImagePair(stem=stem, left_path=left_files[stem], right_path=right_files[stem])
                        for stem in stems
                    ],
                )
            )

    if not groups:
        raise RuntimeError("No nested left/right image groups found.")
    return groups


def _matrix3(name: str, value: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(
            f"{path.name}: {name} must be a 3x3 matrix, got shape {array.shape}"
        )
    return array


def _rotation_into_opencv_convention(
    rotation: np.ndarray, convention: str, path: Path
) -> np.ndarray:
    if convention not in R_CONVENTIONS:
        raise ValueError(
            f"Unknown R convention {convention!r}; expected one of {R_CONVENTIONS}"
        )
    if convention == "matlab":
        return np.ascontiguousarray(rotation.T)
    return rotation


def load_calibration_npz(path: Path, r_convention: str = "matlab") -> StereoCalibration:
    """Load stereo calibration matrices from a NumPy .npz file.

    ``r_convention`` states which convention the stored ``R`` uses:

    * ``"matlab"`` (default): the file was exported from MATLAB, so the
      OpenCV-convention rotation is ``R_cv = R.T``.
    * ``"opencv"``: ``R`` already rotates left-camera points into the right
      camera frame and is used as-is.
    """
    data = np.load(str(path))
    required = {"K1", "D1", "K2", "D2", "R", "T", "image_size"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"Calibration file missing keys: {', '.join(missing)}")

    k1 = _matrix3("K1", data["K1"], path)
    k2 = _matrix3("K2", data["K2"], path)
    r_raw = _matrix3("R", data["R"], path)
    d1 = np.asarray(data["D1"], dtype=np.float64).reshape(-1)
    d2 = np.asarray(data["D2"], dtype=np.float64).reshape(-1)
    if d1.size < 4 or d2.size < 4:
        raise ValueError(
            f"{path.name}: D1/D2 must contain at least 4 distortion coefficients"
        )
    t = np.asarray(data["T"], dtype=np.float64).reshape(3, 1)
    if not np.isfinite(t).all() or np.allclose(t, 0.0):
        raise ValueError(f"{path.name}: T must be a non-zero translation vector")

    image_size_arr = np.asarray(data["image_size"]).reshape(-1)
    if image_size_arr.size != 2:
        raise ValueError("image_size must contain [width, height].")
    width, height = int(image_size_arr[0]), int(image_size_arr[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got [{width}, {height}]")

    r = _rotation_into_opencv_convention(r_raw, r_convention, path)
    return StereoCalibration(
        k1=k1,
        d1=d1,
        k2=k2,
        d2=d2,
        r=r,
        t=t,
        image_size=(width, height),
        r_raw=r_raw,
        r_convention=r_convention,
    )


def build_rectify_maps(
    calibration: StereoCalibration,
    alpha: float = 0.0,
) -> RectificationMaps:
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        calibration.k1,
        calibration.d1,
        calibration.k2,
        calibration.d2,
        calibration.image_size,
        calibration.r,
        calibration.t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        calibration.k1,
        calibration.d1,
        r1,
        p1,
        calibration.image_size,
        cv2.CV_32FC1,
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calibration.k2,
        calibration.d2,
        r2,
        p2,
        calibration.image_size,
        cv2.CV_32FC1,
    )
    return RectificationMaps(
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
        r1=r1,
        r2=r2,
        p1=p1,
        p2=p2,
        q=q,
    )


def rectify_pair(
    pair: ImagePair,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    left = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError(f"Failed to read pair: {pair.left_path} | {pair.right_path}")

    got_left = (left.shape[1], left.shape[0])
    got_right = (right.shape[1], right.shape[0])
    if got_left != image_size or got_right != image_size:
        raise ValueError(
            f"Image size mismatch for {pair.stem}: left={got_left}, right={got_right}, "
            f"expected={image_size}"
        )

    map1x, map1y, map2x, map2y = maps
    return (
        cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR),
        cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR),
    )


def rectify_groups(
    groups: list[RectifyGroup],
    output_root: Path,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    image_size: tuple[int, int],
    dry_run: bool = False,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for group in groups:
        group_output = output_group_root(output_root, group.input_root, group.group_dir)
        left_output_dir = group_output / "left"
        right_output_dir = group_output / "right"
        if not dry_run:
            left_output_dir.mkdir(parents=True, exist_ok=True)
            right_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"{group.relative_group}: {len(group.pairs)} pairs")
        for pair in group.pairs:
            left_out = left_output_dir / f"{pair.stem}.png"
            right_out = right_output_dir / f"{pair.stem}.png"

            if not dry_run:
                rect_left, rect_right = rectify_pair(pair, maps, image_size)
                cv2.imwrite(str(left_out), rect_left)
                cv2.imwrite(str(right_out), rect_right)

            rows.append(
                {
                    "group": str(group.relative_group),
                    "stem": pair.stem,
                    "source_left": str(pair.left_path),
                    "source_right": str(pair.right_path),
                    "output_left": str(left_out),
                    "output_right": str(right_out),
                    "width": image_size[0],
                    "height": image_size[1],
                }
            )

    return rows


def write_manifest(output_root: Path, rows: list[dict[str, str | int]]) -> Path:
    manifest_path = output_root / "rectify_manifest.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def save_rectification_outputs(
    output_root: Path,
    maps: RectificationMaps,
    calibration: StereoCalibration,
    calib_path: Path,
    alpha: float,
) -> tuple[Path, Path]:
    """Save rectification matrices and provenance metadata to ``output_root``."""
    output_root.mkdir(parents=True, exist_ok=True)
    params_path = output_root / "rectification_params.npz"
    np.savez(
        params_path,
        R1=maps.r1,
        R2=maps.r2,
        P1=maps.p1,
        P2=maps.p2,
        Q=maps.q,
        image_size=np.array(calibration.image_size, dtype=np.int32),
    )
    metadata = {
        "calibration_path": str(calib_path),
        "r_convention": calibration.r_convention,
        "r_cv_derivation": (
            "R_cv = R.T (MATLAB convention transposed into OpenCV convention)"
            if calibration.r_convention == "matlab"
            else "R used as-is (already in OpenCV convention)"
        ),
        "alpha": alpha,
        "image_size": list(calibration.image_size),
        "flags": "cv2.CALIB_ZERO_DISPARITY",
    }
    metadata_path = output_root / "rectification_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return params_path, metadata_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch stereo rectify nested left/right datasets")
    parser.add_argument(
        "--calib",
        required=True,
        type=Path,
        help="Calibration .npz with K1,D1,K2,D2,R,T,image_size",
    )
    parser.add_argument(
        "--r-convention",
        choices=R_CONVENTIONS,
        default="matlab",
        help="Convention of the stored R matrix: 'matlab' applies R_cv = R.T "
        "(default), 'opencv' uses R as-is.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        default=[Path("datasets/Single"), Path("datasets/Mixed")],
        help="Input roots to scan recursively",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/Rectified_v2"),
        help="Output root",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="cv2.stereoRectify alpha")
    parser.add_argument("--dry-run", action="store_true", help="Only list work; do not write images")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    calibration = load_calibration_npz(args.calib, r_convention=args.r_convention)
    maps = build_rectify_maps(calibration, alpha=args.alpha)
    groups = collect_rectify_groups(args.input)

    rows = rectify_groups(
        groups=groups,
        output_root=args.output,
        maps=(maps.map1x, maps.map1y, maps.map2x, maps.map2y),
        image_size=calibration.image_size,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        manifest_path = write_manifest(args.output, rows)
        params_path, metadata_path = save_rectification_outputs(
            output_root=args.output,
            maps=maps,
            calibration=calibration,
            calib_path=args.calib,
            alpha=args.alpha,
        )
        print(f"Wrote {len(rows)} rectified pairs")
        print(f"Manifest: {manifest_path}")
        print(f"Params: {params_path}")
        print(f"Metadata: {metadata_path}")
    else:
        print(f"Dry run: found {len(rows)} matched pairs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
