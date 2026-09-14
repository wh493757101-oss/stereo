import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = (1280, 1024)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# MATLAB stereo calibration exported parameters converted to OpenCV format.
# Distortion order here is [k1, k2, p1, p2, k3].
K1 = np.array(
    [
        [3394.9, 0.0, 616.1],
        [0.0, 3395.7, 685.5],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D1 = np.array([0.0405, -0.7705, 0.0089, 0.0, 16.4495], dtype=np.float64)

K2 = np.array(
    [
        [3393.3, 0.0, 667.6],
        [0.0, 3394.4, 634.9],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
D2 = np.array([0.0461, -0.1994, 0.0090, -0.0005, 7.4903], dtype=np.float64)

R = np.array(
    [
        [0.9998, 0.0010, -0.0196],
        [-0.0008, 1.0, 0.0058],
        [0.0196, -0.0058, 0.9998],
    ],
    dtype=np.float64,
)
T = np.array([[-96.2611], [0.2437], [0.0399]], dtype=np.float64)

CROP_LEFT_PIXELS = 0
CROP_RIGHT_PIXELS = 0

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"


@dataclass(frozen=True)
class RectifyJob:
    name: str
    input_root: Path
    output_root: Path


RECTIFY_JOBS = (
    RectifyJob(
        name="PreparedSingle",
        input_root=DATASETS_ROOT / "SingleMaterial",
        output_root=DATASETS_ROOT / "PreparedSingle",
    ),
    RectifyJob(
        name="PreparedMixed",
        input_root=DATASETS_ROOT / "MixedMaterial",
        output_root=DATASETS_ROOT / "PreparedMixed",
    ),
)


def crop_rectified_image(img, crop_left=0, crop_right=0):
    """Crop left/right borders after rectification."""
    _, width = img.shape[:2]

    if crop_left < 0 or crop_right < 0:
        raise ValueError("Crop pixels must be non-negative.")

    if crop_left + crop_right >= width:
        raise ValueError(
            f"Invalid crop width: image_width={width}, crop_left={crop_left}, crop_right={crop_right}"
        )

    start_x = crop_left
    end_x = width - crop_right if crop_right > 0 else width
    return img[:, start_x:end_x]


def build_rectify_maps():
    print("Computing stereo rectification maps...")
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        IMAGE_SIZE,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, r1, p1, IMAGE_SIZE, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, r2, p2, IMAGE_SIZE, cv2.CV_32FC1)
    print("Rectification maps ready.")
    return map1x, map1y, map2x, map2y, p1, p2, q


def collect_group_pairs(root_dir: Path):
    if not root_dir.exists():
        raise RuntimeError(f"Input root not found: {root_dir}")

    group_pairs = []
    for group_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        left_dir = group_dir / "left"
        right_dir = group_dir / "right"
        if not left_dir.exists() or not right_dir.exists():
            print(f"Skipping {group_dir.name}: missing left/right directories.")
            continue

        left_files = {
            path.stem: path
            for path in left_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        right_files = {
            path.stem: path
            for path in right_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        common_stems = sorted(set(left_files) & set(right_files))
        if not common_stems:
            print(f"Skipping {group_dir.name}: no matched left/right stems.")
            continue

        missing_left = sorted(set(right_files) - set(left_files))
        missing_right = sorted(set(left_files) - set(right_files))
        if missing_left:
            print(f"Warning: {group_dir.name} missing left files for stems: {', '.join(missing_left[:5])}")
        if missing_right:
            print(f"Warning: {group_dir.name} missing right files for stems: {', '.join(missing_right[:5])}")

        group_pairs.append(
            {
                "group_name": group_dir.name,
                "pairs": [(stem, left_files[stem], right_files[stem]) for stem in common_stems],
            }
        )

    if not group_pairs:
        raise RuntimeError(f"No valid image groups found in {root_dir}")
    return group_pairs


def make_output_stem(group_name: str, stem: str) -> str:
    # Preserve source group in the flattened filename so later split/group rules stay traceable.
    return f"{group_name}_{stem}"


def rectify_pair(left_path: Path, right_path: Path, map1x, map1y, map2x, map2y):
    img_left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    img_right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if img_left is None or img_right is None:
        raise RuntimeError(f"Failed to read pair: left={left_path}, right={right_path}")

    if (img_left.shape[1], img_left.shape[0]) != IMAGE_SIZE:
        raise RuntimeError(
            f"Unexpected left image size for {left_path.name}: got={img_left.shape[1]}x{img_left.shape[0]}, "
            f"expected={IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
        )
    if (img_right.shape[1], img_right.shape[0]) != IMAGE_SIZE:
        raise RuntimeError(
            f"Unexpected right image size for {right_path.name}: got={img_right.shape[1]}x{img_right.shape[0]}, "
            f"expected={IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
        )

    rectified_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
    cropped_left = crop_rectified_image(rectified_left, CROP_LEFT_PIXELS, CROP_RIGHT_PIXELS)
    cropped_right = crop_rectified_image(rectified_right, CROP_LEFT_PIXELS, CROP_RIGHT_PIXELS)
    return cropped_left, cropped_right


def ensure_output_dirs(root_dir: Path):
    left_dir = root_dir / "left"
    right_dir = root_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    return left_dir, right_dir


def rectify_dataset(job: RectifyJob, map1x, map1y, map2x, map2y):
    print(f"\nProcessing {job.name}")
    print(f"Input root : {job.input_root}")
    print(f"Output root: {job.output_root}")

    group_pairs = collect_group_pairs(job.input_root)
    output_left_dir, output_right_dir = ensure_output_dirs(job.output_root)
    manifest_rows = []
    written_stems = set()

    for group in group_pairs:
        group_name = group["group_name"]
        pairs = group["pairs"]
        print(f"  Group {group_name}: {len(pairs)} pairs")

        for original_stem, left_path, right_path in pairs:
            output_stem = make_output_stem(group_name, original_stem)
            if output_stem in written_stems:
                raise RuntimeError(f"Duplicate flattened stem detected: {output_stem}")
            written_stems.add(output_stem)

            rectified_left, rectified_right = rectify_pair(left_path, right_path, map1x, map1y, map2x, map2y)

            left_output_path = output_left_dir / f"{output_stem}.png"
            right_output_path = output_right_dir / f"{output_stem}.png"
            cv2.imwrite(str(left_output_path), rectified_left)
            cv2.imwrite(str(right_output_path), rectified_right)

            manifest_rows.append(
                {
                    "group_name": group_name,
                    "source_stem": original_stem,
                    "output_stem": output_stem,
                    "source_left": str(left_path),
                    "source_right": str(right_path),
                    "output_left": str(left_output_path),
                    "output_right": str(right_output_path),
                    "width": rectified_left.shape[1],
                    "height": rectified_left.shape[0],
                    "crop_left_pixels": CROP_LEFT_PIXELS,
                    "crop_right_pixels": CROP_RIGHT_PIXELS,
                }
            )

    manifest_path = job.output_root / "rectify_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Finished {job.name}: wrote {len(manifest_rows)} pairs")
    print(f"  Left dir : {output_left_dir}")
    print(f"  Right dir: {output_right_dir}")
    print(f"  Manifest : {manifest_path}")


def main():
    map1x, map1y, map2x, map2y, p1, p2, q = build_rectify_maps()
    fx = float(p1[0, 0])
    baseline = abs(float(p2[0, 3]) / max(abs(float(p2[0, 0])), 1e-12))
    print(f"Rectified fx: {fx:.4f}")
    print(f"Baseline (mm): {baseline:.4f}")
    print(f"Q matrix ready with shape: {q.shape}")

    for job in RECTIFY_JOBS:
        rectify_dataset(job, map1x, map1y, map2x, map2y)

    print("\nAll dataset rectification jobs completed.")


if __name__ == "__main__":
    main()
