import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.polar_utils import (
    IMAGE_EXTENSIONS,
    MAP_EXTENSIONS,
    collect_files_by_stem,
    load_map,
    load_rgb_image,
    require_matching_stems,
    rgb_to_gray_float,
    save_rgb_image,
    warp_right_to_left,
)


def normalize_optional_dir(value):
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def as_2d_float(array, name):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise RuntimeError(f"{name} must be a 2D map, got shape={array.shape}.")
    return array


def save_mask(path, mask):
    # bool mask 落盘为 0/255 PNG，方便后续 OpenCV/人工查看。
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def collect_optional_files(folder, extensions, stems, name):
    folder = normalize_optional_dir(folder)
    if folder is None:
        # 没有可选目录时使用全 1 mask，相当于不引入该类约束。
        print(f"No {name} directory provided; using all-one {name} mask.")
        return None

    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"{name} directory not found: {folder_path}; using all-one {name} mask.")
        return None

    files = collect_files_by_stem(folder_path, extensions)
    if not files:
        print(f"No {name} files found in {folder_path}; using all-one {name} mask.")
        return None

    missing = sorted(set(stems) - set(files.keys()))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Missing {name} files for stems: {preview}")

    return files


def load_object_mask(label_path, expected_shape):
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise RuntimeError(f"Failed to read label mask: {label_path}")
    if label.shape != tuple(expected_shape):
        raise RuntimeError(f"Label shape mismatch: label={label.shape}, expected={tuple(expected_shape)}.")
    return label > 0


def build_edge_safe_mask(object_mask, erode_pixels, has_object_mask):
    # 腐蚀 object mask 会丢掉物体边缘像素，降低边界错位对 polar_norm 的污染。
    if not has_object_mask:
        return np.ones_like(object_mask, dtype=bool)
    if erode_pixels <= 0:
        return object_mask.astype(bool)

    kernel_size = int(erode_pixels) * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1)
    return eroded > 0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate left-view aligned polarization features.")
    parser.add_argument("--left_dir", default="datasets/MyPolarData/left")
    parser.add_argument("--right_dir", default="datasets/MyPolarData/right")
    parser.add_argument("--disp_dir", default="datasets/MyPolarData/disp")
    parser.add_argument("--label_dir", default="datasets/MyPolarData/labels")
    parser.add_argument("--right_warp_dir", default="datasets/MyPolarData/right_warp")
    parser.add_argument("--polar_norm_dir", default="datasets/MyPolarData/polar_norm")
    parser.add_argument("--polar_diff_dir", default="datasets/MyPolarData/polar_diff")
    parser.add_argument("--valid_mask_dir", default="datasets/MyPolarData/valid_mask")
    parser.add_argument("--geometry_valid_mask_dir", default="datasets/MyPolarData/geometry_valid_mask")
    parser.add_argument("--object_mask_dir", default="datasets/MyPolarData/object_mask")
    parser.add_argument("--edge_safe_mask_dir", default="datasets/MyPolarData/edge_safe_mask")
    # edge_erode_pixels 越大，最终 valid_mask 越保守，会直接改变生成的 polar_norm 有效区域。
    parser.add_argument("--edge_erode_pixels", type=int, default=3)
    args = parser.parse_args()

    if args.edge_erode_pixels < 0:
        raise RuntimeError("--edge_erode_pixels must be >= 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stems, left_files, matched = require_matching_stems(
        args.left_dir,
        right=(args.right_dir, IMAGE_EXTENSIONS),
        disp=(args.disp_dir, MAP_EXTENSIONS),
    )
    label_files = collect_optional_files(args.label_dir, (".png",), stems, "label")

    right_warp_dir = Path(args.right_warp_dir)
    polar_norm_dir = Path(args.polar_norm_dir)
    polar_diff_dir = Path(args.polar_diff_dir)
    valid_mask_dir = Path(args.valid_mask_dir)
    geometry_valid_mask_dir = Path(args.geometry_valid_mask_dir)
    object_mask_dir = Path(args.object_mask_dir)
    edge_safe_mask_dir = Path(args.edge_safe_mask_dir)
    # 这些目录都是生成物输出目录；重复运行会覆盖相同 stem 的特征和 mask。
    right_warp_dir.mkdir(parents=True, exist_ok=True)
    polar_norm_dir.mkdir(parents=True, exist_ok=True)
    polar_diff_dir.mkdir(parents=True, exist_ok=True)
    valid_mask_dir.mkdir(parents=True, exist_ok=True)
    geometry_valid_mask_dir.mkdir(parents=True, exist_ok=True)
    object_mask_dir.mkdir(parents=True, exist_ok=True)
    edge_safe_mask_dir.mkdir(parents=True, exist_ok=True)

    mask_pixel_counts = {
        "geometry": 0,
        "object": 0,
        "edge_safe": 0,
        "final": 0,
        "total": 0,
    }

    for stem in tqdm(stems, desc="Generating polarization features"):
        left_rgb = load_rgb_image(left_files[stem]).astype(np.float32)
        right_rgb = load_rgb_image(matched["right"][stem]).astype(np.float32)
        disp = as_2d_float(load_map(matched["disp"][stem]), "disp")

        expected_shape = left_rgb.shape[:2]
        if right_rgb.shape[:2] != expected_shape:
            raise RuntimeError(f"Right image shape mismatch for '{stem}': right={right_rgb.shape[:2]}, left={expected_shape}.")
        if disp.shape != expected_shape:
            raise RuntimeError(f"Disp shape mismatch for '{stem}': disp={disp.shape}, left={expected_shape}.")

        right_warp_rgb, geometry_valid_mask = warp_right_to_left(right_rgb, disp, device)
        geometry_valid_mask = geometry_valid_mask > 0.5

        if label_files is None:
            object_mask = np.ones(expected_shape, dtype=bool)
            has_object_mask = False
        else:
            object_mask = load_object_mask(label_files[stem], expected_shape)
            has_object_mask = True

        edge_safe_mask = build_edge_safe_mask(object_mask, args.edge_erode_pixels, has_object_mask)

        # 最终有效区域同时满足几何可采样、属于目标前景、并避开标注边缘。
        valid_mask = geometry_valid_mask & object_mask & edge_safe_mask
        valid_mask_float = valid_mask.astype(np.float32)
        right_warp_rgb *= valid_mask_float[..., None]
        mask_pixel_counts["geometry"] += int(np.count_nonzero(geometry_valid_mask))
        mask_pixel_counts["object"] += int(np.count_nonzero(object_mask))
        mask_pixel_counts["edge_safe"] += int(np.count_nonzero(edge_safe_mask))
        mask_pixel_counts["final"] += int(np.count_nonzero(valid_mask))
        mask_pixel_counts["total"] += int(valid_mask.size)

        left_gray = rgb_to_gray_float(left_rgb)
        right_gray = rgb_to_gray_float(right_warp_rgb)

        polar_diff = np.abs(left_gray - right_gray).astype(np.float32)
        polar_norm = polar_diff / (left_gray + right_gray + 1e-6)
        # 无效区域归零，后续训练/分析再配合 valid_mask 排除这些像素。
        polar_diff *= valid_mask_float
        polar_norm *= valid_mask_float

        save_rgb_image(right_warp_dir / f"{stem}.png", right_warp_rgb)
        np.save(polar_diff_dir / f"{stem}.npy", polar_diff.astype(np.float32))
        np.save(polar_norm_dir / f"{stem}.npy", polar_norm.astype(np.float32))
        save_mask(geometry_valid_mask_dir / f"{stem}.png", geometry_valid_mask)
        save_mask(object_mask_dir / f"{stem}.png", object_mask)
        save_mask(edge_safe_mask_dir / f"{stem}.png", edge_safe_mask)
        save_mask(valid_mask_dir / f"{stem}.png", valid_mask)

    print(f"Saved warped right images to: {right_warp_dir.resolve()}")
    print(f"Saved polarization features to: {polar_norm_dir.resolve()} and {polar_diff_dir.resolve()}")
    print(f"Saved final valid masks to: {valid_mask_dir.resolve()}")
    print(f"Saved geometry valid masks to: {geometry_valid_mask_dir.resolve()}")
    print(f"Saved object masks to: {object_mask_dir.resolve()}")
    print(f"Saved edge-safe masks to: {edge_safe_mask_dir.resolve()}")
    total_pixels = max(mask_pixel_counts["total"], 1)
    print(
        "Mask valid ratios: "
        f"geometry={mask_pixel_counts['geometry'] / total_pixels:.4f}, "
        f"object={mask_pixel_counts['object'] / total_pixels:.4f}, "
        f"edge_safe={mask_pixel_counts['edge_safe'] / total_pixels:.4f}, "
        f"final={mask_pixel_counts['final'] / total_pixels:.4f}"
    )


if __name__ == "__main__":
    main()
