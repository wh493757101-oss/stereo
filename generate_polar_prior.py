import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from core.polar_utils import collect_files_by_stem


CLASS_IDS = [1, 2, 3]
CLASS_COLORS_BGR = {
    1: np.array([255, 0, 0], dtype=np.float32),
    2: np.array([0, 0, 255], dtype=np.float32),
    3: np.array([0, 255, 0], dtype=np.float32),
}


def load_polar_map(path):
    polar = np.load(str(path)).astype(np.float32)
    if polar.ndim == 3:
        polar = polar[:, :, 0]
    return polar


def load_valid_mask(path):
    valid = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if valid is None:
        raise RuntimeError(f"Failed to read valid mask: {path}")
    return np.asarray(valid > 0, dtype=bool)


def load_prior_model(path):
    # prior_model 由 analyze_polar_distribution.py 生成，包含 grid 和每个材质的密度曲线。
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    grid = np.asarray(model["grid"], dtype=np.float32)
    channel_order = [int(class_id) for class_id in model["channel_order"]]
    densities = [
        np.asarray(model["class_models"][str(class_id)]["density"], dtype=np.float32)
        for class_id in channel_order
    ]
    return model, grid, channel_order, densities


def prior_to_color(prior, channel_order):
    color = np.zeros((prior.shape[0], prior.shape[1], 3), dtype=np.float32)
    for channel_index, class_id in enumerate(channel_order):
        color += prior[:, :, channel_index:channel_index + 1] * CLASS_COLORS_BGR[class_id].reshape(1, 1, 3)
    return np.clip(color, 0, 255).astype(np.uint8)


def hard_prior_to_color(prior, valid_mask, channel_order):
    color = np.zeros((prior.shape[0], prior.shape[1], 3), dtype=np.uint8)
    if np.any(valid_mask):
        hard_index = np.argmax(prior, axis=2)
        for channel_index, class_id in enumerate(channel_order):
            class_mask = (hard_index == channel_index) & valid_mask
            color[class_mask] = CLASS_COLORS_BGR[class_id].astype(np.uint8)
    return color


def main():
    parser = argparse.ArgumentParser(description="Generate 3-channel soft polarization priors from a prior model.")
    parser.add_argument("--polar_dir", default="datasets/MyPolarData/polar_norm")
    parser.add_argument(
        "--prior_model_json",
        default="analysis/priors/single_material_prior_v2/prior_model.json",
    )
    parser.add_argument("--output_dir", default="datasets/MyPolarData")
    parser.add_argument("--valid_mask_dir", default="datasets/MyPolarData/valid_mask")
    args = parser.parse_args()

    polar_files = collect_files_by_stem(args.polar_dir, (".npy",))
    stems = sorted(polar_files.keys())
    if not stems:
        raise RuntimeError(f"No polar maps found in {args.polar_dir}")

    valid_mask_files = None
    if args.valid_mask_dir:
        valid_mask_files = collect_files_by_stem(args.valid_mask_dir, (".png",))
        missing_valid = sorted(set(stems) - set(valid_mask_files.keys()))
        if missing_valid:
            preview = ", ".join(missing_valid[:5])
            raise RuntimeError(f"Missing valid masks for stems: {preview}")

    _, grid, channel_order, densities = load_prior_model(args.prior_model_json)
    grid_min = float(grid[0])
    grid_max = float(grid[-1])

    output_dir = Path(args.output_dir)
    prior_dir = output_dir / "polar_prior"
    prior_vis_dir = output_dir / "polar_prior_vis"
    prior_hard_dir = output_dir / "polar_prior_hard"
    # 这里会生成训练用 soft prior 和两类可视化图，旧的同名 stem 会被覆盖。
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_vis_dir.mkdir(parents=True, exist_ok=True)
    prior_hard_dir.mkdir(parents=True, exist_ok=True)

    for stem in tqdm(stems, desc="Generating polar priors"):
        polar = load_polar_map(polar_files[stem])
        valid_mask = np.isfinite(polar)
        if valid_mask_files is not None:
            valid_mask &= load_valid_mask(valid_mask_files[stem])

        clipped = np.clip(np.nan_to_num(polar, nan=grid_min, posinf=grid_max, neginf=grid_min), grid_min, grid_max)

        density_maps = []
        for density in densities:
            # 对每个像素按 polar_norm 查密度曲线，得到各材质的未归一化似然。
            density_map = np.interp(clipped.reshape(-1), grid, density, left=0.0, right=0.0).reshape(clipped.shape)
            density_maps.append(density_map.astype(np.float32))

        stacked = np.stack(density_maps, axis=2)
        stacked[~valid_mask] = 0.0

        denom = np.sum(stacked, axis=2, keepdims=True)
        # 三个通道归一化成概率形式；无效区域保持全 0。
        prior = np.divide(stacked, denom, out=np.zeros_like(stacked), where=denom > 1e-12)
        prior[~valid_mask] = 0.0

        np.save(prior_dir / f"{stem}.npy", prior.astype(np.float32))
        cv2.imwrite(str(prior_vis_dir / f"{stem}.png"), prior_to_color(prior, channel_order))
        cv2.imwrite(str(prior_hard_dir / f"{stem}.png"), hard_prior_to_color(prior, valid_mask, channel_order))

    print(f"Saved soft priors to: {prior_dir.resolve()}")
    print(f"Saved soft-prior visualizations to: {prior_vis_dir.resolve()}")
    print(f"Saved hard-prior visualizations to: {prior_hard_dir.resolve()}")


if __name__ == "__main__":
    main()
