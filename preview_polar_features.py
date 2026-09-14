import argparse
from pathlib import Path

import cv2
import numpy as np

from core.polar_utils import IMAGE_EXTENSIONS, load_rgb_image


def add_title(image, title):
    canvas = image.copy()
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return canvas


def load_first_stem(folder):
    files = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No images found in {folder}")
    return files[0].stem


def load_polar_map(path):
    polar = np.load(path).astype(np.float32)
    if polar.ndim == 3:
        polar = polar[:, :, 0]
    return polar


def polar_to_color(polar):
    polar = np.clip(polar, 0.0, 1.0)
    polar_u8 = (polar * 255.0).astype(np.uint8)
    return cv2.applyColorMap(polar_u8, cv2.COLORMAP_VIRIDIS)


def valid_mask_to_color(valid_mask):
    return cv2.cvtColor(valid_mask, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description="Build a side-by-side preview for left/right_warp/valid_mask/polar_norm.")
    parser.add_argument("--stem", default=None)
    parser.add_argument("--left_dir", default="datasets/MyPolarData/left")
    parser.add_argument("--right_warp_dir", default="datasets/MyPolarData/right_warp")
    parser.add_argument("--valid_mask_dir", default="datasets/MyPolarData/valid_mask")
    parser.add_argument("--polar_dir", default="datasets/MyPolarData/polar_norm")
    parser.add_argument("--output", default="predict_results/polar_feature_preview.png")
    args = parser.parse_args()

    stem = args.stem or load_first_stem(args.left_dir)

    # 预览图从已生成的 polar/right_warp/mask 读取，不会重新计算特征。
    left = load_rgb_image(Path(args.left_dir) / f"{stem}.png")
    right_warp = load_rgb_image(Path(args.right_warp_dir) / f"{stem}.png")
    valid_mask = cv2.imread(str(Path(args.valid_mask_dir) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    polar = load_polar_map(Path(args.polar_dir) / f"{stem}.npy")

    if valid_mask is None:
        raise RuntimeError(f"Failed to read valid mask for stem: {stem}")

    expected_shape = left.shape[:2]
    for name, image in (
        ("right_warp", right_warp),
        ("valid_mask", valid_mask),
        ("polar_norm", polar),
    ):
        if image.shape[:2] != expected_shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: got {image.shape[:2]}, expected {expected_shape}."
            )

    left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    right_warp_bgr = cv2.cvtColor(right_warp, cv2.COLOR_RGB2BGR)
    valid_bgr = valid_mask_to_color(valid_mask)
    polar_bgr = polar_to_color(polar)

    top = np.hstack([add_title(left_bgr, f"Left | {stem}"), add_title(right_warp_bgr, "Right Warp")])
    bottom = np.hstack([add_title(valid_bgr, "Valid Mask"), add_title(polar_bgr, "Polar Norm")])
    preview = np.vstack([top, bottom])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 只写一张拼接检查图，便于肉眼确认极线对齐和有效区域。
    cv2.imwrite(str(output_path), preview)
    print(f"Saved preview to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
