import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from core.polar_utils import IMAGE_EXTENSIONS, collect_files_by_stem, load_rgb_image


CLASS_NAMES = {
    0: "background",
    1: "pvc",
    2: "metal",
    3: "stone",
}

MASK_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (0, 255, 0),
}


def load_polar_map(path):
    polar = np.load(str(path)).astype(np.float32)
    if polar.ndim == 3:
        polar = polar[:, :, 0]
    return polar


def load_polar_prior(path):
    prior = np.load(str(path)).astype(np.float32)
    if prior.ndim == 2:
        prior = prior[:, :, None]
    elif prior.ndim == 3 and prior.shape[0] == 3 and prior.shape[-1] != 3:
        prior = np.transpose(prior, (1, 2, 0))
    if prior.ndim != 3 or prior.shape[2] < 3:
        raise RuntimeError(f"polar prior must have 3 channels, got shape {prior.shape} from {path}")
    return prior[:, :, :3]


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return np.asarray(mask, dtype=np.uint8)


def load_threshold_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    ordered_classes = [int(class_id) for class_id in spec["ordered_classes"]]
    thresholds = [float(item["value"]) for item in spec["thresholds"]]
    if len(ordered_classes) != len(thresholds) + 1:
        raise RuntimeError("Invalid thresholds.json: ordered_classes and thresholds lengths do not match.")

    return ordered_classes, np.asarray(thresholds, dtype=np.float32)


def colorize_mask(mask):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, bgr in MASK_COLORS.items():
        color[mask == class_id] = bgr
    return color


def build_overlay(image_rgb, color_mask):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image_bgr, 0.6, color_mask, 0.4, 0)


def classify_foreground_with_thresholds(polar, foreground_mask, ordered_classes, thresholds):
    # 旧版硬阈值 baseline：只在已有前景 mask 内分类，背景保持 0。
    pred = np.zeros(foreground_mask.shape, dtype=np.uint8)
    if not np.any(foreground_mask):
        return pred

    foreground_values = polar[foreground_mask]
    class_indices = np.digitize(foreground_values, thresholds, right=False)
    ordered_classes_np = np.asarray(ordered_classes, dtype=np.uint8)
    pred[foreground_mask] = ordered_classes_np[class_indices]
    return pred


def classify_foreground_with_prior(prior, foreground_mask):
    # 推荐 baseline：对 soft prior 做 argmax，同样只修改前景区域的预测值。
    pred = np.zeros(foreground_mask.shape, dtype=np.uint8)
    if not np.any(foreground_mask):
        return pred

    hard_pred = np.argmax(prior, axis=2).astype(np.uint8) + 1
    pred[foreground_mask] = hard_pred[foreground_mask]
    return pred


def ensure_matching_stems(stems, available_files, name):
    missing = sorted(set(stems) - set(available_files.keys()))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Missing {name} files for stems: {preview}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Auxiliary visualization baseline for polarization priors. "
            "Recommended mode reads polar_prior and visualizes argmax(prior) inside an existing foreground mask. "
            "Legacy threshold mode is still supported for comparison."
        )
    )
    parser.add_argument("--mask_dir", default="datasets/MyPolarData/labels")
    parser.add_argument("--output_dir", default="baseline/polar_prior_vis_v2")
    parser.add_argument("--image_dir", default="datasets/MyPolarData/left")
    parser.add_argument("--polar_prior_dir", default="datasets/MyPolarData/polar_prior")
    parser.add_argument("--polar_dir", default=None)
    parser.add_argument("--thresholds_json", default=None)
    args = parser.parse_args()

    use_prior_mode = args.polar_prior_dir is not None
    use_threshold_mode = args.polar_dir is not None and args.thresholds_json is not None
    if not use_prior_mode and not use_threshold_mode:
        raise RuntimeError(
            "Provide either --polar_prior_dir for the recommended prior visualization mode, "
            "or both --polar_dir and --thresholds_json for the legacy threshold mode."
        )

    mask_files = collect_files_by_stem(args.mask_dir, (".png",))
    if not mask_files:
        raise RuntimeError("No masks found in mask_dir.")

    if use_prior_mode:
        feature_files = collect_files_by_stem(args.polar_prior_dir, (".npy",))
        mode_name = "polar_prior_argmax"
    else:
        feature_files = collect_files_by_stem(args.polar_dir, (".npy",))
        mode_name = "threshold_baseline"

    stems = sorted(set(mask_files.keys()) & set(feature_files.keys()))
    if not stems:
        raise RuntimeError("No matching stems found between feature inputs and mask_dir.")

    ensure_matching_stems(stems, mask_files, "mask")
    ensure_matching_stems(stems, feature_files, "feature")

    image_files = None
    if args.image_dir:
        image_files = collect_files_by_stem(args.image_dir, IMAGE_EXTENSIONS)
        ensure_matching_stems(stems, image_files, "image")

    ordered_classes = None
    thresholds = None
    if use_threshold_mode:
        ordered_classes, thresholds = load_threshold_spec(args.thresholds_json)

    output_dir = Path(args.output_dir)
    mask_output_dir = output_dir / "mask"
    color_output_dir = output_dir / "color"
    overlay_output_dir = output_dir / "overlay"
    output_dir.mkdir(parents=True, exist_ok=True)
    # 输出 raw mask、彩色图、overlay 和 summary，用于快速检查先验是否符合直觉。
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    color_output_dir.mkdir(parents=True, exist_ok=True)
    if image_files is not None:
        overlay_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_json = []

    for stem in stems:
        mask = load_mask(mask_files[stem])
        foreground_mask = mask > 0

        if use_prior_mode:
            prior = load_polar_prior(feature_files[stem])
            if prior.shape[:2] != mask.shape[:2]:
                raise RuntimeError(f"Shape mismatch for stem {stem}: prior {prior.shape}, mask {mask.shape}")
            valid_prior_mask = np.isfinite(prior).all(axis=2) & (np.sum(prior, axis=2) > 0)
            # 无效先验像素不参与分类，避免把全 0 prior 当成某个材质。
            foreground_mask = foreground_mask & valid_prior_mask
            pred_mask = classify_foreground_with_prior(prior, foreground_mask)
        else:
            polar = load_polar_map(feature_files[stem])
            if polar.shape[:2] != mask.shape[:2]:
                raise RuntimeError(f"Shape mismatch for stem {stem}: polar {polar.shape}, mask {mask.shape}")
            foreground_mask = foreground_mask & np.isfinite(polar)
            pred_mask = classify_foreground_with_thresholds(polar, foreground_mask, ordered_classes, thresholds)

        color_mask = colorize_mask(pred_mask)

        cv2.imwrite(str(mask_output_dir / f"{stem}.png"), pred_mask)
        cv2.imwrite(str(color_output_dir / f"{stem}.png"), color_mask)

        if image_files is not None:
            overlay = build_overlay(load_rgb_image(image_files[stem]), color_mask)
            cv2.imwrite(str(overlay_output_dir / f"{stem}.png"), overlay)

        counts = {class_id: int(np.sum(pred_mask == class_id)) for class_id in CLASS_NAMES}
        foreground_pixels = int(np.sum(pred_mask > 0))
        row = {
            "stem": stem,
            "background": counts[0],
            "pvc": counts[1],
            "metal": counts[2],
            "stone": counts[3],
            "foreground_pixels": foreground_pixels,
            "mode": mode_name,
        }
        summary_rows.append(row)
        summary_json.append(
            {
                "stem": stem,
                "mode": mode_name,
                "counts": {CLASS_NAMES[class_id]: count for class_id, count in counts.items()},
                "predicted_classes": [CLASS_NAMES[class_id] for class_id in CLASS_NAMES if counts[class_id] > 0],
            }
        )

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stem", "background", "pvc", "metal", "stone", "foreground_pixels", "mode"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"Saved raw masks to: {mask_output_dir.resolve()}")
    print(f"Saved color masks to: {color_output_dir.resolve()}")
    if image_files is not None:
        print(f"Saved overlays to: {overlay_output_dir.resolve()}")
    print(f"Saved summary to: {(output_dir / 'summary.csv').resolve()}")
    print(f"Mode: {mode_name}")


if __name__ == "__main__":
    main()
