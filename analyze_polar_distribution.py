import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from core.polar_utils import collect_files_by_stem

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_NAMES = {
    1: "pvc",
    2: "metal",
    3: "stone",
}

PLOT_COLORS = {
    1: "#2563eb",
    2: "#dc2626",
    3: "#16a34a",
}

MAX_KDE_SAMPLES = 20000
KDE_GRID_SIZE = 1024


def integrate_curve(y, x):
    if y.size <= 1:
        return float(np.sum(y))
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def load_polar_map(path):
    polar = np.load(str(path)).astype(np.float32)
    if polar.ndim == 3:
        polar = polar[:, :, 0]
    return polar


def load_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise RuntimeError(f"Failed to read label: {path}")
    return np.asarray(label, dtype=np.uint8)


def load_valid_mask(path):
    valid = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if valid is None:
        raise RuntimeError(f"Failed to read valid mask: {path}")
    return np.asarray(valid > 0, dtype=bool)


def summarize_values(values):
    values = np.asarray(values, dtype=np.float32)
    percentiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def erode_binary_mask(mask, edge_erode_px):
    mask = np.asarray(mask, dtype=np.uint8)
    if edge_erode_px <= 0 or not np.any(mask):
        return mask.astype(bool)

    kernel_size = edge_erode_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    if np.any(eroded):
        return eroded.astype(bool)
    return mask.astype(bool)


def _subsample_for_kde(values, class_id):
    if values.size <= MAX_KDE_SAMPLES:
        return values.astype(np.float64)
    rng = np.random.default_rng(1000 + int(class_id))
    indices = rng.choice(values.size, size=MAX_KDE_SAMPLES, replace=False)
    return values[indices].astype(np.float64)


def _gaussian_kernel():
    kernel = np.array([1, 4, 7, 10, 7, 4, 1], dtype=np.float64)
    return kernel / kernel.sum()


def build_density_curve(values, class_id, grid):
    # 优先用 KDE 估计连续分布；样本不足或 scipy 不可用时退回平滑直方图。
    samples = _subsample_for_kde(values, class_id)
    method = "gaussian_kde"

    if gaussian_kde is not None and samples.size >= 2 and float(np.std(samples)) >= 1e-8:
        try:
            density = gaussian_kde(samples)(grid)
        except Exception:
            density = None
    else:
        density = None

    if density is None:
        method = "histogram_fallback"
        hist, edges = np.histogram(
            samples,
            bins=max(128, min(512, grid.size // 2)),
            range=(float(grid[0]), float(grid[-1])),
            density=True,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = np.interp(grid, centers, hist, left=0.0, right=0.0)
        density = np.convolve(density, _gaussian_kernel(), mode="same")

    density = np.nan_to_num(np.asarray(density, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    density = np.maximum(density, 0.0)
    area = integrate_curve(density, grid)
    if not np.isfinite(area) or area <= 1e-12:
        density = np.ones_like(grid, dtype=np.float64)
        area = integrate_curve(density, grid)
        method = f"{method}_uniform_fallback"
    density /= max(area, 1e-12)
    return density.astype(np.float32), method


def find_intersection_from_curves(grid, left_density, right_density, left_median, right_median):
    # 相邻材质的密度曲线交点用作阈值；没有可靠交点时退回两个中位数的中点。
    midpoint = float((left_median + right_median) * 0.5)
    diff = left_density - right_density

    candidates = []
    for idx in range(diff.size - 1):
        x0 = grid[idx]
        x1 = grid[idx + 1]
        y0 = diff[idx]
        y1 = diff[idx + 1]

        if y0 == 0.0:
            candidates.append(float(x0))
            continue

        if y0 * y1 < 0.0:
            denom = y1 - y0
            if abs(denom) < 1e-12:
                candidates.append(float((x0 + x1) * 0.5))
            else:
                ratio = -y0 / denom
                candidates.append(float(x0 + ratio * (x1 - x0)))

    if not candidates:
        return midpoint, "median_midpoint_fallback_no_intersection"

    lower = min(left_median, right_median)
    upper = max(left_median, right_median)
    between_medians = [x for x in candidates if lower <= x <= upper]
    target_candidates = between_medians if between_medians else candidates
    threshold = min(target_candidates, key=lambda x: abs(x - midpoint))
    method = "kde_intersection" if between_medians else "kde_intersection_outside_medians"
    return float(threshold), method


def build_prior_model(pixel_stats, pixel_values_by_class):
    # 按 polar_norm 中位数给材质排序，再为相邻类别学习软先验密度和硬阈值。
    ordered = sorted(CLASS_NAMES.keys(), key=lambda class_id: pixel_stats[str(class_id)]["p50"])
    all_values = np.concatenate([pixel_values_by_class[class_id] for class_id in ordered]).astype(np.float32)

    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    if value_max - value_min < 1e-6:
        value_max = value_min + 1e-3
    grid = np.linspace(value_min, value_max, KDE_GRID_SIZE, dtype=np.float64)

    class_models = {}
    density_cache = {}
    for class_id in ordered:
        density, method = build_density_curve(pixel_values_by_class[class_id], class_id, grid)
        density_cache[class_id] = density
        class_models[str(class_id)] = {
            "name": CLASS_NAMES[class_id],
            "density_method": method,
            "sample_count": int(pixel_values_by_class[class_id].size),
            "density": density.astype(np.float32).tolist(),
        }

    thresholds = []
    for left_class, right_class in zip(ordered[:-1], ordered[1:]):
        left_median = pixel_stats[str(left_class)]["p50"]
        right_median = pixel_stats[str(right_class)]["p50"]
        threshold_value, threshold_method = find_intersection_from_curves(
            grid,
            density_cache[left_class],
            density_cache[right_class],
            left_median,
            right_median,
        )
        thresholds.append(
            {
                "between": [left_class, right_class],
                "between_names": [CLASS_NAMES[left_class], CLASS_NAMES[right_class]],
                "value": float(threshold_value),
                "method": threshold_method,
                "median_midpoint": float((left_median + right_median) * 0.5),
            }
        )

    intervals = {}
    for index, class_id in enumerate(ordered):
        lower = thresholds[index - 1]["value"] if index > 0 else None
        upper = thresholds[index]["value"] if index < len(thresholds) else None
        intervals[str(class_id)] = {
            "name": CLASS_NAMES[class_id],
            "lower": None if lower is None else float(lower),
            "upper": None if upper is None else float(upper),
        }

    return {
        "version": "soft_polar_prior_v1",
        "class_names": {str(class_id): name for class_id, name in CLASS_NAMES.items()},
        "channel_order": ordered,
        "ordered_classes": ordered,
        "ordered_names": [CLASS_NAMES[class_id] for class_id in ordered],
        "grid_min": value_min,
        "grid_max": value_max,
        "grid_size": int(grid.size),
        "grid": grid.astype(np.float32).tolist(),
        "threshold_generation": "adjacent_kde_intersections_with_median_fallback",
        "thresholds": thresholds,
        "intervals": intervals,
        "class_models": class_models,
    }


def save_histogram(pixel_values_by_class, prior_model, output_path):
    ordered = prior_model["ordered_classes"]
    all_values = np.concatenate([pixel_values_by_class[class_id] for class_id in ordered])
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    if value_max - value_min < 1e-6:
        value_max = value_min + 1e-3

    bins = np.linspace(value_min, value_max, 128)
    plt.figure(figsize=(10, 6))
    for class_id in ordered:
        plt.hist(
            pixel_values_by_class[class_id],
            bins=bins,
            density=True,
            alpha=0.35,
            color=PLOT_COLORS[class_id],
            label=f"{CLASS_NAMES[class_id]} ({class_id})",
        )

    for item in prior_model["thresholds"]:
        plt.axvline(item["value"], color="#111827", linestyle="--", linewidth=1.5)

    plt.xlabel("polar_norm")
    plt.ylabel("density")
    plt.title("Filtered Pixel-level Polarization Distribution by Material")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=160)
    plt.close()


def save_boxplot(object_values_by_class, ordered, output_path):
    tick_labels = [f"{CLASS_NAMES[class_id]}\n({class_id})" for class_id in ordered]
    plt.figure(figsize=(8, 6))
    boxplot_kwargs = {
        "patch_artist": True,
        "showfliers": False,
    }
    try:
        boxplot = plt.boxplot(
            [object_values_by_class[class_id] for class_id in ordered],
            tick_labels=tick_labels,
            **boxplot_kwargs,
        )
    except TypeError:
        boxplot = plt.boxplot(
            [object_values_by_class[class_id] for class_id in ordered],
            labels=tick_labels,
            **boxplot_kwargs,
        )

    for patch, class_id in zip(boxplot["boxes"], ordered):
        patch.set_facecolor(PLOT_COLORS[class_id])
        patch.set_alpha(0.45)

    plt.ylabel("object_median_polar_norm")
    plt.title("Object-level Polarization Summary by Material")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze polar_norm distributions and build a soft prior model.")
    parser.add_argument("--polar_dir", default="datasets/MyPolarData/polar_norm")
    parser.add_argument("--label_dir", default="datasets/MyPolarData/labels")
    parser.add_argument("--output_dir", default="analysis/priors/single_material_prior_v2")
    parser.add_argument("--valid_mask_dir", default="datasets/MyPolarData/valid_mask")
    # edge_erode_px 会改变参与统计的像素集合，进而改变 prior_model/thresholds。
    parser.add_argument("--edge_erode_px", type=int, default=5)
    args = parser.parse_args()

    polar_files = collect_files_by_stem(args.polar_dir, (".npy",))
    label_files = collect_files_by_stem(args.label_dir, (".png",))

    stems = sorted(set(polar_files.keys()) & set(label_files.keys()))
    if not stems:
        raise RuntimeError("No matching stems found between polar_dir and label_dir.")

    missing_polar = sorted(set(label_files.keys()) - set(polar_files.keys()))
    missing_label = sorted(set(polar_files.keys()) - set(label_files.keys()))
    if missing_polar:
        preview = ", ".join(missing_polar[:5])
        raise RuntimeError(f"Missing polar maps for stems: {preview}")
    if missing_label:
        preview = ", ".join(missing_label[:5])
        raise RuntimeError(f"Missing labels for stems: {preview}")

    valid_mask_files = None
    if args.valid_mask_dir:
        valid_mask_files = collect_files_by_stem(args.valid_mask_dir, (".png",))
        missing_valid = sorted(set(stems) - set(valid_mask_files.keys()))
        if missing_valid:
            preview = ", ".join(missing_valid[:5])
            raise RuntimeError(f"Missing valid masks for stems: {preview}")

    pixel_values_by_class = {class_id: [] for class_id in CLASS_NAMES}
    object_values_by_class = {class_id: [] for class_id in CLASS_NAMES}
    object_records = []

    for stem in stems:
        polar = load_polar_map(polar_files[stem])
        label = load_label(label_files[stem])

        if polar.shape[:2] != label.shape[:2]:
            raise RuntimeError(f"Shape mismatch for stem {stem}: polar {polar.shape}, label {label.shape}")

        valid_mask = np.ones(label.shape, dtype=bool)
        if valid_mask_files is not None:
            valid_mask &= load_valid_mask(valid_mask_files[stem])
        valid_mask &= np.isfinite(polar)

        for class_id in CLASS_NAMES:
            base_mask = (label == class_id) & valid_mask
            if not np.any(base_mask):
                continue

            filtered_mask = erode_binary_mask(base_mask, args.edge_erode_px)
            values = polar[filtered_mask].astype(np.float32)
            if values.size == 0:
                continue

            pixel_values_by_class[class_id].append(values)
            object_value = float(np.median(values))
            object_values_by_class[class_id].append(object_value)
            object_records.append(
                {
                    "stem": stem,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "object_value_median": object_value,
                    "object_value_mean": float(np.mean(values)),
                    "pixel_count": int(values.size),
                }
            )

    missing_classes = [CLASS_NAMES[class_id] for class_id, chunks in pixel_values_by_class.items() if not chunks]
    if missing_classes:
        raise RuntimeError(f"Missing labeled pixels for classes: {', '.join(missing_classes)}")

    pixel_values_by_class = {
        class_id: np.concatenate(chunks, axis=0).astype(np.float32)
        for class_id, chunks in pixel_values_by_class.items()
    }
    object_values_by_class = {
        class_id: np.asarray(values, dtype=np.float32)
        for class_id, values in object_values_by_class.items()
    }

    stats_pixel = {
        "num_images": len(stems),
        "edge_erode_px": int(args.edge_erode_px),
        "uses_valid_mask": bool(args.valid_mask_dir),
        "class_names": {str(class_id): name for class_id, name in CLASS_NAMES.items()},
        "class_stats": {
            str(class_id): {
                "name": CLASS_NAMES[class_id],
                **summarize_values(values),
            }
            for class_id, values in pixel_values_by_class.items()
        },
    }

    stats_object = {
        "num_images": len(stems),
        "edge_erode_px": int(args.edge_erode_px),
        "uses_valid_mask": bool(args.valid_mask_dir),
        "representative_stat": "median",
        "class_names": {str(class_id): name for class_id, name in CLASS_NAMES.items()},
        "class_stats": {
            str(class_id): {
                "name": CLASS_NAMES[class_id],
                **summarize_values(values),
            }
            for class_id, values in object_values_by_class.items()
        },
        "per_sample": object_records,
    }

    prior_model = build_prior_model(stats_pixel["class_stats"], pixel_values_by_class)
    prior_model["edge_erode_px"] = int(args.edge_erode_px)
    prior_model["uses_valid_mask"] = bool(args.valid_mask_dir)
    prior_model["source_stats"] = {
        "pixel_stats_file": "stats_pixel.json",
        "object_stats_file": "stats_object.json",
    }

    threshold_spec = {
        "class_names": prior_model["class_names"],
        "ordered_classes": prior_model["ordered_classes"],
        "ordered_names": prior_model["ordered_names"],
        "threshold_generation": prior_model["threshold_generation"],
        "thresholds": prior_model["thresholds"],
        "intervals": prior_model["intervals"],
    }

    output_dir = Path(args.output_dir)
    # 分析结果会写出 JSON 模型、统计表和图像；重复运行会覆盖同名文件。
    output_dir.mkdir(parents=True, exist_ok=True)

    save_histogram(pixel_values_by_class, prior_model, output_dir / "polar_histogram.png")
    save_boxplot(object_values_by_class, prior_model["ordered_classes"], output_dir / "polar_boxplot.png")

    with open(output_dir / "stats_pixel.json", "w", encoding="utf-8") as f:
        json.dump(stats_pixel, f, ensure_ascii=False, indent=2)
    with open(output_dir / "stats_object.json", "w", encoding="utf-8") as f:
        json.dump(stats_object, f, ensure_ascii=False, indent=2)
    with open(output_dir / "prior_model.json", "w", encoding="utf-8") as f:
        json.dump(prior_model, f, ensure_ascii=False, indent=2)
    with open(output_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(threshold_spec, f, ensure_ascii=False, indent=2)

    print(f"Saved object-level stats to: {(output_dir / 'stats_object.json').resolve()}")
    print(f"Saved pixel-level stats to: {(output_dir / 'stats_pixel.json').resolve()}")
    print(f"Saved prior model to: {(output_dir / 'prior_model.json').resolve()}")
    print(f"Saved thresholds to: {(output_dir / 'thresholds.json').resolve()}")
    print(f"Saved histogram to: {(output_dir / 'polar_histogram.png').resolve()}")
    print(f"Saved boxplot to: {(output_dir / 'polar_boxplot.png').resolve()}")


if __name__ == "__main__":
    main()
