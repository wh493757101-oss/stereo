"""Paired gray-vs-polar ablation evaluation for Model B classifiers.

Loads ``dataset_manifest.csv`` from the gray and polar classification
roots, verifies the two roots describe exactly the same samples (same
relative paths, classes, groups), runs both classifiers on the chosen
split, and emits a JSON report with per-model metrics, the paired
macro-F1 difference, and a capture-group bootstrap confidence interval.

The decision gate claims a polar gain only when:
  1. macro_f1(polar) - macro_f1(gray) >= --min-gain (default 0.03), and
  2. the bootstrap CI lower bound of that difference is > 0.
Otherwise the decision is "use gray-only / no proven gain".

Usage:
    python scripts/eval_ablation.py \
        --gray-weights runs/train/run_20260913_initial/model_b-gray/weights/best.pt \
        --polar-weights runs/train/run_20260913_initial/model_b-polar/weights/best.pt \
        --output analysis/runs/<run-id>/model_b_ablation_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GRAY_ROOT = "datasets/underwater_cls_gray_v2"
DEFAULT_POLAR_ROOT = "datasets/underwater_cls_polar_v2"
DEFAULT_GRAY_WEIGHTS = "runs/train/run_20260913_initial/model_b-gray/weights/best.pt"
DEFAULT_POLAR_WEIGHTS = "runs/train/run_20260913_initial/model_b-polar/weights/best.pt"
DEFAULT_MIN_GAIN = 0.03
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 2026
DEFAULT_CI_LEVEL = 0.95


class ManifestMismatchError(ValueError):
    """Raised when the gray and polar manifests do not describe the same data."""


class ClassMappingError(ValueError):
    """Raised when predictions cannot be mapped to canonical manifest class ids."""


@dataclass(frozen=True)
class PairedSample:
    sample_name: str
    split: str
    group_name: str
    class_id: int
    class_name: str
    relative_path: str

    @property
    def label(self) -> int:
        return self.class_id


def load_manifest(path: str | Path) -> dict[str, dict[str, str]]:
    """Read a dataset_manifest.csv into {sample_name: row}."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["sample_name"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def pair_manifests(
    gray_rows: dict[str, dict[str, str]],
    polar_rows: dict[str, dict[str, str]],
) -> list[PairedSample]:
    """Validate and pair manifest rows; raises on any mismatch."""
    problems: list[str] = []

    only_gray = sorted(set(gray_rows) - set(polar_rows))
    only_polar = sorted(set(polar_rows) - set(gray_rows))
    if only_gray:
        problems.append(f"samples only in gray manifest: {only_gray[:5]}")
    if only_polar:
        problems.append(f"samples only in polar manifest: {only_polar[:5]}")

    pairs: list[PairedSample] = []
    for name in sorted(set(gray_rows) & set(polar_rows)):
        g, p = gray_rows[name], polar_rows[name]
        for field in ("split", "class_id", "class_name", "group_name"):
            if g[field] != p[field]:
                problems.append(f"{name}: {field} mismatch "
                                f"({g[field]!r} vs {p[field]!r})")
        if g["gray_path"] != p["polar_path"]:
            problems.append(f"{name}: relative path mismatch "
                            f"({g['gray_path']!r} vs {p['polar_path']!r})")
        pairs.append(PairedSample(
            sample_name=name,
            split=g["split"],
            group_name=g["group_name"],
            class_id=int(g["class_id"]),
            class_name=g["class_name"],
            relative_path=g["gray_path"],
        ))

    if problems:
        raise ManifestMismatchError(
            "gray/polar manifests do not match; first problems: " + "; ".join(problems)
        )
    return pairs


def select_split(pairs: list[PairedSample], split: str) -> list[PairedSample]:
    selected = [p for p in pairs if p.split == split]
    if not selected:
        raise ValueError(f"No samples found for split '{split}'.")
    return selected


def build_canonical_id_map(pairs: list[PairedSample]) -> dict[str, int]:
    """Build the manifest's canonical one-to-one {class_name: class_id} map.

    Raises ClassMappingError when a name maps to multiple ids, an id maps
    to multiple names, or the ids are not contiguous 0..n-1 (a gap would
    silently skew per-class metrics).
    """
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    for p in pairs:
        seen_id = name_to_id.get(p.class_name)
        if seen_id is not None and seen_id != p.class_id:
            raise ClassMappingError(
                f"class name {p.class_name!r} maps to conflicting ids "
                f"{seen_id} and {p.class_id} in the manifest")
        seen_name = id_to_name.get(p.class_id)
        if seen_name is not None and seen_name != p.class_name:
            raise ClassMappingError(
                f"class id {p.class_id} maps to conflicting names "
                f"{seen_name!r} and {p.class_name!r} in the manifest")
        name_to_id[p.class_name] = p.class_id
        id_to_name[p.class_id] = p.class_name

    ids = sorted(id_to_name)
    if ids != list(range(len(ids))):
        raise ClassMappingError(
            f"manifest class ids must be contiguous 0..n-1, got {ids}")
    return name_to_id


def map_predictions_to_canonical(
    results: list,
    image_paths: list[str],
    canonical: dict[str, int],
    model_label: str,
) -> list[int]:
    """Translate ClassificationResult.top1_name into canonical manifest ids.

    Checkpoint numeric ids (top1_id) are ignored on purpose: Ultralytics
    assigns them from alphabetical folder order, which need not match the
    manifest's class_id space.
    """
    preds: list[int] = []
    for path, result in zip(image_paths, results):
        name = result.top1_name
        if name is None:
            raise ClassMappingError(
                f"{model_label} model returned no class name for {path}")
        if name not in canonical:
            hint = ("; this looks like the checkpoint 'names' fallback for an "
                    "unmapped id; checkpoint names do not cover the manifest "
                    "classes" if re.fullmatch(r"class_\d+", name) else "")
            raise ClassMappingError(
                f"{model_label} model predicted unknown class name {name!r} "
                f"for {path}; expected one of {sorted(canonical)}{hint}")
        preds.append(canonical[name])
    return preds


def confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict:
    """Per-class precision/recall/F1, macro-F1, accuracy, confusion matrix.

    Classes with zero support get precision = recall = f1 = 0.0.
    """
    n_classes = max(len(class_names or []), max(y_true) + 1, max(y_pred) + 1)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    matrix = confusion_matrix(y_true, y_pred, n_classes)
    per_class = {}
    f1_scores = []
    for c in range(n_classes):
        tp = int(matrix[c, c])
        fp = int(matrix[:, c].sum() - tp)
        fn = int(matrix[c, :].sum() - tp)
        support = int(matrix[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[class_names[c]] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        f1_scores.append(f1)

    n = len(y_true)
    correct = int(sum(1 for t, p in zip(y_true, y_pred) if t == p))
    return {
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean(f1_scores)),
        "accuracy": correct / n if n else 0.0,
        "n_samples": n,
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        denom_p, denom_r = tp + fp, tp + fn
        precision = tp / denom_p if denom_p else 0.0
        recall = tp / denom_r if denom_r else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)
    return float(np.mean(f1s))


def group_bootstrap_diff_ci(
    y_true: list[int],
    y_pred_a: list[int],
    y_pred_b: list[int],
    group_names: list[str],
    *,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
    n_classes: int | None = None,
) -> dict:
    """Deterministic capture-group bootstrap of macro-F1 diff (B - A).

    Groups are sampled with replacement; every sample of a sampled group
    is included, so adjacent frames never get bootstrapped individually.
    """
    if n_classes is None:
        n_classes = max(max(y_true), max(y_pred_a), max(y_pred_b)) + 1

    y_true_a = np.asarray(y_true)
    pred_a = np.asarray(y_pred_a)
    pred_b = np.asarray(y_pred_b)

    groups: dict[str, list[int]] = defaultdict(list)
    for idx, g in enumerate(group_names):
        groups[g].append(idx)
    group_arrays = [np.asarray(idxs) for idxs in groups.values()]
    n_groups = len(group_arrays)
    if n_groups == 0:
        raise ValueError("No capture groups to bootstrap.")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        chosen = rng.integers(0, n_groups, size=n_groups)
        idxs = np.concatenate([group_arrays[j] for j in chosen])
        diffs[i] = _macro_f1(y_true_a[idxs], pred_b[idxs], n_classes) - \
            _macro_f1(y_true_a[idxs], pred_a[idxs], n_classes)

    alpha = (1.0 - ci_level) / 2.0
    low, high = np.percentile(diffs, [100 * alpha, 100 * (1.0 - alpha)])
    return {
        "resamples": n_bootstraps,
        "seed": seed,
        "ci_level": ci_level,
        "ci_low": float(low),
        "ci_high": float(high),
        "mean_diff": float(diffs.mean()),
        "n_groups": n_groups,
    }


def make_decision(
    gray_macro_f1: float,
    polar_macro_f1: float,
    ci_low: float,
    *,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> dict:
    """Claim gate: gain >= min_gain AND CI lower bound > 0."""
    diff = polar_macro_f1 - gray_macro_f1
    passed = diff >= min_gain and ci_low > 0
    if passed:
        decision = "use_polar_aided"
        reason = (f"macro-F1 gain {diff:.4f} >= {min_gain:.2f} and "
                  f"CI lower bound {ci_low:.4f} > 0")
    else:
        decision = "use_gray_only"
        reason = (f"no proven gain: macro-F1 diff {diff:.4f} "
                  f"(need >= {min_gain:.2f}) with CI lower bound {ci_low:.4f} "
                  "(need > 0)")
    return {
        "gate": "pass" if passed else "fail",
        "decision": decision,
        "reason": reason,
        "macro_f1_diff": diff,
        "min_gain": min_gain,
    }


def predict_all(model, image_paths: list[Path], root: Path) -> list:
    """Run a ClassificationModel over crops; fail fast on empty results."""
    import cv2

    results = []
    for path in image_paths:
        image = cv2.imread(str(root / path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {root / path}")
        result = model.predict(image)
        if not result.valid or result.top1_id is None:
            raise ValueError(f"Classifier returned no probabilities for {path}")
        results.append(result)
    return results


def evaluate(
    gray_root: Path,
    polar_root: Path,
    gray_weights: str,
    polar_weights: str,
    *,
    split: str = "test",
    device: str = "cpu",
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = DEFAULT_SEED,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> dict:
    from models.classification import ClassificationModel

    gray_rows = load_manifest(gray_root / "dataset_manifest.csv")
    polar_rows = load_manifest(polar_root / "dataset_manifest.csv")
    pairs = select_split(pair_manifests(gray_rows, polar_rows), split)

    canonical = build_canonical_id_map(pairs)
    class_names = sorted(canonical, key=canonical.get)
    n_classes = len(canonical)

    pairs = sorted(pairs, key=lambda p: p.sample_name)
    gray_model = ClassificationModel(gray_weights, device=device)
    polar_model = ClassificationModel(polar_weights, device=device)

    paths = [p.relative_path for p in pairs]
    y_true = [p.class_id for p in pairs]
    gray_results = predict_all(gray_model, paths, gray_root)
    polar_results = predict_all(polar_model, paths, polar_root)
    y_pred_gray = map_predictions_to_canonical(gray_results, paths, canonical, "gray")
    y_pred_polar = map_predictions_to_canonical(polar_results, paths, canonical, "polar")

    gray_metrics = compute_metrics(y_true, y_pred_gray, class_names)
    polar_metrics = compute_metrics(y_true, y_pred_polar, class_names)

    bootstrap = group_bootstrap_diff_ci(
        y_true, y_pred_gray, y_pred_polar,
        [p.group_name for p in pairs],
        n_bootstraps=n_bootstraps, seed=seed, n_classes=n_classes,
    )
    decision = make_decision(
        gray_metrics["macro_f1"], polar_metrics["macro_f1"],
        bootstrap["ci_low"], min_gain=min_gain,
    )

    return {
        "split": split,
        "n_samples": len(pairs),
        "class_names": class_names,
        "gray": {"weights": gray_weights, **gray_metrics},
        "polar": {"weights": polar_weights, **polar_metrics},
        "paired_difference": {
            "macro_f1_diff": polar_metrics["macro_f1"] - gray_metrics["macro_f1"],
        },
        "bootstrap": bootstrap,
        "decision": decision,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gray-root", default=DEFAULT_GRAY_ROOT)
    parser.add_argument("--polar-root", default=DEFAULT_POLAR_ROOT)
    parser.add_argument("--gray-weights", default=DEFAULT_GRAY_WEIGHTS)
    parser.add_argument("--polar-weights", default=DEFAULT_POLAR_WEIGHTS)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu",
                        help="'cpu' or CUDA id; CUDA requires an available GPU.")
    parser.add_argument("--n-bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN)
    parser.add_argument("--output", default=None, help="Write JSON report here.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(
        Path(args.gray_root), Path(args.polar_root),
        args.gray_weights, args.polar_weights,
        split=args.split, device=args.device,
        n_bootstraps=args.n_bootstraps, seed=args.seed, min_gain=args.min_gain,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"Report written to {output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
