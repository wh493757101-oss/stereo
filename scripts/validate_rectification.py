"""Measure vertical epipolar alignment on a rectified stereo dataset.

Reads a rectification manifest CSV (as written by
``scripts/rectify_stereo_dataset.py``), samples evenly spaced pairs per
group, matches SIFT features between the rectified left/right images, and
reports vertical residual statistics. Emits a JSON report and exits
non-zero when the dataset fails the QA gate.

Exit policy (fail-closed):
    * an adequately matched pair whose median or p95 absolute vertical
      residual exceeds its threshold, or
    * a sampled pair with fewer than ``--min-matches`` retained inliers
      (reported with ``status: "insufficient"``, never silently passed)

both fail the gate with a non-zero exit code.

Matching is symmetric (mutual best nearest neighbor in both directions,
each passing the ratio test) and the vertical displacements are filtered
with a deterministic median/MAD outlier gate so a few false
correspondences (e.g. across polarization states) cannot dominate p95.
Reported residuals are computed from the raw displacements of the
retained inliers — the estimated vertical offset is never subtracted,
so a coherent vertical shift still fails the gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MANIFEST_REQUIRED_COLUMNS = ("group", "stem", "output_left", "output_right")
DEFAULT_MIN_MATCHES = 25
DEFAULT_MAX_MEDIAN_Y = 1.0
DEFAULT_MAX_P95_Y = 3.0
DEFAULT_SAMPLES_PER_GROUP = 3
DEFAULT_MAX_FEATURES = 4000
RATIO_TEST_THRESHOLD = 0.75
MAD_SCALE = 1.4826
MAD_SIGMA_MULTIPLIER = 3.0
MAD_FLOOR_PX = 2.0


@dataclass(frozen=True)
class ManifestRow:
    group: str
    stem: str
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class Thresholds:
    min_matches: int
    max_median_y: float
    max_p95_y: float


def resolve_manifest_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve ``value`` against ``base_dir`` when relative, else use as-is."""
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_manifest(manifest_path: Path) -> dict[str, list[ManifestRow]]:
    """Read the manifest and group rows by group, preserving manifest order."""
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [c for c in MANIFEST_REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                f"{manifest_path.name}: manifest is missing required columns: "
                f"{', '.join(missing)}"
            )

        rows_by_group: dict[str, list[ManifestRow]] = {}
        for row in reader:
            rows_by_group.setdefault(row["group"], []).append(
                ManifestRow(
                    group=row["group"],
                    stem=row["stem"],
                    left_path=Path(row["output_left"]),
                    right_path=Path(row["output_right"]),
                )
            )

    if not rows_by_group:
        raise ValueError(f"{manifest_path.name}: manifest contains no data rows")
    return rows_by_group


def select_evenly_spaced(rows: list[ManifestRow], count: int) -> list[ManifestRow]:
    """Deterministically pick ``count`` rows evenly spaced across ``rows``."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    n = len(rows)
    if n == 0:
        return []
    if count >= n:
        return list(rows)
    if count == 1:
        return [rows[n // 2]]
    indices = sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})
    return [rows[i] for i in indices]


def create_sift_detector(max_features: int) -> cv2.Feature2D:
    create = getattr(cv2, "SIFT_create", None)
    if create is None:
        raise RuntimeError(
            "OpenCV SIFT is unavailable in this environment "
            f"(cv2 version {cv2.__version__}). Install opencv-python or "
            "opencv-python-headless >= 4.4 (non-free builds of older "
            "versions exclude SIFT)."
        )
    return create(nfeatures=max_features)


def ratio_test_best(
    knn: list[list[cv2.DMatch]], ratio: float
) -> dict[int, int]:
    """Map each query descriptor to its best train index passing the ratio test.

    A single neighbor is accepted as-is; with two neighbors the Lowe ratio
    test must pass.
    """
    best: dict[int, int] = {}
    for pair in knn:
        if len(pair) == 1:
            best[pair[0].queryIdx] = pair[0].trainIdx
        elif pair[0].distance < ratio * pair[1].distance:
            best[pair[0].queryIdx] = pair[0].trainIdx
    return best


def mutual_matches(
    matcher: cv2.DescriptorMatcher,
    descriptors_left: np.ndarray,
    descriptors_right: np.ndarray,
    ratio: float,
) -> list[tuple[int, int]]:
    """Symmetric best-bet matches (left idx, right idx) with ratio test both ways."""
    best_lr = ratio_test_best(
        matcher.knnMatch(descriptors_left, descriptors_right, k=2), ratio
    )
    best_rl = ratio_test_best(
        matcher.knnMatch(descriptors_right, descriptors_left, k=2), ratio
    )
    return [
        (i, j) for i, j in sorted(best_lr.items()) if best_rl.get(j) == i
    ]


def vertical_inlier_mask(
    dy: np.ndarray,
    floor_px: float = MAD_FLOOR_PX,
    k_mad: float = MAD_SIGMA_MULTIPLIER,
) -> np.ndarray:
    """Deterministic median/MAD gate on signed vertical displacement.

    Displacements within ``max(floor_px, k_mad * scaled MAD)`` of the median
    are retained. The offset itself is never subtracted: a coherent shift
    stays intact and is reported as-is.
    """
    med = float(np.median(dy))
    mad = float(np.median(np.abs(dy - med)))
    cutoff = max(floor_px, k_mad * MAD_SCALE * mad)
    return np.abs(dy - med) <= cutoff


def evaluate_pair(
    sift: cv2.Feature2D,
    matcher: cv2.DescriptorMatcher,
    row: ManifestRow,
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    ratio: float,
) -> dict[str, str | int | float | None]:
    """Match features between rectified left/right images and score alignment."""
    keypoints_left, descriptors_left = sift.detectAndCompute(left_gray, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(right_gray, None)

    match_count = 0
    inlier_count = 0
    median_abs_y: float | None = None
    p95_abs_y: float | None = None
    median_disparity_x: float | None = None

    if (
        descriptors_left is not None
        and descriptors_right is not None
        and len(descriptors_left) >= 2
        and len(descriptors_right) >= 2
    ):
        mutual = mutual_matches(matcher, descriptors_left, descriptors_right, ratio)
        match_count = len(mutual)
        if match_count >= 1:
            height, width = left_gray.shape[:2]
            dx = np.array(
                [
                    keypoints_left[i].pt[0] - keypoints_right[j].pt[0]
                    for i, j in mutual
                ]
            )
            dy = np.array(
                [
                    keypoints_left[i].pt[1] - keypoints_right[j].pt[1]
                    for i, j in mutual
                ]
            )
            plausible = (
                np.isfinite(dx)
                & np.isfinite(dy)
                & (dx >= -width)
                & (dx <= width)
                & (dy >= -height)
                & (dy <= height)
            )
            mask = plausible.copy()
            if np.any(plausible):
                mask[plausible] = vertical_inlier_mask(dy[plausible])
            inlier_count = int(mask.sum())
            if inlier_count >= 1:
                abs_dy = np.abs(dy[mask])
                median_abs_y = float(np.median(abs_dy))
                p95_abs_y = float(np.percentile(abs_dy, 95))
                median_disparity_x = float(np.median(dx[mask]))

    return {
        "group": row.group,
        "stem": row.stem,
        "left": str(row.left_path),
        "right": str(row.right_path),
        "status": "ok" if inlier_count >= 1 else "insufficient",
        "match_count": match_count,
        "inlier_count": inlier_count,
        "median_abs_y": median_abs_y,
        "p95_abs_y": p95_abs_y,
        "median_disparity_x": median_disparity_x,
    }


def aggregate_pair_results(
    pairs: list[dict[str, str | int | float | None]],
) -> dict[str, str | int | float | None]:
    """Summarize per-pair results for a group or the whole dataset."""
    ok = [p for p in pairs if p["status"] == "ok"]
    median_ys = [p["median_abs_y"] for p in ok if p["median_abs_y"] is not None]
    p95_ys = [p["p95_abs_y"] for p in ok if p["p95_abs_y"] is not None]
    disparities = [
        p["median_disparity_x"] for p in ok if p["median_disparity_x"] is not None
    ]
    return {
        "pair_count": len(pairs),
        "ok_count": len(ok),
        "insufficient_count": len(pairs) - len(ok),
        "median_of_median_abs_y": (
            float(np.median(median_ys)) if median_ys else None
        ),
        "max_median_abs_y": max(median_ys) if median_ys else None,
        "max_p95_abs_y": max(p95_ys) if p95_ys else None,
        "median_disparity_x": float(np.median(disparities)) if disparities else None,
    }


def build_report(
    results: list[dict[str, str | int | float | None]],
    thresholds: Thresholds,
    **header: str | int,
) -> dict:
    """Assemble the full JSON report with per-group/overall aggregates and decision."""
    groups: dict[str, list[dict[str, str | int | float | None]]] = {}
    for pair in results:
        groups.setdefault(str(pair["group"]), []).append(pair)

    group_sections = [
        {
            "group": group,
            "pairs": pairs,
            "aggregate": aggregate_pair_results(pairs),
        }
        for group, pairs in groups.items()
    ]

    reasons: list[str] = []
    for pair in results:
        label = f"{pair['group']}/{pair['stem']}"
        if pair["inlier_count"] < thresholds.min_matches:
            reasons.append(
                f"{label}: insufficient inliers ({pair['inlier_count']} retained "
                f"of {pair['match_count']} raw matches < {thresholds.min_matches})"
            )
            continue
        assert pair["median_abs_y"] is not None and pair["p95_abs_y"] is not None
        if pair["median_abs_y"] > thresholds.max_median_y:
            reasons.append(
                f"{label}: median |dy| {pair['median_abs_y']:.3f} px exceeds "
                f"{thresholds.max_median_y:.3f} px"
            )
        if pair["p95_abs_y"] > thresholds.max_p95_y:
            reasons.append(
                f"{label}: p95 |dy| {pair['p95_abs_y']:.3f} px exceeds "
                f"{thresholds.max_p95_y:.3f} px"
            )

    return {
        **header,
        "thresholds": {
            "min_matches": thresholds.min_matches,
            "max_median_y": thresholds.max_median_y,
            "max_p95_y": thresholds.max_p95_y,
        },
        "groups": group_sections,
        "overall": aggregate_pair_results(results),
        "decision": {
            "passed": not reasons,
            "failed_pair_count": len(reasons),
            "reasons": reasons,
        },
    }


def run_validation(
    rows_by_group: dict[str, list[ManifestRow]],
    samples_per_group: int,
    thresholds: Thresholds,
    max_features: int = DEFAULT_MAX_FEATURES,
    ratio: float = RATIO_TEST_THRESHOLD,
) -> dict:
    """Sample pairs, match features, and build the QA report."""
    sift = create_sift_detector(max_features)
    matcher = cv2.BFMatcher()

    results: list[dict[str, str | int | float | None]] = []
    for group, rows in rows_by_group.items():
        for row in select_evenly_spaced(rows, samples_per_group):
            left = cv2.imread(str(row.left_path), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(row.right_path), cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:
                raise RuntimeError(
                    f"Failed to read rectified pair: {row.left_path} | {row.right_path}"
                )
            if left.shape != right.shape:
                raise ValueError(
                    f"{row.group}/{row.stem}: left/right shape mismatch "
                    f"{left.shape} vs {right.shape}"
                )
            results.append(evaluate_pair(sift, matcher, row, left, right, ratio))

    return build_report(
        results,
        thresholds,
        samples_per_group=samples_per_group,
        sift_max_features=max_features,
        ratio_test_threshold=ratio,
        outlier_filter={
            "method": "median_mad_on_dy",
            "sigma_multiplier": MAD_SIGMA_MULTIPLIER,
            "floor_px": MAD_FLOOR_PX,
            "offset_subtracted": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="QA-gate vertical epipolar alignment of a rectified stereo dataset"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Rectification manifest CSV (rectify_manifest.csv)",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=DEFAULT_SAMPLES_PER_GROUP,
        help="Evenly spaced pairs to sample per group (default %(default)s)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=DEFAULT_MIN_MATCHES,
        help="Pairs with fewer retained inlier matches are reported as "
        "insufficient and fail the gate (default %(default)s)",
    )
    parser.add_argument(
        "--max-median-y",
        type=float,
        default=DEFAULT_MAX_MEDIAN_Y,
        help="Maximum allowed median |dy| in px (default %(default)s)",
    )
    parser.add_argument(
        "--max-p95-y",
        type=float,
        default=DEFAULT_MAX_P95_Y,
        help="Maximum allowed p95 |dy| in px (default %(default)s)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="SIFT nfeatures limit (default %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON report to this path",
    )
    args = parser.parse_args(argv)

    base_dir = Path.cwd()
    manifest_path = resolve_manifest_path(args.manifest, base_dir)
    rows_by_group = load_manifest(manifest_path)
    thresholds = Thresholds(
        min_matches=args.min_matches,
        max_median_y=args.max_median_y,
        max_p95_y=args.max_p95_y,
    )

    report = run_validation(
        rows_by_group=rows_by_group,
        samples_per_group=args.samples_per_group,
        thresholds=thresholds,
        max_features=args.max_features,
    )

    if args.output is not None:
        output_path = resolve_manifest_path(args.output, base_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Report: {output_path}")

    overall = report["overall"]
    print(
        f"Pairs: {overall['pair_count']} sampled, {overall['ok_count']} ok, "
        f"{overall['insufficient_count']} insufficient"
    )
    decision = report["decision"]
    if decision["passed"]:
        print("Decision: PASS")
        return 0
    print("Decision: FAIL")
    for reason in decision["reasons"]:
        print(f"  - {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
