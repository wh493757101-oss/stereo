import csv
import json
import tempfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.validate_rectification as vr
from scripts.validate_rectification import (
    aggregate_pair_results,
    select_evenly_spaced,
    vertical_inlier_mask,
)


@pytest.fixture
def workdir() -> Iterator[Path]:
    # The project pins --basetemp=tests/_tmp_pytest, which is currently
    # undeletable (WinError 5), so use a tempfile.TemporaryDirectory instead.
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def make_textured_image(seed: int, width: int = 320, height: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 128, dtype=np.uint8)
    for _ in range(300):
        x = int(rng.integers(6, width - 6))
        y = int(rng.integers(6, height - 6))
        radius = int(rng.integers(2, 5))
        value = int(rng.integers(0, 256))
        cv2.circle(img, (x, y), radius, value, -1)
    return img


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.full_like(img, 128)
    h, w = img.shape
    out[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = img[
        max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
    ]
    return out


def write_manifest(
    path: Path,
    base_dir: Path,
    pairs: list[tuple[str, str, int, int]],
    absolute: bool = True,
) -> None:
    fieldnames = ["group", "stem", "output_left", "output_right"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, stem, dx, dy in pairs:
            left = make_textured_image(zlib.crc32(f"{group}_{stem}".encode()))
            right = shift_image(left, dx, dy)
            left_rel = Path("images") / f"{group}_{stem}_left.png"
            right_rel = Path("images") / f"{group}_{stem}_right.png"
            (base_dir / "images").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(base_dir / left_rel), left)
            cv2.imwrite(str(base_dir / right_rel), right)
            if absolute:
                left_col = (base_dir / left_rel).as_posix()
                right_col = (base_dir / right_rel).as_posix()
            else:
                left_col, right_col = left_rel.as_posix(), right_rel.as_posix()
            writer.writerow(
                {
                    "group": group,
                    "stem": stem,
                    "output_left": left_col,
                    "output_right": right_col,
                }
            )


def run_cli(workdir: Path, pairs: list[tuple[str, str, int, int]], extra: list[str]) -> tuple[int, Path]:
    manifest_path = workdir / "rectify_manifest.csv"
    write_manifest(manifest_path, workdir, pairs)
    output_path = workdir / "report.json"
    code = vr.main(
        [
            "--manifest",
            manifest_path.as_posix(),
            "--output",
            output_path.as_posix(),
            "--min-matches",
            "20",
            *extra,
        ]
    )
    return code, output_path


def first_pair(report: dict) -> dict:
    return report["groups"][0]["pairs"][0]


def test_perfect_horizontal_translation_passes(workdir: Path):
    code, output_path = run_cli(workdir, [("g1", "000", 24, 0)], [])

    assert code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    pair = first_pair(report)
    assert pair["status"] == "ok"
    assert pair["match_count"] >= 20
    assert pair["median_abs_y"] < 1.0
    assert pair["p95_abs_y"] < 3.0
    assert report["decision"]["passed"] is True
    assert report["decision"]["reasons"] == []


def test_injected_vertical_shift_fails(workdir: Path):
    code, output_path = run_cli(workdir, [("g1", "000", 24, 6)], [])

    assert code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    pair = first_pair(report)
    assert pair["status"] == "ok"
    assert pair["median_abs_y"] > 5.0
    assert report["decision"]["passed"] is False
    assert any("median |dy|" in reason for reason in report["decision"]["reasons"])


def test_p95_threshold_alone_fails(workdir: Path):
    code, output_path = run_cli(
        workdir, [("g1", "000", 24, 6)], ["--max-median-y", "10.0", "--max-p95-y", "3.0"]
    )

    assert code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert first_pair(report)["median_abs_y"] < 10.0
    assert any("p95 |dy|" in reason for reason in report["decision"]["reasons"])


def test_deterministic_sampling():
    rows = [
        vr.ManifestRow(group="g", stem=str(i), left_path=Path("l"), right_path=Path("r"))
        for i in range(10)
    ]

    picked = select_evenly_spaced(rows, 3)

    assert [row.stem for row in picked] == ["0", "4", "9"]
    assert [row.stem for row in select_evenly_spaced(rows, 3)] == ["0", "4", "9"]


def test_sampling_edge_cases():
    rows = [
        vr.ManifestRow(group="g", stem=str(i), left_path=Path("l"), right_path=Path("r"))
        for i in range(5)
    ]

    assert select_evenly_spaced(rows, 10) == rows
    assert select_evenly_spaced(rows, 1) == [rows[2]]
    assert select_evenly_spaced([], 3) == []
    with pytest.raises(ValueError):
        select_evenly_spaced(rows, 0)


def test_insufficient_features_are_reported_not_passed(workdir: Path):
    manifest_path = workdir / "rectify_manifest.csv"
    left_rel = Path("images") / "flat_left.png"
    right_rel = Path("images") / "flat_right.png"
    (workdir / "images").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(workdir / left_rel), np.full((240, 320), 128, dtype=np.uint8))
    cv2.imwrite(str(workdir / right_rel), np.full((240, 320), 128, dtype=np.uint8))
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["group", "stem", "output_left", "output_right"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "group": "flat",
                "stem": "000",
                "output_left": (workdir / left_rel).as_posix(),
                "output_right": (workdir / right_rel).as_posix(),
            }
        )

    output_path = workdir / "report.json"
    code = vr.main(
        ["--manifest", manifest_path.as_posix(), "--output", output_path.as_posix()]
    )

    assert code == 1
    report = json.loads(output_path.read_text(encoding="utf-8"))
    pair = first_pair(report)
    assert pair["status"] == "insufficient"
    assert pair["match_count"] == 0
    assert pair["median_abs_y"] is None
    assert pair["p95_abs_y"] is None
    assert report["decision"]["passed"] is False
    assert any("insufficient" in reason for reason in report["decision"]["reasons"])
    assert report["overall"]["insufficient_count"] == 1


def test_json_schema_and_relative_path_resolution(workdir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(workdir)
    write_manifest(
        workdir / "rectify_manifest.csv",
        workdir,
        [("g1", "000", 24, 0), ("g1", "001", 24, 0), ("g2", "000", 24, 0)],
        absolute=False,
    )

    output_path = workdir / "out" / "report.json"
    code = vr.main(
        [
            "--manifest",
            "rectify_manifest.csv",
            "--output",
            "out/report.json",
            "--samples-per-group",
            "2",
            "--min-matches",
            "20",
        ]
    )
    assert code == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["samples_per_group"] == 2
    assert report["thresholds"] == {
        "min_matches": 20,
        "max_median_y": 1.0,
        "max_p95_y": 3.0,
    }
    assert {section["group"] for section in report["groups"]} == {"g1", "g2"}
    for section in report["groups"]:
        assert set(section["aggregate"]) == {
            "pair_count",
            "ok_count",
            "insufficient_count",
            "median_of_median_abs_y",
            "max_median_abs_y",
            "max_p95_abs_y",
            "median_disparity_x",
        }
        for pair in section["pairs"]:
            assert pair["status"] == "ok"
            assert isinstance(pair["match_count"], int)
            assert isinstance(pair["median_abs_y"], float)
            assert isinstance(pair["p95_abs_y"], float)
            assert isinstance(pair["median_disparity_x"], float)
    assert set(report["overall"]) == set(report["groups"][0]["aggregate"])
    assert report["decision"]["passed"] is True
    assert output_path.exists()


def test_missing_manifest_columns_raise(workdir: Path):
    manifest_path = workdir / "bad.csv"
    manifest_path.write_text("group,stem\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        vr.load_manifest(manifest_path)


def test_sift_unavailable_gives_clear_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delattr(cv2, "SIFT_create", raising=False)

    with pytest.raises(RuntimeError, match="SIFT is unavailable"):
        vr.create_sift_detector(100)


def test_aggregate_handles_all_insufficient():
    pairs = [
        {
            "group": "g",
            "stem": "000",
            "status": "insufficient",
            "match_count": 0,
            "inlier_count": 0,
            "median_abs_y": None,
            "p95_abs_y": None,
            "median_disparity_x": None,
        }
    ]

    agg = aggregate_pair_results(pairs)

    assert agg["pair_count"] == 1
    assert agg["ok_count"] == 0
    assert agg["insufficient_count"] == 1
    assert agg["max_median_abs_y"] is None


class OrderedStubSift:
    """Returns pre-computed (keypoints, descriptors) per detectAndCompute call."""

    def __init__(self, results: list[tuple[list, np.ndarray]]):
        self._results = list(results)

    def detectAndCompute(self, _img, _mask):
        return self._results.pop(0)


def one_hot_descriptors(count: int, dim: int = 32) -> np.ndarray:
    desc = np.zeros((count, dim), dtype=np.uint8)
    for i in range(count):
        desc[i, i % dim] = 255
    return desc


def make_keypoint(x: float, y: float) -> cv2.KeyPoint:
    return cv2.KeyPoint(float(x), float(y), 5.0)


def synthetic_matches_with_outliers(
    true_count: int, outlier_dys: list[float], disparity_x: float = 24.0
):
    """Build left/right keypoint sets whose mutual matches include outliers.

    The first ``len(outlier_dys)`` left features match right keypoints placed
    ``outlier_dys[i]`` px away vertically (cross-polarization look-alikes);
    the remaining features match at dy = 0. All horizontal disparity is
    ``disparity_x``. Returns (keypoints_left, descriptors, keypoints_right,
    descriptors) ready for a stub SIFT.
    """
    n = true_count + len(outlier_dys)
    left_kps = [
        make_keypoint(20 + 12 * i, 20 + 6 * i) for i in range(n)
    ]
    right_kps: list[cv2.KeyPoint] = []
    for k, dy in enumerate(outlier_dys):
        x, y = left_kps[k].pt
        right_kps.append(make_keypoint(x - disparity_x, y + dy))
    for i in range(len(outlier_dys), n):
        x, y = left_kps[i].pt
        right_kps.append(make_keypoint(x - disparity_x, y))

    descriptors = one_hot_descriptors(n)
    return left_kps, descriptors, right_kps, descriptors.copy()


def run_evaluate_pair(
    left_kps, desc_l, right_kps, desc_r
) -> dict:
    row = vr.ManifestRow(
        group="g", stem="000", left_path=Path("l"), right_path=Path("r")
    )
    left_gray = np.zeros((240, 320), dtype=np.uint8)
    right_gray = np.zeros((240, 320), dtype=np.uint8)
    sift = OrderedStubSift(
        [(left_kps, desc_l), (right_kps, desc_r)]
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return vr.evaluate_pair(sift, matcher, row, left_gray, right_gray, 0.75)


def test_false_correspondences_do_not_inflate_p95():
    # 6 cross-polarization-style false matches at 150 px vertical offset
    # among 24 mutual matches: without filtering p95 would be ~150 px.
    left_kps, desc_l, right_kps, desc_r = synthetic_matches_with_outliers(
        true_count=18, outlier_dys=[150.0] * 6
    )

    pair = run_evaluate_pair(left_kps, desc_l, right_kps, desc_r)

    assert pair["status"] == "ok"
    assert pair["match_count"] == 24
    assert pair["inlier_count"] == 18
    assert pair["median_abs_y"] < 1.0
    assert pair["p95_abs_y"] < 3.0
    assert abs(pair["median_disparity_x"] - 24.0) < 1.0


def test_all_consistently_displaced_matches_fail_closed():
    # If every mutual match sits 150 px off vertically, the displacement is
    # fully consistent, so the MAD filter keeps it — indistinguishable from
    # a genuine coherent shift. The gate must then fail via the median check.
    left_kps, desc_l, right_kps, desc_r = synthetic_matches_with_outliers(
        true_count=0, outlier_dys=[150.0] * 8
    )

    pair = run_evaluate_pair(left_kps, desc_l, right_kps, desc_r)

    assert pair["match_count"] == 8
    assert pair["inlier_count"] == 8
    assert pair["status"] == "ok"
    assert pair["median_abs_y"] > 149.0
    thresholds = vr.Thresholds(min_matches=8, max_median_y=1.0, max_p95_y=3.0)
    report = vr.build_report([pair], thresholds)
    assert report["decision"]["passed"] is False
    assert any("median |dy|" in r for r in report["decision"]["reasons"])


def test_coherent_vertical_shift_survives_filter_and_fails():
    # A genuine coherent 4 px vertical shift must survive the outlier filter
    # untouched (no offset subtraction) and still report ~4 px.
    left_kps, desc_l, right_kps, desc_r = synthetic_matches_with_outliers(
        true_count=20, outlier_dys=[]
    )
    for kp_left, kp_right in zip(left_kps, right_kps):
        kp_right.pt = (kp_right.pt[0], kp_right.pt[1] - 4.0)

    pair = run_evaluate_pair(left_kps, desc_l, right_kps, desc_r)

    assert pair["status"] == "ok"
    assert pair["match_count"] == pair["inlier_count"]
    assert pair["inlier_count"] == 20
    assert pair["median_abs_y"] > 3.5
    assert pair["p95_abs_y"] > 3.5
    thresholds = vr.Thresholds(min_matches=15, max_median_y=1.0, max_p95_y=3.0)
    report = vr.build_report([pair], thresholds)
    assert report["decision"]["passed"] is False
    assert any("median |dy|" in r for r in report["decision"]["reasons"])


def test_coherent_shift_with_outliers_still_fails():
    # Outliers must not be able to mask a genuine shift either: the retained
    # inliers are the coherent 4 px group, which fails the gate.
    left_kps, desc_l, right_kps, desc_r = synthetic_matches_with_outliers(
        true_count=15, outlier_dys=[150.0] * 5
    )
    for kp_left, kp_right in zip(left_kps[len(left_kps) - 15 :], right_kps[-15:]):
        kp_right.pt = (kp_right.pt[0], kp_right.pt[1] - 4.0)

    pair = run_evaluate_pair(left_kps, desc_l, right_kps, desc_r)

    assert pair["match_count"] == 20
    assert pair["inlier_count"] == 15
    assert pair["median_abs_y"] > 3.5
    thresholds = vr.Thresholds(min_matches=15, max_median_y=1.0, max_p95_y=3.0)
    report = vr.build_report([pair], thresholds)
    assert report["decision"]["passed"] is False


def test_vertical_inlier_filter_on_evidence_like_distribution():
    rng = np.random.default_rng(7)
    inliers = 0.4 + rng.normal(0.0, 0.2, size=30)
    outliers = np.array([100.0, 150.0, 160.0, -120.0, 90.0, 140.0])
    dy = np.concatenate([inliers, outliers])

    mask = vertical_inlier_mask(dy)

    assert mask[:30].all()
    assert not mask[30:].any()
    retained = np.abs(dy[mask])
    assert float(np.percentile(retained, 95)) < 3.0


def test_vertical_inlier_filter_keeps_coherent_offset():
    dy = np.full(20, 4.0)

    assert vertical_inlier_mask(dy).all()


def test_min_matches_applies_to_retained_inliers():
    thresholds = vr.Thresholds(min_matches=25, max_median_y=1.0, max_p95_y=3.0)
    pair = {
        "group": "g",
        "stem": "000",
        "left": "l",
        "right": "r",
        "status": "ok",
        "match_count": 40,
        "inlier_count": 10,
        "median_abs_y": 0.3,
        "p95_abs_y": 0.8,
        "median_disparity_x": 24.0,
    }

    report = vr.build_report([pair], thresholds)

    assert report["decision"]["passed"] is False
    assert any("insufficient inliers" in r for r in report["decision"]["reasons"])
    assert report["overall"]["ok_count"] == 1
