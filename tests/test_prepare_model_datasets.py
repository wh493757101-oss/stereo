"""Tests for the Model-A binary and Model-B paired classification builders."""

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.prepare_binary_seg_dataset as pbsd
import scripts.prepare_cls_paired_datasets as pcls
from core.polar_compute import compute_polar_feature
from core.stereo_matching import (
    StereoMatcher,
    StereoMatcherConfig,
    StereoMatchResult,
)
from scripts.make_polar_dataset import make_crop_window

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish", "real_fish")

POLY_A = ((0.15, 0.15), (0.45, 0.15), (0.45, 0.55), (0.15, 0.55))
POLY_B = ((0.55, 0.20), (0.85, 0.20), (0.85, 0.60), (0.55, 0.60))
POLY_C = ((0.30, 0.60), (0.60, 0.60), (0.60, 0.90), (0.30, 0.90))

FRAME_SPECS = (
    ("g0_a", "train", "GroupA/0 NTU", ((0, POLY_A), (2, POLY_B))),
    ("g0_b", "train", "GroupA/0 NTU", ((3, POLY_C),)),
    ("g1_a", "val", "GroupB/10 NTU", ((1, POLY_A),)),
    ("g2_a", "test", "GroupC/20 NTU", ((0, POLY_B),)),
    ("g2_neg", "test", "GroupC/20 NTU", ()),
)
IMAGE_W, IMAGE_H = 64, 48


@pytest.fixture
def tmp_path() -> Path:
    """Isolated work directory under the system temp (the configured pytest
    basetemp under tests/ can be locked by another process on Windows)."""
    workdir = Path(tempfile.mkdtemp(prefix="prepare_model_datasets_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class RecordingMatcher(StereoMatcher):
    """Real instance_stats/object_disparity_map logic over a stubbed compute."""

    def __init__(self, disparity: float = 8.0, stereo_valid: bool = True) -> None:
        super().__init__(StereoMatcherConfig(max_disparity=32, block_size=3))
        self.disparity = disparity
        self.stereo_valid = stereo_valid
        self.compute_calls = 0
        self.compute_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def compute(self, left: np.ndarray, right: np.ndarray) -> StereoMatchResult:
        self.compute_calls += 1
        self.compute_shapes.append((left.shape, right.shape))
        h, w = left.shape[:2]
        if self.stereo_valid:
            disp = np.full((h, w), self.disparity, dtype=np.float32)
            valid = np.ones((h, w), dtype=np.uint8)
            ratio = 1.0
        else:
            disp = np.zeros((h, w), dtype=np.float32)
            valid = np.zeros((h, w), dtype=np.uint8)
            ratio = 0.0
        return StereoMatchResult(
            disparity=disp, valid=valid, elapsed_s=0.0, valid_ratio=ratio
        )


def make_source_dataset(root: Path, seed: int = 7) -> dict[str, np.ndarray]:
    """Create a tiny four-class YOLO-seg dataset; returns left images by stem."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    lefts: dict[str, np.ndarray] = {}
    for stem, split, group, objects in FRAME_SPECS:
        left = rng.integers(0, 256, size=(IMAGE_H, IMAGE_W), dtype=np.uint8)
        right = np.roll(left, 8, axis=1)
        lefts[stem] = left

        seg_dir = root / "images" / split
        seg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(seg_dir / f"{stem}.png"), left)

        stereo_dir = root / "stereo"
        stereo_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(stereo_dir / f"{stem}_left.png"), left)
        cv2.imwrite(str(stereo_dir / f"{stem}_right.png"), right)

        label_dir = root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for class_id, polygon in objects:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            lines.append(f"{class_id} {coords}")
        (label_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        rows.append(
            {
                "output_stem": stem,
                "split": split,
                "group_name": group,
                "left_path": (stereo_dir / f"{stem}_left.png").resolve().as_posix(),
                "right_path": (stereo_dir / f"{stem}_right.png").resolve().as_posix(),
            }
        )

    with (root / "pair_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["output_stem", "split", "group_name", "left_path", "right_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with (root / "split_assignment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["output_stem", "split", "group_name"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "output_stem": row["output_stem"],
                    "split": row["split"],
                    "group_name": row["group_name"],
                }
            )

    names_block = "\n".join(
        f"  {class_id}: {name}" for class_id, name in enumerate(CLASS_NAMES)
    )
    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    return lefts


def build_cls(tmp_path: Path, matcher, crop_pad: int = 4):
    source = tmp_path / "source"
    return pcls.build_cls_datasets(
        source_root=source,
        gray_root=tmp_path / "cls_gray",
        polar_root=tmp_path / "cls_polar",
        matcher=matcher,
        crop_pad=crop_pad,
    ), source, tmp_path / "cls_gray", tmp_path / "cls_polar"


# ---------------------------------------------------------------------------
# Model A: binary segmentation builder


def test_binary_builder_collapses_class_ids_and_preserves_coordinates(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(source)
    pbsd.build_binary_dataset(
        source_root=source,
        output_root=tmp_path / "binary",
        link_mode="copy",
        clean=False,
    )

    for stem, split, _, objects in FRAME_SPECS:
        source_text = (source / "labels" / split / f"{stem}.txt").read_text("utf-8")
        target_text = (
            tmp_path / "binary" / "labels" / split / f"{stem}.txt"
        ).read_text("utf-8")
        source_lines = [line for line in source_text.splitlines() if line]
        target_lines = [line for line in target_text.splitlines() if line]
        assert len(source_lines) == len(target_lines) == len(objects)
        for source_line, target_line in zip(source_lines, target_lines):
            expected = " ".join(["0", *source_line.split()[1:]])
            assert target_line == expected
            # no source class id other than the collapsed 0 survives
            assert int(target_line.split()[0]) == 0


def test_binary_builder_keeps_empty_negative_labels(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(source)
    summary = pbsd.build_binary_dataset(
        source_root=source,
        output_root=tmp_path / "binary",
        link_mode="copy",
        clean=False,
    )

    negative = tmp_path / "binary" / "labels" / "test" / "g2_neg.txt"
    assert negative.is_file()
    assert negative.read_text("utf-8") == ""
    assert summary["splits"]["test"]["negatives"] == 1
    assert summary["splits"]["test"]["images"] == 2


def test_binary_builder_hardlink_fallback_to_copy(tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source_dataset(source)

    def refuse_link(*_args, **_kwargs):
        raise OSError("hard links not supported")

    monkeypatch.setattr(os, "link", refuse_link)
    summary = pbsd.build_binary_dataset(
        source_root=source,
        output_root=tmp_path / "binary",
        link_mode="hardlink",
        clean=False,
    )

    total_images = len(FRAME_SPECS)
    assert summary["images_reused"]["hardlink"] == 0
    assert summary["images_reused"]["copy"] == total_images
    for stem, split, _, _ in FRAME_SPECS:
        target = tmp_path / "binary" / "images" / split / f"{stem}.png"
        assert target.is_file()
        np.testing.assert_array_equal(
            cv2.imread(str(target), cv2.IMREAD_UNCHANGED),
            cv2.imread(str(source / "images" / split / f"{stem}.png"),
                       cv2.IMREAD_UNCHANGED),
        )


def test_binary_builder_hardlink_mode_records_reuse(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(source)
    summary = pbsd.build_binary_dataset(
        source_root=source,
        output_root=tmp_path / "binary",
        link_mode="hardlink",
        clean=False,
    )

    total_images = len(FRAME_SPECS)
    recorded = summary["images_reused"]["hardlink"] + summary["images_reused"]["copy"]
    assert recorded == total_images
    probe_target = tmp_path / "hardlink_probe"
    try:
        os.link(source / "pair_manifest.csv", probe_target)
    except OSError:
        # Filesystem without hard-link support must fall back to copy.
        assert summary["images_reused"]["copy"] == total_images
    else:
        assert summary["images_reused"]["hardlink"] == total_images


def test_binary_builder_refuses_overlapping_roots(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(source)

    with pytest.raises(ValueError):
        pbsd.build_binary_dataset(
            source_root=source,
            output_root=source / "binary",
            link_mode="copy",
            clean=False,
        )
    with pytest.raises(ValueError):
        pbsd.build_binary_dataset(
            source_root=source,
            output_root=tmp_path,
            link_mode="copy",
            clean=False,
        )
    # nothing was written inside the source tree
    assert not (source / "binary").exists()


def test_binary_builder_preserves_manifests_and_writes_target_yaml(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(source)
    pbsd.build_binary_dataset(
        source_root=source,
        output_root=tmp_path / "binary",
        link_mode="copy",
        clean=False,
    )

    assert (tmp_path / "binary" / "pair_manifest.csv").read_bytes() == (
        source / "pair_manifest.csv"
    ).read_bytes()
    yaml_text = (tmp_path / "binary" / "data.yaml").read_text("utf-8")
    assert "names:" in yaml_text and "0: target" in yaml_text
    assert "images/train" in yaml_text and "images/test" in yaml_text
    for split in ("train", "val", "test"):
        assert (tmp_path / "binary" / "images" / split).is_dir()
        assert (tmp_path / "binary" / "labels" / split).is_dir()


# ---------------------------------------------------------------------------
# Model B: paired gray/polar classification builders


def test_cls_builder_runs_one_dense_compute_per_pair(tmp_path):
    make_source_dataset(tmp_path / "source")
    matcher = RecordingMatcher()
    records, *_ = build_cls(tmp_path, matcher)

    labeled_frames = sum(1 for spec in FRAME_SPECS if spec[3])
    assert matcher.compute_calls == labeled_frames
    assert all(left == right for left, right in matcher.compute_shapes)
    expected_samples = sum(len(spec[3]) for spec in FRAME_SPECS)
    assert len(records) == expected_samples


def test_cls_builder_supports_multiple_objects_per_frame(tmp_path):
    make_source_dataset(tmp_path / "source")
    matcher = RecordingMatcher()
    records, _, gray_root, polar_root = build_cls(tmp_path, matcher)

    frame_records = [r for r in records if r.source_frame == "g0_a"]
    assert len(frame_records) == 2
    assert frame_records[0].class_name == "metal_submarine"
    assert frame_records[1].class_name == "plastic_fish"
    for record in frame_records:
        assert (gray_root / record.gray_path).is_file()
        assert (polar_root / record.polar_path).is_file()


def test_cls_builder_paired_crops_identical_paths_and_content(tmp_path):
    lefts = make_source_dataset(tmp_path / "source")
    matcher = RecordingMatcher(disparity=8.0)
    records, source, gray_root, polar_root = build_cls(tmp_path, matcher)
    crop_pad = 4

    gray_manifest = (gray_root / "dataset_manifest.csv").read_text("utf-8")
    polar_manifest = (polar_root / "dataset_manifest.csv").read_text("utf-8")
    assert gray_manifest == polar_manifest

    for record in records:
        gray_img = cv2.imread(str(gray_root / record.gray_path), cv2.IMREAD_UNCHANGED)
        polar_img = cv2.imread(str(polar_root / record.polar_path), cv2.IMREAD_UNCHANGED)
        assert gray_img is not None and polar_img is not None
        assert gray_img.shape == polar_img.shape and gray_img.ndim == 3

        stem = record.source_frame
        split = record.split
        left = lefts[stem]
        polygons = pcls.read_yolo_polygons(
            source / "labels" / split / f"{stem}.txt", IMAGE_W, IMAGE_H
        )
        annotation = polygons[record.object_index]
        assert annotation.class_id == record.class_id

        window = make_crop_window(annotation.polygon, IMAGE_W, IMAGE_H, crop_pad)
        crop = left[window.y1 : window.y2, window.x1 : window.x2]

        # gray variant is [gray, gray, gray]
        np.testing.assert_array_equal(gray_img[..., 0], crop)
        np.testing.assert_array_equal(gray_img[..., 1], crop)
        np.testing.assert_array_equal(gray_img[..., 2], crop)
        # polar variant shares gray channels with the gray variant
        np.testing.assert_array_equal(polar_img[..., 0], crop)
        np.testing.assert_array_equal(polar_img[..., 2], crop)

        # polar channel equals the constant-disparity polar inside the mask,
        # and is exactly zero outside it
        mask = pcls.build_object_mask(annotation.polygon, IMAGE_W, IMAGE_H)
        obj_map = np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32)
        obj_map[mask > 0] = matcher.disparity
        expected = compute_polar_feature(left, np.roll(left, 8, axis=1), obj_map, mask=mask)
        expected_crop = expected[window.y1 : window.y2, window.x1 : window.x2]
        mask_crop = mask[window.y1 : window.y2, window.x1 : window.x2]
        np.testing.assert_allclose(
            polar_img[..., 1].astype(np.float32) / 255.0,
            expected_crop,
            atol=1.0 / 255.0,
        )
        assert np.all(polar_img[..., 1][mask_crop == 0] == 0)
        assert record.gray_path == record.polar_path


def test_cls_builder_uses_constant_object_disparity_path(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(tmp_path / "source")
    left = cv2.imread(
        str(source / "stereo" / "g0_a_left.png"), cv2.IMREAD_GRAYSCALE
    )
    right = cv2.imread(
        str(source / "stereo" / "g0_a_right.png"), cv2.IMREAD_GRAYSCALE
    )

    matcher = StereoMatcher(StereoMatcherConfig(max_disparity=32, block_size=3))
    annotation = pcls.read_yolo_polygons(
        source / "labels" / "train" / "g0_a.txt", IMAGE_W, IMAGE_H
    )[0]
    mask = pcls.build_object_mask(annotation.polygon, IMAGE_W, IMAGE_H)

    # Varying per-pixel disparity inside the mask; the object path must
    # collapse it to one robust constant before warping.
    raw = np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32)
    raw[mask > 0] = 10.0
    ys, xs = np.where(mask > 0)
    raw[ys[::3], xs[::3]] = 12.0
    raw[ys[1::3], xs[1::3]] = 14.0
    valid = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)
    valid[mask > 0] = 1
    calls = []

    def fake_compute(l: np.ndarray, r: np.ndarray) -> StereoMatchResult:
        calls.append(1)
        return StereoMatchResult(disparity=raw, valid=valid, elapsed_s=0.0, valid_ratio=0.5)

    matcher.compute = fake_compute

    stats = matcher.instance_stats(matcher.compute(left, right), mask, 0)
    object_map = matcher.object_disparity_map(matcher.compute(left, right), mask)

    assert len(calls) == 2  # standalone calls in this test only
    assert stats.valid and stats.disparity > 0
    inside = object_map[mask > 0]
    assert inside.size > 0 and np.all(inside == inside.flat[0])
    assert float(inside.flat[0]) == pytest.approx(stats.disparity)

    # The constant map differs from the raw per-pixel map, so a matching
    # polar field proves the constant fill was used for warping.
    assert not np.allclose(object_map[mask > 0], raw[mask > 0])
    constant_polar = compute_polar_feature(left, right, object_map, mask=mask)
    naive_polar = compute_polar_feature(left, right, raw, mask=mask)
    assert not np.allclose(constant_polar, naive_polar)


def test_cls_builder_retains_invalid_stereo_samples(tmp_path):
    make_source_dataset(tmp_path / "source")
    matcher = RecordingMatcher(stereo_valid=False)
    records, _, gray_root, polar_root = build_cls(tmp_path, matcher)

    expected_samples = sum(len(spec[3]) for spec in FRAME_SPECS)
    assert len(records) == expected_samples
    for record in records:
        assert not record.stereo_valid
        assert record.stereo_reason == "no_valid_disparity"
        assert (gray_root / record.gray_path).is_file()
        assert (polar_root / record.polar_path).is_file()
        polar_image = cv2.imread(
            str(polar_root / record.polar_path), cv2.IMREAD_UNCHANGED
        )
        assert polar_image is not None
        assert np.all(polar_image[..., 1] == 0)

    with (gray_root / "dataset_manifest.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["stereo_valid"] == "false" for row in rows)
    assert all(row["stereo_reason"] == "no_valid_disparity" for row in rows)
    assert all(float(row["valid_ratio"]) == 0.0 for row in rows)


def test_cls_builder_rejects_unknown_class_id(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(source)
    label = source / "labels" / "train" / "g0_b.txt"
    label.write_text("9 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown class id 9"):
        pcls.build_cls_datasets(
            source_root=source,
            gray_root=tmp_path / "cls_gray",
            polar_root=tmp_path / "cls_polar",
            matcher=RecordingMatcher(),
            crop_pad=4,
        )


def test_cls_builder_emits_three_splits_four_classes_and_group_isolation(tmp_path):
    make_source_dataset(tmp_path / "source")
    matcher = RecordingMatcher()
    records, _, gray_root, polar_root = build_cls(tmp_path, matcher)

    for root in (gray_root, polar_root):
        for split in ("train", "val", "test"):
            split_dir = root / split
            assert split_dir.is_dir()
            for class_name in CLASS_NAMES:
                assert (split_dir / class_name).is_dir()

    assert (gray_root / "train" / "metal_submarine" / "g0_a_obj000.png").is_file()
    assert (gray_root / "train" / "plastic_fish" / "g0_a_obj001.png").is_file()
    assert (gray_root / "val" / "plastic_submarine" / "g1_a_obj000.png").is_file()
    assert (gray_root / "test" / "metal_submarine" / "g2_a_obj000.png").is_file()
    assert (polar_root / "test" / "metal_submarine" / "g2_a_obj000.png").is_file()

    group_to_split: dict[str, str] = {}
    for record in records:
        previous = group_to_split.setdefault(record.group_name, record.split)
        assert previous == record.split
    summary = json.loads((gray_root / "dataset_summary.json").read_text("utf-8"))
    assert sorted(summary["splits"]) == ["test", "train", "val"]
    assert summary["splits"]["train"]["samples"] == 3


def test_cls_builder_manifest_is_deterministic(tmp_path):
    make_source_dataset(tmp_path / "source")
    matcher = RecordingMatcher()
    pcls.build_cls_datasets(
        source_root=tmp_path / "source",
        gray_root=tmp_path / "cls_gray",
        polar_root=tmp_path / "cls_polar",
        matcher=matcher,
        crop_pad=4,
    )
    first = {
        name: (tmp_path / "cls_gray" / name).read_bytes()
        for name in ("dataset_manifest.csv", "dataset_summary.json")
    }

    pcls.build_cls_datasets(
        source_root=tmp_path / "source",
        gray_root=tmp_path / "cls_gray",
        polar_root=tmp_path / "cls_polar",
        matcher=RecordingMatcher(),
        crop_pad=4,
        clean=True,
    )
    for name, payload in first.items():
        assert (tmp_path / "cls_gray" / name).read_bytes() == payload


def test_cls_builder_refuses_overlapping_roots(tmp_path):
    source = tmp_path / "source"
    make_source_dataset(source)

    with pytest.raises(ValueError):
        pcls.build_cls_datasets(
            source_root=source,
            gray_root=source / "cls",
            polar_root=tmp_path / "cls_polar",
            matcher=RecordingMatcher(),
        )
    with pytest.raises(ValueError):
        pcls.build_cls_datasets(
            source_root=source,
            gray_root=tmp_path / "cls_gray",
            polar_root=tmp_path / "cls_gray",
            matcher=RecordingMatcher(),
        )
    with pytest.raises(ValueError):
        pcls.build_cls_datasets(
            source_root=source,
            gray_root=tmp_path / "cls_gray",
            polar_root=tmp_path / "cls_gray" / "nested",
            matcher=RecordingMatcher(),
        )


def test_cls_load_gray_normalizes_patched_one_channel_reads(tmp_path, monkeypatch):
    """Ultralytics patches cv2.imread so grayscale reads return (H, W, 1);
    load_gray must still hand back contiguous uint8 (H, W) before any
    crop/mask/channel construction."""
    raw = np.arange(48, dtype=np.uint8).reshape(6, 8, 1)
    monkeypatch.setattr(pcls.cv2, "imread", lambda path, flag: raw)

    img = pcls.load_gray(tmp_path / "frame.png")

    assert img.shape == (6, 8)
    assert img.dtype == np.uint8
    assert img.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(img[:, 0], raw[:, 0, 0])


def test_cls_load_gray_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(pcls.cv2, "imread", lambda path, flag: None)
    with pytest.raises(FileNotFoundError):
        pcls.load_gray(tmp_path / "missing.png")
