import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.make_polar_dataset as mpd
from core.stereo_matching import StereoMatcher, StereoMatcherConfig

WORKSPACE = Path(__file__).resolve().parent / "_tmp_make_polar"


@pytest.fixture
def tmp_path() -> Path:
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True)
    yield WORKSPACE
    shutil.rmtree(WORKSPACE, ignore_errors=True)


class StubMatcher:
    """Stands in for StereoMatcher and records compute() calls."""

    def __init__(self, disparity=32.0):
        self.disparity = disparity
        self.compute_calls = 0
        self.seen_shapes = []

    def compute(self, left, right):
        self.compute_calls += 1
        self.seen_shapes.append(left.shape)
        h, w = left.shape[:2]
        disparity = np.full((h, w), self.disparity, dtype=np.float32)
        from core.stereo_matching import StereoMatchResult

        return StereoMatchResult(
            disparity=disparity,
            valid=np.ones((h, w), dtype=np.uint8),
            elapsed_s=0.0,
            valid_ratio=1.0,
        )


@pytest.fixture
def pair_files(tmp_path: Path):
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 256, size=(64, 96), dtype=np.uint8)
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    cv2.imwrite(str(left), gray)
    cv2.imwrite(str(right), gray)
    return left, right, gray


def test_load_gray_normalizes_patched_one_channel_reads(tmp_path, monkeypatch):
    """Ultralytics patches cv2.imread so grayscale reads return (H, W, 1);
    load_gray must still hand back contiguous uint8 (H, W)."""
    raw = np.arange(64, dtype=np.uint8).reshape(8, 8, 1)
    monkeypatch.setattr(mpd.cv2, "imread", lambda path, flag: raw)

    img = mpd.load_gray(tmp_path / "left.png")

    assert img.shape == (8, 8)
    assert img.dtype == np.uint8
    assert img.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(img[:, 0], raw[:, 0, 0])


def test_load_gray_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mpd.cv2, "imread", lambda path, flag: None)
    with pytest.raises(FileNotFoundError):
        mpd.load_gray(tmp_path / "missing.png")


def test_process_pair_uses_given_matcher_once(pair_files, tmp_path):
    left, right, _ = pair_files
    matcher = StubMatcher(disparity=24.0)
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    outputs = mpd.process_pair(
        left_path=left,
        right_path=right,
        image_dir=image_dir,
        matcher=matcher,
        output_format="png3",
    )

    assert matcher.compute_calls == 1
    assert len(outputs) == 1 and outputs[0].exists()


def test_polar_channel_uses_original_intensities(pair_files, tmp_path):
    """The saved polar channel must be derived from the original fixed
    intensities, not from any CLAHE-normalized matching image."""
    left, right, gray = pair_files
    matcher = StubMatcher(disparity=16.0)
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    mpd.process_pair(
        left_path=left,
        right_path=right,
        image_dir=image_dir,
        matcher=matcher,
        output_format="png3",
    )

    saved = cv2.imread(str(image_dir / "left.png"), cv2.IMREAD_UNCHANGED)
    assert saved is not None and saved.shape == (64, 96, 3)
    # Gray channels must be byte-identical to the original intensities.
    np.testing.assert_array_equal(saved[..., 0], gray)
    np.testing.assert_array_equal(saved[..., 2], gray)
    # Polar channel must equal the ratio computed from the original grays.
    disp = np.full(gray.shape, 16.0, dtype=np.float32)
    expected = mpd.compute_polar_feature(gray, gray, disp)
    np.testing.assert_allclose(
        saved[..., 1].astype(np.float32) / 255.0, expected, atol=1 / 255.0
    )


def test_crop_mode_reuses_one_dense_compute_per_pair(pair_files, tmp_path):
    left, right, gray = pair_files
    label_path = tmp_path / "lbl.txt"
    label_path.write_text("0 0.2 0.2 0.6 0.2 0.4 0.6", encoding="utf-8")
    image_dir = tmp_path / "out" / "images"
    image_dir.mkdir(parents=True)
    matcher = StubMatcher(disparity=8.0)

    count = mpd.process_pair_crops(
        left_path=left,
        right_path=right,
        label_path=label_path,
        image_dir=image_dir,
        matcher=matcher,
        padding=4,
        output_stem="pair0",
    )

    assert count == 1
    assert matcher.compute_calls == 1
    assert (image_dir / "pair0_obj000.png").exists()


def test_cli_defaults_match_new_stereo_settings(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_polar_dataset.py", "--manifest", "m.csv", "--output", "out"],
    )
    args = mpd.parse_args()

    assert args.max_disp == 768
    assert args.block_size == 7
    assert args.scale == 0.25
    assert args.lr_check_threshold == 2.0
    assert args.min_valid_ratio == 0.05


def test_cli_window_flag_maps_to_block_size(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_polar_dataset.py",
            "--manifest",
            "m.csv",
            "--output",
            "out",
            "--window",
            "3",
        ],
    )
    args = mpd.parse_args()
    assert args.block_size == 3


def test_build_matcher_uses_cli_settings(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_polar_dataset.py",
            "--manifest",
            "m.csv",
            "--output",
            "out",
            "--max-disp",
            "512",
            "--scale",
            "0.5",
            "--block-size",
            "5",
        ],
    )
    args = mpd.parse_args()
    matcher = mpd.build_matcher(args)

    assert isinstance(matcher, StereoMatcher)
    assert matcher.config.max_disparity == 512
    assert matcher.config.scale == 0.5
    assert matcher.config.block_size == 5
    assert matcher.config.matcher == "sgbm"


def test_legacy_crop_test_inputs_still_flow_through(monkeypatch, tmp_path):
    """End-to-end guard mirroring the tiny uniform-image crop scenario: the
    matcher must degrade to an all-invalid map instead of raising."""
    image = np.full((16, 20), 128, dtype=np.uint8)
    left = tmp_path / "l.png"
    right = tmp_path / "r.png"
    cv2.imwrite(str(left), image)
    cv2.imwrite(str(right), image)
    label = tmp_path / "l.txt"
    label.write_text("0 0.1 0.1 0.5 0.1 0.3 0.5", encoding="utf-8")
    image_dir = tmp_path / "out" / "images"
    image_dir.mkdir(parents=True)

    matcher = StereoMatcher(StereoMatcherConfig(max_disparity=1, block_size=3))
    count = mpd.process_pair_crops(
        left_path=left,
        right_path=right,
        label_path=label,
        image_dir=image_dir,
        matcher=matcher,
        padding=2,
        output_stem="t",
    )
    assert count == 1
    assert cv2.imread(str(image_dir / "t_obj000.png")) is not None
