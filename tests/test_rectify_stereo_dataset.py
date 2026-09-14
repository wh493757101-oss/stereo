from pathlib import Path
import shutil

import pytest

import json

import cv2
import numpy as np
import pytest

from scripts.rectify_stereo_dataset import (
    build_rectify_maps,
    collect_rectify_groups,
    load_calibration_npz,
    output_group_root,
    parse_args,
    save_rectification_outputs,
)


def test_default_output_root_targets_rectified_v2():
    args = parse_args(["--calib", "stereo_calib.npz"])
    assert args.output == Path("datasets/Rectified_v2")


@pytest.fixture
def workspace_tmp() -> Path:
    root = Path(__file__).resolve().parent / "_tmp_rectify"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_collect_rectify_groups_finds_nested_scene_turbidity_dirs(workspace_tmp: Path):
    sample_dir = workspace_tmp / "Single" / "Metal submarine" / "20 NTU"
    left_dir = sample_dir / "left"
    right_dir = sample_dir / "right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir()
    (left_dir / "000.png").write_bytes(b"left")
    (right_dir / "000.png").write_bytes(b"right")

    groups = collect_rectify_groups([workspace_tmp / "Single"])

    assert len(groups) == 1
    assert groups[0].relative_group == Path("Single") / "Metal submarine" / "20 NTU"
    assert groups[0].pairs[0].stem == "000"


def test_output_group_root_preserves_input_root_name_and_relative_structure(workspace_tmp: Path):
    out = output_group_root(
        output_root=workspace_tmp / "Rectified",
        input_root=workspace_tmp / "Mixed",
        group_dir=workspace_tmp / "Mixed" / "All mixed" / "10 NTU",
    )

    assert out == workspace_tmp / "Rectified" / "Mixed" / "All mixed" / "10 NTU"


@pytest.fixture
def calibration_path(tmp_path: Path) -> Path:
    rotation, _ = cv2.Rodrigues(np.array([0.02, -0.03, 0.01]))
    path = tmp_path / "stereo_calib.npz"
    np.savez(
        path,
        K1=np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        D1=np.array([[-0.1, 0.01, 0.0, 0.0, 0.0]]),
        K2=np.array([[505.0, 0.0, 315.0], [0.0, 505.0, 242.0], [0.0, 0.0, 1.0]]),
        D2=np.array([[-0.12, 0.02, 0.0, 0.0, 0.0]]),
        R=rotation,
        T=np.array([[-0.0989], [0.0001], [0.0002]]),
        image_size=np.array([640, 480]),
    )
    return path


def test_matlab_convention_transposes_rotation(calibration_path: Path):
    calib = load_calibration_npz(calibration_path, r_convention="matlab")
    r_raw = np.load(str(calibration_path))["R"]

    assert calib.r_convention == "matlab"
    assert np.allclose(calib.r, r_raw.T)
    assert np.allclose(calib.r @ r_raw, np.eye(3), atol=1e-12)


def test_opencv_convention_uses_rotation_as_is(calibration_path: Path):
    calib = load_calibration_npz(calibration_path, r_convention="opencv")
    r_raw = np.load(str(calibration_path))["R"]

    assert calib.r_convention == "opencv"
    assert np.allclose(calib.r, r_raw)
    assert not np.allclose(calib.r, r_raw.T)


def test_unknown_convention_is_rejected(calibration_path: Path):
    with pytest.raises(ValueError, match="convention"):
        load_calibration_npz(calibration_path, r_convention="bogus")


@pytest.mark.parametrize(
    "overrides",
    [
        {"R": np.eye(4)},
        {"K1": np.eye(2)},
        {"image_size": np.array([640])},
        {"image_size": np.array([0, 480])},
        {"T": np.zeros((3, 1))},
        {"D1": np.array([[-0.1, 0.01]])},
    ],
)
def test_malformed_calibration_shapes_fail(calibration_path: Path, overrides: dict):
    data = dict(np.load(str(calibration_path)))
    data.update(overrides)
    broken_path = calibration_path.parent / "broken.npz"
    np.savez(broken_path, **data)

    with pytest.raises(ValueError):
        load_calibration_npz(broken_path)


def test_missing_calibration_keys_fail(calibration_path: Path):
    data = dict(np.load(str(calibration_path)))
    data.pop("R")
    broken_path = calibration_path.parent / "missing.npz"
    np.savez(broken_path, **data)

    with pytest.raises(ValueError, match="missing keys"):
        load_calibration_npz(broken_path)


def test_saved_rectification_outputs_include_matrices_and_provenance(
    calibration_path: Path, tmp_path: Path
):
    calib = load_calibration_npz(calibration_path, r_convention="matlab")
    maps = build_rectify_maps(calib, alpha=0.0)
    output_root = tmp_path / "rectified"

    params_path, metadata_path = save_rectification_outputs(
        output_root=output_root,
        maps=maps,
        calibration=calib,
        calib_path=calibration_path,
        alpha=0.0,
    )

    params = np.load(str(params_path))
    for key in ("R1", "R2", "P1", "P2", "Q"):
        assert key in params.files
    assert np.allclose(params["R1"], maps.r1)
    assert params["P1"].shape == (3, 4)
    assert params["Q"].shape == (4, 4)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["r_convention"] == "matlab"
    assert "R_cv = R.T" in metadata["r_cv_derivation"]
    assert metadata["image_size"] == [640, 480]
    assert metadata["calibration_path"] == str(calibration_path)


def test_saved_metadata_records_opencv_provenance(calibration_path: Path, tmp_path: Path):
    calib = load_calibration_npz(calibration_path, r_convention="opencv")
    maps = build_rectify_maps(calib, alpha=0.0)

    _, metadata_path = save_rectification_outputs(
        output_root=tmp_path / "rectified",
        maps=maps,
        calibration=calib,
        calib_path=calibration_path,
        alpha=0.0,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["r_convention"] == "opencv"
    assert "as-is" in metadata["r_cv_derivation"]
