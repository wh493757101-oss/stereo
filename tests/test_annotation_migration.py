import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.migrate_labelme_annotations import (
    empty_labelme_json,
    left_rectified_homography,
    migrate_annotation,
    migrate_annotations,
    transform_points,
    validate_output_root,
)
from scripts.rectify_stereo_dataset import (
    build_rectify_maps,
    load_calibration_npz,
)


@pytest.fixture
def calibration_path(tmp_path: Path) -> Path:
    rotation, _ = cv2.Rodrigues(np.array([0.03, -0.02, 0.01]))
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


def test_same_convention_yields_identity_homography(calibration_path: Path):
    maps = build_rectify_maps(load_calibration_npz(calibration_path, r_convention="opencv"))

    homography = left_rectified_homography(maps, maps)

    assert np.allclose(homography, np.eye(3), atol=1e-9)


def test_matlab_vs_opencv_convention_yields_non_identity_homography(calibration_path: Path):
    old_maps = build_rectify_maps(load_calibration_npz(calibration_path, r_convention="opencv"))
    new_maps = build_rectify_maps(load_calibration_npz(calibration_path, r_convention="matlab"))

    homography = left_rectified_homography(old_maps, new_maps)

    assert np.isfinite(homography).all()
    assert homography[2, 2] == pytest.approx(1.0)
    assert not np.allclose(homography, np.eye(3), atol=1e-6)


def test_homography_matches_analytic_composition(calibration_path: Path):
    old_maps = build_rectify_maps(load_calibration_npz(calibration_path, r_convention="opencv"))
    new_maps = build_rectify_maps(load_calibration_npz(calibration_path, r_convention="matlab"))

    homography = left_rectified_homography(old_maps, new_maps)

    expected = (
        new_maps.p1[:, :3]
        @ new_maps.r1
        @ np.linalg.inv(old_maps.r1)
        @ np.linalg.inv(old_maps.p1[:, :3])
    )
    expected = expected / expected[2, 2]
    assert np.allclose(homography, expected)


def test_transform_points_clips_to_image_bounds():
    translation = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    points = transform_points([[10.0, 5.0]], translation, width=100, height=50)

    assert points == [[100.0, 5.0]]

    points = transform_points([[50.0, 60.0]], translation, width=100, height=50)

    assert points == [[100.0, 50.0]]


def test_migrate_annotation_nulls_image_data_and_updates_metadata():
    data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [
            {
                "label": "metal_submarine",
                "shape_type": "polygon",
                "points": [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]],
            }
        ],
        "imagePath": "../old/000.png",
        "imageData": "aGVsbG8=",
        "imageWidth": 100,
        "imageHeight": 50,
    }
    identity = np.eye(3)

    migrated = migrate_annotation(
        data, identity, width=640, height=480, image_path="left/000.png"
    )

    assert migrated["imageData"] is None
    assert migrated["imagePath"] == "left/000.png"
    assert migrated["imageWidth"] == 640
    assert migrated["imageHeight"] == 480
    assert migrated["shapes"][0]["label"] == "metal_submarine"
    assert migrated["shapes"][0]["points"] == [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]]

    kept = migrate_annotation(
        data, identity, width=640, height=480, image_path="left/000.png",
        keep_image_data=True,
    )
    assert kept["imageData"] == "aGVsbG8="


def test_migrate_annotation_transforms_points_by_homography():
    translation = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    data = {
        "shapes": [
            {
                "label": "plastic_fish",
                "shape_type": "polygon",
                "points": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
            }
        ],
        "imageData": None,
    }

    migrated = migrate_annotation(
        data, translation, width=640, height=480, image_path="left/001.png"
    )

    assert migrated["shapes"][0]["points"] == [[10.0, 20.0], [15.0, 20.0], [15.0, 25.0]]


def test_empty_labelme_json_retains_negative_frame():
    payload = empty_labelme_json("left/002.png", width=640, height=480)

    assert payload["shapes"] == []
    assert payload["imagePath"] == "left/002.png"
    assert payload["imageData"] is None
    assert payload["imageWidth"] == 640
    assert payload["imageHeight"] == 480
    assert payload["version"]


def test_migrate_annotations_writes_migrated_and_empty_frames(
    calibration_path: Path, tmp_path: Path
):
    maps = build_rectify_maps(load_calibration_npz(calibration_path, r_convention="matlab"))
    annotation_root = tmp_path / "annotations"
    image_root = tmp_path / "images"
    new_image_root = tmp_path / "rectified_v2"
    output_root = tmp_path / "annotations_v2"
    (annotation_root / "group_a").mkdir(parents=True)
    (image_root / "group_a" / "left").mkdir(parents=True)
    (new_image_root / "group_a" / "left").mkdir(parents=True)

    annotated = {
        "version": "5.2.1",
        "shapes": [
            {
                "label": "metal_submarine",
                "shape_type": "polygon",
                "points": [[320.0, 240.0], [340.0, 240.0], [330.0, 260.0]],
            }
        ],
        "imagePath": "../old/000.png",
        "imageData": "aGVsbG8=",
        "imageWidth": 640,
        "imageHeight": 480,
    }
    (annotation_root / "group_a" / "000.json").write_text(json.dumps(annotated), encoding="utf-8")
    (image_root / "group_a" / "left" / "000.png").write_bytes(b"fake")
    (new_image_root / "group_a" / "left" / "000.png").write_bytes(b"fake")
    (new_image_root / "group_a" / "left" / "001.png").write_bytes(b"fake")

    rows = migrate_annotations(
        annotation_root=annotation_root,
        new_image_root=new_image_root,
        output_root=output_root,
        homography=np.eye(3),
        width=640,
        height=480,
    )

    statuses = {row["stem"]: row["status"] for row in rows}
    assert statuses == {"000": "migrated", "001": "empty"}

    migrated = json.loads((output_root / "group_a" / "000.json").read_text(encoding="utf-8"))
    assert migrated["imagePath"] == "../../rectified_v2/group_a/left/000.png"
    assert migrated["imageWidth"] == 640
    assert migrated["imageHeight"] == 480
    assert migrated["imageData"] is None
    assert migrated["shapes"][0]["points"][0] == pytest.approx([320.0, 240.0], abs=1e-6)

    negative = json.loads((output_root / "group_a" / "001.json").read_text(encoding="utf-8"))
    assert negative["shapes"] == []
    assert negative["imagePath"] == "../../rectified_v2/group_a/left/001.png"
    assert negative["imageData"] is None


def test_migrate_annotations_image_path_resolves_across_roots(tmp_path: Path):
    annotation_root = tmp_path / "annotations"
    new_image_root = tmp_path / "Rectified_v2"
    output_root = tmp_path / "bz_JSON_v2"
    nested_group = Path("Mixed/All mixed/0 NTU")
    (annotation_root / nested_group).mkdir(parents=True)
    (new_image_root / nested_group / "left").mkdir(parents=True)
    (annotation_root / nested_group / "000.json").write_text(
        json.dumps({"version": "5.2.1", "shapes": []}), encoding="utf-8"
    )
    (new_image_root / nested_group / "left" / "000.png").write_bytes(b"fake")

    migrate_annotations(
        annotation_root=annotation_root,
        new_image_root=new_image_root,
        output_root=output_root,
        homography=np.eye(3),
        width=1280,
        height=1024,
    )

    payload = json.loads(
        (output_root / nested_group / "000.json").read_text(encoding="utf-8")
    )
    image_path = payload["imagePath"]

    assert payload["imageData"] is None
    assert "\\" not in image_path
    assert not Path(image_path).is_absolute()
    resolved = (output_root / nested_group / image_path).resolve()
    assert resolved == (new_image_root / nested_group / "left" / "000.png").resolve()
    assert resolved.is_file()


def test_validate_output_root_refuses_aliasing(tmp_path: Path):
    annotations = tmp_path / "bz_JSON"
    images = tmp_path / "Rectified"
    annotations.mkdir()
    images.mkdir()
    elsewhere = tmp_path / "output"

    with pytest.raises(ValueError):
        validate_output_root(annotations, [annotations, images])

    with pytest.raises(ValueError):
        validate_output_root(annotations / "nested", [annotations, images])

    with pytest.raises(ValueError):
        validate_output_root(tmp_path, [annotations, images])

    validate_output_root(elsewhere, [annotations, images])
