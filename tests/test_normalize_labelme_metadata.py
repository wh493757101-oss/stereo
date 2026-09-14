import json
from pathlib import Path

from scripts.normalize_labelme_metadata import normalize_annotation_tree


def test_normalize_tree_preserves_shapes_and_repairs_metadata(tmp_path: Path):
    annotation_root = tmp_path / "bz_JSON_v2"
    image_root = tmp_path / "Rectified_v2"
    group = Path("Mixed/All mixed/0 NTU")
    json_path = annotation_root / group / "000.json"
    image_path = image_root / group / "left" / "000.png"
    json_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    shapes = [
        {
            "label": "real_fish",
            "points": [[1.25, 2.5], [8.0, 2.0], [4.0, 9.0]],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        }
    ]
    json_path.write_text(
        json.dumps(
            {
                "version": "5.2.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": "..\\wrong\\000.png",
                "imageData": "embedded-base64",
                "imageHeight": 10,
                "imageWidth": 10,
            }
        ),
        encoding="utf-8",
    )

    summary = normalize_annotation_tree(annotation_root, image_root)

    normalized = json.loads(json_path.read_text(encoding="utf-8"))
    assert normalized["shapes"] == shapes
    assert normalized["imageData"] is None
    assert "\\" not in normalized["imagePath"]
    assert not Path(normalized["imagePath"]).is_absolute()
    assert (json_path.parent / normalized["imagePath"]).resolve() == image_path.resolve()
    assert summary == {
        "json_files": 1,
        "changed_files": 1,
        "embedded_image_data_removed": 1,
        "missing_images": 0,
    }


def test_normalize_tree_dry_run_does_not_write(tmp_path: Path):
    annotation_root = tmp_path / "annotations"
    image_root = tmp_path / "images"
    group = Path("Single/Real fish/0 NTU")
    json_path = annotation_root / group / "001.json"
    image_path = image_root / group / "left" / "001.png"
    json_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    original = json.dumps(
        {
            "shapes": [],
            "imagePath": "left/001.png",
            "imageData": "bytes",
            "imageHeight": 10,
            "imageWidth": 10,
        }
    )
    json_path.write_text(original, encoding="utf-8")

    summary = normalize_annotation_tree(annotation_root, image_root, dry_run=True)

    assert json_path.read_text(encoding="utf-8") == original
    assert summary["changed_files"] == 1
    assert summary["embedded_image_data_removed"] == 1


def test_normalize_tree_reports_missing_corresponding_image(tmp_path: Path):
    annotation_root = tmp_path / "annotations"
    image_root = tmp_path / "images"
    image_root.mkdir()
    json_path = annotation_root / "Mixed/group/0 NTU/002.json"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(
        json.dumps({"shapes": [], "imagePath": "missing.png", "imageData": None}),
        encoding="utf-8",
    )

    summary = normalize_annotation_tree(annotation_root, image_root)

    assert summary["missing_images"] == 1
    assert summary["changed_files"] == 0
