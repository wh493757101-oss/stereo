import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.make_polar_dataset import (
    crop_polygon,
    make_crop_window,
    read_yolo_polygons,
)
from scripts.prepare_yolo_dataset import (
    LabeledImage,
    PolygonAnnotation,
    make_output_stem,
    parse_args,
    polygon_for_shape,
    remove_duplicate_annotations,
    split_by_block,
    split_by_group,
    yolo_label_lines,
)


def test_parse_args_defaults_target_v2_paths():
    args = parse_args([])
    assert args.annotations == "datasets/bz_JSON_v2"
    assert args.image_root == "datasets/Rectified_v2"
    assert args.output == "datasets/underwater_seg_v2"


def test_polygon_for_shape_clips_and_normalizes_with_yolo_writer():
    shape = {
        "shape_type": "polygon",
        "points": [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]],
    }

    polygon = polygon_for_shape(shape, width=100, height=50, min_area=1.0)
    lines = yolo_label_lines(
        [PolygonAnnotation(class_id=2, polygon=polygon)], width=100, height=50
    )

    assert lines == ["2 0.100000 0.200000 0.300000 0.200000 0.200000 0.600000"]


def test_make_output_stem_avoids_nested_filename_collisions():
    stem = make_output_stem(Path("Single") / "Metal submarine" / "0 NTU" / "000.json")

    assert stem == "Single_Metal_submarine_0_NTU_000"


def test_remove_duplicate_annotations_drops_exact_duplicates_only():
    polygon = ((2.0, 2.0), (10.0, 2.0), (6.0, 8.0))
    other = ((1.0, 1.0), (5.0, 1.0), (3.0, 4.0))
    annotations = [
        PolygonAnnotation(class_id=0, polygon=polygon),
        PolygonAnnotation(class_id=1, polygon=polygon),
        PolygonAnnotation(class_id=0, polygon=polygon),
        PolygonAnnotation(class_id=0, polygon=other),
        PolygonAnnotation(class_id=1, polygon=polygon),
    ]

    kept, removed = remove_duplicate_annotations(annotations)

    assert kept == (
        PolygonAnnotation(class_id=0, polygon=polygon),
        PolygonAnnotation(class_id=1, polygon=polygon),
        PolygonAnnotation(class_id=0, polygon=other),
    )
    assert removed == 2


def test_remove_duplicate_annotations_keeps_nonduplicates_untouched():
    polygon = ((2.0, 2.0), (10.0, 2.0), (6.0, 8.0))
    other = ((1.0, 1.0), (5.0, 1.0), (3.0, 4.0))
    annotations = [
        PolygonAnnotation(class_id=0, polygon=polygon),
        PolygonAnnotation(class_id=1, polygon=polygon),
        PolygonAnnotation(class_id=0, polygon=other),
    ]

    kept, removed = remove_duplicate_annotations(annotations)

    assert kept == tuple(annotations)
    assert removed == 0


def test_split_by_block_isolates_blocks_and_covers_every_group():
    def item(group: str, stem: str) -> LabeledImage:
        return LabeledImage(
            output_stem=f"{group}_{stem}",
            relative_json=Path(group) / f"{stem}.json",
            group_name=group,
            json_path=Path(group) / f"{stem}.json",
            left_path=Path(f"{stem}.png"),
            right_path=Path(f"{stem}.png"),
            width=100,
            height=100,
            annotations=(),
        )

    items = [item("group_a", f"{index:03d}") for index in range(8)]
    items += [item("group_b", f"{index:03d}") for index in range(8)]

    assignments = split_by_block(
        items=items,
        val_ratio=0.2,
        block_size=8,
        seed=7,
    )

    grouped = {}
    for assignment in assignments:
        grouped.setdefault(assignment.item.group_name, []).append(assignment)

    assert len(grouped) == 2
    for group_assignments in grouped.values():
        splits_by_block = {}
        for assignment in group_assignments:
            splits_by_block.setdefault(assignment.block_id, set()).add(assignment.split)
        assert all(len(splits) == 1 for splits in splits_by_block.values())
        assert (
            sum(next(iter(splits)) == "val" for splits in splits_by_block.values()) == 1
        )


@pytest.fixture
def workspace_tmp() -> Path:
    root = Path(__file__).resolve().parent / "_tmp_prepare_yolo"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_cli_writes_yolo_segmentation_dataset(workspace_tmp: Path, monkeypatch) -> None:
    annotation_root = (
        workspace_tmp / "annotations" / "Single" / "Metal submarine" / "0 NTU"
    )
    image_root = workspace_tmp / "images" / "Single" / "Metal submarine" / "0 NTU"
    annotation_root.mkdir(parents=True)
    (image_root / "left").mkdir(parents=True)
    (image_root / "right").mkdir(parents=True)

    image = np.zeros((16, 20), dtype=np.uint8)
    for stem in ("000", "001", "002", "003"):
        cv2.imwrite(str(image_root / "left" / f"{stem}.png"), image)
        cv2.imwrite(str(image_root / "right" / f"{stem}.png"), image)
        data = {
            "version": "5.2.1",
            "imageWidth": 20,
            "imageHeight": 16,
            "shapes": [
                {
                    "label": "metal_submarine",
                    "shape_type": "polygon",
                    "points": [[2, 2], [10, 2], [6, 8]],
                }
            ],
        }
        (annotation_root / f"{stem}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    output_root = workspace_tmp / "yolo"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_yolo_dataset.py",
            "--annotations",
            str(annotation_root.parent.parent),
            "--image-root",
            str(image_root.parent.parent),
            "--output",
            str(output_root),
            "--classes",
            "metal_submarine",
            "--val-ratio",
            "0.5",
            "--split-by",
            "block",
            "--block-size",
            "2",
            "--clean",
        ],
    )

    from scripts.prepare_yolo_dataset import main

    main()

    yaml_text = (output_root / "data.yaml").read_text(encoding="utf-8")
    assert "0: metal_submarine" in yaml_text
    assert (output_root / "images" / "train").is_dir()
    assert (output_root / "images" / "val").is_dir()

    with (output_root / "split_assignment.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"train", "val"}

    for row in rows:
        label_path = output_root / "labels" / row["split"] / f"{row['output_stem']}.txt"
        values = label_path.read_text(encoding="utf-8").split()
        assert values[0] == "0"
        assert len(values) == 7
        assert all(float(value) <= 1.0 for value in values[1:])

    summary = json.loads(
        (output_root / "dataset_summary.json").read_text(encoding="utf-8")
    )
    assert summary["duplicate_annotations_removed"] == 0


def test_cli_removes_exact_duplicates_and_reports_count(
    workspace_tmp: Path, monkeypatch, capsys
) -> None:
    annotation_root = workspace_tmp / "annotations" / "scene"
    image_root = workspace_tmp / "images" / "scene"
    annotation_root.mkdir(parents=True)
    (image_root / "left").mkdir(parents=True)
    (image_root / "right").mkdir(parents=True)

    image = np.zeros((16, 20), dtype=np.uint8)
    cv2.imwrite(str(image_root / "left" / "000.png"), image)
    cv2.imwrite(str(image_root / "right" / "000.png"), image)
    duplicate_points = [[2, 2], [10, 2], [6, 8]]
    data = {
        "version": "5.2.1",
        "imageWidth": 20,
        "imageHeight": 16,
        "shapes": [
            {"label": "metal_submarine", "shape_type": "polygon",
             "points": duplicate_points},
            # Same class after label normalization and the same converted
            # polygon: an exact duplicate that must be removed.
            {"label": "Metal Submarine", "shape_type": "polygon",
             "points": duplicate_points},
            # Same polygon but a different class: not a duplicate.
            {"label": "plastic_submarine", "shape_type": "polygon",
             "points": duplicate_points},
        ],
    }
    json_path = annotation_root / "000.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    original_json = json_path.read_text(encoding="utf-8")

    output_root = workspace_tmp / "yolo"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_yolo_dataset.py",
            "--annotations",
            str(annotation_root.parent),
            "--image-root",
            str(image_root.parent),
            "--output",
            str(output_root),
            "--classes",
            "metal_submarine,plastic_submarine",
            "--val-ratio",
            "0",
            "--test-ratio",
            "0",
            "--split-by",
            "group",
            "--clean",
        ],
    )

    from scripts.prepare_yolo_dataset import main

    main()

    label_files = sorted((output_root / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 1
    lines = label_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [line.split()[0] for line in lines] == ["0", "1"]

    summary = json.loads(
        (output_root / "dataset_summary.json").read_text(encoding="utf-8")
    )
    assert summary["duplicate_annotations_removed"] == 1
    assert summary["splits"]["train"]["annotations"] == {
        "metal_submarine": 1,
        "plastic_submarine": 1,
    }

    assert "Removed 1 exact duplicate annotations." in capsys.readouterr().out
    assert json_path.read_text(encoding="utf-8") == original_json


def test_crop_helpers_read_and_localize_yolo_polygons(workspace_tmp: Path):
    label_path = workspace_tmp / "sample.txt"
    label_path.write_text("0 0.2 0.2 0.4 0.2 0.3 0.4", encoding="utf-8")
    polygons = read_yolo_polygons(label_path, width=20, height=10)

    assert len(polygons) == 1
    window = make_crop_window(polygons[0].polygon, width=20, height=10, padding=2)
    localized = crop_polygon(polygons[0].polygon, window)

    assert (window.x1, window.y1, window.x2, window.y2) == (2, 0, 10, 6)
    assert localized == ((2.0, 2.0), (6.0, 2.0), (4.0, 4.0))


def test_make_polar_dataset_crop_mode_matches_inference_crops(
    workspace_tmp: Path, monkeypatch
):
    annotation_root = workspace_tmp / "annotations" / "scene"
    image_root = workspace_tmp / "images" / "scene"
    annotation_root.mkdir(parents=True)
    (image_root / "left").mkdir(parents=True)
    (image_root / "right").mkdir(parents=True)

    image = np.full((16, 20), 128, dtype=np.uint8)
    data = {
        "version": "5.2.1",
        "imageWidth": 20,
        "imageHeight": 16,
        "shapes": [
            {
                "label": "metal_submarine",
                "shape_type": "polygon",
                "points": [[2, 2], [10, 2], [6, 8]],
            }
        ],
    }
    for stem in ("000", "001", "002", "003"):
        cv2.imwrite(str(image_root / "left" / f"{stem}.png"), image)
        cv2.imwrite(str(image_root / "right" / f"{stem}.png"), image)
        (annotation_root / f"{stem}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    yolo_root = workspace_tmp / "yolo"
    prepare_argv = [
        "prepare_yolo_dataset.py",
        "--annotations",
        str(annotation_root.parent),
        "--image-root",
        str(image_root.parent),
        "--output",
        str(yolo_root),
        "--classes",
        "metal_submarine",
        "--val-ratio",
        "0.5",
        "--split-by",
        "block",
        "--block-size",
        "2",
        "--clean",
    ]
    monkeypatch.setattr(sys, "argv", prepare_argv)
    from scripts.prepare_yolo_dataset import main as prepare_main

    prepare_main()

    polar_root = workspace_tmp / "polar_crops"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_polar_dataset.py",
            "--manifest",
            str(yolo_root / "pair_manifest.csv"),
            "--labels-root",
            str(yolo_root / "labels"),
            "--output",
            str(polar_root),
            "--workspace",
            ".",
            "--format",
            "png3",
            "--crop",
            "--crop-pad",
            "2",
            "--max-disp",
            "1",
            "--window",
            "3",
            "--clean",
        ],
    )
    from scripts.make_polar_dataset import main as make_main

    make_main()

    for split in ("train", "val"):
        crops = sorted((polar_root / "images" / split).glob("*.png"))
        labels = sorted((polar_root / "labels" / split).glob("*.txt"))
        assert len(crops) == 2
        assert [path.stem for path in crops] == [path.stem for path in labels]
        for crop, label in zip(crops, labels):
            assert cv2.imread(str(crop)) is not None
            values = label.read_text(encoding="utf-8").split()
            assert values[0] == "0"
            assert len(values) == 7
            assert all(0.0 <= float(value) <= 1.0 for value in values[1:])