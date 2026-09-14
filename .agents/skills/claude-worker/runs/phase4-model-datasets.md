# External worker task

## Objective

Add tested data builders for a one-class Model-A segmentation dataset and paired gray-only versus polar-aided Model-B classification datasets, all derived from the leakage-free v2 split without duplicating source imagery unnecessarily.

## Allowed scope

- Inspect: `scripts/prepare_yolo_dataset.py`, `scripts/make_polar_dataset.py`, `core/stereo_matching.py`, `core/polar_compute.py`, `datasets/underwater_seg_v2/data.yaml`, `datasets/underwater_seg_v2/pair_manifest.csv`, `datasets/underwater_seg_v2/dataset_summary.json`, and existing tests.
- Modify: new scripts under `scripts/` for these builders, focused new tests under `tests/`, and `configs/default.yaml` only.
- Do not modify existing generated datasets in this worker task.

## Requirements

- Python 3.11+, pathlib, type hints, deterministic behavior, testable pure helpers.
- Model A builder input is `datasets/underwater_seg_v2`; output defaults to `datasets/underwater_seg_binary_v2`. Preserve train/val/test assignments and manifests. Rewrite every nonempty YOLO-seg label class id to `0`, retain empty negative labels, and write `data.yaml` with one class named `target`. Reuse images with hard links where supported and fall back to `shutil.copy2`; expose `--link-mode hardlink|copy` and refuse overlapping input/output roots.
- Model B builder consumes the pair manifest and segmentation labels from the four-class v2 dataset. It creates two roots in one run: `datasets/underwater_cls_gray_v2` and `datasets/underwater_cls_polar_v2`, each with Ultralytics classification layout `<split>/<class_name>/<sample>.png` for train, val, and test.
- Use one `StereoMatcher.compute` call per stereo pair. For each polygon, build a full-image binary mask, get the robust object disparity via `instance_stats`/`object_disparity_map`, compute polar from original left/right intensities and the constant object disparity inside that mask, then make identical padded crop windows for both variants. Gray variant must be `[gray, gray, gray]`; polar variant `[gray, polar, gray]` with polar zero outside the object mask.
- Always emit paired gray/polar crops with identical relative class paths even when stereo is invalid; record stereo validity, reason, valid ratio, disparity, group, split, class, source frame, and both output paths in a CSV manifest and JSON summary. Do not silently drop difficult examples.
- Classification samples must derive class id from each YOLO polygon, support multiple objects per frame, reject unknown ids, and preserve group-level split isolation.
- Add CLI options for crop padding and all SGBM settings already in `StereoMatcherConfig`. Refuse output roots that overlap each other or source roots. Do not provide HSV/color augmentation here.
- Add config data paths for binary segmentation, gray classification, and polar classification outputs.
- Tests must cover class-id collapse, empty negatives, hardlink fallback, output-root safety, multiple objects, exact paired paths/content, one dense compute per pair, constant object disparity path, invalid stereo retention, three splits/four classes, and deterministic manifests.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install dependencies, access external networks, or perform Git operations.
- Do not generate full datasets in this worker task.
- Do not weaken, delete, or skip existing tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_prepare_model_datasets.py -q`

## Return contract

Return status, files inspected, files changed, exact validation result, and remaining risks.
