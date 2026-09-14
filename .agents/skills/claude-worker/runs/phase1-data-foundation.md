# External worker task

## Objective

Produce a tested, non-destructive v2 data-preparation toolchain that uses the correct MATLAB-to-OpenCV stereo rotation convention, migrates existing Labelme polygons into the new rectified coordinate system, retains negative frames, and creates leakage-free train/validation/test splits by complete capture group.

## Allowed scope

- Inspect: `scripts/rectify_stereo_dataset.py`, `scripts/prepare_yolo_dataset.py`, `tests/`, `configs/default.yaml`, `README.md`, `HANDOVER.md`, `datasets/bd_image/stereo_calib.npz`, and small sampled files under `datasets/bz_JSON/`, `datasets/Rectified/`, `datasets/Single/`, and `datasets/Mixed/` when needed to understand schemas and paths.
- Modify: `.gitignore`, `pyproject.toml`, `scripts/rectify_stereo_dataset.py`, `scripts/prepare_yolo_dataset.py`, new Python modules under `scripts/`, tests under `tests/`, and `configs/default.yaml` only.
- Do not generate or modify bulk data under `datasets/` in this worker task.

## Requirements

- Follow Python 3.11+, PEP 8, pathlib, type hints, small pure/testable functions, and existing project conventions.
- Add tests before or alongside implementation. Preserve every existing test.
- In calibration loading/rectification, make the matrix convention explicit and default this MATLAB-exported file to `R_cv = R_matlab.T`, with a CLI/configurable escape hatch for calibration files already expressed in OpenCV convention. Validate matrix/vector/image-size shapes and save provenance metadata plus `R1`, `R2`, `P1`, `P2`, and `Q` in rectification output.
- Add a deterministic annotation migration command. It must transform Labelme shape points from old rectified-left coordinates to new rectified-left coordinates using camera-ray geometry derived from the old and new rectification maps/matrices (an equivalent analytic homography is acceptable), clip coordinates, preserve labels and relevant Labelme metadata, remove embedded `imageData` by default, update `imagePath`, `imageWidth`, and `imageHeight`, and write only to a distinct output root.
- The migration command must discover every new rectified left image. If a matching old JSON does not exist, emit an empty Labelme JSON so known negative frames are retained. It must refuse an output root that aliases the input annotation root or either rectified image root.
- Extend YOLO preparation to support `train`, `val`, and `test`, default to whole-group isolation, and produce deterministic 17/4/4 group counts for the current 25 groups when all are present. Use a deterministic greedy stratification or similarly transparent algorithm to balance classes and turbidity across splits. No `group_name` may occur in more than one split. Continue supporting empty annotations and write manifests/summaries proving group isolation and class counts.
- Add `.gitignore` entries for datasets, generated runs/models, caches, temporary test outputs, IDE files, and common Python artifacts without hiding source/config/tests/docs.
- Add a minimal `pyproject.toml` for Python 3.11 and pytest configuration; do not invent a dependency lock or claim CUDA availability.
- Update `configs/default.yaml` with versioned paths and rectification convention fields needed by these tools, while keeping later stereo/model settings for subsequent phases.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.
- Do not overwrite `datasets/Rectified`, `datasets/bz_JSON`, `datasets/Single`, or `datasets/Mixed`.
- Do not run full rectification or copy the full image dataset.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_rectify_stereo_dataset.py tests/test_prepare_yolo_dataset.py tests/test_annotation_migration.py -q`
- If the new test filename differs, include that exact test file in the same pytest invocation.
- Run a rectification CLI dry-run only if needed; do not generate bulk data.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- validation commands and results
- unresolved risks or assumptions
