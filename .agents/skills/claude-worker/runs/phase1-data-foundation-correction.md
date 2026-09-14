# External worker task

## Objective

Bring the partially implemented phase-1 data foundation to a fully tested state: all pre-existing tests pass, new rectification/migration/group-split behavior has focused tests, and the missing repository/config scaffolding is present.

## Allowed scope

- Inspect: `scripts/rectify_stereo_dataset.py`, `scripts/migrate_labelme_annotations.py`, `scripts/prepare_yolo_dataset.py`, `tests/test_rectify_stereo_dataset.py`, `tests/test_prepare_yolo_dataset.py`, `tests/conftest.py`, `configs/default.yaml`, and the previous task at `.agents/skills/claude-worker/runs/phase1-data-foundation.md`.
- Modify: `.gitignore`, `pyproject.toml`, the three scripts above, `configs/default.yaml`, the two existing tests above, and new `tests/test_annotation_migration.py` only.

## Requirements

- Treat the current script changes as partial work that must be reviewed, not assumed correct.
- Restore all existing test behaviors. Tests that intentionally exercise block splitting must pass `--split-by block`; the production default remains whole-group splitting.
- Add rectification tests proving MATLAB convention transposes `R`, OpenCV convention does not, malformed shapes fail, and saved output metadata includes `R1`, `R2`, `P1`, `P2`, `Q` plus convention provenance.
- Add annotation migration tests for identity mapping, a known non-identity homography, clipping, `imageData` removal, empty negative JSON creation, metadata updates, and refusal to alias an input/output root.
- Add group-split tests proving deterministic assignment, exact 17/4/4 group counts for 25 groups with ratios 0.16/0.16, no group leakage, and train/val/test manifest and summary output including empty annotations.
- Fix implementation defects revealed by those tests without weakening or removing prior assertions.
- Add `.gitignore` entries for `datasets/`, `runs/`, model weights/engines, Python/test caches, temporary test output, and IDE files while keeping source/config/tests/docs trackable.
- Add a minimal `pyproject.toml` requiring Python >=3.11 with pytest discovery/config only. Do not add fake locks or unverified pinned runtime dependencies.
- Update `configs/default.yaml` with `calibration.file: datasets/bd_image/stereo_calib.npz`, `calibration.r_convention: matlab`, `rectification.enabled: true`, `rectification.output_root: datasets/Rectified_v2`, and annotation/output v2 paths in a coherent data section. Preserve unrelated settings for later phases.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.
- Do not read, generate, or modify bulk data under `datasets/`.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_rectify_stereo_dataset.py tests/test_prepare_yolo_dataset.py tests/test_annotation_migration.py -q`

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- validation commands and results
- unresolved risks or assumptions
