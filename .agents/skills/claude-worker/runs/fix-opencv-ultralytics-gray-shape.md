# External worker task

## Objective

Eliminate the verified Ultralytics/OpenCV import side effect from ordinary model submodule imports and make image boundaries robust to grayscale arrays shaped `(H, W, 1)`, so the full test suite is order-independent and runtime model loading cannot break stereo/polar processing.

## Allowed scope

- Inspect: `models/__init__.py`, `models/export_tensorrt.py`, `models/classification.py`, `core/stereo_matching.py`, `core/polar_compute.py`, `scripts/make_polar_dataset.py`, `scripts/prepare_cls_paired_datasets.py`, and related tests.
- Modify only: `models/__init__.py`, `core/stereo_matching.py`, `core/polar_compute.py`, the two named dataset scripts if genuinely needed, and focused tests under `tests/`.

## Verified defect

- In a fresh process, before importing project models, `cv2.imread(..., IMREAD_GRAYSCALE)` returns `(H, W)`.
- `import models.classification` executes `models/__init__.py`, which eagerly imports `models.export_tensorrt`, which imports Ultralytics and globally replaces `cv2.imread`; afterward grayscale reads return `(H, W, 1)`.
- Running the full suite causes 13 downstream failures, while the affected tests pass alone.

## Requirements

- Preserve `from models import export_model` compatibility without eagerly importing Ultralytics merely from `import models.classification` or `import models`.
- Add a single well-defined core conversion boundary that accepts `(H, W)`, `(H, W, 1)`, and normal 3/4-channel arrays and returns contiguous `uint8` `(H, W)` grayscale. Reject unsupported dimensionality/channel counts with a clear `ValueError`.
- Use that boundary in stereo matching and polar computation so direct callers are robust even after Ultralytics has patched OpenCV.
- Do not silently reinterpret arbitrary multi-channel shapes.
- Add regression tests proving the one-channel case works and ordinary model-submodule import does not eagerly import `models.export_tensorrt`/Ultralytics when those modules were not already loaded.
- Keep API compatibility and all existing tests.

## Prohibited actions

- Do not modify generated datasets, configuration, GUI, camera code, training/evaluation behavior, or files outside the allowed scope.
- Do not install dependencies, access networks, train models, download weights, or perform Git operations.
- Do not weaken or skip tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest -q --basetemp tests/_tmp_full_worker_fix`

## Return contract

Return status, root cause, files changed, exact validation result, and any remaining side-effect risk.

## Bounded retry findings

Codex independently reran the full suite after the first attempt. The result was `8 failed, 162 passed, 1 skipped`. The core conversion now works, but both script-level `load_gray` functions still return Ultralytics-patched `(H, W, 1)` arrays and use them for crop/mask/channel construction before or outside the normalized core calls.

- Update `scripts/make_polar_dataset.py::load_gray` and `scripts/prepare_cls_paired_datasets.py::load_gray` to pass every successful read through the shared `to_gray_u8` boundary.
- Do not duplicate squeezing logic in each script.
- Add focused regression coverage that monkeypatches or simulates `cv2.imread(..., IMREAD_GRAYSCALE)` returning `(H, W, 1)` and proves each loader returns contiguous `uint8 (H, W)`.
- Re-run the exact full-suite validation command. This is the single retry for this task.
