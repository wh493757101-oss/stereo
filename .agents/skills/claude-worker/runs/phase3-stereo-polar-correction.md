# External worker task

## Objective

Fix the two failing crop-mode tests left by the timed-out StereoSGBM task, then verify the whole phase-3 test set without changing stereo algorithms.

## Allowed scope

- Inspect and modify only `scripts/make_polar_dataset.py` and `tests/test_make_polar_dataset.py`.
- Inspect but do not modify `core/stereo_matching.py`, `core/polar_compute.py`, `tests/test_stereo_matching.py`, and `tests/test_polar_compute.py`.

## Requirements

- `process_pair_crops` / `write_polar_crop` must create the derived label output directory before writing labels, for both normal and tiny/uniform-image paths.
- Preserve the intended parallel `images/<split>` and `labels/<split>` layout when `image_dir` is a dataset path, and behave sensibly for a generic test image directory.
- Fix production code, not tests, unless an assertion itself is demonstrably inconsistent with the required directory contract.
- Do not change StereoSGBM matching, thresholds, or polar calculations.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install dependencies, access networks, modify datasets, or perform Git operations.
- Do not weaken, remove, or skip tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_stereo_matching.py tests/test_polar_compute.py tests/test_make_polar_dataset.py -q`

## Return contract

Return status, files changed, exact validation result, and remaining risks.
