# External worker task

## Objective

Complete the original evaluation/defaults implementation left at RED: make the newly added tests pass with production code, without expanding scope.

## Allowed scope

- Inspect and modify only: `scripts/eval_segmentation.py` (new), `tests/test_eval_segmentation.py`, `scripts/eval_speed.py`, `tests/test_eval_speed.py`, `scripts/rectify_stereo_dataset.py`, `scripts/prepare_yolo_dataset.py`, `tests/test_rectify_stereo_dataset.py`, `tests/test_prepare_yolo_dataset.py`, `configs/default.yaml`, `gui/camera_thread.py`, and `tests/test_camera_thread.py`.

## Requirements

- Codex independently confirmed RED with the exact validation command: missing `scripts.eval_segmentation` and missing `parse_args` in `rectify_stereo_dataset`.
- Implement the behavior specified by the tests and the original task: lazy Ultralytics segmentation test evaluation with stable JSON; backward-compatible model speed benchmark plus end-to-end `DualStageInferenceEngine.from_config` mode with synchronized timing and JSON; v2 defaults; remove stale nonexistent `data.polar_root`; accurate current-GUI mock hint.
- Review the tests critically and correct them only if they assert an API shape incompatible with the actual project or Ultralytics metrics contract. Do not reduce coverage.
- Keep Windows compatibility, public existing helpers, type annotations for new functions, and no real model/GPU execution.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not delete/move files or write under `datasets/`, `runs/`, `analysis/`, `archives/`, or project dependency files.
- Do not install/update dependencies, access networks/credentials, or alter Git state.
- Do not weaken, skip, or delete existing tests.

## Validation

- Run: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_eval_segmentation.py tests/test_eval_speed.py tests/test_rectify_stereo_dataset.py tests/test_prepare_yolo_dataset.py tests/test_camera_thread.py -q -W error -p no:cacheprovider`
- Report compileall as not run; Codex will run it because only one Bash family is granted.

## Return contract

Return status, files changed, exact pytest result, and unresolved risks/assumptions.
