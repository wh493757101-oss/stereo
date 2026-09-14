# External worker task

## Objective

Make the training entry point reliable on Windows and make YOLO dataset generation remove exact duplicate annotations without modifying source JSON files.

## Allowed scope

- Inspect: `scripts/train_models.py`, `scripts/prepare_yolo_dataset.py`, `tests/test_model_training.py`, `tests/test_prepare_yolo_dataset.py`, `configs/default.yaml`.
- Modify only: `scripts/train_models.py`, `scripts/prepare_yolo_dataset.py`, `tests/test_model_training.py`, `tests/test_prepare_yolo_dataset.py`, `configs/default.yaml`.

## Requirements

- When `train_stage` passes `project` to Ultralytics, resolve a relative project path against the repository root so `runs/train/<name>` is used instead of an Ultralytics task-prefixed nested path. Preserve caller-supplied absolute paths.
- Add a worker-count training argument and CLI override. The default must be `0` on Windows to avoid CUDA DLL loading in spawned data-loader processes; choose a sensible existing-compatible default on other platforms. Forward it to every stage. Document the Windows default in the config.
- Keep gray and polar Model-B training arguments identical except for their data root and run name.
- During Labelme-to-YOLO conversion, remove only exact duplicates after label normalization and polygon conversion. Equality must include class id and converted polygon. Preserve first occurrence order and leave all source and migrated JSON files unchanged.
- Record the total number of removed duplicate annotations in `dataset_summary.json` and report it in CLI output. Existing nonduplicate behavior and split isolation must remain unchanged.
- Add focused regression tests for relative/absolute project resolution, worker forwarding/default behavior, exact duplicate removal, preservation of same polygon with different classes, and summary reporting where practical.
- Follow existing typing and style. Do not modify generated datasets in this task.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_model_training.py tests/test_prepare_yolo_dataset.py -q`
- If this command is not allowed, report it instead of substituting another command.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- validation commands and results
- unresolved risks or assumptions
