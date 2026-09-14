# External worker task

## Objective

Finish the existing Model-B input-mode change by making the single failing GUI path assertion platform-neutral.

## Allowed scope

- Inspect: `tests/test_inference_panel.py` and the current pytest failure.
- Modify only: `tests/test_inference_panel.py`.

## Requirements

- Replace POSIX-string `endswith` checks for captured absolute Model-B paths in `TestModelBInputMode.test_init_engine_forwards_path_and_mode` with platform-neutral `Path`-based comparison.
- Preserve the path and mode forwarding assertions for both gray and polar selections.
- Do not change product code or weaken any assertion.

## Prohibited actions

- Do not modify any other file, install dependencies, use network access, modify generated artifacts, or perform Git operations.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_inference_engine.py tests/test_inference_panel.py -q`

## Return contract

Return status, file changed, validation result, and unresolved risks.
