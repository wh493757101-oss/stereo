# External worker task

## Objective

Correct the verified training-device forwarding defect in the phase 5 implementation and lock the behavior with focused tests.

## Allowed scope

- Inspect and modify only `scripts/train_models.py` and `tests/test_model_training.py`.

## Requirements

- Ensure the device accepted by `train_stage(..., device=...)` is passed explicitly to `model.train(**kwargs)` after CUDA availability validation.
- Preserve the public behavior of `build_train_kwargs`; choose the smallest implementation change that makes the effective training kwargs explicit and testable.
- Add or update a test proving both CUDA-style device `"0"` and explicit `"cpu"` reach the mocked `model.train` call.
- Keep all prior phase 5 tests passing.

## Prohibited actions

- Do not modify any other file.
- Do not install dependencies, access networks, instantiate a real Ultralytics model, train models, download weights, or perform Git operations.
- Do not weaken, remove, or skip tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_model_training.py tests/test_classification.py tests/test_ablation_eval.py tests/test_eval_speed.py -q`

## Return contract

Return status, files changed, the exact code-level correction, and the exact validation result.
