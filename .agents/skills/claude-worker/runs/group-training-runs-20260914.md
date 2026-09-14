# External worker task

## Objective

Implement and document a grouped training-run layout where each training batch is stored under `runs/train/<run-id>/model_*`, while preserving the current Python API when it is called directly without a run id.

## Allowed scope

- Inspect the repository, especially `scripts/train_models.py`, active checkpoint references, documentation, inventories, and relevant tests.
- Modify only: `scripts/train_models.py`, `scripts/train_seg.py`, `configs/default.yaml`, `gui/inference_panel.py`, `scripts/eval_segmentation.py`, `scripts/eval_ablation.py`, `README.md`, `HANDOVER.md`, `runs/README.md`, `runs/run_inventory.json`, `analysis/project_inventory_v2.json`, and tests directly covering those files.
- You may add a small JSON manifest at `runs/train/run_20260913_initial/run.json` if useful.
- Do not move or modify existing model artifacts under `runs/train/model_*`; Codex will perform and verify that filesystem migration separately.

## Requirements

- The accepted historical model set will live at `runs/train/run_20260913_initial/`, containing `model_baseline`, `model_a`, `model_b-gray`, and `model_b-polar`.
- Update active runtime, evaluation, inventory, and documentation references to those grouped checkpoint paths. Preserve historical evaluation-report JSON files unchanged because they are immutable evidence.
- Add a CLI `--run-id` argument to `scripts/train_models.py`; it must be required for CLI execution and place output under `<project>/<run-id>/<model-name>`.
- Keep `train_stage(..., run_id=None)` backward compatible for direct Python callers. When a run id is supplied, resolve the project as `<project>/<run-id>` before passing it to Ultralytics.
- Validate `run-id` as one safe path component: allow ASCII letters, digits, `.`, `_`, and `-`; reject empty, `.`, `..`, slashes, backslashes, absolute paths, and traversal.
- Keep `exist_ok=False`, current model names, dataset roots, seeds, and all training hyperparameters unchanged.
- Update CLI examples to demonstrate a meaningful id such as `run_20260914_manual_labels`.
- Add tests first and demonstrate a focused RED result before production edits, then make them GREEN.
- Do not start training.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not move, rename, delete, or rewrite model `.pt` files or existing training directories.
- Do not modify historical `analysis/*test*.json` reports.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- Run `python -m pytest tests/test_model_training.py tests/test_inference_panel.py tests/test_eval_segmentation.py tests/test_ablation_eval.py -q`.
- Run a repository text search proving no active code/config defaults still point directly to `runs/train/model_*`; historical reports may remain unchanged.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- RED and GREEN validation commands and results
- unresolved risks or assumptions
