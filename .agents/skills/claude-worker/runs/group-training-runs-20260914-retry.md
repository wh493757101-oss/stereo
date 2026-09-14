# External worker task

## Objective

Complete the partially started grouped training-run implementation so all focused tests pass and active paths use `runs/train/run_20260913_initial/model_*`.

## Allowed scope

- Inspect the repository and the existing tests added by the previous attempt.
- Modify only: `scripts/train_models.py`, `scripts/train_seg.py`, `configs/default.yaml`, `gui/inference_panel.py`, `scripts/eval_segmentation.py`, `scripts/eval_ablation.py`, `README.md`, `HANDOVER.md`, `runs/README.md`, `runs/run_inventory.json`, `analysis/project_inventory_v2.json`, and tests directly covering those files.
- You may add `runs/train/run_20260913_initial/run.json`.
- Do not move or modify existing model artifacts; Codex handles the directory move.

## Requirements

- Treat current tests in `tests/test_model_training.py`, `tests/test_inference_panel.py`, `tests/test_eval_segmentation.py`, and `tests/test_ablation_eval.py` as the acceptance contract. Codex has independently confirmed RED: `TestValidateRunId::test_accepts_safe_ids` fails because `validate_run_id` does not exist.
- Implement `InvalidRunIdError` and safe one-component `validate_run_id` accepting only ASCII letters, digits, `.`, `_`, `-`, while rejecting empty, `.`, `..`, separators, traversal, absolute paths, spaces, punctuation, and non-ASCII.
- Add required CLI `--run-id`; keep direct `train_stage(..., run_id=None)` backward compatible. With a run id, Ultralytics project must be `<resolved project>/<run-id>`, name remains `model_<stage>`, and `exist_ok=False` remains unchanged.
- Update active runtime/evaluation/default/docs/inventory paths to the grouped accepted run. Do not modify historical `analysis/*test*.json` reports.
- Preserve datasets, seeds, hyperparameters, model names, and training behavior otherwise.
- Do not train.
- Use Edit/Write for file changes and Grep/Glob/Read for inspection. Do not attempt `sed`, `grep`, `ls`, shell redirection, or shell pipelines.

## Prohibited actions

- Do not modify files outside allowed scope.
- Do not move, rename, delete, or rewrite `.pt` files or existing run directories.
- Do not install dependencies or access networks/credentials.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not weaken, skip, or delete tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_model_training.py tests/test_inference_panel.py tests/test_eval_segmentation.py tests/test_ablation_eval.py -q -p no:cacheprovider`
- Use the Grep tool, not Bash, to identify remaining direct `runs/train/model_*` active defaults. Historical reports may retain old paths.

## Return contract

Return status, files changed, GREEN result, and unresolved risks. Stop once the focused suite passes and the scoped files are updated.
