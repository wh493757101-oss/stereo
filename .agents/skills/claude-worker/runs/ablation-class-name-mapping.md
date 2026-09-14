# External worker task

## Objective

Fix Model-B ablation evaluation so predictions are compared in the manifest's canonical class-id space even when Ultralytics assigns checkpoint ids in alphabetical folder order.

## Allowed scope

- Inspect: `scripts/eval_ablation.py`, `models/classification.py`, `tests/test_ablation_eval.py`, `tests/test_classification.py`, and the two generated classification manifests/summaries read-only.
- Modify only: `scripts/eval_ablation.py`, `tests/test_ablation_eval.py`.

## Requirements

- Do not compare `ClassificationResult.top1_id` directly with manifest `class_id`.
- Build and validate a one-to-one canonical `{class_name: class_id}` mapping from the selected paired manifest samples.
- Map every model prediction through `ClassificationResult.top1_name` into that canonical id space before metrics/bootstrap.
- Fail fast with a clear error for missing, unknown, duplicate, or conflicting class name/id mappings. Do not silently fall back to checkpoint numeric ids or `class_<id>` names.
- Keep report schema, paired group bootstrap, decision gate, and model runtime wrapper behavior unchanged.
- Add regression coverage where checkpoint numeric ids for two classes are swapped but names are correct; evaluation must still score perfect predictions. Cover unknown/missing names.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not modify generated datasets or reports.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_ablation_eval.py -q`

## Return contract

Return a concise summary containing status, files inspected, files changed, validation command/result, and unresolved risks or assumptions.
