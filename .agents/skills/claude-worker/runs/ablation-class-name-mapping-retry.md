# External worker task

## Objective

Finish the existing class-name mapping fix by correcting the one failing regression test fixture.

## Allowed scope

- Inspect: `scripts/eval_ablation.py`, `tests/test_ablation_eval.py`, and the prior pytest failure supplied by the current files.
- Modify only: `tests/test_ablation_eval.py`. Modify `scripts/eval_ablation.py` only if the exact validation reveals a genuine implementation defect rather than the known fixture issue.

## Requirements

- In `TestSwappedCheckpointIds.test_swapped_ids_but_correct_names_score_perfect`, create each synthetic image under both gray and polar roots. The current loop leaves `root` bound to only the final root.
- Keep the intended swapped checkpoint-id assertion and all class-name mapping behavior intact.
- Do not broaden the implementation or change report semantics.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install dependencies, modify generated data/reports, or use network access.
- Do not perform any Git operations.
- Do not weaken, skip, or delete tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_ablation_eval.py -q`

## Return contract

Return status, files changed, validation result, and unresolved risks.
