# External worker task

## Objective

Produce an evidence-based cleanup inventory that identifies project files and directories which are obsolete, duplicated, failed/smoke-only, cache-only, or unreferenced after the v2 pipeline and formal models were implemented.

## Allowed scope

- Read the whole repository tree and project text/code/config files.
- Inspect directory names, sizes, manifests, summaries, and run metadata.
- Do not recursively read image/binary contents; metadata and representative paths are enough.
- Modify: none.

## Requirements

- Classify candidates into: safe to delete now; move/rename and update references; keep for reproducibility; uncertain and why.
- Check root-level legacy Python scripts for imports, docs/config references, unique behavior, and superseding scripts/modules.
- Compare versionless/legacy dataset directories with current `_v2` roots and identify raw-source/calibration/annotation directories that must remain.
- Identify failed, smoke, duplicate/nested, and formal training run directories; formal checkpoints must be kept.
- Identify caches, temporary test output, `__pycache__`, `.pytest_cache`, and generated transient files.
- Assess root pretrained `.pt` files and `datasets.zip` for reproducibility/value before recommending deletion.
- Treat `.agents`, `.claude`, AGENTS.md, project source/tests/config, formal analysis reports, current v2 datasets, raw data, calibration, and formal weights as protected unless there is strong contrary evidence.
- For every delete candidate provide exact repository-relative path, evidence, and any prerequisite.
- Also map HANDOVER.md planned phases to completed/in-progress/pending based on actual artifacts and code.

## Prohibited actions

- Do not modify or delete anything.
- Do not install dependencies, use network access, or access credentials.
- Do not perform Git operations that change state.
- Do not expose secrets or global configuration.

## Validation

- Read-only cross-reference with repository search. Report commands/checks conceptually; do not run tests.

## Return contract

Return a concise audit with exact paths under the four cleanup classes and a phase-status table for the handover plan. Flag any candidate that cannot be decided safely from repository evidence.
