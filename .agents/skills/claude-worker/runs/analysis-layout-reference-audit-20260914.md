# External worker task

## Objective

Produce a read-only inventory that classifies every current file directly under `analysis/` and identifies every project reference that would need updating if training-specific reports are moved into `analysis/runs/<run-id>/` directories.

## Allowed scope

- Inspect `analysis/`, `runs/train/*/run.json`, `scripts/`, `tests/`, `configs/`, root Markdown files, and other tracked project text files needed to locate path references.
- Files that may be modified: none.

## Requirements

- Classify each direct child file of `analysis/` as data/calibration audit, training-run evaluation, general benchmark, or legacy/duplicate.
- Recommend exact destination paths for training-related reports, preserving filenames unless a collision or ambiguity requires a rename.
- Find literal and programmatic references to every report proposed for movement.
- Flag reports that appear byte-identical or semantically duplicated, but do not delete anything.
- Keep `analysis/runs/<run-id>/` aligned with the existing `runs/train/<run-id>/` grouping.

## Prohibited actions

- Do not modify any file.
- Do not move, rename, or delete files.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- Report all inspected paths and path references found.
- Do not run training, inference, or model evaluation.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- proposed old-to-new path mapping
- references that must be updated
- duplicate/legacy findings
- unresolved risks or assumptions
