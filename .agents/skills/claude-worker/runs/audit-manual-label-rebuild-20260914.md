# External worker task

## Objective

Produce a read-only audit of the safest downstream data rebuild after 83 manually corrected JSON files under `datasets/bz_JSON_v2`, without training any model.

## Allowed scope

- Inspect `datasets/bz_JSON_v2/migration_manifest.csv`, the current generated dataset manifests/summaries, and a small sample of current JSON files.
- Inspect `scripts/prepare_yolo_dataset.py`, `scripts/prepare_binary_seg_dataset.py`, `scripts/prepare_cls_paired_datasets.py`, `scripts/make_polar_dataset.py`, `configs/default.yaml`, and relevant tests.
- Modify none.

## Requirements

- Identify every downstream dataset/artifact whose contents depend on LabelMe polygon geometry.
- Determine whether deterministic group split settings preserve existing train/val/test assignment when the four-class segmentation dataset is rebuilt.
- Identify validation checks needed before and after rebuilding.
- Explicitly prohibit model training and changes to `runs/train`.
- Account for LabelMe-resaved files containing embedded `imageData` and potentially backslash paths; recommend metadata normalization that preserves manual polygon edits.

## Prohibited actions

- Do not modify any file.
- Do not run data preparation or training.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.

## Validation

- Read-only inspection only; do not run tests or write reports.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- required rebuild order
- split-preservation assessment
- pre/post validation checklist
- unresolved risks or assumptions
