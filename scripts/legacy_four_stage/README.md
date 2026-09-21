# legacy_four_stage — archived four-stage training code (2026-09-21)

**Not part of the current mainline.** The active training pipeline is the
three-stage `scripts/train_pipeline.py`
(`model_a` → `gray_fusion` → `fusion_freeze`). This directory preserves the
retired four-stage implementation so historical runs stay reproducible.

## Original stages and purpose

| Stage | Purpose |
|---|---|
| `baseline` | 4-class YOLO-seg reference model (grayscale copies) |
| `a` | binary YOLO-seg, single class `target` (the old Model A) |
| `b-gray` | YOLO-cls on `[gray, gray, gray]` crops (old gray Model B) |
| `b-polar` | YOLO-cls on `[gray, polar, gray]` crops (old polar Model B) |

## Original path → archived path

| Original path | Archived path |
|---|---|
| `scripts/train_models.py` | `scripts/legacy_four_stage/train_models.py` |
| `scripts/train_all_models.py` | `scripts/legacy_four_stage/train_all_models.py` |
| `scripts/train_seg.py` | `scripts/legacy_four_stage/train_seg.py` |

The original paths now hold retirement stubs that only print a notice and
exit non-zero; they never train and never forward to the new pipeline.
Common helpers (run-id validation, device resolution, project anchoring,
base constants) are imported from the active `scripts/training_common.py`
instead of being duplicated.

## Explicit usage (archived code)

```powershell
python scripts/legacy_four_stage/train_all_models.py --run-id <run-id> --device 0
python scripts/legacy_four_stage/train_models.py --stage a --run-id <run-id> --device 0
python scripts/legacy_four_stage/train_seg.py --stage a --data <data.yaml> --run-id <run-id>
```

## Historical weights and reports

Historical four-stage outputs remain exactly where they were produced and
were not moved or rewritten by this archive: the accepted weights under
`runs/train/run_20260913_initial/`, the candidate run
`runs/train/run_20260914_manual_labels/`, and the reports under
`analysis/runs/`. They are read-only historical artifacts.

## Status

Archived code is not maintained, is not covered by the active pipeline's
gates, and must not be used for new formal training.
