# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Underwater stereo-polarization target detection and ranging system (Python 3.11, PyTorch CUDA 13, Ultralytics, PySide6). Runtime pipeline: synchronized left/right frames → stereo rectification → Model A (YOLO-seg binary instance segmentation) → horizontal band SGBM (one dense disparity pass per frame, shared by all instances) → robust median disparity depth + per-pixel polar differential → Model B (YOLO-cls 4-class batch classification, default `gray` input mode) → GUI display.

Authoritative docs: `README.md` (usage), `HANDOVER.md` (acceptance state, invariants, remaining work), `configs/default.yaml` (all runtime/training defaults), `analysis/project/project_inventory_v2.json` (machine-readable asset hashes).

## Critical invariants (do not violate)

1. MATLAB rotation matrices enter OpenCV as `R_cv = R.T`.
2. Left camera = 0° polarizer, right = 90°; polar feature is `|L - warp(R)| / (L + warp(R))`.
3. Class ids are fixed: `metal_submarine=0`, `plastic_submarine=1`, `plastic_fish=2`, `real_fish=3`.
4. Model B crop padding is 10 px (training and inference must match).
5. Channel modes are fixed: `gray` = `[gray, gray, gray]`, `polar` = `[gray, polar, gray]`.
6. Datasets are split by complete capture group (train/val/test = 765/180/180 images, 17/4/4 groups). Never split a continuous capture sequence across splits.
7. Without measurable timestamps, sync status must be reported as `unknown`, never `0 ms`.
8. Training runs require a fresh `--run-id` (ASCII letters/digits/`.`/`_`/`-` only); outputs go to `runs/train/<run-id>/model_<stage>/`. Existing target directories are never overwritten.
9. Never delete: original `datasets/Single`/`Mixed`, `datasets/bd_image`, any v2 datasets, the four official `best.pt` weights under `runs/train/run_20260913_initial/`, or polar ablation assets.
10. Claims like "polarization helps", "real-time", or "depth is accurate" must be backed by the paired ablation bootstrap, full-pipeline benchmark, or distance ground truth respectively. Current honest state: polar gain NOT proven (default is `gray`), pipeline ≈ 6 FPS (NOT 30 FPS real-time), no distance accuracy validation yet.
11. The dev base model is YOLO26nano (`yolo26n-seg.pt` / `yolo26n-cls.pt`, see `SEG_BASE`/`CLS_BASE` in `scripts/train_models.py`). Local `yolo26n.pt` is a detection model and must never be used as a Model A/B base.

## Commands

Environment: validated Python 3.11 / CUDA 13 env at `D:/Python/CondaPkgs/stereo/python.exe`. Use this interpreter for project commands instead of an unspecified `python`. Install deps with `pip install -r requirements.txt` only when the task explicitly authorizes it; export extras (`requirements-export.txt`) only for ONNX/TensorRT.

```powershell
# Full test suite (Windows: -p no:cacheprovider is required)
D:/Python/CondaPkgs/stereo/python.exe -m pytest -q -W error -p no:cacheprovider --basetemp="$env:TEMP\stereo_pytest_tmp"

# Single test
D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_stereo_matching.py::<test_name> -v --basetemp="$env:TEMP\stereo_pytest_tmp"

# Launch GUI
D:/Python/CondaPkgs/stereo/python.exe -m gui.main_window

# Train all four stages (order: baseline, a, b-gray, b-polar), or resume selected stages
D:/Python/CondaPkgs/stereo/python.exe scripts/train_all_models.py --device 0 --run-id <run-id>
D:/Python/CondaPkgs/stereo/python.exe scripts/train_all_models.py --device 0 --run-id <run-id> --stages b-gray b-polar

# Train one stage
D:/Python/CondaPkgs/stereo/python.exe scripts/train_models.py --stage <baseline|a|b-gray|b-polar> --device 0 --run-id <run-id>

# Polar fusion model (V3/V4 npz dataset): dataset → train → eval
D:/Python/CondaPkgs/stereo/python.exe scripts/prepare_cls_fusion_dataset.py --clean   # writes datasets/underwater_cls_fusion_v4_band
D:/Python/CondaPkgs/stereo/python.exe scripts/train_polar_fusion.py --phase freeze    # then --phase joint; --dry-run to validate
D:/Python/CondaPkgs/stereo/python.exe scripts/eval_polar_fusion.py

# Evaluation
D:/Python/CondaPkgs/stereo/python.exe scripts/eval_segmentation.py --device 0 --output analysis/runs/<run>/segmentation_test.json
D:/Python/CondaPkgs/stereo/python.exe scripts/eval_ablation.py --device 0 --output analysis/runs/<run>/model_b_ablation_test.json
D:/Python/CondaPkgs/stereo/python.exe scripts/eval_speed.py --mode pipeline --config configs/default.yaml --left <left.png> --right <right.png> --already-rectified --device 0 --warmup 5 --n 30 --output <out.json>

# Regenerate datasets (rarely needed; order matters; --clean replaces generated dirs)
D:/Python/CondaPkgs/stereo/python.exe scripts/rectify_stereo_dataset.py --calib datasets/bd_image/stereo_calib.npz --r-convention matlab
D:/Python/CondaPkgs/stereo/python.exe scripts/prepare_yolo_dataset.py --clean
D:/Python/CondaPkgs/stereo/python.exe scripts/prepare_binary_seg_dataset.py --clean
D:/Python/CondaPkgs/stereo/python.exe scripts/prepare_cls_paired_datasets.py --clean
D:/Python/CondaPkgs/stereo/python.exe scripts/prepare_cls_fusion_dataset.py --clean
```

Training fixes seed `2026`, deterministic mode, AMP, early stopping. Windows uses `workers=0` (CUDA DLL reload failure in subprocesses).

**Windows pytest quirk:** pyproject.toml sets `--basetemp=tests/_tmp_pytest`, but that directory and `.pytest_cache` can get locked (WinError 5) and break `tmp_path`-based tests. Override with `--basetemp="$env:TEMP\..."` as above. In new test modules, prefer a `tempfile.mkdtemp()`-based fixture over pytest's `tmp_path`. conftest.py adds the repo root to `sys.path` and ignores `_tmp_*` dirs.

Dataset commands using `--clean` replace generated directories. Run them only
when the task explicitly authorizes regeneration and first verify the target
directory; never use them as a routine troubleshooting step.

## Architecture

```
core/
  rectification.py    StereoRectifier: NPZ calib → rectification maps → rectify(left, right)
  stereo_matching.py  StereoMatcher (SGBM), HorizontalBand + build_horizontal_bands/compute_bands,
                      disparity_to_depth, InstanceDepth
  polar_compute.py    compute_polar_features (signed_q/abs_q/valid masks), warp_with_disparity,
                      build_polar_yolo_image
  fusion_dataset.py   V3/V4 fusion npz I/O: FusionSampleArrays, 4-dim quality vector contract
models/
  segmentation.py     SegmentationModel → list[Instance]; predict_polar for [gray, polar, gray]
  classification.py   ClassificationModel with predict / predict_batch
  polar_fusion.py     PolarFusionModel (gray backbone + PolarDeltaNet + PolarGateNet over quality
                      vector), FusionClassifier wrapper
gui/
  main_window.py      MainWindow (PySide6), entry via python -m gui.main_window
  camera_thread.py    CameraThread (QThread): Hikrobot MVS SDK / directory playback / mock
  inference_engine.py DualStageInferenceEngine: orchestrates the full pipeline;
                      process_frame_detailed returns per-stage timings + per-instance polar quality
  inference_panel.py  InferencePanel + InferenceWorker (latest-frame queue, capacity 1)
```

Data flow detail: Model A runs once per frame on a 3-channel gray copy of the left image → instance bboxes merge into non-overlapping horizontal bands → one SGBM pass per frame over merged bands (falls back to single full-frame pass when bands cover nearly the whole image) → all instances reuse the dense disparity; depth uses in-instance robust median disparity, and polar features use per-pixel reliable disparity only (invalid/out-of-bounds pixels → polar 0 with valid_mask).

Fusion pipeline (experimental, gray is still the accepted default): `prepare_cls_fusion_dataset.py` produces numeric npz samples (gray/signed_q/abs_q/valid/quality/class_id + manifest) → `train_polar_fusion.py` trains in `freeze` phase (delta/gate only) then `joint` phase (KL teacher) → `eval_polar_fusion.py` compares gray-only vs fusion with gate statistics.

## Execution handoff protocol

Claude Code is the implementation and execution agent in this repository.
Codex is read-only and supplies a self-contained prompt after the user has
approved the plan. The manually copied prompt is the complete task context.

- A prompt that asks a question, analysis, or plan review is read-only. Edit
  files only when the prompt explicitly authorizes implementation.
- Inspect the actual repository before editing. Preserve the approved design,
  scope, and acceptance criteria; if repository evidence makes them impossible
  or unsafe, stop and report the conflict instead of silently redesigning them.
- Do not add dependencies, change public behavior, or expand scope unless the
  prompt explicitly authorizes it.
- Do not invoke another agent, create an automated worker, or modify Codex,
  Claude Code, CC-Switch, or other global configuration on your own.
- Do not commit, push, reset, clean, checkout, or switch Git state unless the
  prompt explicitly requests it.
- Run the exact validation commands from the prompt when possible. If a
  command cannot run, report the reason and the closest check performed.

## Completion report for Codex

After implementation, return a concise report that the user can paste back to
Codex. Use this structure:

```text
Status: SUCCESS | PARTIAL | BLOCKED | FAILED

Summary:
- What was implemented and the resulting behavior.

Changed files:
- path/to/file.py — short description

Validation:
- `exact command` — PASS/FAIL (include the essential result)

Remaining risks or limitations:
- Known issue, unvalidated assumption, or skipped check; write “None” if none.

Follow-up needed:
- A concrete decision or next task for Codex/user; write “None” if none.
```

Do not claim success when required validation failed or when the approved scope
was not completed.
