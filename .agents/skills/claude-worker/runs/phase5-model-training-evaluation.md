# External worker task

## Objective

Implement tested model wrappers, reproducible training entry points, and paired ablation evaluation for the four-class segmentation baseline, one-class Model A, and gray-only versus polar-aided Model B classifiers.

## Allowed scope

- Inspect: `models/segmentation.py`, `scripts/train_seg.py`, `scripts/eval_speed.py`, `configs/default.yaml`, generated dataset summaries/manifests under `datasets/underwater_*_v2/`, and existing tests.
- Modify: `models/segmentation.py` only if required for compatibility, new `models/classification.py`, `models/__init__.py`, `scripts/train_seg.py`, new training/evaluation scripts under `scripts/`, `scripts/eval_speed.py`, `configs/default.yaml`, and focused tests under `tests/`.
- Do not modify GUI files or generated datasets.

## Requirements

- Add `ClassificationModel` and immutable `ClassificationResult` for Ultralytics classification checkpoints. Accept a pseudo-RGB NumPy crop, pass the configured device, parse `probs.top1`/`top1conf`, and map names robustly. Return an explicit empty/failed result contract when probabilities are absent.
- Keep `SegmentationModel` working for Model A and baseline. Model A output class is `target`; Model B must not use segmentation masks.
- Add one reproducible training CLI supporting stages `baseline`, `a`, `b-gray`, and `b-polar` with defaults: baseline/A `yolov8n-seg.pt`, imgsz 640, batch 8, 100 epochs; B `yolov8n-cls.pt`, imgsz 224, batch 64, 80 epochs. Default data paths must be the generated v2 roots.
- Set seed 2026, deterministic mode, AMP, patience/early stopping, explicit project/name, and save best/last. For B disable HSV/color augmentations (`hsv_h/s/v=0`) because channel semantics are not RGB. Use the same remaining augmentation settings and initialization for b-gray and b-polar.
- Fail fast with a clear message when a CUDA device is requested but `torch.cuda.is_available()` is false. Allow explicit CPU smoke tests without pretending they are production training.
- Keep `scripts/train_seg.py` as a backward-compatible entry point or clear shim; remove unsupported first-convolution surgery from the production path if it conflicts with standard 3-channel datasets.
- Add an ablation evaluation CLI that pairs samples by `dataset_manifest.csv`, verifies identical relative paths/classes/groups across gray/polar roots, runs both classifiers, and outputs JSON containing per-model confusion matrix, per-class precision/recall/F1, macro-F1, accuracy, paired difference, and a deterministic capture-group bootstrap 95% confidence interval. Default at least 2,000 bootstrap resamples, seed 2026. Claim gate is pass only when polar macro-F1 improvement is at least 0.03 and CI lower bound is greater than 0; otherwise decision must say use gray-only/no proven gain.
- Implement pure metric/bootstrap helpers testable without loading real models. Bootstrap groups with replacement and include all samples from sampled groups; do not bootstrap individual adjacent frames.
- Fix `scripts/eval_speed.py` so warmup and benchmark actually pass `device` and `imgsz` to `model.predict`; report p50/p95 latency as well as mean/FPS.
- Update config with baseline, Model A, B-gray, B-polar training/output entries and evaluation gate values. Do not claim models exist or training has completed.
- Tests must mock Ultralytics model objects and cover classification parsing/device use, CUDA fail-fast, paired manifest validation, metrics, deterministic group bootstrap, decision gate, training argument differences/equality, and eval-speed device forwarding.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install dependencies, access networks, train models, download weights, export engines, or perform Git operations.
- Do not weaken, remove, or skip existing tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_model_training.py tests/test_classification.py tests/test_ablation_eval.py tests/test_eval_speed.py -q`

## Return contract

Return status, files inspected, files changed, exact validation result, and remaining runtime assumptions.
