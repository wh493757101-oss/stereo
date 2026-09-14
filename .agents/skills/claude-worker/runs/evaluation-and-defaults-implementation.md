# External worker task

## Objective

Add tested, reproducible CLIs for formal segmentation evaluation and end-to-end stereo pipeline latency measurement, while removing confirmed stale v1 defaults so the runnable code consistently targets the v2 pipeline.

## Allowed scope

- Inspect: `scripts/`, `gui/inference_engine.py`, `gui/camera_thread.py`, `configs/default.yaml`, `tests/`, and v2 dataset metadata/YAML files (manifests and summaries only).
- Modify: `scripts/eval_segmentation.py` (new), `tests/test_eval_segmentation.py` (new), `scripts/eval_speed.py`, `tests/test_eval_speed.py`, `scripts/rectify_stereo_dataset.py`, `scripts/prepare_yolo_dataset.py`, `tests/test_rectify_stereo_dataset.py`, `tests/test_prepare_yolo_dataset.py`, `configs/default.yaml`, `gui/camera_thread.py`, and only directly relevant existing tests.

## Requirements

- Follow test-driven development: add/adjust tests first and run the scoped test command to establish RED before implementation.
- `scripts/eval_segmentation.py` must evaluate the formal `model_baseline` four-class segmentation checkpoint and binary `model_a` checkpoint on their `test` splits through Ultralytics, and write one stable JSON report under a caller-supplied output path. Defaults must match current `runs/train/*` and `datasets/*_v2` paths. Include overall box/mask precision, recall, mAP50 and mAP50-95, per-class box/mask mAP50 and mAP50-95 where exposed by Ultralytics, sample/instance counts where exposed, checkpoint/data/split/device metadata, and clear errors for missing inputs. Structure extraction behind testable functions and lazy-import Ultralytics.
- `scripts/eval_speed.py` must retain the existing single-model benchmark behavior and add an end-to-end mode driven by `DualStageInferenceEngine.from_config`, real left/right image paths, `already_rectified`, configurable warmup/repetitions, and optional JSON output. Measure synchronized wall-clock latency (synchronize CUDA when available), report mean/std/min/max/p50/p95/FPS, and include last-frame instance count, valid-depth count, config/image/device/mode metadata. Structure timing behind testable functions; tests must use fakes/mocks and no real model loading.
- Change `scripts/rectify_stereo_dataset.py` default output to `datasets/Rectified_v2`.
- Change `scripts/prepare_yolo_dataset.py` defaults to `datasets/bz_JSON_v2`, `datasets/Rectified_v2`, and `datasets/underwater_seg_v2`.
- Remove the unused nonexistent `data.polar_root` key from `configs/default.yaml`, and remove/update stale comments claiming formal models do not exist.
- Replace the obsolete `dual_camera_capture.py --mock` hint in `gui/camera_thread.py` with an accurate instruction for the current GUI (do not invent a nonexistent CLI flag).
- Preserve public helpers used by current tests, Windows compatibility, ASCII source unless an existing file already uses Chinese, type annotations for new/changed functions, and project style.
- Do not run formal model evaluation or touch the GPU training session; Codex will run formal commands after training completes.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not delete, move, rename, or write anything under `datasets/`, `runs/`, or `analysis/`.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- RED then GREEN: `D:\Python\CondaPkgs\stereo\python.exe -m pytest tests/test_eval_segmentation.py tests/test_eval_speed.py tests/test_rectify_stereo_dataset.py tests/test_prepare_yolo_dataset.py tests/test_camera_thread.py -q -W error -p no:cacheprovider`
- Compile: `D:\Python\CondaPkgs\stereo\python.exe -m compileall -q scripts gui tests`
- If either command is unavailable or fails for an unrelated environment reason, report exact output and stop.

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- RED and GREEN validation commands and results
- unresolved risks or assumptions
