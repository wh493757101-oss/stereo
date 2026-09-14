# External worker task

## Objective

Replace the obsolete per-instance NCC runtime path with a tested, configuration-driven dual-stage inference core that performs online rectification, one dense SGBM computation per synchronized frame, robust per-instance depth, and true Model-B classification.

## Allowed scope

- Inspect: `gui/inference_engine.py`, `models/segmentation.py`, `models/classification.py`, `core/stereo_matching.py`, `core/polar_compute.py`, `scripts/rectify_stereo_dataset.py`, `scripts/prepare_cls_paired_datasets.py`, `configs/default.yaml`, and existing focused tests.
- Modify: `gui/inference_engine.py`, new `core/rectification.py`, `core/__init__.py` if useful, `models/classification.py`, `configs/default.yaml`, and focused new/updated tests under `tests/`.
- Modify `models/segmentation.py` only if a backwards-compatible optional field is strictly necessary; avoid changing its parsing semantics.
- Do not modify GUI widgets, camera acquisition, generated datasets, or training scripts in this task.

## Requirements

- Add a reusable `StereoRectifier` runtime class which loads the calibration NPZ (`K1,D1,K2,D2,R,T,image_size`), applies the configured rotation convention (`matlab` means `R_cv = R.T`; `opencv` means no transpose), computes `stereoRectify` and both remap pairs exactly once during construction, and exposes `rectify(left, right)` returning contiguous 2-D uint8 images. Validate keys, shapes, convention, and input dimensions with clear errors. Do not import implementation from a `scripts/` module.
- Refactor `DualStageInferenceEngine` to accept 2-D gray, `(H,W,1)`, BGR, or BGRA inputs through the shared `to_gray_u8` boundary. Apply online rectification before inference when enabled.
- Add dependency injection for model A, model B, matcher, and rectifier so tests do not load weights. Preserve constructor use with checkpoint paths and preserve the existing `process_frame(...) -> (instances, depth_results, polar_map)` shape.
- Model A must receive a 3-channel grayscale copy. Model B must be `ClassificationModel`, never `SegmentationModel`, and must receive each crop in the exact offline layout `[gray, polar, gray]` using the shared polar-image builder and the same default crop padding of 10 pixels.
- Extend `ClassificationModel` with an optional inference `imgsz` setting (default compatible with current callers) and forward it only when configured.
- If Model A returns no instances, return without computing dense stereo.
- For a valid synchronized frame with instances, call `StereoMatcher.compute` exactly once. Reuse that result for every instance through `instance_stats`/`object_disparity_map`; use the same robust object disparity for both depth and polar warping. Compute polar from original rectified intensities, only inside each instance mask. Invalid stereo produces zero polar for that instance, matching the generated training data.
- Accept optional `sync_skew_ms` in `process_frame`. Add a configurable default maximum of 2.0 ms. If skew exceeds it, still run Model A and optional gray/zero-polar Model B, but do not call stereo matching; every depth result must be invalid with reason `sync_skew_exceeded`, and polar must contain no pseudo-correspondence signal.
- Each depth record must include `instance_id`, full-resolution disparity, valid ratio/confidence, `depth` (meters or null), `valid`, and a machine-readable `reason`. Never report `0.0 m` as a valid distance.
- Add a YAML-backed constructor/factory using `configs/default.yaml`: calibrated baseline/focal, rectification file/convention/enabled/alpha, the entire `stereo` section, model paths/thresholds/imgsz, crop padding, and sync threshold. Resolve relative paths against project root or config location deterministically.
- Device selection must use `torch.cuda.is_available()`, not OpenCV CUDA. `auto` selects `0` when torch CUDA is usable, otherwise `cpu`; an explicit CUDA request must fail clearly if unavailable.
- Update config runtime model paths to the expected trained checkpoints (`runs/train/model_a/weights/best.pt`, `runs/train/model_b-polar/weights/best.pt`), Model B imgsz 224, and add explicit runtime crop/sync settings. Do not claim the checkpoints exist.
- Keep APIs testable without a display or MVS SDK. Add focused tests for rotation convention/map initialization, input dimensionality, config wiring, torch device selection, grayscale-copy Model-A input, no-instance short-circuit, exactly one dense compute for multiple instances, true classifier usage/layout, invalid stereo zero polar, and sync-skew invalidation.

## Prohibited actions

- Do not modify GUI widgets/panels, camera code, generated datasets, or unrelated modules.
- Do not install dependencies, access networks, instantiate real Ultralytics models in tests, train/download weights, launch a GUI, or perform Git operations.
- Do not weaken, remove, or skip tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_rectification.py tests/test_inference_engine.py tests/test_classification.py tests/test_stereo_matching.py tests/test_polar_compute.py -q`

## Return contract

Return status, files inspected/changed, architecture decisions, exact validation result, and remaining assumptions for GUI/camera integration.

## Bounded retry findings

Codex independently ran the requested tests: `96 passed, 1 skipped`. Code review found the following behavioral gaps. Correct them within the original scope and add focused tests; this is the single bounded retry.

- `resolve_device` must treat numeric IDs (`"0"`, `"1"`) and `"cuda:N"` as explicit CUDA requests. They must raise clearly when `torch.cuda.is_available()` is false. `cpu` remains valid and `auto` remains adaptive.
- Runtime must exactly match the offline builder: only `stats.valid` may produce a depth value or nonzero polar. In particular, a `low_valid_ratio` result can contain a positive robust disparity/object map but is invalid; it must yield `depth=None`, `valid=False`, and an all-zero polar contribution. Add this exact regression case.
- Wire Model A's configured `iou_threshold` into `SegmentationModel` construction and its configured `imgsz` into Model-A `predict`; make both constructor-injectable with backwards-compatible defaults. Extend config wiring tests.
- Normalize per-call synchronization skew with `abs(...)` so a signed timestamp difference cannot bypass the threshold.
- When a valid classifier result has no mapped name, assign a stable fallback such as `class_<id>` rather than `None` to `Instance.class_name`.
- Tighten `StereoRectifier` array validation so malformed `T` length and non-finite distortion vectors produce explicit `ValueError` messages rather than an incidental reshape/OpenCV exception.
- Re-run the exact validation command from this task.
