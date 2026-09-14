# External worker task

## Objective

Integrate the verified runtime engine into the PySide6 application without blocking the UI, and make stereo acquisition metadata/synchronization explicit so unreliable pairs cannot be presented as valid depth.

## Allowed scope

- Inspect: `gui/camera_thread.py`, `gui/inference_panel.py`, `gui/main_window.py`, `gui/capture_panel.py`, `gui/inference_engine.py`, `core/rectification.py`, `configs/default.yaml`, and existing tests.
- Modify: the named `gui/` files, `gui/inference_engine.py` only for a backward-compatible detailed-result/already-rectified API needed by the panel, `configs/default.yaml`, focused tests under `tests/`, and `README.md` only if a short GUI launch note needs correction.
- Do not modify model training, generated datasets, or calibration files.

## Requirements

- Extend `FrameBundle` with deterministic pair/frame ID, left/right frame numbers when available, left/right monotonic or device timestamps in nanoseconds, `sync_skew_ms`, and `rectified`. Preserve existing fields/default construction compatibility.
- Make MVS frame conversion return or expose frame metadata without assuming every SDK struct field exists. Record frame number and timestamp fields through `getattr` fallbacks. Do not claim independent camera device clocks are synchronized.
- Add a configurable trigger source supporting current software trigger and shared hardware line trigger. Configure both cameras consistently; issue `TriggerSoftware` commands only in software mode. Keep software as a safe default unless the existing config says otherwise. Store the host command timestamps used to estimate software-trigger skew; for shared hardware trigger represent skew as unavailable rather than inventing a zero measurement unless comparable timestamps are actually available.
- Add camera config values for maximum trusted skew (default 2.0 ms), trigger source, and whether incoming/playback frames are already rectified. Propagate these into bundles.
- Fix `_make_mock_pair`: for positive left-minus-right disparity, the right-view content must appear to the left by 18 px (`x_right = x_left - disparity`) for base texture, rectangle, and circle. Mock/playback bundles need stable sequential IDs and timestamps.
- Normalize grayscale at camera/playback/statistics boundaries with `to_gray_u8`, including the Ultralytics-patched `(H,W,1)` case.
- Add a backward-compatible detailed engine result containing the rectified left/right grayscale images, instances, depths, and polar map, while preserving existing `process_frame` tuple callers. Accept `already_rectified` so pre-rectified playback is not remapped twice.
- Implement inference off the GUI thread with a `QThread` worker and a bounded latest-frame queue of capacity 1. Submitting a newer frame while inference is busy must replace/drop the stale pending frame, never grow an unbounded queue and never process synchronously in the camera signal handler. Provide clean start/stop/wait lifecycle and surface exceptions through a signal.
- The worker passes bundle `sync_skew_ms` and `rectified` to the engine. A missing/unavailable skew must remain distinguishable from a measured zero; the engine may process it but the panel diagnostic must state sync is unavailable rather than pretending it is measured.
- Initialize panel defaults from `configs/default.yaml`: expected Model A/B paths, calibrated baseline/focal, max disparity 768, block size 7, and sync threshold. Build the engine through `DualStageInferenceEngine.from_config` with explicit UI overrides. Device selection must remain torch-based inside the engine; remove any `cv2.cuda` device decision.
- Ensure max-disparity UI supports at least 768 (prefer a stable numeric spin box or a range up to 1024); enforce odd valid block size.
- Display/draw on the rectified images returned by the detailed result, not raw images under rectified masks. Make QImage objects detach/copy their NumPy backing memory. Handle grayscale/BGR inputs robustly. Invalid depth is shown as unavailable with its reason; never format `None` as a float or show `0.00m` as a valid range.
- Keep the interface operational and compact at the existing 1500x900 window: do not require three vertically stacked 640x480 minimum panels. Use stable responsive image areas and avoid UI overlap.
- Preserve capture-panel behavior. Add new frame/sync fields to capture manifests if straightforward, without breaking old records.
- Add headless/offscreen-safe tests for FrameBundle metadata, mock disparity sign, trigger behavior using fake camera objects, gray boundary, latest-frame replacement, detailed engine compatibility/already-rectified behavior, panel config defaults, and result formatting/handling. Do not require real MVS SDK, cameras, or weights.

## Prohibited actions

- Do not install dependencies, access networks, train/download models, connect to real cameras, launch a persistent GUI, or perform Git operations.
- Do not modify generated datasets or files outside the allowed scope.
- Do not weaken, remove, or skip existing tests.
- Do not add an unbounded queue or run inference directly in `_on_frame`.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_camera_thread.py tests/test_inference_panel.py tests/test_inference_engine.py -q`

## Return contract

Return status, files inspected/changed, thread/queue lifecycle, synchronization semantics, exact validation result, and hardware assumptions still requiring physical validation.

## Bounded retry findings

Codex independently ran the requested tests: `91 passed`, but pytest emitted 24 PySide disconnect warnings. Code review found the following runtime gaps. Correct these within the original scope and add focused tests; this is the single bounded retry.

- Both panels assign the new camera before `_wire_signals`, then try to disconnect signals from the new, never-connected camera. Track the previously wired camera (or add a safe `set_camera`) and disconnect only known old connections. Connect button actions once. Rebinding a camera must produce no PySide disconnect warnings and must leave no callback attached to the old camera. Update `MainWindow` to use the safe rebinding API.
- `InferenceWorker.stop()` must not return while its QThread is still running. Use a clean stopping flag/condition and an unconditional wait for the current inference call to finish in the panel shutdown path. Ignore or disconnect late results after `_running` becomes false so a queued result cannot repopulate cleared UI state. Add a lifecycle test asserting `not worker.isRunning()` after stop and after panel deinit.
- The three inherited `ImageView` widgets each have a 420 px minimum width; placing all three horizontally plus controls exceeds a 1500 px window. Use a compact responsive 2x2/grid arrangement (e.g. left/right on top and polar below) and/or inference-specific smaller stable minimums so the layout's minimum width fits comfortably inside 1500. Add a basic size/layout assertion.
- Expose camera trigger source (`software`/shared hardware line), maximum trusted skew, and already-rectified input as actual capture-panel controls using appropriate combo/spin/checkbox widgets. `_read_config` must return their values. When building the inference engine, use the current camera config's `max_sync_skew_ms` if available so the acquisition setting is not dead data.
- Remove the 24 expected disconnect warnings rather than filtering them. Preserve all passing behavior and rerun the exact validation command.
