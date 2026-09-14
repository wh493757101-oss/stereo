# External worker task

## Objective

Replace the impractical Python NCC paths with one tested OpenCV StereoSGBM implementation shared by offline polar-data generation and online per-instance depth/polar computation, covering the current tank's 0.5 m minimum range.

## Allowed scope

- Inspect: `core/stereo_matching.py`, `core/polar_compute.py`, `scripts/make_polar_dataset.py`, `gui/inference_engine.py` for call contracts only, `configs/default.yaml`, and tests under `tests/`.
- Modify: `core/stereo_matching.py`, `core/polar_compute.py`, `scripts/make_polar_dataset.py`, `configs/default.yaml`, `tests/test_stereo_matching.py`, `tests/test_polar_compute.py`, and focused new tests under `tests/` for this phase only.
- Do not modify GUI files in this task.

## Requirements

- Python 3.11+, type hints, dataclasses for structured results, PEP 8. Preserve public functions where practical with compatibility wrappers.
- Add a reusable `StereoMatcher` configured by full-resolution `max_disparity` (default 768), processing `scale` (default 0.25), odd `block_size` (default 7), uniqueness/speckle parameters, left-right threshold, texture threshold, and minimum valid ratio.
- Internally calculate a low-resolution `numDisparities` rounded up to a positive multiple of 16. Use OpenCV StereoSGBM; no per-pixel or per-disparity Python loops.
- Convert 2-D gray or 3-channel images safely. Matching may use deterministic local contrast normalization such as CLAHE, but return/compute polar values from original fixed intensities.
- Compute both left and right disparity and apply left-right consistency in correct sign convention. Upsample valid disparity to full resolution and divide by scale; invalid values must be zero with a separate boolean/uint8 valid mask.
- Return elapsed time and global valid ratio. For each instance mask, compute median disparity after MAD rejection, valid ratio within the mask, depth in meters, and an explicit validity/reason field. Compute the dense disparity once for all masks.
- Provide an object-level constant disparity map inside a mask for polar warping, derived from the same robust statistic used for depth. Avoid sparse NCC maps.
- `disparity_to_depth` must keep baseline in meters and focal length in full-resolution pixels, return invalid for nonpositive/nonfinite disparity, and avoid rounding inside core calculations.
- Rewrite `compute_full_disparity` as a compatibility wrapper around `StereoMatcher`; remove the nested Python NCC implementation. Keep `compute_disparity_ncc` only as a deprecated compatibility wrapper or replace callers, but it must no longer run Python search loops.
- Update `scripts/make_polar_dataset.py` to instantiate one matcher and use it for every pair, including crop mode. Add CLI options matching the new settings. Offline and online-ready helper paths must call the same matcher/core functions.
- Default config: matcher `sgbm`, `scale: 0.25`, `max_disp: 768`, `block_size: 7`, left-right threshold 2 full-resolution pixels, and minimum valid ratio. Update focal length to 3643.5231322766995; preserve baseline 0.09890970798524992 m.
- Tests must cover validation, image channel handling, output shapes/dtypes, invalid pixels, depth units, mask statistics/MAD rejection, one dense compute for multiple masks, polar uses original intensities, and synthetic textured rectified pairs with full-resolution disparities 0, 64, 384, and 720 pixels. Zero disparity should be handled as invalid/zero-depth evidence, while positive shifts should be recovered within a documented tolerance in a physically visible ROI.
- Add a performance test/benchmark function that can be run manually on 1280x1024 but do not make a fragile wall-clock assertion part of ordinary CI.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install or update dependencies.
- Do not access external networks or credentials.
- Do not perform Git operations.
- Do not generate the full polar dataset or modify existing generated datasets.
- Do not weaken, skip, or delete existing tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_stereo_matching.py tests/test_polar_compute.py tests/test_make_polar_dataset.py -q`
- If `tests/test_make_polar_dataset.py` is not created, run the first two files and explain why coverage is located elsewhere.

## Return contract

Return status, files inspected, files changed, exact tests and results, performance risks, and any compatibility caveats.
