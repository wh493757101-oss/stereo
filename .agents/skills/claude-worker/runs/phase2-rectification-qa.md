# External worker task

## Objective

Add a tested CLI that measures vertical epipolar alignment on a rectified stereo dataset and emits machine-readable QA results suitable for gating downstream dataset preparation.

## Allowed scope

- Inspect: `scripts/rectify_stereo_dataset.py`, `datasets/Rectified_v2/rectify_manifest.csv`, and existing tests for style.
- Modify: new `scripts/validate_rectification.py` and new `tests/test_validate_rectification.py` only.
- Read at most six sampled image pairs under `datasets/Rectified_v2/` if genuinely needed; do not modify data.

## Requirements

- Python 3.11+, pathlib, type hints, deterministic behavior.
- Read the rectification manifest, select a configurable number of evenly spaced pairs per group, and evaluate feature matches between left/right grayscale images.
- Prefer OpenCV SIFT with ratio filtering; provide a clear error when SIFT is unavailable. Keep feature count configurable.
- Report per-pair match count, median absolute vertical residual, p95 absolute vertical residual, and horizontal disparity median. Report aggregate per-group and overall results.
- CLI options must include manifest, samples-per-group (default 3), min-matches, max-median-y (default 1.0 px), max-p95-y (default 3.0 px), output JSON path, and a nonzero exit code when any adequately matched pair violates thresholds. Pairs below min-matches must be reported as insufficient, not silently passed.
- Resolve relative manifest paths relative to the project working directory and absolute paths directly.
- Add focused synthetic tests for perfect horizontal translation, injected vertical shift failure, deterministic sampling, insufficient features, and JSON schema/exit decision. Tests may create small textured images under pytest temporary paths.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not modify or regenerate any dataset.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_validate_rectification.py -q`

## Return contract

Return a concise summary containing:

- status: completed, blocked, or failed
- files inspected
- files changed
- validation commands and results
- unresolved risks or assumptions
