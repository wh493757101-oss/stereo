# External worker task

## Objective

Correct the rectification QA matcher so cross-polarization false correspondences do not dominate p95, while genuine coherent vertical misalignment still fails the configured thresholds.

## Allowed scope

- Inspect and modify only `scripts/validate_rectification.py` and `tests/test_validate_rectification.py`.
- Read `analysis/rectification_qa_v2.json` as evidence; do not modify it.

## Requirements

- Keep the existing CLI and JSON schema backward compatible, adding fields only when useful.
- Use symmetric/mutual descriptor matching in addition to a ratio test.
- Apply a deterministic robust outlier filter based on vertical displacement consistency only (for example median/MAD with a sensible absolute floor). Do not subtract the estimated vertical offset before reporting: a coherent 4 px vertical shift must still report about 4 px and fail.
- Keep horizontal disparity unrestricted except for finite/image-plausible coordinates; targets may occur at different depths.
- Report raw match count and retained inlier count. Apply `min_matches` to retained inliers.
- Add tests with deliberate false correspondences/outlier displacement and ensure they do not inflate p95, plus retain tests proving a genuine vertical shift fails.
- Do not simply loosen the 1 px median or 3 px p95 thresholds.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install dependencies, access networks, modify data, or perform Git operations.
- Do not weaken or remove existing tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_validate_rectification.py -q`

## Return contract

Return status, files changed, exact validation result, and remaining assumptions.
