# External worker task

## Objective

Make runtime Model B input semantics explicit and select the gray-only checkpoint after the formal ablation gate failed to prove a polarization gain.

## Allowed scope

- Inspect: `gui/inference_engine.py`, `gui/inference_panel.py`, `configs/default.yaml`, `tests/test_inference_engine.py`, `tests/test_inference_panel.py`, `core/polar_compute.py`, `analysis/model_b_ablation_test.json`.
- Modify only: `gui/inference_engine.py`, `gui/inference_panel.py`, `configs/default.yaml`, `tests/test_inference_engine.py`, `tests/test_inference_panel.py`.

## Requirements

- Add a validated Model-B input mode with exactly `gray` and `polar` values. Keep constructor backward compatibility by defaulting injected/legacy callers to `polar`.
- In `polar` mode classify `[gray, polar, gray]`; in `gray` mode classify `[gray, gray, gray]`. Preserve crop geometry and all depth/polar-map calculations.
- Read `model_b.input_mode` from YAML in `DualStageInferenceEngine.from_config` and forward it to the engine.
- Change the default config Model B path to `runs/train/model_b-gray/weights/best.pt` and `input_mode: gray`, with a concise comment citing the local formal gate report `analysis/model_b_ablation_test.json` (no web citation). Do not alter the separate evaluation gray/polar paths.
- Expose the mode in the GUI Model B controls using a combo box or other appropriate option control. Make the section title mode-neutral. Ensure `_read_config` passes both selected path and selected mode when creating the engine.
- Tests must verify exact channel content for both modes, invalid-mode rejection, YAML wiring, default panel path/mode, and GUI forwarding. Existing polar-mode tests must remain valid.
- Do not change model weights, generated datasets, or the ablation report.

## Prohibited actions

- Do not modify files outside the allowed scope.
- Do not install or update dependencies.
- Do not commit, push, reset, clean, checkout, or switch Git state.
- Do not access external networks or credentials.
- Do not weaken, skip, or delete existing tests.

## Validation

- Run exactly: `D:/Python/CondaPkgs/stereo/python.exe -m pytest tests/test_inference_engine.py tests/test_inference_panel.py -q`

## Return contract

Return status, files inspected/changed, validation result, and unresolved risks or assumptions.
