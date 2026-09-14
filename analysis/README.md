# Analysis reports

Reports are grouped by ownership instead of being stored directly in this
directory.

- `calibration/`: stereo calibration and rectification QA.
- `data/`: annotation revisions and derived-data audits.
- `project/`: machine-readable project inventory.
- `priors/`: generated polarization-prior analysis.
- `runs/<run-id>/`: evaluation, review, and benchmark reports for one training run.
- `runs/<run-id>/aliases/`: retained duplicate filenames from earlier workflows.

The report directory name must match the corresponding directory under
`runs/train/`. New evaluation commands should write directly into the matching
`analysis/runs/<run-id>/` directory.
