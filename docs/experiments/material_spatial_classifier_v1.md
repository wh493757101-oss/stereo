# Fourth batch: material_spatial_classifier_v1

Frozen on 2026-10-13 after review. The user explicitly authorizes Codex to
implement, validate, run the 36 real fits, and independently verify this batch.
This task-specific authorization overrides the default Claude Code handoff
and manual-training workflow. No Git operations or deployment are authorized.

Objective: evaluate spatial q responses beyond frozen gray G256, common
C14/S16 statistics and spatial coverage. This is exploratory development on
the existing train+val pool, not a new blind test or material-causality claim.
The third-batch stopping decision and all its historical artifacts stand.

## Frozen sources

The complete reviewed spatial input cache is
`analysis/runs/material_spatial_input_v1/review_closeout/20261013T031808Z_advance_922b3f/real_build`.
Its binding SHA256 is
`aaa5ee1f4e8f41cdc905d51204e935be86f26a65a49c6af8d0f247b5cecab0f9`.
The new JSON-compatible YAML config pins the source binding, corrected
statistics, protocol, fold file and nine P0 checkpoint/scaler/oracle files.
Only consumed metadata, grid and permutation shards are read/hash-protected;
this batch does not claim a fresh audit of unconsumed native shards.

## Model and comparison

Seven channels: L interior/band, X interior/band, coverage interior/band/object.
Three Conv2d blocks (16/32/64, kernel3, stride2, padding1, no conv bias),
GroupNorm(4), SiLU. Coverage-object area-weighted pooling yields64 dimensions.
Concatenate standardized G256/C14/S16 (286 dims), then Linear350->64,
SiLU, Linear64->4. The head emits unrestricted logits; P0 is not a residual
base. Gray encoder extraction is not repeated or trained.

Four identical parameterized models: B_stats zeroes all response channels;
B_L uses L plus duplicate L; B_R uses L plus registered R; B_Q uses L plus q.
All retain common C/S and coverage. These are not pure-gray controls.
Original P0 is separately reproduced and compared.

Use the admitted q grid as stored; never reconstruct q from averaged L/R.
Q versus R measures representation, not independent information creation.
Coverage-matched stored q permutations are diagnostics on frozen B_Q heads,
not training augmentation. C/S, coverage, masks and missing flags do not move.
Partial block NaNs become standardized zero plus existing flags. All 105
fully missing instances bypass the new head and exactly use the original P0;
they never enter head loss but remain in validation and outer metrics.

## Training and evaluation

Frozen capture-group roles for three folds. Fit standardization on inner_train
only (G/C all rows, S valid cells). Train head only on available rows. Pick
epoch by inner_val Macro-F1, then accuracy, then earlier epoch. Outer_eval
never enters fit_head. OOF merge includes each sample once per model seed;
three seeds do not triple independent sample count.

Four variants x three folds x seeds2026/2027/2028 =36 fits. GPU0, float32,
batch32, workers0, max60 epochs, patience10, AdamW lr0.001 wd0.001,
CE without class weights, augmentation, scheduler, search or extra seeds.
Initialization and sample-order RNG are identical for the same fold/seed.
CUDA deterministic algorithms, TF32 off. All seeds are reported.

Preflight and four-arm two-epoch bounded smoke are engineering checks.
The formal fit starts one eight-hour deadline shared by admission, all fits,
diagnostics, independent verification and final publication. Each fit has a
ten-minute ceiling. A timeout yields incomplete evidence, never success or
an implicit reduction in the registered epoch budget. No automatic retries.

## Gates and stopping

All six gates required: (1) Q mean fold-F1 minus P0 >=1pp and minus L >=0.5pp;
(2) Q minus R positive mean-F1 and cumulative net corrections;
(3) at least two seeds net correct versus both P0/L;
(4) at least two mean folds nondeclining against both, worst drop <=1pp;
(5) every pooled class seed-mean recall drop <=1pp versus both;
(6) at least two groups net correct versus P0, with concentration reported.
Mean-fold and pooled metrics remain separate. Failure is a valid completed
experiment and stops this model extension. Passing only permits discussion
of real predicted-mask evaluation; no deployment/detection gain claim.

## Evidence and temporary storage

Fresh experiment output directories only. Save config/source identities,
scalers, complete epoch histories, selected weights, all per-instance logits,
fallbacks, permutation predictions and summaries. Independent verification
reloads all36 weights, reproduces normal/permuted outputs and selected val
metrics, checks scaler provenance, identities, missing fallback and selection,
and counts metrics/pairs/groups/gates without production summarizers.

All disposable data stay in a unique project `.test_artifacts` run. Set TEMP
and TMP per process and unique pytest basetemp children. Preserve diagnostics
outside scratch before cleanup; verify targets strictly within this run and
reject links. Reports, retained smoke weights, formal weights and evidence
are retained. Do not modify historical inputs, defaults, labels or Git state.

## Current status

COMPLETED on 2026-10-13. Preflight passed with P0 max error0 for all folds.
Four-arm two-epoch smoke passed. Focused JUnit:80 passed, zero
failures/errors/skips (45 new classifier cases plus35 existing input cases).
RED evidence retained: initial absent module collection refusal; subsequent
3 actual test failures for missing I/O/verifier implementation; one actual
metadata-guard failure, fixed and rerun GREEN. No Git checkpoints were made.

Formal run `analysis/runs/material_spatial_classifier_v1/fit_20261013T060008Z`:
36/36 fits COMPLETED, elapsed591.093s, one training attempt. Shared deadline
2026-10-13T14:23:33Z retained for subsequent verification and publication.

Independent verification:
`analysis/runs/material_spatial_classifier_v1/verify_20261013T060008Z_utcfix`:
PASSED, problem_count0,36 model replays,2446 unique instances,3 folds.
The first verifier launch failed before admission/replay because PowerShell
ConvertFrom-Json converted the UTC string to a DateTime whose argument
serialization was locale-formatted. Its log and empty output directory are
retained. The corrected launch used the same ISO UTC deadline and unchanged
training manifest, input/config/code. No models were retrained, no budget
reset and no output overwritten. Two launch attempts are disclosed; do not
describe verification as succeeding on its first launch.

Three-fold Macro-F1 averaged across all three seeds:

| Reference/variant | Mean Macro-F1 |
|---|---:|
| Frozen P0 |93.278349%|
| B_stats |93.810067%|
| B_L |93.528392%|
| B_R |93.550940%|
| B_Q |93.522799%|

B_Q minus P0:+0.244450pp; minus B_L:-0.005593pp;
minus B_R:-0.028140pp; minus B_stats:-0.287268pp.
Gate1-5 FAIL; Gate6 PASS; advance=false. The experiment is complete despite
failing benefit gates. Stop the current spatial-model extension; no real
predicted-mask experiment or deployment was started. No stable polarization
net benefit established. Do not extrapolate this to absent physical signal.

Retained review:
`analysis/runs/material_spatial_classifier_v1/engineering_20261013T060008Z/final_report_v2.json`.
Changed-code AST statement-start execution coverage89.928% from stdlib line
tracing across unit tests and real integration; neither pytest-cov nor branch
coverage. The environment lacks coverage/pytest-cov; nothing was installed.
Full repository tests NOT_RUN. Reports, checkpoints, logs and failed attempts
are retained outside disposable scratch. Cleanup is recorded separately.

Coverage correction: preliminary final_report.json combined line numbers from
earlier RED/preflight/smoke source revisions, so its93.669% and I/O100% claims
are withdrawn. The retained v2 report only uses final-revision extended units,
formal training and successful verification traces; code hashes match the
formal protected source manifest. Per-file coverage: core96.124%, I/O84.884%,
training91.257%, verifier89.100%. No code/model/input/metric was changed to
produce this correction; the old report remains as historical evidence.
