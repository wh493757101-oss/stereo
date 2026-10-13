# YOLO26 validation of the fourth-batch spatial method

Frozen on 2026-10-13 before any real fitting. The user asks Codex to validate
the fourth-batch method on YOLO26, continuing the explicit instruction that
Codex execute the work instead of handing it to Claude Code. This task-specific
authorization takes precedence over the default read-only project workflow.
No deployment, default configuration change, dependency installation or Git
mutation is part of this experiment.

## Question and sources

Does the admitted spatial method improve a YOLO26 grayscale classifier on
capture-group-isolated evaluation? Reuse the complete fourth-batch native L,
corrected nine-plane grid, C14/S16, availability and frozen three-fold roles.
Every consumed source is hash-protected before and after the experiment.
Old MiT G256 and P0 are not model inputs or fallback predictions here.

The existing `yolo26_formal_001/gray_fusion` checkpoint was trained on original
train and selected on original val. Both sets are in the current development
pool. It cannot be used for fold-isolated feature extraction or fallback.

## Frozen design

For each fold and seed 2026/2027/2028, independently initialize from the local
official ImageNet `yolo26n-cls.pt` (SHA256
`0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`).
Replace its linear output with the fixed four-class order. Train every YOLO26
parameter on inner_train only for 30 epochs, float32 CUDA0, batch32, AdamW
lr0.001/wd0.01, no augmentation/scheduler/class weighting. Select epoch by
inner_val Macro-F1, accuracy, then earlier epoch. Save gradient and actual
backbone/head/BN update evidence. This produces nine grayscale baselines Y0.

Gray preprocessing is native uint8 L, direct INTER_AREA resize to224, /255,
replicated three channels. It matches the established YOLO26 Gray Fusion path.
Freeze the selected model, eval/inference_mode; extract its pre-linear1280
features and raw logits. Verify exact state/BN preservation and reconstruction
of logits from features and the four-class linear head. Native global pooling
differs from the old MiT instance-mask pooling; this is an explicit migration.

For each fold/seed, all four spatial arms share that same frozen YOLO26 model.
Use the fourth-batch seven-channel CNN: Conv7->16->32->64, k3/s2/p1/no bias,
GroupNorm4/SiLU, coverage-object weighted pooling64. Standardized context is
G1280+C14+S16=1310; concat1374->Linear64->SiLU->Linear4, unrestricted logits.
B_stats zeroes all response planes; B_L uses L and duplicate L; B_R uses L
and registered R; B_Q uses L and admitted q. Shared S16 remains in all arms,
so Y0 is the pure-gray reference and B_L is not pure-gray.

Train spatial CNN/head only on available inner_train rows: max60 epochs,
patience10, batch32, float32 CUDA0, AdamW lr0.001/wd0.001, CE, no augmentation,
scheduler/class weighting/search. Standardization fits inner_train only;
epoch selection uses inner_val only. Fully missing105 instances bypass every
spatial arm and exactly return the same fold/seed Y0 logits; they stay in all
evaluation metrics. Partial missing blocks use standardized zero plus flags.
Stored coverage-matched q permutations are frozen-model diagnostics only.

## Evaluation and stopping

Nine gray plus36 spatial fits =45. Same fold/seed initialization and sample
order across spatial arms. Every seed is reported. Each sample occurs once
per model seed in pooled OOF predictions; seeds are not independent new data.
Original test is excluded. This remains existing-data exploratory validation,
not a new blind test, object-identity isolation or material-causality claim.

Inherit all six fourth-batch gates, replacing P0 with fold/seed Y0:
1. Q-Y0 mean-fold F1 >=1pp and Q-L >=0.5pp.
2. Q-R mean-fold F1 and cumulative net corrections both positive.
3. At least two seeds net correct against both Y0/L.
4. At least two mean folds nondeclining against both; worst drop <=1pp.
5. Every pooled class seed-mean recall drop <=1pp against both.
6. At least two capture groups net correct against Y0.
Also require the spatial-contribution condition: Q-stats
mean-fold F1 and cumulative net corrections both positive. Passing the six
gates without this condition cannot establish benefit from spatial q itself.
Report `overall_gain_passed` for the six gates and `spatial_contribution.passed`
separately; `advance` requires both. This stricter new-experiment decision was
reviewed and fixed before formal fitting; the old fourth-batch decision stands.
Failure is a completed experiment and stops extension, not an execution error.

One shared eight-hour deadline includes admission, real fits, diagnostics,
independent replay and publication. Gray single-fit ceiling20min; spatial10min.
No automatic retries, extra seeds, hyperparameter search or implicit budget
reductions. Preflight and bounded two-epoch smoke are engineering checks only.

## Verification and evidence

Retain source identities, complete histories, selected45 weights, gray feature
caches, scalers, all val/outer logits, fallback and permutation records.
Independent verification reloads9 YOLO26 and36 spatial models; recomputes
features, selected-val metrics, normal/permuted predictions, source identities,
scaler provenance, group isolation, missing fallback, pairs, classes and gates.
Its metrics and summarizer are independent of the training summarizer.

Tests follow RED/GREEN. Disposable artifacts use unique task subdirectories
of `.test_artifacts`, process-scoped TEMP/TMP, unique pytest basetemp children.
Retain diagnostics in the experiment evidence tree before safely removing only
verified current-run scratch directories; record cleanup on success or failure.
No Git checkpoint commits because project/user policy forbids Git mutation.

Status: implementation, tests, admission and bounded GPU smoke completed.
Formal fitting and independent verification status are recorded in retained
`analysis/runs/yolo26_spatial_v1/` run artifacts; this frozen protocol stays
unchanged after fitting begins.
