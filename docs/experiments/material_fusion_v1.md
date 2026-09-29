# Material fusion v1 (CMX MiT-B0) — experiment record

Independent CMX-based material-fusion classification experiment on the
V4 band-matched fusion dataset. This document records provenance,
licensing, the verified pretrained weights, the ported code, the fixed
training protocol, and the honest current state.

> **Status: implementation + bounded GPU smoke only. No formal training
> has been run and no polarization benefit has been proven.**

## 1. Upstream source (CMX)

- Repository: `https://github.com/huaaaliu/RGBX_Semantic_Segmentation`
- Pinned commit: `e251d860aebc2f583a6c4919877e6bebe7f1aff3`
- License: MIT, Copyright (c) 2022 Huayao Liu (upstream `LICENSE`,
  SHA256 `a3fb69f7d2d7ab44ce80bba7f8c3a61f3c8a2775a2baac4bef815a60c4d8ba5e`)
- Paper: CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with
  Transformers, `https://arxiv.org/abs/2203.04838`

Files fetched at the pinned commit and their SHA256:

| Upstream file | SHA256 |
|---|---|
| `LICENSE` | `a3fb69f7d2d7ab44ce80bba7f8c3a61f3c8a2775a2baac4bef815a60c4d8ba5e` |
| `README.md` | `a05bb6f4e6ad6ad88f45693697db55c7d3d36ee2ebea8a0ad251acedb9a4abce` |
| `requirements.txt` | `334f826de9b7553601904756ce9ad2f8c94ad4328e209fc82de3b7a5ceedaf50` |
| `models/encoders/dual_segformer.py` | `6d8994d0044929c97d318708223bf6fea48efce4575aa986c72e415ee017e755` |
| `models/net_utils.py` | `ada5e36e14d83c35d9230618c4eb84352a76ec2275a086a624b280e7638d8473` |

Only the encoder and fusion modules are ported; the upstream training
framework (engine, dataloaders, decoders, configs) is not copied.

### Ported files and modifications

`models/cmx/net_utils.py` (FRM/FFM) — verbatim except:

- `from timm.models.layers import trunc_normal_` replaced by
  `models.cmx.timm_compat` (same function, no timm dependency).

`models/cmx/dual_segformer.py` — module names, MiT-B0 parameter table and
every forward computation are identical to upstream, except:

- `timm` imports replaced by `models.cmx.timm_compat`
  (`DropPath`, `to_2tuple`, `trunc_normal_`; timm is Apache-2.0 and the
  three functions are reimplemented with identical semantics, pinned by
  tests in `tests/material_fusion/test_cmx_port.py`).
- `engine.logger` import removed (upstream training framework).
- `load_dualpath_model` replaced by strict loaders
  (`load_mit_b0_pretrained` / `load_mit_b0_checkpoint`) that verify full
  coverage and shapes instead of the upstream `strict=False` silent
  partial load. The upstream loader would leave tensors randomly
  initialized without any signal when the checkpoint does not match.
- `mit_b1` … `mit_b5` omitted (only MiT-B0 is used; the MiT-B0 entry is
  unchanged).
- Added: `MiTEncoder` (single-stream backbone with the same blocks and
  parameter names, used by variant E1), `convert_hf_state_dict`
  (nvidia/mit-b0 Hugging Face key naming → original naming, including
  the q/k/v → fused `kv` merge), `MIT_B0_SPEC` constants.
- The semantic-segmentation decoder is not ported; classification uses a
  multi-scale head instead (below).

`models/cmx/timm_compat.py` — new compatibility shim; no upstream code.

`models/cmx/LICENSE` — the complete upstream MIT license text, verbatim
(byte-identical, SHA256
`a3fb69f7d2d7ab44ce80bba7f8c3a61f3c8a2775a2baac4bef815a60c4d8ba5e`).
`models/cmx/README.md` records the source repository, the pinned commit
and the per-file modification list (upstream code is never presented as
local work and vice versa).

## 2. Pretrained weights (verified)

Official MiT-B0 ImageNet-1K backbone from the SegFormer authors
(NVIDIA). Two independent official channels were fetched and
cross-checked:

| Source | File | Size (bytes) | SHA256 |
|---|---|---|---|
| `https://huggingface.co/nvidia/mit-b0` (official NVIDIA release; fetched via `hf-mirror.com` because huggingface.co is unreachable from this network) | `pytorch_model.bin` | 14,380,029 | `4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc` (equals the HF API LFS `sha256` of the same repo) |
| `https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth` (OpenMMLab official conversion of the same official weights) | `mit_b0_*.pth` | 13,329,391 | `7e0fe6dd43f88b9f7f718ddeec691f00711490058ea17882d9027c822f2f48a8` (prefix `7e0fe6dd` equals the checksum embedded in the official filename) |

Cross-verification: after converting both files to the original key
naming, **all 192 backbone tensors are bit-identical** (192/192 exact
`torch.equal`, zero shape mismatches). The in-repo artifact
`weights/pretrained/mit_b0.pth` is the nvidia/mit-b0 file.

The CMX README additionally links a Google Drive folder of
"pretrained segformer" backbones; Google Drive is unreachable from this
network, so the official NVIDIA HF release (the same weights) was used
and verified against the OpenMMLab conversion instead.

Tensor structure of the nvidia/mit-b0 file: 192 encoder tensors
(`segformer.encoder.patch_embeddings.{0..3}` projections/layer-norms,
`segformer.encoder.block.{0..3}.{0..1}` attention/MLP blocks,
`segformer.encoder.layer_norm.{0..3}`) plus a 1000-class
`classifier.weight/bias` head that is dropped on load. After conversion
to the original naming (q/k/v merged into the fused `kv` projection) the
backbone is 176 tensors / 3,319,392 parameters, which loads into both
the single-stream and dual-stream encoders with full coverage (dual:
352 model tensors filled from 176 checkpoint tensors, both streams
identical, as upstream's `load_dualpath_model` intends).

Download failure is never a silent random initialization: a missing file
raises, and a provided SHA256 that does not match the file raises.

## 3. Model

```
appearance stream (3, 224, 224) ──► MiT-B0 (single) or CMX dual stream
                                    4 levels (32/64/160/256 channels)
                                    FRM + FFM fusion per level (dual only)
per-level 1x1 conv → 128 channels → global average pool → concat (512)
→ LayerNorm → Dropout(0.1) → Linear(4) → raw logits
```

Variants (input mapping from the V4 preprocessed tensors; `G` = the
scaled [0, 1] gray of the same sample, `V` = the V4 valid channel):

| Variant | Encoder | Auxiliary 3-channel input |
|---|---|---|
| E1 `gray` | single-stream MiT-B0 | — |
| E2 `gray_gray_valid` | CMX dual stream | `[G, G, V]` |
| E3 `gray_valid` | CMX dual stream | `[0, 0, V]` |
| E4 `gray_polar` | CMX dual stream | `[signed_q, abs_q, V]` |

Normalization is uniform and explicit: the appearance branch and every
polarization component keep the native V4 scale ([0, 1] gray via
`/255`, polar channels unchanged). No ImageNet mean/std normalization is
applied anywhere, and the quality vector is used only for availability
routing/diagnostics — it is never concatenated into classification
features.

Parameter counts (imgsz 224):

| Variant | Total | Backbone | FRM/FFM | Head |
|---|---|---|---|---|
| E1 `gray` | 3,388,516 | 3,319,392 | 0 | 69,124 |
| E2/E3/E4 | 10,864,844 | 6,638,784 | 4,156,936 | 69,124 |

## 4. Data

- Dataset: `datasets/underwater_cls_fusion_v4_band` (reused as-is, no V5,
  no npz changes), fingerprint
  `8db8998d7a4c280064626961e4d51c92c531878ea2e92f0d1480df46379bd334`
  (pinned in the config; training refuses a different audited state).
- Splits: train 2078 / val 368 / test 362, split by complete capture
  group. Training and model selection read train/val only; the test
  split is never touched by training.
- Preprocessing: identical to the existing V4 pipeline
  (`models.polar_fusion.FusionClsDataset`): direct resize of the stored
  crop to 224 (INTER_AREA for gray, INTER_LINEAR per polar channel).
  The adapter (`core/material_fusion_dataset.py`) is proven
  tensor-identical to the original V4 read by
  `tests/material_fusion/test_material_dataset.py`.
- Preprocessing version string: `v4_band_224_area_linear_v1`.
- No augmentation in the first round.

## 5. Training protocol (formal)

Fixed by `configs/experiments/material_fusion_v1.yaml`:

- imgsz 224, effective batch 16 (physical 8 × accumulation 2; identical
  for every variant and for smoke runs), epochs 30, seed explicit (2026).
- AdamW, weight_decay 0.01, no scheduler, no class weighting, no
  auxiliary loss.
- Parameter groups (every parameter exactly once):

  | Group | Parameters | warmup (epochs 1–2) | finetune (epochs 3–30) |
  |---|---|---|---|
  | `backbone` | pretrained MiT streams | frozen + eval (DropPath off) | trainable, lr 1e-5 |
  | `fusion` | new FRM/FFM layers (empty for E1) | trainable, lr 1e-4, BatchNorm statistics update | trainable, lr 1e-4 |
  | `head` | classification head | trainable, lr 1e-4 | trainable, lr 1e-4 |

  Freezing the backbone never detaches the graph: gradients still flow
  through it to the fusion layers.
- Gradient accumulation: group boundaries are stable (each group is
  exactly `accum` consecutive micro-batches except the last, which covers
  the remainder and is always stepped); optimizer steps =
  `ceil(batches / accum)`; every micro-batch contributes with weight
  `n_i / N_group` so a group's update equals the update on the same
  samples concatenated into one batch; the epoch loss is the
  sample-weighted mean. Each epoch records `gradients_finite` (the run
  refuses to continue on non-finite gradients).
- Model selection: val Macro-F1 primary, accuracy tiebreak; an exact tie
  keeps the earlier epoch. Fixed budget, no on-the-fly extension.
- Outputs: `runs/train/<run-id>/material_fusion/{best.pt,last.pt,
  metrics_history.jsonl,run_config.json}`; an existing run directory is
  always refused. `best.pt` carries the selected epoch's metrics,
  `last.pt` the actual final epoch's metrics/epoch/stage (they differ
  whenever the best epoch is not the last one).
- `run_config.json` stores the resolved run configuration under `config`
  (used by evaluation/verification), the parameter-group LRs, the
  pretrained SHA256 and separate `best`/`last` metric records.
- `--limit-batches > 0` marks a smoke run (bounded train/val batches,
  `smoke=true` and the cap recorded in checkpoint metadata with strict
  types); formal runs use 0.
- `--init-from` is weights-only continuation (clearly recorded as such);
  it is not a full optimizer/RNG resume.
- `--stage-snapshot-dir <dir>` is an observation hook of the production
  path: it writes `init.pt`, `post_warmup.pt`, `post_finetune.pt` and
  `stage_snapshots.json` state snapshots used by the verification harness
  (no extra training is performed).

## 6. Fallback routing (polar-invalid)

The raw classifier and the external fallback routing are separate:

- Pure classifier output = the material-fusion model on every sample.
- Routed output = samples with `valid_ratio == 0` (or an explicit
  `polar_invalid` flag) bypass the model and are classified by the
  independent historical gray Model B
  (`runs/train/yolo26_formal_001/gray_fusion/best.pt`, identity-verified
  at load time: canonical 4-class order, 4-wide head; a missing or
  mismatched file is refused, never substituted).
- Mixed batches keep the exact original sample order (index scatter).
- No confidence threshold is introduced anywhere; routing is a validity
  rule only.
- `scripts/eval_material_fusion.py` reports pure and routed metrics
  separately. `scripts/verify_material_fusion.py` verifies routing
  exactness with explicit invalid flags (bit-exact, no tolerance).
- The all-zero-polar forward of the CMX model is **not** claimed to
  equal the gray model; only the routing is claimed.

## 7. Usage

```powershell
# Formal run (not executed in this task)
D:/Python/CondaPkgs/stereo/python.exe scripts/train_material_fusion.py `
    --run-id <run-id> --variant gray_polar --device 0

# Bounded smoke (one per variant; example for E4)
D:/Python/CondaPkgs/stereo/python.exe scripts/train_material_fusion.py `
    --run-id cmx_smoke_e4_<date> --variant gray_polar --device 0 `
    --batch 8 --accum 2 --epochs 2 --warmup-epochs 1 --limit-batches 4
```

Evaluation resolves the *actual* run configuration from
`run_config.json` next to the checkpoint (or `--run-config`), so the four
variants each have a directly runnable command — no template editing:

```powershell
# E1 gray — pure classifier; the historical gray fallback is never loaded
D:/Python/CondaPkgs/stereo/python.exe scripts/eval_material_fusion.py `
    --checkpoint runs/train/<run-id>/material_fusion/last.pt --split val --device 0 `
    --output analysis/runs/cmx_<run-id>/eval_gray_val.json

# E2 gray_gray_valid / E3 gray_valid / E4 gray_polar — routed (fallback on)
D:/Python/CondaPkgs/stereo/python.exe scripts/eval_material_fusion.py `
    --checkpoint runs/train/<run-id>/material_fusion/last.pt --split val --device 0 `
    --output analysis/runs/cmx_<run-id>/eval_<variant>_val.json

# E2/E3/E4 — pure classifier only (no fallback loaded, no routing)
D:/Python/CondaPkgs/stereo/python.exe scripts/eval_material_fusion.py `
    --checkpoint runs/train/<run-id>/material_fusion/last.pt --split val --device 0 `
    --no-fallback `
    --output analysis/runs/cmx_<run-id>/eval_<variant>_pure_val.json
```

The evaluation report always separates the training-time markers
(`training.smoke` / `training.limit_batches` from the checkpoint) from the
evaluation-time truncation (`evaluation.limit_batches`, `evaluation.samples`,
`evaluation.truncated`); when routing is off, `routing.applied` is false
and `routed` is `null` — never a fabricated routed metric. An existing
report path is refused.

Verification (data admission, pretrained SHA256, stage evidence, gradient
finiteness, reload error, exact fallback routing):

```powershell
D:/Python/CondaPkgs/stereo/python.exe scripts/verify_material_fusion.py `
    --checkpoint runs/train/<run-id>/material_fusion/last.pt `
    --stage-snapshots analysis/runs/cmx_<run-id>/stage_snapshots `
    --device 0 --limit-batches 2 `
    --output analysis/runs/cmx_<run-id>/material_fusion_verify.json
```

The verification fails (exit code 1 with `failed_items` listed) when any
check fails, and it fails (exit code 2) when the actual data or the
pretrained SHA256 does not match the recorded run — a checkpoint without
stage evidence is reported as unprovable and cannot pass.

## 8. Bounded GPU smoke (2026-09-21)

One run per variant, RTX 4060 Laptop (8.6 GB), torch 2.12.1+cu130,
imgsz 224, physical batch 8 × accumulation 2 (effective 16), 2 epochs
(1 warmup + 1 finetune), at most 3 train/val batches per epoch:

```powershell
D:/Python/CondaPkgs/stereo/python.exe scripts/train_material_fusion.py `
    --run-id cmx_smoke_e1_20260921 --variant gray --device 0 `
    --epochs 2 --warmup-epochs 1 --limit-batches 3
# ... e2 gray_gray_valid, e3 gray_valid, e4 gray_polar (same flags)
```

| Variant | Run dir | Wall time | Peak VRAM | Exit |
|---|---|---|---|---|
| E1 gray | `runs/train/cmx_smoke_e1_20260921/material_fusion` | 1.7 s | 339 MB | 0 |
| E2 gray_gray_valid | `runs/train/cmx_smoke_e2_20260921/material_fusion` | 1.9 s | 882 MB | 0 |
| E3 gray_valid | `runs/train/cmx_smoke_e3_20260921/material_fusion` | 1.9 s | 882 MB | 0 |
| E4 gray_polar | `runs/train/cmx_smoke_e4_20260921/material_fusion` | 1.8 s | 882 MB | 0 |

(These times/VRAM are smoke-run bookkeeping only; they are not deployment
FPS estimates and nothing is extrapolated from them.)

Verification per run (`scripts/verify_material_fusion.py --device 0
--limit-batches 2`), all exit 0:

- Pretrained load: 176/176 backbone tensors consumed, 0 missing, both
  dual streams filled (352 model tensors), FRM/FFM reported as
  randomly initialized (156 tensors).
- Save/reload raw-logits comparison on `cuda:0`, float32, eval mode:
  `max_abs_diff = 0.0` for all four (atol 1e-5, rtol 0).
- Param updates: `last.pt` of every variant shows the encoder backbone
  changed during the finetune epoch (176/176 for E1, 352/352 for
  E2–E4, max |Δ| ≈ 2.2e-5 at encoder_lr 1e-5) and the head changed in
  every epoch; `best.pt` for E1/E3/E4 is the warmup-epoch checkpoint
  (that epoch won val Macro-F1) and correctly shows an unchanged encoder,
  while E2's `best.pt` (finetune epoch) shows updated encoder weights —
  the freeze→unfreeze transition is proven in both directions.
- Fallback routing (explicit invalid flags, bounded val batches): invalid
  rows bit-exactly equal to the historical gray model's raw logits, valid
  rows bit-exactly equal to the model's, sample order preserved.
- Fallback identity: `ClassificationModel architecture=yolo26n-cls
  classes=[metal_submarine, plastic_submarine, plastic_fish, real_fish]`.
- Bounded val evaluation (E4, 3 batches, 24 samples): pure and routed
  metrics reported separately; 0 invalid samples in that slice, so routed
  equals pure (routing changed nothing) — expected for a smoke slice.

## 9. First-round fix notes (2026-09-21/22)

The first verification round found and this round fixed: unstable
accumulation group boundaries (a full trailing group was never stepped:
4 batches/accum 2 gave 1 step instead of 2; 260 batches gave 129 instead
of 130), the frozen-encoder protocol wrongly freezing/under-training the
new FRM/FFM layers, the evaluation entry failing without a fallback
(`--no-fallback`) and refusing E1/E2/E3 because the default template
variant was treated as the run's actual configuration, evaluation not
admitting the actual dataset, parameter-update verification comparing
against a *different* random initialization (false positives), and
`last.pt` carrying the best-epoch metrics instead of the final epoch's.

Consequences for older artifacts (kept unchanged, never overwritten):

- `runs/train/cmx_smoke_*_20260921/` — first-round smoke runs. Their
  `last.pt` metadata records the best-epoch metrics (not the final
  epoch), and they carry no stage snapshots, so their parameter updates
  are unprovable under the current verification. Do not use them as
  evidence for anything.
- The fixed runs are `runs/train/cmx_fix_smoke_*` with stage snapshots
  under `analysis/runs/cmx_fix_smoke_*/stage_snapshots/`.

## 9. Verification evidence and guards (2026-09-22, guard round)

### Stage evidence format v2

`--stage-snapshot-dir` writes `init.pt`, `post_warmup.pt`,
`post_finetune.pt` and a `stage_snapshots.json` metadata file that binds
the evidence to the run it belongs to:

- identity: format/version, run_id, variant, seed, data fingerprint,
  imgsz, preprocessing version, upstream commit, pretrained SHA256,
  epochs, warmup_epochs, smoke, limit_batches, physical batch/accum/
  effective batch;
- per-snapshot role/epoch/stage and sha256 of each snapshot file;
- the digest, epoch and stage of the run's `last.pt` and `best.pt`
  (checkpoints never reference the evidence, so there is no circular
  digest dependency);
- strict types everywhere (no `int()`/`bool()` coercion).

Verification (`scripts/verify_material_fusion.py`) refuses to accept
evidence whose identity does not match the verified checkpoint
(cross-run, cross-variant), whose file digests do not match, whose
required roles are missing (a warmup snapshot is required exactly when
the run had warmup epochs), or whose checkpoint file is not one the
evidence records. The verified checkpoint must be bit-identical to the
`post_finetune` snapshot when it is the recorded `last`; an early `best`
must match its recorded epoch/stage and must not be a copy of the final
state. A loaded in-memory model must match the checkpoint exactly, so a
zeroed or modified model can no longer pass. State comparisons require
identical key sets, shapes, dtypes and finite values, and every protocol
group must be non-empty where the variant requires it — an empty group
can never produce a vacuous "frozen" verdict. Comparisons use
`torch.equal` for exactness (no float32 downcast).

The training history is checked against the run protocol: one row per
epoch, no missing/duplicate rows, the protocol stage per epoch, and a
strict boolean `gradients_finite: true` in every row.

A snapshot directory without v2 metadata is reported as **legacy**: the
read-only findings show what the files alone can demonstrate, and the
verification fails — legacy evidence can never be presented as strict
acceptance. The first fix-round artifacts
(`analysis/runs/cmx_fix_smoke_e*/stage_snapshots/`) are legacy in this
sense and must not be cited as strict evidence.

### Config <-> checkpoint consistency (shared by eval and verify)

Both entries resolve the run configuration (v2 `run_config.json` next to
the checkpoint > legacy run config > experiment template) and then run
the same strict contract `check_run_config_checkpoint_consistency`:
variant, imgsz, preprocessing version, upstream commit, pretrained SHA,
data fingerprint, seed, physical batch, accumulation, effective batch,
smoke and limit_batches are type-checked and compared (explicit mapping
`config.batch` <-> `metadata.physical_batch`), and the checkpoint's
epoch/stage must match the run's epoch budget and warmup boundary. A
doctored run config is refused before any report is written; `--variant`
may not contradict the actual run configuration; `--data` may relocate
the dataset only if the fingerprint still matches; `--pretrained` may
relocate the file only if the SHA256 still matches. Legacy sources derive
their values from the checkpoint, are labeled as such in the report, and
list the fields they cannot assert — strict verification requires the
run's own `run_config.json`.

### Snapshot output protection

The `--stage-snapshot-dir` target is canonicalized and refused when it
exists (file, empty or non-empty directory), when it is the project
root, `datasets/`, `weights/`, `runs/`, `runs/train` or another training
directory. Creation is exclusive (a target appearing after the pre-check
fails instead of being reused) and every snapshot/metadata write uses
exclusive-create semantics, so existing evidence is never overwritten.
A target inside the current run's own new directory is allowed; a refusal
happens before any training output is created.

### Final local guards (2026-09-22)

- Device-safe state binding: the in-memory model may live on CUDA while the
  checkpoint is loaded on CPU; comparisons use detached CPU copies per
  tensor, never mutating the model's device/dtype/mode, and still require
  identical key sets, shapes, dtypes, finite values and bit-exact equality
  (no float32 unification, no `allclose`, no relaxed tolerance).
- `warmup_epochs=0` is a legal protocol: the evidence has no `post_warmup`
  snapshot, the warmup checks are explicitly reported as not applicable
  (never fabricated as passed), and the finetune progression is compared
  `init -> post_finetune` with the same backbone/fusion/head update
  requirements. Missing warmup evidence under a warmup>0 protocol fails
  explicitly; a missing snapshot is never read as "no warmup".
- Snapshot targets: the project root and **every descendant** of the
  `datasets`, `weights` and `runs` trees are refused (containment decided
  by resolved path relationships, so `..` aliases and name-similar
  directories cannot bypass or falsely trip it), except a subdirectory of
  the current run's own new directory. Legal targets: new directories under
  `analysis/runs` and inside the run directory.

## 10. Honest state

- Implementation, tests, and bounded GPU smoke only.
- **Not formally trained; no polarization benefit is proven.** Smoke
  metrics on a handful of val samples of an undertrained model are
  bookkeeping, not evidence.
- The gray default of the runtime pipeline is unchanged.


## 11. Fair-comparison orchestrator (2026-09-22)

`scripts/run_material_experiments.py` owns the 4 variants x 3 seeds
(2026/2027/2028) = 12-task formal matrix. Modes are strictly separated:

- `plan` — writes `analysis/runs/material_experiments/plan.json`: the 12
  tasks, exact commands, output dirs and the resource basis (train/val
  sizes, optimizer steps per run). Read-only.
- `preflight` — read-only admission: config vs the fixed protocol
  constants (224 / effective batch 16 / 30 epochs / warmup 2 / AdamW /
  lr 1e-4 + encoder 1e-5 / workers 0 / smoke false), the requested device
  (CUDA availability and the device index are checked against the runtime,
  not merely parsed), dataset fingerprint, pretrained SHA256, fallback
  weights presence, output-dir state. Refuses (exit 1) with an explicit
  difference list instead of self-selecting a protocol.
- `run` — starts only the explicitly selected tasks, serially, stopping on
  the first failure; every attempt (command, status, exit code, start/end
  time) is *appended* to
  `analysis/runs/material_experiments/run_manifest.json` as per-task attempt
  history, so a retry or a later `--skip-completed` pass never erases an
  earlier failure's record. Selection is by `--tasks` with
  `<variant>_s<seed>` ids (there is no seed shortcut and the id separator is
  an underscore, not a colon):

  ```powershell
  & $py scripts/run_material_experiments.py run --device 0 --tasks gray_s2026,gray_gray_valid_s2026,gray_valid_s2026,gray_polar_s2026
  & $py scripts/run_material_experiments.py run --device 0 --tasks gray_s2027,gray_gray_valid_s2027,gray_valid_s2027,gray_polar_s2027
  & $py scripts/run_material_experiments.py run --device 0 --tasks gray_s2028,gray_gray_valid_s2028,gray_valid_s2028,gray_polar_s2028
  ```

  The requested device is validated before the first spawn (the same check
  `preflight` reports), so an unusable device fails once instead of
  producing twelve failed attempts. Existing directories are refused;
  `--skip-completed` skips only runs that pass full completion validation —
  never a directory-name check. A run qualifies only if its resolved config
  matches the fixed protocol field by field, pins a dataset fingerprint and a
  pretrained SHA, its metrics history is contiguous from epoch 1 to the
  configured epoch count with the protocol stage at every epoch, both
  checkpoints agree with the config (variant, image size, preprocessing
  version, upstream commit, seed, batch geometry, pretrained SHA, dataset
  fingerprint), the recorded best epoch *is* the history's selection, and the
  best/last records match the checkpoint metadata. The orchestrator never
  deletes or overwrites an existing run directory, so a task that failed
  part-way must have its partial directory removed explicitly by the operator
  before it can be retried.
- `collect` — read-only aggregation plus the authorized train/val
  inference report into `analysis/runs/material_experiments/collect_*/`:
  per-task best epoch/accuracy/Macro-F1, per-class precision/recall/F1,
  confusion matrix, supports, valid/invalid subsets, pure vs routed
  reported separately, paired flip counts for equal-seed comparisons,
  seed mean/std per variant, and a capture-group cluster bootstrap of the
  Macro-F1 delta (gray_polar - gray). Each task's evaluation data is admitted
  against *that checkpoint's own recorded dataset fingerprint* — `--data`
  relocates the data, it never relaxes the pinned identity — so a run trained
  on a different dataset state is excluded rather than silently compared.
  Smoke artifacts are excluded, the test split is never read, and no
  optimizer is ever created. The val split has only 4 capture groups — the
  report states this uncertainty explicitly.

Test isolation: every disposable test artifact — synthetic datasets,
synthetic pretrained weights, bounded CPU training run directories, stage
snapshots and reports — is written under a unique run directory inside the
project-local `.test_artifacts/` tree, never into the real `runs/train`.
Both output roots are injectable rather than hard-coded: the orchestrator
resolves its run root from the module-level `RUNS_ROOT`, and
`scripts/train_material_fusion.py` does the same for `runs/train/<run-id>/`.
The tests redirect both through those hooks (the production CLI gains no
relaxation flag and its default output locations are unchanged). Cleanup is
decided by resolved path ownership, not by name: a target must resolve
strictly inside the session's own scratch root, no component of its path and
nothing inside the tree may be a symlink, junction or other reparse point
(which `Path.is_symlink` misses on Windows), and the owned root itself is
never a target. A removal failure is reported with the exact path and
exception instead of being ignored, and a run directory that already exists
is refused rather than cleared. A session-level guard records the real
`runs/train` inventory before and after the suite and fails if any test
created or removed real training output.

Auxiliary diagnostics (explicit, separate, never launched by the
orchestrator, never fitted by `collect`):

- `scripts/fit_bias_baseline.py` — frozen gray Model B + (a) four fixed
  class biases, (b) a quality-vector-only linear correction. Fitted on
  train, evaluated on val, bounded step budget, test refused.
- `scripts/ablate_material_shuffle.py` — inference-only shuffled-
  polarization ablation on a fixed checkpoint: reproducible fixed-point-
  free donor bijection over the full split index (batch-size independent,
  per-sample donors recorded), seeds 2026-2030, `full` (content + valid
  swapped, together with the donor's quality vector that validity
  statistics and fallback routing read) vs `content_only` (own valid and own
  quality kept) modes, raw-logit deltas and
  full prediction transition matrices per seed, optional fallback-routing
  effect. Test split refused.

Generated orchestrator/ablation/baseline artifacts stay local
(`analysis/runs/material_experiments/`, `material_shuffle_*`,
`material_bias_*` are git-ignored).

## 12. Honest state

- Implementation, tests, and bounded GPU smoke only.
- **Not formally trained; no polarization benefit is proven.** Smoke
  metrics on a handful of val samples of an undertrained model are
  bookkeeping, not evidence.
- The gray default of the runtime pipeline is unchanged.
