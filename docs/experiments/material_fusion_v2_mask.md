# Material fusion v2 — instance-mask pooling (A1)

Third-round A1 record. It builds the instrument needed to ask a trustworthy
question about polarization; it does **not** produce a new model-quality
conclusion. No formal training was run in this batch.

Scope of this batch, in one line: keep everything from round 2 and change
exactly one training-side factor — the classification head's global average
pooling becomes instance-mask weighted pooling.

## 1. Why the previous round could not answer the question

Round 2 trained the four variants (E1 gray / E2 gray+gray+valid / E3
gray+valid / E4 gray+polar) for three seeds. Mean val Macro-F1 was E1
99.48 %, E2 99.15 %, E3 98.83 %, E4 99.67 % over 368 val samples, 4 capture
groups and 18 real fish. Those limitations stand and are not restated as a
result here. The diagnostic problems that motivated A1:

- The content-only shuffle replaced polarization content but kept the
  sample's own valid channel, and the full shuffle moved both the content
  *and* the valid structure. Neither separates "the measured values" from
  "where a measurement exists".
- The two plastic fish that E4 fixed for seeds 2027/2028 stayed correct under
  all five content-only shuffles, so that fix cannot be attributed to the
  correctly matched differential content.
- The V4 ``valid`` channel was already restricted to the target interior by
  the V4 generation, so it is not an independent measurement-validity map for
  the whole crop; it also carries shape information.
- Aggregate transition counts cannot stand in for per-sample predictions.

## 2. Instance-mask provenance (audited read-only)

The mask is derived from the source annotations, not guessed from the data:

```
sample_name
  -> split / group_name / source_frame        V4 manifest
  -> source image + YOLO label file           source pair_manifest.csv
  -> object_index                             label line order (0-based)
  -> polygon                                  normalised coords * (width, height)
  -> full-image mask                          cv2.fillPoly on int32 points
  -> the identical V4 crop window             floor(min)-pad .. ceil(max)+pad, clipped
  -> cropped instance mask
```

The reconstruction reuses the *same functions* the V4 generation used
(``build_object_mask``, ``make_crop_window``, ``read_yolo_polygons``,
``load_gray``), and every sample is checked against V4 before anything is
written:

| check | result |
|---|---|
| V4 samples resolved to a polygon | 2808 / 2808 |
| unresolved frames | 0 |
| per-split counts equal to V4 | train 2078 / val 368 / test 362 |
| reconstructed crop shape == V4 crop shape | 2808 / 2808 |
| reconstructed gray crop byte-equal to V4 ``gray`` | 2808 / 2808 |
| label class id == manifest class id | 2808 / 2808 |
| mask non-empty inside the crop | 2808 / 2808 |
| V4 ``valid`` pixels outside the polygon | 0 |

Read-only report: ``analysis/runs/material_mask_v5/provenance_check.json``.

Nothing here re-runs SGBM, re-estimates q or the gain, re-splits the data or
changes a class id, an object index or a sample name. The test split was read
for provenance and integrity only; no model was evaluated on it.

## 3. V5 dataset

``datasets/underwater_cls_fusion_v5_instance_mask`` (new directory; the
generator refuses to overwrite an existing one and has no ``--clean``).

- 2808 samples, same splits, same class ids, same object indices, same order.
- ``gray``, ``signed_q``, ``abs_q``, ``valid``, ``quality``, ``class_id`` are
  copied from V4 element for element, with the same dtype and shape. The
  audit verifies this per sample.
- One new array: ``instance_mask`` uint8 (h, w), 1 inside the annotated
  target, same shape as the crop.
- New manifest columns: ``left_path``, ``label_path``, ``image_width``,
  ``image_height``, ``crop_x1..crop_y2``, ``mask_pixels``, ``crop_area``,
  ``mask_area_ratio``.

Three regions stay distinct, because conflating them is the error this batch
exists to avoid:

| region | meaning |
|---|---|
| ``instance_mask`` | annotated target region (added by this batch) |
| ``polar_valid`` | the V4 measurement-valid region, already target-interior in V4 |
| ``instance_mask & polar_valid`` | target-interior measurable region |

**V5 does not restore measurement validity outside the target.** Recomputing
an independent whole-crop validity map is a later step and is not claimed
here.

Audit binding (``dataset_audit.json``):

| field | value |
|---|---|
| V5 fingerprint | `bb9e3661fb1e71e94056893f6dfc50b9e3dee1c604f430ac0727d75362d32982` |
| V4 parent fingerprint | `8db8998d7a4c280064626961e4d51c92c531878ea2e92f0d1480df46379bd334` |
| mask content digest | `46d186cca11805d767849a5ee9ac5fbdf741c310c3e8440fe2fb59d0eacc956f` |
| label source digest | `46579c1ca92834cfc27b91c570df797513f9a2b602d896c388612b33e59d63c8` |
| sample mapping digest | `fd2de70f3b8da08a0b92b1dce813bd1b2b8f29d2af9b32da3ca31107dd9aa7c6` |
| mask schema version | `instance_mask_polygon_crop_v1` |
| mask resize version | `area_crop_to_imgsz_then_area_to_levels_v1` |
| crop pad | 10 (must equal the V4 generation value) |

Mask area ratio over the crop: mean 0.313, min 0.128, max 0.746.
``polar_valid`` coverage inside the mask: mean 0.750, min 0.0, max 1.0.

Quality components after adding the mask — stated honestly:

- **exactly recomputable now**: ``valid_ratio`` (valid pixels / instance-mask
  pixels) and ``mean_abs_q``. Both are recomputed and compared per sample.
- **bound-checked only**: ``in_bounds_ratio`` and
  ``brightness_valid_ratio``. Their per-pixel masks are not stored, so only
  the containment relations (``valid_ratio <= in_bounds_ratio``,
  ``valid_ratio <= brightness_valid_ratio``) are verified. This audit does
  not claim they are exactly recomputable.

Admission (``core.material_mask_dataset.admit_mask_dataset``) re-verifies the
manifest/summary/per-file digests, re-decodes every sample, re-checks the
mask invariants, recomputes the mask content digest, pins the class order and
compares both the V5 fingerprint and the V4 parent fingerprint. An unaudited
or tampered V5 never reaches a model.

## 4. The one training-side change

``MultiScaleClassificationHead`` gains a ``pooling_mode``:

```
pooled[c] = sum(F[c,h,w] * M[h,w]) / sum(M[h,w])
```

- ``M`` comes only from ``instance_mask`` — never from ``polar_valid`` and
  never from q.
- The same ``M`` and the same rule are used for all four variants; no variant
  receives the mask alone.
- ``M`` is area-resampled: the annotated mask is resized into the model input
  frame with the same ``INTER_AREA`` call the V4 channels get, then resampled
  to each encoder level with ``mode="area"`` (an area average over the source
  region mapped to each output cell). No thresholding is used, so a target
  smaller than one output cell still keeps a positive weight; a 3x3 target at
  ``imgsz=224`` has a positive weight even at the 7x7 level.
- A zero weight sum at any level is refused with an explicit error; the head
  never silently falls back to GAP.
- The projection, concat, LayerNorm, Dropout and Linear layers are unchanged,
  and no learnable mask parameter is added.
- An all-ones mask reproduces GAP exactly (measured max |difference| = 0.0 in
  float32).
- Positions whose *level* weight is zero do not affect the pooling output
  (verified by perturbing exactly those positions).

``cv2``'s area resampling is a float32 average and can overshoot 1.0 by one
ulp; the coverage fraction is clipped back into [0, 1] and this is recorded.

### What this does and does not do

The encoder's convolutions and attention already mix spatial information, so
instance pooling cannot guarantee background isolation — it only removes the
*pooling* operator's own background contribution. It also does not remove the
shape information carried by the invalid region of q. Area resampling spreads
a boundary mask value across an output cell, so a pixel just outside the
target can still influence the result through a partially covered cell.

### Versioning and compatibility

- New protocol version ``material_fusion_v2_mask``
  (``configs/experiments/material_fusion_v2_mask.yaml``).
- Every checkpoint records ``pooling_mode``, ``mask_schema``,
  ``mask_resize_version``, the V5 ``data_fingerprint`` and the V4
  ``parent_data_fingerprint``. A ``material_fusion_v2_mask`` checkpoint that
  omits ``pooling_mode`` is refused rather than guessed.
- A GAP checkpoint and a mask-pooling model are never interchangeable: loading
  one as the other is refused by ``pooling_mode`` even though the parameter
  shapes are identical, and a GAP head refuses a mask tensor instead of
  ignoring it.
- v1 checkpoints keep their original loading and inference behaviour: a
  missing ``pooling_mode`` means GAP, which is what the v1 format is, and the
  v1 call convention (``model(rgb, aux)`` / ``model(rgb)``) is preserved.
- A formal v2 run initialises from the same official MiT-B0 pretrained
  weights as v1; it never continues from a v1 ``best`` checkpoint, which
  would import a different training history into the comparison.

Protocol constants are unchanged: imgsz 224, batch 8, accum 2,
effective_batch 16, 30 epochs, warmup 2, AdamW, weight_decay 0.01, lr 1e-4,
encoder_lr 1e-5, seeds 2026/2027/2028, workers 0, Macro-F1 primary with an
accuracy tiebreak and the earlier epoch kept on an exact tie.

## 5. Controlled diagnostics

``scripts/diagnose_material_mask.py``. Perturbations are applied at native npz
resolution and the existing resize is reused; the training preprocessing is
never changed for diagnostic convenience.

- **A** original input (baseline).
- **B** self-content permutation: ``signed_q`` values inside
  ``instance_mask & polar_valid`` are permuted among themselves. The gray,
  the instance mask, the polar-valid support and every invalid region are
  kept, and the multiset of valid values is unchanged. This tests spatial
  arrangement sensitivity only; it does **not** remove the sample-level
  polarization distribution.
- **C** matched-donor content transfer: the recipient keeps its gray, its
  instance mask, its polar-valid support and its invalid regions, and receives
  the *values* of a matched donor's target-interior valid region. Donor values
  are placed through a seeded cyclic permutation (replacement is used exactly
  when the donor has fewer valid pixels than the recipient) and
  ``signed_q``/``abs_q`` move together. This is a controlled numeric
  distribution intervention, **not** a physically real polarization image, and
  it does not by itself prove causation.

Quality handling is self-consistent: the support is unchanged, so
``valid_ratio``/``in_bounds_ratio``/``brightness_valid_ratio`` keep their
values and only ``mean_abs_q`` is recomputed. Both ``original_quality`` and
``intervention_quality`` are recorded per sample. Quality is not fed into the
classifier features. All work happens at native resolution, so no claim is
made that ``abs(resize(q)) == resize(abs(q))``.

### Frozen donor protocol

Written before any model output was inspected (``donor_protocol.json``):

1. candidates come only from the diagnosed split; no train/val/test borrowing;
2. no self-matching;
3. matching features: ``valid_coverage``, ``mask_area_ratio``, valid centroid
   (y, x), valid bbox height/width ratio, mean gray inside the target — all
   computable from the data alone;
4. capture conditions (depth, turbidity, illumination) are **not recorded** in
   this dataset and are reported as ``unknown``, never inferred from a file
   name;
5. the true class, the model output, the logits and the correctness are never
   used to pick a donor;
6. the q mean and the q histogram are deliberately not matched, so the signal
   under test is not pre-removed;
7. weights are fixed (1.0 except 0.5 for the two bbox ratios); the ceiling is
   the 95th percentile of nearest-neighbour distances *within the train
   split*, so the threshold comes from training metadata, never from val model
   outputs;
8. matching is a deterministic greedy one-to-one assignment in sample-name
   order with a name-based tie-break: independent of batch size and iteration
   order, reproducible across processes (the intervention RNG is seeded from a
   sha256 of the protocol seed and the sample names, never from Python's
   randomised ``hash``);
9. a recipient with no eligible donor is ``unmatched`` with a reason; an
   arbitrary donor is never substituted;
10. coverage, unmatched reasons, per-class and per-group coverage, distances
    and donor reuse are all reported;
11. a partial matching is **not** called a bijection; the report states
    whether the matching was complete;
12. every intervention metric is paired against the baseline on the **same
    matched recipient subset** — a matched-subset metric is never subtracted
    from a full-set baseline.

Matching a few statistics does not remove shape, brightness or acquisition
confounding, and the perturbed inputs are partly out-of-distribution. Both
caveats are recorded in the report.

## 6. Per-sample evidence

``per_sample_evidence.jsonl`` (one record per sample per condition per seed)
carries: ``sample_name``, ``source_frame``, ``object_index``, ``group_name``,
``split``, ``label``/``label_name``; ``condition``, ``seed``, ``donor_id``,
``matched``, ``unmatched_reason``, ``match_distance``; the matching features;
``original_quality`` and ``intervention_quality``; all four class logits,
prediction, correctness, the **routed** prediction and correctness separately
from the raw ones, and whether the fallback was triggered. Every aggregate in
``report.json`` is recomputed from these records, and every sample appears
exactly once per condition (the tail batch is not dropped).

## 7. What A1 found so far (instrument validation, not a model conclusion)

Frozen v1 checkpoints were re-run read-only on val (v1 GAP head, V4 input,
original admission) to recover complete per-sample predictions, because the
round-2 aggregate reports do not contain them.

- The two plastic fish fixed by E4 are
  ``Single_Plastic_fish_20_NTU_010_obj001`` and
  ``Single_Plastic_fish_20_NTU_011_obj001`` (group ``Single/Plastic fish/20
  NTU``). Both were misclassified as ``plastic_submarine`` by E1 and correctly
  classified as ``plastic_fish`` by E4 for seeds 2027 and 2028 (and 011 also
  for 2026). Full evidence:
  ``analysis/runs/material_mask_v5/legacy_v1/``.
- On the frozen E4 checkpoint for seed 2027 (368 val samples, 314 matched,
  54 unmatched), both interventions are non-trivial — **every** perturbed
  sample changed its logits (364/364 for B, 314/314 for C) — yet:
  - B broke 2–3 of 314 samples per seed and fixed none;
  - C broke 3 of 314 samples per seed and fixed none;
  - both plastic fish stayed correct under every seed of both interventions.
- **How far that evidence goes.** The correct statement is: *these two samples
  keep their correct classification under the spatial rearrangement and the
  fixed matched-donor substitution used here.* It is **not** established that
  their improvement is independent of the polarization values:
  - the donor that C assigns to each of them is **the other plastic fish
    sample** (010 ↔ 011, same capture group, adjacent frame numbers), so the
    "donor" content is another target of the same class from the same
    acquisition;
  - their original ``mean_abs_q`` values are close (0.04257 and 0.04218), so
    the transfer replaces a polarization value distribution with a similar
    one;
  - consequently a polarization-value feature **shared by both samples** could
    still be what the model uses, and the C intervention would not remove it.
  So the result cannot exclude that a shared polarization-value feature is
  being used, and it does not identify whether the improvement comes from the
  polarization values, the support region, the background, or something else.
- A changed logit vector with an unchanged ``argmax`` is **not** evidence that
  the model ignores the polarization values; it only means the decision
  boundary was not crossed for those samples.
- Nothing here shows "only shape is useful" or "polarization material
  information is ineffective". Both would be over-claims from this evidence.
- Reports: ``analysis/runs/material_mask_v5/controlled_e4_s2027/`` (the frozen
  diagnostic reports are unchanged; the A1 acceptance round only re-read them).

These are diagnostics on frozen v1 weights. The new V5 mask-pooling models
have smoke weights only, and their metrics are **not** used for any model-gain
claim; a real comparison needs A2.

A stronger intervention (for example a donor restricted to a *different*
capture group with a matched, dissimilar value distribution, or a
polarization-value permutation that is independent across the two samples)
would be needed to separate those explanations. It is listed here as a
**proposal for a separately authorised experiment** and was not run.

## 8. Bounded smoke

``scripts/verify_material_mask.py`` runs the real training entry on the real
V5 dataset with a hard ceiling (2 epochs, 1 warmup epoch, 3 batches per phase,
batch 8, accum 2) and refuses anything larger. It verifies: the checkpoint
records the mask protocol and the V5 fingerprint; a GAP load of a mask
checkpoint is refused; the loader really carries masks; save/reload reproduces
identical logits; the mask is not inert (mask pooling differs from GAP on the
same weights); the pooling helper really ran during training; warmup froze the
backbone and trained the new layers, finetune unfroze the backbone; the
independent gray fallback is byte-identical before and after; and the smoke
run is refused by the formal completion admission so it can never enter a
formal comparison.

Command (per variant; the run id is only a name, so the variant must be
requested explicitly):

```powershell
& $py scripts/verify_material_mask.py `
  --config configs/experiments/material_fusion_v2_mask.yaml `
  --data datasets/underwater_cls_fusion_v5_instance_mask `
  --variant gray_polar `
  --device 0 --run-id "mask_a1_smoke_gray_polar_<tag>" --smoke `
  --epochs 2 --warmup-epochs 1 --limit-batches 3 --batch 8 --accum-steps 2
```

Result (all four variants, tag `a1_20260924T162202Z`, every check `ok`):

| variant | parameter groups (backbone/fusion/head) | warmup froze/updated | mask-pooled vs GAP | reload Δlogit |
|---|---|---|---|---|
| gray | 176 / 0 / 12 | 176 frozen, 12 updated | 0.608 | 0.0 |
| gray_gray_valid | 352 / 132 / 12 | 352 frozen, 168 updated | 0.599 | 0.0 |
| gray_valid | 352 / 132 / 12 | 352 frozen, 168 updated | 0.891 | 0.0 |
| gray_polar | 352 / 132 / 12 | 352 frozen, 168 updated | 0.775 | 0.0 |

The single-stream variant really has no fusion group, the dual variants do,
and each variant's logits differ — so the four runs are four models, not one
model under four names. Reports:
``analysis/runs/material_mask_v5/verify_mask_a1_smoke_<variant>_<tag>.json``.

**A bug this validation caught.** The first smoke attempt omitted
``--variant``, so all four runs trained the config's ``gray_polar`` while
carrying run ids that named the other variants. The four misleading
directories were removed and re-run. The verifier now refuses to leave this
silent: it checks that the checkpoint's variant equals the requested variant
and that the parameter groups match the variant (a ``gray`` model must have an
empty fusion group).

## 9. Honest state

- No formal training was run in A1. The 12-run v1 comparison is unchanged and
  untouched.
- The smoke weights are plumbing evidence only; they say nothing about whether
  instance-mask pooling or polarization helps.
- The diagnostics above are instrument validation on frozen v1 weights. They
  do not establish a polarization benefit, and the two plastic fish result is
  a negative finding about the mechanism.
- Donor matching is partial (314/368 here) and matches a few statistics only;
  shape, brightness and acquisition confounding remain.
- The formal orchestrator still pins the v1 protocol constants, so it refuses
  v2 runs. Extending it is A2 work and was deliberately not done here.

## 10. A1 acceptance round (2026-09-26)

Five defects found in review were fixed; no data was regenerated and no
historical artifact was touched. The V5 dataset, the frozen v1 checkpoints and
the frozen diagnostic reports are byte-identical to the previous round.

1. **The eval entry admitted every dataset through the V4 gate.**
   `scripts/eval_material_fusion.py` called `admit_material_dataset`
   unconditionally, so a legal v2 checkpoint was refused because the V5
   dataset's *V4-style* fingerprint (`139ae105…`) is not its real V5
   fingerprint (`bb9e3661…`). Admission now follows the strictly validated
   pooling protocol: `instance_mask` runs go through the V5 gate (digest
   re-verification, mask decode, mask content, annotation source, sample
   mapping, V4 parent, class order), `gap` runs keep the original V4 gate.
   Nothing clears or downgrades the expected fingerprint. The template used
   when a checkpoint has no adjacent `run_config.json` is now selected from
   the checkpoint's recorded protocol version, and the derived config inherits
   `pooling_mode`, `mask_schema`, `mask_resize_version` and the V4 parent
   fingerprint from the checkpoint instead of the v1 template defaults.

2. **Cross-protocol weight continuation and metadata mixing were open.**
   `--init-from` copied a v1 GAP state dict into a v2 mask-pooling model
   (identical shapes, `All keys matched successfully`), and
   `validate_checkpoint_metadata` accepted `version=material_fusion_v2_mask`
   with `pooling_mode=gap`. Now: the v2 protocol refuses `--init-from`
   entirely, before any output directory or training work, because the v2 fair
   comparison must start from the official pretrained weights only; the v1
   continuation path is kept but only accepts a compatible v1/GAP checkpoint;
   the version↔pooling mapping is enforced in both directions (v1 must be GAP,
   a missing `pooling_mode` on a v1 checkpoint is GAP by that format's
   definition, v2 must declare `instance_mask`); `save_material_checkpoint`
   validates the payload with the loader's validator and refuses metadata that
   misdescribes the model's actual pooling; and new `run_config.json` files
   record the real protocol at the top level. The four A1 smoke `run_config`s
   written before this fix carry `version: material_fusion_v1` while their
   validated inner `config.experiment` is already `material_fusion_v2_mask`;
   no reader uses that top-level field, and those historical files are
   deliberately left untouched.

3. **The diagnostics bound only the V4 parent.** A v2 diagnosis now requires
   the checkpoint's concrete V5 fingerprint, its V4 parent fingerprint and its
   mask schema/resize versions to match the data — "the same V4 parent" is not
   enough, since two different mask derivations of one V4 state are different
   datasets. The report records the checkpoint file's SHA256 and the digest of
   the frozen donor protocol. Admission runs before any output directory is
   created, so a refusal leaves nothing behind. The donor matching algorithm,
   features, threshold and seeding are unchanged.

4. **The smoke compared a file with itself.** The save/reload check loaded the
   same checkpoint twice instead of comparing the live model with the saved
   weights, and the "fallback unchanged" check compared a file hash. Now a
   default-off, read-only observation hook in the training entry captures the
   live model's eval logits and the exact bytes written at each production
   save; each capture is compared against the model rebuilt from *those bytes*
   (necessary because a later epoch legitimately overwrites `best.pt`), with
   exact element-wise weight equality asserted separately and a documented
   logit tolerance that only covers cuDNN's algorithm selection on the first
   forward of an input shape. The observation restores module modes and the RNG
   state, and creates no DataLoader iterator inside the window. The fallback is
   now checked functionally through the production routing function
   (`eval_material_fusion.route_batch`, used by `evaluate_with_fallback`): on a
   mixed batch the forced-invalid samples must equal the independent historical
   gray model's output, the valid ones the v2 classifier's, and the fallback
   must receive exactly the invalid rows. The single-stream `gray` variant is
   reported N/A. Finiteness of loss, gradients and metrics is read from the
   training record.

5. **Admission did not re-read the annotation sources.** The audit's
   `label_source_sha256` was only folded into the fingerprint, so editing a
   source annotation after generation went undetected. Admission now recovers
   the annotation mapping from the integrity-checked manifest, resolves each
   path against the project root (refusing absolute, empty or escaping paths),
   re-hashes the files with a per-path cache and recomputes both the label
   source digest and the sample mapping digest with the original algorithm.
   The real V5 dataset still passes with the same digests, and editing or
   deleting a source annotation now refuses admission.

Evidence for the round, all under `analysis/runs/material_mask_a1_fix_<tag>/`:
the four bounded smokes (one per variant, each with every check `ok`), the real
v2 evaluation through the formal CLI, and the focused/full pytest and
`compileall` logs. The four smoke checkpoints are kept under
`runs/train/mask_a1fix_smoke_<variant>_<tag>/`.

## 11. Honest state

- No formal training was run in A1 or in the acceptance round. The 12-run v1
  comparison is unchanged and untouched.
- The smoke weights are plumbing evidence only; they say nothing about whether
  instance-mask pooling or polarization helps.
- The diagnostics are instrument validation on frozen v1 weights. Section 7
  states what the two-plastic-fish result does and does not support.
- Donor matching is partial and matches a few statistics only; shape,
  brightness and acquisition confounding remain.
- The formal orchestrator still pins the v1 protocol constants, so it refuses
  v2 runs. Extending it is A2 work and was deliberately not done here.
  *(Extended in section 13; this bullet describes the A1 acceptance state.)*

## 12. Erratum to the A1 acceptance report (2026-09-26)

The A1 acceptance report (the "Completion report" pasted back to Codex) was
imprecise about the smoke verification in three ways. The measurements
themselves were correct; the prose was not, and it is corrected here rather
than by editing the old report.

1. **"The verifier only uses a stub fallback" was wrong.** The *unit tests*
   use a stub gray model (`_StubGray`) so the routing logic can be checked
   without the real weights. The *actual bounded GPU smoke* loaded the
   historical Gray Model B
   (`runs/train/yolo26_formal_001/gray_fusion/best.pt`) through
   `load_fallback_gray`, i.e. the same identity-checked weights the formal
   evaluation uses.

2. **"The verifier only compares a SHA" was wrong.** The file-hash check is
   one *additional* piece of evidence (that the fallback weights were not
   modified during the run). The primary check is functional: the production
   routing function must send exactly the forced-invalid rows to the fallback
   and the valid rows to the v2 classifier. The SHA is not a substitute for
   that, and it was never the only check.

3. **The mixed-batch size was stated loosely.** The routed check ran on a
   real mixed batch of 8 samples with 2 of them forced invalid
   (`invalid_rows=2/fallback_rows=2` for every dual variant), not on an
   unspecified "mixed batch".

Nothing in the code changed because of this erratum: the smoke evidence stands
as recorded, and the wording in any future report must match it.

## 13. A2 preparation — orchestrator support for the v2 protocol (2026-09-26)

A2 is **not** a completed experiment. This round closes the last admission
gap and extends the experiment orchestrator so a v2 formal run *can* be
started later by the user; it deliberately starts no formal training.

### 13.1 The closed empty-fingerprint gap

Review found that a v2 run could pass admission without ever naming its
dataset: an empty `data_fingerprint` on a v2 config or a v2 checkpoint was
accepted, the diagnostics forwarded that empty value into
`expected_fingerprint`, the comparison was skipped, and the report still
claimed `checkpoint_v5_fingerprint_and_v4_parent` was bound.

A v2 dataset identity is now a strictly validated 64-character lowercase hex
SHA256 everywhere, with no `str()`/`strip()`/zero-padding repair:

- `core.material_mask_dataset.require_fingerprint` is the single validator.
- A `material_fusion_v2_mask` config must pin a concrete `data_fingerprint`
  and a concrete `parent_data_fingerprint` (`validate_material_config`).
- A `material_fusion_v2_mask` checkpoint must record a concrete
  `data_fingerprint` (`validate_checkpoint_metadata`).
- `admit_dataset_for_protocol(pooling_mode="instance_mask", …)` refuses an
  empty or malformed `expected_fingerprint` **or**
  `expected_parent_fingerprint`: the read-only audit may still run without an
  expectation, but a protocol admission that claims a binding may not.
- `check_run_config_checkpoint_consistency` treats an empty fingerprint on
  either side of a mask-pooling pair as a mismatch, not as "unasserted".
- Training, evaluation and diagnostics all route through the protocol
  admission layer, so none of them can bypass it; the diagnostics label a
  binding only *after* the fingerprint was really compared.
- The v1 GAP protocol is unchanged: a v1 config may still leave the
  fingerprint unset, which is what the template/legacy path relies on.

### 13.2 Explicit protocol selection in the orchestrator

`scripts/run_material_experiments.py` now takes
`--protocol {v1,v2_mask}` on all four subcommands. The default stays `v1`,
so every existing command keeps its meaning.

| | `v1` | `v2_mask` |
|---|---|---|
| experiment | `material_fusion_v1` | `material_fusion_v2_mask` |
| config | `configs/experiments/material_fusion_v1.yaml` | `configs/experiments/material_fusion_v2_mask.yaml` |
| dataset | `datasets/underwater_cls_fusion_v4_band` | `datasets/underwater_cls_fusion_v5_instance_mask` |
| pooling | `gap` | `instance_mask` (+ mask schema/resize, V4 parent) |
| run id | `cmx_v1_<variant>_s<seed>` | `cmx_v2_mask_<variant>_s<seed>` |
| output | `runs/train/<run-id>/material_fusion/` | same |
| management | `analysis/runs/material_experiments/` | `analysis/runs/material_experiments_v2_mask/` |

The protocol is carried as an explicit `ProtocolSpec` through every function;
no mode mutates a module global to switch protocols. Each protocol has its own
run manifest, and a manifest recording a different protocol is refused rather
than adopted. A v1 task refuses a v2 run directory and vice versa, even though
both matrices use the same `<variant>_s<seed>` task ids.

`plan` and `preflight` accept an optional `--output` so a review round can
write unique evidence without touching historical files; an existing file is
still never overwritten.

`run` re-derives the essential admission (config, protocol constants, data
identity, dataset admission, pretrained SHA, device) immediately before its
first spawn — a saved `preflight.json` is never trusted in place of that
re-check — and still stops on the first failure and records every attempt.

`collect` now: uses the protocol's own dataset by default; keeps `--data` as a
path relocation that never relaxes the identity (the relocated dataset gets a
structural audit that is explicitly marked as not bound, and each checkpoint
is bound to the concrete fingerprint it records); passes the instance mask
into a v2 model through the shared batch unpacker; and writes
`per_sample_predictions.jsonl` with one record per task per val sample
(`sample_name`, `split`, `group`, `label`, raw logits, raw prediction, routed
prediction, whether the fallback was used, checkpoint SHA256, dataset
fingerprint) so the aggregates in `report.json` are recomputable from the
records alone. A missing fallback still yields `routed: null` with a stated
reason rather than a fabricated routed metric.

### 13.3 Completion validation

`validate_completed_task(task, spec)` validates against the *expected*
protocol, never against the artifact's own claim. For a v2 task it requires
the recorded experiment, the run-id prefix, the pooling mode, the mask
schema/resize versions, well-formed and internally consistent dataset and
parent fingerprints across the config, `run_config.json` and both
checkpoints, an empty `init_from`, a contiguous 30-epoch history with the
protocol stage at every epoch, and a best epoch that really is the history's
selection. A smoke directory is refused on its own recorded budget (2 epochs,
`limit_batches=3`), not merely because a v1 constant was applied to it.

### 13.4 What A2 has not done

- No formal training of any kind. No new GPU smoke. No V4/V5 rebuild, no
  annotation or audit edit, no historical report or weight overwrite.
- v2 is **not** shown to give polarization an independent generalization
  benefit. There is still no v2 result of any kind: the 12-run v2 matrix has
  never been trained.
- The two-plastic-fish reading from section 7 is unchanged and stays cautious:
  the two samples are each other's donors, so a swap that leaves the
  classification correct cannot rule out a shared polarization-value feature.
  The donor protocol was not changed and no new diagnostic was added.
- The orchestrator's auxiliary tasks (bias baseline, shuffle ablation) are
  listed by `plan` but never launched by any mode.

### 13.5 Next step

Codex review, then an explicit user authorization to train. The user runs the
commands manually in a VS Code PowerShell with the `stereo` conda environment
active — see the A2 preparation report for the exact sequence
(plan/preflight, then the seed-2026 four-variant block, then the remaining
seeds after review, then collect). Nothing in this repository starts them
automatically, and there is no logic that reads val scores to change the
protocol or keep only favourable seeds.

## 14. A2 fix round — protocol identity, smoke admission, report digest (2026-09-27)

Three acceptance gaps found in review were closed. No formal training was run,
no V5/V4 data, annotation, audit, checkpoint or historical report was touched,
and no new `cmx_v2_mask` run directory was created.

### 14.1 The chosen protocol is the identity authority

The completion validator checked that a run was *internally* consistent
(config ↔ `run_config.json` ↔ checkpoints) but never that the identity it
recorded was the *chosen protocol's* identity. A complete, self-consistent run
of another dataset state or another pretrained weight therefore validated as
`(True, "ok")`.

`validate_completed_task(task, spec)` now requires the config's
`data_fingerprint`, `parent_data_fingerprint` and `pretrained_sha256` to equal
the `ProtocolSpec`'s, in addition to every previous check (field types, the
fixed budget, smoke markers, `init_from`, the 30-epoch history, best-epoch
selection, mask schema/resize, config↔checkpoint consistency). The rules come
from the expected protocol, never from what the artifact claims about itself.

`preflight`/`run` admission now requires the config's data and parent identity
to match the spec, and the pretrained weight to agree **three ways**: the file's
actual sha256, the config's recorded sha256 and the spec's. A config whose SHA
merely matches a swapped file no longer passes. The dataset admission is issued
with the spec's fingerprint, so a config that disagrees cannot widen it.

`collect` binds the evaluation data to the chosen protocol for **both** the
default path and `--data`: `--data` relocates the read path (a byte-identical
copy has the same fingerprint) but never switches to another dataset version.
The V5 audit (digest re-verification, mask content, annotation sources, sample
mapping, V4 parent, mask schema/resize) still runs, the per-checkpoint data
binding is kept, and an identity failure refuses the whole collection instead
of writing a report. The previous "structural audit that binds nothing" mode is
gone.

Synthetic tests no longer require the production rules to be relaxed: a test
that needs a legitimate run injects its own protocol via
`conftest.build_test_protocol_spec` (a `dataclasses.replace` of a real spec
whose dataset/pretrained identity is the fixture's) and
`conftest.install_test_protocol`. Production defaults are unchanged and there
is no CLI switch that bypasses admission.

### 14.2 The smoke verifier judged a v2 run as v1

`scripts/verify_material_mask.py` built its `ExperimentTask` without a protocol,
so it defaulted to v1 and refused the v2 smoke directory because its experiment
is not `material_fusion_v1` — a reason containing no budget information, which
made the smoke-isolation check fail. The production path now calls
`smoke_formal_admission` (v2_mask task + `V2_MASK_SPEC`), shared with its tests,
and the check `smoke_refused_on_budget` requires the refusal to *be* the
formal/smoke budget separation: it must name the fixed protocol and
`limit_batches`/`smoke`, and it explicitly rejects an unrelated refusal (a
missing directory, a wrong protocol, a foreign identity). All 12 recorded v2
smoke runs are refused on their own budget (2 epochs, `limit_batches=3`).

### 14.3 The per-sample digest is the digest of the file

`collect` hashed the pre-write text while writing it with `Path.write_text`,
which translates LF to CRLF on Windows; the report's `sha256` therefore did not
describe the file on disk. The JSONL is now written as explicit UTF-8/LF bytes
and the digest is taken from the written file (`read_bytes()`). Record fields,
order and content are unchanged, and an existing output directory is still
refused. A regression writes non-ASCII content with an embedded newline and
verifies the report digest against an independent `read_bytes()` hash.

### 14.4 State

The 12 v1 runs still validate under the v1 protocol, the legitimate v2 config
and the relocated-same-identity path still pass, and the real v2 plan/preflight
still pass. v2 remains unproven: nothing here trains, and the 12-run v2 matrix
has still never been run.
