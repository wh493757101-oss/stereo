# Material fusion v2_mask — runtime integration (4A)

Fourth-round engineering record. It wires the frozen CMX v2_mask research
candidates into an independent runtime input mode and proves the *plumbing*:
input consistency, instance association, the independent gray fallback and a
bounded real forward. It does **not** produce a model-quality, polarization,
generalization or FPS conclusion, and no formal evaluation (4B) was started.

## 1. What this batch adds

- A new independent Model B input mode ``material_fusion``
  (``gui/inference_engine.py``). The three legacy modes (``gray``,
  ``polar``, ``fusion``) and the default configuration are untouched;
  ``material_fusion`` is strictly opt-in and must not be confused with the
  old YOLO-residual ``fusion`` mode.
- ``models/material_classifier.py``: the runtime adapter.
  - ``MaterialInstanceInput`` — one instance's runtime input, carrying
    ``frame_id``, ``instance_id``, ``bbox``, ``gray``, ``signed_q``,
    ``abs_q``, ``polar_valid``, ``instance_mask``, ``quality``,
    ``polar_invalid``, ``invalid_reason`` and an explicit ``mask_source``.
  - ``MaterialFusionClassifier`` — loads one frozen v2_mask checkpoint
    through the strict loader and refuses any contract conflict at
    construction: protocol version, pooling mode, canonical class order,
    preprocessing/mask versions, smoke markers (a smoke checkpoint cannot
    serve the runtime), an expected file sha256. Dual variants require an
    explicit independent gray fallback; the single-stream ``gray`` variant
    (E1) refuses one — it is the pure control and never swaps to another
    gray model.
  - ``material_input_tensors`` — the offline-identical preprocessing. It
    reuses ``models.polar_fusion._fusion_tensors`` and the extracted
    ``core.material_fusion_dataset.instance_mask_tensor``; no second copy of
    the training/inference math exists.
- ``configs/experiments/material_runtime_v2_mask.yaml``: the independent
  experiment config pinning Model A, the fallback and all 12 candidate
  identities (sha256), the engineering representative seed (2026 — an
  integration rule, not a selection), the CMX input contract, calibration,
  the accepted V4/V5 stereo/polar parameters, runtime execution and the
  separate training-data vs runtime-data identity fields.
- ``scripts/verify_material_runtime.py``: the bounded two-layer verifier
  (section 3).
- ``tests/material_fusion/test_material_runtime.py``: 33 regression tests
  (TDD: written first, RED confirmed at 33 failed before implementation).

## 2. Instance mask provenance at runtime

The instance mask handed to the models is the **Model A prediction**
(``mask_source="predicted_instance_mask"``), cropped to the same padded
window as every other mode and binarized. Ground-truth masks exist only in
clearly labelled verification inputs (``ground_truth_annotation`` /
``npz_offline_identity``) and can never mix with the live path — the
``MaterialInstanceInput`` constructor refuses an unknown ``mask_source``.

The three regions stay distinct exactly as in V5: ``instance_mask`` (target),
``polar_valid`` (measurement-valid), and their intersection. The runtime
never treats ``polar_valid`` as the instance mask and never recovers a mask
from non-zero q. The polar crops are restricted to the instance mask exactly
as the offline per-instance samples were built.

## 3. The two verification layers

``scripts/verify_material_runtime.py --config configs/experiments/
material_runtime_v2_mask.yaml --device 0 --max-samples 16 --max-frames 8
--output-dir <fresh dir>`` (an existing directory is refused; reports are
never overwritten, and every file's digest is the digest of the bytes
actually on disk).

Hard budget: at most 16 NPZ samples, 8 real frames, 15 minutes of measured
real-forward time. Sample and frame selection is fixed in advance (first N
val rows in manifest order; evenly spread unique val frames in sorted
order) — never chosen from model outputs.

- **Layer A — same V5 NPZ inputs.** The runtime adapter is fed the *same*
  val npz samples the offline dataset yields. The tensors must be exactly
  equal and the eval-mode logits of the four engineering variants (seed
  2026) must agree within the pre-fixed ``atol=rtol=1e-5``. A failure is
  localised, never papered over by widening the tolerance.
- **Layer B — real stereo frames.** The configured Model A runs on the
  rectified val frames; band SGBM + polar features build the runtime inputs;
  the four variants classify them through the production engine with the
  independent gray fallback routing. For the same frames the offline V5
  generation is replayed from the GT annotations (reusing the generator's
  own helpers) and compared against the stored npz (crop window, gray crop,
  mask, quality). Predicted inputs are not required to equal GT inputs;
  provenance and association must be correct.

Every forward is preceded and followed by a parameters+buffers snapshot of
each model and the fallback: inference must not change the weights.

## 4. Fallback and anomaly policy

- ``valid_ratio=0`` or an explicit ``polar_invalid`` → the instance is
  routed to the **independent** gray fallback (the historical
  ``runs/train/yolo26_formal_001/gray_fusion/best.pt``, identity-checked at
  load). The old gate=0 fusion logic is never used to impersonate a CMX
  fallback.
- Non-finite polar arrays → the instance is marked polar unavailable
  (``non_finite_polar``) and routed to the fallback; values are never
  silently replaced and reported as valid polarization.
- Missing timestamps → sync status stays ``unknown`` (never 0 ms,
  never ``assumed_synced``). No new forced fallback is added for unknown
  sync, but polarization already judged unusable by the existing rules is
  never restored.
- A corrupted checkpoint, wrong class order or version conflict →
  construction refused.
- An unusable gray crop, an empty instance mask or a malformed instance →
  an explicit per-instance failure (``valid=False`` with a reason), never a
  fabricated four-class label. Failed instances reach neither model.
- No detections → the classifier is not called (a legal empty output).
- No confidence- or "looks good"-based fallback thresholds exist.

Batch handling: results are scattered back by stable ``instance_id``; the
output count is checked against the input count (a shorter result is an
error, never a silent ``zip`` truncation), and a result whose identity does
not match the instance it lands on is refused. A classification failure
never changes another instance's class or depth association.

## 5. 4B preparation (recorded; NOT started)

The verifier report records the 4B protocol skeleton, and this document is
its working note:

- **Candidates and runtime identities**: all 12 checkpoints with sha256,
  the Model A and fallback identities, are pinned in
  ``configs/experiments/material_runtime_v2_mask.yaml`` and re-verified by
  the verifier before anything runs.
- **GT condition vs predicted condition** (to be fixed before 4B):
  *GT condition* — all annotated objects of a frame (from the label file),
  whether or not an npz sample exists; *predicted condition* — all Model A
  detections of the same frame with predicted instance masks. Metrics:
  per-class precision/recall/macro-F1 under the fixed class order, per
  condition, plus the routed result under the fallback rules.
- **Retrospective val frame-level index**: the val manifest gives
  ``source_frame``/``object_index`` per sample and the label files give all
  polygons per frame, so a complete per-frame GT index (including objects
  without a valid stereo measurement, which have no npz) can be built
  read-only. Gap to close before 4B: this index must be written and its
  coverage (npz vs all GT objects per frame) reported — objects without a
  valid stereo measurement must still enter the GT condition.
- **Admission fields for independent data**: a 64-hex audited dataset
  fingerprint, the V4 parent fingerprint, mask schema/resize versions, the
  canonical class order and capture-group isolation. The training V5
  fingerprint and a new evaluation dataset's fingerprint are verified
  separately — never forced equal, and the existing source checks are never
  globally disabled to admit new data.
- **Thresholds, metrics, fallback rules, performance boundaries**: all to
  be fixed *before* 4B collects anything; none are tuned in this batch.

## 6. Honest state

- This batch is engineering integration only. The val Macro-F1 numbers of
  the 12 candidates (E1 100.00 %, E2 99.75 %, E3 99.51 %, E4 99.09 %) are
  model-selection evidence on 4 val capture groups; they say nothing about
  independent generalization.
- No evidence here shows a polarization benefit. The closeout paired flips
  showed E4 breaking 1–3 val samples per seed and fixing none.
- Nothing here is a real-time or deployment claim; no FPS is measured.
- Model A's training data shares its capture sources with the classifier
  data; its independence is recorded as ``unknown`` in the config, not
  claimed.
- The bounded real forward covers 8 val frames with whatever objects they
  contain; rare cases (no-target frames, high-object-count frames) are
  covered by unit tests on the interface only and are not claimed as real
  coverage.

## 7. Closeout round (2026-09-28)

A closeout round re-verified the 4A artifacts, closed the one evidence gap
(polarization *content* parity), built the complete frame-level GT index 4B
needs, and removed the round's own scratch directory. It ran no training, no
model forward, no 4B evaluation, and it changed no data, annotation,
manifest, audit, weight or historical report.

Tool: ``scripts/closeout_material_runtime.py`` (modes ``all`` /
``verify-evidence`` / ``polar-parity`` / ``gt-index`` / ``cleanup``), with
``tests/material_fusion/test_material_runtime_closeout.py`` (35 tests,
written before the implementation). Evidence:
``analysis/runs/material_runtime_v2_mask/4a_closeout_20260928T073812Z/``.

### 7.1 Historical evidence re-verification — PASS

``evidence_verification.json``. Re-checked against the bytes on disk:

| check | result |
|---|---|
| the four evidence JSONs parse | PASS |
| recorded digests == sha256 of the files on disk | PASS (3 covered files) |
| 12 candidates + Model A + fallback identities recomputed | PASS (no mismatch) |
| report internal consistency (frame/gt_replay counts, statuses) | PASS |
| preserved run logs | 6 logs in the round's scratch, all hashes recorded |

Honest limits: the historical pytest counts (33 / 797 / 1439) have **no
preserved pytest log**, so they stay **UNVERIFIED** in this round — they are
a historical claim, not re-executed here. ``digests.json`` is the digest
index and does not cover itself (no circular hash). The verifier's
``status: PASSED`` is corroborated by ``verify_run6.log``; ``verify_run5.log``
shows a post-report ``KeyError`` (the report had already been written) and
``verify_run1-4.log`` show the intermediate refusals — all consistent with
the recorded history and left untouched.

### 7.2 Polarization content parity — PASS

``polar_content_parity.json``. The original 4A run compared only the valid
support and the quality vector, which cannot show that ``signed_q``/``abs_q``
*content* matches the stored samples. This round rebuilt the polarization
for the **same 8 frames the 4A report lists** (recovered from
``report.json``; if the list could not be recovered the item would have been
abandoned rather than re-chosen) using the generation's own helpers and the
accepted parameters (band bboxes from polygon extrema, SGBM 3way/768/0.25/7,
gain 1.0, min_intensity_sum 10.0, epsilon 1e-6, pad 10).

- 17 GT instances, 8 frames, **17 PASSED, 0 FAILED, 0 without an NPZ**.
- ``signed_q`` and ``abs_q``: **exact equality for 17/17**; max absolute
  error **0.0** over all pixels and inside the target region, at the
  pre-fixed ``atol=1e-6, rtol=0``.
- Invalid-region zeroing contract holds; ``polar_valid ⊆ instance_mask``
  holds; the rebuilt valid support equals the stored one for every instance.

No classifier, no Model A and no NPZ write was involved.

### 7.3 Complete val GT index — COMPLETE

``val_frames_gt.jsonl`` + ``gt_index_summary.json``, built from the
authoritative frame-level split list
(``datasets/underwater_seg_binary_v2/split_assignment.csv``) and the
four-class annotations (``datasets/underwater_seg_v2/labels``); the
single-class binary label set is explicitly refused as a GT source.

| quantity | value |
|---|---|
| authoritative val frames / indexed frames | 180 / 180 |
| frames per capture group | 45 × 4 |
| GT instances | 368 (metal 90, plastic_submarine 63, plastic_fish 197, real_fish 18) |
| V5 val samples | 368 |
| GT without an NPZ | 0 |
| NPZ without a GT source | 0 |
| frames missing an annotation | 0 |
| empty GT frames | 0 |
| duplicates / out-of-range polygons / unknown classes | 0 / 0 / 0 |
| stereo validity | ``not_evaluated`` (no SGBM was run for the index) |

Honest limit: in *this* dataset every val GT object happens to have an NPZ
and no val frame is empty, so the index's handling of "GT without NPZ",
empty GT frames and missing annotations is exercised by the unit tests
(synthetic fixtures), not by real val data. Sync metadata is ``unknown``
(the frame-level sources carry no measurable timestamps).

### 7.4 Cleanup — CLEANED

The round's own scratch ``.test_artifacts/4a_20260928T064802Z`` was verified
(exact literal path, inside ``.test_artifacts``, no symlink/junction, only
the expected ``temp``/``pycache`` plus run logs), its 6 logs were copied to
``preserved_logs/`` with matching sha256 before/after, and only then was the
directory removed. The other task's directory
``.test_artifacts/run_20260924T092544Z_24256`` was not touched.
``pycache`` (2276 files) and ``temp`` were regenerable caches and were not
preserved.

## 8. Entry-point and valid-consistency fix (2026-09-28)

A review of the 4A round found two engineering gaps. Both are closed here;
the historical measurements (17 instances with byte-identical polarization,
180/180 frames, 368 GT instances) are unchanged and were not re-derived.

### 8.1 The experiment config had no construction entry point

Three defects, one root cause — the runtime path was assembled by hand
instead of by a supported entry point:

- `DualStageInferenceEngine.from_config` reads `model_b.path`, which the
  experiment config does not have (`candidates` / `material_model`), so
  passing it there raised `KeyError: 'path'`;
- constructing with `model_b_input_mode="material_fusion"` and a bare
  `model_b_path` produced a dual-stream adapter **with no fallback bound**;
- the earlier smoke only worked because the verifier assembled the engine
  itself, so "the experiment config can start the pipeline" was never
  actually demonstrated.

New entry point: `DualStageInferenceEngine.from_material_config(config_path,
variant=..., seed=..., device=None, model_a_factory=None)`.

- `variant` and `seed` are mandatory and must match exactly one candidate;
  a missing or duplicated pair is refused — never "latest", "best-scoring"
  or "first that loads".
- The selected checkpoint's file sha256, its own metadata and the config's
  declared contract (imgsz, class order, preprocessing/mask/pooling
  versions, upstream commit, V5 fingerprint, V4 parent) must all agree.
- Model A's sha256 and — for E2/E3/E4 — the independent gray fallback's
  sha256 are verified before anything is loaded. E1 (gray) is the pure
  control and binds no fallback; the old default gray is never substituted.
- Every declared value reaches the engine: device, Model A thresholds and
  imgsz, crop padding, calibration/rectification, baseline/focal length,
  band margin and the full SGBM parameter set. Fixed values the engine does
  not expose as parameters (numeric precision, the polarization
  gain/epsilon/min_intensity_sum, the band full-image threshold) are
  checked against the engine's real behaviour and refused on a mismatch.
- Relative paths resolve against the project root, never the cwd.
- `from_config` on a material config now names the factory instead of
  raising `KeyError`, and a bare `model_b_path` construction is refused
  with the same pointer. Injecting a prepared `model_b` (tests, controlled
  verification) still works.
- `scripts/verify_material_runtime.py` builds its engine through the
  factory and its per-variant classifiers through the same
  identity-checked builder the factory uses, so the verified pipeline is
  the production pipeline.

### 8.2 The valid support was recorded but not enforced

`compare_instance_arrays` computed `rebuilt_valid_equals_npz_valid` and
then ignored it: two identical all-zero `signed_q`/`abs_q` arrays with a
stored valid support of all ones and a rebuilt support of all zeros still
returned `PASSED`. The valid channel decides which pixels were a
measurement at all, so it is now part of the verdict: a missing rebuilt
valid, a shape mismatch and a value mismatch all fail, the failure names
its reasons and propagates to the frame, the report and the CLI exit code.
`q == 0` inside a valid pixel remains a legal measurement and is not
re-classified as invalid.

### 8.3 This round's verification

- New tests: `tests/material_fusion/test_material_runtime.py` (+15),
  `tests/material_fusion/test_material_runtime_closeout.py` (+7),
  `tests/test_inference_engine.py` (+2). Focused 164 passed; full suite
  1497 passed, 1 skipped.
- Bounded real run through the factory (2 val NPZ samples, 1 val frame,
  seed 2026): layer A tensor-exact with logits max error 0.0 for all four
  variants; layer B association correct; GT replay PASSED; parameters and
  buffers unchanged; measured real forward 1.9 s.
- Forced polar-invalid check (in memory only, one frame): all four
  instances routed to the fallback, and the routed predictions are exactly
  the locked independent gray model's own output, with instance ids, count
  and order preserved. This is a forced check, **not** a natural fallback
  observation.

Evidence: `analysis/runs/material_runtime_v2_mask/4a_entry_fix_<tag>/`.
No training, no 4B evaluation, no data/weight/config change.

## 9. 4B retrospective evaluation and offline-replay performance (2026-09-28)

Fixed-protocol evaluation and timing over the existing val split. No
training, no re-selection, no test-split use, no new independent dataset
(generalization stays INSUFFICIENT_EVIDENCE). All 12 candidates were
evaluated through the supported factory entry point
(`scripts/eval_material_pipeline.py`,
`scripts/benchmark_material_pipeline.py`).

### 9.1 Quality (evidence: `4b_<tag>/quality/`)

- 180 val frames, 368 four-class GT instances, 380 Model A predictions at
  the fixed operating point (conf 0.25, NMS IoU 0.45, imgsz 640).
- Class-agnostic one-to-one matching at mask IoU >= 0.50: 346 matched,
  22 missed GT, 34 unmatched predictions, 0 failed frames. Detection
  P/R/F1 = 0.910 / 0.940 / 0.925 (fixed point, never called mAP); matched
  IoU mean 0.874 (min 0.662).
- Layer A1 (V5 NPZ entry, E4 s2026): pure Macro-F1 0.9926 — matches the
  recorded offline scores.
- Layer B (real Model A): conditional classification on matched instances
  E4 s2026 accuracy 1.000, E1 s2026 0.997; **end-to-end** Macro-F1
  ≈ 0.77 — the detection losses dominate, which is exactly what the
  end-to-end metric is for.
- Same-seed paired changes (E4 vs E1): s2026 fixed 1 / broke 0; s2027
  0/0; s2028 1/0 on matched instances. Paired cluster bootstrap (4 groups,
  2000 resamples, seed 2026): E4-E1 correctness delta CI straddles or
  touches 0 for every seed — **no polarization benefit is established**.
- Natural fallback (a real observation this time, unlike the 8-frame
  smoke): 3 of ~430 routed instances in E4 s2026 (low_valid_ratio 1,
  no_valid_disparity 2).

### 9.2 Performance (evidence: `4b_<tag>/performance/`)

RTX 4060 Laptop, driver 581.57, torch 2.12.1+cu130, CUDA 13.0, float32,
models preloaded, warmup 10 / measured 100, offline replay of preloaded
already-rectified frames.

| scenario (actual Model A count) | E1 total p50 | E4 total p50 | E1 FPS | E4 FPS |
|---|---|---|---|---|
| 1 target (85 natural frames) | 162 ms | 177 ms | 6.05 | 5.62 |
| 2 targets (25 frames) | 205 ms | 215 ms | 4.81 | 4.63 |
| 5 targets (3 frames) | 285 ms | 290 ms | 3.49 | 3.44 |
| max load (1 frame) | 305 ms | 317 ms | 3.26 | 3.18 |

- Per-stage diagnostics (synchronized): Model A 23-56 ms, band SGBM
  38-42 ms, depth+polar 35-50 ms, E4 classification 39-82 ms (the
  dual-stream CMX is the heaviest classifier stage), E1 15-48 ms, E4
  all-fallback routing 14-23 ms. The stage sum is a diagnostic, never the
  total.
- Peak VRAM 135-171 MB. Standalone online rectification ≈ 8.6 ms (excluded
  from the replay total).
- The 0-target bucket has **no natural val frame** at conf 0.25 — reported
  NOT_COVERED, nothing synthesized.
- Queue latency, drop rate, camera and GUI: NOT_MEASURED. This is **not**
  a camera-to-GUI end-to-end FPS claim. 20-30 FPS is a target, not a
  finding. No distance ground truth exists, so no distance-accuracy claim
  is made.

Two benchmark defects were found and fixed during the round (both
preserved as failed-attempt logs): the survey fed Model A a 1-channel
array instead of the production 3-channel copy, and the end-to-end
lambda returned a result object instead of the elapsed time.

## 10. 4B fix round (2026-09-28)

Five gaps found in review of the first 4B report were closed. Nothing was
trained, no candidate was re-selected, no data/annotation/weight/config was
changed, and the original evidence directories were not modified.

### 10.1 The "end-to-end Macro-F1 bootstrap" was not a Macro-F1 bootstrap

The first report's `bootstrap_end_to_end_macro_f1_e4_vs_e1` averaged a
per-GT 0/1 correctness difference and never computed Macro-F1 or counted
unmatched predictions as false positives. It is now:

- `bootstrap.instance_correctness_delta` — the same per-GT correctness
  delta, under its own accurate name
  (`paired_gt_instance_correctness_delta`);
- `bootstrap.end_to_end_macro_f1_e4_vs_e1` — a real paired cluster
  bootstrap: whole capture groups drawn with replacement (duplicate draws
  weighted, never de-duplicated by frame id), the same draw for both
  candidates, per-class TP/FP/FN re-accumulated per draw (misses and
  unmatched predictions included) and the fixed four-class Macro-F1
  difference recorded. 2000 resamples, seed 2026, percentile interval. The
  point estimate is the full-sample Macro-F1 difference, not the bootstrap
  mean.

Corrected B-routed results (point estimate and 95 % interval):

| seed | E1 Macro-F1 | E4 Macro-F1 | E4-E1 (Macro-F1) | interval | correctness delta |
|---|---|---|---|---|---|
| 2026 | 0.774819 | 0.771314 | -0.003505 | [-0.0236, +0.0855] | +0.002717 |
| 2027 | 0.774397 | 0.770591 | -0.003805 | [-0.0071, +0.0415] | -0.002717 |
| 2028 | 0.767961 | 0.773027 | +0.005066 | [-0.0218, +0.0236] | +0.005435 |

Every interval spans or touches zero: **no polarization benefit is
established**. With 4 val capture groups, and val having been used for model
selection, these intervals are indicative only.

### 10.2 The A2 GT geometry disagreed with V5 for 15 of 368 samples

The first evaluation derived both the band bbox and the crop window from the
rasterised mask's bounding box. The generation contract keeps them apart:
band bbox = raw polygon extrema, crop window = `make_crop_window`
(floor/ceil + padding + clipping). A shared helper (`core/gt_geometry.py`)
now builds both from the source annotations, and the engine's GT path takes
an explicit, GT-only `crop_windows` override (never set by
`process_frame_detailed`, so the Model A path is unchanged).

A second, smaller gap surfaced during the alignment run: the generation
writes a **zero quality vector and zero polar channels** for an instance
whose stereo statistics are invalid, while the live engine reports the
measured in-bounds/brightness ratios. The GT path now reproduces the
generation contract through a second GT-only flag
(`gt_generation_contract`), which is likewise never enabled on the Model A
path. Four of 368 samples needed this.

Alignment result over all 180 frames / 368 GT instances with an NPZ
(`a2_alignment.jsonl`, `a2_alignment_summary.json`): **368/368 PASSED** —
gray, instance_mask, polar_valid, signed_q, abs_q and quality all match the
V5 arrays (quality at atol 1e-6, rtol 0; the images and masks exactly), the
four seed-2026 variants' input tensors are element-wise identical to the
offline dataset's, and their logits agree with the offline forward within
atol = rtol = 1e-5. Tolerances were never widened.

### 10.3 Reporting completeness

- A1 (offline V5 entry), A2 (GT-controlled real frames) and B (real Model A
  predictions) now have separate summaries, each with a `pure` (no-routing
  diagnostic) and a `routed` (production routing) view; E1 stays the pure
  CMX control and is never swapped for the independent gray fallback.
- Per candidate: support, accuracy, Macro-F1, per-class P/R/F1 and the
  confusion matrix, plus the same-seed E4-vs-E1/E2/E3 paired change lists.
- Full-precision raw logits are stored per instance (no rounding).
- Stratification reports metrics (not just counts) per capture group and per
  polar-quality bin; the quality binning is explicitly conditional on a
  match, and unmatched GT are listed separately without a fabricated
  predicted quality.
- All 22 unmatched GT instances are recorded in `unmatched_gt.jsonl` with
  their maximum mask IoU against any prediction and whether that prediction
  was already matched to another GT instance; unmatched predictions and
  unclassified events have their own files. Count conservation holds
  (`matched + missed = GT`, `matched + unmatched = predictions`).
- Real fish: 18 GT, of which only 3 are matched and correct. "Unmatched" is
  not "not detected at all", and a filename is never treated as label
  evidence. No root cause is asserted for the unmatched real fish.

The B-path numbers are **unchanged** by this round (the same 380
predictions, 346 matches, 22 missed GT, 34 unmatched predictions per
candidate and the same per-seed Macro-F1), which is the expected outcome:
the fixes touched the GT path and the reporting, not the predicted path.

### 10.4 Independent-data evaluation is closed, not "complete"

The first report claimed the independent-data admission gate was done. That
was wrong: the helper only checked a manifest's class order and source
overlap and could be mistaken for a passed gate. It has been **removed**;
`--independent-data` now exits non-zero *before* any output directory is
created or any model is loaded, with an explicit "not implemented" message.
Building a real independent evaluation still requires (as future work): a
train/val/diagnosed-test separation that is verified rather than assumed,
Model A's training provenance, capture-sequence and content-duplication
checks, strict audit binding, and a real evaluation-source switch.

### 10.5 Performance measurement corrections

- The selected frame per bucket is written to `scenario_selection.json`
  **before** timing; `available_frames` and `measured_unique_frames` are
  separate fields (85/25/3/1 candidates, 1 frame timed per bucket —
  repeating one frame is not scene coverage).
- Device stages synchronize before the start timestamp and after the stage;
  the whole-frame latency is measured independently and the stage sum is
  never the pipeline latency.
- `classification_wrapper` is named for what it is (the `predict_batch`
  call including its own preprocessing, host-to-device transfer, forward and
  result assembly); crop-only, H2D-only and pure-forward-only are
  NOT_MEASURED, and the earlier "the dual-stream forward is the primary
  bottleneck" claim is **retracted** — it was based on that wrapper time.
- E1 and E4 are resident and measured sequentially (the first engine is
  released before the second is built), with post-load VRAM, peak allocated
  and peak reserved recorded per engine (E1 13 MB loaded / 102 MB peak
  allocated; E4 90 MB loaded / 156 MB peak allocated).
- Raw repeat arrays are saved (`raw_timings.json`) and every summary value
  derives from them; actual batch size (1/2/5/6 instances) is recorded
  separately from the configured `classification_batch=8`.
- Fallback measurements are labelled classifier-level, report the actual
  CMX/fallback routing counts, and are NOT_APPLICABLE for a single-instance
  frame (no mixed batch possible).
- Corrected numbers (100/100 repeats per item, offline replay): 1 target
  E1 124 ms / E4 141 ms (8.0 / 7.1 FPS), 2 targets 154/170 ms, 5 targets
  221/235 ms, maximum load 240/254 ms (4.2/3.9 FPS). The 0-target bucket has
  no natural val frame, so it is NOT_COVERED and the overall report status
  is **PARTIAL** for that reason alone.
- Scope is unchanged and explicit: offline replay of preloaded,
  already-rectified frames; camera, GUI, inter-stage queueing, queue latency
  and drop rate remain NOT_MEASURED. Standalone online rectification is
  ~7.7 ms and is excluded from the replay total. This is still **not** a
  camera-to-GUI end-to-end FPS claim and says nothing about model quality.

### 10.6 This round's verification

- New/updated tests: 27 (eval + speed fix-round coverage). Focused
  863 passed; full suite 1561 passed, 1 skipped; `compileall` and
  `git diff --check` clean.
- Evidence: `analysis/runs/material_runtime_v2_mask/4b_fix_<tag>/`
  (`quality/`, `performance/`, logs, and the preserved first-attempt
  alignment failure). The original `4b_20260928T091849Z` evidence is
  unchanged and is quoted only for comparison.

## 11. 4B closeout round (2026-09-28)

Five residual gaps found in review of the fix round were closed. No model
was loaded, no inference, no benchmark and no training ran in this round;
the historical evidence was read only.

### 11.1 The pure diagnostic crashed on a leading non-finite input

`predict_pure_batch` decided a row's evaluability *after* appending it to
the forward lists and then popped from lists it had not appended to yet —
an `IndexError` on a leading NaN — while the index list used for backfill
was never filtered, so one bad row could shift every later row's result.
The decision is now made before anything is appended, the forward tensors,
routing flags and reasons are indexed by an explicit `forward_indices`
list, results are written back by original index, and a batch with no
forwardable row returns the constructed failures without touching
`torch.stack`, the CMX model or the fallback. Routed behaviour, E1's
independence from the fallback and all-finite results are unchanged
(verified with a micro CPU model whose class depends on each row's own
content, so a backfill-order bug cannot hide behind a constant output).

### 11.2 The fallback summary was wrong

It read a field that does not exist (numerator 0) and mixed pure and
routed rows into the denominator (760 instead of 380). A shared
`summarise_fallback` — used by both the evaluation and the closeout entry —
now counts exactly `layer=B_model_a_predictions, condition=routed` per
variant/seed, takes the numerator from the real `used_fallback` field,
keeps unclassified instances in the denominator and lists them separately,
requires the reason counts to sum to the numerator, refuses duplicate
instance keys, missing fields and unknown identities, and asserts that E1
never falls back. Recomputed from the historical records:

| candidates | fallback | reasons |
|---|---|---|
| E2/E3/E4 (all seeds) | 3 / 380 | `no_valid_disparity` 2, `low_valid_ratio` 1 |
| E1 (all seeds) | 0 / 380 | — |

### 11.3 The alignment verdict was too permissive

`_alignment_summary([])` returned PASSED, a set of NO_NPZ-only rows
returned PASSED, a single UNAVAILABLE check was tolerated, and
`signed_q`/`abs_q` were compared with atol 1e-6 instead of exact equality.
The strict `evaluate_alignment` now requires: exact equality (same shape,
same dtype, no cast/re-interpolation/zeroing/rounding) for `gray`,
`instance_mask`, `polar_valid`, `signed_q`, `abs_q` and the model input
tensors; quality at atol 1e-6/rtol 0; logits at atol = rtol 1e-5;
non-finite values never pass; the expected sample set comes from the GT
correspondence of the verified manifest rather than from the number of rows
produced; duplicates, extras, unknown statuses, missing required checks,
empty checks and a recorded FAILED row all fail; missing coverage, a GT
without an NPZ and an unfinished required check are PARTIAL. The report
status and the exit code now come from that one verdict
(COMPLETED/exit 0, FAILED/non-zero, PARTIAL/non-zero), and the JSON,
Markdown and alignment sub-report agree.

Recheck over the historical rows: the successful attempt is PASSED with
368/368 expected samples covered and 0 mismatches; the preserved first
attempt is **FAILED** even though its recorded `summary.status` said
COMPLETED — that inconsistency is now stated rather than hidden. Three
items cannot be re-verified from the historical records (per-field shape
and dtype, exact-equality recomputation, and the fact that the recorded
verdicts predate the strict contract) and are reported as unverified, not
as newly measured passes.

### 11.4 The zero-overlap interpretation was wrong

With predictions present but every mask IoU zero, an `argmax` over the
all-zero row named the first prediction, and 15 such rows were counted as
"the best prediction is already matched to another GT" — the basis for the
earlier "20 merge/duplicate signals" statement, which is **retracted**.
A zero maximum overlap now keeps `max_mask_iou = 0.0` with a null best
index, a null matched-elsewhere flag and `no_positive_overlap = true`; a
frame with no predictions is reported separately with
`predictions_in_frame = 0` and a null IoU; only a strictly positive maximum
selects a best prediction (stable lowest-index tie-break). Recomputed from
the recorded rows: **22 unmatched GT, of which 16 have no positive overlap
at all and 5 have a positive overlap with a prediction already matched to
another GT** (one row has a positive overlap whose prediction was free).
All 15 unmatched real-fish instances have a maximum mask IoU of exactly
zero. No root cause is asserted.

### 11.5 The CPU stage name was inaccurate

`crop_preprocess_cpu` timed only the CPU resize/normalization of
**already cropped** windows, so it is renamed `cpu_tensor_preparation`
(with `renamed_from`, an explicit `excludes` list and a rename note). Every
timing value, the overall PARTIAL status and the NOT_COVERED /
NOT_MEASURED items are unchanged; this round ran no benchmark.

### 11.6 Verification and evidence

- Focused 896 passed; full suite 1594 passed, 1 skipped; `compileall` and
  `git diff --check` clean.
- One earlier test encoded the reviewed defect (it asserted
  `best_prediction_matched_elsewhere is True` for a zero-overlap row); its
  expectation was corrected to the contract above.
- `scripts/closeout_material_pipeline.py` re-derives the corrected reports
  from the recorded evidence only: it verifies the source manifests and
  required files (refusing escapes and digest mismatches), refuses an
  existing or identical output directory, builds no model and runs no
  forward, and writes `source_provenance.json`,
  `quality_summary_corrected.json`, `alignment_recheck.json`,
  `unmatched_gt_corrected.jsonl`, `performance_report_corrected.json`,
  `raw_timings_corrected.json`, `correction_summary.md` and
  `artifact_manifest.json`.
- Consistency checks confirm the 48 A2/B metric groups, the three
  bootstrap intervals, the per-instance predictions/matches and the eight
  performance timing arrays are unchanged (mismatches: NONE). The
  classification metrics, intervals and timings in this document are
  therefore the same numbers as in section 10.
- Environment note: `CUDA_VISIBLE_DEVICES=-1` (requested for this round)
  makes `torch.cuda.is_available()` raise an access violation on this
  torch 2.12.1+cu130 / Windows build, so the pytest runs were executed
  without that variable. The round still ran no GPU inference: the new
  tests use CPU stubs, and the closeout entry loads no model at all.

## 12. Closeout-entry CLI recovery (2026-09-29)

The closeout entry (`scripts/closeout_material_pipeline.py`) was repaired and
its acceptance path is now covered by synthetic CPU tests. This round ran
**no** real-evidence closeout, **no** full test suite, no model, no training
and no benchmark.

- The three missing helpers are restored with one contract each:
  `expected_keys_from_sources(gt_index, *, npz_keys) -> (keys, gt_without_npz)`,
  `verify_candidates(list, *, expected_shas)`, and
  `compare_source_and_corrected(source, corrected, ...)`.
- `verify_source` now treats each evidence sub-directory's manifest as
  covering its own data files (a manifest never lists itself), validates the
  preserved first attempt against its own manifest, and refuses empty
  listings, missing digests, wrong digests and escaping paths.
- Completeness is enforced against the admitted config: the 12 candidate
  identities must be present, each with four non-empty metric blocks
  (`layer_a2`/`layer_b` x `pure`/`routed`) — 48 compared blocks — plus the
  three named bootstrap seeds. A structure that is merely "missing on both
  sides" no longer passes, and an empty block is not a compared block.
- The post-hoc check re-reads the **files on disk**
  (`quality_summary_corrected.json`, `performance_report_corrected.json`,
  `raw_timings_corrected.json`) and compares them with the source, so a
  self-consistent but altered file is still a failure.
- Config/snapshot/data binding is checked (candidate identities and their
  digests, class order, material-model fields, training-data fingerprint and
  the V5 audit), with duplicate identities detected before any mapping is
  built. Fingerprints recorded by the source and digests computed now are
  reported separately and are not described as a fresh content re-audit.
- `gt_without_npz` now forces PARTIAL instead of being recorded and ignored.
- A legal synthetic fixture (`tests/material_fusion/pipeline_synthetic.py`)
  drives the real `main()`: COMPLETED/exit 0 on the clean case, FAILED/PARTIAL
  with non-zero exits for the single-factor corruptions, and the model
  factories are replaced by fail-if-called probes.
- Focused synthetic CPU results: CLI 14 passed; the four closeout-related
  test modules 108 passed; `compileall` and `git diff --check` clean.

Not done here: the real-evidence closeout run and any full-suite validation
remain separately authorized work.

## 13. Closeout contract fixes (2026-09-29)

Four acceptance gaps in the closeout entry were closed; the round ran only
synthetic CPU tests (no real-evidence closeout, no full suite, no model).

- **Data binding is real.** `audit_passed` must be boolean `True` (strings,
  integers and `None` are refused); the manifest's and summary's SHA256 are
  recomputed and compared with the audit's recorded digests; the V4 parent
  fingerprint must match the admitted config. The real V5 audit carries no
  `dataset_fingerprint` field, so that absence is reported explicitly rather
  than read as `None` and treated as a verified identity; no second
  fingerprint definition was introduced. The GT index is bound through the
  evidence manifest that owns it, and "a digest computed now" is reported
  separately from "an identity recorded by the source".
- **Structure, not just non-emptiness.** Every one of the 48 metric blocks
  must carry the production sections and fields (`counts`, `detection`,
  `conditional_classification`, `end_to_end`), with non-negative integer
  counts, consistent supports and finite rate values; each bootstrap seed
  must carry a real statistical result (`point_estimate`, `ci_low`,
  `ci_high`, a non-inverted interval) or an explicit not-estimable status.
  A block that is merely "missing on both sides" no longer passes, and it is
  not counted as a compared block.
- **Per-instance coverage feeds the verdict.** Records are checked per
  candidate, layer and condition: A2 must correspond exactly to the bound GT
  index (missing objects at best PARTIAL, extra objects FAILED), B must have
  records for every candidate and condition, duplicates and identity
  conflicts are failures, and missing evidence cannot be reported as "zero
  fallbacks, zero errors".
- **Post-write failure cannot stay COMPLETED.** The artifacts on disk are
  re-read and compared with the source; on a mismatch the report is
  re-issued as FAILED (JSON, Markdown and the manifest together) and the exit
  code is non-zero, so no published report can claim success.

Focused synthetic CPU results: CLI 14 passed; the four closeout-related test
modules 108 passed; `compileall` and `git diff --check` clean.

## 14. B instance-identity coverage (2026-09-29)

One confirmed vulnerability was closed: the B coverage check compared record
*counts* only, so replacing a record with `frame_id="nonexistent_frame"` or
`pred_instance_id=999` (keeping the count identical) still passed.

- The expected identity set now comes from the bound frame-level prediction
  evidence. The production entry numbers a frame's detections `0..predictions-1`
  (`eval_material_pipeline` enumerates the routed results), so the expected
  set is `{(frame_id, index)}` and it is never derived from the records
  under review.
- For every candidate, `B/pure` and `B/routed` are compared as identity sets:
  an unknown frame, an out-of-range or non-integer index, a duplicate
  identity inside the same candidate/condition, and any extra identity are
  FAILED; a missing predicted instance is at best PARTIAL. A frame whose
  evidence explicitly reports zero predictions legitimately has no records,
  while missing frame evidence still cannot complete.
- A contradictory record set (duplicates, missing fields, unknown identities)
  is now reported as FAILED instead of being raised out of the entry.

New real-`main()` cases (`TestBIdentityCoverage`, 6 parameterized cases over
pure/routed x unknown-frame/out-of-range/duplicate-identity) were RED before
the fix (all six passed through) and are GREEN after it. Focused synthetic
CPU result: 114 passed; `compileall` and `git diff --check` clean.

Still open (not addressed here): `frames.jsonl` is not yet required to be
covered by the source manifest; the frame evidence lacks structural/type and
duplicate-frame validation; `summary_supports` is not yet wired into `main()`;
the bootstrap not-estimable PARTIAL propagation, the `paired_changes` /
`conclusions` missing-both structure check, and the post-write fault
injection remain unimplemented. The overall 4B contract is therefore **not**
fully accepted.

## 15. B instance-id exception handling and record correction (2026-09-29)

One remaining production gap was closed and the previous round's record was
corrected.

- **Controlled refusal for an illegal B instance id.** A `B_model_a_predictions`
  record whose `pred_instance_id` is `None` reached `summarise_fallback` and
  raised `TypeError` (and a `dict`/`list` value raised an unhashable-type
  error), so `main()` could neither publish the agreed FAILED report nor
  reach the later type check. A validation pass now runs **before** the
  summary and the coverage check: the field must exist and be a real
  non-negative integer (`bool` is not an integer, and no `int()` coercion
  accepts a string or a float). A2 records legitimately carry
  `pred_instance_id=None` and are deliberately untouched.
- The refusal is a report, not a crash: `status=FAILED`, a non-zero return,
  the field-level reason in `instance_record_problem`, the Markdown stating
  FAILED, and no record silently dropped (the illegal set is never summarised
  as "zero fallbacks").
- New real-`main()` cases: `TestIllegalBInstanceId`, 14 parameterized cases
  (pure/routed x `None`, `True`, `0.0`, `"0"`, list, dict, missing field).
  RED before the fix (13 failed, including the `TypeError` escape); GREEN
  after it.
- **Record correction for the previous round** (`4b_b_identity_fix_20260929_065116_9c7378`):
  `pytest_green_02.log` was actually **19 passed, 1 failed** (not "CLI 20
  passed, exit 0"); the round's final passing evidence is
  `pytest_focused_02.log` = **114 passed**. The first two `duplicate_identity`
  cases did not initially construct a duplicate inside the same candidate and
  condition, and were corrected during debugging, so the initial six
  failures are **not** six independently reproduced vulnerabilities: the
  unknown-frame and out-of-range cases are valid RED evidence, while the
  duplicate cases were fixed at the test level. The historical logs were not
  rewritten.
- This round: focused synthetic CPU 128 passed; `compileall` and
  `git diff --check` clean.

Still open: `frames.jsonl` is not required to be covered by the source
manifest; frame-evidence structure/type/duplicate validation; `summary_supports`
not wired into `main()`; bootstrap not-estimable PARTIAL propagation;
`paired_changes`/`conclusions` missing-both check; post-write fault injection.
The overall 4B contract remains **not** fully accepted.

## 16. Frame-evidence source binding (2026-09-29)

One gap was closed: `main()` reads `quality/frames.jsonl` as the basis for
the expected B instance identities, but source admission did not require that
file to be covered by the quality manifest, so an unlisted file could still
be consumed.

- `quality/frames.jsonl` is now a required source file and is part of the
  set the quality manifest must cover, reusing the existing path, SHA256 and
  manifest validation. A file that exists but is not listed, a listed file
  whose digest does not match, and a missing frame-evidence file are all
  refused **before** its content is read; there is no silent fallback to
  "use whatever is on disk".
- The synthetic helper `perturb_alignment_row()` rebuilt the quality manifest
  without `frames.jsonl`; it now keeps the complete coverage of the legal
  fixture, so the alignment FAILED/PARTIAL cases still reach their target
  branch instead of failing on a fixture-induced digest error. Production
  admission was not relaxed for the fixture.
- New real-`main()` cases: `TestFrameEvidenceBinding` (uncovered entry,
  missing file, wrong digest). The uncovered-entry case was RED before the
  fix (`DID NOT RAISE`) and is GREEN after it; the wrong-digest case was
  already refused and is recorded as a regression test, not as a fix.
- This round: CLI 37 passed; four-module focused 131 passed; `compileall` and
  `git diff --check` clean.

Still open (unchanged): frame-evidence internal structure/type/duplicate-frame
validation and complete frame coverage; `summary_supports` not wired into
`main()`; bootstrap not-estimable PARTIAL propagation; `paired_changes` /
`conclusions` missing-both check; general post-write fault injection and final
status publication. The overall 4B contract remains **not** fully accepted.

## 17. Frame-evidence binding: entry-level acceptance (2026-09-29)

Test-level completion of the `frames.jsonl` binding, plus a correction of how
the previous round's evidence was described. **Production code was not
changed in this round.**

- The previous round's three frame-binding tests called
  `verify_source()` directly, so they were *helper-level* evidence — the
  report's wording ("three new real `main()` cases") was inaccurate. Its RED
  evidence (`DID NOT RAISE` for the uncovered entry) is a `verify_source()`
  level observation.
- This round replaces them with four scenarios that all execute
  `scripts.closeout_material_pipeline.main()` on the complete legal synthetic
  fixture, each changing exactly one factor and using its own output
  directory:
  1. file present, manifest entry removed → `ValueError`, message names
     `frames.jsonl` and "not covered";
  2. file removed, entry kept → `FileNotFoundError`, message names
     `frames.jsonl` and "missing";
  3. file and entry both removed → `FileNotFoundError` for the missing
     required source (not merely "a recorded file is absent");
  4. file present, digest well-formed but wrong → `ValueError`, message names
     `frames.jsonl` and "digest mismatch".
- Because the production fix already exists, these entry tests are GREEN on
  first run; **no RED was manufactured by reverting code**. They are reported
  as entry-level acceptance of an existing fix, not as a new reproduction.
- `main()` raising the expected exception is reported as "the entry raised
  the expected exception" — no independent process exit code was measured
  for these refusals.
- Results: the four scenarios 4 passed; full CLI 38 passed; four-module
  focused 132 passed; `compileall` and `git diff --check` clean.

Still open (unchanged, not addressed here): frame-evidence internal
structure/type/duplicate-frame validation and complete frame coverage;
`summary_supports` wiring; bootstrap not-estimable PARTIAL propagation;
`paired_changes`/`conclusions` missing-both check; general post-write fault
injection and final status publication. The overall 4B contract remains
**not** fully accepted.

## 18. Frame-evidence schema and complete frame coverage (2026-09-29)

The frame evidence (`quality/frames.jsonl`) is now validated before it is
turned into a prediction map, so an illegal count, a repeated frame, a frame
outside the frozen scope or a missing frame cannot be laundered into a
"complete" B coverage.

- **Production format (verified read-only).** `eval_material_pipeline` writes
  one record per evaluated frame with a non-empty string `frame_id` (from the
  authoritative split list) and an integer `predictions = len(b_instances)`;
  frames whose instance list is empty are still written, with
  `predictions = 0`. The frozen evaluation scope is therefore the GT index's
  frame list, and the expected frame set is taken from that bound index —
  never from `frames.jsonl` itself and never from the instance records.
- **Removed pre-fix behaviour (reproduced in isolation for the record).**
  The previous mapping used `str(row.get("frame_id"))` and
  `int(row["predictions"])` and overwrote duplicates in a dictionary:
  `123 -> "123"`, `"3" -> 3`, `True -> 1`, a repeated `frame_a` silently
  resolved to the last value, and a frame absent from the file was simply
  absent from the map — i.e. treated as "no predictions" rather than as a
  coverage gap. (`prefix_logic_demo.txt` in the round's evidence directory.)
- **New validation.** `validate_frame_evidence()` requires: each record to be
  a JSON object; a non-empty string `frame_id` with no `str()`/`strip()`
  repair; `predictions` present with `type(value) is int` and `value >= 0`
  (bool, float, string, `None`, list and dict are all refused, with no
  `int()` coercion); no duplicate frame (identical or with a different
  count); and the actual frame set to match the expected set. An unknown or
  extra frame, a duplicate, an illegal identity or an illegal count is
  FAILED; a pure coverage gap is PARTIAL; a contradiction outranks a gap. A
  frame whose evidence explicitly reports zero predictions is a legal frame
  that must have no B instance records, and a record on such a frame is
  FAILED with its own message; a missing frame is never read as zero
  predictions.
- **Entry-level cases.** `TestFrameEvidenceSchema` runs the real `main()`:
  the legal fixture (which now includes a zero-prediction frame whose GT
  instance list is empty) still completes; a zero-prediction frame carrying B
  records fails; a deleted frame record and an empty `frames.jsonl` are
  PARTIAL; an unknown frame replacing a legal one, an extra unknown frame,
  an exact duplicate, a duplicate with a different count, ten illegal
  `predictions` values, six illegal `frame_id` values and a non-object record
  all fail with the target reason and a consistent JSON/Markdown status.
- Results: schema cases 25 passed; full CLI 63 passed; four-module focused
  157 passed; `compileall` and `git diff --check` clean. The pre-fix
  behaviour was confirmed by reading the removed code and reproduced in
  isolation; the new cases were written together with the fix, so they are
  entry-level acceptance of it rather than a captured RED.

Still open (not addressed here): `summary_supports` wiring into `main()`;
bootstrap not-estimable PARTIAL propagation; `paired_changes` /
`conclusions` missing-both check; general post-write fault injection and
final status publication. The overall 4B contract remains **not** fully
accepted.

## 19. Missing frame evidence: unverifiable, not contradictory (2026-09-29)

A frame whose prediction evidence is missing has an **unknown** prediction
count, so its instances must not be judged against a count that was never
recorded.

- **The defect.** `verify_instance_coverage()` built its expected identity
  set only from the frames that *had* evidence, so the legal B records of a
  frame whose record was deleted were reported as "unknown or
  out-of-range" and the run failed — a missing count was misread as a wrong
  identity. Deleting a zero-prediction frame behaved differently again.
- **The fix.** `main()` now passes the independent expected frame set (from
  the bound GT index) into the coverage check, which distinguishes three
  cases per candidate and condition:
  * a frame with legal evidence is range-checked as before (a record on an
    explicit zero-prediction frame, an out-of-range or negative id, a
    duplicate identity are FAILED; a missing predicted instance is PARTIAL);
  * a frame inside the frozen scope whose evidence is missing is
    **unverifiable** — PARTIAL, never zero and never "wrong identity" — while
    count-independent contradictions on that frame (illegal or negative ids,
    duplicate identities) are still FAILED;
  * records on a frame outside the frozen scope are FAILED, and that FAILED
    is not downgraded by any other missing frame.
  The prediction count of a missing frame is never inferred from the largest
  observed index.
- **State priority**: a definite contradiction (FAILED) outranks a coverage
  gap (PARTIAL) outranks a complete pass (PASSED); when frame evidence is
  missing, `instance_coverage` itself can no longer report PASSED.
- **Entry-level cases.** `TestMissingFrameEvidence` (12 cases) runs the real
  `main()`: dropping a frame that has predictions is PARTIAL with no
  "unknown/out-of-range" claim; dropping the zero-prediction frame and
  deleting a frame's records are PARTIAL; an out-of-range id on another
  frame, a duplicate identity on the missing frame, an illegal or negative
  id, and records on an unknown frame are FAILED with the target reason; an
  empty `frames.jsonl` with legal records is PARTIAL; the complete legal
  fixture still completes. RED was captured before the fix (5 of the 12
  failed, including the misjudgement above).
- **Record correction for the previous round**
  (`4b_frame_schema_20260929_075407_03ffe2`): `pytest_schema_01.log` was
  actually **1 failed, 24 passed, 38 deselected** — not "25 passed"; the
  round's final passing evidence is `pytest_cli_03.log` = **63 passed** and
  `pytest_focused_02.log` = **157 passed**. No independent RED was captured
  for that round's schema cases, as stated there. The historical logs were
  not modified.
- This round: missing-frame cases 12 passed; full CLI 75 passed; four-module
  focused 169 passed; `compileall` and `git diff --check` clean.

Still open (not addressed here): `summary_supports` wiring; bootstrap
not-estimable PARTIAL propagation; `paired_changes`/`conclusions`
missing-both check; general post-write fault injection and final status
publication. The overall 4B contract remains **not** fully accepted.

## 20. Four closeout blockers repaired (2026-09-29, explicitly authorized Codex edit)

This supersedes section 19's list of four open code contracts, not its
historical results. No real evidence closeout, inference or training ran.

- `main()` passes each candidate's B/pure and B/routed prediction support
  separately to coverage verification. Supports must be non-negative integers
  (not bool/float/string). With complete independent frame coverage, they must
  equal the frame prediction total; instance identities are checked separately.
  Incomplete frame evidence makes this comparison PARTIAL, not a subtotal
  mismatch. This does not recompute all classification/detection statistics.
- Explicit bootstrap UNVERIFIED / NOT_MEASURED / NOT_ESTIMABLE states propagate
  through consistency checks to a PARTIAL top-level report and nonzero return.
  Contradictions retain FAILED priority. Recorded bootstrap data is not changed.
- Both source and corrected reports must contain a non-empty paired-result
  object list and a non-empty named conclusion mapping. Missing/empty/invalid
  sections cannot pass by agreeing with an equally missing corrected section.
- Publication freezes the intended bytes before writing, verifies every JSON,
  JSONL and Markdown artifact plus the independently computed manifest, and
  announces status only afterwards. A failure triggers one bounded rewrite as
  FAILED and the same verification again. Persistent failure invalidates only
  the new output's quality summary, Markdown and manifest, and attempts to leave
  `publication_failed.json`; the return is nonzero, never success. An underlying
  storage failure may prevent cleanup or the marker too; stderr remains explicit.
- Serialization fault tests cover metrics, bootstrap, timing arrays, performance
  statistics, invalid JSON, Markdown, missing manifest coverage, mutation of the
  manifest argument, persistent corruption and an I/O exception. Model factory
  probes remain enabled in every synthetic entry test.

Validation: first RED 31 failed / 4 passed; initial GREEN 35 passed. An additional
manifest-argument mutation test exposed baseline aliasing (1 failed / 1 passed
with the Markdown regression); isolating that argument fixed it. Final focused
suite: **215 passed** (the original four modules plus 37 new entry tests).
The previous helper-only support fix's nine tests remain included.
Evidence: `analysis/runs/material_runtime_v2_mask/codex_4b_blockers_20260929_c93e61/`.
Coverage NOT_MEASURED (no coverage package installed). Tests are CPU-isolated.

Still pending: real historical evidence admission/closeout and broader validation
under separate authorization. These engineering tests do not establish material
classification benefit, independent generalization, real-time speed or deployment
readiness. No formal configuration, dataset, checkpoint or historical evidence
was modified; no Git state-changing commands were executed.
