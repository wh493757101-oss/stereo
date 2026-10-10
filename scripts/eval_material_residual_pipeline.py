"""Two-layer evaluation of the locked residual candidate under real Model
A predictions (material_residual_runtime_v1).

Layer A replays the candidate's own offline evaluation basis (the full
2446-sample dev pool through the per-fold protocol) through the runtime
adapter and compares every stage against the saved evidence: the gray
feature G, the local C14/S16 statistics, the frozen P0 logits, the final
corrected logits and the missing-S fallback. A stage difference beyond
the pre-fixed tolerances ABORTS the formal quality statistics.

Layer B runs the production pipeline (Model A at the fixed operating
point, band SGBM, per-instance polar features, the residual adapter) on
the 180 val frames. Detection support/misses/false positives, conditional
classification on matched instances, end-to-end metrics including missed
and unmatched predictions, the paired P0-vs-candidate differences,
per-class recall and confusions, per-group and quality-strata breakdowns,
fallback reasons and every per-instance record are reported. Model A's
training data shares its capture sources with the classifier data, so
this is NOT an independent blind test and is never described as one.

The benchmark mode is a bounded, pre-frozen timing round over preloaded
already-rectified frames: selected frames, warmup, repeats and the wall
budget are pinned in the config before any run; incomplete timing is
reported PARTIAL and is never extrapolated into an FPS claim.

No training, no deployment, no checkpoint writes; the candidate is the
COMPLETED fit manual_20261009T185911 and nothing here re-selects a
feature, epoch or seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from core.material_probe import (  # noqa: E402
    authoritative_sample_index,
)
from core.polar_local_features import (  # noqa: E402
    LOCAL_C_WIDTH,
    LOCAL_S_WIDTH,
    verify_local_cache,
)
from models.material_classifier import (  # noqa: E402
    MaterialInstanceInput,
    MASK_SOURCE_NPZ,
)
from models.material_residual_classifier import (  # noqa: E402
    CLASS_NAMES,
    build_frame_fold_map,
    load_residual_runtime_config,
)
from models.polar_fusion import read_manifest_split  # noqa: E402
from scripts.eval_material_pipeline import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MATCH_RULE,
    OPERATING_POINT,
    POLAR_QUALITY_BINS,
    _build_record,
    aggregate_metrics,
    end_to_end_macro_f1_bootstrap,
    instance_correctness_delta_bootstrap,
    match_instances,
    paired_changes,
    paired_cluster_bootstrap,
    prepare_output_dir,
    quality_strata,
    sha256_of,
    summarise_fallback,
    unmatched_gt_records,
    unmatched_prediction_records,
    write_json_bytes,
    write_jsonl_bytes,
)
from scripts.verify_material_runtime import right_image_path  # noqa: E402

VARIANTS = ("R0", "RC", "RCS")
EVIDENCE_VARIANTS = ("R0", "RCS")
MASK_SOURCE_GT_EVAL = "ground_truth_controlled_eval"


# ---------------------------------------------------------------------------
# Stage comparison and the Layer A verdict

STAGE_FINITE = "finite"   # G, C14, P0/final logits: every value must be finite
STAGE_S16 = "s16"         # raw local S16: NaN legal ONLY as a full 8-wide
S_BLOCK_WIDTH = 8         # missing block admitted by the frozen region state


def compare_stage_arrays(actual: np.ndarray | None,
                         expected: np.ndarray | None, *,
                         atol: float, rtol: float,
                         stage: str = STAGE_FINITE) -> dict:
    """One stage comparison under the pre-fixed tolerance contract.

    Contract (fixed before the evaluation ran; never widened):

    - shape must match and be non-empty;
    - +Inf/-Inf/NaN are refused on BOTH sides for finite stages
      (``g``/``c14``/``logits``/``p0``) — ``Inf == Inf`` is not equality
      and a NaN can never silently equal another NaN;
    - for ``s16`` the expected side defines the ADMITTED missing state:
      a block of ``S_BLOCK_WIDTH`` values is either entirely finite
      (compared exactly) or entirely NaN (the frozen explicit-miss rule);
      a partial-NaN block, a one-sided missing block or an Inf inside a
      finite block is a failure;
    - a failure report is JSON-safe: ``max_abs_error`` is null with an
      explicit reason whenever a meaningful error cannot be computed —
      never NaN/Inf, and no ``nan_to_num`` masking anywhere.
    """
    def _fail(reason: str) -> dict:
        return {"passed": False, "max_abs_error": None, "status": "FAILED",
                "reason": reason}

    if actual is None or expected is None:
        return _fail("missing array")
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return _fail(f"shape {actual.shape} != {expected.shape}")
    if actual.size == 0:
        return _fail("empty array")
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    actual_nonfinite = ~np.isfinite(actual64)
    expected_nonfinite = ~np.isfinite(expected64)

    if stage == STAGE_S16:
        blocks = actual64.size // S_BLOCK_WIDTH
        reshaped_a = actual64.reshape(blocks, S_BLOCK_WIDTH) \
            if actual64.ndim == 1 else None
        reshaped_e = expected64.reshape(blocks, S_BLOCK_WIDTH) \
            if expected64.ndim == 1 else None
        if reshaped_a is None:
            return _fail("s16 comparison expects a flat 16-wide row")
        for block in range(blocks):
            a_block, e_block = reshaped_a[block], reshaped_e[block]
            a_all_nan = bool(np.isnan(a_block).all())
            e_all_nan = bool(np.isnan(e_block).all())
            if np.isinf(a_block).any() or np.isinf(e_block).any():
                return _fail(f"s16 block {block} contains Inf")
            if a_block.size and bool(np.isnan(a_block).any()) and \
                    not a_all_nan:
                return _fail(f"s16 actual block {block} has partial NaN")
            if e_block.size and bool(np.isnan(e_block).any()) and \
                    not e_all_nan:
                return _fail(f"s16 expected block {block} has partial NaN")
            if e_all_nan != a_all_nan:
                return _fail(
                    f"s16 block {block}: missing state disagrees "
                    f"(expected all-NaN={e_all_nan}, actual all-NaN="
                    f"{a_all_nan})")
        comparable = ~(np.isnan(actual64) | np.isnan(expected64))
        if not comparable.any():
            return {"passed": True, "max_abs_error": None,
                    "status": "PASSED",
                    "note": "all blocks are legal missing blocks"}
    else:
        if actual_nonfinite.any() or expected_nonfinite.any():
            which = "actual" if actual_nonfinite.any() else "expected"
            return _fail(f"{stage}: {which} side contains non-finite values "
                         "(NaN/Inf are refused on both sides)")
        comparable = np.ones(actual64.shape, dtype=bool)

    diff = np.abs(actual64 - expected64)
    scale = np.maximum(np.abs(expected64), 1e-12)
    exceeded = (diff > float(atol) + float(rtol) * scale) & comparable
    if not bool(exceeded.any()):
        finite_diffs = diff[comparable]
        max_abs = float(finite_diffs.max()) if finite_diffs.size else None
        return {"passed": True, "max_abs_error": max_abs, "status": "PASSED"}
    worst = int(np.argmax(np.where(comparable, diff, -np.inf).ravel()))
    return {"passed": False, "max_abs_error": float(diff.ravel()[worst]),
            "status": "FAILED",
            "reason": "value difference beyond the pre-fixed tolerance"}


REQUIRED_STAGE_ALL = "all_compared_stages"


def layer_a_verdict(rows: Sequence[dict], *, expected_names: Sequence[str],
                    expected_candidates: Sequence[str],
                    required_stages: dict[str, Sequence[str]]) -> dict:
    """The Layer A verdict against the AUTHORITATIVE expectations.

    ``expected_names`` comes from the frozen protocol + manifest and
    ``expected_candidates`` from the frozen candidate registry — never
    from the rows being validated. The verdict is OK only when every
    expected sample appears exactly once for every expected candidate,
    no unknown candidate appears, every required stage is present and
    passed, and no prediction mismatched. Anything else fails and aborts
    Layer B.
    """
    failures: list[dict] = []
    stage_failures: dict[str, int] = {}
    prediction_mismatches = 0
    seen: dict[tuple[str, str], int] = {}
    candidates_seen: set[str] = set()
    names_seen: set[str] = set()
    invalid_s_valid_types = 0
    s_valid_disagreements = 0
    missing_fallback_evidence = 0
    illegal_stage_reports = 0
    for row in rows:
        name = row.get("sample_name")
        candidate = row.get("candidate")
        candidates_seen.add(candidate)
        names_seen.add(name)
        key = (name, candidate)
        seen[key] = seen.get(key, 0) + 1
        if row.get("prediction_mismatch"):
            prediction_mismatches += 1
        # the s_valid contract: strict booleans on both sides, and they
        # must agree — a runtime/offline disagreement is a failure even
        # when every numeric stage passed
        runtime_valid = row.get("s_valid_runtime")
        offline_valid = row.get("s_valid_offline")
        if not isinstance(runtime_valid, bool) or \
                not isinstance(offline_valid, bool):
            invalid_s_valid_types += 1
            if len(failures) < 20:
                failures.append({"sample": name, "candidate": candidate,
                                 "reason": "s_valid fields are not strict "
                                           "booleans"})
        elif runtime_valid != offline_valid:
            s_valid_disagreements += 1
            if len(failures) < 20:
                failures.append({"sample": name, "candidate": candidate,
                                 "reason": "s_valid_runtime disagrees with "
                                           "s_valid_offline"})
        # the missing-S fallback evidence: a row whose admitted offline
        # state is missing must carry ACTUAL exact-fallback evidence
        # (fallback_exact_p0 is True). This applies to every candidate:
        # for the R0/P0 family the check is trivially satisfied because
        # R0 IS the shared frozen P0 — the distinction is that R0 never
        # "falls back" (it is the fallback), while correction candidates
        # gate their delta to exactly zero.
        if isinstance(offline_valid, bool) and offline_valid is False:
            if row.get("fallback_exact_p0") is not True:
                missing_fallback_evidence += 1
                if len(failures) < 20:
                    failures.append({"sample": name, "candidate": candidate,
                                     "reason": "a missing-S row without "
                                               "fallback_exact_p0 is True "
                                               "evidence"})
        reports = row.get("stage_reports")
        if not isinstance(reports, dict) or not reports:
            stage_failures[f"{candidate}:stage_reports"] = \
                stage_failures.get(f"{candidate}:stage_reports", 0) + 1
            failures.append({"sample": name, "candidate": candidate,
                             "reason": "missing stage_reports"})
            continue
        # EVERY present stage report must obey its pass contract — an
        # extra failed stage (e.g. fallback) can never be ignored, and an
        # illegal or self-contradictory report (passed not a strict bool,
        # status outside PASSED/FAILED, or passed disagreeing with
        # status) fails the row
        required = required_stages.get(candidate, ())
        for stage_name in required:
            if stage_name not in reports:
                stage_failures[f"{candidate}:{stage_name}:missing"] = \
                    stage_failures.get(
                        f"{candidate}:{stage_name}:missing", 0) + 1
                if len(failures) < 20:
                    failures.append({"sample": name, "candidate": candidate,
                                     "reason": f"missing stage {stage_name}"})
        for stage_name, report in reports.items():
            if not isinstance(report, dict):
                illegal_stage_reports += 1
                if len(failures) < 20:
                    failures.append({"sample": name, "candidate": candidate,
                                     "stage": stage_name,
                                     "reason": "stage report is not a "
                                               "mapping"})
                continue
            passed = report.get("passed")
            status = report.get("status")
            contradictory = (
                not isinstance(passed, bool)
                or ("status" in report
                    and status not in ("PASSED", "FAILED"))
                or ("status" in report
                    and passed is not (status == "PASSED")))
            if contradictory:
                illegal_stage_reports += 1
                if len(failures) < 20:
                    failures.append({"sample": name, "candidate": candidate,
                                     "stage": stage_name,
                                     "reason": "illegal or self-"
                                               "contradictory stage report"})
                continue
            if passed is False:
                stage_failures[f"{candidate}:{stage_name}"] = \
                    stage_failures.get(f"{candidate}:{stage_name}", 0) + 1
                if len(failures) < 20:
                    failures.append({"sample": name, "candidate": candidate,
                                     "stage": stage_name,
                                     "report": report})
    duplicate = {f"{k[0]}|{k[1]}": v for k, v in seen.items() if v > 1}
    missing_pairs = [(name, candidate)
                     for name in expected_names
                     for candidate in expected_candidates
                     if (name, candidate) not in seen]
    unknown_candidates = sorted(candidates_seen - set(expected_candidates))
    unknown_samples = sorted(names_seen - set(expected_names))
    problems = {
        "duplicate_sample_candidate_pairs": duplicate,
        "missing_sample_candidate_pairs": len(missing_pairs),
        "missing_pair_examples": [
            {"sample": n, "candidate": c}
            for n, c in missing_pairs[:5]],
        "unknown_candidates": unknown_candidates,
        "unknown_samples": unknown_samples[:5],
        "stage_failures": stage_failures,
        "prediction_mismatches": prediction_mismatches,
        "invalid_s_valid_types": invalid_s_valid_types,
        "s_valid_disagreements": s_valid_disagreements,
        "missing_fallback_evidence": missing_fallback_evidence,
        "illegal_stage_reports": illegal_stage_reports,
    }
    status = "OK" if not any(problems.values()) else "FAILED"
    return {
        "status": status,
        "samples_expected": len(expected_names),
        "candidates_expected": list(expected_candidates),
        "rows": len(rows),
        "problems": problems,
        "prediction_mismatches": prediction_mismatches,
        "first_failures": failures,
        "abort_layer_b": status == "FAILED",
    }


def freeze_benchmark_frames(stems: Sequence[str], count: int) -> list[str]:
    """The frozen benchmark frame list: evenly spread unique stems in
    sorted order, decided before any model output exists."""
    stems = sorted(set(stems))
    if count <= 0 or count > len(stems):
        raise ValueError(
            f"cannot freeze {count} benchmark frames from {len(stems)} "
            "unique val frames")
    if count == 1:
        return [stems[0]]
    indices = np.linspace(0, len(stems) - 1, count).round().astype(int)
    return [stems[int(i)] for i in dict.fromkeys(indices)]


class BudgetExceeded(RuntimeError):
    """The wall-clock budget is exhausted. ``watch_scope`` identifies
    WHICH watch fired (structurally, not via string matching): callers
    distinguish the benchmark's own budget from the whole-evaluation
    budget by reading this field, never by parsing the message."""

    def __init__(self, message: str, watch_scope: str = ""):
        super().__init__(message)
        self.watch_scope = watch_scope


class BudgetWatch:
    """A wall-clock budget checked at bounded-operation boundaries.

    ``clock`` is an injectable monotonic-seconds callable (tests use a
    fake clock; production uses ``time.perf_counter``). One watch covers
    the WHOLE evaluation — it is never reset per stage. A check can only
    fire at the boundary it guards; no per-kernel interruption is
    claimed.
    """

    def __init__(self, *, budget_minutes: float, clock=None,
                 scope: str = "whole-evaluation budget"):
        self.scope = scope
        value = float(budget_minutes)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"budget_minutes must be a finite positive number, got "
                f"{budget_minutes!r}")
        self.budget_s = value * 60.0
        self.clock = clock if clock is not None else time.perf_counter
        self.started = self.clock()

    def elapsed_s(self) -> float:
        return self.clock() - self.started

    def check(self, stage: str) -> None:
        if self.elapsed_s() > self.budget_s:
            raise BudgetExceeded(
                f"the {self.scope} "
                f"({self.budget_s / 60.0:.1f} min) is exhausted at {stage}",
                watch_scope=self.scope)


_MISSING = object()


def _clock_factory():
    """The monotonic-clock seam for the whole-evaluation budget (tests
    inject a fake clock; production returns None -> time.perf_counter)."""
    return None


def _strict_budget_number(value: Any, field: str) -> float:
    """A budget number must be a non-bool int/float, finite and positive.

    ``type(x) is int``/``type(x) is float`` semantics: ``bool`` is
    refused even though it subclasses int, and a same-value float never
    substitutes for a required int. No ``float()`` coercion happens — an
    illegal type is refused, not converted.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field} must be an int or float (not bool/str/None), got "
            f"{value!r} of type {type(value).__name__}")
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(
            f"{field} must be finite and positive, got {value!r}")
    return number


def resolve_evaluation_budget(config, cli_value) -> float:
    """The whole-evaluation budget: an explicit legal CLI override wins;
    otherwise the config's ``evaluation_budget.whole_run_minutes``; the
    entry's declared default is 30 minutes. The benchmark's own 15-minute
    budget is NEVER the whole-evaluation budget. bool/str/None/
    non-finite/non-positive values are refused without coercion."""
    if cli_value is not None:
        return _strict_budget_number(cli_value, "--budget-minutes")
    if isinstance(config, dict):
        declared_value = (config.get("evaluation_budget") or {}).get(
            "whole_run_minutes", _MISSING)
    else:
        declared_value = (config.evaluation_budget or {}).get(
            "whole_run_minutes", _MISSING)
    if declared_value is not _MISSING:
        # the key is PRESENT: an illegal value (bool/str/None/non-finite/
        # non-positive) is refused — never silently defaulted
        return _strict_budget_number(declared_value,
                                     "evaluation_budget."
                                     "whole_run_minutes")
    return 30.0


# ---------------------------------------------------------------------------
# Shared setup


def load_protocol(config):
    from core.material_probe import file_sha256

    protocol_path = PROJECT_ROOT / config.protocol_path
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    digest = file_sha256(protocol_path)
    if digest != config.protocol_sha256:
        raise ValueError(
            f"protocol sha256 {digest} != the pinned "
            f"{config.protocol_sha256}")
    return protocol


def _local_cache_path(config) -> str:
    for ref in config.local_cache:
        if ref.path.endswith("cache_binding.json"):
            return str(Path(ref.path).parent)
    raise ValueError("the local cache binding is not pinned")


def npz_to_instance_input(sample, *, frame_id: str, instance_id: int) -> \
        MaterialInstanceInput:
    """A clearly-labelled offline-identity record from one V5 npz sample."""
    return MaterialInstanceInput(
        frame_id=frame_id, instance_id=instance_id, bbox=(0, 0, 0, 0),
        gray=sample.gray, signed_q=sample.signed_q, abs_q=sample.abs_q,
        polar_valid=sample.valid, instance_mask=sample.instance_mask,
        quality=sample.quality, polar_invalid=False, invalid_reason="",
        mask_source=MASK_SOURCE_NPZ)


def build_classifiers(config, protocol, *, device: str, frame_folds,
                      encoder=None):
    """One classifier per frozen registry candidate, sharing one encoder.

    With the complete config the registry is the fixed 9-candidate matrix
    (R0 + linear_local RC/RCS s2026 + conditional RC/RCS s2026/2027/2028);
    with the original config (no registry) the legacy single-seed
    conditional trio is built. The encoder is built once and shared;
    standardized inputs are never shared.
    """
    from models.material_residual_classifier import MaterialResidualClassifier

    if config.candidates:
        specs = list(config.candidates)
    else:
        specs = [None, None, None]  # legacy trio: R0/RC/RCS at s2026
    classifiers: dict[str, Any] = {}
    encoder_builder = None
    for position, spec in enumerate(specs):
        if spec is None:
            variant = ("R0", "RC", "RCS")[position]
            candidate_id = None
        else:
            variant = spec.variant
            candidate_id = spec.candidate_id
        classifier = MaterialResidualClassifier(
            config, variant=variant, device=device,
            frame_folds=frame_folds, encoder=encoder,
            candidate=candidate_id, encoder_builder=encoder_builder)
        if encoder is None:
            encoder = classifier.encoder
            encoder_builder = _shared_builder_hook(encoder)
        classifiers[classifier.candidate_spec.candidate_id] = classifier
    return classifiers


def _shared_builder_hook(encoder):
    """Return a builder that always hands back the already-verified shared
    encoder (the cache is keyed, so same-identity requests hit it)."""

    def _build(config, device):
        return encoder

    return _build


def _read_saved_predictions(config, classifiers):
    """Each candidate's saved outer-eval oracle, keyed by sample_name.

    R0's oracle is the fit's own R0 record (``r0_records``); every
    correction candidate uses the saved predictions pinned in its own
    registry entry.
    """
    saved: dict[str, dict[str, dict[str, dict]]] = {}
    for candidate_id, classifier in classifiers.items():
        spec = classifier.candidate_spec
        per_fold = {}
        if spec.family == "p0":
            for fold, ref in config.r0_records.items():
                path = PROJECT_ROOT / ref.path
                per_fold[fold] = {
                    record["sample_name"]: record
                    for record in _read_jsonl(path)}
        else:
            for fold, files in spec.fold_files.items():
                path = PROJECT_ROOT / files["saved_predictions"].path
                per_fold[fold] = {
                    record["sample_name"]: record
                    for record in _read_jsonl(path)}
        saved[candidate_id] = per_fold
    return saved


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_layer_a(config, protocol, *, device: str, output_dir: Path,
                classifiers, max_samples: int | None = None,
                watch: "BudgetWatch | None" = None) -> dict:
    """Replay EVERY candidate's offline evaluation basis through the
    runtime adapter, stage by stage, against each candidate's OWN saved
    outer-eval oracle.

    Every dev-pool sample is outer_eval of exactly one fold (2446 rows).
    The expected sample set comes from the frozen protocol/manifest and
    the expected candidate set from the frozen registry — never from the
    rows being validated. A stage difference beyond the pre-fixed
    tolerances, any missing/duplicate/unknown identity, or any prediction
    mismatch aborts the formal quality statistics (Layer B never runs).
    ``max_samples`` is an engineering guard for bounded smoke runs only.
    """
    from core.material_mask_dataset import load_mask_sample

    local_dir = PROJECT_ROOT / _local_cache_path(config)
    report = verify_local_cache(local_dir, protocol)
    manifest = json.loads((local_dir / "feature_manifest.json").read_text(
        encoding="utf-8"))
    cache = dict(np.load(local_dir / "feature_cache.npz"))
    samples = manifest["samples"]
    authority = authoritative_sample_index(protocol)
    training_root = PROJECT_ROOT / (config.data_identity or {}).get(
        "training_data_root",
        "datasets/underwater_cls_fusion_v5_instance_mask")

    tol = config.tolerances
    g_atol, g_rtol = float(tol["g_feature_atol"]), float(tol["g_feature_rtol"])
    cs_atol = float(tol["c_s_atol"])
    logit_atol = float(tol["logit_atol"])

    # the fold of each row: outer_eval of exactly one fold
    row_fold = {}
    for name, entry in authority.items():
        outer = [fold for fold, role in entry["roles"].items()
                 if role == "outer_eval"]
        assert len(outer) == 1
        row_fold[name] = outer[0]

    candidate_ids = sorted(classifiers)
    required_stages = {candidate_id: ("g", "c14", "s16", "p0", "final")
                       for candidate_id in candidate_ids}
    saved = _read_saved_predictions(config, classifiers)
    expected_names = [row["sample_name"] for row in samples]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("the local cache manifest contains duplicate "
                         "sample names")

    r0_classifier = classifiers[
        next(cid for cid, clf in classifiers.items()
             if clf.candidate_spec.family == "p0")]
    rows: list[dict] = []
    if max_samples is not None:
        selected_indices = list(range(min(max_samples, len(samples))))
    else:
        selected_indices = list(range(len(samples)))
    batch: list = []
    batch_rows: list[int] = []
    batch_positions: list[int] = []

    def _flush() -> None:
        if watch is not None:
            watch.check("layer_a batch")
        if not batch:
            return
        gray = r0_classifier.gray_features(batch)
        for position, (record_input, row_index) in enumerate(
                zip(batch, batch_rows)):
            row = samples[row_index]
            name = row["sample_name"]
            fold = row_fold[name]
            local = r0_classifier.local_features(record_input)
            p0_logits = r0_classifier.p0_logits_for(
                record_input, fold, gray=gray[position])
            for candidate_id in candidate_ids:
                classifier = classifiers[candidate_id]
                spec = classifier.candidate_spec
                saved_record = saved[candidate_id][fold].get(name)
                stage_reports = {
                    "g": compare_stage_arrays(
                        gray[position], cache["g_features"][row_index],
                        atol=g_atol, rtol=g_rtol, stage=STAGE_FINITE),
                    "c14": compare_stage_arrays(
                        local["c_features"], cache["c_features"][row_index],
                        atol=cs_atol, rtol=0.0, stage=STAGE_FINITE),
                    "s16": compare_stage_arrays(
                        local["s_features"], cache["s_features"][row_index],
                        atol=cs_atol, rtol=0.0, stage=STAGE_S16),
                }
                entry = {
                    "sample_name": name, "candidate": candidate_id,
                    "fold": fold, "row_index": row_index,
                    "s_valid_runtime": bool(local["s_valid"]),
                    "s_valid_offline": bool(cache["s_valid"][row_index]),
                    "stage_reports": stage_reports,
                    "prediction_mismatch": False,
                }
                if saved_record is None:
                    stage_reports["oracle"] = {
                        "passed": False, "max_abs_error": None,
                        "status": "FAILED",
                        "reason": "the saved outer-eval oracle has no row "
                                  "for this sample"}
                    rows.append(entry)
                    continue
                # R0's saved record IS the P0 output (its logits field
                # carries the P0 logits); correction candidates save the
                # P0 logits in their own p0_logits field.
                if spec.family == "p0":
                    p0_reference = saved_record["logits"]
                else:
                    p0_reference = saved_record["p0_logits"]
                stage_reports["p0"] = compare_stage_arrays(
                    np.asarray(p0_logits, dtype=np.float32),
                    np.asarray(p0_reference, dtype=np.float32),
                    atol=logit_atol, rtol=0.0, stage=STAGE_FINITE)
                if spec.family == "p0":
                    final = p0_logits
                else:
                    final_stage = classifier.candidate_final(
                        fold=fold, c14=local["c_features"],
                        s16=local["s_features"],
                        s_valid=local["s_valid"], p0_logits=p0_logits)
                    final = final_stage["final_logits"]
                stage_reports["final"] = compare_stage_arrays(
                    np.asarray(final, dtype=np.float32),
                    np.asarray(saved_record["logits"], dtype=np.float32),
                    atol=logit_atol, rtol=0.0, stage=STAGE_FINITE)
                if int(np.asarray(final).argmax()) != \
                        int(saved_record["prediction"]):
                    entry["prediction_mismatch"] = True
                if not saved_record["has_valid_s"]:
                    fallback_exact = bool(np.array_equal(
                        np.asarray(final, dtype=np.float32),
                        np.asarray(p0_logits, dtype=np.float32)))
                    entry["fallback_exact_p0"] = fallback_exact
                    if not fallback_exact:
                        stage_reports["fallback"] = {
                            "passed": False, "max_abs_error": None,
                            "status": "FAILED",
                            "reason": "missing-S row did not fall back to "
                                      "the shared frozen P0 exactly"}
                rows.append(entry)
        batch.clear()
        batch_rows.clear()

    for iterator_position, row_index in enumerate(selected_indices):
        row = samples[row_index]
        is_last = iterator_position + 1 == len(selected_indices)
        sample = load_mask_sample(training_root / row["npz_path"])
        record_input = npz_to_instance_input(
            sample, frame_id=row["source_frame"],
            instance_id=int(row["object_index"]))
        batch.append(record_input)
        batch_rows.append(row_index)
        if len(batch) < 32 and not is_last:
            continue
        _flush()

    verdict = layer_a_verdict(
        rows, expected_names=[expected_names[i] for i in selected_indices],
        expected_candidates=candidate_ids, required_stages=required_stages)
    integrity_report = {}
    summary = {
        "layer": "A_offline_reproduction",
        "sample_count": len(selected_indices),
        "full_replay": max_samples is None,
        "max_samples": max_samples,
        "candidates": candidate_ids,
        "verdict": verdict,
        "integrity_other_seeds": integrity_report,
        "tolerances": {"g_feature_atol": g_atol, "g_feature_rtol": g_rtol,
                       "c_s_atol": cs_atol, "logit_atol": logit_atol},
        "evidence": {
            "fit_dir": config.fit_dir,
            "summary_sha256": config.summary_sha256,
        },
    }
    write_json_bytes(output_dir, "layer_a_summary.json", summary)
    write_jsonl_bytes(output_dir, "layer_a_samples.jsonl", rows)
    return summary

# ---------------------------------------------------------------------------
# Layer B: the production pipeline under real Model A predictions


def _build_engine(config, *, device: str, rcs_classifier):
    """The production engine, assembled exactly as
    ``from_material_config`` assembles it — same Model A identity and
    operating point, same band SGBM, same crop padding — with the
    residual adapter bound as Model B. The fixed values the engine does
    not take as parameters are checked against the config before
    anything runs."""
    from core.polar_compute import POLAR_EPSILON
    from core.rectification import StereoRectifier
    from core.stereo_matching import StereoMatcher, StereoMatcherConfig
    from gui.inference_engine import DualStageInferenceEngine
    from models.polar_fusion import torch_device_name
    from models.segmentation import SegmentationModel
    import inspect

    model_a_cfg = config.model_a
    model_a_path = PROJECT_ROOT / model_a_cfg["path"]
    digest = sha256_of(model_a_path)
    if digest != model_a_cfg["sha256"]:
        raise ValueError(
            f"Model A sha256 mismatch: {model_a_path} is {digest}, the "
            f"config records {model_a_cfg['sha256']}")
    if str(config.runtime.get("precision")) != "float32":
        raise ValueError("the runtime runs the float32 path only")
    expected_polar = {
        "gain": 1.0,
        "epsilon": float(POLAR_EPSILON),
        "min_intensity_sum": 10.0,
    }
    for field, expected in expected_polar.items():
        declared = config.polarization.get(field)
        if declared is None or float(declared) != expected:
            raise ValueError(
                f"polarization.{field} {declared!r} != the engine's "
                f"{expected!r}; the engine does not take this as a parameter")
    engine_threshold = inspect.signature(
        StereoMatcher.compute_bands).parameters[
        "full_image_threshold"].default
    if float(config.stereo.get("full_image_threshold", -1)) != \
            float(engine_threshold):
        raise ValueError(
            f"stereo.full_image_threshold {config.stereo.get('full_image_threshold')!r} "
            f"!= the engine's {engine_threshold!r}")

    stereo_cfg = StereoMatcherConfig(
        matcher=str(config.stereo["matcher"]),
        mode=str(config.stereo["mode"]),
        max_disparity=int(config.stereo["max_disp"]),
        scale=float(config.stereo["scale"]),
        block_size=int(config.stereo["block_size"]),
        uniqueness_ratio=int(config.stereo["uniqueness_ratio"]),
        speckle_window=int(config.stereo["speckle_window"]),
        speckle_range=int(config.stereo["speckle_range"]),
        texture_threshold=float(config.stereo["texture_threshold"]),
        lr_check_threshold_px=float(config.stereo["lr_check_threshold_px"]),
        min_valid_ratio=float(config.stereo["min_valid_ratio"]),
    )
    calibration = config.calibration
    rectifier = None
    if bool(calibration["rectification"].get("enabled", False)):
        rectifier = StereoRectifier(
            PROJECT_ROOT / calibration["file"],
            r_convention=str(calibration["r_convention"]), alpha=0.0)
    engine = DualStageInferenceEngine(
        model_a=SegmentationModel(
            model_path=model_a_path, in_channels=3,
            device=torch_device_name(str(device)),
            conf_threshold=float(model_a_cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(model_a_cfg.get("iou_threshold", 0.45))),
        model_b=rcs_classifier,
        matcher=StereoMatcher(stereo_cfg),
        rectifier=rectifier,
        rectify_enabled=bool(
            calibration["rectification"].get("enabled", False)),
        device=str(device),
        conf_threshold=float(model_a_cfg.get("conf_threshold", 0.25)),
        model_a_iou_threshold=float(model_a_cfg.get("iou_threshold", 0.45)),
        model_a_imgsz=int(model_a_cfg.get("imgsz", 640)),
        stereo_config=stereo_cfg,
        crop_padding=int(config.crop_padding),
        band_vertical_margin=int(config.stereo.get("band_margin", 20)),
        baseline=float(calibration["baseline"]),
        focal_length=float(calibration["focal_length"]),
        model_b_input_mode="material_fusion",
    )
    return engine


def run_layer_b(config, protocol, *, device: str, output_dir: Path,
                classifiers, frame_folds: dict[str, str],
                split: str = "val", max_frames: int | None = None,
                detection_reference: Path | None = None,
                watch: "BudgetWatch | None" = None) -> dict:
    """The production pipeline on real Model A predictions, evaluated for
    EVERY frozen candidate, with complete per-prediction evidence.

    Model A, band SGBM, the polar features and the per-instance inputs
    are computed ONCE per frame; the raw G feature of each prediction ROI
    is extracted once per frame and shared by the 9 candidates (this is a
    per-ROI forward, never a whole-frame claim). Standardized inputs are
    never shared. The saved evidence covers every prediction of every
    candidate — matched or not — and the per-candidate identity sets are
    generated from the common per-frame prediction evidence, never from a
    candidate's own results.
    """
    import cv2

    from core.gt_geometry import build_gt_geometry
    from core.stereo_matching import to_gray_u8

    detection_identity = config.data_identity or {}
    split_csv = PROJECT_ROOT / detection_identity.get(
        "detection_split_csv",
        "datasets/underwater_seg_binary_v2/split_assignment.csv")
    label_root = PROJECT_ROOT / detection_identity.get(
        "detection_label_root", "datasets/underwater_seg_v2/labels")
    with split_csv.open(encoding="utf-8") as handle:
        split_rows = [row for row in csv.DictReader(handle)
                      if row["split"] == split]
    if max_frames is not None:
        split_rows = split_rows[:max_frames]

    engine = _build_engine(
        config, device=device,
        rcs_classifier=classifiers[_engine_bound_candidate(classifiers)])
    shared_gray = _SharedGrayExtractor(classifiers)

    snapshot = {
        "evaluation": "material_residual_runtime_v1_complete Layer B",
        "class_names": list(CLASS_NAMES),
        "candidates": sorted(classifiers),
        "model_a": {**config.model_a,
                    "data_independence_note": (
                        "Model A's training data shares its capture sources "
                        "with the classifier data; independence is unknown "
                        "and Layer B is NOT an independent blind test")},
        "matching_rule": MATCH_RULE,
        "operating_point": OPERATING_POINT,
        "fallback_rule": config.fallback_rule,
        "sync_status": "unknown",
        "bootstrap": {"seed": BOOTSTRAP_SEED,
                      "resamples": BOOTSTRAP_RESAMPLES,
                      "cluster": "capture group"},
    }
    write_json_bytes(output_dir, "protocol_snapshot.json", snapshot)

    frames_complete: list[dict] = []
    predictions_complete: list[dict] = []
    records: dict[str, list[dict]] = {cid: [] for cid in classifiers}
    fold_of_frame: dict[str, str] = {}
    failed_frames: list[str] = []

    for row in split_rows:
        stem = row["output_stem"]
        group = row["group_name"]
        fold = frame_folds[stem]
        fold_of_frame[stem] = fold
        left_path = PROJECT_ROOT / row["left_path"]
        right_path = right_image_path(left_path)
        if right_path is None or cv2.imread(str(left_path),
                                            cv2.IMREAD_UNCHANGED) is None or \
                cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED) is None:
            # an unreadable frame is NEVER a successful empty-GT frame:
            # the failed-frame evidence is kept and the run is reported
            # as explicitly incomplete (main refuses COMPLETED)
            failed_frames.append(stem)
            for candidate_id in classifiers:
                records[candidate_id].append({
                    "frame_id": stem, "group": group,
                    "frame_error": "frame_unreadable"})
            frames_complete.append({
                "frame_id": stem, "group": group, "fold": fold,
                "frame_error": "frame_unreadable", "gt": [],
                "predictions": [], "matches": [], "missed_gt_ids": [],
                "unmatched_pred_ids": []})
            continue
        left = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        height, width = left_gray.shape
        geometry = build_gt_geometry(
            label_root / split / f"{stem}.txt", width, height, padding=10)
        gt_class_list = [item.class_id for item in geometry]
        gt_masks = [item.mask for item in geometry]

        if watch is not None:
            watch.check("layer_b frame")
        b_result = engine.process_frame_detailed(
            left_gray, right_gray, sync_skew_ms=None, already_rectified=True,
            frame_id=stem)
        b_inputs = list(b_result.material_inputs)
        b_instances = list(b_result.instances)
        matches = match_instances(gt_masks,
                                  [inst.mask for inst in b_instances])
        matched_quality = {
            str(gi): (float(b_inputs[pi].quality[0])
                      if pi < len(b_inputs) and b_inputs[pi] is not None
                      else None)
            for gi, pi, _ in matches}
        matched_by_pred = {pi: gi for gi, pi, _ in matches}

        gray_values = shared_gray.extract(b_inputs)
        results_by_candidate = {}
        for candidate_id, classifier in classifiers.items():
            results_by_candidate[candidate_id] = classifier.predict_batch(
                b_inputs, gray=gray_values)

        frames_complete.append({
            "frame_id": stem, "group": group, "fold": fold,
            "frame_error": None,
            "gt": [{"gt_id": gi, "class_id": int(gt_class_list[gi])}
                   for gi in range(len(gt_class_list))],
            "predictions": [
                {"pred_instance_id": pi,
                 "valid_ratio": float(b_inputs[pi].quality[0])
                 if pi < len(b_inputs) and b_inputs[pi] is not None else None,
                 "mask_source": b_inputs[pi].mask_source
                 if pi < len(b_inputs) and b_inputs[pi] is not None else None}
                for pi in range(len(b_instances))],
            "matches": [[int(gi), int(pi), float(iou)]
                        for gi, pi, iou in matches],
            "missed_gt_ids": [gi for gi in range(len(gt_class_list))
                              if not any(m[0] == gi for m in matches)],
            "unmatched_pred_ids": [pi for pi in range(len(b_instances))
                                   if pi not in matched_by_pred],
        })

        for candidate_id, results in results_by_candidate.items():
            spec = classifiers[candidate_id].candidate_spec
            unclassified = {gi for gi, pi, _ in matches
                            if not results[pi].valid}
            records[candidate_id].append(_build_record(
                frame_id=stem, group=group, gt_classes=gt_class_list,
                matches=matches, results=results,
                unclassified=unclassified, matched_quality=matched_quality))
            for pi, result in enumerate(results):
                predictions_complete.append({
                    "candidate_id": candidate_id,
                    "family": spec.family, "seed": spec.seed,
                    "fold": fold, "frame_id": stem,
                    "pred_instance_id": pi,
                    "matched_gt_id": matched_by_pred.get(pi),
                    "predicted_class": result.class_id if result.valid
                    else None,
                    "p0_logits": None,
                    "final_logits": list(result.raw_logits or []),
                    "status": "valid" if result.valid else "failed",
                    "used_fallback": bool(result.used_fallback),
                    "fallback_reason": result.fallback_reason,
                    "mask_source": b_inputs[pi].mask_source
                    if pi < len(b_inputs) and b_inputs[pi] is not None
                    else None,
                    "valid_ratio": float(b_inputs[pi].quality[0])
                    if pi < len(b_inputs) and b_inputs[pi] is not None
                    else None,
                    "model_identity": result.model_identity,
                    "error": None if result.valid else result.invalid_reason,
                })

    _check_detection_reference(detection_reference, frames_complete,
                               output_dir)

    quality_metrics = {}
    per_group = {}
    for candidate_id in classifiers:
        variant_records = records[candidate_id]
        quality_metrics[candidate_id] = aggregate_metrics(variant_records,
                                                          CLASS_NAMES)
        per_group[candidate_id] = {
            group: aggregate_metrics(group_records, CLASS_NAMES)
            for group, group_records in sorted(
                _group_frames(variant_records).items())}

    paired, bootstrap = _paired_and_bootstrap(classifiers, records)
    fallback_summary = _fallback_over_all_predictions(
        predictions_complete, sorted(classifiers))

    summary = {
        "layer": "B_model_a_predictions",
        "frames": len(split_rows),
        "failed_frames": failed_frames,
        "complete": not failed_frames,
        "fold_of_frame": fold_of_frame,
        "metrics_by_candidate": quality_metrics,
        "per_group": per_group,
        "paired": {name: {k: v for k, v in outcome.items() if k != "changed"}
                   for name, outcome in paired.items()},
        "paired_changes_detail": {
            name: outcome["changed"] for name, outcome in paired.items()},
        "bootstrap": bootstrap,
        "fallback_all_predictions": fallback_summary,
        "quality_strata": {
            "RCS_conditional_s2026": quality_strata(
                records[_pick(classifiers, "conditional", "RCS", 2026)])},
        "polar_quality_bins": [
            {"name": name, "low": low, "high": high}
            for name, low, high in POLAR_QUALITY_BINS],
        "data_independence": "unknown — NOT an independent blind test",
        "prediction_records": len(predictions_complete),
    }
    write_json_bytes(output_dir, "layer_b_summary.json", summary)
    write_jsonl_bytes(output_dir, "frames_complete.jsonl", frames_complete)
    write_jsonl_bytes(output_dir, "predictions_complete.jsonl",
                      predictions_complete)
    write_jsonl_bytes(output_dir, "frames.jsonl", frames_complete)
    return summary


class _SharedGrayExtractor:
    """Extracts the raw G feature of each prediction ROI ONCE per frame so
    the candidates share the encoder output (a per-ROI forward, never a
    whole-frame claim)."""

    def __init__(self, classifiers: dict):
        self._classifier = next(iter(classifiers.values()))

    def extract(self, samples: list) -> list:
        gray_by_position: list = [None] * len(samples)
        forward = [(position, sample) for position, sample in
                   enumerate(samples) if sample is not None]
        if forward:
            features = self._classifier.gray_features(
                [sample for _position, sample in forward])
            for (position, _sample), feature in zip(forward, features):
                gray_by_position[position] = feature
        return gray_by_position


def _engine_bound_candidate(classifiers: dict) -> str:
    """The production engine binds the conditional RCS at seed 2026 (the
    same production-path candidate the original round bound)."""
    return _pick(classifiers, "conditional", "RCS", 2026)


def _pick(classifiers: dict, family: str, variant: str, seed: int) -> str:
    matches = [cid for cid, clf in classifiers.items()
               if clf.candidate_spec.family == family
               and clf.candidate_spec.variant == variant
               and clf.candidate_spec.seed == seed]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {family}/{variant}/s{seed} candidate, "
            f"got {matches}")
    return matches[0]


def _paired_and_bootstrap(classifiers: dict, records: dict) -> tuple[dict,
                                                                      dict]:
    """Paired changes and the frozen bootstrap, per family/seed.

    Every correction candidate is compared against R0 and against the
    SAME family/seed RC; seeds are presented separately (never averaged
    into a best-seed claim). The bootstrap keeps the frozen protocol:
    4 val capture groups, 2000 resamples, seed 2026.
    """
    paired: dict[str, dict] = {}
    bootstrap: dict[str, dict] = {}
    r0_id = _pick(classifiers, "p0", "R0", None) if any(
        clf.candidate_spec.family == "p0"
        for clf in classifiers.values()) else None
    for candidate_id, classifier in classifiers.items():
        spec = classifier.candidate_spec
        if spec.family == "p0":
            continue
        for label, base_id in (
                (f"{candidate_id}_minus_R0", r0_id),
                (f"{candidate_id}_minus_RC",
                 _pick(classifiers, spec.family, "RC", spec.seed))):
            outcome = paired_changes(records[base_id], records[candidate_id])
            paired[label] = outcome
            bootstrap[f"end_to_end_macro_f1_{label}"] = \
                end_to_end_macro_f1_bootstrap(records[candidate_id],
                                              records[base_id])
            bootstrap[f"instance_correctness_{label}"] = \
                instance_correctness_delta_bootstrap(records[candidate_id],
                                                     records[base_id])
    return paired, bootstrap


def _fallback_over_all_predictions(predictions: list[dict],
                                   candidate_ids: list[str]) -> dict:
    """Fallback frequency and reasons over EVERY saved prediction of every
    candidate (matched or not, classified or failed)."""
    summary = {candidate_id: {
        "predictions": 0, "valid": 0, "failed": 0,
        "used_fallback": 0, "reasons": {}, "fallback_ratio": 0.0,
    } for candidate_id in candidate_ids}
    for row in predictions:
        entry = summary[row["candidate_id"]]
        entry["predictions"] += 1
        if row["status"] == "valid":
            entry["valid"] += 1
        else:
            entry["failed"] += 1
        if row.get("used_fallback"):
            entry["used_fallback"] += 1
            reason = str(row.get("fallback_reason") or "")
            if not reason:
                raise ValueError(
                    f"fallback prediction {row['candidate_id']}/"
                    f"{row['frame_id']}/{row['pred_instance_id']} has no "
                    "reason")
            entry["reasons"][reason] = \
                entry["reasons"].get(reason, 0) + 1
    for entry in summary.values():
        if entry["reasons"] and sum(entry["reasons"].values()) != \
                entry["used_fallback"]:
            raise ValueError("fallback reason counts do not sum")
        if not entry["used_fallback"] and entry["reasons"]:
            raise ValueError("reasons without used_fallback")
        entry["fallback_ratio"] = (
            entry["used_fallback"] / entry["predictions"]
            if entry["predictions"] else 0.0)
    return summary


def _check_detection_reference(reference: Path | None,
                               frames_complete: list[dict],
                               output_dir: Path) -> None:
    """Read-only cross-check of the new Model A detections against the
    previous 180-frame record. Differences are REPORTED, never tuned
    away; the classification verdict never claims a detection change as
    a classification gain."""
    if reference is None:
        return
    with Path(reference).open(encoding="utf-8") as handle:
        old_frames = [json.loads(line) for line in handle
                      if line.strip()]
    old_by_frame = {}
    for row in old_frames:
        if row.get("frame_error"):
            old_by_frame[row["frame_id"]] = None
        else:
            old_by_frame[row["frame_id"]] = row
    differences = []
    compared = 0
    for frame in frames_complete:
        old = old_by_frame.get(frame["frame_id"])
        if old is None:
            differences.append({"frame_id": frame["frame_id"],
                                "kind": "missing_or_failed_in_reference"})
            continue
        compared += 1
        old_predictions = len(old.get("predictions", [])) \
            if "predictions" in old else old.get("model_a_predictions")
        new_predictions = len(frame["predictions"])
        if old_predictions != new_predictions:
            differences.append({
                "frame_id": frame["frame_id"],
                "kind": "prediction_count",
                "old": old_predictions, "new": new_predictions})
            continue
        # the old record stores the match COUNT (no per-match rows)
        old_matched = len(old.get("matches", [])) \
            if "matches" in old else old.get("matched")
        new_matched = len(frame["matches"])
        if old_matched != new_matched:
            differences.append({
                "frame_id": frame["frame_id"], "kind": "matched_count",
                "old": old_matched, "new": new_matched})
    report = {
        "reference": str(reference),
        "frames_compared": compared,
        "difference_count": len(differences),
        "differences": differences[:50],
        "note": "read-only cross-check; differences must be located and "
                "reported, never tuned away",
    }
    write_json_bytes(output_dir, "detection_reference_check.json", report)

def _group_frames(records: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["group"], []).append(record)
    return grouped


# ---------------------------------------------------------------------------
# Bounded benchmark


def run_benchmark(config, protocol, *, device: str, output_dir: Path,
                  rcs_classifier, total_watch: "BudgetWatch | None" = None,
                  clock=None) -> dict:
    """The pre-frozen bounded timing round over preloaded frames.

    TWO independent watches run here: the benchmark's own
    ``benchmark.budget_minutes`` (its exhaustion is a PARTIAL timing
    record) and the whole-evaluation watch (its exhaustion PROPAGATES to
    main — later repeats never run). Every warmup and measured repeat is
    bracketed at the same boundary — the TOTAL watch first, then the
    benchmark's own watch — and before the benchmark's PARTIAL record is
    published the total watch is re-checked, so a simultaneous
    exhaustion always propagates instead of being swallowed."""
    total_watch_check = (total_watch.check if total_watch is not None
                         else (lambda stage: None))
    import cv2

    from core.stereo_matching import to_gray_u8

    bench = config.benchmark
    bench_identity = config.data_identity or {}
    split_csv = PROJECT_ROOT / bench_identity.get(
        "detection_split_csv",
        "datasets/underwater_seg_binary_v2/split_assignment.csv")
    with split_csv.open(encoding="utf-8") as handle:
        split_rows = [row for row in csv.DictReader(handle)
                      if row["split"] == "val"]
    stems = freeze_benchmark_frames([row["output_stem"] for row in split_rows],
                                    int(bench["frame_count"]))
    by_stem = {row["output_stem"]: row for row in split_rows}

    engine = _build_engine(config, device=device, rcs_classifier=rcs_classifier)
    preloaded = []
    for stem in stems:
        left = cv2.imread(str(PROJECT_ROOT / by_stem[stem]["left_path"]),
                          cv2.IMREAD_UNCHANGED)
        right_path = right_image_path(PROJECT_ROOT / by_stem[stem]["left_path"])
        right = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        preloaded.append((stem, to_gray_u8(left), to_gray_u8(right)))

    warmups = int(bench["warmup_per_frame"])
    measured = int(bench["measured_per_frame"])
    watch = BudgetWatch(budget_minutes=float(bench["budget_minutes"]),
                        clock=clock, scope="benchmark budget")
    from models.polar_fusion import torch_device_name

    torch_name = torch_device_name(str(device))
    is_cuda = torch_name.startswith("cuda") and torch.cuda.is_available()
    raw = []
    status = "COMPLETED"
    try:
        for stem, left_gray, right_gray in preloaded:
            total_watch_check(f"benchmark frame {stem}")
            for warmup_repeat in range(warmups):
                # every repeat is bracketed at the same boundary: the
                # TOTAL watch first, then the benchmark's own watch
                total_watch_check(f"benchmark warmup {stem} "
                                  f"#{warmup_repeat}")
                watch.check(f"benchmark warmup {stem} #{warmup_repeat}")
                engine.process_frame_detailed(
                    left_gray, right_gray, sync_skew_ms=None,
                    already_rectified=True, frame_id=stem)
                total_watch_check(f"benchmark warmup {stem} "
                                  f"#{warmup_repeat} done")
                watch.check(f"benchmark warmup {stem} "
                            f"#{warmup_repeat} done")
            if is_cuda:
                torch.cuda.synchronize()
            for repeat in range(measured):
                total_watch_check(f"benchmark measured {stem} #{repeat}")
                watch.check(f"benchmark measured {stem} #{repeat}")
                started = time.perf_counter()
                result = engine.process_frame_detailed(
                    left_gray, right_gray, sync_skew_ms=None,
                    already_rectified=True, frame_id=stem)
                if is_cuda:
                    torch.cuda.synchronize()
                total = time.perf_counter() - started
                raw.append({
                    "frame_id": stem, "repeat": repeat,
                    "total_s": total,
                    "model_a_s": result.timings.get("model_a_s"),
                    "stereo_s": result.timings.get("stereo_s"),
                    "depth_s": result.timings.get("depth_s"),
                    "polar_s": result.timings.get("polar_s"),
                    "model_b_s": result.timings.get("model_b_s"),
                    "predictions": len(result.instances),
                })
                total_watch_check(f"benchmark measured {stem} #{repeat} "
                                  f"done")
                watch.check(f"benchmark measured {stem} #{repeat} done")
    except BudgetExceeded as error:
        # only the benchmark's OWN budget exhaustion is a PARTIAL timing
        # record; the whole-evaluation watch's BudgetExceeded propagates
        # to main (never swallowed here). Distinguished by the exception's
        # watch_scope field, never by string matching.
        if getattr(error, "watch_scope", "") != "benchmark budget":
            raise
        # both watches may have exhausted near-simultaneously: re-check
        # the TOTAL watch BEFORE publishing the benchmark's PARTIAL —
        # the total budget is never swallowed by this record
        total_watch_check("benchmark PARTIAL publish gate")
        status = "PARTIAL"
        raw.append({"budget_exceeded": str(error)})

    def _percentiles(values: Sequence[float]) -> dict:
        if not values:
            return {"n": 0}
        array = np.asarray(values, dtype=np.float64) * 1000.0
        return {
            "n": len(values), "mean_ms": float(array.mean()),
            "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)),
        }

    totals = [entry["total_s"] for entry in raw if "total_s" in entry]
    stage_names = ("model_a_s", "stereo_s", "depth_s", "polar_s",
                   "model_b_s")
    stage_summary = {
        name: _percentiles([entry[name] for entry in raw
                            if entry.get(name) is not None])
        for name in stage_names}
    observed_counts: dict[str, int] = {}
    for entry in raw:
        if "predictions" in entry:
            observed_counts[str(entry["predictions"])] = \
                observed_counts.get(str(entry["predictions"]), 0) + 1
    buckets = {}
    for target in bench["scenario_targets"]:
        selected = [entry for entry in raw
                    if entry.get("predictions") == int(target)]
        buckets[f"targets_{target}"] = {
            "measured": len(selected),
            "total": _percentiles([entry["total_s"] for entry in selected]),
            "fps_from_p50": (
                1000.0 / float(np.percentile(
                    np.asarray([e["total_s"] for e in selected],
                               dtype=np.float64) * 1000.0, 50)))
            if selected else None,
            "coverage": "MEASURED" if selected else "NOT_COVERED",
        }
    vram = {}
    if is_cuda:
        vram = {"peak_allocated_mb": torch.cuda.max_memory_allocated()
                / (1024 * 1024),
                "device": torch.cuda.get_device_name(0)}
    summary = {
        "benchmark": "bounded offline replay of preloaded already-rectified "
                     "frames; NOT a camera-to-GUI end-to-end claim",
        "status": status,
        "frozen": {k: v for k, v in bench.items()},
        "frames": stems,
        "total": _percentiles(totals),
        "overall_fps_from_p50": (
            1000.0 / float(np.percentile(
                np.asarray(totals, dtype=np.float64) * 1000.0, 50)))
            if totals else None,
        "stages_ms": stage_summary,
        "observed_prediction_counts": observed_counts,
        "scenario_buckets": buckets,
        "vram": vram,
        "note": "incomplete timing is PARTIAL and is never extrapolated to "
                "20-30 FPS; queue latency, camera and GUI are NOT_MEASURED",
    }
    write_json_bytes(output_dir, "performance_summary.json", summary)
    write_jsonl_bytes(output_dir, "raw_timings.jsonl", raw)
    return summary


# ---------------------------------------------------------------------------
# CLI


# The fixed candidate matrix of the completion batch: a config whose
# registry differs from this set is refused before anything runs.
REQUIRED_CANDIDATE_IDS = (
    "R0",
    "linear_local_RC_s2026",
    "linear_local_RCS_s2026",
    "conditional_RC_s2026",
    "conditional_RCS_s2026",
    "conditional_RC_s2027",
    "conditional_RCS_s2027",
    "conditional_RC_s2028",
    "conditional_RCS_s2028",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Two-layer evaluation of the frozen residual candidate "
                    "matrix (evaluation only; no training).")
    parser.add_argument("--config",
                        default="configs/experiments/"
                                "material_residual_runtime_v1.yaml")
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; reports are never overwritten")
    parser.add_argument("--mode", choices=("evaluate", "layer_a",
                                           "benchmark"),
                        default="evaluate")
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="skip the bounded timing round (the formal "
                             "complete-evaluation run uses this)")
    parser.add_argument("--budget-minutes", type=float, default=None,
                        help="whole-evaluation wall budget; defaults to the "
                             "config's evaluation_budget or 30 minutes")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="engineering guard for bounded smoke runs; the "
                             "formal evaluation runs all val frames")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="engineering guard for bounded smoke runs of "
                             "Layer A; the formal comparison replays every "
                             "dev-pool sample")
    args = parser.parse_args(argv)

    output_dir = prepare_output_dir(args.output_dir)
    config = load_residual_runtime_config(PROJECT_ROOT / args.config)
    if config.candidates:
        registry = sorted(spec.candidate_id for spec in config.candidates)
        if registry != sorted(REQUIRED_CANDIDATE_IDS):
            print(
                "refused: the candidate registry is not the fixed "
                f"9-candidate matrix (got {registry})", file=sys.stderr)
            return 1
    budget_minutes = resolve_evaluation_budget(config, args.budget_minutes)
    watch = BudgetWatch(budget_minutes=budget_minutes,
                        clock=_clock_factory(),
                        scope="whole-evaluation budget")
    run_status = {
        "config": args.config,
        "mode": args.mode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime()),
        "budget_minutes": float(budget_minutes),
        "skip_benchmark": bool(args.skip_benchmark),
        "status": "RUNNING",
        "stages_completed": [],
        "failed_frames": [],
    }
    write_json_bytes(output_dir, "run_status.json", run_status)

    def _finish(status: str, code: int) -> int:
        run_status["status"] = status
        run_status["finished_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run_status["elapsed_s"] = watch.elapsed_s()
        write_json_bytes(output_dir, "run_status.json", run_status)
        return code

    # setup, admission, protocol/split reads and classifier loading all
    # run inside the SAME whole-evaluation watch and the SAME exception
    # boundary — a setup failure is FAILED, never an escaping RUNNING
    try:
        watch.check("before config load")
        protocol = load_protocol(config)
        watch.check("after protocol load")

        # the runtime data identity records what actually flowed through
        from core.material_mask_dataset import admit_dataset_for_protocol

        data_identity = config.data_identity or {}
        training_root = PROJECT_ROOT / data_identity.get(
            "training_data_root",
            "datasets/underwater_cls_fusion_v5_instance_mask")
        watch.check("before admission")
        admission = admit_dataset_for_protocol(
            training_root,
            pooling_mode="instance_mask",
            expected_fingerprint=data_identity.get(
                "training_data_fingerprint",
                "bb9e3661fb1e71e94056893f6dfc50b9e3dee1c604f430ac0727d75362d32982"),
            expected_parent_fingerprint=data_identity.get(
                "training_parent_data_fingerprint",
                "8db8998d7a4c280064626961e4d51c92c531878ea2e92f0d1480df46379bd334"),
            expected_mask_schema=data_identity.get(
                "mask_schema", "instance_mask_polygon_crop_v1"),
            expected_mask_resize_version=data_identity.get(
                "mask_resize_version",
                "area_crop_to_imgsz_then_area_to_levels_v1"),
        )
        watch.check("after admission")
        split_csv = PROJECT_ROOT / data_identity.get(
            "detection_split_csv",
            "datasets/underwater_seg_binary_v2/split_assignment.csv")
        with split_csv.open(encoding="utf-8") as handle:
            val_rows = [row for row in csv.DictReader(handle)
                        if row["split"] == "val"]
        frame_folds = build_frame_fold_map(val_rows, protocol)
        watch.check("after split/protocol reads")

        started = time.perf_counter()
        classifiers = build_classifiers(config, protocol,
                                        device=args.device,
                                        frame_folds=frame_folds)
        watch.check("after classifier load")
    except BudgetExceeded as error:
        print(f"the whole-evaluation budget is exhausted during setup: "
              f"{error}", file=sys.stderr)
        run_status["failure"] = f"BudgetExceeded: {error}"
        return _finish("PARTIAL", 3)
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"setup failed: {type(error).__name__}: {error}",
              file=sys.stderr)
        run_status["failure"] = f"{type(error).__name__}: {error}"
        return _finish("FAILED", 4)

    detection_reference = None
    ref_path = getattr(config, "detection_reference", None)
    if ref_path:
        candidate_path = PROJECT_ROOT / str(ref_path)
        if candidate_path.is_file():
            detection_reference = candidate_path
    setup = {
        "generated_by": "scripts/eval_material_residual_pipeline.py",
        "experiment": config.experiment,
        "candidate_fit_dir": config.fit_dir,
        "summary_sha256": config.summary_sha256,
        "admission_sha256": config.admission_sha256,
        "engineering_seed": config.engineering_seed,
        "candidates": sorted(classifiers),
        "device": args.device,
        "max_samples": args.max_samples,
        "max_frames": args.max_frames,
        "skip_benchmark": bool(args.skip_benchmark),
        "formal_run": args.max_samples is None and args.max_frames is None,
        "detection_reference": str(detection_reference)
        if detection_reference else None,
        "training_data": {
            "root": str(training_root),
            "fingerprint": admission["audit_fingerprint"],
        },
        "model_a_data_independence": config.model_a.get(
            "data_independence", "unknown"),
        "elapsed_setup_s": time.perf_counter() - started,
    }
    write_json_bytes(output_dir, "run_setup.json", setup)
    detection_reference = None
    ref_path = getattr(config, "detection_reference", None)
    if ref_path:
        candidate_path = PROJECT_ROOT / str(ref_path)
        if candidate_path.is_file():
            detection_reference = candidate_path
    setup = {
        "generated_by": "scripts/eval_material_residual_pipeline.py",
        "experiment": config.experiment,
        "candidate_fit_dir": config.fit_dir,
        "summary_sha256": config.summary_sha256,
        "admission_sha256": config.admission_sha256,
        "engineering_seed": config.engineering_seed,
        "candidates": sorted(classifiers),
        "device": args.device,
        "max_samples": args.max_samples,
        "max_frames": args.max_frames,
        "skip_benchmark": bool(args.skip_benchmark),
        "formal_run": args.max_samples is None and args.max_frames is None,
        "detection_reference": str(detection_reference)
        if detection_reference else None,
        "training_data": {
            "root": str(training_root),
            "fingerprint": admission["audit_fingerprint"],
        },
        "model_a_data_independence": config.model_a.get(
            "data_independence", "unknown"),
        "elapsed_setup_s": time.perf_counter() - started,
    }
    write_json_bytes(output_dir, "run_setup.json", setup)

    try:
        if args.mode in ("layer_a", "evaluate"):
            layer_a = run_layer_a(config, protocol, device=args.device,
                                  output_dir=output_dir,
                                  classifiers=classifiers,
                                  max_samples=args.max_samples, watch=watch)
            run_status["stages_completed"].append("layer_a")
            if layer_a["verdict"]["abort_layer_b"]:
                print("Layer A FAILED its stage comparison; the formal "
                      "quality statistics are aborted before Layer B.",
                      file=sys.stderr)
                return _finish("LAYER_A_FAILED", 2)
            watch.check("after layer_a")
        if args.mode in ("evaluate", "benchmark") and not (
                args.mode == "evaluate" and args.skip_benchmark):
            run_benchmark(config, protocol, device=args.device,
                          output_dir=output_dir,
                          rcs_classifier=classifiers[
                              _engine_bound_candidate(classifiers)],
                          total_watch=watch)
            run_status["stages_completed"].append("benchmark")
            watch.check("after benchmark")
        if args.mode == "evaluate":
            layer_b = run_layer_b(
                config, protocol, device=args.device,
                output_dir=output_dir, classifiers=classifiers,
                frame_folds=frame_folds,
                max_frames=args.max_frames,
                detection_reference=detection_reference, watch=watch)
            run_status["stages_completed"].append("layer_b")
            if layer_b.get("failed_frames"):
                run_status["failed_frames"] = layer_b["failed_frames"]
                print(
                    "the evaluation is INCOMPLETE: unreadable frames "
                    f"were recorded ({layer_b['failed_frames'][:5]}); "
                    "refusing to publish a complete result",
                    file=sys.stderr)
                return _finish("INCOMPLETE", 5)
            watch.check("before completion")
    except BudgetExceeded as error:
        print(f"the whole-evaluation budget is exhausted: {error}",
              file=sys.stderr)
        run_status["failure"] = f"BudgetExceeded: {error}"
        return _finish("PARTIAL", 3)
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"evaluation failed: {type(error).__name__}: {error}",
              file=sys.stderr)
        run_status["failure"] = f"{type(error).__name__}: {error}"
        return _finish("FAILED", 4)
    print(f"evaluation complete: {args.mode} -> {output_dir}")
    return _finish("COMPLETED", 0)


if __name__ == "__main__":
    raise SystemExit(main())
