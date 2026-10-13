"""material_residual_scale_v1 — the training-side residual magnitude
experiment (third batch).

Question: on the SAME fix-version inputs (``valid_weighted_legacy_support_v1``),
the same tiny conditional head and the same training budget, is matching
the residual magnitude to the frozen P0's own score scale more effective
than the frozen ±0.5 bound — and does any new gain exceed the gain from
the quality-control information alone?

Pre-registered design (docs/experiments/material_residual_scale_v1.md):

- per fold, the scale is computed from the VERIFIED frozen P0's own
  logits on inner_train ONLY, in float64:
  ``m_i = largest - second_largest`` per sample,
  ``s_f = max(1.0, median(m_i))``. Labels, inner_val and outer_eval are
  never read for the scale; every seed and variant in the fold shares it.
- two magnitude arms over the SAME head/data/budget:
  ``fixed: a_f = 0.5`` and ``scaled: a_f = 0.5 * s_f``;
  ``delta = s_valid * a_f * tanh(head(x))``,
  ``final = P0_logits + delta``,
  ``loss = CE(final, label) + 0.01 * mean(delta^2)``. P0 logits are never
  rescaled and the P0 context is untouched.
- variants: R0 (the frozen P0 itself, no training), RC (C + P0 context,
  the S slot zeroed at the input level) and RCS (C + S + P0 context).
  Missing polarization falls back to P0 EXACTLY under both arms.

This module holds the pure experiment logic (scale rule, magnitudes,
config validation, gate evaluation and independent metric counting). It
never reads project data and never imports production summarizers; the
entries in ``scripts/`` own orchestration and identity protection.
"""

from __future__ import annotations

import os

import numpy as np

EXPERIMENT = "material_residual_scale_v1"
UPSAMPLE_FIX_VERSION = "valid_weighted_legacy_support_v1"
UPSAMPLE_LEGACY_VERSION = "legacy_zero_linear_v1"

ARMS = ("fixed", "scaled")
VARIANTS = ("RC", "RCS")
VARIANT_ORDER = ("R0", "RC", "RCS")
TRAIN_SEEDS = (2026, 2027, 2028)
FIXED_MAGNITUDE = 0.5
DEFAULT_AUX_LOSS_WEIGHT = 0.01
MIN_SCALE = 1.0

CACHE_KIND_FULL = "residual_scale_v1_dev_full"
CACHE_KIND_SMOKE = "residual_scale_v1_smoke"

# the six pre-registered promotion gates (percentage points)
GATE_R0_GAIN_PP = 1.0
GATE_RC_GAIN_PP = 0.5
GATE_WORST_FOLD_DROP_PP = 1.0
GATE_CLASS_RECALL_DROP_PP = 1.0


class ScaleRefused(ValueError):
    """An identity/config/cache admission failure (before any compute)."""


class ScaleMismatch(RuntimeError):
    """A saved-vs-recomputed disagreement found by the verifier."""


def is_link_or_reparse(path) -> bool:
    """A local link/reparse-point probe (keeps the core free of script
    imports; mirrors the registry's own check)."""
    import stat as _stat

    try:
        info = os.lstat(path)
    except OSError:
        return False
    if _stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0)
                & getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


class DeadlineExceeded(RuntimeError):
    """Raised at a safe boundary when the shared deadline has passed."""


class DeadlineWatch:
    """A wall-clock deadline shared by every stage of one entry.

    ``deadline_utc`` is an ISO-8601 instant; ``budget_minutes`` starts the
    clock at construction. Once :meth:`check` raises, the caller marks the
    run PARTIAL/FAILED and the publish phase runs exempt — an overrun can
    shorten a run but never turns it into a success.
    """

    def __init__(self, *, deadline_utc: str | None = None,
                 budget_minutes: float | None = None):
        import time
        from datetime import datetime, timezone

        if (deadline_utc is None) == (budget_minutes is None):
            raise ScaleRefused(
                "pass exactly one of deadline_utc / budget_minutes")
        if deadline_utc is not None:
            deadline = datetime.fromisoformat(
                str(deadline_utc).replace("Z", "+00:00"))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            self._deadline = deadline.timestamp()
        else:
            minutes = float(budget_minutes)
            if not (np.isfinite(minutes) and minutes > 0):
                raise ScaleRefused(
                    f"budget_minutes must be positive, got {budget_minutes!r}")
            self._deadline = time.time() + minutes * 60.0

    def remaining_s(self) -> float:
        import time

        return self._deadline - time.time()

    def check(self, at: str = "") -> None:
        if self.remaining_s() <= 0.0:
            raise DeadlineExceeded(
                "shared deadline exhausted (at "
                f"{at or 'unspecified boundary'}); stopping at this safe "
                "boundary with the finished evidence kept")


# ---------------------------------------------------------------------------
# Config


def load_scale_config(path) -> dict:
    """Load and strictly validate the experiment config."""
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScaleRefused("the experiment config must be a mapping")
    if raw.get("experiment") != EXPERIMENT:
        raise ScaleRefused(
            f"config experiment {raw.get('experiment')!r} is not "
            f"{EXPERIMENT!r}")
    anchors = raw.get("anchors")
    if not isinstance(anchors, dict) or len(anchors) != 5:
        raise ScaleRefused(
            "the config must pin exactly the five frozen anchors")
    for name, anchor in anchors.items():
        if not isinstance(anchor, dict) or \
                not anchor.get("path") or not anchor.get("sha256"):
            raise ScaleRefused(f"anchor {name!r} needs path and sha256")
    fix = raw.get("fix_inputs") or {}
    if fix.get("upsample_version") != UPSAMPLE_FIX_VERSION:
        raise ScaleRefused(
            "fix_inputs.upsample_version must be "
            f"{UPSAMPLE_FIX_VERSION!r}")
    arms = raw.get("arms") or {}
    if not set(ARMS).issubset(arms) or \
            float(arms.get("fixed", {}).get("magnitude_factor", -1)) != 0.5 \
            or float(arms.get("scaled", {}).get("magnitude_factor", -1)) \
            != 0.5:
        raise ScaleRefused(
            "arms must pin fixed/scaled with magnitude_factor 0.5 (scaled "
            "multiplies the fold scale s_f)")
    scale_rule = raw.get("scale_rule") or {}
    if str(scale_rule.get("source")) != "p0_logits_inner_train" or \
            str(scale_rule.get("statistic")) != "median" or \
            str(scale_rule.get("dtype")) != "float64":
        raise ScaleRefused(
            "scale_rule must be median over float64 P0 logits of "
            "inner_train only")
    if float(scale_rule.get("min_scale", -1)) != MIN_SCALE:
        raise ScaleRefused(f"scale_rule.min_scale must be {MIN_SCALE}")
    training = raw.get("training") or {}
    if str(training.get("optimizer")).lower() != "adamw" or \
            not training.get("full_batch") or \
            str(training.get("device")) != "cpu":
        raise ScaleRefused(
            "training must pin cpu/full_batch/AdamW")
    seeds = tuple(int(s) for s in (raw.get("seeds") or ()))
    if seeds != TRAIN_SEEDS:
        raise ScaleRefused(f"seeds must be {list(TRAIN_SEEDS)}")
    shuffle = raw.get("shuffle") or {}
    if tuple(int(s) for s in (shuffle.get("content_seeds") or ())) != \
            tuple(range(2026, 2031)):
        raise ScaleRefused(
            "shuffle.content_seeds must be 2026..2030 inclusive")
    return raw


def matcher_config_of(cfg: dict) -> dict:
    """The production stereo parameters pinned for the fix rebuild."""
    fix = cfg["fix_inputs"]
    keys = ("matcher", "mode", "max_disp", "scale", "block_size",
            "uniqueness_ratio", "speckle_window", "speckle_range",
            "texture_threshold", "lr_check_threshold_px", "min_valid_ratio")
    missing = [key for key in keys if key not in fix]
    if missing:
        raise ScaleRefused(
            f"fix_inputs is missing the production matcher keys {missing}")
    return {key: fix[key] for key in keys}


# ---------------------------------------------------------------------------
# The scale rule


def top2_margins(p0_logits: np.ndarray) -> np.ndarray:
    """``m_i = largest - second_largest`` of each P0 logit row (float64)."""
    logits = np.asarray(p0_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4 or len(logits) == 0:
        raise ScaleRefused(
            f"P0 logits must be a non-empty (n, 4) matrix, got "
            f"{logits.shape}")
    if not np.isfinite(logits).all():
        raise ScaleRefused("P0 logits contain NaN/Inf; refusing the scale")
    ordered = np.sort(logits, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def fold_scale(p0_logits_train: np.ndarray) -> tuple[float, dict]:
    """``s_f = max(1.0, median(m_i))`` over the fold's inner_train P0
    logits (float64, deterministic). Returns the scale and the evidence
    record (median, count, margins summary). Never reads labels, inner_val
    or outer_eval — the caller passes inner_train rows only."""
    margins = top2_margins(p0_logits_train)
    median = float(np.median(margins))
    scale = max(MIN_SCALE, median)
    if not np.isfinite(scale) or scale <= 0:
        raise ScaleRefused(f"illegal fold scale computed: {scale!r}")
    info = {
        "source": "p0_logits_inner_train",
        "dtype": "float64",
        "statistic": "median",
        "min_scale": MIN_SCALE,
        "n": int(len(margins)),
        "margin_median": float(np.float64(median)),
        "margin_p10": float(np.quantile(margins, 0.10)),
        "margin_p90": float(np.quantile(margins, 0.90)),
        "scale": float(np.float64(scale)),
        "scaled_applied": bool(median >= MIN_SCALE),
    }
    return float(np.float64(scale)), info


def magnitude_for_arm(arm: str, scale: float) -> float:
    """``fixed -> 0.5``; ``scaled -> 0.5 * s_f`` (the ONLY design
    difference between the arms)."""
    if arm not in ARMS:
        raise ScaleRefused(f"unknown arm {arm!r}; expected {ARMS}")
    scale = float(scale)
    if not np.isfinite(scale) or scale < MIN_SCALE:
        raise ScaleRefused(f"illegal fold scale {scale!r}")
    if arm == "fixed":
        return FIXED_MAGNITUDE
    return float(np.float64(FIXED_MAGNITUDE * scale))


# ---------------------------------------------------------------------------
# Independent metric counting (the verifier's own loops — never a
# production summarizer)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def count_metrics(labels: list[int], predictions: list[int],
                  num_classes: int = 4) -> dict:
    """Macro-F1, accuracy, confusion and per-class P/R/F1 from plain
    counting loops (the verifier's independent implementation)."""
    labels = [int(v) for v in labels]
    predictions = [int(v) for v in predictions]
    if len(labels) != len(predictions):
        raise ScaleMismatch("label/prediction counts disagree")
    confusion = [[0] * num_classes for _ in range(num_classes)]
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    correct = 0
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1
        if label == prediction:
            correct += 1
            tp[label] += 1
        else:
            fn[label] += 1
            fp[prediction] += 1
    per_class = [_prf(tp[i], fp[i], fn[i]) for i in range(num_classes)]
    macro_f1 = float(np.mean([entry["f1"] for entry in per_class]))
    return {
        "n": len(labels),
        "accuracy": correct / len(labels) if labels else None,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Cache identity markers


def scale_cache_definition(cfg: dict, *, kind: str, complete: bool,
                           frames_built: int, instances_built: int,
                           empty_frames: list[str], data_fingerprint: str,
                           protocol_sha256: str,
                           g_dim: int = 256) -> dict:
    """The feature_definition.json body of this experiment's cache.

    The definition version stays the FROZEN local definition (the C14/S16
    statistics are unchanged); the cache KIND, the fix upsample version,
    the completeness flag and the dev-pool census are this experiment's
    own identity and are what the training entry checks BEFORE the
    generic contract.
    """
    if kind not in (CACHE_KIND_FULL, CACHE_KIND_SMOKE):
        raise ScaleRefused(f"unknown cache kind {kind!r}")
    expectations = cfg["support_expectations"]
    return {
        "feature_definition_version":
            cfg["local_features"]["feature_definition_version"],
        "region_rule": cfg["local_features"]["region_rule"],
        "data_fingerprint": data_fingerprint,
        "protocol_sha256": protocol_sha256,
        "experiment": EXPERIMENT,
        "cache_kind": kind,
        "complete": bool(complete),
        "upsample_version": UPSAMPLE_FIX_VERSION,
        "slot_G": {"dim": int(g_dim),
                   "source": "frozen global cache, copied by "
                             "sample_name, bytes identical"},
        "matcher": matcher_config_of(cfg),
        "band_margin": int(cfg["fix_inputs"]["band_margin"]),
        "full_image_threshold":
            float(cfg["fix_inputs"]["full_image_threshold"]),
        "crop_pad": int(cfg["fix_inputs"]["crop_pad"]),
        "frames_expected": int(expectations["frames"]),
        "instances_expected": int(expectations["instances"]),
        "frames_built": int(frames_built),
        "instances_built": int(instances_built),
        "empty_frames": sorted(empty_frames),
        "note": ("signed_q/abs_q/polar_valid/quality regenerated through "
                 "the PRODUCTION StereoMatcher with "
                 f"{UPSAMPLE_FIX_VERSION}; gray crops, instance masks, "
                 "classes and G256 are the frozen V5/global identities"),
    }


def verify_scale_cache_markers(feature_dir, cfg: dict) -> dict:
    """This experiment's own cache admission, BEFORE the generic contract.

    Refuses directories without this experiment's markers, smoke caches,
    incomplete caches, wrong upsample versions and wrong dev-pool
    censuses — an old cache (no markers at all) is refused here too.
    """
    import json
    from pathlib import Path

    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir():
        raise ScaleRefused(f"feature cache directory missing: {feature_dir}")
    definition_path = feature_dir / "feature_definition.json"
    if not definition_path.is_file():
        raise ScaleRefused(
            f"{definition_path} missing: a directory without this "
            "experiment's markers is never admitted (an old cache carries "
            "none)")
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("experiment") != EXPERIMENT:
        raise ScaleRefused(
            f"cache experiment {definition.get('experiment')!r} != "
            f"{EXPERIMENT!r}")
    if definition.get("cache_kind") != CACHE_KIND_FULL:
        raise ScaleRefused(
            f"cache_kind {definition.get('cache_kind')!r} != "
            f"{CACHE_KIND_FULL!r}; smoke caches are never consumed for "
            "training")
    if definition.get("complete") is not True:
        raise ScaleRefused("the cache is marked incomplete")
    if definition.get("upsample_version") != UPSAMPLE_FIX_VERSION:
        raise ScaleRefused(
            f"cache upsample_version {definition.get('upsample_version')!r} "
            f"!= {UPSAMPLE_FIX_VERSION!r}")
    expectations = cfg["support_expectations"]
    empty = len(definition.get("empty_frames") or [])
    expected_frames = int(expectations["frames"]) - empty
    actual_frames = int(definition.get("frames_built", -1))
    if actual_frames != expected_frames:
        raise ScaleRefused(
            f"cache frames_built {actual_frames} != pinned "
            f"{expected_frames} ({int(expectations['frames'])} dev frames "
            f"minus {empty} empty frames)")
    actual_instances = int(definition.get("instances_built", -1))
    if actual_instances != int(expectations["instances"]):
        raise ScaleRefused(
            f"cache instances_built {actual_instances} != pinned "
            f"{int(expectations['instances'])}")
    return definition


# ---------------------------------------------------------------------------
# Pre-registered gates (mechanical evaluation; never a significance claim)


def evaluate_gates(summary: dict) -> dict:
    """Evaluate the six promotion gates from a fit summary.

    The gates decide whether a FOLLOW-UP real-mask validation batch is
    recommended; they are engineering thresholds, not statistics. All
    values are reported whether or not a gate passes.
    """
    arms = summary["arms"]
    paired = summary["paired"]

    def mean_f1(arm: str, variant: str) -> float:
        return float(arms[arm][variant]["macro_f1_mean_of_seeds"])

    def mean_recall(arm: str, variant: str, class_name: str) -> float:
        values = []
        for seed in map(str, TRAIN_SEEDS):
            pooled = arms[arm][variant]["per_seed"][seed]["pooled"]
            per_class = pooled["per_class_recall"][class_name]
            values.append(0.0 if per_class is None else float(per_class))
        return float(np.mean(values))

    scaled_rcs = mean_f1("scaled", "RCS")
    fixed_rcs = mean_f1("fixed", "RCS")
    r0 = mean_f1("fixed", "R0")
    scaled_rc = mean_f1("scaled", "RC")

    def net(comparison: str) -> int:
        return int(paired[comparison]["total"]["fixed"]
                   - paired[comparison]["total"]["broke"])

    def seed_net(comparison: str) -> dict:
        return {seed: int(entry["fixed"] - entry["broke"])
                for seed, entry in
                paired[comparison]["per_seed"].items()}

    seeds_scaled_minus_r0 = seed_net("scaled_RCS_minus_R0")
    seeds_scaled_minus_rc = seed_net("scaled_RCS_minus_RC")

    # per-fold, per-seed mean macro-F1 for the fold-level gates
    def fold_seed_means(arm: str, variant: str) -> dict:
        out = {}
        for seed in map(str, TRAIN_SEEDS):
            per_seed = arms[arm][variant]["per_seed"][seed]
            for fold_entry in per_seed["folds"]:
                out.setdefault(fold_entry["fold"], {})[seed] = \
                    float(fold_entry["macro_f1"])
        return out

    folds_rcs = fold_seed_means("scaled", "RCS")
    folds_r0 = fold_seed_means("fixed", "R0")
    folds_rc = fold_seed_means("scaled", "RC")

    def fold_seed_mean(table: dict, fold: str) -> float:
        return float(np.mean(list(table[fold].values())))

    folds_no_drop = [
        fold for fold in folds_rcs
        if fold_seed_mean(folds_rcs, fold) >= fold_seed_mean(folds_r0, fold)
        and fold_seed_mean(folds_rcs, fold) >= fold_seed_mean(folds_rc, fold)]
    worst_drop = 0.0
    for fold in folds_rcs:
        rcs = fold_seed_mean(folds_rcs, fold)
        for table in (folds_r0, folds_rc):
            drop = fold_seed_mean(table, fold) - rcs
            worst_drop = max(worst_drop, drop)

    recalls = {}
    worst_recall_drop = 0.0
    for name in summary["class_names"]:
        values = {
            "scaled_RCS": mean_recall("scaled", "RCS", name),
            "R0": mean_recall("fixed", "R0", name),
            "scaled_RC": mean_recall("scaled", "RC", name),
        }
        recalls[name] = values
        worst_recall_drop = max(
            worst_recall_drop,
            values["R0"] - values["scaled_RCS"],
            values["scaled_RC"] - values["scaled_RCS"])

    groups = summary.get("per_group_changes", {}).get(
        "scaled_RCS_minus_R0", {}).get("summed_over_seeds", {})
    positive_groups = sorted(
        group for group, counts in groups.items()
        if int(counts["fixed"]) - int(counts["broke"]) > 0)

    def pp(value: float) -> float:
        return float(value) * 100.0

    gates = {
        "gate1_scaled_rcs_vs_r0_rc": {
            "scaled_rcs_macro_f1": scaled_rcs,
            "r0_macro_f1": r0,
            "scaled_rc_macro_f1": scaled_rc,
            "gain_vs_r0_pp": pp(scaled_rcs - r0),
            "gain_vs_rc_pp": pp(scaled_rcs - scaled_rc),
            "pass": (pp(scaled_rcs - r0) >= GATE_R0_GAIN_PP - 1e-9
                     and pp(scaled_rcs - scaled_rc)
                     >= GATE_RC_GAIN_PP - 1e-9),
        },
        "gate2_scaled_vs_fixed_rcs": {
            "scaled_macro_f1": scaled_rcs,
            "fixed_macro_f1": fixed_rcs,
            "paired_net": net("scaled_RCS_minus_fixed_RCS"),
            "pass": (scaled_rcs > fixed_rcs
                     and net("scaled_RCS_minus_fixed_RCS") > 0),
        },
        "gate3_two_seeds_net_vs_both": {
            "per_seed_net_vs_r0": seeds_scaled_minus_r0,
            "per_seed_net_vs_rc": seeds_scaled_minus_rc,
            "seeds_positive_in_both": sorted(
                seed for seed, net_r0 in seeds_scaled_minus_r0.items()
                if net_r0 > 0
                and seeds_scaled_minus_rc.get(seed, 0) > 0),
            "pass": (sum(1 for seed, net_r0 in
                         seeds_scaled_minus_r0.items()
                         if net_r0 > 0
                         and seeds_scaled_minus_rc.get(seed, 0) > 0) >= 2),
        },
        "gate4_fold_stability": {
            "folds_without_drop": sorted(folds_no_drop),
            "worst_fold_drop_pp": pp(worst_drop),
            "pass": (len(folds_no_drop) >= 2
                     and pp(worst_drop) <= GATE_WORST_FOLD_DROP_PP + 1e-9),
        },
        "gate5_class_recall_floor": {
            "per_class_mean_recall": recalls,
            # the raw drops are FRACTIONS; the gate is defined in
            # PERCENTAGE POINTS, so convert before comparing with 1.0pp
            "worst_recall_drop_pp": pp(worst_recall_drop),
            "pass": pp(worst_recall_drop) <= GATE_CLASS_RECALL_DROP_PP
            + 1e-9,
        },
        "gate6_group_breadth": {
            "positive_groups_vs_r0": positive_groups,
            "positive_group_count": len(positive_groups),
            "pass": len(positive_groups) >= 2,
        },
    }
    gates["recommendation"] = (
        "advance to real-mask validation" if all(
            entry["pass"] for entry in gates.values()) else
        "stop this round of model extension; do not lower thresholds, "
        "select seeds or search")
    return gates


# ---------------------------------------------------------------------------
# Historical binding expansion: the verified anchors record the expected
# identities of files this experiment consumes (the runtime config pins
# the P0 fit's checkpoint/oracle/scaler per fold plus the caches; the
# batch-1 source manifest records after-read digests; the frozen global
# cache carries its own binding). These EXPECTATIONS come from history —
# never from a current re-hash — and registration refuses any consumed
# file whose actual bytes disagree with them.


def expand_historical_bindings(registry, cfg: dict, project_root) -> dict:
    """Populate ``registry.bindings`` from the VERIFIED anchors.

    Bindings are expectations, not verifications: the registry still
    hashes every file at registration and refuses mismatches. Files that
    no historical record covers are registered as this batch's snapshot
    and stay clearly distinguished in the manifest rows. Conflicting
    historical expectations for one path are refused outright.
    """
    import json
    from pathlib import Path

    root = Path(project_root)
    added = {"runtime_config_anchor": 0, "batch1_source_manifest": 0,
             "global_cache_binding": 0}

    def bind(path: str, sha: str, source: str) -> None:
        """Bind one historical expectation under the SAME project-relative
        ``as_posix`` key the registry uses at registration — an absolute
        or differently-spelled key would silently never hit. Paths
        outside the project root and link/reparse points are refused."""
        candidate = Path(str(path))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file() or candidate.is_dir():
            if is_link_or_reparse(candidate) or \
                    is_link_or_reparse(candidate.parent):
                raise ScaleRefused(
                    f"{path}: link or reparse point in a historical "
                    "binding source")
            resolved = candidate.resolve()
            if root != resolved and root not in resolved.parents:
                raise ScaleRefused(
                    f"{path}: a historical binding must stay inside the "
                    "project root")
            rel = candidate.relative_to(root).as_posix()
        else:
            # not on disk (yet): normalise the recorded string the same
            # way so the key still matches a later registration
            rel = candidate.as_posix()
            if rel.startswith("../") or ":/" in rel or ":" + "\\" in rel:
                raise ScaleRefused(
                    f"{path}: a historical binding must stay inside the "
                    "project root")
        existing = registry.bindings.get(rel)
        expected = {"source": source, "expected_sha256": str(sha)}
        if existing is not None and \
                existing.get("expected_sha256") not in (None, expected[
                    "expected_sha256"]):
            raise ScaleRefused(
                f"{rel}: conflicting historical expectations "
                f"({existing.get('source')} vs {source})")
        if existing is None:
            registry.bindings[rel] = expected
            added[source] += 1

    # 1. the runtime config anchor's own pins (P0 fit checkpoint/oracle/
    #    scaler per fold, encoder, caches, residual/r0 files)
    from models.material_residual_classifier import (
        load_residual_runtime_config,
    )

    runtime_cfg = load_residual_runtime_config(
        root / cfg["anchors"]["runtime_config"]["path"])
    bind(runtime_cfg.encoder_path, runtime_cfg.encoder_sha256,
         "runtime_config_anchor")
    for ref in tuple(runtime_cfg.global_cache) + \
            tuple(runtime_cfg.local_cache):
        bind(ref.path, ref.sha256, "runtime_config_anchor")
    for fold, refs in runtime_cfg.p0_models.items():
        for name, ref in refs.items():
            bind(ref.path, ref.sha256, "runtime_config_anchor")
    for seed_map in runtime_cfg.residual_models.values():
        for fold, variants in seed_map.items():
            for variant, files in variants.items():
                for name, ref in files.items():
                    bind(ref.path, ref.sha256, "runtime_config_anchor")
    for fold, ref in runtime_cfg.r0_records.items():
        bind(ref.path, ref.sha256, "runtime_config_anchor")

    # 2. the batch-1 source manifest's consumed_inputs (after-read
    #    digests). The manifest itself is an already-verified anchor; its
    #    registry record exists before this read
    registry.register(root / cfg["anchors"]["batch1_source_manifest"]
                      ["path"], "anchor:batch1_source_manifest")
    batch1 = json.loads((root / cfg["anchors"]["batch1_source_manifest"]
                         ["path"]).read_text(encoding="utf-8"))
    for entry in batch1.get("consumed_inputs", []):
        sha = entry.get("after_sha256") or entry.get("first_read_sha256")
        if sha:
            bind(entry["path"], sha, "batch1_source_manifest")

    # 3. the frozen global cache's own binding record
    global_dir = root / cfg["features_source"]["path"]
    binding_path = global_dir / "cache_binding.json"
    if binding_path.is_file():
        # a NEW source with no historical cover: snapshotted (first-read
        # identity) BEFORE this read, clearly not a historical record
        registry.register(binding_path, "global_cache:cache_binding.json")
        record = json.loads(binding_path.read_text(encoding="utf-8"))
        for name, file_record in (record.get("files") or {}).items():
            bind(str(global_dir / name), file_record["sha256"],
                 "global_cache_binding")
    return {"added": added}


# ---------------------------------------------------------------------------
# Scale provenance for the verifier


def verify_scale_evidence(saved: dict, recomputed: dict) -> list[str]:
    """Byte-level comparison of saved vs independently recomputed scales.

    The verifier recomputes ``s_f`` from its own P0 forward and demands
    EXACT float64 equality with the saved record; any drift is a
    mismatch.
    """
    problems = []
    for fold_id, record in sorted(saved.items()):
        mine = recomputed.get(fold_id)
        if mine is None:
            problems.append(f"{fold_id}: no recomputed scale")
            continue
        if float(record["scale"]) != float(mine["scale"]):
            problems.append(
                f"{fold_id}: scale {record['scale']!r} != recomputed "
                f"{mine['scale']!r}")
        if int(record["n"]) != int(mine["n"]):
            problems.append(
                f"{fold_id}: margin count {record['n']} != recomputed "
                f"{mine['n']}")
    for fold_id in sorted(recomputed):
        if fold_id not in saved:
            problems.append(f"{fold_id}: recomputed scale is not saved")
    return problems
