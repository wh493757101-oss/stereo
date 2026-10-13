"""material_polar_complementarity_v1 — zero-training complementarity
diagnostic (post-third-batch closeout).

The third batch's real 36 fits failed the pre-registered promotion gates
(Gate1-4) and the round was stopped. This module ONLY re-reads the saved
outer-eval prediction rows and the verified fix-version feature cache to
ask whether the polarization channel carries information that the gray
P0's errors could use. It never fits a model, never fits a scaler and
never runs a classifier forward; the only permitted pixel computation is
the production stereo/q recipe for the few frames a panel needs.

Every count in this module is DERIVED from the consumed rows — the
pre-registered expectations are checked against the derivation and any
disagreement is a refusal, never a hardcoded output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EXPERIMENT = "material_polar_complementarity_v1"

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")
ARMS = ("fixed", "scaled")
TRAIN_SEEDS = (2026, 2027, 2028)
VARIANTS = ("RC", "RCS")
FOLD_IDS = ("fold0", "fold1", "fold2")

# The three pre-registered main confusion groups (unique-R0-error counts
# from the third batch review): direction (true_class, r0_prediction).
MAIN_CONFUSION_GROUPS = (
    {"name": "plastic_submarine_vs_plastic_fish",
     "directions": ((1, 2), (2, 1))},
    {"name": "plastic_fish_vs_real_fish",
     "directions": ((2, 3), (3, 2))},
    {"name": "metal_submarine_vs_plastic_submarine",
     "directions": ((0, 1), (1, 0))},
)

# The pre-registered expectations from the third-batch review. They are
# CHECKED against the derivation (verify_expected_counts) — the ledger
# never writes them as its own output.
EXPECTED_COUNTS = {
    "unique_instances": 2446,
    "r0_errors": 150,
    "r0_errors_with_s": 145,
    "corrected_union": 12,
    "broken_union": 17,
    "changed_union": 29,
    "confusion_direction_counts": {
        "plastic_submarine_vs_plastic_fish": {(1, 2): 42, (2, 1): 12},
        "plastic_fish_vs_real_fish": {(2, 3): 24, (3, 2): 28},
        "metal_submarine_vs_plastic_submarine": {(0, 1): 3, (1, 0): 25},
    },
    "corrected_confusions": {(3, 2): 8, (2, 3): 3, (2, 1): 1},
}

# The 39 consumed prediction files, relative to the project root. The
# set digest is computed over EXACTLY this list (sorted) — no rglob.
PREDICTION_FILES = tuple(
    [f"analysis/runs/material_residual_scale_v1/"
     f"fit_cdd17d10ea5b49f4b99c06841ec10320/models/R0/fold{f}_R0/"
     f"outer_eval_predictions.jsonl" for f in (0, 1, 2)]
    + [f"analysis/runs/material_residual_scale_v1/"
       f"fit_cdd17d10ea5b49f4b99c06841ec10320/models/seed_{seed}/"
       f"fold{f}_{arm}_{variant}/outer_eval_predictions.jsonl"
       for arm in ARMS for seed in TRAIN_SEEDS for variant in VARIANTS
       for f in (0, 1, 2)]
)
PREDICTION_SET_SHA256 = (
    "3219de8cd8f6efc6a2a8d1821de3dae222bf3f5a80a43dfb1b153a28e201c508")

MAX_CONTROLS = 12
MAX_PANELS = 41

# Control-direction processing order (fixed, so dedup is deterministic).
CONTROL_DIRECTIONS = ((1, 2), (2, 1), (2, 3), (3, 2), (0, 1), (1, 0))


class ComplementarityRefused(Exception):
    """An identity, count or contract refusal."""


# ---------------------------------------------------------------------------
# Input integrity


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_set_digest(project_root: Path, paths) -> str:
    """The 39-file set digest: '<rel>\t<bytes>\t<sha256>\n' lines in
    sorted order, UTF-8 no BOM, LF endings, hashed with SHA256."""
    lines = []
    for rel in sorted(str(p) for p in paths):
        path = project_root / rel
        lines.append(f"{rel}\t{path.stat().st_size}\t"
                     f"{sha256_of_file(path)}\n")
    payload = "".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Ledger construction (sample-name-keyed; strict identity)


def _read_prediction_rows(path: Path, source: str) -> dict:
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("sample_name")
            if not isinstance(name, str) or not name:
                raise ComplementarityRefused(
                    f"{source}: a row has no sample_name")
            if name in rows:
                raise ComplementarityRefused(
                    f"{source}: duplicate sample_name {name}")
            required = ("class_id", "group", "has_valid_s", "logits",
                        "prediction")
            missing = [k for k in required if k not in row]
            if missing:
                raise ComplementarityRefused(
                    f"{source}: {name} is missing {missing}")
            if len(row["logits"]) != 4:
                raise ComplementarityRefused(
                    f"{source}: {name} logits width != 4")
            rows[name] = row
    if not rows:
        raise ComplementarityRefused(f"{source}: no rows")
    return rows


def _logit_margin(logits) -> float:
    """The raw top1-top2 logit margin (no softmax anywhere)."""
    values = sorted((float(v) for v in logits), reverse=True)
    return values[0] - values[1]


def _correctness(row: dict) -> bool:
    return int(row["prediction"]) == int(row["class_id"])


def build_ledger(fit_dir: Path, cache_arrays: dict, manifest_samples: list,
                 authority: dict) -> tuple[list[dict], dict]:
    """The 2446-row unique-instance ledger from the 39 saved files.

    Every row must match the authoritative cache manifest and protocol
    authority on sample_name / class / group / fold role / has_valid_s.
    Duplicates, missing samples, unknown samples and identity conflicts
    are refused — never skipped.
    """
    from scripts.train_material_local_residual import fold_parts

    names = [sample["sample_name"] for sample in manifest_samples]
    if names != sorted(names) or len(set(names)) != len(names):
        raise ComplementarityRefused(
            "the cache manifest is not uniquely sorted by sample_name")
    row_of = {name: index for index, name in enumerate(names)}
    if cache_arrays["c_features"].shape[0] != len(names) or \
            cache_arrays["s_features"].shape[0] != len(names):
        raise ComplementarityRefused(
            "the cache arrays do not cover exactly the manifest rows")
    fold_of = {}
    for sample in manifest_samples:
        outer = [fold for fold, role in
                 authority[sample["sample_name"]]["roles"].items()
                 if role == "outer_eval"]
        if len(outer) != 1:
            raise ComplementarityRefused(
                f"{sample['sample_name']}: {len(outer)} outer_eval folds")
        roles = sample["fold_roles"]
        if {fold: roles[fold] for fold in FOLD_IDS} != \
                authority[sample["sample_name"]]["roles"]:
            raise ComplementarityRefused(
                f"{sample['sample_name']}: manifest fold_roles differ "
                "from the protocol authority")
        fold_of[sample["sample_name"]] = outer[0]

    r0_by_fold = {}
    for fold_id in FOLD_IDS:
        source = f"models/R0/{fold_id}_R0/outer_eval_predictions.jsonl"
        r0_by_fold[fold_id] = _read_prediction_rows(
            fit_dir / "models" / "R0" / f"{fold_id}_R0" /
            "outer_eval_predictions.jsonl", source)
    r0 = {}
    for fold_id in FOLD_IDS:
        expected_names = {manifest_samples[index]["sample_name"]
                          for index in fold_parts(manifest_samples,
                                                  authority,
                                                  fold_id)["outer_eval"]}
        got = set(r0_by_fold[fold_id])
        if got != expected_names:
            raise ComplementarityRefused(
                f"R0 {fold_id}: rows {len(got)} != the fold's outer_eval "
                f"set (missing {len(expected_names - got)}, unknown "
                f"{len(got - expected_names)})")
        r0.update(r0_by_fold[fold_id])
    if len(r0) != len(names):
        raise ComplementarityRefused(
            f"R0 covers {len(r0)} samples != {len(names)}")

    paths: dict[tuple, dict] = {}
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            for variant in VARIANTS:
                for fold_id in FOLD_IDS:
                    rel = (f"models/seed_{seed}/"
                           f"{fold_id}_{arm}_{variant}/"
                           f"outer_eval_predictions.jsonl")
                    rows = _read_prediction_rows(fit_dir / rel, rel)
                    expected_names = {
                        manifest_samples[index]["sample_name"]
                        for index in fold_parts(manifest_samples,
                                                authority,
                                                fold_id)["outer_eval"]}
                    if set(rows) != expected_names:
                        raise ComplementarityRefused(
                            f"{rel}: the row set differs from the fold's "
                            "outer_eval set")
                    paths[(arm, seed, variant, fold_id)] = rows

    s_features = np.asarray(cache_arrays["s_features"], dtype=np.float32)
    c_features = np.asarray(cache_arrays["c_features"], dtype=np.float32)
    s_valid = np.asarray(cache_arrays["s_valid"], dtype=bool)
    ledger: list[dict] = []
    for sample in manifest_samples:
        name = sample["sample_name"]
        index = row_of[name]
        r0_row = r0[name]
        if int(r0_row["class_id"]) != int(sample["class_id"]) or \
                int(r0_row["class_id"]) != int(cache_arrays["class_id"][index]):
            raise ComplementarityRefused(f"{name}: class_id conflict")
        if r0_row["group"] != sample["group"]:
            raise ComplementarityRefused(f"{name}: group conflict")
        if bool(r0_row["has_valid_s"]) != bool(s_valid[index]):
            raise ComplementarityRefused(
                f"{name}: has_valid_s conflicts with the cache s_valid")
        row = {
            "sample_name": name,
            "source_frame": sample["source_frame"],
            "split": sample["split"],
            "group": sample["group"],
            "class_id": int(sample["class_id"]),
            "class_name": CLASS_NAMES[int(sample["class_id"])],
            "outer_fold": fold_of[name],
            "row": index,
            "s_valid": bool(s_valid[index]),
            "blocks": dict(sample["blocks"]),
            "c_features": [float(v) for v in c_features[index]],
            "s_features": [None if np.isnan(v) else float(v)
                           for v in s_features[index]],
            "r0": {
                "logits": [float(v) for v in r0_row["logits"]],
                "prediction": int(r0_row["prediction"]),
                "correct": _correctness(r0_row),
                "raw_logit_margin": _logit_margin(r0_row["logits"]),
            },
            "paths": {},
            "rcs_vs_r0": {},
            "rcs_vs_rc": {},
        }
        for arm in ARMS:
            for seed in TRAIN_SEEDS:
                for variant in VARIANTS:
                    path_row = paths[(arm, seed, variant, fold_of[name])][
                        name]
                    if int(path_row["class_id"]) != int(sample["class_id"]):
                        raise ComplementarityRefused(
                            f"{name}: {arm}/{seed}/{variant} class_id "
                            "conflict")
                    if path_row["group"] != sample["group"]:
                        raise ComplementarityRefused(
                            f"{name}: {arm}/{seed}/{variant} group conflict")
                    if bool(path_row["has_valid_s"]) != bool(s_valid[index]):
                        raise ComplementarityRefused(
                            f"{name}: {arm}/{seed}/{variant} has_valid_s "
                            "conflict")
                    row["paths"][f"{arm}|{seed}|{variant}"] = {
                        "logits": [float(v) for v in path_row["logits"]],
                        "prediction": int(path_row["prediction"]),
                        "correct": _correctness(path_row),
                    }
                rcs = row["paths"][f"{arm}|{seed}|RCS"]
                rc = row["paths"][f"{arm}|{seed}|RC"]
                for tag, other in (("rcs_vs_r0", row["r0"]),
                                   ("rcs_vs_rc", rc)):
                    fixed = bool(rcs["correct"]) and not bool(other["correct"])
                    broke = bool(other["correct"]) and not bool(rcs["correct"])
                    row[tag][f"{arm}|{seed}"] = {
                        "fixed": fixed, "broke": broke,
                        "net": int(fixed) - int(broke)}
        ledger.append(row)

    stats = derive_counts(ledger)
    return ledger, stats


# ---------------------------------------------------------------------------
# Derived counts (everything below is computed from the ledger rows)


def derive_counts(ledger: list[dict]) -> dict:
    r0_errors = [row for row in ledger if not row["r0"]["correct"]]
    corrected, broken = set(), set()
    per_seed_corrected = {seed: set() for seed in TRAIN_SEEDS}
    per_seed_broken = {seed: set() for seed in TRAIN_SEEDS}
    for row in ledger:
        for arm in ARMS:
            for seed in TRAIN_SEEDS:
                tag = row["rcs_vs_r0"][f"{arm}|{seed}"]
                if tag["fixed"]:
                    corrected.add(row["sample_name"])
                    per_seed_corrected[seed].add(row["sample_name"])
                if tag["broke"]:
                    broken.add(row["sample_name"])
                    per_seed_broken[seed].add(row["sample_name"])
    changed = corrected | broken
    confusion = {}
    for row in r0_errors:
        key = (row["class_id"], row["r0"]["prediction"])
        confusion[key] = confusion.get(key, 0) + 1
    direction_counts = {}
    for group in MAIN_CONFUSION_GROUPS:
        direction_counts[group["name"]] = {
            f"{a}->{b}": int(confusion.get((a, b), 0))
            for a, b in group["directions"]}
    corrected_confusion = {}
    for name in sorted(corrected):
        row = next(r for r in ledger if r["sample_name"] == name)
        key = (row["class_id"], row["r0"]["prediction"])
        corrected_confusion[key] = corrected_confusion.get(key, 0) + 1
    main_total = sum(
        count for group in MAIN_CONFUSION_GROUPS
        for count in direction_counts[group["name"]].values())
    return {
        "unique_instances": len(ledger),
        "r0_errors": len(r0_errors),
        "r0_errors_with_s": sum(1 for row in r0_errors if row["s_valid"]),
        "corrected_union": len(corrected),
        "broken_union": len(broken),
        "changed_union": len(changed),
        "corrected_names": sorted(corrected),
        "broken_names": sorted(broken),
        "changed_names": sorted(changed),
        "per_seed_corrected": {str(seed): sorted(per_seed_corrected[seed])
                               for seed in TRAIN_SEEDS},
        "per_seed_broken": {str(seed): sorted(per_seed_broken[seed])
                            for seed in TRAIN_SEEDS},
        "instance_model_pairs_evaluated": len(ledger) * len(ARMS)
        * len(TRAIN_SEEDS) * len(VARIANTS),
        "corrected_pairs": sum(
            1 for row in ledger for arm in ARMS for seed in TRAIN_SEEDS
            if row["rcs_vs_r0"][f"{arm}|{seed}"]["fixed"]),
        "broken_pairs": sum(
            1 for row in ledger for arm in ARMS for seed in TRAIN_SEEDS
            if row["rcs_vs_r0"][f"{arm}|{seed}"]["broke"]),
        "confusion_all": {f"{a}->{b}": count
                          for (a, b), count in sorted(confusion.items())},
        "main_confusion_total": main_total,
        "other_errors": len(r0_errors) - main_total,
        "direction_counts": direction_counts,
        "corrected_confusions": {f"{a}->{b}": count for (a, b), count
                                 in sorted(corrected_confusion.items())},
    }


def verify_expected_counts(stats: dict) -> list[str]:
    """The derived counts MUST equal the pre-registered expectations."""
    problems = []
    for key in ("unique_instances", "r0_errors", "r0_errors_with_s",
                "corrected_union", "broken_union", "changed_union"):
        if stats[key] != EXPECTED_COUNTS[key]:
            problems.append(
                f"{key}: derived {stats[key]} != expected "
                f"{EXPECTED_COUNTS[key]}")
    for group in MAIN_CONFUSION_GROUPS:
        name = group["name"]
        for a, b in group["directions"]:
            derived = stats["direction_counts"][name][f"{a}->{b}"]
            expected = EXPECTED_COUNTS["confusion_direction_counts"][
                name][(a, b)]
            if derived != expected:
                problems.append(
                    f"confusion {a}->{b}: derived {derived} != expected "
                    f"{expected}")
    for (a, b), expected in EXPECTED_COUNTS["corrected_confusions"].items():
        derived = stats["corrected_confusions"].get(f"{a}->{b}", 0)
        if derived != expected:
            problems.append(
                f"corrected confusion {a}->{b}: derived {derived} != "
                f"expected {expected}")
    if stats["main_confusion_total"] + stats["other_errors"] != \
            stats["r0_errors"]:
        problems.append("the confusion split does not sum to the errors")
    return problems


# ---------------------------------------------------------------------------
# Frozen binning (third-batch semantics: per-fold inner_train terciles of
# C0 quality_valid_ratio and C3 gray_mean_in_instance_mask)


def fold_bin_edges(cache_arrays: dict, manifest_samples: list,
                   authority: dict) -> dict:
    from scripts.train_material_local_residual import fold_parts

    c = np.asarray(cache_arrays["c_features"], dtype=np.float64)
    names = [sample["sample_name"] for sample in manifest_samples]
    edges_by_fold = {}
    for fold_id in FOLD_IDS:
        train_rows = fold_parts(manifest_samples, authority,
                                fold_id)["inner_train"]
        edges_by_fold[fold_id] = {
            "valid_ratio": [float(v) for v in np.quantile(
                c[train_rows][:, 0], [1 / 3, 2 / 3])],
            "gray_mean": [float(v) for v in np.quantile(
                c[train_rows][:, 3], [1 / 3, 2 / 3])],
            "source": ("np.quantile([1/3, 2/3]) over the fold's inner_train "
                       "C0 (quality_valid_ratio) and C3 "
                       "(gray_mean_in_instance_mask) of the fix cache; "
                       "build_donor_bins semantics"),
        }
    return edges_by_fold


def assign_bins(row_values: dict, edges_by_fold: dict) -> dict:
    """One 3x3 bin per instance from its own fold's frozen edges."""
    from core.material_residual import build_donor_bins

    names = sorted(row_values)
    valid_ratio = np.asarray([row_values[n]["valid_ratio"]
                              for n in names], dtype=np.float64)
    gray_mean = np.asarray([row_values[n]["gray_mean"]
                            for n in names], dtype=np.float64)
    folds = [row_values[n]["outer_fold"] for n in names]
    bins = {}
    for fold_id in FOLD_IDS:
        indexes = [i for i, f in enumerate(folds) if f == fold_id]
        if not indexes:
            continue
        sub = build_donor_bins(
            valid_ratio[indexes], gray_mean[indexes],
            edges=edges_by_fold[fold_id])
        for local, index in enumerate(indexes):
            bins[names[index]] = int(sub[local])
    if set(bins) != set(names):
        raise ComplementarityRefused("binning did not cover every sample")
    return bins


# ---------------------------------------------------------------------------
# Deterministic control selection (fixed before any panel is rendered)


def _all_rcs_wrong(row: dict) -> bool:
    return all(not row["paths"][f"{arm}|{seed}|RCS"]["correct"]
               for arm in ARMS for seed in TRAIN_SEEDS)


def _all_rcs_right(row: dict) -> bool:
    return all(row["paths"][f"{arm}|{seed}|RCS"]["correct"]
               for arm in ARMS for seed in TRAIN_SEEDS)


def select_controls(ledger: list[dict], stats: dict) -> tuple[list[dict],
                                                              list[str]]:
    """At most 12 controls, fixed before any panel is rendered.

    For each of the six main directions: one error control (R0 wrong in
    that direction, ALL SIX RCS results still wrong, not in the 29-set)
    whose raw_logit_margin is closest to the eligible set's median, then
    one correct control (same true class, same collection group, R0
    right, ALL SIX RCS right) minimising |C3 diff| + |C0 diff|. Ties go
    to the smaller sample_name. Global dedup; shortages are recorded,
    never backfilled visually.
    """
    by_name = {row["sample_name"]: row for row in ledger}
    changed = set(stats["changed_names"])
    used: set[str] = set(changed)
    selections: list[dict] = []
    notes: list[str] = []
    for a, b in CONTROL_DIRECTIONS:
        direction = f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}"
        eligible = [
            row for row in ledger
            if row["class_id"] == a
            and row["r0"]["prediction"] == b
            and row["sample_name"] not in used
            and _all_rcs_wrong(row)]
        if not eligible:
            notes.append(f"{direction}: no eligible error control")
            continue
        margins = sorted(row["r0"]["raw_logit_margin"]
                         for row in eligible)
        median = float(np.median(margins))
        error_row = min(
            eligible,
            key=lambda r: (abs(r["r0"]["raw_logit_margin"] - median),
                           r["sample_name"]))
        used.add(error_row["sample_name"])
        candidates = [
            row for row in ledger
            if row["class_id"] == a
            and row["group"] == error_row["group"]
            and row["r0"]["correct"]
            and row["sample_name"] not in used
            and _all_rcs_right(row)]
        if not candidates:
            notes.append(
                f"{direction}: no eligible correct control in group "
                f"{error_row['group']!r}")
            selections.append({
                "direction": direction, "kind": "error",
                "sample_name": error_row["sample_name"],
                "reason": ("R0 error in the direction with all six RCS "
                           "results wrong; raw_logit_margin closest to "
                           "the eligible median"),
                "error_control": error_row["sample_name"],
                "correct_control": None,
            })
            continue
        c3, c0 = 3, 0
        correct_row = min(
            candidates,
            key=lambda r: (
                abs(r["c_features"][c3] - error_row["c_features"][c3])
                + abs(r["c_features"][c0] - error_row["c_features"][c0]),
                r["sample_name"]))
        used.add(correct_row["sample_name"])
        selections.append({
            "direction": direction, "kind": "pair",
            "sample_name": error_row["sample_name"],
            "error_control": error_row["sample_name"],
            "correct_control": correct_row["sample_name"],
            "reason": ("error control: direction error with all six RCS "
                       "wrong, margin nearest the eligible median; "
                       "correct control: same class and group, all six "
                       "RCS right, minimal |C3 diff| + |C0 diff|"),
        })
    # flatten one panel entry per selected instance
    panels = []
    for item in selections:
        # '>' is illegal in Windows filenames; the JSON fields keep the
        # arrow, only panel ids are sanitized
        safe_direction = item["direction"].replace("->", "_to_")
        for kind, key in (("error_control", "error_control"),
                          ("correct_control", "correct_control")):
            name = item[key]
            if name is None:
                continue
            panels.append({
                "panel_id": f"control_{safe_direction}_{kind}",
                "sample_name": name, "kind": "control",
                "direction": item["direction"], "role": kind,
                "reason": item["reason"]})
    if len(panels) > MAX_CONTROLS:
        raise ComplementarityRefused(
            f"control selection produced {len(panels)} > {MAX_CONTROLS}")
    return panels, notes


# ---------------------------------------------------------------------------
# Descriptive conditional statistics (unique instances only)


def _finite(values):
    array = np.asarray([v for v in values if v is not None],
                       dtype=np.float64)
    return array[np.isfinite(array)]


def _subset_summary(rows: list[dict], bin_of: dict) -> dict:
    s_keys = ("signed_q_mean", "signed_q_std_ddof0", "signed_q_p10",
              "signed_q_p25", "signed_q_p50", "signed_q_p75",
              "signed_q_p90", "abs_q_mean")
    summary = {
        "support": len(rows),
        "n_groups": len({row["group"] for row in rows}),
        "n_folds": len({row["outer_fold"] for row in rows}),
        "n_missing_s": sum(1 for row in rows if not row["s_valid"]),
        "bins": {},
    }
    counts = {}
    for row in rows:
        counts[bin_of[row["sample_name"]]] = \
            counts.get(bin_of[row["sample_name"]], 0) + 1
    summary["bins"] = {str(k): v for k, v in sorted(counts.items())}
    for block, offset in (("interior", 0), ("band", 8)):
        summary[block] = {}
        for local, key in enumerate(s_keys):
            values = _finite([row["s_features"][offset + local]
                              for row in rows])
            if values.size:
                summary[block][key] = {
                    "n": int(values.size),
                    "median": float(np.median(values)),
                    "iqr": [float(np.percentile(values, 25)),
                            float(np.percentile(values, 75))],
                }
            else:
                summary[block][key] = {"n": 0, "median": None,
                                       "iqr": None}
    summary["c0_valid_ratio_median"] = (
        float(np.median(_finite([row["c_features"][0]
                                 for row in rows])))
        if rows else None)
    summary["c3_gray_mean_median"] = (
        float(np.median(_finite([row["c_features"][3]
                                 for row in rows])))
        if rows else None)
    summary["low_support"] = bool(
        len(rows) < 8 or summary["n_groups"] < 2)
    return summary


def conditional_tables(ledger: list[dict], stats: dict,
                       bin_of: dict) -> dict:
    """Descriptive only: support, groups, missing counts, S16 medians and
    IQRs, C info and per-group change directions for each main confusion
    group. No probe classifiers, no feature selection, no thresholds."""
    by_name = {row["sample_name"]: row for row in ledger}
    corrected = set(stats["corrected_names"])
    broken = set(stats["broken_names"])
    tables = {"note": ("descriptive statistics over unique instances; "
                       "low_support marks subsets with <8 instances or a "
                       "single collection group — such cells are "
                       "'insufficient evidence', never a finding"),
              "groups": {}}
    for group in MAIN_CONFUSION_GROUPS:
        classes = {a for a, _ in group["directions"]}
        directions = set(group["directions"])
        def in_group(row):
            return row["class_id"] in classes
        def r0_error_in_group(row):
            return (row["class_id"], row["r0"]["prediction"]) in directions
        subsets = {
            "all_instances_of_the_pair": [r for r in ledger
                                          if in_group(r)],
            "r0_errors_in_group": [r for r in ledger
                                   if in_group(r)
                                   and r0_error_in_group(r)],
            "r0_correct_in_group": [r for r in ledger
                                    if in_group(r)
                                    and r["r0"]["correct"]],
            "corrected": [by_name[n] for n in sorted(corrected)
                          if in_group(by_name[n])
                          and r0_error_in_group(by_name[n])],
            "broken": [by_name[n] for n in sorted(broken)
                       if in_group(by_name[n])],
            "always_wrong": [
                r for r in ledger
                if in_group(r) and r0_error_in_group(r)
                and _all_rcs_wrong(r)],
        }
        group_table = {}
        for label, rows in subsets.items():
            entry = _subset_summary(rows, bin_of)
            per_group = {}
            for row in rows:
                per_group.setdefault(row["group"], {"support": 0})
                per_group[row["group"]]["support"] += 1
            entry["per_group_support"] = dict(sorted(per_group.items()))
            group_table[label] = entry
        tables["groups"][group["name"]] = {
            "directions": [f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}"
                           for a, b in group["directions"]],
            "subsets": group_table,
        }
    return tables


def _cell_stats(rows: list[dict]) -> dict:
    """One CONDITIONAL cell: unique-instance support, group/fold context,
    missing/validity counts, per-S16-component within-cell median and
    quartiles (missing entries excluded from the statistic and counted),
    C0/C3 quartiles and the explicit insufficiency flags. A cell never
    mixes true classes; bins never mix folds."""
    s_keys = ("signed_q_mean", "signed_q_std_ddof0", "signed_q_p10",
              "signed_q_p25", "signed_q_p50", "signed_q_p75",
              "signed_q_p90", "abs_q_mean")
    cell = {
        "support": len(rows),
        "groups": sorted({row["group"] for row in rows}),
        "outer_folds": sorted({row["outer_fold"] for row in rows}),
        "n_missing_s": sum(1 for row in rows if not row["s_valid"]),
        "n_interior_valid": sum(1 for row in rows
                                if row["blocks"].get("interior_valid")),
        "n_band_valid": sum(1 for row in rows
                            if row["blocks"].get("band_valid")),
    }
    for block, offset in (("interior", 0), ("band", 8)):
        cell[block] = {}
        for local, key in enumerate(s_keys):
            present = [row["s_features"][offset + local] for row in rows
                       if row["s_features"][offset + local] is not None]
            values = _finite(present)
            n_valid = int(values.size)
            n_missing = len(rows) - n_valid
            if n_valid:
                cell[block][key] = {
                    "n_valid": n_valid, "n_missing": n_missing,
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                }
            else:
                cell[block][key] = {"n_valid": 0,
                                    "n_missing": n_missing,
                                    "median": None, "q25": None,
                                    "q75": None}
    for name, column in (("c0_valid_ratio", 0), ("c3_gray_mean", 3)):
        values = _finite([row["c_features"][column] for row in rows])
        if values.size:
            cell[name] = {"n_valid": int(values.size),
                          "median": float(np.median(values)),
                          "q25": float(np.percentile(values, 25)),
                          "q75": float(np.percentile(values, 75))}
        else:
            cell[name] = {"n_valid": 0, "median": None, "q25": None,
                          "q75": None}
    flags = []
    if not rows:
        flags.append("no_support")
    elif len(rows) == 1:
        flags.append("single_instance")
    if len(cell["groups"]) <= 1:
        flags.append("single_group")
    if len(cell["outer_folds"]) > 1:
        flags.append("multi_fold_bin_context")
    cell["flags"] = flags
    return cell


def _median_gap(a: dict, b: dict) -> dict | None:
    """Both sides' supports and valid ns plus the median difference and
    its direction; None when either side has no valid data."""
    if a is None or b is None:
        return None
    if not a.get("n_valid") or not b.get("n_valid"):
        return None
    gap = float(a["median"]) - float(b["median"])
    return {"median_diff": gap,
            "direction": ("first_higher" if gap > 0
                          else "second_higher" if gap < 0 else "same")}


def _side_summary(rows: list[dict]) -> dict:
    cell = _cell_stats(rows)
    return {"support": cell["support"],
            "groups": cell["groups"], "outer_folds": cell["outer_folds"],
            "n_missing_s": cell["n_missing_s"],
            "interior": cell["interior"], "band": cell["band"],
            "c0_valid_ratio": cell["c0_valid_ratio"],
            "c3_gray_mean": cell["c3_gray_mean"],
            "flags": cell["flags"]}


def build_conditional_analysis(ledger: list[dict], stats: dict,
                               bin_of: dict) -> dict:
    """The real conditional analysis over unique instances.

    Layers: per true class; class x outer_fold; class x collection group
    (fold kept); class x fold x C0/C3 joint bin; class x group x fold x
    bin. Bins NEVER mix folds (the same bin number means different
    absolute-brightness edges in different folds) and cells never mix
    true classes. Same-group descriptive comparisons pair corrected vs
    always-wrong, broken vs stable-correct (R0 correct AND all six RCS
    correct — a derived subset of the saved outcomes, no new selection)
    and the two true classes of a pair, within the SAME
    group/fold/bin cell. Everything is descriptive observation: no
    PASS/FAIL threshold, no significance screen, no component picking.
    """
    by_name = {row["sample_name"]: row for row in ledger}
    corrected = set(stats["corrected_names"])
    broken = set(stats["broken_names"])
    stable_correct = {
        row["sample_name"] for row in ledger
        if row["r0"]["correct"] and _all_rcs_right(row)}
    analysis = {
        "note": (
            "descriptive statistics over unique instances. Layered cells "
            "never mix true classes and never mix folds inside a bin; a "
            "missing side is support=0 with null statistics, never a zero "
            "measurement. low_support-style flags are an implementation "
            "display hint, NOT a pre-registered gate and NOT a route "
            "selector. Same-group comparisons are observations, not "
            "PASS/FAIL outcomes."),
        "stable_correct_definition": (
            "R0 correct AND all six RCS paths correct (a derived subset "
            "of the saved outcomes; no new selection or control panel)"),
        "broken_subset_note": (
            "the broken subset is selected by TRUE class belonging to the "
            "pair; the new wrong class of a broken instance need not "
            "belong to the same confusion direction"),
        "groups": {}, "same_group_comparisons": [],
        "comparisons_by_group": {},
    }
    all_comparisons: list = []
    for group in MAIN_CONFUSION_GROUPS:
        classes = sorted({a for a, _ in group["directions"]})
        directions = set(group["directions"])

        def in_pair(row, _classes=classes):
            return row["class_id"] in _classes

        def r0_error_in_pair(row, _directions=directions):
            return (row["class_id"], row["r0"]["prediction"]) in _directions

        subsets = {
            "all_instances_of_the_pair": [r for r in ledger
                                          if in_pair(r)],
            "r0_errors_in_pair": [r for r in ledger
                                  if in_pair(r) and r0_error_in_pair(r)],
            "r0_correct_in_pair": [r for r in ledger
                                   if in_pair(r) and r["r0"]["correct"]],
            "corrected": [by_name[n] for n in sorted(corrected)
                          if in_pair(by_name[n])
                          and r0_error_in_pair(by_name[n])],
            "broken": [by_name[n] for n in sorted(broken)
                       if in_pair(by_name[n])],
            "always_wrong": [r for r in ledger
                             if in_pair(r) and r0_error_in_pair(r)
                             and _all_rcs_wrong(r)],
            "stable_correct": [r for r in ledger
                               if in_pair(r)
                               and r["sample_name"] in stable_correct],
        }
        group_entry = {
            "directions": [f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}"
                           for a, b in group["directions"]],
            "true_classes": [CLASS_NAMES[c] for c in classes],
            "subsets_overview": {label: _side_summary(rows)
                                 for label, rows in subsets.items()},
            "layered": [],
        }
        # the layered cells: the union of every subset's rows, keyed by
        # the actual (class, fold, group, bin) combinations that occur
        subset_labels = {label: {r["sample_name"] for r in rows}
                         for label, rows in subsets.items()}
        cells: dict[tuple, dict] = {}
        for label, names in subset_labels.items():
            for name in names:
                row = by_name[name]
                key = (row["class_id"], row["outer_fold"], row["group"],
                       bin_of[name])
                cells.setdefault(key, []).append((label, row))
        for key in sorted(cells):
            cls_id, fold, grp, bin_id = key
            members = cells[key]
            row0 = members[0][1]
            # the same instance belongs to several subsets — the cell's
            # statistics must count each UNIQUE instance exactly once
            unique_rows = {row["sample_name"]: row
                           for _label, row in members}
            cell = _cell_stats(list(unique_rows.values()))
            cell_entry = {
                "level": "true_class x group x fold x bin",
                "true_class": CLASS_NAMES[cls_id],
                "outer_fold": fold, "group": grp, "bin": int(bin_id),
                **cell,
                "subset_support": {
                    label: sum(1 for l, _ in members if l == label)
                    for label in subsets},
            }
            # the coarser layers are derivable; the class-level and
            # class x fold summaries are kept explicitly for review
            group_entry["layered"].append(cell_entry)
        # same-group comparisons within identical (true class, group,
        # fold, bin) cells — accumulated across ALL confusion groups
        # (the global list is the concatenation of the three groups'
        # lists; nothing is overwritten)
        comparisons = []
        for comparison_type, left_label, right_label in (
                ("corrected_vs_always_wrong", "corrected", "always_wrong"),
                ("broken_vs_stable_correct", "broken", "stable_correct")):
            left_by_cell: dict[tuple, list] = {}
            right_by_cell: dict[tuple, list] = {}
            for name in subset_labels[left_label]:
                row = by_name[name]
                left_by_cell.setdefault(
                    (row["class_id"], row["group"], row["outer_fold"],
                     bin_of[name]), []).append(row)
            for name in subset_labels[right_label]:
                row = by_name[name]
                right_by_cell.setdefault(
                    (row["class_id"], row["group"], row["outer_fold"],
                     bin_of[name]), []).append(row)
            type_index = 0
            for key in sorted(set(left_by_cell) | set(right_by_cell)):
                left_rows = left_by_cell.get(key, [])
                right_rows = right_by_cell.get(key, [])
                left = _side_summary(left_rows)
                right = _side_summary(right_rows)
                cls_id, grp, fold, bin_id = key
                entry = {
                    "confusion_group": group["name"],
                    "comparison_type": comparison_type,
                    "comparison_id": (
                        f"{group['name']}|{comparison_type}|"
                        f"{CLASS_NAMES[cls_id]}|{grp}|{fold}|{bin_id}"),
                    "pair": f"{left_label}_vs_{right_label}",
                    "true_class": CLASS_NAMES[cls_id], "group": grp,
                    "outer_fold": fold, "bin": int(bin_id),
                    "left": left, "right": right,
                    "comparable": bool(left_rows) and bool(right_rows),
                }
                if entry["comparable"]:
                    gaps = {}
                    for block in ("interior", "band"):
                        for s_key in left[block]:
                            gap = _median_gap(left[block][s_key],
                                              right[block][s_key])
                            if gap is not None:
                                gaps[f"{block}.{s_key}"] = gap
                    for c_key in ("c0_valid_ratio", "c3_gray_mean"):
                        gap = _median_gap(left[c_key], right[c_key])
                        if gap is not None:
                            gaps[c_key] = gap
                    entry["median_gaps"] = gaps
                    entry["n_common_valid_components"] = sum(
                        1 for key in gaps
                        if key.startswith(("interior.", "band.")))
                else:
                    entry["median_gaps"] = {}
                    entry["n_common_valid_components"] = 0
                    entry["note"] = (
                        "not comparable: one side has no support in this "
                        "group/fold/bin cell (support recorded, statistics "
                        "null — never a zero measurement)")
                type_index += 1
                comparisons.append(entry)
        # the two true classes of the pair, within the same group/fold/bin;
        # the two directions of one pair are MIRRORS of the same underlying
        # cells and must never be counted as independent support
        for a, b in group["directions"]:
            by_cell: dict[tuple, dict[str, list]] = {}
            for row in ledger:
                if row["class_id"] not in (a, b):
                    continue
                by_cell.setdefault(
                    (row["group"], row["outer_fold"], bin_of[
                        row["sample_name"]]), {}).setdefault(
                    row["class_id"], []).append(row)
            type_index = 0
            for key in sorted(by_cell):
                sides = by_cell[key]
                entry = {
                    "confusion_group": group["name"],
                    "comparison_type": "class_vs_class",
                    "comparison_id": (
                        f"{group['name']}|class_vs_class|"
                        f"{CLASS_NAMES[a]}|{key[0]}|{key[1]}|{key[2]}"),
                    "pair": f"{CLASS_NAMES[a]}_vs_{CLASS_NAMES[b]}",
                    "group": key[0], "outer_fold": key[1],
                    "bin": int(key[2]),
                    "first_class": CLASS_NAMES[a],
                    "second_class": CLASS_NAMES[b],
                    "mirror_note": (
                        "the two directions of one confusion pair are the "
                        "same underlying cells read in opposite order — "
                        "never two independent supports"),
                    "first": _side_summary(sides.get(a, [])),
                    "second": _side_summary(sides.get(b, [])),
                    "comparable": bool(sides.get(a)) and bool(sides.get(b)),
                }
                if entry["comparable"]:
                    gaps = {}
                    for block in ("interior", "band"):
                        for s_key in entry["first"][block]:
                            gap = _median_gap(entry["first"][block][s_key],
                                              entry["second"][block][s_key])
                            if gap is not None:
                                gaps[f"{block}.{s_key}"] = gap
                    for c_key in ("c0_valid_ratio", "c3_gray_mean"):
                        gap = _median_gap(entry["first"][c_key],
                                          entry["second"][c_key])
                        if gap is not None:
                            gaps[c_key] = gap
                    entry["median_gaps"] = gaps
                    entry["n_common_valid_components"] = sum(
                        1 for key in gaps
                        if key.startswith(("interior.", "band.")))
                else:
                    entry["median_gaps"] = {}
                    entry["n_common_valid_components"] = 0
                    entry["note"] = (
                        "not comparable: one true class has no support in "
                        "this group/fold/bin cell")
                type_index += 1
                comparisons.append(entry)
        analysis["groups"][group["name"]] = group_entry
        analysis["comparisons_by_group"][group["name"]] = comparisons
        all_comparisons.extend(comparisons)
    analysis["same_group_comparisons"] = all_comparisons
    analysis["comparison_count_check"] = {
        "global": len(all_comparisons),
        "per_confusion_group": {
            name: len(rows) for name, rows in
            analysis["comparisons_by_group"].items()},
        "rule": ("global_count must equal the sum of the per-group "
                 "counts; each entry carries confusion_group, "
                 "comparison_type, comparison_id and the full condition "
                 "keys"),
    }
    return analysis


SUBSET_LABELS = ("all_instances_of_the_pair", "r0_errors_in_pair",
                 "r0_correct_in_pair", "corrected", "broken",
                 "always_wrong", "stable_correct")

LAYER_LEVELS = ("L1_true_class", "L2_class_x_fold",
                "L3_class_x_group_keeps_fold",
                "L4_class_x_fold_x_bin",
                "L5_class_x_group_x_fold_x_bin")


def subset_layered_analysis(ledger: list[dict], stats: dict,
                            bin_of: dict) -> dict:
    """The five layers FOR EACH confusion group AND EACH subset, every
    cell computed from that subset's own unique members:

    L1 true class; L2 class x outer_fold; L3 class x group (folds kept);
    L4 class x fold x C0/C3 joint bin; L5 class x group x fold x bin.

    A coarser level is recomputed from the members, never merged from
    finer cells; bins never mix folds; cells never mix true classes; an
    instance counts once per (subset, partition). Support conservation
    and the n_valid+n_missing=support identity are checked mechanically
    for every level/class/subset."""
    by_name = {row["sample_name"]: row for row in ledger}
    corrected = set(stats["corrected_names"])
    broken = set(stats["broken_names"])
    stable_correct = {
        row["sample_name"] for row in ledger
        if row["r0"]["correct"] and _all_rcs_right(row)}
    s_keys = ("signed_q_mean", "signed_q_std_ddof0", "signed_q_p10",
              "signed_q_p25", "signed_q_p50", "signed_q_p75",
              "signed_q_p90", "abs_q_mean")
    problems = []

    def partitions(members: list[dict], level: str) -> dict[tuple, list]:
        keyed: dict[tuple, list] = {}
        for row in members:
            cls_id = row["class_id"]
            if level == "L1_true_class":
                key = (cls_id,)
            elif level == "L2_class_x_fold":
                key = (cls_id, row["outer_fold"])
            elif level == "L3_class_x_group_keeps_fold":
                key = (cls_id, row["group"])
            elif level == "L4_class_x_fold_x_bin":
                key = (cls_id, row["outer_fold"],
                       bin_of[row["sample_name"]])
            else:
                key = (cls_id, row["group"], row["outer_fold"],
                       bin_of[row["sample_name"]])
            keyed.setdefault(key, []).append(row)
        return keyed

    def cell_payload(level: str, subset_label: str, group_name: str,
                     key: tuple, members: list[dict]) -> dict:
        cls_id = key[0]
        payload = {
            "level": level, "subset": subset_label,
            "confusion_group": group_name,
            "true_class": CLASS_NAMES[cls_id],
            **_cell_stats(members),
        }
        if level in ("L2_class_x_fold", "L4_class_x_fold_x_bin"):
            payload["outer_fold"] = key[1]
        if level == "L3_class_x_group_keeps_fold":
            payload["group"] = key[1]
            payload["outer_folds"] = sorted({m["outer_fold"]
                                             for m in members})
        if level == "L4_class_x_fold_x_bin":
            payload["bin"] = int(key[2])
        if level == "L5_class_x_group_x_fold_x_bin":
            payload["group"] = key[1]
            payload["outer_fold"] = key[2]
            payload["bin"] = int(key[3])
        return payload

    out = {
        "note": (
            "five layers x seven subsets per confusion group; every cell "
            "is computed from its own subset members (coarser levels are "
            "recomputed, never merged from finer cells); bins never mix "
            "folds; an instance counts once per (subset, partition) and "
            "overlapping subsets must never be added up as independent "
            "samples"),
        "subset_definitions": {
            "all_instances_of_the_pair": "true class belongs to the pair",
            "r0_errors_in_pair": ("true class in the pair AND the R0 "
                                  "error direction belongs to the pair"),
            "r0_correct_in_pair": ("true class in the pair AND R0 "
                                   "correct"),
            "corrected": ("R0 wrong in the pair AND at least one of the "
                          "six RCS paths correct"),
            "broken": ("true class in the pair AND R0 correct AND at "
                       "least one of the six RCS paths wrong (the new "
                       "wrong class need not belong to the pair)"),
            "always_wrong": ("R0 wrong in the pair AND all six RCS paths "
                             "wrong"),
            "stable_correct": ("R0 correct AND all six RCS paths correct "
                               "(derived subset, no new selection)"),
        },
        "groups": {}, "conservation_checks": [],
    }
    for group in MAIN_CONFUSION_GROUPS:
        classes = sorted({a for a, _ in group["directions"]})
        directions = set(group["directions"])

        def in_pair(row, _classes=classes):
            return row["class_id"] in _classes

        def r0_error_in_pair(row, _directions=directions):
            return (row["class_id"], row["r0"]["prediction"]) in _directions

        subsets = {
            "all_instances_of_the_pair": [r for r in ledger
                                          if in_pair(r)],
            "r0_errors_in_pair": [r for r in ledger
                                  if in_pair(r) and r0_error_in_pair(r)],
            "r0_correct_in_pair": [r for r in ledger
                                   if in_pair(r) and r["r0"]["correct"]],
            "corrected": [by_name[n] for n in sorted(corrected)
                          if in_pair(by_name[n])
                          and r0_error_in_pair(by_name[n])],
            "broken": [by_name[n] for n in sorted(broken)
                       if in_pair(by_name[n])],
            "always_wrong": [r for r in ledger
                             if in_pair(r) and r0_error_in_pair(r)
                             and _all_rcs_wrong(r)],
            "stable_correct": [r for r in ledger
                               if in_pair(r)
                               and r["sample_name"] in stable_correct],
        }
        group_out = {"subsets": {}}
        for subset_label, members in subsets.items():
            unique_members = {r["sample_name"]: r for r in members}
            member_rows = [unique_members[n] for n in sorted(unique_members)]
            levels_out = {}
            for level in LAYER_LEVELS:
                keyed = partitions(member_rows, level)
                cells = [cell_payload(level, subset_label, group["name"],
                                      key, keyed[key])
                         for key in sorted(keyed)]
                levels_out[level] = cells
                # mechanical conservation: the partition supports must
                # sum to the unique member count per true class
                per_class_members: dict[int, int] = {}
                for row in member_rows:
                    per_class_members[row["class_id"]] = \
                        per_class_members.get(row["class_id"], 0) + 1
                per_class_cells: dict[int, int] = {}
                for cell in cells:
                    cls_id = CLASS_NAMES.index(cell["true_class"])
                    per_class_cells[cls_id] = per_class_cells.get(
                        cls_id, 0) + cell["support"]
                for cls_id, expected in sorted(per_class_members.items()):
                    got = per_class_cells.get(cls_id, 0)
                    if got != expected:
                        problems.append(
                            f"{group['name']}/{subset_label}/{level}/"
                            f"{CLASS_NAMES[cls_id]}: partition supports "
                            f"sum to {got} != {expected} unique members")
                # n_valid + n_missing == support for every S16 component
                for cell in cells:
                    for block in ("interior", "band"):
                        for s_key in cell[block]:
                            entry = cell[block][s_key]
                            if entry["n_valid"] + entry["n_missing"] != \
                                    cell["support"]:
                                problems.append(
                                    f"{group['name']}/{subset_label}/"
                                    f"{level}/{cell['true_class']}: "
                                    f"{block}.{s_key} n_valid+n_missing "
                                    "!= support")
            group_out["subsets"][subset_label] = {
                "unique_members": len(member_rows),
                "layers": levels_out,
            }
        out["groups"][group["name"]] = group_out
    out["conservation_checks"] = {
        "problems": problems,
        "verdict": "PASS" if not problems else
        f"FAIL ({len(problems)} problems)",
    }
    return out


def q1_fish_pair_common_support(analysis: dict) -> dict:
    """Q1, derived ONLY from confusion_group=plastic_fish_vs_real_fish,
    comparison_type=corrected_vs_always_wrong: the cells where BOTH
    sides have instance support, how many collection groups they span,
    and how many of them share at least one valid S16 component (the
    rest are 'both sides present but no common valid component' — never
    polarization-comparable). Nothing is inferred from the global
    corrected_vs_always_wrong counts."""
    cells = [c for c in analysis.get("same_group_comparisons", [])
             if c.get("confusion_group") == "plastic_fish_vs_real_fish"
             and c.get("comparison_type") == "corrected_vs_always_wrong"]
    both_sides = [c for c in cells if c["comparable"]]
    with_common = [c for c in both_sides
                   if c.get("n_common_valid_components", 0) > 0]
    without_common = [c for c in both_sides
                      if c.get("n_common_valid_components", 0) == 0]
    return {
        "confusion_group": "plastic_fish_vs_real_fish",
        "comparison_type": "corrected_vs_always_wrong",
        "cells_total": len(cells),
        "cells_with_both_sides": len(both_sides),
        "groups_spanned": sorted({c["group"] for c in both_sides}),
        "n_groups_spanned": len({c["group"] for c in both_sides}),
        "cells_with_common_valid_component": len(with_common),
        "groups_with_common_valid_component": sorted(
            {c["group"] for c in with_common}),
        "cells_both_sides_but_no_common_valid_component": [
            {"true_class": c["true_class"], "group": c["group"],
             "outer_fold": c["outer_fold"], "bin": c["bin"],
             "left_support": c["left"]["support"],
             "right_support": c["right"]["support"]}
            for c in without_common],
        "cell_table": [
            {"true_class": c["true_class"], "group": c["group"],
             "outer_fold": c["outer_fold"], "bin": c["bin"],
             "corrected_support": c["left"]["support"],
             "always_wrong_support": c["right"]["support"],
             "n_common_valid_components":
                 c.get("n_common_valid_components", 0)}
            for c in both_sides],
        "reading_rules": (
            "both-sides-present is NOT polarization-comparable by "
            "itself; a cell counts as comparably measured only when the "
            "two sides share at least one valid S16 component. Supports "
            "of 1-3 per side are small-sample descriptions; common "
            "support never becomes a cross-group stable cue by itself."),
    }


def conditional_tables(ledger: list[dict], stats: dict,
                       bin_of: dict) -> dict:
    """Descriptive analysis over unique instances (kept for the full
    entry): subset overviews PLUS the real layered cells and same-group
    comparisons from build_conditional_analysis. No probe classifiers, no
    feature selection, no thresholds, no route inference."""
    return build_conditional_analysis(ledger, stats, bin_of)


# ---------------------------------------------------------------------------
# Postprocess: validation of the saved prior-diagnostic artifacts (the
# offline mode re-derives everything from the saved ledger; it never
# copies numbers, never rebuilds pixels and never renders new panels)

REQUIRED_LEDGER_FIELDS = ("sample_name", "source_frame", "split", "group",
                          "class_id", "class_name", "outer_fold", "row",
                          "s_valid", "blocks", "c_features", "s_features",
                          "r0", "paths", "rcs_vs_r0", "rcs_vs_rc")


def verify_saved_ledger(rows: list[dict]) -> list[str]:
    """Structural validation of the saved ledger before any reuse: unique
    sample names, required fields, C14/S16 widths (None = missing block
    entry), exactly 12 saved prediction paths per row and the missing-S
    semantics (an s_valid=False row carries at least one None; an
    s_valid=True row carries none in at least one block)."""
    problems = []
    names = [row.get("sample_name") for row in rows]
    if any(not isinstance(n, str) or not n for n in names):
        problems.append("a ledger row has no sample_name")
    if len(set(names)) != len(names):
        problems.append("duplicate sample_name in the saved ledger")
    for row in rows:
        name = row.get("sample_name", "<unnamed>")
        missing = [f for f in REQUIRED_LEDGER_FIELDS if f not in row]
        if missing:
            problems.append(f"{name}: missing fields {missing}")
            continue
        if len(row["c_features"]) != 14:
            problems.append(f"{name}: C14 width "
                            f"{len(row['c_features'])} != 14")
        if len(row["s_features"]) != 16:
            problems.append(f"{name}: S16 width "
                            f"{len(row['s_features'])} != 16")
        expected_paths = {f"{arm}|{seed}|{variant}"
                          for arm in ARMS for seed in TRAIN_SEEDS
                          for variant in VARIANTS}
        if set(row["paths"]) != expected_paths:
            problems.append(f"{name}: the saved prediction paths differ "
                            "from the 12-path contract")
        for key, entry in row["paths"].items():
            if set(entry) != {"logits", "prediction", "correct"} or \
                    len(entry.get("logits", [])) != 4:
                problems.append(f"{name}: path {key} is malformed")
        if row["s_valid"] and all(v is None for v in row["s_features"][:8]) \
                and all(v is None for v in row["s_features"][8:]):
            problems.append(
                f"{name}: s_valid=True but both S blocks are missing")
    return problems


def verify_saved_binning(rows: list[dict], binning_doc: dict) -> list[str]:
    """The saved bins must cover every instance and match a re-derivation
    from the saved per-fold edges under the existing build_donor_bins
    semantics."""
    from core.material_residual import build_donor_bins

    problems = []
    edges_by_fold = binning_doc.get("edges_by_fold") or {}
    saved_bins = binning_doc.get("bins") or {}
    names = {row["sample_name"] for row in rows}
    if set(saved_bins) != names:
        missing = sorted(names - set(saved_bins))[:5]
        extra = sorted(set(saved_bins) - names)[:5]
        problems.append(
            f"the saved bins do not cover exactly the ledger instances "
            f"(missing {len(names - set(saved_bins))} e.g. {missing}, "
            f"unknown {len(set(saved_bins) - names)} e.g. {extra})")
        return problems
    by_fold_values: dict[str, dict] = {}
    for row in rows:
        by_fold_values.setdefault(row["outer_fold"], {})[
            row["sample_name"]] = row
    for fold_id, values in sorted(by_fold_values.items()):
        edges = edges_by_fold.get(fold_id)
        if not edges:
            problems.append(f"{fold_id}: no saved edges")
            continue
        local_names = sorted(values)
        valid_ratio = np.asarray(
            [values[n]["c_features"][0] for n in local_names],
            dtype=np.float64)
        gray_mean = np.asarray(
            [values[n]["c_features"][3] for n in local_names],
            dtype=np.float64)
        recomputed = build_donor_bins(valid_ratio, gray_mean,
                                      edges=edges)
        for local, name in enumerate(local_names):
            if int(recomputed[local]) != int(saved_bins[name]):
                problems.append(
                    f"{name}: bin {saved_bins[name]} != recomputed "
                    f"{int(recomputed[local])} from the saved {fold_id} "
                    "edges")
    return problems


def verify_panel_coverage(panels_manifest: dict,
                          selection_manifest: dict,
                          changed_names: list[str],
                          panels_dir: Path) -> list[str]:
    """The saved panels must cover all changed instances, the controls
    must match the selection manifest, every panel file must exist with
    the recorded byte count and every rebuild check must be an EMPTY
    problem list. Only file bytes/stat are read — images are never
    decoded."""
    problems = []
    panels = panels_manifest.get("panels") or []
    by_name = {}
    for panel in panels:
        if panel["sample_name"] in by_name:
            problems.append(
                f"duplicate panel for {panel['sample_name']}")
        by_name[panel["sample_name"]] = panel
    missing = [n for n in changed_names if n not in by_name]
    if missing:
        problems.append(
            f"changed instances without a panel: {sorted(missing)[:5]} "
            f"({len(missing)} total)")
    changed_panels = [p for p in panels if p["kind"] == "changed"]
    if len(changed_panels) != len(changed_names):
        problems.append(
            f"{len(changed_panels)} changed panels != "
            f"{len(changed_names)} changed instances")
    controls = sorted(p["sample_name"] for p in panels
                      if p["kind"] == "control")
    selected_controls = sorted(
        item["sample_name"]
        for item in selection_manifest.get("control_panels", []))
    if controls != selected_controls:
        problems.append(
            "the saved control panels differ from the selection "
            "manifest")
    for panel in panels:
        path = Path(panels_dir) / panel.get("file", "")
        if not path.is_file():
            problems.append(
                f"panel file missing: {path}")
        elif path.stat().st_size != int(panel.get("bytes", -1)):
            problems.append(
                f"panel file bytes changed: {path} "
                f"({path.stat().st_size} != {panel.get('bytes')})")
        check = panel.get("rebuild_check") or {}
        if check.get("problems") != []:
            problems.append(
                f"{panel['sample_name']}: the saved rebuild check is not "
                f"an empty problem list ({check.get('problems')})")
    return problems
