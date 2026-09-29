"""4B systematic pipeline evaluation for the material_fusion runtime.

Fixed-protocol, read-only evaluation over the existing val split. Three
conditions are reported **separately** and never merged under one label:

- **A1** — the offline V5 NPZ entry (cross-check of the recorded scores);
- **A2** — the original stereo frames with GT instances injected through the
  engine's ``process_instances_detailed`` (the production stage code, with
  the generation's own band bbox and crop window);
- **B** — the real Model A detections.

Each condition reports a ``routed`` (production routing, fallback included)
and a ``pure`` (no-routing diagnostic) view; E1 is always the pure CMX
control and is never swapped for the independent gray fallback.

Statistics: the end-to-end Macro-F1 bootstrap resamples **whole capture
groups** with replacement, recomputes per-class TP/FP/FN (including missed
GT and unmatched predictions) and takes the four-class Macro-F1 difference
between E4 and E1 for each draw. The per-GT-instance correctness delta is
reported separately under its own name. With 4 val capture groups the
intervals are indicative, and val was used for model selection — this is
not independent evidence.

Independent-data evaluation is **not implemented**: ``--independent-data``
is refused before any output directory or model is created.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.gt_geometry import build_gt_geometry, gt_crop_windows, gt_instances  # noqa: E402
from core.material_fusion_dataset import CLASS_NAMES, MaterialFusionDataset  # noqa: E402
from core.material_mask_dataset import (  # noqa: E402
    admit_dataset_for_protocol,
    load_mask_sample,
)
from models.material_classifier import (  # noqa: E402
    build_material_classifier_for_candidate,
    load_material_runtime_config,
    material_input_tensors,
    select_candidate,
    verify_material_runtime_fixed_values,
)
from models.polar_fusion import read_manifest_split  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen protocol constants (decided before any model output is inspected).

IOU_THRESHOLD = 0.50
MATCH_RULE = (
    "class-agnostic one-to-one mask IoU >= 0.50 in the rectified left "
    "frame; maximize match count first, then total IoU; deterministic "
    "tie-break by (gt index, prediction index) over id-sorted inputs"
)
OPERATING_POINT = "fixed conf=0.25, NMS IoU=0.45, imgsz=640, match IoU>=0.50"
BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 2000
QUALITY_ATOL = 1e-6
QUALITY_RTOL = 0.0
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5
POLAR_QUALITY_BINS = (
    ("valid_ratio_0", 0.0, 0.0001),
    ("valid_ratio_low", 0.0001, 0.25),
    ("valid_ratio_mid", 0.25, 0.75),
    ("valid_ratio_high", 0.75, 1.0001),
)
MASK_SOURCE_GT_EVAL = "ground_truth_controlled_eval"
MASK_SOURCE_PREDICTED_EVAL = "predicted_model_a_eval"
ENGINEERING_VARIANTS = ("gray", "gray_gray_valid", "gray_valid", "gray_polar")
SEEDS = (2026, 2027, 2028)
ALIGNMENT_VARIANTS = (("gray", 2026), ("gray_gray_valid", 2026),
                      ("gray_valid", 2026), ("gray_polar", 2026))


# ---------------------------------------------------------------------------
# Small report helpers


def prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(
            f"output directory {path} already exists; evaluation reports are "
            "never overwritten — pass a fresh --output-dir"
        )
    path.mkdir(parents=True)
    return path


def write_json_bytes(directory: str | Path, name: str, payload: Any) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = Path(directory) / name
    path.write_bytes(text.encode("utf-8"))
    return path


def write_jsonl_bytes(directory: str | Path, name: str,
                      records: Sequence[Any]) -> Path:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True)
             for record in records]
    path = Path(directory) / name
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))
    return path


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Matching (class-agnostic, one-to-one, count-first then total IoU)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / union


def _max_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-total-weight bipartite assignment (Hungarian, O(n^3))."""
    n = len(weights)
    m = len(weights[0]) if n else 0
    if n == 0 or m == 0:
        return []
    if n > m:
        weights = [row + [0.0] * (n - m) for row in weights]
        m = n
    big = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [big] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = big
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = -weights[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = []
    for j in range(1, m + 1):
        if p[j]:
            i = p[j] - 1
            if weights[i][j - 1] > 0:
                assignment.append((i, j - 1))
    return assignment


def match_instances(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """One-to-one class-agnostic matching by mask IoU."""
    gt_masks = list(gt_masks)
    pred_masks = list(pred_masks)
    if not gt_masks or not pred_masks:
        return []
    gt_areas = [int((np.asarray(m) > 0).sum()) for m in gt_masks]
    pred_areas = [int((np.asarray(m) > 0).sum()) for m in pred_masks]
    gt_boxes = [_mask_bbox(m) for m in gt_masks]
    pred_boxes = [_mask_bbox(m) for m in pred_masks]
    weights: list[list[float]] = []
    for gi, gbox in enumerate(gt_boxes):
        row: list[float] = []
        for pi, pbox in enumerate(pred_boxes):
            iou = 0.0
            if _boxes_overlap(gbox, pbox) and _box_intersection_area(
                gbox, pbox
            ) >= 0.5 * max(gt_areas[gi], pred_areas[pi]):
                iou = mask_iou(gt_masks[gi], pred_masks[pi])
            row.append(1_000_000.0 + iou if iou >= iou_threshold else 0.0)
        weights.append(row)
    matches = [
        (gi, pi, weights[gi][pi] - 1_000_000.0)
        for gi, pi in _max_weight_assignment(weights)
    ]
    matches.sort(key=lambda entry: (entry[0], entry[1]))
    return matches


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if ys.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _box_intersection_area(a, b) -> int:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, dx) * max(0, dy)


# ---------------------------------------------------------------------------
# Metrics


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision/recall/F1 with one explicit zero-denominator rule.

    ``precision = tp / (tp + fp)`` and ``recall = tp / (tp + fn)`` are 0 when
    their denominator is 0; ``f1`` is 0 when ``precision + recall`` is 0.
    The class set is always the fixed four classes — a resample that misses
    a class scores it 0 rather than shrinking the set.
    """
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def per_class_counts(frame_records: Sequence[dict], num_classes: int) -> dict:
    """Accumulate the fixed metric inputs from per-frame records.

    Single source of truth for both the point estimates and the bootstrap:
    classification errors are an FN for the true class and an FP for the
    predicted class, missed GT is an FN, unmatched predictions are FPs, and
    a matched-but-unclassified instance is counted separately *and* as an
    end-to-end failure. Failed frames are counted and excluded from the
    evaluated frames, never hidden.
    """
    detection_tp = 0
    gt_total = pred_total = matched_total = 0
    missed = unmatched = unclassified = failed_frames = evaluated = 0
    iou_values: list[float] = []
    cond_tp = [0] * num_classes
    cond_fn = [0] * num_classes
    cond_fp = [0] * num_classes
    cond_correct = 0
    e2e_tp = [0] * num_classes
    e2e_fp = [0] * num_classes
    e2e_fn = [0] * num_classes
    confusion = [[0] * num_classes for _ in range(num_classes)]

    for record in frame_records:
        if record.get("frame_error"):
            failed_frames += 1
            continue
        evaluated += 1
        gt_classes = list(record["gt_classes"])
        pred_classes = list(record["pred_classes"])
        matches = [tuple(m) for m in record.get("matches", [])]
        unclassified_gt = set(record.get("unclassified_gt_indices", []))
        n_predictions = int(record.get("n_predictions", len(pred_classes)))
        gt_total += len(gt_classes)
        pred_total += n_predictions
        matched_total += len(matches)
        detection_tp += len(matches)
        unmatched += n_predictions - len(matches)
        missed += len(gt_classes) - len(matches)
        for _, _, iou in matches:
            iou_values.append(float(iou))
        matched_pred_indices = {pi for _, pi, _ in matches}
        for gi, pi, _ in matches:
            if gi in unclassified_gt:
                unclassified += 1
                e2e_fn[gt_classes[gi]] += 1
                continue
            true_class = gt_classes[gi]
            pred_class = pred_classes[pi]
            if pred_class == true_class:
                cond_tp[true_class] += 1
                cond_correct += 1
                e2e_tp[true_class] += 1
            else:
                cond_fn[true_class] += 1
                cond_fp[pred_class] += 1
                e2e_fn[true_class] += 1
                e2e_fp[pred_class] += 1
            confusion[true_class][pred_class] += 1
        for gi in range(len(gt_classes)):
            if not any(m[0] == gi for m in matches):
                e2e_fn[gt_classes[gi]] += 1
        for pi in range(n_predictions):
            if pi not in matched_pred_indices:
                pred_class = pred_classes[pi]
                if 0 <= pred_class < num_classes:
                    e2e_fp[pred_class] += 1

    return {
        "counts": {
            "frames": len(frame_records),
            "failed_frames": failed_frames,
            "evaluated_frames": evaluated,
            "gt_instances": gt_total,
            "predictions": pred_total,
            "matched": matched_total,
            "missed_gt": missed,
            "unmatched_predictions": unmatched,
            "unclassified_matched": unclassified,
        },
        "detection": {"tp": detection_tp, "fp": unmatched, "fn": missed},
        "e2e": {"tp": e2e_tp, "fp": e2e_fp, "fn": e2e_fn},
        "conditional": {"tp": cond_tp, "fp": cond_fp, "fn": cond_fn,
                        "correct": cond_correct},
        "confusion": confusion,
        "iou_values": iou_values,
    }


def macro_f1(counts: dict, num_classes: int) -> float:
    """Four-class Macro-F1 from per-class tp/fp/fn (fixed class set)."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return float(
        np.mean([_prf(tp[i], fp[i], fn[i])["f1"] for i in range(num_classes)])
    )


def aggregate_metrics(frame_records: Sequence[dict],
                      class_names: Sequence[str]) -> dict:
    """The fixed metric set built from :func:`per_class_counts`."""
    names = list(class_names)
    raw = per_class_counts(frame_records, len(names))
    counts = raw["counts"]
    detection = _prf(raw["detection"]["tp"], raw["detection"]["fp"],
                     raw["detection"]["fn"])
    detection["operating_point"] = OPERATING_POINT
    detection["note"] = (
        "fixed working point only; no AP sweep was implemented or claimed"
    )
    conditional = raw["conditional"]
    cond_total = sum(conditional["tp"]) + sum(conditional["fn"])
    per_class_conditional = {
        name: _prf(conditional["tp"][i], conditional["fp"][i],
                   conditional["fn"][i])
        for i, name in enumerate(names)
    }
    per_class_e2e = {
        name: _prf(raw["e2e"]["tp"][i], raw["e2e"]["fp"][i], raw["e2e"]["fn"][i])
        for i, name in enumerate(names)
    }
    iou_values = raw["iou_values"]
    return {
        "counts": counts,
        "detection": detection,
        "matched_iou": {
            "n": len(iou_values),
            "mean": float(np.mean(iou_values)) if iou_values else 0.0,
            "min": float(np.min(iou_values)) if iou_values else 0.0,
            "max": float(np.max(iou_values)) if iou_values else 0.0,
        },
        "conditional_classification": {
            "samples": cond_total,
            "accuracy": _safe_div(conditional["correct"], cond_total),
            "macro_f1": macro_f1(conditional, len(names)),
            "per_class": per_class_conditional,
            "confusion_matrix": raw["confusion"],
            "confusion_rows": "true class",
            "confusion_cols": "predicted class",
        },
        "end_to_end": {
            "macro_f1": macro_f1(raw["e2e"], len(names)),
            "per_class": per_class_e2e,
        },
    }


def candidate_pairs(candidates: Sequence[dict]) -> list[tuple[dict, dict]]:
    """Same-seed pairs: (E4, other) for every other variant and seed."""
    pairs = []
    by_key = {(c["variant"], c["seed"]): c for c in candidates}
    for seed in sorted({c["seed"] for c in candidates}):
        e4 = by_key.get(("gray_polar", seed))
        if e4 is None:
            continue
        for variant in ("gray", "gray_gray_valid", "gray_valid"):
            other = by_key.get((variant, seed))
            if other is not None:
                pairs.append((e4, other))
    return pairs


def e1_is_pure_control(candidate: dict) -> bool:
    """E1 (gray) is the pure CMX control — never the independent fallback."""
    return candidate["variant"] == "gray"


def paired_changes(base_records: Sequence[dict],
                   other_records: Sequence[dict]) -> dict:
    """Same-seed paired flips between two candidates over the same frames."""
    base_by_frame = {r["frame_id"]: r for r in base_records}
    fixed = broke = both_correct = both_wrong = 0
    changed_samples: list[dict] = []
    for record in other_records:
        base = base_by_frame.get(record["frame_id"])
        if base is None or base.get("frame_error") or record.get("frame_error"):
            continue
        for gt_id, other_ok in zip(record["gt_ids"], record["gt_correct"]):
            base_ok = (
                base["gt_correct"][base["gt_ids"].index(gt_id)]
                if gt_id in base["gt_ids"] else False
            )
            if base_ok and not other_ok:
                broke += 1
                changed_samples.append(
                    {"frame_id": record["frame_id"], "group": record["group"],
                     "gt_id": gt_id, "change": "broke"}
                )
            elif other_ok and not base_ok:
                fixed += 1
                changed_samples.append(
                    {"frame_id": record["frame_id"], "group": record["group"],
                     "gt_id": gt_id, "change": "fixed"}
                )
            elif base_ok and other_ok:
                both_correct += 1
            else:
                both_wrong += 1
    return {"fixed": fixed, "broke": broke, "both_correct": both_correct,
            "both_wrong": both_wrong, "changed": changed_samples}


# ---------------------------------------------------------------------------
# Bootstrap: a real end-to-end Macro-F1 interval (and the correctness delta)


def _grouped(records: Sequence[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record["group"]), []).append(record)
    return grouped


def end_to_end_macro_f1_bootstrap(
    e4_records: Sequence[dict],
    e1_records: Sequence[dict],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict:
    """Paired cluster bootstrap of the end-to-end Macro-F1 difference.

    Each draw samples whole capture groups **with replacement** (4 draws for
    the 4 val groups) and uses the *same* draw for both candidates, so the
    comparison is paired. A group drawn twice contributes its frames twice —
    duplicate draws are weighted, never de-duplicated by ``frame_id``. For
    each draw the per-class TP/FP/FN are re-accumulated over the drawn
    records (including missed GT and unmatched predictions) and the fixed
    four-class Macro-F1 is computed for both candidates; the recorded
    statistic is ``MacroF1(E4) - MacroF1(E1)``.

    The point estimate is the difference of the two Macro-F1 values on the
    **full** sample — never the bootstrap mean. With only 4 val capture
    groups, and val having been used for model selection, the interval is
    indicative and is not an independent significance test.
    """
    names = list(class_names)
    num_classes = len(names)
    groups = sorted(set(_grouped(e4_records)) | set(_grouped(e1_records)))
    if not groups:
        return {"metric": "end_to_end_macro_f1_delta", "point_estimate": 0.0,
                "ci_low": 0.0, "ci_high": 0.0, "resamples": 0,
                "valid_resamples": 0, "clusters": 0, "seed": seed,
                "classes": names, "note": "no records"}
    by_group_e4 = _grouped(e4_records)
    by_group_e1 = _grouped(e1_records)
    e4_full = per_class_counts(e4_records, num_classes)
    e1_full = per_class_counts(e1_records, num_classes)
    point = macro_f1(e4_full["e2e"], num_classes) - macro_f1(
        e1_full["e2e"], num_classes
    )
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        draws = [groups[rng.randrange(len(groups))] for _ in groups]
        sample_e4 = [r for g in draws for r in by_group_e4.get(g, [])]
        sample_e1 = [r for g in draws for r in by_group_e1.get(g, [])]
        if not sample_e4 or not sample_e1:
            continue
        delta = macro_f1(
            per_class_counts(sample_e4, num_classes)["e2e"], num_classes
        ) - macro_f1(
            per_class_counts(sample_e1, num_classes)["e2e"], num_classes
        )
        deltas.append(delta)
    deltas.sort()
    if deltas:
        low = deltas[max(0, int(0.025 * len(deltas)))]
        high = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)) - 1)]
    else:
        low = high = 0.0
    return {
        "metric": "end_to_end_macro_f1_delta",
        "point_estimate": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "e4_macro_f1": float(macro_f1(e4_full["e2e"], num_classes)),
        "e1_macro_f1": float(macro_f1(e1_full["e2e"], num_classes)),
        "resamples": int(resamples),
        "valid_resamples": len(deltas),
        "clusters": len(groups),
        "seed": int(seed),
        "classes": names,
        "interval": "percentile 2.5%-97.5% of the resampled deltas",
        "counts_e4": e4_full["counts"],
        "counts_e1": e1_full["counts"],
        "note": (
            f"only {len(groups)} val capture groups and val was used for "
            "model selection: the interval is indicative, not an independent "
            "significance test"
        ),
    }


def instance_correctness_delta_bootstrap(
    e4_records: Sequence[dict],
    e1_records: Sequence[dict],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Paired cluster bootstrap of the per-GT end-to-end correctness delta.

    This is **not** a Macro-F1 statistic: it averages a 0/1 correctness
    difference over GT instances and therefore ignores the unmatched
    predictions entirely. It is reported under its own name.
    """
    e1_by_frame = {r["frame_id"]: r for r in e1_records}
    values_a: list[float] = []
    values_b: list[float] = []
    groups: list[str] = []
    for record in e4_records:
        base = e1_by_frame.get(record["frame_id"])
        if base is None or base.get("frame_error"):
            continue
        for gi, gt_id in enumerate(record["gt_ids"]):
            if gi < len(base["gt_correct"]):
                values_a.append(1.0 if record["gt_correct"][gi] else 0.0)
                values_b.append(1.0 if base["gt_correct"][gi] else 0.0)
                groups.append(record["group"])
    result = paired_cluster_bootstrap(values_a, values_b, groups, seed=seed,
                                      resamples=resamples)
    result["metric"] = "paired_gt_instance_correctness_delta"
    return result


def paired_cluster_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    groups: Sequence[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Percentile bootstrap of the mean paired difference over groups."""
    if not values_a:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "resamples": 0, "clusters": 0, "note": "no samples"}
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    groups = np.asarray(groups)
    deltas = values_a - values_b
    unique_groups = sorted(set(groups.tolist()))
    by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = np.concatenate(
            [by_group[unique_groups[rng.randrange(len(unique_groups))]]
             for _ in unique_groups]
        )
        means.append(float(deltas[sample].mean()))
    means.sort()
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(means[int(0.025 * len(means))]),
        "ci_high": float(means[int(0.975 * len(means)) - 1]),
        "resamples": resamples,
        "clusters": len(unique_groups),
        "seed": seed,
        "note": (
            f"only {len(unique_groups)} capture groups; treat the interval as "
            "indicative, consecutive frames are not independent evidence"
        ),
    }


def build_bootstrap_section(
    e4_by_tag: dict[str, Sequence[dict]],
    e1_by_tag: dict[str, Sequence[dict]],
    *,
    seeds: Sequence[int] = SEEDS,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Both bootstrap families, each under the name of what it computes."""
    macro: dict[str, Any] = {}
    correctness: dict[str, Any] = {}
    for seed in seeds:
        e4 = e4_by_tag.get(f"gray_polar_s{seed}")
        e1 = e1_by_tag.get(f"gray_s{seed}")
        if e4 is None or e1 is None:
            continue
        macro[f"seed_{seed}"] = end_to_end_macro_f1_bootstrap(
            e4, e1, seed=BOOTSTRAP_SEED, resamples=resamples
        )
        correctness[f"seed_{seed}"] = instance_correctness_delta_bootstrap(
            e4, e1, seed=BOOTSTRAP_SEED, resamples=resamples
        )
    return {
        "end_to_end_macro_f1_e4_vs_e1": macro,
        "instance_correctness_delta": correctness,
        "note": (
            "the two sections are different statistics: the first recomputes "
            "the four-class Macro-F1 (including misses and false positives) "
            "on each resample, the second averages a per-GT 0/1 correctness "
            "difference and ignores unmatched predictions"
        ),
    }


# ---------------------------------------------------------------------------
# Unmatched events


def unmatched_gt_records(
    *,
    frame_id: str,
    group: str,
    gt_classes: Sequence[int],
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    matches: Sequence[tuple[int, int, float]],
) -> list[dict]:
    """Every GT instance that no prediction matched, with its best overlap.

    "Unmatched" is not "not detected at all". When the GT overlaps no
    prediction at all (every IoU is zero) no best prediction is invented:
    ``max_mask_iou`` stays 0.0, the best index and its matched-elsewhere
    flag stay null and ``no_positive_overlap`` is true. A frame with no
    predictions at all is reported separately with ``predictions_in_frame``
    = 0 and a null IoU. Only a strictly positive maximum selects a best
    prediction, with a stable (lowest index) tie-break.
    """
    matched_gt = {gi for gi, _, _ in matches}
    matched_pred = {pi for _, pi, _ in matches}
    records = []
    for gi, gt_class in enumerate(gt_classes):
        if gi in matched_gt:
            continue
        ious = [mask_iou(gt_masks[gi], pred) for pred in pred_masks]
        if not ious:
            max_iou: float | None = None
            best_index: int | None = None
            matched_elsewhere: bool | None = None
            no_positive_overlap: bool | None = None
        else:
            best_value = max(ious)
            max_iou = float(best_value)
            if best_value > 0.0:
                best_index = ious.index(best_value)   # stable lowest index
                matched_elsewhere = best_index in matched_pred
                no_positive_overlap = False
            else:
                best_index = None
                matched_elsewhere = None
                no_positive_overlap = True
        records.append({
            "frame_id": frame_id,
            "group": group,
            "gt_id": gi,
            "gt_class": int(gt_class),
            "gt_class_name": CLASS_NAMES[gt_class]
            if 0 <= gt_class < len(CLASS_NAMES) else None,
            "max_mask_iou": max_iou,
            "best_prediction_index": best_index,
            "best_prediction_matched_elsewhere": matched_elsewhere,
            "no_positive_overlap": no_positive_overlap,
            "predictions_in_frame": len(pred_masks),
            "note": (
                "not matched at IoU >= 0.50; a zero maximum overlap means "
                "no positive overlap with any prediction, not a merge or a "
                "duplicate signal, and no root cause is asserted"
            ),
        })
    return records


def summarise_fallback(
    instance_records: Sequence[dict],
    *,
    candidates: Sequence[tuple[str, int]],
) -> dict:
    """Fallback counts over exactly one condition, per candidate.

    Only ``layer == "B_model_a_predictions"`` and ``condition ==
    "routed"`` records count; A2, pure and other candidates are excluded.
    The numerator is the real ``used_fallback`` field, the denominator is
    every instance result in that condition (unclassified instances are
    kept and reported separately), the reason counts must sum to the
    numerator, and E1 must never use the fallback. Duplicate instance keys,
    missing fields and unknown identities are refused rather than silently
    deduplicated or read as "no fallback".
    """
    candidate_set = {(str(v), int(s)) for v, s in candidates}
    summary = {
        f"{variant}_s{seed}": {
            "variant": variant, "seed": int(seed), "instances": 0,
            "used_fallback_instances": 0, "unclassified_instances": 0,
            "reasons": {}, "fallback_ratio": 0.0,
            "level": "classifier-level (instance routing)",
        }
        for variant, seed in candidate_set
    }
    seen: set[tuple] = set()
    for row in instance_records:
        if row.get("layer") != "B_model_a_predictions":
            continue
        if row.get("condition") != "routed":
            continue
        for field in ("variant", "seed", "frame_id", "pred_instance_id",
                      "used_fallback"):
            if field not in row:
                raise ValueError(
                    f"instance record {row.get('frame_id')!r} is missing the "
                    f"{field!r} field; refusing to read it as 'no fallback'"
                )
        key = (str(row["variant"]), int(row["seed"]))
        if key not in candidate_set:
            raise ValueError(
                f"instance record identity {key} is not among the candidates; "
                "refusing an unknown variant/seed"
            )
        unique = (key, row["frame_id"], int(row["pred_instance_id"]))
        if unique in seen:
            raise ValueError(
                f"duplicate instance record {unique}; refusing to "
                "deduplicate silently"
            )
        seen.add(unique)
        entry = summary[f"{key[0]}_s{key[1]}"]
        entry["instances"] += 1
        if row.get("error"):
            entry["unclassified_instances"] += 1
        if bool(row["used_fallback"]):
            entry["used_fallback_instances"] += 1
            reason = str(row.get("fallback_reason") or "")
            if not reason:
                raise ValueError(
                    f"fallback instance {unique} has no reason"
                )
            entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1
        elif row.get("fallback_reason"):
            raise ValueError(
                f"instance {unique} records a fallback reason "
                f"{row['fallback_reason']!r} without used_fallback"
            )
    for name, entry in summary.items():
        if sum(entry["reasons"].values()) != entry["used_fallback_instances"]:
            raise ValueError(
                f"{name}: the reason counts do not sum to the fallback count"
            )
        entry["fallback_ratio"] = (
            entry["used_fallback_instances"] / entry["instances"]
            if entry["instances"] else 0.0
        )
        if entry["variant"] == "gray" and entry["used_fallback_instances"]:
            raise ValueError(
                f"{name}: E1 (gray) is the pure CMX control and must never "
                "use the independent gray fallback"
            )
    return summary


def unmatched_prediction_records(
    *,
    frame_id: str,
    group: str,
    pred_classes: Sequence[int],
    matches: Sequence[tuple[int, int, float]],
) -> list[dict]:
    matched_pred = {pi for _, pi, _ in matches}
    return [
        {"frame_id": frame_id, "group": group, "prediction_index": pi,
         "predicted_class": int(pred_classes[pi]),
         "predicted_class_name": CLASS_NAMES[pred_classes[pi]]
         if 0 <= pred_classes[pi] < len(CLASS_NAMES) else None,
         "note": "counted as a false positive for its predicted class"}
        for pi in range(len(pred_classes)) if pi not in matched_pred
    ]


# ---------------------------------------------------------------------------
# Stratification


def _stratify(value: float) -> str:
    for label, low, high in POLAR_QUALITY_BINS:
        if low <= value < high:
            return label
    return "valid_ratio_high"


def stratify_records(records: Sequence[dict],
                     key: Callable[[dict], str]) -> dict[str, list[dict]]:
    strata: dict[str, list[dict]] = {}
    for record in records:
        strata.setdefault(key(record), []).append(record)
    return strata


def quality_strata(records: Sequence[dict]) -> dict[str, dict]:
    """Conditional-classification metrics per polar-quality bin.

    Only **matched** instances have a predicted polarization quality, so the
    binning is explicitly conditional on a match; the unmatched GT are
    reported separately and are never given a fabricated predicted quality.
    """
    strata: dict[str, list[dict]] = {label: [] for label, _, _ in
                                     POLAR_QUALITY_BINS}
    for record in records:
        for gi, pi, _ in [tuple(m) for m in record.get("matches", [])]:
            sample = record.get("matched_quality", {}).get(str(gi))
            if sample is None:
                continue
            strata[_stratify(float(sample))].append({
                "gt_classes": [record["gt_classes"][gi]],
                "pred_classes": [record["pred_classes"][pi]],
                "matches": [[0, 0, 0.0]],
                "n_predictions": 1,
                "unclassified_gt_indices": (
                    [0] if gi in set(record.get("unclassified_gt_indices", []))
                    else []
                ),
                "frame_error": None,
            })
    return {
        label: {
            "conditional": aggregate_metrics(entries, CLASS_NAMES)[
                "conditional_classification"
            ] if entries else None,
            "support": len(entries),
            "conditional_on": "matched instances only; unmatched GT are "
                              "reported separately without a predicted quality",
        }
        for label, entries in strata.items()
    }


# ---------------------------------------------------------------------------
# A2 alignment (arrays, tensors and logits against V5 / the offline entry)


ARRAY_EXACT = "exact"
ARRAY_QUALITY = "quality"
ARRAY_LOGITS = "logits"


def _compare_arrays(actual: np.ndarray | None, expected: np.ndarray | None,
                    *, kind: str = ARRAY_EXACT) -> dict:
    """Compare one alignment field under its own contract.

    ``exact`` (images, masks, valid support, polar differentials and model
    input tensors) requires the same shape, the same dtype and numerically
    exact equality: no cast, re-interpolation, zeroing or rounding may hide
    a difference. ``quality`` allows atol 1e-6 with rtol 0, ``logits``
    atol = rtol 1e-5. Non-finite values never pass, and a missing input is
    UNAVAILABLE, never a pass.
    """
    if actual is None or expected is None:
        return {"status": "UNAVAILABLE", "exact_equal": False,
                "max_abs_err": None, "kind": kind, "reason": "missing input"}
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    result = {"kind": kind, "shape": list(actual.shape),
              "expected_shape": list(expected.shape),
              "dtype": str(actual.dtype), "expected_dtype": str(expected.dtype)}
    if actual.shape != expected.shape:
        result.update({"status": "FAILED", "exact_equal": False,
                       "max_abs_err": None, "reason": "shape mismatch"})
        return result
    if actual.size == 0:
        result.update({"status": "FAILED", "exact_equal": True,
                       "max_abs_err": None,
                       "reason": "empty required input"})
        return result
    if actual.dtype != expected.dtype:
        result.update({"status": "FAILED", "exact_equal": False,
                       "max_abs_err": None, "reason": "dtype mismatch"})
        return result
    # Finiteness is decided *before* any subtraction, so an Inf/NaN pair can
    # never produce a NaN error value or a RuntimeWarning.
    if not (bool(np.isfinite(actual).all())
            and bool(np.isfinite(expected).all())):
        result.update({"status": "FAILED", "exact_equal": False,
                       "max_abs_err": None, "reason": "non-finite values"})
        return result
    exact = bool(np.array_equal(actual, expected))
    if actual.dtype.kind in "fc":
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        max_err = float(diff.max()) if diff.size else 0.0
    else:
        max_err = 0.0 if exact else 1.0
    if kind == ARRAY_QUALITY:
        ok = bool(np.allclose(actual, expected, atol=QUALITY_ATOL,
                              rtol=QUALITY_RTOL))
    elif kind == ARRAY_LOGITS:
        ok = bool(np.allclose(actual, expected, atol=LOGIT_ATOL,
                              rtol=LOGIT_RTOL))
    else:
        ok = exact
    result.update({"status": "PASSED" if ok else "FAILED",
                   "exact_equal": exact, "max_abs_err": max_err})
    return result


def compare_tensors(actual, expected) -> dict:
    """Strict comparison for the model input tensors on the A2 path.

    Same shape, same dtype, all values finite and numerically exact
    equality are required; a missing tensor stays UNAVAILABLE and an empty
    required tensor is a failure, never evidence of alignment.
    """
    if actual is None or expected is None:
        return {"status": "UNAVAILABLE", "exact_equal": False,
                "max_abs_err": None, "reason": "missing input"}
    if not isinstance(actual, torch.Tensor) or not isinstance(
        expected, torch.Tensor
    ):
        return {"status": "FAILED", "exact_equal": False, "max_abs_err": None,
                "reason": "not a tensor"}
    if tuple(actual.shape) != tuple(expected.shape):
        return {"status": "FAILED", "exact_equal": False, "max_abs_err": None,
                "reason": "shape mismatch"}
    if actual.numel() == 0:
        return {"status": "FAILED", "exact_equal": True, "max_abs_err": None,
                "reason": "empty required input"}
    if actual.dtype != expected.dtype:
        return {"status": "FAILED", "exact_equal": False, "max_abs_err": None,
                "reason": "dtype mismatch"}
    if not (bool(torch.isfinite(actual).all())
            and bool(torch.isfinite(expected).all())):
        return {"status": "FAILED", "exact_equal": False, "max_abs_err": None,
                "reason": "non-finite values"}
    equal = bool(torch.equal(actual, expected))
    max_err = 0.0 if equal else float(
        (actual.to(torch.float64) - expected.to(torch.float64)).abs().max()
    )
    return {"status": "PASSED" if equal else "FAILED", "exact_equal": equal,
            "max_abs_err": max_err}


# ---------------------------------------------------------------------------
# Main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--evaluation-kind", default="retrospective",
                        choices=["retrospective"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--independent-data", default=None,
                        help="NOT implemented: passing this is refused before "
                             "any output directory or model is created")
    return parser.parse_args(argv)


def _build_record(*, frame_id: str, group: str, gt_classes, matches,
                  results, unclassified, matched_quality=None) -> dict:
    """One per-frame record built from a result list (pure or routed)."""
    pred_classes = [r.class_id if r.valid else -1 for r in results]
    matched_by_gt = {gi: pi for gi, pi, _ in matches}
    gt_correct = []
    for gi, gt_class in enumerate(gt_classes):
        if gi in matched_by_gt:
            result = results[matched_by_gt[gi]]
            gt_correct.append(bool(result.valid and result.class_id == gt_class))
        else:
            gt_correct.append(False)
    return {
        "frame_id": frame_id,
        "group": group,
        "gt_classes": list(gt_classes),
        "matches": [[gi, pi, float(iou)] for gi, pi, iou in matches],
        "pred_classes": pred_classes,
        "n_predictions": len(results),
        "unclassified_gt_indices": sorted(unclassified),
        "frame_error": None,
        "gt_ids": list(range(len(gt_classes))),
        "gt_correct": gt_correct,
        "matched_quality": matched_quality or {},
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.independent_data:
        print(
            "independent-data evaluation is not implemented in this round: "
            "no admission system exists that could bind a new dataset's "
            "provenance, capture-group isolation and annotation sources, so "
            "running it would either be meaningless or silently re-evaluate "
            "the old val split. Refusing before creating any output "
            "directory or loading any model.",
            file=sys.stderr,
        )
        return 2

    output_dir = prepare_output_dir(args.output_dir)

    config = load_material_runtime_config(args.config)
    verify_material_runtime_fixed_values(config)

    training_root = PROJECT_ROOT / config.training_data_root
    admission = admit_dataset_for_protocol(
        training_root,
        pooling_mode=config.pooling_mode,
        expected_fingerprint=config.data_fingerprint_v5,
        expected_parent_fingerprint=config.parent_data_fingerprint_v4,
        expected_mask_schema=config.mask_schema,
        expected_mask_resize_version=config.mask_resize_version,
    )
    candidates = [
        select_candidate(config, variant=variant, seed=seed)
        for variant in ENGINEERING_VARIANTS for seed in SEEDS
    ]
    checkpoint_shas = {
        (c.variant, c.seed): sha256_of(PROJECT_ROOT / c.checkpoint)
        for c in candidates
    }
    snapshot = {
        "evaluation_kind": args.evaluation_kind,
        "split": args.split,
        "generated_by": "scripts/eval_material_pipeline.py",
        "class_names": list(CLASS_NAMES),
        "conditions": {
            "A1": "offline V5 NPZ entry (recorded-score cross-check)",
            "A2": "original frames + GT instances through the production "
                  "stage code (band bbox = polygon extrema, crop window = "
                  "make_crop_window)",
            "B": "real Model A predictions",
            "pure": "no-routing diagnostic: every evaluable instance goes "
                    "through the CMX model, the fallback is not used",
            "routed": "production routing (polar-invalid -> independent gray)",
        },
        "model_identities": [
            {"variant": c.variant, "seed": c.seed, "checkpoint": c.checkpoint,
             "sha256": checkpoint_shas[(c.variant, c.seed)]}
            for c in candidates
        ],
        "model_a": {
            "path": config.model_a_path,
            "sha256": sha256_of(PROJECT_ROOT / config.model_a_path),
            "conf": config.model_a_conf, "iou": config.model_a_iou,
            "imgsz": config.model_a_imgsz,
        },
        "fallback": {
            "path": config.fallback_path,
            "sha256": sha256_of(PROJECT_ROOT / config.fallback_path),
            "identity": config.fallback_identity,
        },
        "material_model": {
            "imgsz": config.cmx_imgsz, "crop_padding": config.crop_padding,
            "preprocessing_version": config.preprocessing_version,
            "pooling_mode": config.pooling_mode,
            "mask_schema": config.mask_schema,
            "mask_resize_version": config.mask_resize_version,
            "upstream_commit": config.upstream_commit,
        },
        "stereo": dict(config.stereo),
        "polarization": dict(config.polar),
        "training_data_identity": {
            "root": str(training_root),
            "fingerprint": admission["audit_fingerprint"],
            "v4_parent": admission["v4_parent_fingerprint"],
        },
        "matching_rule": MATCH_RULE,
        "operating_point": OPERATING_POINT,
        "polar_quality_bins": [
            {"name": name, "low": low, "high": high}
            for name, low, high in POLAR_QUALITY_BINS
        ],
        "tolerances": {"quality_atol": QUALITY_ATOL, "quality_rtol": QUALITY_RTOL,
                       "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL},
        "alignment_variants": [
            {"variant": v, "seed": s} for v, s in ALIGNMENT_VARIANTS
        ],
        "fallback_rule": (
            "valid_ratio == 0, explicit polar_invalid or non-finite polar "
            "values route to the independent gray fallback (dual variants "
            "only); E1 is the pure control and never routes"
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
            "cluster": "capture group",
            "statistic": "end-to-end four-class Macro-F1 difference",
            "note": "4 val capture groups only; val was used for model "
                    "selection, so intervals are indicative",
        },
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu",
        },
        "sync_status": "unknown",
        "independent_data": "NOT_IMPLEMENTED",
    }
    write_json_bytes(output_dir, "protocol_snapshot.json", snapshot)

    classifiers = {
        (c.variant, c.seed): build_material_classifier_for_candidate(
            config, c, device=args.device
        )
        for c in candidates
    }

    from gui.inference_engine import DualStageInferenceEngine

    engine = DualStageInferenceEngine.from_material_config(
        args.config, variant="gray", seed=2026, device=args.device
    )

    # --- Layer A1: offline V5 NPZ entry ---
    paths, _labels, _names = read_manifest_split(training_root, args.split)
    from scripts.eval_material_fusion import evaluate_with_fallback

    layer_a1: dict[str, dict] = {}
    for candidate in candidates:
        classifier = classifiers[(candidate.variant, candidate.seed)]
        loader = torch.utils.data.DataLoader(
            MaterialFusionDataset(
                paths, variant=candidate.variant, imgsz=config.cmx_imgsz,
                with_instance_mask=bool(classifier.uses_instance_mask),
            ),
            batch_size=32, shuffle=False, num_workers=0,
        )
        result = evaluate_with_fallback(
            classifier.model, classifier.fallback, loader, classifier.device,
            num_classes=len(CLASS_NAMES), routing=classifier.is_dual,
        )
        layer_a1[f"{candidate.variant}_s{candidate.seed}"] = {
            "pure": result["pure_classifier"],
            "routed": result["routed"],
            "routing": result["routing"],
            "samples": result["samples"],
            "invalid_samples": result["invalid_samples"],
            "label_source": "V5 npz labels (offline entry)",
        }

    # --- frame sources ---
    split_csv = PROJECT_ROOT / "datasets/underwater_seg_binary_v2/split_assignment.csv"
    label_root = PROJECT_ROOT / "datasets/underwater_seg_v2/labels"
    with split_csv.open(encoding="utf-8") as handle:
        split_rows = [row for row in csv.DictReader(handle)
                      if row["split"] == args.split]
    with (training_root / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        manifest_rows = [row for row in csv.DictReader(handle)
                         if row["split"] == args.split]
    npz_by_key = {
        (row["source_frame"], int(row["object_index"])): training_root / row["npz_path"]
        for row in manifest_rows
    }
    import cv2

    from core.stereo_matching import to_gray_u8
    from scripts.verify_material_runtime import right_image_path

    # --- per-frame passes ---
    frames_jsonl: list[dict] = []
    instances_jsonl: list[dict] = []
    unmatched_gt_jsonl: list[dict] = []
    unmatched_pred_jsonl: list[dict] = []
    unclassified_jsonl: list[dict] = []
    alignment_rows: list[dict] = []
    records: dict[str, dict[str, list[dict]]] = {
        tag: {"a2_pure": [], "a2_routed": [], "b_pure": [], "b_routed": []}
        for tag in (f"{c.variant}_s{c.seed}" for c in candidates)
    }
    failed_frames: list[str] = []
    # The expected alignment set comes from the GT correspondence of the
    # verified val frames, never from how many rows the run produces.
    expected_alignment_keys: list[tuple[str, int]] = []

    for row in split_rows:
        stem = row["output_stem"]
        group = row["group_name"]
        left_path = PROJECT_ROOT / row["left_path"]
        right_path = right_image_path(left_path)
        if right_path is None:
            failed_frames.append(stem)
            for tag in records:
                for key in records[tag]:
                    records[tag][key].append({
                        "frame_id": stem, "group": group,
                        "frame_error": "right_image_missing",
                    })
            continue
        left = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        if left is None or right is None:
            failed_frames.append(stem)
            for tag in records:
                for key in records[tag]:
                    records[tag][key].append({
                        "frame_id": stem, "group": group,
                        "frame_error": "image_unreadable",
                    })
            continue
        left_gray = to_gray_u8(left)
        right_gray = to_gray_u8(right)
        height, width = left_gray.shape
        geometry = build_gt_geometry(
            label_root / args.split / f"{stem}.txt", width, height,
            padding=config.crop_padding,
        )
        gt_class_list = [item.class_id for item in geometry]
        expected_alignment_keys.extend(
            (stem, item.object_index) for item in geometry
        )
        gt_masks = [item.mask for item in geometry]

        b_result = engine.process_frame_detailed(
            left_gray, right_gray, sync_skew_ms=None, already_rectified=True,
            frame_id=stem,
        )
        a_result = engine.process_instances_detailed(
            left_gray, right_gray, gt_instances(geometry),
            sync_skew_ms=None, already_rectified=True, frame_id=stem,
            crop_windows=gt_crop_windows(geometry),
            gt_generation_contract=True,
        )
        b_inputs = list(b_result.material_inputs)
        b_instances = list(b_result.instances)
        a_inputs = list(a_result.material_inputs)

        matches = match_instances(gt_masks, [inst.mask for inst in b_instances])
        matched_quality = {
            str(gi): (float(b_inputs[pi].quality[0])
                      if pi < len(b_inputs) and b_inputs[pi] is not None else None)
            for gi, pi, _ in matches
        }
        for record in unmatched_gt_records(
            frame_id=stem, group=group, gt_classes=gt_class_list,
            gt_masks=gt_masks, pred_masks=[inst.mask for inst in b_instances],
            matches=matches,
        ):
            unmatched_gt_jsonl.append(record)

        # --- A2 alignment against V5 (arrays) and the offline entry ---
        for index, item in enumerate(geometry):
            npz_path = npz_by_key.get((stem, item.object_index))
            sample_input = a_inputs[index] if index < len(a_inputs) else None
            entry = {
                "frame_id": stem, "group": group,
                "object_index": item.object_index,
                "sample_name": f"{stem}_obj{item.object_index:03d}",
                "npz_path": str(npz_path) if npz_path else None,
                "band_bbox": list(item.band_bbox),
                "crop_window": list(item.crop_window),
                "class_id": item.class_id,
                "mask_source": MASK_SOURCE_GT_EVAL,
                "checks": {},
            }
            if npz_path is None or not npz_path.is_file() or sample_input is None:
                entry["status"] = "NO_NPZ"
                alignment_rows.append(entry)
                continue
            npz = load_mask_sample(npz_path)
            checks = entry["checks"]
            checks["gray"] = _compare_arrays(
                sample_input.gray, npz.gray, kind=ARRAY_EXACT)
            checks["instance_mask"] = _compare_arrays(
                sample_input.instance_mask, npz.instance_mask, kind=ARRAY_EXACT)
            checks["polar_valid"] = _compare_arrays(
                np.asarray(sample_input.polar_valid).astype(np.uint8),
                np.asarray(npz.polar_valid).astype(np.uint8), kind=ARRAY_EXACT)
            checks["signed_q"] = _compare_arrays(
                sample_input.signed_q, npz.signed_q, kind=ARRAY_EXACT)
            checks["abs_q"] = _compare_arrays(
                sample_input.abs_q, npz.abs_q, kind=ARRAY_EXACT)
            checks["quality"] = _compare_arrays(
                sample_input.quality, npz.quality, kind=ARRAY_QUALITY)
            for variant, seed in ALIGNMENT_VARIANTS:
                classifier = classifiers[(variant, seed)]
                runtime_tensors = material_input_tensors(
                    sample_input, config.cmx_imgsz, variant)
                dataset = MaterialFusionDataset(
                    [npz_path], variant=variant, imgsz=config.cmx_imgsz,
                    with_instance_mask=True,
                )
                offline = dataset[0]
                if variant == "gray":
                    off_rgb, off_quality, _cls, off_mask = offline
                    off_aux = None
                else:
                    off_rgb, off_aux, off_quality, _cls, off_mask = offline
                checks[f"{variant}_rgb"] = compare_tensors(
                    runtime_tensors[0], off_rgb)
                checks[f"{variant}_mask"] = compare_tensors(
                    runtime_tensors[2], off_mask)
                if off_aux is not None:
                    checks[f"{variant}_aux"] = compare_tensors(
                        runtime_tensors[1], off_aux)
                with torch.no_grad():
                    rgb = runtime_tensors[0].unsqueeze(0).to(classifier.device)
                    aux = (runtime_tensors[1].unsqueeze(0).to(classifier.device)
                           if runtime_tensors[1] is not None else None)
                    mask_t = runtime_tensors[2].unsqueeze(0).to(classifier.device)
                    runtime_logits = classifier.model(rgb, aux, mask_t).cpu()
                    off_rgb_t = off_rgb.unsqueeze(0).to(classifier.device)
                    off_aux_t = (off_aux.unsqueeze(0).to(classifier.device)
                                 if off_aux is not None else None)
                    off_mask_t = off_mask.unsqueeze(0).to(classifier.device)
                    offline_logits = classifier.model(
                        off_rgb_t, off_aux_t, off_mask_t).cpu()
                checks[f"{variant}_logits"] = _compare_arrays(
                    runtime_logits.numpy(), offline_logits.numpy(),
                    kind=ARRAY_LOGITS)
            statuses = {check.get("status") for check in checks.values()}
            if "FAILED" in statuses:
                entry["status"] = "FAILED"
            elif statuses - {"PASSED"}:
                entry["status"] = "PARTIAL"
            else:
                entry["status"] = "PASSED"
            alignment_rows.append(entry)

        # --- classification for every candidate ---
        for candidate in candidates:
            key = (candidate.variant, candidate.seed)
            tag = f"{candidate.variant}_s{candidate.seed}"
            classifier = classifiers[key]
            a_routed = classifier.predict_batch(a_inputs)
            a_pure = classifier.predict_pure_batch(a_inputs)
            b_routed = classifier.predict_batch(b_inputs)
            b_pure = classifier.predict_pure_batch(b_inputs)
            a_unclassified = {gi for gi, r in enumerate(a_routed) if not r.valid}
            b_unclassified = {gi for gi, pi, _ in matches
                              if not b_routed[pi].valid}
            records[tag]["a2_routed"].append(_build_record(
                frame_id=stem, group=group, gt_classes=gt_class_list,
                matches=[(gi, gi, 1.0) for gi in range(len(gt_class_list))],
                results=a_routed, unclassified=a_unclassified))
            records[tag]["a2_pure"].append(_build_record(
                frame_id=stem, group=group, gt_classes=gt_class_list,
                matches=[(gi, gi, 1.0) for gi in range(len(gt_class_list))],
                results=a_pure, unclassified={
                    gi for gi, r in enumerate(a_pure) if not r.valid}))
            records[tag]["b_routed"].append(_build_record(
                frame_id=stem, group=group, gt_classes=gt_class_list,
                matches=matches, results=b_routed,
                unclassified=b_unclassified, matched_quality=matched_quality))
            records[tag]["b_pure"].append(_build_record(
                frame_id=stem, group=group, gt_classes=gt_class_list,
                matches=matches, results=b_pure,
                unclassified={gi for gi, pi, _ in matches
                              if not b_pure[pi].valid},
                matched_quality=matched_quality))

            for gi, (gt_item, result) in enumerate(zip(geometry, a_routed)):
                instances_jsonl.append({
                    "frame_id": stem, "group": group, "layer": "A2_gt_controlled",
                    "variant": candidate.variant, "seed": candidate.seed,
                    "checkpoint_sha256": checkpoint_shas[key],
                    "condition": "routed", "gt_instance_id": gi,
                    "pred_instance_id": None, "match_iou": 1.0,
                    "gt_class": gt_item.class_id,
                    "pred_class": result.class_id if result.valid else None,
                    "raw_logits": list(result.raw_logits or []),
                    "mask_source": MASK_SOURCE_GT_EVAL,
                    "valid_ratio": float(a_inputs[gi].quality[0])
                    if gi < len(a_inputs) and a_inputs[gi] is not None else None,
                    "used_fallback": bool(result.used_fallback),
                    "fallback_reason": result.fallback_reason,
                    "model_identity": result.model_identity,
                    "error": None if result.valid else result.invalid_reason,
                })
            for gi, (gt_item, result) in enumerate(zip(geometry, a_pure)):
                instances_jsonl.append({
                    "frame_id": stem, "group": group, "layer": "A2_gt_controlled",
                    "variant": candidate.variant, "seed": candidate.seed,
                    "checkpoint_sha256": checkpoint_shas[key],
                    "condition": "pure", "gt_instance_id": gi,
                    "pred_instance_id": None, "match_iou": 1.0,
                    "gt_class": gt_item.class_id,
                    "pred_class": result.class_id if result.valid else None,
                    "raw_logits": list(result.raw_logits or []),
                    "mask_source": MASK_SOURCE_GT_EVAL,
                    "valid_ratio": float(a_inputs[gi].quality[0])
                    if gi < len(a_inputs) and a_inputs[gi] is not None else None,
                    "used_fallback": False,
                    "fallback_reason": "",
                    "model_identity": result.model_identity,
                    "error": None if result.valid else result.invalid_reason,
                })
            matched_by_pred = {pi: gi for gi, pi, _ in matches}
            for pi, result in enumerate(b_routed):
                matched_gt = matched_by_pred.get(pi)
                instances_jsonl.append({
                    "frame_id": stem, "group": group,
                    "layer": "B_model_a_predictions",
                    "variant": candidate.variant, "seed": candidate.seed,
                    "checkpoint_sha256": checkpoint_shas[key],
                    "condition": "routed", "gt_instance_id": matched_gt,
                    "pred_instance_id": pi,
                    "match_iou": next((iou for gi, gpi, iou in matches
                                       if gpi == pi), None),
                    "gt_class": gt_class_list[matched_gt]
                    if matched_gt is not None else None,
                    "pred_class": result.class_id if result.valid else None,
                    "raw_logits": list(result.raw_logits or []),
                    "mask_source": MASK_SOURCE_PREDICTED_EVAL,
                    "valid_ratio": float(b_inputs[pi].quality[0])
                    if pi < len(b_inputs) and b_inputs[pi] is not None else None,
                    "used_fallback": bool(result.used_fallback),
                    "fallback_reason": result.fallback_reason,
                    "model_identity": result.model_identity,
                    "error": None if result.valid else result.invalid_reason,
                })
            for pi, result in enumerate(b_pure):
                matched_gt = matched_by_pred.get(pi)
                instances_jsonl.append({
                    "frame_id": stem, "group": group,
                    "layer": "B_model_a_predictions",
                    "variant": candidate.variant, "seed": candidate.seed,
                    "checkpoint_sha256": checkpoint_shas[key],
                    "condition": "pure", "gt_instance_id": matched_gt,
                    "pred_instance_id": pi,
                    "match_iou": next((iou for gi, gpi, iou in matches
                                       if gpi == pi), None),
                    "gt_class": gt_class_list[matched_gt]
                    if matched_gt is not None else None,
                    "pred_class": result.class_id if result.valid else None,
                    "raw_logits": list(result.raw_logits or []),
                    "mask_source": MASK_SOURCE_PREDICTED_EVAL,
                    "valid_ratio": float(b_inputs[pi].quality[0])
                    if pi < len(b_inputs) and b_inputs[pi] is not None else None,
                    "used_fallback": False,
                    "fallback_reason": "",
                    "model_identity": result.model_identity,
                    "error": None if result.valid else result.invalid_reason,
                })
            for record in unmatched_prediction_records(
                frame_id=stem, group=group,
                pred_classes=[r.class_id if r.valid else -1 for r in b_routed],
                matches=matches,
            ):
                record["variant"] = candidate.variant
                record["seed"] = candidate.seed
                unmatched_pred_jsonl.append(record)
            for gi in sorted(b_unclassified):
                unclassified_jsonl.append({
                    "frame_id": stem, "group": group,
                    "variant": candidate.variant, "seed": candidate.seed,
                    "gt_instance_id": gi,
                    "gt_class": gt_class_list[gi],
                    "pred_instance_id": next(
                        (pi for g, pi, _ in matches if g == gi), None),
                    "reason": b_routed[next(
                        (pi for g, pi, _ in matches if g == gi), 0
                    )].invalid_reason if matches else "",
                    "note": "counted as an end-to-end failure for the true class",
                })

        frames_jsonl.append({
            "frame_id": stem, "group": group,
            "gt_instances": len(geometry),
            "predictions": len(b_instances),
            "matched": len(matches),
            "missed_gt": len(geometry) - len(matches),
            "unmatched_predictions": len(b_instances) - len(matches),
            "gt_classes": gt_class_list,
        })
        print(f"frame {stem}: {len(geometry)} gt, {len(b_instances)} predictions",
              file=sys.stderr)

    # --- aggregation ---
    per_candidate: dict[str, dict] = {}
    b_routed_records = {tag: records[tag]["b_routed"] for tag in records}
    for candidate in candidates:
        tag = f"{candidate.variant}_s{candidate.seed}"
        per_candidate[tag] = {
            "variant": candidate.variant,
            "seed": candidate.seed,
            "checkpoint_sha256": checkpoint_shas[(candidate.variant, candidate.seed)],
            "layer_a1": layer_a1[tag],
            "layer_a2": {
                "routed": aggregate_metrics(records[tag]["a2_routed"], CLASS_NAMES),
                "pure": aggregate_metrics(records[tag]["a2_pure"], CLASS_NAMES),
            },
            "layer_b": {
                "routed": aggregate_metrics(records[tag]["b_routed"], CLASS_NAMES),
                "pure": aggregate_metrics(records[tag]["b_pure"], CLASS_NAMES),
            },
        }

    paired = []
    by_key = {(c.variant, c.seed): c for c in candidates}
    for seed in SEEDS:
        e4_tag = f"gray_polar_s{seed}"
        for variant in ("gray", "gray_gray_valid", "gray_valid"):
            other_tag = f"{variant}_s{seed}"
            if other_tag not in records:
                continue
            changes = paired_changes(b_routed_records[other_tag],
                                     b_routed_records[e4_tag])
            changes["pair"] = f"{e4_tag}_vs_{other_tag}"
            changes["condition"] = "B routed, per-GT end-to-end correctness"
            paired.append(changes)

    bootstrap = build_bootstrap_section(
        {tag: b_routed_records[tag] for tag in b_routed_records},
        {tag: b_routed_records[tag] for tag in b_routed_records},
    )

    e4_s2026 = per_candidate["gray_polar_s2026"]["layer_b"]["routed"]
    stratification = {}
    for candidate in candidates:
        tag = f"{candidate.variant}_s{candidate.seed}"
        group_strata = stratify_records(
            records[tag]["b_routed"], lambda r: str(r["group"]))
        stratification[tag] = {
            "by_capture_group": {
                name: aggregate_metrics(entries, CLASS_NAMES)
                for name, entries in sorted(group_strata.items())
            },
            "by_polar_quality": quality_strata(records[tag]["b_routed"]),
        }

    conservation = {}
    for tag, entry in per_candidate.items():
        counts = entry["layer_b"]["routed"]["counts"]
        conservation[tag] = {
            "gt_conserved": counts["matched"] + counts["missed_gt"]
            == counts["gt_instances"],
            "prediction_conserved": counts["matched"]
            + counts["unmatched_predictions"] == counts["predictions"],
            "unclassified_is_part_of_misses": counts["unclassified_matched"]
            <= counts["matched"],
        }

    fallback_summary = summarise_fallback(
        instances_jsonl,
        candidates=[(c.variant, c.seed) for c in candidates],
    )
    alignment = evaluate_alignment(
        alignment_rows, expected_keys=expected_alignment_keys
    )

    real_fish = {
        "gt_support": e4_s2026["end_to_end"]["per_class"]["real_fish"]["tp"]
        + e4_s2026["end_to_end"]["per_class"]["real_fish"]["fn"],
        "matched_and_correct": e4_s2026["conditional_classification"][
            "per_class"]["real_fish"]["tp"],
        "note": (
            "only a few real-fish GT instances reach the fixed matching "
            "standard; 'unmatched' does not mean 'not detected at all' and "
            "the filename is never used as label evidence"
        ),
    }

    if alignment["status"] == "FAILED":
        final_status = "FAILED"
    elif alignment["status"] == "PARTIAL" or failed_frames:
        final_status = "PARTIAL"
    elif not all(
        all(entry.values()) for entry in conservation.values()
    ):
        final_status = "FAILED"
    else:
        final_status = "COMPLETED"

    summary = {
        "status": final_status,
        "alignment": alignment,
        "evaluation_kind": args.evaluation_kind,
        "split": args.split,
        "frames": len(split_rows),
        "failed_frames": failed_frames,
        "per_candidate": per_candidate,
        "paired_changes": paired,
        "bootstrap": bootstrap,
        "stratification": stratification,
        "count_conservation": conservation,
        "fallback_summary": fallback_summary,
        "unmatched_gt_count": len(unmatched_gt_jsonl),
        "unmatched_prediction_count": len(unmatched_pred_jsonl),
        "unclassified_count": len(unclassified_jsonl),
        "real_fish": real_fish,
        "conclusions": {
            "independent_generalization": "INSUFFICIENT_EVIDENCE",
            "polarization_benefit": "NOT_ESTABLISHED_BY_THIS_EVALUATION",
            "real_time": "NOT_MEASURED_HERE",
            "deployment": "NO_PRE_APPROVED_CRITERIA_NOT_ASSESSED",
            "note": "val was used for model selection and is retrospective "
                    "here; the 4 val capture groups limit every interval",
        },
    }
    write_json_bytes(output_dir, "summary.json", summary)
    (output_dir / "summary.md").write_bytes(
        _render_summary_md(summary).encode("utf-8")
    )
    write_jsonl_bytes(output_dir, "frames.jsonl", frames_jsonl)
    write_jsonl_bytes(output_dir, "instances.jsonl", instances_jsonl)
    write_jsonl_bytes(output_dir, "paired_changes.jsonl", paired)
    write_jsonl_bytes(output_dir, "unmatched_gt.jsonl", unmatched_gt_jsonl)
    write_jsonl_bytes(output_dir, "unmatched_predictions.jsonl",
                      unmatched_pred_jsonl)
    write_jsonl_bytes(output_dir, "unclassified.jsonl", unclassified_jsonl)
    write_jsonl_bytes(output_dir, "a2_alignment.jsonl", alignment_rows)
    write_json_bytes(output_dir, "a2_alignment_summary.json", alignment)
    manifest = {
        "files": {
            p.name: sha256_of(p)
            for p in sorted(output_dir.iterdir()) if p.is_file()
            and p.name != "artifact_manifest.json"
        },
        "index_file": "artifact_manifest.json",
        "index_note": "index of this directory; does not cover itself",
    }
    write_json_bytes(output_dir, "artifact_manifest.json", manifest)
    print(f"evaluation summary: {output_dir / 'summary.json'}")
    print(f"status: {final_status} | a2 alignment: {alignment['status']}")
    return 0 if final_status == "COMPLETED" else 1


def check_kind(name: str) -> str:
    """The comparison contract a recorded check name belongs to."""
    if name == "quality":
        return ARRAY_QUALITY
    if name.endswith("_logits"):
        return ARRAY_LOGITS
    return ARRAY_EXACT


def required_alignment_checks(
    variants: Sequence[tuple[str, int]] = ALIGNMENT_VARIANTS,
) -> list[str]:
    """The checks every aligned sample must carry.

    The single-stream ``gray`` variant has no auxiliary input by protocol,
    so ``gray_aux`` is not required: that is a defined "not applicable",
    not a licence to skip arbitrary fields.
    """
    checks = ["gray", "instance_mask", "polar_valid", "signed_q", "abs_q",
              "quality"]
    for variant, _seed in variants:
        checks += [f"{variant}_rgb", f"{variant}_mask", f"{variant}_logits"]
        if variant != "gray":
            checks.append(f"{variant}_aux")
    return checks


def evaluate_alignment(
    rows: Sequence[dict],
    *,
    expected_keys: Sequence[tuple[str, int]],
    variants: Sequence[tuple[str, int]] = ALIGNMENT_VARIANTS,
) -> dict:
    """The strict alignment verdict over an expected sample set.

    ``expected_keys`` is the ``(frame_id, object_index)`` set the GT
    correspondence requires, derived from the verified manifest rather than
    from however many rows the run produced. Only a non-empty, fully
    covered expected set whose every required check passed is PASSED; a
    mismatch, duplicate or extra record is FAILED; missing coverage, a GT
    instance without an NPZ, or a required check that could not be
    completed is PARTIAL.
    """
    required = required_alignment_checks(variants)
    expected = [f"{frame}|{index}" for frame, index in expected_keys]
    expected_set = set(expected)
    seen: list[str] = []
    mismatches: list[dict] = []
    unfinished: list[dict] = []
    no_npz: list[str] = []
    unknown_status: list[str] = []
    unknown_check_status: list[str] = []
    unverifiable_relative: list[str] = []
    inner_failures = 0
    for row in rows:
        key = f"{row.get('frame_id')}|{row.get('object_index')}"
        seen.append(key)
        status = row.get("status")
        if status not in ("PASSED", "FAILED", "PARTIAL", "NO_NPZ",
                          "UNAVAILABLE"):
            unknown_status.append(key)
        if status == "NO_NPZ":
            no_npz.append(key)
            continue
        checks = row.get("checks") or {}
        missing_checks = [name for name in required if name not in checks]
        if missing_checks:
            mismatches.append({"key": key, "field": "checks",
                               "reason": f"missing checks {missing_checks}"})
        for name, check in checks.items():
            check_status = (check or {}).get("status")
            if check_status not in ("PASSED", "FAILED", "UNAVAILABLE",
                                    "PARTIAL"):
                unknown_check_status.append(f"{key}:{name}={check_status!r}")
                mismatches.append({
                    "key": key, "field": name,
                    "reason": f"unknown check status {check_status!r}",
                })
                continue
            if check_status == "FAILED":
                mismatches.append({
                    "key": key, "field": name,
                    "max_abs_err": (check or {}).get("max_abs_err"),
                    "reason": (check or {}).get("reason", "value mismatch"),
                })
                continue
            if check_status != "PASSED":
                unfinished.append({"key": key, "field": name,
                                   "status": check_status})
                continue
            # A recorded PASSED must not contradict its own fields. Exact
            # comparisons require exact equality and a zero error; the
            # tolerated kinds may differ, but only inside their tolerance.
            # The comparison protocol is fixed by the field name; a
            # recorded kind may never downgrade an exact field to a
            # tolerance, and an unknown kind is not a licence either.
            protocol_kind = check_kind(name)
            recorded_kind = (check or {}).get("kind")
            if recorded_kind is not None and recorded_kind != protocol_kind:
                mismatches.append({
                    "key": key, "field": name,
                    "reason": f"recorded kind {recorded_kind!r} conflicts with "
                              f"the {protocol_kind!r} protocol for this field",
                })
                continue
            kind = protocol_kind
            max_err = (check or {}).get("max_abs_err")
            exact_equal = (check or {}).get("exact_equal")
            contradiction = None
            if kind == ARRAY_EXACT:
                if exact_equal is False:
                    contradiction = "recorded PASSED but exact_equal is false"
                elif max_err is not None and float(max_err) != 0.0:
                    contradiction = (
                        f"recorded PASSED but max_abs_err is {max_err}"
                    )
            elif kind == ARRAY_QUALITY:
                if max_err is not None and float(max_err) > QUALITY_ATOL:
                    contradiction = (
                        f"recorded PASSED but max_abs_err {max_err} exceeds "
                        f"the quality tolerance {QUALITY_ATOL}"
                    )
            else:
                # logits: the contract is
                #   abs(a - e) <= atol + rtol * abs(e)
                # A historical row stores only max_abs_err, so an error
                # above atol alone cannot prove a violation. The recorded
                # verdict is kept and the missing evidence is stated rather
                # than invented.
                if max_err is not None and float(max_err) > LOGIT_ATOL:
                    unverifiable_relative.append(
                        f"{key}:{name} (max_abs_err {max_err} > atol "
                        f"{LOGIT_ATOL}; the rtol-inclusive tolerance cannot "
                        "be recomputed from the stored fields)"
                    )
            if contradiction:
                mismatches.append({"key": key, "field": name,
                                   "max_abs_err": max_err,
                                   "reason": contradiction})
        if status == "PASSED" and any(
            (check or {}).get("status") == "FAILED" for check in checks.values()
        ):
            inner_failures += 1
        if status == "FAILED" and not any(
            (check or {}).get("status") == "FAILED" for check in checks.values()
        ):
            # The row itself says it failed even though its individual
            # checks look clean: a recorded failure is honoured, never
            # silently upgraded by re-reading only the checks.
            mismatches.append({"key": key, "field": "row_status",
                               "reason": "the row is recorded as FAILED"})
        if status in ("PARTIAL", "UNAVAILABLE"):
            unfinished.append({"key": key, "field": "row_status",
                               "status": status})
    counts: dict[str, int] = {}
    for key in seen:
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    extra = sorted(set(seen) - expected_set)
    missing = sorted(expected_set - set(seen))
    if not expected:
        status = "PARTIAL"
    elif mismatches or duplicates or extra or unknown_status or inner_failures:
        status = "FAILED"
    elif missing or no_npz or unfinished:
        status = "PARTIAL"
    else:
        status = "PASSED"
    return {
        "status": status,
        "rows": len(rows),
        "expected_samples": len(expected),
        "covered_samples": len(expected_set & set(seen)),
        "coverage": {"missing": missing, "extra": extra,
                     "duplicates": duplicates, "no_npz": no_npz},
        "mismatches": mismatches[:50],
        "mismatch_count": len(mismatches),
        "unfinished": unfinished[:50],
        "unfinished_count": len(unfinished),
        "unknown_status": unknown_status,
        "unknown_check_status": unknown_check_status,
        "unverifiable_relative_tolerance": unverifiable_relative,
        "required_checks": required,
        "tolerances": {"exact": "numerically exact, same shape and dtype",
                       "quality_atol": QUALITY_ATOL,
                       "quality_rtol": QUALITY_RTOL,
                       "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL},
        "note": "coverage is judged against the expected sample set derived "
                "from the verified manifest, not against the number of rows "
                "the run produced",
    }


def _alignment_summary(rows: Sequence[dict]) -> dict:
    counts: dict[str, dict] = {}
    mismatches: list[dict] = []
    for row in rows:
        status = row.get("status", "UNAVAILABLE")
        counts[status] = counts.get(status, 0) + 1
        for field, check in (row.get("checks") or {}).items():
            field_status = check.get("status")
            entry = counts.setdefault(f"field:{field}", {})
            entry[field_status] = entry.get(field_status, 0) + 1
            if field_status == "FAILED":
                mismatches.append({
                    "frame_id": row["frame_id"],
                    "object_index": row["object_index"],
                    "field": field,
                    "max_abs_err": check.get("max_abs_err"),
                    "reason": check.get("reason", "value mismatch"),
                })
    failed = counts.get("FAILED", 0)
    return {
        "status": "PASSED" if failed == 0 and not mismatches else "FAILED",
        "rows": len(rows),
        "status_counts": counts,
        "mismatches": mismatches[:50],
        "mismatch_count": len(mismatches),
        "tolerances": {"quality_atol": QUALITY_ATOL, "quality_rtol": QUALITY_RTOL,
                       "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL},
        "note": "A2 arrays are compared exactly against the V5 NPZ; the four "
                "seed-2026 input tensors exactly against the offline dataset; "
                "logits within the pre-fixed tolerances against the offline "
                "forward. Tolerances were never widened.",
    }


def _render_summary_md(summary: dict) -> str:
    lines = [
        "# 4B retrospective pipeline evaluation (val split)",
        "",
        f"- status: {summary['status']} "
        f"(A2 alignment: {summary['alignment']['status']})",
        f"- frames: {summary['frames']} (failed: {len(summary['failed_frames'])})",
        f"- candidates: {len(summary['per_candidate'])} (all 12, no re-selection)",
        f"- unmatched GT: {summary['unmatched_gt_count']}, "
        f"unmatched predictions: {summary['unmatched_prediction_count']}, "
        f"unclassified: {summary['unclassified_count']}",
        "",
        "## Layer B routed (real Model A predictions)",
        "",
        "| candidate | matched | missed | unmatched preds | e2e Macro-F1 | cond. acc |",
        "|---|---|---|---|---|---|",
    ]
    for tag, entry in sorted(summary["per_candidate"].items()):
        routed = entry["layer_b"]["routed"]
        counts = routed["counts"]
        lines.append(
            f"| {tag} | {counts['matched']} | {counts['missed_gt']} | "
            f"{counts['unmatched_predictions']} | "
            f"{routed['end_to_end']['macro_f1']:.4f} | "
            f"{routed['conditional_classification']['accuracy']:.4f} |"
        )
    lines += ["", "## Same-seed paired changes (E4 vs others, B routed)",
              "", "| pair | fixed | broke | both correct | both wrong |",
              "|---|---|---|---|---|"]
    for change in summary["paired_changes"]:
        lines.append(
            f"| {change['pair']} | {change['fixed']} | {change['broke']} | "
            f"{change['both_correct']} | {change['both_wrong']} |"
        )
    lines += ["", "## End-to-end Macro-F1 bootstrap (E4 - E1)", "",
              "| seed | point estimate | 95% interval | clusters |",
              "|---|---|---|---|"]
    for seed, entry in sorted(
        summary["bootstrap"]["end_to_end_macro_f1_e4_vs_e1"].items()
    ):
        lines.append(
            f"| {seed} | {entry['point_estimate']:+.4f} | "
            f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] | "
            f"{entry['clusters']} |"
        )
    lines += ["", "## Honest conclusions", ""]
    for key, value in summary["conclusions"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
