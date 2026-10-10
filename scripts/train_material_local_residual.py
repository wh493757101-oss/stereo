"""Local interior/band residual experiment (material_polar_local...v1).

The frozen residual correction model (zero-initialized bias-free linear
head, ``delta = valid_s * 0.5 * tanh(W x)``, ``CE + 0.01*mean(delta^2)``,
full-batch AdamW, epoch-0 candidate, inner_val selection, outer_eval
evaluated once) is applied to the LOCAL polarization representation: the
original measurement region partitioned into the frozen interior and
contour band, each with the existing eight S statistics plus per-region
C controls (see ``core/polar_local_features.py`` and the pre-registered
protocol in ``docs/experiments/material_polar_local_interior_band_v1.md``).

Fixed inputs: the frozen three-fold protocol, the frozen global cache
(P0 source, verified through the original admission chain) and the
independent local feature cache built by
``scripts/prepare_material_local_features.py``. G, P0, fold membership,
the budget and the loss are untouched; only the residual input width
grows (14 -> 30).

Modes: ``--dry-run`` (read-only admission + report) and ``--fit``
(explicit). Exit codes: 0 ok, 1 refused, 2 mismatch, 3 blocked. The real
fit is executed by the user via
``scripts/manual_material_local_residual.ps1`` — never by the agent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from core.material_probe import (  # noqa: E402
    Standardizer,
    accuracy,
    compare_module_states,
    macro_f1,
    snapshot_module_state,
)
from core.material_residual import (  # noqa: E402
    SHUFFLE_SEEDS,
    build_donor_bins,
    build_residual_head,
    compute_delta,
    p0_head_from_payload,
    p0_logits_of,
    residual_logits,
    shuffle_assignment,
    train_residual_head,
)
from core.polar_local_features import (  # noqa: E402
    BAND_BLOCK,
    INTERIOR_BLOCK,
    LOCAL_C_WIDTH,
    LOCAL_FEATURE_DEFINITION_VERSION,
    LOCAL_REGION_RULE,
    LOCAL_S_WIDTH,
    LOCAL_TOTAL_DIM,
    assemble_local_residual_inputs,
    fit_local_scaler,
    verify_local_cache,
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_MISMATCH = 2
EXIT_BLOCKED = 3

VARIANTS = ("RC", "RCS")
VARIANT_ORDER = ("R0", "RC", "RCS")
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")


class LocalMismatch(RuntimeError):
    """A legal, finite saved value disagrees with a recomputation."""


class LocalRefused(RuntimeError):
    """An identity, structure, binding or input-refusal condition."""


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True, allow_nan=False) + "\n")


def _clean(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _metrics(labels: list, predictions: list) -> dict:
    per_class_recall = {}
    for index, name in enumerate(CLASS_NAMES):
        total = sum(1 for label in labels if label == index)
        hits = sum(1 for label, prediction in zip(labels, predictions)
                   if label == index and prediction == index)
        per_class_recall[name] = _clean(hits / total) if total else None
    confusion = [[0] * 4 for _ in range(4)]
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1
    return {
        "macro_f1": _clean(macro_f1(labels, predictions)),
        "accuracy": _clean(accuracy(labels, predictions)),
        "confusion_matrix": confusion,
        "per_class_recall": per_class_recall,
    }


def paired_changes(base_predictions: list, other_predictions: list,
                   labels: list) -> dict:
    fixed = broke = both_correct = both_wrong = 0
    for base, other, label in zip(base_predictions, other_predictions,
                                  labels):
        base_ok = base == label
        other_ok = other == label
        if base_ok and other_ok:
            both_correct += 1
        elif not base_ok and not other_ok:
            both_wrong += 1
        elif other_ok:
            fixed += 1
        else:
            broke += 1
    return {"fixed": fixed, "broke": broke, "both_correct": both_correct,
            "both_wrong": both_wrong}


def _p0_design_for_rows(rows, cache, scaler: Standardizer) -> np.ndarray:
    """The frozen 270-wide P0 design from the GLOBAL cache (read-only).

    The frozen P0 uses ONLY the standardized G columns: the scaler
    transform is followed by ``assemble_design_matrix(..., "P0")``, which
    zeroes the C/S slots exactly. Feeding the full standardized G+C+S
    into the frozen head was the identity bug this entry fixes.
    """
    from core.material_probe import (
        assemble_design_matrix,
        CONTROL_FEATURE_KEYS,
        POLAR_STAT_KEYS,
    )

    g = cache["g_features"][rows]
    c = cache["c_features"][rows]
    s = cache["s_features"][rows]
    s_valid = cache["s_valid"][rows].astype(bool)
    width = g.shape[1] + c.shape[1] + s.shape[1]
    column_valid = np.ones((len(rows), width), dtype=bool)
    column_valid[:, g.shape[1] + c.shape[1]:] = s_valid[:, None]
    raw = np.concatenate([g, c, s], axis=1)
    standardized = scaler.transform(raw, column_valid)
    g_dim = g.shape[1]
    design = assemble_design_matrix(
        standardized[:, :g_dim],
        standardized[:, g_dim:g_dim + len(CONTROL_FEATURE_KEYS)],
        standardized[:, g_dim + len(CONTROL_FEATURE_KEYS):],
        "P0")
    return design.astype(np.float32)


def verify_p0_base_identity(*, fold_id: str, p0_fit_dir: Path,
                            admission: dict, global_cache: dict,
                            local_cache: dict, manifest_samples: list,
                            authority: dict, log) -> dict:
    """Verify the ACTUAL runtime P0 base against the frozen oracle.

    The oracle is ``<P0 fit>/models/<fold>_P0/outer_eval_predictions.jsonl``
    (byte-checked against the admission's recorded file SHA). Checks, in
    order: oracle fields, order, duplicates, self-consistency
    (prediction == logits.argmax, class/group identity), the disabled
    C/S slots of the train/inner_val/outer_eval designs (exactly zero),
    and the runtime logits (absolute tolerance ``LOGIT_ATOL``, rtol 0)
    with exact prediction agreement. A mismatch refuses the run before
    any residual head is built.
    """
    import hashlib

    import torch

    from scripts.verify_p0_baseline import LOGIT_ATOL

    fold_result = admission["fold_results"][fold_id]
    oracle_path = p0_fit_dir / "models" / f"{fold_id}_P0" / \
        "outer_eval_predictions.jsonl"
    if not oracle_path.is_file():
        raise LocalRefused(f"P0 oracle file missing: {oracle_path}")
    recorded_sha = fold_result["files"]["outer_eval_predictions.jsonl"]
    actual_sha = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    if recorded_sha != actual_sha:
        raise LocalMismatch(
            f"fold {fold_id}: the P0 oracle file does not match the "
            f"admitted bytes ({actual_sha} != {recorded_sha})")
    oracle = [json.loads(line) for line in
              oracle_path.read_text(encoding="utf-8").splitlines()]
    names = [r["sample_name"] for r in oracle]
    if len(names) != len(set(names)):
        raise LocalMismatch(
            f"fold {fold_id}: the P0 oracle contains duplicate samples")
    rows = fold_parts(manifest_samples, authority, fold_id)["outer_eval"]
    expected_names = [manifest_samples[row]["sample_name"]
                      for row in rows]
    if names != expected_names:
        raise LocalMismatch(
            f"fold {fold_id}: the P0 oracle rows do not match the "
            "authoritative outer_eval order/set")
    oracle_logits = np.asarray([r["logits"] for r in oracle],
                               dtype=np.float64)
    if oracle_logits.shape != (len(rows), 4) or \
            not np.isfinite(oracle_logits).all():
        raise LocalMismatch(
            f"fold {fold_id}: the P0 oracle logits are not a finite "
            "(n, 4) matrix")
    for index, record in enumerate(oracle):
        if record["prediction"] != int(oracle_logits[index].argmax()):
            raise LocalMismatch(
                f"fold {fold_id}: oracle prediction != logits.argmax at "
                f"{record['sample_name']}")
        if record["class_id"] != int(global_cache["class_id"][rows[index]]):
            raise LocalMismatch(
                f"fold {fold_id}: oracle class_id mismatch at "
                f"{record['sample_name']}")
        if record["group"] != manifest_samples[rows[index]]["group"]:
            raise LocalMismatch(
                f"fold {fold_id}: oracle group mismatch at "
                f"{record['sample_name']}")

    checkpoint = p0_fit_dir / "models" / f"{fold_id}_P0" / "model_state.pt"
    scaler_path = p0_fit_dir / "models" / f"{fold_id}_P0" / "scaler.json"
    p0_head = p0_head_from_payload(
        torch.load(checkpoint, weights_only=False, map_location="cpu"))
    p0_scaler = Standardizer.from_dict(load_json(scaler_path))
    g_dim = global_cache["g_features"].shape[1]
    disabled_slot_max_abs = {}
    for split, part in (("train", "inner_train"),
                        ("inner_val", "inner_val"),
                        ("outer_eval", "outer_eval")):
        part_rows = fold_parts(manifest_samples, authority,
                               fold_id)[part]
        design = _p0_design_for_rows(part_rows, global_cache, p0_scaler)
        disabled_slot_max_abs[split] = _clean(
            float(np.abs(design[:, g_dim:]).max()))
        if disabled_slot_max_abs[split] != 0.0:
            raise LocalMismatch(
                f"fold {fold_id}: the runtime P0 design leaves the C/S "
                f"slots unmasked for {split} (max abs "
                f"{disabled_slot_max_abs[split]}); the frozen P0 reads G "
                "only")
    eval_design = _p0_design_for_rows(rows, global_cache, p0_scaler)
    runtime_logits = p0_logits_of(p0_head, eval_design)
    errors = np.abs(runtime_logits - oracle_logits)
    max_error = float(errors.max())
    if max_error > LOGIT_ATOL:
        worst = names[int(np.argmax(errors.max(axis=1)))]
        raise LocalMismatch(
            f"fold {fold_id}: runtime P0 logits deviate from the frozen "
            f"oracle (max {max_error:.6f} > {LOGIT_ATOL}, worst sample "
            f"{worst})")
    runtime_predictions = [int(v) for v in runtime_logits.argmax(axis=1)]
    mismatched = [names[index] for index in range(len(names))
                  if runtime_predictions[index]
                  != int(oracle[index]["prediction"])]
    if mismatched:
        raise LocalMismatch(
            f"fold {fold_id}: runtime P0 predictions disagree with the "
            f"frozen oracle for {len(mismatched)} sample(s), first: "
            f"{mismatched[:5]}")
    missing_rows = [index for index, row in enumerate(rows)
                    if not local_cache["s_valid"][row]]
    missing_max = _clean(
        float(errors[missing_rows].max())) if missing_rows else 0.0
    record = {
        "fold": fold_id,
        "assembly": ("frozen-scaler transform followed by "
                     "assemble_design_matrix(..., 'P0'); C/S slots "
                     "exactly zero"),
        "oracle_sha256": actual_sha,
        "checkpoint": str(checkpoint),
        "scaler": str(scaler_path),
        "samples": len(rows),
        "max_logit_error": _clean(max_error),
        "logit_atol": LOGIT_ATOL,
        "prediction_mismatches": 0,
        "disabled_slot_max_abs": disabled_slot_max_abs,
        "missing_s_rows": len(missing_rows),
        "missing_s_max_logit_error": missing_max,
    }
    log(f"  P0 identity {fold_id}: {len(rows)} eval rows, max logit "
        f"error {record['max_logit_error']:.3e} (atol {LOGIT_ATOL}), "
        f"{record['missing_s_rows']} missing-S rows checked")
    verified = {
        "head": p0_head,
        "scaler": p0_scaler,
        "eval_logits": runtime_logits,
        "eval_predictions": runtime_predictions,
        "identity": record,
    }
    return verified


def _standardize_local(local_cache: dict, scaler: Standardizer,
                       rows: list) -> np.ndarray:
    c = local_cache["c_features"][rows]
    s = local_cache["s_features"][rows]
    width = LOCAL_C_WIDTH + LOCAL_S_WIDTH
    raw = np.concatenate([c, s], axis=1).astype(np.float64)
    column_valid = np.ones((len(rows), width), dtype=bool)
    column_valid[:, LOCAL_C_WIDTH:] = ~np.isnan(s)
    return scaler.transform(raw, column_valid).astype(np.float32)


def _local_design(standardized: np.ndarray, s_valid: np.ndarray,
                  variant: str) -> np.ndarray:
    c_std = standardized[:, :LOCAL_C_WIDTH]
    s_std = standardized[:, LOCAL_C_WIDTH:]
    return assemble_local_residual_inputs(c_std, s_std, s_valid, variant)


def run_local_admission(config: dict, p0_fit_dir: Path):
    """The frozen old-chain admission (protocol + global cache + P0
    baseline recomputation), the local cache verification, and the
    runtime P0 base identity verification — all BEFORE any residual head
    is built or any training artifact is written."""
    import scripts.train_material_residual as frozen

    if config.get("experiment") != "material_polar_local_interior_band_v1":
        raise LocalRefused(
            f"config experiment {config.get('experiment')!r} is not "
            "material_polar_local_interior_band_v1")
    admission = frozen.run_admission(config, p0_fit_dir)
    protocol = admission["protocol"]
    if protocol["folds"]["status"] != "OK":
        raise LocalRefused(
            "the protocol fold assignment is not OK; fitting requires a "
            "valid group-isolated fold assignment")
    local_dir = _PROJECT_ROOT / config["local_features_source"]["path"]
    local_report = verify_local_cache(local_dir, protocol)
    local_cache = dict(np.load(local_dir / "feature_cache.npz"))
    manifest_samples = admission["manifest"]["samples"]
    authority = admission["authority"]
    all_folds = [fold["fold_id"] for fold in protocol["folds"]["folds"]]

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    p0_verified = {}
    for fold_id in all_folds:
        p0_verified[fold_id] = verify_p0_base_identity(
            fold_id=fold_id, p0_fit_dir=p0_fit_dir, admission=admission,
            global_cache=admission["cache"], local_cache=local_cache,
            manifest_samples=manifest_samples, authority=authority,
            log=log)
    return admission, local_dir, local_report, p0_verified


def fold_parts(manifest_samples: list, authority: dict,
               fold_id: str) -> dict:
    parts = {}
    for part in ("inner_train", "inner_val", "outer_eval"):
        rows = [index for index, sample in enumerate(manifest_samples)
                if authority[sample["sample_name"]]["roles"][fold_id]
                == part]
        if not rows:
            raise LocalRefused(f"fold {fold_id}: {part} is empty")
        parts[part] = rows
    return parts


def fit_one_fold_variant(*, fold_id: str, variant: str, local_cache: dict,
                         global_cache: dict, manifest_samples: list,
                         parts: dict, p0_head, p0_scaler: Standardizer,
                         budget: dict, output_dir: Path, log) -> dict:
    """Fit one local variant on one fold (the frozen loop, wider input)."""
    import torch

    rows_train = parts["inner_train"]
    rows_val = parts["inner_val"]
    rows_eval = parts["outer_eval"]
    scaler, scaler_info = fit_local_scaler(local_cache, rows_train)
    std_train = _standardize_local(local_cache, scaler, rows_train)
    std_val = _standardize_local(local_cache, scaler, rows_val)
    std_eval = _standardize_local(local_cache, scaler, rows_eval)
    valid_train = local_cache["s_valid"][rows_train].astype(bool)
    valid_val = local_cache["s_valid"][rows_val].astype(bool)
    valid_eval = local_cache["s_valid"][rows_eval].astype(bool)

    x_train = _local_design(std_train, valid_train, variant)
    x_val = _local_design(std_val, valid_val, variant)
    x_eval = _local_design(std_eval, valid_eval, variant)
    y_train = [int(v) for v in local_cache["class_id"][rows_train]]
    y_val = [int(v) for v in local_cache["class_id"][rows_val]]
    y_eval = [int(v) for v in local_cache["class_id"][rows_eval]]

    p0_train = p0_logits_of(p0_head, _p0_design_for_rows(
        rows_train, global_cache, p0_scaler))
    p0_val = p0_logits_of(p0_head, _p0_design_for_rows(
        rows_val, global_cache, p0_scaler))
    p0_eval = p0_logits_of(p0_head, _p0_design_for_rows(
        rows_eval, global_cache, p0_scaler))

    model, history, best_state, best = train_residual_head(
        x_train, np.asarray(y_train, dtype=np.int64), p0_train,
        valid_train.astype(np.float32),
        x_val, np.asarray(y_val, dtype=np.int64), p0_val,
        valid_val.astype(np.float32),
        seed=int(budget["seed"]), lr=float(budget["lr"]),
        weight_decay=float(budget["weight_decay"]),
        max_epochs=int(budget["epochs"]),
        aux_loss_weight=float(budget.get("aux_loss_weight", 0.01)),
        input_dim=LOCAL_TOTAL_DIM)
    model.load_state_dict(best_state)
    model.eval()

    with torch.inference_mode():
        eval_final = residual_logits(
            model, torch.from_numpy(x_eval),
            torch.from_numpy(valid_eval.astype(np.float32)),
            torch.from_numpy(p0_eval))
    eval_logits = eval_final.numpy().astype(np.float32)
    predictions = [int(v) for v in eval_logits.argmax(axis=1)]
    delta = eval_logits - p0_eval
    per_sample = []
    for index, row in enumerate(rows_eval):
        sample = manifest_samples[row]
        per_sample.append({
            "sample_name": sample["sample_name"],
            "group": sample["group"],
            "class_id": y_eval[index],
            "has_valid_s": bool(valid_eval[index]),
            "logits": eval_logits[index].tolist(),
            "p0_logits": p0_eval[index].tolist(),
            "delta": delta[index].tolist(),
            "prediction": predictions[index],
        })

    variant_dir = output_dir / "models" / f"{fold_id}_{variant}"
    variant_dir.mkdir(parents=True)
    torch.save({"state_dict": copy.deepcopy(best_state),
                "input_dim": LOCAL_TOTAL_DIM, "num_classes": 4,
                "variant": variant, "fold": fold_id,
                "region_rule": LOCAL_REGION_RULE,
                "seed": int(budget["seed"]),
                "budget": dict(budget)}, variant_dir / "residual_state.pt")
    write_json(variant_dir / "scaler.json", scaler.to_dict())
    write_json(variant_dir / "scaler_info.json", scaler_info)
    write_json(variant_dir / "history.json", {
        "selection": {"epoch": best["epoch"],
                      "inner_val_macro_f1": best["inner_val_macro_f1"],
                      "inner_val_accuracy": best["inner_val_accuracy"]},
        "epochs": history,
    })
    write_jsonl(variant_dir / "outer_eval_predictions.jsonl", per_sample)
    fold_report = {
        "fold": fold_id, "variant": variant, "support": len(rows_eval),
        "selected_epoch": best["epoch"],
        "inner_val_macro_f1": best["inner_val_macro_f1"],
        "inner_val_accuracy": best["inner_val_accuracy"],
        "outer_eval_macro_f1": _clean(macro_f1(y_eval, predictions)),
        "outer_eval_accuracy": _clean(accuracy(y_eval, predictions)),
    }
    write_json(variant_dir / "fold_variant_report.json", fold_report)
    log(f"  {fold_id}/{variant}: epoch {best['epoch']} (val F1 "
        f"{best['inner_val_macro_f1']:.4f}) -> outer macro_f1 "
        f"{fold_report['outer_eval_macro_f1']:.4f}")
    return {
        "fold": fold_id, "variant": variant, "rows": len(rows_eval),
        "labels": y_eval, "predictions": predictions,
        "metrics": _metrics(y_eval, predictions),
        "selected": best, "scaler": scaler, "fold_report": fold_report,
    }


def count_fixed_broke(*, left_predictions: list, right_predictions: list,
                      labels: list, group_by_row: list,
                      comparison: str) -> dict:
    """Per-comparison fixed/broke bookkeeping (never merged across
    comparisons): the same sample correcting against two bases counts
    once in EACH comparison, with per-class and per-group breakdowns."""
    per_class = {name: {"fixed": 0, "broke": 0} for name in CLASS_NAMES}
    per_group: dict = {}
    fixed = broke = both_correct = both_wrong = 0
    for left, right, label, group in zip(left_predictions,
                                         right_predictions, labels,
                                         group_by_row):
        left_ok = left == label
        right_ok = right == label
        if left_ok and right_ok:
            both_correct += 1
        elif not left_ok and not right_ok:
            both_wrong += 1
        elif left_ok:
            # the variant corrected a sample the base got wrong
            fixed += 1
            per_class[CLASS_NAMES[label]]["fixed"] += 1
            slot = per_group.setdefault(group, {"fixed": 0, "broke": 0})
            slot["fixed"] += 1
        else:
            # the variant broke a sample the base had right
            broke += 1
            per_class[CLASS_NAMES[label]]["broke"] += 1
            slot = per_group.setdefault(group, {"fixed": 0, "broke": 0})
            slot["broke"] += 1
    return {
        "comparison": comparison, "fixed": fixed, "broke": broke,
        "both_correct": both_correct, "both_wrong": both_wrong,
        "per_class": per_class, "per_group": per_group,
    }


def run_local_shuffle_for_fold(*, fold_id: str, local_cache: dict,
                               manifest_samples: list, parts: dict,
                               scaler: Standardizer, p0_head, p0_scaler,
                               head, global_cache: dict, budget: dict,
                               output_dir: Path, log) -> dict:
    """The content-shuffle diagnostic over the FROZEN local RCS head.

    Donor matching follows the frozen rule with one CORRECTION (v2,
    pre-registered in the protocol): the grouping is the original
    (split, valid_ratio/gray_mean tertile bin) EXTENDED by the
    (interior_valid, band_valid) validity pattern, encoded as a composite
    bin. Within same split + same original bin + same pattern, the frozen
    seeds 2026-2030 drive a fixed-point-free bijection via the shared
    ``shuffle_assignment``; the FULL standardized S16 comes from that one
    donor (patterns match, so both blocks are exact). The earlier
    per-block guard could not guarantee a per-region bijection (two
    receivers could inherit the same donor block while another donor's
    block went unused); coverage may therefore be lower and is reported
    honestly per region. C, P0 logits and the sample-/block-level missing
    state are never touched; unmatched and invalid rows keep the normal
    inputs and logits exactly.
    """
    import torch

    rows = parts["outer_eval"]
    n = len(rows)
    standardized = _standardize_local(local_cache, scaler, rows)
    c_std = standardized[:, :LOCAL_C_WIDTH]
    s_std = standardized[:, LOCAL_C_WIDTH:]
    s_valid = local_cache["s_valid"][rows].astype(bool)
    s_raw = local_cache["s_features"][rows]
    interior_ok = (~np.isnan(s_raw[:, :INTERIOR_BLOCK[1]])).all(axis=1)
    band_ok = (~np.isnan(s_raw[:, BAND_BLOCK[0]:])).all(axis=1)
    labels = [int(v) for v in local_cache["class_id"][rows]]
    split_labels = [manifest_samples[row]["split"] for row in rows]
    groups = [manifest_samples[row]["group"] for row in rows]
    class_ids = [int(v) for v in local_cache["class_id"][rows]]
    train_rows = parts["inner_train"]
    edges = {
        "valid_ratio": np.quantile(
            global_cache["c_features"][train_rows][:, 0].astype(np.float64),
            [1 / 3, 2 / 3]),
        "gray_mean": np.quantile(
            global_cache["c_features"][train_rows][:, 3].astype(np.float64),
            [1 / 3, 2 / 3]),
    }
    original_bins = build_donor_bins(
        global_cache["c_features"][rows][:, 0].astype(np.float64),
        global_cache["c_features"][rows][:, 3].astype(np.float64),
        edges=edges)
    composite_bins = (original_bins.astype(np.int64) * 4
                      + interior_ok.astype(np.int64) * 2
                      + band_ok.astype(np.int64))

    def x_of(s_matrix: np.ndarray) -> np.ndarray:
        return assemble_local_residual_inputs(c_std, s_matrix, s_valid,
                                              "RCS")

    p0_logits = p0_logits_of(
        p0_head, _p0_design_for_rows(rows, global_cache, p0_scaler))
    p0_tensor = torch.from_numpy(p0_logits)

    def evaluator(x: np.ndarray, valid_vector: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            final = residual_logits(head, torch.from_numpy(x),
                                    torch.from_numpy(valid_vector),
                                    p0_tensor)
        return final.numpy().astype(np.float32)

    normal_x = x_of(s_std)
    normal_logits = np.asarray(
        evaluator(normal_x, s_valid.astype(np.float32)), dtype=np.float32)
    identity_logits = np.asarray(
        evaluator(normal_x, s_valid.astype(np.float32)), dtype=np.float32)
    if not np.array_equal(identity_logits, normal_logits):
        raise LocalMismatch(
            f"fold {fold_id}: the shuffle identity check failed: a second "
            "forward over the unshuffled input did not reproduce the "
            "normal RCS logits exactly")
    if not np.isfinite(normal_logits).all():
        raise LocalRefused(
            f"fold {fold_id}: the normal RCS forward produced non-finite "
            "logits")
    normal_predictions = [int(v) for v in normal_logits.argmax(axis=1)]
    p0_predictions = [int(v) for v in p0_logits.argmax(axis=1)]

    shuffle_dir = output_dir / "shuffle" / fold_id
    shuffle_dir.mkdir(parents=True)
    summary_seeds = {}
    for seed in SHUFFLE_SEEDS:
        receivers = [row for row in range(n) if s_valid[row]]
        assignment = shuffle_assignment(receivers, composite_bins,
                                        split_labels, seed=seed)
        shuffled = s_std.copy()
        records = []
        matched = 0
        same_group = 0
        interior_swapped_count = 0
        band_swapped_count = 0
        for row in range(n):
            base = {
                "sample_name": manifest_samples[rows[row]]["sample_name"],
                "group": groups[row],
                "class_id": class_ids[row],
                "receiver_row": row,
                "receiver_split": split_labels[row],
                "original_bin": int(original_bins[row]),
                "composite_bin": int(composite_bins[row]),
                "interior_valid": bool(interior_ok[row]),
                "band_valid": bool(band_ok[row]),
                "index_meaning": (
                    "0-based row index within this fold's outer_eval rows "
                    "(the frozen row contract order); composite_bin = "
                    "original_bin*4 + interior_valid*2 + band_valid"),
                "has_valid_s": bool(s_valid[row]),
            }
            if not s_valid[row]:
                records.append({**base, "status": "invalid_s",
                                "donor_row": None,
                                "donor_sample_name": None,
                                "donor_split": None,
                                "donor_original_bin": None,
                                "donor_interior_valid": None,
                                "donor_band_valid": None,
                                "interior_block_swapped": False,
                                "band_block_swapped": False})
                continue
            donor = assignment.get(row)
            if donor is None:
                records.append({**base, "status": "unmatched",
                                "unmatched_reason": "single_member_group",
                                "donor_row": None,
                                "donor_sample_name": None,
                                "donor_split": None,
                                "donor_original_bin": None,
                                "donor_interior_valid": None,
                                "donor_band_valid": None,
                                "interior_block_swapped": False,
                                "band_block_swapped": False})
                continue
            donor_interior = bool(interior_ok[donor])
            donor_band = bool(band_ok[donor])
            if donor_interior != bool(interior_ok[row]) or \
                    donor_band != bool(band_ok[row]):
                raise LocalMismatch(
                    f"fold {fold_id}: composite grouping violated for "
                    f"row {row} (donor pattern differs)")
            # the FULL standardized S16 comes from this one donor (the
            # shared pattern makes both blocks exact; invalid cells stay
            # the standardizer's explicit zeros on both sides)
            shuffled[row] = s_std[donor]
            matched += 1
            interior_swapped_count += int(donor_interior)
            band_swapped_count += int(donor_band)
            if groups[donor] == groups[row]:
                same_group += 1
            records.append({
                **base, "status": "matched", "donor_row": donor,
                "donor_sample_name":
                    manifest_samples[rows[donor]]["sample_name"],
                "donor_split": split_labels[donor],
                "donor_original_bin": int(original_bins[donor]),
                "donor_interior_valid": donor_interior,
                "donor_band_valid": donor_band,
                "interior_block_swapped": donor_interior,
                "band_block_swapped": donor_band,
            })
        shuffled_logits = np.asarray(
            evaluator(x_of(shuffled), s_valid.astype(np.float32)),
            dtype=np.float32)
        if not np.isfinite(shuffled_logits).all():
            raise LocalRefused(
                f"fold {fold_id}: the shuffled forward produced non-finite "
                "logits")
        for row in range(n):
            if records[row]["status"] != "matched" and \
                    not np.array_equal(shuffled_logits[row],
                                       normal_logits[row]):
                raise LocalMismatch(
                    f"fold {fold_id}: row {row} with status "
                    f"{records[row]['status']} must reproduce the normal "
                    "logits exactly")
            records[row]["normal_logits"] = normal_logits[row].tolist()
            records[row]["shuffled_logits"] = shuffled_logits[row].tolist()
            records[row]["normal_prediction"] = normal_predictions[row]
            records[row]["shuffled_prediction"] = int(
                shuffled_logits[row].argmax())
        predictions = [int(v) for v in shuffled_logits.argmax(axis=1)]
        vs_p0 = paired_changes(p0_predictions, predictions, labels)
        vs_normal = paired_changes(normal_predictions, predictions, labels)
        interior_denominator = int((interior_ok & s_valid).sum())
        band_denominator = int((band_ok & s_valid).sum())
        seed_report = {
            "seed": int(seed),
            "receivers": len(receivers),
            "matched": matched,
            "coverage": _clean(matched / max(1, len(receivers))),
            "unmatched": len(receivers) - matched,
            "unmatched_reasons": (
                {"single_member_group": len(receivers) - matched}),
            "interior_valid_receivers": interior_denominator,
            "interior_swapped": interior_swapped_count,
            "interior_swap_coverage": _clean(
                interior_swapped_count / interior_denominator)
            if interior_denominator else None,
            "band_valid_receivers": band_denominator,
            "band_swapped": band_swapped_count,
            "band_swap_coverage": _clean(
                band_swapped_count / band_denominator)
            if band_denominator else None,
            "same_group_donor_ratio": _clean(
                same_group / matched) if matched else None,
            "macro_f1": _clean(macro_f1(labels, predictions)),
            "accuracy": _clean(accuracy(labels, predictions)),
            "vs_p0_fixed": vs_p0["fixed"],
            "vs_p0_broke": vs_p0["broke"],
            "vs_normal_rcs_fixed": vs_normal["fixed"],
            "vs_normal_rcs_broke": vs_normal["broke"],
            "grouping_note": (
                "v2 correction: composite bin = original_bin*4 + "
                "interior_valid*2 + band_valid; the full standardized S16 "
                "comes from one donor per receiver. Coverage can be lower "
                "than the row-level guard; unshuffled samples are NOT "
                "evidence of polarization insensitivity"),
        }
        write_json(shuffle_dir / f"seed_{seed}.json",
                   {"seed": int(seed), "samples": records, **seed_report})
        summary_seeds[str(seed)] = seed_report
        log(f"  shuffle {fold_id} seed {seed}: coverage "
            f"{seed_report['coverage']}, vs P0 fixed {vs_p0['fixed']} / "
            f"broke {vs_p0['broke']}")
    write_json(shuffle_dir / "shuffle_summary.json",
               {"fold": fold_id, "seeds": summary_seeds,
                "region_rule": LOCAL_REGION_RULE})
    return {"fold": fold_id, "seeds": summary_seeds}


def run_fit(config: dict, admission: dict, local_dir: Path,
            output_dir: Path, selected_folds: list, log,
            p0_verified: dict) -> dict:
    import torch

    started = time.time()
    manifest_samples = admission["manifest"]["samples"]
    authority = admission["authority"]
    global_cache = admission["cache"]
    local_cache = dict(np.load(local_dir / "feature_cache.npz"))
    budget = {
        "seed": int(config["training"]["seed"]),
        "epochs": int(config["training"]["epochs"]),
        "lr": float(config["training"]["lr"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "aux_loss_weight": float(config["training"].get(
            "aux_loss_weight", 0.01)),
        "device": "cpu",
        "full_batch": True,
    }
    per_variant_results: dict = {variant: [] for variant in VARIANTS}
    shuffle_jobs = []
    for fold_id in selected_folds:
        parts = fold_parts(manifest_samples, authority, fold_id)
        # the SAME verified runtime P0 chain serves R0, the RC/RCS
        # constants, the shuffle and the invalid-S fallback
        verified = p0_verified[fold_id]
        p0_head = verified["head"]
        p0_scaler = verified["scaler"]
        p0_before = snapshot_module_state(p0_head)
        p0_predictions = list(verified["eval_predictions"])
        p0_eval_logits = verified["eval_logits"]
        labels_eval = [int(v) for v in
                       global_cache["class_id"][parts["outer_eval"]]]
        if len(p0_predictions) != len(labels_eval):
            raise LocalRefused(
                f"fold {fold_id}: P0 prediction count does not match the "
                "outer_eval rows")
        fold_variants = {}
        for variant in VARIANTS:
            result = fit_one_fold_variant(
                fold_id=fold_id, variant=variant,
                local_cache=local_cache, global_cache=global_cache,
                manifest_samples=manifest_samples, parts=parts,
                p0_head=p0_head, p0_scaler=p0_scaler, budget=budget,
                output_dir=output_dir, log=log)
            per_variant_results[variant].append(result)
            fold_variants[variant] = result
        # R0 = the frozen P0 on the same rows (explicit control)
        # R0 = the frozen P0 on the same rows (explicit control); its
        # R0 = the frozen P0 on the same rows (explicit control); its
        # logits AND predictions come from the SAME verified runtime P0
        # chain, so prediction == logits.argmax holds for every record
        rows_eval = parts["outer_eval"]
        r0_records = []
        for offset, row in enumerate(rows_eval):
            sample = manifest_samples[row]
            r0_records.append({
                "sample_name": sample["sample_name"],
                "group": sample["group"],
                "class_id": labels_eval[offset],
                "has_valid_s": bool(local_cache["s_valid"][row]),
                "logits": p0_eval_logits[offset].tolist(),
                "prediction": p0_predictions[offset],
                "prediction_source": "verified_runtime_p0",
            })
        r0_dir = output_dir / "models" / f"{fold_id}_R0"
        r0_dir.mkdir(parents=True)
        write_jsonl(r0_dir / "outer_eval_predictions.jsonl", r0_records)
        r0_result = {
            "fold": fold_id, "variant": "R0", "rows": len(labels_eval),
            "labels": labels_eval, "predictions": p0_predictions,
            "metrics": _metrics(labels_eval, p0_predictions),
        }
        per_variant_results.setdefault("R0", []).append(r0_result)
        # P0 frozen evidence: parameters and buffers byte-identical
        p0_after = snapshot_module_state(p0_head)
        check = compare_module_states(p0_before, p0_after)
        if not check["unchanged"]:
            raise LocalRefused(
                f"fold {fold_id}: the P0 head changed during the local "
                f"residual fit: {check['changed_tensors']}")
        if "RCS" in fold_variants:
            shuffle_jobs.append({
                "fold_id": fold_id, "parts": parts, "p0_head": p0_head,
                "p0_scaler": p0_scaler,
                "scaler": fold_variants["RCS"]["scaler"],
            })

    summary_variants = {}
    for variant in VARIANT_ORDER:
        results = per_variant_results[variant]
        folds_report = []
        pooled_labels: list = []
        pooled_predictions: list = []
        for result in results:
            metrics = result["metrics"]
            if variant == "R0":
                folds_report.append({
                    "fold": result["fold"], "rows": result["rows"],
                    **metrics})
            else:
                folds_report.append({
                    "fold": result["fold"], "rows": result["rows"],
                    "support": result["rows"],
                    "selected_epoch": result["selected"]["epoch"],
                    **metrics})
            pooled_labels.extend(result["labels"])
            pooled_predictions.extend(result["predictions"])
        summary_variants[variant] = {
            "folds": folds_report,
            "macro_f1_mean": _clean(np.mean(
                [f["macro_f1"] for f in folds_report])),
            "accuracy_mean": _clean(np.mean(
                [f["accuracy"] for f in folds_report])),
            "pooled": _metrics(pooled_labels, pooled_predictions),
            "pooled_sample_count": len(pooled_labels),
        }
    paired: dict = {}
    class_changes: dict = {}
    group_changes: dict = {}
    for left, right in (("RCS", "R0"), ("RCS", "RC")):
        left_predictions: list = []
        right_predictions: list = []
        labels_all: list = []
        groups_all: list = []
        for lres, rres in zip(per_variant_results[left],
                              per_variant_results[right]):
            if lres["fold"] != rres["fold"]:
                raise LocalRefused(
                    "paired comparison requires identical fold orders")
            left_predictions.extend(lres["predictions"])
            right_predictions.extend(rres["predictions"])
            labels_all.extend(lres["labels"])
            groups_all.extend(
                manifest_samples[row]["group"] for row in
                fold_parts(manifest_samples, authority,
                           lres["fold"])["outer_eval"])
        outcome = count_fixed_broke(
            left_predictions=left_predictions,
            right_predictions=right_predictions,
            labels=labels_all,
            group_by_row=groups_all,
            comparison=f"{left}_minus_{right}")
        # the per-class/per-group bookkeeping must sum to the paired totals
        class_fixed = sum(v["fixed"] for v in outcome["per_class"].values())
        class_broke = sum(v["broke"] for v in outcome["per_class"].values())
        if class_fixed != outcome["fixed"] or class_broke != outcome["broke"]:
            raise LocalMismatch(
                f"{outcome['comparison']}: per-class fixed/broke sums "
                f"({class_fixed}/{class_broke}) do not match the paired "
                f"totals ({outcome['fixed']}/{outcome['broke']})")
        paired[f"{left}_minus_{right}"] = {
            "fixed": outcome["fixed"], "broke": outcome["broke"],
            "both_correct": outcome["both_correct"],
            "both_wrong": outcome["both_wrong"],
            "pooled_macro_f1": {
                right: summary_variants[right]["pooled"]["macro_f1"],
                left: summary_variants[left]["pooled"]["macro_f1"],
            },
        }
        class_changes[outcome["comparison"]] = outcome["per_class"]
        group_changes[outcome["comparison"]] = outcome["per_group"]

    # the old GLOBAL version's reviewed results, as a same-protocol
    # reference only (different feature version; never mixed)
    old_reference = {"available": False}
    reference_path = config.get("old_global_reference", {}).get(
        "fit_summary")
    if reference_path:
        path = _PROJECT_ROOT / reference_path
        payload = load_json(path)
        old_reference = {
            "available": True,
            "fit_summary": str(path),
            "fit_summary_sha256": hashlib.sha256(
                path.read_bytes()).hexdigest(),
            "note": ("reference under the GLOBAL S definition and its own "
                     "protocol version; sample pool and folds match, the "
                     "features do not — never mixed with local numbers"),
            "variants": {name: {
                "macro_f1_mean": payload["variants"][name]["macro_f1_mean"],
                "accuracy_mean": payload["variants"][name][
                    "accuracy_mean"]}
                for name in ("R0", "RC", "RCS")
                if name in payload.get("variants", {})},
            "paired": payload.get("paired", {}),
        }

    # every checkpoint is frozen before the shuffle diagnostic runs; the
    # overall COMPLETED status is only published AFTER the diagnostics
    summary = {
        "mode": "fit",
        "experiment": "material_polar_local_interior_band_v1",
        "region_rule": LOCAL_REGION_RULE,
        "input_dim": LOCAL_TOTAL_DIM,
        "selection_rule": ("inner_val macro_f1 > accuracy > earlier epoch "
                           "(epoch 0 = no correction); outer_eval evaluated "
                           "once per selected model; no refit on the full "
                           "data"),
        "training_status": "FIT_COMPLETE",
        "variants": summary_variants,
        "paired": paired,
        "per_class_changes": class_changes,
        "per_group_changes": group_changes,
        "old_global_reference": old_reference,
        "p0_identity": {fold: p0_verified[fold]["identity"]
                        for fold in selected_folds},
        "elapsed_s": _clean(time.time() - started),
    }
    write_json(output_dir / "summary.json", summary)
    shuffle_summary = {}
    try:
        for job in shuffle_jobs:
            variant_dir = output_dir / "models" / f"{job['fold_id']}_RCS"
            payload = torch.load(variant_dir / "residual_state.pt",
                                 weights_only=False, map_location="cpu")
            head = build_residual_head(seed=int(payload["seed"]),
                                       input_dim=LOCAL_TOTAL_DIM)
            head.load_state_dict(payload["state_dict"])
            head.eval()
            shuffle_summary[job["fold_id"]] = run_local_shuffle_for_fold(
                fold_id=job["fold_id"], local_cache=local_cache,
                manifest_samples=manifest_samples, parts=job["parts"],
                scaler=job["scaler"], p0_head=job["p0_head"],
                p0_scaler=job["p0_scaler"], head=head,
                global_cache=global_cache, budget=budget,
                output_dir=output_dir, log=log)
    except Exception as error:
        # a diagnostic failure must never leave a COMPLETED summary on
        # disk: record the failure phase, keep the fitted artifacts, and
        # re-raise so the entry exits non-zero
        summary["training_status"] = "FAILED"
        summary["failed_phase"] = "shuffle_diagnostic"
        summary["failure"] = f"{type(error).__name__}: {error}"
        write_json(output_dir / "summary.json", summary)
        raise
    summary["training_status"] = "COMPLETED"
    summary["shuffle"] = shuffle_summary
    write_json(output_dir / "summary.json", summary)
    return summary


def yaml_load(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local interior/band residual experiment (dry-run or "
                    "explicit fit).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fit-dir", required=True,
                        help="the completed global-fit directory used by "
                             "the admission chain (read-only)")
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; must not exist")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.fit:
        print("refused: --dry-run and --fit are mutually exclusive",
              file=sys.stderr)
        return EXIT_REFUSED
    if not args.dry_run and not args.fit:
        print("refused: pass exactly one of --dry-run / --fit",
              file=sys.stderr)
        return EXIT_REFUSED

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            raise LocalRefused(
                f"output directory {output_dir} already exists")
        config = yaml_load(Path(args.config))
        # the CLI --fit-dir and the config's P0 source must resolve to the
        # SAME path; a silently different baseline source is refused
        config_p0_dir = _PROJECT_ROOT / config["p0_baseline_source"][
            "fit_dir"]
        cli_p0_dir = Path(args.fit_dir)
        cli_p0_dir = (cli_p0_dir if cli_p0_dir.is_absolute()
                      else _PROJECT_ROOT / cli_p0_dir)
        if cli_p0_dir.resolve() != config_p0_dir.resolve():
            raise LocalRefused(
                f"--fit-dir {cli_p0_dir} does not match the config's P0 "
                f"source {config_p0_dir}; refusing a silently different "
                "baseline source")
        p0_fit_dir = cli_p0_dir
        admission, local_dir, local_report, p0_verified = \
            run_local_admission(config, p0_fit_dir)
        protocol = admission["protocol"]
        all_folds = [fold["fold_id"] for fold in protocol["folds"]["folds"]]
        if args.fold == "all":
            selected_folds = all_folds
        elif args.fold in all_folds:
            selected_folds = [args.fold]
        else:
            raise LocalRefused(
                f"fold {args.fold!r} is not in the protocol folds "
                f"{all_folds}")

        previews = []
        for fold_id in selected_folds:
            parts = fold_parts(admission["manifest"]["samples"],
                               admission["authority"], fold_id)
            previews.append({
                "fold": fold_id,
                "inner_train": len(parts["inner_train"]),
                "inner_val": len(parts["inner_val"]),
                "outer_eval": len(parts["outer_eval"]),
                "p0_baseline": {
                    "macro_f1": admission["fold_results"][fold_id][
                        "recomputed"]["macro_f1"],
                    "selected_epoch": admission["fold_results"][fold_id][
                        "selected_epoch"],
                },
            })
        admission_record = {
            "protocol_sha256": protocol["protocol_sha256"],
            "local_cache": {
                "path": str(local_dir),
                "definition_version": local_report["definition_version"],
                "sample_count": local_report["sample_count"],
                "region_rule": LOCAL_REGION_RULE,
                "input_dim": LOCAL_TOTAL_DIM,
            },
            "p0_fit_dir": str(p0_fit_dir),
            "p0_identity": {fold: p0_verified[fold]["identity"]
                            for fold in all_folds},
            "class_names": list(protocol["data"]["class_names"]),
            "folds": selected_folds,
            "variants": list(VARIANTS),
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "sample_count": len(admission["manifest"]["samples"]),
        }
        output_dir.mkdir(parents=True)
        write_json(output_dir / "admission.json", admission_record)
        if args.dry_run:
            report = {
                "mode": "dry-run",
                "experiment": "material_polar_local_interior_band_v1",
                "region_rule": LOCAL_REGION_RULE,
                "input_dim": LOCAL_TOTAL_DIM,
                "local_cache": admission_record["local_cache"],
                "folds": previews,
                "note": ("no training ran; the real fit is executed by the "
                         "user via scripts/"
                         "manual_material_local_residual.ps1"),
            }
            write_json(output_dir / "report.json", report)
            log(f"dry-run OK: {len(selected_folds)} folds, input_dim "
                f"{LOCAL_TOTAL_DIM}")
            return EXIT_OK
        summary = run_fit(config, admission, local_dir, output_dir,
                          selected_folds, log, p0_verified)
        log(f"fit complete: RCS pooled macro_f1 "
            f"{summary['variants']['RCS']['pooled']['macro_f1']}")
        return EXIT_OK
    except LocalMismatch as error:
        print(f"MISMATCH: {error}", file=sys.stderr)
        return EXIT_MISMATCH
    except (LocalRefused, ValueError, FileNotFoundError) as error:
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        # the summary on disk already carries the FAILED status/phase for
        # fit-stage failures; the CLI still exits non-zero
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
