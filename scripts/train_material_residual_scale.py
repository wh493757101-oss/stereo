"""material_residual_scale_v1 — the training entry (dry-run / fit).

Two magnitude arms over the SAME tiny conditional head, data, budget and
initialization. The ONLY design difference between the arms is the
correction magnitude ``a_f``: ``fixed`` keeps the frozen 0.5 bound,
``scaled`` uses ``0.5 * s_f`` where ``s_f`` is computed per fold from the
VERIFIED frozen P0's own logits on inner_train ONLY (float64 median of
the top-two margins, floored at 1.0). Labels never touch the scale;
inner_val/outer_eval never touch the scale.

- ``--dry-run`` — full admission (anchors, frozen chain, fix cache, P0
  identity per fold), the per-fold scales and the complete 36-path fit
  matrix, WITHOUT calling the training function once (a recording probe
  in the tests proves zero training calls);
- ``--fit``    — the 36 real fits (2 arms x RC/RCS x 3 folds x 3 seeds),
  R0 records from the verified chain, paired comparisons, the content
  shuffle over BOTH arms' RCS heads (shared identity mapping per content
  seed), and the six pre-registered gates. Never run by an agent: the
  user executes scripts/manual_material_residual_scale.ps1 -Mode fit.

One shared deadline (--deadline-utc) or budget (--budget-minutes,
mutually exclusive) covers admission, every fit, evaluation, shuffling
and closeout; an overrun stops NEW computation at a safe boundary,
publishes the finished evidence as PARTIAL and exits non-zero — it never
manufactures a COMPLETED run. Admission failures happen BEFORE any
controlled computation (scaler fitting, optimizer steps) and leave a
refusal report, never a pseudo-completed one.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from core.material_conditional_residual import (  # noqa: E402
    DEFAULT_HIDDEN,
    build_conditional_head,
    conditional_logits,
    train_conditional_residual_head,
)
from core.material_residual_scale import (  # noqa: E402
    ARMS,
    CACHE_KIND_FULL,
    EXPERIMENT,
    DeadlineExceeded,
    DeadlineWatch,
    ScaleRefused,
    UPSAMPLE_FIX_VERSION,
    VARIANTS,
    TRAIN_SEEDS,
    evaluate_gates,
    expand_historical_bindings,
    fold_scale,
    load_scale_config,
    magnitude_for_arm,
    verify_scale_cache_markers,
)
from core.polar_local_features import (  # noqa: E402
    LOCAL_REGION_RULE,
    LOCAL_TOTAL_DIM,
    fit_local_scaler,
    verify_local_cache,
)
from scripts.train_material_conditional_residual import (  # noqa: E402
    _fold_conditional_inputs,
    _metrics,
    _variant_inputs,
    align_paired_samples,
    local_entry_p0_logits,
    verify_normal_matches_saved_rcs,
)
from scripts.train_material_local_residual import (  # noqa: E402
    CLASS_NAMES,
    EXIT_BLOCKED,
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_REFUSED,
    LocalMismatch,
    LocalRefused,
    _clean,
    _standardize_local,
    count_fixed_broke,
    fold_parts,
    load_json,
    verify_p0_base_identity,
    write_json,
    write_jsonl,
)
from scripts.eval_material_upsample_impact import (  # noqa: E402
    IdentityError,
    SourceRegistry,
    dump_json,
    dump_jsonl,
    is_link_or_reparse,
    verify_anchor,
)

PLANNED_FITS = 2 * 2 * 3 * 3  # arms x variants x folds x seeds


def authority_sample_folds(authority: dict) -> list[str]:
    folds = set()
    for entry in authority.values():
        folds.update(entry["roles"])
    return sorted(folds)


def fit_one_scale_path(*, seed: int, fold_id: str, arm: str, variant: str,
                       magnitude: float, shared_inputs: dict, budget: dict,
                       output_dir: Path, log) -> dict:
    """One real fit: (seed x fold x arm x variant). The per-variant
    inputs are assembled here from the shared parts (RC always receives
    its own S-masked tensors); the magnitude enters the training AND the
    outer evaluation — the arms' inputs, head, seed, init and budget are
    identical by construction."""
    import torch

    fold_inputs = _variant_inputs(shared_inputs, variant)
    model, history, best_state, best = train_conditional_residual_head(
        fold_inputs["tensors"]["x_train"].numpy(),
        np.asarray(shared_inputs["labels"]["train"], dtype=np.int64),
        shared_inputs["p0_logits"]["train"],
        shared_inputs["valid"]["train"].astype(np.float32),
        fold_inputs["tensors"]["x_val"].numpy(),
        np.asarray(shared_inputs["labels"]["val"], dtype=np.int64),
        shared_inputs["p0_logits"]["val"],
        shared_inputs["valid"]["val"].astype(np.float32),
        seed=int(seed), lr=float(budget["lr"]),
        weight_decay=float(budget["weight_decay"]),
        max_epochs=int(budget["epochs"]),
        aux_loss_weight=float(budget.get("aux_loss_weight", 0.01)),
        input_dim=int(shared_inputs["input_dim"]),
        hidden=int(budget.get("hidden", DEFAULT_HIDDEN)),
        magnitude=float(magnitude))
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        eval_final = conditional_logits(
            model, torch.from_numpy(fold_inputs["x"]["eval"]),
            torch.from_numpy(shared_inputs["valid"]["eval"]
                             .astype(np.float32)),
            torch.from_numpy(shared_inputs["p0_logits"]["eval"]),
            float(magnitude))
    eval_logits = eval_final.numpy().astype(np.float32)
    predictions = [int(v) for v in eval_logits.argmax(axis=1)]
    delta = eval_logits - shared_inputs["p0_logits"]["eval"]
    per_sample = []
    for index, row in enumerate(shared_inputs["rows"]["eval"]):
        sample = shared_inputs["manifest_samples"][row]
        per_sample.append({
            "sample_name": sample["sample_name"],
            "group": sample["group"],
            "class_id": shared_inputs["labels"]["eval"][index],
            "has_valid_s": bool(shared_inputs["valid"]["eval"][index]),
            "logits": eval_logits[index].tolist(),
            "p0_logits": shared_inputs["p0_logits"]["eval"][index].tolist(),
            "delta": delta[index].tolist(),
            "delta_max_abs": float(np.abs(delta[index]).max()),
            "magnitude": float(magnitude),
            "prediction": predictions[index],
        })
    path_dir = output_dir / "models" / f"seed_{seed}" / \
        f"{fold_id}_{arm}_{variant}"
    path_dir.mkdir(parents=True)
    import torch as _torch

    _torch.save({"state_dict": copy.deepcopy(best_state),
                 "input_dim": int(shared_inputs["input_dim"]),
                 "hidden": int(budget.get("hidden", DEFAULT_HIDDEN)),
                 "num_classes": 4, "variant": variant, "arm": arm,
                 "magnitude": float(magnitude), "fold": fold_id,
                 "seed": int(seed), "region_rule": LOCAL_REGION_RULE,
                 "upsample_version": UPSAMPLE_FIX_VERSION,
                 "budget": dict(budget)},
                path_dir / "residual_state.pt")
    write_json(path_dir / "scaler.json", shared_inputs["scaler"].to_dict())
    write_json(path_dir / "history.json", {
        "selection": {"epoch": best["epoch"],
                      "inner_val_macro_f1": best["inner_val_macro_f1"],
                      "inner_val_accuracy": best["inner_val_accuracy"]},
        "magnitude": float(magnitude),
        "epochs": history,
    })
    write_jsonl(path_dir / "outer_eval_predictions.jsonl", per_sample)
    fold_report = {
        "fold": fold_id, "arm": arm, "variant": variant,
        "seed": int(seed), "magnitude": float(magnitude),
        "support": len(shared_inputs["rows"]["eval"]),
        "selected_epoch": best["epoch"],
        "inner_val_macro_f1": best["inner_val_macro_f1"],
        "inner_val_accuracy": best["inner_val_accuracy"],
        "outer_eval_macro_f1": _clean(_metrics(
            shared_inputs["labels"]["eval"], predictions)["macro_f1"]),
        "outer_eval_accuracy": _clean(_metrics(
            shared_inputs["labels"]["eval"], predictions)["accuracy"]),
    }
    # the shared shuffle binder reads fold_variant_report.json
    write_json(path_dir / "fold_variant_report.json", fold_report)
    saturated = sum(1 for row in per_sample
                    if row["has_valid_s"] and row["delta_max_abs"] >=
                    magnitude * 0.999)
    log(f"  seed {seed} {fold_id}/{arm}/{variant}: epoch {best['epoch']} "
        f"(val F1 {best['inner_val_macro_f1']:.4f}, a_f {magnitude:.4f}, "
        f"saturated {saturated}) -> outer macro_f1 "
        f"{fold_report['outer_eval_macro_f1']:.4f}")
    return {
        "seed": int(seed), "fold": fold_id, "arm": arm, "variant": variant,
        "rows": len(shared_inputs["rows"]["eval"]),
        "labels": shared_inputs["labels"]["eval"],
        "predictions": predictions,
        "metrics": _metrics(shared_inputs["labels"]["eval"], predictions),
        "selected": best, "fold_report": fold_report,
        "per_sample": per_sample,
        "delta_max_abs": float(max(row["delta_max_abs"]
                                   for row in per_sample)),
        "saturated_valid": saturated,
    }


def run_scale_shuffle_for_fold(*, fold_id: str, train_seed: int,
                               arm_heads: dict, fix_cache: dict,
                               manifest_samples: list, parts: dict,
                               scaler, p0_head, p0_scaler,
                               global_cache: dict, output_dir: Path,
                               log) -> dict:
    """The v2 content shuffle over BOTH arms' frozen scaled RCS heads.

    The composite bins, the quantile edges (inner_train only), the donor
    groups and the per-content-seed assignment are computed ONCE and
    shared by the arms — the arms differ ONLY in which frozen head and
    magnitude evaluate the shuffled inputs. Donors: same protocol role,
    same composite bin, polarisation available; no self, no replacement,
    no fixed point; single-member bins stay unmatched with their original
    inputs."""
    import torch

    from core.material_residual import shuffle_assignment as assign
    from core.material_conditional_residual import (assemble_conditional_inputs,
                                                    p0_context)
    from core.polar_local_features import LOCAL_C_WIDTH

    rows = parts["outer_eval"]
    n = len(rows)
    standardized = _standardize_local(fix_cache, scaler, rows)
    c_std = standardized[:, :LOCAL_C_WIDTH]
    s_std = standardized[:, LOCAL_C_WIDTH:]
    s_valid = fix_cache["s_valid"][rows].astype(bool)
    s_raw = fix_cache["s_features"][rows]
    interior_ok = (~np.isnan(s_raw[:, :8])).all(axis=1)
    band_ok = (~np.isnan(s_raw[:, 8:])).all(axis=1)
    labels = [int(v) for v in fix_cache["class_id"][rows]]
    split_labels = [manifest_samples[row]["split"] for row in rows]
    groups = [manifest_samples[row]["group"] for row in rows]
    train_rows = parts["inner_train"]
    # binning edges from the FIX cache C (inner_train only): column 0 is
    # quality_valid_ratio, column 3 is gray_mean_in_instance_mask
    edges = {
        "valid_ratio": np.quantile(
            fix_cache["c_features"][train_rows][:, 0].astype(np.float64),
            [1 / 3, 2 / 3]),
        "gray_mean": np.quantile(
            fix_cache["c_features"][train_rows][:, 3].astype(np.float64),
            [1 / 3, 2 / 3]),
    }
    from core.material_residual import build_donor_bins

    original_bins = build_donor_bins(
        fix_cache["c_features"][rows][:, 0].astype(np.float64),
        fix_cache["c_features"][rows][:, 3].astype(np.float64),
        edges=edges)
    composite_bins = (original_bins.astype(np.int64) * 4
                      + interior_ok.astype(np.int64) * 2
                      + band_ok.astype(np.int64))
    context = p0_context(local_entry_p0_logits(p0_head, p0_scaler,
                                               global_cache, rows))
    p0_logits = local_entry_p0_logits(p0_head, p0_scaler, global_cache,
                                      rows)
    p0_tensor = torch.from_numpy(p0_logits)

    def x_of(s_matrix: np.ndarray) -> np.ndarray:
        return assemble_conditional_inputs(c_std, s_matrix, context,
                                           s_valid, "RCS")

    def evaluator_for(arm: str):
        head = arm_heads[arm]["head"]
        magnitude = float(arm_heads[arm]["magnitude"])

        def evaluate(x: np.ndarray, valid_vector: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                final = conditional_logits(
                    head, torch.from_numpy(x),
                    torch.from_numpy(valid_vector), p0_tensor,
                    magnitude)
            return final.numpy().astype(np.float32)

        return evaluate

    def make_records(shuffled_logits, normal_logits, normal_predictions,
                     assignment):
        records = []
        matched = same_group = 0
        for row in range(n):
            base = {
                "sample_name": manifest_samples[rows[row]]["sample_name"],
                "group": groups[row],
                "receiver_row": row,
                "receiver_split": split_labels[row],
                "original_bin": int(original_bins[row]),
                "composite_bin": int(composite_bins[row]),
                "interior_valid": bool(interior_ok[row]),
                "band_valid": bool(band_ok[row]),
                "has_valid_s": bool(s_valid[row]),
            }
            if not s_valid[row]:
                records.append({**base, "status": "invalid_s",
                                "donor_row": None,
                                "donor_sample_name": None})
                continue
            donor = assignment.get(row)
            if donor is None:
                records.append({**base, "status": "unmatched",
                                "unmatched_reason": "single_member_group",
                                "donor_row": None,
                                "donor_sample_name": None})
                continue
            donor_interior = bool(interior_ok[donor])
            donor_band = bool(band_ok[donor])
            if donor_interior != bool(interior_ok[row]) or \
                    donor_band != bool(band_ok[row]):
                raise LocalMismatch(
                    f"fold {fold_id}: composite grouping violated")
            matched += 1
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
                "normal_logits": normal_logits[row].tolist(),
                "shuffled_logits": shuffled_logits[row].tolist(),
                "normal_prediction": normal_predictions[row],
                "shuffled_prediction": int(shuffled_logits[row].argmax()),
            })
        return records, matched, same_group

    per_seed_reports = {arm: {} for arm in arm_heads}
    for content_seed in (2026, 2027, 2028, 2029, 2030):
        receivers = [row for row in range(n) if s_valid[row]]
        assignment = assign(receivers, composite_bins, split_labels,
                            seed=content_seed)
        for arm in sorted(arm_heads):
            evaluate = evaluator_for(arm)
            normal_x = x_of(s_std)
            normal_logits = np.asarray(
                evaluate(normal_x, s_valid.astype(np.float32)),
                dtype=np.float32)
            identity = np.asarray(
                evaluate(normal_x, s_valid.astype(np.float32)),
                dtype=np.float32)
            if not np.array_equal(identity, normal_logits):
                raise LocalMismatch(
                    f"fold {fold_id} arm {arm} seed {train_seed}: the "
                    "shuffle identity check failed")
            normal_predictions = [int(v) for v in
                                  normal_logits.argmax(axis=1)]
            verify_normal_matches_saved_rcs(
                fold_id=fold_id, train_seed=train_seed,
                variant_dir=output_dir / "models" / f"seed_{train_seed}" /
                f"{fold_id}_{arm}_RCS",
                rows=rows, manifest_samples=manifest_samples,
                normal_logits=normal_logits,
                normal_predictions=normal_predictions)
            shuffled = s_std.copy()
            for row, donor in assignment.items():
                if donor is not None:
                    shuffled[row] = s_std[donor]
            shuffled_logits = np.asarray(
                evaluate(x_of(shuffled), s_valid.astype(np.float32)),
                dtype=np.float32)
            records, matched, same_group = make_records(
                shuffled_logits, normal_logits, normal_predictions,
                assignment)
            for row in range(n):
                if records[row]["status"] != "matched" and \
                        not np.array_equal(shuffled_logits[row],
                                           normal_logits[row]):
                    raise LocalMismatch(
                        f"fold {fold_id} arm {arm}: row {row} with "
                        f"status {records[row]['status']} must reproduce "
                        "the normal logits exactly")
                if records[row]["status"] != "matched":
                    records[row]["normal_logits"] = \
                        normal_logits[row].tolist()
                    records[row]["shuffled_logits"] = \
                        shuffled_logits[row].tolist()
                    records[row]["normal_prediction"] = \
                        normal_predictions[row]
                    records[row]["shuffled_prediction"] = int(
                        shuffled_logits[row].argmax())
            predictions = [int(v) for v in shuffled_logits.argmax(axis=1)]
            report = {
                "content_seed": int(content_seed),
                "arm": arm,
                "receivers": len(receivers),
                "matched": matched,
                "coverage": _clean(matched / max(1, len(receivers))),
                "unmatched": len(receivers) - matched,
                "same_group_donor_ratio": _clean(same_group / matched)
                if matched else None,
                "macro_f1": _clean(_metrics(labels, predictions)
                                   ["macro_f1"]),
                "vs_normal_rcs_fixed": sum(
                    1 for r in records if r["status"] == "matched"
                    and r["shuffled_prediction"]
                    == labels[r["receiver_row"]]
                    and r["normal_prediction"]
                    != labels[r["receiver_row"]]),
                "vs_normal_rcs_broke": sum(
                    1 for r in records if r["status"] == "matched"
                    and r["normal_prediction"]
                    == labels[r["receiver_row"]]
                    and r["shuffled_prediction"]
                    != labels[r["receiver_row"]]),
                "grouping_note": ("composite bin over fix-cache C terciles "
                                  "(inner_train edges) x interior/band "
                                  "validity; the full standardized S16 "
                                  "comes from one donor; the P0 context "
                                  "is untouched"),
            }
            out_dir = (output_dir / "shuffle" / arm /
                       f"seed_{train_seed}" / fold_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(out_dir / f"seed_{content_seed}.json",
                       {"content_seed": int(content_seed),
                        "train_seed": int(train_seed), "arm": arm,
                        "samples": records, **report})
            per_seed_reports[arm][str(content_seed)] = report
            log(f"  shuffle {fold_id} seed {train_seed} arm {arm} "
                f"content {content_seed}: coverage {report['coverage']}")
    summary = {"fold": fold_id, "train_seed": int(train_seed),
               "arms": per_seed_reports, "region_rule": LOCAL_REGION_RULE}
    for arm in sorted(arm_heads):
        out_dir = output_dir / "shuffle" / arm / f"seed_{train_seed}" / \
            fold_id
        write_json(out_dir / "shuffle_summary.json",
                   {"fold": fold_id, "train_seed": int(train_seed),
                    "arm": arm, "seeds": per_seed_reports[arm]})
    return summary


def run_fit(ctx, cfg: dict, admission: dict, feature_dir: Path,
            output_dir: Path, scales: dict, log) -> dict:
    """The 36 real fits, the paired comparisons, the shuffle and the
    gates. outer_eval never enters a training function; the scaler is
    fitted once per fold and shared by every path in the fold."""
    import torch

    watch = ctx["watch"]
    started = time.time()
    manifest_samples = admission["manifest"]["samples"]
    authority = admission["authority"]
    global_cache = admission["cache"]
    fix_cache = dict(np.load(feature_dir / "feature_cache.npz"))
    budget = {
        "epochs": int(cfg["training"]["epochs"]),
        "lr": float(cfg["training"]["lr"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "aux_loss_weight": float(cfg["training"].get("aux_loss_weight",
                                                     0.01)),
        "hidden": int(cfg.get("head", {}).get("hidden", DEFAULT_HIDDEN)),
        "device": "cpu", "full_batch": True,
    }
    results = {arm: {variant: [] for variant in VARIANTS} for arm in ARMS}
    r0_records_by_fold: dict = {}
    shuffle_jobs = []
    for fold_id in sorted(authority_sample_folds(authority)):
        watch.check(f"fold {fold_id} admission")
        parts = fold_parts(manifest_samples, authority, fold_id)
        verified = ctx["p0_verified"][fold_id]
        p0_head = verified["head"]
        p0_scaler = verified["scaler"]
        from core.material_probe import compare_module_states
        from core.material_probe import snapshot_module_state
        p0_before = snapshot_module_state(p0_head)
        scale = float(scales[fold_id]["scale"])
        scaler, _info = fit_local_scaler(fix_cache, parts["inner_train"])
        for seed in TRAIN_SEEDS:
            watch.check(f"fold {fold_id} seed {seed} inputs")
            shared_inputs = _fold_conditional_inputs(
                local_cache=fix_cache, global_cache=global_cache,
                parts=parts, scaler=scaler, p0_head=p0_head,
                p0_scaler=p0_scaler)
            shared_inputs["manifest_samples"] = manifest_samples
            shared_inputs["scaler"] = scaler
            shared_inputs["input_dim"] = 35
            for arm in ARMS:
                magnitude = magnitude_for_arm(arm, scale)
                for variant in VARIANTS:
                    watch.check(f"fit {fold_id}/{arm}/{variant}/{seed}")
                    result = fit_one_scale_path(
                        seed=seed, fold_id=fold_id, arm=arm,
                        variant=variant, magnitude=magnitude,
                        shared_inputs=shared_inputs, budget=budget,
                        output_dir=output_dir, log=log)
                    results[arm][variant].append(result)
        rows_eval = parts["outer_eval"]
        eval_logits = verified["eval_logits"]
        predictions = verified["eval_predictions"]
        labels_eval = [int(v) for v in fix_cache["class_id"][rows_eval]]
        r0_records = []
        for offset, row in enumerate(rows_eval):
            sample = manifest_samples[row]
            r0_records.append({
                "sample_name": sample["sample_name"],
                "group": sample["group"],
                "class_id": labels_eval[offset],
                "has_valid_s": bool(fix_cache["s_valid"][row]),
                "logits": eval_logits[offset].tolist(),
                "prediction": predictions[offset],
                "prediction_source": "verified_runtime_p0",
            })
        r0_dir = output_dir / "models" / "R0" / f"{fold_id}_R0"
        r0_dir.mkdir(parents=True)
        write_jsonl(r0_dir / "outer_eval_predictions.jsonl", r0_records)
        r0_records_by_fold[fold_id] = r0_records
        p0_after = snapshot_module_state(p0_head)
        check = compare_module_states(p0_before, p0_after)
        if not check["unchanged"]:
            raise LocalRefused(
                f"fold {fold_id}: the frozen P0 head changed during the "
                f"fits: {check['changed_tensors']}")
        shuffle_jobs.append({
            "fold_id": fold_id, "parts": parts,
            "p0_head": p0_head, "p0_scaler": p0_scaler, "scaler": scaler,
        })

    # ---- summaries over all seeds (no seed selection) ----
    variants_summary = {}
    r0_labels, r0_predictions = [], []
    for fold_id in r0_records_by_fold:
        for record in r0_records_by_fold[fold_id]:
            r0_labels.append(record["class_id"])
            r0_predictions.append(record["prediction"])
    r0_folds = []
    for fold_id in sorted(r0_records_by_fold):
        fold_records = r0_records_by_fold[fold_id]
        labels = [r["class_id"] for r in fold_records]
        predictions = [r["prediction"] for r in fold_records]
        r0_folds.append({"fold": fold_id, "rows": len(labels),
                         **_metrics(labels, predictions)})
    r0_entry = {
        "note": ("the frozen P0; identical across seeds and arms by "
                 "construction — one independent observation per sample"),
        "per_seed": {str(seed): {"folds": r0_folds,
                                 "macro_f1_mean": _clean(np.mean(
                                     [f["macro_f1"] for f in r0_folds])),
                                 "accuracy_mean": _clean(np.mean(
                                     [f["accuracy"] for f in r0_folds])),
                                 "pooled": _metrics(r0_labels,
                                                    r0_predictions),
                                 "pooled_sample_count": len(r0_labels)}
                     for seed in TRAIN_SEEDS},
        "macro_f1_mean_of_seeds": _clean(np.mean(
            [f["macro_f1"] for f in r0_folds])),
        "macro_f1_seed_range": None,
        "pooled_macro_f1_mean_of_seeds": _clean(_metrics(
            r0_labels, r0_predictions)["macro_f1"]),
        "accuracy_mean": _clean(np.mean(
            [f["accuracy"] for f in r0_folds])),
    }
    arms_summary = {}
    for arm in ARMS:
        variants_summary = {"R0": r0_entry}
        for variant in VARIANTS:
            per_seed = {}
            seed_means = []
            for seed in TRAIN_SEEDS:
                fold_results = [r for r in results[arm][variant]
                                if r["seed"] == seed]
                folds_report = []
                pooled_labels = []
                pooled_predictions = []
                for result in fold_results:
                    folds_report.append({
                        "fold": result["fold"], "rows": result["rows"],
                        "selected_epoch": result["selected"]["epoch"],
                        **result["metrics"]})
                    pooled_labels.extend(result["labels"])
                    pooled_predictions.extend(result["predictions"])
                fold_mean = _clean(np.mean([f["macro_f1"]
                                            for f in folds_report]))
                seed_accuracy_mean = _clean(np.mean(
                    [f["accuracy"] for f in folds_report]))
                seed_means.append(fold_mean)
                per_seed[str(seed)] = {
                    "folds": folds_report,
                    "macro_f1_mean": fold_mean,
                    "accuracy_mean": seed_accuracy_mean,
                    "pooled": _metrics(pooled_labels, pooled_predictions),
                    "pooled_sample_count": len(pooled_labels),
                }
            variants_summary[variant] = {
                "per_seed": per_seed,
                "macro_f1_mean_of_seeds": _clean(np.mean(seed_means)),
                "macro_f1_seed_range": [_clean(min(seed_means)),
                                        _clean(max(seed_means))],
                "pooled_macro_f1_mean_of_seeds": _clean(np.mean(
                    [per_seed[str(seed)]["pooled"]["macro_f1"]
                     for seed in TRAIN_SEEDS])),
                "accuracy_mean": _clean(np.mean(
                    [per_seed[str(seed)]["accuracy_mean"]
                     for seed in TRAIN_SEEDS])),
            }
        arms_summary[arm] = variants_summary

    # ---- paired comparisons, per seed and summed over seeds ----
    manifest_rows = {sample["sample_name"]: {"class_id":
                                             sample["class_id"],
                                             "group": sample["group"]}
                     for sample in manifest_samples}

    def paired_comparison(left_key: str, right_key: str) -> tuple:
        """left/right keys: (arm, variant) or R0."""
        per_seed_counts = {}
        total = {"fixed": 0, "broke": 0, "both_correct": 0,
                 "both_wrong": 0}
        per_class_total = {name: {"fixed": 0, "broke": 0}
                           for name in CLASS_NAMES}
        per_group_total: dict = {}
        all_folds = sorted(r0_records_by_fold)
        for seed in TRAIN_SEEDS:
            left_groups, right_groups, expected_names = [], [], []
            for fold_id in all_folds:
                left_results = [r for r in results[left_key[0]][left_key[1]]
                                if r["seed"] == seed
                                and r["fold"] == fold_id]
                if len(left_results) != 1:
                    raise LocalMismatch(
                        f"seed {seed} fold {fold_id}: expected exactly one "
                        f"{left_key} result")
                left_groups.append(left_results[0]["per_sample"])
                if right_key == "R0":
                    right_groups.append(r0_records_by_fold[fold_id])
                else:
                    right_results = [r for r in
                                     results[right_key[0]][right_key[1]]
                                     if r["seed"] == seed
                                     and r["fold"] == fold_id]
                    if len(right_results) != 1:
                        raise LocalMismatch(
                            f"seed {seed} fold {fold_id}: expected exactly "
                            f"one {right_key} result")
                    right_groups.append(right_results[0]["per_sample"])
                expected_names.extend(
                    manifest_samples[row]["sample_name"]
                    for row in fold_parts(manifest_samples, authority,
                                          fold_id)["outer_eval"])
            aligned_left, aligned_right, labels_all, groups_all = \
                align_paired_samples(left_groups, right_groups,
                                     expected_names=expected_names,
                                     manifest_rows=manifest_rows)
            outcome = count_fixed_broke(
                left_predictions=aligned_left,
                right_predictions=aligned_right,
                labels=labels_all, group_by_row=groups_all,
                comparison=f"{left_key}_minus_{right_key}")
            support = len(expected_names)
            four_sum = (outcome["fixed"] + outcome["broke"]
                        + outcome["both_correct"] + outcome["both_wrong"])
            if four_sum != support:
                raise LocalMismatch(
                    f"{left_key}_minus_{right_key} seed {seed}: counts sum "
                    f"{four_sum} != outer_eval support {support}")
            per_seed_counts[str(seed)] = {
                "fixed": outcome["fixed"], "broke": outcome["broke"],
                "both_correct": outcome["both_correct"],
                "both_wrong": outcome["both_wrong"], "support": support}
            for key in total:
                total[key] += outcome[key]
            for name, counts in outcome["per_class"].items():
                per_class_total[name]["fixed"] += counts["fixed"]
                per_class_total[name]["broke"] += counts["broke"]
            for group, counts in outcome["per_group"].items():
                slot = per_group_total.setdefault(
                    group, {"fixed": 0, "broke": 0})
                slot["fixed"] += counts["fixed"]
                slot["broke"] += counts["broke"]
        paired_entry = {
            "per_seed": per_seed_counts,
            "total": {**total,
                      "note": ("sample-model pair counts across seeds; "
                               "seeds are re-fits, not independent "
                               "captures")},
            "macro_f1_by_path": {
                f"{arm}_{variant}": {
                    "fold_mean_of_seeds":
                        arms_summary[arm][variant]["macro_f1_mean_of_seeds"],
                    "pooled_mean_of_seeds":
                        arms_summary[arm][variant][
                            "pooled_macro_f1_mean_of_seeds"]}
                for arm in ARMS for variant in ("R0", *VARIANTS)},
        }
        per_class_entry = {"summed_over_seeds": per_class_total}
        per_group_entry = {"summed_over_seeds": per_group_total}
        return paired_entry, per_class_entry, per_group_entry

    paired = {}
    class_changes = {}
    group_changes = {}
    for left_arm in ARMS:
        for right in ("R0", "RC"):
            key = f"{left_arm}_RCS_minus_{right}"
            paired[key], class_changes[key], group_changes[key] = \
                paired_comparison((left_arm, "RCS"),
                                  "R0" if right == "R0" else (left_arm,
                                                              "RC"))
    for variant in VARIANTS:
        key = f"scaled_{variant}_minus_fixed_{variant}"
        paired[key], class_changes[key], group_changes[key] = \
            paired_comparison(("scaled", variant), ("fixed", variant))

    # ---- the shuffle over both arms' RCS heads (shared assignments) ----
    summary = {}
    shuffle_summary = {}
    try:
        for job in shuffle_jobs:
            watch.check(f"shuffle fold {job['fold_id']}")
            for seed in TRAIN_SEEDS:
                arm_heads = {}
                for arm in ARMS:
                    variant_dir = (output_dir / "models" / f"seed_{seed}" /
                                   f"{job['fold_id']}_{arm}_RCS")
                    payload = torch.load(
                        variant_dir / "residual_state.pt",
                        weights_only=False, map_location="cpu")
                    head = build_conditional_head(
                        seed=int(payload["seed"]),
                        input_dim=int(payload["input_dim"]),
                        hidden=int(payload["hidden"]))
                    head.load_state_dict(payload["state_dict"])
                    head.eval()
                    arm_heads[arm] = {"head": head,
                                      "magnitude":
                                          float(payload["magnitude"])}
                shuffle_summary.setdefault(f"seed_{seed}", {})
                shuffle_summary[f"seed_{seed}"][job["fold_id"]] = \
                    run_scale_shuffle_for_fold(
                        fold_id=job["fold_id"], train_seed=seed,
                        arm_heads=arm_heads, fix_cache=fix_cache,
                        manifest_samples=manifest_samples,
                        parts=job["parts"], scaler=job["scaler"],
                        p0_head=job["p0_head"],
                        p0_scaler=job["p0_scaler"],
                        global_cache=global_cache,
                        output_dir=output_dir, log=log)
    except DeadlineExceeded as error:
        gates = evaluate_gates(summary) if summary.get("arms") else None
        summary = {
            "mode": "fit", "experiment": EXPERIMENT,
            "training_status": "PARTIAL",
            "interrupted_by_deadline": True,
            "failed_phase": "shuffle_diagnostic",
            "failure": f"{type(error).__name__}: {error}",
            "gates": gates,
            "elapsed_s": _clean(time.time() - started),
        }
        write_json(output_dir / "summary.json", summary)
        raise
    except Exception as error:
        summary = {
            "mode": "fit", "experiment": EXPERIMENT,
            "training_status": "FAILED",
            "failed_phase": "shuffle_diagnostic",
            "failure": f"{type(error).__name__}: {error}",
            "elapsed_s": _clean(time.time() - started),
        }
        write_json(output_dir / "summary.json", summary)
        raise

    summary = {
        "mode": "fit",
        "experiment": EXPERIMENT,
        "region_rule": LOCAL_REGION_RULE,
        "input_dim": 35,
        "upsample_version": UPSAMPLE_FIX_VERSION,
        "scales": scales,
        "magnitude_rule": {"fixed": 0.5,
                           "scaled": "0.5 * s_f (per fold)"},
        "seeds": list(TRAIN_SEEDS),
        "seeds_note": ("all seeds reported; none selected; seeds are "
                       "re-fits, not independent captures"),
        "selection_rule": ("inner_val macro_f1 > accuracy > earlier epoch "
                           "(epoch 0 = no correction); outer_eval evaluated "
                           "once per selected model"),
        "class_names": list(CLASS_NAMES),
        "arms": arms_summary,
        "paired": paired,
        "per_class_changes": class_changes,
        "per_group_changes": group_changes,
        "gates": None,
        "training_status": "FIT_COMPLETE",
        "elapsed_s": _clean(time.time() - started),
    }
    summary["gates"] = evaluate_gates(summary)
    summary["shuffle"] = shuffle_summary
    # the SUCCESS PUBLISH boundary: crossing the deadline after the last
    # fit or shuffle must NOT produce a COMPLETED run — the FULL finished
    # evidence (fits, gates, shuffle) is published as PARTIAL and the
    # exit stays non-zero
    try:
        watch.check("before completion publish")
        summary["training_status"] = "COMPLETED"
    except DeadlineExceeded:
        summary["training_status"] = "PARTIAL"
        summary["interrupted_by_deadline"] = True
        write_json(output_dir / "summary.json", summary)
        raise
    write_json(output_dir / "summary.json", summary)
    return summary


def run_dry_run(ctx, cfg: dict, admission: dict, feature_dir: Path,
                output_dir: Path, scales: dict) -> dict:
    """Admission + scales + the full fit matrix, with ZERO training
    calls (the tests prove it with a recording probe)."""
    folds = sorted(authority_sample_folds(admission["authority"]))
    matrix = [{"fold": fold, "arm": arm, "variant": variant, "seed": seed,
               "magnitude": magnitude_for_arm(arm, scales[fold]["scale"])}
              for fold in folds for arm in ARMS for variant in VARIANTS
              for seed in TRAIN_SEEDS]
    report = {
        "mode": "dry-run",
        "experiment": EXPERIMENT,
        "upsample_version": UPSAMPLE_FIX_VERSION,
        "region_rule": LOCAL_REGION_RULE,
        "input_dim": 35,
        "folds": folds,
        "scales": scales,
        "planned_fits": len(matrix),
        "matrix": matrix,
        "shuffle": {"arms": list(ARMS),
                    "content_seeds": list(range(2026, 2031)),
                    "train_seeds": list(TRAIN_SEEDS)},
        "note": ("no training ran; the real fit is executed by the user "
                 "via scripts/manual_material_residual_scale.ps1 "
                 "-Mode fit"),
    }
    dump_json(output_dir / "report.json", report)
    ctx["log"](f"dry-run OK: {len(folds)} folds, {len(matrix)} planned "
               f"fits, scales "
               f"{ {k: round(v['scale'], 4) for k, v in scales.items()} }")
    return report


def run_admission(ctx, cfg: dict, feature_dir: Path, p0_fit_dir: Path,
                  log) -> dict:
    """Every identity check, BEFORE any controlled computation: the fix
    cache binding (the frozen expectation), this experiment's cache
    markers, the anchors, the frozen chain (protocol/global cache/P0
    summary), the generic cache contract, the per-fold P0 identity and
    the scale records. Returns everything the run needs."""
    import scripts.train_material_residual as frozen

    registry = ctx["registry"]
    # the cache's own binding record is the frozen expectation for the
    # three bound files; register against it BEFORE any further read
    binding_path = feature_dir / "cache_binding.json"
    if not binding_path.is_file():
        raise ScaleRefused(
            f"cache binding record missing: {binding_path}")
    binding = load_json(binding_path)
    for name, record in (binding.get("files") or {}).items():
        # the key MUST be the registry's own project-relative as_posix
        # lookup key; an absolute key would silently never hit
        registry.bindings[(feature_dir / name).relative_to(
            _PROJECT_ROOT).as_posix()] = {
            "source": "scale_cache_binding",
            "expected_sha256": str(record["sha256"]),
            "expected_bytes": int(record["bytes"])}
    for name in ("cache_binding.json", "feature_cache.npz",
                 "feature_manifest.json", "feature_definition.json"):
        registry.register(feature_dir / name, f"fix_cache:{name}")
    definition = verify_scale_cache_markers(feature_dir, cfg)
    log(f"fix cache markers OK: kind={definition['cache_kind']} "
        f"upsample={definition['upsample_version']} "
        f"frames={definition['frames_built']}")
    for name, anchor in cfg["anchors"].items():
        verify_anchor(_PROJECT_ROOT / anchor["path"], anchor["sha256"],
                      registry, f"anchor:{name}")
    ctx["stage"]("anchors", "COMPLETED", names=sorted(cfg["anchors"]))
    # the verified anchors record the HISTORICAL expectations for the
    # consumed P0/cache/identity files; expand them BEFORE any consuming
    # read so a drifted file is refused at registration, never re-hashed
    # into a fresh identity
    ctx["watch"].check("before historical binding expansion")
    expansion = expand_historical_bindings(registry, cfg, _PROJECT_ROOT)
    ctx["stage"]("historical_bindings", "COMPLETED", **expansion["added"])
    # registration (with hash verification against the expanded
    # historical expectations) happens BEFORE the frozen chain consumes
    # anything
    global_dir = _PROJECT_ROOT / cfg["features_source"]["path"]
    for name in ("cache_binding.json", "feature_cache.npz",
                 "feature_manifest.json", "feature_definition.json"):
        path = global_dir / name
        if path.is_file():
            registry.register(path, f"global_cache:{name}")
    registry.register(_PROJECT_ROOT / cfg["protocol_source"]["path"],
                      "protocol_file")
    if not (p0_fit_dir / "summary.json").is_file():
        raise ScaleRefused(
            f"P0 fit summary missing: {p0_fit_dir / 'summary.json'}")
    for name in ("summary.json", "admission.json"):
        path = p0_fit_dir / name
        if path.is_file():
            registry.register(path, f"p0_fit:{name}")
    for fold_dir in sorted((p0_fit_dir / "models").glob("*_P0")):
        for name in ("model_state.pt", "scaler.json", "history.json",
                     "outer_eval_predictions.jsonl"):
            path = fold_dir / name
            if path.is_file():
                registry.register(path,
                                  f"p0_model:{fold_dir.name}:{name}")
    admission = frozen.run_admission(cfg, p0_fit_dir)
    protocol = admission["protocol"]
    report = verify_local_cache(feature_dir, protocol)
    if report["sample_count"] != len(admission["manifest"]["samples"]):
        raise ScaleRefused(
            f"fix cache rows {report['sample_count']} != the frozen "
            f"manifest {len(admission['manifest']['samples'])}")
    ctx["stage"]("fix_cache", "COMPLETED",
                 sample_count=report["sample_count"],
                 definition_version=report["definition_version"])
    manifest_samples = admission["manifest"]["samples"]
    authority = admission["authority"]
    fix_cache = dict(np.load(feature_dir / "feature_cache.npz"))
    p0_verified = {}
    for fold_id in sorted(authority_sample_folds(authority)):
        p0_verified[fold_id] = verify_p0_base_identity(
            fold_id=fold_id, p0_fit_dir=p0_fit_dir, admission=admission,
            global_cache=admission["cache"], local_cache=fix_cache,
            manifest_samples=manifest_samples, authority=authority,
            log=log)
    ctx["stage"]("p0_identity", "COMPLETED", folds=sorted(p0_verified))
    scales = {}
    for fold_id in sorted(p0_verified):
        parts = fold_parts(manifest_samples, authority, fold_id)
        verified = p0_verified[fold_id]
        rows = parts["inner_train"]
        logits = local_entry_p0_logits(verified["head"], verified["scaler"],
                                       admission["cache"], rows)
        scale, info = fold_scale(logits)
        scales[fold_id] = info
        log(f"  scale {fold_id}: s_f = {scale:.6f} (median margin "
            f"{info['margin_median']:.6f}, n = {info['n']})")
    ctx["stage"]("scales", "COMPLETED",
                 **{fold: scales[fold]["scale"] for fold in scales})
    return {"admission": admission, "p0_verified": p0_verified,
            "scales": scales, "definition": definition}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="material_residual_scale_v1 training entry "
                    "(dry-run or explicit fit).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-dir", required=True,
                        help="this experiment's complete fix-version "
                             "feature cache (cache_build/features)")
    parser.add_argument("--p0-fit-dir", required=True,
                        help="the frozen P0 fit directory (must match the "
                             "config's p0_baseline_source)")
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; must not exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--deadline-utc", default=None,
                        help="mutually exclusive with --budget-minutes")
    args = parser.parse_args(argv)
    if args.dry_run and args.fit:
        print("refused: --dry-run and --fit are mutually exclusive",
              file=sys.stderr)
        return EXIT_REFUSED
    if not args.dry_run and not args.fit:
        print("refused: pass exactly one of --dry-run / --fit",
              file=sys.stderr)
        return EXIT_REFUSED
    if (args.budget_minutes is None) == (args.deadline_utc is None):
        print("refused: pass exactly one of --budget-minutes / "
              "--deadline-utc", file=sys.stderr)
        return EXIT_REFUSED
    if args.budget_minutes is not None and (
            not np.isfinite(args.budget_minutes)
            or args.budget_minutes <= 0):
        print("refused: --budget-minutes must be positive", file=sys.stderr)
        return EXIT_REFUSED

    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
              f"{message}", file=sys.stderr, flush=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    try:
        cfg = load_scale_config(config_path)
    except ScaleRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    feature_dir = Path(args.feature_dir)
    if not feature_dir.is_absolute():
        feature_dir = _PROJECT_ROOT / feature_dir
    p0_fit_dir = Path(args.p0_fit_dir)
    if not p0_fit_dir.is_absolute():
        p0_fit_dir = _PROJECT_ROOT / p0_fit_dir
    config_p0_dir = _PROJECT_ROOT / cfg["p0_baseline_source"]["fit_dir"]
    if p0_fit_dir.resolve() != config_p0_dir.resolve():
        print(f"refused: --p0-fit-dir {p0_fit_dir} does not match the "
              f"config's P0 source {config_p0_dir}", file=sys.stderr)
        return EXIT_REFUSED
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    if output_dir.exists():
        print(f"refused: output directory already exists: {output_dir}",
              file=sys.stderr)
        return EXIT_REFUSED
    if is_link_or_reparse(output_dir) or \
            is_link_or_reparse(output_dir.parent):
        print("refused: output directory or parent is a link/reparse point",
              file=sys.stderr)
        return EXIT_REFUSED

    registry = SourceRegistry({})
    try:
        watch = DeadlineWatch(deadline_utc=args.deadline_utc
                              if args.deadline_utc is not None else None,
                              budget_minutes=args.budget_minutes)
    except ScaleRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    stages: dict = {}

    def stage(name: str, status: str, **details) -> None:
        stages[name] = {"status": status, **details}
        log(f"stage {name}: {status}")

    ctx = {"registry": registry, "watch": watch, "stage": stage,
           "log": log, "p0_verified": {}}
    failure = None
    interrupted = False
    mismatch = False
    protection = None
    output_dir.mkdir(parents=True)
    try:
        try:
            registry.register(config_path, "experiment_config",
                              kind="experiment_config")
            wired = run_admission(ctx, cfg, feature_dir, p0_fit_dir, log)
            admission = wired["admission"]
            ctx["p0_verified"] = wired["p0_verified"]
            scales = wired["scales"]
            registry.register(feature_dir / "feature_cache.npz",
                              "fix_cache:feature_cache")
            admission_record = {
                "experiment": EXPERIMENT,
                "protocol_sha256": protocol_sha256_of(admission),
                "upsample_version": UPSAMPLE_FIX_VERSION,
                "fix_cache": {
                    "path": str(feature_dir),
                    "cache_kind": wired["definition"]["cache_kind"],
                    "frames_built": wired["definition"]["frames_built"],
                    "instances_built":
                        wired["definition"]["instances_built"],
                    "region_rule": LOCAL_REGION_RULE,
                    "input_dim": LOCAL_TOTAL_DIM,
                },
                "scales": scales,
                "planned_fits": PLANNED_FITS,
                "matrix": {"arms": list(ARMS),
                           "variants": list(VARIANTS),
                           "folds": sorted(authority_sample_folds(
                               admission["authority"])),
                           "seeds": list(TRAIN_SEEDS)},
                "p0_fit_dir": str(p0_fit_dir),
                "p0_identity": {fold: wired["p0_verified"][fold]["identity"]
                                for fold in sorted(wired["p0_verified"])},
                "class_names": list(CLASS_NAMES),
                "sample_count": len(admission["manifest"]["samples"]),
            }
            dump_json(output_dir / "admission.json", admission_record)
            if args.dry_run:
                ctx["watch"].check("before dry-run matrix publish")
                run_dry_run(ctx, cfg, admission, feature_dir, output_dir,
                            scales)
            else:
                run_fit(ctx, cfg, admission, feature_dir, output_dir,
                        scales, log)
        except DeadlineExceeded as exc:
            interrupted = True
            ctx["watch"] = None  # closeout runs deadline-exempt
            log(f"deadline exceeded: {exc}")
            if not args.dry_run and                     not (output_dir / "summary.json").exists():
                # keep the completed fits visible as PARTIAL evidence
                dump_json(output_dir / "summary.json", {
                    "mode": "fit", "experiment": EXPERIMENT,
                    "training_status": "PARTIAL",
                    "interrupted_by_deadline": True,
                    "note": ("the shared deadline expired during the fit "
                             "loop; the completed model directories are "
                             "kept and no COMPLETED status is published"),
                })
        except LocalMismatch as exc:
            mismatch = True
            failure = f"LocalMismatch: {exc}"
            log(f"mismatch: {failure}")
        except (ScaleRefused, IdentityError, LocalRefused,
                FileNotFoundError, ValueError, OSError, KeyError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            log(f"failure: {failure}")
    except Exception as exc:  # unexpected: still write a status file
        failure = f"UNEXPECTED {type(exc).__name__}: {exc}"
        log(f"unexpected failure: {failure}")

    # closeout: the protection manifest is evidence (kept even after an
    # expiry), but the SUCCESS STATUS needs its own deadline check — the
    # protection verification itself crossing the deadline must never
    # yield a COMPLETED run
    try:
        protection = registry.compare()
        registry.write_manifest(output_dir / "source_manifest.json",
                                protection)
        stage("protection",
              "COMPLETED" if protection["identical"] else "FAILED",
              registered=protection["registered_sources"],
              problems=protection["problems"])
        if not protection["identical"] and failure is None:
            failure = "protection check found changed inputs"
    except Exception as exc:
        log(f"closeout failed: {exc}")
        if failure is None:
            failure = f"CLOSEOUT {type(exc).__name__}: {exc}"
    if not interrupted and ctx["watch"] is not None:
        try:
            ctx["watch"].check("before success status publish")
        except DeadlineExceeded as exc:
            interrupted = True
            ctx["watch"] = None  # only the status write remains
            log(f"deadline crossed before the success status: {exc}")
            if failure is None and (output_dir / "summary.json").is_file():
                # the run would have been COMPLETED: restamp PARTIAL so
                # summary and run_status stay consistent
                try:
                    summary = load_json(output_dir / "summary.json")
                    if summary.get("training_status") == "COMPLETED":
                        summary["training_status"] = "PARTIAL"
                        summary["interrupted_by_deadline"] = True
                        write_json(output_dir / "summary.json", summary)
                except Exception as inner:
                    log(f"summary PARTIAL restamp failed: {inner}")
    if mismatch:
        status, code = "MISMATCH", EXIT_MISMATCH
    elif failure:
        status, code = "FAILED", EXIT_REFUSED
    elif interrupted:
        status, code = "PARTIAL", EXIT_BLOCKED
    else:
        status, code = "COMPLETED", EXIT_OK
    dump_json(output_dir / "run_status.json", {
        "mode": "dry-run" if args.dry_run else "fit",
        "experiment": EXPERIMENT, "stages": stages,
        "interrupted_by_deadline": interrupted, "failure": failure,
        "overall_status": status, "exit_code": code,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": time.time() - started, "protection": protection,
    })
    log(f"overall: {status} (exit {code})")
    return code


def protocol_sha256_of(admission: dict) -> str:
    return str(admission["protocol"]["protocol_sha256"])


if __name__ == "__main__":
    raise SystemExit(main())
