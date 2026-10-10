"""Fixed-P0 residual correction entry (RC/RCS; material_polar_residual_v1).

Two explicit modes, nothing runs without one:

- ``--dry-run`` — full admission (protocol, cache binding, P0 baseline
  re-verification, fold-set identity) plus a task preview; no head is
  built, nothing is written beyond the preview report.
- ``--fit`` — the real experiment: for every requested fold and variant
  (RC = P0 + C correction, RCS = P0 + C,S correction), the residual head
  is trained on the frozen protocol's inner_train with the frozen budget,
  selected on inner_val (epoch 0 included), and evaluated ONCE on
  outer_eval. After ALL selected models are frozen, the protocol's
  polarization shuffle diagnostic runs (5 seeds).

Admission happens BEFORE any head construction, any scaler fit and any
training artifact is written. The P0 head is loaded read-only (CPU), its
logits enter training as detached constants, and a final snapshot proves
the P0 parameters and the saved scaler files are unchanged.

Exit codes: 0 verified run; 1 REFUSED (identity/structure/binding/missing
files/output conflict); 2 MISMATCH (the P0 baseline re-verification
disagrees with the saved values); 3 BLOCKED (protocol folds not OK).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_probe import (  # noqa: E402
    Standardizer,
    accuracy,
    assemble_design_matrix,
    authoritative_sample_index,
    compare_module_states,
    macro_f1,
    snapshot_module_state,
    verify_cache_binding,
)
from core.material_residual import (  # noqa: E402
    DEFAULT_AUX_LOSS_WEIGHT,
    RESIDUAL_INPUT_DIM,
    SHUFFLE_SEEDS,
    VARIANTS,
    assemble_residual_inputs,
    fit_residual_scaler,
    p0_head_from_payload,
    p0_logits_of,
    residual_logits,
    run_shuffle_diagnostic,
    train_residual_head,
)

import scripts.verify_p0_baseline as baseline  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_MISMATCH = 2
EXIT_BLOCKED = 3

METRIC_ATOL = 1e-6
SCALER_ATOL = 1e-6


class ResidualRefused(Exception):
    """Identity/structure/binding/missing-file failure."""


class ResidualMismatch(Exception):
    """A legal finite saved value disagrees with the recomputation."""


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def load_protocol(path: str | Path) -> dict:
    protocol_path = Path(path)
    if protocol_path.is_dir():
        protocol_path = protocol_path / "protocol.json"
    if not protocol_path.is_file():
        raise ResidualRefused(f"protocol not found: {protocol_path}")
    protocol = load_json(protocol_path)
    from core.existing_data_protocol import protocol_digest

    recorded = protocol.get("protocol_sha256", "")
    if not recorded or protocol_digest(protocol) != recorded:
        raise ResidualRefused(
            f"protocol {protocol_path} failed its own digest check")
    return protocol


def run_admission(config: dict, fit_dir: Path):
    """The full read-only admission shared by dry-run and fit.

    Reuses the existing verification machinery (no second identity
    algorithm): protocol digest, cache binding bytes, per-fold authority,
    manifest rows, cache arrays, the P0 fit's admission identity and fold
    records, and a per-fold re-derivation of the P0 baseline (logits,
    metrics, scaler full columns, full history). Any baseline
    disagreement stops the residual experiment here.
    """
    protocol_path = config["protocol_source"]["path"]
    features_path = config["features_source"]["path"]
    protocol = baseline.load_protocol(PROJECT_ROOT / protocol_path)
    binding, definition, manifest, cache, authority = \
        baseline.verify_feature_cache(PROJECT_ROOT / features_path,
                                      protocol)
    if not fit_dir.is_dir():
        raise ResidualRefused(f"P0 fit directory missing: {fit_dir}")
    summary_path = fit_dir / "summary.json"
    if not summary_path.is_file():
        raise ResidualRefused(f"P0 fit summary missing: {summary_path}")
    summary = load_json(summary_path)
    baseline.check_admission_identity(summary, protocol, definition)
    saved_binding = summary.get("admission", {}).get("cache_binding",
                                                     {}).get("files", {})
    for name, record in binding["files"].items():
        if saved_binding.get(name, {}).get("sha256") != record["sha256"]:
            raise ResidualRefused(
                f"P0 fit admission binding for {name} does not match the "
                f"verified cache")
    p0_fold_records = baseline.verify_p0_fold_records(
        summary, protocol, authority, manifest["samples"])
    fold_ids = [fold["fold_id"] for fold in protocol["folds"]["folds"]]
    fold_results = {}
    for fold_id in fold_ids:
        fold_dir = fit_dir / "models" / f"{fold_id}_P0"
        fold_results[fold_id] = baseline.verify_fold(
            fold_id, fold_dir, cache, manifest["samples"], authority,
            summary, list(protocol["data"]["class_names"]),
            p0_fold_records)
    budget = summary["admission"]["budget"]
    residual_budget = config["training"]
    return {
        "protocol": protocol,
        "definition": definition,
        "manifest": manifest,
        "cache": cache,
        "authority": authority,
        "summary": summary,
        "fold_records": p0_fold_records,
        "fold_results": fold_results,
        "budget": budget,
        "residual_budget": residual_budget,
        "binding": binding,
    }


def verify_saved_scaler_matches_source(
    fold_id: str, saved_scaler: Standardizer, cache: dict,
    inner_rows: list[int],
) -> dict:
    """The residual scaler is fitted from inner_train; it must equal the
    saved P0 scaler of the same fold (same data, same rule)."""
    from scripts.verify_p0_baseline import verify_scaler_columns

    return verify_scaler_columns(cache, inner_rows, saved_scaler, fold_id)


def resolve_rows(manifest_samples: list[dict], authority: dict,
                 fold_id: str) -> dict[str, list[int]]:
    parts = {"inner_train": [], "inner_val": [], "outer_eval": []}
    for index, sample in enumerate(manifest_samples):
        role = authority[sample["sample_name"]]["roles"].get(fold_id)
        if role is None:
            raise ResidualRefused(
                f"sample {sample['sample_name']} has no {fold_id} role")
        parts[role].append(index)
    for role, rows in parts.items():
        if not rows:
            raise ResidualRefused(f"fold {fold_id}: {role} part is empty")
    return parts


def fit_one_fold_variant(
    *,
    fold_id: str,
    variant: str,
    cache: dict,
    manifest_samples: list[dict],
    authority: dict,
    parts: dict[str, list[int]],
    p0_payload: dict,
    p0_head,
    budget: dict,
    residual_budget: dict,
    output_dir: Path,
    p0_scaler_path: Path,
    log,
) -> dict:
    """Train one (fold, variant); the P0 baseline stays frozen."""
    import torch

    variant = variant
    fold_variant_dir = output_dir / "models" / f"{fold_id}_{variant}"
    fold_variant_dir.mkdir(parents=True)

    # --- the new standardizer: inner_train only, then cross-checked
    # against the saved P0 scaler of the same fold
    scaler, scaler_info = fit_residual_scaler(cache, parts["inner_train"])
    saved_scaler_payload = load_json(p0_scaler_path)
    saved_scaler = Standardizer.from_dict(saved_scaler_payload)
    # full-column check of the saved P0 scaler against its inner_train
    # source (re-run here so a changed scaler cannot slip through between
    # the baseline verification and this fit)
    scaler_check = verify_saved_scaler_matches_source(
        fold_id, saved_scaler, cache, parts["inner_train"])
    # the residual standardizer = the P0 scaler's C/S column segment
    # (identical inner_train rule; the fresh fit must match it exactly)
    c_start = cache["g_features"].shape[1]
    residual_scaler = Standardizer(
        mean=saved_scaler.mean[c_start:],
        std=saved_scaler.std[c_start:],
        usable=saved_scaler.usable[c_start:])
    if not np.allclose(residual_scaler.mean, scaler.mean, atol=SCALER_ATOL,
                       rtol=0.0) or not np.allclose(
            residual_scaler.std, scaler.std, atol=SCALER_ATOL, rtol=0.0) \
            or not np.array_equal(residual_scaler.usable, scaler.usable):
        raise ResidualRefused(
            f"fold {fold_id}: the freshly fitted residual scaler differs "
            "from the saved P0 scaler's C/S segment; the protocol "
            "requires the identical inner_train standardization")
    scaler_for_run = residual_scaler

    def _slot_inputs(rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
        c = cache["c_features"][rows]
        s = cache["s_features"][rows]
        s_valid = cache["s_valid"][rows].astype(bool)
        width_c = c.shape[1]
        width = width_c + s.shape[1]
        column_valid = np.ones((len(rows), width), dtype=bool)
        column_valid[:, width_c:] = s_valid[:, None]
        raw = np.concatenate([c, s], axis=1)
        standardized = scaler_for_run.transform(raw, column_valid)
        c_std = standardized[:, :width_c]
        s_std = standardized[:, width_c:]
        x14 = assemble_residual_inputs(c_std, s_std, s_valid, variant)
        return x14, s_valid

    x_train, valid_train = _slot_inputs(parts["inner_train"])
    x_val, valid_val = _slot_inputs(parts["inner_val"])
    x_eval, valid_eval = _slot_inputs(parts["outer_eval"])
    labels_train = cache["class_id"][parts["inner_train"]].astype(np.int64)
    labels_val = cache["class_id"][parts["inner_val"]].astype(np.int64)
    labels_eval = cache["class_id"][parts["outer_eval"]].astype(np.int64)

    # --- the frozen P0 design: G slot only, saved scaler semantics
    def _p0_design(rows: list[int]) -> np.ndarray:
        g = cache["g_features"][rows]
        c = cache["c_features"][rows]
        s = cache["s_features"][rows]
        s_valid = cache["s_valid"][rows].astype(bool)
        width_c = c.shape[1]
        width = g.shape[1] + width_c + s.shape[1]
        column_valid = np.ones((len(rows), width), dtype=bool)
        column_valid[:, g.shape[1] + width_c:] = s_valid[:, None]
        raw = np.concatenate([g, c, s], axis=1)
        standardized = saved_scaler.transform(raw, column_valid)
        g_dim = g.shape[1]
        return assemble_design_matrix(
            standardized[:, :g_dim],
            standardized[:, g_dim:g_dim + width_c],
            standardized[:, g_dim + width_c:], "P0")

    p0_design_eval = _p0_design(parts["outer_eval"])
    if p0_design_eval.shape[1] != int(p0_payload["input_dim"]):
        raise ResidualRefused(
            f"fold {fold_id}: P0 design width {p0_design_eval.shape[1]} != "
            f"checkpoint input_dim {p0_payload['input_dim']}")
    p0_train = p0_logits_of(p0_head, _p0_design(parts["inner_train"]))
    p0_val = p0_logits_of(p0_head, _p0_design(parts["inner_val"]))
    p0_eval = p0_logits_of(p0_head, p0_design_eval)

    # --- the frozen residual budget
    head, history, best_state, best = train_residual_head(
        x_train, labels_train, p0_train, valid_train.astype(np.float32),
        x_val, labels_val, p0_val, valid_val.astype(np.float32),
        seed=int(budget["seed"]),
        lr=float(residual_budget["lr"]),
        weight_decay=float(residual_budget["weight_decay"]),
        max_epochs=int(residual_budget["epochs"]),
        aux_loss_weight=float(residual_budget.get("aux_loss_weight",
                                                  DEFAULT_AUX_LOSS_WEIGHT)),
    )
    selection = {
        "epoch": best["epoch"],
        "inner_val_macro_f1": best["inner_val_macro_f1"],
        "inner_val_accuracy": best["inner_val_accuracy"],
        "rule": "inner_val macro_f1 > accuracy > earlier epoch (epoch 0 = "
                "no correction included)",
    }
    head.load_state_dict(best_state)
    head.eval()

    # --- the frozen P0 head snapshot must be byte-identical after training
    p0_after = snapshot_module_state(p0_head)
    p0_check = {
        "unchanged": True,
        "note": "P0 parameters/buffers compared before vs after training",
    }
    # (p0_head has no buffers; the weight check below is exact)

    with torch.no_grad():
        x_eval_t = torch.from_numpy(x_eval)
        valid_eval_t = torch.from_numpy(valid_eval.astype(np.float32))
        delta = head_module_delta(head, x_eval_t, valid_eval_t).numpy()
        final = (torch.from_numpy(p0_eval) + torch.from_numpy(delta)).numpy()
    predictions = [int(v) for v in final.argmax(axis=1)]
    labels_eval_list = [int(v) for v in labels_eval]

    torch.save({
        "state_dict": best_state,
        "variant": variant,
        "fold": fold_id,
        "seed": int(budget["seed"]),
        "input_dim": RESIDUAL_INPUT_DIM,
        "num_classes": 4,
    }, fold_variant_dir / "residual_state.pt")
    write_json(fold_variant_dir / "scaler.json", scaler.to_dict())
    write_json(fold_variant_dir / "scaler_check.json",
               {"recomputed": scaler_info,
                "matches_saved_p0": True,
                "max_diff_note": f"atol {SCALER_ATOL}"})
    write_json(fold_variant_dir / "history.json",
               {"selection": selection, "epochs": history})
    with (fold_variant_dir / "outer_eval_predictions.jsonl").open(
            "w", encoding="utf-8") as handle:
        for index, cache_row in enumerate(parts["outer_eval"]):
            sample = manifest_samples[cache_row]
            handle.write(json.dumps({
                "sample_name": sample["sample_name"],
                "group": sample["group"],
                "class_id": labels_eval_list[index],
                "prediction": predictions[index],
                "logits": [float(v) for v in final[index]],
                "delta": [float(v) for v in delta[index]],
                "p0_logits": [float(v) for v in p0_eval[index]],
                "has_valid_s": bool(valid_eval[index]),
            }, sort_keys=True) + "\n")

    return {
        "fold": fold_id,
        "variant": variant,
        "selection": selection,
        "rows": len(x_eval),
        "labels": labels_eval_list,
        "predictions": predictions,
        "final_logits": final,
        "p0_logits": p0_eval,
        "delta": delta,
        "has_valid_s": valid_eval.astype(bool).tolist(),
        "recomputed": _metrics(labels_eval_list, predictions),
        "scaler_check": scaler_check,
        "dir": fold_variant_dir,
    }


def head_module_delta(head, x14, valid_s):
    import torch

    from core.material_residual import compute_delta

    return compute_delta(head, x14, valid_s)


def _metrics(labels: list[int], predictions: list[int]) -> dict:
    class_names = ("metal_submarine", "plastic_submarine",
                   "plastic_fish", "real_fish")
    confusion = [[0] * 4 for _ in range(4)]
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1
    recalls = {}
    for class_id in range(4):
        total = sum(confusion[class_id])
        recalls[class_names[class_id]] = (
            confusion[class_id][class_id] / total if total else None)
    return {
        "macro_f1": macro_f1(labels, predictions),
        "accuracy": accuracy(labels, predictions),
        "confusion_matrix": confusion,
        "per_class_recall": recalls,
    }


def paired_changes(base_predictions: list[int], other_predictions: list[int],
                   labels: list[int]) -> dict:
    fixed = broke = both_correct = both_wrong = 0
    changes = []
    for index, (base, other, label) in enumerate(
            zip(base_predictions, other_predictions, labels)):
        base_ok = base == label
        other_ok = other == label
        if other_ok and not base_ok:
            fixed += 1
            changes.append({"index": index, "change": "fixed"})
        elif base_ok and not other_ok:
            broke += 1
            changes.append({"index": index, "change": "broke"})
        elif base_ok and other_ok:
            both_correct += 1
        else:
            both_wrong += 1
    return {"fixed": fixed, "broke": broke,
            "both_correct": both_correct, "both_wrong": both_wrong,
            "changes": changes}


def run_shuffle_for_fold(
    *,
    fold_id: str,
    cache: dict,
    manifest_samples: list[dict],
    authority: dict,
    parts: dict[str, list[int]],
    scaler: Standardizer,
    p0_head,
    residual_payload_path: Path,
    budget: dict,
    residual_budget: dict,
    output_dir: Path,
    log,
    capture: dict | None = None,
) -> dict:
    """The protocol shuffle diagnostic over the FROZEN selected RCS head.

    The diagnostic consumes the fold's STANDARDIZED C/S slots (exactly the
    normal RCS evaluation inputs) and compares against the unshuffled RCS
    forward, P0 and — when available — the RC/R0 results, all aligned by
    sample_name. Per-sample logits are KEPT (never discarded).
    """
    import torch

    variant_dir = output_dir / "models" / f"{fold_id}_RCS"
    payload = torch.load(residual_payload_path, weights_only=False,
                         map_location="cpu")
    head = build_fresh_head(int(budget["seed"]))
    head.load_state_dict(payload["state_dict"])
    head.eval()

    rows = parts["outer_eval"]
    c = cache["c_features"][rows]
    s = cache["s_features"][rows]
    s_valid = cache["s_valid"][rows].astype(bool)
    width_c = c.shape[1]
    # the P0 scaler is full-width (G+C+S); use its C/S segment for the
    # residual inputs — the SAME scaler and assembly as the normal
    # RCS evaluation
    g_width = cache["g_features"].shape[1]
    residual_scaler = Standardizer(
        mean=scaler.mean[g_width:],
        std=scaler.std[g_width:],
        usable=scaler.usable[g_width:])
    column_valid = np.ones((len(rows), width_c + s.shape[1]), dtype=bool)
    column_valid[:, width_c:] = s_valid[:, None]
    raw = np.concatenate([c, s], axis=1)
    standardized = residual_scaler.transform(raw, column_valid)
    c_std = standardized[:, :width_c]
    s_std = standardized[:, width_c:]
    labels = [int(v) for v in cache["class_id"][rows]]
    sample_names = [manifest_samples[row]["sample_name"] for row in rows]
    train_c = cache["c_features"][parts["inner_train"]]
    context = {
        "valid_ratio": c[:, 0].astype(np.float64),
        "gray_mean": c[:, 3].astype(np.float64),
        "split": [manifest_samples[row]["split"] for row in rows],
        "group": [manifest_samples[row]["group"] for row in rows],
        "class_id": labels,
        "quantile_edges": {
            "valid_ratio": np.quantile(
                train_c[:, 0].astype(np.float64), [1 / 3, 2 / 3]),
            "gray_mean": np.quantile(
                train_c[:, 3].astype(np.float64), [1 / 3, 2 / 3]),
        },
    }
    # P0 logits for the evaluation rows, computed once (read-only)
    p0_eval = p0_logits_of(p0_head,
                           _p0_design_for_rows(rows, cache, scaler))

    if capture is not None:
        # test hook: expose exactly what the diagnostic receives
        capture["c_std"] = c_std
        capture["s_std"] = s_std

    def evaluator(x14, valid_vector):
        with torch.no_grad():
            final = residual_logits(
                head, torch.from_numpy(x14),
                torch.from_numpy(valid_vector.astype(np.float32)),
                torch.from_numpy(p0_eval))
        return final.numpy()

    report = run_shuffle_diagnostic(
        context=context,
        c_standardized=c_std,
        s_standardized=s_std,
        s_valid=s_valid,
        evaluator=evaluator,
        sample_names=sample_names,
        seeds=SHUFFLE_SEEDS,
    )

    # ---- normal (unshuffled) RCS predictions for the paired statistics
    normal_by_name = {}
    for seed_key, seed_report in report["per_seed"].items():
        for record in seed_report["samples"]:
            normal_by_name[record["sample_name"]] = record[
                "normal_logits"]
    normal_predictions = {
        name: int(np.asarray(logits).argmax())
        for name, logits in normal_by_name.items()}

    # optional comparators: RC and R0 predictions saved by the fit
    comparators = {}
    for variant, filename in (("RC", "outer_eval_predictions.jsonl"),
                              ("R0", "outer_eval_predictions.jsonl")):
        path = output_dir / "models" / f"{fold_id}_{variant}" / filename
        if path.is_file():
            comparators[variant] = {
                json.loads(line)["sample_name"]:
                    int(json.loads(line)["prediction"])
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()}

    # ---- per-seed metrics and pairings, aligned by sample_name
    labels_by_name = {
        manifest_samples[row]["sample_name"]: int(
            cache["class_id"][row]) for row in rows}
    group_by_name = {
        manifest_samples[row]["sample_name"]:
            manifest_samples[row]["group"] for row in rows}
    for seed_key, seed_report in report["per_seed"].items():
        shuffled_predictions = {}
        for record in seed_report["samples"]:
            name = record["sample_name"]
            shuffled_predictions[name] = record["shuffled_prediction"]
            # per-sample correction deltas (normal/shuffled vs P0)
            p0_row = p0_eval[record["receiver_row"]]
            record["normal_delta"] = (
                np.asarray(record["normal_logits"], dtype=np.float32)
                - p0_row).tolist()
            record["shuffled_delta"] = (
                np.asarray(record["shuffled_logits"], dtype=np.float32)
                - p0_row).tolist()
        names = [r["sample_name"] for r in seed_report["samples"]]
        aligned_labels = [labels_by_name[name] for name in names]
        aligned_shuffled = [shuffled_predictions[name] for name in names]
        aligned_normal = [normal_predictions[name] for name in names]
        seed_report["macro_f1"] = macro_f1(aligned_labels,
                                           aligned_shuffled)
        seed_report["accuracy"] = accuracy(aligned_labels,
                                           aligned_shuffled)
        # metrics over the ACTUALLY MATCHED subset only
        matched_names = [r["sample_name"] for r in seed_report["samples"]
                         if r["status"] == "matched"]
        if matched_names:
            seed_report["matched_subset_macro_f1"] = macro_f1(
                [labels_by_name[n] for n in matched_names],
                [shuffled_predictions[n] for n in matched_names])
        else:
            seed_report["matched_subset_macro_f1"] = None
        changes = paired_changes(aligned_normal, aligned_shuffled,
                                 aligned_labels)
        seed_report["vs_normal_rcs"] = {
            key: changes[key] for key in
            ("fixed", "broke", "both_correct", "both_wrong")}
        seed_report["vs_normal_rcs"]["changed_samples"] = [
            {"sample_name": sample_names[change["index"]],
             "group": group_by_name[sample_names[change["index"]]],
             "change": change["change"]}
            for change in changes["changes"]]
        for variant, comparator in comparators.items():
            if set(comparator) >= set(names):
                changes = paired_changes(
                    [comparator[name] for name in names],
                    aligned_shuffled, aligned_labels)
                seed_report[f"vs_{variant}"] = {
                    key: changes[key] for key in
                    ("fixed", "broke", "both_correct", "both_wrong")}
            else:
                seed_report[f"vs_{variant}"] = "NOT_COMPARABLE"
    # per-class and per-group fixed/broke over the seeds
    per_class = {}
    per_group = {}
    for seed_key, seed_report in report["per_seed"].items():
        for record in seed_report["samples"]:
            name = record["sample_name"]
            label = labels_by_name[name]
            class_name = class_names_for(label)
            normal_ok = record["normal_prediction"] == label
            shuffled_ok = record["shuffled_prediction"] == label
            change = ("fixed" if shuffled_ok and not normal_ok
                      else "broke" if normal_ok and not shuffled_ok
                      else "both_correct" if normal_ok
                      else "both_wrong")
            per_class.setdefault(class_name, {}).setdefault(
                seed_key, {}).setdefault(change, 0)
            per_class[class_name][seed_key][change] += 1
            per_group.setdefault(record["group"], {}).setdefault(
                seed_key, {}).setdefault(change, 0)
            per_group[record["group"]][seed_key][change] += 1
    report["per_class_changes"] = per_class
    report["per_group_changes"] = per_group

    shuffle_dir = output_dir / "shuffle" / fold_id
    shuffle_dir.mkdir(parents=True)
    for seed_key, seed_report in report["per_seed"].items():
        write_json(shuffle_dir / f"seed_{seed_key}.json", seed_report)
    write_json(shuffle_dir / "shuffle_summary.json", {
        "seeds": report["seeds"],
        "fold": fold_id,
        "identity_check": report["identity_check"],
        "coverage": {k: v["coverage"] for k, v in
                     report["per_seed"].items()},
        "unmatched": {k: v["unmatched"] for k, v in
                      report["per_seed"].items()},
        "invalid_rows": {k: v["invalid_rows"] for k, v in
                         report["per_seed"].items()},
        "same_group_donor_ratio": {k: v["same_group_donor_ratio"]
                                   for k, v in report["per_seed"].items()},
        "macro_f1": {k: v["macro_f1"] for k, v in
                     report["per_seed"].items()},
        "matched_subset_macro_f1": {k: v["matched_subset_macro_f1"]
                                    for k, v in report["per_seed"].items()},
        "per_class_changes": per_class,
        "per_group_changes": per_group,
        "note": report["note"],
    })
    return {
        "fold": fold_id,
        "coverage": {k: v["coverage"] for k, v in
                     report["per_seed"].items()},
        "macro_f1": {k: v["macro_f1"] for k, v in
                     report["per_seed"].items()},
    }


def class_names_for(label: int) -> str:
    return ("metal_submarine", "plastic_submarine",
            "plastic_fish", "real_fish")[label]


def build_fresh_head(seed: int):
    from core.material_residual import build_residual_head

    return build_residual_head(seed=seed)


def _p0_design_for_rows(rows, cache, scaler):
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
    width_c = c.shape[1]
    return assemble_design_matrix(
        standardized[:, :g_dim],
        standardized[:, g_dim:g_dim + width_c],
        standardized[:, g_dim + width_c:], "P0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fixed-P0 residual correction experiment (RC/RCS).")
    parser.add_argument("--config", required=True,
                        help="material_polar_residual_v1.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", default="all",
                        help="all | fold0 | fold1 | ...")
    parser.add_argument("--variants", nargs="+", default=["RC", "RCS"],
                        choices=["RC", "RCS"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.fit:
        parser.print_help()
        print(
            "\nrefused: choose an explicit mode — --dry-run (admission and "
            "preview only) or --fit (the real experiment). Nothing runs by "
            "default.",
            file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run and args.fit:
        print("refused: --dry-run and --fit are mutually exclusive",
              file=sys.stderr)
        return EXIT_REFUSED
    if Path(args.output_dir).exists():
        print(
            f"refused: output directory {args.output_dir} already exists; "
            "residual runs are never overwritten",
            file=sys.stderr)
        return EXIT_REFUSED

    import yaml

    started = time.perf_counter()

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        config = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8"))
        if config.get("experiment") != "material_polar_residual_v1":
            raise ResidualRefused(
                f"config experiment {config.get('experiment')!r} is not "
                "material_polar_residual_v1")
        fit_dir = PROJECT_ROOT / config["p0_baseline_source"]["fit_dir"]
        admission = run_admission(config, fit_dir)
        protocol = admission["protocol"]
        fold_status = protocol["folds"]["status"]
        all_folds = [fold["fold_id"] for fold in protocol["folds"]["folds"]]
        if args.fold == "all":
            selected_folds = all_folds
        elif args.fold in all_folds:
            selected_folds = [args.fold]
        else:
            raise ResidualRefused(
                f"fold {args.fold!r} is not in the protocol folds "
                f"{all_folds}")
        for variant in args.variants:
            if variant not in VARIANTS:
                raise ResidualRefused(f"unknown variant {variant!r}")
    except baseline.VerificationMismatch as error:
        print(f"MISMATCH: {error}", file=sys.stderr)
        return EXIT_MISMATCH
    except baseline.VerificationRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except ResidualMismatch as error:
        print(f"MISMATCH: {error}", file=sys.stderr)
        return EXIT_MISMATCH
    except (ResidualRefused, ValueError, FileNotFoundError) as error:
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED

    previews = []
    for fold_id in selected_folds:
        parts = resolve_rows(admission["manifest"]["samples"],
                             admission["authority"], fold_id)
        previews.append({
            "fold": fold_id,
            "inner_train": len(parts["inner_train"]),
            "inner_val": len(parts["inner_val"]),
            "outer_eval": len(parts["outer_eval"]),
            "p0_baseline": {
                "macro_f1": admission["fold_results"][fold_id]
                ["recomputed"]["macro_f1"],
                "selected_epoch": admission["fold_results"][fold_id]
                ["selected_epoch"],
            },
            "variants": list(args.variants),
        })

    admission_record = {
        "protocol_sha256": protocol["protocol_sha256"],
        "cache_binding": {
            name: record["sha256"]
            for name, record in admission["binding"]["files"].items()},
        "p0_fit_dir": str(fit_dir),
        "p0_fit_summary_sha256": hashlib.sha256(
            (fit_dir / "summary.json").read_bytes()).hexdigest(),
        "p0_baseline_verification": config["p0_baseline_source"][
            "verification"],
        "class_names": list(protocol["data"]["class_names"]),
        "folds": selected_folds,
        "variants": list(args.variants),
        "budget": {
            "seed": admission["budget"]["seed"],
            "residual": {
                key: admission["residual_budget"][key]
                for key in ("epochs", "lr", "weight_decay",
                            "aux_loss_weight", "device", "full_batch")
                if key in admission["residual_budget"]},
            "p0_max_epochs": admission["budget"]["max_epochs"],
        },
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "sample_count": len(admission["manifest"]["samples"]),
    }

    if args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True)
        write_json(output_dir / "admission.json", admission_record)
        report = {
            "mode": "dry-run",
            "training_status": "NOT_RUN",
            "fold_status": fold_status,
            "admission": admission_record,
            "preview": previews,
            "note": ("no head was built and no training artifact was "
                     "written; a passing dry-run is admission only, NOT "
                     "the experiment result"),
        }
        write_json(output_dir / "report.json", report)
        print(f"dry-run preview written: {output_dir / 'report.json'}",
              file=sys.stderr)
        for entry in previews:
            print(f"  {entry['fold']}: train {entry['inner_train']} / "
                  f"val {entry['inner_val']} / eval {entry['outer_eval']} "
                  f"(P0 macro_f1 {entry['p0_baseline']['macro_f1']:.4f})",
                  file=sys.stderr)
        return EXIT_OK

    if fold_status != "OK":
        print(
            "refused: the protocol fold assignment is not OK; fitting "
            "requires a valid group-isolated fold assignment",
            file=sys.stderr)
        return EXIT_BLOCKED

    # ---------------- fit mode ----------------
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)
    write_json(output_dir / "admission.json", admission_record)
    write_json(output_dir / "provenance.json", {
        "protocol_path": config["protocol_source"]["path"],
        "features_path": config["features_source"]["path"],
        "p0_fit_dir": str(fit_dir),
        "p0_baseline_verification": config["p0_baseline_source"][
            "verification"],
        "generated_by": "scripts/train_material_residual.py",
        "note": ("the P0 head is loaded read-only (CPU); its logits enter "
                 "training as detached constants"),
    })

    try:
        per_variant_results: dict[str, list[dict]] = {
            variant: [] for variant in args.variants}
        pooled: dict[str, dict] = {}
        fold_results_all = admission["fold_results"]
        # per-fold context for the (later) shuffle diagnostic
        shuffle_contexts: dict[str, dict] = {}
        for fold_id in selected_folds:
            parts = resolve_rows(admission["manifest"]["samples"],
                                 admission["authority"], fold_id)
            checkpoint_path = (fit_dir / "models" / f"{fold_id}_P0"
                               / "model_state.pt")
            p0_payload = torch.load(checkpoint_path, weights_only=False,
                                    map_location="cpu")
            p0_head = p0_head_from_payload(p0_payload)
            p0_before = snapshot_module_state(p0_head)
            fold_results = fold_results_all[fold_id]
            p0_predictions = fold_results_predictions(fold_results)
            fold_labels = fold_results["labels"] \
                if "labels" in fold_results else None
            fold_results_all[fold_id].setdefault("labels", fold_labels)
            fold_variant_outputs = {}
            for variant in args.variants:
                result = fit_one_fold_variant(
                    fold_id=fold_id,
                    variant=variant,
                    cache=admission["cache"],
                    manifest_samples=admission["manifest"]["samples"],
                    authority=admission["authority"],
                    parts=parts,
                    p0_payload=p0_payload,
                    p0_head=p0_head,
                    budget=admission["budget"],
                    residual_budget=admission["residual_budget"],
                    output_dir=output_dir,
                    p0_scaler_path=fit_dir / "models" / f"{fold_id}_P0"
                    / "scaler.json",
                    log=log,
                )
                per_variant_results[variant].append(result)
                fold_variant_outputs[variant] = result
                log(f"  {fold_id}/{variant}: epoch "
                    f"{result['selection']['epoch']} (val F1 "
                    f"{result['selection']['inner_val_macro_f1']:.4f}) -> "
                    f"outer macro_f1 {result['recomputed']['macro_f1']:.4f}")

            # R0 = the frozen P0 on the same rows (explicit control)
            p0_scaler = Standardizer.from_dict(load_json(
                fit_dir / "models" / f"{fold_id}_P0" / "scaler.json"))
            p0_design = _p0_design_for_rows(
                parts["outer_eval"], admission["cache"], p0_scaler)
            p0_eval = p0_logits_of(p0_head, p0_design)
            r0_predictions = [int(v) for v in p0_eval.argmax(axis=1)]
            labels_eval = fold_results_all[fold_id]["labels"]
            r0_result = {
                "fold": fold_id, "variant": "R0",
                "rows": len(labels_eval),
                "labels": labels_eval,
                "predictions": r0_predictions,
                "recomputed": _metrics(labels_eval, r0_predictions),
            }
            per_variant_results.setdefault("R0", []).append(r0_result)
            fold_variant_outputs["R0"] = r0_result

            # P0 frozen evidence: parameters and buffers byte-identical
            p0_after = snapshot_module_state(p0_head)
            p0_check = compare_module_states(p0_before, p0_after)
            if not p0_check["unchanged"]:
                raise ResidualRefused(
                    f"fold {fold_id}: the P0 head changed during the "
                    f"residual fit: {p0_check['changed_tensors']}")
            del fold_variant_outputs
            # deferred: the diagnostic runs only after EVERY requested
            # fold/variant has been fit and its checkpoint frozen
            if "RCS" in args.variants:
                shuffle_contexts[fold_id] = {
                    "parts": parts,
                    "p0_head": p0_head,
                }

        # ---- shuffle diagnostic: all requested models are frozen now
        for fold_id, fold_shuffle in shuffle_contexts.items():
            run_shuffle_for_fold(
                fold_id=fold_id,
                cache=admission["cache"],
                manifest_samples=admission["manifest"]["samples"],
                authority=admission["authority"],
                parts=fold_shuffle["parts"],
                scaler=Standardizer.from_dict(load_json(
                    fit_dir / "models" / f"{fold_id}_P0" / "scaler.json")),
                p0_head=fold_shuffle["p0_head"],
                residual_payload_path=output_dir / "models"
                / f"{fold_id}_RCS" / "residual_state.pt",
                budget=admission["budget"],
                residual_budget=admission["residual_budget"],
                output_dir=output_dir,
                log=log,
            )

        # ---- OOF summary per variant (three-fold mean vs pooled, kept
        # separate) and paired deltas on the pooled predictions
        summary_variants = {}
        labels_by_fold = {}
        for fold_id in selected_folds:
            labels_by_fold[fold_id] = fold_results_all[fold_id]["labels"]
        for variant, results in per_variant_results.items():
            folds_report = []
            pooled_labels: list[int] = []
            pooled_predictions: list[int] = []
            for result in results:
                folds_report.append({
                    "fold": result["fold"],
                    "rows": result["rows"],
                    "macro_f1": result["recomputed"]["macro_f1"],
                    "accuracy": result["recomputed"]["accuracy"],
                    "confusion_matrix":
                        result["recomputed"]["confusion_matrix"],
                    "per_class_recall":
                        result["recomputed"]["per_class_recall"],
                    **({"selection": result["selection"]}
                       if "selection" in result else {}),
                })
                pooled_labels.extend(result["labels"])
                pooled_predictions.extend(result["predictions"])
            summary_variants[variant] = {
                "folds": folds_report,
                "macro_f1_mean": float(np.mean(
                    [f["macro_f1"] for f in folds_report])),
                "accuracy_mean": float(np.mean(
                    [f["accuracy"] for f in folds_report])),
                "pooled": _metrics(pooled_labels, pooled_predictions),
                "pooled_sample_count": len(pooled_labels),
            }
        pooled_labels = []
        pooled_predictions = {}
        for variant in summary_variants:
            pooled_predictions[variant] = []
            for result in per_variant_results[variant]:
                pooled_predictions[variant].extend(result["predictions"])
                if variant == args.variants[0]:
                    pooled_labels.extend(result["labels"])
        paired = {}
        if "RC" in summary_variants and "RCS" in summary_variants:
            for base in ("R0", "RC"):
                changes = paired_changes(
                    pooled_predictions.get(base, []),
                    pooled_predictions["RCS"], pooled_labels)
                paired[f"RCS_minus_{base}"] = {
                    key: changes[key] for key in
                    ("fixed", "broke", "both_correct", "both_wrong")}
                paired[f"RCS_minus_{base}"]["pooled_macro_f1"] = {
                    base: summary_variants[base]["pooled"]["macro_f1"]
                    if base in summary_variants else None,
                    "RCS": summary_variants["RCS"]["pooled"]["macro_f1"],
                }

        summary = {
            "mode": "fit",
            "training_status": "COMPLETED",
            "admission": admission_record,
            "variants": summary_variants,
            "paired": paired,
            "selection_rule": (
                "inner_val macro_f1 > accuracy > earlier epoch (epoch 0 = "
                "no correction); outer_eval evaluated once per selected "
                "model; no refit on the full data"),
            "conclusions": {
                "classification_gain": (
                    "engineering thresholds only; not a statistical test "
                    "and not an independent-generalization claim"),
                "polarization_benefit": "NOT_ESTABLISHED_BY_THIS_ROUND",
            },
            "elapsed_s": time.perf_counter() - started,
        }
        write_json(output_dir / "summary.json", summary)
        print(f"fit complete: {output_dir / 'summary.json'}",
              file=sys.stderr)
        return EXIT_OK
    except ResidualMismatch as error:
        print(f"MISMATCH: {error}", file=sys.stderr)
        return EXIT_MISMATCH
    except (ResidualRefused, ValueError, FileNotFoundError) as error:
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


def fold_results_predictions(fold_results: dict) -> list[int]:
    return fold_results["predictions"]


if __name__ == "__main__":
    raise SystemExit(main())
