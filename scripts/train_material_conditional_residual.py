"""Conditional residual experiment (P0-output context).

Question: does the polarization correction need GRAYSCALE context? The
context comes ONLY from the frozen P0's own outputs (four softmax values
+ the top-two margin — model-output features, never calibrated
probabilities), concatenated with the frozen local C/S slots and passed
through ``Linear(input_dim, 8) -> SiLU -> Linear(8, 4, bias=False)``
with the last layer zero-initialized. Everything else is the frozen
residual protocol: ``delta = valid_s * 0.5 * tanh(head(input))``,
``CE + 0.01*mean(delta^2)``, full-batch AdamW, epoch-0 candidate,
inner_val selection, missing-S exact fallback, the RC/RCS contrast, the
v2 content-shuffle diagnostic, and the six engineering thresholds.

Seeds 2026/2027/2028 x 2 variants x 3 folds = 18 small fits; every seed
is reported, none is selected. The real fit is executed by the user via
``scripts/manual_material_conditional_residual.ps1``.
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

from core.material_conditional_residual import (  # noqa: E402
    CONTEXT_DIM,
    DEFAULT_HIDDEN,
    assemble_conditional_inputs,
    build_conditional_head,
    conditional_delta,
    conditional_input_dim,
    conditional_logits,
    p0_context,
    train_conditional_residual_head,
)
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
    shuffle_assignment,
)
from core.polar_local_features import (  # noqa: E402
    LOCAL_C_WIDTH,
    LOCAL_FEATURE_DEFINITION_VERSION,
    LOCAL_REGION_RULE,
    LOCAL_S_WIDTH,
    LOCAL_TOTAL_DIM,
    fit_local_scaler,
    verify_local_cache,
)
from scripts.train_material_local_residual import (  # noqa: E402
    CLASS_NAMES,
    EXIT_BLOCKED,
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_REFUSED,
    LocalMismatch,
    LocalRefused,
    _p0_design_for_rows,
    _standardize_local,
    count_fixed_broke,
    fold_parts,
    load_json,
    verify_p0_base_identity,
    write_json,
    write_jsonl,
    _clean,
)

EXPERIMENT = "material_conditional_residual_v1"
VARIANTS = ("RC", "RCS")
VARIANT_ORDER = ("R0", "RC", "RCS")
SEEDS = (2026, 2027, 2028)
COND_INPUT_DIM = conditional_input_dim(LOCAL_C_WIDTH, LOCAL_S_WIDTH)


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


def run_conditional_admission(config: dict, p0_fit_dir: Path):
    """The frozen old-chain admission, the local cache verification and
    the runtime P0 base identity verification (shared with the local
    entry) — all before any head is built or artifact written."""
    import scripts.train_material_residual as frozen

    if config.get("experiment") != EXPERIMENT:
        raise LocalRefused(
            f"config experiment {config.get('experiment')!r} is not "
            f"{EXPERIMENT!r}")
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
    linear_reference = verify_linear_reference(
        config=config, protocol=protocol, p0_verified=p0_verified,
        all_folds=all_folds)
    log("  linear reference bound: "
        f"{linear_reference['path']} (sha256 "
        f"{linear_reference['sha256'][:16]}..., P0 oracle match)")
    return admission, local_dir, local_report, p0_verified, \
        linear_reference


def align_paired_samples(left_record_groups: list,
                         right_record_groups: list,
                         *, expected_names: list, manifest_rows: dict):
    """Flatten per-fold per-sample records and align by sample_name.

    Returns flat ``(left_predictions, right_predictions, labels, groups)``
    in the authoritative ``expected_names`` order. Refuses duplicates,
    missing or extra samples, and class/group contradictions with a
    :class:`LocalMismatch` — zip must never silently truncate.
    """

    def flatten(groups):
        flat = {}
        for group in groups:
            for record in group:
                name = record["sample_name"]
                if name in flat:
                    raise LocalMismatch(
                        f"duplicate sample in paired evidence: {name}")
                flat[name] = record
        return flat

    left = flatten(left_record_groups)
    right = flatten(right_record_groups)
    if set(left) != set(right):
        difference = sorted(set(left) ^ set(right))[:5]
        raise LocalMismatch(
            f"paired sample sets differ (first: {difference})")
    if set(left) != set(expected_names):
        difference = sorted(set(left) ^ set(expected_names))[:5]
        raise LocalMismatch(
            "paired samples do not match the authoritative outer_eval "
            f"set (first: {difference})")
    if len(expected_names) != len(set(expected_names)):
        raise LocalMismatch(
            "the authoritative outer_eval order contains duplicates")
    out_left, out_right, labels, groups = [], [], [], []
    for name in expected_names:
        left_record = left[name]
        right_record = right[name]
        row = manifest_rows[name]
        for record in (left_record, right_record):
            if record["class_id"] != row["class_id"]:
                raise LocalMismatch(f"class_id contradiction at {name}")
            if record["group"] != row["group"]:
                raise LocalMismatch(f"group contradiction at {name}")
        out_left.append(left_record["prediction"])
        out_right.append(right_record["prediction"])
        labels.append(row["class_id"])
        groups.append(row["group"])
    return out_left, out_right, labels, groups


def verify_linear_reference(*, config: dict, protocol: dict,
                            p0_verified: dict, all_folds: list) -> dict:
    """Bind and identity-check the completed LINEAR local fit.

    The reference must be the fixed-identity local experiment (never the
    invalid historical fit): the configured summary's SHA256 must match
    the recorded expectation, the run must be a COMPLETED fit of the
    local experiment under the SAME protocol, cache definition, folds
    and class order, and its P0 oracle identities must equal ours.
    """
    import hashlib as _hashlib

    reference = config.get("linear_reference", {})
    summary_path_value = reference.get("fit_summary")
    if not summary_path_value:
        raise LocalRefused(
            "linear_reference.fit_summary is not configured; the "
            "linear-vs-conditional comparison needs a bound reference")
    path = _PROJECT_ROOT / summary_path_value
    if not path.is_file():
        raise LocalRefused(
            f"the bound linear reference summary is missing: {path}")
    sha = _hashlib.sha256(path.read_bytes()).hexdigest()
    expected_sha = reference.get("expected_sha256")
    # the expected digest is MANDATORY: a missing / empty / non-string
    # value must refuse the run, never skip the comparison
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or \
            any(char not in "0123456789abcdefABCDEF"
                for char in expected_sha):
        raise LocalRefused(
            "linear_reference.expected_sha256 must be a string of "
            "exactly 64 hexadecimal characters; got "
            f"{expected_sha!r}")
    if sha != expected_sha.lower():
        raise LocalMismatch(
            f"the linear reference summary sha256 {sha} does not match "
            f"the recorded expectation {expected_sha}; refusing to bind "
            "a different file")
    summary = load_json(path)
    if summary.get("mode") != "fit" or \
            summary.get("training_status") != "COMPLETED":
        raise LocalRefused(
            "the linear reference is not a COMPLETED fit "
            f"(mode={summary.get('mode')!r}, status="
            f"{summary.get('training_status')!r})")
    if summary.get("experiment") != "material_polar_local_interior_band_v1":
        raise LocalRefused(
            f"the linear reference experiment is "
            f"{summary.get('experiment')!r}, expected "
            "material_polar_local_interior_band_v1")
    if summary.get("region_rule") != LOCAL_REGION_RULE:
        raise LocalRefused(
            f"the linear reference region rule is "
            f"{summary.get('region_rule')!r}, expected "
            f"{LOCAL_REGION_RULE!r}")
    admission_path = path.parent / "admission.json"
    if not admission_path.is_file():
        raise LocalRefused(
            "the linear reference admission record is missing: "
            f"{admission_path}")
    ref_admission = load_json(admission_path)
    if ref_admission.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise LocalRefused(
            "the linear reference protocol differs from the conditional "
            "experiment's protocol")
    if ref_admission.get("local_cache", {}).get("definition_version") != \
            "existing_data_probe_v1_local_interior_band_v1":
        raise LocalRefused(
            "the linear reference was not trained on the frozen local "
            "feature definition")
    if ref_admission.get("folds") != list(all_folds):
        raise LocalRefused(
            f"the linear reference folds {ref_admission.get('folds')} "
            f"differ from {list(all_folds)}")
    for fold_id in all_folds:
        ours = p0_verified[fold_id]["identity"]["oracle_sha256"]
        theirs = ref_admission.get("p0_identity", {}).get(
            fold_id, {}).get("oracle_sha256")
        if theirs != ours:
            raise LocalRefused(
                f"fold {fold_id}: the linear reference's P0 oracle "
                "identity differs from this run's verified identity; "
                "the two baselines are not comparable")
    return {
        "available": True,
        "path": str(path),
        "sha256": sha,
        "mode": summary.get("mode"),
        "training_status": summary.get("training_status"),
        "experiment": summary.get("experiment"),
        "region_rule": summary.get("region_rule"),
        "protocol_sha256": ref_admission.get("protocol_sha256"),
        "folds": list(all_folds),
        "p0_oracle_match": True,
        "variants": {name: {
            "macro_f1_mean": summary["variants"][name]["macro_f1_mean"],
            "pooled_macro_f1": summary["variants"][name]["pooled"][
                "macro_f1"]}
            for name in ("R0", "RC", "RCS")
            if name in summary.get("variants", {})
            and "pooled" in summary["variants"][name]},
    }


def _fold_conditional_inputs(*, local_cache: dict, global_cache: dict,
                             parts: dict, scaler: Standardizer, p0_head,
                             p0_scaler: Standardizer):
    """The variant-SHARED parts of one fold's conditional inputs:
    standardized C/S (30), the P0-output context (5), the verified P0
    logits, labels, validity and row indices. Per-variant assembly
    happens in :func:`_variant_inputs` — the RC mask is applied there so
    the RCS inputs can never leak into RC."""
    rows_train = parts["inner_train"]
    rows_val = parts["inner_val"]
    rows_eval = parts["outer_eval"]
    std = {name: _standardize_local(local_cache, scaler, rows)
           for name, rows in (("train", rows_train), ("val", rows_val),
                              ("eval", rows_eval))}
    p0_logits = {
        "train": local_entry_p0_logits(p0_head, p0_scaler, global_cache,
                                       rows_train),
        "val": local_entry_p0_logits(p0_head, p0_scaler, global_cache,
                                     rows_val),
        "eval": local_entry_p0_logits(p0_head, p0_scaler, global_cache,
                                      rows_eval),
    }
    context = {name: p0_context(values)
               for name, values in p0_logits.items()}
    valid = {name: local_cache["s_valid"][rows].astype(bool)
             for name, rows in (("train", rows_train), ("val", rows_val),
                                ("eval", rows_eval))}
    labels = {name: [int(v) for v in local_cache["class_id"][rows]]
              for name, rows in (("train", rows_train), ("val", rows_val),
                                 ("eval", rows_eval))}
    return {
        "labels": labels, "valid": valid, "p0_logits": p0_logits,
        "context": context, "std": std,
        "rows": {"train": rows_train, "val": rows_val,
                 "eval": rows_eval},
    }


def _variant_inputs(shared: dict, variant: str) -> dict:
    """Assemble the per-variant conditional inputs from the SHARED parts.

    RC zeroes the S slot [14:30] exactly (the polarization block cannot
    reach the RC head); RCS keeps the standardized S (missing blocks are
    already zero). C, context, P0 logits, labels, row order and the
    missing state are identical between variants; nothing is mutated in
    place.
    """
    import torch

    x = {name: assemble_conditional_inputs(
        shared["std"][name][:, :LOCAL_C_WIDTH],
        shared["std"][name][:, LOCAL_C_WIDTH:],
        shared["context"][name], shared["valid"][name],
        variant) for name in shared["std"]}
    tensors = {
        "x_train": torch.from_numpy(x["train"]),
        "x_val": torch.from_numpy(x["val"]),
        "p0_train": torch.from_numpy(shared["p0_logits"]["train"]),
        "p0_val": torch.from_numpy(shared["p0_logits"]["val"]),
        "valid_train": torch.from_numpy(
            shared["valid"]["train"].astype(np.float32)),
        "valid_val": torch.from_numpy(
            shared["valid"]["val"].astype(np.float32)),
    }
    return {"x": x, "tensors": tensors}


def local_entry_p0_logits(p0_head, p0_scaler, global_cache, rows):
    """Read-only P0 forward through the verified chain (fixed design)."""
    import torch

    design = _p0_design_for_rows(rows, global_cache, p0_scaler)
    with torch.inference_mode():
        logits = p0_head(torch.from_numpy(
            np.ascontiguousarray(design, dtype=np.float32)))
    return logits.cpu().numpy().astype(np.float32)


def fit_one_conditional_variant(*, seed: int, fold_id: str,
                                variant: str, shared_inputs: dict,
                                budget: dict, output_dir: Path, log) -> dict:
    """One tiny conditional fit (seed x fold x variant).

    The per-variant inputs are assembled HERE from the shared parts, so
    RC always receives its own S-masked tensors — never the RCS inputs.
    """
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
        input_dim=COND_INPUT_DIM, hidden=int(budget.get("hidden",
                                                        DEFAULT_HIDDEN)))
    model.load_state_dict(best_state)
    model.eval()

    import torch

    with torch.inference_mode():
        eval_final = conditional_logits(
            model, torch.from_numpy(fold_inputs["x"]["eval"]),
            torch.from_numpy(shared_inputs["valid"]["eval"].astype(
                np.float32)),
            torch.from_numpy(shared_inputs["p0_logits"]["eval"]))
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
            "prediction": predictions[index],
        })
    variant_dir = output_dir / "models" / f"seed_{seed}" / \
        f"{fold_id}_{variant}"
    variant_dir.mkdir(parents=True)
    torch.save({"state_dict": copy.deepcopy(best_state),
                "input_dim": COND_INPUT_DIM,
                "hidden": int(budget.get("hidden", DEFAULT_HIDDEN)),
                "num_classes": 4, "variant": variant, "fold": fold_id,
                "seed": int(seed), "region_rule": LOCAL_REGION_RULE,
                "budget": dict(budget)},
               variant_dir / "residual_state.pt")
    write_json(variant_dir / "scaler.json",
               shared_inputs["scaler"].to_dict())
    write_json(variant_dir / "history.json", {
        "selection": {"epoch": best["epoch"],
                      "inner_val_macro_f1": best["inner_val_macro_f1"],
                      "inner_val_accuracy": best["inner_val_accuracy"]},
        "epochs": history,
    })
    write_jsonl(variant_dir / "outer_eval_predictions.jsonl", per_sample)
    fold_report = {
        "fold": fold_id, "variant": variant, "seed": int(seed),
        "support": len(shared_inputs["rows"]["eval"]),
        "selected_epoch": best["epoch"],
        "inner_val_macro_f1": best["inner_val_macro_f1"],
        "inner_val_accuracy": best["inner_val_accuracy"],
        "outer_eval_macro_f1": _clean(macro_f1(
            shared_inputs["labels"]["eval"], predictions)),
        "outer_eval_accuracy": _clean(accuracy(
            shared_inputs["labels"]["eval"], predictions)),
    }
    write_json(variant_dir / "fold_variant_report.json", fold_report)
    log(f"  seed {seed} {fold_id}/{variant}: epoch {best['epoch']} "
        f"(val F1 {best['inner_val_macro_f1']:.4f}) -> outer macro_f1 "
        f"{fold_report['outer_eval_macro_f1']:.4f}")
    return {
        "seed": int(seed), "fold": fold_id, "variant": variant,
        "rows": len(shared_inputs["rows"]["eval"]),
        "labels": shared_inputs["labels"]["eval"],
        "predictions": predictions,
        "metrics": _metrics(shared_inputs["labels"]["eval"], predictions),
        "selected": best, "fold_report": fold_report,
        "per_sample": per_sample,
    }


def verify_normal_matches_saved_rcs(*, fold_id: str, train_seed: int,
                                    variant_dir: Path, rows: list,
                                    manifest_samples: list,
                                    normal_logits: np.ndarray,
                                    normal_predictions: list) -> None:
    """Bind the diagnostic's NORMAL output to the RCS outer predictions
    saved by the SAME fold x train_seed fit.

    Repeated forward passes being self-consistent does not prove the
    diagnostic walks the same code path as the formal evaluation, so the
    freshly computed normal logits/predictions are compared per sample
    against ``outer_eval_predictions.jsonl`` (identity-checked through
    ``fold_variant_report.json``). Both sides run the same CPU float32
    path, so the comparison is EXACT — a difference is a LocalMismatch
    with the measured error, never a tolerance question.
    """
    report_path = variant_dir / "fold_variant_report.json"
    predictions_path = variant_dir / "outer_eval_predictions.jsonl"
    for path in (report_path, predictions_path):
        if not path.is_file():
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: the saved RCS "
                f"artifact to bind is missing: {path}")
    report = load_json(report_path)
    if (report.get("fold"), report.get("variant"),
            report.get("seed")) != (fold_id, "RCS", int(train_seed)):
        raise LocalMismatch(
            f"fold {fold_id} seed {train_seed}: the shuffle source "
            f"{variant_dir.name} identifies itself as fold "
            f"{report.get('fold')!r} variant {report.get('variant')!r} "
            f"seed {report.get('seed')!r}; refusing to bind a fit from "
            "a different fold, seed or variant")
    saved_records = [json.loads(line) for line in
                     predictions_path.read_text(encoding="utf-8")
                     .splitlines() if line.strip()]
    saved_by_name = {}
    for record in saved_records:
        name = record["sample_name"]
        if name in saved_by_name:
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: duplicate sample in "
                f"the saved RCS predictions: {name}")
        saved_by_name[name] = record
    expected = [manifest_samples[row] for row in rows]
    expected_names = [sample["sample_name"] for sample in expected]
    if len(expected_names) != len(set(expected_names)):
        raise LocalMismatch(
            f"fold {fold_id}: the authoritative outer_eval order "
            "contains duplicate sample names")
    if set(saved_by_name) != set(expected_names):
        missing = sorted(set(expected_names) - set(saved_by_name))[:5]
        extra = sorted(set(saved_by_name) - set(expected_names))[:5]
        raise LocalMismatch(
            f"fold {fold_id} seed {train_seed}: the saved RCS prediction "
            "set differs from the authoritative outer_eval set (missing "
            f"first: {missing}; extra first: {extra})")
    if not np.isfinite(normal_logits).all():
        raise LocalMismatch(
            f"fold {fold_id} seed {train_seed}: the freshly computed "
            "normal logits contain non-finite values")
    for index, sample in enumerate(expected):
        name = sample["sample_name"]
        record = saved_by_name[name]
        if record.get("class_id") != sample["class_id"] or \
                record.get("group") != sample["group"]:
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: the saved RCS record "
                f"contradicts the frozen sample identity at {name}")
        saved_logits = np.asarray(record.get("logits"), dtype=np.float32)
        if saved_logits.shape != (4,) or \
                not np.isfinite(saved_logits).all():
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: the saved RCS logits "
                f"at {name} are not finite four-dimensional data")
        argmax = int(saved_logits.argmax())
        if record.get("prediction") != argmax:
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: the saved RCS "
                f"prediction {record.get('prediction')!r} at {name} is "
                f"not argmax(logits)={argmax}")
        if normal_predictions[index] != \
                int(normal_logits[index].argmax()):
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: the freshly computed "
                f"normal prediction at {name} "
                f"({normal_predictions[index]!r}) is not "
                "argmax(normal_logits)="
                f"{int(normal_logits[index].argmax())}")
        if not np.array_equal(saved_logits, normal_logits[index]):
            error = float(np.max(np.abs(saved_logits
                                        - normal_logits[index])))
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: the freshly computed "
                "normal logits differ from the saved RCS logits at "
                f"{name} (max abs error {error:.3e}; both sides run the "
                "same CPU float32 path and must match exactly)")
        if normal_predictions[index] != record["prediction"]:
            raise LocalMismatch(
                f"fold {fold_id} seed {train_seed}: the freshly computed "
                f"normal prediction at {name} "
                f"({normal_predictions[index]!r}) differs from the saved "
                f"RCS prediction ({record['prediction']!r})")


def run_conditional_shuffle_for_fold(*, fold_id: str, train_seed: int,
                                     local_cache: dict,
                                     manifest_samples: list, parts: dict,
                                     scaler: Standardizer, p0_head,
                                     p0_scaler, head, global_cache: dict,
                                     output_dir: Path, log) -> dict:
    """The v2 content-shuffle diagnostic over one training seed's frozen
    conditional RCS head (composite bin = original_bin*4 + interior*2 +
    band; the full standardized S16 comes from one donor; C, the P0
    context, P0 logits and the missing state are never touched)."""
    import torch

    from core.material_residual import shuffle_assignment as _assign

    rows = parts["outer_eval"]
    n = len(rows)
    standardized = _standardize_local(local_cache, scaler, rows)
    c_std = standardized[:, :LOCAL_C_WIDTH]
    s_std = standardized[:, LOCAL_C_WIDTH:]
    s_valid = local_cache["s_valid"][rows].astype(bool)
    s_raw = local_cache["s_features"][rows]
    interior_ok = (~np.isnan(s_raw[:, :LOCAL_S_WIDTH // 2])).all(axis=1)
    band_ok = (~np.isnan(s_raw[:, LOCAL_S_WIDTH // 2:])).all(axis=1)
    labels = [int(v) for v in local_cache["class_id"][rows]]
    split_labels = [manifest_samples[row]["split"] for row in rows]
    groups = [manifest_samples[row]["group"] for row in rows]
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
    context = p0_context(local_entry_p0_logits(p0_head, p0_scaler,
                                               global_cache, rows))
    p0_logits = local_entry_p0_logits(p0_head, p0_scaler, global_cache,
                                      rows)
    p0_tensor = torch.from_numpy(p0_logits)

    def x_of(s_matrix: np.ndarray) -> np.ndarray:
        return assemble_conditional_inputs(c_std, s_matrix, context,
                                           s_valid, "RCS")

    def evaluator(x: np.ndarray, valid_vector: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            final = conditional_logits(head, torch.from_numpy(x),
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
            f"fold {fold_id} seed {train_seed}: the shuffle identity "
            "check failed")
    normal_predictions = [int(v) for v in normal_logits.argmax(axis=1)]
    # bind the freshly computed normal output to the RCS predictions the
    # fit itself saved (same fold x seed; read-only, no extra forward)
    verify_normal_matches_saved_rcs(
        fold_id=fold_id, train_seed=train_seed,
        variant_dir=output_dir / "models" / f"seed_{train_seed}"
        / f"{fold_id}_RCS",
        rows=rows, manifest_samples=manifest_samples,
        normal_logits=normal_logits,
        normal_predictions=normal_predictions)
    shuffle_dir = output_dir / "shuffle" / f"seed_{train_seed}" / fold_id
    shuffle_dir.mkdir(parents=True)
    summary_seeds = {}
    for content_seed in SHUFFLE_SEEDS:
        receivers = [row for row in range(n) if s_valid[row]]
        assignment = _assign(receivers, composite_bins, split_labels,
                             seed=content_seed)
        shuffled = s_std.copy()
        records = []
        matched = 0
        same_group = 0
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
            shuffled[row] = s_std[donor]
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
            })
        shuffled_logits = np.asarray(
            evaluator(x_of(shuffled), s_valid.astype(np.float32)),
            dtype=np.float32)
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
        vs_p0 = paired_changes(
            [int(v) for v in p0_logits.argmax(axis=1)], predictions,
            labels)
        vs_normal = paired_changes(normal_predictions, predictions, labels)
        interior_denominator = int((interior_ok & s_valid).sum())
        band_denominator = int((band_ok & s_valid).sum())
        seed_report = {
            "content_seed": int(content_seed),
            "receivers": len(receivers),
            "matched": matched,
            "coverage": _clean(matched / max(1, len(receivers))),
            "unmatched": len(receivers) - matched,
            "unmatched_reasons": (
                {"single_member_group": len(receivers) - matched}),
            "interior_valid_receivers": interior_denominator,
            "interior_swapped": sum(
                1 for r in records if r["status"] == "matched"
                and r["interior_valid"]),
            "band_valid_receivers": band_denominator,
            "band_swapped": sum(
                1 for r in records if r["status"] == "matched"
                and r["band_valid"]),
            "same_group_donor_ratio": _clean(
                same_group / matched) if matched else None,
            "macro_f1": _clean(macro_f1(labels, predictions)),
            "accuracy": _clean(accuracy(labels, predictions)),
            "vs_p0_fixed": vs_p0["fixed"],
            "vs_p0_broke": vs_p0["broke"],
            "vs_normal_rcs_fixed": vs_normal["fixed"],
            "vs_normal_rcs_broke": vs_normal["broke"],
            "grouping_note": (
                "v2 composite bin; the full standardized S16 comes from "
                "one donor; the P0 context is untouched by the shuffle"),
        }
        write_json(shuffle_dir / f"seed_{content_seed}.json",
                   {"content_seed": int(content_seed),
                    "train_seed": int(train_seed), "samples": records,
                    **seed_report})
        summary_seeds[str(content_seed)] = seed_report
        log(f"  shuffle seed {train_seed} {fold_id} content "
            f"{content_seed}: coverage {seed_report['coverage']}")
    write_json(shuffle_dir / "shuffle_summary.json",
               {"fold": fold_id, "train_seed": int(train_seed),
                "seeds": summary_seeds, "region_rule": LOCAL_REGION_RULE})
    return {"fold": fold_id, "train_seed": int(train_seed),
            "seeds": summary_seeds}


def run_fit(config: dict, admission: dict, local_dir: Path,
            output_dir: Path, selected_folds: list, log,
            p0_verified: dict, linear_reference: dict) -> dict:
    import torch

    started = time.time()
    manifest_samples = admission["manifest"]["samples"]
    authority = admission["authority"]
    global_cache = admission["cache"]
    local_cache = dict(np.load(local_dir / "feature_cache.npz"))
    budget = {
        "seed_note": "the per-fit seed is the experiment seed, not the budget seed",
        "epochs": int(config["training"]["epochs"]),
        "lr": float(config["training"]["lr"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "aux_loss_weight": float(config["training"].get(
            "aux_loss_weight", 0.01)),
        "hidden": int(config.get("conditional_head", {}).get(
            "hidden", DEFAULT_HIDDEN)),
        "device": "cpu",
        "full_batch": True,
    }
    # the verified P0 contexts and logits per fold (computed once; the
    # context depends only on the frozen P0 outputs)
    per_variant: dict = {variant: [] for variant in VARIANTS}
    r0_records_by_fold: dict = {}
    shuffle_jobs = []
    for fold_id in selected_folds:
        parts = fold_parts(manifest_samples, authority, fold_id)
        verified = p0_verified[fold_id]
        p0_head = verified["head"]
        p0_scaler = verified["scaler"]
        p0_before = snapshot_module_state(p0_head)
        scaler, _scaler_info = fit_local_scaler(local_cache,
                                                parts["inner_train"])
        for seed in SEEDS:
            shared_inputs = _fold_conditional_inputs(
                local_cache=local_cache, global_cache=global_cache,
                parts=parts, scaler=scaler, p0_head=p0_head,
                p0_scaler=p0_scaler)
            shared_inputs["manifest_samples"] = manifest_samples
            shared_inputs["scaler"] = scaler
            for variant in VARIANTS:
                result = fit_one_conditional_variant(
                    seed=seed, fold_id=fold_id, variant=variant,
                    shared_inputs=shared_inputs, budget=budget,
                    output_dir=output_dir, log=log)
                per_variant[variant].append(result)
        # R0 = the verified frozen P0 (identical for every seed)
        rows_eval = parts["outer_eval"]
        eval_logits = verified["eval_logits"]
        predictions = verified["eval_predictions"]
        labels_eval = [int(v) for v in
                       global_cache["class_id"][rows_eval]]
        r0_records = []
        for offset, row in enumerate(rows_eval):
            sample = manifest_samples[row]
            r0_records.append({
                "sample_name": sample["sample_name"],
                "group": sample["group"],
                "class_id": labels_eval[offset],
                "has_valid_s": bool(local_cache["s_valid"][row]),
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
                f"fold {fold_id}: the P0 head changed during the "
                f"conditional fits: {check['changed_tensors']}")
        if "RCS" in VARIANTS:
            for seed in SEEDS:
                shuffle_jobs.append({
                    "fold_id": fold_id, "train_seed": seed,
                    "parts": parts, "p0_head": p0_head,
                    "p0_scaler": p0_scaler, "scaler": scaler,
                })

    # ---- summary over all seeds (no seed selection) ----
    variants_summary = {}
    r0_labels, r0_predictions = [], []
    for fold_id in selected_folds:
        for record in r0_records_by_fold[fold_id]:
            r0_labels.append(record["class_id"])
            r0_predictions.append(record["prediction"])
    r0_folds = []
    for fold_id in selected_folds:
        fold_records = r0_records_by_fold[fold_id]
        labels = [r["class_id"] for r in fold_records]
        predictions = [r["prediction"] for r in fold_records]
        r0_folds.append({"fold": fold_id, "rows": len(labels),
                         **_metrics(labels, predictions)})
    variants_summary["R0"] = {
        "note": ("the frozen P0; identical across training seeds by "
                 "construction"),
        "per_seed": {str(seed): {"folds": r0_folds,
                                 "macro_f1_mean": _clean(np.mean(
                                     [f["macro_f1"] for f in r0_folds])),
                                 "pooled": _metrics(r0_labels,
                                                    r0_predictions)}
                     for seed in SEEDS},
        "macro_f1_mean_of_seeds": _clean(np.mean(
            [f["macro_f1"] for f in r0_folds])),
        "macro_f1_seed_range": None,
        "pooled_macro_f1_mean_of_seeds": _clean(_metrics(
            r0_labels, r0_predictions)["macro_f1"]),
    }
    for variant in VARIANTS:
        per_seed = {}
        seed_means = []
        for seed in SEEDS:
            results = [r for r in per_variant[variant]
                       if r["seed"] == seed]
            folds_report = []
            pooled_labels = []
            pooled_predictions = []
            for result in results:
                folds_report.append({
                    "fold": result["fold"], "rows": result["rows"],
                    "support": result["rows"],
                    "selected_epoch": result["selected"]["epoch"],
                    **result["metrics"]})
                pooled_labels.extend(result["labels"])
                pooled_predictions.extend(result["predictions"])
            fold_mean = _clean(np.mean([f["macro_f1"]
                                        for f in folds_report]))
            seed_means.append(fold_mean)
            per_seed[str(seed)] = {
                "folds": folds_report,
                "macro_f1_mean": fold_mean,
                "accuracy_mean": _clean(np.mean(
                    [f["accuracy"] for f in folds_report])),
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
                 for seed in SEEDS])),
        }

    # paired comparisons, per seed and summed over seeds: every count is
    # aligned PER SAMPLE by sample_name against the authoritative
    # outer_eval order; nested per-fold lists are never compared
    paired = {}
    class_changes: dict = {}
    group_changes: dict = {}
    manifest_rows = {sample["sample_name"]: {"class_id":
                                             sample["class_id"],
                                             "group": sample["group"]}
                     for sample in manifest_samples}
    for left, right in (("RCS", "R0"), ("RCS", "RC")):
        per_seed_counts = {}
        total = {"fixed": 0, "broke": 0, "both_correct": 0,
                 "both_wrong": 0}
        per_class_total = {name: {"fixed": 0, "broke": 0}
                           for name in CLASS_NAMES}
        per_group_total: dict = {}
        for seed in SEEDS:
            left_groups = []
            right_groups = []
            expected_names = []
            for fold_id in selected_folds:
                left_results = [r for r in per_variant[left]
                                if r["seed"] == seed
                                and r["fold"] == fold_id]
                if len(left_results) != 1:
                    raise LocalMismatch(
                        f"seed {seed} fold {fold_id}: expected exactly "
                        f"one {left} result, got {len(left_results)}")
                left_groups.append(left_results[0]["per_sample"])
                if right == "R0":
                    right_groups.append(r0_records_by_fold[fold_id])
                else:
                    right_results = [r for r in per_variant[right]
                                     if r["seed"] == seed
                                     and r["fold"] == fold_id]
                    if len(right_results) != 1:
                        raise LocalMismatch(
                            f"seed {seed} fold {fold_id}: expected "
                            f"exactly one {right} result, got "
                            f"{len(right_results)}")
                    right_groups.append(right_results[0]["per_sample"])
                expected_names.extend(
                    manifest_samples[row]["sample_name"]
                    for row in fold_parts(manifest_samples, authority,
                                          fold_id)["outer_eval"])
            aligned_left, aligned_right, labels_all, groups_all = \
                align_paired_samples(
                    left_groups, right_groups,
                    expected_names=expected_names,
                    manifest_rows=manifest_rows)
            outcome = count_fixed_broke(
                left_predictions=aligned_left,
                right_predictions=aligned_right,
                labels=labels_all, group_by_row=groups_all,
                comparison=f"{left}_minus_{right}")
            support = len(expected_names)
            four_sum = (outcome["fixed"] + outcome["broke"]
                        + outcome["both_correct"] + outcome["both_wrong"])
            if four_sum != support:
                raise LocalMismatch(
                    f"seed {seed} {left}_minus_{right}: the four "
                    f"transition counts sum to {four_sum}, expected the "
                    f"outer_eval support {support}")
            per_seed_counts[str(seed)] = {
                "fixed": outcome["fixed"], "broke": outcome["broke"],
                "both_correct": outcome["both_correct"],
                "both_wrong": outcome["both_wrong"],
                "support": support,
            }
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
        class_fixed = sum(v["fixed"] for v in per_class_total.values())
        class_broke = sum(v["broke"] for v in per_class_total.values())
        if class_fixed != total["fixed"] or class_broke != total["broke"]:
            raise LocalMismatch(
                f"{left}_minus_{right}: per-class sums do not match the "
                "seed-total paired counts")
        paired[f"{left}_minus_{right}"] = {
            "per_seed": per_seed_counts,
            "total": {**total,
                      "note": ("sample-model pair counts across seeds; "
                               "seeds are re-fits, not independent "
                               "captures")},
            "macro_f1_by_variant": {
                name: {
                    "fold_mean_of_seeds":
                        variants_summary[name]["macro_f1_mean_of_seeds"],
                    "pooled_mean_of_seeds":
                        variants_summary[name][
                            "pooled_macro_f1_mean_of_seeds"],
                }
                for name in VARIANT_ORDER},
        }
        class_changes[f"{left}_minus_{right}"] = {
            "summed_over_seeds": per_class_total}
        group_changes[f"{left}_minus_{right}"] = {
            "summed_over_seeds": per_group_total}

    summary = {
        "mode": "fit",
        "experiment": EXPERIMENT,
        "region_rule": LOCAL_REGION_RULE,
        "input_dim": COND_INPUT_DIM,
        "context_note": (
            "context = the frozen P0's four softmax values plus the "
            "top-two margin (model outputs, not calibrated probabilities)"),
        "seeds": list(SEEDS),
        "seeds_note": ("all seeds reported; none selected; seeds are "
                       "re-fits, not independent captures"),
        "selection_rule": ("inner_val macro_f1 > accuracy > earlier epoch "
                           "(epoch 0 = no correction); outer_eval evaluated "
                           "once per selected model"),
        "training_status": "FIT_COMPLETE",
        "variants": variants_summary,
        "paired": paired,
        "per_class_changes": class_changes,
        "per_group_changes": group_changes,
        "linear_reference": linear_reference,
        "p0_identity": {fold: p0_verified[fold]["identity"]
                        for fold in selected_folds},
        "elapsed_s": _clean(time.time() - started),
    }
    write_json(output_dir / "summary.json", summary)
    shuffle_summary = {}
    try:
        for job in shuffle_jobs:
            variant_dir = (output_dir / "models"
                           / f"seed_{job['train_seed']}"
                           / f"{job['fold_id']}_RCS")
            payload = torch.load(variant_dir / "residual_state.pt",
                                 weights_only=False, map_location="cpu")
            head = build_conditional_head(
                seed=int(payload["seed"]), input_dim=int(
                    payload["input_dim"]),
                hidden=int(payload["hidden"]))
            head.load_state_dict(payload["state_dict"])
            head.eval()
            shuffle_summary.setdefault(f"seed_{job['train_seed']}", {})
            shuffle_summary[f"seed_{job['train_seed']}"][
                job["fold_id"]] = run_conditional_shuffle_for_fold(
                    fold_id=job["fold_id"], train_seed=job["train_seed"],
                    local_cache=local_cache,
                    manifest_samples=manifest_samples,
                    parts=job["parts"], scaler=job["scaler"],
                    p0_head=job["p0_head"], p0_scaler=job["p0_scaler"],
                    head=head, global_cache=global_cache,
                    output_dir=output_dir, log=log)
    except Exception as error:
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
        description="Conditional (P0-context) residual experiment "
                    "(dry-run or explicit fit).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fit-dir", required=True,
                        help="the frozen P0/global fit directory (must "
                             "match the config's p0_baseline_source)")
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
        config_p0_dir = _PROJECT_ROOT / config["p0_baseline_source"][
            "fit_dir"]
        cli_p0_dir = Path(args.fit_dir)
        cli_p0_dir = (cli_p0_dir if cli_p0_dir.is_absolute()
                      else _PROJECT_ROOT / cli_p0_dir)
        if cli_p0_dir.resolve() != config_p0_dir.resolve():
            raise LocalRefused(
                f"--fit-dir {cli_p0_dir} does not match the config's P0 "
                f"source {config_p0_dir}")
        p0_fit_dir = cli_p0_dir
        admission, local_dir, local_report, p0_verified, \
            linear_reference = run_conditional_admission(config, p0_fit_dir)
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
        admission_record = {
            "protocol_sha256": protocol["protocol_sha256"],
            "local_cache": {
                "path": str(local_dir),
                "definition_version": local_report["definition_version"],
                "sample_count": local_report["sample_count"],
                "region_rule": LOCAL_REGION_RULE,
                "input_dim": LOCAL_TOTAL_DIM,
            },
            "conditional": {
                "input_dim": COND_INPUT_DIM,
                "context_dim": CONTEXT_DIM,
                "hidden": int(config.get("conditional_head", {}).get(
                    "hidden", DEFAULT_HIDDEN)),
                "seeds": list(SEEDS),
                "planned_fits": len(selected_folds) * len(VARIANTS)
                * len(SEEDS),
            },
            "p0_fit_dir": str(p0_fit_dir),
            "p0_identity": {fold: p0_verified[fold]["identity"]
                            for fold in all_folds},
            "linear_reference": linear_reference,
            "class_names": list(protocol["data"]["class_names"]),
            "folds": selected_folds,
            "sample_count": len(admission["manifest"]["samples"]),
        }
        output_dir.mkdir(parents=True)
        write_json(output_dir / "admission.json", admission_record)
        if args.dry_run:
            report = {
                "mode": "dry-run",
                "experiment": EXPERIMENT,
                "region_rule": LOCAL_REGION_RULE,
                "input_dim": COND_INPUT_DIM,
                "planned_fits": admission_record["conditional"][
                    "planned_fits"],
                "seeds": list(SEEDS),
                "note": ("no training ran; the real fit is executed by the "
                         "user via scripts/"
                         "manual_material_conditional_residual.ps1"),
            }
            write_json(output_dir / "report.json", report)
            log(f"dry-run OK: {len(selected_folds)} folds, "
                f"{admission_record['conditional']['planned_fits']} "
                f"planned fits, input_dim {COND_INPUT_DIM}")
            return EXIT_OK
        summary = run_fit(config, admission, local_dir, output_dir,
                          selected_folds, log, p0_verified,
                          linear_reference)
        log(f"fit complete: RCS macro_f1 mean of seeds "
            f"{summary['variants']['RCS']['macro_f1_mean_of_seeds']}")
        return EXIT_OK
    except LocalMismatch as error:
        print(f"MISMATCH: {error}", file=sys.stderr)
        return EXIT_MISMATCH
    except (LocalRefused, ValueError, FileNotFoundError) as error:
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
