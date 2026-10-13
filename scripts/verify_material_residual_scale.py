"""material_residual_scale_v1 — the independent verification entry.

Re-derives, from the frozen inputs and the saved fit artifacts alone:

- the per-fold scale (its OWN P0 forward over inner_train, float64) and
  demands EXACT equality with the saved scale records;
- every saved model's outer-eval predictions (reload the state, rebuild
  the head, evaluate through the same assembly with the arm magnitude)
  and demands exact agreement with the saved rows;
- the selection history (the shared selection rule applied to the saved
  epoch history must equal the saved selection);
- every reported metric (macro-F1, accuracy, confusion, per-class
  P/R/F1, per-group) by PLAIN COUNTING LOOPS over the saved per-sample
  rows — never a production summarizer;
- every paired comparison and the shuffle bindings (assignment
  reconstruction from the frozen bins + the saved donor identities).

The metric recomputation deliberately reimplements the counting: a fit
summary whose numbers were produced by anything other than the saved
rows cannot pass. Refusals cover wrong scales, wrong roles, extra or
missing samples, selection-history drift and per-class count swaps that
leave the macro-F1 unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.material_conditional_residual import (  # noqa: E402
    assemble_conditional_inputs,
    build_conditional_head,
    conditional_logits,
    p0_context,
)
from core.material_probe import select_epoch  # noqa: E402
from core.material_residual import (  # noqa: E402
    build_donor_bins,
    shuffle_assignment,
)
from core.material_residual_scale import (  # noqa: E402
    expand_historical_bindings,
    ARMS,
    EXPERIMENT,
    TRAIN_SEEDS,
    VARIANTS,
    DeadlineExceeded,
    DeadlineWatch,
    ScaleMismatch,
    ScaleRefused,
    count_metrics,
    evaluate_gates,
    fold_scale,
    load_scale_config,
    magnitude_for_arm,
    verify_scale_cache_markers,
    verify_scale_evidence,
)
from core.polar_local_features import (  # noqa: E402
    LOCAL_C_WIDTH,
    fit_local_scaler,
    verify_local_cache,
)
from scripts.train_material_local_residual import (  # noqa: E402
    CLASS_NAMES,
    EXIT_MISMATCH,
    EXIT_OK,
    EXIT_REFUSED,
    LocalMismatch,
    LocalRefused,
    _clean,
    _standardize_local,
    fold_parts,
    load_json,
    verify_p0_base_identity,
)
from scripts.train_material_conditional_residual import (  # noqa: E402
    local_entry_p0_logits,
)
from scripts.eval_material_upsample_impact import (  # noqa: E402
    IdentityError,
    SourceRegistry,
    dump_json,
    is_link_or_reparse,
    verify_anchor,
)

SHUFFLE_CONTENT_SEEDS = (2026, 2027, 2028, 2029, 2030)

# The saved shuffle report's required fields. The three layers (the
# per-content-seed file, shuffle_summary.json's seeds entry and the
# final summary["shuffle"] entry) carry the same report; every field
# listed here MUST be present — a missing field is never read as a
# legal None.
SHUFFLE_REPORT_COUNT_FIELDS = ("receivers", "matched", "unmatched",
                               "vs_normal_rcs_fixed", "vs_normal_rcs_broke")
SHUFFLE_REPORT_FLOAT_FIELDS = ("coverage", "macro_f1")
SHUFFLE_REPORT_FIELDS = (*SHUFFLE_REPORT_COUNT_FIELDS,
                         *SHUFFLE_REPORT_FLOAT_FIELDS,
                         "same_group_donor_ratio")


def shuffle_report_problems(record, label: str) -> list:
    """Required-field, type and finiteness validation for one saved
    shuffle report. Count fields must be true integers, float fields
    must be finite, and same_group_donor_ratio may be an explicit None
    only when matched == 0."""
    issues = []
    if not isinstance(record, dict):
        return [f"{label}: the shuffle report is not a mapping"]
    for field in SHUFFLE_REPORT_COUNT_FIELDS:
        if field not in record:
            issues.append(f"{label}.{field} is missing")
        elif isinstance(record[field], bool) or \
                not isinstance(record[field], int):
            issues.append(f"{label}.{field} is not an integer count: "
                          f"{record[field]!r}")
    for field in SHUFFLE_REPORT_FLOAT_FIELDS:
        if field not in record:
            issues.append(f"{label}.{field} is missing")
        elif isinstance(record[field], bool) or \
                not isinstance(record[field], (int, float)) or \
                not np.isfinite(float(record[field])):
            issues.append(f"{label}.{field} is not a finite number: "
                          f"{record[field]!r}")
    if "same_group_donor_ratio" not in record:
        issues.append(f"{label}.same_group_donor_ratio is missing")
    else:
        ratio = record["same_group_donor_ratio"]
        matched = record.get("matched")
        if ratio is None:
            if isinstance(matched, int) and \
                    not isinstance(matched, bool) and matched != 0:
                issues.append(
                    f"{label}.same_group_donor_ratio is None while "
                    f"matched == {matched}")
        elif isinstance(ratio, bool) or \
                not isinstance(ratio, (int, float)) or \
                not np.isfinite(float(ratio)):
            issues.append(
                f"{label}.same_group_donor_ratio is not a finite number "
                f"or an explicit None: {ratio!r}")
    return issues


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_saved_scales(fit_dir: Path) -> dict:
    admission = load_json(fit_dir / "admission.json")
    scales = admission.get("scales")
    if not isinstance(scales, dict) or not scales:
        raise ScaleMismatch(f"admission.json carries no scales")
    return scales


def verify_one_model(fit_dir: Path, seed: int, fold_id: str, arm: str,
                     variant: str, magnitude: float, x_eval: np.ndarray,
                     valid_eval: np.ndarray, p0_eval: np.ndarray,
                     labels_eval: list[int], expected_names: list[str],
                     problems: list[str], expected_budget: dict | None = None,
                     shared_scaler=None,
                     authoritative_identity: dict | None = None) -> dict:
    """Reload one saved path, verify its identity/budget/scaler, recompute
    its outer predictions, and check the saved rows, history selection
    and metrics."""
    path_dir = fit_dir / "models" / f"seed_{seed}" / \
        f"{fold_id}_{arm}_{variant}"
    payload = torch.load(path_dir / "residual_state.pt",
                         weights_only=False, map_location="cpu")
    # model identity: the payload must name exactly the path it lives in
    if payload.get("fold") != fold_id or payload.get("seed") != seed \
            or payload.get("arm") != arm or payload.get("variant") != \
            variant:
        problems.append(
            f"{path_dir.name}: payload identity "
            f"(fold={payload.get('fold')!r}, seed={payload.get('seed')!r}, "
            f"arm={payload.get('arm')!r}, "
            f"variant={payload.get('variant')!r}) does not match its "
            "directory")
    magnitude_value = float(payload.get("magnitude", float("nan")))
    if not np.isfinite(magnitude_value) or magnitude_value <= 0:
        problems.append(
            f"{path_dir.name}: payload magnitude {payload.get('magnitude')!r}"
            " is not a finite positive scale")
    elif abs(magnitude_value - float(magnitude)) > 0:
        problems.append(
            f"{path_dir.name}: payload magnitude {magnitude_value!r} != the "
            f"arm magnitude {magnitude!r}")
    if expected_budget is not None:
        saved_budget = payload.get("budget") or {}
        for key in ("epochs", "lr", "weight_decay", "aux_loss_weight",
                    "hidden"):
            if key in expected_budget and \
                    saved_budget.get(key) != expected_budget[key]:
                problems.append(
                    f"{path_dir.name}: budget {key} "
                    f"{saved_budget.get(key)!r} != the pinned "
                    f"{expected_budget[key]!r}")
        if saved_budget.get("device") != "cpu" or \
                saved_budget.get("full_batch") is not True:
            problems.append(
                f"{path_dir.name}: the saved budget is not cpu/full-batch")
    if shared_scaler is not None:
        saved_scaler = load_json(path_dir / "scaler.json")
        if saved_scaler != shared_scaler:
            problems.append(
                f"{path_dir.name}: the saved scaler does not match the "
                "fold's shared standardizer (fitted on inner_train only)")
    head = build_conditional_head(seed=int(payload["seed"]),
                                  input_dim=int(payload["input_dim"]),
                                  hidden=int(payload["hidden"]))
    head.load_state_dict(payload["state_dict"])
    head.eval()
    from core.material_conditional_residual import conditional_logits

    with torch.inference_mode():
        final = conditional_logits(
            head, torch.from_numpy(np.ascontiguousarray(x_eval)),
            torch.from_numpy(valid_eval.astype(np.float32)),
            torch.from_numpy(p0_eval), float(magnitude))
    logits = final.numpy().astype(np.float32)
    predictions = [int(v) for v in logits.argmax(axis=1)]
    saved_rows = _read_jsonl(path_dir / "outer_eval_predictions.jsonl")
    saved_names = [row["sample_name"] for row in saved_rows]
    if saved_names != expected_names:
        problems.append(
            f"{path_dir.name}: saved row order/set does not match the "
            "authoritative outer_eval order")
        return {"labels": labels_eval, "predictions": predictions,
                "metrics": count_metrics(labels_eval, predictions)}
    for index, row in enumerate(saved_rows):
        if authoritative_identity is not None:
            identity = authoritative_identity.get(row["sample_name"])
            if identity is None:
                problems.append(
                    f"{path_dir.name}: {row['sample_name']} is not in the "
                    "authoritative manifest")
            else:
                if int(row.get("class_id", -1)) != identity["class_id"]:
                    problems.append(
                        f"{path_dir.name}: {row['sample_name']} class_id "
                        f"{row.get('class_id')!r} != the authoritative "
                        f"{identity['class_id']}")
                if row.get("group") != identity["group"]:
                    problems.append(
                        f"{path_dir.name}: {row['sample_name']} group "
                        f"{row.get('group')!r} != the authoritative "
                        f"{identity['group']!r}")
        if row["prediction"] != int(logits[index].argmax()):
            problems.append(
                f"{path_dir.name}: {row['sample_name']} saved prediction "
                "!= recomputed argmax")
        if not np.allclose(np.asarray(row["logits"], dtype=np.float64),
                           logits[index].astype(np.float64),
                           rtol=0.0, atol=1e-6):
            problems.append(
                f"{path_dir.name}: {row['sample_name']} saved logits "
                "deviate from the recomputed forward")
        if row["class_id"] != labels_eval[index]:
            problems.append(
                f"{path_dir.name}: {row['sample_name']} class mismatch")
    history = load_json(path_dir / "history.json")
    best = select_epoch(history["epochs"])
    selection = history.get("selection", {})
    if int(selection.get("epoch", -1)) != int(best["epoch"]):
        problems.append(
            f"{path_dir.name}: selection epoch {selection.get('epoch')!r} "
            f"!= the history rule's epoch {best['epoch']}")
    if abs(float(selection.get("inner_val_macro_f1", -1))
           - float(best["inner_val_macro_f1"])) > 1e-9:
        problems.append(
            f"{path_dir.name}: selected inner_val_macro_f1 disagrees with "
            "the history")
    metrics = count_metrics(labels_eval, predictions)
    return {"labels": labels_eval, "predictions": predictions,
            "metrics": metrics, "saved_rows": saved_rows,
            "dir": path_dir}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="material_residual_scale_v1 independent verification.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--p0-fit-dir", required=True)
    parser.add_argument("--experiment-fit-dir", required=True,
                        help="the fit output directory to verify")
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; must not exist")
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--deadline-utc", default=None,
                        help="mutually exclusive with --budget-minutes")
    args = parser.parse_args(argv)
    if args.budget_minutes is not None and args.deadline_utc is not None:
        print("refused: --budget-minutes and --deadline-utc are mutually "
              "exclusive", file=sys.stderr)
        return EXIT_REFUSED
    if args.budget_minutes is not None and (
            not np.isfinite(args.budget_minutes)
            or args.budget_minutes <= 0):
        print("refused: --budget-minutes must be positive", file=sys.stderr)
        return EXIT_REFUSED
    if args.budget_minutes is None and args.deadline_utc is None:
        watch = None  # unbounded verification
    else:
        try:
            watch = DeadlineWatch(
                deadline_utc=args.deadline_utc
                if args.deadline_utc is not None else None,
                budget_minutes=args.budget_minutes)
        except ScaleRefused as error:
            print(f"refused: {error}", file=sys.stderr)
            return EXIT_REFUSED

    started = time.time()

    def log(message: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
              f"{message}", file=sys.stderr, flush=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    feature_dir = Path(args.feature_dir)
    if not feature_dir.is_absolute():
        feature_dir = _PROJECT_ROOT / feature_dir
    p0_fit_dir = Path(args.p0_fit_dir)
    if not p0_fit_dir.is_absolute():
        p0_fit_dir = _PROJECT_ROOT / p0_fit_dir
    fit_dir = Path(args.experiment_fit_dir)
    if not fit_dir.is_absolute():
        fit_dir = _PROJECT_ROOT / fit_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir

    problems: list[str] = []
    failure = None
    interrupted = False

    def check(at: str) -> None:
        if watch is not None:
            watch.check(at)

    try:
        try:
            check("start")
            if output_dir.exists():
                raise ScaleRefused(
                    f"output directory already exists: {output_dir}")
            if is_link_or_reparse(output_dir) or \
                    is_link_or_reparse(output_dir.parent):
                raise ScaleRefused(
                    "output directory or parent is a link/reparse point")
            cfg = load_scale_config(config_path)
            if not fit_dir.is_dir():
                raise ScaleRefused(f"fit directory missing: {fit_dir}")
            summary = load_json(fit_dir / "summary.json")
            if summary.get("experiment") != EXPERIMENT:
                raise ScaleMismatch(
                    f"fit experiment {summary.get('experiment')!r} != "
                    f"{EXPERIMENT!r}")
            if summary.get("training_status") != "COMPLETED":
                raise ScaleMismatch(
                    f"fit training_status "
                    f"{summary.get('training_status')!r} != COMPLETED")

            # identity plumbing (anchors, historical bindings, frozen
            # chain, cache contract) — the verifier protects its own
            # inputs exactly like the other two entries
            definition = verify_scale_cache_markers(feature_dir, cfg)
            import scripts.train_material_residual as frozen

            registry = SourceRegistry({})
            for name, anchor in cfg["anchors"].items():
                verify_anchor(_PROJECT_ROOT / anchor["path"],
                              anchor["sha256"], registry, f"anchor:{name}")
            check("after anchors")
            expansion = expand_historical_bindings(registry, cfg,
                                                   _PROJECT_ROOT)
            log(f"historical bindings: "
                f"{expansion['added']}")
            registry.bindings[(feature_dir / "feature_cache.npz")
                              .relative_to(_PROJECT_ROOT).as_posix()] = {
                "source": "scale_cache_binding",
                "expected_sha256": load_json(
                    feature_dir / "cache_binding.json")["files"][
                    "feature_cache.npz"]["sha256"]}
            registry.register(feature_dir / "cache_binding.json",
                              "fix_cache:cache_binding.json")
            registry.register(feature_dir / "feature_cache.npz",
                              "fix_cache:feature_cache.npz")
            registry.register(feature_dir / "feature_manifest.json",
                              "fix_cache:feature_manifest")
            registry.register(feature_dir / "feature_definition.json",
                              "fix_cache:feature_definition")
            registry.register(_PROJECT_ROOT / cfg["protocol_source"]["path"],
                              "protocol_file")
            # registration BEFORE the frozen chain consumes anything
            for name in ("cache_binding.json", "feature_cache.npz",
                         "feature_manifest.json",
                         "feature_definition.json"):
                path = (_PROJECT_ROOT / cfg["features_source"]["path"] /
                        name)
                if path.is_file():
                    registry.register(path, f"global_cache:{name}")
            if not (p0_fit_dir / "summary.json").is_file():
                raise ScaleRefused(
                    f"P0 fit summary missing: "
                    f"{p0_fit_dir / 'summary.json'}")
            for name in ("summary.json", "admission.json"):
                path = p0_fit_dir / name
                if path.is_file():
                    registry.register(path, f"p0_fit:{name}")
            for fold_dir in sorted((p0_fit_dir / "models").glob("*_P0")):
                for name in ("model_state.pt", "scaler.json",
                             "history.json",
                             "outer_eval_predictions.jsonl"):
                    path = fold_dir / name
                    if path.is_file():
                        registry.register(path,
                                          f"p0_model:{fold_dir.name}:"
                                          f"{name}")
            admission = frozen.run_admission(cfg, p0_fit_dir)
            protocol = admission["protocol"]
            verify_local_cache(feature_dir, protocol)

            manifest_samples = admission["manifest"]["samples"]
            authority = admission["authority"]
            # the authoritative identity map: saved rows are bound to
            # THIS, never to their self-reported group/class
            authoritative_identity = {
                sample["sample_name"]: {
                    "class_id": int(sample["class_id"]),
                    "group": sample["group"],
                    "split": sample["split"]}
                for sample in manifest_samples}
            global_cache = admission["cache"]
            fix_cache = dict(np.load(feature_dir / "feature_cache.npz"))
            p0_verified = {}
            for fold_id in sorted(authority_sample_folds(authority)):
                p0_verified[fold_id] = verify_p0_base_identity(
                    fold_id=fold_id, p0_fit_dir=p0_fit_dir,
                    admission=admission, global_cache=global_cache,
                    local_cache=fix_cache,
                    manifest_samples=manifest_samples,
                    authority=authority, log=log)

            # 1. scales, recomputed from the verified P0 chain
            recomputed_scales = {}
            for fold_id in sorted(p0_verified):
                parts = fold_parts(manifest_samples, authority, fold_id)
                verified = p0_verified[fold_id]
                rows = parts["inner_train"]
                logits = local_entry_p0_logits(verified["head"],
                                               verified["scaler"],
                                               global_cache, rows)
                scale, info = fold_scale(logits)
                recomputed_scales[fold_id] = info
            saved_scales = read_saved_scales(fit_dir)
            problems.extend(verify_scale_evidence(saved_scales,
                                                  recomputed_scales))

            # 2/3/4. per path: reload, recompute, reselect, recount
            check("after scales")
            recomputed = {}
            for fold_id in sorted(p0_verified):
                parts = fold_parts(manifest_samples, authority, fold_id)
                verified = p0_verified[fold_id]
                scale = float(recomputed_scales[fold_id]["scale"])
                scaler, _info = fit_local_scaler(fix_cache,
                                                 parts["inner_train"])
                expected_budget = {
                    "epochs": int(cfg["training"]["epochs"]),
                    "lr": float(cfg["training"]["lr"]),
                    "weight_decay": float(cfg["training"]["weight_decay"]),
                    "aux_loss_weight": float(cfg["training"].get(
                        "aux_loss_weight", 0.01)),
                    "hidden": int(cfg.get("head", {}).get("hidden", 8)),
                }
                shared_scaler_dict = scaler.to_dict()
                expected_names = [manifest_samples[row]["sample_name"]
                                  for row in parts["outer_eval"]]
                labels_eval = [int(v) for v in
                               fix_cache["class_id"][parts["outer_eval"]]]
                context = p0_context(local_entry_p0_logits(
                    verified["head"], verified["scaler"], global_cache,
                    parts["outer_eval"]))
                p0_eval = local_entry_p0_logits(
                    verified["head"], verified["scaler"], global_cache,
                    parts["outer_eval"])
                standardized = _standardize_local(fix_cache, scaler,
                                                  parts["outer_eval"])
                valid_eval = fix_cache["s_valid"][
                    parts["outer_eval"]].astype(bool)
                for arm in ARMS:
                    magnitude = magnitude_for_arm(arm, scale)
                    for variant in VARIANTS:
                        for seed in TRAIN_SEEDS:
                            c_std = standardized[:, :LOCAL_C_WIDTH]
                            s_std = standardized[:, LOCAL_C_WIDTH:]
                            x_eval = assemble_conditional_inputs(
                                c_std, s_std, context, valid_eval,
                                variant)
                            outcome = verify_one_model(
                                fit_dir, seed, fold_id, arm, variant,
                                magnitude, x_eval, valid_eval, p0_eval,
                                labels_eval, expected_names, problems,
                                expected_budget=expected_budget,
                                shared_scaler=shared_scaler_dict,
                                authoritative_identity=(
                                    authoritative_identity))
                            recomputed[(arm, variant, fold_id,
                                        seed)] = outcome
                # R0 rows: the verified chain IS the source
                r0_rows = _read_jsonl(fit_dir / "models" / "R0" /
                                      f"{fold_id}_R0" /
                                      "outer_eval_predictions.jsonl")
                if [row["sample_name"] for row in r0_rows] != \
                        expected_names:
                    problems.append(
                        f"{fold_id}: R0 row order does not match the "
                        "authoritative outer_eval")
                else:
                    for index, row in enumerate(r0_rows):
                        if row["prediction"] != \
                                verified["eval_predictions"][index]:
                            problems.append(
                                f"{fold_id}: R0 prediction mismatch at "
                                f"{row['sample_name']}")
                        if not np.allclose(
                                np.asarray(row["logits"], dtype=np.float64),
                                np.asarray(verified["eval_logits"][index],
                                           dtype=np.float64),
                                rtol=0.0, atol=1e-6):
                            problems.append(
                                f"{fold_id}: R0 logits mismatch at "
                                f"{row['sample_name']}")
                    recomputed[("R0", fold_id)] = {
                        "labels": labels_eval,
                        "predictions": verified["eval_predictions"]}

            # 5. metrics recounted from the saved rows
            check("after models")
            def recount(labels, predictions):
                return count_metrics(labels, predictions)

            for arm in ARMS:
                for variant in ("R0", *VARIANTS):
                    saved_entry = summary["arms"][arm][variant]
                    for seed in map(str, TRAIN_SEEDS):
                        per_seed = saved_entry["per_seed"][seed]
                        if variant == "R0":
                            labels = []
                            predictions = []
                            for fold_id in sorted(p0_verified):
                                rows = _read_jsonl(
                                    fit_dir / "models" / "R0" /
                                    f"{fold_id}_R0" /
                                    "outer_eval_predictions.jsonl")
                                labels.extend(int(r["class_id"])
                                              for r in rows)
                                predictions.extend(int(r["prediction"])
                                                   for r in rows)
                            pooled = recount(labels, predictions)
                        else:
                            pooled_labels = []
                            pooled_predictions = []
                            for fold_id in sorted(p0_verified):
                                outcome = recomputed[(arm, variant,
                                                      fold_id,
                                                      int(seed))]
                                pooled_labels.extend(outcome["labels"])
                                pooled_predictions.extend(
                                    outcome["predictions"])
                            pooled = recount(pooled_labels,
                                             pooled_predictions)
                        if _clean(pooled["macro_f1"]) != \
                                _clean(per_seed["pooled"]["macro_f1"]):
                            problems.append(
                                f"{arm}/{variant}/seed {seed}: pooled "
                                "macro_f1 disagrees with the saved "
                                "summary")
                        if pooled["confusion_matrix"] != \
                                per_seed["pooled"]["confusion_matrix"]:
                            problems.append(
                                f"{arm}/{variant}/seed {seed}: pooled "
                                "confusion disagrees with the summary "
                                "(a per-class swap that leaves the "
                                "macro-F1 unchanged still shows here)")
                        saved_recall = per_seed["pooled"][
                            "per_class_recall"]
                        for index, name in enumerate(CLASS_NAMES):
                            mine = pooled["per_class"][index]
                            theirs = saved_recall[name]
                            expected = 0.0 if theirs is None \
                                else float(theirs)
                            if abs(float(mine["recall"]) - expected) > 1e-9:
                                problems.append(
                                    f"{arm}/{variant}/seed {seed}: "
                                    f"{name} pooled recall disagrees "
                                    "with the summary")
                        for fold_entry in per_seed["folds"]:
                            if variant == "R0":
                                # R0 rows live in the per-fold R0 file;
                                # recount them directly
                                r0_rows = _read_jsonl(
                                    fit_dir / "models" / "R0" /
                                    f"{fold_entry['fold']}_R0" /
                                    "outer_eval_predictions.jsonl")
                                r0_labels = [int(r["class_id"])
                                             for r in r0_rows]
                                r0_preds = [int(r["prediction"])
                                            for r in r0_rows]
                                fold_metrics = count_metrics(
                                    r0_labels, r0_preds)
                            else:
                                outcome = recomputed[(arm, variant,
                                                      fold_entry["fold"],
                                                      int(seed))]
                                fold_metrics = outcome["metrics"]
                            if _clean(fold_metrics["macro_f1"]) != \
                                    _clean(fold_entry["macro_f1"]):
                                problems.append(
                                    f"{arm}/{variant}/seed {seed} "
                                    f"{fold_entry['fold']}: fold macro_f1 "
                                    "disagrees with the summary")
                            if _clean(fold_metrics["accuracy"]) != \
                                    _clean(fold_entry["accuracy"]):
                                problems.append(
                                    f"{arm}/{variant}/seed {seed} "
                                    f"{fold_entry['fold']}: fold accuracy "
                                    "disagrees with the summary")
                            if fold_metrics["confusion_matrix"] != \
                                    fold_entry["confusion_matrix"]:
                                problems.append(
                                    f"{arm}/{variant}/seed {seed} "
                                    f"{fold_entry['fold']}: fold confusion "
                                    "matrix disagrees with the summary")
                            saved_rows_count = fold_entry.get("rows")
                            if saved_rows_count is None or \
                                    int(saved_rows_count) != \
                                    fold_metrics["n"]:
                                problems.append(
                                    f"{arm}/{variant}/seed {seed} "
                                    f"{fold_entry['fold']}: fold rows "
                                    f"{saved_rows_count!r} != recounted "
                                    f"{fold_metrics['n']}")
                            for index, name in enumerate(CLASS_NAMES):
                                mine = fold_metrics["per_class"][index]
                                theirs = (fold_entry.get(
                                    "per_class_recall") or {})[name]
                                theirs = 0.0 if theirs is None \
                                    else float(theirs)
                                if abs(float(mine["recall"])
                                       - theirs) > 1e-9:
                                    problems.append(
                                        f"{arm}/{variant}/seed {seed} "
                                        f"{fold_entry['fold']}: {name} "
                                        "fold recall disagrees with the "
                                        "summary")

            # 6. paired comparisons recounted from the saved rows
            check("after metrics")
            def path_rows(arm: str, variant: str, seed: int):
                rows = []
                for fold_id in sorted(p0_verified):
                    if variant == "R0":
                        rows.extend(_read_jsonl(
                            fit_dir / "models" / "R0" / f"{fold_id}_R0" /
                            "outer_eval_predictions.jsonl"))
                    else:
                        rows.extend(_read_jsonl(
                            fit_dir / "models" / f"seed_{seed}" /
                            f"{fold_id}_{arm}_{variant}" /
                            "outer_eval_predictions.jsonl"))
                return rows


            for key, (left, right) in {
                    "scaled_RCS_minus_R0": (("scaled", "RCS"), "R0"),
                    "scaled_RCS_minus_RC": (("scaled", "RCS"),
                                            ("scaled", "RC")),
                    "scaled_RCS_minus_fixed_RCS": (("scaled", "RCS"),
                                                   ("fixed", "RCS")),
                    "scaled_RC_minus_fixed_RC": (("scaled", "RC"),
                                                 ("fixed", "RC")),
                    "fixed_RCS_minus_R0": (("fixed", "RCS"), "R0"),
                    "fixed_RCS_minus_RC": (("fixed", "RCS"),
                                           ("fixed", "RC")),
            }.items():
                saved = summary["paired"].get(key)
                if saved is None:
                    problems.append(f"paired comparison {key} is missing")
                    continue
                fixed_total = broke_total = 0
                per_class_total = {name: {"fixed": 0, "broke": 0}
                                   for name in CLASS_NAMES}
                for seed in TRAIN_SEEDS:
                    left_rows = path_rows(left[0], left[1], seed)
                    right_rows = (path_rows(right, "R0", seed)
                                  if right == "R0"
                                  else path_rows(right[0], right[1], seed))
                    left_names = [r["sample_name"] for r in left_rows]
                    right_names = [r["sample_name"] for r in right_rows]
                    if left_names != right_names:
                        problems.append(
                            f"{key} seed {seed}: paired rows are not "
                            "aligned by sample_name")
                        continue
                    seed_fixed = seed_broke = 0
                    seed_four = {"both_correct": 0, "both_wrong": 0}
                    for l_row, r_row in zip(left_rows, right_rows):
                        label = int(l_row["class_id"])
                        # the per-group bookkeeping uses the AUTHORITATIVE
                        # group, never the row's self-reported one
                        authoritative_group = authoritative_identity[
                            l_row["sample_name"]]["group"]
                        if label != int(r_row["class_id"]):
                            problems.append(
                                f"{key} seed {seed}: class contradiction "
                                f"at {l_row['sample_name']}")
                            continue
                        l_ok = int(l_row["prediction"]) == label
                        r_ok = int(r_row["prediction"]) == label
                        if l_ok and r_ok:
                            seed_four["both_correct"] += 1
                        elif not l_ok and not r_ok:
                            seed_four["both_wrong"] += 1
                        if l_ok and not r_ok:
                            seed_fixed += 1
                            per_class_total[CLASS_NAMES[label]][
                                "fixed"] += 1
                        elif r_ok and not l_ok:
                            seed_broke += 1
                            per_class_total[CLASS_NAMES[label]][
                                "broke"] += 1
                    saved_seed = saved["per_seed"][str(seed)]
                    if seed_fixed != int(saved_seed["fixed"]) or \
                            seed_broke != int(saved_seed["broke"]):
                        problems.append(
                            f"{key} seed {seed}: recounted fixed/broke "
                            f"{seed_fixed}/{seed_broke} != saved "
                            f"{saved_seed['fixed']}/{saved_seed['broke']}")
                    for counted, key_name in (
                            (seed_four["both_correct"], "both_correct"),
                            (seed_four["both_wrong"], "both_wrong")):
                        if int(counted) != int(saved_seed[key_name]):
                            problems.append(
                                f"{key} seed {seed}: {key_name} "
                                f"{counted} != saved "
                                f"{saved_seed[key_name]}")
                    if int(saved_seed["support"]) != len(left_rows):
                        problems.append(
                            f"{key} seed {seed}: saved support "
                            f"{saved_seed['support']} != the paired row "
                            f"count {len(left_rows)}")
                    fixed_total += seed_fixed
                    broke_total += seed_broke
                if fixed_total != int(saved["total"]["fixed"]) or \
                        broke_total != int(saved["total"]["broke"]):
                    problems.append(
                        f"{key}: summed fixed/broke {fixed_total}/"
                        f"{broke_total} != saved "
                        f"{saved['total']['fixed']}/"
                        f"{saved['total']['broke']}")
                recount_total = {"fixed": 0, "broke": 0,
                                 "both_correct": 0, "both_wrong": 0}
                for seed in TRAIN_SEEDS:
                    left_rows = path_rows(left[0], left[1], seed)
                    right_rows = (path_rows(right, "R0", seed)
                                  if right == "R0"
                                  else path_rows(right[0], right[1], seed))
                    for l_row, r_row in zip(left_rows, right_rows):
                        label = int(l_row["class_id"])
                        l_ok = int(l_row["prediction"]) == label
                        r_ok = int(r_row["prediction"]) == label
                        if l_ok and r_ok:
                            recount_total["both_correct"] += 1
                        elif not l_ok and not r_ok:
                            recount_total["both_wrong"] += 1
                        elif l_ok:
                            recount_total["fixed"] += 1
                        else:
                            recount_total["broke"] += 1
                for key_name in ("both_correct", "both_wrong"):
                    if int(saved["total"].get(key_name, -1)) != \
                            recount_total[key_name]:
                        problems.append(
                            f"{key}: total {key_name} "
                            f"{saved['total'].get(key_name)} != recounted "
                            f"{recount_total[key_name]}")
                saved_classes = summary["per_class_changes"][key][
                    "summed_over_seeds"]
                for name in CLASS_NAMES:
                    if int(per_class_total[name]["fixed"]) != \
                            int(saved_classes[name]["fixed"]) or \
                            int(per_class_total[name]["broke"]) != \
                            int(saved_classes[name]["broke"]):
                        problems.append(
                            f"{key}: per-class change for {name} "
                            "disagrees with the summary (a per-class swap "
                            "with an unchanged macro-F1 is still a "
                            "mismatch)")

            # 7. the shuffle bindings
            check("after paired comparisons")
            for arm in ARMS:
                for seed in TRAIN_SEEDS:
                    for fold_id in sorted(p0_verified):
                        parts = fold_parts(manifest_samples, authority,
                                           fold_id)
                        rows = parts["outer_eval"]
                        s_raw = fix_cache["s_features"][rows]
                        s_valid = fix_cache["s_valid"][rows].astype(bool)
                        interior_ok = (~np.isnan(s_raw[:, :8])).all(axis=1)
                        band_ok = (~np.isnan(s_raw[:, 8:])).all(axis=1)
                        train_rows = parts["inner_train"]
                        edges = {
                            "valid_ratio": np.quantile(
                                fix_cache["c_features"][train_rows][:, 0]
                                .astype(np.float64), [1 / 3, 2 / 3]),
                            "gray_mean": np.quantile(
                                fix_cache["c_features"][train_rows][:, 3]
                                .astype(np.float64), [1 / 3, 2 / 3]),
                        }
                        bins = build_donor_bins(
                            fix_cache["c_features"][rows][:, 0]
                            .astype(np.float64),
                            fix_cache["c_features"][rows][:, 3]
                            .astype(np.float64), edges=edges)
                        composite = (bins.astype(np.int64) * 4
                                     + interior_ok.astype(np.int64) * 2
                                     + band_ok.astype(np.int64))
                        split_labels = [manifest_samples[row]["split"]
                                        for row in rows]
                        names = [manifest_samples[row]["sample_name"]
                                 for row in rows]
                        for content_seed in SHUFFLE_CONTENT_SEEDS:
                            path = (fit_dir / "shuffle" / arm /
                                    f"seed_{seed}" / fold_id /
                                    f"seed_{content_seed}.json")
                            if not path.is_file():
                                problems.append(
                                    f"shuffle file missing: {path}")
                                continue
                            record = load_json(path)
                            receivers = [row for row in range(len(rows))
                                         if s_valid[row]]
                            assignment = shuffle_assignment(
                                receivers, composite, split_labels,
                                seed=content_seed)
                            saved_records = record.get("samples", [])
                            by_name = {r["sample_name"]: r
                                       for r in saved_records}
                            if len(by_name) != len(saved_records):
                                problems.append(
                                    f"{path}: duplicate shuffle records "
                                    "(a sample_name appears twice)")
                                continue
                            if sorted(by_name) != sorted(names):
                                problems.append(
                                    f"{path}: shuffle rows do not match "
                                    "the fold's outer_eval samples")
                                continue
                            matched = 0
                            for row in receivers:
                                donor = assignment.get(row)
                                saved_row = by_name[names[row]]
                                if donor is None:
                                    if saved_row["status"] != "unmatched":
                                        problems.append(
                                            f"{path}: row {names[row]} "
                                            "should be unmatched")
                                    continue
                                donor_name = names[donor]
                                if saved_row.get("donor_sample_name") != \
                                        donor_name:
                                    problems.append(
                                        f"{path}: donor for "
                                        f"{names[row]} != recomputed "
                                        f"{donor_name}")
                                matched += 1
                            if matched != int(record.get("matched", -1)):
                                problems.append(
                                    f"{path}: matched count "
                                    f"{record.get('matched')} != "
                                    f"recomputed {matched}")
                        # cross-arm / cross-train-seed mapping identity:
                        # the same (fold, content seed) must bind the SAME
                        # donor for every arm and every training seed
                        for other_arm in ARMS:
                            if other_arm == arm:
                                continue
                            other_path = (fit_dir / "shuffle" /
                                          other_arm / f"seed_{seed}" /
                                          fold_id /
                                          f"seed_{content_seed}.json")
                            if other_path.is_file():
                                other = load_json(other_path)
                                other_map = {
                                    r["sample_name"]:
                                        r.get("donor_sample_name")
                                    for r in other.get("samples", [])}
                                this_map = {
                                    r["sample_name"]:
                                        r.get("donor_sample_name")
                                    for r in saved_records}
                                if other_map != this_map:
                                    problems.append(
                                        f"{path}: the donor mapping "
                                        f"differs from arm {other_arm}")
                        for other_seed in TRAIN_SEEDS:
                            if other_seed == seed:
                                continue
                            other_path = (fit_dir / "shuffle" / arm /
                                          f"seed_{other_seed}" / fold_id /
                                          f"seed_{content_seed}.json")
                            if other_path.is_file():
                                other = load_json(other_path)
                                other_map = {
                                    r["sample_name"]:
                                        r.get("donor_sample_name")
                                    for r in other.get("samples", [])}
                                this_map = {
                                    r["sample_name"]:
                                        r.get("donor_sample_name")
                                    for r in saved_records}
                                if other_map != this_map:
                                    problems.append(
                                        f"{path}: the donor mapping "
                                        f"differs from training seed "
                                        f"{other_seed}")

            # 7a-2. the per-arm shuffle summaries and the final
            # summary["shuffle"] must agree with the per-seed files
            # field-for-field (receivers, matched, unmatched, coverage,
            # same_group_donor_ratio, macro_f1 and both vs_normal_rcs
            # counts); a missing required entry/field is a problem of
            # its own — never read as a legal None
            for arm in ARMS:
                for seed in TRAIN_SEEDS:
                    for fold_id in sorted(p0_verified):
                        summary_path = (fit_dir / "shuffle" / arm /
                                        f"seed_{seed}" / fold_id /
                                        "shuffle_summary.json")
                        if not summary_path.is_file():
                            problems.append(
                                f"shuffle summary missing: {summary_path}")
                            continue
                        shuffle_summary = load_json(summary_path)
                        seeds_section = shuffle_summary.get("seeds") \
                            if isinstance(shuffle_summary, dict) else None
                        if not isinstance(seeds_section, dict):
                            problems.append(
                                f"{summary_path}: the seeds section is "
                                "missing")
                            seeds_section = {}
                        for content_seed in SHUFFLE_CONTENT_SEEDS:
                            per_seed_path = (fit_dir / "shuffle" / arm /
                                             f"seed_{seed}" / fold_id /
                                             f"seed_{content_seed}.json")
                            if not per_seed_path.is_file():
                                continue
                            per_seed = load_json(per_seed_path)
                            problems.extend(shuffle_report_problems(
                                per_seed, str(per_seed_path)))
                            recorded = seeds_section.get(str(content_seed))
                            if not isinstance(recorded, dict):
                                problems.append(
                                    f"{summary_path}: seeds."
                                    f"{content_seed} is missing")
                            else:
                                problems.extend(shuffle_report_problems(
                                    recorded,
                                    f"{summary_path}: seeds."
                                    f"{content_seed}"))
                                for field in SHUFFLE_REPORT_FIELDS:
                                    if record_or_none(per_seed, field) != \
                                            record_or_none(recorded,
                                                           field):
                                        problems.append(
                                            f"{summary_path}: seeds."
                                            f"{content_seed}.{field} "
                                            "disagrees with the per-seed "
                                            "file")
                            shuffle_root = summary.get("shuffle")
                            if not isinstance(shuffle_root, dict):
                                problems.append(
                                    "summary[shuffle] is missing entirely")
                                continue
                            final_entry = (shuffle_root
                                           .get(f"seed_{seed}", {})
                                           .get(fold_id, {})
                                           .get("arms", {})
                                           .get(arm, {})
                                           .get(str(content_seed)))
                            if not isinstance(final_entry, dict):
                                problems.append(
                                    "summary[shuffle]: the content-seed "
                                    f"entry seed_{seed}/{fold_id}/{arm}/"
                                    f"{content_seed} is missing")
                                continue
                            problems.extend(shuffle_report_problems(
                                final_entry,
                                "summary[shuffle]: "
                                f"seed_{seed}/{fold_id}/{arm}/"
                                f"{content_seed}"))
                            for field in SHUFFLE_REPORT_FIELDS:
                                if record_or_none(per_seed, field) != \
                                        record_or_none(final_entry, field):
                                    problems.append(
                                        "summary[shuffle]: "
                                        f"seed_{seed}/{fold_id}/{arm}/"
                                        f"{content_seed}.{field} "
                                        "disagrees with the per-seed "
                                        "file")

            # 7b. the shuffle outputs rebuilt from reloaded models: the
            # shuffled S16 is reassembled from the recomputed donors and
            # pushed through the SAME frozen head + magnitude
            check("after shuffle bindings")
            for arm in ARMS:
                for seed in TRAIN_SEEDS:
                    for fold_id in sorted(p0_verified):
                        parts = fold_parts(manifest_samples, authority,
                                           fold_id)
                        rows = parts["outer_eval"]
                        verified = p0_verified[fold_id]
                        scale = float(recomputed_scales[fold_id]["scale"])
                        magnitude = magnitude_for_arm(arm, scale)
                        variant_dir = (fit_dir / "models" / f"seed_{seed}" /
                                       f"{fold_id}_{arm}_RCS")
                        payload = torch.load(
                            variant_dir / "residual_state.pt",
                            weights_only=False, map_location="cpu")
                        head = build_conditional_head(
                            seed=int(payload["seed"]),
                            input_dim=int(payload["input_dim"]),
                            hidden=int(payload["hidden"]))
                        head.load_state_dict(payload["state_dict"])
                        head.eval()
                        scaler, _info = fit_local_scaler(
                            fix_cache, parts["inner_train"])
                        standardized = _standardize_local(fix_cache, scaler,
                                                          rows)
                        c_std = standardized[:, :LOCAL_C_WIDTH]
                        s_std = standardized[:, LOCAL_C_WIDTH:]
                        s_valid = fix_cache["s_valid"][rows].astype(bool)
                        s_raw = fix_cache["s_features"][rows]
                        interior_ok = (~np.isnan(s_raw[:, :8])).all(axis=1)
                        band_ok = (~np.isnan(s_raw[:, 8:])).all(axis=1)
                        labels_eval = [int(v) for v in
                                       fix_cache["class_id"][rows]]
                        groups_eval = [manifest_samples[row]["group"]
                                       for row in rows]
                        train_rows = parts["inner_train"]
                        edges = {
                            "valid_ratio": np.quantile(
                                fix_cache["c_features"][train_rows][:, 0]
                                .astype(np.float64), [1 / 3, 2 / 3]),
                            "gray_mean": np.quantile(
                                fix_cache["c_features"][train_rows][:, 3]
                                .astype(np.float64), [1 / 3, 2 / 3]),
                        }
                        bins = build_donor_bins(
                            fix_cache["c_features"][rows][:, 0]
                            .astype(np.float64),
                            fix_cache["c_features"][rows][:, 3]
                            .astype(np.float64), edges=edges)
                        composite = (bins.astype(np.int64) * 4
                                     + interior_ok.astype(np.int64) * 2
                                     + band_ok.astype(np.int64))
                        split_labels = [manifest_samples[row]["split"]
                                        for row in rows]
                        names = [manifest_samples[row]["sample_name"]
                                 for row in rows]
                        context = p0_context(local_entry_p0_logits(
                            verified["head"], verified["scaler"],
                            global_cache, rows))
                        p0_eval = local_entry_p0_logits(
                            verified["head"], verified["scaler"],
                            global_cache, rows)

                        def forward(s_matrix):
                            x_eval = assemble_conditional_inputs(
                                c_std, s_matrix, context, s_valid, "RCS")
                            with torch.inference_mode():
                                final = conditional_logits(
                                    head, torch.from_numpy(
                                        np.ascontiguousarray(x_eval)),
                                    torch.from_numpy(
                                        s_valid.astype(np.float32)),
                                    torch.from_numpy(p0_eval),
                                    float(magnitude))
                            return final.numpy().astype(np.float32)

                        normal_logits = forward(s_std)
                        for content_seed in SHUFFLE_CONTENT_SEEDS:
                            path = (fit_dir / "shuffle" / arm /
                                    f"seed_{seed}" / fold_id /
                                    f"seed_{content_seed}.json")
                            if not path.is_file():
                                continue
                            record = load_json(path)
                            receivers = [row for row in range(len(rows))
                                         if s_valid[row]]
                            assignment = shuffle_assignment(
                                receivers, composite, split_labels,
                                seed=content_seed)
                            shuffled = s_std.copy()
                            for row, donor in assignment.items():
                                if donor is not None:
                                    shuffled[row] = s_std[donor]
                            shuffled_logits = forward(shuffled)
                            saved_records = record.get("samples", [])
                            by_name = {r["sample_name"]: r
                                       for r in saved_records}
                            matched = same_group = net_fixed = net_broke = 0
                            for row in range(len(rows)):
                                saved_row = by_name.get(names[row])
                                if saved_row is None:
                                    continue
                                donor = assignment.get(row)
                                if not s_valid[row]:
                                    expected_status = "invalid_s"
                                elif donor is None:
                                    expected_status = "unmatched"
                                else:
                                    expected_status = "matched"
                                if saved_row.get("status") != \
                                        expected_status:
                                    problems.append(
                                        f"{path}: row {names[row]} status "
                                        f"{saved_row.get('status')!r} != "
                                        f"recomputed {expected_status!r}")
                                    continue
                                if not np.allclose(
                                        np.asarray(
                                            saved_row["normal_logits"],
                                            dtype=np.float64),
                                        normal_logits[row].astype(
                                            np.float64), rtol=0.0,
                                        atol=1e-6):
                                    problems.append(
                                        f"{path}: {names[row]} normal "
                                        "logits deviate from the rebuilt "
                                        "forward")
                                if not np.allclose(
                                        np.asarray(
                                            saved_row["shuffled_logits"],
                                            dtype=np.float64),
                                        shuffled_logits[row].astype(
                                            np.float64), rtol=0.0,
                                        atol=1e-6):
                                    problems.append(
                                        f"{path}: {names[row]} shuffled "
                                        "logits deviate from the rebuilt "
                                        "forward")
                                if int(saved_row.get(
                                        "normal_prediction", -1)) != \
                                        int(normal_logits[row].argmax()):
                                    problems.append(
                                        f"{path}: {names[row]} normal "
                                        "prediction disagrees with the "
                                        "rebuilt forward")
                                if int(saved_row.get(
                                        "shuffled_prediction", -1)) != \
                                        int(shuffled_logits[row].argmax()):
                                    problems.append(
                                        f"{path}: {names[row]} shuffled "
                                        "prediction disagrees with the "
                                        "rebuilt forward")
                                if expected_status == "matched":
                                    matched += 1
                                    if groups_eval[donor] == \
                                            groups_eval[row]:
                                        same_group += 1
                                    label = labels_eval[row]
                                    s_ok = int(shuffled_logits[row]
                                               .argmax()) == label
                                    n_ok = int(normal_logits[row]
                                               .argmax()) == label
                                    if s_ok and not n_ok:
                                        net_fixed += 1
                                    elif n_ok and not s_ok:
                                        net_broke += 1
                            receivers_n = int(s_valid.sum())
                            if matched != int(record.get("matched", -1)):
                                problems.append(
                                    f"{path}: rebuilt matched {matched} != "
                                    f"saved {record.get('matched')}")
                            saved_receivers = record.get("receivers")
                            if saved_receivers is None or \
                                    int(saved_receivers) != receivers_n:
                                problems.append(
                                    f"{path}: receivers "
                                    f"{saved_receivers!r} != rebuilt "
                                    f"{receivers_n}")
                            saved_unmatched = record.get("unmatched")
                            if saved_unmatched is None or \
                                    int(saved_unmatched) != \
                                    receivers_n - matched:
                                problems.append(
                                    f"{path}: unmatched "
                                    f"{saved_unmatched!r} != rebuilt "
                                    f"{receivers_n - matched}")
                            # coverage MUST exist and be finite — a
                            # missing coverage is a problem of its own,
                            # never conflated with a legitimate None
                            saved_coverage = record.get("coverage")
                            if saved_coverage is None:
                                problems.append(
                                    f"{path}: coverage is missing (it is "
                                    "always required and finite)")
                            elif not isinstance(saved_coverage,
                                                (int, float)) or \
                                    not np.isfinite(saved_coverage):
                                problems.append(
                                    f"{path}: coverage "
                                    f"{saved_coverage!r} is not a finite "
                                    "number")
                            else:
                                rebuilt_coverage = _clean(
                                    matched / max(1, receivers_n))
                                if abs(float(saved_coverage) -
                                        float(rebuilt_coverage)) > 1e-9:
                                    problems.append(
                                        f"{path}: coverage "
                                        f"{saved_coverage} != rebuilt "
                                        f"{rebuilt_coverage}")
                            # same_group_donor_ratio is None ONLY when
                            # nothing matched
                            saved_ratio = record.get(
                                "same_group_donor_ratio")
                            rebuilt_ratio = _clean(same_group / matched) \
                                if matched else None
                            if matched == 0:
                                if saved_ratio is not None:
                                    problems.append(
                                        f"{path}: same_group_donor_ratio "
                                        f"{saved_ratio!r} must be None "
                                        "when matched == 0")
                            elif saved_ratio is None:
                                problems.append(
                                    f"{path}: same_group_donor_ratio is "
                                    "missing but matched > 0")
                            elif abs(float(saved_ratio) -
                                     float(rebuilt_ratio)) > 1e-9:
                                problems.append(
                                    f"{path}: same_group_donor_ratio "
                                    f"{saved_ratio} != rebuilt "
                                    f"{rebuilt_ratio}")
                            if net_fixed != int(record.get(
                                    "vs_normal_rcs_fixed", -1)):
                                problems.append(
                                    f"{path}: vs_normal_rcs_fixed "
                                    f"{record.get('vs_normal_rcs_fixed')} "
                                    f"!= rebuilt {net_fixed}")
                            if net_broke != int(record.get(
                                    "vs_normal_rcs_broke", -1)):
                                problems.append(
                                    f"{path}: vs_normal_rcs_broke "
                                    f"{record.get('vs_normal_rcs_broke')} "
                                    f"!= rebuilt {net_broke}")
                            saved_macro = record.get("macro_f1")
                            rebuilt_predictions = [
                                int(v) for v in
                                shuffled_logits.argmax(axis=1)]
                            rebuilt_macro = _clean(count_metrics(
                                labels_eval,
                                rebuilt_predictions)["macro_f1"])
                            if saved_macro is None or \
                                    abs(float(saved_macro)
                                        - float(rebuilt_macro)) > 1e-9:
                                problems.append(
                                    f"{path}: shuffle macro_f1 "
                                    f"{saved_macro} != rebuilt "
                                    f"{rebuilt_macro}")

            # 8. the summary's means, R0 stats, per-group counts and the
            # six gates are RECOMPUTED from the recounted evidence — the
            # saved scalars are never trusted on their own
            check("after shuffle rebuilds")

            def path_rows_of(arm, variant, seed):
                return path_rows(arm, variant, seed)

            recomputed_summary = {
                "class_names": list(CLASS_NAMES),
                "arms": {}, "paired": {}, "per_group_changes": {}}
            for arm in ARMS:
                recomputed_summary["arms"][arm] = {}
                for variant in ("R0", *VARIANTS):
                    per_seed = {}
                    seed_means = []
                    pooled_means = []
                    for seed in TRAIN_SEEDS:
                        rows_all = path_rows_of(arm, variant, seed)
                        labels = [int(r["class_id"]) for r in rows_all]
                        predictions = [int(r["prediction"])
                                       for r in rows_all]
                        pooled = count_metrics(labels, predictions)
                        folds_metrics = []
                        for fold_id in sorted(p0_verified):
                            fold_names = set(
                                manifest_samples[row]["sample_name"]
                                for row in fold_parts(
                                    manifest_samples, authority,
                                    fold_id)["outer_eval"])
                            sub_labels = [int(r["class_id"]) for r in
                                          rows_all
                                          if r["sample_name"] in
                                          fold_names]
                            sub_preds = [int(r["prediction"]) for r in
                                         rows_all
                                         if r["sample_name"] in
                                         fold_names]
                            fold_counts = count_metrics(
                                sub_labels, sub_preds)
                            folds_metrics.append(
                                {"fold": fold_id,
                                 "macro_f1": fold_counts["macro_f1"],
                                 "accuracy": fold_counts["accuracy"],
                                 "rows": fold_counts["n"]})
                        fold_mean = float(np.mean(
                            [f["macro_f1"] for f in folds_metrics]))
                        seed_means.append(fold_mean)
                        pooled_means.append(pooled["macro_f1"])
                        per_seed[str(seed)] = {
                            "folds": folds_metrics,
                            "macro_f1_mean": fold_mean,
                            "accuracy_mean": float(np.mean(
                                [f["accuracy"] for f in folds_metrics])),
                            "pooled": {**pooled, "per_class_recall": {
                                name: pooled["per_class"][i]["recall"]
                                for i, name in enumerate(CLASS_NAMES)}},
                        }
                    saved_entry = summary["arms"][arm][variant]
                    for seed in TRAIN_SEEDS:
                        if _clean(per_seed[str(seed)]["macro_f1_mean"]) != \
                                _clean(saved_entry["per_seed"][
                                    str(seed)]["macro_f1_mean"]):
                            problems.append(
                                f"{arm}/{variant}/seed {seed}: the fold "
                                "mean disagrees with the summary")
                        saved_seed_entry = \
                            saved_entry["per_seed"][str(seed)]
                        if "accuracy_mean" not in saved_seed_entry:
                            problems.append(
                                f"{arm}/{variant}/seed {seed}: per-seed "
                                "accuracy_mean is missing from the "
                                "summary")
                        elif _clean(per_seed[str(seed)]["accuracy_mean"]) \
                                != _clean(saved_seed_entry["accuracy_mean"]):
                            problems.append(
                                f"{arm}/{variant}/seed {seed}: "
                                "per-seed accuracy_mean "
                                f"{saved_seed_entry['accuracy_mean']} != "
                                "the mean of the recounted fold "
                                "accuracies "
                                f"{per_seed[str(seed)]['accuracy_mean']}")
                        recomputed_folds = {
                            f["fold"]: f["macro_f1"]
                            for f in per_seed[str(seed)]["folds"]}
                        for fold_entry in saved_entry["per_seed"][
                                str(seed)]["folds"]:
                            theirs = fold_entry.get("macro_f1")
                            mine = recomputed_folds.get(
                                fold_entry.get("fold"))
                            if mine is None or theirs is None or \
                                    _clean(mine) != _clean(theirs):
                                problems.append(
                                    f"{arm}/{variant}/seed {seed} "
                                    f"{fold_entry.get('fold')}: the saved "
                                    "fold metric disagrees with the "
                                    "recounted rows")
                    recomputed_summary["arms"][arm][variant] = {
                        "per_seed": per_seed,
                        "macro_f1_mean_of_seeds":
                            float(np.mean(seed_means)),
                        "pooled_macro_f1_mean_of_seeds":
                            float(np.mean(pooled_means)),
                    }
                    if _clean(float(np.mean(seed_means))) != \
                            _clean(saved_entry["macro_f1_mean_of_seeds"]):
                        problems.append(
                            f"{arm}/{variant}: the cross-seed mean "
                            "disagrees with the summary")
                    if _clean(float(np.mean(pooled_means))) != \
                            _clean(saved_entry[
                                "pooled_macro_f1_mean_of_seeds"]):
                        problems.append(
                            f"{arm}/{variant}: the pooled cross-seed mean "
                            "disagrees with the summary")
                    saved_range = saved_entry.get("macro_f1_seed_range")
                    if variant != "R0":
                        expected_range = [_clean(min(seed_means)),
                                          _clean(max(seed_means))]
                        if saved_range != expected_range:
                            problems.append(
                                f"{arm}/{variant}: macro_f1_seed_range "
                                f"{saved_range} != recomputed "
                                f"{expected_range}")
                    # the top-level accuracy_mean is checked for EVERY
                    # variant INCLUDING R0: the arithmetic mean of the
                    # recounted per-seed accuracy means. R0's folds are
                    # identical across seeds by construction, so its
                    # three seed means coincide exactly and their mean
                    # IS the common recounted fold mean
                    recounted_seed_acc = [
                        per_seed[str(seed)]["accuracy_mean"]
                        for seed in TRAIN_SEEDS]
                    recomputed_acc_mean = \
                        recounted_seed_acc[0] if variant == "R0" \
                        else float(np.mean(recounted_seed_acc))
                    if "accuracy_mean" not in saved_entry:
                        problems.append(
                            f"{arm}/{variant}: accuracy_mean is missing "
                            "from the summary")
                    elif _clean(recomputed_acc_mean) != \
                            _clean(saved_entry["accuracy_mean"]):
                        problems.append(
                            f"{arm}/{variant}: accuracy_mean "
                            f"recomputed {_clean(recomputed_acc_mean)} != "
                            "saved "
                            f"{_clean(saved_entry['accuracy_mean'])}")
                    for seed in map(str, TRAIN_SEEDS):
                        pooled_entry = saved_entry["per_seed"][seed][
                            "pooled"]
                        if _clean(per_seed[str(seed)]["pooled"][
                                "accuracy"]) != \
                                _clean(pooled_entry["accuracy"]):
                            problems.append(
                                f"{arm}/{variant}/seed {seed}: pooled "
                                "accuracy disagrees with the summary")
                        saved_count = saved_entry["per_seed"][seed].get(
                            "pooled_sample_count")
                        if saved_count is None or \
                                int(saved_count) != \
                                per_seed[str(seed)]["pooled"]["n"]:
                            problems.append(
                                f"{arm}/{variant}/seed {seed}: "
                                f"pooled_sample_count {saved_count!r} != "
                                f"recounted "
                                f"{per_seed[str(seed)]['pooled']['n']}")

            recomputed_summary["paired"] = {}
            recomputed_summary["per_group_changes"] = {}
            for key, (left, right) in {
                    "scaled_RCS_minus_R0": (("scaled", "RCS"), "R0"),
                    "scaled_RCS_minus_RC": (("scaled", "RCS"),
                                            ("scaled", "RC")),
                    "scaled_RCS_minus_fixed_RCS": (("scaled", "RCS"),
                                                   ("fixed", "RCS")),
                    "scaled_RC_minus_fixed_RC": (("scaled", "RC"),
                                                 ("fixed", "RC")),
                    "fixed_RCS_minus_R0": (("fixed", "RCS"), "R0"),
                    "fixed_RCS_minus_RC": (("fixed", "RCS"),
                                           ("fixed", "RC")),
            }.items():
                per_seed_counts = {}
                total = {"fixed": 0, "broke": 0, "both_correct": 0,
                         "both_wrong": 0}
                per_group_total: dict = {}
                for seed in TRAIN_SEEDS:
                    left_rows = path_rows(left[0], left[1], seed)
                    right_rows = (path_rows(right, "R0", seed)
                                  if right == "R0"
                                  else path_rows(right[0], right[1], seed))
                    seed_counts = {"fixed": 0, "broke": 0,
                                   "both_correct": 0, "both_wrong": 0,
                                   "support": len(left_rows)}
                    for l_row, r_row in zip(left_rows, right_rows):
                        label = int(l_row["class_id"])
                        # bind the row to the authoritative identity and
                        # count groups by the MANIFEST group; a row whose
                        # sample_name is not in the manifest is refused,
                        # never skipped
                        identity = authoritative_identity.get(
                            l_row["sample_name"])
                        if identity is None:
                            problems.append(
                                f"{key} seed {seed}: "
                                f"{l_row['sample_name']} is not in the "
                                "authoritative manifest")
                            continue
                        if int(l_row.get("class_id", -1)) != \
                                identity["class_id"] or \
                                int(r_row.get("class_id", -1)) != \
                                identity["class_id"]:
                            problems.append(
                                f"{key} seed {seed}: class contradiction "
                                f"at {l_row['sample_name']}")
                            continue
                        if l_row.get("group") != identity["group"]:
                            problems.append(
                                f"{key} seed {seed}: {l_row['sample_name']} "
                                "self-reported group disagrees with the "
                                "manifest")
                            continue
                        l_ok = int(l_row["prediction"]) == label
                        r_ok = int(r_row["prediction"]) == label
                        if l_ok and r_ok:
                            total["both_correct"] += 1
                            seed_counts["both_correct"] += 1
                        elif not l_ok and not r_ok:
                            total["both_wrong"] += 1
                            seed_counts["both_wrong"] += 1
                        elif l_ok:
                            total["fixed"] += 1
                            seed_counts["fixed"] += 1
                        else:
                            total["broke"] += 1
                            seed_counts["broke"] += 1
                        group = identity["group"]
                        slot = per_group_total.setdefault(
                            group, {"fixed": 0, "broke": 0})
                        if l_ok and not r_ok:
                            slot["fixed"] += 1
                        elif r_ok and not l_ok:
                            slot["broke"] += 1
                    per_seed_counts[str(seed)] = seed_counts
                recomputed_summary["paired"][key] = {
                    "per_seed": per_seed_counts, "total": {**total}}
                recomputed_summary["per_group_changes"][key] = {
                    "summed_over_seeds": per_group_total}
                saved_groups = summary["per_group_changes"].get(
                    key, {}).get("summed_over_seeds", {})
                all_groups = sorted(set(saved_groups)
                                    | set(per_group_total))
                for group in all_groups:
                    mine = per_group_total.get(
                        group, {"fixed": 0, "broke": 0})
                    theirs = saved_groups.get(
                        group, {"fixed": 0, "broke": 0})
                    if int(mine["fixed"]) != int(theirs["fixed"]) or \
                            int(mine["broke"]) != int(theirs["broke"]):
                        problems.append(
                            f"{key}: per-group net-correction counts for "
                            f"{group!r} disagree with the summary")

            if summary.get("gates") is None:
                problems.append("the saved summary carries no gates")
            else:
                recomputed_gates = evaluate_gates(recomputed_summary)
                saved_gates = summary["gates"]
                detailed_fields = {
                    "gate1_scaled_rcs_vs_r0_rc":
                        ("gain_vs_r0_pp", "gain_vs_rc_pp",
                         "scaled_rcs_macro_f1", "r0_macro_f1",
                         "scaled_rc_macro_f1"),
                    "gate2_scaled_vs_fixed_rcs":
                        ("scaled_macro_f1", "fixed_macro_f1",
                         "paired_net"),
                    "gate3_two_seeds_net_vs_both":
                        ("per_seed_net_vs_r0", "per_seed_net_vs_rc",
                         "seeds_positive_in_both"),
                    "gate4_fold_stability":
                        ("folds_without_drop", "worst_fold_drop_pp"),
                    "gate5_class_recall_floor":
                        ("worst_recall_drop_pp",
                         "per_class_mean_recall"),
                    "gate6_group_breadth":
                        ("positive_groups_vs_r0",
                         "positive_group_count"),
                }
                for gate_name in ("gate1_scaled_rcs_vs_r0_rc",
                                  "gate2_scaled_vs_fixed_rcs",
                                  "gate3_two_seeds_net_vs_both",
                                  "gate4_fold_stability",
                                  "gate5_class_recall_floor",
                                  "gate6_group_breadth"):
                    mine = recomputed_gates.get(gate_name, {})
                    theirs = saved_gates.get(gate_name, {})
                    if mine.get("pass") != theirs.get("pass"):
                        problems.append(
                            f"gate {gate_name}: recomputed "
                            f"pass={mine.get('pass')} != "
                            f"saved {theirs.get('pass')}")
                    for field in detailed_fields[gate_name]:
                        mine_value = mine.get(field)
                        theirs_value = theirs.get(field)
                        if isinstance(mine_value, float) and \
                                isinstance(theirs_value, float):
                            drifted = abs(mine_value - theirs_value) > 1e-9
                        else:
                            drifted = mine_value != theirs_value
                        if drifted:
                            problems.append(
                                f"gate {gate_name}: {field} recomputed "
                                f"{mine_value!r} != saved {theirs_value!r}")
                if recomputed_gates["recommendation"] != \
                        saved_gates.get("recommendation"):
                    problems.append(
                        "the final gate recommendation disagrees with the "
                        "recomputed one")

            check("before verification success")
        except DeadlineExceeded as exc:
            interrupted = True
            print(f"deadline exceeded during verification: {exc}",
                  file=sys.stderr)
        except (ScaleMismatch, LocalMismatch) as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
        except (ScaleRefused, IdentityError, LocalRefused,
                FileNotFoundError, ValueError, OSError, KeyError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # unexpected: still write a status file
        failure = f"UNEXPECTED {type(exc).__name__}: {exc}"

    output_dir.mkdir(parents=True)
    report = {
        "experiment": EXPERIMENT,
        "verified_fit_dir": str(fit_dir),
        "problem_count": len(problems),
        "problems": problems[:100],
        "failure": failure,
        "interrupted_by_deadline": interrupted,
        "note": ("all metrics recounted by plain counting loops; the "
                 "production summarizers were never invoked"),
        "elapsed_s": _clean(time.time() - started),
    }
    dump_json(output_dir / "verification_report.json", report)
    # the verifier publishes its own source-protection manifest like the
    # other two entries (kept even on refusal/mismatch)
    try:
        protection = registry.compare()
        registry.write_manifest(output_dir / "source_manifest.json",
                                protection)
    except Exception as exc:  # noqa: BLE001 - evidence best-effort
        print(f"protection manifest failed: {exc}", file=sys.stderr)
    if interrupted:
        log("verification interrupted by the deadline; the collected "
            "report is kept and the run is NOT verified")
        return EXIT_REFUSED
    if failure:
        log(f"refused: {failure}")
        return EXIT_REFUSED
    if problems:
        log(f"MISMATCH: {len(problems)} problem(s), first: "
            f"{problems[0]}")
        return EXIT_MISMATCH
    log("verification OK: scales, models, selections, metrics, paired "
        "counts and shuffle bindings all reproduce")
    return EXIT_OK




def record_or_none(record: dict, field: str):
    """A None-tolerant field getter: missing and present-None both read
    as None so a REQUIRED missing field is caught by the caller's value
    comparison, never skipped."""
    value = (record or {}).get(field)
    return value


def authority_sample_folds(authority: dict) -> list[str]:
    folds = set()
    for entry in authority.values():
        folds.update(entry["roles"])
    return sorted(folds)


if __name__ == "__main__":
    raise SystemExit(main())
