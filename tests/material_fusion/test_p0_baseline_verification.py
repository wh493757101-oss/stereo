"""Entry-level tests for the P0 baseline verification (real main() path).

Every case mutates ONE factor of a fully self-consistent synthetic fit and
asserts the concrete outcome: exit 0 (verified), exit 1 (REFUSED —
identity/structure/binding/non-finite input/scaler-source mismatch/missing
files/output conflict) or exit 2 (MISMATCH — legal, finite saved values
disagreeing with the recomputation). All fits here are tiny synthetic CPU
linear heads over synthetic caches; no real fit, weight or dataset is read.

The "training" inside the fixture is a 5-epoch synthetic linear probe used
ONLY to produce self-consistent artifacts for the verifier to consume.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    Standardizer,
    accuracy,
    assemble_design_matrix,
    authoritative_sample_index,
    macro_f1,
)
from core.material_probe import train_linear_probe

from .test_cache_admission import make_cache_files, make_protocol

FOLD_IDS = ("fold0", "fold1", "fold2")
EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_MISMATCH = 2


# ---------------------------------------------------------------------------
# Synthetic self-consistent fit fixture


def _fold_parts(files, authority, fold_id):
    manifest = files["manifest"]
    names_index = {s["sample_name"]: i for i, s in enumerate(manifest["samples"])}
    parts = {"inner_train": [], "inner_val": [], "outer_eval": []}
    for sample in manifest["samples"]:
        role = authority[sample["sample_name"]]["roles"][fold_id]
        parts[role].append(names_index[sample["sample_name"]])
    return parts


def _standardize(files, scaler, rows):
    g = files["arrays"]["g_features"][rows]
    c = files["arrays"]["c_features"][rows]
    s = files["arrays"]["s_features"][rows]
    valid = files["arrays"]["s_valid"][rows].astype(bool)
    column_valid = np.ones(
        (len(rows), g.shape[1] + c.shape[1] + s.shape[1]), dtype=bool)
    column_valid[:, g.shape[1] + c.shape[1]:] = valid[:, None]
    standardized = scaler.transform(
        np.concatenate([g, c, s], axis=1), column_valid)
    g_dim = g.shape[1]
    design = assemble_design_matrix(
        standardized[:, :g_dim],
        standardized[:, g_dim:g_dim + len(CONTROL_FEATURE_KEYS)],
        standardized[:, g_dim + len(CONTROL_FEATURE_KEYS):], "P0")
    return design


def build_fit(scratch_dir: Path, name: str = "fit",
              train_probe=None) -> dict:
    """A self-consistent synthetic user fit over the synthetic chain.

    Roles follow the production contract: the scaler is fitted on
    inner_train ONLY, training uses inner_train, selection uses inner_val,
    and the saved outer_eval predictions come from the selected model.
    ``train_probe`` (optional) wraps the tiny synthetic CPU trainer so a
    test can assert which design/labels it actually received.
    """
    probe_calls = []
    if train_probe is None:
        def train_probe(X_train, y_train, X_val, y_val, **kwargs):
            probe_calls.append({"X_train": X_train, "y_train": y_train,
                                "X_val": X_val, "y_val": y_val})
            return train_linear_probe(X_train, y_train, X_val, y_val,
                                      **kwargs)

    protocol = make_protocol()
    protocol_dir = scratch_dir / f"{name}_protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True), encoding="utf-8")
    feature_dir = scratch_dir / f"{name}_features"
    files = make_cache_files(feature_dir, protocol)
    authority = authoritative_sample_index(protocol)
    manifest = files["manifest"]

    fit_dir = scratch_dir / name
    (fit_dir / "models").mkdir(parents=True)
    oof_folds = []
    fold_parts = {}
    train_designs = {}
    for fold_id in FOLD_IDS:
        parts = _fold_parts(files, authority, fold_id)
        fold_parts[fold_id] = parts
        g_in = files["arrays"]["g_features"][parts["inner_train"]]
        c_in = files["arrays"]["c_features"][parts["inner_train"]]
        s_in = files["arrays"]["s_features"][parts["inner_train"]]
        valid_in = files["arrays"]["s_valid"][parts["inner_train"]].astype(bool)
        column_valid = np.ones(
            (len(parts["inner_train"]),
             g_in.shape[1] + c_in.shape[1] + s_in.shape[1]), dtype=bool)
        column_valid[:, g_in.shape[1] + c_in.shape[1]:] = valid_in[:, None]
        scaler = Standardizer.fit(
            np.concatenate([g_in, c_in, s_in], axis=1), column_valid)

        design_train = _standardize(files, scaler, parts["inner_train"])
        labels_train = files["arrays"]["class_id"][
            parts["inner_train"]].astype(np.int64)
        design_val = _standardize(files, scaler, parts["inner_val"])
        labels_val = files["arrays"]["class_id"][
            parts["inner_val"]].astype(np.int64)

        model, history, best_state, best = train_probe(
            design_train, labels_train, design_val, labels_val,
            seed=2026, lr=0.01, weight_decay=0.001, max_epochs=5)
        model.load_state_dict(best_state)
        model.eval()

        design_eval = _standardize(files, scaler, parts["outer_eval"])
        labels = files["arrays"]["class_id"][
            parts["outer_eval"]].astype(np.int64)
        with torch.no_grad():
            logits = model(torch.from_numpy(design_eval)).numpy()
        predictions = [int(v) for v in logits.argmax(axis=1)]
        train_designs[fold_id] = design_train

        fold_variant_dir = fit_dir / "models" / f"{fold_id}_P0"
        fold_variant_dir.mkdir()
        torch.save({
            "state_dict": best_state,
            "input_dim": int(design_train.shape[1]),
            "num_classes": 4,
            "variant": "P0",
            "fold": fold_id,
            "seed": 2026,
            "budget": {"seed": 2026, "optimizer": "adamw", "lr": 0.01,
                       "weight_decay": 0.001, "max_epochs": 5,
                       "device": "cpu"},
        }, fold_variant_dir / "model_state.pt")
        (fold_variant_dir / "scaler.json").write_text(
            json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
        (fold_variant_dir / "history.json").write_text(json.dumps({
            "selection": {"epoch": best["epoch"],
                          "inner_val_macro_f1": best["inner_val_macro_f1"],
                          "inner_val_accuracy": best["inner_val_accuracy"]},
            "epochs": history,
        }), encoding="utf-8")
        with (fold_variant_dir / "outer_eval_predictions.jsonl").open(
                "w", encoding="utf-8") as handle:
            for row_index, cache_row in enumerate(parts["outer_eval"]):
                sample = manifest["samples"][cache_row]
                handle.write(json.dumps({
                    "sample_name": sample["sample_name"],
                    "group": sample["group"],
                    "class_id": int(labels[row_index]),
                    "prediction": predictions[row_index],
                    "logits": [float(v) for v in logits[row_index]],
                }, sort_keys=True) + "\n")
        fold_report = {
            "fold": fold_id, "variant": "P0",
            "selected_epoch": best["epoch"],
            "inner_val_macro_f1": best["inner_val_macro_f1"],
            "inner_val_accuracy": best["inner_val_accuracy"],
            "outer_eval_macro_f1": macro_f1(labels, predictions),
            "outer_eval_accuracy": accuracy(labels, predictions),
            "support": len(rows := parts["outer_eval"]),
        }
        (fold_variant_dir / "fold_variant_report.json").write_text(
            json.dumps(fold_report, indent=2), encoding="utf-8")
        oof_folds.append(fold_report)

    from core.material_probe import build_cache_binding

    binding = build_cache_binding(feature_dir, protocol)
    mean_macro = float(np.mean([r["outer_eval_macro_f1"] for r in oof_folds]))
    mean_acc = float(np.mean([r["outer_eval_accuracy"] for r in oof_folds]))
    pooled_labels, pooled_predictions = [], []
    for fold_id in FOLD_IDS:
        for line in (fit_dir / "models" / f"{fold_id}_P0"
                     / "outer_eval_predictions.jsonl").read_text(
                encoding="utf-8").splitlines():
            record = json.loads(line)
            pooled_labels.append(record["class_id"])
            pooled_predictions.append(record["prediction"])
    pooled_macro = macro_f1(pooled_labels, pooled_predictions)
    summary = {
        "mode": "fit",
        "training_status": "COMPLETED",
        "fold_status": "OK",
        "admission": {
            "protocol_sha256": protocol["protocol_sha256"],
            "cache_binding": {
                "binding_version": binding["binding_version"],
                "files": binding["files"],
                "data_fingerprint": binding["data_fingerprint"],
                "class_names": binding["class_names"],
            },
            "class_names": list(protocol["data"]["class_names"]),
            "feature_definition_version": "existing_data_probe_v1",
            "g_dim": int(files["arrays"]["g_features"].shape[1]),
            "imgsz": 224,
            "input_dim": int(files["arrays"]["g_features"].shape[1]
                             + len(CONTROL_FEATURE_KEYS) + 8),
            "preprocessing_version": "v4_band_224_area_linear_v1",
            "sample_count": len(manifest["samples"]),
            "budget": {"seed": 2026, "optimizer": "adamw", "lr": 0.01,
                       "weight_decay": 0.001, "max_epochs": 5,
                       "device": "cpu"},
        },
        "folds": list(FOLD_IDS),
        "variants": ["P0"],
        "oof": {"P0": {"folds": oof_folds,
                       "macro_f1_mean": mean_macro,
                       "accuracy_mean": mean_acc}},
        "paired": {"P2_minus_P0": {
            "fixed": 0, "broke": 0, "both_correct": 0, "both_wrong": 0,
            "pooled_macro_f1": {"P0": pooled_macro, "P2": pooled_macro}}},
    }
    (fit_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "protocol": protocol,
        "protocol_dir": protocol_dir,
        "feature_dir": feature_dir,
        "fit_dir": fit_dir,
        "arrays": files["arrays"],
        "manifest": manifest,
        "fold_parts": fold_parts,
        "train_designs": train_designs,
        "probe_calls": probe_calls,
        "pooled_macro": pooled_macro,
        "mean_macro": mean_macro,
    }


def run_verification(chain, output: Path) -> int:
    from scripts.verify_p0_baseline import main

    return main([
        "--protocol", str(chain["protocol_dir"] / "protocol.json"),
        "--features", str(chain["feature_dir"]),
        "--fit-dir", str(chain["fit_dir"]),
        "--output-dir", str(output),
    ])


def pred_path(chain, fold_id):
    return (chain["fit_dir"] / "models" / f"{fold_id}_P0"
            / "outer_eval_predictions.jsonl")


def rewrite_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8")


def load_predictions(chain, fold_id):
    return [json.loads(line) for line in
            pred_path(chain, fold_id).read_text(
                encoding="utf-8").splitlines() if line.strip()]


def summary_path(chain):
    return chain["fit_dir"] / "summary.json"


def load_summary(chain):
    return json.loads(summary_path(chain).read_text(encoding="utf-8"))


def save_summary(chain, summary):
    summary_path(chain).write_text(json.dumps(summary, indent=2),
                                   encoding="utf-8")


def fold_json(chain, fold_id, name):
    return (chain["fit_dir"] / "models" / f"{fold_id}_P0" / name)


def load_fold_json(chain, fold_id, name):
    return json.loads(fold_json(chain, fold_id, name).read_text(
        encoding="utf-8"))


def save_fold_json(chain, fold_id, name, payload):
    fold_json(chain, fold_id, name).write_text(json.dumps(payload, indent=2),
                                              encoding="utf-8")


# ---------------------------------------------------------------------------
# D. status semantics on the healthy fixture


class TestHealthyFixture:
    def test_main_returns_zero_and_report_complete(self, scratch_dir):
        chain = build_fit(scratch_dir)
        output = scratch_dir / "verify_out"
        assert run_verification(chain, output) == EXIT_OK
        report = json.loads(
            (output / "p0_baseline_verification.json").read_text(
                encoding="utf-8"))
        assert report["status"] == "COMPLETED"
        assert len(report["folds"]) == 3
        for fold in report["folds"]:
            assert fold["rows"] == 3
            assert fold["scaler_check"]["status"] == "VERIFIED"
            assert fold["scaler_check"]["inner_train_rows"] == 3
            assert fold["scaler_check"]["s_valid_rows"] == 3
            assert "max_abs_diff" in fold["scaler_check"]
            assert fold["max_saved_recomputed_logit_diff"] <= 1e-4
            assert set(fold["files"]) == {
                "model_state.pt", "scaler.json", "history.json",
                "outer_eval_predictions.jsonl", "fold_variant_report.json"}
        assert "tolerances" in report
        assert report["tolerances"]["logit_atol"] == 1e-4
        assert report["tolerances"]["metric_atol"] == 1e-6
        assert report["tolerances"]["scaler_atol"] == 1e-6
        assert "summary_aggregates" in report
        assert report["summary_aggregates"][
            "recomputed_mean_macro_f1"] == pytest.approx(chain["mean_macro"])
        # the two Macro-F1 aggregations stay separate
        assert "pooled" in report["summary_aggregates"]
        assert report["summary_aggregates"]["pooled"][
            "macro_f1"] == pytest.approx(chain["pooled_macro"])

    def test_existing_output_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        output = scratch_dir / "verify_twice"
        assert run_verification(chain, output) == EXIT_OK
        assert run_verification(chain, output) == EXIT_REFUSED


# ---------------------------------------------------------------------------
# A. numeric strictness (non-finite inputs must be REFUSED, never pass)


class TestNonFiniteInputs:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"),
                                     float("-inf")])
    def test_non_finite_saved_logit_refused(self, scratch_dir, bad):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold0")
        records[1]["logits"][2] = bad
        rewrite_jsonl(pred_path(chain, "fold0"), records)
        output = scratch_dir / "v_bad_logit"
        assert run_verification(chain, output) == EXIT_REFUSED
        assert not output.exists()

    def test_non_finite_fold_metric_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold0", "fold_variant_report.json")
        report["outer_eval_macro_f1"] = float("nan")
        save_fold_json(chain, "fold0", "fold_variant_report.json", report)
        output = scratch_dir / "v_bad_fold_metric"
        assert run_verification(chain, output) == EXIT_REFUSED
        assert not output.exists()

    def test_non_finite_summary_metric_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["macro_f1_mean"] = float("inf")
        save_summary(chain, summary)
        output = scratch_dir / "v_bad_summary_metric"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_non_finite_weight_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        checkpoint = chain["fit_dir"] / "models" / "fold0_P0" / "model_state.pt"
        payload = torch.load(checkpoint, weights_only=False,
                             map_location="cpu")
        payload["state_dict"]["weight"] = payload["state_dict"][
            "weight"].clone()
        payload["state_dict"]["weight"][0, 0] = float("nan")
        torch.save(payload, checkpoint)
        output = scratch_dir / "v_bad_weight"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_weight_shape_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        checkpoint = chain["fit_dir"] / "models" / "fold0_P0" / "model_state.pt"
        payload = torch.load(checkpoint, weights_only=False,
                             map_location="cpu")
        payload["state_dict"]["weight"] = torch.zeros(4, 17)
        torch.save(payload, checkpoint)
        output = scratch_dir / "v_bad_weight_shape"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_bias_shape_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        checkpoint = chain["fit_dir"] / "models" / "fold0_P0" / "model_state.pt"
        payload = torch.load(checkpoint, weights_only=False,
                             map_location="cpu")
        payload["state_dict"]["bias"] = torch.zeros(7)
        torch.save(payload, checkpoint)
        output = scratch_dir / "v_bad_bias_shape"
        assert run_verification(chain, output) == EXIT_REFUSED


# ---------------------------------------------------------------------------
# A. legal finite disagreement -> MISMATCH (exit 2)


class TestFiniteMismatch:
    def test_small_logit_difference_is_mismatch(self, scratch_dir):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold1")
        records[0]["logits"][1] += 0.01  # finite, > LOGIT_ATOL, < huge
        rewrite_jsonl(pred_path(chain, "fold1"), records)
        output = scratch_dir / "v_diff_logit"
        assert run_verification(chain, output) == EXIT_MISMATCH

    def test_small_metric_difference_is_mismatch(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold2", "fold_variant_report.json")
        report["outer_eval_accuracy"] += 0.01
        save_fold_json(chain, "fold2", "fold_variant_report.json", report)
        output = scratch_dir / "v_diff_metric"
        assert run_verification(chain, output) == EXIT_MISMATCH

    def test_flipped_prediction_is_mismatch(self, scratch_dir):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold0")
        records[0]["prediction"] = (records[0]["prediction"] + 1) % 4
        rewrite_jsonl(pred_path(chain, "fold0"), records)
        output = scratch_dir / "v_flipped"
        assert run_verification(chain, output) == EXIT_MISMATCH


# ---------------------------------------------------------------------------
# B. identity and summary consistency


class TestIdentityAndSummary:
    def test_wrong_group_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold0")
        records[0]["group"] = "no/such/group"
        rewrite_jsonl(pred_path(chain, "fold0"), records)
        output = scratch_dir / "v_bad_group"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_class_id_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold1")
        records[2]["class_id"] = (records[2]["class_id"] + 1) % 4
        rewrite_jsonl(pred_path(chain, "fold1"), records)
        output = scratch_dir / "v_bad_class"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_sample_name_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold0")
        records[0]["sample_name"] = "not_in_protocol"
        rewrite_jsonl(pred_path(chain, "fold0"), records)
        output = scratch_dir / "v_bad_name"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_row_count_mismatch_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold2")
        records = records[:-1]  # one missing outer_eval row
        rewrite_jsonl(pred_path(chain, "fold2"), records)
        output = scratch_dir / "v_row_count"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_fold_report_identity_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold0", "fold_variant_report.json")
        report["fold"] = "fold9"
        save_fold_json(chain, "fold0", "fold_variant_report.json", report)
        output = scratch_dir / "v_bad_report_fold"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_fold_report_variant_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold0", "fold_variant_report.json")
        report["variant"] = "P1"
        save_fold_json(chain, "fold0", "fold_variant_report.json", report)
        output = scratch_dir / "v_bad_report_variant"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_fold_report_support_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold1", "fold_variant_report.json")
        report["support"] = 99
        save_fold_json(chain, "fold1", "fold_variant_report.json", report)
        output = scratch_dir / "v_bad_support"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_bool_support_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold1", "fold_variant_report.json")
        report["support"] = True  # bool is not an int here
        save_fold_json(chain, "fold1", "fold_variant_report.json", report)
        output = scratch_dir / "v_bool_support"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_duplicate_p0_fold_record_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"].append(
            dict(summary["oof"]["P0"]["folds"][0]))
        save_summary(chain, summary)
        output = scratch_dir / "v_dup_fold"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_missing_p0_fold_record_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"] = summary["oof"]["P0"]["folds"][:2]
        save_summary(chain, summary)
        output = scratch_dir / "v_missing_fold"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_only_three_fold_mean_changed_mismatch(self, scratch_dir):
        """A legal finite saved mean disagreeing with the recomputed
        three-fold mean is a MISMATCH (finite saved metric vs recompute)."""
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["macro_f1_mean"] = 0.0
        save_summary(chain, summary)
        output = scratch_dir / "v_mean_zero"
        assert run_verification(chain, output) == EXIT_MISMATCH

    def test_only_saved_p0_pooled_changed_mismatch(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["paired"]["P2_minus_P0"]["pooled_macro_f1"]["P0"] = (
            chain["pooled_macro"] + 0.05)
        save_summary(chain, summary)
        output = scratch_dir / "v_pooled"
        assert run_verification(chain, output) == EXIT_MISMATCH

    def test_selection_epoch_contradiction_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        history = load_fold_json(chain, "fold0", "history.json")
        history["selection"]["epoch"] = 999
        save_fold_json(chain, "fold0", "history.json", history)
        output = scratch_dir / "v_sel_epoch"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_selection_metric_contradiction_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        history = load_fold_json(chain, "fold0", "history.json")
        history["selection"]["inner_val_macro_f1"] = (
            history["selection"]["inner_val_macro_f1"] + 0.5)
        save_fold_json(chain, "fold0", "history.json", history)
        output = scratch_dir / "v_sel_metric"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_best_epoch_recomputed_from_full_history_mismatch(
            self, scratch_dir):
        """The selection record must be the argmax of the FULL history by
        the frozen rule; a selection that is not the history's best is a
        contradiction (REFUSED)."""
        chain = build_fit(scratch_dir)
        history = load_fold_json(chain, "fold1", "history.json")
        best = max(history["epochs"],
                   key=lambda e: (e["inner_val_macro_f1"],
                                  e["inner_val_accuracy"], -e["epoch"]))
        other = next(e for e in history["epochs"]
                     if e["epoch"] != best["epoch"])
        history["selection"] = {
            "epoch": other["epoch"],
            "inner_val_macro_f1": other["inner_val_macro_f1"],
            "inner_val_accuracy": other["inner_val_accuracy"],
        }
        save_fold_json(chain, "fold1", "history.json", history)
        report = load_fold_json(chain, "fold1", "fold_variant_report.json")
        report["selected_epoch"] = other["epoch"]
        save_fold_json(chain, "fold1", "fold_variant_report.json", report)
        summary = load_summary(chain)
        for record in summary["oof"]["P0"]["folds"]:
            if record["fold"] == "fold1":
                record["selected_epoch"] = other["epoch"]
        save_summary(chain, summary)
        output = scratch_dir / "v_not_best"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_admission_class_order_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["admission"]["class_names"] = list(
            reversed(summary["admission"]["class_names"]))
        save_summary(chain, summary)
        output = scratch_dir / "v_adm_classes"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_admission_preprocessing_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["admission"]["preprocessing_version"] = "other_v9"
        save_summary(chain, summary)
        output = scratch_dir / "v_adm_preproc"
        assert run_verification(chain, output) == EXIT_REFUSED


# ---------------------------------------------------------------------------
# C. scaler full-column verification


class TestScalerFullColumn:
    def test_only_c_mean_changed_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        scaler = load_fold_json(chain, "fold0", "scaler.json")
        c_start = chain["arrays"]["g_features"].shape[1]
        scaler["mean"][c_start] += 100.0  # a C column P0 never reads
        save_fold_json(chain, "fold0", "scaler.json", scaler)
        output = scratch_dir / "v_c_mean"
        assert run_verification(chain, output) == EXIT_REFUSED
        assert not output.exists()

    def test_only_s_std_changed_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        scaler = load_fold_json(chain, "fold1", "scaler.json")
        s_start = (chain["arrays"]["g_features"].shape[1]
                   + len(CONTROL_FEATURE_KEYS))
        scaler["std"][s_start + 3] += 0.5
        save_fold_json(chain, "fold1", "scaler.json", scaler)
        output = scratch_dir / "v_s_std"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_only_usable_changed_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        scaler = load_fold_json(chain, "fold2", "scaler.json")
        scaler["usable"][10] = not scaler["usable"][10]
        save_fold_json(chain, "fold2", "scaler.json", scaler)
        output = scratch_dir / "v_usable"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_non_finite_scaler_value_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        scaler = load_fold_json(chain, "fold0", "scaler.json")
        scaler["mean"][5] = float("nan")
        save_fold_json(chain, "fold0", "scaler.json", scaler)
        output = scratch_dir / "v_scaler_nan"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_scaler_length_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        scaler = load_fold_json(chain, "fold0", "scaler.json")
        scaler["mean"] = scaler["mean"][:16]  # shorter than the 18 columns
        save_fold_json(chain, "fold0", "scaler.json", scaler)
        output = scratch_dir / "v_scaler_len"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_scaler_usable_not_bool_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        scaler = load_fold_json(chain, "fold0", "scaler.json")
        scaler["usable"] = [int(v) for v in scaler["usable"]]
        save_fold_json(chain, "fold0", "scaler.json", scaler)
        output = scratch_dir / "v_scaler_usable_type"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_fit_probe_never_called_and_scaler_untouched(
            self, scratch_dir, monkeypatch):
        import core.material_probe as probe_module
        import scripts.verify_p0_baseline as verifier

        chain = build_fit(scratch_dir)
        scaler_paths = [chain["fit_dir"] / "models" / f"{f}_P0" / "scaler.json"
                        for f in FOLD_IDS]
        before = {p: p.read_bytes() for p in scaler_paths}

        def _bomb(*args, **kwargs):
            raise AssertionError("Standardizer.fit must not be called")

        monkeypatch.setattr(probe_module.Standardizer, "fit", _bomb)
        output = scratch_dir / "v_probe"
        assert run_verification(chain, output) == EXIT_OK
        for path, payload in before.items():
            assert path.read_bytes() == payload  # saved scaler untouched
        assert not (output / "models").exists()  # nothing trained/saved

    def test_no_encoder_and_no_training(self, scratch_dir, monkeypatch):
        import scripts.verify_p0_baseline as verifier

        chain = build_fit(scratch_dir)

        def _bomb(*args, **kwargs):
            raise AssertionError("no encoder may be constructed")

        monkeypatch.setattr(verifier, "build_probe_encoder", _bomb,
                            raising=False)
        output = scratch_dir / "v_no_encoder"
        assert run_verification(chain, output) == EXIT_OK


# ---------------------------------------------------------------------------
# Helpers for the remaining-gap single-factor cases


def in_range_variant(value: float) -> float:
    """A legal [0,1] value that differs from ``value`` by more than the
    metric tolerance (metrics may legitimately be exactly 1)."""
    if value <= 0.985:
        return value + 0.01
    return value - 0.01


class TestSummaryFoldSetAndAdmission:
    def test_zero_support_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"][0]["support"] = 0
        save_summary(chain, summary)
        output = scratch_dir / "v_support0"
        assert run_verification(chain, output) == EXIT_REFUSED
        assert not output.exists()

    def test_bool_support_in_summary_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"][1]["support"] = True
        save_summary(chain, summary)
        output = scratch_dir / "v_summary_bool"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_variant_in_summary_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"][2]["variant"] = "P2"
        save_summary(chain, summary)
        output = scratch_dir / "v_variant_p2"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_extra_fold9_record_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        stranger = dict(summary["oof"]["P0"]["folds"][0])
        stranger["fold"] = "fold9"
        summary["oof"]["P0"]["folds"].append(stranger)
        save_summary(chain, summary)
        output = scratch_dir / "v_fold9"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_non_record_entry_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"][0] = "not-a-record"
        save_summary(chain, summary)
        output = scratch_dir / "v_non_record"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_wrong_input_dim_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["admission"]["input_dim"] = 999
        save_summary(chain, summary)
        output = scratch_dir / "v_input_dim"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_input_dim_not_cached_width_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["admission"]["input_dim"] = 17  # != g_dim(4) + 6 + 8
        save_summary(chain, summary)
        output = scratch_dir / "v_input_dim_width"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_input_dim_none_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["admission"]["input_dim"] = None
        save_summary(chain, summary)
        output = scratch_dir / "v_input_dim_none"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_imgsz_none_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["admission"]["imgsz"] = None
        save_summary(chain, summary)
        output = scratch_dir / "v_imgsz_none"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_budget_max_epochs_removed_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        del summary["admission"]["budget"]["max_epochs"]
        save_summary(chain, summary)
        output = scratch_dir / "v_budget_missing"
        assert run_verification(chain, output) == EXIT_REFUSED


class TestFullEpochHistory:
    def test_truncated_to_selection_only_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        history = load_fold_json(chain, "fold0", "history.json")
        history["epochs"] = [dict(history["selection"],
                                  epoch=history["selection"]["epoch"])]
        save_fold_json(chain, "fold0", "history.json", history)
        output = scratch_dir / "v_truncated"
        assert run_verification(chain, output) == EXIT_REFUSED
        assert not output.exists()

    def test_one_non_selected_epoch_missing_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        history = load_fold_json(chain, "fold1", "history.json")
        history["epochs"] = [e for e in history["epochs"]
                             if e["epoch"] != 3]
        save_fold_json(chain, "fold1", "history.json", history)
        output = scratch_dir / "v_missing_epoch"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_duplicate_epoch_replacing_another_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        history = load_fold_json(chain, "fold0", "history.json")
        dup = dict(history["epochs"][1])
        history["epochs"][2] = dup  # equal count, but epoch 3 lost
        save_fold_json(chain, "fold0", "history.json", history)
        output = scratch_dir / "v_dup_epoch"
        assert run_verification(chain, output) == EXIT_REFUSED

    @pytest.mark.parametrize("bad_epoch", [0, 6, True, 2.0, "3"])
    def test_illegal_epoch_values_refused(self, scratch_dir, bad_epoch):
        chain = build_fit(scratch_dir)
        history = load_fold_json(chain, "fold0", "history.json")
        history["epochs"][1] = dict(history["epochs"][1],
                                    epoch=bad_epoch)
        save_fold_json(chain, "fold0", "history.json", history)
        output = scratch_dir / f"v_bad_epoch_{id(bad_epoch)%97}"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_checkpoint_budget_contradiction_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        checkpoint = chain["fit_dir"] / "models" / "fold0_P0" / "model_state.pt"
        payload = torch.load(checkpoint, weights_only=False,
                             map_location="cpu")
        payload["budget"]["max_epochs"] = 2  # summary says 5
        torch.save(payload, checkpoint)
        output = scratch_dir / "v_budget_conflict"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_full_legal_history_passes(self, scratch_dir):
        chain = build_fit(scratch_dir)
        output = scratch_dir / "v_history_ok"
        assert run_verification(chain, output) == EXIT_OK
        report = json.loads(
            (output / "p0_baseline_verification.json").read_text(
                encoding="utf-8"))
        for fold in report["folds"]:
            assert fold["history_check"]["status"] == "VERIFIED"
            assert fold["history_check"]["epochs"] == 5


class TestThreeWaySelectionMetrics:
    def test_fold_report_inner_f1_contradiction_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold0", "fold_variant_report.json")
        report["inner_val_macro_f1"] = in_range_variant(
            report["inner_val_macro_f1"])
        save_fold_json(chain, "fold0", "fold_variant_report.json", report)
        output = scratch_dir / "v_report_inner"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_summary_inner_accuracy_contradiction_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        for record in summary["oof"]["P0"]["folds"]:
            if record["fold"] == "fold1":
                record["inner_val_accuracy"] = in_range_variant(
                    record["inner_val_accuracy"])
        save_summary(chain, summary)
        output = scratch_dir / "v_summary_inner"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_nan_inner_f1_in_fold_report_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold0", "fold_variant_report.json")
        report["inner_val_macro_f1"] = float("nan")
        save_fold_json(chain, "fold0", "fold_variant_report.json", report)
        output = scratch_dir / "v_report_nan"
        assert run_verification(chain, output) == EXIT_REFUSED
        assert not output.exists()

    def test_nan_inner_f1_in_summary_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"][0]["inner_val_macro_f1"] = float("nan")
        save_summary(chain, summary)
        output = scratch_dir / "v_summary_nan"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_bool_inner_metric_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold0", "fold_variant_report.json")
        report["inner_val_accuracy"] = True
        save_fold_json(chain, "fold0", "fold_variant_report.json", report)
        output = scratch_dir / "v_bool_inner"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_out_of_range_inner_metric_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"][0]["inner_val_macro_f1"] = 1.01
        save_summary(chain, summary)
        output = scratch_dir / "v_range_inner"
        assert run_verification(chain, output) == EXIT_REFUSED

    def test_out_of_range_outer_metric_refused(self, scratch_dir):
        chain = build_fit(scratch_dir)
        summary = load_summary(chain)
        summary["oof"]["P0"]["folds"][0]["outer_eval_accuracy"] = -0.1
        save_summary(chain, summary)
        output = scratch_dir / "v_range_outer"
        assert run_verification(chain, output) == EXIT_REFUSED


class TestInMemoryPooled:
    def test_pooled_not_rereading_prediction_files(self, scratch_dir,
                                                   monkeypatch):
        import pathlib

        chain = build_fit(scratch_dir)
        calls = {"count": 0}
        original = pathlib.Path.read_text

        def counting(self, *args, **kwargs):
            if self.name == "outer_eval_predictions.jsonl":
                calls["count"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", counting)
        output = scratch_dir / "v_pool_bomb"
        assert run_verification(chain, output) == EXIT_OK
        # exactly one text read per fold inside verify_fold; main() must
        # not re-read the files to build the pooled metrics
        assert calls["count"] == 3

    def test_pooled_metrics_from_verified_results(self, scratch_dir):
        chain = build_fit(scratch_dir)
        output = scratch_dir / "v_pool_source"
        assert run_verification(chain, output) == EXIT_OK
        report = json.loads(
            (output / "p0_baseline_verification.json").read_text(
                encoding="utf-8"))
        pooled = report["summary_aggregates"]["pooled"]
        total = sum(f["rows"] for f in report["folds"])
        assert pooled["sample_count"] == total
        assert sum(sum(row) for row in
                   pooled["confusion_matrix"]) == total
        assert pooled["macro_f1"] == pytest.approx(chain["pooled_macro"])
        # per-fold coverage recorded
        assert [f["rows"] for f in report["folds"]] == [3, 3, 3]


class TestFixtureRoles:
    def test_trainer_receives_inner_train(self, scratch_dir):
        chain = build_fit(scratch_dir)
        # the probe recorded the actual tensors the trainer received
        assert len(chain["probe_calls"]) == 3
        for fold_id in FOLD_IDS:
            call = chain["probe_calls"][FOLD_IDS.index(fold_id)]
            design = chain["train_designs"][fold_id]
            assert call["X_train"].shape == design.shape
            assert np.array_equal(call["X_train"], design)
            assert call["X_val"].shape[0] == 3  # inner_val, not outer_eval


class TestFiniteMismatchStillHolds:
    """The fixture-role fix may push metrics to exactly 1; the legal
    finite-disagreement cases must stay inside [0,1]."""

    def test_small_metric_difference_is_mismatch(self, scratch_dir):
        chain = build_fit(scratch_dir)
        report = load_fold_json(chain, "fold2", "fold_variant_report.json")
        report["outer_eval_accuracy"] = in_range_variant(
            report["outer_eval_accuracy"])
        save_fold_json(chain, "fold2", "fold_variant_report.json", report)
        output = scratch_dir / "v_diff_metric2"
        assert run_verification(chain, output) == EXIT_MISMATCH

    def test_flipped_prediction_is_mismatch(self, scratch_dir):
        chain = build_fit(scratch_dir)
        records = load_predictions(chain, "fold0")
        records[0]["prediction"] = (records[0]["prediction"] + 1) % 4
        rewrite_jsonl(pred_path(chain, "fold0"), records)
        output = scratch_dir / "v_flipped2"
        assert run_verification(chain, output) == EXIT_MISMATCH
