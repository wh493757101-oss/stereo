"""P0 baseline verification for material_polar_residual_v1.

Re-derives the user-run P0 results from the frozen artifacts, using ONLY:

- the existing admission chain (protocol digest, cache binding, per-fold
  authority, manifest rows, cache arrays) — read-only;
- the per-fold SAVED scaler (loaded from JSON, never refitted, and
  additionally re-derived column by column from the authoritative
  inner_train rows — see ``verify_scaler_columns``);
- the per-fold SAVED P0 linear head (CPU load of a trusted local file);
- a linear forward over the frozen cache rows of that fold's outer_eval.

Strictness: every consumed number (checkpoint tensors, scaler arrays,
saved logits, recomputed logits, metrics, epochs, supports) is checked for
shape, type and finiteness BEFORE any subtraction, argmax or metric
computation. Booleans are never accepted as integers or floats. A single
NaN must refuse the run, not pass a tolerance comparison.

Exit codes (fixed semantics, never derived from error message text):

- 0  — every required check passed;
- 1  (``VerificationRefused``)   — identity, structure, binding,
  non-finite/illegal input, scaler-source mismatch, missing files,
  output conflict;
- 2  (``VerificationMismatch``)  — legal, finite saved logits/predictions/
  metrics that disagree with the recomputation.

The linear forward accumulates in float64 and casts to float32; it is NOT
claimed to be bit-identical to PyTorch's float32 ops, which is why logits
are compared under an explicit tolerance (reported, never loosened).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_probe import (  # noqa: E402
    CONTROL_FEATURE_KEYS,
    FEATURE_DEFINITION_VERSION,
    NUM_CLASSES,
    POLAR_STAT_KEYS,
    Standardizer,
    accuracy,
    assemble_design_matrix,
    authoritative_sample_index,
    macro_f1,
    select_epoch,
    verify_cache_arrays,
    verify_cache_binding,
    verify_cache_manifest,
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_MISMATCH = 2

LOGIT_ATOL = 1e-4
METRIC_ATOL = 1e-6
SCALER_ATOL = 1e-6


class VerificationRefused(Exception):
    """Identity/structure/binding/non-finite/scaler-source failure."""


class VerificationMismatch(Exception):
    """Legal finite saved values disagree with the recomputation."""


# ---------------------------------------------------------------------------
# Strict primitive checks (bool is NOT a number here)


def require_strict_int(name, value, minimum=None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationRefused(
            f"{name} must be a strict integer, got "
            f"{type(value).__name__}: {value!r}")
    if minimum is not None and value < minimum:
        raise VerificationRefused(
            f"{name} must be >= {minimum}, got {value}")
    return value


def require_finite_number(name, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationRefused(
            f"{name} must be a number, got {type(value).__name__}: "
            f"{value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise VerificationRefused(f"{name} is not finite: {value!r}")
    return value


def require_unit_interval(name, value) -> float:
    value = require_finite_number(name, value)
    if not 0.0 <= value <= 1.0:
        raise VerificationRefused(
            f"{name} must lie in [0, 1], got {value!r}")
    return value


def require_finite_array(name, array, expected_shape) -> np.ndarray:
    array = np.asarray(array)
    if array.shape != tuple(expected_shape):
        raise VerificationRefused(
            f"{name} shape {array.shape} != expected "
            f"{tuple(expected_shape)}")
    if not np.issubdtype(array.dtype, np.floating):
        raise VerificationRefused(
            f"{name} dtype {array.dtype} is not a floating type")
    if not np.isfinite(array).all():
        raise VerificationRefused(f"{name} contains NaN/Inf")
    return array


def require_bool_list(name, values) -> list[bool]:
    if not isinstance(values, list):
        raise VerificationRefused(
            f"{name} must be a list, got {type(values).__name__}")
    for index, value in enumerate(values):
        if not isinstance(value, bool):
            raise VerificationRefused(
                f"{name}[{index}] must be a strict boolean, got "
                f"{type(value).__name__}: {value!r}")
    return list(values)


def require_finite_float_list(name, values) -> list[float]:
    if not isinstance(values, list):
        raise VerificationRefused(
            f"{name} must be a list, got {type(values).__name__}")
    return [require_finite_number(f"{name}[{i}]", v)
            for i, v in enumerate(values)]


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_protocol(path: str | Path) -> dict:
    protocol_path = Path(path)
    if protocol_path.is_dir():
        protocol_path = protocol_path / "protocol.json"
    if not protocol_path.is_file():
        raise VerificationRefused(f"protocol not found: {protocol_path}")
    protocol = load_json(protocol_path)
    from core.existing_data_protocol import protocol_digest

    recorded = protocol.get("protocol_sha256", "")
    if not recorded or protocol_digest(protocol) != recorded:
        raise VerificationRefused(
            f"protocol {protocol_path} failed its own digest check")
    return protocol


def verify_feature_cache(feature_dir: str | Path, protocol: dict):
    """The same admission chain the training entry uses (read-only)."""
    feature_dir = Path(feature_dir)
    binding = verify_cache_binding(feature_dir, protocol)
    definition = load_json(feature_dir / "feature_definition.json")
    manifest = load_json(feature_dir / "feature_manifest.json")
    if definition.get("feature_definition_version") != \
            FEATURE_DEFINITION_VERSION:
        raise VerificationRefused("cache feature definition version mismatch")
    if definition.get("data_fingerprint") != protocol["data"]["fingerprint"]:
        raise VerificationRefused(
            "cache data fingerprint does not match the protocol")
    if definition.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise VerificationRefused(
            "cache protocol digest does not match the protocol")
    authority = authoritative_sample_index(protocol)
    samples = verify_cache_manifest(manifest, authority, protocol)
    cache = dict(np.load(feature_dir / "feature_cache.npz"))
    verify_cache_arrays(cache, samples, definition, protocol)
    return binding, definition, manifest, cache, authority


def fold_outer_eval_rows(manifest_samples: list[dict],
                         authority: dict, fold_id: str) -> list[int]:
    """Row indices (the frozen row contract) of one fold's outer_eval."""
    rows = []
    for index, sample in enumerate(manifest_samples):
        if authority[sample["sample_name"]]["roles"].get(fold_id) \
                == "outer_eval":
            rows.append(index)
    if not rows:
        raise VerificationRefused(f"fold {fold_id}: outer_eval part is empty")
    return rows


def fold_inner_train_rows(manifest_samples: list[dict],
                          authority: dict, fold_id: str) -> list[int]:
    rows = []
    for index, sample in enumerate(manifest_samples):
        if authority[sample["sample_name"]]["roles"].get(fold_id) \
                == "inner_train":
            rows.append(index)
    if not rows:
        raise VerificationRefused(
            f"fold {fold_id}: inner_train part is empty")
    return rows


# ---------------------------------------------------------------------------
# Scaler full-column re-derivation (verification of saved parameters, not
# a refit: nothing is written, Standardizer.fit is never called)


def verify_scaler_columns(cache: dict, inner_rows: list[int],
                         saved: Standardizer, fold_id: str) -> dict:
    """Recompute mean/std(ddof=0)/usable for ALL 270 columns read-only.

    G/C statistics use every inner_train row; S statistics only the rows
    with s_valid=true (the existing rule). Any disagreement with the saved
    scaler is a REFUSAL — the saved values are never overwritten.
    """
    g = cache["g_features"][inner_rows]
    c = cache["c_features"][inner_rows]
    s = cache["s_features"][inner_rows]
    s_valid = cache["s_valid"][inner_rows].astype(bool)
    width = g.shape[1] + c.shape[1] + s.shape[1]
    matrix = np.concatenate([g, c, s], axis=1).astype(np.float64)
    column_valid = np.ones((len(inner_rows), width), dtype=bool)
    column_valid[:, g.shape[1] + c.shape[1]:] = s_valid[:, None]

    if saved.mean.shape != (width,) or saved.std.shape != (width,) \
            or saved.usable.shape != (width,):
        raise VerificationRefused(
            f"fold {fold_id}: saved scaler column count "
            f"{saved.mean.shape} does not match the feature layout "
            f"({width},)")
    if not np.isfinite(saved.mean).all() or not np.isfinite(saved.std).all():
        raise VerificationRefused(
            f"fold {fold_id}: saved scaler mean/std contains NaN/Inf")
    if (saved.std < 0).any():
        raise VerificationRefused(
            f"fold {fold_id}: saved scaler has a negative std")

    mean = np.zeros(width, dtype=np.float64)
    std = np.zeros(width, dtype=np.float64)
    usable = np.zeros(width, dtype=bool)
    for column in range(width):
        rows = column_valid[:, column]
        if rows.any():
            values = matrix[rows, column]
            mean[column] = values.mean()
            std[column] = values.std(ddof=0)
            usable[column] = std[column] > 0.0 and np.isfinite(std[column])

    mean_diff = float(np.max(np.abs(mean - saved.mean)))
    std_diff = float(np.max(np.abs(std - saved.std)))
    usable_equal = bool(np.array_equal(usable, saved.usable))
    if mean_diff > SCALER_ATOL or std_diff > SCALER_ATOL or not usable_equal:
        worst_mean = int(np.argmax(np.abs(mean - saved.mean)))
        worst_std = int(np.argmax(np.abs(std - saved.std)))
        raise VerificationRefused(
            f"fold {fold_id}: the saved scaler does not match its "
            f"inner_train source statistics (max mean diff {mean_diff:.3e} "
            f"at column {worst_mean}, max std diff {std_diff:.3e} at column "
            f"{worst_std}, usable equal={usable_equal}, "
            f"atol={SCALER_ATOL}); the saved parameters were not derived "
            "from this fold's inner_train as recorded")

    g_diff = float(np.max(np.abs(mean[:g.shape[1]] - saved.mean[:g.shape[1]])))
    c_diff = float(np.max(np.abs(
        mean[g.shape[1]:g.shape[1] + c.shape[1]]
        - saved.mean[g.shape[1]:g.shape[1] + c.shape[1]])))
    s_diff = float(np.max(np.abs(
        mean[g.shape[1] + c.shape[1]:] - saved.mean[g.shape[1] + c.shape[1]:])))
    return {
        "status": "VERIFIED",
        "inner_train_rows": len(inner_rows),
        "s_valid_rows": int(s_valid.sum()),
        "max_abs_diff": max(mean_diff, std_diff),
        "max_mean_diff_by_slot": {
            "G": g_diff, "C": c_diff, "S": s_diff,
            "all": max(mean_diff, std_diff)},
        "usable_columns_equal": usable_equal,
        "atol": SCALER_ATOL,
        "note": ("read-only re-derivation of the saved scaler's columns "
                 "from the authoritative inner_train rows; G/C use all "
                 "rows, S only s_valid rows; Standardizer.fit was not "
                 "called and nothing was written"),
    }


# ---------------------------------------------------------------------------
# Per-fold verification


def load_saved_scaler(path: Path, fold_id: str) -> Standardizer:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise VerificationRefused(f"fold {fold_id}: scaler.json is not a mapping")
    mean = require_finite_float_list(
        f"fold {fold_id} scaler.mean", payload.get("mean"))
    std = require_finite_float_list(
        f"fold {fold_id} scaler.std", payload.get("std"))
    usable = require_bool_list(
        f"fold {fold_id} scaler.usable", payload.get("usable"))
    if not (len(mean) == len(std) == len(usable)):
        raise VerificationRefused(
            f"fold {fold_id}: scaler mean/std/usable lengths "
            f"{len(mean)}/{len(std)}/{len(usable)} disagree")
    if any(v < 0 for v in std):
        raise VerificationRefused(
            f"fold {fold_id}: scaler has a negative std")
    for column, flag in enumerate(usable):
        if flag and std[column] <= 0.0:
            raise VerificationRefused(
                f"fold {fold_id}: scaler marks column {column} usable with "
                f"std {std[column]}")
    return Standardizer.from_dict(
        {"mean": mean, "std": std, "usable": usable})


def load_checkpoint(path: Path, fold_id: str, expected_seed: int):
    import torch

    if not path.is_file():
        raise VerificationRefused(f"fold {fold_id}: missing model_state.pt")
    payload = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict):
        raise VerificationRefused(
            f"fold {fold_id}: checkpoint payload is not a mapping")
    if payload.get("variant") != "P0" or payload.get("fold") != fold_id:
        raise VerificationRefused(
            f"fold {fold_id}: checkpoint identity "
            f"{payload.get('variant')!r}/{payload.get('fold')!r} does not "
            "match")
    seed = require_strict_int(
        f"fold {fold_id} checkpoint seed", payload.get("seed"))
    if seed != expected_seed:
        raise VerificationRefused(
            f"fold {fold_id}: checkpoint seed {seed} != budget seed "
            f"{expected_seed}")
    input_dim = require_strict_int(
        f"fold {fold_id} checkpoint input_dim", payload.get("input_dim"),
        minimum=1)
    num_classes = require_strict_int(
        f"fold {fold_id} checkpoint num_classes", payload.get("num_classes"))
    if num_classes != NUM_CLASSES:
        raise VerificationRefused(
            f"fold {fold_id}: checkpoint num_classes {num_classes} != "
            f"{NUM_CLASSES}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "weight" not in state \
            or "bias" not in state:
        raise VerificationRefused(
            f"fold {fold_id}: checkpoint state_dict lacks weight/bias")
    weight = require_finite_array(
        f"fold {fold_id} checkpoint weight",
        state["weight"].cpu().numpy(), (NUM_CLASSES, input_dim))
    bias = require_finite_array(
        f"fold {fold_id} checkpoint bias",
        state["bias"].cpu().numpy(), (NUM_CLASSES,))
    budget = payload.get("budget")
    if not isinstance(budget, dict) or "max_epochs" not in budget:
        raise VerificationRefused(
            f"fold {fold_id}: checkpoint carries no budget.max_epochs")
    checkpoint_max_epochs = require_strict_int(
        f"fold {fold_id} checkpoint budget.max_epochs",
        budget.get("max_epochs"), minimum=1)
    return weight, bias, input_dim, checkpoint_max_epochs


def verify_saved_logits(name: str, record: dict) -> list[float]:
    logits = record.get("logits")
    if not isinstance(logits, list) or len(logits) != NUM_CLASSES:
        raise VerificationRefused(
            f"{name}: saved logits must be a list of {NUM_CLASSES} numbers, "
            f"got {type(logits).__name__} "
            f"(len {len(logits) if isinstance(logits, list) else 'n/a'})")
    return [require_finite_number(f"{name} logits[{i}]", v)
            for i, v in enumerate(logits)]


def linear_forward(design: np.ndarray, weight: np.ndarray,
                   bias: np.ndarray) -> np.ndarray:
    """float64 accumulation, cast to float32 (NOT bit-identical to
    PyTorch's float32 ops; logits compare under LOGIT_ATOL)."""
    return (design.astype(np.float64) @ weight.astype(np.float64).T
            + bias.astype(np.float64)).astype(np.float32)


def confusion_and_recalls(labels: list[int], predictions: list[int]) -> dict:
    class_names = ("metal_submarine", "plastic_submarine",
                   "plastic_fish", "real_fish")
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for label, prediction in zip(labels, predictions):
        confusion[label][prediction] += 1
    recalls = {}
    for class_id in range(NUM_CLASSES):
        total = sum(confusion[class_id])
        recalls[class_names[class_id]] = (
            confusion[class_id][class_id] / total if total else None)
    return {
        "confusion_matrix": confusion,
        "confusion_rows": "true class",
        "confusion_cols": "predicted class",
        "per_class_recall": recalls,
    }


def verify_p0_fold_records(summary: dict, protocol: dict,
                           authority: dict,
                           manifest_samples: list) -> dict:
    """ONE-TIME set-level verification of the summary's P0 fold records.

    The fold identities must come EXACTLY from the protocol (no missing,
    extra or duplicated records; illegal entries are refused, never
    filtered away). Each record's variant, support and selected-epoch
    types are checked here; the numeric cross-checks against the history
    happen per fold in ``verify_fold``.
    """
    oof = summary.get("oof", {}).get("P0", {})
    records = oof.get("folds")
    if not isinstance(records, list) or not records:
        raise VerificationRefused(
            "summary.oof.P0.folds must be a non-empty list of records")
    expected_rows = {
        fold["fold_id"]: sum(
            1 for sample in manifest_samples
            if authority[sample["sample_name"]]["roles"].get(
                fold["fold_id"]) == "outer_eval")
        for fold in protocol["folds"]["folds"]
    }
    protocol_folds = set(expected_rows)
    by_fold: dict[str, dict] = {}
    for index, record in enumerate(records):
        name = f"summary.oof.P0.folds[{index}]"
        if not isinstance(record, dict):
            raise VerificationRefused(f"{name} is not a mapping")
        fold_id = record.get("fold")
        if fold_id not in protocol_folds:
            raise VerificationRefused(
                f"{name}: fold {fold_id!r} is not a protocol fold "
                f"{sorted(protocol_folds)}")
        if fold_id in by_fold:
            raise VerificationRefused(
                f"{name}: duplicate record for fold {fold_id!r}")
        if record.get("variant") != "P0":
            raise VerificationRefused(
                f"{name}: variant {record.get('variant')!r} != 'P0'")
        support = require_strict_int(
            f"{name} support", record.get("support"), minimum=0)
        if support != expected_rows[fold_id]:
            raise VerificationRefused(
                f"{name}: support {support} != the authoritative "
                f"outer_eval count {expected_rows[fold_id]}")
        require_strict_int(
            f"{name} selected_epoch", record.get("selected_epoch"),
            minimum=1)
        require_unit_interval(
            f"{name} inner_val_macro_f1",
            record.get("inner_val_macro_f1"))
        require_unit_interval(
            f"{name} inner_val_accuracy",
            record.get("inner_val_accuracy"))
        require_unit_interval(
            f"{name} outer_eval_macro_f1",
            record.get("outer_eval_macro_f1"))
        require_unit_interval(
            f"{name} outer_eval_accuracy",
            record.get("outer_eval_accuracy"))
        by_fold[fold_id] = record
    missing = sorted(protocol_folds - set(by_fold))
    if missing:
        raise VerificationRefused(
            f"summary.oof.P0.folds is missing records for folds {missing}")
    return by_fold


def verify_fold(
    fold_id: str,
    fold_dir: Path,
    cache: dict,
    manifest_samples: list[dict],
    authority: dict,
    summary: dict,
    class_names: list[str],
    p0_fold_records: dict,
) -> dict:
    """One fold's P0 re-derivation; REFUSED on structural problems,
    MISMATCH on finite numeric disagreement."""
    import hashlib

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    paths = {
        "model_state.pt": fold_dir / "model_state.pt",
        "scaler.json": fold_dir / "scaler.json",
        "history.json": fold_dir / "history.json",
        "outer_eval_predictions.jsonl":
            fold_dir / "outer_eval_predictions.jsonl",
        "fold_variant_report.json": fold_dir / "fold_variant_report.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise VerificationRefused(f"fold {fold_id}: missing {name}")

    expected_seed = summary["admission"]["budget"]["seed"]
    weight, bias, checkpoint_dim, checkpoint_max_epochs = load_checkpoint(
        paths["model_state.pt"], fold_id, expected_seed)
    budget_max_epochs = require_strict_int(
        f"summary admission budget.max_epochs",
        summary["admission"]["budget"].get("max_epochs"), minimum=1)
    if checkpoint_max_epochs != budget_max_epochs:
        raise VerificationRefused(
            f"fold {fold_id}: checkpoint budget.max_epochs "
            f"{checkpoint_max_epochs} != the summary admission budget "
            f"{budget_max_epochs}; a single changed budget record must not "
            "legalise a truncated history")

    # --- saved scaler: strict shape/finite/bool checks, then full-column
    # re-derivation against the authoritative inner_train rows
    saved_scaler = load_saved_scaler(paths["scaler.json"], fold_id)
    inner_rows = fold_inner_train_rows(manifest_samples, authority, fold_id)
    scaler_check = verify_scaler_columns(
        cache, inner_rows, saved_scaler, fold_id)

    rows = fold_outer_eval_rows(manifest_samples, authority, fold_id)
    g = cache["g_features"][rows]
    c = cache["c_features"][rows]
    s = cache["s_features"][rows]
    s_valid = cache["s_valid"][rows].astype(bool)
    column_valid = np.ones(
        (len(rows), g.shape[1] + c.shape[1] + s.shape[1]), dtype=bool)
    column_valid[:, g.shape[1] + c.shape[1]:] = s_valid[:, None]
    raw = np.concatenate([g, c, s], axis=1)
    standardized = saved_scaler.transform(raw, column_valid)
    g_dim = g.shape[1]
    design = assemble_design_matrix(
        standardized[:, :g_dim],
        standardized[:, g_dim:g_dim + len(CONTROL_FEATURE_KEYS)],
        standardized[:, g_dim + len(CONTROL_FEATURE_KEYS):], "P0")
    if design.shape[1] != checkpoint_dim:
        raise VerificationRefused(
            f"fold {fold_id}: design width {design.shape[1]} != checkpoint "
            f"input_dim {checkpoint_dim}")
    labels = [int(v) for v in cache["class_id"][rows]]

    logits = linear_forward(design, weight, bias)
    if not np.isfinite(logits).all():
        raise VerificationRefused(
            f"fold {fold_id}: recomputed logits are not finite")
    predictions = [int(v) for v in logits.argmax(axis=1)]

    # --- saved per-sample predictions: identity first, then numbers
    saved = [json.loads(line) for line in
             paths["outer_eval_predictions.jsonl"].read_text(
                 encoding="utf-8").splitlines() if line.strip()]
    if len(saved) != len(rows):
        raise VerificationRefused(
            f"fold {fold_id}: saved predictions {len(saved)} != outer_eval "
            f"rows {len(rows)}")
    seen_names = set()
    mismatched = []
    max_logit_diff = 0.0
    for index, (row, record) in enumerate(zip(rows, saved)):
        if not isinstance(record, dict):
            raise VerificationRefused(
                f"fold {fold_id}: prediction row {index} is not an object")
        name = record.get("sample_name")
        expected_name = manifest_samples[row]["sample_name"]
        if name != expected_name:
            raise VerificationRefused(
                f"fold {fold_id}: row {index} sample_name {name!r} != "
                f"authority {expected_name!r} (row-order contract broken)")
        if name in seen_names:
            raise VerificationRefused(
                f"fold {fold_id}: duplicate sample_name {name!r} in the "
                "saved predictions")
        seen_names.add(name)
        if record.get("group") != manifest_samples[row]["group"]:
            raise VerificationRefused(
                f"fold {fold_id}: {name} saved group "
                f"{record.get('group')!r} != authority "
                f"{manifest_samples[row]['group']!r}")
        saved_class = record.get("class_id")
        require_strict_int(
            f"fold {fold_id} {name} class_id", saved_class, minimum=0)
        if saved_class != labels[index]:
            raise VerificationRefused(
                f"fold {fold_id}: {name} saved class_id {saved_class} != "
                f"cache {labels[index]}")
        saved_logits = verify_saved_logits(
            f"fold {fold_id} {name}", record)
        saved_prediction = require_strict_int(
            f"fold {fold_id} {name} prediction", record.get("prediction"),
            minimum=0)
        row_diff = float(np.max(np.abs(
            np.asarray(saved_logits, dtype=np.float64)
            - logits[index].astype(np.float64))))
        max_logit_diff = max(max_logit_diff, row_diff)
        if row_diff > LOGIT_ATOL:
            mismatched.append({
                "sample_name": name, "kind": "logits",
                "max_abs_diff": row_diff,
                "saved_prediction": saved_prediction,
                "recomputed_prediction": predictions[index]})
            continue
        if saved_prediction != predictions[index]:
            mismatched.append({
                "sample_name": name, "kind": "prediction",
                "saved_prediction": saved_prediction,
                "recomputed_prediction": predictions[index]})
    if mismatched:
        raise VerificationMismatch(
            f"fold {fold_id}: {len(mismatched)} outer_eval rows disagree "
            f"with the recomputed P0 forward (first: {mismatched[0]}); "
            f"max saved-vs-recomputed logit diff {max_logit_diff:.3e} "
            f"(atol {LOGIT_ATOL})")

    recomputed_macro = macro_f1(labels, predictions)
    recomputed_acc = accuracy(labels, predictions)
    for name, value in (("macro_f1", recomputed_macro),
                        ("accuracy", recomputed_acc)):
        if not np.isfinite(value):
            raise VerificationRefused(
                f"fold {fold_id}: recomputed {name} is not finite")

    # --- fold report identity + metrics
    fold_report = load_json(paths["fold_variant_report.json"])
    if not isinstance(fold_report, dict):
        raise VerificationRefused(
            f"fold {fold_id}: fold_variant_report.json is not a mapping")
    if fold_report.get("fold") != fold_id:
        raise VerificationRefused(
            f"fold {fold_id}: fold report records fold "
            f"{fold_report.get('fold')!r}")
    if fold_report.get("variant") != "P0":
        raise VerificationRefused(
            f"fold {fold_id}: fold report records variant "
            f"{fold_report.get('variant')!r}")
    support = require_strict_int(
        f"fold {fold_id} fold report support", fold_report.get("support"))
    if support != len(rows):
        raise VerificationRefused(
            f"fold {fold_id}: fold report support {support} != outer_eval "
            f"rows {len(rows)}")
    report_macro = require_unit_interval(
        f"fold {fold_id} fold report outer_eval_macro_f1",
        fold_report.get("outer_eval_macro_f1"))
    report_acc = require_unit_interval(
        f"fold {fold_id} fold report outer_eval_accuracy",
        fold_report.get("outer_eval_accuracy"))
    report_inner_macro = require_unit_interval(
        f"fold {fold_id} fold report inner_val_macro_f1",
        fold_report.get("inner_val_macro_f1"))
    report_inner_acc = require_unit_interval(
        f"fold {fold_id} fold report inner_val_accuracy",
        fold_report.get("inner_val_accuracy"))
    for name, saved_value, recomputed in (
            ("outer_eval_macro_f1", report_macro, recomputed_macro),
            ("outer_eval_accuracy", report_acc, recomputed_acc)):
        if abs(saved_value - recomputed) > METRIC_ATOL:
            raise VerificationMismatch(
                f"fold {fold_id}: fold report {name} {saved_value} != "
                f"recomputed {recomputed} (atol {METRIC_ATOL})")

    # --- selection: fold report, history selection, and a read-only
    # recomputation of the best epoch from the FULL saved history
    history = load_json(paths["history.json"])
    if not isinstance(history, dict) or not isinstance(
            history.get("epochs"), list) or not history["epochs"]:
        raise VerificationRefused(
            f"fold {fold_id}: history.json carries no epoch records; the "
            "best-epoch recomputation cannot be verified (reported "
            "UNVERIFIED for the selection cross-check, refusing to "
            "fabricate evidence)")
    if len(history["epochs"]) != budget_max_epochs:
        raise VerificationRefused(
            f"fold {fold_id}: history carries {len(history['epochs'])} "
            f"epoch records != the training budget max_epochs "
            f"{budget_max_epochs}; a truncated history is refused instead "
            "of being reported as fully verified")
    epoch_records = []
    seen_epochs: set[int] = set()
    previous_epoch = 0
    for record in history["epochs"]:
        if not isinstance(record, dict):
            raise VerificationRefused(
                f"fold {fold_id}: a history epoch record is not an object")
        epoch = require_strict_int(
            f"fold {fold_id} history epoch", record.get("epoch"), minimum=1)
        if epoch > budget_max_epochs:
            raise VerificationRefused(
                f"fold {fold_id}: history epoch {epoch} exceeds the "
                f"training budget max_epochs {budget_max_epochs}")
        if epoch in seen_epochs:
            raise VerificationRefused(
                f"fold {fold_id}: duplicate history epoch {epoch}")
        if epoch <= previous_epoch:
            raise VerificationRefused(
                f"fold {fold_id}: history epochs are not in the "
                "production ascending order (previous "
                f"{previous_epoch}, current {epoch})")
        seen_epochs.add(epoch)
        previous_epoch = epoch
        macro = require_unit_interval(
            f"fold {fold_id} history epoch {epoch} inner_val_macro_f1",
            record.get("inner_val_macro_f1"))
        acc = require_unit_interval(
            f"fold {fold_id} history epoch {epoch} inner_val_accuracy",
            record.get("inner_val_accuracy"))
        if "train_loss" in record:
            loss = require_finite_number(
                f"fold {fold_id} history epoch {epoch} train_loss",
                record.get("train_loss"))
            if loss < 0.0:
                raise VerificationRefused(
                    f"fold {fold_id}: history epoch {epoch} train_loss "
                    f"{loss} is negative")
        epoch_records.append({"epoch": epoch,
                              "inner_val_macro_f1": macro,
                              "inner_val_accuracy": acc})
    missing_epochs = sorted(
        set(range(1, budget_max_epochs + 1)) - seen_epochs)
    if missing_epochs:
        raise VerificationRefused(
            f"fold {fold_id}: history is missing epochs {missing_epochs}")
    history_check = {
        "status": "VERIFIED",
        "epochs": len(history["epochs"]),
        "budget_max_epochs": budget_max_epochs,
        "checkpoint_budget_max_epochs": checkpoint_max_epochs,
        "epoch_range": f"1..{budget_max_epochs}",
        "ascending_order": True,
        "note": ("read-only completeness check against the admitted "
                 "training budget; the selection was re-derived from the "
                 "FULL history under the frozen rule; no checkpoint was "
                 "re-selected or rewritten"),
    }
    recomputed_best = select_epoch(epoch_records)
    selection = history.get("selection", {})
    if not isinstance(selection, dict):
        raise VerificationRefused(
            f"fold {fold_id}: history.selection is not a mapping")
    saved_selection_epoch = require_strict_int(
        f"fold {fold_id} history selection epoch",
        selection.get("epoch"), minimum=1)
    saved_selection_macro = require_finite_number(
        f"fold {fold_id} history selection inner_val_macro_f1",
        selection.get("inner_val_macro_f1"))
    saved_selection_acc = require_finite_number(
        f"fold {fold_id} history selection inner_val_accuracy",
        selection.get("inner_val_accuracy"))
    if saved_selection_epoch != recomputed_best["epoch"]:
        raise VerificationRefused(
            f"fold {fold_id}: history selection epoch "
            f"{saved_selection_epoch} is not the full-history best under "
            f"the frozen rule (recomputed {recomputed_best['epoch']})")
    for name, saved_value, recomputed_value in (
            ("inner_val_macro_f1", saved_selection_macro,
             recomputed_best["inner_val_macro_f1"]),
            ("inner_val_accuracy", saved_selection_acc,
             recomputed_best["inner_val_accuracy"])):
        if abs(saved_value - recomputed_value) > METRIC_ATOL:
            raise VerificationRefused(
                f"fold {fold_id}: history selection {name} {saved_value} != "
                f"the selected epoch's record {recomputed_value}")
    report_epoch = require_strict_int(
        f"fold {fold_id} fold report selected_epoch",
        fold_report.get("selected_epoch"), minimum=1)
    if report_epoch != saved_selection_epoch:
        raise VerificationRefused(
            f"fold {fold_id}: fold report selected_epoch {report_epoch} != "
            f"history selection {saved_selection_epoch}")
    for name, saved_value, recomputed_value in (
            ("inner_val_macro_f1", report_inner_macro,
             saved_selection_macro),
            ("inner_val_accuracy", report_inner_acc, saved_selection_acc)):
        if abs(saved_value - recomputed_value) > METRIC_ATOL:
            raise VerificationRefused(
                f"fold {fold_id}: fold report {name} {saved_value} != the "
                f"history selection {recomputed_value}")

    # --- summary oof record for this fold (identity checked at set
    # level; the inner metrics cross-check against the history selection)
    summary_record = p0_fold_records[fold_id]
    summary_macro = require_unit_interval(
        f"fold {fold_id} summary outer_eval_macro_f1",
        summary_record.get("outer_eval_macro_f1"))
    summary_acc = require_unit_interval(
        f"fold {fold_id} summary outer_eval_accuracy",
        summary_record.get("outer_eval_accuracy"))
    summary_inner_macro = require_unit_interval(
        f"fold {fold_id} summary inner_val_macro_f1",
        summary_record.get("inner_val_macro_f1"))
    summary_inner_acc = require_unit_interval(
        f"fold {fold_id} summary inner_val_accuracy",
        summary_record.get("inner_val_accuracy"))
    for name, saved_value, recomputed_value in (
            ("inner_val_macro_f1", summary_inner_macro,
             saved_selection_macro),
            ("inner_val_accuracy", summary_inner_acc, saved_selection_acc)):
        if abs(saved_value - recomputed_value) > METRIC_ATOL:
            raise VerificationRefused(
                f"fold {fold_id}: summary {name} {saved_value} != the "
                f"history selection {recomputed_value}")
    for name, saved_value, recomputed_value in (
            ("outer_eval_macro_f1", summary_macro, recomputed_macro),
            ("outer_eval_accuracy", summary_acc, recomputed_acc)):
        if abs(saved_value - recomputed_value) > METRIC_ATOL:
            raise VerificationMismatch(
                f"fold {fold_id}: summary {name} {saved_value} != "
                f"recomputed {recomputed_value} (atol {METRIC_ATOL})")
    summary_epoch = require_strict_int(
        f"fold {fold_id} summary selected_epoch",
        summary_record.get("selected_epoch"), minimum=1)
    if summary_epoch != saved_selection_epoch:
        raise VerificationRefused(
            f"fold {fold_id}: summary selected_epoch {summary_epoch} != "
            f"history selection {saved_selection_epoch}")

    return {
        "fold": fold_id,
        "rows": len(rows),
        "selected_epoch": saved_selection_epoch,
        "inner_val_macro_f1": saved_selection_macro,
        "labels": labels,
        "predictions": predictions,
        "recomputed": {
            "macro_f1": recomputed_macro,
            "accuracy": recomputed_acc,
            **confusion_and_recalls(labels, predictions),
        },
        "history_check": history_check,
        "max_saved_recomputed_logit_diff": max_logit_diff,
        "predictions_status": (
            "predictions match; logits within tolerance"
            if max_logit_diff > 0.0 else
            "predictions match; logits exactly equal"),
        "metrics_status": "metrics match the recomputation",
        "scaler_check": scaler_check,
        "files": {name: _sha(path) for name, path in paths.items()},
    }


def check_summary_aggregates(summary: dict, class_names: list[str],
                             fold_results: list[dict],
                             pooled_labels: list[int],
                             pooled_predictions: list[int]) -> dict:
    """Three-fold means and the saved P0 pooled value, against the
    recomputed in-memory labels/predictions."""
    recomputed_mean_macro = float(np.mean(
        [r["recomputed"]["macro_f1"] for r in fold_results]))
    recomputed_mean_acc = float(np.mean(
        [r["recomputed"]["accuracy"] for r in fold_results]))
    oof = summary.get("oof", {}).get("P0", {})
    saved_mean_macro = require_finite_number(
        "summary oof.P0.macro_f1_mean", oof.get("macro_f1_mean"))
    saved_mean_acc = require_finite_number(
        "summary oof.P0.accuracy_mean", oof.get("accuracy_mean"))
    if abs(saved_mean_macro - recomputed_mean_macro) > METRIC_ATOL:
        raise VerificationMismatch(
            f"summary oof.P0.macro_f1_mean {saved_mean_macro} != the "
            f"recomputed three-fold mean {recomputed_mean_macro} "
            f"(atol {METRIC_ATOL})")
    if abs(saved_mean_acc - recomputed_mean_acc) > METRIC_ATOL:
        raise VerificationMismatch(
            f"summary oof.P0.accuracy_mean {saved_mean_acc} != the "
            f"recomputed three-fold mean {recomputed_mean_acc} "
            f"(atol {METRIC_ATOL})")

    recomputed_pooled = {
        "sample_count": len(pooled_labels),
        "macro_f1": macro_f1(pooled_labels, pooled_predictions),
        "accuracy": accuracy(pooled_labels, pooled_predictions),
        **confusion_and_recalls(pooled_labels, pooled_predictions),
    }
    saved_pooled = None
    paired = summary.get("paired", {})
    for record in paired.values():
        if not isinstance(record, dict):
            continue
        pooled = record.get("pooled_macro_f1")
        if isinstance(pooled, dict) and "P0" in pooled:
            saved_pooled = require_finite_number(
                "summary paired pooled_macro_f1.P0", pooled["P0"])
            break
    pooled_status = "NOT_RECORDED"
    if saved_pooled is not None:
        pooled_status = "VERIFIED"
        if abs(saved_pooled - recomputed_pooled["macro_f1"]) > METRIC_ATOL:
            raise VerificationMismatch(
                f"summary paired pooled_macro_f1.P0 {saved_pooled} != the "
                f"recomputed pooled Macro-F1 {recomputed_pooled['macro_f1']} "
                f"(atol {METRIC_ATOL})")
    return {
        "recomputed_mean_macro_f1": recomputed_mean_macro,
        "recomputed_mean_accuracy": recomputed_mean_acc,
        "saved_mean_macro_f1": saved_mean_macro,
        "saved_mean_accuracy": saved_mean_acc,
        "pooled": recomputed_pooled,
        "saved_pooled_macro_f1": saved_pooled,
        "saved_pooled_status": pooled_status,
    }


def check_admission_identity(summary: dict, protocol: dict,
                             definition: dict) -> None:
    """The admission fields the fit actually recorded and the verifier
    depends on (no demand for fields history never saved)."""
    admission = summary.get("admission", {})
    if not isinstance(admission, dict):
        raise VerificationRefused("summary.admission is not a mapping")
    if admission.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise VerificationRefused(
            "the fit's admission protocol digest does not match this "
            "protocol")
    saved_classes = admission.get("class_names")
    if list(saved_classes or []) != list(protocol["data"]["class_names"]):
        raise VerificationRefused(
            f"summary admission class order {saved_classes!r} != the "
            f"protocol {list(protocol['data']['class_names'])!r}")
    saved_version = admission.get("feature_definition_version")
    if saved_version != definition.get("feature_definition_version"):
        raise VerificationRefused(
            f"summary admission feature_definition_version {saved_version!r} "
            f"!= the cache {definition.get('feature_definition_version')!r}")
    saved_preproc = admission.get("preprocessing_version")
    if saved_preproc is None:
        raise VerificationRefused(
            "summary admission preprocessing_version is missing; a field "
            "the fit recorded cannot be skipped by deletion or None")
    if saved_preproc != definition.get("preprocessing_version"):
        raise VerificationRefused(
            f"summary admission preprocessing_version {saved_preproc!r} != "
            f"the cache {definition.get('preprocessing_version')!r}")
    saved_g_dim = admission.get("g_dim")
    if saved_g_dim is None:
        raise VerificationRefused(
            "summary admission g_dim is missing")
    require_strict_int("summary admission g_dim", saved_g_dim, minimum=1)
    if int(saved_g_dim) != int(definition["slot_G"]["dim"]):
        raise VerificationRefused(
            f"summary admission g_dim {saved_g_dim} != the cache "
            f"{definition['slot_G']['dim']}")
    saved_imgsz = admission.get("imgsz")
    if saved_imgsz is None:
        raise VerificationRefused(
            "summary admission imgsz is missing")
    require_strict_int("summary admission imgsz", saved_imgsz, minimum=1)
    if int(saved_imgsz) != int(definition.get("imgsz", -1)):
        raise VerificationRefused(
            f"summary admission imgsz {saved_imgsz} != the cache "
            f"{definition.get('imgsz')}")
    saved_input_dim = admission.get("input_dim")
    if saved_input_dim is None:
        raise VerificationRefused(
            "summary admission input_dim is missing")
    require_strict_int(
        "summary admission input_dim", saved_input_dim, minimum=1)
    cache_width = (int(definition["slot_G"]["dim"])
                   + len(CONTROL_FEATURE_KEYS) + len(POLAR_STAT_KEYS))
    if int(saved_input_dim) != cache_width:
        raise VerificationRefused(
            f"summary admission input_dim {saved_input_dim} != the cache "
            f"width G+C+S = {cache_width}")
    budget = admission.get("budget")
    if not isinstance(budget, dict):
        raise VerificationRefused(
            "summary admission budget is missing or not a mapping")
    require_strict_int(
        "summary admission budget.seed", budget.get("seed"), minimum=1)
    require_strict_int(
        "summary admission budget.max_epochs", budget.get("max_epochs"),
        minimum=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-derive the user-run P0 baseline (read-only; no "
                    "training, no encoder).")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--fit-dir", required=True,
                        help="the user's fit directory (contains summary.json "
                        "and models/<fold>_P0/)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    if Path(args.output_dir).exists():
        print(
            f"refused: output directory {args.output_dir} already exists",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    try:
        import hashlib

        protocol = load_protocol(args.protocol)
        binding, definition, manifest, cache, authority = \
            verify_feature_cache(args.features, protocol)
        fit_dir = Path(args.fit_dir)
        summary_path = fit_dir / "summary.json"
        if not summary_path.is_file():
            raise VerificationRefused(
                f"fit summary missing: {summary_path}")
        summary = load_json(summary_path)
        check_admission_identity(summary, protocol, definition)
        saved_binding = summary.get("admission", {}).get("cache_binding",
                                                         {}).get("files", {})
        for name, record in binding["files"].items():
            saved = saved_binding.get(name, {})
            if saved.get("sha256") != record["sha256"]:
                raise VerificationRefused(
                    f"fit admission binding for {name} does not match the "
                    f"verified cache ({saved.get('sha256')} vs "
                    f"{record['sha256']})")

        class_names = list(protocol["data"]["class_names"])
        # one-time set-level verification of the summary's P0 fold records
        p0_fold_records = verify_p0_fold_records(
            summary, protocol, authority, manifest["samples"])
        fold_results = []
        pooled_labels: list[int] = []
        pooled_predictions: list[int] = []
        for fold_record in protocol["folds"]["folds"]:
            fold_id = fold_record["fold_id"]
            fold_dir = fit_dir / "models" / f"{fold_id}_P0"
            result = verify_fold(
                fold_id, fold_dir, cache, manifest["samples"], authority,
                summary, class_names, p0_fold_records)
            fold_results.append(result)
            # the pooled aggregation consumes the ALREADY VERIFIED
            # in-memory labels/predictions returned by verify_fold — the
            # prediction JSONL files are never re-parsed here
            pooled_labels.extend(result["labels"])
            pooled_predictions.extend(result["predictions"])
            print(f"  {fold_id}: rows={result['rows']} "
                  f"macro_f1={result['recomputed']['macro_f1']:.4f} "
                  f"acc={result['recomputed']['accuracy']:.4f} "
                  f"scaler={result['scaler_check']['status']}",
                  file=sys.stderr)

        aggregates = check_summary_aggregates(
            summary, class_names, fold_results, pooled_labels,
            pooled_predictions)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True)
        report = {
            "status": "COMPLETED",
            "protocol_sha256": protocol["protocol_sha256"],
            "cache_binding": {
                name: record["sha256"]
                for name, record in binding["files"].items()
            },
            "fit_dir": str(fit_dir),
            "fit_summary_sha256": hashlib.sha256(
                summary_path.read_bytes()).hexdigest(),
            "feature_definition_version": definition[
                "feature_definition_version"],
            "class_names": class_names,
            "tolerances": {
                "logit_atol": LOGIT_ATOL,
                "metric_atol": METRIC_ATOL,
                "scaler_atol": SCALER_ATOL,
                "forward_note": (
                    "the recomputation accumulates in float64 and casts to "
                    "float32; it is NOT bit-identical to PyTorch float32 "
                    "ops, hence the explicit logit tolerance"),
            },
            "folds": fold_results,
            "summary_aggregates": aggregates,
            "boundaries": {
                "verified_identities": [
                    "protocol digest; cache binding bytes; per-fold "
                    "authority and row order; per-sample name/group/class; "
                    "checkpoint variant/fold/seed/shapes; fold report and "
                    "summary fold/variant/support/epochs; selection records "
                    "cross-checked against a read-only recomputation of the "
                    "best epoch over the FULL saved history",
                ],
                "recomputed_values": [
                    "outer_eval logits/predictions; per-fold Macro-F1, "
                    "accuracy, per-class recall and confusion; three-fold "
                    "means; pooled outer_eval metrics (from verified "
                    "in-memory labels/predictions)",
                ],
                "not_verified": [
                    "the residual variants R0/RC/RCS are FROZEN but NOT "
                    "trained or evaluated in this round",
                    "no encoder forward was run (G comes from the cache)",
                    "the scaler was re-derived read-only for verification, "
                    "never refitted or overwritten",
                    "P1/P2 training quality is out of scope (only the P0 "
                    "fields the summary records were consumed)",
                ],
            },
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        }
        text = json.dumps(report, ensure_ascii=False, indent=2,
                          sort_keys=True, allow_nan=False) + "\n"
        (output_dir / "p0_baseline_verification.json").write_text(
            text, encoding="utf-8")
        print(f"verification written: "
              f"{output_dir / 'p0_baseline_verification.json'}",
              file=sys.stderr)
        return EXIT_OK
    except VerificationMismatch as error:
        print(f"MISMATCH: {error}", file=sys.stderr)
        return EXIT_MISMATCH
    except VerificationRefused as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"refused: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
