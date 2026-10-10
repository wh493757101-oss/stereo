"""Frozen-feature probe contract for material_existing_data_v1.

This module holds everything the P0/P1/P2 probe needs that must be
*identical* between the feature-preparation entry and the user-run
training entry:

- the feature definitions of the three fixed slots
  (**G** frozen MiT-B0 last-stage mask-pooled gray feature,
  **C** measurement-quality control vector, **S** polarization statistics
  on ``instance_mask & polar_valid`` of the original crop NPZ);
- the variant slot layout (P0 = G, P1 = G + C, P2 = G + C + S) with
  unused slots zeroed so every variant shares one head dimension, one
  initialization and one budget;
- the per-column standardizer (fit on inner_train only, S columns from
  inner_train rows with a valid S only, deterministic rules for missing
  S and zero-variance columns);
- the linear 4-class softmax head and its fixed full-batch training loop
  with the frozen selection rule (inner_val Macro-F1, then accuracy,
  then the earlier epoch).

Leakage rules enforced here (and unit-tested):

- ``quality[3]`` (``mean_abs_q``) is polarization content and can only
  enter **S**; **S** is computed from the ``signed_q``/``abs_q`` arrays,
  never from the quality vector. Slot C reads quality[0:3] only.
- ``instance_mask`` is never taken from the valid channel or from
  ``q != 0`` positions.

Nothing in this module trains on, loads or touches real project data;
all functions operate on the values handed to them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

FEATURE_DEFINITION_VERSION = "existing_data_probe_v1"

# Slot C: the FIRST THREE quality entries plus two in-mask gray statistics
# and an explicit "S region non-empty" flag. quality[3] (mean_abs_q) is
# deliberately absent — it is polarization content, slot S's business.
CONTROL_FEATURE_KEYS = (
    "quality_valid_ratio",
    "quality_in_bounds_ratio",
    "quality_brightness_valid_ratio",
    "gray_mean_in_instance_mask",
    "gray_std_in_instance_mask",
    "s_region_nonempty_flag",
)

# Slot S: polarization statistics computed on instance_mask & polar_valid
# of the ORIGINAL crop npz (not on the resized tensors).
POLAR_STAT_KEYS = (
    "signed_q_mean",
    "signed_q_std_ddof0",
    "signed_q_p10",
    "signed_q_p25",
    "signed_q_p50",
    "signed_q_p75",
    "signed_q_p90",
    "abs_q_mean",
)
QUANTILE_PERCENTILES = (10.0, 25.0, 50.0, 75.0, 90.0)
QUANTILE_METHOD = "numpy.percentile (linear interpolation)"

SLOTS = ("G", "C", "S")
VARIANT_SLOTS = {
    "P0": (True, False, False),
    "P1": (True, True, False),
    "P2": (True, True, True),
}
NUM_CLASSES = 4


def validate_variant(variant: str) -> str:
    if variant not in VARIANT_SLOTS:
        raise ValueError(
            f"unknown probe variant {variant!r}; expected one of "
            f"{sorted(VARIANT_SLOTS)}"
        )
    return variant


def slot_dimensions(g_dim: int) -> tuple[int, int, int]:
    """The (G, C, S) slot widths for a G feature of dimension ``g_dim``."""
    return (int(g_dim), len(CONTROL_FEATURE_KEYS), len(POLAR_STAT_KEYS))


def total_input_dim(g_dim: int) -> int:
    g, c, s = slot_dimensions(g_dim)
    return g + c + s


# ---------------------------------------------------------------------------
# Slot C / slot S construction (pure numpy, one V5 sample's arrays)


def build_control_features(
    *,
    quality: np.ndarray,
    gray: np.ndarray,
    instance_mask: np.ndarray,
    s_region_nonempty: bool,
) -> np.ndarray:
    """Slot C for one sample (see ``CONTROL_FEATURE_KEYS``).

    ``quality`` is the sample's float32(4,) vector — only entries 0:3 are
    read. The gray statistics use the ORIGINAL uint8 crop scaled to [0, 1]
    inside the annotated instance mask. A missing S region enters only
    through the explicit flag, never as a fake measurement.
    """
    quality = np.asarray(quality, dtype=np.float32)
    if quality.shape != (4,):
        raise ValueError(f"quality must have shape (4,), got {quality.shape}")
    gray = np.asarray(gray)
    mask = np.asarray(instance_mask) > 0
    pixels = gray[mask].astype(np.float64) / 255.0
    if pixels.size:
        gray_mean = float(pixels.mean())
        gray_std = float(pixels.std(ddof=0))
    else:
        # An empty instance mask cannot happen for an admitted V5 sample,
        # but the rule must stay finite and explicit anyway.
        gray_mean, gray_std = 0.0, 0.0
    values = (
        float(quality[0]),
        float(quality[1]),
        float(quality[2]),
        gray_mean,
        gray_std,
        1.0 if s_region_nonempty else 0.0,
    )
    return np.asarray(values, dtype=np.float32)


def build_polar_stats(
    *,
    signed_q: np.ndarray,
    abs_q: np.ndarray,
    valid: np.ndarray,
    instance_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Slot S for one sample, on ``instance_mask & polar_valid`` only.

    Returns ``(values, has_valid_s)``. Without valid pixels the state is
    an explicit miss: ``has_valid_s=False`` and every value NaN — never a
    fabricated zero measurement. NaN/Inf inside the measured region is
    refused (the V5 contract stores zeros outside the valid region; a
    non-finite *measured* value means the sample is broken).
    """
    signed_q = np.asarray(signed_q, dtype=np.float32)
    abs_q = np.asarray(abs_q, dtype=np.float32)
    region = (np.asarray(valid) > 0) & (np.asarray(instance_mask) > 0)
    count = int(region.sum())
    if count == 0:
        return (
            np.full(len(POLAR_STAT_KEYS), np.nan, dtype=np.float32),
            False,
        )
    signed_values = signed_q[region]
    abs_values = abs_q[region]
    if not np.isfinite(signed_values).all() or not np.isfinite(abs_values).all():
        raise ValueError(
            "non-finite polarization value inside instance_mask & "
            "polar_valid; the sample cannot be measured"
        )
    quantiles = np.percentile(signed_values, QUANTILE_PERCENTILES, method="linear")
    values = (
        float(signed_values.mean()),
        float(signed_values.std(ddof=0)),
        *(float(q) for q in quantiles),
        float(abs_values.mean()),
    )
    return np.asarray(values, dtype=np.float32), True


# ---------------------------------------------------------------------------
# Slot assembly


def slot_masks(variant: str, g_dim: int) -> np.ndarray:
    """The boolean column mask of one variant over the [G, C, S] layout."""
    validate_variant(variant)
    g, c, s = slot_dimensions(g_dim)
    enabled = VARIANT_SLOTS[variant]
    return np.concatenate([
        np.full(g, enabled[0], dtype=bool),
        np.full(c, enabled[1], dtype=bool),
        np.full(s, enabled[2], dtype=bool),
    ])


def assemble_design_matrix(
    g_features: np.ndarray,
    c_features: np.ndarray,
    s_features: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Concatenate the standardized slots and zero the unused columns.

    The variant's slot mask is applied here unconditionally — a caller
    cannot forget it, so an unused slot always contributes exactly zeros
    (never a shifted mean). NaN is allowed only inside the S slot of
    samples without a valid S (those columns are zeroed by the
    standardizer before assembly; any surviving NaN/Inf is refused).
    """
    validate_variant(variant)
    g = np.asarray(g_features, dtype=np.float32)
    c = np.asarray(c_features, dtype=np.float32)
    s = np.asarray(s_features, dtype=np.float32)
    if g.ndim != 2 or c.ndim != 2 or s.ndim != 2:
        raise ValueError("slot features must be 2-D (n_samples, width)")
    if not (len(g) == len(c) == len(s)):
        raise ValueError("slot feature matrices disagree on the sample count")
    expected = slot_dimensions(g.shape[1])
    if (c.shape[1], s.shape[1]) != (expected[1], expected[2]):
        raise ValueError(
            f"slot widths {(c.shape[1], s.shape[1])} do not match the "
            f"feature definition {expected[1:]}"
        )
    matrix = np.concatenate([g, c, s], axis=1) * slot_masks(
        variant, g.shape[1]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "design matrix contains NaN/Inf after slot assembly; a missing S "
            "must be standardized to zero before assembly"
        )
    return matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# Standardizer (fit on inner_train only)


@dataclass(frozen=True)
class Standardizer:
    """Per-column standardization with deterministic edge rules.

    - ``mean``/``std`` are computed per column over the rows where that
      column's validity flag is true (for S columns: inner_train samples
      with a valid S only; all other columns: every inner_train sample);
    - ``std == 0`` or no valid row → the column transforms to exactly 0
      for every row (a deterministic, finite rule, recorded as
      ``usable=False``);
    - a row whose column flag is false (a missing S) transforms to 0 and
      is distinguished by slot C's explicit ``s_region_nonempty_flag``.
    """

    mean: np.ndarray
    std: np.ndarray
    usable: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray, column_valid: np.ndarray) -> "Standardizer":
        matrix = np.asarray(matrix, dtype=np.float64)
        column_valid = np.asarray(column_valid, dtype=bool)
        if matrix.ndim != 2 or column_valid.shape != matrix.shape:
            raise ValueError("fit needs the matrix and its validity flag map")
        if not np.isfinite(matrix[column_valid]).all():
            raise ValueError(
                "standardizer fit received NaN/Inf in a column position "
                "marked valid"
            )
        mean = np.zeros(matrix.shape[1], dtype=np.float64)
        std = np.zeros(matrix.shape[1], dtype=np.float64)
        usable = np.zeros(matrix.shape[1], dtype=bool)
        for column in range(matrix.shape[1]):
            rows = column_valid[:, column]
            if not rows.any():
                continue
            values = matrix[rows, column]
            column_mean = float(values.mean())
            column_std = float(values.std(ddof=0))
            mean[column] = column_mean
            std[column] = column_std
            usable[column] = column_std > 0.0 and np.isfinite(column_std)
        return cls(mean=mean, std=std, usable=usable)

    def transform(
        self, matrix: np.ndarray, column_valid: np.ndarray
    ) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        column_valid = np.asarray(column_valid, dtype=bool)
        if matrix.ndim != 2 or column_valid.shape != matrix.shape:
            raise ValueError("transform needs the matrix and its validity map")
        if self.mean.shape != (matrix.shape[1],):
            raise ValueError(
                "standardizer column count does not match the matrix; "
                "refusing a silently mismatched transform"
            )
        finite_checked = matrix[column_valid & self.usable[None, :]]
        if finite_checked.size and not np.isfinite(finite_checked).all():
            raise ValueError(
                "transform received NaN/Inf in a column position marked valid"
            )
        out = np.zeros(matrix.shape, dtype=np.float32)
        for column in range(matrix.shape[1]):
            if not self.usable[column]:
                continue  # zero-variance or empty column → exactly zero
            rows = column_valid[:, column]
            out[rows, column] = (
                (matrix[rows, column] - self.mean[column]) / self.std[column]
            )
        return out

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "usable": self.usable.tolist(),
            "rules": {
                "fit_rows": "inner_train only",
                "s_columns_valid_rows": "inner_train samples with a valid S",
                "missing_s_after_standardization": "zero",
                "zero_variance_or_empty_column": "output_zero",
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Standardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            usable=np.asarray(payload["usable"], dtype=bool),
        )


# ---------------------------------------------------------------------------
# Metrics and the frozen selection rule


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int],
                 num_classes: int = NUM_CLASSES) -> list[float]:
    """F1 per class over the FIXED class set (a missing class scores 0)."""
    f1s = []
    for class_id in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred)
                 if t == class_id and p == class_id)
        fp = sum(1 for t, p in zip(y_true, y_pred)
                 if t != class_id and p == class_id)
        fn = sum(1 for t, p in zip(y_true, y_pred)
                 if t == class_id and p != class_id)
        f1s.append(_prf(tp, fp, fn)[2])
    return f1s


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int],
             num_classes: int = NUM_CLASSES) -> float:
    return float(np.mean(per_class_f1(y_true, y_pred, num_classes)))


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not len(y_true):
        return 0.0
    return float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))


def select_epoch(history: Sequence[dict]) -> dict:
    """inner_val Macro-F1 first, accuracy second, earlier epoch on ties."""
    if not history:
        raise ValueError("empty selection history")
    best = history[0]
    for entry in history[1:]:
        candidate = (
            entry["inner_val_macro_f1"],
            entry["inner_val_accuracy"],
            -entry["epoch"],
        )
        incumbent = (
            best["inner_val_macro_f1"],
            best["inner_val_accuracy"],
            -best["epoch"],
        )
        if candidate > incumbent:
            best = entry
    return best


def paired_changes(
    base_correct: Sequence[bool], other_correct: Sequence[bool]
) -> dict:
    """Fixed/broke counts of ``other`` against ``base`` (paired samples)."""
    if len(base_correct) != len(other_correct):
        raise ValueError("paired change inputs must have equal length")
    fixed = sum(1 for b, o in zip(base_correct, other_correct)
                if o and not b)
    broke = sum(1 for b, o in zip(base_correct, other_correct)
                if b and not o)
    return {
        "fixed": fixed,
        "broke": broke,
        "both_correct": sum(1 for b, o in zip(base_correct, other_correct)
                            if b and o),
        "both_wrong": sum(1 for b, o in zip(base_correct, other_correct)
                          if not b and not o),
    }


# ---------------------------------------------------------------------------
# Module-state snapshot (the no-training evidence for frozen forwards)


def snapshot_module_state(module) -> dict[str, np.ndarray]:
    """``named_parameters`` + ``named_buffers`` as float arrays for compare."""
    state: dict[str, np.ndarray] = {}
    for name, tensor in module.named_parameters():
        state[f"param:{name}"] = (
            tensor.detach().to("cpu", copy=True).numpy()
        )
    for name, tensor in module.named_buffers():
        state[f"buffer:{name}"] = (
            tensor.detach().to("cpu", copy=True).numpy()
        )
    return state


def compare_module_states(before: dict, after: dict) -> dict:
    """Exact equality report of two module snapshots."""
    if set(before) != set(after):
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        return {
            "unchanged": False,
            "added_tensors": added[:20],
            "removed_tensors": removed[:20],
            "changed_tensors": [],
        }
    changed = sorted(
        name for name in before
        if before[name].shape != after[name].shape
        or not np.array_equal(before[name], after[name])
    )
    return {
        "unchanged": not changed,
        "added_tensors": [],
        "removed_tensors": [],
        "changed_tensors": changed[:20],
    }


# ---------------------------------------------------------------------------
# The linear probe head and its fixed training loop


def build_probe_head(input_dim: int, *, seed: int):
    """A fresh linear 4-class head; the seed fixes its initialization."""
    import torch
    from torch import nn

    if int(input_dim) <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim!r}")
    torch.manual_seed(int(seed))
    return nn.Linear(int(input_dim), NUM_CLASSES)


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
):
    """Full-batch AdamW training of the linear head (the frozen budget).

    Every epoch is recorded (loss, inner_val Macro-F1, inner_val
    accuracy); the returned ``best_state`` is the checkpoint of the
    selection rule evaluated ON THE FLY — strictly better on
    (Macro-F1, accuracy, earlier epoch) replaces the incumbent, so ties
    keep the earlier epoch. outer_eval is never passed here and never
    read.
    """
    import torch
    from torch.nn import functional as F

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("probe inputs must be 2-D design matrices")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("train/val input dimensions disagree")
    if len(y_train) != len(X_train) or len(y_val) != len(X_val):
        raise ValueError("feature/label counts disagree")
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "inner_train has fewer than two classes; a 4-class head cannot "
            "be fit meaningfully"
        )
    if int(max_epochs) <= 0:
        raise ValueError(f"max_epochs must be positive, got {max_epochs!r}")

    model = build_probe_head(X_train.shape[1], seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_list = [int(v) for v in y_val]

    history: list[dict] = []
    best_state = None
    best_key = None
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = F.cross_entropy(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).numpy()
        val_pred = [int(v) for v in val_logits.argmax(axis=1)]
        entry = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "inner_val_macro_f1": macro_f1(y_val_list, val_pred),
            "inner_val_accuracy": accuracy(y_val_list, val_pred),
        }
        history.append(entry)
        key = (entry["inner_val_macro_f1"], entry["inner_val_accuracy"],
               -entry["epoch"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())

    return model, history, best_state, select_epoch(history)


# ---------------------------------------------------------------------------
# Cache binding and admission (shared by feature preparation, cache upgrade
# and the training entry). All three must refuse a mismatched cache BEFORE
# any head is built, any scaler is fitted and any training artifact is
# written.

CACHE_BINDING_FILENAME = "cache_binding.json"
CACHE_BINDING_VERSION = "existing_data_probe_cache_binding_v2"
BOUND_CACHE_FILES = (
    "feature_cache.npz",
    "feature_manifest.json",
    "feature_definition.json",
)
HEX64_DIGITS = frozenset("0123456789abcdef")


def _require_hex64(value: str, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or not set(value) <= HEX64_DIGITS):
        raise ValueError(
            f"{field} must be a 64-char lowercase hex sha256, got {value!r}"
        )
    return value


def file_sha256(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cache_binding(feature_dir, protocol: dict) -> dict:
    """The frozen binding record for a finished feature cache.

    Covers the byte digests of the three cache files and binds them to the
    protocol digest, the V5 data fingerprint, the official encoder identity
    and the feature definition version. Written by feature preparation and
    by the cache upgrade; the training entry treats it as the EXPECTED
    state and re-hashes the actual files against it. The expected values
    are never taken from a fresh local computation at admission time.
    """
    import json as json_module
    from pathlib import Path

    feature_dir = Path(feature_dir)
    files = {}
    for name in BOUND_CACHE_FILES:
        path = feature_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"cache file missing: {path}")
        files[name] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    mit = (protocol.get("config", {}).get("baselines", {})
           .get("official_mit_b0", {}))
    definition = json_module.loads(
        (feature_dir / "feature_definition.json").read_text(encoding="utf-8"))
    binding = {
        "binding_version": CACHE_BINDING_VERSION,
        "files": files,
        "protocol_sha256": protocol["protocol_sha256"],
        "data_fingerprint": protocol["data"]["fingerprint"],
        "official_mit_b0": {
            "path": mit.get("path", ""),
            "sha256": mit.get("sha256", ""),
        },
        "feature_definition_version": definition.get(
            "feature_definition_version", ""),
        "class_names": list(protocol["data"]["class_names"]),
    }
    (feature_dir / CACHE_BINDING_FILENAME).write_text(
        json_module.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return binding


def verify_cache_binding(feature_dir, protocol: dict) -> dict:
    """Verify the cache binding against the ACTUAL bytes on disk.

    The binding record is the expected state (saved when the cache was
    finished); this re-hashes the three files and refuses any mismatch,
    any missing/empty record and any identity drift against the protocol.
    """
    import json as json_module
    from pathlib import Path

    feature_dir = Path(feature_dir)
    binding_path = feature_dir / CACHE_BINDING_FILENAME
    if not binding_path.is_file():
        raise FileNotFoundError(
            f"cache binding record missing: {binding_path}; a feature cache "
            "without a binding cannot be admitted for training"
        )
    binding = json_module.loads(binding_path.read_text(encoding="utf-8"))
    if binding.get("binding_version") != CACHE_BINDING_VERSION:
        raise ValueError(
            f"cache binding version {binding.get('binding_version')!r} != "
            f"{CACHE_BINDING_VERSION!r}; regenerate the binding"
        )
    if binding.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError(
            "cache binding protocol digest does not match this protocol"
        )
    if binding.get("data_fingerprint") != protocol["data"]["fingerprint"]:
        raise ValueError(
            "cache binding data fingerprint does not match this protocol"
        )
    mit = (protocol.get("config", {}).get("baselines", {})
           .get("official_mit_b0", {}))
    binding_mit = binding.get("official_mit_b0", {})
    if mit and (binding_mit.get("path") != mit.get("path")
                or binding_mit.get("sha256") != mit.get("sha256")):
        raise ValueError(
            "cache binding encoder identity does not match the protocol's "
            f"official MiT-B0 record (binding={binding_mit}, protocol={mit})"
        )
    if list(binding.get("class_names", [])) != list(
            protocol["data"]["class_names"]):
        raise ValueError(
            "cache binding class order does not match the protocol"
        )
    files = binding.get("files")
    if not isinstance(files, dict) or set(files) != set(BOUND_CACHE_FILES):
        raise ValueError(
            f"cache binding must cover exactly {sorted(BOUND_CACHE_FILES)}"
        )
    for name in BOUND_CACHE_FILES:
        record = files[name]
        if not isinstance(record, dict):
            raise ValueError(
                f"cache binding record for {name} is not a mapping"
            )
        _require_hex64(record.get("sha256", ""), f"binding files.{name}.sha256")
        if not isinstance(record.get("bytes"), int) or record["bytes"] < 0:
            raise ValueError(
                f"binding files.{name}.bytes must be a non-negative integer"
            )
        path = feature_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"bound cache file missing: {path}")
        actual_digest = file_sha256(path)
        if actual_digest != record["sha256"]:
            raise ValueError(
                f"{name} sha256 {actual_digest} does not match the binding "
                f"record {record['sha256']}; the cache changed after binding"
            )
        if path.stat().st_size != record["bytes"]:
            raise ValueError(
                f"{name} byte size {path.stat().st_size} does not match the "
                f"binding record {record['bytes']}"
            )
    return binding


def authoritative_sample_index(protocol: dict) -> dict[str, dict]:
    """The per-sample identity and per-fold role map from the protocol.

    Validates fold by fold (never by total counts alone):
    - every fold's three parts are group-disjoint and together cover the
      whole dev pool;
    - every sample appears exactly once per fold, with one role;
    - a sample's group belongs to that part's group set (no crossing);
    - every sample's original split is a dev split (no test member);
    - the identity fields are identical across folds;
    - every sample is the outer_eval sample of exactly one fold overall.
    """
    folds = protocol["folds"]["folds"]
    if not folds:
        raise ValueError("the protocol has no folds")
    dev_splits = set(protocol["data"]["dev_pool_splits"])
    all_groups = set(protocol.get("group_statistics", {}))
    index: dict[str, dict] = {}
    problems: list[str] = []
    for fold in folds:
        fold_id = fold["fold_id"]
        group_parts: dict[str, str] = {}
        seen_in_fold: dict[str, str] = {}
        for part in ("outer_eval", "inner_train", "inner_val"):
            groups_in_part = set(fold[f"{part}_groups"])
            for group in groups_in_part:
                if group in group_parts:
                    problems.append(
                        f"fold {fold_id}: group {group!r} appears in both "
                        f"{group_parts[group]} and {part}"
                    )
                group_parts[group] = part
            for sample in fold[f"{part}_members"]["samples"]:
                name = sample.get("sample_name")
                if name in seen_in_fold:
                    problems.append(
                        f"{name} appears twice in fold {fold_id} "
                        f"({seen_in_fold[name]} and {part})"
                    )
                seen_in_fold[name] = part
                try:
                    identity = {
                        "class_id": int(sample["class_id"]),
                        "class_name": sample["class_name"],
                        "group": sample["group"],
                        "source_frame": sample["source_frame"],
                        "object_index": int(sample["object_index"]),
                        "split": sample["split"],
                        "npz_path": sample["npz_path"],
                    }
                except (KeyError, TypeError, ValueError) as error:
                    problems.append(
                        f"{name}: malformed member record in fold {fold_id} "
                        f"{part} ({error})"
                    )
                    continue
                if identity["group"] not in groups_in_part:
                    problems.append(
                        f"{name}: group {identity['group']!r} is not among "
                        f"fold {fold_id} {part} groups"
                    )
                if identity["split"] not in dev_splits:
                    problems.append(
                        f"{name}: split {identity['split']!r} is not a dev "
                        "split (test member?)"
                    )
                entry = index.setdefault(name, {"identity": None, "roles": {}})
                if entry["identity"] is None:
                    entry["identity"] = identity
                elif entry["identity"] != identity:
                    problems.append(
                        f"{name}: identity differs between folds "
                        f"{sorted(entry['roles'])} and {fold_id}"
                    )
                if fold_id in entry["roles"]:
                    problems.append(
                        f"{name}: already has a role for fold {fold_id}"
                    )
                entry["roles"][fold_id] = part
        if all_groups and set(group_parts) != all_groups:
            problems.append(
                f"fold {fold_id}: its parts cover groups "
                f"{sorted(set(group_parts) ^ all_groups)} differently from "
                "the dev pool"
            )
        if not seen_in_fold:
            problems.append(f"fold {fold_id}: no members at all")
    fold_ids = {fold["fold_id"] for fold in folds}
    for name, entry in index.items():
        if set(entry["roles"]) != fold_ids:
            missing = sorted(fold_ids - set(entry["roles"]))
            extra = sorted(set(entry["roles"]) - fold_ids)
            problems.append(
                f"{name}: does not hold exactly one role per fold "
                f"(missing={missing}, unknown={extra})"
            )
        outer_evals = [fold for fold, role in entry["roles"].items()
                       if role == "outer_eval"]
        if len(outer_evals) != 1:
            problems.append(
                f"{name}: outer_eval held {len(outer_evals)} times "
                f"(folds {sorted(outer_evals)}), expected exactly once"
            )
    if problems:
        raise ValueError(
            "protocol fold membership failed its per-fold validation "
            f"({len(problems)} problems, first: {problems[:5]})"
        )
    return index


def verify_cache_manifest(manifest: dict, authority: dict,
                          protocol: dict) -> list[dict]:
    """Row-by-row identity check of the cache manifest against the protocol.

    The row order IS the cache row contract: manifest rows must be exactly
    the authoritative index sorted by sample_name. Every identity field and
    every per-fold role must match item by item; total-count equality alone
    is never accepted.
    """
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cache manifest carries no sample rows")
    expected_order = sorted(authority)
    actual_order = [row.get("sample_name") for row in samples]
    if actual_order != expected_order:
        if sorted(actual_order) != expected_order:
            extra = sorted(set(actual_order) - set(expected_order))[:5]
            missing = sorted(set(expected_order) - set(actual_order))[:5]
            raise ValueError(
                "cache manifest sample set differs from the protocol "
                f"(manifest-only={extra}, protocol-only={missing})"
            )
        raise ValueError(
            "cache manifest row order does not match the frozen row "
            "contract (sorted sample_name); re-prepare or re-bind the cache"
        )
    fold_ids = set(authority[next(iter(authority))]["roles"])
    known_roles = {"outer_eval", "inner_train", "inner_val"}
    for row in samples:
        name = row["sample_name"]
        expected = authority[name]["identity"]
        for field in ("class_id", "class_name", "group", "source_frame",
                      "object_index", "split", "npz_path"):
            actual = row.get(field)
            if field in ("class_id", "object_index"):
                if isinstance(actual, bool) or not isinstance(actual, int):
                    raise ValueError(
                        f"{name}: {field} must be an integer, got "
                        f"{type(actual).__name__}: {actual!r}"
                    )
            if actual != expected[field]:
                raise ValueError(
                    f"{name}: manifest {field} {actual!r} != protocol "
                    f"{expected[field]!r}"
                )
        roles = row.get("fold_roles")
        if not isinstance(roles, dict) or set(roles) != fold_ids:
            raise ValueError(
                f"{name}: fold_roles {roles!r} does not cover exactly the "
                f"protocol folds {sorted(fold_ids)}"
            )
        for fold_id, role in roles.items():
            if role not in known_roles:
                raise ValueError(
                    f"{name}: unknown role {role!r} for fold {fold_id}"
                )
            if role != authority[name]["roles"][fold_id]:
                raise ValueError(
                    f"{name}: fold {fold_id} role {role!r} != protocol role "
                    f"{authority[name]['roles'][fold_id]!r}"
                )
        if not isinstance(row.get("has_valid_s"), bool):
            raise ValueError(
                f"{name}: has_valid_s must be a strict boolean, got "
                f"{row.get('has_valid_s')!r}"
            )
    return samples


def verify_cache_arrays(cache: dict, manifest_samples: list,
                        definition: dict, protocol: dict) -> None:
    """Shape/dtype/content checks of the cache arrays against the manifest.

    G/C must be fully finite; valid-S rows must be finite; missing-S rows
    must be exactly NaN (the explicit-miss rule); the C non-empty flag must
    equal s_valid row by row; class_id must be an integer array matching
    the manifest and the protocol class order row by row.
    """
    expected_keys = {"g_features", "c_features", "s_features", "s_valid",
                     "class_id"}
    if set(cache) != expected_keys:
        raise ValueError(
            f"cache npz keys {sorted(cache)} != expected "
            f"{sorted(expected_keys)}"
        )
    n = len(manifest_samples)
    g = cache["g_features"]
    c = cache["c_features"]
    s = cache["s_features"]
    s_valid = cache["s_valid"]
    class_ids = cache["class_id"]
    slot_g = definition.get("slot_G", {})
    if g.dtype != np.float32 or g.ndim != 2 or g.shape[0] != n:
        raise ValueError(
            f"g_features dtype/shape {g.dtype}/{g.shape} does not match the "
            f"contract (float32, ({n}, G_dim))"
        )
    declared_g_dim = slot_g.get("dim")
    if declared_g_dim is not None and int(declared_g_dim) != int(g.shape[1]):
        raise ValueError(
            f"slot_G.dim {declared_g_dim} != actual G width {g.shape[1]}"
        )
    if c.dtype != np.float32 or c.shape != (n, len(CONTROL_FEATURE_KEYS)):
        raise ValueError(
            f"c_features dtype/shape {c.dtype}/{c.shape} does not match "
            f"(float32, ({n}, {len(CONTROL_FEATURE_KEYS)}))"
        )
    if s.dtype != np.float32 or s.shape != (n, len(POLAR_STAT_KEYS)):
        raise ValueError(
            f"s_features dtype/shape {s.dtype}/{s.shape} does not match "
            f"(float32, ({n}, {len(POLAR_STAT_KEYS)}))"
        )
    if s_valid.dtype != np.bool_ or s_valid.shape != (n,):
        raise ValueError(
            f"s_valid dtype/shape {s_valid.dtype}/{s_valid.shape} is not a "
            f"strict boolean ({n},) array"
        )
    if not np.issubdtype(class_ids.dtype, np.integer) or class_ids.shape != (n,):
        raise ValueError(
            f"class_id dtype/shape {class_ids.dtype}/{class_ids.shape} is "
            f"not an integer ({n},) array"
        )
    if class_ids.min() < 0 or class_ids.max() >= NUM_CLASSES:
        raise ValueError(
            f"class_id out of range [{class_ids.min()}, {class_ids.max()}]"
        )
    manifest_ids = np.asarray([row["class_id"] for row in manifest_samples],
                              dtype=np.int64)
    if not np.array_equal(class_ids.astype(np.int64), manifest_ids):
        raise ValueError(
            "cache class_id rows do not match the manifest/protocol classes"
        )
    if not np.isfinite(g).all():
        raise ValueError("g_features contains NaN/Inf")
    if not np.isfinite(c).all():
        raise ValueError("c_features contains NaN/Inf")
    valid = np.asarray(s_valid)
    if valid.any() and not np.isfinite(s[valid]).all():
        raise ValueError("s_features has NaN/Inf in rows marked valid")
    invalid = ~valid
    if invalid.any() and not np.isnan(s[invalid]).all():
        raise ValueError(
            "s_features rows without a valid S must be exactly NaN "
            "(the explicit-miss rule), got other values"
        )
    flag_column = c[:, CONTROL_FEATURE_KEYS.index("s_region_nonempty_flag")]
    if (not np.isin(flag_column, (0.0, 1.0)).all()
            or not np.array_equal(flag_column > 0.5, valid)):
        raise ValueError(
            "c_features s_region_nonempty_flag does not match s_valid row "
            "by row"
        )
    if list(definition.get("slot_C", {}).get("keys", [])) != list(
            CONTROL_FEATURE_KEYS):
        raise ValueError(
            "definition slot_C keys/order do not match the feature contract"
        )
    if list(definition.get("slot_S", {}).get("keys", [])) != list(
            POLAR_STAT_KEYS):
        raise ValueError(
            "definition slot_S keys/order do not match the feature contract"
        )
    features_config = protocol.get("config", {}).get("features", {})
    if (features_config.get("imgsz") is not None
            and int(definition.get("imgsz", -1)) != int(
                features_config["imgsz"])):
        raise ValueError(
            f"definition imgsz {definition.get('imgsz')} != protocol "
            f"{features_config['imgsz']}"
        )
    if (features_config.get("preprocessing_version")
            and definition.get("preprocessing_version")
            != features_config["preprocessing_version"]):
        raise ValueError(
            "definition preprocessing_version does not match the protocol"
        )
    definition_classes = definition.get("class_names")
    if (definition_classes is not None
            and list(definition_classes) != list(
                protocol["data"]["class_names"])):
        raise ValueError(
            "definition class order does not match the protocol"
        )


__all__ = [
    "CONTROL_FEATURE_KEYS",
    "FEATURE_DEFINITION_VERSION",
    "NUM_CLASSES",
    "POLAR_STAT_KEYS",
    "QUANTILE_METHOD",
    "QUANTILE_PERCENTILES",
    "SLOTS",
    "VARIANT_SLOTS",
    "Standardizer",
    "accuracy",
    "assemble_design_matrix",
    "build_control_features",
    "build_polar_stats",
    "build_probe_head",
    "compare_module_states",
    "macro_f1",
    "paired_changes",
    "per_class_f1",
    "select_epoch",
    "slot_dimensions",
    "slot_masks",
    "snapshot_module_state",
    "total_input_dim",
    "train_linear_probe",
    "validate_variant",
]
