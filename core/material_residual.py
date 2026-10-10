"""Fixed-P0 residual correction experiment (material_polar_residual_v1).

Implements the frozen protocol contract on top of the existing probe
infrastructure:

- the residual head is a 14 -> 4 linear layer WITHOUT bias, zero
  initialized; ``delta = valid_s * 0.5 * tanh(W x)`` so a valid-S sample's
  correction is bounded by 0.5 per logit and an invalid-S sample's
  correction is EXACTLY zero (precise P0 fallback);
- ``RC`` zeroes the S slot at the INPUT level (it cannot read polarization
  values); ``RCS`` reads the full 14-dimensional standardized C+S slot;
- the P0 logits enter training as a detached constant — no gradient can
  reach P0, and the frozen budget trains the residual head only;
- epoch 0 (no correction, final == P0) is always the first candidate and
  wins ties through the shared earlier-epoch rule;
- the loss is ``CE(final_logits, label) + aux_loss_weight * mean(delta^2)``.

Nothing here reads project data; every function operates on the values it
is handed. The training loop takes train/val tensors only — there is no
parameter through which outer_eval could enter fitting or selection.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
from torch import nn

from core.material_probe import (
    CONTROL_FEATURE_KEYS,
    POLAR_STAT_KEYS,
    Standardizer,
    accuracy,
    macro_f1,
)

RESIDUAL_INPUT_DIM = 14  # C (6) + S (8)
VARIANTS = ("RC", "RCS")
SHUFFLE_SEEDS = (2026, 2027, 2028, 2029, 2030)
DEFAULT_AUX_LOSS_WEIGHT = 0.01


def validate_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown residual variant {variant!r}; expected one of "
            f"{list(VARIANTS)}")
    return variant


class ResidualHead(nn.Module):
    """The residual correction module: a bias-free 14 -> 4 linear layer.

    ``forward`` exposes the raw pre-tanh projection; the protocol's delta
    and final logits are computed by :func:`compute_delta` /
    :func:`residual_logits` so the valid-S gating stays explicit.
    """

    def __init__(self, input_dim: int = RESIDUAL_INPUT_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.linear = nn.Linear(self.input_dim, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()

    def forward(self, x14):
        return self.linear(x14)


def build_residual_head(*, seed: int,
                        input_dim: int = RESIDUAL_INPUT_DIM) -> ResidualHead:
    """A fresh zero-initialized residual head (deterministic by seed).

    ``input_dim`` defaults to the frozen 14-wide C/S slot; the local
    interior/band variant passes its own width. Zero initialization, the
    missing bias and the seed determinism are identical for both.
    """
    import torch

    torch.manual_seed(int(seed))
    return ResidualHead(input_dim=input_dim)


def compute_delta(head: "nn.Module", x14: "torch.Tensor",
                  valid_s: "torch.Tensor") -> "torch.Tensor":
    """``delta = valid_s * 0.5 * tanh(W x)``; invalid-S rows are exactly 0."""
    import torch

    correction = 0.5 * torch.tanh(head.linear(x14))
    return valid_s.to(correction.dtype).unsqueeze(1) * correction


def residual_logits(head: "nn.Module", x14: "torch.Tensor",
                    valid_s: "torch.Tensor",
                    p0_logits: "torch.Tensor") -> "torch.Tensor":
    """``final_logits = P0_logits + delta`` (P0 stays a detached constant)."""
    import torch

    return p0_logits + compute_delta(head, x14, valid_s)


def assemble_residual_inputs(c_features: np.ndarray, s_features: np.ndarray,
                             s_valid: np.ndarray,
                             variant: str) -> np.ndarray:
    """Build the 14-dimensional residual input from the standardized
    C/S slots.

    ``RC`` zeroes the S slot unconditionally. For both variants, rows with
    ``s_valid=False`` get an exactly-zero S slot (the explicit missing
    rule — never a fabricated measurement). The standardization itself is
    the caller's responsibility (the fold's inner_train scaler).
    """
    validate_variant(variant)
    c = np.asarray(c_features, dtype=np.float32)
    s = np.asarray(s_features, dtype=np.float32)
    valid = np.asarray(s_valid).astype(bool)
    if c.ndim != 2 or c.shape[1] != len(CONTROL_FEATURE_KEYS):
        raise ValueError(
            f"C features must be (n, {len(CONTROL_FEATURE_KEYS)}), got "
            f"{c.shape}")
    if s.ndim != 2 or s.shape[1] != len(POLAR_STAT_KEYS):
        raise ValueError(
            f"S features must be (n, {len(POLAR_STAT_KEYS)}), got {s.shape}")
    if len(c) != len(s) or len(valid) != len(c):
        raise ValueError("C/S/s_valid row counts disagree")
    if not np.isfinite(c).all():
        raise ValueError("C features contain NaN/Inf")
    if np.isfinite(s[valid]).all() if valid.any() else True:
        pass
    else:
        raise ValueError("S features contain NaN/Inf in valid rows")
    s_input = np.zeros_like(s, dtype=np.float32)
    s_input[valid] = s[valid]
    if variant == "RC":
        s_input = np.zeros_like(s_input)
    return np.concatenate([c, s_input], axis=1).astype(np.float32)


def p0_head_from_payload(payload: dict) -> "nn.Module":
    """The frozen P0 linear head from a saved checkpoint payload.

    Every parameter is marked requires_grad=False and the module is put in
    eval mode; the residual training never updates it.
    """
    import torch
    from torch import nn

    state = payload["state_dict"]
    input_dim = int(payload["input_dim"])
    head = nn.Linear(input_dim, 4, bias=True)
    with torch.no_grad():
        head.weight.copy_(state["weight"])
        head.bias.copy_(state["bias"])
    head.requires_grad_(False)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def p0_logits_of(p0_head: "nn.Module", design: np.ndarray) -> np.ndarray:
    """Read-only P0 forward (eval + no_grad), float32."""
    import torch

    with torch.inference_mode():
        logits = p0_head(torch.from_numpy(
            np.ascontiguousarray(design, dtype=np.float32)))
    return logits.cpu().numpy().astype(np.float32)


def fit_residual_scaler(cache: dict, inner_rows: list[int]):
    """Fit the residual standardizer on the fold's inner_train rows only.

    The residual input is the 14-wide C/S slot: C columns use every
    training row, S columns only the rows with ``s_valid=true`` — the same
    rule the P0 scaler used for its C/S segment, so the result can be
    cross-checked against the saved P0 scaler by the entry. Zero variance
    or fully-invalid columns transform to exactly zero (Standardizer's
    deterministic rule). Returns ``(scaler, info)``.
    """
    c = cache["c_features"][inner_rows]
    s = cache["s_features"][inner_rows]
    s_valid = cache["s_valid"][inner_rows].astype(bool)
    width = c.shape[1] + s.shape[1]
    matrix = np.concatenate([c, s], axis=1).astype(np.float64)
    column_valid = np.ones((len(inner_rows), width), dtype=bool)
    column_valid[:, c.shape[1]:] = s_valid[:, None]
    scaler = Standardizer.fit(matrix, column_valid)
    info = {
        "inner_train_rows": len(inner_rows),
        "s_valid_rows": int(s_valid.sum()),
        "width": width,
        "note": ("fitted on inner_train only (C all rows, S valid rows "
                 "only); the entry cross-checks this against the saved P0 "
                 "scaler's C/S segment"),
    }
    return scaler, info


def train_residual_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    p0_train: np.ndarray,
    valid_s_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p0_val: np.ndarray,
    valid_s_val: np.ndarray,
    *,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    aux_loss_weight: float = DEFAULT_AUX_LOSS_WEIGHT,
    input_dim: int = RESIDUAL_INPUT_DIM,
):
    """The frozen full-batch training loop (train/val tensors only — there
    is no outer_eval parameter to misuse).

    Epoch 0 is the untrained head (W=0 => delta=0 => final == P0) and is
    recorded as the first candidate. The selection follows the shared
    rule: inner_val Macro-F1, then accuracy, then the earlier epoch.
    Returns ``(model, history, best_state, best)``.
    """
    import torch
    from torch.nn import functional as F

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    p0_train = np.asarray(p0_train, dtype=np.float32)
    p0_val = np.asarray(p0_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    valid_s_train = np.asarray(valid_s_train, dtype=np.float32)
    valid_s_val = np.asarray(valid_s_val, dtype=np.float32)
    if X_train.shape[1] != input_dim or X_val.shape[1] != input_dim:
        raise ValueError(
            f"residual inputs must have {input_dim} columns")
    if not (len(X_train) == len(y_train) == len(p0_train)
            == len(valid_s_train)):
        raise ValueError("train feature/label/P0/valid counts disagree")
    if not (len(X_val) == len(y_val) == len(p0_val) == len(valid_s_val)):
        raise ValueError("val feature/label/P0/valid counts disagree")
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "inner_train has fewer than two classes; a 4-class residual "
            "head cannot be fit meaningfully")
    max_epochs = int(max_epochs)
    if max_epochs <= 0:
        raise ValueError(f"max_epochs must be positive, got {max_epochs!r}")
    aux = float(aux_loss_weight)
    if aux < 0:
        raise ValueError(f"aux_loss_weight must be >= 0, got {aux!r}")

    model = build_residual_head(seed=seed, input_dim=input_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=float(weight_decay))
    x_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    p0_train_t = torch.from_numpy(p0_train)          # detached constant
    valid_train_t = torch.from_numpy(valid_s_train)
    x_val_t = torch.from_numpy(X_val)
    p0_val_t = torch.from_numpy(p0_val)
    valid_val_t = torch.from_numpy(valid_s_val)
    y_val_list = [int(v) for v in y_val]

    def _val_entry(epoch: int, train_loss: float | None) -> dict:
        with torch.no_grad():
            final = residual_logits(model, x_val_t, valid_val_t, p0_val_t)
            delta = compute_delta(model, x_val_t, valid_val_t)
        predictions = [int(v) for v in final.numpy().argmax(axis=1)]
        return {
            "epoch": epoch,
            "train_loss": train_loss,
            "inner_val_macro_f1": macro_f1(y_val_list, predictions),
            "inner_val_accuracy": accuracy(y_val_list, predictions),
            "delta_max_abs": float(delta.abs().max()),
        }

    history: list[dict] = [_val_entry(0, None)]
    best_state = copy.deepcopy(model.state_dict())
    from core.material_probe import select_epoch

    best = select_epoch(history)
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        final = residual_logits(model, x_train_t, valid_train_t, p0_train_t)
        delta = compute_delta(model, x_train_t, valid_train_t)
        loss = F.cross_entropy(final, y_train_t) + aux * delta.pow(2).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        entry = _val_entry(epoch, float(loss.item()))
        history.append(entry)
        from core.material_probe import select_epoch as _select

        best = _select(history)
        best_state = copy.deepcopy(model.state_dict()) \
            if best["epoch"] == epoch else best_state
    return model, history, best_state, best


# ---------------------------------------------------------------------------
# Polarization shuffle diagnostic (protocol section; donor matching is
# frozen before training and never reads labels or predictions)


def build_donor_bins(valid_ratio, gray_mean, train_rows=None,
                     edges=None) -> np.ndarray:
    """Tertile bins (3x3) over valid_ratio/gray_mean.

    The quantile edges come from ``edges`` ({"valid_ratio": [...],
    "gray_mean": [...]}) when given — the caller estimates them on the
    fold's full inner_train — otherwise from the ``train_rows`` subset.
    """
    valid_ratio = np.asarray(valid_ratio, dtype=np.float64)
    gray_mean = np.asarray(gray_mean, dtype=np.float64)
    if edges is not None:
        vr_edges = np.asarray(edges["valid_ratio"], dtype=np.float64)
        gm_edges = np.asarray(edges["gray_mean"], dtype=np.float64)
    else:
        if train_rows:
            vr_train = valid_ratio[list(train_rows)]
            gm_train = gray_mean[list(train_rows)]
        else:
            vr_train, gm_train = valid_ratio, gray_mean
        vr_edges = np.quantile(vr_train, [1 / 3, 2 / 3])
        gm_edges = np.quantile(gm_train, [1 / 3, 2 / 3])
    vr_bin = np.searchsorted(vr_edges, valid_ratio, side="right")
    gm_bin = np.searchsorted(gm_edges, gray_mean, side="right")
    return (vr_bin * 3 + gm_bin).astype(np.int64)


def shuffle_assignment(valid_receiver_rows: list[int], bins: np.ndarray,
                       split_labels: list[str], *,
                       seed: int) -> dict[int, int | None]:
    """Permutation over the VALID receivers of each (split, bin) group.

    Donor candidates come ONLY from the valid receiver rows themselves
    (an invalid row is never a donor and never crosses its split or bin).
    Each (split, bin) group with >= 2 valid members gets a bijection with
    no fixed point (seeded shuffle + cyclic shift); a single-member group
    is reported as unmatched (``None``). The mapping depends only on the
    frozen bins, the role rows, the validity flags, the stable row order
    and the seed — never on labels, predictions or outcomes.
    """
    import random

    rng = random.Random(int(seed))
    groups: dict[tuple[str, int], list[int]] = {}
    for row in valid_receiver_rows:
        groups.setdefault((split_labels[row], int(bins[row])),
                          []).append(row)
    assignment: dict[int, int | None] = {}
    for key in sorted(groups):
        members = sorted(groups[key])
        if len(members) < 2:
            assignment[members[0]] = None
            continue
        shuffled = members[:]
        rng.shuffle(shuffled)
        # rotate until NO receiver maps to itself (a rotation without a
        # fixed point always exists for n >= 2; verified by construction)
        for shift in range(1, len(members)):
            candidate = [shuffled[(i + shift) % len(members)]
                         for i in range(len(members))]
            if all(c != m for c, m in zip(candidate, members)):
                for receiver, donor in zip(members, candidate):
                    assignment[receiver] = donor
                break
        else:  # pragma: no cover - unreachable for distinct members
            for receiver, donor in zip(members, members[1:] + members[:1]):
                assignment[receiver] = donor
    return assignment


def run_shuffle_diagnostic(
    *,
    context: dict,
    c_standardized: np.ndarray,
    s_standardized: np.ndarray,
    s_valid: np.ndarray,
    evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    sample_names: list[str],
    seeds=SHUFFLE_SEEDS,
) -> dict:
    """Per-seed S-value shuffle over the valid receivers.

    ``c_standardized`` and ``s_standardized`` are the fold's STANDARDIZED
    C/S slots (exactly what the normal RCS evaluation assembles), so the
    receiver's C, the P0 logits and the missing state are untouched by
    the shuffle — only valid standardized S values are exchanged.
    ``evaluator(x14, valid_s)`` recomputes the final logits through the
    SAME frozen P0 head + selected residual head path as the normal
    evaluation.

    The normal (unshuffled) RCS logits are computed here through the same
    evaluator and stored per sample; an identity check re-runs the
    evaluator on the unshuffled input and requires an EXACT match (same
    computation path). Unmatched and invalid rows must reproduce the
    normal output exactly. Every evaluation row gets exactly one record
    per seed (valid, invalid and unmatched alike) carrying full identity,
    donor information and both logits/predictions.
    """
    c_std = np.asarray(c_standardized, dtype=np.float32)
    s_std = np.asarray(s_standardized, dtype=np.float32)
    s_valid = np.asarray(s_valid).astype(bool)
    n = len(s_std)
    if c_std.ndim != 2 or c_std.shape != (n, len(CONTROL_FEATURE_KEYS)):
        raise ValueError(
            f"c_standardized must be (n, {len(CONTROL_FEATURE_KEYS)}), "
            f"got {c_std.shape}")
    if s_std.ndim != 2 or s_std.shape != (n, len(POLAR_STAT_KEYS)):
        raise ValueError(
            f"s_standardized must be (n, {len(POLAR_STAT_KEYS)}), got "
            f"{s_std.shape}")
    if len(sample_names) != n:
        raise ValueError("sample_names must cover every evaluation row")
    if s_valid.any() and not np.isfinite(s_std[s_valid]).all():
        raise ValueError(
            "s_standardized contains NaN/Inf in rows marked valid; "
            "refusing to shuffle broken measurements")
    if not np.isfinite(c_std).all():
        raise ValueError("c_standardized contains NaN/Inf")

    split_labels = list(context["split"])
    groups = context.get("group")
    class_ids = context.get("class_id")
    bins = build_donor_bins(
        context["valid_ratio"], context["gray_mean"],
        train_rows=context.get("train_rows"),
        edges=context.get("quantile_edges"))

    def _assemble(s_matrix: np.ndarray) -> np.ndarray:
        return assemble_residual_inputs(c_std, s_matrix, s_valid, "RCS")

    normal_x14 = _assemble(s_std)
    normal_logits = np.asarray(
        evaluator(normal_x14, s_valid.astype(np.float32)),
        dtype=np.float32)
    if not np.isfinite(normal_logits).all():
        raise ValueError("the normal RCS forward produced non-finite logits")
    # identity check: the same evaluator on the same input, again
    identity_logits = np.asarray(
        evaluator(normal_x14, s_valid.astype(np.float32)),
        dtype=np.float32)
    if not np.array_equal(identity_logits, normal_logits):
        raise ValueError(
            "the diagnostic identity check failed: a second forward over "
            "the unshuffled input did not reproduce the normal RCS logits "
            "exactly")

    per_seed: dict[str, dict] = {}
    for seed in seeds:
        receivers = [row for row in range(n) if s_valid[row]]
        assignment = shuffle_assignment(receivers, bins, split_labels,
                                        seed=seed)
        shuffled = s_std.copy()
        records = []
        matched = 0
        same_group = 0
        for row in range(n):
            base = {
                "sample_name": sample_names[row],
                "group": groups[row] if groups is not None else None,
                "class_id": (int(class_ids[row])
                             if class_ids is not None else None),
                "receiver_row": row,
                "index_meaning": (
                    "0-based row index within this fold's outer_eval "
                    "rows (the frozen row contract order)"),
                "has_valid_s": bool(s_valid[row]),
            }
            if not s_valid[row]:
                records.append({
                    **base,
                    "status": "invalid_s",
                    "donor_row": None,
                    "donor_sample_name": None,
                    "note": ("invalid S: exact P0 fallback; the input is "
                             "unchanged by the shuffle"),
                })
                continue
            donor = assignment.get(row)
            if donor is None:
                records.append({
                    **base,
                    "status": "unmatched",
                    "donor_row": None,
                    "donor_sample_name": None,
                    "note": ("no legal donor in the same split/bin; the "
                             "receiver keeps its own standardized S and "
                             "its output equals the unshuffled RCS "
                             "forward"),
                })
                continue
            matched += 1
            shuffled[row] = s_std[donor]
            same = (groups is not None and groups[row] == groups[donor])
            same_group += int(bool(same))
            records.append({
                **base,
                "status": "matched",
                "donor_row": donor,
                "donor_sample_name": sample_names[donor],
                "donor_split": split_labels[donor],
                "same_group": bool(same),
                "note": ("the donor's standardized S replaces the "
                         "receiver's; C/P0/missing state untouched"),
            })
        shuffled_x14 = _assemble(shuffled)
        shuffled_logits = np.asarray(
            evaluator(shuffled_x14, s_valid.astype(np.float32)),
            dtype=np.float32)
        if not np.isfinite(shuffled_logits).all():
            raise ValueError(
                "the shuffled forward produced non-finite logits")
        for record in records:
            row = record["receiver_row"]
            record["normal_logits"] = normal_logits[row].tolist()
            record["normal_prediction"] = int(normal_logits[row].argmax())
            if record["status"] != "matched":
                # unmatched / invalid rows must reproduce the normal
                # output EXACTLY (same computation path)
                if not np.array_equal(shuffled_logits[row],
                                      normal_logits[row]):
                    raise ValueError(
                        f"row {row} ({record['status']}) must reproduce "
                        "the normal RCS output exactly after the shuffle")
            record["shuffled_logits"] = shuffled_logits[row].tolist()
            record["shuffled_prediction"] = int(
                shuffled_logits[row].argmax())
        per_seed[str(int(seed))] = {
            "seed": int(seed),
            "receivers": len(receivers),
            "matched": matched,
            "unmatched": len(receivers) - matched,
            "invalid_rows": int((~s_valid).sum()),
            "coverage": matched / len(receivers) if receivers else 0.0,
            "same_group_donor_ratio": (
                same_group / matched if matched else None),
            "samples": records,
            "normal_logits": normal_logits.tolist(),
        }
    return {
        "seeds": [int(s) for s in seeds],
        "per_seed": per_seed,
        "identity_check": {
            "exact": True,
            "note": ("a second evaluator forward over the unshuffled "
                     "input reproduced the normal RCS logits exactly"),
        },
        "note": ("only valid standardized S values are exchanged; the "
                 "receiver's C, P0 logits and missing state are "
                 "untouched; a shuffle drop is NOT, on its own, evidence "
                 "of a causal material contribution"),
    }


__all__ = [
    "DEFAULT_AUX_LOSS_WEIGHT",
    "RESIDUAL_INPUT_DIM",
    "SHUFFLE_SEEDS",
    "VARIANTS",
    "assemble_residual_inputs",
    "build_donor_bins",
    "build_residual_head",
    "compute_delta",
    "fit_residual_scaler",
    "p0_head_from_payload",
    "p0_logits_of",
    "residual_logits",
    "run_shuffle_diagnostic",
    "shuffle_assignment",
    "train_residual_head",
    "validate_variant",
]

