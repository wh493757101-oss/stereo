"""Conditional residual correction: P0-output context (batch: conditional).

Question: does the polarization correction need GRAYSCALE context? The
context here comes ONLY from the frozen P0's own outputs — the four
softmax values and the top-two margin. These are model-output features,
never calibrated probabilities. The context is concatenated with the
frozen local C/S slots and passed through one small MLP:

    Linear(input_dim, 8) -> SiLU -> Linear(8, 4, bias=False)

with the LAST layer zero-initialized (epoch 0 = no correction) and the
first layer initialized from the fixed per-fit seed. Otherwise the
frozen residual protocol is unchanged: ``delta = valid_s * 0.5 *
tanh(head(input))``, ``final = P0_logits + delta``,
``CE + 0.01*mean(delta^2)``, full-batch AdamW, epoch-0 candidate,
inner_val selection only, missing S falls back to P0 exactly, and the
RC (S slot zeroed) / RCS (S active) contrast is preserved. No GT
routing, no unfreezing, no architecture or hyperparameter search.
"""

from __future__ import annotations

import copy

import numpy as np

CONTEXT_DIM = 5
CONTEXT_KEYS = (
    "p0_softmax_metal_submarine",
    "p0_softmax_plastic_submarine",
    "p0_softmax_plastic_fish",
    "p0_softmax_real_fish",
    "p0_top_margin",
)
CONTEXT_NOTE = (
    "context comes ONLY from the frozen P0's own output logits: the four "
    "softmax values and the max-minus-second-max margin. These are "
    "model-output features, not calibrated probabilities; they are not "
    "standardized and never read ground truth.")

DEFAULT_HIDDEN = 8
DEFAULT_AUX_LOSS_WEIGHT = 0.01


def p0_context(p0_logits: np.ndarray) -> np.ndarray:
    """The (n, 5) context: four softmax values + top-two margin.

    Deterministic numpy softmax (max-subtracted) over the VERIFIED P0
    logits; identical logits always produce identical context.
    """
    logits = np.asarray(p0_logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError(
            f"P0 logits must be (n, 4), got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("P0 logits contain NaN/Inf")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    sorted_desc = np.sort(softmax, axis=1)
    margin = sorted_desc[:, -1] - sorted_desc[:, -2]
    context = np.concatenate(
        [softmax, margin[:, None]], axis=1).astype(np.float32)
    return context


def conditional_input_dim(c_width: int, s_width: int) -> int:
    return int(c_width) + int(s_width) + CONTEXT_DIM


def build_conditional_head(*, seed: int, input_dim: int,
                           hidden: int = DEFAULT_HIDDEN):
    """Linear(input_dim, hidden) -> SiLU -> Linear(hidden, 4, bias=False).

    The last layer is zero-initialized (epoch 0 = no correction); the
    first layer keeps the seeded default initialization.
    """
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    head = nn.Sequential(
        nn.Linear(int(input_dim), int(hidden)),
        nn.SiLU(),
        nn.Linear(int(hidden), 4, bias=False),
    )
    with torch.no_grad():
        head[2].weight.zero_()
    return head


def _check_magnitude(magnitude: float) -> float:
    """Validate the correction magnitude ``a_f`` (finite and > 0).

    The frozen production value is 0.5; ``material_residual_scale_v1``
    passes ``0.5 * s_f`` for its scaled arm. Existing callers that omit
    the parameter keep the bitwise-original behavior.
    """
    value = float(magnitude)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"correction magnitude must be finite and > 0, got {magnitude!r}")
    return value


def conditional_delta(head, x, valid_s, magnitude: float = 0.5):
    """``delta = valid_s * a_f * tanh(head(x))``; invalid rows are 0.

    ``a_f`` defaults to the frozen production magnitude 0.5 so existing
    callers are unchanged; the scale experiment passes its arm magnitude.
    """
    import torch

    value = _check_magnitude(magnitude)
    correction = value * torch.tanh(head(x))
    return valid_s.to(correction.dtype).unsqueeze(1) * correction


def conditional_logits(head, x, valid_s, p0_logits, magnitude: float = 0.5):
    """``final = P0_logits + delta`` (P0 stays a detached constant)."""
    import torch

    return p0_logits + conditional_delta(head, x, valid_s, magnitude)


def assemble_conditional_inputs(c_features: np.ndarray,
                                s_features: np.ndarray,
                                context: np.ndarray,
                                s_valid: np.ndarray,
                                variant: str) -> np.ndarray:
    """The conditional residual input: [C, S (RC: zeroed), context].

    Rows with ``s_valid=False`` get an exactly-zero S part under BOTH
    variants; the context (P0-derived) and C stay — the explicit
    ``valid_s`` gate makes the correction itself fall back to P0 exactly.
    """
    from core.material_residual import validate_variant

    variant = validate_variant(variant)
    c = np.asarray(c_features, dtype=np.float32)
    s = np.asarray(s_features, dtype=np.float32)
    context = np.asarray(context, dtype=np.float32)
    valid = np.asarray(s_valid).astype(bool)
    if c.ndim != 2 or s.ndim != 2 or context.ndim != 2:
        raise ValueError("C/S/context must be 2-D")
    if len(c) != len(s) or len(valid) != len(c) or len(context) != len(c):
        raise ValueError("C/S/context/s_valid row counts disagree")
    if context.shape[1] != CONTEXT_DIM:
        raise ValueError(
            f"context width must be {CONTEXT_DIM}, got {context.shape[1]}")
    if not np.isfinite(c).all() or not np.isfinite(context).all():
        raise ValueError("C features or context contain NaN/Inf")
    if valid.any() and not np.isfinite(s[valid]).all():
        raise ValueError(
            "S features contain NaN/Inf in rows marked valid; the "
            "standardizer must zero empty-part cells before assembly")
    s_input = np.zeros_like(s, dtype=np.float32)
    s_input[valid] = s[valid]
    if variant == "RC":
        s_input = np.zeros_like(s_input)
    return np.concatenate([c, s_input, context], axis=1).astype(np.float32)


def train_conditional_residual_head(
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
    input_dim: int,
    hidden: int = DEFAULT_HIDDEN,
    magnitude: float = 0.5,
):
    """The frozen full-batch loop with the conditional head.

    Identical contract to ``core.material_residual.train_residual_head``:
    train/val tensors only (there is no outer_eval parameter), epoch 0 is
    the untrained candidate, selection is inner_val Macro-F1 > accuracy >
    earlier epoch, and the loss is ``CE + aux * mean(delta^2)``. The
    correction magnitude ``a_f`` defaults to the frozen 0.5; the scale
    experiment passes its arm magnitude (the ONLY difference between the
    two arms — identical head, seed, init, data and budget).
    """
    import torch
    from torch.nn import functional as F

    from core.material_probe import accuracy, macro_f1, select_epoch

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
            f"conditional residual inputs must have {input_dim} columns")
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

    model = build_conditional_head(seed=seed, input_dim=input_dim,
                                   hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=float(weight_decay))
    x_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    p0_train_t = torch.from_numpy(p0_train)  # detached constant
    valid_train_t = torch.from_numpy(valid_s_train)
    x_val_t = torch.from_numpy(X_val)
    p0_val_t = torch.from_numpy(p0_val)
    valid_val_t = torch.from_numpy(valid_s_val)
    y_val_list = [int(v) for v in y_val]

    def _val_entry(epoch: int, train_loss: float | None) -> dict:
        with torch.no_grad():
            final = conditional_logits(model, x_val_t, valid_val_t,
                                       p0_val_t, magnitude)
            delta = conditional_delta(model, x_val_t, valid_val_t, magnitude)
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
    best = select_epoch(history)
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        final = conditional_logits(model, x_train_t, valid_train_t,
                                   p0_train_t, magnitude)
        delta = conditional_delta(model, x_train_t, valid_train_t, magnitude)
        loss = F.cross_entropy(final, y_train_t) + aux * delta.pow(2).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        entry = _val_entry(epoch, float(loss.item()))
        history.append(entry)
        best = select_epoch(history)
        best_state = copy.deepcopy(model.state_dict()) \
            if best["epoch"] == epoch else best_state
    return model, history, best_state, best
