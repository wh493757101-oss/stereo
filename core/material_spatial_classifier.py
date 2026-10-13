"""Frozen fourth-batch model and training contract; no project I/O."""
from __future__ import annotations

import copy
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('B_stats', 'B_L', 'B_R', 'B_Q')
SEEDS = (2026, 2027, 2028)


def assemble_planes(grid: np.ndarray, variant: str) -> np.ndarray:
    """Select the seven channels; no reconstruction of q from averaged L/R."""
    if variant not in VARIANTS:
        raise ValueError('unknown spatial variant')
    if grid.ndim != 4 or grid.shape[1:] != (9, 128, 128):
        raise ValueError('grid must have shape (N,9,128,128)')
    if grid.dtype != np.float32 or not np.isfinite(grid).all():
        raise ValueError('grid must be finite float32')
    if (grid[:, 6:] < 0).any() or (grid[:, 6:] > 1).any():
        raise ValueError('coverage must lie in [0,1]')
    if (grid[:, 8].sum(axis=(1, 2)) <= 0).any():
        raise ValueError('instance coverage must be nonempty')
    out = np.zeros((len(grid), 7, 128, 128), np.float32)
    out[:, 4:] = grid[:, 6:]
    if variant != 'B_stats':
        out[:, :2] = grid[:, :2]
        source = {'B_L': 0, 'B_R': 2, 'B_Q': 4}[variant]
        out[:, 2:4] = grid[:, source:source + 2]
    return out


class SpatialClassifier(nn.Module):
    """Three convolution blocks and an unrestricted four-class head."""
    def __init__(self) -> None:
        super().__init__()
        layers = []
        for a, b in ((7, 16), (16, 32), (32, 64)):
            layers.extend((nn.Conv2d(a, b, 3, 2, 1, bias=False),
                           nn.GroupNorm(4, b), nn.SiLU()))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(350, 64), nn.SiLU(), nn.Linear(64, 4))

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        weights = F.interpolate(images[:, 6:7], features.shape[-2:], mode='area')
        mass = weights.sum(dim=(2, 3))
        if bool((mass <= 0).any()):
            raise ValueError('empty object pooling mask')
        pooled = (features * weights).sum(dim=(2, 3)) / mass
        return self.head(torch.cat((context, pooled), dim=1))


def classification_metrics(truth: list[int], predictions: list[int]) -> dict:
    if len(truth) != len(predictions) or not len(truth):
        raise ValueError('metrics require aligned nonempty identities')
    if any(type(x) not in (int, np.int64, np.int32) or not 0 <= x < 4
           for x in list(truth) + list(predictions)):
        raise ValueError('class ids must be strict integers in [0,3]')
    cm = np.zeros((4, 4), np.int64)
    np.add.at(cm, (truth, predictions), 1)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    denom = cm.sum(axis=0) + support
    f1 = np.divide(2 * tp, denom, out=np.zeros(4), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support > 0)
    return {'accuracy': float(tp.sum() / cm.sum()), 'macro_f1': float(f1.mean()),
            'recall': recall.tolist(), 'confusion_matrix': cm.tolist(),
            'support': int(cm.sum())}


def require_partition(rows: list[dict], fold: str) -> dict[str, list[int]]:
    parts = {r: [] for r in ('inner_train', 'inner_val', 'outer_eval')}
    group_roles = {}
    for i, row in enumerate(rows):
        role = row['fold_roles'][fold]
        if role not in parts:
            raise ValueError('unknown fold role')
        if row['group'] in group_roles and group_roles[row['group']] != role:
            raise ValueError('capture group leakage')
        group_roles[row['group']] = role
        parts[role].append(i)
    if any(not p for p in parts.values()):
        raise ValueError('empty fold role')
    return parts


def select_best(history: list[dict]) -> dict:
    if not history:
        raise ValueError('empty selection history')
    return max(history, key=lambda r: (r['macro_f1'], r['accuracy'], -r['epoch']))


def predict_with_fallback(model: nn.Module, images: np.ndarray,
                          context: np.ndarray, available: np.ndarray,
                          p0: np.ndarray, device: str, batch: int) -> np.ndarray:
    if available.dtype != bool or available.shape != (len(images),):
        raise ValueError('availability must be an aligned boolean vector')
    if p0.shape != (len(images), 4) or not np.isfinite(p0).all():
        raise ValueError('invalid P0 logits')
    out = p0.copy()
    valid = np.flatnonzero(available)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(valid), batch):
            ids = valid[start:start + batch]
            result = model(torch.from_numpy(images[ids]).to(device),
                           torch.from_numpy(context[ids]).to(device))
            if not bool(torch.isfinite(result).all()):
                raise ValueError('nonfinite model logits')
            out[ids] = result.cpu().numpy()
    return out


def fit_head(train: dict, val: dict, *, seed: int, device: str,
             epochs: int = 60, patience: int = 10, batch: int = 32,
             lr: float = .001, weight_decay: float = .001,
             check: Callable[[], None] = lambda: None,
             epoch_callback: Callable[[dict], None] = lambda _: None) -> tuple:
    """Only train/val are accepted. Missing rows never enter the loss."""
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    model = SpatialClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(train['available'])
    if not len(active):
        raise ValueError('no usable training samples')
    history, best_state, best_key, stale = [], None, None, 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        check()
        model.train()
        order = rng.permutation(active)
        loss_sum = 0.
        for start in range(0, len(order), batch):
            check()
            ids = order[start:start + batch]
            x = torch.from_numpy(train['images'][ids]).to(device)
            c = torch.from_numpy(train['context'][ids]).to(device)
            y = torch.from_numpy(train['labels'][ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x, c), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()):
                raise ValueError('nonfinite gradient')
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        logits = predict_with_fallback(model, val['images'], val['context'],
                                      val['available'], val['p0'], device, batch)
        metrics = classification_metrics(val['labels'].tolist(), logits.argmax(1).tolist())
        record = {'epoch': epoch, **metrics, 'loss': loss_sum / len(active),
                  'train_used': int(len(active)), 'elapsed_s': time.monotonic()-started}
        history.append(record)
        key = (metrics['macro_f1'], metrics['accuracy'], -epoch)
        if best_key is None or key > best_key:
            best_key, stale = key, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        epoch_callback(record)
        check()
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, select_best(history)
