"""YOLO26 gray training and fourth-batch spatial transfer, without project I/O.

Each gray backbone is trained on one inner_train/inner_val pair. Its own
raw logits provide the baseline and missing-measurement fallback; MiT
features and previously trained whole-dataset checkpoints are not used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.material_spatial_classifier import (
    classification_metrics,
    predict_with_fallback,
    select_best,
)
from models.polar_fusion import (
    GrayBackboneAdapter,
    architecture_name,
    prepare_gray_backbone,
)

Y_FEATURE_DIM = 1280
CONTEXT_DIM = Y_FEATURE_DIM + 14 + 16
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish", "real_fish")


def _noop() -> None:
    pass


def _epoch_noop(record: dict) -> None:
    pass


def _seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def state_digest(model: nn.Module) -> str:
    """Hash sorted state tensors, including names, dtype, shape and BN buffers."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(array.dtype), list(array.shape)],
                            separators=(",", ":")).encode("utf-8")
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _validate_gray(gray: np.ndarray) -> None:
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or \
            gray.ndim != 3 or gray.shape[1:] != (224, 224) or len(gray) == 0:
        raise ValueError("gray must be nonempty uint8(N,224,224)")


def gray_batch(gray_uint8_np: np.ndarray, ids: list[int] | np.ndarray,
               device: str) -> torch.Tensor:
    """Convert already INTER_AREA-resized gray images, preserving sample order."""
    _validate_gray(gray_uint8_np)
    gray = torch.from_numpy(np.ascontiguousarray(gray_uint8_np[ids])).to(
        device=device, dtype=torch.float32).div_(255.0)
    return gray.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def build_gray(base_path: str | Path, seed: int,
               device: str) -> GrayBackboneAdapter:
    """Load the local ImageNet YOLO26 classifier and replace only its head."""
    from ultralytics import YOLO

    base_path = Path(base_path)
    if not base_path.is_file():
        raise FileNotFoundError(f"local YOLO26 base missing: {base_path}")
    _seed(seed, device)
    probe = YOLO(str(base_path)).model
    if getattr(probe, "task", None) != "classify" or \
            not architecture_name(probe).startswith("yolo26"):
        raise ValueError("base must actually be a YOLO26 classification checkpoint")
    del probe
    model, _ = prepare_gray_backbone(base_path, "", list(CLASS_NAMES), device)
    model.float()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if model.module.model[-1].linear.in_features != Y_FEATURE_DIM:
        raise ValueError("YOLO26 head feature dimension is not 1280")
    return model


class Yolo26SpatialClassifier(nn.Module):
    """The same spatial CNN, with YOLO26 G1280 replacing MiT G256."""
    def __init__(self) -> None:
        super().__init__()
        layers = []
        for a, b in ((7, 16), (16, 32), (32, 64)):
            layers.extend((nn.Conv2d(a, b, 3, 2, 1, bias=False),
                           nn.GroupNorm(4, b), nn.SiLU()))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(CONTEXT_DIM + 64, 64),
                                  nn.SiLU(), nn.Linear(64, 4))

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        weights = F.interpolate(images[:, 6:7], features.shape[-2:], mode="area")
        mass = weights.sum(dim=(2, 3))
        if bool((mass <= 0).any()):
            raise ValueError("empty object pooling mask")
        pooled = (features * weights).sum(dim=(2, 3)) / mass
        return self.head(torch.cat((context, pooled), dim=1))


def _validate_labels(labels: np.ndarray, length: int) -> None:
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int64 or \
            labels.shape != (length,) or (labels < 0).any() or (labels > 3).any():
        raise ValueError("labels must be aligned int64 class ids in [0,3]")


def _validate_gray_data(data: dict) -> None:
    if set(data) != {"gray", "labels"}:
        raise ValueError("gray training accepts only gray and labels, no outer_eval")
    _validate_gray(data["gray"])
    _validate_labels(data["labels"], len(data["gray"]))


def _validate_spatial_data(data: dict) -> None:
    if set(data) != {"images", "context", "available", "p0", "labels"}:
        raise ValueError("spatial training accepts only the fixed train/val fields")
    n = len(data["images"])
    expected = (("images", (n, 7, 128, 128)), ("context", (n, CONTEXT_DIM)),
                ("p0", (n, 4)))
    for name, shape in expected:
        value = data[name]
        if not n or value.dtype != np.float32 or value.shape != shape or \
                not np.isfinite(value).all():
            raise ValueError(f"invalid {name} shape/dtype/values")
    if data["available"].dtype != bool or data["available"].shape != (n,):
        raise ValueError("availability must be aligned boolean")
    _validate_labels(data["labels"], n)


def _finite_gradients(parameters: list[nn.Parameter]) -> bool:
    values = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    return bool(torch.stack(values).all()) if values else False


def _gray_logits(model: nn.Module, gray: np.ndarray, device: str,
                 batch: int, check: Callable[[], None]) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(gray), batch):
            check()
            logits = model(gray_batch(gray, list(range(start, min(start + batch, len(gray)))), device))
            if logits.shape != (min(batch, len(gray) - start), 4) or \
                    not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite or wrongly shaped gray logits")
            out.append(logits.cpu().numpy())
    return np.concatenate(out).astype(np.float32, copy=False)


def fit_gray(train: dict, val: dict, *, base_path: str | Path,
             seed: int, device: str, epochs: int = 30, batch: int = 32,
             check: Callable[[], None] = _noop,
             epoch_callback: Callable[[dict], None] = _epoch_noop) -> tuple:
    """Train every gray parameter; fixed 30 epochs, inner_val-only selection."""
    _validate_gray_data(train)
    _validate_gray_data(val)
    if epochs < 1 or batch < 1:
        raise ValueError("epochs and batch must be positive")
    check()
    model = build_gray(base_path, seed, device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    covered = {id(p) for g in optimizer.param_groups for p in g["params"]}
    if covered != {id(p) for p in parameters} or not all(p.requires_grad for p in parameters):
        raise RuntimeError("gray optimizer must cover every unfrozen parameter")
    before = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    buffer_before = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
    early_name, early_parameter = next(iter(model.named_parameters()))
    rng = np.random.default_rng(seed)
    history, best_state, best_key = [], None, None
    early_gradient = False
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        check()
        model.train()
        loss_sum = 0.0
        order = rng.permutation(len(train["gray"]))
        for start in range(0, len(order), batch):
            check()
            ids = order[start:start + batch]
            x = gray_batch(train["gray"], ids, device)
            y = torch.from_numpy(train["labels"][ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite gray loss")
            loss.backward()
            if not _finite_gradients(parameters):
                raise ValueError("nonfinite gray gradient")
            if not early_gradient and early_parameter.grad is not None:
                early_gradient = bool((early_parameter.grad != 0).any())
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        logits = _gray_logits(model, val["gray"], device, batch, check)
        metrics = classification_metrics(val["labels"].tolist(), logits.argmax(1).tolist())
        record = {"epoch": epoch, **metrics, "loss": loss_sum / len(order),
                  "train_used": len(order), "elapsed_s": time.monotonic() - started}
        history.append(record)
        key = (metrics["macro_f1"], metrics["accuracy"], -epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_callback(record)
        check()
    trained_parameters = dict(model.named_parameters())
    changed = [k for k, v in before.items() if not torch.equal(v, trained_parameters[k].detach().cpu())]
    head_names = {k for k in before if ".linear." in k}
    buffers = dict(model.named_buffers())
    evidence = {
        "all_parameters_trainable": all(p.requires_grad for p in parameters),
        "optimizer_covers_all": True,
        "early_backbone_parameter": early_name,
        "early_backbone_gradient_nonzero": early_gradient,
        "early_backbone_changed": early_name in changed,
        "backbone_changed_count": sum(k not in head_names for k in changed),
        "head_changed": any(k in head_names for k in changed),
        "bn_buffers_changed_count": sum(not torch.equal(v, buffers[k].detach().cpu())
                                        for k, v in buffer_before.items()),
        "parameters_total": len(parameters),
        "optimizer": "AdamW", "lr": .001, "weight_decay": .01,
    }
    if not (early_gradient and evidence["early_backbone_changed"] and evidence["head_changed"]):
        raise RuntimeError("gray backbone and head must both actually learn")
    model.load_state_dict(best_state)
    model.eval()
    best = {**select_best(history), "training_evidence": evidence,
            "selected_state_sha256": state_digest(model)}
    return model, history, best


def extract_gray(model: GrayBackboneAdapter, gray: np.ndarray, *, device: str,
                 batch: int = 32, check: Callable[[], None] = _noop) -> tuple:
    """Freeze and cache G1280/logits, verifying linear reconstruction and BN state."""
    _validate_gray(gray)
    if batch < 1:
        raise ValueError("batch must be positive")
    model.to(device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.module.model[-1].linear
    if head.in_features != Y_FEATURE_DIM or head.out_features != 4:
        raise ValueError("gray head must be Linear(1280,4)")
    before = state_digest(model)
    captured, features, outputs = [], [], []
    hook = head.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    error = 0.0
    try:
        with torch.inference_mode():
            for start in range(0, len(gray), batch):
                check()
                captured.clear()
                logits = model(gray_batch(gray, list(range(start, min(start + batch, len(gray)))), device))
                if len(captured) != 1 or captured[0].shape != (len(logits), Y_FEATURE_DIM):
                    raise ValueError("wrong or repeated classification feature hook")
                g = captured[0]
                # F.linear avoids recursively triggering the registered hook.
                rebuilt = F.linear(g, head.weight, head.bias)
                if model.perm is not None:
                    rebuilt = rebuilt[:, model.perm]
                if logits.shape != (len(g), 4) or not bool(torch.isfinite(g).all()) or \
                        not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite gray features/logits")
                error = max(error, float((logits - rebuilt).abs().max()))
                features.append(g.cpu().numpy().copy())
                outputs.append(logits.cpu().numpy().copy())
    finally:
        hook.remove()
    after = state_digest(model)
    if after != before or error > 1e-4:
        raise RuntimeError("gray extraction changed state or failed logit reconstruction")
    evidence = {"state_before": before, "state_after": after,
                "state_identical": after == before, "feature_dim": Y_FEATURE_DIM,
                "logit_reconstruction_max_error": error,
                "parameters_frozen": all(not p.requires_grad for p in model.parameters()),
                "eval_mode": not model.training, "samples": len(gray)}
    return np.concatenate(features).astype(np.float32), np.concatenate(outputs).astype(np.float32), evidence


def fit_spatial(train: dict, val: dict, *, seed: int, device: str,
                epochs: int = 60, patience: int = 10, batch: int = 32,
                check: Callable[[], None] = _noop,
                epoch_callback: Callable[[dict], None] = _epoch_noop) -> tuple:
    """Fit the unrestricted spatial head; no fully missing rows enter the loss."""
    _validate_spatial_data(train)
    _validate_spatial_data(val)
    if min(epochs, patience, batch) < 1:
        raise ValueError("epochs, patience and batch must be positive")
    _seed(seed, device)
    model = Yolo26SpatialClassifier().to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.001)
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(train["available"])
    if not len(active):
        raise ValueError("no usable training samples")
    history, best_state, best_key, stale = [], None, None, 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        check()
        model.train()
        loss_sum = 0.0
        order = rng.permutation(active)
        for start in range(0, len(order), batch):
            check()
            ids = order[start:start + batch]
            x = torch.from_numpy(train["images"][ids]).to(device)
            c = torch.from_numpy(train["context"][ids]).to(device)
            y = torch.from_numpy(train["labels"][ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x, c), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite spatial loss")
            loss.backward()
            if not _finite_gradients(parameters):
                raise ValueError("nonfinite spatial gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        check()
        logits = predict_with_fallback(model, val["images"], val["context"],
                                      val["available"], val["p0"], device, batch)
        metrics = classification_metrics(val["labels"].tolist(), logits.argmax(1).tolist())
        record = {"epoch": epoch, **metrics, "loss": loss_sum / len(active),
                  "train_used": len(active), "elapsed_s": time.monotonic() - started}
        history.append(record)
        key = (metrics["macro_f1"], metrics["accuracy"], -epoch)
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
    model.eval()
    return model, history, select_best(history)
