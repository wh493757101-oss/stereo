"""Bounded real-GPU verification of the YOLO26 Gray -> checkpoint -> Fusion
freeze -> save/reload -> weight-continuation chain.

This entry runs the production training paths with hard batch truncation
(2 epochs, at most 3 train + 3 val batches per epoch). It is a smoke
verification only: the truncated metrics prove the chain executes, never
that a model is good, and must not be used as acceptance evidence.

Phases (all outputs kept under runs/train/<run-id>* for review):

A. Gray: scripts.train_gray_fusion with the real YOLO26 classification
   base; checks finite loss/gradients, parameter updates, class order and
   save/reload logits.
B. Fusion freeze: scripts.train_polar_fusion --phase freeze with the smoke
   gray checkpoint as --gray-weights; checks frozen gray parameters and
   BatchNorm buffers, delta updates, gate gradients (once delta is
   non-zero), strict invalid-polar fallback, and save/reload logits via
   the actual fusion rebuild path.
C. Weight continuation: --init-from the smoke fusion checkpoint, freeze
   phase, imgsz inherited; parameters verified identical to the checkpoint
   before the first optimizer update, then 1 train + 1 val batch. A
   conflicting explicit --imgsz must be refused without creating outputs.
   This is weight continuation, not full optimizer/RNG resume.

Usage:
    python scripts/verify_yolo26_fusion.py --run-id verify_... --device 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.train_gray_fusion as tgf
import scripts.train_polar_fusion as tpf
from core.fusion_dataset import file_digest
from models.polar_fusion import (
    FusionClsDataset,
    architecture_name,
    class_names_from_manifest,
    load_fusion_checkpoint,
    prepare_gray_backbone,
    read_fusion_metadata,
    read_manifest_split,
    rebuild_gray_backbone,
)
from scripts.training_common import DeviceUnavailableError, InvalidRunIdError, validate_run_id

DEFAULT_BASE = "yolo26n-cls.pt"
DEFAULT_DATA = "datasets/underwater_cls_fusion_v4_band"
SMOKE_EPOCHS = 2
SMOKE_BATCHES = 3
LOGITS_TOLERANCE = 1e-5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="Smoke run id (fresh).")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args(argv)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _gpu_peak_bytes(device: str) -> int:
    return int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0


def _all_finite(module: nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in module.state_dict().values())


def _fixed_inputs(device: str, imgsz: int):
    generator = torch.Generator(device="cpu").manual_seed(0)
    gray = torch.rand(2, 3, imgsz, imgsz, generator=generator).to(device)
    polar = torch.rand(2, 3, imgsz, imgsz, generator=generator).to(device)
    quality = torch.tensor(
        [[0.5, 0.6, 0.7, 0.3], [0.4, 0.5, 0.6, 0.2]], device=device
    )
    return gray, polar, quality


def _gray_logits(backbone, gray: torch.Tensor) -> torch.Tensor:
    backbone.eval()
    with torch.no_grad():
        return backbone(gray)


def _head_parameter_names(module: nn.Module) -> set[str]:
    """Parameter names of the classification head, located structurally at
    ``module.model[-1]`` (the module ``_replace_cls_head`` swaps), never by
    name-pattern guessing."""
    head = module.model[-1]
    for prefix, submodule in module.named_modules():
        if submodule is head:
            base = f"{prefix}." if prefix else ""
            return {f"{base}{name}" for name, _ in head.named_parameters()}
    raise RuntimeError("classification head not found in the model structure")


def _early_backbone_parameter(module: nn.Module, head_names: set[str]) -> str:
    """First non-head parameter in module order (early feature extractor)."""
    for name, _ in module.named_parameters():
        if name not in head_names:
            return name
    raise RuntimeError("no non-head (backbone) parameter found")


def _update_report(before: dict, after: dict, head_names: set[str]) -> dict:
    """Parameter/buffer update counts between two snapshots.

    Parameters and buffers are tracked separately: BatchNorm running
    statistics changing is NOT a parameter update. The comparison is
    snapshot-to-snapshot (initialized model vs trained model), never
    against the official base checkpoint, so initialization, dtype
    conversion and head replacement cannot create false differences.
    """
    before_params, after_params = before["params"], after["params"]
    before_buffers, after_buffers = before["buffers"], after["buffers"]
    changed = [
        name
        for name in before_params
        if not torch.equal(before_params[name], after_params[name])
    ]
    backbone_changed = [name for name in changed if name not in head_names]
    buffers_changed = [
        name
        for name in before_buffers
        if not torch.equal(before_buffers[name], after_buffers[name])
    ]
    return {
        "params_total": len(before_params),
        "params_changed": changed,
        "params_changed_count": len(changed),
        "head_params_total": len(head_names),
        "head_params_changed_count": len([n for n in changed if n in head_names]),
        "backbone_params_total": len(before_params) - len(head_names),
        "backbone_params_changed": backbone_changed,
        "backbone_params_changed_count": len(backbone_changed),
        "backbone_params_changed_examples": backbone_changed[:5],
        "buffers_total": len(before_buffers),
        "buffers_changed": buffers_changed,
        "buffers_changed_count": len(buffers_changed),
        "buffers_changed_examples": buffers_changed[:5],
    }


def _assert_gray_learning(report: dict, early_name: str, grad_observations: dict) -> dict:
    """Reject head-only or buffer-only "training".

    Requires a real non-head parameter update including the early feature
    layer, and finite non-zero gradients for it, observed in the actual
    training path. Buffers changing never substitutes for parameter
    updates.
    """
    if report["params_changed_count"] < 1:
        raise RuntimeError("no parameter changed during gray training")
    if report["backbone_params_changed_count"] < 1:
        raise RuntimeError(
            "no non-head (backbone) parameter changed during gray training; "
            "the run may have trained only the classification head"
        )
    if early_name not in report["backbone_params_changed"]:
        raise RuntimeError(f"early backbone parameter {early_name} did not change")
    if not all(obs["finite"] for obs in grad_observations.values()):
        raise RuntimeError("non-finite gradients observed during gray training")
    early_obs = grad_observations.get(early_name)
    if early_obs is None or not early_obs["finite"] or early_obs["absmax"] <= 0.0:
        raise RuntimeError(
            f"early backbone parameter {early_name} received no finite "
            "non-zero gradient"
        )
    return {
        "early_backbone_changed": True,
        "backbone_params_changed_count": report["backbone_params_changed_count"],
        "backbone_params_changed_examples": report["backbone_params_changed_examples"],
    }


def _check_gray_frozen(gray_module: nn.Module, trained_state: dict) -> dict:
    """Frozen gray branch check: every parameter AND buffer (BatchNorm
    running statistics included) must be unchanged in the trained fusion
    checkpoint."""
    params = dict(gray_module.named_parameters())
    buffers = dict(gray_module.named_buffers())
    for name, value in params.items():
        key = f"gray_backbone.module.{name}"
        if key not in trained_state:
            raise RuntimeError(f"trained fusion checkpoint lacks {key}")
        if not torch.equal(value.cpu(), trained_state[key].cpu()):
            raise RuntimeError(f"frozen gray parameter drifted during freeze: {name}")
    for name, value in buffers.items():
        key = f"gray_backbone.module.{name}"
        if key not in trained_state:
            raise RuntimeError(f"trained fusion checkpoint lacks {key}")
        if not torch.equal(value.cpu(), trained_state[key].cpu()):
            raise RuntimeError(f"frozen gray buffer drifted during freeze: {name}")
    return {
        "gray_params_unchanged": len(params),
        "gray_buffers_unchanged": len(buffers),
    }


def _delta_last_layer_names(model: nn.Module) -> set[str]:
    """Names of delta_net's final (zero-initialized) layer parameters,
    located structurally."""
    last = model.delta_net.net[-1]
    for prefix, submodule in model.named_modules():
        if submodule is last:
            return {f"{prefix}.{name}" for name, _ in last.named_parameters()}
    raise RuntimeError("delta_net final layer not found in the model structure")


def _delta_update_report(before: dict, after: dict, last_names: set[str]) -> dict:
    """Delta parameter updates by before/after comparison.

    delta_net's conv layers have non-zero initialization, so a non-zero
    weight value alone proves nothing; only an actual before/after
    difference counts as an update. The zero-initialized final layer is
    reported separately.
    """
    changed = [
        name
        for name in before
        if name in after and not torch.equal(before[name].cpu(), after[name].cpu())
    ]
    return {
        "delta_params_total": len(before),
        "delta_params_changed": changed,
        "delta_params_changed_count": len(changed),
        "delta_params_changed_examples": changed[:5],
        "delta_zero_init_last_layer_before_all_zero": all(
            bool((before[name] == 0).all()) for name in last_names
        ),
        "delta_last_layer_changed": any(name in changed for name in last_names),
    }


def _assert_delta_updated(report: dict) -> None:
    if report["delta_params_changed_count"] < 1:
        raise RuntimeError(
            "no delta parameter changed during freeze; non-zero conv "
            "initialization is not evidence of an update"
        )


def _valid_polar_paths(data_root: Path, split: str, batch: int) -> list[Path]:
    """Real split samples whose recorded valid_ratio is positive."""
    paths, _, _ = read_manifest_split(data_root, split)
    import csv

    with (data_root / "dataset_manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_path = {row["npz_path"]: row for row in rows if row["split"] == split}
    selected = [
        path
        for path in paths
        if float(by_path[path.relative_to(data_root).as_posix()]["quality_valid_ratio"]) > 0.0
    ]
    if not selected:
        raise RuntimeError(f"no {split} samples with valid polar pixels found")
    return selected[:batch]


def phase_a_gray(args: argparse.Namespace, device: str, data_root: Path) -> dict:
    """Real YOLO26 gray-only training (truncated) with full-model unfreeze
    and in-path backbone learning evidence."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    run_id = f"{args.run_id}_gray"
    captured: dict[str, dict] = {}
    grad_observations: dict[str, dict] = {}
    hook_handles: list = []

    def save_hook(module: nn.Module, path: Path, info: dict) -> None:
        captured[Path(path).name] = {
            key: value.detach().clone() for key, value in module.state_dict().items()
        }

    def init_hook(model: nn.Module) -> None:
        captured["before"] = {
            "params": {
                name: value.detach().cpu().clone()
                for name, value in model.named_parameters()
            },
            "buffers": {
                name: value.detach().cpu().clone()
                for name, value in model.named_buffers()
            },
        }
        captured["params_total"] = len(captured["before"]["params"])
        captured["params_trainable"] = sum(
            1 for param in model.parameters() if param.requires_grad
        )
        # Gradient observation in the actual training path: record the
        # first backward's gradient per parameter. The hook returns None,
        # so gradients, training modes and BN state are untouched.
        for name, param in model.named_parameters():
            def hook(grad, _name=name):
                if _name not in grad_observations:
                    grad_observations[_name] = {
                        "finite": bool(torch.isfinite(grad).all()),
                        "absmax": float(grad.detach().abs().max()),
                    }
                return None

            hook_handles.append(param.register_hook(hook))

    gray_args = tgf.parse_args(
        [
            "--run-id", run_id,
            "--base", str(args.base),
            "--data", str(data_root),
            "--device", args.device,
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--epochs", str(SMOKE_EPOCHS),
            "--limit-batches", str(SMOKE_BATCHES),
        ]
    )
    try:
        run_dir = tgf.run_training(gray_args, save_hook=save_hook, init_hook=init_hook)
    finally:
        for handle in hook_handles:
            handle.remove()
    _sync(device)
    result: dict = {
        "run_dir": str(run_dir),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "peak_gpu_bytes": _gpu_peak_bytes(device),
    }

    for name in ("best.pt", "last.pt", "train_config.json", "metrics.json"):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"gray smoke output missing: {run_dir / name}")
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if config["architecture"] != "yolo26n-cls":
        raise RuntimeError(f"gray base architecture is {config['architecture']!r}")
    if not config["smoke"] or config["limit_batches"] != SMOKE_BATCHES:
        raise RuntimeError("gray smoke run was not recorded as truncated")
    losses = [float(entry["train_loss"]) for entry in metrics]
    if len(losses) != SMOKE_EPOCHS or not all(np.isfinite(losses)):
        raise RuntimeError(f"non-finite or missing smoke losses: {losses}")
    if any(entry["train_batches"] > SMOKE_BATCHES for entry in metrics):
        raise RuntimeError("train batch truncation was not applied")
    result["train_losses"] = losses
    result["train_batches"] = [entry["train_batches"] for entry in metrics]
    result["val_batches"] = [entry["val_batches"] for entry in metrics]

    # Full-model unfreeze: every parameter must be trainable (the official
    # base loads fully frozen; a head-only run would report 2).
    if captured["params_trainable"] != captured["params_total"]:
        raise RuntimeError(
            f"only {captured['params_trainable']}/{captured['params_total']} gray "
            "parameters were trainable; the whole model must be unfrozen"
        )
    result["params_total"] = captured["params_total"]
    result["params_trainable"] = captured["params_trainable"]

    class_names = class_names_from_manifest(data_root)
    # Save/reload fidelity: state captured at save time vs the file.
    backbone_captured, _ = prepare_gray_backbone(args.base, "", class_names, device)
    backbone_captured.module.load_state_dict(captured["last.pt"], strict=True)
    backbone_reloaded, reload_info = prepare_gray_backbone(
        args.base, str(run_dir / "last.pt"), class_names, device
    )
    if reload_info["gray_class_names"] != class_names:
        raise RuntimeError("reloaded class names differ from the dataset order")
    gray_fixed, _, _ = _fixed_inputs(device, args.imgsz)
    logits_captured = _gray_logits(backbone_captured, gray_fixed)
    logits_reloaded = _gray_logits(backbone_reloaded, gray_fixed)
    if logits_reloaded.shape != (2, len(class_names)):
        raise RuntimeError(f"reloaded logits shape {tuple(logits_reloaded.shape)}")
    max_diff = float((logits_captured - logits_reloaded).abs().max())
    if max_diff > LOGITS_TOLERANCE:
        raise RuntimeError(f"gray save/reload logits differ by {max_diff}")
    result["save_reload_logits_max_diff"] = max_diff

    trained_module = backbone_reloaded.module
    if not _all_finite(trained_module):
        raise RuntimeError("gray checkpoint contains non-finite parameters")

    # Real backbone learning: compare the pre-update snapshot (taken in the
    # actual training path) with the trained model. Parameters and buffers
    # are tracked separately; BN statistics changing is not a parameter
    # update.
    head_names = _head_parameter_names(trained_module)
    early_name = _early_backbone_parameter(trained_module, head_names)
    report = _update_report(
        captured["before"],
        {
            "params": {
                name: value.detach().cpu()
                for name, value in trained_module.named_parameters()
            },
            "buffers": {
                name: value.detach().cpu()
                for name, value in trained_module.named_buffers()
            },
        },
        head_names,
    )
    evidence = _assert_gray_learning(report, early_name, grad_observations)
    result["parameter_updates"] = report
    result["gradient_observations"] = {
        "observed_tensors": len(grad_observations),
        "all_finite": all(obs["finite"] for obs in grad_observations.values()),
        "early_parameter": early_name,
        "early_grad_absmax": float(grad_observations[early_name]["absmax"]),
        "backbone_nonzero_grad_count": sum(
            1
            for name, obs in grad_observations.items()
            if name not in head_names and obs["absmax"] > 0.0
        ),
    }
    result["early_backbone_changed"] = evidence["early_backbone_changed"]
    return result


def phase_b_freeze(args: argparse.Namespace, device: str, data_root: Path, gray_best: Path) -> dict:
    """Fusion freeze with the smoke gray checkpoint (truncated)."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    run_id = f"{args.run_id}_freeze"
    captured: dict[str, dict] = {}
    original_save = tpf.save_fusion_checkpoint
    original_apply = tpf._apply_phase_modes

    def capturing_save(path, model, metadata, extra=None):
        captured[Path(path).name] = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        return original_save(path, model, metadata, extra=extra)

    def capturing_apply(model, joint):
        # First epoch-start call: pre-update delta parameter snapshot.
        if "delta" not in captured:
            captured["delta"] = {
                f"delta_net.{name}": value.detach().cpu().clone()
                for name, value in model.delta_net.named_parameters()
            }
        return original_apply(model, joint)

    tpf.save_fusion_checkpoint = capturing_save
    tpf._apply_phase_modes = capturing_apply
    try:
        fusion_args = tpf.parse_args(
            [
                "--run-id", run_id,
                "--data", str(data_root),
                "--device", args.device,
                "--imgsz", str(args.imgsz),
                "--batch", str(args.batch),
                "--epochs", str(SMOKE_EPOCHS),
                "--limit-batches", str(SMOKE_BATCHES),
                "--phase", "freeze",
                "--gray-weights", str(gray_best),
            ]
        )
        run_dir = tpf.run_training(fusion_args)
    finally:
        tpf.save_fusion_checkpoint = original_save
        tpf._apply_phase_modes = original_apply
    _sync(device)
    result: dict = {
        "run_dir": str(run_dir),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "peak_gpu_bytes": _gpu_peak_bytes(device),
    }
    last_path = run_dir / "last.pt"
    for name in ("best.pt", "last.pt", "train_config.json"):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"fusion smoke output missing: {run_dir / name}")

    # Frozen gray branch: parameters AND BatchNorm buffers (tracked
    # separately) must be unchanged.
    from ultralytics import YOLO

    gray_module = YOLO(str(gray_best)).model
    payload = torch.load(last_path, map_location="cpu", weights_only=False)
    trained_state = payload["state_dict"]
    result.update(_check_gray_frozen(gray_module, trained_state))
    if not all(
        bool(torch.isfinite(value).all()) for value in trained_state.values()
    ):
        raise RuntimeError("fusion checkpoint contains non-finite tensors")

    # Rebuild via the actual production path on GPU for the remaining checks.
    metadata = read_fusion_metadata(last_path)
    backbone = rebuild_gray_backbone(metadata, device)
    model, _, _ = load_fusion_checkpoint(last_path, backbone)
    model = model.to(device)
    result["architecture"] = metadata.architecture

    # Delta updates by pre-update snapshot comparison: non-zero conv
    # initialization alone is not evidence; the zero-init final layer is
    # reported separately.
    last_names = _delta_last_layer_names(model)
    after_delta = {
        key: value for key, value in trained_state.items()
        if key.startswith("delta_net.")
    }
    delta_report = _delta_update_report(captured["delta"], after_delta, last_names)
    _assert_delta_updated(delta_report)
    result["delta_updates"] = delta_report

    # Gradients on a real batch with valid polar pixels (delta is non-zero,
    # so the gate must receive usable gradients; the zero-initialized
    # delta makes the first training step's gate gradient zero by design).
    valid_paths = _valid_polar_paths(data_root, "val", args.batch)
    loader = torch.utils.data.DataLoader(
        FusionClsDataset(valid_paths, imgsz=args.imgsz),
        batch_size=len(valid_paths),
        shuffle=False,
    )
    gray_b, polar_b, quality_b, labels_b = next(iter(loader))
    gray_b, polar_b, quality_b, labels_b = (
        gray_b.to(device), polar_b.to(device), quality_b.to(device), labels_b.to(device)
    )
    model.train()
    model.gray_backbone.eval()  # mirror freeze-phase modes
    model.set_gray_frozen(True)
    model.zero_grad(set_to_none=True)
    out = model(gray_b, polar_b, quality_b)
    delta_absmax = float(out["polar_delta"].detach().abs().max())
    if delta_absmax <= 0.0:
        raise RuntimeError("trained delta still produces zero output")
    loss = nn.CrossEntropyLoss()(out["final_logits"], labels_b)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("fusion smoke loss is not finite")
    gate_grads = [
        p.grad for p in model.gate_net.parameters() if p.grad is not None
    ]
    delta_grads = [
        p.grad for p in model.delta_net.parameters() if p.grad is not None
    ]
    if not gate_grads or not all(bool(torch.isfinite(g).all()) for g in gate_grads):
        raise RuntimeError("gate gradients are missing or not finite")
    gate_grad_norm = math.sqrt(
        sum(float((g.double() ** 2).sum()) for g in gate_grads)
    )
    if gate_grad_norm <= 0.0:
        raise RuntimeError("gate gradients are all zero despite non-zero delta")
    if not delta_grads or not all(bool(torch.isfinite(g).all()) for g in delta_grads):
        raise RuntimeError("delta gradients are missing or not finite")
    result["loss"] = float(loss.detach())
    result["gate_grad_norm"] = gate_grad_norm
    result["delta_output_absmax"] = delta_absmax

    # Strict fallback with non-zero delta: gate exactly 0, final == gray.
    model.eval()
    with torch.no_grad():
        zero_quality = quality_b.clone()
        zero_quality[:, 0] = 0.0  # valid_ratio = 0
        fallback_zero = model(gray_b, polar_b, zero_quality)
        invalid_flag = model(
            gray_b, polar_b, quality_b, polar_invalid=torch.ones(len(gray_b), device=device)
        )
    for name, out_fb in (("valid_ratio_zero", fallback_zero), ("polar_invalid", invalid_flag)):
        if not bool(torch.all(out_fb["gate"] == 0.0)):
            raise RuntimeError(f"{name}: gate is not exactly zero")
        if not torch.equal(out_fb["final_logits"], out_fb["gray_logits"]):
            raise RuntimeError(f"{name}: final_logits != gray_logits")
    result["fallback_valid_ratio_zero_exact"] = True
    result["fallback_polar_invalid_exact"] = True

    # Save/reload via the actual rebuild path: captured state vs the file.
    backbone_captured = rebuild_gray_backbone(metadata, device)
    model_captured, _, _ = load_fusion_checkpoint(last_path, backbone_captured)
    model_captured.load_state_dict(captured["last.pt"])
    model_captured = model_captured.to(device).eval()
    gray_fixed, polar_fixed, quality_fixed = _fixed_inputs(device, args.imgsz)
    with torch.no_grad():
        logits_captured = model_captured(gray_fixed, polar_fixed, quality_fixed)["final_logits"]
        logits_reloaded = model(gray_fixed, polar_fixed, quality_fixed)["final_logits"]
    max_diff = float((logits_captured - logits_reloaded).abs().max())
    if max_diff > LOGITS_TOLERANCE:
        raise RuntimeError(f"fusion save/reload logits differ by {max_diff}")
    result["save_reload_logits_max_diff"] = max_diff
    return result


def phase_c_continue(args: argparse.Namespace, device: str, data_root: Path, fusion_last: Path) -> dict:
    """Weight continuation from the smoke fusion checkpoint (truncated)."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    run_id = f"{args.run_id}_continue"
    captured: dict[str, dict] = {}
    original_apply = tpf._apply_phase_modes

    def capturing_apply(model, joint):
        if "state" not in captured:
            captured["state"] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            captured["param_names"] = {name for name, _ in model.named_parameters()}
            captured["buffer_names"] = {name for name, _ in model.named_buffers()}
        return original_apply(model, joint)

    tpf._apply_phase_modes = capturing_apply
    try:
        cont_args = tpf.parse_args(
            [
                "--run-id", run_id,
                "--data", str(data_root),
                "--device", args.device,
                # --imgsz omitted: must inherit the checkpoint's 224.
                "--batch", str(args.batch),
                "--epochs", "1",
                "--limit-batches", "1",
                "--phase", "freeze",
                "--init-from", str(fusion_last),
            ]
        )
        run_dir = tpf.run_training(cont_args)
    finally:
        tpf._apply_phase_modes = original_apply
    _sync(device)
    result: dict = {
        "run_dir": str(run_dir),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "peak_gpu_bytes": _gpu_peak_bytes(device),
    }

    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    if config["imgsz"] != args.imgsz:
        raise RuntimeError(
            f"continuation imgsz {config['imgsz']} != checkpoint imgsz {args.imgsz}"
        )
    result["inherited_imgsz"] = config["imgsz"]

    # Before the first optimizer update (first epoch-start mode switch) the
    # loaded parameters must equal the checkpoint exactly.
    payload = torch.load(fusion_last, map_location="cpu", weights_only=False)
    checkpoint_state = payload["state_dict"]
    if set(captured["state"]) != set(checkpoint_state):
        raise RuntimeError("continuation state keys differ from the checkpoint")
    mismatched = [
        key
        for key in checkpoint_state
        if not torch.equal(captured["state"][key].cpu(), checkpoint_state[key].cpu())
    ]
    if mismatched:
        raise RuntimeError(f"loaded parameters differ from checkpoint: {mismatched[:5]}")
    result["pre_update_params_matched"] = len(captured["param_names"])
    result["pre_update_buffers_matched"] = len(captured["buffer_names"])
    result["continuation_note"] = (
        "weight continuation only; optimizer/RNG state is not restored"
    )

    # Conflicting explicit imgsz must be refused before creating outputs.
    conflict_run_id = f"{run_id}_imgsz_conflict"
    conflict_args = tpf.parse_args(
        [
            "--run-id", conflict_run_id,
            "--data", str(data_root),
            "--device", args.device,
            "--imgsz", "512",
            "--batch", str(args.batch),
            "--epochs", "1",
            "--limit-batches", "1",
            "--phase", "freeze",
            "--init-from", str(fusion_last),
        ]
    )
    try:
        tpf.run_training(conflict_args)
        raise RuntimeError("conflicting --imgsz was not refused")
    except ValueError as exc:
        if "imgsz" not in str(exc):
            raise
    if (PROJECT_ROOT / "runs" / "train" / conflict_run_id).exists():
        raise RuntimeError("conflicting --imgsz created a run directory")
    result["conflicting_imgsz_refused"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Reject illegal run ids before any path construction, training call or
    # report write (same rule as the production training entries).
    try:
        validate_run_id(args.run_id)
    except InvalidRunIdError as exc:
        print(f"invalid run id: {exc}", file=sys.stderr)
        return 2

    runs_root = (PROJECT_ROOT / "runs" / "train").resolve()
    smoke_root = PROJECT_ROOT / "runs" / "train" / args.run_id
    target_dirs = [
        smoke_root,
        PROJECT_ROOT / "runs" / "train" / f"{args.run_id}_gray",
        PROJECT_ROOT / "runs" / "train" / f"{args.run_id}_freeze",
        PROJECT_ROOT / "runs" / "train" / f"{args.run_id}_continue",
    ]
    for path in target_dirs:
        if not path.resolve().is_relative_to(runs_root):
            print(
                f"run id {args.run_id!r} would escape runs/train; refusing",
                file=sys.stderr,
            )
            return 2

    try:
        device = tpf.torch_device_name(tpf.resolve_device(args.device))
    except DeviceUnavailableError as exc:
        print(f"device unavailable: {exc}", file=sys.stderr)
        return 2

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    base_path = Path(args.base)
    base_path = base_path if base_path.is_absolute() else PROJECT_ROOT / base_path
    if not base_path.is_file():
        print(f"base checkpoint {base_path} is missing; no automatic download", file=sys.stderr)
        return 2

    existing = [str(path) for path in target_dirs if path.exists()]
    if existing:
        print(f"smoke run directories already exist: {existing}; use a new --run-id", file=sys.stderr)
        return 2

    import ultralytics

    report: dict = {
        "run_id": args.run_id,
        "base": str(base_path),
        "base_sha256": file_digest(base_path),
        "base_architecture": architecture_name(
            ultralytics.YOLO(str(base_path)).model
        ),
        "data": str(data_root),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "limits": (
            f"{SMOKE_EPOCHS} epochs x <= {SMOKE_BATCHES} train/val batches per phase; "
            "truncated metrics prove the chain runs, never model quality"
        ),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
    }

    try:
        print("=== phase A: gray-only training ===")
        report["gray"] = phase_a_gray(args, device, data_root)
        gray_best = Path(report["gray"]["run_dir"]) / "best.pt"

        print("=== phase B: fusion freeze ===")
        report["fusion_freeze"] = phase_b_freeze(args, device, data_root, gray_best)

        print("=== phase C: weight continuation ===")
        fusion_last = Path(report["fusion_freeze"]["run_dir"]) / "last.pt"
        report["continuation"] = phase_c_continue(args, device, data_root, fusion_last)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, TypeError) as exc:
        report["status"] = "FAILED"
        report["error"] = str(exc)
        smoke_root.mkdir(parents=True, exist_ok=True)
        (smoke_root / "verify_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"verification FAILED: {exc}", file=sys.stderr)
        return 1

    report["status"] = "PASSED"
    smoke_root.mkdir(parents=True, exist_ok=True)
    report_path = smoke_root / "verify_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"verification report: {report_path}")
    print(
        "gray save/reload logits max diff: "
        f"{report['gray']['save_reload_logits_max_diff']:.3e}; "
        "fusion save/reload logits max diff: "
        f"{report['fusion_freeze']['save_reload_logits_max_diff']:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
