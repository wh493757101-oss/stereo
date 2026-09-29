"""Bounded smoke verification for the instance-mask pooling protocol (A1).

Runs the *real* production training entry on the real V5 dataset with a
strictly bounded budget, then verifies that the mask-pooling path is really
the one that trained:

1. the budget is refused above the smoke ceiling (2 epochs, 1 warmup epoch,
   3 batches per phase, batch 8, accum 2);
2. the resolved config is the v2 mask protocol and the dataset is the audited
   V5 state bound to the formal V4 parent;
3. the head really pools with the instance mask — the pooling helper is
   counted during training, and mask pooling is compared against GAP on the
   same weights;
4. **save-time evidence**: at the moment the production code writes each
   checkpoint, the live model's eval logits on a fixed sample set are
   captured, and each captured vector is compared against the logits of the
   model rebuilt from *that* checkpoint. The observation is read-only
   (module modes and RNG state are snapshotted and restored);
5. the warmup/finetune protocol still freezes and then unfreezes exactly the
   pretrained backbone;
6. **the polar-invalid fallback routing works**: on a mixed batch the
   production routing function sends forced-invalid samples through the
   independent historical gray model and valid samples through the v2
   classifier, checked against independently recomputed outputs (not by
   comparing weight files);
7. the loss, gradients and outputs are finite, from the training record;
8. the independent historical gray fallback file is byte-identical before
   and after, and the smoke run is refused by the formal completion
   admission so it can never enter a formal comparison.

Nothing here selects a model, a threshold or a donor; the smoke metrics are
bookkeeping for the plumbing, not evidence about polarization.

Usage:
    python scripts/verify_material_mask.py \
        --config configs/experiments/material_fusion_v2_mask.yaml \
        --data datasets/underwater_cls_fusion_v5_instance_mask \
        --variant gray_polar \
        --device 0 --run-id mask_a1_smoke_<tag> --smoke \
        --epochs 2 --warmup-epochs 1 --limit-batches 3 --batch 8 --accum-steps 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_fusion_dataset import MaterialFusionDataset
from core.material_mask_dataset import admit_mask_dataset
from models.material_fusion import (
    POOLING_INSTANCE_MASK,
    build_material_model,
    load_material_checkpoint,
    read_material_checkpoint_metadata,
    sha256_of_file,
)
from models.polar_fusion import read_manifest_split, torch_device_name
from scripts.train_material_fusion import _run_dir, load_experiment_config, resolve_config
from scripts.training_common import validate_run_id

# The ceiling this entry point refuses to exceed. Any request above it is a
# different experiment, not a verification.
SMOKE_BUDGET = {
    "epochs": 2,
    "warmup_epochs": 1,
    "limit_batches": 3,
    "batch": 8,
    "accum_steps": 2,
}
DEFAULT_OUTPUT_DIR = "analysis/runs/material_mask_v5"

# Fixed sample set used for the save-time comparison.
OBSERVED_SAMPLES = 4
# Logit tolerance for the save/reload comparison. Exact weight equality is
# verified separately and exactly (the live model's state_dict against the
# bytes written by that save); this tolerance covers the one remaining source
# of difference, which is the cuDNN convolution-algorithm selection on the
# *first* forward of a given input shape (the first observation in a run can
# differ from a fresh rebuild by ~1e-4 while the weights are byte-identical).
# It is deliberately far below any real weight difference: the injected
# corruption regression perturbs a tensor by 0.5 and must be detected.
RELOAD_ATOL = 1e-3


def smoke_formal_admission(run_id: str, run_dir, variant: str, seed) -> tuple[bool, str]:
    """The formal-admission verdict for a bounded smoke artifact.

    The smoke run is a *v2 mask* run, so it is judged against the v2 mask
    protocol explicitly. Validating it as v1 would refuse it for the wrong
    reason (its experiment is not ``material_fusion_v1``), which says nothing
    about the formal/smoke budget separation the check exists to prove.

    Shared by the production verification path and its tests, so there is one
    definition of "how a smoke artifact is judged formally" and no test-only
    re-implementation of it.
    """
    from scripts.run_material_experiments import (
        V2_MASK_SPEC,
        ExperimentTask,
        validate_completed_task,
    )

    task = ExperimentTask(
        task_id=f"{variant}_s{int(seed)}", variant=variant, seed=int(seed),
        run_id=str(run_id), run_dir=str(run_dir), protocol=V2_MASK_SPEC.name,
    )
    return validate_completed_task(task, V2_MASK_SPEC)


def smoke_refused_on_budget(admitted: bool, reason: str) -> bool:
    """True only when the refusal *is* the formal/smoke budget separation.

    An unrelated refusal (a missing directory, a wrong protocol, a foreign
    dataset identity) must not satisfy the check: it would let the smoke
    isolation "pass" without ever demonstrating that the smoke budget is what
    excludes the artifact.
    """
    return (
        not admitted
        and "fixed protocol" in reason
        and ("limit_batches" in reason or "smoke" in reason)
        and "not this task's protocol" not in reason
        and "run directory does not exist" not in reason
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default=None,
                        help="Variant to smoke (defaults to the config value). The "
                             "run id is only a name: without this, every smoke run "
                             "would train whatever variant the config declares.")
    parser.add_argument("--smoke", action="store_true",
                        help="Required: this entry point only runs bounded smoke.")
    parser.add_argument("--epochs", type=int, default=SMOKE_BUDGET["epochs"])
    parser.add_argument("--warmup-epochs", type=int, default=SMOKE_BUDGET["warmup_epochs"])
    parser.add_argument("--limit-batches", type=int, default=SMOKE_BUDGET["limit_batches"])
    parser.add_argument("--batch", type=int, default=SMOKE_BUDGET["batch"])
    parser.add_argument("--accum-steps", type=int, default=SMOKE_BUDGET["accum_steps"])
    parser.add_argument("--output", default="",
                        help="Report path (default: "
                             f"{DEFAULT_OUTPUT_DIR}/verify_<run-id>.json).")
    return parser.parse_args(argv)


def check_budget(args: argparse.Namespace) -> dict:
    """Refuse any request above the smoke ceiling."""
    if not args.smoke:
        raise ValueError(
            "verify_material_mask only runs bounded smoke verification; pass "
            "--smoke (formal training is a separate, explicitly authorised step)"
        )
    requested = {
        "epochs": int(args.epochs),
        "warmup_epochs": int(args.warmup_epochs),
        "limit_batches": int(args.limit_batches),
        "batch": int(args.batch),
        "accum_steps": int(args.accum_steps),
    }
    for field, ceiling in SMOKE_BUDGET.items():
        value = requested[field]
        if value < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be a positive integer")
        if value > ceiling:
            raise ValueError(
                f"--{field.replace('_', '-')} {value} exceeds the smoke budget "
                f"{ceiling}; this entry point refuses larger requests instead of "
                "growing the run until it passes"
            )
    if requested["warmup_epochs"] >= requested["epochs"]:
        raise ValueError("warmup-epochs must be smaller than epochs")
    return requested


class _PoolingSpy:
    """Counts instance-mask pooling calls without changing their behaviour."""

    def __init__(self, module):
        self._module = module
        self._original = module.mask_level_weights
        self.calls = 0
        module.mask_level_weights = self._count

    def _count(self, *args, **kwargs):
        self.calls += 1
        return self._original(*args, **kwargs)

    def reset(self) -> None:
        self.calls = 0

    def stop(self) -> None:
        self._module.mask_level_weights = self._original


# ---------------------------------------------------------------------------
# Save-time observation (read-only)


def materialize_fixed_batch(data_root, variant, imgsz, count, with_mask):
    """One pre-built batch over the first ``count`` val samples.

    The batch is materialized once, outside the observation window: creating
    a DataLoader iterator draws a base seed from the global RNG, so iterating
    one *inside* the observer would perturb the training RNG stream. The
    returned tensors are plain CPU tensors and the sample names are recorded
    so a report can name exactly what was compared.
    """
    from core.material_fusion_dataset import unpack_batch

    val_paths, _, _ = read_manifest_split(data_root, "val")
    chosen = list(val_paths[:count])
    loader = torch.utils.data.DataLoader(
        MaterialFusionDataset(
            chosen, variant=variant, imgsz=imgsz, with_instance_mask=with_mask
        ),
        batch_size=max(len(chosen), 1), shuffle=False, num_workers=0,
    )
    batch = next(iter(loader))
    is_dual = variant != "gray"
    rgb, aux, quality, labels, mask = unpack_batch(batch, is_dual, with_mask)
    return {
        "rgb": rgb,
        "aux": aux,
        "quality": quality,
        "labels": labels,
        "mask": mask,
        "names": [Path(path).stem for path in chosen],
        "variant": variant,
        "with_mask": bool(with_mask),
    }


def _snapshot_module_modes(model) -> dict[str, bool]:
    return {name: module.training for name, module in model.named_modules()}


def _restore_module_modes(model, snapshot: dict[str, bool]) -> None:
    for name, module in model.named_modules():
        if name in snapshot:
            module.training = snapshot[name]


@torch.no_grad()
def forward_fixed_batch(model, fixed, device):
    """Raw logits for the pre-built batch, in eval mode, restoring the modes."""
    was_training = _snapshot_module_modes(model)
    model.eval()
    try:
        rgb = fixed["rgb"].to(device)
        aux = None if fixed["aux"] is None else fixed["aux"].to(device)
        mask = None if fixed["mask"] is None else fixed["mask"].to(device)
        if mask is None:
            out = model(rgb, aux) if model.is_dual else model(rgb)
        else:
            out = model(rgb, aux, mask)
        return out.detach().cpu()
    finally:
        _restore_module_modes(model, was_training)


class _SaveObserver:
    """Captures the live model's eval logits at every production save.

    Read-only by construction: it snapshots and restores every module's
    train/eval flag and the torch RNG state, runs under ``no_grad`` in eval
    mode over a pre-built batch (so no DataLoader iterator is created), and
    never touches parameters, buffers or the optimizer. It records whether the
    RNG state was left exactly as it found it, so non-interference is evidence
    rather than a claim.
    """

    def __init__(self, fixed, device):
        self._fixed = fixed
        self._device = device
        self.records: list[dict] = []

    def __call__(self, path, role, epoch, model):
        # The checkpoint content is captured now, because a later epoch can
        # legitimately overwrite best.pt: the comparison must use the bytes
        # written by *this* save, not whatever file happens to remain.
        payload_bytes = Path(path).read_bytes()
        payload = torch.load(
            __import__("io").BytesIO(payload_bytes), map_location="cpu",
            weights_only=True,
        )
        live_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        saved_state = payload["state_dict"]
        state_dict_matches = (
            set(live_state) == set(saved_state)
            and all(
                torch.equal(live_state[key], saved_state[key]) for key in live_state
            )
        )
        modes = _snapshot_module_modes(model)
        cpu_rng_before = torch.random.get_rng_state()
        cuda_rng_before = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            logits = forward_fixed_batch(model, self._fixed, self._device)
        finally:
            _restore_module_modes(model, modes)
            torch.random.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state_all(cuda_rng_before)
        rng_restored = torch.equal(cpu_rng_before, torch.random.get_rng_state())
        if cuda_rng_before is not None:
            rng_restored = rng_restored and all(
                torch.equal(a, b)
                for a, b in zip(cuda_rng_before, torch.cuda.get_rng_state_all())
            )
        self.records.append(
            {
                "path": str(path),
                "payload_bytes": payload_bytes,
                "role": role,
                "epoch": int(epoch),
                "logits": logits,
                "state_dict_matches_saved_bytes": bool(state_dict_matches),
                "rng_state_unchanged": bool(rng_restored),
                "module_modes_restored": _snapshot_module_modes(model) == modes,
            }
        )


def compare_saved_checkpoints(
    records, fixed, device, variant, imgsz, atol=RELOAD_ATOL, mutate=None,
) -> list[dict]:
    """Rebuild each observed save and compare against the logits captured then.

    ``mutate`` (used only by tests) perturbs the rebuilt model, so the
    comparison can be shown to fail when the reload does not reproduce the
    saved weights.
    """
    import tempfile

    results = []
    for record in records:
        # Rebuild from the bytes captured at that save, through the production
        # loader, so this is a real save/reload comparison even when a later
        # epoch overwrote the file.
        with tempfile.TemporaryDirectory(prefix="mask_reload_") as tmp:
            reload_path = Path(tmp) / f"{record['role']}_epoch{record['epoch']}.pt"
            reload_path.write_bytes(record["payload_bytes"])
            model, metadata, _ = load_material_checkpoint(
                reload_path, variant=variant, imgsz=imgsz,
                pooling_mode=POOLING_INSTANCE_MASK,
            )
            if mutate is not None:
                mutate(model)
            model = model.to(device)
            rebuilt = forward_fixed_batch(model, fixed, device)
            max_abs = float((record["logits"] - rebuilt).abs().max())
        results.append(
            {
                "role": record["role"],
                "epoch": record["epoch"],
                "checkpoint": record["path"],
                "note": "compared against the bytes written by this save",
                "checkpoint_epoch": metadata.epoch,
                "checkpoint_stage": metadata.stage,
                "samples": int(rebuilt.shape[0]),
                "atol": float(atol),
                "max_abs_error": max_abs,
                "ok": max_abs <= atol,
            }
        )
        del model
    return results


# ---------------------------------------------------------------------------
# Fallback routing (functional)


class _CountingFallback:
    """Wraps the fallback and counts how many samples actually reach it."""

    def __init__(self, fallback):
        self._fallback = fallback
        self.rows = 0
        self.calls = 0

    def __call__(self, rgb):
        self.rows += int(rgb.shape[0])
        self.calls += 1
        return self._fallback(rgb)

    def eval(self):
        self._fallback.eval()
        return self


class _ForcedInvalidDataset(torch.utils.data.Dataset):
    """Forces ``valid_ratio == 0`` for chosen samples, in memory only.

    The production routing keys on the quality vector's ``valid_ratio``; this
    wrapper changes that vector for the wrapped item so a mixed valid/invalid
    batch can be built without touching any real NPZ file.
    """

    def __init__(self, base, forced_indices, dual: bool):
        self._base = base
        self._forced = set(int(index) for index in forced_indices)
        # (rgb, aux, quality, ...) for the dual variants, (rgb, quality, ...)
        # for the single stream; the trailing class_id/mask shift the offset.
        self._quality_index = 2 if dual else 1

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index):
        item = list(self._base[index])
        if index in self._forced:
            quality = item[self._quality_index].clone()
            quality[0] = 0.0
            item[self._quality_index] = quality
        return tuple(item)


def verify_fallback_routing(
    model, fallback, loader, device, forced_invalid, route_fn=None,
) -> dict:
    """Functional check of the polar-invalid routing on a mixed batch.

    Runs the production routing function (``eval_material_fusion.route_batch``
    unless ``route_fn`` is injected) on a batch containing both forced-invalid
    and valid samples, then independently recomputes what each sample's
    prediction must be: invalid samples must equal the independent historical
    gray model's output, valid samples the v2 classifier's output. Also checks
    that the fallback received exactly the invalid rows.

    ``route_fn`` exists so a test can inject a wrong routing and prove this
    check fails.
    """
    from scripts.eval_material_fusion import route_batch as production_route
    from scripts.train_material_fusion import _unpack_batch

    route_fn = route_fn or production_route
    model.eval()
    if hasattr(fallback, "eval"):
        fallback.eval()

    for batch in loader:
        rgb, aux, quality, labels, mask = _unpack_batch(
            batch, model.is_dual, device, True
        )
        break
    else:
        raise ValueError("the fallback routing check needs at least one batch")

    invalid = quality[:, 0] <= 0
    expected_invalid = torch.zeros_like(invalid)
    for index in forced_invalid:
        if 0 <= int(index) < expected_invalid.numel():
            expected_invalid[int(index)] = True
    quality_matches_forced = bool(torch.equal(invalid, expected_invalid))

    counter = _CountingFallback(fallback)
    routed_preds, routed_logits, used_fallback = route_fn(
        model, counter, rgb, aux, invalid, mask
    )
    with torch.no_grad():
        if mask is None:
            model_logits = model(rgb, aux) if model.is_dual else model(rgb)
        else:
            model_logits = model(rgb, aux, mask)
        fallback_logits = fallback(rgb)

    expected = torch.where(
        invalid.to(model_logits.device),
        fallback_logits.argmax(dim=1),
        model_logits.argmax(dim=1),
    ).cpu()

    invalid_rows = int(invalid.sum())
    return {
        "samples": int(rgb.shape[0]),
        "forced_invalid": sorted(int(i) for i in forced_invalid),
        "invalid_rows": invalid_rows,
        "quality_matches_forced": quality_matches_forced,
        "fallback_rows": int(counter.rows),
        "fallback_calls": int(counter.calls),
        "used_fallback_matches": bool(
            torch.equal(used_fallback.cpu(), invalid.cpu())
        ),
        "predictions_match_expected": bool(
            torch.equal(routed_preds.cpu(), expected)
        ),
        # Informational: on a barely trained smoke model the two classifiers may
        # or may not agree; the injected-wrong-route regression is what proves
        # this check can fail.
        "model_and_fallback_agree_on_all": bool(
            torch.equal(
                model_logits.argmax(dim=1).cpu(), fallback_logits.argmax(dim=1).cpu()
            )
        ),
        "invalid_predictions_from_fallback": bool(
            torch.equal(
                routed_preds.cpu()[invalid.cpu()], fallback_logits.argmax(dim=1).cpu()[invalid.cpu()]
            )
        ),
        "valid_predictions_from_model": bool(
            torch.equal(
                routed_preds.cpu()[~invalid.cpu()], model_logits.argmax(dim=1).cpu()[~invalid.cpu()]
            )
        ),
    }


def check_finiteness(history_path: Path) -> dict:
    """Loss, gradient and metric finiteness from the training record."""
    entries = [
        json.loads(line)
        for line in Path(history_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_finite = [
        entry["epoch"]
        for entry in entries
        if not math.isfinite(float(entry["train_loss"]))
        or not math.isfinite(float(entry["val_macro_f1"]))
        or not math.isfinite(float(entry["val_accuracy"]))
    ]
    gradients_not_finite = [
        entry["epoch"] for entry in entries if not entry.get("gradients_finite", False)
    ]
    return {
        "epochs": len(entries),
        "non_finite_metrics": non_finite,
        "epochs_with_non_finite_gradients": gradients_not_finite,
        "ok": bool(entries) and not non_finite and not gradients_not_finite,
    }


# ---------------------------------------------------------------------------
# Frozen/unfrozen evidence


def _state_dict_of(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["state_dict"]


def _backbone_keys(state_dict: dict) -> list[str]:
    from models.material_fusion import _BACKBONE_MODULE_PREFIXES, _FUSION_MODULE_PREFIXES

    return [
        key for key in state_dict
        if key.startswith("encoder.")
        and not key.split(".")[1].startswith(_FUSION_MODULE_PREFIXES)
        and key.split(".")[1].startswith(_BACKBONE_MODULE_PREFIXES)
    ]


def _fusion_or_head_keys(state_dict: dict) -> list[str]:
    return [
        key for key in state_dict
        if key.startswith("head.") or "FRMs" in key or "FFMs" in key
    ]


def _equal_on(before: dict, after: dict, keys) -> bool:
    return all(torch.equal(before[key], after[key]) for key in keys)


def _changed_on(before: dict, after: dict, keys) -> bool:
    return any(not torch.equal(before[key], after[key]) for key in keys)


def _val_loader(data_root, variant, imgsz, batch, with_mask):
    val_paths, _, _ = read_manifest_split(data_root, "val")
    return torch.utils.data.DataLoader(
        MaterialFusionDataset(
            val_paths, variant=variant, imgsz=imgsz, with_instance_mask=with_mask
        ),
        batch_size=batch, shuffle=False, num_workers=0,
    )


def run_verification(args: argparse.Namespace) -> dict:
    from scripts.train_material_fusion import main as train_main
    import scripts.train_material_fusion as train_module

    requested = check_budget(args)
    validate_run_id(args.run_id)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    template = load_experiment_config(config_path)
    # The smoke overrides are explicit CLI values, exactly as a smoke run
    # would pass them; the config's formal budget is never edited.
    config = resolve_config(template, argparse.Namespace(
        variant=args.variant, data=str(args.data), pretrained=None,
        batch=requested["batch"], accum=requested["accum_steps"],
        epochs=requested["epochs"], warmup_epochs=requested["warmup_epochs"],
        lr=None, encoder_lr=None, weight_decay=None, seed=None,
        limit_batches=requested["limit_batches"],
    ))
    if config["pooling_mode"] != POOLING_INSTANCE_MASK:
        raise ValueError(
            f"config pooling_mode is {config['pooling_mode']!r}; this verifier "
            "only covers the instance-mask protocol"
        )
    # The effective batch is a protocol constant: a smoke run may truncate the
    # number of batches, but it may not change the batch geometry, or it would
    # no longer be the protocol being verified.
    if requested["batch"] * requested["accum_steps"] != int(config["effective_batch"]):
        raise ValueError(
            f"batch {requested['batch']} x accum {requested['accum_steps']} != the "
            f"config's effective_batch {config['effective_batch']}; the effective "
            "batch is a fixed protocol constant"
        )

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    admission = admit_mask_dataset(
        data_root,
        expected_fingerprint=config["data_fingerprint"],
        expected_parent_fingerprint=config["parent_data_fingerprint"],
    )

    fallback = PROJECT_ROOT / config["fallback_weights"]
    fallback_sha_before = sha256_of_file(fallback) if fallback.is_file() else ""

    run_dir = _run_dir(args.run_id)
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use a unique --run-id"
        )
    snapshot_dir = run_dir / "stage_snapshots"

    import models.material_fusion as model_module

    device = torch_device_name(args.device)
    fixed = materialize_fixed_batch(
        data_root, config["variant"], int(config["imgsz"]), OBSERVED_SAMPLES, True
    )
    fixed_names = fixed["names"]
    observer = _SaveObserver(fixed, device)

    # One forward over the fixed batch before training, so the first
    # observation does not also pay cuDNN's algorithm selection for that input
    # shape (a one-off ~1e-4 effect that is unrelated to the saved weights).
    warmup_model = build_material_model(
        config["variant"], imgsz=int(config["imgsz"]), pooling_mode=POOLING_INSTANCE_MASK
    ).to(device)
    forward_fixed_batch(warmup_model, fixed, device)
    del warmup_model

    spy = _PoolingSpy(model_module)
    previous_observer = train_module.SAVE_OBSERVER
    train_module.SAVE_OBSERVER = observer
    try:
        exit_code = train_main([
            "--run-id", args.run_id,
            "--config", str(config_path),
            "--variant", config["variant"],
            "--data", str(data_root),
            "--device", args.device,
            "--batch", str(requested["batch"]),
            "--accum", str(requested["accum_steps"]),
            "--epochs", str(requested["epochs"]),
            "--warmup-epochs", str(requested["warmup_epochs"]),
            "--limit-batches", str(requested["limit_batches"]),
            "--stage-snapshot-dir", str(snapshot_dir),
        ])
        training_pooling_calls = spy.calls
    finally:
        train_module.SAVE_OBSERVER = previous_observer
        spy.stop()
    if exit_code != 0:
        raise RuntimeError(
            f"the bounded smoke training entry returned {exit_code}; "
            f"see {run_dir} for what it produced"
        )

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- checkpoint contract -------------------------------------------------
    best_meta = read_material_checkpoint_metadata(run_dir / "best.pt")
    record(
        "checkpoint_records_the_mask_protocol",
        best_meta.pooling_mode == POOLING_INSTANCE_MASK
        and best_meta.mask_schema == config["mask_schema"]
        and best_meta.mask_resize_version == config["mask_resize_version"]
        and best_meta.parent_data_fingerprint == config["parent_data_fingerprint"],
        f"pooling_mode={best_meta.pooling_mode} mask_schema={best_meta.mask_schema} "
        f"resize={best_meta.mask_resize_version} "
        f"parent={best_meta.parent_data_fingerprint}",
    )
    record(
        "checkpoint_records_the_v5_fingerprint",
        best_meta.data_fingerprint == admission["audit_fingerprint"],
        f"{best_meta.data_fingerprint} vs {admission['audit_fingerprint']}",
    )
    model_groups = None
    variant_matches = best_meta.variant == config["variant"]
    record(
        "checkpoint_variant_matches_the_requested_variant",
        variant_matches,
        f"checkpoint variant={best_meta.variant} requested={config['variant']} "
        "(a run id is only a name; the variant must be requested explicitly)",
    )
    probe = build_material_model(config["variant"], imgsz=best_meta.imgsz)
    model_groups = {name: len(params) for name, params in probe.parameter_groups().items()}
    del probe
    expects_fusion = config["variant"] != "gray"
    record(
        "parameter_groups_match_the_variant",
        (model_groups["fusion"] > 0) == expects_fusion,
        f"variant={config['variant']} groups={model_groups} "
        f"(fusion layers expected: {expects_fusion})",
    )
    refused_gap = False
    try:
        load_material_checkpoint(run_dir / "best.pt", variant=config["variant"],
                                 pooling_mode="gap")
    except ValueError:
        refused_gap = True
    record(
        "gap_loading_of_a_mask_checkpoint_refused", refused_gap,
        "loading the mask checkpoint as a GAP model is refused",
    )

    model, metadata, _ = load_material_checkpoint(
        run_dir / "best.pt", variant=config["variant"],
        pooling_mode=POOLING_INSTANCE_MASK,
    )
    model = model.to(device)

    # --- save-time evidence --------------------------------------------------
    save_records = compare_saved_checkpoints(
        observer.records, fixed, device, config["variant"], metadata.imgsz
    )
    roles = [entry["role"] for entry in save_records]
    record(
        "save_time_logits_compared_for_every_save",
        len(save_records) == len(observer.records) and bool(save_records),
        f"roles={roles} samples={fixed_names}",
    )
    record(
        "saved_bytes_hold_exactly_the_live_weights",
        all(entry["state_dict_matches_saved_bytes"] for entry in observer.records),
        "the live model's state_dict equals, element for element, the bytes "
        "written by each save",
    )
    record(
        "live_logits_match_the_saved_checkpoints",
        all(entry["ok"] for entry in save_records),
        "; ".join(
            f"{entry['role']}@epoch{entry['epoch']} max_abs={entry['max_abs_error']:.3e} "
            f"(atol={entry['atol']})"
            for entry in save_records
        ),
    )
    record(
        "observation_left_the_rng_and_module_modes_untouched",
        all(
            entry["rng_state_unchanged"] and entry["module_modes_restored"]
            for entry in observer.records
        ),
        "read-only observation: the RNG state and every module's train/eval "
        "flag are exactly as they were before the observation",
    )

    # --- the mask is not inert ----------------------------------------------
    reloaded, _, _ = load_material_checkpoint(
        run_dir / "best.pt", variant=config["variant"],
        pooling_mode=POOLING_INSTANCE_MASK,
    )
    reloaded = reloaded.to(device)
    gap_fixed = materialize_fixed_batch(
        data_root, config["variant"], metadata.imgsz, OBSERVED_SAMPLES, False
    )
    live_logits = forward_fixed_batch(reloaded, fixed, device)
    reloaded.head.pooling_mode = "gap"
    gap_logits = forward_fixed_batch(reloaded, gap_fixed, device)
    reloaded.head.pooling_mode = POOLING_INSTANCE_MASK
    mask_delta = float((live_logits - gap_logits).abs().max())
    record(
        "instance_pooling_differs_from_gap_on_the_same_weights",
        mask_delta > 0.0,
        f"max |mask-pooled - gap| = {mask_delta}",
    )
    record(
        "pooling_helper_ran_during_training",
        training_pooling_calls > 0,
        f"{training_pooling_calls} mask-resampling calls during the smoke run",
    )

    # --- fallback routing (functional) --------------------------------------
    from scripts.eval_material_fusion import load_fallback_gray

    routing: dict | None = None
    if config["variant"] == "gray":
        record(
            "polar_invalid_fallback_routing",
            True,
            "N/A: the single-stream gray variant has no polarization input, so "
            "there is no polar-invalid fallback to exercise",
        )
    else:
        fallback_model, fallback_identity = load_fallback_gray(
            fallback, device, metadata.imgsz
        )
        forced = {0, 2}
        mixed_dataset = _ForcedInvalidDataset(
            MaterialFusionDataset(
                read_manifest_split(data_root, "val")[0][: requested["batch"]],
                variant=config["variant"], imgsz=metadata.imgsz,
                with_instance_mask=True,
            ),
            forced, dual=config["variant"] != "gray",
        )
        mixed_loader = torch.utils.data.DataLoader(
            mixed_dataset, batch_size=requested["batch"], shuffle=False, num_workers=0
        )
        routing = verify_fallback_routing(
            model, fallback_model, mixed_loader, device, forced
        )
        routing["fallback_identity"] = fallback_identity
        record(
            "polar_invalid_fallback_routing",
            routing["quality_matches_forced"]
            and routing["used_fallback_matches"]
            and routing["predictions_match_expected"]
            and routing["invalid_predictions_from_fallback"]
            and routing["valid_predictions_from_model"]
            and routing["fallback_rows"] == routing["invalid_rows"]
            and routing["invalid_rows"] == len(forced),
            f"invalid_rows={routing['invalid_rows']} "
            f"fallback_rows={routing['fallback_rows']} "
            f"predictions_match={routing['predictions_match_expected']} "
            f"used_fallback_matches={routing['used_fallback_matches']}",
        )

    # --- finiteness ----------------------------------------------------------
    finiteness = check_finiteness(run_dir / "metrics_history.jsonl")
    record(
        "loss_gradients_and_metrics_are_finite",
        finiteness["ok"],
        f"epochs={finiteness['epochs']} "
        f"non_finite_metrics={finiteness['non_finite_metrics']} "
        f"non_finite_gradients={finiteness['epochs_with_non_finite_gradients']}",
    )

    # --- the frozen/unfrozen protocol is unchanged ---------------------------
    init = _state_dict_of(snapshot_dir / "init.pt")
    warmup = _state_dict_of(snapshot_dir / "post_warmup.pt")
    finetune = _state_dict_of(snapshot_dir / "post_finetune.pt")
    backbone = _backbone_keys(init)
    trainable = _fusion_or_head_keys(init)
    record(
        "warmup_freezes_the_backbone_and_trains_the_new_layers",
        _equal_on(init, warmup, backbone) and _changed_on(init, warmup, trainable),
        f"{len(backbone)} backbone tensors frozen, "
        f"{len(trainable)} new-layer tensors updated",
    )
    record(
        "finetune_unfreezes_and_trains_the_backbone",
        _changed_on(warmup, finetune, backbone),
        f"{len(backbone)} backbone tensors changed after unfreezing",
    )

    # --- the independent gray fallback is untouched --------------------------
    fallback_sha_after = sha256_of_file(fallback) if fallback.is_file() else ""
    record(
        "independent_gray_fallback_unchanged",
        bool(fallback_sha_before) and fallback_sha_before == fallback_sha_after,
        f"{fallback} sha256 {fallback_sha_before}",
    )

    # --- smoke truncation is recorded and cannot enter a formal comparison ---
    # The run is a v2 mask run and is judged as one; the refusal must be the
    # formal/smoke budget separation, not an unrelated reason.
    admitted, reason = smoke_formal_admission(
        args.run_id, run_dir, config["variant"], config["seed"]
    )
    record(
        "smoke_run_refused_by_the_formal_admission",
        smoke_refused_on_budget(admitted, reason),
        f"formal admission said: {reason}",
    )

    report = {
        "kind": "material_mask_smoke_verification",
        "smoke": True,
        "limit_batches": requested["limit_batches"],
        "smoke_truncation_note": (
            "Bounded smoke run: at most 3 train and 3 val batches per epoch. "
            "These metrics are plumbing bookkeeping and are excluded from any "
            "formal comparison; the formal admission refuses smoke artifacts."
        ),
        "budget": requested,
        "budget_ceiling": SMOKE_BUDGET,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "device": device,
        "variant": config["variant"],
        "parameter_groups": model_groups,
        "seed": int(config["seed"]),
        "pooling_mode": config["pooling_mode"],
        "mask_schema": config["mask_schema"],
        "mask_resize_version": config["mask_resize_version"],
        "parent_data_fingerprint": config["parent_data_fingerprint"],
        "data_fingerprint": admission["audit_fingerprint"],
        "data_report": admission,
        "fallback_weights": str(fallback),
        "fallback_sha256_before": fallback_sha_before,
        "fallback_sha256_after": fallback_sha_after,
        "mask_pooling_calls_during_training": training_pooling_calls,
        "save_time_comparison": {
            "observed_samples": fixed_names,
            "atol": RELOAD_ATOL,
            "atol_note": (
                "Exact weight equality is asserted separately "
                "(saved_bytes_hold_exactly_the_live_weights); the tolerance only "
                "covers cuDNN algorithm selection on the first forward of a shape."
            ),
            "results": save_records,
            "note": (
                "Each entry compares the live model's eval logits captured at the "
                "production save moment against the logits of the model rebuilt "
                "from that same checkpoint file."
            ),
        },
        "fallback_routing": routing,
        "finiteness": finiteness,
        "mask_vs_gap_max_abs_logit_diff": mask_delta,
        "checks": checks,
        "all_ok": all(check["ok"] for check in checks),
    }
    output = args.output or str(Path(DEFAULT_OUTPUT_DIR) / f"verify_{args.run_id}.json")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["report_path"] = str(output_path)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_verification(args)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"mask smoke verification refused: {exc}", file=sys.stderr)
        return 2
    for check in report["checks"]:
        print(f"[{'ok' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
    print(f"report: {report['report_path']}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
