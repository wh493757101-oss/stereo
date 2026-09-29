"""Train the CMX material-fusion classifier on a material-fusion dataset.

Two protocol versions share this entry point and this training algorithm:

- ``material_fusion_v1`` (``configs/experiments/material_fusion_v1.yaml``):
  the V4 dataset, global average pooling head.
- ``material_fusion_v2_mask``
  (``configs/experiments/material_fusion_v2_mask.yaml``): the V5 dataset
  (V4 arrays plus the annotated instance mask), instance-mask weighted
  pooling head. Everything else — encoder, variants, losses, optimizer,
  learning rates, schedule, selection rule and budget — is unchanged.

Protocol (fixed by the experiment config):

- AdamW, weight_decay 0.01, no scheduler, no class weighting, no
  augmentation, single fixed learning-rate scheme.
- ``warmup_epochs`` epochs with the pretrained encoder frozen (eval mode),
  then the remaining epochs with the encoder open at ``encoder_lr`` while
  the new layers keep ``lr``.
- Gradient accumulation keeps the effective batch identical across
  variants; the tail group of every epoch is stepped with its own count.
- Model selection is val Macro-F1 (primary) with accuracy as tiebreak;
  a tie keeps the earlier epoch.
- Only train/val splits are ever read. The test split is never touched by
  training or model selection.

``--limit-batches > 0`` marks a smoke run: bounded train/val batches,
``smoke=True`` and the batch cap recorded in the checkpoint metadata.
Formal runs use ``--limit-batches 0``. Outputs go to
``runs/train/<run-id>/material_fusion/`` and an existing directory is
always refused.

Usage:
    python scripts/train_material_fusion.py --run-id run_x --variant gray_polar --device 0
    python scripts/train_material_fusion.py --run-id smoke_x --variant gray_polar \
        --device 0 --batch 8 --accum 2 --epochs 2 --warmup-epochs 1 --limit-batches 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_fusion_dataset import MaterialFusionDataset
from core.material_mask_dataset import admit_dataset_for_protocol
from models.material_fusion import (
    POOLING_INSTANCE_MASK,
    STAGES,
    build_material_model,
    load_material_checkpoint,
    load_pretrained_into_model,
    material_fusion_metadata,
    read_material_checkpoint_metadata,
    save_material_checkpoint,
    validate_material_config,
    validate_variant,
)
from models.polar_fusion import read_manifest_split, torch_device_name
from scripts.training_common import (
    DeviceUnavailableError,
    InvalidRunIdError,
    resolve_device,
    validate_run_id,
)
from scripts.train_polar_fusion import classification_metrics, validate_dataset

DEFAULT_CONFIG = "configs/experiments/material_fusion_v1.yaml"
RUN_SUBDIR = "material_fusion"

# Output root for training runs. Resolved at call time by ``_run_dir`` so a
# test can redirect every training output into its own scratch directory
# instead of the real ``runs/train``; the production default is unchanged and
# no CLI flag relaxes it.
RUNS_ROOT = PROJECT_ROOT / "runs" / "train"

# Optional, default-off observation hook. When set (by the bounded smoke
# verifier, or by a test), it is called with the live model immediately after
# a checkpoint has been written, so the saved weights can be compared against
# the exact live model state that produced them. The hook is read-only by
# contract: it must not train, must not change the model's parameters,
# buffers, module modes or RNG, and it must restore whatever it touches. The
# production entry point never sets it, and it never copies training logic.
SAVE_OBSERVER = None


def _notify_saved(path: Path, role: str, epoch: int, model) -> None:
    """Call the optional save observer (no-op when unset)."""
    observer = SAVE_OBSERVER
    if observer is None:
        return
    observer(Path(path), str(role), int(epoch), model)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True,
                        help="Run id grouping outputs under runs/train/<run-id>/material_fusion.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Experiment config (default: %(default)s).")
    parser.add_argument("--variant", default=None,
                        help="Variant override; defaults to the config value.")
    parser.add_argument("--data", default=None, help="Dataset root override.")
    parser.add_argument("--pretrained", default=None,
                        help="MiT-B0 pretrained checkpoint override (never auto-downloaded).")
    parser.add_argument("--init-from", default=None,
                        help="Material-fusion checkpoint to continue weights from "
                             "(weights-only init; not a full optimizer/RNG resume).")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=None, help="Physical batch override.")
    parser.add_argument("--accum", type=int, default=None,
                        help="Gradient accumulation steps override.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Frozen-encoder epochs override.")
    parser.add_argument("--lr", type=float, default=None, help="New-layers learning rate.")
    parser.add_argument("--encoder-lr", type=float, default=None,
                        help="Encoder learning rate after unfreezing.")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="Cap train/val batches per epoch (0 = all). >0 marks a smoke run.")
    parser.add_argument("--stage-snapshot-dir", default=None,
                        help="Observation hook: write init/post-warmup/post-finetune "
                             "state-dict snapshots of this production training path "
                             "into the given directory (used by the verification "
                             "harness; no extra training is performed).")
    return parser.parse_args(argv)


def load_experiment_config(path: str | Path) -> dict:
    """Read the experiment YAML config (validated separately afterwards)."""
    import yaml

    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"experiment config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"experiment config {path} must be a YAML mapping")
    config["_config_path"] = str(path)
    return config


def resolve_config(config: dict, args: argparse.Namespace) -> dict:
    """Apply explicit CLI overrides on top of the validated config."""
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    overrides = {
        "variant": args.variant,
        "data": args.data,
        "pretrained": args.pretrained,
        "batch": args.batch,
        "accum_steps": args.accum,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "lr": args.lr,
        "encoder_lr": args.encoder_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    if args.limit_batches is not None:
        resolved["limit_batches"] = args.limit_batches
        resolved["smoke"] = args.limit_batches > 0
    return validate_material_config(resolved)


def stage_for_epoch(epoch: int, warmup_epochs: int) -> str:
    return "warmup" if epoch < int(warmup_epochs) else "finetune"


def apply_stage(model, stage: str) -> None:
    """Epoch-start module modes for the requested stage (model protocol)."""
    model.set_stage(stage)


def accumulation_groups(total_batches: int, accum: int) -> list[tuple[int, int]]:
    """Stable accumulation group boundaries ``[start, end)``.

    Every group covers exactly ``accum`` consecutive micro-batches except
    the last, which covers the remainder and is always stepped; the number
    of groups (and therefore optimizer steps) is ``ceil(total / accum)``.
    """
    if type(accum) is not int or accum < 1:
        raise ValueError(f"accum must be a positive int, got {accum!r}")
    if type(total_batches) is not int or total_batches < 0:
        raise ValueError(f"total_batches must be a non-negative int, got {total_batches!r}")
    groups: list[tuple[int, int]] = []
    start = 0
    while start < total_batches:
        end = min(start + accum, total_batches)
        groups.append((start, end))
        start = end
    return groups


def build_optimizer(model, lr: float, encoder_lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with the protocol groups: backbone / fusion / head.

    The pretrained backbone trains at ``encoder_lr`` (after unfreezing),
    the new FRM/FFM layers and the classification head at ``lr`` in both
    stages. Empty groups (no FRM/FFM in the single-stream variant) are
    omitted; every parameter appears exactly once.
    """
    groups = model.parameter_groups()
    param_groups = []
    for name, group_lr in (("head", lr), ("fusion", lr), ("backbone", encoder_lr)):
        if groups[name]:
            param_groups.append(
                {"name": name, "params": groups[name], "lr": group_lr,
                 "weight_decay": weight_decay}
            )
    return torch.optim.AdamW(param_groups)


def optimizer_group_report(model, optimizer) -> list[dict]:
    """Serializable description of the optimizer's parameter groups."""
    return [
        {"name": group["name"], "lr": group["lr"],
         "weight_decay": group["weight_decay"], "params": len(group["params"])}
        for group in optimizer.param_groups
    ]


def selection_key(metrics: dict) -> tuple[float, float]:
    """Val Macro-F1 primary, accuracy tiebreak."""
    return (float(metrics["macro_f1"]), float(metrics["accuracy"]))


def should_replace_best(best_key: tuple[float, float], new_key: tuple[float, float]) -> bool:
    """Strictly better replaces; an exact tie keeps the earlier epoch."""
    return new_key > best_key


def _unpack_batch(batch, dual: bool, device: str, with_instance_mask: bool = False):
    """Move one collated batch to the device.

    Returns ``(rgb, aux, quality, labels, instance_mask)``; the mask is
    ``None`` for a GAP run so the model never receives one.
    """
    from core.material_fusion_dataset import unpack_batch

    rgb, aux, quality, labels, mask = unpack_batch(
        batch, dual, with_instance_mask
    )
    moved = [
        None if tensor is None else tensor.to(device)
        for tensor in (rgb, aux, quality, labels, mask)
    ]
    return tuple(moved)


def _forward_logits(model, rgb, aux, instance_mask):
    """One model call in the model's own convention.

    Without a mask this is exactly the v1 call (``model(rgb, aux)`` for the
    dual variants, ``model(rgb)`` for the single stream), so a v1 model or a
    test stub keeps working unchanged; the third argument appears only when
    the run really pools with an instance mask.
    """
    if instance_mask is None:
        return model(rgb, aux) if model.is_dual else model(rgb)
    return model(rgb, aux, instance_mask)


def _gradients_finite(model) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    accum: int,
    limit_batches: int,
    stage: str,
    with_instance_mask: bool = False,
) -> dict:
    """One training epoch with stable accumulation groups.

    Each group is processed as a whole: the group's sample count ``N_g``
    is known before any backward, so every micro-batch contributes
    ``loss_i * n_i / N_g`` — exactly the mean loss of the group's samples,
    independent of how the samples were split into micro-batches. The
    last (possibly short) group is always stepped, so no gradient is ever
    left unapplied, and the epoch loss is the sample-weighted mean over
    every processed sample.
    """
    apply_stage(model, stage)
    criterion = nn.CrossEntropyLoss()
    total_batches = len(loader)
    if limit_batches:
        total_batches = min(total_batches, int(limit_batches))
    groups = accumulation_groups(total_batches, int(accum))

    batch_iterator = iter(loader)
    optimizer_steps = 0
    samples_seen = 0
    loss_numerator = 0.0
    gradients_finite = True
    for start, end in groups:
        group_batches = [next(batch_iterator) for _ in range(end - start)]
        unpacked = [
            _unpack_batch(batch, model.is_dual, device, with_instance_mask)
            for batch in group_batches
        ]
        group_samples = sum(int(labels.shape[0]) for *_, labels, _ in unpacked)
        optimizer.zero_grad(set_to_none=True)
        for rgb, aux, quality, labels, instance_mask in unpacked:
            logits = _forward_logits(model, rgb, aux, instance_mask)
            loss = criterion(logits, labels)
            weight = labels.shape[0] / group_samples
            (loss * weight).backward()
            loss_numerator += float(loss.detach()) * int(labels.shape[0])
        gradients_finite = gradients_finite and _gradients_finite(model)
        optimizer.step()
        optimizer_steps += 1
        samples_seen += group_samples
    optimizer.zero_grad(set_to_none=True)
    return {
        "batches": total_batches,
        "optimizer_steps": optimizer_steps,
        "samples": samples_seen,
        "mean_loss": loss_numerator / max(samples_seen, 1),
        "gradients_finite": gradients_finite,
    }


@torch.no_grad()
def evaluate_model(
    model, loader, device: str, limit_batches: int = 0,
    with_instance_mask: bool = False,
) -> dict:
    model.eval()
    preds: list[int] = []
    labels_all: list[int] = []
    total_batches = len(loader)
    if limit_batches:
        total_batches = min(total_batches, int(limit_batches))
    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        rgb, aux, quality, labels, instance_mask = _unpack_batch(
            batch, model.is_dual, device, with_instance_mask
        )
        logits = _forward_logits(model, rgb, aux, instance_mask)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return classification_metrics(labels_all, preds, model.num_classes)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _run_dir(run_id: str) -> Path:
    """This run's output directory under the (injectable) run root."""
    return Path(RUNS_ROOT) / run_id / RUN_SUBDIR


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_snapshot_target(snapshot_dir: str | Path, run_dir: str | Path) -> Path:
    """Canonical, safe stage-snapshot target; refuses anything existing.

    The path is resolved to a canonical absolute path first (aliases like
    ``a/../b`` cannot bypass the checks). The project root and every
    descendant of the ``datasets``, ``weights`` and ``runs`` trees are
    refused (containment is decided by resolved path relationships, never
    string prefixes), except a subdirectory of this run's own new
    directory. An existing file, empty directory or non-empty directory is
    refused without being touched.
    """
    target = Path(snapshot_dir)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target = target.resolve()
    run_dir = Path(run_dir).resolve()
    project_root = PROJECT_ROOT.resolve()

    if target == run_dir:
        raise ValueError(
            f"stage snapshot target {target} is the run directory itself; use "
            "a subdirectory (e.g. <run-dir>/stage_snapshots)"
        )
    if not _is_within(target, run_dir):
        # Containment by resolved path relationships, never string prefixes:
        # the whole datasets/weights/runs trees are protected (the project
        # root itself is refused exactly, not as a subtree).
        if target == project_root:
            raise ValueError(
                f"stage snapshot target {target} is the project root itself; "
                "refusing"
            )
        protected_trees = [
            ("datasets", (project_root / "datasets").resolve()),
            ("weights", (project_root / "weights").resolve()),
            ("runs", (project_root / "runs").resolve()),
        ]
        for name, root in protected_trees:
            if target == root or _is_within(target, root):
                raise ValueError(
                    f"stage snapshot target {target} is inside the protected "
                    f"{name} tree ({root}); refusing to write evidence there"
                )
    if target.exists():
        kind = "file" if target.is_file() else "directory"
        raise FileExistsError(
            f"stage snapshot target {target} already exists ({kind}); refusing "
            "to overwrite existing evidence — choose a new directory"
        )
    return target


def create_snapshot_dir(snapshot_dir: str | Path, run_dir: str | Path) -> Path:
    """Create the snapshot directory with exclusive semantics.

    The leaf is created with ``exist_ok=False`` so a target created
    between the pre-check and creation (race) fails instead of being
    reused; existing content is never deleted, cleared or renamed.
    """
    target = validate_snapshot_target(snapshot_dir, run_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    return target


def write_stage_snapshot(path: str | Path, model, epoch: int, stage: str) -> Path:
    """Observation hook: one state dict of the live production model.

    Written exclusively (``xb``): an existing snapshot file is never
    overwritten.
    """
    path = Path(path)
    payload = {
        "state_dict": {key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()},
        "epoch": int(epoch),
        "stage": stage,
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return path


def write_stage_metadata(path: str | Path, payload: dict) -> Path:
    """Write the evidence metadata file exclusively (never overwritten)."""
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path


def build_stage_evidence(
    *,
    run_id: str,
    variant: str,
    seed: int,
    data_fingerprint: str,
    imgsz: int,
    preprocessing_version: str,
    upstream_commit: str,
    pretrained_sha256: str,
    epochs: int,
    warmup_epochs: int,
    smoke: bool,
    limit_batches: int,
    physical_batch: int,
    accum_steps: int,
    effective_batch: int,
    snapshot_dir: Path,
    run_dir: Path,
    best_epoch: int,
    best_stage: str,
    last_stage: str,
) -> dict:
    """Assemble the v2 stage-evidence metadata (identity + digests).

    Digests cover the snapshot files and the checkpoint files this
    evidence is associated with; checkpoints never reference the evidence,
    so there is no circular dependency.
    """
    from models.material_fusion import (
        SNAPSHOT_FILENAMES,
        SNAPSHOT_STAGES,
        STAGE_SNAPSHOT_FORMAT,
        STAGE_SNAPSHOT_FORMAT_VERSION,
    )

    snapshot_epochs = {
        "init": 0,
        "post_warmup": warmup_epochs,
        "post_finetune": epochs,
    }
    snapshots = {}
    for role, filename in SNAPSHOT_FILENAMES.items():
        path = snapshot_dir / filename
        if not path.is_file():
            continue
        snapshots[role] = {
            "file": filename,
            "epoch": snapshot_epochs[role],
            "stage": SNAPSHOT_STAGES[role],
            "sha256": _sha256_of(path),
        }
    return {
        "format": STAGE_SNAPSHOT_FORMAT,
        "format_version": STAGE_SNAPSHOT_FORMAT_VERSION,
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "data_fingerprint": data_fingerprint,
        "imgsz": imgsz,
        "preprocessing_version": preprocessing_version,
        "upstream_commit": upstream_commit,
        "pretrained_sha256": pretrained_sha256,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "smoke": smoke,
        "limit_batches": limit_batches,
        "physical_batch": physical_batch,
        "accum_steps": accum_steps,
        "effective_batch": effective_batch,
        "snapshots": snapshots,
        "checkpoints": {
            "last": {
                "file": str(run_dir / "last.pt"),
                "epoch": epochs,
                "stage": last_stage,
                "sha256": _sha256_of(run_dir / "last.pt"),
            },
            "best": {
                "file": str(run_dir / "best.pt"),
                "epoch": best_epoch,
                "stage": best_stage,
                "sha256": _sha256_of(run_dir / "best.pt"),
            },
        },
    }


def _build_loaders(
    data_root: Path, variant: str, imgsz: int, batch: int, workers: int, seed: int,
    with_instance_mask: bool = False,
):
    train_paths, _, _ = read_manifest_split(data_root, "train")
    val_paths, _, _ = read_manifest_split(data_root, "val")
    train_loader = torch.utils.data.DataLoader(
        MaterialFusionDataset(
            train_paths, variant=variant, imgsz=imgsz,
            with_instance_mask=with_instance_mask,
        ),
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = torch.utils.data.DataLoader(
        MaterialFusionDataset(
            val_paths, variant=variant, imgsz=imgsz,
            with_instance_mask=with_instance_mask,
        ),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
    )
    return train_loader, val_loader


def run_training(args: argparse.Namespace) -> Path:
    """Full training loop. Formal runs execute the fixed 30-epoch budget."""
    template = load_experiment_config(args.config)
    config = resolve_config(template, args)
    pooling_mode = str(config["pooling_mode"])
    with_instance_mask = pooling_mode == POOLING_INSTANCE_MASK
    pooling_fields = {
        "version": str(config["experiment"]),
        "pooling_mode": pooling_mode,
        "mask_schema": str(config["mask_schema"]),
        "mask_resize_version": str(config["mask_resize_version"]),
        "parent_data_fingerprint": str(config["parent_data_fingerprint"]),
    }
    mask_kwargs = {"with_instance_mask": True} if with_instance_mask else {}
    # Cross-protocol weight continuation is refused here, before any output
    # directory, dataset read or optimizer exists: the v2 fair comparison must
    # initialise from the same official pretrained weights as v1, so no v2 run
    # may inherit another run's training history.
    if with_instance_mask and args.init_from:
        raise ValueError(
            "--init-from is not allowed for the material_fusion_v2_mask protocol: "
            "the v2 comparison initialises from the official pretrained weights "
            "only, so no run inherits a different training history"
        )
    seed = int(config["seed"])
    _set_seed(seed)

    run_dir = _run_dir(args.run_id)
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use a new --run-id "
            "instead of overwriting a previous run"
        )
    # The snapshot target is validated before any training output exists,
    # so a refusal leaves no new run directory behind.
    snapshot_target = (
        validate_snapshot_target(args.stage_snapshot_dir, run_dir)
        if args.stage_snapshot_dir
        else None
    )

    data_root = _resolve_project_path(config["data"])
    if with_instance_mask:
        # The V5 gate re-verifies the recorded digests, re-decodes every mask,
        # recomputes the mask content digest, re-reads the annotation sources
        # and pins the V4 parent state; the protocol admission additionally
        # refuses an empty fingerprint, so a run can never be admitted against
        # "any V5 dataset". An unaudited or tampered V5 dataset never reaches
        # the model.
        dataset_report = admit_dataset_for_protocol(
            data_root,
            pooling_mode=pooling_mode,
            expected_fingerprint=config["data_fingerprint"],
            expected_parent_fingerprint=config["parent_data_fingerprint"],
            expected_mask_schema=config["mask_schema"],
            expected_mask_resize_version=config["mask_resize_version"],
        )
        fingerprint = dataset_report["audit_fingerprint"]
    else:
        dataset_report = validate_dataset(data_root)
        fingerprint = dataset_report.get("audit_fingerprint", "")
        if config["data_fingerprint"] and config["data_fingerprint"] != fingerprint:
            raise ValueError(
                f"dataset fingerprint {fingerprint} does not match the config's "
                f"recorded {config['data_fingerprint']}; refusing to train on a "
                "different dataset state"
            )

    device = torch_device_name(resolve_device(args.device))
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    variant = validate_variant(config["variant"])
    imgsz = int(config["imgsz"])

    pretrained_path = _resolve_project_path(config["pretrained"])
    model = build_material_model(variant, imgsz=imgsz, pooling_mode=pooling_mode)
    pretrained_report = load_pretrained_into_model(
        model, pretrained_path, expected_sha256=config["pretrained_sha256"]
    )
    pretrained_sha256 = _sha256_of(pretrained_path)

    init_from = ""
    if args.init_from:
        init_path = _resolve_project_path(args.init_from)
        init_metadata = read_material_checkpoint_metadata(init_path)
        if init_metadata.variant != variant:
            raise ValueError(
                f"--init-from checkpoint variant {init_metadata.variant!r} does not "
                f"match the requested variant {variant!r}"
            )
        if init_metadata.imgsz != imgsz:
            raise ValueError(
                f"--init-from checkpoint imgsz {init_metadata.imgsz} does not match "
                f"the requested imgsz {imgsz}"
            )
        # Continuation stays inside one protocol. Identical parameter shapes do
        # not make a GAP checkpoint and a mask-pooling model interchangeable,
        # and a v2 checkpoint may never be continued into a v1 run.
        if init_metadata.pooling_mode != pooling_mode:
            raise ValueError(
                f"--init-from checkpoint pooling_mode "
                f"{init_metadata.pooling_mode!r} does not match this run's "
                f"{pooling_mode!r}; refusing to continue across pooling protocols"
            )
        if init_metadata.version != config["experiment"]:
            raise ValueError(
                f"--init-from checkpoint version {init_metadata.version!r} does not "
                f"match this run's protocol {config['experiment']!r}"
            )
        init_model, _, _ = load_material_checkpoint(
            init_path, variant=variant, imgsz=imgsz, pooling_mode=pooling_mode
        )
        model.load_state_dict(init_model.state_dict())
        init_from = str(init_path)

    model = model.to(device)
    optimizer = build_optimizer(
        model, lr=config["lr"], encoder_lr=config["encoder_lr"],
        weight_decay=config["weight_decay"],
    )
    train_loader, val_loader = _build_loaders(
        data_root, variant, imgsz, int(config["batch"]), int(config["workers"]), seed,
        with_instance_mask=with_instance_mask,
    )

    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "metrics_history.jsonl"
    best_key = (-1.0, -1.0)
    best_path = run_dir / "best.pt"
    warmup_epochs = int(config["warmup_epochs"])
    epochs = int(config["epochs"])
    limit_batches = int(config["limit_batches"])
    started = time.time()

    if snapshot_target is not None:
        # Exclusive creation; a target that appeared after the pre-check
        # (race) fails here instead of being reused or overwritten.
        snapshot_target = create_snapshot_dir(snapshot_target, run_dir)
        write_stage_snapshot(
            snapshot_target / "init.pt", model, epoch=0, stage="init"
        )

    best_metrics: dict | None = None
    best_epoch = 0
    best_stage = ""
    last_entry: dict | None = None
    for epoch in range(epochs):
        stage = stage_for_epoch(epoch, warmup_epochs)
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            accum=int(config["accum_steps"]), limit_batches=limit_batches, stage=stage,
            **mask_kwargs,
        )
        if not train_stats["gradients_finite"]:
            raise RuntimeError(
                f"epoch {epoch + 1}: non-finite gradients detected; refusing to "
                "continue training"
            )
        val_metrics = evaluate_model(
            model, val_loader, device, limit_batches, **mask_kwargs
        )
        entry = {
            "epoch": epoch + 1,
            "stage": stage,
            "train_batches": train_stats["batches"],
            "optimizer_steps": train_stats["optimizer_steps"],
            "samples": train_stats["samples"],
            "gradients_finite": train_stats["gradients_finite"],
            "train_loss": train_stats["mean_loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "per_class_recall": val_metrics["per_class_recall"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(
            f"epoch {epoch + 1}/{epochs} stage={stage} "
            f"loss={entry['train_loss']:.4f} "
            f"val_macro_f1={entry['val_macro_f1']:.4f} "
            f"val_acc={entry['val_accuracy']:.4f}"
        )
        last_entry = entry
        if snapshot_target is not None and stage == "warmup" and epoch + 1 == warmup_epochs:
            write_stage_snapshot(
                snapshot_target / "post_warmup.pt", model,
                epoch=epoch + 1, stage="warmup",
            )
        if snapshot_target is not None and stage == "finetune" and epoch + 1 == epochs:
            write_stage_snapshot(
                snapshot_target / "post_finetune.pt", model,
                epoch=epoch + 1, stage="finetune",
            )
        new_key = selection_key(val_metrics)
        if should_replace_best(best_key, new_key):
            best_key = new_key
            best_metrics = val_metrics
            best_epoch = epoch + 1
            best_stage = stage
            save_material_checkpoint(
                best_path,
                model,
                material_fusion_metadata(
                    variant=variant,
                    imgsz=imgsz,
                    data_fingerprint=fingerprint,
                    pretrained_sha256=pretrained_sha256,
                    seed=seed,
                    epoch=epoch + 1,
                    stage=stage,
                    smoke=limit_batches > 0,
                    limit_batches=limit_batches,
                    val_macro_f1=val_metrics["macro_f1"],
                    val_accuracy=val_metrics["accuracy"],
                    physical_batch=int(config["batch"]),
                    accum_steps=int(config["accum_steps"]),
                    effective_batch=int(config["effective_batch"]),
                    drop_path_rate=float(model.drop_path_rate),
                    init_from=init_from,
                    fallback_weights=config["fallback_weights"],
                    fallback_identity=str(config.get("fallback_identity", "")),
                    **pooling_fields,
                ),
            )
            _notify_saved(best_path, "best", epoch + 1, model)

    # last.pt always carries the actual final epoch's metrics, never the
    # best-selection metrics (the best epoch may be an earlier one).
    save_material_checkpoint(
        run_dir / "last.pt",
        model,
        material_fusion_metadata(
            variant=variant,
            imgsz=imgsz,
            data_fingerprint=fingerprint,
            pretrained_sha256=pretrained_sha256,
            seed=seed,
            epoch=epochs,
            stage=stage_for_epoch(epochs - 1, warmup_epochs),
            smoke=limit_batches > 0,
            limit_batches=limit_batches,
            val_macro_f1=last_entry["val_macro_f1"],
            val_accuracy=last_entry["val_accuracy"],
            physical_batch=int(config["batch"]),
            accum_steps=int(config["accum_steps"]),
            effective_batch=int(config["effective_batch"]),
            drop_path_rate=float(model.drop_path_rate),
            init_from=init_from,
            fallback_weights=config["fallback_weights"],
            fallback_identity=str(config.get("fallback_identity", "")),
            **pooling_fields,
        ),
    )
    _notify_saved(run_dir / "last.pt", "last", epochs, model)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "config_path": template.get("_config_path", ""),
                "config": config,
                "variant": variant,
                "data": str(data_root),
                "dataset_fingerprint": fingerprint,
                "pooling_mode": pooling_mode,
                "mask_schema": str(config["mask_schema"]),
                "mask_resize_version": str(config["mask_resize_version"]),
                "parent_data_fingerprint": str(config["parent_data_fingerprint"]),
                "data_report": dataset_report,
                "device": device,
                "pretrained": str(pretrained_path),
                "pretrained_sha256": pretrained_sha256,
                "pretrained_load": pretrained_report,
                "init_from": init_from,
                "param_groups": optimizer_group_report(model, optimizer),
                "best": {
                    "epoch": best_epoch,
                    "stage": best_stage,
                    "val_macro_f1": best_key[0],
                    "val_accuracy": best_key[1],
                },
                "last": {
                    "epoch": epochs,
                    "stage": last_entry["stage"],
                    "val_macro_f1": last_entry["val_macro_f1"],
                    "val_accuracy": last_entry["val_accuracy"],
                },
                # The actual protocol of this run. The four A1 smoke
                # run_configs written before this fix carry the v1 string here
                # while their validated inner "config.experiment" is already
                # correct; no reader uses this top-level field, and the
                # historical files are deliberately left untouched.
                "version": str(config["experiment"]),
                "torch_version": torch.__version__,
                "elapsed_seconds": time.time() - started,
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if device.startswith("cuda")
                    else 0.0
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if snapshot_target is not None:
        # last.pt/best.pt are final at this point; the evidence records their
        # digests and identity so verification can bind checkpoint <-> evidence.
        write_stage_metadata(
            snapshot_target / "stage_snapshots.json",
            build_stage_evidence(
                run_id=args.run_id,
                variant=variant,
                seed=seed,
                data_fingerprint=fingerprint,
                imgsz=imgsz,
                preprocessing_version=config["preprocessing_version"],
                upstream_commit=config["upstream_commit"],
                pretrained_sha256=pretrained_sha256,
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                smoke=limit_batches > 0,
                limit_batches=limit_batches,
                physical_batch=int(config["batch"]),
                accum_steps=int(config["accum_steps"]),
                effective_batch=int(config["effective_batch"]),
                snapshot_dir=snapshot_target,
                run_dir=run_dir,
                best_epoch=best_epoch,
                best_stage=best_stage,
                last_stage=stage_for_epoch(epochs - 1, warmup_epochs),
            ),
        )

    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_run_id(args.run_id)
    except InvalidRunIdError as exc:
        print(f"invalid run id: {exc}", file=sys.stderr)
        return 2
    try:
        run_dir = run_training(args)
    except DeviceUnavailableError as exc:
        print(f"device unavailable: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"training refused: {exc}", file=sys.stderr)
        return 2
    print(f"Material fusion run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
