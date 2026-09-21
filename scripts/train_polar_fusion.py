"""Train the Polar Fusion classification model on the band-matched fusion
dataset (V4).

Phases:
    freeze : train only the polar delta head and the quality gate on top of
             a frozen gray backbone (default, first stage).
    joint  : optional follow-up finetune with the gray backbone unfrozen at
             a lower learning rate.

The gray backbone is the Ultralytics classification model (default base
``yolo26n-cls.pt``) loaded from a local checkpoint. If the base checkpoint
is missing the run is refused with a clear message (no network download).

``--dry-run`` validates the dataset (manifest counts, sample decoding,
quality vector shape) and the fusion structure (delta/gate forward shapes,
forced gate behavior) without training and without creating run outputs.

Usage:
    python scripts/train_polar_fusion.py --run-id run_x --dry-run
    python scripts/train_polar_fusion.py --run-id run_x --phase freeze
    python scripts/train_polar_fusion.py --run-id run_x --phase joint
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import (
    QUALITY_VECTOR_LENGTH,
    dataset_fingerprint,
    load_fusion_sample,
    verify_dataset_integrity,
)
from models.polar_fusion import (
    DEFAULT_BASE_MODEL,
    FUSION_VERSION,
    FusionCheckpointMetadata,
    FusionClsDataset,
    GrayBackboneAdapter,
    PolarFusionModel,
    architecture_name,
    build_gray_class_perm,
    class_names_from_manifest,
    fusion_metadata,
    load_fusion_checkpoint,
    prepare_gray_backbone,
    read_manifest_split,
    rebuild_gray_backbone,
    save_fusion_checkpoint,
    torch_device_name,
)
from scripts.training_common import DeviceUnavailableError, InvalidRunIdError, resolve_device, validate_run_id

DEFAULT_SEED = 2026
DEFAULT_IMGSZ = 224
DEFAULT_DATA = "datasets/underwater_cls_fusion_v4_band"
DEFAULT_PHASE = "freeze"
PHASES = ("freeze", "joint")
# No default gray weights: the formal gray branch must come from a trained
# YOLO26 gray Model B passed explicitly via --gray-weights. The legacy
# YOLOv8 weights are comparison-only and additionally require
# --legacy-gray-weights.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id grouping outputs under runs/train/<run-id>/polar_fusion.",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE_MODEL,
        help=f"Gray backbone architecture base checkpoint (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--gray-weights",
        default=None,
        help="Trained YOLO26 gray Model B checkpoint initializing the gray "
        "branch (required for formal freeze training). Pass an empty string "
        "together with --allow-untrained-gray to start from a freshly "
        "replaced classification head.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Fusion checkpoint to continue from (required for --phase joint "
        "unless freeze/best.pt exists under the same run id).",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="V3 fusion dataset root (default: %(default)s).",
    )
    parser.add_argument("--device", default="0", help="Training device.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input size. New runs default to 224; resume runs inherit the "
        "init checkpoint's imgsz unless explicitly passed (a different "
        "explicit size is refused).",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate.")
    parser.add_argument(
        "--gray-lr", type=float, default=1e-5, help="Gray backbone lr (joint phase)."
    )
    parser.add_argument("--phase", choices=PHASES, default=DEFAULT_PHASE)
    parser.add_argument(
        "--lambda-gray",
        type=float,
        default=1.0,
        help="Weight of the gray-branch CE auxiliary loss "
        "L = CE(fusion) + lambda_gray*CE(gray) + lambda_kd*KL(gray||teacher).",
    )
    parser.add_argument(
        "--lambda-kd",
        type=float,
        default=1.0,
        help="Weight of the KL distillation loss against the frozen gray "
        "teacher (joint phase only).",
    )
    parser.add_argument(
        "--allow-untrained-gray",
        action="store_true",
        help="Explicitly allow --phase freeze with a fresh (untrained) gray "
        "head from the base checkpoint. Refused otherwise: the formal gray "
        "branch must come from a trained gray Model B.",
    )
    parser.add_argument(
        "--legacy-gray-weights",
        action="store_true",
        help="Explicitly allow non-YOLO26 gray weights (e.g. the legacy "
        "yolov8n-cls Model B) for comparison runs. Formal training "
        "requires YOLO26 gray weights.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Verification helper: cap train/val batches per epoch "
        "(0 = all batches).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and structure only; no training, no run outputs.",
    )
    return parser.parse_args(argv)


def validate_dataset(data_root: Path, require_audit: bool = True) -> dict:
    """Dataset gate for dry-run and training: counts, decoding, quality shape.

    Formal training additionally requires the dataset's own audit report
    (``dataset_audit.json`` written by the dataset generation/audit pass)
    to exist and to have passed; the report's fingerprint is returned so
    runs can record which dataset state they were trained on.
    """
    manifest = data_root / "dataset_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest}")

    class_names = class_names_from_manifest(data_root)
    if len(class_names) < 2:
        raise ValueError(f"need >= 2 classes, found {class_names}")

    report: dict = {"class_names": class_names, "splits": {}}
    for split in ("train", "val", "test"):
        paths, class_ids, _ = read_manifest_split(data_root, split)
        report["splits"][split] = len(paths)
        # Decode the first two samples of every split.
        for path in paths[:2]:
            sample = load_fusion_sample(path)
            if sample.quality.shape != (QUALITY_VECTOR_LENGTH,):
                raise ValueError(f"bad quality vector in {path}")
            if int(sample.class_id) not in set(class_ids):
                raise ValueError(f"class id {sample.class_id} out of range in {path}")
    if report["splits"].get("train", 0) == 0 or report["splits"].get("val", 0) == 0:
        raise ValueError("train and val splits must both be non-empty")

    audit_path = data_root / "dataset_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        report["audit_passed"] = bool(audit.get("audit_passed", False))
        if require_audit:
            if not report["audit_passed"]:
                raise ValueError(
                    f"dataset audit did not pass ({audit_path}); regenerate or "
                    "re-audit the dataset before training"
                )
            # The audit verdict alone proves nothing about the current file
            # contents; re-verify manifest/summary/per-file digests first,
            # then record the fingerprint of the verified state.
            integrity = verify_dataset_integrity(data_root)
            report["integrity_verified_files"] = integrity["verified_files"]
            report["audit_fingerprint"] = dataset_fingerprint(data_root)
        else:
            report["integrity_verified_files"] = None
            try:
                report["audit_fingerprint"] = dataset_fingerprint(data_root)
            except (FileNotFoundError, ValueError):
                # Old-format reports cannot be fingerprinted without a
                # re-audit; dry-run only reports, it does not enforce.
                report["audit_fingerprint"] = ""
    elif require_audit:
        raise FileNotFoundError(
            f"missing dataset audit report: {audit_path}; run "
            "scripts/prepare_cls_fusion_dataset.py --audit-only first"
        )
    else:
        report["audit_passed"] = None
    return report


class _StructureStubBackbone(nn.Module):
    """Minimal stand-in backbone for structure validation without weights."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes)
        )

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        return self.net(gray)


def structure_check(num_classes: int, imgsz: int) -> dict:
    """Validate delta/gate/fusion shapes and the forced-gate behavior."""
    torch.manual_seed(0)
    model = PolarFusionModel(_StructureStubBackbone(num_classes), num_classes)
    model.eval()
    batch = 3
    gray = torch.rand(batch, 3, imgsz, imgsz)
    polar = torch.rand(batch, 3, imgsz, imgsz)
    quality = torch.rand(batch, QUALITY_VECTOR_LENGTH)
    with torch.no_grad():
        out = model(gray, polar, quality)
    assert out["final_logits"].shape == (batch, num_classes)
    assert out["gray_logits"].shape == (batch, num_classes)
    assert out["polar_delta"].shape == (batch, num_classes)
    assert out["gate"].shape == (batch, 1)
    assert torch.all((out["gate"] >= 0) & (out["gate"] <= 1))

    # valid_ratio == 0 forces the gate to exactly 0 -> pure gray logits.
    zero_quality = quality.clone()
    zero_quality[:, 0] = 0.0
    with torch.no_grad():
        gated = model(gray, polar, zero_quality)
    assert torch.all(gated["gate"] == 0.0)
    assert torch.equal(gated["final_logits"], gated["gray_logits"])

    # Explicit invalid flag (sync/matching failure) also forces gate 0.
    with torch.no_grad():
        flagged = model(gray, polar, quality, polar_invalid=torch.ones(batch))
    assert torch.all(flagged["gate"] == 0.0)
    return {"num_classes": num_classes, "imgsz": imgsz, "forced_gate_verified": True}


def dry_run(args: argparse.Namespace) -> int:
    data_root = PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    imgsz = int(args.imgsz) if args.imgsz is not None else DEFAULT_IMGSZ
    # Dry-run reports audit status without enforcing it; formal training
    # (run_training) hard-requires a passed audit.
    dataset_report = validate_dataset(data_root, require_audit=False)
    structure_report = structure_check(len(dataset_report["class_names"]), imgsz)
    base_path = PROJECT_ROOT / args.base
    base_present = base_path.is_file()
    gray_weights_present = bool(args.gray_weights) and (PROJECT_ROOT / args.gray_weights).is_file()
    joint_default_init = (
        PROJECT_ROOT / "runs" / "train" / args.run_id
        / "polar_fusion" / "freeze" / "best.pt"
    ).is_file()
    report = {
        "dry_run": True,
        "version": FUSION_VERSION,
        "data_root": str(data_root),
        "dataset": dataset_report,
        "dataset_audit_passed": dataset_report.get("audit_passed"),
        "dataset_fingerprint": dataset_report.get("audit_fingerprint", ""),
        "structure": structure_report,
        "imgsz": imgsz,
        "base_model": args.base,
        "base_checkpoint_present": base_present,
        "base_checkpoint_path": str(base_path),
        "gray_weights": args.gray_weights,
        "gray_weights_present": gray_weights_present,
        "joint_default_init_present": joint_default_init,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not base_present and not args.gray_weights:
        print(
            f"NOTE: base checkpoint {args.base} is not present locally; "
            "the fresh-head build (--gray-weights '') requires it "
            "(no automatic download).",
            file=sys.stderr,
        )
        print(
            "NOTE: with pretrained --gray-weights the base checkpoint is "
            "not needed (the loaded weights carry their own architecture).",
            file=sys.stderr,
        )
    if not args.gray_weights:
        print(
            "NOTE: no --gray-weights provided; formal freeze training "
            "requires a trained YOLO26 gray Model B checkpoint (pass "
            "--gray-weights, or --allow-untrained-gray to start untrained).",
            file=sys.stderr,
        )
    if args.gray_weights and not gray_weights_present:
        print(
            f"NOTE: --gray-weights {args.gray_weights} is not present locally; "
            "training would refuse rather than fall back to an untrained head.",
            file=sys.stderr,
        )
    return 0


@torch.no_grad()
def _verify_backbone_output(backbone: GrayBackboneAdapter, imgsz: int, num_classes: int, device: str) -> None:
    """Fail fast on eval-mode interface or head-width problems."""
    backbone.eval()
    probe = torch.zeros(2, 3, imgsz, imgsz, device=device)
    out = backbone(probe)
    if not isinstance(out, torch.Tensor) or out.shape != (2, num_classes):
        raise RuntimeError(
            f"gray backbone produced {type(out).__name__} {getattr(out, 'shape', None)}; "
            f"expected tensor ({2}, {num_classes})"
        )


def classification_metrics(
    labels: list[int], preds: list[int], num_classes: int
) -> dict:
    """Accuracy, macro-F1 and per-class recall for one split.

    Macro-F1 is the primary model-selection metric: with the imbalanced
    V3/V4 val sets, accuracy is dominated by the majority classes and can
    hide a collapsed minority class (``real_fish``).
    """
    total = len(labels)
    accuracy = sum(1 for l, p in zip(labels, preds) if l == p) / max(total, 1)
    f1s = []
    recalls = {}
    for cls in range(num_classes):
        tp = sum(1 for l, p in zip(labels, preds) if l == cls and p == cls)
        fp = sum(1 for l, p in zip(labels, preds) if l != cls and p == cls)
        fn = sum(1 for l, p in zip(labels, preds) if l == cls and p != cls)
        class_total = tp + fn
        recalls[cls] = tp / class_total if class_total else 0.0
        if tp == 0:
            f1s.append(0.0)
        else:
            precision = tp / (tp + fp)
            recall = tp / class_total
            f1s.append(2 * precision * recall / (precision + recall))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "per_class_recall": {str(cls): recalls[cls] for cls in recalls},
    }


@torch.no_grad()
def _evaluate(model: PolarFusionModel, loader, device: str, limit_batches: int = 0) -> dict:
    """Fusion-branch metrics over a loader (final_logits predictions)."""
    model.eval()
    labels_all: list[int] = []
    preds: list[int] = []
    for batch_index, (gray, polar, quality, labels) in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break
        out = model(gray.to(device), polar.to(device), quality.to(device))
        preds.extend(out["final_logits"].argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.tolist())
    num_classes = model.num_classes
    return classification_metrics(labels_all, preds, num_classes)


def _fusion_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    criterion: nn.Module,
    lambda_gray: float,
    lambda_kd: float,
) -> torch.Tensor:
    """L = CE(fusion, y) + lambda_gray*CE(gray, y) + lambda_kd*KL(gray||teacher).

    The gray CE term keeps the gray branch a competent standalone
    classifier; the KL term pins the (unfrozen) gray branch to the frozen
    pre-joint teacher so joint finetuning cannot drift it away.
    """
    loss = criterion(out["final_logits"], labels)
    gray_logits = out["gray_logits"]
    loss = loss + lambda_gray * criterion(gray_logits, labels)
    if teacher_logits is not None and lambda_kd > 0:
        kd = nn.functional.kl_div(
            nn.functional.log_softmax(gray_logits, dim=-1),
            nn.functional.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        loss = loss + lambda_kd * kd
    return loss


def _phase_run_dir(run_id: str, phase: str) -> Path:
    """Per-phase output directory; refuses to overwrite existing runs."""
    run_dir = PROJECT_ROOT / "runs" / "train" / run_id / "polar_fusion" / phase
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; use a new --run-id "
            "instead of overwriting a previous phase"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_init_checkpoint(args: argparse.Namespace) -> Path | None:
    """Checkpoint to continue from; joint defaults to the freeze-phase best.

    Without this, a follow-up joint run would rebuild from the base and
    silently discard everything the freeze phase learned.
    """
    if args.init_from:
        path = Path(args.init_from)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"--init-from checkpoint not found: {path}")
        return path
    if args.phase == "joint":
        default = (
            PROJECT_ROOT / "runs" / "train" / args.run_id
            / "polar_fusion" / "freeze" / "best.pt"
        )
        if default.is_file():
            return default
        raise FileNotFoundError(
            "--phase joint needs a freeze-phase checkpoint: pass --init-from "
            f"or train the freeze phase first (expected {default})"
        )
    return None


def _apply_phase_modes(model: PolarFusionModel, joint: bool) -> None:
    """Epoch-start module modes for the requested phase.

    In the freeze phase the gray backbone must stay in eval mode while the
    heads train: ``model.train()`` alone would flip BatchNorm statistics
    back to running updates even with ``requires_grad=False``, so the
    "frozen gray" would silently drift.
    """
    model.train()
    if not joint:
        model.gray_backbone.eval()


def _restore_backbone(
    init_payload: dict, class_names: list[str], device: str
) -> GrayBackboneAdapter:
    """Rebuild the gray branch exactly as the init checkpoint was built."""
    metadata = FusionCheckpointMetadata(
        version="",
        class_names=tuple(class_names),
        input_format="",
        base_model=str(init_payload.get("base_model", DEFAULT_BASE_MODEL)),
        quality_vector_keys=(),
        imgsz=0,
        architecture=str(init_payload.get("architecture", "")),
        gray_weights=str(init_payload.get("gray_weights", "")),
        head_replaced=bool(init_payload.get("head_replaced", False)),
    )
    # Weights are then overwritten by the checkpoint state dict.
    return rebuild_gray_backbone(metadata, device)


def _enforce_architecture_gate(
    backbone: GrayBackboneAdapter,
    recorded_architecture: str,
    legacy_allowed: bool,
    context: str,
) -> str:
    """Unified YOLO26/legacy admission for fresh and resumed training.

    The architecture is identified from the actually loaded/restored gray
    backbone (its yaml), never from a requested name. An unidentifiable
    backbone, or a recorded architecture that contradicts the actual one,
    is refused instead of guessed. Non-YOLO26 branches require the explicit
    ``--legacy-gray-weights`` comparison flag. Must run before the training
    output directory is created and before any optimizer update.
    """
    actual = architecture_name(backbone.module)
    if not actual:
        raise ValueError(
            f"cannot identify the gray backbone architecture ({context}); "
            "refusing to train without a verified architecture (no guessing)"
        )
    if recorded_architecture and recorded_architecture != actual:
        raise ValueError(
            f"recorded architecture {recorded_architecture!r} does not match "
            f"the actual gray backbone {actual!r} ({context}); refusing"
        )
    if not actual.startswith("yolo26") and not legacy_allowed:
        raise ValueError(
            f"gray backbone architecture {actual!r} is not YOLO26; formal "
            "training requires a trained YOLO26 gray Model B. Pass "
            "--legacy-gray-weights to use legacy weights for explicit "
            "comparison runs only."
        )
    return actual


def _resolve_imgsz(requested: int | None, checkpoint_imgsz: int | None) -> int:
    """One resolved input size for loaders, forward checks and metadata.

    New runs default to 224. Resume runs inherit the init checkpoint's
    imgsz; an explicit different size is refused (stating both sizes)
    because changing the preprocessing size would silently invalidate the
    checkpoint's contract.
    """
    if checkpoint_imgsz is None:
        return int(requested) if requested is not None else DEFAULT_IMGSZ
    if requested is None:
        return int(checkpoint_imgsz)
    if int(requested) != int(checkpoint_imgsz):
        raise ValueError(
            f"--imgsz {int(requested)} does not match the init checkpoint's "
            f"training imgsz {int(checkpoint_imgsz)}; resume runs must keep "
            "the checkpoint input size (omit --imgsz to inherit it)"
        )
    return int(requested)


def _set_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs so a declared seed actually reproduces
    model init and (via the DataLoader generator) shuffle order."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training(args: argparse.Namespace) -> Path:
    """Full training loop (freeze or joint). Not executed by --dry-run."""
    _set_seed(args.seed)
    data_root = PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    dataset_report = validate_dataset(data_root)

    # resolve_device validates CUDA availability; the result is then mapped
    # to a torch-usable device string ("0" -> "cuda:0").
    device = torch_device_name(resolve_device(args.device))
    class_names = class_names_from_manifest(data_root)
    num_classes = len(class_names)

    init_checkpoint = _resolve_init_checkpoint(args)
    if init_checkpoint is None:
        if args.phase == "freeze" and not args.gray_weights and not args.allow_untrained_gray:
            raise ValueError(
                "--phase freeze without --gray-weights is refused: the formal "
                "gray branch must come from a trained YOLO26 gray Model B. "
                "Pass --gray-weights, or --allow-untrained-gray together with "
                "--gray-weights '' to explicitly start untrained."
            )
        gray_weights = args.gray_weights
        base_path = Path(args.base)
        base_path = base_path if base_path.is_absolute() else PROJECT_ROOT / base_path
        # The base checkpoint is only needed for the fresh-head build; with
        # pretrained gray weights the loaded checkpoint carries its own
        # architecture (which may differ from --base, e.g. YOLOv8 legacy).
        if not gray_weights and not base_path.is_file():
            raise FileNotFoundError(
                f"base checkpoint {base_path} is missing; no automatic download"
            )
        backbone, gray_info = prepare_gray_backbone(
            base_path, gray_weights, class_names, device
        )
        # Unified admission: the architecture is read from the actually
        # loaded gray backbone, never from the requested --base/--gray-weights.
        context = (
            f"gray weights {gray_info['gray_weights']}"
            if gray_info["gray_weights"]
            else f"base checkpoint {gray_info['base_model']}"
        )
        _enforce_architecture_gate(backbone, "", args.legacy_gray_weights, context)
        imgsz = _resolve_imgsz(args.imgsz, None)
        _verify_backbone_output(backbone, imgsz, num_classes, device)
        model = PolarFusionModel(backbone, num_classes=num_classes).to(device)
        metadata: FusionCheckpointMetadata = fusion_metadata(
            class_names,
            base_model=gray_info["base_model"],
            imgsz=imgsz,
            architecture=gray_info["architecture"],
            gray_weights=gray_info["gray_weights"],
            gray_class_names=gray_info["gray_class_names"],
            head_replaced=gray_info["head_replaced"],
            class_permutation=tuple(gray_info["perm"]) if gray_info["perm"] else (),
            limit_batches=int(args.limit_batches),
            smoke=args.limit_batches > 0,
        )
    else:
        init_payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        backbone = _restore_backbone(init_payload, class_names, device)
        model, metadata, _ = load_fusion_checkpoint(init_checkpoint, backbone)
        model = model.to(device)
        if list(metadata.class_names) != class_names:
            raise ValueError(
                f"init checkpoint classes {metadata.class_names} != dataset "
                f"classes {tuple(class_names)}"
            )
        # Same admission as a fresh build: identify from the rebuilt
        # backbone; an old checkpoint without a recorded architecture is
        # identified from the actual weights, a contradictory record is
        # refused. Runs before the output dir exists and before any
        # optimizer update.
        _enforce_architecture_gate(
            backbone,
            metadata.architecture,
            args.legacy_gray_weights,
            context=f"init checkpoint {init_checkpoint}",
        )
        imgsz = _resolve_imgsz(args.imgsz, metadata.imgsz)
        _verify_backbone_output(backbone, imgsz, num_classes, device)
        # The new checkpoint records this run's actual truncation state, so
        # config and checkpoint metadata never contradict each other.
        metadata = dataclasses.replace(
            metadata,
            limit_batches=int(args.limit_batches),
            smoke=args.limit_batches > 0,
        )

    # Phase freeze: gray weights are fixed; joint: everything trains, with
    # the backbone at the lower --gray-lr rate.
    joint = args.phase == "joint"
    model.set_gray_frozen(not joint)
    head_params = [
        {"params": model.delta_net.parameters(), "lr": args.lr},
        {"params": model.gate_net.parameters(), "lr": args.lr},
    ]
    if joint:
        head_params.append(
            {"params": model.gray_backbone.parameters(), "lr": args.gray_lr}
        )
    optimizer = torch.optim.AdamW(head_params)
    criterion = nn.CrossEntropyLoss()

    # Joint-phase teacher: a frozen snapshot of the gray branch before any
    # joint update; the KL term keeps the finetuned branch from drifting
    # away from the accepted gray model (issue 7).
    teacher_backbone = None
    if joint:
        import copy

        teacher_backbone = copy.deepcopy(model.gray_backbone).to(device)
        teacher_backbone.eval()
        for param in teacher_backbone.parameters():
            param.requires_grad_(False)

    train_paths, _, _ = read_manifest_split(data_root, "train")
    val_paths, _, _ = read_manifest_split(data_root, "val")
    train_loader = torch.utils.data.DataLoader(
        FusionClsDataset(train_paths, imgsz=imgsz),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0 if sys.platform == "win32" else 8,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = torch.utils.data.DataLoader(
        FusionClsDataset(val_paths, imgsz=imgsz),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 8,
    )

    run_dir = _phase_run_dir(args.run_id, args.phase)
    best_key = (-1.0, -1.0)  # (macro_f1, accuracy)
    best_path = run_dir / "best.pt"
    for epoch in range(args.epochs):
        _apply_phase_modes(model, joint)
        for batch_index, (gray, polar, quality, labels) in enumerate(train_loader):
            if args.limit_batches and batch_index >= args.limit_batches:
                break
            gray = gray.to(device)
            polar = polar.to(device)
            quality = quality.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(gray, polar, quality)
            teacher_logits = (
                teacher_backbone(gray) if teacher_backbone is not None else None
            )
            loss = _fusion_loss(
                out,
                labels,
                teacher_logits,
                criterion,
                args.lambda_gray,
                args.lambda_kd,
            )
            loss.backward()
            optimizer.step()

        val_metrics = _evaluate(model, val_loader, device, args.limit_batches)
        recalls = " ".join(
            f"{name}={val_metrics['per_class_recall'].get(str(i), 0.0):.3f}"
            for i, name in enumerate(class_names)
        )
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} {recalls}"
        )
        # Macro-F1 primary, accuracy tiebreak (issue 8).
        selection_key = (val_metrics["macro_f1"], val_metrics["accuracy"])
        if selection_key > best_key:
            best_key = selection_key
            save_fusion_checkpoint(
                best_path,
                model,
                metadata,
                extra={
                    "phase": args.phase,
                    "init_from": str(init_checkpoint) if init_checkpoint else "",
                    "epoch": epoch + 1,
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_acc": val_metrics["accuracy"],
                    "per_class_recall": val_metrics["per_class_recall"],
                    "dataset_fingerprint": dataset_report.get("audit_fingerprint", ""),
                },
            )
    save_fusion_checkpoint(
        run_dir / "last.pt",
        model,
        metadata,
        extra={
            "phase": args.phase,
            "init_from": str(init_checkpoint) if init_checkpoint else "",
            "epochs": args.epochs,
            "best_val_macro_f1": best_key[0],
            "best_val_acc": best_key[1],
            "dataset_fingerprint": dataset_report.get("audit_fingerprint", ""),
        },
    )
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "phase": args.phase,
                "base": args.base,
                "gray_weights": args.gray_weights,
                "init_from": str(init_checkpoint) if init_checkpoint else "",
                "data": str(data_root),
                "dataset_fingerprint": dataset_report.get("audit_fingerprint", ""),
                "imgsz": imgsz,
                "batch": args.batch,
                "epochs": args.epochs,
                "lr": args.lr,
                "gray_lr": args.gray_lr,
                "lambda_gray": args.lambda_gray,
                "lambda_kd": args.lambda_kd,
                "seed": args.seed,
                "limit_batches": int(args.limit_batches),
                "smoke": args.limit_batches > 0,
                "device": device,
                "version": FUSION_VERSION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_run_id(args.run_id)
    except InvalidRunIdError as exc:
        print(f"invalid run id: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(args)
    try:
        run_dir = run_training(args)
    except DeviceUnavailableError as exc:
        print(f"device unavailable: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"training refused: {exc}", file=sys.stderr)
        return 2
    print(f"Polar fusion run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
