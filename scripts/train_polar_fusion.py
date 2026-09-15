"""Train the Polar Fusion classification model on the V3 fusion dataset.

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
    load_fusion_sample,
)
from models.polar_fusion import (
    DEFAULT_BASE_MODEL,
    FUSION_VERSION,
    FusionCheckpointMetadata,
    FusionClsDataset,
    GrayBackboneAdapter,
    PolarFusionModel,
    build_gray_class_perm,
    class_names_from_manifest,
    fusion_metadata,
    load_fusion_checkpoint,
    read_manifest_split,
    save_fusion_checkpoint,
)
from scripts.train_models import DeviceUnavailableError, InvalidRunIdError, resolve_device, validate_run_id

DEFAULT_SEED = 2026
DEFAULT_DATA = "datasets/underwater_cls_fusion_v3"
DEFAULT_PHASE = "freeze"
PHASES = ("freeze", "joint")
# Accepted 4-class gray Model B used to initialize the fusion gray branch.
DEFAULT_GRAY_WEIGHTS = "runs/train/run_20260913_initial/model_b-gray/weights/best.pt"


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
        default=DEFAULT_GRAY_WEIGHTS,
        help="Accepted 4-class gray Model B checkpoint initializing the gray "
        "branch (default: the formal run_20260913_initial weights). Pass an "
        "empty string to start from a freshly replaced classification head.",
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
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate.")
    parser.add_argument(
        "--gray-lr", type=float, default=1e-5, help="Gray backbone lr (joint phase)."
    )
    parser.add_argument("--phase", choices=PHASES, default=DEFAULT_PHASE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and structure only; no training, no run outputs.",
    )
    return parser.parse_args(argv)


def validate_dataset(data_root: Path) -> dict:
    """Dataset gate for dry-run and training: counts, decoding, quality shape."""
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
    dataset_report = validate_dataset(data_root)
    structure_report = structure_check(len(dataset_report["class_names"]), args.imgsz)
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
        "structure": structure_report,
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
    if args.gray_weights and not gray_weights_present:
        print(
            f"NOTE: --gray-weights {args.gray_weights} is not present locally; "
            "training would refuse rather than fall back to an untrained head.",
            file=sys.stderr,
        )
    return 0


def torch_device_name(device: str) -> str:
    """Convert an Ultralytics-style device id into a torch-compatible name.

    ``resolve_device`` accepts ``"0"``/``"cuda"``/``"cuda:N"`` for
    Ultralytics, but ``Module.to()`` rejects plain ``"0"`` with
    ``RuntimeError: Invalid device string``; this maps to ``cuda:N``.
    """
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda:0"
    if device.isdigit():
        return f"cuda:{device}"
    if device.startswith("cuda:"):
        return device
    raise ValueError(f"unsupported torch device name: {device!r}")


def _replace_cls_head(classification_model, num_classes: int) -> None:
    """Swap the classification head's final linear for ``num_classes``.

    The stock ``yolo26n-cls.pt`` base is ImageNet-trained; without this the
    gray backbone outputs 1000 logits and the fusion forward fails on the
    first batch. A fresh head starts untrained unless ``--gray-weights``
    provides accepted 4-class weights.
    """
    head = classification_model.model[-1]
    old_linear = head.linear
    head.linear = torch.nn.Linear(old_linear.in_features, num_classes)


def _architecture_name(module) -> str:
    """Actual architecture of an Ultralytics model, e.g. ``yolov8n-cls``.

    Derived from the loaded checkpoint's yaml, so the recorded base model
    reflects what really runs (the legacy gray weights are YOLOv8, not the
    requested YOLO26 default).
    """
    yaml_info = getattr(module, "yaml", None)
    if isinstance(yaml_info, dict):
        yaml_file = str(yaml_info.get("yaml_file", ""))
        if yaml_file:
            return Path(yaml_file).stem
    return ""


def prepare_gray_backbone(
    base_path: Path,
    gray_weights: str,
    class_names: list[str],
    device: str,
) -> tuple[GrayBackboneAdapter, dict]:
    """Build the fusion gray branch and align it with the V3 class order.

    - ``gray_weights`` given: load the accepted 4-class checkpoint, verify
      its head width and derive the column permutation from its class names
      (the accepted weights order plastic_fish/plastic_submarine as 1/2
      while the V3 dataset follows the seg data.yaml order 2/1).
    - ``gray_weights`` empty: build from ``base_path`` with a freshly
      replaced ``num_classes`` head (untrained gray branch).
    A provided-but-missing ``gray_weights`` path is an error, never a
    silent fallback to the untrained base.
    """
    from ultralytics import YOLO

    if gray_weights:
        weights_path = Path(gray_weights)
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"gray-weights checkpoint {weights_path} is missing; pass an "
                "empty --gray-weights to explicitly start untrained (no "
                "automatic download or fallback)"
            )
        module = YOLO(str(weights_path)).model
        head_linear = module.model[-1].linear
        if head_linear.out_features != len(class_names):
            raise ValueError(
                f"gray-weights head outputs {head_linear.out_features} classes "
                f"but the dataset defines {len(class_names)}: {class_names}"
            )
        perm = build_gray_class_perm(module.names, class_names)
        info = {
            # The architecture is whatever the loaded weights carry (e.g.
            # yolov8n-cls for the legacy run), not the requested --base.
            "base_model": _architecture_name(module) or str(weights_path.name),
            "gray_weights": str(weights_path),
            "head_replaced": False,
            "perm": perm,
            "gray_class_names": [
                str(module.names[key]) for key in sorted(module.names)
            ],
        }
    else:
        module = YOLO(str(base_path)).model
        _replace_cls_head(module, len(class_names))
        # The replaced head's class order is now the V3 dataset order.
        module.names = {i: name for i, name in enumerate(class_names)}
        perm = None
        info = {
            "base_model": _architecture_name(module) or Path(str(base_path)).name,
            "gray_weights": "",
            "head_replaced": True,
            "perm": None,
            "gray_class_names": list(class_names),
        }
    module.to(device)
    return GrayBackboneAdapter(module, perm=perm), info


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
    """Rebuild the gray branch exactly as the init checkpoint was built.

    A fresh-head checkpoint was built from the (1000-class) base with a
    replaced head: rebuilding must go through the fresh-head branch again.
    Feeding the recorded base here as gray weights would fail the
    head-width check, and silently swapping in different gray weights
    would break resume equivalence.
    """
    head_replaced = bool(init_payload.get("head_replaced", False))
    restore_gray_weights = (
        "" if head_replaced else str(init_payload.get("gray_weights", ""))
    )
    restore_base = str(init_payload.get("base_model", DEFAULT_BASE_MODEL))
    restore_base_path = Path(restore_base)
    restore_base_path = (
        restore_base_path if restore_base_path.is_absolute()
        else PROJECT_ROOT / restore_base
    )
    if head_replaced and not restore_base_path.is_file():
        raise FileNotFoundError(
            f"base checkpoint {restore_base_path} (recorded in the init "
            "checkpoint) is missing; no automatic download"
        )
    # Weights are then overwritten by the checkpoint state dict.
    backbone, _ = prepare_gray_backbone(
        restore_base_path, restore_gray_weights, class_names, device
    )
    return backbone


def run_training(args: argparse.Namespace) -> Path:
    """Full training loop (freeze or joint). Not executed by --dry-run."""
    data_root = PROJECT_ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    validate_dataset(data_root)

    # resolve_device validates CUDA availability; the result is then mapped
    # to a torch-usable device string ("0" -> "cuda:0").
    device = torch_device_name(resolve_device(args.device))
    class_names = class_names_from_manifest(data_root)
    num_classes = len(class_names)

    init_checkpoint = _resolve_init_checkpoint(args)
    if init_checkpoint is None:
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
        _verify_backbone_output(backbone, args.imgsz, num_classes, device)
        model = PolarFusionModel(backbone, num_classes=num_classes).to(device)
        metadata: FusionCheckpointMetadata = fusion_metadata(
            class_names,
            base_model=gray_info["base_model"],
            imgsz=args.imgsz,
            gray_weights=gray_info["gray_weights"],
            gray_class_names=gray_info["gray_class_names"],
            head_replaced=gray_info["head_replaced"],
            class_permutation=tuple(gray_info["perm"]) if gray_info["perm"] else (),
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

    train_paths, _, _ = read_manifest_split(data_root, "train")
    val_paths, _, _ = read_manifest_split(data_root, "val")
    train_loader = torch.utils.data.DataLoader(
        FusionClsDataset(train_paths, imgsz=args.imgsz),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0 if sys.platform == "win32" else 8,
    )
    val_loader = torch.utils.data.DataLoader(
        FusionClsDataset(val_paths, imgsz=args.imgsz),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 8,
    )

    run_dir = _phase_run_dir(args.run_id, args.phase)
    best_val_acc = -1.0
    best_path = run_dir / "best.pt"
    for epoch in range(args.epochs):
        _apply_phase_modes(model, joint)
        for gray, polar, quality, labels in train_loader:
            gray = gray.to(device)
            polar = polar.to(device)
            quality = quality.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(gray, polar, quality)
            loss = criterion(out["final_logits"], labels)
            loss.backward()
            optimizer.step()

        val_acc = _evaluate_accuracy(model, val_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_fusion_checkpoint(
                best_path,
                model,
                metadata,
                extra={
                    "phase": args.phase,
                    "init_from": str(init_checkpoint) if init_checkpoint else "",
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
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
            "best_val_acc": best_val_acc,
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
                "imgsz": args.imgsz,
                "batch": args.batch,
                "epochs": args.epochs,
                "lr": args.lr,
                "gray_lr": args.gray_lr,
                "seed": args.seed,
                "device": device,
                "version": FUSION_VERSION,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


@torch.no_grad()
def _evaluate_accuracy(model: PolarFusionModel, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for gray, polar, quality, labels in loader:
        out = model(
            gray.to(device), polar.to(device), quality.to(device)
        )
        predictions = out["final_logits"].argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += int(labels.numel())
    return correct / max(total, 1)


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
