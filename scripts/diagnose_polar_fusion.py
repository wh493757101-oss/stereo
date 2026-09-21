"""Diagnostic evaluation of one trained Polar Fusion checkpoint.

Read-only diagnosis of Gray -> Fusion degradation on a single split. It
evaluates, with ``model.eval()`` and ``torch.no_grad()`` only (never
backward, optimizer or training):

    A. Gray-only logits from the fusion model's gray branch, after that
       branch has been verified parameter-for-parameter (and buffer-for-
       buffer) identical to the independent Gray checkpoint.
    B. True-polar fusion with each sample's own gray/polar/quality.
    C. Forced invalid-polar fallbacks (``polar_invalid=True`` and
       ``quality.valid_ratio=0``); the gate must be exactly 0 and the final
       logits exactly equal to the gray logits, otherwise the diagnosis
       status is FAILED.
    D. Polar correspondence shuffled across the full split index (Sattolo
       derangement, one mapping per seed over all samples; polar channels
       and quality always come from the same donor). This breaks the
       polar-to-gray correspondence and is a diagnostic, not a causal
       proof: shuffled results alone cannot claim polarization is helpful
       or useless.

Exactly one checkpoint is evaluated; there is no multi-checkpoint sweep
and nothing is selected or tuned on test scores. Preprocessing is the
shared ``FusionClsDataset`` path (imgsz inherited from the checkpoint).

Usage:
    python scripts/diagnose_polar_fusion.py \
        --checkpoint runs/train/<run-id>/polar_fusion/freeze/best.pt \
        --gray-weights runs/train/<run-id>/gray_fusion/best.pt \
        --data datasets/underwater_cls_fusion_v4_band --split test \
        --device 0 --batch 32 \
        --shuffle-seeds 2026 2027 2028 2029 2030 \
        --output-dir analysis/runs/<run-id>/diagnostics_<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fusion_dataset import QUALITY_VECTOR_KEYS, file_digest, load_fusion_sample
from models.polar_fusion import (
    VALID_RATIO_INDEX,
    _fusion_tensors,
    load_fusion_checkpoint,
    read_fusion_metadata,
    read_manifest_split,
    rebuild_gray_backbone,
)
from scripts.train_polar_fusion import torch_device_name, validate_dataset
from scripts.training_common import DeviceUnavailableError, resolve_device

DEFAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)
DEFAULT_DATA = "datasets/underwater_cls_fusion_v4_band"
CLASS_NAMES = [
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
]
FALLBACK_CONDITIONS = ("valid_ratio_zero", "polar_invalid")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="Fusion best.pt.")
    parser.add_argument(
        "--gray-weights", required=True,
        help="Independent Gray best.pt the fusion branch must match.",
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--shuffle-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Seeds for the polar-correspondence shuffle; every seed is "
        "reported (no best-seed selection).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="New diagnostics directory; an existing directory is refused.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Default: inherit the checkpoint imgsz (recommended).",
    )
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    if not args.shuffle_seeds:
        parser.error("--shuffle-seeds must not be empty")
    return args


def classification_report(labels: Sequence[int], preds: Sequence[int], num_classes: int) -> dict:
    """Accuracy, macro-F1, per-class P/R/F1/support and the confusion matrix
    (rows = true class, columns = predicted class). Zero denominators yield
    0.0 instead of NaN."""
    labels = [int(value) for value in labels]
    preds = [int(value) for value in preds]
    if len(labels) != len(preds):
        raise ValueError("labels and preds must have the same length")
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for label, pred in zip(labels, preds):
        matrix[label][pred] += 1
    per_class: dict[str, dict] = {}
    f1_values = []
    for cls in range(num_classes):
        tp = matrix[cls][cls]
        fp = sum(matrix[row][cls] for row in range(num_classes) if row != cls)
        fn = sum(matrix[cls]) - tp
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(cls)] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
        f1_values.append(f1)
    correct = sum(matrix[cls][cls] for cls in range(num_classes))
    total = len(labels)
    return {
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def change_summary(
    gray_preds: Sequence[int], fusion_preds: Sequence[int], labels: Sequence[int]
) -> dict:
    """Gray->Fusion prediction-change accounting (issue: explain changes)."""
    counts = {
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
        "wrong_to_wrong_changed_class": 0,
    }
    changed_indices = []
    for index, (label, gray, fusion) in enumerate(
        zip(labels, gray_preds, fusion_preds)
    ):
        gray_ok = gray == label
        fusion_ok = fusion == label
        if gray_ok and fusion_ok:
            counts["correct_to_correct"] += 1
        elif gray_ok and not fusion_ok:
            counts["correct_to_wrong"] += 1
            changed_indices.append(index)
        elif not gray_ok and fusion_ok:
            counts["wrong_to_correct"] += 1
            changed_indices.append(index)
        else:
            counts["wrong_to_wrong"] += 1
            if gray != fusion:
                counts["wrong_to_wrong_changed_class"] += 1
                changed_indices.append(index)
    counts["changed_total"] = len(changed_indices)
    counts["changed_indices"] = changed_indices
    return counts


def derangement(n: int, seed: int) -> list[int]:
    """Deterministic Sattolo cycle: a permutation of ``range(n)`` with no
    fixed points, a function of ``(n, seed)`` only (batch independent)."""
    if n < 2:
        raise ValueError(f"cannot build a derangement for n={n}")
    rng = np.random.default_rng(int(seed))
    permutation = list(range(n))
    for index in range(n - 1, 0, -1):
        partner = int(rng.integers(0, index))
        permutation[index], permutation[partner] = (
            permutation[partner], permutation[index],
        )
    return permutation


def batch_slices(n: int, batch: int) -> list[tuple[int, int]]:
    """Index slices covering ``range(n)`` exactly once (no dropped tail)."""
    return [(start, min(start + batch, n)) for start in range(0, n, batch)]


def _load_polar_quality(path: Path, imgsz: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Polar channels + quality of one sample through the shared
    ``_fusion_tensors`` preprocessing (identical to training/inference)."""
    import cv2

    sample = load_fusion_sample(path)
    _gray_t, polar_t, quality_t = _fusion_tensors(sample, imgsz, cv2)
    return polar_t, quality_t


def shuffled_polar_quality(
    paths: Sequence[Path], permutation: Sequence[int], imgsz: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per position: polar channels AND quality from the same donor sample."""
    polars = []
    qualities = []
    for donor in permutation:
        polar_t, quality_t = _load_polar_quality(paths[donor], imgsz)
        polars.append(polar_t)
        qualities.append(quality_t)
    return polars, qualities


def _fallback_report(
    gate: torch.Tensor, final_logits: torch.Tensor, gray_logits: torch.Tensor
) -> dict:
    """Exactness report for one forced-invalid fallback: gate must be
    exactly 0 and final logits exactly equal to the gray logits."""
    gate_failures = int((gate != 0.0).sum())
    logits_failures = int((final_logits != gray_logits).sum())
    max_diff = float((final_logits - gray_logits).abs().max()) if final_logits.numel() else 0.0
    return {
        "gate_failures": gate_failures,
        "logits_failures": logits_failures,
        "max_logits_diff": max_diff,
    }


def verify_gray_branch(fusion_gray_module, independent_gray_module) -> dict:
    """The fusion model's gray branch must match the independent Gray
    checkpoint in every parameter and buffer (BatchNorm statistics
    included). Raises on any drift."""
    fusion_params = dict(fusion_gray_module.named_parameters())
    independent_params = dict(independent_gray_module.named_parameters())
    fusion_buffers = dict(fusion_gray_module.named_buffers())
    independent_buffers = dict(independent_gray_module.named_buffers())
    if set(fusion_params) != set(independent_params):
        raise RuntimeError(
            "gray branch parameter name mismatch between the fusion "
            "checkpoint and the independent Gray weights"
        )
    if set(fusion_buffers) != set(independent_buffers):
        raise RuntimeError(
            "gray branch buffer name mismatch between the fusion "
            "checkpoint and the independent Gray weights"
        )
    for name, value in fusion_params.items():
        if not torch.equal(value.cpu(), independent_params[name].cpu()):
            raise RuntimeError(f"gray branch parameter drift/mismatch: {name}")
    for name, value in fusion_buffers.items():
        if not torch.equal(value.cpu(), independent_buffers[name].cpu()):
            raise RuntimeError(f"gray branch buffer drift/mismatch: {name}")
    return {
        "parameters_compared": len(fusion_params),
        "buffers_compared": len(fusion_buffers),
    }


def check_metadata(
    *,
    metadata,
    fusion_config: dict,
    gray_config: dict,
    fusion_fingerprint: str,
    expected_fingerprint: str,
    gray_weights_path: Path,
    class_names: Sequence[str],
    imgsz: int,
) -> dict:
    """Pre-evaluation admission: dataset fingerprint binding, class order,
    imgsz, formal (non-smoke) markers and this round's Gray source."""
    if not expected_fingerprint:
        raise RuntimeError(
            "current dataset fingerprint is empty; refusing (the strict "
            "dataset gate must return a verified fingerprint)"
        )

    def _check_fingerprint(recorded, owner: str) -> None:
        if not isinstance(recorded, str) or not recorded:
            raise RuntimeError(f"{owner} dataset_fingerprint is missing/empty")
        if recorded != expected_fingerprint:
            raise RuntimeError(
                f"{owner} dataset_fingerprint {recorded!r} != verified "
                f"fingerprint {expected_fingerprint!r}"
            )

    _check_fingerprint(gray_config.get("dataset_fingerprint"), "gray train_config")
    _check_fingerprint(fusion_config.get("dataset_fingerprint"), "fusion train_config")
    _check_fingerprint(fusion_fingerprint, "fusion checkpoint")
    if list(metadata.class_names) != list(class_names):
        raise RuntimeError(
            f"fusion checkpoint class order {list(metadata.class_names)} != "
            f"{list(class_names)}"
        )
    if int(metadata.imgsz) != int(imgsz):
        raise RuntimeError(
            f"fusion checkpoint imgsz {metadata.imgsz} != requested {imgsz}"
        )
    if not str(metadata.architecture).startswith("yolo26"):
        raise RuntimeError(
            f"fusion checkpoint architecture {metadata.architecture!r} is not YOLO26"
        )
    if fusion_config.get("phase") != "freeze":
        raise RuntimeError(f"fusion phase {fusion_config.get('phase')!r} is not freeze")
    for owner, config in (
        ("gray train_config", gray_config),
        ("fusion train_config", fusion_config),
    ):
        if config.get("smoke") is not False:
            raise RuntimeError(
                f"{owner} smoke field must be exactly false for diagnosis; "
                f"got {config.get('smoke')!r}"
            )
        if not (type(config.get("limit_batches")) is int and config.get("limit_batches") == 0):
            raise RuntimeError(
                f"{owner} limit_batches must be the integer 0 for diagnosis; "
                f"got {config.get('limit_batches')!r}"
            )
    expected_gray = Path(gray_weights_path).resolve()
    for owner, recorded in (
        ("fusion train_config gray_weights", fusion_config.get("gray_weights")),
        ("fusion checkpoint gray_weights", metadata.gray_weights),
    ):
        path = Path(str(recorded or ""))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.resolve() != expected_gray:
            raise RuntimeError(
                f"{owner} {path} != --gray-weights {expected_gray}"
            )
    return {
        "architecture": metadata.architecture,
        "class_names": list(metadata.class_names),
        "imgsz": int(metadata.imgsz),
        "phase": "freeze",
    }


def prepare_output_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"output directory already exists: {path}; refusing to overwrite"
        )
    path.mkdir(parents=True)
    return path


def _seed_stats(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in values],
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


@torch.no_grad()
def evaluate(
    model,
    paths: Sequence[Path],
    labels: Sequence[int],
    *,
    imgsz: int,
    device: str,
    batch: int,
    shuffle_seeds: Sequence[int],
    sample_ids: Sequence[str] | None = None,
) -> dict:
    """Run every diagnostic condition; the model is only ever in eval mode
    and its parameters/buffers are never modified."""
    import cv2

    model = model.to(device)
    model.eval()
    num_classes = int(model.num_classes)
    n = len(paths)
    labels_all = [int(label) for label in labels]

    gray_preds: list[int] = []
    fusion_preds: list[int] = []
    fallback_preds: dict[str, list[int]] = {name: [] for name in FALLBACK_CONDITIONS}
    fallback_checks: dict[str, dict] = {
        name: {"gate_failures": 0, "logits_failures": 0, "max_logits_diff": 0.0}
        for name in FALLBACK_CONDITIONS
    }
    records: list[dict] = []

    for start, stop in batch_slices(n, batch):
        grays, polars, qualities = [], [], []
        for index in range(start, stop):
            sample = load_fusion_sample(paths[index])
            gray_t, polar_t, quality_t = _fusion_tensors(sample, imgsz, cv2)
            grays.append(gray_t)
            polars.append(polar_t)
            qualities.append(quality_t)
        gray_b = torch.stack(grays).to(device)
        polar_b = torch.stack(polars).to(device)
        quality_b = torch.stack(qualities).to(device)

        out = model(gray_b, polar_b, quality_b)
        gray_logits = out["gray_logits"].detach()
        final_logits = out["final_logits"].detach()
        gate = out["gate"].detach()
        gate_delta = (gate * out["polar_delta"]).detach()

        zero_quality = quality_b.clone()
        zero_quality[:, VALID_RATIO_INDEX] = 0.0
        fallback_outputs = {
            "valid_ratio_zero": model(gray_b, polar_b, zero_quality),
            "polar_invalid": model(
                gray_b, polar_b, quality_b,
                polar_invalid=torch.ones(stop - start, device=device),
            ),
        }
        for name, fallback in fallback_outputs.items():
            report = _fallback_report(
                fallback["gate"].detach(), fallback["final_logits"].detach(),
                gray_logits,
            )
            fallback_checks[name]["gate_failures"] += report["gate_failures"]
            fallback_checks[name]["logits_failures"] += report["logits_failures"]
            fallback_checks[name]["max_logits_diff"] = max(
                fallback_checks[name]["max_logits_diff"], report["max_logits_diff"]
            )
            fallback_preds[name].extend(
                fallback["final_logits"].detach().argmax(1).tolist()
            )

        gray_probs = torch.softmax(gray_logits, dim=1)
        fusion_probs = torch.softmax(final_logits, dim=1)
        batch_gray_preds = gray_logits.argmax(1).tolist()
        batch_fusion_preds = final_logits.argmax(1).tolist()
        for position, index in enumerate(range(start, stop)):
            gray_preds.append(batch_gray_preds[position])
            fusion_preds.append(batch_fusion_preds[position])
            gray_norm = float(gray_logits[position].norm())
            delta_norm = float(gate_delta[position].norm())
            records.append({
                "index": index,
                "sample_id": (
                    sample_ids[index] if sample_ids is not None
                    else Path(paths[index]).name
                ),
                "npz_path": str(paths[index]),
                "true_id": labels_all[index],
                "gray_pred": batch_gray_preds[position],
                "fusion_pred": batch_fusion_preds[position],
                "gray_logits": gray_logits[position].tolist(),
                "fusion_logits": final_logits[position].tolist(),
                "gray_max_softmax": float(gray_probs[position].max()),
                "fusion_max_softmax": float(fusion_probs[position].max()),
                "gate": float(gate[position, 0]),
                "quality": quality_b[position].tolist(),
                "gate_delta": gate_delta[position].tolist(),
                "gate_delta_norm": delta_norm,
                "gate_delta_relative_to_gray": delta_norm / (gray_norm + 1e-12),
            })

    for record in records:
        for name in FALLBACK_CONDITIONS:
            record[f"fallback_{name}_pred"] = fallback_preds[name][record["index"]]

    shuffles: dict[str, dict] = {}
    for seed in shuffle_seeds:
        seed_key = str(int(seed))
        permutation = derangement(n, int(seed))
        preds: list[int] = []
        for start, stop in batch_slices(n, batch):
            grays, polars, qualities = [], [], []
            for index in range(start, stop):
                gray_t, _, _ = _fusion_tensors(
                    load_fusion_sample(paths[index]), imgsz, cv2
                )
                donor_polar, donor_quality = _load_polar_quality(
                    paths[permutation[index]], imgsz
                )
                grays.append(gray_t)
                polars.append(donor_polar)
                qualities.append(donor_quality)
            out = model(
                torch.stack(grays).to(device),
                torch.stack(polars).to(device),
                torch.stack(qualities).to(device),
            )
            preds.extend(out["final_logits"].detach().argmax(1).tolist())
        shuffles[seed_key] = {
            "seed": int(seed),
            "permutation": permutation,
            "fixed_points": sum(1 for i, p in enumerate(permutation) if p == i),
            "predictions": preds,
            "metrics": classification_report(labels_all, preds, num_classes),
        }
    for record in records:
        record["shuffle_donors"] = {
            key: shuffles[key]["permutation"][record["index"]] for key in shuffles
        }
        record["shuffle_predictions"] = {
            key: shuffles[key]["predictions"][record["index"]] for key in shuffles
        }

    gate_values = [record["gate"] for record in records]
    quality_matrix = (
        np.asarray([record["quality"] for record in records], dtype=np.float64)
        if records else np.zeros((0, len(QUALITY_VECTOR_KEYS)))
    )
    return {
        "samples_evaluated": n,
        "gray": classification_report(labels_all, gray_preds, num_classes),
        "fusion": classification_report(labels_all, fusion_preds, num_classes),
        "changes": change_summary(gray_preds, fusion_preds, labels_all),
        "fallbacks": {
            name: {
                **fallback_checks[name],
                "metrics": classification_report(
                    labels_all, fallback_preds[name], num_classes
                ),
            }
            for name in FALLBACK_CONDITIONS
        },
        "gate": {
            "mean": float(np.mean(gate_values)) if gate_values else 0.0,
            "min": float(np.min(gate_values)) if gate_values else 0.0,
            "max": float(np.max(gate_values)) if gate_values else 0.0,
            "zero_fraction": float(np.mean([g == 0.0 for g in gate_values])) if gate_values else 0.0,
        },
        "quality": {
            "component_mean": quality_matrix.mean(0).tolist() if records else [0.0] * len(QUALITY_VECTOR_KEYS),
            "component_min": quality_matrix.min(0).tolist() if records else [0.0] * len(QUALITY_VECTOR_KEYS),
            "component_max": quality_matrix.max(0).tolist() if records else [0.0] * len(QUALITY_VECTOR_KEYS),
        },
        "shuffles": shuffles,
        "shuffle_summary": {
            "accuracy": _seed_stats([shuffles[k]["metrics"]["accuracy"] for k in shuffles]),
            "macro_f1": _seed_stats([shuffles[k]["metrics"]["macro_f1"] for k in shuffles]),
        },
        "records": records,
        "gray_predictions": gray_preds,
        "fusion_predictions": fusion_preds,
        "labels": labels_all,
        "class_names": list(CLASS_NAMES),
    }


def _write_predictions_csv(path: Path, result: dict) -> None:
    seed_keys = list(result["shuffles"])
    fieldnames = [
        "sample_id", "npz_path", "true_id", "true_name",
        "gray_pred", "gray_pred_name", "fusion_pred", "fusion_pred_name",
        "gray_logits", "fusion_logits", "gray_max_softmax", "fusion_max_softmax",
        "gate",
        *[f"quality_{key}" for key in QUALITY_VECTOR_KEYS],
        "gate_delta", "gate_delta_norm", "gate_delta_relative_to_gray",
        "fallback_valid_ratio_zero_pred", "fallback_polar_invalid_pred",
        *[f"shuffle_donor_{key}" for key in seed_keys],
        *[f"shuffle_pred_{key}" for key in seed_keys],
    ]
    names = result["class_names"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in result["records"]:
            row = {
                "sample_id": record["sample_id"],
                "npz_path": record["npz_path"],
                "true_id": record["true_id"],
                "true_name": names[record["true_id"]],
                "gray_pred": record["gray_pred"],
                "gray_pred_name": names[record["gray_pred"]],
                "fusion_pred": record["fusion_pred"],
                "fusion_pred_name": names[record["fusion_pred"]],
                "gray_logits": json.dumps(record["gray_logits"]),
                "fusion_logits": json.dumps(record["fusion_logits"]),
                "gray_max_softmax": record["gray_max_softmax"],
                "fusion_max_softmax": record["fusion_max_softmax"],
                "gate": record["gate"],
                "gate_delta": json.dumps(record["gate_delta"]),
                "gate_delta_norm": record["gate_delta_norm"],
                "gate_delta_relative_to_gray": record["gate_delta_relative_to_gray"],
                "fallback_valid_ratio_zero_pred": record["fallback_valid_ratio_zero_pred"],
                "fallback_polar_invalid_pred": record["fallback_polar_invalid_pred"],
            }
            for key, value in zip(QUALITY_VECTOR_KEYS, record["quality"]):
                row[f"quality_{key}"] = value
            for key in seed_keys:
                row[f"shuffle_donor_{key}"] = record["shuffle_donors"][key]
                row[f"shuffle_pred_{key}"] = record["shuffle_predictions"][key]
            writer.writerow(row)


def _change_category(label: int, gray: int, fusion: int) -> str:
    if gray == label and fusion != label:
        return "correct_to_wrong"
    if gray != label and fusion == label:
        return "wrong_to_correct"
    return "wrong_to_wrong_changed_class"


def _write_changed_csv(path: Path, result: dict) -> None:
    names = result["class_names"]
    fieldnames = [
        "sample_id", "npz_path", "true_id", "true_name",
        "gray_pred", "gray_pred_name", "fusion_pred", "fusion_pred_name",
        "category", "gate", "gate_delta_norm", "gate_delta_relative_to_gray",
        *[f"quality_{key}" for key in QUALITY_VECTOR_KEYS],
        "gray_logits", "fusion_logits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in result["changes"]["changed_indices"]:
            record = result["records"][index]
            row = {
                "sample_id": record["sample_id"],
                "npz_path": record["npz_path"],
                "true_id": record["true_id"],
                "true_name": names[record["true_id"]],
                "gray_pred": record["gray_pred"],
                "gray_pred_name": names[record["gray_pred"]],
                "fusion_pred": record["fusion_pred"],
                "fusion_pred_name": names[record["fusion_pred"]],
                "category": _change_category(
                    record["true_id"], record["gray_pred"], record["fusion_pred"]
                ),
                "gate": record["gate"],
                "gate_delta_norm": record["gate_delta_norm"],
                "gate_delta_relative_to_gray": record["gate_delta_relative_to_gray"],
                "gray_logits": json.dumps(record["gray_logits"]),
                "fusion_logits": json.dumps(record["fusion_logits"]),
            }
            for key, value in zip(QUALITY_VECTOR_KEYS, record["quality"]):
                row[f"quality_{key}"] = value
            writer.writerow(row)


def _write_panels(output_dir: Path, result: dict, data_root: Path) -> int:
    """Diagnostic panels (gray / signed_q fixed [-1,1] / abs_q fixed [0,1] /
    valid mask) for the changed samples; read-only from the npz files."""
    import cv2

    panels_dir = output_dir / "changed_samples"
    panels_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in result["changes"]["changed_indices"]:
        record = result["records"][index]
        sample = load_fusion_sample(Path(record["npz_path"]))
        height, width = sample.gray.shape
        panel = np.zeros((height, width * 4 + 30, 3), dtype=np.uint8)
        panel[:, :width] = cv2.cvtColor(sample.gray, cv2.COLOR_GRAY2BGR)
        signed_u8 = np.rint(
            np.clip((sample.signed_q.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0) * 255
        ).astype(np.uint8)
        panel[:, width + 10 : 2 * width + 10] = cv2.applyColorMap(
            signed_u8, cv2.COLORMAP_VIRIDIS
        )
        abs_u8 = np.rint(np.clip(sample.abs_q, 0.0, 1.0) * 255).astype(np.uint8)
        panel[:, 2 * width + 20 : 3 * width + 20] = cv2.applyColorMap(
            abs_u8, cv2.COLORMAP_VIRIDIS
        )
        valid_u8 = (sample.valid > 0).astype(np.uint8) * 255
        panel[:, 3 * width + 30 :] = cv2.cvtColor(valid_u8, cv2.COLOR_GRAY2BGR)
        for offset, title in zip(
            (0, width + 10, 2 * width + 20, 3 * width + 30),
            ("gray", "signed_q [-1,1]", "abs_q [0,1]", "valid"),
        ):
            cv2.putText(
                panel, title, (offset + 4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1, cv2.LINE_AA,
            )
        safe_name = record["sample_id"].replace("/", "_").replace("\\", "_")
        cv2.imwrite(str(panels_dir / f"{safe_name}_panel.png"), panel)
        written += 1
    return written


def _condition_rows(result: dict) -> list[tuple[str, float, float]]:
    rows = [
        ("gray (fusion gray branch)", result["gray"]["accuracy"], result["gray"]["macro_f1"]),
        ("fusion (true polar)", result["fusion"]["accuracy"], result["fusion"]["macro_f1"]),
        (
            "fallback valid_ratio=0",
            result["fallbacks"]["valid_ratio_zero"]["metrics"]["accuracy"],
            result["fallbacks"]["valid_ratio_zero"]["metrics"]["macro_f1"],
        ),
        (
            "fallback polar_invalid",
            result["fallbacks"]["polar_invalid"]["metrics"]["accuracy"],
            result["fallbacks"]["polar_invalid"]["metrics"]["macro_f1"],
        ),
    ]
    for key in result["shuffles"]:
        metrics = result["shuffles"][key]["metrics"]
        rows.append((f"shuffled polar (seed {key})", metrics["accuracy"], metrics["macro_f1"]))
    return rows


def _write_summary_md(path: Path, result: dict, run_info: dict, status: str) -> None:
    names = result["class_names"]
    lines: list[str] = []
    lines.append(f"# Polar Fusion diagnostics — run `{run_info['run_id']}` ({run_info['split']})")
    lines.append("")
    lines.append(f"Status: **{status}**")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- checkpoint: `{run_info['checkpoint']}` (sha256 `{run_info['checkpoint_sha256']}`)")
    lines.append(f"- gray weights: `{run_info['gray_weights']}` (sha256 `{run_info['gray_weights_sha256']}`)")
    lines.append(f"- data: `{run_info['data']}` fingerprint `{run_info['dataset_fingerprint']}`")
    lines.append(
        f"- split `{run_info['split']}`, {result['samples_evaluated']} samples, "
        f"imgsz {run_info['imgsz']}, device {run_info['device']}, batch {run_info['batch']}"
    )
    lines.append(
        f"- shuffle seeds: {', '.join(run_info['shuffle_seeds'])} (all reported; "
        "no best-seed selection)"
    )
    lines.append("")
    lines.append("## Conditions")
    lines.append("")
    lines.append("| condition | accuracy | macro-F1 |")
    lines.append("| --- | --- | --- |")
    for name, accuracy, macro_f1 in _condition_rows(result):
        lines.append(f"| {name} | {accuracy:.6f} | {macro_f1:.6f} |")
    lines.append("")
    lines.append(
        "The `gate` below is a learned scalar gate (descriptive statistic), "
        "not a calibrated confidence. Max softmax values in the CSVs are "
        "descriptive only as well."
    )
    lines.append("")
    lines.append("## Per-class metrics (Gray / Fusion)")
    lines.append("")
    lines.append("| class | gray P | gray R | gray F1 | fusion P | fusion R | fusion F1 | support |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cls in range(len(names)):
        gray = result["gray"]["per_class"][str(cls)]
        fusion = result["fusion"]["per_class"][str(cls)]
        lines.append(
            f"| {names[cls]} | {gray['precision']:.4f} | {gray['recall']:.4f} | "
            f"{gray['f1']:.4f} | {fusion['precision']:.4f} | {fusion['recall']:.4f} | "
            f"{fusion['f1']:.4f} | {gray['support']} |"
        )
    lines.append("")
    changes = result["changes"]
    lines.append("## Gray -> Fusion changes")
    lines.append("")
    lines.append(
        f"- correct -> correct: {changes['correct_to_correct']}; "
        f"correct -> wrong: {changes['correct_to_wrong']}; "
        f"wrong -> correct: {changes['wrong_to_correct']}; "
        f"wrong -> wrong: {changes['wrong_to_wrong']} "
        f"(of which changed class: {changes['wrong_to_wrong_changed_class']})"
    )
    lines.append(f"- total changed predictions: {changes['changed_total']}")
    lines.append("")
    if changes["changed_indices"]:
        lines.append("### Changed samples")
        lines.append("")
        lines.append("| sample | true | gray | fusion | category | gate | valid_ratio | mean_abs_q |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for index in changes["changed_indices"]:
            record = result["records"][index]
            quality = record["quality"]
            lines.append(
                f"| {record['sample_id']} | {names[record['true_id']]} | "
                f"{names[record['gray_pred']]} | {names[record['fusion_pred']]} | "
                f"{_change_category(record['true_id'], record['gray_pred'], record['fusion_pred'])} | "
                f"{record['gate']:.4f} | {quality[0]:.4f} | {quality[3]:.4f} |"
            )
        lines.append("")
        lines.append("Clue notes (no root-cause claims without evidence):")
        lines.append("")
        for index in changes["changed_indices"]:
            record = result["records"][index]
            quality = record["quality"]
            clues = []
            if quality[0] < 0.5:
                clues.append(f"low valid_ratio ({quality[0]:.4f})")
            if quality[2] < 0.5:
                clues.append(f"low brightness_valid_ratio ({quality[2]:.4f})")
            if record["gate_delta_relative_to_gray"] > 0.1:
                clues.append(
                    "comparatively large gate*delta vs gray logits "
                    f"({record['gate_delta_relative_to_gray']:.4f})"
                )
            clue_text = "; ".join(clues) if clues else "no obvious quality clue"
            lines.append(f"- `{record['sample_id']}`: {clue_text}")
        lines.append("")
    lines.append("## Fallback and consistency checks")
    lines.append("")
    for name, report in result["fallbacks"].items():
        lines.append(
            f"- fallback `{name}`: gate failures {report['gate_failures']}, "
            f"logits failures {report['logits_failures']}, "
            f"max |final-gray| {report['max_logits_diff']:.3e}"
        )
    lines.append(
        f"- gray branch vs independent Gray weights: "
        f"{run_info['gray_branch']['parameters_compared']} parameters and "
        f"{run_info['gray_branch']['buffers_compared']} buffers identical"
    )
    lines.append(
        f"- gate (descriptive, not calibrated confidence): mean "
        f"{result['gate']['mean']:.4f}, min {result['gate']['min']:.4f}, max "
        f"{result['gate']['max']:.4f}, zero fraction {result['gate']['zero_fraction']:.4f}"
    )
    lines.append("")
    lines.append("## Facts / hypotheses / unchecked")
    lines.append("")
    lines.append("Facts (this run, this split):")
    lines.append("")
    lines.append(
        f"- Gray errors {result['samples_evaluated'] - round(result['gray']['accuracy'] * result['samples_evaluated'])}, "
        f"fusion errors {result['samples_evaluated'] - round(result['fusion']['accuracy'] * result['samples_evaluated'])}, "
        f"changed predictions {changes['changed_total']}."
    )
    lines.append("- Both forced-invalid fallbacks are exact (gate 0, final == gray).")
    lines.append("- The fusion gray branch is parameter- and buffer-identical to the independent Gray checkpoint.")
    lines.append("")
    lines.append("Hypotheses (not proven by this diagnostic):")
    lines.append("")
    lines.append(
        "- Whether the polar channel carries usable class information is only "
        "bounded by the shuffled-correspondence results above; shuffled metrics "
        "near the true-polar metrics suggest the polar input contributes little "
        "beyond gray on this split, but this is not a causal proof."
    )
    lines.append(
        "- The degradation is limited to a handful of changed samples; per-sample "
        "gate/quality clues above are leads, not established root causes."
    )
    lines.append("")
    lines.append("Not checked:")
    lines.append("")
    lines.append("- No label-conditioned analysis, no hyperparameter tuning, no second checkpoint.")
    lines.append("- No distance ground truth, no cross-run generalization, no joint-phase behavior.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_json(path: Path, result: dict, run_info: dict, status: str) -> None:
    payload = {
        "status": status,
        "run": run_info,
        "conditions": {
            "gray": {"accuracy": result["gray"]["accuracy"], "macro_f1": result["gray"]["macro_f1"]},
            "fusion": {"accuracy": result["fusion"]["accuracy"], "macro_f1": result["fusion"]["macro_f1"]},
            "fallback_valid_ratio_zero": {
                "accuracy": result["fallbacks"]["valid_ratio_zero"]["metrics"]["accuracy"],
                "macro_f1": result["fallbacks"]["valid_ratio_zero"]["metrics"]["macro_f1"],
            },
            "fallback_polar_invalid": {
                "accuracy": result["fallbacks"]["polar_invalid"]["metrics"]["accuracy"],
                "macro_f1": result["fallbacks"]["polar_invalid"]["metrics"]["macro_f1"],
            },
        },
        "per_class": {
            "gray": result["gray"]["per_class"],
            "fusion": result["fusion"]["per_class"],
        },
        "confusion_matrix_note": "rows = true class, columns = predicted class",
        "confusion_matrix": {
            "gray": result["gray"]["confusion_matrix"],
            "fusion": result["fusion"]["confusion_matrix"],
        },
        "changes": result["changes"],
        "gate": result["gate"],
        "quality": {
            "component_keys": list(QUALITY_VECTOR_KEYS),
            **result["quality"],
        },
        "fallback_checks": {
            name: {
                "gate_failures": report["gate_failures"],
                "logits_failures": report["logits_failures"],
                "max_logits_diff": report["max_logits_diff"],
            }
            for name, report in result["fallbacks"].items()
        },
        "shuffle_summary": result["shuffle_summary"],
        "shuffles": {
            key: {
                "seed": value["seed"],
                "fixed_points": value["fixed_points"],
                "accuracy": value["metrics"]["accuracy"],
                "macro_f1": value["metrics"]["macro_f1"],
            }
            for key, value in result["shuffles"].items()
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint
    gray_weights = Path(args.gray_weights)
    gray_weights = gray_weights if gray_weights.is_absolute() else PROJECT_ROOT / gray_weights
    data_root = Path(args.data)
    data_root = data_root if data_root.is_absolute() else PROJECT_ROOT / data_root
    output_dir = Path(args.output_dir)
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    if output_dir.exists():
        print(
            f"diagnosis refused: output directory already exists: {output_dir}",
            file=sys.stderr,
        )
        return 2

    try:
        device = torch_device_name(resolve_device(args.device))
    except DeviceUnavailableError as exc:
        print(f"device unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        dataset_report = validate_dataset(data_root)
        fingerprint = dataset_report.get("audit_fingerprint", "")
        metadata = read_fusion_metadata(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        fusion_config = json.loads(
            (checkpoint.parent / "train_config.json").read_text(encoding="utf-8")
        )
        gray_config = json.loads(
            (gray_weights.parent / "train_config.json").read_text(encoding="utf-8")
        )
        imgsz = int(args.imgsz) if args.imgsz is not None else int(metadata.imgsz)
        metadata_checks = check_metadata(
            metadata=metadata,
            fusion_config=fusion_config,
            gray_config=gray_config,
            fusion_fingerprint=str(payload.get("dataset_fingerprint", "")),
            expected_fingerprint=fingerprint,
            gray_weights_path=gray_weights,
            class_names=CLASS_NAMES,
            imgsz=imgsz,
        )
        paths, labels, _ = read_manifest_split(data_root, args.split)
        sample_ids = [path.relative_to(data_root).as_posix() for path in paths]

        backbone = rebuild_gray_backbone(metadata, device)
        model, metadata, payload = load_fusion_checkpoint(checkpoint, backbone)

        from ultralytics import YOLO

        independent_gray = YOLO(str(gray_weights)).model
        gray_branch_report = verify_gray_branch(
            model.gray_backbone.module, independent_gray
        )

        result = evaluate(
            model, paths, labels,
            imgsz=imgsz, device=device, batch=args.batch,
            shuffle_seeds=tuple(args.shuffle_seeds), sample_ids=sample_ids,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"diagnosis refused: {exc}", file=sys.stderr)
        return 2

    status = "PASSED"
    for report in result["fallbacks"].values():
        if report["gate_failures"] or report["logits_failures"]:
            status = "FAILED"

    import ultralytics

    run_info = {
        "run_id": data_root.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "gray_weights": str(gray_weights),
        "gray_weights_sha256": file_digest(gray_weights),
        "data": str(data_root),
        "dataset_fingerprint": fingerprint,
        "split": args.split,
        "class_names": list(CLASS_NAMES),
        "imgsz": imgsz,
        "device": device,
        "batch": args.batch,
        "shuffle_seeds": [str(seed) for seed in args.shuffle_seeds],
        "samples": result["samples_evaluated"],
        "metadata_checks": metadata_checks,
        "gray_branch": gray_branch_report,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": [
            "read-only diagnostic; no training, no backward, no checkpoint selection",
            "gate is a learned scalar gate, not a calibrated confidence",
            "shuffled-polar results break the polar-gray correspondence; they "
            "are diagnostic evidence, not a causal proof",
        ],
    }

    out = prepare_output_dir(output_dir)
    panels = _write_panels(out, result, data_root)
    _write_summary_json(out / "summary.json", result, run_info, status)
    _write_predictions_csv(out / "predictions.csv", result)
    _write_changed_csv(out / "changed_predictions.csv", result)
    _write_summary_md(out / "summary.md", result, run_info, status)

    print(f"Diagnostics report: {out}")
    print(
        f"gray acc={result['gray']['accuracy']:.6f} "
        f"fusion acc={result['fusion']['accuracy']:.6f} "
        f"changed={result['changes']['changed_total']} "
        f"(gray errors "
        f"{result['samples_evaluated'] - sum(1 for l, p in zip(result['labels'], result['gray_predictions']) if l == p)}, "
        f"fusion errors "
        f"{result['samples_evaluated'] - sum(1 for l, p in zip(result['labels'], result['fusion_predictions']) if l == p)})"
    )
    print(f"Panels written: {panels}")
    if status != "PASSED":
        print("diagnosis status FAILED: fallback exactness check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
