"""Controlled support-region diagnostics for the material-fusion classifiers.

The point of this batch is to separate two things the previous round's
shuffling conflated:

- the **support region** (where a polarization measurement exists at all,
  which drives availability statistics and fallback routing), and
- the **measured values** on that support (the signed differential itself).

Three conditions are evaluated on a fixed checkpoint, with the perturbation
applied at native npz resolution and the existing resize reused afterwards:

- ``A`` original input — the unperturbed baseline.
- ``B`` self-content permutation — the ``signed_q`` values *inside*
  ``instance_mask & polar_valid`` are permuted among themselves. The sample
  keeps its gray, its instance mask, its polar-valid support and every
  invalid region; the multiset of valid values is unchanged. This tests
  spatial-arrangement sensitivity only; it does **not** remove the
  sample-level polarization distribution.
- ``C`` matched-donor content transfer — the recipient keeps its gray, its
  instance mask, its polar-valid support and its invalid regions, and
  receives the *values* of a matched donor's target-interior valid region
  through a fixed deterministic resampling rule. It is a controlled numeric
  distribution intervention, not a physically real polarization image, and
  it does not on its own prove a causal effect.

Nothing here trains, fine-tunes or selects anything, and the test split is
never read. Donor matching is frozen before any model output is looked at.

Usage:
    # frozen v1 checkpoints: per-sample evidence + the two plastic fish
    python scripts/diagnose_material_mask.py --mode v1-reproduction \
        --data datasets/underwater_cls_fusion_v5_instance_mask \
        --checkpoints runs/train/cmx_v1_gray_s2027/material_fusion/best.pt \
                      runs/train/cmx_v1_gray_polar_s2027/material_fusion/best.pt \
        --split val --device 0 --output analysis/runs/material_mask_v5/legacy

    # controlled conditions on one fixed checkpoint
    python scripts/diagnose_material_mask.py --mode controlled \
        --data datasets/underwater_cls_fusion_v5_instance_mask \
        --checkpoint runs/train/cmx_v1_gray_polar_s2026/material_fusion/best.pt \
        --split val --device 0 --seeds 2026,2027,2028 \
        --output analysis/runs/material_mask_v5/controlled_e4_s2026
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_mask_dataset import (
    admit_dataset_for_protocol,
    admit_mask_dataset,
    load_mask_sample,
)
from models.material_fusion import (
    CLASS_NAMES,
    POOLING_INSTANCE_MASK,
    load_material_checkpoint,
    read_material_checkpoint_metadata,
    sha256_of_file,
)
from models.polar_fusion import read_manifest_split, torch_device_name

# ---------------------------------------------------------------------------
# Frozen protocol constants. These are part of the protocol and must not be
# changed after looking at any model output.

PROTOCOL_VERSION = "material_mask_diagnostic_v1"
CONDITIONS = ("A", "B", "C")
DEFAULT_SEEDS = (2026, 2027, 2028)

# Matching uses only data-computable quantities: the polarization support
# geometry, the target size and the target brightness. The true class, the
# model output, the correctness and the q mean/histogram are deliberately
# excluded (using them would either leak the label or pre-remove the signal
# under test).
FEATURE_NAMES = (
    "valid_coverage",
    "mask_area_ratio",
    "valid_centroid_y",
    "valid_centroid_x",
    "valid_bbox_h_ratio",
    "valid_bbox_w_ratio",
    "mean_gray",
)
FEATURE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0)

# The donor-distance ceiling is the 95th percentile of the nearest-neighbour
# distance *within the train split*, so the threshold comes from training
# metadata and never from validation model outputs.
THRESHOLD_PERCENTILE = 95.0

# Minimum polar-valid pixels for a sample to take part in B or C: with fewer
# than two valid pixels a permutation is the identity, and a transfer has no
# distribution to impose.
MIN_VALID_PIXELS = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["v1-reproduction", "controlled"], required=True)
    parser.add_argument("--data", default="datasets/underwater_cls_fusion_v5_instance_mask",
                        help="V5 instance-mask dataset (admitted before use).")
    parser.add_argument("--v4", default="datasets/underwater_cls_fusion_v4_band",
                        help="V4 dataset the V5 masks extend (parent binding).")
    parser.add_argument("--checkpoint", default=None,
                        help="controlled mode: the fixed checkpoint to diagnose.")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="v1-reproduction mode: frozen v1 checkpoints.")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Split to diagnose (test is refused).")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Deterministic intervention seeds (default: 2026..2028).")
    parser.add_argument("--output", required=True,
                        help="Output directory (must not exist).")
    args = parser.parse_args(argv)
    if args.mode == "controlled" and not args.checkpoint:
        parser.error("--mode controlled requires --checkpoint")
    if args.mode == "v1-reproduction" and not args.checkpoints:
        parser.error("--mode v1-reproduction requires --checkpoints")
    return args


# ---------------------------------------------------------------------------
# Sample features (data only)


@dataclass(frozen=True)
class SampleFeatures:
    sample_name: str
    split: str
    group_name: str
    class_id: int
    class_name: str
    source_frame: str
    object_index: int
    crop_h: int
    crop_w: int
    mask_pixels: int
    valid_pixels: int
    valid_coverage: float
    mask_area_ratio: float
    valid_centroid_y: float
    valid_centroid_x: float
    valid_bbox_h_ratio: float
    valid_bbox_w_ratio: float
    mean_gray: float

    def vector(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in FEATURE_NAMES], dtype=np.float64)


def _features_from_sample(row: dict, sample) -> SampleFeatures:
    mask = sample.instance_mask > 0
    valid = mask & (sample.valid > 0)
    h, w = mask.shape
    mask_pixels = int(mask.sum())
    valid_pixels = int(valid.sum())
    if valid_pixels:
        rows, cols = np.nonzero(valid)
        centroid_y = float(rows.mean()) / max(h - 1, 1)
        centroid_x = float(cols.mean()) / max(w - 1, 1)
        bbox_h = float(rows.max() - rows.min() + 1) / max(h, 1)
        bbox_w = float(cols.max() - cols.min() + 1) / max(w, 1)
        mean_gray = float(sample.gray[valid].mean()) / 255.0
    else:
        centroid_y = centroid_x = bbox_h = bbox_w = mean_gray = 0.0
    return SampleFeatures(
        sample_name=row["sample_name"],
        split=row["split"],
        group_name=row["group_name"],
        class_id=int(row["class_id"]),
        class_name=row["class_name"],
        source_frame=row["source_frame"],
        object_index=int(row["object_index"]),
        crop_h=h,
        crop_w=w,
        mask_pixels=mask_pixels,
        valid_pixels=valid_pixels,
        valid_coverage=valid_pixels / max(mask_pixels, 1),
        mask_area_ratio=mask_pixels / max(h * w, 1),
        valid_centroid_y=centroid_y,
        valid_centroid_x=centroid_x,
        valid_bbox_h_ratio=bbox_h,
        valid_bbox_w_ratio=bbox_w,
        mean_gray=mean_gray,
    )


def read_split_rows(data_root: Path, split: str) -> list[dict]:
    with (Path(data_root) / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"no {split!r} rows in {data_root / 'dataset_manifest.csv'}")
    return rows


def build_features(data_root: Path, split: str) -> tuple[list[dict], list[SampleFeatures], list]:
    """Manifest rows, matching features and the decoded samples of one split."""
    rows = read_split_rows(data_root, split)
    features: list[SampleFeatures] = []
    samples: list = []
    for row in rows:
        sample = load_mask_sample(Path(data_root) / row["npz_path"])
        samples.append(sample)
        features.append(_features_from_sample(row, sample))
    return rows, features, samples


# ---------------------------------------------------------------------------
# Donor protocol


def weighted_distance(a: np.ndarray, b: np.ndarray) -> float:
    weights = np.asarray(FEATURE_WEIGHTS, dtype=np.float64)
    delta = (a - b) * weights
    return float(np.sqrt(np.sum(delta * delta)))


def distance_threshold(train_features: list[SampleFeatures]) -> dict:
    """Donor ceiling from the *train* split's nearest-neighbour distances."""
    vectors = np.stack([f.vector() for f in train_features])
    if len(vectors) < 2:
        raise ValueError("the train split needs at least two samples for a threshold")
    diff = vectors[:, None, :] - vectors[None, :, :]
    weighted = diff * np.asarray(FEATURE_WEIGHTS, dtype=np.float64)
    dist = np.sqrt(np.sum(weighted * weighted, axis=-1))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    ceiling = float(np.percentile(nearest, THRESHOLD_PERCENTILE))
    return {
        "percentile": THRESHOLD_PERCENTILE,
        "max_distance": ceiling,
        "train_nearest_mean": float(nearest.mean()),
        "train_nearest_max": float(nearest.max()),
        "train_samples": int(len(vectors)),
    }


def match_donors(
    recipients: list[SampleFeatures],
    donors: list[SampleFeatures],
    max_distance: float,
) -> dict[str, dict]:
    """Greedy one-to-one matching in a fixed recipient order.

    Deterministic and independent of batch size and iteration order: the
    recipient order is sorted by sample name, each recipient takes the
    nearest still-unused eligible donor, ties break on the smaller donor
    sample name. A recipient with no eligible donor is ``unmatched`` — the
    protocol never substitutes an arbitrary donor. The result is not a
    bijection in general, and it is not claimed to be one.
    """
    ordered = sorted(recipients, key=lambda f: f.sample_name)
    available = {f.sample_name: f for f in donors}
    matches: dict[str, dict] = {}
    for recipient in ordered:
        if recipient.valid_pixels < MIN_VALID_PIXELS:
            matches[recipient.sample_name] = {
                "matched": False,
                "reason": (
                    f"recipient has {recipient.valid_pixels} polar-valid pixels "
                    f"(< {MIN_VALID_PIXELS}); there is no distribution to replace"
                ),
                "donor": None,
                "distance": None,
            }
            continue
        best = None
        for name in sorted(available):
            donor = available[name]
            if donor.sample_name == recipient.sample_name:
                continue
            if donor.valid_pixels < MIN_VALID_PIXELS:
                continue
            distance = weighted_distance(recipient.vector(), donor.vector())
            if distance > max_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, name)
        if best is None:
            matches[recipient.sample_name] = {
                "matched": False,
                "reason": (
                    "no eligible donor within the frozen distance ceiling "
                    f"({max_distance:.4f}); not substituting an arbitrary donor"
                ),
                "donor": None,
                "distance": None,
            }
            continue
        distance, donor_name = best
        del available[donor_name]
        matches[recipient.sample_name] = {
            "matched": True,
            "reason": "",
            "donor": donor_name,
            "distance": distance,
        }
    return matches


def matching_summary(
    recipients: list[SampleFeatures],
    donors: list[SampleFeatures],
    matches: dict[str, dict],
    threshold: dict,
) -> dict:
    """Coverage, unmatched reasons, per-class/per-group coverage, reuse."""
    matched = [name for name, m in matches.items() if m["matched"]]
    unmatched = {name: m["reason"] for name, m in matches.items() if not m["matched"]}
    used_donors: dict[str, int] = {}
    for m in matches.values():
        if m["matched"]:
            used_donors[m["donor"]] = used_donors.get(m["donor"], 0) + 1
    reused = {name: count for name, count in used_donors.items() if count > 1}

    by_class: dict[str, dict[str, int]] = {}
    by_group: dict[str, dict[str, int]] = {}
    for recipient in recipients:
        is_matched = matches[recipient.sample_name]["matched"]
        for key, bucket in ((recipient.class_name, by_class), (recipient.group_name, by_group)):
            entry = bucket.setdefault(key, {"recipients": 0, "matched": 0})
            entry["recipients"] += 1
            entry["matched"] += int(is_matched)

    distances = [m["distance"] for m in matches.values() if m["matched"]]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "split": recipients[0].split if recipients else "",
        "feature_names": list(FEATURE_NAMES),
        "feature_weights": list(FEATURE_WEIGHTS),
        "min_valid_pixels": MIN_VALID_PIXELS,
        "threshold": threshold,
        "recipients": len(recipients),
        "candidates": len(donors),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "matched_ratio": len(matched) / max(len(recipients), 1),
        "unmatched_reasons": unmatched,
        # A bijection requires every recipient matched to a distinct donor;
        # a partial (or empty) matching is explicitly not claimed to be one.
        "is_bijection": bool(matched)
        and len(used_donors) == len(matched) == len(recipients)
        and not reused,
        "donors_used": len(used_donors),
        "donor_reuse": reused,
        "distance": {
            "mean": float(np.mean(distances)) if distances else None,
            "max": float(np.max(distances)) if distances else None,
        },
        "per_class_coverage": by_class,
        "per_group_coverage": by_group,
        "notes": [
            "Donors come only from the diagnosed split; no train/val/test borrowing.",
            "No self-matching; the true class, the model output and the correctness "
            "are never used to pick a donor.",
            "The q mean and the q histogram are deliberately not matched, so the "
            "signal under test is not pre-removed.",
            "Capture conditions (depth, turbidity, illumination) are not recorded "
            "in this dataset and are reported as unknown, never inferred from the "
            "file name.",
            "Matching a few statistics does not remove shape, brightness or "
            "acquisition confounding.",
        ],
        "capture_conditions": "unknown (not recorded in the dataset metadata)",
    }


# ---------------------------------------------------------------------------
# Conditions


def _stable_rng(*parts) -> np.random.Generator:
    """A process-independent RNG from the protocol seed and sample names.

    ``hash()`` of a string is randomised per process, so it can never be used
    for a reproducible intervention; this derives the seed from a sha256 of
    the parts instead.
    """
    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 8], "big") for i in range(0, 32, 8)]
    return np.random.default_rng(np.random.SeedSequence(words))


def condition_b(sample, name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Permute signed_q inside instance_mask & polar_valid.

    The support, the gray, the invalid regions and the multiset of valid
    values are all preserved; ``abs_q`` is re-derived from the permuted
    signed values so the native signed/abs relation still holds exactly.
    """
    mask = sample.instance_mask > 0
    valid = mask & (sample.valid > 0)
    signed = sample.signed_q.copy()
    abs_q = sample.abs_q.copy()
    values = signed[valid]
    if values.size:
        rng = _stable_rng("B", seed, name)
        order = rng.permutation(values.size)
        signed[valid] = values[order]
        abs_q[valid] = np.abs(signed[valid])
    return signed, abs_q


def _transfer_order(count: int, seed: int, recipient: str, donor: str) -> np.ndarray:
    """Fixed, deterministic placement order of the donor's valid values.

    A seeded permutation of the donor's value indices, cycled with
    replacement when the donor has fewer valid pixels than the recipient.
    The rule depends only on the protocol seed and the two sample names, so
    it is independent of batch size and iteration order.
    """
    rng = _stable_rng("C", seed, recipient, donor)
    order = rng.permutation(count)
    return order


def condition_c(
    sample, name: str, donor_sample, donor_name: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer a matched donor's target-interior valid values.

    The recipient keeps its gray, its instance mask, its polar-valid support
    and its invalid regions. Only the values on the recipient's valid
    positions change. Donor values are placed through a seeded cyclic
    permutation (replacement is used exactly when the donor has fewer valid
    pixels than the recipient), and signed_q/abs_q move together.
    """
    mask = sample.instance_mask > 0
    valid = mask & (sample.valid > 0)
    recipient_index = np.flatnonzero(valid.ravel())
    donor_mask = donor_sample.instance_mask > 0
    donor_valid = donor_mask & (donor_sample.valid > 0)
    donor_values = donor_sample.signed_q[donor_valid]
    signed = sample.signed_q.copy()
    abs_q = sample.abs_q.copy()
    if recipient_index.size and donor_values.size:
        order = _transfer_order(donor_values.size, seed, name, donor_name)
        placed = donor_values[order][np.arange(recipient_index.size) % donor_values.size]
        flat_signed = signed.ravel()
        flat_signed[recipient_index] = placed
        signed = flat_signed.reshape(sample.signed_q.shape)
        flat_abs = abs_q.ravel()
        flat_abs[recipient_index] = np.abs(placed)
        abs_q = flat_abs.reshape(sample.abs_q.shape)
    return signed, abs_q


def intervention_quality(sample, abs_q: np.ndarray) -> dict:
    """Quality vector after an intervention.

    The support is unchanged, so ``valid_ratio`` (and the in-bounds and
    brightness ratios, which describe the gray pair) keep their values; only
    ``mean_abs_q`` is recomputed, from the intervened values. Both the
    original and the intervention vector are reported.
    """
    original = np.asarray(sample.quality, dtype=np.float64)
    valid = (sample.instance_mask > 0) & (sample.valid > 0)
    mean_abs_q = float(abs_q[valid].mean()) if valid.any() else 0.0
    return {
        "original_quality": original.tolist(),
        "intervention_quality": [
            float(original[0]), float(original[1]), float(original[2]), mean_abs_q,
        ],
        "support_unchanged": True,
        "note": (
            "support unchanged -> valid_ratio/in_bounds_ratio/"
            "brightness_valid_ratio keep their values; mean_abs_q is recomputed "
            "from the intervened abs_q. quality is not fed into the classifier."
        ),
    }


# ---------------------------------------------------------------------------
# Inference


def _tensors(sample, imgsz: int, variant: str):
    import cv2

    from core.material_fusion_dataset import variant_inputs
    from models.polar_fusion import _fusion_tensors

    gray_t, polar_t, quality_t = _fusion_tensors(sample, imgsz, cv2)
    rgb, aux = variant_inputs(variant, gray_t, polar_t)
    return rgb, aux, quality_t


def _mask_tensor(mask: np.ndarray, imgsz: int):
    import cv2

    resized = cv2.resize(
        np.ascontiguousarray(mask, dtype=np.float32), (imgsz, imgsz),
        interpolation=cv2.INTER_AREA,
    )
    return torch.from_numpy(np.ascontiguousarray(np.clip(resized, 0.0, 1.0))).float()


@torch.no_grad()
def predict_samples(
    model, metadata, rows, samples, device: str,
    perturbed: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    fallback=None,
):
    """Per-sample logits, predictions and routing for one condition.

    ``perturbed`` maps sample name -> (signed_q, abs_q) at native resolution;
    the perturbation is applied before the existing resize, so the training
    preprocessing is reused unchanged.
    """
    model.eval()
    results = []
    for row, sample in zip(rows, samples):
        signed, abs_q = (sample.signed_q, sample.abs_q)
        if perturbed is not None and row["sample_name"] in perturbed:
            signed, abs_q = perturbed[row["sample_name"]]
            sample = type(sample)(
                gray=sample.gray, signed_q=signed, abs_q=abs_q, valid=sample.valid,
                quality=sample.quality, class_id=sample.class_id,
                instance_mask=sample.instance_mask,
            )
        rgb, aux, quality_t = _tensors(sample, metadata.imgsz, metadata.variant)
        rgb = rgb.unsqueeze(0).to(device)
        aux = None if aux is None else aux.unsqueeze(0).to(device)
        mask = None
        if model.uses_instance_mask:
            mask = _mask_tensor(sample.instance_mask, metadata.imgsz).unsqueeze(0).to(device)
        if mask is None:
            logits = (model(rgb, aux) if model.is_dual else model(rgb))[0].cpu()
        else:
            logits = model(rgb, aux, mask)[0].cpu()
        prediction = int(logits.argmax().item())
        label = int(row["class_id"])
        is_valid = float(quality_t[0]) > 0.0
        routed = prediction
        fallback_used = False
        if fallback is not None and not is_valid:
            fallback_used = True
            routed = int(fallback(rgb).argmax(dim=1)[0].item())
        results.append(
            {
                "sample_name": row["sample_name"],
                "source_frame": row["source_frame"],
                "object_index": int(row["object_index"]),
                "group_name": row["group_name"],
                "split": row["split"],
                "label": label,
                "label_name": CLASS_NAMES[label],
                "logits": [float(v) for v in logits.tolist()],
                "prediction": prediction,
                "prediction_name": CLASS_NAMES[prediction],
                "correct": bool(prediction == label),
                "routed_prediction": routed,
                "routed_correct": bool(routed == label),
                "fallback_triggered": fallback_used,
                "polar_valid": bool(is_valid),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Reporting


def _prepare_output(output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            f"diagnostic output directory already exists: {output}; refusing to "
            "overwrite a previous diagnosis — choose a new --output"
        )
    output.mkdir(parents=True, exist_ok=False)
    return output


def _admit(args: argparse.Namespace, checkpoint_meta) -> dict:
    """Admit the mask dataset and bind it to *this* checkpoint.

    The gate follows the checkpoint's strictly validated pooling protocol:

    - ``instance_mask`` (v2): the checkpoint's concrete V5 fingerprint, its
      V4 parent fingerprint and its mask schema/resize versions are all
      required to match the data. "The same V4 parent" is not enough — two
      different V5 derivations of one V4 state are different datasets.
    - ``gap`` (v1): the V4 dataset is admitted and the checkpoint must have
      been trained on exactly that state; the V5 masks are used only as a
      verified extension of the same V4 state, and the v1 input path and GAP
      behaviour are unchanged.

    Nothing is written here, so a refusal leaves no report behind.
    """
    from core.material_mask_dataset import (
        POOLING_GAP,
        POOLING_INSTANCE_MASK,
        require_fingerprint,
    )

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    if checkpoint_meta.pooling_mode == POOLING_INSTANCE_MASK:
        # Both identities must be concrete before anything is compared: an
        # empty checkpoint fingerprint would make the admission accept any V5
        # derivation, and the report would then claim a binding that never
        # happened.
        require_fingerprint(
            checkpoint_meta.data_fingerprint, "checkpoint data_fingerprint"
        )
        require_fingerprint(
            checkpoint_meta.parent_data_fingerprint,
            "checkpoint parent_data_fingerprint",
        )
        report = admit_dataset_for_protocol(
            data_root,
            pooling_mode=POOLING_INSTANCE_MASK,
            expected_fingerprint=checkpoint_meta.data_fingerprint,
            expected_parent_fingerprint=checkpoint_meta.parent_data_fingerprint,
            expected_mask_schema=checkpoint_meta.mask_schema,
            expected_mask_resize_version=checkpoint_meta.mask_resize_version,
        )
        # The binding label is written only after the fingerprint really was
        # compared against this checkpoint's recorded V5 state.
        if report["audit_fingerprint"] != checkpoint_meta.data_fingerprint:
            raise ValueError(
                f"the admitted V5 dataset {report['audit_fingerprint']} is not the "
                f"one this checkpoint records "
                f"({checkpoint_meta.data_fingerprint}); refusing to label it bound"
            )
        report["binding"] = "checkpoint_v5_fingerprint_and_v4_parent"
        report["data_root"] = str(data_root)
        return report

    if checkpoint_meta.pooling_mode != POOLING_GAP:
        raise ValueError(
            f"unsupported checkpoint pooling_mode {checkpoint_meta.pooling_mode!r}"
        )

    # v1 historical path: verify the actual V4, bind the checkpoint to it, and
    # require the V5 masks to be a verified extension of that same state.
    from core.fusion_dataset import dataset_fingerprint
    from scripts.train_polar_fusion import validate_dataset

    # The formal-constant default is not applied twice here: the binding is
    # the explicit local comparison below, and no V4 admission is bypassed.
    report = admit_mask_dataset(data_root, expected_parent_fingerprint="")
    v4_root = Path(args.v4)
    if not v4_root.is_absolute():
        v4_root = PROJECT_ROOT / v4_root
    validate_dataset(v4_root)
    v4_fingerprint = dataset_fingerprint(v4_root)
    if report["v4_parent_fingerprint"] != v4_fingerprint:
        raise ValueError(
            f"the V5 masks extend V4 {report['v4_parent_fingerprint']} but the "
            f"V4 dataset at {v4_root} fingerprints as {v4_fingerprint}"
        )
    if (checkpoint_meta.data_fingerprint
            and checkpoint_meta.data_fingerprint != v4_fingerprint):
        raise ValueError(
            f"the checkpoint was trained on V4 {checkpoint_meta.data_fingerprint} "
            f"but the local V4 dataset is {v4_fingerprint}"
        )
    report["binding"] = "checkpoint_v4_fingerprint"
    report["data_root"] = str(data_root)
    report["v4_root"] = str(v4_root)
    return report


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarise_condition(records: list[dict]) -> dict:
    correct = sum(1 for r in records if r["correct"])
    routed_correct = sum(1 for r in records if r["routed_correct"])
    return {
        "samples": len(records),
        "accuracy": correct / max(len(records), 1),
        "routed_accuracy": routed_correct / max(len(records), 1),
        "fallback_triggered": sum(1 for r in records if r["fallback_triggered"]),
        "invalid_samples": sum(1 for r in records if not r["polar_valid"]),
    }


def paired_against_baseline(baseline: list[dict], other: list[dict]) -> dict:
    """Paired comparison on the same sample set (never a subset subtraction)."""
    base = {r["sample_name"]: r for r in baseline}
    fixed = [r["sample_name"] for r in other if not base[r["sample_name"]]["correct"] and r["correct"]]
    broken = [r["sample_name"] for r in other if base[r["sample_name"]]["correct"] and not r["correct"]]
    both_wrong = [r["sample_name"] for r in other if not base[r["sample_name"]]["correct"] and not r["correct"]]
    return {
        "samples": len(other),
        "fixed_by_intervention": len(fixed),
        "broken_by_intervention": len(broken),
        "both_wrong": len(both_wrong),
        "fixed_samples": fixed,
        "broken_samples": broken,
        "both_wrong_samples": both_wrong,
    }


def _load_checkpoint(path: Path, device: str):
    metadata = read_material_checkpoint_metadata(path)
    model, metadata, _ = load_material_checkpoint(
        path, variant=metadata.variant, imgsz=metadata.imgsz
    )
    return model.to(device), metadata


def run_controlled(args: argparse.Namespace) -> dict:
    device = torch_device_name(args.device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    model, metadata = _load_checkpoint(checkpoint, device)
    # Admission runs before any output exists: a refused diagnosis must not
    # leave behind a directory that could be mistaken for a report.
    admission = _admit(args, metadata)
    output = _prepare_output(Path(args.output))

    data_root = Path(admission["data_root"])
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds must name at least one seed")

    rows, features, samples = build_features(data_root, args.split)
    _, train_features, _ = build_features(data_root, "train")
    threshold = distance_threshold(train_features)
    matches = match_donors(features, features, threshold["max_distance"])
    summary = matching_summary(features, features, matches, threshold)
    summary["checkpoint"] = str(checkpoint)
    summary["checkpoint_sha256"] = sha256_of_file(checkpoint)
    summary["checkpoint_metadata"] = asdict(metadata)
    summary["data_fingerprint"] = admission["audit_fingerprint"]
    summary["v4_parent_fingerprint"] = admission["v4_parent_fingerprint"]
    summary["dataset_binding"] = admission.get("binding", "")
    summary["train_features_used_for_threshold"] = len(train_features)
    protocol_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    (output / "donor_protocol.json").write_text(protocol_text, encoding="utf-8")
    # Digest of the frozen donor protocol as written, so a report identifies
    # the exact matching protocol it used without changing the algorithm.
    donor_protocol_sha256 = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()

    fallback = None
    fallback_identity = ""
    fallback_error = ""
    fallback_weights = metadata.fallback_weights or ""
    if fallback_weights:
        from scripts.eval_material_fusion import load_fallback_gray

        fallback_file = PROJECT_ROOT / fallback_weights
        try:
            fallback, fallback_identity = load_fallback_gray(
                fallback_file, device, metadata.imgsz
            )
        except (FileNotFoundError, ValueError) as exc:
            fallback_error = str(exc)

    base = asdict
    feature_by_name = {f.sample_name: f for f in features}
    sample_by_name = {row["sample_name"]: sample for row, sample in zip(rows, samples)}

    evidence: list[dict] = []
    baseline_records = predict_samples(
        model, metadata, rows, samples, device, None, fallback
    )
    for record, feature in zip(baseline_records, features):
        record.update(
            {
                "condition": "A",
                "seed": None,
                "donor_id": None,
                "matched": None,
                "unmatched_reason": "",
                "match_distance": None,
                "features": base(feature),
                **intervention_quality(sample_by_name[record["sample_name"]],
                                       sample_by_name[record["sample_name"]].abs_q),
            }
        )
    evidence.extend(baseline_records)

    for seed in seeds:
        # B: self-content permutation
        perturbed_b = {
            row["sample_name"]: condition_b(
                sample_by_name[row["sample_name"]], row["sample_name"], seed
            )
            for row in rows
            if feature_by_name[row["sample_name"]].valid_pixels >= MIN_VALID_PIXELS
        }
        records_b = predict_samples(model, metadata, rows, samples, device, perturbed_b, fallback)
        for record in records_b:
            sample = sample_by_name[record["sample_name"]]
            signed, abs_q = perturbed_b.get(
                record["sample_name"], (sample.signed_q, sample.abs_q)
            )
            record.update(
                {
                    "condition": "B",
                    "seed": seed,
                    "donor_id": None,
                    "matched": record["sample_name"] in perturbed_b,
                    "unmatched_reason": "" if record["sample_name"] in perturbed_b
                    else "fewer than the minimum polar-valid pixels",
                    "match_distance": None,
                    "features": base(feature_by_name[record["sample_name"]]),
                    **intervention_quality(sample, abs_q),
                }
            )
        evidence.extend(records_b)

        # C: matched-donor content transfer
        perturbed_c: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        meta_c: dict[str, dict] = {}
        for row in rows:
            name = row["sample_name"]
            match = matches[name]
            if not match["matched"]:
                meta_c[name] = match
                continue
            donor_sample = sample_by_name[match["donor"]]
            perturbed_c[name] = condition_c(
                sample_by_name[name], name, donor_sample, match["donor"], seed
            )
            meta_c[name] = match
        records_c = predict_samples(model, metadata, rows, samples, device, perturbed_c, fallback)
        for record in records_c:
            name = record["sample_name"]
            match = meta_c[name]
            sample = sample_by_name[name]
            signed, abs_q = perturbed_c.get(name, (sample.signed_q, sample.abs_q))
            record.update(
                {
                    "condition": "C",
                    "seed": seed,
                    "donor_id": match["donor"],
                    "matched": bool(match["matched"]),
                    "unmatched_reason": match["reason"],
                    "match_distance": match["distance"],
                    "features": base(feature_by_name[name]),
                    **intervention_quality(sample, abs_q),
                }
            )
        evidence.extend(records_c)

    write_jsonl(output / "per_sample_evidence.jsonl", evidence)

    matched_names = sorted(name for name, m in matches.items() if m["matched"])
    by_condition: dict[str, dict] = {}
    for condition in CONDITIONS:
        for seed in (None, *seeds):
            key = f"{condition}" + (f"_seed{seed}" if seed is not None else "")
            subset = [
                r for r in evidence
                if r["condition"] == condition and r["seed"] == seed
                and (condition == "A" or r["sample_name"] in matched_names)
            ]
            if not subset:
                continue
            by_condition[key] = summarise_condition(subset)

    baseline_subset = [
        r for r in baseline_records if r["sample_name"] in matched_names
    ]
    paired: dict[str, dict] = {}
    for condition in ("B", "C"):
        for seed in seeds:
            key = f"{condition}_seed{seed}"
            subset = [
                r for r in evidence
                if r["condition"] == condition and r["seed"] == seed
                and r["sample_name"] in matched_names
            ]
            if subset:
                paired[key] = paired_against_baseline(baseline_subset, subset)

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "split": args.split,
        "test_split_used": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "donor_protocol_sha256": donor_protocol_sha256,
        "checkpoint_pooling_mode": metadata.pooling_mode,
        "dataset_binding": admission.get("binding", ""),
        "mask_schema_version": admission.get("mask_schema_version", ""),
        "mask_resize_version": admission.get("mask_resize_version", ""),
        "data_fingerprint": admission["audit_fingerprint"],
        "v4_parent_fingerprint": admission["v4_parent_fingerprint"],
        "seeds": seeds,
        "samples": len(rows),
        "matched_recipients": len(matched_names),
        "unmatched_recipients": len(rows) - len(matched_names),
        "fallback_identity": fallback_identity,
        "fallback_error": fallback_error,
        "conditions": by_condition,
        "paired_against_matched_baseline": paired,
        "pairing_note": (
            "Every intervention metric is paired against the baseline on the "
            "same matched recipient subset; a matched-subset intervention metric "
            "is never subtracted from the full-set baseline."
        ),
        "claims": [
            "B tests spatial-arrangement sensitivity of the polarization values "
            "inside the existing support; it does not remove the sample-level "
            "polarization distribution.",
            "C is a controlled numeric distribution intervention with the "
            "recipient's support preserved; it is not a physically real "
            "polarization image and does not by itself prove causation.",
            "The encoder's convolutions and attention already mix spatial "
            "information, so the mask-pooled model is not fully background "
            "isolated, and the q invalid support region can still carry shape "
            "information.",
            "Matching a few statistics does not remove shape, brightness or "
            "acquisition confounding; the perturbed inputs are partly "
            "out-of-distribution.",
        ],
        "outputs": {
            "per_sample_evidence": str(output / "per_sample_evidence.jsonl"),
            "donor_protocol": str(output / "donor_protocol.json"),
            "donor_protocol_sha256": donor_protocol_sha256,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_v1_reproduction(args: argparse.Namespace) -> dict:
    """Read-only per-sample predictions from frozen v1 checkpoints.

    Reproduces v1 exactly: the v1 GAP head, the V4 input path and the
    original admission. The V5 masks are used only as a verified
    side-channel for support statistics; they never replace the V4 input.
    """
    from core.fusion_dataset import dataset_fingerprint
    from models.material_fusion import POOLING_GAP
    from scripts.train_polar_fusion import validate_dataset

    device = torch_device_name(args.device)
    v4_root = Path(args.v4)
    if not v4_root.is_absolute():
        v4_root = PROJECT_ROOT / v4_root
    v4_report = validate_dataset(v4_root)
    v4_fingerprint = dataset_fingerprint(v4_root)

    mask_admission = None
    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"the verified V5 mask side-channel is required for the support "
            f"statistics: {data_root} is missing"
        )
    mask_admission = admit_mask_dataset(data_root, expected_parent_fingerprint="")
    if mask_admission["v4_parent_fingerprint"] != v4_fingerprint:
        raise ValueError(
            f"the V5 masks extend V4 "
            f"{mask_admission['v4_parent_fingerprint']} but the V4 dataset is "
            f"{v4_fingerprint}"
        )

    from core.fusion_dataset import load_fusion_sample

    # Features (support geometry) come from the admitted V5 masks; the model
    # inputs come from V4 and the V4 admission, never from V5.
    v5_rows, v5_features, _ = build_features(data_root, args.split)
    v4_rows = read_split_rows(v4_root, args.split)
    v4_names = [row["sample_name"] for row in v4_rows]
    v5_names = [row["sample_name"] for row in v5_rows]
    if v4_names != v5_names:
        raise ValueError(
            "the V5 manifest does not list the same samples in the same order as "
            "V4; the mask side-channel cannot be aligned to the V4 inputs"
        )
    features = v5_features
    samples = [load_fusion_sample(v4_root / row["npz_path"]) for row in v4_rows]

    records: list[dict] = []
    for checkpoint_arg in args.checkpoints:
        checkpoint = Path(checkpoint_arg)
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        metadata = read_material_checkpoint_metadata(checkpoint)
        if metadata.pooling_mode != POOLING_GAP:
            raise ValueError(
                f"{checkpoint} records pooling_mode {metadata.pooling_mode!r}; the "
                "v1 reproduction must run the original GAP head and never a "
                "mask-pooling model relabelled as v1"
            )
        if metadata.data_fingerprint and metadata.data_fingerprint != v4_fingerprint:
            raise ValueError(
                f"{checkpoint} was trained on {metadata.data_fingerprint} but the "
                f"local V4 dataset is {v4_fingerprint}"
            )
        model, metadata, _ = load_material_checkpoint(
            checkpoint, variant=metadata.variant, imgsz=metadata.imgsz,
            pooling_mode=POOLING_GAP,
        )
        model = model.to(device)
        fallback = None
        if metadata.fallback_weights:
            from scripts.eval_material_fusion import load_fallback_gray

            try:
                fallback, _ = load_fallback_gray(
                    PROJECT_ROOT / metadata.fallback_weights, device, metadata.imgsz
                )
            except (FileNotFoundError, ValueError):
                fallback = None
        predictions = predict_samples(model, metadata, v4_rows, samples, device, None, fallback)
        for record, feature in zip(predictions, features):
            record["checkpoint"] = str(checkpoint)
            record["run_id"] = str(checkpoint.parent.parent.name)
            record["features"] = asdict(feature)
        records.extend(predictions)

    # Created only after every checkpoint passed its validation, so a refusal
    # never leaves a directory that could look like a finished report.
    output = _prepare_output(Path(args.output))
    write_jsonl(output / "v1_per_sample_predictions.jsonl", records)

    # The two plastic fish: plastic_fish samples that E1 (gray) gets wrong and
    # E4 (gray_polar) gets right for the same seed. Identified from the actual
    # per-sample predictions, never from aggregate transition counts.
    by_run: dict[str, dict[str, dict]] = {}
    for record in records:
        by_run.setdefault(record["run_id"], {})[record["sample_name"]] = record
    gray_runs = {k: v for k, v in by_run.items() if "_gray_s" in k}
    polar_runs = {k: v for k, v in by_run.items() if "_gray_polar_s" in k}
    findings = []
    for gray_id, gray_records in sorted(gray_runs.items()):
        seed = gray_id.rsplit("_s", 1)[-1]
        polar_id = f"cmx_v1_gray_polar_s{seed}"
        polar_records = polar_runs.get(polar_id)
        if polar_records is None:
            continue
        for name, gray_record in sorted(gray_records.items()):
            polar_record = polar_records.get(name)
            if polar_record is None:
                continue
            if (
                gray_record["label_name"] == "plastic_fish"
                and not gray_record["correct"]
                and polar_record["correct"]
            ):
                findings.append(
                    {
                        "seed": int(seed),
                        "sample_name": name,
                        "source_frame": gray_record["source_frame"],
                        "object_index": gray_record["object_index"],
                        "group_name": gray_record["group_name"],
                        "label": gray_record["label"],
                        "label_name": gray_record["label_name"],
                        "gray_prediction": gray_record["prediction"],
                        "gray_prediction_name": gray_record["prediction_name"],
                        "gray_logits": gray_record["logits"],
                        "polar_prediction": polar_record["prediction"],
                        "polar_prediction_name": polar_record["prediction_name"],
                        "polar_logits": polar_record["logits"],
                        "gray_checkpoint": gray_record["checkpoint"],
                        "polar_checkpoint": polar_record["checkpoint"],
                        "features": gray_record["features"],
                    }
                )

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": "v1-reproduction",
        "split": args.split,
        "test_split_used": False,
        "v4_root": str(v4_root),
        "v4_fingerprint": v4_fingerprint,
        "v4_audit_passed": v4_report.get("audit_passed"),
        "v5_mask_admission": mask_admission,
        "checkpoints": [str(Path(c)) for c in args.checkpoints],
        "samples_per_run": len(records) // max(len(args.checkpoints), 1),
        "plastic_fish_fixed_by_polar": findings,
        "note": (
            "Identified from complete per-sample predictions of the frozen v1 "
            "checkpoints (v1 GAP head, V4 input, original admission); aggregate "
            "transition counts are not used to infer per-sample behaviour."
        ),
        "outputs": {"per_sample_predictions": str(output / "v1_per_sample_predictions.jsonl")},
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.split == "test":
        print("diagnostics refuse the test split", file=sys.stderr)
        return 2
    try:
        report = (
            run_controlled(args) if args.mode == "controlled"
            else run_v1_reproduction(args)
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"diagnosis refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
