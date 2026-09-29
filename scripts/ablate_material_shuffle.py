"""Shuffled-polarization inference ablation for a fixed material-fusion checkpoint.

Inference-only: a trained checkpoint is evaluated on one split with the
polarization content replaced by a donor sample's polarization. Nothing is
trained, fine-tuned or selected; no optimizer is ever created.

Design (fixed by the experiment protocol):

- The donor mapping is a fixed-point-free bijection (derangement) over the
  *full split index*, generated from ``numpy.random.default_rng(seed)`` and
  reproducible; it never depends on the batch size, and every sample's
  donor index is recorded to the report directory.
- ``signed_q``, ``abs_q`` and ``valid`` all come from the same donor sample
  (mode ``full``), together with that donor's quality vector — validity
  statistics and fallback routing read the quality vector, so it must move
  with the channels it describes. Mode ``content_only`` keeps the sample's
  own valid channel (and its own quality vector) and only replaces the two
  polarization channels, isolating the effect on polarization *content* from
  the effect on the valid mask and therefore on fallback routing.
- Seeds default to 2026..2030. Each seed is reported against the same
  unshuffled baseline: raw-logit deltas (mean/max absolute, per class) and
  prediction changes (count, ratio, full from->to transition matrix) —
  never accuracy alone.
- The test split is refused; this ablation never becomes a tuning loop.

Usage:
    python scripts/ablate_material_shuffle.py --checkpoint <best.pt> \
        --split val --device 0 --output analysis/runs/material_shuffle_e4/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_fusion_dataset import (
    CLASS_NAMES,
    MaterialFusionDataset,
    admit_material_dataset,
    variant_inputs,
)
from models.material_fusion import load_material_checkpoint, variant_is_dual
from models.polar_fusion import read_manifest_split, torch_device_name

DEFAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Split to ablate (default: val; test is refused so "
                             "the ablation can never become hidden tuning).")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma list of shuffle seeds (default: %(default)s).")
    parser.add_argument("--mode", default="both", choices=["full", "content_only", "both"],
                        help="full: donor supplies signed_q/abs_q/valid; "
                             "content_only: donor supplies only signed_q/abs_q.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--data", default="datasets/underwater_cls_fusion_v4_band",
                        help="Dataset root (default: the V4 fusion dataset).")
    parser.add_argument("--fallback-weights", default=None,
                        help="Historical gray Model B used to report the routing "
                             "effect of valid-mask changes (optional).")
    parser.add_argument("--output", required=True,
                        help="Report JSON path (must not exist).")
    return parser.parse_args(argv)


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A reproducible permutation of range(n) with no fixed point."""
    if n < 2:
        raise ValueError(
            f"a fixed-point-free bijection needs at least 2 samples, got {n}"
        )
    identity = np.arange(n)
    for _ in range(10_000):
        perm = rng.permutation(n)
        if not np.any(perm == identity):
            return perm
    raise RuntimeError("could not build a derangement after 10000 attempts")


class DonorShuffleDataset(torch.utils.data.Dataset):
    """MaterialFusionDataset view where sample i reads donor[i]'s polar.

    Mode ``full`` replaces the whole polarization measurement, so the
    donor's *quality vector* comes along with its channels: validity
    statistics and fallback routing read that vector, and keeping the
    sample's own vector would report "no validity change" for every sample
    by construction. Mode ``content_only`` keeps the sample's own valid
    channel and therefore its own quality vector.
    """

    def __init__(self, paths: list[Path], variant: str, imgsz: int,
                 donor: np.ndarray, content_only: bool):
        from models.polar_fusion import FusionClsDataset

        self._base = FusionClsDataset(paths, imgsz=imgsz)
        self._donor = donor
        self._content_only = content_only
        self.variant = variant
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int):
        gray_t, own_polar, own_quality, class_id = self._base[index]
        _, donor_polar, donor_quality, _ = self._base[int(self._donor[index])]
        if self._content_only:
            polar = torch.stack(
                (donor_polar[0], donor_polar[1], own_polar[2]), dim=0
            )
            quality = own_quality
        else:
            polar = donor_polar
            quality = donor_quality
        rgb, aux = variant_inputs(self.variant, gray_t, polar)
        return rgb, aux, quality, class_id


def _transition_matrix(labels: list[int], base_preds: list[int],
                       new_preds: list[int]) -> dict:
    transitions: dict[str, dict[str, int]] = {}
    for base, new in zip(base_preds, new_preds):
        if base != new:
            row = transitions.setdefault(CLASS_NAMES[base], {})
            row[CLASS_NAMES[new]] = row.get(CLASS_NAMES[new], 0) + 1
    return transitions


@torch.no_grad()
def _forward_all(model, loader, device: str) -> dict:
    model.eval()
    logits_all: list[list[float]] = []
    labels_all: list[int] = []
    preds: list[int] = []
    valid_all: list[bool] = []
    for batch in loader:
        rgb, aux, quality, labels = batch
        logits = model(rgb.to(device), aux.to(device))
        logits_all.extend(logits.cpu().tolist())
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.tolist())
        valid_all.extend(bool(v) for v in (quality[:, 0] > 0).tolist())
    return {"logits": logits_all, "preds": preds, "labels": labels_all,
            "valid": valid_all}


@torch.no_grad()
def _fallback_preds(fallback, loader, device: str) -> list[int]:
    preds: list[int] = []
    for batch in loader:
        preds.extend(fallback(batch[0].to(device)).argmax(dim=1).cpu().tolist())
    return preds


def routed_predictions(base_preds: list[int], fallback_preds: list[int],
                       valid_flags: list[bool]) -> list[int]:
    """Fallback output replaces the model exactly at *invalid*-polar samples.

    ``valid_flags`` follows the dataset's own convention (``True`` == the
    sample's ``valid_ratio`` is above zero), matching the canonical routing
    in ``scripts/eval_material_fusion.py``: invalid samples bypass the model
    and go through the fallback, valid samples go through the model only.
    """
    if len(fallback_preds) != len(valid_flags) or len(base_preds) != len(valid_flags):
        raise ValueError("fallback/flag/prediction lengths must match the split")
    routed = list(base_preds)
    for index, is_valid in enumerate(valid_flags):
        if not is_valid:
            routed[index] = fallback_preds[index]
    return routed


def run_ablation(args: argparse.Namespace) -> dict:
    from scripts.eval_material_fusion import load_fallback_gray

    device = torch_device_name(args.device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    from models.material_fusion import read_material_checkpoint_metadata

    metadata = read_material_checkpoint_metadata(checkpoint)
    if not variant_is_dual(metadata.variant):
        raise ValueError(
            f"checkpoint variant {metadata.variant!r} has no polarization "
            "input; the shuffled-polarization ablation is not applicable"
        )
    model, metadata, _ = load_material_checkpoint(
        checkpoint, variant=metadata.variant
    )
    model = model.to(device)

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    data_report = admit_material_dataset(data_root, metadata.data_fingerprint)
    paths, _, _ = read_manifest_split(data_root, args.split)
    own_loader = torch.utils.data.DataLoader(
        MaterialFusionDataset(paths, variant=metadata.variant, imgsz=metadata.imgsz),
        batch_size=args.batch, shuffle=False, num_workers=0,
    )
    baseline = _forward_all(model, own_loader, device)

    fallback = None
    fallback_identity = ""
    fallback_path = args.fallback_weights
    if fallback_path:
        fallback_file = Path(fallback_path)
        if not fallback_file.is_absolute():
            fallback_file = PROJECT_ROOT / fallback_file
        fallback, fallback_identity = load_fallback_gray(
            fallback_file, device, metadata.imgsz
        )

    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if output.exists():
        raise FileExistsError(f"report {output} already exists; refusing to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        donor = derangement(len(paths), rng)
        seed_entry: dict = {"shuffle_seed": seed, "n_samples": len(paths)}
        for mode in (["full", "content_only"] if args.mode == "both" else [args.mode]):
            loader = torch.utils.data.DataLoader(
                DonorShuffleDataset(
                    paths, metadata.variant, metadata.imgsz, donor,
                    content_only=(mode == "content_only"),
                ),
                batch_size=args.batch, shuffle=False, num_workers=0,
            )
            result = _forward_all(model, loader, device)
            base_logits = np.asarray(baseline["logits"])
            new_logits = np.asarray(result["logits"])
            delta = np.abs(new_logits - base_logits)
            changed = [
                (b, n) for b, n in zip(baseline["preds"], result["preds"]) if b != n
            ]
            mode_entry = {
                "raw_logit_delta": {
                    "mean_abs": float(delta.mean()),
                    "max_abs": float(delta.max()),
                    "per_class_mean_abs": [
                        float(delta[:, cls].mean()) for cls in range(len(CLASS_NAMES))
                    ],
                },
                "changed_predictions": {
                    "count": len(changed),
                    "ratio": len(changed) / max(len(paths), 1),
                    "transitions": _transition_matrix(
                        baseline["labels"], baseline["preds"], result["preds"]
                    ),
                },
                "accuracy": float(np.mean(
                    [l == p for l, p in zip(result["labels"], result["preds"])]
                )),
                "baseline_accuracy": float(np.mean(
                    [l == p for l, p in zip(baseline["labels"], baseline["preds"])]
                )),
            }
            if mode == "full":
                valid_changed = sum(
                    1 for a, b in zip(baseline["valid"], result["valid"]) if a != b
                )
                mode_entry["valid_mask_changed"] = valid_changed
                if fallback is not None:
                    routed_base = routed_predictions(
                        baseline["preds"], _fallback_preds(fallback, own_loader, device),
                        baseline["valid"],
                    )
                    routed_new = routed_predictions(
                        result["preds"], _fallback_preds(fallback, loader, device),
                        result["valid"],
                    )
                    mode_entry["routed_changed_predictions"] = sum(
                        1 for b, n in zip(routed_base, routed_new) if b != n
                    )
            seed_entry[mode] = mode_entry
        donor_path = output.parent / f"donors_seed{seed}.json"
        donor_path.write_text(
            json.dumps(
                {"shuffle_seed": seed, "split": args.split,
                 "donor_index": donor.tolist()},
                ensure_ascii=False, indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        seed_entry["donor_mapping_file"] = str(donor_path)
        per_seed.append(seed_entry)

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_variant": metadata.variant,
        "checkpoint_epoch": metadata.epoch,
        "split": args.split,
        "test_split_used": False,
        "n_samples": len(paths),
        "mode": args.mode,
        "seeds": seeds,
        "baseline": {
            "accuracy": float(np.mean(
                [l == p for l, p in zip(baseline["labels"], baseline["preds"])]
            )),
            "invalid_samples": sum(1 for v in baseline["valid"] if not v),
        },
        "fallback_identity": fallback_identity,
        "data_fingerprint": data_report["audit_fingerprint"],
        "per_seed": per_seed,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_ablation(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ablation refused: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    print(f"shuffle ablation report: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
