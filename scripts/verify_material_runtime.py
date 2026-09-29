"""Bounded engineering verification for the material_fusion runtime (4A).

Two layers, one CLI:

- Layer A — same V5 NPZ inputs: the runtime adapter is fed the *same* val
  npz samples the offline dataset yields, and the tensors and eval-mode
  logits of the four engineering variants (seed 2026) are compared. Tensors
  must be exactly equal; logits must agree within the pre-fixed
  ``atol=rtol=1e-5``. A failure is localised, never papered over by
  widening the tolerance.

- Layer B — real stereo frames: the configured Model A runs on the
  rectified val frames, the band SGBM + polar path builds the runtime
  inputs, and the four variants classify them through the production
  engine (with the independent gray fallback routing). For the same frames
  the offline V5 generation is replayed from the GT annotations and
  compared against the stored npz samples (gray crop, crop window, mask,
  quality) — reproducible differences and their sources are recorded.

This verifier proves *plumbing*, not model quality: it says nothing about
polarization benefit, independent generalization, or FPS.

Bounded budget (hard limits): at most 16 NPZ samples, 8 real frames, and
15 minutes of measured real-forward time per invocation.

Usage:

    python scripts/verify_material_runtime.py \
        --config configs/experiments/material_runtime_v2_mask.yaml \
        --device 0 --max-samples 16 --max-frames 8 \
        --output-dir analysis/runs/material_runtime_v2_mask/4a_<tag>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_mask_dataset import admit_dataset_for_protocol
from core.material_fusion_dataset import CLASS_NAMES, MaterialFusionDataset
from core.stereo_matching import StereoMatcher
from models.material_classifier import (
    MASK_SOURCE_NPZ,
    MaterialInstanceInput,
    load_material_runtime_config,
    material_input_tensors,
)
from models.polar_fusion import read_manifest_split

# Hard budget constants (the CLI caps at these, never above).
MAX_SAMPLES_LIMIT = 16
MAX_FRAMES_LIMIT = 8
REAL_FORWARD_BUDGET_S = 15 * 60.0
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5

ENGINEERING_VARIANTS = ("gray", "gray_gray_valid", "gray_valid", "gray_polar")


# ---------------------------------------------------------------------------
# Small report helpers (unit-tested in test_material_runtime.py).


def prepare_output_dir(output_dir: str | Path) -> Path:
    """Create the report directory; an existing directory is refused."""
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(
            f"output directory {path} already exists; verification reports "
            "are never overwritten — pass a fresh --output-dir"
        )
    path.mkdir(parents=True)
    return path


def write_report_bytes(directory: str | Path, name: str, payload) -> Path:
    """Write one JSON report as UTF-8 bytes; the caller digests read_bytes()."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = Path(directory) / name
    path.write_bytes(text.encode("utf-8"))
    return path


def compare_logits(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = LOGIT_ATOL,
    rtol: float = LOGIT_RTOL,
) -> float:
    """Allclose logits with the pre-fixed tolerances; return the max |error|.

    Raises AssertionError on any mismatch — a difference is something to
    locate, never a reason to widen the tolerance.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"logit shape mismatch: runtime {tuple(actual.shape)} vs offline "
            f"{tuple(expected.shape)}"
        )
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs()
        worst = float(diff.max())
        bad = int(diff.argmax())
        raise AssertionError(
            f"logits exceed atol={atol}, rtol={rtol}: max |diff| {worst:.3e} "
            f"at flat index {bad}"
        )
    return float((actual - expected).abs().max())


def sha256_of_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixed selection rules (decided before any model output is inspected).


def select_val_npz(manifest_rows: list[dict], limit: int) -> list[dict]:
    """The first ``limit`` val rows in manifest order."""
    val_rows = [row for row in manifest_rows if row["split"] == "val"]
    return val_rows[:limit]


def select_val_frames(manifest_rows: list[dict], limit: int) -> list[str]:
    """Evenly spread unique val frames in sorted order."""
    frames = sorted({row["source_frame"] for row in manifest_rows
                     if row["split"] == "val"})
    if len(frames) <= limit:
        return frames
    stride = len(frames) / limit
    picked = sorted({frames[min(int(i * stride), len(frames) - 1)]
                     for i in range(limit)})
    return picked[:limit]


def right_image_path(left_path: Path) -> Path | None:
    """The paired right image of a Rectified_v2 left image (never guessed
    beyond the recorded directory convention)."""
    parts = list(left_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "left":
            parts[index] = "right"
            candidate = Path(*parts)
            return candidate if candidate.is_file() else None
    return None


# ---------------------------------------------------------------------------
# Layer A: runtime adapter vs offline dataset on the same npz samples.


def npz_to_material_input(npz_path: Path, sample) -> MaterialInstanceInput:
    """One offline-identity runtime input built from the npz arrays."""
    height, width = sample.gray.shape
    return MaterialInstanceInput(
        frame_id=npz_path.stem,
        instance_id=0,
        bbox=(0, 0, width, height),
        gray=sample.gray,
        signed_q=sample.signed_q,
        abs_q=sample.abs_q,
        polar_valid=sample.polar_valid.astype(np.uint8),
        instance_mask=sample.instance_mask,
        quality=sample.quality,
        polar_invalid=False,
        invalid_reason="",
        mask_source=MASK_SOURCE_NPZ,
    )


@torch.no_grad()
def run_layer_a(classifiers, npz_paths, device, imgsz, batch: int) -> dict:
    """Tensor identity and logit agreement for all four variants."""
    report: dict = {
        "samples": len(npz_paths),
        "selection_rule": "first N val rows in manifest order",
        "device": device,
        "imgsz": imgsz,
        "logit_atol": LOGIT_ATOL,
        "logit_rtol": LOGIT_RTOL,
        "variants": {},
        "status": "PASSED",
    }
    paths = list(npz_paths)
    for variant, classifier in classifiers.items():
        variant_report: dict = {"tensor_equal": {}, "logit_max_abs_err": {}}
        dataset = MaterialFusionDataset(
            paths, variant=variant, imgsz=imgsz, with_instance_mask=True
        )
        runtime_logits: list[torch.Tensor] = []
        offline_logits: list[torch.Tensor] = []
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            samples = [__import__("core.material_mask_dataset", fromlist=["load_mask_sample"]).load_mask_sample(p) for p in chunk]
            # Runtime tensors through the adapter.
            rgbs, auxs, masks = [], [], []
            for sample in samples:
                rgb, aux, mask, _quality = material_input_tensors(
                    npz_to_material_input(chunk[0], sample), imgsz, variant
                )
                rgbs.append(rgb)
                auxs.append(aux)
                masks.append(mask)
            rgb = torch.stack(rgbs).to(device)
            aux = torch.stack(auxs).to(device) if auxs[0] is not None else None
            mask = torch.stack(masks).to(device)
            model = classifier.model
            if variant == "gray":
                logits = model(rgb, None, mask)
            else:
                logits = model(rgb, aux, mask)
            runtime_logits.append(logits.cpu())

            # Offline tensors through the dataset + a direct model forward.
            offline = [dataset[start + i] for i in range(len(chunk))]
            if variant == "gray":
                o_rgb = torch.stack([row[0] for row in offline])
                o_mask = torch.stack([row[-1] for row in offline])
                o_aux = None
                variant_report["tensor_equal"][f"batch_{start}"] = bool(
                    torch.equal(rgb.cpu(), o_rgb) and torch.equal(mask.cpu(), o_mask)
                )
            else:
                o_rgb = torch.stack([row[0] for row in offline])
                o_aux = torch.stack([row[1] for row in offline])
                o_mask = torch.stack([row[-1] for row in offline])
                variant_report["tensor_equal"][f"batch_{start}"] = bool(
                    torch.equal(rgb.cpu(), o_rgb)
                    and torch.equal(aux.cpu(), o_aux)
                    and torch.equal(mask.cpu(), o_mask)
                )
            if variant == "gray":
                o_logits = model(o_rgb.to(device), None, o_mask.to(device))
            else:
                o_logits = model(
                    o_rgb.to(device), o_aux.to(device), o_mask.to(device)
                )
            offline_logits.append(o_logits.cpu())

        runtime_all = torch.cat(runtime_logits)
        offline_all = torch.cat(offline_logits)
        try:
            max_err = compare_logits(runtime_all, offline_all)
            variant_report["logit_max_abs_err"] = max_err
            variant_report["logits_within_tolerance"] = True
        except AssertionError as exc:
            variant_report["logit_max_abs_err"] = str(exc)
            variant_report["logits_within_tolerance"] = False
            report["status"] = "FAILED"
        report["variants"][variant] = variant_report
    return report


# ---------------------------------------------------------------------------
# Layer B: real frames — GT replay + predicted-mask production forward.


class _ForwardBudget:
    """Hard wall-clock budget over the measured real-forward time."""

    def __init__(self, limit_seconds: float):
        self.limit = limit_seconds
        self.spent = 0.0

    def measure(self, seconds: float) -> None:
        self.spent += seconds
        if self.spent > self.limit:
            raise RuntimeError(
                f"the bounded real-forward budget ({self.limit:.0f}s) was "
                f"exceeded ({self.spent:.1f}s); stopping instead of growing "
                "the run"
            )


def replay_gt_for_frame(
    frame: str,
    rows: list[dict],
    data_root: Path,
    matcher: StereoMatcher,
    band_margin: int,
    npz_by_name: dict[str, Path],
    full_image_threshold: float = 0.9,
) -> dict:
    """Replay the offline V5 generation for one frame and compare its npz.

    Reuses the generator's own helpers (read_yolo_polygons, build_object_mask,
    make_crop_window) so this is the same math, not a re-implementation.
    """
    import cv2

    from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons
    from scripts.prepare_cls_fusion_dataset import build_object_mask

    left_path = PROJECT_ROOT / rows[0]["left_path"]
    right_path = right_image_path(left_path)
    if right_path is None:
        return {"frame": frame, "status": "UNAVAILABLE",
                "reason": "paired right image not found for the recorded "
                          "left_path; the frame is not verified, not guessed"}

    from core.fusion_dataset import quality_vector_from_result
    from core.polar_compute import compute_polar_features
    from core.stereo_matching import to_gray_u8

    # The shared to_gray_u8 boundary (same as the engine): the loaded pair is
    # normalized to 2-D uint8 whatever the file's channel layout is.
    left_gray = to_gray_u8(cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED))
    right_gray = to_gray_u8(cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED))
    if left_gray is None or right_gray is None:
        return {"frame": frame, "status": "UNAVAILABLE",
                "reason": "rectified pair unreadable"}
    height, width = left_gray.shape

    label_path = PROJECT_ROOT / rows[0]["label_path"]
    polygons = read_yolo_polygons(label_path, width, height)
    if len(polygons) < len(rows):
        return {"frame": frame, "status": "UNAVAILABLE",
                "reason": f"label file has {len(polygons)} objects, manifest "
                          f"lists {len(rows)}"}

    bboxes = []
    windows = []
    for row in rows:
        polygon = polygons[int(row["object_index"])].polygon
        # The generation builds band bboxes from the polygon extrema (not
        # from padded crop windows) — this replay must use the same math or
        # the band-restricted SGBM rows differ and so does the disparity.
        xs = [x for x, _ in polygon]
        ys = [y for _, y in polygon]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        windows.append(make_crop_window(polygon, width, height, 10))
    bands = __import__(
        "core.stereo_matching", fromlist=["build_horizontal_bands"]
    ).build_horizontal_bands(
        bboxes, image_height=height, vertical_margin=band_margin
    )
    stereo = matcher.compute_bands(
        left_gray, right_gray, bands, full_image_threshold=full_image_threshold
    )

    comparisons = []
    for row, bbox, window in zip(rows, bboxes, windows):
        polygon = polygons[int(row["object_index"])].polygon
        mask = build_object_mask(polygon, width, height)
        stats = matcher.instance_stats(stereo, mask, int(row["object_index"]))
        entry = {
            "sample_name": row["sample_name"],
            "crop_window_equal": None,
            "gray_equal": None,
            "mask_equal": None,
            "polar_valid_equal": None,
            "quality_max_diff": None,
            "status": "UNAVAILABLE",
        }
        npz_path = npz_by_name.get(row["sample_name"])
        if npz_path is None:
            entry["reason"] = "no npz sample for this object"
            comparisons.append(entry)
            continue
        from core.material_mask_dataset import load_mask_sample

        npz = load_mask_sample(npz_path)
        x1, y1, x2, y2 = window.x1, window.y1, window.x2, window.y2
        crop_gray = left_gray[y1:y2, x1:x2]
        entry["crop_window_equal"] = (
            (x1, y1, x2, y2)
            == tuple(int(row[c]) for c in ("crop_x1", "crop_y1", "crop_x2", "crop_y2"))
        )
        entry["gray_equal"] = bool(np.array_equal(crop_gray, npz.gray))
        entry["mask_equal"] = bool(
            np.array_equal(mask[y1:y2, x1:x2], npz.instance_mask)
        )
        if stats.valid:
            polar = compute_polar_features(
                left_gray, right_gray, stereo.disparity,
                object_mask=mask, disparity_valid=stereo.valid,
            )
            crop_valid = polar.valid_mask[y1:y2, x1:x2].astype(np.uint8)
            entry["polar_valid_equal"] = bool(
                np.array_equal(crop_valid, npz.valid)
            )
            quality, _ = quality_vector_from_result(polar, mask)
            entry["quality_max_diff"] = float(
                np.abs(quality - npz.quality).max()
            )
        else:
            entry["polar_valid_equal"] = None
            entry["reason"] = f"stereo invalid: {stats.reason}"
        passed = (
            entry["crop_window_equal"]
            and entry["gray_equal"]
            and entry["mask_equal"]
            and (entry["polar_valid_equal"] is not False)
            and (entry["quality_max_diff"] is None or entry["quality_max_diff"] <= 1e-6)
        )
        entry["status"] = "PASSED" if passed else "MISMATCH"
        comparisons.append(entry)
    status = "PASSED"
    if any(entry["status"] != "PASSED" for entry in comparisons):
        status = "MISMATCH" if any(
            entry["status"] == "MISMATCH" for entry in comparisons
        ) else "PARTIAL"
    return {"frame": frame, "right_path": str(right_path), "status": status,
            "objects": comparisons}


@torch.no_grad()
def run_layer_b(
    classifiers,
    engine,
    npz_root: Path,
    manifest_rows: list[dict],
    matcher: StereoMatcher,
    band_margin: int,
    max_frames: int,
    budget: _ForwardBudget,
    full_image_threshold: float = 0.9,
) -> dict:
    """GT replay + the real production forward over the selected frames."""
    report: dict = {
        "selection_rule": "evenly spread unique val frames in sorted order",
        "sync_status": "unknown",
        "sync_note": "no measurable timestamps in this verifier; sync is "
                     "reported unknown, never 0 ms or assumed synced",
        "gt_replay": [],
        "frames": [],
        "fallback_totals": {},
        "status": "PASSED",
    }
    val_rows = [row for row in manifest_rows if row["split"] == "val"]
    frames = select_val_frames(manifest_rows, max_frames)
    report["selected_frames"] = frames
    npz_by_name: dict[str, Path] = {}
    for row in val_rows:
        npz_by_name.setdefault(row["sample_name"], npz_root / row["npz_path"])

    for frame in frames:
        rows = [row for row in val_rows if row["source_frame"] == frame]
        gt = replay_gt_for_frame(
            frame, rows, npz_root.parent, matcher, band_margin, npz_by_name,
            full_image_threshold=full_image_threshold,
        )
        report["gt_replay"].append(gt)
        if gt["status"] == "UNAVAILABLE":
            continue

        left_path = PROJECT_ROOT / rows[0]["left_path"]
        right_path = right_image_path(left_path)
        assert right_path is not None  # replay already checked this
        import cv2

        from core.stereo_matching import to_gray_u8

        # to_gray_u8: the Ultralytics-patched cv2.imread may hand back an
        # (H, W, 1) array; the engine normalizes through the same boundary.
        left = to_gray_u8(cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED))
        right = to_gray_u8(cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED))
        frame_report: dict = {"frame": frame, "variants": {}}
        started = time.perf_counter()
        for variant, classifier in classifiers.items():
            engine.model_b = classifier
            detailed = engine.process_frame_detailed(
                left, right, sync_skew_ms=None, already_rectified=True,
                frame_id=frame,
            )
            inputs = detailed.material_inputs
            results = detailed.material_results
            instance_ids = [inst.id for inst in detailed.instances]
            depth_ids = [d["instance_id"] for d in detailed.depths]
            input_ids = [s.instance_id if s is not None else None for s in inputs]
            result_ids = [r.instance_id for r in results]
            association_ok = (
                instance_ids == depth_ids == input_ids == result_ids
            )
            fallback_used = sum(1 for r in results if r.used_fallback)
            fallback_reasons = sorted(
                {r.fallback_reason for r in results if r.used_fallback}
            )
            frame_report["variants"][variant] = {
                "instances": len(instance_ids),
                "instance_ids": instance_ids,
                "association_ok": association_ok,
                "used_fallback": fallback_used,
                "fallback_reasons": fallback_reasons,
                "predicted": [r.class_id if r.valid else None for r in results],
            }
            report["fallback_totals"][variant] = (
                report["fallback_totals"].get(variant, 0) + fallback_used
            )
            if not association_ok:
                report["status"] = "FAILED"
        budget.measure(time.perf_counter() - started)
        report["frames"].append(frame_report)
    report["real_forward_seconds"] = round(budget.spent, 2)
    if any(gt["status"] == "MISMATCH" for gt in report["gt_replay"]):
        report["gt_replay_status"] = "MISMATCH"
    return report


def snapshot_state(module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in list(module.named_parameters())
        + list(module.named_buffers())
    }


# ---------------------------------------------------------------------------
# Main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_LIMIT)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES_LIMIT)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_samples > MAX_SAMPLES_LIMIT or args.max_frames > MAX_FRAMES_LIMIT:
        print(
            f"refused: the bounded budget is {MAX_SAMPLES_LIMIT} npz samples "
            f"and {MAX_FRAMES_LIMIT} frames",
            file=sys.stderr,
        )
        return 2

    try:
        output_dir = prepare_output_dir(args.output_dir)
    except FileExistsError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    from gui.inference_engine import DualStageInferenceEngine

    config = load_material_runtime_config(args.config)

    # --- Identity verification before anything else runs ---
    problems: list[str] = []
    identities: dict = {"model_a": {}, "fallback": {}, "candidates": []}
    model_a_file = PROJECT_ROOT / config.model_a_path
    if not model_a_file.is_file():
        problems.append(f"Model A missing: {model_a_file}")
    else:
        digest = sha256_of_file(model_a_file)
        identities["model_a"] = {
            "path": config.model_a_path,
            "sha256": digest,
            "matches_config": digest == config.model_a_sha256,
        }
        if digest != config.model_a_sha256:
            problems.append("Model A sha256 mismatch")
    fallback_file = PROJECT_ROOT / config.fallback_path
    if not fallback_file.is_file():
        problems.append(f"fallback missing: {fallback_file}")
    else:
        digest = sha256_of_file(fallback_file)
        identities["fallback"] = {
            "path": config.fallback_path,
            "sha256": digest,
            "matches_config": digest == config.fallback_sha256,
            "identity": config.fallback_identity,
        }
        if digest != config.fallback_sha256:
            problems.append("fallback sha256 mismatch")
    for candidate in config.candidates:
        path = PROJECT_ROOT / candidate.checkpoint
        digest = sha256_of_file(path) if path.is_file() else ""
        identities["candidates"].append({
            "variant": candidate.variant,
            "seed": candidate.seed,
            "checkpoint": candidate.checkpoint,
            "sha256": digest,
            "matches_config": digest == candidate.sha256,
        })
        if digest != candidate.sha256:
            problems.append(
                f"candidate {candidate.variant} s{candidate.seed} sha mismatch"
            )
    write_report_bytes(output_dir, "identities.json", identities)
    if problems:
        print("identity verification refused: " + "; ".join(problems),
              file=sys.stderr)
        return 2

    # --- Training-data admission (V5 gate) ---
    training_root = PROJECT_ROOT / config.training_data_root
    try:
        admission = admit_dataset_for_protocol(
            training_root,
            pooling_mode=config.pooling_mode,
            expected_fingerprint=config.data_fingerprint_v5,
            expected_parent_fingerprint=config.parent_data_fingerprint_v4,
            expected_mask_schema=config.mask_schema,
            expected_mask_resize_version=config.mask_resize_version,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"training-data admission refused: {exc}", file=sys.stderr)
        return 2
    write_report_bytes(output_dir, "training_data_admission.json", {
        "data_root": admission["data_root"],
        "audit_fingerprint": admission["audit_fingerprint"],
        "v4_parent_fingerprint": admission["v4_parent_fingerprint"],
        "verified_files": admission["verified_files"],
        "equals_config_training_fingerprint": (
            admission["audit_fingerprint"] == config.data_fingerprint_v5
        ),
    })

    # --- Engineering representatives (seed 2026, fixed rule) ---
    #
    # Every classifier is built through the same identity-checked factory
    # path the engine's from_material_config uses — the verifier must not
    # assemble its own slightly different pipeline.
    from models.material_classifier import (
        build_material_classifier_for_candidate,
        select_candidate,
    )

    representatives = [c for c in config.candidates
                       if c.seed == config.engineering_seed]
    if sorted(c.variant for c in representatives) != sorted(ENGINEERING_VARIANTS):
        print(
            f"expected the four seed-{config.engineering_seed} variants "
            f"{ENGINEERING_VARIANTS}, got "
            f"{sorted(c.variant for c in representatives)}",
            file=sys.stderr,
        )
        return 2
    classifiers = {}
    for variant in ENGINEERING_VARIANTS:
        candidate = select_candidate(
            config, variant=variant, seed=config.engineering_seed
        )
        classifiers[variant] = build_material_classifier_for_candidate(
            config, candidate, device=args.device
        )

    device = next(iter(classifiers.values())).device
    imgsz = config.cmx_imgsz
    snapshots = {
        variant: {
            "model": snapshot_state(classifier.model),
            "fallback": (
                snapshot_state(classifier.fallback)
                if classifier.fallback is not None else {}
            ),
        }
        for variant, classifier in classifiers.items()
    }

    npz_root = training_root
    paths, _, _ = read_manifest_split(npz_root, "val")
    with (npz_root / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    selected = select_val_npz(manifest_rows, args.max_samples)
    selected_paths = [npz_root / row["npz_path"] for row in selected]

    report: dict = {
        "status": "PASSED",
        "generated_by": "scripts/verify_material_runtime.py",
        "budget": {
            "max_samples_limit": MAX_SAMPLES_LIMIT,
            "requested_samples": args.max_samples,
            "max_frames_limit": MAX_FRAMES_LIMIT,
            "requested_frames": args.max_frames,
            "real_forward_budget_s": REAL_FORWARD_BUDGET_S,
        },
        "engineering": {
            "seed": config.engineering_seed,
            "variants": sorted(classifiers),
            "model_identities": {
                variant: c.model_identity
                for variant, c in classifiers.items()
            },
            "fallback_identity": next(iter(classifiers.values())).fallback_identity,
        },
        "training_data_identity": {
            "root": str(training_root),
            "fingerprint": admission["audit_fingerprint"],
            "v4_parent": admission["v4_parent_fingerprint"],
        },
        "runtime_data_identity": {
            "layer_a_root": str(training_root),
            "layer_a_fingerprint": admission["audit_fingerprint"],
            "layer_b_source": "Rectified_v2 val frames referenced by the V5 "
                              "manifest (left/right pair convention)",
            "note": "the runtime identity is reported alongside the training "
                    "identity, never forced equal to it",
        },
        "sync_status": "unknown",
        "sync_note": "no measurable timestamps in this verifier; sync is "
                     "reported unknown, never 0 ms or assumed synced",
        "honest_state": [
            "engineering integration evidence only",
            "no polarization-benefit, generalization, model-quality or FPS "
            "conclusion may be drawn from this report",
        ],
    }

    # --- Layer A ---
    layer_a = run_layer_a(classifiers, selected_paths, device, imgsz,
                          args.batch)
    report["layer_a"] = layer_a

    # --- Layer B ---
    #
    # The engine is built through the supported material entry point (the
    # same one later evaluation runs use), not assembled by hand here. The
    # other variants reuse the classifiers built above through the same
    # identity-checked factory path, so the pipeline under test is one
    # pipeline, not a verifier-only variant of it.
    engine = DualStageInferenceEngine.from_material_config(
        args.config,
        variant="gray_polar",  # a dual variant, so the real fallback binding is exercised
        seed=config.engineering_seed,
        device=args.device,
    )
    matcher = engine.matcher
    band_margin = engine.band_vertical_margin
    budget = _ForwardBudget(REAL_FORWARD_BUDGET_S)
    layer_b = run_layer_b(
        classifiers, engine, npz_root, manifest_rows, matcher, band_margin,
        args.max_frames, budget,
        full_image_threshold=float(config.stereo.get("full_image_threshold", 0.9)),
    )
    report["layer_b"] = layer_b
    report["engine_entry_point"] = {
        "factory": "DualStageInferenceEngine.from_material_config",
        "selection": dict(getattr(engine, "material_selection", {})),
    }

    # --- Immutability after all forwards ---
    changed = []
    for variant, classifier in classifiers.items():
        after_model = snapshot_state(classifier.model)
        for name, tensor in snapshots[variant]["model"].items():
            if not torch.equal(after_model[name], tensor):
                changed.append(f"{variant}:model:{name}")
        if classifier.fallback is not None:
            after_fallback = snapshot_state(classifier.fallback)
            for name, tensor in snapshots[variant]["fallback"].items():
                if not torch.equal(after_fallback[name], tensor):
                    changed.append(f"{variant}:fallback:{name}")
    report["parameters_and_buffers_unchanged"] = not changed
    if changed:
        report["status"] = "FAILED"
        report["changed_tensors"] = changed[:20]

    if layer_a["status"] != "PASSED":
        report["status"] = "FAILED"
    if layer_b["status"] != "PASSED" or layer_b.get("gt_replay_status") == "MISMATCH":
        report["status"] = "FAILED"

    # --- 4B preparation snapshot (recorded; 4B itself is not started) ---
    report["protocol_4b"] = {
        "definition": {
            "gt_condition": "all annotated objects of a val frame (from the "
                            "label file), whether or not an npz sample exists",
            "predicted_condition": "all Model A detections of the same frame "
                                   "with predicted instance masks",
            "thresholds": "to be fixed before 4B starts (none tuned here)",
            "metrics": "per-class precision/recall/macro-F1 under the fixed "
                       "class order, per condition",
            "fallback_rules": "polar-invalid -> independent gray; E1 pure",
            "performance_boundary": "timings are diagnostics, never FPS claims",
        },
        "admission_fields_for_independent_data": [
            "dataset fingerprint (64-hex, audited)",
            "V4 parent fingerprint",
            "mask schema / resize versions",
            "class order",
            "capture-group isolation",
        ],
        "frame_index_note": "the frame-level GT index is written by this "
                            "verifier from the val manifest and label files",
        "known_gaps": [
            "val covers 4 capture groups only; treat uncertainty as indicative",
            "GT objects without a valid stereo measurement have no npz and "
            "must still enter the 4B GT condition",
        ],
    }

    write_report_bytes(output_dir, "report.json", report)

    digests = {
        path.name: sha256_of_file(path)
        for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    write_report_bytes(output_dir, "digests.json", digests)

    print(f"verification report: {output_dir / 'report.json'}")
    print(f"status: {report['status']}")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
