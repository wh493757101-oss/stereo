"""A complete, legal synthetic evidence set for the closeout entry.

Built entirely under the caller's scratch directory: no real data, no real
weights and no model is loaded. The config passes the production parser, the
GT index uses the project's field structure, the synthetic V5 manifest binds
the GT objects to NPZ samples, and the protocol snapshot carries the same 12
candidate identities as the config. Every evidence sub-directory has its own
manifest, which lists its data files and never itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")
VARIANTS = ("gray", "gray_gray_valid", "gray_valid", "gray_polar")
SEEDS = (2026, 2027, 2028)
FRAMES = (("frame_a", "Group/One/0 NTU", 2), ("frame_b", "Group/Two/0 NTU", 2),
          # A legal frame whose instance list is empty and whose evidence
          # explicitly reports zero predictions.
          ("frame_c", "Group/One/0 NTU", 0))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2,
                                 sort_keys=True) + "\n").encode("utf-8"))
    return path


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False,
                                        sort_keys=True) + "\n"
                             for row in rows).encode("utf-8"))
    return path


def _manifest(directory: Path, names) -> Path:
    """A manifest covering the given data files (never itself)."""
    files = {name: _sha(directory / name) for name in names}
    return _write_json(directory / "artifact_manifest.json",
                       {"files": files, "index_file": "artifact_manifest.json"})


def _checks():
    names = ["gray", "instance_mask", "polar_valid", "signed_q", "abs_q",
             "quality"]
    for variant in VARIANTS:
        names += [f"{variant}_rgb", f"{variant}_mask", f"{variant}_logits"]
        if variant != "gray":
            names.append(f"{variant}_aux")
    return {name: {"status": "PASSED", "exact_equal": True, "max_abs_err": 0.0}
            for name in names}


def _metrics(tag: str, view: str) -> dict:
    return {
        "counts": {"frames": 2, "failed_frames": 0, "evaluated_frames": 2,
                   "gt_instances": 4, "predictions": 4, "matched": 4,
                   "missed_gt": 0, "unmatched_predictions": 0,
                   "unclassified_matched": 0},
        "detection": {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                      "tp": 4, "fp": 0, "fn": 0},
        "end_to_end": {"macro_f1": 1.0, "per_class": {}},
        "conditional_classification": {"accuracy": 1.0, "macro_f1": 1.0,
                                       "samples": 4, "per_class": {}},
        "tag": tag, "view": view,
    }


def build_synthetic_source(root: Path) -> dict:
    """Create the config, GT index and evidence tree; return their paths."""
    root.mkdir(parents=True, exist_ok=True)

    # --- placeholder "weights" (never loaded, only hashed) -----------------
    weights = root / "weights"
    weights.mkdir(exist_ok=True)
    model_a = weights / "model_a.pt"
    model_a.write_bytes(b"synthetic model A placeholder\n")
    fallback = weights / "fallback.pt"
    fallback.write_bytes(b"synthetic fallback placeholder\n")
    candidates = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = weights / f"{variant}_s{seed}.pt"
            path.write_bytes(f"synthetic {variant} {seed}\n".encode("utf-8"))
            candidates.append({"variant": variant, "seed": seed,
                               "checkpoint": str(path),
                               "sha256": _sha(path)})

    # --- synthetic V5 dataset ---------------------------------------------
    v5 = root / "v5"
    v5.mkdir(exist_ok=True)
    rows = []
    for frame_id, group, count in FRAMES:
        for index in range(count):
            rows.append({
                "sample_name": f"{frame_id}_obj{index:03d}", "split": "val",
                "group_name": group, "class_id": str(index % 4),
                "class_name": CLASS_NAMES[index % 4],
                "source_frame": frame_id, "object_index": str(index),
                "npz_path": f"val/{frame_id}_obj{index:03d}.npz",
                "left_path": f"left/{frame_id}.png",
                "label_path": f"labels/{frame_id}.txt",
            })
    manifest_path = v5 / "dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(v5 / "dataset_summary.json", {"splits": {"val": len(rows)}})
    # The audit the binding check reads (synthetic; never a real audit file).
    _write_json(v5 / "dataset_audit.json", {
        "audit_passed": True,
        # The real V5 audit binds the manifest and summary by digest; the
        # synthetic one does the same and (like the real file) records no
        # dataset_fingerprint field.
        "manifest_sha256": _sha(manifest_path),
        "summary_sha256": _sha(v5 / "dataset_summary.json"),
        "v4_parent_fingerprint": "8" * 64,
        "v4_manifest_sha256": "7" * 64,
        "mask_content_sha256": "6" * 64,
        "label_source_sha256": "5" * 64,
        "sample_mapping_sha256": "4" * 64,
        "file_digests": {"val/synthetic.npz": "3" * 64},
        "mask_schema_version": "instance_mask_polygon_crop_v1",
        "mask_resize_version": "area_crop_to_imgsz_then_area_to_levels_v1",
        "crop_pad": 10,
    })
    # The V5 identity is produced by the project's own algorithm over the
    # audit metadata; no fixed string is asserted as a legal fingerprint.
    from core.material_mask_dataset import mask_dataset_fingerprint
    v5_fingerprint = mask_dataset_fingerprint(v5)

    # --- config (parsed by the production loader) -------------------------
    config = {
        "experiment": "material_runtime_v2_mask",
        "model_a": {"path": str(model_a), "sha256": _sha(model_a),
                    "conf_threshold": 0.25, "iou_threshold": 0.45,
                    "imgsz": 640, "data_independence": "unknown"},
        "fallback": {"path": str(fallback), "sha256": _sha(fallback),
                     "identity": "synthetic 4-class gray"},
        "candidates": candidates,
        "engineering_seed": 2026,
        "material_model": {
            "imgsz": 224, "crop_padding": 10,
            "preprocessing_version": "v4_band_224_area_linear_v1",
            "pooling_mode": "instance_mask",
            "mask_schema": "instance_mask_polygon_crop_v1",
            "mask_resize_version": "area_crop_to_imgsz_then_area_to_levels_v1",
            "upstream_commit": "e251d860aebc2f583a6c4919877e6bebe7f1aff3",
        },
        "calibration": {"file": "datasets/bd_image/stereo_calib.npz",
                        "r_convention": "matlab", "baseline": 0.0989,
                        "focal_length": 3643.5,
                        "rectification": {"enabled": False,
                                          "already_rectified_frames": False}},
        "stereo": {"matcher": "sgbm", "mode": "3way", "scale": 0.25,
                   "max_disp": 768, "block_size": 7, "uniqueness_ratio": 10,
                   "speckle_window": 100, "speckle_range": 32,
                   "texture_threshold": 10.0, "lr_check_threshold_px": 2.0,
                   "min_valid_ratio": 0.05, "band_margin": 20,
                   "full_image_threshold": 0.9},
        "polarization": {"formula": "signed_q", "gain": 1.0,
                         "epsilon": 1.0e-6, "min_intensity_sum": 10.0},
        "runtime": {"device": "cpu", "precision": "float32",
                    "classification_batch": 8,
                    "fallback_strategy": "route_polar_invalid_to_independent_gray",
                    "class_names": list(CLASS_NAMES)},
        "data_identity": {
            "training_data_root": str(v5),
            "training_data_fingerprint": v5_fingerprint,
            "training_parent_data_fingerprint": "8" * 64,
            "runtime_data_root": "", "runtime_data_fingerprint": "",
        },
    }
    import yaml

    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False),
                           encoding="utf-8")

    # --- GT index (the independent expected set) --------------------------
    gt_index = [
        {"frame_id": frame_id, "split": "val", "capture_group": group,
         "instances": [{"object_index": index, "class_id": index % 4,
                        "gt_instance_id": f"{frame_id}_obj{index:03d}"}
                       for index in range(count)]}
        for frame_id, group, count in FRAMES
    ]
    gt_path = root / "val_frames_gt.jsonl"
    _write_jsonl(gt_path, gt_index)
    # The GT index is bound through its own evidence manifest, exactly as the
    # real 4A closeout directory binds it.
    _manifest(root, ["val_frames_gt.jsonl"])

    # --- evidence tree -----------------------------------------------------
    source = root / "source"
    quality = source / "quality"
    _write_json(quality / "protocol_snapshot.json", {
        "class_names": list(CLASS_NAMES),
        "model_identities": [
            {"variant": c["variant"], "seed": c["seed"], "sha256": c["sha256"]}
            for c in candidates
        ],
        "training_data_identity": {"fingerprint": v5_fingerprint},
        "material_model": config["material_model"],
    })
    per_candidate = {}
    for candidate in candidates:
        tag = f"{candidate['variant']}_s{candidate['seed']}"
        per_candidate[tag] = {
            "variant": candidate["variant"], "seed": candidate["seed"],
            "layer_a1": {"pure": _metrics(tag, "a1_pure"), "routed": None},
            "layer_a2": {"pure": _metrics(tag, "a2_pure"),
                         "routed": _metrics(tag, "a2_routed")},
            "layer_b": {"pure": _metrics(tag, "b_pure"),
                        "routed": _metrics(tag, "b_routed")},
        }
    _write_json(quality / "summary.json", {
        "status": "COMPLETED",
        "per_candidate": per_candidate,
        "bootstrap": {"end_to_end_macro_f1_e4_vs_e1": {
            f"seed_{seed}": {"point_estimate": -0.003, "ci_low": -0.02,
                             "ci_high": 0.08, "clusters": 4}
            for seed in SEEDS}},
        "paired_changes": [{"pair": "e4_vs_e1", "fixed": 1, "broke": 0}],
        "conclusions": {"polarization_benefit": "NOT_ESTABLISHED"},
    })
    instance_rows = []
    for candidate in candidates:
        for frame_id, _group, count in FRAMES:
            for index in range(count):
                common = {"variant": candidate["variant"],
                          "seed": candidate["seed"], "frame_id": frame_id}
                for layer, condition in (
                    ("A2_gt_controlled", "routed"),
                    ("A2_gt_controlled", "pure"),
                    ("B_model_a_predictions", "routed"),
                    ("B_model_a_predictions", "pure"),
                ):
                    row = dict(common)
                    row.update({
                        "layer": layer, "condition": condition,
                        "gt_instance_id": index,
                        "pred_instance_id": index if layer.startswith("B")
                        else None,
                        "used_fallback": False, "fallback_reason": "",
                        "pred_class": index % 4, "error": None,
                    })
                    instance_rows.append(row)
    _write_jsonl(quality / "instances.jsonl", instance_rows)
    alignment_rows = [
        {"frame_id": frame_id, "object_index": index, "status": "PASSED",
         "npz_path": f"val/{frame_id}_obj{index:03d}.npz", "checks": _checks()}
        for frame_id, _group, count in FRAMES for index in range(count)
    ]
    _write_jsonl(quality / "a2_alignment.jsonl", alignment_rows)
    _write_json(quality / "a2_alignment_summary.json", {"status": "PASSED"})
    _write_jsonl(quality / "frames.jsonl", [
        {"frame_id": frame_id, "group": group, "gt_instances": count,
         "predictions": count, "matched": count, "missed_gt": 0,
         "unmatched_predictions": 0}
        for frame_id, group, count in FRAMES
    ])
    _write_jsonl(quality / "unmatched_gt.jsonl", [])
    _manifest(quality, ["summary.json", "protocol_snapshot.json",
                        "instances.jsonl", "a2_alignment_summary.json",
                        "a2_alignment.jsonl", "unmatched_gt.jsonl",
                        "frames.jsonl"])

    failed = source / "quality_attempt1_alignment_failed"
    failed_rows = [dict(row) for row in alignment_rows]
    failed_rows[0]["status"] = "FAILED"
    _write_jsonl(failed / "a2_alignment.jsonl", failed_rows)
    _write_json(failed / "summary.json", {"status": "COMPLETED"})
    _manifest(failed, ["summary.json", "a2_alignment.jsonl"])

    performance = source / "performance"
    _write_json(performance / "performance_report.json", {
        "status": "PARTIAL",
        "coverage": {"0": {"status": "NOT_COVERED"},
                     "1": {"status": "MEASURED", "available_frames": 2,
                           "measured_unique_frames": 1}},
        "residency": [{"variant": "gray", "peak_allocated_mb": 87.0}],
        "conditions": {"stage_definitions": {
            "crop_preprocess_cpu": {"status": "MEASURED"}}},
        "scenarios": {"1": {"bucket": "1", "variants": {"gray": {
            "status": "COMPLETED", "throughput_fps": 8.0,
            "total": {"p50": 0.12, "p95": 0.13, "n": 100},
            "stages": {"crop_preprocess_cpu": {"p50": 0.002}}}}}},
    })
    _write_json(performance / "raw_timings.json", {
        "raw": {"1": {"gray": {"total": [0.12, 0.121],
                               "stages": {"crop_preprocess_cpu": [0.002]}}}},
        "units": "seconds",
    })
    _manifest(performance, ["performance_report.json", "raw_timings.json"])

    return {"root": root, "source": source, "config": config_path,
            "gt_index": gt_path, "candidates": candidates, "v5": v5,
            "alignment_rows": alignment_rows}


def refresh_manifest(directory: Path, names) -> Path:
    """Re-write a manifest after an intentional fixture mutation."""
    return _manifest(directory, names)


def perturb_alignment_row(source: Path, *, index: int, status: str,
                          drop_checks: bool = False) -> None:
    rows = [json.loads(line) for line in (
        source / "quality" / "a2_alignment.jsonl").read_bytes()
        .decode().splitlines()]
    rows[index]["status"] = status
    if drop_checks:
        rows[index]["checks"] = {}
    _write_jsonl(source / "quality" / "a2_alignment.jsonl", rows)
    _manifest(source / "quality", ["summary.json", "protocol_snapshot.json",
                                   "instances.jsonl", "a2_alignment_summary.json",
                                   "a2_alignment.jsonl", "unmatched_gt.jsonl",
                                   "frames.jsonl"])
