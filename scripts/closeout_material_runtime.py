"""4A closeout: evidence verification, bounded polar parity, GT index, cleanup.

This tool only *checks* and *records*; it never trains, never loads a
classification model, never runs Model A and never writes into a dataset.
Its four jobs:

1. ``verify-evidence`` — re-verify a 4A evidence directory: JSON parse,
   per-file sha256 against the recorded digests (of the bytes actually on
   disk), candidate/Model A/fallback identities recomputed from the real
   files, internal consistency of the report, and which preserved logs
   support which claim.
2. ``polar-parity`` — the bounded GT polarization rebuild for the *same*
   frames the original 4A run used (recovered from its report, never
   re-chosen), comparing ``signed_q``/``abs_q`` (and the valid support)
   against the stored V5 NPZ arrays with a pre-fixed ``atol=1e-6, rtol=0``.
3. ``gt-index`` — the complete val frame-level GT index built from the
   authoritative frame-level split list and the four-class annotations,
   keeping instances without an NPZ, empty GT frames, missing annotations
   and every anomaly.
4. ``cleanup`` — an explicitly verified literal-path removal with log
   preservation. A path read out of a report is never treated as a delete
   instruction.

Nothing here is a model-quality statement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = (
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
)

# Pre-fixed parity tolerances. A difference is a finding to locate — the
# tolerance is never widened after seeing one.
PARITY_ATOL = 1e-6
PARITY_RTOL = 0.0

# The single-class segmentation labels; using them as "GT classes" would
# replace the four-class annotation with a binary target.
SINGLE_CLASS_LABEL_ROOT_MARKER = "underwater_seg_binary_v2"

EVIDENCE_REQUIRED_FILES = (
    "report.json",
    "identities.json",
    "training_data_admission.json",
    "digests.json",
)


# ---------------------------------------------------------------------------
# Byte-level report helpers


def sha256_of(path: str | Path) -> str:
    """sha256 of the file's bytes as they are on disk."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report_bytes(directory: str | Path, name: str, payload: Any) -> Path:
    """Write one JSON report as UTF-8 bytes (digests are taken from the file)."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = Path(directory) / name
    path.write_bytes(text.encode("utf-8"))
    return path


def write_jsonl_bytes(directory: str | Path, name: str, records: Iterable[Any]) -> Path:
    """Write one JSONL report as UTF-8/LF bytes."""
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
    ]
    path = Path(directory) / name
    path.write_bytes(("".join(line + "\n" for line in lines)).encode("utf-8"))
    return path


def prepare_output_dir(output_dir: str | Path) -> Path:
    """Create the evidence directory; an existing directory is refused."""
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(
            f"output directory {path} already exists; closeout evidence is "
            "never overwritten — pass a fresh --output-dir"
        )
    path.mkdir(parents=True)
    return path


def prepare_cleanup_dir(output_dir: str | Path) -> Path:
    """The cleanup mode may append to a fresh closeout directory.

    The directory itself may already exist (the closeout run created it),
    but the cleanup report and the preserved logs must not exist yet: an
    existing cleanup report is never overwritten.
    """
    path = Path(output_dir)
    if not path.exists():
        path.mkdir(parents=True)
    for name in ("cleanup_report.json",):
        if (path / name).exists():
            raise FileExistsError(
                f"refusing cleanup: {path / name} already exists; cleanup "
                "evidence is never overwritten"
            )
    return path


# ---------------------------------------------------------------------------
# Evidence verification


def _load_json(path: Path, problems: list[str]) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        problems.append(f"{path.name} is not readable JSON: {exc}")
        return None


def verify_evidence(
    evidence_dir: str | Path,
    *,
    identity_root: str | Path = PROJECT_ROOT,
    log_dirs: Sequence[str | Path] = (),
) -> dict:
    """Re-verify a 4A evidence directory against the bytes on disk.

    Returns a report with ``status`` (PASS/FAIL), the per-file digest
    comparison, the recomputed identities, the report's internal
    consistency and the log traceability. It never writes into the evidence
    directory.
    """
    evidence_dir = Path(evidence_dir)
    identity_root = Path(identity_root)
    problems: list[str] = []
    payloads: dict[str, Any] = {}
    for name in EVIDENCE_REQUIRED_FILES:
        path = evidence_dir / name
        if not path.is_file():
            problems.append(f"missing evidence file: {name}")
            continue
        payloads[name] = _load_json(path, problems)

    # --- digests: the recorded value must equal the file's actual bytes ---
    digest_mismatches: list[str] = []
    digests = payloads.get("digests.json") or {}
    if isinstance(digests, dict):
        for name, recorded in sorted(digests.items()):
            path = evidence_dir / name
            if not path.is_file():
                digest_mismatches.append(f"{name}: recorded but missing")
                continue
            actual = sha256_of(path)
            if actual != recorded:
                digest_mismatches.append(
                    f"{name}: recorded {recorded[:16]}… actual {actual[:16]}…"
                )
    else:
        problems.append("digests.json is not a mapping")

    # --- identities: recompute from the real files ---
    identity_mismatches: list[str] = []
    identities = payloads.get("identities.json") or {}
    entries: list[tuple[str, dict]] = []
    if isinstance(identities, dict):
        for key in ("model_a", "fallback"):
            if isinstance(identities.get(key), dict):
                entries.append((key, identities[key]))
        for candidate in identities.get("candidates", []) or []:
            entries.append(
                (
                    f"candidate {candidate.get('variant')} s{candidate.get('seed')}",
                    candidate,
                )
            )
    for label, entry in entries:
        raw = str(entry.get("path") or entry.get("checkpoint") or "")
        if not raw:
            identity_mismatches.append(f"{label}: no path recorded")
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = identity_root / path
        if not path.is_file():
            identity_mismatches.append(f"{label}: file missing ({raw})")
            continue
        actual = sha256_of(path)
        if actual != entry.get("sha256"):
            identity_mismatches.append(
                f"{label}: recorded {str(entry.get('sha256'))[:16]}… actual "
                f"{actual[:16]}…"
            )

    # --- report internal consistency ---
    consistency_problems: list[str] = []
    report = payloads.get("report.json") or {}
    if isinstance(report, dict) and report:
        layer_a = report.get("layer_a") or {}
        layer_b = report.get("layer_b") or {}
        frames = layer_b.get("selected_frames") or []
        frame_entries = layer_b.get("frames") or []
        gt_replay = layer_b.get("gt_replay") or []
        if len(frames) != len(frame_entries):
            consistency_problems.append(
                f"layer_b: {len(frames)} selected frames but "
                f"{len(frame_entries)} frame entries"
            )
        if len(frames) != len(gt_replay):
            consistency_problems.append(
                f"layer_b: {len(frames)} selected frames but "
                f"{len(gt_replay)} gt_replay entries"
            )
        unknown = sorted({g.get("frame") for g in gt_replay} - set(frames))
        if unknown:
            consistency_problems.append(
                f"layer_b: gt_replay frames not in the selected list: {unknown}"
            )
        if not layer_a.get("selection_rule"):
            consistency_problems.append("layer_a: no selection rule recorded")
        variants = layer_a.get("variants") or {}
        for name, entry in sorted(variants.items()):
            if not entry.get("logits_within_tolerance"):
                consistency_problems.append(
                    f"layer_a: {name} logits were not within tolerance"
                )
        if report.get("status") == "PASSED":
            if layer_a.get("status") != "PASSED":
                consistency_problems.append("status PASSED but layer_a is not")
            if layer_b.get("status") != "PASSED":
                consistency_problems.append("status PASSED but layer_b is not")
            if report.get("parameters_and_buffers_unchanged") is not True:
                consistency_problems.append(
                    "status PASSED but parameters/buffers changed"
                )
    else:
        consistency_problems.append("report.json missing or empty")

    # --- log traceability ---
    log_entries: list[dict] = []
    searched = [evidence_dir, *(Path(d) for d in log_dirs)]
    for directory in searched:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.log")):
            text = path.read_bytes().decode("utf-8", errors="replace")
            log_entries.append(
                {
                    "path": str(path),
                    "sha256": sha256_of(path),
                    "mentions_passed": "status: PASSED" in text,
                    "mentions_traceback": "Traceback" in text,
                }
            )
    pytest_logs = [entry for entry in log_entries if "pytest" in Path(entry["path"]).name]
    log_notes = []
    if not log_entries:
        log_notes.append(
            "no preserved run logs were found for this evidence directory"
        )
    if not pytest_logs:
        log_notes.append(
            "no preserved pytest log: the test counts in the historical 4A "
            "report cannot be re-derived from artifacts and stay UNVERIFIED"
        )
    if log_entries and not any(entry["mentions_passed"] for entry in log_entries):
        log_notes.append(
            "no preserved log shows 'status: PASSED'; the PASSED verdict rests "
            "on report.json alone"
        )

    if problems or digest_mismatches or identity_mismatches or consistency_problems:
        status = "FAIL"
    else:
        status = "PASS"
    return {
        "status": status,
        "evidence_dir": str(evidence_dir),
        "files": sorted(payloads),
        "problems": problems,
        "digests": {
            "covered": sorted(digests) if isinstance(digests, dict) else [],
            "index_file": "digests.json",
            "index_note": "the digest index does not cover itself (no circular hash)",
            "mismatches": digest_mismatches,
        },
        "identity_mismatches": identity_mismatches,
        "consistency": {"problems": consistency_problems},
        "logs": {"entries": log_entries, "notes": log_notes},
    }


# ---------------------------------------------------------------------------
# Polar content parity


def compare_float_arrays(
    actual: np.ndarray | None,
    expected: np.ndarray | None,
    *,
    atol: float = PARITY_ATOL,
    rtol: float = PARITY_RTOL,
    target_mask: np.ndarray | None = None,
) -> dict:
    """Compare two float arrays without hiding anything.

    No absolute-value trick, no cropping away of the differing region, no
    dtype change and no mean-based summary: the exact-equality verdict, the
    full-pixel maximum error, the error inside the target region and the
    number of pixels beyond the pre-fixed tolerance are all reported.
    """
    if actual is None or expected is None:
        return {"status": "UNAVAILABLE", "reason": "array missing"}
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    result: dict[str, Any] = {
        "shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "finite": bool(np.isfinite(actual).all()),
        "expected_finite": bool(np.isfinite(expected).all()),
    }
    if actual.shape != expected.shape:
        result.update(
            {"status": "FAILED", "exact_equal": False,
             "reason": "shape mismatch", "n_over_tolerance": None,
             "max_abs_err": None, "max_abs_err_in_target": None}
        )
        return result
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = atol + rtol * np.abs(expected.astype(np.float64))
    over = diff > tolerance
    result.update(
        {
            "exact_equal": bool(np.array_equal(actual, expected)),
            "max_abs_err": float(diff.max()) if diff.size else 0.0,
            "n_over_tolerance": int(over.sum()),
            "tolerance": {"atol": atol, "rtol": rtol},
        }
    )
    if target_mask is not None:
        target = np.asarray(target_mask).astype(bool)
        result["max_abs_err_in_target"] = (
            float(diff[target].max()) if target.any() else 0.0
        )
    else:
        result["max_abs_err_in_target"] = result["max_abs_err"]
    failed = (
        not result["finite"]
        or not result["expected_finite"]
        or result["n_over_tolerance"] > 0
    )
    result["status"] = "FAILED" if failed else "PASSED"
    if failed:
        worst = np.unravel_index(int(diff.argmax()), diff.shape)
        result["first_counterexample"] = {
            "index": [int(i) for i in worst],
            "actual": float(actual[worst]),
            "expected": float(expected[worst]),
        }
    return result


def recover_4a_frames(evidence_dir: str | Path) -> list[str]:
    """The frame list of the original 4A run, recovered from its report.

    The list is never re-chosen: if it cannot be recovered, the caller must
    stop instead of substituting frames that happen to pass.
    """
    report_path = Path(evidence_dir) / "report.json"
    if not report_path.is_file():
        raise ValueError(
            f"cannot recover the original 4A frame list: {report_path} is missing"
        )
    report = json.loads(report_path.read_bytes().decode("utf-8"))
    frames = (report.get("layer_b") or {}).get("selected_frames")
    if not frames:
        raise ValueError(
            "cannot recover the original 4A frame list: the report has no "
            "layer_b.selected_frames"
        )
    return [str(frame) for frame in frames]


def compare_instance_arrays(
    *,
    frame_id: str,
    sample_name: str,
    object_index: int,
    rebuilt_signed: np.ndarray | None,
    rebuilt_abs: np.ndarray | None,
    rebuilt_valid: np.ndarray | None,
    npz,
    source_files: dict | None = None,
    crop: dict | None = None,
    mask_source: str = "gt_polygon_annotation",
) -> dict:
    """Per-instance signed_q/abs_q parity against the stored NPZ sample."""
    record: dict[str, Any] = {
        "frame_id": frame_id,
        "sample_name": sample_name,
        "object_index": int(object_index),
        "mask_source": mask_source,
        "source_files": source_files or {},
        "crop": crop or {},
        "npz_path": None,
    }
    if npz is None:
        record.update(
            {
                "status": "NO_NPZ",
                "reason": "no NPZ sample corresponds to this GT instance; the "
                          "comparison cannot be made (the instance is kept, not "
                          "dropped and not passed)",
                "signed_q": None,
                "abs_q": None,
            }
        )
        return record

    record["npz_path"] = str(getattr(npz, "path", "")) or None
    target = np.asarray(npz.instance_mask) > 0
    signed = compare_float_arrays(rebuilt_signed, npz.signed_q, target_mask=target)
    absolute = compare_float_arrays(rebuilt_abs, npz.abs_q, target_mask=target)
    record["signed_q"] = signed
    record["abs_q"] = absolute

    # The invalid-region contract: wherever the measurement is not valid,
    # both the rebuilt and the stored signed_q/abs_q must be exactly zero
    # (an invalid pixel is never reported as a measured value).
    invalid = np.asarray(npz.polar_valid) == 0
    zero_ok = True
    if rebuilt_signed is not None and np.asarray(rebuilt_signed).shape == invalid.shape:
        zero_ok &= bool(np.all(np.asarray(rebuilt_signed)[invalid] == 0))
    if rebuilt_abs is not None and np.asarray(rebuilt_abs).shape == invalid.shape:
        zero_ok &= bool(np.all(np.asarray(rebuilt_abs)[invalid] == 0))
    zero_ok &= bool(np.all(np.asarray(npz.signed_q)[invalid] == 0))
    zero_ok &= bool(np.all(np.asarray(npz.abs_q)[invalid] == 0))
    record["invalid_region_zero_ok"] = bool(zero_ok)

    # valid must stay inside the instance mask (V5 invariant).
    valid_inside_mask = bool(
        not np.any((np.asarray(npz.polar_valid) > 0) & ~target)
    )
    record["valid_inside_instance_mask"] = valid_inside_mask

    # The valid support is part of the contract, not a diagnostic: a
    # matching signed_q/abs_q pair with a different (or missing) valid
    # channel still describes a different measurement. A missing rebuilt
    # valid, a shape mismatch and a value mismatch all fail.
    rebuilt_valid_matches = False
    if rebuilt_valid is not None:
        rebuilt_array = np.asarray(rebuilt_valid)
        stored_array = np.asarray(npz.polar_valid)
        if rebuilt_array.shape == stored_array.shape:
            rebuilt_valid_matches = bool(
                np.array_equal(rebuilt_array.astype(np.uint8),
                               stored_array.astype(np.uint8))
            )
    record["rebuilt_valid_equals_npz_valid"] = rebuilt_valid_matches

    passed = (
        signed["status"] == "PASSED"
        and absolute["status"] == "PASSED"
        and record["invalid_region_zero_ok"]
        and valid_inside_mask
        and rebuilt_valid_matches
    )
    record["status"] = "PASSED" if passed else "FAILED"
    if not passed:
        record["failure_reasons"] = [
            reason
            for reason, ok in (
                ("signed_q_mismatch", signed["status"] == "PASSED"),
                ("abs_q_mismatch", absolute["status"] == "PASSED"),
                ("invalid_region_not_zeroed", record["invalid_region_zero_ok"]),
                ("valid_outside_instance_mask", valid_inside_mask),
                ("valid_support_mismatch", rebuilt_valid_matches),
            )
            if not ok
        ]
    return record


def _right_image_path(left_path: Path) -> Path | None:
    parts = list(left_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "left":
            parts[index] = "right"
            candidate = Path(*parts)
            return candidate if candidate.is_file() else None
    return None


def run_polar_parity(
    frames: Sequence[str],
    *,
    data_root: str | Path,
    runtime_config,
    project_root: str | Path = PROJECT_ROOT,
) -> dict:
    """Bounded GT polarization rebuild for exactly ``frames``.

    Reuses the generation's own helpers and the accepted stereo/polar
    parameters. No classifier, no Model A, no NPZ write, no dataset change.
    """
    import cv2

    from core.fusion_dataset import quality_vector_from_result  # noqa: F401
    from core.material_mask_dataset import load_mask_sample
    from core.polar_compute import compute_polar_features
    from core.stereo_matching import (
        StereoMatcher,
        StereoMatcherConfig,
        build_horizontal_bands,
        to_gray_u8,
    )
    from scripts.make_polar_dataset import make_crop_window, read_yolo_polygons
    from scripts.prepare_cls_fusion_dataset import build_object_mask

    project_root = Path(project_root)
    data_root = Path(data_root)
    manifest_path = data_root / "dataset_manifest.csv"
    with manifest_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    val_rows = [row for row in rows if row["split"] == "val"]
    npz_by_name = {row["sample_name"]: data_root / row["npz_path"] for row in val_rows}

    stereo_cfg = StereoMatcherConfig(
        matcher=str(runtime_config.stereo.get("matcher", "sgbm")),
        mode=str(runtime_config.stereo.get("mode", "3way")),
        max_disparity=int(runtime_config.stereo.get("max_disp", 768)),
        scale=float(runtime_config.stereo.get("scale", 0.25)),
        block_size=int(runtime_config.stereo.get("block_size", 7)),
        uniqueness_ratio=int(runtime_config.stereo.get("uniqueness_ratio", 10)),
        speckle_window=int(runtime_config.stereo.get("speckle_window", 100)),
        speckle_range=int(runtime_config.stereo.get("speckle_range", 32)),
        texture_threshold=float(runtime_config.stereo.get("texture_threshold", 10.0)),
        lr_check_threshold_px=float(
            runtime_config.stereo.get("lr_check_threshold_px", 2.0)
        ),
        min_valid_ratio=float(runtime_config.stereo.get("min_valid_ratio", 0.05)),
    )
    matcher = StereoMatcher(stereo_cfg)
    band_margin = int(runtime_config.stereo.get("band_margin", 20))
    full_image_threshold = float(
        runtime_config.stereo.get("full_image_threshold", 0.9)
    )
    gain = float(runtime_config.polar.get("gain", 1.0))
    min_intensity_sum = float(runtime_config.polar.get("min_intensity_sum", 10.0))
    epsilon = float(runtime_config.polar.get("epsilon", 1e-6))
    pad = int(runtime_config.crop_padding)

    report: dict = {
        "frames_requested": list(frames),
        "tolerances": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        "stereo_parameters": {
            "band_margin": band_margin,
            "full_image_threshold": full_image_threshold,
            "max_disp": stereo_cfg.max_disparity,
            "scale": stereo_cfg.scale,
            "block_size": stereo_cfg.block_size,
        },
        "polar_parameters": {
            "gain": gain,
            "min_intensity_sum": min_intensity_sum,
            "epsilon": epsilon,
        },
        "frames": [],
        "instances": [],
        "status": "PASSED",
    }

    for frame in frames:
        frame_rows = [row for row in val_rows if row["source_frame"] == frame]
        entry: dict = {"frame_id": frame, "objects": len(frame_rows)}
        if not frame_rows:
            entry.update(
                {"status": "UNAVAILABLE",
                 "reason": "no val manifest rows for this frame; the original "
                           "frame list is not recoverable from this dataset"}
            )
            report["frames"].append(entry)
            report["status"] = "FAILED"
            continue
        left_path = project_root / frame_rows[0]["left_path"]
        right_path = _right_image_path(left_path)
        if right_path is None:
            entry.update(
                {"status": "UNAVAILABLE",
                 "reason": "paired right image not found; not guessed"}
            )
            report["frames"].append(entry)
            report["status"] = "FAILED"
            continue
        left_gray = to_gray_u8(cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED))
        right_gray = to_gray_u8(cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED))
        height, width = left_gray.shape
        label_path = project_root / frame_rows[0]["label_path"]
        polygons = read_yolo_polygons(label_path, width, height)

        bboxes = []
        windows = []
        for row in frame_rows:
            polygon = polygons[int(row["object_index"])].polygon
            xs = [x for x, _ in polygon]
            ys = [y for _, y in polygon]
            # The generation contract: band bboxes come from the polygon
            # extrema, not from padded crop windows.
            bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            windows.append(make_crop_window(polygon, width, height, pad))
        bands = build_horizontal_bands(
            bboxes, image_height=height, vertical_margin=band_margin
        )
        stereo = matcher.compute_bands(
            left_gray, right_gray, bands, full_image_threshold=full_image_threshold
        )

        entry.update(
            {
                "left_path": str(left_path),
                "left_sha256": sha256_of(left_path),
                "right_path": str(right_path),
                "right_sha256": sha256_of(right_path),
                "label_path": str(label_path),
                "label_sha256": sha256_of(label_path),
                "image_size": [width, height],
                "already_rectified": True,
                "band_bbox_source": "polygon_extrema",
                "status": "PASSED",
            }
        )
        for row, window in zip(frame_rows, windows):
            object_index = int(row["object_index"])
            polygon = polygons[object_index].polygon
            mask = build_object_mask(polygon, width, height)
            stats = matcher.instance_stats(stereo, mask, object_index)
            if stats.valid:
                polar = compute_polar_features(
                    left_gray, right_gray, stereo.disparity,
                    object_mask=mask, disparity_valid=stereo.valid,
                    right_gain=gain, min_intensity_sum=min_intensity_sum,
                    epsilon=epsilon,
                )
                x1, y1, x2, y2 = window.x1, window.y1, window.x2, window.y2
                rebuilt_signed = polar.signed_q[y1:y2, x1:x2]
                rebuilt_abs = polar.abs_q[y1:y2, x1:x2]
                rebuilt_valid = polar.valid_mask[y1:y2, x1:x2].astype(np.uint8)
            else:
                shape = (window.y2 - window.y1, window.x2 - window.x1)
                rebuilt_signed = np.zeros(shape, dtype=np.float32)
                rebuilt_abs = np.zeros(shape, dtype=np.float32)
                rebuilt_valid = np.zeros(shape, dtype=np.uint8)
            npz_path = npz_by_name.get(row["sample_name"])
            npz = load_mask_sample(npz_path) if npz_path is not None else None
            record = compare_instance_arrays(
                frame_id=frame,
                sample_name=row["sample_name"],
                object_index=object_index,
                rebuilt_signed=rebuilt_signed,
                rebuilt_abs=rebuilt_abs,
                rebuilt_valid=rebuilt_valid,
                npz=npz,
                source_files={
                    "left": {"path": str(left_path), "sha256": entry["left_sha256"]},
                    "right": {"path": str(right_path), "sha256": entry["right_sha256"]},
                    "label": {"path": str(label_path), "sha256": entry["label_sha256"]},
                },
                crop={
                    "x1": window.x1, "y1": window.y1,
                    "x2": window.x2, "y2": window.y2,
                    "padding": pad,
                },
                mask_source="gt_polygon_annotation",
            )
            record["stereo_valid"] = bool(stats.valid)
            record["stereo_reason"] = stats.reason
            report["instances"].append(record)
            if record["status"] not in ("PASSED", "NO_NPZ"):
                entry["status"] = "FAILED"
        report["frames"].append(entry)

    statuses = {record["status"] for record in report["instances"]}
    if "FAILED" in statuses:
        report["status"] = "FAILED"
    report["counts"] = {
        "frames": len(report["frames"]),
        "instances": len(report["instances"]),
        "passed": sum(1 for r in report["instances"] if r["status"] == "PASSED"),
        "failed": sum(1 for r in report["instances"] if r["status"] == "FAILED"),
        "no_npz": sum(1 for r in report["instances"] if r["status"] == "NO_NPZ"),
    }
    return report


# ---------------------------------------------------------------------------
# Complete frame-level GT index


def _read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _refuse_single_class_label_root(label_root: Path) -> None:
    resolved = str(Path(label_root).resolve())
    if SINGLE_CLASS_LABEL_ROOT_MARKER in resolved.replace("\\", "/"):
        raise ValueError(
            f"label_root {resolved} is the single-class (binary) segmentation "
            "label set; the four-class GT annotations live in "
            "datasets/underwater_seg_v2/labels"
        )


def parse_gt_label_file(
    path: Path,
    width: int,
    height: int,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
) -> list[dict]:
    """Parse one YOLO-polygon label file into raw instances (no repair)."""
    instances: list[dict] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_index, line in enumerate(text.splitlines()):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or len(values) % 2 == 0:
            instances.append(
                {
                    "object_index": None,
                    "malformed_line": line_index,
                    "raw": line,
                }
            )
            continue
        class_id = int(values[0])
        points = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
        pixels = points * np.asarray([width, height], dtype=np.float64)
        xs = pixels[:, 0]
        ys = pixels[:, 1]
        instances.append(
            {
                "object_index": None,  # assigned by the index builder
                "class_id": class_id,
                "class_name": (
                    class_names[class_id]
                    if 0 <= class_id < len(class_names)
                    else None
                ),
                "polygon_normalized": points.tolist(),
                "polygon": pixels.tolist(),
                "bbox": [float(xs.min()), float(ys.min()),
                         float(xs.max()), float(ys.max())],
                "coordinate_convention": (
                    "yolo_normalized_xy_scaled_by_(width,height); "
                    "pixels in the rectified left image frame"
                ),
                "annotation_line": line_index,
            }
        )
    return instances


def build_val_gt_index(
    *,
    split_csv: str | Path,
    label_root: str | Path,
    v5_manifest: str | Path,
    data_root: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    split: str = "val",
    image_size_reader: Callable[[Path], tuple[int, int]] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> tuple[list[dict], dict]:
    """The complete frame-level GT index for one split.

    Built from the authoritative frame-level split list and the four-class
    annotations — never reverse-engineered from the V5 NPZ list, and never
    from the single-class segmentation labels.
    """
    label_root = Path(label_root)
    _refuse_single_class_label_root(label_root)
    split_csv = Path(split_csv)
    project_root = Path(project_root)
    data_root = Path(data_root)

    def read_size(path: Path) -> tuple[int, int]:
        if image_size_reader is not None:
            return image_size_reader(path)
        import cv2

        from core.stereo_matching import to_gray_u8

        image = to_gray_u8(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
        return int(image.shape[1]), int(image.shape[0])

    frames_rows = [
        row for row in _read_csv_rows(split_csv) if row.get("split") == split
    ]
    manifest_rows = [
        row for row in _read_csv_rows(Path(v5_manifest)) if row.get("split") == split
    ]
    npz_by_key: dict[tuple[str, int], dict] = {}
    duplicate_samples: list[str] = []
    for row in manifest_rows:
        key = (row["source_frame"], int(row["object_index"]))
        if key in npz_by_key:
            duplicate_samples.append(row["sample_name"])
        npz_by_key[key] = row

    index: list[dict] = []
    used_npz: set[str] = set()
    summary: dict[str, Any] = {
        "split": split,
        "authoritative_source": str(split_csv),
        "label_root": str(label_root),
        "class_names": list(class_names),
        "authoritative_frames": len(frames_rows),
        "indexed_frames": 0,
        "gt_instances": 0,
        "gt_instances_per_class": {name: 0 for name in class_names},
        "frames_per_group": {},
        "npz_samples": len(manifest_rows),
        "gt_without_npz": 0,
        "npz_without_gt_source": 0,
        "frames_missing_annotation": 0,
        "empty_gt_frames": 0,
        "duplicate_instances": 0,
        "duplicate_npz_samples": sorted(duplicate_samples),
        "out_of_range_polygons": 0,
        "unknown_class_instances": 0,
        "malformed_lines": 0,
        "missing_images": 0,
        "anomalies_total": 0,
        "stereo_validity": "not_evaluated",
        "sync_metadata": "unknown",
    }

    for row in frames_rows:
        stem = row["output_stem"]
        group = row["group_name"]
        summary["frames_per_group"][group] = summary["frames_per_group"].get(group, 0) + 1
        left = project_root / row["left_path"]
        right = project_root / row["right_path"]
        label_path = label_root / split / f"{stem}.txt"
        anomalies: list[str] = []
        integrity: dict[str, Any] = {}

        for name, path in (("left", left), ("right", right)):
            if not path.is_file():
                anomalies.append(f"missing_{name}_image")
                summary["missing_images"] += 1
            else:
                integrity[f"{name}_sha256"] = sha256_of(path)
                try:
                    width, height = read_size(path)
                    integrity[f"{name}_size"] = [width, height]
                except Exception as exc:  # unreadable image
                    anomalies.append(f"unreadable_{name}_image: {exc}")

        if not label_path.is_file():
            annotation_state = "missing"
            summary["frames_missing_annotation"] += 1
            instances: list[dict] = []
            integrity["label_sha256"] = None
            anomalies.append("missing_annotation")
        else:
            integrity["label_sha256"] = sha256_of(label_path)
            raw_instances = parse_gt_label_file(
                label_path,
                integrity.get("left_size", [0, 0])[0],
                integrity.get("left_size", [0, 0])[1],
                class_names=class_names,
            )
            annotation_state = "empty" if not raw_instances else "present"
            if annotation_state == "empty":
                summary["empty_gt_frames"] += 1
            instances = []
            seen_polygons: set[tuple] = set()
            for position, instance in enumerate(raw_instances):
                if instance.get("malformed_line") is not None:
                    anomalies.append(f"malformed_label_line:{instance['malformed_line']}")
                    summary["malformed_lines"] += 1
                    continue
                instance = dict(instance)
                instance["object_index"] = position
                instance["gt_instance_id"] = f"{stem}_obj{position:03d}"
                normalized = np.asarray(instance["polygon_normalized"])
                if bool((normalized < 0).any()) or bool((normalized > 1).any()):
                    anomalies.append(f"out_of_range_polygon:{position}")
                    summary["out_of_range_polygons"] += 1
                if instance["class_name"] is None:
                    anomalies.append(f"unknown_class_id:{instance['class_id']}")
                    summary["unknown_class_instances"] += 1
                key = (instance["class_id"], tuple(np.round(normalized, 9).ravel().tolist()))
                if key in seen_polygons:
                    anomalies.append(f"duplicate_instance:{position}")
                    summary["duplicate_instances"] += 1
                seen_polygons.add(key)
                manifest_row = npz_by_key.get((stem, position))
                instance["v5_sample_name"] = (
                    manifest_row["sample_name"] if manifest_row else None
                )
                instance["has_npz"] = manifest_row is not None
                instance["npz_path"] = (
                    manifest_row["npz_path"] if manifest_row else None
                )
                instance["npz_absence_reason"] = (
                    None
                    if manifest_row
                    else "no V5 sample maps to this GT instance "
                         "(object_index/frame pair absent from the V5 manifest)"
                )
                if manifest_row:
                    used_npz.add(manifest_row["sample_name"])
                else:
                    summary["gt_without_npz"] += 1
                instance["stereo_validity"] = "not_evaluated"
                instance["anomalies"] = []
                instances.append(instance)
                summary["gt_instances"] += 1
                name = instance["class_name"]
                if name in summary["gt_instances_per_class"]:
                    summary["gt_instances_per_class"][name] += 1

        index.append(
            {
                "split": split,
                "capture_group": group,
                "frame_id": stem,
                "left_path": row["left_path"],
                "right_path": row["right_path"],
                "image_size": integrity.get("left_size"),
                "already_rectified": True,
                "coordinate_system": "rectified left image (datasets/Rectified_v2)",
                "label_path": str(label_path.relative_to(project_root))
                if label_path.is_relative_to(project_root) else str(label_path),
                "label_sha256": integrity.get("label_sha256"),
                "annotation_state": annotation_state,
                "gt_instance_count": len(instances) if annotation_state != "missing" else None,
                "instances": instances,
                "integrity": integrity,
                "sync_metadata": {"status": "unknown",
                                  "note": "no measurable timestamps in the "
                                          "frame-level sources"},
                "anomalies": anomalies,
            }
        )
        summary["anomalies_total"] += len(anomalies)

    summary["indexed_frames"] = len(index)
    summary["npz_without_gt_source"] = len(
        [row for row in manifest_rows if row["sample_name"] not in used_npz]
    )
    summary["gt_without_npz"] = int(summary["gt_without_npz"])
    # "Complete" means the frame coverage was actually demonstrated: every
    # authoritative frame is indexed, every frame's images and annotation
    # were readable, and no label line was malformed. Recorded anomalies
    # (duplicates, out-of-range polygons, unknown classes, GT without an
    # NPZ) do not make the index incomplete — they are findings, reported
    # rather than repaired.
    summary["complete"] = bool(
        summary["indexed_frames"] == summary["authoritative_frames"]
        and summary["missing_images"] == 0
        and summary["frames_missing_annotation"] == 0
        and summary["malformed_lines"] == 0
    )
    summary["completeness_note"] = (
        "frame coverage is demonstrated against the authoritative frame-level "
        "split list; anomalies are recorded, never repaired or filtered"
    )
    return index, summary


# ---------------------------------------------------------------------------
# Cleanup (explicit literal path only)


def _reparse_points(root: Path) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames) + list(filenames):
            path = Path(dirpath) / name
            try:
                info = os.lstat(path)
            except OSError:
                continue
            attributes = getattr(info, "st_file_attributes", 0)
            if stat.S_ISLNK(info.st_mode) or (
                attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                found.append(str(path))
    return found


def verify_cleanup_target(
    target: str | Path,
    *,
    allowed_parent: str | Path,
    expected_names: set[str] | None = None,
) -> dict:
    """Prove the target is exactly the expected, disposable directory.

    The path must resolve strictly inside ``allowed_parent``, must be a real
    directory, must contain no symlink/junction, and — when
    ``expected_names`` is given — may only contain those names plus run
    logs. Anything else stops the cleanup.
    """
    target = Path(target)
    allowed_parent = Path(allowed_parent)
    if not allowed_parent.is_dir():
        raise ValueError(f"allowed parent does not exist: {allowed_parent}")
    allowed_resolved = allowed_parent.resolve()
    target_resolved = target.resolve()
    if target_resolved == allowed_resolved or allowed_resolved not in target_resolved.parents:
        raise ValueError(
            f"refusing {target_resolved}: it is outside the allowed parent "
            f"{allowed_resolved} (or is the parent itself)"
        )
    if not target_resolved.is_dir():
        raise ValueError(f"refusing {target_resolved}: not an existing directory")
    links = _reparse_points(target_resolved)
    if links:
        raise ValueError(
            f"refusing {target_resolved}: it contains symlink/junction entries "
            f"{links[:5]}; resolve them manually"
        )
    top_level = sorted(entry.name for entry in target_resolved.iterdir())
    if expected_names is not None:
        unexpected = [
            name
            for name in top_level
            if name not in expected_names and not name.endswith(".log")
        ]
        if unexpected:
            raise ValueError(
                f"refusing {target_resolved}: unexpected entries {unexpected}; "
                "only the known disposable layout plus logs may be removed"
            )
    return {
        "target": str(target_resolved),
        "allowed_parent": str(allowed_resolved),
        "top_level": top_level,
        "reparse_points": links,
        "verified": True,
    }


def preserve_logs(target: str | Path, destination: str | Path) -> dict:
    """Copy the target's run logs out, verifying the copied bytes."""
    target = Path(target)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted(target.glob("*.log")):
        source_bytes = path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        copied = destination / path.name
        copied.write_bytes(source_bytes)
        entries.append(
            {
                "name": path.name,
                "source": str(path),
                "destination": str(copied),
                "source_sha256": source_sha,
                "destination_sha256": sha256_of(copied),
                "sha256_match": sha256_of(copied) == source_sha,
                "mentions_passed": b"status: PASSED" in source_bytes,
                "mentions_traceback": b"Traceback" in source_bytes,
            }
        )
    return {
        "files": len(entries),
        "entries": entries,
        "all_digests_match": all(entry["sha256_match"] for entry in entries),
    }


def remove_cleanup_target(target: str | Path, *, verified: dict | None) -> dict:
    """Remove the verified target only; requires a prior verification."""
    if not verified or not verified.get("verified"):
        raise ValueError(
            "refusing to remove anything without a verified target: run "
            "verify_cleanup_target first"
        )
    target = Path(target).resolve()
    if str(target) != str(Path(verified["target"])):
        raise ValueError(
            f"refusing {target}: it does not match the verified target "
            f"{verified['target']}"
        )
    links = _reparse_points(target)
    if links:
        raise ValueError(f"refusing {target}: reparse points appeared ({links[:5]})")
    shutil.rmtree(target)
    return {"removed": True, "target": str(target), "existed_after": target.exists()}


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="all",
                        choices=["all", "verify-evidence", "polar-parity",
                                 "gt-index", "cleanup"],
                        help="'all' runs verification, polar parity and the GT "
                             "index into one fresh evidence directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence-dir", default=None,
                        help="the original 4A evidence directory (read-only)")
    parser.add_argument("--log-dir", action="append", default=[],
                        help="extra directory to search for preserved logs")
    parser.add_argument("--config",
                        default="configs/experiments/material_runtime_v2_mask.yaml")
    parser.add_argument("--data-root",
                        default="datasets/underwater_cls_fusion_v5_instance_mask")
    parser.add_argument("--split-csv",
                        default="datasets/underwater_seg_binary_v2/split_assignment.csv")
    parser.add_argument("--label-root", default="datasets/underwater_seg_v2/labels")
    parser.add_argument("--cleanup-target", default=None)
    parser.add_argument("--cleanup-parent", default=None)
    parser.add_argument("--cleanup-expect", action="append", default=[])
    return parser.parse_args(argv)


def _run_verify_evidence(args, output_dir: Path) -> dict:
    if not args.evidence_dir:
        raise ValueError("--evidence-dir is required for verify-evidence")
    result = verify_evidence(
        args.evidence_dir, identity_root=PROJECT_ROOT, log_dirs=args.log_dir
    )
    write_report_bytes(output_dir, "evidence_verification.json", result)
    return result


def _run_polar_parity(args, output_dir: Path) -> dict:
    if not args.evidence_dir:
        raise ValueError("--evidence-dir is required for polar-parity")
    from models.material_classifier import load_material_runtime_config

    config = load_material_runtime_config(args.config)
    frames = recover_4a_frames(args.evidence_dir)
    result = run_polar_parity(
        frames, data_root=PROJECT_ROOT / args.data_root, runtime_config=config
    )
    write_report_bytes(output_dir, "polar_content_parity.json", result)
    return result


def _run_gt_index(args, output_dir: Path) -> tuple[list[dict], dict]:
    index, summary = build_val_gt_index(
        split_csv=PROJECT_ROOT / args.split_csv,
        label_root=PROJECT_ROOT / args.label_root,
        v5_manifest=PROJECT_ROOT / args.data_root / "dataset_manifest.csv",
        data_root=PROJECT_ROOT / args.data_root,
        project_root=PROJECT_ROOT,
    )
    write_jsonl_bytes(output_dir, "val_frames_gt.jsonl", index)
    write_report_bytes(output_dir, "gt_index_summary.json", summary)
    return index, summary


def _write_closeout_summary(
    output_dir: Path,
    *,
    args,
    evidence_result: dict | None,
    parity_result: dict | None,
    gt_summary: dict | None,
    extra: dict | None = None,
) -> dict:
    """The closeout report plus the artifact manifest of this directory."""
    sections = {
        "historical_evidence": (
            "PASS" if evidence_result and evidence_result["status"] == "PASS"
            else "FAIL" if evidence_result else "UNVERIFIED"
        ),
        "polar_content_parity": (
            "PASS" if parity_result and parity_result["status"] == "PASSED"
            else "FAIL" if parity_result else "UNVERIFIED"
        ),
        "gt_index": (
            "COMPLETE" if gt_summary and gt_summary.get("complete")
            else "INCOMPLETE" if gt_summary else "UNVERIFIED"
        ),
    }
    report = {
        "generated_by": "scripts/closeout_material_runtime.py",
        "mode": args.mode,
        "original_4a_evidence_dir": args.evidence_dir,
        "sections": sections,
        "evidence_verification": (
            {
                "status": evidence_result["status"],
                "digest_mismatches": evidence_result["digests"]["mismatches"],
                "identity_mismatches": evidence_result["identity_mismatches"],
                "consistency_problems": evidence_result["consistency"]["problems"],
                "log_notes": evidence_result["logs"]["notes"],
                "logs": evidence_result["logs"]["entries"],
            }
            if evidence_result else None
        ),
        "polar_content_parity": (
            {
                "status": parity_result["status"],
                "counts": parity_result["counts"],
                "frames": parity_result["frames_requested"],
                "tolerances": parity_result["tolerances"],
            }
            if parity_result else None
        ),
        "gt_index": gt_summary,
        "notes": [
            "this round re-verified historical artifacts; it did not re-run the "
            "4A test suites, the bounded smoke or any model forward",
            "no training, no 4B evaluation, no deployment",
            "polar parity is a data-generation consistency check, not a "
            "polarization-benefit result",
        ],
        "honest_state": [
            "real-frame natural fallback stayed 0 in the original 4A run: that "
            "is an observation, NOT a validation of the fallback path",
            "no polarization benefit, independent generalization, real-time "
            "performance or deployment readiness is claimed",
        ],
    }
    if extra:
        report.update(extra)
    write_report_bytes(output_dir, "closeout_report.json", report)
    (output_dir / "closeout_report.md").write_bytes(
        _render_closeout_markdown(report).encode("utf-8")
    )
    manifest = {
        "files": {
            path.name: sha256_of(path)
            for path in sorted(output_dir.iterdir()) if path.is_file()
        },
        "note": "artifact_manifest.json is written last and does not cover itself "
                "(no circular hash)",
    }
    write_report_bytes(output_dir, "artifact_manifest.json", manifest)
    return report


def _render_closeout_markdown(report: dict) -> str:
    lines = [
        "# 4A closeout",
        "",
        f"- mode: `{report['mode']}`",
        f"- original 4A evidence: `{report['original_4a_evidence_dir']}`",
        "",
        "## Section status",
        "",
        "| section | status |",
        "|---|---|",
    ]
    for name, status in report["sections"].items():
        lines.append(f"| {name} | {status} |")
    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in report["notes"]]
    lines += ["", "## Honest state", ""]
    lines += [f"- {note}" for note in report["honest_state"]]
    lines += [""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (
        prepare_cleanup_dir(args.output_dir)
        if args.mode == "cleanup"
        else prepare_output_dir(args.output_dir)
    )

    if args.mode == "verify-evidence":
        result = _run_verify_evidence(args, output_dir)
        print(f"evidence verification: {result['status']}")
        return 0 if result["status"] == "PASS" else 1

    if args.mode == "polar-parity":
        result = _run_polar_parity(args, output_dir)
        print(
            f"polar parity: {result['status']} "
            f"({result['counts']['passed']} passed, "
            f"{result['counts']['failed']} failed, "
            f"{result['counts']['no_npz']} without npz)"
        )
        return 0 if result["status"] == "PASSED" else 1

    if args.mode == "gt-index":
        _index, summary = _run_gt_index(args, output_dir)
        print(
            f"gt index: {summary['indexed_frames']}/{summary['authoritative_frames']} "
            f"frames, {summary['gt_instances']} instances"
        )
        return 0 if summary["complete"] else 1

    if args.mode == "all":
        evidence_result = _run_verify_evidence(args, output_dir)
        parity_result = _run_polar_parity(args, output_dir)
        _index, gt_summary = _run_gt_index(args, output_dir)
        report = _write_closeout_summary(
            output_dir, args=args, evidence_result=evidence_result,
            parity_result=parity_result, gt_summary=gt_summary,
        )
        print(f"sections: {report['sections']}")
        return 0 if all(
            status in ("PASS", "COMPLETE") for status in report["sections"].values()
        ) else 1

    if args.mode == "cleanup":
        if not args.cleanup_target or not args.cleanup_parent:
            print("--cleanup-target and --cleanup-parent are required",
                  file=sys.stderr)
            return 2
        preserved = preserve_logs(args.cleanup_target, output_dir / "preserved_logs")
        verified = verify_cleanup_target(
            args.cleanup_target, allowed_parent=args.cleanup_parent,
            expected_names=set(args.cleanup_expect) or None,
        )
        if not preserved["all_digests_match"]:
            print("refusing cleanup: preserved log digests do not match",
                  file=sys.stderr)
            return 2
        result = remove_cleanup_target(args.cleanup_target, verified=verified)
        write_report_bytes(
            output_dir, "cleanup_report.json",
            {"preserved_logs": preserved, "verification": verified,
             "removal": result},
        )
        print(f"cleanup: removed={result['removed']} target={result['target']}")
        return 0

    print(f"unknown mode {args.mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
