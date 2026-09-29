"""4B offline-replay performance benchmark for the material_fusion runtime.

Fixed representatives: E1 s2026 and E4 s2026 (an integration rule, not a
"whichever was fastest" choice). Scenario buckets come from the **actual**
Model A predictions over the val split (0, 1, 2, 5 targets and the
highest-load frame); a bucket with no natural frame is reported NOT_COVERED.

Measurement bookkeeping this round fixes:

- the selected frame per bucket is written to ``scenario_selection.json``
  **before** any timing starts, so the choice cannot follow the numbers;
- ``available_frames`` (candidates in the bucket) and
  ``measured_unique_frames`` (frames actually timed) are separate fields —
  repeating one frame 100 times is not 100 independent scenes;
- device stages are synchronized **before** the start timestamp and after
  the stage, so the measurement covers the stage's device work;
- the whole-frame latency is measured independently of the per-stage
  diagnostics and the stage sum is never reported as the pipeline latency;
- stage names say what was measured: ``classification_wrapper`` is the
  ``predict_batch`` call including its own preprocessing, host→device
  transfer, forward and result assembly — it is **not** a pure CMX forward
  time, and crop-only / H2D-only / pure-forward-only are NOT_MEASURED;
- E1 and E4 are resident and measured **sequentially** (the first engine is
  released before the second is built), with post-load VRAM, peak allocated
  and peak reserved recorded per engine;
- the raw repeat arrays are saved, and every summary value is derived from
  them;
- fallback measurements are labelled classifier-level and report the actual
  CMX / fallback routing counts; a single-instance frame cannot form a
  mixed batch (NOT_APPLICABLE);
- a shortfall, an incomplete repeat set or a failure never summarizes as
  COMPLETED.

Scope: offline replay of preloaded, already-rectified frames — no camera,
no GUI, no queue (NOT_MEASURED). Not a camera-to-GUI end-to-end FPS claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPRESENTATIVES = (("gray", 2026), ("gray_polar", 2026))
SCENARIO_TARGETS = (0, 1, 2, 5)


# ---------------------------------------------------------------------------
# Small report helpers


def prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.exists():
        raise FileExistsError(
            f"output directory {path} already exists; performance reports "
            "are never overwritten — pass a fresh --output-dir"
        )
    path.mkdir(parents=True)
    return path


def write_json_bytes(directory: str | Path, name: str, payload: Any) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = Path(directory) / name
    path.write_bytes(text.encode("utf-8"))
    return path


def write_jsonl_bytes(directory: str | Path, name: str,
                      records: Sequence[Any]) -> Path:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True)
             for record in records]
    path = Path(directory) / name
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))
    return path


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile_summary(samples: Sequence[float]) -> dict:
    """Nearest-rank percentiles (deterministic, no interpolation)."""
    values = sorted(float(s) for s in samples)
    if not values:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0,
                "mean": 0.0}

    def nearest_rank(q: float) -> float:
        return values[max(0, int(np.ceil(q * len(values))) - 1)]

    return {
        "n": len(values),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "min": values[0],
        "max": values[-1],
        "mean": float(np.mean(values)),
    }


# ---------------------------------------------------------------------------
# Timing helpers


def timed_stage(
    stage: Callable[[], Any],
    perf_counter: Callable[[], float],
    cuda_synchronize: Callable[[], None],
    device: str,
) -> float:
    """Time one stage, synchronizing before and after on CUDA devices.

    Synchronizing before the start timestamp makes sure no earlier work is
    still in flight when the clock starts; synchronizing after the stage
    makes sure its device work has completed before the clock stops. Both
    are part of the *diagnostic* measurement (and one reason the stage sum
    is never the end-to-end latency).
    """
    is_cuda = str(device) not in ("cpu", "") and (
        "cuda" in str(device) or str(device).isdigit()
    )
    if is_cuda:
        cuda_synchronize()
    started = perf_counter()
    stage()
    if is_cuda:
        cuda_synchronize()
    return perf_counter() - started


def run_repeats(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    max_seconds: float | None = None,
) -> Any:
    """Warmup runs then measured runs; the warmup is never in the samples."""
    for _ in range(warmup):
        fn()
    samples: list = []
    if max_seconds is None:
        for _ in range(repeats):
            samples.append(fn())
        return samples
    deadline = time.perf_counter() + float(max_seconds)
    completed = 0
    for _ in range(repeats):
        if time.perf_counter() >= deadline:
            break
        samples.append(fn())
        completed += 1
        if time.perf_counter() >= deadline:
            break
    return {
        "completed": completed,
        "requested": repeats,
        "status": "COMPLETED" if completed >= repeats else "PARTIAL",
        "samples": samples,
    }


def summarise_repeats(stage_samples: dict[str, Sequence[float]],
                      total_samples: Sequence[float]) -> dict:
    """Stage diagnostics and the end-to-end total, side by side."""
    stage_p50 = {
        name: percentile_summary(values)
        for name, values in stage_samples.items()
    }
    total = percentile_summary(total_samples)
    return {
        "total": total,
        "stages": stage_p50,
        "stage_sum_p50": float(sum(entry["p50"] for entry in stage_p50.values())),
        "stage_sum_note": (
            "diagnostic only: per-stage synchronization and stage overlap "
            "make the sum differ from the end-to-end latency; it is never "
            "reported as the pipeline latency"
        ),
    }


def stage_definitions() -> dict:
    """What each measured stage name actually covers."""
    return {
        "model_a": {
            "status": "MEASURED",
            "includes": "Ultralytics Model A predict on the 3-channel gray copy",
        },
        "band_sgbm": {
            "status": "MEASURED",
            "includes": "band building + one band-restricted SGBM pass",
        },
        "depth_polar": {
            "status": "MEASURED",
            "includes": "per-instance disparity stats + frame-level polar "
                        "differential",
        },
        "cpu_tensor_preparation": {
            "status": "MEASURED",
            "includes": "CPU-side resize/normalization of the ALREADY CROPPED "
                        "instance windows into input tensors",
            "excludes": "cropping, host-to-device transfer, model forward",
            "renamed_from": "crop_preprocess_cpu",
            "rename_note": "the old name implied the crop was inside the "
                           "timed region; it is not",
        },
        "classification_wrapper": {
            "status": "MEASURED",
            "includes": "MaterialFusionClassifier.predict_batch: its own "
                        "preprocessing, host->device transfer, forward and "
                        "result assembly",
            "excludes": "pure CMX forward (not isolated), host->device "
                        "transfer alone, crop alone",
            "warning": "this is a wrapper total, not a pure forward time",
        },
        "classification_all_fallback": {
            "status": "MEASURED",
            "includes": "predict_batch with every instance forced polar-invalid "
                        "(classifier-level routing to the independent gray)",
        },
        "classification_mixed_fallback": {
            "status": "MEASURED",
            "includes": "predict_batch with alternating valid/invalid instances "
                        "(classifier-level routing); NOT_APPLICABLE when the "
                        "frame has fewer than two instances",
        },
        "crop_only": {"status": "NOT_MEASURED"},
        "host_to_device_only": {"status": "NOT_MEASURED"},
        "pure_forward_only": {"status": "NOT_MEASURED"},
    }


def model_a_predict_kwargs(imgsz: int | None) -> dict:
    """The configured Model A imgsz, forwarded exactly as the engine does."""
    return {"imgsz": imgsz} if imgsz is not None else {}


def persist_scenario_selection(directory: str | Path,
                               selection: dict) -> Path:
    """Write the chosen frame per bucket before any timing starts."""
    return write_json_bytes(directory, "scenario_selection.json", {
        "selection": selection,
        "written_before_measurement": True,
        "rule": "first frame in sorted frame-id order per predicted-count "
                "bucket; the choice is never revised after seeing timings",
    })


def bucket_coverage(buckets: dict, measured: dict) -> dict:
    """Available candidates vs frames actually timed, per bucket."""
    coverage = {}
    for bucket, frame_ids in buckets.items():
        if not frame_ids:
            coverage[bucket] = {
                "status": "NOT_COVERED",
                "available_frames": 0,
                "measured_unique_frames": 0,
                "reason": "no natural val frame with this predicted instance "
                          "count; nothing was synthesized",
            }
            continue
        timed = measured.get(bucket) or []
        coverage[bucket] = {
            "status": "MEASURED" if timed else "PARTIAL",
            "available_frames": len(frame_ids),
            "measured_unique_frames": len(set(timed)),
            "measured_frames": sorted(set(timed)),
            "note": "repeats are not independent scenes",
        }
    return coverage


def save_raw_timings(directory: str | Path, raw: dict) -> Path:
    """The raw repeat arrays; every summary value derives from these."""
    return write_json_bytes(directory, "raw_timings.json", {
        "raw": raw,
        "units": "seconds",
        "note": "per-repeat arrays; p50/p95/mean/throughput are computed from "
                "these and never re-measured",
    })


def overall_status(statuses: Sequence[dict]) -> str:
    """COMPLETED only when every item completed; otherwise PARTIAL."""
    states = [entry.get("status") for entry in statuses]
    if states and all(state == "COMPLETED" for state in states):
        return "COMPLETED"
    return "PARTIAL"


def mixed_fallback_status(instance_count: int) -> str:
    """A single instance cannot form a mixed batch."""
    return "MEASURED" if int(instance_count) >= 2 else "NOT_APPLICABLE"


def residency_record(
    *,
    variant: str,
    seed: int,
    components: Sequence[str],
    loaded_vram_mb: float | None,
    peak_allocated_mb: float | None,
    peak_reserved_mb: float | None,
    released: bool,
) -> dict:
    """One engine's residency, recorded when it is released for the next."""
    return {
        "variant": variant,
        "seed": seed,
        "components": list(components),
        "loaded_vram_mb": loaded_vram_mb,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "released_before_next_engine": bool(released),
        "note": "engines are resident and measured sequentially, never in "
                "parallel",
    }


def performance_scope(scope: str) -> dict:
    """What the measurement covers and what it explicitly does not."""
    return {
        "scope": scope,
        "includes": [
            "in-memory preloaded, already-rectified stereo frames",
            "Model A, band SGBM, depth, polarization, classification and "
            "the independent gray fallback routing",
        ],
        "excludes": [
            "disk image decoding (frames are preloaded before timing)",
            "camera",
            "GUI",
            "inter-stage queueing",
        ],
        "queue_latency": "NOT_MEASURED",
        "drop_rate": "NOT_MEASURED",
        "note": "offline replay; not a camera-to-GUI end-to-end FPS claim",
    }


# ---------------------------------------------------------------------------
# Main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--scope", default="offline-replay",
                        choices=["offline-replay"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _vram_mb() -> dict:
    if not torch.cuda.is_available():
        return {"allocated": None, "reserved": None}
    return {
        "allocated": float(torch.cuda.memory_allocated()) / (1024 * 1024),
        "reserved": float(torch.cuda.memory_reserved()) / (1024 * 1024),
    }


def _survey_frames(engine, split_rows, left_rights, imgsz):
    """One Model A pass per val frame with the configured imgsz."""
    import cv2

    frames = []
    kwargs = model_a_predict_kwargs(imgsz)
    for row in split_rows:
        left_gray, _ = left_rights[row["output_stem"]]
        bgr_left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
        started = time.perf_counter()
        instances = engine.model_a.predict(bgr_left, **kwargs)
        frames.append({
            "frame_id": row["output_stem"],
            "group": row["group_name"],
            "predicted_instances": len(instances),
            "model_a_seconds": time.perf_counter() - started,
        })
    return frames


def bucket_by_predictions(frames: Sequence[dict],
                          wanted: Sequence[int]) -> dict:
    """Bucket frame ids by their *actual* predicted instance counts."""
    buckets: dict[str, Any] = {}
    max_count = max((int(f["predicted_instances"]) for f in frames), default=-1)
    for target in wanted:
        buckets[str(target)] = [
            f["frame_id"] for f in frames
            if int(f["predicted_instances"]) == target
        ] or None
    buckets["max_available"] = [
        f["frame_id"] for f in frames
        if int(f["predicted_instances"]) == max_count
    ] if max_count >= 0 else None
    return buckets


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = prepare_output_dir(args.output_dir)

    from gui.inference_engine import DualStageInferenceEngine
    from models.material_classifier import (
        load_material_runtime_config,
        verify_material_runtime_fixed_values,
    )

    config = load_material_runtime_config(args.config)
    verify_material_runtime_fixed_values(config)
    budget_started = time.perf_counter()

    def budget_left() -> float:
        return args.max_seconds - (time.perf_counter() - budget_started)

    cuda_available = torch.cuda.is_available()
    conditions = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "driver": _driver_version() if cuda_available else "not_applicable",
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu": platform.processor(),
        "precision": config.precision,
        "model_a_imgsz": config.model_a_imgsz,
        "cmx_imgsz": config.cmx_imgsz,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_seconds": args.max_seconds,
        "timing_clock": "time.perf_counter (monotonic high-resolution)",
        "gpu_sync": "torch.cuda.synchronize before the start timestamp and "
                    "after each diagnostic stage; the end-to-end pass "
                    "synchronizes once at the end of the whole frame",
        "residency": "E1 and E4 are loaded and measured sequentially; the "
                     "first engine is released before the second is built",
        "scope": performance_scope(args.scope),
        "stage_definitions": stage_definitions(),
        "config_identities": {
            "model_a_sha256": sha256_of(PROJECT_ROOT / config.model_a_path),
            "fallback_sha256": sha256_of(PROJECT_ROOT / config.fallback_path),
            "fallback_identity": config.fallback_identity,
        },
    }
    write_json_bytes(output_dir, "benchmark_conditions.json", conditions)

    # --- frame source (preloaded; disk decode is excluded from timing) ---
    split_csv = PROJECT_ROOT / "datasets/underwater_seg_binary_v2/split_assignment.csv"
    with split_csv.open(encoding="utf-8") as handle:
        split_rows = [row for row in csv.DictReader(handle)
                      if row["split"] == args.split]
    import cv2

    from core.stereo_matching import to_gray_u8
    from scripts.verify_material_runtime import right_image_path

    left_rights = {}
    for row in split_rows:
        left_path = PROJECT_ROOT / row["left_path"]
        right_path = right_image_path(left_path)
        left = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED) \
            if right_path else None
        if left is None or right is None:
            continue
        left_rights[row["output_stem"]] = (to_gray_u8(left), to_gray_u8(right))

    def _synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    residency: list[dict] = []
    raw_timings: dict[str, dict] = {}
    scenarios_report: dict[str, dict] = {}
    measured_frames: dict[str, list[str]] = {}
    rectification_report: dict = {"status": "NOT_MEASURED",
                                  "reason": "no rectifier in offline replay"}

    for variant, seed in REPRESENTATIVES:
        # --- one engine resident at a time ---
        engine = DualStageInferenceEngine.from_material_config(
            args.config, variant=variant, seed=seed, device=args.device,
        )
        loaded = _vram_mb()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if not scenarios_report:
            # survey + fixed selection happen once, before any timing
            survey = _survey_frames(engine, split_rows, left_rights,
                                    config.model_a_imgsz)
            write_jsonl_bytes(output_dir, "scenario_survey.jsonl", survey)
            buckets = bucket_by_predictions(survey, wanted=SCENARIO_TARGETS)
            selection = {}
            for bucket, frame_ids in buckets.items():
                if not frame_ids:
                    selection[bucket] = None
                    continue
                frame_id = sorted(frame_ids)[0]
                row = next(r for r in split_rows
                           if r["output_stem"] == frame_id)
                selection[bucket] = {
                    "frame_id": frame_id,
                    "group": row["group_name"],
                    "left_path": row["left_path"],
                    "left_sha256": sha256_of(PROJECT_ROOT / row["left_path"]),
                }
            persist_scenario_selection(output_dir, selection)

        # standalone rectification diagnostic (excluded from the replay total)
        if variant == "gray" and engine.rectifier is not None:
            manifest = PROJECT_ROOT / "datasets/Rectified_v2/rectify_manifest.csv"
            raw_row = None
            if manifest.is_file():
                with manifest.open(encoding="utf-8") as handle:
                    raw_row = next(iter(csv.DictReader(handle)), None)
            if raw_row is not None:
                raw_left = cv2.imread(
                    str(PROJECT_ROOT / raw_row["source_left"]),
                    cv2.IMREAD_UNCHANGED)
                raw_right = cv2.imread(
                    str(PROJECT_ROOT / raw_row["source_right"]),
                    cv2.IMREAD_UNCHANGED)
                if raw_left is not None and raw_right is not None:
                    def _one_rectify():
                        started = time.perf_counter()
                        engine.rectifier.rectify(raw_left, raw_right)
                        return time.perf_counter() - started

                    samples = run_repeats(_one_rectify, warmup=3, repeats=20)
                    rectification_report = percentile_summary(samples)
                    rectification_report["scope"] = (
                        "standalone online-rectification diagnostic on one raw "
                        "pair; the offline replay path uses already_rectified "
                        "frames, so this stage is excluded from the total"
                    )
                    raw_timings["rectification_standalone"] = {
                        "total": samples}

        for bucket in [str(t) for t in SCENARIO_TARGETS] + ["max_available"]:
            frame_ids = buckets.get(bucket)
            entry = scenarios_report.setdefault(bucket, {"bucket": bucket})
            if not frame_ids:
                entry.setdefault("status", "NOT_COVERED")
                continue
            frame_id = sorted(frame_ids)[0]
            row = next(r for r in split_rows if r["output_stem"] == frame_id)
            left_gray, right_gray = left_rights[frame_id]
            variant_report = entry.setdefault("variants", {}).setdefault(
                variant, {})
            if budget_left() <= 60:
                variant_report.update({
                    "status": "PARTIAL",
                    "reason": "performance budget spent",
                })
                continue
            measured_frames.setdefault(bucket, []).append(frame_id)

            stage_samples: dict[str, list[float]] = {}

            def _stage(name: str, fn: Callable[[], Any], repeats: int = 5):
                samples = [timed_stage(fn, time.perf_counter, _synchronize,
                                       args.device)]
                for _ in range(repeats - 1):
                    samples.append(timed_stage(fn, time.perf_counter,
                                               _synchronize, args.device))
                stage_samples[name] = samples

            instances_cache: list = []

            def _model_a():
                found = engine.model_a.predict(
                    cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR),
                    **model_a_predict_kwargs(config.model_a_imgsz),
                )
                instances_cache.clear()
                instances_cache.extend(found)
                return found

            _stage("model_a", _model_a)
            instances = list(instances_cache)
            from core.stereo_matching import build_horizontal_bands

            def _sgbm():
                bands = build_horizontal_bands(
                    (inst.bbox for inst in instances),
                    image_height=left_gray.shape[0],
                    vertical_margin=engine.band_vertical_margin,
                )
                if hasattr(engine.matcher, "compute_bands"):
                    return engine.matcher.compute_bands(
                        left_gray, right_gray, bands)
                return engine.matcher.compute(left_gray, right_gray)

            _stage("band_sgbm", _sgbm)
            stereo_result = _sgbm()
            from core.polar_compute import compute_polar_features

            def _depth_polar():
                polar_mask = np.zeros(left_gray.shape, dtype=bool)
                for inst in instances:
                    stats = engine.matcher.instance_stats(
                        stereo_result, inst.mask, instance_id=inst.id)
                    if stats.valid:
                        polar_mask |= inst.mask > 0
                if np.any(polar_mask):
                    compute_polar_features(
                        left_gray, right_gray, stereo_result.disparity,
                        object_mask=polar_mask,
                        disparity_valid=stereo_result.valid,
                    )

            _stage("depth_polar", _depth_polar)

            stats_list = [
                engine.matcher.instance_stats(stereo_result, inst.mask,
                                              instance_id=inst.id)
                for inst in instances
            ]
            polar_mask = np.zeros(left_gray.shape, dtype=bool)
            for inst, stats in zip(instances, stats_list):
                if stats.valid:
                    polar_mask |= inst.mask > 0
            polar_result = compute_polar_features(
                left_gray, right_gray, stereo_result.disparity,
                object_mask=polar_mask if np.any(polar_mask)
                else np.zeros(left_gray.shape, dtype=bool),
                disparity_valid=stereo_result.valid,
            ) if instances else None
            material_inputs = [
                engine._instance_material_input(
                    inst, left_gray, polar_result,
                    polar_invalid=not stats.valid,
                )
                for inst, stats in zip(instances, stats_list)
            ]

            def _preprocess_cpu():
                from models.material_classifier import material_input_tensors

                for sample in material_inputs:
                    if sample is not None:
                        material_input_tensors(
                            sample, engine.model_b.imgsz, engine.model_b.variant)

            _stage("cpu_tensor_preparation", _preprocess_cpu)

            def _classify():
                return engine.model_b.predict_batch(material_inputs)

            _stage("classification_wrapper", _classify)
            routing_counts = {"cmx": 0, "fallback": 0, "not_evaluable": 0}
            for result in _classify():
                if not result.valid:
                    routing_counts["not_evaluable"] += 1
                elif result.used_fallback:
                    routing_counts["fallback"] += 1
                else:
                    routing_counts["cmx"] += 1
            actual_batch = sum(
                1 for sample in material_inputs if sample is not None)

            if engine.model_b.fallback is not None and material_inputs:
                import dataclasses

                forced = [
                    dataclasses.replace(
                        s, polar_invalid=True,
                        invalid_reason="forced_invalid_benchmark")
                    for s in material_inputs if s is not None
                ]

                def _all_fallback():
                    return engine.model_b.predict_batch(forced)

                _stage("classification_all_fallback", _all_fallback)
                mixed_status = mixed_fallback_status(len(forced))
                if mixed_status == "MEASURED":
                    mixed = [
                        dataclasses.replace(
                            s, polar_invalid=(index % 2 == 0),
                            invalid_reason="forced_invalid_benchmark")
                        for index, s in enumerate(forced)
                    ]

                    def _mixed_fallback():
                        return engine.model_b.predict_batch(mixed)

                    _stage("classification_mixed_fallback", _mixed_fallback)
            else:
                mixed_status = "NOT_APPLICABLE"

            # --- end-to-end: one synchronization at the end of the frame ---
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            def _one_full_frame():
                started = time.perf_counter()
                engine.process_frame_detailed(
                    left_gray, right_gray, sync_skew_ms=None,
                    already_rectified=True, frame_id=frame_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                return time.perf_counter() - started

            total = run_repeats(
                _one_full_frame, warmup=args.warmup, repeats=args.repeats,
                max_seconds=max(60.0, budget_left()),
            )
            total_samples = total["samples"] if isinstance(total, dict) else total
            summary = summarise_repeats(stage_samples, total_samples)
            summary["throughput_fps"] = (
                len(total_samples) / sum(total_samples) if total_samples else 0.0
            )
            summary["measured_repeats"] = len(total_samples)
            summary["requested_repeats"] = args.repeats
            summary["warmup"] = args.warmup
            summary["status"] = (
                "COMPLETED"
                if (not isinstance(total, dict) or total["status"] == "COMPLETED")
                and len(total_samples) >= args.repeats
                else "PARTIAL"
            )
            summary["partial_reason"] = (
                None if summary["status"] == "COMPLETED"
                else "repeat budget reached before the requested count"
            )
            summary["actual_batch"] = actual_batch
            summary["configured_classification_batch"] = config.classification_batch
            summary["batch_note"] = (
                "actual instances per classification call on this frame; "
                "classification_batch=8 is a configured value, not a batching "
                "that was applied here"
            )
            summary["routing_counts"] = routing_counts
            summary["fallback_measurement_level"] = "classifier-level"
            summary["mixed_fallback"] = mixed_status
            summary["instances"] = len(instances)
            summary["target_area_fraction"] = (
                float(np.logical_or.reduce([inst.mask > 0 for inst in instances]
                                           ).sum())
                / (left_gray.shape[0] * left_gray.shape[1])
                if instances else 0.0
            )
            summary["peak_vram"] = _vram_mb()
            summary["peak_allocated_mb"] = (
                float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
                if torch.cuda.is_available() else None
            )
            summary["peak_reserved_mb"] = (
                float(torch.cuda.max_memory_reserved()) / (1024 * 1024)
                if torch.cuda.is_available() else None
            )
            variant_report.update(summary)
            raw_timings.setdefault(bucket, {})[variant] = {
                "total": [float(x) for x in total_samples],
                "stages": {name: [float(x) for x in values]
                           for name, values in stage_samples.items()},
            }

        residency.append(residency_record(
            variant=variant, seed=seed,
            components=["model_a", engine.model_b.model_identity]
            + (["independent_gray_fallback"]
               if getattr(engine.model_b, "fallback", None) is not None else []),
            loaded_vram_mb=loaded.get("allocated"),
            peak_allocated_mb=(
                float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
                if torch.cuda.is_available() else None),
            peak_reserved_mb=(
                float(torch.cuda.max_memory_reserved()) / (1024 * 1024)
                if torch.cuda.is_available() else None),
            released=True,
        ))
        # release this engine before the next one is built
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_raw_timings(output_dir, raw_timings)
        write_json_bytes(output_dir, "performance_report.json", {
            "status": "RUNNING",
            "conditions": conditions,
            "residency": residency,
            "scenario_buckets": {k: v for k, v in buckets.items()},
            "scenarios": scenarios_report,
        })

    coverage = bucket_coverage(buckets, measured_frames)
    item_statuses = []
    for bucket, entry in scenarios_report.items():
        if entry.get("status") == "NOT_COVERED":
            item_statuses.append({"status": "NOT_COVERED"})
            continue
        for variant_report in (entry.get("variants") or {}).values():
            item_statuses.append(variant_report)
    status = overall_status(item_statuses)

    report = {
        "status": status,
        "generated_by": "scripts/benchmark_material_pipeline.py",
        "representatives": [
            {"variant": variant, "seed": seed} for variant, seed in REPRESENTATIVES
        ],
        "conditions": conditions,
        "residency": residency,
        "rectification_standalone": rectification_report,
        "scenario_buckets": {k: v for k, v in buckets.items()},
        "coverage": coverage,
        "scenarios": scenarios_report,
        "raw_timings_file": "raw_timings.json",
        "honest_state": [
            "offline replay of preloaded frames; queue latency and drop rate "
            "are NOT_MEASURED, and this is not a camera-to-GUI end-to-end "
            "FPS claim",
            "the per-stage sum is a diagnostic, never the pipeline latency; "
            "classification_wrapper is a wrapper total, not a pure forward "
            "time, and no conclusion about the primary bottleneck is drawn "
            "from it",
            "repeating one frame is not independent scene coverage; the "
            "measured_unique_frames field states the real coverage",
            "20-30 FPS is a target, not a conclusion of this report",
            "no distance ground truth exists; no distance-accuracy claim is "
            "made here",
        ],
    }
    write_json_bytes(output_dir, "performance_report.json", report)
    manifest = {
        "files": {
            p.name: sha256_of(p)
            for p in sorted(output_dir.iterdir()) if p.is_file()
            and p.name != "artifact_manifest.json"
        },
        "index_file": "artifact_manifest.json",
        "index_note": "index of this directory; does not cover itself",
    }
    write_json_bytes(output_dir, "artifact_manifest.json", manifest)
    print(f"performance report: {output_dir / 'performance_report.json'}")
    print(f"status: {status}")
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
