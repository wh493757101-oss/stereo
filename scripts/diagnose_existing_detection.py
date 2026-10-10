"""Fixed-condition Model A detection error attribution (Part 2A).

Four frozen conditions on the ORIGINAL val split — no grid search, no
threshold tuning, no classifier, no SGBM, no training:

    D0  imgsz=640, conf=0.25   (the historical operating point)
    D1  imgsz=640, conf=0.10
    D2  imgsz=640, conf=0.05
    D3  imgsz=960,  conf=0.25

Everything else (precision, NMS IoU, mask recovery, matching) is the
production path. D0 runs FIRST and is checked against the historical
4B evidence; when the reproduction fails, later conditions do not run
and nothing is re-tuned to force the counts to agree.

Matching reuses :func:`scripts.eval_material_pipeline.match_instances`
(class-agnostic one-to-one, mask IoU >= 0.50, count-first then total
IoU) with the threshold NEVER lowered. GT identity across conditions is
``(frame_id, object_index)``; prediction ids are per-condition and never
treated as the same target across conditions.

Attribution outputs are rule-based SIGNS (``待确认``), never a claim of
human review and never a fabricated NMS/low-confidence root cause.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.existing_data_protocol import (  # noqa: E402
    BudgetExceeded,
    BudgetWatch,
    load_experiment_config,
    verify_baseline_identities,
)
from core.gt_geometry import build_gt_geometry  # noqa: E402
from scripts.eval_material_pipeline import (  # noqa: E402
    IOU_THRESHOLD,
    match_instances,
    mask_iou,
)

EXIT_OK = 0
EXIT_REFUSED = 1

ATTRIBUTION_SIGN_NOTE = (
    "rule-based sign only; no human review was performed and no root "
    "cause (NMS or otherwise) is asserted"
)
CLASS_NAMES = ("metal_submarine", "plastic_submarine",
               "plastic_fish", "real_fish")


def write_json(path: Path, payload) -> Path:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def write_jsonl(path: Path, records) -> Path:
    path.write_bytes("".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True,
                   allow_nan=False) + "\n"
        for record in records).encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# Condition plan and historical baseline


def build_condition_plan(config: dict) -> list[dict]:
    """The frozen condition matrix, in the configured order."""
    detection = config["detection"]
    conditions = detection["conditions"]
    order = list(detection.get("order", sorted(conditions)))
    match_iou = float(detection["match_iou_threshold"])
    if match_iou != IOU_THRESHOLD:
        raise ValueError(
            f"match_iou_threshold {match_iou} does not match the frozen "
            f"evaluation matching threshold {IOU_THRESHOLD}; lowering it is "
            "not allowed"
        )
    plan = []
    for name in order:
        condition = conditions[name]
        plan.append({
            "name": name,
            "imgsz": int(condition["imgsz"]),
            "conf": float(condition["conf"]),
            "nms_iou": float(detection["nms_iou"]),
            "match_iou": match_iou,
        })
    if not plan:
        raise ValueError("the detection config defines no conditions")
    return plan


def load_historical_d0(config: dict, project_root: Path, resolver=None) -> dict:
    """Aggregate + per-frame D0 baseline from the recorded 4B evidence.

    The counts are read from the evidence file (never hardcoded); the
    reference candidate/layer comes from the config.
    """
    import json as json_module

    root = Path(project_root)
    resolve = resolver or (lambda relative: root / relative)
    evidence = config["baselines"]["historical_d0_evidence"]
    frames_path = resolve(evidence["frames_jsonl"])
    if not frames_path.is_file():
        raise FileNotFoundError(
            f"historical D0 evidence not found: {frames_path}"
        )
    per_frame = {}
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json_module.loads(line)
        per_frame[record["frame_id"]] = {
            "predictions": int(record["predictions"]),
            "matched": int(record["matched"]),
            "gt_instances": int(record["gt_instances"]),
        }
    summary_path = resolve(evidence.get("summary_json", ""))
    reference = None
    if summary_path and Path(summary_path).is_file():
        summary = json_module.loads(Path(summary_path).read_text(encoding="utf-8"))
        candidate = summary["per_candidate"][evidence["reference_candidate"]]
        layer, condition = evidence["reference_layer"].split(".")
        reference = {
            "candidate": evidence["reference_candidate"],
            "layer": evidence["reference_layer"],
            "counts": candidate[layer][condition]["counts"],
        }
    return {
        "source": str(frames_path),
        "reference": reference,
        "frames": len(per_frame),
        "gt_instances": sum(v["gt_instances"] for v in per_frame.values()),
        "predictions": sum(v["predictions"] for v in per_frame.values()),
        "matched": sum(v["matched"] for v in per_frame.values()),
        "per_frame": per_frame,
    }


def d0_consistency_report(observed: dict, historical: dict) -> dict:
    """Compare the reproduced D0 with the historical baseline.

    Aggregate counts and every per-frame prediction/matched count must
    agree; a mismatch is reported concretely and (in ``require`` mode)
    stops the remaining conditions.
    """
    mismatches = []
    for field in ("frames", "gt_instances", "predictions", "matched"):
        if observed.get(field) != historical.get(field):
            mismatches.append(
                f"{field}: observed {observed.get(field)} != historical "
                f"{historical.get(field)}"
            )
    observed_frames = observed.get("per_frame", {})
    for frame_id, expected in historical.get("per_frame", {}).items():
        actual = observed_frames.get(frame_id)
        if actual is None:
            mismatches.append(f"frame {frame_id}: missing from the run")
            continue
        for field in ("predictions", "matched"):
            if actual.get(field) != expected.get(field):
                mismatches.append(
                    f"frame {frame_id}: {field} {actual.get(field)} != "
                    f"historical {expected.get(field)}"
                )
    return {
        "consistent": not mismatches,
        "mismatches": mismatches[:50],
        "mismatch_count": len(mismatches),
        "historical_source": historical.get("source"),
        "note": (
            "the historical counts are a baseline to re-check, not numbers "
            "new inputs must satisfy; a reproduction failure stops the "
            "comparison instead of re-tuning the rules"
        ),
    }


# ---------------------------------------------------------------------------
# Attribution rules (signs)


def attribution_for_unmatched_gt(signs: dict) -> str:
    """One priority-ordered sign per unmatched GT (never a root cause).

    Priority: rescued by lower confidence -> rescued by larger input
    scale -> best D0 overlap already matched elsewhere (duplicate/merge
    sign) -> positive but insufficient overlap -> no positive overlap
    with predictions present -> no prediction in the frame at all.
    """
    if signs.get("matched_D1") or signs.get("matched_D2"):
        return "confidence_related"
    if signs.get("matched_D3"):
        return "input_scale_related"
    best_elsewhere = signs.get("d0_best_matched_elsewhere")
    max_iou = signs.get("d0_max_iou")
    if max_iou is not None and max_iou > 0.0 and best_elsewhere:
        return "duplicate_or_merge_sign"
    if max_iou is not None and max_iou > 0.0:
        return "insufficient_mask_overlap_sign"
    if int(signs.get("d0_predictions_in_frame") or 0) == 0:
        return "no_prediction_in_frame"
    return "no_corresponding_prediction_sign"


# ---------------------------------------------------------------------------
# Diagnosis runner (model injected; the real entry passes Model A)


class _ConditionedModelA:
    """The production Model A adapter for the fixed conditions.

    The wrapper's confidence threshold is the wrapper's own protocol; the
    condition's conf is set on it before each call and the condition's
    imgsz is forwarded as the predict kwarg — exactly the path
    ``gui.inference_engine`` uses (``_predict_kwargs``).
    """

    def __init__(self, model_path: str | Path, nms_iou: float, device: str):
        from models.segmentation import SegmentationModel

        self._wrapper = SegmentationModel(
            model_path,
            device=device,
            conf_threshold=0.25,
            iou_threshold=float(nms_iou),
        )
        # the underlying torch module, for the parameter/buffer snapshot
        # evidence (ultralytics YOLO.model is an nn.Module)
        self.model = self._wrapper.model

    def predict(self, image_bgr: np.ndarray, *, conf: float, imgsz: int):
        self._wrapper.conf_threshold = float(conf)
        return self._wrapper.predict(image_bgr, imgsz=int(imgsz))


def run_detection_diagnosis(
    *,
    config: dict,
    output_dir: str | Path,
    frames_provider,
    model_factory,
    historical: dict | None,
    device: str = "cpu",
    log=lambda message: None,
    budget: "BudgetWatch | None" = None,
) -> dict:
    """Run the frozen conditions over the frames and write the evidence.

    ``model_factory(path, conf, iou, imgsz, device)`` returns an object
    with ``predict(image_bgr, conf=..., imgsz=...)`` and a ``.model``
    torch module (snapshotted for the no-change evidence). ``historical``
    binds the D0 reproduction check; ``None`` skips the check (synthetic
    runs) — the real entry always passes it. ``budget`` is the shared
    monotonic budget checked at every condition and frame boundary; on
    exhaustion the run stops with at least a PARTIAL status and keeps the
    finished evidence.
    """
    from core.existing_data_protocol import BudgetExceeded
    output_dir = Path(output_dir)
    # the caller owns the existence check (refuse-never-overwrite); this
    # only creates the tree for direct API use
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_condition_plan(config)
    frames = frames_provider()
    if not frames:
        raise ValueError("the frame provider returned no frames")

    model_a_config = config["baselines"]["model_a"]
    first = plan[0]
    model = model_factory(
        model_a_config["path"], first["conf"], first["nms_iou"],
        first["imgsz"], device,
    )
    if not hasattr(model, "model"):
        raise ValueError(
            "the model factory must expose the underlying torch module as "
            ".model for the parameter/buffer snapshot evidence"
        )
    from core.material_probe import compare_module_states, snapshot_module_state

    # First-load / first-inference initialization is allowed to change
    # structure (lazy buffers etc.); the evidence baseline is established
    # AFTER each condition's first forward and must then stay byte-stable.
    init_baseline = snapshot_module_state(model.model)
    init_changes_record: dict[str, dict] = {}

    conditions_summary: dict[str, dict] = {}
    d0_per_gt: list[dict] = []
    d0_unmatched: list[dict] = []
    d0_report = None
    status = "COMPLETED"

    budget_status = None
    for condition in plan:
        if budget is not None:
            try:
                budget.check(f"condition {condition['name']} start")
            except BudgetExceeded as error:
                budget_status = str(error)
                status = "PARTIAL_BUDGET_EXCEEDED"
                break
        name = condition["name"]
        condition_dir = output_dir / "conditions" / name
        condition_dir.mkdir(parents=True, exist_ok=True)
        per_gt_rows: list[dict] = []
        prediction_rows: list[dict] = []
        frame_rows: list[dict] = []
        unmatched_gt_rows: list[dict] = []
        unmatched_pred_rows: list[dict] = []
        counts = {
            "frames": 0, "gt_instances": 0, "predictions": 0,
            "matched": 0, "missed_gt": 0, "unmatched_gt": 0,
            "unmatched_predictions": 0,
        }
        matched_by_class = {name_: 0 for name_ in CLASS_NAMES}
        gt_by_class = {name_: 0 for name_ in CLASS_NAMES}
        iou_values: list[float] = []
        per_frame_state: dict[str, dict] = {}
        mask_dir = condition_dir / "masks"
        mask_dir.mkdir(exist_ok=True)

        for frame in frames:
            if budget is not None:
                try:
                    budget.check(f"frame {frame['frame_id']}")
                except BudgetExceeded as error:
                    budget_status = str(error)
                    status = "PARTIAL_BUDGET_EXCEEDED"
                    break
            counts["frames"] += 1
            gt_masks = [gt["mask"] for gt in frame["gt"]]
            for gt in frame["gt"]:
                class_name = (gt["class_name"] if 0 <= gt["class_id"]
                              < len(CLASS_NAMES) else None)
                if class_name in gt_by_class:
                    gt_by_class[class_name] += 1
            height, width = frame["image"].shape[:2]
            instances = model.predict(
                frame["image"], conf=condition["conf"],
                imgsz=condition["imgsz"])
            if counts["frames"] == 1:
                warm_state = snapshot_module_state(model.model)
                init_report = compare_module_states(init_baseline, warm_state)
                init_changes_record[name] = {
                    "allowed_first_forward_changes": init_report,
                }
                init_baseline = warm_state  # the evidence baseline from here
            pred_masks = []
            for instance in instances:
                mask = np.asarray(instance.mask)
                if mask.shape != (height, width):
                    raise ValueError(
                        f"prediction mask shape {mask.shape} does not match "
                        f"the frame {(height, width)}; the mask recovery "
                        "must restore the original size"
                    )
                pred_masks.append(mask)
            matches = match_instances(
                gt_masks, pred_masks, iou_threshold=condition["match_iou"])
            matched_gt = {gi for gi, _, _ in matches}
            matched_pred = {pi for _, pi, _ in matches}

            for gi, gt in enumerate(frame["gt"]):
                match = next((m for m in matches if m[0] == gi), None)
                ious = [mask_iou(gt_masks[gi], pred) for pred in pred_masks]
                max_iou = max(ious) if ious else None
                best_index = (int(np.argmax(ious))
                              if ious and max_iou > 0.0 else None)
                row = {
                    "frame_id": frame["frame_id"],
                    "group": frame["group"],
                    "gt_id": gt["object_index"],
                    "class_id": gt["class_id"],
                    "class_name": gt["class_name"],
                    "matched": match is not None,
                    "match_iou": match[2] if match else None,
                    "max_iou_any_prediction": max_iou,
                    "best_prediction_index": best_index,
                    "best_prediction_matched_elsewhere": (
                        best_index in matched_pred
                        if best_index is not None else None),
                    "predictions_in_frame": len(pred_masks),
                }
                per_gt_rows.append(row)
                if match is not None:
                    counts["matched"] += 1
                    iou_values.append(match[2])
                    if gt["class_name"] in matched_by_class:
                        matched_by_class[gt["class_name"]] += 1
                else:
                    unmatched_gt_rows.append({
                        "frame_id": frame["frame_id"],
                        "group": frame["group"],
                        "gt_id": gt["object_index"],
                        "class_id": gt["class_id"],
                        "class_name": gt["class_name"],
                        "max_iou_any_prediction": max_iou,
                        "best_prediction_index": best_index,
                        "best_prediction_matched_elsewhere": (
                            best_index in matched_pred
                            if best_index is not None else None),
                        "no_positive_overlap": (
                            max_iou is not None and max_iou == 0.0),
                        "predictions_in_frame": len(pred_masks),
                        "attribution": None,  # filled after all conditions
                        "note": ATTRIBUTION_SIGN_NOTE,
                    })
            for pi, instance in enumerate(instances):
                gt_ious = [mask_iou(pred_masks[pi], gt_mask)
                           for gt_mask in gt_masks]
                max_gt_iou = max(gt_ious) if gt_ious else None
                matched_gt_id = next(
                    (gi for gi, p, _ in matches if p == pi), None)
                prediction_rows.append({
                    "frame_id": frame["frame_id"],
                    "group": frame["group"],
                    "prediction_index": pi,
                    "bbox": list(instance.bbox),
                    "confidence": float(instance.confidence),
                    "class_id": int(instance.class_id),
                    "mask_pixels": int(pred_masks[pi].sum()),
                    "matched_gt_id": matched_gt_id,
                    "max_iou_any_gt": max_gt_iou,
                })
                if pi not in matched_pred:
                    unmatched_pred_rows.append({
                        "frame_id": frame["frame_id"],
                        "group": frame["group"],
                        "prediction_index": pi,
                        "confidence": float(instance.confidence),
                        "class_id": int(instance.class_id),
                        "bbox": list(instance.bbox),
                        "mask_pixels": int(pred_masks[pi].sum()),
                        "max_iou_any_gt": max_gt_iou,
                        "note": (
                            "no GT reached the frozen match threshold; may "
                            "be background, duplicate or a contour-quality "
                            "failure — sign only"),
                    })

            np.savez_compressed(
                mask_dir / f"{frame['frame_id']}.npz",
                masks=np.packbits(
                    np.stack([m > 0 for m in pred_masks]).reshape(
                        len(pred_masks), -1), axis=1)
                if pred_masks else np.zeros((0, (height * width + 7) // 8),
                                            dtype=np.uint8),
                shape=np.asarray([height, width]),
                confidences=np.asarray(
                    [float(i.confidence) for i in instances], dtype=np.float32),
                bboxes=np.asarray([list(i.bbox) for i in instances],
                                  dtype=np.int32)
                if instances else np.zeros((0, 4), dtype=np.int32),
            )
            frame_rows.append({
                "frame_id": frame["frame_id"],
                "group": frame["group"],
                "gt_instances": len(frame["gt"]),
                "predictions": len(pred_masks),
                "matched": len(matches),
                "missed_gt": len(frame["gt"]) - len(matches),
                "unmatched_predictions": len(pred_masks) - len(matches),
            })
            per_frame_state[frame["frame_id"]] = {
                "predictions": len(pred_masks), "matched": len(matches),
            }
            counts["gt_instances"] += len(frame["gt"])
            counts["predictions"] += len(pred_masks)

        counts["missed_gt"] = counts["gt_instances"] - counts["matched"]
        counts["unmatched_gt"] = counts["gt_instances"] - counts["matched"]
        counts["unmatched_predictions"] = (
            counts["predictions"] - counts["matched"])

        observed = {
            "frames": counts["frames"],
            "gt_instances": counts["gt_instances"],
            "predictions": counts["predictions"],
            "matched": counts["matched"],
            "per_frame": per_frame_state,
        }
        consistency = None
        if name == "D0" and historical is not None:
            consistency = d0_consistency_report(observed, historical)
            d0_report = consistency
            write_json(output_dir / "d0_consistency.json", consistency)
            if not consistency["consistent"]:
                detection_config = config["detection"]
                if detection_config.get("d0_reproduction", "require") == "require":
                    status = "FAILED_D0_REPRODUCTION"

        per_gt_index = {
            (row["frame_id"], row["gt_id"]): row for row in per_gt_rows
        }
        conditions_summary[name] = {
            "conf": condition["conf"],
            "imgsz": condition["imgsz"],
            "nms_iou": condition["nms_iou"],
            "match_iou": condition["match_iou"],
            "counts": counts,
            "recall_by_class": {
                class_name: (
                    matched_by_class[class_name] / gt_by_class[class_name]
                    if gt_by_class[class_name] else None)
                for class_name in CLASS_NAMES
            },
            "gt_by_class": gt_by_class,
            "matched_iou": {
                "n": len(iou_values),
                "mean": float(np.mean(iou_values)) if iou_values else None,
                "min": float(np.min(iou_values)) if iou_values else None,
                "max": float(np.max(iou_values)) if iou_values else None,
            },
            "unmatched_gt": len(unmatched_gt_rows),
            "unmatched_predictions": len(unmatched_pred_rows),
        }

        if name == "D0":
            d0_per_gt = per_gt_rows
            d0_unmatched = unmatched_gt_rows

        write_jsonl(condition_dir / "frames.jsonl", frame_rows)
        write_jsonl(condition_dir / "per_gt.jsonl", per_gt_rows)
        write_jsonl(condition_dir / "predictions.jsonl", prediction_rows)
        write_jsonl(condition_dir / "unmatched_gt.jsonl", unmatched_gt_rows)
        write_jsonl(condition_dir / "unmatched_predictions.jsonl",
                    unmatched_pred_rows)
        log(f"  {name}: {counts['predictions']} predictions, "
            f"{counts['matched']} matched, {counts['unmatched_gt']} "
            f"unmatched GT")

        state_report = compare_module_states(
            init_baseline, snapshot_module_state(model.model))
        if not state_report["unchanged"]:
            raise ValueError(
                f"Model A parameters/buffers changed during condition "
                f"{name} after its first forward: "
                f"{state_report['changed_tensors']}"
            )
        if budget_status is not None:
            status = "PARTIAL_BUDGET_EXCEEDED"
            break
        if status != "COMPLETED":
            break

    # ---- cross-condition attribution for the D0 unmatched GT ----
    run_attribution = (d0_report is None) or d0_report["consistent"]
    if run_attribution:
        later = {
            cond_name: {
                (row["frame_id"], row["gt_id"]): row["matched"]
                for row in _read_jsonl_if_exists(
                    output_dir / "conditions" / cond_name / "per_gt.jsonl")
            }
            for cond_name in ("D1", "D2", "D3")
        }
        d0_index = {
            (row["frame_id"], row["gt_id"]): row for row in d0_per_gt
        }
        for row in d0_unmatched:
            key = (row["frame_id"], row["gt_id"])
            signs = {
                "matched_D1": later["D1"].get(key, False),
                "matched_D2": later["D2"].get(key, False),
                "matched_D3": later["D3"].get(key, False),
                "d0_max_iou": d0_index[key]["max_iou_any_prediction"],
                "d0_best_matched_elsewhere":
                    d0_index[key]["best_prediction_matched_elsewhere"],
                "d0_predictions_in_frame":
                    d0_index[key]["predictions_in_frame"],
            }
            row["attribution"] = attribution_for_unmatched_gt(signs)
            row["signs"] = signs
        d0_out_dir = output_dir / "conditions" / "D0"
        d0_out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "conditions" / "D0" / "unmatched_gt.jsonl",
                    d0_unmatched)
        attribution_counts: dict[str, int] = {}
        for row in d0_unmatched:
            attribution_counts[row["attribution"]] = (
                attribution_counts.get(row["attribution"], 0) + 1)
        if "D0" in conditions_summary:
            conditions_summary["D0"]["attribution_counts"] = attribution_counts

    # ---- atlas (D0 events + fixed-order matched controls), redrawn from
    # the SAVED evidence so predictions are actually drawn on every panel
    try:
        atlas_report = rebuild_atlas_from_evidence(
            output_dir, frames, output_dir,
            condition=plan[0]["name"],
            controls_per_class=int(
                config["detection"]["atlas"]["matched_controls_per_class"]),
            margin=int(config["detection"]["atlas"].get("margin", 48)),
        )
        if atlas_report["status"] != "COMPLETED":
            status = status if status != "COMPLETED" else "PARTIAL"
    except Exception as error:  # noqa: BLE001 - panels must not kill evidence
        log(f"  atlas rendering failed (evidence kept): {error}")
        write_json(output_dir / "atlas_error.json",
                   {"error": f"{type(error).__name__}: {error}"})
        atlas_report = {"status": "FAILED",
                        "error": f"{type(error).__name__}: {error}"}
        status = status if status != "COMPLETED" else "PARTIAL"

    summary = {
        "status": status,
        "conditions": conditions_summary,
        "d0_consistency": d0_report,
        "first_forward_init_changes": init_changes_record,
        "atlas": atlas_report,
        "budget_exhausted": budget_status,
        "attribution_note": ATTRIBUTION_SIGN_NOTE,
        "matching_rule": (
            "class-agnostic one-to-one mask IoU >= 0.50 (the frozen "
            "evaluation threshold), count-first then total IoU; reused "
            "from scripts/eval_material_pipeline, never lowered"),
        "note": (
            "detection-only diagnosis of the frozen Model A; no classifier "
            "is run and no four-class accuracy is derived from it"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _read_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# Atlas rendering


def render_event_panel(image: np.ndarray, gt_entries, prediction_entries,
                       path: Path, title: str, note: str = "",
                       margin: int = 48) -> Path:
    """One atlas panel that REALLY draws what it claims.

    ``gt_entries`` items carry ``mask`` (and optional ``polygon``/``label``);
    they are drawn as GREEN contours. ``prediction_entries`` items carry
    ``mask``, ``confidence`` and ``label``; they are drawn as RED contours
    with their per-prediction confidence text. The crop is framed on the
    union of everything drawn, so a panel never silently shows only the
    original image.
    """
    import cv2

    image = image.copy()
    regions = []
    for entry in gt_entries:
        mask = np.asarray(entry["mask"])
        ys, xs = np.nonzero(mask > 0)
        if xs.size:
            regions.append((int(xs.min()), int(ys.min()),
                            int(xs.max()) + 1, int(ys.max()) + 1))
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 200, 0), 1)
        if "polygon" in entry:
            for x, y in entry["polygon"]:
                cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
    for prediction in prediction_entries:
        mask = np.asarray(prediction["mask"])
        ys, xs = np.nonzero(mask > 0)
        if xs.size:
            regions.append((int(xs.min()), int(ys.min()),
                            int(xs.max()) + 1, int(ys.max()) + 1))
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
        label = prediction.get("label", "")
        text = f"pred {label} conf={prediction.get('confidence', 0):.2f}"
        if xs.size:
            anchor = (int(xs.min()), max(10, int(ys.min()) - 4))
            cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 255), 1, cv2.LINE_AA)
    if not regions:
        x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
    else:
        x1 = max(0, min(r[0] for r in regions) - margin)
        y1 = max(0, min(r[1] for r in regions) - margin)
        x2 = min(image.shape[1], max(r[2] for r in regions) + margin)
        y2 = min(image.shape[0], max(r[3] for r in regions) + margin)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image
    cv2.putText(crop, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    if note:
        cv2.putText(crop, note, (4, crop.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    return path


def _load_condition_masks(evidence_dir: Path, condition: str) -> dict:
    """frame_id -> list of prediction masks, rebuilt from the saved npz."""
    import numpy as np

    masks_by_frame: dict[str, list] = {}
    mask_dir = evidence_dir / "conditions" / condition / "masks"
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"saved prediction masks missing: {mask_dir}"
        )
    for npz_path in sorted(mask_dir.glob("*.npz")):
        with np.load(npz_path) as payload:
            shape = tuple(int(v) for v in payload["shape"])
            packed = payload["masks"]
            count = int(payload["confidences"].shape[0])
        if count == 0:
            masks_by_frame[npz_path.stem] = []
            continue
        bits = np.unpackbits(packed, axis=1)[:, :shape[0] * shape[1]]
        masks_by_frame[npz_path.stem] = [
            bits[index].reshape(shape).astype(np.uint8)
            for index in range(count)
        ]
    return masks_by_frame


def rebuild_atlas_from_evidence(
    evidence_dir: Path,
    frames,
    output_dir: Path,
    *,
    condition: str = "D0",
    controls_per_class: int = 3,
    margin: int = 48,
) -> dict:
    """Rebuild the atlas from SAVED evidence without any model call.

    Reads the condition's masks npz + jsonl files, the original frames
    (image + GT) and redraws every event panel with its predictions
    actually drawn. Returns a status report with the input digests and the
    produced panel counts; missing inputs yield FAILED/PARTIAL, never a
    pretended complete success.
    """
    import hashlib
    import json as json_module

    output_dir = Path(output_dir)
    atlas_dir = output_dir / "atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    by_frame = {frame["frame_id"]: frame for frame in frames}
    condition_dir = evidence_dir / "conditions" / condition
    inputs = {
        "predictions.jsonl": condition_dir / "predictions.jsonl",
        "per_gt.jsonl": condition_dir / "per_gt.jsonl",
        "unmatched_gt.jsonl": condition_dir / "unmatched_gt.jsonl",
        "unmatched_predictions.jsonl":
            condition_dir / "unmatched_predictions.jsonl",
    }
    problems: list[str] = []
    digests: dict[str, str] = {}
    for name, path in inputs.items():
        if not path.is_file():
            problems.append(f"missing input {name}")
            continue
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        masks_by_frame = _load_condition_masks(evidence_dir, condition)
        mask_npz_digests = {
            npz_path.name: hashlib.sha256(npz_path.read_bytes()).hexdigest()
            for npz_path in sorted(
                (condition_dir / "masks").glob("*.npz"))
        }
    except FileNotFoundError as error:
        masks_by_frame = {}
        mask_npz_digests = {}
        problems.append(str(error))

    status = "COMPLETED"
    if problems:
        status = "FAILED" if len(problems) >= len(inputs) or not masks_by_frame \
            else "PARTIAL"
        report = {
            "status": status, "problems": problems,
            "inputs": digests, "mask_files": mask_npz_digests,
            "panels": 0,
        }
        write_json(output_dir / "atlas_manifest.json", report)
        return report

    predictions_by_frame: dict[str, list] = {}
    for line in (inputs["predictions.jsonl"]).read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json_module.loads(line)
        predictions_by_frame.setdefault(record["frame_id"], []).append(record)
    unmatched_gt = [
        json_module.loads(line) for line in
        inputs["unmatched_gt.jsonl"].read_text(encoding="utf-8").splitlines()
        if line.strip()]
    unmatched_pred = [
        json_module.loads(line) for line in
        inputs["unmatched_predictions.jsonl"].read_text(
            encoding="utf-8").splitlines() if line.strip()]
    per_gt = [
        json_module.loads(line) for line in
        inputs["per_gt.jsonl"].read_text(encoding="utf-8").splitlines()
        if line.strip()]

    sign_by_gt: dict[tuple, str] = {}
    for row in unmatched_gt:
        sign_by_gt[(row["frame_id"], row["gt_id"])] = row.get(
            "attribution") or "unavailable"

    panels = 0
    for row in unmatched_gt:
        frame = by_frame.get(row["frame_id"])
        if frame is None:
            problems.append(f"frame {row['frame_id']} unavailable")
            continue
        gt_entry = next((gt for gt in frame["gt"]
                         if gt["object_index"] == row["gt_id"]), None)
        if gt_entry is None:
            problems.append(
                f"gt {row['gt_id']} of {row['frame_id']} unavailable")
            continue
        frame_predictions = predictions_by_frame.get(row["frame_id"], [])
        frame_masks = masks_by_frame.get(row["frame_id"], [])
        related = []
        for record in frame_predictions:
            index = record["prediction_index"]
            if index >= len(frame_masks):
                continue
            mask = frame_masks[index]
            if row["max_iou_any_prediction"] and row[
                    "max_iou_any_prediction"] > 0:
                overlap = mask_iou(gt_entry["mask"], mask)
                if overlap > 0:
                    related.append(record)
            else:
                related.append(record)
        related = related[:3] if related else []
        prediction_entries = [
            {"mask": frame_masks[record["prediction_index"]],
             "confidence": record["confidence"],
             "label": f"p{record['prediction_index']}"}
            for record in related
        ]
        note = (
            f"attribution={sign_by_gt[(row['frame_id'], row['gt_id'])]} "
            f"(sign, not human-reviewed); group={row['group']}"
        )
        if not prediction_entries:
            note += "; NO prediction overlaps this GT"
        title = (f"UNMATCHED GT {row['class_name']} gt{row['gt_id']} "
                 f"frame={row['frame_id']} iou_max="
                 f"{row['max_iou_any_prediction']}")
        render_event_panel(
            frame["image"], [gt_entry], prediction_entries,
            atlas_dir / f"unmatched_gt_{row['frame_id']}_gt{row['gt_id']}.png",
            title, note, margin)
        panels += 1
    for row in unmatched_pred:
        frame = by_frame.get(row["frame_id"])
        if frame is None:
            problems.append(f"frame {row['frame_id']} unavailable")
            continue
        frame_masks = masks_by_frame.get(row["frame_id"], [])
        index = row["prediction_index"]
        if index >= len(frame_masks):
            problems.append(
                f"prediction {index} of {row['frame_id']} not in the saved "
                "masks")
            continue
        prediction_entries = [{
            "mask": frame_masks[index],
            "confidence": row["confidence"],
            "label": f"p{index}",
        }]
        gt_entries = [
            {"mask": gt["mask"], "label": gt["class_name"]}
            for gt in frame["gt"]]
        title = (f"UNMATCHED PREDICTION p{index} conf={row['confidence']:.2f} "
                 f"frame={row['frame_id']} iou_gt_max="
                 f"{row['max_iou_any_gt']}")
        note = (f"no GT reached the frozen threshold (sign only, "
                f"not human-reviewed); group={row['group']}")
        render_event_panel(
            frame["image"], gt_entries, prediction_entries,
            atlas_dir / f"unmatched_pred_{row['frame_id']}_p{index}.png",
            title, note, margin)
        panels += 1
    matched_controls = [row for row in per_gt if row["matched"]]
    matched_controls.sort(
        key=lambda r: (CLASS_NAMES.index(r["class_name"])
                       if r["class_name"] in CLASS_NAMES else 99,
                       -r["match_iou"], r["frame_id"], r["gt_id"]))
    taken: dict[str, int] = {}
    for row in matched_controls:
        class_name = row["class_name"]
        if taken.get(class_name, 0) >= controls_per_class:
            continue
        frame = by_frame.get(row["frame_id"])
        if frame is None:
            continue
        gt_entry = next((gt for gt in frame["gt"]
                         if gt["object_index"] == row["gt_id"]), None)
        matched_record = next(
            (record for record in predictions_by_frame.get(
                row["frame_id"], [])
             if record["matched_gt_id"] == row["gt_id"]), None)
        frame_masks = masks_by_frame.get(row["frame_id"], [])
        if gt_entry is None or matched_record is None \
                or matched_record["prediction_index"] >= len(frame_masks):
            continue
        taken[class_name] = taken.get(class_name, 0) + 1
        prediction_entries = [{
            "mask": frame_masks[matched_record["prediction_index"]],
            "confidence": matched_record["confidence"],
            "label": f"p{matched_record['prediction_index']}",
        }]
        title = (f"MATCHED {class_name} gt{row['gt_id']} "
                 f"frame={row['frame_id']} iou={row['match_iou']:.3f}")
        note = (f"matched control; group={row['group']} "
                "(sign context, not human-reviewed)")
        render_event_panel(
            frame["image"], [gt_entry], prediction_entries,
            atlas_dir / f"control_{class_name}_{row['frame_id']}"
            f"_gt{row['gt_id']}.png",
            title, note, margin)
        panels += 1

    report = {
        "status": status if not problems else "PARTIAL",
        "problems": problems[:20],
        "inputs": digests,
        "mask_files": mask_npz_digests,
        "panels": panels,
        "unmatched_gt_panels": len(unmatched_gt),
        "unmatched_prediction_panels": len(unmatched_pred),
        "note": ("redrawn from the SAVED evidence with predictions actually "
                 "drawn; sign-only annotations, no human review claimed"),
    }
    write_json(output_dir / "atlas_manifest.json", report)
    return report


# ---------------------------------------------------------------------------
# Real entry


def real_frames_provider(config: dict, project_root: Path):
    """Val frames (image + GT geometry) from the authoritative sources."""
    import cv2

    from core.stereo_matching import to_gray_u8

    data_config = config["data"]
    split_assignment = project_root / data_config["split_assignment"]
    label_root = project_root / data_config["label_root"]
    split = config["detection"]["split"]
    from core.existing_data_protocol import read_split_rows

    rows = [row for row in read_split_rows(split_assignment, [split])
            if row["split"] == split]
    frames = []
    for row in rows:
        left_path = project_root / row["left_path"]
        image = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(
                f"left image unreadable: {left_path}; the diagnosis refuses "
                "to continue with a partial frame set"
            )
        gray = to_gray_u8(image)
        height, width = gray.shape
        label_path = label_root / split / f"{row['frame_id']}.txt"
        geometry = build_gt_geometry(
            label_path, width, height, padding=10)
        frames.append({
            "frame_id": row["frame_id"],
            "group": row["group"],
            "split": split,
            # the production Model A input: a 3-channel BGR copy of the
            # normalized left gray (Rectified_v2 is never re-rectified)
            "image": np.ascontiguousarray(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
            "gray": gray,
            "gt": [
                {
                    "object_index": item.object_index,
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "mask": item.mask,
                    "polygon": item.polygon,
                }
                for item in geometry
            ],
        })
    return frames


def measure_timing(model, frames, plan, config, device: str, *,
                   budget: "BudgetWatch | None" = None,
                   progress: dict | None = None) -> dict:
    """Warm, synced per-condition Model A stage timing (NOT system FPS).

    The device is NORMALIZED through the project's device helper before
    any CUDA decision: raw ids like ``"0"`` map to ``"cuda:0"`` and get a
    real ``torch.cuda.synchronize`` before and after every measured call;
    the CPU path never touches ``torch.cuda`` at all. Each timing repeat
    checks the shared budget before it starts.
    """
    import torch

    from models.polar_fusion import torch_device_name

    timing_config = config["detection"]["timing"]
    if not timing_config.get("enabled", False):
        return {"enabled": False}
    warmup = int(timing_config.get("warmup", 5))
    repeats = int(timing_config.get("repeats", 30))
    frame = frames[len(frames) // 2]
    normalized = torch_device_name(str(device))
    use_cuda = normalized.startswith("cuda")
    report = progress if progress is not None else {}
    report.update(enabled=True, device=device, device_normalized=normalized,
                  synchronized=use_cuda, frame_id=frame["frame_id"],
                  status="RUNNING", conditions={})
    results = report["conditions"]

    def check(at):
        report["stage"] = at
        if budget is not None:
            budget.check(at)

    for condition in plan:
        check(f"timing condition {condition['name']} start")
        row = {
            "imgsz": condition["imgsz"], "conf": condition["conf"],
            "warmup": warmup, "repeats": repeats,
            "completed_warmup": 0, "completed_repeats": 0,
            "samples_ms": [], "p50_ms": None, "p95_ms": None,
            "status": "PARTIAL", "device_normalized": normalized,
            "synchronized": use_cuda,
            "stage": "Model A predict only (preprocessing to parsed masks)",
            "note": "single stage on one frame; NOT the system FPS",
        }
        results[condition["name"]] = row
        for _ in range(warmup):
            check(f"timing warmup {condition['name']}")
            model.predict(frame["image"], conf=condition["conf"],
                          imgsz=condition["imgsz"])
            row["completed_warmup"] += 1
            check(f"timing warmup {condition['name']} completed")
        for _ in range(repeats):
            check(f"timing repeat {condition['name']}")
            if use_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            model.predict(frame["image"], conf=condition["conf"],
                          imgsz=condition["imgsz"])
            if use_cuda:
                torch.cuda.synchronize()
            row["samples_ms"].append((time.perf_counter() - started) * 1000.0)
            row["completed_repeats"] += 1
            check(f"timing repeat {condition['name']} completed")
        array = np.asarray(row["samples_ms"])
        row.update(p50_ms=float(np.percentile(array, 50)),
                   p95_ms=float(np.percentile(array, 95)), status="COMPLETED")
    report["status"] = "COMPLETED"
    return report


def probe_ultralytics_identity(device: str) -> dict:
    """The installed Ultralytics version, probed in an isolated subprocess.

    Importing ultralytics in-process globally patches ``cv2.imread``
    (grayscale reads start returning (H, W, 1)), which would silently
    corrupt any later mask construction in the same interpreter. A short
    subprocess reads the version without that side effect; a probe failure
    is recorded as an explicit error, never swallowed.
    """
    import subprocess as subprocess_module

    probe_lines = [
        "import json, torch",
        "out = {'torch_version': torch.__version__}",
        "try:",
        "    import ultralytics",
        "    out['ultralytics_version'] = ultralytics.__version__",
        "except Exception as error:",
        "    out['ultralytics_version_error'] = "
        "f'{type(error).__name__}: {error}'",
        "print(json.dumps(out))",
    ]
    code = "\n".join(probe_lines)
    try:
        completed = subprocess_module.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            timeout=120, cwd=str(PROJECT_ROOT),
        )
    except Exception as error:  # noqa: BLE001
        return {"device": device, "runtime_identity_error":
                f"probe launch failed: {type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "device": device,
            "runtime_identity_error":
                f"probe exited {completed.returncode}: "
                f"{completed.stderr.strip()[:300]}",
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        return {"device": device, "runtime_identity_error":
                f"probe output unreadable: {error}"}
    payload["device"] = device
    payload["postprocess_note"] = (
        "the conf/iou values are passed to the installed Ultralytics "
        "predict implementation; no NMS parameter is assumed to take "
        "effect beyond what the runtime does")
    return payload



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fixed-condition Model A detection diagnosis "
                    "(D0/D1/D2/D3; read-only).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--rebuild-atlas", default=None, metavar="EVIDENCE_DIR",
                        help="redraw the atlas for a SAVED diagnosis evidence "
                        "directory (no model call); --output-dir receives the "
                        "new atlas and manifest")
    args = parser.parse_args(argv)

    if args.rebuild_atlas is not None:
        if Path(args.output_dir).exists():
            print(
                f"refused: output directory {args.output_dir} already exists",
                file=sys.stderr,
            )
            return EXIT_REFUSED
        try:
            evidence_dir = Path(args.rebuild_atlas)
            if not evidence_dir.is_dir():
                raise FileNotFoundError(
                    f"evidence directory not found: {evidence_dir}")
            config = load_experiment_config(args.config)
            frames = real_frames_provider(config, PROJECT_ROOT)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True)
            report = rebuild_atlas_from_evidence(
                evidence_dir, frames, output_dir,
                controls_per_class=int(
                    config["detection"]["atlas"]["matched_controls_per_class"]),
                margin=int(config["detection"]["atlas"].get("margin", 48)),
            )
            write_json(output_dir / "atlas_manifest.json", report)
            print(f"atlas rebuilt: {output_dir} (status {report['status']}, "
                  f"{report['panels']} panels)")
            return EXIT_OK if report["status"] == "COMPLETED" else 2
        except Exception as error:  # noqa: BLE001
            print(f"failed: {type(error).__name__}: {error}", file=sys.stderr)
            return EXIT_REFUSED

    if Path(args.output_dir).exists():
        print(
            f"refused: output directory {args.output_dir} already exists; "
            "diagnosis evidence is never overwritten",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    started = time.perf_counter()

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        config = load_experiment_config(args.config)
        verify_baseline_identities(config, PROJECT_ROOT)
        historical = load_historical_d0(config, PROJECT_ROOT)
        plan = build_condition_plan(config)
        log(f"conditions: {[(c['name'], c['conf'], c['imgsz']) for c in plan]}")

        import torch

        from models.segmentation import SegmentationModel

        device = args.device

        def model_factory(path, conf, iou, imgsz, device_):
            adapter = _ConditionedModelA(
                PROJECT_ROOT / path, iou, device_)
            # the first D0 call happens inside the runner; record the
            # runtime identity here once
            return adapter

        frames = real_frames_provider(config, PROJECT_ROOT)
        log(f"frames: {len(frames)} ({config['detection']['split']} split)")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True)
        budget = BudgetWatch(
            float(config["detection"].get("max_runtime_minutes", 30)))
        summary = run_detection_diagnosis(
            config=config,
            output_dir=output_dir,
            frames_provider=lambda: frames,
            model_factory=model_factory,
            historical=historical,
            device=device,
            log=log,
            budget=budget,
        )

        # runtime identity of the actual post-processing implementation.
        # Ultralytics patches cv2.imread globally at import time, so the
        # version probe runs in a SUBPROCESS and never mutates this
        # process's cv2 (a failed probe is recorded, never silently
        # skipped).
        runtime = probe_ultralytics_identity(device)
        # On ANY non-COMPLETED status (D0 reproduction failure, budget
        # exhaustion, partial atlas) the timing forward passes do NOT run:
        # no further model is constructed and no condition is re-predicted.
        if summary["status"] == "COMPLETED":
            timing = {"enabled": bool(config["detection"]["timing"].get("enabled", False)),
                      "conditions": {}, "stage": "before timing model construction"}
            summary["diagnosis_status"] = "COMPLETED"
            try:
                budget.check(timing["stage"])
                if timing["enabled"]:
                    timing = measure_timing(
                        model_factory(config["baselines"]["model_a"]["path"],
                                      plan[0]["conf"], plan[0]["nms_iou"],
                                      plan[0]["imgsz"], device),
                        frames, plan, config, device, budget=budget,
                        progress=timing)
            except BudgetExceeded as error:
                summary["status"] = "PARTIAL_BUDGET_EXCEEDED"
                timing.update(status=summary["status"], reason=str(error))
            except Exception as error:  # preserve failure, not a budget event
                summary["status"] = "FAILED"
                timing.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
        else:
            timing = {
                "enabled": False,
                "skipped": summary["status"],
                "note": ("timing was not run because the diagnosis did not "
                         "complete; no further model forward was performed"),
            }
        write_json(output_dir / "runtime.json", runtime)
        write_json(output_dir / "timing.json", timing)
        summary["elapsed_s"] = time.perf_counter() - started
        summary["runtime"] = runtime
        write_json(output_dir / "summary.json", summary)
        log(f"summary: {output_dir / 'summary.json'}")
        if summary["status"] == "FAILED":
            return EXIT_REFUSED
        return EXIT_OK if summary["status"] == "COMPLETED" else 2
    except Exception as error:  # noqa: BLE001 - CLI boundary report
        print(f"failed: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
