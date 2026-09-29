"""Read-only closeout over a 4B evidence directory.

Re-derives the corrected reports from the *recorded* evidence — it loads no
checkpoint, builds no engine, runs no model forward, no SGBM and no
benchmark, and it never writes into the source directory.

What it corrects (each with the source digest recorded):

- the fallback summary, recomputed with the shared
  :func:`scripts.eval_material_pipeline.summarise_fallback` over exactly
  ``layer=B_model_a_predictions, condition=routed``;
- the alignment verdict, re-evaluated with the strict
  :func:`scripts.eval_material_pipeline.evaluate_alignment` for both the
  successful run and the preserved failed attempt;
- the unmatched-GT records, where a zero maximum overlap is now stated as
  ``no_positive_overlap`` instead of naming an arbitrary first prediction;
- the performance stage name (``crop_preprocess_cpu`` →
  ``cpu_tensor_preparation``); the timing values are untouched.

What it cannot do is re-run anything: fields the historical records never
stored cannot be re-verified, and those items are marked unverified rather
than reported as newly measured.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.material_fusion_dataset import CLASS_NAMES  # noqa: E402
from models.material_classifier import (  # noqa: E402
    load_material_runtime_config,
)
from scripts.eval_material_pipeline import (  # noqa: E402
    evaluate_alignment,
    summarise_fallback,
)

STAGE_RENAMES = {"crop_preprocess_cpu": "cpu_tensor_preparation"}
# Files that must exist, grouped by the evidence sub-directory that owns
# them. A manifest covers its own sub-directory's *data* files; the manifest
# files themselves are inputs, not entries of their own listing.
SOURCE_REQUIRED = (
    "quality/summary.json",
    "quality/protocol_snapshot.json",
    "quality/instances.jsonl",
    "quality/a2_alignment_summary.json",
    "quality/a2_alignment.jsonl",
    "quality/frames.jsonl",
    "quality/artifact_manifest.json",
    "performance/performance_report.json",
    "performance/raw_timings.json",
    "performance/artifact_manifest.json",
)
SOURCE_MANIFEST_DIRS = ("quality", "performance")
# Data files each manifest must cover, per sub-directory.
MANIFEST_MUST_COVER = {
    "quality": ("summary.json", "protocol_snapshot.json", "instances.jsonl",
                "a2_alignment_summary.json", "a2_alignment.jsonl",
                "frames.jsonl"),
    "performance": ("performance_report.json", "raw_timings.json"),
}
# Optional evidence that is validated against its own manifest when present.
FAILED_ATTEMPT_DIR = "quality_attempt1_alignment_failed"
FAILED_ATTEMPT_MUST_COVER = ("summary.json", "a2_alignment.jsonl")


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_output_dir(output_dir: str | Path, source_dir: str | Path) -> Path:
    path = Path(output_dir)
    resolved = path.resolve()
    source_resolved = Path(source_dir).resolve()
    if resolved == source_resolved or source_resolved in resolved.parents:
        raise ValueError(
            f"the output directory {resolved} is the source evidence "
            "directory or inside it; the source tree is read-only"
        )
    if path.exists():
        raise FileExistsError(
            f"output directory {path} already exists; closeout reports are "
            "never overwritten — pass a fresh --output-dir"
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().decode("utf-8")
            .splitlines() if line.strip()]


def _within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root.resolve() or root.resolve() in resolved.parents


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _verify_manifest(sub_dir: Path, *, root: Path, must_cover) -> dict:
    """Validate one evidence sub-directory's manifest and return its digests."""
    manifest_path = sub_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(
            f"{manifest_path}: 'files' must be a non-empty mapping, got "
            f"{type(files).__name__}"
        )
    covered: dict[str, str] = {}
    for name, recorded in files.items():
        if not isinstance(name, str) or not isinstance(recorded, str):
            raise ValueError(
                f"{manifest_path}: entries must be string pairs, got "
                f"{name!r}: {recorded!r}"
            )
        if not _HEX64.match(recorded):
            raise ValueError(
                f"{manifest_path}: digest for {name!r} is not a 64-hex sha256"
            )
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(
                f"{manifest_path}: {name!r} is an absolute or escaping path"
            )
        target = sub_dir / name
        if not _within(target, sub_dir):
            raise ValueError(
                f"{manifest_path}: {name!r} resolves outside {sub_dir.name}/"
            )
        if not target.is_file():
            raise ValueError(f"{manifest_path}: {name!r} is recorded but missing")
        if sha256_of(target) != recorded:
            raise ValueError(f"{manifest_path}: digest mismatch for {name!r}")
        covered[name] = recorded
    missing = [name for name in must_cover if name not in covered]
    if missing:
        raise ValueError(
            f"{manifest_path}: consumed files are not covered by any digest: "
            f"{missing}"
        )
    return {
        "manifest": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256_of(manifest_path),
        "covered": sorted(covered),
    }


def verify_source(source_dir: Path) -> dict:
    """Admission over the source evidence before anything is read.

    Each evidence sub-directory owns its own manifest, and a manifest lists
    its data files — never itself. The failed first attempt is validated
    against its own manifest when it exists. Every consumed file must be
    covered with a matching digest; absolute, escaping and out-of-root
    paths are refused.
    """
    missing = [name for name in SOURCE_REQUIRED
               if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source evidence is missing: {missing}")
    manifests: dict[str, dict] = {}
    for sub in SOURCE_MANIFEST_DIRS:
        manifests[sub] = _verify_manifest(
            source_dir / sub, root=source_dir, must_cover=MANIFEST_MUST_COVER[sub]
        )
    failed_dir = source_dir / FAILED_ATTEMPT_DIR
    if failed_dir.is_dir():
        manifests[FAILED_ATTEMPT_DIR] = _verify_manifest(
            failed_dir, root=source_dir, must_cover=FAILED_ATTEMPT_MUST_COVER
        )
    return {
        "source_dir": str(source_dir.resolve()),
        "verified_files": {
            name: sha256_of(source_dir / name) for name in SOURCE_REQUIRED
        },
        "manifests": manifests,
        "covered_by_manifest": sorted(
            f"{sub}/{name}"
            for sub, entry in manifests.items() for name in entry["covered"]
        ),
        "manifests_verified": [entry["manifest"] for entry in manifests.values()],
        "manifest_note": "a manifest lists its data files and never itself; "
                         "its own digest is recorded here",
    }


def expected_keys_from_sources(gt_index, *, npz_keys):
    """The expected alignment set from the independent GT index.

    Returns ``(expected_keys, gt_without_npz)``: both are sorted lists of
    ``(frame_id, object_index)``. The set comes from the frozen GT index
    (every annotated instance of every val frame), never from the A2 or
    alignment outputs under review; an object without an NPZ stays in the
    set and is additionally listed in ``gt_without_npz``.
    """
    problems: list[str] = []
    keys: list[tuple[str, int]] = []
    for frame in gt_index:
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            problems.append(f"GT index frame has no usable frame_id: {frame!r}")
            continue
        instances = frame.get("instances")
        if not isinstance(instances, list):
            problems.append(f"{frame_id}: 'instances' must be a list")
            continue
        for instance in instances:
            index = instance.get("object_index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                problems.append(
                    f"{frame_id}: illegal object_index {index!r}"
                )
                continue
            keys.append((frame_id, index))
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        problems.append(f"duplicate GT keys: {duplicates[:5]}")
    if problems:
        raise ValueError("GT index admission failed: " + "; ".join(problems[:8]))
    expected = sorted(set(keys))
    npz_set = {(str(frame), int(index)) for frame, index in npz_keys}
    without_npz = [key for key in expected if key not in npz_set]
    extra_npz = sorted(npz_set - set(expected))
    if extra_npz:
        raise ValueError(
            f"the V5 manifest maps {len(extra_npz)} NPZ samples to GT objects "
            f"that the GT index does not contain (first: {extra_npz[:5]})"
        )
    return expected, without_npz


METRIC_BLOCK_REQUIRED = {
    "counts": ("frames", "failed_frames", "evaluated_frames", "gt_instances",
               "predictions", "matched", "missed_gt",
               "unmatched_predictions", "unclassified_matched"),
    "detection": ("precision", "recall", "f1", "tp", "fp", "fn"),
    "conditional_classification": ("accuracy", "macro_f1", "samples"),
    "end_to_end": ("macro_f1",),
}
BOOTSTRAP_REQUIRED = ("point_estimate", "ci_low", "ci_high")
NOT_ESTIMABLE = ("NOT_ESTIMABLE", "NOT_MEASURED", "UNVERIFIED")


def _finite(value) -> bool:
    import math
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def validate_metric_block(block) -> list[str]:
    """The minimum production structure of one metric block.

    A dictionary that merely exists is not a result: the required sections
    and their fields must be present, counts must be non-negative integers,
    the rate fields must be finite and the supports must add up.
    """
    problems: list[str] = []
    if not isinstance(block, dict) or not block:
        return ["the block is missing or empty"]
    for section, fields in METRIC_BLOCK_REQUIRED.items():
        value = block.get(section)
        if not isinstance(value, dict) or not value:
            problems.append(f"{section}: missing or empty")
            continue
        missing = [field for field in fields if field not in value]
        if missing:
            problems.append(f"{section}: missing fields {missing}")
    counts = block.get("counts")
    if isinstance(counts, dict):
        for field, value in counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(f"counts.{field}: not a non-negative integer")
        matched = counts.get("matched")
        gt = counts.get("gt_instances")
        preds = counts.get("predictions")
        if all(isinstance(v, int) for v in (matched, gt, preds)):
            if matched > gt:
                problems.append("counts: matched exceeds gt_instances")
            if matched > preds:
                problems.append("counts: matched exceeds predictions")
    for section, fields in (("detection", ("precision", "recall", "f1")),
                            ("conditional_classification", ("accuracy",
                                                            "macro_f1")),
                            ("end_to_end", ("macro_f1",))):
        values = block.get(section)
        if isinstance(values, dict):
            for field in fields:
                if field not in values:
                    continue
                value = values[field]
                if not _finite(value):
                    problems.append(f"{section}.{field}: not a finite number")
                elif not 0.0 <= float(value) <= 1.0:
                    problems.append(
                        f"{section}.{field}: {value!r} is outside [0, 1]"
                    )
    # Integer supports and the conservation relations the report is built on.
    for section, fields in (("detection", ("tp", "fp", "fn")),
                            ("conditional_classification", ("samples",))):
        values = block.get(section)
        if isinstance(values, dict):
            for field in fields:
                value = values.get(field)
                if field in values and (not isinstance(value, int)
                                        or isinstance(value, bool)
                                        or value < 0):
                    problems.append(f"{section}.{field}: not a non-negative "
                                    "integer")
    counts = block.get("counts") if isinstance(block.get("counts"), dict) else {}
    detection = (block.get("detection")
                 if isinstance(block.get("detection"), dict) else {})
    conditional = (block.get("conditional_classification")
                   if isinstance(block.get("conditional_classification"), dict)
                   else {})
    def _ints(*values):
        return all(isinstance(v, int) and not isinstance(v, bool)
                   for v in values)
    if _ints(counts.get("matched"), counts.get("missed_gt"),
             counts.get("gt_instances")):
        if counts["matched"] + counts["missed_gt"] != counts["gt_instances"]:
            problems.append(
                "counts: matched + missed_gt != gt_instances "
                f"({counts['matched']} + {counts['missed_gt']} != "
                f"{counts['gt_instances']})"
            )
    if _ints(counts.get("matched"), counts.get("unmatched_predictions"),
             counts.get("predictions")):
        if (counts["matched"] + counts["unmatched_predictions"]
                != counts["predictions"]):
            problems.append(
                "counts: matched + unmatched_predictions != predictions"
            )
    if _ints(counts.get("matched"), detection.get("tp")):
        if detection["tp"] > counts["matched"]:
            problems.append("detection.tp exceeds counts.matched")
    if _ints(conditional.get("samples"), counts.get("matched")):
        if conditional["samples"] > counts["matched"]:
            problems.append(
                "conditional_classification.samples exceeds counts.matched"
            )
    return problems


def validate_bootstrap_entry(entry) -> list[str]:
    """One seed's bootstrap result must carry a real statistical outcome."""
    if not isinstance(entry, dict) or not entry:
        return ["missing or empty result"]
    status = entry.get("status")
    if status in NOT_ESTIMABLE:
        return []          # an explicit not-estimable state is allowed
    missing = [field for field in BOOTSTRAP_REQUIRED if field not in entry]
    if missing:
        return [f"missing statistical fields {missing} and no explicit "
                f"{NOT_ESTIMABLE} status"]
    problems = [f"{field}: not a finite number"
                for field in BOOTSTRAP_REQUIRED if not _finite(entry[field])]
    if _finite(entry.get("ci_low")) and _finite(entry.get("ci_high")):
        if entry["ci_low"] > entry["ci_high"]:
            problems.append("the interval is inverted")
    return problems


ALLOWED_LAYERS = ("A2_gt_controlled", "B_model_a_predictions",
                  "A1_offline_npz")
ALLOWED_CONDITIONS = ("pure", "routed")


def validate_instance_identities(records) -> list[str]:
    """Validate the B instance id before anything keys or converts it.

    Only ``B_model_a_predictions`` records are covered: their
    ``pred_instance_id`` must exist and be a real integer (``bool`` is not
    an integer, and no ``int()`` coercion may accept a string or a float).
    An A2 record legitimately carries ``pred_instance_id=None``, so A2 is
    deliberately untouched here.
    """
    problems: list[str] = []
    for row in records:
        if row.get("layer") != "B_model_a_predictions":
            continue
        if row.get("condition") not in ALLOWED_CONDITIONS:
            continue
        where = (f"variant={row.get('variant')!r} seed={row.get('seed')!r} "
                 f"frame_id={row.get('frame_id')!r} "
                 f"condition={row.get('condition')!r}")
        if "pred_instance_id" not in row:
            problems.append(f"B record ({where}) is missing pred_instance_id")
            continue
        value = row["pred_instance_id"]
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(
                f"B record ({where}) has an illegal pred_instance_id "
                f"{value!r} of type {type(value).__name__}; it must be a "
                "non-negative integer"
            )
        elif value < 0:
            problems.append(
                f"B record ({where}) has a negative pred_instance_id {value}"
            )
    return problems


def validate_frame_evidence(rows, *, expected_frame_ids) -> dict:
    """Validate ``frames.jsonl`` before it is turned into a prediction map.

    The production writer emits one record per evaluated frame with a
    non-empty string ``frame_id`` and an integer ``predictions`` count, so
    this validator requires exactly that: no ``str()``/``int()`` coercion of
    an illegal identity, no dictionary overwrite of a repeated frame, no
    silent skip of a record whose count is missing or of the wrong type, and
    no assumption that an absent frame means zero predictions. The expected
    frame set comes from the bound GT index (the frozen evaluation scope),
    never from this file.
    """
    failures: list[str] = []
    partials: list[str] = []
    positions: dict[str, int] = {}
    predictions: dict[str, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(
                f"frame record #{position} is not a JSON object "
                f"(got {type(row).__name__}); refusing to interpret it"
            )
            continue
        if "frame_id" not in row:
            failures.append(f"frame record #{position} has no frame_id")
            continue
        frame_id = row["frame_id"]
        if not isinstance(frame_id, str) or not frame_id.strip():
            failures.append(
                f"frame record #{position} has an illegal frame_id "
                f"{frame_id!r} of type {type(frame_id).__name__}; a non-empty "
                "string is required"
            )
            continue
        if "predictions" not in row:
            failures.append(f"frame {frame_id!r}: predictions is missing")
            continue
        value = row["predictions"]
        if type(value) is not int or value < 0:
            failures.append(
                f"frame {frame_id!r}: illegal predictions {value!r} of type "
                f"{type(value).__name__}; a non-negative integer is required"
            )
            continue
        if frame_id in positions:
            failures.append(
                f"duplicate frame record for {frame_id!r} (positions "
                f"{positions[frame_id]} and {position})"
            )
            continue
        positions[frame_id] = position
        predictions[frame_id] = value

    expected = list(expected_frame_ids)
    expected_set = set(expected)
    extra = sorted(set(positions) - expected_set)
    missing = sorted(expected_set - set(positions))
    if extra:
        failures.append(
            f"frame evidence contains {len(extra)} frame(s) outside the "
            f"frozen evaluation scope: {extra[:5]}"
        )
    if missing:
        message = (
            f"frame evidence is missing {len(missing)} expected frame(s): "
            f"{missing[:5]}"
        )
        if failures:
            failures.append(message)      # a contradiction outranks a gap
        else:
            partials.append(message)
    status = "FAILED" if failures else ("PARTIAL" if partials else "PASSED")
    return {
        "status": status,
        "failures": failures[:20],
        "partials": partials[:20],
        "records": len(rows),
        "expected_frames": len(expected),
        "missing_frames": missing[:20],
        "extra_frames": extra[:20],
        "frame_predictions": predictions if status != "FAILED" else {},
        "note": "an absent frame is a coverage gap, never an implicit "
                "zero-prediction frame",
    }


def verify_instance_coverage(records, *, candidates, expected_a2_keys,
                             frame_predictions=None,
                             expected_frame_ids=None,
                             summary_supports=None) -> dict:
    """Per-candidate, per-layer, per-condition record coverage.

    A2 records must correspond exactly to the bound GT index; B records must
    exist for every candidate and condition (a candidate whose summary
    reports predictions but has no records is a contradiction, not a zero).
    Duplicates, extra records and identity conflicts are failures; missing
    evidence is at best PARTIAL.
    """
    failures: list[str] = []
    partials: list[str] = []
    groups: dict[tuple, list[dict]] = {}
    seen: set[tuple] = set()
    allowed_candidates = {(str(v), int(s)) for v, s in candidates}
    for row in records:
        try:
            key = (str(row.get("variant")), int(row.get("seed")),
                   row.get("layer"), row.get("condition"))
        except (TypeError, ValueError):
            failures.append(f"instance record has an illegal identity: {row!r}")
            continue
        if key[:2] not in allowed_candidates:
            failures.append(f"instance record for an unknown candidate {key[:2]}")
            continue
        if key[2] not in ALLOWED_LAYERS:
            failures.append(f"instance record for an unknown layer {key[2]!r}")
            continue
        if key[3] not in ALLOWED_CONDITIONS:
            failures.append(
                f"instance record for an unknown condition {key[3]!r}"
            )
            continue
        for field in ("variant", "seed", "frame_id"):
            if field not in row:
                failures.append(f"instance record is missing {field!r}: {row!r}")
        groups.setdefault(key, []).append(row)
        if key[2] == "A1_offline_npz":
            continue          # a legal layer, but not part of A2/B coverage
        # Illegal (unhashable or non-integer) identity values are already
        # reported by validate_instance_identities; repr() keeps the
        # duplicate bookkeeping hashable instead of raising TypeError here.
        def _hashable(value):
            return value if isinstance(value, (int, str, type(None))) else repr(value)
        if key[2] == "B_model_a_predictions":
            unique = (key, row.get("frame_id"),
                      _hashable(row.get("pred_instance_id")))
        else:
            unique = (key, row.get("frame_id"),
                      _hashable(row.get("gt_instance_id")))
        if unique in seen:
            failures.append(f"duplicate instance record {unique}")
        seen.add(unique)
    expected = {(str(frame), int(index)) for frame, index in expected_a2_keys}
    expected_predictions = frame_predictions or {}
    expected_frames = (set(str(frame) for frame in expected_frame_ids)
                       if expected_frame_ids is not None else None)
    for variant, seed in candidates:
        for layer in ("A2_gt_controlled", "B_model_a_predictions"):
            for condition in ALLOWED_CONDITIONS:
                key = (str(variant), int(seed), layer, condition)
                rows = groups.get(key) or []
                if layer == "A2_gt_controlled":
                    if not rows:
                        partials.append(f"{key}: no instance records")
                        continue
                    actual = {(str(row["frame_id"]), int(row["gt_instance_id"]))
                              for row in rows if row.get("gt_instance_id")
                              is not None}
                    missing = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    if missing:
                        partials.append(f"{key}: missing GT objects {missing[:5]}")
                    if extra:
                        failures.append(f"{key}: unexpected GT objects {extra[:5]}")
                    continue
                # B: the expected identity set comes from the frame-level
                # prediction evidence. The production entry numbers a
                # frame's detections 0..predictions-1
                # (eval_material_pipeline enumerates the routed results), so
                # the identity is (frame_id, index) and an equal-count
                # substitution with an unknown frame or an out-of-range
                # index is a contradiction, not a coverage gap.
                expected_identities = {
                    (str(frame), index)
                    for frame, count in expected_predictions.items()
                    for index in range(int(count))
                }
                actual_identities: set[tuple[str, int]] = set()
                for row in rows:
                    value = row.get("pred_instance_id")
                    if not isinstance(value, int) or isinstance(value, bool):
                        failures.append(
                            f"{key}: illegal pred_instance_id {value!r} in a "
                            "record"
                        )
                        continue
                    if value < 0:
                        failures.append(
                            f"{key}: negative pred_instance_id {value} in a "
                            "record"
                        )
                        continue
                    actual_identities.add((str(row.get("frame_id")), value))
                if len(actual_identities) != len(rows):
                    failures.append(
                        f"{key}: duplicate instance identities within the "
                        "same candidate and condition"
                    )
                # (a) records on frames outside the frozen evaluation scope.
                if expected_frames is not None:
                    outside_scope = sorted(
                        {frame for frame, _index in actual_identities
                         if frame not in expected_frames}
                    )
                    if outside_scope:
                        failures.append(
                            f"{key}: instance records on frames outside the "
                            "frozen evaluation scope: "
                            f"{outside_scope[:5]}"
                        )
                # (b) expected frames whose prediction evidence is missing:
                # their count is unknown, so their coverage is unverifiable —
                # never read as zero predictions and never reported as a
                # wrong identity.
                if expected_frames is not None:
                    without_evidence = sorted(
                        frame for frame in expected_frames
                        if frame not in expected_predictions
                    )
                    if without_evidence:
                        partials.append(
                            f"{key}: {len(without_evidence)} frame(s) have no "
                            "prediction evidence, so their instance coverage "
                            f"cannot be verified: {without_evidence[:5]}"
                        )
                # (c) an explicit zero-prediction frame must have no records.
                zero_prediction_frames = {
                    str(frame) for frame, count in expected_predictions.items()
                    if int(count) == 0
                }
                on_zero_prediction_frames = sorted(
                    identity for identity in actual_identities
                    if identity[0] in zero_prediction_frames
                )
                if on_zero_prediction_frames:
                    failures.append(
                        f"{key}: {len(on_zero_prediction_frames)} record(s) "
                        "for frames whose evidence reports zero predictions: "
                        f"{on_zero_prediction_frames[:5]}"
                    )
                # (d) only frames with evidence can be range-checked.
                known_actual = {
                    identity for identity in actual_identities
                    if identity[0] in expected_predictions
                }
                extra = sorted(known_actual - expected_identities)
                missing = sorted(expected_identities - known_actual)
                if extra:
                    failures.append(
                        f"{key}: unknown or out-of-range instance identities "
                        f"{extra[:5]}"
                    )
                if missing:
                    partials.append(
                        f"{key}: missing predicted instances {missing[:5]}"
                    )
                if summary_supports is not None:
                    supported = summary_supports.get(f"{variant}_s{seed}")
                    if isinstance(supported, dict):
                        supported = supported.get(condition)
                    if type(supported) is not int or supported < 0:
                        failures.append(
                            f"{key}: summary prediction support must be a "
                            "non-negative integer for this condition"
                        )
                    else:
                        # A subtotal is not comparable to a full-run support.
                        # Only the independent scope can establish completeness.
                        if (expected_frames is None
                                or set(expected_predictions) != expected_frames):
                            partials.append(
                                f"{key}: the summary prediction total cannot be "
                                "verified without complete frame evidence and "
                                "an independent expected frame scope"
                            )
                        else:
                            expected_count = len(expected_identities)
                            if supported != expected_count:
                                failures.append(
                                    f"{key}: the summary reports {supported} "
                                    "predictions but the frame evidence reports "
                                    f"{expected_count}"
                                )
    status = "FAILED" if failures else ("PARTIAL" if partials else "PASSED")
    return {"status": status, "failures": failures[:20],
            "partials": partials[:20], "groups": len(groups),
            "records": len(records),
            "expected_b_instances": sum(int(c) for c
                                        in (frame_predictions or {}).values())}


def verify_candidates(candidates, *, expected_shas) -> dict:
    """The frozen 4-variant x 3-seed matrix, checked against the snapshot."""
    variants = ("gray", "gray_gray_valid", "gray_valid", "gray_polar")
    seeds = (2026, 2027, 2028)
    wanted = {(variant, seed) for variant in variants for seed in seeds}
    seen: list[tuple[str, int]] = []
    problems: list[str] = []
    for candidate in candidates:
        variant = candidate.get("variant")
        seed = candidate.get("seed")
        if not isinstance(variant, str) or not isinstance(seed, int):
            problems.append(f"illegal candidate record {candidate!r}")
            continue
        key = (variant, seed)
        seen.append(key)
        expected = expected_shas.get(key)
        if expected is None:
            problems.append(f"candidate {key} is not in the frozen identity list")
            continue
        if not _HEX64.match(str(candidate.get("sha256"))):
            problems.append(f"candidate {key} has no valid sha256")
            continue
        if str(candidate["sha256"]) != expected:
            problems.append(
                f"candidate {key} sha256 does not match the recorded identity"
            )
    duplicates = sorted({key for key in seen if seen.count(key) > 1})
    if duplicates:
        problems.append(f"duplicate candidate identities: {duplicates}")
    missing = sorted(wanted - set(seen))
    extra = sorted(set(seen) - wanted)
    if missing:
        problems.append(f"missing candidates: {missing}")
    if extra:
        problems.append(f"unexpected candidates: {extra}")
    if problems:
        raise ValueError("candidate identity check failed: "
                         + "; ".join(problems[:8]))
    return {
        "candidates": len(seen),
        "variants": list(variants),
        "seeds": list(seeds),
        "identities_verified": sorted(f"{variant}_s{seed}"
                                      for variant, seed in seen),
        "sha256_verified": len(seen),
    }


def _diff_paths(source, corrected, path="") -> list[str]:
    """Recursive difference paths between two report structures."""
    if isinstance(source, dict) and isinstance(corrected, dict):
        diffs = []
        for key in sorted(set(source) | set(corrected)):
            child = f"{path}.{key}" if path else str(key)
            if key not in source:
                diffs.append(f"{child}: unexpected in the corrected report")
            elif key not in corrected:
                diffs.append(f"{child}: missing from the corrected report")
            else:
                diffs.extend(_diff_paths(source[key], corrected[key], child))
        return diffs
    if isinstance(source, list) and isinstance(corrected, list):
        if len(source) != len(corrected):
            return [f"{path}: length {len(source)} -> {len(corrected)}"]
        diffs = []
        for index, (left, right) in enumerate(zip(source, corrected)):
            diffs.extend(_diff_paths(left, right, f"{path}[{index}]"))
        return diffs
    if source != corrected:
        return [f"{path}: {source!r} -> {corrected!r}"]
    return []


REQUIRED_VIEWS = ("pure", "routed")
REQUIRED_CONDITIONS = ("layer_a2", "layer_b")
BOOTSTRAP_SEEDS = ("seed_2026", "seed_2027", "seed_2028")


def compare_source_and_corrected(source_summary, corrected_summary, *,
                                 source_timings, corrected_timings,
                                 source_performance=None,
                                 corrected_performance=None,
                                 required_candidates=None) -> dict:
    """Compare the corrected report against the source it claims to copy.

    Every per-candidate metric block (12 candidates x A2/B x pure/routed =
    48 blocks), the bootstrap section, the paired changes, the conclusions,
    the performance report statistics and the timing arrays are compared in
    both directions. In the timing arrays only the approved stage rename is
    permitted; scenario/variant/field sets, order, values and rounding must
    not move. An empty input is never a pass.
    """
    mismatches: list[str] = []
    partials: list[str] = []
    source_candidates = source_summary.get("per_candidate") or {}
    corrected_candidates = corrected_summary.get("per_candidate") or {}
    if not source_candidates:
        mismatches.append("per_candidate: the source has no metric blocks")
    if required_candidates is not None:
        # The identity set comes from the admitted config, never from the
        # source agreeing with itself.
        wanted = set(required_candidates)
        missing = sorted(wanted - set(source_candidates))
        extra = sorted(set(source_candidates) - wanted)
        if missing:
            mismatches.append(f"per_candidate: missing candidates {missing}")
        if extra:
            mismatches.append(f"per_candidate: unexpected candidates {extra}")
    if set(source_candidates) != set(corrected_candidates):
        mismatches.extend(_diff_paths(source_candidates, corrected_candidates,
                                      "per_candidate"))
    compared_blocks = 0
    for tag in sorted(set(source_candidates) & set(corrected_candidates)):
        for condition in REQUIRED_CONDITIONS:
            block = source_candidates[tag].get(condition)
            if not isinstance(block, dict) or not block:
                mismatches.append(
                    f"{tag}.{condition}: the source block is missing or empty"
                )
                continue
            for view in REQUIRED_VIEWS:
                source_view = block.get(view)
                corrected_view = (corrected_candidates[tag].get(condition)
                                  or {}).get(view)
                structure = validate_metric_block(source_view)
                if structure:
                    mismatches.append(
                        f"{tag}.{condition}.{view}: the source block does not "
                        f"meet the report structure: {structure[:3]}"
                    )
                    continue
                corrected_structure = validate_metric_block(corrected_view)
                if corrected_structure:
                    mismatches.append(
                        f"{tag}.{condition}.{view}: the corrected block does "
                        f"not meet the report structure: "
                        f"{corrected_structure[:3]}"
                    )
                    continue
                compared_blocks += 1
                mismatches.extend(_diff_paths(
                    source_view, corrected_view, f"{tag}.{condition}.{view}"))
        if "layer_a1" in source_candidates[tag] or "layer_a1" in (
            corrected_candidates[tag] or {}
        ):
            mismatches.extend(_diff_paths(
                source_candidates[tag].get("layer_a1"),
                corrected_candidates[tag].get("layer_a1"),
                f"{tag}.layer_a1 (A1, not part of the 48)",
            ))
    for section in ("bootstrap", "paired_changes", "conclusions"):
        mismatches.extend(_diff_paths(source_summary.get(section),
                                      corrected_summary.get(section), section))
    for owner, payload in (("source", source_summary),
                           ("corrected", corrected_summary)):
        paired = payload.get("paired_changes")
        if (not isinstance(paired, list) or not paired
                or any(not isinstance(row, dict) or not row
                       for row in paired)):
            mismatches.append(f"{owner}.paired_changes: required non-empty "
                              "list of result objects")
        conclusions = payload.get("conclusions")
        if (not isinstance(conclusions, dict) or not conclusions
                or any(not isinstance(key, str) or not key.strip()
                       or not isinstance(value, str) or not value.strip()
                       for key, value in conclusions.items())):
            mismatches.append(f"{owner}.conclusions: required non-empty "
                              "mapping of named conclusions")
    bootstrap = source_summary.get("bootstrap")
    macro = (bootstrap.get("end_to_end_macro_f1_e4_vs_e1")
             if isinstance(bootstrap, dict) else None)
    if not isinstance(macro, dict):
        mismatches.append("bootstrap: missing or invalid seed result mapping")
        macro = {}
    seeds = sorted(macro)
    if seeds != sorted(BOOTSTRAP_SEEDS):
        mismatches.append(
            f"bootstrap: expected {list(BOOTSTRAP_SEEDS)}, found {seeds}"
        )
    for seed in BOOTSTRAP_SEEDS:
        entry = macro.get(seed)
        for problem in validate_bootstrap_entry(entry):
            mismatches.append(f"bootstrap.{seed}: {problem}")
        if isinstance(entry, dict) and entry.get("status") in NOT_ESTIMABLE:
            partials.append(f"bootstrap.{seed}: {entry['status']}; "
                            "the statistical result cannot be verified")
    if source_performance is not None or corrected_performance is not None:
        mismatches.extend(_diff_paths(source_performance, corrected_performance,
                                      "performance_report"))
    timing_mismatches: list[str] = []
    source_raw = (source_timings or {}).get("raw") or {}
    corrected_raw = (corrected_timings or {}).get("raw") or {}
    if not source_raw:
        timing_mismatches.append("raw: the source has no timing arrays")
    for bucket in sorted(set(source_raw) | set(corrected_raw)):
        variants = source_raw.get(bucket)
        corrected_variants = corrected_raw.get(bucket)
        if variants is None:
            timing_mismatches.append(f"raw.{bucket}: added in the corrected data")
            continue
        if corrected_variants is None:
            timing_mismatches.append(f"raw.{bucket}: missing in corrected data")
            continue
        if not isinstance(variants, dict) or not isinstance(corrected_variants,
                                                            dict):
            if variants != corrected_variants:
                timing_mismatches.append(f"raw.{bucket}: changed")
            continue
        for variant in sorted(set(variants) | set(corrected_variants)):
            entry = variants.get(variant)
            corrected_entry = corrected_variants.get(variant)
            if entry is None:
                timing_mismatches.append(
                    f"raw.{bucket}.{variant}: added in the corrected data")
                continue
            if corrected_entry is None:
                timing_mismatches.append(
                    f"raw.{bucket}.{variant}: missing in corrected data")
                continue
            if not isinstance(entry, dict) or not isinstance(corrected_entry,
                                                             dict):
                if entry != corrected_entry:
                    timing_mismatches.append(
                        f"raw.{bucket}.{variant}: changed")
                continue
            for field in sorted(set(entry) | set(corrected_entry)):
                if field == "stages":
                    continue
                if field not in entry:
                    timing_mismatches.append(
                        f"raw.{bucket}.{variant}.{field}: added")
                elif field not in corrected_entry:
                    timing_mismatches.append(
                        f"raw.{bucket}.{variant}.{field}: missing")
                elif entry[field] != corrected_entry[field]:
                    timing_mismatches.append(
                        f"raw.{bucket}.{variant}.{field}: changed")
            source_stages = entry.get("stages")
            corrected_stages = corrected_entry.get("stages")
            if not isinstance(source_stages, dict) or not isinstance(
                corrected_stages, dict
            ):
                if source_stages != corrected_stages:
                    timing_mismatches.append(
                        f"raw.{bucket}.{variant}.stages: changed")
                continue
            if set(corrected_stages) != {
                STAGE_RENAMES.get(name, name) for name in source_stages
            }:
                timing_mismatches.append(
                    f"raw.{bucket}.{variant}.stages: stage set changed beyond "
                    "the approved rename")
                continue
            for name, values in source_stages.items():
                renamed = STAGE_RENAMES.get(name, name)
                if values != corrected_stages[renamed]:
                    timing_mismatches.append(
                        f"raw.{bucket}.{variant}.stages.{renamed}: values "
                        "changed")
    mismatches.extend(timing_mismatches)
    return {
        "status": "FAILED" if mismatches else ("PARTIAL" if partials else "PASSED"),
        "partials": partials,
        "compared_metric_blocks": compared_blocks,
        "compared_candidates": len(set(source_candidates)
                                   & set(corrected_candidates)),
        "bootstrap_seeds": seeds,
        "mismatches": mismatches,
        "timing_mismatches": timing_mismatches,
        "timings_compared": bool(source_raw),
        "renamed_stages": sorted(STAGE_RENAMES),
        "note": "computed from the actual source and corrected objects; the "
                "approved stage rename is the only permitted timing change",
    }


def expected_alignment_keys(instances: Sequence[dict]) -> list[tuple[str, int]]:
    """The GT correspondence, taken from the recorded A2 instance rows.

    The expected set comes from the recorded GT-controlled instances, not
    from the alignment rows themselves, so the verdict cannot be satisfied
    by however many rows the alignment happened to produce.
    """
    keys = {
        (str(row["frame_id"]), int(row["gt_instance_id"]))
        for row in instances
        if row.get("layer") == "A2_gt_controlled"
        and row.get("condition") == "routed"
        and row.get("gt_instance_id") is not None
    }
    return sorted(keys)


def recheck_alignment(rows: Sequence[dict], keys) -> dict:
    """Re-evaluate recorded alignment rows under the strict verdict.

    The historical rows carry status/exact_equal/max_abs_err but not the
    shape, dtype and tolerance metadata the strict contract now requires,
    so checks that cannot be re-verified are reported as unverified rather
    than as newly measured passes.
    """
    verdict = evaluate_alignment(rows, expected_keys=keys)
    unverifiable = [
        "shape and dtype per field (not stored in the historical rows)",
        "exact-equality recomputation for signed_q/abs_q (only the recorded "
        "exact_equal/max_abs_err verdicts exist)",
        "the recorded values were produced before this round's strict "
        "contract; the recheck cannot upgrade them",
    ]
    verdict["recheck_note"] = (
        "this is a re-evaluation of the *recorded* verdicts under the strict "
        "rules; it is not a new inference and it cannot verify what the "
        "historical records never stored"
    )
    verdict["unverifiable_items"] = unverifiable
    return verdict


def correct_unmatched_gt(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """State zero overlap explicitly instead of naming an arbitrary index."""
    corrected = []
    zero_overlap = 0
    positive_and_matched_elsewhere = 0
    for row in rows:
        entry = dict(row)
        max_iou = entry.get("max_mask_iou")
        predictions = int(entry.get("predictions_in_frame") or 0)
        if predictions == 0:
            entry["best_prediction_index"] = None
            entry["best_prediction_matched_elsewhere"] = None
            entry["no_positive_overlap"] = None
            entry["reason"] = "no predictions in the frame"
        elif max_iou is None or float(max_iou) <= 0.0:
            entry["max_mask_iou"] = 0.0
            entry["best_prediction_index"] = None
            entry["best_prediction_matched_elsewhere"] = None
            entry["no_positive_overlap"] = True
            entry["reason"] = "no positive overlap with any prediction"
            zero_overlap += 1
        else:
            entry["no_positive_overlap"] = False
            if entry.get("best_prediction_matched_elsewhere"):
                positive_and_matched_elsewhere += 1
        entry["correction_note"] = (
            "zero overlap is not evidence of a merge or duplicate; no root "
            "cause is asserted"
        )
        corrected.append(entry)
    return corrected, {
        "records": len(corrected),
        "no_positive_overlap": zero_overlap,
        "positive_overlap_and_matched_elsewhere": positive_and_matched_elsewhere,
        "with_predictions": sum(1 for row in corrected
                                if int(row.get("predictions_in_frame") or 0) > 0),
    }


def correct_performance(report: dict, raw: dict) -> tuple[dict, dict, dict]:
    """Rename the mislabelled stage; every timing value is preserved."""
    corrected = json.loads(json.dumps(report))
    stage_map: dict[str, str] = {}
    for bucket, entry in (corrected.get("scenarios") or {}).items():
        for variant, variant_entry in (entry.get("variants") or {}).items():
            if not isinstance(variant_entry, dict):
                continue
            stages = variant_entry.get("stages")
            if not isinstance(stages, dict):
                continue
            for old, new in STAGE_RENAMES.items():
                if old in stages:
                    stages[new] = stages.pop(old)
                    stage_map[old] = new
    definitions = (corrected.get("conditions") or {}).get("stage_definitions")
    if isinstance(definitions, dict):
        for old, new in STAGE_RENAMES.items():
            if old in definitions:
                definitions[new] = definitions.pop(old)
    corrected["stage_name_corrections"] = stage_map
    corrected["correction_note"] = (
        "only stage names changed; every timing value is the original "
        "measurement and this round ran no benchmark"
    )
    raw_corrected = json.loads(json.dumps(raw))
    raw_map: dict[str, str] = {}
    for bucket, variants in (raw_corrected.get("raw") or {}).items():
        if not isinstance(variants, dict):
            continue
        for variant, entry in variants.items():
            if not isinstance(entry, dict):
                continue          # e.g. the standalone rectification entry
            stages = entry.get("stages")
            if not isinstance(stages, dict):
                continue
            for old, new in STAGE_RENAMES.items():
                if old in stages:
                    stages[new] = stages.pop(old)
                    raw_map[old] = new
    raw_corrected["stage_name_corrections"] = raw_map
    return corrected, raw_corrected, {"report": stage_map, "raw": raw_map}


def performance_projection(report: dict) -> dict:
    """The parts of a performance report that must survive the correction.

    Stage keys are mapped through the approved rename; everything else —
    status, coverage, residency and every per-scenario statistic — is
    compared as it stands, so a changed latency, throughput, memory or
    coverage state is detected.
    """
    report = report or {}
    projection: dict = {
        "status": report.get("status"),
        "coverage": report.get("coverage"),
        "residency": report.get("residency"),
        "scenarios": {},
    }
    for bucket, entry in (report.get("scenarios") or {}).items():
        if not isinstance(entry, dict):
            projection["scenarios"][bucket] = entry
            continue
        item = {key: value for key, value in entry.items()
                if key != "variants"}
        variants = {}
        for variant, variant_entry in (entry.get("variants") or {}).items():
            if not isinstance(variant_entry, dict):
                variants[variant] = variant_entry
                continue
            stats = {key: value for key, value in variant_entry.items()
                     if key != "stages"}
            stats["stages"] = {
                STAGE_RENAMES.get(name, name): values
                for name, values in (variant_entry.get("stages") or {}).items()
            }
            variants[variant] = stats
        item["variants"] = variants
        projection["scenarios"][bucket] = item
    return projection


def consistency_checks(source_summary, corrected_summary, *,
                       source_timings, corrected_timings,
                       source_performance=None,
                       corrected_performance=None,
                       required_candidates=None) -> dict:
    """The real source-vs-corrected comparison plus the coverage counts."""
    comparison = compare_source_and_corrected(
        source_summary, corrected_summary,
        source_timings=source_timings, corrected_timings=corrected_timings,
        source_performance=source_performance,
        corrected_performance=corrected_performance,
        required_candidates=required_candidates,
    )
    counts = {
        tag: ((entry.get("layer_b") or {}).get("routed") or {}).get("counts")
        or {}
        for tag, entry in (source_summary.get("per_candidate") or {}).items()
    }
    return {
        **comparison,
        "source_per_candidate_counts": counts,
        "bootstrap_present": bool(
            (source_summary.get("bootstrap") or {}).get(
                "end_to_end_macro_f1_e4_vs_e1")
        ),
        "performance_status_preserved": (
            (source_performance or {}).get("status")
        ),
    }


def _write_verified_publication(output_dir: Path, reports: dict,
                                jsonl_reports: dict, markdown: str) -> list[str]:
    """Write once, then compare every byte to an independent frozen payload.

    The manifest is derived from intended bytes, not from possibly corrupted
    writes. Freeze before invoking any writer so mutations at that boundary
    cannot also change our verification baseline.
    """
    expected = {
        name: (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
               + "\n").encode("utf-8")
        for name, payload in reports.items()
    }
    expected.update({
        name: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                      for row in rows).encode("utf-8")
        for name, rows in jsonl_reports.items()
    })
    expected["correction_summary.md"] = markdown.encode("utf-8")
    manifest = {
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(expected.items())},
        "index_file": "artifact_manifest.json",
        "index_note": "index of this directory; does not cover itself",
    }
    try:
        for name in reports:
            write_json_bytes(output_dir, name, json.loads(expected[name]))
        for name in jsonl_reports:
            rows = [json.loads(line) for line in expected[name].splitlines()]
            write_jsonl_bytes(output_dir, name, rows)
        (output_dir / "correction_summary.md").write_bytes(
            expected["correction_summary.md"])
        write_json_bytes(output_dir, "artifact_manifest.json",
                         json.loads(json.dumps(manifest)))
        problems = [f"{name}: written content differs from intended content"
                    for name, data in expected.items()
                    if (output_dir / name).read_bytes() != data]
        if _read_json(output_dir / "artifact_manifest.json") != manifest:
            problems.append("artifact_manifest.json: content or coverage mismatch")
        if {p.name for p in output_dir.iterdir()} != set(expected) | {
                "artifact_manifest.json"}:
            problems.append("unexpected or missing publication artifacts")
        return problems
    except (OSError, ValueError) as exc:
        return [f"publication I/O or decoding error: {type(exc).__name__}: {exc}"]


def _invalidate_publication(output_dir: Path, problems: list[str]) -> None:
    """Do not leave a success-looking report after a failed bounded repair.

    Only this invocation's new output files are invalidated, never the source.
    Storage failure can prevent even a failure marker: stderr/nonzero remains
    authoritative in that case; no success message is emitted.
    """
    for name in ("quality_summary_corrected.json", "correction_summary.md",
                 "artifact_manifest.json"):
        try:
            (output_dir / name).unlink(missing_ok=True)
        except OSError as exc:
            problems.append(f"cannot invalidate {name}: {exc}")
    try:
        (output_dir / "publication_failed.json").write_bytes(
            (json.dumps({"status": "FAILED", "problems": problems},
                        ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    except OSError as exc:
        print(f"cannot write publication failure marker: {exc}", file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True,
                        help="the frozen experiment config (identity source)")
    parser.add_argument("--gt-index", required=True,
                        help="the complete val GT index (expected-set source)")
    args = parser.parse_args(argv)

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"source evidence directory not found: {source_dir}",
              file=sys.stderr)
        return 2
    output_dir = prepare_output_dir(args.output_dir, source_dir)

    # --- admission: source manifests, config identity, independent GT set ---
    provenance = verify_source(source_dir)
    provenance["correction_script_sha256"] = sha256_of(Path(__file__))
    provenance["shared_module_sha256"] = sha256_of(
        PROJECT_ROOT / "scripts/eval_material_pipeline.py")
    provenance["no_inference_this_round"] = (
        "no checkpoint was loaded, no engine built, no model forward, no "
        "SGBM and no benchmark was executed"
    )
    config_path = Path(args.config)
    gt_index_path = Path(args.gt_index)
    provenance["config"] = {
        "path": str(config_path),
        "sha256": sha256_of(config_path),
    }
    provenance["gt_index"] = {
        "path": str(gt_index_path),
        "sha256": sha256_of(gt_index_path),
    }
    config = load_material_runtime_config(config_path)
    v5_root = PROJECT_ROOT / config.training_data_root
    manifest_path = v5_root / "dataset_manifest.csv"
    provenance["v5_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_of(manifest_path),
        "dataset_fingerprint": config.data_fingerprint_v5,
    }
    with manifest_path.open(encoding="utf-8") as handle:
        manifest_rows = [row for row in csv.DictReader(handle)
                         if row.get("split") == "val"]
    npz_keys = {(row["source_frame"], int(row["object_index"]))
                for row in manifest_rows}
    gt_index = _read_jsonl(gt_index_path)
    expected_keys, without_npz = expected_keys_from_sources(
        gt_index, npz_keys=npz_keys)
    provenance["expected_set"] = {
        "frames": len(gt_index),
        "gt_instances": len(expected_keys),
        "gt_without_npz": len(without_npz),
        "source": "frozen GT index + bound V5 manifest (never the outputs "
                  "under review)",
    }

    quality_dir = source_dir / "quality"
    summary = _read_json(quality_dir / "summary.json")
    snapshot = _read_json(quality_dir / "protocol_snapshot.json")
    identities = snapshot.get("model_identities") or []
    raw_keys = [(str(entry.get("variant")), int(entry.get("seed")))
                for entry in identities if isinstance(entry, dict)
                and entry.get("variant") is not None
                and entry.get("seed") is not None]
    repeated = sorted({key for key in raw_keys if raw_keys.count(key) > 1})
    if repeated:
        raise ValueError(
            f"the protocol snapshot lists duplicate candidate identities "
            f"{repeated}; refusing to collapse them into a mapping"
        )
    snapshot_shas = {
        (str(entry["variant"]), int(entry["seed"])): str(entry["sha256"])
        for entry in identities
    }
    if len(snapshot_shas) != 12:
        raise ValueError(
            "the protocol snapshot does not record the 12 frozen candidate "
            "identities; the identity check cannot be completed"
        )
    binding_problems = []
    if list(snapshot.get("class_names") or []) != list(CLASS_NAMES):
        binding_problems.append("class order differs between the snapshot and "
                                "the project's fixed order")
    material = snapshot.get("material_model") or {}
    for field in ("imgsz", "preprocessing_version", "pooling_mode",
                  "mask_schema", "mask_resize_version"):
        expected = getattr(config, field, None)
        if field == "imgsz":
            expected = config.cmx_imgsz
        if expected is not None and str(material.get(field)) != str(expected):
            binding_problems.append(
                f"{field}: snapshot {material.get(field)!r} != config "
                f"{expected!r}"
            )
    identity = snapshot.get("training_data_identity") or {}
    if str(identity.get("fingerprint") or "") != config.data_fingerprint_v5:
        binding_problems.append(
            "the snapshot's training-data fingerprint does not match the "
            "admitted config"
        )
    audit_path = v5_root / "dataset_audit.json"
    if not audit_path.is_file():
        raise ValueError(f"the V5 audit file is missing: {audit_path}")
    audit = _read_json(audit_path)
    audit_binding = {"path": str(audit_path), "sha256": sha256_of(audit_path)}
    if audit.get("audit_passed") is not True:
        binding_problems.append(
            f"audit_passed must be boolean True, got "
            f"{type(audit.get('audit_passed')).__name__}: "
            f"{audit.get('audit_passed')!r}"
        )
    for name, field in (("dataset_manifest.csv", "manifest_sha256"),
                        ("dataset_summary.json", "summary_sha256")):
        recorded = audit.get(field)
        if not isinstance(recorded, str) or not _HEX64.match(recorded):
            binding_problems.append(
                f"the audit records no valid {field} for {name}"
            )
            continue
        actual = sha256_of(v5_root / name)
        if actual != recorded:
            binding_problems.append(
                f"{name} digest {actual[:12]}… != the audit's {recorded[:12]}…"
            )
    parent = audit.get("v4_parent_fingerprint")
    if not isinstance(parent, str) or not _HEX64.match(parent):
        binding_problems.append("the audit records no valid V4 parent "
                                "fingerprint")
    elif parent != config.parent_data_fingerprint_v4:
        binding_problems.append(
            "the audit's V4 parent fingerprint differs from the admitted config"
        )
    # The real V5 audit does not carry a `dataset_fingerprint` field; the
    # chain is established through the recorded digests and the parent, and
    # its absence is reported rather than invented.
    audit_binding["recorded_fields"] = sorted(
        key for key in ("audit_passed", "manifest_sha256", "summary_sha256",
                        "v4_parent_fingerprint", "mask_schema_version",
                        "mask_resize_version", "crop_pad")
        if key in audit
    )
    audit_binding["has_dataset_fingerprint"] = "dataset_fingerprint" in audit
    if "dataset_fingerprint" not in audit:
        audit_binding["dataset_fingerprint_note"] = (
            "the source audit does not record a dataset_fingerprint field; "
            "the identity chain is established from the recorded digests, the "
            "V4 parent and the config's own fingerprint — no new fingerprint "
            "definition was introduced"
        )
    from core.material_mask_dataset import mask_dataset_fingerprint

    try:
        computed_v5 = mask_dataset_fingerprint(v5_root)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise ValueError(
            f"the V5 identity cannot be computed from the audit metadata: "
            f"{exc}"
        ) from exc
    audit_binding["computed_v5_fingerprint"] = computed_v5
    audit_binding["fingerprint_source"] = (
        "core.material_mask_dataset.mask_dataset_fingerprint over the audit's "
        "recorded digests (no NPZ content re-audit)"
    )
    if computed_v5 != config.data_fingerprint_v5:
        binding_problems.append(
            "the V5 fingerprint computed from the audit metadata differs from "
            "the admitted config"
        )
    if str(identity.get("fingerprint") or "") != computed_v5:
        binding_problems.append(
            "the protocol snapshot's V5 fingerprint differs from the audit "
            "metadata"
        )
    for field, expected in (("mask_schema_version", config.mask_schema),
                            ("mask_resize_version",
                             config.mask_resize_version)):
        recorded = audit.get(field)
        if recorded is not None and str(recorded) != str(expected):
            binding_problems.append(
                f"{field}: the audit records {recorded!r}, the config expects "
                f"{expected!r}"
            )
    gt_dir = Path(args.gt_index).parent
    gt_manifest = gt_dir / "artifact_manifest.json"
    gt_binding = {"path": str(args.gt_index),
                  "sha256": sha256_of(args.gt_index)}
    if gt_manifest.is_file():
        manifest = _read_json(gt_manifest)
        recorded = (manifest.get("files") or {}).get(Path(args.gt_index).name)
        gt_binding["manifest"] = str(gt_manifest)
        if not isinstance(recorded, str) or not _HEX64.match(recorded):
            gt_binding["status"] = "UNVERIFIED"
            binding_problems.append(
                "the GT index is not covered by its evidence manifest"
            )
        elif recorded != gt_binding["sha256"]:
            gt_binding["status"] = "MISMATCH"
            binding_problems.append("the GT index digest differs from its "
                                    "evidence manifest")
        else:
            gt_binding["status"] = "BOUND"
    else:
        gt_binding["status"] = "UNVERIFIED"
        binding_problems.append(
            f"no evidence manifest next to the GT index: {gt_manifest}"
        )
    provenance["binding"] = {
        "audit": audit_binding,
        "gt_index": gt_binding,
        "note": "digests computed now and identities *recorded by the source* "
                "are reported separately; this is not a fresh content re-audit",
    }
    if binding_problems:
        raise ValueError("config/snapshot/data binding failed: "
                         + "; ".join(binding_problems))
    candidate_check = verify_candidates(
        [{"variant": candidate.variant, "seed": candidate.seed,
          "sha256": candidate.sha256} for candidate in config.candidates],
        expected_shas=snapshot_shas,
    )
    instances = _read_jsonl(quality_dir / "instances.jsonl")
    identity_problems = validate_instance_identities(instances)
    instance_record_problem = "; ".join(identity_problems[:5])
    try:
        if identity_problems:
            # An illegal instance identity never reaches the summary; it is
            # reported as FAILED with the field-level reason.
            raise ValueError(instance_record_problem)
        fallback_summary = summarise_fallback(
            instances,
            candidates=[(candidate.variant, candidate.seed)
                        for candidate in config.candidates],
        )
    except ValueError as exc:
        # A contradictory instance record set (duplicates, missing fields,
        # unknown identities) is a refusal, reported as FAILED rather than
        # raised out of the entry.
        instance_record_problem = "; ".join(
            part for part in (instance_record_problem, str(exc)) if part
        )
        fallback_summary = {}

    # The expected frame set is the frozen evaluation scope recorded by the
    # bound GT index — including frames whose instance list is legitimately
    # empty — and never the frames.jsonl file itself.
    expected_frame_ids: list[str] = []
    for entry in gt_index:
        frame_id = entry.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError(
                f"the GT index contains an illegal frame_id {frame_id!r}; the "
                "expected frame set cannot be established"
            )
        expected_frame_ids.append(frame_id)
    if len(set(expected_frame_ids)) != len(expected_frame_ids):
        raise ValueError(
            "the GT index lists duplicate frame_ids; the expected frame set "
            "would be ambiguous"
        )
    frame_rows = _read_jsonl(quality_dir / "frames.jsonl")
    frame_verdict = validate_frame_evidence(
        frame_rows, expected_frame_ids=expected_frame_ids
    )
    frame_predictions = frame_verdict["frame_predictions"]
    # Keep pure/routed supports separate; neither is borrowed from the other.
    summary_supports = {}
    for candidate in config.candidates:
        tag = f"{candidate.variant}_s{candidate.seed}"
        entry = (summary.get("per_candidate") or {}).get(tag)
        layer = entry.get("layer_b") if isinstance(entry, dict) else None
        supports = {}
        for condition in ALLOWED_CONDITIONS:
            block = layer.get(condition) if isinstance(layer, dict) else None
            counts = block.get("counts") if isinstance(block, dict) else None
            supports[condition] = (counts.get("predictions")
                                   if isinstance(counts, dict) else None)
        summary_supports[tag] = supports
    coverage = verify_instance_coverage(
        instances,
        candidates=[(candidate.variant, candidate.seed)
                    for candidate in config.candidates],
        expected_a2_keys=expected_keys,
        frame_predictions=frame_predictions,
        expected_frame_ids=expected_frame_ids,
        summary_supports=summary_supports,
    )
    alignment_rows = _read_jsonl(quality_dir / "a2_alignment.jsonl")
    alignment = recheck_alignment(alignment_rows, expected_keys)

    failed_dir = source_dir / "quality_attempt1_alignment_failed"
    failed_attempt = None
    if (failed_dir / "a2_alignment.jsonl").is_file():
        failed_rows = _read_jsonl(failed_dir / "a2_alignment.jsonl")
        failed_attempt = recheck_alignment(failed_rows, expected_keys)
        failed_attempt["recorded_summary_status"] = _read_json(
            failed_dir / "summary.json").get("status")
        failed_attempt["note"] = (
            "the first attempt is preserved as a failure; it is not deleted "
            "or rewritten into a success, and it does not veto a later "
            "qualified success"
        )

    unmatched_rows = _read_jsonl(quality_dir / "unmatched_gt.jsonl")
    corrected_unmatched, unmatched_summary = correct_unmatched_gt(unmatched_rows)

    performance_report = _read_json(
        source_dir / "performance/performance_report.json")
    raw_timings = _read_json(source_dir / "performance/raw_timings.json")
    performance_corrected, raw_corrected, stage_map = correct_performance(
        performance_report, raw_timings)
    performance_corrected["conditions"]["stage_definitions"] = _stage_definitions(
        performance_corrected["conditions"].get("stage_definitions") or {})

    # --- the verdict comes from this round's own checks ---------------------
    # The corrected report is built first; the comparison then checks it
    # against the source (never the source against itself).
    quality_corrected = {
        "status": None,  # filled in below, once the verdict is computed
        "source_status": summary.get("status"),
        "per_candidate": summary.get("per_candidate"),
        "bootstrap": summary.get("bootstrap"),
        "paired_changes": summary.get("paired_changes"),
        "conclusions": summary.get("conclusions"),
    }
    checks = consistency_checks(
        summary, quality_corrected,
        source_timings=raw_timings, corrected_timings=raw_corrected,
        required_candidates=[
            f"{candidate.variant}_s{candidate.seed}"
            for candidate in config.candidates
        ],
        source_performance=performance_projection(performance_report),
        corrected_performance=performance_projection(performance_corrected),
    )
    if (alignment["status"] == "FAILED" or checks["mismatches"]
            or coverage["status"] == "FAILED" or instance_record_problem
            or frame_verdict["status"] == "FAILED"):
        final_status = "FAILED"
    elif (alignment["status"] == "PARTIAL" or without_npz
          or coverage["status"] == "PARTIAL"
          or checks["status"] == "PARTIAL"
          or frame_verdict["status"] == "PARTIAL"):
        final_status = "PARTIAL"
    else:
        final_status = "COMPLETED"

    quality_corrected.update({
        "status": final_status,
        "status_basis": (
            "computed from this round's admission, alignment recheck and "
            "consistency comparison; the source status is recorded only as "
            "history"
        ),
        "admission": {
            "expected_set": provenance["expected_set"],
            "candidate_check": candidate_check,
            "covered_by_manifest": provenance["covered_by_manifest"],
        },
        "fallback_summary": fallback_summary,
        "instance_record_problem": instance_record_problem,
        "instance_coverage": coverage,
        "frame_evidence": frame_verdict,
        "alignment": alignment,
        "failed_attempt_alignment": failed_attempt,
        "unmatched_gt_summary": unmatched_summary,
        "bootstrap": summary.get("bootstrap"),
        "paired_changes": summary.get("paired_changes"),
        "per_candidate": summary.get("per_candidate"),
        "conclusions": summary.get("conclusions"),
        "consistency_checks": checks,
        "correction_note": (
            "the classification metrics, bootstrap intervals, per-instance "
            "predictions and matches are copied from the source evidence "
            "unchanged; this round re-ran no inference"
        ),
    })

    reports = {
        "source_provenance.json": provenance,
        "quality_summary_corrected.json": quality_corrected,
        "alignment_recheck.json": {
            "successful_attempt": alignment,
            "failed_attempt": failed_attempt,
            "expected_keys_source": "frozen GT index + bound V5 manifest",
            "expected_sample_count": len(expected_keys),
            "gt_without_npz": without_npz,
        },
        "performance_report_corrected.json": performance_corrected,
        "raw_timings_corrected.json": raw_corrected,
    }
    jsonl_reports = {"unmatched_gt_corrected.jsonl": corrected_unmatched}
    problems = _write_verified_publication(
        output_dir, reports, jsonl_reports,
        _render_markdown(provenance, quality_corrected, unmatched_summary,
                         stage_map, checks),
    )
    if problems:
        final_status = "FAILED"
        quality_corrected["status"] = "FAILED"
        quality_corrected["post_write_verification"] = {
            "status": "FAILED",
            "problems": problems,
            "note": "initial publication failed; one bounded FAILED "
                    "rewrite is attempted and independently rechecked",
        }
        repair_problems = _write_verified_publication(
            output_dir, reports, jsonl_reports,
            _render_markdown(provenance, quality_corrected, unmatched_summary,
                             stage_map, checks),
        )
        if repair_problems:
            _invalidate_publication(output_dir, problems + repair_problems)
            print("status: FAILED (publication could not be verified)", file=sys.stderr)
            return 1
    # No provisional success is announced before all bytes are verified.
    print(f"closeout written to {output_dir}")
    print(f"status: {final_status} | alignment: {alignment['status']}")
    print(f"consistency mismatches: {checks['mismatches'] or 'NONE'}")
    return 0 if final_status == "COMPLETED" else 1


def _stage_definitions(definitions: dict) -> dict:
    """The approved wording for the renamed CPU stage."""
    definitions = dict(definitions)
    entry = dict(definitions.get("cpu_tensor_preparation") or {})
    entry.update({
        "status": "MEASURED",
        "includes": "CPU-side resize/normalization/tensor preparation of the "
                    "ALREADY CROPPED instance windows",
        "excludes": "crop-only, host-to-device-only and pure model forward "
                    "(each NOT_MEASURED)",
        "renamed_from": "crop_preprocess_cpu",
        "rename_note": "the old name implied the crop was inside the timed "
                       "region; it is not",
    })
    definitions["cpu_tensor_preparation"] = entry
    return definitions


def _render_markdown(provenance, quality, unmatched, stage_map,
                     checks) -> str:
    lines = [
        "# 4B closeout corrections",
        "",
        f"- source: `{provenance['source_dir']}`",
        f"- status: **{quality['status']}** "
        f"(source status, history only: {quality['source_status']})",
        f"- expected set: {provenance['expected_set']['gt_instances']} GT "
        f"instances over {provenance['expected_set']['frames']} frames",
        "- no inference, no benchmark, no training was run in this round",
        "",
        "## Fallback summary (B routed only)",
        "",
        "| candidate | instances | fallback | reasons |",
        "|---|---|---|---|",
    ]
    for tag, entry in sorted(quality["fallback_summary"].items()):
        lines.append(
            f"| {tag} | {entry['instances']} | "
            f"{entry['used_fallback_instances']} | {entry['reasons']} |"
        )
    lines += [
        "",
        "## Alignment recheck",
        "",
        f"- successful attempt: {quality['alignment']['status']} "
        f"(covered {quality['alignment']['covered_samples']}/"
        f"{quality['alignment']['expected_samples']})",
        f"- preserved failed attempt: "
        f"{(quality['failed_attempt_alignment'] or {}).get('status')}",
        "- items the historical records cannot support are listed as "
        "unverifiable, never as newly measured passes",
        "",
        "## Unmatched GT correction",
        "",
        f"- records: {unmatched['records']}",
        f"- no positive overlap: {unmatched['no_positive_overlap']}",
        f"- positive overlap with an already-matched prediction: "
        f"{unmatched['positive_overlap_and_matched_elsewhere']}",
        "",
        "## Performance stage rename",
        "",
        f"- mapping: {stage_map}",
        "- timing values unchanged; the report status is preserved",
        "",
        "## Consistency",
        "",
        f"- mismatches: {checks['mismatches'] or 'NONE'}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
