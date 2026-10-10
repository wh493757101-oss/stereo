"""Independent recomputation of the complete per-prediction evidence
(residrt_finish batch).

Reads ONLY raw saved evidence, after binding every consumed file through
the source artifact manifest (recorded sha256/bytes verified against the
actual bytes BEFORE anything is read), and rebuilds every reported
statistic with its own boolean/numpy math. It never imports the
production summarizers or the evaluation entry for METRICS, so an
agreement between the two is independent evidence and a difference is a
finding — never something to reconcile by editing the raw records.

Authority chain (nothing is hardcoded here):
  --config            the frozen experiment config: the AUTHORITATIVE
                      candidate registry, class order, the protocol file
                      expectation (protocol.path + protocol.sha256), the
                      local cache pin and the legal fallback reasons
  --csv-path          the bound split CSV; its EXPECTED sha256/bytes come
                      from the protection snapshots (protected_before/
                      protected_after.json) bound and byte-verified through
                      the source manifest — the current file's own digest
                      is never substituted for the saved expectation
  protocol vs CSV     val frame sets and capture groups must agree; a
                      conflict is a refusal (RAW_INVALID), never a warning
  raw admission       the verified authority routing (frame→group→fold),
                      candidate identities (family/variant/seed) and the
                      legal fallback reasons constrain every raw row

CLI:
  --evidence-dir       the run's evaluate/ directory (raw evidence)
  --source-manifest    the source artifact_manifest.json (identity bind)
  --config             the frozen experiment config (see above)
  --csv-path           the bound split CSV (default: the real dataset path)
  --output-dir         a FRESH directory, never inside any input, never
                       overwriting history
  --expect-config-sha256 / --expect-source-manifest-sha256
                       optional identity anchors; an actual mismatch
                       stops the run (never auto-updated)

Exit codes: 0 = full admission + full comparison + zero differences
(PASSED); 2 = the raw evidence failed its structural admission (including
any authority-chain failure); 3 = the raw evidence and the source summary
disagree; 4 = refused (layout/anchor problems). Missing evidence is never
wrapped up as "zero differences".

The historical predictions_complete.jsonl stores p0_logits as None. The
historical records are NOT modified and no GPU forward is re-run: the
fallback rows' final logits are instead verified against the R0
candidate's own saved final logits, joined by
(frame_id, pred_instance_id) — a derived re-check whose source is
reported as ``R0_join_by_frame_and_pred_instance``. Missing or duplicate
R0 identities are refused; this does NOT claim the historical payload
stored P0 logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CLASS_NAMES = ("metal_submarine", "plastic_submarine", "plastic_fish",
               "real_fish")
BOOTSTRAP_SEED = 2026
BOOTSTRAP_RESAMPLES = 2000
NUM_CLASSES = len(CLASS_NAMES)
_CANDIDATE_ID_PATTERN = re.compile(
    r"^(linear_local|conditional)_([A-Z]+)_s(\d+)$")

# comparison contract, fixed before the recomputation ran
FLOAT_ATOL = 1e-12
FLOAT_RTOL = 0.0

CONSUMED_EVIDENCE = (
    "layer_a_samples.jsonl",
    "layer_a_summary.json",
    "frames_complete.jsonl",
    "predictions_complete.jsonl",
    "layer_b_summary.json",
    "protocol_snapshot.json",
    "run_setup.json",
    "run_status.json",
)

PROTECTION_SNAPSHOTS = ("protected_before.json", "protected_after.json")

EXIT_PASSED = 0
EXIT_RAW_INVALID = 2
EXIT_MISMATCH = 3
EXIT_REFUSED = 4

# the nine agreed count fields of the source contract
COUNTS_FIELDS = ("frames", "evaluated_frames", "failed_frames",
                 "gt_instances", "predictions", "matched", "missed_gt",
                 "unmatched_predictions", "unclassified_matched")

# descriptive fields the source carries that carry NO recomputable
# quantity; they are documented here and deliberately NOT compared —
# this list is the complete set of fields left uncompared
EXCLUDED_DESCRIPTIVE_FIELDS = (
    "metrics_by_candidate.*.detection.note",
    "metrics_by_candidate.*.detection.operating_point",
    "metrics_by_candidate.*.conditional_classification.confusion_rows",
    "metrics_by_candidate.*.conditional_classification.confusion_cols",
    "bootstrap.end_to_end_*.metric",
    "bootstrap.end_to_end_*.interval",
    "bootstrap.end_to_end_*.note",
    "bootstrap.end_to_end_*.classes",
    "bootstrap.instance_correctness_*.metric",
    "bootstrap.instance_correctness_*.note",
    "top-level summary: layer, data_independence, polar_quality_bins, "
    "prediction_records, quality_strata, paired_changes_detail, "
    "candidates, frames, failed_frames, fold_of_frame",
)


def resolve_project_path(value: str) -> Path:
    """PROJECT_ROOT-relative for relative values; absolute passthrough so
    synthetic worlds can bind their own files without touching the frozen
    real config."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_finite_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and bool(np.isfinite(float(value)))


def _strict_int(value: Any, path: str, differences: list[str]) -> None:
    """type(x) is int; bool and same-value float are refused."""
    if type(value) is not int:
        differences.append(
            f"{path}: expected strict int, got {value!r} of type "
            f"{type(value).__name__}")


def _finite_ratio(value: Any, path: str, differences: list[str]) -> None:
    """Finite non-bool number in [0, 1] (for rates, ratios, F1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        differences.append(
            f"{path}: expected a finite number, got {value!r} of type "
            f"{type(value).__name__}")
        return
    if not np.isfinite(float(value)):
        differences.append(f"{path}: non-finite value {value!r}")
    elif not 0.0 <= float(value) <= 1.0:
        differences.append(f"{path}: ratio {value!r} outside [0, 1]")


def _close(a: Any, b: Any) -> bool:
    """Deterministic metric comparison: atol=1e-12, rtol=0 (a testing
    tolerance for re-derived floats — never the model logits tolerance)."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (np.isfinite(float(a)) and np.isfinite(float(b))):
            return False  # NaN/Inf never pass, even both-sided
        return abs(float(a) - float(b)) <= FLOAT_ATOL
    return a == b


# ---------------------------------------------------------------------------
# Binding: the source manifest pins every consumed historical file


def bind_source_manifest(source_manifest_path: Path,
                         evidence_dir: Path) -> dict:
    """Verify every consumed evidence file's recorded digest/bytes
    against the actual bytes BEFORE anything is read. The manifest keys
    the files relative to ITS parent, so the evidence directory's
    relative position supplies the prefix (falling back to bare names)."""
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("the source manifest carries no files section")
    try:
        prefix = evidence_dir.resolve().relative_to(
            source_manifest_path.resolve().parent).as_posix() + "/"
    except ValueError:
        prefix = ""
    binding = {"manifest_sha256": sha256_of_file(source_manifest_path),
               "key_prefix": prefix, "verified": {}, "problems": []}
    for name in CONSUMED_EVIDENCE:
        record = files.get(prefix + name) or files.get(name)
        if not isinstance(record, dict):
            binding["problems"].append(
                f"{name}: not bound by the source manifest")
            continue
        path = evidence_dir / name
        if not path.is_file():
            binding["problems"].append(f"{name}: file missing on disk")
            continue
        digest = sha256_of_file(path)
        size = path.stat().st_size
        if digest != record.get("sha256"):
            binding["problems"].append(
                f"{name}: sha256 {digest} != the recorded "
                f"{record.get('sha256')}")
            continue
        if size != record.get("bytes"):
            binding["problems"].append(
                f"{name}: bytes {size} != the recorded {record.get('bytes')}")
            continue
        binding["verified"][name] = {"sha256": digest, "bytes": size}
    required = set(CONSUMED_EVIDENCE)
    if set(binding["verified"]) != required:
        binding["problems"].append(
            f"consumed set incomplete: verified "
            f"{sorted(binding['verified'])} != required {sorted(required)}")
    return binding


def check_output_location(output_dir: Path, inputs: Sequence[Path]) -> None:
    """The output must be fresh and never inside (or equal to) any input:
    directory inputs exclude everything inside them; file inputs exclude
    only their own exact path."""
    if output_dir.exists():
        raise ValueError(
            f"output directory {output_dir} already exists; recomputation "
            "reports are never overwritten")
    resolved = output_dir.resolve()
    for source in inputs:
        source = Path(source).resolve()
        if source.is_dir():
            if resolved == source or source in resolved.parents:
                raise ValueError(
                    f"output directory {output_dir} lies inside the input "
                    f"{source}; refusing to write reports into consumed "
                    "inputs")
        else:
            if resolved == source:
                raise ValueError(
                    f"output directory {output_dir} would overwrite the "
                    f"input file {source}")
    resolved.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Independent authority: protocol file, protection-bound CSV, class order,
# candidate registry identity


def verify_protocol_file(config: dict) -> tuple[dict, Path]:
    """The expected protocol file SHA256 comes from the BOUND config's
    ``protocol.sha256`` — never computed from the current bytes. Format
    checks run first, then the byte digest, then the content is read."""
    spec = config.get("protocol")
    if not isinstance(spec, dict):
        raise ValueError("the config declares no protocol section")
    path_value = spec.get("path")
    expected_sha = spec.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("the config's protocol.path is missing or empty")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or \
            any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError(
            f"the config's protocol.sha256 is not a legal sha256 hex "
            f"digest: {expected_sha!r}")
    protocol_path = resolve_project_path(path_value)
    if not protocol_path.is_file():
        raise ValueError(f"protocol file missing: {protocol_path}")
    digest = sha256_of_file(protocol_path)
    if digest != expected_sha:
        raise ValueError(
            f"protocol file sha256 {digest} != the config-pinned "
            f"{expected_sha}")
    return json.loads(protocol_path.read_text(encoding="utf-8")), \
        protocol_path


def extract_authoritative_frames(protocol: dict) -> tuple[dict, dict]:
    """frame→(group, fold) from the protocol's OWN inventory and frozen
    folds: every val frame's capture group must route to exactly one
    fold's outer_eval. Returns (routing, group_fold_map)."""
    group_fold: dict[str, str] = {}
    folds = (protocol.get("folds") or {}).get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("the protocol declares no folds")
    for fold in folds:
        fold_id = fold.get("fold_id")
        if not isinstance(fold_id, str) or not fold_id:
            raise ValueError("a protocol fold carries no fold_id")
        for group in fold.get("outer_eval_groups", []):
            if group in group_fold:
                raise ValueError(
                    f"capture group {group!r} routes to more than one "
                    "fold (ambiguous authority routing)")
            group_fold[group] = fold_id
    routing: dict[str, dict] = {}
    frames = (protocol.get("inventory") or {}).get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("the protocol inventory carries no frames")
    for frame in frames:
        frame_id = frame.get("frame_id")
        group = frame.get("group")
        if not isinstance(frame_id, str) or not frame_id or \
                not isinstance(group, str) or not group:
            raise ValueError(
                "the protocol inventory carries an empty frame identity")
        if frame_id in routing:
            raise ValueError(
                f"protocol frame {frame_id!r} appears more than once")
        if frame.get("split") != "val":
            continue
        fold = group_fold.get(group)
        if fold is None:
            raise ValueError(
                f"protocol val frame {frame_id!r}: its capture group "
                f"{group!r} belongs to no fold's outer_eval")
        routing[frame_id] = {"group": group, "fold": fold}
    if not routing:
        raise ValueError("the protocol declares no val frames")
    return routing, group_fold


def csv_snapshot_key(csv_path: Path) -> str:
    """The protection snapshots key targets by PROJECT_ROOT-relative
    posix path (the real dataset convention); a file outside the project
    (synthetic world) is keyed by its absolute posix path."""
    resolved = csv_path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def verify_protection_chain(source_manifest_path: Path,
                            csv_path: Path) -> dict:
    """The CSV expectation chain: BOTH protection snapshots must be bound
    by the source manifest and byte-verified BEFORE they are read; their
    CSV target records must exist, be typed legally, and agree; the
    resulting expectation is then checked against the current CSV bytes.
    The current file's own digest is never substituted for the saved
    expectation, and an unbound snapshot is never read."""
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_files = manifest.get("files", {})
    snapshots: dict[str, dict] = {}
    for name in PROTECTION_SNAPSHOTS:
        record = manifest_files.get(name)
        if not isinstance(record, dict):
            raise ValueError(
                f"{name}: not bound by the source manifest; refusing to "
                "read an unbound protection snapshot")
        path = source_manifest_path.parent / name
        if not path.is_file():
            raise ValueError(f"{name} missing: {path}")
        digest = sha256_of_file(path)
        size = path.stat().st_size
        if digest != record.get("sha256") or size != record.get("bytes"):
            raise ValueError(
                f"{name}: actual bytes (sha256 {digest}, {size}B) do not "
                f"match the source-manifest binding "
                f"({record.get('sha256')}, {record.get('bytes')}B)")
        snapshots[name] = json.loads(path.read_text(encoding="utf-8"))
    key = csv_snapshot_key(csv_path)
    expectations: dict[str, tuple[str, int]] = {}
    for name, snapshot in snapshots.items():
        targets = snapshot.get("targets")
        if not isinstance(targets, dict):
            raise ValueError(f"{name}: carries no targets mapping")
        record = targets.get(key)
        if not isinstance(record, dict):
            raise ValueError(
                f"{name}: no CSV target record for {key} — the CSV "
                "expectation is missing")
        sha = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(sha, str) or len(sha) != 64 or \
                any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(
                f"{name}: illegal CSV sha256 expectation {sha!r}")
        if type(size) is not int or size <= 0:
            raise ValueError(
                f"{name}: illegal CSV byte-length expectation {size!r}")
        expectations[name] = (sha, size)
    before = expectations["protected_before.json"]
    after = expectations["protected_after.json"]
    if before != after:
        raise ValueError(
            "the two protection snapshots disagree on the bound CSV "
            f"identity: {before} != {after}")
    expected_sha, expected_bytes = before
    if not csv_path.is_file():
        raise ValueError(f"the bound CSV is missing: {csv_path}")
    actual_sha = sha256_of_file(csv_path)
    actual_bytes = csv_path.stat().st_size
    if actual_sha != expected_sha:
        raise ValueError(
            f"the current CSV sha256 {actual_sha} differs from the saved "
            f"protection expectation {expected_sha}")
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"the current CSV byte length {actual_bytes} differs from the "
            f"saved protection expectation {expected_bytes}")
    return {"csv_key": key, "expected_sha256": expected_sha,
            "expected_bytes": expected_bytes,
            "snapshots": {name: {"sha256": sha256_of_file(
                source_manifest_path.parent / name)}
                for name in PROTECTION_SNAPSHOTS}}


def read_split_csv(csv_path: Path) -> list[dict]:
    import csv as csv_module

    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv_module.DictReader(handle))


def cross_check_csv_protocol(csv_rows: Sequence[dict],
                             routing: dict) -> None:
    """The bound CSV and the protocol must agree exactly on the val frame
    set and every val capture group; a conflict is a hard refusal — it
    can never degrade to a recorded warning with PASSED as the outcome."""
    by_stem: dict[str, dict] = {}
    for row in csv_rows:
        stem = row.get("output_stem")
        if not stem:
            raise ValueError("the bound CSV carries an empty output_stem")
        if stem in by_stem:
            raise ValueError(
                f"the bound CSV duplicates output_stem {stem!r}")
        by_stem[stem] = row
    val_stems = sorted(stem for stem, row in by_stem.items()
                       if row.get("split") == "val")
    protocol_stems = sorted(routing)
    if val_stems != protocol_stems:
        raise ValueError(
            "csv_protocol_conflict: the bound CSV and the protocol "
            "disagree on the val frame set " + json.dumps({
                "protocol_only": sorted(set(protocol_stems)
                                        - set(val_stems))[:5],
                "csv_only": sorted(set(val_stems)
                                   - set(protocol_stems))[:5]}))
    disagreements = [frame_id for frame_id, info in sorted(routing.items())
                     if by_stem[frame_id].get("group_name")
                     != info["group"]]
    if disagreements:
        raise ValueError(
            "csv_protocol_conflict: capture groups disagree for "
            f"{disagreements[:5]}")


def verify_class_order(config: dict, protocol: dict,
                       protocol_snapshot: dict) -> dict:
    """Every existing class-order declaration must equal the fixed four
    class order: the config's candidate and runtime sections, the
    protocol snapshot, and the protocol's OWN existing class fields
    (data.class_names, plus admission.class_names when present — no
    historical file is required to grow new fields)."""
    sources = [
        ("candidate.class_names",
         (config.get("candidate") or {}).get("class_names")),
        ("runtime.class_names",
         (config.get("runtime") or {}).get("class_names")),
        ("protocol_snapshot.class_names",
         protocol_snapshot.get("class_names")),
        ("protocol.data.class_names",
         (protocol.get("data") or {}).get("class_names")),
    ]
    admission = (protocol.get("admission") or {}).get("class_names")
    if admission is not None:
        sources.append(("protocol.admission.class_names", admission))
    problems = []
    for name, value in sources:
        if not isinstance(value, list) or \
                [str(entry) for entry in value] != list(CLASS_NAMES):
            problems.append(
                f"{name}: {value!r} != {list(CLASS_NAMES)}")
    return {"problems": problems, "passed": not problems,
            "compared_sources": [name for name, _ in sources]}


def verify_candidate_registry(entries: Any) -> tuple[dict, list]:
    """Registry identity: no empty/duplicate ids, and every entry's
    family/variant/seed must agree with the identity its candidate_id
    declares. R0 keeps seed=None; every correction candidate follows the
    family_VARIANT_sSEED pattern. Returns ({cid: spec}, problems)."""
    specs: dict[str, dict] = {}
    problems: list[dict] = []
    if not isinstance(entries, list) or not entries:
        return specs, [{"reason": "the config declares no candidate "
                                  "registry"}]
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append({"reason": "registry entry is not a mapping"})
            continue
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            problems.append({"reason": "a registry entry carries an empty "
                                       "candidate_id"})
            continue
        if candidate_id in seen:
            problems.append({"candidate": candidate_id,
                             "reason": "duplicate candidate_id"})
            continue
        seen.add(candidate_id)
        if candidate_id == "R0":
            if entry.get("seed") is not None:
                problems.append({"candidate": candidate_id,
                                 "reason": "R0 must keep seed=None"})
            if entry.get("family") != "p0":
                problems.append({"candidate": candidate_id,
                                 "reason": "R0 family must be p0"})
            specs[candidate_id] = {"family": "p0", "variant": "R0",
                                   "seed": None}
            continue
        match = _CANDIDATE_ID_PATTERN.match(candidate_id)
        if match is None:
            problems.append({"candidate": candidate_id,
                             "reason": "candidate_id does not follow the "
                                       "family_VARIANT_sSEED pattern"})
            continue
        family, variant, seed_text = match.groups()
        seed = int(seed_text)
        if entry.get("family") != family or entry.get("variant") != variant:
            problems.append({"candidate": candidate_id,
                             "reason": "registry family/variant disagree "
                                       "with the candidate_id"})
        if type(entry.get("seed")) is not int or entry.get("seed") != seed:
            problems.append({"candidate": candidate_id,
                             "reason": "registry seed disagrees with the "
                                       "candidate_id"})
        specs[candidate_id] = {"family": family, "variant": variant,
                               "seed": seed}
    if "R0" not in specs:
        problems.append({"candidate": "R0",
                         "reason": "the registry has no R0 baseline"})
    return specs, problems


# ---------------------------------------------------------------------------
# Raw structural admission (constrained by the VERIFIED authority)


def validate_raw(frames: list[dict], predictions: list[dict], *,
                 candidate_specs: dict, frame_routing: dict,
                 fallback_reasons: Sequence[str]) -> dict:
    """Structural admission of the raw evidence against the VERIFIED
    authority: the frame→group→fold routing extracted from the protocol,
    the candidate identity specs (family/variant/seed) and the legal
    fallback reasons — never just a candidate_id list. Returns a problems
    dict; an empty dict means the raw rows are admissible."""
    problems: dict[str, Any] = {}

    frame_ids = [frame.get("frame_id") for frame in frames]
    if any(fid is None for fid in frame_ids):
        problems["frames_missing_id"] = True
    duplicated_frames = sorted({fid for fid in frame_ids
                                if frame_ids.count(fid) > 1})
    if duplicated_frames:
        problems["duplicate_frames"] = duplicated_frames[:5]
    frame_by_id = {}
    for frame in frames:
        frame_by_id[frame["frame_id"]] = frame

    expected_frame_set = set(frame_routing)
    raw_frame_set = set(frame_ids)
    if raw_frame_set != expected_frame_set:
        problems["frame_set_mismatch"] = {
            "missing": sorted(expected_frame_set - raw_frame_set)[:5],
            "extra": sorted(raw_frame_set - expected_frame_set)[:5]}

    legal_reasons = set(fallback_reasons)
    for frame in frames:
        stem = frame.get("frame_id")
        routing = frame_routing.get(stem)
        if routing is None:
            problems.setdefault("frame_not_in_authority", []).append(stem)
        else:
            if frame.get("group") != routing["group"]:
                problems.setdefault("frame_group_mismatch", []).append(stem)
            if frame.get("fold") != routing["fold"]:
                problems.setdefault("frame_fold_mismatch", []).append(stem)
        if frame.get("frame_error"):
            continue
        gt_ids = [g.get("gt_id") for g in frame.get("gt", [])]
        if len(gt_ids) != len(set(gt_ids)):
            problems.setdefault("duplicate_gt_ids", []).append(stem)
        for g in frame.get("gt", []):
            class_id = g.get("class_id")
            if not isinstance(class_id, int) or isinstance(class_id, bool) \
                    or not 0 <= class_id < NUM_CLASSES:
                problems.setdefault("gt_class_out_of_range", []).append(stem)
        pred_ids = [p.get("pred_instance_id") for p in
                    frame.get("predictions", [])]
        if len(pred_ids) != len(set(pred_ids)):
            problems.setdefault("duplicate_pred_ids", []).append(stem)
        if any(type(pid) is not int or pid < 0 for pid in pred_ids):
            problems.setdefault("illegal_pred_ids", []).append(stem)
        matched_gt = [m[0] for m in frame.get("matches", [])]
        matched_pred = [m[1] for m in frame.get("matches", [])]
        if len(matched_gt) != len(set(matched_gt)) or \
                len(matched_pred) != len(set(matched_pred)):
            problems.setdefault("matching_not_one_to_one", []).append(stem)
        for m in frame.get("matches", []):
            if len(m) != 3 or not _is_finite_float(m[2]) or \
                    not 0.0 <= float(m[2]) <= 1.0:
                problems.setdefault("illegal_iou", []).append(stem)
            if m[0] not in gt_ids or m[1] not in pred_ids:
                problems.setdefault("match_references_unknown_identity",
                                    []).append(stem)
        missed = frame.get("missed_gt_ids")
        derived_missed = sorted(set(gt_ids) - set(matched_gt))
        if sorted(missed or []) != derived_missed:
            problems.setdefault("missed_ids_inconsistent", []).append(stem)
        unmatched = frame.get("unmatched_pred_ids")
        derived_unmatched = sorted(set(pred_ids) - set(matched_pred))
        if sorted(unmatched or []) != derived_unmatched:
            problems.setdefault("unmatched_ids_inconsistent", []).append(
                stem)

    match_map_by_frame: dict[str, dict[int, int]] = {}
    for frame in frames:
        match_map_by_frame[frame["frame_id"]] = {
            m[1]: m[0] for m in frame.get("matches", [])
            if type(m[1]) is int and type(m[0]) is int}

    by_candidate: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for row in predictions:
        candidate = row.get("candidate_id")
        if candidate not in candidate_specs:
            problems.setdefault("unknown_candidate_rows", []).append(
                candidate)
            continue
        spec = candidate_specs[candidate]
        key = (row.get("frame_id"), row.get("pred_instance_id"))
        if key in by_candidate[candidate]:
            problems.setdefault("duplicate_prediction_rows", []).append(
                {"candidate": candidate, "key": list(key)})
            continue
        by_candidate[candidate][key] = row
        # the ROW's instance id must itself be a strict non-negative int
        # (0 is legal; False and same-value floats are not) — the frame
        # records are checked separately below
        row_instance_id = row.get("pred_instance_id")
        if type(row_instance_id) is not int or row_instance_id < 0:
            problems.setdefault("illegal_pred_ids", []).append(
                {"candidate": candidate, "frame": row.get("frame_id"),
                 "pred_instance_id": row_instance_id})
        frame = frame_by_id.get(row.get("frame_id"))
        if frame is None:
            problems.setdefault("prediction_unknown_frame", []).append(
                row.get("frame_id"))
            continue
        # prediction identity: family and fold must match the verified
        # authority; seed must match the registry (R0 keeps None)
        if row.get("family") != spec["family"]:
            problems.setdefault("prediction_family_mismatch", []).append(
                {"candidate": candidate, "frame": row.get("frame_id"),
                 "family": row.get("family")})
        routing = frame_routing.get(row["frame_id"])
        if routing is not None and row.get("fold") != routing["fold"]:
            problems.setdefault("prediction_fold_mismatch", []).append(
                {"candidate": candidate, "frame": row.get("frame_id"),
                 "fold": row.get("fold")})
        seed = row.get("seed")
        if candidate == "R0":
            if seed is not None:
                problems.setdefault("prediction_seed_mismatch", []).append(
                    {"candidate": candidate, "frame": row.get("frame_id"),
                     "seed": seed})
        elif type(seed) is not int or seed != spec["seed"]:
            problems.setdefault("prediction_seed_mismatch", []).append(
                {"candidate": candidate, "frame": row.get("frame_id"),
                 "seed": seed})
        if type(row.get("used_fallback")) is not bool:
            problems.setdefault("used_fallback_not_bool", []).append(
                {"candidate": candidate, "frame": row.get("frame_id"),
                 "used_fallback": row.get("used_fallback")})
        pred_ids = [p.get("pred_instance_id")
                    for p in frame.get("predictions", [])]
        if row.get("pred_instance_id") not in pred_ids:
            problems.setdefault("prediction_unknown_instance", []).append(
                {"frame": row["frame_id"],
                 "pred_instance_id": row.get("pred_instance_id")})
        # matched_gt_id must correspond EXACTLY to this frame's matches:
        # a matched instance may not drop to None or move to another GT,
        # and only an unmatched instance may carry None
        instance_id = row.get("pred_instance_id")
        if type(instance_id) is int:
            expected_gt = match_map_by_frame[row["frame_id"]].get(
                instance_id)
            actual_gt = row.get("matched_gt_id")
            if not (type(actual_gt) is type(expected_gt)
                    and actual_gt == expected_gt):
                problems.setdefault("matched_gt_correspondence",
                                    []).append(
                    {"candidate": candidate, "frame": row["frame_id"],
                     "pred_instance_id": instance_id,
                     "matched_gt_id": actual_gt, "expected": expected_gt})
        logits = row.get("final_logits")
        if row.get("status") == "valid":
            if not isinstance(logits, list) or len(logits) != NUM_CLASSES \
                    or not all(_is_finite_float(v) for v in logits):
                problems.setdefault("illegal_logits", []).append(
                    {"candidate": candidate, "frame": row.get("frame_id")})
            else:
                argmax = int(np.asarray(logits).argmax())
                if row.get("predicted_class") != argmax:
                    problems.setdefault(
                        "predicted_class_not_argmax", []).append(
                        {"candidate": candidate,
                         "frame": row.get("frame_id")})
            class_id = row.get("predicted_class")
            if not isinstance(class_id, int) or isinstance(class_id, bool) \
                    or not 0 <= class_id < NUM_CLASSES:
                problems.setdefault("predicted_class_out_of_range",
                                    []).append(stem)
        if row.get("status") not in ("valid", "failed"):
            problems.setdefault("illegal_status", []).append(stem)
        if row.get("used_fallback") and not row.get("fallback_reason"):
            problems.setdefault("fallback_without_reason", []).append(stem)
        if row.get("fallback_reason") and not row.get("used_fallback"):
            problems.setdefault("reason_without_fallback", []).append(stem)
        if row.get("used_fallback") is True:
            reason = row.get("fallback_reason")
            if not isinstance(reason, str) or reason not in legal_reasons:
                problems.setdefault("illegal_fallback_reason", []).append(
                    {"candidate": candidate, "frame": row.get("frame_id"),
                     "fallback_reason": reason})

    for candidate in candidate_specs:
        rows = by_candidate.get(candidate, {})
        missing = [(frame["frame_id"], pred["pred_instance_id"])
                   for frame in frames
                   for pred in frame.get("predictions", [])
                   if (frame["frame_id"], pred["pred_instance_id"])
                   not in rows]
        if missing:
            problems.setdefault("incomplete_candidate_coverage", {})[
                candidate] = [
                {"frame_id": fid, "pred_instance_id": pid}
                for fid, pid in missing[:5]]
        if expected_frame_set and \
                set(key[0] for key in rows) - raw_frame_set:
            problems.setdefault("candidate_extra_frames",
                                {}).setdefault(candidate, True)

    return problems


def verify_fallback_against_r0(predictions: list[dict]) -> dict:
    """Derived re-check of the fallback rows: every correction
    candidate's fallback prediction must carry EXACTLY the R0
    candidate's final logits (both are the pure P0 output), joined by
    (frame_id, pred_instance_id). Source:
    ``R0_join_by_frame_and_pred_instance`` — this does NOT claim the
    historical payload stored P0 logits."""
    r0_by_key: dict[tuple, list[float]] = {}
    for row in predictions:
        if row.get("candidate_id") == "R0" and row.get("status") == "valid":
            key = (row.get("frame_id"), row.get("pred_instance_id"))
            if key in r0_by_key:
                return {"passed": False,
                        "reason": "duplicate R0 identity for the join",
                        "checked": len(r0_by_key), "problems": [
                            {"key": list(key),
                             "reason": "duplicate R0 identity"}]}
            r0_by_key[key] = row.get("final_logits")
    problems = []
    checked = 0
    for row in predictions:
        if not row.get("used_fallback"):
            continue
        key = (row.get("frame_id"), row.get("pred_instance_id"))
        reference = r0_by_key.get(key)
        if reference is None:
            problems.append({"key": list(key),
                             "reason": "no R0 row for the fallback join"})
            continue
        checked += 1
        if row.get("final_logits") != reference:
            problems.append({"key": list(key),
                             "reason": "fallback final logits differ from "
                                       "the R0 base"})
    return {"passed": not problems, "checked": checked,
            "problems": problems[:10],
            "source": "R0_join_by_frame_and_pred_instance"}


# ---------------------------------------------------------------------------
# Independent metric recomputation (own math only)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _macro_f1(tp: Sequence[int], fp: Sequence[int], fn: Sequence[int]) -> float:
    return float(np.mean([_prf(tp[i], fp[i], fn[i])["f1"]
                          for i in range(len(tp))]))


def _candidate_view(frames: list[dict], predictions: list[dict],
                    candidate_id: str) -> dict[tuple, dict]:
    view: dict[tuple, dict] = {}
    for row in predictions:
        if row["candidate_id"] == candidate_id:
            view[(row["frame_id"], row["pred_instance_id"])] = row
    return view


def _counts_for(frames: list[dict], view: dict[tuple, dict]) -> dict:
    """Count over the EXACT frame list given: a frame appearing twice is
    weighted twice (bootstrap group draws are never de-duplicated)."""
    num = NUM_CLASSES
    cond_tp = [0] * num
    cond_fp = [0] * num
    cond_fn = [0] * num
    cond_correct = 0
    e2e_tp = [0] * num
    e2e_fp = [0] * num
    e2e_fn = [0] * num
    confusion = [[0] * num for _ in range(num)]
    detection_tp = 0
    gt_total = pred_total = 0
    iou_values: list[float] = []
    failed_classifications = 0
    for frame in frames:
        if frame.get("frame_error"):
            continue
        gt_classes = {g["gt_id"]: g["class_id"] for g in frame["gt"]}
        gt_total += len(gt_classes)
        pred_total += len(frame["predictions"])
        detection_tp += len(frame["matches"])
        iou_values.extend(float(m[2]) for m in frame["matches"])
        matched_gt = {m[0] for m in frame["matches"]}
        matched_pred = {m[1] for m in frame["matches"]}
        for gt_id, pred_id, _iou in frame["matches"]:
            true_class = gt_classes[gt_id]
            row = view.get((frame["frame_id"], pred_id))
            if row is None or row.get("status") != "valid":
                failed_classifications += 1
                e2e_fn[true_class] += 1
                continue
            pred_class = int(row["predicted_class"])
            confusion[true_class][pred_class] += 1
            if pred_class == true_class:
                cond_tp[true_class] += 1
                cond_correct += 1
                e2e_tp[true_class] += 1
            else:
                cond_fn[true_class] += 1
                cond_fp[pred_class] += 1
                e2e_fn[true_class] += 1
                e2e_fp[pred_class] += 1
        for gt_id, true_class in gt_classes.items():
            if gt_id not in matched_gt:
                e2e_fn[true_class] += 1
        for pred in frame["predictions"]:
            if pred["pred_instance_id"] not in matched_pred:
                row = view.get((frame["frame_id"],
                                pred["pred_instance_id"]))
                if row is not None and row.get("status") == "valid":
                    e2e_fp[int(row["predicted_class"])] += 1
    total_frames = len(frames)
    failed = sum(1 for f in frames if f.get("frame_error"))
    return {
        "counts": {
            "frames": total_frames,
            "evaluated_frames": total_frames - failed,
            "failed_frames": failed,
            "gt_instances": gt_total,
            "predictions": pred_total,
            "matched": detection_tp,
            "missed_gt": gt_total - detection_tp,
            "unmatched_predictions": pred_total - detection_tp,
            "unclassified_matched": failed_classifications,
        },
        "detection": _prf(detection_tp, pred_total - detection_tp,
                          gt_total - detection_tp),
        "conditional": {
            "samples": sum(cond_tp) + sum(cond_fn),
            "accuracy": cond_correct / (sum(cond_tp) + sum(cond_fn))
            if (sum(cond_tp) + sum(cond_fn)) else 0.0,
            "macro_f1": _macro_f1(cond_tp, cond_fp, cond_fn),
            "per_class": {name: _prf(cond_tp[i], cond_fp[i], cond_fn[i])
                          for i, name in enumerate(CLASS_NAMES)},
            "confusion_matrix": confusion,
        },
        "end_to_end": {
            "macro_f1": _macro_f1(e2e_tp, e2e_fp, e2e_fn),
            "per_class": {name: _prf(e2e_tp[i], e2e_fp[i], e2e_fn[i])
                          for i, name in enumerate(CLASS_NAMES)},
        },
        "matched_iou": {
            "n": len(iou_values),
            "mean": float(np.mean(iou_values)) if iou_values else 0.0,
            "min": float(np.min(iou_values)) if iou_values else 0.0,
            "max": float(np.max(iou_values)) if iou_values else 0.0,
        },
    }


def _paired_counts(frames: list[dict], views: dict[str, dict[tuple, dict]],
                   other_id: str, base_id: str) -> dict:
    """fixed/broke/both pairs aligned by (frame_id, gt_id), under the
    established production semantics (an unmatched GT counts as
    incorrect on BOTH sides); matched-only sub-counts are exposed
    separately."""
    base_view = views[base_id]
    other_view = views[other_id]
    fixed = broke = both_correct = both_wrong = 0
    matched_keys = 0
    for frame in frames:
        if frame.get("frame_error"):
            continue
        gt_classes = {g["gt_id"]: g["class_id"] for g in frame["gt"]}
        matched_by_gt = {m[0]: m[1] for m in frame["matches"]}
        for gt_id, true_class in gt_classes.items():
            base_row = base_view.get((frame["frame_id"],
                                      matched_by_gt.get(gt_id)))
            other_row = other_view.get((frame["frame_id"],
                                        matched_by_gt.get(gt_id)))
            base_ok = bool(
                base_row is not None and base_row.get("status") == "valid"
                and int(base_row["predicted_class"]) == true_class)
            other_ok = bool(
                other_row is not None and other_row.get("status") == "valid"
                and int(other_row["predicted_class"]) == true_class)
            if gt_id in matched_by_gt:
                matched_keys += 1
            if base_ok and other_ok:
                both_correct += 1
            elif not base_ok and not other_ok:
                both_wrong += 1
            elif other_ok:
                fixed += 1
            else:
                broke += 1
    return {"fixed": fixed, "broke": broke, "both_correct": both_correct,
            "both_wrong": both_wrong, "matched_pairs": matched_keys}


def _per_frame_correctness(frames: list[dict], views: dict[str, dict],
                           candidate_id: str) -> tuple[list[float], list[str]]:
    values: list[float] = []
    groups: list[str] = []
    view = views[candidate_id]
    for frame in frames:
        if frame.get("frame_error"):
            continue
        gt_classes = {g["gt_id"]: g["class_id"] for g in frame["gt"]}
        matched_by_gt = {m[0]: m[1] for m in frame["matches"]}
        for gt_id, true_class in gt_classes.items():
            row = view.get((frame["frame_id"], matched_by_gt.get(gt_id)))
            ok = bool(row is not None and row.get("status") == "valid"
                      and int(row["predicted_class"]) == true_class)
            values.append(1.0 if ok else 0.0)
            groups.append(frame["group"])
    return values, groups


def _cluster_bootstrap(values_a: Sequence[float],
                       values_b: Sequence[float],
                       groups: Sequence[str]) -> dict:
    """Own implementation of the frozen paired cluster bootstrap: whole
    capture groups drawn with replacement, the SAME draw for both sides,
    seed 2026, 2000 resamples, percentile 2.5%/97.5%."""
    if not values_a:
        return {"delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "resamples": 0, "clusters": 0, "seed": BOOTSTRAP_SEED,
                "note": "no samples"}
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    groups_array = np.asarray(groups)
    deltas = values_a - values_b
    unique = sorted(set(groups))
    by_group = {g: np.where(groups_array == g)[0] for g in unique}
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = np.concatenate(
            [by_group[unique[rng.randrange(len(unique))]]
             for _ in unique])
        means.append(float(deltas[sample].mean()))
    means.sort()
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(means[int(0.025 * len(means))]),
        "ci_high": float(means[int(0.975 * len(means)) - 1]),
        "resamples": BOOTSTRAP_RESAMPLES,
        "clusters": len(unique),
        "seed": BOOTSTRAP_SEED,
    }


def _end_to_end_bootstrap(frames: list[dict], views: dict[str, dict],
                          other_id: str, base_id: str) -> dict:
    """Own implementation of the end-to-end Macro-F1 delta bootstrap:
    whole capture groups drawn with replacement (same draw for both
    sides), the four-class Macro-F1 re-accumulated per draw, seed 2026,
    2000 resamples. The full-sample counts of BOTH sides are reported
    (counts_e4 = the candidate, counts_e1 = the base) exactly as the
    source declares them."""
    groups = sorted({frame["group"] for frame in frames})
    by_group: dict[str, list[dict]] = defaultdict(list)
    for frame in frames:
        by_group[frame["group"]].append(frame)
    full_other = _counts_for(frames, views[other_id])
    full_base = _counts_for(frames, views[base_id])
    point = full_other["end_to_end"]["macro_f1"] - \
        full_base["end_to_end"]["macro_f1"]
    rng = random.Random(BOOTSTRAP_SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draws = [groups[rng.randrange(len(groups))] for _ in groups]
        # duplicate group draws are WEIGHTED (the drawn frames list keeps
        # every occurrence), never de-duplicated
        selected = [frame for g in draws for frame in by_group[g]]
        delta = _counts_for(selected, views[other_id])[
            "end_to_end"]["macro_f1"] \
            - _counts_for(selected, views[base_id])[
                "end_to_end"]["macro_f1"]
        deltas.append(delta)
    deltas.sort()
    if deltas:
        low = float(deltas[max(0, int(0.025 * len(deltas)))])
        high = float(deltas[min(len(deltas) - 1,
                                int(0.975 * len(deltas)) - 1)])
    else:
        low = high = 0.0
    return {
        "point_estimate": float(point),
        "ci_low": low,
        "ci_high": high,
        "resamples": BOOTSTRAP_RESAMPLES,
        "valid_resamples": len(deltas),
        "clusters": len(groups),
        "seed": BOOTSTRAP_SEED,
        "e4_macro_f1": float(full_other["end_to_end"]["macro_f1"]),
        "e1_macro_f1": float(full_base["end_to_end"]["macro_f1"]),
        "counts_e4": dict(full_other["counts"]),
        "counts_e1": dict(full_base["counts"]),
    }


def recompute(frames: list[dict], predictions: list[dict],
              candidate_ids: Sequence[str]) -> dict:
    """The full independent recomputation over the raw evidence."""
    views = {cid: _candidate_view(frames, predictions, cid)
             for cid in candidate_ids}
    report: dict = {"candidates": {}, "per_group": {}, "paired": {},
                    "bootstrap": {}, "fallback": {}}
    for candidate_id in candidate_ids:
        counts = _counts_for(frames, views[candidate_id])
        report["candidates"][candidate_id] = counts
        by_group: dict[str, list[dict]] = defaultdict(list)
        for frame in frames:
            by_group[frame["group"]].append(frame)
        report["per_group"][candidate_id] = {
            group: _counts_for(group_frames, views[candidate_id])
            for group, group_frames in sorted(by_group.items())}
    r0 = "R0" if "R0" in candidate_ids else None
    for candidate_id in candidate_ids:
        spec_family = ("conditional"
                       if candidate_id.startswith("conditional")
                       else "linear_local"
                       if candidate_id.startswith("linear_local")
                       else None)
        if spec_family is None:
            continue
        seed = candidate_id.rsplit("_s", 1)[-1]
        base_rc = f"{spec_family}_RC_s{seed}"
        if r0:
            report["paired"][f"{candidate_id}_minus_R0"] = _paired_counts(
                frames, views, candidate_id, r0)
            report["bootstrap"][
                f"end_to_end_macro_f1_{candidate_id}_minus_R0"] = \
                _end_to_end_bootstrap(frames, views, candidate_id, r0)
            va, ga = _per_frame_correctness(frames, views, candidate_id)
            vb, _gb = _per_frame_correctness(frames, views, r0)
            report["bootstrap"][
                f"instance_correctness_{candidate_id}_minus_R0"] = \
                _cluster_bootstrap(va, vb, ga)
        if base_rc in candidate_ids:
            # the self-pair labels (RC vs itself) are part of the
            # source's declared comparison set and are reproduced
            report["paired"][f"{candidate_id}_minus_RC"] = _paired_counts(
                frames, views, candidate_id, base_rc)
            report["bootstrap"][
                f"end_to_end_macro_f1_{candidate_id}_minus_RC"] = \
                _end_to_end_bootstrap(frames, views, candidate_id, base_rc)
            va, ga = _per_frame_correctness(frames, views, candidate_id)
            vb, _gb = _per_frame_correctness(frames, views, base_rc)
            report["bootstrap"][
                f"instance_correctness_{candidate_id}_minus_RC"] = \
                _cluster_bootstrap(va, vb, ga)
    for candidate_id in candidate_ids:
        view = views[candidate_id]
        rows = list(view.values())
        fallback = [row for row in rows if row.get("used_fallback")]
        reasons: dict[str, int] = defaultdict(int)
        for row in fallback:
            reasons[str(row.get("fallback_reason") or "")] += 1
        report["fallback"][candidate_id] = {
            "predictions": len(rows),
            "valid": sum(1 for row in rows if row.get("status") == "valid"),
            "failed": sum(1 for row in rows if row.get("status") != "valid"),
            "used_fallback": len(fallback),
            "reasons": dict(reasons),
            "fallback_ratio": len(fallback) / len(rows) if rows else 0.0,
        }
    return report


# ---------------------------------------------------------------------------
# Full comparison against the source's declared fields


def _walk_difference(path: str, mine: Any, own: Any,
                     differences: list[str]) -> None:
    if isinstance(mine, dict) and isinstance(own, dict):
        for key in sorted(set(mine) - set(own)):
            differences.append(f"{path}.{key}: missing from summary")
        for key in sorted(set(own) - set(mine)):
            differences.append(f"{path}.{key}: extra in summary")
        for key in sorted(set(mine) & set(own)):
            _walk_difference(f"{path}.{key}", mine[key], own[key],
                             differences)
        return
    if isinstance(mine, list) and isinstance(own, list):
        if len(mine) != len(own):
            differences.append(
                f"{path}: length {len(mine)} != {len(own)}")
            return
        for index, (m, o) in enumerate(zip(mine, own)):
            _walk_difference(f"{path}[{index}]", m, o, differences)
        return
    if isinstance(mine, (int, float)) and isinstance(own, (int, float)) \
            and not isinstance(mine, bool) and not isinstance(own, bool):
        if not _close(mine, own):
            differences.append(f"{path}: {mine} != {own}")
        return
    if mine != own:
        differences.append(f"{path}: {mine!r} != {own!r}")


_MISSING = object()


def _both_sides_int(mine: Any, own: Any, path: str,
                    differences: list[str]) -> None:
    """A count field: strict int on BOTH sides (an equal-value float on
    the summary side is refused), missing on either or both sides is a
    difference — never silently skipped."""
    if mine is _MISSING or own is _MISSING:
        side = ("both sides" if mine is _MISSING and own is _MISSING
                else "the recompute side" if mine is _MISSING
                else "the summary")
        differences.append(f"{path}: missing from {side}")
        return
    _strict_int(mine, path, differences)
    _strict_int(own, path, differences)
    if type(mine) is int and type(own) is int and mine != own:
        differences.append(f"{path}: {mine} != {own}")


def _both_sides_ratio(mine: Any, own: Any, path: str,
                      differences: list[str]) -> None:
    """A [0,1]-bounded ratio: finite and in range on BOTH sides (both
    sides equally broken never passes), then compared at atol=1e-12."""
    if mine is _MISSING or own is _MISSING:
        side = ("both sides" if mine is _MISSING and own is _MISSING
                else "the recompute side" if mine is _MISSING
                else "the summary")
        differences.append(f"{path}: missing from {side}")
        return
    _finite_ratio(mine, path, differences)
    _finite_ratio(own, path, differences)
    if not _close(mine, own):
        differences.append(f"{path}: {mine} != {own}")


def _both_sides_float(mine: Any, own: Any, path: str,
                      differences: list[str]) -> None:
    """A signed float (bootstrap point estimates and interval bounds may
    legally be negative): finite on BOTH sides, then atol-compared."""
    if mine is _MISSING or own is _MISSING:
        side = ("both sides" if mine is _MISSING and own is _MISSING
                else "the recompute side" if mine is _MISSING
                else "the summary")
        differences.append(f"{path}: missing from {side}")
        return
    for label, value in (("recompute", mine), ("summary", own)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            differences.append(
                f"{path}: expected a finite number on the {label} side, "
                f"got {value!r} of type {type(value).__name__}")
        elif not np.isfinite(float(value)):
            differences.append(
                f"{path}: non-finite on the {label} side: {value!r}")
    if not _close(mine, own):
        differences.append(f"{path}: {mine} != {own}")


def _compare_counts(mine_counts: Any, own_counts: Any, path: str,
                    differences: list[str]) -> None:
    if not isinstance(mine_counts, dict) or not isinstance(own_counts,
                                                           dict):
        differences.append(f"{path}: counts block missing on one side")
        return
    for key in COUNTS_FIELDS:
        _both_sides_int(mine_counts.get(key, _MISSING),
                        own_counts.get(key, _MISSING),
                        f"{path}.counts.{key}", differences)


def _compare_per_class(mine_map: Any, own_map: Any, path: str,
                       differences: list[str]) -> None:
    """The per-class blocks: the class SET is checked for
    missing/extra, then every class's tp/fp/fn (strict int) and
    precision/recall/f1 (ratio) on both sides."""
    mine_classes = sorted(mine_map) if isinstance(mine_map, dict) else []
    own_classes = sorted(own_map) if isinstance(own_map, dict) else []
    if mine_classes != own_classes:
        differences.append(
            f"{path}: class set {mine_classes} != {own_classes}")
        return
    for name in CLASS_NAMES:
        mine_entry = (mine_map or {}).get(name)
        own_entry = (own_map or {}).get(name)
        if not isinstance(mine_entry, dict) or not isinstance(own_entry,
                                                              dict):
            differences.append(
                f"{path}.{name}: per-class block missing on one side")
            continue
        for key in ("tp", "fp", "fn"):
            _both_sides_int(mine_entry.get(key, _MISSING),
                            own_entry.get(key, _MISSING),
                            f"{path}.{name}.{key}", differences)
        for key in ("precision", "recall", "f1"):
            _both_sides_ratio(mine_entry.get(key, _MISSING),
                              own_entry.get(key, _MISSING),
                              f"{path}.{name}.{key}", differences)


def _compare_confusion(mine_cm: Any, own_cm: Any, path: str,
                       differences: list[str]) -> None:
    """The full 4x4 confusion matrix: every cell a strict int on BOTH
    sides (bool and same-value float refused), every cell equal."""
    if not isinstance(mine_cm, list) or not isinstance(own_cm, list):
        differences.append(f"{path}: confusion matrix missing on one side")
        return
    if len(mine_cm) != NUM_CLASSES or len(own_cm) != NUM_CLASSES:
        differences.append(
            f"{path}: confusion matrix has {len(mine_cm)}/{len(own_cm)} "
            f"rows, expected {NUM_CLASSES}")
        return
    for i in range(NUM_CLASSES):
        mine_row = mine_cm[i]
        own_row = own_cm[i]
        if not isinstance(mine_row, list) or not isinstance(own_row, list) \
                or len(mine_row) != NUM_CLASSES \
                or len(own_row) != NUM_CLASSES:
            differences.append(
                f"{path}[{i}]: confusion row malformed")
            continue
        for j in range(NUM_CLASSES):
            _both_sides_int(mine_row[j], own_row[j],
                            f"{path}[{i}][{j}]", differences)


def _compare_candidate_block(mine: dict, own: dict, path: str,
                             differences: list[str]) -> None:
    """One candidate's full declared block: the nine counts, detection
    tp/fp/fn + rates, the complete conditional block, end_to_end and
    matched_iou — overall AND per group share this contract."""
    _compare_counts(mine.get("counts"), own.get("counts"), path,
                    differences)

    mine_det = mine.get("detection")
    own_det = own.get("detection")
    if not isinstance(mine_det, dict) or not isinstance(own_det, dict):
        differences.append(f"{path}.detection: block missing on one side")
    else:
        for key in ("tp", "fp", "fn"):
            _both_sides_int(mine_det.get(key, _MISSING),
                            own_det.get(key, _MISSING),
                            f"{path}.detection.{key}", differences)
        for key in ("precision", "recall", "f1"):
            _both_sides_ratio(mine_det.get(key, _MISSING),
                              own_det.get(key, _MISSING),
                              f"{path}.detection.{key}", differences)

    mine_cond = mine.get("conditional")
    own_cond = own.get("conditional_classification")
    if not isinstance(mine_cond, dict) or not isinstance(own_cond, dict):
        differences.append(
            f"{path}.conditional: block missing on one side")
    else:
        _both_sides_int(mine_cond.get("samples", _MISSING),
                        own_cond.get("samples", _MISSING),
                        f"{path}.conditional.samples", differences)
        for key in ("accuracy", "macro_f1"):
            _both_sides_ratio(mine_cond.get(key, _MISSING),
                              own_cond.get(key, _MISSING),
                              f"{path}.conditional.{key}", differences)
        _compare_confusion(mine_cond.get("confusion_matrix"),
                           own_cond.get("confusion_matrix"),
                           f"{path}.conditional.confusion", differences)
        _compare_per_class(mine_cond.get("per_class"),
                           own_cond.get("per_class"),
                           f"{path}.conditional.per_class", differences)

    mine_e2e = mine.get("end_to_end")
    own_e2e = own.get("end_to_end")
    if not isinstance(mine_e2e, dict) or not isinstance(own_e2e, dict):
        differences.append(
            f"{path}.end_to_end: block missing on one side")
    else:
        _both_sides_ratio(mine_e2e.get("macro_f1", _MISSING),
                          own_e2e.get("macro_f1", _MISSING),
                          f"{path}.end_to_end.macro_f1", differences)
        _compare_per_class(mine_e2e.get("per_class"),
                           own_e2e.get("per_class"),
                           f"{path}.end_to_end.per_class", differences)

    mine_iou = mine.get("matched_iou")
    own_iou = own.get("matched_iou")
    if not isinstance(mine_iou, dict) or not isinstance(own_iou, dict):
        differences.append(
            f"{path}.matched_iou: block missing on one side")
    else:
        _both_sides_int(mine_iou.get("n", _MISSING),
                        own_iou.get("n", _MISSING),
                        f"{path}.matched_iou.n", differences)
        for key in ("mean", "min", "max"):
            _both_sides_ratio(mine_iou.get(key, _MISSING),
                              own_iou.get(key, _MISSING),
                              f"{path}.matched_iou.{key}", differences)


def compare_with_summary(report: dict, summary: dict,
                         candidate_ids: Sequence[str]) -> dict:
    """The full comparison contract: every quantitative field the source
    declares is mapped and compared on BOTH sides — overall and per
    group, candidate set, class set, group set, paired/bootstrap/fallback
    labels. Strict ints exclude bool and same-value float on both sides;
    ratios are finite and in [0,1]; bootstrap point estimates and bounds
    may legally be negative; NaN/Inf and None never pass, even
    both-sided. Descriptive fields are excluded explicitly (see
    EXCLUDED_DESCRIPTIVE_FIELDS) and are never claimed as verified."""
    differences: list[str] = []

    summary_candidates = summary.get("metrics_by_candidate")
    if not isinstance(summary_candidates, dict):
        return {"difference_count": 1,
                "differences": ["metrics_by_candidate missing from summary"]}
    _walk_difference("metrics_by_candidate.candidates",
                     sorted(summary_candidates), sorted(candidate_ids),
                     differences)
    for candidate_id in candidate_ids:
        mine = report["candidates"][candidate_id]
        own = summary_candidates.get(candidate_id)
        if not isinstance(own, dict):
            differences.append(f"{candidate_id}: missing from summary")
            continue
        _compare_candidate_block(mine, own, candidate_id, differences)

    own_per_group = summary.get("per_group")
    if not isinstance(own_per_group, dict):
        differences.append("per_group missing from summary")
    else:
        # the per_group candidate set must match the admitted registry
        # EXACTLY — an extra or missing candidate is a difference, and a
        # non-mapping per-group container is a structural difference, not
        # a crash and never a pass-by-empty-default
        _walk_difference("per_group.candidates", sorted(candidate_ids),
                         sorted(own_per_group), differences)
        for candidate_id in candidate_ids:
            own_container = own_per_group.get(candidate_id)
            if not isinstance(own_container, dict):
                differences.append(
                    f"per_group.{candidate_id}: expected a per-group "
                    f"mapping, got "
                    f"{type(own_container).__name__}")
                continue
            _walk_difference(f"per_group.{candidate_id}.groups",
                             sorted(report["per_group"][candidate_id]),
                             sorted(own_container), differences)
            for group, counts in report["per_group"][candidate_id].items():
                own = own_container.get(group)
                if not isinstance(own, dict):
                    differences.append(
                        f"per_group.{candidate_id}.{group}: missing")
                    continue
                _compare_candidate_block(
                    counts, own,
                    f"per_group.{candidate_id}.{group}", differences)

    own_paired = summary.get("paired")
    if not isinstance(own_paired, dict):
        differences.append("paired missing from summary")
    else:
        _walk_difference("paired.labels", sorted(report["paired"]),
                         sorted(own_paired), differences)
        for label, outcome in report["paired"].items():
            own = own_paired.get(label)
            if not isinstance(own, dict):
                # an ABSENT label is already flagged by the label-set walk
                # above; a PRESENT but non-mapping result block is a
                # structural difference of its own — never silently
                # skipped and never substituted with zeros
                if label in own_paired:
                    differences.append(
                        f"paired.{label}: expected a result mapping, got "
                        f"{type(own).__name__}")
                continue
            for key in ("fixed", "broke", "both_correct", "both_wrong"):
                _both_sides_int(outcome.get(key, _MISSING),
                                own.get(key, _MISSING),
                                f"paired.{label}.{key}", differences)

    own_bootstrap = summary.get("bootstrap")
    if not isinstance(own_bootstrap, dict):
        differences.append("bootstrap missing from summary")
    else:
        expected_labels = sorted(report["bootstrap"])
        _walk_difference("bootstrap.labels", expected_labels,
                         sorted(own_bootstrap), differences)
        for label, mine in report["bootstrap"].items():
            own = own_bootstrap.get(label)
            if not isinstance(own, dict):
                # an ABSENT label is already flagged by the label-set walk
                # above; a PRESENT but non-mapping result block is a
                # structural difference — this guards BOTH the end-to-end
                # and the instance-correctness families before any field
                # comparison runs
                if label in own_bootstrap:
                    differences.append(
                        f"bootstrap.{label}: expected a result mapping, "
                        f"got {type(own).__name__}")
                continue
            if label.startswith("end_to_end_"):
                for key in ("point_estimate", "ci_low", "ci_high"):
                    _both_sides_float(mine.get(key, _MISSING),
                                      own.get(key, _MISSING),
                                      f"bootstrap.{label}.{key}",
                                      differences)
                for key in ("resamples", "valid_resamples", "clusters",
                            "seed"):
                    _both_sides_int(mine.get(key, _MISSING),
                                    own.get(key, _MISSING),
                                    f"bootstrap.{label}.{key}",
                                    differences)
                for key in ("e4_macro_f1", "e1_macro_f1"):
                    _both_sides_ratio(mine.get(key, _MISSING),
                                      own.get(key, _MISSING),
                                      f"bootstrap.{label}.{key}",
                                      differences)
                for key in ("counts_e4", "counts_e1"):
                    _compare_counts(mine.get(key), own.get(key),
                                    f"bootstrap.{label}.{key}", differences)
            else:
                for key in ("delta_mean", "ci_low", "ci_high"):
                    _both_sides_float(mine.get(key, _MISSING),
                                      own.get(key, _MISSING),
                                      f"bootstrap.{label}.{key}",
                                      differences)
                for key in ("resamples", "clusters", "seed"):
                    _both_sides_int(mine.get(key, _MISSING),
                                    own.get(key, _MISSING),
                                    f"bootstrap.{label}.{key}",
                                    differences)

    own_fallback = summary.get("fallback_all_predictions")
    if not isinstance(own_fallback, dict):
        differences.append("fallback_all_predictions missing from summary")
    else:
        for candidate_id in candidate_ids:
            mine = report["fallback"][candidate_id]
            own = own_fallback.get(candidate_id)
            if not isinstance(own, dict):
                differences.append(
                    f"fallback.{candidate_id}: missing from summary")
                continue
            for key in ("predictions", "valid", "failed", "used_fallback"):
                _both_sides_int(mine.get(key, _MISSING),
                                own.get(key, _MISSING),
                                f"fallback.{candidate_id}.{key}",
                                differences)
            _both_sides_ratio(mine.get("fallback_ratio", _MISSING),
                              own.get("fallback_ratio", _MISSING),
                              f"fallback.{candidate_id}.fallback_ratio",
                              differences)
            mine_reasons = mine.get("reasons", {})
            own_reasons = own.get("reasons", {})
            _walk_difference(f"fallback.{candidate_id}.reasons",
                             mine_reasons, own_reasons, differences)
            if isinstance(own_reasons, dict):
                for reason, count in own_reasons.items():
                    _strict_int(count,
                                f"fallback.{candidate_id}.reasons.{reason}",
                                differences)

    return {"difference_count": len(differences),
            "differences": differences[:80]}


# ---------------------------------------------------------------------------
# Frozen Layer A record verdict (no forward, no model)


def layer_a_record_verdict(evidence_dir: Path, config: dict) -> dict:
    """Re-judge the frozen layer_a_samples.jsonl against the AUTHORITATIVE
    expectations derived from the bound protocol-side sources: the
    candidate registry from the frozen config and the expected sample
    names from the config-bound local cache manifest (whose digest is
    pinned in the config). The verdict code is the production
    layer_a_verdict logic re-implemented here... it is IMPORTED instead —
    the verdict is an admission rule, not a metric: importing it does not
    weaken independence of the METRIC recomputation.
    """
    from scripts.eval_material_residual_pipeline import layer_a_verdict

    local_cache = (config.get("features", {}).get("local_cache", {}))
    manifest_ref_digest = (local_cache.get("files", {})
                           .get("feature_manifest.json"))
    if not local_cache.get("path") or not manifest_ref_digest:
        return {"status": "REFUSED",
                "reason": "the config binds no local cache manifest digest"}
    manifest_path = resolve_project_path(
        str(local_cache["path"]).rstrip("/\\")) / "feature_manifest.json"
    if not manifest_path.is_file():
        return {"status": "REFUSED",
                "reason": f"the config-bound local cache manifest is "
                          f"missing: {manifest_path}"}
    digest = sha256_of_file(manifest_path)
    if digest != manifest_ref_digest:
        return {"status": "REFUSED",
                "reason": "the local cache manifest digest does not match "
                          "the frozen config pin"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = [row["sample_name"] for row in manifest["samples"]]
    candidate_ids = [entry["candidate_id"]
                     for entry in config.get("candidates", [])]
    required_stages = {candidate: ("g", "c14", "s16", "p0", "final")
                       for candidate in candidate_ids}
    rows = _read_jsonl(evidence_dir / "layer_a_samples.jsonl")
    verdict = layer_a_verdict(rows, expected_names=expected_names,
                              expected_candidates=candidate_ids,
                              required_stages=required_stages)
    recorded = json.loads((evidence_dir / "layer_a_summary.json")
                          .read_text(encoding="utf-8")).get("verdict", {})
    agree = verdict["status"] == recorded.get("status")
    return {"status": verdict["status"], "recorded_status":
            recorded.get("status"), "agrees_with_recorded": agree,
            "rows": len(rows), "problems": verdict["problems"]}


# ---------------------------------------------------------------------------
# CLI


def _write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recompute_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _refuse(message: str) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Independent recomputation of the raw per-prediction "
                    "evidence, bound through the source manifest "
                    "(read-only).")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--config", required=True,
                        help="the frozen experiment config: the "
                             "authoritative candidate registry, class "
                             "order and protocol/CSV expectations")
    parser.add_argument("--csv-path",
                        default="datasets/underwater_seg_binary_v2/"
                                "split_assignment.csv",
                        help="the bound split CSV; its expectation comes "
                             "from the protection snapshots, never from "
                             "the current bytes")
    parser.add_argument("--output-dir", required=True,
                        help="fresh directory; never inside any input")
    parser.add_argument("--expect-config-sha256", default=None,
                        help="identity anchor for the config; an actual "
                             "mismatch stops the run")
    parser.add_argument("--expect-source-manifest-sha256", default=None,
                        help="identity anchor for the source manifest")
    parser.add_argument("--skip-layer-a-verdict", action="store_true",
                        help="engineering guard for synthetic tests whose "
                             "config binds no real local cache")
    args = parser.parse_args(argv)

    report: dict = {"anchors": {}, "binding": {}, "raw_problems": {},
                    "comparison": {}, "layer_a_record_verdict": {},
                    "comparison_contract": {
                        "float_atol": FLOAT_ATOL,
                        "float_rtol": FLOAT_RTOL,
                        "counts_fields": list(COUNTS_FIELDS),
                        "int_semantics": "type(x) is int on BOTH sides; "
                                         "bool and same-value float are "
                                         "refused",
                        "excluded_descriptive_fields": list(
                            EXCLUDED_DESCRIPTIVE_FIELDS)},
                    "status": "PASSED"}

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        return _refuse(f"config not found: {config_path}")
    config_digest = sha256_of_file(config_path)
    report["anchors"]["config_sha256"] = config_digest
    if args.expect_config_sha256 and \
            config_digest != args.expect_config_sha256:
        return _refuse(
            f"the config sha256 {config_digest} does not match the "
            f"accepted anchor {args.expect_config_sha256}; the anchor is "
            "never auto-updated")

    source_manifest_path = Path(args.source_manifest)
    if not source_manifest_path.is_absolute():
        source_manifest_path = PROJECT_ROOT / source_manifest_path
    if not source_manifest_path.is_file():
        return _refuse(
            f"source manifest not found: {source_manifest_path}")
    manifest_digest = sha256_of_file(source_manifest_path)
    report["anchors"]["source_manifest_sha256"] = manifest_digest
    if args.expect_source_manifest_sha256 and \
            manifest_digest != args.expect_source_manifest_sha256:
        return _refuse(
            f"the source manifest sha256 {manifest_digest} does not match "
            f"the accepted anchor {args.expect_source_manifest_sha256}")

    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.is_absolute():
        evidence_dir = PROJECT_ROOT / evidence_dir
    if not evidence_dir.is_dir():
        return _refuse(f"evidence dir not found: {evidence_dir}")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    # discover actual consumed inputs for the output guard
    import yaml as _yaml

    _cfg = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _input_guards = [evidence_dir, source_manifest_path, config_path]
    _protocol_path = resolve_project_path(
        (_cfg or {}).get("protocol", {}).get("path", ""))
    if _protocol_path.is_file():
        _input_guards.append(_protocol_path)
    _csv_path = resolve_project_path(args.csv_path)
    if _csv_path.is_file():
        _input_guards.append(_csv_path)
    for _snapshot in PROTECTION_SNAPSHOTS:
        _snapshot_path = source_manifest_path.parent / _snapshot
        if _snapshot_path.is_file():
            _input_guards.append(_snapshot_path)
    _local_cache = (_cfg or {}).get("features", {}).get("local_cache", {})
    if _local_cache.get("path"):
        _input_guards.append(resolve_project_path(_local_cache["path"]))
    _global_cache = (_cfg or {}).get("features", {}).get("global_cache", {})
    if _global_cache.get("path"):
        _input_guards.append(resolve_project_path(_global_cache["path"]))
    _source_root = source_manifest_path.parent
    if _source_root.is_dir():
        _input_guards.append(_source_root)
    try:
        check_output_location(output_dir, _input_guards)
    except ValueError as error:
        return _refuse(str(error))

    # 1. the config is the authoritative candidate registry + class order
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate_ids = [entry["candidate_id"]
                     for entry in config.get("candidates", [])]
    if not candidate_ids:
        return _refuse("the config declares no candidate registry")
    report["authoritative_candidates"] = candidate_ids

    # 2. bind every consumed historical file through the source manifest
    try:
        binding = bind_source_manifest(source_manifest_path, evidence_dir)
    except ValueError as error:
        return _refuse(str(error))
    report["binding"] = binding
    if binding["problems"]:
        report["status"] = "RAW_INVALID"
        report["raw_problems"] = {"binding": binding["problems"]}
        _write_report(output_dir, report)
        print("RAW_INVALID: the source binding is incomplete; see the "
              "report", file=sys.stderr)
        return EXIT_RAW_INVALID

    # 3. the independent authority chain — protocol file (expectation from
    # the config), protocol routing, protection-bound CSV chain, class
    # order and candidate registry identity — ALL before any raw admission
    protocol_snapshot = json.loads(
        (evidence_dir / "protocol_snapshot.json").read_text(
            encoding="utf-8"))
    authority_report: dict = {"protocol_verified": False,
                              "csv_verified": False,
                              "expected_frame_count": 0,
                              "group_fold_map": {},
                              "csv_val_rows": 0}
    authority_error = None
    protocol = None
    frame_routing = None
    candidate_specs = None
    fallback_reasons: list[str] = []
    try:
        protocol, protocol_path = verify_protocol_file(config)
        authority_report["protocol_verified"] = True
        authority_report["protocol_path"] = str(protocol_path)
        frame_routing, group_fold_map = extract_authoritative_frames(
            protocol)
        authority_report["expected_frame_count"] = len(frame_routing)
        authority_report["group_fold_map"] = group_fold_map
        csv_path = resolve_project_path(args.csv_path)
        authority_report["csv_chain"] = verify_protection_chain(
            source_manifest_path, csv_path)
        csv_rows = read_split_csv(csv_path)
        authority_report["csv_val_rows"] = len(csv_rows)
        cross_check_csv_protocol(csv_rows, frame_routing)
        authority_report["csv_verified"] = True
        class_order = verify_class_order(config, protocol,
                                         protocol_snapshot)
        authority_report["class_order"] = class_order
        if not class_order["passed"]:
            raise ValueError(
                "class order verification failed: "
                + "; ".join(class_order["problems"]))
        candidate_specs, registry_problems = verify_candidate_registry(
            config.get("candidates", []))
        authority_report["registry_problems"] = registry_problems
        if registry_problems:
            raise ValueError(
                "candidate registry identity failed: "
                + json.dumps(registry_problems[:5]))
        raw_reasons = (config.get("candidate") or {}).get("fallback", {})\
            .get("reasons")
        if not isinstance(raw_reasons, list) or \
                not all(isinstance(reason, str) and reason
                        for reason in raw_reasons):
            raise ValueError(
                "the config declares no legal fallback reason list "
                "(candidate.fallback.reasons)")
        fallback_reasons = list(raw_reasons)
        authority_report["fallback_reasons"] = fallback_reasons
    except ValueError as error:
        authority_error = str(error)
    report["authority"] = authority_report
    if authority_error is not None:
        report["status"] = "RAW_INVALID"
        report["raw_problems"] = {"authority": authority_error}
        _write_report(output_dir, report)
        print(f"RAW_INVALID: the authority chain failed: "
              f"{authority_error}", file=sys.stderr)
        return EXIT_RAW_INVALID

    # the config's registry must agree with what the run declared
    summary_path = evidence_dir / "layer_b_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    declared = summary.get("candidates")
    if not declared:
        declared = protocol_snapshot.get("candidates", [])
    declared_candidates = [
        c if isinstance(c, str) else c.get("candidate_id")
        for c in declared]
    if sorted(declared_candidates) != sorted(candidate_ids):
        report["status"] = "RAW_INVALID"
        report["raw_problems"] = {
            "registry_mismatch": {
                "config": sorted(candidate_ids),
                "declared": sorted(declared_candidates)}}
        _write_report(output_dir, report)
        print("RAW_INVALID: the candidate registry disagrees with the "
              "declared run", file=sys.stderr)
        return EXIT_RAW_INVALID

    # 4. read the raw evidence and admit it against the verified authority
    frames = _read_jsonl(evidence_dir / "frames_complete.jsonl")
    predictions = _read_jsonl(evidence_dir / "predictions_complete.jsonl")
    raw_problems = validate_raw(frames, predictions,
                                candidate_specs=candidate_specs,
                                frame_routing=frame_routing,
                                fallback_reasons=fallback_reasons)
    fallback_join = verify_fallback_against_r0(predictions)
    report["fallback_r0_join"] = fallback_join
    if fallback_join.get("passed") is False:
        raw_problems["fallback_r0_join"] = fallback_join.get(
            "problems", [{"reason": fallback_join.get("reason")}])
    report["raw_problems"] = raw_problems
    if raw_problems:
        report["status"] = "RAW_INVALID"
        _write_report(output_dir, report)
        print("RAW_INVALID: the raw evidence failed its structural "
              "admission; see the report", file=sys.stderr)
        return EXIT_RAW_INVALID

    # 5. the frozen Layer A records re-judged (no forward, no model)
    if not args.skip_layer_a_verdict:
        report["layer_a_record_verdict"] = layer_a_record_verdict(
            evidence_dir, config)

    # 6. the full independent recomputation and comparison
    recomputed = recompute(frames, predictions, candidate_ids)
    report["recompute"] = recomputed
    comparison = compare_with_summary(recomputed, summary, candidate_ids)
    report["comparison"] = comparison

    layer_a_ok = report.get("layer_a_record_verdict", {}).get("status") \
        in ("OK", None)
    if comparison["difference_count"] == 0 and layer_a_ok:
        report["status"] = "PASSED"
    else:
        report["status"] = "MISMATCH"

    _write_report(output_dir, report)
    if report["status"] == "PASSED":
        print(f"PASSED: the independent recomputation agrees with the "
              f"source on every compared field ({output_dir})")
        return EXIT_PASSED
    print(f"MISMATCH: {comparison['difference_count']} differences; "
          f"layer_a verdict "
          f"{report['layer_a_record_verdict'].get('status')}; see the "
          f"report", file=sys.stderr)
    return EXIT_MISMATCH


if __name__ == "__main__":
    raise SystemExit(main())
