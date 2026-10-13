#!/usr/bin/env python
"""material_measurement_diagnostic_v1 — batch-1 measurement-chain and
architecture-limitation diagnostic entry.

Answers three questions WITHOUT training, WITHOUT loading any model
weights and WITHOUT changing production behaviour:

A. Residual capacity: how many of the frozen P0's misclassifications could
   the frozen ±0.5/logit correction rule ever flip (per-sample m = logit
   gap), on the saved outer-eval oracles, plus the runtime pipeline join
   by (frame_id, matched_gt_id)?
B. Disparity upsample chain: analytic synthetic counterexamples for the
   production zero-fill + INTER_LINEAR upsample, with a valid-weight
   normalized control (diagnostic only, not a fix).
C. A frozen, deterministic frame selection (required coverage + cap 64).
D. Single-factor real-image recomputation on the selected frames:
   GT-conditional and prediction-conditional measurement under
   (1) the production path, (2) only the upsample swapped, (3) ±1px right
   shifts with full re-matching, (4) the same shifts with the baseline
   disparity held fixed (warp/sampling only).
E. The 15 real-fish missed-GT cases: per-case evidence table from the
   saved D0-D3 records and atlas, with honest review status.

Every stage is gated by one shared wall budget; finished evidence is kept
on exhaustion and the run reports PARTIAL with a nonzero exit. The three
frozen SHA256 anchors are verified before anything else. Consumed inputs
get a same-window BEFORE/AFTER protection record.
"""

from __future__ import annotations

import argparse
import csv
import json
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.existing_data_protocol import BudgetExceeded, BudgetWatch, sha256_of_file
from core.gt_geometry import build_gt_geometry, gt_crop_windows, gt_instances
from core.measurement_diagnostic import (
    CLASS_NAMES,
    IdentityError,
    PRODUCTION_UPSAMPLE,
    compare_measurements,
    compute_polar_features,
    edge_unmeasurable_mask,
    instance_boundary_split,
    instance_paired_comparison,
    load_and_merge_p0_oracles,
    pipeline_gt_prediction_join,
    residual_capacity_stats,
    run_synthetic_scenarios,
    select_diagnostic_frames,
    SelectionConflict,
    shift_right_image,
    upsample_lowres,
    run_sgbm_lowres,
    DiagnosticUpsampleMatcher,
)
from core.stereo_matching import (
    StereoMatcher,
    StereoMatcherConfig,
    build_horizontal_bands,
    disparity_to_depth,
    to_gray_u8,
)

MODES = ("saved", "synthetic", "recompute", "all")
MODE_STAGES = {
    "saved": ("verify", "selection", "saved", "protection"),
    "synthetic": ("verify", "synthetic", "protection"),
    "recompute": ("verify", "selection", "recompute", "protection"),
    "all": ("verify", "selection", "saved", "synthetic", "recompute",
            "review", "protection"),
}
EXIT_OK = 0
EXIT_PARTIAL = 2
EXIT_FAILED = 3
EXIT_USAGE = 4

_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_FLAG)


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def dump_json(path: Path, payload: Any) -> None:
    """JSON dump that refuses NaN/Inf anywhere in the payload."""

    def check(node: Any, where: str) -> None:
        if isinstance(node, float):
            if not np.isfinite(node):
                raise ValueError(
                    f"refusing to write non-finite float at {where}")
        elif isinstance(node, dict):
            for key, value in node.items():
                check(value, f"{where}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                check(value, f"{where}[{index}]")

    check(payload, path.name)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    lines = []
    for index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(
                    f"refusing non-finite float in row {index} field {key!r}")
        lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Run context


@dataclass
class RunContext:
    config: dict
    config_path: Path
    output_dir: Path
    mode: str
    budget_minutes: float
    max_frames: int
    run_id: str
    watch: BudgetWatch
    stages: dict = field(default_factory=dict)
    # identity recorded at each input's FIRST read (consume time, before
    # the consuming stage parses the file); end-of-run comparison detects
    # any change AFTER that first read. Reads made before this recording
    # are limited to the three anchors (byte-verified against config pins)
    # and the artifact-manifest/frozen-YAML binding lookups (themselves
    # byte-verified anchors whose recorded expectations are checked before
    # anything else reads the bound files).
    first_read: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    amendments: list = field(default_factory=list)
    frame_rows: list = field(default_factory=list)
    selection: dict | None = None
    protocol: dict | None = None
    frozen_runtime: dict | None = None
    artifact_manifest: dict | None = None

    def stage(self, name: str, status: str, **details: Any) -> None:
        self.stages[name] = {"status": status, **details}
        log(f"stage {name}: {status}")

    def consume(self, path: Path, role: str, pinned: str | None = None,
                binding: dict | None = None) -> dict:
        """Record an input's identity at its FIRST read.

        ``binding`` carries a frozen expectation (source, expected sha256,
        optional expected bytes); the current bytes are checked against it
        at this first read and a mismatch is refused before any stage
        consumes the content. Current hashes are NEVER written down as
        frozen expectations — expectations only ever come from the verified
        anchors. Entry-level paths may be given relative to the project
        root and are normalized first.
        """
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in self.first_read:
            return self.first_read[rel]
        if is_link_or_reparse(path) or is_link_or_reparse(path.parent):
            raise IdentityError(f"{rel}: link or reparse point")
        resolved = path.resolve()
        root = PROJECT_ROOT.resolve()
        if root != resolved and root not in resolved.parents:
            raise IdentityError(f"{rel}: escapes the project root")
        if not path.is_file():
            raise IdentityError(f"{rel}: missing at first read")
        digest = sha256_of_file(path)
        size = path.stat().st_size
        if binding is not None:
            if digest != binding["expected_sha256"]:
                raise IdentityError(
                    f"{rel}: bytes sha256 {digest} != frozen expectation "
                    f"{binding['expected_sha256']} from {binding['source']}; "
                    "refusing to consume")
            if binding.get("expected_bytes") is not None and \
                    size != int(binding["expected_bytes"]):
                raise IdentityError(
                    f"{rel}: bytes {size} != frozen expectation "
                    f"{binding['expected_bytes']} from {binding['source']}")
        record = {
            "path": rel,
            "role": role,
            "bytes": size,
            "sha256": digest,
            "pinned_sha256": pinned,
            "binding": binding,
            "recorded_at": "first_read",
        }
        self.first_read[rel] = record
        return record

    def snapshot_after(self) -> list[dict]:
        """Fresh end-of-run hash of every first-read input."""
        records = []
        for rel, meta in sorted(self.first_read.items()):
            path = PROJECT_ROOT / rel
            if not path.is_file():
                raise IdentityError(f"{rel}: missing at end-of-run snapshot")
            records.append({
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_of_file(path),
                "role": meta["role"],
                "pinned_sha256": meta.get("pinned_sha256"),
            })
        return records


# ---------------------------------------------------------------------------
# Stage 1: source verification


def verify_sources(ctx: RunContext) -> None:
    cfg = ctx.config
    for name, anchor in cfg["anchors"].items():
        path = PROJECT_ROOT / anchor["path"]
        if is_link_or_reparse(path):
            raise IdentityError(f"anchor {name} is a link/reparse point")
        digest = sha256_of_file(path)
        if digest != anchor["sha256"]:
            raise IdentityError(
                f"anchor {name} ({anchor['path']}): bytes sha256 {digest} != "
                f"pinned {anchor['sha256']}")
        ctx.consume(path, f"anchor:{name}", pinned=anchor["sha256"],
                    binding={"source": "config anchors",
                             "expected_sha256": anchor["sha256"]})
        log(f"anchor verified: {name} = {digest[:16]}…")

    frozen = yaml.safe_load(
        (PROJECT_ROOT / cfg["anchors"]["frozen_runtime_config"]["path"])
        .read_text(encoding="utf-8"))
    ctx.frozen_runtime = frozen
    artifact_manifest = json.loads(
        (PROJECT_ROOT / cfg["anchors"]["artifact_manifest"]["path"])
        .read_text(encoding="utf-8"))
    ctx.artifact_manifest = artifact_manifest
    ctx.protocol = json.loads(
        (PROJECT_ROOT / cfg["anchors"]["protocol"]["path"])
        .read_text(encoding="utf-8"))

    # cross-check the diagnostic pins against the verified frozen config
    string_keys = ("matcher", "mode")
    numeric_keys = ("scale", "max_disp", "block_size", "uniqueness_ratio",
                    "speckle_window", "speckle_range", "texture_threshold",
                    "lr_check_threshold_px", "min_valid_ratio", "band_margin")
    for key in string_keys:
        if str(cfg["stereo"][key]) != str(frozen["stereo"][key]):
            raise IdentityError(
                f"stereo.{key} pin {cfg['stereo'][key]!r} != frozen runtime "
                f"{frozen['stereo'][key]!r}")
    for key in numeric_keys:
        mine = cfg["stereo"][key]
        theirs = frozen["stereo"].get(key)
        if float(mine) != float(theirs):
            raise IdentityError(
                f"stereo.{key} pin {mine!r} != frozen runtime {theirs!r}")
    if float(cfg["stereo"]["full_image_threshold"]) != 0.9:
        raise IdentityError("stereo.full_image_threshold pin must be 0.9")
    for key in ("gain", "epsilon", "min_intensity_sum"):
        if float(cfg["polarization"][key]) != float(frozen["polarization"][key]):
            raise IdentityError(
                f"polarization.{key} pin != frozen runtime")
    if list(cfg["class_names"]) != list(frozen["candidate"]["class_names"]):
        raise IdentityError("class order pin != frozen runtime class order")
    if int(cfg["crop_padding"]) != 10:
        raise IdentityError("crop padding pin must be 10 (invariant 4)")

    # --- frozen bindings resolved BEFORE any stage reads the bound files --
    # (a) runtime evidence: path/length/sha256 pinned by the VERIFIED
    #     artifact_manifest (a byte-verified anchor); expectations come
    #     from that manifest, never from current re-hashes.
    manifest_dir = (PROJECT_ROOT / cfg["anchors"]["artifact_manifest"]["path"]
                   ).parent
    manifest_files = artifact_manifest.get("files")
    if not isinstance(manifest_files, dict) or not manifest_files:
        raise IdentityError(
            "artifact_manifest.files is missing or empty; cannot bind the "
            "runtime evidence identities")
    runtime_bindings = {}
    for key, manifest_rel in (
            ("runtime_frames", "evaluate/frames_complete.jsonl"),
            ("runtime_predictions", "evaluate/predictions_complete.jsonl"),
            ("runtime_layer_b_summary", "evaluate/layer_b_summary.json")):
        entry = manifest_files.get(manifest_rel)
        if not isinstance(entry, dict) or "sha256" not in entry \
                or "bytes" not in entry:
            raise IdentityError(
                f"artifact_manifest has no usable entry for {manifest_rel!r} "
                f"(needed for sources.{key}); refusing to consume it "
                "unbound")
        declared = manifest_dir / manifest_rel
        configured = (PROJECT_ROOT / cfg["sources"][key]).resolve()
        if declared.resolve() != configured:
            raise IdentityError(
                f"sources.{key} {cfg['sources'][key]!r} does not resolve to "
                f"the artifact_manifest entry {manifest_rel!r} "
                f"({declared})")
        runtime_bindings[key] = {
            "source": f"artifact_manifest.files[{manifest_rel!r}]",
            "expected_sha256": str(entry["sha256"]),
            "expected_bytes": int(entry["bytes"]),
        }
    # (b) local feature manifest: sha256 pinned by the VERIFIED frozen
    #     runtime YAML (features.local_cache.files).
    local_cache = frozen.get("features", {}).get("local_cache", {})
    local_files = local_cache.get("files", {})
    local_entry = local_files.get("feature_manifest.json")
    if not isinstance(local_entry, str) or not local_entry:
        raise IdentityError(
            "frozen runtime YAML features.local_cache.files has no "
            "sha256 entry for feature_manifest.json; refusing to consume "
            "it unbound")
    local_declared = (PROJECT_ROOT / str(local_cache["path"]) /
                      "feature_manifest.json")
    local_configured = (PROJECT_ROOT /
                        cfg["sources"]["local_feature_manifest"]).resolve()
    if local_declared.resolve() != local_configured:
        raise IdentityError(
            f"sources.local_feature_manifest "
            f"{cfg['sources']['local_feature_manifest']!r} does not resolve "
            f"to the frozen local cache path ({local_declared})")

    # P0 oracle identities are pinned in the frozen config; bind them here
    oracles = cfg["sources"]["p0_oracles"]
    for fold, rel in sorted(oracles.items()):
        pinned = frozen["p0_models"][fold]["oracle"]["sha256"]
        frozen_path = frozen["p0_models"][fold]["oracle"]["path"]
        if Path(frozen_path).as_posix() != Path(rel).as_posix():
            raise IdentityError(
                f"p0 oracle {fold}: config path {rel!r} != frozen {frozen_path!r}")
        ctx.consume(PROJECT_ROOT / rel, f"p0_oracle:{fold}", pinned=pinned,
                    binding={"source": f"frozen_runtime_config.p0_models."
                                        f"{fold}.oracle",
                             "expected_sha256": pinned})
        digest = sha256_of_file(PROJECT_ROOT / rel)
        if digest != pinned:
            raise IdentityError(
                f"p0 oracle {fold}: bytes sha256 {digest} != frozen pin "
                f"{pinned}")

    static_roles = {
        "runtime_frames": ("runtime_frames_complete",
                           runtime_bindings["runtime_frames"]),
        "runtime_predictions": ("runtime_predictions_complete",
                                runtime_bindings["runtime_predictions"]),
        "runtime_layer_b_summary": ("runtime_layer_b_summary",
                                    runtime_bindings["runtime_layer_b_summary"]),
        "local_feature_manifest": (
            "local_feature_manifest",
            {"source": "frozen_runtime_config.features.local_cache.files",
             "expected_sha256": local_entry}),
    }
    for key, (role, binding) in static_roles.items():
        ctx.consume(PROJECT_ROOT / cfg["sources"][key], role, binding=binding)
    for key, role in (("detection_dir", None), ("atlas_manifest", None),
                      ("split_csv", None)):
        if key == "detection_dir":
            ctx.consume(PROJECT_ROOT / cfg["sources"]["detection_dir"] /
                        "summary.json", "detection_summary")
        else:
            ctx.consume(PROJECT_ROOT / cfg["sources"][key], role or key)
    for cond in ("D0", "D1", "D2", "D3"):
        for name in ("per_gt.jsonl", "predictions.jsonl", "unmatched_gt.jsonl",
                     "unmatched_predictions.jsonl", "frames.jsonl"):
            ctx.consume(
                PROJECT_ROOT / cfg["sources"]["detection_dir"] / "conditions"
                / cond / name, f"detection_{cond}_{name}")
    if ctx.protocol.get("protocol_sha256") is None:
        raise IdentityError("protocol.json lacks its own protocol_sha256")
    expectations = cfg["protocol_expectations"]
    inventory_frames = ctx.protocol["inventory"]["frames"]
    val_frames = [f for f in inventory_frames if f["split"] == "val"]
    if len(val_frames) != int(expectations["val_frames"]):
        raise IdentityError(
            f"protocol val frames {len(val_frames)} != pinned "
            f"{expectations['val_frames']}")
    dev_splits = set(ctx.protocol["data"]["dev_pool_splits"])
    if dev_splits != set(expectations["dev_splits"]):
        raise IdentityError(f"unexpected dev pool splits: {dev_splits!r}")

    unbound_roles = sorted(
        r["role"] for r in ctx.first_read.values() if not r["binding"])
    ctx.stage("verify", "COMPLETED",
              anchors_verified=sorted(cfg["anchors"]),
              protocol_val_frames=len(val_frames),
              frozen_bindings=sorted(
                  r["role"] for r in ctx.first_read.values() if r["binding"]),
              unbound_disclosed=unbound_roles)


# ---------------------------------------------------------------------------
# Stage 2: frozen selection


def atlas_control_frames(ctx: RunContext) -> list[str]:
    atlas_dir = PROJECT_ROOT / ctx.config["sources"]["atlas_dir"]
    class_names = tuple(ctx.config["class_names"])
    frames = []
    for panel in sorted(atlas_dir.glob("control_*.png")):
        name = panel.stem[len("control_"):]
        for cls in class_names:
            if name.startswith(cls + "_"):
                name = name[len(cls) + 1:]
                break
        else:
            raise IdentityError(f"unparseable control panel name: {panel.name}")
        suffix = "_gt"
        if suffix not in name:
            raise IdentityError(f"control panel lacks _gt marker: {panel.name}")
        frames.append(name[:name.rfind(suffix)])
    return frames


def run_selection(ctx: RunContext) -> None:
    manifest_path = PROJECT_ROOT / ctx.config["sources"]["local_feature_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_complete = read_jsonl(PROJECT_ROOT / ctx.config["sources"]["runtime_frames"])
    r0_rows = [row for row in
               read_jsonl(PROJECT_ROOT / ctx.config["sources"]["runtime_predictions"])
               if row.get("candidate_id") == "R0"]
    join = pipeline_gt_prediction_join(frames_complete, r0_rows)
    m_lt1_frames = sorted({e["frame_id"] for e in join["errors"]
                           if not e["m_ge_1"]})
    selection = select_diagnostic_frames(
        ctx.protocol["inventory"]["frames"],
        atlas_control_frames(ctx),
        manifest["samples"],
        m_lt1_frames,
        max_frames=ctx.max_frames,
    )
    ctx.selection = selection

    split_rows = _read_split_csv(ctx)
    by_stem = {row["output_stem"]: row for row in split_rows}
    missing = [f["frame_id"] for f in selection["frames"]
               if f["frame_id"] not in by_stem]
    if missing:
        raise IdentityError(
            f"selected frames absent from the split CSV: {missing[:5]}")

    # bind per-frame consumed inputs (images, labels, D0 masks, panels)
    det = PROJECT_ROOT / ctx.config["sources"]["detection_dir"]
    for frame in selection["frames"]:
        fid = frame["frame_id"]
        row = by_stem[fid]
        ctx.consume(PROJECT_ROOT / row["left_path"], f"image_left:{fid}")
        ctx.consume(PROJECT_ROOT / row["right_path"], f"image_right:{fid}")
        ctx.consume(
            PROJECT_ROOT / ctx.config["sources"]["label_root"] / frame["split"]
            / f"{fid}.txt", f"gt_label:{fid}")
        if frame["split"] == "val":
            ctx.consume(det / "conditions" / "D0" / "masks" / f"{fid}.npz",
                        f"d0_masks:{fid}")
    for panel in sorted(
            (PROJECT_ROOT / ctx.config["sources"]["atlas_dir"]).glob("*.png")):
        if panel.stem.startswith(("unmatched_gt_", "control_")):
            ctx.consume(panel, f"atlas_panel:{panel.stem}")

    payload = {
        "selection_rules": ctx.config["selection"]["required_rules"],
        "fill_rule": ctx.config["selection"]["fill_rule"],
        "max_frames_cap": ctx.config["selection"]["max_frames_cap"],
        "max_frames_requested": ctx.max_frames,
        "pipeline_m_below_1_frames": m_lt1_frames,
        **selection,
    }
    dump_json(ctx.output_dir / "selection_manifest.json", payload)
    ctx.stage("selection", "COMPLETED",
              required=selection["required_count"],
              selected=selection["selected_count"],
              groups_covered=selection["groups_covered"],
              groups_total=selection["groups_total"])


def _read_split_csv(ctx: RunContext) -> list[dict]:
    path = PROJECT_ROOT / ctx.config["sources"]["split_csv"]
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Stage 3: saved-evidence statistics (task A)


def run_saved_stage(ctx: RunContext) -> None:
    cfg = ctx.config
    manifest_path = PROJECT_ROOT / cfg["sources"]["local_feature_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_and_merge_p0_oracles(
        {fold: str(PROJECT_ROOT / rel) for fold, rel in
         cfg["sources"]["p0_oracles"].items()},
        manifest["samples"])
    # attach has_valid_s from the manifest (join by sample_name, verified)
    by_name = {s["sample_name"]: s for s in manifest["samples"]}
    if len(by_name) != len(manifest["samples"]):
        raise IdentityError("duplicate sample_name in the feature manifest")
    for row in rows:
        row["has_valid_s"] = bool(by_name[row["sample_name"]]["has_valid_s"])
    stats = residual_capacity_stats(rows)
    for key in ("fold0", "fold1", "fold2"):
        bucket = stats["per_fold"].get(key)
        if bucket is None:
            raise IdentityError(f"outer-eval fold {key} produced no rows")
    join_frames = read_jsonl(PROJECT_ROOT / cfg["sources"]["runtime_frames"])
    join_preds = read_jsonl(PROJECT_ROOT / cfg["sources"]["runtime_predictions"])
    r0_rows = [row for row in join_preds if row.get("candidate_id") == "R0"]
    join = pipeline_gt_prediction_join(join_frames, r0_rows)

    summary = {
        "stage": "saved_residual_capacity",
        "oracle_identity": {
            fold: {"path": rel,
                   "sha256": sha256_of_file(PROJECT_ROOT / rel)}
            for fold, rel in cfg["sources"]["p0_oracles"].items()},
        "outer_eval": stats,
        "pipeline_join": join,
        "reading_notes": [
            "outer_eval rows are the frozen P0 on V5 GT crops; the pipeline "
            "join rows are the runtime R0 on Model A predicted instances - "
            "different inputs, both reported separately",
            "m >= 1 is a per-sample accuracy capacity bound under the frozen "
            "residual rule; no Macro-F1 bound and no realized gain follows",
            "missing-S outer-eval rows fall back to the exact frozen P0, so "
            "the residual cannot act on them regardless of m",
        ],
    }
    dump_json(ctx.output_dir / "residual_capacity_summary.json", summary)
    dump_jsonl(ctx.output_dir / "residual_capacity_samples.jsonl", [
        {"sample_name": r["sample_name"], "fold": r["fold"],
         "group": r["group"], "true_class": r["class_id"],
         "prediction": r["prediction"], "logits": r["logits"],
         "has_valid_s": r["has_valid_s"]}
        for r in rows])
    dump_jsonl(ctx.output_dir / "pipeline_join_errors.jsonl", join["errors"])
    totals = {**stats["totals"], "pipeline": join["totals"]}
    log(f"saved stats: {json.dumps(totals)}")
    ctx.stage("saved", "COMPLETED", totals=totals)


# ---------------------------------------------------------------------------
# Stage 4: synthetic diagnostics (task B)


def run_synthetic_stage(ctx: RunContext) -> None:
    cfg = ctx.config
    stereo_cfg = _stereo_config(cfg)
    scenarios = run_synthetic_scenarios(
        stereo_cfg, w_min=float(cfg["interventions"]["upsample_control"]["w_min"]),
        seed=int(cfg["synthetic"]["seed"]))
    # serialise: strip array payloads, keep scalars
    out: dict[str, Any] = {}
    for name, result in scenarios.items():
        cleaned: dict[str, Any] = {}
        for key, value in result.items():
            if isinstance(value, dict):
                cleaned[key] = {
                    k: (v if not isinstance(v, np.ndarray) else "<array>")
                    for k, v in value.items()}
            else:
                cleaned[key] = value
        out[name] = cleaned
    report = {
        "stage": "synthetic_upsample_chain",
        "ground_truth": "independent analytic correspondence; expected "
                        "values are derived in the tests, never from the "
                        "code under test",
        "upsample_methods": [PRODUCTION_UPSAMPLE, "valid_weighted"],
        "w_min": float(cfg["interventions"]["upsample_control"]["w_min"]),
        "scenarios": out,
        "reading_notes": [
            "the depth-discontinuity scenario shows production and control "
            "are IDENTICAL when no invalid low-res pixels exist: the "
            "control does not repair valid-valid boundary mixing",
            "the in-situ scenario attributes each biased valid pixel: a "
            "control-restored value indicates zero-fill contribution; an "
            "unchanged value indicates SGBM's own low-res edge response",
        ],
    }
    dump_json(ctx.output_dir / "synthetic_report.json", report)
    for name, result in out.items():
        scalars = {k: v for k, v in result.items() if not isinstance(v, dict)}
        log(f"synthetic {name}: {json.dumps(scalars)}")
    ctx.stage("synthetic", "COMPLETED", scenarios=sorted(out))


# ---------------------------------------------------------------------------
# Stage 5: real-image recomputation (task D)


class _ModelAProbe:
    """Recording probe injected as Model A: the diagnostic must never run
    detection, so any call is recorded and then raised, never swallowed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, image: np.ndarray, **kwargs: Any) -> list:
        self.calls.append({"shape": list(np.asarray(image).shape),
                           "kwargs": sorted(kwargs)})
        raise AssertionError(
            "Model A probe was invoked; the measurement diagnostic must "
            "reuse saved masks/GT geometry instead of running detection")


def _stereo_config(cfg: dict) -> StereoMatcherConfig:
    s = cfg["stereo"]
    return StereoMatcherConfig(
        matcher=str(s["matcher"]), mode=str(s["mode"]),
        max_disparity=int(s["max_disp"]), scale=float(s["scale"]),
        block_size=int(s["block_size"]),
        uniqueness_ratio=int(s["uniqueness_ratio"]),
        speckle_window=int(s["speckle_window"]),
        speckle_range=int(s["speckle_range"]),
        texture_threshold=float(s["texture_threshold"]),
        lr_check_threshold_px=float(s["lr_check_threshold_px"]),
        min_valid_ratio=float(s["min_valid_ratio"]))


def _load_condition_masks(path: Path) -> list[np.ndarray]:
    """The saved D0 prediction masks (bit-packed npz), same decoding as
    scripts/diagnose_existing_detection._load_condition_masks."""
    with np.load(path) as payload:
        shape = tuple(int(v) for v in payload["shape"])
        packed = payload["masks"]
        count = int(payload["confidences"].shape[0])
        bboxes = payload["bboxes"]
        confidences = payload["confidences"]
    if count == 0:
        return []
    bits = np.unpackbits(packed, axis=1)[:, :shape[0] * shape[1]]
    return [
        (bits[i].reshape(shape).astype(np.uint8),
         tuple(int(v) for v in bboxes[i]), float(confidences[i]))
        for i in range(count)
    ]


class _WorstCaseTracker:
    """Keeps the crops of the current worst (frame, instance) per
    comparison kind, ranked by the instance's max |signed dq| on its
    common region. Crops stay small (instance neighbourhood only)."""

    def __init__(self) -> None:
        self.best: dict[str, dict] = {}

    def offer(self, kind: str, score: float, payload: dict) -> None:
        current = self.best.get(kind)
        if current is None or score > current["score"]:
            self.best[kind] = {"score": score, **payload}


def _crop_worst_payload(left, right, instance_mask, stereo0, polar0, stereo,
                        polar, margin: int = 24) -> dict:
    """Crop the instance neighbourhood out of both arms' measurement maps
    for the worst-case panel.

    The peak pixel is restricted to the instance's COMMON region
    ``instance_mask & baseline_qvalid & arm_qvalid``: a larger difference
    outside that region (invalid on either side, or outside the mask) is
    not a paired difference and must never be selected. An empty common
    region yields no peak coordinates at all."""
    ys, xs = np.nonzero(instance_mask > 0)
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(left.shape[0], int(ys.max()) + 1 + margin)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(left.shape[1], int(xs.max()) + 1 + margin)
    sl = np.s_[y1:y2, x1:x2]
    base_q = polar0.signed_q[sl]
    arm_q = polar.signed_q[sl]
    base_qvalid = polar0.valid_mask[sl] > 0
    arm_qvalid = polar.valid_mask[sl] > 0
    inst = instance_mask[sl] > 0
    common = inst & base_qvalid & arm_qvalid
    abs_dq = np.abs(arm_q.astype(np.float64) - base_q.astype(np.float64))
    masked_abs_dq = np.where(common, abs_dq, 0.0)
    if common.any():
        peak = np.unravel_index(int(np.argmax(masked_abs_dq)),
                                masked_abs_dq.shape)
        peak_pixel_global = [int(x1 + peak[1]), int(y1 + peak[0])]
        peak_abs_dq = float(abs_dq[peak])
    else:
        peak_pixel_global = None
        peak_abs_dq = None
    return {
        "bbox": [x1, y1, x2, y2],
        "gray": left[sl],
        "instance_mask": inst.astype(np.uint8),
        "base_q": base_q,
        "base_qvalid": base_qvalid,
        "arm_q": arm_q,
        "arm_qvalid": arm_qvalid,
        "common_region": common,
        "abs_dq": masked_abs_dq,
        "peak_pixel_global": peak_pixel_global,
        "peak_abs_dq": peak_abs_dq,
    }


def _measure_condition(
    ctx: RunContext | None,
    *,
    stereo_cfg: StereoMatcherConfig,
    engine: Any,
    probe: _ModelAProbe,
    left: np.ndarray,
    right: np.ndarray,
    instances: list,
    crop_windows: dict[int, tuple[int, int, int, int]] | None,
    w_min: float,
    shifts: list[tuple[int, int]],
    engine_check: bool,
    band_margin: int = 20,
    frame_id: str = "",
    conditioning: str = "unconditioned",
    trackers: "_WorstCaseTracker | None" = None,
) -> dict[str, Any]:
    """One conditioning (GT or prediction) through all measurement arms.

    Masks, bands and crop windows are FIXED across arms; each arm changes
    exactly one declared factor. The engine run is the production
    reference; the orchestration baseline is cross-checked against it
    bitwise so arm differences are attributable to the intended factor.
    Per-instance paired statistics (fixed regions, common validity) are
    computed for every arm; ``trackers`` collects worst-case crops.
    """
    h, w = left.shape
    bboxes = [inst.bbox for inst in instances]
    bands = build_horizontal_bands(
        bboxes, image_height=h, vertical_margin=int(band_margin))
    production_matcher = StereoMatcher(stereo_cfg)

    # --- arm E: the production engine (reference) -------------------------
    engine_record: dict[str, Any] = {"ran": False}
    if engine_check:
        probe_start = len(probe.calls)
        engine.matcher = production_matcher
        eng = engine.process_instances_detailed(
            left, right, instances, already_rectified=True,
            frame_id=frame_id, crop_windows=crop_windows)
        engine_record = {
            "ran": True,
            "model_a_probe_calls": len(probe.calls) - probe_start,
            "n_instances": len(eng.instances),
        }

    # --- arm 0: orchestration baseline (production upsample semantics) ---
    base_matcher = DiagnosticUpsampleMatcher(
        stereo_cfg, method=PRODUCTION_UPSAMPLE)
    stereo0 = base_matcher.compute_bands(left, right, bands)
    stats0 = [base_matcher.instance_stats(stereo0, inst.mask, instance_id=inst.id)
              for inst in instances]
    polar_mask0 = np.zeros((h, w), dtype=bool)
    for inst, st in zip(instances, stats0):
        if st.valid:
            polar_mask0 |= inst.mask > 0
    polar0 = None
    if polar_mask0.any():
        polar0 = compute_polar_features(
            left, right, stereo0.disparity, object_mask=polar_mask0,
            disparity_valid=stereo0.valid)

    # engine cross-check: bitwise identity of the whole measurement chain
    crosscheck: dict[str, Any] = {"required": bool(engine_check)}
    if engine_check:
        eng_polar = eng.polar_result
        checks = {
            "engine_polar_result_present": eng_polar is not None,
            "orchestration_polar_present": polar0 is not None,
        }
        checks["instance_stats_equal"] = _depth_records_equal(
            eng.depths, stats0)
        checks["model_a_never_called"] = (
            len(probe.calls) == probe_start)
        if eng_polar is None and polar0 is None:
            # no measurable instance in this frame under either path: the q
            # chain cannot be compared, the depth records can
            checks["both_polar_none"] = True
            identical = checks["instance_stats_equal"] \
                and checks["model_a_never_called"]
        elif eng_polar is not None and polar0 is not None:
            checks["signed_q_bitwise"] = bool(np.array_equal(
                eng_polar.signed_q, polar0.signed_q))
            checks["abs_q_bitwise"] = bool(np.array_equal(
                eng_polar.abs_q, polar0.abs_q))
            checks["valid_mask_bitwise"] = bool(np.array_equal(
                eng_polar.valid_mask, polar0.valid_mask))
            identical = (checks["signed_q_bitwise"] and checks["abs_q_bitwise"]
                         and checks["valid_mask_bitwise"]
                         and checks["instance_stats_equal"]
                         and checks["model_a_never_called"])
        else:
            checks["polar_presence_mismatch"] = True
            identical = False
        crosscheck["checks"] = checks
        crosscheck["identical"] = bool(identical)
        engine_record["crosscheck"] = crosscheck

    # --- arm 1: only the upsample swapped ---------------------------------
    wmatcher = DiagnosticUpsampleMatcher(
        stereo_cfg, method="valid_weighted", w_min=w_min)
    stereo1 = wmatcher.compute_bands(left, right, bands)
    polar1 = None
    if polar_mask0.any():
        polar1 = compute_polar_features(
            left, right, stereo1.disparity, object_mask=polar_mask0,
            disparity_valid=stereo1.valid)

    arms: dict[str, dict[str, Any]] = {}
    interiors: dict[int, np.ndarray] = {}
    boundaries: dict[int, np.ndarray] = {}
    for inst in instances:
        interior, boundary = instance_boundary_split(inst.mask)
        interiors[inst.id] = interior
        boundaries[inst.id] = boundary

    def paired_records(stereo, polar):
        """Per-instance paired statistics on fixed regions (empty common
        regions report n=0 with null statistics)."""
        out = []
        for inst in instances:
            if polar is None or polar0 is None:
                out.append({
                    "instance_id": int(inst.id), "n": 0,
                    "reason": "polar maps absent on one side"})
                continue
            paired = instance_paired_comparison(
                inst.mask, interiors[inst.id], boundaries[inst.id],
                stereo0.disparity, stereo0.valid,
                polar0.signed_q, polar0.abs_q, polar0.valid_mask,
                stereo.disparity, stereo.valid,
                polar.signed_q, polar.abs_q, polar.valid_mask)
            paired["instance_id"] = int(inst.id)
            out.append(paired)
        return out

    def offer_worst(kind: str, arm_key: str, stereo, polar,
                    paired_list: list[dict]) -> None:
        if trackers is None or not paired_list:
            return
        worst = max(
            (p for p in paired_list if p.get("signed_dq", {}).get("n", 0) > 0),
            key=lambda p: p["signed_dq"]["abs_max"], default=None)
        if worst is None:
            return
        inst = next(i for i in instances
                    if int(i.id) == int(worst["instance_id"]))
        crop = _crop_worst_payload(left, right, inst.mask,
                                   stereo0, polar0, stereo, polar)
        # the recorded peak must be the instance's common-region maximum:
        # the payload peak and the paired abs_max cover the same pixels
        assert crop["peak_abs_dq"] is not None and abs(
            crop["peak_abs_dq"] - worst["signed_dq"]["abs_max"]) <= 1e-9, \
            "worst-case peak drifted from the paired common-region maximum"
        trackers.offer(kind, worst["signed_dq"]["abs_max"], {
            "frame_id": frame_id, "conditioning": conditioning,
            "arm": arm_key, "instance_id": int(inst.id),
            "score_abs_dq_max": worst["signed_dq"]["abs_max"],
            "paired": worst,
            "crop": crop,
        })

    def arm_record(name, stereo, polar, *, note=None, arm_key=None):
        arm_key = arm_key or name
        record: dict[str, Any] = {
            "valid_pixels": int((stereo.valid > 0).sum()),
            "valid_ratio_frame": float((stereo.valid > 0).mean()),
            "polar_valid_pixels": int(polar.valid_mask.sum()) if polar else 0,
        }
        if polar is not None and polar0 is not None:
            # frame/band-level context only; per-instance paired statistics
            # below are the reported comparison
            record["vs_baseline_frame_context"] = compare_measurements(
                stereo0.disparity, stereo0.valid, polar0.signed_q,
                polar0.valid_mask, stereo.disparity, stereo.valid,
                polar.signed_q, polar.valid_mask)
        record["per_instance"] = _per_instance_records(instances, stereo, polar)
        record["paired"] = paired_records(stereo, polar)
        record["note"] = note or ""
        offer_worst(f"{conditioning}|{arm_key}", arm_key, stereo, polar,
                    record["paired"])
        return record

    arms["upsample_control"] = arm_record(
        "upsample_control", stereo1, polar1,
        note="only the final upsample swapped; masks/bands/crops identical")

    # --- arms 3/4: right-image +-1px shifts --------------------------------
    # The unmeasurable-edge mask MUST come from the disparity actually
    # sampled by each arm's warp: full_recompute samples with the
    # re-matched stereo3.disparity, fixed_baseline_warp with the baseline
    # disparity. Fill content never counts as valid q on either arm.
    shift_records: dict[str, Any] = {}
    for dx, dy in shifts:
        shifted, fill = shift_right_image(right, dx, dy)
        # arm 3: full re-matching on the shifted right image
        stereo3 = production_matcher.compute_bands(left, shifted, bands)
        edge3 = edge_unmeasurable_mask((h, w), stereo3.disparity, dx, dy)
        polar3 = None
        if polar_mask0.any():
            polar3 = compute_polar_features(
                left, shifted, stereo3.disparity, object_mask=polar_mask0,
                disparity_valid=stereo3.valid & ~edge3)
        key = f"shift_dx{dx:+d}_dy{dy:+d}".replace("+", "p").replace("-", "m")
        full = arm_record(
            key, stereo3, polar3,
            note="re-matched SGBM + warp + q on the shifted right image; "
                 "edge mask from the re-matched disparity",
            arm_key=f"{key}:full_recompute")
        full["edge_mask_source"] = "stereo3.disparity (re-matched)"
        full["edge_excluded_pixels"] = int((edge3 & stereo3.valid).sum())
        shift_records[key] = {"shift": [dx, dy], "full_recompute": full}
        # arm 4: baseline disparity fixed, only warp + q recomputed
        edge4 = edge_unmeasurable_mask((h, w), stereo0.disparity, dx, dy)
        polar4 = None
        if polar_mask0.any():
            polar4 = compute_polar_features(
                left, shifted, stereo0.disparity, object_mask=polar_mask0,
                disparity_valid=stereo0.valid & ~edge4)
        fixed = {
            "shift": [dx, dy],
            "valid_pixels": int(polar4.valid_mask.sum()) if polar4 else 0,
            "vs_baseline_frame_context": (
                compare_measurements(
                    stereo0.disparity, stereo0.valid, polar0.signed_q,
                    polar0.valid_mask, stereo0.disparity, stereo0.valid,
                    polar4.signed_q, polar4.valid_mask)
                if polar4 is not None and polar0 is not None else {}),
            "per_instance": _per_instance_records(instances, stereo0, polar4),
            "paired": paired_records(stereo0, polar4),
            "edge_mask_source": "stereo0.disparity (baseline)",
            "edge_excluded_pixels": int((edge4 & stereo0.valid).sum()),
            "note": "baseline disparity held fixed; only warp sampling and "
                    "q recomputed on the shifted right image; edge mask "
                    "from the baseline disparity",
        }
        offer_worst(f"{conditioning}|{key}:fixed_baseline_warp",
                    f"{key}:fixed_baseline_warp",
                    stereo0, polar4, fixed["paired"])
        shift_records[key]["fixed_baseline_warp"] = fixed
        shift_records[key]["fill_pixels"] = int(fill.sum())

    return {
        "bands": [[int(b.y1), int(b.y2)] for b in bands],
        "baseline": {
            "valid_pixels": int((stereo0.valid > 0).sum()),
            "polar_valid_pixels": int(polar0.valid_mask.sum()) if polar0 else 0,
            "per_instance": _per_instance_records(instances, stereo0, polar0),
        },
        "engine_reference": engine_record,
        "arms": arms,
        "shifts": shift_records,
    }


def _depth_records_equal(engine_depths: list[dict], stats: list) -> bool:
    if len(engine_depths) != len(stats):
        return False
    for rec, st in zip(engine_depths, stats):
        if int(rec["instance_id"]) != int(st.instance_id):
            return False
        if float(rec["disparity"]) != round(st.disparity, 2):
            return False
        if float(rec["valid_ratio"]) != round(st.valid_ratio, 3):
            return False
        if rec["reason"] != st.reason:
            return False
    return True


def _per_instance_records(instances, stereo, polar) -> list[dict]:
    from core.measurement_diagnostic import instance_boundary_split

    records = []
    for inst in instances:
        mask = inst.mask > 0
        valid = mask & (stereo.valid > 0)
        interior, boundary = instance_boundary_split(inst.mask)
        rec = {
            "instance_id": int(inst.id),
            "class_name": getattr(inst, "class_name", None),
            "mask_pixels": int(mask.sum()),
            "disparity_valid_pixels": int(valid.sum()),
            "valid_ratio": float(valid.sum() / mask.sum()) if mask.any() else 0.0,
        }
        values = stereo.disparity[valid]
        rec["disparity"] = float(np.median(values)) if values.size else None
        if polar is not None:
            qv = polar.valid_mask & mask
            q = polar.signed_q[qv].astype(np.float64)
            abs_q = polar.abs_q[qv].astype(np.float64)
            rec["q_valid_pixels"] = int(qv.sum())
            rec["signed_q"] = _stats(q)
            rec["abs_q"] = _stats(abs_q)
            for label, region in (("interior", interior), ("boundary", boundary)):
                sel = qv & region
                rec[f"signed_q_{label}"] = _stats(polar.signed_q[sel].astype(np.float64))
                rec[f"abs_q_{label}"] = _stats(polar.abs_q[sel].astype(np.float64))
                rec[f"q_valid_pixels_{label}"] = int(sel.sum())
        else:
            for key in ("q_valid_pixels", "signed_q", "abs_q",
                        "signed_q_interior", "abs_q_interior",
                        "signed_q_boundary", "abs_q_boundary"):
                rec[key] = None
            rec["q_valid_pixels_interior"] = rec["q_valid_pixels_boundary"] = 0
        records.append(rec)
    return records


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None,
                "max_abs": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max_abs": float(np.max(np.abs(values))),
    }


# ---------------------------------------------------------------------------
# Stage 5 driver: real-image recomputation


def run_recompute_stage(ctx: RunContext) -> None:
    import cv2

    from gui.inference_engine import DualStageInferenceEngine

    cfg = ctx.config
    stereo_cfg = _stereo_config(cfg)
    calib = cfg["calibration"]
    probe = _ModelAProbe()
    engine = DualStageInferenceEngine(
        model_a=probe, model_b=None,
        matcher=StereoMatcher(stereo_cfg),
        rectifier=None, rectify_enabled=False,
        device="cpu",
        crop_padding=int(cfg["crop_padding"]),
        band_vertical_margin=int(cfg["stereo"]["band_margin"]),
        baseline=float(calib["baseline"]),
        focal_length=float(calib["focal_length"]),
    )
    w_min = float(cfg["interventions"]["upsample_control"]["w_min"])
    shifts = [tuple(int(v) for v in s)
              for s in cfg["interventions"]["right_shifts_px"]]
    split_rows = {row["output_stem"]: row for row in _read_split_csv(ctx)}
    det_dir = PROJECT_ROOT / cfg["sources"]["detection_dir"]
    d0_predictions: dict[str, dict] = {}
    for row in read_jsonl(det_dir / "conditions" / "D0" / "predictions.jsonl"):
        d0_predictions.setdefault(row["frame_id"], {})[
            int(row["prediction_index"])] = row

    records_path = ctx.output_dir / "recompute_instances.jsonl"
    frame_rows: list[dict] = []
    engine_check_failures: list[str] = []
    budget_exceeded = False
    trackers = _WorstCaseTracker()
    handle = records_path.open("w", encoding="utf-8")

    def emit_frame_instance(row: dict) -> None:
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"non-finite float in instance record {key!r}")
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    try:
        for index, frame in enumerate(ctx.selection["frames"]):
            ctx.watch.check(f"recompute frame {frame['frame_id']}")
            fid = frame["frame_id"]
            row = split_rows[fid]
            left_path = PROJECT_ROOT / row["left_path"]
            right_path = PROJECT_ROOT / row["right_path"]
            left_img = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
            right_img = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
            if left_img is None or right_img is None:
                raise IdentityError(
                    f"frame {fid}: unreadable left/right image (never treated "
                    "as a successful empty frame)")
            left = to_gray_u8(left_img)
            right = to_gray_u8(right_img)
            if left.shape != right.shape:
                raise IdentityError(f"frame {fid}: left/right shape mismatch")
            height, width = left.shape
            frame_record: dict[str, Any] = {
                "frame_id": fid,
                "group": frame["group"],
                "split": frame["split"],
                "image_size": [height, width],
                "image_source": {
                    "left": row["left_path"], "right": row["right_path"],
                    "already_rectified": bool(
                        cfg["image_source"]["already_rectified"]),
                    "re_rectified": False,
                    "exposure_gain_gamma": "NOT_AVAILABLE",
                    "sync_status": "unknown",
                },
            }

            # ---- GT-conditional measurement ------------------------------
            geometry = build_gt_geometry(
                PROJECT_ROOT / cfg["sources"]["label_root"] / frame["split"]
                / f"{fid}.txt", width, height,
                padding=int(cfg["crop_padding"]))
            if geometry:
                ctx.watch.check(f"recompute gt-arms {fid}")
                result = _measure_condition(
                    ctx, stereo_cfg=stereo_cfg, engine=engine, probe=probe,
                    left=left, right=right,
                    instances=gt_instances(geometry),
                    crop_windows=gt_crop_windows(geometry),
                    w_min=w_min, shifts=shifts, engine_check=True,
                    band_margin=int(cfg["stereo"]["band_margin"]),
                    frame_id=fid, conditioning="gt_conditional",
                    trackers=trackers)
                if not result["engine_reference"]["crosscheck"]["identical"]:
                    engine_check_failures.append(f"{fid}:gt")
                frame_record["gt_conditional"] = {
                    "mask_source": "gt_generation_geometry",
                    "n_instances": len(geometry),
                    **result,
                }
            else:
                frame_record["gt_conditional"] = {
                    "mask_source": "gt_generation_geometry",
                    "n_instances": 0,
                    "note": "no GT instances (no-target frame); no bands, "
                            "no SGBM, nothing measurable",
                }

            # ---- prediction-conditional measurement ----------------------
            pred_conditional = None
            if frame["split"] == "val":
                mask_file = det_dir / "conditions" / "D0" / "masks" / f"{fid}.npz"
                if mask_file.is_file():
                    payload = _load_condition_masks(mask_file)
                    if payload:
                        instances = []
                        for pred_id, (mask, bbox, conf) in enumerate(payload):
                            meta = d0_predictions.get(fid, {}).get(pred_id, {})
                            instances.append(_PredInstance(
                                pred_id, tuple(int(v) for v in bbox), mask,
                                conf, meta.get("matched_gt_id")))
                        ctx.watch.check(f"recompute pred-arms {fid}")
                        result = _measure_condition(
                            ctx, stereo_cfg=stereo_cfg, engine=engine,
                            probe=probe, left=left, right=right,
                            instances=instances, crop_windows=None,
                            w_min=w_min, shifts=shifts, engine_check=False,
                            band_margin=int(cfg["stereo"]["band_margin"]),
                            frame_id=fid, conditioning="pred_conditional",
                            trackers=trackers)
                        pred_conditional = {
                            "mask_source": "saved_d0_prediction_masks",
                            "n_instances": len(instances),
                            **result,
                        }
                    else:
                        pred_conditional = {
                            "mask_source": "saved_d0_prediction_masks",
                            "n_instances": 0,
                            "note": "no D0 predictions for this frame"}
            frame_record["pred_conditional"] = pred_conditional
            frame_rows.append(frame_record)
            for row in _frame_instance_rows(fid, frame_record):
                emit_frame_instance(row)
            handle.flush()
            log(f"recompute {index + 1}/{len(ctx.selection['frames'])}: {fid}")
    except BudgetExceeded:
        budget_exceeded = True
        log("budget exhausted inside the recompute frame loop; finished "
            "frames are kept")
    finally:
        handle.close()

    dump_jsonl(ctx.output_dir / "recompute_frames.jsonl", frame_rows)
    paired_summary = aggregate_paired_stats(frame_rows)
    dump_json(ctx.output_dir / "recompute_paired_summary.json", paired_summary)
    panels = _render_worst_case_panels(ctx.output_dir / "panels", trackers)
    if budget_exceeded:
        ctx.stage("recompute", "PARTIAL",
                  frames_completed=len(frame_rows),
                  frames_requested=len(ctx.selection["frames"]),
                  engine_check_failures=engine_check_failures,
                  panels=panels,
                  note="shared budget exhausted; remaining frames NOT run")
        raise BudgetExceeded("recompute stage stopped at the frame boundary")

    ctx.stage("recompute", "COMPLETED",
              frames=len(frame_rows),
              engine_check_failures=engine_check_failures,
              panels=panels,
              paired_summary_top=paired_summary["overall"])
    if engine_check_failures:
        raise IdentityError(
            f"engine cross-check failures: {engine_check_failures}")


class _PredInstance:
    """A saved D0 prediction rendered as an engine-compatible instance."""

    def __init__(self, pred_id: int, bbox: tuple, mask: np.ndarray,
                 confidence: float, matched_gt_id: int | None) -> None:
        from models.segmentation import Instance

        self.ref = Instance(
            id=pred_id, bbox=bbox, mask=mask, confidence=confidence,
            class_id=0, class_name="d0_prediction")
        self.matched_gt_id = matched_gt_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ref, name)


def _frame_instance_rows(fid: str, frame_record: dict) -> list[dict]:
    """One flat record per (conditioning, arm, instance) for the JSONL."""
    rows = []
    for conditioning in ("gt_conditional", "pred_conditional"):
        block = frame_record.get(conditioning)
        if not block or block.get("n_instances", 0) == 0:
            continue
        arms: dict[str, dict] = {"baseline": block["baseline"]}
        arms.update(block.get("arms", {}))
        for key, rec in block.get("shifts", {}).items():
            for sub in ("full_recompute", "fixed_baseline_warp"):
                if sub in rec:
                    arms[f"shift:{key}:{sub}"] = rec[sub]
        for arm_name, arm in arms.items():
            if arm_name != "baseline" and "per_instance" not in arm:
                continue
            paired = arm.get("paired") or []
            for inst, pr in zip(arm.get("per_instance", []), paired):
                rows.append({
                    "frame_id": fid,
                    "conditioning": conditioning,
                    "mask_source": block["mask_source"],
                    "arm": arm_name,
                    "instance_id": inst.get("instance_id"),
                    "class_name": inst.get("class_name"),
                    "group": frame_record.get("group"),
                    "mask_pixels": inst.get("mask_pixels"),
                    "disparity_valid_pixels": inst.get("disparity_valid_pixels"),
                    "valid_ratio": inst.get("valid_ratio"),
                    "disparity": inst.get("disparity"),
                    "q_valid_pixels": inst.get("q_valid_pixels"),
                    "signed_q_mean": (inst.get("signed_q") or {}).get("mean"),
                    "abs_q_mean": (inst.get("abs_q") or {}).get("mean"),
                    "abs_q_max": (inst.get("abs_q") or {}).get("max_abs"),
                    "paired": pr,
                })
    return rows


def aggregate_paired_stats(frame_rows: list[dict]) -> dict:
    """Aggregate per-instance paired records from the in-memory frame rows
    (the same data written to recompute_frames.jsonl / instances.jsonl).

    Two aggregations are reported with their weights stated explicitly:
    - pixel_weighted: sums over common pixels (sum of |dq| / sum of n);
    - instance_weighted: unweighted mean over instances with n>0.
    Both are descriptive aggregations of path differences only.
    """

    def collect():
        for row in frame_rows:
            for conditioning in ("gt_conditional", "pred_conditional"):
                block = row.get(conditioning) or {}
                if block.get("n_instances", 0) == 0:
                    continue
                arms: dict[str, dict] = {}
                arms.update(block.get("arms", {}))
                for key, rec in block.get("shifts", {}).items():
                    for sub in ("full_recompute", "fixed_baseline_warp"):
                        if sub in rec:
                            arms[f"shift:{key}:{sub}"] = rec[sub]
                for arm_name, arm in arms.items():
                    paired = arm.get("paired") or []
                    per_inst = arm.get("per_instance") or []
                    for pr, pi in zip(paired, per_inst):
                        yield conditioning, arm_name, row.get("group"), \
                            pi.get("class_name"), pr

    def summarise(records):
        """Means/percentiles use only records with a non-empty common
        region; validity counts and flip totals cumulate over ALL records
        (an instance whose common region is empty can still lose every
        baseline-valid pixel). Flip totals are cumulated over
        instance-by-intervention RECORDS — they are not unique pixel
        counts. The p95 field is the pixel-count-weighted mean of
        PER-INSTANCE p95 values, never a pooled pixel p95 (pooled pixel
        p95 is NOT_COMPUTED here)."""
        with_common = [r for r in records
                       if r.get("signed_dq", {}).get("n", 0) > 0]
        n_pix = sum(r["signed_dq"]["n"] for r in with_common)
        with_counts = [r for r in records if "q_valid_counts" in r]
        out = {
            "instances_total": len(records),
            "instances_with_common_region": len(with_common),
            "common_pixels_total": n_pix,
            "weighting": {
                "pixel_weighted": "sum(stat*n)/sum(n) over instances with "
                                  "n>0; means and percentiles use only "
                                  "these records",
                "instance_weighted": "unweighted mean over instances with "
                                     "n>0",
            },
            "flip_semantics": "lost/gained cumulate over instance-by-"
                              "intervention records (including records "
                              "with an empty common region); they are NOT "
                              "unique pixel counts",
            "pooled_pixel_p95": "NOT_COMPUTED (only the weighted mean of "
                                "per-instance p95 values is reported)",
        }
        out["q_lost_total"] = int(sum(
            r["q_valid_counts"]["lost"] for r in with_counts))
        out["q_gained_total"] = int(sum(
            r["q_valid_counts"]["gained"] for r in with_counts))
        if not with_common:
            out.update({"signed_dq_mean_pixel_weighted": None,
                        "abs_dq_mean_pixel_weighted": None,
                        "abs_dq_mean_instance_weighted": None,
                        "abs_dq_instance_p95_pixelcount_weighted": None,
                        "dabs_q_mean_pixel_weighted": None})
            return out
        out["signed_dq_mean_pixel_weighted"] = float(sum(
            r["signed_dq"]["signed_mean"] * r["signed_dq"]["n"]
            for r in with_common) / n_pix)
        out["abs_dq_mean_pixel_weighted"] = float(sum(
            r["signed_dq"]["abs_mean"] * r["signed_dq"]["n"]
            for r in with_common) / n_pix)
        out["abs_dq_instance_p95_pixelcount_weighted"] = float(sum(
            r["signed_dq"]["abs_p95"] * r["signed_dq"]["n"]
            for r in with_common) / n_pix)
        out["abs_dq_mean_instance_weighted"] = float(sum(
            r["signed_dq"]["abs_mean"] for r in with_common)
            / len(with_common))
        out["dabs_q_mean_pixel_weighted"] = float(sum(
            r["dabs_q"]["signed_mean"] * r["dabs_q"]["n"]
            for r in with_common) / n_pix)
        return out

    per_cond_arm: dict[str, dict[str, list]] = {}
    per_cond_class: dict[str, dict[str, list]] = {}
    per_cond_group: dict[str, dict[str, list]] = {}
    by_cond: dict[str, list] = {}
    mixed_arm: dict[str, list] = {}
    mixed_class: dict[str, list] = {}
    mixed_group: dict[str, list] = {}
    all_records = []
    for conditioning, arm_name, group, class_name, pr in collect():
        all_records.append(pr)
        by_cond.setdefault(conditioning, []).append(pr)
        per_cond_arm.setdefault(conditioning, {}).setdefault(
            arm_name, []).append(pr)
        mixed_arm.setdefault(arm_name, []).append(pr)
        if class_name:
            per_cond_class.setdefault(conditioning, {}).setdefault(
                class_name, []).append(pr)
            mixed_class.setdefault(class_name, []).append(pr)
        if group:
            per_cond_group.setdefault(conditioning, {}).setdefault(
                group, []).append(pr)
            mixed_group.setdefault(group, []).append(pr)
    return {
        "definition": {
            "common_region": "fixed instance_mask & baseline_qvalid & "
                             "arm_qvalid (disparity: & baseline_valid & "
                             "arm_valid)",
            "signed_dq": "arm_q - baseline_q per pixel; signed_mean is a "
                         "mean difference (positive and negative changes "
                         "cancel), abs_mean of the same difference is the "
                         "per-pixel MAE on the common region",
            "dabs_q": "arm_abs_q - baseline_abs_q per pixel; signed_mean is "
                      "a feature-magnitude change, not an error",
            "p95_field": "abs_dq_instance_p95_pixelcount_weighted is the "
                         "pixel-count-weighted mean of per-instance p95 "
                         "values; the pooled pixel p95 is NOT_COMPUTED",
            "flip_semantics": "lost/gained cumulate over instance-by-"
                              "intervention records; not unique pixels",
            "per_arm_means": "kept in recompute_instances.jsonl per arm; "
                             "mean differences must not be read as pixel "
                             "MAE",
        },
        "overall": summarise(all_records),
        "by_conditioning": {k: summarise(v) for k, v in sorted(by_cond.items())},
        "by_arm_per_conditioning": {
            cond: {arm: summarise(v) for arm, v in sorted(arms.items())}
            for cond, arms in sorted(per_cond_arm.items())},
        "by_class_per_conditioning": {
            cond: {cls: summarise(v) for cls, v in sorted(cls_map.items())}
            for cond, cls_map in sorted(per_cond_class.items())},
        "by_group_per_conditioning": {
            cond: {grp: summarise(v) for grp, v in sorted(grp_map.items())}
            for cond, grp_map in sorted(per_cond_group.items())},
        "overview_mixed_conditionings": {
            "note": "arms/classes/groups MIXED across conditionings; "
                    "overview only, never quote alongside the separated "
                    "blocks",
            "by_arm": {k: summarise(v) for k, v in sorted(mixed_arm.items())},
            "by_class": {k: summarise(v) for k, v in sorted(mixed_class.items())},
            "by_group": {k: summarise(v) for k, v in sorted(mixed_group.items())},
        },
    }


def _heat(values: np.ndarray, mask: np.ndarray, colormap: int) -> np.ndarray:
    """Colormapped 8-bit rendering of one scalar map inside ``mask``."""
    import cv2

    render = np.zeros(values.shape[:2], np.uint8)
    region = mask > 0
    if region.any():
        lo = float(np.min(values[region]))
        hi = float(np.max(values[region]))
        if hi > lo:
            norm = np.clip((values[region] - lo) / (hi - lo), 0.0, 1.0)
            render[region] = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(render, colormap)


def _render_worst_case_panels(panel_dir: Path,
                              trackers: "_WorstCaseTracker") -> list:
    """Readable panels for the worst instance per comparison kind: instance
    contour, baseline q, intervention q, |dq| heat and the valid-set change
    overlay, with the peak-pixel coordinate recorded."""
    import cv2

    panel_dir.mkdir(parents=True, exist_ok=True)
    written = []
    best_per_comparison: dict[str, dict] = {}
    for kind, payload in trackers.best.items():
        conditioning, _, arm_key = kind.partition("|")
        if arm_key == "upsample_control":
            family = "upsample_control"
        elif arm_key.endswith(":full_recompute"):
            family = "shift_full_recompute"
        else:
            family = "shift_fixed_baseline_warp"
        group_key = f"{conditioning}|{family}"
        current = best_per_comparison.get(group_key)
        if current is None or payload["score"] > current["score"]:
            best_per_comparison[group_key] = {
                "score": payload["score"], "kind": kind,
                "arm_key": arm_key, "payload": payload}
    for key, entry in sorted(best_per_comparison.items()):
        conditioning, _, family = key.partition("|")
        payload = entry["payload"]
        crop = payload["crop"]
        h, w = crop["gray"].shape[:2]
        gray3 = cv2.cvtColor(to_gray_u8(crop["gray"]), cv2.COLOR_GRAY2BGR)
        base_q = crop["base_q"].astype(np.float64)
        arm_q = crop["arm_q"].astype(np.float64)
        inst = crop["instance_mask"] > 0
        bvalid = (crop["base_qvalid"] > 0) & inst
        avalid = (crop["arm_qvalid"] > 0) & inst
        common = bvalid & avalid
        lost = bvalid & ~avalid
        gained = avalid & ~bvalid
        contour_mask = (crop["instance_mask"] > 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def framed(map_img: np.ndarray) -> np.ndarray:
            img = map_img.copy()
            cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
            return img

        panels_row1 = [
            framed(gray3),
            framed(_heat(base_q, bvalid, cv2.COLORMAP_TURBO)),
            framed(_heat(arm_q, avalid, cv2.COLORMAP_TURBO)),
            framed(_heat(crop["abs_dq"], common, cv2.COLORMAP_HOT)),
        ]
        change = framed(gray3)
        change[lost] = (0, 0, 255)
        change[gained] = (255, 0, 0)
        both_valid = common & ~lost & ~gained
        change[both_valid] = (
            0.4 * change[both_valid].astype(np.float64)
            + np.array([0, 120, 0])).astype(np.uint8)
        labels_row1 = ["left gray + instance contour", "baseline signed_q",
                       "arm signed_q", "|dq| on common region"]
        labels_row2 = ["valid set change: red=lost, blue=gained, "
                       "green tint=common"]
        scale = min(1.0, 320.0 / max(h, 1))
        cell_h, cell_w = h, w
        if scale < 1.0:
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            panels_row1 = [cv2.resize(p, new_size,
                                      interpolation=cv2.INTER_NEAREST)
                           for p in panels_row1]
            cell_h, cell_w = new_size[1], new_size[0]
        change_small = cv2.resize(change, (cell_w, cell_h),
                                  interpolation=cv2.INTER_NEAREST)
        row2 = [change_small,
                np.zeros((cell_h, cell_w, 3), np.uint8),
                np.zeros((cell_h, cell_w, 3), np.uint8),
                np.zeros((cell_h, cell_w, 3), np.uint8)]
        canvas = np.zeros((cell_h * 2 + 8, cell_w * 4 + 24, 3), np.uint8)

        def paste(img, r, c):
            y, x = r * (cell_h + 4), c * (cell_w + 4)
            canvas[y:y + img.shape[0], x:x + img.shape[1]] = img

        for c, img in enumerate(panels_row1):
            paste(img, 0, c)
        for c, img in enumerate(row2):
            paste(img, 1, c)
        for c, label in enumerate(labels_row1):
            cv2.putText(canvas, label, (c * (cell_w + 4) + 2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.putText(canvas, labels_row2[0], (2, cell_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA)
        name = f"worst_{conditioning}_{family}.png"
        cv2.imwrite(str(panel_dir / name), canvas)
        record = {
            "comparison": family,
            "conditioning": conditioning,
            "frame_id": payload["frame_id"],
            "arm": entry["arm_key"],
            "selection_metric": "max |signed dq| on the instance's common "
                                "region",
            "instance_id": payload["instance_id"],
            "max_abs_signed_dq": payload["score"],
            "peak_pixel_global_xy": crop["peak_pixel_global"],
            "peak_abs_dq": crop["peak_abs_dq"],
            "crop_bbox_xyxy": crop["bbox"],
            "panel_png": f"panels/{name}",
            "paired": payload["paired"],
            "note": "path difference only; no accuracy claim on real "
                    "images",
        }
        written.append(record)
    dump_json(panel_dir / "worst_cases.json", {"panels": written})
    return written


# ---------------------------------------------------------------------------
# Stage 6: real-fish missed-detection review (task E)


def _condition_candidate(cond_dir: Path, frame_id: str, gt_id: int,
                         per_gt_cache: dict, pred_cache: dict) -> dict:
    per_gt = per_gt_cache.setdefault(
        cond_dir.name, read_jsonl(cond_dir / "per_gt.jsonl"))
    preds = pred_cache.setdefault(
        cond_dir.name, read_jsonl(cond_dir / "predictions.jsonl"))
    row = next((r for r in per_gt if r["frame_id"] == frame_id
                and int(r["gt_id"]) == int(gt_id)), None)
    if row is None:
        return {"condition": cond_dir.name, "present": False}
    best = row.get("best_prediction_index")
    conf = None
    if best is not None:
        pred = next((p for p in preds if p["frame_id"] == frame_id
                     and int(p["prediction_index"]) == int(best)), None)
        conf = float(pred["confidence"]) if pred else None
    return {
        "condition": cond_dir.name,
        "present": True,
        "matched": bool(row["matched"]),
        "max_iou_any_prediction": float(row["max_iou_any_prediction"]),
        "best_prediction_confidence": conf,
        "best_prediction_matched_elsewhere": bool(
            row.get("best_prediction_matched_elsewhere", False)),
        "predictions_in_frame": int(row.get("predictions_in_frame", 0)),
    }


def run_review_stage(ctx: RunContext, observations: list[dict] | None) -> None:
    cfg = ctx.config
    det_dir = PROJECT_ROOT / cfg["sources"]["detection_dir"]
    atlas_dir = PROJECT_ROOT / cfg["sources"]["atlas_dir"]
    unmatched = read_jsonl(det_dir / "conditions" / "D0" / "unmatched_gt.jsonl")
    cases = [row for row in unmatched if row["class_name"] == "real_fish"]
    per_gt_cache: dict[str, list] = {}
    pred_cache: dict[str, list] = {}
    obs_by_key: dict[tuple, dict] = {}
    for obs in observations or []:
        key = (obs["frame_id"], int(obs["gt_id"]))
        if key in obs_by_key:
            raise IdentityError(f"duplicate visual observation for {key!r}")
        if obs.get("human_reviewed") is not False:
            raise IdentityError(
                f"visual observation for {key!r} must carry "
                "human_reviewed=false; user confirmation is required first")
        obs_by_key[key] = obs

    csv_rows: list[dict] = []
    for row in sorted(cases, key=lambda r: (r["frame_id"], int(r["gt_id"]))):
        fid, gt_id = row["frame_id"], int(row["gt_id"])
        candidates = [
            _condition_candidate(det_dir / "conditions" / c, fid, gt_id,
                                 per_gt_cache, pred_cache)
            for c in ("D0", "D1", "D2", "D3")]
        positive = [c for c in candidates
                    if c.get("present") and c["max_iou_any_prediction"] > 0.0]
        matched_lower = [c["condition"] for c in candidates
                         if c.get("present") and c["matched"]
                         and c["condition"] != "D0"]
        signs = row.get("signs", {})
        sign_list = []
        if not positive:
            sign_list.append("no_candidate_any_condition")
        else:
            if matched_lower:
                sign_list.append("low_score_candidate")
            best_elsewhere = any(
                c.get("best_prediction_matched_elsewhere") for c in positive)
            if best_elsewhere:
                sign_list.append("overlap_taken_elsewhere")
            if not matched_lower and not best_elsewhere and positive:
                sign_list.append("contour_or_threshold_insufficient")
        for extra in ("confidence_related", "input_scale_related",
                      "duplicate_or_merge_sign", "no_corresponding_prediction_sign"):
            if row.get("attribution") == extra and extra not in sign_list:
                sign_list.append(f"saved_attribution:{extra}")
        obs = obs_by_key.get((fid, gt_id))
        panel = atlas_dir / f"unmatched_gt_{fid}_gt{gt_id}.png"
        csv_rows.append({
            "case_id": f"{fid}#gt{gt_id}",
            "frame_id": fid,
            "gt_id": gt_id,
            "group": row.get("group"),
            "class_name": row["class_name"],
            "atlas_panel_path": str(panel.relative_to(PROJECT_ROOT))
            if panel.is_file() else "MISSING",
            "d0_max_iou_any_prediction": float(row["max_iou_any_prediction"]),
            "d0_no_positive_overlap": bool(row.get("no_positive_overlap")),
            "matched_D1": bool(signs.get("matched_D1")),
            "matched_D2": bool(signs.get("matched_D2")),
            "matched_D3": bool(signs.get("matched_D3")),
            "per_condition_evidence": json.dumps(candidates, ensure_ascii=False),
            "rule_based_signs": ";".join(sign_list),
            "saved_attribution": row.get("attribution"),
            "visual_observation": (obs or {}).get("observations"),
            "evidence_strength": (obs or {}).get("evidence_strength"),
            "items_to_confirm": (obs or {}).get("items_to_confirm"),
            "review_source": (cfg["review"]["visual_review_source"]
                              if obs else cfg["review"]["record_based_source"]),
            "human_reviewed": "false",
        })
    missing_panels = [r["case_id"] for r in csv_rows
                      if r["atlas_panel_path"] == "MISSING"]
    unmatched_obs = [k for k in obs_by_key
                     if k not in {(r["frame_id"], int(r["gt_id"]))
                                  for r in csv_rows}]
    if unmatched_obs:
        raise IdentityError(
            f"visual observations reference unknown real-fish cases: "
            f"{sorted(unmatched_obs)}")

    out_path = ctx.output_dir / "real_fish_review.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    dump_json(ctx.output_dir / "real_fish_review_summary.json", {
        "stage": "real_fish_missed_gt_review",
        "cases": len(csv_rows),
        "missing_atlas_panels": missing_panels,
        "visual_reviewed_cases": sum(
            1 for r in csv_rows
            if r["review_source"] == cfg["review"]["visual_review_source"]),
        "human_reviewed": False,
        "note": "rule-based signs are SIGNS only; no NMS/capture root cause "
                "is asserted and no label change is proposed",
    })
    log(f"review: {len(csv_rows)} real-fish cases written "
        f"({len(missing_panels)} missing panels)")
    ctx.stage("review", "COMPLETED", cases=len(csv_rows),
              missing_panels=missing_panels,
              visual_reviewed=sum(
                  1 for r in csv_rows
                  if r["review_source"] == cfg["review"]["visual_review_source"]))


def load_visual_observations(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("observations")
    if not isinstance(data, list) or not data:
        raise IdentityError(
            "visual observations must be a non-empty list of per-case dicts")
    for obs in data:
        for key in ("frame_id", "gt_id", "observations", "evidence_strength",
                    "items_to_confirm"):
            if key not in obs:
                raise IdentityError(
                    f"visual observation missing {key!r}: {obs!r}")
        obs["gt_id"] = int(obs["gt_id"])
        if obs.get("human_reviewed") is None:
            obs["human_reviewed"] = False
    return data


# ---------------------------------------------------------------------------
# Protection


def run_protection(ctx: RunContext, phase: str) -> None:
    if phase == "before":
        # the "before" side is NOT a late snapshot: it is the per-input
        # identity recorded at each file's FIRST read (see RunContext
        # .consume). No re-hashing happens here.
        records = [
            {"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"],
             "role": r["role"], "pinned_sha256": r.get("pinned_sha256"),
             "binding": r.get("binding"),
             "recorded_at": "first_read"}
            for r in sorted(ctx.first_read.values(), key=lambda r: r["path"])]
        dump_json(ctx.output_dir / "protected_before.json", {
            "phase": "before (identity at each input's first read)",
            "taken_at": utc_now(),
            "files": records,
            "window_disclosure": "reads made before this recording are "
                                 "limited to the three anchors (verified "
                                 "against config pins) and the "
                                 "artifact-manifest/frozen-YAML binding "
                                 "lookups (expectations checked at that "
                                 "point); this run does NOT claim the "
                                 "recording covers anything read earlier "
                                 "by other processes",
        })
    else:
        records = ctx.snapshot_after()
        dump_json(ctx.output_dir / "protected_after.json", {
            "phase": "after (fresh end-of-run hash)",
            "taken_at": utc_now(),
            "files": records,
        })
        ctx.after = {r["path"]: r for r in records}


def compare_protection(ctx: RunContext) -> dict:
    problems = []
    first_read = ctx.first_read
    only_before = sorted(set(first_read) - set(ctx.after))
    only_after = sorted(set(ctx.after) - set(first_read))
    for rel in only_before:
        problems.append(f"{rel}: recorded at first read, missing at end")
    for rel in only_after:
        problems.append(f"{rel}: appeared only in the end snapshot")
    consumed_records = []
    for rel in sorted(first_read):
        before = first_read[rel]
        after = ctx.after.get(rel)
        if after is None:
            problems.append(f"{rel}: missing on the end side")
            continue
        match = (before["sha256"] == after["sha256"]
                 and before["bytes"] == after["bytes"])
        if not match:
            problems.append(f"{rel}: CHANGED after its first read during "
                            "this run")
        pinned = before.get("pinned_sha256")
        pin_match = None
        if pinned:
            pin_match = before["sha256"] == pinned
            if not pin_match:
                problems.append(f"{rel}: current bytes do not match the "
                                "frozen pinned sha256")
        binding = before.get("binding")
        consumed_records.append({
            "path": rel,
            "role": before["role"],
            "bytes": before["bytes"],
            "pinned_sha256": pinned,
            "pinned_match": pin_match,
            "binding": binding,
            "binding_verified_at_first_read": bool(binding),
            "first_read_sha256": before["sha256"],
            "after_sha256": after["sha256"],
            "unchanged_after_first_read": match,
        })
    result = {
        "files_compared": len(first_read),
        "identical": not problems,
        "problems": problems,
        "window": "per-input identity is recorded at each input's FIRST "
                  "read and compared against a fresh end-of-run hash; any "
                  "change after the first read is reported. This does NOT "
                  "claim coverage of reads made before the recording by "
                  "other processes; the three anchors and every frozen-"
                  "bound input are additionally checked against their "
                  "pinned expectations at the recording point.",
        "scope_note": "the protected set is this run's CONSUMED inputs only; "
                      "no whole-project protection is claimed",
    }
    dump_json(ctx.output_dir / "protection_check.json", result)
    dump_json(ctx.output_dir / "source_manifest.json", {
        "run_id": ctx.run_id,
        "mode": ctx.mode,
        "anchors": ctx.config["anchors"],
        "consumed_inputs": consumed_records,
        "protection": result,
    })
    return result


# ---------------------------------------------------------------------------
# Report and run status


def write_summary_report(ctx: RunContext, protection: dict) -> None:
    """Auto-summarised from the run's own machine artifacts — every number
    below is read back from the JSON this run wrote, never hand-typed."""
    cfg = ctx.config
    stages = ctx.stages
    lines = [
        "# material_measurement_diagnostic_v1 — run report (auto-summary)",
        "",
        f"- run_id: `{ctx.run_id}`",
        f"- mode: `{ctx.mode}`",
        f"- finished: {utc_now()}",
        f"- budget: {ctx.budget_minutes} minutes (shared, never reset)",
        "",
        "## Stage status",
        "",
    ]
    for name, record in stages.items():
        lines.append(f"- **{name}**: {record['status']}")
    not_run = [s for s in MODE_STAGES[ctx.mode] if s not in stages]
    if not_run:
        lines.append(f"- NOT RUN in this mode: {', '.join(not_run)}")
    not_in_mode = [s for s in ("verify", "selection", "saved", "synthetic",
                               "recompute", "review", "protection")
                   if s not in MODE_STAGES[ctx.mode]]
    if not_in_mode:
        lines.append(f"- not part of this mode: {', '.join(not_in_mode)}")

    summary_doc: dict[str, Any] = {"run_id": ctx.run_id}
    if "saved" in stages and (ctx.output_dir /
                              "residual_capacity_summary.json").exists():
        saved = json.loads((ctx.output_dir / "residual_capacity_summary.json")
                           .read_text(encoding="utf-8"))
        oe = saved["outer_eval"]["totals"]
        pj = saved["pipeline_join"]["totals"]
        pct = (100.0 * oe["m_ge_1"] / oe["errors"]
               if oe["errors"] else None)
        lines += ["", "## Task A — residual capacity (from "
                  "`residual_capacity_summary.json`)", "",
                  f"- outer_eval: {oe['samples']} samples, {oe['errors']} "
                  f"errors, {oe['m_ge_1']} m>=1 "
                  f"({pct:.1f}% of errors), {oe['m_lt_1']} m<1",
                  f"- errors with missing polar S: {oe['errors_with_missing_polar']}"
                  f" (fall back to exact P0 regardless of m); m<1 with valid "
                  f"polar: {oe['m_lt_1_with_valid_polar']}",
                  f"- pipeline join: {pj['frames']} frames / {pj['gt_instances']}"
                  f" GT / {pj['predictions']} predictions / {pj['matches']} "
                  f"matches; matched errors {pj['errors']} "
                  f"({pj['m_ge_1']} m>=1, {pj['m_lt_1']} m<1)",
                  "- m>=1 is a per-sample accuracy CAPACITY bound only; no "
                  "Macro-F1 bound and no realized classification gain follows",
                  ]
        summary_doc["task_a"] = {"outer_eval": oe, "pipeline_join": pj,
                                 "m_ge_1_pct_of_errors": pct}

    if "synthetic" in stages and \
            (ctx.output_dir / "synthetic_report.json").exists():
        lines += ["", "## Task B — synthetic upsample chain", "",
                  "See `synthetic_report.json`; the in-situ scenario "
                  "attributions (zero-fill pull vs SGBM edge response) are "
                  "recorded there."]
    if "recompute" in stages and \
            (ctx.output_dir / "recompute_paired_summary.json").exists():
        paired = json.loads(
            (ctx.output_dir / "recompute_paired_summary.json")
            .read_text(encoding="utf-8"))
        ov = paired["overall"]
        lines += ["", "## Task D — paired per-instance statistics (from "
                  "`recompute_paired_summary.json`)", "",
                  f"- instances: {ov['instances_total']} measured, "
                  f"{ov['instances_with_common_region']} with a non-empty "
                  f"common region; common q pixels {ov['common_pixels_total']}",
                  f"- abs_dq_mean (pixel MAE on common regions), "
                  f"pixel-weighted: "
                  f"{_fmt(ov.get('abs_dq_mean_pixel_weighted'))}",
                  f"- signed_dq mean (cancellation-prone mean difference, "
                  f"NOT an error metric), pixel-weighted: "
                  f"{_fmt(ov.get('signed_dq_mean_pixel_weighted'))}",
                  f"- dabs_q mean (feature-magnitude change, NOT an error "
                  f"metric), pixel-weighted: "
                  f"{_fmt(ov.get('dabs_q_mean_pixel_weighted'))}",
                  f"- abs_dq_instance_p95_pixelcount_weighted (weighted "
                  f"mean of PER-INSTANCE p95; pooled pixel p95 is "
                  f"NOT_COMPUTED): "
                  f"{_fmt(ov.get('abs_dq_instance_p95_pixelcount_weighted'))}"
                  f"; worst instances per comparison in "
                  f"`panels/worst_cases.json` with peak pixel coordinates",
                  f"- q validity flips cumulated over instance-by-"
                  f"intervention RECORDS (NOT unique pixels): "
                  f"lost {ov.get('q_lost_total')} / "
                  f"gained {ov.get('q_gained_total')}",
                  "- per-conditioning / per-class / per-group blocks with "
                  "explicit weights are in the same JSON; real images have "
                  "no ground truth, so these are PATH DIFFERENCES only",
                  ]
        by_cond = {k: {kk: _fmt(vv) for kk, vv in v.items()
                       if kk in ("abs_dq_mean_pixel_weighted",
                                 "q_lost_total", "q_gained_total")}
                   for k, v in paired["by_conditioning"].items()}
        lines += ["", "### by conditioning", ""] + [
            f"- {k}: {json.dumps(v)}" for k, v in sorted(by_cond.items())]
        for cond, arms in sorted(
                paired.get("by_arm_per_conditioning", {}).items()):
            arm_view = {arm: {kk: _fmt(vv) for kk, vv in s.items()
                              if kk in ("instances_with_common_region",
                                        "abs_dq_mean_pixel_weighted",
                                        "q_lost_total", "q_gained_total")}
                        for arm, s in arms.items()}
            lines += [f"", f"### arms per conditioning — {cond}", ""] + [
                f"- {arm}: {json.dumps(view)}"
                for arm, view in sorted(arm_view.items())]
        lines += ["", "NOTE: the `overview_mixed_conditionings` block in "
                  "`recompute_paired_summary.json` mixes conditionings and "
                  "is overview-only; quote the separated blocks above.",
                  "NOTE: flip totals cumulate over instance-by-intervention "
                  "records and are not unique pixel counts."]
        summary_doc["task_d_overall"] = ov

    if "review" in stages:
        lines += ["", "## Task E — real-fish review", "",
                  "See `real_fish_review.csv`; `human_reviewed` is false "
                  "everywhere until the user confirms cases."]

    lines += ["", "## Protection (from `protection_check.json`)", "",
              f"- registered sources first-read-recorded and re-compared: "
              f"{protection['files_compared']}",
              f"- unchanged after first read: {protection['identical']}",
              "- the protected set is the REGISTERED sources listed in "
              "`source_manifest.json`; exclusions are disclosed there"]
    if protection["problems"]:
        lines += ["- PROBLEMS:"] + [f"  - {p}" for p in protection["problems"]]
    lines += ["", "## Disclosures", "",
              f"- {cfg['provenance_disclosure']}",
              "- shift interventions are sensitivity probes, not measured "
              "registration errors.",
              "- the upsample control is diagnostic-only, not a proposed "
              "default fix.",
              "- per-arm means are kept per instance; a mean difference is "
              "not a pixel MAE and must not be quoted as one."]
    (ctx.output_dir / "diagnostic_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    dump_json(ctx.output_dir / "report_summary.json", summary_doc)


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def finalize(ctx: RunContext, *, interrupted: bool, failure: str | None,
             protection: dict | None) -> int:
    requested = list(MODE_STAGES[ctx.mode])
    not_run = [s for s in requested if s not in ctx.stages]
    not_in_mode = [s for s in ("verify", "selection", "saved", "synthetic",
                               "recompute", "review", "protection")
                   if s not in requested]
    failed = [s for s, rec in ctx.stages.items() if rec["status"] == "FAILED"]
    partial = [s for s, rec in ctx.stages.items() if rec["status"] == "PARTIAL"]
    if failure:
        status = "FAILED"
        exit_code = EXIT_FAILED
    elif interrupted:
        status = "PARTIAL"
        exit_code = EXIT_PARTIAL
    elif failed:
        status = "FAILED"
        exit_code = EXIT_FAILED
    elif partial:
        status = "PARTIAL"
        exit_code = EXIT_PARTIAL
    else:
        status = "COMPLETED"
        exit_code = EXIT_OK
    payload = {
        "run_id": ctx.run_id,
        "mode": ctx.mode,
        "requested_stages": requested,
        "stages": ctx.stages,
        "stages_not_run": not_run,
        "stages_not_in_mode": not_in_mode,
        "interrupted_by_budget": interrupted,
        "failure": failure,
        "amendments": ctx.amendments,
        "overall_status": status,
        "exit_code": exit_code,
        "started_at": ctx.stages.get("verify", {}).get("started_at"),
        "finished_at": utc_now(),
        "protection": (protection if protection is not None else
                       {"note": "protection did not complete"}),
    }
    dump_json(ctx.output_dir / "run_status.json", payload)
    log(f"overall: {status} (exit {exit_code}); stages not run: {not_run}")
    return exit_code


# ---------------------------------------------------------------------------
# Entry point


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="material_measurement_diagnostic_v1 entry (batch 1)")
    parser.add_argument("--config", required=True,
                        help="experiment config path")
    parser.add_argument("--output-dir", required=True,
                        help="final output directory; must NOT exist")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--budget-minutes", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--record-visual-observations", default=None,
                        help="path to a per-case visual observation JSON "
                             "(claude_visual_review; human_reviewed=false)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_minutes is None or not (
            type(args.budget_minutes) in (int, float)
            and float(args.budget_minutes) > 0
            and np.isfinite(args.budget_minutes)):
        log("invalid --budget-minutes")
        return EXIT_USAGE
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cap = int(config["selection"]["max_frames_cap"])
    if type(args.max_frames) is not int or args.max_frames <= 0:
        log("--max-frames must be a positive int")
        return EXIT_USAGE
    if args.max_frames > cap:
        log(f"--max-frames {args.max_frames} exceeds the frozen cap {cap}; "
            "the selection can only be shrunk, never widened")
        return EXIT_USAGE

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists():
        log(f"refusing: output directory already exists: {output_dir}")
        return EXIT_USAGE
    if is_link_or_reparse(output_dir) or is_link_or_reparse(output_dir.parent):
        log("refusing: output directory or parent is a link/reparse point")
        return EXIT_USAGE
    rel_output = output_dir.relative_to(PROJECT_ROOT).as_posix()
    probe = subprocess.run(
        ["git", "check-ignore", "-q", "--", rel_output],
        cwd=str(PROJECT_ROOT))
    if probe.returncode != 0:
        log(f"refusing: output directory {rel_output!r} is not covered by "
            ".gitignore")
        return EXIT_USAGE
    output_dir.mkdir(parents=True)

    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    ctx = RunContext(
        config=config, config_path=config_path, output_dir=output_dir,
        mode=args.mode, budget_minutes=float(args.budget_minutes),
        max_frames=args.max_frames, run_id=run_id,
        watch=BudgetWatch(float(args.budget_minutes)))
    # register the diagnostic config itself: it was read once above to
    # build this run, so the recorded identity is "at registration" (a
    # change between that read and this registration is not detectable —
    # disclosed in the protection window)
    ctx.consume(config_path, "diagnostic_config")

    observations = None
    failure = None
    interrupted = False
    protection = None
    try:
        try:
            observations = load_visual_observations(
                Path(args.record_visual_observations)
                if args.record_visual_observations else None)
            if args.record_visual_observations:
                # identity recorded immediately after the read above; same
                # registration semantics as the diagnostic config
                ctx.consume(Path(args.record_visual_observations),
                            "visual_observations")
            if observations is not None:
                ctx.amendments.append({
                    "type": "visual_observations_merged",
                    "file": Path(args.record_visual_observations).resolve()
                    .as_posix(),
                    "cases": len(observations),
                    "review_source": ctx.config["review"][
                        "visual_review_source"],
                    "human_reviewed": False,
                    "note": "claude_visual_review merged into "
                            "real_fish_review.csv; user confirmation still "
                            "pending",
                })
            ctx.watch.check("before verify")
            ctx.stages.setdefault("verify", {"status": "RUNNING"})
            verify_sources(ctx)
            ctx.stages["verify"]["status"] = "COMPLETED"

            if "selection" in MODE_STAGES[args.mode]:
                ctx.watch.check("before selection")
                run_selection(ctx)

            if "saved" in MODE_STAGES[args.mode]:
                ctx.watch.check("before saved stage")
                run_saved_stage(ctx)

            if "synthetic" in MODE_STAGES[args.mode]:
                ctx.watch.check("before synthetic stage")
                run_synthetic_stage(ctx)

            if "recompute" in MODE_STAGES[args.mode]:
                ctx.watch.check("before recompute stage")
                run_protection(ctx, "before")
                run_recompute_stage(ctx)
            if "review" in MODE_STAGES[args.mode]:
                ctx.watch.check("before review stage")
                run_review_stage(ctx, observations)

            ctx.watch.check("before protection snapshot")
            # the "before" side (identity at each input's first read) is
            # dumped for the record in every mode; the end comparison uses
            # it either way
            run_protection(ctx, "before")
            run_protection(ctx, "after")
            protection_result = compare_protection(ctx)
            protection = protection_result
            ctx.stage("protection",
                      "COMPLETED" if protection_result["identical"]
                      else "FAILED",
                      files_compared=protection_result["files_compared"],
                      problems=protection_result["problems"])
            if not protection_result["identical"]:
                failure = "protection check found changed inputs"
            write_summary_report(ctx, protection_result)
        except BudgetExceeded as exc:
            interrupted = True
            log(f"budget exhausted: {exc}")
            try:
                run_protection(ctx, "before")
                run_protection(ctx, "after")
                protection = compare_protection(ctx)
            except IdentityError as inner:
                log(f"protection after exhaustion failed: {inner}")
            write_summary_report(
                ctx, protection or {"files_compared": 0,
                                    "identical": None, "problems": []})
        except (IdentityError, SelectionConflict, FileNotFoundError,
                ValueError, OSError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            log(f"stage failure: {failure}")
    except Exception as exc:  # unexpected failure still leaves a status file
        failure = f"UNEXPECTED {type(exc).__name__}: {exc}"
        log(f"unexpected failure: {failure}")
    try:
        exit_code = finalize(ctx, interrupted=interrupted, failure=failure,
                             protection=protection)
    except ValueError as exc:
        # e.g. a NaN sneaked into the status payload itself
        log(f"failed to write run_status.json: {exc}")
        return EXIT_FAILED
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
