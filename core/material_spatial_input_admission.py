"""material_spatial_input_v1 — admission helpers for the review-fix batch.

This module holds ONLY path handling, source registration, the cache
contract (definition/binding/ledgers) and identity validation shared by
the builder and the independent verifier. It deliberately contains none
of the grid/permutation execution math: the builder keeps the SAT form
and the verifier keeps its own independent weight-matrix reference, and
neither may be routed through here.

Path keys are canonical project-relative POSIX keys (case-folded on
Windows so path aliases cannot split one file into two expectations).
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np

from core.material_spatial_input import (
    EXPERIMENT,
    GRID_PLANES,
    GRID_SCHEMA,
    GRID_SIZE,
    PERMUTATION_SCHEMA,
    REGIONS,
    NativeInstanceArrays,
    SpatialRefused,
    canvas_offset,
    scaled_size,
)

# --- the new cache contract (review_fix_v1) ---------------------------
CACHE_SCHEMA = "material_spatial_input_cache_v1"
IMPLEMENTATION_REVISION = "review_fix_v1"
KIND_FULL = "material_spatial_input_full_v1"
KIND_SMOKE = "material_spatial_input_smoke_v1"
NATIVE_ARRAY_KEYS = ("L", "R", "q", "M", "V", "region", "quality")
META_RELS = ("cache/definition.json", "cache/samples.jsonl",
             "cache/frames.jsonl")
BINDING_REL = "cache/binding.json"
ARTIFACT_KINDS = ("metadata", "native", "grid", "permutation")

_REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class AdmissionRefused(SpatialRefused):
    """A path, registration, cache-contract or identity refusal."""


# ---------------------------------------------------------------------------
# canonical path keys and ancestor checks


def is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return bool(getattr(info, "st_file_attributes", 0)
                & _REPARSE_FLAG)


def canonical_rel(path: Path, project_root: Path) -> str:
    """The canonical project-relative POSIX key (case-folded on Windows)."""
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    rel = resolved.relative_to(root).as_posix()
    return rel.lower() if os.name == "nt" else rel


def check_consumed_path(path: Path, project_root: Path) -> str:
    """Refuse anything outside the project, any '..' traversal and any
    link/reparse point on the file or on ANY ancestor. The ancestor
    walk uses the RAW path (resolve() would silently traverse a
    junction and hide it). Returns the canonical rel key."""
    root = Path(project_root).resolve()
    raw_abs = Path(os.path.abspath(str(path)))
    if ".." in Path(str(path)).parts:
        raise AdmissionRefused(f"{path}: '..' traversal in the raw path")
    node = raw_abs
    root_raw = Path(os.path.abspath(str(project_root)))
    while node.resolve() != root:
        if is_link_or_reparse(node):
            raise AdmissionRefused(
                f"{path}: link or reparse point at {node}")
        parent = node.parent
        if parent == node:
            break
        node = parent
    resolved = raw_abs.resolve()
    try:
        rel = resolved.relative_to(root).as_posix()
    except ValueError:
        raise AdmissionRefused(
            f"{path} resolves outside the project root {root}")
    if not resolved.is_file():
        raise AdmissionRefused(f"{path}: missing at registration")
    return rel.lower() if os.name == "nt" else rel


def validate_new_run_root(run_root: Path, project_root: Path) -> Path:
    """The run root must be absolute, inside the project, not existing,
    free of '..'/'~' alias segments and free of link/reparse ancestors."""
    raw = str(run_root)
    if not Path(raw).is_absolute():
        raise AdmissionRefused(f"run root must be absolute: {raw}")
    parts = Path(raw).parts
    for part in parts:
        if part in ("..", "."):
            raise AdmissionRefused(f"run root {raw}: '{part}' segment")
        if "~" in part:
            raise AdmissionRefused(
                f"run root {raw}: 8.3-alias segment {part!r}")
    resolved = Path(raw).resolve()
    root = Path(project_root).resolve()
    if resolved == root or root not in resolved.parents:
        raise AdmissionRefused(
            f"run root {resolved} is not a new directory inside {root}")
    if Path(raw).exists():
        raise AdmissionRefused(f"run root already exists: {resolved}")
    probe = resolved
    while probe != root:
        if probe.exists() and is_link_or_reparse(probe):
            raise AdmissionRefused(
                f"link or reparse point on the run-root path: {probe}")
        probe = probe.parent
    return resolved


def register_consumed(registry, path: Path, role: str,
                      project_root: Path,
                      expectation: dict | None = None) -> dict:
    """Register one consumed file after the full ancestor check; the
    expectation (if any) is verified by the registry against the CURRENT
    bytes, before any consumer may read the file."""
    check_consumed_path(path, project_root)
    return registry.register(path, role, binding=expectation)


def validated_scratch_cleanup_paths(scratch_root: Path,
                                    project_root: Path) -> tuple[Path, ...]:
    """Validate root, every ancestor and every descendant before deletion.

    The caller owns this exact scratch root. Returned children are in
    leaf-first order; callers use unlink/rmdir, never a recursive delete.
    This function does not delete or traverse any reparse point.
    """
    raw = Path(os.path.abspath(str(scratch_root)))
    project = Path(os.path.abspath(str(project_root)))
    tree = project / ".test_artifacts"
    if ".." in Path(scratch_root).parts or raw == tree or tree not in raw.parents:
        raise AdmissionRefused(f"unsafe scratch cleanup root: {scratch_root}")
    node = raw
    root_info = None
    while True:
        try:
            info = node.lstat()
        except FileNotFoundError:
            if node != raw:
                raise AdmissionRefused(f"missing scratch ancestor: {node}")
        else:
            if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0) & _REPARSE_FLAG):
                raise AdmissionRefused(f"link/reparse cleanup ancestor: {node}")
            if node == raw:
                root_info = info
        if node == project:
            break
        node = node.parent
    if root_info is None:
        return ()
    if not stat.S_ISDIR(root_info.st_mode):
        raise AdmissionRefused(f"scratch root is not a directory: {raw}")
    resolved = raw.resolve(strict=True)
    if tree.resolve(strict=True) not in resolved.parents:
        raise AdmissionRefused(f"scratch root resolves outside ownership: {raw}")
    paths, stack = [], [raw]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or (
                        getattr(info, "st_file_attributes", 0) & _REPARSE_FLAG):
                    raise AdmissionRefused(f"link/reparse cleanup child: {path}")
                if resolved not in path.resolve(strict=True).parents:
                    raise AdmissionRefused(f"cleanup child escaped ownership: {path}")
                if stat.S_ISDIR(info.st_mode):
                    stack.append(path)
                elif not stat.S_ISREG(info.st_mode):
                    raise AdmissionRefused(f"special cleanup child: {path}")
                paths.append(path)
    return tuple(sorted(paths, key=lambda p: (-len(p.parts), p.as_posix())))


# ---------------------------------------------------------------------------
# canonical expectation layer (conflict detection across path aliases)


class ExpectationBook:
    """Expected sha256 per canonical rel key. Historical expansions and
    cache bindings are merged here; a conflicting expectation for the
    same canonical path is refused instead of silently split across two
    path aliases."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.expected: dict[str, dict] = {}

    def add(self, path: Path, source: str, sha256: str,
            bytes_: int | None = None) -> dict:
        rel = canonical_rel(path, self.project_root)
        record = {"source": source, "expected_sha256": str(sha256)}
        if bytes_ is not None:
            record["expected_bytes"] = int(bytes_)
        existing = self.expected.get(rel)
        if existing is not None:
            if existing.get("expected_sha256") !=                     record["expected_sha256"]:
                raise AdmissionRefused(
                    f"{rel}: conflicting expectations "
                    f"({existing.get('source')} vs {source})")
            return dict(existing, added=False)
        # the registry's own keys are the raw project-relative POSIX
        # strings, so keep the exact path that produced this key
        record["rel_raw"] = Path(path).resolve().relative_to(
            Path(self.project_root).resolve()).as_posix()
        self.expected[rel] = record
        return dict(record, added=True)

    def binding_for(self, path: Path) -> dict | None:
        return self.expected.get(canonical_rel(path, self.project_root))


# ---------------------------------------------------------------------------
# binding construction and inventory verification


def npz_key_records(path: Path) -> list[dict]:
    """dtype/shape for every key of an npz, in stored order."""
    with np.load(path) as payload:
        return [{"key": key, "dtype": str(payload[key].dtype),
                 "shape": list(payload[key].shape)}
                for key in payload.files]


def build_cache_binding(run_root: Path, records: list[dict]) -> dict:
    """The cache binding covers definition.json, samples.jsonl,
    frames.jsonl and every native/grid/permutation shard; binding.json
    excludes itself (its sha256 is bound by the builder status)."""
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        raise AdmissionRefused("duplicate artifact path in the binding")
    for rel in META_RELS:
        if rel not in paths:
            raise AdmissionRefused(f"the binding must cover {rel}")
    for record in records:
        if record["kind"] not in ARTIFACT_KINDS:
            raise AdmissionRefused(
                f"{record['path']}: unknown artifact kind "
                f"{record['kind']!r}")
        if not record["path"].startswith("cache/"):
            raise AdmissionRefused(
                f"{record['path']}: artifacts must live under cache/")
        if record["kind"] == "metadata" and record.get("npz_keys"):
            raise AdmissionRefused(
                f"{record['path']}: a non-array file must not carry "
                "dtype/shape records")
    binding = {
        "experiment": EXPERIMENT,
        "cache_schema": CACHE_SCHEMA,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "artifact_count": len(records),
        "artifacts": sorted(records, key=lambda item: item["path"]),
        "note": ("binding.json excludes itself; its sha256 is bound by "
                 "the builder status and re-checked by the verifier and "
                 "the wrapper"),
    }
    return binding


def verify_inventory(run_root: Path, binding: dict, registry,
                     project_root: Path) -> dict:
    """Full inventory check: metadata present, no duplicate/unknown
    entries, no extra file under cache/, every artifact verified against
    its recorded bytes/sha BEFORE it is loaded, npz keys exactly equal
    to the recorded dtype/shape set, non-array files without dtype/shape
    records. Every artifact is registered for final-state protection."""
    run_root = Path(run_root)
    artifacts: dict[str, dict] = {}
    for record in binding.get("artifacts", []):
        rel = record["path"]
        if rel in artifacts:
            raise AdmissionRefused(f"duplicate inventory entry: {rel}")
        artifacts[rel] = record
    for rel in META_RELS:
        if rel not in artifacts:
            raise AdmissionRefused(f"metadata missing from the binding: "
                                   f"{rel}")
    cache_root = run_root / "cache"
    actual = {}
    for path in sorted(cache_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(run_root).as_posix()
            if rel == BINDING_REL:
                continue
            actual[rel] = path
    missing = sorted(set(artifacts) - set(actual))
    extra = sorted(set(actual) - set(artifacts))
    if missing:
        raise AdmissionRefused(f"inventory artifacts missing: "
                               f"{missing[:5]}")
    if extra:
        raise AdmissionRefused(f"extra cache files not in the binding: "
                               f"{extra[:5]}")
    for rel, record in sorted(artifacts.items()):
        path = run_root / rel
        expectation = {"source": "cache_binding",
                       "expected_sha256": str(record["sha256"]),
                       "expected_bytes": int(record["bytes"])}
        register_consumed(registry, path, f"cache:{record['kind']}:{rel}",
                          project_root, expectation)
        if record["kind"] == "metadata":
            if record.get("npz_keys"):
                raise AdmissionRefused(
                    f"{rel}: a metadata file must not carry npz key "
                    "records")
            continue
        if not rel.endswith((".npz", ".json")):
            raise AdmissionRefused(f"{rel}: unexpected artifact suffix")
        if rel.endswith(".npz"):
            recorded = record.get("npz_keys")
            if not isinstance(recorded, list) or not recorded:
                raise AdmissionRefused(
                    f"{rel}: npz artifacts must record every key's "
                    "dtype/shape")
            with np.load(path) as payload:
                actual_keys = {key: {"dtype": str(payload[key].dtype),
                                     "shape": list(payload[key].shape)}
                               for key in payload.files}
            recorded_map = {entry["key"]: entry for entry in recorded}
            if set(actual_keys) != set(recorded_map):
                only_saved = sorted(set(actual_keys) - set(recorded_map))
                only_rec = sorted(set(recorded_map) - set(actual_keys))
                raise AdmissionRefused(
                    f"{rel}: npz keys differ from the binding "
                    f"(undocumented: {only_saved[:3]}, "
                    f"missing: {only_rec[:3]})")
            for key, entry in actual_keys.items():
                want = recorded_map[key]
                if entry["dtype"] != want["dtype"] or \
                        entry["shape"] != list(want["shape"]):
                    raise AdmissionRefused(
                        f"{rel}|{key}: dtype/shape {entry} != recorded "
                        f"{want}")
        else:
            if record.get("npz_keys"):
                raise AdmissionRefused(
                    f"{rel}: a JSON artifact must not carry npz key "
                    "records")
    return {"artifacts": len(artifacts)}


# ---------------------------------------------------------------------------
# shared admission stages (identical rules for the builder and verifier)


def run_v5_admission(registry, book: "ExpectationBook", protocol: dict,
                     cfg: dict, project_root: Path,
                     log=None) -> dict:
    """The FULL V5 admission (existing production gate). The gate's whole
    read chain — audit/manifest/summary, every npz the audit digests and
    every annotation source the manifest names — registers (hash-verified
    against the expanded historical expectations) BEFORE the gate runs.
    Test-split reads are integrity reads only: they contribute no sample,
    statistic, cache row or training input. The frozen fingerprint
    comparison happens inside the gate itself."""
    from core.material_mask_dataset import (
        MASK_RESIZE_VERSION,
        MASK_SCHEMA_VERSION,
        admit_dataset_for_protocol,
    )

    data_root = Path(project_root) / cfg["frozen_sources"]["v5_data"][
        "root"]
    for name in ("dataset_audit.json", "dataset_manifest.csv",
                 "dataset_summary.json"):
        register_consumed(registry, data_root / name, f"v5_gate:{name}",
                          project_root, book.binding_for(data_root / name))
    audit = json.loads((data_root / "dataset_audit.json").read_text(
        encoding="utf-8"))
    digests = audit.get("file_digests") or {}
    if not digests:
        raise AdmissionRefused("the V5 audit carries no per-file digests")
    for rel in sorted(digests):
        register_consumed(registry, data_root / rel, f"v5_gate:npz:{rel}",
                          project_root, book.binding_for(data_root / rel))
    import csv

    with (data_root / "dataset_manifest.csv").open(
            encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # label paths are project-relative (like left_path in the manifest);
    # npz paths are relative to the V5 root
    for rel in sorted({row["label_path"] for row in rows}):
        register_consumed(registry, Path(project_root) / rel,
                          f"v5_gate:label:{rel}", project_root,
                          book.binding_for(Path(project_root) / rel))
    parent = (protocol.get("admission") or {}).get(
        "v4_parent_fingerprint", "")
    if not parent:
        raise AdmissionRefused("the frozen protocol does not carry the "
                               "V4 parent fingerprint for the V5 gate")
    report = admit_dataset_for_protocol(
        data_root,
        pooling_mode="instance_mask",
        expected_fingerprint=cfg["frozen_sources"]["v5_data"][
            "fingerprint"],
        expected_parent_fingerprint=parent,
        expected_mask_schema=MASK_SCHEMA_VERSION,
        expected_mask_resize_version=MASK_RESIZE_VERSION)
    if report.get("audit_fingerprint") != \
            cfg["frozen_sources"]["v5_data"]["fingerprint"]:
        raise AdmissionRefused(
            "the V5 gate fingerprint does not match the pinned "
            "fingerprint")
    if log is not None:
        log(f"full V5 admission passed (fingerprint "
            f"{report['audit_fingerprint'][:12]}…; integrity reads cover "
            "the whole dataset including test)")
    return {"audit_fingerprint": report["audit_fingerprint"],
            "v4_parent_fingerprint": parent,
            "mask_schema_version": report.get("mask_schema_version"),
            "mask_resize_version": report.get("mask_resize_version"),
            "registered_v5_files": len(digests)}


def dev_census(cfg: dict, registry, book: "ExpectationBook",
               project_root: Path) -> dict:
    """The dev-pool census against the pinned expectations, with the
    empty frames' consumed files registered first."""
    import csv

    data_root = Path(project_root) / cfg["frozen_sources"]["v5_data"][
        "root"]
    manifest_path = data_root / "dataset_manifest.csv"
    register_consumed(registry, manifest_path,
                      "v5_record:dataset_manifest.csv", project_root,
                      book.binding_for(manifest_path))
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    splits = set(cfg["dev_pool"]["splits"])
    dev_rows = [row for row in all_rows if row["split"] in splits]
    excluded = sorted({row["split"] for row in all_rows} - splits)
    if excluded != ["test"]:
        raise AdmissionRefused(f"unexpected non-dev splits: {excluded}")
    pair_path = Path(project_root) / cfg["frozen_sources"][
        "pair_manifest"]["path"]
    register_consumed(registry, pair_path, "source_pair_manifest",
                      project_root, book.binding_for(pair_path))
    pairs = {}
    with pair_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pairs[str(row["output_stem"]).strip()] = row
    frames = sorted({row["source_frame"] for row in dev_rows})
    frame_rows: dict[str, list] = {}
    for row in dev_rows:
        frame_rows.setdefault(row["source_frame"], []).append(row)
    empty_frames = sorted(set(cfg["dev_pool"]["empty_frame_names"]))
    for stem in empty_frames:
        if stem in frame_rows:
            raise AdmissionRefused(
                f"empty frame {stem} unexpectedly has rows")
        if stem not in pairs:
            raise AdmissionRefused(
                f"empty frame {stem} missing from the pair manifest")
    for stem in empty_frames:
        row = pairs[stem]
        for column, role in (("left_path", f"image_left:{stem}"),
                             ("right_path", f"image_right:{stem}")):
            register_consumed(registry, Path(project_root) / row[column],
                              role, project_root,
                              book.binding_for(Path(project_root) /
                                               row[column]))
    expectations = cfg["dev_pool"]
    groups = sorted({row["group_name"] for row in dev_rows})
    checks = {
        "frames": (len(frames) + len(empty_frames),
                   int(expectations["frames"])),
        "nonempty_frames": (len(frames),
                            int(expectations["nonempty_frames"])),
        "instances": (len(dev_rows), int(expectations["instances"])),
        "groups": (len(groups), int(expectations["groups"])),
        "empty_frames": (len(empty_frames),
                         int(expectations["empty_frames"])),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise AdmissionRefused(f"dev-pool census {name}: {actual} != "
                                   f"pinned {expected}")
    return {"dev_rows": dev_rows, "frames": frames,
            "frame_rows": frame_rows, "empty_frames": empty_frames,
            "pairs": pairs, "groups": groups, "data_root": data_root}


# ---------------------------------------------------------------------------
# identity: authoritative sets and ledger validation


def authoritative_frame_instances(manifest_samples: list[dict]) -> dict:
    """Per-frame authoritative instance-name sets derived from the frozen
    feature manifest (which is bound to the V5 rows and the protocol)."""
    frames: dict[str, set[str]] = {}
    for index, sample in enumerate(manifest_samples):
        frames.setdefault(sample["source_frame"], set()).add(
            sample["sample_name"])
    return frames


def expected_shard_keys(frame_instances: set[str]) -> dict:
    """The exact shard key sets one frame must carry."""
    return {
        "native": {f"{name}|{key}" for name in frame_instances
                   for key in NATIVE_ARRAY_KEYS},
        "grid": {f"{name}|planes" for name in frame_instances},
    }


def cache_mode_of(mode: str) -> str:
    """The unified mode semantics: the saved cache only knows
    build|smoke while the verifier CLI uses full|smoke. Every kind,
    complete, built-count and empty-frame branch derives from this one
    mapping; an unknown mode is refused outright."""
    if mode == "full":
        return "build"
    if mode == "smoke":
        return "smoke"
    raise AdmissionRefused(f"unknown mode {mode!r} (expected full|smoke)")


SAMPLE_LEDGER_FIELDS = ("sample_name", "frame", "cache_row", "class_id",
                        "class_name", "group", "split", "crop_window",
                        "stereo_valid", "stereo_reason", "has_valid_s",
                        "blocks", "w2", "h2", "offset", "fold_roles")
FRAME_LEDGER_FIELDS = ("frame", "split", "group", "instances", "note")
LEGAL_ROLES = ("inner_train", "inner_val", "outer_eval")


def require_true_int(value, field: str, minimum: int | None = None,
                     maximum: int | None = None) -> int:
    """A true integer: type(value) is int — bool/float/numeric strings
    are refused, never coerced."""
    if type(value) is not int:
        raise AdmissionRefused(
            f"{field}: expected a true int, got {type(value).__name__} "
            f"{value!r}")
    if minimum is not None and value < minimum:
        raise AdmissionRefused(f"{field}: {value} < {minimum}")
    if maximum is not None and value > maximum:
        raise AdmissionRefused(f"{field}: {value} > {maximum}")
    return value


def require_true_bool(value, field: str) -> bool:
    if type(value) is not bool:
        raise AdmissionRefused(
            f"{field}: expected a true bool, got {type(value).__name__} "
            f"{value!r}")
    return value


def require_str(value, field: str) -> str:
    if not isinstance(value, str) or isinstance(value, bool):
        raise AdmissionRefused(
            f"{field}: expected a string, got {type(value).__name__} "
            f"{value!r}")
    return value


def _check_closed_keys(row: dict, fields: tuple[str, ...],
                       label: str) -> None:
    keys = set(row)
    want = set(fields)
    missing = sorted(want - keys)
    extra = sorted(keys - want)
    if missing:
        raise AdmissionRefused(f"{label}: missing fields {missing[:5]}")
    if extra:
        raise AdmissionRefused(f"{label}: unknown fields {extra[:5]}")


def check_samples_ledger(rows: list[dict], manifest_samples: list[dict],
                         authority: dict, run_root_label: str,
                         require_complete: bool = True) -> None:
    """Every samples-ledger row must match the frozen manifest row, the
    protocol authority and the letterbox geometry, field by field, with
    strict JSON types and a closed field set."""
    if require_complete and len(rows) != len(manifest_samples):
        raise AdmissionRefused(
            f"{run_root_label}: {len(rows)} sample ledger rows != "
            f"{len(manifest_samples)} frozen manifest rows")
    by_name = {sample["sample_name"]: (index, sample)
               for index, sample in enumerate(manifest_samples)}
    seen = set()
    for row in rows:
        _check_closed_keys(row, SAMPLE_LEDGER_FIELDS, "samples ledger")
        name = require_str(row.get("sample_name"), "sample_name")
        if name not in by_name:
            raise AdmissionRefused(
                f"{run_root_label}: ledger row {name!r} is not in the "
                "frozen manifest")
        if name in seen:
            raise AdmissionRefused(f"{run_root_label}: duplicate ledger "
                                   f"row {name}")
        seen.add(name)
        index, manifest_row = by_name[name]
        require_true_int(row["cache_row"], f"{name}.cache_row", 0,
                         len(manifest_samples) - 1)
        if row["cache_row"] != index:
            raise AdmissionRefused(
                f"{name}: cache_row {row['cache_row']} != the frozen "
                f"manifest row {index} (the G256 reference)")
        require_true_int(row["class_id"], f"{name}.class_id", 0, 3)
        require_str(row["class_name"], f"{name}.class_name")
        require_str(row["group"], f"{name}.group")
        require_str(row["split"], f"{name}.split")
        require_str(row["frame"], f"{name}.frame")
        identity = authority[name]["identity"]
        for field in ("class_id", "class_name", "group", "split"):
            if row[field] != identity[field]:
                raise AdmissionRefused(
                    f"{name}: ledger {field} {row[field]!r} != protocol "
                    f"authority {identity[field]!r}")
        if row["frame"] != identity["source_frame"]:
            raise AdmissionRefused(f"{name}: ledger frame mismatch")
        window = row["crop_window"]
        if not isinstance(window, list) or len(window) != 4 or \
                not all(type(v) is int for v in window):
            raise AdmissionRefused(
                f"{name}: crop_window must be four true ints")
        frozen_window = tuple(int(v) for v in
                              manifest_row["fix_polar"]["crop_window"])
        if tuple(window) != frozen_window:
            raise AdmissionRefused(f"{name}: crop_window differs from "
                                   "the frozen manifest")
        require_true_bool(row["stereo_valid"], f"{name}.stereo_valid")
        require_str(row["stereo_reason"], f"{name}.stereo_reason")
        if bool(row["stereo_valid"]) != \
                bool(manifest_row["fix_polar"]["stereo_valid"]):
            raise AdmissionRefused(f"{name}: stereo_valid differs from "
                                   "the frozen manifest")
        if str(row["stereo_reason"]) != \
                str(manifest_row["fix_polar"]["stereo_reason"]):
            raise AdmissionRefused(f"{name}: stereo_reason differs from "
                                   "the frozen manifest")
        require_true_bool(row["has_valid_s"], f"{name}.has_valid_s")
        if row["has_valid_s"] != bool(manifest_row["has_valid_s"]):
            raise AdmissionRefused(f"{name}: has_valid_s differs")
        blocks = row["blocks"]
        if not isinstance(blocks, dict) or set(blocks) != \
                {"interior_valid", "band_valid"}:
            raise AdmissionRefused(f"{name}: blocks must carry exactly "
                                   "interior_valid/band_valid")
        for key in blocks:
            require_true_bool(blocks[key], f"{name}.blocks.{key}")
        if {k: v for k, v in blocks.items()} != \
                {k: bool(v) for k, v in manifest_row["blocks"].items()}:
            raise AdmissionRefused(f"{name}: blocks differ")
        require_true_int(row["w2"], f"{name}.w2", 1, 128)
        require_true_int(row["h2"], f"{name}.h2", 1, 128)
        offset = row["offset"]
        if not isinstance(offset, list) or len(offset) != 2 or \
                not all(type(v) is int for v in offset):
            raise AdmissionRefused(
                f"{name}: offset must be two true ints")
        roles = row["fold_roles"]
        if not isinstance(roles, dict) or \
                set(roles) != set(authority[name]["roles"]):
            raise AdmissionRefused(
                f"{name}: fold_roles must cover exactly the protocol "
                "folds")
        for fold_id, role in roles.items():
            require_str(role, f"{name}.fold_roles.{fold_id}")
            if role not in LEGAL_ROLES:
                raise AdmissionRefused(
                    f"{name}: fold_roles.{fold_id} {role!r} is not a "
                    "legal role")
            if role != authority[name]["roles"][fold_id]:
                raise AdmissionRefused(
                    f"{name}: fold_roles.{fold_id} {role!r} != protocol "
                    f"{authority[name]['roles'][fold_id]!r}")
        w, h = frozen_window[2] - frozen_window[0], \
            frozen_window[3] - frozen_window[1]
        expect_w2, expect_h2 = scaled_size(w, h)
        expect_offset = canvas_offset(expect_w2, expect_h2)
        if (row["w2"], row["h2"]) != (expect_w2, expect_h2) or \
                tuple(offset) != expect_offset:
            raise AdmissionRefused(
                f"{name}: ledger w2/h2/offset do not match the frozen "
                "letterbox geometry")
        NativeInstanceArrays.validate_geometry(w, h, row["w2"],
                                               row["h2"],
                                               tuple(offset))


def check_frames_ledger(rows: list[dict], expected_frames: dict,
                        empty_frame_names: list[str]) -> None:
    """The frame ledger is a closed set: frame/split/group/instances/
    note, with true-int counts, exact frame coverage (no extra,
    duplicate or missing frames), per-item group/split for non-empty
    frames and zero instances for the fixed empty frames. expected_frames
    maps frame -> {count, split, group} (empty frames need count 0)."""
    seen = {}
    for row in rows:
        _check_closed_keys(row, FRAME_LEDGER_FIELDS, "frames ledger")
        frame = require_str(row.get("frame"), "frame")
        if frame in seen:
            raise AdmissionRefused(f"duplicate frame ledger row {frame}")
        require_true_int(row["instances"], f"{frame}.instances", 0)
        if row["instances"] > 0:
            require_str(row["split"], f"{frame}.split")
            require_str(row["group"], f"{frame}.group")
        seen[frame] = row
    extra = sorted(set(seen) - set(expected_frames))
    if extra:
        raise AdmissionRefused(
            f"frame ledger has extra frames: {extra[:5]}")
    for frame, expected in expected_frames.items():
        if frame not in seen:
            raise AdmissionRefused(f"frame ledger lacks {frame}")
        row = seen[frame]
        if row["instances"] != int(expected["count"]):
            raise AdmissionRefused(
                f"frame {frame}: ledger instances {row['instances']} != "
                f"expected {expected['count']}")
        if int(expected["count"]) > 0:
            if row["split"] != expected["split"] or \
                    row["group"] != expected["group"]:
                raise AdmissionRefused(
                    f"frame {frame}: ledger group/split differ from the "
                    "census")
    for frame in empty_frame_names:
        if frame not in seen:
            raise AdmissionRefused(
                f"empty frame {frame} missing from the frame ledger")
        if seen[frame]["instances"] != 0:
            raise AdmissionRefused(
                f"empty frame {frame} has instances")


DEFINITION_FIELDS = ("experiment", "cache_schema",
                     "implementation_revision", "kind", "mode",
                     "complete", "grid_schema", "grid_size",
                     "permutation_schema", "upsample_version",
                     "content_seed", "planes", "recipe",
                     "recipe_source", "recipe_source_sha256",
                     "source_identity", "native_arrays", "dev_pool",
                     "built", "run_id")
SOURCE_IDENTITY_FIELDS = ("config_sha256", "protocol_sha256",
                          "v5_fingerprint")


def check_definition(definition: dict, *, mode: str,
                     config: dict, recipe: dict,
                     recipe_source_sha: str, planes: tuple,
                     built_frames: int, built_instances: int,
                     empty_frames: list[str]) -> None:
    """The saved definition must carry the closed field set with the
    expected values under the unified cache_mode mapping; an old,
    partial or over-specified definition can never pass."""
    _check_closed_keys(definition, DEFINITION_FIELDS, "definition")
    cache_mode = cache_mode_of(mode)
    expected = {
        "experiment": EXPERIMENT,
        "cache_schema": CACHE_SCHEMA,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "kind": KIND_FULL if cache_mode == "build" else KIND_SMOKE,
        "mode": cache_mode,
        "complete": cache_mode == "build",
        "grid_schema": GRID_SCHEMA,
        "grid_size": GRID_SIZE,
        "permutation_schema": PERMUTATION_SCHEMA,
        "content_seed": 2026,
        "upsample_version": recipe["upsample_version"],
        "planes": list(planes),
        "recipe": {key: recipe[key] for key in sorted(recipe)},
        "recipe_source_sha256": recipe_source_sha,
    }
    for field, want in expected.items():
        got = definition.get(field)
        if got != want:
            raise AdmissionRefused(
                f"definition.{field}: got {got!r} != expected {want!r}")
    require_true_bool(definition["complete"], "definition.complete")
    require_true_int(definition["content_seed"],
                     "definition.content_seed")
    require_true_int(definition["grid_size"], "definition.grid_size")
    identity = definition.get("source_identity")
    if not isinstance(identity, dict) or \
            set(identity) != set(SOURCE_IDENTITY_FIELDS):
        raise AdmissionRefused(
            "definition.source_identity must carry exactly "
            "config_sha256/protocol_sha256/v5_fingerprint")
    for field in SOURCE_IDENTITY_FIELDS:
        require_str(identity[field],
                    f"definition.source_identity.{field}")
    sources = config["frozen_sources"]
    for field, expected_value in (
            ("protocol_sha256", sources["protocol"]["file_sha256"]),
            ("v5_fingerprint", sources["v5_data"]["fingerprint"])):
        if identity[field] != expected_value:
            raise AdmissionRefused(f"definition.source_identity.{field} differs "
                                   "from the frozen source")
    require_str(definition["recipe_source"], "definition.recipe_source")
    if definition["recipe_source"] != sources["old_experiment_config"]["path"]:
        raise AdmissionRefused("definition.recipe_source differs from the frozen source")
    expected_native = {"L": "uint8", "R": "float32", "q": "float32",
                       "M": "uint8", "V": "uint8", "region": "uint8",
                       "quality": "float32(4)"}
    if definition["native_arrays"] != expected_native:
        raise AdmissionRefused("definition.native_arrays differs from the native contract")
    pool = definition.get("dev_pool")
    if not isinstance(pool, dict):
        raise AdmissionRefused("definition.dev_pool missing")
    pinned = config["dev_pool"]
    for field in ("frames", "nonempty_frames", "empty_frames",
                  "instances", "groups", "s_valid_true",
                  "s_valid_false"):
        require_true_int(pool.get(field), f"definition.dev_pool.{field}")
        if pool[field] != int(pinned[field]):
            raise AdmissionRefused(
                f"definition.dev_pool.{field}: {pool[field]!r} != "
                f"pinned {pinned[field]!r}")
    if sorted(pool.get("empty_frame_names", [])) != \
            sorted(pinned["empty_frame_names"]):
        raise AdmissionRefused("definition.dev_pool.empty_frame_names "
                               "differ from the pinned names")
    built = definition.get("built")
    if not isinstance(built, dict) or \
            set(built) != {"frames", "instances", "empty_frames"}:
        raise AdmissionRefused("definition.built must carry exactly "
                               "frames/instances/empty_frames")
    require_true_int(built["frames"], "definition.built.frames", 0)
    require_true_int(built["instances"], "definition.built.instances", 0)
    if not isinstance(built["empty_frames"], list):
        raise AdmissionRefused("definition.built.empty_frames must be "
                               "a list")
    if cache_mode == "build":
        if built["frames"] != built_frames or \
                built["instances"] != built_instances:
            raise AdmissionRefused(
                "definition.built does not match the approved counts")
        if sorted(built["empty_frames"]) != sorted(empty_frames):
            raise AdmissionRefused("definition.built.empty_frames "
                                   "differ from the approved frames")
    require_str(definition["run_id"], "definition.run_id")


# ---------------------------------------------------------------------------
# binding: the expected artifact set and the strict inventory


def expected_artifact_paths(authoritative_frames: dict,
                            built_nonempty_frames) -> list[str]:
    """The artifact path set derived from the authoritative identity:
    three metadata files plus one native/grid/permutation shard per
    non-empty frame that had to be built. The binding's own report is
    never the source of the allowed set."""
    rels = list(META_RELS)
    for stem in sorted(built_nonempty_frames):
        if stem not in authoritative_frames or \
                not authoritative_frames[stem]:
            raise AdmissionRefused(
                f"built frame {stem} is not in the authoritative set")
        rels.append(f"cache/native/frame_{stem}.npz")
        rels.append(f"cache/grid/frame_{stem}.npz")
        rels.append(f"cache/permutation/frame_{stem}.json")
    return rels


def verify_inventory(run_root: Path, binding: dict, registry,
                     project_root: Path,
                     expected_rels: list[str]) -> dict:
    """Full inventory check against the DERIVED expected set: metadata
    present, no duplicate/unknown entries, no alias duplicates, no
    duplicate npz-key records, no extra file under cache/, every
    artifact verified against its recorded bytes/sha BEFORE it is
    loaded, npz keys exactly equal to the recorded dtype/shape set,
    non-array files without dtype/shape records. Every artifact is
    registered for final-state protection."""
    run_root = Path(run_root)
    if binding.get("experiment") != EXPERIMENT or \
            binding.get("cache_schema") != CACHE_SCHEMA or \
            binding.get("implementation_revision") != \
            IMPLEMENTATION_REVISION:
        raise AdmissionRefused("the binding does not carry the "
                               "review-fix cache contract")
    artifacts: dict[str, dict] = {}
    for record in binding.get("artifacts", []):
        rel = record["path"]
        if not isinstance(rel, str) or not rel.startswith("cache/"):
            raise AdmissionRefused(
                f"artifact path outside cache/: {rel!r}")
        if rel in artifacts:
            raise AdmissionRefused(f"duplicate inventory entry: {rel}")
        canonical = rel.lower()
        if canonical != rel and canonical in artifacts:
            raise AdmissionRefused(
                f"path alias duplicate: {rel} vs "
                f"{artifacts[canonical]['path']}")
        artifacts[canonical] = record
    want = {rel.lower(): rel for rel in expected_rels}
    missing = sorted(set(want) - set(artifacts))
    if missing:
        raise AdmissionRefused(
            f"expected artifacts missing from the binding: "
            f"{missing[:5]}")
    extra = sorted(set(artifacts) - set(want))
    if extra:
        raise AdmissionRefused(
            f"binding carries unexpected artifacts: {extra[:5]}")
    for rel in META_RELS:
        if rel not in artifacts:
            raise AdmissionRefused(
                f"metadata missing from the binding: {rel}")
    if binding.get("artifact_count") != len(binding.get("artifacts", [])):
        raise AdmissionRefused("binding.artifact_count does not match "
                               "the artifact list")
    cache_root = run_root / "cache"
    actual = {}
    for path in sorted(cache_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(run_root).as_posix()
            if rel == BINDING_REL:
                continue
            actual[rel.lower()] = path
    extra_files = sorted(set(actual) - set(artifacts))
    if extra_files:
        raise AdmissionRefused(
            f"extra cache files not in the binding: {extra_files[:5]}")
    for rel, record in sorted(artifacts.items()):
        path = run_root / record["path"]
        expectation = {"source": "cache_binding",
                       "expected_sha256": str(record["sha256"]),
                       "expected_bytes": int(record["bytes"])}
        register_consumed(registry, path,
                          f"cache:{record['kind']}:{rel}",
                          project_root, expectation)
        if record["kind"] not in ARTIFACT_KINDS:
            raise AdmissionRefused(f"{rel}: unknown artifact kind")
        if record["kind"] == "metadata":
            if record["path"].count("/") != 1:
                raise AdmissionRefused(
                    f"{rel}: metadata must sit directly under cache/")
            if record.get("npz_keys"):
                raise AdmissionRefused(
                    f"{rel}: a metadata file must not carry npz key "
                    "records")
            continue
        expected_dir = {"native": "cache/native/",
                        "grid": "cache/grid/",
                        "permutation": "cache/permutation/"}[
            record["kind"]]
        if not record["path"].startswith(expected_dir):
            raise AdmissionRefused(
                f"{record['path']}: kind {record['kind']} must live "
                f"under {expected_dir}")
        if not rel.endswith((".npz", ".json")):
            raise AdmissionRefused(f"{rel}: unexpected artifact suffix")
        if rel.endswith(".npz"):
            recorded = record.get("npz_keys")
            if not isinstance(recorded, list) or not recorded:
                raise AdmissionRefused(
                    f"{rel}: npz artifacts must record every key's "
                    "dtype/shape")
            seen_keys = set()
            for entry in recorded:
                key = entry.get("key")
                if key in seen_keys:
                    raise AdmissionRefused(
                        f"{rel}: duplicate npz-key record {key!r}")
                seen_keys.add(key)
            with np.load(path) as payload:
                actual_keys = {key: {"dtype": str(payload[key].dtype),
                                     "shape": list(payload[key].shape)}
                               for key in payload.files}
            recorded_map = {entry["key"]: entry for entry in recorded}
            if set(actual_keys) != set(recorded_map):
                only_saved = sorted(set(actual_keys) -
                                    set(recorded_map))
                only_rec = sorted(set(recorded_map) -
                                  set(actual_keys))
                raise AdmissionRefused(
                    f"{rel}: npz keys differ from the binding "
                    f"(undocumented: {only_saved[:3]}, "
                    f"missing: {only_rec[:3]})")
            for key, entry in actual_keys.items():
                want_entry = recorded_map[key]
                if entry["dtype"] != want_entry["dtype"] or \
                        entry["shape"] != list(want_entry["shape"]):
                    raise AdmissionRefused(
                        f"{rel}|{key}: dtype/shape {entry} != recorded "
                        f"{want_entry}")
        else:
            if record.get("npz_keys"):
                raise AdmissionRefused(
                    f"{rel}: a JSON artifact must not carry npz key "
                    "records")
    return {"artifacts": len(artifacts)}


# ---------------------------------------------------------------------------
# native array contract (independent of the binding records)


def check_native_arrays(saved_native: dict, name: str, crop_w: int,
                        crop_h: int) -> None:
    """The loaded native arrays must satisfy the actual dtype/shape/
    value contract — the binding records are never trusted instead."""
    expected = {"L": np.uint8, "R": np.float32, "q": np.float32,
                "M": np.uint8, "V": np.uint8, "region": np.uint8,
                "quality": np.float32}
    for key in NATIVE_ARRAY_KEYS:
        array = saved_native.get(f"{name}|{key}")
        if array is None:
            raise AdmissionRefused(f"{name}|{key}: missing")
        if array.dtype != expected[key]:
            raise AdmissionRefused(
                f"{name}|{key}: dtype {array.dtype} != "
                f"{expected[key]}")
    for key in ("L", "R", "q", "M", "V", "region"):
        if saved_native[f"{name}|{key}"].shape != (crop_h, crop_w):
            raise AdmissionRefused(
                f"{name}|{key}: shape "
                f"{saved_native[f'{name}|{key}'].shape} != "
                f"{(crop_h, crop_w)}")
    quality = saved_native[f"{name}|quality"]
    if quality.shape != (4,):
        raise AdmissionRefused(f"{name}|quality: shape {quality.shape} "
                               "!= (4,)")
    if not np.all(np.isfinite(quality)):
        raise AdmissionRefused(f"{name}|quality: non-finite values")
    mask = saved_native[f"{name}|M"]
    valid = saved_native[f"{name}|V"]
    region = saved_native[f"{name}|region"]
    if not np.all((mask == 0) | (mask == 1)):
        raise AdmissionRefused(f"{name}|M: not binary")
    if not np.all((valid == 0) | (valid == 1)):
        raise AdmissionRefused(f"{name}|V: not binary")
    if not np.all((region == 0) | (region == 1) | (region == 2)):
        raise AdmissionRefused(f"{name}|region: values outside 0/1/2")
    if not np.all((valid > 0) <= (mask > 0)):
        raise AdmissionRefused(f"{name}|V is not a subset of M")
    invalid = valid == 0
    if not np.all(saved_native[f"{name}|q"][invalid] ==
                  np.float32(0.0)):
        raise AdmissionRefused(
            f"{name}|q: not +0 at invalid positions")
    if not np.all(saved_native[f"{name}|R"][invalid] ==
                  np.float32(0.0)):
        raise AdmissionRefused(
            f"{name}|R: not +0 at invalid positions")


# ---------------------------------------------------------------------------
# permutation plan validation


PLAN_TOP_FIELDS = ("sample_name", "schema", "content_seed", "regions")
PLAN_REGION_FIELDS = ("movable_cells", "groups", "moved_indices",
                      "value_changed_cells")
PLAN_GROUP_FIELDS = ("coverage_bits_hex8", "cells", "seed64", "permuted",
                     "moved_indices", "value_changed_cells",
                     "cell_indices", "permutation")
PLAN_SINGLETON_FIELDS = ("coverage_bits_hex8", "cells", "permuted",
                         "moved_indices", "value_changed_cells")


def check_permutation_plan(plan: dict, name: str,
                           frame_instances: set[str]) -> dict:
    """Strict closed-set/type/bijection validation of one saved
    permutation plan against the frozen rule; returns the plan. The
    per-instance plan set equality is checked by the caller."""
    _check_closed_keys(plan, PLAN_TOP_FIELDS, f"{name} plan top level")
    require_str(plan["sample_name"], f"{name}.plan.sample_name")
    require_str(plan["schema"], f"{name}.plan.schema")
    if plan["sample_name"] != name:
        raise AdmissionRefused(f"{name}: plan.sample_name mismatch")
    if plan["schema"] != PERMUTATION_SCHEMA:
        raise AdmissionRefused(f"{name}: plan.schema mismatch")
    require_true_int(plan["content_seed"], f"{name}.plan.content_seed")
    if plan["content_seed"] != 2026:
        raise AdmissionRefused(f"{name}: plan.content_seed != 2026")
    regions = plan["regions"]
    if not isinstance(regions, dict) or set(regions) != set(REGIONS):
        raise AdmissionRefused(f"{name}: plan regions must be exactly "
                               "interior/band")
    for region in REGIONS:
        record = regions[region]
        _check_closed_keys(record, PLAN_REGION_FIELDS,
                           f"{name}.{region} plan")
        require_true_int(record["movable_cells"],
                         f"{name}.{region}.movable_cells", 0)
        require_true_int(record["moved_indices"],
                         f"{name}.{region}.moved_indices", 0)
        require_true_int(record["value_changed_cells"],
                         f"{name}.{region}.value_changed_cells", 0)
        groups = record["groups"]
        if not isinstance(groups, dict):
            raise AdmissionRefused(f"{name}.{region}: groups must be "
                                   "a dict")
        movable_seen = 0
        moved_seen = 0
        changed_seen = 0
        all_members: list[int] = []
        for bits, group in groups.items():
            if not (isinstance(bits, str) and len(bits) == 8 and
                    all(c in "0123456789abcdef" for c in bits)):
                raise AdmissionRefused(
                    f"{name}.{region}: group key {bits!r} is not an "
                    "8-hex coverage bit pattern")
            require_true_int(group["cells"], f"{name}.{region}.cells",
                             1)
            if group["cells"] > 1:
                _check_closed_keys(group, PLAN_GROUP_FIELDS,
                                   f"{name}.{region}[{bits}]")
                require_true_bool(group["permuted"],
                                  f"{name}.{region}[{bits}].permuted")
                if group["permuted"] is not True:
                    raise AdmissionRefused(
                        f"{name}.{region}[{bits}]: a multi-cell group "
                        "must be permuted=true")
                require_true_int(group["seed64"],
                                 f"{name}.{region}[{bits}].seed64",
                                 0, 2 ** 64 - 1)
                members = group["cell_indices"]
                perm = group["permutation"]
                if not isinstance(members, list) or \
                        not all(type(v) is int for v in members) or \
                        not isinstance(perm, list) or \
                        not all(type(v) is int for v in perm):
                    raise AdmissionRefused(
                        f"{name}.{region}[{bits}]: cell_indices/"
                        "permutation must be lists of true ints")
                if len(members) != group["cells"] or \
                        len(perm) != group["cells"]:
                    raise AdmissionRefused(
                        f"{name}.{region}[{bits}]: member/perm counts "
                        "disagree with cells")
                if sorted(perm) != list(range(group["cells"])):
                    raise AdmissionRefused(
                        f"{name}.{region}[{bits}]: the permutation is "
                        "not a bijection")
                require_true_int(
                    group["moved_indices"],
                    f"{name}.{region}[{bits}].moved_indices",
                    0, group["cells"])
                require_true_int(
                    group["value_changed_cells"],
                    f"{name}.{region}[{bits}].value_changed_cells",
                    0, group["cells"])
                movable_seen += group["cells"]
                moved_seen += group["moved_indices"]
                changed_seen += group["value_changed_cells"]
                all_members.extend(members)
            else:
                _check_closed_keys(group, PLAN_SINGLETON_FIELDS,
                                   f"{name}.{region}[{bits}]")
                require_true_bool(group["permuted"],
                                  f"{name}.{region}[{bits}].permuted")
                if group["permuted"] is not False:
                    raise AdmissionRefused(
                        f"{name}.{region}[{bits}]: a singleton group "
                        "must be permuted=false")
                for field in ("moved_indices", "value_changed_cells"):
                    require_true_int(group[field],
                                     f"{name}.{region}[{bits}].{field}", 0, 0)
                movable_seen += 1
                # Singleton plans deliberately omit cell_indices. Their
                # actual position is recovered from saved coverage by the
                # independent plan comparison; a shared -1 sentinel would
                # falsely make every pair of singleton groups overlap.
        if movable_seen != record["movable_cells"]:
            raise AdmissionRefused(
                f"{name}.{region}: group cells sum {movable_seen} != "
                f"movable_cells {record['movable_cells']}")
        if moved_seen != record["moved_indices"]:
            raise AdmissionRefused(
                f"{name}.{region}: group moved sum != moved_indices")
        if changed_seen != record["value_changed_cells"]:
            raise AdmissionRefused(
                f"{name}.{region}: group changed sum != "
                "value_changed_cells")
        if len(all_members) != len(set(all_members)):
            raise AdmissionRefused(
                f"{name}.{region}: a cell belongs to multiple groups")
    return plan
