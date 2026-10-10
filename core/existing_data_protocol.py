"""Existing-data protocol: baseline identities, dev-pool inventory, folds.

The round works on the data that already exists (old train + val only; the
old test split is excluded from every dev use). This module is the single
place that defines

- **baseline identities** — SHA256 verification of the explicitly pinned
  weight/evidence paths (never "the newest directory");
- **the dev-pool inventory** — frames, GT instances, V5 NPZ samples and
  their correspondence, with empty-GT frames and NPZ-less GT instances kept
  and listed instead of silently dropped;
- **the group-isolated fold assignment** — whole capture groups (each one a
  continuous 45-frame sequence) assigned to 3 outer folds by a
  deterministic, seeded, class-instance-count-balanced greedy pass, each
  outer-train part further split into inner_train/inner_val at the group
  level.

It reuses the project's existing fingerprint and admission machinery
(:mod:`core.material_mask_dataset`, :mod:`core.fusion_dataset`); it does
not define a second fingerprint. Nothing here modifies the underlying
datasets: all outputs are new manifests that reference the original files.

Honesty boundary (recorded in every protocol report): the fold protocol
proves *training-level group isolation* over a pool whose original val part
already took part in project development. It is not a new blind test, and
weights trained on the old train/val are not OOF evidence under it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_VERSION = "material_existing_data_v1"
FOLD_STATUS_OK = "OK"
FOLD_STATUS_BLOCKED = "BLOCKED"

CLASS_IDS = (0, 1, 2, 3)

BASELINE_KEYS = (
    "model_a",
    "e1_candidate",
    "e4_candidate",
    "independent_gray_fallback",
    "official_mit_b0",
)

# Configuration fields that must be present before anything runs. Missing
# fields are a hard error: a half-specified protocol must not silently
# degrade into defaults.
REQUIRED_CONFIG_FIELDS = (
    ("experiment",),
    ("data", "root"),
    ("data", "fingerprint"),
    ("data", "parent_v4_fingerprint"),
    ("data", "dev_pool_splits"),
    ("data", "excluded_splits"),
    ("data", "split_assignment"),
    ("data", "label_root"),
    ("baselines", "model_a", "path"),
    ("baselines", "model_a", "sha256"),
    ("baselines", "e1_candidate", "path"),
    ("baselines", "e1_candidate", "sha256"),
    ("baselines", "e4_candidate", "path"),
    ("baselines", "e4_candidate", "sha256"),
    ("baselines", "independent_gray_fallback", "path"),
    ("baselines", "independent_gray_fallback", "sha256"),
    ("baselines", "official_mit_b0", "path"),
    ("baselines", "official_mit_b0", "sha256"),
    ("protocol", "n_outer_folds"),
    ("protocol", "seed"),
    ("detection", "match_iou_threshold"),
    ("features", "feature_definition_version"),
)


# ---------------------------------------------------------------------------
# Configuration


def load_experiment_config(path: str | Path) -> dict:
    """Load and structurally validate the experiment YAML.

    Every field in ``REQUIRED_CONFIG_FIELDS`` must be present with a
    non-empty value; there is no default-filling. Returns the raw dict.
    """
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"experiment config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"experiment config {path} is not a mapping")
    missing = [
        ".".join(field_path)
        for field_path in REQUIRED_CONFIG_FIELDS
        if _lookup(config, field_path) in (None, "", [])
    ]
    if missing:
        raise ValueError(
            f"experiment config {path} is missing required fields: {missing}"
        )
    return config


def _lookup(mapping: Any, field_path: Sequence[str]) -> Any:
    value = mapping
    for key in field_path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def sha256_of_file(path: str | Path) -> str:
    """sha256 hex digest of one file (reuses the project's own helper)."""
    from core.fusion_dataset import file_digest

    return file_digest(path)


# ---------------------------------------------------------------------------
# Baseline identities


def verify_baseline_identities(
    config: dict, project_root: Path | None = None
) -> list[dict]:
    """Verify every pinned baseline file exists and matches its SHA256.

    Paths come from the config (never re-resolved from disk by freshness).
    A missing file raises ``FileNotFoundError``; a digest mismatch raises
    ``ValueError``. The historical D0 evidence files are also recorded
    (with their digests) so the detection step can bind the baseline it
    re-checks against, but they carry no expected digest.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    baselines = config["baselines"]
    records: list[dict] = []
    for key in BASELINE_KEYS:
        entry = baselines.get(key)
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"experiment config baselines.{key} is missing")
        resolved = root / entry["path"]
        if not resolved.is_file():
            raise FileNotFoundError(
                f"baseline {key} ({entry['path']}) not found at {resolved}; "
                "the affected steps are refused rather than re-resolved"
            )
        digest = sha256_of_file(resolved)
        expected = entry.get("sha256", "")
        if not expected:
            raise ValueError(f"baselines.{key} records no sha256")
        if digest != expected:
            raise ValueError(
                f"baseline {key} sha256 {digest} does not match the recorded "
                f"{expected} ({entry['path']}); refusing to build a protocol "
                "on a changed weight file"
            )
        records.append(
            {
                "key": key,
                "path": entry["path"],
                "resolved": str(resolved),
                "sha256": digest,
                "identity_fields": {
                    name: value
                    for name, value in entry.items()
                    if name not in ("path", "sha256")
                },
            }
        )
    evidence = baselines.get("historical_d0_evidence")
    if isinstance(evidence, dict):
        for field_name in ("frames_jsonl", "summary_json"):
            relative = evidence.get(field_name)
            if not relative:
                continue
            resolved = root / relative
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"historical D0 evidence {field_name} ({relative}) not "
                    f"found at {resolved}; the D0 reproduction check cannot "
                    "be bound to its baseline"
                )
            records.append(
                {
                    "key": f"historical_d0_{field_name}",
                    "path": relative,
                    "resolved": str(resolved),
                    "sha256": sha256_of_file(resolved),
                    "identity_fields": {
                        name: value
                        for name, value in evidence.items()
                        if field_name not in (name,)
                    },
                }
            )
    return records


# ---------------------------------------------------------------------------
# Dev-pool inventory


def read_split_rows(
    split_assignment: str | Path, splits: Iterable[str]
) -> list[dict]:
    """The frame rows of the authoritative split assignment for ``splits``.

    Each row carries the frame stem, its capture group and the recorded
    block id (the only recorded sequence structure). Duplicate stems are
    refused: a frame belongs to exactly one split and one group.
    """
    wanted = set(splits)
    path = Path(split_assignment)
    if not path.is_file():
        raise FileNotFoundError(f"split assignment not found: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen: dict[str, str] = {}
    selected: list[dict] = []
    for row in rows:
        if row["split"] not in wanted:
            continue
        stem = row["output_stem"]
        if stem in seen:
            raise ValueError(
                f"split assignment lists frame {stem!r} twice "
                f"({seen[stem]!r} and {row['split']!r})"
            )
        seen[stem] = row["split"]
        selected.append(
            {
                "frame_id": stem,
                "split": row["split"],
                "group": row["group_name"],
                "block_id": row.get("block_id", ""),
                "left_path": row.get("left_path", ""),
            }
        )
    if not selected:
        raise ValueError(f"no frames for splits {sorted(wanted)} in {path}")
    return selected


def read_label_class_ids(label_path: Path) -> list[int]:
    """Parse one YOLO segmentation label file into its per-line class ids.

    Inventory-level counting only (the authoritative geometry is built
    elsewhere by ``core.gt_geometry.build_gt_geometry``). A malformed line
    raises ``ValueError`` naming the file; a missing file returns ``[]``
    with the caller recording the absence.
    """
    if not label_path.is_file():
        raise FileNotFoundError(str(label_path))
    class_ids: list[int] = []
    for line_number, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = raw.strip()
        if not text:
            continue
        head = text.split()[0]
        try:
            class_id = int(head)
        except ValueError as error:
            raise ValueError(
                f"{label_path}:{line_number}: class id {head!r} is not an integer"
            ) from error
        if class_id not in CLASS_IDS:
            raise ValueError(
                f"{label_path}:{line_number}: unknown class id {class_id}"
            )
        class_ids.append(class_id)
    return class_ids


def read_manifest_rows(data_root: str | Path, splits: Iterable[str]) -> list[dict]:
    """The V5 manifest rows for ``splits`` (dicts, order preserved)."""
    wanted = set(splits)
    path = Path(data_root) / "dataset_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in wanted]
    if not rows:
        raise ValueError(f"no manifest rows for splits {sorted(wanted)} in {path}")
    return rows


def build_dev_inventory(
    *,
    split_rows: Sequence[dict],
    manifest_rows: Sequence[dict],
    label_root: str | Path,
    data_root: str | Path,
    project_root: Path | None = None,
    class_names: Sequence[str] = ("metal_submarine", "plastic_submarine",
                                   "plastic_fish", "real_fish"),
) -> dict:
    """Inventory the dev pool: frames, GT instances, NPZ samples, mapping.

    Everything the round must not lose is kept and single-listed:
    frames without any GT instance, GT instances without an NPZ sample and
    NPZ samples without a GT source are recorded explicitly, never dropped.
    Physical-object identity has no source in this dataset, so it is
    recorded as ``unknown`` (never guessed from file names).
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    label_root = Path(label_root)
    data_root = Path(data_root)

    frames: list[dict] = []
    anomalies: list[dict] = []
    for row in split_rows:
        stem = row["frame_id"]
        label_path = Path(label_root) / row["split"] / f"{stem}.txt"
        resolved_label = label_path if label_path.is_absolute() else root / label_path
        try:
            ids = read_label_class_ids(resolved_label)
            error = None
            label_exists = True
        except FileNotFoundError:
            # A frame without a label file is a legitimate no-target frame:
            # it stays listed (empty GT) and is not an anomaly. If a label
            # was expected, the NPZ correspondence below flags the mismatch.
            ids, error, label_exists = [], None, False
        except ValueError as error:
            ids, error, label_exists = [], str(error), True
        counts: dict[int, int] = {}
        for class_id in ids:
            counts[class_id] = counts.get(class_id, 0) + 1
        frames.append(
            {
                "frame_id": stem,
                "split": row["split"],
                "group": row["group"],
                "block_id": row["block_id"],
                "label_path": str(label_path).replace("\\", "/"),
                "label_exists": label_exists,
                "label_error": error,
                "gt_instances": len(ids),
                "gt_class_ids": ids,
                "gt_class_counts": {str(k): v for k, v in sorted(counts.items())},
            }
        )
        if error:
            anomalies.append(
                {"frame_id": stem, "kind": "label_error", "reason": error}
            )

    frame_by_id = {frame["frame_id"]: frame for frame in frames}
    if len(frame_by_id) != len(frames):  # already enforced by read_split_rows
        raise ValueError("duplicate frame ids in the dev pool")

    samples: list[dict] = []
    gt_without_npz: list[dict] = []
    npz_without_gt: list[dict] = []
    npz_by_key: dict[tuple[str, int], dict] = {}
    for row in manifest_rows:
        stem = row["source_frame"]
        object_index = int(row["object_index"])
        class_id = int(row["class_id"])
        npz_path = row["npz_path"]
        if not (data_root / npz_path).is_file():
            anomalies.append(
                {
                    "sample_name": row["sample_name"],
                    "kind": "npz_missing",
                    "reason": f"{npz_path} does not exist under {data_root}",
                }
            )
            continue
        sample = {
            "sample_name": row["sample_name"],
            "npz_path": npz_path,
            "split": row["split"],
            "group": row["group_name"],
            "source_frame": stem,
            "object_index": object_index,
            "class_id": class_id,
            "class_name": row["class_name"],
            "physical_object_id": "unknown",
        }
        samples.append(sample)
        key = (stem, object_index)
        if key in npz_by_key:
            anomalies.append(
                {
                    "sample_name": row["sample_name"],
                    "kind": "duplicate_npz_key",
                    "reason": f"(frame, object_index) {key} already held by "
                    f"{npz_by_key[key]['sample_name']}",
                }
            )
            continue
        npz_by_key[key] = sample

    for frame in frames:
        for object_index, class_id in enumerate(frame["gt_class_ids"]):
            sample = npz_by_key.get((frame["frame_id"], object_index))
            if sample is None:
                gt_without_npz.append(
                    {
                        "frame_id": frame["frame_id"],
                        "object_index": object_index,
                        "class_id": class_id,
                        "class_name": class_names[class_id]
                        if 0 <= class_id < len(class_names)
                        else None,
                    }
                )
                continue
            if sample["class_id"] != class_id:
                anomalies.append(
                    {
                        "sample_name": sample["sample_name"],
                        "kind": "class_mismatch",
                        "reason": f"manifest class {sample['class_id']} != label "
                        f"class {class_id}",
                    }
                )
            if sample["split"] != frame["split"] or sample["group"] != frame["group"]:
                anomalies.append(
                    {
                        "sample_name": sample["sample_name"],
                        "kind": "provenance_mismatch",
                        "reason": "manifest split/group differs from the split "
                        "assignment row of its source frame",
                    }
                )
    matched_keys = {
        (frame["frame_id"], object_index)
        for frame in frames
        for object_index in range(len(frame["gt_class_ids"]))
    }
    for key, sample in npz_by_key.items():
        if key not in matched_keys:
            npz_without_gt.append(
                {
                    "sample_name": sample["sample_name"],
                    "source_frame": key[0],
                    "object_index": key[1],
                }
            )

    class_counts = {str(c): 0 for c in CLASS_IDS}
    for frame in frames:
        for class_text, count in frame["gt_class_counts"].items():
            class_counts[class_text] = class_counts.get(class_text, 0) + count
    sample_class_counts = {str(c): 0 for c in CLASS_IDS}
    for sample in samples:
        sample_class_counts[str(sample["class_id"])] += 1
    empty_gt_frames = [f["frame_id"] for f in frames if f["gt_instances"] == 0]
    return {
        "frames": frames,
        "samples": samples,
        "anomalies": anomalies,
        "gt_without_npz": gt_without_npz,
        "npz_without_gt": npz_without_gt,
        "summary": {
            "frames": len(frames),
            "samples": len(samples),
            "empty_gt_frames": empty_gt_frames,
            "gt_instances": sum(f["gt_instances"] for f in frames),
            "gt_instances_by_class": class_counts,
            "samples_by_class": sample_class_counts,
            "gt_without_npz_count": len(gt_without_npz),
            "npz_without_gt_count": len(npz_without_gt),
            "anomaly_count": len(anomalies),
        },
    }


# ---------------------------------------------------------------------------
# Group statistics and fold assignment


def group_statistics(inventory: dict) -> dict[str, dict]:
    """Per-capture-group class instance counts over the dev pool.

    A group name that appears under two different original splits is a
    conflict that must not be smoothed over (the old split was
    group-exclusive), so it is refused here.
    """
    groups: dict[str, dict] = {}
    for frame in inventory["frames"]:
        name = frame["group"]
        entry = groups.setdefault(
            name,
            {
                "group": name,
                "splits": set(),
                "blocks": set(),
                "frames": 0,
                "empty_gt_frames": 0,
                "gt_instances": 0,
                "gt_instances_by_class": {str(c): 0 for c in CLASS_IDS},
                "samples": 0,
            },
        )
        entry["splits"].add(frame["split"])
        entry["blocks"].add(str(frame["block_id"]))
        entry["frames"] += 1
        entry["gt_instances"] += frame["gt_instances"]
        if frame["gt_instances"] == 0:
            entry["empty_gt_frames"] += 1
        for class_text, count in frame["gt_class_counts"].items():
            entry["gt_instances_by_class"][class_text] += count
    for sample in inventory["samples"]:
        groups[sample["group"]]["samples"] += 1
    for name, entry in groups.items():
        if len(entry["splits"]) > 1:
            raise ValueError(
                f"capture group {name!r} spans original splits "
                f"{sorted(entry['splits'])}; the old split was group-exclusive "
                "and this conflict must be resolved before folding"
            )
        entry["splits"] = sorted(entry["splits"])
        entry["blocks"] = sorted(entry["blocks"])
        entry["sequence_note"] = (
            "one continuous capture sequence per block; blocks are never "
            "split across folds or inner parts"
        )
    if not groups:
        raise ValueError("the dev pool has no capture groups")
    return groups


def _class_balance_score(
    counts: Sequence[int], global_totals: Sequence[int], n_bins: int
) -> float:
    """Squared relative deviation of one bin from its proportional share."""
    score = 0.0
    for current, total in zip(counts, global_totals):
        target = total / n_bins
        if target > 0:
            score += ((current - target) / target) ** 2
        elif current:  # a class with no instances must not appear anywhere
            score += 1e9
    return score


def assign_outer_folds(
    group_stats: dict[str, dict],
    *,
    n_folds: int,
    seed: int,
    class_ids: Sequence[int] = CLASS_IDS,
) -> tuple[dict[str, int], list[list[int]]]:
    """Deterministic greedy assignment of whole groups to ``n_folds`` folds.

    Groups are processed in a seeded-shuffled order (``random.Random(seed)``
    over the sorted group names); each group goes to the fold that keeps
    the *whole* assignment closest to the per-class proportional share —
    the objective is the summed squared relative deviation of **all**
    folds, so early groups spread instead of piling into one fold. Model
    scores are never consulted. Returns the ``group -> fold`` mapping and
    the per-fold class-count totals.
    """
    names = sorted(group_stats)
    order = list(names)
    random.Random(seed).shuffle(order)
    global_totals = [
        sum(group_stats[name]["gt_instances_by_class"].get(str(c), 0)
            for name in names)
        for c in class_ids
    ]
    fold_counts = [[0] * len(class_ids) for _ in range(n_folds)]
    assignment: dict[str, int] = {}
    for name in order:
        counts = [
            group_stats[name]["gt_instances_by_class"].get(str(c), 0)
            for c in class_ids
        ]
        best_score = None
        best_fold = 0
        for fold in range(n_folds):
            projected = [row[:] for row in fold_counts]
            for index, add in enumerate(counts):
                projected[fold][index] += add
            score = sum(
                _class_balance_score(row, global_totals, n_folds)
                for row in projected
            )
            if best_score is None or score < best_score - 1e-12:
                best_score = score
                best_fold = fold
        assignment[name] = best_fold
        for index, add in enumerate(counts):
            fold_counts[best_fold][index] += add
    return assignment, fold_counts


def split_inner(
    outer_train_groups: Sequence[str],
    group_stats: dict[str, dict],
    *,
    fraction: float,
    seed: int,
    class_ids: Sequence[int] = CLASS_IDS,
) -> tuple[list[str], list[str]]:
    """Split one fold's outer-train groups into inner_train / inner_val.

    Whole groups only (a continuous sequence is never split). Groups are
    processed in a seeded-shuffled order; each goes to the part whose
    per-class counts are closer to its proportional share, with the
    inner_val part capped at ``ceil(len * fraction) >= 1`` groups.
    """
    names = sorted(outer_train_groups)
    order = list(names)
    random.Random(seed).shuffle(order)
    n = len(order)
    if n < 2:
        raise ValueError(
            "an outer-train part needs at least 2 groups to form an "
            "inner_train/inner_val pair"
        )
    target_val = max(1, math.ceil(n * fraction))
    totals = [
        sum(group_stats[name]["gt_instances_by_class"].get(str(c), 0)
            for name in names)
        for c in class_ids
    ]
    val_target_counts = [total * fraction for total in totals]
    train_target_counts = [total * (1.0 - fraction) for total in totals]
    val_counts = [0] * len(class_ids)
    train_counts = [0] * len(class_ids)
    val_groups: list[str] = []
    train_groups: list[str] = []
    for name in order:
        counts = [
            group_stats[name]["gt_instances_by_class"].get(str(c), 0)
            for c in class_ids
        ]
        can_val = len(val_groups) < target_val
        must_val = len(train_groups) >= n - target_val
        # Global two-bin objective: the summed deviation of BOTH parts
        # after the choice, so early groups spread instead of piling up.
        score_after_val = _class_balance_score(
            [v + a for v, a in zip(val_counts, counts)],
            [2 * t for t in val_target_counts], 2,
        ) + _class_balance_score(
            train_counts, [2 * t for t in train_target_counts], 2,
        )
        score_after_train = _class_balance_score(
            [t + a for t, a in zip(train_counts, counts)],
            [2 * t for t in train_target_counts], 2,
        ) + _class_balance_score(
            val_counts, [2 * t for t in val_target_counts], 2,
        )
        val_score = score_after_val if can_val else float("inf")
        train_score = score_after_train if not must_val else float("inf")
        if val_score <= train_score:
            val_groups.append(name)
            for index, add in enumerate(counts):
                val_counts[index] += add
        else:
            train_groups.append(name)
            for index, add in enumerate(counts):
                train_counts[index] += add
    if not val_groups or not train_groups:
        raise ValueError(
            "inner split produced an empty part; cannot form "
            "inner_train/inner_val for this fold"
        )
    return sorted(train_groups), sorted(val_groups)


def build_folds(
    group_stats: dict[str, dict],
    *,
    n_folds: int,
    seed: int,
    fraction: float,
    inner_seed_base: int,
    require_outer_eval_all_classes: bool = True,
    require_inner_train_all_classes: bool = True,
    require_inner_val_all_classes: bool = False,
    class_ids: Sequence[int] = CLASS_IDS,
) -> dict:
    """Assign folds, split inner parts, and audit the result.

    Returns ``{"status": OK | BLOCKED, "folds": [...], "conflicts": [...]}``.
    A blocked assignment is never auto-repaired by splitting groups or by
    falling back to random frame-level assignment: the concrete conflicts
    are reported and the fitting admission refuses.
    """
    assignment, fold_counts = assign_outer_folds(
        group_stats, n_folds=n_folds, seed=seed, class_ids=class_ids
    )
    conflicts: list[dict] = []
    folds: list[dict] = []
    for fold in range(n_folds):
        outer_eval = sorted(name for name, f in assignment.items() if f == fold)
        outer_train = sorted(name for name, f in assignment.items() if f != fold)
        if not outer_eval or not outer_train:
            conflicts.append(
                {
                    "fold": fold,
                    "reason": f"outer_eval has {len(outer_eval)} groups, "
                    f"outer_train {len(outer_train)}; a fold would be empty",
                }
            )
            continue
        try:
            inner_train, inner_val = split_inner(
                outer_train,
                group_stats,
                fraction=fraction,
                seed=inner_seed_base + fold,
                class_ids=class_ids,
            )
        except ValueError as error:
            conflicts.append({"fold": fold, "reason": str(error)})
            continue
        folds.append(
            {
                "fold_id": f"fold{fold}",
                "outer_eval_groups": outer_eval,
                "inner_train_groups": inner_train,
                "inner_val_groups": inner_val,
            }
        )
    if len(folds) != n_folds:
        conflicts.append(
            {
                "fold": None,
                "reason": f"only {len(folds)} of {n_folds} folds could be formed",
            }
        )

    # Class coverage of every level.
    for fold in folds:
        for part, required in (
            ("outer_eval_groups", require_outer_eval_all_classes),
            ("inner_train_groups", require_inner_train_all_classes),
            ("inner_val_groups", require_inner_val_all_classes),
        ):
            counts = {
                str(c): sum(
                    group_stats[name]["gt_instances_by_class"].get(str(c), 0)
                    for name in fold[part]
                )
                for c in class_ids
            }
            fold[f"{part.removesuffix('_groups')}_class_counts"] = counts
            missing = [c for c, count in counts.items() if count == 0]
            if required and missing:
                conflicts.append(
                    {
                        "fold": fold["fold_id"],
                        "reason": f"{part} lacks classes {missing}; a fold "
                        "level must cover every class for this protocol",
                    }
                )

    status = FOLD_STATUS_OK if not conflicts else FOLD_STATUS_BLOCKED
    return {"status": status, "folds": folds, "conflicts": conflicts}


def audit_fold_assignment(folds: dict, group_stats: dict[str, dict]) -> dict:
    """Re-check the hard isolation properties of a built assignment.

    - the outer_eval levels partition the dev pool: every dev group is the
      outer_eval group of exactly one fold (no crossing, none lost);
    - within every fold, inner_train and inner_val are disjoint and
      together hold exactly that fold's outer-train groups (its own
      outer_eval groups excluded), so no continuous sequence is split;
    - no group appears twice inside one fold level.
    """
    problems: list[str] = []
    outer_eval_owner: dict[str, str] = {}
    for fold in folds["folds"]:
        for name in fold["outer_eval_groups"]:
            if name in outer_eval_owner:
                problems.append(
                    f"group {name!r} is the outer_eval group of both "
                    f"{outer_eval_owner[name]} and {fold['fold_id']}"
                )
            outer_eval_owner[name] = fold["fold_id"]
    if set(outer_eval_owner) != set(group_stats):
        problems.append(
            "outer_eval groups do not cover the dev pool exactly: "
            f"missing={sorted(set(group_stats) - set(outer_eval_owner))}, "
            f"unknown={sorted(set(outer_eval_owner) - set(group_stats))}"
        )
    for fold in folds["folds"]:
        own_eval = set(fold["outer_eval_groups"])
        inner_train = set(fold["inner_train_groups"])
        inner_val = set(fold["inner_val_groups"])
        expected_outer_train = set(group_stats) - own_eval
        if inner_train & inner_val:
            problems.append(
                f"fold {fold['fold_id']}: groups appear in both inner parts: "
                f"{sorted(inner_train & inner_val)}"
            )
        if inner_train | inner_val != expected_outer_train:
            problems.append(
                f"fold {fold['fold_id']}: inner parts do not partition the "
                f"outer-train groups (missing="
                f"{sorted(expected_outer_train - (inner_train | inner_val))}, "
                f"unexpected={sorted((inner_train | inner_val) - expected_outer_train)})"
            )
    return {
        "group_crossing_violations": problems,
        "groups_checked": len(group_stats),
        "folds_checked": len(folds["folds"]),
        "passed": not problems,
    }


# ---------------------------------------------------------------------------
# Protocol assembly


def protocol_digest(protocol: dict) -> str:
    """Stable sha256 over the protocol content (volatile fields excluded)."""
    payload = {
        key: value
        for key, value in protocol.items()
        if key not in ("generated_at", "protocol_sha256")
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_protocol(
    *,
    config: dict,
    config_path: str | Path,
    admission: dict,
    baselines: Sequence[dict],
    inventory: dict,
    folds: dict,
    project_root: Path | None = None,
) -> dict:
    """Assemble the protocol report (the dict written as protocol.json).

    The report binds the data fingerprint, the admission result, the
    baseline identities, the inventory and the fold assignment. It asserts
    the hard isolation invariants one last time before writing.
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    group_stats = group_statistics(inventory)
    audit = audit_fold_assignment(folds, group_stats)
    if not audit["passed"] and folds["status"] == FOLD_STATUS_OK:
        raise ValueError(
            "fold assignment failed its own audit: "
            f"{audit['group_crossing_violations']}"
        )
    data_config = config["data"]
    config_resolved = Path(config_path)
    if not config_resolved.is_absolute():
        config_resolved = root / config_resolved
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_by": "scripts/prepare_existing_data_protocol.py",
        "config_path": str(config_path),
        "config_sha256": (
            sha256_of_file(config_resolved)
            if config_resolved.is_file() else ""
        ),
        "config": config,
        "data": {
            "root": data_config["root"],
            "fingerprint": admission["audit_fingerprint"],
            "parent_v4_fingerprint": admission.get("v4_parent_fingerprint", ""),
            "dev_pool_splits": list(data_config["dev_pool_splits"]),
            "excluded_splits": list(data_config["excluded_splits"]),
            "class_names": list(data_config["class_names"]),
        },
        "admission": {
            "data_root": admission["data_root"],
            "audit_fingerprint": admission["audit_fingerprint"],
            "v4_parent_fingerprint": admission.get("v4_parent_fingerprint", ""),
            "verified_files": admission.get("verified_files"),
            "mask_schema_version": admission.get("mask_schema_version"),
            "mask_resize_version": admission.get("mask_resize_version"),
            "class_names": admission.get("class_names"),
        },
        "baselines": list(baselines),
        "inventory": inventory,
        "group_statistics": {
            name: {
                "splits": entry["splits"],
                "blocks": entry["blocks"],
                "frames": entry["frames"],
                "samples": entry["samples"],
                "gt_instances": entry["gt_instances"],
                "gt_instances_by_class": entry["gt_instances_by_class"],
                "empty_gt_frames": entry["empty_gt_frames"],
                "sequence_note": entry["sequence_note"],
            }
            for name, entry in sorted(group_stats.items())
        },
        "folds": folds,
        "fold_audit": audit,
        "interpretation_boundaries": {
            "training_level_group_isolation_only": (
                "the fold protocol proves group isolation for training on "
                "the existing pool; the original val part already took part "
                "in project development, so no fold is a new blind test"
            ),
            "no_independent_capture_generalization": (
                "no claim of independent-capture generalization is made by "
                "this protocol or by anything evaluated under it"
            ),
            "historical_weights_not_oof": (
                "diagnostics that use weights trained on the old train/val "
                "are not OOF evidence under this protocol"
            ),
            "physical_object_identity": (
                "no source records physical-object identity; it is recorded "
                "as unknown and never guessed from file names; cross-group "
                "object relations are unconfirmed and not treated as leakage"
            ),
        },
    }
    protocol["protocol_sha256"] = protocol_digest(protocol)
    return protocol


# ---------------------------------------------------------------------------
# Shared execution budget (detection conditions, timing repeats, feature
# extraction batches). One shared start, one deadline, checked at safe
# boundaries — never a fresh budget per step.


class BudgetExceeded(RuntimeError):
    """Raised at a safe boundary when the shared budget is exhausted."""


class BudgetWatch:
    """Monotonic-clock budget shared by every stage of one entry point.

    ``clock`` is injectable for tests (a fake monotonic clock); production
    uses ``time.monotonic``. ``check`` raises :class:`BudgetExceeded` once
    the shared deadline has passed — the caller stops at the next safe
    boundary (condition / frame / timing repeat / extraction batch),
    keeps the finished evidence and reports at least PARTIAL.
    """

    def __init__(self, minutes: float, *, clock=None):
        if clock is None:
            import time

            clock = time.monotonic
        minutes = float(minutes)
        if not minutes > 0:
            raise ValueError(
                f"budget minutes must be positive, got {minutes!r}"
            )
        self._clock = clock
        self.shared_start = clock()
        self._deadline = self.shared_start + minutes * 60.0
        self.minutes = minutes

    def remaining_s(self) -> float:
        return self._deadline - self._clock()

    def check(self, at: str = "") -> None:
        if self.remaining_s() <= 0.0:
            raise BudgetExceeded(
                f"shared budget of {self.minutes} minutes exhausted "
                f"(at {at or 'unspecified boundary'}); stopping at this "
                "safe boundary with the finished evidence kept"
            )


__all__ = [
    "BASELINE_KEYS",
    "CLASS_IDS",
    "BudgetExceeded",
    "BudgetWatch",
    "FOLD_STATUS_BLOCKED",
    "FOLD_STATUS_OK",
    "PROTOCOL_VERSION",
    "assign_outer_folds",
    "audit_fold_assignment",
    "build_dev_inventory",
    "build_folds",
    "build_protocol",
    "group_statistics",
    "load_experiment_config",
    "protocol_digest",
    "read_label_class_ids",
    "read_manifest_rows",
    "read_split_rows",
    "sha256_of_file",
    "split_inner",
    "verify_baseline_identities",
]
