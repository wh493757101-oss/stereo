"""Synthetic-tree tests for the existing-data protocol (no real data).

Every fixture here is fully synthetic: tiny label files, a tiny split
assignment and, for the CLI-level tests, a complete source -> V4 -> V5
chain built by the production generators inside ``.test_artifacts``. No
real project dataset, weight or report is read by these tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from core.existing_data_protocol import (
    CLASS_IDS,
    FOLD_STATUS_BLOCKED,
    FOLD_STATUS_OK,
    assign_outer_folds,
    audit_fold_assignment,
    build_dev_inventory,
    build_folds,
    build_protocol,
    group_statistics,
    load_experiment_config,
    protocol_digest,
    read_label_class_ids,
    read_split_rows,
    split_inner,
    verify_baseline_identities,
)

from .conftest import (
    build_synthetic_source,
    build_synthetic_v4,
    build_synthetic_v5,
)


# ---------------------------------------------------------------------------
# Minimal config fixtures


def _write_config(path: Path, overrides: dict | None = None) -> Path:
    config = {
        "experiment": "material_existing_data_v1",
        "data": {
            "root": "datasets/synthetic_v5",
            "fingerprint": "a" * 64,
            "parent_v4_fingerprint": "b" * 64,
            "dev_pool_splits": ["train", "val"],
            "excluded_splits": ["test"],
            "split_assignment": "split_assignment.csv",
            "label_root": "labels",
            "class_names": ["metal_submarine", "plastic_submarine",
                            "plastic_fish", "real_fish"],
        },
        "baselines": {
            "model_a": {"path": "weights/model_a.pt",
                        "sha256": "1" * 64},
            "e1_candidate": {"path": "weights/e1.pt", "sha256": "2" * 64},
            "e4_candidate": {"path": "weights/e4.pt", "sha256": "3" * 64},
            "independent_gray_fallback": {"path": "weights/fb.pt",
                                          "sha256": "4" * 64},
            "official_mit_b0": {"path": "weights/mit_b0.pth",
                                "sha256": "5" * 64},
            "historical_d0_evidence": {
                "frames_jsonl": "evidence/frames.jsonl",
            },
        },
        "protocol": {"n_outer_folds": 3, "seed": 2026},
        "detection": {"match_iou_threshold": 0.50},
        "features": {"feature_definition_version": "existing_data_probe_v1"},
    }
    if overrides:
        for section, values in overrides.items():
            config.setdefault(section, {}).update(values)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_weight(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class TestLoadExperimentConfig:
    def test_missing_file_refused(self, scratch_dir):
        with pytest.raises(FileNotFoundError):
            load_experiment_config(scratch_dir / "nope.yaml")

    def test_missing_required_field_refused(self, scratch_dir):
        path = _write_config(scratch_dir / "config.yaml")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        del config["baselines"]["e4_candidate"]["sha256"]
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="e4_candidate.sha256"):
            load_experiment_config(path)

    def test_valid_config_loads(self, scratch_dir):
        config = load_experiment_config(_write_config(scratch_dir / "c.yaml"))
        assert config["experiment"] == "material_existing_data_v1"
        assert config["protocol"]["seed"] == 2026


class TestBaselineIdentities:
    def test_all_identities_verified_and_recorded(self, scratch_dir):
        import hashlib

        weights = {
            "weights/model_a.pt": b"model-a-bytes",
            "weights/e1.pt": b"e1-bytes",
            "weights/e4.pt": b"e4-bytes",
            "weights/fb.pt": b"fallback-bytes",
            "weights/mit_b0.pth": b"mit-bytes",
            "evidence/frames.jsonl": b'{"frame": 1}\n',
        }
        digests = {}
        for relative, payload in weights.items():
            path = _write_weight(scratch_dir, relative, payload)
            digests[relative] = hashlib.sha256(payload).hexdigest()
        config = load_experiment_config(_write_config(
            scratch_dir / "config.yaml",
            {
                "baselines": {
                    "model_a": {"path": "weights/model_a.pt",
                                "sha256": digests["weights/model_a.pt"]},
                    "e1_candidate": {"path": "weights/e1.pt",
                                     "sha256": digests["weights/e1.pt"]},
                    "e4_candidate": {"path": "weights/e4.pt",
                                     "sha256": digests["weights/e4.pt"]},
                    "independent_gray_fallback": {
                        "path": "weights/fb.pt",
                        "sha256": digests["weights/fb.pt"]},
                    "official_mit_b0": {
                        "path": "weights/mit_b0.pth",
                        "sha256": digests["weights/mit_b0.pth"]},
                },
            },
        ))
        records = verify_baseline_identities(config, scratch_dir)
        by_key = {record["key"]: record for record in records}
        assert set(by_key) == {
            "model_a", "e1_candidate", "e4_candidate",
            "independent_gray_fallback", "official_mit_b0",
            "historical_d0_frames_jsonl",
        }
        assert by_key["model_a"]["sha256"] == digests["weights/model_a.pt"]

    def test_missing_baseline_file_refused(self, scratch_dir):
        config = load_experiment_config(_write_config(scratch_dir / "c.yaml"))
        with pytest.raises(FileNotFoundError, match="model_a"):
            verify_baseline_identities(config, scratch_dir)

    def test_sha_mismatch_refused(self, scratch_dir):
        _write_weight(scratch_dir, "weights/model_a.pt", b"changed")
        for relative, payload in {
            "weights/e1.pt": b"e1", "weights/e4.pt": b"e4",
            "weights/fb.pt": b"fb", "weights/mit_b0.pth": b"mit",
            "evidence/frames.jsonl": b"x",
        }.items():
            _write_weight(scratch_dir, relative, payload)
        config = load_experiment_config(_write_config(scratch_dir / "c.yaml"))
        with pytest.raises(ValueError, match="does not match"):
            verify_baseline_identities(config, scratch_dir)


# ---------------------------------------------------------------------------
# Inventory


def _label(root: Path, split: str, stem: str, class_ids: list[int]) -> str:
    path = root / "labels" / split / f"{stem}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{cid} 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n" for cid in class_ids
        ),
        encoding="utf-8",
    )
    return path


def _manifest_row(sample: str, split: str, group: str, frame: str,
                  obj: int, class_id: int) -> dict:
    class_names = ("metal_submarine", "plastic_submarine",
                   "plastic_fish", "real_fish")
    return {
        "sample_name": sample,
        "split": split,
        "group_name": group,
        "class_id": str(class_id),
        "class_name": class_names[class_id],
        "source_frame": frame,
        "object_index": str(obj),
        "npz_path": f"{split}/{class_names[class_id]}/{sample}.npz",
    }


def _touch_npz(root: Path, row: dict) -> None:
    path = root / row["npz_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"npz")


def _basic_split_rows() -> list[dict]:
    rows = []
    for index in range(3):
        rows.append({"frame_id": f"train_{index:03d}", "split": "train",
                     "group": "train/g_a", "block_id": "0",
                     "left_path": f"left/train/{index}.png"})
    rows.append({"frame_id": "train_100", "split": "train",
                 "group": "train/g_empty", "block_id": "0",
                 "left_path": "left/train/100.png"})
    for index in range(2):
        rows.append({"frame_id": f"val_{index:03d}", "split": "val",
                     "group": "val/g_b", "block_id": "0",
                     "left_path": f"left/val/{index}.png"})
    return rows


class TestReadSplitRows:
    def test_selects_only_requested_splits(self, scratch_dir):
        path = scratch_dir / "split_assignment.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "output_stem", "split", "group_name", "block_id",
                "left_path"])
            writer.writeheader()
            writer.writerows([
                {"output_stem": "a", "split": "train", "group_name": "g1",
                 "block_id": "0", "left_path": "a.png"},
                {"output_stem": "b", "split": "test", "group_name": "g2",
                 "block_id": "0", "left_path": "b.png"},
                {"output_stem": "c", "split": "val", "group_name": "g3",
                 "block_id": "0", "left_path": "c.png"},
            ])
        rows = read_split_rows(path, ["train", "val"])
        assert [r["frame_id"] for r in rows] == ["a", "c"]

    def test_duplicate_stem_refused(self, scratch_dir):
        path = scratch_dir / "split_assignment.csv"
        path.write_text(
            "output_stem,split,group_name,block_id\n"
            "a,train,g1,0\na,val,g2,0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="twice"):
            read_split_rows(path, ["train", "val"])


class TestReadLabelClassIds:
    def test_missing_file_raises(self, scratch_dir):
        with pytest.raises(FileNotFoundError):
            read_label_class_ids(scratch_dir / "nope.txt")

    def test_malformed_line_refused(self, scratch_dir):
        path = scratch_dir / "bad.txt"
        path.write_text("0 0.1 0.1\nfish 0.2 0.2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not an integer"):
            read_label_class_ids(path)

    def test_unknown_class_refused(self, scratch_dir):
        path = scratch_dir / "bad2.txt"
        path.write_text("7 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="unknown class"):
            read_label_class_ids(path)

    def test_class_ids_parsed(self, scratch_dir):
        path = _label(scratch_dir, "train", "f000", [0, 3, 3])
        assert read_label_class_ids(path) == [0, 3, 3]


class TestBuildDevInventory:
    def test_counts_and_correspondence(self, scratch_dir):
        _label(scratch_dir, "train", "train_000", [0, 1])
        _label(scratch_dir, "train", "train_001", [2])
        _label(scratch_dir, "val", "val_000", [3, 3, 3])
        # train_100 deliberately has no label file: an empty-GT frame.
        manifest = [
            _manifest_row("train_000_obj000", "train", "train/g_a",
                          "train_000", 0, 0),
            _manifest_row("train_000_obj001", "train", "train/g_a",
                          "train_000", 1, 1),
            _manifest_row("train_001_obj000", "train", "train/g_a",
                          "train_001", 0, 2),
            _manifest_row("val_000_obj000", "val", "val/g_b",
                          "val_000", 0, 3),
        ]
        for row in manifest:
            _touch_npz(scratch_dir / "v5", row)
        inventory = build_dev_inventory(
            split_rows=_basic_split_rows(),
            manifest_rows=manifest,
            label_root=scratch_dir / "labels",
            data_root=scratch_dir / "v5",
            project_root=scratch_dir,
        )
        summary = inventory["summary"]
        assert summary["frames"] == 6
        assert summary["samples"] == 4
        assert summary["gt_instances"] == 6
        assert summary["gt_instances_by_class"] == {
            "0": 1, "1": 1, "2": 1, "3": 3,
        }
        assert summary["empty_gt_frames"] == ["train_002", "train_100",
                                              "val_001"]
        assert summary["gt_without_npz_count"] == 2  # val_000 obj1, obj2
        assert summary["npz_without_gt_count"] == 0
        assert summary["anomaly_count"] == 0
        assert all(s["physical_object_id"] == "unknown"
                   for s in inventory["samples"])

    def test_gt_without_npz_kept_and_listed(self, scratch_dir):
        _label(scratch_dir, "train", "train_000", [0, 0])
        manifest = [_manifest_row("train_000_obj000", "train", "train/g_a",
                                  "train_000", 0, 0)]
        _touch_npz(scratch_dir / "v5", manifest[0])
        inventory = build_dev_inventory(
            split_rows=_basic_split_rows(),
            manifest_rows=manifest,
            label_root=scratch_dir / "labels",
            data_root=scratch_dir / "v5",
            project_root=scratch_dir,
        )
        missing = inventory["gt_without_npz"]
        assert len(missing) == 1
        assert missing[0] == {
            "frame_id": "train_000", "object_index": 1,
            "class_id": 0, "class_name": "metal_submarine",
        }

    def test_npz_without_gt_source_flagged(self, scratch_dir):
        _label(scratch_dir, "train", "train_000", [0])
        manifest = [
            _manifest_row("train_000_obj000", "train", "train/g_a",
                          "train_000", 0, 0),
            _manifest_row("train_000_obj007", "train", "train/g_a",
                          "train_000", 7, 1),
        ]
        for row in manifest:
            _touch_npz(scratch_dir / "v5", row)
        inventory = build_dev_inventory(
            split_rows=_basic_split_rows(),
            manifest_rows=manifest,
            label_root=scratch_dir / "labels",
            data_root=scratch_dir / "v5",
            project_root=scratch_dir,
        )
        assert len(inventory["npz_without_gt"]) == 1
        assert inventory["npz_without_gt"][0]["object_index"] == 7

    def test_class_mismatch_flagged(self, scratch_dir):
        _label(scratch_dir, "train", "train_000", [0])
        manifest = [_manifest_row("train_000_obj000", "train", "train/g_a",
                                  "train_000", 0, 1)]
        _touch_npz(scratch_dir / "v5", manifest[0])
        inventory = build_dev_inventory(
            split_rows=_basic_split_rows(),
            manifest_rows=manifest,
            label_root=scratch_dir / "labels",
            data_root=scratch_dir / "v5",
            project_root=scratch_dir,
        )
        kinds = [a["kind"] for a in inventory["anomalies"]]
        assert "class_mismatch" in kinds

    def test_missing_npz_file_anomaly(self, scratch_dir):
        _label(scratch_dir, "train", "train_000", [0])
        manifest = [_manifest_row("train_000_obj000", "train", "train/g_a",
                                  "train_000", 0, 0)]
        inventory = build_dev_inventory(
            split_rows=_basic_split_rows(),
            manifest_rows=manifest,
            label_root=scratch_dir / "labels",
            data_root=scratch_dir / "v5",
            project_root=scratch_dir,
        )
        assert [a["kind"] for a in inventory["anomalies"]] == ["npz_missing"]
        assert inventory["summary"]["samples"] == 0


class TestGroupStatistics:
    def test_refuses_group_spanning_splits(self):
        inventory = {
            "frames": [
                {"frame_id": "a", "split": "train", "group": "g", "block_id": "0",
                 "gt_instances": 1, "gt_class_counts": {"0": 1},
                 "gt_class_ids": [0]},
                {"frame_id": "b", "split": "val", "group": "g", "block_id": "0",
                 "gt_instances": 1, "gt_class_counts": {"0": 1},
                 "gt_class_ids": [0]},
            ],
            "samples": [],
        }
        with pytest.raises(ValueError, match="spans original splits"):
            group_statistics(inventory)

    def test_aggregates_frames_and_samples(self):
        inventory = {
            "frames": [
                {"frame_id": "a", "split": "train", "group": "g1",
                 "block_id": "0", "gt_instances": 2,
                 "gt_class_counts": {"0": 1, "3": 1}, "gt_class_ids": [0, 3]},
                {"frame_id": "b", "split": "train", "group": "g2",
                 "block_id": "0", "gt_instances": 0,
                 "gt_class_counts": {}, "gt_class_ids": []},
                {"frame_id": "c", "split": "val", "group": "g3",
                 "block_id": "0", "gt_instances": 1,
                 "gt_class_counts": {"2": 1}, "gt_class_ids": [2]},
            ],
            "samples": [
                {"sample_name": "a_obj000", "group": "g1", "class_id": 0},
                {"sample_name": "c_obj000", "group": "g3", "class_id": 2},
            ],
        }
        stats = group_statistics(inventory)
        assert stats["g1"]["gt_instances_by_class"] == {
            "0": 1, "1": 0, "2": 0, "3": 1,
        }
        assert stats["g1"]["samples"] == 1
        assert stats["g2"]["empty_gt_frames"] == 1
        assert stats["g3"]["splits"] == ["val"]


def _group_stats(class_layout: dict[str, dict[int, int]]) -> dict[str, dict]:
    stats = {}
    for name, counts in class_layout.items():
        stats[name] = {
            "group": name,
            "splits": ["train"],
            "blocks": ["0"],
            "frames": 1,
            "empty_gt_frames": 0,
            "gt_instances": sum(counts.values()),
            "gt_instances_by_class": {
                str(c): counts.get(c, 0) for c in CLASS_IDS
            },
            "samples": sum(counts.values()),
        }
    return stats


class TestAssignOuterFolds:
    def test_deterministic_and_total(self):
        layout = {f"g{i}": {i % 4: 10, (i + 1) % 4: 5} for i in range(12)}
        stats = _group_stats(layout)
        first, totals = assign_outer_folds(stats, n_folds=3, seed=2026)
        second, _ = assign_outer_folds(stats, n_folds=3, seed=2026)
        assert first == second
        assert sorted(first) == sorted(stats)
        assert set(first.values()) == {0, 1, 2}

    def test_classes_covered_on_feasible_layout(self):
        layout = {f"g{i}": {c: 10 + i for c in CLASS_IDS} for i in range(9)}
        stats = _group_stats(layout)
        assignment, totals = assign_outer_folds(stats, n_folds=3, seed=2026)
        for fold in range(3):
            for class_index in range(4):
                assert totals[fold][class_index] > 0


class TestSplitInner:
    def test_partition_and_determinism(self):
        layout = {f"g{i}": {c: 10 + i for c in CLASS_IDS} for i in range(8)}
        stats = _group_stats(layout)
        train_a, val_a = split_inner(
            sorted(stats), stats, fraction=0.25, seed=2026)
        train_b, val_b = split_inner(
            sorted(stats), stats, fraction=0.25, seed=2026)
        assert (train_a, val_a) == (train_b, val_b)
        assert sorted(train_a + val_a) == sorted(stats)
        assert not set(train_a) & set(val_a)
        assert len(val_a) == 2  # ceil(8 * 0.25)

    def test_whole_groups_only(self):
        layout = {f"g{i}": {c: 10 for c in CLASS_IDS} for i in range(6)}
        stats = _group_stats(layout)
        train, val = split_inner(sorted(stats), stats, fraction=0.34, seed=7)
        assert set(train) | set(val) == set(stats)
        assert not set(train) & set(val)

    def test_single_group_refused(self):
        stats = _group_stats({"g0": {0: 5}})
        with pytest.raises(ValueError, match="at least 2 groups"):
            split_inner(["g0"], stats, fraction=0.25, seed=2026)


class TestBuildFolds:
    def test_ok_on_feasible_layout(self):
        layout = {f"g{i}": {c: 10 + i for c in CLASS_IDS} for i in range(9)}
        folds = build_folds(
            _group_stats(layout), n_folds=3, seed=2026, fraction=0.25,
            inner_seed_base=2026,
        )
        assert folds["status"] == FOLD_STATUS_OK
        assert len(folds["folds"]) == 3
        assert folds["conflicts"] == []

    def test_blocked_when_class_cannot_cover_folds(self):
        # class 3 exists in only 2 groups: 3 outer folds can never all
        # evaluate it -> BLOCKED with concrete conflicts, no auto-repair.
        layout = {
            "g0": {0: 10, 1: 10, 2: 10, 3: 5},
            "g1": {0: 10, 1: 10, 2: 10, 3: 5},
            "g2": {0: 10, 1: 10, 2: 10},
            "g3": {0: 10, 1: 10, 2: 10},
        }
        folds = build_folds(
            _group_stats(layout), n_folds=3, seed=2026, fraction=0.25,
            inner_seed_base=2026,
        )
        assert folds["status"] == FOLD_STATUS_BLOCKED
        assert folds["conflicts"]
        assert any("outer_eval_groups lacks classes" in c["reason"]
                   for c in folds["conflicts"])

    def test_inner_train_missing_class_blocked(self):
        layout = {
            "g0": {0: 10, 1: 10, 2: 10, 3: 10},
            "g1": {0: 10, 1: 10, 2: 10},
            "g2": {0: 10, 1: 10, 2: 10},
        }
        folds = build_folds(
            _group_stats(layout), n_folds=2, seed=2026, fraction=0.5,
            inner_seed_base=2026,
        )
        # one fold gets g0 as outer_eval, the other holds g1+g2 with no
        # class 3 anywhere outside -> its inner_train lacks class 3.
        assert folds["status"] == FOLD_STATUS_BLOCKED


class TestAuditFoldAssignment:
    def test_passes_on_built_folds(self):
        layout = {f"g{i}": {c: 10 + i for c in CLASS_IDS} for i in range(9)}
        stats = _group_stats(layout)
        folds = build_folds(stats, n_folds=3, seed=2026, fraction=0.25,
                            inner_seed_base=2026)
        audit = audit_fold_assignment(folds, stats)
        assert audit["passed"], audit["group_crossing_violations"]

    def test_detects_group_crossing(self):
        layout = {f"g{i}": {c: 10 for c in CLASS_IDS} for i in range(6)}
        stats = _group_stats(layout)
        folds = build_folds(stats, n_folds=2, seed=2026, fraction=0.5,
                            inner_seed_base=2026)
        folds["folds"][0]["outer_eval_groups"].append(
            folds["folds"][1]["outer_eval_groups"][0])
        audit = audit_fold_assignment(folds, stats)
        assert not audit["passed"]
        assert any("outer_eval group of both" in p
                   for p in audit["group_crossing_violations"])


# ---------------------------------------------------------------------------
# Protocol assembly


def _protocol_weights(scratch_dir: Path) -> dict[str, str]:
    """The dummy baseline files the config identity check requires.

    Returns the real sha256 digests so a config can be written that
    actually verifies.
    """
    import hashlib

    digests = {}
    for relative, payload in {
        "weights/model_a.pt": b"model-a",
        "weights/e1.pt": b"e1",
        "weights/e4.pt": b"e4",
        "weights/fb.pt": b"fallback",
        "weights/mit_b0.pth": b"mit",
        "evidence/frames.jsonl": b'{"frame": 1}\n',
    }.items():
        _write_weight(scratch_dir, relative, payload)
        digests[relative] = hashlib.sha256(payload).hexdigest()
    return digests


def _config_with_verified_identities(scratch_dir: Path, name: str = "c.yaml") -> Path:
    digests = _protocol_weights(scratch_dir)
    path = _write_config(scratch_dir / name)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["baselines"]["model_a"]["sha256"] = digests["weights/model_a.pt"]
    config["baselines"]["e1_candidate"]["sha256"] = digests["weights/e1.pt"]
    config["baselines"]["e4_candidate"]["sha256"] = digests["weights/e4.pt"]
    config["baselines"]["independent_gray_fallback"]["sha256"] = digests["weights/fb.pt"]
    config["baselines"]["official_mit_b0"]["sha256"] = digests["weights/mit_b0.pth"]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _protocol_inputs(scratch_dir: Path):
    layout = {f"g{i}": {c: 10 + i for c in CLASS_IDS} for i in range(9)}
    stats = _group_stats(layout)
    inventory = {
        "frames": [
            {"frame_id": f"{name}_000", "split": "train", "group": name,
             "block_id": "0", "gt_instances": 4,
             "gt_class_ids": [0, 1, 2, 3],
             "gt_class_counts": {"0": 1, "1": 1, "2": 1, "3": 1}}
            for name in stats
        ],
        "samples": [
            {"sample_name": f"{name}_000_obj{i:03d}", "group": name,
             "split": "train", "source_frame": f"{name}_000",
             "object_index": i, "class_id": i,
             "class_name": ["metal_submarine", "plastic_submarine",
                            "plastic_fish", "real_fish"][i],
             "physical_object_id": "unknown"}
            for name in stats for i in range(4)
        ],
        "anomalies": [],
        "gt_without_npz": [],
        "npz_without_gt": [],
        "summary": {"frames": len(stats), "samples": 4 * len(stats),
                    "empty_gt_frames": [], "gt_instances": 4 * len(stats),
                    "gt_instances_by_class": {"0": 90, "1": 90, "2": 90,
                                              "3": 90},
                    "gt_without_npz_count": 0, "npz_without_gt_count": 0,
                    "anomaly_count": 0},
    }
    return stats, inventory


class TestBuildProtocol:
    def test_assembles_and_digest_is_stable(self, scratch_dir):
        config = load_experiment_config(
            _config_with_verified_identities(scratch_dir))
        baselines = verify_baseline_identities(config, scratch_dir)
        stats, inventory = _protocol_inputs(scratch_dir)
        folds = build_folds(stats, n_folds=3, seed=2026, fraction=0.25,
                            inner_seed_base=2026)
        protocol = build_protocol(
            config=config, config_path="c.yaml",
            admission={"data_root": "v5", "audit_fingerprint": "a" * 64,
                       "v4_parent_fingerprint": "b" * 64},
            baselines=baselines, inventory=inventory, folds=folds,
            project_root=scratch_dir,
        )
        assert protocol["protocol_version"] == "material_existing_data_v1"
        assert protocol["folds"]["status"] == FOLD_STATUS_OK
        assert protocol["protocol_sha256"] == protocol_digest(protocol)
        # JSON must round-trip without NaN/Inf
        text = json.dumps(protocol, allow_nan=False, sort_keys=True)
        assert json.loads(text)["protocol_sha256"] == protocol["protocol_sha256"]
        # volatile fields are excluded from the digest
        changed = dict(protocol)
        changed["generated_at"] = "2030-01-01T00:00:00Z"
        assert protocol_digest(changed) == protocol["protocol_sha256"]

    def test_audit_failure_on_ok_folds_refused(self, scratch_dir):
        config = load_experiment_config(
            _config_with_verified_identities(scratch_dir))
        baselines = verify_baseline_identities(config, scratch_dir)
        stats, inventory = _protocol_inputs(scratch_dir)
        folds = build_folds(stats, n_folds=3, seed=2026, fraction=0.25,
                            inner_seed_base=2026)
        folds["folds"][0]["inner_train_groups"] = (
            folds["folds"][0]["inner_train_groups"]
            + folds["folds"][1]["inner_val_groups"]
        )
        folds["folds"][1]["inner_val_groups"] = []
        with pytest.raises(ValueError, match="audit"):
            build_protocol(
                config=config, config_path="c.yaml",
                admission={"data_root": "v5",
                           "audit_fingerprint": "a" * 64},
                baselines=baselines, inventory=inventory, folds=folds,
                project_root=scratch_dir,
            )


# ---------------------------------------------------------------------------
# CLI end-to-end over a full synthetic V5 chain


@pytest.fixture(scope="module")
def protocol_chain(module_scratch):
    """source -> V4 -> V5 with a matching split_assignment.csv.

    Every synthetic group contains all four classes, so any whole-group
    fold assignment is valid; the CLI-level tests exercise plumbing, not
    the balancing objective (covered by the unit tests above).
    """
    workspace = module_scratch.resolve()
    source = build_synthetic_source(
        module_scratch / "source", counts=(6, 3, 2), size=64,
        objects_per_frame=4,
    )
    v4 = build_synthetic_v4(source, module_scratch / "v4", workspace)
    v5 = build_synthetic_v5(source, v4, module_scratch / "v5", workspace)

    from core.fusion_dataset import file_digest

    project_root = Path(__file__).resolve().parents[2]
    weights = {}
    for name, payload in (
        ("model_a.pt", b"model-a"),
        ("e1.pt", b"e1"),
        ("e4.pt", b"e4"),
        ("fb.pt", b"fallback"),
        ("mit_b0.pth", b"mit"),
    ):
        path = module_scratch / "weights" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        weights[name] = (path, file_digest(path))

    split_assignment = source / "split_assignment.csv"
    with split_assignment.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "output_stem", "split", "group_name", "block_id",
            "source_json", "left_path", "right_path"])
        writer.writeheader()
        for split in ("train", "val", "test"):
            for index in range({"train": 6, "val": 3, "test": 2}[split]):
                stem = f"{split}_frame_{index:03d}"
                writer.writerow({
                    "output_stem": stem,
                    "split": split,
                    "group_name": f"{split}/group_{index}",
                    "block_id": "0",
                    "source_json": "",
                    "left_path": (source / "images" / split / f"{stem}.png").as_posix(),
                    "right_path": (source / "images_right" / split / f"{stem}.png").as_posix(),
                })
    return {
        "workspace": workspace,
        "source": source,
        "v4": v4,
        "v5": v5,
        "weights": weights,
        "split_assignment": split_assignment,
        "project_root": project_root,
    }


def _write_chain_config(protocol_chain, path: Path) -> Path:
    """An experiment config bound to the synthetic chain (project-relative)."""
    import hashlib

    from core.fusion_dataset import dataset_fingerprint, file_digest
    from core.material_mask_dataset import mask_dataset_fingerprint

    root = protocol_chain["project_root"]
    v5_rel = protocol_chain["v5"].relative_to(root).as_posix()
    split_rel = protocol_chain["split_assignment"].relative_to(root).as_posix()
    labels_rel = (protocol_chain["source"] / "labels").relative_to(root).as_posix()
    weights = protocol_chain["weights"]

    def wrel(name):
        return weights[name][0].relative_to(root).as_posix()

    config = {
        "experiment": "material_existing_data_v1",
        "data": {
            "root": v5_rel,
            "fingerprint": mask_dataset_fingerprint(protocol_chain["v5"]),
            "parent_v4_fingerprint": dataset_fingerprint(protocol_chain["v4"]),
            "dev_pool_splits": ["train", "val"],
            "excluded_splits": ["test"],
            "split_assignment": split_rel,
            "label_root": labels_rel,
            "class_names": ["metal_submarine", "plastic_submarine",
                            "plastic_fish", "real_fish"],
        },
        "baselines": {
            "model_a": {"path": wrel("model_a.pt"),
                        "sha256": weights["model_a.pt"][1]},
            "e1_candidate": {"path": wrel("e1.pt"),
                             "sha256": weights["e1.pt"][1]},
            "e4_candidate": {"path": wrel("e4.pt"),
                             "sha256": weights["e4.pt"][1]},
            "independent_gray_fallback": {"path": wrel("fb.pt"),
                                          "sha256": weights["fb.pt"][1]},
            "official_mit_b0": {"path": wrel("mit_b0.pth"),
                                "sha256": weights["mit_b0.pth"][1]},
            "historical_d0_evidence": {
                "frames_jsonl": wrel("model_a.pt"),
            },
        },
        "protocol": {
            "n_outer_folds": 3,
            "seed": 2026,
            "assignment_objective": "class_instance_count_greedy",
            "inner_val_fraction": 0.25,
            "inner_seed_base": 2026,
        },
        "detection": {"match_iou_threshold": 0.50},
        "features": {
            "imgsz": 224,
            "preprocessing_version": "v4_band_224_area_linear_v1",
            "gray_input_mode": "gray replicated to [G,G,G]",
            "encoder": "official MiT-B0 (identity from baselines)",
            "pooling": "instance_mask weighted mean on the last stage",
            "pooling_background_note": "test config",
            "feature_definition_version": "existing_data_probe_v1",
        },
        "probe": {
            "variants": {"P0": {"slots": ["G"]},
                         "P1": {"slots": ["G", "C"]},
                         "P2": {"slots": ["G", "C", "S"]}},
            "head": "linear_softmax_4class",
            "seed": 2026,
            "optimizer": "adamw",
            "lr": 0.01,
            "weight_decay": 0.001,
            "max_epochs": 8,
            "device": "cpu",
            "standardization": "per_column_inner_train_only",
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class TestPrepareProtocolCli:
    def test_end_to_end_writes_protocol(self, protocol_chain, scratch_dir):
        from scripts.prepare_existing_data_protocol import main

        config_path = _write_chain_config(
            protocol_chain, scratch_dir / "config.yaml")
        output = scratch_dir / "protocol_out"
        code = main([
            "--config", str(config_path),
            "--output-dir", str(output),
            "--project-root", str(protocol_chain["project_root"]),
        ])
        assert code == 0
        protocol = json.loads(
            (output / "protocol.json").read_text(encoding="utf-8"))
        folds = json.loads((output / "folds.json").read_text(encoding="utf-8"))
        identities = json.loads(
            (output / "baseline_identities.json").read_text(encoding="utf-8"))
        assert protocol["protocol_version"] == "material_existing_data_v1"
        assert protocol["data"]["fingerprint"]
        assert protocol["folds"]["status"] == FOLD_STATUS_OK
        assert len(folds["folds"]) == 3
        assert len(identities) >= 5
        # members reference the original files; nothing was copied or moved
        for fold in folds["folds"]:
            for part in ("outer_eval", "inner_train", "inner_val"):
                members = fold[f"{part}_members"]
                assert members["samples"]
                assert all(s["npz_path"] for s in members["samples"])
        v5_manifest = protocol_chain["v5"] / "dataset_manifest.csv"
        assert v5_manifest.is_file()  # original data untouched
        assert (output / "README.md").is_file()
        assert (output / "group_audit.json").is_file()

    def test_existing_output_dir_refused(self, protocol_chain, scratch_dir):
        from scripts.prepare_existing_data_protocol import main

        config_path = _write_chain_config(
            protocol_chain, scratch_dir / "config.yaml")
        output = scratch_dir / "protocol_out"
        assert main(["--config", str(config_path),
                     "--output-dir", str(output),
                     "--project-root", str(protocol_chain["project_root"])]) == 0
        code = main(["--config", str(config_path),
                     "--output-dir", str(output),
                     "--project-root", str(protocol_chain["project_root"])])
        assert code != 0

    def test_wrong_fingerprint_refused(self, protocol_chain, scratch_dir):
        from scripts.prepare_existing_data_protocol import main

        config_path = _write_chain_config(
            protocol_chain, scratch_dir / "config.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["data"]["fingerprint"] = "f" * 64
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        code = main(["--config", str(config_path),
                     "--output-dir", str(scratch_dir / "out_bad_fp"),
                     "--project-root", str(protocol_chain["project_root"])])
        assert code != 0
        assert not (scratch_dir / "out_bad_fp").exists()

    def test_missing_baseline_refused(self, protocol_chain, scratch_dir):
        from scripts.prepare_existing_data_protocol import main

        config_path = _write_chain_config(
            protocol_chain, scratch_dir / "config.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["baselines"]["e4_candidate"]["sha256"] = "9" * 64
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        code = main(["--config", str(config_path),
                     "--output-dir", str(scratch_dir / "out_bad_sha"),
                     "--project-root", str(protocol_chain["project_root"])])
        assert code != 0
        assert not (scratch_dir / "out_bad_sha").exists()
