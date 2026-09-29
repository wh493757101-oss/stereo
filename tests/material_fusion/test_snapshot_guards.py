"""Stage-snapshot output protection: no overwriting of existing evidence.

Every path a test writes to lives under this session's ``.test_artifacts``
run directory. The real ``runs/train`` tree appears below only as an
*argument* to the validators, which must refuse it whether or not it exists.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from models.material_fusion import build_material_model
from scripts.train_material_fusion import (
    create_snapshot_dir,
    validate_snapshot_target,
    write_stage_metadata,
    write_stage_snapshot,
)

from .conftest import ROOT, material_run_dir, run_production_training


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison_run_dir(run_id: str):
    """A path used only as the run-dir *argument* of the validators.

    It is never created: the target-protection tests below assert that such
    locations are refused, and a refusal must not depend on the directory
    existing.
    """
    return ROOT / "runs" / "train" / run_id / "material_fusion"


class TestTargetValidation:
    def test_existing_empty_directory_refused(self, scratch_dir):
        target = scratch_dir / "snaps"
        target.mkdir()
        with pytest.raises(FileExistsError):
            validate_snapshot_target(target, _comparison_run_dir("guard_target_a"))

    def test_existing_nonempty_directory_refused(self, scratch_dir):
        target = scratch_dir / "snaps"
        target.mkdir()
        marker = target / "init.pt"
        marker.write_bytes(b"precious")
        with pytest.raises(FileExistsError):
            validate_snapshot_target(target, _comparison_run_dir("guard_target_b"))
        assert marker.read_bytes() == b"precious"

    def test_existing_file_refused(self, scratch_dir):
        target = scratch_dir / "snaps"
        target.write_bytes(b"a file, not a directory")
        with pytest.raises(FileExistsError):
            validate_snapshot_target(target, _comparison_run_dir("guard_target_c"))
        assert target.read_bytes() == b"a file, not a directory"

    @pytest.mark.parametrize(
        "relative",
        ["", "datasets", "weights", "runs", "runs/train"],
    )
    def test_forbidden_locations_refused(self, relative):
        target = (ROOT / relative) if relative else ROOT
        with pytest.raises(ValueError, match="protected|refusing|forbidden"):
            validate_snapshot_target(target, _comparison_run_dir("guard_target_d"))

    def test_other_training_dir_refused(self, scratch_dir):
        target = ROOT / "runs" / "train" / "some_other_run" / "stage_snapshots"
        with pytest.raises(ValueError):
            validate_snapshot_target(target, _comparison_run_dir("guard_target_e"))

    def test_new_target_inside_this_run_dir_allowed(self, scratch_dir):
        run_dir = material_run_dir(scratch_dir, "guard_target_f")
        target = run_dir / "stage_snapshots"
        resolved = validate_snapshot_target(target, run_dir)
        assert resolved == target.resolve()

    def test_alias_path_resolved_before_check(self, scratch_dir):
        """A path spelled with '..' must resolve to the same forbidden target."""
        sneaky = ROOT / "runs" / "train" / "guard_target_g" / ".." / "guard_target_g" / "x"
        with pytest.raises(ValueError):
            validate_snapshot_target(sneaky, _comparison_run_dir("guard_target_g"))


class TestExclusiveCreation:
    def test_create_snapshot_dir_is_exclusive(self, scratch_dir):
        target = scratch_dir / "snaps"
        create_snapshot_dir(target, _comparison_run_dir("guard_create_a"))
        assert target.is_dir()
        # Second creation (simulating a post-precheck race) must not overwrite.
        with pytest.raises(FileExistsError):
            create_snapshot_dir(target, _comparison_run_dir("guard_create_a"))

    def test_snapshot_write_refuses_overwrite(self, scratch_dir):
        target = scratch_dir / "snaps"
        target.mkdir()
        model = build_material_model("gray", imgsz=64, drop_path_rate=0.0)
        path = target / "init.pt"
        write_stage_snapshot(path, model, epoch=0, stage="init")
        first = _sha256(path)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        with pytest.raises(FileExistsError):
            write_stage_snapshot(path, model, epoch=0, stage="init")
        assert _sha256(path) == first

    def test_metadata_write_refuses_overwrite(self, scratch_dir):
        path = scratch_dir / "stage_snapshots.json"
        write_stage_metadata(path, {"format": "x"})
        first = _sha256(path)
        with pytest.raises(FileExistsError):
            write_stage_metadata(path, {"format": "y"})
        assert _sha256(path) == first


class TestTrainingEntryProtection:
    def test_existing_snapshot_dir_refused_before_training(self, scratch_dir):
        run_id = "guard_snapdir_existing"
        target = scratch_dir / "existing_snaps"
        target.mkdir()
        marker = target / "init.pt"
        marker.write_bytes(b"precious")
        result = run_production_training(
            scratch_dir, run_id, epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=target,
        )
        assert result["exit_code"] == 2
        assert marker.read_bytes() == b"precious"
        assert not result["run_dir"].exists()

    def test_existing_file_target_refused(self, scratch_dir):
        run_id = "guard_snapdir_file"
        target = scratch_dir / "a_file"
        target.write_bytes(b"data")
        result = run_production_training(
            scratch_dir, run_id, epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=target,
        )
        assert result["exit_code"] == 2
        assert target.read_bytes() == b"data"
        assert not result["run_dir"].exists()

    def test_old_run_snapshot_dir_with_new_run_id_refused(self, scratch_dir):
        first = run_production_training(
            scratch_dir, "guard_snapdir_old", epochs=2, warmup_epochs=1,
            limit_batches=1, snapshot_dir=scratch_dir / "old_snaps",
        )
        assert first["exit_code"] == 0
        hashes = {
            name: _sha256(first["snapshot_dir"] / name)
            for name in ("init.pt", "post_warmup.pt", "stage_snapshots.json")
        }
        second = run_production_training(
            scratch_dir, "guard_snapdir_new", epochs=2, warmup_epochs=1,
            limit_batches=1, snapshot_dir=first["snapshot_dir"],
        )
        assert second["exit_code"] == 2
        assert not second["run_dir"].exists()
        for name, digest in hashes.items():
            assert _sha256(first["snapshot_dir"] / name) == digest

    def test_new_target_passes_and_writes_evidence(self, scratch_dir):
        run_id = "guard_snapdir_new_ok"
        target = scratch_dir / "fresh_snaps"
        result = run_production_training(
            scratch_dir, run_id, epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=target,
        )
        assert result["exit_code"] == 0
        for name in ("init.pt", "post_warmup.pt", "post_finetune.pt",
                     "stage_snapshots.json"):
            assert (target / name).is_file(), name

    def test_target_inside_this_run_dir_allowed(self, scratch_dir):
        run_id = "guard_snapdir_inside"
        run_dir = material_run_dir(scratch_dir, run_id)
        result = run_production_training(
            scratch_dir, run_id, epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=run_dir / "stage_snapshots",
        )
        assert result["exit_code"] == 0
        assert (run_dir / "stage_snapshots" / "init.pt").is_file()

    def test_forbidden_target_refused_without_run_dir(self, scratch_dir):
        run_id = "guard_snapdir_forbidden"
        result = run_production_training(
            scratch_dir, run_id, epochs=2, warmup_epochs=1, limit_batches=1,
            snapshot_dir=ROOT / "runs" / "train" / "guard_other_run" / "snaps",
        )
        assert result["exit_code"] == 2
        assert not result["run_dir"].exists()


class TestProtectedSubtrees:
    """Every descendant of the protected trees is refused, not just the root."""

    @pytest.mark.parametrize(
        "relative",
        [
            "datasets/new_dir",
            "datasets/a/b/c",
            "datasets/underwater_cls_fusion_v4_band/extra",
            "weights/new_dir",
            "weights/pretrained/nested",
        ],
    )
    def test_protected_subpaths_refused(self, relative):
        target = ROOT / relative
        with pytest.raises(ValueError):
            validate_snapshot_target(target, _comparison_run_dir("guard_subpath_a"))

    def test_dotdot_alias_into_protected_tree_refused(self, scratch_dir):
        sneaky = ROOT / "analysis" / ".." / "datasets" / "sneaky"
        with pytest.raises(ValueError):
            validate_snapshot_target(sneaky, _comparison_run_dir("guard_subpath_b"))
        sneaky_weights = ROOT / "runs" / ".." / "weights" / "sneaky"
        with pytest.raises(ValueError):
            validate_snapshot_target(sneaky_weights, _comparison_run_dir("guard_subpath_c"))

    def test_name_similar_directories_not_misjudged(self, scratch_dir):
        """A directory merely named like a protected tree is not a descendant."""
        for name in ("datasets_extra", "datasets", "weights_backup", "weights"):
            target = scratch_dir / name / "snaps"
            resolved = validate_snapshot_target(target, _comparison_run_dir("guard_subpath_d"))
            assert resolved == target.resolve()

    def test_analysis_runs_new_directory_allowed(self, scratch_dir):
        target = scratch_dir / "analysis" / "runs" / "snaps" / "stage_snapshots"
        resolved = validate_snapshot_target(target, _comparison_run_dir("guard_subpath_e"))
        assert resolved == target.resolve()

    def test_other_run_subdirectory_refused(self, scratch_dir):
        target = ROOT / "runs" / "train" / "guard_other_run_x" / "material_fusion" / "snaps"
        with pytest.raises(ValueError):
            validate_snapshot_target(target, _comparison_run_dir("guard_subpath_f"))

    def test_runs_root_descendant_outside_this_run_refused(self, scratch_dir):
        target = ROOT / "runs" / "guard_other_place" / "snaps"
        with pytest.raises(ValueError):
            validate_snapshot_target(target, _comparison_run_dir("guard_subpath_g"))

    def test_this_run_subdirectory_still_allowed(self, scratch_dir):
        run_dir = material_run_dir(scratch_dir, "guard_subpath_h")
        target = run_dir / "stage_snapshots"
        assert validate_snapshot_target(target, run_dir) == target.resolve()
