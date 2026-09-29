"""Run-config / checkpoint consistency contract (guard round).

Evaluation and verification must call the same strict contract: every
field of the resolved run configuration is type-checked and compared
against the checkpoint metadata (with explicit field mapping, e.g.
``config.batch`` <-> ``metadata.physical_batch``); a doctored run config
must be refused before any report is written. Legacy/template sources are
labeled and never presented as confirmed historical facts.
"""

from __future__ import annotations

import json
import shutil

import pytest

from core.fusion_dataset import dataset_fingerprint
from models.material_fusion import (
    CLASS_NAMES,
    MATERIAL_FUSION_VERSION,
    PREPROCESSING_VERSION,
    UPSTREAM_COMMIT,
    check_run_config_checkpoint_consistency,
    expected_stage_for_epoch,
)

from .conftest import build_v4_dataset, run_production_training
from .test_eval import GrayFallbackStub


def _base_config(**overrides):
    config = {
        "experiment": MATERIAL_FUSION_VERSION,
        "variant": "gray_polar",
        "data": "datasets/underwater_cls_fusion_v4_band",
        "data_fingerprint": "f" * 64,
        "pretrained": "weights/pretrained/mit_b0.pth",
        "pretrained_sha256": "a" * 64,
        "upstream_commit": UPSTREAM_COMMIT,
        "preprocessing_version": PREPROCESSING_VERSION,
        "imgsz": 64,
        "batch": 2,
        "accum_steps": 2,
        "effective_batch": 4,
        "epochs": 4,
        "warmup_epochs": 2,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "lr": 1e-4,
        "encoder_lr": 1e-5,
        "seed": 2026,
        "workers": 0,
        "fallback_weights": "",
        "fallback_identity": "",
        "smoke": False,
        "limit_batches": 0,
    }
    config.update(overrides)
    return config


def _base_metadata(**overrides):
    metadata = {
        "version": MATERIAL_FUSION_VERSION,
        "variant": "gray_polar",
        "class_names": list(CLASS_NAMES),
        "imgsz": 64,
        "preprocessing_version": PREPROCESSING_VERSION,
        "data_fingerprint": "f" * 64,
        "upstream_commit": UPSTREAM_COMMIT,
        "pretrained_sha256": "a" * 64,
        "seed": 2026,
        "epoch": 3,
        "stage": "finetune",
        "smoke": False,
        "limit_batches": 0,
        "val_macro_f1": 0.1,
        "val_accuracy": 0.2,
        "physical_batch": 2,
        "accum_steps": 2,
        "effective_batch": 4,
        "drop_path_rate": 0.0,
        "init_from": "",
        "fallback_weights": "",
        "fallback_identity": "",
    }
    metadata.update(overrides)
    return metadata


class TestConsistencyContract:
    def test_valid_pair_passes(self):
        report = check_run_config_checkpoint_consistency(
            _base_config(), _base_metadata()
        )
        assert report["asserted"]["batch"] == 2

    @pytest.mark.parametrize(
        "config_override,metadata_override",
        [
            ({"variant": "gray"}, {}),
            ({"imgsz": 448}, {}),
            ({"preprocessing_version": "v4_band_224_area_linear_v0"}, {}),
            ({"upstream_commit": "0" * 40}, {}),
            ({"pretrained_sha256": "b" * 64}, {}),
            ({"data_fingerprint": "0" * 64}, {}),
            ({"seed": 1}, {}),
            ({"batch": 4}, {}),          # config.batch <-> metadata.physical_batch
            ({"accum_steps": 4, "effective_batch": 8}, {}),
            ({"smoke": True, "limit_batches": 4}, {}),
            ({}, {"physical_batch": 4}),
            ({}, {"seed": 7}),
            ({}, {"limit_batches": 3, "smoke": True}),
        ],
    )
    def test_single_field_conflict_rejected(self, config_override, metadata_override):
        with pytest.raises(ValueError):
            check_run_config_checkpoint_consistency(
                _base_config(**config_override), _base_metadata(**metadata_override)
            )

    @pytest.mark.parametrize(
        "config_override,metadata_override",
        [
            ({"limit_batches": False}, {}),           # bool where int required
            ({"smoke": 0}, {}),                       # int where bool required
            ({"effective_batch": 4.0, "batch": 2.0, "accum_steps": 2.0}, {}),
            ({"imgsz": 64.0}, {}),
            ({"seed": True}, {}),
            ({}, {"limit_batches": False}),
            ({}, {"physical_batch": True}),
        ],
    )
    def test_type_conflicts_rejected_even_when_values_match(
        self, config_override, metadata_override
    ):
        with pytest.raises(ValueError):
            check_run_config_checkpoint_consistency(
                _base_config(**config_override), _base_metadata(**metadata_override)
            )

    def test_epoch_stage_boundary_enforced(self):
        # warmup_epochs=2: epoch 2 must be warmup, epoch 3 finetune.
        check_run_config_checkpoint_consistency(
            _base_config(), _base_metadata(epoch=2, stage="warmup")
        )
        check_run_config_checkpoint_consistency(
            _base_config(), _base_metadata(epoch=3, stage="finetune")
        )
        with pytest.raises(ValueError, match="stage"):
            check_run_config_checkpoint_consistency(
                _base_config(), _base_metadata(epoch=2, stage="finetune")
            )
        with pytest.raises(ValueError):
            check_run_config_checkpoint_consistency(
                _base_config(), _base_metadata(epoch=5, stage="finetune")  # > epochs
            )
        with pytest.raises(ValueError):
            check_run_config_checkpoint_consistency(
                _base_config(), _base_metadata(epoch=0, stage="warmup")
            )

    def test_early_best_epoch_allowed(self):
        """best may be an earlier epoch; only the stage boundary is enforced."""
        report = check_run_config_checkpoint_consistency(
            _base_config(), _base_metadata(epoch=1, stage="warmup")
        )
        assert report["asserted"]["epoch"] == 1

    def test_unasserted_fields_recorded_not_silently_masked(self):
        report = check_run_config_checkpoint_consistency(
            _base_config(pretrained_sha256="", data_fingerprint=""),
            _base_metadata(),
        )
        assert "pretrained_sha256" in report["unasserted"]
        assert "data_fingerprint" in report["unasserted"]

    def test_expected_stage_helper(self):
        assert expected_stage_for_epoch(1, 2) == "warmup"
        assert expected_stage_for_epoch(2, 2) == "warmup"
        assert expected_stage_for_epoch(3, 2) == "finetune"
        assert expected_stage_for_epoch(1, 0) == "finetune"


@pytest.fixture(scope="module")
def guarded_run(module_scratch):
    import torch

    scratch = module_scratch
    result = run_production_training(
        scratch, "guard_consistency_run", variant="gray_polar",
        epochs=2, warmup_epochs=1, limit_batches=1,
        snapshot_dir=scratch / "stage_snapshots",
    )
    assert result["exit_code"] == 0
    fallback = scratch / "gray_fallback.pt"
    torch.save({"model": GrayFallbackStub(imgsz=64)}, fallback)
    result["fallback"] = fallback
    yield result


def _doctored_run_dir(scratch_dir, run, mutate):
    target = scratch_dir / "doctored"
    target.mkdir()
    shutil.copy(run["run_dir"] / "last.pt", target / "last.pt")
    payload = json.loads((run["run_dir"] / "run_config.json").read_text(encoding="utf-8"))
    mutate(payload["config"])
    (target / "run_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return target / "last.pt"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: cfg.update(imgsz=448),
        lambda cfg: cfg.update(pretrained_sha256="0" * 64),
        lambda cfg: cfg.update(batch=16, effective_batch=32),
        lambda cfg: cfg.update(smoke=False, limit_batches=0),
        lambda cfg: cfg.update(variant="gray"),
        lambda cfg: cfg.update(seed=1),
        lambda cfg: cfg.update(preprocessing_version="v4_band_224_area_linear_v0"),
        lambda cfg: cfg.update(upstream_commit="0" * 40),
        lambda cfg: cfg.update(smoke=False),  # type/value conflict with metadata
    ],
)
def test_eval_refuses_doctored_run_config(guarded_run, scratch_dir, mutate):
    from scripts.eval_material_fusion import main as eval_main

    checkpoint = _doctored_run_dir(scratch_dir, guarded_run, mutate)
    output = scratch_dir / "eval.json"
    code = eval_main([
        "--checkpoint", str(checkpoint),
        "--split", "val",
        "--device", "cpu",
        "--no-fallback",
        "--output", str(output),
    ])
    assert code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: cfg.update(imgsz=448),
        lambda cfg: cfg.update(batch=16, effective_batch=32),
        lambda cfg: cfg.update(smoke=False, limit_batches=0),
    ],
)
def test_verify_refuses_doctored_run_config(guarded_run, scratch_dir, mutate):
    from scripts.verify_material_fusion import main as verify_main

    checkpoint = _doctored_run_dir(scratch_dir, guarded_run, mutate)
    output = scratch_dir / "verify.json"
    code = verify_main([
        "--checkpoint", str(checkpoint),
        "--stage-snapshots", str(guarded_run["snapshot_dir"]),
        "--device", "cpu",
        "--limit-batches", "1",
        "--output", str(output),
    ])
    assert code == 2
    assert not output.exists()


def test_explicit_variant_conflicting_with_run_config_refused(guarded_run, scratch_dir):
    from scripts.eval_material_fusion import main as eval_main

    output = scratch_dir / "eval_variant_conflict.json"
    code = eval_main([
        "--checkpoint", str(guarded_run["run_dir"] / "last.pt"),
        "--variant", "gray",           # run config says gray_polar
        "--split", "val",
        "--device", "cpu",
        "--output", str(output),
    ])
    assert code == 2
    assert not output.exists()


def test_data_relocation_with_same_fingerprint_allowed(guarded_run, scratch_dir):
    from scripts.eval_material_fusion import main as eval_main

    moved = scratch_dir / "moved_v4"
    shutil.copytree(guarded_run["data_root"], moved)
    output = scratch_dir / "eval_moved.json"
    code = eval_main([
        "--checkpoint", str(guarded_run["run_dir"] / "last.pt"),
        "--data", str(moved),
        "--split", "val",
        "--device", "cpu",
        "--no-fallback",
        "--output", str(output),
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data"]["root"] == str(moved)
    assert report["data"]["audit_fingerprint"] == dataset_fingerprint(moved)


def test_pretrained_relocation_same_content_allowed(guarded_run, scratch_dir):
    from scripts.verify_material_fusion import main as verify_main

    moved = scratch_dir / "moved_mit.pth"
    shutil.copy(guarded_run["pretrained"], moved)
    output = scratch_dir / "verify_moved_pretrained.json"
    code = verify_main([
        "--checkpoint", str(guarded_run["run_dir"] / "last.pt"),
        "--stage-snapshots", str(guarded_run["snapshot_dir"]),
        "--pretrained", str(moved),
        "--fallback-weights", str(guarded_run["fallback"]),
        "--device", "cpu",
        "--limit-batches", "1",
        "--output", str(output),
    ])
    assert code == 0


def test_pretrained_relocation_different_content_refused(guarded_run, scratch_dir):
    from scripts.verify_material_fusion import main as verify_main

    moved = scratch_dir / "different_mit.pth"
    moved.write_bytes(b"not the recorded checkpoint")
    output = scratch_dir / "verify_bad_pretrained.json"
    code = verify_main([
        "--checkpoint", str(guarded_run["run_dir"] / "last.pt"),
        "--stage-snapshots", str(guarded_run["snapshot_dir"]),
        "--pretrained", str(moved),
        "--fallback-weights", str(guarded_run["fallback"]),
        "--device", "cpu",
        "--limit-batches", "1",
        "--output", str(output),
    ])
    assert code == 2
    assert not output.exists()


def test_eval_batch_and_limit_may_differ_from_training(guarded_run, scratch_dir):
    from scripts.eval_material_fusion import main as eval_main

    output = scratch_dir / "eval_batch_diff.json"
    code = eval_main([
        "--checkpoint", str(guarded_run["run_dir"] / "last.pt"),
        "--split", "val",
        "--device", "cpu",
        "--batch", "3",
        "--limit-batches", "2",
        "--no-fallback",
        "--output", str(output),
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["training"]["limit_batches"] == 1
    assert report["evaluation"]["limit_batches"] == 2
