"""Training-loop mechanics: stages, optimizer coverage, selection, artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from core.material_fusion_dataset import MaterialFusionDataset
from models.material_fusion import (
    CLASS_NAMES,
    MATERIAL_FUSION_VERSION,
    build_material_model,
    read_material_checkpoint_metadata,
)
from models.polar_fusion import read_manifest_split
from scripts.train_material_fusion import (
    apply_stage,
    build_optimizer,
    load_experiment_config,
    main,
    run_training,
    should_replace_best,
    stage_for_epoch,
    train_one_epoch,
)

from .conftest import (
    build_v4_dataset,
    write_experiment_config,
    write_synthetic_pretrained,
)


@pytest.fixture()
def training_setup(scratch_dir, train_runs_root):
    """Synthetic data plus a redirected output root for the real entry point."""
    data_root = build_v4_dataset(scratch_dir / "v4")
    pretrained = write_synthetic_pretrained(scratch_dir / "mit_b0_syn.pth")
    config_path = write_experiment_config(
        scratch_dir / "config.yaml", data_root, pretrained
    )
    return {
        "data_root": data_root,
        "pretrained": pretrained,
        "config_path": config_path,
        "runs_root": train_runs_root,
    }


def _run_dir(runs_root, run_id):
    """This test's training output directory under the redirected root."""
    return Path(runs_root) / run_id / "material_fusion"


class TestStages:
    def test_stage_for_epoch(self):
        assert stage_for_epoch(0, 2) == "warmup"
        assert stage_for_epoch(1, 2) == "warmup"
        assert stage_for_epoch(2, 2) == "finetune"
        assert stage_for_epoch(29, 2) == "finetune"
        assert stage_for_epoch(0, 0) == "finetune"

    def test_apply_stage_freezes_backbone_only_then_unfreezes(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        apply_stage(model, "warmup")
        groups = model.parameter_groups()
        assert all(not p.requires_grad for p in groups["backbone"])
        assert all(p.requires_grad for p in groups["fusion"])
        assert all(p.requires_grad for p in groups["head"])
        assert not model.encoder.patch_embed1.training
        assert model.encoder.FRMs[0].training
        apply_stage(model, "finetune")
        assert all(p.requires_grad for p in groups["backbone"])
        assert model.encoder.patch_embed1.training


class TestOptimizer:
    def test_three_param_groups_with_protocol_lrs(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        optimizer = build_optimizer(model, lr=1e-4, encoder_lr=1e-5, weight_decay=0.01)
        by_name = {group["name"]: group for group in optimizer.param_groups}
        assert set(by_name) == {"backbone", "fusion", "head"}
        assert by_name["backbone"]["lr"] == 1e-5
        assert by_name["fusion"]["lr"] == 1e-4
        assert by_name["head"]["lr"] == 1e-4
        assert all(g["weight_decay"] == 0.01 for g in optimizer.param_groups)

    def test_single_stream_has_no_fusion_group(self):
        model = build_material_model("gray", imgsz=64, drop_path_rate=0.0)
        optimizer = build_optimizer(model, lr=1e-4, encoder_lr=1e-5, weight_decay=0.01)
        names = {group["name"] for group in optimizer.param_groups}
        assert names == {"backbone", "head"}

    @pytest.mark.parametrize("variant", ("gray", "gray_gray_valid", "gray_valid", "gray_polar"))
    def test_optimizer_covers_every_parameter_exactly_once(self, variant):
        model = build_material_model(variant, imgsz=64, drop_path_rate=0.0)
        optimizer = build_optimizer(model, lr=1e-4, encoder_lr=1e-5, weight_decay=0.01)
        covered = [id(p) for group in optimizer.param_groups for p in group["params"]]
        model_params = [id(p) for p in model.parameters()]
        assert sorted(covered) == sorted(model_params)
        assert len(covered) == len(set(covered))

    def test_warmup_freezes_backbone_but_trains_fusion_and_head(self, scratch_dir):
        data_root = build_v4_dataset(scratch_dir / "v4")
        paths, _, _ = read_manifest_split(data_root, "train")
        loader = torch.utils.data.DataLoader(
            MaterialFusionDataset(paths, variant="gray_polar", imgsz=64),
            batch_size=2, shuffle=False,
        )
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        optimizer = build_optimizer(model, lr=1e-3, encoder_lr=1e-3, weight_decay=0.0)
        groups = model.parameter_groups()
        backbone_before = [p.detach().clone() for p in groups["backbone"]]
        fusion_before = [p.detach().clone() for p in groups["fusion"]]
        head_before = [p.detach().clone() for p in groups["head"]]
        stats = train_one_epoch(
            model, loader, optimizer, "cpu", accum=1, limit_batches=2, stage="warmup"
        )
        assert stats["gradients_finite"] is True
        assert all(
            torch.equal(before, after)
            for before, after in zip(backbone_before, groups["backbone"])
        )
        assert any(
            not torch.equal(before, after)
            for before, after in zip(fusion_before, groups["fusion"])
        )
        assert any(
            not torch.equal(before, after)
            for before, after in zip(head_before, groups["head"])
        )

    def test_finetune_updates_backbone_fusion_and_head(self, scratch_dir):
        data_root = build_v4_dataset(scratch_dir / "v4")
        paths, _, _ = read_manifest_split(data_root, "train")
        loader = torch.utils.data.DataLoader(
            MaterialFusionDataset(paths, variant="gray_polar", imgsz=64),
            batch_size=2, shuffle=False,
        )
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        optimizer = build_optimizer(model, lr=1e-3, encoder_lr=1e-3, weight_decay=0.0)
        groups = model.parameter_groups()
        before_by_group = {
            name: [p.detach().clone() for p in groups[name]]
            for name in ("backbone", "fusion", "head")
        }
        train_one_epoch(model, loader, optimizer, "cpu", accum=1, limit_batches=2, stage="finetune")
        for name in ("backbone", "fusion", "head"):
            assert any(
                not torch.equal(old, new)
                for old, new in zip(before_by_group[name], groups[name])
            ), name

    def test_accumulation_tail_batch_steps_optimizer(self, scratch_dir):
        data_root = build_v4_dataset(scratch_dir / "v4")
        paths, _, _ = read_manifest_split(data_root, "train")
        loader = torch.utils.data.DataLoader(
            MaterialFusionDataset(paths, variant="gray_polar", imgsz=64),
            batch_size=2, shuffle=False,
        )
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        optimizer = build_optimizer(model, lr=1e-2, encoder_lr=1e-2, weight_decay=0.0)
        head_before = [p.detach().clone() for p in model.parameter_groups()["head"]]
        # 3 physical batches with accum=2: one full group + one tail group.
        stats = train_one_epoch(
            model, loader, optimizer, "cpu", accum=2, limit_batches=3, stage="finetune"
        )
        assert stats["optimizer_steps"] == 2
        assert stats["batches"] == 3
        assert any(
            not torch.equal(before, after)
            for before, after in zip(head_before, model.parameter_groups()["head"])
        )


class TestSelection:
    def test_strictly_greater_replaces(self):
        assert should_replace_best((0.5, 0.8), (0.6, 0.1)) is True
        assert should_replace_best((0.5, 0.8), (0.5, 0.9)) is True
        assert should_replace_best((0.5, 0.8), (0.5, 0.7)) is False

    def test_tie_keeps_earlier_epoch(self):
        assert should_replace_best((0.5, 0.8), (0.5, 0.8)) is False
        assert should_replace_best((0.0, 0.0), (0.0, 0.0)) is False


class TestEndToEnd:
    def test_smoke_training_writes_artifacts_and_metadata(self, training_setup, scratch_dir):
        args = [
            "--run-id", "test_smoke_e4",
            "--config", str(training_setup["config_path"]),
            "--device", "cpu",
            "--epochs", "2",
            "--warmup-epochs", "1",
            "--limit-batches", "1",
        ]
        assert main(args) == 0
        run_dir = _run_dir(training_setup["runs_root"], "test_smoke_e4")
        assert (run_dir / "best.pt").is_file()
        assert (run_dir / "last.pt").is_file()
        history = [
            json.loads(line)
            for line in (run_dir / "metrics_history.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(history) == 2
        assert [entry["stage"] for entry in history] == ["warmup", "finetune"]
        assert all("val_macro_f1" in entry for entry in history)
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        assert run_config["config"]["limit_batches"] == 1
        assert run_config["config"]["smoke"] is True
        assert run_config["config"]["variant"] == "gray_polar"
        assert run_config["variant"] == "gray_polar"
        assert len(run_config["pretrained_sha256"]) == 64

        metadata = read_material_checkpoint_metadata(run_dir / "best.pt")
        assert metadata.variant == "gray_polar"
        assert metadata.smoke is True
        assert metadata.limit_batches == 1
        assert metadata.class_names == CLASS_NAMES
        assert metadata.version == MATERIAL_FUSION_VERSION
        assert metadata.effective_batch == 4
        assert metadata.data_fingerprint
        assert metadata.epoch in (1, 2)
        assert metadata.stage in ("warmup", "finetune")
    def test_training_never_reads_test_split(self, training_setup, monkeypatch):
        import scripts.train_material_fusion as train_module

        seen_splits = []
        original = train_module.read_manifest_split

        def recording_read(data_root, split):
            seen_splits.append(split)
            return original(data_root, split)

        monkeypatch.setattr(train_module, "read_manifest_split", recording_read)
        args = [
            "--run-id", "test_splits_guard",
            "--config", str(training_setup["config_path"]),
            "--device", "cpu",
            "--epochs", "2",
            "--warmup-epochs", "1",
            "--limit-batches", "1",
        ]
        assert main(args) == 0
        assert "test" not in seen_splits
        assert "train" in seen_splits and "val" in seen_splits

    def test_existing_run_dir_is_refused(self, training_setup):
        run_dir = _run_dir(training_setup["runs_root"], "test_existing")
        run_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "--run-id", "test_existing",
            "--config", str(training_setup["config_path"]),
            "--device", "cpu",
            "--limit-batches", "1",
        ]
        assert main(args) == 2

    def test_bad_run_id_refused(self, training_setup):
        args = [
            "--run-id", "../escape",
            "--config", str(training_setup["config_path"]),
            "--device", "cpu",
        ]
        assert main(args) == 2

    def test_missing_pretrained_is_refused_not_random_init(self, training_setup, scratch_dir):
        config_path = write_experiment_config(
            scratch_dir / "config_missing.yaml",
            training_setup["data_root"],
            scratch_dir / "does_not_exist.pth",
        )
        args = [
            "--run-id", "test_missing_pretrained",
            "--config", str(config_path),
            "--device", "cpu",
            "--limit-batches", "1",
        ]
        assert main(args) == 2
        run_dir = (
            _run_dir(training_setup["runs_root"], "test_missing_pretrained")
        )
        assert not run_dir.exists()

    def test_config_dataset_fingerprint_mismatch_refused(self, training_setup, scratch_dir):
        import yaml

        config = yaml.safe_load(training_setup["config_path"].read_text(encoding="utf-8"))
        config["data_fingerprint"] = "0" * 64
        path = scratch_dir / "config_fp.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        args = [
            "--run-id", "test_fp_mismatch",
            "--config", str(path),
            "--device", "cpu",
            "--limit-batches", "1",
        ]
        assert main(args) == 2

    def test_load_experiment_config_reads_yaml(self, training_setup):
        config = load_experiment_config(training_setup["config_path"])
        assert config["variant"] == "gray_polar"
        assert config["imgsz"] == 64


class TestBestLastAndSnapshots:
    def _run(self, training_setup, run_id, extra_args=None):
        args = [
            "--run-id", run_id,
            "--config", str(training_setup["config_path"]),
            "--device", "cpu",
            "--epochs", "2",
            "--warmup-epochs", "1",
            "--limit-batches", "1",
        ] + (extra_args or [])
        assert main(args) == 0
        return _run_dir(training_setup["runs_root"], run_id)

    def test_last_checkpoint_matches_last_epoch_not_best(self, training_setup, monkeypatch):
        """Scripted metrics: epoch 1 wins, epoch 2 is worse -> best != last."""
        import scripts.train_material_fusion as train_module

        scripted = [
            {"accuracy": 0.9, "macro_f1": 0.9, "per_class_recall": {"0": 0.9}},
            {"accuracy": 0.1, "macro_f1": 0.1, "per_class_recall": {"0": 0.1}},
        ]
        calls = {"n": 0}

        def fake_evaluate(model, loader, device, limit_batches=0):
            result = scripted[min(calls["n"], len(scripted) - 1)]
            calls["n"] += 1
            return result

        monkeypatch.setattr(train_module, "evaluate_model", fake_evaluate)
        run_dir = self._run(training_setup, "test_best_last")
        history = [
            json.loads(line)
            for line in (run_dir / "metrics_history.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        last_meta = read_material_checkpoint_metadata(run_dir / "last.pt")
        best_meta = read_material_checkpoint_metadata(run_dir / "best.pt")
        assert last_meta.epoch == 2
        assert last_meta.stage == "finetune"
        assert last_meta.val_macro_f1 == history[-1]["val_macro_f1"] == 0.1
        assert last_meta.val_accuracy == history[-1]["val_accuracy"] == 0.1
        assert best_meta.epoch == 1
        assert best_meta.stage == "warmup"
        assert best_meta.val_macro_f1 == 0.9
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        assert run_config["best"]["epoch"] == 1
        assert run_config["best"]["val_macro_f1"] == 0.9
        assert run_config["last"]["epoch"] == 2
        assert run_config["last"]["val_macro_f1"] == 0.1
        assert run_config["best"] != run_config["last"]
    def test_stage_snapshots_capture_real_training_states(self, training_setup, scratch_dir):
        snapshot_dir = scratch_dir / "stage_snapshots"
        run_dir = self._run(
            training_setup, "test_snapshots", ["--stage-snapshot-dir", str(snapshot_dir)]
        )
        init_path = snapshot_dir / "init.pt"
        warmup_path = snapshot_dir / "post_warmup.pt"
        finetune_path = snapshot_dir / "post_finetune.pt"
        meta_path = snapshot_dir / "stage_snapshots.json"
        for path in (init_path, warmup_path, finetune_path, meta_path):
            assert path.is_file(), path

        init = torch.load(init_path, map_location="cpu", weights_only=True)
        warmup = torch.load(warmup_path, map_location="cpu", weights_only=True)
        finetune = torch.load(finetune_path, map_location="cpu", weights_only=True)
        assert init["stage"] == "init" and init["epoch"] == 0
        assert warmup["stage"] == "warmup" and warmup["epoch"] == 1
        assert finetune["stage"] == "finetune" and finetune["epoch"] == 2

        backbone_keys = [
            key for key in init["state_dict"] if key.startswith("encoder.")
            and "FRMs" not in key and "FFMs" not in key
        ]
        fusion_keys = [key for key in init["state_dict"] if "FRMs" in key or "FFMs" in key]
        # Warmup froze the backbone but trained the fusion layers.
        assert all(
            torch.equal(init["state_dict"][key], warmup["state_dict"][key])
            for key in backbone_keys
        )
        assert any(
            not torch.equal(init["state_dict"][key], warmup["state_dict"][key])
            for key in fusion_keys
        )
        # Finetune unfroze and trained the backbone.
        assert any(
            not torch.equal(warmup["state_dict"][key], finetune["state_dict"][key])
            for key in backbone_keys
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["warmup_epochs"] == 1
        assert meta["epochs"] == 2
    def test_history_records_gradient_finiteness_and_steps(self, training_setup):
        run_dir = self._run(training_setup, "test_history_fields")
        history = [
            json.loads(line)
            for line in (run_dir / "metrics_history.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(history) == 2
        for entry in history:
            assert entry["gradients_finite"] is True
            assert entry["optimizer_steps"] == 1  # 1 batch, accum 2 -> ceil(1/2)=1
            assert entry["samples"] > 0
    def test_run_config_separates_training_and_param_groups(self, training_setup):
        run_dir = self._run(training_setup, "test_run_config_groups")
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        groups = {g["name"]: g for g in run_config["param_groups"]}
        assert groups["backbone"]["lr"] == 1e-5
        assert groups["fusion"]["lr"] == 1e-4
        assert groups["head"]["lr"] == 1e-4
        assert groups["backbone"]["params"] > 0
        assert run_config["config"]["variant"] == "gray_polar"
        assert run_config["config"]["effective_batch"] == 4
