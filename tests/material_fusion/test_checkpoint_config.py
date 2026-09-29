"""Checkpoint and config validation: strict types, separation, consistency."""

from __future__ import annotations

import pytest
import torch

from models.material_fusion import (
    MATERIAL_FUSION_VERSION,
    PREPROCESSING_VERSION,
    UPSTREAM_COMMIT,
    build_material_model,
    check_config_checkpoint_consistency,
    load_material_checkpoint,
    material_fusion_metadata,
    read_material_checkpoint_metadata,
    save_material_checkpoint,
    validate_checkpoint_metadata,
    validate_material_config,
)


def _valid_config(**overrides):
    config = {
        "experiment": MATERIAL_FUSION_VERSION,
        "variant": "gray_polar",
        "data": "datasets/underwater_cls_fusion_v4_band",
        "data_fingerprint": "",
        "pretrained": "weights/pretrained/mit_b0.pth",
        "pretrained_sha256": "",
        "upstream_commit": UPSTREAM_COMMIT,
        "preprocessing_version": PREPROCESSING_VERSION,
        "imgsz": 64,
        "batch": 2,
        "accum_steps": 2,
        "effective_batch": 4,
        "epochs": 2,
        "warmup_epochs": 1,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "lr": 1e-4,
        "encoder_lr": 1e-5,
        "seed": 2026,
        "workers": 0,
        "fallback_weights": "runs/train/yolo26_formal_001/gray_fusion/best.pt",
        "smoke": False,
        "limit_batches": 0,
    }
    config.update(overrides)
    return config


def _metadata(**overrides):
    values = dict(
        version=MATERIAL_FUSION_VERSION,
        variant="gray_polar",
        class_names=("metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"),
        imgsz=64,
        preprocessing_version=PREPROCESSING_VERSION,
        data_fingerprint="abc123",
        upstream_commit=UPSTREAM_COMMIT,
        pretrained_sha256="0" * 64,
        seed=2026,
        epoch=2,
        stage="finetune",
        smoke=False,
        limit_batches=0,
        val_macro_f1=0.5,
        val_accuracy=0.6,
        physical_batch=2,
        accum_steps=2,
        effective_batch=4,
        drop_path_rate=0.1,
        init_from="",
        fallback_weights="runs/train/yolo26_formal_001/gray_fusion/best.pt",
        fallback_identity="yolo26n-cls 4-class",
    )
    values.update(overrides)
    return material_fusion_metadata(**values)


class TestConfigValidation:
    def test_valid_config_passes(self):
        result = validate_material_config(_valid_config())
        assert result["variant"] == "gray_polar"

    def test_bad_variant_rejected(self):
        with pytest.raises(ValueError, match="variant"):
            validate_material_config(_valid_config(variant="rgb"))

    def test_smoke_must_be_bool(self):
        with pytest.raises(ValueError, match="smoke"):
            validate_material_config(_valid_config(smoke=1))
        with pytest.raises(ValueError, match="smoke"):
            validate_material_config(_valid_config(smoke="true"))

    def test_limit_batches_must_be_strict_int(self):
        for bad in (True, 1.0, "3"):
            with pytest.raises(ValueError, match="limit_batches"):
                validate_material_config(_valid_config(limit_batches=bad))

    def test_warmup_must_be_below_epochs(self):
        with pytest.raises(ValueError, match="warmup"):
            validate_material_config(_valid_config(epochs=2, warmup_epochs=2))

    def test_batch_times_accum_must_equal_effective(self):
        with pytest.raises(ValueError, match="effective_batch"):
            validate_material_config(
                _valid_config(batch=3, accum_steps=2, effective_batch=4)
            )

    def test_upstream_commit_pinned(self):
        with pytest.raises(ValueError, match="upstream_commit"):
            validate_material_config(_valid_config(upstream_commit="main"))


class TestCheckpointMetadataValidation:
    def test_valid_metadata_passes(self):
        payload = {"metadata": _metadata().__dict__, "state_dict": {}}
        result = validate_checkpoint_metadata(payload)
        assert result["smoke"] is False
        assert result["limit_batches"] == 0

    def test_smoke_strict_bool(self):
        for bad in (1, 0, "true", None):
            payload = {"metadata": _metadata(smoke=bad).__dict__, "state_dict": {}}
            with pytest.raises(ValueError, match="smoke"):
                validate_checkpoint_metadata(payload)

    def test_limit_batches_strict_int(self):
        for bad in (True, False, 1.0, "3"):
            payload = {"metadata": _metadata(limit_batches=bad).__dict__, "state_dict": {}}
            with pytest.raises(ValueError, match="limit_batches"):
                validate_checkpoint_metadata(payload)

    def test_formal_smoke_consistency(self):
        payload = {"metadata": _metadata(smoke=True, limit_batches=0).__dict__, "state_dict": {}}
        with pytest.raises(ValueError, match="limit_batches"):
            validate_checkpoint_metadata(payload)
        payload = {"metadata": _metadata(smoke=False, limit_batches=3).__dict__, "state_dict": {}}
        with pytest.raises(ValueError, match="limit_batches"):
            validate_checkpoint_metadata(payload)

    def test_missing_field_rejected(self):
        meta = _metadata().__dict__.copy()
        del meta["data_fingerprint"]
        with pytest.raises(ValueError, match="data_fingerprint"):
            validate_checkpoint_metadata({"metadata": meta, "state_dict": {}})

    def test_bad_stage_rejected(self):
        payload = {"metadata": _metadata(stage="freeze").__dict__, "state_dict": {}}
        with pytest.raises(ValueError, match="stage"):
            validate_checkpoint_metadata(payload)


class TestConsistency:
    def test_consistent_pair_passes(self):
        config = _valid_config(pretrained_sha256="0" * 64)
        check_config_checkpoint_consistency(config, _metadata().__dict__)

    def test_variant_mismatch_rejected(self):
        config = _valid_config(variant="gray")
        with pytest.raises(ValueError, match="variant"):
            check_config_checkpoint_consistency(config, _metadata().__dict__)

    def test_imgsz_mismatch_rejected(self):
        config = _valid_config(imgsz=224)
        with pytest.raises(ValueError, match="imgsz"):
            check_config_checkpoint_consistency(config, _metadata().__dict__)

    def test_effective_batch_mismatch_rejected(self):
        config = _valid_config(batch=8, accum_steps=2, effective_batch=16)
        with pytest.raises(ValueError, match="effective_batch"):
            check_config_checkpoint_consistency(config, _metadata().__dict__)

    def test_data_fingerprint_mismatch_rejected(self):
        config = _valid_config(data_fingerprint="expected")
        with pytest.raises(ValueError, match="data_fingerprint"):
            check_config_checkpoint_consistency(config, _metadata(data_fingerprint="actual").__dict__)

    def test_pretrained_sha_mismatch_rejected(self):
        config = _valid_config(pretrained_sha256="1" * 64)
        with pytest.raises(ValueError, match="pretrained_sha256"):
            check_config_checkpoint_consistency(config, _metadata(pretrained_sha256="2" * 64).__dict__)


class TestSaveLoadRoundtrip:
    def test_save_read_load_roundtrip(self, scratch_dir):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        metadata = _metadata()
        path = save_material_checkpoint(scratch_dir / "best.pt", model, metadata)
        read_back = read_material_checkpoint_metadata(path)
        assert read_back.variant == "gray_polar"
        assert read_back.limit_batches == 0
        assert read_back.smoke is False

        restored, restored_meta, payload = load_material_checkpoint(path, variant="gray_polar", imgsz=64)
        assert restored_meta.class_names == metadata.class_names
        original = model.state_dict()
        for key, value in restored.state_dict().items():
            assert torch.equal(value, original[key]), key

    def test_save_refuses_class_count_mismatch(self, scratch_dir):
        model = build_material_model("gray_polar", imgsz=64, num_classes=4, drop_path_rate=0.0)
        metadata = _metadata(class_names=("a", "b"))
        with pytest.raises(ValueError, match="class_names"):
            save_material_checkpoint(scratch_dir / "bad.pt", model, metadata)

    def test_load_refuses_variant_mismatch(self, scratch_dir):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        path = save_material_checkpoint(scratch_dir / "best.pt", model, _metadata())
        with pytest.raises(ValueError, match="variant"):
            load_material_checkpoint(path, variant="gray", imgsz=64)

    def test_load_refuses_imgsz_mismatch(self, scratch_dir):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        path = save_material_checkpoint(scratch_dir / "best.pt", model, _metadata())
        with pytest.raises(ValueError, match="imgsz"):
            load_material_checkpoint(path, variant="gray_polar", imgsz=224)

    def test_save_does_not_change_model_mode_or_device(self, scratch_dir):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        model.train()
        before_mode = model.training
        before_dtype = next(model.parameters()).dtype
        before_device = next(model.parameters()).device
        save_material_checkpoint(scratch_dir / "best.pt", model, _metadata())
        assert model.training == before_mode
        assert next(model.parameters()).dtype == before_dtype
        assert next(model.parameters()).device == before_device
