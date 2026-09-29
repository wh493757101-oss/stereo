"""Material-fusion model tests: variants, head, freeze stages, parameters."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.material_fusion import (
    CLASS_NAMES,
    HEAD_DROPOUT,
    HEAD_PROJ_DIM,
    VARIANTS,
    MultiScaleClassificationHead,
    build_material_model,
    validate_variant,
)


class TestVariants:
    def test_variant_names_are_fixed(self):
        assert VARIANTS == ("gray", "gray_gray_valid", "gray_valid", "gray_polar")

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError, match="polar_fusion"):
            validate_variant("polar_fusion")
        with pytest.raises(ValueError):
            validate_variant(None)

    def test_class_order_is_fixed(self):
        assert CLASS_NAMES == (
            "metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"
        )


class TestHead:
    def test_head_projects_each_level_to_128_and_outputs_4(self):
        head = MultiScaleClassificationHead(in_dims=(32, 64, 160, 256), num_classes=4)
        features = [
            torch.rand(2, 32, 16, 16),
            torch.rand(2, 64, 8, 8),
            torch.rand(2, 160, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        logits = head(features)
        assert logits.shape == (2, 4)
        assert HEAD_PROJ_DIM == 128
        assert HEAD_DROPOUT == 0.1
        # 4 levels * 128 projected channels -> 512-dim pooled vector.
        assert head.norm.normalized_shape == (512,)
        assert head.fc.in_features == 512

    def test_head_rejects_wrong_level_count(self):
        head = MultiScaleClassificationHead(in_dims=(32, 64, 160, 256), num_classes=4)
        with pytest.raises(ValueError):
            head([torch.rand(1, 32, 8, 8)])


class TestModelForward:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_forward_returns_raw_logits(self, variant):
        model = build_material_model(variant, imgsz=64, drop_path_rate=0.0).eval()
        rgb = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            if variant == "gray":
                logits = model(rgb)
            else:
                logits = model(rgb, torch.rand(2, 3, 64, 64))
        assert logits.shape == (2, len(CLASS_NAMES))
        assert logits.dtype == torch.float32

    def test_dual_variant_requires_aux(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        with pytest.raises(ValueError, match="aux"):
            model(torch.rand(1, 3, 64, 64))

    def test_gray_variant_rejects_aux(self):
        model = build_material_model("gray", imgsz=64, drop_path_rate=0.0)
        with pytest.raises(ValueError, match="aux"):
            model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))

    def test_dual_model_has_four_fused_levels(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0).eval()
        with torch.no_grad():
            features = model.encoder(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
        assert [f.shape[1] for f in features] == [32, 64, 160, 256]

    def test_single_stream_model_uses_single_encoder(self):
        from models.cmx.dual_segformer import MiTEncoder

        model = build_material_model("gray", imgsz=64, drop_path_rate=0.0)
        assert isinstance(model.encoder, MiTEncoder)

    def test_param_counts(self):
        single = build_material_model("gray", imgsz=64, drop_path_rate=0.0)
        dual = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        head_params = sum(p.numel() for p in single.head.parameters())
        assert sum(p.numel() for p in single.parameters()) == 3_319_392 + head_params
        assert sum(p.numel() for p in dual.parameters()) > sum(
            p.numel() for p in single.parameters()
        )


class TestParameterGroups:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_groups_cover_every_parameter_exactly_once(self, variant):
        model = build_material_model(variant, imgsz=64, drop_path_rate=0.0)
        groups = model.parameter_groups()
        assert set(groups) == {"backbone", "fusion", "head"}
        covered = [id(p) for params in groups.values() for p in params]
        all_params = [id(p) for p in model.parameters()]
        assert sorted(covered) == sorted(all_params)
        assert len(covered) == len(set(covered))

    def test_single_stream_has_no_fusion_group_members(self):
        model = build_material_model("gray", imgsz=64, drop_path_rate=0.0)
        assert model.parameter_groups()["fusion"] == []
        assert model.parameter_groups()["backbone"]

    def test_dual_variants_have_fusion_members(self):
        for variant in ("gray_gray_valid", "gray_valid", "gray_polar"):
            model = build_material_model(variant, imgsz=64, drop_path_rate=0.0)
            assert model.parameter_groups()["fusion"], variant

    def test_backbone_excludes_frm_ffm(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        groups = model.parameter_groups()
        backbone_ids = {id(p) for p in groups["backbone"]}
        backbone_names = [
            name for name, param in model.named_parameters() if id(param) in backbone_ids
        ]
        assert backbone_names
        assert not any("FRMs" in name or "FFMs" in name for name in backbone_names)
        fusion_ids = {id(p) for p in groups["fusion"]}
        fusion_names = [
            name for name, param in model.named_parameters() if id(param) in fusion_ids
        ]
        assert fusion_names and all(
            "FRMs" in name or "FFMs" in name for name in fusion_names
        )


class TestStageProtocol:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_warmup_freezes_backbone_only(self, variant):
        model = build_material_model(variant, imgsz=64, drop_path_rate=0.1)
        model.set_stage("warmup")
        groups = model.parameter_groups()
        assert all(not p.requires_grad for p in groups["backbone"])
        assert all(p.requires_grad for p in groups["fusion"])
        assert all(p.requires_grad for p in groups["head"])
        # Backbone modules run in eval (DropPath identity); fusion and head
        # stay in train so their BatchNorm statistics keep updating.
        assert not model.encoder.patch_embed1.training
        assert model.head.training
        if variant != "gray":
            assert model.encoder.FRMs[0].training
            assert model.encoder.FFMs[0].training

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_finetune_opens_everything(self, variant):
        model = build_material_model(variant, imgsz=64, drop_path_rate=0.0)
        model.set_stage("warmup")
        model.set_stage("finetune")
        groups = model.parameter_groups()
        assert all(p.requires_grad for p in groups["backbone"])
        assert all(p.requires_grad for p in groups["fusion"])
        assert all(p.requires_grad for p in groups["head"])
        assert model.training

    def test_unknown_stage_rejected(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        with pytest.raises(ValueError, match="stage"):
            model.set_stage("freeze")

    def test_warmup_gradients_flow_to_fusion_and_head_but_not_backbone(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        model.set_stage("warmup")
        logits = model(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
        logits.sum().backward()
        groups = model.parameter_groups()
        assert all(p.grad is None for p in groups["backbone"])
        for param in groups["fusion"] + groups["head"]:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_warmup_updates_fusion_and_head_params_only(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        model.set_stage("warmup")
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        groups = model.parameter_groups()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        logits = model(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        optimizer.step()
        after = model.state_dict()
        # Backbone parameters are untouched.
        backbone_names = {
            name for name, param in model.named_parameters()
            if id(param) in {id(p) for p in groups["backbone"]}
        }
        for name in backbone_names:
            assert torch.equal(before[name], after[name]), name
        trained_names = {
            name for name, param in model.named_parameters()
            if id(param) in {id(q) for q in groups["fusion"] + groups["head"]}
        }
        assert any(
            not torch.equal(before[name], after[name]) for name in trained_names
        )

    def test_warmup_updates_fusion_batch_norm_buffers(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        model.set_stage("warmup")
        fusion_buffer_names = [
            name for name, _ in model.named_buffers() if "FRMs" in name or "FFMs" in name
        ]
        assert fusion_buffer_names, "fusion layers must carry BatchNorm buffers"
        before = {name: model.state_dict()[name].clone() for name in fusion_buffer_names}
        with torch.no_grad():
            model(torch.rand(4, 3, 64, 64), torch.rand(4, 3, 64, 64))
        after = model.state_dict()
        changed = [
            name for name in fusion_buffer_names
            if not torch.equal(before[name], after[name])
        ]
        assert changed, "fusion BatchNorm buffers must update during warmup"

    def test_backbone_buffers_stay_unchanged_during_warmup(self):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        model.set_stage("warmup")
        backbone_buffer_names = [
            name for name, _ in model.named_buffers()
            if "FRMs" not in name and "FFMs" not in name
        ]
        before = {name: model.state_dict()[name].clone() for name in backbone_buffer_names}
        with torch.no_grad():
            model(torch.rand(4, 3, 64, 64), torch.rand(4, 3, 64, 64))
        after = model.state_dict()
        for name in backbone_buffer_names:
            assert torch.equal(before[name], after[name]), name
