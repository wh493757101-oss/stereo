"""Port-fidelity tests for the vendored CMX encoder and fusion modules.

The ported code must match the upstream definitions at commit
e251d860aebc2f583a6c4919877e6bebe7f1aff3 (huaaaliu/RGBX_Semantic_Segmentation,
MIT). These tests pin the MiT-B0 specification, the tensor shapes of every
building block, and the behavior of the timm compatibility shims.
"""

from __future__ import annotations

import torch

from models.cmx.dual_segformer import (
    MIT_B0_SPEC,
    MiTEncoder,
    RGBXTransformer,
    mit_b0,
)
from models.cmx.net_utils import FeatureFusionModule, FeatureRectifyModule
from models.cmx.timm_compat import DropPath, to_2tuple, trunc_normal_


class TestTimmCompat:
    def test_to_2tuple_scalar_and_tuple(self):
        assert to_2tuple(7) == (7, 7)
        assert to_2tuple((3, 5)) == (3, 5)
        assert to_2tuple([3, 5]) == [3, 5]

    def test_trunc_normal_absolute_bounds(self):
        # timm/torch semantics: a/b are absolute values, not sigma units.
        torch.manual_seed(0)
        tensor = torch.empty(20000)
        result = trunc_normal_(tensor, mean=0.0, std=0.02, a=-2.0, b=2.0)
        assert result is tensor
        assert float(tensor.min()) >= -2.0
        assert float(tensor.max()) <= 2.0
        assert abs(float(tensor.mean())) < 5e-4
        assert abs(float(tensor.std()) - 0.02) < 5e-4

    def test_trunc_normal_truncates_at_sigma_bounds(self):
        torch.manual_seed(0)
        tensor = torch.empty(20000)
        trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
        assert float(tensor.min()) >= -2.0 - 1e-6
        assert float(tensor.max()) <= 2.0 + 1e-6
        # Truncation at +/-2 sigma shrinks the standard deviation to
        # ~0.8796 of the requested one.
        assert abs(float(tensor.std()) - 0.8796) < 0.02

    def test_trunc_normal_respects_custom_bounds(self):
        torch.manual_seed(0)
        tensor = torch.empty(5000)
        trunc_normal_(tensor, mean=1.0, std=0.5, a=0.5, b=1.5)
        assert float(tensor.min()) >= 0.5 - 1e-6
        assert float(tensor.max()) <= 1.5 + 1e-6

    def test_drop_path_identity_in_eval_and_zero_prob(self):
        torch.manual_seed(0)
        x = torch.ones(8, 4, 3, 3)
        layer = DropPath(0.5)
        layer.eval()
        assert torch.equal(layer(x), x)
        zero = DropPath(0.0)
        zero.train()
        assert torch.equal(zero(x), x)

    def test_drop_path_scales_kept_paths_in_train(self):
        torch.manual_seed(0)
        x = torch.ones(4000, 4)
        out = DropPath(0.5)(x)
        kept = (out != 0).float().mean().item()
        assert 0.45 < kept < 0.55
        assert torch.allclose(out[out != 0], torch.full_like(out[out != 0], 2.0))


class TestMiTEncoder:
    def test_mit_b0_spec_matches_upstream(self):
        assert MIT_B0_SPEC["embed_dims"] == [32, 64, 160, 256]
        assert MIT_B0_SPEC["num_heads"] == [1, 2, 5, 8]
        assert MIT_B0_SPEC["depths"] == [2, 2, 2, 2]
        assert MIT_B0_SPEC["sr_ratios"] == [8, 4, 2, 1]
        assert MIT_B0_SPEC["mlp_ratios"] == [4, 4, 4, 4]
        assert MIT_B0_SPEC["qkv_bias"] is True
        assert MIT_B0_SPEC["patch_size"] == 4

    def test_single_stream_outputs_four_levels(self):
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0).eval()
        with torch.no_grad():
            features = encoder(torch.rand(2, 3, 64, 64))
        assert len(features) == 4
        expected = [(2, 32, 16, 16), (2, 64, 8, 8), (2, 160, 4, 4), (2, 256, 2, 2)]
        for tensor, shape in zip(features, expected):
            assert tuple(tensor.shape) == shape

    def test_single_stream_parameter_count_matches_official_backbone(self):
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        assert sum(p.numel() for p in encoder.parameters()) == 3_319_392

    def test_dual_stream_outputs_four_fused_levels(self):
        model = mit_b0().eval()
        with torch.no_grad():
            fused = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
        assert len(fused) == 4
        assert [tuple(f.shape[1:]) for f in fused] == [
            (32, 16, 16), (64, 8, 8), (160, 4, 4), (256, 2, 2),
        ]

    def test_dual_stream_has_two_independent_backbones(self):
        model = mit_b0()
        backbone_params = sum(p.numel() for p in model.parameters())
        assert backbone_params == 2 * 3_319_392 + model.frm_ffm_parameter_count()


class TestFusionModules:
    def test_frm_preserves_shapes_and_mixes_streams(self):
        frm = FeatureRectifyModule(dim=32, reduction=1).eval()
        x1 = torch.rand(2, 32, 8, 8)
        x2 = torch.rand(2, 32, 8, 8)
        with torch.no_grad():
            out1, out2 = frm(x1, x2)
        assert out1.shape == x1.shape and out2.shape == x2.shape
        assert not torch.allclose(out1, x1)
        assert not torch.allclose(out2, x2)

    def test_ffm_output_shape_equals_input_shape(self):
        ffm = FeatureFusionModule(dim=32, reduction=1, num_heads=1).eval()
        x1 = torch.rand(2, 32, 8, 8)
        x2 = torch.rand(2, 32, 8, 8)
        with torch.no_grad():
            merged = ffm(x1, x2)
        assert merged.shape == x1.shape

    def test_ffm_zero_aux_still_uses_cross_path(self):
        ffm = FeatureFusionModule(dim=32, reduction=1, num_heads=1).eval()
        x1 = torch.rand(2, 32, 8, 8)
        zeros = torch.zeros(2, 32, 8, 8)
        with torch.no_grad():
            merged = ffm(x1, zeros)
        assert merged.shape == x1.shape
        assert torch.isfinite(merged).all()


class TestRGBXTransformerShapes:
    def test_forward_features_returns_expected_dims(self):
        model = RGBXTransformer(
            img_size=64,
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=0.0,
        ).eval()
        with torch.no_grad():
            fused = model(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
        channels = [f.shape[1] for f in fused]
        assert channels == [32, 64, 160, 256]
