"""CMX MiT encoder port (dual-stream RGBX transformer + single-stream MiT).

Vendored from huaaaliu/RGBX_Semantic_Segmentation at commit
e251d860aebc2f583a6c4919877e6bebe7f1aff3 (``models/encoders/dual_segformer.py``,
MIT License, Copyright (c) 2022 Huayao Liu). Changes against upstream:

- ``timm`` imports replaced by the local compatibility shim
  ``models.cmx.timm_compat`` (same three functions, no dependency).
- ``engine.logger`` import removed (upstream training framework).
- ``load_dualpath_model`` replaced by :func:`load_mit_b0_pretrained` /
  :func:`load_mit_b0_checkpoint`, which verify load coverage strictly
  instead of the upstream ``strict=False`` silent partial load.
- ``mit_b1`` .. ``mit_b5`` omitted (only MiT-B0 is used by this
  experiment; the parameter tables are otherwise unchanged).
- Added :class:`MiTEncoder` (single-stream backbone with the same blocks
  and parameter names) and :func:`convert_hf_state_dict` (nvidia/mit-b0
  Hugging Face key naming -> original naming).

The dual-stream module/parameter names (``patch_embed*``, ``extra_patch_embed*``,
``block*``, ``extra_block*``, ``norm*``, ``extra_norm*``, ``FRMs``, ``FFMs``)
and every forward computation are identical to upstream.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from models.cmx.net_utils import FeatureFusionModule as FFM
from models.cmx.net_utils import FeatureRectifyModule as FRM
from models.cmx.timm_compat import DropPath, to_2tuple, trunc_normal_

UPSTREAM_COMMIT = "e251d860aebc2f583a6c4919877e6bebe7f1aff3"
UPSTREAM_SOURCE = (
    "https://github.com/huaaaliu/RGBX_Semantic_Segmentation/blob/"
    f"{UPSTREAM_COMMIT}/models/encoders/dual_segformer.py"
)

# MiT-B0 specification exactly as upstream ``class mit_b0``.
MIT_B0_SPEC = {
    "patch_size": 4,
    "embed_dims": [32, 64, 160, 256],
    "num_heads": [1, 2, 5, 8],
    "mlp_ratios": [4, 4, 4, 4],
    "qkv_bias": True,
    "depths": [2, 2, 2, 2],
    "sr_ratios": [8, 4, 2, 1],
    "drop_rate": 0.0,
    "drop_path_rate": 0.1,
}


class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()  # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # B C H W -> B N C

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block:
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)

        return x, H, W


class MiTEncoder(nn.Module):
    """Single-stream MiT-B0 backbone returning the four stage features.

    Not present upstream (CMX only ships the dual-stream transformer); the
    blocks, names and stage outputs mirror the single path of
    :class:`RGBXTransformer` so the official MiT-B0 backbone checkpoint
    loads into it one-to-one.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dims=None, num_heads=None, mlp_ratios=None, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, depths=None, sr_ratios=None):
        super().__init__()
        embed_dims = embed_dims or MIT_B0_SPEC["embed_dims"]
        num_heads = num_heads or MIT_B0_SPEC["num_heads"]
        mlp_ratios = mlp_ratios or MIT_B0_SPEC["mlp_ratios"]
        depths = depths or MIT_B0_SPEC["depths"]
        sr_ratios = sr_ratios or MIT_B0_SPEC["sr_ratios"]

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4,
                                              in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
            norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                                    embed_dim=embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                                    embed_dim=embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                                    embed_dim=embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                                    embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.extra_block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.extra_block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + 1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])

        cur += depths[1]

        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.extra_block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])

        cur += depths[2]

        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.extra_block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        cur += depths[3]

        self.FRMs = nn.ModuleList([
            FRM(dim=embed_dims[0], reduction=1),
            FRM(dim=embed_dims[1], reduction=1),
            FRM(dim=embed_dims[2], reduction=1),
            FRM(dim=embed_dims[3], reduction=1)])

        self.FFMs = nn.ModuleList([
            FFM(dim=embed_dims[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
            FFM(dim=embed_dims[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
            FFM(dim=embed_dims[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
            FFM(dim=embed_dims[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def frm_ffm_parameter_count(self) -> int:
        """Parameters of the newly initialized FRM/FFM fusion modules."""
        return sum(
            p.numel() for name, p in self.named_parameters()
            if name.startswith("FRMs.") or name.startswith("FFMs.")
        )

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x N x H x W
        """
        B = x_rgb.shape[0]
        outs = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        # B H*W/16 C
        x_e, _, _ = self.extra_patch_embed1(x_e)
        for i, blk in enumerate(self.block1):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block1):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm1(x_rgb)
        x_e = self.extra_norm1(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[0](x_rgb, x_e)
        x_fused = self.FFMs[0](x_rgb, x_e)
        outs.append(x_fused)

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e, _, _ = self.extra_patch_embed2(x_e)
        for i, blk in enumerate(self.block2):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block2):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm2(x_rgb)
        x_e = self.extra_norm2(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[1](x_rgb, x_e)
        x_fused = self.FFMs[1](x_rgb, x_e)
        outs.append(x_fused)

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e, _, _ = self.extra_patch_embed3(x_e)
        for i, blk in enumerate(self.block3):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block3):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm3(x_rgb)
        x_e = self.extra_norm3(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[2](x_rgb, x_e)
        x_fused = self.FFMs[2](x_rgb, x_e)
        outs.append(x_fused)

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e, _, _ = self.extra_patch_embed4(x_e)
        for i, blk in enumerate(self.block4):
            x_rgb = blk(x_rgb, H, W)
        for i, blk in enumerate(self.extra_block4):
            x_e = blk(x_e, H, W)
        x_rgb = self.norm4(x_rgb)
        x_e = self.extra_norm4(x_e)

        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[3](x_rgb, x_e)
        x_fused = self.FFMs[3](x_rgb, x_e)
        outs.append(x_fused)

        return outs

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out


class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, drop_path_rate=MIT_B0_SPEC["drop_path_rate"], **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=drop_path_rate)


# ---------------------------------------------------------------------------
# Checkpoint loading (replaces upstream ``load_dualpath_model``)

_HF_ENCODER_PREFIX = "segformer.encoder."
_BACKBONE_PREFIXES = ("patch_embed", "block", "norm")

# HF block-tail prefixes -> original naming (explicit, prefix-anchored).
_HF_BLOCK_TAIL_MAP = (
    ("layer_norm_1.", "norm1."),
    ("layer_norm_2.", "norm2."),
    ("attention.self.query.", "attn.q."),
    ("attention.output.dense.", "attn.proj."),
    ("attention.self.sr.", "attn.sr."),
    ("attention.self.layer_norm.", "attn.norm."),
    ("mlp.dense1.", "mlp.fc1."),
    ("mlp.dense2.", "mlp.fc2."),
)


def _convert_hf_block_tail(tail: str) -> str:
    for hf_prefix, original_prefix in _HF_BLOCK_TAIL_MAP:
        if tail.startswith(hf_prefix):
            return original_prefix + tail[len(hf_prefix):]
    return tail


def convert_hf_state_dict(state_dict: dict) -> dict:
    """Normalize a MiT backbone state dict to the original naming.

    Accepts the nvidia/mit-b0 Hugging Face key naming
    (``segformer.encoder.*`` with ``attention.self.query`` etc.) as well as
    already-original keys (``patch_embed1.proj.weight``, ...). The HF
    checkpoint stores q/k/v as three separate projections while the original
    (and the CMX port) use one ``q`` and one fused ``kv`` projection, so
    k/v are concatenated. The 1000-class ``classifier.*`` head of the HF
    checkpoint is dropped. Any unrecognized key is refused.
    """
    converted: dict = {}
    pending_kv: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if key.startswith("classifier."):
            continue
        if not key.startswith(_HF_ENCODER_PREFIX):
            converted[key] = value
            continue
        rest = key[len(_HF_ENCODER_PREFIX):]
        if rest.startswith("patch_embeddings."):
            _, index, tail = rest.split(".", 2)
            tail = tail.replace("layer_norm.", "norm.")
            converted[f"patch_embed{int(index) + 1}.{tail}"] = value
        elif rest.startswith("block."):
            _, index, layer, tail = rest.split(".", 3)
            block_prefix = f"block{int(index) + 1}.{layer}"
            if tail.startswith(("attention.self.key.", "attention.self.value.")):
                leaf = tail.split(".")[-1]
                side = "k" if tail.startswith("attention.self.key.") else "v"
                pending_kv.setdefault((block_prefix, leaf), {})[side] = value
                continue
            converted[f"{block_prefix}.{_convert_hf_block_tail(tail)}"] = value
        elif rest.startswith("layer_norm."):
            _, index, tail = rest.split(".", 2)
            converted[f"norm{int(index) + 1}.{tail}"] = value
        else:
            raise ValueError(f"unrecognized Hugging Face encoder key: {key}")

    for (block_prefix, leaf), sides in pending_kv.items():
        if set(sides) != {"k", "v"}:
            raise ValueError(
                f"incomplete attention key/value pair for {block_prefix}.{leaf}: "
                f"found {sorted(sides)}"
            )
        converted[f"{block_prefix}.attn.kv.{leaf}"] = torch.cat(
            [sides["k"], sides["v"]], dim=0
        )
    return converted


def _backbone_targets(model: nn.Module) -> dict[str, str]:
    """Map model parameter name -> checkpoint key for backbone parameters."""
    targets: dict[str, str] = {}
    for name in model.state_dict():
        if isinstance(model, RGBXTransformer) and name.startswith(("FRMs.", "FFMs.")):
            continue
        if name.startswith("extra_"):
            targets[name] = name[len("extra_"):]
        else:
            targets[name] = name
    return targets


def load_mit_b0_pretrained(model: nn.Module, state_dict: dict) -> dict:
    """Strictly load a MiT-B0 backbone checkpoint into a model.

    Every backbone parameter of ``model`` must be covered by the checkpoint
    with an exactly matching shape; anything else raises. For the dual-stream
    :class:`RGBXTransformer` both streams are filled from the same single
    backbone (upstream behavior); FRM/FFM parameters stay randomly
    initialized and are reported.

    Returns a report dict: ``loaded``, ``missing``, ``unexpected``,
    ``dropped``, ``random_init``.
    """
    raw = state_dict.get("state_dict", state_dict) if isinstance(state_dict, dict) else state_dict
    converted = convert_hf_state_dict(raw)
    dropped = sorted(key for key in raw if key.startswith("classifier."))

    targets = _backbone_targets(model)
    model_state = model.state_dict()
    missing = []
    shape_mismatch = []
    for name, checkpoint_key in targets.items():
        if checkpoint_key not in converted:
            missing.append(name)
            continue
        if tuple(converted[checkpoint_key].shape) != tuple(model_state[name].shape):
            shape_mismatch.append(
                (name, tuple(model_state[name].shape), tuple(converted[checkpoint_key].shape))
            )
    if missing:
        raise ValueError(
            f"pretrained checkpoint is missing {len(missing)} backbone tensors "
            f"(e.g. {missing[:5]}); refusing a partial/silent load"
        )
    if shape_mismatch:
        raise ValueError(
            f"pretrained checkpoint shape mismatch for {len(shape_mismatch)} tensors "
            f"(e.g. {shape_mismatch[:3]}); wrong backbone variant or corrupted file"
        )

    used = set()
    load_dict = {}
    for name, checkpoint_key in targets.items():
        load_dict[name] = converted[checkpoint_key]
        used.add(checkpoint_key)
    model.load_state_dict(load_dict, strict=False)

    unexpected = sorted(
        key for key in converted
        if key not in used and key.startswith(_BACKBONE_PREFIXES)
    )
    random_init = sorted(name for name in model_state if name not in targets)
    return {
        # Number of model parameter tensors filled (dual-stream models fill
        # both streams from the same checkpoint, so this is 2x the
        # checkpoint's backbone tensor count).
        "loaded": len(load_dict),
        "checkpoint_tensors_used": len(used),
        "missing": missing,
        "unexpected": unexpected,
        "dropped": dropped,
        "random_init": random_init,
    }


def load_mit_b0_checkpoint(model: nn.Module, path: str | Path,
                           expected_sha256: str = "") -> dict:
    """Load a MiT-B0 checkpoint file with optional SHA256 verification.

    A missing file raises ``FileNotFoundError`` (never a silent random
    initialization); a SHA256 mismatch raises ``ValueError``.
    """
    import hashlib

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pretrained MiT-B0 checkpoint {path} is missing; download the "
            "official nvidia/mit-b0 weights and record their SHA256 "
            "(no automatic download, no random initialization fallback)"
        )
    if expected_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"pretrained checkpoint {path} sha256 {digest} does not match "
                f"the recorded {expected_sha256}"
            )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    report = load_mit_b0_pretrained(model, state_dict)
    report["path"] = str(path)
    return report
