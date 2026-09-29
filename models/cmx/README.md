# models/cmx — vendored CMX code (provenance and modifications)

This directory contains code ported from an upstream open-source project.
**It is not original work and it is not unmodified upstream code** — the
file headers list the exact changes per file.

## Upstream

- Repository: `https://github.com/huaaaliu/RGBX_Semantic_Segmentation`
- Pinned commit: `e251d860aebc2f583a6c4919877e6bebe7f1aff3`
- License: MIT — full text in `LICENSE` (verbatim upstream file, SHA256
  `a3fb69f7d2d7ab44ce80bba7f8c3a61f3c8a2775a2baac4bef815a60c4d8ba5e`)
- Paper: CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with
  Transformers, `https://arxiv.org/abs/2203.04838`
- Upstream source files and their SHA256 at the pinned commit:
  - `models/encoders/dual_segformer.py` —
    `6d8994d0044929c97d318708223bf6fea48efce4575aa986c72e415ee017e755`
  - `models/net_utils.py` —
    `ada5e36e14d83c35d9230618c4eb84352a76ec2275a086a624b280e7638d8473`

## Local files

| File | Origin | Modifications |
|---|---|---|
| `net_utils.py` | upstream `models/net_utils.py` | `timm` import replaced by `timm_compat`; otherwise verbatim |
| `dual_segformer.py` | upstream `models/encoders/dual_segformer.py` | `timm`/`engine.logger` imports replaced or removed; strict pretrained loaders replace `load_dualpath_model`; `mit_b1`–`mit_b5` omitted; `MiTEncoder` (single-stream backbone), `convert_hf_state_dict`, `MIT_B0_SPEC` added |
| `timm_compat.py` | local | New file: faithful reimplementations of `to_2tuple`, `trunc_normal_`, `DropPath` (timm is Apache-2.0) so the port needs no `timm` dependency |
| `LICENSE` | upstream `LICENSE` | Verbatim, unmodified |

The dual-stream module names, the MiT-B0 parameter table and every forward
computation are unchanged from upstream.
