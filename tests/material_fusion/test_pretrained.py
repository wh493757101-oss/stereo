"""Pretrained MiT-B0 loading: real-file coverage and wrong-spec rejection.

The real weights live at ``weights/pretrained/mit_b0.pth`` (SHA256
4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc, the
official nvidia/mit-b0 release). Tests that need the file skip with an
explicit reason when it is absent; the synthetic-path tests always run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.cmx.dual_segformer import (
    MiTEncoder,
    convert_hf_state_dict,
    load_mit_b0_pretrained,
    mit_b0,
)

from .conftest import build_synthetic_mit_state_dict, write_synthetic_pretrained

REAL_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "pretrained" / "mit_b0.pth"
REAL_SHA256 = "4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc"

requires_real_weights = pytest.mark.skipif(
    not REAL_WEIGHTS.is_file(),
    reason=(
        f"official MiT-B0 weights not present at {REAL_WEIGHTS}; download from "
        "https://huggingface.co/nvidia/mit-b0 and verify SHA256 "
        f"{REAL_SHA256}"
    ),
)


class TestSyntheticLoading:
    def test_single_stream_full_coverage(self):
        state = build_synthetic_mit_state_dict(seed=1)
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        report = load_mit_b0_pretrained(encoder, state)
        # 176 backbone tensors: the original naming fuses k/v into one kv
        # projection (the HF checkpoint splits them into two).
        assert report["loaded"] == 176
        assert report["checkpoint_tensors_used"] == 176
        assert report["missing"] == []
        for key, value in state.items():
            assert torch.equal(encoder.state_dict()[key], value), key

    def test_dual_stream_fills_both_paths_identically(self):
        state = build_synthetic_mit_state_dict(seed=2)
        model = mit_b0()
        report = load_mit_b0_pretrained(model, state)
        assert report["loaded"] == 352
        assert report["checkpoint_tensors_used"] == 176
        params = dict(model.named_parameters())

        def extra_name(key: str) -> str:
            for prefix in ("patch_embed", "block", "norm"):
                if key.startswith(prefix):
                    return "extra_" + key
            return key

        for key, value in state.items():
            if key.endswith("weight") or key.endswith("bias"):
                assert torch.equal(params[key], value), key
                assert torch.equal(params[extra_name(key)], value), extra_name(key)

    def test_frm_ffm_and_head_stay_randomly_initialized(self):
        state = build_synthetic_mit_state_dict(seed=3)
        model = mit_b0()
        report = load_mit_b0_pretrained(model, state)
        assert report["random_init"], "FRM/FFM parameters must be reported"
        assert all(
            "FRMs" in key or "FFMs" in key for key in report["random_init"]
        )

    def test_missing_key_is_rejected(self):
        state = build_synthetic_mit_state_dict(seed=4)
        del state["block1.0.attn.q.weight"]
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        with pytest.raises(ValueError, match="block1.0.attn.q.weight"):
            load_mit_b0_pretrained(encoder, state)

    def test_wrong_shape_is_rejected(self):
        state = build_synthetic_mit_state_dict(seed=5)
        state["patch_embed1.proj.weight"] = torch.zeros(64, 3, 7, 7)
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        with pytest.raises(ValueError, match="patch_embed1.proj.weight"):
            load_mit_b0_pretrained(encoder, state)

    def test_foreign_checkpoint_is_rejected(self):
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        with pytest.raises(ValueError):
            load_mit_b0_pretrained(encoder, {"backbone.0.weight": torch.zeros(4)})

    def test_missing_file_is_not_silent_random_init(self, scratch_dir):
        from models.cmx.dual_segformer import load_mit_b0_checkpoint

        missing = scratch_dir / "nope.pth"
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        with pytest.raises(FileNotFoundError):
            load_mit_b0_checkpoint(encoder, missing)

    def test_synthetic_checkpoint_file_roundtrip(self, scratch_dir):
        from models.cmx.dual_segformer import load_mit_b0_checkpoint

        path = write_synthetic_pretrained(scratch_dir / "syn.pth", seed=6)
        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        report = load_mit_b0_checkpoint(encoder, path)
        assert report["loaded"] == 176


class TestHfNamingConversion:
    def test_hf_keys_convert_to_original_naming(self):
        hf_state = {
            "segformer.encoder.patch_embeddings.0.proj.weight": torch.zeros(32, 3, 7, 7),
            "segformer.encoder.patch_embeddings.0.layer_norm.bias": torch.zeros(32),
            "segformer.encoder.block.0.0.layer_norm_1.weight": torch.zeros(32),
            "segformer.encoder.block.0.0.attention.self.query.weight": torch.zeros(32, 32),
            "segformer.encoder.block.0.0.attention.self.key.weight": torch.zeros(32, 32),
            "segformer.encoder.block.0.0.attention.self.value.weight": torch.zeros(32, 32),
            "segformer.encoder.block.0.0.attention.self.sr.bias": torch.zeros(32),
            "segformer.encoder.block.0.0.attention.self.layer_norm.weight": torch.zeros(32),
            "segformer.encoder.block.0.0.attention.output.dense.bias": torch.zeros(32),
            "segformer.encoder.block.0.0.layer_norm_2.weight": torch.zeros(32),
            "segformer.encoder.block.0.0.mlp.dense1.weight": torch.zeros(128, 32),
            "segformer.encoder.block.0.0.mlp.dwconv.dwconv.bias": torch.zeros(128),
            "segformer.encoder.block.0.0.mlp.dense2.bias": torch.zeros(32),
            "segformer.encoder.layer_norm.0.weight": torch.zeros(32),
        }
        converted = convert_hf_state_dict(hf_state)
        assert set(converted) == {
            "patch_embed1.proj.weight",
            "patch_embed1.norm.bias",
            "block1.0.norm1.weight",
            "block1.0.attn.q.weight",
            "block1.0.attn.kv.weight",
            "block1.0.attn.sr.bias",
            "block1.0.attn.norm.weight",
            "block1.0.attn.proj.bias",
            "block1.0.norm2.weight",
            "block1.0.mlp.fc1.weight",
            "block1.0.mlp.dwconv.dwconv.bias",
            "block1.0.mlp.fc2.bias",
            "norm1.weight",
        }
        # k/v concatenate into the fused kv projection.
        assert converted["block1.0.attn.kv.weight"].shape == (64, 32)

    def test_original_naming_passes_through(self):
        state = build_synthetic_mit_state_dict(seed=8)
        converted = convert_hf_state_dict(state)
        assert set(converted) == set(state)
        assert torch.equal(converted["patch_embed2.proj.bias"], state["patch_embed2.proj.bias"])

    def test_hf_classifier_keys_are_dropped(self):
        state = build_synthetic_mit_state_dict(seed=9)
        hf_state = dict(state)
        hf_state["classifier.weight"] = torch.zeros(1000, 256)
        hf_state["classifier.bias"] = torch.zeros(1000)
        converted = convert_hf_state_dict(hf_state)
        assert "classifier.weight" not in converted
        assert "classifier.bias" not in converted
        assert len(converted) == 176


@requires_real_weights
class TestRealPretrainedWeights:
    def test_file_sha256_matches_recorded_release(self):
        import hashlib

        digest = hashlib.sha256(REAL_WEIGHTS.read_bytes()).hexdigest()
        assert digest == REAL_SHA256

    def test_single_stream_full_coverage_from_real_file(self):
        from models.cmx.dual_segformer import load_mit_b0_checkpoint

        encoder = MiTEncoder(img_size=64, drop_path_rate=0.0)
        report = load_mit_b0_checkpoint(encoder, REAL_WEIGHTS)
        assert report["loaded"] == 176
        assert report["checkpoint_tensors_used"] == 176
        assert report["missing"] == []
        assert report["dropped"] == ["classifier.bias", "classifier.weight"]

    def test_dual_stream_from_real_file_fills_both_paths(self):
        from models.cmx.dual_segformer import load_mit_b0_checkpoint

        model = mit_b0()
        report = load_mit_b0_checkpoint(model, REAL_WEIGHTS)
        assert report["loaded"] == 352
        params = dict(model.named_parameters())
        assert torch.equal(
            params["patch_embed1.proj.weight"], params["extra_patch_embed1.proj.weight"]
        )
        assert torch.equal(
            params["block4.1.mlp.fc2.bias"], params["extra_block4.1.mlp.fc2.bias"]
        )
