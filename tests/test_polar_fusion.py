"""Tests for models.polar_fusion (structure, gating, checkpointing, dataset)."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.fusion_dataset import (
    QUALITY_VECTOR_KEYS,
    QUALITY_VECTOR_LENGTH,
    build_quality_vector,
    save_fusion_sample,
)
from models.polar_fusion import (
    DEFAULT_BASE_MODEL,
    FUSION_VERSION,
    FusionCheckpointMetadata,
    FusionClsDataset,
    GrayBackboneAdapter,
    PolarDeltaNet,
    PolarFusionModel,
    PolarGateNet,
    build_gray_class_perm,
    class_names_from_manifest,
    fusion_metadata,
    load_fusion_checkpoint,
    read_fusion_metadata,
    read_manifest_split,
    save_fusion_checkpoint,
)


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="polar_fusion_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class TinyBackbone(nn.Module):
    """Deterministic stand-in for the YOLO classification backbone."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes)
        )
        torch.manual_seed(7)
        for param in self.parameters():
            nn.init.normal_(param, std=0.01)

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        return self.net(gray)


def make_inputs(batch=4, num_classes=4, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    gray = torch.rand(batch, 3, size, size, generator=g)
    polar = torch.rand(batch, 3, size, size, generator=g)
    quality = torch.rand(batch, QUALITY_VECTOR_LENGTH, generator=g)
    return gray, polar, quality


class TestGrayBackboneAdapter:
    def test_eval_tuple_returns_raw_logits(self):
        """Real Ultralytics classification models return (probs, logits) in
        eval mode; the adapter must expose the raw logits."""

        class TupleBackbone(nn.Module):
            def forward(self, gray):
                logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]])
                return (logits.softmax(-1), logits)

        adapter = GrayBackboneAdapter(TupleBackbone().eval())
        out = adapter(torch.zeros(1, 3, 8, 8))
        assert isinstance(out, torch.Tensor)
        torch.testing.assert_close(out, torch.tensor([[1.0, 2.0, 0.5, -1.0]]))

    def test_train_mode_tensor_passthrough(self):
        class TensorBackbone(nn.Module):
            def forward(self, gray):
                return torch.ones(1, 4)

        adapter = GrayBackboneAdapter(TensorBackbone())
        out = adapter(torch.zeros(1, 3, 8, 8))
        torch.testing.assert_close(out, torch.ones(1, 4))

    def test_perm_reorders_columns_to_target_class_order(self):
        class FixedBackbone(nn.Module):
            def forward(self, gray):
                return torch.tensor([[10.0, 20.0, 30.0, 40.0]])

        # V3 order (plastic_submarine=1, plastic_fish=2) vs gray order
        # (plastic_fish=1, plastic_submarine=2): V3 column i takes gray col.
        perm = [0, 2, 1, 3]
        adapter = GrayBackboneAdapter(FixedBackbone(), perm=perm)
        out = adapter(torch.zeros(1, 3, 8, 8))
        torch.testing.assert_close(out, torch.tensor([[10.0, 30.0, 20.0, 40.0]]))

    def test_fusion_with_real_style_eval_backbone(self):
        """End-to-end: eval-mode tuple backbone + permutation inside the
        fusion forward must not raise (review issue 3 regression)."""

        class UltralyticsStyleBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 4)

            def forward(self, gray):
                logits = self.linear(gray.mean(dim=(2, 3)))
                if self.training:
                    return logits
                return (logits.softmax(-1), logits)

        adapter = GrayBackboneAdapter(
            UltralyticsStyleBackbone().eval(), perm=[0, 2, 1, 3]
        )
        model = PolarFusionModel(adapter, num_classes=4)
        model.eval()
        gray, polar, quality = make_inputs()
        out = model(gray, polar, quality)
        assert out["final_logits"].shape == (4, 4)
        torch.testing.assert_close(
            out["final_logits"], out["gray_logits"] + out["gate"] * out["polar_delta"]
        )


class TestGrayClassPerm:
    GRAY_NAMES = {
        0: "metal_submarine",
        1: "plastic_fish",
        2: "plastic_submarine",
        3: "real_fish",
    }
    V3_NAMES = [
        "metal_submarine",
        "plastic_submarine",
        "plastic_fish",
        "real_fish",
    ]

    def test_perm_fixes_plastic_fish_submarine_swap(self):
        assert build_gray_class_perm(self.GRAY_NAMES, self.V3_NAMES) == [0, 2, 1, 3]

    def test_identity_when_orders_match(self):
        names = {0: "a", 1: "b", 2: "c", 3: "d"}
        assert build_gray_class_perm(names, ["a", "b", "c", "d"]) == [0, 1, 2, 3]

    def test_missing_target_class_raises(self):
        with pytest.raises(ValueError, match="not present"):
            build_gray_class_perm(self.GRAY_NAMES, ["metal_submarine", "unknown"])

    def test_duplicate_gray_names_raise(self):
        with pytest.raises(ValueError, match="not unique"):
            build_gray_class_perm({0: "a", 1: "a"}, ["a"])


class TestGrayInitMetadata:
    def test_roundtrip_preserves_gray_init(self, tmp_path):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        meta = fusion_metadata(
            ["w", "x", "y", "z"],
            gray_weights="runs/train/run_20260913_initial/model_b-gray/weights/best.pt",
            gray_class_names=[
                "metal_submarine",
                "plastic_fish",
                "plastic_submarine",
                "real_fish",
            ],
            head_replaced=False,
            class_permutation=[0, 2, 1, 3],
        )
        path = save_fusion_checkpoint(tmp_path / "fusion.pt", model, meta)
        assert read_fusion_metadata(path) == meta

    def test_metadata_defaults_empty(self):
        meta = fusion_metadata(["w", "x", "y", "z"])
        assert meta.gray_weights == ""
        assert meta.gray_class_names == ()
        assert meta.head_replaced is False
        assert meta.class_permutation == ()

    def test_fresh_head_metadata_records_replacement(self, tmp_path):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        meta = fusion_metadata(
            ["w", "x", "y", "z"],
            base_model="yolo26n-cls.pt",
            head_replaced=True,
        )
        path = save_fusion_checkpoint(tmp_path / "fusion.pt", model, meta)
        loaded = read_fusion_metadata(path)
        assert loaded.head_replaced is True
        assert loaded.gray_weights == ""
        assert loaded.base_model == "yolo26n-cls.pt"


class TestSmokeMetadataFields:
    """limit_batches/smoke round-trip; old checkpoints read as None so they
    can never default-pass formal admission (review issue 3)."""

    def test_roundtrip_preserves_smoke_fields(self, tmp_path):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        meta = fusion_metadata(
            ["w", "x", "y", "z"], limit_batches=3, smoke=True
        )
        path = save_fusion_checkpoint(tmp_path / "fusion.pt", model, meta)
        loaded = read_fusion_metadata(path)
        assert loaded == meta
        assert loaded.limit_batches == 3
        assert loaded.smoke is True

    def test_defaults_are_none(self):
        meta = fusion_metadata(["w", "x", "y", "z"])
        assert meta.limit_batches is None
        assert meta.smoke is None

    def test_old_checkpoint_without_fields_reads_none(self, tmp_path):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        meta = fusion_metadata(["w", "x", "y", "z"])
        path = save_fusion_checkpoint(tmp_path / "fusion.pt", model, meta)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        del payload["limit_batches"]
        del payload["smoke"]
        torch.save(payload, path)
        loaded = read_fusion_metadata(path)
        assert loaded.limit_batches is None
        assert loaded.smoke is None


class TestHeads:
    def test_delta_net_output_shape(self):
        delta = PolarDeltaNet(num_classes=4)
        out = delta(torch.rand(2, 3, 48, 48))
        assert out.shape == (2, 4)

    def test_gate_net_is_mlp_sigmoid_in_unit_range(self):
        gate = PolarGateNet()
        assert isinstance(gate.net[-1], nn.Sigmoid)
        out = gate(torch.rand(5, QUALITY_VECTOR_LENGTH) * 2)
        assert out.shape == (5, 1)
        assert torch.all((out >= 0) & (out <= 1))

    def test_gate_input_dim_matches_quality_vector(self):
        gate = PolarGateNet()
        assert gate.net[0].in_features == QUALITY_VECTOR_LENGTH == 4


class TestFusionForward:
    def test_output_shapes(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        gray, polar, quality = make_inputs()
        out = model(gray, polar, quality)
        assert out["final_logits"].shape == (4, 4)
        assert out["gray_logits"].shape == (4, 4)
        assert out["polar_delta"].shape == (4, 4)
        assert out["gate"].shape == (4, 1)

    def test_fusion_equals_gray_plus_gate_times_delta(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        model.eval()
        gray, polar, quality = make_inputs(seed=3)
        with torch.no_grad():
            out = model(gray, polar, quality)
            expected = out["gray_logits"] + out["gate"] * out["polar_delta"]
        torch.testing.assert_close(out["final_logits"], expected)

    def test_zero_valid_ratio_forces_gate_zero(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        model.eval()
        gray, polar, quality = make_inputs(seed=1)
        quality[:, QUALITY_VECTOR_KEYS.index("valid_ratio")] = 0.0
        with torch.no_grad():
            out = model(gray, polar, quality)
        assert torch.all(out["gate"] == 0.0)
        torch.testing.assert_close(out["final_logits"], out["gray_logits"])

    def test_explicit_invalid_flag_forces_gate_zero(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        model.eval()
        gray, polar, quality = make_inputs(seed=2)
        invalid = torch.tensor([True, False, True, False])
        with torch.no_grad():
            out = model(gray, polar, quality, polar_invalid=invalid)
        assert torch.all(out["gate"][invalid] == 0.0)
        # Untouched samples keep a learned gate value (>= 0, learned)
        assert torch.all(out["gate"][~invalid] >= 0.0)

    def test_positive_valid_ratio_does_not_force_zero(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        model.eval()
        gray, polar, quality = make_inputs(seed=4)
        quality[:, 0] = 0.5  # valid_ratio > 0
        with torch.no_grad():
            out = model(gray, polar, quality)
        # Sigmoid can in principle emit exactly 0 only at -inf; with random
        # positive-quality inputs the gate must stay strictly positive here.
        assert torch.all(out["gate"] > 0.0)

    def test_num_classes_validation(self):
        with pytest.raises(ValueError, match="num_classes"):
            PolarFusionModel(TinyBackbone(1), num_classes=1)

    def test_backbone_logit_mismatch_raises(self):
        model = PolarFusionModel(TinyBackbone(3), num_classes=4)
        gray, polar, quality = make_inputs()
        with pytest.raises(ValueError, match="logits"):
            model(gray, polar, quality)


class TestFreezePhases:
    def test_freeze_excludes_gray_parameters(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        params = model.trainable_parameters(include_gray=False)
        param_ids = {id(p) for p in params}
        assert all(id(p) not in param_ids for p in model.gray_backbone.parameters())
        assert any(id(p) in param_ids for p in model.delta_net.parameters())
        assert any(id(p) in param_ids for p in model.gate_net.parameters())

    def test_joint_includes_gray_parameters(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        params = model.trainable_parameters(include_gray=True)
        param_ids = {id(p) for p in params}
        assert any(id(p) in param_ids for p in model.gray_backbone.parameters())

    def test_set_gray_frozen_toggles_requires_grad(self):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        model.set_gray_frozen(True)
        assert all(not p.requires_grad for p in model.gray_backbone.parameters())
        assert all(p.requires_grad for p in model.delta_net.parameters())
        model.set_gray_frozen(False)
        assert all(p.requires_grad for p in model.gray_backbone.parameters())


class TestCheckpointing:
    def test_metadata_contains_required_audit_fields(self):
        meta = fusion_metadata(["a", "b", "c", "d"], base_model="yolo26n-cls.pt", imgsz=224)
        assert meta.version == FUSION_VERSION
        assert meta.class_names == ("a", "b", "c", "d")
        assert "gray" in meta.input_format and "signed_q" in meta.input_format
        assert meta.base_model == "yolo26n-cls.pt"
        assert meta.quality_vector_keys == tuple(QUALITY_VECTOR_KEYS)
        assert meta.imgsz == 224

    def test_default_base_is_yolo26n_cls(self):
        assert DEFAULT_BASE_MODEL == "yolo26n-cls.pt"

    def test_save_load_roundtrip(self, tmp_path):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        meta = fusion_metadata(["w", "x", "y", "z"])
        path = save_fusion_checkpoint(tmp_path / "fusion.pt", model, meta)

        fresh = PolarFusionModel(TinyBackbone(4), num_classes=4)
        loaded, loaded_meta, payload = load_fusion_checkpoint(path, fresh.gray_backbone)
        torch.testing.assert_close(
            loaded.delta_net.state_dict()[list(loaded.delta_net.state_dict())[0]],
            model.delta_net.state_dict()[list(model.delta_net.state_dict())[0]],
        )
        assert loaded_meta == meta
        assert payload["version"] == FUSION_VERSION

    def test_read_metadata_without_weights(self, tmp_path):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        meta = fusion_metadata(["w", "x", "y", "z"])
        path = save_fusion_checkpoint(tmp_path / "fusion.pt", model, meta)
        assert read_fusion_metadata(path) == meta

    def test_save_rejects_class_count_mismatch(self, tmp_path):
        model = PolarFusionModel(TinyBackbone(4), num_classes=4)
        with pytest.raises(ValueError, match="class_names"):
            save_fusion_checkpoint(
                tmp_path / "bad.pt", model, fusion_metadata(["only", "two"])
            )

    def test_load_rejects_checkpoint_missing_metadata(self, tmp_path):
        torch.save({"state_dict": {}}, tmp_path / "bare.pt")
        with pytest.raises(ValueError, match="missing field"):
            load_fusion_checkpoint(tmp_path / "bare.pt", TinyBackbone(4))


def write_sample(root: Path, split: str, class_name: str, name: str, class_id: int, seed: int):
    rng = np.random.default_rng(seed)
    shape = (24, 20)
    gray = rng.integers(0, 256, size=shape, dtype=np.uint8)
    signed = rng.uniform(-1, 1, size=shape).astype(np.float32)
    save_fusion_sample(
        root / split / class_name / f"{name}.npz",
        gray=gray,
        signed_q=signed,
        abs_q=np.abs(signed),
        valid=(rng.uniform(size=shape) > 0.5).astype(np.uint8),
        quality=build_quality_vector(0.5, 0.6, 0.7, 0.2),
        class_id=class_id,
    )
    return {
        "sample_name": name,
        "split": split,
        "group_name": f"group_{class_name}",
        "class_id": str(class_id),
        "class_name": class_name,
        "source_frame": name,
        "object_index": "0",
        "stereo_valid": "true",
        "stereo_reason": "ok",
        "stereo_valid_ratio": "0.900000",
        "disparity": "12.0000",
        "polar_valid_ratio": "0.500000",
        "quality_valid_ratio": "0.500000",
        "quality_in_bounds_ratio": "0.600000",
        "quality_brightness_valid_ratio": "0.700000",
        "quality_mean_abs_q": "0.200000",
        "npz_path": f"{split}/{class_name}/{name}.npz",
    }


def write_manifest(root: Path, rows):
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    (root / "dataset_manifest.csv").write_text(buffer.getvalue(), encoding="utf-8")


class TestFusionDatasetAndManifest:
    @pytest.fixture
    def mini_dataset(self, tmp_path):
        root = tmp_path / "fusion_v3"
        rows = []
        for i in range(3):
            rows.append(
                write_sample(root, "train", "metal", f"m{i}", 0, seed=i)
            )
        for i in range(2):
            rows.append(
                write_sample(root, "val", "plastic", f"p{i}", 1, seed=10 + i)
            )
        write_manifest(root, rows)
        return root

    def test_read_manifest_split(self, mini_dataset):
        paths, class_ids, class_names = read_manifest_split(mini_dataset, "train")
        assert len(paths) == 3
        assert class_ids == [0, 0, 0]
        assert class_names == ["metal"] * 3
        assert all(p.is_file() for p in paths)

    def test_read_manifest_split_missing_raises(self, mini_dataset):
        with pytest.raises(ValueError, match="test"):
            read_manifest_split(mini_dataset, "test")

    def test_class_names_ordered_by_id(self, mini_dataset):
        assert class_names_from_manifest(mini_dataset) == ["metal", "plastic"]

    def test_dataset_tensors(self, mini_dataset):
        paths, _, _ = read_manifest_split(mini_dataset, "train")
        dataset = FusionClsDataset(paths, imgsz=16)
        assert len(dataset) == 3
        gray, polar, quality, class_id = dataset[0]
        assert gray.shape == (3, 16, 16)
        assert polar.shape == (3, 16, 16)
        assert quality.shape == (QUALITY_VECTOR_LENGTH,)
        assert 0.0 <= gray.min() and gray.max() <= 1.0
        assert class_id == 0

    def test_dataset_feeds_fusion_model(self, mini_dataset):
        paths, _, _ = read_manifest_split(mini_dataset, "train")
        loader = torch.utils.data.DataLoader(
            FusionClsDataset(paths, imgsz=16), batch_size=2
        )
        model = PolarFusionModel(TinyBackbone(2), num_classes=2)
        model.eval()
        batch = next(iter(loader))
        with torch.no_grad():
            out = model(*batch[:3])
        assert out["final_logits"].shape == (2, 2)
