"""Evaluation: routing, pure-mode entry, data admission, report contract."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from core.fusion_dataset import dataset_fingerprint
from core.material_fusion_dataset import MaterialFusionDataset
from models.material_fusion import (
    CLASS_NAMES,
    MATERIAL_FUSION_VERSION,
    PREPROCESSING_VERSION,
    UPSTREAM_COMMIT,
    build_material_model,
    material_fusion_metadata,
    save_material_checkpoint,
)
from models.polar_fusion import read_manifest_split
from scripts.eval_material_fusion import (
    evaluate_with_fallback,
    load_fallback_gray,
    main,
)

from .conftest import build_v4_dataset, write_experiment_config


class FixedLogitsModel(nn.Module):
    """Returns a fixed per-sample logit vector; dual-style interface."""

    def __init__(self, logits_by_class: list[float]):
        super().__init__()
        self.is_dual = True
        self.register_buffer("logits", torch.tensor(logits_by_class))

    def forward(self, rgb, aux=None):
        return self.logits.expand(rgb.shape[0], -1)


class GrayFallbackStub(nn.Module):
    """Minimal identity-correct fallback: canonical names + 4-class head."""

    def __init__(self, imgsz: int = 64):
        super().__init__()
        self.names = {i: name for i, name in enumerate(CLASS_NAMES)}
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(3 * imgsz * imgsz, 4))

    def forward(self, x):
        return self.model(x)


class WrongNamesModel(nn.Module):
    """Fallback stub whose class names do not match the canonical order."""

    def __init__(self):
        super().__init__()
        self.names = {0: "cat", 1: "dog", 2: "fish", 3: "sub"}
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 4))

    def forward(self, x):
        return self.model(x)


class WrongHeadModel(nn.Module):
    """Fallback stub with a 1000-wide classification head."""

    def __init__(self):
        super().__init__()
        self.names = {i: name for i, name in enumerate(CLASS_NAMES)}
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 1000))

    def forward(self, x):
        return self.model(x)


def _loader_for(data_root, batch_size=3):
    paths, _, _ = read_manifest_split(data_root, "val")
    dataset = MaterialFusionDataset(paths, variant="gray_polar", imgsz=64)
    markers = [float(dataset[i][0].mean()) for i in range(len(paths))]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False), markers


@pytest.fixture()
def routing_setup(scratch_dir):
    data_root = build_v4_dataset(
        scratch_dir / "v4", counts=(4, 6, 4), invalid_indices={"val": {1, 4}}
    )
    loader, markers = _loader_for(data_root)
    return {"data_root": data_root, "loader": loader, "markers": markers}


def build_eval_artifacts(
    scratch_dir,
    variant: str = "gray_polar",
    data_root=None,
    counts=(4, 6, 4),
    invalid_indices=None,
    batch=3,
):
    """Checkpoint + run_config + dataset with matching fingerprints."""
    if data_root is None:
        data_root = build_v4_dataset(
            scratch_dir / "v4", counts=counts, invalid_indices=invalid_indices or {}
        )
    fingerprint = dataset_fingerprint(data_root)
    model = build_material_model(variant, imgsz=64, drop_path_rate=0.0)
    metadata = material_fusion_metadata(
        variant=variant, imgsz=64, data_fingerprint=fingerprint,
        pretrained_sha256="0" * 64, seed=2026, epoch=2, stage="finetune",
        smoke=True, limit_batches=4, val_macro_f1=0.1, val_accuracy=0.2,
        physical_batch=2, accum_steps=2, effective_batch=4, drop_path_rate=0.0,
    )
    checkpoint = save_material_checkpoint(scratch_dir / "best.pt", model, metadata)
    run_config = {
        "config": {
            "experiment": MATERIAL_FUSION_VERSION,
            "variant": variant,
            "data": str(data_root),
            "data_fingerprint": fingerprint,
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
            "fallback_weights": "",
            "fallback_identity": "",
            "smoke": True,
            "limit_batches": 4,
        },
        "run_id": "test_run",
        "best": {"epoch": 1, "val_macro_f1": 0.1, "val_accuracy": 0.2},
        "last": {"epoch": 2, "val_macro_f1": 0.1, "val_accuracy": 0.2},
    }
    (scratch_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    return {"checkpoint": checkpoint, "data_root": data_root, "config": run_config["config"]}


class TestFallbackRouting:
    def test_routing_bypasses_cmx_for_invalid_samples(self, routing_setup):
        cmx = FixedLogitsModel([0.0, 1.0, 0.0, 0.0])       # always class 1
        fallback = FixedLogitsModel([2.0, 0.0, 0.0, 0.0])  # always class 0
        report = evaluate_with_fallback(
            cmx, fallback, routing_setup["loader"], "cpu", num_classes=4, routing=True
        )
        assert report["routing"]["applied"] is True
        assert report["invalid_samples"] == 2
        assert report["samples"] == 6
        assert report["pure_classifier"]["per_class_recall"]["1"] == 1.0
        # Val labels are [0,1,2,3,0,1]; invalid samples 1 and 4 get class 0,
        # so class 0 recall is 1/2 (sample 0 stays with the CMX model).
        assert report["routing"]["changed_predictions"] == 2
        assert report["routed"]["per_class_recall"]["0"] == 0.5

    def test_pure_mode_never_calls_fallback(self, routing_setup):
        cmx = FixedLogitsModel([0.0, 1.0, 0.0, 0.0])

        class CountingFallback(FixedLogitsModel):
            def __init__(self):
                super().__init__([2.0, 0.0, 0.0, 0.0])
                self.calls = 0

            def forward(self, rgb, aux=None):
                self.calls += 1
                return super().forward(rgb, aux)

        fallback = CountingFallback()
        report = evaluate_with_fallback(
            cmx, fallback, routing_setup["loader"], "cpu", num_classes=4, routing=False
        )
        assert fallback.calls == 0
        assert report["routing"]["applied"] is False
        assert report["routed"] is None
        assert report["invalid_samples"] == 2  # still measured for the report
        assert "changed_predictions" not in report["routing"]

    def test_pure_mode_allows_fallback_none(self, routing_setup):
        cmx = FixedLogitsModel([0.0, 1.0, 0.0, 0.0])
        report = evaluate_with_fallback(
            cmx, None, routing_setup["loader"], "cpu", num_classes=4, routing=False
        )
        # Labels are [0,1,2,3,0,1]; the model always predicts class 1.
        assert report["pure_classifier"]["accuracy"] == pytest.approx(2 / 6)
        assert report["routing"]["applied"] is False

    def test_routing_requires_fallback(self, routing_setup):
        cmx = FixedLogitsModel([0.0, 1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="fallback"):
            evaluate_with_fallback(
                cmx, None, routing_setup["loader"], "cpu", num_classes=4, routing=True
            )

    def test_mixed_batch_preserves_sample_order(self, routing_setup):
        class OrderProbe(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_dual = True
                self.seen = []

            def forward(self, rgb, aux=None):
                self.seen.append(rgb.mean(dim=(1, 2, 3)).detach().clone())
                return torch.zeros(rgb.shape[0], 4)

        cmx = OrderProbe()
        fallback = OrderProbe()
        evaluate_with_fallback(
            cmx, fallback, routing_setup["loader"], "cpu", num_classes=4, routing=True
        )
        markers = routing_setup["markers"]
        # batch_size 3, 6 samples, invalid indices {1, 4}: per batch the CMX
        # model is called first for the pure pass (all samples) and then only
        # for the valid samples, in original order; the fallback only ever
        # sees the invalid samples, in original order.
        expected_calls = [
            markers[0:3],
            [markers[0], markers[2]],
            markers[3:6],
            [markers[3], markers[5]],
        ]
        assert len(cmx.seen) == len(expected_calls)
        for seen, expected in zip(cmx.seen, expected_calls):
            assert seen.tolist() == pytest.approx(expected, abs=1e-5)
        assert torch.cat(fallback.seen).tolist() == pytest.approx(
            [markers[1], markers[4]], abs=1e-5
        )

    def test_all_invalid_batch_routes_everything(self, scratch_dir):
        data_root = build_v4_dataset(
            scratch_dir / "v4", counts=(4, 6, 4), invalid_indices={"val": set(range(6))}
        )
        loader, _ = _loader_for(data_root, batch_size=4)
        cmx = FixedLogitsModel([0.0, 1.0, 0.0, 0.0])
        fallback = FixedLogitsModel([0.0, 0.0, 3.0, 0.0])
        report = evaluate_with_fallback(
            cmx, fallback, loader, "cpu", num_classes=4, routing=True
        )
        assert report["invalid_samples"] == 6
        assert report["routed"]["per_class_recall"]["2"] == 1.0
        assert report["routing"]["changed_predictions"] == 6

    def test_tail_batch_routing(self, scratch_dir):
        data_root = build_v4_dataset(
            scratch_dir / "v4", counts=(4, 5, 4), invalid_indices={"val": {4}}
        )
        loader, _ = _loader_for(data_root, batch_size=3)  # batches of 3 + 2
        cmx = FixedLogitsModel([0.0, 1.0, 0.0, 0.0])
        fallback = FixedLogitsModel([2.0, 0.0, 0.0, 0.0])
        report = evaluate_with_fallback(
            cmx, fallback, loader, "cpu", num_classes=4, routing=True
        )
        assert report["samples"] == 5
        assert report["invalid_samples"] == 1
        assert report["routing"]["changed_predictions"] == 1

    def test_eval_does_not_change_parameters_or_buffers(self, routing_setup):
        model = build_material_model("gray_polar", imgsz=64, drop_path_rate=0.0)
        fallback = FixedLogitsModel([0.0, 0.0, 1.0, 0.0])
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        before_fb = {k: v.detach().clone() for k, v in fallback.state_dict().items()}
        evaluate_with_fallback(
            model, fallback, routing_setup["loader"], "cpu", num_classes=4, routing=True
        )
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]), key
        for key, value in fallback.state_dict().items():
            assert torch.equal(value, before_fb[key]), key
        assert not model.training and not fallback.training


class TestFallbackLoading:
    def test_missing_fallback_refused(self, scratch_dir):
        with pytest.raises(FileNotFoundError, match="fallback"):
            load_fallback_gray(scratch_dir / "missing.pt", "cpu", imgsz=64)

    def test_wrong_class_names_refused(self, scratch_dir):
        path = scratch_dir / "wrong.pt"
        torch.save({"model": WrongNamesModel()}, path)
        with pytest.raises(ValueError, match="names"):
            load_fallback_gray(path, "cpu", imgsz=64)

    def test_wrong_head_width_refused(self, scratch_dir):
        path = scratch_dir / "wrong_head.pt"
        torch.save({"model": WrongHeadModel()}, path)
        with pytest.raises(ValueError, match="class"):
            load_fallback_gray(path, "cpu", imgsz=64)

    def test_correct_fallback_loads_with_identity(self, scratch_dir):
        path = scratch_dir / "gray_best.pt"
        torch.save({"model": GrayFallbackStub(imgsz=64)}, path)
        adapter, identity = load_fallback_gray(path, "cpu", imgsz=64)
        assert adapter is not None
        assert "GrayFallbackStub" in identity
        logits = adapter(torch.rand(2, 3, 64, 64))
        assert logits.shape == (2, 4)


class TestEvalEntry:
    def test_dual_routed_eval_uses_run_config(self, scratch_dir):
        artifacts = build_eval_artifacts(scratch_dir)
        fallback_path = scratch_dir / "gray_best.pt"
        torch.save({"model": GrayFallbackStub(imgsz=64)}, fallback_path)
        output = scratch_dir / "eval.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--batch", "3",
            "--fallback-weights", str(fallback_path),
            "--output", str(output),
        ])
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["variant"] == "gray_polar"
        assert report["routing"]["applied"] is True
        assert report["routed"] is not None
        assert report["samples"] == 6
        assert report["evaluation"]["samples"] == 6
        assert report["evaluation"]["truncated"] is False
        assert report["training"]["limit_batches"] == 4
        assert report["training"]["smoke"] is True
        assert report["evaluation"]["limit_batches"] == 0
        assert report["data"]["audit_fingerprint"] == artifacts["config"]["data_fingerprint"]

    def test_dual_no_fallback_pure_eval(self, scratch_dir):
        artifacts = build_eval_artifacts(scratch_dir)
        output = scratch_dir / "eval_pure.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--batch", "3",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["routing"]["applied"] is False
        assert report["routed"] is None
        assert report["fallback_weights"] == ""
        assert report["pure_classifier"]["accuracy"] >= 0.0

    def test_gray_variant_pure_eval_without_fallback(self, scratch_dir):
        """E1 has no polarization input: pure evaluation must not need the
        historical gray weights at all."""
        artifacts = build_eval_artifacts(scratch_dir, variant="gray")
        output = scratch_dir / "eval_gray.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--batch", "3",
            "--output", str(output),
        ])
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["variant"] == "gray"
        assert report["routing"]["applied"] is False
        assert report["routing"]["reason"]
        assert report["routed"] is None

    def test_explicit_variant_conflict_still_refused(self, scratch_dir):
        artifacts = build_eval_artifacts(scratch_dir, variant="gray_polar")
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--variant", "gray",
            "--split", "val",
            "--device", "cpu",
            "--no-fallback",
            "--output", str(scratch_dir / "conflict.json"),
        ])
        assert code == 2
        assert not (scratch_dir / "conflict.json").exists()

    def test_eval_truncation_recorded(self, scratch_dir):
        artifacts = build_eval_artifacts(scratch_dir)
        output = scratch_dir / "eval_trunc.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--batch", "2",
            "--limit-batches", "2",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["evaluation"]["limit_batches"] == 2
        assert report["evaluation"]["truncated"] is True
        assert report["evaluation"]["samples"] == 4
        assert report["samples"] == 4

    def test_existing_output_refused(self, scratch_dir):
        artifacts = build_eval_artifacts(scratch_dir)
        output = scratch_dir / "eval_exists.json"
        output.write_text("{}", encoding="utf-8")
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 2
        assert output.read_text(encoding="utf-8") == "{}"

    def test_data_fingerprint_mismatch_refused(self, scratch_dir):
        """Config/checkpoint agree with each other but not with the data."""
        artifacts = build_eval_artifacts(scratch_dir)
        other_root = build_v4_dataset(scratch_dir / "other_v4", counts=(4, 6, 4), seed=99)
        output = scratch_dir / "eval_wrong_data.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(other_root),
            "--split", "val",
            "--device", "cpu",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 2
        assert not output.exists()

    def test_missing_audit_refused(self, scratch_dir):
        artifacts = build_eval_artifacts(scratch_dir)
        (artifacts["data_root"] / "dataset_audit.json").unlink()
        output = scratch_dir / "eval_no_audit.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 2
        assert not output.exists()

    def test_corrupted_sample_refused(self, scratch_dir):
        import numpy as np

        artifacts = build_eval_artifacts(scratch_dir)
        sample = next(artifacts["data_root"].rglob("*.npz"))
        with np.load(sample) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["gray"] = (arrays["gray"].astype(np.uint16) + 3).astype(np.uint8)
        np.savez_compressed(sample, **arrays)
        output = scratch_dir / "eval_corrupt.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 2
        assert not output.exists()

    def test_missing_run_config_falls_back_to_template(self, scratch_dir):
        """Old checkpoints without run_config.json still evaluate through the
        explicit template + CLI variant resolution."""
        artifacts = build_eval_artifacts(scratch_dir)
        (scratch_dir / "run_config.json").unlink()
        template = write_experiment_config(
            scratch_dir / "template.yaml", artifacts["data_root"],
            scratch_dir / "mit.pth", variant="gray_polar", effective_batch=4, imgsz=64,
        )
        output = scratch_dir / "eval_template.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--config", str(template),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["variant"] == "gray_polar"

    def test_legacy_run_config_falls_back_to_template(self, scratch_dir):
        """Old run configs (no resolved-config section) stay evaluable via
        the explicit template, and the report says which path was used."""
        artifacts = build_eval_artifacts(scratch_dir)
        (scratch_dir / "run_config.json").write_text(
            json.dumps({"run_id": "legacy"}), encoding="utf-8"
        )
        template = write_experiment_config(
            scratch_dir / "template.yaml", artifacts["data_root"],
            scratch_dir / "mit.pth", variant="gray_polar", effective_batch=4, imgsz=64,
        )
        output = scratch_dir / "eval_legacy.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--config", str(template),
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--no-fallback",
            "--output", str(output),
        ])
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["config_source"]["source"] == "run_config_legacy"
        assert report["config_source"]["derived_fields"]
        assert report["variant"] == "gray_polar"

    def test_legacy_run_config_with_explicit_variant(self, scratch_dir):
        """An old E1 checkpoint evaluates with --variant gray even though the
        template default is gray_polar."""
        artifacts = build_eval_artifacts(scratch_dir, variant="gray")
        (scratch_dir / "run_config.json").write_text(
            json.dumps({"run_id": "legacy"}), encoding="utf-8"
        )
        template = write_experiment_config(
            scratch_dir / "template.yaml", artifacts["data_root"],
            scratch_dir / "mit.pth", variant="gray_polar", effective_batch=4, imgsz=64,
        )
        output = scratch_dir / "eval_legacy_gray.json"
        code = main([
            "--checkpoint", str(artifacts["checkpoint"]),
            "--config", str(template),
            "--variant", "gray",
            "--data", str(artifacts["data_root"]),
            "--split", "val",
            "--device", "cpu",
            "--output", str(output),
        ])
        assert code == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["variant"] == "gray"
        assert report["routing"]["applied"] is False
