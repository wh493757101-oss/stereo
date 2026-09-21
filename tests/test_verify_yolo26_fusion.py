"""Tests for scripts.verify_yolo26_fusion judgment logic and the gray-only
full-model unfreeze fix. No real training, no network: the Ultralytics
loader is faked.
"""

import copy
import csv
import io
import json
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

import scripts.train_gray_fusion as tgf
import scripts.train_polar_fusion as tpf
import scripts.verify_yolo26_fusion as vf
from core.fusion_dataset import build_quality_vector, save_fusion_sample

V3_CLASS_NAMES = [
    "metal_submarine",
    "plastic_submarine",
    "plastic_fish",
    "real_fish",
]


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="verify_yolo26_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def write_sample(root: Path, split: str, class_name: str, name: str, class_id: int, seed: int):
    rng = np.random.default_rng(seed)
    shape = (20, 16)
    signed = rng.uniform(-1, 1, size=shape).astype(np.float32)
    save_fusion_sample(
        root / split / class_name / f"{name}.npz",
        gray=rng.integers(0, 256, size=shape, dtype=np.uint8),
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


@pytest.fixture
def mini_dataset(tmp_path):
    """4-class mini dataset (V3 order) with train + val + test splits."""
    root = tmp_path / "fusion_v3names"
    rows = []
    for class_id, name in enumerate(V3_CLASS_NAMES):
        rows.append(
            write_sample(root, "train", name, f"tr{class_id}", class_id, seed=class_id)
        )
    for class_id, name in enumerate(V3_CLASS_NAMES[:2]):
        rows.append(
            write_sample(root, "val", name, f"va{class_id}", class_id, seed=40 + class_id)
        )
    for class_id, name in enumerate(V3_CLASS_NAMES[:2]):
        rows.append(
            write_sample(root, "test", name, f"te{class_id}", class_id, seed=60 + class_id)
        )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    (root / "dataset_manifest.csv").write_text(buffer.getvalue(), encoding="utf-8")
    return root


class FakeClassifyHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features):
        return self.linear(features)


class TrainableFakeModel(nn.Module):
    """Classification-like model with real input dependence: conv feature
    extractor (early layer), BatchNorm and a linear head at ``model[-1]``.

    ``frozen=True`` simulates the official Ultralytics checkpoint, which
    loads with every parameter ``requires_grad=False``.
    """

    def __init__(self, out_features, names, yaml_file, task="classify", frozen=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            FakeClassifyHead(8, out_features),
        )
        self.names = names
        self.yaml = {"yaml_file": yaml_file}
        self.task = task
        if frozen:
            for param in self.parameters():
                param.requires_grad_(False)

    def forward(self, gray):
        logits = self.model(gray)
        if self.training:
            return logits
        return (logits.softmax(-1), logits)


class FakeYOLO:
    last_weights: str | None = None

    def __init__(self, weights):
        FakeYOLO.last_weights = weights
        path = Path(str(weights))
        payload = None
        if path.is_file():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                payload = None  # plain placeholder file (fake base)
        if isinstance(payload, dict) and "model" in payload:
            self.model = payload["model"]
        else:
            # Official base: 1000-class classification model, fully frozen.
            self.model = TrainableFakeModel(
                1000, {i: f"imagenet_{i}" for i in range(1000)}, "yolo26n-cls.yaml",
                frozen=True,
            )


@pytest.fixture
def fake_ultralytics(monkeypatch):
    import types

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    FakeYOLO.last_weights = None
    return module


def allow_training(monkeypatch):
    monkeypatch.setattr(tgf, "validate_dataset", lambda root: {})
    monkeypatch.setattr(tgf, "resolve_device", lambda *a, **kw: "cpu")


def make_base_file(tmp_path: Path) -> Path:
    base = tmp_path / "yolo26n-cls.pt"
    base.write_bytes(b"fake")
    return base


# ---------------------------------------------------------------------------
# Synthetic snapshot helpers for the judgment-logic tests. The verifier must
# judge updates by comparing parameter/buffer snapshots, never by comparing
# to the official base checkpoint.
# ---------------------------------------------------------------------------

HEAD_NAMES = {"model.5.linear.weight", "model.5.linear.bias"}


def _synthetic_snapshots():
    before = {
        "params": {
            "model.0.weight": torch.zeros(4),
            "model.0.bias": torch.zeros(4),
            "model.5.linear.weight": torch.ones(4, 8),
            "model.5.linear.bias": torch.zeros(4),
        },
        "buffers": {
            "model.1.running_mean": torch.zeros(8),
            "model.1.running_var": torch.ones(8),
            "model.1.num_batches_tracked": torch.tensor(0),
        },
    }
    head_only = copy.deepcopy(before)
    head_only["params"]["model.5.linear.weight"] += 1.0
    head_only["params"]["model.5.linear.bias"] += 1.0
    head_only["buffers"]["model.1.running_mean"] += 0.5
    head_only["buffers"]["model.1.num_batches_tracked"] += 1

    buffers_only = copy.deepcopy(before)
    buffers_only["buffers"]["model.1.running_var"] *= 1.5

    backbone_updated = copy.deepcopy(head_only)
    backbone_updated["params"]["model.0.weight"] += 0.25
    return before, head_only, buffers_only, backbone_updated


def _ok_grads(name="model.0.weight", absmax=0.5):
    return {name: {"finite": True, "absmax": absmax}}


class TestVerifierJudgment:
    """The verifier must reject head-only or buffer-only "updates" and
    accept only real backbone parameter updates (issue 2)."""

    def test_rejects_head_only_update_with_buffer_changes(self):
        before, head_only, _, _ = _synthetic_snapshots()
        report = vf._update_report(before, head_only, HEAD_NAMES)
        assert report["params_changed_count"] == 2  # only the head
        assert report["backbone_params_changed_count"] == 0
        assert report["buffers_changed_count"] >= 1
        with pytest.raises(RuntimeError, match="non-head"):
            vf._assert_gray_learning(report, "model.0.weight", _ok_grads())

    def test_buffers_only_changes_do_not_count_as_parameter_updates(self):
        before, _, buffers_only, _ = _synthetic_snapshots()
        report = vf._update_report(before, buffers_only, HEAD_NAMES)
        assert report["params_changed_count"] == 0
        assert report["buffers_changed_count"] >= 1
        with pytest.raises(RuntimeError, match="no parameter changed"):
            vf._assert_gray_learning(report, "model.0.weight", _ok_grads())

    def test_rejects_missing_early_gradient(self):
        before, _, _, backbone_updated = _synthetic_snapshots()
        report = vf._update_report(before, backbone_updated, HEAD_NAMES)
        with pytest.raises(RuntimeError, match="gradient"):
            vf._assert_gray_learning(report, "model.0.weight", {})

    def test_rejects_nonfinite_gradient(self):
        before, _, _, backbone_updated = _synthetic_snapshots()
        report = vf._update_report(before, backbone_updated, HEAD_NAMES)
        grads = _ok_grads()
        grads["model.5.linear.weight"] = {"finite": False, "absmax": 1.0}
        with pytest.raises(RuntimeError, match="finite"):
            vf._assert_gray_learning(report, "model.0.weight", grads)

    def test_accepts_real_backbone_update(self):
        before, _, _, backbone_updated = _synthetic_snapshots()
        report = vf._update_report(before, backbone_updated, HEAD_NAMES)
        evidence = vf._assert_gray_learning(
            report, "model.0.weight", _ok_grads()
        )
        assert evidence["early_backbone_changed"] is True
        assert report["backbone_params_changed_count"] == 1
        assert report["backbone_params_changed"] == ["model.0.weight"]
        assert "model.1.running_mean" in report["buffers_changed"]


class TestGrayFrozenCheck:
    def _module_and_state(self):
        module = TrainableFakeModel(
            4, {i: n for i, n in enumerate(V3_CLASS_NAMES)}, "yolo26n-cls.yaml",
            frozen=False,
        )
        state = {
            f"gray_backbone.module.{name}": value.detach().clone()
            for name, value in module.state_dict().items()
        }
        return module, state

    def test_unchanged_gray_passes(self):
        module, state = self._module_and_state()
        result = vf._check_gray_frozen(module, state)
        assert result["gray_params_unchanged"] == sum(
            1 for _ in module.named_parameters()
        )
        assert result["gray_buffers_unchanged"] == sum(
            1 for _ in module.named_buffers()
        )

    def test_parameter_drift_rejected(self):
        module, state = self._module_and_state()
        state["gray_backbone.module.model.0.weight"] += 1.0
        with pytest.raises(RuntimeError, match="parameter drifted"):
            vf._check_gray_frozen(module, state)

    def test_buffer_drift_rejected(self):
        module, state = self._module_and_state()
        state["gray_backbone.module.model.1.running_mean"] += 1.0
        with pytest.raises(RuntimeError, match="buffer drifted"):
            vf._check_gray_frozen(module, state)


class TestDeltaUpdateReport:
    LAST = {"delta_net.net.8.weight", "delta_net.net.8.bias"}

    def _snapshots(self):
        before = {
            "delta_net.net.0.weight": torch.full((2, 3, 5, 5), 0.1),
            "delta_net.net.8.weight": torch.zeros(4, 64),
            "delta_net.net.8.bias": torch.zeros(4),
        }
        unchanged = copy.deepcopy(before)
        updated = copy.deepcopy(before)
        updated["delta_net.net.0.weight"] += 0.01
        updated["delta_net.net.8.weight"] += 0.02
        return before, unchanged, updated

    def test_nonzero_conv_init_without_updates_rejected(self):
        """Non-zero conv initialization must not be mistaken for an update:
        the zero-init last layer and all other parameters are unchanged."""
        before, unchanged, _ = self._snapshots()
        report = vf._delta_update_report(before, unchanged, self.LAST)
        assert report["delta_params_changed_count"] == 0
        assert report["delta_zero_init_last_layer_before_all_zero"] is True
        assert report["delta_last_layer_changed"] is False
        with pytest.raises(RuntimeError, match="delta"):
            vf._assert_delta_updated(report)

    def test_real_updates_pass_and_report_last_layer(self):
        before, _, updated = self._snapshots()
        report = vf._delta_update_report(before, updated, self.LAST)
        assert report["delta_params_changed_count"] == 2
        assert report["delta_last_layer_changed"] is True
        vf._assert_delta_updated(report)  # must not raise


class TestGrayUnfreezeEndToEnd:
    """phase_a_gray must prove the whole model was unfrozen and the real
    backbone learned (issue 1)."""

    def _run_phase_a(self, tmp_path, mini_dataset, monkeypatch):
        allow_training(monkeypatch)
        base = make_base_file(tmp_path)
        run_id = "test_vf_phase_a"
        args = vf.parse_args(
            [
                "--run-id", run_id,
                "--base", str(base),
                "--data", str(mini_dataset),
                "--device", "cpu",
                "--imgsz", "16",
                "--batch", "2",
            ]
        )
        try:
            result = vf.phase_a_gray(args, "cpu", mini_dataset)
        finally:
            shutil.rmtree(
                tpf.PROJECT_ROOT / "runs" / "train" / f"{run_id}_gray",
                ignore_errors=True,
            )
        return result

    def test_unfreezes_and_updates_backbone(
        self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch
    ):
        result = self._run_phase_a(tmp_path, mini_dataset, monkeypatch)
        assert result["params_trainable"] == result["params_total"]
        updates = result["parameter_updates"]
        assert updates["backbone_params_changed_count"] >= 1
        assert "model.0.weight" in updates["backbone_params_changed"]
        grads = result["gradient_observations"]
        assert grads["all_finite"] is True
        assert grads["early_parameter"] == "model.0.weight"
        assert grads["early_grad_absmax"] > 0.0
        assert grads["backbone_nonzero_grad_count"] >= 1
        # BN running statistics are tracked separately from parameters.
        assert updates["buffers_changed_count"] >= 1
        assert updates["buffers_total"] >= 3
        # Save/reload fidelity and truncation are still recorded.
        assert result["save_reload_logits_max_diff"] <= vf.LOGITS_TOLERANCE
        assert result["train_batches"] == [2, 2]

    def test_verifier_rejects_when_backbone_never_updates(
        self, tmp_path, mini_dataset, fake_ultralytics, monkeypatch
    ):
        """End-to-end negative: if the training path trains only the head
        (simulated by keeping the backbone frozen), phase_a_gray must fail
        instead of reporting PASSED."""
        allow_training(monkeypatch)
        base = make_base_file(tmp_path)
        original_prepare = tgf.prepare_gray_backbone

        def prepare_then_refreeze(base_path, gray_weights, class_names, device):
            backbone, info = original_prepare(base_path, gray_weights, class_names, device)
            for param in backbone.parameters():
                param.requires_grad_(False)
            # Re-enable only the head, like the pre-fix behavior.
            head = backbone.module.model[-1]
            for param in head.parameters():
                param.requires_grad_(True)
            return backbone, info

        monkeypatch.setattr(tgf, "prepare_gray_backbone", prepare_then_refreeze)
        # Neutralize the entry's explicit unfreeze so the simulated
        # regression actually reaches the verifier (the entry guard and the
        # verifier both reject it).
        monkeypatch.setattr(
            tgf,
            "_unfreeze_gray_model",
            lambda model: sum(1 for p in model.parameters() if p.requires_grad),
        )
        args = vf.parse_args(
            [
                "--run-id", "test_vf_phase_a_negative",
                "--base", str(base),
                "--data", str(mini_dataset),
                "--device", "cpu",
                "--imgsz", "16",
                "--batch", "2",
            ]
        )
        try:
            with pytest.raises(RuntimeError, match="unfrozen|non-head|backbone"):
                vf.phase_a_gray(args, "cpu", mini_dataset)
        finally:
            shutil.rmtree(
                tpf.PROJECT_ROOT / "runs" / "train" / "test_vf_phase_a_negative_gray",
                ignore_errors=True,
            )


class TestRunIdValidation:
    @pytest.mark.parametrize(
        "bad_id",
        ["", ".", "..", "../escape", "..\\escape", "C:\\absolute\\path", "with space", "a/b"],
    )
    def test_invalid_run_id_rejected_before_any_work(
        self, tmp_path, monkeypatch, capsys, bad_id
    ):
        monkeypatch.setattr(vf, "PROJECT_ROOT", tmp_path)

        def boom(*args, **kwargs):
            raise AssertionError("no work may run for an invalid run id")

        monkeypatch.setattr(tpf, "resolve_device", boom)
        monkeypatch.setattr(vf, "phase_a_gray", boom)
        exit_code = vf.main(
            ["--run-id", bad_id, "--base", "yolo26n-cls.pt", "--data", "datasets/x"]
        )
        assert exit_code == 2
        assert "invalid run id" in capsys.readouterr().err
        assert not any(tmp_path.rglob("*"))

    def test_derived_dirs_must_stay_under_runs_train(self, tmp_path, monkeypatch, capsys):
        """Defense in depth: even if a bad id slipped past validation, a
        derived directory escaping runs/train must be refused before any
        output exists."""
        monkeypatch.setattr(vf, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(vf, "validate_run_id", lambda run_id: run_id)
        exit_code = vf.main(["--run-id", "..", "--base", "yolo26n-cls.pt", "--data", "datasets/x"])
        assert exit_code == 2
        assert "runs/train" in capsys.readouterr().err
        assert not (tmp_path / "runs").exists()

    def test_valid_run_id_passes_validation(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(vf, "PROJECT_ROOT", tmp_path)
        exit_code = vf.main(
            [
                "--run-id", "valid_smoke_123",
                "--base", "missing-base.pt",
                "--data", "datasets/x",
                "--device", "cpu",
            ]
        )
        assert exit_code == 2  # missing base, not an invalid run id
        err = capsys.readouterr().err
        assert "missing" in err
        assert "invalid run id" not in err
        assert not any(tmp_path.rglob("*"))
