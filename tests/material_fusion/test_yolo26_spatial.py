"""Information-barrier and learning checks for the YOLO26 spatial extension."""
import importlib
import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.polar_fusion import GrayBackboneAdapter


@pytest.fixture
def subject():
    return importlib.import_module("core.yolo26_spatial")


@pytest.fixture(autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


class TinyHead(nn.Module):
    def __init__(self, outputs=1000):
        super().__init__()
        self.expand = nn.Linear(4, 1280)
        self.drop = nn.Dropout(0.0)
        self.linear = nn.Linear(1280, outputs)

    def forward(self, x):
        return self.linear(self.drop(self.expand(x).relu()))


class TinyGray(nn.Module):
    def __init__(self, task="classify", architecture="yolo26n-cls"):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), TinyHead()])
        self.task = task
        self.yaml = {"yaml_file": architecture + ".yaml"}
        self.names = {i: str(i) for i in range(1000)}
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = self.model[1](self.model[0](x)).relu().mean((2, 3))
        logits = self.model[2](x)
        return logits if self.training else (logits.softmax(1), logits)


@pytest.fixture
def fake_base(monkeypatch, tmp_path):
    path = tmp_path / "yolo26n-cls.pt"
    path.write_bytes(b"local fake base")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: SimpleNamespace(model=TinyGray())))
    return path


def gray_data():
    levels = np.array([15, 65, 150, 230], np.uint8)
    return {"gray": np.broadcast_to(levels[:, None, None], (4, 224, 224)).copy(),
            "labels": np.arange(4, dtype=np.int64)}


def spatial_data():
    return {"images": np.ones((4, 7, 128, 128), np.float32),
            "context": np.zeros((4, 1310), np.float32),
            "available": np.array([True, True, True, False]),
            "p0": np.arange(16, dtype=np.float32).reshape(4, 4),
            "labels": np.arange(4, dtype=np.int64)}


def test_gray_conversion_and_order(subject):
    data = gray_data()["gray"]
    before = data.copy()
    tensor = subject.gray_batch(data, [3, 0], "cpu")
    assert tensor.shape == (2, 3, 224, 224) and tensor.dtype == torch.float32
    assert tensor[0, 1, 100, 100].item() == pytest.approx(230 / 255)
    assert torch.equal(tensor[:, 0], tensor[:, 2])
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("bad", [np.zeros((1, 224, 224), np.float32), np.zeros((1, 223, 224), np.uint8)])
def test_gray_conversion_refuses_wrong_contract(subject, bad):
    with pytest.raises(ValueError):
        subject.gray_batch(bad, [0], "cpu")


def test_base_is_real_yolo26_classify_and_full_model_unfrozen(subject, fake_base):
    model = subject.build_gray(fake_base, 2026, "cpu")
    assert isinstance(model, GrayBackboneAdapter)
    assert all(p.requires_grad for p in model.parameters())
    assert model.module.model[-1].linear.out_features == 4
    assert list(model.module.names.values()) == ["metal_submarine", "plastic_submarine", "plastic_fish", "real_fish"]


@pytest.mark.parametrize("task,architecture", [("detect", "yolo26n"), ("classify", "yolov8n-cls")])
def test_wrong_base_is_refused(subject, fake_base, monkeypatch, task, architecture):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: SimpleNamespace(model=TinyGray(task, architecture))))
    with pytest.raises(ValueError):
        subject.build_gray(fake_base, 2026, "cpu")


def test_missing_base_refused_before_loader(subject, tmp_path):
    with pytest.raises(FileNotFoundError):
        subject.build_gray(tmp_path / "absent.pt", 1, "cpu")


def test_spatial_head_learns_all_branches(subject):
    model = subject.Yolo26SpatialClassifier()
    out = model(torch.ones(2, 7, 128, 128), torch.ones(2, 1310))
    assert out.shape == (2, 4)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head[0].in_features == 1374


def test_spatial_pool_refuses_empty_mask(subject):
    with pytest.raises(ValueError, match="empty"):
        subject.Yolo26SpatialClassifier()(torch.zeros(1, 7, 128, 128), torch.zeros(1, 1310))


def test_gray_training_is_full_model_and_fixed_length(subject, fake_base):
    calls = []
    model, history, best = subject.fit_gray(gray_data(), gray_data(), base_path=fake_base,
                                           seed=2026, device="cpu", epochs=2, batch=4,
                                           epoch_callback=lambda r: calls.append(r["epoch"]))
    assert calls == [1, 2] and len(history) == 2
    assert model.training is False
    evidence = best["training_evidence"]
    assert evidence["all_parameters_trainable"] and evidence["optimizer_covers_all"]
    assert evidence["early_backbone_gradient_nonzero"] and evidence["early_backbone_changed"]
    assert evidence["backbone_changed_count"] > 0 and evidence["head_changed"]
    assert evidence["bn_buffers_changed_count"] > 0
    assert best["epoch"] == max(history, key=lambda r: (r["macro_f1"], r["accuracy"], -r["epoch"]))["epoch"]


def test_gray_trainer_cannot_accept_outer_eval(subject):
    assert "outer_eval" not in inspect.signature(subject.fit_gray).parameters
    assert "outer_eval" not in inspect.signature(subject.fit_spatial).parameters
    invalid = gray_data()
    invalid["outer_eval"] = gray_data()
    with pytest.raises(ValueError):
        subject.fit_gray(invalid, gray_data(), base_path="unused", seed=1, device="cpu", epochs=1)


def test_feature_hook_reconstructs_logits_and_freezes_buffers(subject, fake_base):
    model = subject.build_gray(fake_base, 2026, "cpu")
    before = subject.state_digest(model)
    g, logits, evidence = subject.extract_gray(model, gray_data()["gray"], device="cpu", batch=2)
    assert g.shape == (4, 1280) and logits.shape == (4, 4)
    assert g.dtype == logits.dtype == np.float32
    assert evidence["state_before"] == evidence["state_after"] == before
    assert evidence["state_identical"] and evidence["parameters_frozen"] and evidence["eval_mode"]
    assert evidence["logit_reconstruction_max_error"] < 1e-5
    assert not model.module.model[-1].linear._forward_pre_hooks
    assert all(not p.requires_grad for p in model.parameters())


def test_hook_is_removed_when_extraction_interrupts(subject, fake_base):
    model = subject.build_gray(fake_base, 1, "cpu")
    def stop():
        raise TimeoutError("bounded cancellation")
    with pytest.raises(TimeoutError):
        subject.extract_gray(model, gray_data()["gray"], device="cpu", check=stop)
    assert not model.module.model[-1].linear._forward_pre_hooks


def test_state_hash_includes_batchnorm_buffers(subject):
    model = TinyGray()
    before = subject.state_digest(model)
    assert subject.state_digest(model) == before
    model.model[1].running_mean.add_(1)
    assert subject.state_digest(model) != before


def test_spatial_training_omits_missing_and_uses_new_context(subject):
    from core.material_spatial_classifier import predict_with_fallback
    data = spatial_data()
    model, history, best = subject.fit_spatial(data, data, seed=2026, device="cpu", epochs=2, batch=2)
    assert len(history) == 2 and all(r["train_used"] == 3 for r in history)
    assert best["epoch"] in (1, 2)
    out = predict_with_fallback(model, data["images"], data["context"], data["available"], data["p0"], "cpu", 2)
    np.testing.assert_array_equal(out[-1], data["p0"][-1])


def test_spatial_no_available_training_is_refused(subject):
    data = spatial_data()
    data["available"][:] = False
    with pytest.raises(ValueError, match="usable"):
        subject.fit_spatial(data, data, seed=1, device="cpu", epochs=1)


def test_spatial_wrong_context_is_refused(subject):
    data = spatial_data()
    data["context"] = np.zeros((4, 286), np.float32)
    with pytest.raises(ValueError):
        subject.fit_spatial(data, spatial_data(), seed=1, device="cpu", epochs=1)


def test_gray_nonfinite_output_is_refused(subject, fake_base, monkeypatch):
    model = subject.build_gray(fake_base, 1, "cpu")
    with torch.no_grad():
        model.module.model[-1].linear.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        subject.extract_gray(model, gray_data()["gray"], device="cpu")


def test_spatial_early_stopping_selects_first_tied_epoch(subject, monkeypatch):
    monkeypatch.setattr(subject, "classification_metrics", lambda *_: {"macro_f1": .5, "accuracy": .5})
    data = spatial_data()
    _, history, best = subject.fit_spatial(data, data, seed=1, device="cpu", epochs=10, patience=2, batch=4)
    assert len(history) == 3 and best["epoch"] == 1
