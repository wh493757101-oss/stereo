"""Tests for models.classification (ClassificationModel / ClassificationResult)."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.classification import ClassificationModel, ClassificationResult


class FakeProbs:
    def __init__(self, top1=None, top1conf=None):
        self.top1 = top1
        self.top1conf = top1conf


class FakeResult:
    def __init__(self, probs):
        self.probs = probs


class FakeYOLO:
    """Stands in for ultralytics.YOLO and records predict() kwargs."""

    instances: list["FakeYOLO"] = []

    def __init__(self, weights: str):
        self.weights = weights
        self.names = {0: "metal_submarine", 1: "plastic_fish"}
        self.predict_calls: list[dict] = []
        self.result = FakeResult(FakeProbs(top1=1, top1conf=np.float32(0.8731)))
        FakeYOLO.instances.append(self)

    def predict(self, source=None, **kwargs):
        self.predict_calls.append({"source": source, **kwargs})
        return [self.result]


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """Install a fake ultralytics module so no real YOLO package is needed."""
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    FakeYOLO.instances.clear()
    return module


def test_predict_parses_top1_and_confidence(fake_ultralytics):
    model = ClassificationModel("fake_cls.pt", device="cpu")
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result = model.predict(image)

    assert isinstance(result, ClassificationResult)
    assert result.valid is True
    assert result.top1_id == 1
    assert result.top1_name == "plastic_fish"
    assert result.top1_conf == pytest.approx(0.8731, abs=1e-4)
    assert result.probs_available is True


def test_predict_passes_configured_device(fake_ultralytics):
    model = ClassificationModel("fake_cls.pt", device="cuda:0")
    model.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    yolo = FakeYOLO.instances[-1]
    assert yolo.predict_calls[0]["device"] == "cuda:0"
    assert yolo.weights == "fake_cls.pt"


def test_predict_forwards_imgsz_when_configured(fake_ultralytics):
    model = ClassificationModel("fake_cls.pt", device="cpu", imgsz=224)
    model.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    yolo = FakeYOLO.instances[-1]
    assert yolo.predict_calls[0]["imgsz"] == 224


def test_predict_omits_imgsz_by_default(fake_ultralytics):
    model = ClassificationModel("fake_cls.pt", device="cpu")
    model.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    yolo = FakeYOLO.instances[-1]
    assert "imgsz" not in yolo.predict_calls[0]


def test_predict_maps_names_when_list_shaped(fake_ultralytics):
    yolo_module = fake_ultralytics

    class ListNamesYOLO(FakeYOLO):
        def __init__(self, weights):
            super().__init__(weights)
            self.names = ["metal_submarine", "plastic_fish"]

    yolo_module.YOLO = ListNamesYOLO
    model = ClassificationModel("fake_cls.pt", device="cpu")
    result = model.predict(np.zeros((4, 4, 3), dtype=np.uint8))
    assert result.top1_name == "plastic_fish"


def test_predict_returns_empty_result_without_probs(fake_ultralytics):
    yolo_module = fake_ultralytics

    class NoProbsYOLO(FakeYOLO):
        def __init__(self, weights):
            super().__init__(weights)
            self.result = FakeResult(probs=None)

    yolo_module.YOLO = NoProbsYOLO
    model = ClassificationModel("fake_cls.pt", device="cpu")
    result = model.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    assert result.valid is False
    assert result.top1_id is None
    assert result.top1_name is None
    assert result.top1_conf is None


def test_predict_returns_empty_result_for_bad_input(fake_ultralytics):
    model = ClassificationModel("fake_cls.pt", device="cpu")
    for bad in (None, np.zeros((4, 4), dtype=np.uint8)):
        result = model.predict(bad)
        assert result.valid is False
    assert FakeYOLO.instances[-1].predict_calls == []


def test_predict_falls_back_to_class_id_name_when_unmapped(fake_ultralytics):
    yolo_module = fake_ultralytics

    class MissingNameYOLO(FakeYOLO):
        def __init__(self, weights):
            super().__init__(weights)
            self.names = {0: "metal_submarine"}  # no entry for id 1

    yolo_module.YOLO = MissingNameYOLO
    model = ClassificationModel("fake_cls.pt", device="cpu")
    result = model.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    assert result.valid is True
    assert result.top1_id == 1
    assert result.top1_name == "class_1"


def test_result_is_immutable():
    result = ClassificationResult(1, "plastic_fish", 0.5, True)
    with pytest.raises(Exception):
        result.top1_id = 2


def test_empty_factory():
    empty = ClassificationResult.empty()
    assert empty.valid is False
    assert empty.top1_id is None and empty.top1_name is None and empty.top1_conf is None
