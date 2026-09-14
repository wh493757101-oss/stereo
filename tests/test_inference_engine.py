"""Tests for gui.inference_engine (DualStageInferenceEngine).

All models, the matcher and the rectifier are faked; no Ultralytics
weights, CUDA or display are required.
"""

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.polar_compute import compute_polar_feature
from core.stereo_matching import InstanceDepth, StereoMatchResult
from models.classification import ClassificationResult
from models.segmentation import Instance

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path() -> Path:
    """Isolated work dir under the system temp (pytest basetemp under tests/
    can be locked by another process on Windows)."""
    workdir = Path(tempfile.mkdtemp(prefix="inference_engine_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _load_engine_module():
    # Import through the gui package when possible; fall back to a direct
    # file load if the gui package pulls in unavailable GUI dependencies.
    try:
        import gui.inference_engine as module

        return module
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "_engine_under_test", ROOT / "gui" / "inference_engine.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["_engine_under_test"] = module
        spec.loader.exec_module(module)
        return module


engine_module = _load_engine_module()
DualStageInferenceEngine = engine_module.DualStageInferenceEngine
resolve_device = engine_module.resolve_device

HEIGHT, WIDTH = 48, 64


def make_gray(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8)


def make_instance(inst_id=0, seed=1):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[10:30, 20:40] = 255
    return Instance(
        id=inst_id,
        bbox=(20, 10, 40, 30),
        mask=mask,
        confidence=0.9,
        class_id=0,
        class_name="object",
    )


class FakeSegmenter:
    def __init__(self, instances=None):
        self.instances = [] if instances is None else instances
        self.calls = []
        self.call_kwargs = []

    def predict(self, image, **kwargs):
        self.calls.append(image)
        self.call_kwargs.append(kwargs)
        return self.instances


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, image):
        self.calls.append(image)
        return ClassificationResult(
            top1_id=3, top1_name="metal", top1_conf=0.87, probs_available=True
        )


class NamelessFakeClassifier(FakeClassifier):
    def predict(self, image):
        self.calls.append(image)
        return ClassificationResult(
            top1_id=3, top1_name=None, top1_conf=0.87, probs_available=True
        )


class FakeMatcher:
    """Stands in for StereoMatcher with a pre-set dense result."""

    def __init__(self, valid=True, disparity=12.0):
        if valid:
            disparity_map = np.full((HEIGHT, WIDTH), disparity, dtype=np.float32)
            valid_map = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
            self.stats = InstanceDepth(0, disparity, 0.8, None, True, "ok")
        else:
            disparity_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
            valid_map = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            self.stats = InstanceDepth(0, 0.0, 0.0, None, False, "no_valid_disparity")
        self.result = StereoMatchResult(
            disparity=disparity_map,
            valid=valid_map,
            elapsed_s=0.0,
            valid_ratio=0.8 if valid else 0.0,
        )
        self.compute_calls = []
        self.stats_calls = []

    def compute(self, left, right):
        self.compute_calls.append((left, right))
        return self.result

    def instance_stats(self, result, mask, instance_id=0):
        self.stats_calls.append(instance_id)
        import dataclasses

        return dataclasses.replace(self.stats, instance_id=instance_id)

    def object_disparity_map(self, result, mask):
        filled = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        if self.stats.valid:
            filled[mask > 0] = self.stats.disparity
        return filled


class FakeRectifier:
    def __init__(self):
        self.calls = []

    def rectify(self, left, right):
        self.calls.append((left, right))
        return left, right


def make_engine(**overrides):
    kwargs = dict(
        model_a=FakeSegmenter(instances=[make_instance()]),
        model_b=FakeClassifier(),
        matcher=FakeMatcher(),
        device="cpu",
    )
    kwargs.update(overrides)
    return DualStageInferenceEngine(**kwargs), kwargs


class TestInputDimensionality:
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_accepts_gray_single_and_color(self, channels):
        engine, _ = make_engine(model_a=FakeSegmenter(instances=[make_instance()]))
        gray = make_gray(2)
        if channels == 1:
            left = right = gray
        else:
            import cv2

            code = {3: cv2.COLOR_GRAY2BGR, 4: cv2.COLOR_GRAY2BGRA}[channels]
            left = right = cv2.cvtColor(gray, code)

        instances, depths, polar = engine.process_frame(left, right)

        assert len(instances) == 1 and len(depths) == 1
        assert polar.shape == (HEIGHT, WIDTH)

    def test_accepts_hwc1(self):
        engine, _ = make_engine(model_a=FakeSegmenter(instances=[make_instance()]))
        gray = make_gray(3)
        instances, depths, _ = engine.process_frame(
            gray[:, :, None], gray[:, :, None]
        )
        assert len(depths) == 1


class TestModelAInput:
    def test_receives_3ch_grayscale_copy(self):
        import cv2

        engine, kwargs = make_engine()
        left = make_gray(4)
        engine.process_frame(left, make_gray(5))

        model_a = kwargs["model_a"]
        assert len(model_a.calls) == 1
        sent = model_a.calls[0]
        assert sent.shape == (HEIGHT, WIDTH, 3)
        np.testing.assert_array_equal(sent[:, :, 0], left)
        np.testing.assert_array_equal(sent[:, :, 1], left)
        np.testing.assert_array_equal(sent[:, :, 2], left)

    def test_predict_omits_imgsz_by_default(self):
        engine, kwargs = make_engine()
        engine.process_frame(make_gray(40), make_gray(41))
        assert kwargs["model_a"].call_kwargs == [{}]

    def test_predict_forwards_imgsz_when_configured(self):
        engine, kwargs = make_engine(model_a_imgsz=512)
        engine.process_frame(make_gray(42), make_gray(43))
        assert kwargs["model_a"].call_kwargs == [{"imgsz": 512}]


class TestNoInstanceShortCircuit:
    def test_skips_stereo_and_polar(self):
        engine, kwargs = make_engine(model_a=FakeSegmenter())
        instances, depths, polar = engine.process_frame(make_gray(6), make_gray(7))

        assert instances == [] and depths == [] and polar is None
        assert kwargs["matcher"].compute_calls == []
        assert kwargs["model_b"].calls == []


class TestSingleDenseComputation:
    def test_one_compute_reused_for_all_instances(self):
        engine, kwargs = make_engine()
        two = [make_instance(0), make_instance(1, seed=9)]
        kwargs["model_a"].instances = two

        _, depths, _ = engine.process_frame(make_gray(8), make_gray(10))

        assert len(kwargs["matcher"].compute_calls) == 1
        assert kwargs["matcher"].stats_calls == [0, 1]
        assert [d["instance_id"] for d in depths] == [0, 1]
        assert all(d["valid"] for d in depths)
        assert all(d["depth"] is not None and d["depth"] > 0 for d in depths)

    def test_one_polar_compute_reused_for_all_valid_instances(self, monkeypatch):
        first = make_instance(0)
        second_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        second_mask[15:35, 35:55] = 255
        second = Instance(
            id=1,
            bbox=(35, 15, 55, 35),
            mask=second_mask,
            confidence=0.8,
            class_id=0,
            class_name="object",
        )
        engine, kwargs = make_engine(
            model_a=FakeSegmenter(instances=[first, second]),
            model_b=None,
        )
        left, right = make_gray(80), make_gray(81)
        calls = []

        def recording_compute(left_gray, right_gray, disparity, mask=None):
            calls.append((disparity.copy(), mask.copy()))
            return compute_polar_feature(left_gray, right_gray, disparity, mask=mask)

        monkeypatch.setattr(engine_module, "compute_polar_feature", recording_compute)

        _, depths, polar = engine.process_frame(left, right)

        assert len(depths) == 2
        assert len(calls) == 1
        combined_mask = (first.mask > 0) | (second.mask > 0)
        expected_disparity = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        expected_disparity[combined_mask] = kwargs["matcher"].stats.disparity
        np.testing.assert_array_equal(calls[0][0], expected_disparity)
        np.testing.assert_array_equal(calls[0][1], combined_mask)
        expected = compute_polar_feature(
            left, right, expected_disparity, mask=combined_mask
        )
        np.testing.assert_allclose(polar, expected)

    def test_depth_record_fields(self):
        engine, _ = make_engine()
        _, depths, _ = engine.process_frame(make_gray(11), make_gray(12))
        record = depths[0]
        assert set(record) == {
            "instance_id",
            "disparity",
            "valid_ratio",
            "confidence",
            "depth",
            "valid",
            "reason",
        }
        assert record["reason"] == "ok"


class TestModelBUsage:
    def test_true_classifier_receives_gray_polar_gray_crop(self):
        engine, kwargs = make_engine()
        engine.process_frame(make_gray(13), make_gray(14))

        model_b = kwargs["model_b"]
        assert len(model_b.calls) == 1
        crop = model_b.calls[0]
        # bbox (20, 10, 40, 30) padded by 10 -> x [10, 50), y [0, 40)
        assert crop.shape == (40, 40, 3)
        np.testing.assert_array_equal(crop[:, :, 0], crop[:, :, 2])
        assert crop[:, :, 1].min() >= 0 and crop[:, :, 1].max() <= 255

        instances, _, _ = engine.process_frame(make_gray(13), make_gray(14))
        assert instances[0].class_name == "metal"
        assert instances[0].class_id == 3

    def test_classifier_not_run_without_model_b(self):
        engine, _ = make_engine(model_b=None)
        instances, _, _ = engine.process_frame(make_gray(15), make_gray(16))
        assert instances[0].class_name == "object"

    def test_missing_classifier_name_uses_stable_fallback(self):
        engine, _ = make_engine(model_b=NamelessFakeClassifier())
        instances, _, _ = engine.process_frame(make_gray(44), make_gray(45))
        assert instances[0].class_name == "class_3"


class TestModelBInputMode:
    # bbox (20, 10, 40, 30) padded by 10 -> crop x [10, 50), y [0, 40)
    CROP = (slice(0, 40), slice(10, 50))

    def test_default_is_polar_for_legacy_callers(self):
        engine, _ = make_engine()
        assert engine.model_b_input_mode == "polar"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="model_b_input_mode"):
            make_engine(model_b_input_mode="rgb")

    def test_polar_mode_exact_channel_content(self):
        engine, kwargs = make_engine(model_b_input_mode="polar")
        left, right = make_gray(70), make_gray(71)
        engine.process_frame(left, right)

        crop = kwargs["model_b"].calls[0]
        assert crop.shape == (40, 40, 3)
        gray_crop = left[self.CROP]
        object_disp = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        object_disp[make_instance().mask > 0] = 12.0
        polar = compute_polar_feature(left, right, object_disp, mask=make_instance().mask)
        expected_polar_u8 = np.rint(
            np.clip(polar[self.CROP], 0.0, 1.0) * 255
        ).astype(np.uint8)

        np.testing.assert_array_equal(crop[:, :, 0], gray_crop)
        np.testing.assert_array_equal(crop[:, :, 2], gray_crop)
        np.testing.assert_array_equal(crop[:, :, 1], expected_polar_u8)

    def test_gray_mode_exact_channel_content(self):
        engine, kwargs = make_engine(model_b_input_mode="gray")
        left = make_gray(72)
        engine.process_frame(left, make_gray(73))

        crop = kwargs["model_b"].calls[0]
        assert crop.shape == (40, 40, 3)
        gray_crop = left[self.CROP]
        for channel in range(3):
            np.testing.assert_array_equal(crop[:, :, channel], gray_crop)

    def test_gray_mode_preserves_polar_map_and_depth(self):
        engine, kwargs = make_engine(model_b_input_mode="gray")
        _, depths, polar = engine.process_frame(make_gray(74), make_gray(75))
        assert depths[0]["valid"] is True and depths[0]["depth"] > 0
        assert polar.shape == (HEIGHT, WIDTH)
        assert np.any(polar > 0)
        assert len(kwargs["matcher"].compute_calls) == 1


class TestInvalidStereo:
    def test_invalid_stereo_gives_zero_polar_and_invalid_depth(self):
        engine, kwargs = make_engine(matcher=FakeMatcher(valid=False))
        _, depths, polar = engine.process_frame(make_gray(17), make_gray(18))

        assert depths[0]["valid"] is False
        assert depths[0]["depth"] is None
        assert depths[0]["reason"] == "no_valid_disparity"
        np.testing.assert_array_equal(polar, np.zeros((HEIGHT, WIDTH), np.float32))

    def test_low_valid_ratio_is_invalid_despite_positive_object_map(self):
        """A low_valid_ratio result carries a positive robust disparity and
        object map, but is invalid: no depth value and all-zero polar."""
        low_ratio_stats = InstanceDepth(0, 12.0, 0.02, None, False, "low_valid_ratio")

        class LowRatioMatcher(FakeMatcher):
            def instance_stats(self, result, mask, instance_id=0):
                self.stats_calls.append(instance_id)
                import dataclasses

                return dataclasses.replace(low_ratio_stats, instance_id=instance_id)

            def object_disparity_map(self, result, mask):
                filled = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
                filled[mask > 0] = 12.0  # positive, as instance_stats saw valid px
                return filled

        engine, _ = make_engine(matcher=LowRatioMatcher())
        _, depths, polar = engine.process_frame(make_gray(29), make_gray(30))

        assert depths[0]["valid"] is False
        assert depths[0]["depth"] is None
        assert depths[0]["reason"] == "low_valid_ratio"
        assert depths[0]["disparity"] == pytest.approx(12.0)
        np.testing.assert_array_equal(polar, np.zeros((HEIGHT, WIDTH), np.float32))


class TestSyncSkew:
    def test_exceeded_skew_skips_stereo_and_invalidates_depth(self):
        engine, kwargs = make_engine()
        instances, depths, polar = engine.process_frame(
            make_gray(19), make_gray(20), sync_skew_ms=5.0
        )

        assert kwargs["matcher"].compute_calls == []
        assert len(depths) == 1
        assert depths[0]["valid"] is False
        assert depths[0]["depth"] is None
        assert depths[0]["reason"] == "sync_skew_exceeded"
        np.testing.assert_array_equal(polar, np.zeros((HEIGHT, WIDTH), np.float32))
        # Model A and optional gray/zero-polar Model B still run.
        assert len(kwargs["model_a"].calls) == 1
        assert len(kwargs["model_b"].calls) == 1

    def test_within_skew_runs_stereo(self):
        engine, kwargs = make_engine()
        engine.process_frame(make_gray(21), make_gray(22), sync_skew_ms=1.0)
        assert len(kwargs["matcher"].compute_calls) == 1

    def test_negative_skew_is_normalized_to_absolute(self):
        engine, kwargs = make_engine(max_sync_skew_ms=0.5)
        _, depths, polar = engine.process_frame(
            make_gray(31), make_gray(32), sync_skew_ms=-5.0
        )
        assert depths[0]["reason"] == "sync_skew_exceeded"
        assert depths[0]["valid"] is False
        np.testing.assert_array_equal(polar, np.zeros((HEIGHT, WIDTH), np.float32))
        assert kwargs["matcher"].compute_calls == []

    def test_default_uses_configured_max(self):
        engine, kwargs = make_engine(max_sync_skew_ms=0.5)
        _, depths, _ = engine.process_frame(
            make_gray(23), make_gray(24), sync_skew_ms=0.75
        )
        assert depths[0]["reason"] == "sync_skew_exceeded"


class TestRectificationWiring:
    def test_rectifier_applied_when_enabled(self):
        rectifier = FakeRectifier()
        engine, _ = make_engine(rectifier=rectifier, rectify_enabled=True)
        left, right = make_gray(25), make_gray(26)
        engine.process_frame(left, right)

        assert rectifier.calls == [(left, right)]

    def test_rectifier_not_applied_when_disabled(self):
        rectifier = FakeRectifier()
        engine, _ = make_engine(rectifier=rectifier, rectify_enabled=False)
        engine.process_frame(make_gray(27), make_gray(28))
        assert rectifier.calls == []

    def test_enabled_without_rectifier_raises(self):
        with pytest.raises(ValueError, match="rectifier"):
            make_engine(rectify_enabled=True)


class TestDeviceResolution:
    def test_auto_selects_cuda0_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == "0"

    def test_auto_selects_cpu_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto") == "cpu"

    def test_explicit_cuda_fails_clearly_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
            resolve_device("cuda")

    def test_explicit_cuda_ok_with_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("cuda") == "0"

    def test_cpu_passes_through(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("cpu") == "cpu"

    def test_numeric_id_is_explicit_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("0") == "0"
        assert resolve_device("1") == "1"

    def test_numeric_id_raises_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
            resolve_device("1")

    def test_cuda_index_raises_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
            resolve_device("cuda:1")

    def test_cuda_index_kept_with_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("cuda:1") == "cuda:1"

    def test_engine_uses_resolved_device(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        engine, _ = make_engine(device="auto")
        assert engine.device == "cpu"


class TestFromConfig:
    @pytest.fixture
    def calib_npz(self, tmp_path):
        k = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
        path = tmp_path / "stereo_calib.npz"
        np.savez(
            str(path),
            K1=k,
            D1=np.zeros(5),
            K2=k,
            D2=np.zeros(5),
            R=np.eye(3),
            T=np.array([[0.05], [0.0], [0.0]]),
            image_size=np.array([WIDTH, HEIGHT]),
        )
        return path

    @pytest.fixture
    def fake_model_classes(self, monkeypatch):
        created = {}

        class FakeSeg:
            def __init__(self, **kwargs):
                created["model_a"] = kwargs

            def predict(self, image, **kwargs):
                return []

        class FakeCls:
            def __init__(self, **kwargs):
                created["model_b"] = kwargs

            def predict(self, image):
                return ClassificationResult.empty()

        monkeypatch.setattr(engine_module, "SegmentationModel", FakeSeg)
        monkeypatch.setattr(engine_module, "ClassificationModel", FakeCls)
        return created

    def make_config(self, tmp_path, calib_npz):
        config = tmp_path / "engine.yaml"
        config.write_text(
            """
model_a:
  path: "runs/train/model_a/weights/best.pt"
  conf_threshold: 0.3
  iou_threshold: 0.55
  imgsz: 512
model_b:
  path: "runs/train/model_b-polar/weights/best.pt"
  imgsz: 224
  input_mode: "gray"
calibration:
  file: "STUB_CALIB"
  r_convention: "matlab"
  baseline: 0.11
  focal_length: 3600.0
rectification:
  enabled: true
  alpha: 0.0
stereo:
  matcher: "sgbm"
  mode: "hh"
  max_disp: 256
  scale: 0.5
  block_size: 9
  min_valid_ratio: 0.1
runtime:
  device: "cpu"
  crop_padding: 12
  sync_skew_ms: 3.5
""".replace("STUB_CALIB", str(calib_npz).replace("\\", "/")),
            encoding="utf-8",
        )
        return config

    def test_config_wiring(self, tmp_path, calib_npz, fake_model_classes):
        config = self.make_config(tmp_path, calib_npz)
        engine = DualStageInferenceEngine.from_config(config)

        model_a_kwargs = fake_model_classes["model_a"]
        model_b_kwargs = fake_model_classes["model_b"]

        assert model_a_kwargs["model_path"] == (
            ROOT / "runs/train/model_a/weights/best.pt"
        )
        assert model_a_kwargs["conf_threshold"] == pytest.approx(0.3)
        assert model_a_kwargs["iou_threshold"] == pytest.approx(0.55)
        assert engine.model_a_imgsz == 512
        assert model_b_kwargs["imgsz"] == 224
        assert engine.model_b_input_mode == "gray"
        assert engine.baseline == pytest.approx(0.11)
        assert engine.focal_length == pytest.approx(3600.0)
        assert engine.crop_padding == 12
        assert engine.max_sync_skew_ms == pytest.approx(3.5)
        assert engine.rectify_enabled is True
        assert engine.rectifier is not None
        assert engine.rectifier.image_size == (WIDTH, HEIGHT)
        assert engine.matcher.config.max_disparity == 256
        assert engine.matcher.config.mode == "hh"
        assert engine.matcher.config.scale == pytest.approx(0.5)
        assert engine.matcher.config.block_size == 9
        assert engine.matcher.config.min_valid_ratio == pytest.approx(0.1)

    def test_rectification_disabled_builds_no_rectifier(
        self, tmp_path, calib_npz, fake_model_classes
    ):
        config = self.make_config(tmp_path, calib_npz)
        text = config.read_text(encoding="utf-8").replace("enabled: true", "enabled: false")
        config.write_text(text, encoding="utf-8")

        engine = DualStageInferenceEngine.from_config(config)
        assert engine.rectifier is None
        assert engine.rectify_enabled is False

    def test_overrides_apply(self, tmp_path, calib_npz, fake_model_classes):
        config = self.make_config(tmp_path, calib_npz)
        engine = DualStageInferenceEngine.from_config(config, crop_padding=5)
        assert engine.crop_padding == 5

    def test_input_mode_defaults_to_polar_when_absent(
        self, tmp_path, calib_npz, fake_model_classes
    ):
        config = self.make_config(tmp_path, calib_npz)
        text = config.read_text(encoding="utf-8").replace('  input_mode: "gray"\n', "")
        config.write_text(text, encoding="utf-8")

        engine = DualStageInferenceEngine.from_config(config)
        assert engine.model_b_input_mode == "polar"

    def test_missing_calib_file_raises(self, tmp_path, fake_model_classes):
        config = tmp_path / "engine.yaml"
        config.write_text(
            """
model_a:
  path: "a.pt"
model_b:
  path: "b.pt"
calibration:
  file: "does/not/exist.npz"
rectification:
  enabled: true
""",
            encoding="utf-8",
        )
        with pytest.raises(FileNotFoundError):
            DualStageInferenceEngine.from_config(config)


class TestDetailedResult:
    def test_process_frame_tuple_callers_unchanged(self):
        engine, _ = make_engine()
        out = engine.process_frame(make_gray(50), make_gray(51))
        assert isinstance(out, tuple) and len(out) == 3
        instances, depths, polar = out
        assert len(instances) == 1 and len(depths) == 1
        assert polar.shape == (HEIGHT, WIDTH)

    def test_detailed_result_contains_rectified_grays_and_state(self):
        engine, _ = make_engine()
        left, right = make_gray(52), make_gray(53)
        result = engine.process_frame_detailed(left, right)

        assert isinstance(result, engine_module.DetailedInferenceResult)
        np.testing.assert_array_equal(result.left_gray, left)
        np.testing.assert_array_equal(result.right_gray, right)
        assert len(result.instances) == 1
        assert len(result.depths) == 1
        assert result.polar_map is not None
        assert result.polar_map.shape == (HEIGHT, WIDTH)
        assert result.already_rectified is False

    def test_detailed_result_grays_are_normalized_gray_u8(self):
        import cv2

        engine, _ = make_engine()
        bgr = cv2.cvtColor(make_gray(54), cv2.COLOR_GRAY2BGR)
        result = engine.process_frame_detailed(bgr, bgr)
        assert result.left_gray.ndim == 2
        assert result.left_gray.dtype == np.uint8

    def test_detailed_result_short_circuit_no_instances(self):
        engine, _ = make_engine(model_a=FakeSegmenter())
        result = engine.process_frame_detailed(make_gray(55), make_gray(56))
        assert result.instances == []
        assert result.depths == []
        assert result.polar_map is None
        assert result.left_gray.shape == (HEIGHT, WIDTH)


class TestDetailedSyncSkewSemantics:
    def test_unavailable_skew_echoed_as_none_and_processed(self):
        engine, kwargs = make_engine()
        result = engine.process_frame_detailed(
            make_gray(57), make_gray(58), sync_skew_ms=None
        )
        assert result.sync_skew_ms is None
        # Unavailable skew is processed (stereo runs) but is distinguishable
        # from a measured zero.
        assert len(kwargs["matcher"].compute_calls) == 1

    def test_measured_zero_echoed_as_zero(self):
        engine, _ = make_engine()
        result = engine.process_frame_detailed(
            make_gray(59), make_gray(60), sync_skew_ms=0.0
        )
        assert result.sync_skew_ms == 0.0

    def test_exceeded_skew_echoed_in_detailed_result(self):
        engine, _ = make_engine()
        result = engine.process_frame_detailed(
            make_gray(61), make_gray(62), sync_skew_ms=5.0
        )
        assert result.sync_skew_ms == 5.0
        assert result.depths[0]["reason"] == "sync_skew_exceeded"


class TestAlreadyRectified:
    def test_already_rectified_skips_online_rectification(self):
        rectifier = FakeRectifier()
        engine, _ = make_engine(rectifier=rectifier, rectify_enabled=True)
        engine.process_frame_detailed(
            make_gray(63), make_gray(64), already_rectified=True
        )
        assert rectifier.calls == []

    def test_not_already_rectified_uses_rectifier(self):
        rectifier = FakeRectifier()
        engine, _ = make_engine(rectifier=rectifier, rectify_enabled=True)
        engine.process_frame_detailed(make_gray(65), make_gray(66))
        assert len(rectifier.calls) == 1

    def test_process_frame_forwards_already_rectified(self):
        rectifier = FakeRectifier()
        engine, _ = make_engine(rectifier=rectifier, rectify_enabled=True)
        engine.process_frame(make_gray(67), make_gray(68), already_rectified=True)
        assert rectifier.calls == []
