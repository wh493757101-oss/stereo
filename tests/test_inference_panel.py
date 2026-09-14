"""Headless/offscreen tests for gui.inference_panel and gui.capture_panel.

Covers: config-driven UI defaults, latest-frame worker replacement,
synchronous-free _on_frame, detailed-result formatting (depth None /
unavailable sync), QImage memory detachment, worker/panel lifecycle,
safe camera rebinding without disconnect warnings, capture-panel trigger
controls and the compact 2x2 layout. No real weights, cameras or MVS
SDK required.
"""

import os
import shutil
import tempfile
import threading
import time
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import gui.capture_panel as cp
import gui.inference_panel as ip
from gui.camera_thread import CameraConfig, CameraThread, FrameBundle
from gui.capture_panel import ImageView
from gui.inference_engine import DetailedInferenceResult
from core.stereo_matching import StereoMatcher, StereoMatcherConfig, to_gray_u8

_APP = QApplication.instance() or QApplication([])

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path() -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="inference_panel_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture
def camera():
    thread = CameraThread(source=None)  # mock config, never started
    yield thread
    thread._running = False


def make_bundle(value: int = 0, pair_id: int = 0, sync_skew_ms=None, rectified=False):
    left = np.full((16, 16), value, np.uint8)
    right = np.full((16, 16), value, np.uint8)
    return FrameBundle(
        left=left,
        right=right,
        pair_id=pair_id,
        sync_skew_ms=sync_skew_ms,
        rectified=rectified,
        timestamp="2026-01-01T00:00:00",
    )


class BlockingEngine:
    """Fake engine: blocks inside inference until released."""

    def __init__(self) -> None:
        self.calls = []          # (left_array, thread_id, sync_skew_ms, already_rectified)
        self.started = threading.Event()
        self.release = threading.Event()
        self.matcher = StereoMatcher(StereoMatcherConfig(max_disparity=64, block_size=7))
        self.max_sync_skew_ms = 2.0

    def process_frame_detailed(self, left, right, sync_skew_ms=None, already_rectified=False):
        self.calls.append((left.copy(), threading.get_ident(), sync_skew_ms, already_rectified))
        self.started.set()
        self.release.wait(timeout=5.0)
        return DetailedInferenceResult(
            left_gray=to_gray_u8(left),
            right_gray=to_gray_u8(right),
            sync_skew_ms=sync_skew_ms,
            already_rectified=already_rectified,
        )


class TestConfigDefaults:
    def test_defaults_loaded_from_yaml(self):
        defaults = ip._load_gui_defaults()
        assert defaults["model_a_path"] == "runs/train/run_20260913_initial/model_a/weights/best.pt"
        assert defaults["model_b_path"] == "runs/train/run_20260913_initial/model_b-gray/weights/best.pt"
        assert defaults["model_b_input_mode"] == "gray"
        assert defaults["baseline"] == pytest.approx(0.09890970798524992)
        assert defaults["focal_length"] == pytest.approx(3643.5231322766995)
        assert defaults["max_disp"] == 768
        assert defaults["block_size"] == 7
        assert defaults["sync_skew_ms"] == pytest.approx(2.0)

    def test_defaults_fall_back_without_config_file(self, tmp_path):
        defaults = ip._load_gui_defaults(tmp_path / "missing.yaml")
        assert defaults["model_b_path"] == "runs/train/run_20260913_initial/model_b-gray/weights/best.pt"
        assert defaults["model_b_input_mode"] == "gray"
        assert defaults["max_disp"] == 768
        assert defaults["block_size"] == 7
        assert defaults["sync_skew_ms"] == pytest.approx(2.0)

    def test_panel_ui_initialized_from_defaults(self, camera):
        panel = ip.InferencePanel(camera)
        assert panel.model_a_edit.text() == "runs/train/run_20260913_initial/model_a/weights/best.pt"
        assert panel.model_b_edit.text() == "runs/train/run_20260913_initial/model_b-gray/weights/best.pt"
        assert panel.model_b_mode_combo.currentData() == "gray"
        assert float(panel.baseline_edit.text()) == pytest.approx(0.09890970798524992, abs=1e-5)
        assert float(panel.focal_edit.text()) == pytest.approx(3643.5231322766995, abs=0.01)
        assert panel.max_disp_spin.value() == 768
        assert panel.block_size_spin.value() == 7

    def test_max_disp_spin_supports_768_and_1024(self, camera):
        panel = ip.InferencePanel(camera)
        assert panel.max_disp_spin.maximum() >= 1024
        panel.max_disp_spin.setValue(768)
        assert panel.max_disp_spin.value() == 768
        panel.max_disp_spin.setValue(1024)
        assert panel.max_disp_spin.value() == 1024

    def test_block_size_spin_enforces_odd_values(self, camera):
        panel = ip.InferencePanel(camera)
        assert panel.block_size_spin.minimum() >= 3
        # Stepping through the full range never produces an even value.
        values = set()
        panel.block_size_spin.setMinimum(3)
        panel.block_size_spin.setMaximum(21)
        panel.block_size_spin.setValue(3)
        for _ in range(10):
            values.add(panel.block_size_spin.value())
            panel.block_size_spin.stepUp()
        values.add(panel.block_size_spin.value())
        assert all(v % 2 == 1 for v in values)


class TestEngineInit:
    def _patch_from_config(self, monkeypatch, engine):
        monkeypatch.setattr(
            ip.DualStageInferenceEngine,
            "from_config",
            classmethod(lambda cls, *a, **k: engine),
        )

    def test_init_engine_builds_worker_and_applies_ui_overrides(self, camera, monkeypatch):
        engine = BlockingEngine()
        self._patch_from_config(monkeypatch, engine)
        panel = ip.InferencePanel(camera)
        try:
            assert panel.init_engine() is True
            assert panel._running is True
            assert panel._worker is not None
            # UI stereo overrides replace the config's matcher.
            assert panel._engine.matcher.config.max_disparity == 768
            assert panel._engine.matcher.config.block_size == 7
        finally:
            engine.release.set()
            panel.deinit_engine()
        assert panel._worker is None
        assert panel._running is False

    def test_init_engine_rejects_missing_model_path(self, camera):
        panel = ip.InferencePanel(camera)
        panel.model_a_edit.setText("")
        assert panel.init_engine() is False
        assert panel._running is False


class TestModelBInputMode:
    def test_combo_offers_only_valid_modes(self, camera):
        panel = ip.InferencePanel(camera)
        modes = {
            panel.model_b_mode_combo.itemData(i)
            for i in range(panel.model_b_mode_combo.count())
        }
        assert modes == {"gray", "polar"}

    def test_section_title_is_mode_neutral(self, camera):
        panel = ip.InferencePanel(camera)
        title = panel.model_b_mode_combo.parentWidget().title()
        assert "Model B" in title
        assert "polar" not in title.lower()

    def test_init_engine_forwards_path_and_mode(self, camera, monkeypatch):
        engine = BlockingEngine()
        captured = {}

        def fake_from_config(cls, config_path=None, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return engine

        monkeypatch.setattr(
            ip.DualStageInferenceEngine, "from_config", classmethod(fake_from_config)
        )
        panel = ip.InferencePanel(camera)
        try:
            assert panel.init_engine() is True
            assert captured["model_b_input_mode"] == "gray"
            assert Path(captured["model_b_path"]).as_posix().endswith(
                "runs/train/run_20260913_initial/model_b-gray/weights/best.pt"
            )
        finally:
            engine.release.set()
            panel.deinit_engine()

        # Switching the combo to polar is forwarded on the next engine build.
        panel.model_b_mode_combo.setCurrentIndex(
            panel.model_b_mode_combo.findData("polar")
        )
        panel.model_b_edit.setText("runs/train/run_20260913_initial/model_b-polar/weights/best.pt")
        try:
            assert panel.init_engine() is True
            assert captured["model_b_input_mode"] == "polar"
            assert Path(captured["model_b_path"]).as_posix().endswith(
                "runs/train/run_20260913_initial/model_b-polar/weights/best.pt"
            )
        finally:
            engine.release.set()
            panel.deinit_engine()


class TestWorkerLatestFrame:
    def test_busy_worker_replaces_pending_frame(self):
        engine = BlockingEngine()
        worker = ip.InferenceWorker(engine)
        worker.start()
        try:
            first = make_bundle(value=1)
            second = make_bundle(value=2)
            third = make_bundle(value=3)

            worker.submit(first)
            assert engine.started.wait(5.0)  # worker busy on `first`
            worker.submit(second)            # pending
            worker.submit(third)             # replaces second before pickup
            # Give the worker a moment to consume pending (must pick third).
            engine.release.set()

            # Wait until the worker records the second call.
            for _ in range(200):
                if len(engine.calls) >= 2:
                    break
                time.sleep(0.01)
            assert len(engine.calls) == 2
            assert engine.calls[0][0][0, 0] == 1
            assert engine.calls[1][0][0, 0] == 3  # stale second dropped
        finally:
            engine.release.set()
            worker.stop()

    def test_queue_capacity_is_one_never_grows(self):
        engine = BlockingEngine()
        worker = ip.InferenceWorker(engine)
        worker.start()
        try:
            worker.submit(make_bundle(value=1))
            assert engine.started.wait(5.0)
            for k in range(20):
                worker.submit(make_bundle(value=10 + k))
            engine.release.set()
            for _ in range(200):
                if len(engine.calls) >= 2:
                    break
                time.sleep(0.01)
            # Capacity-1 slot: only one additional frame processed.
            assert len(engine.calls) == 2
            assert engine.calls[1][0][0, 0] == 29  # last submitted wins
        finally:
            engine.release.set()
            worker.stop()

    def test_worker_passes_sync_and_rectified_to_engine(self):
        engine = BlockingEngine()
        worker = ip.InferenceWorker(engine)
        worker.start()
        try:
            worker.submit(make_bundle(value=7, sync_skew_ms=0.42, rectified=True))
            assert engine.started.wait(5.0)
            engine.release.set()
            for _ in range(200):
                if len(engine.calls) >= 1 and engine.calls[-1][2] == 0.42:
                    break
                time.sleep(0.01)
            left, _tid, sync, rect = engine.calls[-1]
            assert sync == pytest.approx(0.42)
            assert rect is True
        finally:
            engine.release.set()
            worker.stop()


class TestPanelFrameFlow:
    def _make_panel(self, camera, monkeypatch):
        engine = BlockingEngine()
        monkeypatch.setattr(
            ip.DualStageInferenceEngine,
            "from_config",
            classmethod(lambda cls, *a, **k: engine),
        )
        panel = ip.InferencePanel(camera)
        assert panel.init_engine() is True
        return panel, engine

    def test_on_frame_never_runs_inference_on_gui_thread(self, camera, monkeypatch):
        panel, engine = self._make_panel(camera, monkeypatch)
        gui_thread_id = threading.get_ident()
        try:
            panel._on_frame(make_bundle(value=5))
            assert engine.started.wait(5.0)
            engine.release.set()
            for _ in range(200):
                if engine.calls:
                    break
                time.sleep(0.01)
            assert engine.calls, "worker never picked up submitted frame"
            # Inference must have run on the worker thread, not the caller.
            assert all(tid != gui_thread_id for _, tid, _, _ in engine.calls)
        finally:
            engine.release.set()
            panel.deinit_engine()

    def test_on_frame_skips_incomplete_pairs(self, camera, monkeypatch):
        panel, engine = self._make_panel(camera, monkeypatch)
        try:
            bundle = make_bundle(value=1)
            bundle.left = None
            panel._on_frame(bundle)
            assert not engine.calls
        finally:
            engine.release.set()
            panel.deinit_engine()


class TestResultFormatting:
    def test_format_depth_none_is_unavailable(self):
        assert ip._format_depth(None) == "unavailable"
        assert ip._format_depth(1.234) == "1.23m"

    def test_format_depths_valid_and_invalid(self):
        depths = [
            {"instance_id": 0, "valid": True, "depth": 1.234},
            {"instance_id": 1, "valid": False, "depth": None, "reason": "sync_skew_exceeded"},
        ]
        text = ip._format_depths(depths)
        assert "#0: 1.23m" in text
        assert "#1: N/A (sync_skew_exceeded)" in text

    def test_format_depths_empty(self):
        assert ip._format_depths([]) == "--"

    def test_result_never_formats_none_depth_as_number(self, camera):
        panel = ip.InferencePanel(camera)
        result = DetailedInferenceResult(
            left_gray=np.zeros((8, 8), np.uint8),
            right_gray=np.zeros((8, 8), np.uint8),
            depths=[{"instance_id": 0, "valid": False, "depth": None,
                     "reason": "no_valid_disparity", "disparity": 0.0,
                     "valid_ratio": 0.0, "confidence": 0.0}],
            sync_skew_ms=None,
        )
        panel._on_inference_result({"bundle": make_bundle(), "result": result})
        depth_text = panel.depth_label.text()
        assert "None" not in depth_text
        assert "unavailable" in depth_text or "N/A" in depth_text

    def test_unavailable_sync_reported_as_unavailable(self, camera):
        panel = ip.InferencePanel(camera)
        result = DetailedInferenceResult(
            left_gray=np.zeros((8, 8), np.uint8),
            right_gray=np.zeros((8, 8), np.uint8),
            sync_skew_ms=None,
        )
        panel._on_inference_result({"bundle": make_bundle(), "result": result})
        assert "不可用" in panel.sync_label.text()

    def test_measured_zero_sync_reported_as_measured(self, camera):
        panel = ip.InferencePanel(camera)
        result = DetailedInferenceResult(
            left_gray=np.zeros((8, 8), np.uint8),
            right_gray=np.zeros((8, 8), np.uint8),
            sync_skew_ms=0.0,
        )
        panel._on_inference_result({"bundle": make_bundle(), "result": result})
        assert "0.00" in panel.sync_label.text()
        assert "不可用" not in panel.sync_label.text()

    def test_rectified_note_updates(self, camera):
        panel = ip.InferencePanel(camera)
        result = DetailedInferenceResult(
            left_gray=np.zeros((8, 8), np.uint8),
            right_gray=np.zeros((8, 8), np.uint8),
            already_rectified=True,
        )
        panel._on_inference_result(
            {"bundle": make_bundle(rectified=True), "result": result}
        )
        assert "已校正" in panel.rectified_note.text()


class TestQImageSafety:
    def test_pixmap_survives_source_mutation(self):
        img = np.zeros((4, 4), np.uint8)
        pixmap = ImageView._to_pixmap(img)
        img[:] = 255
        qimage = pixmap.toImage()
        assert qimage.pixelColor(0, 0).red() == 0

    def test_bgr_input_converted(self):
        bgr = np.zeros((4, 4, 3), np.uint8)
        bgr[:, :, 2] = 200  # red channel in BGR
        pixmap = ImageView._to_pixmap(bgr)
        assert pixmap.width() == 4 and pixmap.height() == 4

    def test_gray_input_accepted(self):
        gray = np.full((4, 4), 128, np.uint8)
        pixmap = ImageView._to_pixmap(gray)
        assert pixmap.width() == 4 and pixmap.height() == 4


class TestDisplayRefresh:
    def test_refresh_display_uses_detailed_result(self, camera):
        panel = ip.InferencePanel(camera)
        left = np.full((32, 32), 100, np.uint8)
        right = np.full((32, 32), 150, np.uint8)
        polar = np.full((32, 32), 0.5, np.float32)
        result = DetailedInferenceResult(
            left_gray=left,
            right_gray=right,
            polar_map=polar,
        )
        panel._on_inference_result({"bundle": make_bundle(), "result": result})
        assert panel.left_view.pixmap() is not None
        assert panel.right_view.pixmap() is not None
        assert panel.polar_view.pixmap() is not None


# ---------------------------------------------------------------------------
# Worker / panel lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_stop_waits_until_thread_finished(self):
        engine = BlockingEngine()
        worker = ip.InferenceWorker(engine)
        worker.start()
        worker.submit(make_bundle(value=1))
        assert engine.started.wait(5.0)
        engine.release.set()
        worker.stop()
        assert not worker.isRunning()
        assert worker.isFinished()

    def test_deinit_engine_waits_for_worker_thread(self, camera, monkeypatch):
        engine = BlockingEngine()
        monkeypatch.setattr(
            ip.DualStageInferenceEngine,
            "from_config",
            classmethod(lambda cls, *a, **k: engine),
        )
        panel = ip.InferencePanel(camera)
        assert panel.init_engine() is True
        worker = panel._worker
        assert worker is not None
        worker.submit(make_bundle(value=2))
        assert engine.started.wait(5.0)
        engine.release.set()
        panel.deinit_engine()
        assert not worker.isRunning()
        assert worker.isFinished()
        assert panel._worker is None

    def test_late_result_after_deinit_is_ignored(self, camera, monkeypatch):
        engine = BlockingEngine()
        monkeypatch.setattr(
            ip.DualStageInferenceEngine,
            "from_config",
            classmethod(lambda cls, *a, **k: engine),
        )
        panel = ip.InferencePanel(camera)
        assert panel.init_engine() is True
        stale_generation = panel._engine_generation
        engine.release.set()
        panel.deinit_engine()

        result = DetailedInferenceResult(
            left_gray=np.zeros((8, 8), np.uint8),
            right_gray=np.zeros((8, 8), np.uint8),
            depths=[{"instance_id": 0, "valid": True, "depth": 1.5,
                     "disparity": 2.0, "valid_ratio": 1.0, "confidence": 1.0}],
        )
        panel._on_inference_result({
            "generation": stale_generation,
            "bundle": make_bundle(),
            "result": result,
        })
        assert panel.depth_label.text() == "距离: --"
        assert panel._last_result is None
        assert panel._last_bundle is None


# ---------------------------------------------------------------------------
# Safe camera rebinding
# ---------------------------------------------------------------------------

def _disconnect_warnings(caught):
    return [w for w in caught if "disconnect" in str(w.message).lower()]


class TestCameraRebinding:
    def test_set_camera_emits_no_disconnect_warnings(self, camera):
        panel = ip.InferencePanel(camera)
        new_camera = CameraThread(source=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            panel.set_camera(new_camera)
        assert not _disconnect_warnings(caught)
        assert panel._camera is new_camera
        assert panel._wired_camera is new_camera

    def test_old_camera_has_no_callback_after_rebind(self, camera):
        panel = ip.InferencePanel(camera)
        new_camera = CameraThread(source=None)
        panel.set_camera(new_camera)

        camera.frame_ready.emit(make_bundle(value=3))
        assert panel._last_bundle is None

        new_camera.frame_ready.emit(make_bundle(value=4))
        assert panel._last_bundle is not None
        assert panel._last_bundle.left[0, 0] == 4

    def test_set_camera_same_camera_is_noop(self, camera):
        panel = ip.InferencePanel(camera)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            panel.set_camera(camera)
        assert not _disconnect_warnings(caught)
        camera.frame_ready.emit(make_bundle(value=5))
        assert panel._last_bundle is not None

    def test_capture_panel_set_camera_rebinds_cleanly(self, camera):
        panel = cp.CapturePanel(camera)
        new_camera = CameraThread(source=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            panel.set_camera(new_camera)
        assert not _disconnect_warnings(caught)

        camera.frame_ready.emit(make_bundle(value=1))
        assert panel._latest_frame is None

        new_camera.frame_ready.emit(make_bundle(value=2))
        assert panel._latest_frame is not None
        assert panel._latest_frame.left[0, 0] == 2

    def test_capture_panel_buttons_wired_once(self, camera):
        panel = cp.CapturePanel(camera)
        # The same action must not fire twice per click after a rebind.
        panel.set_camera(CameraThread(source=None))
        clicks = []
        panel._on_capture = lambda: clicks.append(1)  # noqa: F841 (sanity only)
        receivers = panel.capture_btn.receivers("clicked()")
        assert receivers <= 1

    def test_main_window_panels_use_safe_rebinding(self):
        from gui.main_window import MainWindow

        window = MainWindow()
        assert window._capture_panel._wired_camera is window._camera
        assert window._inference_panel._wired_camera is window._camera
        assert window._capture_panel._camera is window._camera
        assert window._inference_panel._camera is window._camera


# ---------------------------------------------------------------------------
# Compact 2x2 layout
# ---------------------------------------------------------------------------

class TestLayout:
    def test_minimum_size_fits_1500x900_window(self, camera):
        panel = ip.InferencePanel(camera)
        min_hint = panel.minimumSizeHint()
        assert min_hint.width() <= 1500
        assert min_hint.height() <= 900

    def test_views_have_stable_compact_minimums(self, camera):
        panel = ip.InferencePanel(camera)
        for view in (panel.left_view, panel.right_view, panel.polar_view):
            assert view.minimumWidth() <= 420
            assert view.minimumHeight() <= 320


# ---------------------------------------------------------------------------
# Capture panel trigger / sync controls
# ---------------------------------------------------------------------------

class TestCapturePanelControls:
    def test_read_config_defaults(self, camera):
        panel = cp.CapturePanel(camera)
        cfg = panel._read_config()
        assert cfg.trigger_source == "software"
        assert cfg.max_sync_skew_ms == pytest.approx(2.0)
        assert cfg.already_rectified is False

    def test_controls_update_config(self, camera):
        panel = cp.CapturePanel(camera)
        panel.trigger_source_combo.setCurrentIndex(1)  # hardware
        panel.max_skew_spin.setValue(5.5)
        panel.already_rectified_check.setChecked(True)
        cfg = panel._read_config()
        assert cfg.trigger_source == "hardware"
        assert cfg.max_sync_skew_ms == pytest.approx(5.5)
        assert cfg.already_rectified is True

    def test_trigger_combo_only_offers_valid_sources(self, camera):
        panel = cp.CapturePanel(camera)
        for i in range(panel.trigger_source_combo.count()):
            assert panel.trigger_source_combo.itemData(i) in ("software", "hardware")


# ---------------------------------------------------------------------------
# Engine uses the camera acquisition config
# ---------------------------------------------------------------------------

class TestEngineUsesCameraConfig:
    def test_init_engine_uses_camera_max_sync_skew(self, camera, monkeypatch):
        camera._config = CameraConfig(mock=True, max_sync_skew_ms=5.5)
        engine = BlockingEngine()
        captured = {}

        def fake_from_config(cls, config_path=None, **kwargs):
            captured.update(kwargs)
            return engine

        monkeypatch.setattr(
            ip.DualStageInferenceEngine, "from_config", classmethod(fake_from_config)
        )
        panel = ip.InferencePanel(camera)
        try:
            assert panel.init_engine() is True
            assert captured["max_sync_skew_ms"] == pytest.approx(5.5)
        finally:
            engine.release.set()
            panel.deinit_engine()
