"""Tests for gui.camera_thread: bundle metadata, mock disparity sign,
trigger configuration, MVS metadata fallbacks and gray boundary.

No real MVS SDK, cameras or weights are required.
"""

import os
import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import gui.camera_thread as ct
from gui.camera_thread import (
    CameraConfig,
    FrameBundle,
    FrameMeta,
    _make_mock_pair,
    compute_image_stats,
)

_APP = QApplication.instance() or QApplication([])


@pytest.fixture
def tmp_path() -> Path:
    """Isolated work dir under the system temp (pytest basetemp under tests/
    can be locked by another process on Windows)."""
    workdir = Path(tempfile.mkdtemp(prefix="camera_thread_test_"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# FrameBundle metadata / compatibility
# ---------------------------------------------------------------------------

class TestFrameBundleCompat:
    def test_default_construction_unchanged(self):
        bundle = FrameBundle()
        assert bundle.left is None
        assert bundle.right is None
        assert bundle.config is None
        assert bundle.timestamp == ""
        assert bundle.fps == 0.0

    def test_new_sync_fields_default_to_unavailable(self):
        bundle = FrameBundle()
        assert bundle.pair_id == 0
        assert bundle.left_frame_no is None
        assert bundle.right_frame_no is None
        assert bundle.left_timestamp_ns is None
        assert bundle.right_timestamp_ns is None
        assert bundle.sync_skew_ms is None
        assert bundle.rectified is False

    def test_legacy_keyword_construction_still_works(self):
        img = np.zeros((2, 2), np.uint8)
        bundle = FrameBundle(left=img, right=img, fps=30.0)
        assert bundle.fps == 30.0
        assert bundle.sync_skew_ms is None

    def test_camera_config_defaults(self):
        cfg = CameraConfig()
        assert cfg.trigger_source == "software"
        assert cfg.max_sync_skew_ms == pytest.approx(2.0)
        assert cfg.already_rectified is False


# ---------------------------------------------------------------------------
# Mock pair disparity sign
# ---------------------------------------------------------------------------

class TestMockPair:
    def test_base_texture_right_is_shifted_left_by_18(self):
        """Positive left-minus-right disparity: x_right = x_left - 18."""
        left, right, _, _ = _make_mock_pair(0)
        h, w = left.shape
        # Rows above the rectangle (cy=160) and the MOCK text (~y<=115).
        np.testing.assert_array_equal(right[0:40, 0:w - 19], left[0:40, 18:w - 1])

    def test_rectangle_right_is_left_of_left_rectangle(self):
        left, right, _, _ = _make_mock_pair(3)
        cx = 180 + (3 * 9) % (left.shape[1] - 360)
        cy = 160 + (3 * 5) % (left.shape[0] - 320)
        assert left[cy + 10, cx + 10] == 215
        assert right[cy + 10, cx + 10 - 18] == 205

    def test_circle_right_is_left_of_left_circle(self):
        left, right, _, _ = _make_mock_pair(2)
        h, w = left.shape
        top = h // 2 - 110
        # Circle outline topmost point: right center is 18 px to the left.
        assert 60 in right[top - 2:top + 3, w // 2 - 18]
        assert 55 in left[top - 2:top + 3, w // 2]


# ---------------------------------------------------------------------------
# MVS frame metadata via getattr fallbacks
# ---------------------------------------------------------------------------

class TestFrameMeta:
    def test_missing_fields_all_none(self):
        info = types.SimpleNamespace(nFrameLen=4, nHeight=2, nWidth=2)
        meta = ct._frame_meta_from_info(info)
        assert meta.frame_no is None
        assert meta.timestamp_ns is None

    def test_high_low_timestamp_pair_combined(self):
        info = types.SimpleNamespace(
            nFrameNum=17,
            nDevTimeStampHigh=1,
            nDevTimeStampLow=2,
        )
        meta = ct._frame_meta_from_info(info)
        assert meta.frame_no == 17
        assert meta.timestamp_ns == (1 << 32) | 2

    def test_direct_timestamp_fallback(self):
        info = types.SimpleNamespace(nDevTimeStamp=12345)
        meta = ct._frame_meta_from_info(info)
        assert meta.timestamp_ns == 12345

    def test_host_timestamp_last_resort(self):
        info = types.SimpleNamespace(nHostTimeStamp=999)
        meta = ct._frame_meta_from_info(info)
        assert meta.timestamp_ns == 999

    def test_nonpositive_timestamp_treated_as_missing(self):
        info = types.SimpleNamespace(nDevTimeStamp=0)
        meta = ct._frame_meta_from_info(info)
        assert meta.timestamp_ns is None

    def test_frame_meta_defaults(self):
        meta = FrameMeta()
        assert meta.frame_no is None
        assert meta.timestamp_ns is None
        assert meta.exposure_time_us is None
        assert meta.gain_db is None


# ---------------------------------------------------------------------------
# Trigger configuration with fake cameras
# ---------------------------------------------------------------------------

class FakeTriggerCam:
    def __init__(self) -> None:
        self.enum_calls = []

    def MV_CC_SetEnumValueByString(self, key: str, value: str) -> int:
        self.enum_calls.append((key, value))
        return 0

    def MV_CC_SetEnumValue(self, key: str, value: int) -> int:
        self.enum_calls.append((key, value))
        return 0


class TestConfigureTrigger:
    def test_software_mode_sets_software_source(self):
        cam = FakeTriggerCam()
        ct._configure_trigger(cam, "software")
        keys = dict(cam.enum_calls)
        assert keys["TriggerMode"] == "On"
        assert keys["TriggerSource"] == "Software"

    def test_hardware_mode_sets_line0_and_never_software(self):
        cam = FakeTriggerCam()
        ct._configure_trigger(cam, "hardware")
        keys = dict(cam.enum_calls)
        assert keys["TriggerMode"] == "On"
        assert keys["TriggerSource"] == "Line0"
        assert not any(v == "Software" for _, v in cam.enum_calls)


# ---------------------------------------------------------------------------
# Run-loop trigger behavior with fake cameras
# ---------------------------------------------------------------------------

class FakeMvCamera:
    @staticmethod
    def MV_CC_Initialize() -> int:
        return 0

    @staticmethod
    def MV_CC_Finalize() -> int:
        return 0


class FakeApi:
    MvCamera = FakeMvCamera


class FakeLiveCam:
    def __init__(self) -> None:
        self.commands = []

    def MV_CC_SetCommandValue(self, name: str) -> int:
        self.commands.append(name)
        return 0


def _run_one_real_frame(monkeypatch, trigger_source: str) -> FrameBundle:
    """Run _run_real until the first frame with fully faked cameras."""
    cams = {"192.168.1.11": FakeLiveCam(), "192.168.1.12": FakeLiveCam()}
    frame_counter = {"n": 0}

    def fake_convert(cam, payload_size, api):
        frame_counter["n"] += 1
        img = np.full((4, 4), frame_counter["n"], np.uint8)
        meta = FrameMeta(frame_no=frame_counter["n"], timestamp_ns=1000 * frame_counter["n"])
        return img, 0, meta

    monkeypatch.setattr(ct, "load_mvs_api", lambda: FakeApi())
    monkeypatch.setattr(
        ct, "init_and_open_camera",
        lambda ip, exp, gain, api, trigger_source="software": (cams[ip], "Fake"),
    )
    monkeypatch.setattr(ct, "get_payload_size", lambda cam, api: 16)
    monkeypatch.setattr(ct, "convert_to_cv2_image", fake_convert)

    thread = ct.CameraThread(
        source=CameraConfig(
            mock=False,
            trigger_source=trigger_source,
            left_ip="192.168.1.11",
            right_ip="192.168.1.12",
        )
    )
    got: list[FrameBundle] = []
    done = threading.Event()

    def on_frame(bundle: FrameBundle) -> None:
        got.append(bundle)
        thread._running = False
        done.set()

    thread.frame_ready.connect(on_frame, Qt.DirectConnection)
    thread._running = True
    worker = threading.Thread(target=thread._run_real, daemon=True)
    worker.start()
    assert done.wait(10.0), "no frame produced by fake cameras"
    worker.join(10.0)
    assert got, "frame bundle missing"
    return got[0]


class TestRunLoopTrigger:
    def test_software_mode_issues_commands_and_measures_skew(self, monkeypatch):
        bundle = _run_one_real_frame(monkeypatch, "software")
        assert bundle.left is not None and bundle.right is not None
        # TriggerSoftware issued exactly once per camera in software mode.
        assert bundle.trigger_ret_left == 0
        assert bundle.trigger_ret_right == 0
        # Skew measured from host command timestamps (0.0 is a valid
        # measurement); must never be None when both commands succeed.
        assert bundle.sync_skew_ms is not None
        assert bundle.sync_skew_ms >= 0.0

    def test_hardware_mode_never_issues_software_trigger(self, monkeypatch):
        # Re-run with command capture to inspect camera command lists.
        cams = {"192.168.1.11": FakeLiveCam(), "192.168.1.12": FakeLiveCam()}
        monkeypatch.setattr(ct, "load_mvs_api", lambda: FakeApi())
        monkeypatch.setattr(
            ct, "init_and_open_camera",
            lambda ip, exp, gain, api, trigger_source="software": (cams[ip], "Fake"),
        )
        monkeypatch.setattr(ct, "get_payload_size", lambda cam, api: 16)
        monkeypatch.setattr(
            ct, "convert_to_cv2_image",
            lambda cam, p, api: (
                np.zeros((4, 4), np.uint8), 0,
                FrameMeta(frame_no=1, timestamp_ns=1000),
            ),
        )
        thread = ct.CameraThread(
            source=CameraConfig(mock=False, trigger_source="hardware")
        )
        got: list[FrameBundle] = []
        done = threading.Event()

        def on_frame(bundle: FrameBundle) -> None:
            got.append(bundle)
            thread._running = False
            done.set()

        thread.frame_ready.connect(on_frame, Qt.DirectConnection)
        thread._running = True
        worker = threading.Thread(target=thread._run_real, daemon=True)
        worker.start()
        assert done.wait(10.0)
        worker.join(10.0)

        for cam in cams.values():
            assert "TriggerSoftware" not in cam.commands
        assert got[0].sync_skew_ms is None  # unavailable, not a fake zero
        assert got[0].trigger_ret_left == 0
        assert got[0].trigger_ret_right == 0

    def test_real_mode_propagates_metadata_and_rectified_flag(self, monkeypatch):
        bundle = _run_one_real_frame(monkeypatch, "software")
        assert bundle.pair_id == 0
        # The shared fake counter increments per camera grab (left first).
        assert bundle.left_frame_no == 1
        assert bundle.right_frame_no == 2
        assert bundle.left_timestamp_ns == 1000
        assert bundle.right_timestamp_ns == 2000
        assert bundle.rectified is False

        bundle_rect = _run_one_real_frame_rectified(monkeypatch)
        assert bundle_rect.rectified is True


def _run_one_real_frame_rectified(monkeypatch) -> FrameBundle:
    cams = {"192.168.1.11": FakeLiveCam(), "192.168.1.12": FakeLiveCam()}
    monkeypatch.setattr(ct, "load_mvs_api", lambda: FakeApi())
    monkeypatch.setattr(
        ct, "init_and_open_camera",
        lambda ip, exp, gain, api, trigger_source="software": (cams[ip], "Fake"),
    )
    monkeypatch.setattr(ct, "get_payload_size", lambda cam, api: 16)
    monkeypatch.setattr(
        ct, "convert_to_cv2_image",
        lambda cam, p, api: (
            np.zeros((4, 4), np.uint8), 0, FrameMeta(frame_no=1, timestamp_ns=1)
        ),
    )
    thread = ct.CameraThread(
        source=CameraConfig(mock=False, already_rectified=True)
    )
    got: list[FrameBundle] = []
    done = threading.Event()

    def on_frame(bundle: FrameBundle) -> None:
        got.append(bundle)
        thread._running = False
        done.set()

    thread.frame_ready.connect(on_frame, Qt.DirectConnection)
    thread._running = True
    worker = threading.Thread(target=thread._run_real, daemon=True)
    worker.start()
    assert done.wait(10.0)
    worker.join(10.0)
    return got[0]


# ---------------------------------------------------------------------------
# MVS load failure hint
# ---------------------------------------------------------------------------

class TestMvsApiErrorHint:
    def test_mvs_load_failure_points_to_gui_capture_tab(self, monkeypatch):
        """The error hint must describe the current GUI, not the removed
        dual_camera_capture.py CLI."""
        monkeypatch.setattr(ct, "_MVS_API", None)
        monkeypatch.setattr(ct, "_prepare_mvs_import", lambda: None)
        # A None entry in sys.modules makes the MvImport import fail
        # deterministically, with or without the SDK installed.
        monkeypatch.setitem(sys.modules, "MvImport", None)

        with pytest.raises(RuntimeError) as excinfo:
            ct.load_mvs_api()

        message = str(excinfo.value)
        assert "dual_camera_capture" not in message
        assert "--mock" not in message
        assert "采集" in message
        assert "启动相机" in message


# ---------------------------------------------------------------------------
# Playback bundles
# ---------------------------------------------------------------------------

class TestPlayback:
    def _make_dataset(self, root: Path, pairs: int = 2) -> None:
        left_dir = root / "left"
        right_dir = root / "right"
        left_dir.mkdir(parents=True)
        right_dir.mkdir(parents=True)
        for i in range(pairs):
            cv2.imwrite(str(left_dir / f"{i:03d}.png"), np.full((8, 8), 10 + i, np.uint8))
            cv2.imwrite(str(right_dir / f"{i:03d}.png"), np.full((8, 8), 20 + i, np.uint8))

    def test_playback_bundles_have_sequential_ids_and_unmeasured_sync(self, tmp_path):
        self._make_dataset(tmp_path)
        thread = ct.CameraThread(source=str(tmp_path))
        got: list[FrameBundle] = []

        def on_frame(bundle: FrameBundle) -> None:
            got.append(bundle)
            if len(got) >= 2:
                thread._running = False

        thread.frame_ready.connect(on_frame, Qt.DirectConnection)
        thread._running = True
        thread._run_playback(str(tmp_path))

        assert len(got) == 2
        for i, bundle in enumerate(got):
            assert bundle.pair_id == i
            assert bundle.left_timestamp_ns is not None
            assert bundle.right_timestamp_ns is not None
            # File playback has no device clocks: skew must stay unavailable.
            assert bundle.sync_skew_ms is None
        assert got[0].left.shape == (8, 8)


# ---------------------------------------------------------------------------
# Gray boundary at statistics
# ---------------------------------------------------------------------------

class TestGrayBoundary:
    def test_stats_on_hwc1_matches_squeezed_gray(self):
        gray = np.random.default_rng(0).integers(0, 256, (12, 16), dtype=np.uint8)
        stats_2d = compute_image_stats(gray)
        stats_1ch = compute_image_stats(gray[:, :, None])
        assert stats_1ch == stats_2d

    def test_stats_on_bgr_matches_gray_conversion(self):
        rng = np.random.default_rng(1)
        bgr = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        stats = compute_image_stats(bgr)
        expected = compute_image_stats(gray)
        assert stats == expected

    def test_stats_on_gray_passthrough(self):
        gray = np.full((8, 8), 128, np.uint8)
        stats = compute_image_stats(gray)
        assert stats.mean == pytest.approx(128.0)
        assert stats.width == 8 and stats.height == 8
