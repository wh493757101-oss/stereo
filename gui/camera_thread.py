"""
双目相机采集线程（PySide6）。

支持三种模式：
- 海康 MVS SDK 实时采集（GigE 工业相机）
- 目录回放（调试用）
- Mock 模式（无相机调试）

封装海康 SDK 双相机采集逻辑，并通过 PySide6 Signal 传递帧数据。
"""

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.stereo_matching import to_gray_u8


FRAME_TIMEOUT_MS = 1000

TRIGGER_SOFTWARE = "software"
TRIGGER_HARDWARE = "hardware"
VALID_TRIGGER_SOURCES = (TRIGGER_SOFTWARE, TRIGGER_HARDWARE)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    left_ip: str = "192.168.1.11"
    right_ip: str = "192.168.1.12"
    left_exposure_us: float = 40000.0
    right_exposure_us: float = 40000.0
    left_gain_db: float = 0.0
    right_gain_db: float = 0.0
    mock: bool = False
    trigger_source: str = TRIGGER_SOFTWARE
    max_sync_skew_ms: float = 2.0
    already_rectified: bool = False


@dataclass
class ImageStats:
    mean: float = 0.0
    min_value: int = 0
    max_value: int = 0
    std: float = 0.0
    dark_pct: float = 0.0
    saturated_pct: float = 0.0
    width: int = 0
    height: int = 0


@dataclass
class FrameBundle:
    """一帧双目数据，包含原始灰度图、BGR 彩色图、图像统计和诊断信息。

    Synchronization fields:
        pair_id: monotonic per-pair counter assigned by the capture thread.
        left_frame_no / right_frame_no: device frame numbers when available.
        left_timestamp_ns / right_timestamp_ns: monotonic or device
            timestamps in nanoseconds when available, else None.
        sync_skew_ms: measured left/right skew in milliseconds; None means
            the skew is unavailable (e.g. shared hardware trigger) and must
            not be presented as a measured zero.
        rectified: True when the frames are already rectified and must not
            be remapped again downstream.
    """
    left: Optional[np.ndarray] = None
    right: Optional[np.ndarray] = None
    left_bgr: Optional[np.ndarray] = None
    right_bgr: Optional[np.ndarray] = None
    left_stats: Optional[ImageStats] = None
    right_stats: Optional[ImageStats] = None
    fps: float = 0.0
    left_fail_count: int = 0
    right_fail_count: int = 0
    trigger_ret_left: int = 0
    trigger_ret_right: int = 0
    grab_ret_left: int = 0
    grab_ret_right: int = 0
    config: Optional[CameraConfig] = None
    timestamp: str = ""
    pair_id: int = 0
    left_frame_no: Optional[int] = None
    right_frame_no: Optional[int] = None
    left_timestamp_ns: Optional[int] = None
    right_timestamp_ns: Optional[int] = None
    sync_skew_ms: Optional[float] = None
    rectified: bool = False


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _format_ret(ret: Optional[int]) -> str:
    if ret is None:
        return "-"
    if ret == 0:
        return "0x0"
    return f"0x{int(ret) & 0xFFFFFFFF:08X}"


def compute_image_stats(img: np.ndarray) -> ImageStats:
    """计算单通道、(H, W, 1) 或 BGR 图像的灰度统计。"""
    gray = to_gray_u8(img)
    h, w = gray.shape[:2]
    total = max(int(gray.size), 1)
    dark = int(np.count_nonzero(gray <= 5))
    saturated = int(np.count_nonzero(gray >= 250))
    return ImageStats(
        mean=float(np.mean(gray)),
        min_value=int(np.min(gray)),
        max_value=int(np.max(gray)),
        std=float(np.std(gray)),
        dark_pct=dark * 100.0 / total,
        saturated_pct=saturated * 100.0 / total,
        width=w,
        height=h,
    )


# ---------------------------------------------------------------------------
# 海康 MVS SDK 入口
# ---------------------------------------------------------------------------

_MVS_API = None


def _prepare_mvs_import() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_candidates = [
        current_dir,
        os.path.join(current_dir, ".."),
        os.path.join(current_dir, "..", "third_party"),
        r"D:\MVS\MVS\Development\Samples\Python",
        r"D:\MVS\MVS\Development\MVFG\Samples\Python",
        r"C:\Program Files\MVS\Development\Samples\Python",
        r"C:\Program Files (x86)\MVS\Development\Samples\Python",
        r"C:\Program Files\Common Files\MVS\Development\Samples\Python",
        r"C:\Program Files (x86)\Common Files\MVS\Development\Samples\Python",
    ]
    for path in python_candidates:
        if os.path.isdir(os.path.join(path, "MvImport")) and path not in sys.path:
            sys.path.insert(0, path)

    if hasattr(os, "add_dll_directory"):
        dll_candidates = [
            r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64",
            r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64",
            r"C:\Program Files\MVS\Runtime\Win64_x64",
            r"C:\Program Files (x86)\MVS\Runtime\Win64_x64",
        ]
        for path in dll_candidates:
            if os.path.isdir(path):
                try:
                    os.add_dll_directory(path)
                except OSError:
                    pass


def load_mvs_api():
    global _MVS_API
    if _MVS_API is not None:
        return _MVS_API

    _prepare_mvs_import()
    try:
        from MvImport.MvCameraControl_class import MvCamera
        from MvImport.CameraParams_const import (
            MV_ACCESS_Control,
            MV_ACCESS_Exclusive,
            MV_GIGE_DEVICE,
            MV_USB_DEVICE,
        )
        from MvImport.CameraParams_header import (
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_FRAME_OUT_INFO_EX,
            MVCC_INTVALUE,
        )
    except Exception as exc:
        raise RuntimeError(
            "未找到或无法加载海康 MVS Python 接口 MvImport。\n"
            "请确认已安装 MVS 开发包，或把 MvImport 文件夹放到项目根目录/third_party。\n"
            "缺少 SDK 时，主界面「采集」页的「启动相机」无法进行实时采集。"
        ) from exc

    @dataclass
    class MvsApi:
        MvCamera: Any
        MV_GIGE_DEVICE: int
        MV_USB_DEVICE: int
        MV_ACCESS_Exclusive: int
        MV_ACCESS_Control: int
        MV_CC_DEVICE_INFO: Any
        MV_CC_DEVICE_INFO_LIST: Any
        MVCC_INTVALUE: Any
        MV_FRAME_OUT_INFO_EX: Any

    _MVS_API = MvsApi(
        MvCamera=MvCamera,
        MV_GIGE_DEVICE=MV_GIGE_DEVICE,
        MV_USB_DEVICE=MV_USB_DEVICE,
        MV_ACCESS_Exclusive=MV_ACCESS_Exclusive,
        MV_ACCESS_Control=MV_ACCESS_Control,
        MV_CC_DEVICE_INFO=MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST=MV_CC_DEVICE_INFO_LIST,
        MVCC_INTVALUE=MVCC_INTVALUE,
        MV_FRAME_OUT_INFO_EX=MV_FRAME_OUT_INFO_EX,
    )
    return _MVS_API


def _close_camera(cam) -> None:
    if cam is None:
        return
    for method_name in ("MV_CC_StopGrabbing", "MV_CC_CloseDevice", "MV_CC_DestroyHandle"):
        try:
            getattr(cam, method_name)()
        except Exception:
            pass


def _set_enum_by_string_with_fallback(cam, key: str, value: str, fallback_value=None) -> int:
    ret = cam.MV_CC_SetEnumValueByString(key, value)
    if ret == 0:
        return ret
    if fallback_value is not None:
        return cam.MV_CC_SetEnumValue(key, fallback_value)
    return ret


def _set_camera_optics(cam, exposure_time_us: float, gain_db: float) -> list[str]:
    warnings = []
    ret = cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    if ret != 0:
        warnings.append(f"关闭自动曝光失败: {_format_ret(ret)}")
    ret = cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time_us))
    if ret != 0:
        warnings.append(f"设置曝光失败: {_format_ret(ret)}")
    ret = cam.MV_CC_SetEnumValue("GainAuto", 0)
    if ret != 0:
        warnings.append(f"关闭自动增益失败: {_format_ret(ret)}")
    ret = cam.MV_CC_SetFloatValue("Gain", float(gain_db))
    if ret != 0:
        warnings.append(f"设置增益失败: {_format_ret(ret)}")
    return warnings


def _ip_from_device_info(device_info) -> str:
    current_ip = device_info.SpecialInfo.stGigEInfo.nCurrentIp
    return (
        f"{(current_ip & 0xFF000000) >> 24}."
        f"{(current_ip & 0x00FF0000) >> 16}."
        f"{(current_ip & 0x0000FF00) >> 8}."
        f"{current_ip & 0x000000FF}"
    )


def _configure_trigger(cam, trigger_source: str) -> None:
    """Set trigger mode/source consistently on one camera.

    Software mode: TriggerMode=On, TriggerSource=Software; the host issues
    ``TriggerSoftware`` commands per frame. Hardware mode: TriggerMode=On,
    TriggerSource=Line0 (shared hardware line); the host never issues
    software trigger commands.
    """
    _set_enum_by_string_with_fallback(cam, "TriggerMode", "On", 1)
    if trigger_source == TRIGGER_HARDWARE:
        _set_enum_by_string_with_fallback(cam, "TriggerSource", "Line0", 0)
    else:
        _set_enum_by_string_with_fallback(cam, "TriggerSource", "Software", 7)


def init_and_open_camera(
    ip_address: str,
    exposure_time_us: float,
    gain_db: float,
    api,
    trigger_source: str = TRIGGER_SOFTWARE,
):
    cam = api.MvCamera()
    device_list = api.MV_CC_DEVICE_INFO_LIST()
    ret = api.MvCamera.MV_CC_EnumDevices(api.MV_GIGE_DEVICE | api.MV_USB_DEVICE, device_list)
    if ret != 0:
        raise RuntimeError(f"枚举设备失败: {_format_ret(ret)}")

    target_device_info = None
    for i in range(device_list.nDeviceNum):
        info = ctypes.cast(device_list.pDeviceInfo[i], ctypes.POINTER(api.MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType != api.MV_GIGE_DEVICE:
            continue
        if _ip_from_device_info(info) == ip_address:
            target_device_info = info
            break

    if target_device_info is None:
        raise RuntimeError(f"未找到 IP {ip_address}，请检查网线、相机 IP 和 MVS 配置。")

    handle_created = False
    try:
        ret = cam.MV_CC_CreateHandle(target_device_info)
        if ret != 0:
            raise RuntimeError(f"相机 {ip_address} 创建句柄失败: {_format_ret(ret)}")
        handle_created = True

        access_mode_used = "Exclusive"
        ret = cam.MV_CC_OpenDevice(api.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            access_mode_used = "Control"
            ret = cam.MV_CC_OpenDevice(api.MV_ACCESS_Control, 0)
        if ret != 0:
            raise RuntimeError(f"相机 {ip_address} 打开失败: {_format_ret(ret)}")

        packet_size = cam.MV_CC_GetOptimalPacketSize()
        if int(packet_size) > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

        _set_enum_by_string_with_fallback(cam, "PixelFormat", "Mono8")
        _set_camera_optics(cam, exposure_time_us, gain_db)
        _configure_trigger(cam, trigger_source)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"相机 {ip_address} 开始取流失败: {_format_ret(ret)}")

        return cam, access_mode_used
    except Exception:
        if handle_created:
            _close_camera(cam)
        raise


def get_payload_size(cam, api) -> int:
    st_param = api.MVCC_INTVALUE()
    ret = cam.MV_CC_GetIntValue("PayloadSize", st_param)
    if ret != 0:
        raise RuntimeError(f"获取 PayloadSize 失败: {_format_ret(ret)}")
    return int(st_param.nCurValue)


@dataclass
class FrameMeta:
    """Per-frame SDK metadata collected via getattr fallbacks.

    SDK structs vary between MVS versions; every field is optional and None
    means the camera/SDK did not expose it.
    """

    frame_no: Optional[int] = None
    timestamp_ns: Optional[int] = None
    exposure_time_us: Optional[float] = None
    gain_db: Optional[float] = None


def _frame_meta_from_info(frame_info) -> FrameMeta:
    def _int(*names: str) -> Optional[int]:
        for name in names:
            try:
                value = getattr(frame_info, name, None)
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return None

    frame_no = _int("nFrameNum")

    # Device timestamps: high/low word pairs (ns since an arbitrary device
    # epoch) or a direct 64-bit ns field. Host timestamp is a wall-clock
    # fallback; all are best-effort and may be missing entirely.
    ts_ns: Optional[int] = None
    high = _int("nDevTimeStampHigh")
    low = _int("nDevTimeStampLow")
    if high is not None and low is not None:
        ts_ns = (high << 32) | low
    else:
        ts_ns = _int("nDevTimeStamp", "timestamp_ns")
    if ts_ns is None:
        ts_ns = _int("nHostTimeStamp")
    if ts_ns is not None and ts_ns <= 0:
        ts_ns = None

    exposure = getattr(frame_info, "fExposureTime", None)
    gain = getattr(frame_info, "fAnalogGain", None)
    return FrameMeta(
        frame_no=frame_no,
        timestamp_ns=ts_ns,
        exposure_time_us=float(exposure) if exposure is not None else None,
        gain_db=float(gain) if gain is not None else None,
    )


def convert_to_cv2_image(cam, payload_size: int, api):
    data_buf = (ctypes.c_ubyte * payload_size)()
    frame_info = api.MV_FRAME_OUT_INFO_EX()
    ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, frame_info, FRAME_TIMEOUT_MS)
    if ret != 0:
        return None, ret, FrameMeta()

    frame_len = int(frame_info.nFrameLen)
    height = int(frame_info.nHeight)
    width = int(frame_info.nWidth)
    img_data = np.frombuffer(data_buf, count=frame_len, dtype=np.uint8)
    try:
        img_cv = img_data.reshape((height, width)).copy()
    except ValueError:
        return None, -1, FrameMeta()
    return img_cv, ret, _frame_meta_from_info(frame_info)


# ---------------------------------------------------------------------------
# Mock 相机（调试用）
# ---------------------------------------------------------------------------

def _make_mock_pair(frame_index: int, width: int = 1280, height: int = 1024):
    """Build a synthetic pair with positive left-minus-right disparity.

    With positive disparity the right-view content appears to the LEFT of
    the left-view content: ``x_right = x_left - disparity`` (18 px here).
    """
    x = np.linspace(0, 255, width, dtype=np.uint16)
    y = np.linspace(0, 70, height, dtype=np.uint16)[:, None]
    base = ((x[None, :] + y + frame_index * 3) % 256).astype(np.uint8)
    left_gray = base.copy()
    right_gray = np.roll(base, shift=-18, axis=1).copy()
    cx = 180 + (frame_index * 9) % max(width - 360, 1)
    cy = 160 + (frame_index * 5) % max(height - 320, 1)
    cv2.rectangle(left_gray, (cx, cy), (cx + 180, cy + 120), 215, -1)
    cv2.rectangle(right_gray, (cx - 18, cy), (cx + 162, cy + 120), 205, -1)
    cv2.circle(left_gray, (width // 2, height // 2), 110, 55, 4)
    cv2.circle(right_gray, (width // 2 - 18, height // 2), 110, 60, 4)
    cv2.putText(left_gray, "MOCK L", (48, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 240, 3)
    cv2.putText(right_gray, "MOCK R", (48, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 230, 3)
    left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
    return left_gray, right_gray, left_bgr, right_bgr


# ---------------------------------------------------------------------------
# 采集线程
# ---------------------------------------------------------------------------

class CameraThread(QThread):
    """双目相机采集线程，支持海康 MVS 实时采集 / 目录回放 / Mock 模式。"""

    frame_ready = Signal(FrameBundle)
    fps_updated = Signal(float)
    status = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        source: str | int | CameraConfig | None = None,
        parent=None,
    ):
        """
        Args:
            source:
                - CameraConfig: 海康 MVS 实时采集配置。
                - str: 图像目录路径，需包含 left/ 和 right/ 子目录（回放模式）。
                - None / int: 等价于 CameraConfig(mock=True)。
        """
        super().__init__(parent)
        self._running = False
        self._config: CameraConfig | None = None
        self._playback_dir: str | None = None

        if isinstance(source, CameraConfig):
            self._config = source
        elif isinstance(source, str) and Path(source).is_dir():
            self._playback_dir = source
        else:
            self._config = CameraConfig(mock=True)

    def run(self):
        self._running = True
        if self._playback_dir:
            self._run_playback(self._playback_dir)
        elif self._config and self._config.mock:
            self._run_mock()
        else:
            self._run_real()

    # ---- Mock 模式 ----
    def _run_mock(self):
        config = self._config or CameraConfig(mock=True)
        frame_index = 0
        last_time = time.monotonic()
        self.status.emit("模拟相机已启动")

        try:
            while self._running:
                left_gray, right_gray, left_bgr, right_bgr = _make_mock_pair(frame_index)
                now = time.monotonic()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                ts_ns = time.monotonic_ns()
                self.fps_updated.emit(fps)
                self.frame_ready.emit(FrameBundle(
                    left=left_gray,
                    right=right_gray,
                    left_bgr=left_bgr,
                    right_bgr=right_bgr,
                    left_stats=compute_image_stats(left_gray),
                    right_stats=compute_image_stats(right_gray),
                    fps=fps,
                    config=config,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    pair_id=frame_index,
                    left_frame_no=frame_index,
                    right_frame_no=frame_index,
                    left_timestamp_ns=ts_ns,
                    right_timestamp_ns=ts_ns,
                    sync_skew_ms=0.0,
                    rectified=config.already_rectified,
                ))
                frame_index += 1
                self.msleep(33)
        except Exception as exc:
            self.error.emit(f"模拟相机运行失败: {exc}")

    # ---- 目录回放 ----
    def _run_playback(self, root_dir: str):
        root = Path(root_dir)
        left_dir = root / "left"
        right_dir = root / "right"
        if not left_dir.exists() or not right_dir.exists():
            self.error.emit(f"回放模式需要 {root_dir} 下有 left/ 和 right/ 子目录")
            return

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        left_map = {f.stem: f for f in sorted(left_dir.iterdir()) if f.suffix.lower() in exts}
        right_map = {f.stem: f for f in sorted(right_dir.iterdir()) if f.suffix.lower() in exts}
        common = sorted(set(left_map) & set(right_map))
        if not common:
            self.error.emit("left/ 和 right/ 中没有匹配的图像对")
            return

        self.status.emit(f"回放模式: 找到 {len(common)} 对图像")
        last_time = time.monotonic()
        config = CameraConfig(mock=False)
        for pair_id, stem in enumerate(common):
            if not self._running:
                break
            left_gray = cv2.imread(str(left_map[stem]), cv2.IMREAD_GRAYSCALE)
            right_gray = cv2.imread(str(right_map[stem]), cv2.IMREAD_GRAYSCALE)
            if left_gray is None or right_gray is None:
                continue
            left_gray = to_gray_u8(left_gray)
            right_gray = to_gray_u8(right_gray)
            left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
            right_bgr = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
            now = time.monotonic()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            ts_ns = time.monotonic_ns()
            self.fps_updated.emit(fps)
            # Playback has no device timestamps and no measured skew: both
            # stay None so downstream can report sync as unavailable.
            self.frame_ready.emit(FrameBundle(
                left=left_gray,
                right=right_gray,
                left_bgr=left_bgr,
                right_bgr=right_bgr,
                left_stats=compute_image_stats(left_gray),
                right_stats=compute_image_stats(right_gray),
                fps=fps,
                config=config,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                pair_id=pair_id,
                left_timestamp_ns=ts_ns,
                right_timestamp_ns=ts_ns,
                sync_skew_ms=None,
                rectified=config.already_rectified,
            ))
            self.msleep(33)

    # ---- 海康 MVS 实时采集 ----
    def _run_real(self):
        api = None
        cam_left = None
        cam_right = None
        config = self._config or CameraConfig()
        trigger_source = (
            config.trigger_source
            if config.trigger_source in VALID_TRIGGER_SOURCES
            else TRIGGER_SOFTWARE
        )
        left_fail_count = 0
        right_fail_count = 0
        last_time = time.monotonic()

        try:
            api = load_mvs_api()
            api.MvCamera.MV_CC_Initialize()
            self.status.emit("正在连接相机...")

            cam_left, left_access = init_and_open_camera(
                config.left_ip, config.left_exposure_us, config.left_gain_db, api,
                trigger_source=trigger_source,
            )
            self.status.emit(f"左目已连接 ({left_access})")

            cam_right, right_access = init_and_open_camera(
                config.right_ip, config.right_exposure_us, config.right_gain_db, api,
                trigger_source=trigger_source,
            )
            self.status.emit(f"右目已连接 ({right_access})")

            payload_left = get_payload_size(cam_left, api)
            payload_right = get_payload_size(cam_right, api)
            if trigger_source == TRIGGER_HARDWARE:
                self.status.emit("开始采集 (硬件触发)...")
            else:
                self.status.emit("开始采集 (软触发)...")
            pair_id = 0

            while self._running:
                # Software trigger only in software mode; the host command
                # times are the only common clock for skew estimation.
                if trigger_source == TRIGGER_SOFTWARE:
                    t_left_ns = time.monotonic_ns()
                    trigger_ret_left = cam_left.MV_CC_SetCommandValue("TriggerSoftware")
                    t_right_ns = time.monotonic_ns()
                    trigger_ret_right = cam_right.MV_CC_SetCommandValue("TriggerSoftware")
                    if trigger_ret_left == 0 and trigger_ret_right == 0:
                        sync_skew_ms = abs(t_left_ns - t_right_ns) / 1e6
                    else:
                        sync_skew_ms = None
                else:
                    trigger_ret_left = 0
                    trigger_ret_right = 0
                    # Shared hardware line: no comparable host timestamps
                    # exist, so skew is genuinely unavailable, not zero.
                    sync_skew_ms = None

                img_l, grab_ret_left, meta_l = convert_to_cv2_image(cam_left, payload_left, api)
                img_r, grab_ret_right, meta_r = convert_to_cv2_image(cam_right, payload_right, api)

                left_fail_count = left_fail_count + 1 if img_l is None else 0
                right_fail_count = right_fail_count + 1 if img_r is None else 0

                left_gray = to_gray_u8(img_l) if img_l is not None else None
                right_gray = to_gray_u8(img_r) if img_r is not None else None
                left_bgr = (
                    cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
                    if left_gray is not None else None
                )
                right_bgr = (
                    cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)
                    if right_gray is not None else None
                )

                now = time.monotonic()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now

                self.fps_updated.emit(fps)
                self.frame_ready.emit(FrameBundle(
                    left=left_gray,
                    right=right_gray,
                    left_bgr=left_bgr,
                    right_bgr=right_bgr,
                    left_stats=compute_image_stats(left_gray) if left_gray is not None else None,
                    right_stats=compute_image_stats(right_gray) if right_gray is not None else None,
                    fps=fps,
                    left_fail_count=left_fail_count,
                    right_fail_count=right_fail_count,
                    trigger_ret_left=trigger_ret_left,
                    trigger_ret_right=trigger_ret_right,
                    grab_ret_left=grab_ret_left,
                    grab_ret_right=grab_ret_right,
                    config=config,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    pair_id=pair_id,
                    left_frame_no=meta_l.frame_no,
                    right_frame_no=meta_r.frame_no,
                    left_timestamp_ns=meta_l.timestamp_ns,
                    right_timestamp_ns=meta_r.timestamp_ns,
                    sync_skew_ms=sync_skew_ms,
                    rectified=config.already_rectified,
                ))
                pair_id += 1
                self.msleep(5)

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            _close_camera(cam_left)
            _close_camera(cam_right)
            if api is not None:
                try:
                    api.MvCamera.MV_CC_Finalize()
                except Exception:
                    pass
            self.status.emit("相机已断开")

    def stop(self):
        self._running = False
        self.wait(3000)
