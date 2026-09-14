import argparse
import csv
import ctypes
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)


LEFT_IP = "192.168.1.11"
RIGHT_IP = "192.168.1.12"
LEFT_EXPOSURE_US = 40000.0
RIGHT_EXPOSURE_US = 40000.0
LEFT_GAIN_DB = 0.0
RIGHT_GAIN_DB = 0.0

FRAME_TIMEOUT_MS = 1000
SESSION_ROOT_NAME = ".capture_sessions"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_MVS_API = None


@dataclass
class CameraConfig:
    left_ip: str
    right_ip: str
    left_exposure_us: float
    right_exposure_us: float
    left_gain_db: float
    right_gain_db: float
    mock: bool = False


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


@dataclass
class ImageStats:
    mean: float
    min_value: int
    max_value: int
    std: float
    dark_pct: float
    saturated_pct: float
    width: int
    height: int


@dataclass
class FrameBundle:
    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    left_stats: Optional[ImageStats]
    right_stats: Optional[ImageStats]
    fps: float
    left_fail_count: int
    right_fail_count: int
    trigger_ret_left: int
    trigger_ret_right: int
    grab_ret_left: int
    grab_ret_right: int
    settings: CameraConfig
    timestamp: str


def _format_ret(ret: Optional[int]) -> str:
    if ret is None:
        return "-"
    if ret == 0:
        return "0x0"
    return f"0x{int(ret) & 0xFFFFFFFF:08X}"


def _prepare_mvs_import() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_candidates = [
        current_dir,
        os.path.join(current_dir, "third_party"),
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


def load_mvs_api() -> MvsApi:
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
            "如果只是测试界面和保存流程，请运行: python dual_camera_capture.py --mock"
        ) from exc

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


def _set_enum_by_string_with_fallback(cam: Any, key: str, value: str, fallback_value: Optional[int] = None) -> int:
    ret = cam.MV_CC_SetEnumValueByString(key, value)
    if ret == 0:
        return ret
    if fallback_value is not None:
        return cam.MV_CC_SetEnumValue(key, fallback_value)
    return ret


def _set_camera_optics(cam: Any, exposure_time_us: float, gain_db: float) -> List[str]:
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


def _close_camera(cam: Any) -> None:
    if cam is None:
        return
    for method_name in ("MV_CC_StopGrabbing", "MV_CC_CloseDevice", "MV_CC_DestroyHandle"):
        try:
            getattr(cam, method_name)()
        except Exception:
            pass


def _ip_from_device_info(device_info: Any) -> str:
    current_ip = device_info.SpecialInfo.stGigEInfo.nCurrentIp
    return (
        f"{(current_ip & 0xFF000000) >> 24}."
        f"{(current_ip & 0x00FF0000) >> 16}."
        f"{(current_ip & 0x0000FF00) >> 8}."
        f"{current_ip & 0x000000FF}"
    )


def init_and_open_camera(ip_address: str, exposure_time_us: float, gain_db: float, api: MvsApi) -> Tuple[Any, str]:
    cam = api.MvCamera()
    device_list = api.MV_CC_DEVICE_INFO_LIST()
    ret = api.MvCamera.MV_CC_EnumDevices(api.MV_GIGE_DEVICE | api.MV_USB_DEVICE, device_list)
    if ret != 0:
        raise RuntimeError(f"枚举设备失败: {_format_ret(ret)}")

    target_device_info = None
    for i in range(device_list.nDeviceNum):
        device_info = ctypes.cast(device_list.pDeviceInfo[i], ctypes.POINTER(api.MV_CC_DEVICE_INFO)).contents
        if device_info.nTLayerType != api.MV_GIGE_DEVICE:
            continue
        if _ip_from_device_info(device_info) == ip_address:
            target_device_info = device_info
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
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
            if ret != 0:
                print(f"[警告] 相机 {ip_address} 设置最优包大小失败: {_format_ret(ret)}")

        ret = _set_enum_by_string_with_fallback(cam, "PixelFormat", "Mono8")
        if ret != 0:
            print(f"[警告] 相机 {ip_address} 设置 PixelFormat=Mono8 失败: {_format_ret(ret)}")

        for warning in _set_camera_optics(cam, exposure_time_us, gain_db):
            print(f"[警告] 相机 {ip_address} {warning}")

        ret = _set_enum_by_string_with_fallback(cam, "TriggerMode", "On", 1)
        if ret != 0:
            print(f"[警告] 相机 {ip_address} 设置 TriggerMode=On 失败: {_format_ret(ret)}")
        ret = _set_enum_by_string_with_fallback(cam, "TriggerSource", "Software", 7)
        if ret != 0:
            print(f"[警告] 相机 {ip_address} 设置 TriggerSource=Software 失败: {_format_ret(ret)}")

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"相机 {ip_address} 开始取流失败: {_format_ret(ret)}")
        return cam, access_mode_used
    except Exception:
        if handle_created:
            _close_camera(cam)
        raise


def get_payload_size(cam: Any, api: MvsApi) -> int:
    st_param = api.MVCC_INTVALUE()
    ret = cam.MV_CC_GetIntValue("PayloadSize", st_param)
    if ret != 0:
        raise RuntimeError(f"获取 PayloadSize 失败: {_format_ret(ret)}")
    return int(st_param.nCurValue)


def convert_to_cv2_image(cam: Any, payload_size: int, api: MvsApi) -> Tuple[Optional[np.ndarray], int]:
    data_buf = (ctypes.c_ubyte * payload_size)()
    frame_info = api.MV_FRAME_OUT_INFO_EX()
    ret = cam.MV_CC_GetOneFrameTimeout(data_buf, payload_size, frame_info, FRAME_TIMEOUT_MS)
    if ret != 0:
        return None, ret

    frame_len = int(frame_info.nFrameLen)
    height = int(frame_info.nHeight)
    width = int(frame_info.nWidth)
    img_data = np.frombuffer(data_buf, count=frame_len, dtype=np.uint8)
    try:
        img_cv = img_data.reshape((height, width)).copy()
    except ValueError:
        return None, -1
    return img_cv, ret


def compute_image_stats(img_gray: np.ndarray) -> ImageStats:
    if img_gray.ndim != 2:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    height, width = img_gray.shape[:2]
    pixel_count = max(int(img_gray.size), 1)
    dark_count = int(np.count_nonzero(img_gray <= 5))
    saturated_count = int(np.count_nonzero(img_gray >= 250))
    return ImageStats(
        mean=float(np.mean(img_gray)),
        min_value=int(np.min(img_gray)),
        max_value=int(np.max(img_gray)),
        std=float(np.std(img_gray)),
        dark_pct=dark_count * 100.0 / pixel_count,
        saturated_pct=saturated_count * 100.0 / pixel_count,
        width=width,
        height=height,
    )


def _make_mock_pair(frame_index: int, width: int = 1280, height: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 255, width, dtype=np.uint16)
    y = np.linspace(0, 70, height, dtype=np.uint16)[:, None]
    base = ((x[None, :] + y + frame_index * 3) % 256).astype(np.uint8)
    left = base.copy()
    right = np.roll(base, shift=18, axis=1).copy()
    cx = 180 + (frame_index * 9) % max(width - 360, 1)
    cy = 160 + (frame_index * 5) % max(height - 320, 1)
    cv2.rectangle(left, (cx, cy), (cx + 180, cy + 120), 215, -1)
    cv2.rectangle(right, (cx + 18, cy), (cx + 198, cy + 120), 205, -1)
    cv2.circle(left, (width // 2, height // 2), 110, 55, 4)
    cv2.circle(right, (width // 2 + 18, height // 2), 110, 60, 4)
    cv2.putText(left, "MOCK L", (48, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 240, 3)
    cv2.putText(right, "MOCK R", (48, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 230, 3)
    return left, right


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    connected = pyqtSignal(object)
    stopped = pyqtSignal()

    def __init__(self, config: CameraConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._running = False
        self._lock = threading.Lock()
        self._pending_config: Optional[CameraConfig] = None

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def request_settings(self, config: CameraConfig) -> None:
        with self._lock:
            self._pending_config = config

    def _should_run(self) -> bool:
        with self._lock:
            return self._running

    def _take_pending_config(self) -> Optional[CameraConfig]:
        with self._lock:
            pending = self._pending_config
            self._pending_config = None
        return pending

    def run(self) -> None:
        with self._lock:
            self._running = True
        if self._config.mock:
            self._run_mock()
        else:
            self._run_real()

    def _run_mock(self) -> None:
        config = self._config
        frame_index = 0
        last_time = time.monotonic()
        self.connected.emit({"mode": "mock", "message": "模拟相机已启动"})

        try:
            while self._should_run():
                pending = self._take_pending_config()
                if pending is not None:
                    config = pending
                    self.status.emit("模拟参数已更新")

                left, right = _make_mock_pair(frame_index)
                now = time.monotonic()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                self.frame_ready.emit(
                    FrameBundle(
                        left=left,
                        right=right,
                        left_stats=compute_image_stats(left),
                        right_stats=compute_image_stats(right),
                        fps=fps,
                        left_fail_count=0,
                        right_fail_count=0,
                        trigger_ret_left=0,
                        trigger_ret_right=0,
                        grab_ret_left=0,
                        grab_ret_right=0,
                        settings=config,
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                    )
                )
                frame_index += 1
                self.msleep(33)
        except Exception as exc:
            self.error.emit(f"模拟相机运行失败: {exc}")
        finally:
            self.stopped.emit()

    def _run_real(self) -> None:
        api = None
        cam_left = None
        cam_right = None
        config = self._config
        left_fail_count = 0
        right_fail_count = 0
        last_time = time.monotonic()

        try:
            api = load_mvs_api()
            api.MvCamera.MV_CC_Initialize()
            cam_left, left_access = init_and_open_camera(
                config.left_ip,
                config.left_exposure_us,
                config.left_gain_db,
                api,
            )
            cam_right, right_access = init_and_open_camera(
                config.right_ip,
                config.right_exposure_us,
                config.right_gain_db,
                api,
            )
            payload_size_left = get_payload_size(cam_left, api)
            payload_size_right = get_payload_size(cam_right, api)
            self.connected.emit(
                {
                    "mode": "real",
                    "left_access": left_access,
                    "right_access": right_access,
                    "payload_size_left": payload_size_left,
                    "payload_size_right": payload_size_right,
                }
            )

            while self._should_run():
                pending = self._take_pending_config()
                if pending is not None:
                    config = pending
                    warnings = []
                    warnings.extend(
                        f"左相机 {item}"
                        for item in _set_camera_optics(cam_left, config.left_exposure_us, config.left_gain_db)
                    )
                    warnings.extend(
                        f"右相机 {item}"
                        for item in _set_camera_optics(cam_right, config.right_exposure_us, config.right_gain_db)
                    )
                    self.status.emit("；".join(warnings) if warnings else "相机曝光/增益已应用")

                trigger_ret_left = cam_left.MV_CC_SetCommandValue("TriggerSoftware")
                trigger_ret_right = cam_right.MV_CC_SetCommandValue("TriggerSoftware")
                img_l, grab_ret_left = convert_to_cv2_image(cam_left, payload_size_left, api)
                img_r, grab_ret_right = convert_to_cv2_image(cam_right, payload_size_right, api)

                left_fail_count = left_fail_count + 1 if img_l is None else 0
                right_fail_count = right_fail_count + 1 if img_r is None else 0
                now = time.monotonic()
                fps = 1.0 / max(now - last_time, 1e-6)
                last_time = now
                self.frame_ready.emit(
                    FrameBundle(
                        left=img_l,
                        right=img_r,
                        left_stats=compute_image_stats(img_l) if img_l is not None else None,
                        right_stats=compute_image_stats(img_r) if img_r is not None else None,
                        fps=fps,
                        left_fail_count=left_fail_count,
                        right_fail_count=right_fail_count,
                        trigger_ret_left=trigger_ret_left,
                        trigger_ret_right=trigger_ret_right,
                        grab_ret_left=grab_ret_left,
                        grab_ret_right=grab_ret_right,
                        settings=config,
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                    )
                )
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
            self.stopped.emit()


class CaptureSession:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        base_session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
        self.session_id = base_session_id
        self.root = project_dir / SESSION_ROOT_NAME / self.session_id
        suffix = 1
        while self.root.exists():
            self.session_id = f"{base_session_id}_{suffix:02d}"
            self.root = project_dir / SESSION_ROOT_NAME / self.session_id
            suffix += 1
        self.left_dir = self.root / "left"
        self.right_dir = self.root / "right"
        self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir.mkdir(parents=True, exist_ok=True)
        self.samples: List[Dict[str, Any]] = []

    def capture(self, left: np.ndarray, right: np.ndarray, frame: FrameBundle) -> Dict[str, Any]:
        stem = f"{len(self.samples):03d}"
        left_path = self.left_dir / f"{stem}.png"
        right_path = self.right_dir / f"{stem}.png"
        left_ok = cv2.imwrite(str(left_path), left)
        right_ok = cv2.imwrite(str(right_path), right)
        if not left_ok or not right_ok:
            for path in (left_path, right_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise IOError("写入本轮缓存图片失败")

        left_stats = frame.left_stats or compute_image_stats(left)
        right_stats = frame.right_stats or compute_image_stats(right)
        record = {
            "session_id": self.session_id,
            "temp_stem": stem,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "left_temp_path": str(left_path),
            "right_temp_path": str(right_path),
            "left_mean": left_stats.mean,
            "right_mean": right_stats.mean,
            "mean_diff": abs(left_stats.mean - right_stats.mean),
            "left_min": left_stats.min_value,
            "left_max": left_stats.max_value,
            "right_min": right_stats.min_value,
            "right_max": right_stats.max_value,
            "left_std": left_stats.std,
            "right_std": right_stats.std,
            "left_dark_pct": left_stats.dark_pct,
            "right_dark_pct": right_stats.dark_pct,
            "left_saturated_pct": left_stats.saturated_pct,
            "right_saturated_pct": right_stats.saturated_pct,
            "width": left_stats.width,
            "height": left_stats.height,
            "left_exposure_us": frame.settings.left_exposure_us,
            "right_exposure_us": frame.settings.right_exposure_us,
            "left_gain_db": frame.settings.left_gain_db,
            "right_gain_db": frame.settings.right_gain_db,
            "exported": False,
        }
        self.samples.append(record)
        return record

    def undo_last(self) -> Dict[str, Any]:
        if not self.samples:
            raise ValueError("当前没有可撤销的样本")
        if self.samples[-1].get("exported"):
            raise ValueError("最近一组样本已经导出，不能只撤销缓存记录")
        record = self.samples.pop()
        for key in ("left_temp_path", "right_temp_path"):
            try:
                Path(record[key]).unlink(missing_ok=True)
            except OSError:
                pass
        return record

    def unexported_count(self) -> int:
        return sum(1 for sample in self.samples if not sample.get("exported"))

    def is_empty(self) -> bool:
        return not self.samples

    def export_to(self, target_dir: Path) -> Tuple[int, Path]:
        pending = [sample for sample in self.samples if not sample.get("exported")]
        if not pending:
            return 0, target_dir

        left_dir = target_dir / "left"
        right_dir = target_dir / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        next_index = self._next_export_index(left_dir, right_dir)
        exported_at = datetime.now().isoformat(timespec="seconds")
        rows = []

        for offset, sample in enumerate(pending):
            final_stem = f"{next_index + offset:03d}"
            final_left = left_dir / f"{final_stem}.png"
            final_right = right_dir / f"{final_stem}.png"
            shutil.copy2(sample["left_temp_path"], final_left)
            shutil.copy2(sample["right_temp_path"], final_right)
            sample["exported"] = True
            sample["exported_at"] = exported_at
            sample["final_stem"] = final_stem
            sample["left_file"] = str(final_left)
            sample["right_file"] = str(final_right)
            rows.append(
                {
                    "session_id": sample["session_id"],
                    "temp_stem": sample["temp_stem"],
                    "final_stem": final_stem,
                    "captured_at": sample["captured_at"],
                    "exported_at": exported_at,
                    "left_file": str(final_left),
                    "right_file": str(final_right),
                    "left_mean": f"{sample['left_mean']:.4f}",
                    "right_mean": f"{sample['right_mean']:.4f}",
                    "mean_diff": f"{sample['mean_diff']:.4f}",
                    "left_min": sample["left_min"],
                    "left_max": sample["left_max"],
                    "right_min": sample["right_min"],
                    "right_max": sample["right_max"],
                    "left_std": f"{sample['left_std']:.4f}",
                    "right_std": f"{sample['right_std']:.4f}",
                    "left_dark_pct": f"{sample['left_dark_pct']:.4f}",
                    "right_dark_pct": f"{sample['right_dark_pct']:.4f}",
                    "left_saturated_pct": f"{sample['left_saturated_pct']:.4f}",
                    "right_saturated_pct": f"{sample['right_saturated_pct']:.4f}",
                    "width": sample["width"],
                    "height": sample["height"],
                    "left_exposure_us": sample["left_exposure_us"],
                    "right_exposure_us": sample["right_exposure_us"],
                    "left_gain_db": sample["left_gain_db"],
                    "right_gain_db": sample["right_gain_db"],
                }
            )

        manifest_path = target_dir / "capture_manifest.csv"
        write_header = not manifest_path.exists()
        with manifest_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        return len(rows), target_dir

    def delete_cache(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    @staticmethod
    def _next_export_index(left_dir: Path, right_dir: Path) -> int:
        max_index = -1
        for side_dir in (left_dir, right_dir):
            if not side_dir.exists():
                continue
            for path in side_dir.iterdir():
                if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.isdigit():
                    max_index = max(max_index, int(path.stem))
        return max_index + 1


class ImageView(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(QSize(420, 320))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QLabel { background: #16181c; color: #aeb4be; border: 1px solid #30343b; }")

    def set_image(self, img_gray: np.ndarray) -> None:
        self._pixmap = self._to_pixmap(img_gray)
        self._rescale()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @staticmethod
    def _to_pixmap(img_gray: np.ndarray) -> QPixmap:
        if img_gray.ndim == 2:
            rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimage)


class CameraPanel(QGroupBox):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.view = ImageView("等待画面")
        self.metrics: Dict[str, QLabel] = {}
        metric_names = [
            ("mean", "平均灰度"),
            ("range", "最小/最大"),
            ("std", "标准差"),
            ("clip", "过暗/过曝"),
            ("size", "尺寸"),
            ("fail", "失败计数"),
            ("ret", "返回码"),
        ]
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(12)
        form.setLabelAlignment(Qt.AlignRight)
        for key, label in metric_names:
            value = QLabel("-")
            value.setMinimumWidth(96)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metrics[key] = value
            form.addRow(label, value)

        metric_widget = QWidget()
        metric_widget.setLayout(form)
        metric_widget.setMinimumWidth(220)
        metric_widget.setMaximumWidth(240)
        metric_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        layout = QHBoxLayout(self)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(metric_widget, stretch=0, alignment=Qt.AlignTop)

    def update_frame(
        self,
        img: Optional[np.ndarray],
        stats: Optional[ImageStats],
        fail_count: int,
        trigger_ret: int,
        grab_ret: int,
    ) -> None:
        if img is not None:
            self.view.set_image(img)
        if stats is not None:
            self.metrics["mean"].setText(f"{stats.mean:.2f}")
            self.metrics["range"].setText(f"{stats.min_value} / {stats.max_value}")
            self.metrics["std"].setText(f"{stats.std:.2f}")
            self.metrics["clip"].setText(f"{stats.dark_pct:.2f}% / {stats.saturated_pct:.2f}%")
            self.metrics["size"].setText(f"{stats.width} x {stats.height}")
        self.metrics["fail"].setText(str(fail_count))
        self.metrics["ret"].setText(f"T {_format_ret(trigger_ret)} | G {_format_ret(grab_ret)}")


class CaptureMainWindow(QMainWindow):
    def __init__(self, mock: bool = False, auto_start: bool = False) -> None:
        super().__init__()
        self.project_dir = Path(__file__).resolve().parent
        self.mock = mock
        self.session: Optional[CaptureSession] = None
        self.worker: Optional[CameraWorker] = None
        self.latest_frame: Optional[FrameBundle] = None
        self.setWindowTitle("双目图像采集")
        self._build_ui()
        self._wire_shortcuts()
        self._set_running_state(False)
        self._update_session_label()
        if auto_start:
            QTimer.singleShot(150, self.start_capture)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self._build_parameter_group(), stretch=3)
        top_layout.addWidget(self._build_control_group(), stretch=2)
        root_layout.addLayout(top_layout)

        panels_layout = QHBoxLayout()
        self.left_panel = CameraPanel("Left Camera")
        self.right_panel = CameraPanel("Right Camera")
        panels_layout.addWidget(self.left_panel, stretch=1)
        panels_layout.addWidget(self.right_panel, stretch=1)
        root_layout.addLayout(panels_layout, stretch=1)

        bottom_group = QGroupBox("Session")
        bottom_layout = QVBoxLayout(bottom_group)
        self.session_label = QLabel("-")
        self.session_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_label = QLabel("FPS: - | 灰度差: - | 样本: 0")
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(110)
        bottom_layout.addWidget(self.session_label)
        bottom_layout.addWidget(self.summary_label)
        bottom_layout.addWidget(self.log_view)
        root_layout.addWidget(bottom_group)
        self.setCentralWidget(root)
        self.resize(1420, 900)

    def _build_parameter_group(self) -> QGroupBox:
        group = QGroupBox("Camera Parameters")
        grid = QGridLayout(group)
        self.left_ip_edit = QLineEdit(LEFT_IP)
        self.right_ip_edit = QLineEdit(RIGHT_IP)
        self.left_exposure_spin = self._make_float_spin(1.0, 1_000_000.0, LEFT_EXPOSURE_US, " us")
        self.right_exposure_spin = self._make_float_spin(1.0, 1_000_000.0, RIGHT_EXPOSURE_US, " us")
        self.left_gain_spin = self._make_float_spin(0.0, 48.0, LEFT_GAIN_DB, " dB")
        self.right_gain_spin = self._make_float_spin(0.0, 48.0, RIGHT_GAIN_DB, " dB")

        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(QLabel("Left"), 0, 1)
        grid.addWidget(QLabel("Right"), 0, 2)
        grid.addWidget(QLabel("IP"), 1, 0)
        grid.addWidget(self.left_ip_edit, 1, 1)
        grid.addWidget(self.right_ip_edit, 1, 2)
        grid.addWidget(QLabel("Exposure"), 2, 0)
        grid.addWidget(self.left_exposure_spin, 2, 1)
        grid.addWidget(self.right_exposure_spin, 2, 2)
        grid.addWidget(QLabel("Gain"), 3, 0)
        grid.addWidget(self.left_gain_spin, 3, 1)
        grid.addWidget(self.right_gain_spin, 3, 2)
        self.apply_params_btn = QPushButton("应用参数")
        self.apply_params_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.apply_params_btn.clicked.connect(self.apply_parameters)
        grid.addWidget(self.apply_params_btn, 4, 2)
        return group

    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("Controls")
        layout = QGridLayout(group)
        self.start_btn = QPushButton("启动模拟" if self.mock else "连接相机")
        self.start_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_btn.clicked.connect(self.stop_capture)
        self.capture_btn = QPushButton("拍摄")
        self.capture_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.capture_btn.clicked.connect(self.capture_sample)
        self.undo_btn = QPushButton("撤销上一组")
        self.undo_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.undo_btn.clicked.connect(self.undo_last)
        self.export_btn = QPushButton("导出")
        self.export_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.export_btn.clicked.connect(self.export_session)
        self.exit_btn = QPushButton("退出")
        self.exit_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        self.exit_btn.clicked.connect(self.close)
        buttons = [self.start_btn, self.stop_btn, self.capture_btn, self.undo_btn, self.export_btn, self.exit_btn]
        for button in buttons:
            button.setAutoDefault(False)
            button.setDefault(False)
        for index, button in enumerate(buttons):
            layout.addWidget(button, index // 2, index % 2)
        return group

    def _wire_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.capture_sample)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_last)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self.export_session)

    @staticmethod
    def _make_float_spin(minimum: float, maximum: float, value: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(1)
        spin.setSingleStep(1000.0 if suffix.strip() == "us" else 0.5)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def _read_config(self) -> CameraConfig:
        return CameraConfig(
            left_ip=self.left_ip_edit.text().strip(),
            right_ip=self.right_ip_edit.text().strip(),
            left_exposure_us=self.left_exposure_spin.value(),
            right_exposure_us=self.right_exposure_spin.value(),
            left_gain_db=self.left_gain_spin.value(),
            right_gain_db=self.right_gain_spin.value(),
            mock=self.mock,
        )

    def start_capture(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        config = self._read_config()
        if not self.mock and (not config.left_ip or not config.right_ip):
            QMessageBox.warning(self, "参数不完整", "请填写左右相机 IP。")
            return
        self.latest_frame = None
        self.worker = CameraWorker(config)
        self.worker.connected.connect(self.on_worker_connected)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status.connect(self.log)
        self.worker.error.connect(self.on_worker_error)
        self.worker.stopped.connect(self.on_worker_stopped)
        self.worker.start()
        self._set_running_state(True)
        self.log("正在启动采集线程...")

    def stop_capture(self) -> None:
        if self.worker is None:
            return
        self.log("正在停止采集线程...")
        self.worker.stop()
        self.worker.wait(3500)
        if self.worker is not None and self.worker.isRunning():
            self.log("采集线程仍在等待相机超时返回，请稍候。")

    def apply_parameters(self) -> None:
        config = self._read_config()
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_settings(config)
            self.log("参数更新已提交")
        else:
            self.log("参数已更新，下一次启动时生效")

    def on_worker_connected(self, info: Dict[str, Any]) -> None:
        self._ensure_session()
        if info.get("mode") == "mock":
            self.log("模拟相机已启动")
        else:
            self.log(
                "相机已连接 | "
                f"left={info.get('left_access')} payload={info.get('payload_size_left')} | "
                f"right={info.get('right_access')} payload={info.get('payload_size_right')}"
            )
        self._update_session_label()

    def on_worker_error(self, message: str) -> None:
        self.log(f"错误: {message}")
        QMessageBox.critical(self, "采集错误", message)

    def on_worker_stopped(self) -> None:
        self._set_running_state(False)
        self.worker = None
        self.log("采集线程已停止")

    def on_frame_ready(self, frame: FrameBundle) -> None:
        self.latest_frame = frame
        self.left_panel.update_frame(
            frame.left,
            frame.left_stats,
            frame.left_fail_count,
            frame.trigger_ret_left,
            frame.grab_ret_left,
        )
        self.right_panel.update_frame(
            frame.right,
            frame.right_stats,
            frame.right_fail_count,
            frame.trigger_ret_right,
            frame.grab_ret_right,
        )
        mean_diff = "-"
        if frame.left_stats is not None and frame.right_stats is not None:
            mean_diff = f"{abs(frame.left_stats.mean - frame.right_stats.mean):.2f}"
        sample_count = len(self.session.samples) if self.session is not None else 0
        unexported = self.session.unexported_count() if self.session is not None else 0
        self.summary_label.setText(
            f"FPS: {frame.fps:.1f} | 灰度差: {mean_diff} | 样本: {sample_count} | 待导出: {unexported}"
        )
        self.capture_btn.setEnabled(frame.left is not None and frame.right is not None)

    def capture_sample(self) -> None:
        if self.latest_frame is None or self.latest_frame.left is None or self.latest_frame.right is None:
            return
        self._ensure_session()
        assert self.session is not None
        try:
            record = self.session.capture(self.latest_frame.left.copy(), self.latest_frame.right.copy(), self.latest_frame)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            self.log(f"保存失败: {exc}")
            return
        self.log(
            f"已拍摄 {record['temp_stem']} | "
            f"L={record['left_mean']:.2f} R={record['right_mean']:.2f} diff={record['mean_diff']:.2f}"
        )
        self._refresh_action_buttons()
        self._update_session_label()

    def undo_last(self) -> None:
        if self.session is None or not self.session.samples:
            return
        try:
            record = self.session.undo_last()
        except ValueError as exc:
            QMessageBox.information(self, "无法撤销", str(exc))
            self.log(f"撤销被拒绝: {exc}")
            return
        except Exception as exc:
            QMessageBox.critical(self, "撤销失败", str(exc))
            self.log(f"撤销失败: {exc}")
            return
        self.log(f"已撤销 {record['temp_stem']}")
        QTimer.singleShot(0, self._refresh_action_buttons)
        self._update_session_label()

    def export_session(self) -> None:
        if self.session is None or not self.session.samples:
            QMessageBox.information(self, "没有样本", "当前没有可导出的采集样本。")
            return
        if self.session.unexported_count() <= 0:
            QMessageBox.information(self, "无需导出", "当前样本都已经导出。")
            return
        target = QFileDialog.getExistingDirectory(self, "选择最终保存文件夹", str(self.project_dir))
        if not target:
            return
        try:
            count, target_dir = self.session.export_to(Path(target))
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            self.log(f"导出失败: {exc}")
            return
        previous_session = self.session
        self.log(f"已导出 {count} 组样本到 {target_dir}")
        QMessageBox.information(
            self,
            "导出完成",
            f"已导出 {count} 组样本。\n\n左图: {Path(target_dir) / 'left'}\n右图: {Path(target_dir) / 'right'}",
        )
        previous_session_id = previous_session.session_id
        self.session = CaptureSession(self.project_dir)
        self.log(f"已结束会话 {previous_session_id}")
        remove_reply = QMessageBox.question(
            self,
            "删除上一轮缓存",
            f"是否删除上一轮缓存？\n\n{previous_session.root}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if remove_reply == QMessageBox.Yes:
            try:
                previous_session.delete_cache()
            except Exception as exc:
                QMessageBox.warning(self, "删除缓存失败", str(exc))
                self.log(f"删除上一轮缓存失败: {exc}")
            else:
                self.log(f"已删除上一轮缓存: {previous_session.root}")
        else:
            self.log(f"保留上一轮缓存: {previous_session.root}")
        self._delete_empty_session_if_needed()
        if self.session is None:
            self.log("当前无活动会话，下一次拍摄时将自动创建新会话")
        self._refresh_action_buttons()
        self._update_session_label()

    def _delete_empty_session_if_needed(self) -> None:
        if self.session is None or not self.session.is_empty():
            return
        empty_session_root = self.session.root
        empty_session_id = self.session.session_id
        try:
            self.session.delete_cache()
        except Exception as exc:
            self.log(f"删除空会话失败 {empty_session_id}: {exc}")
            return
        self.log(f"已删除空会话缓存: {empty_session_root}")
        self.session = None
        self._refresh_action_buttons()
        self._update_session_label()

    def _ensure_session(self) -> None:
        if self.session is None:
            self.session = CaptureSession(self.project_dir)
            self.log(f"本轮缓存目录: {self.session.root}")
            self._update_session_label()

    def _set_running_state(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.capture_btn.setEnabled(False)
        self._refresh_action_buttons()

    def _refresh_action_buttons(self) -> None:
        has_samples = self.session is not None and bool(self.session.samples)
        can_undo = has_samples and not self.session.samples[-1].get("exported")
        can_export = self.session is not None and self.session.unexported_count() > 0
        self.undo_btn.setEnabled(can_undo)
        self.export_btn.setEnabled(can_export)

    def _update_session_label(self) -> None:
        if self.session is None:
            self.session_label.setText("本轮缓存目录: -")
            return
        self.session_label.setText(
            f"本轮缓存目录: {self.session.root} | "
            f"已拍: {len(self.session.samples)} | 待导出: {self.session.unexported_count()}"
        )

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.session is not None and self.session.unexported_count() > 0:
            reply = QMessageBox.question(
                self,
                "还有未导出样本",
                "当前还有未导出的采集样本，确认退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self.stop_capture()
        self._delete_empty_session_if_needed()
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双目相机 PyQt 采集工具")
    parser.add_argument("--mock", action="store_true", help="使用合成图像测试界面和保存流程")
    parser.add_argument("--auto-start", action="store_true", help="启动后自动连接相机或启动模拟源")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication([sys.argv[0]])
    window = CaptureMainWindow(mock=args.mock, auto_start=args.auto_start)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
