"""
采集面板 - 双目相机图像采集与保存。

使用 PySide6 实现双相机采集控制与预览。
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from gui.camera_thread import CameraConfig, CameraThread, FrameBundle, ImageStats, compute_image_stats


SESSION_ROOT_NAME = ".capture_sessions"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Session 管理
# ---------------------------------------------------------------------------

class CaptureSession:
    """管理采集样本的缓存、导出和撤销。"""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        base_id = datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
        self.session_id = base_id
        self.root = project_dir / SESSION_ROOT_NAME / self.session_id
        suffix = 1
        while self.root.exists():
            self.session_id = f"{base_id}_{suffix:02d}"
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
        ok_l = cv2.imwrite(str(left_path), left)
        ok_r = cv2.imwrite(str(right_path), right)
        if not ok_l or not ok_r:
            for p in (left_path, right_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            raise IOError("写入缓存图片失败")

        ls = frame.left_stats or compute_image_stats(left)
        rs = frame.right_stats or compute_image_stats(right)
        record = {
            "session_id": self.session_id, "temp_stem": stem,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "left_temp_path": str(left_path), "right_temp_path": str(right_path),
            "left_mean": ls.mean, "right_mean": rs.mean,
            "mean_diff": abs(ls.mean - rs.mean),
            "left_min": ls.min_value, "left_max": ls.max_value,
            "right_min": rs.min_value, "right_max": rs.max_value,
            "left_std": ls.std, "right_std": rs.std,
            "left_dark_pct": ls.dark_pct, "right_dark_pct": rs.dark_pct,
            "left_saturated_pct": ls.saturated_pct, "right_saturated_pct": rs.saturated_pct,
            "width": ls.width, "height": ls.height,
            "left_exposure_us": frame.config.left_exposure_us if frame.config else 0,
            "right_exposure_us": frame.config.right_exposure_us if frame.config else 0,
            "left_gain_db": frame.config.left_gain_db if frame.config else 0,
            "right_gain_db": frame.config.right_gain_db if frame.config else 0,
            "pair_id": getattr(frame, "pair_id", None),
            "left_frame_no": getattr(frame, "left_frame_no", None),
            "right_frame_no": getattr(frame, "right_frame_no", None),
            "left_timestamp_ns": getattr(frame, "left_timestamp_ns", None),
            "right_timestamp_ns": getattr(frame, "right_timestamp_ns", None),
            "sync_skew_ms": getattr(frame, "sync_skew_ms", None),
            "rectified": bool(getattr(frame, "rectified", False)),
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
        return sum(1 for s in self.samples if not s.get("exported"))

    def is_empty(self) -> bool:
        return not self.samples

    def export_to(self, target_dir: Path) -> tuple[int, Path]:
        pending = [s for s in self.samples if not s.get("exported")]
        if not pending:
            return 0, target_dir

        left_dir = target_dir / "left"
        right_dir = target_dir / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        next_idx = self._next_index(left_dir, right_dir)
        exported_at = datetime.now().isoformat(timespec="seconds")
        rows = []

        for offset, sample in enumerate(pending):
            stem = f"{next_idx + offset:03d}"
            fl = left_dir / f"{stem}.png"
            fr = right_dir / f"{stem}.png"
            shutil.copy2(sample["left_temp_path"], fl)
            shutil.copy2(sample["right_temp_path"], fr)
            sample["exported"] = True
            sample["exported_at"] = exported_at
            sample["final_stem"] = stem
            sample["left_file"] = str(fl)
            sample["right_file"] = str(fr)
            rows.append({
                "session_id": sample["session_id"],
                "temp_stem": sample["temp_stem"],
                "final_stem": stem,
                "captured_at": sample["captured_at"],
                "exported_at": exported_at,
                "left_file": str(fl), "right_file": str(fr),
                "left_mean": f"{sample['left_mean']:.4f}",
                "right_mean": f"{sample['right_mean']:.4f}",
                "mean_diff": f"{sample['mean_diff']:.4f}",
                "left_min": sample["left_min"], "right_min": sample["right_min"],
                "left_max": sample["left_max"], "right_max": sample["right_max"],
                "left_std": f"{sample['left_std']:.4f}",
                "right_std": f"{sample['right_std']:.4f}",
                "left_dark_pct": f"{sample['left_dark_pct']:.4f}",
                "right_dark_pct": f"{sample['right_dark_pct']:.4f}",
                "left_saturated_pct": f"{sample['left_saturated_pct']:.4f}",
                "right_saturated_pct": f"{sample['right_saturated_pct']:.4f}",
                "width": sample["width"], "height": sample["height"],
                "left_exposure_us": sample["left_exposure_us"],
                "right_exposure_us": sample["right_exposure_us"],
                "left_gain_db": sample["left_gain_db"],
                "right_gain_db": sample["right_gain_db"],
                # New sync/frame metadata; old records may lack these keys.
                "pair_id": sample.get("pair_id", ""),
                "left_frame_no": sample.get("left_frame_no", ""),
                "right_frame_no": sample.get("right_frame_no", ""),
                "left_timestamp_ns": sample.get("left_timestamp_ns", ""),
                "right_timestamp_ns": sample.get("right_timestamp_ns", ""),
                "sync_skew_ms": (
                    f"{sample['sync_skew_ms']:.3f}"
                    if sample.get("sync_skew_ms") is not None else ""
                ),
                "rectified": sample.get("rectified", ""),
            })

        manifest_path = target_dir / "capture_manifest.csv"
        write_header = not manifest_path.exists()
        with manifest_path.open("a", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        return len(rows), target_dir

    def delete_cache(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    @staticmethod
    def _next_index(left_dir: Path, right_dir: Path) -> int:
        max_idx = -1
        for d in (left_dir, right_dir):
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.isdigit():
                    max_idx = max(max_idx, int(p.stem))
        return max_idx + 1


# ---------------------------------------------------------------------------
# 图像预览组件
# ---------------------------------------------------------------------------

class ImageView(QLabel):
    """自适应缩放的图像预览控件。"""

    def __init__(self, title: str = "") -> None:
        super().__init__(title)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QLabel { background: #16181c; border: 1px solid #30343b; }")

    def set_image(self, img: np.ndarray) -> None:
        self._pixmap = self._to_pixmap(img)
        self._rescale()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    @staticmethod
    def _to_pixmap(img: np.ndarray) -> QPixmap:
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# 单目预览面板
# ---------------------------------------------------------------------------

class _CameraMetricPanel(QGroupBox):
    """单侧相机预览：图像 + 灰度统计指标。"""

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.view = ImageView("等待画面")
        self.metrics: Dict[str, QLabel] = {}
        metric_names = [
            ("mean", "平均灰度"), ("range", "最小/最大"), ("std", "标准差"),
            ("clip", "过暗/过曝"), ("size", "尺寸"),
            ("fail", "失败计数"), ("ret", "返回码"),
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

        def _fmt(v):
            if v is None:
                return "-"
            if v == 0:
                return "0x0"
            return f"0x{v & 0xFFFFFFFF:08X}"
        self.metrics["ret"].setText(f"T {_fmt(trigger_ret)} | G {_fmt(grab_ret)}")


# ---------------------------------------------------------------------------
# 采集面板
# ---------------------------------------------------------------------------

class CapturePanel(QWidget):
    """完整的采集面板：参数配置 + 预览 + 拍摄/撤销/导出。"""

    status_msg = Signal(str)

    DEFAULT_LEFT_IP = "192.168.1.11"
    DEFAULT_RIGHT_IP = "192.168.1.12"
    DEFAULT_EXPOSURE = 40000.0
    DEFAULT_GAIN = 0.0
    DEFAULT_TRIGGER_SOURCE = "software"
    DEFAULT_MAX_SYNC_SKEW_MS = 2.0
    DEFAULT_ALREADY_RECTIFIED = False

    def __init__(self, camera: CameraThread, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._wired_camera: Optional[CameraThread] = None
        self._session: Optional[CaptureSession] = None
        self._latest_frame: Optional[FrameBundle] = None
        self._project_dir = Path(__file__).resolve().parent.parent

        self._build_ui()
        self._wire_camera_signals()
        self._wire_shortcuts()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ---- 参数区 ----
        param_group = QGroupBox("相机参数")
        grid = QGridLayout(param_group)
        self.left_ip_edit = QLineEdit(self.DEFAULT_LEFT_IP)
        self.right_ip_edit = QLineEdit(self.DEFAULT_RIGHT_IP)
        self.left_exposure_spin = self._make_spin(1.0, 1_000_000.0, self.DEFAULT_EXPOSURE, " us")
        self.right_exposure_spin = self._make_spin(1.0, 1_000_000.0, self.DEFAULT_EXPOSURE, " us")
        self.left_gain_spin = self._make_spin(0.0, 48.0, self.DEFAULT_GAIN, " dB")
        self.right_gain_spin = self._make_spin(0.0, 48.0, self.DEFAULT_GAIN, " dB")

        self.trigger_source_combo = QComboBox()
        self.trigger_source_combo.addItem("软件触发", "software")
        self.trigger_source_combo.addItem("硬件触发 Line0", "hardware")
        self.max_skew_spin = QDoubleSpinBox()
        self.max_skew_spin.setRange(0.0, 1000.0)
        self.max_skew_spin.setDecimals(2)
        self.max_skew_spin.setSingleStep(0.5)
        self.max_skew_spin.setValue(self.DEFAULT_MAX_SYNC_SKEW_MS)
        self.max_skew_spin.setSuffix(" ms")
        self.max_skew_spin.setToolTip("超过该左/右时间戳偏差(ms)的帧对不再作为有效测距")
        self.max_skew_spin.setKeyboardTracking(False)
        self.already_rectified_check = QCheckBox("输入已校正（回放已校正数据集时勾选）")

        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(QLabel("左相机"), 0, 1)
        grid.addWidget(QLabel("右相机"), 0, 2)
        grid.addWidget(QLabel("IP"), 1, 0)
        grid.addWidget(self.left_ip_edit, 1, 1)
        grid.addWidget(self.right_ip_edit, 1, 2)
        grid.addWidget(QLabel("曝光时间"), 2, 0)
        grid.addWidget(self.left_exposure_spin, 2, 1)
        grid.addWidget(self.right_exposure_spin, 2, 2)
        grid.addWidget(QLabel("增益"), 3, 0)
        grid.addWidget(self.left_gain_spin, 3, 1)
        grid.addWidget(self.right_gain_spin, 3, 2)
        grid.addWidget(QLabel("触发源"), 4, 0)
        grid.addWidget(self.trigger_source_combo, 4, 1, 1, 2)
        grid.addWidget(QLabel("最大同步偏差"), 5, 0)
        grid.addWidget(self.max_skew_spin, 5, 1, 1, 2)
        grid.addWidget(self.already_rectified_check, 6, 0, 1, 3)

        self.apply_btn = QPushButton("应用参数")
        self.apply_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        grid.addWidget(self.apply_btn, 7, 2)
        layout.addWidget(param_group)

        # ---- 预览区 ----
        panels_layout = QHBoxLayout()
        self.left_panel = _CameraMetricPanel("左相机")
        self.right_panel = _CameraMetricPanel("右相机")
        panels_layout.addWidget(self.left_panel, stretch=1)
        panels_layout.addWidget(self.right_panel, stretch=1)
        layout.addLayout(panels_layout, stretch=1)

        # ---- 控制按钮 ----
        btn_group = QGroupBox("采集控制")
        btn_layout = QGridLayout(btn_group)
        self.capture_btn = QPushButton("拍摄 (Space)")
        self.capture_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.undo_btn = QPushButton("撤销 (Ctrl+Z)")
        self.undo_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.export_btn = QPushButton("导出 (Ctrl+E)")
        self.export_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        for b in [self.capture_btn, self.undo_btn, self.export_btn]:
            b.setAutoDefault(False)
            b.setDefault(False)
        btn_layout.addWidget(self.capture_btn, 0, 0)
        btn_layout.addWidget(self.undo_btn, 0, 1)
        btn_layout.addWidget(self.export_btn, 0, 2)
        layout.addWidget(btn_group)

        # ---- Session 信息 ----
        info_group = QGroupBox("会话")
        info_layout = QVBoxLayout(info_group)
        self.session_label = QLabel("缓存目录: -")
        self.session_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary_label = QLabel("FPS: - | 灰度差: - | 样本: 0 | 待导出: 0")
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_layout.addWidget(self.session_label)
        info_layout.addWidget(self.summary_label)
        layout.addWidget(info_group)

        # Button actions are wired exactly once (the panel keeps its buttons
        # across camera rebinds).
        self.apply_btn.clicked.connect(self._on_apply_params)
        self.capture_btn.clicked.connect(self._on_capture)
        self.undo_btn.clicked.connect(self._on_undo)
        self.export_btn.clicked.connect(self._on_export)

        self._update_action_buttons()

    def set_camera(self, camera: CameraThread) -> None:
        """Rebind the camera thread; only connections made by this panel to
        the previously wired camera are disconnected, so rebinding never
        emits PySide disconnect warnings."""
        if self._wired_camera is not None and self._wired_camera is not camera:
            self._unwire_camera(self._wired_camera)
        self._camera = camera
        if self._wired_camera is not camera:
            self._wire_camera_signals()

    def _unwire_camera(self, camera: CameraThread) -> None:
        try:
            camera.frame_ready.disconnect(self._on_frame)
            camera.status.disconnect(self.status_msg.emit)
            camera.error.disconnect(self.status_msg.emit)
        except (TypeError, RuntimeError):
            pass

    def _wire_camera_signals(self) -> None:
        self._camera.frame_ready.connect(self._on_frame)
        self._camera.status.connect(self.status_msg.emit)
        self._camera.error.connect(self.status_msg.emit)
        self._wired_camera = self._camera

    def _wire_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._on_capture)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._on_undo)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._on_export)

    @staticmethod
    def _make_spin(min_val: float, max_val: float, value: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(1)
        spin.setSingleStep(1000.0 if suffix.strip() == "us" else 0.5)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def _read_config(self) -> CameraConfig:
        trigger = self.trigger_source_combo.currentData()
        return CameraConfig(
            left_ip=self.left_ip_edit.text().strip(),
            right_ip=self.right_ip_edit.text().strip(),
            left_exposure_us=self.left_exposure_spin.value(),
            right_exposure_us=self.right_exposure_spin.value(),
            left_gain_db=self.left_gain_spin.value(),
            right_gain_db=self.right_gain_spin.value(),
            mock=False,
            trigger_source=trigger if trigger in ("software", "hardware") else "software",
            max_sync_skew_ms=self.max_skew_spin.value(),
            already_rectified=self.already_rectified_check.isChecked(),
        )

    # ---- CameraThread 回调 ----

    def _on_frame(self, frame: FrameBundle) -> None:
        self._latest_frame = frame
        self.left_panel.update_frame(
            frame.left, frame.left_stats,
            frame.left_fail_count, frame.trigger_ret_left, frame.grab_ret_left,
        )
        self.right_panel.update_frame(
            frame.right, frame.right_stats,
            frame.right_fail_count, frame.trigger_ret_right, frame.grab_ret_right,
        )
        mean_diff = "-"
        if frame.left_stats and frame.right_stats:
            mean_diff = f"{abs(frame.left_stats.mean - frame.right_stats.mean):.2f}"
        total = len(self._session.samples) if self._session else 0
        unexported = self._session.unexported_count() if self._session else 0
        self.summary_label.setText(
            f"FPS: {frame.fps:.1f} | 灰度差: {mean_diff} | 样本: {total} | 待导出: {unexported}"
        )
        self.capture_btn.setEnabled(frame.left is not None and frame.right is not None)

    # ---- 操作 ----

    def _on_apply_params(self) -> None:
        config = self._read_config()
        if self._camera.isRunning():
            self._camera._config = config
            self.status_msg.emit("参数已应用")
        else:
            self.status_msg.emit("相机未运行，参数将在下次启动时生效")

    def _ensure_session(self) -> CaptureSession:
        if self._session is None:
            self._session = CaptureSession(self._project_dir)
            self.status_msg.emit(f"缓存目录: {self._session.root}")
            self._update_session_label()
        return self._session

    def _on_capture(self) -> None:
        if self._latest_frame is None:
            return
        f = self._latest_frame
        if f.left is None or f.right is None:
            return
        session = self._ensure_session()
        try:
            rec = session.capture(f.left.copy(), f.right.copy(), f)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            self.status_msg.emit(f"保存失败: {exc}")
            return
        self.status_msg.emit(
            f"已拍摄 {rec['temp_stem']} | L={rec['left_mean']:.2f} R={rec['right_mean']:.2f}"
        )
        self._update_action_buttons()
        self._update_session_label()

    def _on_undo(self) -> None:
        if self._session is None or not self._session.samples:
            return
        try:
            rec = self._session.undo_last()
        except ValueError as exc:
            QMessageBox.information(self, "无法撤销", str(exc))
            return
        self.status_msg.emit(f"已撤销 {rec['temp_stem']}")
        self._update_action_buttons()
        self._update_session_label()

    def _on_export(self) -> None:
        if self._session is None or self._session.unexported_count() <= 0:
            QMessageBox.information(self, "无需导出", "当前没有可导出的样本。")
            return
        target = QFileDialog.getExistingDirectory(self, "选择导出目录", str(self._project_dir))
        if not target:
            return
        try:
            count, target_dir = self._session.export_to(Path(target))
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        prev = self._session
        self.status_msg.emit(f"已导出 {count} 组样本到 {target_dir}")
        QMessageBox.information(
            self, "导出完成",
            f"已导出 {count} 组样本。\n\n左图: {target_dir / 'left'}\n右图: {target_dir / 'right'}",
        )
        self._session = CaptureSession(self._project_dir)
        self.status_msg.emit(f"已结束会话 {prev.session_id}")
        reply = QMessageBox.question(
            self, "删除缓存", f"是否删除上一轮缓存？\n\n{prev.root}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                prev.delete_cache()
            except Exception as exc:
                QMessageBox.warning(self, "删除失败", str(exc))
            else:
                self.status_msg.emit(f"已删除缓存: {prev.root}")
        self._update_action_buttons()
        self._update_session_label()

    def _update_action_buttons(self) -> None:
        has = self._session is not None and bool(self._session.samples)
        can_undo = has and not self._session.samples[-1].get("exported")
        can_export = self._session is not None and self._session.unexported_count() > 0
        self.undo_btn.setEnabled(can_undo)
        self.export_btn.setEnabled(can_export)

    def _update_session_label(self) -> None:
        if self._session is None:
            self.session_label.setText("缓存目录: -")
            return
        self.session_label.setText(
            f"缓存目录: {self._session.root} | "
            f"已拍: {len(self._session.samples)} | 待导出: {self._session.unexported_count()}"
        )

    def has_unsaved(self) -> bool:
        return self._session is not None and self._session.unexported_count() > 0

    def cleanup(self) -> None:
        if self._session is not None and self._session.is_empty():
            try:
                self._session.delete_cache()
            except Exception:
                pass
