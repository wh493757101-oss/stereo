"""
推理面板 - 双阶段分割 + 立体匹配 + 实时显示。

接收 CameraThread 帧流，推理在独立的 QThread 工作线程中执行：
- GUI 线程只提交最新帧（容量 1 的"最新帧"槽，旧帧被直接替换/丢弃）
- 工作线程调用 DualStageInferenceEngine.process_frame_detailed
- 结果通过 Signal 返回 GUI 线程刷新显示

Stage 1: YOLO-A (3ch 灰度) -> mask + bbox
Stage 2: SGBM 匹配 -> disparity -> polar feature
Stage 3: YOLO-B（输入模式 polar/gray 可选）-> 材质分类
"""

import dataclasses
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.stereo_matching import StereoMatcher
from gui.camera_thread import CameraThread, FrameBundle
from gui.capture_panel import ImageView
from gui.inference_engine import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_BASELINE_M,
    DEFAULT_FOCAL_PX,
    DetailedInferenceResult,
    DualStageInferenceEngine,
)
from models.segmentation import Instance


def _load_gui_defaults(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """UI 默认值取自 configs/default.yaml；文件缺失时回退到引擎常量。"""
    defaults = {
        "model_a_path": "runs/train/run_20260913_initial/model_a/weights/best.pt",
        "model_b_path": "runs/train/run_20260913_initial/model_b-gray/weights/best.pt",
        "model_b_input_mode": "gray",
        "baseline": DEFAULT_BASELINE_M,
        "focal_length": DEFAULT_FOCAL_PX,
        "max_disp": 768,
        "block_size": 7,
        "sync_skew_ms": 2.0,
    }
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError:
        return defaults

    model_a = cfg.get("model_a") or {}
    model_b = cfg.get("model_b") or {}
    calibration = cfg.get("calibration") or {}
    stereo = cfg.get("stereo") or {}
    runtime = cfg.get("runtime") or {}
    if model_a.get("path"):
        defaults["model_a_path"] = str(model_a["path"])
    if model_b.get("path"):
        defaults["model_b_path"] = str(model_b["path"])
    if model_b.get("input_mode"):
        defaults["model_b_input_mode"] = str(model_b["input_mode"])
    if "baseline" in calibration:
        defaults["baseline"] = float(calibration["baseline"])
    if "focal_length" in calibration:
        defaults["focal_length"] = float(calibration["focal_length"])
    defaults["max_disp"] = int(stereo.get("max_disp", defaults["max_disp"]))
    defaults["block_size"] = int(stereo.get("block_size", defaults["block_size"]))
    defaults["sync_skew_ms"] = float(runtime.get("sync_skew_ms", defaults["sync_skew_ms"]))
    return defaults


def _resolve_model_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def _format_depth(depth: Optional[float]) -> str:
    if depth is None:
        return "unavailable"
    return f"{depth:.2f}m"


def _format_depths(depths: list[dict]) -> str:
    if not depths:
        return "--"
    parts = []
    for d in depths:
        depth = d.get("depth")
        if d.get("valid") and depth is not None:
            parts.append(f"#{d['instance_id']}: {_format_depth(depth)}")
        else:
            reason = d.get("reason") or "invalid"
            parts.append(f"#{d['instance_id']}: N/A ({reason})")
    return ", ".join(parts)


def _draw_instances(
    image: np.ndarray,
    instances: list[Instance],
    depth_results: Optional[list[dict]] = None,
) -> np.ndarray:
    vis = image.copy()
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for i, inst in enumerate(instances):
        color = colors[i % len(colors)]
        mask_overlay = np.zeros_like(vis)
        mask_overlay[inst.mask > 128] = color
        vis = cv2.addWeighted(vis, 1.0, mask_overlay, 0.4, 0)
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = inst.class_name
        if inst.confidence > 0:
            label += f" {inst.confidence:.2f}"
        if depth_results and i < len(depth_results):
            d = depth_results[i]
            if d.get("valid") and d.get("depth") is not None:
                label += f" | {d['depth']:.2f}m"
            else:
                label += " | depth N/A"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return vis


class InferenceWorker(QThread):
    """在独立线程运行推理；只保留最新提交的一帧。

    submit() 在推理忙碌时用新帧覆盖未处理的旧帧（容量 1），
    队列永不增长；异常通过 error 信号上抛，不阻塞 GUI 线程。

    ``generation`` 标识创建该 worker 的引擎会话；结果携带它，
    面板据此丢弃已停机会话的迟到结果。
    """

    result_ready = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        engine: DualStageInferenceEngine,
        generation: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine = engine
        self._generation = int(generation)
        self._cond = threading.Condition()
        self._pending: Optional[FrameBundle] = None
        self._stopping = False

    def submit(self, bundle: FrameBundle) -> None:
        with self._cond:
            self._pending = bundle
            self._cond.notify_all()

    def stop(self) -> None:
        """Stop the worker and wait until the thread has fully finished.

        The wait is unconditional: stop() never returns while the thread
        (including an in-flight inference call) is still running.
        """
        with self._cond:
            self._stopping = True
            self._pending = None
            self._cond.notify_all()
        self.wait()

    def run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopping:
                    self._cond.wait(timeout=0.1)
                if self._stopping:
                    return
                bundle = self._pending
                self._pending = None
            try:
                result = self._engine.process_frame_detailed(
                    bundle.left,
                    bundle.right,
                    sync_skew_ms=bundle.sync_skew_ms,
                    already_rectified=bundle.rectified,
                )
            except Exception as exc:
                self.error.emit(f"Inference error: {exc}")
                continue
            self.result_ready.emit({
                "generation": self._generation,
                "bundle": bundle,
                "result": result,
            })


class InferencePanel(QWidget):
    """双阶段推理 + 显示面板。"""

    status_msg = Signal(str)

    def __init__(self, camera: CameraThread, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._camera = camera
        self._wired_camera: Optional[CameraThread] = None
        self._defaults = _load_gui_defaults()
        self._engine: Optional[DualStageInferenceEngine] = None
        self._worker: Optional[InferenceWorker] = None
        self._engine_generation = 0
        self._last_bundle: Optional[FrameBundle] = None
        self._last_result: Optional[DetailedInferenceResult] = None
        self._fps_timer = time.time()
        self._frame_count = 0
        self._running = False

        self._build_ui()
        self._wire_camera_signals()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        # ---- 左侧：2x2 紧凑网格（左右目在上，Polar 在下），适应 1500x900 窗口 ----
        image_grid = QGridLayout()
        image_grid.setSpacing(4)

        left_group = QGroupBox("左目 - 分割 + 材质 (校正后)")
        left_layout = QVBoxLayout(left_group)
        self.left_view = ImageView()
        self.left_view.setMinimumSize(320, 240)
        left_layout.addWidget(self.left_view)
        image_grid.addWidget(left_group, 0, 0)

        right_group = QGroupBox("右目 - 校正后")
        right_layout = QVBoxLayout(right_group)
        self.right_view = ImageView()
        self.right_view.setMinimumSize(320, 240)
        right_layout.addWidget(self.right_view)
        image_grid.addWidget(right_group, 0, 1)

        polar_group = QGroupBox("Polar 特征图")
        polar_layout = QVBoxLayout(polar_group)
        self.polar_view = ImageView()
        self.polar_view.setMinimumSize(320, 180)
        polar_layout.addWidget(self.polar_view)
        image_grid.addWidget(polar_group, 1, 0, 1, 2)

        image_grid.setRowStretch(0, 3)
        image_grid.setRowStretch(1, 2)
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)

        image_widget = QWidget()
        image_widget.setLayout(image_grid)
        layout.addWidget(image_widget, stretch=3)

        # ---- 右侧：控制面板 ----
        control_panel = QVBoxLayout()
        control_panel.setSpacing(8)

        model_a_group = QGroupBox("Model A (Stage 1 - 灰度分割)")
        model_a_layout = QVBoxLayout(model_a_group)
        model_a_layout.addWidget(QLabel("Model A 路径:"))
        self.model_a_edit = QLineEdit(self._defaults["model_a_path"])
        model_a_layout.addWidget(self.model_a_edit)
        control_panel.addWidget(model_a_group)

        model_b_group = QGroupBox("Model B (Stage 3 - 材质分类)")
        model_b_layout = QVBoxLayout(model_b_group)
        model_b_layout.addWidget(QLabel("Model B 路径:"))
        self.model_b_edit = QLineEdit(self._defaults["model_b_path"])
        model_b_layout.addWidget(self.model_b_edit)
        self.model_b_mode_combo = QComboBox()
        self.model_b_mode_combo.addItem("Polar ([gray, polar, gray])", "polar")
        self.model_b_mode_combo.addItem("Gray ([gray, gray, gray])", "gray")
        default_mode = self._defaults.get("model_b_input_mode", "polar")
        self.model_b_mode_combo.setCurrentIndex(
            max(self.model_b_mode_combo.findData(default_mode), 0)
        )
        mode_form = QFormLayout()
        mode_form.addRow("输入模式:", self.model_b_mode_combo)
        model_b_layout.addLayout(mode_form)
        control_panel.addWidget(model_b_group)

        calib_group = QGroupBox("标定")
        calib_layout = QHBoxLayout(calib_group)
        calib_layout.addWidget(QLabel("基线(m):"))
        self.baseline_edit = QLineEdit(f"{self._defaults['baseline']:.6f}")
        calib_layout.addWidget(self.baseline_edit)
        calib_layout.addWidget(QLabel("焦距(px):"))
        self.focal_edit = QLineEdit(f"{self._defaults['focal_length']:.2f}")
        calib_layout.addWidget(self.focal_edit)
        control_panel.addWidget(calib_group)

        stereo_group = QGroupBox("立体匹配")
        stereo_form = QFormLayout(stereo_group)
        self.max_disp_spin = QSpinBox()
        self.max_disp_spin.setRange(16, 1024)
        self.max_disp_spin.setSingleStep(16)
        self.max_disp_spin.setValue(int(self._defaults["max_disp"]))
        self.max_disp_spin.setKeyboardTracking(False)
        stereo_form.addRow("最大视差(px):", self.max_disp_spin)
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3, 21)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(int(self._defaults["block_size"]))
        self.block_size_spin.setKeyboardTracking(False)
        stereo_form.addRow("窗口(px):", self.block_size_spin)
        control_panel.addWidget(stereo_group)

        self.sync_group = QGroupBox("同步")
        sync_form = QFormLayout(self.sync_group)
        self.sync_threshold_label = QLabel(f"{self._defaults['sync_skew_ms']:.2f} ms")
        sync_form.addRow("允许偏差:", self.sync_threshold_label)
        self.sync_label = QLabel("同步: --")
        sync_form.addRow("当前:", self.sync_label)
        control_panel.addWidget(self.sync_group)

        self.rectified_note = QLabel("")
        self.rectified_note.setWordWrap(True)
        self.rectified_note.setStyleSheet("color: #888;")
        control_panel.addWidget(self.rectified_note)

        control_panel.addStretch()

        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout(status_group)
        self.fps_label = QLabel("FPS: --")
        self.instance_label = QLabel("Targets: --")
        self.depth_label = QLabel("距离: --")
        self.material_label = QLabel("材质: --")
        self.diag_label = QLabel("Camera: --")
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.instance_label)
        status_layout.addWidget(self.depth_label)
        status_layout.addWidget(self.material_label)
        status_layout.addWidget(self.diag_label)
        control_panel.addWidget(status_group)

        layout.addLayout(control_panel, stretch=1)

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
            camera.error.disconnect(self.status_msg.emit)
        except (TypeError, RuntimeError):
            pass

    def _wire_camera_signals(self) -> None:
        self._camera.frame_ready.connect(self._on_frame)
        self._camera.error.connect(self.status_msg.emit)
        self._wired_camera = self._camera

    # ---- engine management ----

    def _read_config(self) -> dict:
        """UI 当前选择的引擎构建参数（含 Model B 路径与输入模式）。"""
        return dict(
            model_a_path=str(_resolve_model_path(self.model_a_edit.text().strip())),
            model_b_path=str(_resolve_model_path(self.model_b_edit.text().strip())),
            model_b_input_mode=str(self.model_b_mode_combo.currentData()),
            baseline=float(self.baseline_edit.text()),
            focal_length=float(self.focal_edit.text()),
        )

    def init_engine(self) -> bool:
        model_a = self.model_a_edit.text().strip()
        model_b = self.model_b_edit.text().strip()
        if not model_a:
            self.status_msg.emit("请输入 Model A 路径")
            return False
        if not model_b:
            self.status_msg.emit("请输入 Model B 路径")
            return False
        try:
            overrides = self._read_config()
        except ValueError:
            self.status_msg.emit("标定参数无效")
            return False

        # 采集面板配置的最大同步偏差是有效数据：构建引擎时优先采用。
        cam_cfg = getattr(self._camera, "_config", None)
        cam_max_skew = getattr(cam_cfg, "max_sync_skew_ms", None)
        if cam_max_skew is not None:
            overrides["max_sync_skew_ms"] = float(cam_max_skew)

        try:
            # 设备选择由引擎内部通过 torch 解析（from_config 的 runtime.device），
            # UI 只传递显式覆盖项（含 Model B 输入模式）。
            self._engine = DualStageInferenceEngine.from_config(
                DEFAULT_CONFIG_PATH, **overrides
            )
        except Exception as exc:
            self.status_msg.emit(f"引擎初始化失败: {exc}")
            self._engine = None
            return False

        # UI 的视差/窗口参数覆盖 YAML stereo 配置。
        matcher_cfg = dataclasses.replace(
            self._engine.matcher.config,
            max_disparity=self.max_disp_spin.value(),
            block_size=self.block_size_spin.value(),
        )
        self._engine.matcher = StereoMatcher(matcher_cfg)

        self.sync_threshold_label.setText(
            f"{self._engine.max_sync_skew_ms:.2f} ms"
        )

        self._engine_generation += 1
        self._worker = InferenceWorker(self._engine, generation=self._engine_generation)
        self._worker.result_ready.connect(self._on_inference_result)
        self._worker.error.connect(self.status_msg.emit)
        self._worker.start()

        self._running = True
        self._frame_count = 0
        self._fps_timer = time.time()
        return True

    def deinit_engine(self) -> None:
        self._running = False
        if self._worker is not None:
            # 无条件等待当前推理调用结束，stop() 不会在线程仍运行时返回。
            self._worker.stop()
            self._worker = None
        # 递增代会话代号：已在事件队列中的迟到结果将被 _on_inference_result 丢弃。
        self._engine_generation += 1
        self._engine = None
        self._last_result = None
        self._last_bundle = None
        self.fps_label.setText("FPS: --")
        self.instance_label.setText("Targets: --")
        self.depth_label.setText("距离: --")
        self.material_label.setText("材质: --")
        self.diag_label.setText("Camera: --")
        self.sync_label.setText("同步: --")
        self.left_view.clear()
        self.polar_view.clear()
        self.right_view.clear()

    # ---- frame handling (GUI thread; never runs inference here) ----

    def _on_frame(self, frame: FrameBundle) -> None:
        if frame.left is None or frame.right is None:
            return

        self._last_bundle = frame

        if self._running and self._worker is not None:
            # 容量 1 的最新帧槽：忙碌时旧帧被覆盖，永不排队堆积。
            self._worker.submit(frame)

        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self._frame_count = 0
            self._fps_timer = time.time()

        hs = frame.left_stats
        if hs:
            self.diag_label.setText(
                f"L mean={hs.mean:.0f} dark={hs.dark_pct:.1f}% sat={hs.saturated_pct:.1f}%"
            )

    # ---- worker results ----

    def _on_inference_result(self, payload: object) -> None:
        # 丢弃已停机会话的迟到结果（worker 停止前已入队的事件）。
        generation = payload.get("generation", self._engine_generation) \
            if isinstance(payload, dict) else self._engine_generation
        if generation != self._engine_generation:
            return
        bundle: FrameBundle = payload["bundle"]
        result: DetailedInferenceResult = payload["result"]
        self._last_bundle = bundle
        self._last_result = result

        self.instance_label.setText(f"Targets: {len(result.instances)}")
        self.depth_label.setText("距离: " + _format_depths(result.depths))

        if result.instances:
            m_strs = [
                f"#{i + 1}: {inst.class_name}" for i, inst in enumerate(result.instances)
            ]
            self.material_label.setText("材质: " + " | ".join(m_strs))
        else:
            self.material_label.setText("材质: --")

        if bundle.rectified:
            self.rectified_note.setText("输入已校正 (跳过在线重映射)")
        else:
            self.rectified_note.setText("")

        # None = 未测量（如硬件触发/回放），不能当作已测 0 值展示。
        skew = result.sync_skew_ms
        if skew is None:
            self.sync_label.setText("同步: 不可用（未测量）")
        else:
            threshold = self._engine.max_sync_skew_ms if self._engine else None
            if threshold is not None and abs(skew) > threshold:
                self.sync_label.setText(f"同步: {skew:.2f} ms (超限)")
            else:
                self.sync_label.setText(f"同步: {skew:.2f} ms")

        self.refresh_display()

    # ---- display ----

    def refresh_display(self) -> None:
        """Called by external timer or after each inference result."""
        result = self._last_result
        if result is not None:
            left = cv2.cvtColor(result.left_gray, cv2.COLOR_GRAY2BGR)
            if result.instances:
                left = _draw_instances(left, result.instances, result.depths)
            self.left_view.set_image(left)

            if result.polar_map is not None:
                self.polar_view.set_image(_polar_to_colormap(result.polar_map))

            self.right_view.set_image(result.right_gray)
        elif self._last_bundle is not None:
            frame = self._last_bundle
            if frame.left_bgr is not None:
                self.left_view.set_image(frame.left_bgr)
            if frame.right_bgr is not None:
                self.right_view.set_image(frame.right_bgr)


def _polar_to_colormap(polar: np.ndarray) -> np.ndarray:
    vis = (np.clip(polar, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)
