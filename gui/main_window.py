"""
上位机主窗口 - 统一界面。

通过顶部标签页切换：
- 采集：相机参数配置、实时预览、拍摄/撤销/导出
- 推理：实例分割 + 立体匹配 + 测距显示

两个面板共享同一个 CameraThread 实例。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.camera_thread import CameraConfig, CameraThread
from gui.capture_panel import CapturePanel
from gui.inference_panel import InferencePanel


class MainWindow(QMainWindow):
    """统一主窗口：采集 + 推理标签页。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("双目水下目标检测与测距系统")
        self.resize(1500, 900)

        self._camera: CameraThread | None = None
        self._display_timer: QTimer | None = None
        self._status_label: QLabel | None = None

        # 创建临时相机用于初始化面板（不会启动）
        self._default_config = CameraConfig(mock=True)
        self._camera = CameraThread(source=self._default_config)

        self._capture_panel = CapturePanel(self._camera)
        self._inference_panel = InferencePanel(self._camera)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---- 顶部工具栏 ----
        toolbar_layout = QHBoxLayout()

        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("color: #888;")
        toolbar_layout.addWidget(self._status_label)
        toolbar_layout.addStretch()

        self.start_btn = QPushButton("启动相机")
        self.start_btn.setMinimumHeight(32)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setMinimumHeight(32)
        self.stop_btn.setEnabled(False)
        toolbar_layout.addWidget(self.start_btn)
        toolbar_layout.addWidget(self.stop_btn)

        layout.addLayout(toolbar_layout)

        # ---- 标签页 ----
        self._tabs = QTabWidget()
        self._tabs.addTab(self._capture_panel, "📷 采集")
        self._tabs.addTab(self._inference_panel, "🔍 检测")
        layout.addWidget(self._tabs, stretch=1)

        # ---- 信号 ----
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _on_start(self) -> None:
        config = self._capture_panel._read_config()

        # 重建相机线程（停止旧的）
        if self._camera and self._camera.isRunning():
            self._camera.stop()
        self._camera = CameraThread(source=config)
        self._camera.status.connect(self._on_status)
        self._camera.error.connect(self._on_error)

        # 把新相机传给两个面板（安全重绑定：只断开自己建立的旧连接）
        self._capture_panel.set_camera(self._camera)
        self._inference_panel.set_camera(self._camera)

        # 显示定时器
        self._display_timer = QTimer()
        self._display_timer.timeout.connect(self._update_display)
        self._display_timer.start(33)

        self._camera.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._status_label.setText("正在连接相机...")
        self._status_label.setStyleSheet("color: #cc0;")

    def _on_stop(self) -> None:
        if self._inference_panel:
            self._inference_panel.deinit_engine()

        if self._camera and self._camera.isRunning():
            self._camera.stop()

        if self._display_timer:
            self._display_timer.stop()
            self._display_timer = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._status_label.setText("已停止")
        self._status_label.setStyleSheet("color: #888;")

    def _on_tab_changed(self, index: int) -> None:
        if index == 1 and self._inference_panel and not self._inference_panel._running:
            self._inference_panel.init_engine()
        elif index == 0 and self._inference_panel:
            self._inference_panel.deinit_engine()

    def _on_status(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("color: #0a0;")

    def _on_error(self, msg: str) -> None:
        self._status_label.setText(msg)
        self._status_label.setStyleSheet("color: #c00;")

    def _update_display(self) -> None:
        if self._inference_panel and self._inference_panel._running:
            self._inference_panel.refresh_display()

    def closeEvent(self, event) -> None:
        if self._capture_panel and self._capture_panel.has_unsaved():
            reply = QMessageBox.question(
                self,
                "还有未导出样本",
                "当前采集面板还有未导出的样本，确认退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        self._on_stop()
        if self._capture_panel:
            self._capture_panel.cleanup()
        event.accept()


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication([sys.argv[0]])
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
