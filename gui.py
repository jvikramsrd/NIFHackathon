"""
gui.py — Geo-Intel Pipeline Desktop App
Run with: python gui.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────


class PipelineTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["preprocess", "train_stage1", "train_stage2", "train_all", "evaluate", "infer"]
        )
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Data root picker
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Data Root:"))
        self.data_label = QLabel(str(ROOT / "dataset"))
        self.data_label.setStyleSheet("border: 1px solid gray; padding: 2px; background: white;")
        data_row.addWidget(self.data_label, stretch=1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._pick_data_root)
        data_row.addWidget(browse_btn)
        layout.addLayout(data_row)

        # Run / Stop row
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedHeight(36)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Live log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Courier New", 9))
        layout.addWidget(self.log_view, stretch=1)

        # QProcess for subprocess
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)

    def _pick_data_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Root", str(ROOT))
        if d:
            self.data_label.setText(d)

    def _run(self):
        mode = self.mode_combo.currentText()
        data_root = self.data_label.text()
        self.log_view.clear()
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.process.start(
            sys.executable,
            [str(ROOT / "run_pipeline.py"), "--mode", mode, "--data_root", data_root],
        )

    def _stop(self):
        self.process.kill()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_view.appendPlainText("\n[Stopped by user]")

    def _on_stdout(self):
        text = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._append_log(text)

    def _on_stderr(self):
        text = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        self._append_log(text)

    def _append_log(self, text: str):
        self.log_view.appendPlainText(text.rstrip())
        for line in text.splitlines():
            if "Ep " in line and "/" in line:
                try:
                    part = line.split("Ep ")[1].split()[0]
                    curr, total = part.split("/")
                    self.progress.setValue(int(int(curr) / int(total) * 100))
                except Exception:
                    pass
            elif "epoch" in line.lower():
                try:
                    import re
                    m = re.search(r"(\d+)/(\d+)", line)
                    if m:
                        self.progress.setValue(int(int(m.group(1)) / int(m.group(2)) * 100))
                except Exception:
                    pass

    def _on_finished(self, exit_code: int, _exit_status):
        self.log_view.appendPlainText(f"\n[Process finished - exit code {exit_code}]")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if exit_code == 0 else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Map / Image Viewer
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_COLORS = np.array(
    [
        [0, 0, 0],        # 0: background - black
        [255, 60, 60],    # 1: building - red
        [160, 160, 160],  # 2: road - gray
        [60, 100, 255],   # 3: waterbody - blue
    ],
    dtype=np.uint8,
)


class MapViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        open_btn = QPushButton("Open TIF...")
        open_btn.clicked.connect(self._open_tif)
        toolbar.addWidget(open_btn)

        load_mask_btn = QPushButton("Load Mask...")
        load_mask_btn.clicked.connect(self._load_mask)
        toolbar.addWidget(load_mask_btn)

        toolbar.addWidget(QLabel("Overlay opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFixedWidth(140)
        self.opacity_slider.valueChanged.connect(self._update_display)
        toolbar.addWidget(self.opacity_slider)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Side-by-side image panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_label = QLabel("Open a TIF to begin")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_label.setStyleSheet("background: #111; color: #aaa;")
        self.left_label.setMinimumSize(300, 300)

        self.right_label = QLabel("Load a prediction mask to see overlay")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_label.setStyleSheet("background: #111; color: #aaa;")
        self.right_label.setMinimumSize(300, 300)

        splitter.addWidget(self.left_label)
        splitter.addWidget(self.right_label)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        # Legend
        legend_row = QHBoxLayout()
        for color_hex, name in [("#3c3c3c", "Background"), ("#ff3c3c", "Building"),
                                 ("#a0a0a0", "Road"), ("#3c64ff", "Waterbody")]:
            dot = QLabel("*")
            dot.setStyleSheet(f"color: {color_hex}; font-size: 18px; font-weight: bold;")
            legend_row.addWidget(dot)
            legend_row.addWidget(QLabel(name))
            legend_row.addSpacing(12)
        legend_row.addStretch()
        layout.addLayout(legend_row)

        self._rgb: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    def _open_tif(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GeoTIFF", str(ROOT / "dataset"), "GeoTIFF (*.tif *.tiff)"
        )
        if not path:
            return
        try:
            import rasterio
            with rasterio.open(path) as src:
                bands = min(src.count, 3)
                data = src.read(list(range(1, bands + 1)))
            rgb = np.transpose(data, (1, 2, 0))
            if rgb.dtype != np.uint8:
                rgb = self._to_uint8(rgb)
            if rgb.shape[2] < 3:
                rgb = np.stack([rgb[:, :, 0]] * 3, axis=-1)
            self._rgb = rgb
            self._update_display()
        except Exception as e:
            self.left_label.setText(f"Error: {e}")

    def _load_mask(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Segmentation Mask", str(ROOT / "outputs"), "Image (*.png *.tif *.tiff)"
        )
        if not path:
            return
        try:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                import rasterio
                with rasterio.open(path) as src:
                    mask = src.read(1)
            self._mask = mask
            self._update_display()
        except Exception as e:
            self.right_label.setText(f"Error: {e}")

    def _update_display(self):
        if self._rgb is None:
            return
        rgb_disp = self._fit(self._rgb, 1024)
        self._set_pixmap(self.left_label, rgb_disp)

        if self._mask is not None:
            mask_disp = cv2.resize(
                self._mask, (rgb_disp.shape[1], rgb_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = _CLASS_COLORS[np.clip(mask_disp, 0, len(_CLASS_COLORS) - 1)]
            alpha = self.opacity_slider.value() / 100.0
            blended = (
                rgb_disp.astype(np.float32) * (1 - alpha)
                + overlay.astype(np.float32) * alpha
            ).clip(0, 255).astype(np.uint8)
            self._set_pixmap(self.right_label, blended)
        else:
            self._set_pixmap(self.right_label, rgb_disp)

    @staticmethod
    def _fit(img: np.ndarray, max_px: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(max_px / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(min(arr.shape[2], 3)):
            ch = arr[:, :, i].astype(np.float32)
            mn, mx = ch.min(), ch.max()
            out[:, :, i] = 0 if mx == mn else (ch - mn) / (mx - mn) * 255
        return out.astype(np.uint8)

    @staticmethod
    def _set_pixmap(label: QLabel, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        label.setPixmap(
            QPixmap.fromImage(img).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────


class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_results)
        browse_btn = QPushButton("Browse Outputs...")
        browse_btn.clicked.connect(lambda: os.startfile(str(ROOT / "outputs")))
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(browse_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Stage", "mIoU", "Dice / F1", "Accuracy", "mAP@0.5"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        fig = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas = FigureCanvas(fig)
        self.ax = fig.add_subplot(111)
        layout.addWidget(self.canvas, stretch=1)

        self._load_results()

    def _load_results(self):
        results_path = ROOT / "outputs" / "results.json"
        if not results_path.exists():
            self.table.setRowCount(0)
            return

        with open(results_path) as f:
            results = json.load(f)

        self.table.setRowCount(0)
        chart_labels: list = []
        chart_values: list = []

        for stage_key, data in results.items():
            row = self.table.rowCount()
            self.table.insertRow(row)

            def _cell(v) -> str:
                return f"{v:.4f}" if isinstance(v, float) else str(v)

            self.table.setItem(row, 0, QTableWidgetItem(stage_key))
            self.table.setItem(row, 1, QTableWidgetItem(_cell(data.get("mIoU", ""))))
            self.table.setItem(row, 2, QTableWidgetItem(_cell(data.get("dice", ""))))
            self.table.setItem(row, 3, QTableWidgetItem(_cell(data.get("accuracy", ""))))
            self.table.setItem(row, 4, QTableWidgetItem(_cell(data.get("mAP_50", ""))))

            primary = data.get("mIoU") or data.get("accuracy") or data.get("mAP_50")
            if isinstance(primary, float):
                chart_labels.append(stage_key)
                chart_values.append(primary)

        self.ax.clear()
        if chart_labels:
            bars = self.ax.bar(chart_labels, chart_values, color=["#4e79a7", "#f28e2b", "#59a14f"])
            self.ax.set_ylim(0, 1)
            self.ax.set_ylabel("Score")
            self.ax.set_title("Primary Metric per Stage")
            for bar, val in zip(bars, chart_values):
                self.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geo-Intel Pipeline")
        self.resize(1100, 780)

        tabs = QTabWidget()
        tabs.addTab(PipelineTab(), "Pipeline Runner")
        tabs.addTab(MapViewerTab(), "Map Viewer")
        tabs.addTab(ResultsTab(), "Results")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
