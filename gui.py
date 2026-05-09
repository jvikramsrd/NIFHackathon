"""
gui.py — Geo-Intel Pipeline Operator Console
Run with: python gui.py

Design system: dark-only, Geo-Intel palette.
Cross-platform: Windows / macOS / Linux.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QProcess, QTimer
from PyQt6.QtGui import QFont, QFontDatabase, QImage, QPixmap, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSplitter, QStatusBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QFrame,
    QGridLayout, QHeaderView,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ROOT = Path(__file__).parent


def _find_pipeline_python() -> str:
    """Return the Python executable that has the pipeline deps installed.

    When running from a PyInstaller bundle sys.executable points to the frozen
    launcher, which cannot import torch.  We walk PATH looking for a Python
    that can import torch, preferring the one that matches the active venv.
    Falls back to sys.executable (works fine in normal dev mode).
    """
    if not getattr(sys, "frozen", False):
        return sys.executable  # ordinary dev run — sys.executable is correct

    # Bundle mode: probe candidates in priority order.
    candidates: list[str] = []

    # 1. VIRTUAL_ENV / CONDA_PREFIX from the environment the user activated
    for env_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env_root = os.environ.get(env_var)
        if env_root:
            for rel in ("bin/python3", "bin/python", "Scripts/python.exe"):
                p = Path(env_root) / rel
                if p.is_file():
                    candidates.append(str(p))

    # 2. System PATH
    for name in ("python3", "python", "python3.11", "python3.10"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-c", "import torch"],
                capture_output=True, timeout=8,
            )
            if result.returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            continue

    return sys.executable  # last resort

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM PALETTE
# ─────────────────────────────────────────────────────────────────────────────

PAL = {
    "bg_app":      "#070A10",
    "bg_panel":    "#0F141C",
    "bg_surface":  "#161D27",
    "bg_input":    "#1A2230",
    "border":      "#2D3A52",
    "fg_primary":  "#F0F4FA",
    "fg_secondary":"#C8D2E0",
    "fg_tertiary": "#8595AD",
    "fg_quart":    "#5C6878",
    "accent":      "#FFB547",
    # status
    "ok":          "#4ADE80",
    "running":     "#5FD1FF",
    "warn":        "#FBBF24",
    "err":         "#F87171",
    "info":        "#60A5FA",
    # class colours (sacred — never reassign)
    "cls_building":"#E63946",
    "cls_road":    "#8B96AA",
    "cls_water":   "#4A90E2",
    "cls_bg":      "#2D3A52",
}

# Class overlay colours as uint8 RGB arrays (for numpy blending)
_CLASS_COLORS = np.array([
    [0x2D, 0x3A, 0x52],   # 0 background
    [0xE6, 0x39, 0x46],   # 1 building
    [0x8B, 0x96, 0xAA],   # 2 road
    [0x4A, 0x90, 0xE2],   # 3 waterbody
], dtype=np.uint8)

QSS = f"""
QWidget {{
    background-color: {PAL['bg_app']};
    color: {PAL['fg_primary']};
    font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background-color: {PAL['bg_app']};
}}
QTabWidget::pane {{
    border: 1px solid {PAL['border']};
    background-color: {PAL['bg_panel']};
}}
QTabBar::tab {{
    background: {PAL['bg_surface']};
    color: {PAL['fg_tertiary']};
    padding: 8px 20px;
    border: 1px solid {PAL['border']};
    border-bottom: none;
    min-width: 120px;
}}
QTabBar::tab:selected {{
    background: {PAL['bg_panel']};
    color: {PAL['fg_primary']};
    border-top: 2px solid {PAL['accent']};
}}
QTabBar::tab:hover:!selected {{
    color: {PAL['fg_secondary']};
    background: {PAL['bg_input']};
}}
QPushButton {{
    background-color: {PAL['bg_surface']};
    color: {PAL['fg_primary']};
    border: 1px solid {PAL['border']};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: {PAL['bg_input']};
    border-color: {PAL['fg_tertiary']};
}}
QPushButton:pressed {{
    background-color: {PAL['border']};
}}
QPushButton:disabled {{
    color: {PAL['fg_quart']};
    border-color: {PAL['bg_surface']};
}}
QPushButton#run_btn {{
    background-color: {PAL['accent']};
    color: {PAL['bg_app']};
    font-weight: 600;
    border: none;
}}
QPushButton#run_btn:hover {{
    background-color: #FFC76A;
}}
QPushButton#run_btn:disabled {{
    background-color: {PAL['border']};
    color: {PAL['fg_quart']};
}}
QPushButton#stop_btn {{
    background-color: transparent;
    color: {PAL['err']};
    border: 1px solid {PAL['err']};
}}
QPushButton#stop_btn:hover {{
    background-color: rgba(248,113,113,0.12);
}}
QComboBox {{
    background-color: {PAL['bg_input']};
    color: {PAL['fg_primary']};
    border: 1px solid {PAL['border']};
    border-radius: 4px;
    padding: 5px 10px;
    min-width: 160px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {PAL['bg_surface']};
    color: {PAL['fg_primary']};
    selection-background-color: {PAL['border']};
    border: 1px solid {PAL['border']};
}}
QLabel {{
    color: {PAL['fg_secondary']};
    background: transparent;
}}
QLabel#label_heading {{
    color: {PAL['fg_primary']};
    font-size: 14px;
    font-weight: 600;
}}
QLabel#label_mono {{
    color: {PAL['fg_primary']};
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-variant-numeric: tabular-nums;
}}
QLabel#path_label {{
    background-color: {PAL['bg_input']};
    color: {PAL['fg_secondary']};
    border: 1px solid {PAL['border']};
    border-radius: 3px;
    padding: 4px 8px;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-size: 11px;
}}
QPlainTextEdit {{
    background-color: {PAL['bg_app']};
    color: {PAL['fg_secondary']};
    border: 1px solid {PAL['border']};
    border-radius: 3px;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-size: 11px;
    selection-background-color: {PAL['border']};
}}
QProgressBar {{
    background-color: {PAL['bg_input']};
    border: 1px solid {PAL['border']};
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {PAL['accent']};
    border-radius: 3px;
}}
QSlider::groove:horizontal {{
    background: {PAL['bg_input']};
    height: 4px;
    border-radius: 2px;
    border: 1px solid {PAL['border']};
}}
QSlider::handle:horizontal {{
    background: {PAL['accent']};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {PAL['accent']};
    border-radius: 2px;
}}
QTableWidget {{
    background-color: {PAL['bg_panel']};
    alternate-background-color: {PAL['bg_surface']};
    color: {PAL['fg_secondary']};
    border: 1px solid {PAL['border']};
    gridline-color: {PAL['border']};
    font-size: 12px;
}}
QTableWidget::item:selected {{
    background-color: {PAL['border']};
    color: {PAL['fg_primary']};
}}
QHeaderView::section {{
    background-color: {PAL['bg_surface']};
    color: {PAL['fg_tertiary']};
    border: none;
    border-bottom: 1px solid {PAL['border']};
    border-right: 1px solid {PAL['border']};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QScrollBar:vertical {{
    background: {PAL['bg_app']};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {PAL['border']};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {PAL['bg_app']};
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {PAL['border']};
    border-radius: 4px;
}}
QSplitter::handle {{
    background: {PAL['border']};
    width: 1px;
}}
QFrame#divider {{
    background: {PAL['border']};
    max-height: 1px;
}}
QStatusBar {{
    background: {PAL['bg_panel']};
    color: {PAL['fg_tertiary']};
    border-top: 1px solid {PAL['border']};
    font-size: 11px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def open_folder(path: Path | str):
    """Open a folder in the system file manager, cross-platform."""
    p = str(Path(path).resolve())
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", p], check=False)
    else:
        # Use the XDG Desktop Portal via gdbus — proper Wayland-native IPC.
        # The portal brokers the request through the compositor rather than
        # shelling out to xdg-open, which may not have a Wayland display env.
        uri = Path(p).as_uri()  # file:///absolute/path
        result = subprocess.run(
            [
                "gdbus", "call", "--session",
                "--dest",        "org.freedesktop.portal.Desktop",
                "--object-path", "/org/freedesktop/portal/desktop",
                "--method",      "org.freedesktop.portal.OpenURI.OpenURI",
                "",              # parent window handle (empty = no parent)
                uri,
                "{}",            # options dict a{sv}
            ],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            # gdbus unavailable (non-GNOME compositor) — fall back to gio
            subprocess.run(["gio", "open", p], check=False)


def mono_font(size: int = 10) -> QFont:
    """JetBrains Mono with a safe fallback chain."""
    for name in ("JetBrains Mono", "Consolas", "Courier New", "Monospace"):
        f = QFont(name, size)
        if QFontDatabase.hasFamily(name):
            f.setFixedPitch(True)
            f.setStyleHint(QFont.StyleHint.Monospace)
            return f
    f = QFont()
    f.setFixedPitch(True)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(size)
    return f


def _pill(text: str, color: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(
        f"background: transparent; color: {color}; border: 1px solid {color};"
        f" border-radius: 10px; padding: 1px 8px; font-size: 11px; font-weight:600;"
    )
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        f"color: {PAL['fg_quart']}; font-size: 10px; font-weight:700;"
        f" letter-spacing: 1px; padding: 8px 0 4px 0;"
    )
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background: {PAL['border']}; max-height:1px;")
    return f


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT STATUS PANEL  (shared between Pipeline and Results tabs)
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(240)
        self.setStyleSheet(f"background:{PAL['bg_panel']}; border-left:1px solid {PAL['border']};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        layout.addWidget(_section_label("Checkpoints"))

        self._rows: list[tuple[str, QLabel, QLabel]] = []
        stages = [
            ("Stage 1", "stage1_best.pth"),
            ("Stage 2A", "stage2a_best.pth"),
        ]
        for label, fname in stages:
            row = QHBoxLayout()
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(f"color:{PAL['fg_secondary']}; font-size:12px;")
            status_lbl = QLabel("—")
            status_lbl.setFont(mono_font(10))
            status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(name_lbl)
            row.addStretch()
            row.addWidget(status_lbl)
            layout.addLayout(row)
            self._rows.append((fname, name_lbl, status_lbl))

        # Stage 2B YOLO (finds best.pt under checkpoints/)
        row = QHBoxLayout()
        n2 = QLabel("Stage 2B")
        n2.setStyleSheet(f"color:{PAL['fg_secondary']}; font-size:12px;")
        self._s2b_status = QLabel("—")
        self._s2b_status.setFont(mono_font(10))
        self._s2b_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(n2)
        row.addStretch()
        row.addWidget(self._s2b_status)
        layout.addLayout(row)

        layout.addWidget(_divider())
        layout.addWidget(_section_label("Data Dirs"))

        dirs_info = [
            ("Patches", ROOT / "dataset" / "patches"),
            ("Crops",   ROOT / "dataset" / "building_crops"),
            ("YOLO",    ROOT / "dataset" / "yolo_infra"),
            ("Outputs", ROOT / "outputs"),
        ]
        for name, path in dirs_info:
            row2 = QHBoxLayout()
            n = QLabel(name)
            n.setStyleSheet(f"color:{PAL['fg_secondary']}; font-size:12px;")
            btn = QPushButton("open")
            btn.setFixedHeight(20)
            btn.setStyleSheet(
                f"font-size:10px; padding:1px 6px; border:1px solid {PAL['border']};"
                f" background:{PAL['bg_surface']}; color:{PAL['fg_tertiary']};"
                f" border-radius:3px;"
            )
            _path = path  # capture
            btn.clicked.connect(lambda _, p=_path: open_folder(p) if p.exists() else None)
            row2.addWidget(n)
            row2.addStretch()
            row2.addWidget(btn)
            layout.addLayout(row2)

        layout.addStretch()

        # Auto-refresh every 5 s
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(5000)
        self.refresh()

    def refresh(self):
        ckpt_dir = ROOT / "checkpoints"
        for fname, _, status_lbl in self._rows:
            p = ckpt_dir / fname
            if p.exists():
                mb = p.stat().st_size / 1_048_576
                status_lbl.setText(f"{mb:.0f} MB")
                status_lbl.setStyleSheet(f"color:{PAL['ok']}; font-size:11px;")
            else:
                status_lbl.setText("missing")
                status_lbl.setStyleSheet(f"color:{PAL['fg_quart']}; font-size:11px;")

        # YOLO: look for any best.pt under checkpoints/
        best_pts = list((ckpt_dir).glob("**/best.pt")) if ckpt_dir.exists() else []
        if best_pts:
            mb = best_pts[0].stat().st_size / 1_048_576
            self._s2b_status.setText(f"{mb:.0f} MB")
            self._s2b_status.setStyleSheet(f"color:{PAL['ok']}; font-size:11px;")
        else:
            self._s2b_status.setText("missing")
            self._s2b_status.setStyleSheet(f"color:{PAL['fg_quart']}; font-size:11px;")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

_EPOCH_PATTERNS = [
    re.compile(r"Ep\s+(\d+)/(\d+)"),           # Stage 1:  Ep 012/150
    re.compile(r"[Ee]poch[: ]+(\d+)/(\d+)"),    # YOLO:     Epoch 1/200
    re.compile(r"\b(\d+)/(\d+)\b"),             # generic   42/100
]


class PipelineTab(QWidget):
    def __init__(self, ckpt_panel: CheckpointPanel, status_bar: QStatusBar):
        super().__init__()
        self._ckpt_panel = ckpt_panel
        self._status_bar = status_bar

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── main content ─────────────────────────────────────────────────────
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # ── Mode row ─────────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        lbl = QLabel("Mode")
        lbl.setFixedWidth(72)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "preprocess", "train_stage1", "train_stage2",
            "train_all", "evaluate", "infer",
        ])
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        mode_row.addWidget(lbl)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ── Data root (shown for preprocess / train / evaluate) ───────────
        self._data_root_widget = QWidget()
        dr = QHBoxLayout(self._data_root_widget)
        dr.setContentsMargins(0, 0, 0, 0)
        lbl2 = QLabel("Data Root")
        lbl2.setFixedWidth(72)
        self.data_label = QLabel(str(ROOT / "dataset"))
        self.data_label.setObjectName("path_label")
        self.data_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        dr_btn = QPushButton("Browse")
        dr_btn.setFixedWidth(70)
        dr_btn.clicked.connect(self._pick_data_root)
        dr.addWidget(lbl2)
        dr.addWidget(self.data_label, stretch=1)
        dr.addWidget(dr_btn)
        layout.addWidget(self._data_root_widget)

        # ── Infer-specific rows (TIF + output dir) ────────────────────────
        self._infer_widget = QWidget()
        iv = QVBoxLayout(self._infer_widget)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(6)

        tif_row = QHBoxLayout()
        lbl3 = QLabel("Input TIF")
        lbl3.setFixedWidth(72)
        self.tif_label = QLabel("(none selected)")
        self.tif_label.setObjectName("path_label")
        self.tif_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        tif_btn = QPushButton("Browse")
        tif_btn.setFixedWidth(70)
        tif_btn.clicked.connect(self._pick_tif)
        tif_row.addWidget(lbl3)
        tif_row.addWidget(self.tif_label, stretch=1)
        tif_row.addWidget(tif_btn)
        iv.addLayout(tif_row)

        out_row = QHBoxLayout()
        lbl4 = QLabel("Output Dir")
        lbl4.setFixedWidth(72)
        self.out_label = QLabel(str(ROOT / "outputs" / "infer"))
        self.out_label.setObjectName("path_label")
        self.out_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        out_btn = QPushButton("Browse")
        out_btn.setFixedWidth(70)
        out_btn.clicked.connect(self._pick_out_dir)
        out_row.addWidget(lbl4)
        out_row.addWidget(self.out_label, stretch=1)
        out_row.addWidget(out_btn)
        iv.addLayout(out_row)

        layout.addWidget(self._infer_widget)
        self._infer_widget.hide()

        # ── Run / Stop ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.setFixedHeight(34)
        self.run_btn.setFixedWidth(100)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setFixedHeight(34)
        self.stop_btn.setFixedWidth(80)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self._status_pill = _pill("idle", PAL["fg_quart"])
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addSpacing(12)
        btn_row.addWidget(self._status_pill)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Progress ──────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(6)
        layout.addWidget(self.progress)

        # ── Log ───────────────────────────────────────────────────────────
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(mono_font(10))
        layout.addWidget(self.log_view, stretch=1)

        outer.addWidget(content, stretch=1)
        outer.addWidget(ckpt_panel)

        # QProcess
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)

    # ── Mode switching ────────────────────────────────────────────────────────
    def _on_mode_change(self, mode: str):
        is_infer = mode == "infer"
        self._data_root_widget.setVisible(not is_infer)
        self._infer_widget.setVisible(is_infer)

    # ── Pickers ───────────────────────────────────────────────────────────────
    def _pick_data_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Root", str(ROOT / "dataset"))
        if d:
            self.data_label.setText(d)

    def _pick_tif(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Input TIF", str(ROOT / "dataset"),
            "GeoTIFF (*.tif *.tiff);;All files (*)",
        )
        if p:
            self.tif_label.setText(p)

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", str(ROOT / "outputs"))
        if d:
            self.out_label.setText(d)

    # ── Run / Stop ────────────────────────────────────────────────────────────
    def _run(self):
        mode = self.mode_combo.currentText()
        self.log_view.clear()
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_pill("running", PAL["running"])
        self._status_bar.showMessage(f"Running: {mode}")

        args = [str(ROOT / "run_pipeline.py"), "--mode", mode]
        if mode == "infer":
            tif = self.tif_label.text()
            if not tif or tif == "(none selected)":
                self.log_view.appendPlainText("[ERROR] Select a TIF file first.")
                self._on_finished(1, None)
                return
            args += ["--tif", tif, "--out", self.out_label.text()]
        else:
            args += ["--data_root", self.data_label.text()]

        self.process.start(_find_pipeline_python(), args)

    def _stop(self):
        self.process.kill()

    def _set_pill(self, text: str, color: str):
        self._status_pill.setText(text)
        self._status_pill.setStyleSheet(
            f"background: transparent; color: {color}; border: 1px solid {color};"
            f" border-radius: 10px; padding: 1px 8px; font-size: 11px; font-weight:600;"
        )

    # ── Output handling ───────────────────────────────────────────────────────
    def _on_output(self):
        raw = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self.log_view.appendPlainText(raw.rstrip())
        # scroll to bottom
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())
        # parse progress from every line
        for line in raw.splitlines():
            for pat in _EPOCH_PATTERNS:
                m = pat.search(line)
                if m:
                    try:
                        curr, total = int(m.group(1)), int(m.group(2))
                        if 0 < total <= 10_000:
                            self.progress.setValue(int(curr / total * 100))
                    except (ValueError, ZeroDivisionError):
                        pass
                    break

    def _on_finished(self, exit_code: int, _status):
        ok = exit_code == 0
        self.log_view.appendPlainText(
            f"\n[Process finished — exit code {exit_code}]"
        )
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if ok else 0)
        self._set_pill("done" if ok else "failed", PAL["ok"] if ok else PAL["err"])
        self._status_bar.showMessage("Finished" if ok else f"Failed (exit {exit_code})", 8000)
        self._ckpt_panel.refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Map / Image Viewer
# ─────────────────────────────────────────────────────────────────────────────

class MapViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        open_btn = QPushButton("Open TIF")
        open_btn.clicked.connect(self._open_tif)
        mask_btn = QPushButton("Load Mask")
        mask_btn.clicked.connect(self._load_mask)
        toolbar.addWidget(open_btn)
        toolbar.addWidget(mask_btn)
        toolbar.addSpacing(16)

        toolbar.addWidget(QLabel("Overlay"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFixedWidth(120)
        self.opacity_slider.valueChanged.connect(self._update_display)
        toolbar.addWidget(self.opacity_slider)
        self.opacity_lbl = QLabel("50%")
        self.opacity_lbl.setObjectName("label_mono")
        self.opacity_lbl.setFixedWidth(36)
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_lbl.setText(f"{v}%")
        )
        toolbar.addWidget(self.opacity_lbl)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # ── Side-by-side panels ───────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.left_label = self._make_image_label("Open a TIF to begin")
        self.right_label = self._make_image_label("Load a prediction mask to see overlay")

        splitter.addWidget(self._wrap_image_panel("RGB", self.left_label))
        splitter.addWidget(self._wrap_image_panel("Overlay", self.right_label))
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        # ── Legend ────────────────────────────────────────────────────────────
        legend = QHBoxLayout()
        entries = [
            (PAL["cls_bg"],       "Background"),
            (PAL["cls_building"], "Building"),
            (PAL["cls_road"],     "Road"),
            (PAL["cls_water"],    "Waterbody"),
        ]
        for color, name in entries:
            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background:{color}; border-radius:2px;")
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{PAL['fg_tertiary']}; font-size:11px;")
            legend.addWidget(dot)
            legend.addWidget(lbl)
            legend.addSpacing(16)
        legend.addStretch()
        layout.addLayout(legend)

        self._rgb: np.ndarray | None = None
        self._mask: np.ndarray | None = None

    @staticmethod
    def _make_image_label(placeholder: str) -> QLabel:
        lbl = QLabel(placeholder)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            f"background:{PAL['bg_panel']}; color:{PAL['fg_quart']};"
            f" border:1px solid {PAL['border']};"
        )
        lbl.setMinimumSize(280, 280)
        return lbl

    @staticmethod
    def _wrap_image_panel(title: str, img_label: QLabel) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        t = QLabel(title.upper())
        t.setStyleSheet(
            f"color:{PAL['fg_quart']}; font-size:10px; font-weight:700; letter-spacing:1px;"
        )
        v.addWidget(t)
        v.addWidget(img_label, stretch=1)
        return w

    def _open_tif(self):
        # Search dataset dir and patches dir
        start = str(ROOT / "dataset")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GeoTIFF", start, "GeoTIFF (*.tif *.tiff);;All files (*)"
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
                rgb = _to_uint8(rgb)
            if rgb.shape[2] < 3:
                rgb = np.stack([rgb[:, :, 0]] * 3, axis=-1)
            self._rgb = rgb
            self._update_display()
        except Exception as exc:
            self.left_label.setText(f"Error: {exc}")

    def _load_mask(self):
        start = str(ROOT / "outputs")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Segmentation Mask", start,
            "Image (*.png *.tif *.tiff);;All files (*)",
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
        except Exception as exc:
            self.right_label.setText(f"Error: {exc}")

    def _update_display(self):
        if self._rgb is None:
            return
        rgb_disp = _fit(self._rgb, 1280)
        _set_pixmap(self.left_label, rgb_disp)

        if self._mask is not None:
            mask_r = cv2.resize(
                self._mask, (rgb_disp.shape[1], rgb_disp.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            overlay = _CLASS_COLORS[np.clip(mask_r, 0, len(_CLASS_COLORS) - 1)]
            alpha = self.opacity_slider.value() / 100.0
            blended = (
                rgb_disp.astype(np.float32) * (1 - alpha)
                + overlay.astype(np.float32) * alpha
            ).clip(0, 255).astype(np.uint8)
            _set_pixmap(self.right_label, blended)
        else:
            _set_pixmap(self.right_label, rgb_disp)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Results Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_results)
        browse_btn = QPushButton("Browse Outputs")
        browse_btn.clicked.connect(lambda: open_folder(ROOT / "outputs"))
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(browse_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        layout.addWidget(_divider())

        # ── Metrics table ─────────────────────────────────────────────────────
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Stage", "mIoU", "Dice / F1", "Accuracy", "mAP@0.5"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, 5):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setFixedHeight(180)
        layout.addWidget(self.table)

        # ── Bar chart (dark) ──────────────────────────────────────────────────
        fig = Figure(figsize=(6, 3), facecolor=PAL["bg_panel"])
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet(f"background:{PAL['bg_panel']};")
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor(PAL["bg_surface"])
        layout.addWidget(self.canvas, stretch=1)

        self._load_results()

    def _load_results(self):
        results: dict = {}

        # Primary source: outputs/results.json
        results_json = ROOT / "outputs" / "results.json"
        if results_json.exists():
            try:
                with open(results_json) as f:
                    results = json.load(f)
            except Exception:
                pass

        # Supplement Stage 2B from YOLO's own results.csv if not already present
        if "stage2b" not in results or "error" in results.get("stage2b", {}):
            csvs = list((ROOT / "checkpoints").glob("**/results.csv"))
            if csvs:
                try:
                    import csv
                    with open(csvs[0], newline="") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    if rows:
                        # strip whitespace from header keys
                        header = [k.strip() for k in rows[0].keys()]
                        map50_col = next(
                            (h for h in header if "map50" in h.lower() and "95" not in h.lower()),
                            None,
                        )
                        if map50_col:
                            vals = [
                                float(row[map50_col].strip())
                                for row in rows
                                if row.get(map50_col, "").strip()
                            ]
                            if vals:
                                results["stage2b"] = {"mAP_50": max(vals)}
                except Exception:
                    pass

        self.table.setRowCount(0)
        chart_labels: list[str] = []
        chart_values: list[float] = []
        chart_colors: list[str] = []
        stage_colors = [PAL["cls_water"], PAL["cls_building"], PAL["cls_road"]]

        for i, (stage_key, data) in enumerate(results.items()):
            row = self.table.rowCount()
            self.table.insertRow(row)

            def _cell(v) -> QTableWidgetItem:
                text = f"{v:.4f}" if isinstance(v, float) else str(v)
                item = QTableWidgetItem(text)
                item.setFont(mono_font(11))
                item.setForeground(QColor(PAL["fg_primary"]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                return item

            name_item = QTableWidgetItem(stage_key)
            name_item.setForeground(QColor(PAL["fg_secondary"]))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, _cell(data.get("mIoU", "")))
            self.table.setItem(row, 2, _cell(data.get("dice", "")))
            self.table.setItem(row, 3, _cell(data.get("accuracy", "")))
            self.table.setItem(row, 4, _cell(data.get("mAP_50", "")))

            primary = data.get("mIoU") or data.get("accuracy") or data.get("mAP_50")
            if isinstance(primary, float):
                chart_labels.append(stage_key)
                chart_values.append(primary)
                chart_colors.append(stage_colors[i % len(stage_colors)])

        # ── Redraw chart ──────────────────────────────────────────────────────
        self.ax.clear()
        self.ax.set_facecolor(PAL["bg_surface"])
        if chart_labels:
            bars = self.ax.bar(chart_labels, chart_values, color=chart_colors, width=0.5)
            self.ax.set_ylim(0, 1.05)
            self.ax.set_ylabel("Score", color=PAL["fg_tertiary"], fontsize=10)
            self.ax.tick_params(colors=PAL["fg_tertiary"], labelsize=9)
            self.ax.spines[:].set_color(PAL["border"])
            for spine in self.ax.spines.values():
                spine.set_linewidth(0.5)
            self.ax.yaxis.grid(True, color=PAL["border"], linewidth=0.5, alpha=0.6)
            self.ax.set_axisbelow(True)
            for bar, val in zip(bars, chart_values):
                self.ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9, color=PAL["fg_secondary"],
                    fontfamily="monospace",
                )
        self.canvas.figure.patch.set_facecolor(PAL["bg_panel"])
        self.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────

def _fit(img: np.ndarray, max_px: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_px / max(h, w, 1), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    out = np.zeros((*arr.shape[:2], 3), dtype=np.float32)
    for i in range(min(arr.shape[2] if arr.ndim == 3 else 1, 3)):
        ch = arr[:, :, i].astype(np.float32)
        mn, mx = ch.min(), ch.max()
        out[:, :, i] = 0 if mx == mn else (ch - mn) / (mx - mn) * 255
    return out.astype(np.uint8)


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
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Geo-Intel — Pipeline Operator Console")
        self.resize(1200, 820)

        status = QStatusBar()
        status.showMessage("Ready")
        self.setStatusBar(status)

        ckpt_panel = CheckpointPanel()

        tabs = QTabWidget()
        tabs.addTab(PipelineTab(ckpt_panel, status), "Pipeline Runner")
        tabs.addTab(MapViewerTab(), "Map Viewer")
        tabs.addTab(ResultsTab(), "Results")
        # Refresh results when switching to that tab
        tabs.currentChanged.connect(
            lambda i: tabs.widget(i)._load_results()
            if hasattr(tabs.widget(i), "_load_results") else None
        )
        self.setCentralWidget(tabs)


def main():
    # High-DPI on Windows/Linux
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)

    # Force dark palette for native widgets that don't honour QSS fully
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(PAL["bg_app"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(PAL["fg_primary"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(PAL["bg_input"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(PAL["bg_surface"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(PAL["fg_primary"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(PAL["bg_surface"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(PAL["fg_primary"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(PAL["border"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(PAL["fg_primary"]))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
