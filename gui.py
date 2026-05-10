"""
gui.py — Geo-Intel Pipeline Operator Console
Run with: python gui.py

Design system: light/dark theme, Geo-Intel palette.
Cross-platform: Windows / macOS / Linux.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    Qt, QProcess, QTimer, QPropertyAnimation, QEasingCurve,
    pyqtSignal, pyqtProperty, QObject, QRectF,
)
from PyQt6.QtGui import (
    QFont, QFontDatabase, QImage, QPixmap, QColor, QPalette,
    QTextCursor, QTextCharFormat, QPainter, QPainterPath,
)
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton,
    QSizePolicy, QSlider, QSplitter, QStatusBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QFrame,
    QGridLayout, QHeaderView, QTextEdit,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ROOT = Path(__file__).parent


def _find_pipeline_python() -> str:
    """Return Python with pipeline deps; handles PyInstaller bundle mode."""
    if not getattr(sys, "frozen", False):
        return sys.executable

    candidates: list[str] = []
    for env_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env_root = os.environ.get(env_var)
        if env_root:
            for rel in ("bin/python3", "bin/python", "Scripts/python.exe"):
                p = Path(env_root) / rel
                if p.is_file():
                    candidates.append(str(p))

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

    return sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM PALETTE
# ─────────────────────────────────────────────────────────────────────────────

DARK_PAL = {
    "bg_app":      "#080C15",
    "bg_panel":    "#0D1420",
    "bg_surface":  "#131C2C",
    "bg_input":    "#192233",
    "border":      "#253347",
    "fg_primary":  "#F2F6FC",
    "fg_secondary":"#C2CDE0",
    "fg_tertiary": "#7A90A8",
    "fg_quart":    "#4D6070",
    "accent":      "#F59E0B",
    "ok":          "#34D399",
    "running":     "#38BDF8",
    "warn":        "#FBBF24",
    "err":         "#F87171",
    "info":        "#60A5FA",
    # class colours (sacred — never reassign)
    "cls_building":"#EF4444",
    "cls_road":    "#94A3B8",
    "cls_water":   "#3B82F6",
    "cls_bg":      "#253347",
}

LIGHT_PAL = {
    "bg_app":      "#F2F5FA",
    "bg_panel":    "#FFFFFF",
    "bg_surface":  "#EAF0F8",
    "bg_input":    "#DDE6F2",
    "border":      "#BCC8DC",
    "fg_primary":  "#0F1929",
    "fg_secondary":"#253348",
    "fg_tertiary": "#4E6070",
    "fg_quart":    "#7A90A8",
    "accent":      "#D97706",
    "ok":          "#059669",
    "running":     "#0284C7",
    "warn":        "#B45309",
    "err":         "#DC2626",
    "info":        "#2563EB",
    # class colours (sacred — never reassign)
    "cls_building":"#EF4444",
    "cls_road":    "#94A3B8",
    "cls_water":   "#3B82F6",
    "cls_bg":      "#253347",
}

PAL: dict[str, str] = dict(DARK_PAL)  # mutable current palette

# Class overlay colours as uint8 RGB arrays (for numpy blending)
_CLASS_COLORS = np.array([
    [0x2D, 0x3A, 0x52],   # 0 background
    [0xE6, 0x39, 0x46],   # 1 building
    [0x8B, 0x96, 0xAA],   # 2 road
    [0x4A, 0x90, 0xE2],   # 3 waterbody
], dtype=np.uint8)


def build_qss() -> str:
    """Build and return the full application QSS, reading from the global PAL dict."""
    return f"""
QWidget {{
    background-color: {PAL['bg_app']};
    color: {PAL['fg_primary']};
    font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    font-weight: 450;
}}
QMainWindow, QDialog {{
    background-color: {PAL['bg_app']};
}}
QTabWidget::pane {{
    border: 1px solid {PAL['border']};
    border-top: none;
    background-color: {PAL['bg_panel']};
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background: {PAL['bg_surface']};
    color: {PAL['fg_tertiary']};
    padding: 11px 26px;
    border: 1px solid {PAL['border']};
    border-bottom: none;
    min-width: 130px;
    font-size: 13px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {PAL['bg_panel']};
    color: {PAL['fg_primary']};
    border-top: 3px solid {PAL['accent']};
    font-weight: 650;
}}
QTabBar::tab:hover:!selected {{
    color: {PAL['fg_secondary']};
    background: {PAL['bg_input']};
}}
QPushButton {{
    background-color: {PAL['bg_surface']};
    color: {PAL['fg_primary']};
    border: 1px solid {PAL['border']};
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 550;
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
    background-color: {PAL['bg_surface']};
}}
QPushButton#run_btn {{
    background-color: {PAL['accent']};
    color: #080C15;
    font-size: 14px;
    font-weight: 700;
    border: none;
    letter-spacing: 0.4px;
}}
QPushButton#run_btn:hover {{
    background-color: #FBBF24;
}}
QPushButton#run_btn:disabled {{
    background-color: {PAL['border']};
    color: {PAL['fg_quart']};
}}
QPushButton#stop_btn {{
    background-color: transparent;
    color: {PAL['err']};
    border: 1.5px solid {PAL['err']};
    font-weight: 600;
}}
QPushButton#stop_btn:hover {{
    background-color: rgba(248,113,113,0.14);
}}
QPushButton#check_btn {{
    background-color: {PAL['info']};
    color: #080C15;
    font-size: 14px;
    font-weight: 700;
    border: none;
    letter-spacing: 0.4px;
}}
QPushButton#check_btn:hover {{
    background-color: #82BAFF;
}}
QPushButton#check_btn:disabled {{
    background-color: {PAL['border']};
    color: {PAL['fg_quart']};
}}
QPushButton#dir_btn {{
    font-size: 11px;
    font-weight: 550;
    padding: 3px 9px;
    border: 1px solid {PAL['border']};
    background: {PAL['bg_surface']};
    color: {PAL['fg_tertiary']};
    border-radius: 6px;
}}
QComboBox {{
    background-color: {PAL['bg_input']};
    color: {PAL['fg_primary']};
    border: 1px solid {PAL['border']};
    border-radius: 8px;
    padding: 7px 12px;
    min-width: 160px;
    font-size: 13px;
    font-weight: 500;
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background-color: {PAL['bg_surface']};
    color: {PAL['fg_primary']};
    selection-background-color: {PAL['border']};
    border: 1px solid {PAL['border']};
    padding: 4px;
}}
QLabel {{
    color: {PAL['fg_secondary']};
    background: transparent;
    font-size: 14px;
}}
QLabel#label_heading {{
    color: {PAL['fg_primary']};
    font-size: 17px;
    font-weight: 700;
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
    border-radius: 6px;
    padding: 6px 10px;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-size: 12px;
}}
QLabel#section_lbl {{
    color: {PAL['fg_quart']};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 10px 0 6px 0;
}}
QLabel#op_label {{
    color: {PAL['fg_secondary']};
    font-size: 13px;
    font-weight: 500;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
}}
QLabel#metric_key {{
    color: {PAL['fg_quart']};
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.8px;
}}
QLabel#img_placeholder {{
    background: {PAL['bg_panel']};
    color: {PAL['fg_quart']};
    border: 1px solid {PAL['border']};
    border-radius: 8px;
    font-size: 14px;
}}
QLabel#img_panel_title {{
    color: {PAL['fg_quart']};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#legend_lbl {{
    color: {PAL['fg_tertiary']};
    font-size: 12px;
    font-weight: 500;
}}
QFrame#info_card {{
    background: {PAL['bg_panel']};
    border: 1px solid {PAL['border']};
    border-radius: 10px;
}}
QLabel#info_key {{
    color: {PAL['fg_quart']};
    font-size: 12px;
    font-weight: 650;
    min-width: 60px;
}}
QLabel#info_val {{
    color: {PAL['fg_secondary']};
    font-size: 12px;
    font-weight: 500;
}}
QTextEdit {{
    background-color: {PAL['bg_app']};
    color: {PAL['fg_secondary']};
    border: 1px solid {PAL['border']};
    border-radius: 8px;
    font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
    font-size: 12px;
    selection-background-color: {PAL['border']};
}}
QProgressBar {{
    background-color: {PAL['bg_input']};
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {PAL['accent']};
    border-radius: 5px;
}}
QSlider::groove:horizontal {{
    background: {PAL['bg_input']};
    height: 5px;
    border-radius: 3px;
    border: 1px solid {PAL['border']};
}}
QSlider::handle:horizontal {{
    background: {PAL['accent']};
    width: 17px;
    height: 17px;
    margin: -6px 0;
    border-radius: 9px;
}}
QSlider::sub-page:horizontal {{
    background: {PAL['accent']};
    border-radius: 3px;
}}
QTableWidget {{
    background-color: {PAL['bg_panel']};
    alternate-background-color: {PAL['bg_surface']};
    color: {PAL['fg_secondary']};
    border: 1px solid {PAL['border']};
    border-radius: 8px;
    gridline-color: {PAL['border']};
    font-size: 13px;
    font-weight: 450;
}}
QTableWidget::item {{
    padding: 7px 10px;
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
    padding: 9px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
}}
QScrollBar:vertical {{
    background: {PAL['bg_app']};
    width: 10px;
    border: none;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {PAL['border']};
    border-radius: 5px;
    min-height: 28px;
    margin: 2px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {PAL['bg_app']};
    height: 10px;
    border: none;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {PAL['border']};
    border-radius: 5px;
    margin: 2px;
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
    font-size: 12px;
    font-weight: 500;
}}
QWidget#checkpoint_panel {{
    background: {PAL['bg_panel']};
    border-left: 1px solid {PAL['border']};
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# THEME MANAGER
# ─────────────────────────────────────────────────────────────────────────────

_THEME_FILE = ROOT / ".geo_intel_theme"


def _apply_app_palette(app: QApplication):
    """Set the Qt QPalette on the app, reading from the current global PAL."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(PAL["bg_app"]))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(PAL["fg_primary"]))
    palette.setColor(QPalette.ColorRole.Base,            QColor(PAL["bg_input"]))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(PAL["bg_surface"]))
    palette.setColor(QPalette.ColorRole.Text,            QColor(PAL["fg_primary"]))
    palette.setColor(QPalette.ColorRole.Button,          QColor(PAL["bg_surface"]))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(PAL["fg_primary"]))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(PAL["border"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(PAL["fg_primary"]))
    app.setPalette(palette)


class ThemeManager(QObject):
    changed = pyqtSignal()
    _inst: "ThemeManager | None" = None

    def __init__(self):
        super().__init__()
        self.is_dark = True

    @classmethod
    def get(cls) -> "ThemeManager":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def set_dark(self, dark: bool):
        if dark == self.is_dark:
            return
        self.is_dark = dark
        PAL.update(DARK_PAL if dark else LIGHT_PAL)
        app = QApplication.instance()
        if app:
            app.setStyleSheet(build_qss())
            _apply_app_palette(app)
        try:
            _THEME_FILE.write_text("dark" if dark else "light")
        except OSError:
            pass
        self.changed.emit()


# ─────────────────────────────────────────────────────────────────────────────
# THEME TOGGLE WIDGET
# ─────────────────────────────────────────────────────────────────────────────

class ThemeToggle(QWidget):
    """Animated pill toggle — left=dark, right=light."""
    W, H, D, PAD = 52, 28, 20, 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.W, self.H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Toggle dark / light theme")
        self._pos = 0.0  # 0.0=dark/left, 1.0=light/right

        self._anim = QPropertyAnimation(self, b"pos_val", self)
        self._anim.setDuration(240)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        ThemeManager.get().changed.connect(self._sync)

    def get_pos(self) -> float:
        return self._pos

    def set_pos(self, v: float):
        self._pos = max(0.0, min(1.0, v))
        self.update()

    pos_val = pyqtProperty(float, get_pos, set_pos)

    def _sync(self):
        target = 0.0 if ThemeManager.get().is_dark else 1.0
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(target)
        self._anim.start()

    def mousePressEvent(self, e):
        ThemeManager.get().set_dark(not ThemeManager.get().is_dark)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h, pos = self.W, self.H, self._pos

        # Interpolate track color: amber (dark) ↔ steel blue (light)
        ca = QColor("#FFB547")
        cb = QColor("#60A5FA")
        tr = int(ca.red()   + (cb.red()   - ca.red())   * pos)
        tg = int(ca.green() + (cb.green() - ca.green()) * pos)
        tb = int(ca.blue()  + (cb.blue()  - ca.blue())  * pos)
        track_color = QColor(tr, tg, tb)

        # Track pill
        track_h = h * 0.75
        track_y = (h - track_h) / 2
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, track_y, w, track_h), track_h / 2, track_h / 2)
        p.fillPath(path, track_color)

        # White thumb circle
        travel = w - self.PAD * 2 - self.D
        tx = self.PAD + pos * travel
        ty = (h - self.D) / 2
        # Subtle shadow
        p.setBrush(QColor(0, 0, 0, 30))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(tx + 0.5, ty + 1.0, self.D, self.D))
        # Thumb
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QRectF(tx, ty, self.D, self.D))

        # Icon on thumb: moon (dark) or sun (light)
        icon = "☀" if pos >= 0.5 else "\U0001F319"
        icon_color = QColor(30, 80, 160) if pos >= 0.5 else QColor(90, 60, 0)
        p.setPen(icon_color)
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        p.drawText(QRectF(tx, ty, self.D, self.D), Qt.AlignmentFlag.AlignCenter, icon)


# ─────────────────────────────────────────────────────────────────────────────
# REQUIREMENTS CHECK SCRIPT  (runs in a subprocess, prints JSON)
# ─────────────────────────────────────────────────────────────────────────────

_REQUIREMENTS_CHECK_SCRIPT = r"""
import json, sys, importlib, os

PACKAGES = [
    ("torch",                       "PyTorch",                       True),
    ("torchvision",                 "torchvision",                   True),
    ("cv2",                         "OpenCV",                        True),
    ("numpy",                       "NumPy",                         True),
    ("matplotlib",                  "matplotlib",                    True),
    ("PyQt6",                       "PyQt6",                         True),
    ("albumentations",              "albumentations",                True),
    ("segmentation_models_pytorch", "segmentation-models-pytorch",   True),
    ("timm",                        "timm",                          True),
    ("ultralytics",                 "ultralytics",                   True),
    ("rasterio",                    "rasterio",                      True),
    ("fiona",                       "fiona",                         False),
    ("geopandas",                   "geopandas",                     False),
    ("shapely",                     "shapely",                       True),
    ("sahi",                        "sahi",                          True),
    ("pandas",                      "pandas",                        True),
    ("scipy",                       "scipy",                         True),
    ("sklearn",                     "scikit-learn",                  True),
    ("skimage",                     "scikit-image",                  True),
    ("psutil",                      "psutil",                        True),
    ("tensorboard",                 "tensorboard",                   False),
    ("onnx",                        "onnx",                          False),
    ("pydensecrf",                  "pydensecrf2",                   False),
]

results = {"packages": {}}
for mod, name, required in PACKAGES:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "installed")
        if mod == "PyQt6":
            from PyQt6.QtCore import PYQT_VERSION_STR
            ver = PYQT_VERSION_STR
        results["packages"][name] = {"ok": True, "version": str(ver), "required": required}
    except ImportError as e:
        results["packages"][name] = {"ok": False, "error": str(e), "required": required}

try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        results["gpu"] = {
            "backend": "CUDA", "name": p.name,
            "vram_gb": round(p.total_memory / 1024**3, 1),
            "cuda_version": torch.version.cuda or "?",
            "device_count": torch.cuda.device_count(),
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        results["gpu"] = {"backend": "MPS", "name": "Apple Silicon (Metal)"}
    else:
        results["gpu"] = {"backend": "CPU", "name": "No GPU detected"}
except Exception as e:
    results["gpu"] = {"backend": "error", "error": str(e)}

try:
    import shutil as _sh
    root_path = os.path.abspath(os.sep)
    total, used, free = _sh.disk_usage(root_path)
    results["disk"] = {
        "total_gb": round(total / 1024**3, 1),
        "free_gb": round(free / 1024**3, 1),
        "path": root_path,
    }
except Exception as e:
    results["disk"] = {"error": str(e)}

results["python"] = {
    "version": sys.version.split()[0],
    "executable": sys.executable,
}
print(json.dumps(results))
"""


# ─────────────────────────────────────────────────────────────────────────────
# LOG / METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_PATTERNS: dict[str, re.Pattern] = {
    "mIoU": re.compile(r"m[Ii]o[Uu][:\s=]+([0-9.]+)"),
    "Loss": re.compile(r"\b[Ll]oss[:\s=]+([0-9.]+)"),
    "Acc":  re.compile(r"[Aa]cc(?:uracy)?[:\s=]+([0-9.]+)"),
    "LR":   re.compile(r"\blr[:\s=]+([0-9.e+\-]+)", re.I),
    "mAP":  re.compile(r"mAP(?:[@\d.]*)?[:\s=]+([0-9.]+)"),
}

_EPOCH_PATTERNS = [
    re.compile(r"Ep\s+(\d+)/(\d+)"),
    re.compile(r"[Ee]poch[: ]+(\d+)/(\d+)"),
]


def _line_color(line: str) -> str:
    if re.search(r"\b(?:ERROR|CRITICAL|FAILED?|Traceback|Exception)\b", line, re.I):
        return PAL["err"]
    if re.search(r"\b(?:WARN(?:ING)?)\b", line, re.I):
        return PAL["warn"]
    if re.search(r"\b(?:done|success(?:fully)?|complet(?:ed?|ing)|finish(?:ed)?)\b", line, re.I):
        return PAL["ok"]
    if re.search(r"m[Ii]o[Uu][:\s=]|\b[Ll]oss[:\s=]|mAP[:\s=]", line):
        return PAL["accent"]
    if re.search(r"(?:^|\s)(?:Stage|Epoch|Training|Evaluating|Preprocessing|Processing)\b", line, re.I):
        return PAL["running"]
    return PAL["fg_secondary"]


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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
        # XDG Desktop Portal via gdbus — proper Wayland-native IPC.
        uri = Path(p).as_uri()
        result = subprocess.run(
            [
                "gdbus", "call", "--session",
                "--dest",        "org.freedesktop.portal.Desktop",
                "--object-path", "/org/freedesktop/portal/desktop",
                "--method",      "org.freedesktop.portal.OpenURI.OpenURI",
                "", uri, "{}",
            ],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            subprocess.run(["gio", "open", p], check=False)


def mono_font(size: int = 10) -> QFont:
    try:
        _families = set(QFontDatabase.families())
    except Exception:
        _families = set()
    for name in ("JetBrains Mono", "Consolas", "Courier New", "Monospace"):
        if name in _families:
            f = QFont(name, size)
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
        f"background: transparent; color: {color}; border: 1.5px solid {color};"
        f" border-radius: 10px; padding: 2px 10px; font-size: 12px; font-weight:650;"
    )
    lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return lbl


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("section_lbl")
    return lbl


def _divider() -> QFrame:
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.Shape.HLine)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT STATUS PANEL
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("checkpoint_panel")
        self.setFixedWidth(264)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        layout.addWidget(_section_label("Checkpoints"))

        self._rows: list[tuple[str, QLabel, QLabel]] = []
        stages = [
            ("Stage 1", "stage1_best.pth"),
            ("Stage 2A", "stage2a_best.pth"),
        ]
        for label, fname in stages:
            row = QHBoxLayout()
            name_lbl = QLabel(label)
            name_lbl.setStyleSheet(f"color:{PAL['fg_secondary']}; font-size:13px; font-weight:500;")
            status_lbl = QLabel("—")
            status_lbl.setFont(mono_font(11))
            status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(name_lbl)
            row.addStretch()
            row.addWidget(status_lbl)
            layout.addLayout(row)
            self._rows.append((fname, name_lbl, status_lbl))

        row = QHBoxLayout()
        n2 = QLabel("Stage 2B")
        n2.setStyleSheet(f"color:{PAL['fg_secondary']}; font-size:13px; font-weight:500;")
        self._s2b_status = QLabel("—")
        self._s2b_status.setFont(mono_font(11))
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
            n.setStyleSheet(f"color:{PAL['fg_secondary']}; font-size:13px; font-weight:500;")
            btn = QPushButton("open")
            btn.setObjectName("dir_btn")
            btn.setFixedHeight(20)
            _path = path
            btn.clicked.connect(lambda _, p=_path: open_folder(p) if p.exists() else None)
            row2.addWidget(n)
            row2.addStretch()
            row2.addWidget(btn)
            layout.addLayout(row2)

        layout.addStretch()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(5000)
        self.refresh()

        ThemeManager.get().changed.connect(self._apply_theme)

    def _apply_theme(self):
        self.refresh()

    def refresh(self):
        ckpt_dir = ROOT / "checkpoints"
        for fname, _, status_lbl in self._rows:
            p = ckpt_dir / fname
            if p.exists():
                mb = p.stat().st_size / 1_048_576
                status_lbl.setText(f"{mb:.0f} MB")
                status_lbl.setStyleSheet(f"color:{PAL['ok']}; font-size:12px; font-weight:600;")
            else:
                status_lbl.setText("missing")
                status_lbl.setStyleSheet(f"color:{PAL['fg_quart']}; font-size:12px; font-weight:500;")

        best_pts = list((ckpt_dir).glob("**/best.pt")) if ckpt_dir.exists() else []
        if best_pts:
            mb = best_pts[0].stat().st_size / 1_048_576
            self._s2b_status.setText(f"{mb:.0f} MB")
            self._s2b_status.setStyleSheet(f"color:{PAL['ok']}; font-size:12px; font-weight:600;")
        else:
            self._s2b_status.setText("missing")
            self._s2b_status.setStyleSheet(f"color:{PAL['fg_quart']}; font-size:12px; font-weight:500;")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE STRIP — visual indicator of pipeline stage progress
# ─────────────────────────────────────────────────────────────────────────────

class StageStrip(QWidget):
    _STAGES = [
        ("Stage 1", "Segmentation"),
        ("Stage 2A", "Classifier"),
        ("Stage 2B", "Detector"),
    ]

    def __init__(self):
        super().__init__()
        self.setFixedHeight(46)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._labels: list[QLabel] = []
        self._states: list[str] = []
        for num, name in self._STAGES:
            lbl = QLabel(f"  {num}  ·  {name}  ")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            lbl.setFixedHeight(38)
            self._labels.append(lbl)
            self._states.append("pending")
            layout.addWidget(lbl)

        self.reset()
        ThemeManager.get().changed.connect(self._apply_theme)

    def reset(self):
        self._states = ["pending"] * len(self._labels)
        for lbl in self._labels:
            self._apply(lbl, "pending")

    def set_active(self, idx: int):
        self._states = [
            "done" if i < idx else ("active" if i == idx else "pending")
            for i in range(len(self._labels))
        ]
        for i, lbl in enumerate(self._labels):
            self._apply(lbl, self._states[i])

    def all_done(self):
        self._states = ["done"] * len(self._labels)
        for lbl in self._labels:
            self._apply(lbl, "done")

    def _apply_theme(self):
        for lbl, state in zip(self._labels, self._states):
            self._apply(lbl, state)

    @staticmethod
    def _apply(lbl: QLabel, state: str):
        if state == "active":
            bg, fg, border = PAL["accent"], "#080C15", PAL["accent"]
            weight = "700"
            size = "13px"
        elif state == "done":
            bg, fg, border = f"rgba(52,211,153,0.12)", PAL["ok"], PAL["ok"]
            weight = "650"
            size = "12px"
        else:
            bg, fg, border = PAL["bg_surface"], PAL["fg_quart"], PAL["border"]
            weight = "500"
            size = "12px"
        lbl.setStyleSheet(
            f"background:{bg}; color:{fg}; border:1.5px solid {border};"
            f" border-radius:8px; font-size:{size}; font-weight:{weight}; letter-spacing:0.3px;"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

class PipelineTab(QWidget):
    def __init__(self, ckpt_panel: CheckpointPanel, status_bar: QStatusBar):
        super().__init__()
        self._ckpt_panel = ckpt_panel
        self._status_bar = status_bar
        self._start_time: float | None = None
        self._current_epoch = 0
        self._total_epochs = 0
        self._log_has_content = False

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 18, 20, 18)
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

        # ── Stage strip ───────────────────────────────────────────────────────
        self._stage_strip = StageStrip()
        layout.addWidget(self._stage_strip)

        # ── Data root ─────────────────────────────────────────────────────────
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

        # ── Infer-specific rows ───────────────────────────────────────────────
        self._infer_widget = QWidget()
        iv = QVBoxLayout(self._infer_widget)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(6)

        tif_row = QHBoxLayout()
        lbl3 = QLabel("Input Raster")
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

        # ── Operation description ─────────────────────────────────────────────
        self._op_label = QLabel("Select a mode and press Run")
        self._op_label.setObjectName("op_label")
        layout.addWidget(self._op_label)

        # ── Run / Stop row ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.setFixedHeight(42)
        self.run_btn.setFixedWidth(120)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setFixedHeight(42)
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self._status_pill = _pill("idle", PAL["fg_quart"])
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addSpacing(12)
        btn_row.addWidget(self._status_pill)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Live metrics row ──────────────────────────────────────────────────
        metrics_row = QHBoxLayout()
        self._metric_labels: dict[str, QLabel] = {}
        self._metric_key_labels: list[QLabel] = []
        for key in ("mIoU", "Loss", "Acc", "LR", "mAP"):
            k_lbl = QLabel(key)
            k_lbl.setObjectName("metric_key")
            v_lbl = QLabel("—")
            v_lbl.setFont(mono_font(12))
            v_lbl.setStyleSheet(f"color:{PAL['accent']}; min-width:68px; font-size:15px; font-weight:700;")
            metrics_row.addWidget(k_lbl)
            metrics_row.addWidget(v_lbl)
            metrics_row.addSpacing(12)
            self._metric_labels[key] = v_lbl
            self._metric_key_labels.append(k_lbl)
        metrics_row.addStretch()
        layout.addLayout(metrics_row)

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(10)
        layout.addWidget(self.progress)

        # ── Elapsed / ETA row ─────────────────────────────────────────────────
        time_row = QHBoxLayout()
        self._elapsed_lbl = QLabel("Elapsed: —")
        self._elapsed_lbl.setFont(mono_font(11))
        self._elapsed_lbl.setStyleSheet(f"color:{PAL['fg_tertiary']}; font-size:12px; font-weight:500;")
        self._eta_lbl = QLabel("ETA: —")
        self._eta_lbl.setFont(mono_font(11))
        self._eta_lbl.setStyleSheet(f"color:{PAL['fg_tertiary']}; font-size:12px; font-weight:500;")
        time_row.addWidget(self._elapsed_lbl)
        time_row.addSpacing(24)
        time_row.addWidget(self._eta_lbl)
        time_row.addStretch()
        layout.addLayout(time_row)

        # ── Log view (colored QTextEdit) ──────────────────────────────────────
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(mono_font(11))
        layout.addWidget(self.log_view, stretch=1)

        outer.addWidget(content, stretch=1)
        outer.addWidget(ckpt_panel)

        # QProcess
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        # Elapsed timer (1 s)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        self._on_mode_change(self.mode_combo.currentText())

        ThemeManager.get().changed.connect(self._apply_theme)

    def _apply_theme(self):
        self._elapsed_lbl.setStyleSheet(f"color:{PAL['fg_tertiary']}; font-size:12px; font-weight:500;")
        self._eta_lbl.setStyleSheet(f"color:{PAL['fg_tertiary']}; font-size:12px; font-weight:500;")
        for v_lbl in self._metric_labels.values():
            v_lbl.setStyleSheet(f"color:{PAL['accent']}; min-width:68px; font-size:15px; font-weight:700;")

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_mode_change(self, mode: str):
        is_infer = mode == "infer"
        self._data_root_widget.setVisible(not is_infer)
        self._infer_widget.setVisible(is_infer)

        train_modes = {"train_stage1", "train_stage2", "train_all"}
        self._stage_strip.setVisible(mode in train_modes)

        if mode == "train_stage1":
            self._stage_strip.set_active(0)
        elif mode == "train_stage2":
            self._stage_strip.set_active(1)
        elif mode == "train_all":
            self._stage_strip.reset()

        self._op_label.setText(f"Ready  —  {mode.replace('_', ' ')}")

    # ── Pickers ───────────────────────────────────────────────────────────────

    def _pick_data_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data Root", str(ROOT / "dataset"))
        if d:
            self.data_label.setText(d)

    def _pick_tif(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Input Raster", str(ROOT / "dataset"),
            "Raster files (*.tif *.tiff *.ecw);;GeoTIFF (*.tif *.tiff);;ECW (*.ecw);;All files (*)",
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
        self._log_has_content = False
        self.progress.setValue(0)
        self._reset_metrics()
        self._current_epoch = 0
        self._total_epochs = 0
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_pill("running", PAL["running"])
        self._status_bar.showMessage(f"Running: {mode}")
        self._op_label.setText(f"Starting  {mode.replace('_', ' ')}…")
        self._start_time = time.monotonic()
        self._elapsed_timer.start()

        if mode == "train_stage1" or mode == "train_all":
            self._stage_strip.set_active(0)
        elif mode == "train_stage2":
            self._stage_strip.set_active(1)

        args = [str(ROOT / "run_pipeline.py"), "--mode", mode]
        if mode == "infer":
            tif = self.tif_label.text()
            if not tif or tif == "(none selected)":
                self._append_log("[ERROR] Select a TIF file first.", PAL["err"])
                self._on_finished(1, None)
                return
            args += ["--tif", tif, "--out", self.out_label.text()]
        else:
            args += ["--data_root", self.data_label.text()]

        self.process.start(_find_pipeline_python(), args)

    def _stop(self):
        self.process.kill()

    # ── Status pill ───────────────────────────────────────────────────────────

    def _set_pill(self, text: str, color: str):
        self._status_pill.setText(text)
        self._status_pill.setStyleSheet(
            f"background: transparent; color: {color}; border: 1px solid {color};"
            f" border-radius: 10px; padding: 1px 8px; font-size: 11px; font-weight:600;"
        )

    # ── Elapsed timer ─────────────────────────────────────────────────────────

    def _update_elapsed(self):
        if self._start_time is None:
            return
        elapsed = time.monotonic() - self._start_time
        self._elapsed_lbl.setText(f"Elapsed: {_fmt_time(elapsed)}")
        if self._current_epoch > 0 and self._total_epochs > 0:
            rate = elapsed / self._current_epoch
            remaining = rate * (self._total_epochs - self._current_epoch)
            self._eta_lbl.setText(f"ETA: {_fmt_time(remaining)}")

    def _reset_metrics(self):
        for v in self._metric_labels.values():
            v.setText("—")
        self._elapsed_lbl.setText("Elapsed: —")
        self._eta_lbl.setText("ETA: —")

    # ── Colored log append ────────────────────────────────────────────────────

    def _append_log(self, line: str, color: str | None = None):
        color = color or _line_color(line)
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        if self._log_has_content:
            cursor.insertText("\n" + line)
        else:
            cursor.insertText(line)
            self._log_has_content = True
        self.log_view.setTextCursor(cursor)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    # ── Live metric extraction ────────────────────────────────────────────────

    def _update_metrics(self, line: str):
        for key, pat in _METRIC_PATTERNS.items():
            m = pat.search(line)
            if m:
                self._metric_labels[key].setText(m.group(1))

    # ── Output handling ───────────────────────────────────────────────────────

    def _on_output(self):
        raw = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        mode = self.mode_combo.currentText()

        for line in raw.splitlines():
            if not line.strip():
                continue
            self._append_log(line)
            self._update_metrics(line)

            # Advance stage strip based on log content
            if re.search(r"\b(?:stage.?2a|classifier|convnext|arcface)\b", line, re.I):
                self._stage_strip.set_active(1)
            elif re.search(r"\b(?:stage.?2b|yolo|detector|sahi)\b", line, re.I):
                self._stage_strip.set_active(2)

            # Epoch progress + operation label
            for pat in _EPOCH_PATTERNS:
                m = pat.search(line)
                if m:
                    try:
                        curr, total = int(m.group(1)), int(m.group(2))
                        if 0 < total <= 10_000:
                            self._current_epoch = curr
                            self._total_epochs = total
                            self.progress.setValue(int(curr / total * 100))
                            label = mode.replace("_", " ").title()
                            self._op_label.setText(f"{label}  —  Epoch {curr} / {total}")
                    except (ValueError, ZeroDivisionError):
                        pass
                    break

    def _on_finished(self, exit_code: int, _status):
        ok = exit_code == 0
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        self._elapsed_timer.stop()

        suffix = f"  {_fmt_time(elapsed)}" if elapsed else ""
        self._append_log(
            f"\n[Process finished — exit code {exit_code}{suffix}]",
            PAL["ok"] if ok else PAL["err"],
        )

        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100 if ok else 0)
        self._set_pill("done" if ok else "failed", PAL["ok"] if ok else PAL["err"])
        self._status_bar.showMessage("Finished" if ok else f"Failed (exit {exit_code})", 8000)

        if ok:
            self._stage_strip.all_done()
            self._op_label.setText("Completed successfully")
        else:
            self._op_label.setText(f"Failed with exit code {exit_code}")

        if elapsed:
            self._elapsed_lbl.setText(f"Elapsed: {_fmt_time(elapsed)}")

        self._ckpt_panel.refresh()

    def _on_process_error(self, error):
        self._elapsed_timer.stop()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_pill("error", PAL["err"])
        self._append_log(f"\n[Process error: {self.process.errorString()}]", PAL["err"])
        self._op_label.setText("Failed to start process")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Map / Image Viewer
# ─────────────────────────────────────────────────────────────────────────────

class MapViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_label = self._make_image_label("Open a TIF to begin")
        self.right_label = self._make_image_label("Load a prediction mask to see overlay")
        splitter.addWidget(self._wrap_image_panel("RGB", self.left_label))
        splitter.addWidget(self._wrap_image_panel("Overlay", self.right_label))
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, stretch=1)

        self._legend_labels: list[QLabel] = []
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
            lbl.setObjectName("legend_lbl")
            self._legend_labels.append(lbl)
            legend.addWidget(dot)
            legend.addWidget(lbl)
            legend.addSpacing(16)
        legend.addStretch()
        layout.addLayout(legend)

        self._rgb: np.ndarray | None = None
        self._mask: np.ndarray | None = None

        ThemeManager.get().changed.connect(self._apply_theme)

    def _apply_theme(self):
        self._update_display()

    @staticmethod
    def _make_image_label(placeholder: str) -> QLabel:
        lbl = QLabel(placeholder)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setObjectName("img_placeholder")
        lbl.setMinimumSize(280, 280)
        return lbl

    @staticmethod
    def _wrap_image_panel(title: str, img_label: QLabel) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        t = QLabel(title.upper())
        t.setObjectName("img_panel_title")
        v.addWidget(t)
        v.addWidget(img_label, stretch=1)
        return w

    def _open_tif(self):
        start = str(ROOT / "dataset")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Raster", start, "Raster files (*.tif *.tiff *.ecw);;GeoTIFF (*.tif *.tiff);;ECW (*.ecw);;All files (*)"
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

        fig = Figure(figsize=(6, 3), facecolor=PAL["bg_panel"])
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet(f"background:{PAL['bg_panel']};")
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor(PAL["bg_surface"])
        layout.addWidget(self.canvas, stretch=1)

        self._load_results()

        ThemeManager.get().changed.connect(self._apply_theme)

    def _apply_theme(self):
        self.canvas.setStyleSheet(f"background:{PAL['bg_panel']};")
        self._load_results()

    def _load_results(self):
        results: dict = {}

        results_json = ROOT / "outputs" / "results.json"
        if results_json.exists():
            try:
                with open(results_json) as f:
                    results = json.load(f)
            except Exception:
                pass

        if "stage2b" not in results or "error" in results.get("stage2b", {}):
            csvs = list((ROOT / "checkpoints").glob("**/results.csv"))
            if csvs:
                try:
                    import csv
                    with open(csvs[0], newline="") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    if rows:
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
# Tab 4: Requirements Checker
# ─────────────────────────────────────────────────────────────────────────────

class RequirementsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._last_data: dict | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Environment & Requirements")
        title.setObjectName("label_heading")
        self._check_btn = QPushButton("Check All")
        self._check_btn.setObjectName("check_btn")
        self._check_btn.setFixedWidth(100)
        self._check_btn.setFixedHeight(32)
        self._check_btn.clicked.connect(self._run_check)
        self._status_pill = _pill("not checked", PAL["fg_quart"])
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self._status_pill)
        hdr.addSpacing(8)
        hdr.addWidget(self._check_btn)
        layout.addLayout(hdr)

        layout.addWidget(_divider())

        # Info panels row
        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        self._gpu_frame = self._make_info_panel("Accelerator")
        self._gpu_backend_lbl = self._add_info_row(self._gpu_frame, "Backend", "—")
        self._gpu_name_lbl = self._add_info_row(self._gpu_frame, "Device", "—")
        self._gpu_vram_lbl = self._add_info_row(self._gpu_frame, "Memory", "—")
        self._gpu_frame.layout().addStretch()
        info_row.addWidget(self._gpu_frame)

        self._disk_frame = self._make_info_panel("Disk Space")
        self._disk_free_lbl = self._add_info_row(self._disk_frame, "Free", "—")
        self._disk_total_lbl = self._add_info_row(self._disk_frame, "Total", "—")
        self._disk_frame.layout().addStretch()
        info_row.addWidget(self._disk_frame)

        self._py_frame = self._make_info_panel("Python")
        self._py_ver_lbl = self._add_info_row(self._py_frame, "Version", "—")
        self._py_exe_lbl = self._add_info_row(self._py_frame, "Path", "—")
        self._py_exe_lbl.setWordWrap(True)
        self._py_frame.layout().addStretch()
        info_row.addWidget(self._py_frame, stretch=2)

        layout.addLayout(info_row)

        # Package table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Package", "Required", "Status", "Version / Error"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

        # QProcess for the check script
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.finished.connect(self._on_check_finished)
        self._proc.errorOccurred.connect(self._on_check_error)

        ThemeManager.get().changed.connect(self._apply_theme)

    def _apply_theme(self):
        if self._last_data is not None:
            self._populate_data(self._last_data)

    @staticmethod
    def _make_info_panel(title: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("info_card")
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(12, 10, 12, 10)
        vl.setSpacing(6)
        t = QLabel(title.upper())
        t.setObjectName("section_lbl")
        vl.addWidget(t)
        return frame

    @staticmethod
    def _add_info_row(panel: QFrame, key: str, default: str) -> QLabel:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        k = QLabel(f"{key}:")
        k.setObjectName("info_key")
        v = QLabel(default)
        v.setFont(mono_font(10))
        v.setObjectName("info_val")
        row.addWidget(k)
        row.addWidget(v, stretch=1)
        panel.layout().addLayout(row)
        return v

    def _set_pill(self, text: str, color: str):
        self._status_pill.setText(text)
        self._status_pill.setStyleSheet(
            f"background: transparent; color: {color}; border: 1px solid {color};"
            f" border-radius: 10px; padding: 1px 8px; font-size: 11px; font-weight:600;"
        )

    def _run_check(self):
        self._check_btn.setEnabled(False)
        self._set_pill("checking…", PAL["running"])
        self._table.setRowCount(0)
        self._proc.start(_find_pipeline_python(), ["-c", _REQUIREMENTS_CHECK_SCRIPT])

    def _on_check_error(self, error):
        self._check_btn.setEnabled(True)
        self._set_pill("error", PAL["err"])
        self._table.setRowCount(0)
        err_row = QTableWidgetItem(self._proc.errorString())
        err_row.setForeground(QColor(PAL["err"]))
        self._table.insertRow(0)
        self._table.setItem(0, 3, err_row)

    def _on_check_finished(self, exit_code: int, _status):
        self._check_btn.setEnabled(True)
        raw = self._proc.readAllStandardOutput().data().decode("utf-8", errors="replace").strip()
        if exit_code != 0:
            self._set_pill("check failed", PAL["err"])
            self._table.setRowCount(1)
            item = QTableWidgetItem(raw or f"Process exited with code {exit_code}")
            item.setForeground(QColor(PAL["err"]))
            self._table.setItem(0, 3, item)
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._set_pill("parse error", PAL["err"])
            self._table.setRowCount(1)
            item = QTableWidgetItem(raw[:200] if raw else "Empty output from check script")
            item.setForeground(QColor(PAL["err"]))
            self._table.setItem(0, 3, item)
            return

        self._last_data = data
        self._populate_data(data)

    def _populate_data(self, data: dict):
        # GPU panel
        gpu = data.get("gpu", {})
        backend = gpu.get("backend", "?")
        if backend == "CUDA":
            cnt = gpu.get("device_count", 1)
            self._gpu_backend_lbl.setText(f"CUDA {gpu.get('cuda_version','?')} ({cnt} device{'s' if cnt>1 else ''})")
            self._gpu_backend_lbl.setStyleSheet(
                f"color:{PAL['ok']}; font-size:11px; font-weight:600; border:none; background:transparent;"
                f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            )
            self._gpu_name_lbl.setText(gpu.get("name", "?"))
            self._gpu_name_lbl.setStyleSheet(
                f"color:{PAL['fg_primary']}; font-size:11px; border:none; background:transparent;"
                f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            )
            self._gpu_vram_lbl.setText(f"{gpu.get('vram_gb','?')} GB VRAM")
            self._gpu_vram_lbl.setStyleSheet(
                f"color:{PAL['fg_tertiary']}; font-size:11px; border:none; background:transparent;"
                f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            )
        elif backend == "MPS":
            self._gpu_backend_lbl.setText("Apple Silicon MPS")
            self._gpu_backend_lbl.setStyleSheet(
                f"color:{PAL['ok']}; font-size:11px; font-weight:600; border:none; background:transparent;"
                f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            )
            self._gpu_name_lbl.setText(gpu.get("name", "Apple Silicon"))
            self._gpu_name_lbl.setStyleSheet(
                f"color:{PAL['fg_primary']}; font-size:11px; border:none; background:transparent;"
                f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            )
            self._gpu_vram_lbl.setText("Unified Memory")
            self._gpu_vram_lbl.setStyleSheet(
                f"color:{PAL['fg_tertiary']}; font-size:11px; border:none; background:transparent;"
            )
        else:
            label = gpu.get("name", gpu.get("error", backend))
            self._gpu_backend_lbl.setText(label)
            self._gpu_backend_lbl.setStyleSheet(
                f"color:{PAL['warn']}; font-size:11px; font-weight:600; border:none; background:transparent;"
            )
            self._gpu_name_lbl.setText("—")
            self._gpu_vram_lbl.setText("—")

        # Disk panel
        disk = data.get("disk", {})
        if "free_gb" in disk:
            free = disk["free_gb"]
            total = disk["total_gb"]
            disk_color = PAL["ok"] if free > 20 else (PAL["warn"] if free > 5 else PAL["err"])
            self._disk_free_lbl.setText(f"{free:.1f} GB")
            self._disk_free_lbl.setStyleSheet(
                f"color:{disk_color}; font-size:12px; font-weight:700; border:none; background:transparent;"
                f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            )
            self._disk_total_lbl.setText(f"{total:.1f} GB total")
            self._disk_total_lbl.setStyleSheet(
                f"color:{PAL['fg_tertiary']}; font-size:11px; border:none; background:transparent;"
                f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            )

        # Python panel
        py = data.get("python", {})
        self._py_ver_lbl.setText(f"Python {py.get('version', '?')}")
        self._py_ver_lbl.setStyleSheet(
            f"color:{PAL['fg_primary']}; font-size:12px; font-weight:700; border:none; background:transparent;"
            f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
        )
        self._py_exe_lbl.setText(py.get("executable", "?"))
        self._py_exe_lbl.setStyleSheet(
            f"color:{PAL['fg_tertiary']}; font-size:10px; border:none; background:transparent;"
            f" font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
        )

        # Package table
        pkgs = data.get("packages", {})
        self._table.setRowCount(len(pkgs))
        missing_required = 0

        for row, (name, info) in enumerate(pkgs.items()):
            ok = info.get("ok", False)
            required = info.get("required", False)

            name_item = QTableWidgetItem(name)
            name_item.setForeground(QColor(PAL["fg_primary"]))

            req_item = QTableWidgetItem("required" if required else "optional")
            req_item.setForeground(QColor(PAL["fg_secondary"] if required else PAL["fg_quart"]))
            req_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if ok:
                status_item = QTableWidgetItem("✓  installed")
                status_item.setForeground(QColor(PAL["ok"]))
            else:
                status_item = QTableWidgetItem("✗  missing")
                status_item.setForeground(QColor(PAL["err"] if required else PAL["warn"]))
                if required:
                    missing_required += 1
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            ver = info.get("version", info.get("error", ""))
            ver_item = QTableWidgetItem(str(ver))
            ver_item.setFont(mono_font(10))
            ver_item.setForeground(QColor(PAL["fg_tertiary"] if ok else PAL["err"]))

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, req_item)
            self._table.setItem(row, 2, status_item)
            self._table.setItem(row, 3, ver_item)

        if missing_required == 0:
            self._set_pill("all good", PAL["ok"])
        else:
            self._set_pill(f"{missing_required} required missing", PAL["err"])


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
    # Keep _buf alive until QPixmap.fromImage() has copied the data.
    # QImage wraps the buffer without copying, so the bytes object must not be GC'd first.
    _buf = bytes(np.ascontiguousarray(rgb).data)
    img = QImage(_buf, w, h, w * 3, QImage.Format.Format_RGB888)
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
        self.resize(1320, 900)

        status = QStatusBar()
        status.showMessage("Ready")
        self.setStatusBar(status)

        ckpt_panel = CheckpointPanel()

        tabs = QTabWidget()
        tabs.addTab(PipelineTab(ckpt_panel, status), "Pipeline Runner")
        tabs.addTab(MapViewerTab(), "Map Viewer")
        tabs.addTab(ResultsTab(), "Results")
        tabs.addTab(RequirementsTab(), "Requirements")

        tabs.currentChanged.connect(
            lambda i: tabs.widget(i)._load_results()
            if hasattr(tabs.widget(i), "_load_results") else None
        )

        # Theme toggle corner widget
        corner = QWidget()
        corner.setObjectName("corner_widget")
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 10, 0)
        cl.setSpacing(6)
        self._mode_lbl = QLabel("Dark")
        self._mode_lbl.setStyleSheet(
            f"color:{PAL['fg_tertiary']}; font-size:11px; background:transparent;"
        )
        self._toggle = ThemeToggle()
        cl.addWidget(self._mode_lbl)
        cl.addWidget(self._toggle)
        tabs.setCornerWidget(corner)

        ThemeManager.get().changed.connect(self._on_theme_changed)

        self.setCentralWidget(tabs)

    def _on_theme_changed(self):
        dark = ThemeManager.get().is_dark
        self._mode_lbl.setText("Dark" if dark else "Light")
        self._mode_lbl.setStyleSheet(
            f"color:{PAL['fg_tertiary']}; font-size:11px; background:transparent;"
        )


def main():
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load saved theme before building QSS
    try:
        saved = _THEME_FILE.read_text().strip()
        if saved == "light":
            PAL.update(LIGHT_PAL)
            ThemeManager.get().is_dark = False
    except (OSError, ValueError):
        pass

    app.setStyleSheet(build_qss())
    _apply_app_palette(app)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
