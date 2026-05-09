# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Geo-Intel GUI binary.

The binary bundles only the GUI shell (gui.py) + the pipeline Python source
files. PyTorch, rasterio, and other heavy ML libraries are NOT bundled —
they live in the user's Python environment and are invoked via subprocess.

This keeps the binary under ~400 MB on all platforms.

Build with:
    python build.py
    OR
    pyinstaller geo_intel.spec
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)

# ── Source files to embed (pipeline code runs as subprocess but needs to be
#    findable relative to the binary's _MEIPASS temp dir) ─────────────────────
pipeline_sources = [
    (str(ROOT / "config.py"),          "."),
    (str(ROOT / "run_pipeline.py"),    "."),
    (str(ROOT / "infer_folder.py"),    "."),
    (str(ROOT / "export_models.py"),   "."),
    (str(ROOT / "data"),               "data"),
    (str(ROOT / "models"),             "models"),
    (str(ROOT / "train"),              "train"),
    (str(ROOT / "inference"),          "inference"),
    (str(ROOT / "utils"),              "utils"),
]

# ── Collect data files from libraries that need them at runtime ───────────────
lib_datas = []
for pkg in ("matplotlib", "pyproj", "certifi"):
    try:
        lib_datas += collect_data_files(pkg)
    except Exception:
        pass

# rasterio ships GDAL data and proj databases inside the wheel
for pkg in ("rasterio", "fiona"):
    try:
        lib_datas += collect_data_files(pkg, include_py_files=False)
    except Exception:
        pass

all_datas = pipeline_sources + lib_datas

# ── Hidden imports — modules discovered at runtime that PyInstaller misses ────
hidden_imports = [
    # PyQt6 essentials
    "PyQt6.sip",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtPrintSupport",
    # matplotlib Qt backend
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt",
    # cv2
    "cv2",
    # numpy
    "numpy",
    "numpy.core._multiarray_umath",
    # rasterio / GDAL (loaded as plugins)
    "rasterio._shim",
    "rasterio.control",
    "rasterio.crs",
    "rasterio.sample",
    "rasterio.vrt",
    "rasterio._features",
    # json / stdlib
    "json",
    "csv",
    "re",
    "subprocess",
    "shutil",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "gui.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML libs — they run in the user's env, not the bundle
        "torch", "torchvision", "torchaudio",
        "tensorflow", "tensorflow_core",
        "jax",
        "sklearn", "scikit_learn",
        "scipy",
        "albumentations",
        "segmentation_models_pytorch",
        "timm",
        "ultralytics",
        "sahi",
        "geopandas",
        "fiona",
        "shapely",
        "pandas",
        "PIL",
        "Pillow",
        "wandb",
        "tensorboard",
        "onnx",
        "onnxruntime",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="GeoIntel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    disable_windowed_traceback=False,
    target_arch=None,       # host arch; override via --target-arch on Apple Silicon
    codesign_identity=None,
    entitlements_file=None,
    icon=None,              # add a .ico / .icns here when available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="GeoIntel",
)

# macOS .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="GeoIntel.app",
        icon=None,
        bundle_identifier="com.geointel.pipeline",
        info_plist={
            "NSPrincipalClass": "NSApplication",
            "NSAppleScriptEnabled": False,
            "CFBundleDisplayName": "Geo-Intel",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
