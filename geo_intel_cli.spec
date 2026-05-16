# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the Geo-Intel CLI binary (``geo-intel``).

Wraps ``run_pipeline.py`` as a console executable. Same architecture as the
GUI binary (``geo_intel.spec``): the heavy ML libraries (PyTorch, ultralytics,
segmentation_models_pytorch, etc.) are NOT bundled — they live in the user's
Python environment and are invoked in-process or via subprocess. This keeps
the binary under ~150 MB on all platforms.

Build with:
    python build.py --cli
    OR
    pyinstaller geo_intel_cli.spec
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH)

# ── Source files to embed — pipeline code lives next to the binary ───────────
pipeline_sources = [
    (str(ROOT / "config.py"),          "."),
    (str(ROOT / "run_pipeline.py"),    "."),
    (str(ROOT / "infer_folder.py"),    "."),
    (str(ROOT / "export_models.py"),   "."),
    (str(ROOT / "run_stage2b.py"),     "."),
    (str(ROOT / "data"),               "data"),
    (str(ROOT / "models"),             "models"),
    (str(ROOT / "train"),              "train"),
    (str(ROOT / "inference"),          "inference"),
    (str(ROOT / "utils"),              "utils"),
]

# ── Data files from libs we DO bundle (small, geospatial-critical) ───────────
lib_datas = []
for pkg in ("pyproj", "certifi"):
    try:
        lib_datas += collect_data_files(pkg)
    except Exception:
        pass
for pkg in ("rasterio", "fiona"):
    try:
        lib_datas += collect_data_files(pkg, include_py_files=False)
    except Exception:
        pass

all_datas = pipeline_sources + lib_datas

# ── Hidden imports — modules loaded dynamically by name ──────────────────────
hidden_imports = [
    # cv2 / numpy core
    "cv2",
    "numpy",
    "numpy.core._multiarray_umath",
    # rasterio / GDAL plugins
    "rasterio._shim",
    "rasterio.control",
    "rasterio.crs",
    "rasterio.sample",
    "rasterio.vrt",
    "rasterio._features",
    # stdlib called by reflection
    "json",
    "csv",
    "re",
    "subprocess",
    "shutil",
    "argparse",
    "logging",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "run_pipeline.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML libs — they run in the user's Python env, not the
        # bundle. The CLI imports them lazily inside subcommand handlers, so
        # the binary itself only needs them present at run-time, not bundled.
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
        # GUI-only — keep CLI binary lean
        "PyQt6",
        "matplotlib",
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
    name="geo-intel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,           # CLI: keep the terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="geo-intel",
)
