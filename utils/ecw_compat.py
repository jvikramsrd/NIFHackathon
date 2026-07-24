"""
utils/ecw_compat.py  —  ECW support for the Geo-Intel pipeline
═══════════════════════════════════════════════════════════════

WHY THIS EXISTS
───────────────
pip-installed rasterio ships its own private GDAL DLL.  The QGIS ECW plugin
(gdal_ECW_JP2ECW.dll) links against QGIS's gdal312.dll — a different GDAL
instance.  You cannot inject one into the other.

HOW WE HANDLE IT
────────────────
Use QGIS's gdal_translate.exe to convert ECW → a temporary GeoTIFF.
The temp TIF is created in the system TEMP folder and deleted automatically
when processing of that raster finishes.

CRITICAL: convert ECW files ONE AT A TIME.
The Hexagon ECW SDK 5.5 (inside QGIS) silently fails when multiple
gdal_translate processes open it concurrently.  Always call ecw_to_tif()
sequentially — never in parallel.  The caller (preprocess_folder) ensures
this by converting all ECW files before spawning worker processes.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from utils.core import get_logger

log = get_logger(__name__)

ECW_EXTENSIONS = {".ecw", ".ECW"}

# Cached result of ensure_ecw_driver() — None = not yet checked.
_ecw_ready: Optional[bool] = None


def is_ecw(path: Path) -> bool:
    """Return True if path is an ECW file."""
    return path.suffix.lower() in ECW_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# QGIS binary discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_gdal_translate() -> Optional[str]:
    """
    Return the path to a QGIS (ECW-capable) gdal_translate.exe, or None.

    Searches QGIS install directories, OSGeo4W, and then PATH.
    """
    import shutil

    candidates: List[str] = []

    # QGIS on Windows: Program Files / Program Files (x86)
    for pf in [Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)")]:
        if pf.exists():
            for qdir in sorted(pf.glob("QGIS*"), reverse=True):
                exe = qdir / "bin" / "gdal_translate.exe"
                if exe.exists():
                    candidates.append(str(exe))

    for osgeo in [Path(r"C:\OSGeo4W"), Path(r"C:\OSGeo4W64")]:
        exe = osgeo / "bin" / "gdal_translate.exe"
        if exe.exists():
            candidates.append(str(exe))

    # conda environments
    for base in [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / "AppData" / "Local" / "miniconda3",
        Path("C:/ProgramData/miniconda3"),
    ]:
        if base.exists():
            exe = base / "Library" / "bin" / "gdal_translate.exe"
            if exe.exists():
                candidates.append(str(exe))

    # Anything on PATH
    in_path = shutil.which("gdal_translate")
    if in_path:
        candidates.append(in_path)

    # Return first that reports ECW support
    for binary in candidates:
        env = _qgis_env(binary)
        try:
            r = subprocess.run(
                [binary, "--formats"],
                capture_output=True, text=True, env=env, timeout=15,
            )
            if "ECW" in r.stdout:
                return binary
        except Exception:
            continue
    return None


def _qgis_env(binary: str) -> dict:
    """Build a subprocess environment with GDAL_DATA/PROJ_LIB set for a binary."""
    env = os.environ.copy()
    bin_dir = Path(binary).parent
    root = bin_dir.parent

    # Put QGIS bin first so NCSEcw.dll is found by gdal_ECW_JP2ECW.dll
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

    for candidate in [root / "apps" / "gdal" / "share" / "gdal",
                      root / "share" / "gdal"]:
        if candidate.exists():
            env["GDAL_DATA"] = str(candidate)
            break

    for candidate in [root / "apps" / "gdal" / "lib" / "gdalplugins",
                      bin_dir / "gdalplugins",
                      root / "lib" / "gdalplugins"]:
        if candidate.exists():
            env["GDAL_DRIVER_PATH"] = str(candidate)
            break

    for candidate in [root / "apps" / "proj" / "share" / "proj",
                      root / "share" / "proj"]:
        if candidate.exists():
            env["PROJ_LIB"] = str(candidate)
            break

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ensure_ecw_driver() -> bool:
    """
    Return True if this machine can convert ECW files (via QGIS gdal_translate).

    Idempotent — checks at most once per process.
    """
    global _ecw_ready
    if _ecw_ready is not None:
        return _ecw_ready

    binary = _find_gdal_translate()
    if binary:
        root_name = Path(binary).parent.parent.name
        import multiprocessing
        if multiprocessing.current_process().name == "MainProcess":
            log.info(
                "[ECW] %s gdal_translate supports ECW — "
                "ECW files will be pre-converted to temp TIFs before processing "
                "(sequential, auto-deleted, zero permanent disk overhead).",
                root_name,
            )
        else:
            log.debug("[ECW] %s gdal_translate supports ECW", root_name)
        _ecw_ready = True
    else:
        log.warning(
            "[ECW] No ECW-capable gdal_translate found.\n"
            "  Windows: install QGIS (free) → https://qgis.org/download\n"
            "  Linux:   conda install -c conda-forge libgdal-ecw\n"
            "  Non-ECW rasters (GeoTIFF, IMG) work without this."
        )
        _ecw_ready = False

    return _ecw_ready


def ecw_to_tif(ecw_path: Path, out_dir: Path) -> Path:
    """
    Convert *ecw_path* to a DEFLATE-compressed GeoTIFF in *out_dir*.

    Returns the Path to the written TIF.  Raises RuntimeError on failure.

    IMPORTANT: Call this from a single thread/process only — never in parallel.
    The Hexagon ECW SDK silently corrupts output when multiple conversions run
    simultaneously.
    """
    binary = _find_gdal_translate()
    if not binary:
        raise RuntimeError(
            f"Cannot read {ecw_path.name}: no ECW-capable gdal_translate found.\n"
            f"Install QGIS: https://qgis.org/download"
        )

    out_tif = out_dir / (re.sub(r"[^\w]", "_", ecw_path.stem) + "__ecw.tif")
    partial = out_tif.with_suffix(".partial.tif")

    env = _qgis_env(binary)
    cmd = [
        binary,
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-co", "PREDICTOR=2",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        "-co", "BIGTIFF=IF_SAFER",
        str(ecw_path), str(partial),
    ]

    log.info("[ECW] Converting %s → temp TIF …", ecw_path.name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"gdal_translate timed out after 60 min for {ecw_path.name}")

    if result.returncode != 0 or not partial.exists() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"gdal_translate failed (rc={result.returncode}) for {ecw_path.name}\n"
            f"  stdout: {stdout[:300] or '(empty)'}\n"
            f"  stderr: {stderr[:300] or '(empty)'}"
        )

    partial.rename(out_tif)
    log.info("[ECW] → %s  (%.1f GB)", out_tif.name, out_tif.stat().st_size / 1e9)
    return out_tif
