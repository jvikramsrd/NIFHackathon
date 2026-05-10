"""
ECW-to-GeoTIFF conversion helper.

ECW is a proprietary Hexagon format. GDAL must be built with the Hexagon ECW
SDK to read it. This module tries several strategies in order:

  1. rasterio.open()         — works if the installed GDAL has the ECW driver
  2. subprocess gdal_translate — tries EVERY found binary (QGIS first, then
                                  OSGeo4W, then conda) with correct env vars
  3. osgeo.gdal Python bindings — if installed separately

QGIS is tried first because it ships with Hexagon's free read-only ECW SDK.
OSGeo4W only has ECW support if the licensed gdal-ecw package is installed.

If all strategies fail a clear RuntimeError is raised with install instructions.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

ECW_EXTENSIONS = {".ecw", ".ECW"}


# ---------------------------------------------------------------------------
# Binary discovery — returns ALL candidates, QGIS before OSGeo4W
# ---------------------------------------------------------------------------

def _find_all_gdal_translate() -> List[str]:
    """
    Return every gdal_translate binary found on this machine.
    QGIS installations come first because they include Hexagon's free
    read-only ECW SDK.  OSGeo4W is listed after (ECW there requires a
    paid licensed plugin).
    """
    found: List[str] = []
    seen: set = set()

    def _add(p) -> None:
        s = str(p)
        if s not in seen and Path(s).exists():
            seen.add(s)
            found.append(s)

    # 1. QGIS — scan Program Files for any version, newest first
    for pf in [Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)")]:
        if pf.exists():
            for d in sorted(pf.glob("QGIS*"), reverse=True):
                _add(d / "bin" / "gdal_translate.exe")

    # 2. OSGeo4W
    _add(r"C:\OSGeo4W\bin\gdal_translate.exe")
    _add(r"C:\OSGeo4W64\bin\gdal_translate.exe")

    # 3. Conda / Miniconda environments
    conda_roots = []
    for root_candidate in [
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
        Path.home() / "AppData" / "Local" / "miniconda3",
        Path.home() / "AppData" / "Local" / "anaconda3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/ProgramData/anaconda3"),
    ]:
        if root_candidate.exists():
            conda_roots.append(root_candidate)

    for root in conda_roots:
        _add(root / "Library" / "bin" / "gdal_translate.exe")
        for env_dir in sorted((root / "envs").glob("*")):
            _add(env_dir / "Library" / "bin" / "gdal_translate.exe")

    # 4. System PATH — append last (unknown ECW capability)
    in_path = shutil.which("gdal_translate")
    if in_path:
        _add(in_path)

    return found


# ---------------------------------------------------------------------------
# Per-binary environment setup
# ---------------------------------------------------------------------------

def _env_for_binary(binary: str) -> dict:
    """
    Build a subprocess environment tuned for a specific gdal_translate binary.
    Sets GDAL_DATA, GDAL_DRIVER_PATH, and PROJ_LIB so the binary can locate
    its plugin DLLs (including the ECW plugin) regardless of system state.
    """
    env = os.environ.copy()
    bin_dir = Path(binary).parent
    root = bin_dir.parent  # e.g. C:\Program Files\QGIS 3.40 or C:\OSGeo4W

    # Prepend the binary's directory so it loads its own DLLs first
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

    # GDAL_DATA — GDAL's internal data files (projections, formats, etc.)
    for candidate in [
        root / "apps" / "gdal" / "share" / "gdal",  # QGIS layout
        root / "share" / "gdal",                     # OSGeo4W / conda layout
    ]:
        if candidate.exists():
            env["GDAL_DATA"] = str(candidate)
            break

    # GDAL_DRIVER_PATH — directory where gdal_ECW_JP2ECW.dll lives
    for candidate in [
        bin_dir / "gdalplugins",                        # QGIS 3.x
        root / "apps" / "gdal" / "lib" / "gdalplugins", # QGIS alt
        bin_dir / "gdal-plugins",                       # OSGeo4W
        root / "lib" / "gdalplugins",                   # conda
    ]:
        if candidate.exists():
            env["GDAL_DRIVER_PATH"] = str(candidate)
            break

    # PROJ_LIB — coordinate reference system data
    for candidate in [
        root / "apps" / "proj" / "share" / "proj",  # QGIS layout
        root / "share" / "proj",                     # OSGeo4W / conda
    ]:
        if candidate.exists():
            env["PROJ_LIB"] = str(candidate)
            break

    return env


# ---------------------------------------------------------------------------
# Conversion strategies
# ---------------------------------------------------------------------------

def is_ecw(path: Path) -> bool:
    return path.suffix.lower() in ECW_EXTENSIONS


def _try_rasterio(ecw_path: Path, out_tif: Path) -> bool:
    """Try converting via rasterio (succeeds only if GDAL has ECW driver)."""
    try:
        import rasterio

        with rasterio.open(str(ecw_path)) as src:
            profile = src.profile.copy()
            profile.update(driver="GTiff", compress="lzw", tiled=True,
                           blockxsize=256, blockysize=256)
            with rasterio.open(str(out_tif), "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("ecw", "driver", "not recognized", "not supported")):
            return False
        raise


def _try_gdal_translate(ecw_path: Path, out_tif: Path) -> bool:
    """
    Try every found gdal_translate binary with correct env vars until one
    successfully converts the ECW file.  QGIS binaries are tried first.
    """
    candidates = _find_all_gdal_translate()
    if not candidates:
        log.debug("[ECW] No gdal_translate binary found anywhere on this machine")
        return False

    cmd_base = ["-of", "GTiff", "-co", "COMPRESS=LZW", "-co", "TILED=YES"]

    for binary in candidates:
        env = _env_for_binary(binary)
        cmd = [binary] + cmd_base + [str(ecw_path), str(out_tif)]

        root_name = Path(binary).parent.parent.name
        log.debug("[ECW] Trying %s (GDAL_DRIVER_PATH=%s)",
                  root_name, env.get("GDAL_DRIVER_PATH", "<not set>"))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=300
            )
        except subprocess.TimeoutExpired:
            log.warning("[ECW] gdal_translate timed out for %s", root_name)
            if out_tif.exists():
                out_tif.unlink(missing_ok=True)
            continue

        if result.returncode == 0 and out_tif.exists():
            log.info("[ECW] Converted via gdal_translate (%s)", root_name)
            return True

        # Log stderr at debug so normal runs stay quiet
        stderr = result.stderr.strip()
        if stderr:
            log.debug("[ECW] %s stderr: %s", root_name, stderr[:400])

        # Remove partial output before trying next binary
        if out_tif.exists():
            out_tif.unlink(missing_ok=True)

    return False


def _try_osgeo_gdal(ecw_path: Path, out_tif: Path) -> bool:
    """Try converting via osgeo.gdal Python bindings."""
    try:
        from osgeo import gdal
        drv_names = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
        if "ECW" not in drv_names:
            return False
        ds = gdal.Open(str(ecw_path))
        if ds is None:
            return False
        gdal.Translate(str(out_tif), ds, format="GTiff",
                       creationOptions=["COMPRESS=LZW", "TILED=YES"])
        ds = None
        return out_tif.exists()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ecw_to_tif(ecw_path: Path, tmp_dir: Path) -> Path:
    """
    Convert an ECW file to a GeoTIFF in tmp_dir.
    Returns the Path to the converted TIF.
    Raises RuntimeError if no ECW-capable backend is found.
    """
    out_tif = tmp_dir / (ecw_path.stem + "__ecw_converted.tif")

    log.info("[ECW] Converting %s → GeoTIFF …", ecw_path.name)

    for strategy, fn in [
        ("rasterio (native ECW driver)", _try_rasterio),
        ("gdal_translate (subprocess)",  _try_gdal_translate),
        ("osgeo.gdal Python bindings",   _try_osgeo_gdal),
    ]:
        try:
            if fn(ecw_path, out_tif):
                log.info("[ECW] Converted via %s: %s", strategy, out_tif.name)
                return out_tif
        except Exception as exc:
            log.warning("[ECW] Strategy '%s' raised: %s", strategy, exc)

    raise RuntimeError(
        f"\n{'='*65}\n"
        f"  Cannot read ECW file: {ecw_path.name}\n"
        f"  ECW is a proprietary Hexagon format requiring the ECW SDK.\n"
        f"\n"
        f"  RECOMMENDED FIX — Install QGIS (free, includes ECW support):\n"
        f"    1. Download: https://qgis.org/download\n"
        f"    2. Run the installer with default settings\n"
        f"    3. Re-run this pipeline — ECW files will convert automatically\n"
        f"\n"
        f"  NOTE: OSGeo4W alone is NOT enough — it requires a paid\n"
        f"  Hexagon license for ECW.  QGIS includes the free read-only SDK.\n"
        f"\n"
        f"  ALTERNATIVE: Open each .ecw in QGIS and export as GeoTIFF\n"
        f"    (Raster › Convert › Translate…)\n"
        f"{'='*65}"
    )
