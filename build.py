"""
build.py — cross-platform binary builder for Geo-Intel GUI
==========================================================
Run from the project root:

    python build.py            # build for the current platform
    python build.py --clean    # remove dist/ and build/ first
    python build.py --check    # verify deps, don't build
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"

REQUIRED_FOR_BUILD = [
    "PyQt6",
    "cv2",
    "numpy",
    "matplotlib",
    "PyInstaller",
]


def _check_deps():
    missing = []
    for mod in REQUIRED_FOR_BUILD:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[ERROR] Missing build dependencies: {', '.join(missing)}")
        print("       Install with:")
        print("         pip install PyQt6 opencv-python numpy matplotlib pyinstaller")
        return False
    print("[OK] All build dependencies present.")
    return True


def _clean():
    for d in (DIST, BUILD):
        if d.exists():
            shutil.rmtree(d)
            print(f"[clean] removed {d}")


def _build():
    spec = ROOT / "geo_intel.spec"
    if not spec.exists():
        print(f"[ERROR] Spec file not found: {spec}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec),
    ]
    print(f"[build] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("[ERROR] PyInstaller failed.")
        sys.exit(result.returncode)

    _report()


def _report():
    platform = sys.platform
    if platform == "win32":
        binary = DIST / "GeoIntel" / "GeoIntel.exe"
    elif platform == "darwin":
        binary = DIST / "GeoIntel.app"
    else:
        binary = DIST / "GeoIntel" / "GeoIntel"

    if binary.exists():
        size_mb = sum(
            f.stat().st_size for f in binary.rglob("*") if f.is_file()
        ) / 1_048_576 if binary.is_dir() else binary.stat().st_size / 1_048_576
        print(f"\n[done] Binary: {binary}")
        print(f"       Size:   {size_mb:.0f} MB")
        print()
        if platform == "win32":
            print("  Launch:  dist\\GeoIntel\\GeoIntel.exe")
        elif platform == "darwin":
            print("  Launch:  open dist/GeoIntel.app")
            print("           OR: dist/GeoIntel/GeoIntel")
        else:
            print("  Launch:  ./dist/GeoIntel/GeoIntel")
    else:
        print(f"[warn] Expected binary not found at {binary}")
        print("       Check dist/ for the actual output.")


def main():
    ap = argparse.ArgumentParser(description="Build Geo-Intel GUI binary")
    ap.add_argument("--clean", action="store_true", help="Remove dist/ and build/ first")
    ap.add_argument("--check", action="store_true", help="Check deps only, do not build")
    args = ap.parse_args()

    if args.clean:
        _clean()

    if not _check_deps():
        sys.exit(1)

    if args.check:
        print("[check] Done. Pass --build or omit flags to actually build.")
        return

    _build()


if __name__ == "__main__":
    main()
