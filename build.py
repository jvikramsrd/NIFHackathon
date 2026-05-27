"""
build.py — cross-platform binary builder for Geo-Intel
======================================================
Builds two executables from the same PyInstaller setup:

    GeoIntel(.exe)   ← GUI shell, double-click to launch (geo_intel.spec)
    geo-intel(.exe)  ← CLI runner for preprocess/train/infer (geo_intel_cli.spec)

Run from the project root:

    python build.py                  # build BOTH GUI and CLI
    python build.py --gui            # build only the GUI binary
    python build.py --cli            # build only the CLI binary
    python build.py --clean          # remove dist/ and build/ first
    python build.py --check          # verify deps, don't build

Both binaries follow the same architecture: the heavy ML libraries (PyTorch,
ultralytics, segmentation_models_pytorch, etc.) are NOT bundled. They live in
the user's Python environment, which the binary auto-detects at runtime.
This keeps each binary under ~400 MB instead of ~5 GB.
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


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────


def _check_deps() -> bool:
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


def _clean() -> None:
    for d in (DIST, BUILD):
        if d.exists():
            shutil.rmtree(d)
            print(f"[clean] removed {d}")


# ─────────────────────────────────────────────────────────────────────────────
# Build one binary from its .spec
# ─────────────────────────────────────────────────────────────────────────────


def _build_spec(spec_filename: str, label: str) -> None:
    spec = ROOT / spec_filename
    if not spec.exists():
        print(f"[ERROR] Spec file not found: {spec}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec),
    ]
    print(f"\n[build:{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ERROR] PyInstaller failed for {label} binary.")
        sys.exit(result.returncode)


def _report(binary_name: str, label: str, console: bool) -> None:
    platform = sys.platform
    if platform == "win32":
        binary = DIST / binary_name / f"{binary_name}.exe"
    elif platform == "darwin":
        # GUI -> .app bundle; CLI -> regular folder/binary
        if not console:
            binary = DIST / f"{binary_name}.app"
        else:
            binary = DIST / binary_name / binary_name
    else:
        binary = DIST / binary_name / binary_name

    if binary.exists():
        size_mb = (
            sum(f.stat().st_size for f in binary.rglob("*") if f.is_file())
            if binary.is_dir()
            else binary.stat().st_size
        ) / 1_048_576
        print(f"\n[done:{label}] Binary: {binary}")
        print(f"            Size:   {size_mb:.0f} MB")
    else:
        print(f"\n[warn:{label}] Expected binary not found at {binary}")
        print(f"             Check {DIST} for the actual output.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Geo-Intel binaries")
    ap.add_argument("--gui",   action="store_true", help="Build only the GUI binary")
    ap.add_argument("--cli",   action="store_true", help="Build only the CLI binary")
    ap.add_argument("--clean", action="store_true", help="Remove dist/ and build/ first")
    ap.add_argument("--check", action="store_true", help="Check deps only, do not build")
    args = ap.parse_args()

    if args.clean:
        _clean()

    if not _check_deps():
        sys.exit(1)

    if args.check:
        print("[check] Done. Omit flags to actually build.")
        return

    # Default (no flags): build both. Otherwise honour explicit flags.
    build_gui = args.gui or not (args.gui or args.cli)
    build_cli = args.cli or not (args.gui or args.cli)

    if build_gui:
        _build_spec("geo_intel.spec", "gui")
        _report("GeoIntel", "gui", console=False)

    if build_cli:
        _build_spec("geo_intel_cli.spec", "cli")
        _report("geo-intel", "cli", console=True)

    print()
    print("  ─" * 30)
    print("  Outputs:")
    if build_gui:
        print(f"    GUI : {DIST / 'GeoIntel'}/")
    if build_cli:
        print(f"    CLI : {DIST / 'geo-intel'}/")
    print()
    print("  Launch (Windows):")
    if build_gui:
        print("    dist\\GeoIntel\\GeoIntel.exe")
    if build_cli:
        print("    dist\\geo-intel\\geo-intel.exe --mode infer --tif ortho.tif --out ./out")
    print()
    print("  Launch (macOS / Linux):")
    if build_gui and sys.platform == "darwin":
        print("    open dist/GeoIntel.app")
    elif build_gui:
        print("    ./dist/GeoIntel/GeoIntel")
    if build_cli:
        print("    ./dist/geo-intel/geo-intel --mode infer --tif ortho.tif --out ./out")
    print("  ─" * 30)


if __name__ == "__main__":
    main()
