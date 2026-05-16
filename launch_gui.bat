@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  Geo-Intel Pipeline — Launch GUI
REM
REM  Resolution order:
REM    1. If dist\GeoIntel\GeoIntel.exe exists (a built binary), launch it.
REM    2. Else activate venv and run gui.py.
REM
REM  Double-click this file to open the Operator Console.
REM ═══════════════════════════════════════════════════════════════════════════
setlocal

cd /d "%~dp0"

REM ── 1. Prefer the built GUI binary if present ───────────────────────────────
if exist "dist\GeoIntel\GeoIntel.exe" (
    start "" "dist\GeoIntel\GeoIntel.exe" %*
    exit /b 0
)

REM ── 2. Fallback: activate venv (if any) and run gui.py ──────────────────────
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [warn] No built binary in dist\ and no venv found.
    echo        Run `setup_venv.bat` once to provision the Python environment,
    echo        or `python build.py --gui` to produce dist\GeoIntel\GeoIntel.exe.
)

python gui.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with an error.  See output above.
    pause
)
