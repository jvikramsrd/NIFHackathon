@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  Geo-Intel Pipeline — Launch GUI
REM  Double-click this file to open the Operator Console.
REM ═══════════════════════════════════════════════════════════════════════════
setlocal

cd /d "%~dp0"

REM ── Activate venv if present, otherwise fall back to system Python ──────────
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [warn] venv not found — using system Python.  Run setup_venv.bat first.
)

REM ── Launch the GUI ──────────────────────────────────────────────────────────
python gui.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with an error.  See output above.
    pause
)
