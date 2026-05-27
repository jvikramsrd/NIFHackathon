@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  Geo-Intel Pipeline — CLI wrapper
REM
REM  Resolution order:
REM    1. If dist\geo-intel\geo-intel.exe exists (a built binary), use it.
REM    2. Else fall through to the venv Python + run_pipeline.py.
REM
REM  Usage examples:
REM    run.bat --mode preprocess --data_root .\dataset
REM    run.bat --mode train_all
REM    run.bat --mode infer --tif .\village.tif --out .\outputs\village
REM    run.bat --mode evaluate
REM ═══════════════════════════════════════════════════════════════════════════
setlocal

cd /d "%~dp0"

REM ── 1. Prefer the built CLI binary if present ───────────────────────────────
if exist "dist\geo-intel\geo-intel.exe" (
    "dist\geo-intel\geo-intel.exe" %*
    exit /b %errorlevel%
)

REM ── 2. Fallback: activate venv (if any) and invoke run_pipeline.py ──────────
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [warn] No built binary in dist\ and no venv found.
    echo        Run `setup_venv.bat` once to provision the Python environment,
    echo        or `python build.py` to produce dist\geo-intel\geo-intel.exe.
)

python run_pipeline.py %*
