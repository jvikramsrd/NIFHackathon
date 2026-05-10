@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  Geo-Intel Pipeline — CLI wrapper
REM
REM  Usage examples:
REM    run.bat --mode preprocess --data_root .\dataset
REM    run.bat --mode train_all
REM    run.bat --mode infer --tif .\village.tif --out .\outputs\village
REM    run.bat --mode evaluate
REM ═══════════════════════════════════════════════════════════════════════════
setlocal

cd /d "%~dp0"

REM ── Activate venv if present, otherwise fall back to system Python ──────────
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [warn] venv not found — using system Python.  Run setup_venv.bat first.
)

REM ── Pass all arguments straight to run_pipeline.py ──────────────────────────
python run_pipeline.py %*
