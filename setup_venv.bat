@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  Geo-Intel Pipeline — Setup (Windows)
REM
REM  Auto-detects and selects the correct PyTorch requirements file:
REM    NVIDIA CUDA 12.x  →  requirements-torch-cuda.txt   (cu121)
REM    NVIDIA CUDA 11.x  →  requirements-torch-cuda11.txt  (cu118)
REM    CPU / no GPU      →  requirements-torch-cpu.txt
REM
REM  Options:
REM    setup_venv.bat [--cpu] [--cuda VER]
REM    --cpu        Force CPU-only install
REM    --cuda VER   Force CUDA version  (e.g. --cuda 12.1  or  --cuda 11.8)
REM ═══════════════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion

set FORCE_CPU=0
set "FORCE_CUDA="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--cpu"  (set FORCE_CPU=1 & shift & goto parse_args)
if /i "%~1"=="--cuda" (set "FORCE_CUDA=%~2" & shift & shift & goto parse_args)
echo [warn] Unknown flag: %~1
shift
goto parse_args
:args_done

echo.
echo   ========================================
echo     Geo-Intel Pipeline - Setup (Windows)
echo   ========================================
echo.

REM ── 1/6  Check Python ──────────────────────────────────────────────────────
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10+ not found.  https://python.org
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo   Python %PYVER%

REM ── 2/6  Virtual environment ────────────────────────────────────────────────
echo.
echo [2/6] Setting up virtual environment...
if exist venv\ (
    echo   Using existing venv
) else (
    python -m venv venv
    if errorlevel 1 ( echo [ERROR] venv creation failed & pause & exit /b 1 )
    echo   Created venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip "setuptools>=69,<82" wheel --quiet
echo   pip upgraded

REM ── 3/6  Detect accelerator — select the right torch requirements file ──────
echo.
echo [3/6] Detecting accelerator...

set "TORCH_FILE=requirements-torch-cpu.txt"
set "TORCH_URL="
set "TORCH_LABEL=CPU (no GPU detected)"

REM ── Forced CPU ──────────────────────────────────────────────────────────────
if %FORCE_CPU%==1 (
    set "TORCH_LABEL=CPU (forced)"
    goto torch_selected
)

REM ── Forced CUDA ─────────────────────────────────────────────────────────────
if not "!FORCE_CUDA!"=="" (
    for /f "tokens=1 delims=." %%m in ("!FORCE_CUDA!") do set "CUDA_MAJ=%%m"
    if !CUDA_MAJ! GEQ 12 (
        set "TORCH_FILE=requirements-torch-cuda.txt"
    ) else (
        set "TORCH_FILE=requirements-torch-cuda11.txt"
    )
    set "TORCH_LABEL=NVIDIA CUDA !FORCE_CUDA! (forced)"
    goto torch_selected
)

REM ── Auto-detect NVIDIA GPU ──────────────────────────────────────────────────
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    python -c "import subprocess,re;r=subprocess.run(['nvidia-smi'],capture_output=True,text=True);m=re.search(r'CUDA Version: (\d+)',r.stdout);print(m.group(1) if m else '0')" >"%TEMP%\geo_cuda.txt" 2>nul
    set /p CUDA_MAJ=<"%TEMP%\geo_cuda.txt"
    del "%TEMP%\geo_cuda.txt" 2>nul

    if "!CUDA_MAJ!"=="" set "CUDA_MAJ=0"
    if !CUDA_MAJ! GEQ 12 (
        set "TORCH_FILE=requirements-torch-cuda.txt"
        set "TORCH_LABEL=NVIDIA CUDA !CUDA_MAJ!.x (detected)"
    ) else if !CUDA_MAJ! GEQ 11 (
        set "TORCH_FILE=requirements-torch-cuda11.txt"
        set "TORCH_LABEL=NVIDIA CUDA !CUDA_MAJ!.x (detected → cu118)"
    ) else (
        set "TORCH_LABEL=NVIDIA GPU (CUDA detection failed — using CPU)"
    )
)

:torch_selected
echo   !TORCH_LABEL!
if not "!TORCH_FILE!"==""   echo   Using: !TORCH_FILE!
if not "!TORCH_URL!"==""    echo   Using: !TORCH_URL!

REM ── 4/6  Install PyTorch ────────────────────────────────────────────────────
echo.
echo [4/6] Installing PyTorch...
if not "!TORCH_URL!"=="" (
    pip install torch torchvision torchaudio --index-url "!TORCH_URL!" --quiet
) else (
    pip install -r "!TORCH_FILE!" --quiet
)
if errorlevel 1 ( echo [ERROR] PyTorch installation failed & pause & exit /b 1 )

python _setup_verify.py --torch
echo   PyTorch ready

REM ── 5/6  Geospatial stack ───────────────────────────────────────────────────
echo.
echo [5/6] Installing geospatial stack...
conda --version >nul 2>&1
if not errorlevel 1 (
    echo   conda found - installing with ECW support
    conda install -c conda-forge libgdal-ecw rasterio fiona geopandas --yes --quiet 2>nul
    if errorlevel 1 conda install -c conda-forge rasterio fiona geopandas --yes --quiet
    echo   Installed via conda
) else (
    echo   conda not found - using pip
    pip install rasterio fiona geopandas --quiet
    if errorlevel 1 (
        echo   Retrying with pre-built wheels...
        pip install rasterio fiona --find-links https://github.com/cgohlke/geospatial-wheels/releases --quiet
        pip install geopandas --quiet
    )
    echo   Installed via pip
)

REM ── 6/6  Pipeline dependencies + CRF + verify ───────────────────────────────
echo.
echo [6/6] Installing pipeline dependencies (requirements.txt)...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo   onnxruntime-gpu unavailable - retrying with CPU onnxruntime
    findstr /v "onnxruntime" requirements.txt > "%TEMP%\geo_reqs.txt"
    pip install -r "%TEMP%\geo_reqs.txt" --quiet
    pip install onnxruntime --quiet 2>nul
    del "%TEMP%\geo_reqs.txt" 2>nul
)
echo   Dependencies installed

echo.
echo   pydensecrf2 (optional CRF)...
pip install pydensecrf2 --quiet 2>nul
if errorlevel 1 ( echo   Skipped - CRF will be disabled ) else ( echo   Installed )

REM ── Verify ──────────────────────────────────────────────────────────────────
echo.
echo   Checking ECW...
python _setup_verify.py --ecw

echo.
echo   ========================================
echo     Verification
echo   ========================================
python _setup_verify.py --verify

echo.
echo   ========================================
echo     Setup complete!
echo.
echo     Activate venv :  venv\Scripts\activate.bat
echo     Launch GUI    :  python gui.py
echo     Preprocess    :  python run_pipeline.py --mode preprocess --data_root .\dataset
echo     Train         :  python run_pipeline.py --mode train_all
echo   ========================================
echo.
pause