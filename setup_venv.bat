@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  Geo-Intel A4000 Pipeline — Virtual Environment Setup
REM  Run ONCE from inside the geo_intel_a4000 folder:
REM      cd geo_intel_a4000
REM      setup_venv.bat
REM ═══════════════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║      Geo-Intel A4000 — Environment Setup                ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

python --version >nul 2>&1
if errorlevel 1 ( echo [ERROR] Python not found. Install 3.10/3.11 from python.org & pause & exit /b 1 )
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%

echo.
echo [1/7] Creating venv ...
if exist venv ( echo       Already exists. ) else (
    python -m venv venv
    if errorlevel 1 ( echo [ERROR] venv failed & pause & exit /b 1 )
    echo [OK] Created.
)

echo.
echo [2/7] Activating ...
call venv\Scripts\activate.bat
echo [OK] Active.

echo.
echo [3/7] Upgrading pip ...
python -m pip install --upgrade pip setuptools wheel --quiet
echo [OK] Done.

echo.
echo [4/7] Installing PyTorch 2.x + CUDA 12.1 (~2.5 GB) ...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 ( echo [ERROR] PyTorch failed. & pause & exit /b 1 )
python -c "import torch; ok=torch.cuda.is_available(); print('  CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'N/A')"

echo.
echo [5/7] Installing geospatial stack ...
echo       Checking for conda (needed for ECW support) ...
conda --version >nul 2>&1
if errorlevel 1 (
    echo       Conda not found — using pip.
    pip install rasterio fiona geopandas --quiet 2>nul
    if errorlevel 1 (
        echo       Trying pre-built wheels ...
        pip install rasterio --find-links https://github.com/cgohlke/geospatial-wheels/releases --quiet
        pip install fiona     --find-links https://github.com/cgohlke/geospatial-wheels/releases --quiet
        pip install geopandas --quiet
    )
) else (
    echo       Conda found — installing with ECW support ...
    conda install -c conda-forge libgdal-ecw rasterio fiona geopandas --yes --quiet
    echo [OK] Geospatial + ECW installed.
)

echo.
echo [6/7] Installing ML packages ...
pip install ^
    segmentation-models-pytorch>=0.3.3 ^
    timm>=0.9.12 ^
    ultralytics>=8.1.0 ^
    shapely>=2.0.3 pyproj>=3.6.1 ^
    albumentations>=1.4.0 opencv-python>=4.9.0 Pillow>=10.2.0 ^
    numpy>=1.26.0 pandas>=2.1.0 scipy>=1.12.0 scikit-learn>=1.4.0 ^
    tqdm>=4.66.1 pyyaml>=6.0.1 matplotlib>=3.8.0 tensorboard>=2.16.0 ^
    dbfread lxml --quiet
if errorlevel 1 ( echo [ERROR] Some packages failed. & pause & exit /b 1 )
echo [OK] Done.

echo.
echo [7/7] pydensecrf2 (optional) ...
pip install pydensecrf2 --quiet 2>nul
if errorlevel 1 ( echo       Not available — CRF skipped automatically. ) else ( echo [OK] Installed. )

echo.
echo       Checking ECW driver ...
python -c "
try:
    from osgeo import gdal
    d=[gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
    print('  ECW driver:', 'YES' if 'ECW' in d else 'NO — run: conda install -c conda-forge libgdal-ecw')
except: print('  ECW: cannot check')
"

echo.
echo ════════════════════════════════════════════════════════════
echo  Summary
echo ════════════════════════════════════════════════════════════
python -c "
import torch, cv2, numpy, albumentations, segmentation_models_pytorch, timm, rasterio, geopandas, ultralytics
p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
items = [
    ('torch',    torch.__version__),
    ('CUDA',     str(torch.cuda.is_available())),
    ('GPU',      p.name if p else 'N/A'),
    ('VRAM',     f'{p.total_memory/1024**3:.1f} GB' if p else 'N/A'),
    ('rasterio', rasterio.__version__),
    ('geopandas',geopandas.__version__),
    ('timm',     timm.__version__),
    ('ultralytics', ultralytics.__version__),
]
[print(f'  {k:<20} {v}') for k,v in items]
"
echo.
echo ════════════════════════════════════════════════════════════
echo  Setup complete!
echo.
echo  Next steps:
echo    1. Activate next time: activate.bat
echo    2. Preprocess your data:
echo       python run_pipeline.py --mode preprocess --data_root C:\Users\Vikram\Downloads\dataset
echo    3. Train:
echo       python run_pipeline.py --mode train_all
echo ════════════════════════════════════════════════════════════
echo.
pause
