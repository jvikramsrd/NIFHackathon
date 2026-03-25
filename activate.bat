@echo off
REM Quick activate shortcut — double-click or run from project root
REM Opens a new CMD session with the venv active and shows status

call venv\Scripts\activate.bat

echo.
echo  Geo-Intel venv active
echo  Python: %VIRTUAL_ENV%
echo.
python -c "import torch; print(f'  PyTorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

REM Keep window open if double-clicked
if "%1"=="" cmd /k
