@echo off
rem ===================================================================
rem  P16 stage 2: one-click run on Windows (NVIDIA GPU required)
rem  Double-click this file. Safe to re-run: finished steps are skipped.
rem  Heavy files go to %USERPROFILE%\cv-p16-data (outside OneDrive).
rem  Optional arguments are passed to scripts\run_all.py, e.g.
rem      run_baseline.bat --epochs 3 --train-fraction 0.5
rem ===================================================================
setlocal
cd /d "%~dp0"
set "WORK=%USERPROFILE%\cv-p16-data"
set "VENV=%WORK%\venv"

echo.
echo [1] Checking NVIDIA GPU...
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if errorlevel 1 (
  echo.
  echo No NVIDIA GPU / driver found. Install the NVIDIA driver, or use the Kaggle notebook instead.
  goto :fail
)

if exist "%VENV%\Scripts\python.exe" goto :run

echo.
echo [2] Finding Python 3.10-3.13...
set "PY="
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.10-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH"^) and run this file again.
  goto :fail
)
echo Using: %PY%

echo.
echo [3] Creating environment in %VENV% and installing PyTorch with CUDA 12.8 (~3 GB, once; needed for RTX 50xx)...
if not exist "%WORK%" mkdir "%WORK%"
%PY% -m venv "%VENV%" || goto :fail
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
"%VENV%\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :fail
"%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail

:run
"%VENV%\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch - update the NVIDIA driver'; print('PyTorch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))" || goto :fail

echo.
echo [4] Running the pipeline (download, splits, smoke test, baseline A, reference)...
"%VENV%\Scripts\python.exe" -u scripts\run_all.py --work "%WORK%" %* || goto :fail

echo.
echo ================= ALL DONE =================
echo Results: %~dp0results   Archive: %~dp0stage2_outputs.zip
pause
exit /b 0

:fail
echo.
echo ================= STOPPED WITH AN ERROR =================
echo Log: %~dp0run_log.txt
pause
exit /b 1
