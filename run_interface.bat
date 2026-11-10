@echo off
rem Starts the browser interface: drop a photo, get the detections drawn on it.
rem   run_interface.bat [checkpoint.pt]
rem The page opens at http://127.0.0.1:8765 automatically. Close this window to stop.
setlocal
if defined P16_PYTHON (set "PY=%P16_PYTHON%") else if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else if exist "C:\Users\spesh\cv-p16-data\venv\Scripts\python.exe" (set "PY=C:\Users\spesh\cv-p16-data\venv\Scripts\python.exe") else (set "PY=python")

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=%P16_CKPT%"
if "%CKPT%"=="" if exist "models\Bp\best.pt" set "CKPT=models\Bp\best.pt"
if "%CKPT%"=="" if exist "C:\Users\spesh\cv-p16-data\runs\Bp12_s0\best.pt" set "CKPT=C:\Users\spesh\cv-p16-data\runs\Bp12_s0\best.pt"
if "%CKPT%"=="" (
  echo No checkpoint found. Run:  run_interface.bat path\to\best.pt
  pause & exit /b 2
)

"%PY%" scripts\web_demo.py --ckpt "%CKPT%" --threshold 0.9
pause
