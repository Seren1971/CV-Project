@echo off
rem Runs the final detector on a photo or a folder of photos (CLI, no browser).
rem   run_model.bat [checkpoint.pt] [image-or-folder]
rem Defaults: checkpoint from P16_CKPT or the usual local paths, images from demo\images.
setlocal
if defined P16_PYTHON (set "PY=%P16_PYTHON%") else if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else if exist "C:\Users\spesh\cv-p16-data\venv\Scripts\python.exe" (set "PY=C:\Users\spesh\cv-p16-data\venv\Scripts\python.exe") else (set "PY=python")

set "CKPT=%~1"
if "%CKPT%"=="" set "CKPT=%P16_CKPT%"
if "%CKPT%"=="" if exist "models\Bp\best.pt" set "CKPT=models\Bp\best.pt"
if "%CKPT%"=="" if exist "C:\Users\spesh\cv-p16-data\runs\Bp12_s0\best.pt" set "CKPT=C:\Users\spesh\cv-p16-data\runs\Bp12_s0\best.pt"
if "%CKPT%"=="" (
  echo No checkpoint found. Run:  run_model.bat path\to\best.pt [photo.jpg]
  pause & exit /b 2
)

set "IMGS=%~2"
if "%IMGS%"=="" set "IMGS=demo\images"

"%PY%" scripts\demo.py --ckpt "%CKPT%" --threshold 0.9 --images "%IMGS%" --out demo\out
pause
