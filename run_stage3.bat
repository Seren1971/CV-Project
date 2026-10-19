@echo off
rem ===================================================================
rem  P16 stage 3: variant B (targeted augmentation) + comparison with A.
rem  Run run_baseline.bat first (it installs everything and trains A).
rem  Double-click. Safe to re-run: finished steps are skipped.
rem ===================================================================
setlocal
cd /d "%~dp0"
set "WORK=%USERPROFILE%\cv-p16-data"
set "PYEXE=%WORK%\venv\Scripts\python.exe"

if not exist "%PYEXE%" (
  echo Environment not found in %WORK%\venv. Run run_baseline.bat first.
  goto :fail
)
"%PYEXE%" -u scripts\run_stage3.py --work "%WORK%" %* || goto :fail

echo.
echo ================= ALL DONE =================
echo Comparison: %~dp0results\comparison_A_vs_B.md
pause
exit /b 0

:fail
echo.
echo ================= STOPPED WITH AN ERROR =================
echo Log: %~dp0run_log.txt
pause
exit /b 1
