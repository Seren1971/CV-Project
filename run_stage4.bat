@echo off
rem Stage 4: baseline A and variant B' (Bp), 12 epochs each, then all predictions.
cd /d C:\Users\spesh\OneDrive\Desktop\Study\CV
set PY=C:\Users\spesh\cv-p16-data\venv\Scripts\python.exe
set IMG=C:\Users\spesh\cv-p16-data\raw\eccv_18_all_images_sm
set LOG=C:\Users\spesh\cv-p16-data\stage4_log.txt

echo === stage 4 started %DATE% %TIME% === >> %LOG%

echo --- train A 12 epochs --- >> %LOG%
%PY% -m src.train --variant A --epochs 12 --seed 0 --workers 2 --image-dir %IMG% --out C:\Users\spesh\cv-p16-data\runs\A12_s0 >> %LOG% 2>&1

echo --- train Bp 12 epochs --- >> %LOG%
%PY% -m src.train --variant Bp --epochs 12 --seed 0 --workers 2 --image-dir %IMG% --out C:\Users\spesh\cv-p16-data\runs\Bp12_s0 >> %LOG% 2>&1

for %%M in (A12_s0 Bp12_s0) do (
  for %%S in (val test test_cis) do (
    echo --- predict %%M %%S --- >> %LOG%
    %PY% -m src.predict --ckpt C:\Users\spesh\cv-p16-data\runs\%%M\best.pt --split %%S --image-dir %IMG% --out C:\Users\spesh\cv-p16-data\runs\%%M\preds_%%S.json >> %LOG% 2>&1
  )
)

echo === ALL DONE %DATE% %TIME% === >> %LOG%
