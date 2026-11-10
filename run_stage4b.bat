@echo off
rem Stage 4 retry: variant B' 12 epochs + its predictions (A12 already done).
cd /d C:\Users\spesh\OneDrive\Desktop\Study\CV
set PY=C:\Users\spesh\cv-p16-data\venv\Scripts\python.exe
set IMG=C:\Users\spesh\cv-p16-data\raw\eccv_18_all_images_sm
set LOG=C:\Users\spesh\cv-p16-data\stage4_log.txt

echo --- train Bp 12 epochs (retry) %DATE% %TIME% --- >> %LOG%
%PY% -m src.train --variant Bp --epochs 12 --seed 0 --workers 2 --image-dir %IMG% --out C:\Users\spesh\cv-p16-data\runs\Bp12_s0 >> %LOG% 2>&1

for %%S in (val test test_cis) do (
  echo --- predict Bp12_s0 %%S --- >> %LOG%
  %PY% -m src.predict --ckpt C:\Users\spesh\cv-p16-data\runs\Bp12_s0\best.pt --split %%S --image-dir %IMG% --out C:\Users\spesh\cv-p16-data\runs\Bp12_s0\preds_%%S.json >> %LOG% 2>&1
)

echo === BP DONE %DATE% %TIME% === >> %LOG%
