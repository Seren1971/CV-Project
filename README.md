# P16 — Wildlife Detection in Camera-Trap Images (Computer Vision 2026 project)

Dataset: CCT20 subset of Caltech Camera Traps (LILA BC). Details: [docs/dataset_preparation.md](docs/dataset_preparation.md).

## Project plan

The consolidated plan is in [docs/project_plan.md](docs/project_plan.md). Individual sections: [scope](docs/project_scope.md), [dataset](docs/dataset_preparation.md), [implementation](docs/implementation_plan.md), [experiments](docs/experimental_plan.md), and [responsibilities](docs/team_responsibilities.md).

## Quick start on Kaggle (recommended)

Open [notebooks/baseline_kaggle.ipynb](notebooks/baseline_kaggle.ipynb) on Kaggle, set **Accelerator: GPU** and **Internet: On**, then *Run All*. It downloads the data, builds the splits, runs a smoke test, the COCO reference and baseline A, and saves `stage2_outputs.zip` (splits + results) to commit here.

## One-click run on Windows (NVIDIA GPU)

Double-click `run_baseline.bat`. It installs PyTorch with CUDA into `%USERPROFILE%\cv-p16-data\venv`, downloads CCT20 there, builds the splits and runs the smoke test, baseline A and the COCO reference. Re-running skips finished steps. Output: `results/`, `data/splits/`, `run_log.txt`. Same pipeline on any OS: `python scripts/run_all.py`.

## Setup (local)

```bash
pip install -r requirements.txt
```

## Data

```bash
bash scripts/download_cct20.sh                    # images (~6 GB) + annotations into data/raw/
python -m src.data.prepare_cct20 --with-cis-test  # splits -> data/splits/
python -m src.data.tag_conditions --split test    # night / dark / small / truncated tags
```

`data/raw/` is not committed. `data/splits/` (small JSON/CSV files) is committed so everyone uses the same split.

## Baseline A and COCO reference

```bash
# untrained COCO reference (class-agnostic)
python -m src.predict --coco-reference --split val  --out runs/reference/preds_val.json
python -m src.predict --coco-reference --split test --out runs/reference/preds_test.json
python -m src.evaluate --name "COCO reference" --agnostic-only \
    --val-preds runs/reference/preds_val.json --test-preds runs/reference/preds_test.json --out results/reference

# baseline A: Faster R-CNN R50-FPN, COCO weights, standard augmentation
python -m src.train --variant A --epochs 20 --seed 0 --out runs/A_s0
python -m src.predict --ckpt runs/A_s0/best.pt --split val  --out runs/A_s0/preds_val.json
python -m src.predict --ckpt runs/A_s0/best.pt --split test --out runs/A_s0/preds_test.json
python -m src.evaluate --name "Baseline A" \
    --val-preds runs/A_s0/preds_val.json --test-preds runs/A_s0/preds_test.json --out results/baseline_A
```

The confidence threshold is chosen on val (best F1) and then fixed for test. `results/<model>/summary.md` holds mAP@0.5, mAP@[.5:.95], precision, recall, F1, false positives per empty image, image-level recall, per-class AP and per-condition recall.

## Repository layout

```
docs/        project plan sections
notebooks/   Kaggle notebook: data + reference + baseline A end to end
scripts/     data download
src/data/    split preparation, condition tagging, PyTorch dataset, box-aware transforms
src/models/  Faster R-CNN builder
src/         train.py, predict.py, evaluate.py, metrics.py
data/        raw data (ignored) and prepared splits
results/     metrics and figures (committed; summary: docs/stage2_baseline_results.md); runs/ holds checkpoints (ignored)
```
