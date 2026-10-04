# P16 — Wildlife Detection in Camera-Trap Images (Computer Vision 2026 project)

Dataset: CCT20 subset of Caltech Camera Traps (LILA BC). Details: [docs/dataset_preparation.md](docs/dataset_preparation.md).

## Setup

```bash
pip install -r requirements.txt
```

## Data

```bash
bash scripts/download_cct20.sh                    # images (~6 GB) + annotations into data/raw/
python -m src.data.prepare_cct20 --with-cis-test  # splits -> data/splits/
python -m src.data.tag_conditions --split test    # night / dark / small / truncated tags
```

`data/raw/` is not committed. `data/splits/` (small JSON/CSV files) can be committed so everyone uses the same split.

## Experiments

Plan: [docs/experimental_plan.md](docs/experimental_plan.md).

```bash
python -m src.train --variant A --seed 0 --out runs/A_s0     # baseline
python -m src.train --variant B --seed 0 --out runs/B_s0     # targeted augmentation
python -m src.evaluate predict --ckpt runs/A_s0/best.pt --split val  --out runs/A_s0/preds_val.json
python -m src.evaluate threshold --preds runs/A_s0/preds_val.json --split val
python -m src.evaluate predict --ckpt runs/A_s0/best.pt --split test --out runs/A_s0/preds_test.json
python -m src.evaluate report --preds runs/A_s0/preds_test.json --split test --threshold 0.5 --out runs/A_s0/report_test.csv
```

Optional manual tags (vegetation occlusion, pose) go in `data/splits/test_manual_tags.csv` with columns `image_id,occluded,pose`; `report` picks them up automatically.

## Repository layout

```
docs/       project plan sections
scripts/    data download
src/data/   split preparation, condition tagging, PyTorch datasets
src/        augment.py (variants A/B), train.py, evaluate.py (metrics, error analysis)
data/       raw data (ignored) and prepared splits
```
