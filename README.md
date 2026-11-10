# P16 — Wildlife Detection in Camera-Trap Images (Computer Vision 2026 project)

Detecting and classifying animals in camera-trap images on **unseen camera locations**.
Dataset: CCT20 subset of Caltech Camera Traps (LILA BC). Details: [docs/dataset_preparation.md](docs/dataset_preparation.md).

**Final report:** [docs/final_summary.md](docs/final_summary.md) (two-page technical summary) · **Slides:** `presentation/` · **Demo:** `scripts/demo.py`

## Results at a glance

Faster R-CNN R50-FPN (COCO weights), fine-tuned on CCT20. `test` = 5,842 images from camera locations never seen in training; thresholds chosen on val only.

| Model | Epochs | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image |
|---|---|---|---|---|---|---|---|
| COCO reference (untrained, class-agnostic) | – | 0.377 | 0.183 | 0.706 | 0.356 | 0.473 | 0.023 |
| Baseline A (standard augmentation) | 6 | 0.603 | 0.367 | 0.701 | 0.623 | 0.660 | 0.239 |
| Variant B (targeted augmentation) | 6 | 0.560 | 0.329 | 0.648 | 0.580 | 0.612 | 0.244 |
| Baseline A | 12 | 0.607 | 0.367 | 0.821 | 0.563 | 0.668 | **0.080** |
| **Variant B′ (fixed targeted augmentation)** | 12 | **0.609** | **0.368** | **0.821** | **0.569** | **0.672** | 0.115 |

Class-agnostic (find any animal): B′ 12 ep reaches **0.837 mAP@0.5** vs 0.820 for A at the same
budget, and significantly better recall on small animals (+5.2 points, 95 % CI excludes 0).

Full tables, per-class AP, per-condition recall and 95 % bootstrap intervals:
[results/comparison_A_vs_B.md](results/comparison_A_vs_B.md) (stage 3, 6 epochs) and
[results/stage4/comparison_A12_vs_Bp12.md](results/stage4/comparison_A12_vs_Bp12.md) (stage 4, 12 epochs).
Failure cases with images: `results/failure_cases/`. Report on stages:
[docs/stage2_baseline_results.md](docs/stage2_baseline_results.md), [docs/stage3_variant_B.md](docs/stage3_variant_B.md),
[docs/stage4_longer_training.md](docs/stage4_longer_training.md).

## Live demo

```bash
python scripts/demo.py --ckpt <path-to-best.pt> --images demo/images --out demo/out
```

Runs the detector on the committed sample images (or any folder of JPG/PNG), prints a
detection table with inference time, and saves annotated images + `detections.json`.
A pre-recorded 32-second video of this demo is in `demo/demo_video.mp4`
(re-record with `scripts/make_demo_video.py`, requires `imageio imageio-ffmpeg`).

**Browser interface** (drop any photo, get the detections drawn on it):

```bash
python scripts/web_demo.py --ckpt <path-to-best.pt>   # opens http://127.0.0.1:8765
```

On Windows just double-click: **`run_model.bat`** (CLI: photo/folder → annotated output)
or **`run_interface.bat`** (browser UI with drag & drop and a threshold slider).
Both auto-detect the checkpoint (`models\Bp\best.pt`, `P16_CKPT` or the usual local
paths) and the Python environment; pass a checkpoint path as the first argument to override.
Checkpoints are not stored in git (158 MB each); retrain with the commands below or use
`p16_checkpoints_A_B.zip` from the team's shared storage.

## Quick start on Kaggle (recommended for reproduction)

Open [notebooks/baseline_kaggle.ipynb](notebooks/baseline_kaggle.ipynb) on Kaggle, set **Accelerator: GPU** and **Internet: On**, then *Run All*. It downloads the data, builds the splits, runs a smoke test, the COCO reference and baseline A, and saves `stage2_outputs.zip` (splits + results) to commit here.

## One-click run on Windows (NVIDIA GPU)

Double-click `run_baseline.bat`. It installs PyTorch with CUDA into `%USERPROFILE%\cv-p16-data\venv`, downloads CCT20 there, builds the splits and runs the smoke test, baseline A and the COCO reference. Re-running skips finished steps. Output: `results/`, `data/splits/`, `run_log.txt`. Same pipeline on any OS: `python scripts/run_all.py`.

## Stage 3: variant B (targeted augmentation)

After the baseline run, double-click `run_stage3.bat` (or `python scripts/run_stage3.py`). It trains variant B with the same budget as A, evaluates A and B on `test` and `test_cis`, and writes `results/comparison_A_vs_B.md`. Description: [docs/stage3_variant_B.md](docs/stage3_variant_B.md).

## Stage 4: longer training and variant B′

`run_stage4.bat` trains baseline A and variant B′ for 12 epochs each (~2 h per run on an RTX 5070), then predicts val/test/test_cis. `python scripts/run_stage4.py` evaluates both and writes `results/stage4/`. Variant B′ fixes the diagnosed problems of B: darkening is applied only to colour (daytime) frames, grayscale is dropped, and the border crop keeps clipped boxes (≥ 15 % visible) instead of leaving unlabeled animal parts. Code: `targeted_train_v2` in `src/data/transforms.py`, `border_crop_window_v2` in `src/data/box_geometry.py`, `--variant Bp` in `src/train.py`.

## Setup (local)

```bash
pip install -r requirements.txt
```

Runtime environment and training times: [results/environment.txt](results/environment.txt).

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

## Failure analysis

```bash
python scripts/failure_analysis.py --preds runs/A12_s0/preds_test.json \
    --results results/stage4/baseline_A12 --image-dir data/raw/eccv_18_all_images_sm \
    --out results/failure_cases
```

Prints the error taxonomy (correct / class confusion / background FP / missed) and renders
representative failure and success cases with GT (green) and predicted (red/orange) boxes.

## Repository layout

```
docs/        project plan sections, stage reports, final technical summary
notebooks/   Kaggle notebook: data + reference + baseline A end to end
scripts/     data download, stage runners, demo, failure analysis
src/data/    split preparation, condition tagging, PyTorch dataset, box-aware transforms
src/models/  Faster R-CNN builder
src/         train.py, predict.py, evaluate.py, compare.py, metrics.py
demo/        sample images for the live demo (from CCT20, CDLA-permissive)
data/        raw data (ignored) and prepared splits
results/     metrics and figures (committed); runs/ holds checkpoints (ignored)
```

## Team

| Member | Contribution |
|---|---|
| Sergey Peshkov | Dataset selection and preparation, leakage-free splits, condition tagging, training/evaluation pipeline, experiment runs (A, B, B′), demo, repository maintenance |
| Arina Nikolaeva | Model and implementation specification: Faster R-CNN R50-FPN choice, pretrained weights, software stack and GPU pipeline |
| Anton Kropotov | Experimental design: hypotheses, run configuration, metrics protocol, targeted-augmentation idea, error-analysis protocol |
| Arseniy Laptev | Project scope and documentation: research question, boundaries, plan assembly, consistency checks |
