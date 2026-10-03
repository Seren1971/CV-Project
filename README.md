# Animal recognition in camera-trap images (Computer Vision 2026 project)

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

## Repository layout

```
docs/       project plan sections
scripts/    data download
src/data/   split preparation, condition tagging, PyTorch datasets
data/       raw data (ignored) and prepared splits
```
