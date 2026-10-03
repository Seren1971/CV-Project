# Dataset preparation

*Owner: Sergey Peshkov*

## Dataset choice

We use **CCT20**, the benchmark subset of **Caltech Camera Traps** (Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018), distributed by LILA BC under the Community Data License Agreement (permissive).

Why this dataset fits our topic:

- **Real failure conditions are built in.** The images come from motion-triggered camera traps in the American Southwest, so they naturally contain the conditions we want to analyse: animals hidden by vegetation, night-time infrared frames, animals partly outside the frame or very close to the lens, and unusual poses. The dataset paper itself lists illumination, motion blur, small regions of interest, occlusion, camouflage and perspective as the main nuisance factors.
- **Labels for both classification and detection.** Each animal has a class label and a bounding box (COCO-CameraTraps JSON), so the same split works whichever model the Implementation plan selects.
- **Small enough for our resources.** CCT20 has 57,868 images from 20 camera locations; the resized version is about 6 GB, so all experiments fit a single GPU (Colab/Kaggle) without large-scale training.
- **An official leakage-free split by location** already exists, which we reuse (see below).

## Classes and subset

- CCT20 contains 15 classes plus "empty". We **drop `empty` and `car`** (not animals) and keep the animal classes that have **at least 100 training images**, so every class can be learned and evaluated meaningfully. The exact class list is produced by `prepare_cct20.py` and saved to `data/splits/classes.json`.
- **Training set:** all images of the kept classes (no subsampling).
- **Validation and test sets:** capped at **600 images per class**, sampled by whole sequences, to keep evaluation time short and classes less imbalanced. The cap is a script parameter (`--cap-eval`).
- Images with several animals keep all boxes; for classification the image label is the class of the largest box.

## Train / validation / test split

| Our split | Official CCT20 split | Camera locations | Purpose |
|---|---|---|---|
| train | `train` | 10 training locations | model training / fine-tuning |
| val | `cis_val` | same as train, different sequences | model selection, early stopping, thresholds |
| test | `trans_test` | locations never seen in training | final metrics and failure analysis |
| test_cis (optional) | `cis_test` | seen locations | measures the "new location" generalisation gap |

The real image counts per split and class are written to `data/splits/split_stats.csv` by the preparation script and will be copied here after the first run.

### Leakage prevention

Camera traps record **bursts of near-identical frames** (sequences) and the **background of each camera is fixed**, so a random image-level split would leak: the model would see almost the same picture in train and test. We prevent this by:

1. keeping every sequence entirely inside one split;
2. using test images only from **camera locations absent from training**, so the model cannot rely on memorised backgrounds;
3. an automatic check in `prepare_cct20.py` that fails if any image, sequence, or train–test location is shared.

## Preprocessing

- We use the resized "_sm" images (max 1024 px on the long side). The official boxes refer to the original resolution, so they are **rescaled to the actual image size** when loaded (`src/data/dataset.py`).
- Model-specific resizing and normalisation follow the chosen pretrained model (e.g. ImageNet mean/std, 224 px for classifiers, native input size for detectors).
- Training augmentation (if the model is fine-tuned): random resized crop, horizontal flip, mild colour jitter. **No augmentation on val/test.** Grayscale night images are kept as 3-channel images, not removed.

## Condition tags for failure analysis

To analyse errors by condition, every test image is tagged by `tag_conditions.py`:

| Condition | How it is tagged |
|---|---|
| Darkness / night | automatic: IR frames are grayscale (R≈G≈B); plus mean brightness below a threshold |
| Partial visibility | automatic: bounding box touches the image border |
| Small / distant animal | automatic: box area < 1% of the image |
| Vegetation occlusion | manual, on inspected failure cases |
| Unusual pose | manual, on inspected failure cases |

This lets the Experimental plan report metrics per condition (e.g. accuracy on night vs day images) and pick representative failure cases instead of random ones.

## Reproducibility

```bash
bash scripts/download_cct20.sh                    # ~6 GB images + annotations
python -m src.data.prepare_cct20 --with-cis-test  # builds data/splits/
python -m src.data.tag_conditions --split test    # condition tags for failure analysis
```

All random sampling uses a fixed seed (42).
