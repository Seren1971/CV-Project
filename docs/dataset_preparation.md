# Dataset preparation

*Project P16 — Wildlife Detection in Camera-Trap Images. Owner: Sergey Peshkov*

## Dataset choice

We use **CCT20**, the benchmark subset of **Caltech Camera Traps** (Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018), distributed by LILA BC under the Community Data License Agreement (permissive).

Why this dataset fits the task:

- **Bounding boxes for every animal.** Annotations are in COCO-CameraTraps JSON (class + box per instance), so detection metrics such as mAP, precision and recall can be computed directly.
- **The failure conditions we study are built in.** Images come from motion-triggered camera traps in the American Southwest: animals hidden by vegetation, night-time infrared frames, animals partly outside the frame, and unusual poses. The dataset paper lists illumination, motion blur, small regions of interest, occlusion, camouflage and perspective as its main nuisance factors.
- **Empty frames are labelled.** Camera traps are often triggered by moving grass or branches, producing frames with no animal. These are exactly where false detections caused by vegetation appear.
- **Small enough for our resources.** 57,868 images from 20 camera locations; the resized version is about 6 GB, so all experiments fit on a single Colab/Kaggle GPU.
- **An official location-based split** already exists, which we reuse to avoid leakage.

## Selected categories and subset

- CCT20 has 15 classes plus `empty`. We drop `car` (not an animal) and keep the **animal classes with at least 100 training images**, so each category has enough boxes to train and evaluate. The final list is produced by `prepare_cct20.py` and saved to `data/splits/classes.json`.
- **Train:** all images of the selected classes. Images whose only label is a dropped class are removed.
- **Validation and test:** capped at **600 images per class**, sampled by whole sequences, to keep evaluation fast and classes less imbalanced.
- **Empty frames in validation and test:** added with the same cap of 600. They contain no animals, so any detection on them is a false positive. This lets us measure false detections caused by vegetation and background directly.
- Images with several animals keep all their boxes.

## Train / validation / test split

| Our split | Official CCT20 split | Camera locations | Purpose |
|---|---|---|---|
| train | `train` | 10 training locations | fine-tuning the detector |
| val | `cis_val` | same as train, different sequences | model selection, confidence threshold |
| test | `trans_test` | locations never seen in training | final metrics and failure analysis |
| test_cis (optional) | `cis_test` | seen locations | measures the "new location" gap |

Exact image and box counts per split and class are written to `data/splits/split_stats.csv` and will be added here after the first run.

### Leakage prevention

Camera traps record **bursts of near-identical frames** (sequences), and **each camera has a fixed background**. A random image-level split would put almost identical pictures into both train and test. We prevent this in three ways:

1. Each sequence is kept entirely inside one split.
2. Test images come only from **camera locations absent from training**, so the detector cannot rely on memorised backgrounds.
3. `prepare_cct20.py` runs an automatic check and stops if any image, sequence, or train/test location is shared.

## Preprocessing

- We use the resized "_sm" images (max 1024 px on the long side). The official boxes refer to the original resolution, so **boxes are rescaled to the actual image size** at loading time (`src/data/dataset.py`). Boxes are converted from COCO `[x, y, w, h]` to `[x1, y1, x2, y2]`.
- Input size and normalisation follow the chosen pretrained detector.
- Training augmentation, if the model is fine-tuned: horizontal flip, scale jitter, mild colour jitter, all box-aware. **No augmentation is applied to val/test.**
- Night IR frames are grayscale. They are kept and fed as 3-channel images, not removed.

## Condition tags for failure analysis

Every test image is tagged by `tag_conditions.py`, so errors can be counted per condition rather than only shown:

| Condition | How it is tagged |
|---|---|
| Darkness | automatic: IR frames are grayscale (R≈G≈B), plus mean brightness below a threshold |
| Partial visibility | automatic: a box touches the image border |
| Small / distant animal | automatic: box area below 1% of the image |
| Vegetation | partly automatic: false positives on `empty` frames; occluded animals tagged manually on inspected cases |
| Unusual pose | manual, on inspected failure cases |

## Reproducibility

```bash
bash scripts/download_cct20.sh                    # ~6 GB images + annotations
python -m src.data.prepare_cct20 --with-cis-test  # builds data/splits/
python -m src.data.tag_conditions --split test    # condition tags
```

All random sampling uses a fixed seed (42).
