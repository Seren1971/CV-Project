# P16: Wildlife Detection in Camera-Trap Images

## Project plan

This planning document combines the team sections. Dataset and experiment sections reproduce their current repository versions; the implementation section consolidates the supplied team outline. Numerical results are not yet available.

## Project scope

### Objective

Detect and classify selected animal categories in CCT20 camera-trap images using a pretrained Faster R-CNN detector, and analyse false detections and missed animals caused by vegetation, darkness, partial visibility and unusual poses.

### Research question

Does fine-tuning Faster R-CNN with augmentations that simulate difficult camera-trap conditions improve detection performance over fine-tuning with standard augmentations, particularly on dark and partially visible animals at previously unseen camera locations?

### Scope and comparison

The main task is object detection: predict an animal category, bounding box and confidence for each detected instance. We use the selected CCT20 animal classes and the official split policy described in [Dataset preparation](dataset_preparation.md). Empty validation and test frames are retained to measure false alarms.

The main comparison is between two Faster R-CNN ResNet-50 FPN models initialised with COCO-pretrained weights from torchvision:

- **Baseline A:** fine-tuning with standard augmentation.
- **Variant B:** the same training setup with additional augmentation for darkness, partial visibility, occlusion and small animals.

The architecture, splits and training budget are held constant. This comparison evaluates the augmentation package as a whole; it does not isolate the effect of each individual transformation. A COCO-pretrained detector without CCT20 fine-tuning is an additional class-agnostic reference, subject to an explicit COCO-to-animal label mapping.

Checkpoints and confidence thresholds are selected using validation data. The test set is reserved for final evaluation. We report mAP@0.5, mAP@[0.5:0.95], precision, recall and F1, together with false positives per empty image and condition-specific error analysis. At least three representative failures will be explained. A concise sensitivity experiment will also be included; its exact protocol must be agreed with the experiment owner before execution.

### Boundaries and limitations

The project uses an existing public dataset, pretrained weights and a single Colab or Kaggle GPU. Training a detector from scratch, collecting a new dataset, deployment, video tracking and additional detector architectures are outside the core scope. Evaluation on the optional `test_cis` split and repeated training seeds depend on the available compute budget.

Automatic condition tags are proxies: grayscale is not proof of night-time capture, border-touching boxes cover truncation rather than every form of partial visibility, and false positives on empty frames are background errors whose cause must be inspected before attributing them to vegetation. Occlusion and unusual poses require manual inspection. Conclusions will describe observed results and limitations rather than assume that the proposed augmentation improves performance.

### Expected outputs

- Reproducible code or a notebook with dataset download instructions, environment information and exact run commands.
- Saved evaluation metrics, predictions and a compact comparison table or figure.
- Failure examples with explanations of likely causes.
- A two-page technical summary with individual contributions.
- A five-minute demonstration with at most three supporting slides, followed by individual questions.

This document describes planned work. Training, evaluation and numerical results are not yet claimed.

---

## Dataset preparation

*Project P16 — Wildlife Detection in Camera-Trap Images. Owner: Sergey Peshkov*

### Dataset choice

We use **CCT20**, the benchmark subset of **Caltech Camera Traps** (Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018), distributed by LILA BC under the Community Data License Agreement (permissive).

Why this dataset fits the task:

- **Bounding boxes for every animal.** Annotations are in COCO-CameraTraps JSON (class + box per instance), so detection metrics such as mAP, precision and recall can be computed directly.
- **The failure conditions we study are built in.** Images come from motion-triggered camera traps in the American Southwest: animals hidden by vegetation, night-time infrared frames, animals partly outside the frame, and unusual poses. The dataset paper lists illumination, motion blur, small regions of interest, occlusion, camouflage and perspective as its main nuisance factors.
- **Empty frames are labelled.** Camera traps are often triggered by moving grass or branches, producing frames with no animal. These are exactly where false detections caused by vegetation appear.
- **Small enough for our resources.** 57,868 images from 20 camera locations; the resized version is about 6 GB, so all experiments fit on a single Colab/Kaggle GPU.
- **An official location-based split** already exists, which we reuse to avoid leakage.

### Selected categories and subset

- CCT20 has 15 classes plus `empty`. We drop `car` (not an animal) and keep the **animal classes with at least 100 training images**, so each category has enough boxes to train and evaluate. The final list is produced by `prepare_cct20.py` and saved to `data/splits/classes.json`.
- **Train:** all images of the selected classes. Images whose only label is a dropped class are removed.
- **Validation and test:** capped at **600 images per class**, sampled by whole sequences, to keep evaluation fast and classes less imbalanced.
- **Empty frames in validation and test:** added with the same cap of 600. They contain no animals, so any detection on them is a false positive. This lets us measure false detections caused by vegetation and background directly.
- Images with several animals keep all their boxes.

### Train / validation / test split

| Our split | Official CCT20 split | Camera locations | Purpose |
|---|---|---|---|
| train | `train` | 10 training locations | fine-tuning the detector |
| val | `cis_val` | same as train, different sequences | model selection, confidence threshold |
| test | `trans_test` | locations never seen in training | final metrics and failure analysis |
| test_cis (optional) | `cis_test` | seen locations | measures the "new location" gap |

Exact image and box counts per split and class are written to `data/splits/split_stats.csv` and will be added here after the first run.

#### Leakage prevention

Camera traps record **bursts of near-identical frames** (sequences), and **each camera has a fixed background**. A random image-level split would put almost identical pictures into both train and test. We prevent this in three ways:

1. Each sequence is kept entirely inside one split.
2. Test images come only from **camera locations absent from training**, so the detector cannot rely on memorised backgrounds.
3. `prepare_cct20.py` runs an automatic check and stops if any image, sequence, or train/test location is shared.

### Preprocessing

- We use the resized "_sm" images (max 1024 px on the long side). The official boxes refer to the original resolution, so **boxes are rescaled to the actual image size** at loading time (`src/data/dataset.py`). Boxes are converted from COCO `[x, y, w, h]` to `[x1, y1, x2, y2]`.
- Input size and normalisation follow the chosen pretrained detector.
- Training augmentation, if the model is fine-tuned: horizontal flip, scale jitter, mild colour jitter, all box-aware. **No augmentation is applied to val/test.**
- Night IR frames are grayscale. They are kept and fed as 3-channel images, not removed.

### Condition tags for failure analysis

Every test image is tagged by `tag_conditions.py`, so errors can be counted per condition rather than only shown:

| Condition | How it is tagged |
|---|---|
| Darkness | automatic: IR frames are grayscale (R≈G≈B), plus mean brightness below a threshold |
| Partial visibility | automatic: a box touches the image border |
| Small / distant animal | automatic: box area below 1% of the image |
| Vegetation | partly automatic: false positives on `empty` frames; occluded animals tagged manually on inspected cases |
| Unusual pose | manual, on inspected failure cases |

### Reproducibility

```bash
bash scripts/download_cct20.sh                    # ~6 GB images + annotations
python -m src.data.prepare_cct20 --with-cis-test  # builds data/splits/
python -m src.data.tag_conditions --split test    # condition tags
```

All random sampling uses a fixed seed (42).

---

## Implementation plan

*Owner: Arina Nikolaeva. Consolidated from the implementation outline supplied by the team.*

### Model and environment

Use Faster R-CNN with a ResNet-50 FPN backbone and COCO-pretrained weights from torchvision. The software stack is Python, PyTorch, torchvision, Pillow and NumPy. Training and evaluation run on a GPU in Google Colab or Kaggle.

### Pipeline

1. Download CCT20 images and annotations and prepare the agreed splits.
2. Load images and bounding boxes with the custom PyTorch dataset.
3. Adapt the detector output to the selected CCT20 categories and fine-tune it on the training split.
4. Run the baseline and augmentation variant according to the experimental plan.
5. Select a checkpoint and confidence threshold using validation data.
6. Evaluate the selected models on test data and save metrics, predictions and failure examples.

The exact torchvision model constructor, weight identifier, package versions and input transforms will be recorded by the implementation owner before execution. Images and boxes must be transformed together during spatial augmentation. The unchanged COCO reference uses a separate class-agnostic evaluation mapping rather than the newly initialised CCT20 prediction head.

---

## Experimental plan

*Project P16: Wildlife Detection in Camera-Trap Images.*

We want to know how well a detector finds animals in CCT20, and why it fails when it does. We look at four hard cases: animals hidden by vegetation, dark (night) images, animals cut off by the image border, and unusual poses. Data, splits and condition tags are described in [dataset_preparation.md](dataset_preparation.md).

### Models

We compare two models and add one untrained reference.

- **Reference.** A Faster R-CNN (ResNet-50 FPN) with COCO weights and no training on our data. All animal classes are merged into one "animal" class. This shows what a generic detector can do without help.
- **Baseline (A).** The same Faster R-CNN, fine-tuned on the CCT20 training split with standard augmentation: horizontal flip, scale jitter and mild colour jitter.
- **Variant (B).** The same model, hyper-parameters and seed as A. Only the augmentation changes. B adds transformations that imitate the hard cases:
  - random darkening, gamma change and grayscale, to imitate night and infrared frames;
  - random crops that cut boxes at the image border (a box is kept if at least 40% of it stays visible), to imitate partly visible animals;
  - random erasing over part of a box (never the whole box), to imitate grass and branches in front of the animal;
  - wider scale jitter, for small and distant animals.

We chose this pair because it changes one thing. If B does better on night images, the cause is the augmentation and nothing else. A different architecture (for example YOLOv8 or RetinaNet) could be a third model if time allows, with the same protocol.

### Experimental setup

- **Data.** Train, val and test come from `data/splits/`. The test set is the official `trans_test`, which contains camera locations never seen in training. We can also report `test_cis` (seen locations) to measure the gap between new and known locations.
- **Classes.** The animal classes in `data/splits/classes.json`. We evaluate in two ways: class-agnostic (is there an animal, and where?) and per class.
- **Training.** The table below lists what is the same and what differs between the runs. We plan three seeds per model and report mean and standard deviation. If compute only allows one seed, we treat differences below about 1 mAP point as noise.
- **Model selection.** We pick the epoch with the best mAP@0.5 on val. We also pick the confidence threshold on val (the one that maximises F1) and keep it fixed on test.
- **Test set use.** We never tune anything on test. Each final model is evaluated on it once.
- **Hardware.** One Colab or Kaggle GPU. The split and seeds are fixed (seed 42), so runs can be repeated.

#### Run configurations

| Setting | Reference | Baseline (A) | Variant (B) |
|---|---|---|---|
| Model | Faster R-CNN R50-FPN | same | same |
| Initial weights | COCO | COCO | COCO |
| Trained on CCT20 | no | yes | yes |
| Classes | 1 (animal), merged from COCO animal classes | CCT20 animal classes | CCT20 animal classes |
| Epochs | n/a | 20 | 20 |
| Optimiser | n/a | SGD, momentum 0.9, weight decay 1e-4 | same |
| Learning rate | n/a | 0.01, cosine decay | same |
| Batch size | n/a | 4 to 8 | same |
| Input size | long side 1024 | long side 1024 | long side 1024 |
| Scale jitter (shorter side) | none | 640 to 896 | 480 to 960 |
| Flip, colour jitter | none | yes | yes |
| Darkening, grayscale | none | no | yes, p = 0.3 each |
| Border crop | none | no | yes, p = 0.3 |
| Partial erasing | none | no | yes, p = 0.3 |
| Seeds | 1 | 3 | 3 |
| Model selection | n/a | best val mAP@0.5 | best val mAP@0.5 |

All values are starting points. If a run does not fit into the available GPU time, we reduce epochs for A and B together.

### Metrics

On the full test set:

- mAP@0.5 and mAP@[0.5:0.95] (COCO protocol), per class and averaged;
- precision, recall and F1 at the fixed threshold, with IoU ≥ 0.5;
- false positives per image on empty frames, which measures false alarms from moving grass and branches;
- image-level recall: the share of animal images where at least one box is correct. This is the number that matters if the detector is used to filter camera-trap data.

We then compute the same metrics on each condition subset (using the tags from `tag_conditions.py`) and compare it with the rest of the test set.

### Error analysis

Each detection above the threshold gets one label, checked in this order:

1. **Correct:** IoU ≥ 0.5 with a ground-truth box, right class.
2. **Duplicate:** IoU ≥ 0.5 with a ground-truth box of the right class that another detection already matched.
3. **Localisation error:** right class, IoU between 0.1 and 0.5.
4. **Class confusion:** IoU ≥ 0.5, wrong class.
5. **Background false positive:** IoU < 0.1 with every ground-truth box. On empty frames, these are the vegetation false alarms.
6. **Missed animal:** a ground-truth box that no detection matches.

We count these per model and per condition. The four conditions are handled as follows.

| Condition | Test subset | What we measure |
|---|---|---|
| Vegetation | (a) all empty frames; (b) animal images where we tag an occluder by hand | (a) false positives per image and the confidence of the false boxes; (b) recall of occluded vs. clearly visible animals |
| Darkness | `is_night` and `is_dark` tags | mAP and recall at night vs. day, and whether B narrows the gap |
| Partial visibility | `truncated` tag (box touches the border) | recall and mean IoU vs. non-truncated animals, and the share of localisation errors |
| Unusual poses | failure cases tagged by hand (lying, curled up, seen from behind, jumping) | how many failures fall into each pose type, with example images; there is no automatic metric for this |

We also report recall by box size (the `is_small` tag), because small animals are hard on their own and can distort the other results.

Procedure:

1. Run A and B on the test set and save every detection.
2. Compute overall and per-condition metrics. For each condition, report the gap between the subset and the rest, and whether B makes it smaller.
3. Look at the 50 most confident false positives and all missed animals in each condition. Tag them by hand (type of occluder, type of pose).
4. Show side-by-side examples of A and B, successes and failures.
5. Conditions overlap: a night image can also be truncated. We report a few combined subsets (for example night and truncated) so that we do not blame one factor for the whole drop.

Some subsets will be small, so we give 95% bootstrap confidence intervals for precision, recall and image-level recall (mAP is a point estimate only). We resample by sequence, not by image, because frames in one burst are nearly identical. We call a difference between A and B real only if the intervals do not overlap.

### Hypotheses

We state them before running anything, so the results can show that we were wrong.

- **H1: fine-tuning beats the reference.** A has clearly higher mAP@0.5 than the untrained reference, mostly because CCT20 contains species and poses that COCO does not cover well. If it does not, the training setup is the problem.
- **H2: night images are harder than day images for A.** Recall on `is_night` is lower than on the rest of the test set. B reduces this gap, because it saw darkened and grayscale images in training.
- **H3: truncated animals are found less often and localised worse.** A has lower recall and lower mean IoU on `truncated` images. B improves both, since border crops give it examples of cut-off animals. We expect the localisation errors to shrink more than the misses.
- **H4: vegetation false alarms stay about the same.** False positives per empty image differ little between A and B, because empty frames are not changed by our augmentation. Erasing patches may even slightly raise false alarms. If this holds, hard negative training is the next step.
- **H5: occlusion and unusual poses hurt both models.** Occluded animals have lower recall than clear ones in A and B. B helps a little at most, since erasing patches only roughly imitate real grass. Pose failures should be spread over many different poses, with no single dominant type.
- **H6: new locations are harder.** Both models score lower on `test` (new locations) than on `test_cis` (seen locations). The gap is similar for A and B, since augmentation does not change the background.
- **Overall.** B gives a small gain on overall mAP (about 1 to 3 points) and larger gains on the night and truncated subsets. A gain only on the full test set, with no change on the subsets, would not support our explanation.

If a hypothesis fails, we report that and look at the error counts to find the reason.

### Risks
- Manual tags (occlusion, pose) are subjective. We will write a short tagging guide, and two people will tag the same sample of 50 images so we can check how often they agree.
- The resized `_sm` images lose detail on small animals. We will state this as a limitation.
- Compute limits the number of seeds, which limits how much we can trust small differences.

---

## Team responsibilities

| Member | Responsibility | Planning contribution |
|---|---|---|
| Sergey Peshkov | Dataset preparation | Select the dataset and animal categories; inspect annotations; define preprocessing and train/validation/test splits; report subset sizes and leakage checks. |
| Arina Nikolaeva | Model and implementation | Specify Faster R-CNN ResNet-50 FPN and pretrained weights; describe the software stack, GPU resources and implementation pipeline. |
| Anton Kropotov | Experiments and evaluation | Define baseline and comparison, training conditions, metrics, sensitivity experiment and failure-analysis protocol. |
| avlaptev | Project scope and document assembly | Formulate the objective, research question and boundaries; document responsibilities; combine the sections and check consistency. |

Each member reviews their section after integration. Model and experiment owners agree on the class mapping, augmentation settings and evaluation protocol before implementation. The dataset owner supplies the final class list and split statistics after preparation. The document assembly owner records these decisions in the shared plan.

These are planning responsibilities, not claims of completed implementation or experiments. Actual contributions will be updated in the final technical summary. Every member prepares to explain the overall model, metrics, results and failure cases during the individual understanding check.

## Integration checks before execution

- **Dataset:** publish the final category list and actual split sizes after running preparation. Whole-sequence sampling may exceed the nominal per-class cap, so report actual counts.
- **Implementation:** confirm the exact model constructor, weights and versions, image-size policy and identical training budgets for A and B.
- **Reference:** compare the unchanged COCO model with fine-tuned models only under the same class-agnostic animal metric and matching protocol. Its COCO categories do not directly provide CCT20 species predictions.
- **Sensitivity experiment:** the experiment owner must specify one concise experiment. A validation-only confidence-threshold sweep with fixed predictions is a possible low-cost option; this is a proposal, not an agreed protocol.
- **Seeds:** distinguish the fixed data-sampling seed (42) from the list of training seeds; record the list if three runs are used.
- **Error analysis:** use confidence-ordered, one-to-one ground-truth matching; define how every unmatched prediction is counted. Empty frames alone do not establish vegetation as the cause of a false alarm.
- **Interpretation:** evaluate the augmentation package as a whole. A gain cannot be assigned to one transformation without an ablation. Avoid a universal 1 mAP noise cutoff, treating non-overlapping intervals as the sole significance criterion, or assuming that a failed hypothesis proves a training defect.
- **Final summary:** replace planned settings with executed settings and results, include actual contributions and verify the course submission checklist.
