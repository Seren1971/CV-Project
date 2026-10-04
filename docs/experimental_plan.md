# Experimental plan

*Project P16: Wildlife Detection in Camera-Trap Images.*

We want to know how well a detector finds animals in CCT20, and why it fails when it does. We look at four hard cases: animals hidden by vegetation, dark (night) images, animals cut off by the image border, and unusual poses. Data, splits and condition tags are described in [dataset_preparation.md](dataset_preparation.md).

## Models

We compare two models and add one untrained reference.

- **Reference.** A Faster R-CNN (ResNet-50 FPN) with COCO weights and no training on our data. All animal classes are merged into one "animal" class. This shows what a generic detector can do without help.
- **Baseline (A).** The same Faster R-CNN, fine-tuned on the CCT20 training split with standard augmentation: horizontal flip, scale jitter and mild colour jitter.
- **Variant (B).** The same model, hyper-parameters and seed as A. Only the augmentation changes. B adds transformations that imitate the hard cases:
  - random darkening, gamma change and grayscale, to imitate night and infrared frames;
  - random crops that cut boxes at the image border (a box is kept if at least 40% of it stays visible), to imitate partly visible animals;
  - random erasing over part of a box (never the whole box), to imitate grass and branches in front of the animal;
  - wider scale jitter, for small and distant animals.

We chose this pair because it changes one thing. If B does better on night images, the cause is the augmentation and nothing else. A different architecture (for example YOLOv8 or RetinaNet) could be a third model if time allows, with the same protocol.

## Experimental setup

- **Data.** Train, val and test come from `data/splits/`. The test set is the official `trans_test`, which contains camera locations never seen in training. We can also report `test_cis` (seen locations) to measure the gap between new and known locations.
- **Classes.** The animal classes in `data/splits/classes.json`. We evaluate in two ways: class-agnostic (is there an animal, and where?) and per class.
- **Training.** The table below lists what is the same and what differs between the runs. We plan three seeds per model and report mean and standard deviation. If compute only allows one seed, we treat differences below about 1 mAP point as noise.
- **Model selection.** We pick the epoch with the best mAP@0.5 on val. We also pick the confidence threshold on val (the one that maximises F1) and keep it fixed on test.
- **Test set use.** We never tune anything on test. Each final model is evaluated on it once.
- **Hardware.** One Colab or Kaggle GPU. The split and seeds are fixed (seed 42), so runs can be repeated.

### Run configurations

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

## Metrics

On the full test set:

- mAP@0.5 and mAP@[0.5:0.95] (COCO protocol), per class and averaged;
- precision, recall and F1 at the fixed threshold, with IoU ≥ 0.5;
- false positives per image on empty frames, which measures false alarms from moving grass and branches;
- image-level recall: the share of animal images where at least one box is correct. This is the number that matters if the detector is used to filter camera-trap data.

We then compute the same metrics on each condition subset (using the tags from `tag_conditions.py`) and compare it with the rest of the test set.

## Error analysis

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

## Hypotheses

We state them before running anything, so the results can show that we were wrong.

- **H1: fine-tuning beats the reference.** A has clearly higher mAP@0.5 than the untrained reference, mostly because CCT20 contains species and poses that COCO does not cover well. If it does not, the training setup is the problem.
- **H2: night images are harder than day images for A.** Recall on `is_night` is lower than on the rest of the test set. B reduces this gap, because it saw darkened and grayscale images in training.
- **H3: truncated animals are found less often and localised worse.** A has lower recall and lower mean IoU on `truncated` images. B improves both, since border crops give it examples of cut-off animals. We expect the localisation errors to shrink more than the misses.
- **H4: vegetation false alarms stay about the same.** False positives per empty image differ little between A and B, because empty frames are not changed by our augmentation. Erasing patches may even slightly raise false alarms. If this holds, hard negative training is the next step.
- **H5: occlusion and unusual poses hurt both models.** Occluded animals have lower recall than clear ones in A and B. B helps a little at most, since erasing patches only roughly imitate real grass. Pose failures should be spread over many different poses, with no single dominant type.
- **H6: new locations are harder.** Both models score lower on `test` (new locations) than on `test_cis` (seen locations). The gap is similar for A and B, since augmentation does not change the background.
- **Overall.** B gives a small gain on overall mAP (about 1 to 3 points) and larger gains on the night and truncated subsets. A gain only on the full test set, with no change on the subsets, would not support our explanation.

If a hypothesis fails, we report that and look at the error counts to find the reason.

## Risks
- Manual tags (occlusion, pose) are subjective. We will write a short tagging guide, and two people will tag the same sample of 50 images so we can check how often they agree.
- The resized `_sm` images lose detail on small animals. We will state this as a limitation.
- Compute limits the number of seeds, which limits how much we can trust small differences.
