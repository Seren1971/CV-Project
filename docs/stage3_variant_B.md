# Stage 3: second approach — targeted augmentation (variant B)

*P16 — Wildlife Detection in Camera-Trap Images.*

## New approach

Variant B keeps the baseline detector unchanged: Faster R-CNN R50-FPN with COCO weights, fine-tuned on the same CCT20 training split. Only the **training augmentation** changes. In addition to the baseline flip and colour jitter, B applies four transforms that imitate the conditions behind the baseline's errors, each with probability 0.3:

| Transform | What it does | Condition it imitates |
|---|---|---|
| Border crop | Crops the image so that one edge cuts through an animal and 40–80% of it stays visible. Boxes are clipped; a box is kept only if ≥ 40% of its area remains. | Animals partly outside the frame |
| Partial erase | Covers 20–50% of one animal's box (never the whole box) with a patch copied from another part of the same image. | Vegetation or rocks in front of the animal |
| Darken | Brightness × 0.3–0.7, then gamma 1.2–2.0. | Night and under-exposed frames |
| Grayscale | Converts the image to 3-channel grayscale. | Infrared night frames |
| Wider scale jitter | Shorter side sampled from 480–960 px instead of 640–896. | Small and distant animals |

Examples of each transform with the resulting boxes: `results/variant_B/augmentation_examples.png`.

## Motivation

The baseline (stage 2) showed where the detector fails on new camera locations:

- **Partial visibility** is the clearest weakness: recall is 0.39 on animals cut by the image border versus 0.67 on the rest.
- **Small animals:** recall is 0.43 versus 0.68.
- **Darkness** was the main expected difficulty, and the untrained COCO model fails badly at night (recall 0.22). After fine-tuning, the baseline's night recall (0.70) was not lower than its day recall (0.54). B tests whether extra darkening and grayscale still help, or whether the remaining day/night difference comes from something else, such as the species mix.

These cases are under-represented in the training images. Augmentation creates more of them without new data or a new architecture. Because everything else stays fixed, any change in the results can be attributed to the augmentation package as a whole.

**Expected improvement** (hypotheses H2, H3 and H4 in the experimental plan): higher recall and IoU on truncated animals, higher recall on small animals, and a similar or slightly better result at night. Overall mAP is expected to change little (+1 to 3 points). False alarms on empty frames should not change much, because empty frames are not targeted. Partial erasing could slightly increase them.

## Implementation

Differences from baseline A:

| | Baseline A | Variant B |
|---|---|---|
| Model, weights, classes | Faster R-CNN R50-FPN, COCO_V1, 11 classes | same |
| Optimiser, lr schedule, batch, epochs, seed | SGD 0.01, warm-up + cosine, batch 4, 6 epochs, seed 0 | same |
| Flip, colour jitter | yes | yes |
| Border crop / partial erase / darken / grayscale | no | yes, p = 0.3 each |
| Scale jitter (shorter side) | 640–896 | 480–960 |
| Model selection, threshold | best val mAP@0.5; threshold = best val F1 | same |

Code: `src/data/box_geometry.py` (crop and erase geometry, pure Python, property-tested on 20,000 random cases), `src/data/transforms.py` (`targeted_train`), `--variant B` in `src/train.py`. The comparison (`src/compare.py`) evaluates both models on the same test images and reports differences with 95% bootstrap intervals, resampling whole sequences. Both models are also evaluated on `test_cis` (seen camera locations) to measure the new-location gap.

Reproduce: `run_stage3.bat` (Windows) or `python scripts/run_stage3.py`.

## Initial results

*To be filled in after the run.*
