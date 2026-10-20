# Stage 3: second approach — targeted augmentation (variant B)

*P16 — Wildlife Detection in Camera-Trap Images. Code and results: https://github.com/Seren1971/CV-Project*

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

- **Partial visibility** is the clearest weakness: in images that contain an animal cut by the image border, recall is 0.39, versus 0.67 in the other images.
- **Small animals:** in images that contain a small animal (box < 1% of the image), recall is 0.43, versus 0.68.
- **Darkness** was the main expected difficulty, and the untrained COCO model fails badly at night (recall 0.22). After fine-tuning, the baseline's night recall (0.70) was not lower than its day recall (0.54). B tests whether extra darkening and grayscale still help, or whether the remaining day/night difference comes from something else, such as the species mix.

Small animals are less common in training than in testing: 11% of training boxes versus 20% of test boxes are smaller than 1% of the image. Truncated animals are about as common in both (14% vs 16%), so for them the aim is robustness rather than more examples. Augmentation produces such cases without new data or a new architecture. Because everything else stays fixed, any change in the results can be attributed to the augmentation package as a whole, but not to one specific transform.

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

Run on 19 Oct 2026, RTX 5070, 6 epochs, seed 0, 59 minutes of training (A: 55 minutes). Full tables: `results/comparison_A_vs_B.md`. Each model uses its own confidence threshold chosen on val (best F1): per class 0.80 for A and 0.75 for B; class-agnostic 0.70 for both. Differences are B − A with 95% bootstrap intervals over sequences, paired on the same test images. Intervals are computed for precision, recall and false positives per empty frame, not for mAP.

### Overall

| Test set | Model | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image |
|---|---|---|---|---|---|---|---|
| test (new cameras), per class | A | 0.603 | 0.367 | 0.701 | 0.623 | 0.660 | 0.239 |
| | B | 0.560 | 0.329 | 0.648 | 0.580 | 0.612 | 0.244 |
| test, class-agnostic | A | 0.827 | 0.501 | 0.771 | 0.797 | 0.784 | 0.349 |
| | B | 0.804 | 0.468 | 0.759 | 0.748 | 0.754 | 0.302 |
| test_cis (seen cameras), per class | A | 0.862 | 0.537 | 0.866 | 0.809 | 0.837 | 0.100 |
| | B | 0.840 | 0.509 | 0.853 | 0.771 | 0.810 | 0.075 |

On `test`, per class: recall −0.044 [−0.056, −0.033], precision −0.052 [−0.068, −0.037]; both intervals exclude 0. False positives per empty frame: +0.005 [−0.056, +0.066], which includes 0.

### Target conditions (recall on `test`, per class)

These subsets are **image-level**: an image is selected if it contains at least one animal with the condition, and recall is computed over all annotated animals in the selected images. They therefore describe images, not individual small or truncated animals.

| Images containing… | Animals in these images | Recall A | Recall B | B − A [95% CI] | Recall on other images: A → B |
|---|---|---|---|---|---|
| a small animal | 1,176 | 0.430 | 0.466 | +0.036 [+0.013, +0.061] | 0.675 → 0.610 |
| a truncated animal | 934 | 0.389 | 0.357 | −0.032 [−0.062, −0.005] | 0.671 → 0.625 |
| night (IR frames) | 2,931 | 0.696 | 0.653 | −0.043 [−0.058, −0.027] | 0.544 → 0.500 |
| empty frames: FP / image | 602 frames | 0.239 | 0.244 | +0.005 [−0.056, +0.066] | – |

For images with small animals, the gap to the other images narrowed from 0.245 to 0.144. This happened partly because recall improved in images containing small animals (0.430 → 0.466) and partly because recall dropped on the remaining images (0.675 → 0.610). The narrower gap should therefore not be read only as an improvement on the hard cases.

The largest per-class drops in AP@0.5 are dog (−0.154), bird (−0.110) and bobcat (−0.069; night recall 0.50 → 0.28). Rabbit recall improved both at night (0.66 → 0.74) and by day (0.63 → 0.72).

### Comparison with the baseline

**With the same 6-epoch budget, variant B does not improve on the baseline.** It has lower overall mAP@0.5 (−4.3 points per class, −2.3 class-agnostic). The paired sequence-bootstrap intervals for the per-class precision and recall differences exclude zero. For mAP, only point estimates are available. B is also lower on `test_cis`. Of the targeted subsets, B is higher only on images containing small animals. Each outcome below describes the whole augmentation package: the run does not show which transform caused it.

| Hypothesis | Outcome |
|---|---|
| H2: night is harder; B narrows the gap | **Not supported.** Night images are not harder for A, and B's recall on them is lower than A's. |
| H3: B improves truncated animals | **Not supported.** Recall on images with truncated animals is slightly lower for B. |
| Small animals improve | **Partly consistent.** Recall on images with small animals is +3.6 points, but recall on other images fell. This may be related to the wider scale jitter; an ablation is needed to isolate its contribution. |
| H4: empty-frame false alarms unchanged | **No clear evidence of a change** on `test` (interval includes 0; this does not prove equality). On `test_cis`, B has 0.075 per empty frame versus 0.100 for A, without an interval. |
| H6: new locations are harder | **Supported** for both models. Per-class mAP@0.5 on seen vs new cameras: A 0.86 vs 0.60, B 0.84 vs 0.56. Class-agnostic: A 0.91 vs 0.83, B 0.90 vs 0.80, so the gap is mostly species confusion. |

**Why the baseline was better at night.** Per-class recall shows that night and day are almost equal for most classes: coyote 0.77 / 0.72, dog 0.72 / 0.72, bobcat 0.50 / 0.51, rabbit 0.66 / 0.63. The overall night advantage comes mainly from the species mix: bird (all 180 test boxes) and squirrel (567 of 600), two of the hardest classes, appear almost only by day, and cats are found less often by day (0.20 vs 0.49). Darkness was therefore not the bottleneck that B was designed for.

**Likely reasons B underperforms** (to be tested in stage 4):

1. **Equal training budget.** B's val mAP rises more slowly (0.65 vs 0.80 after the first epoch) and almost catches up by epoch 6 (0.894 vs 0.903) on seen cameras, but not on new ones. Stronger augmentation usually needs a longer schedule than 6 epochs (the plan foresaw 20).
2. **Darkening is also applied to frames that are already dark.** It is applied with the same probability to night IR frames, and darkening them again produces near-black images (see `augmentation_examples.png`) that do not resemble real test frames.
3. **Grayscale removes colour cues.** The class-aware drop (−4.3) is about twice the class-agnostic drop (−2.3). B still finds animals, but confuses species more often, especially those identified by colour by day (dog, bird, bobcat).
4. **Border crop leaves unlabeled animal parts.** Boxes with less than 40% visible are removed, but the visible fragment stays in the image as "background". This can teach the model to ignore partly visible animals, which matches the lower truncated recall.

Limitations: one seed and 6 epochs per model; the intervals cover variation across test images, not run-to-run variation between seeds.

### Next steps (stage 4, 27 Oct)

- Train A and B for longer (12 epochs) to separate the training-budget effect from the augmentation effect.
- Variant B′: darken only colour (daytime) frames, drop grayscale, keep the wider scale jitter, and keep clipped boxes from the border crop instead of turning them into background.
- Error taxonomy (localisation / class confusion / background) and at least three explained failure examples.
