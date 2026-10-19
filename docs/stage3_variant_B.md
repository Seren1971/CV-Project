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

Run on 19 Oct 2026, RTX 5070, 6 epochs, seed 0, 59 minutes of training (A: 55 minutes). Full tables: `results/comparison_A_vs_B.md`. Differences are B − A with 95% bootstrap intervals over sequences, paired on the same test images; **bold** means the interval excludes 0.

### Overall

| Test set | Model | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image |
|---|---|---|---|---|---|---|---|
| test (new cameras), per class | A | **0.603** | **0.367** | **0.701** | **0.623** | **0.660** | 0.239 |
| | B | 0.560 | 0.329 | 0.648 | 0.580 | 0.612 | 0.244 |
| test, class-agnostic | A | **0.827** | **0.501** | 0.771 | **0.797** | **0.784** | 0.349 |
| | B | 0.804 | 0.468 | 0.759 | 0.748 | 0.754 | 0.302 |
| test_cis (seen cameras), per class | A | **0.862** | **0.537** | 0.866 | 0.809 | 0.837 | 0.100 |
| | B | 0.840 | 0.509 | 0.853 | 0.771 | 0.810 | 0.075 |

On `test`, per class: recall **−0.044 [−0.056, −0.033]**, precision **−0.052 [−0.068, −0.037]**, false positives per empty frame +0.005 [−0.056, +0.066].

### Target conditions (recall on `test`, per class)

| Condition | Boxes | A | B | B − A [95% CI] | Gap A → Gap B |
|---|---|---|---|---|---|
| Small animals | 1,176 | 0.430 | 0.466 | **+0.036 [+0.013, +0.061]** | 0.245 → **0.144** |
| Truncated | 934 | 0.389 | 0.357 | **−0.032 [−0.062, −0.005]** | 0.282 → 0.268 |
| Night (IR) | 2,931 | 0.696 | 0.653 | **−0.043 [−0.058, −0.027]** | night is not harder |
| Empty frames: FP / image | 602 | 0.239 | 0.244 | +0.005 [−0.056, +0.066] | – |

The largest per-class drops in AP@0.5 are dog (−0.154), bird (−0.110) and bobcat (−0.069; night recall 0.50 → 0.28). Rabbit recall improved both at night (0.66 → 0.74) and by day (0.63 → 0.72).

### Comparison with the baseline

**With the same 6-epoch budget, variant B does not improve on the baseline.** It is significantly worse overall (−4.3 mAP@0.5 points per class, −2.3 class-agnostic). Of the targeted conditions, it improves only **small animals**, where it closes 40% of the recall gap. The result held on both test sets. Against the hypotheses in the experimental plan:

| Hypothesis | Outcome |
|---|---|
| H2: night is harder; B narrows the gap | **Not supported.** Night is not harder; B is worse at night. |
| H3: B improves truncated animals | **Not supported.** Recall decreased slightly. |
| Small animals improve (wider scale jitter) | **Supported.** +3.6 recall points. |
| H4: empty-frame false alarms unchanged | **Supported** on `test`. On `test_cis` B even has fewer (0.075 vs 0.100). |
| H6: new locations are harder | **Supported** for both models: per-class mAP@0.5 is 0.86 on seen cameras and 0.60 on new ones. Class-agnostic it is 0.91 vs 0.83, so the gap is mostly species confusion. |

**Why the baseline was better at night.** Per-class recall shows that night and day are almost equal for most classes: coyote 0.77 / 0.72, dog 0.72 / 0.72, bobcat 0.50 / 0.51, rabbit 0.66 / 0.63. The overall night advantage comes mainly from the species mix and from daytime-specific difficulties. Two of the hardest classes appear almost only by day: bird (all 180 test boxes) and squirrel (567 of 600). Cats are found much less often by day than at night (0.20 vs 0.49). Darkness was therefore not the bottleneck that B was designed for.

**Likely reasons B underperforms** (to be tested in stage 4):

1. **Equal training budget.** B's training loss stays higher and its val mAP rises more slowly (0.65 vs 0.80 after the first epoch). By epoch 6 it almost catches up on seen cameras (0.894 vs 0.903 val mAP@0.5) but not on new ones. Stronger augmentation usually needs a longer schedule than the 6 epochs used here; the plan's 20 epochs were not used.
2. **Darkening is applied to frames that are already dark.** Most training frames are night IR images. Darkening them again produces near-black images (see `augmentation_examples.png`) that do not resemble real test frames.
3. **Grayscale removes colour cues.** The class-aware drop (−4.3) is about twice the class-agnostic drop (−2.3). B still finds animals, but confuses species more often, especially those identified by colour by day (dog, bird, bobcat).
4. **Border crop leaves unlabeled animal parts.** Boxes with less than 40% visible are removed, but the visible fragment stays in the image as "background". This can teach the model to ignore partly visible animals, which matches the lower truncated recall.

Limitations: one seed and 6 epochs per model. The differences are significant across test images, but run-to-run variation (different seeds) has not been measured.

### Next steps (stage 4, 27 Oct)

- Train A and B for longer (12 epochs) to separate the training-budget effect from the augmentation effect.
- Variant B′: darken only colour (daytime) frames, drop grayscale, keep the wider scale jitter, and keep clipped boxes from the border crop instead of turning them into background.
- Error taxonomy (localisation / class confusion / background) and at least three explained failure examples.
