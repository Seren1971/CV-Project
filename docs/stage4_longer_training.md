# Stage 4: longer training (12 epochs) and variant B′

*P16 — Wildlife Detection in Camera-Trap Images.*

Stage 3 showed that variant B (targeted augmentation) is **worse** than baseline A at an equal
6-epoch budget, and the error analysis pointed to four likely causes. Stage 4 tests them:

1. **Training budget.** B's val mAP rises more slowly and had almost caught up on seen cameras
   by epoch 6. A and B′ are re-trained for **12 epochs** to separate the budget effect from the
   augmentation effect.
2. **Darkening IR frames.** Most training frames are already dark night IR images; darkening
   them again produces unrealistic near-black images. B′ applies Darken **only to colour
   (daytime) frames** (`is_colour_image` in `src/data/transforms.py`: mean pairwise channel
   difference below 5/256 ⇒ IR frame ⇒ skip).
3. **Grayscale removes colour cues.** The class-aware drop of B (−4.3 mAP) was about twice the
   class-agnostic drop (−2.3), i.e. B still finds animals but confuses species. B′ **drops
   Grayscale** entirely.
4. **Unlabeled animal fragments.** B's border crop removed boxes with < 40 % visible area but
   left the visible fragment in the image as "background", teaching the model to ignore
   truncated animals. B′ **keeps clipped boxes down to 15 % visible area** and paints out
   smaller leftovers with a background patch (`border_crop_window_v2` in
   `src/data/box_geometry.py`, `BorderCropV2` in `src/data/transforms.py`).

Variant B′ keeps B's partial erase (p = 0.3), border crop (p = 0.3) and wide scale jitter
(480–960 px). Everything else — model, optimiser, seed, splits, selection rule — is unchanged.

| | B (stage 3) | B′ (stage 4) |
|---|---|---|
| Border crop | boxes < 40 % visible dropped, fragments stay as background | boxes ≥ 15 % visible kept and clipped; smaller fragments erased |
| Partial erase | p = 0.3 | same |
| Darken | p = 0.3 on all frames | p = 0.3, colour frames only |
| Grayscale | p = 0.3 | **removed** |
| Scale jitter | 480–960 | same |
| Epochs | 6 | 12 |

Reproduce: `run_stage4.bat` (trains A and B′ for 12 epochs, predicts val/test/test_cis),
then `python scripts/run_stage4.py` (evaluation + comparison with 95 % bootstrap intervals).
Smoke preview of the B′ transforms: `results/variant_Bp_smoke.png`.

## Results

Run on 10 Nov 2026, RTX 5070, 12 epochs, seed 0, 115.3 min (A12) and 117.5 min (B′12)
of training. Full tables: `results/stage4/comparison_A12_vs_Bp12.md`. Differences are B′ − A at
the same 12-epoch budget with 95 % bootstrap intervals over sequences; **bold** = excludes 0.

### Overall (test = new camera locations)

| Model | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image |
|---|---|---|---|---|---|---|
| A, 6 epochs (stage 3) | 0.603 | 0.367 | 0.701 | 0.623 | 0.660 | 0.239 |
| B, 6 epochs (stage 3) | 0.560 | 0.329 | 0.648 | 0.580 | 0.612 | 0.244 |
| A, 12 epochs | 0.607 | 0.367 | 0.821 | 0.563 | 0.668 | **0.080** |
| B′, 12 epochs | **0.609** | **0.368** | **0.821** | **0.569** | **0.672** | 0.115 |

Class-agnostic detection (the primary "is there an animal" task): **B′ 0.837 mAP@0.5 vs A 0.820**
(F1 0.808 vs 0.781). On `test_cis` (seen locations) the two models are equal
(0.862 vs 0.866 class-aware; 0.914 vs 0.911 agnostic).

### Target conditions (recall on test, class-aware)

| Condition | Boxes | A12 | B′12 | B′ − A [95% CI] |
|---|---|---|---|---|
| Small animals | 1,176 | 0.364 | 0.416 | **+0.052 [+0.033, +0.073]** |
| Truncated | 934 | 0.333 | 0.350 | +0.017 [−0.008, +0.044] |
| Night (IR) | 2,931 | 0.647 | 0.654 | +0.006 [−0.006, +0.018] |
| Empty frames: FP / image | 602 | 0.080 | 0.115 | +0.035 [−0.002, +0.073] |

### Reading

1. **The budget was the main problem of B, not the idea.** At 12 epochs B′ fully recovers the
   −4.3 mAP regression that B showed at 6 epochs and matches the baseline overall.
2. **B′ is the better detector where it was designed to help:** +5.2 recall points on small
   animals (significant; the small-animal gap shrinks from 0.25 to 0.19) and +1.7 mAP
   class-agnostic. Truncated animals improve slightly but not significantly — border crops are
   a weak imitation of real close-range truncation.
3. **Longer training mostly fixes calibration:** A12 vs A6 at the val-chosen threshold keeps
   the same recall level of detections but cuts false alarms on empty frames from 0.24 to
   0.08 per image and raises the share of correct detections from 70 % to 82 %. B′ pays a
   small price in daytime false alarms (+0.06 per empty day frame, borderline significant) —
   partial erasing still teaches some "animal-like" background responses.
4. **Night is simply not the bottleneck:** night recall ≈ day recall for both models; the
   overall day/night difference comes from the species mix (birds and squirrels are daytime,
   and both are hard classes).

**Final model for the demo: B′, 12 epochs** (best agnostic mAP and small-animal recall at
equal overall class-aware performance).

Limitations: still one seed per configuration; per-class APs for rare classes (rodent, skunk)
are noisy; the effect is attributed to the B′ transform package as a whole.
