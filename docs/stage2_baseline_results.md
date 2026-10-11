# Stage 2: working dataset code and first baseline results

*P16 — Wildlife Detection in Camera-Trap Images. Run date: 11 Oct 2026.*

## What was run

- **Data:** CCT20, 11 animal classes (bird, bobcat, cat, coyote, dog, opossum, rabbit, raccoon, rodent, skunk, squirrel). Train 11,382 images / 11,891 boxes; val 2,148 images (600 empty); test = `trans_test`, 5,842 images (602 empty) from 9 camera locations never seen in training. Full statistics: [dataset_preparation.md](dataset_preparation.md).
- **COCO reference:** torchvision Faster R-CNN R50-FPN with `COCO_V1` weights, no CCT20 training. COCO animal classes are merged into one "animal" class, so it is evaluated class-agnostically only.
- **Baseline A:** the same model fine-tuned on CCT20 train with standard augmentation (horizontal flip, mild colour jitter, shorter-side scale jitter 640–896, long side ≤ 1024). SGD with momentum 0.9 and weight decay 1e-4, lr 0.01 with 500-iteration warm-up and cosine decay, batch 4, mixed precision, **6 epochs, 1 seed (0)**. The best epoch was chosen by val mAP@0.5 on a fixed 1,500-image val subset (epoch 5, val mAP@0.5 = 0.903).
- **Thresholds:** the confidence threshold maximising F1 on val, then fixed for test. Precision, recall and F1 use one-to-one greedy matching at IoU ≥ 0.5.
- **Hardware:** NVIDIA RTX 5070 (12 GB), PyTorch 2.11 + CUDA 12.8. Training took 55 minutes (≈9.2 min per epoch). Environment details: `results/environment.txt`.

Reproduce: `run_baseline.bat` (Windows) or `python scripts/run_all.py --epochs 6`.

## Main results on test (new camera locations)

| Model | Mode | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| COCO reference | class-agnostic | 0.377 | 0.183 | 0.706 | 0.356 | 0.473 | 0.023 | 0.357 |
| Baseline A | class-agnostic | **0.827** | **0.501** | 0.771 | **0.797** | **0.784** | 0.349 | **0.807** |
| Baseline A | per class | 0.603 | 0.367 | 0.701 | 0.623 | 0.660 | 0.239 | 0.640 |

AP@0.5 per class (baseline A): opossum 0.919, skunk 0.859, coyote 0.817, raccoon 0.792, rabbit 0.742, dog 0.724, bobcat 0.642, squirrel 0.475, cat 0.388, bird 0.243, rodent 0.036.

## Condition subsets (baseline A, per class, test)

| Condition | Subset | Images | Recall | Image recall | FP / empty image |
|---|---|---|---|---|---|
| Night (grayscale IR) | yes / no | 3,046 / 2,796 | 0.696 / 0.544 | 0.702 / 0.567 | 0.182 / 0.271 |
| Dark (low brightness) | yes / no | 2,147 / 3,695 | 0.709 / 0.575 | 0.717 / 0.594 | 0.213 / 0.249 |
| Truncated (box at border) | yes / no | 859 / 4,983 | **0.389** / 0.670 | 0.387 / 0.690 | – |
| Small (< 1% of image) | yes / no | 1,007 / 4,835 | **0.430** / 0.675 | 0.477 / 0.679 | – |

## First observations

1. **Fine-tuning helps a lot (H1 supported).** The untrained COCO model finds only 36% of the animals. It is very conservative (0.02 false positives per empty frame), but at night its recall falls to 0.22. After fine-tuning, class-agnostic mAP@0.5 rises from 0.38 to 0.83.
2. **New locations hurt classification more than localisation.** Val (seen cameras) reaches mAP@0.5 = 0.90, while test reaches 0.60 per class but 0.83 class-agnostic. The model finds the animals but often confuses species at unseen cameras. Similar small or look-alike classes are the weakest: rodent (only 24 test boxes), bird, cat and squirrel. The `test_cis` split is prepared, so this gap can be measured cleanly against seen locations in the next stage.
3. **Night is not the main problem for baseline A (H2 not supported so far).** Recall is higher on night IR frames (0.70) than on day frames (0.54), and night frames also produce fewer false alarms. Possible reasons to check: the IR flash lights up nearby animals against a dark, uniform background; daytime frames contain more vegetation, shadows and camouflage; and the night subset contains more nocturnal species (opossum, raccoon, skunk), which have the highest AP. A per-class night/day comparison is needed before drawing a conclusion.
4. **Partial visibility and small animals are the clearest weaknesses (consistent with H3).** Recall drops from 0.67 to 0.39 for animals cut by the image border and to 0.43 for small animals.
5. **False alarms on empty frames.** At the chosen threshold, 21% of empty frames get at least one false detection (0.24 per empty frame), and more often in daylight (0.27 vs 0.18 at night). These frames will be inspected to check whether vegetation is the cause.

## Limitations of this first run

- 6 epochs instead of the planned 20, and one seed. The val curve was still rising slowly (0.884 → 0.903 → 0.903 over the last three epochs), so longer training may add a little.
- Epoch selection used a 1,500-image val subset. The threshold was chosen on the full val split.
- Condition tags are automatic proxies (see the dataset section). Vegetation occlusion and unusual poses are not tagged yet.

## Next steps (stage 3, 20 Oct)

- Variant B (targeted augmentation) with the same budget, then A vs B on all subsets.
- Per-class night/day breakdown and evaluation on `test_cis`.
- Error taxonomy (localisation, class confusion, background) and inspection of the most confident false positives on empty frames.
