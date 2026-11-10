# Baseline A vs variant B on `test`

Same model, data, epochs and seed; only the training augmentation differs. Thresholds are chosen on val for each model. Intervals: 95% bootstrap over sequences (1000 resamples), paired on the same images; **bold** = interval excludes 0.

## Overall

| Model | Mode | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| A | class-aware | 0.607 | 0.367 | 0.821 | 0.563 | 0.668 | 0.080 | 0.582 |
| B | class-aware | 0.609 | 0.368 | 0.821 | 0.569 | 0.672 | 0.115 | 0.587 |
| A12 (test_cis) | class-aware | 0.866 | 0.541 | 0.920 | 0.762 | 0.834 | 0.047 | 0.789 |
| B'12 (test_cis) | class-aware | 0.862 | 0.536 | 0.907 | 0.770 | 0.833 | 0.035 | 0.791 |
| A | agnostic | 0.820 | 0.496 | 0.776 | 0.786 | 0.781 | 0.324 | 0.801 |
| B | agnostic | 0.837 | 0.501 | 0.827 | 0.790 | 0.808 | 0.233 | 0.802 |
| A12 (test_cis) | agnostic | 0.911 | 0.573 | 0.911 | 0.878 | 0.894 | 0.154 | 0.900 |
| B'12 (test_cis) | agnostic | 0.914 | 0.570 | 0.895 | 0.882 | 0.889 | 0.150 | 0.904 |

Difference B - A on test (per class): recall +0.006 [-0.004, +0.014], precision +0.000 [-0.013, +0.014], FP per empty image +0.035 [-0.002, +0.073].

## AP@0.5 per class

| Class | A | B | B - A |
|---|---|---|---|
| bird | 0.200 | 0.144 | -0.056 |
| bobcat | 0.662 | 0.693 | +0.031 |
| cat | 0.403 | 0.377 | -0.026 |
| coyote | 0.808 | 0.812 | +0.005 |
| dog | 0.704 | 0.774 | +0.070 |
| opossum | 0.909 | 0.913 | +0.004 |
| rabbit | 0.744 | 0.755 | +0.012 |
| raccoon | 0.791 | 0.818 | +0.028 |
| rodent | 0.092 | 0.044 | -0.048 |
| skunk | 0.861 | 0.866 | +0.005 |
| squirrel | 0.508 | 0.501 | -0.006 |

## Condition subsets (recall, per class)

Gap = recall on the rest of the test set minus recall on the subset (smaller is better).

| Condition | Animal boxes | Recall A | Recall B | B - A [95% CI] | Gap A | Gap B |
|---|---|---|---|---|---|---|
| Night (IR) | 2931 | 0.647 | 0.654 | +0.006 [-0.006, +0.018] | -0.177 | -0.178 |
| Dark | 2028 | 0.664 | 0.677 | +0.013 [-0.001, +0.027] | -0.159 | -0.169 |
| Truncated | 934 | 0.333 | 0.350 | +0.017 [-0.008, +0.044] | 0.276 | 0.263 |
| Small | 1176 | 0.364 | 0.416 | **+0.052 [+0.033, +0.073]** | 0.252 | 0.194 |

False alarms on empty frames (FP per empty image):

| Empty frames | Count | A | B | B - A [95% CI] |
|---|---|---|---|---|
| all | 602 | 0.080 | 0.115 | +0.035 [-0.002, +0.073] |
| night | 214 | 0.033 | 0.028 | -0.005 [-0.037, +0.025] |
| day | 388 | 0.106 | 0.162 | +0.057 [+0.000, +0.109] |

## Recall per class, night vs day

Checks whether the night/day difference comes from the species mix.

| Class | Night boxes | Day boxes | A night | A day | B night | B day |
|---|---|---|---|---|---|---|
| bird | 0 | 180 | n/a | 0.172 | n/a | 0.233 |
| bobcat | 361 | 250 | 0.490 | 0.436 | 0.418 | 0.440 |
| cat | 304 | 372 | 0.428 | 0.175 | 0.372 | 0.167 |
| coyote | 340 | 298 | 0.676 | 0.614 | 0.726 | 0.567 |
| dog | 65 | 656 | 0.662 | 0.640 | 0.738 | 0.642 |
| opossum | 599 | 7 | 0.845 | 0.571 | 0.883 | 0.571 |
| rabbit | 309 | 316 | 0.625 | 0.557 | 0.689 | 0.646 |
| raccoon | 642 | 0 | 0.648 | n/a | 0.642 | n/a |
| rodent | 21 | 3 | 0.143 | 0.000 | 0.000 | 0.000 |
| skunk | 257 | 16 | 0.774 | 0.875 | 0.790 | 0.938 |
| squirrel | 33 | 567 | 0.000 | 0.444 | 0.000 | 0.425 |
