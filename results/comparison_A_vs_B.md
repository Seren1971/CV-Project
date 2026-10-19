# Baseline A vs variant B on `test`

Same model, data, epochs and seed; only the training augmentation differs. Thresholds are chosen on val for each model. Intervals: 95% bootstrap over sequences (1000 resamples), paired on the same images; **bold** = interval excludes 0.

## Overall

| Model | Mode | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| A | class-aware | 0.603 | 0.367 | 0.701 | 0.623 | 0.660 | 0.239 | 0.640 |
| B | class-aware | 0.560 | 0.329 | 0.648 | 0.580 | 0.612 | 0.244 | 0.598 |
| A (test_cis) | class-aware | 0.862 | 0.537 | 0.866 | 0.809 | 0.837 | 0.100 | 0.829 |
| B (test_cis) | class-aware | 0.840 | 0.509 | 0.853 | 0.771 | 0.810 | 0.075 | 0.790 |
| A | agnostic | 0.827 | 0.501 | 0.771 | 0.797 | 0.784 | 0.349 | 0.807 |
| B | agnostic | 0.804 | 0.468 | 0.759 | 0.748 | 0.754 | 0.302 | 0.762 |
| A (test_cis) | agnostic | 0.912 | 0.574 | 0.890 | 0.887 | 0.889 | 0.143 | 0.908 |
| B (test_cis) | agnostic | 0.903 | 0.554 | 0.903 | 0.850 | 0.876 | 0.086 | 0.872 |

Difference B - A on test (per class): recall **-0.044 [-0.056, -0.033]**, precision **-0.052 [-0.068, -0.037]**, FP per empty image +0.005 [-0.056, +0.066].

## AP@0.5 per class

| Class | A | B | B - A |
|---|---|---|---|
| bird | 0.243 | 0.132 | -0.110 |
| bobcat | 0.642 | 0.573 | -0.069 |
| cat | 0.388 | 0.354 | -0.035 |
| coyote | 0.817 | 0.777 | -0.040 |
| dog | 0.724 | 0.570 | -0.154 |
| opossum | 0.919 | 0.904 | -0.015 |
| rabbit | 0.742 | 0.739 | -0.003 |
| raccoon | 0.792 | 0.779 | -0.013 |
| rodent | 0.036 | 0.038 | +0.002 |
| skunk | 0.859 | 0.829 | -0.030 |
| squirrel | 0.475 | 0.461 | -0.014 |

## Condition subsets (recall, per class)

Gap = recall on the rest of the test set minus recall on the subset (smaller is better).

| Condition | Animal boxes | Recall A | Recall B | B - A [95% CI] | Gap A | Gap B |
|---|---|---|---|---|---|---|
| Night (IR) | 2931 | 0.696 | 0.653 | **-0.043 [-0.058, -0.027]** | -0.152 | -0.153 |
| Dark | 2028 | 0.709 | 0.660 | **-0.049 [-0.069, -0.031]** | -0.134 | -0.126 |
| Truncated | 934 | 0.389 | 0.357 | **-0.032 [-0.062, -0.005]** | 0.282 | 0.268 |
| Small | 1176 | 0.430 | 0.466 | **+0.036 [+0.013, +0.061]** | 0.245 | 0.144 |

False alarms on empty frames (FP per empty image):

| Empty frames | Count | A | B | B - A [95% CI] |
|---|---|---|---|---|
| all | 602 | 0.239 | 0.244 | +0.005 [-0.056, +0.066] |
| night | 214 | 0.182 | 0.215 | +0.033 [-0.064, +0.137] |
| day | 388 | 0.271 | 0.260 | -0.010 [-0.087, +0.065] |

## Recall per class, night vs day

Checks whether the night/day difference comes from the species mix.

| Class | Night boxes | Day boxes | A night | A day | B night | B day |
|---|---|---|---|---|---|---|
| bird | 0 | 180 | n/a | 0.289 | n/a | 0.161 |
| bobcat | 361 | 250 | 0.504 | 0.512 | 0.280 | 0.416 |
| cat | 304 | 372 | 0.487 | 0.199 | 0.480 | 0.245 |
| coyote | 340 | 298 | 0.774 | 0.715 | 0.715 | 0.681 |
| dog | 65 | 656 | 0.723 | 0.716 | 0.754 | 0.572 |
| opossum | 599 | 7 | 0.868 | 0.571 | 0.886 | 0.429 |
| rabbit | 309 | 316 | 0.657 | 0.630 | 0.741 | 0.715 |
| raccoon | 642 | 0 | 0.710 | n/a | 0.646 | n/a |
| rodent | 21 | 3 | 0.143 | 0.000 | 0.000 | 0.000 |
| skunk | 257 | 16 | 0.848 | 0.938 | 0.770 | 0.938 |
| squirrel | 33 | 567 | 0.000 | 0.519 | 0.030 | 0.504 |
