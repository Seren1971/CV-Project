# Variant B: results on `test`

| Mode | Threshold (val) | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| class_aware | 0.75 | 0.560 | 0.329 | 0.648 | 0.580 | 0.612 | 0.244 | 0.598 |
| agnostic | 0.70 | 0.804 | 0.468 | 0.759 | 0.748 | 0.754 | 0.302 | 0.762 |

AP@0.5 per class:

| Class | AP@0.5 |
|---|---|
| bird | 0.132 |
| bobcat | 0.573 |
| cat | 0.354 |
| coyote | 0.777 |
| dog | 0.570 |
| opossum | 0.904 |
| rabbit | 0.739 |
| raccoon | 0.779 |
| rodent | 0.038 |
| skunk | 0.829 |
| squirrel | 0.461 |

Condition subsets (class_aware, test threshold from val):

| Condition | Subset | Images | Recall | Image recall | FP / empty image |
|---|---|---|---|---|---|
| is_night | yes | 3046 | 0.653 | 0.657 | 0.215 |
| is_night | no | 2796 | 0.500 | 0.529 | 0.260 |
| is_dark | yes | 2147 | 0.660 | 0.665 | 0.201 |
| is_dark | no | 3695 | 0.534 | 0.558 | 0.261 |
| truncated | yes | 859 | 0.356 | 0.357 | n/a |
| truncated | no | 4983 | 0.625 | 0.646 | 0.244 |
| is_small | yes | 1007 | 0.466 | 0.519 | n/a |
| is_small | no | 4835 | 0.610 | 0.617 | 0.244 |
