# Baseline A: results on `test`

| Mode | Threshold (val) | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| class_aware | 0.80 | 0.603 | 0.367 | 0.701 | 0.623 | 0.660 | 0.239 | 0.640 |
| agnostic | 0.70 | 0.827 | 0.501 | 0.771 | 0.797 | 0.784 | 0.349 | 0.807 |

AP@0.5 per class:

| Class | AP@0.5 |
|---|---|
| bird | 0.243 |
| bobcat | 0.642 |
| cat | 0.388 |
| coyote | 0.817 |
| dog | 0.724 |
| opossum | 0.919 |
| rabbit | 0.742 |
| raccoon | 0.792 |
| rodent | 0.036 |
| skunk | 0.859 |
| squirrel | 0.475 |

Condition subsets (class_aware, test threshold from val):

| Condition | Subset | Images | Recall | Image recall | FP / empty image |
|---|---|---|---|---|---|
| is_night | yes | 3046 | 0.696 | 0.702 | 0.182 |
| is_night | no | 2796 | 0.544 | 0.567 | 0.271 |
| is_dark | yes | 2147 | 0.709 | 0.717 | 0.213 |
| is_dark | no | 3695 | 0.575 | 0.594 | 0.249 |
| truncated | yes | 859 | 0.389 | 0.387 | n/a |
| truncated | no | 4983 | 0.670 | 0.690 | 0.239 |
| is_small | yes | 1007 | 0.430 | 0.477 | n/a |
| is_small | no | 4835 | 0.675 | 0.679 | 0.239 |
