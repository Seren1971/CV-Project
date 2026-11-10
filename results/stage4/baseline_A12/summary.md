# Baseline A (12 epochs): results on `test`

| Mode | Threshold (val) | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| class_aware | 0.95 | 0.607 | 0.367 | 0.821 | 0.563 | 0.668 | 0.080 | 0.582 |
| agnostic | 0.75 | 0.820 | 0.496 | 0.776 | 0.786 | 0.781 | 0.324 | 0.801 |

AP@0.5 per class:

| Class | AP@0.5 |
|---|---|
| bird | 0.200 |
| bobcat | 0.662 |
| cat | 0.403 |
| coyote | 0.808 |
| dog | 0.704 |
| opossum | 0.909 |
| rabbit | 0.744 |
| raccoon | 0.791 |
| rodent | 0.092 |
| skunk | 0.861 |
| squirrel | 0.508 |

Condition subsets (class_aware, test threshold from val):

| Condition | Subset | Images | Recall | Image recall | FP / empty image |
|---|---|---|---|---|---|
| is_night | yes | 3046 | 0.647 | 0.655 | 0.033 |
| is_night | no | 2796 | 0.470 | 0.497 | 0.106 |
| is_dark | yes | 2147 | 0.664 | 0.672 | 0.041 |
| is_dark | no | 3695 | 0.506 | 0.528 | 0.095 |
| truncated | yes | 859 | 0.333 | 0.334 | n/a |
| truncated | no | 4983 | 0.609 | 0.631 | 0.080 |
| is_small | yes | 1007 | 0.364 | 0.412 | n/a |
| is_small | no | 4835 | 0.616 | 0.623 | 0.080 |
