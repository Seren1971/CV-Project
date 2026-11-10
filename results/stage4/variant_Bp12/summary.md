# Variant B' (12 epochs): results on `test`

| Mode | Threshold (val) | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| class_aware | 0.90 | 0.609 | 0.368 | 0.821 | 0.569 | 0.672 | 0.115 | 0.587 |
| agnostic | 0.70 | 0.837 | 0.501 | 0.827 | 0.790 | 0.808 | 0.233 | 0.802 |

AP@0.5 per class:

| Class | AP@0.5 |
|---|---|
| bird | 0.144 |
| bobcat | 0.693 |
| cat | 0.377 |
| coyote | 0.812 |
| dog | 0.774 |
| opossum | 0.913 |
| rabbit | 0.755 |
| raccoon | 0.818 |
| rodent | 0.044 |
| skunk | 0.866 |
| squirrel | 0.501 |

Condition subsets (class_aware, test threshold from val):

| Condition | Subset | Images | Recall | Image recall | FP / empty image |
|---|---|---|---|---|---|
| is_night | yes | 3046 | 0.654 | 0.660 | 0.028 |
| is_night | no | 2796 | 0.476 | 0.502 | 0.162 |
| is_dark | yes | 2147 | 0.677 | 0.683 | 0.035 |
| is_dark | no | 3695 | 0.508 | 0.529 | 0.145 |
| truncated | yes | 859 | 0.350 | 0.352 | n/a |
| truncated | no | 4983 | 0.613 | 0.634 | 0.115 |
| is_small | yes | 1007 | 0.416 | 0.464 | n/a |
| is_small | no | 4835 | 0.610 | 0.617 | 0.115 |
