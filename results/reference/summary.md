# COCO reference: results on `test`

| Mode | Threshold (val) | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |
|---|---|---|---|---|---|---|---|---|
| agnostic | 0.35 | 0.377 | 0.183 | 0.706 | 0.356 | 0.473 | 0.023 | 0.357 |

Condition subsets (agnostic, test threshold from val):

| Condition | Subset | Images | Recall | Image recall | FP / empty image |
|---|---|---|---|---|---|
| is_night | yes | 3046 | 0.218 | 0.221 | 0.028 |
| is_night | no | 2796 | 0.507 | 0.517 | 0.021 |
| is_dark | yes | 2147 | 0.227 | 0.230 | 0.030 |
| is_dark | no | 3695 | 0.429 | 0.435 | 0.021 |
| truncated | yes | 859 | 0.312 | 0.311 | n/a |
| truncated | no | 4983 | 0.364 | 0.367 | 0.023 |
| is_small | yes | 1007 | 0.179 | 0.179 | n/a |
| is_small | no | 4835 | 0.403 | 0.400 | 0.023 |
