# Saved predictions for error analysis

These are the saved inference outputs from the team's A/B error-analysis package. They are copied unchanged, so the detections can be inspected without model weights or rerunning inference. The existing result folders still hold the metrics, validation thresholds and training configuration.

| Directory | Model | Available splits |
| --- | --- | --- |
| `baseline_A/` | Faster R-CNN ResNet-50 FPN, initialized with COCO weights and fine-tuned on CCT20 with standard augmentation (seed 0) | `val`, `test`, `test_cis` |
| `variant_B/` | Same architecture and initialization, fine-tuned on CCT20 with targeted augmentation (seed 0) | `val`, `test`, `test_cis` |
| `coco_reference/` | COCO-pretrained Faster R-CNN ResNet-50 FPN without CCT20 fine-tuning; COCO animal detections are collapsed to one `animal` class | `val`, `test` |

Each `preds_<split>.json` has one entry per `image_id` in [`data/splits/<split>.json`](../../data/splits/). `val` is the `cis_val` split (2,148 images); `test` is `trans_test` at unseen camera locations (5,842 images); `test_cis` is the test split at seen locations (6,076 images). There is no `test_cis` file for the COCO reference in the package.

## JSON format and box coordinates

Each file is a JSON object keyed by image ID. An entry looks like this (illustrative values):

```json
{
  "example-image-id": {
    "size": [1024, 747],
    "boxes": [[12.4, 30.0, 190.1, 204.8]],
    "scores": [0.86],
    "labels": [1]
  }
}
```

- `size` is `[width, height]` of the image loaded by the dataset, before the detector's internal resize. The detector returns boxes in that loaded image's pixel coordinates. Use the size stored for each image: these files include both `1024 × 747` and `1024 × 768` images.
- `boxes` uses `[x_min, y_min, x_max, y_max]`, with the origin at the top left. `boxes`, `scores` and `labels` have matching indices and can all be empty.
- A and B use one-based label IDs in [`classes.json`](../../data/splits/classes.json), so class name `classes[label - 1]` matches a prediction. Background ID `0` is not exported. For the COCO reference, `1` means `animal`, not the first CCT20 class.

Join predictions to the prepared split records by `image_id`. Use the split record's `file_name` to locate the corresponding image. Ground-truth boxes in the split records use `[x, y, width, height]` in the annotation coordinate system. To compare them to predictions, convert to corner coordinates, then scale x coordinates by `size[0] / record.width` and y coordinates by `size[1] / record.height`. This is the conversion used by [`scaled_boxes`](../../src/data/dataset.py) and [`gt_boxes`](../../src/metrics.py). For an overlay on the resized CCT20 image loaded by the dataset, prediction boxes can be drawn directly in that image's coordinates.

Use the `threshold_from_val` for the matching evaluation mode from the existing metrics: [A](../baseline_A/metrics.json), [B](../variant_B/metrics.json), or [COCO reference](../reference/metrics.json). For `test_cis`, see [A CIS](../baseline_A_cis/metrics.json) and [B CIS](../variant_B_cis/metrics.json). The COCO reference can be compared only in class-agnostic mode. [`src/metrics.py`](../../src/metrics.py) contains the project's box matching at IoU 0.5, and [`test_conditions.csv`](../../data/splits/test_conditions.csv) contains condition tags for `test`.

## Original CCT20 annotations

Use the [CCT20 benchmark metadata on LILA BC](https://lila.science/datasets/caltech-camera-traps/#cct20-benchmark-subset). From the repository root, download just the metadata:

```sh
bash scripts/download_cct20.sh --annotations-only
```

The script extracts the [public metadata archive](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz) into `data/raw/eccv_18_annotation_files/`. The original test files are `trans_test_annotations.json` and `cis_test_annotations.json`. Use the prepared split files for the exact image sets evaluated by this project; the original annotation files include records outside those prepared sets. Downloading the metadata does not download images.
