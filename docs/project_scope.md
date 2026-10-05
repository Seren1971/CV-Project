# Project scope

## Objective

Detect and classify selected animal categories in CCT20 camera-trap images using a pretrained Faster R-CNN detector, and analyse false detections and missed animals caused by vegetation, darkness, partial visibility and unusual poses.

## Research question

Does fine-tuning Faster R-CNN with augmentations that simulate difficult camera-trap conditions improve detection performance over fine-tuning with standard augmentations, particularly on dark and partially visible animals at previously unseen camera locations?

## Scope and comparison

The main task is object detection: predict an animal category, bounding box and confidence for each detected instance. We use the selected CCT20 animal classes and the official split policy described in [Dataset preparation](dataset_preparation.md). Empty validation and test frames are retained to measure false alarms.

The main comparison is between two Faster R-CNN ResNet-50 FPN models initialised with COCO-pretrained weights from torchvision:

- **Baseline A:** fine-tuning with standard augmentation.
- **Variant B:** the same training setup with additional augmentation for darkness, partial visibility, occlusion and small animals.

The architecture, splits and training budget are held constant. This comparison evaluates the augmentation package as a whole; it does not isolate the effect of each individual transformation. A COCO-pretrained detector without CCT20 fine-tuning is an additional class-agnostic reference, subject to an explicit COCO-to-animal label mapping.

Checkpoints and confidence thresholds are selected using validation data. The test set is reserved for final evaluation. We report mAP@0.5, mAP@[0.5:0.95], precision, recall and F1, together with false positives per empty image and condition-specific error analysis. At least three representative failures will be explained. A concise sensitivity experiment will also be included; its exact protocol must be agreed with the experiment owner before execution.

## Boundaries and limitations

The project uses an existing public dataset, pretrained weights and a single Colab or Kaggle GPU. Training a detector from scratch, collecting a new dataset, deployment, video tracking and additional detector architectures are outside the core scope. Evaluation on the optional `test_cis` split and repeated training seeds depend on the available compute budget.

Automatic condition tags are proxies: grayscale is not proof of night-time capture, border-touching boxes cover truncation rather than every form of partial visibility, and false positives on empty frames are background errors whose cause must be inspected before attributing them to vegetation. Occlusion and unusual poses require manual inspection. Conclusions will describe observed results and limitations rather than assume that the proposed augmentation improves performance.

## Expected outputs

- Reproducible code or a notebook with dataset download instructions, environment information and exact run commands.
- Saved evaluation metrics, predictions and a compact comparison table or figure.
- Failure examples with explanations of likely causes.
- A two-page technical summary with individual contributions.
- A five-minute demonstration with at most three supporting slides, followed by individual questions.

This document describes planned work. Training, evaluation and numerical results are not yet claimed.
