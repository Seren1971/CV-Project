"""Faster R-CNN ResNet-50 FPN (torchvision) with COCO weights.

Constructor:  torchvision.models.detection.fasterrcnn_resnet50_fpn
Weights:      FasterRCNN_ResNet50_FPN_Weights.COCO_V1
"""
from __future__ import annotations

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MAX_SIZE = 1024             # long side, same for every run
EVAL_MIN_SIZE = 768         # shorter side at val/test time
TRAIN_SCALES = {            # shorter side sampled per image during training
    "A": (640, 704, 768, 832, 896),
    "B": (480, 576, 672, 768, 864, 960),   # wider range for small / distant animals
}

# COCO category ids of animals, used by the untrained reference model
COCO_ANIMALS = {16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
                22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe"}


def build_model(num_classes: int | None, variant: str = "A", pretrained: bool = True):
    """num_classes=None keeps the original 91-class COCO head (reference model)."""
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None,
                                    min_size=TRAIN_SCALES.get(variant, (EVAL_MIN_SIZE,)),
                                    max_size=MAX_SIZE)
    if num_classes is not None:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def set_eval_size(model):
    model.transform.min_size = (EVAL_MIN_SIZE,)


def set_train_size(model, variant: str):
    model.transform.min_size = TRAIN_SCALES[variant]
