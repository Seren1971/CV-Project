"""Box-aware transforms for detection: each takes and returns (image, target).

Images are PIL until `ToTensor`; target["boxes"] is a float tensor [N, 4] in xyxy pixels.

Baseline A (docs/experimental_plan.md): horizontal flip + mild colour jitter.
Scale jitter is done inside the detector (`min_size` tuple), not here.
Variant B adds its own transforms on top of `baseline_train()`.
"""
from __future__ import annotations

import random

import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            w = img.width
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - target["boxes"][:, [2, 0]]
            target = {**target, "boxes": boxes}
            img = F.hflip(img)
        return img, target


class ColourJitter:
    def __init__(self, strength: float = 0.2):
        self.cj = ColorJitter(brightness=strength, contrast=strength, saturation=strength)

    def __call__(self, img, target):
        return self.cj(img), target


def baseline_train() -> Compose:
    return Compose([HorizontalFlip(0.5), ColourJitter(0.2), ToTensor()])


def eval_transform() -> Compose:
    return Compose([ToTensor()])
