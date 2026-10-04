"""Box-aware augmentations for detection training.

Two pipelines, matching docs/experimental_plan.md:
    baseline  (A): horizontal flip + mild colour jitter (scale jitter is set in the model)
    targeted  (B): A + darkening / grayscale, border crops, partial occlusion

Every transform takes and returns (image, target). Images are PIL until `ToTensor`,
boxes are xyxy in pixels.
"""
from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

MIN_VISIBLE = 0.4  # a box cut by a border crop is kept if >= 40% of it stays visible


class Compose:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, img, target):
        for t in self.ts:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class HFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            w = img.width
            b = target["boxes"].clone()
            b[:, [0, 2]] = w - target["boxes"][:, [2, 0]]
            target = {**target, "boxes": b}
            img = F.hflip(img)
        return img, target


class Colour:
    def __init__(self, strength=0.2):
        self.cj = ColorJitter(brightness=strength, contrast=strength, saturation=strength)

    def __call__(self, img, target):
        return self.cj(img), target


class Darken:
    """Imitates night / under-exposed frames: lower brightness and a gamma > 1."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.adjust_gamma(img, random.uniform(1.5, 3.0))
            img = F.adjust_brightness(img, random.uniform(0.3, 0.7))
        return img, target


class Gray:
    """Imitates infrared frames (kept as 3 channels)."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.rgb_to_grayscale(img, num_output_channels=3)
        return img, target


class BorderCrop:
    """Crops a window covering 60-90% of each side so animals get cut by the border.

    Boxes keep only the visible part; boxes with < MIN_VISIBLE of their area left are
    dropped. If that would remove every box of a non-empty image, the crop is skipped.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target
        w, h = img.size
        cw, ch = int(w * random.uniform(0.6, 0.9)), int(h * random.uniform(0.6, 0.9))
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        boxes = target["boxes"]
        clipped = boxes.clone()
        clipped[:, [0, 2]] = (boxes[:, [0, 2]] - x0).clamp(0, cw)
        clipped[:, [1, 3]] = (boxes[:, [1, 3]] - y0).clamp(0, ch)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = area(clipped) >= MIN_VISIBLE * area(boxes)
        if len(boxes) and not keep.any():
            return img, target
        img = F.crop(img, y0, x0, ch, cw)
        out = {k: v for k, v in target.items()}
        out["boxes"], out["labels"] = clipped[keep], target["labels"][keep]
        return img, out


class PartialOcclusion:
    """Paints a patch over part of a random box (never the whole box), like grass in front."""

    def __init__(self, p=0.3, frac=(0.2, 0.5)):
        self.p, self.frac = p, frac

    def __call__(self, img, target):
        if random.random() >= self.p or len(target["boxes"]) == 0:
            return img, target
        x1, y1, x2, y2 = target["boxes"][random.randrange(len(target["boxes"]))].tolist()
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            return img, target
        f = random.uniform(*self.frac) ** 0.5  # patch area ~ frac of the box area
        pw, ph = int(bw * f), int(bh * f)
        px = int(random.uniform(x1, x2 - pw))
        py = int(random.uniform(y1, y2 - ph))
        img = img.clone()
        noise = torch.rand(img.shape[0], 1, 1) * 0.5  # dark vegetation-like patch
        img[:, py:py + ph, px:px + pw] = noise + 0.1 * torch.rand(img.shape[0], ph, pw)
        return img, target


def build_train_transform(variant: str):
    if variant == "A":
        return Compose([Colour(0.2), HFlip(), ToTensor()])
    if variant == "B":
        return Compose([Colour(0.2), Darken(), Gray(), BorderCrop(), HFlip(),
                        ToTensor(), PartialOcclusion()])
    raise ValueError(f"unknown variant {variant!r}")


def eval_transform():
    return Compose([ToTensor()])
