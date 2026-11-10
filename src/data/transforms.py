"""Box-aware transforms for detection: each takes and returns (image, target).

Images are PIL until `ToTensor`; target["boxes"] is a float tensor [N, 4] in xyxy pixels.

Baseline A (docs/experimental_plan.md): horizontal flip + mild colour jitter.
Variant B: A + transforms that imitate the hard cases (all with p = 0.3):
    BorderCrop    crop whose edge cuts through an animal (40-80% stays visible) -> truncation
    PartialErase  cover 20-50% of one box with a patch of the same image's background -> occlusion
    Darken        lower brightness + gamma > 1 -> night / under-exposed frames
    Grayscale     3-channel grayscale -> infrared frames
Scale jitter is done inside the detector (`min_size` tuple), not here; B uses a wider range.
"""
from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

from src.data.box_geometry import border_crop_window, border_crop_window_v2, erase_rect


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


class BorderCrop:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p or len(target["boxes"]) == 0:
            return img, target
        res = border_crop_window(img.width, img.height, target["boxes"].tolist())
        if res is None:
            return img, target
        win, boxes, kept = res
        img = img.crop(win)
        target = {**target,
                  "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": target["labels"][torch.tensor(kept, dtype=torch.int64)]}
        return img, target


class PartialErase:
    """Cover part of one animal with a background patch taken from the same image."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img, target):
        if random.random() >= self.p or len(target["boxes"]) == 0:
            return img, target
        box = target["boxes"][random.randrange(len(target["boxes"]))].tolist()
        r = erase_rect(box)
        if r is None:
            return img, target
        w, h = r[2] - r[0], r[3] - r[1]
        if w < 1 or h < 1 or w >= img.width or h >= img.height:
            return img, target
        sx, sy = random.randint(0, img.width - w), random.randint(0, img.height - h)
        patch = img.crop((sx, sy, sx + w, sy + h))
        img = img.copy()
        img.paste(patch, (r[0], r[1]))
        return img, target


class Darken:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.adjust_brightness(img, random.uniform(0.3, 0.7))
            img = F.adjust_gamma(img, random.uniform(1.2, 2.0))
        return img, target


class Grayscale:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.rgb_to_grayscale(img, num_output_channels=3)
        return img, target


def baseline_train() -> Compose:
    return Compose([HorizontalFlip(0.5), ColourJitter(0.2), ToTensor()])


def targeted_train(p: float = 0.3) -> Compose:
    """Variant B. Same flip and colour jitter as A, plus the four targeted transforms."""
    return Compose([BorderCrop(p), HorizontalFlip(0.5), PartialErase(p),
                    ColourJitter(0.2), Darken(p), Grayscale(p), ToTensor()])


# ---------------------------------------------------------------------------
# Variant B' (stage 4): fixes the three diagnosed problems of B.
#   1. Darken is applied only to colour (daytime) frames, not to IR night frames
#      that are already dark.
#   2. Grayscale is dropped: it removed the colour cues used to tell species apart.
#   3. The border crop keeps clipped boxes down to 15 % visible area (instead of
#      turning them into unlabeled background) and paints out smaller leftovers.
#   PartialErase, the wider scale jitter (in the detector) and p = 0.3 are kept.
# ---------------------------------------------------------------------------

COLOUR_SPREAD_MIN = 5.0   # mean |R-G|+|R-B|+|G-B| (0-255) below this = IR / grayscale frame


def is_colour_image(img) -> bool:
    """True for colour (daytime) frames, False for near-grayscale IR night frames."""
    if isinstance(img, torch.Tensor):
        img = F.to_pil_image(img)
    small = img.resize((64, 64))
    r, g, b = small.split()
    import itertools
    import PIL.ImageChops as Chops
    import PIL.ImageStat as Stat
    spread = sum(Stat.Stat(Chops.difference(c1, c2)).mean[0]
                 for c1, c2 in itertools.combinations((r, g, b), 2)) / 3
    return spread >= COLOUR_SPREAD_MIN


class DaytimeDarken(Darken):
    """Darken applied only to colour frames (IR frames are already dark)."""

    def __call__(self, img, target):
        if not is_colour_image(img):
            return img, target
        return super().__call__(img, target)


class BorderCropV2:
    """Border crop that keeps clipped boxes (>=15% visible) and erases smaller leftovers,
    so no unlabeled animal parts remain in the frame."""

    def __init__(self, p: float = 0.3, min_visible: float = 0.15):
        self.p = p
        self.min_visible = min_visible

    def __call__(self, img, target):
        if random.random() >= self.p or len(target["boxes"]) == 0:
            return img, target
        res = border_crop_window_v2(img.width, img.height, target["boxes"].tolist(),
                                    min_visible=self.min_visible)
        if res is None:
            return img, target
        win, boxes, kept, dropped = res
        img = img.crop(win)
        for r in dropped:                       # paint out unlabeled fragments with background
            r = [int(round(v)) for v in r]      # geometry works in floats; pixels are ints
            w, h = r[2] - r[0], r[3] - r[1]
            if w < 1 or h < 1 or w >= img.width or h >= img.height:
                continue
            sx, sy = random.randint(0, img.width - w), random.randint(0, img.height - h)
            img.paste(img.crop((sx, sy, sx + w, sy + h)), (r[0], r[1]))
        target = {**target,
                  "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": target["labels"][torch.tensor(kept, dtype=torch.int64)]}
        return img, target


def targeted_train_v2(p: float = 0.3) -> Compose:
    """Variant B': B without Grayscale, Darken only on colour frames, clipped boxes kept."""
    return Compose([BorderCropV2(p), HorizontalFlip(0.5), PartialErase(p),
                    ColourJitter(0.2), DaytimeDarken(p), ToTensor()])


def eval_transform() -> Compose:
    return Compose([ToTensor()])
