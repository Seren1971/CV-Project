"""PyTorch datasets for the prepared CCT20 splits.

Important: the downloadable "_sm" images are resized to max 1024 px on the long
side, while bounding boxes in the official JSON refer to the ORIGINAL resolution
(width/height fields). `scaled_boxes` rescales them to the image actually on disk.

Two modes:
    CCT20Classification - one label per image (class of the largest box);
                          `crop=True` classifies the largest box crop instead.
    CCT20Detection      - torchvision-style target dict (boxes in xyxy, labels).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # allows the prepare/tag scripts to run without torch
    torch = None
    Dataset = object


def scaled_boxes(rec: dict, w: int, h: int) -> list[list[float]]:
    """Boxes [x, y, w, h] rescaled from annotation resolution to (w, h)."""
    sx = w / rec["width"] if rec.get("width") else 1.0
    sy = h / rec["height"] if rec.get("height") else 1.0
    return [[b["bbox"][0] * sx, b["bbox"][1] * sy, b["bbox"][2] * sx, b["bbox"][3] * sy]
            for b in rec["boxes"]]


class _Base(Dataset):
    def __init__(self, split: str, image_dir: str | Path,
                 splits_dir: str | Path = "data/splits", transform=None):
        splits_dir = Path(splits_dir)
        self.records = json.load(open(splits_dir / f"{split}.json"))
        self.classes = json.load(open(splits_dir / "classes.json"))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, rec: dict) -> Image.Image:
        return Image.open(self.image_dir / rec["file_name"]).convert("RGB")


class CCT20Classification(_Base):
    def __init__(self, *args, crop: bool = False, pad: float = 0.1, **kw):
        super().__init__(*args, **kw)
        self.crop, self.pad = crop, pad

    def __getitem__(self, i):
        rec = self.records[i]
        img = self._load(rec)
        if self.crop:
            boxes = scaled_boxes(rec, *img.size)
            x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
            px, py = bw * self.pad, bh * self.pad
            img = img.crop((max(0, x - px), max(0, y - py),
                            min(img.width, x + bw + px), min(img.height, y + bh + py)))
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[rec["label"]]


class CCT20Detection(_Base):
    def __getitem__(self, i):
        rec = self.records[i]
        img = self._load(rec)
        boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh in scaled_boxes(rec, *img.size)]
        # label 0 is reserved for background in torchvision detectors
        labels = [self.class_to_idx[b["category"]] + 1 for b in rec["boxes"]]
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32),
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": rec["image_id"]}
        if self.transform:
            img = self.transform(img)
        return img, target
