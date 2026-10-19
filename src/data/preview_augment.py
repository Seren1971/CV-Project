"""Save a grid of variant-B augmentations with their boxes (sanity check + report figure).

    python -m src.data.preview_augment --image-dir <images> --out results/variant_B/augmentation_examples.png

Columns: original | border crop | partial erase | darken + grayscale | full B pipeline.
Each transform in columns 2-4 is forced (p = 1) so its effect is visible.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image

from src.data.dataset import CCT20Detection
from src.data.transforms import (BorderCrop, Compose, Darken, Grayscale, PartialErase, ToTensor,
                                 targeted_train)

COLUMNS = {
    "original": Compose([ToTensor()]),
    "border crop": Compose([BorderCrop(1.0), ToTensor()]),
    "partial erase": Compose([PartialErase(1.0), ToTensor()]),
    "darken + grayscale": Compose([Darken(1.0), Grayscale(1.0), ToTensor()]),
    "full B (random)": targeted_train(),
}


def draw(img_t, target, classes):
    im = to_pil_image(img_t)
    d = ImageDraw.Draw(im)
    for b, l in zip(target["boxes"].tolist(), target["labels"].tolist()):
        d.rectangle(b, outline=(255, 40, 40), width=4)
        d.text((b[0] + 4, b[1] + 4), classes[l - 1], fill=(255, 40, 40))
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/raw/eccv_18_all_images_sm")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/variant_B/augmentation_examples.png"))
    args = p.parse_args()

    random.seed(args.seed)
    ds = CCT20Detection("train", args.image_dir, args.splits_dir)
    # prefer medium-size animals so the effect of the crop/erase is easy to see
    def share(r):
        return max(b["bbox"][2] * b["bbox"][3] for b in r["boxes"]) / (r["width"] * r["height"])
    idx = [i for i, r in enumerate(ds.records) if r["boxes"] and 0.02 <= share(r) <= 0.3]
    random.shuffle(idx)
    fig, axes = plt.subplots(args.rows, len(COLUMNS), figsize=(4 * len(COLUMNS), 3.2 * args.rows))
    n_boxes = 0
    for row, i in enumerate(idx[:args.rows]):
        for col, (name, tf) in enumerate(COLUMNS.items()):
            ds.transform = tf
            img, target = ds[i]
            assert img.shape[0] == 3 and len(target["boxes"]) == len(target["labels"])
            b = target["boxes"]
            if len(b):
                assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
                assert b[:, 0].min() >= 0 and b[:, 1].min() >= 0
                assert b[:, 2].max() <= img.shape[2] + 1 and b[:, 3].max() <= img.shape[1] + 1
            n_boxes += len(b)
            ax = axes[row, col]
            ax.imshow(draw(img, target, ds.classes))
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=12)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=80)
    print(f"augmentation preview OK ({n_boxes} boxes checked) -> {args.out}")


if __name__ == "__main__":
    main()
