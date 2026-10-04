"""Fine-tune Faster R-CNN on CCT20 (baseline A or targeted-augmentation variant B).

Usage:
    python -m src.train --variant A --seed 0 --out runs/A_s0
    python -m src.train --variant B --seed 0 --out runs/B_s0

Same schedule for both variants; only the augmentation (and the wider scale jitter
of B) differs. The best epoch by val mAP@0.5 is saved to <out>/best.pt.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.augment import build_train_transform, eval_transform
from src.data.dataset import CCT20Detection

# shorter-side sizes sampled per batch during training (scale jitter)
SCALES = {"A": (640, 704, 768, 832, 896), "B": (480, 576, 672, 768, 864, 960)}


def build_model(num_classes: int, variant: str = "A", pretrained: bool = True):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None, min_size=SCALES[variant], max_size=1024)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)  # +1 background
    return model


def collate(batch):
    return tuple(zip(*batch))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def val_map(model, loader, device):
    model.eval()
    # at test time use a single fixed size, the middle of the training range
    model.transform.min_size = (800,)
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    for imgs, targets in loader:
        out = model([i.to(device) for i in imgs])
        metric.update([{k: v.cpu() for k, v in o.items()} for o in out],
                      [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets])
    return metric.compute()["map_50"].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["A", "B"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-dir", default="data/raw/eccv_18_all_images_sm")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CCT20Detection("train", args.image_dir, args.splits_dir,
                              transform=build_train_transform(args.variant))
    val_ds = CCT20Detection("val", args.image_dir, args.splits_dir, transform=eval_transform())
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, 8, num_workers=args.workers, collate_fn=collate)

    model = build_model(len(train_ds.classes), args.variant).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, log = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        model.transform.min_size = SCALES[args.variant]
        total = 0.0
        for imgs, targets in train_dl:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items() if k != "image_id"} for t in targets]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()
        m = val_map(model, val_dl, device)
        log.append({"epoch": epoch, "train_loss": total / len(train_dl), "val_map50": m})
        print(log[-1], flush=True)
        if m > best:
            best = m
            torch.save({"model": model.state_dict(), "classes": train_ds.classes,
                        "variant": args.variant, "epoch": epoch}, args.out / "best.pt")
        json.dump(log, open(args.out / "log.json", "w"), indent=1)
    print(f"best val mAP@0.5 = {best:.4f}")


if __name__ == "__main__":
    main()
