"""Fine-tune Faster R-CNN R50-FPN on CCT20 (baseline A or targeted-augmentation variant B).

    python -m src.train --variant A --epochs 20 --seed 0 --out runs/A_s0
    # quick first run (subset of train sequences, fewer epochs)
    python -m src.train --variant A --epochs 4 --train-fraction 0.5 --out runs/A_quick

Settings follow the run table in docs/experimental_plan.md: SGD (momentum 0.9,
weight decay 1e-4), lr 0.01 with linear warm-up and cosine decay, long side 1024,
shorter-side scale jitter 640-896, flip + mild colour jitter.

After each epoch the model is evaluated on a fixed subset of val (`--val-max`
images, whole sequences) and the epoch with the best val mAP@0.5 is saved to
<out>/best.pt. Final threshold selection and test evaluation: src/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import CCT20Detection
from src.data.transforms import baseline_train, eval_transform, targeted_train
from src.metrics import coco_map
from src.models.detector import build_model, set_train_size
from src.predict import collate, run_inference

TRANSFORMS = {"A": baseline_train, "B": targeted_train}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sequence_subset(records: list[dict], fraction: float = 1.0, max_images: int = 0,
                    seed: int = 0) -> list[int]:
    """Indices of a random subset made of whole sequences (no partial bursts)."""
    by_seq = defaultdict(list)
    for i, r in enumerate(records):
        by_seq[r["seq_id"]].append(i)
    seqs = sorted(by_seq)
    random.Random(seed).shuffle(seqs)
    limit = max_images or math.ceil(fraction * len(records))
    out = []
    for s in seqs:
        if len(out) >= limit:
            break
        out.extend(by_seq[s])
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(TRANSFORMS), default="A")
    p.add_argument("--seed", type=int, default=0, help="training seed (data split seed is 42)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="share of train images to use (whole sequences)")
    p.add_argument("--val-max", type=int, default=1500,
                   help="val images used for per-epoch model selection (0 = all)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-dir", default="data/raw/eccv_18_all_images_sm")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--no-pretrained", action="store_true", help="debug only")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_ds = CCT20Detection("train", args.image_dir, args.splits_dir,
                              transform=TRANSFORMS[args.variant]())
    train_ds.records = [r for r in train_ds.records if r["boxes"]]
    if args.train_fraction < 1.0:
        idx = sequence_subset(train_ds.records, args.train_fraction, seed=42)
        train_ds.records = [train_ds.records[i] for i in idx]
    val_ds = CCT20Detection("val", args.image_dir, args.splits_dir, transform=eval_transform())
    val_idx = sequence_subset(val_ds.records, max_images=args.val_max, seed=42) \
        if args.val_max else list(range(len(val_ds)))
    val_records = [val_ds.records[i] for i in val_idx]
    classes = train_ds.classes
    print(f"classes ({len(classes)}): {classes}")
    print(f"train images: {len(train_ds)}, val images for selection: {len(val_idx)}")

    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          collate_fn=collate, drop_last=True, generator=g,
                          pin_memory=device.type == "cuda")

    model = build_model(len(classes), args.variant, pretrained=not args.no_pretrained).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total_iters = args.epochs * len(train_dl)
    warm = min(args.warmup_iters, max(1, total_iters // 10))

    def lr_lambda(it):
        if it < warm:
            return 0.001 + (1 - 0.001) * it / warm
        return 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total_iters - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(classes=classes, train_images=len(train_ds), val_images=len(val_idx),
                  torch=torch.__version__, device=str(device))
    json.dump(config, open(args.out / "config.json", "w"), indent=1)

    best, log = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        set_train_size(model, args.variant)
        t0, total, n = time.time(), 0.0, 0
        for imgs, targets in train_dl:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
                       for t in targets]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = sum(model(imgs, targets).values())
            if not torch.isfinite(loss):
                raise SystemExit(f"loss is {loss.item()} at epoch {epoch}, iter {n}: lower --lr")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            n += 1
            if n % 200 == 0:
                print(f"  epoch {epoch} iter {n}/{len(train_dl)} loss {total / n:.4f}", flush=True)
        train_time = time.time() - t0

        preds = run_inference(model, Subset(val_ds, val_idx), device, workers=args.workers,
                              log_every=0)
        m = coco_map(val_records, preds, classes)
        log.append({"epoch": epoch, "train_loss": total / max(n, 1),
                    "val_map50": m["map50"], "val_map": m["map"],
                    "lr": opt.param_groups[0]["lr"], "train_minutes": round(train_time / 60, 2)})
        print(log[-1], flush=True)
        ckpt = {"model": model.state_dict(), "classes": classes, "variant": args.variant,
                "epoch": epoch, "val_map50": m["map50"]}
        if m["map50"] > best:
            best = m["map50"]
            torch.save(ckpt, args.out / "best.pt")
        json.dump(log, open(args.out / "log.json", "w"), indent=1)
    print(f"best val mAP@0.5 = {best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
