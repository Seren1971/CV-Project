"""Run a detector on a split and save all detections with score >= 0.05.

    # fine-tuned model
    python -m src.predict --ckpt runs/A_s0/best.pt --split test --out runs/A_s0/preds_test.json
    # untrained COCO reference (COCO animal classes merged into one "animal" class)
    python -m src.predict --coco-reference --split test --out runs/reference/preds_test.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CCT20Detection
from src.data.transforms import eval_transform
from src.models.detector import COCO_ANIMALS, build_model, set_eval_size

SCORE_MIN = 0.05


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=8, workers=2, coco_reference=False,
                  log_every=200) -> dict:
    model.eval()
    set_eval_size(model)
    model.roi_heads.score_thresh = SCORE_MIN
    dl = DataLoader(dataset, batch_size, num_workers=workers, collate_fn=collate)
    preds, done = {}, 0
    for imgs, targets in dl:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model([i.to(device) for i in imgs])
        for t, o in zip(targets, out):
            boxes, scores, labels = o["boxes"].float().cpu(), o["scores"].float().cpu(), o["labels"].cpu()
            if coco_reference:
                keep = torch.tensor([int(l) in COCO_ANIMALS for l in labels], dtype=torch.bool)
                boxes, scores = boxes[keep], scores[keep]
                labels = torch.ones(len(scores), dtype=torch.int64)
            preds[t["image_id"]] = {"size": list(t["size"]),
                                    "boxes": [[round(v, 1) for v in b] for b in boxes.tolist()],
                                    "scores": [round(s, 4) for s in scores.tolist()],
                                    "labels": labels.tolist()}
        done += len(imgs)
        if log_every and done % log_every < len(imgs):
            print(f"  predicted {done}/{len(dataset)}", flush=True)
    return preds


def load_model(ckpt: str | None, coco_reference: bool, device):
    if coco_reference:
        model = build_model(None, pretrained=True)
        classes = None
    else:
        ck = torch.load(ckpt, map_location="cpu")
        model = build_model(len(ck["classes"]), ck.get("variant", "A"), pretrained=False)
        model.load_state_dict(ck["model"])
        classes = ck["classes"]
    return model.to(device), classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--coco-reference", action="store_true")
    p.add_argument("--split", default="test")
    p.add_argument("--image-dir", default="data/raw/eccv_18_all_images_sm")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-images", type=int, default=0, help="first N images only (smoke test)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not args.coco_reference and not args.ckpt:
        p.error("--ckpt or --coco-reference is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.ckpt, args.coco_reference, device)
    ds = CCT20Detection(args.split, args.image_dir, args.splits_dir, transform=eval_transform())
    if args.max_images:
        ds.records = ds.records[:args.max_images]
    preds = run_inference(model, ds, device, args.batch_size, args.workers, args.coco_reference)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(args.out, "w"))
    print(f"{len(preds)} images -> {args.out}")


if __name__ == "__main__":
    main()
