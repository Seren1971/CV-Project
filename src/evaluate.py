"""Inference, threshold selection and per-condition error analysis.

Three steps (see docs/experimental_plan.md):

    # 1. detections for a split (saved with score >= 0.05)
    python -m src.evaluate predict --ckpt runs/A_s0/best.pt --split val  --out runs/A_s0/preds_val.json
    python -m src.evaluate predict --ckpt runs/A_s0/best.pt --split test --out runs/A_s0/preds_test.json

    # 2. confidence threshold = max F1 on val, then kept fixed for test
    python -m src.evaluate threshold --preds runs/A_s0/preds_val.json --split val

    # 3. overall + per-condition metrics, error types, bootstrap CIs
    python -m src.evaluate report --preds runs/A_s0/preds_test.json --split test \
        --threshold 0.5 --out runs/A_s0/report_test.csv

Add --agnostic to ignore class labels ("is there an animal, and where").

Error types for each detection above the threshold (checked in this order):
    tp          IoU >= 0.5 with an unmatched GT box of the same class
    duplicate   IoU >= 0.5 with a GT box of the same class that is already matched
    confusion   IoU >= 0.5 with a GT box of another class
    localisation same class, 0.1 <= IoU < 0.5
    background  IoU < 0.1 with every GT box (on empty frames: vegetation false alarm)
and `fn` = GT boxes no detection matched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou

from src.data.dataset import scaled_boxes

IOU_TP, IOU_LOC = 0.5, 0.1
ERR_TYPES = ["tp", "duplicate", "confusion", "localisation", "background", "fn"]
# condition tags from tag_conditions.py (+ optional manual tags, see load_tags)
AUTO_TAGS = ["is_night", "is_dark", "truncated", "is_small"]


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(args):
    from torch.utils.data import DataLoader

    from src.augment import eval_transform
    from src.data.dataset import CCT20Detection
    from src.train import build_model, collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(len(ck["classes"]), ck["variant"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.transform.min_size = (800,)
    model.roi_heads.score_thresh = 0.05
    model.to(device).eval()

    ds = CCT20Detection(args.split, args.image_dir, args.splits_dir, transform=eval_transform())
    dl = DataLoader(ds, 8, num_workers=4, collate_fn=collate)
    preds = {}
    for imgs, targets in dl:
        out = model([i.to(device) for i in imgs])
        for t, o in zip(targets, out):
            preds[t["image_id"]] = {"boxes": o["boxes"].cpu().tolist(),
                                    "scores": o["scores"].cpu().tolist(),
                                    "labels": o["labels"].cpu().tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(args.out, "w"))
    print(f"{len(preds)} images -> {args.out}")


# ---------------------------------------------------------------- matching
def gt_for(rec, classes_idx, agnostic):
    """GT boxes (xyxy) and labels in the coordinate frame of the image on disk."""
    # predictions are in pixels of the "_sm" image; reuse the rescaling of the dataset
    # (record width/height -> actual size is taken from the saved image size).
    boxes = [[x, y, x + w, y + h] for x, y, w, h in scaled_boxes(rec, rec["_w"], rec["_h"])]
    labels = [0 if agnostic else classes_idx[b["category"]] + 1 for b in rec["boxes"]]
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.int64))


def match_image(gt_b, gt_l, pred, thr, agnostic):
    """Count error types for one image."""
    c = dict.fromkeys(ERR_TYPES, 0)
    keep = [i for i, s in enumerate(pred["scores"]) if s >= thr]
    keep.sort(key=lambda i: -pred["scores"][i])
    pb = torch.tensor([pred["boxes"][i] for i in keep], dtype=torch.float32).reshape(-1, 4)
    pl = torch.tensor([0 if agnostic else pred["labels"][i] for i in keep], dtype=torch.int64)
    matched = torch.zeros(len(gt_b), dtype=torch.bool)
    iou = box_iou(pb, gt_b) if len(pb) and len(gt_b) else torch.zeros(len(pb), len(gt_b))
    for d in range(len(pb)):
        if len(gt_b) == 0:
            c["background"] += 1
            continue
        same = gt_l == pl[d]
        free = same & ~matched
        if free.any() and iou[d][free].max() >= IOU_TP:
            j = torch.where(free)[0][iou[d][free].argmax()]
            matched[j] = True
            c["tp"] += 1
        elif same.any() and iou[d][same].max() >= IOU_TP:
            c["duplicate"] += 1
        elif (~same).any() and iou[d][~same].max() >= IOU_TP:
            c["confusion"] += 1
        elif same.any() and iou[d][same].max() >= IOU_LOC:
            c["localisation"] += 1
        else:
            c["background"] += 1
    c["fn"] = int((~matched).sum())
    return c


def per_image_table(records, preds, classes, thr, agnostic):
    cidx = {c: i for i, c in enumerate(classes)}
    rows = []
    for r in records:
        gt_b, gt_l = gt_for(r, cidx, agnostic)
        c = match_image(gt_b, gt_l, preds[r["image_id"]], thr, agnostic)
        rows.append({"image_id": r["image_id"], "seq_id": r["seq_id"], "label": r["label"],
                     "n_gt": len(gt_b), **c})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- aggregation
def summarise(df: pd.DataFrame) -> dict:
    s = df[ERR_TYPES + ["n_gt"]].sum()
    det = s.tp + s.duplicate + s.confusion + s.localisation + s.background
    prec = s.tp / det if det else float("nan")
    rec = s.tp / s.n_gt if s.n_gt else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    animal = df[df.n_gt > 0]
    empty = df[df.n_gt == 0]
    return {"images": len(df), "gt_boxes": int(s.n_gt), "precision": prec, "recall": rec, "f1": f1,
            "image_recall": (animal.tp > 0).mean() if len(animal) else float("nan"),
            "fp_per_empty_img": empty.background.mean() if len(empty) else float("nan"),
            **{k: int(s[k]) for k in ERR_TYPES}}


def bootstrap_ci(df, keys=("precision", "recall", "image_recall"), n=500, seed=42):
    """95% CI, resampling whole sequences (frames of one burst are near-identical)."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("seq_id")]
    vals = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        s = summarise(pd.concat([groups[i] for i in pick]))
        for k in keys:
            vals[k].append(s[k])
    return {f"{k}_ci": f"[{np.nanpercentile(v, 2.5):.3f}, {np.nanpercentile(v, 97.5):.3f}]"
            for k, v in vals.items()}


def map_for(records, preds, classes, agnostic):
    from torchmetrics.detection import MeanAveragePrecision
    cidx = {c: i for i, c in enumerate(classes)}
    m = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    for r in records:
        gb, gl = gt_for(r, cidx, agnostic)
        p = preds[r["image_id"]]
        pl = torch.zeros(len(p["labels"]), dtype=torch.int64) if agnostic else torch.tensor(p["labels"], dtype=torch.int64)
        m.update([{"boxes": torch.tensor(p["boxes"], dtype=torch.float32).reshape(-1, 4),
                   "scores": torch.tensor(p["scores"]), "labels": pl}],
                 [{"boxes": gb, "labels": gl}])
    out = m.compute()
    return out["map_50"].item(), out["map"].item()


def load_records(args):
    from PIL import Image
    recs = json.load(open(Path(args.splits_dir) / f"{args.split}.json"))
    for r in recs:  # actual size on disk, needed to rescale boxes
        with Image.open(Path(args.image_dir) / r["file_name"]) as im:
            r["_w"], r["_h"] = im.size
    classes = json.load(open(Path(args.splits_dir) / "classes.json"))
    return recs, classes


def load_tags(args, df):
    """Merge auto tags (tag_conditions.py) and optional manual tags.

    Manual file data/splits/<split>_manual_tags.csv: image_id, occluded (0/1), pose
    (free text, e.g. lying / curled / back_view / jumping). Missing file = skipped.
    """
    cond = pd.read_csv(Path(args.splits_dir) / f"{args.split}_conditions.csv")
    df = df.merge(cond.drop(columns=["label"]), on="image_id", how="left")
    tags = list(AUTO_TAGS)
    manual = Path(args.splits_dir) / f"{args.split}_manual_tags.csv"
    if manual.exists():
        m = pd.read_csv(manual)
        df = df.merge(m, on="image_id", how="left")
        df["occluded"] = df["occluded"].fillna(0).astype(bool)
        tags.append("occluded")
        for pose in sorted(m["pose"].dropna().unique()):
            df[f"pose_{pose}"] = df["pose"] == pose
            tags.append(f"pose_{pose}")
    return df, tags


# ---------------------------------------------------------------- commands
def threshold_cmd(args):
    recs, classes = load_records(args)
    preds = json.load(open(args.preds))
    best = (-1, 0.5)
    for thr in np.arange(0.1, 0.96, 0.05):
        f1 = summarise(per_image_table(recs, preds, classes, float(thr), args.agnostic))["f1"]
        print(f"thr {thr:.2f}  F1 {f1:.3f}")
        if f1 > best[0]:
            best = (f1, float(thr))
    print(f"best threshold {best[1]:.2f} (F1 {best[0]:.3f})")


def report_cmd(args):
    recs, classes = load_records(args)
    preds = json.load(open(args.preds))
    df = per_image_table(recs, preds, classes, args.threshold, args.agnostic)
    df, tags = load_tags(args, df)
    by_id = {r["image_id"]: r for r in recs}

    rows = []

    def add(name, sub):
        row = {"subset": name, **summarise(sub), **bootstrap_ci(sub)}
        sub_recs = [by_id[i] for i in sub.image_id]
        row["map50"], row["map50_95"] = map_for(sub_recs, preds, classes, args.agnostic)
        rows.append(row)

    add("all", df)
    add("empty_frames", df[df.n_gt == 0])
    for t in tags:
        mask = df[t].fillna(False).astype(bool)
        add(f"{t}=1", df[mask & (df.n_gt > 0)])
        add(f"{t}=0", df[~mask & (df.n_gt > 0)])
    # overlap between conditions: do not blame one factor for the whole drop
    for a, b in [("is_night", "truncated"), ("is_night", "is_small"), ("truncated", "is_small")]:
        add(f"{a}&{b}", df[df[a].astype(bool) & df[b].astype(bool) & (df.n_gt > 0)])
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(out[["subset", "images", "map50", "precision", "recall", "f1", "image_recall",
               "fp_per_empty_img"]].round(3).to_string(index=False))
    print(f"written to {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("predict", "threshold", "report"):
        s = sub.add_parser(name)
        s.add_argument("--split", default="test")
        s.add_argument("--splits-dir", default="data/splits")
        s.add_argument("--image-dir", default="data/raw/eccv_18_all_images_sm")
        s.add_argument("--agnostic", action="store_true")
        if name == "predict":
            s.add_argument("--ckpt", required=True)
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--preds", required=True)
        if name == "report":
            s.add_argument("--threshold", type=float, required=True)
            s.add_argument("--out", required=True)
    args = p.parse_args()
    {"predict": predict, "threshold": threshold_cmd, "report": report_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
