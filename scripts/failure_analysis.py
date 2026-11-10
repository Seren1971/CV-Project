"""Select representative failure (and success) cases on `test` and render them.

    python scripts/failure_analysis.py \
        --preds C:/Users/spesh/cv-p16-data/runs/A12_s0/preds_test.json \
        --results results/baseline_A12 \
        --image-dir C:/Users/spesh/cv-p16-data/raw/eccv_18_all_images_sm \
        --out results/failure_cases

Case categories (at least one image each):
    missed_small       GT box tagged `is_small`, no detection at IoU >= 0.5
    missed_truncated   GT box tagged `truncated`, no detection at IoU >= 0.5
    class_confusion    detection with IoU >= 0.5 to a GT box of another class
    false_positive     empty frame with a high-confidence detection
    success_day/night  clean correct detections (for contrast and the demo)

Green = ground truth, red = prediction (label + score), orange = wrong-class prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.dataset import scaled_boxes
from src.metrics import iou_matrix

GREEN, RED, ORANGE = (60, 220, 60), (255, 60, 40), (255, 170, 0)


def load_tags(path: Path) -> dict:
    with open(path) as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def per_image_detail(recs, preds, classes, thr):
    """Per image: matched GT, missed GT, FP preds and wrong-class preds at IoU >= 0.5."""
    cidx = {c: i for i, c in enumerate(classes)}
    out = {}
    for rec in recs:
        p = preds.get(rec["image_id"])
        if p is None:
            continue
        w, h = p["size"]
        gb = np.array([[x, y, x + bw, y + bh] for x, y, bw, bh in scaled_boxes(rec, w, h)],
                      dtype=np.float32).reshape(-1, 4)
        gl = np.array([cidx[b["category"]] + 1 for b in rec["boxes"]], dtype=np.int64)
        pb = np.array(p["boxes"], dtype=np.float32).reshape(-1, 4)
        ps = np.array(p["scores"], dtype=np.float32)
        pl = np.array(p["labels"], dtype=np.int64)
        keep = np.where(ps >= thr)[0]
        keep = keep[np.argsort(-ps[keep])]
        iou = iou_matrix(pb[keep], gb)
        matched, pairs = np.zeros(len(gb), bool), []
        wrong, fp = [], []
        for k, d in enumerate(keep):
            if len(gb) == 0:
                fp.append(d)
                continue
            ok = (gl == pl[d]) & ~matched & (iou[k] >= 0.5)
            if ok.any():
                j = np.where(ok)[0][iou[k][ok].argmax()]
                matched[j] = True
                pairs.append((d, j))
                continue
            anyc = (~matched) & (iou[k] >= 0.5)     # right place, wrong class
            if anyc.any():
                j = np.where(anyc)[0][iou[k][anyc].argmax()]
                matched[j] = True                  # still counts as localised
                wrong.append((d, j))
            else:
                fp.append(d)
        out[rec["image_id"]] = {"rec": rec, "size": (w, h), "gb": gb, "gl": gl,
                                "pb": pb, "ps": ps, "pl": pl, "keep": keep,
                                "missed": np.where(~matched)[0], "pairs": pairs,
                                "wrong": wrong, "fp": fp}
    return out


def render(det, classes, path: Path, title: str):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    sx, sy = img.width / det["size"][0], img.height / det["size"][1]
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.width // 70))
    for b, l in zip(det["gb"], det["gl"]):                       # GT: green
        x1, y1, x2, y2 = b * [sx, sy, sx, sy]
        d.rectangle([x1, y1, x2, y2], outline=GREEN, width=3)
        d.text((x1 + 3, y1 + 3), f"GT {classes[l - 1]}", fill=GREEN, font=font)
    for k in det["keep"]:                                        # predictions
        x1, y1, x2, y2 = det["pb"][k] * [sx, sy, sx, sy]
        wrong = any(k == dk for dk, _ in det["wrong"])
        col = ORANGE if wrong else RED
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        d.text((x1 + 3, y2 - 18), f"{classes[det['pl'][k] - 1]} {det['ps'][k]:.2f}",
               fill=col, font=font)
    d.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    d.text((6, 6), title, fill=(255, 255, 255), font=font)
    return img


def pick(cands, key, reverse=True):
    return sorted(cands, key=key, reverse=reverse)[0] if cands else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True)
    p.add_argument("--results", type=Path, required=True,
                   help="results/<model> dir with metrics.json (for the val threshold)")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=Path("results/failure_cases"))
    args = p.parse_args()

    classes = json.load(open(args.splits_dir / "classes.json"))
    recs = json.load(open(args.splits_dir / f"{args.split}.json"))
    tags = load_tags(args.splits_dir / f"{args.split}_conditions.csv")
    preds = json.load(open(args.preds))
    thr = json.load(open(args.results / "metrics.json"))["class_aware"]["threshold_from_val"]
    recs = [r for r in recs if r["image_id"] in preds]
    det = per_image_detail(recs, preds, classes, thr)
    print(f"{len(det)} images, threshold {thr}")

    # ---- error taxonomy over the whole test set (docs/experimental_plan.md)
    tax = {"correct": 0, "class_confusion": 0, "background_fp": 0, "missed": 0, "gt": 0}
    for d in det.values():
        tax["correct"] += len(d["pairs"])
        tax["class_confusion"] += len(d["wrong"])
        tax["background_fp"] += len(d["fp"])
        tax["missed"] += len(d["missed"])
        tax["gt"] += len(d["gb"])
    n_pred = sum(tax[k] for k in ("correct", "class_confusion", "background_fp"))
    print("error taxonomy (test, thresholded):")
    print(f"  correct           {tax['correct']:6d}  ({tax['correct'] / max(n_pred, 1):.1%} of detections)")
    print(f"  class confusion   {tax['class_confusion']:6d}  ({tax['class_confusion'] / max(n_pred, 1):.1%} of detections)")
    print(f"  background FP     {tax['background_fp']:6d}  ({tax['background_fp'] / max(n_pred, 1):.1%} of detections)")
    print(f"  missed animals    {tax['missed']:6d}  ({tax['missed'] / max(tax['gt'], 1):.1%} of ground truth)")

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    cases = {}
    cases["missed_small"] = pick(
        [d for i, d in det.items()
         if tags.get(i, {}).get("is_small") == "True" and len(d["missed"])],
        lambda d: max(area(d["gb"][j]) for j in d["missed"]))
    cases["missed_truncated"] = pick(
        [d for i, d in det.items()
         if tags.get(i, {}).get("truncated") == "True" and len(d["missed"])],
        lambda d: max(area(d["gb"][j]) for j in d["missed"]))
    cases["class_confusion"] = pick(
        [d for d in det.values() if d["wrong"]],
        lambda d: max(d["ps"][k] for k, _ in d["wrong"]))
    cases["false_positive"] = pick(
        [d for d in det.values() if len(d["gb"]) == 0 and d["fp"]],
        lambda d: max(d["ps"][k] for k in d["fp"]))
    good = [d for d in det.values() if len(d["gb"]) >= 1 and not len(d["missed"])
            and not d["wrong"] and not d["fp"]]
    cases["success_night"] = pick(
        [d for i, d in det.items() if d in good and tags.get(i, {}).get("is_night") == "True"],
        lambda d: len(d["gb"]))
    cases["success_day"] = pick(
        [d for i, d in det.items() if d in good and tags.get(i, {}).get("is_night") == "False"],
        lambda d: len(d["gb"]))

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, d in cases.items():
        if d is None:
            print(f"!! no case found for {name}")
            continue
        iid = d["rec"]["image_id"]
        img = render(d, classes, Path(args.image_dir) / d["rec"]["file_name"], name)
        img.save(args.out / f"{name}.jpg", quality=92)
        manifest[name] = {"image_id": iid, "file_name": d["rec"]["file_name"],
                          "n_gt": len(d["gb"]), "n_pred": len(d["keep"]),
                          "missed_gt": len(d["missed"]), "wrong_class": len(d["wrong"]),
                          "tags": {k: tags.get(iid, {}).get(k) for k in
                                   ("is_night", "is_dark", "is_small", "truncated")}}
        print(f"{name}: {iid} gt={len(d['gb'])} pred={len(d['keep'])} "
              f"missed={len(d['missed'])} wrong={len(d['wrong'])}")
    json.dump({"threshold": thr, "taxonomy": tax, "cases": manifest},
              open(args.out / "manifest.json", "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
