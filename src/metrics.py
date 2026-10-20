"""Detection metrics used in docs/experimental_plan.md.

Predictions format (written by src/predict.py), one entry per image:
    {image_id: {"size": [w, h], "boxes": [[x1, y1, x2, y2], ...],
                "scores": [...], "labels": [...]}}
Labels are 1..K (index in classes.json + 1). Boxes are pixels of the image on disk.

Metrics:
    coco_map         mAP@0.5, mAP@[0.5:0.95] and per-class AP@0.5 (pycocotools)
    match_images     one-to-one greedy matching by confidence at IoU >= 0.5
    summarise        precision, recall, F1, false positives per empty image,
                     image-level recall (animal images with >= 1 correct box)
    pick_threshold   confidence threshold with the best F1 (use on val only)
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

from src.data.dataset import scaled_boxes

IOU_TP = 0.5


def gt_boxes(rec: dict, size, classes_idx: dict, agnostic: bool):
    """GT boxes (xyxy, pixels of the image on disk) and labels for one record."""
    w, h = size
    boxes = np.array([[x, y, x + bw, y + bh] for x, y, bw, bh in scaled_boxes(rec, w, h)],
                     dtype=np.float32).reshape(-1, 4)
    labels = np.array([1 if agnostic else classes_idx[b["category"]] + 1 for b in rec["boxes"]],
                      dtype=np.int64)
    return boxes, labels


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _pred_arrays(p: dict, agnostic: bool):
    boxes = np.array(p["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.array(p["scores"], dtype=np.float32)
    labels = np.ones(len(scores), dtype=np.int64) if agnostic else np.array(p["labels"], dtype=np.int64)
    return boxes, scores, labels


# ---------------------------------------------------------------- COCO mAP
def coco_map(records: list[dict], preds: dict, classes: list[str], agnostic: bool = False) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cidx = {c: i for i, c in enumerate(classes)}
    cats = [{"id": 1, "name": "animal"}] if agnostic else [
        {"id": i + 1, "name": c} for i, c in enumerate(classes)]
    images, anns, dets = [], [], []
    for n, rec in enumerate(r for r in records if r["image_id"] in preds):
        p = preds[rec["image_id"]]
        w, h = p["size"]
        images.append({"id": n, "width": w, "height": h})
        gb, gl = gt_boxes(rec, (w, h), cidx, agnostic)
        for b, l in zip(gb, gl):
            anns.append({"id": len(anns) + 1, "image_id": n, "category_id": int(l), "iscrowd": 0,
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "area": float((b[2] - b[0]) * (b[3] - b[1]))})
        pb, ps, pl = _pred_arrays(p, agnostic)
        for b, s, l in zip(pb, ps, pl):
            dets.append({"image_id": n, "category_id": int(l), "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": cats}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        if not dets:
            return {"map50": 0.0, "map": 0.0, "ap50_per_class": {}}
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # precision: [T(iou), R(recall), K(class), A(area), M(maxDets)]
    prec = ev.eval["precision"][0, :, :, 0, -1]          # IoU=0.5, all areas, 100 dets
    per_class = {}
    for k, cat in enumerate(cats):
        col = prec[:, k]
        col = col[col > -1]
        per_class[cat["name"]] = float(col.mean()) if col.size else float("nan")
    return {"map50": float(ev.stats[1]), "map": float(ev.stats[0]), "ap50_per_class": per_class}


# ---------------------------------------------------------------- matching
def match_images(records: list[dict], preds: dict, classes: list[str], thr: float,
                 agnostic: bool = False) -> list[dict]:
    """Per-image counts at confidence threshold `thr`."""
    cidx = {c: i for i, c in enumerate(classes)}
    rows = []
    for rec in records:
        p = preds.get(rec["image_id"])
        if p is None:
            continue
        gb, gl = gt_boxes(rec, p["size"], cidx, agnostic)
        pb, ps, pl = _pred_arrays(p, agnostic)
        keep = np.where(ps >= thr)[0]
        keep = keep[np.argsort(-ps[keep])]
        iou = iou_matrix(pb[keep], gb)
        matched = np.zeros(len(gb), dtype=bool)
        tp = 0
        for d in range(len(keep)):
            if len(gb) == 0:
                break
            ok = (gl == pl[keep[d]]) & ~matched & (iou[d] >= IOU_TP)
            if ok.any():
                j = np.where(ok)[0][iou[d][ok].argmax()]
                matched[j] = True
                tp += 1
        gt_cls = [b["category"] for b in rec["boxes"]]
        rows.append({"image_id": rec["image_id"], "seq_id": rec.get("seq_id", rec["image_id"]),
                     "empty": len(gb) == 0,
                     "n_gt": len(gb), "n_pred": len(keep), "tp": tp,
                     "fp": len(keep) - tp, "fn": int((~matched).sum()),
                     "gt_matched": matched.tolist(),
                     # per ground-truth class: [matched, total]
                     "by_class": {c: [sum(1 for k, cc in enumerate(gt_cls) if cc == c and matched[k]),
                                      gt_cls.count(c)] for c in set(gt_cls)}})
    return rows


def summarise(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    animal = [r for r in rows if not r["empty"]]
    empty = [r for r in rows if r["empty"]]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": len(rows), "animal_images": len(animal), "empty_images": len(empty),
        "precision": precision, "recall": recall, "f1": f1,
        "fp_per_empty_image": sum(r["fp"] for r in empty) / len(empty) if empty else float("nan"),
        "empty_images_with_fp": sum(r["fp"] > 0 for r in empty) / len(empty) if empty else float("nan"),
        "image_recall": sum(r["tp"] > 0 for r in animal) / len(animal) if animal else float("nan"),
    }


def pick_threshold(records, preds, classes, agnostic=False,
                   grid=np.round(np.arange(0.05, 0.96, 0.05), 2)) -> tuple[float, list[dict]]:
    curve = []
    for t in grid:
        s = summarise(match_images(records, preds, classes, float(t), agnostic))
        curve.append({"threshold": float(t), **s})
    best = max(curve, key=lambda c: c["f1"])
    return best["threshold"], curve
