"""Pick the confidence threshold on val, then evaluate once on test.

    python -m src.evaluate --name "Baseline A" \
        --val-preds runs/A_s0/preds_val.json --test-preds runs/A_s0/preds_test.json \
        --out results/baseline_A
    # COCO reference: no CCT20 classes, class-agnostic metrics only
    python -m src.evaluate --name "COCO reference" --agnostic-only \
        --val-preds runs/reference/preds_val.json --test-preds runs/reference/preds_test.json \
        --out results/reference

Writes to --out:
    metrics.json         all numbers (class-aware and class-agnostic)
    per_condition.csv    recall / FP on condition subsets (night, dark, truncated, small)
    threshold_curve.csv  P / R / F1 on val for every threshold
    summary.md           short tables for the report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.metrics import coco_map, match_images, pick_threshold, summarise

CONDITIONS = ["is_night", "is_dark", "truncated", "is_small"]


def load_conditions(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["image_id"]: {c: r[c] == "True" for c in CONDITIONS} for r in csv.DictReader(f)}


def condition_rows(rows: list[dict], tags: dict) -> list[dict]:
    out = []
    for c in CONDITIONS:
        for value in (True, False):
            sub = [r for r in rows if r["image_id"] in tags and tags[r["image_id"]][c] == value]
            if not sub:
                continue
            s = summarise(sub)
            out.append({"condition": c, "value": value, "images": s["images"],
                        "animal_images": s["animal_images"], "empty_images": s["empty_images"],
                        "recall": round(s["recall"], 4), "image_recall": round(s["image_recall"], 4),
                        "precision": round(s["precision"], 4),
                        "fp_per_empty_image": round(s["fp_per_empty_image"], 4)})
    return out


def evaluate(val_recs, test_recs, val_preds, test_preds, classes, agnostic):
    thr, curve = pick_threshold(val_recs, val_preds, classes, agnostic)
    rows = match_images(test_recs, test_preds, classes, thr, agnostic)
    res = {"threshold_from_val": thr, **summarise(rows), **coco_map(test_recs, test_preds, classes, agnostic)}
    return res, rows, curve


def fmt(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--val-preds", required=True)
    p.add_argument("--test-preds", required=True)
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--test-split", default="test")
    p.add_argument("--agnostic-only", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    classes = json.load(open(args.splits_dir / "classes.json"))
    val_recs = json.load(open(args.splits_dir / "val.json"))
    test_recs = json.load(open(args.splits_dir / f"{args.test_split}.json"))
    val_preds = json.load(open(args.val_preds))
    test_preds = json.load(open(args.test_preds))
    tags = load_conditions(args.splits_dir / f"{args.test_split}_conditions.csv")
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = {"name": args.name, "test_split": args.test_split}
    modes = [("agnostic", True)] if args.agnostic_only else [("class_aware", False), ("agnostic", True)]
    cond_all, curves = [], {}
    for mode, agn in modes:
        res, rows, curve = evaluate(val_recs, test_recs, val_preds, test_preds, classes, agn)
        metrics[mode] = res
        curves[mode] = curve
        for r in condition_rows(rows, tags):
            cond_all.append({"mode": mode, **r})
        print(f"[{mode}] thr={res['threshold_from_val']:.2f} mAP@0.5={res['map50']:.4f} "
              f"mAP={res['map']:.4f} P={res['precision']:.3f} R={res['recall']:.3f} "
              f"F1={res['f1']:.3f} FP/empty={res['fp_per_empty_image']:.3f} "
              f"img-recall={res['image_recall']:.3f}")

    json.dump(metrics, open(args.out / "metrics.json", "w"), indent=1)
    if cond_all:
        with open(args.out / "per_condition.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cond_all[0]))
            w.writeheader()
            w.writerows(cond_all)
    with open(args.out / "threshold_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "threshold", "precision", "recall", "f1", "fp_per_empty_image"])
        for mode, curve in curves.items():
            for c in curve:
                w.writerow([mode, c["threshold"], round(c["precision"], 4), round(c["recall"], 4),
                            round(c["f1"], 4), round(c["fp_per_empty_image"], 4)])

    # ---- markdown summary
    lines = [f"# {args.name}: results on `{args.test_split}`", "",
             "| Mode | Threshold (val) | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |",
             "|---|---|---|---|---|---|---|---|---|"]
    for mode, _ in modes:
        r = metrics[mode]
        lines.append(f"| {mode} | {r['threshold_from_val']:.2f} | {fmt(r['map50'])} | {fmt(r['map'])} | "
                     f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | "
                     f"{fmt(r['fp_per_empty_image'])} | {fmt(r['image_recall'])} |")
    if "class_aware" in metrics:
        lines += ["", "AP@0.5 per class:", "", "| Class | AP@0.5 |", "|---|---|"]
        lines += [f"| {c} | {fmt(v)} |" for c, v in metrics["class_aware"]["ap50_per_class"].items()]
    if cond_all:
        mode = modes[0][0]
        lines += ["", f"Condition subsets ({mode}, test threshold from val):", "",
                  "| Condition | Subset | Images | Recall | Image recall | FP / empty image |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['condition']} | {'yes' if r['value'] else 'no'} | {r['images']} | "
                  f"{fmt(r['recall'])} | {fmt(r['image_recall'])} | {fmt(r['fp_per_empty_image'])} |"
                  for r in cond_all if r["mode"] == mode]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
