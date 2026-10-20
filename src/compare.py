"""Compare baseline A and variant B on the same test images.

    python -m src.compare --a-preds runs/A_s0/preds_test.json --b-preds runs/B_s0/preds_test.json \
        --a-results results/baseline_A --b-results results/variant_B --out results/comparison_A_vs_B.md

Uses each model's own val-selected threshold (from <results>/metrics.json), per-class
matching at IoU >= 0.5. Differences B - A on recall / precision / FP per empty frame get a
95% bootstrap interval obtained by resampling camera-trap *sequences* (bursts of
near-identical frames), paired on the same images.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.dataset import scaled_boxes
from src.metrics import match_images

CONDITIONS = [("is_night", "Night (IR)"), ("is_dark", "Dark"), ("truncated", "Truncated"),
              ("is_small", "Small")]
SMALL_SHARE = 0.01     # same thresholds as src/data/tag_conditions.py
BORDER_PX = 2


def box_flags(rec: dict, size) -> list[dict]:
    """Per ground-truth box: is it small / truncated? (box-level, not image-level)"""
    w, h = size
    out = []
    for x, y, bw, bh in scaled_boxes(rec, w, h):
        out.append({"small": bw * bh / (w * h) < SMALL_SHARE,
                    "truncated": x <= BORDER_PX or y <= BORDER_PX
                    or x + bw >= w - BORDER_PX or y + bh >= h - BORDER_PX})
    return out


def box_seq_table(recs, preds, rows_a, rows_b, flag, value):
    """Per-sequence [tp_a, tp_b, n_gt] over boxes whose flag == value."""
    agg = defaultdict(lambda: np.zeros(3))
    for rec, ra, rb in zip(recs, rows_a, rows_b):
        if not rec["boxes"]:
            continue
        flags = box_flags(rec, preds[rec["image_id"]]["size"])
        a = agg[ra["seq_id"]]
        for k, f in enumerate(flags):
            if f[flag] == value:
                a += [ra["gt_matched"][k], rb["gt_matched"][k], 1]
    return np.array(list(agg.values())) if agg else np.zeros((0, 3))


def box_recall_ci(t, n=1000, seed=0):
    if len(t) < 2:
        return None
    rng = np.random.default_rng(seed)
    s = rng.multinomial(len(t), np.full(len(t), 1 / len(t)), size=n) @ t
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (s[:, 1] - s[:, 0]) / s[:, 2]
    return tuple(np.nanpercentile(d, [2.5, 97.5]))


def load_tags(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["image_id"]: {c: r[c] == "True" for c, _ in CONDITIONS} for r in csv.DictReader(f)}


def seq_table(rows_a, rows_b, keep):
    """Per-sequence sums for the images selected by keep(image_id) -> arrays [n_seq, k]."""
    agg = defaultdict(lambda: np.zeros(8))
    for ra, rb in zip(rows_a, rows_b):
        assert ra["image_id"] == rb["image_id"]
        if not keep(ra["image_id"]):
            continue
        a = agg[ra["seq_id"]]
        a += [ra["tp"], rb["tp"], ra["tp"] + ra["fn"], ra["n_pred"], rb["n_pred"],
              ra["fp"] if ra["empty"] else 0, rb["fp"] if rb["empty"] else 0, ra["empty"]]
    return np.array(list(agg.values())) if agg else np.zeros((0, 8))


def stats(t):
    tpa, tpb, ngt, npa, npb, fpea, fpeb, ne = t.sum(axis=0) if len(t) else np.zeros(8)
    div = lambda x, y: x / y if y else float("nan")
    return {"recall_a": div(tpa, ngt), "recall_b": div(tpb, ngt),
            "prec_a": div(tpa, npa), "prec_b": div(tpb, npb),
            "fpe_a": div(fpea, ne), "fpe_b": div(fpeb, ne), "n_gt": int(ngt), "n_empty": int(ne)}


def bootstrap(t, n=1000, seed=0):
    """95% intervals for B - A of recall, precision and FP per empty image."""
    if len(t) < 2:
        return {}
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(t), np.full(len(t), 1 / len(t)), size=n)   # [n, n_seq]
    s = w @ t                                                            # [n, 8]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = {"recall": (s[:, 1] - s[:, 0]) / s[:, 2],
             "prec": s[:, 1] / s[:, 4] - s[:, 0] / s[:, 3],
             "fpe": (s[:, 6] - s[:, 5]) / s[:, 7]}
    return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).any() else None
            for k, v in d.items()}


def f3(x):
    return "n/a" if x is None or x != x else f"{x:.3f}"


def delta(b, a, ci=None):
    if a != a or b != b:
        return "n/a"
    s = f"{b - a:+.3f}"
    if ci:
        s += f" [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        if ci[0] > 0 or ci[1] < 0:
            s = f"**{s}**"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-preds", required=True)
    p.add_argument("--b-preds", required=True)
    p.add_argument("--a-results", type=Path, default=Path("results/baseline_A"))
    p.add_argument("--b-results", type=Path, default=Path("results/variant_B"))
    p.add_argument("--extra-results", nargs="*", default=[],
                   help="label=dir pairs for extra rows of the main table (e.g. test_cis runs)")
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--split", default="test")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("results/comparison_A_vs_B.md"))
    args = p.parse_args()

    classes = json.load(open(args.splits_dir / "classes.json"))
    recs = json.load(open(args.splits_dir / f"{args.split}.json"))
    tags = load_tags(args.splits_dir / f"{args.split}_conditions.csv")
    ma = json.load(open(args.a_results / "metrics.json"))
    mb = json.load(open(args.b_results / "metrics.json"))
    pa, pb = json.load(open(args.a_preds)), json.load(open(args.b_preds))
    recs = [r for r in recs if r["image_id"] in pa and r["image_id"] in pb]
    ra = match_images(recs, pa, classes, ma["class_aware"]["threshold_from_val"])
    rb = match_images(recs, pb, classes, mb["class_aware"]["threshold_from_val"])

    L = [f"# Baseline A vs variant B on `{args.split}`", "",
         "Same model, data, epochs and seed; only the training augmentation differs. "
         "Thresholds are chosen on val for each model. Intervals: 95% bootstrap over "
         f"sequences ({args.n_boot} resamples), paired on the same images; **bold** = interval excludes 0.", ""]

    # ---- main table
    L += ["## Overall", "",
          "| Model | Mode | mAP@0.5 | mAP@[.5:.95] | Precision | Recall | F1 | FP / empty image | Image recall |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = [("A", ma), ("B", mb)]
    for item in args.extra_results:
        label, d = item.split("=", 1)
        rows.append((label, json.load(open(Path(d) / "metrics.json"))))
    for mode in ["class_aware", "agnostic"]:
        for name, m in rows:
            if mode not in m:
                continue
            r = m[mode]
            L.append(f"| {name} | {mode.replace('_', '-')} | {f3(r['map50'])} | {f3(r['map'])} | "
                     f"{f3(r['precision'])} | {f3(r['recall'])} | {f3(r['f1'])} | "
                     f"{f3(r['fp_per_empty_image'])} | {f3(r['image_recall'])} |")
    allt = seq_table(ra, rb, lambda i: True)
    st, ci = stats(allt), bootstrap(allt, args.n_boot)
    L += ["", f"Difference B - A on {args.split} (per class): recall {delta(st['recall_b'], st['recall_a'], ci.get('recall'))}, "
          f"precision {delta(st['prec_b'], st['prec_a'], ci.get('prec'))}, "
          f"FP per empty image {delta(st['fpe_b'], st['fpe_a'], ci.get('fpe'))}.", ""]

    # ---- per class AP
    L += ["## AP@0.5 per class", "", "| Class | A | B | B - A |", "|---|---|---|---|"]
    apa, apb = ma["class_aware"]["ap50_per_class"], mb["class_aware"]["ap50_per_class"]
    for c in classes:
        L.append(f"| {c} | {f3(apa.get(c))} | {f3(apb.get(c))} | {delta(apb.get(c, float('nan')), apa.get(c, float('nan')))} |")

    # ---- conditions
    if tags:
        L += ["", "## Condition subsets (recall, per class)", "",
              "Gap = recall on the rest of the test set minus recall on the subset (smaller is better).", "",
              "| Condition | Animal boxes | Recall A | Recall B | B - A [95% CI] | Gap A | Gap B |",
              "|---|---|---|---|---|---|---|"]
        for c, label in CONDITIONS:
            sub = seq_table(ra, rb, lambda i, c=c: i in tags and tags[i][c])
            rest = seq_table(ra, rb, lambda i, c=c: i in tags and not tags[i][c])
            s, r = stats(sub), stats(rest)
            L.append(f"| {label} | {s['n_gt']} | {f3(s['recall_a'])} | {f3(s['recall_b'])} | "
                     f"{delta(s['recall_b'], s['recall_a'], bootstrap(sub, args.n_boot).get('recall'))} | "
                     f"{f3(r['recall_a'] - s['recall_a'])} | {f3(r['recall_b'] - s['recall_b'])} |")
        L += ["", "Note: the rows above are image-level subsets (all animals in images that contain at least "
              "one animal with the condition). Box-level recall for the individual animals:", "",
              "| Animals | Boxes | Recall A | Recall B | B - A [95% CI] | Other boxes: A | Other boxes: B |",
              "|---|---|---|---|---|---|---|"]
        for flag, label in [("small", "Small boxes (< 1% of image)"), ("truncated", "Truncated boxes (touch border)")]:
            sub = box_seq_table(recs, pa, ra, rb, flag, True)
            rest = box_seq_table(recs, pa, ra, rb, flag, False)
            ss, rr = sub.sum(axis=0), rest.sum(axis=0)
            L.append(f"| {label} | {int(ss[2])} | {f3(ss[0] / ss[2])} | {f3(ss[1] / ss[2])} | "
                     f"{delta(ss[1] / ss[2], ss[0] / ss[2], box_recall_ci(sub, args.n_boot))} | "
                     f"{f3(rr[0] / rr[2])} | {f3(rr[1] / rr[2])} |")
        L += ["", "False alarms on empty frames (FP per empty image):", "",
              "| Empty frames | Count | A | B | B - A [95% CI] |", "|---|---|---|---|---|"]
        for label, keep in [("all", lambda i: True),
                            ("night", lambda i: i in tags and tags[i]["is_night"]),
                            ("day", lambda i: i in tags and not tags[i]["is_night"])]:
            t = seq_table(ra, rb, keep)
            s = stats(t)
            L.append(f"| {label} | {s['n_empty']} | {f3(s['fpe_a'])} | {f3(s['fpe_b'])} | "
                     f"{delta(s['fpe_b'], s['fpe_a'], bootstrap(t, args.n_boot).get('fpe'))} |")

        # ---- night / day per class
        L += ["", "## Recall per class, night vs day", "",
              "Checks whether the night/day difference comes from the species mix.", "",
              "| Class | Night boxes | Day boxes | A night | A day | B night | B day |",
              "|---|---|---|---|---|---|---|"]
        for c in classes:
            cnt = {}
            for name, rows_ in [("a", ra), ("b", rb)]:
                for night in (True, False):
                    m = n = 0
                    for r in rows_:
                        if r["image_id"] in tags and tags[r["image_id"]]["is_night"] == night and c in r["by_class"]:
                            m += r["by_class"][c][0]
                            n += r["by_class"][c][1]
                    cnt[(name, night)] = (m, n)
            rec = lambda k: cnt[k][0] / cnt[k][1] if cnt[k][1] else float("nan")
            L.append(f"| {c} | {cnt[('a', True)][1]} | {cnt[('a', False)][1]} | {f3(rec(('a', True)))} | "
                     f"{f3(rec(('a', False)))} | {f3(rec(('b', True)))} | {f3(rec(('b', False)))} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
