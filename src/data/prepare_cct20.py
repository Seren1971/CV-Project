"""Prepare the CCT20 (Caltech Camera Traps, ECCV'18 benchmark) subset for the project.

Reads the official COCO-CameraTraps annotation files, keeps the selected animal
classes, builds train / val / test splits, checks for leakage and writes
compact JSON + CSV files to data/splits/.

Split policy (default):
    train = official `train`        (10 camera locations)
    val   = official `cis_val`      (same locations as train, different sequences)
    test  = official `trans_test`   (camera locations never seen in training)
Optional extra test set:
    test_cis = official `cis_test`  (seen locations) -- to measure the
               "new location" generalisation gap.

Usage:
    python -m src.data.prepare_cct20 --ann-dir data/raw/eccv_18_annotation_files \
        --out-dir data/splits
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OFFICIAL_SPLITS = ["train", "cis_val", "trans_val", "cis_test", "trans_test"]
SPLIT_MAP = {"train": "train", "val": "cis_val", "test": "trans_test", "test_cis": "cis_test"}
NON_ANIMAL = {"empty", "car"}


def find_annotation_file(ann_dir: Path, split: str) -> Path:
    hits = sorted(ann_dir.rglob(f"{split}_annotations.json"))
    if not hits:
        raise FileNotFoundError(f"{split}_annotations.json not found under {ann_dir}")
    return hits[0]


def load_split(ann_dir: Path, split: str) -> dict:
    with open(find_annotation_file(ann_dir, split), encoding="utf-8") as f:
        return json.load(f)


def to_records(coco: dict, keep: set[str], include_empty: bool = False) -> list[dict]:
    """One record per image with all its boxes.

    Images whose only labels are dropped classes are skipped. If include_empty,
    images labelled `empty` are kept with no boxes (label "empty"): they are
    needed to measure false detections on vegetation / background.
    """
    cat_name = {c["id"]: c["name"].lower() for c in coco["categories"]}
    anns = defaultdict(list)
    empty_ids, other_ids = set(), set()
    for a in coco["annotations"]:
        name = cat_name[a["category_id"]]
        if name in keep and a.get("bbox"):
            anns[a["image_id"]].append({"category": name, "bbox": [float(v) for v in a["bbox"]]})
        elif name == "empty":
            empty_ids.add(a["image_id"])
        else:
            other_ids.add(a["image_id"])
    records = []
    for img in coco["images"]:
        boxes = anns.get(img["id"])
        if boxes:
            # image-level label = class of the largest box
            main = max(boxes, key=lambda b: b["bbox"][2] * b["bbox"][3])["category"]
        elif include_empty and img["id"] in empty_ids and img["id"] not in other_ids:
            boxes, main = [], "empty"
        else:
            continue
        records.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "location": str(img.get("location", "")),
            "seq_id": str(img.get("seq_id", img["id"])),
            "frame_num": img.get("frame_num", 0),
            "date_captured": img.get("date_captured", ""),
            "width": img.get("width"),
            "height": img.get("height"),
            "label": main,
            "multi_class": len({b["category"] for b in boxes}) > 1,
            "boxes": boxes,
        })
    return records


def cap_per_class(records: list[dict], cap: int | None, seed: int) -> list[dict]:
    """Subsample to <= cap images per class, keeping whole sequences together."""
    if not cap:
        return records
    rng = random.Random(seed)
    by_cls_seq = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_cls_seq[r["label"]][r["seq_id"]].append(r)
    out = []
    for cls, seqs in sorted(by_cls_seq.items()):
        ids = sorted(seqs)
        rng.shuffle(ids)
        n = 0
        for sid in ids:
            if n >= cap:
                break
            out.extend(seqs[sid])
            n += len(seqs[sid])
    return out


def check_leakage(splits: dict[str, list[dict]]) -> list[str]:
    msgs = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            img = {r["image_id"] for r in splits[a]} & {r["image_id"] for r in splits[b]}
            seq = {r["seq_id"] for r in splits[a]} & {r["seq_id"] for r in splits[b]}
            if img:
                msgs.append(f"LEAK: {len(img)} identical images in {a} and {b}")
            if seq:
                msgs.append(f"LEAK: {len(seq)} shared sequences in {a} and {b}")
    if "test" in splits:
        shared = {r["location"] for r in splits["train"]} & {r["location"] for r in splits["test"]}
        if shared:
            msgs.append(f"LEAK: test shares camera locations with train: {sorted(shared)}")
    return msgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir", type=Path, default=Path("data/raw/eccv_18_annotation_files"))
    p.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--classes", nargs="*", default=None,
                   help="Classes to keep (default: every animal class with >= --min-train images)")
    p.add_argument("--min-train", type=int, default=100)
    p.add_argument("--cap-train", type=int, default=0, help="max images/class in train (0 = all)")
    p.add_argument("--cap-eval", type=int, default=600, help="max images/class in val/test")
    p.add_argument("--with-cis-test", action="store_true")
    p.add_argument("--no-empty-eval", action="store_true",
                   help="do not add empty frames to val/test (they are added by default, "
                        "capped like a class, to measure false detections)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = {s: load_split(args.ann_dir, s) for s in OFFICIAL_SPLITS
           if s != "cis_test" or args.with_cis_test}

    # choose classes from train counts
    all_names = {c["name"].lower() for c in raw["train"]["categories"]} - NON_ANIMAL
    train_counts = Counter(r["label"] for r in to_records(raw["train"], all_names))
    keep = set(args.classes) if args.classes else {
        c for c, n in train_counts.items() if n >= args.min_train}
    print("train images per animal class:", dict(train_counts.most_common()))
    print("kept classes:", sorted(keep))

    wanted = dict(SPLIT_MAP)
    if not args.with_cis_test:
        wanted.pop("test_cis")
    splits = {}
    for ours, official in wanted.items():
        recs = to_records(raw[official], keep,
                          include_empty=(ours != "train" and not args.no_empty_eval))
        cap = args.cap_train if ours == "train" else args.cap_eval
        splits[ours] = cap_per_class(recs, cap, args.seed)

    problems = check_leakage(splits)
    for m in problems:
        print(m)
    if problems:
        raise SystemExit("Leakage detected - fix before training.")
    print("leakage check: OK (no shared images, sequences, or train/test locations)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(keep)
    with open(args.out_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)

    stats_rows = []
    for name, recs in splits.items():
        with open(args.out_dir / f"{name}.json", "w") as f:
            json.dump(recs, f)
        with open(args.out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "file_name", "label", "location", "seq_id", "n_boxes", "multi_class"])
            for r in recs:
                w.writerow([r["image_id"], r["file_name"], r["label"], r["location"],
                            r["seq_id"], len(r["boxes"]), r["multi_class"]])
        img_cnt = Counter(r["label"] for r in recs)
        box_cnt = Counter(b["category"] for r in recs for b in r["boxes"])
        stats_rows.append({
            "split": name, "images": len(recs),
            "empty_images": img_cnt.get("empty", 0),
            "boxes": sum(len(r["boxes"]) for r in recs),
            "sequences": len({r["seq_id"] for r in recs}),
            "locations": len({r["location"] for r in recs}),
            **{f"boxes_{c}": box_cnt.get(c, 0) for c in classes},
        })

    with open(args.out_dir / "split_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0]))
        w.writeheader()
        w.writerows(stats_rows)
    for row in stats_rows:
        print(row)
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
