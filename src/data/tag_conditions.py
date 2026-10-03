"""Tag every image of a split with the conditions used in the failure analysis.

Columns written to data/splits/<split>_conditions.csv:
    is_night      - IR/night frame (image is grayscale: R, G, B channels ~equal)
    brightness    - mean gray level 0..255 (low = dark)
    is_dark       - brightness < --dark-thr
    min_box_area  - smallest box area as fraction of the image (small / distant animal)
    is_small      - min_box_area < --small-thr
    truncated     - some box touches the image border (animal partly outside the frame)

Occlusion by vegetation and unusual poses cannot be derived automatically from
CCT20 labels; they are tagged manually on the inspected failure cases
(see docs/dataset_preparation.md, section "Condition tags").

Usage:
    python -m src.data.tag_conditions --split test --image-dir data/raw/eccv_18_all_images_sm
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset import scaled_boxes


def analyse(path: Path, rec: dict, border_px: int = 2) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        small = np.asarray(im.resize((128, 96)), dtype=np.float32)
    chan_diff = float(np.abs(small[..., 0] - small[..., 1]).mean()
                      + np.abs(small[..., 1] - small[..., 2]).mean())
    boxes = scaled_boxes(rec, w, h)
    areas = [bw * bh / (w * h) for _, _, bw, bh in boxes]
    trunc = any(x <= border_px or y <= border_px or x + bw >= w - border_px or y + bh >= h - border_px
                for x, y, bw, bh in boxes)
    return {
        "is_night": chan_diff < 3.0,
        "brightness": round(float(small.mean()), 1),
        "min_box_area": round(min(areas), 5) if areas else None,
        "truncated": trunc,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--image-dir", type=Path, default=Path("data/raw/eccv_18_all_images_sm"))
    p.add_argument("--dark-thr", type=float, default=60.0)
    p.add_argument("--small-thr", type=float, default=0.01)
    args = p.parse_args()

    recs = json.load(open(args.splits_dir / f"{args.split}.json"))
    out = args.splits_dir / f"{args.split}_conditions.csv"
    cols = ["image_id", "label", "is_night", "brightness", "is_dark",
            "min_box_area", "is_small", "truncated"]
    missing = 0
    summary = {"is_night": 0, "is_dark": 0, "is_small": 0, "truncated": 0}
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            path = args.image_dir / r["file_name"]
            if not path.exists():
                missing += 1
                continue
            a = analyse(path, r)
            a["is_dark"] = a["brightness"] < args.dark_thr
            a["is_small"] = a["min_box_area"] is not None and a["min_box_area"] < args.small_thr
            for k in summary:
                summary[k] += int(bool(a[k]))
            w.writerow({"image_id": r["image_id"], "label": r["label"], **a})
    n = len(recs) - missing
    print(f"{args.split}: {n} images tagged, {missing} missing files")
    for k, v in summary.items():
        print(f"  {k}: {v} ({100 * v / max(n, 1):.1f}%)")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
