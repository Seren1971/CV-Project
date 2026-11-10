"""Live demo: run the trained detector on arbitrary images and draw the detections.

    python scripts/demo.py --ckpt path/to/best.pt --images demo/images --out demo/out

- `--images` accepts a folder or individual image paths (any JPG/PNG, no dataset needed).
- The confidence threshold defaults to the value stored inside the checkpoint's results
  (pass --threshold to override).
- Annotated copies and detections.json are written to --out; a per-image table is printed.

During the presentation run it on demo/images (committed to the repo) or on any new
camera-trap photo to show the pipeline live.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

import torch
from PIL import Image, ImageDraw, ImageFont

from src.predict import load_model
from src.models.detector import set_eval_size

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9a6324"]


@torch.no_grad()
def detect(model, img: Image.Image, device, thr: float):
    from torchvision.transforms.functional import to_tensor
    t = to_tensor(img).to(device)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        out = model([t])[0]
    keep = out["scores"] >= thr
    return (out["boxes"][keep].cpu(), out["scores"][keep].cpu(), out["labels"][keep].cpu())


def draw(img: Image.Image, boxes, scores, labels, classes) -> Image.Image:
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(14, img.width // 70))
    for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        col = COLORS[(l - 1) % len(COLORS)]
        d.rectangle(b, outline=col, width=3)
        d.text((b[0] + 3, b[1] + 3), f"{classes[l - 1]} {s:.2f}", fill=col, font=font)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", required=True, help="folders and/or image files")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", type=Path, default=Path("demo/out"))
    args = p.parse_args()

    paths = []
    for pat in args.images:
        q = Path(pat)
        if q.is_dir():
            paths += sorted(f for f in q.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        elif q.is_file():
            paths.append(q)
    if not paths:
        raise SystemExit("no images found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.ckpt, coco_reference=False, device=device)
    model.eval()
    set_eval_size(model)
    ck = torch.load(args.ckpt, map_location="cpu")
    thr = args.threshold if args.threshold is not None else ck.get("threshold", 0.7)
    print(f"model: {args.ckpt} (variant {ck.get('variant')}, val mAP@0.5 {ck.get('val_map50'):.3f})")
    print(f"device: {device}, threshold: {thr}, images: {len(paths)}")

    args.out.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for path in paths:
        img = Image.open(path).convert("RGB")
        t0 = time.time()
        boxes, scores, labels = detect(model, img, device, thr)
        ms = (time.time() - t0) * 1000
        dets = [{"box": [round(v, 1) for v in b], "score": round(s, 3),
                 "label": classes[l - 1]} for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist())]
        results[path.name] = dets
        draw(img, boxes, scores, labels, classes).save(args.out / f"{path.stem}_detected.jpg", quality=92)
        found = ", ".join(f"{d['label']} {d['score']:.2f}" for d in dets) or "(nothing above threshold)"
        rows.append((path.name, len(dets), ms, found))
        print(f"{path.name:50s} {len(dets):2d} det in {ms:6.0f} ms : {found}")
    json.dump(results, open(args.out / "detections.json", "w"), indent=1)
    print(f"\nannotated images + detections.json -> {args.out}")


if __name__ == "__main__":
    main()
