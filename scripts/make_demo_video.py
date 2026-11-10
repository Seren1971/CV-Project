"""Record the demo as an MP4 video: title card, per-image live-style detection, outro.

    python scripts/make_demo_video.py --ckpt path/to/best.pt --images demo/images --out demo/demo_video.mp4

Renders at 1280x720 @ 30 fps. Each image scene shows the raw frame briefly
("running inference...") and then the detections with class, score and timing.
Requires imageio + imageio-ffmpeg (pip install imageio imageio-ffmpeg).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.predict import load_model
from src.models.detector import set_eval_size
from scripts.demo import COLORS, detect

W, H, FPS = 1280, 720, 30
FONT_DIR = Path(r"C:\Windows\Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default(size=size)


F_TITLE = font("arialbd.ttf", 44)
F_SUB = font("arial.ttf", 26)
F_MONO = font("consola.ttf", 20)
F_MONO_SM = font("consola.ttf", 17)
F_BOX = font("arialbd.ttf", 16)


def card(lines, canvas=None) -> np.ndarray:
    img = Image.new("RGB", (W, H), "#101418")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill="#3cb44b")
    y = 150
    for text, f, col, gap in lines:
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col)
        y += f.size + gap
    return np.array(img)


def fit(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    img = img.convert("RGB")
    r = min(box_w / img.width, box_h / img.height)
    return img.resize((round(img.width * r), round(img.height * r)), Image.LANCZOS)


def scene_frame(img: Image.Image, boxes, scores, labels, classes, header: str,
                status: str, annotated: bool) -> np.ndarray:
    canvas = Image.new("RGB", (W, H), "#101418")
    d = ImageDraw.Draw(canvas)
    d.text((28, 16), header, font=F_MONO, fill="#9aa5b1")
    view = fit(img, W - 56, H - 150)
    x0, y0 = (W - view.width) // 2, 78
    canvas.paste(view, (x0, y0))
    if annotated:
        vd = ImageDraw.Draw(canvas)
        sx, sy = view.width / img.width, view.height / img.height
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            x1, y1, x2, y2 = x0 + b[0] * sx, y0 + b[1] * sy, x0 + b[2] * sx, y0 + b[3] * sy
            col = COLORS[(l - 1) % len(COLORS)]
            vd.rectangle([x1, y1, x2, y2], outline=col, width=3)
            vd.text((x1 + 4, y1 + 4), f"{classes[l - 1]} {s:.2f}", font=F_BOX, fill=col)
    d.rectangle([0, H - 56, W, H], fill="#1a2027")
    d.text((28, H - 42), status, font=F_MONO_SM, fill="#e8edf2" if annotated else "#ffd966")
    return np.array(canvas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", default=["demo/images"])
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", type=Path, default=Path("demo/demo_video.mp4"))
    p.add_argument("--title", default="P16 · CCT20 animal detection — live demo")
    args = p.parse_args()

    paths = []
    for pat in args.images:
        q = Path(pat)
        paths += sorted(q.iterdir()) if q.is_dir() else [q]
    paths = [f for f in paths if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.ckpt, coco_reference=False, device=device)
    model.eval()
    set_eval_size(model)
    ck = torch.load(args.ckpt, map_location="cpu")
    thr = args.threshold if args.threshold is not None else ck.get("threshold", 0.7)

    cmd = f"$ python scripts/demo.py --ckpt .../Bp12_s0/best.pt --threshold {thr:.2f} --images demo/images"
    writer = imageio.get_writer(str(args.out), fps=FPS, codec="libx264", quality=8)

    # warm up CUDA kernels so the first on-screen timing is representative
    warm = Image.open(paths[0]).convert("RGB")
    detect(model, warm, device, thr)
    if device.type == "cuda":
        torch.cuda.synchronize()

    def hold(frame: np.ndarray, seconds: float):
        for _ in range(round(seconds * FPS)):
            writer.append_data(frame)

    hold(card([
        (args.title, F_TITLE, "#ffffff", 26),
        ("variant B' · Faster R-CNN (ResNet-50 FPN) · 12 epochs", F_SUB, "#3cb44b", 18),
        (f"val mAP@0.5 {ck.get('val_map50', 0):.3f} · test mAP@0.5 0.609 · threshold {thr:.2f}",
         F_SUB, "#9aa5b1", 18),
        (f"{len(paths)} camera-trap frames · GPU {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
         F_SUB, "#9aa5b1", 0),
    ]), 3.0)
    hold(card([(cmd, F_MONO, "#e8edf2", 0)]), 2.0)

    for path in paths:
        img = Image.open(path).convert("RGB")
        raw = scene_frame(img, None, None, None, classes, cmd,
                          f"> {path.name} — running inference...", annotated=False)
        t0 = time.time()
        boxes, scores, labels = detect(model, img, device, thr)
        ms = (time.time() - t0) * 1000
        found = ", ".join(f"{classes[l - 1]} {s:.2f}" for s, l in zip(scores.tolist(), labels.tolist())) \
            or "(nothing above threshold)"
        det = scene_frame(img, boxes, scores, labels, classes, cmd,
                          f"> {path.name} — {len(boxes)} det in {ms:.0f} ms : {found}", annotated=True)
        hold(raw, 0.8)
        hold(det, 3.2)

    hold(card([
        ("Done — detections.json written", F_TITLE, "#ffffff", 26),
        ("82% correct · 8% class confusion · 10% background FP (test set)", F_SUB, "#3cb44b", 18),
        ("github.com/Seren1971/CV-Project", F_SUB, "#9aa5b1", 0),
    ]), 3.0)
    writer.close()
    print(f"video -> {args.out}")


if __name__ == "__main__":
    main()
