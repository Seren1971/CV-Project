"""Whole stage-2 pipeline in one command (Windows / Linux / macOS).

    python scripts/run_all.py                 # everything with default settings
    python scripts/run_all.py --epochs 3 --train-fraction 0.5   # faster

Steps (each one is skipped if its output already exists, so the script can be
re-run after an interruption and continues where it stopped):
    1. download CCT20 images + annotations into <work>/raw   (~6 GB, resumable)
    2. build splits -> data/splits/, condition tags for test
    3. smoke test (tiny training + inference, ~2-5 min)
    4. baseline A: train -> predict val/test -> evaluate -> results/baseline_A/
    5. COCO reference (untrained, class-agnostic) -> results/reference/
    6. training curve, environment info, stage2_outputs.zip in the repo root

Heavy files (raw data, checkpoints, predictions) go to --work, default
~/cv-p16-data, outside the repo (and outside OneDrive).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
ARCHIVES = ["eccv_18_annotations.tar.gz", "eccv_18_all_images_sm.tar.gz"]


LOG = REPO / "run_log.txt"


class Tee:
    """Print to the console and append to run_log.txt."""
    def __init__(self, stream):
        self.stream, self.file = stream, open(LOG, "a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        self.stream.flush()


def step(title):
    print(f"\n{'=' * 70}\n{time.strftime('%H:%M:%S')}  {title}\n{'=' * 70}", flush=True)


def run(*args):
    cmd = [sys.executable, "-u", "-m", *map(str, args)]
    print(">", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        sys.exit(f"\nFAILED: {' '.join(cmd)}\nSee {LOG}")


def download(url: str, dest: Path):
    """Resumable download with progress."""
    done = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return
        raise
    if done and resp.status != 206:  # server ignored Range: start over
        done = 0
    total = done + int(resp.headers.get("Content-Length", 0))
    mode = "ab" if done else "wb"
    last = 0
    with open(dest, mode) as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done - last >= 200 << 20 or done == total:
                last = done
                print(f"  {dest.name}: {done / 2**30:.2f} / {total / 2**30:.2f} GB", flush=True)


def extract(archive: Path, to: Path):
    print(f"  extracting {archive.name} ...", flush=True)
    with tarfile.open(archive) as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(to, filter="data")
        else:
            tar.extractall(to)


def find_dirs(raw: Path):
    ann = next(raw.rglob("train_annotations.json")).parent
    img = max((d for d in raw.iterdir() if d.is_dir() and d != ann and ann not in d.parents
               and d not in ann.parents),
              key=lambda d: sum(1 for _ in d.glob("*.jpg")))
    return ann, img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, default=Path.home() / "cv-p16-data")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-reference", action="store_true")
    args = p.parse_args()
    sys.stdout = Tee(sys.stdout)
    print(f"\n##### run started {time.strftime('%Y-%m-%d %H:%M:%S')} #####")

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"torch {torch.__version__} | GPU: {gpu or 'NONE'} | work dir: {args.work}")
    if not gpu:
        sys.exit("No CUDA GPU visible to PyTorch. Training on CPU would take days - stopping.")

    raw, runs = args.work / "raw", args.work / "runs"
    raw.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    step("1/6 Download CCT20 (~6 GB)")
    for name in ARCHIVES:
        marker = raw / f".{name}.done"
        if marker.exists():
            print(f"  {name}: already extracted")
            continue
        free = shutil.disk_usage(args.work).free / 2**30
        if "images" in name and free < 14:
            sys.exit(f"Need ~14 GB free in {args.work}, only {free:.1f} GB. Use --work D:\\somewhere")
        archive = args.work / name
        download(f"{BASE}/{name}", archive)
        extract(archive, raw)
        archive.unlink()
        marker.touch()
    ann_dir, img_dir = find_dirs(raw)
    print(f"  annotations: {ann_dir}\n  images: {img_dir}")

    # 2. splits -------------------------------------------------------------
    step("2/6 Splits and condition tags")
    if not (REPO / "data/splits/split_stats.csv").exists():
        run("src.data.prepare_cct20", "--ann-dir", ann_dir, "--with-cis-test")
    if not (REPO / "data/splits/test_conditions.csv").exists():
        run("src.data.tag_conditions", "--split", "test", "--image-dir", img_dir)
    print((REPO / "data/splits/split_stats.csv").read_text())

    common = ["--image-dir", img_dir, "--workers", args.workers]

    # 3. smoke test -----------------------------------------------------------
    step("3/6 Smoke test")
    smoke = runs / "smoke"
    if not (smoke / "preds_test.json").exists():
        run("src.train", "--epochs", 1, "--train-fraction", 0.01, "--val-max", 40,
            "--batch-size", min(args.batch_size, 2), *common, "--out", smoke)
        run("src.predict", "--ckpt", smoke / "best.pt", "--split", "test", "--max-images", 40,
            *common, "--out", smoke / "preds_test.json")
    print("  smoke test OK")

    # 4. baseline A -----------------------------------------------------------
    step(f"4/6 Baseline A: {args.epochs} epochs")
    run_dir = runs / f"A_s{args.seed}"
    if not (run_dir / "train.done").exists():
        run("src.train", "--variant", "A", "--epochs", args.epochs, "--seed", args.seed,
            "--train-fraction", args.train_fraction, "--batch-size", args.batch_size,
            *common, "--out", run_dir)
        (run_dir / "train.done").touch()
    for split in ["val", "test"]:
        if not (run_dir / f"preds_{split}.json").exists():
            run("src.predict", "--ckpt", run_dir / "best.pt", "--split", split, *common,
                "--out", run_dir / f"preds_{split}.json")
    run("src.evaluate", "--name", "Baseline A", "--val-preds", run_dir / "preds_val.json",
        "--test-preds", run_dir / "preds_test.json", "--out", REPO / "results/baseline_A")

    # 5. reference -------------------------------------------------------------
    if not args.skip_reference:
        step("5/6 COCO reference (no training)")
        ref = runs / "reference"
        for split in ["val", "test"]:
            if not (ref / f"preds_{split}.json").exists():
                run("src.predict", "--coco-reference", "--split", split, *common,
                    "--out", ref / f"preds_{split}.json")
        run("src.evaluate", "--name", "COCO reference", "--agnostic-only",
            "--val-preds", ref / "preds_val.json", "--test-preds", ref / "preds_test.json",
            "--out", REPO / "results/reference")

    # 6. outputs -------------------------------------------------------------
    step("6/6 Save outputs")
    out = REPO / "results/baseline_A"
    for f in ["log.json", "config.json"]:
        shutil.copy(run_dir / f, out / f)
    log = json.load(open(run_dir / "log.json"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot([r["epoch"] for r in log], [r["train_loss"] for r in log], marker="o")
        ax[0].set(title="Train loss", xlabel="epoch")
        ax[1].plot([r["epoch"] for r in log], [r["val_map50"] for r in log], marker="o")
        ax[1].set(title="Val mAP@0.5 (selection subset)", xlabel="epoch")
        plt.tight_layout()
        plt.savefig(out / "training_curve.png", dpi=120)
    except Exception as e:  # plotting is optional
        print("  plot skipped:", e)
    import numpy
    import PIL
    import torchvision
    (REPO / "results/environment.txt").write_text(
        f"python {platform.python_version()} ({platform.system()})\n"
        f"torch {torch.__version__}\ntorchvision {torchvision.__version__}\n"
        f"numpy {numpy.__version__}\npillow {PIL.__version__}\nGPU {gpu}\n"
        f"epochs {args.epochs}, train_fraction {args.train_fraction}, batch {args.batch_size}\n"
        f"train minutes total {sum(r['train_minutes'] for r in log):.1f}\n")
    zpath = REPO / "stage2_outputs.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["data/splits", "results"]:
            for f in (REPO / folder).rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(REPO))
    print((out / "summary.md").read_text())
    print(f"\nDONE. Results: {REPO / 'results'}\nArchive: {zpath}")


if __name__ == "__main__":
    main()
