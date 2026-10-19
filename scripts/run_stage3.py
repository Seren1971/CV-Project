"""Stage 3: variant B (targeted augmentation) and comparison with baseline A.

    python scripts/run_stage3.py              # same budget as A: 6 epochs, seed 0

Needs the stage-2 run first (data in <work>/raw, baseline A in <work>/runs/A_s0).
Each step is skipped if its output exists, so the script can be re-run.
    1. augmentation preview with box checks -> results/variant_B/augmentation_examples.png
    2. smoke test of B (tiny training)
    3. train B -> predict val / test / test_cis
    4. predict A on test_cis
    5. evaluate B (test), A and B (test_cis) -> results/variant_B, results/*_cis
    6. A vs B comparison with bootstrap intervals -> results/comparison_A_vs_B.md
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import REPO, Tee, find_dirs, run, step  # noqa: E402


def plot_curves(runs: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, log in runs.items():
        ax[0].plot([r["epoch"] for r in log], [r["train_loss"] for r in log], marker="o", label=name)
        ax[1].plot([r["epoch"] for r in log], [r["val_map50"] for r in log], marker="o", label=name)
    ax[0].set(title="Train loss", xlabel="epoch")
    ax[1].set(title="Val mAP@0.5 (selection subset)", xlabel="epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, default=Path.home() / "cv-p16-data")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sys.stdout = Tee(sys.stdout)
    print(f"\n##### stage 3 started {time.strftime('%Y-%m-%d %H:%M:%S')} #####")

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"torch {torch.__version__} | GPU: {gpu or 'NONE'} | work dir: {args.work}")
    if not gpu:
        sys.exit("No CUDA GPU visible to PyTorch - stopping.")
    runs = args.work / "runs"
    a_dir, b_dir = runs / f"A_s{args.seed}", runs / f"B_s{args.seed}"
    if not (a_dir / "best.pt").exists():
        sys.exit(f"Baseline A checkpoint not found in {a_dir}. Run run_baseline.bat first.")
    _, img_dir = find_dirs(args.work / "raw")
    common = ["--image-dir", img_dir, "--workers", args.workers]
    res = REPO / "results"

    step("1/6 Augmentation preview (checks boxes after every transform)")
    run("src.data.preview_augment", "--image-dir", img_dir,
        "--out", res / "variant_B/augmentation_examples.png")

    step("2/6 Smoke test of variant B")
    smoke = runs / "smoke_B"
    if not (smoke / "best.pt").exists():
        run("src.train", "--variant", "B", "--epochs", 1, "--train-fraction", 0.01, "--val-max", 40,
            "--batch-size", 2, *common, "--out", smoke)
    print("  smoke test OK")

    step(f"3/6 Variant B: {args.epochs} epochs (same budget as A)")
    if not (b_dir / "train.done").exists():
        run("src.train", "--variant", "B", "--epochs", args.epochs, "--seed", args.seed,
            "--batch-size", args.batch_size, *common, "--out", b_dir)
        (b_dir / "train.done").touch()
    for split in ["val", "test", "test_cis"]:
        if not (b_dir / f"preds_{split}.json").exists():
            run("src.predict", "--ckpt", b_dir / "best.pt", "--split", split, *common,
                "--out", b_dir / f"preds_{split}.json")

    step("4/6 Baseline A on test_cis (seen camera locations)")
    for split in ["val", "test", "test_cis"]:
        if not (a_dir / f"preds_{split}.json").exists():
            run("src.predict", "--ckpt", a_dir / "best.pt", "--split", split, *common,
                "--out", a_dir / f"preds_{split}.json")

    step("5/6 Evaluate")
    run("src.evaluate", "--name", "Variant B", "--val-preds", b_dir / "preds_val.json",
        "--test-preds", b_dir / "preds_test.json", "--out", res / "variant_B")
    for name, d in [("Baseline A", a_dir), ("Variant B", b_dir)]:
        tag = "baseline_A_cis" if name.startswith("Baseline") else "variant_B_cis"
        run("src.evaluate", "--name", f"{name} (test_cis)", "--test-split", "test_cis",
            "--val-preds", d / "preds_val.json", "--test-preds", d / "preds_test_cis.json",
            "--out", res / tag)

    step("6/6 Compare A and B")
    run("src.compare", "--a-preds", a_dir / "preds_test.json", "--b-preds", b_dir / "preds_test.json",
        "--a-results", res / "baseline_A", "--b-results", res / "variant_B",
        "--extra-results", f"A (test_cis)={res / 'baseline_A_cis'}", f"B (test_cis)={res / 'variant_B_cis'}",
        "--out", res / "comparison_A_vs_B.md")

    for f in ["log.json", "config.json"]:
        shutil.copy(b_dir / f, res / "variant_B" / f)
    logs = {"A": json.load(open(a_dir / "log.json")), "B": json.load(open(b_dir / "log.json"))}
    try:
        plot_curves(logs, res / "variant_B/training_curves_A_vs_B.png")
    except Exception as e:
        print("  plot skipped:", e)
    with open(res / "environment.txt", "a") as f:
        f.write(f"\nstage 3 ({time.strftime('%Y-%m-%d')}): variant B, epochs {args.epochs}, "
                f"batch {args.batch_size}, python {platform.python_version()}, torch {torch.__version__}, "
                f"train minutes total {sum(r['train_minutes'] for r in logs['B']):.1f}\n")
    print(f"\nDONE. Comparison: {res / 'comparison_A_vs_B.md'}")


if __name__ == "__main__":
    main()
