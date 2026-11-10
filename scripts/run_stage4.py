"""Stage 4: evaluate the 12-epoch runs (baseline A, variant B') and compare them.

    python scripts/run_stage4.py            # after run_stage4.bat finished training

Needs runs/A12_s0 and runs/Bp12_s0 (best.pt + preds_*.json) in <work>/runs.
Each evaluation step is skipped if its output exists. Writes:
    results/stage4/baseline_A12/   results/stage4/variant_Bp12/   (+ _cis)
    results/stage4/comparison_A12_vs_Bp12.md   95% bootstrap intervals over sequences
    results/stage4/training_curves.png         A6 / B6 / A12 / B'12 val curves
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
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, default=Path.home() / "cv-p16-data")
    args = p.parse_args()
    sys.stdout = Tee(sys.stdout)
    print(f"\n##### stage 4 evaluation started {time.strftime('%Y-%m-%d %H:%M:%S')} #####")

    runs = args.work / "runs"
    a_dir, b_dir = runs / "A12_s0", runs / "Bp12_s0"
    for d in (a_dir, b_dir):
        for split in ["val", "test", "test_cis"]:
            if not (d / f"preds_{split}.json").exists():
                sys.exit(f"missing {d / f'preds_{split}.json'} - run run_stage4.bat first")
    res = REPO / "results" / "stage4"

    step("1/3 Evaluate A12 and B'12 on test and test_cis")
    run("src.evaluate", "--name", "Baseline A (12 epochs)",
        "--val-preds", a_dir / "preds_val.json", "--test-preds", a_dir / "preds_test.json",
        "--out", res / "baseline_A12")
    run("src.evaluate", "--name", "Variant B' (12 epochs)",
        "--val-preds", b_dir / "preds_val.json", "--test-preds", b_dir / "preds_test.json",
        "--out", res / "variant_Bp12")
    for name, d, tag in [("Baseline A (12 ep)", a_dir, "baseline_A12_cis"),
                         ("Variant B' (12 ep)", b_dir, "variant_Bp12_cis")]:
        run("src.evaluate", "--name", f"{name} (test_cis)", "--test-split", "test_cis",
            "--val-preds", d / "preds_val.json", "--test-preds", d / "preds_test_cis.json",
            "--out", res / tag)

    step("2/3 Compare A12 and B'12 (95% bootstrap intervals over sequences)")
    run("src.compare", "--a-preds", a_dir / "preds_test.json", "--b-preds", b_dir / "preds_test.json",
        "--a-results", res / "baseline_A12", "--b-results", res / "variant_Bp12",
        "--extra-results", f"A12 (test_cis)={res / 'baseline_A12_cis'}",
        f"B'12 (test_cis)={res / 'variant_Bp12_cis'}",
        "--out", res / "comparison_A12_vs_Bp12.md")

    step("3/3 Figures, configs, environment")
    for d, tag in [(a_dir, "baseline_A12"), (b_dir, "variant_Bp12")]:
        for f in ["log.json", "config.json"]:
            shutil.copy(d / f, res / tag / f)
    logs = {}
    for name, d in [("A (6 ep)", runs / "A_s0"), ("B (6 ep)", runs / "B_s0"),
                    ("A (12 ep)", a_dir), ("B' (12 ep)", b_dir)]:
        try:
            logs[name] = json.load(open(d / "log.json"))
        except FileNotFoundError:
            print(f"  no log for {name}, skipped in the figure")
    try:
        plot_curves(logs, res / "training_curves.png")
    except Exception as e:
        print("  plot skipped:", e)
    import torch
    with open(REPO / "results" / "environment.txt", "a") as f:
        mins = {k: sum(r["train_minutes"] for r in v) for k, v in logs.items()}
        a12 = mins.get("A (12 ep)", float("nan"))
        bp12 = mins.get("B' (12 ep)", float("nan"))
        f.write(f"\nstage 4 ({time.strftime('%Y-%m-%d')}): A and B' at 12 epochs, batch 4, "
                f"python {platform.python_version()}, torch {torch.__version__}, "
                f"train minutes: A12 {a12:.1f}, B'12 {bp12:.1f}\n")
    print(f"\nDONE. Comparison: {res / 'comparison_A12_vs_Bp12.md'}")


if __name__ == "__main__":
    main()
