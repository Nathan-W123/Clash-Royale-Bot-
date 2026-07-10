"""Learning-curve plots from runs/<run>/train_log.csv and eval_log.csv.

CLI:  uv run python -m src.eval.plots --run r1
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_eval(rows: list[dict], out: Path) -> None:
    by_bot: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_bot[r["bot"]].append((int(r["step"]), float(r["win_rate"])))
    fig, ax = plt.subplots(figsize=(9, 5))
    for bot, pts in sorted(by_bot.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=bot)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("win rate (deterministic eval)")
    ax.set_ylim(0, 1)
    ax.set_title("Benchmark suite win rate over training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def plot_train(rows: list[dict], out: Path) -> None:
    steps = [int(r["step"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("win_rate", "training win rate", axes[0][0]),
        ("reward_mean", "mean reward/step", axes[0][1]),
        ("card_usage_entropy", "card usage entropy (nats)", axes[1][0]),
        ("leak_per_match", "elixir leaked / match", axes[1][1]),
    ]
    for key, title, ax in panels:
        ax.plot(steps, [float(r[key]) for r in rows])
        ax.set_title(title)
        ax.set_xlabel("training step")
    # Stage transition markers on the win-rate panel.
    last = None
    for r in rows:
        if r["stage"] != last:
            axes[0][0].axvline(int(r["step"]), color="red", lw=0.7, ls=":")
            axes[0][0].text(int(r["step"]), 0.02, r["stage"], rotation=90,
                            fontsize=7, color="red")
            last = r["stage"]
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run1")
    args = parser.parse_args()
    run_dir = Path("runs") / args.run
    eval_rows = read_csv(run_dir / "eval_log.csv") if (run_dir / "eval_log.csv").exists() else []
    train_rows = read_csv(run_dir / "train_log.csv") if (run_dir / "train_log.csv").exists() else []
    if eval_rows:
        plot_eval(eval_rows, run_dir / "eval_curves.png")
    if train_rows:
        plot_train(train_rows, run_dir / "train_curves.png")
    if not eval_rows and not train_rows:
        raise SystemExit(f"no logs found under {run_dir}")


if __name__ == "__main__":
    main()
