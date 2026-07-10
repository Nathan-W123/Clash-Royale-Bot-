"""Checkpoint regression gate against the frozen benchmark suite.

A candidate checkpoint is promoted to checkpoints/best/model.pt only if its
win rate against every benchmark bot is within `tolerance` of the current
best (and strictly better on average). Exits non-zero on regression so it
can gate a training pipeline.

CLI:
  uv run python -m src.eval.regression --candidate runs/r1/latest.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.agent.selfplay import PolicyBot
from src.decks.catalog import DeckCatalog
from src.eval.benchmark import run_benchmark

BEST_DIR = Path("checkpoints/best")
BEST_MODEL = BEST_DIR / "model.pt"
BEST_SCORES = BEST_DIR / "scores.json"


def benchmark_policy(ckpt: Path, deck: str, matches: int, seed: int) -> dict[str, float]:
    bot = PolicyBot.load(ckpt, deterministic=True)
    reporter = run_benchmark(bot, deck, matches_per_opponent=matches,
                             catalog=DeckCatalog(), seed=seed,
                             run_name=f"regression_{ckpt.stem}")
    return reporter.win_rates_by_bot()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--deck", default="training_mirror")
    parser.add_argument("--matches", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    candidate = Path(args.candidate)
    scores = benchmark_policy(candidate, args.deck, args.matches, args.seed)
    print(f"candidate {candidate}:")
    for bot, wr in scores.items():
        print(f"  vs {bot:10s} {wr:.3f}")

    if BEST_SCORES.exists():
        best = json.loads(BEST_SCORES.read_text())
        regressions = {b: (best[b], scores.get(b, 0.0)) for b in best
                       if scores.get(b, 0.0) < best[b] - args.tolerance}
        if regressions:
            print("REGRESSION — candidate rejected:")
            for bot, (old, new) in regressions.items():
                print(f"  vs {bot}: {old:.3f} -> {new:.3f}")
            sys.exit(1)
        old_avg = sum(best.values()) / len(best)
        new_avg = sum(scores.values()) / len(scores)
        print(f"suite avg: {old_avg:.3f} -> {new_avg:.3f}")

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, BEST_MODEL)
    BEST_SCORES.write_text(json.dumps(scores, indent=1))
    print(f"promoted {candidate} -> {BEST_MODEL}")


if __name__ == "__main__":
    main()
