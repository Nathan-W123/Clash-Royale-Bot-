"""Re-run the frozen benchmark and write a fresh baseline.

Run this after any balance change (see `scripts/sync_card_stats.py`). Every
win rate recorded before such a change is measured against different physics
and is not comparable afterwards — the frozen bots are the regression
tripwire, and a tripwire calibrated to the old game is worse than none.

    python -m scripts.rebenchmark --matches 40

Writes one JSON per agent archetype plus a combined summary, so a later run
can diff against it.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.eval.benchmark import load_benchmark_opponents, run_benchmark
from src.training.config import load_training_config

ARCHETYPES = ("rusher", "control", "siege", "beatdown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=40,
                        help="matches per benchmark opponent, per archetype")
    parser.add_argument("--agents", nargs="*", default=list(ARCHETYPES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark_baseline.json"))
    parser.add_argument("--tag", default="post-card-stat-sync")
    args = parser.parse_args()

    catalog = DeckCatalog()
    training = load_training_config()
    opponents = load_benchmark_opponents()
    print(f"frozen roster: {', '.join(f'{o.bot_name}/{o.deck_name}' for o in opponents)}")
    print(f"{args.matches} matches per opponent x {len(opponents)} opponents "
          f"x {len(args.agents)} archetypes\n")

    results = {}
    for name in args.agents:
        rng = np.random.default_rng(args.seed)
        bot = get_bot(name, catalog=catalog, deck_name=name, rng=rng,
                      skill_tier=training.opponents.skill_tier)
        reporter = run_benchmark(bot, name, matches_per_opponent=args.matches,
                                 catalog=catalog, seed=args.seed,
                                 run_name=f"baseline_{name}")
        overall = reporter.overall
        results[name] = {
            "win_rate": overall.win_rate,
            "wins": overall.wins,
            "losses": overall.losses,
            "draws": overall.draws,
            "card_entropy": reporter.card_entropy,
            "vs_bot": reporter.win_rates_by_bot(),
        }
        print(f"{name:<10} WR {overall.win_rate:6.1%}  "
              f"({overall.wins}W {overall.losses}L {overall.draws}D)  "
              f"entropy {reporter.card_entropy:.2f}")
        print("           " + "  ".join(
            f"vs {b}={wr:.0%}" for b, wr in sorted(reporter.win_rates_by_bot().items())))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tag": args.tag,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matches_per_opponent": args.matches,
        "seed": args.seed,
        "opponents": [{"bot": o.bot_name, "deck": o.deck_name} for o in opponents],
        "agents": results,
    }, indent=2), encoding="utf-8")
    print(f"\nbaseline written to {args.out}")


if __name__ == "__main__":
    main()
