"""CLI entry point: run benchmark suite and print training report."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.agent.selfplay import PolicyBot
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.eval.benchmark import run_benchmark
from src.training.config import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CR Bot benchmark and print report")
    parser.add_argument("--agent-bot", default="generic",
                        help="Scripted bot archetype to evaluate (ignored if --checkpoint is given)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained policy checkpoint to evaluate instead of a scripted bot")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample the checkpoint's policy instead of acting greedily")
    parser.add_argument("--agent-deck", default="training_mirror", help="Deck name")
    parser.add_argument("--matches", type=int, default=None, help="Matches per opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="Export JSON report path")
    parser.add_argument("--tensorboard", type=Path, default=None, help="TensorBoard log dir")
    parser.add_argument("--step", type=int, default=0, help="Training step for TB logging")
    args = parser.parse_args()

    catalog = DeckCatalog()
    training = load_training_config()
    if args.checkpoint is not None:
        agent_bot = PolicyBot.load(args.checkpoint, deterministic=not args.stochastic)
    else:
        rng = __import__("numpy").random.default_rng(args.seed)
        agent_bot = get_bot(
            args.agent_bot,
            catalog=catalog,
            deck_name=args.agent_deck,
            rng=rng,
            skill_tier=training.opponents.skill_tier,
        )

    reporter = run_benchmark(
        agent_bot,
        args.agent_deck,
        matches_per_opponent=args.matches,
        catalog=catalog,
        seed=args.seed,
        run_name=f"benchmark_{args.checkpoint.stem if args.checkpoint else args.agent_bot}",
    )
    print(reporter.summary())

    if args.json:
        path = reporter.export_json(args.json)
        print(f"\nJSON report: {path}")
    if args.tensorboard:
        reporter.log_tensorboard(args.tensorboard, args.step)
        print(f"TensorBoard scalars logged to {args.tensorboard}")


if __name__ == "__main__":
    main()
