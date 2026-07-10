"""Watch a match in ASCII: bot vs bot, or a policy checkpoint vs a bot.

Usage:
  uv run python scripts/watch_match.py --bottom rusher --top control
  uv run python scripts/watch_match.py --bottom runs/r1/latest.pt --top control
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena
from src.simulator.constants import MatchResult, Side
from src.simulator.engine import BattleEngine
from src.simulator.render import render_ascii
from src.training.config import load_training_config


def resolve_actor(spec: str, deck_arg: str | None, catalog, training, seed: int):
    if spec.endswith(".pt"):
        from src.agent.selfplay import PolicyBot
        return (PolicyBot.load(spec, deterministic=True),
                deck_arg or "training_mirror")
    deck = deck_arg or (spec if spec in catalog.decks else "training_mirror")
    bot = get_bot(spec, catalog=catalog, deck_name=deck,
                  rng=np.random.default_rng(seed),
                  skill_tier=training.opponents.skill_tier)
    return bot, deck


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bottom", default="rusher")
    parser.add_argument("--top", default="control")
    parser.add_argument("--deck-bottom", default=None)
    parser.add_argument("--deck-top", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--quiet", action="store_true", help="print only the result")
    args = parser.parse_args()

    arena, catalog = load_arena(), DeckCatalog()
    training = load_training_config()
    bottom, deck_b = resolve_actor(args.bottom, args.deck_bottom, catalog, training, args.seed)
    top, deck_t = resolve_actor(args.top, args.deck_top, catalog, training, args.seed + 1)

    eng = BattleEngine(catalog.resolve(deck_b), catalog.resolve(deck_t),
                       arena, seed=args.seed)
    step = 0
    while eng.result == MatchResult.ONGOING and step < 600:
        for side, actor in ((Side.BOTTOM, bottom), (Side.TOP, top)):
            action = actor.decide(eng, side)
            if action is not None:
                p = eng.players[side]
                if (p.can_afford(action.slot)
                        and eng.legal_deploy(side, p.hand[action.slot], action.x, action.y)):
                    eng.play_card(side, action.slot, action.x, action.y)
        for _ in range(5):
            eng.tick()
            if eng.result != MatchResult.ONGOING:
                break
        step += 1
        if not args.quiet:
            print("\033[2J\033[H" + render_ascii(eng))
            time.sleep(1.0 / args.fps)
    print(render_ascii(eng))
    print(f"\n{args.bottom} (bottom, {deck_b}) vs {args.top} (top, {deck_t})")
    print(f"Result: {eng.result.name} in {eng.time:.0f}s")


if __name__ == "__main__":
    main()
