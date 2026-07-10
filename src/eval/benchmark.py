"""Fixed benchmark suite — regression tripwire for training progress."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.bots.base import Bot
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.eval.reporter import TrainingReporter
from src.simulator.cards import CONFIG_DIR, load_arena
from src.simulator.constants import Side
from src.training.config import load_training_config
from src.training.match_runner import run_match_detailed


@dataclass(frozen=True)
class BenchmarkOpponent:
    bot_name: str
    deck_name: str


def load_benchmark_opponents(path: Path | None = None) -> list[BenchmarkOpponent]:
    path = path or CONFIG_DIR / "eval.yaml"
    raw = yaml.safe_load(path.read_text())
    return [
        BenchmarkOpponent(bot_name=o["bot"], deck_name=o["deck"])
        for o in raw["benchmark"]["opponents"]
    ]


def run_benchmark(
    agent_bot: Bot,
    agent_deck_name: str,
    *,
    matches_per_opponent: int | None = None,
    catalog: DeckCatalog | None = None,
    seed: int = 0,
    run_name: str = "benchmark",
) -> TrainingReporter:
    """Pit an agent bot against the frozen benchmark roster."""
    catalog = catalog or DeckCatalog()
    arena = load_arena()
    training = load_training_config()
    matches = matches_per_opponent or training.raw["eval"]["matches_per_opponent"]
    opponents = load_benchmark_opponents()
    reporter = TrainingReporter(run_name=run_name)
    rng = np.random.default_rng(seed)

    agent_deck = catalog.resolve(agent_deck_name)
    for opp in opponents:
        opp_deck = catalog.resolve(opp.deck_name)
        opp_bot = get_bot(
            opp.bot_name,
            catalog=catalog,
            deck_name=opp.deck_name,
            rng=rng,
            skill_tier=training.opponents.skill_tier,
        )
        for i in range(matches):
            report = run_match_detailed(
                arena,
                agent_deck,
                opp_deck,
                agent_bot,
                opp_bot,
                seed=int(rng.integers(0, 2**31)),
                bottom_deck_name=agent_deck_name,
                top_deck_name=opp.deck_name,
            )
            reporter.record_report(
                report,
                agent_side=Side.BOTTOM,
                agent_deck=agent_deck_name,
                opponent_deck=opp.deck_name,
                opponent_bot=opp.bot_name,
                opponent_kind="benchmark",
                stage="benchmark",
            )
    return reporter
