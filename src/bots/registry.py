"""Bot factory keyed by opponent config names."""
from __future__ import annotations

import numpy as np

from src.bots.archetypes import (
    BeatdownBot,
    ControlBot,
    GenericChampionBot,
    RusherBot,
    SiegeBot,
    bot_for_archetype,
)
from src.bots.base import Bot
from src.bots.champion import ChampionBot, RandomBot
from src.decks.catalog import DeckCatalog


def get_bot(
    name: str,
    catalog: DeckCatalog | None = None,
    deck_name: str | None = None,
    rng: np.random.Generator | None = None,
    skill_tier: str = "ultimate_champion",
) -> Bot:
    """Resolve a bot by config name. UC tier uses champion heuristics."""
    if name == "random":
        if rng is None:
            rng = np.random.default_rng()
        return RandomBot(rng)

    if skill_tier == "ultimate_champion":
        if name == "champion" and catalog and deck_name:
            return ChampionBot.for_deck_name(catalog, deck_name)
        if name in ("rusher", "control", "siege", "beatdown", "generic"):
            return bot_for_archetype(name)
        if name == "champion":
            return ChampionBot()

    # Named archetype aliases
    aliases = {
        "rusher": RusherBot,
        "control": ControlBot,
        "siege": SiegeBot,
        "beatdown": BeatdownBot,
    }
    if name in aliases:
        return aliases[name]()
    return GenericChampionBot()
