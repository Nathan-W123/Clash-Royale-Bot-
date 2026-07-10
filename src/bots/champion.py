"""Champion-tier bot: picks archetype logic from deck composition."""
from __future__ import annotations

from src.bots import base
from src.bots.archetypes import bot_for_archetype
from src.bots.base import Action, Bot
from src.decks.catalog import DeckCatalog
from src.simulator.constants import Side
from src.simulator.engine import BattleEngine


class ChampionBot:
    """Ultimate Champion skill tier — reactive defense + archetype offense."""

    name = "champion"

    def __init__(self, archetype: str = "generic"):
        self._inner = bot_for_archetype(archetype)
        self.archetype = archetype

    @classmethod
    def for_deck_name(cls, catalog: DeckCatalog, deck_name: str) -> "ChampionBot":
        return cls(catalog.archetype_for_deck(deck_name))

    def decide(self, engine: BattleEngine, side: Side) -> Action | None:
        return self._inner.decide(engine, side)


class RandomBot:
    """Weak baseline — random legal deploys. Not UC tier."""

    name = "random"

    def __init__(self, rng):
        self.rng = rng

    def decide(self, engine: BattleEngine, side: Side) -> Action | None:
        player = engine.players[side]
        slots = base.affordable_slots(player)
        if not slots:
            return None
        if self.rng.random() > 0.15:
            return None
        slot = int(self.rng.choice(slots))
        card = player.hand[slot]
        a = engine.arena
        for _ in range(20):
            x = float(self.rng.uniform(1, a.width - 1))
            y = float(self.rng.uniform(1, a.height - 1))
            if engine.legal_deploy(side, card, x, y):
                return Action(slot, x, y)
        return None
