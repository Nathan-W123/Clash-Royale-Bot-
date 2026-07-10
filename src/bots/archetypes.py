"""Archetype-specific champion bots (Ultimate Champion tier heuristics)."""
from __future__ import annotations

from src.bots import base, counters, placement
from src.bots.base import Action, Bot
from src.simulator.constants import CardType, Side, TargetType
from src.simulator.engine import BattleEngine


class _ArchetypeBot:
    """Shared champion fundamentals with archetype-specific offense."""

    archetype: str = "generic"
    push_lanes: tuple[str, ...] = ("right",)
    win_cards: tuple[str, ...] = ()
    siege_cards: tuple[str, ...] = ()

    def __init__(self, name: str | None = None):
        self.name = name or f"champion_{self.archetype}"

    def decide(self, engine: BattleEngine, side: Side) -> Action | None:
        player = engine.players[side]
        threats = base.threat_units(engine, side)
        hand = {c.name for c in player.hand}

        spell = base.best_spell_action(engine, side, min_value=2.0 if threats else 2.5)
        if spell is not None:
            return spell

        if threats:
            counter = self._counter_threat(engine, side, threats, hand)
            if counter is not None:
                return counter

        if base.should_wait_for_elixir(player, engine, side, threats):
            return None

        offense = self._offense(engine, side, hand)
        if offense is not None:
            return offense

        return self._cycle(engine, side)

    def _counter_threat(
        self,
        engine: BattleEngine,
        side: Side,
        threats: list,
        hand: set[str],
    ) -> Action | None:
        player = engine.players[side]
        threat = max(threats, key=lambda u: u.elixir_value)
        counter_name = counters.pick_counter(threat.stats.name, hand)
        slot = base.hand_slot(player, counter_name) if counter_name else None
        if slot is None or not player.can_afford(slot):
            slot = counters.generic_counter(threat, player)
        if slot is None:
            return None
        lane = "left" if threat.x < engine.arena.width / 2 else "right"
        x, y = placement.own_defense(engine, side, lane)
        card = player.hand[slot]
        if card.type == CardType.BUILDING:
            x, y = placement.own_back(engine, side, lane)
        if engine.legal_deploy(side, card, x, y):
            return Action(slot, x, y)
        return None

    def _offense(self, engine: BattleEngine, side: Side, hand: set[str]) -> Action | None:
        player = engine.players[side]
        for win in self.win_cards:
            if win not in hand:
                continue
            slot = base.hand_slot(player, win)
            if slot is None or not player.can_afford(slot):
                continue
            card = player.hand[slot]
            lane = self.push_lanes[0]
            if card.type == CardType.TROOP and card.targets == TargetType.BUILDINGS_ONLY:
                x, y = placement.own_back(engine, side, lane)
            else:
                x, y = placement.bridge_push(engine, side, lane)
            if player.elixir >= card.cost + 2 and engine.legal_deploy(side, card, x, y):
                return Action(slot, x, y)
        for siege in self.siege_cards:
            if siege not in hand:
                continue
            slot = base.hand_slot(player, siege)
            if slot is None or not player.can_afford(slot):
                continue
            card = player.hand[slot]
            x, y = placement.siege_spot(engine, side)
            if player.elixir >= card.cost + 1 and engine.legal_deploy(side, card, x, y):
                return Action(slot, x, y)
        return None

    def _cycle(self, engine: BattleEngine, side: Side) -> Action | None:
        player = engine.players[side]
        slots = base.affordable_slots(player)
        if not slots:
            return None
        slot = min(slots, key=lambda s: player.hand[s].cost)
        card = player.hand[slot]
        x, y = placement.own_back(engine, side, "right")
        if engine.legal_deploy(side, card, x, y):
            return Action(slot, x, y)
        return None


class RusherBot(_ArchetypeBot):
    archetype = "rusher"
    push_lanes = ("right", "left")
    win_cards = ("hog_rider", "giant")


class ControlBot(_ArchetypeBot):
    archetype = "control"
    push_lanes = ("right",)
    win_cards = ("giant", "baby_dragon")
    siege_cards = ("cannon",)


class SiegeBot(_ArchetypeBot):
    archetype = "siege"
    push_lanes = ("right",)
    win_cards = ()
    siege_cards = ("tesla", "cannon")


class BeatdownBot(_ArchetypeBot):
    archetype = "beatdown"
    push_lanes = ("right",)
    win_cards = ("giant",)


class GenericChampionBot(_ArchetypeBot):
    archetype = "generic"
    push_lanes = ("right",)
    win_cards = ("giant", "hog_rider", "mini_pekka")


def bot_for_archetype(archetype: str) -> Bot:
    mapping = {
        "rusher": RusherBot,
        "control": ControlBot,
        "siege": SiegeBot,
        "beatdown": BeatdownBot,
        "generic": GenericChampionBot,
        "training_mirror": GenericChampionBot,
    }
    cls = mapping.get(archetype, GenericChampionBot)
    return cls()
