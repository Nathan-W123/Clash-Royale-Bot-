"""Shared bot utilities and the Bot protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.simulator.cards import CardStats
from src.simulator.constants import CardType, Side
from src.simulator.engine import BattleEngine
from src.simulator.entities import Tower, Unit


@dataclass(frozen=True)
class Action:
    slot: int
    x: float
    y: float


class Bot(Protocol):
    name: str

    def decide(self, engine: BattleEngine, side: Side) -> Action | None:
        ...


def hand_slot(player, card_name: str) -> int | None:
    for i, c in enumerate(player.hand):
        if c.name == card_name:
            return i
    return None


def affordable_slots(player) -> list[int]:
    return [i for i, c in enumerate(player.hand) if player.can_afford(i)]


def card_in_hand(player, name: str) -> bool:
    return any(c.name == name for c in player.hand)


def enemy_units(engine: BattleEngine, side: Side) -> list[Unit]:
    return engine.units_of(side.other)


def friendly_units(engine: BattleEngine, side: Side) -> list[Unit]:
    return engine.units_of(side)


def alive_princesses(engine: BattleEngine, side: Side) -> list[Tower]:
    return [t for t in engine.towers_of(side) if not t.is_king and t.hp > 0]


def threat_units(engine: BattleEngine, side: Side) -> list[Unit]:
    """Enemy units that have crossed the river toward our towers."""
    a = engine.arena
    river_mid = (a.river_y_min + a.river_y_max) / 2
    threats = []
    for u in enemy_units(engine, side):
        if u.is_building:
            continue
        if side == Side.BOTTOM and u.y < river_mid:
            threats.append(u)
        elif side == Side.TOP and u.y >= river_mid:
            threats.append(u)
    return threats


def spell_value(engine: BattleEngine, side: Side, card: CardStats, x: float, y: float) -> float:
    """Elixir-value estimate for a spell at (x, y).

    Only credits value the spell actually converts: full value for units it
    kills, half for units it takes below half HP, tower chip only when the
    spell meaningfully damages a tower.
    """
    if card.type != CardType.SPELL:
        return 0.0
    score = 0.0
    r = card.spell_radius
    for u in enemy_units(engine, side):
        if np.hypot(u.x - x, u.y - y) <= r + u.radius:
            if card.spell_damage >= u.hp:
                score += u.elixir_value
            elif card.spell_damage >= 0.5 * u.hp:
                score += 0.5 * u.elixir_value
    tower_chip = card.spell_damage * card.tower_multiplier
    for t in engine.towers_of(side.other):
        if np.hypot(t.x - x, t.y - y) <= r + t.radius:
            if tower_chip >= t.hp:
                score += 10.0  # finisher
            elif tower_chip >= 80.0:
                score += 1.0
    return score


def best_spell_action(
    engine: BattleEngine,
    side: Side,
    min_value: float = 1.5,
) -> Action | None:
    player = engine.players[side]
    best: Action | None = None
    best_score = min_value
    for slot in affordable_slots(player):
        card = player.hand[slot]
        if card.type != CardType.SPELL:
            continue
        # A spell must beat its own cost, not just the caller's floor.
        threshold = max(best_score, card.cost + 0.5)
        for u in threat_units(engine, side) or enemy_units(engine, side):
            score = spell_value(engine, side, card, u.x, u.y)
            if score > threshold:
                best_score = threshold = score
                best = Action(slot, u.x, u.y)
        for t in alive_princesses(engine, side.other):
            score = spell_value(engine, side, card, t.x, t.y)
            if score > threshold:
                best_score = threshold = score
                best = Action(slot, t.x, t.y)
    return best


def should_wait_for_elixir(player, engine: BattleEngine, side: Side, threats: list[Unit]) -> bool:
    """Champion discipline: don't leak elixir, but defend under pressure."""
    if threats:
        return False
    if player.elixir >= player.elixir_max - 0.5:
        return False
    if player.elixir >= 8.0 and engine.double_elixir:
        return False
    # No pressure: bank toward a real push instead of drip-feeding cards.
    return player.elixir < 7.0
