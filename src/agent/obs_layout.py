"""Observation layout for the policy network (numpy, no torch).

Tensor shapes
-------------
``encode_spatial(engine, side)`` -> ``(SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)``
    Coarse 9x16 grid (2x2 arena tiles per cell). Channels:

    0  friendly troop HP density
    1  enemy troop HP density
    2  friendly building HP density
    3  enemy building HP density
    4  friendly tower HP
    5  enemy tower HP
    6  friendly pending spell damage
    7  enemy pending spell damage
    8  friendly unit presence
    9  enemy unit presence

``encode_hand(player, card_to_id)`` -> ``(CARD_FEATURE_DIM,)`` float32
    ``[hand_id_0, ..., hand_id_3, next_id, elixir / 10]``
    Unknown cards map to index 0.
"""
from __future__ import annotations

import numpy as np

from src.simulator.constants import HAND_SIZE, PLACE_COLS, PLACE_ROWS, Side
from src.simulator.engine import BattleEngine
from src.simulator.player import PlayerState

from src.agent.masking import frame_y

SPATIAL_CHANNELS = 10
CARD_FEATURE_DIM = HAND_SIZE + 2  # 4 hand slots + next card + elixir scalar


def _grid_cell(side: Side, x: float, y: float, arena_height: float) -> tuple[int, int] | None:
    fy = frame_y(side, y, arena_height)
    col = int(x // 2)
    row = int(fy // 2)
    if 0 <= col < PLACE_COLS and 0 <= row < PLACE_ROWS:
        return col, row
    return None


def encode_spatial(engine: BattleEngine, side: Side) -> np.ndarray:
    """Multi-channel coarse arena grid from ``side``'s perspective."""
    spatial = np.zeros((SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS), np.float32)
    h = engine.arena.height

    for u in engine.units:
        if u.hp <= 0:
            continue
        cell = _grid_cell(side, u.x, u.y, h)
        if cell is None:
            continue
        col, row = cell
        friendly = u.side == side
        ch = (0 if friendly else 1) + (2 if u.is_building else 0)
        spatial[ch, row, col] += u.hp / 1000.0
        spatial[8 if friendly else 9, row, col] += 0.2

    for t in engine.towers:
        if t.hp <= 0:
            continue
        cell = _grid_cell(side, t.x, t.y, h)
        if cell is None:
            continue
        col, row = cell
        ch = 4 if t.side == side else 5
        spatial[ch, row, col] += t.hp / 1000.0

    for s in engine.spells:
        cell = _grid_cell(side, s.x, s.y, h)
        if cell is None:
            continue
        col, row = cell
        ch = 6 if s.side == side else 7
        rad = int(np.ceil(s.radius / 2.0))
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < PLACE_ROWS and 0 <= cc < PLACE_COLS:
                    spatial[ch, rr, cc] += s.damage / 1000.0

    np.clip(spatial, 0.0, 30.0, out=spatial)
    return spatial


def _card_id(card, card_to_id: dict[str, int]) -> float:
    return float(card_to_id.get(card.name, 0))


def encode_hand(player: PlayerState, card_to_id: dict[str, int]) -> np.ndarray:
    """Hand card ids, next card id, and normalized elixir."""
    ids = [_card_id(c, card_to_id) for c in player.hand]
    ids.append(_card_id(player.next_card, card_to_id))
    ids.append(player.elixir / 10.0)
    return np.asarray(ids, dtype=np.float32)


SCALAR_DIM = 17


def encode_scalars(engine: BattleEngine, side: Side) -> np.ndarray:
    """Match-state scalars: elixir, clock/phase, hand costs, tower status.

    ``(SCALAR_DIM,)`` float32:
    [own_elixir, opp_elixir, cost_0..3, affordable_0..3, time_remaining,
     double_elixir, overtime, own_king_activated, enemy_king_activated,
     own_princess_count/2, enemy_princess_count/2]
    """
    me = engine.players[side]
    opp = engine.players[side.other]

    def king(s: Side):
        return next(t for t in engine.towers if t.side == s and t.is_king)

    def princesses(s: Side) -> float:
        return sum(1 for t in engine.towers
                   if t.side == s and not t.is_king and t.hp > 0) / 2.0

    return np.asarray(
        [me.elixir / 10.0, opp.elixir / 10.0]
        + [me.hand[i].cost / 10.0 for i in range(HAND_SIZE)]
        + [float(me.can_afford(i)) for i in range(HAND_SIZE)]
        + [max(0.0, engine.regulation - engine.time) / max(engine.regulation, 1.0),
           float(engine.double_elixir), float(engine.overtime),
           float(king(side).activated), float(king(side.other).activated),
           princesses(side), princesses(side.other)],
        dtype=np.float32)


def encode_obs(engine: BattleEngine, side: Side, card_to_id: dict[str, int]) -> dict[str, np.ndarray]:
    """Canonical policy observation: spatial grid + hand card ids + scalars.

    The single encoding shared by the gym env, BC data generation, and
    policy-backed opponents/bots — keep them identical.
    """
    player = engine.players[side]
    ids = [int(card_to_id.get(c.name, 0)) for c in player.hand]
    ids.append(int(card_to_id.get(player.next_card.name, 0)))
    return {
        "spatial": encode_spatial(engine, side),
        "cards": np.asarray(ids, dtype=np.int64),
        "vector": encode_scalars(engine, side),
    }
