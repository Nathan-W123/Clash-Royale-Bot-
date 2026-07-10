"""Standard deploy coordinates for scripted bots."""
from __future__ import annotations

from src.simulator.constants import Side
from src.simulator.engine import BattleEngine


def lane_x(arena, lane: str) -> float:
    return arena.princess_left_pos[0] if lane == "left" else arena.princess_right_pos[0]


def own_back(engine: BattleEngine, side: Side, lane: str = "right") -> tuple[float, float]:
    a = engine.arena
    x = lane_x(a, lane)
    y = 4.0 if side == Side.BOTTOM else a.height - 4.0
    return x, y


def own_defense(engine: BattleEngine, side: Side, lane: str = "right") -> tuple[float, float]:
    a = engine.arena
    x = lane_x(a, lane)
    y = 8.0 if side == Side.BOTTOM else a.height - 8.0
    return x, y


def bridge_push(engine: BattleEngine, side: Side, lane: str = "right") -> tuple[float, float]:
    a = engine.arena
    bridge_x = a.bridge_xs[0] if lane == "left" else a.bridge_xs[1]
    y = a.river_y_min - 1.0 if side == Side.BOTTOM else a.river_y_max + 1.0
    return bridge_x, y


def siege_spot(engine: BattleEngine, side: Side) -> tuple[float, float]:
    a = engine.arena
    x = a.width / 2
    y = 10.0 if side == Side.BOTTOM else a.height - 10.0
    return x, y


def pocket_push(engine: BattleEngine, side: Side, lane: str) -> tuple[float, float]:
    a = engine.arena
    x = lane_x(a, lane)
    princess_y = a.princess_left_pos[1] if lane == "left" else a.princess_right_pos[1]
    y = princess_y + 2.0 if side == Side.BOTTOM else a.height - princess_y - 2.0
    return x, y
