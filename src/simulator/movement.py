"""Unit movement: straight-line waypoints with bridge routing across the river."""
from __future__ import annotations

import math

from src.simulator.cards import ArenaConfig
from src.simulator.entities import Unit


def _needs_bridge(unit: Unit, ty: float, arena: ArenaConfig) -> bool:
    if unit.flying:
        return False
    below = unit.y < arena.river_y_min
    above = unit.y > arena.river_y_max
    t_below = ty < arena.river_y_min
    t_above = ty > arena.river_y_max
    return (below and not t_below) or (above and not t_above)


def waypoint_toward(unit: Unit, tx: float, ty: float, arena: ArenaConfig) -> tuple[float, float]:
    """Next point to walk toward. Ground units route via the nearest bridge."""
    if not _needs_bridge(unit, ty, arena):
        return tx, ty
    bx = min(arena.bridge_xs, key=lambda b: abs(b - unit.x))
    on_bridge_line = abs(unit.x - bx) <= arena.bridge_half_width
    if on_bridge_line:
        # Walk straight through the river band along the bridge.
        exit_y = arena.river_y_max + 0.1 if unit.y < arena.river_y_min else arena.river_y_min - 0.1
        return bx, exit_y
    # Walk along own side to the bridge entrance first (no corner-cutting into water).
    entry_y = arena.river_y_min - 0.1 if unit.y < arena.river_y_min else arena.river_y_max + 0.1
    return bx, entry_y


def step_toward(unit: Unit, wx: float, wy: float, dt: float) -> None:
    dx, dy = wx - unit.x, wy - unit.y
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return
    move = min(unit.stats.speed * dt, d)
    unit.x += dx / d * move
    unit.y += dy / d * move
