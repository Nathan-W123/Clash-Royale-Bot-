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


def next_position(unit: Unit, wx: float, wy: float, dt: float,
                  extra_speed: float = 1.0) -> tuple[float, float]:
    """Where `unit` would end up walking toward (wx, wy) this tick, ignoring collision.

    `extra_speed` carries transient multipliers that aren't part of the
    unit's persistent state — a Prince's charge (see `mechanics.charge_speed`)
    — separately from `speed_multiplier`, which belongs to Rage.
    """
    dx, dy = wx - unit.x, wy - unit.y
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return unit.x, unit.y
    move = min(unit.stats.speed * unit.speed_multiplier * extra_speed * dt, d)
    return unit.x + dx / d * move, unit.y + dy / d * move


# Sidestep angles (radians) tried, in order, when the direct step is blocked:
# progressively more lateral. The near-perpendicular entries matter — against
# a body directly ahead, any forward-leaning step still closes the distance,
# so only a mostly-sideways one is actually clear.
_SIDESTEP_ANGLES = (0.6, -0.6, 1.2, -1.2, 1.6, -1.6)


def avoid_obstacle(unit: Unit, nx: float, ny: float, all_units: list[Unit],
                   arena: ArenaConfig) -> tuple[float, float]:
    """Steer around a blocking body instead of standing still against it.

    Straight-line movement plus reject-on-overlap means a unit that walks
    into the side of a tank simply stops and stays stopped: its waypoint
    never changes, so it re-proposes the same blocked step forever. Real
    units flow around each other, and the difference shows up in exactly the
    situations players engineer on purpose — funnelling a push into one lane,
    or a tank shielding the support behind it.

    A cheap fix rather than a full navmesh: try the direct step, then a few
    rotations of the step vector, and take the first that is clear. That
    recovers "walk around it" behaviour without a pathfinder, and if
    everything is blocked the unit holds position exactly as before.
    """
    from src.simulator import collision

    if not collision.blocked(unit, nx, ny, all_units):
        return nx, ny
    dx, dy = nx - unit.x, ny - unit.y
    step = math.hypot(dx, dy)
    if step < 1e-9:
        return nx, ny
    base = math.atan2(dy, dx)
    for offset in _SIDESTEP_ANGLES:
        angle = base + offset
        cx = min(max(unit.x + math.cos(angle) * step, 0.0), arena.width - 0.01)
        cy = min(max(unit.y + math.sin(angle) * step, 0.0), arena.height - 0.01)
        if _crosses_river(unit, cy, arena):
            continue
        if not collision.blocked(unit, cx, cy, all_units):
            return cx, cy
    return nx, ny


def _crosses_river(unit: Unit, cy: float, arena: ArenaConfig) -> bool:
    """Don't let a sidestep walk a ground unit into the water."""
    return not unit.flying and arena.river_y_min <= cy <= arena.river_y_max


def step_toward(unit: Unit, wx: float, wy: float, dt: float) -> None:
    unit.x, unit.y = next_position(unit, wx, wy, dt)
