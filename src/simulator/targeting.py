"""Target acquisition, lock-on, and retarget rules for units and towers."""
from __future__ import annotations

import math

from src.simulator.constants import LOCK_BREAK_FACTOR, TargetType
from src.simulator.entities import Tower, Unit


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def edge_dist(a, b) -> float:
    """Center distance minus body radii (edge-to-edge, floored at 0)."""
    return max(0.0, dist(a.x, a.y, b.x, b.y) - a.radius - b.radius)


def unit_acquire(unit: Unit, enemies: list[Unit], enemy_towers: list[Tower]) -> int | None:
    """Pick a target for a unit: nearest legal enemy in sight, else nearest enemy tower.

    Buildings-only units consider enemy buildings in sight, else nearest tower.
    """
    best_id, best_d = None, float("inf")
    for e in enemies:
        if not unit.can_target(e.flying, e.is_building):
            continue
        d = edge_dist(unit, e)
        if d <= unit.stats.sight_range and d < best_d:
            best_id, best_d = e.id, d
    if best_id is not None:
        return best_id
    # No unit in sight: push toward the nearest enemy tower (any distance).
    for t in enemy_towers:
        d = edge_dist(unit, t)
        if d < best_d:
            best_id, best_d = t.id, d
    return best_id


def unit_keeps_lock(unit: Unit, target) -> bool:
    """Locks persist until death or the target leaves 1.5x sight range."""
    if target is None or getattr(target, "hp", 0) <= 0:
        return False
    if isinstance(target, Tower):
        return True  # towers don't move; lock until destroyed
    return edge_dist(unit, target) <= unit.stats.sight_range * LOCK_BREAK_FACTOR


def tower_acquire(tower: Tower, enemies: list[Unit]) -> int | None:
    """Towers hit air and ground; nearest enemy within attack range."""
    best_id, best_d = None, float("inf")
    for e in enemies:
        d = edge_dist(tower, e)
        if d <= tower.stats.range and d < best_d:
            best_id, best_d = e.id, d
    return best_id


def tower_keeps_lock(tower: Tower, target: Unit | None) -> bool:
    if target is None or target.hp <= 0:
        return False
    return edge_dist(tower, target) <= tower.stats.range
