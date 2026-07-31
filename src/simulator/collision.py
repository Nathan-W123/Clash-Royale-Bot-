"""Ground-body collision: troops and buildings can't occupy the same space.

This is what makes tanking, funneling, and sheltering possible at all: without
it, every unit moves through every other unit's body in a straight line, so
"the Giant absorbs hits while the Musketeer shelters behind it" only ever
worked by accident of nearest-target selection, never because anything was
physically in the way.

Scope, deliberately: flying units and towers are excluded. Flying troops
don't collide with anything in the real game (different lane). Towers live
in `BattleEngine.towers`, not `.units`, and this simulator's movement model
already routes ground units to towers via bridge waypoints rather than
letting them wander near a king tower's position, so tower collision would
add complexity for a case the routing model rarely produces; it's a known,
intentional gap, not an oversight.

Approach: reject-on-overlap rather than exact segment/circle clamping. A
unit's per-tick travel distance is well under one body radius for every card
in the roster (`speed * dt` maxes out around 0.2 tiles at dt=0.1s vs. a
0.5+0.5 tile radius sum), so "don't move if the destination overlaps
someone" converges to the same standoff distance as an exact clamp within a
fraction of a tile, at a fraction of the complexity.
"""
from __future__ import annotations

from src.simulator.entities import Unit


def blocked(mover: Unit, nx: float, ny: float, all_units: list[Unit]) -> bool:
    """True if moving `mover` to (nx, ny) would overlap another ground body.

    The mover's own current attack target is exempt — a unit must always be
    able to close the last distance onto whatever it's trying to hit.

    Hot path: this runs per moving unit per tick, so it's O(n^2) per tick over
    the unit list. Kept tight deliberately — squared distances (no sqrt), an
    axis-aligned reject before the multiply, and locals hoisted out of the
    loop. Same predicate as the obvious `dist(...) < r1 + r2` version.
    """
    if mover.flying:
        return False
    mover_radius = mover.radius
    target_id = mover.target_id
    for other in all_units:
        if other is mover or other.hp <= 0 or other.flying:
            continue
        if target_id is not None and other.id == target_id:
            continue
        reach = mover_radius + other.radius
        dx = nx - other.x
        if dx > reach or dx < -reach:
            continue
        dy = ny - other.y
        if dy > reach or dy < -reach:
            continue
        if dx * dx + dy * dy < reach * reach:
            return True
    return False
