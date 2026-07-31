"""Special-cased spell effects beyond generic area damage: freeze, rage,
tornado, poison, clone.

``combat.resolve_spell`` only ever applies flat area damage, which is correct
for fireball/zap/arrows/etc. but silently drops the signature mechanic of a
few spells (freeze's stun, rage's buff, tornado's pull, poison's persistent
tick damage, clone's fragile duplicates) — those cards were configured with
placeholder near-zero ``spell_damage`` and, until this module, never did
anything else. Dispatched by card name from ``BattleEngine.tick()``.

Persistent spells (poison, tornado, rage) re-apply over several ticks via
``PendingSpell.ticks_left``; ``PERSISTENT_TICKS`` is the single table of how
many applications each gets and how far apart. `BattleEngine.tick()` calls
`reschedule` after each application.

Remaining simplifications, noted because they trade fidelity for scope:
  - Rage speeds up movement and boosts damage per hit; it does not shorten
    attack cooldown (real Rage also speeds up attack rate).
  - Tornado pulls at a constant rate over its duration rather than with the
    real game's ease-in toward the centre.
  - Clone excludes heroes/champions, to sidestep duplicating hero-ability
    charge state, and gives copies no special post-clone deploy-delay
    exemption — they get the same ~1s inert window as any freshly spawned
    unit, a minor simplification.

A spell constructed by hand with ``ticks_left = 0`` (as engine-level tests
do) still applies its whole effect in one step: the per-tick share is derived
from ``ticks_left``, so single-shot and persistent forms stay consistent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.simulator.targeting import dist

if TYPE_CHECKING:
    from src.simulator.engine import BattleEngine
    from src.simulator.entities import PendingSpell, Unit

FREEZE_DURATION = 4.0
RAGE_DURATION = 6.0
RAGE_SPEED_MULTIPLIER = 1.35
RAGE_DAMAGE_MULTIPLIER = 1.35
TORNADO_PULL_DISTANCE = 3.0
TORNADO_DURATION = 1.0
POISON_TOTAL_TICKS = 8
POISON_TICK_INTERVAL = 1.0
CLONE_HP = 1.0

SPECIAL_EFFECTS = frozenset({"freeze", "rage", "tornado", "poison", "clone"})

# card name -> (total applications, seconds between them).
#   poison  damages on every tick, which is its whole identity.
#   tornado pulls a share of the total distance per tick, so units are
#           dragged *through* the zone rather than teleported — that is what
#           makes a tornado-into-king-tower activation a real play.
#   rage    re-applies so units entering the zone later are caught too,
#           instead of only whoever happened to be standing there at cast.
PERSISTENT_TICKS: dict[str, tuple[int, float]] = {
    "poison": (POISON_TOTAL_TICKS, POISON_TICK_INTERVAL),
    "tornado": (10, TORNADO_DURATION / 10),
    "rage": (12, RAGE_DURATION / 12),
}


def total_ticks(card_name: str) -> int:
    """Applications a freshly-cast `card_name` gets (0 = single-shot)."""
    return PERSISTENT_TICKS.get(card_name, (0, 0.0))[0]


def tick_interval(card_name: str) -> float:
    return PERSISTENT_TICKS.get(card_name, (0, 0.0))[1]


def damages_this_tick(spell: "PendingSpell") -> bool:
    """Whether this application also deals its area damage.

    Poison's damage is per-tick; every other persistent spell deals its
    damage once, on the first application, and later ticks carry only the
    effect.
    """
    if spell.card_name == "poison":
        return True
    total = total_ticks(spell.card_name)
    return total == 0 or spell.ticks_left >= total


def reschedule(spell: "PendingSpell", engine: "BattleEngine") -> None:
    """Queue the next application of a persistent spell, if any remain."""
    if spell.ticks_left <= 1:
        return
    from src.simulator.entities import PendingSpell as _PendingSpell

    engine.spells.append(_PendingSpell(
        side=spell.side, x=spell.x, y=spell.y, radius=spell.radius,
        damage=spell.damage, tower_multiplier=spell.tower_multiplier,
        resolve_at=engine.time + tick_interval(spell.card_name),
        card_name=spell.card_name, ticks_left=spell.ticks_left - 1,
        zone_until=spell.zone_until))


def apply_freeze(spell: "PendingSpell", engine: "BattleEngine", enemy_units: list["Unit"]) -> None:
    """Stun affected troops/buildings: no targeting, movement, or attacks.

    Also resets charge, inferno ramp, and the attack wind-up — see
    `mechanics.on_stun`. Freeze is just a very long stun.
    """
    from src.simulator import mechanics

    for u in enemy_units:
        if u.hp > 0 and dist(spell.x, spell.y, u.x, u.y) <= spell.radius + u.radius:
            u.frozen_until = engine.time + FREEZE_DURATION
            mechanics.on_stun(u)


def apply_rage(spell: "PendingSpell", engine: "BattleEngine", friendly_units: list["Unit"]) -> None:
    """Buff allied units in the zone: faster movement and harder hits.

    The buff expires when the *zone* does, not `RAGE_DURATION` after whenever
    this particular tick happened — otherwise each re-application would push
    the expiry out and a 6-second rage would last twelve.
    """
    until = spell.zone_until or (engine.time + RAGE_DURATION)
    for u in friendly_units:
        if u.hp > 0 and dist(spell.x, spell.y, u.x, u.y) <= spell.radius + u.radius:
            u.speed_multiplier = RAGE_SPEED_MULTIPLIER
            u.damage_multiplier = RAGE_DAMAGE_MULTIPLIER
            u.buff_until = until


def apply_tornado(spell: "PendingSpell", engine: "BattleEngine", enemy_units: list["Unit"]) -> None:
    """Drag enemy troops (not buildings) toward the cast point.

    Each application moves its share of `TORNADO_PULL_DISTANCE`, so a
    persistent cast drags units continuously across its duration while a
    hand-built single-tick spell still moves them the full distance at once.
    """
    a = engine.arena
    share = TORNADO_PULL_DISTANCE / max(spell.ticks_left, 1)
    for u in enemy_units:
        if u.hp <= 0 or u.is_building:
            continue
        d = dist(spell.x, spell.y, u.x, u.y)
        if d <= spell.radius + u.radius and d > 1e-6:
            pull = min(share, d)
            u.x = min(max(u.x + (spell.x - u.x) / d * pull, 0.0), a.width - 0.01)
            u.y = min(max(u.y + (spell.y - u.y) / d * pull, 0.0), a.height - 0.01)


def apply_clone(spell: "PendingSpell", engine: "BattleEngine", friendly_units: list["Unit"]) -> None:
    """Spawn a 1-HP copy of every allied troop (not building, not hero) in radius."""
    for u in list(friendly_units):
        if u.hp <= 0 or u.is_building or u.is_hero:
            continue
        if dist(spell.x, spell.y, u.x, u.y) <= spell.radius + u.radius:
            engine.spawn_units(u.stats, u.side, u.x, u.y, count=1, hp_override=CLONE_HP)
