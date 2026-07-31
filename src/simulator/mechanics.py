"""Unit-specific mechanics that change which card a good player picks (#36).

The simulator's generic move/target/hit loop covers most of the roster, but a
handful of cards are *defined* by a mechanic outside it, and those mechanics
are exactly the ones that drive card choice:

**Charge** (Prince, Dark Prince, Battle Ram, Ram Rider; the Bandit and Royal
Ghost dashes). Moving unimpeded for a few tiles makes the unit faster and its
next hit far harder. Without it, "throw a Skeleton in front of the Prince" —
one of the cheapest positive trades in the game — is worth nothing, and the
policy has no reason to learn it.

**Ramp-up** (Inferno Tower, Inferno Dragon). Damage grows while locked on a
single target and resets on retarget or stun. Without it, Inferno is just a
mediocre single-target tower and the entire "swarm/zap the inferno" and
"never tank with a Golem into an unzapped Inferno" layer of play disappears.

**Stun resets wind-ups.** Zap on a Sparky and Lightning on an Inferno are
skill-defining plays *because* the stun resets the charge, the ramp, and the
attack wind-up. Modelled generically here rather than special-cased per card.

**Death effects** (Lava Hound pups, Golem golemites, Balloon and Giant
Skeleton death damage). A card whose value is mostly posthumous is priced
completely wrong without them.

**Tunnelling** (Miner, Goblin Drill). Deployable anywhere, which is a
placement rule rather than a combat one, but belongs in the same audit.

Everything is config-driven from `configs/cards.yaml`; a card with none of
these fields set behaves exactly as before.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.simulator import combat
from src.simulator.targeting import dist

if TYPE_CHECKING:
    from src.simulator.engine import BattleEngine
    from src.simulator.entities import Unit


# --------------------------------------------------------------- charge


def has_charge(unit: "Unit") -> bool:
    return unit.stats.charge_distance > 0


def advance_charge(unit: "Unit", distance_moved: float) -> None:
    """Accumulate unimpeded travel toward the charge threshold."""
    if not has_charge(unit) or unit.charged:
        return
    unit.charge_progress += distance_moved
    if unit.charge_progress >= unit.stats.charge_distance:
        unit.charged = True


def break_charge(unit: "Unit") -> None:
    """Reset the wind-up: blocked movement, a landed hit, or a stun.

    Resetting on *blocked movement* is what makes a cheap blocker work: the
    Prince that walks into a Skeleton stops accumulating and loses the charge
    it had banked.
    """
    unit.charge_progress = 0.0
    unit.charged = False


def charge_speed(unit: "Unit") -> float:
    return unit.stats.charge_speed_multiplier if unit.charged and has_charge(unit) else 1.0


# ----------------------------------------------------------------- ramp


def has_ramp(unit: "Unit") -> bool:
    return unit.stats.ramp_up_time > 0 and unit.stats.ramp_up_multiplier > 1.0


def advance_ramp(unit: "Unit", target_id: int | None, dt: float) -> None:
    """Grow the ramp while locked on one target; reset when it changes."""
    if not has_ramp(unit):
        return
    if target_id is None or target_id != unit.ramp_target_id:
        unit.ramp_target_id = target_id
        unit.ramp_time = 0.0
        return
    unit.ramp_time += dt


def break_ramp(unit: "Unit") -> None:
    unit.ramp_time = 0.0
    unit.ramp_target_id = None


def ramp_multiplier(unit: "Unit") -> float:
    if not has_ramp(unit):
        return 1.0
    progress = min(1.0, unit.ramp_time / unit.stats.ramp_up_time)
    return 1.0 + progress * (unit.stats.ramp_up_multiplier - 1.0)


# ---------------------------------------------------------------- attack


def attack_scale(unit: "Unit") -> float:
    """Damage multiplier from charge and ramp for the hit about to land."""
    scale = ramp_multiplier(unit)
    if unit.charged and has_charge(unit):
        scale *= unit.stats.charge_damage_multiplier
    return scale


def on_attack(unit: "Unit") -> None:
    """A charge is spent by connecting; the ramp is not."""
    if unit.charged:
        break_charge(unit)


# ----------------------------------------------------------------- stun


def on_stun(unit: "Unit") -> None:
    """Everything a stun resets.

    The attack cooldown goes back to full, which is why a 0.5s Zap costs a
    Sparky its entire 4-second wind-up rather than half a second.
    """
    break_charge(unit)
    break_ramp(unit)
    unit.cooldown = unit.stats.hit_speed


def apply_stun(engine: "BattleEngine", units: list["Unit"], x: float, y: float,
               radius: float, duration: float) -> None:
    """Stun every unit in radius for `duration` seconds."""
    if duration <= 0:
        return
    for u in units:
        if u.hp > 0 and dist(x, y, u.x, u.y) <= radius + u.radius:
            u.frozen_until = max(u.frozen_until, engine.time + duration)
            on_stun(u)


# ----------------------------------------------------------------- death


def on_unit_death(unit: "Unit", engine: "BattleEngine", events: list[dict]) -> None:
    """Death spawns and death damage.

    Death damage hits *enemies of the dying unit* — a Balloon's payload and a
    Giant Skeleton's bomb are offensive, so they must not clip the owner's own
    push.
    """
    stats = unit.stats
    if stats.death_spawn and stats.death_spawn in engine.cards:
        spawn = engine.cards[stats.death_spawn]
        count = stats.death_spawn_count or spawn.count
        engine.spawn_units(spawn, unit.side, unit.x, unit.y, count)
        events.append({"type": "death_spawn", "side": unit.side,
                       "card": stats.name, "spawned": stats.death_spawn})

    if stats.death_damage > 0 and stats.death_damage_radius > 0:
        radius = stats.death_damage_radius
        for e in engine.units_of(unit.side.other):
            if dist(unit.x, unit.y, e.x, e.y) <= radius + e.radius:
                combat.damage_unit(e, stats.death_damage, events)
        for t in engine.towers_of(unit.side.other):
            if dist(unit.x, unit.y, t.x, t.y) <= radius + t.radius:
                combat.damage_tower(t, stats.death_damage, events)
        events.append({"type": "death_damage", "side": unit.side,
                       "card": stats.name, "amount": stats.death_damage})
