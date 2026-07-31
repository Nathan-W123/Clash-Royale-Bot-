"""Damage application: unit attacks (with splash) and spell resolution.

All damage flows through these helpers so the event log stays consistent.
Events are dicts consumed by rewards/metrics:
  {"type": "tower_damage", "side": <tower side>, "amount": float}
  {"type": "tower_fall",   "side": <tower side>, "kind": str}
  {"type": "death",        "side": <unit side>,  "value": float}   # elixir value
"""
from __future__ import annotations

from src.simulator.entities import PendingSpell, Tower, Unit
from src.simulator.targeting import dist, edge_dist


def damage_tower(tower: Tower, amount: float, events: list[dict]) -> None:
    amount = min(amount, tower.hp)
    if amount <= 0:
        return
    tower.hp -= amount
    events.append({"type": "tower_damage", "side": tower.side, "amount": amount})
    if tower.hp <= 0:
        events.append({"type": "tower_fall", "side": tower.side, "kind": tower.kind})


def damage_unit(unit: Unit, amount: float, events: list[dict]) -> None:
    if unit.hp <= 0:
        return
    if unit.shield_hp > 0:
        absorbed = min(amount, unit.shield_hp)
        unit.shield_hp -= absorbed
        amount -= absorbed
    if amount <= 0:
        return
    unit.hp -= amount
    if unit.hp <= 0:
        events.append({"type": "death", "side": unit.side, "value": unit.elixir_value})


def apply_attack(
    attacker: Unit | Tower,
    target: Unit | Tower,
    enemy_units: list[Unit],
    events: list[dict],
    damage_scale: float = 1.0,
) -> None:
    """Deal one hit; splash attackers damage everything legal around the target.

    `damage_scale` carries the per-hit multipliers that live outside the
    attacker's persistent state — charge and inferno ramp (see
    `src.simulator.mechanics`) — so those mechanics don't have to smuggle
    themselves through `damage_multiplier`, which belongs to Rage and hero
    buffs and has a different lifetime.
    """
    dmg = attacker.stats.damage * damage_scale
    if isinstance(attacker, Unit):
        dmg *= attacker.damage_multiplier
    splash = getattr(attacker.stats, "splash_radius", 0.0)
    if isinstance(target, Tower):
        damage_tower(target, dmg, events)
    else:
        damage_unit(target, dmg, events)
    if splash > 0 and not isinstance(target, Tower):
        for e in enemy_units:
            if e.id == target.id or e.hp <= 0:
                continue
            if not isinstance(attacker, Tower) and not attacker.can_target(e.flying, e.is_building):
                continue
            if dist(target.x, target.y, e.x, e.y) <= splash:
                damage_unit(e, dmg, events)


def resolve_spell(
    spell: PendingSpell,
    enemy_units: list[Unit],
    enemy_towers: list[Tower],
    events: list[dict],
) -> None:
    """Area damage to all enemy units and towers in radius (reduced vs towers)."""
    for e in enemy_units:
        if e.hp > 0 and dist(spell.x, spell.y, e.x, e.y) <= spell.radius + e.radius:
            damage_unit(e, spell.damage, events)
    for t in enemy_towers:
        if t.hp > 0 and dist(spell.x, spell.y, t.x, t.y) <= spell.radius + t.radius:
            damage_tower(t, spell.damage * spell.tower_multiplier, events)
