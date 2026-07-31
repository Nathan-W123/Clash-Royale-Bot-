"""Hero ability resolution: dash, spawn, damage_buff, shield, taunt, split_on_death."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.simulator import combat
from src.simulator.targeting import dist

if TYPE_CHECKING:
    from src.simulator.engine import BattleEngine
    from src.simulator.entities import Unit


def tick_hero_abilities(engine: BattleEngine, dt: float, events: list[dict]) -> None:
    """Charge heroes on arena and auto-trigger at full charge."""
    for u in engine.units:
        if u.hp <= 0 or not u.is_hero:
            continue
        charge_time = u.stats.ability_charge
        if charge_time <= 0:
            continue
        u.ability_charge = min(1.0, u.ability_charge + dt / charge_time)
        if u.ability_charge >= 1.0:
            u.ability_charge = 0.0
            trigger_ability(u, engine, events)

    for u in engine.units:
        if u.buff_until and engine.time >= u.buff_until:
            u.damage_multiplier = 1.0
            u.speed_multiplier = 1.0
            u.buff_until = 0.0
        if u.shield_until and engine.time >= u.shield_until:
            u.shield_hp = 0.0
            u.shield_until = 0.0
        if u.frozen_until and engine.time >= u.frozen_until:
            u.frozen_until = 0.0


def trigger_ability(hero: Unit, engine: BattleEngine, events: list[dict]) -> None:
    ability = hero.stats.ability
    if ability == "dash":
        _dash(hero, engine, events)
    elif ability == "spawn":
        _spawn(hero, engine, events)
    elif ability == "damage_buff":
        _damage_buff(hero, engine, events)
    elif ability == "shield":
        _shield(hero, engine, events)
    elif ability == "taunt":
        _taunt(hero, engine, events)
    elif ability == "split_on_death":
        _split_spawn(hero, engine, events)
    events.append({
        "type": "hero_ability",
        "side": hero.side,
        "hero": hero.stats.name,
        "ability": ability,
    })


def on_unit_death(unit: Unit, engine: BattleEngine, events: list[dict]) -> None:
    """Passive death effects (e.g. Hero Ice Golem splits)."""
    if unit.stats.ability == "split_on_death":
        _split_spawn(unit, engine, events)


def _dash(hero: Unit, engine: BattleEngine, events: list[dict]) -> None:
    enemies = engine.units_of(hero.side.other)
    towers = engine.towers_of(hero.side.other)
    target = None
    best_d = float("inf")
    for e in enemies:
        d = dist(hero.x, hero.y, e.x, e.y)
        if d < best_d:
            best_d, target = d, e
    for t in towers:
        d = dist(hero.x, hero.y, t.x, t.y)
        if d < best_d:
            best_d, target = d, t
    if target is None:
        return
    hero.x, hero.y = target.x, target.y
    hero.target_id = None
    radius = hero.stats.ability_radius
    dmg = hero.stats.ability_damage
    for e in enemies:
        if e.hp > 0 and dist(hero.x, hero.y, e.x, e.y) <= radius + e.radius:
            combat.damage_unit(e, dmg, events)
    for t in towers:
        if t.hp > 0 and dist(hero.x, hero.y, t.x, t.y) <= radius + t.radius:
            combat.damage_tower(t, dmg * 0.35, events)


def _spawn(hero: Unit, engine: BattleEngine, events: list[dict]) -> None:
    spawn_name = hero.stats.ability_spawn
    if not spawn_name or spawn_name not in engine.cards:
        return
    card = engine.cards[spawn_name]
    count = hero.stats.ability_spawn_count or card.count
    engine.spawn_units(card, hero.side, hero.x, hero.y, count)


def _split_spawn(unit: Unit, engine: BattleEngine, events: list[dict]) -> None:
    spawn_name = unit.stats.ability_spawn
    if not spawn_name or spawn_name not in engine.cards:
        return
    card = engine.cards[spawn_name]
    count = unit.stats.ability_spawn_count or 2
    engine.spawn_units(card, unit.side, unit.x, unit.y, count)


def _damage_buff(hero: Unit, engine: BattleEngine, events: list[dict]) -> None:
    mult = hero.stats.ability_damage if hero.stats.ability_damage > 0 else 1.5
    duration = hero.stats.ability_duration if hero.stats.ability_duration > 0 else 3.0
    hero.damage_multiplier = mult
    hero.buff_until = engine.time + duration


def _shield(hero: Unit, engine: BattleEngine, events: list[dict]) -> None:
    amount = hero.stats.ability_damage if hero.stats.ability_damage > 0 else 400.0
    duration = hero.stats.ability_duration if hero.stats.ability_duration > 0 else 5.0
    hero.shield_hp = amount
    hero.shield_until = engine.time + duration


def _taunt(hero: Unit, engine: BattleEngine, events: list[dict]) -> None:
    """Royal Taunt — spawn decoy swarm to draw fire."""
    _spawn(hero, engine, events)
