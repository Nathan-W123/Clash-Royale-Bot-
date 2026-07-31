"""Freeze/rage/tornado: signature effects beyond the generic spell-damage path."""
import pytest

from src.simulator.constants import Side
from src.simulator.entities import PendingSpell
from src.simulator import spell_effects
from tests.conftest import dummy_stats, make_engine, spawn_unit


def test_freeze_stuns_movement_and_attacks(cards, arena):
    eng = make_engine(cards, arena)
    knight = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 20.0)
    knight.target_id = None
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=20.0, radius=3.0,
                                   damage=1.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="freeze"))
    eng.tick()
    assert knight.frozen_until == pytest.approx(spell_effects.FREEZE_DURATION)

    y_after_freeze = knight.y
    for _ in range(20):
        eng.tick()
    assert knight.y == pytest.approx(y_after_freeze)  # never moved while frozen

    for _ in range(int(spell_effects.FREEZE_DURATION / arena.dt) + 5):
        eng.tick()
    assert knight.frozen_until == 0.0  # cleared once expired


def test_frozen_unit_can_still_be_damaged(cards, arena):
    eng = make_engine(cards, arena)
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 20.0)
    bag.frozen_until = 100.0
    from src.simulator import combat
    events = []
    combat.damage_unit(bag, 500.0, events)
    assert bag.hp == pytest.approx(1_000_000.0 - 500.0)


def test_rage_buffs_own_units_not_enemy(cards, arena):
    eng = make_engine(cards, arena)
    ally = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 10.0)
    enemy = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 20.0)
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=10.0, radius=5.0,
                                   damage=1.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="rage"))
    eng.tick()
    assert ally.speed_multiplier == pytest.approx(spell_effects.RAGE_SPEED_MULTIPLIER)
    assert ally.damage_multiplier == pytest.approx(spell_effects.RAGE_DAMAGE_MULTIPLIER)
    assert enemy.speed_multiplier == pytest.approx(1.0)  # rage never hits enemies
    assert enemy.hp == cards["knight"].hp  # and deals no damage


def test_rage_expires(cards, arena):
    eng = make_engine(cards, arena)
    ally = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 10.0)
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=10.0, radius=5.0,
                                   damage=1.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="rage"))
    eng.tick()
    assert ally.speed_multiplier > 1.0
    for _ in range(int(spell_effects.RAGE_DURATION / arena.dt) + 5):
        eng.tick()
    assert ally.speed_multiplier == pytest.approx(1.0)
    assert ally.damage_multiplier == pytest.approx(1.0)


def test_tornado_pulls_enemy_troops_toward_center(cards, arena):
    eng = make_engine(cards, arena)
    enemy = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 25.0)
    start_y = enemy.y
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=20.0, radius=5.5,
                                   damage=40.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="tornado"))
    eng.tick()
    assert enemy.y < start_y  # pulled toward y=20
    assert start_y - enemy.y == pytest.approx(spell_effects.TORNADO_PULL_DISTANCE, abs=0.05)


def test_tornado_does_not_pull_buildings(cards, arena):
    eng = make_engine(cards, arena)
    cannon = spawn_unit(eng, cards["cannon"], Side.TOP, 9.0, 25.0)
    start = (cannon.x, cannon.y)
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=20.0, radius=5.5,
                                   damage=40.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="tornado"))
    eng.tick()
    assert (cannon.x, cannon.y) == start
