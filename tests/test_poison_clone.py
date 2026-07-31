"""Poison (persistent DoT, not one instant hit) and Clone (fragile duplicates)."""
import pytest

from src.simulator import spell_effects
from src.simulator.constants import Side
from src.simulator.entities import PendingSpell
from tests.conftest import dummy_stats, make_engine, spawn_unit


def _cast_poison(eng, x, y, radius=3.5, damage=40.0, side=Side.BOTTOM, resolve_at=0.0):
    eng.spells.append(PendingSpell(
        side=side, x=x, y=y, radius=radius, damage=damage, tower_multiplier=0.35,
        resolve_at=resolve_at, card_name="poison", ticks_left=spell_effects.POISON_TOTAL_TICKS))


def test_poison_deals_damage_over_multiple_ticks_not_one_hit(cards, arena):
    eng = make_engine(cards, arena)
    victim = spawn_unit(eng, dummy_stats(speed=0.0, hp=100_000.0), Side.TOP, 9.0, 20.0)
    _cast_poison(eng, 9.0, 20.0)

    eng.tick()  # first application lands immediately (spell_delay 0 in this test)
    after_one = 100_000.0 - victim.hp
    assert after_one == pytest.approx(40.0)  # a single tick, not the whole 8-tick total

    for _ in range(int(spell_effects.POISON_TICK_INTERVAL / arena.dt) + 1):
        eng.tick()
    after_two = 100_000.0 - victim.hp
    assert after_two == pytest.approx(80.0)  # second application landed ~1s later


def test_poison_stops_after_its_total_duration(cards, arena):
    eng = make_engine(cards, arena)
    victim = spawn_unit(eng, dummy_stats(speed=0.0, hp=100_000.0), Side.TOP, 9.0, 20.0)
    _cast_poison(eng, 9.0, 20.0)

    ticks_per_application = int(spell_effects.POISON_TICK_INTERVAL / arena.dt) + 1
    for _ in range(spell_effects.POISON_TOTAL_TICKS * ticks_per_application):
        eng.tick()
    total_after_full_duration = 100_000.0 - victim.hp
    assert total_after_full_duration == pytest.approx(
        40.0 * spell_effects.POISON_TOTAL_TICKS, rel=0.05)

    for _ in range(ticks_per_application * 3):  # well past the duration
        eng.tick()
    assert 100_000.0 - victim.hp == pytest.approx(total_after_full_duration)  # no further damage


def test_poison_hits_units_that_enter_the_zone_after_the_cast(cards, arena):
    """Proves the zone re-scans each tick rather than snapshotting who was
    standing there at cast time — unlike Rage's intentionally-snapshot buff."""
    eng = make_engine(cards, arena)
    _cast_poison(eng, 9.0, 20.0)
    eng.tick()  # zone is live, nobody there yet

    late_arrival = spawn_unit(eng, dummy_stats(speed=0.0, hp=100_000.0), Side.TOP, 9.0, 20.0)
    for _ in range(int(spell_effects.POISON_TICK_INTERVAL / arena.dt) + 1):
        eng.tick()
    assert late_arrival.hp < 100_000.0  # caught by a later tick despite arriving after cast


def test_clone_spawns_fragile_copy_of_ally_in_radius(cards, arena):
    eng = make_engine(cards, arena)
    original = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 10.0)
    before = len(eng.units_of(Side.BOTTOM))
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=10.0, radius=3.5,
                                   damage=1.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="clone"))
    eng.tick()

    after = eng.units_of(Side.BOTTOM)
    assert len(after) == before + 1
    clone = next(u for u in after if u.id != original.id)
    assert clone.stats.name == "knight"
    assert clone.hp == pytest.approx(spell_effects.CLONE_HP)
    assert original.hp == cards["knight"].hp  # the original is untouched


def test_clone_excludes_buildings_and_heroes(cards, arena):
    eng = make_engine(cards, arena)
    building = spawn_unit(eng, cards["cannon"], Side.BOTTOM, 9.0, 10.0)
    hero = spawn_unit(eng, cards["golden_knight"], Side.BOTTOM, 9.0, 10.5)
    before = len(eng.units_of(Side.BOTTOM))
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=10.0, radius=3.5,
                                   damage=1.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="clone"))
    eng.tick()
    assert len(eng.units_of(Side.BOTTOM)) == before  # neither got cloned


def test_clone_never_damages_anyone(cards, arena):
    eng = make_engine(cards, arena)
    # y=13.5-14.0 is outside every tower's edge-distance range (see
    # test_combat.py's identical choice), so nothing else can damage `enemy`.
    spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 13.5)
    enemy = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 14.0)  # inside the same radius
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=13.5, radius=3.5,
                                   damage=1.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="clone"))
    eng.tick()
    assert enemy.hp == enemy.stats.hp  # clone is a pure self-buff, never offensive
