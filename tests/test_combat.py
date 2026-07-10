import math

import pytest

from src.simulator import combat
from src.simulator.constants import Side
from src.simulator.entities import PendingSpell
from tests.conftest import dummy_stats, make_engine, slot_of, spawn_unit


def test_dps_matches_hit_speed_with_initial_cooldown(cards, arena):
    eng = make_engine(cards, arena)
    # Mid-arena spot chosen to be outside every tower's range.
    knight = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 13.5)
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 14.4)  # in melee reach
    seconds = 22.0
    for _ in range(int(seconds / arena.dt)):
        eng.tick()
    expected_hits = math.floor(seconds / knight.stats.hit_speed)
    assert bag.hp == pytest.approx(1_000_000.0 - expected_hits * knight.stats.damage)


def test_splash_hits_radius_not_outside(cards, arena):
    eng = make_engine(cards, arena)
    valk = spawn_unit(eng, cards["valkyrie"], Side.BOTTOM, 9.0, 10.0)
    primary = spawn_unit(eng, dummy_stats("bag1"), Side.TOP, 9.0, 11.0)
    close = spawn_unit(eng, dummy_stats("bag2"), Side.TOP, 10.0, 11.5)   # within 2.0
    far = spawn_unit(eng, dummy_stats("bag3"), Side.TOP, 9.0, 16.0)      # outside
    events = []
    combat.apply_attack(valk, primary, eng.units_of(Side.TOP), events)
    dmg = valk.stats.damage
    assert primary.hp == 1_000_000.0 - dmg
    assert close.hp == 1_000_000.0 - dmg
    assert far.hp == 1_000_000.0


def test_ground_splash_does_not_hit_air(cards, arena):
    eng = make_engine(cards, arena)
    valk = spawn_unit(eng, cards["valkyrie"], Side.BOTTOM, 9.0, 10.0)
    primary = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 11.0)
    flyer = spawn_unit(eng, cards["minions"], Side.TOP, 9.5, 11.0)
    events = []
    combat.apply_attack(valk, primary, eng.units_of(Side.TOP), events)
    assert flyer.hp == cards["minions"].hp


def test_spell_delay_and_tower_multiplier(cards, arena):
    fb = cards["fireball"]
    eng = make_engine(cards, arena, deck=[fb] * 8)
    tower = next(t for t in eng.towers if t.side == Side.TOP and t.kind == "princess_left")
    eng.play_card(Side.BOTTOM, 0, tower.x, tower.y)
    ticks_before = int(fb.spell_delay / arena.dt) - 1
    for _ in range(ticks_before):
        eng.tick()
    assert tower.hp == tower.stats.hp  # not resolved yet
    eng.tick()
    eng.tick()
    assert tower.hp == pytest.approx(tower.stats.hp - fb.spell_damage * fb.tower_multiplier)


def test_spell_kills_units_full_damage(cards, arena):
    eng = make_engine(cards, arena)
    gob = spawn_unit(eng, cards["goblins"], Side.TOP, 9.0, 20.0)
    events = []
    spell = PendingSpell(side=Side.BOTTOM, x=9.0, y=20.0, radius=2.5, damage=325.0,
                         tower_multiplier=0.35, resolve_at=0.0, card_name="fireball")
    combat.resolve_spell(spell, eng.units_of(Side.TOP), eng.towers_of(Side.TOP), events)
    assert gob.hp <= 0
    deaths = [e for e in events if e["type"] == "death"]
    assert len(deaths) == 1
    assert deaths[0]["value"] == pytest.approx(cards["goblins"].cost / cards["goblins"].count)


def test_death_event_elixir_ledger(cards, arena):
    eng = make_engine(cards, arena)
    knight = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 20.0)
    events = []
    combat.damage_unit(knight, 10_000.0, events)
    assert events == [{"type": "death", "side": Side.TOP, "value": pytest.approx(3.0)}]


def test_building_expires_without_death_credit(cards, arena):
    eng = make_engine(cards, arena)
    cannon = spawn_unit(eng, cards["cannon"], Side.BOTTOM, 9.0, 10.0)
    cannon.expires_at = 1.0
    all_events = []
    for _ in range(15):
        all_events += eng.tick()
    assert cannon not in eng.units
    assert not any(e["type"] == "death" for e in all_events)
