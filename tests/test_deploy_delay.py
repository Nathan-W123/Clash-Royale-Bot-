"""Post-deploy delay: freshly spawned troops/buildings sit inert before acting.

Duration is per-card (`CardStats.deploy_time`, set in cards.yaml), defaulting
to DEFAULT_DEPLOY_TIME — siege buildings and a few wind-up troops take
materially longer, which is a real tempo cost the agent has to plan around.
"""
import pytest

from src.simulator.constants import DEFAULT_DEPLOY_TIME as DEPLOY_DELAY
from src.simulator.constants import Side
from tests.conftest import dummy_stats, make_engine, spawn_unit


def test_deploy_delay_prevents_immediate_attack(cards, arena):
    eng = make_engine(cards, arena)
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 14.4)
    # Spawn via the real engine path so `deployed_at` is stamped at eng.time,
    # matching how a normally-deployed troop behaves.
    knight = eng.spawn_units(cards["knight"], Side.BOTTOM, 9.0, 13.5)[0]

    ticks_during_delay = int(DEPLOY_DELAY / arena.dt) - 2
    for _ in range(ticks_during_delay):
        eng.tick()
    assert bag.hp == bag.stats.hp  # knight hasn't been able to attack yet
    assert knight.x == pytest.approx(9.0) and knight.y == pytest.approx(13.5)  # hasn't moved

    for _ in range(50):
        eng.tick()
    assert bag.hp < bag.stats.hp  # active now, landed a hit


def test_deploy_delay_prevents_immediate_movement_toward_a_distant_target(cards, arena):
    eng = make_engine(cards, arena)
    far = spawn_unit(eng, dummy_stats("far", speed=0.0), Side.TOP, 9.0, 20.0)
    knight = eng.spawn_units(cards["knight"], Side.BOTTOM, 9.0, 10.0)[0]
    knight.target_id = far.id

    ticks_during_delay = int(DEPLOY_DELAY / arena.dt) - 2
    for _ in range(ticks_during_delay):
        eng.tick()
    assert (knight.x, knight.y) == (9.0, 10.0)

    for _ in range(20):
        eng.tick()
    assert knight.y > 10.0  # moving now that the delay has expired


def test_deploy_delay_applies_to_buildings(cards, arena):
    eng = make_engine(cards, arena)
    # y=13.5/14.4 is far enough from every tower's edge-distance range
    # (see test_combat.py's identical choice) that they don't confound this.
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 14.4)
    cannon = eng.spawn_units(cards["cannon"], Side.BOTTOM, 9.0, 13.5)[0]

    ticks_during_delay = int(DEPLOY_DELAY / arena.dt) - 2
    for _ in range(ticks_during_delay):
        eng.tick()
    assert bag.hp == bag.stats.hp  # cannon hasn't fired yet

    for _ in range(50):
        eng.tick()
    assert bag.hp < bag.stats.hp  # firing now


def test_deploy_delay_expires_exactly_around_one_second(cards, arena):
    eng = make_engine(cards, arena)
    knight = eng.spawn_units(cards["knight"], Side.BOTTOM, 9.0, 10.0)[0]
    assert eng._is_deploying(knight)
    for _ in range(int(DEPLOY_DELAY / arena.dt) + 1):
        eng.tick()
    assert not eng._is_deploying(knight)


# ------------------------------------------------------- per-card durations


def test_siege_buildings_take_longer_than_the_default(cards):
    """Mortar/X-Bow wind up slowly; that gap is the point of per-card timing."""
    assert cards["mortar"].deploy_time > DEPLOY_DELAY
    assert cards["x_bow"].deploy_time > DEPLOY_DELAY
    assert cards["knight"].deploy_time == DEPLOY_DELAY


def test_a_slow_card_is_still_inert_when_a_default_card_is_already_active(cards, arena):
    """The whole point: at the same instant, the fast card acts and the slow
    one cannot. A single global constant could not express this."""
    eng = make_engine(cards, arena)
    knight = eng.spawn_units(cards["knight"], Side.BOTTOM, 8.0, 10.0)[0]
    mortar = eng.spawn_units(cards["mortar"], Side.BOTTOM, 10.0, 10.0)[0]

    # Step just past the default deploy time but well short of the mortar's.
    for _ in range(int(DEPLOY_DELAY / arena.dt) + 2):
        eng.tick()
    assert not eng._is_deploying(knight)
    assert eng._is_deploying(mortar)

    for _ in range(int(cards["mortar"].deploy_time / arena.dt) + 2):
        eng.tick()
    assert not eng._is_deploying(mortar)


def test_slow_building_cannot_fire_during_its_wind_up(cards, arena):
    """Behavioral check, not just the flag: a mortar in range of a target
    deals no damage until its longer deploy_time has elapsed."""
    eng = make_engine(cards, arena)
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 14.4)
    eng.spawn_units(cards["mortar"], Side.BOTTOM, 9.0, 13.5)

    for _ in range(int(DEPLOY_DELAY / arena.dt) + 2):
        eng.tick()
    assert bag.hp == bag.stats.hp  # past the default delay, still winding up

    for _ in range(int(cards["mortar"].deploy_time / arena.dt) + 60):
        eng.tick()
    assert bag.hp < bag.stats.hp  # firing once wound up


def test_evolution_inherits_base_card_deploy_time(cards):
    if "mortar_evo" in cards:
        assert cards["mortar_evo"].deploy_time == cards["mortar"].deploy_time
