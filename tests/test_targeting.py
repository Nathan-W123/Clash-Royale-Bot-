from src.simulator import targeting
from src.simulator.constants import Side
from src.simulator.entities import Tower
from tests.conftest import dummy_stats, make_engine, spawn_unit


def test_giant_ignores_troops_walks_to_tower(cards, arena):
    eng = make_engine(cards, arena)
    giant = spawn_unit(eng, cards["giant"], Side.BOTTOM, 14.5, 14.0)
    spawn_unit(eng, cards["knight"], Side.TOP, 14.5, 14.5)  # right next to it
    eng.tick()
    target = eng._by_id[giant.target_id]
    assert isinstance(target, Tower)


def test_ground_only_never_targets_air(cards, arena):
    eng = make_engine(cards, arena)
    knight = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 10.0)
    spawn_unit(eng, cards["minions"], Side.TOP, 9.0, 10.5)  # flying, adjacent
    eng.tick()
    target = eng._by_id[knight.target_id]
    assert isinstance(target, Tower)  # skipped the minion, pushes a tower


def test_nearest_acquire(cards, arena):
    eng = make_engine(cards, arena)
    musk = spawn_unit(eng, cards["musketeer"], Side.BOTTOM, 9.0, 10.0)
    far = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 15.0)
    near = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 12.0)
    eng.tick()
    assert musk.target_id == near.id
    assert far.hp == cards["knight"].hp


def test_lock_breaks_beyond_factor(cards, arena):
    eng = make_engine(cards, arena)
    musk = spawn_unit(eng, cards["musketeer"], Side.BOTTOM, 9.0, 10.0)
    prey = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 12.0)
    eng.tick()
    assert musk.target_id == prey.id
    # Still inside 1.5x sight: lock persists even beyond plain sight range.
    prey.y = 10.0 + musk.stats.sight_range * 1.3
    assert targeting.unit_keeps_lock(musk, prey)
    # Beyond 1.5x sight: lock breaks.
    prey.y = 10.0 + musk.stats.sight_range * 1.8
    assert not targeting.unit_keeps_lock(musk, prey)


def test_retarget_on_target_death(cards, arena):
    eng = make_engine(cards, arena)
    musk = spawn_unit(eng, cards["musketeer"], Side.BOTTOM, 9.0, 10.0)
    prey = spawn_unit(eng, cards["knight"], Side.TOP, 9.0, 12.0, hp=1.0)
    other = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 13.0)
    for _ in range(15):  # enough ticks for first hit (1.1s) to land and kill
        eng.tick()
    assert prey.hp <= 0
    assert musk.target_id == other.id


def test_tower_aggro_range(cards, arena):
    eng = make_engine(cards, arena)
    tower = next(t for t in eng.towers if t.side == Side.TOP and t.kind == "princess_left")
    intruder = spawn_unit(eng, dummy_stats(), Side.BOTTOM,
                          tower.x, tower.y - tower.stats.range - 3.0)
    eng.tick()
    assert tower.target_id is None
    intruder.y = tower.y - tower.stats.range  # edge distance now within range
    eng.tick()
    assert tower.target_id == intruder.id


def test_buildings_only_pulled_by_building_in_sight(cards, arena):
    eng = make_engine(cards, arena)
    giant = spawn_unit(eng, cards["giant"], Side.BOTTOM, 9.0, 20.0)
    cannon = spawn_unit(eng, cards["cannon"], Side.TOP, 9.0, 23.0)
    eng.tick()
    assert giant.target_id == cannon.id
