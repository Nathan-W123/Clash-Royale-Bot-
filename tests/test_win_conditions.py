import pytest

from src.simulator.constants import MatchResult, Side
from src.simulator.entities import PendingSpell
from tests.conftest import make_engine


def princess(eng, side, kind="princess_left"):
    return next(t for t in eng.towers if t.side == side and t.kind == kind)


def king(eng, side):
    return next(t for t in eng.towers if t.side == side and t.is_king)


def drop_spell(eng, side, x, y, damage=1000.0):
    eng.spells.append(PendingSpell(side=side, x=x, y=y, radius=1.0, damage=damage,
                                   tower_multiplier=1.0, resolve_at=eng.time,
                                   card_name="test"))


def test_king_death_instant_win(cards, arena):
    eng = make_engine(cards, arena)
    k = king(eng, Side.TOP)
    k.hp = 1.0
    drop_spell(eng, Side.BOTTOM, k.x, k.y)
    eng.tick()
    assert eng.result == MatchResult.BOTTOM_WIN


def test_simultaneous_king_destruction_is_draw(cards, arena):
    eng = make_engine(cards, arena)
    king(eng, Side.TOP).hp = 1.0
    king(eng, Side.BOTTOM).hp = 1.0
    kt, kb = king(eng, Side.TOP), king(eng, Side.BOTTOM)
    drop_spell(eng, Side.BOTTOM, kt.x, kt.y)
    drop_spell(eng, Side.TOP, kb.x, kb.y)
    eng.tick()
    assert eng.result == MatchResult.DRAW


def test_more_towers_at_regulation_wins(cards, arena):
    eng = make_engine(cards, arena)
    princess(eng, Side.TOP).hp = 1.0
    drop_spell(eng, Side.BOTTOM, princess(eng, Side.TOP).x, princess(eng, Side.TOP).y)
    eng.tick()
    assert eng.result == MatchResult.ONGOING
    eng.time = eng.regulation - arena.dt
    eng.tick()
    assert eng.result == MatchResult.BOTTOM_WIN


def test_tie_at_regulation_goes_to_overtime(cards, arena):
    eng = make_engine(cards, arena)
    eng.time = eng.regulation - arena.dt
    eng.tick()
    assert eng.result == MatchResult.ONGOING
    assert eng.overtime
    assert eng.double_elixir


def test_overtime_first_tower_fall_wins(cards, arena):
    eng = make_engine(cards, arena)
    eng.time = eng.regulation + 5.0
    p = princess(eng, Side.BOTTOM)
    p.hp = 1.0
    drop_spell(eng, Side.TOP, p.x, p.y)
    eng.tick()
    assert eng.result == MatchResult.TOP_WIN


def test_overtime_simultaneous_falls_cancel(cards, arena):
    eng = make_engine(cards, arena)
    eng.time = eng.regulation + 5.0
    pb, pt = princess(eng, Side.BOTTOM), princess(eng, Side.TOP)
    pb.hp = 1.0
    pt.hp = 1.0
    drop_spell(eng, Side.TOP, pb.x, pb.y)
    drop_spell(eng, Side.BOTTOM, pt.x, pt.y)
    eng.tick()
    assert eng.result == MatchResult.ONGOING  # 1-1, sudden death continues


def test_overtime_expiry_tiebreaker_lowest_tower_hp(cards, arena):
    eng = make_engine(cards, arena)
    princess(eng, Side.TOP).hp = 100.0  # TOP has the weakest tower
    eng.time = eng.regulation + arena.overtime - arena.dt
    eng.tick()
    assert eng.result == MatchResult.BOTTOM_WIN


def test_overtime_expiry_equal_state_is_draw(cards, arena):
    eng = make_engine(cards, arena)
    eng.time = eng.regulation + arena.overtime - arena.dt
    eng.tick()
    assert eng.result == MatchResult.DRAW


def test_king_activates_on_princess_fall(cards, arena):
    eng = make_engine(cards, arena)
    assert not king(eng, Side.TOP).activated
    p = princess(eng, Side.TOP)
    p.hp = 1.0
    drop_spell(eng, Side.BOTTOM, p.x, p.y)
    eng.tick()
    assert king(eng, Side.TOP).activated
    assert not king(eng, Side.BOTTOM).activated


def test_king_activates_on_chip_damage(cards, arena):
    eng = make_engine(cards, arena)
    k = king(eng, Side.TOP)
    drop_spell(eng, Side.BOTTOM, k.x, k.y, damage=10.0)
    eng.tick()
    assert k.activated


def test_single_lane_mode_has_two_towers_per_side(cards, arena):
    eng = make_engine(cards, arena, lanes="right")
    assert len(eng.towers_of(Side.BOTTOM)) == 2
    assert len(eng.towers_of(Side.TOP)) == 2
