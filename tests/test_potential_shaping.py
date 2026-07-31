"""Potential-based shaping (#28).

The claim that justifies leaving this on permanently is *policy invariance*
(Ng, Harada & Russell 1999): shaped return differs from true return by a
constant that depends only on the start and end states, never on the path.
These tests check that identity numerically rather than trusting the algebra.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agent import rewards as R
from src.agent.train import ShapedRewardFn
from src.simulator.constants import MatchResult, Side
from tests.conftest import make_engine


@pytest.fixture()
def cfg():
    return R.RewardConfig(mode=R.MODE_POTENTIAL, potential_scale=0.5, gamma=0.99)


# ------------------------------------------------------------------ potential


def test_potential_is_zero_at_a_symmetric_start(cards, arena):
    """Relied on directly: it lets a fresh engine default phi to 0.0 with no
    reset hook."""
    eng = make_engine(cards, arena)
    assert R.tower_potential(eng, Side.BOTTOM, R.RewardConfig()) == pytest.approx(0.0)
    assert R.tower_potential(eng, Side.TOP, R.RewardConfig()) == pytest.approx(0.0)


def test_potential_is_antisymmetric(cards, arena, cfg):
    eng = make_engine(cards, arena)
    next(t for t in eng.towers if t.side == Side.TOP).hp *= 0.25
    a = R.tower_potential(eng, Side.BOTTOM, cfg)
    b = R.tower_potential(eng, Side.TOP, cfg)
    assert a > 0 and b < 0
    assert a == pytest.approx(-b)


def test_potential_rises_as_the_enemy_loses_tower_hp(cards, arena, cfg):
    eng = make_engine(cards, arena)
    before = R.tower_potential(eng, Side.BOTTOM, cfg)
    for t in eng.towers:
        if t.side == Side.TOP:
            t.hp *= 0.5
    after = R.tower_potential(eng, Side.BOTTOM, cfg)
    assert after > before


def test_potential_is_bounded_by_scale(cards, arena, cfg):
    eng = make_engine(cards, arena)
    for t in eng.towers:
        if t.side == Side.TOP:
            t.hp = 0.0
    assert R.tower_potential(eng, Side.BOTTOM, cfg) == pytest.approx(cfg.potential_scale)


def test_destroyed_towers_do_not_produce_negative_hp(cards, arena, cfg):
    eng = make_engine(cards, arena)
    for t in eng.towers:
        t.hp = -50.0  # over-kill damage leaves hp below zero
    assert R.tower_potential(eng, Side.BOTTOM, cfg) == pytest.approx(0.0)


# ------------------------------------------------------------- the invariant


def test_shaped_return_telescopes_to_a_path_independent_constant(cfg):
    """The core guarantee. Two different trajectories between the same start
    and end potentials must accumulate the *same* discounted shaping."""

    def discounted_total(potentials: list[float]) -> float:
        total, disc = 0.0, 1.0
        for prev, curr in zip(potentials, potentials[1:]):
            total += disc * R.potential_shaping(prev, curr, cfg, terminated=False)
            disc *= cfg.gamma
        return total

    smooth = [0.0, 0.1, 0.2, 0.3, 0.4]
    jagged = [0.0, 0.4, -0.2, 0.35, 0.4]  # same endpoints, wild middle
    assert discounted_total(smooth) == pytest.approx(discounted_total(jagged), abs=1e-9)


def test_closed_form_matches_the_telescoped_sum(cfg):
    """Sum of gamma^t * F_t should equal gamma^T*phi(s_T) - phi(s_0)."""
    potentials = [0.0, 0.2, -0.1, 0.45, 0.3]
    total, disc = 0.0, 1.0
    for prev, curr in zip(potentials, potentials[1:]):
        total += disc * R.potential_shaping(prev, curr, cfg, terminated=False)
        disc *= cfg.gamma
    closed = disc * potentials[-1] - potentials[0]
    assert total == pytest.approx(closed, abs=1e-9)


def test_terminal_transition_uses_zero_successor_potential(cfg):
    """Absorbing states must have phi = 0, else a state-dependent bonus leaks
    into the terminal step and invariance is lost."""
    assert R.potential_shaping(0.3, 0.9, cfg, terminated=True) == pytest.approx(-0.3)
    assert R.potential_shaping(0.3, 0.9, cfg, terminated=False) == pytest.approx(
        cfg.gamma * 0.9 - 0.3)


def test_a_round_trip_in_potential_nets_out(cfg):
    """Going up then back down must not pay out — that is precisely the
    reward-farming loop non-potential shaping permits."""
    up = R.potential_shaping(0.0, 0.4, cfg, terminated=False)
    down = R.potential_shaping(0.4, 0.0, cfg, terminated=False)
    assert up + down == pytest.approx((cfg.gamma - 1.0) * 0.4, abs=1e-9)
    assert up + down <= 0.0  # never a net gain


# --------------------------------------------------------------- integration


def test_shaped_reward_fn_tracks_potential_per_side(cards, arena, cfg):
    eng = make_engine(cards, arena)
    fn = ShapedRewardFn(R.RewardShaper(config=cfg))
    r0 = fn([], Side.BOTTOM, eng)
    assert r0 == pytest.approx(0.0, abs=1e-9)  # symmetric start, nothing changed

    for t in eng.towers:
        if t.side == Side.TOP:
            t.hp *= 0.5
    r1 = fn([], Side.BOTTOM, eng)
    assert r1 > 0.0  # gained ground
    assert getattr(eng, f"_potential_{int(Side.BOTTOM)}") > 0.0


def test_both_seats_keep_independent_potential_state(cards, arena, cfg):
    eng = make_engine(cards, arena)
    fn = ShapedRewardFn(R.RewardShaper(config=cfg))
    fn([], Side.BOTTOM, eng)
    fn([], Side.TOP, eng)
    b = getattr(eng, f"_potential_{int(Side.BOTTOM)}")
    t = getattr(eng, f"_potential_{int(Side.TOP)}")
    assert b == pytest.approx(-t)


def test_events_mode_is_unchanged(cards, arena):
    """Legacy path must behave exactly as before, including annealing."""
    eng = make_engine(cards, arena)
    cfg = R.RewardConfig(mode=R.MODE_EVENTS, tower_damage=0.3)
    fn = ShapedRewardFn(R.RewardShaper(config=cfg))
    fn.weight = 1.0
    events = [{"type": "tower_damage", "side": Side.TOP, "amount": R.TOWER_HP_SCALE}]
    assert fn(events, Side.BOTTOM, eng) == pytest.approx(0.3)
    fn.weight = 0.0
    assert fn(events, Side.BOTTOM, eng) == pytest.approx(0.0)


def test_terminal_outcome_still_dominates(cards, arena, cfg):
    """Shaping must not out-vote actually winning."""
    eng = make_engine(cards, arena)
    fn = ShapedRewardFn(R.RewardShaper(config=cfg))
    for t in eng.towers:
        if t.side == Side.TOP:
            t.hp = 0.0
    eng.result = MatchResult.BOTTOM_WIN
    r = fn([], Side.BOTTOM, eng)
    assert r > 0.5  # terminal 1.0 dominates the <=0.5 shaping term


def test_unknown_mode_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("reward:\n  mode: nonsense\n")
    with pytest.raises(ValueError, match="unknown reward.mode"):
        R.load_reward_config(path)


def test_gamma_defaults_to_the_ppo_discount(tmp_path):
    """Invariance only holds when the two discounts agree, so the default
    must follow the ppo block rather than a hardcoded constant."""
    path = tmp_path / "cfg.yaml"
    path.write_text("ppo:\n  gamma: 0.912\nreward:\n  mode: potential\n")
    cfg = R.load_reward_config(path)
    assert cfg.gamma == pytest.approx(0.912)
