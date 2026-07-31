"""Tests for the three observation tiers (full / human / restricted)."""
from __future__ import annotations

import numpy as np
import pytest

from src.agent.obs_layout import (
    HUMAN_SCALAR_DIM,
    MAX_ENTITIES,
    RESTRICTED_SCALAR_DIM,
    SCALAR_DIM,
    SPATIAL_CHANNELS,
    TIER_FULL,
    TIER_HUMAN,
    TIER_RESTRICTED,
    UNIT_FEATURE_DIM,
    encode_obs,
    encode_scalars,
    encode_units,
    resolve_tier,
    scalar_dim_for,
    unit_views,
)
from src.simulator.constants import PLACE_COLS, PLACE_ROWS, Side
from tests.conftest import make_engine, spawn_unit


def _card_to_id(cards):
    return {name: i for i, name in enumerate(cards)}


# ------------------------------------------------------------- tier plumbing


def test_use_spatial_bool_still_resolves_to_a_tier():
    assert resolve_tier(True) == TIER_FULL
    assert resolve_tier(False) == TIER_RESTRICTED
    assert resolve_tier(None) == TIER_FULL
    assert resolve_tier("HUMAN") == TIER_HUMAN


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        resolve_tier("cheating")


def test_scalar_widths_are_not_ordered_by_tier():
    # Guards the trap called out in the handoff: restricted is *wider* than
    # full, so any "tiers are nested / ordered by width" assumption is wrong.
    assert scalar_dim_for(TIER_RESTRICTED) > scalar_dim_for(TIER_FULL)
    assert scalar_dim_for(TIER_HUMAN) == HUMAN_SCALAR_DIM
    assert (SCALAR_DIM, HUMAN_SCALAR_DIM, RESTRICTED_SCALAR_DIM) == (17, 20, 18)


# ------------------------------------------------------------------- shapes


def test_full_path_unchanged_shape(cards, arena):
    engine = make_engine(cards, arena)
    obs = encode_obs(engine, Side.BOTTOM, _card_to_id(cards))
    assert obs["spatial"].shape == (SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)
    assert obs["vector"].shape == (SCALAR_DIM,)
    assert obs["cards"].shape == (5,)


def test_human_path_keeps_real_spatial_grid(cards, arena):
    engine = make_engine(cards, arena)
    spawn_unit(engine, cards["knight"], Side.TOP, 9.0, 20.0)
    obs = encode_obs(engine, Side.BOTTOM, _card_to_id(cards), TIER_HUMAN)
    assert obs["spatial"].shape == (SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)
    assert obs["spatial"].any(), "human tier must see rendered enemy troops"
    assert obs["vector"].shape == (HUMAN_SCALAR_DIM,)

    full = encode_obs(engine, Side.BOTTOM, _card_to_id(cards), TIER_FULL)
    np.testing.assert_array_equal(obs["spatial"], full["spatial"])


def test_restricted_path_shapes_and_zero_spatial(cards, arena):
    engine = make_engine(cards, arena)
    obs = encode_obs(engine, Side.BOTTOM, _card_to_id(cards), use_spatial=False)
    assert obs["spatial"].shape == (SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)
    assert not obs["spatial"].any()
    assert obs["vector"].shape == (RESTRICTED_SCALAR_DIM,)
    assert obs["cards"].shape == (5,)


# --------------------------------------------------- the opponent-elixir line


def test_opponent_elixir_leaks_only_in_full_path(cards, arena):
    engine = make_engine(cards, arena)
    engine.players[Side.TOP].elixir = 7.25

    full = encode_scalars(engine, Side.BOTTOM, TIER_FULL)
    assert 7.25 / 10.0 in full

    for tier in (TIER_HUMAN, TIER_RESTRICTED):
        assert 7.25 / 10.0 not in encode_scalars(engine, Side.BOTTOM, tier)


@pytest.mark.parametrize("tier", [TIER_HUMAN, TIER_RESTRICTED])
def test_live_legal_tiers_are_invariant_to_opponent_elixir(cards, arena, tier):
    """The load-bearing test: two engines identical except for the opponent's
    elixir must encode *identically*. A width check alone would not catch a
    regression that swapped some other field into the freed slot."""
    low = make_engine(cards, arena, seed=7)
    high = make_engine(cards, arena, seed=7)
    low.players[Side.TOP].elixir = 0.0
    high.players[Side.TOP].elixir = 10.0

    a = encode_obs(low, Side.BOTTOM, _card_to_id(cards), tier)
    b = encode_obs(high, Side.BOTTOM, _card_to_id(cards), tier)
    for key in ("spatial", "cards", "vector"):
        np.testing.assert_array_equal(a[key], b[key])


def test_full_tier_does_react_to_opponent_elixir(cards, arena):
    """Counterpart to the invariance test — proves it is testing something."""
    low = make_engine(cards, arena, seed=7)
    high = make_engine(cards, arena, seed=7)
    low.players[Side.TOP].elixir = 0.0
    high.players[Side.TOP].elixir = 10.0
    assert not np.array_equal(
        encode_scalars(low, Side.BOTTOM, TIER_FULL),
        encode_scalars(high, Side.BOTTOM, TIER_FULL))


# ------------------------------------------------------------- tower scalars


def test_restricted_per_tower_alive_booleans(cards, arena):
    engine = make_engine(cards, arena)
    for t in engine.towers:
        if t.side == Side.BOTTOM and t.kind == "princess_left":
            t.hp = 0.0

    vector = encode_scalars(engine, Side.BOTTOM, TIER_RESTRICTED)
    # Layout: [elixir, cost x4, afford x4, time, double_elixir, overtime,
    #          own_king_act, enemy_king_act,
    #          own_left, own_right, enemy_left, enemy_right]
    own_left, own_right, enemy_left, enemy_right = vector[-4:]
    assert own_left == 0.0
    assert own_right == 1.0
    assert enemy_left == 1.0
    assert enemy_right == 1.0


def test_human_carries_both_princess_counts_and_alive_flags(cards, arena):
    engine = make_engine(cards, arena)
    for t in engine.towers:
        if t.side == Side.TOP and t.kind == "princess_right":
            t.hp = 0.0

    vector = encode_scalars(engine, Side.BOTTOM, TIER_HUMAN)
    own_left, own_right, enemy_left, enemy_right = vector[-4:]
    assert (own_left, own_right, enemy_left, enemy_right) == (1.0, 1.0, 1.0, 0.0)
    own_frac, enemy_frac = vector[-6:-4]
    assert (own_frac, enemy_frac) == (1.0, 0.5)


# -------------------------------------------------------------- entity views


def test_unit_views_cover_units_and_towers(cards, arena):
    engine = make_engine(cards, arena)
    spawn_unit(engine, cards["knight"], Side.BOTTOM, 4.0, 6.0)
    views = unit_views(engine, Side.BOTTOM)
    towers = [v for v in views if v.is_tower]
    troops = [v for v in views if not v.is_tower]
    assert len(towers) == len(engine.towers)
    assert [v.card for v in troops] == ["knight"]
    assert troops[0].friendly is True


def test_encode_units_pads_and_flags_presence(cards, arena):
    engine = make_engine(cards, arena)
    spawn_unit(engine, cards["knight"], Side.TOP, 9.0, 22.0)
    units = encode_units(engine, Side.BOTTOM, _card_to_id(cards))
    assert units.shape == (MAX_ENTITIES, UNIT_FEATURE_DIM)
    present = units[:, 1] > 0
    assert present.sum() == len(unit_views(engine, Side.BOTTOM))
    assert not units[present.sum():].any(), "padding rows must be all-zero"


def test_encode_units_is_mirrored_for_the_top_seat(cards, arena):
    """Both seats must see their own king at the same normalized y, or the
    policy silently learns a flipped board for one of them."""
    engine = make_engine(cards, arena)
    ids = _card_to_id(cards)
    bottom = encode_units(engine, Side.BOTTOM, ids)
    top = encode_units(engine, Side.TOP, ids)

    def own_king_y(mat):
        rows = [r for r in mat if r[1] > 0 and r[8] > 0 and r[6] > 0]
        return min(r[3] for r in rows)

    assert own_king_y(bottom) == pytest.approx(own_king_y(top))
