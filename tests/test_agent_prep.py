"""Tests for agent prep modules (rewards, masking, observations)."""
from __future__ import annotations

import numpy as np
import pytest

from src.agent import (
    CARD_FEATURE_DIM,
    RewardShaper,
    SPATIAL_CHANNELS,
    action_mask_cards,
    action_mask_placement,
    anneal_factor,
    compute_step_reward,
    encode_hand,
    encode_spatial,
    load_reward_config,
)
from src.agent.masking import N_CELLS
from src.simulator.constants import PLACE_COLS, PLACE_ROWS, MatchResult, Side
from tests.conftest import force_hand, make_engine


def test_reward_from_mock_events():
    cfg = load_reward_config()
    # Enemy princess takes 700 damage -> +0.15 at weight 0.3 / 1400 scale
    events = [
        {"type": "tower_damage", "side": Side.TOP, "amount": 700.0},
        {"type": "death", "side": Side.TOP, "value": 3.0},
        {"type": "leak", "side": Side.BOTTOM, "amount": 0.5},
    ]
    r = compute_step_reward(events, Side.BOTTOM, None, None, cfg)
    assert r == pytest.approx(0.3 * 700 / 1400 + 0.02 * 3 - 0.005 * 0.5)

    shaper = RewardShaper(cfg)
    assert shaper.terminal_reward(MatchResult.BOTTOM_WIN, Side.BOTTOM) == 1.0
    assert shaper.terminal_reward(MatchResult.TOP_WIN, Side.BOTTOM) == -1.0
    assert shaper.terminal_reward(MatchResult.DRAW, Side.BOTTOM) == 0.0


def test_anneal_factor_endpoints():
    cfg = load_reward_config()
    assert anneal_factor(0, cfg) == cfg.anneal_start
    assert anneal_factor(cfg.anneal_over_steps, cfg) == cfg.anneal_end
    mid = anneal_factor(cfg.anneal_over_steps // 2, cfg)
    assert cfg.anneal_end < mid < cfg.anneal_start


def test_mask_unaffordable_card(cards, arena):
    engine = make_engine(cards, arena)
    player = engine.players[Side.BOTTOM]
    force_hand(player, cards, ["giant", "knight", "archers", "goblins"])
    player.elixir = 2.0  # giant costs 5
    mask = action_mask_cards(player)
    assert not mask[0]
    assert mask[1:].any()
    assert not action_mask_placement(engine, Side.BOTTOM, 0).any()


def test_spell_mask_full_grid(cards, arena):
    engine = make_engine(cards, arena)
    player = engine.players[Side.BOTTOM]
    force_hand(player, cards, ["knight", "cannon", "fireball", "giant"])
    player.elixir = 10.0
    slot = next(i for i, c in enumerate(player.hand) if c.name == "fireball")
    mask = action_mask_placement(engine, Side.BOTTOM, slot)
    assert mask.shape == (N_CELLS,)
    assert mask.all()


def test_obs_shapes(cards, arena):
    engine = make_engine(cards, arena)
    card_to_id = {name: i + 1 for i, name in enumerate(cards)}
    spatial = encode_spatial(engine, Side.BOTTOM)
    assert spatial.shape == (SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)
    assert spatial.dtype == np.float32

    hand = encode_hand(engine.players[Side.BOTTOM], card_to_id)
    assert hand.shape == (CARD_FEATURE_DIM,)
    assert hand.dtype == np.float32
    assert hand[-1] == pytest.approx(engine.players[Side.BOTTOM].elixir / 10.0)
