"""Inference-time rollout search (#40)."""
from __future__ import annotations

import copy

import pytest
import torch

from src.agent.search import (
    RolloutSearch,
    SearchBot,
    SearchConfig,
    _tower_differential,
    candidate_actions,
    leaf_value,
)
from src.simulator.constants import MatchResult, Side
from tests.conftest import make_engine

SMALL = {"tier": "full", "conv_channels": (8,), "cnn_out": 32, "fusion_mlp": 64}
FAST = SearchConfig(top_k_cards=2, cells_per_card=1, max_candidates=2,
                    rollouts_per_candidate=1, horizon_seconds=1.0)


@pytest.fixture()
def net(cards):
    from src.agent.network import make_network
    torch.manual_seed(0)
    return make_network(len(cards), SMALL)


@pytest.fixture()
def card_names(cards):
    return list(cards)


# ---------------------------------------------------------------- scoring


def test_tower_differential_favours_the_healthier_side(cards, arena):
    engine = make_engine(cards, arena)
    assert _tower_differential(engine, Side.BOTTOM) == pytest.approx(0.0)
    for t in engine.towers:
        if t.side == Side.TOP and t.kind == "princess_left":
            t.hp = 1.0
    assert _tower_differential(engine, Side.BOTTOM) > 0
    assert _tower_differential(engine, Side.TOP) < 0


def test_crowns_dominate_the_differential(cards, arena):
    engine = make_engine(cards, arena)
    for t in engine.towers:
        if t.side == Side.TOP and t.kind == "princess_left":
            t.hp = 0.0
    assert _tower_differential(engine, Side.BOTTOM) > 1.0


def test_leaf_value_short_circuits_on_a_finished_match(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    engine.result = MatchResult.BOTTOM_WIN
    ids = {n: i for i, n in enumerate(card_names)}
    assert leaf_value(net, engine, Side.BOTTOM, ids) == 1.0
    assert leaf_value(net, engine, Side.TOP, ids) == -1.0
    engine.result = MatchResult.DRAW
    assert leaf_value(net, engine, Side.BOTTOM, ids) == 0.0


# ------------------------------------------------------------- candidates


def test_candidates_are_legal_and_bounded(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    ids = {n: i for i, n in enumerate(card_names)}
    from src.agent.masking import build_action_masks

    masks = build_action_masks(engine, Side.BOTTOM)
    actions = candidate_actions(net, engine, Side.BOTTOM, ids, FAST)
    assert 0 < len(actions) <= FAST.max_candidates
    for choice, cell in actions:
        if choice == 0:
            continue
        assert masks["card"][choice - 1]
        assert masks["place"][choice - 1, cell]


def test_candidate_search_ignores_the_raw_action_product(net, card_names, cards, arena):
    """The whole point: 5 x 144 joint actions are never enumerated."""
    engine = make_engine(cards, arena)
    ids = {n: i for i, n in enumerate(card_names)}
    wide = SearchConfig(top_k_cards=5, cells_per_card=4, max_candidates=20)
    assert len(candidate_actions(net, engine, Side.BOTTOM, ids, wide)) <= 20


# ---------------------------------------------------------------- rollouts


def test_search_returns_one_of_its_candidates(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    search = RolloutSearch(net, card_names, FAST)
    chosen = search.choose(engine, Side.BOTTOM)
    assert chosen in candidate_actions(net, engine, Side.BOTTOM, search.card_to_id, FAST)


def test_search_does_not_mutate_the_root_engine(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    before = copy.deepcopy(engine)
    RolloutSearch(net, card_names, FAST).choose(engine, Side.BOTTOM)
    assert engine.time == before.time
    assert len(engine.units) == len(before.units)
    assert engine.players[Side.BOTTOM].elixir == before.players[Side.BOTTOM].elixir
    assert [t.hp for t in engine.towers] == [t.hp for t in before.towers]


def test_every_candidate_gets_a_score(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    search = RolloutSearch(net, card_names, FAST)
    scores = search.evaluate_candidates(engine, Side.BOTTOM)
    assert scores
    assert all(isinstance(v, float) for v in scores.values())


def test_time_budget_stops_early(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    budgeted = SearchConfig(top_k_cards=5, cells_per_card=3, max_candidates=15,
                            rollouts_per_candidate=3, horizon_seconds=3.0,
                            time_budget_s=0.0)
    scores = RolloutSearch(net, card_names, budgeted).evaluate_candidates(
        engine, Side.BOTTOM)
    assert len(scores) == 1, "budget exhausted after the first candidate"


def test_search_still_answers_when_only_noop_is_legal(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    engine.players[Side.BOTTOM].elixir = 0.0
    choice, _ = RolloutSearch(net, card_names, FAST).choose(engine, Side.BOTTOM)
    assert choice == 0


# ------------------------------------------------------------------- bot


def test_search_bot_plugs_into_the_bot_interface(net, card_names, cards, arena):
    engine = make_engine(cards, arena)
    bot = SearchBot(net, card_names, FAST)
    action = bot.decide(engine, Side.BOTTOM)
    if action is not None:
        assert 0 <= action.slot < 4
        assert engine.legal_deploy(Side.BOTTOM,
                                   engine.players[Side.BOTTOM].hand[action.slot],
                                   action.x, action.y)


def test_search_bot_opts_out_of_batched_opponent_resolution(net, card_names):
    """`SyncVecEnv` groups opponents by `batch_key` and runs the *plain*
    policy for the group. A searching bot must not be grouped, or its search
    is silently discarded."""
    bot = SearchBot(net, card_names, FAST)
    assert getattr(bot, "batch_key", None) is None
