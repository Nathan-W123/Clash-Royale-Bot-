"""Batched opponent inference must be a pure optimization.

SyncVecEnv groups envs whose opponents share a policy network and evaluates
them in one forward pass instead of N batch-of-1 calls. That is worth ~7x on
the forward pass, but only if it produces the same decisions the per-env path
would have.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agent.network import make_network
from src.agent.selfplay import BotOpponent, PolicyBot, policy_actions_batched
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.constants import Side
from src.simulator.engine import BattleEngine
from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv


@pytest.fixture(scope="module")
def bits():
    cards = load_cards()
    arena = load_arena()
    catalog = DeckCatalog()
    names = list(cards.keys())
    net = make_network(len(names), {"use_spatial": False})
    return cards, arena, catalog, names, net


def _engines(cards, arena, catalog, n):
    deck = catalog.resolve("training_mirror")
    out = []
    for i in range(n):
        eng = BattleEngine(deck, list(deck), arena, seed=i)
        for _ in range(20 + i * 5):  # diverge the states
            eng.tick()
        out.append(eng)
    return out


def test_batched_matches_per_engine_decisions(bits):
    """Deterministic policy: batched rows must equal one-at-a-time results."""
    cards, arena, catalog, names, net = bits
    bot = PolicyBot(net, names, name="t", deterministic=True)
    engines = _engines(cards, arena, catalog, 6)

    rows = policy_actions_batched(net, engines, Side.TOP, bot.card_to_id,
                                  deterministic=True)
    batched = [bot.decode_row(e, Side.TOP, r) for e, r in zip(engines, rows)]
    individual = [bot.decide(e, Side.TOP) for e in engines]

    assert len(batched) == len(individual) == 6
    for b, i in zip(batched, individual):
        assert (b is None) == (i is None)
        if b is not None:
            assert (b.slot, b.x, b.y) == (i.slot, i.x, i.y)


def test_batched_handles_empty_group(bits):
    _, _, _, names, net = bits
    out = policy_actions_batched(net, [], Side.TOP, {n: i for i, n in enumerate(names)})
    assert out.shape == (0, 2)


def test_vec_env_step_matches_unbatched_path(bits):
    """End-to-end: a vec-env using the batched path must reach the same
    engine state as one that resolves opponents per-env."""
    cards, arena, catalog, names, net = bits
    deck = catalog.resolve("training_mirror")

    def make(batched: bool):
        bot = PolicyBot(net, names, name="opp", deterministic=True)
        if not batched:
            # Hide batch_key so SyncVecEnv falls back to per-env resolution.
            bot.batch_key = None
        envs = SyncVecEnv([
            (lambda b=bot: CRBattleEnv(cards, arena, deck, list(deck),
                                       opponent=b, use_spatial=False))
            for _ in range(4)
        ])
        envs.reset(seed=123)
        return envs

    a = make(batched=True)
    b = make(batched=False)
    rng = np.random.default_rng(0)
    for _ in range(25):
        actions = np.stack([
            np.array([rng.integers(0, 5), rng.integers(0, 144)]) for _ in range(4)
        ])
        a.step(actions)
        b.step(actions)

    for ea, eb in zip(a.envs, b.envs):
        assert ea.engine.time == pytest.approx(eb.engine.time)
        assert len(ea.engine.units) == len(eb.engine.units)
        for ua, ub in zip(ea.engine.units, eb.engine.units):
            assert ua.stats.name == ub.stats.name
            assert (ua.x, ua.y) == pytest.approx((ub.x, ub.y))
            assert ua.hp == pytest.approx(ub.hp)


def test_scripted_opponents_still_resolved_per_env(bits):
    """Scripted bots have no batch_key and must keep working untouched."""
    cards, arena, catalog, names, net = bits
    deck = catalog.resolve("training_mirror")
    bot = BotOpponent(get_bot("rusher", catalog=catalog, deck_name="training_mirror",
                              rng=np.random.default_rng(0)))
    envs = SyncVecEnv([
        (lambda: CRBattleEnv(cards, arena, deck, list(deck),
                             opponent=bot, use_spatial=False))
        for _ in range(3)
    ])
    envs.reset(seed=7)
    resolved = envs._batched_opponent_actions()
    from src.simulator.env import UNSET
    assert all(r is UNSET for r in resolved)  # left for the env to handle

    obs, r, d, m, infos = envs.step(np.zeros((3, 2), dtype=np.int64))
    assert len(infos) == 3  # and the step still completes


def test_mixed_policy_and_scripted_opponents(bits):
    """A vec-env with both kinds must batch only the policy ones."""
    cards, arena, catalog, names, net = bits
    deck = catalog.resolve("training_mirror")
    policy = PolicyBot(net, names, name="opp", deterministic=True)
    scripted = BotOpponent(get_bot("rusher", catalog=catalog,
                                   deck_name="training_mirror",
                                   rng=np.random.default_rng(0)))
    opponents = [policy, scripted, policy, scripted]
    envs = SyncVecEnv([
        (lambda o=o: CRBattleEnv(cards, arena, deck, list(deck),
                                 opponent=o, use_spatial=False))
        for o in opponents
    ])
    envs.reset(seed=11)
    from src.simulator.env import UNSET
    resolved = envs._batched_opponent_actions()
    assert resolved[1] is UNSET and resolved[3] is UNSET  # scripted deferred
    assert resolved[0] is not UNSET and resolved[2] is not UNSET  # policy batched
