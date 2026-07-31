"""Asymmetric actor-critic (#26): privileged value, human-legal policy.

The critic may read simulator ground truth the deployed actor will never
have. The load-bearing property is that this stays a *training-time* aid: the
policy's output must depend only on its own tier's observation, so the only
path from privileged state to behaviour is the advantage signal.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent import obs_layout
from src.agent.network import (
    critic_view,
    make_network,
    masks_to_tensors,
    obs_to_tensors,
)
from src.agent.selfplay import PolicyBot, load_checkpoint, save_checkpoint
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.constants import Side
from src.simulator.env import CRBattleEnv


@pytest.fixture(scope="module")
def world():
    cards = load_cards()
    arena = load_arena()
    deck = DeckCatalog().resolve("training_mirror")
    return cards, arena, deck


def _env(world, **kw):
    cards, arena, deck = world
    return CRBattleEnv(cards, arena, deck, list(deck), **kw)


def _net(cards, **kw):
    return make_network(len(cards), kw)


# --------------------------------------------------------------- env plumbing


def test_env_emits_critic_observation(world):
    cards, _, _ = world
    env = _env(world, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    obs, _ = env.reset(seed=0)
    assert "critic_spatial" in obs and "critic_vector" in obs and "critic_cards" in obs
    assert obs["vector"].shape[0] == obs_layout.RESTRICTED_SCALAR_DIM
    assert obs["critic_vector"].shape[0] == obs_layout.SCALAR_DIM


def test_no_critic_keys_when_not_configured(world):
    env = _env(world, tier=obs_layout.TIER_RESTRICTED)
    obs, _ = env.reset(seed=0)
    assert not any(k.startswith("critic_") for k in obs)


def test_actor_observation_is_unchanged_by_enabling_the_critic(world):
    """Turning the critic on must not perturb what the actor sees."""
    plain = _env(world, tier=obs_layout.TIER_RESTRICTED)
    asym = _env(world, tier=obs_layout.TIER_RESTRICTED,
                critic_tier=obs_layout.TIER_FULL)
    a, _ = plain.reset(seed=7)
    b, _ = asym.reset(seed=7)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"actor obs '{k}' changed"


def test_critic_sees_opponent_elixir_and_actor_does_not(world):
    """The whole point, stated as a test: perturb only the opponent's elixir
    and check exactly one of the two observations moves."""
    env = _env(world, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    before, _ = env.reset(seed=3)
    before = {k: v.copy() for k, v in before.items()}

    env.engine.players[Side.TOP].elixir = 1.0
    low = env.build_obs(Side.BOTTOM)
    env.engine.players[Side.TOP].elixir = 9.0
    high = env.build_obs(Side.BOTTOM)

    assert np.array_equal(low["vector"], high["vector"]), \
        "actor vector moved with opponent elixir — that is a privileged leak"
    assert not np.array_equal(low["critic_vector"], high["critic_vector"]), \
        "critic vector ignored opponent elixir — privileged info is not reaching it"


# ------------------------------------------------------------------- network


def test_critic_view_rekeys_only_critic_entries():
    obs = {"spatial": 1, "vector": 2, "critic_spatial": 3, "critic_vector": 4}
    assert critic_view(obs) == {"spatial": 3, "vector": 4}


def test_asymmetric_network_builds_separate_stacks(world):
    cards, _, _ = world
    net = _net(cards, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    assert net.config.asymmetric
    assert hasattr(net, "critic_fusion") and hasattr(net, "critic_card_embed")
    # Separate embeddings keep the information boundary auditable.
    assert net.critic_card_embed is not net.card_embed
    # Restricted actor has no CNN; full critic does.
    assert not hasattr(net, "cnn")
    assert hasattr(net, "critic_cnn")


def test_policy_output_ignores_critic_observation(world):
    """Strongest form: scramble the critic observation and confirm the sampled
    action distribution is bit-identical. Anything else means the privileged
    tensor reached the policy heads."""
    cards, _, _ = world
    env = _env(world, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    obs, info = env.reset(seed=11)
    net = _net(cards, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    net.eval()
    device = torch.device("cpu")
    masks = masks_to_tensors(info["masks"], device)

    t_clean = obs_to_tensors(obs, device)
    scrambled = {k: (v.copy() if not k.startswith("critic_") else
                     np.zeros_like(v) if k != "critic_cards" else v)
                 for k, v in obs.items()}
    t_dirty = obs_to_tensors(scrambled, device)

    with torch.no_grad():
        feat_a, feat_b = net.trunk(t_clean), net.trunk(t_dirty)
        logits_a = net.card_logits(feat_a, masks["card"])
        logits_b = net.card_logits(feat_b, masks["card"])
    assert torch.equal(logits_a, logits_b)


def test_value_actually_uses_the_critic_trunk(world):
    """Converse of the above: the value head *must* respond to critic input,
    otherwise the privileged trunk is dead weight."""
    cards, _, _ = world
    env = _env(world, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    obs, _ = env.reset(seed=13)
    net = _net(cards, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    net.eval()
    device = torch.device("cpu")

    t_clean = obs_to_tensors(obs, device)
    bumped = {k: v.copy() for k, v in obs.items()}
    bumped["critic_vector"] = bumped["critic_vector"] + 0.5
    t_bumped = obs_to_tensors(bumped, device)

    with torch.no_grad():
        v_a = net.value_of(t_clean, net.trunk(t_clean))
        v_b = net.value_of(t_bumped, net.trunk(t_bumped))
    assert not torch.equal(v_a, v_b)


def test_value_falls_back_when_critic_obs_absent(world):
    """`PolicyBot` opponents call `act` for an action only and never build a
    critic observation; that must work rather than KeyError."""
    cards, _, _ = world
    env = _env(world, tier=obs_layout.TIER_RESTRICTED)  # no critic keys
    obs, info = env.reset(seed=5)
    net = _net(cards, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    device = torch.device("cpu")
    actions, log_probs, values = net.act(
        obs_to_tensors(obs, device), masks_to_tensors(info["masks"], device))
    assert torch.isfinite(values).all()
    assert actions.shape == (1, 2)


def test_policy_bot_still_drives_an_asymmetric_net(world):
    cards, arena, deck = world
    net = _net(cards, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    bot = PolicyBot(net, list(cards.keys()), deterministic=True)
    env = _env(world, tier=obs_layout.TIER_RESTRICTED)
    env.reset(seed=2)
    action = bot.decide(env.engine, Side.TOP)
    assert action is None or 0 <= action.slot < 4


# ---------------------------------------------------------- checkpointing


def test_symmetric_checkpoints_still_load(tmp_path, world):
    """Additive-only module tree: a net saved before #26 must load strictly."""
    cards, _, _ = world
    names = list(cards.keys())
    plain = _net(cards, tier=obs_layout.TIER_RESTRICTED)
    path = tmp_path / "plain.pt"
    save_checkpoint(plain, names, path)
    loaded, _ = load_checkpoint(path)
    assert not loaded.config.asymmetric


def test_asymmetric_checkpoint_round_trips(tmp_path, world):
    cards, _, _ = world
    names = list(cards.keys())
    net = _net(cards, tier=obs_layout.TIER_RESTRICTED, critic_tier=obs_layout.TIER_FULL)
    path = tmp_path / "asym.pt"
    save_checkpoint(net, names, path)
    loaded, _ = load_checkpoint(path)
    assert loaded.config.asymmetric
    assert loaded.config.critic_tier == obs_layout.TIER_FULL
    assert loaded.config.tier == obs_layout.TIER_RESTRICTED
