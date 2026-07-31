"""Permutation-invariant set encoder over the entity list (#39)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent import obs_layout
from src.agent.network import (
    N_CARD_CHOICES,
    N_CELLS,
    make_network,
    masks_to_tensors,
    obs_to_tensors,
)
from src.agent.obs_noise import ObservationNoise, ObsNoiseConfig
from src.simulator.constants import HAND_SIZE, PLACE_COLS, PLACE_ROWS, Side
from tests.conftest import make_engine, spawn_unit

N_CARDS = 16
DEVICE = torch.device("cpu")
SMALL = {"tier": "human", "use_set_encoder": True, "unit_hidden": 32,
         "cnn_out": 32, "fusion_mlp": 64}


def _batch(rng, batch=6, n_units=5):
    units = np.zeros((batch, obs_layout.MAX_ENTITIES, obs_layout.UNIT_FEATURE_DIM),
                     np.float32)
    units[:, :n_units, 0] = rng.integers(0, N_CARDS, (batch, n_units))
    units[:, :n_units, 1] = 1.0
    units[:, :n_units, 2:] = rng.random(
        (batch, n_units, obs_layout.UNIT_FEATURE_DIM - 2)).astype(np.float32)
    obs = {
        "spatial": rng.random(
            (batch, obs_layout.SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)).astype(np.float32),
        "cards": rng.integers(0, N_CARDS, (batch, HAND_SIZE + 1)),
        "vector": rng.random((batch, obs_layout.HUMAN_SCALAR_DIM)).astype(np.float32),
        "units": units,
    }
    card_mask = np.zeros((batch, N_CARD_CHOICES), bool)
    card_mask[:, 0] = True
    card_mask[:, 1] = True
    place_mask = rng.random((batch, HAND_SIZE, N_CELLS)) < 0.4
    place_mask[:, 0, 0] = True
    return obs, {"card": card_mask, "place": place_mask}


# ----------------------------------------------------------------- topology


def test_set_encoder_replaces_the_cnn():
    net = make_network(N_CARDS, SMALL)
    assert hasattr(net, "set_encoder")
    assert not hasattr(net, "cnn")


def test_grid_path_is_still_available_for_ablation():
    net = make_network(N_CARDS, {"tier": "human"})
    assert hasattr(net, "cnn")
    assert not hasattr(net, "set_encoder")


def test_set_encoder_shares_the_hand_card_embedding():
    """A card on the arena and the same card in hand should be the same
    vector, not two independently-learned ones."""
    net = make_network(N_CARDS, SMALL)
    assert net.set_encoder.card_embed is net.card_embed


def test_missing_units_key_fails_loudly():
    rng = np.random.default_rng(0)
    net = make_network(N_CARDS, SMALL)
    obs, _ = _batch(rng)
    del obs["units"]
    with pytest.raises(KeyError, match="with_units"):
        net.trunk(obs_to_tensors(obs, DEVICE))


# ------------------------------------------------------------- invariance


def test_output_is_permutation_invariant():
    rng = np.random.default_rng(1)
    net = make_network(N_CARDS, SMALL)
    obs, _ = _batch(rng, batch=1, n_units=6)
    a = net.trunk(obs_to_tensors(obs, DEVICE))

    shuffled = obs["units"].copy()
    order = rng.permutation(6)
    shuffled[0, :6] = shuffled[0, order]
    obs["units"] = shuffled
    b = net.trunk(obs_to_tensors(obs, DEVICE))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_padding_rows_do_not_affect_the_output():
    rng = np.random.default_rng(2)
    net = make_network(N_CARDS, SMALL)
    obs, _ = _batch(rng, batch=1, n_units=4)
    a = net.trunk(obs_to_tensors(obs, DEVICE))

    # Garbage in the padded rows, presence flag still zero.
    noisy = obs["units"].copy()
    noisy[0, 4:, 0] = rng.integers(0, N_CARDS, obs_layout.MAX_ENTITIES - 4)
    noisy[0, 4:, 2:] = rng.random((obs_layout.MAX_ENTITIES - 4,
                                   obs_layout.UNIT_FEATURE_DIM - 2))
    obs["units"] = noisy
    b = net.trunk(obs_to_tensors(obs, DEVICE))
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_empty_entity_list_is_finite():
    """An all-absent row must pool to zero, not to -inf from the max branch."""
    net = make_network(N_CARDS, SMALL)
    rng = np.random.default_rng(3)
    obs, _ = _batch(rng, batch=2, n_units=0)
    feat = net.trunk(obs_to_tensors(obs, DEVICE))
    assert torch.isfinite(feat).all()


def test_content_actually_changes_the_output():
    rng = np.random.default_rng(4)
    net = make_network(N_CARDS, SMALL)
    obs, _ = _batch(rng, batch=1, n_units=3)
    a = net.trunk(obs_to_tensors(obs, DEVICE))
    obs["units"][0, 0, 2] += 0.5   # move one unit
    b = net.trunk(obs_to_tensors(obs, DEVICE))
    assert not torch.allclose(a, b)


def test_set_encoder_ignores_the_rasterized_grid():
    rng = np.random.default_rng(5)
    net = make_network(N_CARDS, SMALL)
    obs, _ = _batch(rng)
    a = net.trunk(obs_to_tensors(obs, DEVICE))
    obs["spatial"] = rng.random(obs["spatial"].shape).astype(np.float32) * 50
    torch.testing.assert_close(a, net.trunk(obs_to_tensors(obs, DEVICE)))


# ------------------------------------------------------------- action path


def test_act_and_evaluate_still_respect_masks():
    rng = np.random.default_rng(6)
    net = make_network(N_CARDS, SMALL)
    obs_np, masks_np = _batch(rng, batch=8)
    obs = obs_to_tensors(obs_np, DEVICE)
    masks = masks_to_tensors(masks_np, DEVICE)
    actions, log_prob, value = net.act(obs, masks)
    for b in range(8):
        card, cell = int(actions[b, 0]), int(actions[b, 1])
        assert masks_np["card"][b, card]
        if card > 0:
            assert masks_np["place"][b, card - 1, cell]
    log_prob2, entropy, _ = net.evaluate_actions(obs, masks, actions)
    torch.testing.assert_close(log_prob, log_prob2, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(entropy).all()


def test_single_observation_is_batched_consistently():
    rng = np.random.default_rng(7)
    net = make_network(N_CARDS, SMALL)
    obs_np, _ = _batch(rng, batch=1)
    single = {k: v[0] for k, v in obs_np.items()}
    tensors = obs_to_tensors(single, DEVICE)
    assert tensors["units"].shape[0] == 1
    assert net.trunk(tensors).shape[0] == 1


# -------------------------------------------------------------- integration


def test_env_emits_units_when_asked(cards, arena, decks):
    from src.simulator.env import CRBattleEnv

    deck = [cards[n] for n in decks["training_mirror"]]
    env = CRBattleEnv(cards, arena, deck, list(deck), tier=obs_layout.TIER_HUMAN,
                      with_units=True, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs["units"].shape == (obs_layout.MAX_ENTITIES, obs_layout.UNIT_FEATURE_DIM)
    assert "units" in env.observation_space.spaces
    assert (obs["units"][:, 1] > 0).sum() == len(env.engine.towers)


def test_restricted_tier_gets_a_zeroed_entity_list(cards, arena):
    engine = make_engine(cards, arena)
    spawn_unit(engine, cards["knight"], Side.TOP, 9.0, 22.0)
    ids = {n: i for i, n in enumerate(cards)}
    obs = obs_layout.encode_obs(engine, Side.BOTTOM, ids,
                                obs_layout.TIER_RESTRICTED, with_units=True)
    assert not obs["units"].any()


def test_noise_drops_list_entries_rather_than_smearing_a_grid(cards, arena):
    """The #32 interaction: a set encoder consumes detections directly, so
    randomization is a list edit."""
    engine = make_engine(cards, arena)
    for i in range(4):
        spawn_unit(engine, cards["knight"], Side.TOP, 4.0 + i, 22.0)
    ids = {n: i for i, n in enumerate(cards)}
    noise = ObservationNoise(ObsNoiseConfig(enabled=True, p_miss=1.0), seed=0)
    views = noise.perturb(obs_layout.unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    units = obs_layout.encode_units(engine, Side.BOTTOM, ids, views)
    present = units[units[:, 1] > 0]
    assert len(present) == len(engine.towers)
    hostile = present[present[:, 6] == 0]
    assert (hostile[:, 8] == 1.0).all(), "only enemy towers survive; troops were dropped"


def test_ppo_buffer_carries_the_units_key():
    """Guards the plumbing: RolloutBuffer must round-trip whatever keys the
    observation has, not a hardcoded triple."""
    from src.agent.ppo import RolloutBuffer

    rng = np.random.default_rng(8)
    obs, masks = _batch(rng, batch=2)
    buffer = RolloutBuffer(1, 2, {k: v.shape[1:] for k, v in obs.items()},
                           {k: v.shape[1:] for k, v in masks.items()})
    buffer.add(obs, masks, np.zeros((2, 2), np.int64), np.zeros(2, np.float32),
               np.zeros(2, np.float32), np.zeros(2, np.float32), np.zeros(2, bool))
    np.testing.assert_array_equal(buffer.obs["units"][0], obs["units"])


def test_deploying_and_frozen_are_separate_features(cards, arena):
    """Both stop a unit acting, but a deploy lock always expires on its own
    while a freeze can be re-applied — and one is a future threat where the
    other is a present opportunity. Folding them into one flag would make
    those indistinguishable to the policy."""
    engine = make_engine(cards, arena)
    ids = {n: i for i, n in enumerate(cards)}
    deploying = spawn_unit(engine, cards["knight"], Side.TOP, 9.0, 22.0)
    deploying.deployed_at = engine.time          # still in its deploy lock
    frozen = spawn_unit(engine, cards["knight"], Side.TOP, 11.0, 22.0)
    frozen.frozen_until = engine.time + 4.0

    views = {(v.x, v.y): v for v in obs_layout.unit_views(engine, Side.BOTTOM)}
    assert views[(9.0, 22.0)].deploying and not views[(9.0, 22.0)].frozen
    assert views[(11.0, 22.0)].frozen and not views[(11.0, 22.0)].deploying
    assert views[(11.0, 22.0)].inert

    cols = {name: i for i, name in enumerate(obs_layout.UNIT_FEATURES)}
    units = obs_layout.encode_units(engine, Side.BOTTOM, ids)
    present = units[units[:, 1] > 0]
    assert present[:, cols["deploying"]].sum() == 1
    assert present[:, cols["frozen"]].sum() == 1
