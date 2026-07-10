import numpy as np
from gymnasium.utils.env_checker import check_env

from src.simulator.constants import HAND_SIZE, Side
from src.simulator.env import N_CARD_CHOICES, N_CELLS, CRBattleEnv
from tests.conftest import spawn_unit


def make_env(cards, arena, **kwargs):
    deck = [cards[n] for n in ["knight", "archers", "goblins", "giant",
                               "musketeer", "minions", "fireball", "cannon"]]
    return CRBattleEnv(cards, arena, deck, list(deck), seed=3, **kwargs)


def test_gymnasium_check_env(cards, arena):
    check_env(make_env(cards, arena), skip_render_check=True)


def test_obs_and_mask_shapes(cards, arena):
    env = make_env(cards, arena)
    obs, info = env.reset(seed=0)
    assert obs["spatial"].shape == (10, 16, 9)
    assert obs["cards"].shape == (HAND_SIZE + 1,)
    assert obs["vector"].shape == (17,)
    assert info["masks"]["card"].shape == (N_CARD_CHOICES,)
    assert info["masks"]["place"].shape == (HAND_SIZE, N_CELLS)


def _force_equal_hands(env):
    pb, pt = env.engine.players[Side.BOTTOM], env.engine.players[Side.TOP]
    pt.hand = list(pb.hand)
    pt._queue = list(pb._queue)


def test_perspective_flip_symmetry(cards, arena):
    """A fully mirrored state must produce identical obs from either seat."""
    env = make_env(cards, arena)
    env.reset(seed=0)
    _force_equal_hands(env)
    spawn_unit(env.engine, cards["knight"], Side.BOTTOM, 4.0, 10.0)
    spawn_unit(env.engine, cards["knight"], Side.TOP, 4.0, 22.0)  # mirror position
    obs_b, obs_t = env.build_obs(Side.BOTTOM), env.build_obs(Side.TOP)
    for key in obs_b:
        np.testing.assert_array_equal(obs_b[key], obs_t[key])
    masks_b, masks_t = env.build_masks(Side.BOTTOM), env.build_masks(Side.TOP)
    np.testing.assert_array_equal(masks_b["card"], masks_t["card"])
    np.testing.assert_array_equal(masks_b["place"], masks_t["place"])


def test_episode_metrics_on_termination(cards, arena):
    env = make_env(cards, arena, regulation=5.0)
    env.reset(seed=0)
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step((0, 0))
    m = info["episode_metrics"]
    assert m["result"] == "DRAW"
    assert m["match_time"] >= 5.0  # idle match runs to overtime expiry
