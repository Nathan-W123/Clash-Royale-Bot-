"""Gym env smoke tests: reset/step, masks, episode termination."""
import numpy as np

from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.constants import Side
from src.simulator.env import N_CARD_CHOICES, N_CELLS, CRBattleEnv
from src.agent.selfplay import BotOpponent
def _make_env(cards, arena):
    deck = [cards[n] for n in ["knight", "archers", "goblins", "giant",
                               "musketeer", "minions", "fireball", "cannon"]]
    catalog = DeckCatalog()
    bot = get_bot("rusher", catalog=catalog, deck_name="rusher",
                  rng=np.random.default_rng(0))
    return CRBattleEnv(
        cards, arena, deck, list(deck),
        opponent=BotOpponent(bot), seed=1,
    )


def test_reset_returns_obs_and_masks(cards, arena):
    env = _make_env(cards, arena)
    obs, info = env.reset(seed=0)
    assert set(obs) == {"spatial", "cards", "vector"}
    assert info["masks"]["card"].shape == (N_CARD_CHOICES,)
    assert info["masks"]["place"].shape == (4, N_CELLS)
    assert info["masks"]["card"][0]  # no-op always legal


def test_step_advances_and_refreshes_masks(cards, arena):
    env = _make_env(cards, arena)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step((0, 0))
    assert not truncated
    assert isinstance(reward, float)
    assert set(obs) == {"spatial", "cards", "vector"}
    if not terminated:
        assert "masks" in info
        assert info["masks"]["card"].dtype == bool


def test_episode_terminates_with_metrics(cards, arena):
    env = _make_env(cards, arena)
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated and steps < 5000:
        masks = env.build_masks(Side.BOTTOM)
        card = int(np.argmax(masks["card"]))
        cell = 0
        if card > 0 and masks["place"][card - 1].any():
            cell = int(np.flatnonzero(masks["place"][card - 1])[0])
        _, _, terminated, _, info = env.step((card, cell))
        steps += 1
    assert terminated
    assert "episode_metrics" in info
    assert info["episode_metrics"]["result"] in {
        "BOTTOM_WIN", "TOP_WIN", "DRAW",
    }
