"""Fast PPO smoke test: tiny rollout + one optimizer update."""
import numpy as np
import torch

from src.agent.network import make_network, masks_to_tensors, obs_to_tensors
from src.agent.ppo import PPOConfig, RolloutBuffer, ppo_update
from src.agent.rewards import RewardShaper
from src.agent.selfplay import BotOpponent
from src.agent.train import ShapedRewardFn, make_setup_fn
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv
from src.training.config import load_training_config
from src.training.curriculum import load_curriculum


def test_ppo_rollout_and_update():
    cards = load_cards()
    arena = load_arena()
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    stage = next(s for s in load_curriculum() if s.name == "one_lane")
    card_names = list(cards.keys())
    net = make_network(len(card_names), training_cfg.raw.get("network"))
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    shaper = RewardShaper()
    reward_fn = ShapedRewardFn(shaper)
    bot = get_bot("rusher", catalog=catalog, deck_name="stage0",
                  rng=np.random.default_rng(0))
    setup_fn = make_setup_fn(stage, catalog, training_cfg, pool=None, latest_bot=None)

    def env_fn():
        deck = catalog.resolve("stage0")
        return CRBattleEnv(
            cards, arena, deck, list(deck), reward_fn=reward_fn,
            opponent=BotOpponent(bot), lanes="right", regulation=90.0,
            setup_fn=setup_fn,
        )

    n_envs = 2
    n_steps = 8
    envs = SyncVecEnv([env_fn for _ in range(n_envs)])
    obs, masks = envs.reset(seed=0)
    obs_shapes = {k: v.shape[1:] for k, v in obs.items()}
    mask_shapes = {k: v.shape[1:] for k, v in masks.items()}
    buffer = RolloutBuffer(n_steps, n_envs, obs_shapes, mask_shapes)

    device = torch.device("cpu")
    for _ in range(n_steps):
        obs_t = obs_to_tensors(obs, device)
        masks_t = masks_to_tensors(masks, device)
        actions, log_probs, values = net.act(obs_t, masks_t)
        next_obs, rewards, dones, next_masks, _ = envs.step(actions.cpu().numpy())
        buffer.add(obs, masks, actions.cpu().numpy(), log_probs.cpu().numpy(),
                   values.cpu().numpy(), rewards, dones)
        obs, masks = next_obs, next_masks

    with torch.no_grad():
        _, _, last_values = net.act(obs_to_tensors(obs, device),
                                    masks_to_tensors(masks, device))
    ppo_cfg = PPOConfig(n_steps=n_steps, batch_size=16, n_epochs=1)
    buffer.compute_returns(last_values.cpu().numpy(), ppo_cfg.gamma, ppo_cfg.gae_lambda)
    stats = ppo_update(net, optimizer, buffer, ppo_cfg, ent_coef=0.01,
                       device=device, rng=np.random.default_rng(0))

    assert np.isfinite(rewards).all()
    assert stats["policy_loss"] == stats["policy_loss"]  # not NaN
    assert stats["value_loss"] >= 0.0
