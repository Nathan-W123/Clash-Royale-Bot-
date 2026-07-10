"""Minimal PPO (clipped surrogate + GAE) over the factored masked policy.

Kept deliberately small: the interesting complexity in this project is
opponent sampling and the environment, not the RL algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.agent.network import PolicyNetwork, masks_to_tensors, obs_to_tensors

OBS_KEYS = ("spatial", "cards", "vector")


@dataclass(frozen=True)
class PPOConfig:
    n_steps: int = 512
    batch_size: int = 1024
    n_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.002
    max_grad_norm: float = 0.5
    total_steps: int = 3_000_000

    def ent_coef(self, global_step: int) -> float:
        t = min(1.0, global_step / max(self.total_steps, 1))
        return self.ent_coef_start + t * (self.ent_coef_end - self.ent_coef_start)


class RolloutBuffer:
    """(T, N, ...) storage for one on-policy rollout."""

    def __init__(self, n_steps: int, n_envs: int, obs_shapes: dict, mask_shapes: dict):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.obs = {k: np.zeros((n_steps, n_envs, *shape),
                                np.int64 if k == "cards" else np.float32)
                    for k, shape in obs_shapes.items()}
        self.masks = {k: np.zeros((n_steps, n_envs, *shape), bool)
                      for k, shape in mask_shapes.items()}
        self.actions = np.zeros((n_steps, n_envs, 2), np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), np.float32)
        self.values = np.zeros((n_steps, n_envs), np.float32)
        self.rewards = np.zeros((n_steps, n_envs), np.float32)
        self.dones = np.zeros((n_steps, n_envs), bool)
        self.t = 0

    def add(self, obs, masks, actions, log_probs, values, rewards, dones) -> None:
        i = self.t
        for k in OBS_KEYS:
            self.obs[k][i] = obs[k]
        for k in ("card", "place"):
            self.masks[k][i] = masks[k]
        self.actions[i] = actions
        self.log_probs[i] = log_probs
        self.values[i] = values
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.t += 1

    def compute_returns(self, last_values: np.ndarray, gamma: float, lam: float):
        """GAE; `dones[t]` marks that step t ended an episode (no bootstrap across)."""
        adv = np.zeros_like(self.rewards)
        gae = np.zeros(self.n_envs, np.float32)
        for t in reversed(range(self.n_steps)):
            next_values = last_values if t == self.n_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t].astype(np.float32)
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + self.values


def ppo_update(
    net: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    config: PPOConfig,
    ent_coef: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = buffer.n_steps * buffer.n_envs
    flat_obs = {k: v.reshape(n, *v.shape[2:]) for k, v in buffer.obs.items()}
    flat_masks = {k: v.reshape(n, *v.shape[2:]) for k, v in buffer.masks.items()}
    actions = torch.as_tensor(buffer.actions.reshape(n, 2), device=device)
    old_log_probs = torch.as_tensor(buffer.log_probs.reshape(n), device=device)
    advantages = buffer.advantages.reshape(n)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(buffer.returns.reshape(n), device=device)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    batches = 0
    for _ in range(config.n_epochs):
        order = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            idx = order[start:start + config.batch_size]
            obs = obs_to_tensors({k: flat_obs[k][idx] for k in OBS_KEYS}, device)
            masks = masks_to_tensors(
                {"card": flat_masks["card"][idx], "place": flat_masks["place"][idx]},
                device)
            idx_t = torch.as_tensor(idx, device=device)
            log_probs, entropy, values = net.evaluate_actions(obs, masks, actions[idx_t])

            ratio = torch.exp(log_probs - old_log_probs[idx_t])
            adv = advantages[idx_t]
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns[idx_t])
            loss = (policy_loss + config.vf_coef * value_loss
                    - ent_coef * entropy.mean())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy.mean())
                stats["clip_frac"] += float(
                    ((ratio - 1).abs() > config.clip_range).float().mean())
            batches += 1
    return {k: v / max(batches, 1) for k, v in stats.items()}
