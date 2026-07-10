"""Synchronous vectorized wrapper over N CRBattleEnv instances.

Single-process: the pure-Python engine is fast enough that spawn overhead on
Windows isn't worth it. Auto-resets finished episodes; the terminal
observation is replaced by the fresh reset obs (episode metrics arrive via
`infos`), which is what an on-policy learner wants.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.simulator.env import CRBattleEnv


def _stack_obs(obs_list: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _stack_masks(mask_list: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([m[k] for m in mask_list]) for k in mask_list[0]}


class SyncVecEnv:
    def __init__(self, env_fns: list[Callable[[], CRBattleEnv]]):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)

    def reset(self, seed: int | None = None):
        obs, masks = [], []
        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=None if seed is None else seed + i)
            obs.append(o)
            masks.append(info["masks"])
        return _stack_obs(obs), _stack_masks(masks)

    def step(self, actions: np.ndarray):
        obs, rewards, dones, masks, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            o, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                o, reset_info = env.reset()
                info["masks"] = reset_info["masks"]
            obs.append(o)
            rewards.append(r)
            dones.append(done)
            masks.append(info["masks"])
            infos.append(info)
        return (_stack_obs(obs), np.asarray(rewards, np.float32),
                np.asarray(dones, bool), _stack_masks(masks), infos)

    def set_attr(self, name: str, value) -> None:
        for env in self.envs:
            setattr(env, name, value)
