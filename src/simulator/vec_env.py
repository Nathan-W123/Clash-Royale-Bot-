"""Synchronous vectorized wrapper over N CRBattleEnv instances.

Single-process, and still the default. Measured on an idle machine at 8 envs
this beats the subprocess variant (see `subproc_vec_env.py` and
`scripts/bench_throughput.py`); process parallelism only starts paying at
around 32 envs, because splitting envs across workers narrows the batched
opponent forward that `_batched_opponent_actions` depends on.

Auto-resets finished episodes; the terminal observation is replaced by the
fresh reset obs (episode metrics arrive via `infos`), which is what an
on-policy learner wants.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.simulator.constants import Side
from src.simulator.env import UNSET, CRBattleEnv


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

    def _batched_opponent_actions(self) -> list:
        """Resolve policy-backed opponents for all envs in as few forwards as possible.

        Each env used to run its own opponent forward pass, so a vectorized
        step issued N batch-of-1 calls into torch. Those are dominated by
        per-call overhead, and profiling attributed ~90% of training
        wall-clock to torch rather than the simulator. Here envs whose
        opponents share a network (the common case: many envs facing the same
        `latest` policy, or the same pooled checkpoint) are grouped and
        evaluated in one call.

        Scripted bots are left alone — they are cheap Python heuristics with
        no tensor work to amortize. Their envs get UNSET so `CRBattleEnv.step`
        resolves them itself, exactly as before.
        """
        resolved: list = [UNSET] * self.n
        groups: dict = {}
        for i, env in enumerate(self.envs):
            opponent = env.opponent
            key_fn = getattr(opponent, "batch_key", None)
            if key_fn is None or env.engine is None:
                continue  # scripted bot, or no opponent: leave to the env
            key = key_fn()
            if key is None:
                continue  # opted out of batching
            groups.setdefault(key, []).append(i)

        for _, idxs in groups.items():
            bot = self.envs[idxs[0]].opponent
            engines = [self.envs[i].engine for i in idxs]
            # `batched_rows` also threads per-engine hidden state, so
            # recurrent opponents keep both their memory and the batching win.
            rows = bot.batched_rows(engines, Side.TOP)
            for i, row in zip(idxs, rows):
                action = bot.decode_row(self.envs[i].engine, Side.TOP, row)
                resolved[i] = (None if action is None
                               else (action.slot, action.x, action.y))
        return resolved

    def step(self, actions: np.ndarray):
        opp_actions = self._batched_opponent_actions()
        obs, rewards, dones, masks, infos = [], [], [], [], []
        for env, action, opp_action in zip(self.envs, actions, opp_actions):
            o, r, terminated, truncated, info = env.step(
                action, opponent_action=opp_action)
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
