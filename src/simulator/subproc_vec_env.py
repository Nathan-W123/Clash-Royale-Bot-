"""Subprocess-parallel vectorized envs.

`SyncVecEnv`'s docstring claimed spawn overhead wasn't worth it. That was
true when the engine ran at ~400 steps/sec; it stopped being true once
matches got longer (collision + deploy delay) and measured throughput fell to
~41 steps/sec, where a 1M-step run costs hours. This splits the envs across
worker processes while keeping the exact `SyncVecEnv` API.

Design notes that are load-bearing on Windows:

* **Spawn only.** Windows has no `fork`, so every worker re-imports the world
  and receives its state by pickle. The `env_fn` closures in `train.py`
  capture a catalog, an arena, a reward function and a `setup_fn` — none of
  which plain `pickle` can handle — so the payload goes through
  `cloudpickle`.

* **Object identity across the boundary.** `env_fns` and the live policy are
  cloudpickled as a *single* payload on purpose. Cloudpickle preserves shared
  references within one payload, so the `latest` bot a worker's `setup_fn`
  closes over is the same object the worker holds for weight syncing.
  Pickling them separately would silently give each worker two unrelated
  copies, and weight updates would apply to a policy nothing was using.

* **Weight sync is exact, not approximate.** PPO does not mutate the network
  during rollout collection, so broadcasting `state_dict` at update
  boundaries reproduces single-process behaviour precisely rather than
  merely approximating it.

* **One torch thread per worker.** N workers each spawning a full thread pool
  oversubscribes the machine badly; the small nets here get nothing from
  intra-op parallelism anyway.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import cloudpickle
import numpy as np

from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv, _stack_masks, _stack_obs


def _worker(remote, payload: bytes, torch_threads: int) -> None:
    """Own a slice of envs; serve commands until told to close."""
    import torch

    torch.set_num_threads(max(1, torch_threads))
    env_fns, live_bot = cloudpickle.loads(payload)
    vec = SyncVecEnv(env_fns)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(vec.step(data))
            elif cmd == "reset":
                remote.send(vec.reset(seed=data))
            elif cmd == "set_weights":
                if live_bot is not None:
                    live_bot.net.load_state_dict(data)
                remote.send(True)
            elif cmd == "set_attr":
                vec.set_attr(*data)
                remote.send(True)
            elif cmd == "call":
                name, args, kwargs = data
                remote.send([getattr(e, name)(*args, **kwargs) for e in vec.envs])
            elif cmd == "close":
                remote.send(True)
                break
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        remote.close()


def _split(n_items: int, n_groups: int) -> list[slice]:
    """Contiguous, near-even slices; keeps env ordering stable across workers."""
    base, extra = divmod(n_items, n_groups)
    out, start = [], 0
    for i in range(n_groups):
        size = base + (1 if i < extra else 0)
        out.append(slice(start, start + size))
        start += size
    return out


class SubprocVecEnv:
    """Drop-in parallel replacement for `SyncVecEnv`.

    Each worker runs its own `SyncVecEnv`, so the batched-opponent-inference
    optimization applies *within* a worker automatically — the batch is
    `envs_per_worker` wide rather than `n_envs` wide, which is the one real
    cost of splitting.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], CRBattleEnv]],
        n_workers: int | None = None,
        live_bot=None,
        torch_threads: int = 1,
    ):
        self.n = len(env_fns)
        n_workers = min(n_workers or 4, self.n)
        self.slices = _split(self.n, n_workers)

        ctx = mp.get_context("spawn")
        self.remotes: list = []
        self.procs: list = []
        for sl in self.slices:
            parent, child = ctx.Pipe()
            # env_fns slice and live_bot travel together so cloudpickle keeps
            # the shared reference between them intact (see module docstring).
            payload = cloudpickle.dumps((env_fns[sl], live_bot))
            proc = ctx.Process(target=_worker, args=(child, payload, torch_threads),
                               daemon=True)
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(proc)
        self.closed = False

    # ---------------------------------------------------------------- api

    def reset(self, seed: int | None = None):
        for remote, sl in zip(self.remotes, self.slices):
            # Offset the seed by the slice start so env i gets the same seed
            # it would have received single-process.
            remote.send(("reset", None if seed is None else seed + sl.start))
        obs, masks = [], []
        for remote in self.remotes:
            o, m = remote.recv()
            obs.append(o)
            masks.append(m)
        return _concat_dicts(obs), _concat_dicts(masks)

    def step(self, actions: np.ndarray):
        for remote, sl in zip(self.remotes, self.slices):
            remote.send(("step", actions[sl]))
        obs, rewards, dones, masks, infos = [], [], [], [], []
        for remote in self.remotes:
            o, r, d, m, i = remote.recv()
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            masks.append(m)
            infos.extend(i)
        return (_concat_dicts(obs), np.concatenate(rewards),
                np.concatenate(dones), _concat_dicts(masks), infos)

    def set_weights(self, state_dict) -> None:
        """Push updated live-policy weights to every worker.

        Call after each PPO update. Without this, self-play opponents keep
        playing whatever policy existed when the workers were spawned.
        """
        cpu_sd = {k: v.detach().cpu() for k, v in state_dict.items()}
        for remote in self.remotes:
            remote.send(("set_weights", cpu_sd))
        for remote in self.remotes:
            remote.recv()

    def set_attr(self, name: str, value) -> None:
        for remote in self.remotes:
            remote.send(("set_attr", (name, value)))
        for remote in self.remotes:
            remote.recv()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
                remote.recv()
            except (OSError, EOFError):
                pass
            remote.close()
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():  # pragma: no cover - only on a wedged worker
                proc.terminate()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _concat_dicts(parts: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
