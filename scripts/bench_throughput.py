"""Measure env throughput (steps/sec) for the real training configuration.

Kept in the repo because throughput is now a tracked metric: matches got
longer with collision + deploy delay, and a 1M-step run is hours, so
regressions here matter as much as win-rate regressions.

Run on an *idle* machine. Contention from other processes swamps the signal —
readings on a loaded box have varied by 4x for identical code.

    python -m scripts.bench_throughput --steps 256 --n-envs 8
    python -m scripts.bench_throughput --steps 256 --n-envs 8 --workers 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.agent.league import CheckpointPool
from src.agent.network import make_network, masks_to_tensors, obs_to_tensors
from src.agent.rewards import RewardShaper
from src.agent.selfplay import PolicyBot
from src.agent.train import ShapedRewardFn, make_setup_fn
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv
from src.training.config import load_training_config
from src.training.curriculum import load_curriculum

_TMP_POOL = Path("runs/_bench_pool")


def build_env_fn(stage_name: str = "full_pool", tier: str = "restricted"):
    """Return (env_fn, live_bot) matching how train.py builds envs."""
    cards = load_cards()
    arena = load_arena()
    catalog = DeckCatalog()
    cfg = load_training_config(Path("configs/training_restricted.yaml"))
    stage = next(s for s in load_curriculum() if s.name == stage_name)
    names = list(cards.keys())
    net = make_network(len(names), cfg.raw.get("network"))
    reward_fn = ShapedRewardFn(RewardShaper())
    pool = CheckpointPool(_TMP_POOL)
    live_bot = PolicyBot(net, names, name="latest", deterministic=False)
    setup_fn = make_setup_fn(stage, catalog, cfg, pool, live_bot)

    def env_fn():
        deck = catalog.resolve("training_mirror")
        return CRBattleEnv(cards, arena, deck, list(deck), reward_fn=reward_fn,
                           lanes="both", regulation=None, setup_fn=setup_fn,
                           use_spatial=False)

    return env_fn, live_bot, net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = SyncVecEnv (single process); >0 = SubprocVecEnv")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="0 = leave torch defaults alone")
    args = ap.parse_args()

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    env_fn, live_bot, net = build_env_fn()
    env_fns = [env_fn for _ in range(args.n_envs)]

    if args.workers:
        from src.simulator.subproc_vec_env import SubprocVecEnv
        envs = SubprocVecEnv(env_fns, n_workers=args.workers, live_bot=live_bot)
        label = f"SubprocVecEnv({args.workers} workers)"
    else:
        envs = SyncVecEnv(env_fns)
        label = "SyncVecEnv"

    try:
        obs, masks = envs.reset(seed=0)
        device = torch.device("cpu")
        # Warm up so import/JIT/alloc costs don't land in the timed window.
        for _ in range(5):
            a, _, _ = net.act(obs_to_tensors(obs, device), masks_to_tensors(masks, device))
            obs, _, _, masks, _ = envs.step(a.cpu().numpy())

        t0 = time.perf_counter()
        for _ in range(args.steps):
            a, _, _ = net.act(obs_to_tensors(obs, device), masks_to_tensors(masks, device))
            obs, _, _, masks, _ = envs.step(a.cpu().numpy())
        dt = time.perf_counter() - t0
    finally:
        if hasattr(envs, "close"):
            envs.close()

    total = args.steps * args.n_envs
    sps = total / dt
    print(f"{label:28s} {args.n_envs} envs  {total} steps in {dt:6.1f}s "
          f"-> {sps:6.1f} sps   1M steps = {1e6 / sps / 3600:.1f} h")


if __name__ == "__main__":
    main()
