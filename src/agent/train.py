"""PPO training CLI with curriculum stages and league self-play.

Examples:
  uv run python -m src.agent.train --run r1 --stage one_lane --steps 400000
  uv run python -m src.agent.train --run r1 --auto --bc-init checkpoints/bc_init.pt
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.agent.league import CheckpointPool
from src.agent.network import make_network, masks_to_tensors, obs_to_tensors
from src.agent.ppo import PPOConfig, RolloutBuffer, ppo_update
from src.agent.rewards import RewardShaper
from src.agent.selfplay import BotOpponent, PolicyBot, load_checkpoint, save_checkpoint
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.constants import MatchResult, Side
from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv
from src.training.config import load_training_config
from src.training.curriculum import CurriculumStage, load_curriculum
from src.training.match_runner import run_match_detailed

DEVICE = torch.device("cpu")


class ShapedRewardFn:
    """Env-side reward: annealed dense shaping + terminal. Shared across envs
    so the trainer can move `weight` once per rollout."""

    def __init__(self, shaper: RewardShaper):
        self.shaper = shaper
        self.weight = shaper.config.anneal_start

    def __call__(self, events, side, engine) -> float:
        r = self.weight * self.shaper.compute_step_reward(events, side)
        if engine.result != MatchResult.ONGOING:
            r += self.shaper.terminal_reward(engine.result, side)
        return r


def resolve_stage_decks(stage: CurriculumStage, catalog: DeckCatalog, rng):
    if stage.deck:
        name_b = stage.deck
    else:
        pool_name = stage.agent_deck_pool or stage.deck_pool or "stage2_pool"
        name_b = str(rng.choice(catalog.pool(pool_name)))
    if stage.opponent_deck:
        name_t = stage.opponent_deck
    elif stage.opponent_deck_pool or stage.deck_pool:
        pool_name = stage.opponent_deck_pool or stage.deck_pool
        name_t = str(rng.choice(catalog.pool(pool_name)))
    else:
        name_t = name_b
    return name_b, name_t


def make_setup_fn(stage, catalog, training_cfg, pool, latest_bot):
    """Per-episode (deck_b, deck_t, opponent) sampler for one env."""
    scripted_names = list(stage.opponents) or list(training_cfg.opponents.scripted_bots)
    cfg = training_cfg.opponents

    def scripted(rng, deck_t_name):
        name = str(rng.choice(scripted_names))
        # Archetype bots fight best on their archetype deck when it exists.
        deck_name = name if name in catalog.decks else deck_t_name
        bot = get_bot(name, catalog=catalog, deck_name=deck_name, rng=rng,
                      skill_tier=cfg.skill_tier)
        return BotOpponent(bot), deck_name

    def setup(rng):
        name_b, name_t = resolve_stage_decks(stage, catalog, rng)
        if not stage.selfplay:
            opponent, name_t = scripted(rng, name_t)
        else:
            r = float(rng.random())
            if r < cfg.sample_scripted:
                opponent, name_t = scripted(rng, name_t)
            elif r < cfg.sample_scripted + cfg.sample_latest or not pool.members():
                opponent = latest_bot
            else:
                opponent = pool.sample_opponent(rng) or latest_bot
        return catalog.resolve(name_b), catalog.resolve(name_t), opponent

    return setup


def quick_eval(net, card_names, stage, catalog, arena, training_cfg,
               bots: list[str], matches: int, seed: int) -> dict[str, float]:
    """Deterministic policy vs each scripted bot; returns win rates
    (draws count half)."""
    policy = PolicyBot(net, card_names, name="eval", deterministic=True)
    rng = np.random.default_rng(seed)
    out = {}
    for bot_name in bots:
        deck_t_name = bot_name if bot_name in catalog.decks else \
            resolve_stage_decks(stage, catalog, rng)[1]
        score = 0.0
        for i in range(matches):
            name_b, _ = resolve_stage_decks(stage, catalog, rng)
            bot = get_bot(bot_name, catalog=catalog, deck_name=deck_t_name,
                          rng=rng, skill_tier=training_cfg.opponents.skill_tier)
            report = run_match_detailed(
                arena, catalog.resolve(name_b), catalog.resolve(deck_t_name),
                policy, bot, seed=int(rng.integers(2**31)),
                lanes=stage.single_lane or "both", regulation=stage.match_time)
            won = report.agent_won(Side.BOTTOM)
            score += 0.5 if won is None else float(won)
        out[bot_name] = score / matches
    return out


def train_stage(
    net,
    optimizer,
    stage: CurriculumStage,
    *,
    card_names,
    catalog,
    arena,
    cards,
    training_cfg,
    ppo_cfg: PPOConfig,
    pool: CheckpointPool,
    run_dir: Path,
    global_step: int,
    step_budget: int,
    n_envs: int,
    seed: int,
) -> int:
    shaper = RewardShaper()
    reward_fn = ShapedRewardFn(shaper)
    latest_bot = PolicyBot(net, card_names, name="latest", deterministic=False)
    setup_fn = make_setup_fn(stage, catalog, training_cfg, pool, latest_bot)

    def env_fn():
        deck = catalog.resolve(stage.deck or "training_mirror")
        return CRBattleEnv(cards, arena, deck, list(deck), reward_fn=reward_fn,
                           lanes=stage.single_lane or "both",
                           regulation=stage.match_time, setup_fn=setup_fn)

    envs = SyncVecEnv([env_fn for _ in range(n_envs)])
    obs, masks = envs.reset(seed=seed)
    rng = np.random.default_rng(seed)

    league_cfg = training_cfg.raw.get("league", {})
    eval_cfg = training_cfg.raw.get("eval", {})
    snapshot_every = int(league_cfg.get("snapshot_every", 100_000))
    eval_every = int(eval_cfg.get("every_steps", 100_000))
    next_snapshot = global_step + snapshot_every
    next_eval = global_step + eval_every

    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(str(run_dir / "tb"))
    except ImportError:
        pass
    for path, header in ((train_csv, ["step", "stage", "win_rate", "draw_rate",
                                      "reward_mean", "entropy", "policy_loss",
                                      "value_loss", "leak_per_match", "match_time",
                                      "card_usage_entropy", "steps_per_sec"]),
                         (eval_csv, ["step", "stage", "bot", "win_rate"])):
        if not path.exists():
            path.write_text(",".join(header) + "\n")

    stage_start = global_step
    obs_shapes = {k: v.shape[1:] for k, v in obs.items()}
    mask_shapes = {k: v.shape[1:] for k, v in masks.items()}
    ep_metrics: list[dict] = []

    while global_step - stage_start < step_budget:
        buffer = RolloutBuffer(ppo_cfg.n_steps, n_envs, obs_shapes, mask_shapes)
        t0 = time.perf_counter()
        reward_sum = 0.0
        for _ in range(ppo_cfg.n_steps):
            obs_t = obs_to_tensors(obs, DEVICE)
            masks_t = masks_to_tensors(masks, DEVICE)
            actions, log_probs, values = net.act(obs_t, masks_t)
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, next_masks, infos = envs.step(actions_np)
            buffer.add(obs, masks, actions_np, log_probs.cpu().numpy(),
                       values.cpu().numpy(), rewards, dones)
            reward_sum += float(rewards.sum())
            for info in infos:
                m = info.get("episode_metrics")
                if m:
                    ep_metrics.append(m)
                    opp = m["opponent"]
                    if opp.startswith(("step_", "anchor_")):
                        won = None if m["draw"] else bool(m["win"])
                        pool.record_result(f"{opp}.pt", won)
            obs, masks = next_obs, next_masks

        with torch.no_grad():
            _, _, last_values = net.act(obs_to_tensors(obs, DEVICE),
                                        masks_to_tensors(masks, DEVICE))
        buffer.compute_returns(last_values.cpu().numpy(), ppo_cfg.gamma,
                               ppo_cfg.gae_lambda)
        stats = ppo_update(net, optimizer, buffer, ppo_cfg,
                           ppo_cfg.ent_coef(global_step), DEVICE, rng)
        global_step += ppo_cfg.n_steps * n_envs
        reward_fn.weight = shaper.anneal_factor(global_step)
        sps = ppo_cfg.n_steps * n_envs / (time.perf_counter() - t0)

        if ep_metrics:
            wins = np.mean([m["win"] for m in ep_metrics])
            draws = np.mean([m["draw"] for m in ep_metrics])
            leak = np.mean([m["elixir_leaked"] for m in ep_metrics])
            mtime = np.mean([m["match_time"] for m in ep_metrics])
            usage = {}
            for m in ep_metrics:
                for k, v in m["card_usage"].items():
                    usage[k] = usage.get(k, 0) + v
            total = sum(usage.values()) or 1
            probs = np.array([v / total for v in usage.values()])
            usage_entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        else:
            wins = draws = leak = mtime = usage_entropy = 0.0
        with train_csv.open("a", newline="") as f:
            csv.writer(f).writerow(
                [global_step, stage.name, f"{wins:.3f}", f"{draws:.3f}",
                 f"{reward_sum / (ppo_cfg.n_steps * n_envs):.4f}",
                 f"{stats['entropy']:.3f}", f"{stats['policy_loss']:.4f}",
                 f"{stats['value_loss']:.4f}", f"{leak:.2f}", f"{mtime:.0f}",
                 f"{usage_entropy:.3f}", f"{sps:.0f}"])
        if tb_writer is not None:
            tb_writer.add_scalar("train/win_rate", wins, global_step)
            tb_writer.add_scalar("train/reward_mean",
                                 reward_sum / (ppo_cfg.n_steps * n_envs), global_step)
            tb_writer.add_scalar("train/entropy", stats["entropy"], global_step)
            tb_writer.add_scalar("train/policy_loss", stats["policy_loss"], global_step)
            tb_writer.add_scalar("train/value_loss", stats["value_loss"], global_step)
            tb_writer.add_scalar("train/reward_anneal", reward_fn.weight, global_step)
            tb_writer.add_scalar("train/steps_per_sec", sps, global_step)
        print(f"[{stage.name}] step {global_step:>8}  win {wins:.2f}  draw {draws:.2f}  "
              f"ent {stats['entropy']:.2f}  anneal {reward_fn.weight:.2f}  {sps:.0f} sps  "
              f"eps {len(ep_metrics)}")
        ep_metrics = []

        if global_step >= next_snapshot:
            pool.snapshot(net, card_names, global_step)
            pool.save_ledger()
            save_checkpoint(net, card_names, run_dir / "latest.pt")
            next_snapshot += snapshot_every

        if global_step >= next_eval:
            bots = list(stage.opponents) or list(training_cfg.opponents.scripted_bots)
            scores = quick_eval(net, card_names, stage, catalog, arena, training_cfg,
                                bots, matches=20, seed=global_step)
            with eval_csv.open("a", newline="") as f:
                for bot, wr in scores.items():
                    csv.writer(f).writerow([global_step, stage.name, bot, f"{wr:.3f}"])
            if tb_writer is not None:
                for bot, wr in scores.items():
                    tb_writer.add_scalar(f"eval/win_rate/{bot}", wr, global_step)
            print(f"[{stage.name}] eval @ {global_step}: "
                  + "  ".join(f"{b}={wr:.2f}" for b, wr in scores.items()))
            next_eval += eval_every

            promote = stage.promote
            if promote and scores.get(promote.vs, 0.0) >= promote.win_rate:
                confirm = quick_eval(net, card_names, stage, catalog, arena,
                                     training_cfg, [promote.vs],
                                     matches=promote.matches, seed=global_step + 1)
                if confirm[promote.vs] >= promote.win_rate:
                    print(f"[{stage.name}] PROMOTED: {promote.vs} "
                          f"win rate {confirm[promote.vs]:.2f} "
                          f">= {promote.win_rate}")
                    save_checkpoint(net, card_names, run_dir / f"{stage.name}_final.pt")
                    if tb_writer is not None:
                        tb_writer.close()
                    return global_step
    save_checkpoint(net, card_names, run_dir / f"{stage.name}_final.pt")
    if tb_writer is not None:
        tb_writer.close()
    return global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run1")
    parser.add_argument("--stage", default=None, help="stage name; omit with --auto")
    parser.add_argument("--auto", action="store_true", help="advance through all stages")
    parser.add_argument("--steps", type=int, default=None,
                        help="per-stage step budget (default: ppo.total_steps)")
    parser.add_argument("--bc-init", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cards = load_cards()
    arena = load_arena()
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    raw_ppo = training_cfg.raw.get("ppo", {})
    ppo_cfg = PPOConfig(
        n_steps=int(raw_ppo.get("n_steps", 512)),
        batch_size=int(raw_ppo.get("batch_size", 1024)),
        n_epochs=int(raw_ppo.get("n_epochs", 4)),
        lr=float(raw_ppo.get("lr", 3e-4)),
        gamma=float(raw_ppo.get("gamma", 0.997)),
        gae_lambda=float(raw_ppo.get("gae_lambda", 0.95)),
        clip_range=float(raw_ppo.get("clip_range", 0.2)),
        vf_coef=float(raw_ppo.get("vf_coef", 0.5)),
        ent_coef_start=float(raw_ppo.get("ent_coef_start", 0.01)),
        ent_coef_end=float(raw_ppo.get("ent_coef_end", 0.002)),
        max_grad_norm=float(raw_ppo.get("max_grad_norm", 0.5)),
        total_steps=int(raw_ppo.get("total_steps", 3_000_000)),
    )
    n_envs = args.n_envs or int(raw_ppo.get("n_envs", 8))
    card_names = list(cards.keys())

    if args.resume:
        net, card_names = load_checkpoint(Path(args.resume))
        net.train()
    elif args.bc_init:
        net, card_names = load_checkpoint(Path(args.bc_init))
        net.train()
        print(f"[train] initialized from BC checkpoint {args.bc_init}")
    else:
        net = make_network(len(card_names),
                           training_cfg.raw.get("network"))
    optimizer = torch.optim.Adam(net.parameters(), lr=ppo_cfg.lr)

    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    league_cfg = training_cfg.raw.get("league", {})
    pool = CheckpointPool(Path("checkpoints") / args.run,
                          pool_max=int(league_cfg.get("pool_max", 20)),
                          anchor_every=int(league_cfg.get("anchor_every", 500_000)))

    stages = load_curriculum()
    if args.auto:
        selected = stages
    else:
        name = args.stage or stages[0].name
        selected = [s for s in stages if s.name == name]
        if not selected:
            raise SystemExit(f"unknown stage {name}")

    global_step = 0
    budget = args.steps or ppo_cfg.total_steps
    for stage in selected:
        print(f"=== stage {stage.name} (budget {budget}) ===")
        global_step = train_stage(
            net, optimizer, stage,
            card_names=card_names, catalog=catalog, arena=arena, cards=cards,
            training_cfg=training_cfg, ppo_cfg=ppo_cfg, pool=pool,
            run_dir=run_dir, global_step=global_step, step_budget=budget,
            n_envs=n_envs, seed=args.seed)
    save_checkpoint(net, card_names, run_dir / "final.pt")
    print(f"[train] done at step {global_step}; saved {run_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
