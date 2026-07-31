"""Privileged teacher -> human student distillation (#37).

A `full`-tier teacher reads the opponent's exact elixir. That makes it
simulator-only, but it also makes it a much better *supervisor* than any
signal the student can generate alone: it can be trained to high strength
first, then used to shape a `human`-tier student that only ever consumes
live-legal inputs at inference.

Three things this gets right that a naive setup gets wrong:

**Distil on states the student visits.** Supervising on teacher-visited
states is ordinary behaviour cloning and hits the classic distribution
mismatch — the student is never trained on the states its own mistakes
produce. Rollouts here are collected by the *student*; the teacher is only
ever asked "what would you do here?".

**Same underlying state, two encodings.** Teacher and student read one
engine, encoded at two tiers, so the KL compares like with like. The `human`
and `full` tiers share an identical spatial grid and card ids and differ
only in the scalar vector, which is why only the teacher's `vector` needs
storing alongside the rollout.

**Anneal lambda to zero.** The teacher makes plays that are correct *given*
knowledge of opponent elixir — committing a big push into a bar it knows is
empty. The student cannot know that, and imitating it forever would bake in
a superstition. Two guards: the KL term is masked out where the teacher's
and student's value estimates disagree sharply (a decent proxy for "the
teacher is acting on information the student lacks"), and lambda decays so
the student ends up optimizing its own return.

The PPO update is reimplemented here rather than parameterizing
`src/agent/ppo.py`, which is owned by the concurrent throughput work.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.agent import obs_layout
from src.agent.network import (
    PolicyNetwork,
    make_network,
    masks_to_tensors,
    obs_to_tensors,
)
from src.agent.ppo import PPOConfig, RolloutBuffer
from src.agent.selfplay import load_checkpoint, save_checkpoint
from src.simulator.constants import Side

DEVICE = torch.device("cpu")


@dataclass(frozen=True)
class DistillConfig:
    lambda_start: float = 1.0
    lambda_end: float = 0.0
    anneal_over_steps: int = 1_500_000
    # Skip the KL term where |V_teacher - V_student| exceeds this. Large
    # disagreement is the signature of a teacher acting on the one thing the
    # student structurally cannot see.
    value_gap_mask: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict | None) -> "DistillConfig":
        raw = raw or {}
        return cls(
            lambda_start=float(raw.get("lambda_start", 1.0)),
            lambda_end=float(raw.get("lambda_end", 0.0)),
            anneal_over_steps=int(raw.get("anneal_over_steps", 1_500_000)),
            value_gap_mask=float(raw.get("value_gap_mask", 0.5)),
        )

    def weight(self, global_step: int) -> float:
        t = min(1.0, global_step / max(self.anneal_over_steps, 1))
        return self.lambda_start + t * (self.lambda_end - self.lambda_start)


class TeacherObs:
    """Rollout-aligned storage for the teacher's view of the same states.

    Only what actually differs between the tiers is kept. For a `human`
    student that is the scalar vector alone; for a `restricted` student the
    spatial grid is zero-filled on the student side and must be carried too.
    """

    def __init__(self, n_steps: int, n_envs: int, scalar_dim: int,
                 share_spatial: bool, spatial_shape):
        self.share_spatial = share_spatial
        self.vector = np.zeros((n_steps, n_envs, scalar_dim), np.float32)
        self.spatial = (None if share_spatial
                        else np.zeros((n_steps, n_envs, *spatial_shape), np.float32))

    def add(self, t: int, vector: np.ndarray, spatial: np.ndarray | None) -> None:
        self.vector[t] = vector
        if self.spatial is not None:
            self.spatial[t] = spatial

    def flat(self, n: int, student_flat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(student_flat)
        out["vector"] = self.vector.reshape(n, *self.vector.shape[2:])
        if self.spatial is not None:
            out["spatial"] = self.spatial.reshape(n, *self.spatial.shape[2:])
        return out


def teacher_view(env, side: Side = Side.BOTTOM) -> tuple[np.ndarray, np.ndarray]:
    """The `full`-tier encoding of an env's current state.

    Reads the live engine rather than reconstructing from the student's
    observation, because the whole point is the fields the student's
    encoding dropped.
    """
    obs = obs_layout.encode_obs(env.engine, side, env.card_index, obs_layout.TIER_FULL)
    return obs["vector"], obs["spatial"]


def _factored_kl(
    teacher: PolicyNetwork,
    student: PolicyNetwork,
    teacher_obs: dict[str, torch.Tensor],
    student_obs: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(per-state KL(teacher || student), teacher values).

    The policy is factored — card choice, then placement conditioned on that
    card — so the KL is the card-distribution KL plus the placement KL under
    the card actually taken, gated off for no-op rows exactly as the log-prob
    is elsewhere.
    """
    with torch.no_grad():
        t_feat = teacher.trunk(teacher_obs)
        t_card = teacher.card_logits(t_feat, masks["card"])
        t_place = teacher.place_logits(t_feat, teacher_obs["cards"],
                                       actions[:, 0], masks["place"])
        t_value = teacher.value_head(t_feat).squeeze(-1)

    s_feat = student.trunk(student_obs)
    s_card = student.card_logits(s_feat, masks["card"])
    s_place = student.place_logits(s_feat, student_obs["cards"],
                                   actions[:, 0], masks["place"])

    kl = F.kl_div(F.log_softmax(s_card, dim=-1),
                  F.log_softmax(t_card, dim=-1),
                  log_target=True, reduction="none").sum(-1)
    played = (actions[:, 0] > 0).float()
    kl = kl + played * F.kl_div(F.log_softmax(s_place, dim=-1),
                                F.log_softmax(t_place, dim=-1),
                                log_target=True, reduction="none").sum(-1)
    return kl, t_value


def distill_update(
    student: PolicyNetwork,
    teacher: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    teacher_obs: TeacherObs,
    ppo_cfg: PPOConfig,
    distill_cfg: DistillConfig,
    ent_coef: float,
    kl_weight: float,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    """PPO step plus `kl_weight * KL(teacher || student)` on student states."""
    n = buffer.n_steps * buffer.n_envs
    flat_obs = {k: v.reshape(n, *v.shape[2:]) for k, v in buffer.obs.items()}
    flat_teacher = teacher_obs.flat(n, flat_obs)
    flat_masks = {k: v.reshape(n, *v.shape[2:]) for k, v in buffer.masks.items()}
    actions = torch.as_tensor(buffer.actions.reshape(n, 2), device=device)
    old_log_probs = torch.as_tensor(buffer.log_probs.reshape(n), device=device)
    advantages = buffer.advantages.reshape(n)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = torch.as_tensor(advantages, device=device)
    returns = torch.as_tensor(buffer.returns.reshape(n), device=device)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "distill_kl": 0.0, "kl_masked_frac": 0.0}
    batches = 0
    for _ in range(ppo_cfg.n_epochs):
        order = rng.permutation(n)
        for start in range(0, n, ppo_cfg.batch_size):
            idx = order[start:start + ppo_cfg.batch_size]
            obs = obs_to_tensors({k: v[idx] for k, v in flat_obs.items()}, device)
            t_obs = obs_to_tensors({k: v[idx] for k, v in flat_teacher.items()}, device)
            masks = masks_to_tensors(
                {"card": flat_masks["card"][idx], "place": flat_masks["place"][idx]},
                device)
            idx_t = torch.as_tensor(idx, device=device)
            log_probs, entropy, values = student.evaluate_actions(
                obs, masks, actions[idx_t])

            ratio = torch.exp(log_probs - old_log_probs[idx_t])
            adv = advantages[idx_t]
            policy_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - ppo_cfg.clip_range, 1 + ppo_cfg.clip_range) * adv,
            ).mean()
            value_loss = F.mse_loss(values, returns[idx_t])
            loss = (policy_loss + ppo_cfg.vf_coef * value_loss
                    - ent_coef * entropy.mean())

            kl_mean = torch.zeros((), device=device)
            masked_frac = 0.0
            if kl_weight > 0.0:
                kl, t_value = _factored_kl(teacher, student, t_obs, obs, masks,
                                           actions[idx_t])
                keep = ((t_value - values.detach()).abs()
                        <= distill_cfg.value_gap_mask).float()
                masked_frac = float(1.0 - keep.mean())
                denom = keep.sum().clamp(min=1.0)
                kl_mean = (kl * keep).sum() / denom
                loss = loss + kl_weight * kl_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), ppo_cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy.mean())
                stats["distill_kl"] += float(kl_mean)
                stats["kl_masked_frac"] += masked_frac
            batches += 1
    return {k: v / max(batches, 1) for k, v in stats.items()}


def check_compatible(teacher: PolicyNetwork, student: PolicyNetwork) -> None:
    """Fail loudly on the pairings that silently produce garbage."""
    if teacher.config.tier != obs_layout.TIER_FULL:
        raise ValueError(
            f"teacher must be the privileged `full` tier, got {teacher.config.tier!r}; "
            "distilling from a non-privileged teacher has no information advantage")
    if student.config.tier == obs_layout.TIER_FULL:
        raise ValueError(
            "student is also `full` tier — it would inherit the opponent-elixir "
            "leak and could never be used live")
    if teacher.config.n_cards != student.config.n_cards:
        raise ValueError("teacher and student were built for different card rosters")


def distill_stage(
    student,
    teacher,
    optimizer,
    stage,
    *,
    card_names,
    catalog,
    arena,
    cards,
    training_cfg,
    ppo_cfg: PPOConfig,
    distill_cfg: DistillConfig,
    run_dir: Path,
    global_step: int,
    step_budget: int,
    n_envs: int,
    seed: int,
) -> int:
    """One curriculum stage of student-collected, teacher-supervised PPO."""
    from src.agent.obs_noise import ObservationNoise, ObsNoiseConfig
    from src.agent.rewards import RewardShaper
    from src.agent.selfplay import PolicyBot
    from src.agent.train import ShapedRewardFn, make_setup_fn
    from src.simulator.env import CRBattleEnv
    from src.simulator.vec_env import SyncVecEnv
    from src.agent.league import CheckpointPool

    check_compatible(teacher, student)
    teacher.eval()

    shaper = RewardShaper()
    reward_fn = ShapedRewardFn(shaper)
    latest_bot = PolicyBot(student, card_names, name="latest", deterministic=False)
    pool = CheckpointPool(Path("checkpoints") / run_dir.name)
    setup_fn = make_setup_fn(stage, catalog, training_cfg, pool, latest_bot)

    noise_cfg = ObsNoiseConfig.from_dict(training_cfg.raw.get("obs_noise"))
    tier = student.config.tier
    env_seed = seed

    def env_fn():
        nonlocal env_seed
        deck = catalog.resolve(stage.deck or "training_mirror")
        noise = (ObservationNoise(noise_cfg, seed=env_seed)
                 if noise_cfg.enabled and obs_layout.tier_uses_spatial(tier) else None)
        env_seed += 1
        return CRBattleEnv(cards, arena, deck, list(deck), reward_fn=reward_fn,
                           lanes=stage.single_lane or "both",
                           regulation=stage.match_time, setup_fn=setup_fn,
                           tier=tier, obs_noise=noise)

    envs = SyncVecEnv([env_fn for _ in range(n_envs)])
    obs, masks = envs.reset(seed=seed)
    rng = np.random.default_rng(seed)
    obs_shapes = {k: v.shape[1:] for k, v in obs.items()}
    mask_shapes = {k: v.shape[1:] for k, v in masks.items()}
    share_spatial = tier == obs_layout.TIER_HUMAN
    stage_start = global_step

    while global_step - stage_start < step_budget:
        buffer = RolloutBuffer(ppo_cfg.n_steps, n_envs, obs_shapes, mask_shapes)
        t_obs = TeacherObs(ppo_cfg.n_steps, n_envs,
                           obs_layout.SCALAR_DIM, share_spatial,
                           obs_shapes["spatial"])
        t0 = time.perf_counter()
        for t in range(ppo_cfg.n_steps):
            views = [teacher_view(e) for e in envs.envs]
            t_obs.add(t, np.stack([v for v, _ in views]),
                      None if share_spatial else np.stack([s for _, s in views]))
            actions, log_probs, values = student.act(
                obs_to_tensors(obs, DEVICE), masks_to_tensors(masks, DEVICE))
            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, next_masks, _ = envs.step(actions_np)
            buffer.add(obs, masks, actions_np, log_probs.cpu().numpy(),
                       values.cpu().numpy(), rewards, dones)
            obs, masks = next_obs, next_masks

        with torch.no_grad():
            _, _, last_values = student.act(obs_to_tensors(obs, DEVICE),
                                            masks_to_tensors(masks, DEVICE))
        buffer.compute_returns(last_values.cpu().numpy(), ppo_cfg.gamma,
                               ppo_cfg.gae_lambda)
        kl_weight = distill_cfg.weight(global_step)
        stats = distill_update(student, teacher, optimizer, buffer, t_obs,
                               ppo_cfg, distill_cfg, ppo_cfg.ent_coef(global_step),
                               kl_weight, DEVICE, rng)
        global_step += ppo_cfg.n_steps * n_envs
        reward_fn.weight = shaper.anneal_factor(global_step)
        sps = ppo_cfg.n_steps * n_envs / (time.perf_counter() - t0)
        print(f"[distill/{stage.name}] step {global_step:>8}  "
              f"lambda {kl_weight:.3f}  KL {stats['distill_kl']:.4f}  "
              f"masked {stats['kl_masked_frac']:.2f}  ent {stats['entropy']:.2f}  "
              f"{sps:.0f} sps")

    save_checkpoint(student, card_names, run_dir / f"{stage.name}_final.pt")
    return global_step


def main() -> None:  # pragma: no cover - CLI
    from src.decks.catalog import DeckCatalog
    from src.simulator.cards import load_arena, load_cards
    from src.training.config import load_training_config
    from src.training.curriculum import load_curriculum

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, help="full-tier checkpoint")
    parser.add_argument("--run", default="distill1")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--student", default=None,
                        help="resume from a student checkpoint instead of a fresh net")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/training_human.yaml")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cards = load_cards()
    arena = load_arena()
    catalog = DeckCatalog()
    training_cfg = load_training_config(Path(args.config))
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
    distill_cfg = DistillConfig.from_dict(training_cfg.raw.get("distill"))

    teacher, card_names = load_checkpoint(Path(args.teacher))
    if args.student:
        student, card_names = load_checkpoint(Path(args.student))
    else:
        student = make_network(len(card_names), training_cfg.raw.get("network"))
    student.train()
    check_compatible(teacher, student)

    optimizer = torch.optim.Adam(student.parameters(), lr=ppo_cfg.lr)
    run_dir = Path("runs") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = load_curriculum()
    name = args.stage or stages[0].name
    selected = [s for s in stages if s.name == name]
    if not selected:
        raise SystemExit(f"unknown stage {name}")

    global_step = 0
    for stage in selected:
        global_step = distill_stage(
            student, teacher, optimizer, stage,
            card_names=card_names, catalog=catalog, arena=arena, cards=cards,
            training_cfg=training_cfg, ppo_cfg=ppo_cfg, distill_cfg=distill_cfg,
            run_dir=run_dir, global_step=global_step,
            step_budget=args.steps or ppo_cfg.total_steps,
            n_envs=args.n_envs or int(raw_ppo.get("n_envs", 8)), seed=args.seed)
    save_checkpoint(student, card_names, run_dir / "final.pt")
    print(f"[distill] done at step {global_step}; saved {run_dir / 'final.pt'}")


if __name__ == "__main__":  # pragma: no cover
    main()
