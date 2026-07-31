# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

# Project: Clash Royale RL Agent

## Overview
A machine learning agent that learns to play Clash Royale through reinforcement
learning, starting entirely against a custom-built simulator. The agent improves
over time via self-play. This is a personal/educational RL project, not a
competitive-play tool.

## Current Phase: Simulator Only
We are in Phase 1. Scope is strictly limited to:
- A custom Clash Royale battle simulator (`src/simulator/`)
- Scripted baseline bots for imitation bootstrapping and evaluation (`src/bots/`)
- An RL agent trained via self-play against the simulator (`src/agent/`)
- An evaluation harness benchmarking the agent against fixed scripted opponents (`src/eval/`)

## Live-Play Bridge (re-scoped 2026-07-14)
`src/live/` is an explicitly re-scoped exception to the simulator-only phase:
a calibrated Windows-desktop bridge (screen capture + synthetic touch input)
that watches an already-running local Clash Royale match and taps a
configured preset deck. It does not start matchmaking, read game memory, or
touch network traffic — see "Live-play scope" in README.md for the boundary.
This is personal/educational use against the user's own local client only.

The end goal is the trained policy driving live play, not the hand-rolled
heuristic in `src/live/runner.py`. That required a second re-scope
(2026-07-14): the existing full-vision observation
(`src/agent/obs_layout.py`'s default/`use_spatial=True` path) reads the
opponent's exact elixir count, which a real player can never see, so it is
simulator-only. `src/agent`/`src/simulator` therefore also support a
**restricted observation variant** (tier `"restricted"`; see
`_encode_scalars_restricted` in `obs_layout.py`) using only own hand, own
elixir, per-tower alive/dead status, king-activated flags, and match-clock
flags.

Implemented 2026-07-31: the tier is now a string
(`"full" | "human" | "restricted"`) threaded through `NetworkConfig.tier`,
`CRBattleEnv(tier=...)`, and the training configs. The old `use_spatial`
bool is still accepted everywhere as a deprecated alias (True -> `full`,
False -> `restricted`) so checkpoints written before the change keep
loading. Route width lookups through `obs_layout.scalar_dim_for` — the tiers
are *not* ordered by width.

Superseded in part on 2026-07-31: the earlier claim that the `(10,16,9)`
troop/tower/spell grid "isn't realistically reconstructable from screen
pixels" conflated *hard* with *off-limits*. Rendered troops are visible to a
human, so reconstructing that grid from the screen is legitimate and is now
the target — see "On-Screen Visual Perception" below. The `restricted` tier
remains the fallback for when vision is unavailable or untrusted.

Card identity for that restricted observation does not need visual
recognition: `src/live/runner.py`'s `HandCycle` already deterministically
simulates the known 8-card deck cycle from the configured
`opening_hand`/`draw_order`. The remaining live-vision work (own elixir bar,
own/opponent tower presence, confirming a tap consumed elixir) is a
lightweight read of the player's own already-visible UI — explicitly
distinct from reading opponent-hidden state, memory, or packets, which
stays out of scope below.

## On-Screen Visual Perception (re-scoped 2026-07-31)
Reading the rendered arena — **including enemy troop type and position** — is
in scope. Troops on screen are not hidden state: a human player sees them by
looking at the display, so a vision system that reads them replicates
ordinary perception rather than extracting anything the client conceals.
This is the same category as the already-permitted elixir-bar and tower
reads, just harder to implement.

The line that still matters is *derived vs. read*, not *easy vs. hard* and
not *how accurate*.

Opponent elixir and hand are **derivable, and strong players derive them
exactly**: elixir is deterministic arithmetic (start value + known regen +
observed card costs), and the 8-card cycle is deterministic once revealed,
so perfect tracking yields the opponent's actual hand rather than a guess.
Computing these from observed play is therefore fully in scope and is a core
skill to implement, not an approximation to tolerate — see the deterministic
opponent tracker task.

What stays out of scope is the *channel*, not the quantity. Memory and
packet access do not merely reveal elixir; they hand over the entire state,
including things no skill can yield — exact per-unit HP, spawn timers, the
opponent's full deck before a single card is played, RNG seeds, their
opening hand. A derived tracker is also wrong precisely when perception is
wrong (missed cast, off-screen spell), which is the same failure mode a
human has; a memory read never is.

Target observation tier for live play (`human`): spatial troop/tower grid
from vision + own hand/elixir + tower states — everything a skilled human
perceives, and nothing more. This sits between the existing `full` tier
(which also reads opponent elixir, so it is simulator-only and suitable as a
privileged *critic* input) and the `restricted` tier (no spatial grid at
all, retained as the fallback when vision is unavailable or untrusted).

Policies trained against perfect simulator positions degrade badly on noisy
real detections, so vision-facing training must apply domain randomization
(positional jitter, missed detections, false positives, identity confusion,
occlusion dropout) to the simulator observation.

Implemented 2026-07-31 (tasks #31–#41):
- `human` tier — `obs_layout._encode_scalars_human`, `configs/training_human.yaml`
- domain randomization — `src/agent/obs_noise.py` (enemy entities only,
  training only; the shipped rates are placeholders pending calibration
  against real detections)
- perception — `src/live/homography.py`, `src/live/vision.py`,
  `src/live/identify.py`
- deterministic opponent tracker — `src/agent/opponent_tracker.py`
- distillation, set encoder, search, PBT — `src/agent/distill.py`,
  `network.use_set_encoder`, `src/agent/search.py`, `src/agent/pbt.py`
- simulator fidelity audit and fixes — `docs/SIM_FIDELITY.md`,
  `src/simulator/mechanics.py`

## Out of Scope — Do Not Implement Without Explicit Re-Scoping
- **No memory reading and no packet inspection/manipulation.** These extract
  state the client holds but never displays, which is categorically
  different from reading the screen.
- **No reading opponent-hidden state** — specifically the opponent's elixir
  count and hand, which a player cannot see. (Inferring likely opponent
  elixir from *observed* plays is fine; reading the true value is not.)
- **No automated matchmaking.** The bridge joins an already-running match
  started by the user; it does not queue, accept, or restart games.

Note: these are *information-source* limits, not capability limits. Nothing
here restricts how strong the agent may become — see "Reaching Human-Parity
Ceiling" for the techniques that close the gap legitimately, including
inferring opponent elixir rather than reading it.

## ML Approach
Two-stage training pipeline (see design discussion in project history):
1. **Imitation bootstrap**: scripted bots (rusher/control/siege archetypes)
   generate rollouts; behavior cloning (+ optional DAgger) gives the policy a
   non-random starting point.
2. **Self-play PPO with a league**: PPO (clipped surrogate + GAE), sampling
   opponents from a pool of past checkpoints (not just the latest) to avoid
   strategy cycling. Periodic "exploiter" agents trained against the current
   main policy help surface blind spots.

Reward shaping decays over training: dense signals early (elixir trade
efficiency, tower HP delta), fading toward pure win/loss as the agent matures.

## Architecture Notes
- **Observation**: split spatial (arena grid: troop/tower positions, HP, type —
  fed through a small CNN) from non-spatial (hand, elixir, next card — small
  embedding + MLP), fused before the policy/value heads.
- **Action space**: factored/autoregressive — sample card choice, then condition
  placement on that choice. Action masking is required (elixir legality, valid
  placement zones) — do not train on an unmasked flat action space.
- **Vectorized envs**: the simulator's `step()`/`reset()` should support batched
  rollouts from the start; matches are short, so parallelism is where sample
  efficiency comes from.

## Project Structure
- `src/simulator/` — battle engine, card/tower stats, match rules (config-driven
  via `configs/*.yaml`, not hardcoded, so balance changes don't require code edits)
- `src/bots/` — scripted baseline strategies
- `src/agent/` — RL training code (BC pretraining + PPO self-play + league management)
- `src/eval/` — fixed benchmark suite, regression checks, training metrics/plots
- `tests/` — unit tests for elixir regen, targeting, damage calc, win-condition
  edge cases (simultaneous tower destruction, overtime)

## Evaluation Discipline
- Self-play win rate alone is not a valid progress signal (trends to ~50% by
  construction). Track win rate against the frozen scripted-bot benchmark suite
  over time as the real metric.
- Track card-usage entropy / strategy diversity to catch mode collapse.
- Never delete the original benchmark bots even after the agent surpasses them —
  they're the permanent regression tripwire.