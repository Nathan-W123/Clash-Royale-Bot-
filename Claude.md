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

## Out of Scope — Do Not Implement Without Explicit Re-Scoping
- **No real Clash Royale client integration** (no screen capture, OCR, memory
  reading, packet manipulation, or input injection into the actual game/emulator).
- Keep simulator interfaces generic enough that a future driver *could*
  theoretically be swapped in, but do not build one preemptively.

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