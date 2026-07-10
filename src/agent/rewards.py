"""Dense reward shaping from engine events + terminal outcome.

Pure functions — no torch dependency. Weights load from ``configs/training.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.simulator.cards import CONFIG_DIR
from src.simulator.constants import MatchResult, Side

# Princess tower HP scale used to normalize damage differentials.
TOWER_HP_SCALE = 1400.0


@dataclass(frozen=True)
class RewardConfig:
    terminal: float = 1.0
    tower_damage: float = 0.3
    tower_fall: float = 0.15
    elixir_trade: float = 0.02
    leak: float = 0.005
    anneal_start: float = 1.0
    anneal_end: float = 0.0
    anneal_over_steps: int = 2_000_000


def load_reward_config(path: Path | None = None) -> RewardConfig:
    path = path or CONFIG_DIR / "training.yaml"
    raw = yaml.safe_load(path.read_text()).get("reward", {})
    return RewardConfig(
        terminal=float(raw.get("terminal", 1.0)),
        tower_damage=float(raw.get("tower_damage", 0.3)),
        tower_fall=float(raw.get("tower_fall", 0.15)),
        elixir_trade=float(raw.get("elixir_trade", 0.02)),
        leak=float(raw.get("leak", 0.005)),
        anneal_start=float(raw.get("anneal_start", 1.0)),
        anneal_end=float(raw.get("anneal_end", 0.0)),
        anneal_over_steps=int(raw.get("anneal_over_steps", 2_000_000)),
    )


def anneal_factor(global_step: int, config: RewardConfig) -> float:
    """Linear blend weight for dense shaping (1.0 early -> 0.0 late)."""
    if config.anneal_over_steps <= 0:
        return config.anneal_end
    t = min(1.0, max(0.0, global_step / config.anneal_over_steps))
    return config.anneal_start + t * (config.anneal_end - config.anneal_start)


def terminal_reward(result: MatchResult, side: Side, config: RewardConfig) -> float:
    if result in (MatchResult.ONGOING, MatchResult.DRAW):
        return 0.0
    sign = 1.0 if result == MatchResult.win_for(side) else -1.0
    return sign * config.terminal


def compute_step_reward(
    events: list[dict[str, Any]],
    side: Side,
    prev_state: dict[str, Any] | None,
    curr_state: dict[str, Any] | None,
    config: RewardConfig,
) -> float:
    """Dense per-step reward from tick events (perspective of ``side``).

    ``prev_state`` / ``curr_state`` are optional snapshots (e.g. tower HP maps)
    for callers that also log state; event payloads drive the reward.
    """
    del prev_state, curr_state  # reserved for future state-delta shaping
    enemy = side.other
    reward = 0.0

    for ev in events:
        et = ev["type"]
        ev_side = ev["side"]
        if et == "tower_damage":
            sign = 1.0 if ev_side == enemy else -1.0
            reward += sign * config.tower_damage * ev["amount"] / TOWER_HP_SCALE
        elif et == "tower_fall":
            sign = 1.0 if ev_side == enemy else -1.0
            reward += sign * config.tower_fall
        elif et == "death":
            sign = 1.0 if ev_side == enemy else -1.0
            reward += sign * config.elixir_trade * ev["value"]
        elif et == "leak":
            sign = -1.0 if ev_side == side else 1.0
            reward += sign * config.leak * ev["amount"]

    return reward


class RewardShaper:
    """Loads reward weights and blends dense shaping with terminal outcome."""

    def __init__(self, config: RewardConfig | None = None, path: Path | None = None):
        self.config = config or load_reward_config(path)

    def anneal_factor(self, global_step: int) -> float:
        return anneal_factor(global_step, self.config)

    def terminal_reward(self, result: MatchResult, side: Side) -> float:
        return terminal_reward(result, side, self.config)

    def compute_step_reward(
        self,
        events: list[dict[str, Any]],
        side: Side,
        prev_state: dict[str, Any] | None = None,
        curr_state: dict[str, Any] | None = None,
    ) -> float:
        return compute_step_reward(events, side, prev_state, curr_state, self.config)

    def shaped_step_reward(
        self,
        events: list[dict[str, Any]],
        side: Side,
        global_step: int,
        prev_state: dict[str, Any] | None = None,
        curr_state: dict[str, Any] | None = None,
    ) -> float:
        """Dense reward scaled by the anneal schedule."""
        return self.anneal_factor(global_step) * self.compute_step_reward(
            events, side, prev_state, curr_state
        )

    def episode_reward(
        self,
        events: list[dict[str, Any]],
        side: Side,
        result: MatchResult,
        global_step: int,
        prev_state: dict[str, Any] | None = None,
        curr_state: dict[str, Any] | None = None,
    ) -> float:
        """Dense (annealed) + terminal on the final transition."""
        dense = self.shaped_step_reward(events, side, global_step, prev_state, curr_state)
        if result == MatchResult.ONGOING:
            return dense
        return dense + self.terminal_reward(result, side)
