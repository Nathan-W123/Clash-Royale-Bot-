"""Dense reward shaping from engine events + terminal outcome.

Two shaping modes, selected by ``reward.mode`` in the training yaml:

``events`` (legacy)
    Sums per-event bonuses (tower damage dealt, trades won, elixir leaked).
    Simple and effective early, but **not** policy-invariant: it adds return
    to behaviours that are merely correlated with winning, so the optimal
    policy under it differs from the optimal policy under the true objective.
    That is why it has to anneal to zero — and annealing throws the signal
    away exactly when the agent is finally good enough to use it.

``potential`` (preferred)
    Potential-based shaping, ``F(s, s') = gamma * phi(s') - phi(s)``
    (Ng, Harada & Russell 1999). Telescoping means the total shaped return
    over an episode collapses to ``gamma^T * phi(s_T) - phi(s_0)``, a
    constant offset independent of the path taken, so the optimal policy is
    provably unchanged. It can therefore stay on permanently at full weight
    instead of being annealed out.

The potential used here is the normalized tower-HP differential: a state is
"good" in proportion to how far ahead the agent is on towers. Progress toward
winning yields positive shaping automatically, without ever paying the agent
for an action that does not actually improve its position.

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


MODE_EVENTS = "events"
MODE_POTENTIAL = "potential"


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
    mode: str = MODE_EVENTS
    # Scale of the potential function. Multiplies the tower-HP differential,
    # which is already in [-1, 1], so this is directly comparable to
    # `terminal`. Keep it well below `terminal`: shaping should bias the
    # search, not out-vote actually winning.
    potential_scale: float = 0.5
    # gamma must match the PPO discount, or the telescoping identity does not
    # hold and the invariance guarantee is lost.
    gamma: float = 0.997

    @property
    def is_potential(self) -> bool:
        return self.mode == MODE_POTENTIAL


def load_reward_config(path: Path | None = None) -> RewardConfig:
    path = path or CONFIG_DIR / "training.yaml"
    doc = yaml.safe_load(path.read_text())
    raw = doc.get("reward", {})
    # Default the shaping discount to the PPO discount in the same file:
    # potential-based invariance only holds when the two agree, and silently
    # defaulting to a different value would break the guarantee invisibly.
    ppo_gamma = float(doc.get("ppo", {}).get("gamma", 0.997))
    mode = str(raw.get("mode", MODE_EVENTS)).lower()
    if mode not in (MODE_EVENTS, MODE_POTENTIAL):
        raise ValueError(
            f"unknown reward.mode {mode!r}; expected {MODE_EVENTS!r} or {MODE_POTENTIAL!r}")
    return RewardConfig(
        terminal=float(raw.get("terminal", 1.0)),
        tower_damage=float(raw.get("tower_damage", 0.3)),
        tower_fall=float(raw.get("tower_fall", 0.15)),
        elixir_trade=float(raw.get("elixir_trade", 0.02)),
        leak=float(raw.get("leak", 0.005)),
        anneal_start=float(raw.get("anneal_start", 1.0)),
        anneal_end=float(raw.get("anneal_end", 0.0)),
        anneal_over_steps=int(raw.get("anneal_over_steps", 2_000_000)),
        mode=str(raw.get("mode", MODE_EVENTS)).lower(),
        potential_scale=float(raw.get("potential_scale", 0.5)),
        gamma=float(raw.get("gamma", ppo_gamma)),
    )


def anneal_factor(global_step: int, config: RewardConfig) -> float:
    """Linear blend weight for dense shaping (1.0 early -> 0.0 late)."""
    if config.anneal_over_steps <= 0:
        return config.anneal_end
    t = min(1.0, max(0.0, global_step / config.anneal_over_steps))
    return config.anneal_start + t * (config.anneal_end - config.anneal_start)


def tower_potential(engine, side: Side, config: RewardConfig) -> float:
    """Phi(s): normalized tower-HP advantage, in [-potential_scale, +scale].

    Uses *remaining tower HP* rather than crowns because HP is continuous:
    chipping a tower from 100% to 40% is real progress that a crown count
    cannot express, and a potential that only moves on tower kills would give
    almost no gradient through the part of the match where positioning
    actually decides things.

    Destroyed towers contribute zero, so this stays well-defined once a tower
    falls, and the king tower is included — losing it is the whole game.
    """
    def side_hp(s: Side) -> float:
        return sum(max(0.0, t.hp) for t in engine.towers if t.side == s)

    mine, theirs = side_hp(side), side_hp(side.other)
    total = mine + theirs
    if total <= 0:
        return 0.0
    return config.potential_scale * (mine - theirs) / total


def potential_shaping(
    prev_potential: float,
    curr_potential: float,
    config: RewardConfig,
    terminated: bool = False,
) -> float:
    """``F = gamma * phi(s') - phi(s)``.

    On termination the successor is absorbing and its potential is defined to
    be 0. Using phi(s_terminal) instead would leak a state-dependent bonus
    into the terminal transition and break the invariance the whole scheme
    exists to provide.
    """
    next_potential = 0.0 if terminated else curr_potential
    return config.gamma * next_potential - prev_potential


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
