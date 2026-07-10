"""Match outcome records and aggregate statistics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WinLossRecord:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        if decided == 0:
            return 0.0
        return self.wins / decided

    @property
    def wl_ratio(self) -> float:
        if self.losses == 0:
            return float(self.wins) if self.wins else 0.0
        return self.wins / self.losses

    def with_result(self, won: bool | None) -> WinLossRecord:
        if won is True:
            return WinLossRecord(self.wins + 1, self.losses, self.draws)
        if won is False:
            return WinLossRecord(self.wins, self.losses + 1, self.draws)
        return WinLossRecord(self.wins, self.losses, self.draws + 1)


@dataclass
class MatchRecord:
    """One finished game from the agent's perspective."""

    won: bool | None
    agent_crowns: int
    opponent_crowns: int
    agent_deck: str
    opponent_deck: str
    opponent_bot: str
    opponent_kind: str = "scripted"
    stage: str = ""
    duration_sec: float = 0.0
    cards_played: dict[str, int] = field(default_factory=dict)
    elixir_spent: float = 0.0
    elixir_leaked: float = 0.0
    training_step: int | None = None


def card_usage_entropy(counts: dict[str, int]) -> float:
    """Shannon entropy of card play distribution (nats). Higher = more diverse."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for n in counts.values():
        p = n / total
        entropy -= p * math.log(p)
    return entropy


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for name, n in source.items():
        target[name] = target.get(name, 0) + n
