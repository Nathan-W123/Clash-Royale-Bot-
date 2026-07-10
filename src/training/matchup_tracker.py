"""Track agent performance vs opponent decks for weakness-weighted sampling."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchupTracker:
    """Records wins/losses per opponent deck name."""

    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)

    def record(self, opponent_deck_name: str, won: bool) -> None:
        if won:
            self.wins[opponent_deck_name] = self.wins.get(opponent_deck_name, 0) + 1
        else:
            self.losses[opponent_deck_name] = self.losses.get(opponent_deck_name, 0) + 1

    def loss_rate(self, deck_name: str) -> float:
        w = self.wins.get(deck_name, 0)
        l = self.losses.get(deck_name, 0)
        total = w + l
        if total == 0:
            return 0.5
        return l / total

    def sampling_weights(self, deck_names: list[str], weakness_weight: float) -> dict[str, float]:
        """Higher weight for decks the agent loses to more often."""
        weights: dict[str, float] = {}
        for name in deck_names:
            lr = self.loss_rate(name)
            weights[name] = 1.0 + weakness_weight * lr
        return weights
