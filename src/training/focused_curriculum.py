"""Focused rotation curriculum: beat one opponent deck at a time."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FocusedRotationConfig:
    wins_per_deck: int = 5
    target_win_rate: float = 0.65
    min_matches_per_deck: int = 10
    min_win_rate_per_deck: float = 0.55


class FocusedRotationManager:
    """Rotate through ladder decks; advance after X wins or min matches + WR."""

    def __init__(
        self,
        deck_names: list[str],
        config: FocusedRotationConfig | None = None,
    ):
        if not deck_names:
            raise ValueError("deck_names must not be empty")
        self.deck_names = list(deck_names)
        self.config = config or FocusedRotationConfig()
        self.current_index = 0
        self.wins_vs_current = 0
        self.matches_vs_current = 0
        self.total_wins = 0
        self.total_matches = 0
        self.cycle = 0
        self._completed_cycles = 0

    def current_opponent_deck(self) -> str:
        return self.deck_names[self.current_index]

    @property
    def current_deck_win_rate(self) -> float:
        if self.matches_vs_current == 0:
            return 0.0
        return self.wins_vs_current / self.matches_vs_current

    @property
    def overall_win_rate(self) -> float:
        if self.total_matches == 0:
            return 0.0
        return self.total_wins / self.total_matches

    def record_result(self, won: bool) -> None:
        self.total_matches += 1
        self.matches_vs_current += 1
        if won:
            self.total_wins += 1
            self.wins_vs_current += 1
        if self._should_advance():
            self._advance()

    def _should_advance(self) -> bool:
        cfg = self.config
        if self.wins_vs_current >= cfg.wins_per_deck:
            return True
        if self.matches_vs_current >= cfg.min_matches_per_deck:
            return self.current_deck_win_rate >= cfg.min_win_rate_per_deck
        return False

    def _advance(self) -> None:
        self.current_index += 1
        self.wins_vs_current = 0
        self.matches_vs_current = 0
        if self.current_index >= len(self.deck_names):
            self.current_index = 0
            self.cycle += 1
            self._completed_cycles += 1

    def should_continue_training(self) -> bool:
        """Stop when a full cycle completes at or above target overall WR."""
        if self._completed_cycles == 0:
            return True
        return self.overall_win_rate < self.config.target_win_rate

    def progress(self) -> dict:
        return {
            "current_deck": self.current_opponent_deck(),
            "deck_index": self.current_index,
            "deck_total": len(self.deck_names),
            "wins_vs_current": self.wins_vs_current,
            "matches_vs_current": self.matches_vs_current,
            "current_deck_win_rate": self.current_deck_win_rate,
            "overall_win_rate": self.overall_win_rate,
            "total_matches": self.total_matches,
            "cycle": self.cycle,
            "completed_cycles": self._completed_cycles,
        }
