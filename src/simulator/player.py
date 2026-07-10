"""Per-player state: elixir economy and the 4-card hand cycle."""
from __future__ import annotations

import numpy as np

from src.simulator.cards import CardStats
from src.simulator.constants import HAND_SIZE


class PlayerState:
    """Elixir regen/cap/leak plus the deck cycle.

    The hand holds 4 cards; the next card is visible. A played card goes to
    the back of the cycle queue and its slot is refilled from the front.
    """

    def __init__(
        self,
        deck: list[CardStats],
        rng: np.random.Generator,
        elixir_start: float,
        elixir_max: float,
        regen_interval: float,
    ):
        if len(deck) < HAND_SIZE + 1:
            raise ValueError(f"deck needs at least {HAND_SIZE + 1} cards, got {len(deck)}")
        order = list(rng.permutation(len(deck)))
        self._deck = [deck[i] for i in order]
        self.hand: list[CardStats] = self._deck[:HAND_SIZE]
        self._queue: list[CardStats] = self._deck[HAND_SIZE:]

        self.elixir = elixir_start
        self.elixir_max = elixir_max
        self.regen_interval = regen_interval
        self.leaked = 0.0        # cumulative elixir lost to regen-at-cap
        self.spent = 0.0         # cumulative elixir spent on cards

    @property
    def next_card(self) -> CardStats:
        return self._queue[0]

    def regen(self, dt: float, double: bool) -> float:
        """Advance elixir regen by dt seconds. Returns elixir leaked this tick."""
        rate = (2.0 if double else 1.0) / self.regen_interval
        gained = rate * dt
        new = self.elixir + gained
        leak = max(0.0, new - self.elixir_max)
        self.elixir = min(new, self.elixir_max)
        self.leaked += leak
        return leak

    def can_afford(self, slot: int) -> bool:
        return self.elixir >= self.hand[slot].cost

    def play(self, slot: int) -> CardStats:
        card = self.hand[slot]
        if self.elixir < card.cost:
            raise ValueError(f"cannot afford {card.name}: {self.elixir:.2f} < {card.cost}")
        self.elixir -= card.cost
        self.spent += card.cost
        self.hand[slot] = self._queue.pop(0)
        self._queue.append(card)
        return card
