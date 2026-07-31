"""Deterministic opponent elixir + cycle tracker (#38).

**Do not learn what you can compute.** The opponent's elixir is not a hidden
quantity to be estimated — it is arithmetic over things a player can see:
a known start value, a known regeneration rate that doubles on a known
clock, and the known cost of every card they have played. Likewise the
8-card cycle is fully determined once the deck reveals itself, so a tracker
that has watched the whole match knows their *actual* hand, not a guess.
Strong human players do exactly this. See CLAUDE.md, "On-Screen Visual
Perception".

**The invariant that keeps this in scope.** This module consumes only what
the perception pipeline reported: observed deploys, observed units, observed
spell effects, and the clock. It must never read
``engine.players[opponent].elixir`` or any other engine-side ground truth.
That is the whole difference between deriving and peeking — and it shows up
as a real behavioural difference, because a derived tracker is *wrong
exactly when perception was wrong* (a cast missed off-screen, a unit
misidentified), which is the same failure mode a human has. A memory read
never is. `tests/test_opponent_tracker.py` pins this down by feeding a
partial history and asserting the tracker's output diverges from engine
truth.

**Self-correction** is what makes it robust rather than brittle. Vision sees
*units*, not only deploy animations, so a unit that appears without a
matching observed deploy is evidence of a play that was missed: the tracker
infers it retroactively, subtracts the cost, and raises `uncertainty`. A
missed spell is caught the same way when its effect lands.

Sim ground truth may be used *only* to measure tracker accuracy as an
auxiliary training signal — never as an input.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from src.simulator.cards import ArenaConfig, CardStats

HAND_SIZE = 4
DECK_SIZE = 8

# An Elixir Collector's lifetime payout, and how often it pays. Real CR pays
# out in discrete lumps rather than continuously, and the lump timing is what
# makes a collector-punish window exist at all.
COLLECTOR_INTERVAL = 8.5
COLLECTOR_PAYOUTS = 8
COLLECTOR_CARDS = frozenset({"elixir_collector", "elixir_pump"})

# Cards that gift elixir to the *enemy* when they die. If we play one, the
# opponent's count goes up — the classic way a hand-rolled counter silently
# drifts low for the rest of the match.
ELIXIR_GIFT_ON_DEATH = {"elixir_golem": 1.0, "elixir_golemite": 0.5}

# Mirror costs "the mirrored card + 1", so it has no fixed cost of its own.
MIRROR_CARDS = frozenset({"mirror"})


@dataclass(frozen=True)
class TrackerConfig:
    elixir_start: float
    elixir_max: float
    regen_interval: float
    double_time: float
    regulation: float

    @classmethod
    def from_arena(cls, arena: ArenaConfig) -> "TrackerConfig":
        return cls(
            elixir_start=arena.elixir_start,
            elixir_max=arena.elixir_max,
            regen_interval=arena.elixir_regen_interval,
            double_time=arena.double_time,
            regulation=arena.regulation,
        )


@dataclass
class _Collector:
    deployed_at: float
    paid: int = 0


@dataclass
class TrackerState:
    """Everything the tracker believes, as plain data (easy to log/compare)."""

    elixir: float
    uncertainty: int
    leaked: float
    hand: tuple[str | None, ...]
    queue: tuple[str | None, ...]
    deck: tuple[str, ...]
    inferred_plays: tuple[str, ...] = field(default=())


class OpponentTracker:
    """Derives the opponent's elixir and cycle from observations alone.

    Feed it, in whatever order they are perceived:
      - `advance(now)`               the match clock (drives regen)
      - `observe_play(card, ...)`    a deploy you actually saw
      - `observe_unit(card, key)`    a unit on the arena, keyed by detection id
      - `observe_spell(card)`        a spell effect landing
      - `observe_elixir_gift(n)`     elixir handed to them (our Elixir Golem)

    Read back `elixir`, `elixir_range`, `hand`, `next_card`, `uncertainty`.
    """

    def __init__(
        self,
        config: TrackerConfig,
        cards: Mapping[str, CardStats],
        deck: Iterable[str] | None = None,
    ):
        self.config = config
        self.cards = cards
        self.elixir = config.elixir_start
        self.leaked = 0.0
        self.time = 0.0
        self.uncertainty = 0
        self.inferred_plays: list[str] = []

        # Cycle model, mirroring `PlayerState`: 4-card hand, 4-card queue,
        # a played card goes to the back and its slot refills from the front.
        # `None` marks a slot whose card has not been revealed yet.
        known = list(deck) if deck else []
        self.hand: list[str | None] = (known[:HAND_SIZE] + [None] * HAND_SIZE)[:HAND_SIZE]
        self.queue: list[str | None] = (known[HAND_SIZE:] + [None] * HAND_SIZE)[:HAND_SIZE]
        self.deck: list[str] = list(dict.fromkeys(known))

        self._collectors: list[_Collector] = []
        self._seen_units: set = set()
        # Units we expect to see because we watched the deploy that spawns
        # them; consumed by `observe_unit` so a seen deploy is not
        # double-charged when its troops render a frame later.
        self._pending_spawns: Counter[str] = Counter()
        self._last_play_cost: float = 0.0

    # ------------------------------------------------------------- read-out

    @property
    def double_elixir(self) -> bool:
        return self.time >= self.config.double_time or self.time >= self.config.regulation

    @property
    def next_card(self) -> str | None:
        return self.queue[0]

    @property
    def deck_known(self) -> bool:
        return len(self.deck) >= DECK_SIZE

    @property
    def elixir_range(self) -> tuple[float, float]:
        """Confidence band around the point estimate.

        Each retroactive inference means one play was perceived late or not
        at all, so the point value is exact only if nothing was missed. The
        band widens by the cheapest possible unseen play per inference — a
        deliberately loose bound, since the alternative is a downstream
        consumer treating a drifting number as exact.
        """
        margin = float(self.uncertainty)
        return (max(0.0, self.elixir - margin),
                min(self.config.elixir_max, self.elixir + margin))

    def state(self) -> TrackerState:
        return TrackerState(
            elixir=self.elixir,
            uncertainty=self.uncertainty,
            leaked=self.leaked,
            hand=tuple(self.hand),
            queue=tuple(self.queue),
            deck=tuple(self.deck),
            inferred_plays=tuple(self.inferred_plays),
        )

    def cost_of(self, card: str) -> float:
        """Elixir cost of one play of `card`.

        Mirror has no cost of its own — it costs whatever was mirrored, plus
        one — so it is resolved against the previous play.
        """
        if card in MIRROR_CARDS:
            return self._last_play_cost + 1.0
        stats = self.cards.get(card)
        return float(stats.cost) if stats is not None else 0.0

    # --------------------------------------------------------------- clock

    def advance(self, now: float) -> None:
        """Regenerate elixir up to absolute match time `now`.

        Handles the double-elixir boundary exactly rather than picking one
        rate for the whole interval, so a long step that straddles the
        transition does not silently lose (or invent) elixir.
        """
        if now <= self.time:
            self.time = max(self.time, now)
            return
        boundary = min(self.config.double_time, self.config.regulation)
        segments: list[tuple[float, bool]] = []
        start = self.time
        if start < boundary < now:
            segments.append((boundary - start, False))
            segments.append((now - boundary, True))
        else:
            segments.append((now - start, self.double_elixir or start >= boundary))
        for span, double in segments:
            rate = (2.0 if double else 1.0) / self.config.regen_interval
            self._gain(rate * span)
        self.time = now
        self._pay_collectors()

    def _gain(self, amount: float) -> None:
        """Add elixir, discarding whatever spills past the cap."""
        if amount <= 0:
            return
        new = self.elixir + amount
        overflow = max(0.0, new - self.config.elixir_max)
        self.elixir = min(new, self.config.elixir_max)
        self.leaked += overflow

    def _pay_collectors(self) -> None:
        for c in self._collectors:
            while (c.paid < COLLECTOR_PAYOUTS
                   and self.time >= c.deployed_at + COLLECTOR_INTERVAL * (c.paid + 1)):
                c.paid += 1
                self._gain(1.0)
        self._collectors = [c for c in self._collectors if c.paid < COLLECTOR_PAYOUTS]

    # -------------------------------------------------------- observations

    def observe_play(self, card: str, at: float | None = None,
                     inferred: bool = False) -> float:
        """Record a play we saw (or inferred). Returns the elixir charged."""
        if at is not None:
            self.advance(at)
        cost = self.cost_of(card)
        # Elixir cannot go negative: if the count says they could not have
        # afforded this, our count was too low, and the play is the ground
        # truth — clamp and flag rather than carrying a negative forward.
        if cost > self.elixir + 1e-6:
            self.uncertainty += 1
            self.elixir = 0.0
        else:
            self.elixir -= cost
        self._last_play_cost = cost
        self._advance_cycle(card)
        if card in COLLECTOR_CARDS:
            self._collectors.append(_Collector(deployed_at=self.time))
        stats = self.cards.get(card)
        if stats is not None and not inferred:
            self._pending_spawns[card] += max(1, stats.count)
        if inferred:
            self.uncertainty += 1
            self.inferred_plays.append(card)
        return cost

    def observe_unit(self, card: str, key) -> bool:
        """Report a unit currently on the arena, keyed by a stable detection id.

        Returns True when this sighting forced a retroactive play inference —
        i.e. the unit exists but we never saw it deployed, which is exactly
        the self-correction that keeps the count from drifting after a missed
        frame. Repeat sightings of the same `key` are free.
        """
        if key in self._seen_units:
            return False
        self._seen_units.add(key)
        if self._pending_spawns[card] > 0:
            self._pending_spawns[card] -= 1
            return False
        self.observe_play(card, inferred=True)
        # A swarm deploy produces several units from one play; credit the
        # rest so the remaining sightings do not each infer another play.
        stats = self.cards.get(card)
        if stats is not None and stats.count > 1:
            self._pending_spawns[card] += stats.count - 1
        return True

    def observe_spell(self, card: str, key=None) -> bool:
        """A spell effect landed. Same retroactive path as `observe_unit`."""
        return self.observe_unit(card, key if key is not None else ("spell", card, self.time))

    def observe_elixir_gift(self, amount: float) -> None:
        """Elixir handed *to* the opponent by something we did — our Elixir
        Golem dying is the case that exists in this roster."""
        self._gain(amount)

    def observe_our_unit_death(self, card: str) -> None:
        """Convenience wrapper: charge the enemy-gift table for our unit."""
        gift = ELIXIR_GIFT_ON_DEATH.get(card)
        if gift:
            self.observe_elixir_gift(gift)

    # ---------------------------------------------------------------- cycle

    def _advance_cycle(self, card: str) -> None:
        """Play `card`: it leaves the hand, the queue front refills the slot,
        and the card goes to the back of the queue."""
        if card not in self.deck and len(self.deck) < DECK_SIZE:
            self.deck.append(card)
        try:
            slot = self.hand.index(card)
        except ValueError:
            # Not yet known to be in hand: it must have occupied one of the
            # unresolved slots, so bind it there.
            slot = self.hand.index(None) if None in self.hand else 0
        self.hand[slot] = self.queue.pop(0)
        self.queue.append(card)

    def possible_hand(self) -> list[str]:
        """Cards we know are in hand right now (unresolved slots omitted)."""
        return [c for c in self.hand if c is not None]

    def candidate_cards(self) -> list[str]:
        """Cards revealed so far — the candidate set a vision classifier
        should be constrained to (#35). Empty means "no prior yet"."""
        return list(self.deck)


def track_from_events(
    tracker: OpponentTracker,
    events: Iterable[dict],
    now: float,
    opponent,
    drop: set[str] | None = None,
) -> None:
    """Feed simulator *events* through the tracker as if they were perceived.

    This is the sim-side harness for measuring tracker accuracy, and for
    building the auxiliary training signal. `drop` names cards whose deploy
    events are withheld, which is how a missed-perception scenario is
    constructed in tests — the resulting divergence from engine truth is the
    point, not a bug.
    """
    drop = drop or set()
    for ev in events:
        if ev.get("type") != "deploy" or ev.get("side") != opponent:
            continue
        if ev["card"] in drop:
            continue
        tracker.observe_play(ev["card"], at=now)
    tracker.advance(now)
