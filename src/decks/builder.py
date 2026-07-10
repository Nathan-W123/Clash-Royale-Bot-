"""Adaptive agent deck builder from per-card performance scores."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.decks.catalog import DECK_SIZE, DeckCatalog
from src.simulator.cards import CONFIG_DIR, CardStats

_CATEGORY_ORDER = (
    "win_condition",
    "spell",
    "building",
    "cycle",
    "support",
    "swarm",
    "air",
    "heavy",
)


@dataclass
class CardPerformance:
    plays: int = 0
    wins: int = 0
    crowns: float = 0.0
    elixir_spent: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.plays == 0:
            return 0.5
        return self.wins / self.plays

    @property
    def crowns_per_play(self) -> float:
        if self.plays == 0:
            return 0.0
        return self.crowns / self.plays

    @property
    def elixir_efficiency(self) -> float:
        if self.elixir_spent <= 0:
            return 0.0
        return self.crowns / self.elixir_spent


class CardScoreTracker:
    """Track per-card outcomes when the card appears in match play stats."""

    def __init__(self):
        self._stats: dict[str, CardPerformance] = {}

    def record(
        self,
        cards_played: dict[str, int],
        *,
        won: bool,
        crowns: int = 0,
        elixir_spent: float = 0.0,
    ) -> None:
        if not cards_played:
            return
        total_plays = sum(cards_played.values())
        crown_share = crowns / total_plays if total_plays else 0.0
        elixir_share = elixir_spent / total_plays if total_plays else 0.0
        for card, count in cards_played.items():
            perf = self._stats.setdefault(card, CardPerformance())
            perf.plays += count
            if won:
                perf.wins += count
            perf.crowns += crown_share * count
            perf.elixir_spent += elixir_share * count

    def composite_score(self, card_name: str) -> float:
        perf = self._stats.get(card_name)
        if perf is None or perf.plays == 0:
            return 0.5
        return (
            0.55 * perf.win_rate
            + 0.30 * min(perf.crowns_per_play / 2.0, 1.0)
            + 0.15 * min(perf.elixir_efficiency * 3.0, 1.0)
        )

    def to_dict(self) -> dict[str, dict]:
        return {
            name: {
                "plays": p.plays,
                "wins": p.wins,
                "win_rate": p.win_rate,
                "crowns": p.crowns,
                "elixir_spent": p.elixir_spent,
                "composite_score": self.composite_score(name),
            }
            for name, p in self._stats.items()
        }


def load_card_categories(path: Path | None = None) -> dict[str, list[str]]:
    path = path or CONFIG_DIR / "card_categories.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    return {k: list(v) for k, v in raw.get("categories", {}).items()}


def _load_assembly_order(path: Path | None = None) -> list[str]:
    path = path or CONFIG_DIR / "card_categories.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    return list(raw.get("assembly_order", _CATEGORY_ORDER))


@dataclass
class AdaptiveDeckBuilderConfig:
    rebuild_every_matches: int = 25
    plateau_window: int = 20
    plateau_threshold: float = 0.02
    seed_deck: list[str] = field(
        default_factory=lambda: [
            "hog_rider",
            "fireball",
            "ice_spirit",
            "skeletons",
            "ice_golem",
            "musketeer",
            "log",
            "cannon",
        ]
    )


class AdaptiveDeckBuilder:
    """Assemble an 8-card deck from category winners and composite scores."""

    def __init__(
        self,
        catalog: DeckCatalog,
        config: AdaptiveDeckBuilderConfig | None = None,
        categories: dict[str, list[str]] | None = None,
    ):
        self.catalog = catalog
        self.config = config or AdaptiveDeckBuilderConfig()
        self.categories = categories or load_card_categories()
        self.assembly_order = _load_assembly_order()
        self.tracker = CardScoreTracker()
        self._current_names = list(self.config.seed_deck)
        self._matches_since_rebuild = 0
        self._recent_win_rates: list[float] = []

    @property
    def current_deck_name(self) -> str:
        return "adaptive_agent"

    def current_deck(self) -> list[CardStats]:
        return self.catalog.validate_names(self._current_names)

    def record_match(
        self,
        cards_played: dict[str, int],
        *,
        won: bool,
        crowns: int = 0,
        elixir_spent: float = 0.0,
    ) -> None:
        self.tracker.record(
            cards_played, won=won, crowns=crowns, elixir_spent=elixir_spent
        )
        self._matches_since_rebuild += 1
        self._recent_win_rates.append(1.0 if won else 0.0)
        window = self.config.plateau_window
        if len(self._recent_win_rates) > window:
            self._recent_win_rates = self._recent_win_rates[-window:]
        if self._should_rebuild():
            self.rebuild()

    def _should_rebuild(self) -> bool:
        if self._matches_since_rebuild >= self.config.rebuild_every_matches:
            return True
        window = self.config.plateau_window
        if len(self._recent_win_rates) < window:
            return False
        first_half = self._recent_win_rates[: window // 2]
        second_half = self._recent_win_rates[window // 2 :]
        if not first_half or not second_half:
            return False
        delta = abs(sum(second_half) / len(second_half) - sum(first_half) / len(first_half))
        return delta < self.config.plateau_threshold

    def rebuild(self) -> list[str]:
        self._current_names = self.build_deck(self.tracker, self.catalog)
        self._matches_since_rebuild = 0
        return list(self._current_names)

    def build_deck(
        self,
        scores: CardScoreTracker,
        catalog: DeckCatalog,
    ) -> list[str]:
        chosen: list[str] = []
        has_hero = False
        has_evo = False

        def can_add(name: str) -> bool:
            if name not in catalog.cards or name in chosen:
                return False
            card = catalog.cards[name]
            if card.is_hero:
                if has_hero:
                    return False
            if card.is_evolution:
                if has_evo:
                    return False
                base = card.evolution_of
                if base and base in chosen:
                    return False
            return True

        def add_best(candidates: list[str]) -> None:
            nonlocal has_hero, has_evo
            ranked = sorted(
                [c for c in candidates if can_add(c)],
                key=lambda n: scores.composite_score(n),
                reverse=True,
            )
            if ranked:
                name = ranked[0]
                chosen.append(name)
                card = catalog.cards[name]
                if card.is_hero:
                    has_hero = True
                if card.is_evolution:
                    has_evo = True

        for category in self.assembly_order:
            if len(chosen) >= DECK_SIZE:
                break
            add_best(self.categories.get(category, []))

        remaining = sorted(
            catalog.cards.keys(),
            key=lambda n: scores.composite_score(n),
            reverse=True,
        )
        for name in remaining:
            if len(chosen) >= DECK_SIZE:
                break
            if can_add(name):
                chosen.append(name)
                card = catalog.cards[name]
                if card.is_hero:
                    has_hero = True
                if card.is_evolution:
                    has_evo = True

        if len(chosen) < DECK_SIZE:
            for name in self.config.seed_deck:
                if len(chosen) >= DECK_SIZE:
                    break
                if can_add(name):
                    chosen.append(name)

        if len(chosen) != DECK_SIZE:
            raise ValueError(f"could not assemble {DECK_SIZE}-card deck, got {len(chosen)}")
        return chosen
