"""Deck catalog: named decks, pools, and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.simulator.cards import CONFIG_DIR, CardStats, load_cards, load_decks, ladder_deck_names

DECK_SIZE = 8


@dataclass(frozen=True)
class DeckTemplate:
    name: str
    required: tuple[str, ...]
    fill_from: tuple[str, ...]


class DeckCatalog:
    """Loads cards, named decks, pools, and generation templates from config."""

    def __init__(
        self,
        cards: dict[str, CardStats] | None = None,
        decks: dict[str, list[str]] | None = None,
        pools: dict[str, list[str]] | None = None,
        templates: dict[str, DeckTemplate] | None = None,
    ):
        self.cards = cards or load_cards()
        self.decks = decks or load_decks()
        self.pools = pools or _load_pools()
        self.templates = templates or _load_templates()

    def resolve(self, deck_name: str) -> list[CardStats]:
        if deck_name not in self.decks:
            raise KeyError(f"unknown deck: {deck_name}")
        return self.validate_names(self.decks[deck_name])

    def validate_names(self, names: list[str]) -> list[CardStats]:
        if len(names) != len(set(names)):
            raise ValueError("deck contains duplicate cards")
        if len(names) < 5:
            raise ValueError(f"deck needs at least 5 cards, got {len(names)}")
        missing = [n for n in names if n not in self.cards]
        if missing:
            raise ValueError(f"unknown cards: {missing}")
        heroes = [n for n in names if self.cards[n].is_hero]
        if len(heroes) > 1:
            raise ValueError(f"deck can have at most one hero, got {heroes}")
        evos = [n for n in names if self.cards[n].is_evolution]
        if len(evos) > 1:
            raise ValueError(f"deck can have at most one evolution, got {evos}")
        for evo_name in evos:
            base = self.cards[evo_name].evolution_of
            if base in names:
                raise ValueError(f"deck cannot include both {base} and {evo_name}")
        return [self.cards[n] for n in names]

    def pool(self, pool_name: str) -> list[str]:
        if pool_name not in self.pools:
            raise KeyError(f"unknown deck pool: {pool_name}")
        return list(self.pools[pool_name])

    def archetype_for_deck(self, deck_name: str) -> str:
        """Best-effort archetype label for a named deck."""
        if deck_name in self.templates:
            return deck_name
        if deck_name in {"rusher", "control", "siege", "beatdown"}:
            return deck_name
        ladder_arch = _ladder_archetypes().get(deck_name)
        if ladder_arch:
            return ladder_arch
        return "generic"


def _ladder_archetypes(path: Path | None = None) -> dict[str, str]:
    path = path or CONFIG_DIR / "ladder_decks.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {
        spec["name"]: str(spec.get("archetype", "meta"))
        for spec in raw.get("decks", [])
    }


def _load_pools(path: Path | None = None) -> dict[str, list[str]]:
    path = path or CONFIG_DIR / "decks.yaml"
    raw = yaml.safe_load(path.read_text())
    pools = {name: list(cards) for name, cards in raw.get("pools", {}).items()}
    ladder = ladder_deck_names()
    if ladder:
        pools["ladder_top50"] = ladder
    return pools


def _load_templates(path: Path | None = None) -> dict[str, DeckTemplate]:
    path = path or CONFIG_DIR / "deck_templates.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {
        name: DeckTemplate(
            name=name,
            required=tuple(spec.get("required", [])),
            fill_from=tuple(spec.get("fill_from", [])),
        )
        for name, spec in raw.items()
    }
