"""Procedural deck generation and outer-loop deck search scaffolding."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.decks.catalog import DECK_SIZE, DeckCatalog
from src.simulator.cards import CardStats


def generate_from_template(
    catalog: DeckCatalog,
    template_name: str,
    rng: np.random.Generator,
) -> list[str]:
    if template_name not in catalog.templates:
        raise KeyError(f"unknown deck template: {template_name}")
    tpl = catalog.templates[template_name]
    chosen = list(tpl.required)
    pool = [c for c in tpl.fill_from if c not in chosen and c in catalog.cards]
    rng.shuffle(pool)
    for card in pool:
        if len(chosen) >= DECK_SIZE:
            break
        chosen.append(card)
    if len(chosen) < DECK_SIZE:
        extras = [n for n in catalog.cards if n not in chosen]
        rng.shuffle(extras)
        for card in extras:
            if len(chosen) >= DECK_SIZE:
                break
            chosen.append(card)
    if len(chosen) != DECK_SIZE:
        raise ValueError(f"template {template_name} could not fill {DECK_SIZE} cards")
    return chosen


def generate_deck(
    catalog: DeckCatalog,
    rng: np.random.Generator,
    template_name: str | None = None,
) -> list[CardStats]:
    if template_name:
        names = generate_from_template(catalog, template_name, rng)
    else:
        names = list(rng.choice(list(catalog.cards), size=DECK_SIZE, replace=False))
    return catalog.validate_names(names)


@dataclass
class DeckCandidate:
    names: list[str]
    score: float


class DeckSearch:
    """Evaluate and mutate decks against a benchmark suite.

    The evaluator receives a deck (CardStats list) and returns a score in [0, 1].
    Intended for post-training outer loops once a generalist policy exists; works
    today with bot-vs-bot proxies.
    """

    def __init__(
        self,
        catalog: DeckCatalog,
        evaluate: Callable[[list[CardStats]], float],
        rng: np.random.Generator | None = None,
    ):
        self.catalog = catalog
        self.evaluate = evaluate
        self.rng = rng or np.random.default_rng()

    def score_deck(self, names: list[str]) -> DeckCandidate:
        try:
            deck = self.catalog.validate_names(names)
        except ValueError:
            return DeckCandidate(names=list(names), score=float("-inf"))
        return DeckCandidate(names=list(names), score=self.evaluate(deck))

    def mutate(self, names: list[str], swaps: int = 1) -> list[str]:
        out = list(names)
        unused = [n for n in self.catalog.cards if n not in out]
        if not unused:
            return out
        for _ in range(swaps):
            if not unused:
                break
            i = int(self.rng.integers(len(out)))
            j = int(self.rng.integers(len(unused)))
            out[i] = unused.pop(j)
        return out

    def search(
        self,
        seed_names: list[str],
        iterations: int = 50,
        population: int = 8,
    ) -> DeckCandidate:
        seeds = [seed_names]
        for name in self.catalog.templates:
            try:
                seeds.append(generate_from_template(self.catalog, name, self.rng))
            except ValueError:
                continue
        candidates = [self.score_deck(s) for s in seeds[:population]]
        best = max(candidates, key=lambda c: c.score)
        for _ in range(iterations):
            trial = self.mutate(best.names, swaps=1)
            cand = self.score_deck(trial)
            if cand.score >= best.score:
                best = cand
        return best
