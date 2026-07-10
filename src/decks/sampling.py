"""Deck sampling for training episodes."""
from __future__ import annotations

import numpy as np

from src.decks.catalog import DeckCatalog
from src.simulator.cards import CardStats


def sample_deck_name(
    catalog: DeckCatalog,
    pool_name: str,
    rng: np.random.Generator,
    weights: dict[str, float] | None = None,
) -> str:
    names = catalog.pool(pool_name)
    if not names:
        raise ValueError(f"empty deck pool: {pool_name}")
    if weights:
        w = np.array([max(weights.get(n, 1.0), 0.01) for n in names], dtype=float)
        w /= w.sum()
        idx = int(rng.choice(len(names), p=w))
        return names[idx]
    return str(rng.choice(names))


def sample_deck(
    catalog: DeckCatalog,
    pool_name: str,
    rng: np.random.Generator,
    weights: dict[str, float] | None = None,
) -> tuple[str, list[CardStats]]:
    name = sample_deck_name(catalog, pool_name, rng, weights=weights)
    return name, catalog.resolve(name)


def sample_match_decks(
    catalog: DeckCatalog,
    agent_pool: str,
    opponent_pool: str,
    rng: np.random.Generator,
    opponent_weights: dict[str, float] | None = None,
    agent_weights: dict[str, float] | None = None,
) -> tuple[tuple[str, list[CardStats]], tuple[str, list[CardStats]]]:
    agent = sample_deck(catalog, agent_pool, rng, weights=agent_weights)
    opponent = sample_deck(catalog, opponent_pool, rng, weights=opponent_weights)
    return agent, opponent
