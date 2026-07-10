"""Tests for ladder meta deck loading and validation."""
from __future__ import annotations

import yaml

from src.decks.catalog import DeckCatalog
from src.simulator.cards import CONFIG_DIR, load_cards, load_ladder_decks


def test_ladder_decks_load_count():
    decks = load_ladder_decks()
    assert len(decks) >= 45


def test_ladder_top50_pool_matches_catalog():
    cat = DeckCatalog()
    pool = cat.pool("ladder_top50")
    assert len(pool) >= 45
    assert set(pool) <= set(cat.decks.keys())


def test_all_ladder_decks_valid_eight_cards():
    cards = load_cards()
    decks = load_ladder_decks()
    for name, card_names in decks.items():
        assert len(card_names) == 8, f"{name}: expected 8 cards, got {len(card_names)}"
        assert len(set(card_names)) == 8, f"{name}: duplicate cards"
        missing = [c for c in card_names if c not in cards]
        assert not missing, f"{name}: unknown cards {missing}"


def test_ladder_decks_have_archetype_tags():
    raw = yaml.safe_load((CONFIG_DIR / "ladder_decks.yaml").read_text())
    for spec in raw["decks"]:
        assert "archetype" in spec
        assert "name" in spec
        assert len(spec["cards"]) == 8


def test_all_ladder_decks_pass_catalog_validation():
    cat = DeckCatalog()
    for name in cat.pool("ladder_top50"):
        resolved = cat.resolve(name)
        assert len(resolved) == 8
