"""Tests for adaptive deck builder."""
from __future__ import annotations

from src.decks.builder import AdaptiveDeckBuilder, CardScoreTracker
from src.decks.catalog import DeckCatalog


def test_card_score_tracker_composite():
    tracker = CardScoreTracker()
    tracker.record({"hog_rider": 3}, won=True, crowns=2, elixir_spent=12.0)
    tracker.record({"hog_rider": 2}, won=False, crowns=0, elixir_spent=8.0)
    score = tracker.composite_score("hog_rider")
    assert 0.0 < score < 1.0
    assert tracker.to_dict()["hog_rider"]["plays"] == 5


def test_build_deck_respects_eight_cards(cards):
    cat = DeckCatalog(cards=cards)
    builder = AdaptiveDeckBuilder(cat)
    tracker = CardScoreTracker()
    for card in ["hog_rider", "fireball", "zap", "knight", "archers", "goblins", "cannon", "musketeer"]:
        tracker.record({card: 2}, won=True, crowns=1, elixir_spent=6.0)
    names = builder.build_deck(tracker, cat)
    assert len(names) == 8
    assert len(set(names)) == 8
    cat.validate_names(names)


def test_build_deck_max_one_hero_and_evo(cards):
    cat = DeckCatalog(cards=cards)
    builder = AdaptiveDeckBuilder(cat)
    tracker = CardScoreTracker()
    for card in cat.cards:
        tracker.record({card: 1}, won=True, crowns=1, elixir_spent=3.0)
    names = builder.build_deck(tracker, cat)
    heroes = [n for n in names if cat.cards[n].is_hero]
    evos = [n for n in names if cat.cards[n].is_evolution]
    assert len(heroes) <= 1
    assert len(evos) <= 1


def test_rebuild_after_interval(cards):
    cat = DeckCatalog(cards=cards)
    builder = AdaptiveDeckBuilder(cat)
    initial = list(builder._current_names)
    for i in range(25):
        builder.record_match({"hog_rider": 1}, won=i % 2 == 0, crowns=1, elixir_spent=4.0)
    assert builder._matches_since_rebuild == 0
    assert len(builder.current_deck()) == 8
