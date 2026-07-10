import numpy as np

from src.decks.catalog import DeckCatalog
from src.decks.generator import DeckSearch, generate_deck, generate_from_template
from src.decks.sampling import sample_deck, sample_match_decks


def test_catalog_loads_pools_and_decks():
    cat = DeckCatalog()
    assert "rusher" in cat.decks
    assert len(cat.resolve("rusher")) == 8
    assert "all_archetypes" in cat.pools
    assert "rusher" in cat.pool("all_archetypes")


def test_stage2_pool_is_not_a_playable_deck():
    cat = DeckCatalog()
    assert "stage2_pool" not in cat.decks


def test_sample_deck_from_pool(cards):
    cat = DeckCatalog(cards=cards)
    rng = np.random.default_rng(0)
    name, deck = sample_deck(cat, "all_archetypes", rng)
    assert name in cat.pool("all_archetypes")
    assert len(deck) == 8


def test_sample_match_decks_different_pools(cards):
    cat = DeckCatalog(cards=cards)
    rng = np.random.default_rng(1)
    (an, _), (on, _) = sample_match_decks(
        cat, "stage2_pool", "all_archetypes", rng
    )
    assert an in cat.pool("stage2_pool")
    assert on in cat.pool("all_archetypes")


def test_generate_from_template(cards):
    cat = DeckCatalog(cards=cards)
    rng = np.random.default_rng(2)
    names = generate_from_template(cat, "rusher", rng)
    assert len(names) == 8
    assert "hog_rider" in names
    assert len(set(names)) == 8


def test_deck_search_runs(cards):
    cat = DeckCatalog(cards=cards)
    rng = np.random.default_rng(3)

    def evaluate(deck):
        return sum(c.cost for c in deck) / len(deck)

    search = DeckSearch(cat, evaluate, rng=rng)
    seed = cat.decks["rusher"]
    best = search.search(seed, iterations=20, population=4)
    assert len(best.names) == 8
    assert best.score > 0
