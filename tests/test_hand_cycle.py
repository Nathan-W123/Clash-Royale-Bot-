import numpy as np

from src.simulator.constants import HAND_SIZE
from src.simulator.player import PlayerState


def make_player(cards, seed=0):
    deck = [cards[n] for n in ["knight", "archers", "goblins", "giant",
                               "musketeer", "minions", "fireball", "cannon"]]
    return PlayerState(deck, np.random.default_rng(seed), 100.0, 100.0, 2.8)


def test_hand_size_and_next_visible(cards):
    p = make_player(cards)
    assert len(p.hand) == HAND_SIZE
    assert p.next_card is not None
    names = {c.name for c in p.hand} | {p.next_card.name}
    assert len(names) == HAND_SIZE + 1  # all distinct


def test_played_card_cycles_to_back(cards):
    p = make_player(cards)
    first = p.hand[0]
    expected_next = p.next_card
    p.play(0)
    assert p.hand[0] is expected_next        # slot refilled from queue front
    # After 3 more plays the first card is at the front of the queue again.
    for _ in range(3):
        p.play(1)
    assert p.next_card is first
    p.play(2)
    assert first in p.hand                   # back in hand on the 5th play


def test_shuffle_deterministic_under_seed(cards):
    a, b = make_player(cards, seed=7), make_player(cards, seed=7)
    assert [c.name for c in a.hand] == [c.name for c in b.hand]
    c = make_player(cards, seed=8)
    # Different seed almost surely produces a different order.
    assert ([x.name for x in a.hand] != [x.name for x in c.hand]
            or a.next_card.name != c.next_card.name)
