import numpy as np
import pytest

from src.simulator.player import PlayerState


def make_player(cards, rng, start=5.0, cap=10.0, interval=2.8):
    deck = [cards[n] for n in ["knight", "archers", "goblins", "giant",
                               "musketeer", "minions", "fireball", "cannon"]]
    return PlayerState(deck, rng, start, cap, interval)


def test_regen_rate_exact(cards, rng):
    p = make_player(cards, rng, start=0.0)
    for _ in range(28):  # 2.8s at 10 Hz
        p.regen(0.1, double=False)
    assert p.elixir == pytest.approx(1.0)


def test_double_elixir_rate(cards, rng):
    p = make_player(cards, rng, start=0.0)
    for _ in range(14):  # 1.4s doubled == one elixir
        p.regen(0.1, double=True)
    assert p.elixir == pytest.approx(1.0)


def test_cap_and_leak(cards, rng):
    p = make_player(cards, rng, start=9.95)
    leaked = sum(p.regen(0.1, double=False) for _ in range(56))  # 2 elixir worth
    assert p.elixir == 10.0
    assert leaked == pytest.approx(p.leaked)
    assert p.leaked == pytest.approx(2.0 - 0.05)


def test_spend_rejection(cards, rng):
    p = make_player(cards, rng, start=1.0)
    expensive = next(i for i, c in enumerate(p.hand) if c.cost > 1)
    assert not p.can_afford(expensive)
    with pytest.raises(ValueError):
        p.play(expensive)
    assert p.elixir == 1.0  # unchanged after rejection


def test_spend_deducts(cards, rng):
    p = make_player(cards, rng, start=10.0)
    cost = p.hand[0].cost
    p.play(0)
    assert p.elixir == pytest.approx(10.0 - cost)
    assert p.spent == pytest.approx(cost)


def test_deck_too_small_rejected(cards, rng):
    with pytest.raises(ValueError):
        PlayerState([cards["knight"]] * 4, rng, 5, 10, 2.8)
