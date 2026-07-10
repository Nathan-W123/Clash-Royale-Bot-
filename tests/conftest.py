import numpy as np
import pytest

from src.simulator.cards import load_arena, load_cards, load_decks


@pytest.fixture(scope="session")
def cards():
    return load_cards()


@pytest.fixture(scope="session")
def arena():
    return load_arena()


@pytest.fixture(scope="session")
def decks():
    return load_decks()


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


def make_engine(cards, arena, deck=None, **kwargs):
    from src.simulator.engine import BattleEngine
    deck = deck or [cards[n] for n in
                    ["knight", "archers", "goblins", "giant",
                     "musketeer", "minions", "fireball", "cannon"]]
    return BattleEngine(deck, list(deck), arena, **kwargs)


def spawn_unit(engine, stats, side, x, y, hp=None):
    """Inject a unit directly, bypassing hand/elixir — for engine-level tests."""
    from src.simulator.entities import Unit
    u = Unit(id=engine._new_id(), stats=stats, side=side, x=x, y=y,
             hp=hp if hp is not None else stats.hp,
             cooldown=stats.hit_speed, elixir_value=stats.cost / stats.count)
    engine.units.append(u)
    engine._by_id[u.id] = u
    return u


def dummy_stats(name="dummy", hp=1_000_000.0, **overrides):
    """Inert punching-bag troop: never moves, effectively never attacks."""
    from src.simulator.cards import CardStats
    from src.simulator.constants import CardType, TargetType
    spec = dict(name=name, type=CardType.TROOP, cost=3, hp=hp, damage=1.0,
                hit_speed=10_000.0, range=0.5, sight_range=5.5, speed=0.0,
                targets=TargetType.GROUND, flying=False, count=1)
    spec.update(overrides)
    return CardStats(**spec)


def slot_of(player, name):
    return next(i for i, c in enumerate(player.hand) if c.name == name)


def force_hand(player, cards, names):
    """Pin a player's hand to a known card order (queue gets the rest)."""
    deck = {c.name: c for c in player.hand} | {c.name: c for c in player._queue}
    player.hand = [cards[n] for n in names]
    player._queue = [c for c in deck.values() if c.name not in names] or [cards[n] for n in names]
