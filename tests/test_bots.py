import numpy as np

from src.bots.archetypes import RusherBot
from src.bots.base import affordable_slots
from src.bots.champion import ChampionBot, RandomBot
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.constants import Side
from tests.conftest import make_engine


def test_champion_bot_returns_legal_action(cards, arena):
    engine = make_engine(cards, arena)
    bot = RusherBot()
    player = engine.players[Side.BOTTOM]
    # Give enough elixir to play
    player.elixir = 10.0
    action = bot.decide(engine, Side.BOTTOM)
    if action is not None:
        card = player.hand[action.slot]
        assert engine.legal_deploy(Side.BOTTOM, card, action.x, action.y)


def test_champion_for_deck_name(cards):
    cat = DeckCatalog(cards=cards)
    bot = ChampionBot.for_deck_name(cat, "rusher")
    assert bot.archetype == "rusher"


def test_registry_uc_tier(cards):
    cat = DeckCatalog(cards=cards)
    bot = get_bot("control", catalog=cat, skill_tier="ultimate_champion")
    assert bot.name == "champion_control"


def test_random_bot_sometimes_waits(cards, arena, rng):
    engine = make_engine(cards, arena)
    bot = RandomBot(rng)
    player = engine.players[Side.BOTTOM]
    player.elixir = 10.0
    waits = sum(1 for _ in range(50) if bot.decide(engine, Side.BOTTOM) is None)
    assert waits > 10


def test_affordable_slots_respects_elixir(cards, arena, rng):
    engine = make_engine(cards, arena)
    player = engine.players[Side.BOTTOM]
    player.elixir = 2.0
    slots = affordable_slots(player)
    assert all(player.hand[s].cost <= 2 for s in slots)
