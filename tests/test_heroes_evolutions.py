import pytest

from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_cards
from src.simulator.constants import Side
from src.simulator.heroes import trigger_ability
from tests.conftest import make_engine, spawn_unit


def test_load_cards_includes_heroes_and_evolutions():
    cards = load_cards()
    assert "golden_knight" in cards
    assert cards["golden_knight"].is_hero
    assert "knight_evo" in cards
    assert cards["knight_evo"].is_evolution
    assert cards["knight_evo"].evolution_of == "knight"
    assert cards["knight_evo"].hp > cards["knight"].hp


def test_evolution_stat_overrides(cards):
    assert cards["mortar_evo"].hp > cards["mortar"].hp
    assert cards["skeletons_evo"].count == 4
    assert cards["wall_breakers_evo"].speed > cards["wall_breakers"].speed


def test_deck_validation_one_hero(cards):
    cat = DeckCatalog(cards=cards)
    cat.validate_names(
        ["golden_knight", "knight", "archers", "goblins", "giant", "musketeer", "fireball", "cannon"]
    )
    with pytest.raises(ValueError, match="at most one hero"):
        cat.validate_names(
            ["golden_knight", "skeleton_king", "knight", "archers", "goblins",
             "giant", "musketeer", "fireball"]
        )


def test_deck_validation_one_evolution(cards):
    cat = DeckCatalog(cards=cards)
    cat.validate_names(
        ["knight_evo", "archers", "goblins", "giant", "musketeer", "minions", "fireball", "cannon"]
    )
    with pytest.raises(ValueError, match="at most one evolution"):
        cat.validate_names(
            ["knight_evo", "mortar_evo", "archers", "goblins", "giant",
             "musketeer", "fireball", "cannon"]
        )


def test_deck_validation_no_base_and_evo(cards):
    cat = DeckCatalog(cards=cards)
    with pytest.raises(ValueError, match="cannot include both"):
        cat.validate_names(
            ["knight", "knight_evo", "archers", "goblins", "giant",
             "musketeer", "fireball", "cannon"]
        )


def test_named_decks_with_heroes_resolve(cards):
    cat = DeckCatalog(cards=cards)
    for deck_name in ("rusher", "control", "siege", "beatdown"):
        deck = cat.resolve(deck_name)
        assert len(deck) == 8
        assert sum(1 for c in deck if c.is_hero) == 1
        assert sum(1 for c in deck if c.is_evolution) == 1


def test_hero_ability_dash_deals_damage(cards, arena):
    engine = make_engine(cards, arena)
    gk = cards["golden_knight"]
    hero = spawn_unit(engine, gk, Side.BOTTOM, 9.0, 10.0)
    enemy = spawn_unit(engine, cards["knight"], Side.TOP, 9.0, 12.0)
    hp_before = enemy.hp
    trigger_ability(hero, engine, events := [])
    assert any(e["type"] == "hero_ability" for e in events)
    assert enemy.hp < hp_before or hero.x != 10.0


def test_hero_spawn_ability(cards, arena):
    engine = make_engine(cards, arena)
    sk = cards["skeleton_king"]
    hero = spawn_unit(engine, sk, Side.BOTTOM, 9.0, 10.0)
    before = len(engine.units)
    trigger_ability(hero, engine, events := [])
    assert len(engine.units) > before
    assert any(e["type"] == "hero_ability" for e in events)


def test_hero_damage_buff(cards, arena):
    engine = make_engine(cards, arena)
    aq = cards["archer_queen"]
    hero = spawn_unit(engine, aq, Side.BOTTOM, 9.0, 10.0)
    trigger_ability(hero, engine, [])
    assert hero.damage_multiplier > 1.0
    assert hero.buff_until > engine.time


def test_hero_charges_in_tick(cards, arena):
    engine = make_engine(cards, arena)
    gk = cards["golden_knight"]
    hero = spawn_unit(engine, gk, Side.BOTTOM, 9.0, 10.0)
    charge_time = gk.ability_charge
    ticks = int(charge_time / arena.dt) + 1
    events = []
    for _ in range(ticks):
        events.extend(engine.tick())
    assert any(e["type"] == "hero_ability" for e in events)


def test_deploy_evolution_uses_evo_stats(cards, arena):
    engine = make_engine(
        cards,
        arena,
        deck=[cards["knight_evo"]] * 8,
    )
    player = engine.players[Side.BOTTOM]
    player.elixir = 10.0
    slot = next(i for i, c in enumerate(player.hand) if c.name == "knight_evo")
    engine.play_card(Side.BOTTOM, slot, 9.0, 8.0)
    unit = next(u for u in engine.units if u.stats.name == "knight_evo")
    assert unit.stats.hp == cards["knight_evo"].hp
    assert unit.stats.splash_radius > 0
