"""Card-level scaling.

The feature exists so simulated breakpoints match the ones the player
actually owns, so the tests are written around breakpoints and mixed levels
rather than around individual numbers.
"""
from __future__ import annotations

import pytest

from src.simulator.cards import load_arena, load_cards
from src.simulator.constants import Side
from src.simulator.levels import (
    CardLevels,
    describe,
    growth,
    load_card_levels,
    load_reference,
    scale_arena,
    scale_cards,
)
from tests.conftest import make_engine, spawn_unit


@pytest.fixture(scope="module")
def reference():
    return load_reference()


# ------------------------------------------------------------------ config


def test_level_one_is_a_no_op(cards, arena):
    levels = CardLevels()
    assert levels.is_uniform_level_one
    scaled = scale_cards(cards, levels)
    assert all(scaled[n].hp == cards[n].hp for n in cards)
    assert scale_arena(arena, levels) is arena


def test_evolutions_inherit_their_base_card_level():
    levels = CardLevels(default=9, overrides={"hog_rider": 13})
    assert levels.level_of("hog_rider") == 13
    assert levels.level_of("hog_rider_evo") == 13
    assert levels.level_of("knight_evo") == 9


def test_explicit_evolution_override_wins():
    levels = CardLevels(default=9, overrides={"knight": 11, "knight_evo": 14})
    assert levels.level_of("knight_evo") == 14


def test_levels_outside_the_real_range_are_rejected():
    with pytest.raises(ValueError, match="outside 1-15"):
        CardLevels(default=16)
    with pytest.raises(ValueError, match="outside 1-15"):
        CardLevels(default=0)
    with pytest.raises(ValueError, match="outside 1-15"):
        CardLevels(default=11, overrides={"knight": 99})


def test_non_integer_levels_are_rejected():
    with pytest.raises(ValueError, match="must be an integer"):
        CardLevels(default=11.5)


def test_round_trips_through_a_dict():
    levels = CardLevels(default=11, tower=12, overrides={"hog_rider": 13})
    assert CardLevels.from_dict(levels.to_dict()) == levels


def test_tower_defaults_to_the_card_level():
    assert CardLevels.from_dict({"default": 11}).tower == 11


def test_loads_the_shipped_config():
    levels = load_card_levels(path="configs/card_levels.yaml")
    assert 1 <= levels.default <= 15
    assert 1 <= levels.tower <= 15


# ----------------------------------------------------------------- scaling


def test_scaling_matches_the_real_ladder(cards, reference):
    """Knight's level-11 HP is 1766 in the datamined table; scaling by the
    ladder ratio against a config that already matches level 1 must land
    exactly there."""
    scaled = scale_cards(cards, CardLevels(default=11), reference)
    assert scaled["knight"].hp == 1766
    assert scaled["knight"].damage == 202


def test_scaling_is_proportional_so_hand_tuning_survives(cards, reference):
    """Scaling multiplies by the ladder *ratio* rather than substituting the
    ladder value, so a deliberate config deviation is preserved instead of
    being silently reverted by a level change."""
    from dataclasses import replace

    tweaked = dict(cards)
    tweaked["knight"] = replace(cards["knight"], hp=1380.0)  # 2x the real value
    scaled = scale_cards(tweaked, CardLevels(default=11), reference)
    assert scaled["knight"].hp == pytest.approx(1766 * 2, rel=0.001)


def test_only_level_dependent_fields_move(cards, reference):
    scaled = scale_cards(cards, CardLevels(default=13), reference)
    for name in ("knight", "musketeer", "hog_rider"):
        a, b = cards[name], scaled[name]
        assert b.hp > a.hp and b.damage > a.damage
        assert (b.hit_speed, b.range, b.sight_range, b.speed, b.count) == \
               (a.hit_speed, a.range, a.sight_range, a.speed, a.count)


def test_spell_damage_scales(cards, reference):
    scaled = scale_cards(cards, CardLevels(default=11), reference)
    assert scaled["fireball"].spell_damage == 832
    assert scaled["fireball"].spell_radius == cards["fireball"].spell_radius


def test_death_damage_scales_with_the_attack_ladder(cards, reference):
    """Balloon's payload is damage it deals; leaving it at level 1 would make
    the card quietly weaker at every level above 1."""
    scaled = scale_cards(cards, CardLevels(default=11), reference)
    assert scaled["balloon"].death_damage > cards["balloon"].death_damage


def test_cards_without_a_ladder_still_scale(cards, reference):
    """Death-spawn products and heroes have no upstream ladder; they fall
    back to compounding growth rather than being left at level 1 while
    everything around them scales."""
    scaled = scale_cards(cards, CardLevels(default=11), reference)
    assert scaled["lava_pup"].hp > cards["lava_pup"].hp


def test_towers_scale_on_their_own_level(arena, reference):
    scaled = scale_arena(arena, CardLevels(default=1, tower=11), reference)
    assert scaled.princess.hp == 3584
    assert scaled.king.hp == 6144
    assert scaled.princess.damage == 128
    assert scaled.princess.range == arena.princess.range


def test_tower_level_is_independent_of_card_level(arena, reference):
    cards_only = scale_arena(arena, CardLevels(default=13, tower=1), reference)
    assert cards_only.princess.hp == arena.princess.hp


def test_growth_falls_back_without_a_ladder():
    assert growth(None, 1) == pytest.approx(1.0)
    assert growth(None, 3) == pytest.approx(1.21)
    assert growth([100, 110, 121], 3) == pytest.approx(1.21)


def test_growth_clamps_past_the_end_of_a_short_ladder():
    assert growth([100, 110], 9) == pytest.approx(1.1)


# ------------------------------------------------------------- breakpoints


def test_uniform_levels_preserve_breakpoints(cards, reference):
    """The reason a global rescale was the wrong fix: at a uniform level the
    same spells kill the same troops."""
    def kills(table):
        return {(s, t): table[s].spell_damage >= table[t].hp
                for s in ("fireball", "arrows", "zap", "rocket")
                for t in ("musketeer", "wizard", "minions", "skeletons", "barbarians")}

    base = kills(cards)
    for level in (7, 11, 14):
        assert kills(scale_cards(cards, CardLevels(default=level), reference)) == base


def test_mixed_levels_do_move_breakpoints(cards, reference):
    """...and the reason this feature exists: they do *not* survive a deck
    with uneven levels."""
    even = scale_cards(cards, CardLevels(default=11), reference)
    uneven = scale_cards(cards, CardLevels(default=11, overrides={"musketeer": 14}),
                         reference)
    assert even["fireball"].spell_damage == uneven["fireball"].spell_damage
    assert uneven["musketeer"].hp > even["musketeer"].hp


# ------------------------------------------------------------- integration


def test_engine_uses_the_scaled_table_for_death_spawns(cards, arena, reference):
    """A level-13 Lava Hound must not leave level-1 pups behind."""
    from src.simulator.engine import BattleEngine

    scaled = scale_cards(cards, CardLevels(default=13), reference)
    deck = [scaled[n] for n in ("knight", "archers", "goblins", "giant",
                                "musketeer", "minions", "fireball", "cannon")]
    engine = BattleEngine(deck, list(deck), arena, cards=scaled)
    hound = spawn_unit(engine, scaled["lava_hound"], Side.TOP, 9.0, 20.0)
    hound.hp = 0.0
    engine.tick()
    pups = [u for u in engine.units if u.stats.name == "lava_pup"]
    assert pups
    assert pups[0].hp == scaled["lava_pup"].hp
    assert pups[0].hp > cards["lava_pup"].hp


def test_engine_defaults_to_the_unscaled_table(cards, arena):
    engine = make_engine(cards, arena)
    assert engine.cards["knight"].hp == load_cards()["knight"].hp


def test_env_threads_its_card_table_into_the_engine(cards, arena, decks, reference):
    from src.simulator.env import CRBattleEnv

    scaled = scale_cards(cards, CardLevels(default=12), reference)
    deck = [scaled[n] for n in decks["training_mirror"]]
    env = CRBattleEnv(scaled, scale_arena(arena, CardLevels(default=12, tower=12), reference),
                      deck, list(deck), seed=0)
    env.reset(seed=0)
    assert env.engine.cards["knight"].hp == scaled["knight"].hp
    assert env.engine.arena.princess.hp > arena.princess.hp


# ------------------------------------------------------------- provenance


def test_checkpoint_records_and_returns_levels(tmp_path):
    from src.agent.network import make_network
    from src.agent.selfplay import checkpoint_card_levels, save_checkpoint

    levels = CardLevels(default=11, tower=12, overrides={"hog_rider": 13})
    net = make_network(8, {"conv_channels": (4,), "cnn_out": 8, "fusion_mlp": 16})
    path = tmp_path / "policy.pt"
    save_checkpoint(net, [f"c{i}" for i in range(8)], path, card_levels=levels)
    assert checkpoint_card_levels(path) == levels


def test_checkpoints_predating_the_feature_read_as_level_one(tmp_path):
    from src.agent.network import make_network
    from src.agent.selfplay import checkpoint_card_levels, save_checkpoint

    net = make_network(8, {"conv_channels": (4,), "cnn_out": 8, "fusion_mlp": 16})
    path = tmp_path / "old.pt"
    save_checkpoint(net, [f"c{i}" for i in range(8)], path)
    assert checkpoint_card_levels(path).is_uniform_level_one


def test_describe_is_readable():
    assert "level 1" in describe(CardLevels())
    text = describe(CardLevels(default=11, tower=12, overrides={"hog_rider": 13}))
    assert "11" in text and "12" in text and "hog_rider=13" in text
