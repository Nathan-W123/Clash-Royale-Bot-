"""Unit-specific mechanics: charge, inferno ramp, stun resets, death effects,
tunnelling, and obstacle steering (#36).

These are the mechanics that change *which card a good player picks*, so each
test is written around the decision it enables rather than around the field
it sets.
"""
from __future__ import annotations

import pytest

from src.simulator import mechanics, movement, spell_effects
from src.simulator.constants import Side
from src.simulator.entities import PendingSpell
from tests.conftest import dummy_stats, force_hand, make_engine, slot_of, spawn_unit


def kings_only(engine):
    """Drop the princess towers.

    Tower range gates on *edge* distance (7.5 + both radii), so almost every
    mid-board tile is inside some tower's reach and stray tower fire quietly
    corrupts damage measurements. Kings start deactivated, so leaving only
    them gives a quiet arena while keeping the win-condition checks valid.
    """
    engine.towers = [t for t in engine.towers if t.is_king]
    return engine


# ------------------------------------------------------------------ charge


def test_charge_accumulates_over_distance_then_completes(cards):
    prince = _fake(cards["prince"])
    mechanics.advance_charge(prince, 2.0)
    assert not prince.charged
    mechanics.advance_charge(prince, 2.0)
    assert prince.charged


def test_non_charging_cards_never_charge(cards):
    knight = _fake(cards["knight"])
    mechanics.advance_charge(knight, 50.0)
    assert not knight.charged
    assert mechanics.attack_scale(knight) == 1.0


def test_blocked_movement_resets_a_banked_charge(cards):
    prince = _fake(cards["prince"])
    mechanics.advance_charge(prince, 3.0)
    assert prince.charge_progress == 3.0
    mechanics.break_charge(prince)
    assert prince.charge_progress == 0.0
    assert not prince.charged


def test_prince_charges_up_over_a_clear_run(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    prince = spawn_unit(eng, cards["prince"], Side.BOTTOM, 9.0, 5.0)
    spawn_unit(eng, dummy_stats(hp=50_000.0), Side.TOP, 9.0, 13.0)
    for _ in range(100):
        eng.tick()
        if prince.charged:
            break
    assert prince.charged
    assert prince.y > 5.0, "it charged by actually walking"


def test_charged_hit_lands_double_damage_and_spends_the_charge(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    prince = spawn_unit(eng, cards["prince"], Side.BOTTOM, 9.0, 6.0)
    bag = spawn_unit(eng, dummy_stats(hp=50_000.0), Side.TOP, 9.0, 7.5)
    prince.cooldown = 0.0
    prince.charged = True

    before = bag.hp
    eng.tick()
    assert before - bag.hp == pytest.approx(cards["prince"].damage * 2.0)
    assert not prince.charged, "a connected hit spends the charge"

    prince.cooldown = 0.0
    mid = bag.hp
    eng.tick()
    assert mid - bag.hp == pytest.approx(cards["prince"].damage)


def test_a_cheap_blocker_eats_the_charge(cards, arena):
    """The decision this exists for: a 1-elixir body in front of a Prince is
    a good trade *because* it absorbs the charged hit."""
    eng = kings_only(make_engine(cards, arena))
    prince = spawn_unit(eng, cards["prince"], Side.BOTTOM, 9.0, 6.0)
    blocker = spawn_unit(eng, dummy_stats(hp=50_000.0), Side.TOP, 9.0, 7.5)
    prince.charged = True
    prince.cooldown = 0.0
    eng.tick()
    assert 50_000.0 - blocker.hp == pytest.approx(cards["prince"].damage * 2.0)
    assert not prince.charged


def test_charge_speeds_the_unit_up(cards):
    prince = _fake(cards["prince"])
    assert mechanics.charge_speed(prince) == 1.0
    prince.charged = True
    assert mechanics.charge_speed(prince) == cards["prince"].charge_speed_multiplier


# -------------------------------------------------------------- inferno ramp


def test_inferno_damage_ramps_while_locked_on_one_target(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    tower = spawn_unit(eng, cards["inferno_tower"], Side.BOTTOM, 9.0, 10.0)
    bag = spawn_unit(eng, dummy_stats(hp=500_000.0), Side.TOP, 9.0, 12.0)
    tower.cooldown = 0.0

    hits = []
    previous = bag.hp
    for _ in range(60):
        eng.tick()
        if bag.hp < previous:
            hits.append(previous - bag.hp)
            previous = bag.hp
    assert len(hits) > 5
    assert hits[0] == pytest.approx(cards["inferno_tower"].damage, rel=0.2)
    assert hits[-1] > hits[0] * 4, "the beam must actually bite by the end"


def test_ramp_resets_when_the_target_changes(cards):
    inferno = _fake(cards["inferno_tower"])
    mechanics.advance_ramp(inferno, 7, dt=0.1)   # acquires the lock
    mechanics.advance_ramp(inferno, 7, dt=3.0)   # holds it
    assert mechanics.ramp_multiplier(inferno) > 5.0
    mechanics.advance_ramp(inferno, 9, dt=0.1)
    assert mechanics.ramp_multiplier(inferno) == pytest.approx(1.0)


def test_ramp_is_capped_at_the_configured_multiplier(cards):
    inferno = _fake(cards["inferno_tower"])
    mechanics.advance_ramp(inferno, 1, dt=0.1)
    mechanics.advance_ramp(inferno, 1, dt=100.0)
    assert mechanics.ramp_multiplier(inferno) == pytest.approx(
        cards["inferno_tower"].ramp_up_multiplier)


def test_cards_without_ramp_are_unaffected(cards):
    knight = _fake(cards["knight"])
    mechanics.advance_ramp(knight, 3, dt=10.0)
    assert mechanics.ramp_multiplier(knight) == 1.0


# -------------------------------------------------------------------- stun


def test_stun_resets_charge_ramp_and_the_attack_wind_up(cards):
    sparky = _fake(cards["sparky"])
    sparky.cooldown = 0.1
    sparky.charged = True
    sparky.ramp_time = 5.0
    mechanics.on_stun(sparky)
    assert sparky.cooldown == cards["sparky"].hit_speed
    assert not sparky.charged
    assert sparky.ramp_time == 0.0


def test_zap_resets_a_sparky_about_to_fire(cards, arena):
    """The play: a 2-elixir Zap costs Sparky its whole 4-second wind-up."""
    eng = kings_only(make_engine(cards, arena))
    sparky = spawn_unit(eng, cards["sparky"], Side.TOP, 9.0, 12.0)
    sparky.cooldown = 0.05
    eng.spells.append(PendingSpell(
        side=Side.BOTTOM, x=9.0, y=12.0, radius=cards["zap"].spell_radius,
        damage=cards["zap"].spell_damage,
        tower_multiplier=cards["zap"].tower_multiplier,
        resolve_at=0.0, card_name="zap"))
    eng.tick()
    assert sparky.cooldown == pytest.approx(cards["sparky"].hit_speed)
    assert sparky.frozen_until > eng.time


def test_zap_resets_a_ramped_inferno(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    inferno = spawn_unit(eng, cards["inferno_tower"], Side.TOP, 9.0, 12.0)
    inferno.ramp_target_id = 1234
    inferno.ramp_time = 3.0
    eng.spells.append(PendingSpell(
        side=Side.BOTTOM, x=9.0, y=12.0, radius=cards["zap"].spell_radius,
        damage=cards["zap"].spell_damage,
        tower_multiplier=cards["zap"].tower_multiplier,
        resolve_at=0.0, card_name="zap"))
    eng.tick()
    assert mechanics.ramp_multiplier(inferno) == pytest.approx(1.0)


def test_freeze_also_resets_wind_ups(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    sparky = spawn_unit(eng, cards["sparky"], Side.TOP, 9.0, 12.0)
    sparky.cooldown = 0.05
    eng.spells.append(PendingSpell(
        side=Side.BOTTOM, x=9.0, y=12.0, radius=3.0, damage=1.0,
        tower_multiplier=0.35, resolve_at=0.0, card_name="freeze"))
    eng.tick()
    assert sparky.cooldown == pytest.approx(cards["sparky"].hit_speed)


# ------------------------------------------------------------------- death


def test_lava_hound_spawns_pups_on_death(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    hound = spawn_unit(eng, cards["lava_hound"], Side.TOP, 9.0, 12.0)
    hound.hp = 0.0
    eng.tick()
    pups = [u for u in eng.units if u.stats.name == "lava_pup"]
    assert len(pups) == cards["lava_hound"].death_spawn_count
    assert all(p.side == Side.TOP for p in pups)


def test_golem_splits_into_golemites(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    golem = spawn_unit(eng, cards["golem"], Side.BOTTOM, 9.0, 10.0)
    golem.hp = 0.0
    eng.tick()
    assert len([u for u in eng.units if u.stats.name == "golemite"]) == 2


def test_balloon_death_damage_hits_enemies_not_the_owner(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    balloon = spawn_unit(eng, cards["balloon"], Side.TOP, 9.0, 12.0)
    victim = spawn_unit(eng, dummy_stats(hp=5_000.0), Side.BOTTOM, 9.0, 12.5)
    ally = spawn_unit(eng, dummy_stats(name="ally", hp=5_000.0), Side.TOP, 9.5, 12.5)
    balloon.hp = 0.0
    eng.tick()
    assert 5_000.0 - victim.hp == pytest.approx(cards["balloon"].death_damage)
    assert ally.hp == 5_000.0, "a death bomb must not clip its own push"


def test_giant_skeleton_bomb_reaches_a_tower(cards, arena):
    eng = make_engine(cards, arena)
    tower = next(t for t in eng.towers if t.side == Side.TOP and t.kind == "princess_left")
    skeleton = spawn_unit(eng, cards["giant_skeleton"], Side.BOTTOM, tower.x, tower.y - 1.0)
    before = tower.hp
    skeleton.hp = 0.0
    eng.tick()
    assert before - tower.hp >= cards["giant_skeleton"].death_damage


def test_cards_without_death_effects_leave_nothing_behind(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    knight = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 10.0)
    knight.hp = 0.0
    eng.tick()
    assert eng.units == []


# -------------------------------------------------------------- tunnelling


def test_miner_deploys_anywhere(cards, arena):
    eng = make_engine(cards, arena)
    behind_enemy_tower = (14.5, 27.0)
    assert eng.legal_deploy(Side.BOTTOM, cards["miner"], *behind_enemy_tower)
    assert not eng.legal_deploy(Side.BOTTOM, cards["knight"], *behind_enemy_tower)


def test_goblin_drill_deploys_anywhere(cards, arena):
    eng = make_engine(cards, arena)
    assert eng.legal_deploy(Side.BOTTOM, cards["goblin_drill"], 9.0, 26.0)


def test_tunnelling_still_respects_the_arena_bounds(cards, arena):
    eng = make_engine(cards, arena)
    assert not eng.legal_deploy(Side.BOTTOM, cards["miner"], -1.0, 10.0)
    assert not eng.legal_deploy(Side.BOTTOM, cards["miner"], 9.0, 99.0)


# -------------------------------------------------------- persistent spells


def test_tornado_drags_units_across_its_duration(cards, arena):
    """A single teleport and a continuous drag are different plays: the drag
    is what pulls a push into king-tower range."""
    eng = kings_only(make_engine(cards, arena))
    bottom = eng.players[Side.BOTTOM]
    force_hand(bottom, cards, ["tornado", "knight", "archers", "goblins"])
    bottom.elixir = 10.0
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 13.0)

    eng.play_card(Side.BOTTOM, slot_of(bottom, "tornado"), 9.0, 10.0)
    positions = []
    for _ in range(int(spell_effects.TORNADO_DURATION / arena.dt) + 4):
        eng.tick()
        positions.append(bag.y)

    moved_steps = sum(1 for a, b in zip(positions, positions[1:]) if b < a - 1e-9)
    assert moved_steps > 1, "pull must be spread over several ticks"
    assert 13.0 - bag.y == pytest.approx(spell_effects.TORNADO_PULL_DISTANCE, abs=0.2)


def test_tornado_damage_lands_only_once(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    bottom = eng.players[Side.BOTTOM]
    force_hand(bottom, cards, ["tornado", "knight", "archers", "goblins"])
    bottom.elixir = 10.0
    bag = spawn_unit(eng, dummy_stats(hp=50_000.0), Side.TOP, 9.0, 11.0)

    eng.play_card(Side.BOTTOM, slot_of(bottom, "tornado"), 9.0, 10.0)
    for _ in range(30):
        eng.tick()
    assert 50_000.0 - bag.hp == pytest.approx(cards["tornado"].spell_damage, abs=1.0)


def test_rage_zone_catches_units_that_arrive_later(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    bottom = eng.players[Side.BOTTOM]
    force_hand(bottom, cards, ["rage", "knight", "archers", "goblins"])
    bottom.elixir = 10.0

    eng.play_card(Side.BOTTOM, slot_of(bottom, "rage"), 9.0, 10.0)
    eng.tick()
    latecomer = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 10.0)
    assert latecomer.speed_multiplier == pytest.approx(1.0)
    for _ in range(int(spell_effects.tick_interval("rage") / arena.dt) + 2):
        eng.tick()
    assert latecomer.speed_multiplier == pytest.approx(
        spell_effects.RAGE_SPEED_MULTIPLIER)


def test_rage_does_not_outlast_its_duration(cards, arena):
    """Re-applying every tick must not keep pushing the expiry out."""
    eng = kings_only(make_engine(cards, arena))
    bottom = eng.players[Side.BOTTOM]
    force_hand(bottom, cards, ["rage", "knight", "archers", "goblins"])
    bottom.elixir = 10.0
    ally = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 10.0)

    eng.play_card(Side.BOTTOM, slot_of(bottom, "rage"), 9.0, 10.0)
    for _ in range(int((spell_effects.RAGE_DURATION + 1.0) / arena.dt)):
        eng.tick()
    assert ally.speed_multiplier == pytest.approx(1.0)


def test_single_tick_spells_keep_their_old_semantics(cards, arena):
    """Hand-built spells with ticks_left=0 still apply everything at once, so
    engine-level tests that construct them directly stay meaningful."""
    eng = kings_only(make_engine(cards, arena))
    bag = spawn_unit(eng, dummy_stats(), Side.TOP, 9.0, 13.0)
    eng.spells.append(PendingSpell(side=Side.BOTTOM, x=9.0, y=10.0, radius=5.5,
                                   damage=40.0, tower_multiplier=0.35,
                                   resolve_at=0.0, card_name="tornado"))
    eng.tick()
    assert 13.0 - bag.y == pytest.approx(spell_effects.TORNADO_PULL_DISTANCE, abs=0.05)


# ---------------------------------------------------------------- steering


def test_units_steer_around_a_blocking_body(cards, arena):
    """Without this, a unit that bumps a body re-proposes the same blocked
    step forever and stands there for the rest of the match."""
    from src.simulator import collision

    eng = kings_only(make_engine(cards, arena))
    walker = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 8.0)
    walker.target_id = 999  # not the blocker, so collision does not exempt it
    spawn_unit(eng, dummy_stats(name="wall"), Side.BOTTOM, 9.0, 9.05)

    direct = movement.next_position(walker, 9.0, 12.0, arena.dt)
    assert collision.blocked(walker, *direct, eng.units)
    steered = movement.avoid_obstacle(walker, *direct, eng.units, arena)
    assert steered != direct
    assert not collision.blocked(walker, *steered, eng.units)


def test_steering_leaves_a_clear_path_alone(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    walker = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 8.0)
    direct = movement.next_position(walker, 9.0, 12.0, arena.dt)
    assert movement.avoid_obstacle(walker, *direct, eng.units, arena) == direct


def test_steering_holds_position_when_every_way_is_blocked(cards, arena):
    eng = kings_only(make_engine(cards, arena))
    walker = spawn_unit(eng, cards["knight"], Side.BOTTOM, 9.0, 8.0)
    walker.target_id = 999
    for dx, dy in ((0.0, 0.95), (0.95, 0.3), (-0.95, 0.3), (0.7, -0.7), (-0.7, -0.7)):
        spawn_unit(eng, dummy_stats(name=f"wall{dx}{dy}"), Side.BOTTOM,
                   9.0 + dx, 8.0 + dy)
    direct = movement.next_position(walker, 9.0, 12.0, arena.dt)
    assert movement.avoid_obstacle(walker, *direct, eng.units, arena) == direct


def test_ground_sidesteps_never_enter_the_river(cards, arena):
    walker = _fake(cards["knight"])
    assert movement._crosses_river(walker, 16.0, arena)
    assert not movement._crosses_river(walker, 14.0, arena)


def test_flying_units_may_sidestep_over_the_river(cards, arena):
    assert not movement._crosses_river(_fake(cards["minions"]), 16.0, arena)


# ----------------------------------------------------------------- helpers


def _fake(stats):
    from src.simulator.entities import Unit

    return Unit(id=1, stats=stats, side=Side.BOTTOM, x=0.0, y=0.0, hp=stats.hp,
                cooldown=0.0, elixir_value=float(stats.cost))
