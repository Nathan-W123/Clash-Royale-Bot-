"""Deterministic opponent tracker (#38).

The load-bearing tests here are the two at the bottom: the tracker must be
*exact* when perception is complete, and must *diverge* from engine truth
when perception is not. Together they prove it derives rather than peeks.
"""
from __future__ import annotations

import pytest

from src.agent.opponent_tracker import (
    COLLECTOR_INTERVAL,
    COLLECTOR_PAYOUTS,
    OpponentTracker,
    TrackerConfig,
    track_from_events,
)
from src.simulator.constants import Side
from tests.conftest import make_engine


@pytest.fixture()
def tracker(cards, arena):
    return OpponentTracker(TrackerConfig.from_arena(arena), cards)


# ------------------------------------------------------------------- elixir


def test_regen_is_deterministic_arithmetic(tracker, arena):
    tracker.advance(10.0)
    assert tracker.elixir == pytest.approx(
        min(arena.elixir_max, arena.elixir_start + 10.0 / arena.elixir_regen_interval))


def test_regen_doubles_after_double_time(tracker, arena):
    single = 1.0 / arena.elixir_regen_interval
    tracker.advance(arena.double_time)
    tracker.elixir = 0.0
    tracker.advance(arena.double_time + 10.0)
    assert tracker.elixir == pytest.approx(min(arena.elixir_max, 20.0 * single))


def test_step_straddling_the_double_boundary_splits_the_rate(tracker, arena):
    """One coarse step across the boundary must not pick a single rate for
    the whole interval — that is a silent, permanent drift."""
    start = arena.double_time - 5.0
    tracker.advance(start)
    tracker.elixir = 0.0
    tracker.advance(start + 10.0)
    expected = (5.0 + 2.0 * 5.0) / arena.elixir_regen_interval
    assert tracker.elixir == pytest.approx(min(arena.elixir_max, expected))


def test_overflow_at_cap_is_discarded_not_banked(tracker, arena):
    tracker.advance(600.0)
    assert tracker.elixir == pytest.approx(arena.elixir_max)
    assert tracker.leaked > 0
    tracker.observe_play("knight")
    assert tracker.elixir == pytest.approx(arena.elixir_max - 3)


def test_play_subtracts_the_card_cost(tracker, cards):
    tracker.advance(30.0)
    before = tracker.elixir
    tracker.observe_play("giant")
    assert tracker.elixir == pytest.approx(before - cards["giant"].cost)


def test_elixir_never_goes_negative_and_flags_uncertainty(tracker):
    tracker.elixir = 1.0
    tracker.observe_play("golem")
    assert tracker.elixir == 0.0
    assert tracker.uncertainty == 1


def test_collector_pays_out_in_lumps(tracker, cards):
    tracker.advance(60.0)
    tracker.observe_play("elixir_collector")
    tracker.elixir = 0.0
    tracker.advance(60.0 + COLLECTOR_INTERVAL * 2.5)
    regen = (COLLECTOR_INTERVAL * 2.5) / tracker.config.regen_interval
    assert tracker.elixir == pytest.approx(min(10.0, regen + 2.0))


def test_collector_stops_after_its_payout_budget(tracker):
    tracker.observe_play("elixir_collector")
    tracker.advance(COLLECTOR_INTERVAL * (COLLECTOR_PAYOUTS + 4))
    assert not tracker._collectors


def test_our_elixir_golem_death_gifts_them_elixir(tracker):
    tracker.elixir = 4.0
    tracker.observe_our_unit_death("elixir_golem")
    assert tracker.elixir == pytest.approx(5.0)


def test_mirror_costs_the_previous_play_plus_one(tracker):
    tracker.elixir = 10.0
    tracker.observe_play("knight")     # 3
    tracker.observe_play("mirror")     # 3 + 1
    assert tracker.elixir == pytest.approx(10.0 - 3 - 4)


# -------------------------------------------------------------------- cycle


def test_cycle_resolves_to_the_exact_hand_once_the_deck_is_revealed(tracker):
    deck = ["knight", "archers", "goblins", "giant",
            "musketeer", "minions", "fireball", "cannon"]
    tracker.elixir = 10.0
    for card in deck:
        tracker.elixir = 10.0
        tracker.observe_play(card)
    assert tracker.deck_known
    # After one full pass the first four played are back in hand, in order.
    assert tracker.possible_hand() == deck[:4]
    assert tracker.next_card == deck[4]


def test_known_deck_seeds_the_cycle(cards, arena):
    deck = ["knight", "archers", "goblins", "giant",
            "musketeer", "minions", "fireball", "cannon"]
    t = OpponentTracker(TrackerConfig.from_arena(arena), cards, deck=deck)
    assert t.possible_hand() == deck[:4]
    assert t.next_card == "musketeer"
    t.elixir = 10.0
    t.observe_play("knight")
    assert t.next_card == "minions"
    assert "musketeer" in t.possible_hand()


def test_candidate_set_narrows_as_cards_reveal(tracker):
    assert tracker.candidate_cards() == []
    tracker.elixir = 10.0
    tracker.observe_play("hog_rider")
    tracker.elixir = 10.0
    tracker.observe_play("fireball")
    assert tracker.candidate_cards() == ["hog_rider", "fireball"]


# ----------------------------------------------------------- self-correction


def test_unseen_unit_forces_a_retroactive_play(tracker, cards):
    tracker.advance(30.0)
    before = tracker.elixir
    assert tracker.observe_unit("giant", key=1) is True
    assert tracker.elixir == pytest.approx(before - cards["giant"].cost)
    assert tracker.uncertainty == 1
    assert tracker.inferred_plays == ["giant"]


def test_seen_deploy_is_not_double_charged_when_its_unit_renders(tracker, cards):
    tracker.advance(30.0)
    before = tracker.elixir
    tracker.observe_play("giant")
    assert tracker.observe_unit("giant", key=1) is False
    assert tracker.elixir == pytest.approx(before - cards["giant"].cost)
    assert tracker.uncertainty == 0


def test_repeat_sightings_of_the_same_unit_are_free(tracker):
    tracker.advance(30.0)
    tracker.observe_unit("knight", key="u7")
    after = tracker.elixir
    for _ in range(5):
        assert tracker.observe_unit("knight", key="u7") is False
    assert tracker.elixir == pytest.approx(after)


def test_swarm_deploy_infers_one_play_not_one_per_unit(tracker, cards):
    tracker.advance(40.0)
    before = tracker.elixir
    for i in range(cards["goblins"].count):
        tracker.observe_unit("goblins", key=("gob", i))
    assert tracker.elixir == pytest.approx(before - cards["goblins"].cost)
    assert tracker.uncertainty == 1


def test_missed_spell_is_caught_when_its_effect_lands(tracker, cards):
    tracker.advance(30.0)
    before = tracker.elixir
    assert tracker.observe_spell("fireball") is True
    assert tracker.elixir == pytest.approx(before - cards["fireball"].cost)


def test_uncertainty_widens_the_reported_range(tracker):
    tracker.advance(40.0)
    lo, hi = tracker.elixir_range
    assert lo == hi == tracker.elixir
    tracker.observe_unit("knight", key=1)
    lo, hi = tracker.elixir_range
    assert lo < tracker.elixir < hi


# ------------------------------------------------- the derived/read boundary


def test_tracker_never_touches_engine_elixir(cards, arena):
    """Structural check: the module must contain no read of the opponent's
    engine-side elixir. This is a scope boundary, not an optimization."""
    import ast
    from pathlib import Path

    import src.agent.opponent_tracker as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    for node in ast.walk(tree):  # drop docstrings, which discuss the rule
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body.pop(0)
    code = ast.unparse(tree)
    assert "players[" not in code
    assert "engine." not in code


def test_complete_perception_matches_engine_truth_exactly(cards, arena):
    """With every play observed, the derived count equals ground truth. This
    is the "strong players derive it exactly" claim, made concrete."""
    engine = make_engine(cards, arena, seed=3)
    t = OpponentTracker(TrackerConfig.from_arena(arena), cards)
    opp = engine.players[Side.TOP]

    for step in range(200):
        events = engine.tick()
        if step % 20 == 0:
            slot = next((i for i in range(4) if opp.can_afford(i)), None)
            if slot is not None:
                card = opp.hand[slot]
                if engine.legal_deploy(Side.TOP, card, 9.0, 26.0):
                    events += engine.play_card(Side.TOP, slot, 9.0, 26.0)
        track_from_events(t, events, engine.time, Side.TOP)

    assert t.uncertainty == 0
    assert t.elixir == pytest.approx(opp.elixir, abs=1e-6)


def test_partial_perception_diverges_from_engine_truth(cards, arena):
    """The tracker is wrong exactly when perception was wrong — the same
    failure mode a human has, and the observable difference between deriving
    and reading memory. If this ever passes by matching ground truth, the
    tracker has started peeking."""
    engine = make_engine(cards, arena, seed=3)
    t = OpponentTracker(TrackerConfig.from_arena(arena), cards)
    opp = engine.players[Side.TOP]
    dropped = 0

    for step in range(200):
        events = engine.tick()
        if step % 20 == 0:
            slot = next((i for i in range(4) if opp.can_afford(i)), None)
            if slot is not None:
                card = opp.hand[slot]
                if engine.legal_deploy(Side.TOP, card, 9.0, 26.0):
                    events += engine.play_card(Side.TOP, slot, 9.0, 26.0)
                    dropped += 1
        # Withhold every deploy: perception saw nothing at all.
        unseen = {e["card"] for e in events if e.get("type") == "deploy"}
        track_from_events(t, events, engine.time, Side.TOP, drop=unseen)

    assert dropped > 0
    assert t.elixir != pytest.approx(opp.elixir, abs=1e-6)
    assert t.elixir > opp.elixir  # unspent in our model, actually spent in truth
