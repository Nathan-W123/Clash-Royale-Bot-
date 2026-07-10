"""Tests for focused rotation curriculum."""
from __future__ import annotations

from src.training.focused_curriculum import FocusedRotationConfig, FocusedRotationManager


def test_advances_after_wins_per_deck():
    mgr = FocusedRotationManager(
        ["deck_a", "deck_b"],
        FocusedRotationConfig(
            wins_per_deck=3,
            target_win_rate=0.65,
            min_matches_per_deck=10,
            min_win_rate_per_deck=0.55,
        ),
    )
    assert mgr.current_opponent_deck() == "deck_a"
    for _ in range(3):
        mgr.record_result(True)
    assert mgr.current_opponent_deck() == "deck_b"
    assert mgr.wins_vs_current == 0


def test_advances_after_min_matches_and_win_rate():
    mgr = FocusedRotationManager(
        ["deck_a", "deck_b"],
        FocusedRotationConfig(
            wins_per_deck=10,
            target_win_rate=0.65,
            min_matches_per_deck=4,
            min_win_rate_per_deck=0.5,
        ),
    )
    for _ in range(3):
        mgr.record_result(True)
    assert mgr.current_opponent_deck() == "deck_a"
    mgr.record_result(False)  # 3/4 = 75%
    assert mgr.current_opponent_deck() == "deck_b"


def test_cycles_and_stops_at_target_win_rate():
    decks = ["d1", "d2"]
    mgr = FocusedRotationManager(
        decks,
        FocusedRotationConfig(
            wins_per_deck=1,
            target_win_rate=0.65,
            min_matches_per_deck=1,
            min_win_rate_per_deck=0.5,
        ),
    )
    # First cycle: 2 wins / 2 matches = 100%
    mgr.record_result(True)
    mgr.record_result(True)
    assert mgr.cycle == 1
    assert mgr.should_continue_training() is False


def test_continues_when_below_target_after_cycle():
    mgr = FocusedRotationManager(
        ["d1", "d2"],
        FocusedRotationConfig(
            wins_per_deck=1,
            target_win_rate=0.65,
            min_matches_per_deck=1,
            min_win_rate_per_deck=0.5,
        ),
    )
    mgr.record_result(True)
    mgr.record_result(False)  # 50% overall after cycle
    assert mgr.should_continue_training() is True
