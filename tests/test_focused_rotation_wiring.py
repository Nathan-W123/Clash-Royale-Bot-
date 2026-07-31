"""`focused_ladder` must actually drive focused rotation + adaptive decks.

Before this wiring the stage was named for behavior it did not have: the
FocusedRotationManager / AdaptiveDeckBuilder pair lived only in
`src/training/session.py`, which the PPO trainer never calls. These tests pin
the stage to the behavior its name promises.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agent.train import FocusedRotationState, make_setup_fn
from src.decks.catalog import DeckCatalog
from src.training.config import load_training_config
from src.training.curriculum import load_curriculum


@pytest.fixture()
def focused_stage():
    return next(s for s in load_curriculum() if s.name == "focused_ladder")


def test_focused_ladder_stage_is_flagged(focused_stage):
    assert focused_stage.focused_rotation is True


def test_other_stages_are_not_flagged():
    for stage in load_curriculum():
        if stage.name != "focused_ladder":
            assert stage.focused_rotation is False, stage.name


def test_setup_fn_serves_the_current_rotation_deck(focused_stage):
    """Every episode in the stage faces the rotation's current deck, rather
    than an independently sampled one."""
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    focused = FocusedRotationState(focused_stage, catalog, training_cfg)
    setup_fn = make_setup_fn(focused_stage, catalog, training_cfg, pool=None,
                             latest_bot=None, focused=focused)

    current = focused.rotation.current_opponent_deck()
    expected = [c.name for c in catalog.resolve(current)]
    rng = np.random.default_rng(0)
    for _ in range(5):
        _, opp_deck, _ = setup_fn(rng)
        assert [c.name for c in opp_deck] == expected


def test_recording_results_advances_the_rotation(focused_stage):
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    focused = FocusedRotationState(focused_stage, catalog, training_cfg)
    first = focused.rotation.current_opponent_deck()

    metrics = {"win": 1.0, "draw": 0.0, "crowns_for": 2,
               "elixir_spent": 40.0, "card_usage": {"knight": 3, "giant": 2}}
    for _ in range(training_cfg.focused_rotation.wins_per_deck):
        focused.record(metrics)

    assert focused.rotation.current_opponent_deck() != first
    assert focused.rotation.total_matches == training_cfg.focused_rotation.wins_per_deck


def test_setup_fn_follows_the_rotation_after_it_advances(focused_stage):
    """The opponent served by setup_fn tracks the rotation, so advancing
    actually changes what the agent trains against."""
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    focused = FocusedRotationState(focused_stage, catalog, training_cfg)
    setup_fn = make_setup_fn(focused_stage, catalog, training_cfg, pool=None,
                             latest_bot=None, focused=focused)
    rng = np.random.default_rng(0)

    _, before, _ = setup_fn(rng)
    metrics = {"win": 1.0, "draw": 0.0, "crowns_for": 2,
               "elixir_spent": 40.0, "card_usage": {"knight": 3}}
    for _ in range(training_cfg.focused_rotation.wins_per_deck):
        focused.record(metrics)
    _, after, _ = setup_fn(rng)

    assert [c.name for c in before] != [c.name for c in after]


def test_adaptive_builder_receives_card_usage(focused_stage):
    """Per-card scores must actually accumulate from episode metrics —
    otherwise the 'adaptive' deck never adapts."""
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    focused = FocusedRotationState(focused_stage, catalog, training_cfg)

    focused.record({"win": 1.0, "draw": 0.0, "crowns_for": 3,
                    "elixir_spent": 50.0, "card_usage": {"knight": 4}})
    scores = focused.builder.tracker.to_dict()
    assert scores["knight"]["plays"] == 4
    assert scores["knight"]["wins"] == 4


def test_agent_deck_is_the_adaptive_build(focused_stage):
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    focused = FocusedRotationState(focused_stage, catalog, training_cfg)
    setup_fn = make_setup_fn(focused_stage, catalog, training_cfg, pool=None,
                             latest_bot=None, focused=focused)
    agent_deck, _, _ = setup_fn(np.random.default_rng(0))
    assert [c.name for c in agent_deck] == \
        [c.name for c in focused.builder.current_deck()]
