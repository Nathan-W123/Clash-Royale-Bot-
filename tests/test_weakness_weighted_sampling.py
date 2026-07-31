"""Scripted-opponent sampling in the real PPO loop should oversample whichever
archetype the matchup tracker says the agent is currently losing to most —
this is what's supposed to retarget training at a specific blind spot,
verified against the actual `make_setup_fn` used by src.agent.train, not a
reimplementation of the sampling logic."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.agent.train import make_setup_fn
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.training.config import load_training_config
from src.training.curriculum import load_curriculum
from src.training.matchup_tracker import MatchupTracker


@pytest.fixture()
def stage():
    # selfplay=False so make_setup_fn's `scripted()` path is always taken —
    # a selfplay stage would mostly return the (here, None) latest/pool bot.
    return next(s for s in load_curriculum() if s.name == "one_lane")


def _sample_many(stage, catalog, training_cfg, tracker, n=4000, seed=0):
    setup_fn = make_setup_fn(stage, catalog, training_cfg, pool=_EmptyPool(),
                             latest_bot=None, matchup_tracker=tracker)
    rng = np.random.default_rng(seed)
    counts = Counter()
    for _ in range(n):
        _, _, opponent = setup_fn(rng)
        counts[opponent.name] += 1
    return counts


class _EmptyPool:
    def members(self):
        return []


def test_uniform_without_a_tracker(stage):
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    counts = _sample_many(stage, catalog, training_cfg, tracker=None)
    display_names = {
        get_bot(n, catalog=catalog, rng=np.random.default_rng(0),
               skill_tier=training_cfg.opponents.skill_tier).name
        for n in stage.opponents  # ("rusher", "control") for one_lane
    }
    fracs = [counts[d] / sum(counts.values()) for d in display_names]
    # Two bots, no bias: each should land near 50%, not wildly skewed.
    for f in fracs:
        assert 0.40 < f < 0.60


def test_oversamples_the_archetype_the_agent_is_losing_to(stage):
    catalog = DeckCatalog()
    training_cfg = load_training_config()
    skill_tier = training_cfg.opponents.skill_tier
    rusher_name = get_bot("rusher", catalog=catalog, rng=np.random.default_rng(0),
                          skill_tier=skill_tier).name
    control_name = get_bot("control", catalog=catalog, rng=np.random.default_rng(0),
                           skill_tier=skill_tier).name

    tracker = MatchupTracker()
    # Simulated history: the agent is crushed by rusher, dominant vs control.
    for _ in range(50):
        tracker.record(rusher_name, won=False)
    for _ in range(50):
        tracker.record(control_name, won=True)

    counts = _sample_many(stage, catalog, training_cfg, tracker=tracker)
    total = sum(counts.values())
    rusher_frac = counts[rusher_name] / total
    control_frac = counts[control_name] / total
    # weakness_weight=0.35 (configs/training.yaml) against a 100%-loss-rate
    # history gives weight 1.35 vs 1.0 -> a theoretical ~57/43 split, not a
    # dramatic skew — it's a deliberately mild bias knob, not a hard switch.
    assert rusher_frac > control_frac
    assert rusher_frac > 0.52
