"""BC data collection matches the observation schema it's collected for."""
from __future__ import annotations

import src.agent.bc as bc_module
from src.agent.bc import collect_bot_matches
from src.agent.obs_layout import RESTRICTED_SCALAR_DIM, SCALAR_DIM


def test_collect_bot_matches_restricted_vector_width():
    data = collect_bot_matches(n_matches=2, seed=0, use_spatial=False)
    arrays = data.arrays()
    assert arrays["vector"].shape[-1] == RESTRICTED_SCALAR_DIM
    assert not arrays["spatial"].any()


def test_collect_bot_matches_full_vector_width_unchanged():
    data = collect_bot_matches(n_matches=2, seed=0)
    arrays = data.arrays()
    assert arrays["vector"].shape[-1] == SCALAR_DIM


def test_focus_opponent_biases_the_top_seat_toward_that_archetype(monkeypatch):
    """collect_bot_matches calls get_bot(bot_b_name, ...) then
    get_bot(bot_t_name, ...) exactly once each, in that order, every match —
    so every second recorded name is the TOP-seat (opponent) choice."""
    requested = []
    original_get_bot = bc_module.get_bot

    def spy(name, **kwargs):
        requested.append(name)
        return original_get_bot(name, **kwargs)

    monkeypatch.setattr(bc_module, "get_bot", spy)
    collect_bot_matches(n_matches=60, seed=0, focus_opponent="rusher", focus_weight=0.9)

    top_seat_names = requested[1::2]
    rusher_frac = top_seat_names.count("rusher") / len(top_seat_names)
    assert rusher_frac > 0.7  # close to the configured 0.9, allowing for RNG noise


def test_no_focus_opponent_preserves_original_round_robin(monkeypatch):
    requested = []
    original_get_bot = bc_module.get_bot

    def spy(name, **kwargs):
        requested.append(name)
        return original_get_bot(name, **kwargs)

    monkeypatch.setattr(bc_module, "get_bot", spy)
    collect_bot_matches(n_matches=10, seed=0)

    top_seat_names = requested[1::2]
    # Original fixed round robin: pairs[(i*3+1) % 5] for i in 0..9.
    expected = ["control", "champion", "siege", "rusher", "beatdown"] * 2
    assert top_seat_names == expected
