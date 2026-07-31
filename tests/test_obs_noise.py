"""Domain randomization (#32): degrade enemy detections, never own state."""
from __future__ import annotations

import numpy as np
import pytest

from src.agent.obs_layout import TIER_HUMAN, encode_spatial, unit_views
from src.agent.obs_noise import ObservationNoise, ObsNoiseConfig
from src.simulator.constants import Side
from tests.conftest import make_engine, spawn_unit


def _engine_with_units(cards, arena):
    engine = make_engine(cards, arena)
    for i in range(6):
        spawn_unit(engine, cards["knight"], Side.TOP, 3.0 + i * 2.0, 22.0)
        spawn_unit(engine, cards["archers"], Side.BOTTOM, 3.0 + i * 2.0, 8.0)
    return engine


def _noise(**kwargs):
    return ObservationNoise(ObsNoiseConfig(enabled=True, **kwargs), seed=0)


def test_disabled_config_is_identity(cards, arena):
    engine = _engine_with_units(cards, arena)
    views = unit_views(engine, Side.BOTTOM)
    noise = ObservationNoise(ObsNoiseConfig(), seed=0)
    assert noise.perturb(views, engine, Side.BOTTOM) is views


@pytest.mark.parametrize("kwargs", [
    {"jitter_tiles": 1.0},
    {"p_miss": 0.9},
    {"p_false_positive": 0.9},
    {"p_identity_confusion": 1.0},
    {"p_occlusion": 1.0, "occlusion_radius": 20.0},
    {"hp_error_frac": 0.5},
])
def test_own_units_and_all_towers_are_never_perturbed(cards, arena, kwargs):
    """Own state comes from the player's own UI and the deterministic hand
    cycle; towers are static and unmissable. Noise there models an error that
    does not exist and makes the policy timid."""
    engine = _engine_with_units(cards, arena)
    views = unit_views(engine, Side.BOTTOM)
    before = [v for v in views if v.friendly or v.is_tower]
    noise = _noise(**kwargs)
    for _ in range(20):
        out = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
        after = [v for v in out if v.friendly or v.is_tower]
        assert after == before


def test_missed_detections_drop_enemies(cards, arena):
    engine = _engine_with_units(cards, arena)
    noise = _noise(p_miss=1.0)
    out = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    assert not [v for v in out if not v.friendly and not v.is_tower]


def test_jitter_moves_enemies_but_keeps_them_in_the_arena(cards, arena):
    engine = _engine_with_units(cards, arena)
    noise = _noise(jitter_tiles=1.0)
    out = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    enemies = [v for v in out if not v.friendly and not v.is_tower]
    original = {(v.x, v.y) for v in unit_views(engine, Side.BOTTOM) if not v.friendly}
    assert any((v.x, v.y) not in original for v in enemies)
    assert all(0.0 <= v.x < arena.width and 0.0 <= v.y < arena.height for v in enemies)


def test_jitter_is_applied_before_grid_binning(cards, arena):
    """Jitter must change which *cell* a unit lands in — i.e. it happens in
    tile space, not as a smear over the finished grid."""
    engine = _engine_with_units(cards, arena)
    clean = encode_spatial(engine, Side.BOTTOM)
    noise = _noise(jitter_tiles=2.0)
    moved = False
    for _ in range(20):
        dirty = encode_spatial(
            engine, Side.BOTTOM,
            noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM))
        # Enemy troop density channel only.
        if not np.array_equal(clean[1], dirty[1]):
            moved = True
            break
    assert moved


def test_false_positives_add_phantom_enemies(cards, arena):
    engine = _engine_with_units(cards, arena)
    noise = _noise(p_false_positive=1.0, max_false_positives=2)
    out = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    enemies = [v for v in out if not v.friendly and not v.is_tower]
    assert len(enemies) == 6 + 2
    assert all(0.0 <= v.x < arena.width for v in enemies)


def test_identity_confusion_swaps_for_a_real_card(cards, arena):
    engine = _engine_with_units(cards, arena)
    noise = _noise(p_identity_confusion=1.0)
    out = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    enemies = [v for v in out if not v.friendly and not v.is_tower]
    assert all(v.card in cards for v in enemies)
    assert any(v.card != "knight" for v in enemies)


def test_occlusion_drops_a_contiguous_patch(cards, arena):
    engine = make_engine(cards, arena)
    for i in range(4):
        spawn_unit(engine, cards["knight"], Side.TOP, 4.0 + i * 0.3, 22.0)
    spawn_unit(engine, cards["knight"], Side.TOP, 16.0, 22.0)  # far away
    noise = _noise(p_occlusion=1.0, occlusion_radius=2.0)
    out = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    enemies = [v for v in out if not v.friendly and not v.is_tower]
    assert len(enemies) == 1
    assert enemies[0].x == pytest.approx(16.0)


def test_staleness_reuses_the_previous_frame(cards, arena):
    engine = _engine_with_units(cards, arena)
    noise = _noise(p_stale=1.0)
    first = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    for u in engine.units:
        if u.side == Side.TOP:
            u.y -= 5.0
    second = noise.perturb(unit_views(engine, Side.BOTTOM), engine, Side.BOTTOM)
    stale = [v for v in second if not v.friendly and not v.is_tower]
    fresh = [v for v in first if not v.friendly and not v.is_tower]
    assert [(v.x, v.y) for v in stale] == [(v.x, v.y) for v in fresh]


def test_config_from_dict_rejects_typos():
    with pytest.raises(ValueError):
        ObsNoiseConfig.from_dict({"p_mis": 0.1})


def test_config_from_dict_defaults_to_enabled():
    cfg = ObsNoiseConfig.from_dict({"p_miss": 0.2})
    assert cfg.enabled is True
    assert cfg.p_miss == 0.2
    assert ObsNoiseConfig.from_dict(None).enabled is False
    assert ObsNoiseConfig.from_dict({"enabled": False, "p_miss": 0.2}).enabled is False


def test_env_applies_noise_only_when_configured(cards, arena, decks):
    from src.simulator.env import CRBattleEnv

    deck = [cards[n] for n in decks["training_mirror"]]
    clean = CRBattleEnv(cards, arena, deck, list(deck), tier=TIER_HUMAN, seed=1)
    clean.reset(seed=1)
    for i in range(5):
        spawn_unit(clean.engine, cards["knight"], Side.TOP, 3.0 + i, 22.0)
    baseline = clean.build_obs(Side.BOTTOM)["spatial"]

    clean.obs_noise = _noise(p_miss=1.0)
    degraded = clean.build_obs(Side.BOTTOM)["spatial"]
    assert baseline[1].any()
    assert not degraded[1].any()
    np.testing.assert_array_equal(baseline[0], degraded[0])  # own troops intact
