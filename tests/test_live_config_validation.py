"""Live-config calibration guards.

These target the failure mode that actually bit this config: coordinates
measured against one capture geometry left behind after switching to another,
sitting dormant because the active decision_mode ignored them.
"""
from __future__ import annotations

import textwrap

import pytest

from src.live.config import Rect, load_live_config


def _write(tmp_path, body: str):
    path = tmp_path / "live.yaml"
    path.write_text(textwrap.dedent(body))
    return path


BASE = """\
live:
  transport: desktop
  desktop_capture: window
  window_title: Clash Royale
  reference_size: [556, 1028]
  match_indicator: [160, 995, 350, 25]
  match_min_saturation: 0.5
  card_slots: [[167, 917], [278, 917], [386, 917], [492, 917]]
  card_ready_regions: [[140, 851, 55, 57], [250, 851, 55, 57], [358, 851, 55, 57], [464, 851, 55, 57]]
  decision_mode: dynamic_slots
  dynamic_target: [278, 550]
"""


def test_baseline_config_is_valid(tmp_path):
    cfg = load_live_config(_write(tmp_path, BASE))
    assert cfg.reference_size == (556, 1028)
    assert cfg.decision_mode == "dynamic_slots"


def test_rejects_placement_outside_reference_frame_even_when_mode_ignores_it(tmp_path):
    """The exact live bug: full-desktop x=1450 left in a 556-wide frame,
    unused under dynamic_slots, would misfire the moment mode changed."""
    body = BASE + "  placements:\n    giant: [1450, 770]\n"
    with pytest.raises(ValueError, match=r"placements\['giant'\].*outside reference_size"):
        load_live_config(_write(tmp_path, body))


def test_rejects_card_slot_outside_reference_frame(tmp_path):
    body = BASE.replace("[[167, 917], [278, 917], [386, 917], [492, 917]]",
                        "[[167, 917], [278, 917], [386, 917], [9999, 917]]")
    with pytest.raises(ValueError, match=r"card_slots\[3\].*outside reference_size"):
        load_live_config(_write(tmp_path, body))


def test_rejects_dynamic_target_outside_reference_frame(tmp_path):
    body = BASE.replace("dynamic_target: [278, 550]", "dynamic_target: [278, 5000]")
    with pytest.raises(ValueError, match=r"dynamic_target.*outside reference_size"):
        load_live_config(_write(tmp_path, body))


def test_rejects_rect_extending_past_reference_frame(tmp_path):
    body = BASE.replace("match_indicator: [160, 995, 350, 25]",
                        "match_indicator: [160, 995, 350, 500]")
    with pytest.raises(ValueError, match=r"match_indicator.*extends past reference_size"):
        load_live_config(_write(tmp_path, body))


def test_rejects_duplicate_keys(tmp_path):
    """YAML silently keeps the last duplicate; for a calibration file that
    means a stray second reference_size rescales every other coordinate."""
    body = BASE + "  reference_size: [1920, 1080]\n"
    with pytest.raises(ValueError, match="duplicate key 'reference_size'"):
        load_live_config(_write(tmp_path, body))


def test_rect_rejects_negative_origin():
    with pytest.raises(ValueError, match="must be non-negative"):
        Rect.from_values([-5, 10, 20, 20], "match_indicator")


def test_shipped_configs_are_valid():
    """Both tracked configs must satisfy the contract they document."""
    for path in ("configs/live_play.yaml", "configs/live_play.example.yaml"):
        cfg = load_live_config(path)
        assert cfg.reference_size[0] > 0


def test_known_deck_mode_still_validates_placements_against_frame(tmp_path):
    """In-frame placements are accepted under known_deck (the positive case,
    so the guard above isn't just rejecting everything)."""
    body = (
        BASE.replace("decision_mode: dynamic_slots", "decision_mode: known_deck")
        + "  preset_deck: [giant, musketeer, mini_pekka, minions, knight, goblins, fireball, cannon]\n"
        + "  opening_hand: [giant, musketeer, mini_pekka, minions]\n"
        + "  draw_order: [knight, goblins, fireball, cannon]\n"
        + "  placements:\n    giant: [278, 600]\n"
        + "  priority: [giant]\n"
    )
    cfg = load_live_config(_write(tmp_path, body))
    assert cfg.placements["giant"] == (278, 600)


def test_known_deck_rejects_a_config_that_could_never_play_anything(tmp_path):
    """Empty placements under known_deck is a silent no-op runner, not an
    error at runtime — so it has to be caught at load time."""
    body = (
        BASE.replace("decision_mode: dynamic_slots", "decision_mode: known_deck")
        + "  preset_deck: [giant, musketeer, mini_pekka, minions, knight, goblins, fireball, cannon]\n"
        + "  opening_hand: [giant, musketeer, mini_pekka, minions]\n"
        + "  draw_order: [knight, goblins, fireball, cannon]\n"
        + "  placements: {}\n"
        + "  priority: []\n"
    )
    with pytest.raises(ValueError, match="can never play anything"):
        load_live_config(_write(tmp_path, body))


def test_known_deck_rejects_priority_and_placements_that_do_not_overlap(tmp_path):
    body = (
        BASE.replace("decision_mode: dynamic_slots", "decision_mode: known_deck")
        + "  preset_deck: [giant, musketeer, mini_pekka, minions, knight, goblins, fireball, cannon]\n"
        + "  opening_hand: [giant, musketeer, mini_pekka, minions]\n"
        + "  draw_order: [knight, goblins, fireball, cannon]\n"
        + "  placements:\n    giant: [278, 600]\n"
        + "  priority: [musketeer]\n"  # names a card with no placement
    )
    with pytest.raises(ValueError, match="can never play anything"):
        load_live_config(_write(tmp_path, body))
