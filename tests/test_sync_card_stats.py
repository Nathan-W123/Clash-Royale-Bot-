"""Tests for the card-stat sync tool (scripts/sync_card_stats.py).

The tool rewrites the balance file from an external table, so its failure
mode is silently writing *wrong* numbers into `cards.yaml` — which no other
test would catch, because the simulator would happily run with them. Each
test here pins one of the judgement calls that keeps that from happening.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import sync_card_stats as sync  # noqa: E402


# ------------------------------------------------------------- projectile link


def _projectiles():
    return {
        "WizardProjectile": {"radius": 1500, "damage": 133,
                             "aoe_to_ground": True, "aoe_to_air": True},
        "MusketeerProjectile": {"radius": 900, "damage": 103},
        "HunterPellet": {"radius": 70, "damage": 53, "aoe_to_ground": True},
    }


def test_splash_follows_the_character_to_projectile_link():
    """Upstream reports area_damage_radius 0 for ranged splash units; the
    blast lives on the projectile they fire."""
    wizard = {"area_damage_radius": 0, "projectile": "WizardProjectile"}
    assert sync._splash(wizard, _projectiles()) == pytest.approx(1.5)


def test_splash_prefers_a_value_on_the_character_itself():
    knight = {"area_damage_radius": 1300, "projectile": "WizardProjectile"}
    assert sync._splash(knight, _projectiles()) == pytest.approx(1.3)


def test_single_target_projectiles_do_not_become_splash():
    """The load-bearing guard: a single-target projectile still has a body
    `radius`, so without the aoe flags Musketeer and Princess would acquire
    splash they do not have."""
    musketeer = {"area_damage_radius": 0, "projectile": "MusketeerProjectile"}
    assert sync._splash(musketeer, _projectiles()) == 0.0


def test_missing_projectile_is_not_an_error():
    assert sync._splash({"area_damage_radius": 0, "projectile": "Nope"}, {}) == 0.0
    assert sync._splash({}, {}) == 0.0


# ------------------------------------------------------------------- spells


def test_spell_fields_resolve_across_both_rows():
    """Lightning keeps its radius on the spell row but its damage and crown
    reduction on the projectile, so each field is resolved independently."""
    projectiles = {"LighningSpell": {"damage": 660, "crown_tower_damage_percent": -70}}
    row = {"radius": 3500, "projectile": "LighningSpell"}
    out = sync._spell_row(row, projectiles)
    assert out["spell_radius"] == pytest.approx(3.5)
    assert out["spell_damage"] == pytest.approx(660)
    assert out["tower_multiplier"] == pytest.approx(0.30)


def test_crown_reduction_converts_to_a_surviving_fraction():
    out = sync._spell_row({"damage": 700, "radius": 2000,
                           "crown_tower_damage_percent": -75}, {})
    assert out["tower_multiplier"] == pytest.approx(0.25)


def test_absent_crown_reduction_is_unknown_not_full_damage():
    """Rows without an explicit reduction carry `UseSpellsTowerDamageMul`,
    i.e. 'apply the game-wide multiplier', which this table does not have.
    Reading the absence as 1.0 would hand Barbarian Barrel and Royal Delivery
    full tower damage."""
    out = sync._spell_row({"damage": 151, "radius": 1300}, {})
    assert out["tower_multiplier"] is None


def test_none_reference_values_are_skipped_by_the_diff(monkeypatch):
    reference = {"cards": {"knight": {"kind": "unit", "hp": 690.0, "damage": 79.0,
                                      "hit_speed": 1.2, "speed": 1.0, "count": 1,
                                      "splash_radius": None, "range": 1.2,
                                      "sight_range": 5.5}}}
    rows = sync.diff(reference)
    knight = next(r for r in rows if r["card"] == "knight")
    assert "splash_radius" not in knight["fields"]
    assert "splash_radius" not in knight.get("review", {})


# --------------------------------------------------------------- exclusions


def test_per_field_exclusions_are_honoured():
    reference = {"cards": {"arrows": {"kind": "spell", "spell_damage": 48.0,
                                      "spell_radius": 1.4,
                                      "tower_multiplier": 0.3}}}
    rows = sync.diff(reference)
    arrows = next(r for r in rows if r["card"] == "arrows")
    assert "spell_radius" not in arrows["fields"], "volley spread is not a blast radius"


def test_chain_and_shotgun_cards_keep_their_stand_in_splash():
    assert "splash_radius" in sync.FIELD_EXCLUSIONS["electro_dragon"]
    assert "splash_radius" in sync.FIELD_EXCLUSIONS["hunter"]
    assert "splash_radius" in sync.FIELD_EXCLUSIONS["magic_archer"]


def test_effect_spells_are_never_synced():
    reference = {"cards": {"rage": {"kind": "spell", "spell_damage": 0.0,
                                    "spell_radius": 3.0, "tower_multiplier": None}}}
    rage = next(r for r in sync.diff(reference) if r["card"] == "rage")
    assert rage["status"] == "effect-spell"
    assert rage["fields"] == {}


def test_zero_into_a_positive_field_is_treated_as_a_mapping_failure():
    """CardStats rejects hp/damage of 0, so a zero from upstream is always a
    mapping failure. It must be reported, never written."""
    reference = {"cards": {"knight": {"kind": "unit", "hp": 690.0, "damage": 0.0,
                                      "hit_speed": 1.2, "speed": 1.0, "count": 1,
                                      "splash_radius": 0.0, "range": 1.2,
                                      "sight_range": 5.5}}}
    knight = next(r for r in sync.diff(reference) if r["card"] == "knight")
    assert "damage" not in knight["fields"]
    assert "damage" in knight["review"]


# -------------------------------------------------------------------- apply


def _write(tmp_path, text):
    path = tmp_path / "cards.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_apply_rewrites_values_and_keeps_comments(tmp_path, monkeypatch):
    path = _write(tmp_path, "knight:\n  hp: 500\n  damage: 79   # tuned\n")
    monkeypatch.setattr(sync, "CARDS_YAML", path)
    changed = sync.apply([{"card": "knight", "status": "differs",
                           "fields": {"hp": (500.0, 690.0)}}])
    assert changed == 1
    text = path.read_text(encoding="utf-8")
    assert "hp: 690" in text
    assert "damage: 79   # tuned" in text


def test_apply_appends_fields_the_card_never_declared(tmp_path, monkeypatch):
    """Buildings like Mortar have no `splash_radius:` line at all, so a
    rewrite-only pass would drop exactly the corrections that add a missing
    mechanic."""
    path = _write(tmp_path, "mortar:\n  hp: 535\n  damage: 104\n\nknight:\n  hp: 690\n")
    monkeypatch.setattr(sync, "CARDS_YAML", path)
    changed = sync.apply([{"card": "mortar", "status": "differs",
                           "fields": {"splash_radius": (0.0, 2.0)}}])
    assert changed == 1
    import yaml
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["mortar"]["splash_radius"] == 2
    assert parsed["knight"]["hp"] == 690, "the next card must stay intact"


def test_apply_keeps_the_file_parseable_with_mixed_edits(tmp_path, monkeypatch):
    path = _write(tmp_path,
                  "a_card:\n  hp: 1\n  damage: 2\n\n"
                  "b_card:\n  hp: 3\n\n"
                  "c_card:\n  hp: 5\n  splash_radius: 0.0\n")
    monkeypatch.setattr(sync, "CARDS_YAML", path)
    sync.apply([
        {"card": "a_card", "status": "differs", "fields": {"hp": (1.0, 10.0)}},
        {"card": "b_card", "status": "differs", "fields": {"splash_radius": (0.0, 1.5)}},
        {"card": "c_card", "status": "differs", "fields": {"splash_radius": (0.0, 2.5)}},
    ])
    import yaml
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parsed["a_card"] == {"hp": 10, "damage": 2}
    assert parsed["b_card"] == {"hp": 3, "splash_radius": 1.5}
    assert parsed["c_card"] == {"hp": 5, "splash_radius": 2.5}


def test_apply_is_a_noop_without_changes(tmp_path, monkeypatch):
    original = "knight:\n  hp: 690\n"
    path = _write(tmp_path, original)
    monkeypatch.setattr(sync, "CARDS_YAML", path)
    assert sync.apply([{"card": "knight", "status": "ok", "fields": {}}]) == 0
    assert path.read_text(encoding="utf-8") == original


# ----------------------------------------------------------------- reference


def test_shipped_reference_is_present_and_has_provenance():
    table = sync.load_reference()
    assert table["_source"].startswith("https://")
    assert len(table["cards"]) > 100
    assert table["towers"]["princess"]["hp"] == 1400


def test_shipped_reference_agrees_with_the_arena_config():
    """Independent evidence that the simulator is at the right scale: the
    towers in arena.yaml match the datamined table exactly."""
    from src.simulator.cards import load_arena

    arena = load_arena()
    towers = sync.load_reference()["towers"]
    assert arena.princess.hp == towers["princess"]["hp"]
    assert arena.princess.damage == towers["princess"]["damage"]
    assert arena.king.hp == towers["king"]["hp"]
    assert arena.king.damage == towers["king"]["damage"]


def test_cards_yaml_has_no_outstanding_auto_syncable_drift():
    """The sync contract: every field the tool can resolve unambiguously is
    already applied. Regressions here mean someone hand-edited a stat that
    the real table disagrees with."""
    rows = sync.diff(sync.load_reference())
    outstanding = {r["card"]: r["fields"] for r in rows if r["fields"]}
    assert not outstanding, f"run `python -m scripts.sync_card_stats --apply`: {outstanding}"
