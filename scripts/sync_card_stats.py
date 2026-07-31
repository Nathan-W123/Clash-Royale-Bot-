"""Diff (and optionally sync) configs/cards.yaml against real Clash Royale stats.

Source of truth
---------------
`configs/reference_card_stats.json`, distilled from RoyaleAPI's `cr-api-data`
mirror of Supercell's card table:

    https://github.com/RoyaleAPI/cr-api-data  (json/cards_stats.json)

Regenerate it with `--refresh`, which downloads that file and re-distils.

Why level-1 values are the right target
---------------------------------------
Clash Royale scales every troop by one per-level curve, so level-1 values
preserve every troop-vs-troop breakpoint exactly — "does Arrows kill Minions"
has the same answer at level 1 and level 11. The simulator is therefore
*already at the right scale*, and this is confirmed independently:
`configs/arena.yaml`'s towers (princess 1400/50/0.8s/7.5, king 2400/50/1.0s/7.0)
match the datamined values to the digit.

That is what makes a global "rescale" the wrong operation. Rescaling multiplies
both sides of every breakpoint and changes nothing. What the audit actually
found (docs/SIM_FIDELITY.md) is per-card drift: a chunk of `cards.yaml` is
exactly right, and the rest was filled in by estimate and ranges from about
-74% to +277% off. Only a per-card correction fixes that, and only a real
source can supply one.

Usage
-----
    python -m scripts.sync_card_stats                # diff report (default)
    python -m scripts.sync_card_stats --apply        # rewrite cards.yaml
    python -m scripts.sync_card_stats --refresh      # re-download reference

`--apply` rewrites only the stat fields below. Everything hand-authored stays:
card `type`, `targets`, `flying`, `cost`, `lifetime`, `deploy_time`, and every
#36 mechanics field (charge, ramp, death effects, tunnelling, stun).

**Applying this is a balance change.** Every recorded benchmark win rate
predates it and is not comparable afterwards; re-benchmark the frozen bots
before reading anything into a number.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
REFERENCE = CONFIG_DIR / "reference_card_stats.json"
CARDS_YAML = CONFIG_DIR / "cards.yaml"
UPSTREAM = "https://royaleapi.github.io/cr-api-data/json/cards_stats.json"

# Fields this tool owns. Anything not listed is hand-authored and preserved.
#
# Split by how well the two schemas line up:
#
# SAFE — one upstream number, one meaning, same units. These are also the
# fields breakpoints depend on, so they are what actually matters.
#
# REVIEW — upstream splits across objects that this simulator flattens.
# A unit's splash lives on its *projectile*, not the character, so upstream
# reports `area_damage_radius: 0` for Wizard and Sparky; a spell's radius may
# be the projectile's own body rather than its area of effect (upstream gives
# Arrows a 1.4-tile radius, which is not its blast). Syncing these blindly
# would replace correct values with zeros and call it a fidelity improvement.
# Reported, never auto-applied, unless --include-review is passed.
SAFE_UNIT_FIELDS = ("hp", "damage", "hit_speed", "speed", "count")
REVIEW_UNIT_FIELDS = ("range", "sight_range", "splash_radius")
SAFE_SPELL_FIELDS = ("spell_damage",)
REVIEW_SPELL_FIELDS = ("spell_radius", "tower_multiplier")

UNIT_FIELDS = SAFE_UNIT_FIELDS + REVIEW_UNIT_FIELDS
SPELL_FIELDS = SAFE_SPELL_FIELDS + REVIEW_SPELL_FIELDS

# Cards whose upstream row does not describe the same entity our config does,
# so even the "safe" fields are wrong for them. Spawner buildings attack by
# summoning (upstream hit_speed 10s, sight 0); Rascals' row covers only the
# boy, not the pair of girls the card also summons.
SCHEMA_MISMATCH = frozenset({
    # Spawner buildings: upstream models the attack as summoning (hit_speed
    # 10s, sight 0), where this simulator gives them a nominal attack.
    "tombstone", "goblin_hut", "furnace", "barbarian_hut", "goblin_cage",
    # Delivery spells whose upstream row carries no damage — the payload is
    # a separate spawn. This simulator flattens them into a damage spell.
    "goblin_barrel", "skeleton_barrel", "goblin_drill",
    # Upstream row covers only part of the card: Rascals' boy without the two
    # girls; Princess and Firecracker keep their damage on their projectile.
    "rascals", "princess", "firecracker",
    # Upstream models these as spell projectiles; this simulator models them
    # as the little troops that carry them.
    "fire_spirits", "heal_spirit",
})

# `CardStats.__post_init__` requires these to be positive, so a zero coming
# out of the reference is always a mapping failure (the real value lives on a
# linked projectile or spawner row), never data. Refusing to write it turns a
# whole class of silent mis-syncs into a visible warning.
POSITIVE_FIELDS = frozenset({"hp", "damage", "spell_damage", "hit_speed", "count"})

# Spell cards whose stats live in the projectile table under a different name.
PROJECTILE_SPELLS = {
    "fireball": "FireballSpell",
    "arrows": "ArrowsSpell",
    "rocket": "RocketSpell",
    "log": "LogProjectileRolling",
    "giant_snowball": "SnowballSpell",
    "goblin_barrel": "GoblinBarrelSpell",
    "royal_delivery": "RoyalDeliveryProjectile",
    "fire_spirits": "FireSpiritsProjectile",
    "heal_spirit": "HealSpiritProjectile",
    "barbarian_barrel": "BarbLogProjectileRolling",
}

# Cards whose signature effect this simulator implements explicitly rather
# than as area damage (see src/simulator/spell_effects.py). Their upstream
# `damage` is 0 or an unrelated internal value, so syncing it would silently
# disarm them.
EFFECT_SPELLS = frozenset({"rage", "clone", "graveyard", "poison", "tornado",
                           "freeze", "earthquake"})


# ------------------------------------------------------------------ distil


def _damage(row: dict, projectiles: dict) -> float:
    dmg = row.get("damage") or 0
    if not dmg and row.get("projectile"):
        p = projectiles.get(row["projectile"])
        if p:
            dmg = p.get("damage") or 0
    return float(dmg)


def distil(raw: dict) -> dict:
    """Upstream dump -> the small table this project actually consumes."""
    chars = {c["name"]: c for c in raw["characters"] if c.get("name")}
    projectiles = {p["name"]: p for p in raw["projectile"] if p.get("name")}
    troops = {t["key"].replace("-", "_"): t for t in raw["troop"] if t.get("key")}
    buildings = {b["key"].replace("-", "_"): b for b in raw["building"] if b.get("key")}
    spells = {s["key"].replace("-", "_"): s for s in raw["spell"] if s.get("key")}

    out: dict[str, dict] = {}
    for key, troop in troops.items():
        c = chars.get(troop.get("summon_character") or troop.get("name"))
        if not c:
            continue
        out[key] = {
            "kind": "unit",
            "hp": float(c["hitpoints"]),
            "damage": _damage(c, projectiles),
            "hit_speed": (c.get("hit_speed") or 0) / 1000,
            "range": (c.get("range") or 0) / 1000,
            "sight_range": (c.get("sight_range") or 0) / 1000,
            # Upstream speed is in sixtieths of a tile per second.
            "speed": (c.get("speed") or 0) / 60.0,
            "count": troop.get("summon_number") or 1,
            "splash_radius": (c.get("area_damage_radius") or 0) / 1000,
        }
    for key, b in buildings.items():
        out[key] = {
            "kind": "unit",
            "hp": float(b.get("hitpoints") or 0),
            "damage": _damage(b, projectiles),
            "hit_speed": (b.get("hit_speed") or 0) / 1000,
            "range": (b.get("range") or 0) / 1000,
            "sight_range": (b.get("sight_range") or 0) / 1000,
            "speed": 0.0,
            "count": 1,
            "splash_radius": (b.get("area_damage_radius") or 0) / 1000,
        }
    for key, s in spells.items():
        row = _spell_row(s, projectiles)
        if row:
            out[key] = row
    for key, name in PROJECTILE_SPELLS.items():
        p = projectiles.get(name)
        if p:
            out[key] = _spell_row(p, projectiles)

    towers = {}
    for name, label in (("PrincessTower", "princess"), ("KingTower", "king")):
        row = next((b for b in raw["building"] if b.get("name") == name), None)
        if row:
            towers[label] = {
                "hp": float(row["hitpoints"]),
                "damage": _damage(row, projectiles),
                "hit_speed": (row.get("hit_speed") or 0) / 1000,
                "range": (row.get("range") or 0) / 1000,
            }
    return {"_source": UPSTREAM,
            "_note": "Level-1 base values. Troop scaling is uniform per level, "
                     "so these preserve every breakpoint.",
            "towers": towers,
            "cards": out}


def _spell_row(row: dict, projectiles: dict) -> dict:
    radius = (row.get("radius") or 0) / 1000
    damage = _damage(row, projectiles)
    if row.get("projectile"):
        p = projectiles.get(row["projectile"])
        if p and not radius:
            radius = (p.get("radius") or 0) / 1000
    crown = row.get("crown_tower_damage_percent") or 0
    return {
        "kind": "spell",
        "spell_damage": float(damage),
        "spell_radius": float(radius),
        # Upstream states the *reduction* against crown towers as a negative
        # percentage; the simulator stores the surviving fraction.
        "tower_multiplier": round(1.0 + crown / 100.0, 3),
    }


def refresh() -> dict:
    with urllib.request.urlopen(UPSTREAM, timeout=60) as fh:
        raw = json.load(fh)
    table = distil(raw)
    REFERENCE.write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")
    return table


def load_reference() -> dict:
    if not REFERENCE.exists():
        raise SystemExit(f"{REFERENCE} is missing; run with --refresh")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- diff


def diff(reference: dict, tolerance: float = 0.02, include_review: bool = False) -> list[dict]:
    """Per-card field differences between cards.yaml and the reference."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.simulator.cards import load_cards

    cards = load_cards()
    table = reference["cards"]
    rows = []
    for key, card in sorted(cards.items()):
        if card.is_evolution or card.is_hero:
            continue
        ref = table.get(key)
        if not ref:
            rows.append({"card": key, "status": "no-reference", "fields": {}})
            continue
        if key in EFFECT_SPELLS:
            rows.append({"card": key, "status": "effect-spell", "fields": {}})
            continue
        if key in SCHEMA_MISMATCH:
            rows.append({"card": key, "status": "schema-mismatch", "fields": {}})
            continue
        spell = ref["kind"] == "spell"
        safe = SAFE_SPELL_FIELDS if spell else SAFE_UNIT_FIELDS
        review = REVIEW_SPELL_FIELDS if spell else REVIEW_UNIT_FIELDS

        changed, flagged = {}, {}
        for f in safe + review:
            ours = float(getattr(card, f))
            theirs = float(ref[f])
            if abs(ours - theirs) <= tolerance * max(abs(theirs), 1.0):
                continue
            if f in POSITIVE_FIELDS and theirs <= 0:
                flagged[f] = (ours, theirs)   # mapping failure, never data
                continue
            if f in safe or include_review:
                changed[f] = (ours, theirs)
            else:
                flagged[f] = (ours, theirs)
        rows.append({"card": key,
                     "status": "ok" if not changed and not flagged else "differs",
                     "fields": changed, "review": flagged})
    return rows


def _fmt(fields: dict) -> str:
    return ", ".join(
        f"{f} {a:g}->{b:g} ({(a / b - 1) * 100:+.0f}%)" if b else f"{f} {a:g}->{b:g}"
        for f, (a, b) in fields.items())


def report(rows: list[dict], reference: dict) -> None:
    differs = [r for r in rows if r["status"] == "differs"]
    ok = [r for r in rows if r["status"] == "ok"]
    none = [r for r in rows if r["status"] == "no-reference"]
    effect = [r for r in rows if r["status"] == "effect-spell"]
    mismatch = [r for r in rows if r["status"] == "schema-mismatch"]
    n_safe = sum(len(r["fields"]) for r in differs)
    n_review = sum(len(r.get("review", {})) for r in differs)

    print(f"{len(ok)} cards match, {len(differs)} differ "
          f"({n_safe} auto-syncable fields, {n_review} needing review), "
          f"{len(none)} unreferenced, {len(effect)} effect-spells, "
          f"{len(mismatch)} schema mismatches\n")
    for r in differs:
        if r["fields"]:
            print(f"  {r['card']:<22} {_fmt(r['fields'])}")
        if r.get("review"):
            print(f"  {'':<22} review: {_fmt(r['review'])}")
    if none:
        print(f"\nno reference entry (left untouched): {', '.join(r['card'] for r in none)}")
    if mismatch:
        print(f"\nschema mismatch (never synced): {', '.join(r['card'] for r in mismatch)}")
    towers = reference.get("towers", {})
    if towers:
        print("\ntowers (compare against configs/arena.yaml):")
        for label, t in towers.items():
            print(f"  {label:<10} hp={t['hp']:g} damage={t['damage']:g} "
                  f"hit_speed={t['hit_speed']:g} range={t['range']:g}")


# ------------------------------------------------------------------ apply


def apply(rows: list[dict]) -> int:
    """Rewrite only the changed stat lines, in place, preserving comments."""
    text = CARDS_YAML.read_text(encoding="utf-8")
    lines = text.split("\n")
    edits = {r["card"]: r["fields"] for r in rows if r["status"] == "differs"}
    if not edits:
        return 0

    current: str | None = None
    changed = 0
    for i, line in enumerate(lines):
        header = re.match(r"^([a-z_0-9]+):\s*$", line)
        if header:
            current = header.group(1)
            continue
        if current not in edits:
            continue
        field = re.match(r"^(\s+)([a-z_]+):\s*([^#\n]*?)(\s*#.*)?$", line)
        if not field:
            continue
        indent, name, _, comment = field.groups()
        if name not in edits[current]:
            continue
        value = edits[current][name][1]
        rendered = f"{value:g}" if value % 1 else str(int(value))
        lines[i] = f"{indent}{name}: {rendered}{comment or ''}"
        changed += 1

    CARDS_YAML.write_text("\n".join(lines), encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download and re-distil the reference table")
    parser.add_argument("--apply", action="store_true",
                        help="rewrite cards.yaml (a balance change: re-benchmark after)")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--include-review", action="store_true",
                        help="also sync fields whose schemas don't cleanly align "
                             "(range, sight, splash, spell radius, tower multiplier)")
    args = parser.parse_args()

    reference = refresh() if args.refresh else load_reference()
    rows = diff(reference, args.tolerance, include_review=args.include_review)
    report(rows, reference)
    if args.apply:
        n = apply(rows)
        print(f"\nrewrote {n} stat lines in {CARDS_YAML}")
        print("Existing benchmark win rates are no longer comparable — "
              "re-run the frozen benchmark before reading anything into them.")


if __name__ == "__main__":
    main()
