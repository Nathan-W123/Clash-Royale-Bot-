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
# SAFE — resolvable to one unambiguous upstream number. These are also the
# fields breakpoints depend on, so they are what actually matters.
#
# `splash_radius`, `spell_radius` and `tower_multiplier` were initially in
# REVIEW because upstream splits an attack across a character row and its
# *projectile* row: `area_damage_radius` is 0 for Wizard and Bowler, and a
# spell's damage and crown-tower reduction can sit on either object.
# `distil()` now follows that link (see `_splash` and `_spell_row`), which
# resolves them exactly — and incidentally found that Witch's 1.0 splash was
# missing from this simulator entirely. The handful of cards whose geometry
# genuinely does not map are handled per-field by FIELD_EXCLUSIONS rather
# than by holding back the whole category.
#
# REVIEW — still genuinely ambiguous. `range`/`sight_range` differ by
# convention: upstream measures to the target's edge, this simulator adds
# both body radii in `BattleEngine.tick`, so the two are not the same
# quantity and a blind copy would silently change every engagement distance.
SAFE_UNIT_FIELDS = ("hp", "damage", "hit_speed", "speed", "count", "splash_radius")
REVIEW_UNIT_FIELDS = ("range", "sight_range")
SAFE_SPELL_FIELDS = ("spell_damage", "spell_radius", "tower_multiplier")
REVIEW_SPELL_FIELDS = ()

# Per-field opt-outs, for cards where one number does not map but the rest do.
FIELD_EXCLUSIONS: dict[str, frozenset[str]] = {
    # Arrows is a *volley*: upstream's 1.4 is one arrow's blast, and the
    # spread across the target area lives in the spawn pattern, not the
    # table. The flattened 4.0-tile area this simulator uses is correct;
    # its damage (48) is not affected and does sync.
    "arrows": frozenset({"spell_radius"}),
    # Rolling projectiles: a swept 1.95x0.6-tile rectangle travelling 10
    # tiles, which this simulator approximates as one circle. Not the same
    # shape, so the radius is a modelling choice rather than a fact.
    "log": frozenset({"spell_radius"}),
    "barbarian_barrel": frozenset({"spell_radius"}),
    # Chain lightning, not area damage: upstream carries no radius at all, so
    # syncing would silently turn Electro Dragon single-target. This
    # simulator's 2.0 splash is a deliberate stand-in for the chain.
    "electro_dragon": frozenset({"splash_radius"}),
    # Shotguns and piercing shots. Hunter fires 10 pellets of 0.07 radius
    # each; Magic Archer's arrow pierces along a line. Neither is a blast,
    # and copying the per-projectile body radius would imply one.
    "hunter": frozenset({"splash_radius"}),
    "magic_archer": frozenset({"splash_radius"}),
}

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


def _splash(row: dict, projectiles: dict) -> float:
    """Area-damage radius, following the character -> projectile link.

    A ranged splash unit carries its blast on the projectile it fires, so the
    character row reports `area_damage_radius: 0` for Wizard, Bomber, Bowler
    and friends. The projectile's `aoe_to_ground` / `aoe_to_air` flags are
    what distinguish a blast radius from a single-target projectile's own
    body — without that check, Musketeer and Princess would pick up splash
    they do not have.
    """
    direct = row.get("area_damage_radius") or 0
    if direct:
        return direct / 1000
    p = projectiles.get(row.get("projectile") or "")
    if p and (p.get("aoe_to_ground") or p.get("aoe_to_air")):
        return (p.get("radius") or 0) / 1000
    return 0.0


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
            "splash_radius": _splash(c, projectiles),
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
            "splash_radius": _splash(b, projectiles),
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
    """Flatten a spell across its own row and its projectile.

    A spell can be described entirely on one object (Zap: radius and crown
    reduction on the spell row) or split across two (Lightning: radius on the
    spell row, damage and crown reduction on `LighningSpell`). Each field is
    therefore resolved independently, taking whichever object actually
    carries it, rather than picking one row and hoping.
    """
    projectile = projectiles.get(row.get("projectile") or "") or {}

    def pick(field):
        value = row.get(field)
        return value if value else projectile.get(field)

    radius = (pick("radius") or 0) / 1000
    # Upstream states the *reduction* against crown towers as a negative
    # percentage. An absent value is not "no reduction": those rows carry
    # `deflect_behaviour: UseSpellsTowerDamageMul`, i.e. "apply the game-wide
    # spell multiplier", which this table does not contain. Reporting it as
    # 1.0 would hand Barbarian Barrel and Royal Delivery full tower damage.
    crown = pick("crown_tower_damage_percent")
    return {
        "kind": "spell",
        "spell_damage": _damage(row, projectiles),
        "spell_radius": float(radius),
        "tower_multiplier": round(1.0 + crown / 100.0, 3) if crown else None,
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

        excluded = FIELD_EXCLUSIONS.get(key, frozenset())
        changed, flagged = {}, {}
        for f in safe + review:
            if f in excluded or ref.get(f) is None:
                continue   # None = upstream has no value for it, not zero
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


def _render(value: float) -> str:
    return f"{value:g}" if value % 1 else str(int(value))


def apply(rows: list[dict]) -> int:
    """Rewrite the changed stat lines in place, preserving comments.

    Fields absent from a card's block are *appended* rather than skipped.
    That matters: a building like Mortar has no `splash_radius:` line at all,
    so a rewrite-only pass would silently drop the very corrections that add
    a mechanic the card was missing.
    """
    lines = CARDS_YAML.read_text(encoding="utf-8").split("\n")
    edits = {r["card"]: r["fields"] for r in rows if r["status"] == "differs"}
    if not edits:
        return 0

    out: list[str] = []
    current: str | None = None
    pending: dict[str, tuple[float, float]] = {}
    indent = "  "
    changed = 0

    def flush() -> None:
        """Append fields the block never declared, before leaving it."""
        nonlocal changed
        while out and not out[-1].strip():
            out.pop()          # keep the appended lines inside the block
        for name, (_, value) in pending.items():
            out.append(f"{indent}{name}: {_render(value)}")
            changed += 1
        pending.clear()

    for line in lines:
        header = re.match(r"^([a-z_0-9]+):\s*$", line)
        if header:
            if current in edits:
                flush()
                out.append("")
            current = header.group(1)
            pending = dict(edits.get(current, {}))
            out.append(line)
            continue

        field = re.match(r"^(\s+)([a-z_]+):\s*([^#\n]*?)(\s*#.*)?$", line)
        if field and current in edits and field.group(2) in pending:
            lead, name, _, comment = field.groups()
            indent = lead
            value = pending.pop(name)[1]
            out.append(f"{lead}{name}: {_render(value)}{comment or ''}")
            changed += 1
            continue
        out.append(line)

    if current in edits:
        flush()
        out.append("")

    CARDS_YAML.write_text("\n".join(out), encoding="utf-8")
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
