"""Counter tables for champion-level reactive defense."""
from __future__ import annotations

# Threat card -> preferred counters (checked in order against hand).
COUNTERS: dict[str, list[str]] = {
    "giant": ["cannon", "mini_pekka", "tesla", "valkyrie"],
    "hog_rider": ["cannon", "tesla", "mini_pekka", "goblins"],
    "minions": ["arrows", "zap", "musketeer", "archers"],
    "goblins": ["zap", "arrows", "valkyrie"],
    "skeletons": ["zap", "arrows", "valkyrie"],
    "baby_dragon": ["musketeer", "archers", "minions"],
    "musketeer": ["fireball", "mini_pekka", "knight"],
    "valkyrie": ["knight", "mini_pekka", "musketeer"],
    "mini_pekka": ["goblins", "skeletons", "knight"],
}

WIN_CONDITIONS = {"giant", "hog_rider", "balloon"}  # balloon not in catalog yet


def pick_counter(threat_name: str, hand_names: set[str]) -> str | None:
    for counter in COUNTERS.get(threat_name, []):
        if counter in hand_names:
            return counter
    return None


def generic_counter(threat, player) -> int | None:
    """Stats-based fallback when the table has no entry for this threat.

    Returns a hand *slot*: best affordable troop/building by DPS + bulk,
    respecting air-targeting and preferring splash against swarms.
    """
    from src.simulator.constants import CardType, TargetType

    best_slot, best_score = None, 0.0
    for slot, c in enumerate(player.hand):
        if c.type == CardType.SPELL or not player.can_afford(slot):
            continue
        if c.type == CardType.TROOP and c.targets == TargetType.BUILDINGS_ONLY:
            continue  # never burn a win condition on defense
        if threat.flying and c.targets not in (TargetType.AIR, TargetType.BOTH):
            continue
        dps = c.damage * c.count / max(c.hit_speed, 0.1)
        score = dps + 0.3 * c.hp * c.count
        if threat.stats.count >= 3 and (c.splash_radius > 0 or c.count >= 3):
            score *= 1.8
        score /= 1.0 + c.cost
        if score > best_score:
            best_slot, best_score = slot, score
    return best_slot
