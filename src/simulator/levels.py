"""Scale card and tower stats to chosen Clash Royale levels.

Why this exists
---------------
`configs/cards.yaml` holds **level-1** values, and Clash Royale scales every
card on a per-level ladder. If every card in the game moved by the same
factor, the level would not matter at all: breakpoints are ratios, and
multiplying both sides changes nothing.

Two things break that convenience, and both are the reason this module
exists:

1. **Mixed levels.** A deck with a level-13 Hog Rider and a level-11
   Musketeer does not behave like an all-level-11 deck. Every
   spell-kills-troop threshold in that deck shifts. A policy trained at
   uniform level 1 has learned thresholds the player does not actually own.
2. **Rounding is per-card.** The ladders are nominally ~10% per level, but
   each is rounded independently — there are 59 distinct HP ladders across
   the roster. Low-HP cards are the worst: Skeletons at 32 HP step in
   visible jumps. So even a *uniform* level shift is not exactly a no-op.

Tower level is separate because the player's King level is separate in the
real game, and the tower-to-troop HP ratio is what decides how many hits a
win condition needs to connect.

How it scales
-------------
By **ratio against the ladder's own level-1 entry**, not by substituting the
ladder value outright. When `cards.yaml` matches the reference (the normal
case) the two are identical to the digit. When it deliberately differs —
a hand-tuned card, or one of the entries with no upstream match — the
deviation is preserved instead of being silently overwritten by a level
change. A level selection should never quietly undo a balance decision.

Only `hp`, `damage` and `spell_damage` scale. Hit speed, range, sight range,
speed and spawn count are level-invariant in the real game.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from src.simulator.cards import ArenaConfig, CardStats, TowerStats

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
REFERENCE_PATH = CONFIG_DIR / "reference_card_stats.json"

MIN_LEVEL = 1
MAX_LEVEL = 15  # real-game cap; ladders carry a few extra entries beyond it

# Used only for entries with no upstream ladder (heroes, death-spawn
# products, cards with no datamined match). ~10% per level compounding,
# which is the shape every real ladder follows before rounding.
_FALLBACK_GROWTH = 1.1


@dataclass(frozen=True)
class CardLevels:
    """Which level each card is played at, and the player's tower level."""

    default: int = 1
    tower: int = 1
    overrides: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in [("default", self.default), ("tower", self.tower)]:
            _check(value, label)
        for name, value in self.overrides.items():
            _check(value, f"overrides[{name!r}]")

    def level_of(self, card_name: str) -> int:
        """Level for one card. Evolutions inherit their base card's level,
        which is how the real game works — you upgrade the card, and the
        evolution rides along."""
        if card_name in self.overrides:
            return int(self.overrides[card_name])
        if card_name.endswith("_evo"):
            base = card_name[: -len("_evo")]
            if base in self.overrides:
                return int(self.overrides[base])
        return int(self.default)

    @property
    def is_uniform_level_one(self) -> bool:
        """True when this is a no-op — the state `cards.yaml` is already in."""
        return (self.default == 1 and self.tower == 1
                and all(int(v) == 1 for v in self.overrides.values()))

    def to_dict(self) -> dict:
        return {"default": self.default, "tower": self.tower,
                "overrides": {k: int(v) for k, v in sorted(self.overrides.items())}}

    @classmethod
    def from_dict(cls, raw: dict | None) -> "CardLevels":
        raw = raw or {}
        return cls(default=int(raw.get("default", 1)),
                   tower=int(raw.get("tower", raw.get("default", 1))),
                   overrides={str(k): int(v) for k, v in (raw.get("overrides") or {}).items()})


def _check(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"card level {label} must be an integer, got {value!r}")
    if not MIN_LEVEL <= value <= MAX_LEVEL:
        raise ValueError(
            f"card level {label}={value} is outside {MIN_LEVEL}-{MAX_LEVEL}")


def load_reference(path: Path | None = None) -> dict:
    path = path or REFERENCE_PATH
    if not path.exists():
        return {"cards": {}, "towers": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(ladder: list[int] | None, level: int) -> float | None:
    """Growth factor from level 1 to `level`, or None without a ladder."""
    if not ladder or not ladder[0]:
        return None
    index = min(level, len(ladder)) - 1
    return ladder[index] / ladder[0]


def growth(ladder: list[int] | None, level: int) -> float:
    """Ladder ratio, falling back to compounding growth when absent."""
    exact = _ratio(ladder, level)
    return exact if exact is not None else _FALLBACK_GROWTH ** (level - 1)


def scale_cards(
    cards: Mapping[str, CardStats],
    levels: CardLevels,
    reference: dict | None = None,
) -> dict[str, CardStats]:
    """Return `cards` with hp/damage/spell_damage scaled to `levels`."""
    if levels.is_uniform_level_one:
        return dict(cards)
    table = (reference or load_reference()).get("cards", {})
    out: dict[str, CardStats] = {}
    for name, card in cards.items():
        level = levels.level_of(name)
        if level == 1:
            out[name] = card
            continue
        ref = table.get(name) or table.get(
            name[: -len("_evo")] if name.endswith("_evo") else name) or {}
        hp_growth = growth(ref.get("hp_per_level"), level)
        dmg_growth = growth(ref.get("damage_per_level"), level)
        spell_growth = growth(ref.get("spell_damage_per_level"), level)
        out[name] = replace(
            card,
            hp=round(card.hp * hp_growth) if card.hp else card.hp,
            damage=round(card.damage * dmg_growth) if card.damage else card.damage,
            spell_damage=(round(card.spell_damage * spell_growth)
                          if card.spell_damage else card.spell_damage),
            # Death damage rides the attack ladder: it is damage the card
            # deals, so leaving it at level 1 would make Balloon and Giant
            # Skeleton quietly weaker at every level above 1.
            death_damage=(round(card.death_damage * dmg_growth)
                          if card.death_damage else card.death_damage),
        )
    return out


def scale_arena(
    arena: ArenaConfig,
    levels: CardLevels,
    reference: dict | None = None,
) -> ArenaConfig:
    """Return `arena` with both towers scaled to `levels.tower`."""
    if levels.tower == 1:
        return arena
    towers = (reference or load_reference()).get("towers", {})

    def scaled(stats: TowerStats, key: str) -> TowerStats:
        ref = towers.get(key, {})
        return TowerStats(
            hp=round(stats.hp * growth(ref.get("hp_per_level"), levels.tower)),
            damage=round(stats.damage * growth(ref.get("damage_per_level"), levels.tower)),
            hit_speed=stats.hit_speed,
            range=stats.range,
        )

    return replace(arena,
                   princess=scaled(arena.princess, "princess"),
                   king=scaled(arena.king, "king"))


def load_card_levels(raw: dict | None = None, path: Path | str | None = None) -> CardLevels:
    """Build from a training config's `card_levels:` block, or from a file."""
    if raw is not None:
        return CardLevels.from_dict(raw)
    if path is None:
        return CardLevels()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return CardLevels.from_dict(data.get("card_levels", data))


def describe(levels: CardLevels) -> str:
    """One-line summary for training logs and checkpoint provenance."""
    if levels.is_uniform_level_one:
        return "card levels: all level 1 (config values as-is)"
    extra = (f", overrides: " + " ".join(f"{k}={v}" for k, v in sorted(levels.overrides.items()))
             if levels.overrides else "")
    return f"card levels: default {levels.default}, towers {levels.tower}{extra}"
