"""Card stat definitions loaded from configs/cards.yaml."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import yaml

from src.simulator.constants import DEFAULT_DEPLOY_TIME, CardType, TargetType

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Alternate deck/catalog names → canonical card key in cards.yaml.
CARD_ALIASES: dict[str, str] = {
    "the_log": "log",
}


@dataclass(frozen=True)
class CardStats:
    name: str
    type: CardType
    cost: int
    # Troop / building fields (0 for spells)
    hp: float = 0.0
    damage: float = 0.0
    hit_speed: float = 1.0
    range: float = 0.5
    sight_range: float = 5.5
    speed: float = 0.0            # tiles/sec; buildings don't move
    targets: TargetType = TargetType.GROUND
    flying: bool = False
    count: int = 1                # units spawned per deploy (swarms)
    splash_radius: float = 0.0    # 0 = single target
    lifetime: float = 0.0         # buildings only; seconds until self-destruct
    # Seconds after placement before this card can target, move, or attack.
    # ~1s covers most of the roster; siege buildings (mortar, x_bow) and a few
    # slow-winding troops take materially longer, which is a real tempo cost
    # the agent should have to plan around. Override per card in cards.yaml.
    deploy_time: float = DEFAULT_DEPLOY_TIME
    # Unit-specific mechanics (#36). Zero/empty disables each one, so cards
    # that don't have the mechanic need no entry in cards.yaml.
    #   charge_*      Prince/Battle Ram/Dark Prince/Ram Rider, and the
    #                 Bandit/Royal Ghost dash: move `charge_distance` tiles
    #                 unimpeded, then hit harder and move faster until the
    #                 charge is spent or broken.
    #   ramp_up_*     Inferno Tower/Dragon: damage grows while locked on one
    #                 target and resets on retarget or stun.
    #   death_*       Lava Hound pups, Golem golemites, Balloon/Giant
    #                 Skeleton death damage.
    #   deploy_anywhere  Miner/Goblin Drill tunnel past placement rules.
    charge_distance: float = 0.0
    charge_damage_multiplier: float = 2.0
    charge_speed_multiplier: float = 1.6
    ramp_up_time: float = 0.0
    ramp_up_multiplier: float = 1.0
    death_spawn: str = ""
    death_spawn_count: int = 0
    death_damage: float = 0.0
    death_damage_radius: float = 0.0
    deploy_anywhere: bool = False
    # Spell fields
    spell_damage: float = 0.0
    spell_radius: float = 0.0
    spell_delay: float = 0.0
    tower_multiplier: float = 1.0
    stun_duration: float = 0.0    # zap/lightning: brief stun that resets wind-ups
    # Hero / evolution metadata
    is_hero: bool = False
    is_evolution: bool = False
    evolution_of: str = ""
    ability: str = ""             # dash | spawn | damage_buff | shield | taunt | split_on_death
    ability_charge: float = 0.0   # seconds on arena to reach full charge
    ability_damage: float = 0.0 # burst damage (dash) or multiplier (buff)
    ability_radius: float = 0.0
    ability_duration: float = 0.0
    ability_spawn: str = ""       # base card name for spawn ability
    ability_spawn_count: int = 0

    def __post_init__(self) -> None:
        if self.cost < 1:
            raise ValueError(f"{self.name}: cost must be >= 1")
        if self.is_hero and (not self.ability or self.ability_charge <= 0):
            raise ValueError(f"{self.name}: hero needs ability and ability_charge")
        if self.type == CardType.SPELL:
            if self.spell_damage <= 0 or self.spell_radius <= 0:
                raise ValueError(f"{self.name}: spell needs spell_damage and spell_radius")
        else:
            if self.hp <= 0 or self.damage <= 0:
                raise ValueError(f"{self.name}: troop/building needs hp and damage")
            if self.type == CardType.BUILDING and self.lifetime <= 0:
                raise ValueError(f"{self.name}: building needs lifetime")
            if self.sight_range < self.range:
                raise ValueError(f"{self.name}: sight_range must cover attack range")


def _parse_card(name: str, spec: dict) -> CardStats:
    spec = dict(spec)
    spec["type"] = CardType(spec["type"])
    if "targets" in spec:
        spec["targets"] = TargetType(spec["targets"])
    return CardStats(name=name, **spec)


def _load_yaml_cards(path: Path) -> dict[str, CardStats]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {name: _parse_card(name, spec) for name, spec in raw.items()}


@lru_cache(maxsize=8)
def _load_cards_cached(base_path: Path) -> dict[str, CardStats]:
    """Parse-once backing store for `load_cards`.

    `BattleEngine.__init__` calls `load_cards()`, so this ran on every engine
    construction — i.e. every episode reset, times every parallel env. Parsing
    ~1500 lines of YAML per reset was a large share of training wall-clock for
    data that never changes within a run.
    """
    cards = _load_yaml_cards(base_path)
    cards.update(_load_yaml_cards(CONFIG_DIR / "heroes.yaml"))

    evo_path = CONFIG_DIR / "evolutions.yaml"
    if evo_path.exists():
        overrides = yaml.safe_load(evo_path.read_text()) or {}
        for base_name, delta in overrides.items():
            if base_name not in cards:
                continue
            base = cards[base_name]
            evo_name = f"{base_name}_evo"
            cards[evo_name] = replace(
                base,
                name=evo_name,
                is_evolution=True,
                evolution_of=base_name,
                **delta,
            )

    for alias, canonical in CARD_ALIASES.items():
        if canonical in cards and alias not in cards:
            cards[alias] = replace(cards[canonical], name=alias)

    return cards


def load_cards(path: Path | None = None) -> dict[str, CardStats]:
    """Load base cards, heroes, and build evolution variants.

    Returns a fresh dict each call (CardStats itself is frozen, so the values
    are safe to share) — callers that mutate the mapping must not corrupt the
    cache behind it.
    """
    return dict(_load_cards_cached(path or CONFIG_DIR / "cards.yaml"))


def load_decks(path: Path | None = None) -> dict[str, list[str]]:
    path = path or CONFIG_DIR / "decks.yaml"
    raw = yaml.safe_load(path.read_text())
    raw.pop("pools", None)
    decks = {name: list(cards) for name, cards in raw.items()}
    decks.update(load_ladder_decks())
    return decks


def load_ladder_decks(path: Path | None = None) -> dict[str, list[str]]:
    """Load curated top-ladder decks from configs/ladder_decks.yaml."""
    path = path or CONFIG_DIR / "ladder_decks.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {spec["name"]: list(spec["cards"]) for spec in raw.get("decks", [])}


def ladder_deck_names(path: Path | None = None) -> list[str]:
    return list(load_ladder_decks(path).keys())


@dataclass(frozen=True)
class TowerStats:
    hp: float
    damage: float
    hit_speed: float
    range: float


@dataclass(frozen=True)
class ArenaConfig:
    width: float
    height: float
    river_y_min: float
    river_y_max: float
    bridge_xs: tuple[float, ...]
    bridge_half_width: float
    princess: TowerStats
    king: TowerStats
    princess_left_pos: tuple[float, float]
    princess_right_pos: tuple[float, float]
    king_pos: tuple[float, float]
    elixir_start: float
    elixir_max: float
    elixir_regen_interval: float
    double_time: float
    dt: float
    regulation: float
    overtime: float
    tiebreaker: str


@lru_cache(maxsize=8)
def load_arena(path: Path | None = None) -> ArenaConfig:
    """Cached: ArenaConfig is frozen, and this is re-read per engine build."""
    path = path or CONFIG_DIR / "arena.yaml"
    raw = yaml.safe_load(path.read_text())
    a, t, e, c = raw["arena"], raw["towers"], raw["elixir"], raw["clock"]
    return ArenaConfig(
        width=float(a["width"]),
        height=float(a["height"]),
        river_y_min=float(a["river_y_min"]),
        river_y_max=float(a["river_y_max"]),
        bridge_xs=tuple(float(x) for x in a["bridge_xs"]),
        bridge_half_width=float(a["bridge_half_width"]),
        princess=TowerStats(**{k: float(v) for k, v in t["princess"].items()}),
        king=TowerStats(**{k: float(v) for k, v in t["king"].items()}),
        princess_left_pos=tuple(t["positions"]["princess_left"]),
        princess_right_pos=tuple(t["positions"]["princess_right"]),
        king_pos=tuple(t["positions"]["king"]),
        elixir_start=float(e["start"]),
        elixir_max=float(e["max"]),
        elixir_regen_interval=float(e["regen_interval"]),
        double_time=float(e["double_time"]),
        dt=float(c["dt"]),
        regulation=float(c["regulation"]),
        overtime=float(c["overtime"]),
        tiebreaker=str(c["tiebreaker"]),
    )
