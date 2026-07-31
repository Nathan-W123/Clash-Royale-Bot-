"""Runtime entities: units (troops + buildings), towers, pending spells."""
from __future__ import annotations

from dataclasses import dataclass

from src.simulator.cards import CardStats, TowerStats
from src.simulator.constants import CardType, Side, TargetType

UNIT_RADIUS = 0.5
TOWER_RADIUS = 1.0


@dataclass
class Unit:
    id: int
    stats: CardStats
    side: Side
    x: float
    y: float
    hp: float
    cooldown: float               # seconds until next attack allowed
    elixir_value: float           # card cost / spawn count, for trade ledger
    target_id: int | None = None
    expires_at: float = 0.0       # buildings: sim time of self-destruct; 0 = never
    radius: float = UNIT_RADIUS
    ability_charge: float = 0.0   # 0–1 progress toward next hero ability
    damage_multiplier: float = 1.0
    buff_until: float = 0.0       # sim time when damage/speed buff expires
    shield_hp: float = 0.0        # temporary HP buffer from shield ability
    shield_until: float = 0.0     # sim time when shield expires
    speed_multiplier: float = 1.0
    frozen_until: float = 0.0     # sim time when a freeze/stun effect expires
    deployed_at: float = 0.0      # sim time this unit was spawned
    # Charge/ramp state (#36). Both are "wind-ups" that a stun resets, which
    # is the mechanic behind zapping a Sparky or an Inferno Tower.
    charge_progress: float = 0.0  # tiles moved unimpeded toward a target
    charged: bool = False         # charge complete; next hit is the charged one
    ramp_time: float = 0.0        # seconds locked onto `ramp_target_id`
    ramp_target_id: int | None = None

    @property
    def is_building(self) -> bool:
        return self.stats.type == CardType.BUILDING

    @property
    def is_hero(self) -> bool:
        return self.stats.is_hero

    @property
    def flying(self) -> bool:
        return self.stats.flying

    def can_target(self, flying: bool, is_structure: bool) -> bool:
        t = self.stats.targets
        if t == TargetType.BUILDINGS_ONLY:
            return is_structure
        if flying:
            return t in (TargetType.AIR, TargetType.BOTH)
        return t in (TargetType.GROUND, TargetType.BOTH)


@dataclass
class Tower:
    id: int
    kind: str                     # 'princess_left' | 'princess_right' | 'king'
    side: Side
    x: float
    y: float
    hp: float
    stats: TowerStats
    cooldown: float = 0.0
    target_id: int | None = None
    activated: bool = False       # king starts False until triggered; always set explicitly at construction
    radius: float = TOWER_RADIUS

    @property
    def is_king(self) -> bool:
        return self.kind == "king"


@dataclass
class PendingSpell:
    side: Side                    # caster
    x: float
    y: float
    radius: float
    damage: float
    tower_multiplier: float
    resolve_at: float             # sim time
    card_name: str
    ticks_left: int = 0           # remaining applications (0 = one-shot spell)
    zone_until: float = 0.0       # rage: sim time the buff zone ends (0 = unset)
