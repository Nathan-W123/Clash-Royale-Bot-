"""Observation layout for the policy network (numpy, no torch).

Observation tiers
-----------------
Which facts the encoder is allowed to read is a *scope* decision, not a
performance one — see CLAUDE.md's "On-Screen Visual Perception". Three tiers:

===========  ============  ==============  ==============  ===========
tier         spatial grid  own hand/elix.  opponent elix.  legal live?
===========  ============  ==============  ==============  ===========
`full`       real          yes             **yes**         no (cheats)
`human`      real          yes             no              yes
`restricted` zero-filled   yes             no              yes
===========  ============  ==============  ==============  ===========

`full` reads the opponent's exact elixir, which no player can see, so it is
simulator-only — useful as a privileged critic or as the #37 distillation
teacher. `human` is the live-play target: everything a skilled human
perceives from the screen and nothing more. `restricted` is the fallback for
when vision is unavailable or untrusted; it sees no enemy troops at all,
which makes the environment a severe POMDP.

The scalar widths are *not* ordered by tier — ``RESTRICTED_SCALAR_DIM (18) >
SCALAR_DIM (17)`` because restricted drops opponent elixir but adds four
per-tower alive flags. Code that assumes a width ordering (or that
restricted ⊂ full) produces silently misaligned tensors; always route
through `scalar_dim_for`.

Tensor shapes
-------------
``encode_spatial(engine, side)`` -> ``(SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)``
    Coarse 9x16 grid (2x2 arena tiles per cell). Channels:

    0  friendly troop HP density
    1  enemy troop HP density
    2  friendly building HP density
    3  enemy building HP density
    4  friendly tower HP
    5  enemy tower HP
    6  friendly pending spell damage
    7  enemy pending spell damage
    8  friendly unit presence
    9  enemy unit presence

``encode_units(engine, side, card_to_id)`` -> ``(MAX_ENTITIES, UNIT_FEATURE_DIM)``
    Padded permutation-invariant entity list for the set encoder (#39); see
    `UNIT_FEATURES` for the column layout.

``encode_hand(player, card_to_id)`` -> ``(CARD_FEATURE_DIM,)`` float32
    ``[hand_id_0, ..., hand_id_3, next_id, elixir / 10]``
    Unknown cards map to index 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulator.constants import HAND_SIZE, PLACE_COLS, PLACE_ROWS, Side
from src.simulator.engine import BattleEngine
from src.simulator.player import PlayerState

from src.agent.masking import frame_y

SPATIAL_CHANNELS = 10
CARD_FEATURE_DIM = HAND_SIZE + 2  # 4 hand slots + next card + elixir scalar

# --------------------------------------------------------------------- tiers

TIER_FULL = "full"
TIER_HUMAN = "human"
TIER_RESTRICTED = "restricted"
TIERS = (TIER_FULL, TIER_HUMAN, TIER_RESTRICTED)

# Each width is stated explicitly rather than derived from another tier: the
# tiers differ by more than one field in each direction, so any "full minus
# one" arithmetic would be wrong the moment a field moves.
SCALAR_DIM = 17
HUMAN_SCALAR_DIM = 20
RESTRICTED_SCALAR_DIM = 18

_SCALAR_DIMS = {
    TIER_FULL: SCALAR_DIM,
    TIER_HUMAN: HUMAN_SCALAR_DIM,
    TIER_RESTRICTED: RESTRICTED_SCALAR_DIM,
}


def resolve_tier(tier) -> str:
    """Normalize a tier name, tolerating the deprecated `use_spatial` bool.

    `use_spatial` predates the `human` tier and conflated two independent
    questions ("is there a spatial grid?" and "may I read opponent elixir?").
    It is still accepted — old checkpoints store it in their saved config and
    must keep loading — mapping True -> `full`, False -> `restricted`.
    """
    if tier is None:
        return TIER_FULL
    if isinstance(tier, bool) or isinstance(tier, np.bool_):
        return TIER_FULL if tier else TIER_RESTRICTED
    name = str(tier).lower()
    if name not in _SCALAR_DIMS:
        raise ValueError(f"unknown observation tier {tier!r}; expected one of {TIERS}")
    return name


def scalar_dim_for(tier) -> int:
    return _SCALAR_DIMS[resolve_tier(tier)]


def tier_uses_spatial(tier) -> bool:
    """True when the tier gets a real spatial grid (`full` and `human`)."""
    return resolve_tier(tier) != TIER_RESTRICTED


def tier_sees_opponent_elixir(tier) -> bool:
    """True only for `full`, the simulator-only/critic tier."""
    return resolve_tier(tier) == TIER_FULL


# ---------------------------------------------------------------- entity view


@dataclass
class UnitView:
    """One perceivable entity, in arena coordinates.

    Deliberately decoupled from `Unit`/`Tower`: this is the record a vision
    pipeline would produce and the only thing the domain-randomization
    wrapper (#32) is allowed to perturb, so both the grid rasterizer and the
    set encoder (#39) consume the same intermediate.
    """

    card: str
    x: float
    y: float
    hp: float
    max_hp: float
    friendly: bool
    is_building: bool = False
    is_tower: bool = False
    flying: bool = False
    # Kept apart rather than folded into one "inert" flag: both stop the unit
    # acting, but they end differently. A deploy lock always expires on its
    # own; a freeze can be extended by another spell, and a unit that is
    # merely deploying is a *future* threat while a frozen one is a current
    # opportunity. A policy that cannot tell them apart cannot time either.
    deploying: bool = False
    frozen: bool = False
    shielded: bool = False

    @property
    def inert(self) -> bool:
        return self.deploying or self.frozen


def unit_views(engine: BattleEngine, side: Side) -> list[UnitView]:
    """Units and towers currently on the arena, from ``side``'s perspective."""
    views: list[UnitView] = []
    for u in engine.units:
        if u.hp <= 0:
            continue
        views.append(UnitView(
            card=u.stats.name, x=u.x, y=u.y, hp=u.hp, max_hp=max(u.stats.hp, 1.0),
            friendly=u.side == side, is_building=u.is_building, flying=u.flying,
            deploying=engine._is_deploying(u), frozen=engine._is_frozen(u),
            shielded=u.shield_hp > 0,
        ))
    for t in engine.towers:
        if t.hp <= 0:
            continue
        views.append(UnitView(
            card=t.kind, x=t.x, y=t.y, hp=t.hp, max_hp=max(t.stats.hp, 1.0),
            friendly=t.side == side, is_building=True, is_tower=True,
        ))
    return views


# --------------------------------------------------------------- spatial grid


def _grid_cell(side: Side, x: float, y: float, arena_height: float) -> tuple[int, int] | None:
    fy = frame_y(side, y, arena_height)
    col = int(x // 2)
    row = int(fy // 2)
    if 0 <= col < PLACE_COLS and 0 <= row < PLACE_ROWS:
        return col, row
    return None


def encode_spatial(
    engine: BattleEngine,
    side: Side,
    views: list[UnitView] | None = None,
) -> np.ndarray:
    """Multi-channel coarse arena grid from ``side``'s perspective.

    ``views`` overrides what is rasterized — the hook the domain-
    randomization wrapper uses to degrade *enemy* detections before binning,
    so jitter lands in tile space rather than smearing a finished grid.
    """
    spatial = np.zeros((SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS), np.float32)
    h = engine.arena.height
    views = unit_views(engine, side) if views is None else views

    for v in views:
        cell = _grid_cell(side, v.x, v.y, h)
        if cell is None:
            continue
        col, row = cell
        if v.is_tower:
            spatial[4 if v.friendly else 5, row, col] += v.hp / 1000.0
            continue
        ch = (0 if v.friendly else 1) + (2 if v.is_building else 0)
        spatial[ch, row, col] += v.hp / 1000.0
        spatial[8 if v.friendly else 9, row, col] += 0.2

    for s in engine.spells:
        cell = _grid_cell(side, s.x, s.y, h)
        if cell is None:
            continue
        col, row = cell
        ch = 6 if s.side == side else 7
        rad = int(np.ceil(s.radius / 2.0))
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < PLACE_ROWS and 0 <= cc < PLACE_COLS:
                    spatial[ch, rr, cc] += s.damage / 1000.0

    np.clip(spatial, 0.0, 30.0, out=spatial)
    return spatial


# ------------------------------------------------------------------ unit set

# Column layout of the set-encoder entity matrix. Column 0 holds a *card id*
# that the network casts to long and feeds through its card embedding; the
# rest are already-normalized floats.
UNIT_FEATURES = (
    "card_id", "present", "x", "y", "hp_frac", "hp_scaled",
    "friendly", "building", "tower", "flying", "deploying", "shielded",
    "frozen",
)
UNIT_FEATURE_DIM = len(UNIT_FEATURES)
MAX_ENTITIES = 40


def encode_units(
    engine: BattleEngine,
    side: Side,
    card_to_id: dict[str, int],
    views: list[UnitView] | None = None,
) -> np.ndarray:
    """``(MAX_ENTITIES, UNIT_FEATURE_DIM)`` padded entity matrix.

    Rows past the entity count are all-zero, so column 1 (`present`) doubles
    as the padding mask — no separate mask key, which keeps the observation
    dict a flat mapping of same-dtype arrays.

    Overflow past `MAX_ENTITIES` keeps the entities *nearest the acting
    player's own king*, because the ones about to hit something matter more
    than a straggler at the far end of the arena.
    """
    out = np.zeros((MAX_ENTITIES, UNIT_FEATURE_DIM), np.float32)
    views = unit_views(engine, side) if views is None else views
    a = engine.arena
    if len(views) > MAX_ENTITIES:
        views = sorted(views, key=lambda v: frame_y(side, v.y, a.height))[:MAX_ENTITIES]
    for i, v in enumerate(views):
        out[i] = (
            float(card_to_id.get(v.card, 0)),
            1.0,
            v.x / a.width,
            frame_y(side, v.y, a.height) / a.height,
            min(1.0, v.hp / v.max_hp),
            min(30.0, v.hp / 1000.0),
            float(v.friendly),
            float(v.is_building),
            float(v.is_tower),
            float(v.flying),
            float(v.deploying),
            float(v.shielded),
            float(v.frozen),
        )
    return out


# -------------------------------------------------------------------- scalars


def _card_id(card, card_to_id: dict[str, int]) -> float:
    return float(card_to_id.get(card.name, 0))


def encode_hand(player: PlayerState, card_to_id: dict[str, int]) -> np.ndarray:
    """Hand card ids, next card id, and normalized elixir."""
    ids = [_card_id(c, card_to_id) for c in player.hand]
    ids.append(_card_id(player.next_card, card_to_id))
    ids.append(player.elixir / 10.0)
    return np.asarray(ids, dtype=np.float32)


def _king(engine: BattleEngine, side: Side):
    return next(t for t in engine.towers if t.side == side and t.is_king)


def _tower_alive(engine: BattleEngine, side: Side, kind: str) -> float:
    return float(any(t.side == side and t.kind == kind and t.hp > 0 for t in engine.towers))


def _time_remaining_frac(engine: BattleEngine) -> float:
    return max(0.0, engine.regulation - engine.time) / max(engine.regulation, 1.0)


def _princess_fraction(engine: BattleEngine, side: Side) -> float:
    return sum(1 for t in engine.towers
               if t.side == side and not t.is_king and t.hp > 0) / 2.0


def _hand_scalars(engine: BattleEngine, side: Side) -> list[float]:
    me = engine.players[side]
    return ([me.elixir / 10.0]
            + [me.hand[i].cost / 10.0 for i in range(HAND_SIZE)]
            + [float(me.can_afford(i)) for i in range(HAND_SIZE)])


def _clock_scalars(engine: BattleEngine, side: Side) -> list[float]:
    return [_time_remaining_frac(engine),
            float(engine.double_elixir), float(engine.overtime),
            float(_king(engine, side).activated),
            float(_king(engine, side.other).activated)]


def _tower_alive_scalars(engine: BattleEngine, side: Side) -> list[float]:
    other = side.other
    return [_tower_alive(engine, side, "princess_left"),
            _tower_alive(engine, side, "princess_right"),
            _tower_alive(engine, other, "princess_left"),
            _tower_alive(engine, other, "princess_right")]


def _encode_scalars_full(engine: BattleEngine, side: Side) -> np.ndarray:
    """``(SCALAR_DIM,)`` float32:
    [own_elixir, opp_elixir, cost_0..3, affordable_0..3, time_remaining,
     double_elixir, overtime, own_king_activated, enemy_king_activated,
     own_princess_count/2, enemy_princess_count/2]

    Includes the opponent's exact elixir count, which is only meaningful for
    simulator self-play (both sides' ground truth is known) — never valid
    for a live match, where the opponent's elixir isn't observable. See
    ``_encode_scalars_human`` for the live-legal variant that keeps the
    spatial grid, and ``_encode_scalars_restricted`` for the no-vision one.
    """
    opp = engine.players[side.other]
    head = _hand_scalars(engine, side)
    return np.asarray(
        [head[0], opp.elixir / 10.0] + head[1:]
        + _clock_scalars(engine, side)
        + [_princess_fraction(engine, side), _princess_fraction(engine, side.other)],
        dtype=np.float32)


def _encode_scalars_human(engine: BattleEngine, side: Side) -> np.ndarray:
    """``(HUMAN_SCALAR_DIM,)`` float32 — `full` minus the opponent-elixir
    leak, plus the per-tower alive flags `restricted` carries:
    [own_elixir, cost_0..3, affordable_0..3, time_remaining, double_elixir,
     overtime, own_king_activated, enemy_king_activated,
     own_princess_count/2, enemy_princess_count/2,
     own_princess_left_alive, own_princess_right_alive,
     enemy_princess_left_alive, enemy_princess_right_alive]

    Every field here is something a player reads off their own screen. The
    princess *counts* and the per-tower *flags* are redundant with each
    other; both are kept so this tier is a strict superset of what the other
    two legally see, which is what makes it a drop-in student for a `full`
    teacher.
    """
    return np.asarray(
        _hand_scalars(engine, side)
        + _clock_scalars(engine, side)
        + [_princess_fraction(engine, side), _princess_fraction(engine, side.other)]
        + _tower_alive_scalars(engine, side),
        dtype=np.float32)


def _encode_scalars_restricted(engine: BattleEngine, side: Side) -> np.ndarray:
    """``(RESTRICTED_SCALAR_DIM,)`` float32, using only information honestly
    observable in a live match (no opponent elixir, no aggregate tower-HP):
    [own_elixir, cost_0..3, affordable_0..3, time_remaining, double_elixir,
     overtime, own_king_activated, enemy_king_activated,
     own_princess_left_alive, own_princess_right_alive,
     enemy_princess_left_alive, enemy_princess_right_alive]
    """
    return np.asarray(
        _hand_scalars(engine, side)
        + _clock_scalars(engine, side)
        + _tower_alive_scalars(engine, side),
        dtype=np.float32)


_SCALAR_ENCODERS = {
    TIER_FULL: _encode_scalars_full,
    TIER_HUMAN: _encode_scalars_human,
    TIER_RESTRICTED: _encode_scalars_restricted,
}


def encode_scalars(
    engine: BattleEngine,
    side: Side,
    tier=TIER_FULL,
    *,
    use_spatial: bool | None = None,
) -> np.ndarray:
    """Match-state scalars for `tier`. See the per-tier `_encode_scalars_*`."""
    if use_spatial is not None:
        tier = use_spatial
    return _SCALAR_ENCODERS[resolve_tier(tier)](engine, side)


def encode_obs(
    engine: BattleEngine,
    side: Side,
    card_to_id: dict[str, int],
    tier=TIER_FULL,
    *,
    use_spatial: bool | None = None,
    views: list[UnitView] | None = None,
    with_units: bool = False,
) -> dict[str, np.ndarray]:
    """Canonical policy observation: spatial grid + hand card ids + scalars.

    The single encoding shared by the gym env, BC data generation, and
    policy-backed opponents/bots — keep them identical.

    For `restricted` the ``"spatial"`` grid is zero-filled (never computed —
    a network with a matching config never reads it) and ``"vector"`` drops
    to the narrow live-safe schema.

    ``views`` supplies an already-perturbed entity list (see
    `src.agent.obs_noise`); ``with_units`` additionally emits the padded
    ``"units"`` matrix consumed by the set encoder.
    """
    tier = resolve_tier(use_spatial if use_spatial is not None else tier)
    player = engine.players[side]
    ids = [int(card_to_id.get(c.name, 0)) for c in player.hand]
    ids.append(int(card_to_id.get(player.next_card.name, 0)))

    spatial_tier = tier_uses_spatial(tier)
    if spatial_tier and (views is not None or with_units):
        views = unit_views(engine, side) if views is None else views

    spatial = (encode_spatial(engine, side, views) if spatial_tier
               else np.zeros((SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS), np.float32))
    obs = {
        "spatial": spatial,
        "cards": np.asarray(ids, dtype=np.int64),
        "vector": encode_scalars(engine, side, tier),
    }
    if with_units:
        obs["units"] = (encode_units(engine, side, card_to_id, views) if spatial_tier
                        else np.zeros((MAX_ENTITIES, UNIT_FEATURE_DIM), np.float32))
    return obs
