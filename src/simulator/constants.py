"""Enums and structural constants for the simulator.

Numeric balance values live in configs/*.yaml, not here.
"""
from __future__ import annotations

import enum


class Side(enum.IntEnum):
    BOTTOM = 0
    TOP = 1

    @property
    def other(self) -> "Side":
        return Side(1 - self)


class CardType(enum.StrEnum):
    TROOP = "troop"
    SPELL = "spell"
    BUILDING = "building"


class TargetType(enum.StrEnum):
    GROUND = "ground"
    AIR = "air"
    BOTH = "both"
    BUILDINGS_ONLY = "buildings_only"


class MatchResult(enum.IntEnum):
    ONGOING = -1
    DRAW = 0
    BOTTOM_WIN = 1
    TOP_WIN = 2

    @staticmethod
    def win_for(side: Side) -> "MatchResult":
        return MatchResult.BOTTOM_WIN if side == Side.BOTTOM else MatchResult.TOP_WIN


# Retarget when a locked target moves beyond this multiple of sight range.
LOCK_BREAK_FACTOR = 1.5

# Tolerance for accumulated floating-point error in time/cooldown comparisons.
TIME_EPS = 1e-9

# Placement grid used by the action space: arena downsampled 2x2.
PLACE_COLS = 9   # 18 / 2
PLACE_ROWS = 16  # 32 / 2

HAND_SIZE = 4

# Seconds a freshly-deployed troop/building sits inert before it can target,
# move, or attack — the real game's post-deploy animation lock. This is only
# the fallback for cards that don't override it; per-card values live in
# cards.yaml as `deploy_time`, because the spread matters tactically (a
# mortar that is useful ~1s after landing is a very different card from one
# that takes several seconds to wind up).
DEFAULT_DEPLOY_TIME = 1.0
