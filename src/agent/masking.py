"""Action masks for the factored card-then-placement policy."""
from __future__ import annotations

import numpy as np

from src.simulator.constants import HAND_SIZE, PLACE_COLS, PLACE_ROWS, CardType, Side
from src.simulator.engine import BattleEngine
from src.simulator.player import PlayerState

N_CELLS = PLACE_COLS * PLACE_ROWS


def frame_y(side: Side, y: float, arena_height: float) -> float:
    return y if side == Side.BOTTOM else arena_height - y


def cell_to_xy(side: Side, col: int, row: int, arena_height: float) -> tuple[float, float]:
    """Placement grid cell ``(col, row)`` in the acting player's frame -> arena coords."""
    x = 2.0 * col + 1.0
    y_frame = 2.0 * row + 1.0
    return x, frame_y(side, y_frame, arena_height)


def cell_index(col: int, row: int) -> int:
    return row * PLACE_COLS + col


def legal_card_slots(player: PlayerState) -> list[int]:
    return [slot for slot in range(HAND_SIZE) if player.can_afford(slot)]


def legal_placements(
    engine: BattleEngine,
    side: Side,
    card,
) -> list[tuple[int, int]]:
    """Legal ``(col, row)`` cells for deploying ``card`` from ``side``."""
    a = engine.arena
    cells: list[tuple[int, int]] = []
    for row in range(PLACE_ROWS):
        for col in range(PLACE_COLS):
            x, y = cell_to_xy(side, col, row, a.height)
            if engine.legal_deploy(side, card, x, y):
                cells.append((col, row))
    return cells


def action_mask_cards(player: PlayerState) -> np.ndarray:
    """Affordable hand slots, shape ``(HAND_SIZE,)`` bool."""
    mask = np.zeros(HAND_SIZE, dtype=bool)
    for slot in legal_card_slots(player):
        mask[slot] = True
    return mask


def action_mask_placement(engine: BattleEngine, side: Side, slot: int) -> np.ndarray:
    """Legal cells for hand ``slot``, shape ``(PLACE_COLS * PLACE_ROWS,)`` bool."""
    mask = np.zeros(N_CELLS, dtype=bool)
    player = engine.players[side]
    if not player.can_afford(slot):
        return mask
    for col, row in legal_placements(engine, side, player.hand[slot]):
        mask[cell_index(col, row)] = True
    return mask


def build_action_masks(engine: BattleEngine, side: Side) -> dict[str, np.ndarray]:
    """Convenience bundle: affordable slots + per-slot placement masks."""
    player = engine.players[side]
    card = action_mask_cards(player)
    place = np.zeros((HAND_SIZE, N_CELLS), dtype=bool)
    type_masks: dict[CardType, np.ndarray] = {}
    for slot in range(HAND_SIZE):
        if not card[slot]:
            continue
        ctype = player.hand[slot].type
        tm = type_masks.get(ctype)
        if tm is None:
            tm = action_mask_placement(engine, side, slot)
            type_masks[ctype] = tm
        if tm.any():
            place[slot] = tm
    return {"card": card, "place": place}
