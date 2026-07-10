"""RL agent utilities: rewards, masking, observations."""
from __future__ import annotations

from src.agent.masking import (
    N_CELLS,
    action_mask_cards,
    action_mask_placement,
    build_action_masks,
    cell_index,
    cell_to_xy,
    frame_y,
    legal_card_slots,
    legal_placements,
)
from src.agent.obs_layout import (
    CARD_FEATURE_DIM,
    SPATIAL_CHANNELS,
    encode_hand,
    encode_spatial,
)
from src.agent.rewards import (
    RewardConfig,
    RewardShaper,
    anneal_factor,
    compute_step_reward,
    load_reward_config,
    terminal_reward,
)

__all__ = [
    "CARD_FEATURE_DIM",
    "N_CELLS",
    "RewardConfig",
    "RewardShaper",
    "SPATIAL_CHANNELS",
    "action_mask_cards",
    "action_mask_placement",
    "anneal_factor",
    "build_action_masks",
    "cell_index",
    "cell_to_xy",
    "compute_step_reward",
    "encode_hand",
    "encode_spatial",
    "frame_y",
    "legal_card_slots",
    "legal_placements",
    "load_reward_config",
    "terminal_reward",
]
