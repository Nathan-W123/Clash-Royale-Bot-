"""Checkpoints written before the tier rework must keep loading.

`save_checkpoint` stores `asdict(net.config)` and `load_checkpoint` pops
`n_cards` and hands the rest to `make_network`. Every new `NetworkConfig`
field therefore needs a safe default, and every *removed* key (here:
`use_spatial`) needs to stay readable — otherwise every checkpoint on disk
breaks at once, silently in the case of a mis-sized scalar vector.
"""
from __future__ import annotations

import torch

from src.agent.network import make_network
from src.agent.obs_layout import (
    HUMAN_SCALAR_DIM,
    RESTRICTED_SCALAR_DIM,
    SCALAR_DIM,
    TIER_FULL,
    TIER_HUMAN,
    TIER_RESTRICTED,
)
from src.agent.selfplay import load_checkpoint, save_checkpoint

N_CARDS = 12
CARD_NAMES = [f"card_{i}" for i in range(N_CARDS)]
SMALL = {"conv_channels": (4,), "cnn_out": 16, "fusion_mlp": 32}


def _write_legacy(tmp_path, config: dict, state_from):
    """A checkpoint in the pre-tier format: `use_spatial`, no `tier`."""
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": state_from.state_dict(),
                "config": config,
                "card_names": CARD_NAMES}, path)
    return path


def test_new_checkpoints_round_trip(tmp_path):
    for tier, dim in ((TIER_FULL, SCALAR_DIM), (TIER_HUMAN, HUMAN_SCALAR_DIM),
                      (TIER_RESTRICTED, RESTRICTED_SCALAR_DIM)):
        net = make_network(N_CARDS, dict(SMALL, tier=tier))
        path = tmp_path / f"{tier}.pt"
        save_checkpoint(net, CARD_NAMES, path)
        loaded, names = load_checkpoint(path)
        assert loaded.config.tier == tier
        assert loaded.config.scalar_dim == dim
        assert names == CARD_NAMES


def test_legacy_full_checkpoint_loads_as_full_tier(tmp_path):
    net = make_network(N_CARDS, dict(SMALL, tier=TIER_FULL))
    path = _write_legacy(tmp_path, {
        "n_cards": N_CARDS, "conv_channels": (4,), "cnn_out": 16,
        "card_embed_dim": 16, "hand_mlp": 64, "fusion_mlp": 32,
        "use_spatial": True, "scalar_dim": SCALAR_DIM,
    }, net)
    loaded, _ = load_checkpoint(path)
    assert loaded.config.tier == TIER_FULL
    assert loaded.config.scalar_dim == SCALAR_DIM
    assert loaded.config.use_spatial is True


def test_legacy_restricted_checkpoint_loads_as_restricted_tier(tmp_path):
    net = make_network(N_CARDS, dict(SMALL, tier=TIER_RESTRICTED))
    path = _write_legacy(tmp_path, {
        "n_cards": N_CARDS, "conv_channels": (4,), "cnn_out": 16,
        "card_embed_dim": 16, "hand_mlp": 64, "fusion_mlp": 32,
        "use_spatial": False, "scalar_dim": RESTRICTED_SCALAR_DIM,
    }, net)
    loaded, _ = load_checkpoint(path)
    assert loaded.config.tier == TIER_RESTRICTED
    assert loaded.config.scalar_dim == RESTRICTED_SCALAR_DIM
    assert loaded.config.use_spatial is False


def test_unknown_config_keys_do_not_crash_make_network():
    """A checkpoint from a future (or concurrent) branch carrying a field this
    build has never heard of must still load rather than take the run down."""
    net = make_network(N_CARDS, dict(SMALL, tier=TIER_HUMAN,
                                     some_future_knob=17, another="yes"))
    assert net.config.tier == TIER_HUMAN


def test_missing_config_keys_fall_back_to_defaults():
    net = make_network(N_CARDS, {})
    assert net.config.tier == TIER_FULL
    assert net.config.scalar_dim == SCALAR_DIM


def test_saved_config_no_longer_contains_the_deprecated_key(tmp_path):
    """`use_spatial` is a property now, so `asdict` must not write it back —
    otherwise it would shadow `tier` on the next load."""
    net = make_network(N_CARDS, dict(SMALL, tier=TIER_HUMAN))
    path = tmp_path / "h.pt"
    save_checkpoint(net, CARD_NAMES, path)
    config = torch.load(path, weights_only=False)["config"]
    assert "tier" in config
    assert "use_spatial" not in config
