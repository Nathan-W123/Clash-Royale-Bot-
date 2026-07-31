"""Checkpoint round-trip and PolicyBot behavior for both obs schemas."""
from __future__ import annotations

from pathlib import Path

import torch

from src.agent.network import make_network
from src.agent.selfplay import PolicyBot, load_checkpoint, save_checkpoint
from src.simulator.constants import Side
from tests.conftest import make_engine

N_CARDS = 16
REAL_CHECKPOINT = Path("checkpoints/full_pool_60m.pt")


def test_restricted_checkpoint_round_trip_and_decide(cards, arena, tmp_path):
    net = make_network(N_CARDS, {"use_spatial": False})
    card_names = [f"card_{i}" for i in range(N_CARDS)]
    path = tmp_path / "restricted.pt"
    save_checkpoint(net, card_names, path)

    loaded, loaded_names = load_checkpoint(path)
    assert loaded.config.use_spatial is False
    assert loaded_names == card_names

    bot = PolicyBot(loaded, card_names, name="restricted-test", deterministic=True)
    engine = make_engine(cards, arena)
    action = bot.decide(engine, Side.BOTTOM)  # must not raise; None (no-op) is valid
    assert action is None or hasattr(action, "slot")


def test_old_checkpoint_without_new_keys_loads_with_full_defaults(tmp_path):
    """Simulate a checkpoint saved before use_spatial/scalar_dim existed."""
    net = make_network(N_CARDS)
    old_style_config = {
        k: v for k, v in vars(net.config).items()
        if k not in ("use_spatial", "scalar_dim")
    }
    path = tmp_path / "old_style.pt"
    torch.save({
        "state_dict": net.state_dict(),
        "config": old_style_config,
        "card_names": [f"card_{i}" for i in range(N_CARDS)],
    }, path)

    loaded, _ = load_checkpoint(path)
    assert loaded.config.use_spatial is True
    assert loaded.config.scalar_dim == 17


def test_real_frozen_checkpoint_still_loads_and_decides(cards, arena):
    if not REAL_CHECKPOINT.exists():
        return  # optional artifact; skip if this clone doesn't have it
    bot = PolicyBot.load(REAL_CHECKPOINT, deterministic=True)
    assert bot.net.config.use_spatial is True
    engine = make_engine(cards, arena)
    action = bot.decide(engine, Side.BOTTOM)
    assert action is None or hasattr(action, "slot")
