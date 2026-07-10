"""Adapters that let a policy network play anywhere a Bot can, and vice versa.

- PolicyBot: implements the src.bots.base.Bot protocol (decide(engine, side)),
  so trained checkpoints plug into match_runner, benchmark, and the league.
- BotOpponent / PolicyOpponent: implement the env Opponent protocol
  (act(env, side)) for opponents living inside CRBattleEnv during training.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.agent import masking, obs_layout
from src.agent.network import PolicyNetwork, make_network, masks_to_tensors, obs_to_tensors
from src.bots.base import Action, Bot
from src.simulator.cards import load_cards
from src.simulator.constants import PLACE_COLS, Side
from src.simulator.engine import BattleEngine


def save_checkpoint(net: PolicyNetwork, card_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "config": asdict(net.config),
        "card_names": list(card_names),
    }, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[PolicyNetwork, list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    n_cards = cfg.pop("n_cards")
    cfg["conv_channels"] = tuple(cfg["conv_channels"])
    net = make_network(n_cards, cfg)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["card_names"]


def policy_action(
    net: PolicyNetwork,
    engine: BattleEngine,
    side: Side,
    card_to_id: dict[str, int],
    deterministic: bool = False,
) -> tuple[int, int]:
    """(card_choice, cell) for the given engine state."""
    device = next(net.parameters()).device
    obs = obs_to_tensors(obs_layout.encode_obs(engine, side, card_to_id), device)
    m = masking.build_action_masks(engine, side)
    card = np.concatenate(([True], m["card"] & m["place"].any(axis=1)))
    masks = masks_to_tensors({"card": card, "place": m["place"]}, device)
    actions, _, _ = net.act(obs, masks, deterministic=deterministic)
    return int(actions[0, 0]), int(actions[0, 1])


class PolicyBot:
    """A trained policy exposed through the scripted-bot interface."""

    def __init__(self, net: PolicyNetwork, card_names: list[str],
                 name: str = "policy", deterministic: bool = False):
        self.net = net
        self.card_to_id = {n: i for i, n in enumerate(card_names)}
        self.name = name
        self.deterministic = deterministic

    @classmethod
    def load(cls, path: Path, name: str | None = None,
             deterministic: bool = False) -> "PolicyBot":
        net, card_names = load_checkpoint(Path(path))
        return cls(net, card_names, name=name or Path(path).stem,
                   deterministic=deterministic)

    def decide(self, engine: BattleEngine, side: Side) -> Action | None:
        choice, cell = policy_action(self.net, engine, side, self.card_to_id,
                                     deterministic=self.deterministic)
        if choice == 0:
            return None
        row, col = divmod(cell, PLACE_COLS)
        x, y = masking.cell_to_xy(side, col, row, engine.arena.height)
        return Action(choice - 1, x, y)

    # Also usable directly as an env opponent.
    def act(self, env, side: Side) -> tuple[int, float, float] | None:
        action = self.decide(env.engine, side)
        return None if action is None else (action.slot, action.x, action.y)


class BotOpponent:
    """Wrap a scripted Bot as an env Opponent."""

    def __init__(self, bot: Bot):
        self.bot = bot
        self.name = bot.name

    def act(self, env, side: Side) -> tuple[int, float, float] | None:
        action = self.bot.decide(env.engine, side)
        return None if action is None else (action.slot, action.x, action.y)


def default_card_names() -> list[str]:
    return list(load_cards().keys())
