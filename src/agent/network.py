"""Policy/value network: CNN spatial encoder + hand embedding, fused MLP,
factored autoregressive action head (card choice -> placement | card).

Action = (card_choice, cell):
  card_choice 0 = no-op, 1..4 = hand slot; cell in the 9x16 placement grid.
The placement head is conditioned on the embedding of the *card* in the
chosen slot, and masked with that slot's legality mask. Joint log-prob is
log p(card) + log p(cell | card) (no-op contributes only the card term).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.agent.obs_layout import SCALAR_DIM, SPATIAL_CHANNELS
from src.simulator.constants import HAND_SIZE, PLACE_COLS, PLACE_ROWS

N_CELLS = PLACE_COLS * PLACE_ROWS
N_CARD_CHOICES = HAND_SIZE + 1

NEG_INF = -1e9


@dataclass(frozen=True)
class NetworkConfig:
    n_cards: int
    conv_channels: tuple[int, ...] = (32, 64, 64)
    cnn_out: int = 256
    card_embed_dim: int = 16
    hand_mlp: int = 64
    fusion_mlp: int = 256


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Numpy obs (single or batched) -> batched torch tensors."""
    spatial = torch.as_tensor(obs["spatial"], dtype=torch.float32, device=device)
    cards = torch.as_tensor(obs["cards"], dtype=torch.long, device=device)
    vector = torch.as_tensor(obs["vector"], dtype=torch.float32, device=device)
    if spatial.dim() == 3:
        spatial, cards, vector = spatial[None], cards[None], vector[None]
    return {"spatial": spatial, "cards": cards, "vector": vector}


def masks_to_tensors(masks: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    card = torch.as_tensor(masks["card"], dtype=torch.bool, device=device)
    place = torch.as_tensor(masks["place"], dtype=torch.bool, device=device)
    if card.dim() == 1:
        card, place = card[None], place[None]
    return {"card": card, "place": place}


class PolicyNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        chans = [SPATIAL_CHANNELS, *config.conv_channels]
        convs: list[nn.Module] = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            convs += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
        convs += [nn.Flatten()]
        self.cnn = nn.Sequential(*convs)
        cnn_flat = config.conv_channels[-1] * PLACE_ROWS * PLACE_COLS
        self.cnn_proj = nn.Sequential(nn.Linear(cnn_flat, config.cnn_out), nn.ReLU())

        self.card_embed = nn.Embedding(config.n_cards, config.card_embed_dim)
        hand_in = (HAND_SIZE + 1) * config.card_embed_dim + SCALAR_DIM
        self.hand_mlp = nn.Sequential(nn.Linear(hand_in, config.hand_mlp), nn.ReLU())

        fused_in = config.cnn_out + config.hand_mlp
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, config.fusion_mlp), nn.ReLU(),
            nn.Linear(config.fusion_mlp, config.fusion_mlp), nn.ReLU(),
        )
        self.card_head = nn.Linear(config.fusion_mlp, N_CARD_CHOICES)
        self.place_head = nn.Sequential(
            nn.Linear(config.fusion_mlp + config.card_embed_dim, config.fusion_mlp), nn.ReLU(),
            nn.Linear(config.fusion_mlp, N_CELLS),
        )
        self.value_head = nn.Linear(config.fusion_mlp, 1)

    # ------------------------------------------------------------- trunk

    def trunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial = self.cnn_proj(self.cnn(obs["spatial"]))
        emb = self.card_embed(obs["cards"]).flatten(1)
        hand = self.hand_mlp(torch.cat([emb, obs["vector"]], dim=1))
        return self.fusion(torch.cat([spatial, hand], dim=1))

    def card_logits(self, feat: torch.Tensor, card_mask: torch.Tensor) -> torch.Tensor:
        logits = self.card_head(feat)
        return logits.masked_fill(~card_mask, NEG_INF)

    def place_logits(
        self,
        feat: torch.Tensor,
        obs_cards: torch.Tensor,
        card_choice: torch.Tensor,
        place_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Placement logits conditioned on the chosen slot's card embedding.

        For no-op rows the conditioning/mask are dummies; callers must zero
        out their contribution to log-probs/entropy.
        """
        slot = (card_choice - 1).clamp(min=0)
        chosen_ids = obs_cards.gather(1, slot[:, None]).squeeze(1)
        cond = torch.cat([feat, self.card_embed(chosen_ids)], dim=1)
        logits = self.place_head(cond)
        slot_mask = place_mask.gather(
            1, slot[:, None, None].expand(-1, 1, N_CELLS)).squeeze(1)
        # Keep no-op rows finite so log_softmax stays NaN-free; contribution
        # is zeroed by the played gate downstream.
        safe = slot_mask.any(dim=1, keepdim=True)
        slot_mask = torch.where(safe, slot_mask, torch.ones_like(slot_mask))
        return logits.masked_fill(~slot_mask, NEG_INF)

    # ------------------------------------------------------------- API

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (B, 2) actions; returns (actions, log_probs, values)."""
        feat = self.trunk(obs)
        card_dist = torch.distributions.Categorical(
            logits=self.card_logits(feat, masks["card"]))
        card = card_dist.probs.argmax(-1) if deterministic else card_dist.sample()

        place_logits = self.place_logits(feat, obs["cards"], card, masks["place"])
        place_dist = torch.distributions.Categorical(logits=place_logits)
        cell = place_dist.probs.argmax(-1) if deterministic else place_dist.sample()

        played = (card > 0).float()
        log_prob = card_dist.log_prob(card) + played * place_dist.log_prob(cell)
        value = self.value_head(feat).squeeze(-1)
        return torch.stack([card, cell], dim=1), log_prob, value

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log_probs, entropy, values) for stored actions — PPO update path."""
        feat = self.trunk(obs)
        card, cell = actions[:, 0], actions[:, 1]
        card_dist = torch.distributions.Categorical(
            logits=self.card_logits(feat, masks["card"]))
        place_dist = torch.distributions.Categorical(
            logits=self.place_logits(feat, obs["cards"], card, masks["place"]))
        played = (card > 0).float()
        log_prob = card_dist.log_prob(card) + played * place_dist.log_prob(cell)
        entropy = card_dist.entropy() + played * place_dist.entropy()
        value = self.value_head(feat).squeeze(-1)
        return log_prob, entropy, value

    def bc_logits(
        self,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        card_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(card_logits, place_logits) with placement conditioned on the
        *target* card — teacher forcing for behavior cloning."""
        feat = self.trunk(obs)
        return (self.card_logits(feat, masks["card"]),
                self.place_logits(feat, obs["cards"], card_targets, masks["place"]))


def make_network(n_cards: int, config: dict | None = None) -> PolicyNetwork:
    cfg = config or {}
    return PolicyNetwork(NetworkConfig(
        n_cards=n_cards,
        conv_channels=tuple(cfg.get("conv_channels", (32, 64, 64))),
        cnn_out=int(cfg.get("cnn_out", 256)),
        card_embed_dim=int(cfg.get("card_embed_dim", 16)),
        hand_mlp=int(cfg.get("hand_mlp", 64)),
        fusion_mlp=int(cfg.get("fusion_mlp", 256)),
    ))
