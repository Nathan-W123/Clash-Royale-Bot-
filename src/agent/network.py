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

from src.agent.obs_layout import (
    MAX_ENTITIES,
    SCALAR_DIM,
    SPATIAL_CHANNELS,
    TIER_FULL,
    UNIT_FEATURE_DIM,
    resolve_tier,
    scalar_dim_for,
    tier_uses_spatial,
)
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
    tier: str = TIER_FULL
    scalar_dim: int = SCALAR_DIM
    # #39: swap the CNN-over-grid for a permutation-invariant encoder over a
    # variable-length entity list. Kept behind a flag so the two can be
    # ablated against each other on the frozen benchmark.
    use_set_encoder: bool = False
    unit_hidden: int = 128
    # #26 asymmetric actor-critic: when set, the value head reads a *separate*
    # trunk fed this (richer, usually `full`) tier while the policy keeps
    # reading `tier`. Training happens in a simulator where ground truth is
    # free, so the critic can see what the deployed actor never will; this
    # cuts value-estimation variance without the actor ever consuming
    # privileged input. None keeps the classic shared-trunk critic.
    critic_tier: str | None = None
    critic_scalar_dim: int = SCALAR_DIM
    # #27 recurrence. The restricted tier cannot see enemy troops at all, so
    # a feedforward policy has no mechanism to represent "their elixir
    # dropped 5 two seconds ago, a push is coming" — that information exists
    # only across time. A GRU sits between the fused trunk and the heads.
    #
    # `gru_hidden` defaults to `fusion_mlp` so the existing heads keep their
    # input width and their parameter names are untouched.
    use_recurrence: bool = False
    gru_hidden: int = 0
    # Truncated BPTT window. Backpropagating through a full 512-step rollout
    # is a memory problem; chunking detaches the hidden state every N steps.
    # 0 means "no chunking" (backprop the whole rollout).
    bptt_chunk: int = 0

    @property
    def asymmetric(self) -> bool:
        return self.critic_tier is not None

    @property
    def hidden_size(self) -> int:
        return self.gru_hidden or self.fusion_mlp

    @property
    def use_spatial(self) -> bool:
        """Deprecated alias kept for call sites that only care whether a
        spatial grid exists. It is a *property*, not a field, so `asdict`
        never writes it back into a checkpoint — new checkpoints record
        `tier` instead, and `make_network` still reads the old key."""
        return tier_uses_spatial(self.tier)


def obs_to_tensors(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Numpy obs (single or batched) -> batched torch tensors.

    Batched-ness is decided by the spatial grid's rank and applied to every
    key, so optional entries (`units` for the set encoder, the `critic_*`
    mirror for the asymmetric critic) can never end up batched differently
    from the rest. Any key ending in `cards` is an index tensor; everything
    else is float.
    """
    spatial = torch.as_tensor(obs["spatial"], dtype=torch.float32, device=device)
    batched = spatial.dim() == 4
    out = {
        k: torch.as_tensor(
            v, dtype=torch.long if k.endswith("cards") else torch.float32, device=device)
        for k, v in obs.items()
    }
    if not batched:
        out = {k: v[None] for k, v in out.items()}
    return out


CRITIC_PREFIX = "critic_"


def critic_view(obs: dict) -> dict:
    """Re-key the `critic_*` entries as a plain observation the trunk can eat."""
    return {k[len(CRITIC_PREFIX):]: v for k, v in obs.items()
            if k.startswith(CRITIC_PREFIX)}


def masks_to_tensors(masks: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    card = torch.as_tensor(masks["card"], dtype=torch.bool, device=device)
    place = torch.as_tensor(masks["place"], dtype=torch.bool, device=device)
    if card.dim() == 1:
        card, place = card[None], place[None]
    return {"card": card, "place": place}


class SetEncoder(nn.Module):
    """Deep-sets encoder over a padded entity list (#39).

    The 9x16 grid quantises positions into 2x2-tile cells and sums HP into
    density channels, discarding both exact position and per-unit identity —
    and placement precision is a real skill in this game. A per-entity
    embedding with symmetric pooling keeps both, consumes a variable number
    of entities natively, and (usefully for #32) is what a detector actually
    outputs, so dropping or jittering *list entries* models detection noise
    far more naturally than perturbing a rasterised grid.

    Mean and max pooling are both taken: mean carries "how much stuff is
    there", max carries "is there anything with this property at all" — the
    latter is what a single lethal unit looks like, and mean alone washes it
    out.
    """

    def __init__(self, card_embed: nn.Embedding, hidden: int, out_dim: int):
        super().__init__()
        self.card_embed = card_embed
        in_dim = card_embed.embedding_dim + UNIT_FEATURE_DIM - 1
        self.per_unit = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(2 * hidden, out_dim), nn.ReLU())

    def forward(self, units: torch.Tensor) -> torch.Tensor:
        # Column 0 is a card id; column 1 is the presence/padding flag.
        ids = units[:, :, 0].clamp(min=0).long().clamp(max=self.card_embed.num_embeddings - 1)
        present = units[:, :, 1] > 0
        feats = torch.cat([self.card_embed(ids), units[:, :, 1:]], dim=-1)
        encoded = self.per_unit(feats) * present.unsqueeze(-1)

        count = present.sum(dim=1, keepdim=True).clamp(min=1)
        mean = encoded.sum(dim=1) / count
        # Absent rows must not win the max; an all-absent batch row pools to
        # zero rather than to -inf.
        masked = encoded.masked_fill(~present.unsqueeze(-1), float("-inf"))
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum,
                              torch.zeros_like(maximum))
        return self.proj(torch.cat([mean, maximum], dim=1))


class PolicyNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        if config.use_spatial and not config.use_set_encoder:
            chans = [SPATIAL_CHANNELS, *config.conv_channels]
            convs: list[nn.Module] = []
            for cin, cout in zip(chans[:-1], chans[1:]):
                convs += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
            convs += [nn.Flatten()]
            self.cnn = nn.Sequential(*convs)
            cnn_flat = config.conv_channels[-1] * PLACE_ROWS * PLACE_COLS
            self.cnn_proj = nn.Sequential(nn.Linear(cnn_flat, config.cnn_out), nn.ReLU())

        self.card_embed = nn.Embedding(config.n_cards, config.card_embed_dim)
        if config.use_spatial and config.use_set_encoder:
            # Shares the card embedding with the hand encoder on purpose: a
            # card on the arena and the same card in hand should live at the
            # same point in embedding space.
            self.set_encoder = SetEncoder(self.card_embed, config.unit_hidden,
                                          config.cnn_out)
        hand_in = (HAND_SIZE + 1) * config.card_embed_dim + config.scalar_dim
        self.hand_mlp = nn.Sequential(nn.Linear(hand_in, config.hand_mlp), nn.ReLU())

        fused_in = (config.cnn_out if config.use_spatial else 0) + config.hand_mlp
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

        # Recurrent core (#27), added only when enabled so non-recurrent
        # checkpoints keep their exact parameter set.
        if config.use_recurrence:
            self.gru = nn.GRU(config.fusion_mlp, config.hidden_size)
            if config.hidden_size != config.fusion_mlp:
                # Heads are sized to fusion_mlp; project back rather than
                # rebuild them, so head parameter names stay stable.
                self.gru_proj = nn.Linear(config.hidden_size, config.fusion_mlp)

        # Asymmetric critic (#26). Built only when configured, and only ever
        # *added* to the module tree — the actor's parameter names are
        # untouched so every pre-existing checkpoint still loads strictly.
        #
        # The critic gets its own card embedding rather than sharing the
        # actor's. Sharing would work (and is common), but a separate
        # embedding makes the information boundary auditable: no tensor the
        # critic computes is ever consumed by the policy heads, so the only
        # channel from privileged state to the actor is the advantage signal,
        # which is exactly the intended one.
        if config.asymmetric:
            critic_spatial = tier_uses_spatial(config.critic_tier)
            self.critic_card_embed = nn.Embedding(config.n_cards, config.card_embed_dim)
            if critic_spatial and not config.use_set_encoder:
                chans = [SPATIAL_CHANNELS, *config.conv_channels]
                convs = []
                for cin, cout in zip(chans[:-1], chans[1:]):
                    convs += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
                convs += [nn.Flatten()]
                self.critic_cnn = nn.Sequential(*convs)
                self.critic_cnn_proj = nn.Sequential(
                    nn.Linear(config.conv_channels[-1] * PLACE_ROWS * PLACE_COLS,
                              config.cnn_out), nn.ReLU())
            elif critic_spatial:
                self.critic_set_encoder = SetEncoder(
                    self.critic_card_embed, config.unit_hidden, config.cnn_out)
            critic_hand_in = ((HAND_SIZE + 1) * config.card_embed_dim
                              + config.critic_scalar_dim)
            self.critic_hand_mlp = nn.Sequential(
                nn.Linear(critic_hand_in, config.hand_mlp), nn.ReLU())
            critic_fused = (config.cnn_out if critic_spatial else 0) + config.hand_mlp
            self.critic_fusion = nn.Sequential(
                nn.Linear(critic_fused, config.fusion_mlp), nn.ReLU(),
                nn.Linear(config.fusion_mlp, config.fusion_mlp), nn.ReLU(),
            )

    # ------------------------------------------------------------- trunk

    def trunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        emb = self.card_embed(obs["cards"]).flatten(1)
        hand = self.hand_mlp(torch.cat([emb, obs["vector"]], dim=1))
        if not self.config.use_spatial:
            return self.fusion(hand)
        if self.config.use_set_encoder:
            if "units" not in obs:
                raise KeyError(
                    "set-encoder policy needs the 'units' observation; build the "
                    "env with with_units=True (or encode_obs(..., with_units=True))")
            arena = self.set_encoder(obs["units"])
        else:
            arena = self.cnn_proj(self.cnn(obs["spatial"]))
        return self.fusion(torch.cat([arena, hand], dim=1))

    # --------------------------------------------------------- recurrence

    def initial_hidden(self, batch: int, device: torch.device) -> torch.Tensor | None:
        """Zero hidden state, or None for a feedforward policy."""
        if not self.config.use_recurrence:
            return None
        return torch.zeros(1, batch, self.config.hidden_size, device=device)

    def recurrent_step(
        self,
        feat: torch.Tensor,
        hidden: torch.Tensor,
        done: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the GRU one timestep, resetting where an episode ended.

        `done` marks envs whose *previous* step terminated. Their hidden state
        belongs to a finished match and must be zeroed before this step —
        the vec-env has already auto-reset them, so carrying memory across
        would blend two unrelated games and quietly poison the recurrence.
        """
        if done is not None:
            hidden = hidden * (~done).to(hidden.dtype).view(1, -1, 1)
        out, hidden = self.gru(feat.unsqueeze(0), hidden)
        out = out.squeeze(0)
        if hasattr(self, "gru_proj"):
            out = self.gru_proj(out)
        return out, hidden

    @torch.no_grad()
    def act_recurrent(
        self,
        obs: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        done: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        """`act`, carrying hidden state. Returns (actions, log_probs, values, hidden).

        Separate from `act` rather than an extra argument on it: `act` is
        called from several places that have no notion of hidden state
        (`PolicyBot`, batched opponents, BC), and widening its return arity
        would churn all of them.
        """
        feat, hidden = self.recurrent_step(self.trunk(obs), hidden, done)
        card_dist = torch.distributions.Categorical(
            logits=self.card_logits(feat, masks["card"]))
        card = card_dist.probs.argmax(-1) if deterministic else card_dist.sample()
        place_dist = torch.distributions.Categorical(
            logits=self.place_logits(feat, obs["cards"], card, masks["place"]))
        cell = place_dist.probs.argmax(-1) if deterministic else place_dist.sample()
        played = (card > 0).float()
        log_prob = card_dist.log_prob(card) + played * place_dist.log_prob(cell)
        # The critic stays feedforward even here: a privileged critic already
        # sees the true state, so it has far less need to infer history.
        value = self.value_of(obs, feat)
        return torch.stack([card, cell], dim=1), log_prob, value, hidden

    def evaluate_actions_recurrent(
        self,
        obs_seq: dict[str, torch.Tensor],
        masks_seq: dict[str, torch.Tensor],
        actions_seq: torch.Tensor,
        initial_hidden: torch.Tensor,
        dones_seq: torch.Tensor,
        bptt_chunk: int = 0,
    ):
        """Re-unroll a stored (T, B, ...) rollout under current parameters.

        Stored hidden states go stale the moment the policy updates, so the
        sequence is replayed from its start rather than reusing them — the
        standard fix, and the reason recurrent PPO must minibatch over whole
        env-trajectories instead of over shuffled transitions.

        `bptt_chunk` detaches the hidden state every N steps, bounding the
        backward graph without breaking the forward recurrence.
        """
        n_steps = actions_seq.shape[0]
        hidden = initial_hidden
        log_probs, entropies, values = [], [], []

        for t in range(n_steps):
            if bptt_chunk and t and t % bptt_chunk == 0:
                hidden = hidden.detach()
            obs_t = {k: v[t] for k, v in obs_seq.items()}
            masks_t = {k: v[t] for k, v in masks_seq.items()}
            # dones[t-1] gates step t: it marks the episode that ended just
            # before this observation was produced.
            done_t = dones_seq[t - 1] if t > 0 else None
            feat, hidden = self.recurrent_step(self.trunk(obs_t), hidden, done_t)

            card, cell = actions_seq[t, :, 0], actions_seq[t, :, 1]
            card_dist = torch.distributions.Categorical(
                logits=self.card_logits(feat, masks_t["card"]))
            place_dist = torch.distributions.Categorical(
                logits=self.place_logits(feat, obs_t["cards"], card, masks_t["place"]))
            played = (card > 0).float()
            log_probs.append(card_dist.log_prob(card) + played * place_dist.log_prob(cell))
            entropies.append(card_dist.entropy() + played * place_dist.entropy())
            values.append(self.value_of(obs_t, feat))

        return (torch.stack(log_probs), torch.stack(entropies), torch.stack(values))

    def critic_trunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Trunk for the privileged value function (#26).

        Mirrors `trunk` but over the `critic_*` observation and the critic's
        own encoder stack.
        """
        cobs = critic_view(obs)
        emb = self.critic_card_embed(cobs["cards"]).flatten(1)
        hand = self.critic_hand_mlp(torch.cat([emb, cobs["vector"]], dim=1))
        if not tier_uses_spatial(self.config.critic_tier):
            return self.critic_fusion(hand)
        if self.config.use_set_encoder:
            arena = self.critic_set_encoder(cobs["units"])
        else:
            arena = self.critic_cnn_proj(self.critic_cnn(cobs["spatial"]))
        return self.critic_fusion(torch.cat([arena, hand], dim=1))

    def value_of(self, obs: dict[str, torch.Tensor],
                 actor_feat: torch.Tensor) -> torch.Tensor:
        """Value estimate, from the privileged trunk when one is configured.

        Falls back to the actor features when the critic observation is
        absent. That case is normal, not a bug: opponents driven by
        `PolicyBot` call `act` purely for an action and never build critic
        observations, and their returned value is discarded.
        """
        if self.config.asymmetric and any(
                k.startswith(CRITIC_PREFIX) for k in obs):
            return self.value_head(self.critic_trunk(obs)).squeeze(-1)
        return self.value_head(actor_feat).squeeze(-1)

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
        value = self.value_of(obs, feat)
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
        value = self.value_of(obs, feat)
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
    """Build a policy from a plain dict — training yaml `network:` blocks and
    the `config` blob stored inside a checkpoint both land here.

    Unknown keys are ignored and the deprecated `use_spatial` bool is still
    honored (as a tier alias) so every checkpoint written before the `human`
    tier existed keeps loading; `tier` wins when both are present.
    """
    cfg = config or {}
    tier = resolve_tier(cfg["tier"] if cfg.get("tier") else cfg.get("use_spatial", True))
    critic_tier = cfg.get("critic_tier")
    critic_tier = resolve_tier(critic_tier) if critic_tier else None
    return PolicyNetwork(NetworkConfig(
        n_cards=n_cards,
        conv_channels=tuple(cfg.get("conv_channels", (32, 64, 64))),
        cnn_out=int(cfg.get("cnn_out", 256)),
        card_embed_dim=int(cfg.get("card_embed_dim", 16)),
        hand_mlp=int(cfg.get("hand_mlp", 64)),
        fusion_mlp=int(cfg.get("fusion_mlp", 256)),
        tier=tier,
        scalar_dim=int(cfg.get("scalar_dim", scalar_dim_for(tier))),
        use_set_encoder=bool(cfg.get("use_set_encoder", False)),
        unit_hidden=int(cfg.get("unit_hidden", 128)),
        critic_tier=critic_tier,
        critic_scalar_dim=int(cfg.get("critic_scalar_dim",
                                      scalar_dim_for(critic_tier or tier))),
        use_recurrence=bool(cfg.get("use_recurrence", False)),
        gru_hidden=int(cfg.get("gru_hidden", 0)),
        bptt_chunk=int(cfg.get("bptt_chunk", 0)),
    ))
