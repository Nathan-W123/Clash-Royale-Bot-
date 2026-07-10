"""Behavior-cloning bootstrap: imitate champion-tier scripted bots, then one
or more DAgger rounds so PPO starts from a legal, purposeful policy.

Dataset: (obs, masks, card_choice, cell) at the agent decision cadence
(every `decision_ticks` engine ticks), for both seats of bot-vs-bot matches.
No-op frames vastly outnumber card plays, so they are subsampled ~2:1.

CLI:  uv run python -m src.agent.bc --matches 150 --out checkpoints/bc_init.pt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.agent import masking, obs_layout
from src.agent.network import PolicyNetwork, make_network, masks_to_tensors, obs_to_tensors
from src.agent.selfplay import PolicyBot, save_checkpoint
from src.bots.base import Action, Bot
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.constants import HAND_SIZE, PLACE_COLS, MatchResult, Side
from src.simulator.engine import BattleEngine

MAX_DECISIONS = 600
NOOP_KEEP_RATIO = 2.0  # kept no-op frames per card-play frame


@dataclass
class Dataset:
    spatial: list = field(default_factory=list)
    cards: list = field(default_factory=list)
    vector: list = field(default_factory=list)
    card_mask: list = field(default_factory=list)
    place_mask: list = field(default_factory=list)
    target_card: list = field(default_factory=list)
    target_cell: list = field(default_factory=list)

    def add(self, obs, masks, card, cell) -> None:
        self.spatial.append(obs["spatial"])
        self.cards.append(obs["cards"])
        self.vector.append(obs["vector"])
        self.card_mask.append(masks["card"])
        self.place_mask.append(masks["place"])
        self.target_card.append(card)
        self.target_cell.append(cell)

    def __len__(self) -> int:
        return len(self.target_card)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "spatial": np.stack(self.spatial),
            "cards": np.stack(self.cards),
            "vector": np.stack(self.vector),
            "card_mask": np.stack(self.card_mask),
            "place_mask": np.stack(self.place_mask),
            "target_card": np.asarray(self.target_card, np.int64),
            "target_cell": np.asarray(self.target_cell, np.int64),
        }


def action_to_target(engine: BattleEngine, side: Side, action: Action | None,
                     place_mask: np.ndarray) -> tuple[int, int] | None:
    """Bot Action -> (card_choice, cell); None if it doesn't snap to a legal cell."""
    if action is None:
        return 0, 0
    fy = masking.frame_y(side, action.y, engine.arena.height)
    col, row = int(action.x // 2), int(fy // 2)
    cell = masking.cell_index(min(col, PLACE_COLS - 1), row)
    if 0 <= cell < place_mask.shape[1] and place_mask[action.slot, cell]:
        return action.slot + 1, cell
    return None


def full_masks(engine: BattleEngine, side: Side) -> dict[str, np.ndarray]:
    m = masking.build_action_masks(engine, side)
    card = np.concatenate(([True], m["card"] & m["place"].any(axis=1)))
    return {"card": card, "place": m["place"]}


def collect_bot_matches(
    n_matches: int,
    seed: int,
    decision_ticks: int = 5,
    catalog: DeckCatalog | None = None,
    student: PolicyBot | None = None,
    expert_name: str = "champion",
) -> Dataset:
    """Bot-vs-bot rollouts (or student rollouts relabeled by the expert when
    `student` is given — the DAgger case)."""
    catalog = catalog or DeckCatalog()
    arena = load_arena()
    card_to_id = {n: i for i, n in enumerate(load_cards())}
    rng = np.random.default_rng(seed)
    pairs = [("rusher", "rusher"), ("control", "control"),
             ("siege", "siege"), ("beatdown", "beatdown"),
             ("champion", "training_mirror")]
    data = Dataset()
    plays = kept_noops = 0

    for i in range(n_matches):
        (bot_b_name, deck_b_name) = pairs[i % len(pairs)]
        (bot_t_name, deck_t_name) = pairs[(i * 3 + 1) % len(pairs)]
        deck_b = catalog.resolve(deck_b_name)
        deck_t = catalog.resolve(deck_t_name)
        expert_b: Bot = get_bot(bot_b_name, catalog=catalog, deck_name=deck_b_name, rng=rng)
        bot_t: Bot = get_bot(bot_t_name, catalog=catalog, deck_name=deck_t_name, rng=rng)
        actor_b: Bot = student if student is not None else expert_b
        engine = BattleEngine(deck_b, deck_t, arena, seed=int(rng.integers(2**31)))

        for _ in range(MAX_DECISIONS):
            if engine.result != MatchResult.ONGOING:
                break
            for side, actor, expert in ((Side.BOTTOM, actor_b, expert_b),
                                        (Side.TOP, bot_t, bot_t)):
                masks = full_masks(engine, side)
                label_action = expert.decide(engine, side)
                target = action_to_target(engine, side, label_action, masks["place"])
                record = target is not None
                if record and target == (0, 0):
                    if kept_noops >= NOOP_KEEP_RATIO * max(plays, 10):
                        record = False
                    else:
                        kept_noops += 1
                elif record:
                    plays += 1
                if record:
                    obs = obs_layout.encode_obs(engine, side, card_to_id)
                    data.add(obs, masks, *target)

                # Advance the game with the *actor's* action (student on DAgger).
                play = label_action if actor is expert else actor.decide(engine, side)
                if play is not None:
                    player = engine.players[side]
                    if (player.can_afford(play.slot)
                            and engine.legal_deploy(side, player.hand[play.slot],
                                                    play.x, play.y)):
                        engine.play_card(side, play.slot, play.x, play.y)
            for _ in range(decision_ticks):
                engine.tick()
                if engine.result != MatchResult.ONGOING:
                    break
    return data


def train_bc(
    net: PolicyNetwork,
    data: Dataset,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = 0,
    device: str = "cpu",
) -> list[float]:
    arrays = data.arrays()
    n = len(data)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    epoch_losses = []
    for _ in range(epochs):
        order = rng.permutation(n)
        total, batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            obs = obs_to_tensors({k: arrays[k][idx] for k in ("spatial", "cards", "vector")},
                                 torch.device(device))
            masks = masks_to_tensors(
                {"card": arrays["card_mask"][idx], "place": arrays["place_mask"][idx]},
                torch.device(device))
            card_t = torch.as_tensor(arrays["target_card"][idx], device=device)
            cell_t = torch.as_tensor(arrays["target_cell"][idx], device=device)
            card_logits, place_logits = net.bc_logits(obs, masks, card_t)
            loss = F.cross_entropy(card_logits, card_t)
            played = card_t > 0
            if played.any():
                loss = loss + F.cross_entropy(place_logits[played], cell_t[played])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        epoch_losses.append(total / max(batches, 1))
    return epoch_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=150)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--dagger-matches", type=int, default=75)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/bc_init.pt")
    args = parser.parse_args()

    card_names = list(load_cards().keys())
    net = make_network(len(card_names))

    print(f"[bc] collecting {args.matches} expert matches...")
    data = collect_bot_matches(args.matches, seed=args.seed)
    print(f"[bc] dataset: {len(data)} frames "
          f"({int((np.asarray(data.target_card) > 0).sum())} card plays)")
    losses = train_bc(net, data, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch_size, seed=args.seed)
    print(f"[bc] BC losses/epoch: {[f'{l:.3f}' for l in losses]}")

    for r in range(args.dagger_rounds):
        student = PolicyBot(net, card_names, name="student")
        print(f"[bc] DAgger round {r + 1}: {args.dagger_matches} student rollouts...")
        extra = collect_bot_matches(args.dagger_matches, seed=args.seed + 1000 + r,
                                    student=student)
        for key in ("spatial", "cards", "vector", "card_mask", "place_mask",
                    "target_card", "target_cell"):
            getattr(data, key).extend(getattr(extra, key))
        losses = train_bc(net, data, epochs=max(2, args.epochs // 2), lr=args.lr / 2,
                          batch_size=args.batch_size, seed=args.seed + r)
        print(f"[bc] post-DAgger losses/epoch: {[f'{l:.3f}' for l in losses]}")

    save_checkpoint(net, card_names, Path(args.out))
    print(f"[bc] saved {args.out}")


if __name__ == "__main__":
    main()
