"""Inference-time search over the policy's top-k actions (#40).

**Simulator-only.** Each candidate action costs a `deepcopy` of the engine
plus several seconds of simulated play; that does not fit inside the 0.5s
live decision cadence and is not meant to. Its value is (a) a materially
stronger sim agent for benchmarking and (b) a stronger *teacher* for the
#37 distillation, where the student pays none of the cost.

**Do not search the raw action space.** It is 5 card choices x 144 cells =
720 joint actions per decision, and most are nonsense the policy already
assigns ~0 probability. Search is restricted to the policy's top-k masked
actions, which is where essentially all of the value is.

`BattleEngine` is plain Python and deterministic given its rng, so
`deepcopy` is a workable clone. Card stats are `lru_cache`-backed frozen
dataclasses shared across clones, so the copy cost is units and towers
rather than the whole card roster — but it is still the dominant cost here,
which is why `SearchConfig.time_budget_s` exists.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch

from src.agent.masking import cell_to_xy
from src.agent.network import PolicyNetwork, masks_to_tensors, obs_to_tensors
from src.agent.selfplay import PolicyBot, _obs_and_masks, policy_action
from src.bots.base import Action
from src.simulator.constants import PLACE_COLS, MatchResult, Side
from src.simulator.engine import BattleEngine


@dataclass(frozen=True)
class SearchConfig:
    top_k_cards: int = 3          # card choices considered (including no-op)
    cells_per_card: int = 3       # placements considered per card
    max_candidates: int = 8
    rollouts_per_candidate: int = 2
    horizon_seconds: float = 6.0
    decision_ticks: int = 5
    # Weight on the simulated tower-HP swing relative to the value head. The
    # value head is the better long-horizon estimate; the HP term is what
    # makes short rollouts discriminate at all.
    tower_weight: float = 1.0
    time_budget_s: float | None = None
    deterministic_rollouts: bool = True


def _tower_differential(engine: BattleEngine, side: Side) -> float:
    """(our tower HP - their tower HP), normalized. Crowns dominate."""
    mine = sum(t.hp for t in engine.towers if t.side == side)
    theirs = sum(t.hp for t in engine.towers if t.side == side.other)
    total = max(mine + theirs, 1.0)
    crowns = engine.crowns(side) - engine.crowns(side.other)
    return crowns + (mine - theirs) / total


def leaf_value(net: PolicyNetwork, engine: BattleEngine, side: Side,
               card_to_id: dict[str, int]) -> float:
    """The critic's estimate at a rollout leaf."""
    if engine.result != MatchResult.ONGOING:
        if engine.result == MatchResult.DRAW:
            return 0.0
        return 1.0 if engine.result == MatchResult.win_for(side) else -1.0
    device = next(net.parameters()).device
    obs, masks = _obs_and_masks(net, engine, side, card_to_id)
    with torch.no_grad():
        _, _, value = net.act(obs_to_tensors(obs, device),
                              masks_to_tensors(masks, device), deterministic=True)
    return float(value[0])


def candidate_actions(
    net: PolicyNetwork,
    engine: BattleEngine,
    side: Side,
    card_to_id: dict[str, int],
    config: SearchConfig,
) -> list[tuple[int, int]]:
    """The policy's most-likely legal ``(card_choice, cell)`` pairs."""
    device = next(net.parameters()).device
    obs, masks = _obs_and_masks(net, engine, side, card_to_id)
    obs_t = obs_to_tensors(obs, device)
    masks_t = masks_to_tensors(masks, device)
    with torch.no_grad():
        feat = net.trunk(obs_t)
        card_logits = net.card_logits(feat, masks_t["card"])
        legal_cards = int(masks_t["card"][0].sum())
        k = min(config.top_k_cards, legal_cards)
        top_cards = torch.topk(card_logits[0], k).indices.tolist()

        out: list[tuple[int, int]] = []
        for choice in top_cards:
            if choice == 0:
                out.append((0, 0))
                continue
            place = net.place_logits(feat, obs_t["cards"],
                                     torch.tensor([choice], device=device),
                                     masks_t["place"])[0]
            legal = int(masks_t["place"][0, choice - 1].sum())
            if legal == 0:
                continue
            for cell in torch.topk(place, min(config.cells_per_card, legal)).indices.tolist():
                out.append((choice, int(cell)))
    return out[:config.max_candidates]


def _apply(engine: BattleEngine, side: Side, choice: int, cell: int) -> None:
    if choice == 0:
        return
    row, col = divmod(int(cell), PLACE_COLS)
    x, y = cell_to_xy(side, col, row, engine.arena.height)
    slot = choice - 1
    player = engine.players[side]
    if player.can_afford(slot) and engine.legal_deploy(side, player.hand[slot], x, y):
        engine.play_card(side, slot, x, y)


class RolloutSearch:
    """Rollout-and-average over the policy's top-k actions.

    Deliberately not MCTS: with a 6-second horizon and a value head at the
    leaf, the tree is one ply deep and the extra bookkeeping of selection and
    backup buys nothing. Root-parallel MCTS is the natural upgrade if the
    horizon ever grows.
    """

    def __init__(
        self,
        net: PolicyNetwork,
        card_names: list[str],
        config: SearchConfig | None = None,
        opponent_net: PolicyNetwork | None = None,
    ):
        self.net = net
        self.card_to_id = {n: i for i, n in enumerate(card_names)}
        self.config = config or SearchConfig()
        # Whom we assume the opponent to be during a playout. Self-play
        # against our own policy is the honest default; a specific frozen
        # checkpoint models a known opponent better when we have one.
        self.opponent_net = opponent_net or net

    def _playout(self, engine: BattleEngine, side: Side,
                 choice: int, cell: int) -> float:
        cfg = self.config
        clone = copy.deepcopy(engine)
        _apply(clone, side, choice, cell)

        step_seconds = clone.arena.dt * cfg.decision_ticks
        steps = max(1, int(cfg.horizon_seconds / max(step_seconds, 1e-6)))
        before = _tower_differential(clone, side)
        for _ in range(steps):
            if clone.result != MatchResult.ONGOING:
                break
            for actor, net in ((side, self.net), (side.other, self.opponent_net)):
                c, e = policy_action(net, clone, actor, self.card_to_id,
                                     deterministic=cfg.deterministic_rollouts)
                _apply(clone, actor, c, e)
            for _ in range(cfg.decision_ticks):
                clone.tick()
                if clone.result != MatchResult.ONGOING:
                    break
        swing = _tower_differential(clone, side) - before
        return leaf_value(self.net, clone, side, self.card_to_id) + cfg.tower_weight * swing

    def evaluate_candidates(self, engine: BattleEngine, side: Side) -> dict[tuple[int, int], float]:
        cfg = self.config
        started = time.perf_counter()
        scores: dict[tuple[int, int], float] = {}
        for action in candidate_actions(self.net, engine, side, self.card_to_id, cfg):
            total = 0.0
            runs = 0
            for _ in range(cfg.rollouts_per_candidate):
                total += self._playout(engine, side, *action)
                runs += 1
                if cfg.time_budget_s is not None and \
                        time.perf_counter() - started > cfg.time_budget_s:
                    break
            scores[action] = total / max(runs, 1)
            if cfg.time_budget_s is not None and \
                    time.perf_counter() - started > cfg.time_budget_s:
                # Out of budget: return what has been scored so far rather
                # than an incomplete-and-silently-biased full sweep.
                break
        return scores

    def choose(self, engine: BattleEngine, side: Side) -> tuple[int, int]:
        scores = self.evaluate_candidates(engine, side)
        if not scores:
            return policy_action(self.net, engine, side, self.card_to_id,
                                 deterministic=True)
        return max(scores.items(), key=lambda kv: kv[1])[0]


class SearchBot(PolicyBot):
    """A searching policy behind the scripted-bot interface.

    Slots into `match_runner`, the frozen benchmark, and the league exactly
    like `PolicyBot`, so "policy + search" can be benchmarked against "policy
    alone" with no other change. Sim-only — see the module docstring.
    """

    def __init__(self, net: PolicyNetwork, card_names: list[str],
                 config: SearchConfig | None = None,
                 name: str = "search", opponent_net: PolicyNetwork | None = None):
        super().__init__(net, card_names, name=name, deterministic=True)
        self.search = RolloutSearch(net, card_names, config, opponent_net)

    # Search is per-state and cannot share a forward pass across envs.
    # `SyncVecEnv` treats a missing/None `batch_key` as "resolve this env
    # yourself", which is exactly right; a method returning None would
    # instead group these envs and silently run the *unsearched* policy.
    batch_key = None

    def decide(self, engine: BattleEngine, side: Side) -> Action | None:
        return self.decode_row(engine, side, self.search.choose(engine, side))
