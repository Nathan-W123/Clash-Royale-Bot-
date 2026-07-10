"""Gymnasium adapter over BattleEngine.

The learning agent always occupies the BOTTOM seat; the opponent (scripted bot
or frozen policy) lives inside the env and acts on the TOP seat each decision
step. Observations are always encoded from the acting player's perspective
(y flipped for TOP), so one policy can play either seat.

Action = (card_choice, cell):
  card_choice: 0 = no-op, 1..4 = hand slot
  cell: index into the 9x16 coarse placement grid (2x2 tiles per cell),
        in the acting player's frame. Ignored for no-op.
"""
from __future__ import annotations

from typing import Callable, Protocol

import gymnasium as gym
import numpy as np

from src.simulator.cards import ArenaConfig, CardStats
from src.simulator.constants import (
    HAND_SIZE,
    PLACE_COLS,
    PLACE_ROWS,
    CardType,
    MatchResult,
    Side,
)
from src.simulator.engine import BattleEngine
from src.agent import masking, obs_layout

N_CARD_CHOICES = HAND_SIZE + 1  # no-op + 4 slots
N_CELLS = PLACE_COLS * PLACE_ROWS


class Opponent(Protocol):
    def act(self, env: "CRBattleEnv", side: Side) -> tuple[int, float, float] | None:
        """Return (hand_slot, x, y) in absolute coords, or None for no-op."""


RewardFn = Callable[[list[dict], Side, BattleEngine], float]


def terminal_reward(events: list[dict], side: Side, engine: BattleEngine) -> float:
    if engine.result == MatchResult.ONGOING or engine.result == MatchResult.DRAW:
        return 0.0
    return 1.0 if engine.result == MatchResult.win_for(side) else -1.0


class CRBattleEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        cards: dict[str, CardStats],
        arena: ArenaConfig,
        deck_bottom: list[CardStats],
        deck_top: list[CardStats],
        opponent: Opponent | None = None,
        reward_fn: RewardFn | None = None,
        lanes: str = "both",
        regulation: float | None = None,
        decision_ticks: int = 5,
        setup_fn: Callable[
            [np.random.Generator],
            tuple[list[CardStats], list[CardStats], "Opponent | None"]] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.cards = cards
        self.card_index = {name: i for i, name in enumerate(cards)}
        self.arena = arena
        self.deck_bottom = deck_bottom
        self.deck_top = deck_top
        self.setup_fn = setup_fn
        self.opponent = opponent
        self.reward_fn = reward_fn or terminal_reward
        self.lanes = lanes
        self.regulation = regulation
        self.decision_ticks = decision_ticks
        self._rng = np.random.default_rng(seed)
        self.engine: BattleEngine | None = None
        self._usage: dict[str, int] = {}
        self._illegal = 0

        n_cards = len(cards)
        self.observation_space = gym.spaces.Dict({
            "spatial": gym.spaces.Box(
                0.0, 30.0, (obs_layout.SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS), np.float32),
            "cards": gym.spaces.MultiDiscrete([n_cards] * (HAND_SIZE + 1)),
            "vector": gym.spaces.Box(0.0, 2.0, (obs_layout.SCALAR_DIM,), np.float32),
        })
        self.action_space = gym.spaces.MultiDiscrete([N_CARD_CHOICES, N_CELLS])

    # ------------------------------------------------------------ helpers

    def cell_to_xy(self, side: Side, cell: int) -> tuple[float, float]:
        """Placement cell (acting player's frame) -> absolute coords."""
        row, col = divmod(int(cell), PLACE_COLS)
        return masking.cell_to_xy(side, col, row, self.arena.height)

    def build_obs(self, side: Side) -> dict[str, np.ndarray]:
        return obs_layout.encode_obs(self.engine, side, self.card_index)

    def build_masks(self, side: Side) -> dict[str, np.ndarray]:
        """card: (5,) bool with no-op at index 0; place: (4, 144) bool."""
        m = masking.build_action_masks(self.engine, side)
        card = np.concatenate(([True], m["card"] & m["place"].any(axis=1)))
        return {"card": card, "place": m["place"]}

    def decode_action(self, side: Side, card_choice: int, cell: int) -> tuple[int, float, float] | None:
        if card_choice == 0:
            return None
        slot = card_choice - 1
        x, y = self.cell_to_xy(side, cell)
        return slot, x, y

    def _apply(self, side: Side, action: tuple[int, float, float] | None,
               events: list[dict]) -> bool:
        """Apply a deploy; returns False if it was illegal (treated as no-op)."""
        if action is None:
            return True
        slot, x, y = action
        eng = self.engine
        card = eng.players[side].hand[slot]
        if not eng.players[side].can_afford(slot) or not eng.legal_deploy(side, card, x, y):
            return False
        events += eng.play_card(side, slot, x, y)
        if side == Side.BOTTOM:
            self._usage[card.name] = self._usage.get(card.name, 0) + 1
        return True

    # ------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        deck_b, deck_t = self.deck_bottom, self.deck_top
        if self.setup_fn is not None:
            deck_b, deck_t, self.opponent = self.setup_fn(self._rng)
        self.engine = BattleEngine(
            deck_b, deck_t, self.arena, seed=self._rng,
            lanes=self.lanes, regulation=self.regulation)
        self._usage = {}
        self._illegal = 0
        return self.build_obs(Side.BOTTOM), {"masks": self.build_masks(Side.BOTTOM)}

    def step(self, action):
        card_choice, cell = int(action[0]), int(action[1])
        events: list[dict] = []

        opp_action = None
        if self.opponent is not None:
            opp_action = self.opponent.act(self, Side.TOP)

        if not self._apply(Side.BOTTOM, self.decode_action(Side.BOTTOM, card_choice, cell), events):
            self._illegal += 1
        self._apply(Side.TOP, opp_action, events)

        for _ in range(self.decision_ticks):
            events += self.engine.tick()
            if self.engine.result != MatchResult.ONGOING:
                break

        reward = float(self.reward_fn(events, Side.BOTTOM, self.engine))
        terminated = self.engine.result != MatchResult.ONGOING
        info: dict = {"events": events}
        if terminated:
            info["episode_metrics"] = self.episode_metrics()
        else:
            info["masks"] = self.build_masks(Side.BOTTOM)
        return self.build_obs(Side.BOTTOM), reward, terminated, False, info

    def episode_metrics(self) -> dict:
        eng = self.engine
        me = eng.players[Side.BOTTOM]
        result = eng.result
        return {
            "result": result.name,
            "opponent": getattr(self.opponent, "name", "none"),
            "win": float(result == MatchResult.BOTTOM_WIN),
            "draw": float(result == MatchResult.DRAW),
            "crowns_for": eng.crowns(Side.BOTTOM),
            "crowns_against": eng.crowns(Side.TOP),
            "match_time": eng.time,
            "elixir_spent": me.spent,
            "elixir_leaked": me.leaked,
            "card_usage": dict(self._usage),
            "illegal_actions": self._illegal,
        }

    def render(self):
        from src.simulator.render import render_ascii
        return render_ascii(self.engine)
