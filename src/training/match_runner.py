"""Headless bot-vs-bot match runner for eval and deck search."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.bots.base import Bot
from src.simulator.cards import ArenaConfig, CardStats
from src.simulator.constants import MatchResult, Side
from src.simulator.engine import BattleEngine


@dataclass(frozen=True)
class MatchOutcome:
    result: MatchResult
    ticks: int
    bottom_crowns: int
    top_crowns: int


@dataclass
class MatchReport:
    """Full match statistics for training/eval reporting."""

    outcome: MatchOutcome
    duration_sec: float
    bottom_deck_name: str = ""
    top_deck_name: str = ""
    bottom_cards_played: dict[str, int] = field(default_factory=dict)
    top_cards_played: dict[str, int] = field(default_factory=dict)
    bottom_elixir_spent: float = 0.0
    top_elixir_spent: float = 0.0
    bottom_elixir_leaked: float = 0.0
    top_elixir_leaked: float = 0.0

    def agent_won(self, agent_side: Side = Side.BOTTOM) -> bool | None:
        r = self.outcome.result
        if r == MatchResult.DRAW:
            return None
        if agent_side == Side.BOTTOM:
            return r == MatchResult.BOTTOM_WIN
        return r == MatchResult.TOP_WIN

    def agent_crowns(self, agent_side: Side = Side.BOTTOM) -> int:
        if agent_side == Side.BOTTOM:
            return self.outcome.bottom_crowns
        return self.outcome.top_crowns

    def opponent_crowns(self, agent_side: Side = Side.BOTTOM) -> int:
        if agent_side == Side.BOTTOM:
            return self.outcome.top_crowns
        return self.outcome.bottom_crowns

    def cards_for(self, side: Side) -> dict[str, int]:
        return self.bottom_cards_played if side == Side.BOTTOM else self.top_cards_played


def run_match(
    arena: ArenaConfig,
    bottom_deck: list[CardStats],
    top_deck: list[CardStats],
    bottom_bot: Bot,
    top_bot: Bot,
    seed: int = 0,
    max_ticks: int = 20_000,
    lanes: str = "both",
    regulation: float | None = None,
    bottom_deck_name: str = "",
    top_deck_name: str = "",
    detailed: bool = False,
) -> MatchOutcome | MatchReport:
    report = run_match_detailed(
        arena,
        bottom_deck,
        top_deck,
        bottom_bot,
        top_bot,
        seed=seed,
        max_ticks=max_ticks,
        lanes=lanes,
        regulation=regulation,
        bottom_deck_name=bottom_deck_name,
        top_deck_name=top_deck_name,
    )
    if detailed:
        return report
    return report.outcome


def run_match_detailed(
    arena: ArenaConfig,
    bottom_deck: list[CardStats],
    top_deck: list[CardStats],
    bottom_bot: Bot,
    top_bot: Bot,
    seed: int = 0,
    max_ticks: int = 20_000,
    lanes: str = "both",
    regulation: float | None = None,
    bottom_deck_name: str = "",
    top_deck_name: str = "",
    decision_ticks: int = 5,
) -> MatchReport:
    engine = BattleEngine(
        bottom_deck, top_deck, arena, seed=seed, lanes=lanes, regulation=regulation
    )
    bottom_plays: dict[str, int] = {}
    top_plays: dict[str, int] = {}
    ticks = 0
    while engine.result == MatchResult.ONGOING and ticks < max_ticks:
        # Same 0.5 s decision cadence as the training env, so bot difficulty
        # is identical in eval and training.
        if ticks % decision_ticks == 0:
            for side, bot in ((Side.BOTTOM, bottom_bot), (Side.TOP, top_bot)):
                action = bot.decide(engine, side)
                if action is not None:
                    card = engine.players[side].hand[action.slot]
                    if engine.legal_deploy(side, card, action.x, action.y):
                        try:
                            engine.play_card(side, action.slot, action.x, action.y)
                            bucket = bottom_plays if side == Side.BOTTOM else top_plays
                            bucket[card.name] = bucket.get(card.name, 0) + 1
                        except ValueError:
                            pass
        engine.tick()
        ticks += 1

    pb, pt = engine.players[Side.BOTTOM], engine.players[Side.TOP]
    return MatchReport(
        outcome=MatchOutcome(
            result=engine.result,
            ticks=ticks,
            bottom_crowns=engine.crowns(Side.BOTTOM),
            top_crowns=engine.crowns(Side.TOP),
        ),
        duration_sec=ticks * arena.dt,
        bottom_deck_name=bottom_deck_name,
        top_deck_name=top_deck_name,
        bottom_cards_played=bottom_plays,
        top_cards_played=top_plays,
        bottom_elixir_spent=pb.spent,
        top_elixir_spent=pt.spent,
        bottom_elixir_leaked=pb.leaked,
        top_elixir_leaked=pt.leaked,
    )
