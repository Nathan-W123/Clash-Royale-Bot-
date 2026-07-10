"""Sample opponents and decks for a training episode."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.bots.base import Bot
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.decks.sampling import sample_deck, sample_match_decks
from src.simulator.cards import CardStats
from src.training.config import TrainingConfig
from src.training.curriculum import CurriculumStage
from src.training.matchup_tracker import MatchupTracker


class OpponentKind(Enum):
    SCRIPTED = "scripted"
    LEAGUE = "league"
    SELF = "self"


@dataclass(frozen=True)
class MatchSetup:
    stage_name: str
    agent_deck_name: str
    agent_deck: list[CardStats]
    opponent_deck_name: str
    opponent_deck: list[CardStats]
    opponent_kind: OpponentKind
    opponent_bot: Bot
    single_lane: str | None
    match_time: float | None
    selfplay: bool


class OpponentSampler:
    """Builds per-episode match setups with diverse UC-tier opponents."""

    def __init__(
        self,
        catalog: DeckCatalog,
        training: TrainingConfig,
        tracker: MatchupTracker | None = None,
    ):
        self.catalog = catalog
        self.training = training
        self.tracker = tracker or MatchupTracker()

    def sample(self, stage: CurriculumStage, rng: np.random.Generator) -> MatchSetup:
        agent_pool, opp_pool = self._deck_pools(stage)
        opp_weights = None
        if self.tracker and opp_pool:
            names = self.catalog.pool(opp_pool)
            opp_weights = self.tracker.sampling_weights(
                names, self.training.opponents.weakness_weight
            )

        if stage.deck_pool or (stage.agent_deck_pool and stage.opponent_deck_pool):
            (agent_name, agent_deck), (opp_name, opp_deck) = sample_match_decks(
                self.catalog, agent_pool, opp_pool, rng, opponent_weights=opp_weights
            )
        elif stage.deck and stage.opponent_deck:
            agent_name, opp_name = stage.deck, stage.opponent_deck
            agent_deck = self.catalog.resolve(agent_name)
            opp_deck = self.catalog.resolve(opp_name)
        else:
            agent_name, agent_deck = sample_deck(self.catalog, agent_pool, rng)
            opp_name, opp_deck = sample_deck(
                self.catalog, opp_pool, rng, weights=opp_weights
            )

        kind, bot_name = self._sample_opponent_kind(stage, rng)
        bot = get_bot(
            bot_name,
            catalog=self.catalog,
            deck_name=opp_name,
            rng=rng,
            skill_tier=self.training.opponents.skill_tier,
        )

        return MatchSetup(
            stage_name=stage.name,
            agent_deck_name=agent_name,
            agent_deck=agent_deck,
            opponent_deck_name=opp_name,
            opponent_deck=opp_deck,
            opponent_kind=kind,
            opponent_bot=bot,
            single_lane=stage.single_lane,
            match_time=stage.match_time,
            selfplay=stage.selfplay and kind != OpponentKind.SCRIPTED,
        )

    def setup_focused(
        self,
        stage: CurriculumStage,
        *,
        agent_deck_name: str,
        agent_deck: list[CardStats],
        opponent_deck_name: str,
        rng: np.random.Generator,
    ) -> MatchSetup:
        """Fixed agent + opponent decks for focused rotation training."""
        opp_deck = self.catalog.resolve(opponent_deck_name)
        bot_name = str(rng.choice(stage.opponents)) if stage.opponents else "champion"
        bot = get_bot(
            bot_name,
            catalog=self.catalog,
            deck_name=opponent_deck_name,
            rng=rng,
            skill_tier=self.training.opponents.skill_tier,
        )
        return MatchSetup(
            stage_name=stage.name,
            agent_deck_name=agent_deck_name,
            agent_deck=agent_deck,
            opponent_deck_name=opponent_deck_name,
            opponent_deck=opp_deck,
            opponent_kind=OpponentKind.SCRIPTED,
            opponent_bot=bot,
            single_lane=stage.single_lane,
            match_time=stage.match_time,
            selfplay=False,
        )

    def _deck_pools(self, stage: CurriculumStage) -> tuple[str, str]:
        cfg = self.training.opponents
        if stage.agent_deck_pool and stage.opponent_deck_pool:
            return stage.agent_deck_pool, stage.opponent_deck_pool
        if stage.deck_pool:
            return stage.deck_pool, stage.deck_pool
        return cfg.agent_deck_pool, cfg.opponent_deck_pool

    def _sample_opponent_kind(
        self, stage: CurriculumStage, rng: np.random.Generator
    ) -> tuple[OpponentKind, str]:
        if not stage.selfplay:
            bot = str(rng.choice(stage.opponents)) if stage.opponents else "champion"
            return OpponentKind.SCRIPTED, bot

        cfg = self.training.opponents
        r = float(rng.random())
        if r < cfg.sample_scripted:
            bot = str(rng.choice(cfg.scripted_bots))
            return OpponentKind.SCRIPTED, bot
        if r < cfg.sample_scripted + cfg.sample_latest:
            return OpponentKind.SELF, "champion"
        return OpponentKind.LEAGUE, "champion"
