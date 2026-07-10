"""Run sampled training episodes and collect reports."""
from __future__ import annotations

import numpy as np

from src.bots.base import Bot
from src.decks.builder import AdaptiveDeckBuilder, AdaptiveDeckBuilderConfig
from src.decks.catalog import DeckCatalog
from src.eval.reporter import TrainingReporter
from src.simulator.cards import load_arena
from src.simulator.constants import Side
from src.training.config import FocusedRotationConfig, load_training_config
from src.training.curriculum import CurriculumStage, load_curriculum, stage_by_name
from src.training.focused_curriculum import FocusedRotationManager
from src.training.match_runner import run_match_detailed
from src.training.opponents import OpponentSampler


def run_episodes(
    agent_bot: Bot,
    n_episodes: int,
    *,
    stage: CurriculumStage | str = "full_pool",
    catalog: DeckCatalog | None = None,
    seed: int = 0,
    run_name: str = "training",
) -> TrainingReporter:
    """Play N sampled matches and return a populated TrainingReporter."""
    catalog = catalog or DeckCatalog()
    arena = load_arena()
    training = load_training_config()
    if isinstance(stage, str):
        stage = stage_by_name(load_curriculum(), stage)
    sampler = OpponentSampler(catalog, training)
    reporter = TrainingReporter(run_name=run_name)
    rng = np.random.default_rng(seed)

    for _ in range(n_episodes):
        setup = sampler.sample(stage, rng)
        lanes = setup.single_lane or "both"
        report = run_match_detailed(
            arena,
            setup.agent_deck,
            setup.opponent_deck,
            agent_bot,
            setup.opponent_bot,
            seed=int(rng.integers(0, 2**31)),
            lanes=lanes,
            regulation=setup.match_time,
            bottom_deck_name=setup.agent_deck_name,
            top_deck_name=setup.opponent_deck_name,
        )
        reporter.record_report(
            report,
            agent_side=Side.BOTTOM,
            agent_deck=setup.agent_deck_name,
            opponent_deck=setup.opponent_deck_name,
            opponent_bot=setup.opponent_bot.name,
            opponent_kind=setup.opponent_kind.value,
            stage=setup.stage_name,
        )
    return reporter


def run_focused_training(
    agent_bot: Bot,
    *,
    max_episodes: int = 10_000,
    stage: CurriculumStage | str = "focused_ladder",
    catalog: DeckCatalog | None = None,
    seed: int = 0,
    run_name: str = "focused_training",
    rotation: FocusedRotationConfig | None = None,
) -> TrainingReporter:
    """Focused curriculum: fixed adaptive agent deck vs one ladder deck at a time."""
    catalog = catalog or DeckCatalog()
    arena = load_arena()
    training = load_training_config()
    if isinstance(stage, str):
        stage = stage_by_name(load_curriculum(), stage)

    rot_cfg = rotation or training.focused_rotation
    if rot_cfg is None:
        rot_cfg = FocusedRotationConfig(
            wins_per_deck=5,
            target_win_rate=0.65,
            min_matches_per_deck=10,
            min_win_rate_per_deck=0.55,
        )

    ladder_names = catalog.pool(stage.opponent_deck_pool or training.opponents.opponent_deck_pool)
    rotation_mgr = FocusedRotationManager(ladder_names, rot_cfg)

    adaptive_cfg = training.adaptive_deck
    builder = AdaptiveDeckBuilder(
        catalog,
        config=AdaptiveDeckBuilderConfig(
            rebuild_every_matches=adaptive_cfg.rebuild_every_matches if adaptive_cfg else 25,
            plateau_window=adaptive_cfg.plateau_window if adaptive_cfg else 20,
            plateau_threshold=adaptive_cfg.plateau_threshold if adaptive_cfg else 0.02,
        ),
    )

    sampler = OpponentSampler(catalog, training)
    reporter = TrainingReporter(run_name=run_name)
    rng = np.random.default_rng(seed)

    for _ in range(max_episodes):
        if not rotation_mgr.should_continue_training():
            break

        opp_name = rotation_mgr.current_opponent_deck()
        agent_deck = builder.current_deck()
        setup = sampler.setup_focused(
            stage,
            agent_deck_name=builder.current_deck_name,
            agent_deck=agent_deck,
            opponent_deck_name=opp_name,
            rng=rng,
        )
        lanes = setup.single_lane or "both"
        report = run_match_detailed(
            arena,
            setup.agent_deck,
            setup.opponent_deck,
            agent_bot,
            setup.opponent_bot,
            seed=int(rng.integers(0, 2**31)),
            lanes=lanes,
            regulation=setup.match_time,
            bottom_deck_name=setup.agent_deck_name,
            top_deck_name=setup.opponent_deck_name,
        )
        won = report.agent_won(Side.BOTTOM)
        rotation_mgr.record_result(bool(won))
        builder.record_match(
            dict(report.cards_for(Side.BOTTOM)),
            won=bool(won),
            crowns=report.agent_crowns(Side.BOTTOM),
            elixir_spent=report.bottom_elixir_spent,
        )
        reporter.record_report(
            report,
            agent_side=Side.BOTTOM,
            agent_deck=setup.agent_deck_name,
            opponent_deck=setup.opponent_deck_name,
            opponent_bot=setup.opponent_bot.name,
            opponent_kind=setup.opponent_kind.value,
            stage=setup.stage_name,
        )

    reporter.card_scores = builder.tracker.to_dict()
    reporter.focused_progress = rotation_mgr.progress()
    return reporter
