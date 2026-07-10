import numpy as np

from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.training.config import load_training_config
from src.training.curriculum import load_curriculum, stage_by_name
from src.training.match_runner import run_match
from src.training.matchup_tracker import MatchupTracker
from src.training.opponents import OpponentKind, OpponentSampler
from src.simulator.constants import MatchResult, Side


def test_training_config_opponents():
    cfg = load_training_config()
    assert cfg.opponents.skill_tier == "ultimate_champion"
    assert "beatdown" in cfg.opponents.scripted_bots


def test_opponent_sampler_full_pool(cards, arena):
    cat = DeckCatalog(cards=cards)
    training = load_training_config()
    stages = load_curriculum()
    stage = stage_by_name(stages, "full_pool")
    sampler = OpponentSampler(cat, training, MatchupTracker())
    rng = np.random.default_rng(0)
    setups = [sampler.sample(stage, rng) for _ in range(30)]
    opp_decks = {s.opponent_deck_name for s in setups}
    assert len(opp_decks) >= 3
    assert all(s.opponent_bot.name.startswith("champion") for s in setups)


def test_weakness_weighting_favors_losses(cards):
    cat = DeckCatalog(cards=cards)
    tracker = MatchupTracker()
    tracker.record("rusher", won=False)
    tracker.record("rusher", won=False)
    tracker.record("control", won=True)
    weights = tracker.sampling_weights(
        ["rusher", "control"], weakness_weight=0.5
    )
    assert weights["rusher"] > weights["control"]


def test_champion_vs_champion_match_completes(cards, arena):
    cat = DeckCatalog(cards=cards)
    bottom = get_bot("rusher", catalog=cat, deck_name="rusher")
    top = get_bot("control", catalog=cat, deck_name="control")
    outcome = run_match(
        arena,
        cat.resolve("rusher"),
        cat.resolve("control"),
        bottom,
        top,
        seed=42,
        max_ticks=5000,
    )
    assert outcome.result != MatchResult.ONGOING
    assert outcome.ticks > 0


def test_one_lane_stage_uses_scripted_only(cards):
    cat = DeckCatalog(cards=cards)
    training = load_training_config()
    stage = stage_by_name(load_curriculum(), "one_lane")
    sampler = OpponentSampler(cat, training)
    setup = sampler.sample(stage, np.random.default_rng(1))
    assert setup.opponent_kind == OpponentKind.SCRIPTED
    assert setup.single_lane == "right"
