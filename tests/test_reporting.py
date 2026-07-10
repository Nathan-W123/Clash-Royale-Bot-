import json

import numpy as np

from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.eval.benchmark import load_benchmark_opponents, run_benchmark
from src.eval.metrics import MatchRecord, WinLossRecord, card_usage_entropy
from src.eval.reporter import TrainingReporter
from src.simulator.constants import Side
from src.training.match_runner import run_match_detailed
from src.training.session import run_episodes


def test_win_loss_record():
    wl = WinLossRecord()
    wl = wl.with_result(True).with_result(False).with_result(None)
    assert wl.wins == 1 and wl.losses == 1 and wl.draws == 1
    assert wl.win_rate == 0.5
    assert wl.wl_ratio == 1.0


def test_card_entropy_uniform():
    assert card_usage_entropy({"a": 1, "b": 1, "c": 1, "d": 1}) > 1.3


def test_reporter_summary(cards, arena):
    reporter = TrainingReporter(run_name="test")
    reporter.record(
        MatchRecord(
            won=True,
            agent_crowns=2,
            opponent_crowns=0,
            agent_deck="training_mirror",
            opponent_deck="rusher",
            opponent_bot="champion_rusher",
            cards_played={"knight": 3, "fireball": 2},
            duration_sec=120.0,
            elixir_spent=40.0,
        )
    )
    reporter.record(
        MatchRecord(
            won=False,
            agent_crowns=0,
            opponent_crowns=1,
            agent_deck="training_mirror",
            opponent_deck="control",
            opponent_bot="champion_control",
            cards_played={"knight": 1, "giant": 4},
        )
    )
    text = reporter.summary()
    assert "Win rate: 50.0%" in text
    assert "vs Opponent Deck" in text
    assert "knight" in text


def test_reporter_json_export(cards, arena, tmp_path):
    reporter = TrainingReporter(run_name="export_test")
    reporter.record(
        MatchRecord(
            won=True,
            agent_crowns=1,
            opponent_crowns=0,
            agent_deck="rusher",
            opponent_deck="control",
            opponent_bot="control",
        )
    )
    path = reporter.export_json(tmp_path / "report.json")
    data = json.loads(path.read_text())
    assert data["overall"]["wins"] == 1
    assert data["overall"]["win_rate"] == 1.0


def test_match_report_agent_perspective(cards, arena):
    from src.bots.archetypes import RusherBot

    report = run_match_detailed(
        arena,
        [cards["knight"]] * 8,
        [cards["archers"]] * 8,
        RusherBot(),
        RusherBot(),
        seed=1,
        max_ticks=500,
        bottom_deck_name="a",
        top_deck_name="b",
    )
    assert report.duration_sec > 0
    assert report.agent_won(Side.BOTTOM) in (True, False, None)


def test_run_episodes_produces_report(cards, arena):
    cat = DeckCatalog(cards=cards)
    bot = get_bot("rusher", catalog=cat, deck_name="rusher")
    reporter = run_episodes(bot, 6, stage="one_lane", catalog=cat, seed=0)
    assert reporter.overall.total == 6
    assert "Win rate" in reporter.summary()


def test_benchmark_runs(cards, arena):
    cat = DeckCatalog(cards=cards)
    bot = get_bot("control", catalog=cat, deck_name="control")
    opponents = load_benchmark_opponents()
    assert len(opponents) >= 4
    reporter = run_benchmark(
        bot, "control", matches_per_opponent=2, catalog=cat, seed=42
    )
    assert reporter.overall.total == len(opponents) * 2
