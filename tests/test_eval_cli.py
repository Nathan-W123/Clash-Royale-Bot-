"""src.eval CLI: --checkpoint lets the benchmark evaluate a trained policy
directly, instead of only scripted-bot archetypes."""
from __future__ import annotations

import sys

import pytest

from src.agent.network import make_network
from src.agent.selfplay import save_checkpoint
from src.simulator.cards import load_cards


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    card_names = list(load_cards().keys())
    net = make_network(len(card_names), {"use_spatial": False, "conv_channels": [4, 4]})
    path = tmp_path / "tiny.pt"
    save_checkpoint(net, card_names, path)
    return path


def test_checkpoint_flag_runs_benchmark(tiny_checkpoint, monkeypatch, capsys):
    from src.eval.__main__ import main

    monkeypatch.setattr(sys, "argv", [
        "prog", "--checkpoint", str(tiny_checkpoint), "--matches", "1",
    ])
    main()
    out = capsys.readouterr().out
    assert "Training Report: benchmark_tiny" in out
    assert "Matches: 4" in out  # 1 match x 4 benchmark opponents
