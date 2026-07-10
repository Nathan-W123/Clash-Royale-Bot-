"""Training configuration loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.simulator.cards import CONFIG_DIR


@dataclass(frozen=True)
class FocusedRotationConfig:
    wins_per_deck: int
    target_win_rate: float
    min_matches_per_deck: int
    min_win_rate_per_deck: float


@dataclass(frozen=True)
class AdaptiveDeckConfig:
    rebuild_every_matches: int
    plateau_window: int
    plateau_threshold: float


@dataclass(frozen=True)
class OpponentConfig:
    skill_tier: str
    agent_deck_pool: str
    opponent_deck_pool: str
    weakness_weight: float
    sample_scripted: float
    sample_pool: float
    sample_latest: float
    scripted_bots: tuple[str, ...]


@dataclass(frozen=True)
class TrainingConfig:
    opponents: OpponentConfig
    focused_rotation: FocusedRotationConfig | None
    adaptive_deck: AdaptiveDeckConfig | None
    raw: dict = field(repr=False)


def load_training_config(path: Path | None = None) -> TrainingConfig:
    path = path or CONFIG_DIR / "training.yaml"
    raw = yaml.safe_load(path.read_text())
    opp = raw.get("opponents", {})
    league = raw.get("league", {})
    focused_raw = raw.get("focused_rotation")
    adaptive_raw = raw.get("adaptive_deck")
    focused = None
    if focused_raw:
        focused = FocusedRotationConfig(
            wins_per_deck=int(focused_raw.get("wins_per_deck", 5)),
            target_win_rate=float(focused_raw.get("target_win_rate", 0.65)),
            min_matches_per_deck=int(focused_raw.get("min_matches_per_deck", 10)),
            min_win_rate_per_deck=float(focused_raw.get("min_win_rate_per_deck", 0.55)),
        )
    adaptive = None
    if adaptive_raw:
        adaptive = AdaptiveDeckConfig(
            rebuild_every_matches=int(adaptive_raw.get("rebuild_every_matches", 25)),
            plateau_window=int(adaptive_raw.get("plateau_window", 20)),
            plateau_threshold=float(adaptive_raw.get("plateau_threshold", 0.02)),
        )
    return TrainingConfig(
        opponents=OpponentConfig(
            skill_tier=str(opp.get("skill_tier", "ultimate_champion")),
            agent_deck_pool=str(opp.get("agent_deck_pool", "stage2_pool")),
            opponent_deck_pool=str(opp.get("opponent_deck_pool", "ladder_top50")),
            weakness_weight=float(opp.get("weakness_weight", 0.35)),
            sample_scripted=float(league.get("sample_scripted", 0.15)),
            sample_pool=float(league.get("sample_pool", 0.65)),
            sample_latest=float(league.get("sample_latest", 0.20)),
            scripted_bots=tuple(opp.get("scripted_bots", ["rusher", "control", "siege", "beatdown"])),
        ),
        focused_rotation=focused,
        adaptive_deck=adaptive,
        raw=raw,
    )
