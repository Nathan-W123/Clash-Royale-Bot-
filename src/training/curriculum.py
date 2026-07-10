"""Curriculum stage definitions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.simulator.cards import CONFIG_DIR


@dataclass(frozen=True)
class PromoteCriteria:
    vs: str
    win_rate: float
    matches: int


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    deck: str | None
    opponent_deck: str | None
    deck_pool: str | None
    agent_deck_pool: str | None
    opponent_deck_pool: str | None
    single_lane: str | None
    match_time: float | None
    opponents: tuple[str, ...]
    selfplay: bool
    promote: PromoteCriteria | None


def load_curriculum(path: Path | None = None) -> list[CurriculumStage]:
    path = path or CONFIG_DIR / "curriculum.yaml"
    raw = yaml.safe_load(path.read_text())
    stages = []
    for spec in raw["stages"]:
        promote = spec.get("promote")
        stages.append(
            CurriculumStage(
                name=spec["name"],
                deck=spec.get("deck"),
                opponent_deck=spec.get("opponent_deck"),
                deck_pool=spec.get("deck_pool"),
                agent_deck_pool=spec.get("agent_deck_pool"),
                opponent_deck_pool=spec.get("opponent_deck_pool"),
                single_lane=spec.get("single_lane"),
                match_time=spec.get("match_time"),
                opponents=tuple(spec.get("opponents", ["champion"])),
                selfplay=bool(spec.get("selfplay", False)),
                promote=PromoteCriteria(**promote) if promote else None,
            )
        )
    return stages


def stage_by_name(stages: list[CurriculumStage], name: str) -> CurriculumStage:
    for s in stages:
        if s.name == name:
            return s
    raise KeyError(f"unknown curriculum stage: {name}")
