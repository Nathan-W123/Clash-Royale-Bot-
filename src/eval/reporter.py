"""Training and eval reporting: W/L, matchups, card usage, TensorBoard."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.eval.metrics import MatchRecord, WinLossRecord, card_usage_entropy, merge_counts
from src.simulator.constants import Side
from src.training.match_runner import MatchReport


@dataclass
class TrainingReporter:
    """Accumulates match results and produces human-readable + JSON reports."""

    run_name: str = "default"
    records: list[MatchRecord] = field(default_factory=list)
    _overall: WinLossRecord = field(default_factory=WinLossRecord)
    _by_opponent_deck: dict[str, WinLossRecord] = field(default_factory=dict)
    _by_opponent_bot: dict[str, WinLossRecord] = field(default_factory=dict)
    _by_agent_deck: dict[str, WinLossRecord] = field(default_factory=dict)
    _matchup: dict[str, dict[str, WinLossRecord]] = field(default_factory=dict)
    _card_plays: dict[str, int] = field(default_factory=dict)
    _total_crowns_for: int = 0
    _total_crowns_against: int = 0
    _total_duration: float = 0.0
    _total_elixir_spent: float = 0.0
    _total_elixir_leaked: float = 0.0
    card_scores: dict | None = None
    focused_progress: dict | None = None

    def record(self, match: MatchRecord) -> None:
        self.records.append(match)
        self._overall = self._overall.with_result(match.won)
        self._by_opponent_deck[match.opponent_deck] = self._by_opponent_deck.get(
            match.opponent_deck, WinLossRecord()
        ).with_result(match.won)
        self._by_opponent_bot[match.opponent_bot] = self._by_opponent_bot.get(
            match.opponent_bot, WinLossRecord()
        ).with_result(match.won)
        self._by_agent_deck[match.agent_deck] = self._by_agent_deck.get(
            match.agent_deck, WinLossRecord()
        ).with_result(match.won)
        row = self._matchup.setdefault(match.agent_deck, {})
        row[match.opponent_deck] = row.get(match.opponent_deck, WinLossRecord()).with_result(
            match.won
        )
        merge_counts(self._card_plays, match.cards_played)
        self._total_crowns_for += match.agent_crowns
        self._total_crowns_against += match.opponent_crowns
        self._total_duration += match.duration_sec
        self._total_elixir_spent += match.elixir_spent
        self._total_elixir_leaked += match.elixir_leaked

    def record_report(
        self,
        report: MatchReport,
        *,
        agent_side: Side = Side.BOTTOM,
        agent_deck: str = "",
        opponent_deck: str = "",
        opponent_bot: str = "",
        opponent_kind: str = "scripted",
        stage: str = "",
        training_step: int | None = None,
    ) -> None:
        deck_agent = agent_deck or report.bottom_deck_name
        deck_opp = opponent_deck or report.top_deck_name
        if agent_side == Side.TOP:
            deck_agent = agent_deck or report.top_deck_name
            deck_opp = opponent_deck or report.bottom_deck_name
        self.record(
            MatchRecord(
                won=report.agent_won(agent_side),
                agent_crowns=report.agent_crowns(agent_side),
                opponent_crowns=report.opponent_crowns(agent_side),
                agent_deck=deck_agent,
                opponent_deck=deck_opp,
                opponent_bot=opponent_bot,
                opponent_kind=opponent_kind,
                stage=stage,
                duration_sec=report.duration_sec,
                cards_played=dict(report.cards_for(agent_side)),
                elixir_spent=(
                    report.bottom_elixir_spent
                    if agent_side == Side.BOTTOM
                    else report.top_elixir_spent
                ),
                elixir_leaked=(
                    report.bottom_elixir_leaked
                    if agent_side == Side.BOTTOM
                    else report.top_elixir_leaked
                ),
                training_step=training_step,
            )
        )

    def win_rates_by_bot(self) -> dict[str, float]:
        """Win rate (decided games) against each opponent bot."""
        return {name: rec.win_rate for name, rec in self._by_opponent_bot.items()}

    @property
    def overall(self) -> WinLossRecord:
        return self._overall

    @property
    def card_entropy(self) -> float:
        return card_usage_entropy(self._card_plays)

    def summary(self) -> str:
        o = self._overall
        lines = [
            f"=== Training Report: {self.run_name} ===",
            f"Matches: {o.total}  |  W {o.wins}  L {o.losses}  D {o.draws}",
            f"Win rate: {o.win_rate:.1%}  |  W/L ratio: {o.wl_ratio:.2f}",
        ]
        if o.total:
            n = o.total
            lines.append(
                f"Avg crowns: {self._total_crowns_for / n:.2f} for  "
                f"{self._total_crowns_against / n:.2f} against"
            )
            lines.append(
                f"Avg match: {self._total_duration / n:.1f}s  |  "
                f"Elixir spent: {self._total_elixir_spent / n:.1f}  "
                f"leaked: {self._total_elixir_leaked / n:.2f}"
            )
            lines.append(f"Card usage entropy: {self.card_entropy:.2f} nats")

        if self._by_opponent_deck:
            lines.append("")
            lines.append("--- vs Opponent Deck ---")
            for name, wl in sorted(self._by_opponent_deck.items()):
                lines.append(
                    f"  {name:16s}  {wl.wins:4d}W {wl.losses:4d}L  "
                    f"WR {wl.win_rate:5.1%}  W/L {wl.wl_ratio:.2f}"
                )

        if self._by_opponent_bot:
            lines.append("")
            lines.append("--- vs Opponent Bot ---")
            for name, wl in sorted(self._by_opponent_bot.items()):
                lines.append(
                    f"  {name:20s}  {wl.wins:4d}W {wl.losses:4d}L  "
                    f"WR {wl.win_rate:5.1%}"
                )

        if self._card_plays:
            lines.append("")
            lines.append("--- Card Usage (agent) ---")
            total = sum(self._card_plays.values())
            for name, count in sorted(self._card_plays.items(), key=lambda x: -x[1]):
                pct = 100.0 * count / total
                lines.append(f"  {name:14s}  {count:4d}  ({pct:4.1f}%)")

        weak = [
            (deck, wl)
            for deck, wl in self._by_opponent_deck.items()
            if wl.total >= 3 and wl.win_rate < 0.5
        ]
        if weak:
            lines.append("")
            lines.append("--- Needs Work (WR < 50%, 3+ games) ---")
            for deck, wl in sorted(weak, key=lambda x: x[1].win_rate):
                lines.append(f"  {deck}: {wl.win_rate:.1%} ({wl.wins}W-{wl.losses}L)")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        o = self._overall
        out = {
            "run_name": self.run_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall": {
                "wins": o.wins,
                "losses": o.losses,
                "draws": o.draws,
                "win_rate": o.win_rate,
                "wl_ratio": o.wl_ratio,
                "matches": o.total,
            },
            "avg_crowns_for": self._total_crowns_for / o.total if o.total else 0.0,
            "avg_crowns_against": self._total_crowns_against / o.total if o.total else 0.0,
            "avg_duration_sec": self._total_duration / o.total if o.total else 0.0,
            "card_entropy": self.card_entropy,
            "card_usage": dict(self._card_plays),
            "by_opponent_deck": {
                k: {"wins": v.wins, "losses": v.losses, "draws": v.draws, "win_rate": v.win_rate}
                for k, v in self._by_opponent_deck.items()
            },
            "by_opponent_bot": {
                k: {"wins": v.wins, "losses": v.losses, "draws": v.draws, "win_rate": v.win_rate}
                for k, v in self._by_opponent_bot.items()
            },
            "matchup_matrix": {
                agent: {
                    opp: {"wins": wl.wins, "losses": wl.losses, "win_rate": wl.win_rate}
                    for opp, wl in opps.items()
                }
                for agent, opps in self._matchup.items()
            },
        }
        if self.card_scores is not None:
            out["card_scores"] = self.card_scores
        if self.focused_progress is not None:
            out["focused_progress"] = self.focused_progress
        return out

    def export_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def log_tensorboard(self, log_dir: Path | str, step: int) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return
        writer = SummaryWriter(str(log_dir))
        o = self._overall
        writer.add_scalar("eval/win_rate", o.win_rate, step)
        writer.add_scalar("eval/wl_ratio", o.wl_ratio, step)
        writer.add_scalar("eval/wins", o.wins, step)
        writer.add_scalar("eval/losses", o.losses, step)
        writer.add_scalar("eval/card_entropy", self.card_entropy, step)
        if o.total:
            writer.add_scalar("eval/avg_crowns_for", self._total_crowns_for / o.total, step)
            writer.add_scalar("eval/avg_crowns_against", self._total_crowns_against / o.total, step)
        for deck, wl in self._by_opponent_deck.items():
            writer.add_scalar(f"eval/vs_deck/{deck}/win_rate", wl.win_rate, step)
        writer.flush()
        writer.close()
