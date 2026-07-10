"""League of frozen checkpoints with PFSP opponent sampling.

Pool layout on disk:
  checkpoints/<run>/pool/step_<N>.pt      rolling window (pool_max newest)
  checkpoints/<run>/pool/anchor_<N>.pt    permanent anchors (every anchor_every)
  checkpoints/<run>/ledger.json           win-rate ledger vs each pool member

PFSP weight w ∝ p(1-p) prioritizes near-even opponents; unseen members get
the maximum weight so they are probed quickly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.agent.network import PolicyNetwork
from src.agent.selfplay import PolicyBot, load_checkpoint, save_checkpoint


class CheckpointPool:
    def __init__(self, root: Path, pool_max: int = 20, anchor_every: int = 500_000):
        self.root = Path(root)
        self.pool_dir = self.root / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "ledger.json"
        self.pool_max = pool_max
        self.anchor_every = anchor_every
        self._next_anchor = anchor_every
        self.ledger: dict[str, dict[str, float]] = {}
        if self.ledger_path.exists():
            self.ledger = json.loads(self.ledger_path.read_text())
        self._cache: dict[str, PolicyBot] = {}

    # ------------------------------------------------------------- pool ops

    def members(self) -> list[Path]:
        return sorted(self.pool_dir.glob("*.pt"))

    def snapshot(self, net: PolicyNetwork, card_names: list[str], step: int) -> Path:
        if step >= self._next_anchor:
            path = self.pool_dir / f"anchor_{step}.pt"
            self._next_anchor += self.anchor_every
        else:
            path = self.pool_dir / f"step_{step}.pt"
        save_checkpoint(net, card_names, path)
        self._prune()
        return path

    def _prune(self) -> None:
        rolling = sorted(self.pool_dir.glob("step_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
        for path in rolling[:-self.pool_max]:
            path.unlink()
            self.ledger.pop(path.name, None)
            self._cache.pop(path.name, None)

    # ------------------------------------------------------------- sampling

    def _pfsp_weights(self, names: list[str]) -> np.ndarray:
        w = np.empty(len(names))
        for i, name in enumerate(names):
            rec = self.ledger.get(name)
            if not rec or rec.get("games", 0) < 5:
                w[i] = 0.25  # max of p(1-p): probe unknowns first
            else:
                p = rec["wins"] / rec["games"]
                w[i] = max(p * (1.0 - p), 0.01)
        return w / w.sum()

    def sample_opponent(self, rng: np.random.Generator) -> PolicyBot | None:
        members = self.members()
        if not members:
            return None
        names = [p.name for p in members]
        idx = int(rng.choice(len(members), p=self._pfsp_weights(names)))
        return self._load_bot(members[idx])

    def anchors(self) -> list[Path]:
        return sorted(self.pool_dir.glob("anchor_*.pt"),
                      key=lambda p: int(p.stem.split("_")[1]))

    def _load_bot(self, path: Path) -> PolicyBot:
        bot = self._cache.get(path.name)
        if bot is None:
            net, card_names = load_checkpoint(path)
            bot = PolicyBot(net, card_names, name=path.stem, deterministic=False)
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[path.name] = bot
        return bot

    # ------------------------------------------------------------- ledger

    def record_result(self, opponent_name: str, won: bool | None) -> None:
        """won=None records a draw (half a win)."""
        rec = self.ledger.setdefault(opponent_name, {"wins": 0.0, "games": 0})
        rec["games"] += 1
        rec["wins"] += 0.5 if won is None else float(won)

    def save_ledger(self) -> None:
        self.ledger_path.write_text(json.dumps(self.ledger, indent=1))
