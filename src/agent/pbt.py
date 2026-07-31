"""Population-based training over PPO hyperparameters (#41).

The current settings (`lr 3e-4`, `clip 0.2`, `gamma 0.997`, the entropy
schedule, the reward weights) were hand-picked and never searched. PBT
trains a population in parallel, periodically copies the weights *and*
hyperparameters of strong members onto weak ones (exploit), and perturbs the
copied hyperparameters (explore) — so the schedule is discovered rather than
guessed, and it can be non-stationary, which a one-shot sweep cannot
express.

**Selection is on the frozen benchmark, never on self-play win rate.**
Self-play win rate trends to ~50% by construction (CLAUDE.md, "Evaluation
Discipline"); ranking a population by it ranks noise, and it is the single
easiest way to waste a week of compute. `benchmark_fitness` here goes
through `src/eval/benchmark.py` and the frozen `configs/eval.yaml` roster
for exactly that reason, and `PBTPopulation.record` refuses a fitness that
did not come from it.

**Throughput caveat.** PBT needs many concurrent runs to be worth anything,
so it is gated on the parallel-rollout work (#24/#25). `run_pbt` therefore
takes a `train_fn`, and the shipped `sequential_scheduler` trains members
round-robin in one process: correct, and slow enough that it is really only
for validating the machinery. Hand it a process-pool scheduler once
throughput lands.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

# Search space. Bounds are wide on purpose: the point is to discover that a
# hand-picked value was wrong, which a narrow box around it cannot do.
# `log` sampling for anything spanning orders of magnitude.
DEFAULT_SPACE: dict[str, tuple[float, float, str]] = {
    "lr": (5e-5, 1e-3, "log"),
    "clip_range": (0.1, 0.3, "linear"),
    "gamma": (0.99, 0.999, "linear"),
    "gae_lambda": (0.9, 0.99, "linear"),
    "ent_coef_start": (0.002, 0.03, "log"),
    "vf_coef": (0.25, 1.0, "linear"),
    "reward.tower_damage": (0.1, 0.6, "linear"),
    "reward.elixir_trade": (0.005, 0.05, "log"),
    "reward.leak": (0.001, 0.02, "log"),
}

# Fraction of the population replaced each PBT round.
DEFAULT_QUANTILE = 0.25
# Multiplicative jitter applied to an inherited hyperparameter.
PERTURB_FACTORS = (0.8, 1.25)

BENCHMARK_SOURCE = "frozen_benchmark"


def sample_hypers(space: Mapping[str, tuple[float, float, str]],
                  rng: np.random.Generator) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (low, high, scale) in space.items():
        if scale == "log":
            out[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            out[name] = float(rng.uniform(low, high))
    return out


def perturb(hypers: Mapping[str, float],
            space: Mapping[str, tuple[float, float, str]],
            rng: np.random.Generator) -> dict[str, float]:
    """Jitter inherited hyperparameters, clipped back into the space.

    Multiplicative rather than additive so a learning rate and a reward
    weight are perturbed on comparable relative scales.
    """
    out = dict(hypers)
    for name, (low, high, _) in space.items():
        if name not in out:
            continue
        factor = float(rng.choice(PERTURB_FACTORS))
        out[name] = float(np.clip(out[name] * factor, low, high))
    return out


@dataclass
class Member:
    member_id: int
    hypers: dict[str, float]
    step: int = 0
    fitness: float | None = None
    checkpoint: str | None = None
    generation: int = 0
    ancestry: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class PBTPopulation:
    """Bookkeeping for a PBT run: sampling, ranking, exploit, explore."""

    def __init__(
        self,
        size: int,
        space: Mapping[str, tuple[float, float, str]] | None = None,
        seed: int = 0,
        quantile: float = DEFAULT_QUANTILE,
    ):
        if size < 2:
            raise ValueError("PBT needs at least two members to exploit anything")
        self.space = dict(space or DEFAULT_SPACE)
        self.rng = np.random.default_rng(seed)
        self.quantile = quantile
        self.members = [Member(member_id=i, hypers=sample_hypers(self.space, self.rng))
                        for i in range(size)]
        self.generation = 0

    def __len__(self) -> int:
        return len(self.members)

    # ------------------------------------------------------------- fitness

    def record(self, member_id: int, fitness: float, step: int,
               source: str, checkpoint: str | None = None) -> None:
        """Attach a benchmark score to a member.

        `source` must be `BENCHMARK_SOURCE`. This is a guard rail with teeth:
        the failure mode PBT invites is quietly ranking members by whatever
        number is closest to hand — which in this codebase is the self-play
        win rate sitting in `train_log.csv`, and which carries no signal.
        """
        if source != BENCHMARK_SOURCE:
            raise ValueError(
                f"PBT fitness must come from the frozen benchmark ({BENCHMARK_SOURCE!r}), "
                f"got {source!r}. Self-play win rate trends to 50% by construction and "
                f"is not a progress signal — see CLAUDE.md, Evaluation Discipline.")
        member = self.members[member_id]
        member.fitness = float(fitness)
        member.step = int(step)
        if checkpoint is not None:
            member.checkpoint = str(checkpoint)

    def ranked(self) -> list[Member]:
        """Best first. Unscored members sort last."""
        return sorted(self.members,
                      key=lambda m: (m.fitness is not None,
                                     m.fitness if m.fitness is not None else 0.0),
                      reverse=True)

    # ---------------------------------------------------- exploit / explore

    def exploit_and_explore(self) -> list[tuple[int, int]]:
        """Replace the bottom quantile from the top. Returns (loser, winner)
        pairs so the caller can copy weights; hyperparameters are already
        updated in place here."""
        scored = [m for m in self.ranked() if m.fitness is not None]
        if len(scored) < 2:
            return []
        n_replace = max(1, int(len(scored) * self.quantile))
        winners = scored[:n_replace]
        losers = scored[-n_replace:]
        pairs: list[tuple[int, int]] = []
        for loser, winner in zip(losers, winners):
            if loser.member_id == winner.member_id:
                continue
            loser.hypers = perturb(winner.hypers, self.space, self.rng)
            loser.checkpoint = winner.checkpoint
            loser.fitness = None          # must be re-scored before ranking again
            loser.generation += 1
            loser.ancestry = [*winner.ancestry, winner.member_id]
            pairs.append((loser.member_id, winner.member_id))
        self.generation += 1
        return pairs

    # -------------------------------------------------------------- state

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"generation": self.generation,
             "space": {k: list(v) for k, v in self.space.items()},
             "members": [m.to_dict() for m in self.members]}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PBTPopulation":
        raw = json.loads(Path(path).read_text())
        space = {k: (float(v[0]), float(v[1]), str(v[2])) for k, v in raw["space"].items()}
        pop = cls(size=len(raw["members"]), space=space)
        pop.generation = int(raw["generation"])
        pop.members = [Member(**m) for m in raw["members"]]
        return pop


# ------------------------------------------------------------------ fitness


def benchmark_fitness(
    bot,
    agent_deck_name: str = "training_mirror",
    matches_per_opponent: int = 20,
    seed: int = 0,
) -> float:
    """Win rate against the frozen benchmark roster (draws count half).

    Goes through `run_benchmark` so the opponents are exactly the permanent
    tripwire set in `configs/eval.yaml` — never a stage's current opponents,
    which move as the curriculum advances and would make scores from
    different generations incomparable.
    """
    from src.eval.benchmark import run_benchmark

    reporter = run_benchmark(bot, agent_deck_name,
                             matches_per_opponent=matches_per_opponent,
                             seed=seed, run_name="pbt")
    return float(reporter.overall.win_rate)


def apply_hypers(ppo_cfg, hypers: Mapping[str, float]):
    """Return a `PPOConfig` with the non-`reward.*` hyperparameters applied."""
    fields = {k: v for k, v in hypers.items() if not k.startswith("reward.")}
    return replace(ppo_cfg, **{k: type(getattr(ppo_cfg, k))(v)
                               for k, v in fields.items() if hasattr(ppo_cfg, k)})


def reward_overrides(hypers: Mapping[str, float]) -> dict[str, float]:
    """The `reward.*` entries, stripped of their prefix."""
    return {k.split(".", 1)[1]: v for k, v in hypers.items() if k.startswith("reward.")}


# ----------------------------------------------------------------- driver

TrainFn = Callable[[Member, int], tuple[str, float]]


def sequential_scheduler(train_fn: TrainFn, members: list[Member],
                         steps: int) -> list[tuple[int, str, float]]:
    """Train every member in turn, in this process.

    Enough to validate the machinery end to end, and far too slow to search
    with — PBT's value comes from concurrency. Swap in a process-pool
    scheduler once #24/#25 land.
    """
    return [(m.member_id, *train_fn(m, steps)) for m in members]


def run_pbt(
    population: PBTPopulation,
    train_fn: TrainFn,
    rounds: int,
    steps_per_round: int,
    scheduler=sequential_scheduler,
    state_path: Path | str | None = None,
    log: Callable[[str], None] = print,
) -> PBTPopulation:
    """Alternate {train every member} and {exploit + explore}.

    `train_fn(member, steps)` trains one member for `steps` environment
    steps starting from `member.checkpoint` (None = fresh) under
    `member.hypers`, and returns `(checkpoint_path, benchmark_win_rate)`.
    """
    for round_index in range(rounds):
        results = scheduler(train_fn, population.members, steps_per_round)
        for member_id, checkpoint, fitness in results:
            population.record(member_id, fitness,
                              step=population.members[member_id].step + steps_per_round,
                              source=BENCHMARK_SOURCE, checkpoint=checkpoint)
        best = population.ranked()[0]
        log(f"[pbt] round {round_index + 1}/{rounds}  best member {best.member_id} "
            f"benchmark WR {best.fitness:.3f}  hypers "
            + " ".join(f"{k}={v:.4g}" for k, v in sorted(best.hypers.items())))
        pairs = population.exploit_and_explore()
        for loser, winner in pairs:
            log(f"[pbt]   member {loser} <- {winner} (exploit + perturb)")
        if state_path is not None:
            population.save(state_path)
    return population
