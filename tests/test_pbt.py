"""Population-based training (#41)."""
from __future__ import annotations

import numpy as np
import pytest

from src.agent.pbt import (
    BENCHMARK_SOURCE,
    DEFAULT_SPACE,
    PBTPopulation,
    apply_hypers,
    perturb,
    reward_overrides,
    run_pbt,
    sample_hypers,
)
from src.agent.ppo import PPOConfig


def _scored(size=8, seed=0):
    pop = PBTPopulation(size=size, seed=seed)
    for i, m in enumerate(pop.members):
        pop.record(m.member_id, fitness=i / size, step=1000,
                   source=BENCHMARK_SOURCE, checkpoint=f"ckpt_{i}.pt")
    return pop


# ------------------------------------------------------------------- space


def test_samples_stay_inside_the_space():
    rng = np.random.default_rng(0)
    for _ in range(50):
        hypers = sample_hypers(DEFAULT_SPACE, rng)
        for name, (low, high, _) in DEFAULT_SPACE.items():
            assert low <= hypers[name] <= high


def test_log_scaled_axes_actually_span_orders_of_magnitude():
    rng = np.random.default_rng(1)
    lrs = [sample_hypers(DEFAULT_SPACE, rng)["lr"] for _ in range(200)]
    assert min(lrs) < 1e-4 < max(lrs)


def test_perturbation_is_clipped_back_into_the_space():
    rng = np.random.default_rng(2)
    extreme = {name: high for name, (_, high, _) in DEFAULT_SPACE.items()}
    for _ in range(20):
        extreme = perturb(extreme, DEFAULT_SPACE, rng)
        for name, (low, high, _) in DEFAULT_SPACE.items():
            assert low <= extreme[name] <= high


def test_perturbation_actually_moves_values():
    rng = np.random.default_rng(3)
    start = sample_hypers(DEFAULT_SPACE, rng)
    moved = perturb(start, DEFAULT_SPACE, rng)
    assert any(moved[k] != start[k] for k in start)


# ------------------------------------------------------- evaluation discipline


def test_fitness_from_self_play_is_rejected():
    """The whole trap this module exists to avoid: self-play win rate trends
    to 50% by construction and ranks noise."""
    pop = PBTPopulation(size=4, seed=0)
    with pytest.raises(ValueError, match="frozen benchmark"):
        pop.record(0, 0.51, step=1000, source="self_play")


def test_fitness_from_the_frozen_benchmark_is_accepted():
    pop = PBTPopulation(size=4, seed=0)
    pop.record(0, 0.62, step=1000, source=BENCHMARK_SOURCE)
    assert pop.members[0].fitness == 0.62


def test_benchmark_fitness_uses_the_frozen_roster():
    import inspect

    from src.agent import pbt

    source = inspect.getsource(pbt.benchmark_fitness)
    assert "run_benchmark" in source


# -------------------------------------------------------- exploit / explore


def test_population_needs_at_least_two_members():
    with pytest.raises(ValueError, match="at least two"):
        PBTPopulation(size=1)


def test_ranking_puts_the_best_first():
    pop = _scored()
    ranked = pop.ranked()
    assert ranked[0].fitness == max(m.fitness for m in pop.members)
    assert ranked[0].fitness > ranked[-1].fitness


def test_unscored_members_rank_last():
    pop = PBTPopulation(size=4, seed=0)
    pop.record(2, 0.4, step=1, source=BENCHMARK_SOURCE)
    assert pop.ranked()[0].member_id == 2
    assert pop.ranked()[-1].fitness is None


def test_losers_inherit_the_winners_checkpoint_and_hypers():
    pop = _scored(size=8)
    winner_before = {m.member_id: dict(m.hypers) for m in pop.ranked()}
    pairs = pop.exploit_and_explore()
    assert pairs
    for loser_id, winner_id in pairs:
        loser = pop.members[loser_id]
        assert loser.checkpoint == f"ckpt_{winner_id}.pt"
        assert loser.ancestry[-1] == winner_id
        # Inherited then perturbed, so close to the winner's but not identical.
        assert loser.hypers != winner_before[winner_id]


def test_exploited_members_must_be_rescored_before_ranking_again():
    """Carrying the winner's fitness onto the perturbed copy would let a
    hyperparameter change that made things worse survive a round unpunished."""
    pop = _scored(size=8)
    pairs = pop.exploit_and_explore()
    for loser_id, _ in pairs:
        assert pop.members[loser_id].fitness is None


def test_exploit_replaces_the_configured_quantile():
    pop = _scored(size=8)
    pop.quantile = 0.25
    assert len(pop.exploit_and_explore()) == 2


def test_exploit_is_a_noop_without_scores():
    pop = PBTPopulation(size=4, seed=0)
    assert pop.exploit_and_explore() == []


# ---------------------------------------------------------------- plumbing


def test_hypers_split_into_ppo_and_reward_parts():
    hypers = {"lr": 1e-4, "clip_range": 0.15, "reward.tower_damage": 0.4}
    cfg = apply_hypers(PPOConfig(), hypers)
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.clip_range == pytest.approx(0.15)
    assert reward_overrides(hypers) == {"tower_damage": 0.4}


def test_unknown_hyperparameters_are_ignored_not_crashed():
    cfg = apply_hypers(PPOConfig(), {"lr": 2e-4, "not_a_field": 3.0})
    assert cfg.lr == pytest.approx(2e-4)


def test_state_round_trips_through_disk(tmp_path):
    pop = _scored(size=6)
    pop.exploit_and_explore()
    path = pop.save(tmp_path / "pbt.json")
    loaded = PBTPopulation.load(path)
    assert loaded.generation == pop.generation
    assert [m.to_dict() for m in loaded.members] == [m.to_dict() for m in pop.members]


# ------------------------------------------------------------------ driver


def test_run_pbt_converges_toward_the_better_region():
    """A toy fitness landscape (lower lr is better) to exercise the loop
    without touching torch."""
    pop = PBTPopulation(size=8, seed=5)

    def train_fn(member, steps):
        return f"ckpt_{member.member_id}.pt", 1.0 - member.hypers["lr"] * 1000.0

    lr_before = float(np.mean([m.hypers["lr"] for m in pop.members]))
    run_pbt(pop, train_fn, rounds=6, steps_per_round=1000, log=lambda _: None)
    lr_after = float(np.mean([m.hypers["lr"] for m in pop.members]))
    assert lr_after < lr_before


def test_run_pbt_persists_state_each_round(tmp_path):
    pop = PBTPopulation(size=4, seed=6)
    path = tmp_path / "state.json"
    run_pbt(pop, lambda m, s: ("c.pt", 0.5), rounds=2, steps_per_round=10,
            state_path=path, log=lambda _: None)
    assert path.exists()
    assert PBTPopulation.load(path).generation == 2


def test_steps_accumulate_across_rounds():
    pop = PBTPopulation(size=4, seed=7)
    run_pbt(pop, lambda m, s: ("c.pt", 0.5), rounds=3, steps_per_round=100,
            log=lambda _: None)
    assert all(m.step == 300 for m in pop.members)
