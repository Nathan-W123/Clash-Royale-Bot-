"""SubprocVecEnv must be behaviourally identical to SyncVecEnv.

Parallelism is only worth anything if it does not change results, so these
compare the two implementations directly rather than testing the parallel one
in isolation. Kept small because each test pays real process-spawn cost on
Windows.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.simulator.subproc_vec_env import SubprocVecEnv, _split
from src.simulator.vec_env import SyncVecEnv


def _make_env():
    """Top-level factory: must be importable by name in a spawned worker."""
    from src.decks.catalog import DeckCatalog
    from src.simulator.cards import load_arena, load_cards
    from src.simulator.env import CRBattleEnv

    cards = load_cards()
    arena = load_arena()
    deck = DeckCatalog().resolve("training_mirror")
    return CRBattleEnv(cards, arena, deck, list(deck), use_spatial=False)


def test_split_is_contiguous_and_covers_everything():
    for n, k in ((8, 4), (8, 3), (5, 5), (7, 2), (1, 1)):
        slices = _split(n, k)
        assert len(slices) == k
        assert slices[0].start == 0
        assert slices[-1].stop == n
        for a, b in zip(slices, slices[1:]):
            assert a.stop == b.start  # no gaps, no overlap
        assert sum(s.stop - s.start for s in slices) == n
        # near-even: no worker carries 2+ more envs than another
        sizes = [s.stop - s.start for s in slices]
        assert max(sizes) - min(sizes) <= 1


@pytest.mark.slow
def test_matches_sync_vec_env_step_for_step():
    n = 4
    sync = SyncVecEnv([_make_env for _ in range(n)])
    sub = SubprocVecEnv([_make_env for _ in range(n)], n_workers=2)
    try:
        o_sync, m_sync = sync.reset(seed=42)
        o_sub, m_sub = sub.reset(seed=42)
        for k in o_sync:
            assert np.allclose(o_sync[k], o_sub[k]), f"reset obs mismatch: {k}"
        for k in m_sync:
            assert np.array_equal(m_sync[k], m_sub[k]), f"reset mask mismatch: {k}"

        rng = np.random.default_rng(0)
        for step in range(12):
            actions = np.stack([
                np.array([rng.integers(0, 5), rng.integers(0, 144)]) for _ in range(n)
            ])
            a_obs, a_r, a_d, a_m, _ = sync.step(actions)
            b_obs, b_r, b_d, b_m, _ = sub.step(actions)
            assert np.allclose(a_r, b_r), f"reward mismatch at step {step}"
            assert np.array_equal(a_d, b_d), f"done mismatch at step {step}"
            for k in a_obs:
                assert np.allclose(a_obs[k], b_obs[k]), f"obs {k} at step {step}"
    finally:
        sub.close()


@pytest.mark.slow
def test_shapes_and_ordering_preserved():
    n = 5  # deliberately not divisible by worker count
    sub = SubprocVecEnv([_make_env for _ in range(n)], n_workers=2)
    try:
        obs, masks = sub.reset(seed=1)
        assert all(v.shape[0] == n for v in obs.values())
        assert all(v.shape[0] == n for v in masks.values())
        o, r, d, m, infos = sub.step(np.zeros((n, 2), dtype=np.int64))
        assert r.shape == (n,) and d.shape == (n,)
        assert len(infos) == n
        assert all(v.shape[0] == n for v in o.values())
    finally:
        sub.close()


@pytest.mark.slow
def test_set_attr_reaches_workers():
    sub = SubprocVecEnv([_make_env for _ in range(2)], n_workers=2)
    try:
        sub.reset(seed=0)
        sub.set_attr("decision_ticks", 7)
        vals = []
        for remote in sub.remotes:
            remote.send(("call", ("__getattribute__", ("decision_ticks",), {})))
            vals.extend(remote.recv())
        assert vals == [7, 7]
    finally:
        sub.close()


@pytest.mark.slow
def test_close_is_idempotent_and_reaps_workers():
    sub = SubprocVecEnv([_make_env for _ in range(2)], n_workers=2)
    sub.reset(seed=0)
    sub.close()
    sub.close()  # must not raise
    assert all(not p.is_alive() for p in sub.procs)
