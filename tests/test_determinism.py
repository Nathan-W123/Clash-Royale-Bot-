import numpy as np

from src.simulator.constants import Side
from src.simulator.env import CRBattleEnv


def snapshot(env):
    eng = env.engine
    return (
        round(eng.time, 6),
        eng.result,
        tuple(round(t.hp, 4) for t in eng.towers),
        tuple((u.stats.name, round(u.x, 4), round(u.y, 4), round(u.hp, 4))
              for u in eng.units),
        tuple(round(eng.players[s].elixir, 6) for s in (Side.BOTTOM, Side.TOP)),
    )


def run_episode(cards, arena, seed):
    deck = [cards[n] for n in ["knight", "archers", "goblins", "giant",
                               "musketeer", "minions", "fireball", "cannon"]]
    env = CRBattleEnv(cards, arena, deck, list(deck), regulation=60.0)
    _, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    trace = []
    masks = info["masks"]
    for _ in range(200):
        choice = int(rng.choice(np.flatnonzero(masks["card"])))
        cell = int(rng.choice(np.flatnonzero(masks["place"][choice - 1]))) if choice else 0
        _, reward, terminated, _, info = env.step((choice, cell))
        trace.append(reward)
        if terminated:
            break
        masks = info["masks"]
    return snapshot(env), tuple(trace)


def test_same_seed_same_history(cards, arena):
    a = run_episode(cards, arena, seed=123)
    b = run_episode(cards, arena, seed=123)
    assert a == b


def test_different_seed_diverges(cards, arena):
    a = run_episode(cards, arena, seed=123)
    b = run_episode(cards, arena, seed=456)
    assert a != b
