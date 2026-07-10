import numpy as np

from src.simulator.constants import HAND_SIZE, PLACE_COLS, CardType, Side
from src.simulator.env import CRBattleEnv
from tests.conftest import force_hand, slot_of


def make_env(cards, arena, **kwargs):
    deck = [cards[n] for n in ["knight", "archers", "goblins", "giant",
                               "musketeer", "minions", "fireball", "cannon"]]
    return CRBattleEnv(cards, arena, deck, list(deck), seed=3, **kwargs)


def rows_of(mask):
    return {c // PLACE_COLS for c in np.flatnonzero(mask)}


def test_noop_always_legal_and_broke_player_fully_masked(cards, arena):
    env = make_env(cards, arena)
    env.reset(seed=0)
    env.engine.players[Side.BOTTOM].elixir = 0.5
    masks = env.build_masks(Side.BOTTOM)
    assert masks["card"][0]
    assert not masks["card"][1:].any()
    assert not masks["place"].any()


def test_troop_mask_own_half_no_river(cards, arena):
    env = make_env(cards, arena)
    env.reset(seed=0)
    force_hand(env.engine.players[Side.BOTTOM], cards,
               ["knight", "cannon", "fireball", "giant"])
    env.engine.players[Side.BOTTOM].elixir = 10.0
    masks = env.build_masks(Side.BOTTOM)
    troop_slot = 0
    rows = rows_of(masks["place"][troop_slot])
    # Own half: cell centers y=1..13 are rows 0..6; row 7 (y=15) is river.
    assert rows == set(range(7))


def test_spell_mask_full_grid(cards, arena):
    env = make_env(cards, arena)
    env.reset(seed=0)
    force_hand(env.engine.players[Side.BOTTOM], cards,
               ["knight", "cannon", "fireball", "giant"])
    env.engine.players[Side.BOTTOM].elixir = 10.0
    masks = env.build_masks(Side.BOTTOM)
    assert masks["place"][2].all()


def test_building_mask_own_half_even_with_pocket(cards, arena):
    env = make_env(cards, arena)
    env.reset(seed=0)
    for t in env.engine.towers:
        if t.side == Side.TOP and t.kind == "princess_left":
            t.hp = 0.0
    force_hand(env.engine.players[Side.BOTTOM], cards,
               ["knight", "cannon", "fireball", "giant"])
    env.engine.players[Side.BOTTOM].elixir = 10.0
    masks = env.build_masks(Side.BOTTOM)
    b_slot, t_slot = 1, 0
    assert rows_of(masks["place"][b_slot]) == set(range(7))
    troop_rows = rows_of(masks["place"][t_slot])
    assert troop_rows > set(range(7))  # pocket rows unlocked for troops
    # Pocket only on the left lane (cols with x<9) and up to the princess row.
    pocket_cells = np.flatnonzero(masks["place"][t_slot])
    pocket = [(c // PLACE_COLS, c % PLACE_COLS) for c in pocket_cells if c // PLACE_COLS > 7]
    assert pocket and all(col <= 3 for _, col in pocket)
    assert all(row <= 12 for row, _ in pocket)  # princess line y=25.5 -> row 12


def test_perspective_flip_mask_for_top(cards, arena):
    env = make_env(cards, arena)
    env.reset(seed=0)
    force_hand(env.engine.players[Side.TOP], cards,
               ["knight", "cannon", "fireball", "giant"])
    env.engine.players[Side.TOP].elixir = 10.0
    masks = env.build_masks(Side.TOP)
    # In TOP's own frame the own half is also rows 0..6.
    assert rows_of(masks["place"][0]) == set(range(7))


def test_single_lane_masks_left_half(cards, arena):
    env = make_env(cards, arena, lanes="right")
    env.reset(seed=0)
    env.engine.players[Side.BOTTOM].elixir = 10.0
    masks = env.build_masks(Side.BOTTOM)
    hand = env.engine.players[Side.BOTTOM].hand
    for slot in range(HAND_SIZE):
        cols = {c % PLACE_COLS for c in np.flatnonzero(masks["place"][slot])}
        assert all(col >= 4 for col in cols), f"{hand[slot].name} allowed in left half"


def test_unmasked_actions_never_illegal(cards, arena):
    """Property sweep: sampling only unmasked actions never trips the
    illegal-action fallback across a full episode."""
    env = make_env(cards, arena)
    _, info = env.reset(seed=1)
    rng = np.random.default_rng(1)
    masks = info["masks"]
    for _ in range(300):
        legal_choices = np.flatnonzero(masks["card"])
        choice = int(rng.choice(legal_choices))
        cell = 0
        if choice > 0:
            cell = int(rng.choice(np.flatnonzero(masks["place"][choice - 1])))
        _, _, terminated, _, info = env.step((choice, cell))
        if terminated:
            break
        masks = info["masks"]
    assert env._illegal == 0
