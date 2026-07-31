"""Privileged-teacher -> human-student distillation (#37)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent import obs_layout
from src.agent.distill import (
    DistillConfig,
    TeacherObs,
    _factored_kl,
    check_compatible,
    distill_update,
    teacher_view,
)
from src.agent.network import make_network, masks_to_tensors, obs_to_tensors
from src.agent.ppo import PPOConfig, RolloutBuffer
from src.simulator.constants import HAND_SIZE, PLACE_COLS, PLACE_ROWS, Side
from tests.conftest import make_engine

N_CARDS = 16
DEVICE = torch.device("cpu")
SPATIAL_SHAPE = (obs_layout.SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)
N_CELLS = PLACE_ROWS * PLACE_COLS


def _rollout(rng, n_steps=4, n_envs=4, scalar_dim=obs_layout.HUMAN_SCALAR_DIM):
    obs_shapes = {"spatial": SPATIAL_SHAPE, "cards": (HAND_SIZE + 1,),
                  "vector": (scalar_dim,)}
    mask_shapes = {"card": (HAND_SIZE + 1,), "place": (HAND_SIZE, N_CELLS)}
    buffer = RolloutBuffer(n_steps, n_envs, obs_shapes, mask_shapes)
    for _ in range(n_steps):
        card_mask = np.zeros((n_envs, HAND_SIZE + 1), bool)
        card_mask[:, 0] = True
        card_mask[:, 1] = True
        place_mask = rng.random((n_envs, HAND_SIZE, N_CELLS)) < 0.5
        place_mask[:, 0, 0] = True
        obs = {
            "spatial": rng.random((n_envs, *SPATIAL_SHAPE)).astype(np.float32),
            "cards": rng.integers(0, N_CARDS, (n_envs, HAND_SIZE + 1)),
            "vector": rng.random((n_envs, scalar_dim)).astype(np.float32),
        }
        actions = np.stack([np.ones(n_envs, np.int64),
                            rng.integers(0, N_CELLS, n_envs)], axis=1)
        for e in range(n_envs):
            actions[e, 1] = int(np.flatnonzero(place_mask[e, 0])[0])
        buffer.add(obs, {"card": card_mask, "place": place_mask}, actions,
                   rng.standard_normal(n_envs).astype(np.float32),
                   rng.standard_normal(n_envs).astype(np.float32),
                   rng.standard_normal(n_envs).astype(np.float32),
                   np.zeros(n_envs, bool))
    buffer.compute_returns(np.zeros(n_envs, np.float32), 0.99, 0.95)
    return buffer


# ------------------------------------------------------------ compatibility


def test_teacher_must_be_privileged():
    student = make_network(N_CARDS, dict(_SMALL, tier="human"))
    with pytest.raises(ValueError, match="privileged"):
        check_compatible(make_network(N_CARDS, dict(_SMALL, tier="human")), student)


def test_student_must_not_be_full_tier():
    teacher = make_network(N_CARDS, dict(_SMALL, tier="full"))
    with pytest.raises(ValueError, match="could never be used live"):
        check_compatible(teacher, make_network(N_CARDS, dict(_SMALL, tier="full")))


def test_roster_mismatch_rejected():
    with pytest.raises(ValueError, match="card rosters"):
        check_compatible(make_network(N_CARDS, dict(_SMALL, tier="full")),
                         make_network(N_CARDS + 1, dict(_SMALL, tier="human")))


def test_human_student_pairs_with_full_teacher():
    check_compatible(make_network(N_CARDS, dict(_SMALL, tier="full")),
                     make_network(N_CARDS, dict(_SMALL, tier="human")))


# ---------------------------------------------------------------- encoding


def test_teacher_view_is_the_full_tier_encoding(cards, arena, decks):
    from src.simulator.env import CRBattleEnv

    deck = [cards[n] for n in decks["training_mirror"]]
    env = CRBattleEnv(cards, arena, deck, list(deck), tier=obs_layout.TIER_HUMAN, seed=0)
    env.reset(seed=0)
    env.engine.players[Side.TOP].elixir = 6.5

    vector, spatial = teacher_view(env)
    assert vector.shape == (obs_layout.SCALAR_DIM,)
    assert spatial.shape == SPATIAL_SHAPE
    assert 0.65 in vector, "teacher must see the opponent elixir the student cannot"
    assert 0.65 not in env.build_obs(Side.BOTTOM)["vector"]


def test_human_student_shares_the_teachers_spatial_grid(cards, arena):
    """Why only the scalar vector is stored alongside the rollout."""
    engine = make_engine(cards, arena)
    ids = {n: i for i, n in enumerate(cards)}
    human = obs_layout.encode_obs(engine, Side.BOTTOM, ids, obs_layout.TIER_HUMAN)
    full = obs_layout.encode_obs(engine, Side.BOTTOM, ids, obs_layout.TIER_FULL)
    np.testing.assert_array_equal(human["spatial"], full["spatial"])
    np.testing.assert_array_equal(human["cards"], full["cards"])
    assert human["vector"].shape != full["vector"].shape


def test_teacher_obs_buffer_shares_or_stores_spatial():
    shared = TeacherObs(2, 3, obs_layout.SCALAR_DIM, True, SPATIAL_SHAPE)
    assert shared.spatial is None
    separate = TeacherObs(2, 3, obs_layout.SCALAR_DIM, False, SPATIAL_SHAPE)
    assert separate.spatial is not None
    student_flat = {"spatial": np.zeros((6, *SPATIAL_SHAPE), np.float32),
                    "cards": np.zeros((6, 5), np.int64),
                    "vector": np.zeros((6, obs_layout.HUMAN_SCALAR_DIM), np.float32)}
    flat = shared.flat(6, student_flat)
    assert flat["vector"].shape == (6, obs_layout.SCALAR_DIM)
    assert flat["spatial"] is student_flat["spatial"]


# ------------------------------------------------------------------ anneal


def test_lambda_anneals_to_zero():
    cfg = DistillConfig(lambda_start=1.0, lambda_end=0.0, anneal_over_steps=1000)
    assert cfg.weight(0) == 1.0
    assert cfg.weight(500) == pytest.approx(0.5)
    assert cfg.weight(1000) == 0.0
    assert cfg.weight(5000) == 0.0, "must not go negative past the horizon"


def test_config_from_dict_defaults():
    cfg = DistillConfig.from_dict(None)
    assert cfg.lambda_start == 1.0 and cfg.lambda_end == 0.0
    assert DistillConfig.from_dict({"lambda_start": 0.4}).lambda_start == 0.4


# -------------------------------------------------------------------- loss


def test_kl_is_zero_for_identical_policies():
    rng = np.random.default_rng(0)
    net = make_network(N_CARDS, dict(_SMALL, tier="human"))
    buffer = _rollout(rng)
    n = buffer.n_steps * buffer.n_envs
    obs = obs_to_tensors({k: v.reshape(n, *v.shape[2:])
                          for k, v in buffer.obs.items()}, DEVICE)
    masks = masks_to_tensors({k: v.reshape(n, *v.shape[2:])
                              for k, v in buffer.masks.items()}, DEVICE)
    actions = torch.as_tensor(buffer.actions.reshape(n, 2))
    kl, _ = _factored_kl(net, net, obs, obs, masks, actions)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def _mean_kl(teacher, student, buffer, t_obs) -> float:
    n = buffer.n_steps * buffer.n_envs
    flat = {k: v.reshape(n, *v.shape[2:]) for k, v in buffer.obs.items()}
    obs = obs_to_tensors(flat, DEVICE)
    t = obs_to_tensors(t_obs.flat(n, flat), DEVICE)
    masks = masks_to_tensors({k: v.reshape(n, *v.shape[2:])
                              for k, v in buffer.masks.items()}, DEVICE)
    with torch.no_grad():
        kl, _ = _factored_kl(teacher, student, t, obs, masks,
                             torch.as_tensor(buffer.actions.reshape(n, 2)))
    return float(kl.mean())


# Small trunks: these tests are about the loss, not capacity, and the full
# CNN makes them minutes long.
_SMALL = {"conv_channels": (8,), "cnn_out": 32, "fusion_mlp": 64}


def _opinionated_teacher(seed: int):
    """A teacher with a *peaked* policy. Two freshly-initialized networks
    already agree (both near-uniform), so distilling one into the other would
    measure nothing — the teacher has to actually have an opinion."""
    torch.manual_seed(seed)
    teacher = make_network(N_CARDS, dict(_SMALL, tier="full"))
    with torch.no_grad():
        for head in (teacher.card_head, teacher.place_head[-1]):
            head.weight.mul_(8.0)
            head.bias.normal_(0.0, 4.0)
    return teacher


def test_distillation_pulls_the_student_toward_the_teacher():
    """The actual claim: the KL term makes the student agree with the teacher
    more than the same PPO update without it, on student-visited states."""
    rng = np.random.default_rng(1)
    teacher = _opinionated_teacher(11)
    buffer = _rollout(rng)
    t_obs = TeacherObs(buffer.n_steps, buffer.n_envs, obs_layout.SCALAR_DIM,
                       True, SPATIAL_SHAPE)
    for t in range(buffer.n_steps):
        t_obs.add(t, rng.random((buffer.n_envs, obs_layout.SCALAR_DIM)).astype(np.float32), None)

    ppo_cfg = PPOConfig(n_steps=buffer.n_steps, batch_size=8, n_epochs=4)
    cfg = DistillConfig(value_gap_mask=1e9)  # never mask, for a clean signal

    def run(kl_weight: float) -> float:
        torch.manual_seed(7)
        student = make_network(N_CARDS, dict(_SMALL, tier="human"))
        opt = torch.optim.Adam(student.parameters(), lr=3e-3)
        before = _mean_kl(teacher, student, buffer, t_obs)
        for _ in range(4):
            distill_update(student, teacher, opt, buffer, t_obs, ppo_cfg, cfg,
                           ent_coef=0.0, kl_weight=kl_weight, device=DEVICE,
                           rng=np.random.default_rng(2))
        return _mean_kl(teacher, student, buffer, t_obs) / before

    with_kl = run(1.0)
    without_kl = run(0.0)
    assert with_kl < without_kl
    assert with_kl < 1.0, "the KL term must actually close the gap"


def test_value_gap_mask_drops_disagreeing_states():
    """States where teacher and student value estimates diverge are where the
    teacher is most likely acting on opponent elixir — the superstition the
    student must not copy."""
    rng = np.random.default_rng(3)
    teacher = make_network(N_CARDS, dict(_SMALL, tier="full"))
    student = make_network(N_CARDS, dict(_SMALL, tier="human"))
    buffer = _rollout(rng)
    t_obs = TeacherObs(buffer.n_steps, buffer.n_envs, obs_layout.SCALAR_DIM,
                       True, SPATIAL_SHAPE)
    ppo_cfg = PPOConfig(n_steps=buffer.n_steps, batch_size=16, n_epochs=1)
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)

    strict = distill_update(student, teacher, opt, buffer, t_obs, ppo_cfg,
                            DistillConfig(value_gap_mask=0.0), 0.0, 1.0,
                            DEVICE, np.random.default_rng(4))
    loose = distill_update(student, teacher, opt, buffer, t_obs, ppo_cfg,
                           DistillConfig(value_gap_mask=1e9), 0.0, 1.0,
                           DEVICE, np.random.default_rng(4))
    assert strict["kl_masked_frac"] >= loose["kl_masked_frac"]
    assert loose["kl_masked_frac"] == 0.0


def test_zero_lambda_is_plain_ppo():
    rng = np.random.default_rng(5)
    teacher = make_network(N_CARDS, dict(_SMALL, tier="full"))
    student = make_network(N_CARDS, dict(_SMALL, tier="human"))
    buffer = _rollout(rng)
    t_obs = TeacherObs(buffer.n_steps, buffer.n_envs, obs_layout.SCALAR_DIM,
                       True, SPATIAL_SHAPE)
    ppo_cfg = PPOConfig(n_steps=buffer.n_steps, batch_size=16, n_epochs=1)
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)
    stats = distill_update(student, teacher, opt, buffer, t_obs, ppo_cfg,
                           DistillConfig(), 0.0, 0.0, DEVICE,
                           np.random.default_rng(6))
    assert stats["distill_kl"] == 0.0
