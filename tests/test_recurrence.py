"""Recurrent policy (#27).

The restricted tier cannot see enemy troops, so the only way to represent
"a push is coming" is across time. Recurrence is easy to add and easy to get
subtly wrong, so these target the specific failure modes: memory that leaks
across episode boundaries, stale hidden states in the update, and a policy
that is silently evaluated as if it had none.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent import obs_layout
from src.agent.network import make_network, masks_to_tensors, obs_to_tensors
from src.agent.ppo import (
    PPOConfig,
    RecurrentRolloutBuffer,
    ppo_update_recurrent,
)
from src.agent.selfplay import PolicyBot
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.constants import Side
from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv

RECURRENT = {"tier": obs_layout.TIER_RESTRICTED, "use_recurrence": True}


@pytest.fixture(scope="module")
def world():
    cards = load_cards()
    return cards, load_arena(), DeckCatalog().resolve("training_mirror")


@pytest.fixture(scope="module")
def net(world):
    cards, _, _ = world
    return make_network(len(cards), RECURRENT)


def _env(world, **kw):
    cards, arena, deck = world
    return CRBattleEnv(cards, arena, deck, list(deck),
                       tier=obs_layout.TIER_RESTRICTED, **kw)


# ------------------------------------------------------------------- shapes


def test_recurrent_network_builds_a_gru(net):
    assert net.config.use_recurrence
    assert hasattr(net, "gru")
    # Default hidden width matches fusion_mlp so the heads are untouched.
    assert net.config.hidden_size == net.config.fusion_mlp
    assert not hasattr(net, "gru_proj")


def test_feedforward_network_has_no_gru(world):
    cards, _, _ = world
    plain = make_network(len(cards), {"tier": obs_layout.TIER_RESTRICTED})
    assert not plain.config.use_recurrence
    assert not hasattr(plain, "gru")
    assert plain.initial_hidden(4, torch.device("cpu")) is None


def test_custom_hidden_width_projects_back_to_head_width(world):
    cards, _, _ = world
    n = make_network(len(cards), {**RECURRENT, "gru_hidden": 96})
    assert n.config.hidden_size == 96
    assert hasattr(n, "gru_proj")  # heads still expect fusion_mlp


def test_initial_hidden_shape(net):
    h = net.initial_hidden(5, torch.device("cpu"))
    assert h.shape == (1, 5, net.config.hidden_size)
    assert torch.count_nonzero(h) == 0


# ------------------------------------------------------------------- memory


def test_hidden_state_actually_evolves(world, net):
    env = _env(world)
    obs, info = env.reset(seed=0)
    device = torch.device("cpu")
    h = net.initial_hidden(1, device)
    _, _, _, h1 = net.act_recurrent(
        obs_to_tensors(obs, device), masks_to_tensors(info["masks"], device), h)
    assert not torch.equal(h, h1), "GRU produced no state change"


def test_identical_input_different_memory_gives_different_output(world, net):
    """The point of recurrence: the same observation can mean different
    things depending on what came before."""
    env = _env(world)
    obs, info = env.reset(seed=1)
    device = torch.device("cpu")
    o = obs_to_tensors(obs, device)
    m = masks_to_tensors(info["masks"], device)

    zero = net.initial_hidden(1, device)
    other = torch.randn_like(zero)
    with torch.no_grad():
        f_a, _ = net.recurrent_step(net.trunk(o), zero)
        f_b, _ = net.recurrent_step(net.trunk(o), other)
    assert not torch.allclose(f_a, f_b)


def test_done_flag_zeroes_memory(net):
    """Carrying memory across a match boundary would blend two unrelated
    games — the vec-env has already auto-reset those envs."""
    device = torch.device("cpu")
    feat = torch.randn(3, net.config.fusion_mlp)
    hidden = torch.randn(1, 3, net.config.hidden_size)
    done = torch.tensor([True, False, True])
    with torch.no_grad():
        out_reset, _ = net.recurrent_step(feat, hidden, done)
        # Envs 0 and 2 must behave exactly as if starting from zero memory.
        zeroed = hidden * (~done).float().view(1, -1, 1)
        out_manual, _ = net.recurrent_step(feat, zeroed, None)
    assert torch.allclose(out_reset, out_manual)


def test_reset_env_matches_a_fresh_start(net):
    """Stronger form: a done env's next output equals what it would produce
    from a genuinely fresh hidden state."""
    device = torch.device("cpu")
    feat = torch.randn(2, net.config.fusion_mlp)
    dirty = torch.randn(1, 2, net.config.hidden_size)
    with torch.no_grad():
        after_done, _ = net.recurrent_step(feat, dirty, torch.tensor([True, True]))
        from_scratch, _ = net.recurrent_step(feat, net.initial_hidden(2, device), None)
    assert torch.allclose(after_done, from_scratch)


# -------------------------------------------------------------- PPO update


def _rollout(world, net, n_steps=6, n_envs=4):
    envs = SyncVecEnv([lambda: _env(world) for _ in range(n_envs)])
    obs, masks = envs.reset(seed=3)
    device = torch.device("cpu")
    buf = RecurrentRolloutBuffer(n_steps, n_envs,
                                 {k: v.shape[1:] for k, v in obs.items()},
                                 {k: v.shape[1:] for k, v in masks.items()},
                                 net.config.hidden_size)
    hidden = net.initial_hidden(n_envs, device)
    buf.set_initial_hidden(hidden)
    prev_dones = None
    for _ in range(n_steps):
        done_t = torch.as_tensor(prev_dones) if prev_dones is not None else None
        a, lp, v, hidden = net.act_recurrent(
            obs_to_tensors(obs, device), masks_to_tensors(masks, device),
            hidden, done_t)
        a_np = a.cpu().numpy()
        obs2, r, d, masks2, _ = envs.step(a_np)
        buf.add(obs, masks, a_np, lp.cpu().numpy(), v.cpu().numpy(), r, d)
        prev_dones = d
        obs, masks = obs2, masks2
    buf.compute_returns(np.zeros(n_envs, np.float32), 0.99, 0.95)
    return buf


def test_buffer_records_the_rollout_start_hidden(world, net):
    buf = _rollout(world, net)
    assert buf.initial_hidden.shape == (1, buf.n_envs, net.config.hidden_size)


def test_recurrent_update_runs_and_changes_weights(world, net):
    buf = _rollout(world, net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    stats = ppo_update_recurrent(
        net, opt, buf, PPOConfig(n_epochs=1, batch_size=256), 0.01,
        torch.device("cpu"), np.random.default_rng(0))
    assert all(np.isfinite(v) for v in stats.values())
    gru_moved = any(not torch.equal(p, before[n])
                    for n, p in net.named_parameters() if n.startswith("gru"))
    assert gru_moved, "GRU parameters did not receive gradient"


def test_evaluate_recurrent_reproduces_collection_logprobs(world):
    """Replaying an unchanged policy over the stored rollout must recover the
    log-probs recorded during collection — the invariant that makes the PPO
    ratio meaningful. A mishandled reset or off-by-one in the done gating
    shows up here immediately."""
    cards, _, _ = world
    net = make_network(len(cards), RECURRENT)
    net.eval()
    buf = _rollout(world, net)
    device = torch.device("cpu")
    with torch.no_grad():
        lp, _, _ = net.evaluate_actions_recurrent(
            {k: torch.as_tensor(v, device=device) for k, v in buf.obs.items()},
            {k: torch.as_tensor(v, device=device) for k, v in buf.masks.items()},
            torch.as_tensor(buf.actions, device=device),
            torch.as_tensor(buf.initial_hidden, device=device),
            torch.as_tensor(buf.dones, device=device),
        )
    assert torch.allclose(lp, torch.as_tensor(buf.log_probs), atol=1e-4)


def test_bptt_chunking_does_not_change_the_forward_pass(world):
    """Chunking only detaches gradients; the values must be identical."""
    cards, _, _ = world
    net = make_network(len(cards), RECURRENT)
    net.eval()
    buf = _rollout(world, net, n_steps=8)
    device = torch.device("cpu")
    args = (
        {k: torch.as_tensor(v, device=device) for k, v in buf.obs.items()},
        {k: torch.as_tensor(v, device=device) for k, v in buf.masks.items()},
        torch.as_tensor(buf.actions, device=device),
        torch.as_tensor(buf.initial_hidden, device=device),
        torch.as_tensor(buf.dones, device=device),
    )
    with torch.no_grad():
        whole = net.evaluate_actions_recurrent(*args, bptt_chunk=0)[0]
        chunked = net.evaluate_actions_recurrent(*args, bptt_chunk=3)[0]
    assert torch.allclose(whole, chunked)


# ----------------------------------------------------------------- policy bot


def test_policy_bot_carries_memory_across_decisions(world, net):
    """A recurrent checkpoint evaluated statelessly would silently understate
    every benchmark number."""
    cards, arena, deck = world
    bot = PolicyBot(net, list(cards.keys()), deterministic=True)
    env = _env(world)
    env.reset(seed=4)
    bot.decide(env.engine, Side.TOP)
    key = (id(env.engine), int(Side.TOP))
    assert key in bot._hidden
    first = bot._hidden[key].clone()
    for _ in range(5):
        env.engine.tick()
    bot.decide(env.engine, Side.TOP)
    assert not torch.equal(first, bot._hidden[key]), "memory did not advance"


def test_recurrent_bot_opts_out_of_batched_opponent_path(world, net):
    """`policy_actions_batched` has no hidden state to thread, so a recurrent
    bot must decline batching rather than be played memoryless."""
    cards, _, _ = world
    rec = PolicyBot(net, list(cards.keys()))
    plain = PolicyBot(make_network(len(cards), {"tier": obs_layout.TIER_RESTRICTED}),
                      list(cards.keys()))
    assert rec.batch_key() is None
    assert plain.batch_key() is not None


def test_vec_env_defers_recurrent_opponents(world, net):
    cards, _, _ = world
    from src.simulator.env import UNSET
    bot = PolicyBot(net, list(cards.keys()), deterministic=True)
    envs = SyncVecEnv([lambda: _env(world, opponent=bot) for _ in range(3)])
    envs.reset(seed=5)
    assert all(r is UNSET for r in envs._batched_opponent_actions())
    obs, r, d, m, infos = envs.step(np.zeros((3, 2), dtype=np.int64))
    assert len(infos) == 3  # and the step still completes correctly
