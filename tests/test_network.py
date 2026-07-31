import numpy as np
import torch

from src.agent.network import (
    N_CARD_CHOICES,
    N_CELLS,
    make_network,
    masks_to_tensors,
    obs_to_tensors,
)
from src.agent.obs_layout import RESTRICTED_SCALAR_DIM, SCALAR_DIM, SPATIAL_CHANNELS
from src.simulator.constants import HAND_SIZE, PLACE_COLS, PLACE_ROWS

N_CARDS = 16
DEVICE = torch.device("cpu")


def random_batch(rng, batch=8, scalar_dim=SCALAR_DIM):
    obs = {
        "spatial": rng.random((batch, SPATIAL_CHANNELS, PLACE_ROWS, PLACE_COLS)).astype(np.float32),
        "cards": rng.integers(0, N_CARDS, (batch, HAND_SIZE + 1)),
        "vector": rng.random((batch, scalar_dim)).astype(np.float32),
    }
    card_mask = np.zeros((batch, N_CARD_CHOICES), bool)
    card_mask[:, 0] = True
    place_mask = rng.random((batch, HAND_SIZE, N_CELLS)) < 0.4
    for b in range(batch):
        for slot in range(HAND_SIZE):
            if rng.random() < 0.6 and place_mask[b, slot].any():
                card_mask[b, 1 + slot] = True
    return obs, {"card": card_mask, "place": place_mask}


def test_act_respects_masks():
    rng = np.random.default_rng(0)
    net = make_network(N_CARDS)
    obs_np, masks_np = random_batch(rng, batch=16)
    obs = obs_to_tensors(obs_np, DEVICE)
    masks = masks_to_tensors(masks_np, DEVICE)
    for _ in range(20):
        actions, log_prob, value = net.act(obs, masks)
        for b in range(16):
            card, cell = int(actions[b, 0]), int(actions[b, 1])
            assert masks_np["card"][b, card]
            if card > 0:
                assert masks_np["place"][b, card - 1, cell]
        assert log_prob.shape == (16,)
        assert value.shape == (16,)
        assert torch.isfinite(log_prob).all()


def test_evaluate_matches_act_logprob():
    rng = np.random.default_rng(1)
    net = make_network(N_CARDS)
    obs_np, masks_np = random_batch(rng)
    obs = obs_to_tensors(obs_np, DEVICE)
    masks = masks_to_tensors(masks_np, DEVICE)
    actions, log_prob, _ = net.act(obs, masks)
    log_prob2, entropy, value = net.evaluate_actions(obs, masks, actions)
    torch.testing.assert_close(log_prob, log_prob2, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(entropy).all()
    assert (entropy >= 0).all()


def test_bc_loss_decreases_on_fixture():
    rng = np.random.default_rng(2)
    net = make_network(N_CARDS)
    obs_np, masks_np = random_batch(rng, batch=32)
    obs = obs_to_tensors(obs_np, DEVICE)
    masks = masks_to_tensors(masks_np, DEVICE)
    # Fabricate legal targets.
    cards, cells = [], []
    for b in range(32):
        legal = np.flatnonzero(masks_np["card"][b])
        c = int(legal[-1])
        cards.append(c)
        cells.append(int(np.flatnonzero(masks_np["place"][b, c - 1])[0]) if c > 0 else 0)
    card_t = torch.as_tensor(cards)
    cell_t = torch.as_tensor(cells)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        card_logits, place_logits = net.bc_logits(obs, masks, card_t)
        loss = torch.nn.functional.cross_entropy(card_logits, card_t)
        played = card_t > 0
        if played.any():
            loss = loss + torch.nn.functional.cross_entropy(
                place_logits[played], cell_t[played])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_restricted_network_has_no_cnn_params():
    net = make_network(N_CARDS, {"use_spatial": False})
    assert not hasattr(net, "cnn")
    assert not hasattr(net, "cnn_proj")
    assert net.config.scalar_dim == RESTRICTED_SCALAR_DIM


def test_restricted_network_ignores_spatial_content():
    rng = np.random.default_rng(3)
    net = make_network(N_CARDS, {"use_spatial": False})
    obs_np, masks_np = random_batch(rng, batch=8, scalar_dim=RESTRICTED_SCALAR_DIM)
    obs = obs_to_tensors(obs_np, DEVICE)
    masks = masks_to_tensors(masks_np, DEVICE)
    feat_a = net.trunk(obs)

    obs_np["spatial"] = rng.random(obs_np["spatial"].shape).astype(np.float32) * 100.0
    obs_b = obs_to_tensors(obs_np, DEVICE)
    feat_b = net.trunk(obs_b)

    torch.testing.assert_close(feat_a, feat_b)


def test_full_network_default_config_unchanged():
    net = make_network(N_CARDS)
    assert net.config.use_spatial is True
    assert net.config.scalar_dim == SCALAR_DIM
    assert hasattr(net, "cnn")
