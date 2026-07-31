"""The live bridge: perception -> shadow engine -> policy -> tap.

Tested against simulator-built states rather than live frames, which is the
point of the shadow-engine design — if the policy sees the same observation
it would see in-sim, the bridge is correct regardless of where the pixels
came from.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agent import masking, obs_layout
from src.agent.network import make_network
from src.live.bridge import (
    LiveObservation,
    PerceivedUnit,
    PolicyDriver,
    ShadowEngine,
    check_tier,
)
from src.simulator.constants import HAND_SIZE, Side

DECK = ["knight", "archers", "goblins", "giant",
        "musketeer", "minions", "fireball", "cannon"]
SMALL = {"tier": "human", "conv_channels": (4,), "cnn_out": 16, "fusion_mlp": 32}


@pytest.fixture()
def shadow(cards, arena):
    return ShadowEngine(cards, arena, DECK)


def _obs(**kwargs):
    base = dict(hand=DECK[:4], next_card=DECK[4], own_elixir=10.0)
    base.update(kwargs)
    return LiveObservation(**base)


# ------------------------------------------------------------ shadow engine


def test_hand_and_elixir_come_from_perception(shadow):
    engine = shadow.build(_obs(hand=["giant", "fireball", "knight", "minions"],
                               next_card="cannon", own_elixir=6.5))
    me = engine.players[Side.BOTTOM]
    assert [c.name for c in me.hand] == ["giant", "fireball", "knight", "minions"]
    assert me.next_card.name == "cannon"
    assert me.elixir == pytest.approx(6.5)


def test_elixir_is_clamped_to_the_cap(shadow, arena):
    assert shadow.build(_obs(own_elixir=99.0)).players[Side.BOTTOM].elixir == arena.elixir_max
    assert shadow.build(_obs(own_elixir=-3.0)).players[Side.BOTTOM].elixir == 0.0


def test_detected_units_land_on_the_right_side_and_tile(shadow):
    engine = shadow.build(_obs(units=[
        PerceivedUnit(card="hog_rider", tile_x=4.0, tile_y=22.0, hostile=True),
        PerceivedUnit(card="musketeer", tile_x=9.0, tile_y=8.0, hostile=False),
    ]))
    hostile = [u for u in engine.units if u.side == Side.TOP]
    friendly = [u for u in engine.units if u.side == Side.BOTTOM]
    assert [u.stats.name for u in hostile] == ["hog_rider"]
    assert (hostile[0].x, hostile[0].y) == (4.0, 22.0)
    assert [u.stats.name for u in friendly] == ["musketeer"]


def test_perceived_units_are_immediately_active(shadow):
    """A detection is by definition already on the arena, so it must not sit
    in a deploy lock the policy would read as 'not a threat yet'."""
    engine = shadow.build(_obs(units=[
        PerceivedUnit(card="knight", tile_x=9.0, tile_y=20.0, hostile=True)]))
    assert not engine._is_inactive(engine.units[0])


def test_hp_fraction_becomes_unit_hp(shadow, cards):
    engine = shadow.build(_obs(units=[
        PerceivedUnit(card="giant", tile_x=9.0, tile_y=20.0, hostile=True,
                      hp_fraction=0.5, hp_confident=True)]))
    assert engine.units[0].hp == pytest.approx(cards["giant"].hp * 0.5)


def test_unidentified_detections_still_appear(shadow):
    """An abstaining classifier must not make a unit vanish — position and
    team alone recover most of the observation."""
    engine = shadow.build(_obs(units=[
        PerceivedUnit(card="", tile_x=9.0, tile_y=20.0, hostile=True)]))
    assert len(engine.units) == 1
    assert engine.units[0].side == Side.TOP


def test_fallen_towers_are_reflected_and_activate_the_king(shadow):
    engine = shadow.build(_obs(enemy_left_alive=False))
    dead = [t for t in engine.towers if t.side == Side.TOP and t.kind == "princess_left"]
    assert dead[0].hp == 0.0
    assert engine._king_of(Side.TOP).activated
    assert engine.pocket_unlocked(Side.BOTTOM, "left")


def test_a_fallen_tower_opens_the_placement_mask(shadow, cards):
    """The masks must react to observed tower state, or the policy can never
    use the pocket it just earned."""
    closed = masking.build_action_masks(shadow.build(_obs()), Side.BOTTOM)
    opened = masking.build_action_masks(
        shadow.build(_obs(enemy_left_alive=False)), Side.BOTTOM)
    assert opened["place"].sum() > closed["place"].sum()


def test_rejects_a_deck_with_unknown_cards(cards, arena):
    with pytest.raises(ValueError, match="not in the card table"):
        ShadowEngine(cards, arena, ["knight", "not_a_card", "goblins", "giant", "minions"])


def test_rejects_a_deck_too_small_to_cycle(cards, arena):
    with pytest.raises(ValueError, match="at least 5"):
        ShadowEngine(cards, arena, ["knight", "archers"])


def test_each_frame_rebuilds_rather_than_accumulating(shadow):
    """No reliable cross-frame identity, so carrying units forward would
    accumulate ghosts."""
    first = shadow.build(_obs(units=[
        PerceivedUnit(card="knight", tile_x=9.0, tile_y=20.0, hostile=True)]))
    second = shadow.build(_obs(units=[]))
    assert len(first.units) == 1
    assert second.units == []


# -------------------------------------------------- observation equivalence


def test_shadow_observation_matches_a_real_engine(cards, arena):
    """The load-bearing claim: a policy fed from perception sees the same
    thing it would see in the simulator for an equivalent board."""
    from tests.conftest import spawn_unit

    shadow = ShadowEngine(cards, arena, DECK)
    live = shadow.build(_obs(hand=DECK[:4], next_card=DECK[4], own_elixir=7.0,
                             units=[PerceivedUnit(card="hog_rider", tile_x=4.0,
                                                  tile_y=22.0, hostile=True)]))

    reference = shadow.build(_obs(hand=DECK[:4], next_card=DECK[4], own_elixir=7.0))
    spawn_unit(reference, cards["hog_rider"], Side.TOP, 4.0, 22.0)

    ids = {n: i for i, n in enumerate(cards)}
    a = obs_layout.encode_obs(live, Side.BOTTOM, ids, obs_layout.TIER_HUMAN)
    b = obs_layout.encode_obs(reference, Side.BOTTOM, ids, obs_layout.TIER_HUMAN)
    for key in ("spatial", "cards", "vector"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6)


def test_human_tier_observation_hides_opponent_elixir(shadow):
    """Even though the shadow engine has a slot for it, the human tier must
    not surface it."""
    lo = shadow.build(_obs(opponent_elixir=1.0))
    hi = shadow.build(_obs(opponent_elixir=9.0))
    ids = {"knight": 0}
    a = obs_layout.encode_obs(lo, Side.BOTTOM, ids, obs_layout.TIER_HUMAN)
    b = obs_layout.encode_obs(hi, Side.BOTTOM, ids, obs_layout.TIER_HUMAN)
    np.testing.assert_array_equal(a["vector"], b["vector"])


# ---------------------------------------------------------------- the driver


def test_full_tier_checkpoints_are_refused():
    with pytest.raises(ValueError, match="simulator-only"):
        check_tier(make_network(8, dict(SMALL, tier="full")))


def test_human_and_restricted_tiers_are_accepted():
    check_tier(make_network(8, dict(SMALL, tier="human")))
    check_tier(make_network(8, dict(SMALL, tier="restricted")))


def test_driver_returns_a_legal_slot_and_tile(cards, arena):
    torch.manual_seed(0)
    net = make_network(len(cards), SMALL)
    driver = PolicyDriver(net, list(cards), cards, arena, DECK)
    action = driver.decide(_obs(own_elixir=10.0))
    if action is not None:
        assert 0 <= action.slot < HAND_SIZE
        assert action.card == DECK[action.slot]
        engine = driver.last_engine
        assert engine.legal_deploy(Side.BOTTOM, cards[action.card], *action.tile)


def test_driver_cannot_play_what_it_cannot_afford(cards, arena):
    """Affordability masking is why reading the elixir bar matters."""
    torch.manual_seed(1)
    net = make_network(len(cards), SMALL)
    driver = PolicyDriver(net, list(cards), cards, arena, DECK)
    for _ in range(15):
        assert driver.decide(_obs(own_elixir=0.0)) is None


def test_driver_respects_the_hand_it_was_given(cards, arena):
    torch.manual_seed(2)
    net = make_network(len(cards), SMALL)
    driver = PolicyDriver(net, list(cards), cards, arena, DECK)
    hand = ["giant", "fireball", "knight", "minions"]
    for _ in range(10):
        action = driver.decide(_obs(hand=hand, own_elixir=10.0))
        if action is not None:
            assert action.card == hand[action.slot]


def test_driver_is_deterministic_by_default(cards, arena):
    torch.manual_seed(3)
    net = make_network(len(cards), SMALL)
    driver = PolicyDriver(net, list(cards), cards, arena, DECK)
    first = driver.decide(_obs(own_elixir=8.0))
    for _ in range(5):
        assert driver.decide(_obs(own_elixir=8.0)) == first


# ------------------------------------------------- runner integration


class _FakeDevice:
    def __init__(self, image):
        self.image = image
        self.taps = []

    def screenshot(self):
        return self.image

    def tap(self, x, y):
        self.taps.append((x, y))


def _policy_config(tmp_path, arena, **overrides):
    import yaml

    from src.live.config import load_live_config
    from src.live.homography import ANCHOR_NAMES, anchor_tiles
    from tests.live_frames import perspective_camera

    project = perspective_camera(arena, (556, 1028))
    tiles = anchor_tiles(arena)
    body = {
        "transport": "desktop",
        "reference_size": [556, 1028],
        "dynamic_target": [278, 550],
        "match_indicator": [160, 995, 350, 25],
        "card_slots": [[167, 917], [278, 917], [386, 917], [492, 917]],
        "card_ready_regions": [[140, 851, 55, 57], [250, 851, 55, 57],
                               [358, 851, 55, 57], [464, 851, 55, 57]],
        "decision_mode": "policy",
        "elixir_bar": [100, 960, 350, 20],
        "checkpoint": "dummy.pt",
        "deck": DECK,
        "homography_anchors": {n: [round(v) for v in project(*tiles[n])]
                               for n in ANCHOR_NAMES},
    }
    body.update(overrides)
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump({"live": body}))
    return load_live_config(path)


def test_policy_mode_requires_its_calibration(tmp_path, arena):
    for missing, pattern in (("homography_anchors", "homography_anchors"),
                             ("elixir_bar", "elixir_bar"),
                             ("checkpoint", "checkpoint")):
        with pytest.raises(ValueError, match=pattern):
            _policy_config(tmp_path, arena, **{missing: None})


def test_policy_mode_rejects_a_deck_that_is_not_eight_cards(tmp_path, arena):
    with pytest.raises(ValueError, match="exactly the eight cards"):
        _policy_config(tmp_path, arena, deck=DECK[:6])


def test_runner_falls_back_when_perception_fails(tmp_path, arena, cards):
    """The reason the heuristic is kept: a fabricated observation is worse
    than a dumb-but-safe decision."""
    from PIL import Image

    from src.live.runner import LiveMatchRunner

    config = _policy_config(tmp_path, arena)
    torch.manual_seed(0)
    net = make_network(len(cards), SMALL)
    driver = PolicyDriver(net, list(cards), cards, arena, DECK)

    # Bright and featureless: neither magenta fill nor dark trough, so the
    # elixir bar cannot be located at all.
    blank = Image.new("RGB", (556, 1028), (240, 240, 240))
    messages = []
    runner = LiveMatchRunner(config, _FakeDevice(blank), armed=False,
                             log=messages.append, driver=driver)
    runner._was_in_match = True
    assert runner._step_policy(blank) is False
    assert any("falling back" in m for m in messages)


def test_runner_taps_slot_then_tile_when_armed(tmp_path, arena, cards):
    from PIL import Image

    from src.live.runner import LiveMatchRunner

    config = _policy_config(tmp_path, arena)
    torch.manual_seed(0)
    net = make_network(len(cards), SMALL)
    driver = PolicyDriver(net, list(cards), cards, arena, DECK)

    # A frame with a full elixir bar so the policy can afford something.
    frame = np.zeros((1028, 556, 3), np.uint8)
    frame[:, :] = (96, 128, 84)
    frame[960:980, 100:450] = (220, 40, 200)   # magenta elixir bar, full
    image = Image.fromarray(frame)

    device = _FakeDevice(image)
    runner = LiveMatchRunner(config, device, armed=True, log=lambda _: None,
                             driver=driver)
    runner._was_in_match = True
    handled = runner._step_policy(image)
    assert handled is True
    if device.taps:
        assert len(device.taps) == 2, "one tap to pick the card, one to place it"


def test_dry_run_does_not_advance_the_hand_cycle(tmp_path, arena, cards):
    """Advancing the deterministic cycle on an unarmed run would leave the
    simulated hand permanently out of step with the real game."""
    from PIL import Image

    from src.live.runner import LiveMatchRunner

    config = _policy_config(tmp_path, arena)
    torch.manual_seed(0)
    net = make_network(len(cards), SMALL)
    driver = PolicyDriver(net, list(cards), cards, arena, DECK)

    frame = np.zeros((1028, 556, 3), np.uint8)
    frame[:, :] = (96, 128, 84)
    frame[960:980, 100:450] = (220, 40, 200)
    image = Image.fromarray(frame)

    runner = LiveMatchRunner(config, _FakeDevice(image), armed=False,
                             log=lambda _: None, driver=driver)
    runner._was_in_match = True
    before = list(runner.hand.hand)
    runner._step_policy(image)
    assert runner.hand.hand == before
