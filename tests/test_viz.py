"""Tests for the 3D network viewer (`src/viz`).

The load-bearing properties are: the graph reflects the *actual* module tree
across every architecture variant, the probe reads real signal without
touching training, and the telemetry bus never blocks a producer.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from src.agent.network import make_network
from src.agent.obs_layout import MAX_ENTITIES, UNIT_FEATURE_DIM
from src.viz import telemetry
from src.viz.graph import MAX_NODES_PER_LAYER, build_graph, describe_layers
from src.viz.probe import NetworkProbe

VARIANTS = [
    pytest.param({"tier": "human"}, id="human-cnn"),
    pytest.param({"tier": "full"}, id="full-cnn"),
    pytest.param({"tier": "restricted"}, id="restricted-no-spatial"),
    pytest.param({"tier": "human", "use_set_encoder": True}, id="set-encoder"),
    pytest.param({"tier": "human", "use_recurrence": True}, id="recurrent"),
    pytest.param({"tier": "human", "critic_tier": "full"}, id="asymmetric-critic"),
]


@pytest.fixture(autouse=True)
def _clean_bus():
    telemetry.reset()
    telemetry.set_enabled(False)
    yield
    telemetry.reset()
    telemetry.set_enabled(False)


def make_obs(net, batch: int = 2) -> tuple[dict, dict]:
    cfg = net.config
    obs = {
        "spatial": torch.randn(batch, 10, 16, 9),
        "cards": torch.randint(0, cfg.n_cards, (batch, 5)),
        "vector": torch.randn(batch, cfg.scalar_dim),
        "units": torch.rand(batch, MAX_ENTITIES, UNIT_FEATURE_DIM),
    }
    if cfg.asymmetric:
        obs |= {
            "critic_spatial": torch.randn(batch, 10, 16, 9),
            "critic_cards": torch.randint(0, cfg.n_cards, (batch, 5)),
            "critic_vector": torch.randn(batch, cfg.critic_scalar_dim),
            "critic_units": torch.rand(batch, MAX_ENTITIES, UNIT_FEATURE_DIM),
        }
    masks = {"card": torch.ones(batch, 5, dtype=torch.bool),
             "place": torch.ones(batch, 5, 144, dtype=torch.bool)}
    return obs, masks


# ------------------------------------------------------------------- graph

@pytest.mark.parametrize("cfg", VARIANTS)
def test_graph_builds_for_every_architecture(cfg):
    graph = build_graph(make_network(60, cfg))
    assert graph["nodes"], "no nodes"
    assert graph["edges"], "no edges"
    ids = {n["id"] for n in graph["nodes"]}
    assert ids == set(range(len(graph["nodes"]))), "node ids must be dense"
    for src, dst, weight in graph["edges"]:
        assert src in ids and dst in ids, "edge references a missing node"
        assert 0.0 <= weight <= 1.0


def test_graph_tracks_config_rather_than_a_hardcoded_diagram():
    """The picture must change when the network changes, or it is a lie."""
    plain = {layer["key"] for layer in build_graph(make_network(60, {"tier": "human"}))["layers"]}
    recurrent = {layer["key"] for layer in
                 build_graph(make_network(60, {"tier": "human", "use_recurrence": True}))["layers"]}
    sets = {layer["key"] for layer in
            build_graph(make_network(60, {"tier": "human", "use_set_encoder": True}))["layers"]}
    critic = {layer["key"] for layer in
              build_graph(make_network(60, {"tier": "human", "critic_tier": "full"}))["layers"]}

    assert "gru" in recurrent and "gru" not in plain
    assert any(k.startswith("set.") for k in sets)
    assert not any(k.startswith("set.") for k in sets & plain)
    assert any(k.startswith("c.") for k in critic)
    assert not any(k.startswith("c.") for k in plain)


def test_restricted_tier_has_no_spatial_branch():
    """`restricted` zero-fills the grid, so drawing a live CNN would misrepresent it."""
    graph = build_graph(make_network(60, {"tier": "restricted"}))
    keys = {layer["key"] for layer in graph["layers"]}
    assert not any(k.startswith(("cnn", "obs.spatial", "set.")) for k in keys)
    assert graph["meta"]["arch"] == "scalar-only"


def test_layer_scalar_width_follows_the_tier():
    """Widths are not ordered by tier; the graph must not assume they are."""
    human = build_graph(make_network(60, {"tier": "human"}))
    restricted = build_graph(make_network(60, {"tier": "restricted"}))

    def scalar_size(graph):
        return next(l["size"] for l in graph["layers"] if l["key"] == "obs.vector")

    assert scalar_size(human) == 20
    assert scalar_size(restricted) == 18


def test_wide_layers_are_sampled_and_say_so():
    graph = build_graph(make_network(60, {"tier": "human"}))
    fusion = next(l for l in graph["layers"] if l["key"] == "fusion.0")
    assert fusion["size"] == 256
    assert fusion["shown"] == MAX_NODES_PER_LAYER
    assert len(fusion["nodes"]) == fusion["shown"]
    assert graph["meta"]["sampled"] is True


def test_sampled_nodes_are_real_unit_indices():
    graph = build_graph(make_network(60, {"tier": "human"}))
    for layer in graph["layers"]:
        units = [graph["nodes"][i]["unit"] for i in layer["nodes"]]
        assert units == sorted(set(units)), "sampled units must be distinct/ordered"
        assert all(0 <= u < layer["size"] for u in units)


def test_edges_flow_from_shallower_to_deeper_layers():
    graph = build_graph(make_network(60, {"tier": "human"}))
    depth = {layer["key"]: layer["depth"]
             for layer in graph["layers"]}
    for src, dst, _ in graph["edges"]:
        a = graph["nodes"][src]["layer"]
        b = graph["nodes"][dst]["layer"]
        assert depth[a] < depth[b], f"{a} -> {b} is not forward"


def test_critic_island_is_disconnected_from_the_actor():
    """No drawn edge may cross from the privileged critic into the policy path.

    This mirrors the real information boundary (`src/agent/network.py`): the
    critic may read opponent elixir, the actor may not, and a picture that
    implied otherwise would misrepresent the one property that makes the
    asymmetric setup legitimate.
    """
    graph = build_graph(make_network(60, {"tier": "human", "critic_tier": "full"}))
    nodes = graph["nodes"]
    for src, dst, _ in graph["edges"]:
        assert nodes[src]["critic"] == nodes[dst]["critic"], (
            "edge crosses the actor/critic boundary")


# ------------------------------------------------------------------- probe

@pytest.mark.parametrize("cfg", VARIANTS)
def test_probe_reports_activation_after_a_forward_pass(cfg):
    net = make_network(60, cfg)
    graph = build_graph(net)
    obs, masks = make_obs(net)

    with NetworkProbe(net, graph) as probe:
        assert probe.activation_frame() is None, "nothing has run yet"
        net.act(obs, masks)
        frame = probe.activation_frame()

    assert frame is not None
    assert len(frame) == len(graph["nodes"])
    assert all(0.0 <= v <= 1.0 for v in frame)
    assert max(frame) > 0.0, "a forward pass produced no activation anywhere"


def test_probe_activation_responds_to_the_observation():
    """Different inputs must give different pictures, or it is just decoration."""
    net = make_network(60, {"tier": "human"})
    graph = build_graph(net)
    masks = {"card": torch.ones(2, 5, dtype=torch.bool),
             "place": torch.ones(2, 5, 144, dtype=torch.bool)}

    with NetworkProbe(net, graph) as probe:
        obs, _ = make_obs(net)
        obs["spatial"] = torch.zeros_like(obs["spatial"])
        net.act(obs, masks)
        quiet = np.array(probe.activation_frame())

        obs["spatial"] = torch.randn_like(obs["spatial"]) * 5
        net.act(obs, masks)
        loud = np.array(probe.activation_frame())

    assert not np.allclose(quiet, loud)


def test_reveal_grows_only_when_weights_move():
    net = make_network(60, {"tier": "human"})
    graph = build_graph(net)
    obs, masks = make_obs(net)

    with NetworkProbe(net, graph) as probe:
        first = probe.learning_frame()
        assert max(first["reveal"]) == 0.0, "nothing has moved yet"

        # A second read with untouched weights must not invent movement.
        assert max(probe.learning_frame()["reveal"]) == 0.0

        optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
        log_probs, _, values = net.evaluate_actions(
            obs, masks, torch.zeros(2, 2, dtype=torch.long))
        (log_probs.sum() + values.sum()).backward()
        optimizer.step()

        moved = probe.learning_frame()

    assert max(moved["reveal"]) > 0.0, "training moved weights but reveal did not"
    assert all(0.0 <= v <= 1.0 for v in moved["reveal"])


def test_reveal_is_monotonic():
    """Reveal records that a unit *has been* reshaped, so it must never fall."""
    net = make_network(60, {"tier": "human"})
    graph = build_graph(net)
    obs, masks = make_obs(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)

    with NetworkProbe(net, graph) as probe:
        history = []
        for _ in range(3):
            log_probs, _, values = net.evaluate_actions(
                obs, masks, torch.zeros(2, 2, dtype=torch.long))
            (log_probs.sum() + values.sum()).backward()
            optimizer.step()
            optimizer.zero_grad()
            history.append(np.array(probe.learning_frame()["reveal"]))

    for earlier, later in zip(history, history[1:]):
        assert np.all(later >= earlier - 1e-6)


def test_probe_detach_removes_all_hooks():
    """A left-behind hook would silently tax every future forward pass."""
    net = make_network(60, {"tier": "human"})
    probe = NetworkProbe(net, build_graph(net)).attach()
    probe.detach()
    assert not any(m._forward_hooks for m in net.modules())


def test_probe_does_not_perturb_the_network():
    """Observation must not change behaviour — same seed, same action."""
    net = make_network(60, {"tier": "human"})
    obs, masks = make_obs(net)

    torch.manual_seed(7)
    before = net.act(obs, masks)[0].clone()

    with NetworkProbe(net, build_graph(net)):
        torch.manual_seed(7)
        during = net.act(obs, masks)[0].clone()

    assert torch.equal(before, during)


# ------------------------------------------------------------ live attach

def test_attach_live_driver_emits_a_frame_per_decision():
    """The live graph must animate off the same forward pass that taps."""
    from src.viz import attach

    net = make_network(60, {"tier": "human"})
    obs, masks = make_obs(net, batch=1)
    decisions = []

    class FakeDriver:
        def __init__(self):
            self.net = net

        def decide(self, observation):
            decisions.append(observation)
            net.act(obs, masks)
            return "tap"

    telemetry.set_enabled(True)
    sub_id, sub, _ = telemetry.subscribe()
    driver = FakeDriver()
    probe = attach.attach_live_driver(driver)
    try:
        assert driver.decide("frame-1") == "tap", "wrapper must not swallow the action"
        assert decisions == ["frame-1"], "wrapper must not swallow the observation"

        kinds = []
        while not sub.q.empty():
            kinds.append(sub.q.get_nowait()["t"])
        assert "graph" in kinds
        assert "learn" in kinds, "frozen weights still need one maturity frame"
        assert "act" in kinds, "no activation frame for the decision"
    finally:
        attach.detach(probe)
        telemetry.unsubscribe(sub_id)


def test_attach_live_driver_is_inert_without_a_viewer():
    from src.viz import attach

    net = make_network(60, {"tier": "human"})

    class FakeDriver:
        def __init__(self):
            self.net = net

        def decide(self, observation):
            return "tap"

    telemetry.set_enabled(False)
    driver = FakeDriver()
    assert attach.attach_live_driver(driver) is None
    # `driver.decide` builds a fresh bound method on every access, so identity
    # comparison proves nothing. The wrapper works by shadowing the class
    # attribute with an instance one, so its absence is the real check.
    assert "decide" not in driver.__dict__, "live play must not be wrapped for nobody"


# --------------------------------------------------------------- telemetry

def test_emit_is_inert_when_no_viewer_is_attached():
    telemetry.set_enabled(False)
    calls = []

    def payload():
        calls.append(1)
        return {"x": 1}

    telemetry.emit("act", payload)
    assert calls == [], "payload built despite no viewer — that is the hot path"


def test_slow_subscriber_never_blocks_the_producer():
    """Training must not stall because a browser tab fell behind."""
    telemetry.set_enabled(True)
    sub_id, sub, _ = telemetry.subscribe()
    try:
        for i in range(2000):        # far beyond the queue bound
            telemetry.emit("act", {"i": i})
        assert sub.q.full()
        assert sub.dropped > 0, "expected frames to be dropped, not buffered"
    finally:
        telemetry.unsubscribe(sub_id)


def test_late_viewer_receives_the_sticky_graph():
    telemetry.set_enabled(True)
    telemetry.emit("graph", {"nodes": [], "edges": []})
    telemetry.emit("log", {"line": "hello", "level": "info"})

    sub_id, _, backlog = telemetry.subscribe()
    try:
        kinds = [e["t"] for e in backlog]
        assert "graph" in kinds, "a tab opened mid-run would see an empty scene"
        assert any(e.get("line") == "hello" for e in backlog)
    finally:
        telemetry.unsubscribe(sub_id)


def test_tee_logger_prints_and_publishes():
    telemetry.set_enabled(True)
    printed = []
    sub_id, sub, _ = telemetry.subscribe()
    try:
        telemetry.TeeLogger(printed.append)("match detected")
        assert printed == ["match detected"]
        assert sub.q.get_nowait()["line"] == "match detected"
    finally:
        telemetry.unsubscribe(sub_id)


# ------------------------------------------------------------------ server

@pytest.fixture
def running_server():
    from src.viz.server import VizController, serve

    controller = VizController({"live": _FakeSource(), "training": _FakeSource()})
    httpd = serve(port=0, host="127.0.0.1", controller=controller, block=False)
    yield httpd, controller
    telemetry.close_all()
    controller.shutdown()
    httpd.shutdown()
    httpd.server_close()


class _FakeSource:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def _get(httpd, path: str) -> tuple[int, bytes]:
    url = f"http://127.0.0.1:{httpd.server_port}{path}"
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


def test_server_serves_the_page_and_assets(running_server):
    httpd, _ = running_server
    for path in ("/", "/app.js", "/gl.js", "/style.css"):
        status, body = _get(httpd, path)
        assert status == 200 and body, path


def test_server_refuses_paths_outside_static(running_server):
    """Loopback is not a reason to hand out checkpoints on request."""
    httpd, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(httpd, "/../../configs/cards.yaml")
    assert excinfo.value.code in (403, 404)


def test_mode_switch_starts_one_source_and_stops_the_other(running_server):
    httpd, controller = running_server
    _get(httpd, "/api/mode?m=live")
    assert controller.sources["live"].started == 1

    _get(httpd, "/api/mode?m=training")
    assert controller.sources["live"].stopped == 1
    assert controller.sources["training"].started == 1
    assert controller.mode == "training"


def test_unknown_mode_is_rejected(running_server):
    httpd, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(httpd, "/api/mode?m=nonsense")
    assert excinfo.value.code == 400


def test_event_stream_delivers_frames(running_server):
    httpd, _ = running_server
    url = f"http://127.0.0.1:{httpd.server_port}/events"
    received: list[dict] = []

    def reader():
        with urllib.request.urlopen(url, timeout=8) as response:
            for raw in response:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    received.append(json.loads(line[6:]))
                    if len(received) >= 2:
                        return

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for _ in range(50):
        telemetry.emit("act", {"nodes": [0.5]})
        telemetry.emit("log", {"line": "tick", "level": "info"})
        thread.join(timeout=0.1)
        if not thread.is_alive():
            break

    assert received, "no events reached the SSE client"
    assert all("t" in event for event in received)
