"""Three-line integration surface for code that is busy doing something else.

`src.agent.train` and `src.live` should not have to know about graphs,
probes, or SSE. They call `attach_to_network` once, then `emit_*` on
whatever cadence suits them, and every function here is a no-op when no
viewer is connected — so the viz import costs nothing at runtime and can
stay unconditional.
"""
from __future__ import annotations

import itertools

from src.viz import telemetry
from src.viz.graph import build_graph
from src.viz.probe import NetworkProbe

# Streaming an activation frame on every single env step would spend more
# time serialising than training. Once every N steps still looks continuous
# at a 512-step rollout.
ACT_EVERY_STEPS = 8


def attach_to_network(net, *, label: str = "") -> NetworkProbe | None:
    """Publish the architecture and start reading it. None if nobody is watching."""
    if not telemetry.is_enabled():
        return None
    graph = build_graph(net)
    telemetry.emit("graph", graph)
    if label:
        telemetry.log(f"[viz] attached to {label} — "
                      f"{graph['meta']['params']:,} params, "
                      f"{graph['meta']['tier']} tier", "good")
    return NetworkProbe(net, graph).attach()


def detach(probe: NetworkProbe | None) -> None:
    if probe is not None:
        probe.detach()


def emit_act(probe: NetworkProbe | None, step: int, **extra) -> None:
    """One activation frame from the most recent forward pass."""
    if probe is None:
        return
    frame = probe.activation_frame()
    if frame is not None:
        telemetry.emit("act", {"nodes": frame, "step": step, **extra})


def emit_learn(probe: NetworkProbe | None, step: int, **extra) -> None:
    """One weight-movement frame. Cheap enough to call per PPO update."""
    if probe is None:
        return
    telemetry.emit("learn", {**probe.learning_frame(), "step": step, **extra})


def attach_live_driver(driver, *, label: str = "live policy") -> NetworkProbe | None:
    """Animate the graph from a live-play `PolicyDriver`'s forward passes.

    Wraps `decide` rather than touching `LiveMatchRunner`: the runner calls
    the driver exactly once per screen capture, which is precisely the
    cadence at which a new forward pass exists to show.
    """
    probe = attach_to_network(driver.net, label=label)
    if probe is None:
        return None

    inner = driver.decide
    counter = itertools.count()

    def decide(observation):
        action = inner(observation)
        emit_act(probe, next(counter))
        return action

    driver.decide = decide
    # Weights never move during live play, so `reveal` would stay flat and the
    # graph would render as a bare skeleton. This one frame carries
    # `maturity`, which reads structure out of the weights themselves.
    emit_learn(probe, 0, frozen=True)
    return probe


def emit_stats(scope: str, metrics: dict) -> None:
    if telemetry.is_enabled():
        telemetry.emit("stats", {"scope": scope, "metrics": metrics})


def start_server(port: int | None, host: str = "127.0.0.1", *,
                 mode_note: str = "attached to this process",
                 activity: str = "training") -> None:
    """Boot the viewer inside the current process, if a port was requested.

    `activity` is what this process is doing (`training` or `live`). The mode
    is always `"attached"` here, so without it the UI cannot tell the two
    apart — see `VizController.state`.
    """
    if not port:
        return
    from src.viz.server import VizController, serve
    from src.viz.sources import AttachedSource

    controller = VizController(
        {"attached": AttachedSource("viz", mode_note, activity=activity)})
    controller.set_mode("attached")
    serve(port=port, host=host, controller=controller, block=False)
