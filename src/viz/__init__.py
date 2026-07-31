"""Live 3D visualisation of the policy network (training + live play).

The visualiser is strictly an observer. It reads the network the rest of the
project already builds, never constructs its own, and never feeds anything
back into training or live play. Everything is opt-in: with no viz server
attached, `telemetry.emit` is a couple of predictable branches and the probe
is never installed at all.

Modules
-------
`telemetry`  process-wide event bus (ring buffer + subscriber queues)
`graph`      PolicyNetwork -> 3D node/edge scene, derived from real modules
`probe`      torch forward hooks (activations) + weight-movement tracker
`server`     stdlib HTTP + Server-Sent Events, serves `static/`
`sources`    drivers that produce frames (simulator rollout, attach-to-file)
"""
from src.viz.telemetry import emit, is_enabled, set_enabled

__all__ = ["emit", "is_enabled", "set_enabled"]
