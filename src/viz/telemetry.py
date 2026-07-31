"""Process-wide event bus between whatever is running (training, live play)
and any attached viewers.

Design constraints that shaped this:

* **Off costs nothing.** Training loops call `emit` on a hot-ish path. When
  no viewer is attached `_ENABLED` is False and `emit` returns before its
  payload is built — which is why `emit` takes a callable for expensive
  payloads rather than the payload itself.
* **A slow viewer must never stall training.** Subscriber queues are bounded
  and *drop oldest* when full. A browser tab that falls behind loses frames;
  it does not apply backpressure to the PPO loop. Dropped frames are counted
  so the UI can be honest about it.
* **No dependencies.** `queue` + `threading` from the stdlib.
"""
from __future__ import annotations

import itertools
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

# Frames are pure state snapshots, so dropping a stale one loses nothing that
# the next frame will not immediately restate. Log lines are the exception —
# they are a sequence, not a snapshot — so they get a deeper allowance.
_QUEUE_MAX = 256

_lock = threading.Lock()
_subscribers: dict[int, "_Subscriber"] = {}
_ids = itertools.count(1)
_ENABLED = False

# Latest frame of each "sticky" kind, replayed to a viewer the moment it
# connects. Without this a browser opening mid-run stares at an empty scene
# until the next graph rebuild, which for training is minutes away.
_STICKY_KINDS = ("graph", "mode", "learn", "act", "stats")
_sticky: dict[str, dict] = {}
_recent_logs: list[dict] = []
_RECENT_LOG_MAX = 200


class _Subscriber:
    __slots__ = ("q", "dropped")

    def __init__(self) -> None:
        self.q: queue.Queue[dict] = queue.Queue(maxsize=_QUEUE_MAX)
        self.dropped = 0

    def put(self, event: dict) -> None:
        try:
            self.q.put_nowait(event)
        except queue.Full:
            # Drop the oldest, not the newest: the newest frame is the one
            # that reflects reality.
            try:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait(event)
            except (queue.Empty, queue.Full):  # pragma: no cover - race only
                self.dropped += 1


def is_enabled() -> bool:
    return _ENABLED


def set_enabled(value: bool) -> None:
    """Force the bus on/off regardless of subscribers.

    The server turns this on at startup so events produced *before* the first
    browser connects still populate the sticky cache.
    """
    global _ENABLED
    _ENABLED = value


def _refresh_enabled() -> None:
    global _ENABLED
    _ENABLED = bool(_subscribers)


def emit(kind: str, payload: dict | Callable[[], dict] | None = None) -> None:
    """Publish one event. `payload` may be a callable, evaluated only when the
    bus is live — that is what keeps the disabled path free."""
    if not _ENABLED:
        return
    data = payload() if callable(payload) else (payload or {})
    event = {"t": kind, "ts": time.time(), **data}
    with _lock:
        if kind in _STICKY_KINDS:
            _sticky[kind] = event
        elif kind == "log":
            _recent_logs.append(event)
            del _recent_logs[:-_RECENT_LOG_MAX]
        for sub in _subscribers.values():
            sub.put(event)


def log(line: str, level: str = "info") -> None:
    """Convenience for the terminal pane. Safe to pass as a `log=` callable."""
    emit("log", {"line": str(line), "level": level})


def subscribe() -> tuple[int, _Subscriber, list[dict]]:
    """Register a viewer. Returns (id, subscriber, replay backlog)."""
    with _lock:
        sub_id = next(_ids)
        sub = _Subscriber()
        _subscribers[sub_id] = sub
        backlog = [_sticky[k] for k in _STICKY_KINDS if k in _sticky]
        backlog += _recent_logs[-60:]
        _refresh_enabled()
    return sub_id, sub, backlog


def unsubscribe(sub_id: int) -> None:
    with _lock:
        _subscribers.pop(sub_id, None)
        _refresh_enabled()


def subscriber_count() -> int:
    with _lock:
        return len(_subscribers)


def reset() -> None:
    """Drop all state. Tests only."""
    with _lock:
        _subscribers.clear()
        _sticky.clear()
        _recent_logs.clear()
    _refresh_enabled()


def stream(sub: _Subscriber, backlog: list[dict],
           heartbeat: float = 15.0) -> Iterator[bytes]:
    """Yield SSE-framed bytes forever.

    The heartbeat comment is not decorative: without traffic, proxies and
    some browsers quietly close an idle event stream, and a training run
    between rollouts can legitimately say nothing for a while.
    """
    for event in backlog:
        yield _sse(event)
    while True:
        try:
            event = sub.q.get(timeout=heartbeat)
        except queue.Empty:
            yield b": keepalive\n\n"
            continue
        if event.get("t") == "__close__":
            return
        if sub.dropped:
            event = {**event, "dropped": sub.dropped}
            sub.dropped = 0
        yield _sse(event)


def _sse(event: dict) -> bytes:
    return b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"


def close_all() -> None:
    """Wake every stream so the server can shut down promptly."""
    with _lock:
        for sub in _subscribers.values():
            sub.put({"t": "__close__"})


class TeeLogger:
    """`log=` callable that both prints and publishes.

    `LiveMatchRunner` already takes a `log` callable, so the live terminal
    needs no changes to the runner beyond passing one of these.
    """

    def __init__(self, underlying: Callable[[str], Any] | None = print,
                 level: str = "info") -> None:
        self.underlying = underlying
        self.level = level

    def __call__(self, line: str) -> None:
        if self.underlying is not None:
            self.underlying(line)
        log(line, self.level)
