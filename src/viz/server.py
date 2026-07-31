"""Stdlib HTTP server + Server-Sent Events stream for the 3D viewer.

Why SSE and not WebSockets: the data flow is essentially one-way (backend →
browser at 5-20 Hz), SSE needs no handshake or frame codec, browsers
reconnect automatically, and it is implementable on `http.server` without
adding a dependency to a project whose install is currently numpy/torch/yaml.
The handful of control messages that go the other way (mode switch, camera
reset) are ordinary GETs.

Binding: loopback only by default. This streams the internals of a running
process and, in live mode, a description of what is on the user's screen —
none of which should be reachable from the network. `--host` can override it
for a deliberate LAN setup, and the server says so loudly when it is not
bound to localhost.
"""
from __future__ import annotations

import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.viz import telemetry

STATIC_DIR = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
}


class VizHandler(BaseHTTPRequestHandler):
    server_version = "CRBotViz/1.0"
    controller: "VizController | None" = None

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook
        """Silence per-request logging; it would drown the training console."""

    # ------------------------------------------------------------ routing

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        route = urlparse(self.path)
        path = route.path
        if path == "/events":
            self._serve_events()
        elif path == "/api/state":
            self._serve_json(self.controller.state() if self.controller else {})
        elif path == "/api/mode":
            self._serve_mode(parse_qs(route.query))
        elif path in ("/", "/index.html"):
            self._serve_static("index.html")
        else:
            self._serve_static(path.lstrip("/"))

    # ----------------------------------------------------------- handlers

    def _serve_mode(self, query: dict) -> None:
        mode = (query.get("m") or [""])[0]
        if self.controller is None:
            self._serve_json({"ok": False, "error": "no controller"}, 503)
            return
        try:
            self.controller.set_mode(mode)
        except ValueError as error:
            self._serve_json({"ok": False, "error": str(error)}, 400)
            return
        self._serve_json({"ok": True, "mode": self.controller.mode})

    def _serve_events(self) -> None:
        sub_id, sub, backlog = telemetry.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            for chunk in telemetry.stream(sub, backlog):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass          # tab closed; entirely routine
        finally:
            telemetry.unsubscribe(sub_id)

    def _serve_static(self, name: str) -> None:
        # Resolve inside STATIC_DIR and verify containment: this server binds
        # to loopback but still must not serve `../../checkpoints/*` to a
        # stray request.
        target = (STATIC_DIR / name).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",
                         _CONTENT_TYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class VizController:
    """Owns the currently running source and lets the UI switch between them.

    Sources are objects with `start()`, `stop()` and a `name`. Only one runs
    at a time — two sources emitting into one bus would interleave two
    unrelated networks' activations into the same scene.
    """

    def __init__(self, sources: dict) -> None:
        self.sources = sources
        self.mode = "idle"
        self._active = None
        self._lock = threading.Lock()

    def set_mode(self, mode: str) -> None:
        if mode not in self.sources and mode != "idle":
            raise ValueError(
                f"unknown mode {mode!r}; have {sorted(self.sources) + ['idle']}")
        with self._lock:
            if mode == self.mode:
                return
            if self._active is not None:
                self._active.stop()
                self._active = None
            self.mode = mode
            if mode != "idle":
                self._active = self.sources[mode]
                self._active.start()
        telemetry.emit("mode", {"mode": self.mode,
                                "available": sorted(self.sources)})

    def state(self) -> dict:
        return {"mode": self.mode, "available": sorted(self.sources),
                "viewers": telemetry.subscriber_count()}

    def shutdown(self) -> None:
        with self._lock:
            if self._active is not None:
                self._active.stop()
                self._active = None
            self.mode = "idle"


def serve(port: int = 8770, host: str = "127.0.0.1",
          controller: VizController | None = None,
          block: bool = True) -> ThreadingHTTPServer:
    """Start the viewer. Returns the server (already serving) when block=False."""
    telemetry.set_enabled(True)
    handler = partial(VizHandler)
    handler.controller = controller          # type: ignore[attr-defined]
    VizHandler.controller = controller

    httpd = ThreadingHTTPServer((host, port), VizHandler)
    httpd.daemon_threads = True
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{httpd.server_port}"
    print(f"[viz] serving {url}")
    if host not in ("127.0.0.1", "localhost"):
        print(f"[viz] WARNING: bound to {host}, not loopback — this exposes "
              "network internals and live-screen state to your network.")

    if not block:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True,
                                  name="viz-server")
        thread.start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] stopping")
    finally:
        telemetry.close_all()
        if controller is not None:
            controller.shutdown()
        httpd.shutdown()
        httpd.server_close()
    return httpd


def serve_in_background(port: int = 8770, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Attach a viewer to a process that is already doing something else.

    This is how `src.agent.train --viz-port` and `src.live --viz-port` work:
    the real workload stays on the main thread and the server rides along.
    """
    return serve(port=port, host=host, controller=None, block=False)
