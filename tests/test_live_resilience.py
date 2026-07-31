"""Runner resilience and the vectorized letterbox-trim.

`run_forever` must survive the transient failures an unattended watcher
actually hits (window minimized, a blocked grab) without ending the session,
while still refusing to spin forever on a genuinely broken setup.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.live.device import WindowsDesktopDevice
from src.live.runner import LiveMatchRunner
from tests.test_live_runner import make_config


class _Device:
    """Screenshot source scripted to fail on chosen calls."""

    def __init__(self, failures: set[int], error=RuntimeError("client area is empty")):
        self.failures = failures
        self.error = error
        self.calls = 0
        self.taps: list[tuple[int, int]] = []

    def screenshot(self):
        self.calls += 1
        if self.calls in self.failures:
            raise self.error
        return Image.new("RGB", (100, 100), (0, 0, 0))

    def tap(self, x, y):
        self.taps.append((x, y))


def _runner(device, logs, **kwargs):
    config = make_config()
    object.__setattr__(config, "poll_seconds", 0.0)  # frozen dataclass; no real sleeping
    return LiveMatchRunner(config, device, log=logs.append, **kwargs)


def test_transient_error_does_not_end_the_session(monkeypatch):
    """A failure followed by recovery keeps polling rather than propagating."""
    device = _Device(failures={1, 2})
    logs: list[str] = []
    runner = _runner(device, logs)

    # Stop the infinite loop once we've seen enough polls to prove recovery.
    def fake_sleep(_seconds):
        if device.calls >= 5:
            raise KeyboardInterrupt

    monkeypatch.setattr("src.live.runner.time.sleep", fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        runner.run_forever()

    assert device.calls >= 5  # kept going past the two failures
    assert sum("Recoverable error" in m for m in logs) == 2


def test_persistent_failure_gives_up_instead_of_spinning(monkeypatch):
    device = _Device(failures=set(range(1, 100)))
    logs: list[str] = []
    runner = _runner(device, logs)
    monkeypatch.setattr("src.live.runner.time.sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="Giving up after 3 consecutive errors"):
        runner.run_forever(max_consecutive_errors=3)
    assert device.calls == 3


def test_error_counter_resets_after_a_success(monkeypatch):
    """Intermittent failures must not accumulate toward the give-up limit."""
    device = _Device(failures={1, 3, 5})  # alternating fail/succeed
    logs: list[str] = []
    runner = _runner(device, logs)

    def fake_sleep(_seconds):
        if device.calls >= 7:
            raise KeyboardInterrupt

    monkeypatch.setattr("src.live.runner.time.sleep", fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        runner.run_forever(max_consecutive_errors=2)  # would trip if not reset
    assert device.calls >= 7


def test_keyboard_interrupt_propagates_immediately(monkeypatch):
    device = _Device(failures=set(), error=KeyboardInterrupt())
    logs: list[str] = []
    runner = _runner(device, logs)
    monkeypatch.setattr("src.live.runner.time.sleep", lambda _s: None)

    def interrupt():
        raise KeyboardInterrupt

    runner.step = interrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run_forever()


# ---------------------------------------------------------------- viewport


def _letterboxed(width, height, bar):
    """Bright portrait content flanked by solid-black bars."""
    image = Image.new("RGB", (width, height), (0, 0, 0))
    for x in range(bar, width - bar):
        for y in range(height):
            image.putpixel((x, y), (200, 200, 200))
    return image


def test_content_viewport_trims_side_bars():
    image = _letterboxed(200, 400, bar=40)
    left, top, right, bottom = WindowsDesktopDevice._content_viewport(image)
    assert (top, bottom) == (0, 400)
    assert left == pytest.approx(40, abs=2)
    assert right == pytest.approx(160, abs=2)


def test_content_viewport_keeps_full_frame_when_there_are_no_bars():
    image = Image.new("RGB", (200, 400), (200, 200, 200))
    assert WindowsDesktopDevice._content_viewport(image) == (0, 0, 200, 400)


def test_content_viewport_ignores_an_all_dark_frame():
    """A transient black screen must not collapse the crop to nothing."""
    image = Image.new("RGB", (200, 400), (0, 0, 0))
    assert WindowsDesktopDevice._content_viewport(image) == (0, 0, 200, 400)


def test_content_viewport_rejects_an_implausibly_narrow_result():
    image = _letterboxed(200, 400, bar=95)  # only ~10px of content
    assert WindowsDesktopDevice._content_viewport(image) == (0, 0, 200, 400)
