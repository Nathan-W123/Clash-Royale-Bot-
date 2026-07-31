"""Synthetic annotated frames for the live-vision tests (#34).

There is no way to unit-test perception against a live match, and running
against live servers is out of scope for testing, so the fixtures are saved
frames. Real captures are the goal; these synthetic ones exist so the
pipeline has deterministic coverage from day one and so the *fixture format*
is pinned down before anyone hand-labels a screenshot.

Drop real annotated captures into ``tests/fixtures/live/`` as a
``<name>.png`` plus a ``<name>.json`` of the form::

    {"reference_size": [556, 1028],
     "homography_anchors": {"own_king": [278, 830], ...},
     "units": [{"team": "hostile", "tile": [9.0, 20.0], "hp_fraction": 0.5}, ...]}

`test_vision_blobs.py` picks them up automatically and holds them to the same
assertions as the synthetic frame.

Run this module to (re)generate the synthetic fixture on disk::

    python -m tests.live_frames
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "live"
FRAME_SIZE = (556, 1028)  # (width, height)

# Colours chosen to sit inside `vision.DEFAULT_TEAM_COLORS` hue windows.
HOSTILE_RGB = (222, 40, 38)
FRIENDLY_RGB = (44, 96, 226)
TROUGH_RGB = (26, 26, 30)          # unfilled bar remainder: dark, desaturated
ARENA_RGB = (96, 128, 84)          # muted grass; low saturation on purpose
BAR_WIDTH = 22
BAR_HEIGHT = 4


@dataclass(frozen=True)
class PlannedUnit:
    team: str
    tile: tuple[float, float]
    hp_fraction: float


def perspective_camera(arena, size=FRAME_SIZE):
    """Tile -> pixel with a genuine perspective term (far half narrower)."""
    w, h = size

    def project(tx, ty):
        depth = 1.0 + 0.45 * (ty / arena.height)
        px = w / 2 + (tx - arena.width / 2) / depth * (w / arena.width) * 0.9
        py = h - (ty / arena.height) * h * 0.82 / depth - 40.0
        return px, py

    return project


def anchor_pixels(arena, project) -> dict[str, list[int]]:
    from src.live.homography import anchor_tiles

    return {name: [round(v) for v in project(*tile)]
            for name, tile in anchor_tiles(arena).items()}


def _draw_rect(frame: np.ndarray, x: int, y: int, w: int, h: int, rgb) -> None:
    h_img, w_img = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = rgb


def render_frame(arena, units: list[PlannedUnit], size=FRAME_SIZE,
                 bar_to_feet_px: int = 14, with_hud_distractor: bool = True):
    """Render a frame plus the ground truth needed to score it."""
    project = perspective_camera(arena, size)
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    frame[:, :] = ARENA_RGB

    for unit in units:
        feet_x, feet_y = project(*unit.tile)
        bar_cx = feet_x
        bar_cy = feet_y - bar_to_feet_px
        filled = max(1, int(round(BAR_WIDTH * unit.hp_fraction)))
        left = int(round(bar_cx - BAR_WIDTH / 2))
        top = int(round(bar_cy - BAR_HEIGHT / 2))
        # Trough first, filled portion over it — same z-order the game uses,
        # and what makes the fill fraction recoverable.
        _draw_rect(frame, left, top, BAR_WIDTH, BAR_HEIGHT, TROUGH_RGB)
        rgb = HOSTILE_RGB if unit.team == "hostile" else FRIENDLY_RGB
        _draw_rect(frame, left, top, filled, BAR_HEIGHT, rgb)

    if with_hud_distractor:
        # A saturated red HUD chip below the arena. Anything that maps
        # off-board must be discarded, or the elixir bar becomes a permanent
        # phantom enemy parked on the agent's own side.
        _draw_rect(frame, 40, size[1] - 25, 60, 6, HOSTILE_RGB)

    return Image.fromarray(frame), {
        "reference_size": list(size),
        "homography_anchors": anchor_pixels(arena, project),
        "units": [{"team": u.team, "tile": list(u.tile), "hp_fraction": u.hp_fraction}
                  for u in units],
    }


DEFAULT_UNITS = [
    PlannedUnit("hostile", (4.0, 21.0), 1.0),
    PlannedUnit("hostile", (13.5, 24.0), 0.5),
    PlannedUnit("hostile", (9.0, 18.0), 0.25),
    PlannedUnit("friendly", (5.0, 9.0), 1.0),
    PlannedUnit("friendly", (14.0, 11.0), 0.75),
]


def write_default_fixture(arena, directory: Path = FIXTURE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image, meta = render_frame(arena, DEFAULT_UNITS)
    meta["synthetic"] = True
    image.save(directory / "synthetic_push.png")
    (directory / "synthetic_push.json").write_text(json.dumps(meta, indent=2))
    return directory / "synthetic_push.png"


if __name__ == "__main__":  # pragma: no cover - fixture regeneration
    from src.simulator.cards import load_arena

    print(write_default_fixture(load_arena()))
