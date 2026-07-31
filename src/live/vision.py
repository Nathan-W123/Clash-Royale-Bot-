"""Screen reading for live play: calibrated UI checks plus arena perception.

Two layers live here:

**Calibrated region checks** (`mean_luma`, `mean_saturation`) — the original
cheap tests for "are we in a match" and "does this card slot look ready".

**Team-tinted blob detection** (#34) — the first real arena perception cut:
"there is a hostile unit at tile (x, y)", without card identity. Reading the
rendered arena is in scope precisely because a human reads it by looking at
the display; see CLAUDE.md, "On-Screen Visual Perception".

Why health bars rather than the unit sprites themselves: CR tints each
unit's health bar by team, and those bars are solid, high-saturation,
axis-aligned rectangles against a busy but desaturated arena. Sprites are
animated, overlapping, and share palettes across cards. Segmenting bars is
both far more robust and enough on its own — the observation's spatial
channels are HP density and presence, so position plus team recovers most of
the grid. Card identity (#35) is a refinement layered on top, not a
prerequisite.

**Known fidelity loss, stated plainly.** HP is read from the health-bar fill
fraction, which needs the bar's *unfilled* remainder to be visible. When the
remainder can't be found the detection is reported at full HP with
`hp_confident=False`. That biases perceived enemy HP upward, which makes the
policy value trades slightly wrong — it will read a nearly-dead unit as
healthy and over-commit to finishing it. Downstream consumers should prefer
`hp_confident` detections when weighting anything HP-sensitive.

Nothing here reads game memory or network traffic; it only looks at pixels
that are already on screen.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.live.config import Rect

# --------------------------------------------------------- calibrated checks


def crop_array(image: Image.Image, rect: Rect) -> np.ndarray:
    array = np.asarray(image)
    if rect.x < 0 or rect.y < 0 or rect.x + rect.width > image.width or rect.y + rect.height > image.height:
        raise ValueError(f"calibration rectangle {rect} is outside screenshot {image.size}")
    return array[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]


def mean_saturation(image: Image.Image, rect: Rect) -> float:
    pixels = crop_array(image, rect).astype(np.float32) / 255.0
    high = pixels.max(axis=2)
    low = pixels.min(axis=2)
    return float(np.mean(np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)))


def mean_luma(image: Image.Image, rect: Rect) -> float:
    pixels = crop_array(image, rect).astype(np.float32)
    return float(np.mean(0.2126 * pixels[:, :, 0] + 0.7152 * pixels[:, :, 1] + 0.0722 * pixels[:, :, 2]))


# The elixir bar's filled portion is a solid magenta; the empty remainder is
# a dark trough. Same segmentation idea as health bars, one dimension.
ELIXIR_HUE_CENTER = 310.0
ELIXIR_HUE_TOLERANCE = 35.0
ELIXIR_MIN_SATURATION = 0.35
ELIXIR_MAX = 10.0


def read_elixir(
    image: Image.Image | np.ndarray,
    rect: Rect,
    max_elixir: float = ELIXIR_MAX,
    hue_center: float = ELIXIR_HUE_CENTER,
    hue_tolerance: float = ELIXIR_HUE_TOLERANCE,
    min_saturation: float = ELIXIR_MIN_SATURATION,
) -> float | None:
    """Own elixir, read off the bar. None when the bar can't be found.

    Returns a *continuous* value rather than snapping to whole pips: elixir
    regenerates continuously and the affordability mask only cares whether
    the count has crossed a card's cost, so rounding to pips would make a
    card look unaffordable for up to a second after it actually became
    playable.

    None is a meaningful answer, not an error — it means the HUD is not
    visible (between matches, or the window is occluded), and the caller
    should hold rather than act on a fabricated zero.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] >= 3:
        crop = array[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]
    else:
        return None
    if crop.size == 0:
        return None

    hue, sat, val = rgb_to_hsv(crop)
    delta = np.abs(((hue - hue_center) + 180.0) % 360.0 - 180.0)
    filled = (delta <= hue_tolerance) & (sat >= min_saturation)
    # The unfilled remainder is the bar's dark trough. A region that is
    # neither magenta nor trough is not an elixir bar at all — the HUD is
    # occluded, or the window is showing something else — and that has to be
    # distinguishable from a genuinely empty bar, which *is* mostly trough.
    trough = (sat < min_saturation) & (val < 0.55)
    if (filled | trough).mean() < 0.3:
        return None
    if not filled.any():
        return 0.0

    # Fraction of *columns* that are filled, not of pixels: the bar is one
    # horizontal gauge, and a column-wise read is unaffected by the pip
    # dividers and rounded end-caps that break up a pixel count.
    column_filled = filled.mean(axis=0) > 0.5
    return float(column_filled.sum()) / column_filled.size * max_elixir


# ------------------------------------------------------------------- colour

TEAM_HOSTILE = "hostile"
TEAM_FRIENDLY = "friendly"


def rgb_to_hsv(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized HSV for an ``(H, W, 3)`` uint8 image.

    Hue in degrees [0, 360), saturation and value in [0, 1]. Hand-rolled
    rather than pulled from a dependency: this is the only colour conversion
    the project needs and it keeps `src/live` importable without OpenCV.
    """
    rgb = array[:, :, :3].astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    high = rgb.max(axis=2)
    low = rgb.min(axis=2)
    span = high - low

    hue = np.zeros_like(high)
    nz = span > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        r_max = nz & (high == r)
        g_max = nz & (high == g) & ~r_max
        b_max = nz & ~r_max & ~g_max
        hue[r_max] = ((g - b)[r_max] / span[r_max]) % 6.0
        hue[g_max] = ((b - r)[g_max] / span[g_max]) + 2.0
        hue[b_max] = ((r - g)[b_max] / span[b_max]) + 4.0
    hue *= 60.0

    sat = np.zeros_like(high)
    np.divide(span, high, out=sat, where=high > 1e-6)
    return hue, sat, high


@dataclass(frozen=True)
class TeamColor:
    """Hue window identifying one team's health-bar tint."""

    team: str
    hue_center: float
    hue_tolerance: float
    min_saturation: float
    min_value: float

    def mask(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
        delta = np.abs(((hue - self.hue_center) + 180.0) % 360.0 - 180.0)
        return (delta <= self.hue_tolerance) & (sat >= self.min_saturation) & (val >= self.min_value)


# Starting points, not measurements. Re-fit these against real captures
# before trusting a session: exact tints shift with arena skin, the
# day/night variants, and any display colour management.
DEFAULT_TEAM_COLORS = (
    TeamColor(TEAM_HOSTILE, hue_center=0.0, hue_tolerance=18.0,
              min_saturation=0.55, min_value=0.35),
    TeamColor(TEAM_FRIENDLY, hue_center=215.0, hue_tolerance=25.0,
              min_saturation=0.50, min_value=0.35),
)


@dataclass(frozen=True)
class VisionConfig:
    """Blob-detection thresholds. Every value is display-dependent."""

    # Deliberately permissive on width: the *filled* portion is what gets
    # segmented, so a nearly-dead unit's bar is only a few pixels wide.
    # Tightening this to "looks like a full bar" silently blinds the agent to
    # exactly the units it is about to finish off.
    min_bar_width: int = 4
    max_bar_height: int = 8
    min_aspect_ratio: float = 1.5   # bars are wider than they are tall
    min_area: int = 6
    # Health bars float above the unit; its feet — the position that matters
    # for arena coordinates — sit this many pixels lower.
    bar_to_feet_px: int = 14
    # Unfilled bar remainder: dark and desaturated.
    empty_max_saturation: float = 0.35
    empty_max_value: float = 0.45
    max_empty_scan_px: int = 60
    teams: tuple[TeamColor, ...] = DEFAULT_TEAM_COLORS


@dataclass(frozen=True)
class Blob:
    """A connected run of team-coloured pixels, in image coordinates."""

    x0: int
    y0: int
    x1: int   # inclusive
    y1: int   # inclusive
    area: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


@dataclass(frozen=True)
class Detection:
    """One perceived unit."""

    team: str
    tile_x: float
    tile_y: float
    pixel: tuple[float, float]
    hp_fraction: float
    hp_confident: bool
    blob: Blob

    @property
    def hostile(self) -> bool:
        return self.team == TEAM_HOSTILE


# ------------------------------------------------------- connected components


def connected_components(mask: np.ndarray) -> list[Blob]:
    """8-connected components of a boolean mask, as bounding boxes.

    Hand-rolled BFS over only the set pixels (health bars cover a tiny
    fraction of the frame), which keeps this dependency-free and comfortably
    inside the 0.5s decision budget.
    """
    ys, xs = np.nonzero(mask)
    remaining = set(zip(ys.tolist(), xs.tolist()))
    height, width = mask.shape
    blobs: list[Blob] = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        y0 = y1 = seed[0]
        x0 = x1 = seed[1]
        area = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            y0, y1 = min(y0, y), max(y1, y)
            x0, x1 = min(x0, x), max(x1, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and (ny, nx) in remaining:
                        remaining.discard((ny, nx))
                        queue.append((ny, nx))
        blobs.append(Blob(x0=x0, y0=y0, x1=x1, y1=y1, area=area))
    blobs.sort(key=lambda b: (b.y0, b.x0))
    return blobs


def _looks_like_health_bar(blob: Blob, config: VisionConfig) -> bool:
    return (blob.area >= config.min_area
            and blob.width >= config.min_bar_width
            and blob.height <= config.max_bar_height
            and blob.width >= config.min_aspect_ratio * blob.height)


def _fill_fraction(
    blob: Blob,
    sat: np.ndarray,
    val: np.ndarray,
    config: VisionConfig,
) -> tuple[float, bool]:
    """Fraction of the bar that is filled, and whether we actually measured it.

    The coloured blob is the *filled* part; the remainder is the bar's dark
    trough. Scanning outward along the bar's centre row for trough pixels
    recovers the full width. When no trough is found the unit is either at
    full HP or its bar is occluded — indistinguishable from pixels alone, so
    it reports 1.0 with `confident=False` rather than pretending.
    """
    row = int(round((blob.y0 + blob.y1) / 2.0))
    height, width = sat.shape
    if not (0 <= row < height):
        return 1.0, False
    empty = (sat[row] <= config.empty_max_saturation) & (val[row] <= config.empty_max_value)

    extra = 0
    for x in range(blob.x1 + 1, min(width, blob.x1 + 1 + config.max_empty_scan_px)):
        if not empty[x]:
            break
        extra += 1
    for x in range(blob.x0 - 1, max(-1, blob.x0 - 1 - config.max_empty_scan_px), -1):
        if not empty[x]:
            break
        extra += 1
    if extra == 0:
        return 1.0, False
    total = blob.width + extra
    return float(blob.width) / float(total), True


# ------------------------------------------------------------------ pipeline


def find_team_blobs(
    image: Image.Image | np.ndarray,
    config: VisionConfig | None = None,
) -> dict[str, list[Blob]]:
    """Team-tinted health-bar blobs, keyed by team name."""
    config = config or VisionConfig()
    array = np.asarray(image)
    hue, sat, val = rgb_to_hsv(array)
    out: dict[str, list[Blob]] = {}
    for team in config.teams:
        blobs = connected_components(team.mask(hue, sat, val))
        out[team.team] = [b for b in blobs if _looks_like_health_bar(b, config)]
    return out


def detect_units(
    image: Image.Image | np.ndarray,
    homography,
    config: VisionConfig | None = None,
    arena=None,
) -> list[Detection]:
    """Perceived units in arena-tile coordinates.

    `homography` maps pixels to tiles (see `src.live.homography`) and must
    already be rebased onto this capture's size. Detections whose feet fall
    outside the arena are dropped — those are HUD elements (the elixir bar,
    card rarity chips) that happened to match a team hue.
    """
    config = config or VisionConfig()
    array = np.asarray(image)
    hue, sat, val = rgb_to_hsv(array)
    detections: list[Detection] = []
    for team in config.teams:
        for blob in connected_components(team.mask(hue, sat, val)):
            if not _looks_like_health_bar(blob, config):
                continue
            cx, cy = blob.center
            feet = (cx, cy + config.bar_to_feet_px)
            try:
                tile_x, tile_y = homography.pixel_to_tile(*feet)
            except ValueError:
                continue
            if arena is not None and not (0.0 <= tile_x < arena.width
                                          and 0.0 <= tile_y < arena.height):
                continue
            fraction, confident = _fill_fraction(blob, sat, val, config)
            detections.append(Detection(
                team=team.team, tile_x=tile_x, tile_y=tile_y, pixel=feet,
                hp_fraction=fraction, hp_confident=confident, blob=blob))
    detections.sort(key=lambda d: (d.team, d.tile_y, d.tile_x))
    return detections
