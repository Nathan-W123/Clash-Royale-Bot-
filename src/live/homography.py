"""Screen-pixel <-> arena-tile mapping via a planar homography (#33).

Clash Royale renders the arena in perspective: the far half of the board is
visibly narrower than the near half. An affine scale (what `scaled_point` in
`runner.py` does for UI widgets) is right for the HUD, which is drawn flat,
but wrong for the playfield — it puts a tile-accurate placement several tiles
off at the far end. A full 3x3 homography is the correct model for a plane
viewed by a pinhole camera, and it is only four correspondences of work.

Both directions matter:
  * **pixel -> tile** for perception (#34): blob centroids become arena
    coordinates the observation encoder can bin.
  * **tile -> pixel** for control: `runner.py` currently taps one fixed
    configured point per card, which is what caps live play at a
    hand-written heuristic. Tile->pixel is what lets a trained policy pick
    an arbitrary placement cell and have it land where it meant.

Calibration anchors: the six tower centres plus the two bridge ends. They
are visually distinctive, spread across the board (the far towers are what
pin down the perspective term), and their tile coordinates are already known
from `configs/arena.yaml`, so the user only has to click eight pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulator.cards import ArenaConfig

# Tile-space anchors, resolved against the arena config. "own" is the bottom
# player — the seat a live session always occupies.
ANCHOR_NAMES = (
    "own_king",
    "own_princess_left",
    "own_princess_right",
    "enemy_king",
    "enemy_princess_left",
    "enemy_princess_right",
    "bridge_left",
    "bridge_right",
)

MIN_ANCHORS = 4
# Max allowed round-trip error, in tiles, when re-projecting the calibration
# anchors through the solved matrix.
FIT_TOLERANCE_TILES = 0.75


def anchor_tiles(arena: ArenaConfig) -> dict[str, tuple[float, float]]:
    """Arena-tile coordinates of every named calibration anchor."""
    h = arena.height
    river_y = (arena.river_y_min + arena.river_y_max) / 2.0
    left_x, right_x = min(arena.bridge_xs), max(arena.bridge_xs)
    kx, ky = arena.king_pos
    plx, ply = arena.princess_left_pos
    prx, pry = arena.princess_right_pos
    return {
        "own_king": (kx, ky),
        "own_princess_left": (plx, ply),
        "own_princess_right": (prx, pry),
        "enemy_king": (kx, h - ky),
        "enemy_princess_left": (plx, h - ply),
        "enemy_princess_right": (prx, h - pry),
        "bridge_left": (left_x, river_y),
        "bridge_right": (right_x, river_y),
    }


def _normalizing_transform(points: np.ndarray) -> np.ndarray:
    """Hartley normalization: centre at the origin, mean distance sqrt(2).

    Skipping this is the classic way to get a homography that "works" on the
    calibration points and drifts badly elsewhere — pixel coordinates in the
    hundreds and tile coordinates in the tens differ by two orders of
    magnitude, which conditions the DLT system terribly.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_dist = float(np.mean(np.hypot(centred[:, 0], centred[:, 1])))
    if mean_dist < 1e-9:
        raise ValueError("degenerate calibration: all points are coincident")
    s = np.sqrt(2.0) / mean_dist
    return np.array([[s, 0.0, -s * centroid[0]],
                     [0.0, s, -s * centroid[1]],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _apply(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    vec = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    w = vec[2]
    if abs(w) < 1e-12:
        raise ValueError(f"point ({x}, {y}) maps to the horizon; calibration is degenerate")
    return float(vec[0] / w), float(vec[1] / w)


@dataclass(frozen=True, eq=False)
class Homography:
    """Pixel->tile plane mapping plus its inverse."""

    matrix: np.ndarray          # pixel -> tile
    inverse_matrix: np.ndarray  # tile -> pixel

    # ------------------------------------------------------------- solving

    @classmethod
    def solve(cls, pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> "Homography":
        """Fit from ``[(pixel_xy, tile_xy), ...]``; needs at least four pairs.

        Raises on degenerate input (fewer than four pairs, coincident points,
        collinear anchors) rather than returning a matrix that silently maps
        the whole board onto a line.
        """
        if len(pairs) < MIN_ANCHORS:
            raise ValueError(
                f"a homography needs at least {MIN_ANCHORS} correspondences, got {len(pairs)}")
        src = np.asarray([p for p, _ in pairs], dtype=np.float64)
        dst = np.asarray([t for _, t in pairs], dtype=np.float64)
        _reject_collinear(src, "pixel")
        _reject_collinear(dst, "tile")

        t_src = _normalizing_transform(src)
        t_dst = _normalizing_transform(dst)
        src_n = np.array([_apply(t_src, x, y) for x, y in src])
        dst_n = np.array([_apply(t_dst, x, y) for x, y in dst])

        rows = []
        for (u, v), (x, y) in zip(src_n, dst_n):
            rows.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
            rows.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
        _, singular, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        # A well-posed DLT system has a *one*-dimensional null space. A wider
        # one means the anchors admit a family of solutions and the SVD's
        # arbitrary pick would be garbage.
        if singular[-2] < 1e-8 * singular[0]:
            raise ValueError("degenerate calibration anchors: no unique homography")
        h_norm = vt[-1].reshape(3, 3)

        matrix = np.linalg.inv(t_dst) @ h_norm @ t_src
        if abs(matrix[2, 2]) > 1e-12:
            matrix = matrix / matrix[2, 2]
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("degenerate calibration anchors: singular homography")

        homography = cls(matrix=matrix, inverse_matrix=np.linalg.inv(matrix))
        homography._check_fit(pairs)
        return homography

    @classmethod
    def from_anchors(
        cls,
        arena: ArenaConfig,
        pixel_anchors: dict[str, tuple[float, float]],
    ) -> "Homography":
        """Fit from named anchors (see `ANCHOR_NAMES`) to their pixel points."""
        tiles = anchor_tiles(arena)
        unknown = set(pixel_anchors) - set(tiles)
        if unknown:
            raise ValueError(
                f"unknown homography anchors {sorted(unknown)}; expected {list(ANCHOR_NAMES)}")
        pairs = [(tuple(map(float, pixel_anchors[name])), tiles[name])
                 for name in ANCHOR_NAMES if name in pixel_anchors]
        return cls.solve(pairs)

    def _check_fit(self, pairs) -> None:
        worst = max(
            float(np.hypot(*(np.subtract(self.pixel_to_tile(*p), t))))
            for p, t in pairs)
        if worst > FIT_TOLERANCE_TILES:
            raise ValueError(
                f"homography does not fit its own anchors (worst residual "
                f"{worst:.2f} tiles > {FIT_TOLERANCE_TILES}); the anchors are "
                f"inconsistent or mismeasured")

    # ------------------------------------------------------------- mapping

    def pixel_to_tile(self, px: float, py: float) -> tuple[float, float]:
        return _apply(self.matrix, float(px), float(py))

    def tile_to_pixel(self, tx: float, ty: float) -> tuple[float, float]:
        return _apply(self.inverse_matrix, float(tx), float(ty))

    def scaled_to(self, reference_size: tuple[int, int],
                  image_size: tuple[int, int]) -> "Homography":
        """Rebase onto a differently-sized capture.

        Anchors are calibrated once against `reference_size` and the live
        window is whatever it is, exactly like every other coordinate in
        `live_play.yaml` — so the same rescale discipline applies here rather
        than forcing recalibration whenever the window changes.
        """
        sx = image_size[0] / reference_size[0]
        sy = image_size[1] / reference_size[1]
        scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        matrix = self.matrix @ np.linalg.inv(scale)
        return Homography(matrix=matrix, inverse_matrix=np.linalg.inv(matrix))


def _reject_collinear(points: np.ndarray, label: str) -> None:
    """Raise if the points span fewer than two dimensions.

    Four collinear anchors still produce *a* matrix from the SVD — one that
    maps the entire board onto a line. Silently returning that is much worse
    than refusing to calibrate.
    """
    centred = points - points.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[0] < 1e-9:
        raise ValueError(f"degenerate {label} anchors: all points are coincident")
    if singular[1] < 1e-6 * singular[0]:
        raise ValueError(f"degenerate {label} anchors: points are collinear")
