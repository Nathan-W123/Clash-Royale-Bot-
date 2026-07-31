"""Screen<->arena homography (#33)."""
from __future__ import annotations

import numpy as np
import pytest

from src.live.homography import (
    ANCHOR_NAMES,
    Homography,
    anchor_tiles,
)


def _perspective_camera(arena, size=(556, 1028)):
    """A synthetic 'render': tile -> pixel with a real perspective term, so
    an affine fit provably cannot reproduce it."""
    w, h = size

    def project(tx, ty):
        # Far half (large ty) is compressed toward the top of the screen and
        # narrowed toward the centre — the same effect CR's camera has.
        depth = 1.0 + 0.45 * (ty / arena.height)
        px = w / 2 + (tx - arena.width / 2) / depth * (w / arena.width) * 0.9
        py = h - (ty / arena.height) * h * 0.82 / depth - 40.0
        return px, py

    return project


def _pairs(arena, project, names=ANCHOR_NAMES):
    tiles = anchor_tiles(arena)
    return [(project(*tiles[n]), tiles[n]) for n in names]


def test_anchor_tiles_covers_every_named_anchor(arena):
    tiles = anchor_tiles(arena)
    assert set(tiles) == set(ANCHOR_NAMES)
    assert tiles["own_king"][1] < tiles["enemy_king"][1]
    assert tiles["bridge_left"][0] < tiles["bridge_right"][0]


def test_round_trips_synthetic_anchors(arena):
    project = _perspective_camera(arena)
    h = Homography.solve(_pairs(arena, project))
    for tx in (1.0, 4.5, 9.0, 13.0, 17.0):
        for ty in (1.0, 8.0, 16.0, 24.0, 31.0):
            px, py = h.tile_to_pixel(tx, ty)
            rx, ry = h.pixel_to_tile(px, py)
            assert (rx, ry) == pytest.approx((tx, ty), abs=1e-6)


def test_recovers_the_camera_it_was_calibrated_from(arena):
    project = _perspective_camera(arena)
    h = Homography.solve(_pairs(arena, project))
    for tx, ty in [(2.0, 3.0), (9.0, 20.0), (16.0, 29.0), (5.0, 16.0)]:
        assert h.tile_to_pixel(tx, ty) == pytest.approx(project(tx, ty), abs=1e-4)
        assert h.pixel_to_tile(*project(tx, ty)) == pytest.approx((tx, ty), abs=1e-4)


def test_four_anchors_are_enough(arena):
    project = _perspective_camera(arena)
    names = ("own_princess_left", "own_princess_right",
             "enemy_princess_left", "enemy_princess_right")
    h = Homography.solve(_pairs(arena, project, names))
    assert h.pixel_to_tile(*project(9.0, 16.0)) == pytest.approx((9.0, 16.0), abs=1e-4)


def test_affine_scaling_is_not_sufficient(arena):
    """Justifies the whole module: a pure scale fitted to the near half is
    tiles off at the far end."""
    project = _perspective_camera(arena)
    h = Homography.solve(_pairs(arena, project))
    near_px, near_py = project(9.0, 2.5)
    scale_x = near_px / 9.0
    scale_y = near_py / 2.5
    far_px, far_py = project(9.0, 29.5)
    affine_guess = (far_px / scale_x, far_py / scale_y)
    exact = h.pixel_to_tile(far_px, far_py)
    assert exact == pytest.approx((9.0, 29.5), abs=1e-4)
    assert np.hypot(*np.subtract(affine_guess, exact)) > 2.0


def test_rescaling_to_a_different_capture_size(arena):
    project = _perspective_camera(arena, size=(556, 1028))
    h = Homography.solve(_pairs(arena, project))
    bigger = h.scaled_to((556, 1028), (1112, 2056))
    px, py = project(9.0, 16.0)
    assert bigger.pixel_to_tile(px * 2, py * 2) == pytest.approx((9.0, 16.0), abs=1e-4)


# ------------------------------------------------------------- degeneracies


def test_too_few_anchors_raises(arena):
    project = _perspective_camera(arena)
    with pytest.raises(ValueError, match="at least 4"):
        Homography.solve(_pairs(arena, project, ANCHOR_NAMES[:3]))


def test_collinear_anchors_raise_rather_than_returning_garbage(arena):
    tiles = anchor_tiles(arena)
    pairs = [((100.0 + 10 * i, 200.0 + 10 * i), tiles[n])
             for i, n in enumerate(ANCHOR_NAMES[:4])]
    with pytest.raises(ValueError, match="collinear"):
        Homography.solve(pairs)


def test_coincident_anchors_raise(arena):
    tiles = anchor_tiles(arena)
    pairs = [((100.0, 200.0), tiles[n]) for n in ANCHOR_NAMES[:4]]
    with pytest.raises(ValueError, match="coincident"):
        Homography.solve(pairs)


def test_inconsistent_anchors_are_rejected(arena):
    """Four points always admit *a* homography, so the guard that matters is
    the over-determined fit residual: swap two anchors and the eight-point
    solve can no longer explain them."""
    project = _perspective_camera(arena)
    pairs = _pairs(arena, project)
    pairs[0], pairs[3] = (pairs[0][0], pairs[3][1]), (pairs[3][0], pairs[0][1])
    with pytest.raises(ValueError, match="does not fit"):
        Homography.solve(pairs)


def test_unknown_anchor_name_rejected(arena):
    with pytest.raises(ValueError, match="unknown homography anchors"):
        Homography.from_anchors(arena, {"midfield": (1, 2), "own_king": (3, 4),
                                        "enemy_king": (5, 6), "bridge_left": (7, 8)})


# -------------------------------------------------------- config integration


def test_live_config_validates_and_solves_anchors(tmp_path, arena):
    import yaml

    from src.live.config import load_live_config

    project = _perspective_camera(arena, size=(556, 1028))
    tiles = anchor_tiles(arena)
    anchors = {name: [round(v) for v in project(*tiles[name])] for name in ANCHOR_NAMES}
    raw = {
        "live": {
            "transport": "desktop",
            "reference_size": [556, 1028],
            "dynamic_target": [278, 550],
            "match_indicator": [160, 995, 350, 25],
            "card_slots": [[167, 917], [278, 917], [386, 917], [492, 917]],
            "card_ready_regions": [[140, 851, 55, 57], [250, 851, 55, 57],
                                   [358, 851, 55, 57], [464, 851, 55, 57]],
            "homography_anchors": anchors,
        }
    }
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_live_config(path)
    h = config.homography()
    assert h is not None
    assert h.pixel_to_tile(*project(9.0, 16.0)) == pytest.approx((9.0, 16.0), abs=0.5)


def test_live_config_rejects_out_of_frame_anchor(tmp_path, arena):
    import yaml

    from src.live.config import load_live_config

    project = _perspective_camera(arena, size=(556, 1028))
    tiles = anchor_tiles(arena)
    anchors = {name: [round(v) for v in project(*tiles[name])] for name in ANCHOR_NAMES}
    anchors["own_king"] = [9999, 10]
    raw = {
        "live": {
            "transport": "desktop",
            "reference_size": [556, 1028],
            "dynamic_target": [278, 550],
            "match_indicator": [160, 995, 350, 25],
            "card_slots": [[167, 917], [278, 917], [386, 917], [492, 917]],
            "card_ready_regions": [[140, 851, 55, 57], [250, 851, 55, 57],
                                   [358, 851, 55, 57], [464, 851, 55, 57]],
            "homography_anchors": anchors,
        }
    }
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="outside reference_size"):
        load_live_config(path)


def _live_config(tmp_path, arena, with_anchors=True):
    import yaml

    from src.live.config import load_live_config

    project = _perspective_camera(arena, size=(556, 1028))
    tiles = anchor_tiles(arena)
    body = {
        "transport": "desktop",
        "reference_size": [556, 1028],
        "dynamic_target": [278, 550],
        "match_indicator": [160, 995, 350, 25],
        "card_slots": [[167, 917], [278, 917], [386, 917], [492, 917]],
        "card_ready_regions": [[140, 851, 55, 57], [250, 851, 55, 57],
                               [358, 851, 55, 57], [464, 851, 55, 57]],
    }
    if with_anchors:
        body["homography_anchors"] = {
            name: [round(v) for v in project(*tiles[name])] for name in ANCHOR_NAMES}
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump({"live": body}))
    return load_live_config(path), project


def test_runner_maps_tiles_to_tap_points(tmp_path, arena):
    """The piece that lets a trained policy pick an arbitrary placement
    instead of the one fixed configured target."""
    from src.live.runner import LiveMatchRunner

    config, project = _live_config(tmp_path, arena)
    runner = LiveMatchRunner(config, device=None)
    assert runner.homography is not None
    tap = runner.tile_to_pixel((9.0, 10.0), (556, 1028))
    assert tap == pytest.approx(tuple(round(v) for v in project(9.0, 10.0)), abs=2)


def test_runner_rescales_tap_points_for_a_bigger_window(tmp_path, arena):
    from src.live.runner import LiveMatchRunner

    config, project = _live_config(tmp_path, arena)
    runner = LiveMatchRunner(config, device=None)
    tap = runner.tile_to_pixel((9.0, 10.0), (1112, 2056))
    expected = tuple(round(v * 2) for v in project(9.0, 10.0))
    assert tap == pytest.approx(expected, abs=4)


def test_runner_without_calibration_keeps_legacy_behaviour(tmp_path, arena):
    from src.live.runner import LiveMatchRunner

    config, _ = _live_config(tmp_path, arena, with_anchors=False)
    runner = LiveMatchRunner(config, device=None)
    assert runner.homography is None
    assert runner.tile_to_pixel((9.0, 10.0), (556, 1028)) is None
