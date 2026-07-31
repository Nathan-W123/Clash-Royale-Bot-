"""Team-tinted blob detection (#34), against saved frames only.

Every assertion runs on a rendered fixture; nothing here touches a live
match. See `tests/live_frames.py` for the fixture format and for how to add
real annotated captures alongside the synthetic one.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from src.live.homography import Homography
from src.live.vision import (
    TEAM_FRIENDLY,
    TEAM_HOSTILE,
    VisionConfig,
    connected_components,
    detect_units,
    find_team_blobs,
    rgb_to_hsv,
)
from tests.live_frames import (
    DEFAULT_UNITS,
    FIXTURE_DIR,
    PlannedUnit,
    render_frame,
)


@pytest.fixture()
def frame(arena):
    return render_frame(arena, DEFAULT_UNITS)


def _homography(arena, meta):
    return Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})


# -------------------------------------------------------------- primitives


def test_rgb_to_hsv_matches_known_colours():
    array = np.array([[[255, 0, 0], [0, 0, 255], [128, 128, 128], [0, 0, 0]]], np.uint8)
    hue, sat, val = rgb_to_hsv(array)
    assert hue[0, 0] == pytest.approx(0.0)
    assert hue[0, 1] == pytest.approx(240.0)
    assert sat[0, 0] == pytest.approx(1.0)
    assert sat[0, 2] == pytest.approx(0.0)
    assert val[0, 3] == pytest.approx(0.0)


def test_connected_components_separates_and_bounds():
    mask = np.zeros((10, 20), bool)
    mask[2:4, 3:9] = True
    mask[7:9, 12:18] = True
    blobs = connected_components(mask)
    assert len(blobs) == 2
    first = blobs[0]
    assert (first.x0, first.y0, first.x1, first.y1) == (3, 2, 8, 3)
    assert first.area == 12
    assert first.center == (5.5, 2.5)


def test_connected_components_joins_diagonally():
    mask = np.zeros((6, 6), bool)
    mask[1, 1] = mask[2, 2] = True
    assert len(connected_components(mask)) == 1


# ------------------------------------------------------------- segmentation


def test_team_blobs_are_separated_by_tint(frame):
    image, meta = frame
    blobs = find_team_blobs(image)
    expected_hostile = sum(1 for u in meta["units"] if u["team"] == TEAM_HOSTILE)
    expected_friendly = sum(1 for u in meta["units"] if u["team"] == TEAM_FRIENDLY)
    # +1 hostile: the HUD distractor is still a red bar at this stage; it is
    # discarded later, by arena bounds, not by colour.
    assert len(blobs[TEAM_HOSTILE]) == expected_hostile + 1
    assert len(blobs[TEAM_FRIENDLY]) == expected_friendly


def test_arena_background_produces_no_blobs(arena):
    image, _ = render_frame(arena, [], with_hud_distractor=False)
    blobs = find_team_blobs(image)
    assert blobs[TEAM_HOSTILE] == []
    assert blobs[TEAM_FRIENDLY] == []


def test_aspect_ratio_filter_rejects_square_patches(arena):
    image, _ = render_frame(arena, [], with_hud_distractor=False)
    array = np.asarray(image).copy()
    array[300:340, 100:140] = (222, 40, 38)  # big square, not a bar
    assert find_team_blobs(Image.fromarray(array))[TEAM_HOSTILE] == []


# ---------------------------------------------------------------- detection


def test_detections_recover_tile_positions(frame, arena):
    image, meta = frame
    h = _homography(arena, meta)
    detections = detect_units(image, h, arena=arena)
    assert len(detections) == len(meta["units"])
    for planned in meta["units"]:
        near = [d for d in detections
                if d.team == planned["team"]
                and np.hypot(d.tile_x - planned["tile"][0],
                             d.tile_y - planned["tile"][1]) < 0.75]
        assert near, f"no detection near {planned}"


def test_off_board_hud_elements_are_discarded(frame, arena):
    image, meta = frame
    h = _homography(arena, meta)
    with_bounds = detect_units(image, h, arena=arena)
    without_bounds = detect_units(image, h, arena=None)
    assert len(without_bounds) == len(with_bounds) + 1


def test_hp_fraction_is_read_from_bar_fill(frame, arena):
    image, meta = frame
    h = _homography(arena, meta)
    detections = detect_units(image, h, arena=arena)
    for planned in meta["units"]:
        if planned["hp_fraction"] >= 1.0:
            continue
        match = min(detections,
                    key=lambda d: np.hypot(d.tile_x - planned["tile"][0],
                                           d.tile_y - planned["tile"][1]))
        assert match.hp_confident
        assert match.hp_fraction == pytest.approx(planned["hp_fraction"], abs=0.1)


def test_full_hp_bar_is_reported_but_flagged_unconfident(arena):
    """The documented fidelity loss: a full bar has no visible remainder, so
    'full HP' and 'bar occluded' are the same pixels. Report it, flag it."""
    image, meta = render_frame(arena, [PlannedUnit("hostile", (9.0, 20.0), 1.0)],
                               with_hud_distractor=False)
    h = _homography(arena, meta)
    detections = detect_units(image, h, arena=arena)
    assert len(detections) == 1
    assert detections[0].hp_fraction == 1.0
    assert detections[0].hp_confident is False


def test_detection_survives_a_larger_capture(arena):
    """Anchors are calibrated once against a reference size; a bigger window
    must not require recalibration."""
    image, meta = render_frame(arena, DEFAULT_UNITS, with_hud_distractor=False)
    doubled = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
    h = _homography(arena, meta).scaled_to(tuple(meta["reference_size"]), doubled.size)
    config = VisionConfig(bar_to_feet_px=28, min_bar_width=12)
    detections = detect_units(doubled, h, config=config, arena=arena)
    assert len(detections) == len(meta["units"])


# ------------------------------------------- real captures, when they exist


def _saved_fixtures():
    if not FIXTURE_DIR.exists():
        return []
    return sorted(p for p in FIXTURE_DIR.glob("*.json"))


@pytest.mark.parametrize("meta_path", _saved_fixtures(),
                         ids=lambda p: p.stem)
def test_saved_fixture_frames(meta_path, arena):
    """Runs over every committed frame, synthetic or hand-labelled."""
    meta = json.loads(meta_path.read_text())
    image = Image.open(meta_path.with_suffix(".png")).convert("RGB")
    h = _homography(arena, meta).scaled_to(tuple(meta["reference_size"]), image.size)
    detections = detect_units(image, h, arena=arena)
    for planned in meta["units"]:
        near = [d for d in detections
                if d.team == planned["team"]
                and np.hypot(d.tile_x - planned["tile"][0],
                             d.tile_y - planned["tile"][1]) < 1.0]
        assert near, f"{meta_path.stem}: nothing detected near {planned}"
