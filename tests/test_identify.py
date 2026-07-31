"""Card identity constrained by the deck prior (#35)."""
from __future__ import annotations

import numpy as np
import pytest

from src.agent.opponent_tracker import OpponentTracker, TrackerConfig
from src.live.identify import (
    DeckPriorClassifier,
    TemplateLibrary,
    extract_patch,
)

DECK = ["hog_rider", "fireball", "musketeer", "cannon",
        "ice_spirit", "skeletons", "log", "ice_golem"]
OFF_DECK = ["golem", "lava_hound", "mega_knight"]


def _sprite(card: str, seed_shift: int = 0, noise: float = 0.0) -> np.ndarray:
    """A deterministic pseudo-sprite: stable per card, different across cards."""
    rng = np.random.default_rng(abs(hash(card)) % (2**32) + seed_shift)
    base = rng.integers(0, 256, (40, 40, 3), dtype=np.uint16)
    if noise:
        base = base + rng.normal(0, noise * 255, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture()
def library():
    lib = TemplateLibrary()
    for card in DECK + OFF_DECK:
        lib.add(card, _sprite(card))
    return lib


@pytest.fixture()
def tracker(cards, arena):
    return OpponentTracker(TrackerConfig.from_arena(arena), cards)


def _reveal(tracker, cards_played):
    for card in cards_played:
        tracker.elixir = 10.0
        tracker.observe_play(card)


# ---------------------------------------------------------------- templates


def test_library_round_trips_through_disk(library, tmp_path):
    library.save(tmp_path)
    loaded = TemplateLibrary.load(tmp_path)
    assert loaded.cards == library.cards
    assert len(loaded) == len(library)
    vector_source = _sprite("hog_rider")
    a = DeckPriorClassifier(library).classify(vector_source)
    b = DeckPriorClassifier(loaded).classify(vector_source)
    assert a.card == b.card == "hog_rider"


def test_matching_is_brightness_invariant(library):
    """Zero-meaning the patches is what stops "both are bright" from scoring
    as a match; the two arena halves are not lit the same."""
    clf = DeckPriorClassifier(library)
    dim = (_sprite("musketeer").astype(np.float32) * 0.55).astype(np.uint8)
    assert clf.classify(dim).card == "musketeer"


# ------------------------------------------------------------- the deck prior


def test_without_a_tracker_the_whole_library_is_in_play(library):
    clf = DeckPriorClassifier(library)
    result = clf.classify(_sprite("golem"))
    assert result.card == "golem"
    assert result.restricted is False
    assert result.n_candidates == len(library.cards)


def test_fully_revealed_deck_restricts_the_candidate_set(library, tracker):
    _reveal(tracker, DECK)
    clf = DeckPriorClassifier(library, tracker=tracker)
    candidates, restricted = clf.candidates()
    assert restricted is True
    assert sorted(candidates) == sorted(DECK)


def test_off_deck_card_is_never_returned_once_the_deck_is_known(library, tracker):
    """The strongest consequence of the prior: with all eight revealed, a
    patch that genuinely looks like an off-deck card must resolve to a deck
    card or to None — never to something they cannot be holding."""
    _reveal(tracker, DECK)
    clf = DeckPriorClassifier(library, tracker=tracker)
    result = clf.classify(_sprite("golem"))
    assert result.card != "golem"
    assert result.card is None or result.card in DECK


def test_candidate_set_narrows_as_the_match_goes_on(library, tracker):
    clf = DeckPriorClassifier(library, tracker=tracker)
    assert clf.candidates()[0] == library.cards
    _reveal(tracker, DECK[:4])
    assert clf.candidates()[1] is False  # not yet fully revealed
    _reveal(tracker, DECK[4:])
    cards, restricted = clf.candidates()
    assert restricted is True
    assert len(cards) == 8


def test_prior_breaks_a_tie_toward_a_revealed_card(cards, arena):
    """Before the deck is fully revealed the prior is a nudge, not a filter."""
    lib = TemplateLibrary()
    shared = _sprite("ambiguous")
    lib.add("hog_rider", shared)
    lib.add("golem", shared)
    tracker = OpponentTracker(TrackerConfig.from_arena(arena), cards)
    _reveal(tracker, ["hog_rider"])
    clf = DeckPriorClassifier(lib, tracker=tracker, min_margin=0.01)
    assert clf.classify(shared).card == "hog_rider"


def test_unrevealed_cards_stay_reachable_before_the_deck_is_known(library, tracker):
    _reveal(tracker, DECK[:3])
    clf = DeckPriorClassifier(library, tracker=tracker)
    assert clf.classify(_sprite("ice_golem")).card == "ice_golem"


# ----------------------------------------------------------- abstaining


def test_unrecognizable_patch_returns_none(library):
    clf = DeckPriorClassifier(library)
    blank = np.full((40, 40, 3), 128, np.uint8)
    assert clf.classify(blank).card is None


def test_ambiguous_match_returns_none_rather_than_guessing(library):
    """A wrong identity corrupts the tracker's cycle for the rest of the
    match, so a thin margin must abstain."""
    lib = TemplateLibrary()
    shared = _sprite("twins")
    lib.add("hog_rider", shared)
    lib.add("ram_rider", shared)
    clf = DeckPriorClassifier(lib)
    assert clf.classify(shared).card is None


# ------------------------------------------------------------- patch + label


def test_extract_patch_crops_below_the_health_bar(arena):
    from src.live.homography import Homography
    from src.live.vision import detect_units
    from tests.live_frames import DEFAULT_UNITS, render_frame

    image, meta = render_frame(arena, DEFAULT_UNITS, with_hud_distractor=False)
    h = Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})
    detection = detect_units(image, h, arena=arena)[0]
    patch = extract_patch(image, detection, width=20, height=20)
    assert patch.shape[2] == 3
    assert patch.shape[0] > 0 and patch.shape[1] > 0


def test_own_deploy_labelling_grows_the_library(arena):
    from src.live.homography import Homography
    from src.live.vision import TEAM_FRIENDLY, detect_units
    from tests.live_frames import DEFAULT_UNITS, render_frame

    image, meta = render_frame(arena, DEFAULT_UNITS, with_hud_distractor=False)
    h = Homography.from_anchors(
        arena, {k: tuple(v) for k, v in meta["homography_anchors"].items()})
    friendly = [d for d in detect_units(image, h, arena=arena) if d.team == TEAM_FRIENDLY]
    lib = TemplateLibrary()
    clf = DeckPriorClassifier(lib)
    clf.learn_from_own_deploy(image, friendly[0], "knight")
    assert lib.cards == ["knight"]
    assert len(lib) == 1
