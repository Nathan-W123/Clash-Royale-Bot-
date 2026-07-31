"""Configuration and validation for a calibrated live-play session."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_values(cls, values: list[int], name: str) -> "Rect":
        if len(values) != 4:
            raise ValueError(f"{name} must contain [x, y, width, height]")
        x, y, width, height = (int(value) for value in values)
        if width <= 0 or height <= 0:
            raise ValueError(f"{name} width and height must be positive")
        if x < 0 or y < 0:
            # A negative origin survives scaling and only blows up later, deep
            # in `vision.crop_array`, mid-session.
            raise ValueError(f"{name} x and y must be non-negative")
        return cls(x, y, width, height)


@dataclass(frozen=True)
class LiveConfig:
    transport: str
    adb_path: str | None
    device_serial: str | None
    poll_seconds: float
    action_cooldown_seconds: float
    match_indicator: Rect
    match_min_saturation: float
    card_slots: tuple[tuple[int, int], ...]
    card_ready_regions: tuple[Rect, ...]
    card_ready_min_luma: float
    preset_deck: tuple[str, ...]
    opening_hand: tuple[str, ...]
    draw_order: tuple[str, ...]
    placements: dict[str, tuple[int, int]]
    priority: tuple[str, ...]
    reference_size: tuple[int, int] = (1920, 1080)
    decision_mode: str = "dynamic_slots"
    dynamic_target: tuple[int, int] = (960, 700)
    slot_priority: tuple[int, ...] = (0, 1, 2, 3)
    desktop_capture: str = "virtual_desktop"
    window_title: str | None = None
    tap_delay_seconds: float = 0.15
    # Named arena anchors -> pixel points in the `reference_size` frame; see
    # src/live/homography.py. Optional: without it the bridge falls back to
    # fixed configured tap targets and cannot do arena perception.
    homography_anchors: tuple[tuple[str, tuple[int, int]], ...] = ()
    # The elixir bar, for reading own elixir. Required by decision_mode
    # 'policy': without it there is no affordability mask and the policy
    # would be asked to choose from cards it cannot pay for.
    elixir_bar: Rect | None = None
    checkpoint: str | None = None
    deck: tuple[str, ...] = ()

    def homography(self):
        """Solve the calibrated pixel<->tile mapping, or None if uncalibrated."""
        if not self.homography_anchors:
            return None
        from src.live.homography import Homography
        from src.simulator.cards import load_arena
        return Homography.from_anchors(load_arena(), dict(self.homography_anchors))

    def validate(self) -> None:
        if self.transport not in {"desktop", "adb"}:
            raise ValueError("transport must be either 'desktop' or 'adb'")
        if self.transport == "adb" and not self.adb_path:
            raise ValueError("adb_path is required when transport is 'adb'")
        if self.desktop_capture not in {"virtual_desktop", "window"}:
            raise ValueError("desktop_capture must be either 'virtual_desktop' or 'window'")
        if len(self.reference_size) != 2 or any(value <= 0 for value in self.reference_size):
            raise ValueError("reference_size must contain positive [width, height]")
        if self.decision_mode not in {"dynamic_slots", "known_deck", "policy"}:
            raise ValueError(
                "decision_mode must be 'policy', 'dynamic_slots' or 'known_deck'")
        if not self.slot_priority or any(slot not in range(4) for slot in self.slot_priority):
            raise ValueError("slot_priority must contain card-slot indexes from 0 through 3")
        if len(self.card_slots) != 4 or len(self.card_ready_regions) != 4:
            raise ValueError("live play requires exactly four card slots and ready regions")
        self._validate_geometry()
        if self.decision_mode == "policy":
            self._validate_policy_mode()
            return
        if self.decision_mode == "dynamic_slots":
            return
        if len(self.preset_deck) != 8:
            raise ValueError("preset_deck must contain exactly eight cards")
        if len(self.opening_hand) != 4:
            raise ValueError("opening_hand must contain exactly four cards")
        if len(set(self.preset_deck)) != len(self.preset_deck):
            raise ValueError("preset_deck may not contain duplicate card names")
        if not set(self.opening_hand).issubset(self.preset_deck):
            raise ValueError("opening_hand cards must belong to preset_deck")
        if len(self.draw_order) != 4 or not set(self.draw_order).issubset(self.preset_deck):
            raise ValueError("draw_order must contain the four remaining preset_deck cards")
        if set(self.opening_hand) | set(self.draw_order) != set(self.preset_deck):
            raise ValueError("opening_hand and draw_order must contain each preset_deck card exactly once")
        if not set(self.priority).issubset(self.preset_deck):
            raise ValueError("priority cards must belong to preset_deck")
        if not set(self.placements).issubset(self.preset_deck):
            raise ValueError("placements may only name cards in preset_deck")
        # `_choose_action` only ever returns a card that is both in `priority`
        # and in `placements`, so an empty intersection is a runner that polls
        # forever and never plays anything — a silent no-op rather than an
        # error, which is exactly the kind of failure this config should catch
        # at load time.
        playable = set(self.priority) & set(self.placements)
        if not playable:
            raise ValueError(
                "decision_mode 'known_deck' needs at least one card present in both "
                "`priority` and `placements`; otherwise the runner can never play "
                "anything. Calibrate placements (see configs/live_play.example.yaml) "
                "or use decision_mode 'dynamic_slots'."
            )

    def _validate_policy_mode(self) -> None:
        """A policy needs more calibration than the heuristics do.

        Each of these is checked at load time rather than mid-match because
        the failure modes are all silent: no homography means every placement
        lands at the same wrong pixel, no elixir bar means the affordability
        mask is fabricated, and a deck that does not match the one equipped
        in-game means the hand cycle drifts a card at a time.
        """
        if not self.homography_anchors:
            raise ValueError(
                "decision_mode 'policy' needs `homography_anchors`: the policy "
                "chooses an arena tile, and without a calibrated homography "
                "there is no way to turn that into a tap. See "
                "configs/live_play.example.yaml.")
        if self.elixir_bar is None:
            raise ValueError(
                "decision_mode 'policy' needs `elixir_bar`: without reading own "
                "elixir the affordability mask is fabricated and the policy will "
                "pick cards it cannot pay for.")
        if len(self.deck) != 8:
            raise ValueError(
                f"decision_mode 'policy' needs `deck` to list exactly the eight "
                f"cards equipped in-game, got {len(self.deck)}. The hand cycle is "
                f"simulated from it, so a mismatch drifts one card at a time.")
        if len(set(self.deck)) != len(self.deck):
            raise ValueError("`deck` may not contain duplicate cards")
        if not self.checkpoint:
            raise ValueError(
                "decision_mode 'policy' needs `checkpoint` (or --checkpoint) "
                "pointing at a trained `human`-tier policy.")

    def _validate_geometry(self) -> None:
        """Every calibrated coordinate must fall inside `reference_size`.

        Coordinates are recorded against one capture geometry and rescaled to
        whatever the live capture turns out to be, so a coordinate outside the
        reference frame is always a calibration mistake — usually a leftover
        from a previous reference (e.g. full-desktop coords kept after
        switching to window-local capture). Untouched, such a coordinate
        scales to a tap far outside the game window that silently does
        nothing, and the runner retries it forever.

        Checked for every mode, including coordinates the *current*
        decision_mode ignores: `placements` is dead weight under
        `dynamic_slots`, but leaving it unvalidated is exactly how it rots
        until someone flips the mode and every deploy misses.
        """
        width, height = self.reference_size

        def check_point(point: tuple[int, int], name: str) -> None:
            x, y = point
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(
                    f"{name} {tuple(point)} is outside reference_size "
                    f"{tuple(self.reference_size)} — recalibrate it against the "
                    f"capture geometry named by reference_size"
                )

        for slot, point in enumerate(self.card_slots):
            check_point(point, f"card_slots[{slot}]")
        check_point(self.dynamic_target, "dynamic_target")
        for name, point in sorted(self.placements.items()):
            check_point(point, f"placements[{name!r}]")
        for slot, rect in enumerate(self.card_ready_regions):
            if rect.x + rect.width > width or rect.y + rect.height > height:
                raise ValueError(
                    f"card_ready_regions[{slot}] {rect} extends past reference_size "
                    f"{tuple(self.reference_size)}"
                )
        if self.elixir_bar is not None:
            e = self.elixir_bar
            if e.x + e.width > width or e.y + e.height > height:
                raise ValueError(
                    f"elixir_bar {e} extends past reference_size "
                    f"{tuple(self.reference_size)}")
        r = self.match_indicator
        if r.x + r.width > width or r.y + r.height > height:
            raise ValueError(
                f"match_indicator {r} extends past reference_size "
                f"{tuple(self.reference_size)}"
            )
        self._validate_homography(check_point)

    def _validate_homography(self, check_point) -> None:
        """Anchors must be in-frame, named, enough, and actually solvable.

        Solving here rather than at first use is deliberate: a mismeasured
        anchor set produces a matrix that maps the board onto a line, and the
        symptom — every deploy landing in the same wrong place — is far
        harder to diagnose mid-session than a load-time error.
        """
        if not self.homography_anchors:
            return
        from src.live.homography import ANCHOR_NAMES, MIN_ANCHORS

        names = [name for name, _ in self.homography_anchors]
        unknown = sorted(set(names) - set(ANCHOR_NAMES))
        if unknown:
            raise ValueError(
                f"unknown homography_anchors {unknown}; expected any of {list(ANCHOR_NAMES)}")
        if len(names) < MIN_ANCHORS:
            raise ValueError(
                f"homography_anchors needs at least {MIN_ANCHORS} anchors, got {len(names)}")
        for name, point in self.homography_anchors:
            check_point(point, f"homography_anchors[{name!r}]")
        self.homography()  # raises on degenerate/inconsistent anchors


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    Plain YAML keeps the last occurrence silently. In a calibration file
    that's dangerous: a stray second `reference_size` overrides the one you
    think is in effect and every derived coordinate is quietly rescaled
    against the wrong frame.
    """


def _no_duplicate_keys(loader: yaml.Loader, node: yaml.MappingNode) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ValueError(
                f"duplicate key {key!r} in live config — YAML keeps only the "
                f"last one, so remove the earlier occurrence explicitly"
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def load_live_config(path: Path | str) -> LiveConfig:
    raw = yaml.load(Path(path).read_text(), Loader=_StrictLoader) or {}
    config = raw.get("live", raw)
    ready_regions = tuple(
        Rect.from_values(values, "card_ready_regions")
        for values in config["card_ready_regions"]
    )
    live = LiveConfig(
        transport=str(config.get("transport", "desktop")).lower(),
        adb_path=str(config["adb_path"]) if config.get("adb_path") else None,
        device_serial=config.get("device_serial"),
        poll_seconds=float(config.get("poll_seconds", 0.5)),
        action_cooldown_seconds=float(config.get("action_cooldown_seconds", 1.2)),
        match_indicator=Rect.from_values(config["match_indicator"], "match_indicator"),
        match_min_saturation=float(config.get("match_min_saturation", 0.12)),
        card_slots=tuple(tuple(map(int, point)) for point in config["card_slots"]),
        card_ready_regions=ready_regions,
        card_ready_min_luma=float(config.get("card_ready_min_luma", 55.0)),
        preset_deck=tuple(config.get("preset_deck", ())),
        opening_hand=tuple(config.get("opening_hand", ())),
        draw_order=tuple(config.get("draw_order", ())),
        placements={name: tuple(map(int, point)) for name, point in config.get("placements", {}).items()},
        priority=tuple(config.get("priority", ())),
        reference_size=tuple(map(int, config.get("reference_size", [1920, 1080]))),
        decision_mode=str(config.get("decision_mode", "dynamic_slots")).lower(),
        dynamic_target=tuple(map(int, config.get("dynamic_target", [960, 700]))),
        slot_priority=tuple(map(int, config.get("slot_priority", [0, 1, 2, 3]))),
        desktop_capture=str(config.get("desktop_capture", "virtual_desktop")).lower(),
        window_title=str(config["window_title"]) if config.get("window_title") else None,
        tap_delay_seconds=float(config.get("tap_delay_seconds", 0.15)),
        elixir_bar=(Rect.from_values(config["elixir_bar"], "elixir_bar")
                    if config.get("elixir_bar") else None),
        checkpoint=str(config["checkpoint"]) if config.get("checkpoint") else None,
        deck=tuple(config.get("deck", ())),
        homography_anchors=tuple(
            (str(name), (int(point[0]), int(point[1])))
            for name, point in (config.get("homography_anchors") or {}).items()),
    )
    live.validate()
    return live
