"""Conservative live-match loop for a known eight-card deck."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from src.live.config import LiveConfig, Rect
from src.live.device import LiveDevice
from src.live.vision import mean_luma, mean_saturation


def scaled_point(config: LiveConfig, point: tuple[int, int], image_size: tuple[int, int]) -> tuple[int, int]:
    scale_x = image_size[0] / config.reference_size[0]
    scale_y = image_size[1] / config.reference_size[1]
    return (round(point[0] * scale_x), round(point[1] * scale_y))


def scaled_rect(config: LiveConfig, rect: Rect, image_size: tuple[int, int]) -> Rect:
    x, y = scaled_point(config, (rect.x, rect.y), image_size)
    width, height = scaled_point(config, (rect.width, rect.height), image_size)
    return type(rect)(x, y, max(1, width), max(1, height))


@dataclass
class HandCycle:
    hand: list[str]
    next_cards: list[str]

    @classmethod
    def from_config(cls, config: LiveConfig) -> "HandCycle":
        return cls(list(config.opening_hand), list(config.draw_order))

    def play(self, slot: int) -> str:
        card = self.hand[slot]
        self.hand[slot] = self.next_cards.pop(0)
        self.next_cards.append(card)
        return card


class LiveMatchRunner:
    """Watch an existing match and deploy configured preset-deck cards.

    The runner does not start matchmaking. It only taps after the calibrated
    match indicator is visible, and only when launched with ``armed=True``.
    """

    def __init__(self, config: LiveConfig, device: LiveDevice, armed: bool = False, log: Callable[[str], None] = print):
        self.config = config
        self.device = device
        self.armed = armed
        self.log = log
        self.hand = HandCycle.from_config(config)
        self._was_in_match = False
        self._last_action_at = 0.0
        self._last_logged_choice: tuple[int, str] | None = None
        # None when `homography_anchors` isn't calibrated; the runner then
        # keeps its legacy fixed-target behaviour.
        self.homography = config.homography()

    def tile_to_pixel(self, tile: tuple[float, float],
                      image_size: tuple[int, int]) -> tuple[int, int] | None:
        """Arena tile -> tap point on this capture, or None if uncalibrated.

        This is the piece that lets a *policy* drive live play instead of the
        heuristic below: the policy picks a placement cell, and this turns it
        into a tap. The arena is drawn in perspective, so it goes through the
        calibrated homography rather than `scaled_point`, which is only
        correct for the flat HUD.
        """
        if self.homography is None:
            return None
        matrix = self.homography.scaled_to(self.config.reference_size, image_size)
        px, py = matrix.tile_to_pixel(*tile)
        return round(px), round(py)

    def run_forever(self, max_consecutive_errors: int = 10) -> None:
        """Poll until interrupted, surviving transient capture failures.

        A watcher runs unattended for a whole match, so a single recoverable
        hiccup — the window momentarily minimized (empty client area), a
        blocked screen grab, a dropped touch injection — must not end the
        session. Persistent failure still stops rather than spinning forever,
        because that means the setup is genuinely broken (game closed, window
        title changed) and retrying can't fix it.
        """
        consecutive_errors = 0
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                self.log("Interrupted; stopping.")
                raise
            except (RuntimeError, OSError, ValueError) as error:
                consecutive_errors += 1
                self.log(
                    f"Recoverable error ({consecutive_errors}/{max_consecutive_errors}): {error}"
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"Giving up after {max_consecutive_errors} consecutive errors; "
                        f"last was: {error}"
                    ) from error
            else:
                consecutive_errors = 0
            time.sleep(self.config.poll_seconds)

    def step(self) -> None:
        image = self.device.screenshot()
        in_match = mean_saturation(image, scaled_rect(self.config, self.config.match_indicator, image.size)) >= self.config.match_min_saturation
        if in_match and not self._was_in_match:
            self.hand = HandCycle.from_config(self.config)
            self._last_logged_choice = None
            self.log("Match detected; hand cycle reset.")
        self._was_in_match = in_match
        if not in_match or time.monotonic() - self._last_action_at < self.config.action_cooldown_seconds:
            return
        ready_slots = [
            slot for slot, region in enumerate(self.config.card_ready_regions)
            if mean_luma(image, scaled_rect(self.config, region, image.size)) >= self.config.card_ready_min_luma
        ]
        choice = self._choose_action(ready_slots)
        if choice is None:
            return
        slot, card = choice
        if not self.armed and choice == self._last_logged_choice:
            return
        if self.config.decision_mode == "dynamic_slots":
            target = self.config.dynamic_target
        else:
            target = self.config.placements.get(card)
            if target is None:
                return
        card_slot = scaled_point(self.config, self.config.card_slots[slot], image.size)
        target = scaled_point(self.config, target, image.size)
        self.log(f"{'Playing' if self.armed else 'Would play'} {card} from slot {slot + 1} at {target}.")
        if self.armed:
            self.device.tap(*card_slot)
            time.sleep(self.config.tap_delay_seconds)
            self.device.tap(*target)
            if self.config.decision_mode == "known_deck":
                self.hand.play(slot)
        else:
            self._last_logged_choice = choice
        self._last_action_at = time.monotonic()

    def _choose_action(self, ready_slots: list[int]) -> tuple[int, str] | None:
        if self.config.decision_mode == "dynamic_slots":
            for slot in self.config.slot_priority:
                if slot in ready_slots:
                    return slot, f"visible slot {slot + 1}"
            return None
        for card in self.config.priority:
            for slot in ready_slots:
                if self.hand.hand[slot] == card and card in self.config.placements:
                    return slot, card
        return None
