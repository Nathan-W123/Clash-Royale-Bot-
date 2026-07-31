from PIL import Image

from src.live.config import LiveConfig, Rect
from src.live.runner import HandCycle, LiveMatchRunner


def make_config() -> LiveConfig:
    return LiveConfig(
        transport="desktop",
        adb_path=None,
        device_serial=None,
        poll_seconds=0.5,
        action_cooldown_seconds=0.0,
        match_indicator=Rect(0, 0, 10, 10),
        match_min_saturation=0.1,
        card_slots=((10, 90), (30, 90), (50, 90), (70, 90)),
        card_ready_regions=(Rect(10, 70, 10, 10), Rect(30, 70, 10, 10), Rect(50, 70, 10, 10), Rect(70, 70, 10, 10)),
        card_ready_min_luma=50.0,
        preset_deck=("giant", "musketeer", "mini_pekka", "minions", "knight", "goblins", "fireball", "cannon"),
        opening_hand=("giant", "musketeer", "mini_pekka", "minions"),
        draw_order=("knight", "goblins", "fireball", "cannon"),
        placements={"giant": (50, 50)},
        priority=("giant",),
        reference_size=(100, 100),
        decision_mode="known_deck",
    )


def test_hand_cycle_replaces_played_card_with_next_card():
    cycle = HandCycle.from_config(make_config())
    assert cycle.play(0) == "giant"
    assert cycle.hand == ["knight", "musketeer", "mini_pekka", "minions"]
    assert cycle.next_cards[-1] == "giant"


def test_runner_only_taps_when_armed():
    class Device:
        def __init__(self):
            self.taps = []

        def screenshot(self):
            image = Image.new("RGB", (100, 100), (0, 0, 0))
            for x in range(10):
                for y in range(10):
                    image.putpixel((x, y), (255, 0, 0))
            for x in range(10, 20):
                for y in range(70, 80):
                    image.putpixel((x, y), (255, 255, 255))
            return image

        def tap(self, x, y):
            self.taps.append((x, y))

    device = Device()
    runner = LiveMatchRunner(make_config(), device, armed=False, log=lambda _: None)
    runner.step()
    assert device.taps == []
    runner = LiveMatchRunner(make_config(), device, armed=True, log=lambda _: None)
    runner.step()
    assert device.taps == [(10, 90), (50, 50)]
