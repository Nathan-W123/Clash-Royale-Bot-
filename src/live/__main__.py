"""Run the calibrated live-play bridge."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.live.config import LiveConfig, load_live_config
from src.live.device import ADBDevice, LiveDevice, WindowsDesktopDevice
from src.live.runner import LiveMatchRunner, scaled_rect
from src.live.vision import mean_luma, mean_saturation


def diagnose(config: LiveConfig, device: LiveDevice, out_path: Path) -> None:
    """Capture one screenshot and print the raw values match/ready detection uses.

    Point this at whatever screen is giving false positives (e.g. the home
    screen) and compare its numbers against a capture taken mid-match to pick
    a `match_indicator` rect and thresholds that actually separate the two.
    """
    image = device.screenshot()
    image.save(out_path)
    print(f"Captured {image.size[0]}x{image.size[1]} screenshot -> {out_path}")

    indicator_rect = scaled_rect(config, config.match_indicator, image.size)
    saturation = mean_saturation(image, indicator_rect)
    verdict = "IN MATCH" if saturation >= config.match_min_saturation else "not in match"
    print(
        f"match_indicator {indicator_rect}: saturation={saturation:.3f} "
        f"(threshold={config.match_min_saturation:.3f}) -> {verdict}"
    )

    for slot, region in enumerate(config.card_ready_regions):
        rect = scaled_rect(config, region, image.size)
        luma = mean_luma(image, rect)
        ready = "ready" if luma >= config.card_ready_min_luma else "not ready"
        print(f"card_ready_regions[{slot}] {rect}: luma={luma:.1f} (threshold={config.card_ready_min_luma:.1f}) -> {ready}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an already-running Clash Royale match through a calibrated display.")
    parser.add_argument("--config", type=Path, default=Path("configs/live_play.yaml"))
    parser.add_argument("--armed", action="store_true", help="Send clicks/taps to the configured target.")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Capture one screenshot, print match/ready detection values for it, save it, and exit.",
    )
    parser.add_argument("--diagnose-out", type=Path, default=Path("live_diagnostic.png"))
    args = parser.parse_args()
    config = load_live_config(args.config)
    device: LiveDevice
    if config.transport == "desktop":
        device = WindowsDesktopDevice(config.desktop_capture, config.window_title)
    else:
        device = ADBDevice(config.adb_path or "adb", config.device_serial)
    if args.diagnose:
        diagnose(config, device, args.diagnose_out)
        return
    runner = LiveMatchRunner(config, device, armed=args.armed)
    runner.run_forever()


if __name__ == "__main__":
    main()
