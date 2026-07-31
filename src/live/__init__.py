"""Live-match adapters for a user-controlled Clash Royale session.

The simulator remains the source of truth for training.  This package bridges
an already-running match to a calibrated Windows desktop or Android device.
"""

from src.live.config import LiveConfig, load_live_config
from src.live.runner import LiveMatchRunner

__all__ = ["LiveConfig", "LiveMatchRunner", "load_live_config"]
