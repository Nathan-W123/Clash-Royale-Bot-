"""ASCII renderer for debugging and match watching."""
from __future__ import annotations

from src.simulator.constants import Side
from src.simulator.engine import BattleEngine


def render_ascii(engine: BattleEngine) -> str:
    a = engine.arena
    w, h = int(a.width), int(a.height)
    grid = [["." for _ in range(w)] for _ in range(h)]
    for y in range(h):
        if a.river_y_min <= y < a.river_y_max:
            for x in range(w):
                on_bridge = any(abs(x + 0.5 - bx) <= a.bridge_half_width for bx in a.bridge_xs)
                grid[y][x] = "=" if on_bridge else "~"
    for t in engine.towers:
        if t.hp <= 0:
            continue
        ch = "K" if t.is_king else "T"
        if t.side == Side.TOP:
            ch = ch.lower()
        grid[int(t.y)][int(t.x)] = ch
    for u in engine.units:
        if u.hp <= 0:
            continue
        ch = u.stats.name[0].upper() if u.side == Side.BOTTOM else u.stats.name[0].lower()
        grid[min(int(u.y), h - 1)][min(int(u.x), w - 1)] = ch
    # Top of arena printed first (TOP player at top of screen).
    rows = ["".join(r) for r in reversed(grid)]
    pb, pt = engine.players[Side.BOTTOM], engine.players[Side.TOP]
    header = (f"t={engine.time:6.1f}s  result={engine.result.name}  "
              f"elixir B={pb.elixir:4.1f} T={pt.elixir:4.1f}  "
              f"crowns B={engine.crowns(Side.BOTTOM)} T={engine.crowns(Side.TOP)}")
    return header + "\n" + "\n".join(rows)
