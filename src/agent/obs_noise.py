"""Domain randomization for vision-facing observations (#32).

A policy trained on exact simulator coordinates falls over on real frames:
detections arrive jittered, late, occasionally missing, occasionally
hallucinated, and sometimes with the wrong card attached. This module
degrades the *observation* to look like the output of `src.live.vision`
before it is rasterized into the spatial grid (or packed into the set
encoder's entity list).

Two design rules that are easy to get wrong:

**Randomize enemy entities only.** Own units, own elixir, and own hand come
from the player's own UI and the deterministic hand-cycle tracker, which are
near-perfectly observable. Own *towers* and enemy *towers* are static, huge,
and always on screen, so they are not perturbed either. Injecting noise into
things that are actually reliable teaches the policy to distrust them and
makes it needlessly timid.

**Perturb in tile space, before binning.** Jitter is applied to the
`UnitView` coordinates that `obs_layout.encode_spatial` then bins, so a
half-tile error moves a unit across a cell boundary exactly as often as it
would in reality. Smearing the finished 9x16 grid instead would model a
completely different (and non-physical) error process.

**Training only.** `CRBattleEnv` takes this via `obs_noise=`; evaluation and
the frozen benchmark leave it None, otherwise benchmark win rates stop being
comparable across runs.

**Calibrate, do not guess.** The defaults here are placeholders. Once #34
produces real detections, measure `p_miss` and positional error on recorded
frames and fit these numbers to them — noise that is qualitatively wrong
(e.g. a systematic homography bias modelled as zero-mean jitter) transfers
worse than no randomization at all.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from src.agent.obs_layout import UnitView
from src.simulator.constants import CardType, Side

# Look-alikes a real detector genuinely mixes up: same silhouette, same
# footprint, often the same swarm count. Symmetric — registered both ways at
# import time.
_CONFUSION_PAIRS = (
    ("skeletons", "goblins"),
    ("minions", "bats"),
    ("knight", "valkyrie"),
    ("archers", "spear_goblins"),
    ("giant", "golem"),
    ("hog_rider", "ram_rider"),
    ("mini_pekka", "prince"),
    ("wizard", "electro_wizard"),
    ("musketeer", "dart_goblin"),
    ("baby_dragon", "inferno_dragon"),
    ("barbarians", "elite_barbarians"),
    ("cannon", "tesla"),
)

SIMILAR_CARDS: dict[str, tuple[str, ...]] = {}
for _a, _b in _CONFUSION_PAIRS:
    SIMILAR_CARDS.setdefault(_a, ())
    SIMILAR_CARDS.setdefault(_b, ())
    SIMILAR_CARDS[_a] += (_b,)
    SIMILAR_CARDS[_b] += (_a,)


@dataclass(frozen=True)
class ObsNoiseConfig:
    """Per-frame detection error rates. All probabilities are per entity
    except `p_stale` and `p_occlusion`, which are per frame."""

    enabled: bool = False
    jitter_tiles: float = 0.0          # Gaussian sigma of positional error, in tiles
    p_miss: float = 0.0                # enemy entity not detected this frame
    p_false_positive: float = 0.0      # chance of injecting one phantom enemy
    max_false_positives: int = 2
    p_identity_confusion: float = 0.0  # card swapped for a look-alike
    p_occlusion: float = 0.0           # chance a whole arena patch is occluded
    occlusion_radius: float = 3.0      # tiles
    p_stale: float = 0.0               # reuse previous frame's enemy detections
    hp_error_frac: float = 0.0         # relative error reading the health bar

    @classmethod
    def from_dict(cls, raw: dict | None) -> "ObsNoiseConfig":
        """Build from a training-yaml `obs_noise:` block.

        A present-but-silent block means "on with these values" — writing out
        rates and then having them ignored because `enabled` was omitted is a
        failure mode that costs a whole training run to notice.
        """
        if not raw:
            return cls()
        defaults = {f.name: f.default for f in fields(cls)}
        unknown = set(raw) - set(defaults)
        if unknown:
            raise ValueError(f"unknown obs_noise keys: {sorted(unknown)}")
        kwargs = {key: type(defaults[key])(value)
                  for key, value in raw.items() if key != "enabled"}
        return cls(enabled=bool(raw.get("enabled", True)), **kwargs)


class ObservationNoise:
    """Stateful per-env degrader. Hold one per env; it carries the previous
    frame's detections for the staleness model."""

    def __init__(self, config: ObsNoiseConfig, seed: int | None = None,
                 rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self._last: dict[Side, list[UnitView]] = {}

    def reset(self) -> None:
        self._last.clear()

    # ------------------------------------------------------------- helpers

    def _similar_card(self, view: UnitView, engine) -> str:
        """A card this one plausibly gets misread as."""
        explicit = SIMILAR_CARDS.get(view.card)
        if explicit:
            known = [c for c in explicit if c in engine.cards]
            if known:
                return str(self.rng.choice(known))
        stats = engine.cards.get(view.card)
        if stats is None:
            return view.card
        pool = [name for name, c in engine.cards.items()
                if name != view.card and c.type == stats.type
                and c.flying == stats.flying and abs(c.cost - stats.cost) <= 1]
        return str(self.rng.choice(pool)) if pool else view.card

    def _phantom(self, engine, side: Side, seen: list[UnitView]) -> UnitView:
        """A hallucinated enemy at a plausible spot.

        Anchored to a real detection when one exists — false positives in
        practice are duplicated/split blobs off an actual unit far more often
        than they are units conjured out of empty grass.
        """
        a = engine.arena
        anchor = seen[self.rng.integers(len(seen))] if seen else None
        if anchor is not None:
            x = anchor.x + float(self.rng.normal(0.0, 1.5))
            y = anchor.y + float(self.rng.normal(0.0, 1.5))
            card, max_hp = anchor.card, anchor.max_hp
        else:
            x = float(self.rng.uniform(0.0, a.width))
            # Enemy half in the acting player's frame.
            y = (float(self.rng.uniform(a.river_y_max, a.height)) if side == Side.BOTTOM
                 else float(self.rng.uniform(0.0, a.river_y_min)))
            troops = [n for n, c in engine.cards.items() if c.type == CardType.TROOP]
            card = str(self.rng.choice(troops)) if troops else "knight"
            max_hp = max(engine.cards[card].hp, 1.0) if card in engine.cards else 1000.0
        return UnitView(
            card=card,
            x=min(max(x, 0.0), a.width - 0.01),
            y=min(max(y, 0.0), a.height - 0.01),
            hp=max_hp * float(self.rng.uniform(0.4, 1.0)),
            max_hp=max_hp,
            friendly=False,
        )

    # ---------------------------------------------------------------- main

    def perturb(self, views: list[UnitView], engine, side: Side) -> list[UnitView]:
        """Return a degraded copy of `views`. Friendly entities and all
        towers pass through untouched."""
        cfg = self.config
        if not cfg.enabled:
            return views

        keep = [v for v in views if v.friendly or v.is_tower]
        enemies = [v for v in views if not v.friendly and not v.is_tower]

        if cfg.p_stale > 0 and side in self._last and self.rng.random() < cfg.p_stale:
            # Capture/inference lag: the enemy half of the frame is one
            # decision old while own state is current.
            return keep + list(self._last[side])

        if cfg.p_occlusion > 0 and enemies and self.rng.random() < cfg.p_occlusion:
            centre = enemies[self.rng.integers(len(enemies))]
            r = cfg.occlusion_radius
            enemies = [v for v in enemies
                       if np.hypot(v.x - centre.x, v.y - centre.y) > r]

        a = engine.arena
        out: list[UnitView] = []
        for v in enemies:
            if cfg.p_miss > 0 and self.rng.random() < cfg.p_miss:
                continue
            x, y = v.x, v.y
            if cfg.jitter_tiles > 0:
                x += float(self.rng.normal(0.0, cfg.jitter_tiles))
                y += float(self.rng.normal(0.0, cfg.jitter_tiles))
            hp = v.hp
            if cfg.hp_error_frac > 0:
                hp *= 1.0 + float(self.rng.normal(0.0, cfg.hp_error_frac))
            card = v.card
            if cfg.p_identity_confusion > 0 and self.rng.random() < cfg.p_identity_confusion:
                card = self._similar_card(v, engine)
            out.append(UnitView(
                card=card,
                x=min(max(x, 0.0), a.width - 0.01),
                y=min(max(y, 0.0), a.height - 0.01),
                hp=min(max(hp, 1.0), v.max_hp),
                max_hp=v.max_hp,
                friendly=False,
                is_building=v.is_building,
                is_tower=False,
                flying=v.flying,
                deploying=v.deploying,
                frozen=v.frozen,
                shielded=v.shielded,
            ))

        for _ in range(cfg.max_false_positives):
            if cfg.p_false_positive <= 0 or self.rng.random() >= cfg.p_false_positive:
                break
            out.append(self._phantom(engine, side, out))

        self._last[side] = out
        return keep + out
