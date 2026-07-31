"""BattleEngine: the deterministic core simulation.

Reward-agnostic: `tick()` returns an event list; callers (env, metrics)
interpret it. Symmetric API — both sides deploy through the same methods —
so a different driver (PettingZoo shim, future real-client adapter) can wrap
it without changes.
"""
from __future__ import annotations

import numpy as np

from src.simulator import (
    collision,
    combat,
    heroes,
    mechanics,
    movement,
    spell_effects,
    targeting,
)
from src.simulator.cards import ArenaConfig, CardStats, load_cards
from src.simulator.constants import TIME_EPS, CardType, MatchResult, Side
from src.simulator.entities import PendingSpell, Tower, Unit
from src.simulator.player import PlayerState

# Deterministic spawn offsets for swarm deploys, indexed by unit count.
_SWARM_OFFSETS = {
    1: [(0.0, 0.0)],
    2: [(-0.5, 0.0), (0.5, 0.0)],
    3: [(0.0, 0.5), (-0.5, -0.4), (0.5, -0.4)],
}


class BattleEngine:
    def __init__(
        self,
        deck_bottom: list[CardStats],
        deck_top: list[CardStats],
        arena: ArenaConfig,
        seed: int | np.random.Generator = 0,
        lanes: str = "both",              # "both" | "right" (curriculum stage 0)
        regulation: float | None = None,  # override match length (stage 0)
        cards: dict[str, CardStats] | None = None,
    ):
        self.arena = arena
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.lanes = lanes
        self.regulation = regulation if regulation is not None else arena.regulation

        self.players = {
            side: PlayerState(deck, self.rng, arena.elixir_start, arena.elixir_max,
                              arena.elixir_regen_interval)
            for side, deck in ((Side.BOTTOM, deck_bottom), (Side.TOP, deck_top))
        }
        self.time = 0.0
        self.result = MatchResult.ONGOING
        self.units: list[Unit] = []
        self.spells: list[PendingSpell] = []
        self.towers: list[Tower] = []
        self._next_id = 1
        self._by_id: dict[int, Unit | Tower] = {}
        # The table used for *derived* spawns and lookups: death-spawns,
        # hero summons, spell stun durations. It must be the same table the
        # decks came from, or a level-scaled run would spawn level-1 Lava
        # Pups behind a level-13 Lava Hound.
        self.cards = cards if cards is not None else load_cards()
        self._build_towers()

    # ------------------------------------------------------------- setup

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _build_towers(self) -> None:
        a = self.arena
        specs = [("princess_right", a.princess_right_pos, a.princess)]
        if self.lanes == "both":
            specs.insert(0, ("princess_left", a.princess_left_pos, a.princess))
        specs.append(("king", a.king_pos, a.king))
        for side in (Side.BOTTOM, Side.TOP):
            for kind, (x, y), stats in specs:
                if side == Side.TOP:
                    x, y = self.mirror(x, y)
                t = Tower(id=self._new_id(), kind=kind, side=side, x=x, y=y,
                          hp=stats.hp, stats=stats, activated=(kind != "king"))
                self.towers.append(t)
                self._by_id[t.id] = t

    def mirror(self, x: float, y: float) -> tuple[float, float]:
        # y-only: lanes stay aligned (both sides' left princess share an x).
        return x, self.arena.height - y

    # ------------------------------------------------------------- queries

    @property
    def overtime(self) -> bool:
        return self.time >= self.regulation and self.result == MatchResult.ONGOING

    @property
    def double_elixir(self) -> bool:
        return self.time >= self.arena.double_time or self.time >= self.regulation

    def towers_of(self, side: Side, alive: bool = True) -> list[Tower]:
        return [t for t in self.towers if t.side == side and (t.hp > 0 or not alive)]

    def units_of(self, side: Side) -> list[Unit]:
        return [u for u in self.units if u.side == side and u.hp > 0]

    def crowns(self, side: Side) -> int:
        """Enemy towers this side has destroyed."""
        return sum(1 for t in self.towers if t.side == side.other and t.hp <= 0)

    def pocket_unlocked(self, side: Side, lane: str) -> bool:
        """A lane's enemy pocket opens when that enemy princess tower falls."""
        kind = f"princess_{lane}"
        return any(t.side == side.other and t.kind == kind and t.hp <= 0 for t in self.towers)

    def legal_deploy(self, side: Side, card: CardStats, x: float, y: float) -> bool:
        a = self.arena
        if not (0 <= x < a.width and 0 <= y < a.height):
            return False
        if self.lanes == "right" and x < a.width / 2:
            return False  # single-lane curriculum: left half closed entirely
        if card.type == CardType.SPELL:
            return True
        if card.deploy_anywhere:
            # Miner/Goblin Drill tunnel: the whole board is legal, which is
            # the entire reason those cards exist.
            return True
        # Perspective: own half for BOTTOM is below the river, for TOP above.
        # Closed band so the geometry is exactly mirror-symmetric (15 <-> 17).
        if a.river_y_min <= y <= a.river_y_max:
            return False
        own_half = y < a.river_y_min if side == Side.BOTTOM else y > a.river_y_max
        if own_half:
            return True
        if card.type == CardType.BUILDING:
            return False
        # Troop in enemy territory: only in an unlocked pocket, up to the
        # enemy princess-tower line.
        lane = "left" if x < a.width / 2 else "right"
        if self.lanes == "right" and lane == "left":
            return False
        if not self.pocket_unlocked(side, lane):
            return False
        princess_y = a.princess_left_pos[1]
        limit = a.height - princess_y  # enemy princess row from bottom's view
        return y <= limit if side == Side.BOTTOM else y >= a.height - limit

    # ------------------------------------------------------------- actions

    def play_card(self, side: Side, slot: int, x: float, y: float) -> list[dict]:
        """Deploy the card in hand slot at (x, y). Raises on illegal action."""
        player = self.players[side]
        card = player.hand[slot]
        if not self.legal_deploy(side, card, x, y):
            raise ValueError(f"illegal deploy: {card.name} at ({x:.1f},{y:.1f}) for {side.name}")
        card = player.play(slot)  # validates elixir
        events: list[dict] = [{"type": "deploy", "side": side, "card": card.name,
                               "cost": card.cost, "x": x, "y": y}]
        if card.type == CardType.SPELL:
            ticks = spell_effects.total_ticks(card.name)
            zone_until = (self.time + card.spell_delay + spell_effects.RAGE_DURATION
                          if card.name == "rage" else 0.0)
            self.spells.append(PendingSpell(
                side=side, x=x, y=y, radius=card.spell_radius, damage=card.spell_damage,
                tower_multiplier=card.tower_multiplier,
                resolve_at=self.time + card.spell_delay, card_name=card.name,
                ticks_left=ticks, zone_until=zone_until))
            return events
        spawned = self.spawn_units(card, side, x, y, card.count)
        for u in spawned:
            events.append({"type": "spawn", "side": side, "card": card.name, "unit_id": u.id})
        return events

    def spawn_units(
        self,
        card: CardStats,
        side: Side,
        x: float,
        y: float,
        count: int | None = None,
        hp_override: float | None = None,
    ) -> list[Unit]:
        """Spawn troops/buildings at (x, y). Used by deploy, hero abilities,
        and Clone (`hp_override` gives Clone's fragile 1-HP copies)."""
        count = count if count is not None else card.count
        offsets = _SWARM_OFFSETS.get(count) or [
            (float(dx), float(dy)) for dx, dy in self.rng.normal(0, 0.5, (count, 2))
        ]
        expires = self.time + card.lifetime if card.type == CardType.BUILDING else 0.0
        spawned: list[Unit] = []
        for dx, dy in offsets:
            u = Unit(
                id=self._new_id(),
                stats=card,
                side=side,
                x=min(max(x + dx, 0.0), self.arena.width - 0.01),
                y=min(max(y + dy, 0.0), self.arena.height - 0.01),
                hp=hp_override if hp_override is not None else card.hp,
                cooldown=card.hit_speed,
                elixir_value=card.cost / max(count, 1),
                expires_at=expires,
                deployed_at=self.time,
            )
            self.units.append(u)
            self._by_id[u.id] = u
            spawned.append(u)
        return spawned

    # ------------------------------------------------------------- tick

    def tick(self) -> list[dict]:
        """Advance one dt. Returns this tick's events."""
        if self.result != MatchResult.ONGOING:
            return []
        events: list[dict] = []
        dt = self.arena.dt

        # 1. Elixir
        for side, p in self.players.items():
            leak = p.regen(dt, self.double_elixir)
            if leak > 0:
                events.append({"type": "leak", "side": side, "amount": leak})

        # 2. Spells due
        due = [s for s in self.spells if s.resolve_at <= self.time + TIME_EPS]
        self.spells = [s for s in self.spells if s.resolve_at > self.time + TIME_EPS]
        for s in due:
            if s.card_name == "rage":
                # Self-buff: targets the caster's own units, deals no damage.
                spell_effects.apply_rage(s, self, self.units_of(s.side))
                spell_effects.reschedule(s, self)
                continue
            if s.card_name == "clone":
                # Self-buff too: duplicates the caster's own units, no damage.
                spell_effects.apply_clone(s, self, self.units_of(s.side))
                continue
            # Damage lands once per application for poison (that's the point
            # of its ticks) but only on the first for the other persistent
            # spells, whose later ticks carry the *effect*, not the hit.
            if spell_effects.damages_this_tick(s):
                combat.resolve_spell(s, self.units_of(s.side.other),
                                     self.towers_of(s.side.other), events)
            if s.card_name == "freeze":
                spell_effects.apply_freeze(s, self, self.units_of(s.side.other))
            elif s.card_name == "tornado":
                spell_effects.apply_tornado(s, self, self.units_of(s.side.other))
            # Zap/Lightning-style stuns reset charge, inferno ramp, and the
            # attack wind-up; see mechanics.on_stun.
            stats = self.cards.get(s.card_name)
            if stats is not None and stats.stun_duration > 0:
                mechanics.apply_stun(self, self.units_of(s.side.other),
                                     s.x, s.y, s.radius, stats.stun_duration)
            spell_effects.reschedule(s, self)

        # 3. Tower attacks
        for t in self.towers:
            if t.hp <= 0 or not t.activated:
                continue
            t.cooldown = max(0.0, t.cooldown - dt)
            tgt = self._by_id.get(t.target_id) if t.target_id else None
            if not targeting.tower_keeps_lock(t, tgt):
                t.target_id = targeting.tower_acquire(t, self.units_of(t.side.other))
                tgt = self._by_id.get(t.target_id) if t.target_id else None
            if tgt is not None and t.cooldown <= TIME_EPS:
                combat.apply_attack(t, tgt, self.units_of(t.side.other), events)
                t.cooldown = t.stats.hit_speed

        # 4. Unit targeting
        for u in self.units:
            if u.hp <= 0 or self._is_inactive(u):
                continue
            tgt = self._by_id.get(u.target_id) if u.target_id else None
            if not targeting.unit_keeps_lock(u, tgt):
                u.target_id = targeting.unit_acquire(
                    u, self.units_of(u.side.other), self.towers_of(u.side.other))

        # 5. Movement
        for u in self.units:
            if u.hp <= 0 or u.is_building or self._is_inactive(u):
                continue
            tgt = self._by_id.get(u.target_id) if u.target_id else None
            if tgt is None:
                continue
            reach = u.stats.range + u.radius + tgt.radius
            if targeting.dist(u.x, u.y, tgt.x, tgt.y) > reach:
                wx, wy = movement.waypoint_toward(u, tgt.x, tgt.y, self.arena)
                nx, ny = movement.next_position(
                    u, wx, wy, dt, extra_speed=mechanics.charge_speed(u))
                nx, ny = movement.avoid_obstacle(u, nx, ny, self.units, self.arena)
                if collision.blocked(u, nx, ny, self.units):
                    # Body-blocked: a banked charge is lost, which is what
                    # makes a cheap blocker a real answer to a Prince.
                    mechanics.break_charge(u)
                else:
                    moved = targeting.dist(u.x, u.y, nx, ny)
                    u.x, u.y = nx, ny
                    mechanics.advance_charge(u, moved)

        # 6. Unit attacks
        for u in self.units:
            if u.hp <= 0 or self._is_inactive(u):
                continue
            u.cooldown = max(0.0, u.cooldown - dt)
            tgt = self._by_id.get(u.target_id) if u.target_id else None
            mechanics.advance_ramp(u, u.target_id if tgt is not None else None, dt)
            if tgt is None or getattr(tgt, "hp", 0) <= 0 or u.cooldown > TIME_EPS:
                continue
            reach = u.stats.range + u.radius + tgt.radius
            if targeting.dist(u.x, u.y, tgt.x, tgt.y) <= reach:
                combat.apply_attack(u, tgt, self.units_of(u.side.other), events,
                                    damage_scale=mechanics.attack_scale(u))
                mechanics.on_attack(u)
                u.cooldown = u.stats.hit_speed

        # 6b. Hero abilities
        heroes.tick_hero_abilities(self, dt, events)

        # 7. Cleanup: deaths, building expiry, king activation
        for ev in events:
            if ev["type"] == "tower_fall" and ev["kind"] != "king":
                self._king_of(ev["side"]).activated = True
        for t in self.towers:
            if t.is_king and t.hp < t.stats.hp:
                t.activated = True
        gone = [u for u in self.units
                if u.hp <= 0 or (u.expires_at and self.time + TIME_EPS >= u.expires_at)]
        for u in gone:
            if u.hp <= 0:
                heroes.on_unit_death(u, self, events)
                mechanics.on_unit_death(u, self, events)
            self._by_id.pop(u.id, None)
        self.units = [u for u in self.units if u not in gone]

        # 8. Clock + win check
        self.time += dt
        self._check_result(events)
        return events

    def _king_of(self, side: Side) -> Tower:
        return next(t for t in self.towers if t.side == side and t.is_king)

    def _is_frozen(self, u: Unit) -> bool:
        return bool(u.frozen_until) and self.time < u.frozen_until - TIME_EPS

    def _is_deploying(self, u: Unit) -> bool:
        return self.time < u.deployed_at + u.stats.deploy_time - TIME_EPS

    def _is_inactive(self, u: Unit) -> bool:
        return self._is_frozen(u) or self._is_deploying(u)

    def _check_result(self, events: list[dict]) -> None:
        bottom_king_dead = self._king_of(Side.BOTTOM).hp <= 0
        top_king_dead = self._king_of(Side.TOP).hp <= 0
        if bottom_king_dead and top_king_dead:
            self.result = MatchResult.DRAW
            return
        if top_king_dead:
            self.result = MatchResult.BOTTOM_WIN
            return
        if bottom_king_dead:
            self.result = MatchResult.TOP_WIN
            return

        in_overtime = self.time >= self.regulation
        if in_overtime and any(e["type"] == "tower_fall" for e in events):
            # Sudden death: a tower fell this tick. Simultaneous falls on both
            # sides cancel out; compare totals.
            cb, ct = self.crowns(Side.BOTTOM), self.crowns(Side.TOP)
            if cb != ct:
                self.result = MatchResult.BOTTOM_WIN if cb > ct else MatchResult.TOP_WIN
            return

        if abs(self.time - self.regulation) < self.arena.dt / 2:
            # Regulation just ended: more crowns wins, tie goes to overtime.
            cb, ct = self.crowns(Side.BOTTOM), self.crowns(Side.TOP)
            if cb != ct:
                self.result = MatchResult.BOTTOM_WIN if cb > ct else MatchResult.TOP_WIN
            return

        if self.time >= self.regulation + self.arena.overtime:
            self._tiebreak()

    def _tiebreak(self) -> None:
        cb, ct = self.crowns(Side.BOTTOM), self.crowns(Side.TOP)
        if cb != ct:
            self.result = MatchResult.BOTTOM_WIN if cb > ct else MatchResult.TOP_WIN
            return
        if self.arena.tiebreaker == "lowest_tower_hp":
            def key(side: Side) -> tuple[float, float]:
                alive = self.towers_of(side)
                return (min(t.hp for t in alive), sum(t.hp for t in alive))
            kb, kt = key(Side.BOTTOM), key(Side.TOP)
            if kb != kt:
                self.result = MatchResult.BOTTOM_WIN if kb > kt else MatchResult.TOP_WIN
                return
        self.result = MatchResult.DRAW
