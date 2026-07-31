# Simulator dynamics fidelity audit (#36)

Once the policy can *see* the board (#31/#34), the binding constraint on
real-match strength stops being perception and becomes whether the simulated
dynamics match real Clash Royale. A policy that is superhuman against a
simulator with the wrong physics is superhuman at the wrong game.

## The filter used here

Not "is this different from the real game" — almost everything is, somewhere
in the fourth decimal place. The question asked of every candidate was:

> **Does this change which card a good player would choose?**

Charge mechanics do: a Skeleton in front of a Prince is a good trade only
because it eats the charged hit. Inferno ramp-up does: it is the reason you
do not tank with a Golem into an unzapped Inferno. A 3% stat difference does
not.

Items are grouped by whether they were fixed, and the ones that were not are
listed with the reason, because "we looked and decided no" is much more
useful to the next person than silence.

---

## Fixed

### Charge — Prince, Dark Prince, Battle Ram, Ram Rider, Bandit, Royal Ghost

Moving `charge_distance` tiles unimpeded makes the unit faster
(`charge_speed_multiplier`) and its next hit far harder
(`charge_damage_multiplier`). Connecting spends the charge; being
body-blocked or stunned resets it.

*Why it matters:* without charge, the entire "cheap block" layer of defence
is worth nothing and the policy has no gradient toward learning it. A Prince
was previously just an expensive 200-damage troop.

Implementation: `src/simulator/mechanics.py` (`advance_charge`,
`break_charge`, `charge_speed`, `attack_scale`, `on_attack`), driven from the
movement and attack phases of `BattleEngine.tick`.

### Inferno ramp-up — Inferno Tower, Inferno Dragon

Damage grows from 1x to `ramp_up_multiplier` over `ramp_up_time` seconds
locked on one target, and resets on retarget or stun.

*Why it matters:* this single mechanic is the reason swarms counter Inferno,
the reason Zap/Lightning on an Inferno is a real play, and the reason
tanks respect it at all. Without it, Inferno was a weak single-target tower.

### Stun resets wind-ups — Zap, Lightning, Freeze

Stuns now reset charge, inferno ramp, **and the attack cooldown**. That last
one is why a 0.5-second Zap costs a Sparky its whole 4-second wind-up rather
than half a second of it.

Modelled generically via `CardStats.stun_duration` and `mechanics.on_stun`,
rather than special-casing Sparky — every current and future card with a
wind-up gets the interaction for free.

### Death effects — Lava Hound, Golem, Balloon, Giant Skeleton

`death_spawn` / `death_spawn_count` and `death_damage` /
`death_damage_radius`. Two new non-playable cards (`lava_pup`, `golemite`)
exist purely as death products.

*Why it matters:* a card whose value is mostly posthumous is priced
completely wrong without it. A Lava Hound with no pups is a bad 7-elixir
tank; a Balloon with no death damage is much easier to defend than the real
one. Death damage hits only enemies of the dying unit, so a death bomb never
clips its own push.

### Tunnelling — Miner, Goblin Drill

`deploy_anywhere` bypasses the own-half/pocket placement rules. This is the
entire reason those cards exist, and it changes the action mask, so it
affects the policy directly rather than only the simulation.

### Pathing — steer around bodies

Straight-line movement plus reject-on-overlap meant a unit that walked into
the side of a tank simply stopped: its waypoint never changed, so it
re-proposed the same blocked step forever. `movement.avoid_obstacle` tries a
few rotations of the step vector and takes the first clear one, with ground
units forbidden from sidestepping into the river.

*Why it matters:* funnelling a push into one lane and sheltering support
behind a tank are deliberate plays whose payoff depends on units flowing
around each other rather than deadlocking.

### Tornado pulls continuously; Rage is a persistent zone

Tornado now drags units a share of its total pull per tick across its
duration, instead of teleporting them once — the difference between "a pull"
and "an activation into king-tower range". Rage re-applies over its duration
so units entering the zone later are caught, and the buff expires when the
*zone* does rather than being pushed out by each re-application.

Both are driven by `spell_effects.PERSISTENT_TICKS`, the single table of how
many applications a persistent spell gets and how far apart. Spells built by
hand with `ticks_left = 0` still apply their whole effect at once, so
engine-level tests that construct them directly keep working.

---

## Examined and deliberately not changed

### Card stats — the header comment is wrong, and rescaling is the wrong fix

`configs/cards.yaml` claims its values are "roughly half of real Clash
Royale values". Checked against Supercell's card table (via RoyaleAPI's
[`cr-api-data`](https://github.com/RoyaleAPI/cr-api-data) mirror), that is
not true, in a way that matters:

- **The scale is already exact.** `configs/arena.yaml`'s towers match the
  datamined values to the digit — princess 1400 HP / 50 damage / 0.8s /
  7.5 range, king 2400 / 50 / 1.0 / 7.0 — and Knight (690/79), Skeletons
  (32/32), Tesla (450/90), Fireball (325) and Rocket (700) are exactly
  right too. The file was clearly *started* from level-1 datamined values.
- **Rescaling would therefore fix nothing.** Breakpoints are ratios;
  multiplying every HP and damage by a constant leaves every one of them
  where it was. Clash Royale scales all troops on one per-level curve, so
  level-1 values already preserve every breakpoint the real game has.
- **The actual defect is per-card drift.** Only 3 of 94 cards match on both
  HP and damage. The rest were filled in by estimate and range from -74%
  (Rascals HP) to +277% (Hunter damage). No scale factor produces that; only
  a per-card correction from a real source does.

**Tooling, not guesses:** `scripts/sync_card_stats.py` diffs `cards.yaml`
against a vendored, reviewable distillation of that table
(`configs/reference_card_stats.json`, regenerate with `--refresh`) and can
rewrite it with `--apply`.

It deliberately splits the fields:

- **auto-syncable** (`hp`, `damage`, `hit_speed`, `speed`, `count`,
  `splash_radius`, `spell_radius`, `tower_multiplier`) — resolvable to one
  unambiguous upstream number, and the fields breakpoints depend on.
- **needs review** (`range`, `sight_range`) — upstream measures to the
  target's edge; this simulator adds both body radii at reach time. Not the
  same quantity, so copying would change every engagement distance.
- **never synced** — the effect-spells this simulator implements explicitly
  (rage, clone, graveyard, poison, tornado, freeze, earthquake), thirteen
  cards whose upstream row describes a different entity than ours (spawner
  buildings, delivery spells, Rascals, Princess), and a handful of
  per-field opt-outs where one number does not map but the rest do.

### Following the projectile link

Upstream splits an attack across a character row and the *projectile* it
fires, which is why `area_damage_radius` reads 0 for Wizard, Bowler and
Bomber. `distil()` follows that link, gated on the projectile's
`aoe_to_ground` / `aoe_to_air` flags so a single-target projectile's own
body radius is never mistaken for a blast — without that check, Musketeer
and Princess would acquire splash they do not have.

That converted most of the review backlog into exact values, and turned up
mechanics this simulator was missing outright: **Witch, Mortar, Bomb Tower,
Wall Breakers, Royal Ghost and Skeleton Dragons all had no splash at all.**
Six more had the wrong radius (Mega Knight 2.5 -> 1.3, Sparky 2.5 -> 1.8,
Executioner 2.0 -> 1.0, Dark Prince 2.0 -> 1.1, Bowler 2.0 -> 1.8, Baby
Dragon 1.5 -> 1.2). Zap and Lightning got their real radii (2.5 and 3.5),
and every spell's crown-tower multiplier is now per-card rather than a flat
0.35 guess (Rocket 0.25, Log 0.20, Fireball/Arrows/Zap/Lightning 0.30).

Three deliberate opt-outs, because the link resolves to something that is
not splash:

- **Electro Dragon** — chain lightning, no upstream radius at all. Syncing
  would silently make it single-target; the 2.0 here is a stand-in.
- **Hunter** — a 10-pellet shotgun (`multiple_projectiles: 10`), each pellet
  0.07 tiles. A spread, not a blast.
- **Magic Archer** — a piercing shot along a line.

And one guard worth keeping: an *absent* `crown_tower_damage_percent` is not
"no reduction". Those rows carry `deflect_behaviour: UseSpellsTowerDamageMul`
— "apply the game-wide spell multiplier", which this table does not contain.
Reporting it as 1.0 would have handed Barbarian Barrel and Royal Delivery
full tower damage, so a missing value is treated as unknown and skipped.

**Applied 2026-07-31.** The 211 auto-syncable stat lines were written; the
tool now reports zero auto-syncable differences remaining. The ~93 review
fields were left alone and are still outstanding — see "What to do next".

Two things the apply run surfaced, kept here because they will recur on the
next sync:

- **Zeros are mapping failures, not data.** `goblin_barrel` and `goblin_cage`
  came through with `damage: 0`, because upstream models them as
  delivery/spawner rows whose payload lives on another object.
  `CardStats.__post_init__` rejected the file on load, which is how it was
  caught. The tool now refuses to write a zero into `hp`, `damage`,
  `spell_damage`, `hit_speed` or `count` and reports it instead.
- **Match pacing changed materially.** Troops got substantially tankier
  (Sparky 600 -> 1200 HP, Mega Knight 2000 -> 3300, Golem 2600 -> 3200), and
  bot-vs-bot matches now run the full 180s regulation where most previously
  ended early on a king-tower kill. That is plausible for real Clash Royale,
  where most ladder games do go to time, but it roughly tripled test-suite
  wall clock (325s -> 940s) and will slow training rollouts by a similar
  factor. Worth coordinating with the throughput work (#24/#25).

Every benchmark number recorded before this date is measured against
different physics and is not comparable. `scripts/rebenchmark.py` writes a
fresh baseline (`artifacts/benchmark_baseline.json`); do that before reading
anything into a win rate.

### Targeting quirks — aggro ranges, retarget rules, first-target preference

`targeting.py` approximates these with nearest-legal-in-sight plus a
1.5x-sight lock break. Real CR has per-card aggro ranges and more specific
retarget rules. This *does* change play at the margin (kiting a Pekka with a
building), but the current model already produces the qualitative behaviour,
and the remaining gap is much smaller than the ones fixed above. Worth
revisiting after a real-match error analysis rather than speculatively.

### Electro-family on-hit stuns — Electro Wizard, Electro Dragon, Zappies

Their stun is applied on *hit* rather than by a spell, which needs a
different hook than `stun_duration` (a spell-side field). The machinery
(`mechanics.on_stun`) is already generic, so this is a small follow-up:
add an on-hit stun field and call `on_stun` from `combat.apply_attack`.
Left out here to keep the change reviewable.

### Turn radii and unit acceleration

Real units do not change direction instantly. Modelling it would mostly
affect how sharply a unit rounds a bridge. Low decision-relevance; high
implementation cost. Skipped.

### Clone's copies get the normal deploy delay

Pre-existing simplification, documented in `spell_effects.py`. Clone is rare
and the delay is ~1s. Skipped.

---

## What to do next

1. **The remaining `range` / `sight_range` fields** (~64). These are the last
   category still held back, and for a real reason: upstream measures range
   to the target's *edge*, while `BattleEngine.tick` computes
   `stats.range + attacker.radius + target.radius`. The two are not the same
   quantity, so a blind copy would change every engagement distance in the
   game. Settling it means deciding the convention once and converting, not
   copying.
2. **Electro-family on-hit stuns** — smallest remaining *mechanics* fix with
   real decision relevance, and the hook already exists.
3. **Real-match error analysis** — once the live bridge runs a `human`-tier
   policy end to end, log the states where sim-predicted and observed
   outcomes diverge. That measurement should drive the next round of this
   list instead of intuition; every item above was prioritised by argument,
   not by data.

Tests: `tests/test_mechanics.py`.
