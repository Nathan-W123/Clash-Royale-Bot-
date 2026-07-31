# Handoff spec: tasks #31–#41

Written for an agent picking this up cold. Assumes no context from the audit
session. Read `CLAUDE.md` first — especially "On-Screen Visual Perception"
and "Out of Scope", which define what information the agent is allowed to
consume and why.

**Owned elsewhere:** #24–#30 (throughput, asymmetric critic, recurrence,
potential-based shaping, league exploiters, re-benchmark). Do not edit
`src/simulator/vec_env.py`, `src/agent/ppo.py`, or `src/agent/rewards.py`
without coordinating — those are actively being changed.

---

## Ground truth you need before touching anything

**Observation tiers.** `src/agent/obs_layout.py` currently has two, selected
by a `use_spatial` bool threaded through `NetworkConfig`, `CRBattleEnv`, and
the training configs:

| Tier | `use_spatial` | Spatial grid | Own elixir/hand | Opponent elixir | Legal live? |
|---|---|---|---|---|---|
| `full` | `True` | real | yes | **yes** | No — cheats |
| `restricted` | `False` | zero-filled | yes | no | Yes |

`SCALAR_DIM = 17` (full), `RESTRICTED_SCALAR_DIM = 18` (restricted). They
differ because restricted drops opponent elixir but adds per-tower
alive flags. Do not assume restricted ⊂ full.

**Spatial grid shape:** `(SPATIAL_CHANNELS=10, PLACE_ROWS=16, PLACE_COLS=9)`
— the arena downsampled 2×2 tiles per cell. Channels are documented at the
top of `obs_layout.py`. Note it encodes **HP density and presence**, not card
identity — this matters a lot for #34/#35 scoping.

**Coordinate frames.** Arena is 18×32 tiles, `y` increasing away from the
BOTTOM player. Observations are always encoded from the acting player's
perspective via `masking.frame_y`, which mirrors `y` for TOP. Any new
observation code must respect that or the policy silently learns a mirrored
board for one seat.

**Engine facts that bite people:**
- Towers gate on *edge distance* (`targeting.edge_dist`: centre distance
  minus both radii), and tower radius is 1.0. A unit that looks outside a
  tower's 7.5-tile range on paper is often inside it. This has already
  broken several tests; place test units well clear.
- Ground units collide (`src/simulator/collision.py`); flying units do not.
- Units are inert for `CardStats.deploy_time` after spawn (default 1.0s,
  3.5s for mortar/x_bow — see `configs/cards.yaml`).
- `BattleEngine.tick()` is `dt=0.1s`; the agent decides every
  `decision_ticks=5` (0.5s).

**Test conventions.** `tests/conftest.py` has `make_engine`, `spawn_unit`
(back-dates `deployed_at` so injected units are immediately active),
`dummy_stats` (inert punching bag), `force_hand`. Prefer these over
hand-rolling engine state.

---

# Group A: #31–#36 — perception and fidelity

## #31 — `human` observation tier

**Goal.** A third tier: spatial grid **+** own hand/elixir/tower states,
**minus** opponent elixir. This is the correct live-play target — everything
a skilled human perceives and nothing more. It is not a variant of
`restricted`; it is `full` with the cheating scalar removed.

**Why it matters.** `restricted` cannot see enemy troops *at all*, which
makes the environment a severe POMDP and is the leading hypothesis for why
the restricted policy sits at ~10% win rate vs rusher/beatdown while the
full-vision policy reaches 72%.

**Implementation.**
1. Replace the `use_spatial: bool` plumbing with a tier enum/string
   (`"full" | "human" | "restricted"`). Keep `use_spatial` accepted as a
   deprecated alias so existing checkpoints still load —
   `selfplay.load_checkpoint` reads `config` straight out of the `.pt` and
   passes it to `make_network`, so an unknown/missing key must not crash.
2. Add `_encode_scalars_human`: identical to `_encode_scalars_full` minus
   `opp.elixir`, plus the per-tower alive flags that `restricted` has.
   Define `HUMAN_SCALAR_DIM` explicitly; do not compute it by subtraction.
3. `encode_obs(engine, side, card_to_id, tier)` returns the real spatial
   grid for `full` and `human`, zeros for `restricted`.
4. `NetworkConfig` needs `scalar_dim` to follow the tier
   (`make_network` already has this pattern).
5. Add `configs/training_human.yaml` mirroring `training_restricted.yaml`.

**Test that must exist.** Assert opponent elixir is *absent* from the human
vector — construct two engines identical except for opponent elixir and
assert the encoded vectors are equal. A dimension check alone will not catch
a regression that swaps a field in.

**Gotcha.** `RESTRICTED_SCALAR_DIM (18) > SCALAR_DIM (17)`. Anyone assuming
tiers are ordered by width will produce silently misaligned tensors.

## #32 — Domain-randomization wrapper

**Goal.** Train the `human` tier against *degraded* observations so the
policy survives real detection noise. This is the phase that decides whether
vision transfers at all; a policy trained on perfect coordinates falls over
on real frames.

**Implementation.** A wrapper around `encode_obs` for `human`/`full` tiers,
config-driven, applied **only during training** (clean for eval — otherwise
benchmark numbers stop being comparable):
- positional jitter: Gaussian offset before grid binning (do it in tile
  space *before* `_grid_cell`, not by perturbing the finished grid)
- missed detections: drop each enemy unit with probability `p_miss`
- false positives: inject phantom units at plausible positions
- identity confusion: swap a unit's card for a visually similar one
- occlusion dropout: drop all units inside a random arena patch
- staleness: with probability `p_stale`, reuse the previous frame's enemy
  channels (models capture/inference lag)

**Critical design note.** Randomize **enemy** channels only. Own units, own
elixir, and own hand come from the deterministic hand-cycle tracker and the
player's own UI, which are near-perfectly observable. Randomizing them
models noise that does not exist and will make the policy needlessly timid.

**Calibrate, do not guess.** Once #34 produces real detections, fit
`p_miss`/jitter to measured error on recorded frames. Guessed noise that is
qualitatively wrong (systematic bias vs. random jitter) transfers worse than
no randomization.

## #33 — Screen→arena homography

**Goal.** Map screen pixels to arena tiles. CR renders the arena in
perspective, so an affine scale is not enough — you need a full homography.

**Implementation.**
- Calibration from ≥4 known correspondences; the six tower centres plus the
  two bridge ends are the natural anchors and are visually distinctive.
- Solve for the 3×3 matrix (`numpy.linalg.lstsq` on the standard DLT setup;
  no OpenCV dependency needed, though `cv2.findHomography` is fine if you
  already have it).
- Provide **both directions**: pixel→tile for perception, tile→pixel for
  deploy targeting. `src/live/runner.py` currently taps fixed configured
  points; tile→pixel is what lets the policy choose an arbitrary placement.
- Persist calibration in `configs/live_play.yaml` under the existing
  `reference_size` discipline. `LiveConfig.validate()` already rejects
  out-of-frame coordinates — extend it to validate the homography anchors.

**Test.** Round-trip synthetic anchors: tile→pixel→tile within a small
tolerance, plus a degenerate-input case (collinear anchors must raise, not
silently return garbage).

## #34 — Team-tinted blob detection

**Goal.** First real vision cut: "hostile unit at (x, y)" without identity.

**Why this is enough to start.** The spatial channels are mostly HP-density
and presence. Presence + position alone recovers most of the observation.
Identity (#35) is a refinement, not a prerequisite.

**Implementation.** In `src/live/vision.py`, which currently has only
`mean_luma`/`mean_saturation`:
- Segment by team colour (CR tints health bars and unit outlines
  red/blue). Health bars are the most reliable cue — they are solid,
  high-saturation, and roughly axis-aligned.
- Connected components → centroids → homography → tile coordinates.
- Approximate HP from health-bar fill fraction where visible; assume full HP
  otherwise. **Document this as a known fidelity loss** — it is real and
  affects how the policy values trades.

**Test against saved frames, never a live match.** Commit a small set of
annotated screenshots as fixtures. There is no way to unit-test against a
live game, and running against live servers is out of scope for testing.

## #35 — Card identity via deck prior

**Goal.** Which card, not just which team.

**Key insight that makes this tractable.** A deck is 8 cards and reveals
itself over a match. Constrain classification to the observed candidate set
rather than the full ~108-card roster. Track revealed cards per match and
narrow as you go — this is the same information the deterministic tracker
(#38) maintains, so share that state rather than duplicating it.

**Escalation path.** Template matching on the candidate set first. Only
fine-tune a detector (YOLOv8n/RT-DETR) if that is measurably limiting.
Semi-auto-label your own units from known deploys; opponent units need hand
labelling, which is the real cost.

## #36 — Sim dynamics fidelity

**This is the task that actually gates real-match strength.** Once the
policy can see the board, the binding constraint becomes whether simulated
*dynamics* match real CR. Audit and prioritise by how much each mismatch
changes decision-making:

- **Pathing.** `movement.waypoint_toward` is straight-line-with-bridge-
  routing. Real units path around obstacles and have turn radii. Probably
  the largest remaining gap.
- **Card stats.** `configs/cards.yaml` is explicitly "roughly half of real
  Clash Royale values". Internally consistent, but any policy reasoning
  about absolute breakpoints (does fireball kill musketeer?) learns wrong
  thresholds. Consider rescaling to true values.
- **Targeting quirks.** Real CR has aggro ranges, retarget rules, and
  first-target preferences that `targeting.py` approximates.
- **Unit-specific behaviour** not modelled at all: Prince/Battle Ram charge,
  Bandit/Royal Ghost dash, Miner tunnelling, Sparky reset-on-stun, Inferno
  ramp-up damage, Lava Hound death-spawn, Balloon death-damage.
- **Spell nuance.** `spell_effects.py` documents its simplifications —
  tornado pulls in one instant step rather than continuously, rage is a
  snapshot rather than a persistent zone.

**Method.** Do not fix everything. For each candidate, ask: does this change
*which card a good player would choose*? Charge mechanics and Inferno ramp
do; a 3% stat difference does not.

---

# Group B: #37–#41 — learning quality

## #37 — Privileged teacher → human student distillation ⭐

**Expected to be the single largest quality lever after throughput.**

**Idea.** Train a `full`-tier teacher (sees opponent elixir, perfect
positions) to high strength in simulation, then distil it into a
`human`-tier student. The teacher's supervision transfers strategy the
student could not discover alone, while the student only ever consumes
human-legal inputs at inference.

**Implementation.**
- Teacher: existing pipeline with the `full` tier. Optionally strengthen
  with #40 search.
- Student loss: `PPO_loss + λ · KL(teacher_π ‖ student_π)` on states visited
  by the **student** (on-policy distillation / DAgger-style). Distilling on
  teacher-visited states causes distribution mismatch — the classic
  behaviour-cloning failure this project already hit once.
- Both networks must see the *same underlying state*, encoded at different
  tiers, so run them over one engine and encode twice.
- Anneal λ down so the student eventually optimises its own return rather
  than imitating a teacher whose information it will never have.

**Watch for.** The teacher will make plays that are only correct *given*
knowledge of opponent elixir (e.g. committing a big push into a known-empty
bar). The student cannot know that and will learn a superstition. Mitigate
by weighting the KL term by teacher-value confidence, or masking states
where teacher and student value estimates diverge sharply.

## #38 — Deterministic opponent tracker ⭐

**Read the full task description — the design was revised after a good
argument that elixir counting is deterministic arithmetic, not estimation.**

**Do not learn what you can compute.** Opponent elixir is exactly derivable
from observed plays: start value + known regen (2× after `double_time`) +
known card costs, capped at `elixir_max` with overflow discarded. Cycle is
likewise deterministic once the 8-card deck is revealed.

**Self-correction is what makes it robust.** Vision sees *units*, not just
deploy animations. If a unit appears that was never observed being deployed,
infer the play retroactively, subtract its cost, and raise an uncertainty
counter. A missed spell gets caught when its effect lands.

**Edge cases that will silently corrupt the count:** Elixir Collector
income, Elixir Golem granting elixir to the *enemy* on death, Mirror's +1
cost, and elixir overflow at cap.

**Hard invariant.** The tracker consumes only what the observation pipeline
saw. It must never read `engine.players[opponent].elixir`. Add a test that
constructs a divergence (feed it a partial play history) and asserts the
tracker's output differs from engine ground truth — proving it is deriving,
not peeking. Sim ground truth may be used *only* as an auxiliary training
signal to measure tracker accuracy.

## #39 — Set-based unit encoder

**Goal.** Replace/augment the CNN-over-grid with a permutation-invariant
encoder over a variable-length unit list.

**Why.** The 9×16 grid quantises positions into 2×2-tile cells and sums
HP into density channels, discarding exact position and per-unit identity.
Placement precision matters in CR.

**Implementation.** Per-unit embedding of (card id, hp fraction, x, y, side,
flying, deploy-state, shield/freeze flags) → deep-sets (mean/max pool) or a
small transformer encoder. Keep the grid path behind a config flag for
ablation and compare on the frozen benchmark, not on self-play win rate.

**Interaction with #32.** A set encoder consumes detections directly, which
makes domain randomisation *more* natural (drop/jitter list entries) than
perturbing a rasterised grid. Coordinate with whoever owns #32.

## #40 — Inference-time search

**Sim-only.** Too slow for live play; its value is (a) a much stronger
sim agent and (b) a better teacher for #37.

**Implementation.** `BattleEngine` is plain Python and deterministic given a
seed, so `copy.deepcopy` gives a workable clone — but profile it, since the
engine holds `cards` (now `lru_cache`-backed, so shared) and per-unit
objects. Root-parallel MCTS or simple rollout-and-average over the top-k
policy actions. Budget against the 0.5s decision cadence *for sim eval
only*.

**Reality check.** The action space is 5 card choices × 144 cells. Do not
search the raw product — restrict to the policy's top-k masked actions.

## #41 — Population-based training

Current hyperparameters (`lr 3e-4`, `clip 0.2`, `gamma 0.997`, entropy
schedule, reward weights) were hand-set and never searched.

**Select on the frozen benchmark, never on self-play win rate** — the latter
trends to 50% by construction and is not a progress signal. This is stated
in `CLAUDE.md` under Evaluation Discipline and is the single easiest way to
waste a week of compute.

Blocked on throughput (#24/#25) — PBT needs many concurrent runs.

---

## Cross-cutting

- **Never delete the frozen benchmark bots.** They are the permanent
  regression tripwire (`configs/eval.yaml`, `src/eval/benchmark.py`).
- **Checkpoint compatibility.** `save_checkpoint` stores
  `asdict(net.config)`; `load_checkpoint` pops `n_cards` and passes the rest
  to `make_network`. Any new `NetworkConfig` field needs a safe default or
  every existing checkpoint breaks.
- **Existing checkpoints predate the current physics** (collision, deploy
  delay, per-card deploy times, freeze/rage/tornado/poison/clone). Their
  recorded win rates are not comparable to new numbers. #30 re-benchmarks
  them; do not treat `checkpoints/README.md`'s 88% as a live baseline.
- **Run the full suite before claiming done** — currently 173 tests. Some
  take minutes; a fast inner loop is
  `pytest tests/test_collision.py tests/test_deploy_delay.py -q`.
