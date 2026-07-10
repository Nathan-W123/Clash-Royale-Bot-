# CR Bot

A reinforcement learning agent that learns to play Clash Royale entirely inside a custom battle simulator. This is an educational RL project — not a real-game bot.

**Current phase:** simulator, scripted opponents, eval reporting, and training infrastructure. The PPO agent is next.

---

## What it does

| Component | Status | Description |
|-----------|--------|-------------|
| Battle simulator | ✅ | Config-driven engine (`src/simulator/`) |
| UC-tier scripted bots | ✅ | Rusher, control, siege, beatdown heuristics |
| Deck pools & generation | ✅ | Ladder meta pool, adaptive builder, focused rotation |
| **Training reports** | ✅ | W/L, win rate, matchups, card usage, TensorBoard |
| Heroes & evolutions | ✅ | Config-driven; abilities + deck rules |
| PPO agent | ✅ | Masked factored policy + league self-play (`src/agent/`) |

---

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or pip

### Install

```bash
cd "CR Bot"
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

### Run tests

```bash
pytest tests/ -q
```

### Run benchmark + print report

Pits a bot against the **frozen benchmark suite** (rusher, control, siege, beatdown) and prints a full report:

```bash
python -m src.eval --agent-bot control --agent-deck control --matches 20
```

Example output:

```
=== Training Report: benchmark_control ===
Matches: 80  |  W 42  L 35  D 3
Win rate: 54.5%  |  W/L ratio: 1.20
Avg crowns: 1.12 for  0.98 against
Avg match: 145.3s  |  Elixir spent: 38.2  leaked: 1.40
Card usage entropy: 2.01 nats

--- vs Opponent Deck ---
  beatdown          12W    8L  WR  60.0%  W/L 1.50
  control           10W   10L  WR  50.0%  W/L 1.00
  ...
```

Export JSON or log to TensorBoard:

```bash
python -m src.eval --agent-bot rusher --agent-deck rusher --matches 50 \
  --json runs/report.json \
  --tensorboard runs/tb --step 100000
tensorboard --logdir runs/tb
```

---

## Start training

Activate the venv, then launch PPO from the first curriculum stage (`one_lane`):

```bash
.venv\Scripts\activate
python -m src.agent.train --run my_run --stage one_lane --seed 0
```

Useful flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--run` | `run1` | Run name; logs go to `runs/<run>/`, checkpoints to `checkpoints/<run>/` |
| `--stage` | first stage | Curriculum stage (`one_lane`, `full_arena`, `full_pool`, …) |
| `--auto` | off | Run all curriculum stages in sequence |
| `--steps` | `ppo.total_steps` (3M) | Per-stage environment-step budget |
| `--n-envs` | `8` | Parallel envs (see `configs/training.yaml`) |
| `--bc-init` | — | Warm-start from a BC checkpoint (`python -m src.agent.bc`) |
| `--resume` | — | Continue from a saved `.pt` checkpoint |

Quick smoke run (~30s):

```bash
python -m src.agent.train --run smoke --stage one_lane --steps 4096 --n-envs 2 --seed 0
```

Monitor training:

```bash
tensorboard --logdir runs/my_run/tb
```

Outputs:

- `runs/<run>/train_log.csv` — per-rollout metrics (win rate, entropy, losses)
- `runs/<run>/eval_log.csv` — periodic eval vs scripted bots
- `runs/<run>/tb/` — TensorBoard scalars
- `runs/<run>/latest.pt` and `final.pt` — policy checkpoints
- `checkpoints/<run>/pool/` — league snapshots for self-play stages

Optional BC bootstrap before PPO:

```bash
python -m src.agent.bc --matches 150 --out checkpoints/bc_init.pt
python -m src.agent.train --run my_run --stage one_lane --bc-init checkpoints/bc_init.pt
```

---

## Training reports

Reports track what the agent (or stand-in bot) learns over time.

### Metrics collected per match

- **Win / loss / draw** and **win rate**
- **W/L ratio** (wins ÷ losses)
- **Crowns** scored vs conceded
- **Match duration** and **elixir** spent / leaked
- **Card usage** counts and **entropy** (strategy diversity)
- Breakdowns by **opponent deck**, **opponent bot**, and **agent deck**
- **Matchup matrix** (your deck × their deck)
- **Weak matchups** flagged when WR < 50% over 3+ games

### Using the reporter in code

```python
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.training.session import run_episodes, run_focused_training

catalog = DeckCatalog()
bot = get_bot("rusher", catalog=catalog, deck_name="rusher")

# Random ladder-meta opponents
reporter = run_episodes(bot, n_episodes=100, stage="full_pool", seed=0)

# Focused rotation: beat each ladder deck in sequence, adaptive agent deck
focused = run_focused_training(bot, max_episodes=500, stage="focused_ladder", seed=0)
print(focused.summary())
focused.export_json("runs/focused_training.json")
```

When the RL agent lands, call `TrainingReporter.record()` or `record_report()` after each episode — same API.

### What to watch during training

| Signal | Good sign | Bad sign |
|--------|-----------|----------|
| Win rate vs benchmark bots | Trending up | Flat or falling |
| Self-play win rate | ~50% (expected) | Don't use as sole metric |
| Card entropy | Stable or rising | Collapsing → mode collapse |
| Per-deck WR | Balanced across archetypes | One deck at 90%, others at 30% |

---

## Deck system

### Ladder meta opponents (`configs/ladder_decks.yaml`)

70 curated Ultimate Champion decks sourced from RoyaleAPI TopRanked (7d) and Season 81 leaderboard data (March–April 2026). Each deck has a name, archetype tag, and 8 cards. The `ladder_top50` pool is auto-built from this file and is the default opponent pool in `configs/training.yaml`.

### Focused rotation curriculum

`run_focused_training()` plays matches against **one fixed ladder deck at a time**. The agent advances to the next deck after:

- **X wins** vs the current deck (`focused_rotation.wins_per_deck`), or
- **min matches** with per-deck win rate above `min_win_rate_per_deck`

After cycling through all ladder decks, training stops when overall win rate reaches **`target_win_rate` (Y)**; otherwise the cycle restarts.

Config (`configs/training.yaml`):

```yaml
focused_rotation:
  wins_per_deck: 5
  target_win_rate: 0.65
  min_matches_per_deck: 10
  min_win_rate_per_deck: 0.55
```

### Adaptive agent deck builder

During focused training the agent does **not** use a fixed deck. `AdaptiveDeckBuilder` tracks per-card performance (win rate when played, crowns, elixir efficiency) and rebuilds an 8-card deck by picking the best card in each role category (`configs/card_categories.yaml`):

- win condition → spell → building → cycle → support → swarm → air → heavy
- Max 1 hero and 1 evolution per deck

Rebuild triggers (`adaptive_deck` in `training.yaml`):

- Every `rebuild_every_matches` games, or
- When win rate plateaus over `plateau_window` matches

Card scores are exported in JSON reports under `card_scores`.

---

```
configs/
  cards.yaml          # Card stats (balance lives here)
  ladder_decks.yaml   # Top UC ladder meta decks (~70)
  card_categories.yaml # Roles for adaptive deck builder
  decks.yaml          # Named decks + pools
  deck_templates.yaml # Procedural deck generation
  arena.yaml          # Arena geometry & clock
  curriculum.yaml     # Training stages (incl. focused_ladder)
  training.yaml       # PPO hyperparams, focused rotation, adaptive deck
  eval.yaml           # Frozen benchmark opponents

src/
  simulator/          # Battle engine
  bots/               # Scripted UC-tier opponents
  decks/              # Catalog, sampling, adaptive builder, deck search
  training/           # Opponent sampling, focused curriculum, sessions
  agent/              # Gym env adapter, PPO, policy network, BC, league
  eval/               # Reports, benchmarks, TensorBoard

tests/                # Unit + integration tests
```

---

## Training pipeline

1. **Imitation bootstrap (optional)** — `python -m src.agent.bc` from scripted bot rollouts
2. **PPO self-play** — `python -m src.agent.train`; league of past checkpoints in self-play stages
3. **Eval discipline** — always benchmark against frozen bots in `configs/eval.yaml`

Opponents train at **Ultimate Champion** tier: elixir discipline, counter-play, spell value, archetype offense.

Curriculum stages (`configs/curriculum.yaml`):

1. `one_lane` — single lane, small deck, UC scripted bots
2. `full_arena` — full map, mirror deck, self-play mix
3. `full_pool` — both sides sample from ladder meta pool
4. `focused_ladder` — focused rotation vs one ladder deck at a time + adaptive agent deck

---

## Heroes & evolutions

The simulator supports CR-style **heroes** and **evolutions** via config:

| Feature | Config | Rules |
|---------|--------|-------|
| **Heroes** | `configs/heroes.yaml` | Max 1 per deck; ability charges on arena, auto-fires at full charge |
| **Evolutions** | `configs/evolutions.yaml` | Max 1 per deck; loaded as `{base}_evo` cards with stat overrides |

**Hero abilities** (simplified): `dash` (leap + burst damage), `spawn` (summon troops), `damage_buff` (temporary attack boost).

**Evolution examples:** `knight_evo`, `hog_rider_evo`, `skeletons_evo`, `fireball_evo`, etc.

Deck validation enforces: no duplicate base+evo pair, one hero, one evolution. Archetype decks (`rusher`, `control`, etc.) already include a hero and evolution each.

---

## Configuration

Key files — edit YAML, not code, for balance and training tweaks:

- **`configs/cards.yaml`** — HP, damage, cost, range
- **`configs/heroes.yaml`** — hero troops and abilities
- **`configs/evolutions.yaml`** — evolution stat overrides
- **`configs/decks.yaml`** — deck lists and pools
- **`configs/ladder_decks.yaml`** — UC meta opponent decks
- **`configs/card_categories.yaml`** — adaptive builder role mapping
- **`configs/training.yaml`** — PPO settings, `focused_rotation`, `adaptive_deck`
- **`configs/eval.yaml`** — benchmark roster (never delete opponents)

---

## Out of scope

No integration with the real Clash Royale client (screen capture, OCR, input injection, etc.). Simulator only.

---

## License

Personal / educational project.
