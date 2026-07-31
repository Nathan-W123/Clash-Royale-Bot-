# Trained checkpoint (this branch)

`checkpoints/full_pool_60m.pt` — frozen weights from `my_run` after ~60M steps
on `full_pool`, using the **`full` observation tier** (reads opponent elixir,
so it is simulator-only and not deployable to live play).

## Benchmark: 50 matches x 4 opponents, `training_mirror` deck

The same checkpoint, the same protocol, measured against two different
simulators. Nothing about the model changed between these columns.

| Opponent | Old physics | **Current physics** | Delta |
|----------|------------:|--------------------:|------:|
| Control  | 100% | **96%** | -4% |
| Siege    | 100% | **92%** | -8% |
| Rusher   |  72% | **56%** | -16% |
| Beatdown |  80% | **54%** | -26% |
| **Overall** | **88%** | **74%** | **-14%** |

**This drop is not a regression.** It is the simulator getting harder and more
realistic. Between the two measurements the engine gained body collision with
obstacle steering, per-card deploy times, real freeze/rage/tornado/poison/clone
effects, and unit mechanics (charge, ramp-up, death spawns, stun). The old
number was recorded against a world where a tank could not block, siege
buildings fired the instant they landed, and Tornado was a mild damage spell.

Corroborating detail: average crowns *for* fell 1.75 -> 0.86 while crowns
*against* fell 0.57 -> 0.18, and matches got longer (180s -> 194s). Both sides
find it much harder to close out a game, which is exactly what stronger
defense should look like — not a policy that got worse at attacking.

The largest losses are against **rusher (-16%) and beatdown (-26%)**, the two
aggressive archetypes. That is consistent with the known blind spot: punishing
a rush now requires body-blocking and deploy-timing skills that did not exist
in the world this policy trained in.

### What to do with this

Treat **74% as the live baseline** for anything measured today. The 88% figure
is only meaningful against a simulator that no longer exists, and comparing new
runs to it will make real progress look like regression.

Retraining under current physics is expected to recover a good part of the gap:
these numbers measure a policy playing a game it was never trained on.

Raw reports: `artifacts/benchmark_report.json` (old),
`artifacts/benchmark_full_pool_60m_newphysics.json` (current).

## Load

```bash
.\.venv\Scripts\activate
python -m src.eval --checkpoint checkpoints/full_pool_60m.pt --matches 50
```

Or resume training:

```bash
python -m src.agent.train --run continue --stage full_pool \
  --resume checkpoints/full_pool_60m.pt --seed 0 --global-step-start 60000000
```

`--global-step-start` matters on a resume: the reward-anneal and entropy
schedules are keyed off the global step, so omitting it restarts both from
their step-0 values on a network that is already deep into training.

## Regression gate baseline (`checkpoints/best/`)

Established 2026-07-31 under current physics via:

```bash
python -m src.eval.regression --candidate <ckpt> --matches 50
```

`checkpoints/best/model.pt` currently holds `full_pool_60m.pt`, with
`scores.json` as the bar a candidate must clear. The gate exits **1** on
regression, so it can front a training pipeline. Verified working: the
17.1M-step restricted policy is correctly rejected against this baseline.

Note the baseline scores are noisier than the headline benchmark above
(fewer matches per opponent); use `--matches 50` when promoting for real.
