# Trained checkpoint (this branch)

`checkpoints/full_pool_60m.pt` — frozen weights from `my_run` after ~60M steps on `full_pool`.

## Benchmark (50 matches × 4 opponents, `training_mirror` deck)

| Opponent | Win rate |
|----------|----------|
| Control  | 100% |
| Siege    | 100% |
| Beatdown | 80% |
| Rusher   | 72% |
| **Overall** | **88%** (176–24) |

## Load

```bash
.\.venv\Scripts\activate
python -c "from src.agent.selfplay import PolicyBot; b=PolicyBot.load('checkpoints/full_pool_60m.pt', deterministic=True); print(b.name)"
```

Or resume training:

```bash
python -m src.agent.train --run continue --stage full_pool --resume checkpoints/full_pool_60m.pt --seed 0
```
