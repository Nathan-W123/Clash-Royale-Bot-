"""Launch the 3D network viewer.

    python -m src.viz                                   # untrained net, watch it learn
    python -m src.viz --checkpoint checkpoints/full_pool_60m.pt --mode live
    python -m src.viz --mode training --tier restricted

Then open http://localhost:8770.

To watch a *real* training run or a *real* live match instead of the
self-contained sources here, attach the viewer to that process:

    python -m src.agent.train --run r1 --stage full_pool --viz-port 8770
    python -m src.live --config configs/live_play.yaml --viz-port 8770
"""
from __future__ import annotations

import argparse

from src.viz import telemetry
from src.viz.server import VizController, serve
from src.viz.sources import SimulationSource, TrainingSource


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D policy-network viewer (training + live).")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback by default; anything else exposes "
                             "network internals to your LAN")
    parser.add_argument("--checkpoint", default=None,
                        help="policy to watch; omit for a fresh untrained net "
                             "(which is what makes the training view legible)")
    parser.add_argument("--tier", default="human",
                        choices=["full", "human", "restricted"],
                        help="observation tier for a fresh network")
    parser.add_argument("--mode", default="live", choices=["live", "training", "idle"],
                        help="which source to start with")
    parser.add_argument("--opponent", default="rusher",
                        help="scripted benchmark bot to play against")
    parser.add_argument("--deck", default="training_mirror")
    parser.add_argument("--fps", type=float, default=6.0,
                        help="live-view decisions per second (the sim runs "
                             "faster than this; the cap is for watchability)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="parallel envs for the built-in training source")
    parser.add_argument("--n-steps", type=int, default=128,
                        help="rollout length per PPO update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    telemetry.set_enabled(True)
    controller = VizController({
        "live": SimulationSource(args.checkpoint, opponent=args.opponent,
                                 deck=args.deck, tier=args.tier, fps=args.fps,
                                 seed=args.seed),
        "training": TrainingSource(args.checkpoint, tier=args.tier,
                                   n_envs=args.n_envs, n_steps=args.n_steps,
                                   opponent=args.opponent, deck=args.deck,
                                   lr=args.lr, seed=args.seed),
    })
    controller.set_mode(args.mode)
    serve(port=args.port, host=args.host, controller=controller)


if __name__ == "__main__":
    main()
