"""Drivers that make the graph move.

Two of them, and the difference matters:

`SimulationSource` — plays real matches in the simulator with a real policy
and streams the activations of each forward pass. This is what "live" looks
like *without needing a live Clash Royale match running*: the network, the
observation encoding, and the forward pass are all the production ones, only
the pixels-to-observation step is replaced by the simulator. It is the
honest stand-in, and it is also what you want for inspecting a checkpoint.

`TrainingSource` — runs a real PPO loop (the project's own `RolloutBuffer`
and `ppo_update`, not a reimplementation) so weight movement is genuine.

Neither is the production training path. For a real run, attach the viewer
to the real trainer with `python -m src.agent.train --viz-port 8770`; these
two exist so the viewer is useful on its own and so the UI can be exercised
without a multi-hour run.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from src.agent import obs_layout
from src.agent.network import make_network, masks_to_tensors, obs_to_tensors
from src.agent.ppo import PPOConfig, RolloutBuffer, ppo_update
from src.agent.selfplay import BotOpponent, load_checkpoint
from src.bots.registry import get_bot
from src.decks.catalog import DeckCatalog
from src.simulator.cards import load_arena, load_cards
from src.simulator.constants import Side
from src.simulator.env import CRBattleEnv
from src.simulator.vec_env import SyncVecEnv
from src.viz import telemetry
from src.viz.graph import build_graph
from src.viz.probe import NetworkProbe

DEVICE = torch.device("cpu")


def load_policy(checkpoint: str | Path | None, tier: str = "human"):
    """A network to watch: a checkpoint if given, else a fresh one.

    A fresh network is genuinely useful here rather than a fallback — the
    "reveal as it learns" view is only interesting when the weights start
    untrained, and an untrained policy is the correct starting point for
    watching training fill the graph in.
    """
    if checkpoint:
        net, card_names = load_checkpoint(Path(checkpoint))
        return net, card_names, str(checkpoint)
    card_names = list(load_cards().keys())
    net = make_network(len(card_names), {"tier": tier})
    return net, card_names, f"untrained ({tier} tier)"


class _ThreadedSource:
    """Common lifecycle: one background thread, cooperative stop."""

    name = "source"

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._guarded, daemon=True,
                                        name=f"viz-{self.name}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)

    def _guarded(self) -> None:
        try:
            self.run()
        except Exception as error:                    # noqa: BLE001
            # A crashed source must surface in the UI, not vanish into a dead
            # daemon thread while the graph sits there looking fine.
            telemetry.log(f"[{self.name}] stopped: {error!r}", "error")
            raise

    def run(self) -> None:                            # pragma: no cover
        raise NotImplementedError


class SimulationSource(_ThreadedSource):
    """Play matches in the simulator, streaming one activation frame per decision."""

    name = "live"

    def __init__(self, checkpoint: str | Path | None = None,
                 opponent: str = "rusher", deck: str = "training_mirror",
                 tier: str = "human", fps: float = 6.0, seed: int = 0) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.opponent_name = opponent
        self.deck_name = deck
        self.tier = tier
        # The agent decides every 0.5s of simulated time; running the stream
        # slower than real time makes the flow legible instead of a strobe.
        self.interval = 1.0 / max(fps, 0.5)
        self.seed = seed

    def run(self) -> None:
        net, card_names, label = load_policy(self.checkpoint, self.tier)
        net.eval()
        cards, arena, catalog = load_cards(), load_arena(), DeckCatalog()
        graph = build_graph(net)
        telemetry.emit("graph", graph)
        telemetry.log(f"[live] policy: {label} | tier={net.config.tier} "
                      f"| {graph['meta']['params']:,} params", "good")

        deck = catalog.resolve(self.deck_name)
        rng = np.random.default_rng(self.seed)
        bot = get_bot(self.opponent_name, catalog=catalog, deck_name=self.opponent_name
                      if self.opponent_name in catalog.decks else self.deck_name, rng=rng)
        env = CRBattleEnv(cards, arena, deck, list(deck),
                          opponent=BotOpponent(bot), tier=net.config.tier,
                          with_units=net.config.use_set_encoder)

        probe = NetworkProbe(net, graph).attach()
        # Weights are frozen during play, so movement is zero forever and the
        # graph would render as a bare skeleton. One frame at attach carries
        # `maturity`, which reads structure out of the weights themselves —
        # a trained checkpoint should look like the trained network it is.
        telemetry.emit("learn", {**probe.learning_frame(), "step": 0,
                                 "frozen": True})
        try:
            match = 0
            while not self._stop.is_set():
                match += 1
                obs, info = env.reset(seed=int(rng.integers(2**31)))
                masks = info["masks"]
                telemetry.log(f"[live] match {match} vs {self.opponent_name} "
                              f"({self.deck_name})", "good")
                self._play(env, net, probe, obs, masks, card_names)
        finally:
            probe.detach()

    def _play(self, env, net, probe, obs, masks, card_names) -> None:
        step = 0
        while not self._stop.is_set():
            actions, _, values = net.act(obs_to_tensors(obs, DEVICE),
                                         masks_to_tensors(masks, DEVICE))
            action = actions[0].cpu().numpy()
            frame = probe.activation_frame()
            engine = env.engine
            me = engine.players[Side.BOTTOM]

            if frame is not None:
                telemetry.emit("act", {
                    "nodes": frame,
                    "step": step,
                    "value": round(float(values[0]), 3),
                    "clock": round(float(getattr(engine, "time", 0.0)), 1),
                    "elixir": round(float(me.elixir), 2),
                    "action": int(action[0]),
                    "cell": int(action[1]),
                })
            if action[0] > 0:
                slot = int(action[0]) - 1
                card = me.hand[slot].name if slot < len(me.hand) else "?"
                row, col = divmod(int(action[1]), obs_layout.PLACE_COLS)
                telemetry.log(
                    f"  t={float(getattr(engine, 'time', 0.0)):6.1f}s  "
                    f"play {card:<18} cell ({col},{row})  "
                    f"elixir {float(me.elixir):.1f}  V={float(values[0]):+.2f}")

            obs, _, terminated, _, info = env.step(action)
            step += 1
            if terminated:
                metrics = info.get("episode_metrics", {})
                telemetry.log(f"[live] match over: {metrics.get('result', '?')} "
                              f"in {metrics.get('match_time', 0):.0f}s", "good")
                telemetry.emit("stats", {"scope": "live", "metrics": {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, str))}})
                return
            masks = info["masks"]
            self._stop.wait(self.interval)


class TrainingSource(_ThreadedSource):
    """A real (small) PPO loop, so weight movement on screen is real movement."""

    name = "training"

    def __init__(self, checkpoint: str | Path | None = None,
                 tier: str = "human", n_envs: int = 4, n_steps: int = 128,
                 opponent: str = "rusher", deck: str = "training_mirror",
                 lr: float = 3e-4, seed: int = 0) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.tier = tier
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.opponent_name = opponent
        self.deck_name = deck
        self.lr = lr
        self.seed = seed

    def run(self) -> None:
        net, card_names, label = load_policy(self.checkpoint, self.tier)
        net.train()
        cards, arena, catalog = load_cards(), load_arena(), DeckCatalog()
        graph = build_graph(net)
        telemetry.emit("graph", graph)
        telemetry.log(f"[train] policy: {label} | tier={net.config.tier} "
                      f"| {graph['meta']['params']:,} params", "good")
        telemetry.log(f"[train] {self.n_envs} envs x {self.n_steps} steps "
                      f"per update, lr={self.lr}", "good")

        deck = catalog.resolve(self.deck_name)
        rng = np.random.default_rng(self.seed)

        def env_fn():
            bot = get_bot(self.opponent_name, catalog=catalog,
                          deck_name=self.deck_name, rng=rng)
            return CRBattleEnv(cards, arena, deck, list(deck),
                               opponent=BotOpponent(bot), tier=net.config.tier,
                               with_units=net.config.use_set_encoder)

        envs = SyncVecEnv([env_fn for _ in range(self.n_envs)])
        obs, masks = envs.reset(seed=self.seed)
        obs_shapes = {k: v.shape[1:] for k, v in obs.items()}
        mask_shapes = {k: v.shape[1:] for k, v in masks.items()}
        cfg = PPOConfig(n_steps=self.n_steps, batch_size=256, n_epochs=4, lr=self.lr)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        probe = NetworkProbe(net, graph).attach()

        update, global_step, wins = 0, 0, []
        try:
            while not self._stop.is_set():
                buffer = RolloutBuffer(cfg.n_steps, self.n_envs, obs_shapes, mask_shapes)
                t0 = time.perf_counter()
                reward_sum = 0.0

                for i in range(cfg.n_steps):
                    if self._stop.is_set():
                        return
                    actions, log_probs, values = net.act(
                        obs_to_tensors(obs, DEVICE), masks_to_tensors(masks, DEVICE))
                    actions_np = actions.cpu().numpy()
                    next_obs, rewards, dones, next_masks, infos = envs.step(actions_np)
                    buffer.add(obs, masks, actions_np, log_probs.cpu().numpy(),
                               values.cpu().numpy(), rewards, dones)
                    reward_sum += float(rewards.sum())
                    for info in infos:
                        m = info.get("episode_metrics")
                        if m:
                            wins.append(float(m["win"]))
                    obs, masks = next_obs, next_masks

                    # Stream activations during rollout collection, not only
                    # at update boundaries — otherwise the graph is frozen for
                    # the whole rollout and only twitches once per update.
                    if i % 8 == 0:
                        frame = probe.activation_frame()
                        if frame is not None:
                            telemetry.emit("act", {"nodes": frame,
                                                   "step": global_step + i})

                with torch.no_grad():
                    _, _, last_values = net.act(obs_to_tensors(obs, DEVICE),
                                                masks_to_tensors(masks, DEVICE))
                buffer.compute_returns(last_values.cpu().numpy(), cfg.gamma,
                                       cfg.gae_lambda)
                stats = ppo_update(net, optimizer, buffer, cfg,
                                   cfg.ent_coef(global_step), DEVICE,
                                   np.random.default_rng(self.seed + update))
                global_step += cfg.n_steps * self.n_envs
                update += 1
                sps = cfg.n_steps * self.n_envs / (time.perf_counter() - t0)
                win_rate = float(np.mean(wins[-50:])) if wins else 0.0

                learn = probe.learning_frame()
                telemetry.emit("learn", {**learn, "step": global_step,
                                         "update": update})
                telemetry.emit("stats", {"scope": "training", "metrics": {
                    "step": global_step, "update": update,
                    "win_rate": round(win_rate, 3),
                    "reward": round(reward_sum / (cfg.n_steps * self.n_envs), 4),
                    "entropy": round(stats["entropy"], 3),
                    "policy_loss": round(stats["policy_loss"], 4),
                    "value_loss": round(stats["value_loss"], 4),
                    "sps": round(sps),
                    "revealed": round(float(np.mean(learn["reveal"])), 4),
                }})
                telemetry.log(
                    f"[train] update {update:>4}  step {global_step:>8}  "
                    f"win {win_rate:.2f}  ent {stats['entropy']:.2f}  "
                    f"pl {stats['policy_loss']:+.4f}  vl {stats['value_loss']:.4f}  "
                    f"{sps:.0f} sps  revealed {np.mean(learn['reveal']):.1%}")
        finally:
            probe.detach()


class AttachedSource:
    """Placeholder for 'something else in this process is emitting'.

    Used when the viewer rides along inside `src.agent.train` or `src.live`:
    the controller must not start a competing source, but the mode still
    needs to exist so the UI can select it.
    """

    def __init__(self, name: str, note: str, activity: str = "training") -> None:
        self.name = name
        self.note = note
        # What the host process is doing, so the UI can distinguish a
        # training run from live play — both arrive as mode "attached".
        self.activity = activity

    def start(self) -> None:
        telemetry.log(f"[{self.name}] {self.note}", "good")

    def stop(self) -> None:
        pass
