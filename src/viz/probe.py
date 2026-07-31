"""Read two different things off a live network, per drawn node.

**Activation** (`act` frames) — what fired on the last forward pass. This is
the live-play signal: the pulse travelling through the graph is a real
forward pass over a real observation.

**Weight movement** (`learn` frames) — how far each unit's incoming weights
have travelled since the probe attached. This is what drives the "grows as
it learns" behaviour, and it is worth being precise about what it is and
isn't:

    The architecture is fixed. Training does not add units. So nothing here
    literally grows. What grows is how much of the network has *moved* —
    `reveal` is a monotonic, normalised record of cumulative weight travel
    per unit. A freshly initialised network starts dark and fills in as
    training reshapes it; a unit whose weights never move stays dark, which
    is itself informative.

`reveal` is measured *from attach*, so resuming a 60M-step checkpoint starts
at zero again — that is the honest reading (nothing has moved yet this
session), and `maturity` is sent alongside for the "how developed is this
network right now" view, derived from weight-norm spread rather than from
motion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9
# Running peak decays so a single outlier activation does not permanently
# flatten the display, but recent history still sets the scale.
_PEAK_DECAY = 0.99
# Full brightness at this much relative weight travel. Chosen so early PPO
# updates are visible without saturating within the first few rollouts;
# it is a display scale, not a claim about the optimiser.
_REVEAL_FULL = 0.75


@dataclass
class _Slot:
    """Where one layer's per-unit vector comes from on a forward pass."""
    layer: str
    source: str      # "output" | "input" | custom tag
    reduce: str      # how to collapse the tensor to per-unit values
    index: int = 0
    offset: int = 0
    length: int = 0


class NetworkProbe:
    """Forward hooks + a weight-movement tracker over a built graph."""

    def __init__(self, net, graph: dict):
        self.net = net
        self.graph = graph
        self.n_nodes = len(graph["nodes"])
        self._node_of: dict[str, list[tuple[int, int]]] = {}
        for layer in graph["layers"]:
            key = layer["key"]
            ids = layer["nodes"]
            units = [graph["nodes"][i]["unit"] for i in ids]
            self._node_of[key] = list(zip(ids, units))

        self._latest: dict[str, np.ndarray] = {}
        self._peak: dict[str, float] = {}
        self._handles: list = []
        self._reveal = np.zeros(self.n_nodes, np.float32)
        self._tracked: dict[str, tuple] = {}
        self._maturity = np.zeros(self.n_nodes, np.float32)
        self.attached = False

    # ------------------------------------------------------------ hooks

    def attach(self) -> "NetworkProbe":
        if self.attached:
            return self
        for module, slots in self._hook_plan().items():
            self._handles.append(
                module.register_forward_hook(self._make_hook(slots)))
        self._snapshot_weights()
        self.attached = True
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.attached = False

    def __enter__(self) -> "NetworkProbe":
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    def _hook_plan(self) -> dict:
        """module -> slots it can satisfy.

        Pseudo-layers (`obs.*`) are read from the *inputs* of the first real
        module that consumes them, which avoids having to wrap `trunk` itself.
        """
        from src.simulator.constants import HAND_SIZE

        net, cfg = self.net, self.net.config
        plan: dict = {}

        def want(module, *slots: _Slot) -> None:
            if module is None:
                return
            plan.setdefault(module, []).extend(slots)

        keys = {layer["key"] for layer in self.graph["layers"]}

        for prefix, embed, hand_first, cnn, cnn_proj, fusion, set_enc in (
            ("", net.card_embed, net.hand_mlp[0],
             getattr(net, "cnn", None), getattr(net, "cnn_proj", None),
             net.fusion, getattr(net, "set_encoder", None)),
            ("c.", getattr(net, "critic_card_embed", None),
             net.critic_hand_mlp[0] if hasattr(net, "critic_hand_mlp") else None,
             getattr(net, "critic_cnn", None), getattr(net, "critic_cnn_proj", None),
             getattr(net, "critic_fusion", None),
             getattr(net, "critic_set_encoder", None)),
        ):
            if f"{prefix}card_embed" not in keys:
                continue
            want(embed,
                 _Slot(f"{prefix}obs.cards", "output", "norm_last"),
                 _Slot(f"{prefix}card_embed", "output", "norm_mid"))
            scalar_off = (HAND_SIZE + 1) * cfg.card_embed_dim
            want(hand_first,
                 _Slot(f"{prefix}obs.vector", "input", "slice",
                       offset=scalar_off),
                 _Slot(f"{prefix}hand_mlp", "output", "features"))
            if cnn is not None:
                convs = [m for m in cnn if m.__class__.__name__ == "Conv2d"]
                for i, conv in enumerate(convs):
                    slots = [_Slot(f"{prefix}cnn.{i}", "output", "channels")]
                    if i == 0:
                        slots.append(
                            _Slot(f"{prefix}obs.spatial", "input", "channels"))
                    want(conv, *slots)
            if cnn_proj is not None:
                want(cnn_proj[0], _Slot(f"{prefix}cnn_proj", "output", "features"))
            if set_enc is not None:
                want(set_enc.per_unit[0],
                     _Slot(f"{prefix}obs.spatial", "input", "channels"),
                     _Slot(f"{prefix}set.per_unit.0", "output", "entities"))
                want(set_enc.per_unit[2],
                     _Slot(f"{prefix}set.per_unit.2", "output", "entities"))
                want(set_enc.proj[0], _Slot(f"{prefix}set.proj", "output", "features"))
            if fusion is not None:
                want(fusion[0], _Slot(f"{prefix}fusion.0", "output", "features"))
                want(fusion[2], _Slot(f"{prefix}fusion.2", "output", "features"))

        if cfg.use_recurrence:
            want(net.gru, _Slot("gru", "output", "gru"))
        want(net.card_head, _Slot("card_head", "output", "features"))
        want(net.place_head[0], _Slot("place_head.0", "output", "features"))
        want(net.place_head[2], _Slot("place_head.2", "output", "features"))
        want(net.value_head, _Slot("value_head", "output", "features"))
        return plan

    def _make_hook(self, slots: list[_Slot]):
        def hook(module, inputs, output):
            for slot in slots:
                tensor = (inputs[slot.index] if slot.source == "input"
                          else output)
                vec = _reduce(tensor, slot)
                if vec is not None:
                    self._latest[slot.layer] = vec
        return hook

    # ------------------------------------------------------- activations

    def activation_frame(self) -> list[float] | None:
        """Per-node activation in 0..1, or None if nothing has run yet."""
        if not self._latest:
            return None
        out = np.zeros(self.n_nodes, np.float32)
        for key, vec in self._latest.items():
            pairs = self._node_of.get(key)
            if not pairs:
                continue
            peak = max(float(np.abs(vec).max()), _EPS)
            running = max(peak, self._peak.get(key, 0.0) * _PEAK_DECAY)
            self._peak[key] = running
            for node_id, unit in pairs:
                if unit < vec.shape[0]:
                    out[node_id] = abs(float(vec[unit])) / running
        return [round(float(v), 3) for v in np.clip(out, 0.0, 1.0)]

    # ---------------------------------------------------- weight movement

    def _snapshot_weights(self) -> None:
        """Reference copy of every weight matrix a drawn layer owns."""
        import torch

        from src.viz.graph import describe_layers

        with torch.no_grad():
            for layer in describe_layers(self.net):
                module = layer.module
                if module is None or not hasattr(module, "weight"):
                    weight = getattr(module, "weight_ih_l0", None)
                    if weight is None:
                        continue
                else:
                    weight = module.weight
                detached = weight.detach()
                # Full clone, not just row norms: two weight vectors can have
                # identical norms and point in completely different
                # directions, and rotation is exactly what "this unit learned
                # something else" looks like.
                self._tracked[layer.key] = (
                    weight, detached.clone(), _row_norms(detached).clone())
        self._update_maturity()

    def _update_maturity(self) -> None:
        """A static "how developed is this network" read.

        Row-norm dispersion within a layer: at initialisation every unit has
        a near-identical norm, and training pulls them apart. It is a weaker
        signal than movement, but unlike movement it survives a restart, so a
        loaded checkpoint renders as the trained network it is.
        """
        import torch

        with torch.no_grad():
            for key, (weight, _, _) in self._tracked.items():
                pairs = self._node_of.get(key)
                if not pairs:
                    continue
                rows = _row_norms(weight.detach())
                median = float(rows.median()) or 1.0
                spread = (rows / median - 1.0).abs()
                scale = max(float(spread.max()), _EPS)
                # Row-norm dispersion is heavily right-skewed — a handful of
                # outlier units dominate and everything else compresses to
                # near zero, which renders a trained network as an empty
                # skeleton. The square root spreads the mid-range back out.
                # Display scaling only; the ordering is untouched.
                for node_id, unit in pairs:
                    if unit < rows.shape[0]:
                        self._maturity[node_id] = min(
                            1.0, math.sqrt(float(spread[unit]) / scale))

    def learning_frame(self) -> dict:
        """Weight travel since attach: `delta` (recent) and `reveal` (cumulative)."""
        import torch

        delta = np.zeros(self.n_nodes, np.float32)
        with torch.no_grad():
            for key, (weight, reference, ref_rows) in self._tracked.items():
                pairs = self._node_of.get(key)
                if not pairs:
                    continue
                moved_rows = _row_norms(weight.detach() - reference)
                rel = moved_rows / (ref_rows + _EPS)
                for node_id, unit in pairs:
                    if unit < rel.shape[0]:
                        delta[node_id] = float(rel[unit])
        scaled = np.clip(delta / _REVEAL_FULL, 0.0, 1.0)
        # Reveal only ever grows: it records that a unit *has been* reshaped,
        # not that it is being reshaped right now.
        self._reveal = np.maximum(self._reveal, scaled)
        self._update_maturity()
        return {
            "delta": [round(float(v), 4) for v in scaled],
            "reveal": [round(float(v), 4) for v in self._reveal],
            "maturity": [round(float(v), 3) for v in self._maturity],
        }

    def reset_reference(self) -> None:
        """Re-baseline movement to the current weights."""
        self._tracked.clear()
        self._reveal[:] = 0.0
        self._snapshot_weights()


# --------------------------------------------------------------- helpers


def _row_norms(weight):
    """L2 norm per output unit, for 2D and 4D (conv) weights alike."""
    flat = weight.reshape(weight.shape[0], -1)
    return flat.norm(dim=1)


def _reduce(tensor, slot: _Slot):
    """Collapse a hooked tensor to one non-negative value per unit."""
    import torch

    if isinstance(tensor, tuple):
        tensor = tensor[0]
    if not torch.is_tensor(tensor):
        return None
    with torch.no_grad():
        x = tensor.detach().float()
        if x.dtype in (torch.int64, torch.int32):
            x = x.float()
        if slot.reduce == "channels" and x.dim() == 4:      # (B, C, H, W)
            return x.abs().mean(dim=(0, 2, 3)).cpu().numpy()
        if slot.reduce == "channels" and x.dim() == 3:      # set-encoder input
            return x.abs().mean(dim=(0, 1)).cpu().numpy()
        if slot.reduce == "entities" and x.dim() == 3:      # (B, N, H)
            return x.abs().mean(dim=(0, 1)).cpu().numpy()
        if slot.reduce == "norm_last" and x.dim() == 3:     # (B, slots, embed)
            return x.norm(dim=2).mean(dim=0).cpu().numpy()
        if slot.reduce == "norm_mid" and x.dim() == 3:
            return x.norm(dim=1).mean(dim=0).cpu().numpy()
        if slot.reduce == "gru":
            if x.dim() == 3:
                x = x.squeeze(0)
            return x.abs().mean(dim=0).cpu().numpy()
        if slot.reduce == "slice":
            x = x[..., slot.offset:]
        if x.dim() >= 2:
            return x.abs().mean(dim=tuple(range(x.dim() - 1))).cpu().numpy()
        return x.abs().cpu().numpy()
