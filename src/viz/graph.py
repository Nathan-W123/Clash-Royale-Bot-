"""Turn a live `PolicyNetwork` into a 3D node/edge scene.

The graph is *derived from the real module tree*, not hand-drawn. If someone
flips `use_set_encoder`, enables recurrence, or attaches an asymmetric
critic, the picture changes because the network changed — a diagram that had
to be maintained by hand would be lying within a week.

Two honesty constraints worth stating up front, because both are visible in
the UI:

**Nodes are sampled.** A 256-unit layer is drawn with at most
`MAX_NODES_PER_LAYER` nodes, evenly strided. Every node is a real unit with
a real index; there are just more units than dots. Each layer reports
`size` (true) alongside `shown`.

**Edges are the strongest few, not all of them.** 256x256 fully connected is
65k lines per layer pair and renders as an opaque grey slab. We keep the
top-`EDGES_PER_NODE` incoming connections per drawn node, ranked by |weight|.
That is a real and interpretable subset — the dominant pathways — but it is
a subset, and the UI says so.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_NODES_PER_LAYER = 48
EDGES_PER_NODE = 3
MAX_EDGES = 9000

# Layout constants (world units).
DEPTH_SPACING = 8.0
NODE_SPACING = 1.15
BRANCH_Y = {"spatial": 9.5, "hand": -9.5, "trunk": 0.0,
            "head_card": 7.0, "head_place": 0.0, "head_value": -7.5}
CRITIC_Z = -20.0


@dataclass
class Edge:
    """One drawn connection. `src`/`dst` index into the flat node list."""
    src: int
    dst: int
    weight: float


@dataclass
class Layer:
    key: str
    label: str
    kind: str            # input | conv | linear | embed | gru | head
    branch: str
    size: int            # true unit count
    detail: str = ""
    module: Any = None
    shape: tuple[int, int] | None = None   # lay out as a real 2D grid
    inputs: list["LayerLink"] = field(default_factory=list)
    depth: int = 0
    shown: list[int] = field(default_factory=list)   # sampled unit indices
    node_ids: list[int] = field(default_factory=list)
    critic: bool = False


@dataclass
class LayerLink:
    """An incoming connection, plus how to find its weights.

    `col_map` turns a *source unit index* into the weight-matrix columns it
    occupies. The default is a straight offset, but flattened concatenations
    need more: `hand_mlp` eats `[5 slots x 16 embed dims] ++ [scalar_dim]`, so
    embedding dimension `d` lives in columns `{s * 16 + d}` for every slot
    `s`, and the scalar block starts at column 80.
    """
    src: str
    col_map: Callable[[int], Sequence[int]] | None = None
    gate_rows: int = 1   # GRU packs 3 gates into one weight matrix


def _torch():
    import torch
    return torch


def _sample(n: int, cap: int = MAX_NODES_PER_LAYER) -> list[int]:
    """Evenly strided unit indices, always including first and last."""
    if n <= cap:
        return list(range(n))
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


def _offset_map(offset: int) -> Callable[[int], Sequence[int]]:
    return lambda j: (offset + j,)


def _embed_slot_map(embed_dim: int, n_slots: int) -> Callable[[int], Sequence[int]]:
    return lambda d: tuple(s * embed_dim + d for s in range(n_slots))


def describe_layers(net) -> list[Layer]:
    """The ordered layer list for this exact network configuration."""
    from src.simulator.constants import HAND_SIZE

    cfg = net.config
    n_slots = HAND_SIZE + 1
    embed_dim = cfg.card_embed_dim
    layers: list[Layer] = []

    def add(layer: Layer) -> Layer:
        layers.append(layer)
        return layer

    spatial = cfg.use_spatial
    arena_out_key: str | None = None

    if spatial:
        add(Layer("obs.spatial", "arena grid", "input", "spatial",
                  size=_spatial_channels(), detail="10ch x 16 x 9",
                  module=None))
        if cfg.use_set_encoder:
            enc = net.set_encoder
            add(Layer("set.per_unit.0", "entity MLP", "linear", "spatial",
                      size=cfg.unit_hidden, module=enc.per_unit[0],
                      detail="deep-sets, per entity",
                      inputs=[LayerLink("obs.spatial")]))
            add(Layer("set.per_unit.2", "entity MLP", "linear", "spatial",
                      size=cfg.unit_hidden, module=enc.per_unit[2],
                      inputs=[LayerLink("set.per_unit.0")]))
            add(Layer("set.proj", "set pool -> proj", "linear", "spatial",
                      size=cfg.cnn_out, module=enc.proj[0],
                      detail="mean ++ max pooling",
                      inputs=[LayerLink("set.per_unit.2")]))
            arena_out_key = "set.proj"
        else:
            prev = "obs.spatial"
            for i, conv in enumerate(m for m in net.cnn if _is_conv(m)):
                key = f"cnn.{i}"
                add(Layer(key, f"conv{i + 1} 3x3", "conv", "spatial",
                          size=conv.out_channels, module=conv,
                          detail=f"{conv.in_channels}->{conv.out_channels}",
                          inputs=[LayerLink(prev)]))
                prev = key
            add(Layer("cnn_proj", "cnn projection", "linear", "spatial",
                      size=cfg.cnn_out, module=net.cnn_proj[0],
                      detail="flatten -> dense",
                      inputs=[LayerLink(prev)]))
            arena_out_key = "cnn_proj"

    add(Layer("obs.cards", "hand slots", "input", "hand", size=n_slots,
              detail="4 in hand + next"))
    add(Layer("card_embed", "card embedding", "embed", "hand", size=embed_dim,
              module=net.card_embed, detail=f"{cfg.n_cards} cards -> {embed_dim}d",
              inputs=[LayerLink("obs.cards")]))
    add(Layer("obs.vector", "scalars", "input", "hand", size=cfg.scalar_dim,
              detail=f"{cfg.tier} tier"))
    add(Layer("hand_mlp", "hand + scalar MLP", "linear", "hand",
              size=cfg.hand_mlp, module=net.hand_mlp[0],
              inputs=[
                  LayerLink("card_embed", _embed_slot_map(embed_dim, n_slots)),
                  LayerLink("obs.vector", _offset_map(n_slots * embed_dim)),
              ]))

    fusion_inputs = [LayerLink("hand_mlp", _offset_map(cfg.cnn_out if spatial else 0))]
    if arena_out_key:
        fusion_inputs.insert(0, LayerLink(arena_out_key, _offset_map(0)))
    add(Layer("fusion.0", "fusion", "linear", "trunk", size=cfg.fusion_mlp,
              module=net.fusion[0], detail="arena ++ hand", inputs=fusion_inputs))
    add(Layer("fusion.2", "fusion", "linear", "trunk", size=cfg.fusion_mlp,
              module=net.fusion[2], inputs=[LayerLink("fusion.0")]))

    trunk_out = "fusion.2"
    if cfg.use_recurrence:
        add(Layer("gru", "GRU memory", "gru", "trunk", size=cfg.hidden_size,
                  module=net.gru, detail="carries history across steps",
                  inputs=[LayerLink("fusion.2", gate_rows=3)]))
        trunk_out = "gru"

    add(Layer("card_head", "card choice", "head", "head_card", size=_n_card_choices(),
              module=net.card_head, detail="no-op + 4 slots",
              inputs=[LayerLink(trunk_out)]))
    add(Layer("place_head.0", "placement hidden", "linear", "head_place",
              size=cfg.fusion_mlp, module=net.place_head[0],
              detail="conditioned on chosen card",
              inputs=[LayerLink(trunk_out, _offset_map(0)),
                      LayerLink("card_embed", _offset_map(cfg.fusion_mlp))]))
    rows, cols = _place_shape()
    add(Layer("place_head.2", "placement grid", "head", "head_place",
              size=rows * cols, module=net.place_head[2], shape=(rows, cols),
              detail=f"{cols} x {rows} cells",
              inputs=[LayerLink("place_head.0")]))
    add(Layer("value_head", "value", "head", "head_value", size=1,
              module=net.value_head, detail="expected return",
              inputs=[LayerLink(trunk_out)]))

    if cfg.asymmetric:
        layers.extend(_critic_layers(net))
    return layers


def _critic_layers(net) -> list[Layer]:
    """The privileged value trunk, drawn on its own plane behind the actor.

    It is deliberately a *separate* island: no tensor it computes reaches the
    policy heads, and the picture should make that obvious at a glance.
    """
    from src.agent.obs_layout import tier_uses_spatial
    from src.simulator.constants import HAND_SIZE

    cfg = net.config
    n_slots = HAND_SIZE + 1
    embed_dim = cfg.card_embed_dim
    out: list[Layer] = []
    spatial = tier_uses_spatial(cfg.critic_tier)
    arena_key = None

    if spatial:
        out.append(Layer("c.obs.spatial", "arena grid", "input", "spatial",
                         size=_spatial_channels(), critic=True,
                         detail=f"{cfg.critic_tier} tier"))
        if cfg.use_set_encoder:
            out.append(Layer("c.set.proj", "set encoder", "linear", "spatial",
                             size=cfg.cnn_out, module=net.critic_set_encoder.proj[0],
                             critic=True, inputs=[LayerLink("c.obs.spatial")]))
            arena_key = "c.set.proj"
        else:
            prev = "c.obs.spatial"
            for i, conv in enumerate(m for m in net.critic_cnn if _is_conv(m)):
                key = f"c.cnn.{i}"
                out.append(Layer(key, f"conv{i + 1}", "conv", "spatial",
                                 size=conv.out_channels, module=conv, critic=True,
                                 inputs=[LayerLink(prev)]))
                prev = key
            out.append(Layer("c.cnn_proj", "cnn projection", "linear", "spatial",
                             size=cfg.cnn_out, module=net.critic_cnn_proj[0],
                             critic=True, inputs=[LayerLink(prev)]))
            arena_key = "c.cnn_proj"

    out.append(Layer("c.obs.cards", "hand slots", "input", "hand",
                     size=n_slots, critic=True))
    out.append(Layer("c.card_embed", "card embedding", "embed", "hand",
                     size=embed_dim, module=net.critic_card_embed, critic=True,
                     inputs=[LayerLink("c.obs.cards")]))
    out.append(Layer("c.obs.vector", "scalars", "input", "hand",
                     size=cfg.critic_scalar_dim, critic=True,
                     detail="privileged"))
    out.append(Layer("c.hand_mlp", "hand + scalar MLP", "linear", "hand",
                     size=cfg.hand_mlp, module=net.critic_hand_mlp[0], critic=True,
                     inputs=[LayerLink("c.card_embed", _embed_slot_map(embed_dim, n_slots)),
                             LayerLink("c.obs.vector", _offset_map(n_slots * embed_dim))]))
    fin = [LayerLink("c.hand_mlp", _offset_map(cfg.cnn_out if spatial else 0))]
    if arena_key:
        fin.insert(0, LayerLink(arena_key, _offset_map(0)))
    out.append(Layer("c.fusion.0", "critic fusion", "linear", "trunk",
                     size=cfg.fusion_mlp, module=net.critic_fusion[0], critic=True,
                     inputs=fin))
    out.append(Layer("c.fusion.2", "critic fusion", "linear", "trunk",
                     size=cfg.fusion_mlp, module=net.critic_fusion[2], critic=True,
                     inputs=[LayerLink("c.fusion.0")]))
    return out


def _is_conv(module) -> bool:
    import torch.nn as nn
    return isinstance(module, nn.Conv2d)


def _spatial_channels() -> int:
    from src.agent.obs_layout import SPATIAL_CHANNELS
    return SPATIAL_CHANNELS


def _n_card_choices() -> int:
    from src.agent.network import N_CARD_CHOICES
    return N_CARD_CHOICES


def _place_shape() -> tuple[int, int]:
    from src.simulator.constants import PLACE_COLS, PLACE_ROWS
    return PLACE_ROWS, PLACE_COLS


def _assign_depths(layers: list[Layer]) -> None:
    """Longest-path depth, so merging branches land in the same column."""
    by_key = {layer.key: layer for layer in layers}
    resolved: dict[str, int] = {}

    def depth_of(key: str, seen: frozenset[str] = frozenset()) -> int:
        if key in resolved:
            return resolved[key]
        layer = by_key[key]
        if key in seen:   # defensive: the architecture is a DAG today
            return 0
        if not layer.inputs:
            d = 0
        else:
            d = 1 + max(depth_of(link.src, seen | {key})
                        for link in layer.inputs if link.src in by_key)
        resolved[key] = d
        return d

    for layer in layers:
        layer.depth = depth_of(layer.key)


def _layout(layers: list[Layer]) -> list[dict]:
    """Place sampled units in 3D and produce the flat node list."""
    nodes: list[dict] = []
    for layer in layers:
        layer.shown = _sample(layer.size)
        n = len(layer.shown)
        if layer.shape and n == layer.size:
            rows, cols = layer.shape
        else:
            cols = max(1, math.ceil(math.sqrt(n)))
            rows = math.ceil(n / cols)
        x = layer.depth * DEPTH_SPACING
        y0 = BRANCH_Y.get(layer.branch, 0.0)
        z0 = CRITIC_Z if layer.critic else 0.0
        w = (cols - 1) * NODE_SPACING
        h = (rows - 1) * NODE_SPACING
        layer.node_ids = []
        for i, unit in enumerate(layer.shown):
            r, c = divmod(i, cols)
            node_id = len(nodes)
            layer.node_ids.append(node_id)
            nodes.append({
                "id": node_id,
                "layer": layer.key,
                "unit": unit,
                "x": round(x, 3),
                "y": round(y0 + h / 2 - r * NODE_SPACING, 3),
                "z": round(z0 - w / 2 + c * NODE_SPACING, 3),
                "kind": layer.kind,
                "branch": layer.branch,
                "critic": layer.critic,
            })
    return nodes


def _weight_matrix(layer: Layer, link: LayerLink):
    """|W| reduced to a 2D [out_units, in_columns] tensor, or None."""
    torch = _torch()
    module = layer.module
    if module is None or not hasattr(module, "weight"):
        if layer.kind == "gru":
            w = getattr(module, "weight_ih_l0", None)
            if w is None:
                return None
            gates = w.detach().abs().reshape(link.gate_rows, -1, w.shape[1])
            return gates.max(dim=0).values
        return None
    w = module.weight.detach()
    if w.dim() == 4:                       # conv: [out, in, kh, kw]
        return w.abs().flatten(2).norm(dim=2)
    if w.dim() == 2:
        return w.abs()
    return None


def _build_edges(layers: list[Layer]) -> list[Edge]:
    torch = _torch()
    by_key = {layer.key: layer for layer in layers}
    edges: list[Edge] = []

    with torch.no_grad():
        for layer in layers:
            for link in layer.inputs:
                src = by_key.get(link.src)
                if src is None or not src.node_ids:
                    continue
                mat = _weight_matrix(layer, link)
                col_map = link.col_map or _offset_map(0)

                if mat is None:
                    # No weight matrix relates these (embedding lookup, or a
                    # pseudo-input). The relationship is a full copy, so draw
                    # it as one — uniformly weighted.
                    for d, dst_id in enumerate(layer.node_ids):
                        for s_id in src.node_ids[:EDGES_PER_NODE * 2]:
                            edges.append(Edge(s_id, dst_id, 0.5))
                    continue

                n_cols = mat.shape[1]
                cols, valid_src = [], []
                for j, unit in enumerate(src.shown):
                    mapped = [c for c in col_map(unit) if 0 <= c < n_cols]
                    if mapped:
                        cols.append(mapped)
                        valid_src.append(j)
                if not cols:
                    continue

                # Reduce each source unit's columns to a single score per
                # target row, then keep the top-K sources for that row.
                scores = torch.stack(
                    [mat[:, c].max(dim=1).values for c in cols], dim=1)
                rows = torch.as_tensor(
                    [u for u in layer.shown], dtype=torch.long)
                rows = rows.clamp(max=scores.shape[0] - 1)
                sub = scores.index_select(0, rows)
                k = min(EDGES_PER_NODE, sub.shape[1])
                top = sub.topk(k, dim=1)
                peak = float(sub.max()) or 1.0
                for d, dst_id in enumerate(layer.node_ids):
                    for slot in range(k):
                        j = valid_src[int(top.indices[d, slot])]
                        edges.append(Edge(src.node_ids[j], dst_id,
                                          round(float(top.values[d, slot]) / peak, 4)))
    if len(edges) > MAX_EDGES:
        edges.sort(key=lambda e: e.weight, reverse=True)
        del edges[MAX_EDGES:]
    return edges


def build_graph(net) -> dict:
    """The full scene description sent to the browser once per connection."""
    layers = describe_layers(net)
    _assign_depths(layers)
    nodes = _layout(layers)
    edges = _build_edges(layers)
    cfg = net.config
    n_params = sum(p.numel() for p in net.parameters())

    return {
        "nodes": nodes,
        "edges": [[e.src, e.dst, e.weight] for e in edges],
        "layers": [{
            "key": layer.key,
            "label": layer.label,
            "detail": layer.detail,
            "kind": layer.kind,
            "branch": layer.branch,
            "depth": layer.depth,
            "size": layer.size,
            "shown": len(layer.shown),
            "critic": layer.critic,
            "nodes": layer.node_ids,
        } for layer in layers],
        "meta": {
            "tier": cfg.tier,
            "critic_tier": cfg.critic_tier,
            "recurrent": cfg.use_recurrence,
            "set_encoder": cfg.use_set_encoder,
            "n_cards": cfg.n_cards,
            "params": n_params,
            "sampled": any(len(layer.shown) < layer.size for layer in layers),
            "edges_per_node": EDGES_PER_NODE,
            "arch": ("set-encoder" if cfg.use_set_encoder
                     else "cnn" if cfg.use_spatial else "scalar-only"),
        },
    }
