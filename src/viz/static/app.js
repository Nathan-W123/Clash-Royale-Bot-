/* 3D policy-network viewer.
 *
 * Two signals drive the picture, and they are deliberately different:
 *
 *   activation ("act")  — what fired on the last forward pass. Drives the
 *                         travelling pulses along edges and the flare on
 *                         nodes. Fast attack, slow release, so a firing unit
 *                         fades in and out rather than strobing.
 *
 *   reveal ("learn")    — how far a unit's weights have moved since the
 *                         probe attached. Monotonic. Drives node size and
 *                         opacity, so an untrained network starts as a faint
 *                         skeleton and fills in as training reshapes it.
 *
 * The network does not gain units while training; nothing here pretends it
 * does. What grows is how much of it has demonstrably changed.
 */
'use strict';

const BRANCH_COLOR = {
  spatial:    [0.22, 0.82, 1.00],
  hand:       [1.00, 0.70, 0.33],
  trunk:      [0.66, 0.49, 1.00],
  head_card:  [0.36, 0.95, 0.66],
  head_place: [1.00, 0.48, 0.72],
  head_value: [1.00, 0.85, 0.40],
};
const CRITIC_COLOR = [0.36, 0.45, 0.62];

const BRANCH_LABEL = {
  spatial: 'arena / spatial',
  hand: 'hand + scalars',
  trunk: 'shared trunk',
  head_card: 'card head',
  head_place: 'placement head',
  head_value: 'value head',
};

/* Snapshot mode (`?snapshot`): draw the last known state, then stop
 * streaming. A live SSE connection never goes idle, which means headless
 * screenshot tools (and anything else waiting on network-idle) hang forever
 * on this page. It also holds activations instead of fading them out, so a
 * still shows the network mid-pulse rather than at rest. */
const SNAPSHOT = new URLSearchParams(location.search).has('snapshot');

/* ------------------------------------------------------------------ state */

const S = {
  graph: null,
  nodeCount: 0,
  act: null, actTarget: null,
  reveal: null, revealTarget: null,
  edges: null,
  mode: 'idle',
  stats: {},
  labelsOn: true,
  lastFrameAt: 0,
  hover: -1,
};

const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
if (!gl) {
  document.body.innerHTML =
    '<p style="padding:24px">WebGL2 is unavailable in this browser.</p>';
  throw new Error('no webgl2');
}

/* ---------------------------------------------------------------- shaders */

const NODE_VS = `#version 300 es
in vec3 aPos;
in vec3 aColor;
in float aAct;
in float aReveal;
uniform mat4 uMVP;
uniform float uPointScale;
out vec3 vColor;
out float vAct;
out float vReveal;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  vColor = aColor;
  vAct = aAct;
  vReveal = aReveal;
  // Unrevealed units stay small but never vanish: the skeleton of the
  // architecture should always be readable.
  float grow = mix(0.42, 1.0, aReveal);
  float flare = 1.0 + aAct * 1.35;
  float s = uPointScale * grow * flare / max(gl_Position.w, 0.6);
  gl_PointSize = clamp(s, 1.5, 70.0);
}`;

const NODE_FS = `#version 300 es
precision highp float;
in vec3 vColor;
in float vAct;
in float vReveal;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r = length(d) * 2.0;
  if (r > 1.0) discard;
  float core = smoothstep(1.0, 0.35, r);
  float halo = smoothstep(1.0, 0.0, r);
  vec3 hot = mix(vColor, vec3(1.0), 0.72);
  vec3 col = mix(vColor * 0.62, hot, vAct);
  float a = (0.22 + 0.78 * vReveal) * (halo * 0.40 + core * 0.95);
  a += vAct * 0.40 * halo;
  fragColor = vec4(col, clamp(a, 0.0, 1.0));
}`;

const EDGE_VS = `#version 300 es
in vec3 aPos;
in vec3 aColor;
in float aT;
in float aAct;
in float aReveal;
in float aWeight;
in float aPhase;
uniform mat4 uMVP;
out vec3 vColor;
out float vT;
out float vAct;
out float vReveal;
out float vWeight;
out float vPhase;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  vColor = aColor; vT = aT; vAct = aAct;
  vReveal = aReveal; vWeight = aWeight; vPhase = aPhase;
}`;

const EDGE_FS = `#version 300 es
precision highp float;
in vec3 vColor;
in float vT;
in float vAct;
in float vReveal;
in float vWeight;
in float vPhase;
uniform float uTime;
uniform float uFlow;
out vec4 fragColor;
void main() {
  // A bright head travelling source -> target. Each edge carries its own
  // phase so the whole graph does not pulse in lockstep, which reads as a
  // flashing screen rather than as flow.
  float head = fract(uTime * uFlow + vPhase);
  float d = abs(vT - head);
  d = min(d, 1.0 - d);
  float pulse = exp(-d * d * 260.0);
  float rest = (0.030 + 0.075 * vWeight) * (0.25 + 0.75 * vReveal);
  float a = rest + pulse * vAct * 0.95;
  vec3 col = mix(vColor * 0.55, mix(vColor, vec3(1.0), 0.6), vAct);
  fragColor = vec4(col, clamp(a, 0.0, 1.0));
}`;

const nodeProg = GLU.program(gl, NODE_VS, NODE_FS);
const edgeProg = GLU.program(gl, EDGE_VS, EDGE_FS);

/* ------------------------------------------------------------------ scene */

const buffers = {};
let nodeVAO = null, edgeVAO = null, edgeVertexCount = 0;

function buildScene(graph) {
  S.graph = graph;
  const nodes = graph.nodes;
  const n = nodes.length;
  S.nodeCount = n;

  S.act = new Float32Array(n);
  S.actTarget = new Float32Array(n);
  S.reveal = new Float32Array(n);
  S.revealTarget = new Float32Array(n);

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const nd = nodes[i];
    pos[i * 3] = nd.x; pos[i * 3 + 1] = nd.y; pos[i * 3 + 2] = nd.z;
    const c = nd.critic ? CRITIC_COLOR : (BRANCH_COLOR[nd.branch] || [0.6, 0.7, 0.9]);
    col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
  }

  const e = graph.edges;
  const m = e.length;
  edgeVertexCount = m * 2;
  const epos = new Float32Array(m * 6);
  const ecol = new Float32Array(m * 6);
  const et = new Float32Array(m * 2);
  const ew = new Float32Array(m * 2);
  const eph = new Float32Array(m * 2);
  S.edges = e;

  for (let i = 0; i < m; i++) {
    const [a, b, w] = e[i];
    const na = nodes[a], nb = nodes[b];
    epos.set([na.x, na.y, na.z, nb.x, nb.y, nb.z], i * 6);
    const c = nb.critic ? CRITIC_COLOR : (BRANCH_COLOR[nb.branch] || [0.6, 0.7, 0.9]);
    ecol.set([c[0], c[1], c[2], c[0], c[1], c[2]], i * 6);
    et[i * 2] = 0; et[i * 2 + 1] = 1;
    ew[i * 2] = w; ew[i * 2 + 1] = w;
    const phase = (i * 0.6180339887) % 1;
    eph[i * 2] = phase; eph[i * 2 + 1] = phase;
  }

  buffers.nodePos = GLU.buffer(gl, pos);
  buffers.nodeCol = GLU.buffer(gl, col);
  buffers.nodeAct = GLU.buffer(gl, S.act, gl.DYNAMIC_DRAW);
  buffers.nodeRev = GLU.buffer(gl, S.reveal, gl.DYNAMIC_DRAW);

  buffers.edgePos = GLU.buffer(gl, epos);
  buffers.edgeCol = GLU.buffer(gl, ecol);
  buffers.edgeT = GLU.buffer(gl, et);
  buffers.edgeW = GLU.buffer(gl, ew);
  buffers.edgePh = GLU.buffer(gl, eph);
  buffers.edgeActData = new Float32Array(m * 2);
  buffers.edgeRevData = new Float32Array(m * 2);
  buffers.edgeAct = GLU.buffer(gl, buffers.edgeActData, gl.DYNAMIC_DRAW);
  buffers.edgeRev = GLU.buffer(gl, buffers.edgeRevData, gl.DYNAMIC_DRAW);

  nodeVAO = gl.createVertexArray();
  gl.bindVertexArray(nodeVAO);
  GLU.attrib(gl, nodeProg, 'aPos', buffers.nodePos, 3);
  GLU.attrib(gl, nodeProg, 'aColor', buffers.nodeCol, 3);
  GLU.attrib(gl, nodeProg, 'aAct', buffers.nodeAct, 1);
  GLU.attrib(gl, nodeProg, 'aReveal', buffers.nodeRev, 1);

  edgeVAO = gl.createVertexArray();
  gl.bindVertexArray(edgeVAO);
  GLU.attrib(gl, edgeProg, 'aPos', buffers.edgePos, 3);
  GLU.attrib(gl, edgeProg, 'aColor', buffers.edgeCol, 3);
  GLU.attrib(gl, edgeProg, 'aT', buffers.edgeT, 1);
  GLU.attrib(gl, edgeProg, 'aWeight', buffers.edgeW, 1);
  GLU.attrib(gl, edgeProg, 'aPhase', buffers.edgePh, 1);
  GLU.attrib(gl, edgeProg, 'aAct', buffers.edgeAct, 1);
  GLU.attrib(gl, edgeProg, 'aReveal', buffers.edgeRev, 1);
  gl.bindVertexArray(null);

  frameCamera(nodes);
  buildLegend(graph);
  buildLabels(graph);
  document.getElementById('arch-label').textContent =
    `${graph.meta.arch} · ${graph.meta.tier} tier · ` +
    `${graph.meta.params.toLocaleString()} params`;
}

/* ----------------------------------------------------------------- camera */

const cam = { tx: 0, ty: 0, tz: 0, yaw: -0.52, pitch: 0.17, dist: 90 };
const keys = new Set();

function frameCamera(nodes) {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity;
  let x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (const n of nodes) {
    x0 = Math.min(x0, n.x); x1 = Math.max(x1, n.x);
    y0 = Math.min(y0, n.y); y1 = Math.max(y1, n.y);
    z0 = Math.min(z0, n.z); z1 = Math.max(z1, n.z);
  }
  cam.tx = (x0 + x1) / 2; cam.ty = (y0 + y1) / 2; cam.tz = (z0 + z1) / 2;
  cam.dist = Math.max(40, Math.hypot(x1 - x0, y1 - y0, z1 - z0) * 0.92);
  cam.yaw = -0.52; cam.pitch = 0.13;
}

function eyePosition() {
  const cp = Math.cos(cam.pitch);
  return [
    cam.tx + cam.dist * cp * Math.sin(cam.yaw),
    cam.ty + cam.dist * Math.sin(cam.pitch),
    cam.tz + cam.dist * cp * Math.cos(cam.yaw),
  ];
}

const view = M4.create(), proj = M4.create(), mvp = M4.create();

function updateMatrices() {
  const aspect = canvas.width / Math.max(canvas.height, 1);
  M4.perspective(proj, Math.PI / 4, aspect, 0.5, 4000);
  M4.lookAt(view, eyePosition(), [cam.tx, cam.ty, cam.tz], [0, 1, 0]);
  M4.multiply(mvp, proj, view);
}

/* Mouse: left-drag orbits, right-drag (or shift) pans, wheel dollies. */
let dragging = null, lastX = 0, lastY = 0;

canvas.addEventListener('mousedown', (ev) => {
  dragging = (ev.button === 2 || ev.shiftKey) ? 'pan' : 'orbit';
  lastX = ev.clientX; lastY = ev.clientY;
  canvas.classList.add('dragging');
  ev.preventDefault();
});

window.addEventListener('mouseup', () => {
  dragging = null;
  canvas.classList.remove('dragging');
});

window.addEventListener('mousemove', (ev) => {
  if (!dragging) { hoverTest(ev); return; }
  const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
  lastX = ev.clientX; lastY = ev.clientY;
  if (dragging === 'orbit') {
    cam.yaw -= dx * 0.006;
    // Clamped just short of the poles: at exactly +/-90 the up vector and
    // the view direction become parallel and lookAt degenerates.
    cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.005));
  } else {
    const k = cam.dist * 0.0016;
    const right = [Math.cos(cam.yaw), 0, -Math.sin(cam.yaw)];
    const up = [
      -Math.sin(cam.pitch) * Math.sin(cam.yaw),
      Math.cos(cam.pitch),
      -Math.sin(cam.pitch) * Math.cos(cam.yaw),
    ];
    cam.tx -= (right[0] * dx - up[0] * dy) * k;
    cam.ty -= (right[1] * dx - up[1] * dy) * k;
    cam.tz -= (right[2] * dx - up[2] * dy) * k;
  }
});

canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

canvas.addEventListener('wheel', (ev) => {
  cam.dist = Math.max(6, Math.min(1200, cam.dist * Math.exp(ev.deltaY * 0.0012)));
  ev.preventDefault();
}, { passive: false });

window.addEventListener('keydown', (ev) => {
  const tag = (ev.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea') return;
  const k = ev.key.toLowerCase();
  keys.add(k);
  if (k === 'r' && S.graph) frameCamera(S.graph.nodes);
  if (k === 'f') {
    S.labelsOn = !S.labelsOn;
    document.getElementById('labels').style.display = S.labelsOn ? '' : 'none';
  }
});
window.addEventListener('keyup', (ev) => keys.delete(ev.key.toLowerCase()));
window.addEventListener('blur', () => keys.clear());

function flyStep(dt) {
  if (!keys.size) return;
  const speed = cam.dist * 0.9 * dt;
  const fwd = [
    -Math.cos(cam.pitch) * Math.sin(cam.yaw),
    -Math.sin(cam.pitch),
    -Math.cos(cam.pitch) * Math.cos(cam.yaw),
  ];
  const right = [Math.cos(cam.yaw), 0, -Math.sin(cam.yaw)];
  let mx = 0, my = 0, mz = 0;
  if (keys.has('w')) { mx += fwd[0]; my += fwd[1]; mz += fwd[2]; }
  if (keys.has('s')) { mx -= fwd[0]; my -= fwd[1]; mz -= fwd[2]; }
  if (keys.has('d')) { mx += right[0]; mz += right[2]; }
  if (keys.has('a')) { mx -= right[0]; mz -= right[2]; }
  if (keys.has('e')) my += 1;
  if (keys.has('q')) my -= 1;
  cam.tx += mx * speed; cam.ty += my * speed; cam.tz += mz * speed;
}

/* --------------------------------------------------------------- hovering */

const tooltip = document.getElementById('tooltip');

function hoverTest(ev) {
  if (!S.graph) return;
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  if (mx < 0 || my < 0 || mx > rect.width || my > rect.height) {
    tooltip.classList.remove('on'); S.hover = -1; return;
  }
  let best = -1, bestD = 14 * 14;
  const nodes = S.graph.nodes;
  for (let i = 0; i < nodes.length; i++) {
    const p = M4.project(mvp, nodes[i].x, nodes[i].y, nodes[i].z);
    if (!p) continue;
    const sx = (p[0] * 0.5 + 0.5) * rect.width;
    const sy = (1 - (p[1] * 0.5 + 0.5)) * rect.height;
    const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my);
    if (d < bestD) { bestD = d; best = i; }
  }
  S.hover = best;
  if (best < 0) { tooltip.classList.remove('on'); return; }

  const nd = nodes[best];
  const layer = S.graph.layers.find((l) => l.key === nd.layer);
  tooltip.innerHTML =
    `<b>${layer ? layer.label : nd.layer}</b> `
    + `<span class="k">unit ${nd.unit}</span><br>`
    + `<span class="k">act</span> ${S.act[best].toFixed(2)} `
    + `<span class="k">moved</span> ${(S.reveal[best] * 100).toFixed(0)}%`
    + (layer && layer.shown < layer.size
        ? `<br><span class="k">showing ${layer.shown} of ${layer.size}</span>` : '');
  tooltip.style.left = mx + 'px';
  tooltip.style.top = my + 'px';
  tooltip.classList.add('on');
}

/* ----------------------------------------------------------------- labels */

function buildLabels(graph) {
  let host = document.getElementById('labels');
  if (!host) {
    host = document.createElement('div');
    host.id = 'labels';
    Object.assign(host.style, {
      position: 'absolute', inset: '0', pointerEvents: 'none', zIndex: '2',
    });
    document.getElementById('stage').appendChild(host);
  }
  host.innerHTML = '';
  host.style.display = S.labelsOn ? '' : 'none';

  for (const layer of graph.layers) {
    if (!layer.nodes.length) continue;
    let x = 0, y = 0, z = 0, yMin = Infinity, yMax = -Infinity;
    for (const id of layer.nodes) {
      const n = graph.nodes[id];
      x += n.x; y += n.y; z += n.z;
      yMin = Math.min(yMin, n.y); yMax = Math.max(yMax, n.y);
    }
    const el = document.createElement('div');
    el.className = 'layer-label';
    el.innerHTML = `<span>${layer.label}</span><em>${layer.size}${
      layer.shown < layer.size ? ` · ${layer.shown} shown` : ''}</em>`;
    Object.assign(el.style, {
      position: 'absolute',
      transform: 'translate(-50%, -50%)',
      font: '10.5px ui-monospace, monospace',
      color: layer.critic ? '#8394b8' : '#cfe0f7',
      textShadow: '0 0 9px #06080f, 0 0 4px #06080f, 0 1px 2px #06080f',
      whiteSpace: 'nowrap', textAlign: 'center', lineHeight: '1.25',
    });
    el._w = [x / layer.nodes.length, y / layer.nodes.length, z / layer.nodes.length];
    // Push the label clear of its own nodes, outward from the trunk, and
    // stagger consecutive depths. Without the stagger, adjacent same-height
    // layers (the two `fusion` blocks, the two placement blocks) print their
    // labels on top of each other.
    const outward = (layer.branch === 'hand' || layer.branch === 'head_value') ? -1 : 1;
    const half = (yMax - yMin) / 2;
    el._off = outward * (half + 2.6 + (layer.depth % 2 ? 3.2 : 0));
    host.appendChild(el);
  }
  if (!document.getElementById('label-style')) {
    const st = document.createElement('style');
    st.id = 'label-style';
    st.textContent =
      '.layer-label em{display:block;font-style:normal;opacity:.55;font-size:9.5px}';
    document.head.appendChild(st);
  }
}

function positionLabels() {
  const host = document.getElementById('labels');
  if (!host || !S.labelsOn) return;
  const rect = canvas.getBoundingClientRect();
  for (const el of host.children) {
    const w = el._w;
    const p = M4.project(mvp, w[0], w[1] + el._off, w[2]);
    if (!p) { el.style.display = 'none'; continue; }
    el.style.display = '';
    el.style.left = ((p[0] * 0.5 + 0.5) * rect.width) + 'px';
    el.style.top = ((1 - (p[1] * 0.5 + 0.5)) * rect.height) + 'px';
    el.style.opacity = String(Math.max(0.42, Math.min(1.0, 150 / p[2])));
  }
}

/* ----------------------------------------------------------------- legend */

function buildLegend(graph) {
  const seen = new Map();
  for (const layer of graph.layers) {
    const key = layer.critic ? 'critic' : layer.branch;
    if (!seen.has(key)) seen.set(key, { units: 0, layers: 0 });
    const e = seen.get(key);
    e.units += layer.size; e.layers += 1;
  }
  const list = document.getElementById('legend-list');
  list.innerHTML = '';
  for (const [key, info] of seen) {
    const c = key === 'critic' ? CRITIC_COLOR : (BRANCH_COLOR[key] || [0.6, 0.7, 0.9]);
    const hex = `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`;
    const li = document.createElement('li');
    li.innerHTML =
      `<span class="swatch" style="background:${hex};color:${hex}"></span>`
      + `<b>${key === 'critic' ? 'privileged critic' : (BRANCH_LABEL[key] || key)}</b>`
      + `<span class="count">${info.units.toLocaleString()} units</span>`;
    list.appendChild(li);
  }
  const note = document.getElementById('sampling-note');
  note.textContent =
    `Nodes are sampled: layers wider than 48 units are drawn with 48 evenly `
    + `spaced real units. Edges are the ${graph.meta.edges_per_node} strongest `
    + `incoming weights per drawn node, not all connections.`;
}

/* ------------------------------------------------------------------ stats */

const STAT_LABEL = {
  step: 'step', update: 'update', win_rate: 'win rate', reward: 'reward',
  entropy: 'entropy', policy_loss: 'policy loss', value_loss: 'value loss',
  sps: 'steps/sec', revealed: 'weights moved', value: 'V(s)', elixir: 'elixir',
  clock: 'match clock', result: 'result', match_time: 'match time',
};

function renderStats() {
  const dl = document.getElementById('stats-list');
  dl.innerHTML = '';
  for (const [k, v] of Object.entries(S.stats)) {
    const dt = document.createElement('dt');
    dt.textContent = STAT_LABEL[k] || k;
    const dd = document.createElement('dd');
    dd.textContent = typeof v === 'number'
      ? (k === 'win_rate' || k === 'revealed'
          ? (v * 100).toFixed(1) + '%'
          : (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3)))
      : String(v);
    dl.append(dt, dd);
  }
}

/* -------------------------------------------------------------- terminal */

const logEl = document.getElementById('log');
const autoscroll = document.getElementById('autoscroll');
let logLines = 0;

function appendLog(line, level) {
  const div = document.createElement('div');
  div.className = 'line ' + (level || 'info');
  const t = new Date().toLocaleTimeString('en-GB', { hour12: false });
  div.innerHTML = `<span class="t">${t}</span>`;
  div.appendChild(document.createTextNode(line));
  logEl.appendChild(div);
  if (++logLines > 1500) { logEl.removeChild(logEl.firstChild); logLines--; }
  if (autoscroll.checked) logEl.scrollTop = logEl.scrollHeight;
}

document.getElementById('clear-log').addEventListener('click', () => {
  logEl.innerHTML = ''; logLines = 0;
});

/* ------------------------------------------------------------------- SSE */

function connect() {
  const es = new EventSource('/events');
  const dot = document.getElementById('conn-dot');
  let closing = false;

  es.onopen = () => dot.classList.add('on');
  es.onerror = () => dot.classList.remove('on');
  es.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    handle(msg);
    // The server replays the sticky graph/act/learn frames the instant a
    // viewer connects, so by the time the graph lands the scene is already
    // complete and the stream can be dropped.
    if (SNAPSHOT && msg.t === 'graph' && !closing) {
      closing = true;
      setTimeout(() => { es.close(); dot.classList.remove('on'); }, 1200);
    }
  };
  return es;
}

function handle(msg) {
  switch (msg.t) {
    case 'graph':
      buildScene(msg);
      break;

    case 'act': {
      if (!S.actTarget || !msg.nodes) break;
      const a = msg.nodes;
      for (let i = 0; i < S.actTarget.length && i < a.length; i++) {
        S.actTarget[i] = a[i];
      }
      S.lastFrameAt = performance.now();
      for (const k of ['value', 'elixir', 'clock']) {
        if (msg[k] !== undefined) S.stats[k] = msg[k];
      }
      renderStats();
      break;
    }

    case 'learn': {
      if (!S.revealTarget || !msg.reveal) break;
      const r = msg.reveal;
      const mat = msg.maturity || [];
      for (let i = 0; i < S.revealTarget.length && i < r.length; i++) {
        // `maturity` keeps a loaded checkpoint from rendering as an empty
        // skeleton: movement is measured from attach, so a trained network
        // has not moved yet even though it is plainly trained.
        S.revealTarget[i] = Math.max(r[i], (mat[i] || 0) * 0.62);
      }
      break;
    }

    case 'stats':
      S.stats = { ...S.stats, ...(msg.metrics || {}) };
      renderStats();
      break;

    case 'log':
      appendLog(msg.line, msg.level);
      break;

    case 'mode':
      setModeUI(msg.mode);
      break;
  }
  if (msg.dropped) {
    appendLog(`[viz] dropped ${msg.dropped} frame(s) — viewer behind producer`,
              'warn');
  }
}

/* ------------------------------------------------------------------ modes */

function setModeUI(mode) {
  S.mode = mode;
  for (const b of document.querySelectorAll('button.mode')) {
    b.classList.toggle('active', b.dataset.mode === mode);
  }
}

for (const b of document.querySelectorAll('button.mode')) {
  b.addEventListener('click', async () => {
    const mode = b.dataset.mode;
    try {
      const res = await fetch('/api/mode?m=' + encodeURIComponent(mode));
      const body = await res.json();
      if (!body.ok) appendLog('[viz] ' + (body.error || 'mode change failed'), 'error');
      else setModeUI(body.mode);
    } catch (err) {
      appendLog('[viz] ' + err, 'error');
    }
  });
}

/* ------------------------------------------------------------------ frame */

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
}

let prev = performance.now();

function render(now) {
  const dt = Math.min((now - prev) / 1000, 0.1);
  prev = now;
  resize();
  flyStep(dt);
  updateMatrices();

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.024, 0.031, 0.059, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  if (S.graph) {
    animate(dt, now);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);   // additive: overlapping flow adds up
    gl.depthMask(false);
    gl.disable(gl.DEPTH_TEST);

    gl.useProgram(edgeProg);
    gl.uniformMatrix4fv(edgeProg.u.uMVP, false, mvp);
    gl.uniform1f(edgeProg.u.uTime, now / 1000);
    gl.uniform1f(edgeProg.u.uFlow, 0.42);
    gl.bindVertexArray(edgeVAO);
    gl.drawArrays(gl.LINES, 0, edgeVertexCount);

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(nodeProg);
    gl.uniformMatrix4fv(nodeProg.u.uMVP, false, mvp);
    gl.uniform1f(nodeProg.u.uPointScale, 36 * (canvas.height / 900));
    gl.bindVertexArray(nodeVAO);
    gl.drawArrays(gl.POINTS, 0, S.nodeCount);
    gl.bindVertexArray(null);

    positionLabels();
  }
  requestAnimationFrame(render);
}

function animate(dt, now) {
  // Asymmetric smoothing is the whole "fade in / fade out" feel: a unit
  // lights up quickly and decays slowly, so a pulse remains legible between
  // frames that arrive only 6-20 times a second.
  const attack = 1 - Math.exp(-dt * 14);
  const release = 1 - Math.exp(-dt * 2.6);
  const revealRate = 1 - Math.exp(-dt * 1.8);
  // With no frames arriving, drift activation to zero rather than freezing
  // the scene mid-pulse — a paused source should look paused.
  const stale = !SNAPSHOT && (now - S.lastFrameAt) > 2500;

  for (let i = 0; i < S.nodeCount; i++) {
    const target = stale ? 0 : S.actTarget[i];
    const k = target > S.act[i] ? attack : release;
    S.act[i] += (target - S.act[i]) * k;
    S.reveal[i] += (S.revealTarget[i] - S.reveal[i]) * revealRate;
  }

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.nodeAct);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, S.act);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.nodeRev);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, S.reveal);

  const ea = buffers.edgeActData, er = buffers.edgeRevData;
  for (let i = 0; i < S.edges.length; i++) {
    const [a, b] = S.edges[i];
    // The pulse belongs to the connection, so it is driven by the weaker of
    // the two endpoints: a wire out of a silent unit should not glow just
    // because its destination is loud for other reasons.
    const v = Math.min(S.act[a], S.act[b]);
    ea[i * 2] = v; ea[i * 2 + 1] = v;
    const rv = Math.min(S.reveal[a], S.reveal[b]);
    er[i * 2] = rv; er[i * 2 + 1] = rv;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.edgeAct);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, ea);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.edgeRev);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, er);
}

/* ------------------------------------------------------------------- boot */

gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LEQUAL);
appendLog('[viz] viewer ready — waiting for a graph…', 'good');
connect();
fetch('/api/state').then((r) => r.json()).then((s) => setModeUI(s.mode)).catch(() => {});
requestAnimationFrame(render);
