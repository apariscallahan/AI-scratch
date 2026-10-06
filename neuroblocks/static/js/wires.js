// Node-editor style wires: from a 📦 weights block's ● to a 'start output' block's ○.
// A wire is stored as the in-port's value (the weights block id), so saving, undo/redo and
// copy/paste go through Blockly's normal field machinery; this module only draws and edits them.
/* global Blockly */
import { portHandlers } from './blocks.js';
import { bus } from './util.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const SNAP_PX = 30; // how close (on screen) a drop must be to a port

let ws = null;
let layer = null; // <g> inside the block canvas, behind the blocks
const paths = new Map(); // hat id -> {g, path, flow, label, text, bg}
const summaries = new Map(); // weights block id -> what it held in the last run
const flowing = new Set(); // hat ids whose output stack is running
let drag = null;
let raf = 0;
let dragLoop = false;

function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

// Screen pixels -> coordinates inside the wire layer (which zooms and scrolls with the blocks).
function toLayer(p) {
  const m = layer.getScreenCTM();
  if (!m) return p;
  const pt = new DOMPoint(p.x, p.y).matrixTransform(m.inverse());
  return { x: pt.x, y: pt.y };
}

function curve(a, b) {
  const dx = Math.max(50, Math.abs(b.x - a.x) * 0.5, Math.abs(b.y - a.y) * 0.25);
  const c1 = { x: a.x + dx, y: a.y };
  const c2 = { x: b.x - dx, y: b.y };
  const mid = { x: (a.x + 3 * c1.x + 3 * c2.x + b.x) / 8, y: (a.y + 3 * c1.y + 3 * c2.y + b.y) / 8 };
  return { d: `M ${a.x} ${a.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${b.x} ${b.y}`, mid };
}

function ports() {
  const outs = new Map();
  const ins = [];
  for (const b of ws.getAllBlocks(false)) {
    const f = b.getField && b.getField('PORT');
    if (!f || !f.screenCenter) continue;
    if (b.type === 'nb_weights') outs.set(b.id, { block: b, field: f });
    else if (b.type === 'nb_when_output') ins.push({ block: b, field: f, src: f.getValue() || '' });
  }
  return { outs, ins };
}

function labelText(weightsBlock) {
  const model = weightsBlock.getFieldValue('MODEL') || 'model';
  const s = summaries.get(weightsBlock.id);
  return s ? `📦 ${model} · ${s}` : `📦 ${model}`;
}

// --- drawing ------------------------------------------------------------------------------
export function redraw() {
  if (!ws || !layer) return;
  const { outs, ins } = ports();
  const used = new Set();
  const live = new Set();
  for (const p of ins) {
    const src = outs.get(p.src);
    p.field.setConnected(!!src);
    if (!src) continue;
    used.add(p.src);
    const a = src.field.screenCenter();
    const b = p.field.screenCenter();
    if (!a || !b) continue;
    const { d, mid } = curve(toLayer(a), toLayer(b));
    let w = paths.get(p.block.id);
    if (!w) {
      const g = svg('g', { class: 'nb-wire' }, layer);
      w = { g, casing: svg('path', { class: 'nb-wire-casing' }, g), path: svg('path', { class: 'nb-wire-core' }, g),
        flow: svg('path', { class: 'nb-wire-flow' }, g), label: svg('g', { class: 'nb-wire-label' }, g) };
      w.bg = svg('rect', { rx: 9, ry: 9 }, w.label);
      w.text = svg('text', {}, w.label);
      w.g.addEventListener('dblclick', (e) => { e.stopPropagation(); unplug(p.block.id); });
      w.g.append(Object.assign(document.createElementNS(SVG_NS, 'title'), { textContent: 'Double-click to unplug this wire' }));
      paths.set(p.block.id, w);
    }
    for (const el of [w.casing, w.path, w.flow]) el.setAttribute('d', d);
    const off = !p.block.isEnabled() || !src.block.isEnabled();
    w.g.classList.toggle('off', off);
    w.g.classList.toggle('flowing', flowing.has(p.block.id));
    const text = labelText(src.block);
    if (w.text.textContent !== text) w.text.textContent = text;
    w.text.setAttribute('x', mid.x);
    w.text.setAttribute('y', mid.y + 4);
    const tw = Math.min(w.text.getComputedTextLength ? w.text.getComputedTextLength() : text.length * 6.5, 600) || text.length * 6.5;
    w.bg.setAttribute('x', mid.x - tw / 2 - 8);
    w.bg.setAttribute('y', mid.y - 10);
    w.bg.setAttribute('width', tw + 16);
    w.bg.setAttribute('height', 20);
    live.add(p.block.id);
  }
  for (const [id, w] of paths) if (!live.has(id)) { w.g.remove(); paths.delete(id); }
  for (const [id, o] of outs) o.field.setConnected(used.has(id));
}

export function schedule() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    redraw();
    if (dragLoop) schedule();
  });
}

// --- editing ---------------------------------------------------------------------------------
function setWire(hatField, value) {
  if (hatField.getValue() !== value) hatField.setValue(value);
}

export function unplug(hatId) {
  const hat = ws.getBlockById(hatId);
  if (!hat) return;
  Blockly.Events.setGroup(true);
  try { setWire(hat.getField('PORT'), ''); } finally { Blockly.Events.setGroup(false); }
  schedule();
}

function nearest(candidates, pt) {
  let best = null;
  let bestD = SNAP_PX;
  for (const c of candidates) {
    const q = c.field.screenCenter();
    if (!q) continue;
    const dist = Math.hypot(q.x - pt.x, q.y - pt.y);
    if (dist < bestD) { best = c; bestD = dist; }
  }
  if (best) return best;
  // Dropping anywhere on the target block (or the stack under a start block) works too.
  for (const c of candidates) {
    const r = c.block.getSvgRoot && c.block.getSvgRoot().getBoundingClientRect();
    if (r && pt.x >= r.left && pt.x <= r.right && pt.y >= r.top && pt.y <= r.bottom) return c;
  }
  return null;
}

function startDrag(field, e) {
  const { outs, ins } = ports();
  const block = field.getSourceBlock();
  let fixed; // the end of the wire that stays put
  let wantIn; // looking for an in-port (true) or an out-port (false)
  let detachedFrom = null;
  Blockly.Events.setGroup(true); // everything this drag changes is one undo step
  if (field.direction === 'out') {
    fixed = { field, block, id: block.id };
    wantIn = true;
  } else {
    const src = outs.get(field.getValue());
    if (src) { // pick up the wire's end and carry it somewhere else (or drop it to unplug)
      detachedFrom = block.id;
      setWire(field, '');
      fixed = { field: src.field, block: src.block, id: src.block.id };
      wantIn = true;
    } else {
      fixed = { field, block, id: block.id };
      wantIn = false;
    }
  }
  const temp = svg('g', { class: 'nb-wire dragging' }, layer);
  const casing = svg('path', { class: 'nb-wire-casing' }, temp);
  const core = svg('path', { class: 'nb-wire-core' }, temp);
  drag = { fixed, wantIn, candidates: wantIn ? ins : [...outs.values()], temp, target: null, detachedFrom };
  document.body.classList.add('nb-wiring');
  const move = (ev) => {
    const a = fixed.field.screenCenter() || { x: ev.clientX, y: ev.clientY };
    const pt = { x: ev.clientX, y: ev.clientY };
    const target = nearest(drag.candidates, pt);
    if (drag.target !== target) {
      if (drag.target) drag.target.block.getSvgRoot().classList.remove('nb-wire-target');
      if (target) target.block.getSvgRoot().classList.add('nb-wire-target');
      drag.target = target;
    }
    const end = target ? (target.field.screenCenter() || pt) : pt;
    const [p1, p2] = wantIn ? [a, end] : [end, a];
    const { d } = curve(toLayer(p1), toLayer(p2));
    casing.setAttribute('d', d);
    core.setAttribute('d', d);
  };
  const finish = (ev, cancel = false) => {
    document.removeEventListener('pointermove', move, true);
    document.removeEventListener('pointerup', up, true);
    document.removeEventListener('keydown', key, true);
    document.body.classList.remove('nb-wiring');
    temp.remove();
    const target = cancel ? null : (ev ? nearest(drag.candidates, { x: ev.clientX, y: ev.clientY }) : null);
    if (drag.target) drag.target.block.getSvgRoot().classList.remove('nb-wire-target');
    try {
      if (target) {
        if (wantIn) setWire(target.field, fixed.id);
        else setWire(fixed.field, target.block.id);
      } else if (cancel && detachedFrom) {
        const hat = ws.getBlockById(detachedFrom);
        if (hat) setWire(hat.getField('PORT'), fixed.id);
      }
    } finally {
      Blockly.Events.setGroup(false);
      drag = null;
      schedule();
    }
  };
  const up = (ev) => finish(ev);
  const key = (ev) => { if (ev.key === 'Escape') { ev.stopPropagation(); finish(null, true); } };
  document.addEventListener('pointermove', move, true);
  document.addEventListener('pointerup', up, true);
  document.addEventListener('keydown', key, true);
  move(e);
}

function registerContextMenu() {
  try {
    Blockly.ContextMenuRegistry.registry.register({
      id: 'nb_unplug_wire',
      scopeType: Blockly.ContextMenuRegistry.ScopeType.BLOCK,
      displayText: () => '✂️ Unplug the wire',
      weight: -19,
      preconditionFn: (scope) => {
        const b = scope.block || scope.focusedNode;
        return b && b.type === 'nb_when_output' && !b.isInFlyout && b.getFieldValue('PORT') ? 'enabled' : 'hidden';
      },
      callback: (scope) => unplug((scope.block || scope.focusedNode).id),
    });
  } catch (e) { /* already registered */ }
}

// --- run-time feedback -----------------------------------------------------------------------
function setFlowing(hatId, on) {
  if (on) flowing.add(hatId); else flowing.delete(hatId);
  schedule();
}

export function init(workspace) {
  ws = workspace;
  const canvas = ws.getCanvas();
  layer = svg('g', { class: 'nb-wires' });
  canvas.insertBefore(layer, canvas.firstChild);
  portHandlers.down = startDrag;
  registerContextMenu();
  ws.addChangeListener((e) => {
    if (e.type === Blockly.Events.BLOCK_DRAG) {
      dragLoop = !!e.isStart;
      schedule();
      return;
    }
    if (e.type === Blockly.Events.VIEWPORT_CHANGE || !e.isUiEvent || e.type === Blockly.Events.FINISHED_LOADING) schedule();
  });
  window.addEventListener('resize', schedule);
  bus.on('ev:weights', (ev) => { if (ev.block) { summaries.set(ev.block, ev.summary); schedule(); } });
  bus.on('ev:output_start', (ev) => ev.block && setFlowing(ev.block, true));
  bus.on('ev:output_end', (ev) => ev.block && setFlowing(ev.block, false));
  bus.on('ev:done', () => { flowing.clear(); schedule(); });
  bus.on('project:loaded', () => { summaries.clear(); flowing.clear(); schedule(); });
  // Blocks render asynchronously: draw again once they have their final size.
  setInterval(() => { if (!drag && paths.size) schedule(); }, 1000);
  schedule();
}
