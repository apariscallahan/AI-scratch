// Output tab: what output stacks make — what's in the 📦 weights, text written token by token,
// the top choices, pictures, and 'ask … and wait' boxes. Newest at the bottom, like a conversation.
import { send } from '../api.js';
import { $, bus, el, esc } from '../util.js';
import { revealBlock } from '../workspace.js';

let active = null; // the running output stack: {label}
const keyed = new Map(); // result key -> element (a block's result is updated in place)
const asks = new Map(); // ask id -> {box, finish}

function list() { return $('#output-list'); }
function page() { return $('#tab-output'); }

// Keep the newest output in view (unless the user scrolled up to read something).
function add(node, parent = list()) {
  const p = page();
  const atBottom = p.scrollHeight - p.scrollTop - p.clientHeight < 60;
  parent.append(node);
  $('#output-empty').style.display = 'none';
  if (atBottom) requestAnimationFrame(() => { p.scrollTop = p.scrollHeight; });
  bus.emit('tab:new', 'output');
  return node;
}

function head(title, block, sub) {
  const h = el('h4', {}, title, sub ? el('span', { class: 'sub' }, sub) : null);
  if (block) {
    h.style.cursor = 'pointer';
    h.title = 'Show the block that made this';
    h.onclick = () => revealBlock(block);
  }
  return h;
}

function keyedCard(key, make) {
  let c = keyed.get(key);
  if (!c || !c.isConnected) {
    c = make();
    keyed.set(key, c);
    add(c);
  }
  return c;
}

function note(text, kind = '') { add(el('div', { class: `out-note ${kind}` }, text)); }

// --- the start of an output stack: what came through the wire --------------------------
function start(ev) {
  active = { label: ev.label };
  const rows = (ev.details || []).map(([k, v]) => el('tr', {}, el('th', {}, k), el('td', {}, v)));
  const card = el('div', { class: 'card out-start' },
    head(el('span', {}, '○ Output · using ', el('b', {}, `📦 ${ev.label}`)), ev.block, ev.summary),
    el('details', { class: 'body', open: true },
      el('summary', {}, 'What came through the wire?'),
      el('table', { class: 'out-facts' }, el('tbody', {}, rows))));
  add(card);
  bus.emit('tab:show', 'output');
}

function end() {
  if (!active) return;
  note('✓ output finished', 'end');
  active = null;
}

// --- results ---------------------------------------------------------------------------------
function text(ev) {
  const c = keyedCard(ev.key, () => el('div', { class: 'card' }, head('📝 Output text', ev.block),
    el('div', { class: 'body' }, el('div', { class: 'gen-text out-text' }))));
  const box = c.querySelector('.out-text');
  box.innerHTML = `${esc(ev.text)}<span class="cursor"></span>`;
  clearTimeout(c._t);
  c._t = setTimeout(() => { const cur = box.querySelector('.cursor'); if (cur) cur.remove(); }, 700);
}

function bars(ev) {
  const c = keyedCard(ev.key, () => el('div', { class: 'card' }, head(ev.title, ev.block),
    el('div', { class: 'body' }, el('div', { class: 'out-bars' }), el('div', { class: 'caption' }))));
  const items = ev.items || [];
  const top = ev.scores ? Math.max(1e-9, ...items.map((i) => Math.abs(i.p))) : 1;
  const box = c.querySelector('.out-bars');
  box.innerHTML = '';
  for (const [n, i] of items.entries()) {
    const w = Math.max(0, Math.min(1, Math.abs(i.p) / top)) * 100;
    box.append(el('div', { class: `out-bar ${n === 0 ? 'top' : ''}` },
      el('span', { class: 'lbl', title: `choice ${i.choice}` }, i.label),
      el('div', { class: 'track' }, el('div', { class: 'fill', style: { width: `${w}%` } })),
      el('span', { class: 'val' }, ev.scores ? String(Number(i.p.toPrecision(3))) : `${(i.p * 100).toFixed(1)}%`)));
  }
  c.querySelector('.caption').textContent = ev.scores
    ? 'These look like raw scores, not probabilities — turn them into probabilities first to compare chances.'
    : `Choice numbers: ${items.map((i) => i.choice).join(', ')}`;
}

// Pictures from the same block (e.g. in a loop) collect into one gallery.
function picture(ev) {
  const c = keyedCard(ev.key, () => el('div', { class: 'card' }, head('🖼️ Pictures', ev.block),
    el('div', { class: 'body' }, el('div', { class: 'out-gallery' }), el('div', { class: 'caption' }))));
  const gallery = c.querySelector('.out-gallery');
  gallery.append(el('figure', {}, el('img', { class: 'pixel', src: `data:image/png;base64,${ev.png}` }),
    ev.label != null ? el('figcaption', {}, ev.label) : null));
  while (gallery.childElementCount > 60) gallery.firstElementChild.remove();
  gallery.classList.toggle('many', gallery.childElementCount > 1);
  c.querySelector('.caption').textContent = gallery.childElementCount > 1
    ? `${gallery.childElementCount} pictures (${ev.caption.replace(/^a /, '')})` : (ev.caption || '');
}

// --- ask … and wait ----------------------------------------------------------------------------
function drawingPad(onSend) {
  const size = 280;
  const pad = el('canvas', { class: 'draw-pad', width: size, height: size });
  const ctx = pad.getContext('2d');
  const reset = () => { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, size, size); };
  reset();
  let drawing = false;
  let last = null;
  const brush = el('input', { type: 'range', min: '8', max: '40', value: '20' });
  const pos = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * (size / r.width), (e.clientY - r.top) * (size / r.height)]; };
  const stroke = (p) => {
    ctx.strokeStyle = '#fff'; ctx.lineWidth = parseInt(brush.value, 10); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(...(last || p)); ctx.lineTo(...p); ctx.stroke();
    last = p;
  };
  pad.addEventListener('pointerdown', (e) => {
    drawing = true; last = null;
    try { pad.setPointerCapture(e.pointerId); } catch (_) { /* pointer already gone */ }
    stroke(pos(e));
  });
  pad.addEventListener('pointermove', (e) => { if (drawing) stroke(pos(e)); });
  const up = () => { drawing = false; last = null; };
  pad.addEventListener('pointerup', up);
  pad.addEventListener('pointerleave', up);
  const pixels = () => {
    const n = 28;
    const small = el('canvas', { width: n, height: n });
    const sctx = small.getContext('2d');
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(pad, 0, 0, n, n);
    const data = sctx.getImageData(0, 0, n, n).data;
    const out = [];
    for (let i = 0; i < n * n; i++) out.push(Math.round((data[i * 4] / 255) * 1000) / 1000);
    return { pixels: out, size: n };
  };
  const send_ = el('button', { class: 'btn primary', onclick: () => onSend(pixels()) }, 'Send drawing ▶');
  const clear = el('button', { class: 'btn', onclick: reset }, '🧽 Clear');
  return el('div', { class: 'draw-row' }, pad,
    el('div', { class: 'out-pad-tools' }, send_, clear, el('label', { style: { fontSize: '12px' } }, 'brush ', brush)));
}

function ask(ev) {
  if (ev.headless) return; // asked in a terminal run (e.g. on a cloud GPU), not here
  const card = el('div', { class: 'card out-ask' }, head(`✋ ${ev.question || 'Your turn'}`, ev.block, 'the program is waiting'));
  const body = el('div', { class: 'body' });
  card.append(body);
  const reply = (data) => {
    send({ cmd: 'interact', rid: Date.now(), data: { ask: ev.id, ...data } });
    for (const b of body.querySelectorAll('button, input, textarea, canvas')) b.disabled = true;
    card.classList.add('sent');
  };
  if (ev.kind === 'draw') {
    body.append(drawingPad((d) => reply(d)));
  } else {
    const input = el('textarea', { rows: '2', placeholder: 'Type your answer and press Enter (Shift+Enter for a new line)…' });
    const go = () => reply({ answer: input.value });
    input.addEventListener('keydown', (e) => {
      e.stopPropagation(); // keep Blockly's keyboard shortcuts out of the text box
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); go(); }
    });
    body.append(el('div', { class: 'chat-input' }, input, el('button', { class: 'btn primary', onclick: go }, 'Send ▶')));
    setTimeout(() => input.focus(), 60);
  }
  asks.set(ev.id, { card, question: ev.question || '' });
  add(card);
  bus.emit('tab:show', 'output');
}

function asked(ev) {
  const a = asks.get(ev.id);
  if (!a) return;
  asks.delete(ev.id);
  const you = ev.png
    ? el('div', { class: 'chat-msg me' }, el('img', { class: 'pixel', src: `data:image/png;base64,${ev.png}`, style: { width: '84px', height: '84px', display: 'block' } }))
    : el('div', { class: 'chat-msg me' }, ev.answer === '' ? '(empty answer)' : ev.answer);
  a.card.replaceWith(el('div', { class: 'out-you' }, el('div', { class: 'out-q' }, a.question), you));
}

function cancelAsks() {
  for (const { card } of asks.values()) {
    card.classList.add('sent');
    for (const b of card.querySelectorAll('button, input, textarea')) b.disabled = true;
  }
  asks.clear();
}

export function init() {
  const clear = () => {
    keyed.clear();
    asks.clear();
    list().innerHTML = '';
    active = null;
    $('#output-empty').style.display = $('#play-area').childElementCount ? 'none' : '';
  };
  $('#output-clear').onclick = clear;
  bus.on('run:reset', clear);
  bus.on('ev:output_start', start);
  bus.on('ev:output_end', end);
  bus.on('ev:output_text', text);
  bus.on('ev:bars', bars);
  bus.on('ev:output_picture', picture);
  bus.on('ev:ask', ask);
  bus.on('ev:ask_done', asked);
  bus.on('ev:done', () => { cancelAsks(); end(); });
  // While an output stack runs, its messages belong here too.
  bus.on('ev:say', (ev) => { if (active) add(el('div', { class: 'bubble' }, ev.text)); });
  bus.on('ev:log', (ev) => { if (active && ['info', 'warn', 'success'].includes(ev.level || 'info')) note(ev.text, ev.level); });
  bus.on('ev:error', (ev) => {
    if (!active) return;
    add(el('div', { class: 'c-error' }, el('div', { class: 'msg' }, `❌ ${ev.message}`), ev.hint ? el('div', { class: 'hint' }, `💡 ${ev.hint}`) : null));
  });
}
