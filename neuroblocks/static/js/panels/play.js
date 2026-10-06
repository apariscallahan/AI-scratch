// Play tab: interactive testing sessions started by 'let me …' blocks.
import { send } from '../api.js';
import { $, bus, debounce, el, esc } from '../util.js';
import { SimRenderer } from './sim.js';

let session = null; // {id, kind, card, pending: Map(rid -> callback), ...}
let rid = 0;

function area() { return $('#play-area'); }

function request(data, cb) {
  if (!session) return;
  const r = ++rid;
  session.pending.set(r, cb);
  send({ cmd: 'interact', rid: r, data });
  return r;
}

function endSession(fromUser) {
  if (!session) return;
  if (fromUser) send({ cmd: 'interact_end' });
  if (session.cleanup) session.cleanup();
  const s = session;
  session = null;
  if (s.card) {
    s.card.style.opacity = '0.6';
    const done = s.card.querySelector('.done-btn');
    if (done) { done.disabled = true; done.textContent = 'Finished'; }
  }
}

function shell(title, ...body) {
  const doneBtn = el('button', { class: 'small-btn primary done-btn', onclick: () => endSession(true) }, 'Done — continue the program ▶');
  const card = el('div', { class: 'card play-card' }, el('h4', {}, title, el('span', { class: 'sub' }, doneBtn)), el('div', { class: 'body' }, ...body));
  area().innerHTML = '';
  area().append(card);
  $('#play-empty').style.display = 'none';
  return card;
}

// --- chat / prompt ---------------------------------------------------------------
function startChat(ev) {
  const log = el('div', { class: 'chat-log' });
  const input = el('textarea', { placeholder: 'Type the start of a text, e.g. "ROMEO:" — the model continues it…' });
  const len = el('input', { type: 'range', min: '20', max: '1000', step: '10', value: String(ev.options?.length || 200) });
  const lenLbl = el('span', {}, len.value);
  len.oninput = () => { lenLbl.textContent = len.value; };
  const temp = el('input', { type: 'range', min: '0', max: '1.5', step: '0.05', value: String(ev.options?.temperature ?? 0.8) });
  const tempLbl = el('span', {}, temp.value);
  temp.oninput = () => { tempLbl.textContent = temp.value; };
  const btn = el('button', { class: 'btn primary' }, 'Write ✍️');
  const go = () => {
    const prompt = input.value;
    log.append(el('div', { class: 'chat-msg me' }, prompt || '(empty prompt)'));
    const out = el('div', { class: 'chat-msg ai' }, '…');
    log.append(out);
    log.scrollTop = log.scrollHeight;
    btn.disabled = true;
    const r = request({ prompt, length: parseInt(len.value, 10), temperature: parseFloat(temp.value) }, (res) => {
      btn.disabled = false;
      if (!res.ok) { out.textContent = '⚠️ ' + res.error; return; }
      out.innerHTML = `<b>${esc(res.result.prompt)}</b>${esc(res.result.text)}`;
      log.scrollTop = log.scrollHeight;
    });
    session.streams.set(`chat-${r}`, out);
  };
  btn.onclick = go;
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); go(); } });
  session.card = shell(`💬 Prompt ${ev.label}`, log,
    el('div', { class: 'chat-input' }, input, btn),
    el('div', { class: 'chat-opts' }, el('label', {}, 'length ', len, ' ', lenLbl), el('label', {}, 'creativity ', temp, ' ', tempLbl),
      el('span', {}, 'Ctrl+Enter to send')));
}

// --- drawing pad --------------------------------------------------------------------
function startDraw(ev) {
  const size = 280;
  const pad = el('canvas', { class: 'draw-pad', width: size, height: size });
  const ctx = pad.getContext('2d');
  const reset = () => { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, size, size); };
  reset();
  const probs = el('div', { class: 'probs' }, el('div', { style: { color: 'var(--ink-3)' } }, 'Draw something on the black square…'));
  const seen = el('img', { style: { width: '84px', height: '84px', imageRendering: 'pixelated', borderRadius: '6px', background: '#000', display: 'none' } });
  let drawing = false, last = null, ink = false;
  const pos = (e) => { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * (size / r.width), (e.clientY - r.top) * (size / r.height)]; };
  const brush = el('input', { type: 'range', min: '8', max: '40', value: '20' });
  const stroke = (p) => {
    ctx.strokeStyle = '#fff'; ctx.lineWidth = parseInt(brush.value, 10); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); ctx.moveTo(...(last || p)); ctx.lineTo(...p); ctx.stroke();
    last = p; ink = true;
  };
  const predict = () => {
    if (!ink) return;
    const n = 28;
    const small = el('canvas', { width: n, height: n });
    const sctx = small.getContext('2d');
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = 'high';
    sctx.drawImage(pad, 0, 0, n, n);
    const data = sctx.getImageData(0, 0, n, n).data;
    const pixels = [];
    for (let i = 0; i < n * n; i++) pixels.push(Math.round((data[i * 4] / 255) * 1000) / 1000);
    request({ pixels, size: n }, (res) => {
      if (!res.ok) { probs.innerHTML = `<div style="color:var(--err)">⚠️ ${esc(res.error)}</div>`; return; }
      probs.innerHTML = '';
      const top = res.result.probs[0];
      res.result.probs.forEach((p) => {
        probs.append(el('div', { class: `prob ${p === top ? 'top' : ''}` }, el('span', { class: 'nm' }, p.label),
          el('div', { class: 'b', style: { width: `${Math.max(2, p.p * 160)}px` } }), el('span', {}, `${(p.p * 100).toFixed(1)}%`)));
      });
      if (res.result.seen) { seen.src = `data:image/png;base64,${res.result.seen}`; seen.style.display = 'block'; }
    });
  };
  const predictSoon = debounce(predict, 120);
  pad.addEventListener('pointerdown', (e) => { drawing = true; last = null; pad.setPointerCapture(e.pointerId); stroke(pos(e)); });
  pad.addEventListener('pointermove', (e) => { if (drawing) { stroke(pos(e)); predictSoon(); } });
  const up = () => { if (drawing) { drawing = false; last = null; predict(); } };
  pad.addEventListener('pointerup', up);
  pad.addEventListener('pointerleave', up);
  const clear = el('button', { class: 'btn' }, '🧽 Clear');
  clear.onclick = () => { reset(); ink = false; probs.innerHTML = ''; seen.style.display = 'none'; };
  session.card = shell(`✏️ Draw for ${ev.label}`,
    el('div', { class: 'draw-row' }, el('div', {}, pad, el('div', { style: { display: 'flex', gap: '8px', marginTop: '8px', alignItems: 'center' } }, clear, el('label', { style: { fontSize: '12px' } }, 'brush ', brush))),
      el('div', { class: 'probs' }, el('div', { style: { fontWeight: 700, marginBottom: '6px' } }, 'The model thinks it is…'), probs,
        el('div', { style: { marginTop: '12px', fontSize: '12px', color: 'var(--ink-3)' } }, 'What the model sees (shrunk & centred):'), seen)));
}

// --- form ---------------------------------------------------------------------------------
function startForm(ev) {
  const fields = ev.options?.fields || [];
  const grid = el('div', { class: 'form-grid' });
  const inputs = [];
  for (const f of fields) {
    const inp = f.kind === 'text'
      ? el('textarea', { rows: '3', style: { width: '100%' }, placeholder: 'Type a sentence…' })
      : el('input', { type: 'number', value: String(f.value), step: String(f.step || 'any') });
    inputs.push(inp);
    grid.append(el('label', {}, f.name), inp);
  }
  const result = el('div', { class: 'form-result' }, '');
  const go = () => {
    request({ values: inputs.map((i) => i.value) }, (res) => {
      result.textContent = res.ok ? `${ev.options?.target || 'prediction'}: ${res.result.prediction}` : `⚠️ ${res.error}`;
    });
  };
  const goSoon = debounce(go, 250);
  inputs.forEach((i) => i.addEventListener('input', goSoon));
  session.card = shell(`🔢 Try ${ev.label}`, grid, el('div', { style: { marginTop: '10px' } }, el('button', { class: 'btn primary', onclick: go }, 'Predict')), result);
  go();
}

// --- game (human control of a simulation) ------------------------------------------------------
function startGame(ev) {
  const canvas = el('canvas');
  const box = el('div', { class: 'game-box', tabindex: '0' }, canvas);
  const renderer = new SimRenderer(canvas);
  const keys = {};
  let frame = null;
  let raf = 0;
  let liveTime = 0;
  let liveDt = 1 / 30;
  const helpText = Object.entries(ev.options?.keys || {}).map(([k, v]) => `${k === ' ' ? 'Space' : k.replace('Arrow', '')}: ${v}`).join(' · ');
  const sendKeys = () => send({ cmd: 'keys', keys: { ...keys } });
  const down = (e) => {
    // Keep keys away from Blockly's document-wide shortcuts (Delete would delete a selected block).
    e.stopPropagation();
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' ', 'Backspace', 'Delete'].includes(e.key)) e.preventDefault();
    if (!keys[e.key]) { keys[e.key] = true; sendKeys(); }
  };
  const up = (e) => { e.stopPropagation(); if (keys[e.key]) { delete keys[e.key]; sendKeys(); } };
  box.addEventListener('keydown', down);
  box.addEventListener('keyup', up);
  box.addEventListener('blur', () => { for (const k of Object.keys(keys)) delete keys[k]; sendKeys(); });
  const tick = () => {
    raf = requestAnimationFrame(tick);
    if (renderer.replay && frame) {
      renderer.replay.frames = [frame];
      renderer.draw(0, { time: liveTime });
    }
  };
  raf = requestAnimationFrame(tick);
  session.onLive = (lv) => {
    if (lv.scene) renderer.setReplay({ scene: lv.scene, frames: [], agents: [{ alpha: 1 }], dt: 1 / 30, kind: lv.kind || 'car', hud: [] });
    if (lv.reset) { liveTime = 0; liveDt = lv.dt || liveDt; }
    if (lv.frame) { frame = [lv.frame]; if (!lv.reset) liveTime += liveDt; }
    if (lv.frames && lv.frames.length) frame = [lv.frames[lv.frames.length - 1]];
    if (lv.hud !== undefined && renderer.replay) renderer.replay.hud = [lv.hud];
    if (renderer.replay) renderer.replay.fx = lv.fx ? [[lv.fx]] : undefined;
  };
  session.cleanup = () => cancelAnimationFrame(raf);
  session.card = shell(`🕹️ Play ${ev.label}`, box, el('div', { class: 'keys-help' }, `Click the picture, then use the keyboard. ${helpText}`));
  setTimeout(() => box.focus(), 50);
}

export function init() {
  bus.on('run:reset', () => { if (session) endSession(false); });
  bus.on('ev:interactive', (ev) => {
    if (session) endSession(false);
    session = { id: ev.id, kind: ev.kind, pending: new Map(), streams: new Map() };
    if (ev.kind === 'chat') startChat(ev);
    else if (ev.kind === 'draw') startDraw(ev);
    else if (ev.kind === 'form') startForm(ev);
    else if (ev.kind === 'game') startGame(ev);
    bus.emit('tab:show', 'play');
  });
  bus.on('ev:interactive_result', (ev) => {
    if (!session || ev.id !== session.id) return;
    const cb = session.pending.get(ev.rid);
    session.pending.delete(ev.rid);
    if (cb) cb(ev);
  });
  bus.on('ev:text_stream', (ev) => {
    if (!session || !String(ev.id || '').startsWith('chat-')) return;
    const out = session.streams.get(ev.id);
    if (out) out.innerHTML = `<b>${esc(ev.prompt || '')}</b>${esc(ev.text || '')}<span class="cursor"></span>`;
  });
  bus.on('ev:sim_live', (ev) => { if (session && session.onLive && ev.id === session.id) session.onLive(ev); });
  bus.on('ev:interactive_end', (ev) => { if (session && ev.id === session.id) endSession(false); });
  bus.on('ev:done', () => { if (session) endSession(false); });
}
