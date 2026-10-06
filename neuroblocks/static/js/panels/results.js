// Results tab: images, tables, confusion matrices, scatter plots, predictions, generated text.
/* global Chart */
import { $, bus, el, esc, fmtNum, PALETTE } from '../util.js';
import { revealBlock } from '../workspace.js';

const keyed = new Map(); // key -> card (cards with the same key are replaced, e.g. decision maps during training)
const streams = new Map(); // text id -> {pre}
const chartsInCards = new Set();

function list() { return $('#results-list'); }

function card(title, body, { key, block, sub } = {}) {
  const h = el('h4', {}, title, sub ? el('span', { class: 'sub' }, sub) : null);
  if (block) {
    h.style.cursor = 'pointer';
    h.title = 'Show the block that made this';
    h.onclick = () => revealBlock(block);
  }
  const c = el('div', { class: 'card' }, h, el('div', { class: 'body' }, body));
  if (key && keyed.has(key)) {
    const old = keyed.get(key);
    old.replaceWith(c);
    for (const canvas of old.querySelectorAll('canvas')) {
      const ch = Chart.getChart(canvas);
      if (ch) { ch.destroy(); chartsInCards.delete(ch); }
    }
  } else {
    list().prepend(c);
  }
  if (key) keyed.set(key, c);
  $('#results-empty').style.display = 'none';
  bus.emit('tab:new', 'results');
  return c;
}

function table(columns, rows, footer) {
  const t = el('table', { class: 'grid' });
  t.append(el('thead', {}, el('tr', {}, columns.map((c) => el('th', {}, c)))));
  const tb = el('tbody');
  for (const r of rows) {
    tb.append(el('tr', {}, r.map((v) => {
      const num = typeof v === 'number';
      let text = num ? fmtNum(v) : String(v ?? '');
      const td = el('td', { class: num ? 'num' : '' }, text);
      if (v === '✓') td.style.color = '#1d7a45';
      if (v === '✗') td.style.color = '#b42318';
      return td;
    })));
  }
  t.append(tb);
  const wrap = el('div', { class: 'scroll-x' }, t);
  return footer ? el('div', {}, wrap, el('div', { class: 'footer-note' }, footer)) : wrap;
}

function confusion(labels, matrix) {
  const n = labels.length;
  const max = Math.max(1, ...matrix.flat());
  const t = el('table', { class: 'grid confusion' });
  const head = el('tr', {}, el('th', { class: 'axis' }, 'true ↓ / predicted →'), labels.map((l) => el('th', {}, l)));
  t.append(el('thead', {}, head));
  const tb = el('tbody');
  for (let i = 0; i < n; i++) {
    const row = el('tr', {}, el('th', { class: 'rowh' }, labels[i]));
    for (let j = 0; j < n; j++) {
      const v = matrix[i][j];
      const a = v / max;
      const good = i === j;
      const bg = v === 0 ? '#fff' : good ? `rgba(47,168,106,${0.15 + 0.75 * a})` : `rgba(229,72,77,${0.12 + 0.75 * a})`;
      row.append(el('td', { style: { background: bg, color: a > 0.55 ? '#fff' : '#2b2d42', fontWeight: good ? 700 : 400 } }, v));
    }
    tb.append(row);
  }
  t.append(tb);
  return el('div', { class: 'scroll-x' }, t);
}

function scatterChart(ev) {
  const canvas = el('canvas');
  const box = el('div', { class: 'chart-box', style: { height: '300px' } }, canvas);
  const datasets = [];
  const pts = ev.points || [];
  if (ev.continuous) {
    const vals = pts.map((p) => p[2]);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const col = (v) => {
      const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      return `rgb(${Math.round(78 + t * (242 - 78))},${Math.round(121 + t * (142 - 121))},${Math.round(167 + t * (43 - 167))})`;
    };
    datasets.push({ label: 'examples', data: pts.map((p) => ({ x: p[0], y: p[1] })), pointBackgroundColor: pts.map((p) => col(p[2])), pointRadius: 3 });
  } else {
    const groups = new Map();
    for (const p of pts) {
      const k = p[2] ?? 0;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push({ x: p[0], y: p[1] });
    }
    const classes = ev.classes || [];
    for (const [k, data] of [...groups.entries()].sort((a, b) => a[0] - b[0])) {
      const c = PALETTE[k % PALETTE.length];
      datasets.push({ label: classes[k] ?? String(k), data, backgroundColor: c + 'cc', borderColor: c, pointRadius: 3 });
    }
  }
  if (ev.line && ev.line.length) {
    datasets.push({ type: 'line', label: 'model', data: ev.line.map((p) => ({ x: p[0], y: p[1] })), borderColor: '#6c5ce7', borderWidth: 3, pointRadius: 0, fill: false, tension: 0.1 });
  }
  setTimeout(() => {
    const ch = new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: { datasets },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false, parsing: true,
        scales: { x: { title: { display: !!ev.xlabel, text: ev.xlabel } }, y: { title: { display: !!ev.ylabel, text: ev.ylabel } } },
        plugins: { legend: { display: datasets.length > 1 && datasets.length < 12, labels: { boxWidth: 10 } } },
      },
    });
    chartsInCards.add(ch);
  }, 0);
  return box;
}

function plotChart(ev) {
  const canvas = el('canvas');
  const box = el('div', { class: 'chart-box', style: { height: ev.kind === 'bar' ? `${Math.max(180, (ev.values || []).length * 22)}px` : '240px' } }, canvas);
  const labels = ev.labels || (ev.values || []).map((_, i) => i + 1);
  setTimeout(() => {
    const ch = new Chart(canvas.getContext('2d'), {
      type: ev.kind === 'bar' ? 'bar' : (ev.kind === 'scatter' ? 'scatter' : 'line'),
      data: ev.kind === 'scatter'
        ? { datasets: [{ label: ev.title, data: (ev.values || []).map((v, i) => ({ x: i + 1, y: v })), backgroundColor: '#6c5ce7' }] }
        : { labels, datasets: [{ label: ev.title, data: ev.values || [], backgroundColor: '#8e7dffcc', borderColor: '#6c5ce7', borderWidth: 2, pointRadius: 3 }] },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false, indexAxis: ev.kind === 'bar' && labels.length > 6 ? 'y' : 'x',
        plugins: { legend: { display: false } },
      },
    });
    chartsInCards.add(ch);
  }, 0);
  return box;
}

function predictions(ev) {
  if (ev.kind === 'pairs') {
    return el('div', { class: 'pred-grid' }, (ev.items || []).map((it) => el('div', { class: 'pred' },
      el('div', { class: 'pair' }, el('img', { src: `data:image/png;base64,${it.png}` }), el('img', { src: `data:image/png;base64,${it.png2}` })),
      el('div', { class: 'tr' }, it.label || ''))));
  }
  return el('div', { class: 'pred-grid' }, (ev.items || []).map((it) => {
    const hasTruth = it.true !== undefined;
    return el('div', { class: `pred ${hasTruth ? (it.ok ? 'ok' : 'bad') : ''}`, title: it.conf != null ? `confidence ${(it.conf * 100).toFixed(0)}%` : '' },
      el('img', { src: `data:image/png;base64,${it.png}` }),
      el('div', { class: 'p' }, it.pred ?? ''),
      hasTruth ? el('div', { class: 'tr' }, it.ok ? '✓ correct' : `really: ${it.true}`) : null);
  }));
}

function textBody(ev) {
  const pre = el('div', { class: 'gen-text' });
  pre.innerHTML = (ev.prompt ? `<b>${esc(ev.prompt)}</b>` : '') + esc(ev.text || '');
  return pre;
}

export function init() {
  $('#results-clear').onclick = clearAll;
  bus.on('run:reset', clearAll);
  bus.on('ev:image', (ev) => {
    const img = el('img', { class: 'result', src: `data:image/png;base64,${ev.png}` });
    card(ev.title, el('div', {}, img, ev.caption ? el('div', { class: 'caption' }, ev.caption) : null), { key: ev.key || ev.title, block: ev.block });
  });
  bus.on('ev:table', (ev) => card(ev.title, table(ev.columns || [], ev.rows || [], ev.footer), { block: ev.block }));
  bus.on('ev:confusion', (ev) => card(ev.title, el('div', {}, confusion(ev.labels, ev.matrix),
    el('div', { class: 'caption' }, 'Rows: the real class. Columns: what the model said. Green diagonal = correct.')), { key: ev.title, block: ev.block }));
  bus.on('ev:scatter', (ev) => card(ev.title, scatterChart(ev), { key: ev.key, block: ev.block }));
  bus.on('ev:plot', (ev) => card(ev.title, plotChart(ev), { key: ev.title, block: ev.block }));
  bus.on('ev:predictions', (ev) => card(ev.title, predictions(ev), { block: ev.block }));
  bus.on('ev:text_stream', (ev) => {
    if (String(ev.id || '').startsWith('chat-')) return; // shown in the Output tab
    let s = streams.get(ev.id);
    if (!s) {
      const pre = el('div', { class: 'gen-text' });
      const c = card('✍️ writing…', pre, { key: `text-${ev.id}` });
      s = { pre, c };
      streams.set(ev.id, s);
    }
    s.pre.innerHTML = (ev.prompt ? `<b>${esc(ev.prompt)}</b>` : '') + esc(ev.text || '') + '<span class="cursor"></span>';
    s.pre.scrollTop = s.pre.scrollHeight;
  });
  bus.on('ev:text', (ev) => {
    card(ev.title || 'Text', textBody(ev), { key: ev.id ? `text-${ev.id}` : undefined, block: ev.block,
      sub: ev.seconds != null ? `${ev.seconds}s` : '' });
    streams.delete(ev.id);
  });
}

function clearAll() {
  for (const ch of chartsInCards) ch.destroy();
  chartsInCards.clear();
  keyed.clear();
  streams.clear();
  list().innerHTML = '';
  $('#results-empty').style.display = '';
}
