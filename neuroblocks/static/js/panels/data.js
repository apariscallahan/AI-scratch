// Data tab: dataset previews + your files (upload CSV / text / image folders).
/* global Chart */
import { api } from '../api.js';
import { $, bus, el, esc, fmtBytes, fmtNum, PALETTE, toast } from '../util.js';
import { revealBlock } from '../workspace.js';

const cards = new Map();
const charts = new Set();

function previewNode(p, info) {
  if (!p) return null;
  if (p.kind === 'images') {
    return el('div', {}, el('img', { class: 'result pixel', src: `data:image/png;base64,${p.png}` }),
      el('div', { class: 'caption' }, 'First examples: ' + (p.labels || []).slice(0, 16).join(', ') + ((p.labels || []).length > 16 ? '…' : '')));
  }
  if (p.kind === 'table') {
    const t = el('table', { class: 'grid' });
    t.append(el('thead', {}, el('tr', {}, p.columns.map((c) => el('th', {}, c)))));
    t.append(el('tbody', {}, p.rows.map((r) => el('tr', {}, r.map((v) => el('td', { class: typeof v === 'number' ? 'num' : '' }, typeof v === 'number' ? fmtNum(v) : v))))));
    return el('div', { class: 'scroll-x' }, t);
  }
  if (p.kind === 'text') {
    const d = el('div', { class: 'gen-text' });
    d.innerHTML = esc(p.text);
    return d;
  }
  if (p.kind === 'points' || p.kind === 'series') {
    const canvas = el('canvas');
    const box = el('div', { class: 'chart-box', style: { height: '240px' } }, canvas);
    setTimeout(() => {
      let cfg;
      if (p.kind === 'series') {
        cfg = { type: 'line', data: { labels: p.values.map((_, i) => i), datasets: [{ data: p.values, borderColor: '#6c5ce7', pointRadius: 0, borderWidth: 1.5 }] },
          options: { animation: false, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { ticks: { maxTicksLimit: 8 } } } } };
      } else if (p.dims === 1) {
        cfg = { type: 'scatter', data: { datasets: [{ data: p.points.map((q) => ({ x: q[0], y: q[2] })), backgroundColor: '#6c5ce7aa', pointRadius: 3 }] },
          options: { animation: false, maintainAspectRatio: false, plugins: { legend: { display: false } } } };
      } else {
        const groups = new Map();
        const classes = p.classes || [];
        for (const q of p.points) {
          const k = classes.length ? Math.max(0, classes.indexOf(q[2])) : 0;
          if (!groups.has(k)) groups.set(k, []);
          groups.get(k).push({ x: q[0], y: q[1] });
        }
        cfg = { type: 'scatter', data: { datasets: [...groups.entries()].map(([k, data]) => ({ label: classes[k] || 'examples', data, backgroundColor: PALETTE[k % PALETTE.length] + 'cc', pointRadius: 3 })) },
          options: { animation: false, maintainAspectRatio: false, plugins: { legend: { labels: { boxWidth: 10 } } } } };
      }
      charts.add(new Chart(canvas.getContext('2d'), cfg));
    }, 0);
    return box;
  }
  return null;
}

function render(ev) {
  const info = ev.info || {};
  const body = el('div', { class: 'body' });
  if (info.description) body.append(el('div', { style: { marginBottom: '6px' } }, info.description));
  body.append(el('div', { class: 'caption', style: { marginTop: 0, marginBottom: '8px' } }, info.summary || ''));
  if (info.tokenizer) body.append(el('div', { class: 'caption' }, `Tokens: ${info.tokenizer}`));
  if (info.classes && info.class_counts) {
    const max = Math.max(1, ...info.class_counts);
    const bars = el('div', { class: 'class-bars' });
    info.classes.slice(0, 20).forEach((c, i) => bars.append(el('div', { class: 'class-bar' }, el('span', { class: 'nm', title: c }, c),
      el('div', { class: 'b', style: { width: `${(info.class_counts[i] / max) * 140}px`, background: PALETTE[i % PALETTE.length] } }),
      el('span', {}, fmtNum(info.class_counts[i])))));
    body.append(el('div', { style: { margin: '6px 0 10px' } }, el('b', { style: { fontSize: '12px' } }, 'Examples per class'), bars));
  }
  const prev = previewNode(info.preview, info);
  if (prev) body.append(prev);
  const head = el('h4', {}, `📦 ${ev.name}`, el('span', { class: 'sub' }, info.source || ''));
  if (ev.block) { head.style.cursor = 'pointer'; head.onclick = () => revealBlock(ev.block); }
  return el('div', { class: 'card' }, head, body);
}

async function refreshFiles() {
  try {
    const res = await api.get('/api/files');
    $('#home-path').textContent = res.home;
    const list = $('#file-list');
    list.innerHTML = '';
    const all = [...res.data, ...res.models];
    if (!all.length) { list.append(el('div', { class: 'caption' }, 'No files yet. Uploaded data goes in the data folder; saved models in models.')); return; }
    for (const f of all) {
      list.append(el('div', { class: 'f' }, el('span', {}, f.kind === 'model' ? '🧠' : '📄'),
        el('span', { class: 'nm', title: 'Click to copy the name', style: { cursor: 'pointer' },
          onclick: () => { navigator.clipboard && navigator.clipboard.writeText(f.name); toast(`Copied “${f.name}” — paste it into a file slot`); } }, f.path),
        el('span', { class: 'sz' }, fmtBytes(f.size))));
    }
  } catch (e) { /* server not ready */ }
}

async function upload(files) {
  for (const f of files) {
    const isZip = f.name.toLowerCase().endsWith('.zip');
    try {
      const q = new URLSearchParams({ path: `data/${f.name}` });
      if (isZip) q.set('unzip', '1');
      const res = await api.raw(`/api/upload?${q}`, { method: 'POST', body: f });
      const j = await (res.json ? res.json() : res);
      toast(`Uploaded ${j.path || f.name}`, 'ok');
    } catch (e) {
      toast(`Upload failed: ${e.message}`, 'err');
    }
  }
  refreshFiles();
}

export function init() {
  bus.on('run:reset', () => {
    for (const c of charts) c.destroy();
    charts.clear();
    cards.clear();
    $('#data-list').innerHTML = '';
  });
  bus.on('ev:dataset', (ev) => {
    const c = render(ev);
    if (cards.has(ev.name)) cards.get(ev.name).replaceWith(c); else $('#data-list').append(c);
    cards.set(ev.name, c);
    bus.emit('tab:new', 'data');
  });
  bus.on('ev:file', refreshFiles);
  bus.on('ev:done', refreshFiles);
  bus.on('tab:shown', (name) => { if (name === 'data') refreshFiles(); });
  const input = $('#upload-input');
  $('#upload-btn').onclick = () => input.click();
  input.onchange = () => { upload([...input.files]); input.value = ''; };
  const dz = $('#drop-zone');
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('over'); upload([...e.dataTransfer.files]); });
  refreshFiles();
}
