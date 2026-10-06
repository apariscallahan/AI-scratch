// Charts tab: live training curves (Chart.js).
/* global Chart */
import { $, bus, el, fmtNum, SERIES_COLORS } from '../util.js';

const charts = new Map(); // name -> {chart, series: Map(name -> {raw: [], ds}), card}
let dirty = new Set();
let smooth = 0.6;
let logLoss = false;
const MAX_POINTS = 5000;

const TITLES = {
  loss: 'Loss — how wrong the model is (lower is better)',
  accuracy: 'Accuracy (higher is better)',
  error: 'Average error (lower is better)',
  reward: 'Reward per try (higher is better)',
  distance: 'Distance travelled (m)',
  length: 'Try length (steps)',
};

function colorFor(chartObj, name) {
  const idx = chartObj.series.size;
  return SERIES_COLORS[idx % SERIES_COLORS.length];
}

function makeChart(name, xlabel) {
  const canvas = el('canvas');
  const card = el('div', { class: 'card' }, el('h4', {}, TITLES[name] || name), el('div', { class: 'body' }, el('div', { class: 'chart-box' }, canvas)));
  $('#charts-grid').append(card);
  $('#charts-empty').style.display = 'none';
  const isAcc = name === 'accuracy';
  const chart = new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: { datasets: [] },
    options: {
      animation: false,
      parsing: false,
      normalized: true,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      scales: {
        x: { type: 'linear', title: { display: true, text: xlabel || 'step' }, ticks: { maxTicksLimit: 7 } },
        y: {
          type: (logLoss && name === 'loss') ? 'logarithmic' : 'linear',
          ticks: { maxTicksLimit: 6, callback: (v) => (isAcc ? `${Math.round(v * 1000) / 10}%` : fmtNum(v, 3)) },
          ...(isAcc ? { suggestedMin: 0, suggestedMax: 1 } : {}),
        },
      },
      plugins: {
        legend: { labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${isAcc ? (c.parsed.y * 100).toFixed(1) + '%' : fmtNum(c.parsed.y)}` } },
      },
    },
  });
  const obj = { chart, series: new Map(), card, name };
  charts.set(name, obj);
  return obj;
}

function smoothed(raw, isTrain) {
  if (!isTrain || smooth <= 0 || raw.length < 3) return raw;
  const out = new Array(raw.length);
  let s = raw[0].y;
  let w = 0;
  for (let i = 0; i < raw.length; i++) {
    // Debiased exponential moving average (like TensorBoard).
    s = smooth * s + (1 - smooth) * raw[i].y;
    w = smooth * w + (1 - smooth);
    out[i] = { x: raw[i].x, y: i === 0 ? raw[0].y : s / (w || 1) };
  }
  return out;
}

function addPoint(ev) {
  const name = ev.chart || 'chart';
  const obj = charts.get(name) || makeChart(name, ev.xlabel || (ev.group === 'custom' ? 'x' : 'step'));
  const sname = ev.series || name;
  let s = obj.series.get(sname);
  if (!s) {
    const color = colorFor(obj, sname);
    const ds = { label: sname, data: [], borderColor: color, backgroundColor: color + '33', borderWidth: 2, pointRadius: 0, tension: 0.15, spanGaps: true };
    obj.chart.data.datasets.push(ds);
    s = { raw: [], ds, train: /train|average/i.test(sname) && name !== 'accuracy' };
    obj.series.set(sname, s);
  }
  if (ev.y === null || ev.y === undefined) return;
  s.raw.push({ x: ev.x, y: ev.y });
  if (s.raw.length > MAX_POINTS) {
    const merged = [];
    for (let i = 0; i < s.raw.length - 1; i += 2) merged.push({ x: s.raw[i + 1].x, y: (s.raw[i].y + s.raw[i + 1].y) / 2 });
    s.raw = merged;
  }
  dirty.add(obj);
}

function refresh() {
  for (const obj of dirty) {
    for (const s of obj.series.values()) {
      s.ds.data = smoothed(s.raw, s.train);
      s.ds.pointRadius = s.raw.length <= 30 ? 3 : 0;
    }
    obj.chart.update('none');
  }
  dirty = new Set();
}

function clearAll() {
  for (const obj of charts.values()) { obj.chart.destroy(); obj.card.remove(); }
  charts.clear();
  dirty.clear();
  $('#charts-empty').style.display = '';
}

export function init() {
  setInterval(refresh, 300);
  $('#charts-clear').onclick = clearAll;
  $('#chart-smooth').oninput = (e) => { smooth = parseFloat(e.target.value); for (const o of charts.values()) dirty.add(o); refresh(); };
  $('#chart-log').onchange = (e) => {
    logLoss = e.target.checked;
    const o = charts.get('loss');
    if (o) { o.chart.options.scales.y.type = logLoss ? 'logarithmic' : 'linear'; o.chart.update('none'); }
  };
  bus.on('run:reset', clearAll);
  bus.on('ev:metric', (ev) => { addPoint(ev); bus.emit('tab:new', 'charts'); bus.emit('tab:auto', 'charts'); });
}
