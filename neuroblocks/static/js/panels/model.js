// Model tab: layer-by-layer summary of every model that was built.
import { $, bus, el, fmtNum } from '../util.js';
import { revealBlock } from '../workspace.js';

const cards = new Map();

function render(ev) {
  const info = ev.info || {};
  const rows = info.rows || [];
  const maxParams = Math.max(1, ...rows.map((r) => r.params || 0));
  const t = el('table', { class: 'grid' });
  t.append(el('thead', {}, el('tr', {}, el('th', {}, 'layer'), el('th', {}, 'details'), el('th', {}, 'output shape'), el('th', {}, 'parameters'), el('th', {}, ''))));
  const tb = el('tbody');
  for (const r of rows) {
    const shape = r.shape ? r.shape.join(' × ') : '—';
    const name = el('td', { style: { paddingLeft: `${8 + (r.depth || 0) * 14}px` } }, r.name,
      r.auto ? el('span', { class: 'badge auto', title: 'Added automatically' }, 'auto') : null,
      r.repeat ? el('span', { class: 'badge' }, r.repeat) : null);
    const tr = el('tr', { class: `layer-row ${r.block ? 'clickable' : ''}`, title: r.block ? 'Click to find this block' : '' },
      name, el('td', {}, r.detail || ''), el('td', { class: 'num' }, shape), el('td', { class: 'num' }, r.params == null ? '' : fmtNum(r.params)),
      el('td', { style: { width: '90px' } }, r.params ? el('div', { class: 'layer-bar', style: { width: `${Math.max(2, (r.params / maxParams) * 80)}px` } }) : null));
    if (r.block) tr.onclick = () => revealBlock(r.block);
    tb.append(tr);
  }
  t.append(tb);
  const total = info.total_params;
  const head = el('h4', {}, `🧠 ${info.label || ev.name}`, el('span', { class: 'sub' }, total != null ? `${fmtNum(total)} parameters` : ''));
  const input = info.input_shape ? el('div', { class: 'caption', style: { marginBottom: '6px' } }, `Input: ${info.input_shape.join(' × ')} · task: ${info.task || '?'}`) : null;
  const sizeNote = total ? el('div', { class: 'caption' }, `≈ ${(total * 4 / 1e6).toFixed(total * 4 > 1e6 ? 1 : 3)} MB of weights. ${total > 5e6 ? 'Big model: a GPU helps a lot.' : ''}`) : null;
  return el('div', { class: 'card' }, head, el('div', { class: 'body' }, input, el('div', { class: 'scroll-x' }, t), sizeNote));
}

export function init() {
  bus.on('run:reset', () => { cards.clear(); $('#model-list').innerHTML = ''; $('#model-empty').style.display = ''; });
  bus.on('ev:model', (ev) => {
    const c = render(ev);
    const key = ev.name;
    if (cards.has(key)) cards.get(key).replaceWith(c); else $('#model-list').append(c);
    cards.set(key, c);
    $('#model-empty').style.display = 'none';
    bus.emit('tab:new', 'model');
  });
}
