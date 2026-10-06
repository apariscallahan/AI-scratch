// Console tab: say bubbles, logs, progress bars, errors, test results.
import { $, bus, clock, el, fmtTime } from '../util.js';
import { revealBlock } from '../workspace.js';

const list = () => $('#console-list');
const progressEls = new Map();
let showDebug = false;
const MAX_LINES = 4000;

function atBottom() {
  const page = $('#tab-console');
  return page.scrollHeight - page.scrollTop - page.clientHeight < 60;
}

function add(node) {
  const stick = atBottom();
  list().append(node);
  while (list().childElementCount > MAX_LINES) list().firstElementChild.remove();
  if (stick) { const page = $('#tab-console'); page.scrollTop = page.scrollHeight; }
  bus.emit('tab:new', 'console');
}

function blockLink(id) {
  if (!id) return null;
  return el('span', { class: 'link', onclick: () => revealBlock(id) }, 'show block');
}

export function line(level, text, blockId) {
  const node = el('div', { class: `c-line ${level || 'info'}` }, el('span', { class: 't' }, clock()), text, blockLink(blockId));
  if (level === 'debug' && !showDebug) node.style.display = 'none';
  add(node);
  return node;
}

function separator(text) {
  add(el('div', { class: 'c-line sep' }, text));
}

export function init() {
  $('#console-clear').onclick = () => { list().innerHTML = ''; progressEls.clear(); };
  try { showDebug = localStorage.getItem('nb-console-details') === '1'; } catch (e) { /* storage blocked */ }
  $('#console-debug').checked = showDebug;
  $('#console-debug').onchange = (e) => {
    showDebug = e.target.checked;
    try { localStorage.setItem('nb-console-details', showDebug ? '1' : '0'); } catch (err) { /* ignore */ }
    for (const n of list().querySelectorAll('.c-line.debug')) n.style.display = showDebug ? '' : 'none';
  };

  bus.on('run:reset', (info) => {
    progressEls.clear();
    separator(`▶ Run started ${clock()}${info && info.project ? ' · ' + info.project : ''}${info && info.quick ? ' · quick test' : ''}${info && info.remote ? ' · on ' + info.remote.name : ''}`);
  });
  bus.on('ev:log', (ev) => line(ev.level, ev.text, ev.block));
  bus.on('ev:say', (ev) => add(el('div', {}, el('div', { class: 'bubble' }, ev.text))));
  bus.on('ev:device', (ev) => {
    $('#device-label').textContent = ev.label || '';
    line('debug', `Computing on: ${ev.label}`);
  });
  bus.on('ev:error', (ev) => {
    const card = el('div', { class: 'c-error' },
      el('div', { class: 'msg' }, '❌ ', ev.message || 'Error', blockLink(ev.block)),
      ev.hint ? el('div', { class: 'hint' }, '💡 ', ev.hint) : null,
      ev.traceback ? el('details', {}, el('summary', {}, 'technical details'), el('pre', {}, ev.traceback)) : null);
    add(card);
  });
  bus.on('ev:check', (ev) => line(ev.passed ? 'check-pass' : 'check-fail', `${ev.passed ? '✅' : '❌'} check ${ev.index}: ${ev.text}`, ev.block));
  bus.on('ev:check_summary', (ev) => line(ev.passed === ev.total ? 'success' : 'check-fail', `Tests: ${ev.passed} of ${ev.total} checks passed.`));
  bus.on('ev:progress', (ev) => {
    let p = progressEls.get(ev.id);
    if (!p) {
      const bar = el('div');
      const lbl = el('span');
      const info = el('span');
      const node = el('div', { class: 'c-progress' }, el('div', { class: 'lbl' }, lbl, info), el('div', { class: 'bar' }, bar));
      p = { node, bar, lbl, info };
      progressEls.set(ev.id, p);
      add(node);
    }
    p.lbl.textContent = ev.label || '';
    p.info.textContent = ev.info || '';
    const pct = ev.total ? Math.min(100, (100 * ev.current) / ev.total) : 0;
    p.bar.style.width = (ev.total ? pct : 100) + '%';
    p.node.classList.toggle('done', !!ev.total && ev.current >= ev.total);
  });
  bus.on('ev:dataset', (ev) => line('info', `📦 ${ev.info && ev.info.summary}`, ev.block));
  bus.on('ev:model', (ev) => line('info', `🧠 ${ev.info && ev.info.summary}`, ev.block));
  bus.on('ev:text', (ev) => {
    const t = (ev.prompt || '') + (ev.text || '');
    line('info', `✍️ ${ev.title}: “${t.slice(0, 140).replace(/\n/g, ' ⏎ ')}${t.length > 140 ? '…' : ''}” (full text in Results)`, ev.block);
  });
  bus.on('ev:table', (ev) => { if (!ev.quiet) line('debug', `📊 ${ev.title} → Results tab`, ev.block); });
  bus.on('ev:file', (ev) => line('debug', `💾 ${ev.path}`));
  bus.on('ev:done', (ev) => {
    const secs = ev.seconds != null ? ` in ${fmtTime(ev.seconds)}` : '';
    const word = { finished: '■ Finished', stopped: '■ Stopped', error: '■ Stopped with an error' }[ev.status] || '■ Done';
    separator(`${word}${secs} · ${clock()}`);
  });
}
