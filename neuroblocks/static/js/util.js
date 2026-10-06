// Small DOM / UI helpers shared by every module.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else if (k === 'html') e.innerHTML = v;
    else if (k === 'text') e.textContent = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function fmtNum(v, digits = 4) {
  if (v === null || v === undefined || v === '') return '';
  if (typeof v !== 'number') return String(v);
  if (!isFinite(v)) return String(v);
  if (Number.isInteger(v)) return v.toLocaleString();
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  return String(Number(v.toPrecision(digits)));
}

export function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1024 ** 3) return (n / 1024 / 1024).toFixed(1) + ' MB';
  return (n / 1024 ** 3).toFixed(2) + ' GB';
}

export function fmtTime(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec >= 3600) return `${Math.floor(sec / 3600)}h${String(Math.floor((sec % 3600) / 60)).padStart(2, '0')}m`;
  if (sec >= 60) return `${Math.floor(sec / 60)}m${String(sec % 60).padStart(2, '0')}s`;
  return `${sec}s`;
}

export function clock(ts = Date.now()) {
  const d = new Date(ts);
  return d.toTimeString().slice(0, 8);
}

// --- tiny event bus -------------------------------------------------------
const handlers = {};
export const bus = {
  on(type, fn) { (handlers[type] ||= []).push(fn); },
  emit(type, data) { for (const fn of handlers[type] || []) { try { fn(data); } catch (e) { console.error(type, e); } } },
};

// --- toasts -----------------------------------------------------------------
export function toast(msg, kind = '', ms = 2600) {
  const t = el('div', { class: `toast ${kind}` }, msg);
  $('#toasts').append(t);
  setTimeout(() => t.remove(), ms);
}

// --- modal dialogs ------------------------------------------------------------
export function modal({ title, body, buttons = [], wide = false, onClose } = {}) {
  const root = $('#modal-root');
  const back = el('div', { class: 'modal-back' });
  const box = el('div', { class: 'modal' + (wide ? ' wide' : '') });
  const close = () => { back.remove(); document.removeEventListener('keydown', onKey); onClose && onClose(); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  const header = el('header', {}, title || '', el('button', { class: 'x', onclick: close, title: 'Close' }, '×'));
  const content = el('div', { class: 'content' });
  if (typeof body === 'string') content.innerHTML = body; else if (body) content.append(body);
  box.append(header, content);
  if (buttons.length) {
    const footer = el('footer');
    for (const b of buttons) {
      footer.append(el('button', {
        class: `btn ${b.kind || ''}`,
        onclick: async () => { const keep = b.onClick ? await b.onClick(content) : false; if (keep !== true) close(); },
      }, b.label));
    }
    box.append(footer);
  }
  back.append(box);
  back.addEventListener('mousedown', (e) => { if (e.target === back) close(); });
  root.append(back);
  setTimeout(() => { const f = content.querySelector('input, textarea'); if (f) f.focus(); }, 30);
  return { close, content };
}

export function ask(title, label, value = '', okLabel = 'OK') {
  return new Promise((resolve) => {
    const input = el('input', { class: 'text', value });
    let done = false;
    const m = modal({
      title,
      body: el('div', {}, el('label', { class: 'field' }, label), input),
      buttons: [
        { label: 'Cancel', onClick: () => { done = true; resolve(null); } },
        { label: okLabel, kind: 'primary', onClick: () => { done = true; resolve(input.value); } },
      ],
      onClose: () => { if (!done) resolve(null); },
    });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { done = true; resolve(input.value); m.close(); } });
  });
}

export function confirmBox(title, text, okLabel = 'OK', kind = 'primary') {
  return new Promise((resolve) => {
    let done = false;
    modal({
      title, body: el('p', {}, text),
      buttons: [
        { label: 'Cancel', onClick: () => { done = true; resolve(false); } },
        { label: okLabel, kind, onClick: () => { done = true; resolve(true); } },
      ],
      onClose: () => { if (!done) resolve(false); },
    });
  });
}

export function downloadBlob(blob, filename) {
  const a = el('a', { href: URL.createObjectURL(blob), download: filename });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

export function downloadText(text, filename, type = 'text/plain') {
  downloadBlob(new Blob([text], { type }), filename);
}

// Colour palette shared with the Python side (Tableau 10).
export const PALETTE = ['#F28E2B', '#4E79A7', '#59A14F', '#B07AA1', '#FF9DA7', '#9C755F', '#76B7B2', '#EDC948', '#E15759', '#BAB0AC'];
export const SERIES_COLORS = ['#6c5ce7', '#f28e2b', '#2fa86a', '#e15759', '#4e79a7', '#b07aa1', '#edc948', '#76b7b2', '#9c755f', '#ff9da7'];
