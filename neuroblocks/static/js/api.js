// Talking to the NeuroBlocks server: REST calls and the live event WebSocket.
import { bus } from './util.js';

const params = new URLSearchParams(location.search);
export const TOKEN = params.get('token') || sessionStorage.getItem('nb-token') || '';
if (TOKEN) sessionStorage.setItem('nb-token', TOKEN);

function headers(extra = {}) {
  const h = { ...extra };
  if (TOKEN) h.Authorization = `Bearer ${TOKEN}`;
  return h;
}

async function handle(r) {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) { /* not json */ }
    throw new Error(msg);
  }
  const ct = r.headers.get('content-type') || '';
  if (ct.includes('application/json')) return r.json();
  return r;
}

export const api = {
  get: (url) => fetch(url, { headers: headers() }).then(handle),
  post: (url, body) => fetch(url, { method: 'POST', headers: headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body ?? {}) }).then(handle),
  put: (url, body) => fetch(url, { method: 'PUT', headers: headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body ?? {}) }).then(handle),
  del: (url) => fetch(url, { method: 'DELETE', headers: headers() }).then(handle),
  raw: (url, opts = {}) => fetch(url, { ...opts, headers: headers(opts.headers || {}) }).then(handle),
  text: (url) => fetch(url, { headers: headers() }).then(handle).then((r) => r.text()),
};

// --- live events ---------------------------------------------------------------
let ws = null;
let retry = 0;
export function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/ws${TOKEN ? `?token=${encodeURIComponent(TOKEN)}` : ''}`;
  ws = new WebSocket(url);
  ws.onopen = () => { retry = 0; bus.emit('ws:open'); };
  ws.onmessage = (m) => {
    let msg;
    try { msg = JSON.parse(m.data); } catch (_) { return; }
    if (msg.type === 'hello') { bus.emit('ws:hello', msg); return; }
    if (msg.event) bus.emit('run:event', { run: msg.run, event: msg.event });
  };
  ws.onclose = () => {
    bus.emit('ws:close');
    retry = Math.min(retry + 1, 6);
    setTimeout(connect, 400 * retry);
  };
}

export function send(msg) {
  if (ws && ws.readyState === 1) { ws.send(JSON.stringify(msg)); return true; }
  return false;
}
