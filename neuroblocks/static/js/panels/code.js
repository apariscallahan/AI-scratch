// Python tab: the program your blocks make, as real Python.
import { api } from '../api.js';
import { $, bus, downloadText, esc, toast } from '../util.js';

let latest = '';
let visible = false;
let stale = true;

const KW = new Set(['def', 'return', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and', 'or', 'import', 'from', 'as',
  'global', 'pass', 'break', 'continue', 'lambda', 'True', 'False', 'None', 'with', 'try', 'except', 'class']);

export function highlight(code) {
  const out = [];
  const re = /("""[\s\S]*?"""|'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|#[^\n]*|\b\d+(?:\.\d+)?(?:e-?\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b|[\s\S])/g;
  let m;
  while ((m = re.exec(code))) {
    const t = m[0];
    if (t.startsWith('#')) out.push(`<span class="tok-com">${esc(t)}</span>`);
    else if (t.startsWith('"') || t.startsWith("'")) out.push(`<span class="tok-str">${esc(t)}</span>`);
    else if (/^\d/.test(t)) out.push(`<span class="tok-num">${t}</span>`);
    else if (KW.has(t)) out.push(`<span class="tok-kw">${t}</span>`);
    else if (t === 'nb') out.push(`<span class="tok-nb">nb</span>`);
    else if (/^[A-Za-z_]/.test(t) && code[re.lastIndex] === '(') out.push(`<span class="tok-fn">${t}</span>`);
    else out.push(esc(t));
  }
  return out.join('');
}

function show(res) {
  latest = res.code || '';
  $('#code-view').innerHTML = highlight(latest);
  stale = false;
}

export function init() {
  bus.on('compiled', (res) => { if (visible) show(res); else stale = true; });
  bus.on('tab:shown', async (name) => {
    visible = name === 'python';
    if (visible && stale) {
      const res = await api.post('/api/compile', { workspace: bus.workspaceJson(), project: bus.projectName() });
      show(res);
    }
  });
  $('#code-copy').onclick = async () => {
    try { await navigator.clipboard.writeText(latest); toast('Copied the Python code', 'ok'); } catch (e) { toast('Copy failed', 'err'); }
  };
  $('#code-download').onclick = async () => {
    const res = await api.post('/api/export/python', { workspace: bus.workspaceJson(), project: bus.projectName() });
    downloadText(res.code, res.filename, 'text/x-python');
  };
  $('#code-bundle').onclick = () => bus.emit('action', 'export-bundle');
}
