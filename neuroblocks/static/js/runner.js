// Running programs: buttons, status, live event routing, tabs, sounds, past runs.
import { api, send } from './api.js';
import { $, $$, bus, confirmBox, el, fmtTime, toast } from './util.js';
import { project } from './project.js';
import * as WS from './workspace.js';

const state = { runId: null, status: 'idle', started: 0, autoSwitched: new Set(), viewing: false, lastProgress: 0 };

// --- tabs ---------------------------------------------------------------------------
let activeTab = 'console';
export function showTab(name) {
  activeTab = name;
  for (const b of $$('#tabs button')) {
    const on = b.dataset.tab === name;
    b.classList.toggle('active', on);
    if (on) b.classList.remove('has-new');
  }
  for (const p of $$('.tab-page')) p.classList.toggle('active', p.id === `tab-${name}`);
  bus.emit('tab:shown', name);
}

function initTabs() {
  for (const b of $$('#tabs button')) b.onclick = () => showTab(b.dataset.tab);
  bus.on('tab:new', (name) => {
    if (name === activeTab) return;
    const b = $(`#tabs button[data-tab="${name}"]`);
    if (b) b.classList.add('has-new');
  });
  bus.on('tab:show', (name) => showTab(name));
  // Jump to Charts / Sim the first time they get something in a run (not while viewing old runs).
  bus.on('tab:auto', (name) => {
    if (state.viewing || state.autoSwitched.has(name)) return;
    state.autoSwitched.add(name);
    if (name === 'sim' || (name === 'charts' && activeTab === 'console')) showTab(name);
  });
}

// --- status & buttons ----------------------------------------------------------------------
function setStatus(kind, text) {
  const s = $('#status');
  s.className = `status ${kind}`;
  s.textContent = text;
  const running = kind === 'running';
  $('#btn-run').classList.toggle('running', running);
  $('#btn-stop').disabled = !running;
  $('#btn-skip').disabled = !running;
  if (!running) $('#progress-line').style.width = '0';
}

let timer = null;
function startTimer() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (state.status !== 'running') return;
    const secs = (Date.now() - state.started) / 1000;
    if (Date.now() - state.lastProgress > 1500) $('#status').textContent = `Running · ${fmtTime(secs)}`;
  }, 1000);
}

// --- sounds (tiny synthesizer, no files needed) ----------------------------------------------
let audio = null;
export function playSound(name = 'ding') {
  try {
    audio ||= new (window.AudioContext || window.webkitAudioContext)();
    const notes = { ding: [[880, 0, 0.25], [1320, 0.12, 0.35]], success: [[523, 0, 0.15], [659, 0.1, 0.15], [784, 0.2, 0.15], [1046, 0.3, 0.4]],
      oops: [[440, 0, 0.18], [330, 0.15, 0.3]], pop: [[660, 0, 0.06]] }[name] || [[880, 0, 0.2]];
    const now = audio.currentTime;
    for (const [f, start, dur] of notes) {
      const o = audio.createOscillator();
      const g = audio.createGain();
      o.type = 'sine'; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, now + start);
      g.gain.exponentialRampToValueAtTime(0.25, now + start + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, now + start + dur);
      o.connect(g); g.connect(audio.destination);
      o.start(now + start); o.stop(now + start + dur + 0.05);
    }
  } catch (e) { /* audio blocked */ }
}

// --- running ---------------------------------------------------------------------------------
export async function run() {
  if (state.status === 'running') {
    if (!(await confirmBox('Still running', 'The program is still running. Stop it and start again?', 'Stop & run again'))) return;
    await stop();
    for (let i = 0; i < 50 && state.status === 'running'; i++) await new Promise((r) => setTimeout(r, 100));
  }
  const target = $('#target').value || 'local';
  const quick = $('#quick').checked;
  setStatus('running', 'Starting…');
  state.viewing = false;
  let res;
  try {
    res = await api.post('/api/run', { workspace: WS.save(), project: project.name, quick, target });
  } catch (e) {
    setStatus('error', 'Could not start');
    toast(e.message, 'err', 6000);
    return;
  }
  if (!res.ok) {
    setStatus('error', 'Fix the marked blocks');
    WS.applyDiagnostics(res.diagnostics || []);
    const errs = (res.diagnostics || []).filter((d) => d.level === 'error');
    bus.emit('run:reset', { project: project.name });
    for (const d of errs) bus.emit('ev:error', { message: d.message, block: d.block_id, hint: d.block_id ? 'The block is marked with a ⚠ warning sign.' : null });
    if (errs[0] && errs[0].block_id) WS.revealBlock(errs[0].block_id);
    showTab('console');
    playSound('oops');
    return;
  }
  WS.applyDiagnostics(res.diagnostics || []);
}

export async function stop() {
  try { await api.post('/api/stop', { run: state.runId }); } catch (e) { toast(e.message, 'err'); }
}

function skip() {
  send({ cmd: 'skip' });
  toast('Finishing the current training early…');
}

// --- event routing ------------------------------------------------------------------------------
function handle(ev, live = true) {
  switch (ev.type) {
    case 'run_start':
      state.runId = ev.run && ev.run.id;
      state.status = 'running';
      state.started = Date.now();
      state.autoSwitched = new Set();
      state.lastProgress = 0;
      bus.emit('run:reset', { project: ev.run && ev.run.project, quick: ev.quick, remote: ev.run && ev.run.remote });
      if (live) { setStatus('running', 'Running…'); startTimer(); }
      return;
    case 'block':
      if (live) WS.highlight(ev.id);
      break;
    case 'progress': {
      if (!live) break;
      const pct = ev.total ? Math.min(100, (100 * ev.current) / ev.total) : 0;
      $('#progress-line').style.width = ev.total && ev.current < ev.total ? `${pct}%` : '0';
      if (state.status === 'running') {
        $('#status').textContent = `${ev.label || ''} ${ev.total ? Math.round(pct) + '%' : ''}`.trim();
        state.lastProgress = Date.now();
      }
      break;
    }
    case 'error':
      if (live) {
        if (ev.block) WS.markError(ev.block, ev.message + (ev.hint ? `\n💡 ${ev.hint}` : ''));
        showTab('console');
        playSound('oops');
      }
      break;
    case 'sound':
      if (live) playSound(ev.name);
      break;
    case 'interactive':
      if (live) { $('#status').textContent = '🎮 Your turn — see the Play tab'; state.lastProgress = Date.now() + 1e9; }
      break;
    case 'interactive_end':
      state.lastProgress = 0;
      break;
    case 'done': {
      state.status = ev.status || 'finished';
      if (live) {
        WS.highlight(null);
        const secs = (Date.now() - state.started) / 1000;
        if (ev.status === 'error') setStatus('error', 'Error — see Console');
        else if (ev.status === 'stopped') setStatus('idle', 'Stopped');
        else {
          setStatus('finished', `Finished in ${fmtTime(ev.seconds ?? secs)}`);
          if (secs > 30) { playSound('success'); toast('✅ Finished!', 'ok'); }
        }
      }
      break;
    }
    default:
  }
  bus.emit(`ev:${ev.type}`, ev);
}

async function viewRun(runId) {
  state.viewing = true;
  let text;
  try { text = await api.text(`/api/runs/${encodeURIComponent(runId)}/events`); } catch (e) { toast(e.message, 'err'); return; }
  const events = text.split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l); } catch (_) { return null; } }).filter(Boolean);
  bus.emit('run:reset', { project: `results of ${runId}` });
  for (const ev of events) if (ev.type !== 'run_start') handle(ev, false);
  setStatus('idle', `Viewing ${runId}`);
  toast(`Showing ${events.length.toLocaleString()} recorded events from ${runId}`, 'ok', 4000);
}

// --- targets ---------------------------------------------------------------------------------------
export async function refreshTargets(selectId) {
  const sel = $('#target');
  const prev = selectId || sel.value || localStorage.getItem('nb-target') || 'local';
  let info = {};
  try { info = await api.get('/api/info'); } catch (e) { /* ignore */ }
  const d = info.device || {};
  const local = d.cuda ? `This computer (GPU: ${(d.gpus || [])[0] || 'CUDA'})` : d.mps ? 'This computer (Apple GPU)' : d.threads ? `This computer (CPU, ${d.threads} threads)` : 'This computer';
  sel.innerHTML = '';
  sel.append(el('option', { value: 'local' }, local));
  try {
    for (const t of await api.get('/api/targets')) {
      const g = t.device && t.device.gpus && t.device.gpus.length ? ` (${t.device.gpus[0]})` : '';
      sel.append(el('option', { value: t.id }, `☁️ ${t.name}${g}`));
    }
  } catch (e) { /* ignore */ }
  sel.append(el('option', { value: '__add' }, '➕ Add remote server…'));
  sel.value = [...sel.options].some((o) => o.value === prev) ? prev : 'local';
  if (!d.threads && !d.error) setTimeout(() => refreshTargets(), 2500);  // device probe still running
}

export function init() {
  initTabs();
  $('#btn-run').onclick = run;
  $('#btn-stop').onclick = stop;
  $('#btn-skip').onclick = skip;
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); run(); }
  });
  $('#target').onchange = (e) => {
    if (e.target.value === '__add') { e.target.value = 'local'; bus.emit('action', 'targets'); return; }
    localStorage.setItem('nb-target', e.target.value);
  };
  bus.on('targets:changed', (id) => refreshTargets(id));
  bus.on('run:event', ({ event }) => handle(event, true));
  bus.on('ws:hello', (hello) => {
    const cur = hello.current;
    if (!cur || !hello.backlog || !hello.backlog.length) return;
    // Reconnected (or page reloaded) while a run exists: replay what happened so far.
    for (const ev of hello.backlog) handle(ev, cur.status === 'running');
    if (cur.status !== 'running') setStatus(cur.status === 'error' ? 'error' : 'idle', cur.status === 'error' ? 'Last run had an error' : 'Ready');
  });
  bus.on('ws:close', () => { if (state.status === 'running') $('#status').textContent = 'Reconnecting…'; });
  bus.on('view-run', viewRun);
  refreshTargets();
}
