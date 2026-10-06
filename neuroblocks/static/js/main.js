// NeuroBlocks editor bootstrap.
import { api, connect } from './api.js';
import { registerAll } from './blocks.js';
import * as WS from './workspace.js';
import * as Project from './project.js';
import * as Runner from './runner.js';
import { $, bus, toast } from './util.js';
import * as ConsolePanel from './panels/console.js';
import * as ChartsPanel from './panels/charts.js';
import * as ResultsPanel from './panels/results.js';
import * as SimPanel from './panels/sim.js';
import * as PlayPanel from './panels/play.js';
import * as ModelPanel from './panels/model.js';
import * as DataPanel from './panels/data.js';
import * as CodePanel from './panels/code.js';

function initSplitter() {
  const split = $('#splitter');
  const panel = $('#panel');
  const saved = parseFloat(localStorage.getItem('nb-panel-width') || '0');
  if (saved > 200) panel.style.width = `${saved}px`;
  let dragging = false;
  split.addEventListener('pointerdown', (e) => { dragging = true; split.classList.add('drag'); split.setPointerCapture(e.pointerId); });
  split.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const w = Math.min(window.innerWidth * 0.75, Math.max(320, window.innerWidth - e.clientX));
    panel.style.width = `${w}px`;
    WS.resize();
  });
  split.addEventListener('pointerup', () => {
    dragging = false;
    split.classList.remove('drag');
    localStorage.setItem('nb-panel-width', String(panel.getBoundingClientRect().width));
    WS.resize();
    window.dispatchEvent(new Event('resize'));
  });
}

async function boot() {
  let bundle;
  try {
    bundle = await api.get('/api/blocks');
  } catch (e) {
    document.body.innerHTML = `<div class="empty"><span class="big">⚠️</span>Couldn't reach the NeuroBlocks server: ${e.message}</div>`;
    return;
  }
  const theme = registerAll(bundle);
  WS.inject(bundle, theme);
  initSplitter();
  Project.init();
  Runner.init();
  for (const p of [ConsolePanel, ChartsPanel, ResultsPanel, SimPanel, PlayPanel, ModelPanel, DataPanel, CodePanel]) p.init();

  const params = new URLSearchParams(location.search);
  if (!Project.restore()) {
    Project.loadProject({ name: 'My first AI', workspace: WS.starter() });
    setTimeout(() => Project.helpDialog(), 400);
  }
  connect();
  if (params.get('view')) setTimeout(() => bus.emit('view-run', params.get('view')), 300);
  if (params.get('example')) {
    try { Project.loadProject(await api.get(`/api/examples/${encodeURIComponent(params.get('example'))}`), { fromExample: true }); } catch (e) { toast(e.message, 'err'); }
  }
  setTimeout(() => WS.resize(), 50);
}

boot();
