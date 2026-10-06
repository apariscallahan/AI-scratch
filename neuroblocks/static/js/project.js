// Projects: new / open / save / examples / exports / past runs / remote servers / help.
import { api } from './api.js';
import { $, ask, bus, confirmBox, debounce, downloadBlob, downloadText, el, modal, toast } from './util.js';
import * as WS from './workspace.js';

export const project = { name: 'Untitled project', file: null, description: '', dirty: false };
const AUTOSAVE_KEY = 'neuroblocks-autosave-v1';

function setTitle(name) {
  project.name = name || 'Untitled project';
  $('#project-title').value = project.name;
  document.title = `${project.name} · NeuroBlocks`;
}

function setDirty(d) {
  project.dirty = d;
  $('#save-state').textContent = d ? (project.file ? 'unsaved changes' : '') : (project.file ? 'saved ✓' : '');
}

export function projectJson() {
  return { format: 'neuroblocks-project', version: 1, name: project.name, description: project.description, workspace: WS.save() };
}

const autosave = debounce(() => {
  try {
    localStorage.setItem(AUTOSAVE_KEY, JSON.stringify({ ...projectJson(), file: project.file, when: Date.now() }));
  } catch (e) { /* storage full or blocked */ }
}, 800);

export function loadProject(data, { file = null, fromExample = false } = {}) {
  try {
    WS.load(data.workspace || {});
  } catch (e) {
    console.error(e);
    toast(`Some blocks in “${data.name || 'this project'}” couldn't be loaded: ${String(e.message || e).split('\n')[0]}`,
      'err', 8000);
  }
  setTitle(data.name || 'Untitled project');
  project.description = data.description || '';
  project.file = file;
  setDirty(false);
  autosave();
  bus.emit('project:loaded', data);
  if (fromExample && data.description) toast(`Loaded “${data.name}”. ${data.minutes ? `Takes about ${data.minutes} min on a laptop.` : ''}`, 'ok', 4200);
}

export function restore() {
  try {
    const raw = localStorage.getItem(AUTOSAVE_KEY);
    if (raw) {
      const d = JSON.parse(raw);
      loadProject(d, { file: d.file || null });
      if (d.file) setDirty(false);
      return true;
    }
  } catch (e) { /* ignore */ }
  return false;
}

export async function newProject() {
  if (project.dirty && !(await confirmBox('New project', 'Start a new project? Unsaved changes will be lost (they are in the autosave until you change something).', 'Start new'))) return;
  loadProject({ name: 'Untitled project', workspace: WS.starter() });
}

export async function save(asNew = false) {
  let file = project.file;
  if (!file || asNew) {
    const name = await ask('Save project', 'Project name', project.name, 'Save');
    if (!name) return;
    setTitle(name);
    file = name.replace(/[^A-Za-z0-9 _-]+/g, '').trim().replace(/\s+/g, '_').toLowerCase() || 'project';
    file += '.nblk';
  }
  try {
    const res = await api.put(`/api/projects/${encodeURIComponent(file)}`, projectJson());
    project.file = res.file;
    setDirty(false);
    autosave();
    toast(`Saved ${res.file}`, 'ok');
  } catch (e) {
    toast(`Save failed: ${e.message}`, 'err');
  }
}

export async function openDialog() {
  const list = await api.get('/api/projects');
  const body = el('div');
  if (!list.length) body.append(el('p', {}, 'No saved projects yet. Use File ▸ Save, or start from an example.'));
  for (const p of list) {
    const row = el('div', { class: 'proj-row' }, el('span', {}, '📄'), el('span', { class: 'nm' }, p.name),
      el('span', { class: 'dt' }, new Date(p.modified * 1000).toLocaleString()),
      el('button', { class: 'small-btn', title: 'Move to trash', onclick: async (e) => {
        e.stopPropagation();
        if (await confirmBox('Delete project', `Move “${p.name}” to the trash folder?`, 'Delete', 'danger')) { await api.del(`/api/projects/${encodeURIComponent(p.file)}`); row.remove(); }
      } }, '🗑'));
    row.onclick = async () => {
      m.close();
      const data = await api.get(`/api/projects/${encodeURIComponent(p.file)}`);
      loadProject(data, { file: p.file });
    };
    body.append(row);
  }
  const m = modal({ title: '📂 Open a project', body, buttons: [{ label: 'Open file from computer…', onClick: () => { uploadProject(); } }, { label: 'Close' }] });
}

function uploadProject() {
  const input = el('input', { type: 'file', accept: '.nblk,.json' });
  input.onchange = async () => {
    const f = input.files[0];
    if (!f) return;
    try {
      const data = JSON.parse(await f.text());
      loadProject(data.workspace ? data : { name: f.name.replace(/\.[^.]+$/, ''), workspace: data });
      toast(`Opened ${f.name}`, 'ok');
    } catch (e) { toast(`Couldn't read ${f.name}: ${e.message}`, 'err'); }
  };
  input.click();
}

export async function examplesDialog() {
  const list = await api.get('/api/examples');
  const body = el('div');
  body.append(el('p', { style: { marginTop: 0 } }, 'Pick an example to load it, look at the blocks, then press ▶ Run. Change things and see what happens!'));
  const cats = [...new Set(list.map((e) => e.category))];
  for (const c of cats) {
    body.append(el('div', { class: 'ex-cat' }, c));
    const grid = el('div', { class: 'ex-grid' });
    for (const e of list.filter((x) => x.category === c)) {
      grid.append(el('div', { class: 'ex-card', onclick: async () => {
        if (project.dirty && !(await confirmBox('Load example', 'Replace the current blocks with this example? (Save first if you want to keep them.)', 'Load example'))) return;
        m.close();
        const data = await api.get(`/api/examples/${encodeURIComponent(e.id)}`);
        loadProject(data, { fromExample: true });
      } }, el('div', { class: 'nm' }, e.name), el('div', { class: 'ds' }, e.description),
        el('div', { class: 'meta' }, [e.level, e.minutes ? `~${e.minutes} min on a laptop` : ''].filter(Boolean).join(' · '))));
    }
    body.append(grid);
  }
  const m = modal({ title: '✨ Examples', body, wide: true });
}

async function exportPython() {
  const res = await api.post('/api/export/python', { workspace: WS.save(), project: project.name });
  if (!res.ok) toast('Note: the program has errors (see the red blocks).', 'err');
  downloadText(res.code, res.filename, 'text/x-python');
}

async function exportBundle() {
  try {
    const r = await api.raw('/api/export/bundle', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace: WS.save(), project: project.name, description: project.description }),
    });
    const blob = await r.blob();
    const cd = r.headers.get('content-disposition') || '';
    const fname = (cd.match(/filename="([^"]+)"/) || [])[1] || 'bundle.zip';
    downloadBlob(blob, fname);
    let files = [];
    try { files = JSON.parse(r.headers.get('x-bundle-files') || '[]'); } catch (e) { /* ignore */ }
    modal({
      title: '☁️ Cloud GPU bundle ready',
      body: `<p>Downloaded <b>${fname}</b>. It contains your blocks, the same program as Python, the NeuroBlocks runtime${files.length ? `, and ${files.length} data/model file(s)` : ''}.</p>
        <p>On a Linux machine with a GPU (Lambda, RunPod, Vast.ai, Colab, …):</p>
        <pre class="gen-text">unzip ${fname} -d job && cd job
bash run.sh --quick     # 1-minute smoke test
bash run.sh             # the real run, on the GPU</pre>
        <p>Trained models end up in <code>models/</code> — copy them back and use <i>load model from file</i>.
        Open the run's charts with <code>neuroblocks view runs/&lt;folder&gt;</code>.</p>
        <p>Want live charts while it trains? Run <code>python -m neuroblocks serve --host 0.0.0.0</code> there and add it under File ▸ Remote servers.</p>`,
      buttons: [{ label: 'Great', kind: 'primary' }],
    });
  } catch (e) {
    toast(`Export failed: ${e.message}`, 'err', 5000);
  }
}

function downloadProject() {
  const name = (project.name || 'project').replace(/[^A-Za-z0-9 _-]+/g, '').trim().replace(/\s+/g, '_') || 'project';
  downloadText(JSON.stringify(projectJson(), null, 1), `${name}.nblk`, 'application/json');
}

async function runsDialog() {
  const list = await api.get('/api/runs');
  const body = el('div');
  body.append(el('p', { style: { marginTop: 0 } }, 'Every run is logged. Open one to see its charts, results and replays again (also works for runs copied back from a cloud GPU into the runs folder).'));
  if (!list.length) body.append(el('p', {}, 'No runs yet.'));
  for (const r of list) {
    const row = el('div', { class: 'proj-row' }, el('span', {}, '🕘'), el('span', { class: 'nm' }, r.project || r.id), el('span', { class: 'dt' }, r.id.slice(0, 15)),
      r.has_events ? el('button', { class: 'small-btn', onclick: async (e) => { e.stopPropagation(); m.close(); bus.emit('view-run', r.id); } }, 'View results') : null,
      el('button', { class: 'small-btn', onclick: async (e) => {
        e.stopPropagation();
        try { const p = await api.get(`/api/runs/${encodeURIComponent(r.id)}/project`); m.close(); loadProject(p); toast('Loaded the blocks used in that run'); } catch (err) { toast('No blocks were saved with that run', 'err'); }
      } }, 'Blocks'));
    body.append(row);
  }
  const m = modal({ title: '🕘 Past runs', body, wide: true, buttons: [{ label: 'Close' }] });
}

export async function targetsDialog() {
  const list = await api.get('/api/targets');
  const body = el('div');
  body.append(el('p', { style: { marginTop: 0 } }, 'Run your blocks on a more powerful computer (like a cloud GPU) while you watch here. On that machine run:'));
  body.append(el('pre', { class: 'gen-text' }, 'pip install <the neuroblocks bundle or folder>\npython -m neuroblocks serve --host 0.0.0.0 --port 8765\n# it prints an access token'));
  body.append(el('p', {}, 'Safer: keep the server private and tunnel over SSH: ssh -L 9000:localhost:8765 user@gpu-machine, then add http://localhost:9000 here.'));
  for (const t of list) {
    const dev = t.device && t.device.gpus && t.device.gpus.length ? t.device.gpus.join(', ') : (t.device && t.device.cuda === false ? 'CPU only' : '');
    body.append(el('div', { class: 'proj-row' }, el('span', {}, '🖥️'), el('span', { class: 'nm' }, `${t.name} — ${t.url}`), el('span', { class: 'dt' }, dev),
      el('button', { class: 'small-btn', onclick: async () => { await api.del(`/api/targets/${t.id}`); m.close(); bus.emit('targets:changed'); toast('Removed'); } }, 'Remove')));
  }
  const name = el('input', { class: 'text', placeholder: 'e.g. Big GPU' });
  const url = el('input', { class: 'text', placeholder: 'http://localhost:9000' });
  const token = el('input', { class: 'text', placeholder: 'token printed by "neuroblocks serve"' });
  body.append(el('label', { class: 'field' }, 'Name'), name, el('label', { class: 'field' }, 'Address (URL)'), url, el('label', { class: 'field' }, 'Access token'), token);
  const m = modal({
    title: '🖥️ Remote servers', body, buttons: [{ label: 'Close' }, {
      label: 'Add server', kind: 'primary', onClick: async () => {
        try {
          const t = await api.post('/api/targets', { name: name.value, url: url.value, token: token.value });
          toast(`Connected to ${t.name}`, 'ok');
          bus.emit('targets:changed', t.id);
          return false;
        } catch (e) { toast(e.message, 'err', 5000); return true; }
      },
    }],
  });
}

export function helpDialog() {
  modal({
    title: '👋 Welcome to NeuroBlocks', wide: true,
    body: `<p><b>NeuroBlocks</b> is like Scratch, but for building artificial intelligence. A project has two stacks,
      joined by a wire: one that <b>trains</b> a model and one that <b>uses</b> it.</p>
      <div class="help-block"><b>1 · Training</b> — under the yellow <b>start training when ▶ clicked</b><ol style="margin:6px 0 0">
        <li><b>Data</b> — load or make a dataset (toy dots, tables, images, text…)</li>
        <li><b>Neural Nets / Layers</b> — build a model by stacking layers (or use a ready-made GPT)</li>
        <li><b>Training</b> — <i>train model on data</i>, with settings inside it; <b>Test</b> — how good is it?</li>
        <li>End the stack with <b>📦 weights of trained …</b> (from <b>Output</b>): that's what training produces.</li></ol></div>
      <div class="help-block"><b>2 · Output</b> — drag a wire from the 📦 block's <b>●</b> to the <b>○</b> of a teal
        <b>start output</b> block, then build what the trained model should do, step by step. For a GPT: turn text into
        <i>tokens</i>, get <i>scores for the next token</i>, turn them into <i>probabilities</i>, <i>pick</i> one, add it,
        repeat. Results appear in the <b>Output</b> tab. Output stacks run after training has finished.</div>
      <div class="help-block"><b>Simulations</b>: build a world (a car on rough terrain, a rocket, a maze…), give it abilities,
        senses and rewards, then <i>train model to play in world</i>. Its output stack senses → runs the model → acts.
        Replays show up in the <b>Sim</b> tab.</div>
      <div class="help-block"><b>Tips</b><ul style="margin:6px 0 0">
        <li>Wires: drag from a <b>○</b> to unplug or move a wire; double-click a wire to remove it.</li>
        <li>Right-click a block → <i>What does this block do?</i></li>
        <li>Blocks only fit where they make sense — settings go <i>inside</i> the block they configure.</li>
        <li><b>quick test</b> runs every training for just a few steps, to check everything works.</li>
        <li><b>⏭ Skip</b> ends the current training early and carries on.</li>
        <li>The <b>Python</b> tab shows your blocks as real code. File ▸ <i>Export cloud GPU bundle</i> makes a zip you can run on a big GPU.</li>
        <li>Shortcuts: <span class="kbd-key">Ctrl</span>+<span class="kbd-key">Enter</span> run · <span class="kbd-key">Ctrl</span>+<span class="kbd-key">S</span> save · <span class="kbd-key">Ctrl</span>+<span class="kbd-key">Z</span> undo</li></ul></div>`,
    buttons: [{ label: 'Show me the examples', onClick: () => { setTimeout(examplesDialog, 50); } }, { label: "Let's go!", kind: 'primary' }],
  });
}

export function init() {
  setTitle(project.name);
  $('#project-title').addEventListener('change', (e) => { setTitle(e.target.value.trim()); setDirty(true); autosave(); });
  bus.projectName = () => project.name;
  bus.workspaceJson = () => WS.save();
  bus.on('workspace:changed', () => { setDirty(true); autosave(); });
  const actions = {
    new: newProject, open: openDialog, save: () => save(false), 'save-as': () => save(true), download: downloadProject,
    upload: uploadProject, 'export-py': exportPython, 'export-bundle': exportBundle, runs: runsDialog, targets: targetsDialog,
  };
  bus.on('action', (name) => actions[name] && actions[name]());
  const menu = $('#menu-file');
  menu.querySelector('[data-menu]').onclick = (e) => { e.stopPropagation(); menu.classList.toggle('open'); };
  document.addEventListener('click', () => menu.classList.remove('open'));
  for (const b of menu.querySelectorAll('[data-act]')) b.onclick = () => { menu.classList.remove('open'); bus.emit('action', b.dataset.act); };
  $('#btn-examples').onclick = examplesDialog;
  $('#btn-help').onclick = helpDialog;
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') { e.preventDefault(); save(false); }
  });
  window.addEventListener('beforeunload', () => autosave());
}
