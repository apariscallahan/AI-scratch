// The Blockly workspace: injection, live checks, name propagation, highlighting.
/* global Blockly */
import { api } from './api.js';
import { BUNDLE, DEFINERS } from './blocks.js';
import { bus, debounce } from './util.js';

export let workspace = null;
let suppress = 0;
let lastWarnings = new Set();

export function inject(bundle, theme) {
  workspace = Blockly.inject('blockly-div', {
    toolbox: bundle.toolbox,
    renderer: 'zelos',
    theme,
    media: '/static/vendor/blockly/media/',
    zoom: { controls: true, wheel: true, startScale: 0.68, maxScale: 2, minScale: 0.25, scaleSpeed: 1.12, pinch: true },
    move: { scrollbars: true, drag: true, wheel: true },
    grid: { spacing: 40, length: 2, colour: '#dde1ee', snap: false },
    trashcan: true,
    sounds: true,
    oneBasedIndex: true,
  });
  workspace.registerToolboxCategoryCallback('NB_VARIABLES', (ws) => {
    let base = [];
    try { base = Blockly.Variables.flyoutCategory(ws); } catch (e) { console.warn(e); }
    return [...base, { kind: 'label', text: 'Settings you can change from the command line' },
      ...bundle.variables_extra.map((b) => JSON.parse(JSON.stringify(b)))];
  });
  workspace.addChangeListener(onChange);
  window.addEventListener('resize', () => Blockly.svgResize(workspace));
  return workspace;
}

export function resize() {
  if (workspace) Blockly.svgResize(workspace);
}

export function save() {
  return Blockly.serialization.workspaces.save(workspace);
}

export function load(json) {
  suppress++;
  try {
    workspace.clear();
    if (json && Object.keys(json).length) Blockly.serialization.workspaces.load(json, workspace);
  } finally {
    suppress--;
  }
  setTimeout(() => {
    workspace.scrollCenter();
    const tops = workspace.getTopBlocks(true);
    if (tops.length) {
      const hat = tops.find((b) => b.type === 'nb_when_run') || tops[0];
      const xy = hat.getRelativeToSurfaceXY();
      workspace.scroll(-xy.x * workspace.scale + 60, -xy.y * workspace.scale + 40);
    }
  }, 30);
  scheduleCheck();
}

export function starter() {
  return {
    blocks: {
      languageVersion: 0,
      blocks: [{
        type: 'nb_when_run', x: 60, y: 60,
        next: { block: { type: 'nb_say', inputs: { TEXT: { shadow: { type: 'text', fields: { TEXT: 'Hello! Drag blocks from the left to build an AI.' } } } } } },
      }],
    },
  };
}

// --- change handling ------------------------------------------------------------
const UI_EVENTS = new Set([Blockly.Events.UI, Blockly.Events.CLICK, Blockly.Events.SELECTED, Blockly.Events.VIEWPORT_CHANGE,
  Blockly.Events.TOOLBOX_ITEM_SELECT, Blockly.Events.THEME_CHANGE, Blockly.Events.BUBBLE_OPEN, Blockly.Events.TRASHCAN_OPEN,
  Blockly.Events.BLOCK_DRAG].filter(Boolean));

function onChange(e) {
  if (UI_EVENTS.has(e.type) || e.isUiEvent) return;
  if (e.type === Blockly.Events.BLOCK_CHANGE && e.element === 'field') propagateRename(e);
  if (suppress) return;
  bus.emit('workspace:changed', e);
  scheduleCheck();
}

function propagateRename(e) {
  const block = workspace.getBlockById(e.blockId);
  if (!block) return;
  const d = DEFINERS[block.type];
  if (!d || d.field !== e.name) return;
  const oldName = e.oldValue;
  const newName = e.newValue;
  if (!oldName || oldName === newName) return;
  // Another block still defines the old name? Then leave references alone.
  const stillDefined = workspace.getAllBlocks(false).some((b) => b.id !== block.id && DEFINERS[b.type]
    && DEFINERS[b.type].kind === d.kind && b.getFieldValue(DEFINERS[b.type].field) === oldName);
  if (stillDefined) return;
  const refs = BUNDLE.refs || {};
  Blockly.Events.setGroup(e.group || true);
  try {
    for (const b of workspace.getAllBlocks(false)) {
      const fields = refs[b.type];
      if (!fields) continue;
      for (const [fname, kind] of Object.entries(fields)) {
        if (kind === d.kind && b.getFieldValue(fname) === oldName) b.setFieldValue(newName, fname);
      }
    }
  } finally {
    Blockly.Events.setGroup(false);
  }
}

// --- live checks: compile in the background and mark problem blocks ------------------
export const scheduleCheck = debounce(check, 700);
let checkSeq = 0;

export async function check() {
  if (!workspace) return null;
  const seq = ++checkSeq;
  let res;
  try {
    res = await api.post('/api/compile', { workspace: save(), project: bus.projectName ? bus.projectName() : 'Untitled' });
  } catch (e) {
    return null;
  }
  if (seq !== checkSeq) return res;
  applyDiagnostics(res.diagnostics || []);
  bus.emit('compiled', res);
  return res;
}

export function applyDiagnostics(diags) {
  const byBlock = new Map();
  for (const d of diags) {
    if (!d.block_id || d.level === 'info') continue;
    const prev = byBlock.get(d.block_id);
    byBlock.set(d.block_id, prev ? `${prev}\n${d.message}` : d.message);
  }
  Blockly.Events.disable();
  try {
    for (const id of lastWarnings) {
      if (!byBlock.has(id)) {
        const b = workspace.getBlockById(id);
        if (b) b.setWarningText(null);
      }
    }
    for (const [id, msg] of byBlock) {
      const b = workspace.getBlockById(id);
      if (b) b.setWarningText(msg);
    }
  } finally {
    Blockly.Events.enable();
  }
  lastWarnings = new Set(byBlock.keys());
}

// --- running block glow & errors ------------------------------------------------------
let glowing = null;
export function highlight(id) {
  if (!workspace) return;
  if (glowing === id) return;
  glowing = id;
  try { workspace.highlightBlock(id || null); } catch (e) { /* block removed */ }
}

export function markError(id, message) {
  const b = id && workspace.getBlockById(id);
  if (!b) return false;
  Blockly.Events.disable();
  try { b.setWarningText(message); } finally { Blockly.Events.enable(); }
  lastWarnings.add(id);
  revealBlock(id);
  return true;
}

export function revealBlock(id) {
  const b = id && workspace.getBlockById(id);
  if (!b) return;
  try {
    workspace.centerOnBlock(id);
    b.select && b.select();
    workspace.highlightBlock(id);
    setTimeout(() => { if (glowing !== id) workspace.highlightBlock(glowing || null); }, 1600);
  } catch (e) { /* ignore */ }
}
