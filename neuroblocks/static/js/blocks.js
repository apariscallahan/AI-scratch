// Registers NeuroBlocks' blocks with Blockly: theme, custom fields, definitions, help.
/* global Blockly */
import { el, modal } from './util.js';

export let BUNDLE = null;
export let DEFINERS = {};

// A text field that names a thing (dataset / model / world) the block creates.
class FieldNbName extends Blockly.FieldTextInput {
  constructor(value, validator, config) {
    super(value, validator, config);
    this.kind = (config && config.kind) || 'model';
    this.SERIALIZABLE = true;
  }
  static fromJson(options) {
    return new FieldNbName(options.text ?? '', undefined, options);
  }
  doClassValidation_(v) {
    if (typeof v !== 'string') return null;
    v = v.replace(/[\r\n\t]/g, ' ').slice(0, 40);
    return v.trim() ? v : null;
  }
}

// A dropdown listing the names of things of one kind defined anywhere in the project.
class FieldNbRef extends Blockly.FieldDropdown {
  constructor(kind, value, config) {
    super(function () { return this.nbOptions(); }, undefined, config);
    this.kind = kind;
    if (value) this.setValue(value);
  }
  static fromJson(options) {
    return new FieldNbRef(options.kind, options.value);
  }
  nbOptions() {
    const cur = this.getValue ? this.getValue() : null;
    if (!this.kind) return [[cur || '?', cur || '?']];
    const names = new Set();
    const block = this.getSourceBlock && this.getSourceBlock();
    let ws = block && block.workspace;
    if (ws && ws.isFlyout && ws.targetWorkspace) ws = ws.targetWorkspace;
    if (ws && ws.getAllBlocks) {
      for (const b of ws.getAllBlocks(false)) {
        const d = DEFINERS[b.type];
        if (d && d.kind === this.kind && !b.isInFlyout) {
          const v = b.getFieldValue(d.field);
          if (v) names.add(v);
        }
      }
    }
    if (cur) names.add(cur);
    if (!names.size) names.add(this.kind === 'dataset' ? 'data' : this.kind);
    return [...names].sort((a, b) => a.localeCompare(b)).map((n) => [n, n]);
  }
  getOptions() { return this.nbOptions(); }
  doClassValidation_(v) { return (typeof v === 'string' && v.length) ? v : null; }
}

// A wire socket, as in node editors: ● 'out' on the 📦 weights block, ○ 'in' on 'start output'.
// The in-port's value is the id of the weights block it is wired to ('' = not wired).
// Dragging from a port draws a wire (wires.js) instead of moving the block.
const PORT_SIZE = 30;
export const portHandlers = { down: null }; // set by wires.js

export class FieldNbPort extends Blockly.Field {
  constructor(value, validator, config) {
    super(value ?? '', validator, config);
    this.direction = (config && config.direction) === 'out' ? 'out' : 'in';
    this.SERIALIZABLE = this.direction === 'in';
    this.EDITABLE = false;
    this.connected = false;
  }
  static fromJson(options) {
    return new FieldNbPort(options.value ?? '', undefined, options);
  }
  doClassValidation_(v) { return typeof v === 'string' ? v : ''; }
  initView() {
    const svg = Blockly.utils.dom.createSvgElement;
    const c = PORT_SIZE / 2;
    this.fieldGroup_.classList.add('nb-port-field');
    this.ring = svg('circle', { class: `nb-port nb-port-${this.direction}`, cx: c, cy: c, r: 9 }, this.fieldGroup_);
    this.dot = svg('circle', { class: 'nb-port-dot', cx: c, cy: c, r: 4.5 }, this.fieldGroup_);
    this.hit = svg('circle', { class: 'nb-port-hit', cx: c, cy: c, r: 18 }, this.fieldGroup_);
    const tip = svg('title', {}, this.hit);
    tip.textContent = this.direction === 'out'
      ? 'Drag a wire from here to the ○ of a "start output" block'
      : 'Drop a wire from a 📦 weights block here (drag away to unplug it)';
    this.hit.addEventListener('pointerdown', (e) => {
      const block = this.getSourceBlock();
      if (!block || block.isInFlyout || !portHandlers.down) return; // in the toolbox: drag the block as usual
      e.stopPropagation();
      e.preventDefault();
      portHandlers.down(this, e);
    });
    this.setConnected(this.connected);
  }
  updateSize_() { this.size_ = new Blockly.utils.Size(PORT_SIZE, PORT_SIZE); }
  render_() { this.updateSize_(); }
  getText() { return this.direction === 'out' ? '●' : (this.getValue() ? '●' : '○'); }
  setConnected(on) {
    this.connected = !!on;
    if (this.ring) {
      this.ring.classList.toggle('on', this.connected);
      this.dot.style.display = (this.direction === 'out' || this.connected) ? '' : 'none';
    }
  }
  // Centre of the socket on screen (null while hidden, e.g. in a collapsed block).
  screenCenter() {
    if (!this.hit || !this.hit.isConnected) return null;
    const r = this.hit.getBoundingClientRect();
    if (!r.width) return null;
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }
}

function registerFields() {
  try { Blockly.fieldRegistry.register('field_nbname', FieldNbName); } catch (e) { /* already */ }
  try { Blockly.fieldRegistry.register('field_nbref', FieldNbRef); } catch (e) { /* already */ }
  try { Blockly.fieldRegistry.register('field_nbport', FieldNbPort); } catch (e) { /* already */ }
}

function makeTheme(bundle) {
  const blockStyles = {};
  const categoryStyles = {};
  for (const c of bundle.categories) {
    const s = { colourPrimary: c.colour, colourSecondary: c.secondary, colourTertiary: c.tertiary };
    blockStyles[`nb_${c.key}`] = s;
    blockStyles[`nb_${c.key}_hat`] = { ...s, hat: 'cap' };
    categoryStyles[`nb_${c.key}_category`] = { colour: c.colour };
  }
  for (const [style, colour] of Object.entries(bundle.builtin_styles)) {
    const c = bundle.categories.find((x) => x.colour === colour) || bundle.categories[0];
    blockStyles[style] = { colourPrimary: colour, colourSecondary: c.secondary, colourTertiary: c.tertiary };
  }
  return Blockly.Theme.defineTheme('neuroblocks', {
    name: 'neuroblocks',
    base: Blockly.Themes.Zelos || Blockly.Themes.Classic,
    blockStyles,
    categoryStyles,
    componentStyles: {
      workspaceBackgroundColour: '#f9f9fc',
      toolboxBackgroundColour: '#ffffff',
      toolboxForegroundColour: '#575e75',
      flyoutBackgroundColour: '#f2f3f8',
      flyoutForegroundColour: '#575e75',
      flyoutOpacity: 1,
      scrollbarColour: '#cdd2e6',
      scrollbarOpacity: 0.7,
      insertionMarkerColour: '#000000',
      insertionMarkerOpacity: 0.2,
    },
    fontStyle: { family: '"Helvetica Neue", Helvetica, Arial, sans-serif', weight: '500', size: 12 },
    startHats: false,
  });
}

export function showBlockHelp(type) {
  const h = BUNDLE && BUNDLE.help[type];
  if (!h) return;
  const cat = BUNDLE.categories.find((c) => c.key === h.category);
  const body = el('div', {},
    cat ? el('div', { style: { display: 'inline-block', background: cat.colour, color: '#fff', borderRadius: '8px', padding: '2px 10px', fontWeight: 700, fontSize: '12px' } }, cat.name) : null,
    el('p', { style: { fontSize: '15px' } }, h.tooltip || ''),
    h.help ? el('div', { class: 'help-block' }, h.help) : null,
  );
  modal({ title: 'About this block', body, buttons: [{ label: 'Got it', kind: 'primary' }] });
}

function registerContextMenu() {
  const reg = Blockly.ContextMenuRegistry.registry;
  try { reg.unregister('blockHelp'); } catch (e) { /* not present */ }
  try {
    reg.register({
      id: 'nb_help',
      scopeType: Blockly.ContextMenuRegistry.ScopeType.BLOCK,
      displayText: () => '❓ What does this block do?',
      weight: -20,
      preconditionFn: (scope) => {
        const b = scope.block || scope.focusedNode;
        return b && BUNDLE.help[b.type] ? 'enabled' : 'hidden';
      },
      callback: (scope) => showBlockHelp((scope.block || scope.focusedNode).type),
    });
  } catch (e) { console.warn('context menu', e); }
}

export function registerAll(bundle) {
  BUNDLE = bundle;
  DEFINERS = bundle.definers || {};
  registerFields();
  Blockly.common.defineBlocksWithJsonArray(bundle.blocks);
  registerContextMenu();
  return makeTheme(bundle);
}
