// Sim tab: plays back physics replays sent by the runtime (and powers the live game view).
import { $, bus, el, fmtNum } from '../util.js';

function shade(hex, amt) {
  const h = (hex || '#888888').replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  r = Math.max(0, Math.min(255, Math.round(r + amt * 255)));
  g = Math.max(0, Math.min(255, Math.round(g + amt * 255)));
  b = Math.max(0, Math.min(255, Math.round(b + amt * 255)));
  return `rgb(${r},${g},${b})`;
}

export class SimRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.cam = null;
    this.follow = true;
    this.replay = null;
  }

  setReplay(replay) {
    this.replay = replay;
    this.cam = null;
    const ground = (replay.scene.static || []).find((s) => s.type === 'ground');
    this.ground = ground ? ground.points : null;
  }

  groundYAt(x) {
    const g = this.ground;
    if (!g || !g.length) return 0;
    let lo = 0, hi = g.length - 1;
    if (x <= g[0][0]) return g[0][1];
    if (x >= g[hi][0]) return g[hi][1];
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (g[m][0] <= x) lo = m; else hi = m; }
    const t = (x - g[lo][0]) / ((g[hi][0] - g[lo][0]) || 1);
    return g[lo][1] + t * (g[hi][1] - g[lo][1]);
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, this.canvas.clientWidth);
    const h = Math.max(10, this.canvas.clientHeight);
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    this.W = w; this.H = h; this.dpr = dpr;
  }

  leaderPose(frame) {
    const a0 = frame && frame[0];
    return a0 ? [a0[0], a0[1]] : [0, 0];
  }

  camera(frame, jump) {
    const sc = this.replay.scene;
    const cam = sc.camera || {};
    const b = sc.bounds || [-10, -2, 10, 10];
    if (cam.mode === 'follow' && this.follow) {
      const width = cam.width || 22;
      const s = this.W / width;
      const [lx, ly] = this.leaderPose(frame);
      const visH = this.H / s;
      const tx = lx + width * 0.12;
      const gy = this.ground ? this.groundYAt(lx) : ly;
      const ty = Math.max(ly, gy) + visH * 0.12;
      if (!this.cam || jump) this.cam = { x: tx, y: ty, s };
      else {
        this.cam.x += (tx - this.cam.x) * 0.15;
        this.cam.y += (ty - this.cam.y) * 0.08;
        this.cam.s = s;
      }
    } else {
      const pad = 0.06;
      const bw = (b[2] - b[0]) * (1 + 2 * pad);
      const bh = (b[3] - b[1]) * (1 + 2 * pad);
      const s = Math.min(this.W / bw, this.H / bh);
      this.cam = { x: (b[0] + b[2]) / 2, y: (b[1] + b[3]) / 2, s };
    }
    return this.cam;
  }

  X(x) { return (x - this.cam.x) * this.cam.s + this.W / 2; }
  Y(y) { return this.H / 2 - (y - this.cam.y) * this.cam.s; }

  draw(frameIndex, opts = {}) {
    const r = this.replay;
    if (!r) return;
    this.resize();
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const frames = r.frames || [];
    const fi = Math.max(0, Math.min(frames.length - 1, frameIndex | 0));
    const frame = frames[fi] || [];
    this.camera(frame, opts.jump);
    const sc = r.scene;
    // Sky.
    const sky = ctx.createLinearGradient(0, 0, 0, this.H);
    sky.addColorStop(0, sc.sky || '#bfe6ff');
    sky.addColorStop(1, '#f4fbff');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, this.W, this.H);
    if ((sc.camera || {}).mode === 'follow') this.drawParallax();
    for (const s of sc.static || []) this.drawStatic(s);
    if (r.kind === 'car' || sc.finish_x != null) this.drawMarkers(sc.finish_x);
    if (r.overlay && r.overlay.arrows) this.drawArrows(r.overlay.arrows);
    // Agents: ghosts first, leader (index 0) last so it is on top.
    const agents = r.agents || [{ alpha: 1 }];
    const order = [...frame.keys()].reverse();
    for (const ai of order) {
      const pose = frame[ai];
      if (!pose) continue;
      const alpha = (agents[ai] && agents[ai].alpha != null) ? agents[ai].alpha : (ai === 0 ? 1 : 0.3);
      if (opts.leaderOnly && ai > 0) continue;
      ctx.globalAlpha = alpha;
      const fx = r.fx && r.fx[fi] && r.fx[fi][ai];
      if (fx) this.drawEffects(pose, fx);
      (sc.bodies || []).forEach((body, bi) => this.drawBody(body, pose[bi * 3], pose[bi * 3 + 1], pose[bi * 3 + 2]));
      ctx.globalAlpha = 1;
    }
    // HUD.
    const hud = r.hud && r.hud[Math.min(fi, r.hud.length - 1)];
    const tsec = (opts.time != null ? opts.time : fi * (r.dt || 1 / 30)).toFixed(1);
    this.drawHud(hud ? `${hud}` : '', `${tsec} s`);
  }

  drawParallax() {
    const ctx = this.ctx;
    const off = this.cam.x * this.cam.s * 0.25;
    ctx.fillStyle = 'rgba(120,170,210,0.35)';
    ctx.beginPath();
    ctx.moveTo(0, this.H);
    for (let px = 0; px <= this.W + 20; px += 20) {
      const wx = (px + off) / 120;
      const y = this.H * 0.55 - 40 * Math.sin(wx) - 25 * Math.sin(wx * 2.3 + 1);
      ctx.lineTo(px, y);
    }
    ctx.lineTo(this.W, this.H);
    ctx.fill();
  }

  drawStatic(s) {
    const ctx = this.ctx;
    const k = this.cam.s;
    switch (s.type) {
      case 'ground': {
        const pts = s.points || [];
        if (!pts.length) return;
        const x0 = this.cam.x - this.W / k, x1 = this.cam.x + this.W / k;
        ctx.beginPath();
        let started = false;
        let first = null, last = null;
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          const nxt = pts[i + 1];
          if (p[0] < x0 && nxt && nxt[0] < x0) continue;
          if (p[0] > x1 && last) break;
          const X = this.X(p[0]), Y = this.Y(p[1]);
          if (!started) { ctx.moveTo(X, Y); started = true; first = X; } else ctx.lineTo(X, Y);
          last = X;
        }
        if (!started) return;
        const groundFill = this.replay.scene.ground_fill || '#8d6e63';
        ctx.lineTo(last, this.H + 10);
        ctx.lineTo(first, this.H + 10);
        ctx.closePath();
        ctx.fillStyle = groundFill;
        ctx.fill();
        ctx.beginPath();
        started = false;
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          const nxt = pts[i + 1];
          if (p[0] < x0 && nxt && nxt[0] < x0) continue;
          if (p[0] > x1 && started) { ctx.lineTo(this.X(p[0]), this.Y(p[1])); break; }
          if (!started) { ctx.moveTo(this.X(p[0]), this.Y(p[1])); started = true; } else ctx.lineTo(this.X(p[0]), this.Y(p[1]));
        }
        ctx.strokeStyle = s.color || '#5aa83c';
        ctx.lineWidth = Math.max(2, 0.16 * k);
        ctx.lineJoin = 'round';
        ctx.stroke();
        break;
      }
      case 'line': {
        const pts = s.points || [];
        if (pts.length < 2) return;
        ctx.beginPath();
        ctx.moveTo(this.X(pts[0][0]), this.Y(pts[0][1]));
        for (const p of pts.slice(1)) ctx.lineTo(this.X(p[0]), this.Y(p[1]));
        ctx.strokeStyle = s.color || '#333';
        ctx.lineWidth = Math.max(1, (s.width || 0.05) * k);
        ctx.stroke();
        break;
      }
      case 'rect':
        ctx.fillStyle = s.color || '#555';
        ctx.fillRect(this.X(s.x), this.Y(s.y + s.h), s.w * k + 0.5, s.h * k + 0.5);
        break;
      case 'circle':
        ctx.beginPath();
        ctx.arc(this.X(s.x), this.Y(s.y), Math.max(1, s.r * k), 0, Math.PI * 2);
        ctx.fillStyle = s.color || '#555';
        ctx.fill();
        break;
      case 'flag': {
        const x = this.X(s.x), y = this.Y(s.y);
        const h = 2.4 * k;
        ctx.strokeStyle = '#444'; ctx.lineWidth = Math.max(2, 0.08 * k);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - h); ctx.stroke();
        const fw = 1.1 * k, fh = 0.7 * k;
        const cells = 4;
        for (let i = 0; i < cells; i++) for (let j = 0; j < 2; j++) {
          ctx.fillStyle = (i + j) % 2 ? '#222' : '#fff';
          ctx.fillRect(x + (i * fw) / cells, y - h + (j * fh) / 2, fw / cells + 0.5, fh / 2 + 0.5);
        }
        if (s.label) { ctx.fillStyle = '#333'; ctx.font = `bold ${Math.max(10, 0.4 * k)}px sans-serif`; ctx.fillText(s.label, x + 4, y - h - 4); }
        break;
      }
      case 'text':
        ctx.fillStyle = s.color || '#333';
        ctx.font = `${Math.max(9, (s.size || 0.5) * k)}px sans-serif`;
        ctx.fillText(s.text, this.X(s.x), this.Y(s.y));
        break;
      default:
    }
  }

  drawMarkers(finish) {
    if (!this.ground) return;
    const ctx = this.ctx;
    const k = this.cam.s;
    const x0 = Math.floor((this.cam.x - this.W / (2 * k)) / 10) * 10;
    const x1 = this.cam.x + this.W / (2 * k);
    ctx.font = `bold ${Math.max(10, Math.min(14, 0.38 * k))}px sans-serif`;
    for (let x = Math.max(0, x0); x <= x1; x += 10) {
      if (finish != null && x > finish + 0.01) break;
      const gy = this.groundYAt(x);
      const X = this.X(x), Y = this.Y(gy);
      ctx.strokeStyle = 'rgba(60,60,60,.55)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X, Y - 0.9 * k); ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,.85)';
      const label = `${x} m`;
      const w = ctx.measureText(label).width + 8;
      ctx.fillRect(X - w / 2, Y - 0.9 * k - 16, w, 15);
      ctx.fillStyle = '#333';
      ctx.fillText(label, X - w / 2 + 4, Y - 0.9 * k - 4);
    }
  }

  drawArrows(arrows) {
    const ctx = this.ctx;
    const k = this.cam.s;
    const vals = arrows.map((a) => a[4]).filter((v) => v != null);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    for (const [x, y, dx, dy, v] of arrows) {
      const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      ctx.strokeStyle = `hsl(${Math.round(t * 120)}, 70%, 42%)`;
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = Math.max(1.5, 0.07 * k);
      const L = 0.32;
      const sx = this.X(x - dx * L * 0.5), sy = this.Y(y - dy * L * 0.5);
      const ex = this.X(x + dx * L * 0.5), ey = this.Y(y + dy * L * 0.5);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
      const ang = Math.atan2(ey - sy, ex - sx);
      const hs = Math.max(4, 0.13 * k);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - hs * Math.cos(ang - 0.5), ey - hs * Math.sin(ang - 0.5));
      ctx.lineTo(ex - hs * Math.cos(ang + 0.5), ey - hs * Math.sin(ang + 0.5));
      ctx.fill();
    }
  }

  drawBody(body, x, y, a) {
    if (x === undefined || x === null) return;
    const ctx = this.ctx;
    const k = this.cam.s;
    ctx.save();
    ctx.translate(this.X(x), this.Y(y));
    ctx.rotate(-a);
    const color = body.color || '#e8463a';
    ctx.lineWidth = Math.max(1, 0.04 * k);
    ctx.strokeStyle = shade(color, -0.25);
    ctx.fillStyle = color;
    if (body.shape === 'circle') {
      const r = Math.max(1, body.r * k);
      ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (body.spokes) {
        ctx.fillStyle = '#9aa0a6';
        ctx.beginPath(); ctx.arc(0, 0, r * 0.55, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#555'; ctx.lineWidth = Math.max(1, r * 0.12);
        for (let i = 0; i < 3; i++) {
          const ang = (i * Math.PI) / 3;
          ctx.beginPath(); ctx.moveTo(-Math.cos(ang) * r * 0.55, -Math.sin(ang) * r * 0.55);
          ctx.lineTo(Math.cos(ang) * r * 0.55, Math.sin(ang) * r * 0.55); ctx.stroke();
        }
        ctx.fillStyle = '#333'; ctx.beginPath(); ctx.arc(0, 0, r * 0.15, 0, Math.PI * 2); ctx.fill();
      }
    } else if (body.shape === 'rect') {
      const w = body.w * k, h = body.h * k;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(-w / 2, -h / 2, w, h, Math.min(w, h) * 0.15); else ctx.rect(-w / 2, -h / 2, w, h);
      ctx.fill(); ctx.stroke();
    } else {
      const pts = body.points || [];
      if (pts.length) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0] * k, -pts[0][1] * k);
        for (const p of pts.slice(1)) ctx.lineTo(p[0] * k, -p[1] * k);
        ctx.closePath();
        ctx.fill(); ctx.stroke();
        if (body.name === 'chassis' || body.window) {
          // a little window so you can tell front from back
          let maxx = -1e9, maxy = -1e9, miny = 1e9;
          for (const p of pts) { maxx = Math.max(maxx, p[0]); maxy = Math.max(maxy, p[1]); miny = Math.min(miny, p[1]); }
          const hgt = (maxy - miny);
          ctx.fillStyle = 'rgba(200,235,255,.9)';
          ctx.beginPath();
          ctx.rect((maxx - hgt * 1.3) * k, -(maxy - hgt * 0.2) * k, hgt * 0.8 * k, hgt * 0.45 * k);
          ctx.fill();
        }
      }
    }
    ctx.restore();
  }

  drawEffects(pose, fx) {
    const effects = this.replay.scene.effects || [];
    const ctx = this.ctx;
    const k = this.cam.s;
    effects.forEach((e, i) => {
      const v = fx[i] || 0;
      if (v < 0.02) return;
      const bi = e.body || 0;
      const bx = pose[bi * 3], by = pose[bi * 3 + 1], ba = pose[bi * 3 + 2];
      if (bx === undefined) return;
      const cos = Math.cos(ba), sin = Math.sin(ba);
      const ax = bx + (e.x || 0) * cos - (e.y || 0) * sin;
      const ay = by + (e.x || 0) * sin + (e.y || 0) * cos;
      const dir = ba + ((e.angle || -90) * Math.PI) / 180;
      const len = (e.size || 1) * v * (0.85 + Math.random() * 0.3);
      const tx = ax + Math.cos(dir) * len, ty = ay + Math.sin(dir) * len;
      const wdt = Math.max(0.12, (e.size || 1) * 0.28);
      const nx = -Math.sin(dir) * wdt * 0.5, ny = Math.cos(dir) * wdt * 0.5;
      const grad = ctx.createLinearGradient(this.X(ax), this.Y(ay), this.X(tx), this.Y(ty));
      grad.addColorStop(0, 'rgba(255,240,120,.95)');
      grad.addColorStop(0.5, 'rgba(255,150,30,.85)');
      grad.addColorStop(1, 'rgba(255,60,0,0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(this.X(ax + nx), this.Y(ay + ny));
      ctx.lineTo(this.X(tx), this.Y(ty));
      ctx.lineTo(this.X(ax - nx), this.Y(ay - ny));
      ctx.closePath();
      ctx.fill();
      void k;
    });
  }

  drawHud(text, right) {
    const ctx = this.ctx;
    ctx.font = 'bold 13px "Helvetica Neue", Arial, sans-serif';
    if (text) {
      const w = ctx.measureText(text).width + 16;
      ctx.fillStyle = 'rgba(255,255,255,.85)';
      ctx.fillRect(8, 8, w, 24);
      ctx.fillStyle = '#2b2d42';
      ctx.fillText(text, 16, 25);
    }
    if (right) {
      const w = ctx.measureText(right).width + 16;
      ctx.fillStyle = 'rgba(255,255,255,.7)';
      ctx.fillRect(this.W - w - 8, 8, w, 24);
      ctx.fillStyle = '#575e75';
      ctx.fillText(right, this.W - w, 25);
    }
  }
}

// ---------------------------------------------------------------------------------
// The Sim tab player
// ---------------------------------------------------------------------------------
const replays = [];
let current = null;
let renderer = null;
let playing = true;
let t = 0;
let lastTs = 0;
let speed = 1;
let visible = false;

function frameCount() { return current ? (current.frames || []).length : 0; }

function select(i, jump = true) {
  current = replays[i] || null;
  if (!current) return;
  renderer.setReplay(current);
  t = 0;
  playing = true;
  $('#sim-play').textContent = '⏸';
  $('#sim-scrub').max = Math.max(0, frameCount() - 1);
  $('#sim-title').textContent = current.title || '';
  const st = current.stats || {};
  $('#sim-stats').textContent = Object.entries(st).filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `${k}: ${typeof v === 'number' ? fmtNum(v) : v}`).join(' · ');
  $('#sim-list').value = String(i);
  $('#sim-empty').style.display = 'none';
  renderer.draw(0, { jump });
}

function rebuildList() {
  const sel = $('#sim-list');
  sel.innerHTML = '';
  replays.forEach((r, i) => sel.append(el('option', { value: String(i) }, r.title || `replay ${i + 1}`)));
}

function loop(ts) {
  requestAnimationFrame(loop);
  const dt = lastTs ? Math.min(0.1, (ts - lastTs) / 1000) : 0;
  lastTs = ts;
  if (!visible || !current) return;
  const n = frameCount();
  const fdt = current.dt || 1 / 30;
  if (playing) {
    t += dt * speed;
    if (t / fdt >= n - 1) {
      t = (n - 1) * fdt;
      playing = false;
      $('#sim-play').textContent = '▶';
    }
  }
  const f = Math.floor(t / fdt);
  $('#sim-scrub').value = String(f);
  renderer.draw(f);
}

export function init() {
  renderer = new SimRenderer($('#sim-canvas'));
  requestAnimationFrame(loop);
  bus.on('tab:shown', (name) => { visible = name === 'sim'; if (visible && current) renderer.draw(Math.floor(t / (current.dt || 1 / 30)), { jump: true }); });
  $('#sim-play').onclick = () => {
    if (!current) return;
    if (!playing && Math.floor(t / (current.dt || 1 / 30)) >= frameCount() - 1) t = 0;
    playing = !playing;
    $('#sim-play').textContent = playing ? '⏸' : '▶';
  };
  $('#sim-restart').onclick = () => { t = 0; playing = true; $('#sim-play').textContent = '⏸'; if (current) renderer.draw(0, { jump: true }); };
  $('#sim-scrub').oninput = (e) => {
    if (!current) return;
    t = parseInt(e.target.value, 10) * (current.dt || 1 / 30);
    playing = false;
    $('#sim-play').textContent = '▶';
    renderer.draw(parseInt(e.target.value, 10), { jump: true });
  };
  $('#sim-speed').onchange = (e) => { speed = parseFloat(e.target.value); };
  $('#sim-follow').onchange = (e) => { renderer.follow = e.target.checked; if (current) renderer.draw(Math.floor(t / (current.dt || 1 / 30)), { jump: true }); };
  $('#sim-list').onchange = (e) => select(parseInt(e.target.value, 10));
  bus.on('run:reset', () => {
    replays.length = 0;
    current = null;
    rebuildList();
    $('#sim-title').textContent = '';
    $('#sim-stats').textContent = '';
    $('#sim-empty').style.display = '';
    const c = $('#sim-canvas');
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
  });
  bus.on('ev:sim_replay', (ev) => {
    if (!ev.scene || !ev.frames) return;
    replays.push(ev);
    if (replays.length > 24) replays.splice(0, replays.length - 24);
    rebuildList();
    const isPlaying = current && playing;
    if ($('#sim-latest').checked && !(isPlaying && visible && t < 1.5)) select(replays.length - 1);
    else if (!current) select(replays.length - 1);
    bus.emit('tab:new', 'sim');
    bus.emit('tab:auto', 'sim');
  });
  window.addEventListener('resize', () => { if (current && visible) renderer.draw(Math.floor(t / (current.dt || 1 / 30)), { jump: true }); });
}
