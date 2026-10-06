"""Replays: recorded episodes that the editor's Sim tab animates (``sim_replay`` events).

Schema (a contract with the browser renderer)::

    {"id", "world", "kind", "title", "block", "dt",
     "scene": {"bounds", "camera", "sky", "ground_fill", "static": [...], "bodies": [...], "effects": [...]},
     "agents": [{"label", "alpha"}, ...],
     "frames": [[[x, y, angle, ...per body] ...per agent] ...per frame],
     "fx": [[[value, ...per effect] ...per agent] ...per frame],     # optional
     "hud": ["text", ...per frame],                                   # optional
     "overlay": {"arrows": [[x, y, dx, dy, value], ...]},            # optional
     "stats": {"reward", "distance", "steps", "finished"}}
"""
from __future__ import annotations

import itertools
import json
import re
import time

import numpy as np

from ..events import emit, emitter

MAX_NUMBERS = 170_000      # ≈ 1.2 MB of JSON
_COUNTER = itertools.count(1)


class Track:
    """The recorded frames of one agent."""

    __slots__ = ("label", "alpha", "frames", "fx", "hud", "stats")

    def __init__(self, label="agent", alpha=1.0):
        self.label = label
        self.alpha = alpha
        self.frames: list = []
        self.fx: list = []
        self.hud: list = []
        self.stats: dict = {}

    def add(self, env, hud=True, first=False):
        frames = [env.frame()] if first else env.frames()
        f = env.fx()
        h = env.hud() if hud else None
        for fr in frames:
            self.frames.append(fr)
            if f is not None:
                self.fx.append(f)
            if h is not None:
                self.hud.append(h)

    def __len__(self):
        return len(self.frames)


def frame_dt(env, stride=None) -> float:
    """Seconds between recorded frames."""
    return env.DT * (stride or env.RECORD_EVERY) / env.FRAME_SUBSTEPS


def record_episode(env, act, seed, *, label="agent", max_steps=None, stride=None):
    """Run one episode with ``act(obs) -> action`` and record it. Returns (track, info)."""
    stride = stride or env.RECORD_EVERY
    obs = env.reset(seed)
    tr = Track(label, 1.0)
    tr.add(env, first=True)
    n = 0
    while True:
        obs, r, term, trunc = env.step(act(obs))
        n += 1
        done = term or trunc or (max_steps is not None and n >= max_steps)
        if n % stride == 0 or done:
            tr.add(env)
        if done:
            break
    info = env.episode_info()
    tr.stats = info
    return tr, info


def build(world, scene, tracks, *, title, block=None, dt, stats=None, overlay=None) -> dict:
    """Assemble a replay dict from tracks (padding short ones, thinning long ones to stay small)."""
    tracks = [t for t in tracks if len(t)]
    if not tracks:
        raise ValueError("nothing recorded")
    n_frames = max(len(t) for t in tracks)
    per_frame = sum(len(t.frames[0]) + (len(t.fx[0]) if t.fx else 0) for t in tracks)
    stride = 1
    while n_frames / stride * per_frame > MAX_NUMBERS and stride < 64:
        stride += 1
    idx = list(range(0, n_frames, stride))
    if idx[-1] != n_frames - 1:
        idx.append(n_frames - 1)
    frames_arr = []
    fx_arr = []
    has_fx = all(t.fx for t in tracks) and bool(tracks[0].fx)
    for t in tracks:
        f = np.asarray(t.frames, dtype=np.float64)
        if len(f) < n_frames:
            f = np.concatenate([f, np.repeat(f[-1:], n_frames - len(f), axis=0)])
        frames_arr.append(f[idx])
        if has_fx:
            g = np.asarray(t.fx, dtype=np.float64)
            if len(g) < n_frames:
                g = np.concatenate([g, np.zeros((n_frames - len(g), g.shape[1]))])
            fx_arr.append(g[idx])
    frames = np.round(np.stack(frames_arr, axis=1), 3).tolist()   # (T, A, 3B)
    replay = {
        "id": f"{world.name}-{next(_COUNTER)}-{time.time_ns() % 10**9}",
        "world": world.name,
        "kind": world.kind,
        "title": title,
        "block": block,
        "dt": round(dt * stride, 5),
        "scene": scene,
        "agents": [{"label": t.label, "alpha": t.alpha} for t in tracks],
        "frames": frames,
    }
    if has_fx:
        replay["fx"] = np.round(np.clip(np.stack(fx_arr, axis=1), 0.0, 1.0), 3).tolist()
    hud = tracks[0].hud
    if hud:
        hud = hud + [hud[-1]] * (n_frames - len(hud))
        replay["hud"] = [hud[i] for i in idx]
    if overlay:
        replay["overlay"] = overlay
    st = dict(stats if stats is not None else tracks[0].stats)
    replay["stats"] = {
        "reward": round(float(st.get("reward", 0.0)), 3),
        "distance": round(float(st.get("distance", 0.0)), 3),
        "steps": int(st.get("steps", 0)),
        "finished": bool(st.get("finished", False)),
        **{k: v for k, v in st.items() if k not in ("reward", "distance", "steps", "finished", "reason")},
    }
    return replay


def emit_replay(replay: dict):
    """Send a replay to the editor (and save it next to the run's events on the command line)."""
    em = emitter()
    ev = dict(replay)
    if em.mode == "cli" and em.run_dir is not None:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{replay.get('world', 'world')}-{replay.get('title', 'replay')}")
        slug = slug.strip("-").lower()[:70] or "replay"
        d = em.run_dir / "replays"
        try:
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{slug}-{replay['id'][-6:]}.json"
            path.write_text(json.dumps(replay, separators=(",", ":")), encoding="utf-8")
            ev["saved"] = str(path)
        except OSError:
            pass
    emit("sim_replay", **ev)
    return ev
