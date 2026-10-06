"""'play it yourself': drive the world with the keyboard in the editor."""
from __future__ import annotations

import queue
import time

from ..core import STATE, gui_mode
from ..events import emit
from . import replay as R
from .api import _check_world

__all__ = ["play_yourself"]


def _drain():
    while True:
        try:
            STATE.interact_q.get_nowait()
        except queue.Empty:
            return


def _should_stop() -> bool:
    while True:
        try:
            msg = STATE.interact_q.get_nowait()
        except queue.Empty:
            return False
        if msg.get("cmd") == "interact_end":
            return True


def play_yourself(world, _bid=None, **unused):
    """Play the world with the keyboard (Sim tab) until you press Done. Returns your best score."""
    _check_world(world, _bid)
    if not gui_mode():
        emit("log", level="info", text="(Playing a world yourself only works in the NeuroBlocks editor, so this "
                                        "block was skipped.)", block=_bid)
        return 0.0
    env = world.make_env()
    sid = f"game-{time.time_ns()}"
    keys = dict(env.KEYS)
    emit("interactive", id=sid, kind="game", model="", label=world.name, block=_bid,
         options={"keys": keys, "world": world.kind, "title": world.title})
    emit("log", level="info", block=_bid,
         text=f"Play '{world.name}' yourself in the Play tab (click the picture first) — " + ", ".join(
             f"{'Space' if k == ' ' else k}: {v}" for k, v in keys.items()) + ". Press Done to continue.")
    _drain()
    scores = []
    episode = 0
    dt_frame = R.frame_dt(env)

    def start(ep):
        seed = world.seed if (world.same_world or ep == 0) else (world.seed + 7919 * ep) & 0x7FFFFFFF
        env.reset(seed)
        emit("sim_live", id=sid, reset=True, scene=env.scene(), kind=world.kind, world=world.name, dt=dt_frame,
             agents=[{"label": "you", "alpha": 1.0}], frame=_r(env.frame()), fx=_fx(env), hud=env.hud())

    start(episode)
    next_t = time.perf_counter()
    last_move = 0.0
    pause_until = None
    waiting = True  # nothing moves until the player presses a key (so an idle try isn't "stuck")
    while not _should_stop():
        now = time.perf_counter()
        if pause_until is not None:
            if now < pause_until:
                time.sleep(0.03)
                continue
            pause_until = None
            episode += 1
            start(episode)
            waiting = True
            next_t = time.perf_counter()
        pressed = dict(STATE.keys or {})
        if waiting:
            if not any(pressed.values()):
                time.sleep(0.03)
                continue
            waiting = False
            next_t = time.perf_counter()
        action = env.human_action(pressed)
        if env.TURN_BASED:
            if action is None or now - last_move < 0.18:
                time.sleep(0.02)
                continue
            last_move = now
        obs, r, term, trunc = env.step(action)
        frames = env.frames()
        for k, fr in enumerate(frames):
            if env.steps % env.RECORD_EVERY == 0 or term or trunc:
                emit("sim_live", id=sid, frame=_r(fr), fx=_fx(env), hud=env.hud())
            if len(frames) > 1 and k < len(frames) - 1:
                time.sleep(env.DT / len(frames) * 0.6)
        if term or trunc:
            info = env.episode_info()
            scores.append(info["reward"])
            what = f"{info['distance']:.1f} m, " if world.kind == "car" else ""
            emit("log", level="info", block=_bid,
                 text=f"Your try {episode + 1}: {what}reward {info['reward']:.1f} ({info['reason'] or 'time up'}).")
            pause_until = time.perf_counter() + 1.2
            continue
        if not env.TURN_BASED:
            next_t += env.DT
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                next_t = time.perf_counter()
    emit("interactive_end", id=sid)
    return float(max(scores)) if scores else 0.0


def _r(frame):
    return [round(float(v), 3) for v in frame]


def _fx(env):
    f = env.fx()
    return None if f is None else [round(float(v), 3) for v in f]
