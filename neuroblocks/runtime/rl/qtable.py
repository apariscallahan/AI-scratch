"""Tabular Q-learning for mazes: a table with one score per (cell, move)."""
from __future__ import annotations

import math

import numpy as np

from ..core import STATE
from ..events import emit
from . import replay as R
from .session import Seeds

CHUNK = 1000


def run(sess, model, world, agent, *, steps, lr, gamma, explore):
    steps = int(steps)
    chunk = CHUNK
    if STATE.quick:
        steps = min(steps, 2000)
        chunk = 500
    iters = max(1, math.ceil(steps / chunk))
    alpha = float(lr) if lr is not None and 0.01 <= float(lr) <= 1.0 else 0.5
    if lr is not None and alpha != float(lr):
        emit("log", level="info", text=f"Q-tables learn with a bigger step than networks: using learning rate "
                                        f"{alpha:g} instead of {float(lr):g}.")
    eps_final = float(explore) if explore is not None else 0.1
    eps_final = min(max(eps_final, 0.0), 1.0)
    Q = agent.qtable
    env = world.make_env()
    n_states = agent.n_states
    seeds = Seeds(world, salt=agent.steps + 5)
    rng = np.random.default_rng(seeds.next())
    obs = env.reset(seeds.next())
    s = int(obs[:n_states].argmax())
    sess.plan(iters, "round", f"{n_states} cells × 4 moves, learning rate {alpha:g}, {iters} rounds of {chunk:,} steps")
    it = 0
    for k in range(1, steps + 1):
        eps = max(eps_final, 1.0 - k / max(1.0, 0.3 * steps))
        if rng.random() < eps:
            a = int(rng.integers(Q.shape[1]))
        else:
            row = Q[s]
            a = int(rng.choice(np.flatnonzero(row == row.max())))
        o, r, term, trunc = env.step(a)
        s2 = int(o[:n_states].argmax())
        target = r if term else r + gamma * Q[s2].max()
        Q[s, a] += alpha * (target - Q[s, a])
        s = s2
        if term or trunc:
            sess.episode_done(env.episode_info())
            o = env.reset(seeds.next())
            s = int(o[:n_states].argmax())
        if k % chunk == 0 or k == steps:
            it += 1
            n = chunk if k % chunk == 0 else k % chunk
            agent.steps += n
            model.steps_done += n
            sess.report(it, agent.steps, k / steps, extra=f"exploring {eps * 100:.0f}%")
            if sess.replay_due(it):
                ev = sess.eval_env()
                tr, info = R.record_episode(ev, lambda ob: agent.act(ob), world.seed, label=sess.label,
                                            max_steps=150 if STATE.quick else None)
                rep = R.build(world, ev.scene(), [tr], title=f"Round {it}/{iters} · {sess.describe_result(info)}",
                              block=sess.bid, dt=R.frame_dt(ev), stats=info, overlay=ev.overlay(qtable=Q))
                R.emit_replay(rep)
                sess.replays += 1
                sess.last_replay = info
            sess.fire(it)
            if sess.skip():
                break
    return {"rounds": it}
