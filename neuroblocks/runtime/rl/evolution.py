"""Neuro-evolution (a genetic algorithm on network weights).

Each generation, every member of the population (a copy of the policy network with its own
weights) plays one episode in the same world (same seed). The best members survive unchanged
(elitism); the rest of the next generation are mutated copies (gaussian noise) of the top quarter.
All members act in lock-step and their networks run as one batched computation.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..core import STATE
from ..events import emit
from . import replay as R
from .nets import Population, reinit

MAX_GHOSTS = 20


def run(sess, model, world, agent, *, generations, population, mutation, steps=None):
    P = int(population or 50)
    P = max(4, min(P, 1000))
    sigma = float(mutation if mutation is not None else 0.1)
    max_steps = world.max_steps
    if generations is None:
        if steps:
            generations = int(round(int(steps) / (P * max_steps * 0.5)))
            generations = max(3, min(generations, 500))
            emit("log", level="info", text=f"{int(steps):,} steps ≈ {generations} generations of {P} "
                                            f"(each episode is up to {max_steps:,} steps).")
        else:
            generations = 30
    G = max(1, int(generations))
    if STATE.quick:
        G = min(G, 2)
        P = min(P, 8)
        max_steps = min(max_steps, 150)
    net = agent.actor
    pop = Population(net)
    D = pop.dim
    gen = torch.Generator().manual_seed(int(world.seed) * 31 + agent.generations + 7)
    rng = np.random.default_rng(int(world.seed) * 17 + agent.generations + 3)
    # initial population: the current network (if it already learned something) plus fresh random ones
    theta = torch.empty(P, D)
    trained = agent.generations > 0 or agent.steps > 0
    start = pop.flat()
    with torch.no_grad():
        for i in range(P):
            if trained:
                theta[i] = start if i == 0 else start + sigma * torch.randn(D, generator=gen)
            else:
                reinit(net)
                theta[i] = pop.flat()
    pop.load(theta[0])
    n_elite = max(1, P // 10)
    n_parents = max(2, P // 4)
    envs = [world.make_env() for _ in range(P)]
    discrete = agent.action.discrete
    norm = agent.norm
    sess.plan(G, "generation", f"population {P}, {G} generations, mutation {sigma:g}")
    best_ever = -math.inf
    last = {}
    for g in range(1, G + 1):
        # every member of a generation plays the same episode; the next generation gets a new start
        # (with 'same world' the layout stays the same — only starting positions/noise change)
        seed = world.seed if g == 1 and agent.generations == 0 else \
            (world.seed * 1009 + agent.generations + 1) & 0x7FFFFFFF
        record = sess.replay_due(g)
        returns, infos, tracks, used = _evaluate(pop, theta, envs, seed, norm, discrete, max_steps, record)
        order = np.argsort(-returns, kind="stable")
        best = int(order[0])
        agent.generations += 1
        agent.steps += used
        model.steps_done += used
        for inf in infos:
            sess.episode_done(inf)
        best_ever = max(best_ever, float(returns[best]))
        pop.load(theta[best])                       # the agent always holds the current champion
        agent.best = float(returns[best])
        sess.report(g, agent.generations, g / G, best_reward=float(returns[best]), infos=infos,
                    extra=f"best reward {returns[best]:.1f}")
        last = {"best": infos[best], "order": order, "returns": returns}
        if record:
            # as many ghosts as fit in a replay at full frame rate (worlds with many bodies get fewer)
            longest = max(len(t) for t in tracks)
            per_ghost = longest * (len(tracks[best].frames[0]) + (len(tracks[best].fx[0]) if tracks[best].fx else 0))
            n_ghosts = max(1, min(MAX_GHOSTS, R.MAX_NUMBERS // max(1, per_ghost)))
            keep = [int(i) for i in order[:n_ghosts]]
            chosen = []
            for rank, i in enumerate(keep):
                tr = tracks[i]
                tr.label = "best" if rank == 0 else f"#{rank + 1}"
                tr.alpha = 1.0 if rank == 0 else 0.25
                chosen.append(tr)
            env0 = envs[best]
            title = f"Generation {agent.generations} · best {sess.describe_result(infos[best])}"
            overlay = env0.overlay(agent) if world.kind == "maze" else None
            rep = R.build(world, env0.scene(), chosen, title=title, block=sess.bid, dt=R.frame_dt(env0),
                          stats={**infos[best], "generation": agent.generations, "population": P},
                          overlay=overlay)
            R.emit_replay(rep)
            sess.replays += 1
        sess.fire(g)
        if g >= G or sess.skip():
            break
        # next generation
        with torch.no_grad():
            elite = theta[torch.from_numpy(order[:n_elite].copy())]
            parents = theta[torch.from_numpy(order[:n_parents].copy())]
            pick = torch.from_numpy(rng.integers(0, n_parents, P - n_elite))
            children = parents[pick] + sigma * torch.randn(P - n_elite, D, generator=gen)
            theta = torch.cat([elite, children])
    sess.last_generation = last
    return {"generations": agent.generations, "population": P, "best_ever": best_ever}


def _evaluate(pop, theta, envs, seed, norm, discrete, max_steps, record):
    P = theta.shape[0]
    obs = np.stack([e.reset(seed) for e in envs]).astype(np.float32)
    alive = np.ones(P, bool)
    returns = np.zeros(P)
    infos = [None] * P
    tracks = None
    if record:
        tracks = []
        for e in envs:
            tr = R.Track()
            tr.add(e, hud=True, first=True)
            tracks.append(tr)
    used = 0
    n = 0
    stride = envs[0].RECORD_EVERY
    while alive.any():
        n += 1
        idx = np.flatnonzero(alive)
        if len(idx) <= P // 2:   # only a few still playing: just run their networks
            out = pop.forward(theta[torch.from_numpy(idx)], norm(obs[idx]))
        else:
            out = pop.forward(theta, norm(obs))[idx]
        acts = out.argmax(1) if discrete else np.tanh(out)
        for j, i in enumerate(idx):
            env = envs[i]
            o, r, term, trunc = env.step(acts[j])
            used += 1
            obs[i] = o
            done = term or trunc or n >= max_steps
            if record and (n % stride == 0 or done):
                tracks[i].add(env)
            if done:
                alive[i] = False
                infos[i] = env.episode_info()
                returns[i] = env.ep_return
    return returns, infos, tracks, used
