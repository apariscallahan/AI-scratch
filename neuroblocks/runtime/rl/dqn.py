"""Deep Q-learning (DQN) with experience replay, a target network and double-DQN targets.

Continuous worlds are turned into a menu of choices (each control set to e.g. −1, 0 or 1).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import STATE
from ..errors import NBError
from ..events import emit
from .nets import FastForward
from .session import Seeds

CHUNK = 1000  # environment steps per reported iteration


def run(sess, model, world, agent, *, steps, lr, gamma, parallel, explore):
    steps = int(steps)
    n_envs = max(1, min(4, int(parallel or 4)))
    chunk = CHUNK
    if STATE.quick:
        steps = min(steps, 600)
        n_envs = min(n_envs, 2)
        chunk = 200
    iters = max(1, math.ceil(steps / chunk))
    q = agent.qnet
    q_target = copy.deepcopy(q)
    q_target.eval()
    opt = torch.optim.Adam(q.parameters(), lr=lr, foreach=True)
    fq = FastForward(q)
    choices = agent.choices
    n_actions = len(choices)
    eps_final = float(explore) if explore is not None else 0.05
    eps_final = min(max(eps_final, 0.0), 1.0)
    explore_steps = max(1, int(0.3 * steps))
    learn_start = max(100 if STATE.quick else 500, min(2000, steps // 20))
    batch = 64
    cap = max(1000, min(100_000, steps))
    obs_dim = agent.obs_size
    B_obs = np.zeros((cap, obs_dim), np.float32)
    B_next = np.zeros((cap, obs_dim), np.float32)
    B_act = np.zeros(cap, np.int64)
    B_rew = np.zeros(cap, np.float32)
    B_term = np.zeros(cap, np.float32)
    size = pos = 0
    envs = [world.make_env() for _ in range(n_envs)]
    seeds = Seeds(world, salt=agent.steps + 3)
    rng = np.random.default_rng(seeds.next())
    obs = np.stack([e.reset(seeds.next()) for e in envs]).astype(np.float32)
    done_steps = 0
    grad_steps = 0
    discrete = agent.action.discrete
    sess.plan(iters, "round", f"{n_actions} possible actions, {n_envs} copies of the world, {iters} rounds of "
                              f"{chunk:,} steps")
    it = 0
    reported = 0
    loss_v = float("nan")
    best = (-math.inf, None, 0)   # (greedy score, Q-network weights, round)
    while done_steps < steps:
        eps = max(eps_final, 1.0 - (1.0 - eps_final) * done_steps / explore_steps)
        greedy = fq(obs).argmax(1)
        rand = rng.random(n_envs) < eps
        a_idx = np.where(rand, rng.integers(0, n_actions, n_envs), greedy)
        for i, env in enumerate(envs):
            act = int(a_idx[i]) if discrete else choices[int(a_idx[i])]
            o, r, term, trunc = env.step(act)
            B_obs[pos] = obs[i]
            B_next[pos] = o
            B_act[pos] = a_idx[i]
            B_rew[pos] = r
            B_term[pos] = 1.0 if term else 0.0
            pos = (pos + 1) % cap
            size = min(size + 1, cap)
            if term or trunc:
                sess.episode_done(env.episode_info())
                o = env.reset(seeds.next())
            obs[i] = o
        prev = done_steps
        done_steps += n_envs
        if done_steps >= learn_start and size >= batch:
            for _ in range(max(1, n_envs // 2)):
                idx = rng.integers(0, size, batch)
                loss_v = _learn(q, q_target, opt, B_obs[idx], B_act[idx], B_rew[idx], B_next[idx], B_term[idx],
                                gamma)
                grad_steps += 1
                if grad_steps % 250 == 0:
                    q_target.load_state_dict(q.state_dict())
        if done_steps // chunk > prev // chunk or done_steps >= steps:
            it += 1
            agent.steps += done_steps - reported
            model.steps_done += done_steps - reported
            reported = done_steps
            sess.report(it, agent.steps, min(1.0, done_steps / steps), extra=f"exploring {eps * 100:.0f}%")
            if grad_steps:
                # DQN can get worse again later on: remember the best version (one greedy test episode)
                score = _greedy_return(agent, sess.eval_env(), world.seed, 150 if STATE.quick else None)
                if score >= best[0]:
                    best = (score, copy.deepcopy(q.state_dict()), it)
            if sess.replay_due(it):
                sess.eval_replay(agent, f"Round {it}/{iters}")
            sess.fire(it)
            if sess.skip():
                break
    if best[1] is not None and best[2] != it:
        now = _greedy_return(agent, sess.eval_env(), world.seed, 150 if STATE.quick else None)
        if best[0] > now:
            q.load_state_dict(best[1])
            emit("log", level="info", text=f"{sess.label}: kept the best version of the Q-network (from round "
                                            f"{best[2]}, score {best[0]:.1f} instead of {now:.1f}).")
            sess.eval_replay(agent, f"Best version (round {best[2]})")
    return {"rounds": it, "gradient_steps": grad_steps, "last_loss": loss_v}


def _greedy_return(agent, env, seed, max_steps=None):
    obs = env.reset(seed)
    n = 0
    while True:
        obs, r, term, trunc = env.step(agent.act(obs))
        n += 1
        if term or trunc or (max_steps is not None and n >= max_steps):
            return env.ep_return


def _learn(q, q_target, opt, obs, act, rew, nxt, term, gamma):
    o = torch.from_numpy(obs)
    n = torch.from_numpy(nxt)
    a = torch.from_numpy(act)
    r = torch.from_numpy(rew)
    t = torch.from_numpy(term)
    q.train()
    with torch.no_grad():
        best = q(n).argmax(1, keepdim=True)
        target = r + gamma * (1.0 - t) * q_target(n).gather(1, best).squeeze(1)
    pred = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(pred, target)
    if not torch.isfinite(loss):
        raise NBError("Learning blew up (the numbers became infinite).",
                      hint="Try a smaller learning rate, or smaller rewards.")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    opt.step()
    q.eval()
    return float(loss.detach())
