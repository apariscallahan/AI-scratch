"""Proximal Policy Optimization (PPO) with several copies of the world running side by side.

Clipped surrogate objective, generalised advantage estimation (GAE), observation normalisation,
reward scaling, value bootstrapping when an episode is cut off by its time limit, an entropy bonus
for discrete actions and a linearly decaying learning rate.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import STATE
from ..errors import NBError
from .agent import RunningNorm
from .nets import FastForward
from .session import Seeds

LOG2PI = math.log(2 * math.pi)


def run(sess, model, world, agent, *, steps, lr, gamma, parallel, lam=0.95, clip=0.2):
    hints = world.env_cls.HINTS
    n_envs = max(1, min(64, int(parallel or 8)))
    n_steps = max(8, int(hints.get("ppo_rollout", 2048)) // n_envs)
    epochs = int(hints.get("ppo_epochs", 10))
    minibatch = int(hints.get("ppo_minibatch", 256))
    steps = int(steps)
    if STATE.quick:
        steps = min(steps, 1024)
        n_envs = min(n_envs, 4)
        n_steps = min(n_steps, 64)
        epochs = 2
    batch = n_envs * n_steps
    minibatch = max(16, min(minibatch, batch))
    updates = max(1, math.ceil(steps / batch))
    discrete = agent.action.discrete
    agent.ensure_critic(model)
    actor, critic = agent.actor, agent.critic
    params = list(actor.parameters()) + list(critic.parameters())
    if not discrete:
        params.append(agent.log_std)
    opt = torch.optim.Adam(params, lr=lr, eps=1e-5, foreach=True)
    ent_coef = 0.01 if discrete else 0.0
    fa, fc = FastForward(actor), FastForward(critic)
    norm = agent.norm
    ret_norm = RunningNorm(1)
    ret_acc = np.zeros(n_envs)
    envs = [world.make_env() for _ in range(n_envs)]
    seeds = Seeds(world, salt=agent.steps + 1)
    rng = np.random.default_rng(seeds.next())
    obs = np.stack([e.reset(seeds.next()) for e in envs]).astype(np.float32)
    adim = agent.action.size
    n_out = adim
    sess.plan(updates, "update", f"{n_envs} copies of the world at once, {n_steps} steps each per update, "
                                 f"{updates} updates")

    for it in range(1, updates + 1):
        for g in opt.param_groups:
            g["lr"] = lr * max(0.05, 1.0 - (it - 1) / updates)
        b_obs = np.zeros((n_steps, n_envs, agent.obs_size), np.float32)
        b_act = np.zeros((n_steps, n_envs) if discrete else (n_steps, n_envs, adim), np.int64 if discrete
                         else np.float32)
        b_logp = np.zeros((n_steps, n_envs), np.float32)
        b_val = np.zeros((n_steps, n_envs), np.float32)
        b_rew = np.zeros((n_steps, n_envs), np.float32)
        b_done = np.zeros((n_steps, n_envs), np.float32)
        std = None if discrete else np.exp(agent.log_std.detach().numpy())
        log_std_sum = 0.0 if discrete else float(agent.log_std.detach().sum())
        for t in range(n_steps):
            if norm.enabled:
                norm.update(obs)
            x = norm(obs)
            out = fa(x)
            v = fc(x)[:, 0]
            if discrete:
                z = out - out.max(1, keepdims=True)
                logp_all = z - np.log(np.exp(z).sum(1, keepdims=True))
                cum = np.cumsum(np.exp(logp_all), 1)
                a = (rng.random((n_envs, 1)) < cum).argmax(1)
                logp = logp_all[np.arange(n_envs), a]
                acts = a
            else:
                mean = np.tanh(out)
                noise = rng.standard_normal((n_envs, n_out)).astype(np.float32)
                a = mean + std * noise
                logp = -0.5 * (noise * noise).sum(1) - log_std_sum - 0.5 * n_out * LOG2PI
                acts = a
            b_obs[t] = x
            b_act[t] = a
            b_logp[t] = logp
            b_val[t] = v
            rew = np.zeros(n_envs)
            done = np.zeros(n_envs, bool)
            cut = []
            for i, env in enumerate(envs):
                o, r, term, trunc = env.step(acts[i])
                rew[i] = r
                if term or trunc:
                    sess.episode_done(env.episode_info())
                    if trunc and not term:
                        cut.append((i, o))
                    o = env.reset(seeds.next())
                    done[i] = True
                obs[i] = o
            ret_acc = ret_acc * gamma + rew
            ret_norm.update(ret_acc[:, None])
            scaled = np.clip(rew / math.sqrt(float(ret_norm.var[0]) + 1e-8), -10.0, 10.0)
            ret_acc[done] = 0.0
            if cut:
                idx = [i for i, _ in cut]
                vf = fc(norm(np.stack([o for _, o in cut])))[:, 0]
                scaled[idx] += gamma * vf
            b_rew[t] = scaled
            b_done[t] = done
        # advantages (GAE)
        last_v = fc(norm(obs))[:, 0]
        adv = np.zeros((n_steps, n_envs), np.float32)
        gae = np.zeros(n_envs, np.float32)
        for t in reversed(range(n_steps)):
            next_v = last_v if t == n_steps - 1 else b_val[t + 1]
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + gamma * next_v * nonterm - b_val[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + b_val
        stats = _update(agent, opt, params, b_obs, b_act, b_logp, adv, ret, epochs, minibatch, clip, ent_coef,
                        discrete, rng)
        agent.steps += batch
        model.steps_done += batch
        sess.report(it, agent.steps, it / updates,
                    extra=f"entropy {stats['entropy']:.2f}" if discrete else f"noise {float(np.mean(std)):.2f}")
        if sess.replay_due(it):
            sess.eval_replay(agent, f"Update {it}/{updates}")
        sess.fire(it)
        if sess.skip():
            break
    return {"updates": it, "envs": n_envs}


def _update(agent, opt, params, b_obs, b_act, b_logp, adv, ret, epochs, minibatch, clip, ent_coef, discrete, rng):
    actor, critic = agent.actor, agent.critic
    B = b_obs.shape[0] * b_obs.shape[1]
    obs = torch.from_numpy(b_obs.reshape(B, -1))
    act = torch.from_numpy(b_act.reshape(B) if discrete else b_act.reshape(B, -1))
    old_logp = torch.from_numpy(b_logp.reshape(B))
    advs = torch.from_numpy(adv.reshape(B))
    rets = torch.from_numpy(ret.reshape(B))
    actor.train()
    critic.train()
    ent_val = 0.0
    try:
        for _ in range(epochs):
            perm = torch.from_numpy(rng.permutation(B))
            stop = False
            for s in range(0, B, minibatch):
                idx = perm[s:s + minibatch]
                if len(idx) < 2:
                    continue
                out = actor(obs[idx])
                if discrete:
                    logp_all = F.log_softmax(out, -1)
                    logp = logp_all.gather(1, act[idx].unsqueeze(1)).squeeze(1)
                    ent = -(logp_all.exp() * logp_all).sum(-1).mean()
                else:
                    mean = torch.tanh(out)
                    log_std = agent.log_std
                    z = (act[idx] - mean) / log_std.exp()
                    logp = (-0.5 * z * z - log_std - 0.5 * LOG2PI).sum(-1)
                    ent = (log_std + 0.5 + 0.5 * LOG2PI).sum()
                a = advs[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                v = critic(obs[idx]).squeeze(-1)
                v_loss = F.mse_loss(v, rets[idx])
                loss = pg + 0.5 * v_loss - ent_coef * ent
                if not torch.isfinite(loss):
                    raise NBError("Learning blew up (the numbers became infinite).",
                                  hint="Try a smaller learning rate, or smaller rewards.")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()
                if not discrete:
                    with torch.no_grad():
                        agent.log_std.clamp_(-3.0, 0.5)
                ent_val = float(ent.detach())
                with torch.no_grad():
                    kl = float(((ratio - 1) - log_ratio).mean())
                if kl > 0.05:
                    stop = True
                    break
            if stop:
                break
    finally:
        actor.eval()
        critic.eval()
    return {"entropy": ent_val}
