"""The simulation blocks: train … to play in …, watch, show world, average score."""
from __future__ import annotations

import math
import time

import numpy as np

from ..core import STATE, quick_cap
from ..errors import NBError
from ..events import emit
from ..train import fmt_time
from . import replay as R
from .agent import Agent
from .nets import few_threads
from .session import ALGO_NAMES, Session
from .world import World

__all__ = ["train", "watch", "show_world", "average_score"]

_ALGO_ALIASES = {
    "ppo": "ppo", "proximal_policy_optimization": "ppo", "policy_gradient": "ppo",
    "evolution": "evolution", "neuroevolution": "evolution", "neuro_evolution": "evolution", "genetic": "evolution",
    "genetic_algorithm": "evolution", "ga": "evolution", "evolve": "evolution",
    "dqn": "dqn", "deep_q": "dqn", "deep_q_learning": "dqn", "q_network": "dqn",
    "qtable": "qtable", "q_table": "qtable", "q_learning": "qtable", "qlearning": "qtable", "tabular": "qtable",
}
_AGENT_KIND = {"ppo": "policy", "evolution": "policy", "dqn": "q", "qtable": "table"}
_DEFAULT_LR = {"ppo": 3e-4, "dqn": 1e-3, "qtable": 0.5, "evolution": None}


def _check_model(model, bid, what="train"):
    if model is None or not hasattr(model, "specs") or getattr(model, "family", None) != "neural":
        if isinstance(model, World):
            raise NBError("The first slot needs a model (a neural network), and the second one the world.",
                          block_id=bid)
        if getattr(model, "family", None) == "classic":
            raise NBError("Only neural networks can learn in a simulation.",
                          hint="Use a 'create neural network' model (dense layers work well).", block_id=bid)
        raise NBError(f"To {what} in a simulation, the first slot must be a neural network model.", block_id=bid)


def _check_world(world, bid):
    if not isinstance(world, World):
        if hasattr(world, "task") and hasattr(world, "x_train"):
            raise NBError("That's a dataset — simulations need a world.",
                          hint="Use 'create world' from the Simulations blocks, or the normal 'train' block for "
                               "datasets.", block_id=bid)
        raise NBError("The world slot needs a world (from 'create world').", block_id=bid)


def _algo(name, bid):
    key = str(name or "ppo").strip().lower().replace("-", "_").replace(" ", "_")
    algo = _ALGO_ALIASES.get(key)
    if algo is None:
        raise NBError(f"Unknown learning method '{name}'.", hint="Use ppo, evolution, dqn or qtable.", block_id=bid)
    return algo


def _announce(model, agent, world):
    total = agent.num_params()
    what = {"policy": "policy network", "q": "Q-network", "table": "Q-table"}[agent.kind]
    if agent.kind == "table":
        summary = f"{model.label}: {what} with {agent.n_states} cells × {agent.action.size} moves"
    else:
        n_out = agent.action.size if agent.kind == "policy" else len(agent.choices)
        summary = (f"{model.label}: {what}, {len(agent.rows)} layers, {total:,} parameters "
                   f"({world.obs_size} senses → {n_out} outputs)")
    emit("model", name=model.name, block=model.bid,
         info={"summary": summary, "rows": agent.rows, "total_params": total, "input_shape": [world.obs_size],
               "task": "control", "label": model.label, "family": "neural"})


def _set_meta(model, agent, world):
    model.meta = {"task": "control", "world": world.kind, "world_name": world.name, "algorithm": agent.algorithm,
                  "obs_size": agent.obs_size, "obs_names": list(agent.obs_names), "action": agent.action.to_dict(),
                  "input_shape": [agent.obs_size], "modality": "vector"}


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(model, world, algorithm="ppo", steps=None, lr=None, gamma=0.99, parallel=8, replay_every=None, label=None,
          every=None, generations=None, population=None, mutation=None, explore=None, _bid=None, **unused):
    """Let ``model`` learn to act in ``world`` by trial and error (reinforcement learning).

    The model's layer blocks become the policy network (an empty model gets 2 dense layers of 64).
    ``algorithm``: ``"ppo"`` (default), ``"evolution"`` (neuro-evolution), ``"dqn"`` or ``"qtable"`` (mazes).
    ``steps``: environment steps (PPO 100,000 · DQN 50,000 · Q-table 20,000 by default).
    ``generations`` / ``population`` / ``mutation``: neuro-evolution (30 / 50 / 0.1; generations are derived
    from ``steps`` when only steps are given). ``lr``: learning rate (PPO 3e-4, DQN 1e-3, Q-table 0.5).
    ``gamma``: discount. ``parallel``: copies of the world stepped side by side (PPO, DQN).
    ``explore``: final exploration rate (DQN 0.05, Q-table 0.1). ``replay_every``: send a replay every N
    updates/generations (default: about 8 per training). ``every=[(n, "steps", fn)]``: call ``fn`` every n
    updates/generations. Training again continues from what the model already learned (same world type).
    Returns a dict of results (score, rewards, distances, steps, seconds, …).
    """
    for k in unused:
        if not k.startswith("_"):
            emit("log", level="warn", text=f"The '{k}' setting doesn't apply to simulations, so it was ignored.",
                 block=_bid)
    _check_model(model, _bid)
    _check_world(world, _bid)
    algo = _algo(algorithm, _bid)
    if algo == "qtable" and world.kind != "maze":
        raise NBError("The Q-table method only works in the maze world (it needs a small number of places to "
                      "be in).", hint="Use PPO, evolution or DQN for this world.", block_id=_bid)
    gamma = float(gamma if gamma is not None else 0.99)
    if not 0.0 < gamma <= 1.0:
        raise NBError("The discount (gamma) must be between 0 and 1, e.g. 0.99.", block_id=_bid)
    lr = float(lr) if lr is not None else _DEFAULT_LR[algo]
    if lr is not None and not (lr > 0 and math.isfinite(lr)):
        raise NBError("The learning rate must be a positive number.", block_id=_bid)
    agent = model.agent
    if agent is not None:
        reason = agent.mismatch(world)
        if reason is None and agent.kind == _AGENT_KIND[algo]:
            emit("log", level="info", text=f"{model.label} already learned something, so it keeps learning "
                                            f"from there ({agent.steps:,} steps so far).", block=_bid)
        else:
            if reason is None:
                reason = f"it was trained with {ALGO_NAMES.get(agent.algorithm, agent.algorithm)}"
            emit("log", level="warn", text=f"{model.label} starts learning from scratch because {reason}.",
                 block=_bid)
            agent = None
    with few_threads(1):
        fresh = agent is None
        if fresh:
            agent = Agent.new(model, world, algo)
            if algo == "ppo":
                agent.norm.enabled = True
        agent.algorithm = algo
        if fresh:
            _announce(model, agent, world)
        model.agent = agent
        _set_meta(model, agent, world)
        sess = Session(model, world, algo, label=label, every=every, replay_every=replay_every, bid=_bid)
        if steps is None and algo != "evolution":
            steps = {"ppo": 100_000, "dqn": 50_000, "qtable": 20_000}[algo]
        if steps is not None:
            steps = int(steps)
            if steps < 1:
                raise NBError("Train for at least 1 step.", block_id=_bid)
        if algo == "evolution" and generations is not None:
            generations = int(generations)
        what = (f"{generations or 'about 30'} generations of {int(population or 50)}" if algo == "evolution"
                and (generations is not None or steps is None) else f"{steps:,} steps")
        emit("log", level="info", text=f"Training {sess.label} in {world.describe()} with {ALGO_NAMES[algo]} "
                                        f"for {what}…", block=_bid)
        t0 = time.time()
        if algo == "ppo":
            from . import ppo
            res = ppo.run(sess, model, world, agent, steps=steps, lr=lr, gamma=gamma, parallel=parallel)
        elif algo == "dqn":
            from . import dqn
            res = dqn.run(sess, model, world, agent, steps=steps, lr=lr, gamma=gamma, parallel=parallel,
                          explore=explore)
        elif algo == "qtable":
            from . import qtable
            res = qtable.run(sess, model, world, agent, steps=steps, lr=lr, gamma=gamma, explore=explore)
        else:
            from . import evolution
            res = evolution.run(sess, model, world, agent, generations=generations, population=population,
                                mutation=mutation, steps=steps if generations is None else None)
        # a last look at what it learned (the last update/generation already sent one unless Skip was pressed)
        if sess.skipped or sess.replays == 0:
            title = f"Generation {agent.generations} · champion" if algo == "evolution" else "After training"
            final = sess.eval_replay(agent, title)
        elif algo == "evolution":
            final = sess.last_generation.get("best")
        else:
            final = sess.last_replay
    dt = time.time() - t0
    model.train_seconds = getattr(model, "train_seconds", 0.0) + dt
    sess.ms.flush()
    sess.ms.progress(sess.total, sess.total, "done", force=True)
    avg_r, avg_d, best_d = sess.averages(sess.recent)
    stats = {
        "algorithm": algo, "steps": int(agent.steps), "episodes": int(sess.episodes), "seconds": round(dt, 2),
        "reward": None if avg_r is None else round(avg_r, 3),
        "best_reward": None if sess.best_reward == -math.inf else round(sess.best_reward, 3),
        "score": None if not final else round(final["reward"], 3),
    }
    if world.kind == "car":
        stats["distance"] = None if avg_d is None else round(avg_d, 3)
        stats["best_distance"] = None if sess.best_progress == -math.inf else round(sess.best_progress, 3)
        stats["final_distance"] = None if not final else round(final.get("distance", 0.0), 3)
        stats["finished"] = bool(final and final.get("finished"))
    if algo == "evolution":
        stats["generations"] = int(agent.generations)
    stats.update({k: v for k, v in (res or {}).items() if k not in stats})
    summary = f"Finished training {sess.label} in {fmt_time(dt)}"
    if final:
        summary += f" — it now scores {final['reward']:.1f}"
        if world.kind == "car":
            summary += f" and drives {final.get('distance', 0.0):.1f} m"
            if final.get("finished"):
                summary += " (it reaches the finish!)"
        elif world.kind == "flappy":
            summary += f" ({final.get('pipes', 0)} pipes)"
        elif final.get("finished"):
            summary += " (success!)"
    emit("log", level="success", text=summary + ".", block=_bid)
    emit("train_done", model=model.name, label=sess.label, metrics=stats, seconds=round(dt, 2))
    return stats


# ---------------------------------------------------------------------------
# watch / average score / show world
# ---------------------------------------------------------------------------


def _agent_for(model, world, bid, what="watch"):
    _check_model(model, bid, what)
    _check_world(world, bid)
    agent = getattr(model, "agent", None)
    if agent is None:
        emit("log", level="warn", text=f"{model.label} hasn't learned to play in a world yet, so it moves at random "
                                        f"(untrained network).", block=bid)
        with few_threads(1):
            agent = Agent.new(model, world, "evolution")
        return agent
    reason = agent.mismatch(world)
    if reason is not None:
        raise NBError(f"{model.label} can't play in '{world.name}': {reason}.",
                      hint="Train it in this world first (or watch it in the world it learned in).", block_id=bid)
    return agent


def _seed(world, i):
    return world.seed if i == 0 else (world.seed + 1000 * i + 7) & 0x7FFFFFFF


def watch(model, world, episodes=3, _bid=None, **unused):
    """Play some episodes with the model's best actions and show them as replays."""
    agent = _agent_for(model, world, _bid)
    n = max(1, int(episodes or 1))
    n = quick_cap(n, 1)
    env = world.make_env()
    rewards = []
    with few_threads(1):
        for i in range(n):
            tr, info = R.record_episode(env, lambda o: agent.act(o), _seed(world, i), label=model.label,
                                        max_steps=150 if STATE.quick else None)
            rewards.append(info["reward"])
            res = _result_text(world, info)
            overlay = env.overlay(agent) if world.kind == "maze" else None
            rep = R.build(world, env.scene(), [tr], title=f"{model.label} · episode {i + 1}/{n} · {res}", block=_bid,
                          dt=R.frame_dt(env), stats=info, overlay=overlay)
            R.emit_replay(rep)
            emit("log", level="info", text=f"{model.label} in '{world.name}', episode {i + 1}: {res} "
                                            f"(reward {info['reward']:.1f}, {info['reason'] or 'time up'}).",
                 block=_bid)
    return float(np.mean(rewards))


def _result_text(world, info):
    if world.kind == "car":
        return f"{info.get('distance', 0.0):.1f} m" + (" · finished!" if info.get("finished") else "")
    if world.kind == "flappy":
        return f"{info.get('pipes', 0)} pipes"
    return f"reward {info['reward']:.1f}" + (" · success!" if info.get("finished") else "")


def average_score(model, world, episodes=5, _bid=None, **unused):
    """Mean total reward over some episodes, always taking the best action (no exploration)."""
    agent = _agent_for(model, world, _bid, "score")
    n = max(1, int(episodes or 1))
    n = quick_cap(n, 2)
    env = world.make_env()
    total = 0.0
    with few_threads(1):
        for i in range(n):
            obs = env.reset(_seed(world, i))
            k = 0
            while True:
                obs, r, term, trunc = env.step(agent.act(obs))
                k += 1
                if term or trunc or (STATE.quick and k >= 150):
                    break
            total += env.ep_return
    return float(total / n)


def show_world(world, seconds=None, _bid=None, **unused):
    """Preview a world before training: a few seconds of it with nobody at the controls."""
    _check_world(world, _bid)
    env = world.make_env()
    secs = float(seconds) if seconds else 4.0
    if STATE.quick:
        secs = min(secs, 1.0)
    n = max(1, int(round(secs / env.DT)))
    if env.TURN_BASED:  # e.g. the maze: nothing moves until someone presses a key
        env.reset(world.seed)
        tr = R.Track("preview", 1.0)
        for _ in range(max(2, int(secs / R.frame_dt(env)))):
            tr.add(env, first=True)
        info = tr.stats = env.episode_info()
    else:               # nobody at the controls (cart-pole: no push, lander: engines off, …)
        tr, info = R.record_episode(env, lambda o: env.human_action({}), world.seed, label="preview", max_steps=n)
    rep = R.build(world, env.scene(), [tr], title=f"Preview of '{world.name}'", block=_bid, dt=R.frame_dt(env),
                  stats=info)
    R.emit_replay(rep)
    emit("log", level="info", text=world.summary(), block=_bid)
    return rep["id"]
