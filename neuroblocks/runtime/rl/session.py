"""Bookkeeping shared by every learning algorithm: charts, progress, logs, replays, callbacks, Skip."""
from __future__ import annotations

import math
import time
from collections import deque

from ..core import STATE
from ..events import emit
from ..train import MetricStream, _Callbacks, _check_skip, fmt_time
from . import replay as R

ALGO_NAMES = {"ppo": "PPO", "evolution": "neuro-evolution", "dqn": "DQN", "qtable": "Q-table learning"}


class Seeds:
    """A reproducible stream of episode seeds for one training run."""

    def __init__(self, world, salt: int = 0):
        self.base = (int(world.seed) * 7919 + salt * 104729 + 17) & 0x7FFFFFFF
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return (self.base + self.n * 2654435761) & 0x7FFFFFFF


class Session:
    def __init__(self, model, world, algorithm: str, label=None, every=None, replay_every=None, bid=None):
        self.model = model
        self.world = world
        self.algorithm = algorithm
        self.label = label or model.label
        self.ms = MetricStream(self.label)
        self.cb = _Callbacks(every)
        self.bid = bid
        self.replay_every_arg = replay_every
        self.replay_every = 1
        self.total = 1
        self.unit = "update"
        self.t0 = time.time()
        self.episodes = 0
        self.recent = deque(maxlen=20)
        self.window: list = []          # episodes finished since the last report
        self.best_reward = -math.inf
        self.best_progress = -math.inf
        self.is_car = world.kind == "car"
        self.unit_suffix = world.env_cls.PROGRESS_UNIT
        self._eval_env = None
        self.last_replay = None
        self.replays = 0
        self.skipped = False
        self.last_generation: dict = {}

    # -- schedule ---------------------------------------------------------------
    def plan(self, total: int, unit: str, info: str = ""):
        self.total = max(1, int(total))
        self.unit = unit
        if info:
            emit("log", level="debug", text=f"{self.label}: {info}")
        if self.replay_every_arg:
            self.replay_every = max(1, int(self.replay_every_arg))
        else:
            self.replay_every = max(1, int(round(self.total * 0.125)))
        if STATE.quick:
            self.replay_every = max(self.replay_every, self.total)

    def replay_due(self, it: int) -> bool:
        return it % self.replay_every == 0 or it >= self.total

    # -- episodes -----------------------------------------------------------------
    def episode_done(self, info: dict):
        self.episodes += 1
        self.recent.append(info)
        self.window.append(info)
        self.best_reward = max(self.best_reward, info["reward"])
        self.best_progress = max(self.best_progress, info.get("distance", 0.0))

    def averages(self, infos=None):
        infos = list(infos if infos is not None else (self.window or self.recent))
        if not infos:
            return None, None, None
        avg_r = sum(i["reward"] for i in infos) / len(infos)
        avg_d = sum(i.get("distance", 0.0) for i in infos) / len(infos)
        best_d = max(i.get("distance", 0.0) for i in infos)
        return avg_r, avg_d, best_d

    # -- reporting ------------------------------------------------------------------
    def eta(self, done_frac: float) -> str:
        el = time.time() - self.t0
        if done_frac <= 0:
            return "?"
        return fmt_time(el / done_frac * (1 - done_frac))

    def report(self, it: int, x, done_frac: float, extra: str = "", best_reward=None, infos=None):
        """Charts + progress + a debug line after one update/generation."""
        avg_r, avg_d, best_d = self.averages(infos)
        self.window = []
        model = self.model
        bits = [f"{self.unit} {it}/{self.total}"]
        if avg_r is not None:
            self.ms.add("reward", "average", x, avg_r, force=True)
            model.record("episode_reward", x, avg_r)
            bits.append(f"avg reward {avg_r:.1f}")
            if best_reward is not None:
                self.ms.add("reward", "best", x, best_reward, force=True)
                model.record("best_reward", x, best_reward)
            if self.is_car:
                self.ms.add("distance", "average", x, avg_d, force=True)
                self.ms.add("distance", "best", x, best_d, force=True)
                model.record("distance", x, avg_d)
                model.record("best_distance", x, best_d)
                bits.append(f"best {best_d:.1f} m")
        if extra:
            bits.append(extra)
        bits.append(f"ETA {self.eta(done_frac)}")
        info = " · ".join(bits)
        self.ms.progress(it, self.total, info, force=True)
        emit("log", level="debug", text=f"{self.label}: {info}")
        return avg_r, avg_d, best_d

    def fire(self, it: int):
        self.cb.fire("steps", it)
        self.cb.fire("epochs", it)

    def skip(self) -> bool:
        if _check_skip():
            self.skipped = True
            return True
        return False

    # -- replays ----------------------------------------------------------------------
    def eval_env(self):
        if self._eval_env is None:
            self._eval_env = self.world.make_env()
        return self._eval_env

    def describe_result(self, info: dict) -> str:
        if self.is_car:
            s = f"{info.get('distance', 0.0):.1f} m"
            return s + (" · finished!" if info.get("finished") else "")
        if self.world.kind == "flappy":
            return f"{info.get('pipes', 0)} pipes"
        s = f"reward {info.get('reward', 0.0):.1f}"
        return s + (" · success!" if info.get("finished") else "")

    def eval_replay(self, agent, title: str, max_steps=None):
        """Play one episode with the agent's best actions on the world's standard layout and send the replay."""
        env = self.eval_env()
        if STATE.quick and max_steps is None:
            max_steps = 150
        tr, info = R.record_episode(env, lambda o: agent.act(o), self.world.seed, label=self.label,
                                    max_steps=max_steps)
        overlay = env.overlay(agent) if self.world.kind == "maze" else None
        rep = R.build(self.world, env.scene(), [tr], title=f"{title} · {self.describe_result(info)}", block=self.bid,
                      dt=R.frame_dt(env), stats=info, overlay=overlay)
        R.emit_replay(rep)
        self.replays += 1
        self.last_replay = info
        # Each replay is a test drive with the current best actions: chart it too.
        x = getattr(agent, "generations", 0) if self.algorithm == "evolution" else getattr(agent, "steps", 0)
        self.ms.add("reward", "test run", x, float(info.get("reward", 0.0)), force=True)
        if self.is_car:
            self.ms.add("distance", "test run", x, float(info.get("distance", 0.0)), force=True)
        return info

    def elapsed(self) -> float:
        return time.time() - self.t0
