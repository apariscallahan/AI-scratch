"""Agents: what a model learned in a world (``model.agent``).

Three kinds:

* ``policy`` (PPO, neuro-evolution): an actor network. Continuous worlds: action = tanh(output);
  discrete worlds: the output with the highest score (or a sample from the softmax while learning).
* ``q`` (DQN): a Q-network scoring a menu of actions; the best one is taken.
* ``table`` (tabular Q-learning, mazes): a table of scores per cell and move.

Observations are normalised with running statistics learned during training.
"""
from __future__ import annotations

import numpy as np
import torch

from ..errors import NBError
from .env_base import ActionSpec
from .nets import FastForward, build_net

FORMAT = "neuroblocks-agent"


class RunningNorm:
    """Running mean/variance of observations (or returns) — normalises inputs to about ±1."""

    def __init__(self, size, enabled=True):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4
        self.enabled = enabled
        self._scale = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None]
        n = x.shape[0]
        if n == 0:
            return
        bm, bv = x.mean(0), x.var(0)
        tot = self.count + n
        delta = bm - self.mean
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.count + bv * n + delta * delta * self.count * n / tot) / tot
        self.count = tot
        self._scale = None

    def __call__(self, x):
        if not self.enabled:
            return np.asarray(x, dtype=np.float32)
        if self._scale is None:
            self._scale = (1.0 / np.sqrt(self.var + 1e-8)).astype(np.float32)
            self._mean32 = self.mean.astype(np.float32)
        return np.clip((np.asarray(x, dtype=np.float32) - self._mean32) * self._scale, -10.0, 10.0)

    def state(self) -> dict:
        return {"mean": torch.tensor(self.mean), "var": torch.tensor(self.var), "count": float(self.count),
                "enabled": bool(self.enabled)}

    @classmethod
    def from_state(cls, d, size):
        n = cls(size, bool(d.get("enabled", True)))
        n.mean = np.asarray(d["mean"], dtype=np.float64).reshape(size)
        n.var = np.asarray(d["var"], dtype=np.float64).reshape(size)
        n.count = float(d.get("count", 1.0))
        return n


class Agent:
    """A policy that picks actions from observations."""

    def __init__(self, kind: str, obs_size: int, action: ActionSpec, *, world: str = "", obs_names=None,
                 algorithm: str = "", actor=None, critic=None, log_std=None, qnet=None, qtable=None,
                 norm: RunningNorm | None = None, n_states=None):
        self.kind = kind
        self.obs_size = int(obs_size)
        self.action = action
        self.world = world
        self.obs_names = list(obs_names or [])
        self.algorithm = algorithm
        self.actor = actor
        self.critic = critic
        self.log_std = log_std
        self.qnet = qnet
        self.qtable = qtable
        self.norm = norm if norm is not None else RunningNorm(self.obs_size, enabled=False)
        self.n_states = n_states
        self.choices = action.choices() if kind == "q" else None
        self.steps = 0
        self.episodes = 0
        self.generations = 0
        self.best = None
        self.rows: list = []
        self._fast = None

    # -- building -----------------------------------------------------------------
    @classmethod
    def new(cls, model, world, algorithm: str):
        """A fresh, untrained agent for ``world`` whose networks follow the model's layer blocks."""
        spec = world.action_spec
        common = dict(world=world.kind, obs_names=world.obs_names, algorithm=algorithm)
        if algorithm == "qtable":
            env = world.make_env()
            n_states = int(getattr(env, "n_cells", 0))
            return cls("table", world.obs_size, spec, qtable=np.zeros((n_states, spec.size)), n_states=n_states,
                       **common)
        if algorithm == "dqn":
            n = len(spec.choices())
            if n > 512:
                raise NBError(f"DQN would have to choose between {n} action combinations — too many.",
                              hint="Give the car fewer abilities, or use PPO or evolution.")
            qnet, rows = build_net(model, world.obs_size, n)
            ag = cls("q", world.obs_size, spec, qnet=qnet, **common)
            ag.rows = rows
            return ag
        out = spec.size
        actor, rows = build_net(model, world.obs_size, out, out_gain=0.01 if algorithm == "ppo" else None)
        ag = cls("policy", world.obs_size, spec, actor=actor, norm=RunningNorm(world.obs_size, enabled=False),
                 **common)
        ag.rows = rows
        return ag

    def ensure_critic(self, model):
        if self.critic is None:
            self.critic, _ = build_net(model, self.obs_size, 1, out_gain=1.0)
        if self.log_std is None and not self.action.discrete:
            self.log_std = torch.nn.Parameter(torch.full((self.action.size,), -0.5))

    def num_params(self) -> int:
        net = self.actor if self.actor is not None else self.qnet
        if net is not None:
            return sum(p.numel() for p in net.parameters())
        return int(self.qtable.size) if self.qtable is not None else 0

    def describe(self) -> str:
        what = {"policy": "policy network", "q": "Q-network", "table": "Q-table"}[self.kind]
        return f"{what} for the {self.world} world ({self.algorithm})"

    # -- acting -----------------------------------------------------------------------
    def _forward(self, net, x):
        if self._fast is None or self._fast.net is not net:
            self._fast = FastForward(net)
        return self._fast(x)

    def act_batch(self, obs, deterministic=True, rng=None):
        """Actions for a batch of observations (list of action vectors, or array of choices)."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None]
        if self.kind == "table":
            states = obs[:, : self.n_states].argmax(1)
            q = self.qtable[states]
            return q.argmax(1)
        x = self.norm(obs)
        if self.kind == "q":
            q = self._forward(self.qnet, x)
            idx = q.argmax(1)
            if self.action.discrete:
                return idx
            return [self.choices[i] for i in idx]
        out = self._forward(self.actor, x)
        if self.action.discrete:
            if deterministic:
                return out.argmax(1)
            rng = rng or np.random.default_rng()
            z = out - out.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            return np.array([rng.choice(len(pi), p=pi) for pi in p])
        mean = np.tanh(out)
        if deterministic or self.log_std is None:
            return mean
        rng = rng or np.random.default_rng()
        std = np.exp(self.log_std.detach().numpy())
        return mean + std * rng.standard_normal(mean.shape)

    def act(self, obs, deterministic=True):
        a = self.act_batch(np.asarray(obs, dtype=np.float32)[None], deterministic)
        return a[0]

    def greedy_with_value(self, obs):
        """(best choice, its score) for each observation — used for the maze's arrows."""
        obs = np.asarray(obs, dtype=np.float32)
        if self.kind == "table":
            q = self.qtable[obs[:, : self.n_states].argmax(1)]
            return q.argmax(1), q.max(1)
        x = self.norm(obs)
        out = self._forward(self.qnet if self.kind == "q" else self.actor, x)
        if self.kind == "policy":
            z = out - out.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            return out.argmax(1), p.max(1)
        return out.argmax(1), out.max(1)

    # -- compatibility & files ----------------------------------------------------------
    def mismatch(self, world) -> str | None:
        """Why this agent can't act in ``world`` (None if it can)."""
        if world.obs_size != self.obs_size:
            return (f"it learned with {self.obs_size} senses but the world '{world.name}' gives "
                    f"{world.obs_size}")
        a = world.action_spec
        if a.kind != self.action.kind or a.size != self.action.size:
            return (f"it learned to control {self.action.size} {self.action.kind} action(s) but the world "
                    f"'{world.name}' has {a.size} {a.kind} action(s)")
        if self.kind == "table" and getattr(world.make_env(), "n_cells", None) != self.n_states:
            return "its Q-table was made for a maze of a different size"
        return None

    def to_payload(self) -> dict:
        def sd(net):
            return None if net is None else {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        return {
            "format": FORMAT, "version": 1, "kind": self.kind, "algorithm": self.algorithm, "world": self.world,
            "obs_size": self.obs_size, "obs_names": list(self.obs_names), "action": self.action.to_dict(),
            "norm": self.norm.state(), "actor": sd(self.actor), "critic": sd(self.critic),
            "log_std": None if self.log_std is None else self.log_std.detach().cpu().clone(),
            "qnet": sd(self.qnet), "qtable": None if self.qtable is None else torch.tensor(self.qtable),
            "n_states": self.n_states, "steps": int(self.steps), "episodes": int(self.episodes),
            "generations": int(self.generations), "best": self.best,
        }

    @classmethod
    def from_payload(cls, payload: dict, model):
        if not isinstance(payload, dict) or payload.get("format") != FORMAT:
            raise NBError("The saved simulation agent isn't in a format this version understands.")
        action = ActionSpec.from_dict(payload["action"])
        obs = int(payload["obs_size"])
        kind = payload["kind"]
        norm = RunningNorm.from_state(payload["norm"], obs) if payload.get("norm") else None
        common = dict(world=payload.get("world", ""), obs_names=payload.get("obs_names"),
                      algorithm=payload.get("algorithm", ""), norm=norm)
        try:
            if kind == "table":
                ag = cls("table", obs, action, qtable=np.asarray(payload["qtable"], dtype=np.float64),
                         n_states=payload.get("n_states"), **common)
            elif kind == "q":
                qnet, rows = build_net(model, obs, len(action.choices()))
                qnet.load_state_dict(payload["qnet"])
                ag = cls("q", obs, action, qnet=qnet, **common)
                ag.rows = rows
            else:
                actor, rows = build_net(model, obs, action.size)
                actor.load_state_dict(payload["actor"])
                critic = None
                if payload.get("critic") is not None:
                    critic, _ = build_net(model, obs, 1)
                    critic.load_state_dict(payload["critic"])
                log_std = payload.get("log_std")
                ag = cls("policy", obs, action, actor=actor, critic=critic,
                         log_std=None if log_std is None else torch.nn.Parameter(log_std.clone().float()), **common)
                ag.rows = rows
        except (RuntimeError, KeyError) as e:
            raise NBError("The saved agent doesn't fit this model's layers.",
                          hint="Load it into a model with the same layer blocks it was trained with.") from e
        ag.steps = int(payload.get("steps", 0))
        ag.episodes = int(payload.get("episodes", 0))
        ag.generations = int(payload.get("generations", 0))
        ag.best = payload.get("best")
        return ag
