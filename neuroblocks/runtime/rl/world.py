"""Worlds: the 'create world' block.

``World("car", name="world", parts=[...])`` checks the parts, fills in sensible
defaults and can then make any number of independent copies of itself
(``world.make_env()``) for agents to act in.
"""
from __future__ import annotations

import importlib
import zlib

from ..core import STATE, register
from ..errors import NBError
from ..events import emit
from . import parts as P

__all__ = ["World", "WORLD_KINDS", "warn_part"]

# kind -> (module, class)
WORLD_KINDS = {
    "car": ("env_car", "CarEnv"),
    "cartpole": ("env_classic", "CartPoleEnv"),
    "mountaincar": ("env_classic", "MountainCarEnv"),
    "pendulum": ("env_classic", "PendulumEnv"),
    "lander": ("env_lander", "LanderEnv"),
    "maze": ("env_maze", "MazeEnv"),
    "flappy": ("env_flappy", "FlappyEnv"),
}

_ALIASES = {
    "cart_pole": "cartpole", "balance": "cartpole", "pole": "cartpole", "balance_a_pole": "cartpole",
    "mountain_car": "mountaincar", "mountain": "mountaincar", "swing_up": "pendulum", "rocket": "lander",
    "rocket_lander": "lander", "lunar_lander": "lander", "moon_lander": "lander", "grid": "maze",
    "grid_world": "maze", "gridworld": "maze", "flappy_bird": "flappy", "bird": "flappy", "car_on_terrain": "car",
    "terrain": "car",
}


def env_class(kind: str):
    mod, cls = WORLD_KINDS[kind]
    return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)


def normalize_kind(kind) -> str:
    k = str(kind or "car").strip().lower().replace("-", "_").replace(" ", "_")
    k = _ALIASES.get(k, k)
    return k


def warn_part(part, text: str):
    emit("log", level="warn", text=text, block=getattr(part, "bid", None))


def _all_reward_kinds() -> set:
    kinds = set(P.REWARD_KINDS)
    for k in WORLD_KINDS:
        try:
            kinds |= set(env_class(k).REWARD_KINDS)
        except Exception:  # noqa: BLE001
            pass
    return kinds


def _flatten(parts, bid):
    out = []
    stack = list(parts or [])[::-1]
    while stack:
        p = stack.pop()
        if p is None:
            continue
        if isinstance(p, (list, tuple)):
            stack.extend(list(p)[::-1])
            continue
        if not isinstance(p, P.Part):
            raise NBError(f"'{p}' can't go inside a world — only world parts (terrain, sensors, rewards, …) can.",
                          block_id=bid)
        out.append(p)
    return out


class WorldConfig:
    """Everything an environment needs to know, resolved from the world's parts."""

    def __init__(self, kind: str, name: str, bid):
        self.kind = kind
        self.name = name
        self.bid = bid
        self.seed = int(STATE.seed) if STATE.seed is not None else zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF
        self.same_world = True
        self.gravity = None
        self.time_limit = None
        self.stuck = None
        self.rewards: list = []    # (kind, weight, end, block id)
        self.customs: list = []    # (fn, block id)
        self.end_whens: list = []  # (fn, block id)
        self.default_rewards = False
        self.default_notes: list = []


class World:
    """A simulated world that agents can learn in."""

    is_world = True

    def __init__(self, kind="car", name="world", parts=(), _bid=None, **unused):
        k = normalize_kind(kind)
        if k not in WORLD_KINDS:
            raise NBError(f"Unknown world '{kind}'.", hint=f"Choose one of: {', '.join(WORLD_KINDS)}.", block_id=_bid)
        for key in unused:
            if not key.startswith("_"):
                emit("log", level="warn", text=f"The '{key}' setting doesn't apply to worlds, so it was ignored.",
                     block=_bid)
        self.kind = k
        self.name = str(name or "world")
        self.bid = _bid
        self.env_cls = env_class(k)
        self.title = self.env_cls.TITLE
        self.parts = _flatten(parts, _bid)
        self.cfg = self._configure()
        probe = self.make_env()
        self.obs_size = probe.obs_size
        self.obs_names = list(probe.obs_names)
        self.action_spec = probe.action_spec
        self.dt = probe.DT
        self.max_steps = probe.max_steps
        if self.obs_size == 0:
            raise NBError("The world gives the agent nothing to sense.", block_id=_bid)
        for note in self.cfg.default_notes:
            emit("log", level="info", text=note, block=_bid)
        register(self)

    # -- parts -----------------------------------------------------------------
    def _configure(self) -> WorldConfig:
        E = self.env_cls
        cfg = WorldConfig(self.kind, self.name, self.bid)
        specific = []
        known_kinds = None
        for p in self.parts:
            if isinstance(p, P.SameWorld):
                cfg.same_world = p.same
            elif isinstance(p, P.Gravity):
                if E.USES_GRAVITY:
                    cfg.gravity = p.g
                else:
                    warn_part(p, f"Gravity doesn't matter in the {E.TITLE} world, so it was ignored.")
            elif isinstance(p, P.EndAfter):
                cfg.time_limit = p.seconds
            elif isinstance(p, P.EndIfStuck):
                if E.SUPPORTS_STUCK:
                    cfg.stuck = p.seconds
                else:
                    warn_part(p, "'end if stuck' only works in the car world, so it was ignored.")
            elif isinstance(p, P.EndWhen):
                cfg.end_whens.append((p.fn, p.bid))
            elif isinstance(p, P.CustomReward):
                cfg.customs.append((p.fn, p.bid))
            elif isinstance(p, P.Reward):
                if p.kind not in E.REWARD_KINDS:
                    if known_kinds is None:
                        known_kinds = _all_reward_kinds()
                    if p.kind in known_kinds:
                        warn_part(p, f"The '{p.kind}' reward doesn't apply in the {E.TITLE} world, so it was ignored. "
                                     f"Rewards here: {', '.join(E.REWARD_KINDS)}.")
                        continue
                    raise NBError(f"Unknown reward '{p.kind}'.",
                                  hint=f"In the {E.TITLE} world you can use: {', '.join(E.REWARD_KINDS)}.",
                                  block_id=p.bid)
                end = p.end
                if end and p.kind not in E.EVENT_KINDS:
                    warn_part(p, f"Only one-off rewards ({', '.join(k for k in E.EVENT_KINDS if k in E.REWARD_KINDS)})"
                                 f" can end the episode; '{p.kind}' happens every step, so 'end' was ignored.")
                    end = False
                cfg.rewards.append((p.kind, p.weight, end, p.bid))
            else:
                specific.append(p)
        for p in E.configure(cfg, specific):
            warn_part(p, f"'{p.label}' doesn't do anything in the {E.TITLE} world, so it was ignored.")
        if not cfg.rewards and not cfg.customs:
            cfg.rewards = [(k, w, e, None) for k, w, e in E.DEFAULT_REWARDS]
            cfg.default_rewards = True
        if cfg.time_limit is None:
            cfg.time_limit = E.default_time(cfg)
        if E.SUPPORTS_STUCK and cfg.stuck is None:
            cfg.stuck = E.DEFAULT_STUCK
        return cfg

    # -- environments ------------------------------------------------------------
    def make_env(self):
        return self.env_cls(self.cfg)

    @property
    def same_world(self) -> bool:
        return self.cfg.same_world

    @property
    def seed(self) -> int:
        return self.cfg.seed

    def describe(self) -> str:
        return f"world '{self.name}' ({self.title})"

    def __repr__(self):
        return f"<World {self.name}: {self.kind}, {len(self.parts)} parts>"

    def reward_text(self) -> str:
        bits = []
        for kind, w, end, _ in self.cfg.rewards:
            bits.append(f"{kind} {w:+g}" + (" (ends)" if end else ""))
        if self.cfg.customs:
            bits.append(f"{len(self.cfg.customs)} custom")
        return ", ".join(bits)

    def summary(self) -> str:
        sense = ", ".join(self.obs_names[:12]) + (" …" if len(self.obs_names) > 12 else "")
        return (f"{self.describe()}: the agent senses {self.obs_size} number{'s' if self.obs_size != 1 else ''} "
                f"({sense}) and {self.action_spec.describe()}. Rewards: {self.reward_text()}. "
                f"Episodes end after {self.cfg.time_limit:g} s.")

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "title": self.title, "obs_size": self.obs_size,
                "obs_names": self.obs_names, "action": self.action_spec.to_dict(), "dt": self.dt,
                "seconds": self.cfg.time_limit, "same_world": self.cfg.same_world, "rewards": self.reward_text()}
