"""The base class every simulated world ("environment") builds on.

An environment is one copy of a world that one agent acts in::

    obs = env.reset(seed)
    obs, reward, terminated, truncated = env.step(action)

Rewards and episode endings are assembled here from the world's parts
(``Reward``, ``CustomReward``, ``EndWhen``, ``EndAfter``) so every world handles
them the same way; a world only reports its *reward terms* for each step (how far
the car moved, whether it flipped, …).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ..errors import NBError

STATE_KEYS = ("distance", "speed", "tilt", "spin", "height", "x", "y", "time", "fuel", "touching_ground")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


class ActionSpec:
    """What an agent controls.

    * ``continuous``: ``size`` numbers, each between -1 and 1 (throttle, lean, …)
    * ``discrete``: one choice out of ``size`` (push left / push right, …)
    """

    def __init__(self, kind: str, size: int, names, dim_values=None, noop: int = 0):
        self.kind = kind
        self.size = int(size)
        self.names = list(names)
        # For continuous actions: the values each dimension takes when DQN chooses from a menu.
        self.dim_values = [list(v) for v in dim_values] if dim_values else [[-1.0, 0.0, 1.0]] * self.size
        self.noop = int(noop)

    @property
    def discrete(self) -> bool:
        return self.kind == "discrete"

    def choices(self) -> list:
        """Continuous actions turned into a menu of fixed choices (for DQN)."""
        if self.discrete:
            return list(range(self.size))
        return [list(c) for c in itertools.product(*self.dim_values)]

    def clean(self, action):
        """Turn whatever the agent produced into a valid action: an int, or a list of floats in [-1, 1]."""
        if self.discrete:
            if isinstance(action, (int, np.integer)) and not isinstance(action, bool):
                a = int(action)
            else:
                try:
                    a = int(np.asarray(action).reshape(-1)[0])
                except (TypeError, ValueError, IndexError):
                    raise NBError(f"This world expects one choice (a whole number), got {action!r}.") from None
            return min(max(a, 0), self.size - 1)
        if isinstance(action, np.ndarray):
            vals = action.reshape(-1).tolist()
        elif isinstance(action, (list, tuple)):
            vals = list(action)
        else:
            vals = [action]
        if len(vals) != self.size:
            if len(vals) == 1 and self.size > 1:
                vals = vals * self.size
            else:
                raise NBError(f"This world expects {self.size} action numbers, got {len(vals)}.")
        out = []
        for v in vals:
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise NBError(f"Actions must be numbers, got {v!r}.") from None
            if v != v:
                v = 0.0
            out.append(-1.0 if v < -1.0 else (1.0 if v > 1.0 else v))
        return out

    def zero(self):
        return self.noop if self.discrete else [0.0] * self.size

    def sample(self, rng):
        if self.discrete:
            return int(rng.integers(self.size))
        return rng.uniform(-1.0, 1.0, self.size).tolist()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "size": self.size, "names": list(self.names), "dim_values": self.dim_values,
                "noop": self.noop}

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpec":
        return cls(d["kind"], d["size"], d.get("names") or [], d.get("dim_values"), d.get("noop", 0))

    def describe(self) -> str:
        if self.discrete:
            return f"chooses one of {self.size} actions ({', '.join(self.names)})"
        return f"controls {self.size} number{'s' if self.size != 1 else ''} ({', '.join(self.names)})"


# ---------------------------------------------------------------------------
# Calling the user's expressions safely
# ---------------------------------------------------------------------------


def _user_call(fn, state, bid, what):
    try:
        return fn(state)
    except NBError as e:
        if e.nb_block is None:
            e.nb_block = bid
        raise
    except KeyError as e:
        raise NBError(f"Your {what} used {e} but the world doesn't have that value.",
                      hint=f"Available values: {', '.join(sorted(state))}.", block_id=bid) from None
    except Exception as e:  # noqa: BLE001 - user code may raise anything
        raise NBError(f"Your {what} caused an error: {type(e).__name__}: {e}", block_id=bid) from None


def user_number(fn, state, bid, what="custom reward") -> float:
    v = _user_call(fn, state, bid, what)
    if isinstance(v, bool):
        return float(v)
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise NBError(f"Your {what} must give a number, but it gave {v!r}.", block_id=bid) from None
    if not math.isfinite(v):
        raise NBError(f"Your {what} gave {v}, which isn't a normal number.", block_id=bid)
    return v


def user_bool(fn, state, bid, what="'end when' condition") -> bool:
    v = _user_call(fn, state, bid, what)
    try:
        return bool(v)
    except Exception:  # noqa: BLE001
        raise NBError(f"Your {what} must be true or false, but it gave {v!r}.", block_id=bid) from None


# ---------------------------------------------------------------------------
# Environment base class
# ---------------------------------------------------------------------------


class Env:
    KIND = ""
    TITLE = ""
    DT = 1 / 30                 # seconds per decision step
    RECORD_EVERY = 1            # replay keeps every N-th step
    FRAME_SUBSTEPS = 1          # frames recorded per step (the maze slides smoothly between cells)
    TURN_BASED = False          # people play it one key press at a time
    DEFAULT_TIME = 20.0         # default 'end after' (seconds)
    DEFAULT_GRAVITY = 9.8
    USES_GRAVITY = True
    SUPPORTS_STUCK = False
    DEFAULT_STUCK = None
    HANDLES = ()                # names of the part classes this world understands (besides the generic ones)
    REWARD_KINDS: dict = {"time": "every step"}
    EVENT_KINDS = ("finish", "flip")   # one-off rewards that can also end the episode
    DEFAULT_REWARDS = (("time", 1.0, False),)
    KEYS: dict = {}
    PROGRESS_UNIT = ""          # e.g. " m" when the world's 'distance' is in metres
    # Hints for the learning algorithms (rollout sizes etc.).
    HINTS: dict = {}

    def __init__(self, cfg):
        self.cfg = cfg
        self.rewards = [(k, float(w), bool(e)) for (k, w, e, _bid) in cfg.rewards]
        self.customs = list(cfg.customs)
        self.end_whens = list(cfg.end_whens)
        self.gravity = float(cfg.gravity) if cfg.gravity is not None else float(self.DEFAULT_GRAVITY)
        self.max_steps = max(1, int(round(float(cfg.time_limit) / self.DT)))
        self.t = 0.0
        self.steps = 0
        self.ep_return = 0.0
        self.last_reward = 0.0
        self.end_reason = ""
        self.finished = False
        self.seed = 0
        self.obs_names: list = []
        self.action_spec: ActionSpec | None = None
        self._setup()
        self.obs_size = len(self.obs_names)

    # -- to implement ---------------------------------------------------------
    @classmethod
    def configure(cls, cfg, parts) -> list:
        """Read world-specific parts into ``cfg``; return the parts that don't apply."""
        return list(parts)

    @classmethod
    def default_time(cls, cfg) -> float:
        return cls.DEFAULT_TIME

    def _setup(self):
        raise NotImplementedError

    def _reset(self, seed: int):
        raise NotImplementedError

    def _step(self, action) -> tuple[dict, bool]:
        """Advance one step. Return (reward terms, episode over because of the world itself)."""
        raise NotImplementedError

    def observe(self) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> dict:
        raise NotImplementedError

    def scene(self) -> dict:
        raise NotImplementedError

    def frame(self) -> list:
        raise NotImplementedError

    def frames(self) -> list:
        """The frames for the step just taken (FRAME_SUBSTEPS of them)."""
        return [self.frame()]

    def fx(self):
        return None

    def hud(self) -> str:
        return f"reward {self.ep_return:.1f} · {self.t:.1f} s"

    def progress(self) -> float:
        """The world's 'distance' for charts (metres for the car, task-specific elsewhere)."""
        return float(self.state().get("distance", 0.0))

    def human_action(self, keys: dict):
        return self.action_spec.zero()

    def overlay(self, agent=None):
        return None

    # -- shared machinery -----------------------------------------------------
    def reset(self, seed: int = 0) -> np.ndarray:
        self.t = 0.0
        self.steps = 0
        self.ep_return = 0.0
        self.last_reward = 0.0
        self.end_reason = ""
        self.finished = False
        self.seed = int(seed)
        self._reset(self.seed)
        return self.observe()

    def step(self, action):
        a = self.action_spec.clean(action)
        self.steps += 1
        self.t = self.steps * self.DT
        terms, over = self._step(a)
        r = 0.0
        terminated = bool(over)
        for kind, w, end in self.rewards:
            v = terms.get(kind)
            if v:
                r += w * v
                if end and not terminated:
                    terminated = True
                    self.end_reason = self.end_reason or kind
        if self.customs or self.end_whens:
            st = self.state()
            for fn, bid in self.customs:
                r += user_number(fn, st, bid)
            if not terminated:
                for fn, bid in self.end_whens:
                    if user_bool(fn, st, bid):
                        terminated = True
                        self.end_reason = "your 'end when' rule"
                        break
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated and not self.end_reason:
            self.end_reason = "time"
        self.ep_return += r
        self.last_reward = r
        return self.observe(), r, terminated, truncated

    def base_state(self) -> dict:
        return {"distance": 0.0, "speed": 0.0, "tilt": 0.0, "spin": 0.0, "height": 0.0, "x": 0.0, "y": 0.0,
                "time": round(self.t, 4), "fuel": 0.0, "touching_ground": 0, "reward": self.ep_return,
                "steps": self.steps}

    def episode_info(self) -> dict:
        return {"reward": float(self.ep_return), "distance": float(self.progress()), "steps": int(self.steps),
                "finished": bool(self.finished), "reason": self.end_reason}
