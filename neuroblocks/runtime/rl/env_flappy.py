"""Flappy bird: flap to fly through the gaps between the pipes.

The pipes in replays are a fixed number of recycled bodies: a pipe that leaves the screen on the
left jumps (off-screen) to the right and becomes the next pipe.
"""
from __future__ import annotations

import math

import numpy as np

from .env_base import ActionSpec, Env
from .parts import Gap, PipeSpeed

WIDTH, HEIGHT = 16.0, 10.0
FLOOR = 0.8
BIRD_X = 4.0
BIRD_R = 0.35
PIPE_W = 1.2
PIPE_H = 12.0          # drawn pipe length (extends off-screen)
SPACING = 5.5          # metres between pipes
FIRST_PIPE = 12.0
SLOTS = 5              # recycled pipe bodies (pairs) in replays
FLAP_SPEED = 6.5


class FlappyEnv(Env):
    KIND = "flappy"
    TITLE = "flappy bird"
    DT = 1 / 30
    DEFAULT_TIME = 30.0
    DEFAULT_GRAVITY = 20.0
    HANDLES = ("Gap", "PipeSpeed")
    REWARD_KINDS = {
        "pipe": "once for every pipe the bird flies through",
        "time": "1 every step the bird keeps flying",
        "distance": "metres flown",
        "flip": "once, when the bird crashes",
        "finish": "once, if the bird is still flying when the time runs out",
        "energy": "1 for every flap",
    }
    DEFAULT_REWARDS = (("pipe", 1.0, False), ("time", 0.1, False), ("flip", -1.0, True))
    KEYS = {" ": "flap", "ArrowUp": "flap"}
    HINTS = {"ppo_rollout": 2048, "ppo_epochs": 10, "ppo_minibatch": 256}

    @classmethod
    def configure(cls, cfg, parts):
        cfg.gap = 3.0
        cfg.pipe_speed = 3.5
        unused = []
        for p in parts:
            if isinstance(p, Gap):
                cfg.gap = p.size
            elif isinstance(p, PipeSpeed):
                cfg.pipe_speed = p.speed
            else:
                unused.append(p)
        return unused

    def _setup(self):
        self.gap = float(self.cfg.gap)
        self.speed = float(self.cfg.pipe_speed)
        self.obs_names = ["height", "vertical speed", "next pipe distance", "next gap offset",
                          "pipe after distance", "pipe after gap offset"]
        self.action_spec = ActionSpec("discrete", 2, ["glide", "flap"], noop=0)
        self._layout = None
        self.gaps = []
        self._flap_was = False

    def _make_layout(self, seed):
        n = int(self.max_steps * self.DT * self.speed / SPACING) + SLOTS + 4
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 31])
        lo, hi = FLOOR + self.gap / 2 + 0.6, HEIGHT - self.gap / 2 - 0.6
        if hi < lo:
            lo = hi = (FLOOR + HEIGHT) / 2
        gaps = []
        prev = (lo + hi) / 2
        for _ in range(n):
            g = float(np.clip(prev + rng.uniform(-3.0, 3.0), lo, hi))
            gaps.append(g)
            prev = g
        self.gaps = gaps
        self._layout = seed

    def _reset(self, seed):
        lseed = self.cfg.seed if self.cfg.same_world else seed
        if self._layout != lseed:
            self._make_layout(lseed)
        self.y = (FLOOR + HEIGHT) / 2 + 0.5
        self.vy = 0.0
        self.scroll = 0.0
        self.passed = 0
        self.crashed = False
        self.flapped = False

    def _pipe_x(self, k):
        return FIRST_PIPE + SPACING * k - self.scroll

    def _step(self, a):
        terms = {"time": 1.0, "distance": self.speed * self.DT}
        self.flapped = a == 1
        if self.flapped:
            self.vy = FLAP_SPEED
            terms["energy"] = 1.0
        self.vy = max(self.vy - self.gravity * self.DT, -12.0)
        self.y += self.vy * self.DT
        self.scroll += self.speed * self.DT
        over = False
        # passed pipes
        while self._pipe_x(self.passed) + PIPE_W / 2 < BIRD_X - BIRD_R:
            self.passed += 1
            terms["pipe"] = terms.get("pipe", 0.0) + 1.0
        # crash?
        hit = self.y - BIRD_R < FLOOR or self.y + BIRD_R > HEIGHT + 0.5
        k = self.passed
        r2 = BIRD_R * BIRD_R
        for j in (k, k + 1):
            if hit or j >= len(self.gaps):
                break
            px = self._pipe_x(j)
            cx = min(max(BIRD_X, px - PIPE_W / 2), px + PIPE_W / 2)   # closest x on the pipe
            dx2 = (BIRD_X - cx) ** 2
            if dx2 >= r2:
                continue
            g = self.gaps[j]
            bottom, top = g - self.gap / 2, g + self.gap / 2
            # closest points on the lower pipe (y ≤ bottom) and the upper pipe (y ≥ top)
            if dx2 + (self.y - min(self.y, bottom)) ** 2 < r2 or dx2 + (self.y - max(self.y, top)) ** 2 < r2:
                hit = True
        if hit:
            self.crashed = True
            terms["flip"] = 1.0
            self.end_reason = "crashed"
            over = True
        elif self.steps >= self.max_steps:
            terms["finish"] = 1.0
            self.finished = True
        return terms, over

    def observe(self):
        a, b = self.passed, self.passed + 1
        ga = self.gaps[a] if a < len(self.gaps) else HEIGHT / 2
        gb = self.gaps[b] if b < len(self.gaps) else HEIGHT / 2
        return np.asarray([self.y / HEIGHT, self.vy / 10.0, (self._pipe_x(a) - BIRD_X) / 6.0, (ga - self.y) / 5.0,
                           (self._pipe_x(b) - BIRD_X) / 6.0, (gb - self.y) / 5.0], dtype=np.float32)

    def state(self):
        st = self.base_state()
        st.update({"distance": self.scroll, "speed": self.speed,
                   "tilt": math.degrees(math.atan2(self.vy, self.speed)), "height": self.y - FLOOR, "x": BIRD_X,
                   "y": self.y, "pipes": self.passed, "vertical_speed": self.vy,
                   "touching_ground": 1 if self.y - BIRD_R <= FLOOR else 0})
        return st

    def progress(self):
        return float(self.passed)

    def episode_info(self):
        info = super().episode_info()
        info["distance"] = round(self.scroll, 3)
        info["pipes"] = self.passed
        return info

    def scene(self):
        bodies = [{"name": "bird", "shape": "circle", "r": BIRD_R, "color": "#f1c40f"}]
        for j in range(SLOTS):
            bodies.append({"name": f"pipe {j + 1} bottom", "shape": "rect", "w": PIPE_W, "h": PIPE_H,
                           "color": "#2ecc71"})
            bodies.append({"name": f"pipe {j + 1} top", "shape": "rect", "w": PIPE_W, "h": PIPE_H,
                           "color": "#2ecc71"})
        return {
            "bounds": [0.0, 0.0, WIDTH, HEIGHT],
            "camera": {"mode": "fixed"},
            "sky": "#70c5ce", "ground_fill": "#ded895",
            "static": [
                {"type": "ground", "points": [[0.0, FLOOR], [WIDTH, FLOOR]], "color": "#5aa83c"},
            ],
            "bodies": bodies,
        }

    def frame(self):
        tilt = max(-1.0, min(0.5, math.atan2(self.vy, self.speed * 2.0)))
        out = [BIRD_X, self.y, tilt]
        # slot j shows pipe k where k ≡ j (mod SLOTS), the earliest one that hasn't left the screen
        first = max(0, int((self.scroll - FIRST_PIPE - PIPE_W) // SPACING) + 1)
        for j in range(SLOTS):
            k = first + ((j - first) % SLOTS)
            px = self._pipe_x(k)
            g = self.gaps[k] if k < len(self.gaps) else HEIGHT / 2
            out += [px, g - self.gap / 2 - PIPE_H / 2, 0.0, px, g + self.gap / 2 + PIPE_H / 2, 0.0]
        return out

    def hud(self):
        s = f"pipes {self.passed} · {self.t:.1f} s"
        if self.crashed:
            s += " · crashed"
        return s

    def human_action(self, keys):
        k = keys or {}
        down = bool(k.get(" ") or k.get("ArrowUp") or k.get("Space"))
        flap = down and not self._flap_was   # one flap per key press
        self._flap_was = down
        return 1 if flap else 0
