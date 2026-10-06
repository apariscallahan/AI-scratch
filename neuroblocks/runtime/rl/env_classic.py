"""Classic control worlds: cart-pole, mountain car and pendulum (the standard textbook equations)."""
from __future__ import annotations

import math

import numpy as np

from .env_base import ActionSpec, Env
from .parts import PoleLength


def _rng(seed):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, 11])


# ---------------------------------------------------------------------------
# Cart-pole: balance a pole on a cart by pushing it left or right
# ---------------------------------------------------------------------------


class CartPoleEnv(Env):
    KIND = "cartpole"
    TITLE = "cart-pole"
    DT = 0.02
    RECORD_EVERY = 2
    DEFAULT_TIME = 10.0          # 500 steps
    HANDLES = ("PoleLength",)
    REWARD_KINDS = {
        "time": "1 every step the pole stays up",
        "upright": "1 when the pole points straight up, every step",
        "finish": "once, if the pole is still up when the time runs out",
        "flip": "once, when the pole falls over or the cart leaves the track",
        "distance": "how far the cart is from the middle of the track (m), every step",
        "speed": "how fast the cart moves (m/s), every step",
    }
    DEFAULT_REWARDS = (("time", 1.0, False),)
    KEYS = {"ArrowLeft": "push the cart left", "ArrowRight": "push the cart right"}
    HINTS = {"ppo_rollout": 512, "ppo_epochs": 10, "ppo_minibatch": 128}
    X_LIMIT = 2.4
    THETA_LIMIT = 12 * 2 * math.pi / 360

    @classmethod
    def configure(cls, cfg, parts):
        cfg.pole_length = 1.0
        unused = []
        for p in parts:
            if isinstance(p, PoleLength):
                cfg.pole_length = p.length
            else:
                unused.append(p)
        return unused

    def _setup(self):
        self.pole_len = float(self.cfg.pole_length)
        self.half = self.pole_len / 2
        self.masscart, self.masspole = 1.0, 0.1
        self.total_mass = self.masscart + self.masspole
        self.pml = self.masspole * self.half
        self.force_mag = 10.0
        self.obs_names = ["cart position", "cart speed", "pole angle", "pole spin"]
        self.action_spec = ActionSpec("discrete", 2, ["push left", "push right"], noop=0)
        self.s = [0.0, 0.0, 0.0, 0.0]
        self._alt = 0

    def _reset(self, seed):
        self.s = _rng(seed).uniform(-0.05, 0.05, 4).tolist()
        self.fallen = False

    def _step(self, a):
        x, x_dot, th, th_dot = self.s
        force = self.force_mag if a == 1 else -self.force_mag
        ct, st = math.cos(th), math.sin(th)
        temp = (force + self.pml * th_dot * th_dot * st) / self.total_mass
        th_acc = (self.gravity * st - ct * temp) / (self.half * (4.0 / 3.0 - self.masspole * ct * ct / self.total_mass))
        x_acc = temp - self.pml * th_acc * ct / self.total_mass
        tau = self.DT
        x += tau * x_dot
        x_dot += tau * x_acc
        th += tau * th_dot
        th_dot += tau * th_acc
        self.s = [x, x_dot, th, th_dot]
        terms = {"time": 1.0, "upright": math.cos(th), "distance": abs(x), "speed": abs(x_dot)}
        over = False
        if abs(x) > self.X_LIMIT or abs(th) > self.THETA_LIMIT:
            terms["flip"] = 1.0
            self.fallen = True
            self.end_reason = "the pole fell" if abs(th) > self.THETA_LIMIT else "the cart left the track"
            over = True
        elif self.steps >= self.max_steps:
            terms["finish"] = 1.0
            self.finished = True
        return terms, over

    def observe(self):
        x, x_dot, th, th_dot = self.s
        return np.asarray([x / self.X_LIMIT, x_dot / 3.0, th / self.THETA_LIMIT, th_dot / 3.0], dtype=np.float32)

    def state(self):
        x, x_dot, th, th_dot = self.s
        st = self.base_state()
        st.update({"distance": x, "speed": x_dot, "tilt": math.degrees(th), "spin": math.degrees(th_dot),
                   "height": self.pole_len * math.cos(th), "x": x, "y": 0.0, "touching_ground": 1,
                   "pole_angle": math.degrees(th), "cart_position": x})
        return st

    def progress(self):
        return float(self.steps)

    def episode_info(self):
        info = super().episode_info()
        info["distance"] = 0.0
        info["finished"] = bool(self.finished)
        return info

    def scene(self):
        L = self.pole_len
        lim = self.X_LIMIT
        return {
            "bounds": [-lim - 0.8, -0.6, lim + 0.8, max(1.6, L + 0.9)],
            "camera": {"mode": "fixed"},
            "sky": "#eef6ff", "ground_fill": "#d7ccc8",
            "static": [
                {"type": "ground", "points": [[-lim - 0.8, -0.05], [lim + 0.8, -0.05]], "color": "#8d6e63"},
                {"type": "line", "points": [[-lim - 0.3, 0.0], [lim + 0.3, 0.0]], "color": "#333333", "width": 0.04},
                {"type": "rect", "x": -lim - 0.35, "y": 0.0, "w": 0.06, "h": 0.3, "color": "#555555"},
                {"type": "rect", "x": lim + 0.29, "y": 0.0, "w": 0.06, "h": 0.3, "color": "#555555"},
            ],
            "bodies": [
                {"name": "cart", "shape": "rect", "w": 0.5, "h": 0.3, "color": "#34495e"},
                {"name": "pole", "shape": "rect", "w": 0.08, "h": round(L, 3), "color": "#c0392b"},
            ],
        }

    def frame(self):
        x, _, th, _ = self.s
        px, py = x, 0.3
        return [x, 0.15, 0.0, px + self.half * math.sin(th), py + self.half * math.cos(th), -th]

    def hud(self):
        return f"{self.t:.1f} s · pole {math.degrees(self.s[2]):+.1f}°"

    def human_action(self, keys):
        k = keys or {}
        if k.get("ArrowRight") and not k.get("ArrowLeft"):
            return 1
        if k.get("ArrowLeft") and not k.get("ArrowRight"):
            return 0
        self._alt ^= 1  # no key: alternate tiny left/right pushes, which cancel out
        return self._alt


# ---------------------------------------------------------------------------
# Mountain car: too weak to drive up the hill — swing back and forth to build momentum
# ---------------------------------------------------------------------------


class MountainCarEnv(Env):
    KIND = "mountaincar"
    TITLE = "mountain car"
    DT = 1 / 30
    DEFAULT_TIME = 10.0          # 300 steps
    REWARD_KINDS = {
        "finish": "once, when the car reaches the flag",
        "height": "metres of height gained above the best so far",
        "distance": "metres closer to the flag than ever before",
        "speed": "how fast the car moves (m/s), every step",
        "energy": "how hard the motor works (throttle²), every step",
        "time": "1 every step",
    }
    EVENT_KINDS = ("finish",)
    DEFAULT_REWARDS = (("finish", 100.0, True), ("height", 2.0, False))
    KEYS = {"ArrowLeft": "drive left", "ArrowRight": "drive right"}
    HINTS = {"ppo_rollout": 2048, "ppo_epochs": 10, "ppo_minibatch": 256}
    MIN_POS, MAX_POS, MAX_SPEED, GOAL, POWER = -1.2, 0.6, 0.07, 0.45, 0.0015
    SX, SY = 10.0, 4.5           # drawing scale: metres per unit of position / height

    def _setup(self):
        self.obs_names = ["position", "velocity"]
        self.action_spec = ActionSpec("continuous", 1, ["push"], [[-1.0, 0.0, 1.0]])
        self.pos, self.vel = -0.5, 0.0

    def _reset(self, seed):
        self.pos = float(_rng(seed).uniform(-0.6, -0.4))
        self.vel = 0.0
        self.best_x = self.pos
        self.best_h = self._h(self.pos)

    def _h(self, pos):
        return self.SY * math.sin(3 * pos)

    def _step(self, a):
        f = a[0]
        g = 0.0025 * self.gravity / 9.8
        self.vel += f * self.POWER - g * math.cos(3 * self.pos)
        self.vel = min(max(self.vel, -self.MAX_SPEED), self.MAX_SPEED)
        self.pos += self.vel
        if self.pos < self.MIN_POS:
            self.pos = self.MIN_POS
            if self.vel < 0:
                self.vel = 0.0
        self.pos = min(self.pos, self.MAX_POS)
        h = self._h(self.pos)
        terms = {"time": 1.0, "energy": f * f, "speed": abs(self.vel) * self.SX / self.DT}
        if self.pos > self.best_x:
            terms["distance"] = (self.pos - self.best_x) * self.SX
            self.best_x = self.pos
        if h > self.best_h:
            terms["height"] = h - self.best_h
            self.best_h = h
        over = False
        if self.pos >= self.GOAL:
            terms["finish"] = 1.0
            self.finished = True
            self.end_reason = "reached the flag"
            over = True
        return terms, over

    def observe(self):
        return np.asarray([(self.pos + 0.3) / 0.9, self.vel / self.MAX_SPEED], dtype=np.float32)

    def _slope(self):
        return math.atan(self.SY * 3 * math.cos(3 * self.pos) / self.SX)

    def state(self):
        st = self.base_state()
        st.update({"distance": self.pos * self.SX, "speed": self.vel * self.SX / self.DT,
                   "tilt": math.degrees(self._slope()), "height": self._h(self.pos), "x": self.pos * self.SX,
                   "y": self._h(self.pos), "touching_ground": 1, "position": self.pos, "velocity": self.vel})
        return st

    def progress(self):
        return float(self.best_x * self.SX)

    def scene(self):
        xs = np.linspace(self.MIN_POS, self.MAX_POS, 120)
        pts = [[round(float(x * self.SX), 3), round(float(self._h(x)), 3)] for x in xs]
        gx = self.GOAL * self.SX
        return {
            "bounds": [round(self.MIN_POS * self.SX - 0.5, 2), -6.5, round(self.MAX_POS * self.SX + 0.5, 2), 7.0],
            "camera": {"mode": "fixed"},
            "sky": "#dff1ff", "ground_fill": "#a1887f",
            "static": [
                {"type": "ground", "points": pts, "color": "#6d9b3a"},
                {"type": "flag", "x": round(gx, 3), "y": round(self._h(self.GOAL), 3), "label": "goal"},
            ],
            "bodies": [
                {"name": "car", "shape": "rect", "w": 1.3, "h": 0.5, "color": "#2e86de"},
                {"name": "rear wheel", "shape": "circle", "r": 0.22, "color": "#333333", "spokes": True},
                {"name": "front wheel", "shape": "circle", "r": 0.22, "color": "#333333", "spokes": True},
            ],
        }

    def frame(self):
        X, Y = self.pos * self.SX, self._h(self.pos)
        phi = self._slope()
        tx, ty = math.cos(phi), math.sin(phi)
        nx, ny = -ty, tx
        roll = -X / 0.22
        out = [X + 0.6 * nx, Y + 0.6 * ny, phi]
        for s in (-0.42, 0.42):
            out += [X + s * tx + 0.24 * nx, Y + s * ty + 0.24 * ny, roll]
        return out

    def hud(self):
        return f"{self.t:.1f} s · height {self._h(self.pos):+.1f} m"

    def human_action(self, keys):
        k = keys or {}
        return [float(bool(k.get("ArrowRight"))) - float(bool(k.get("ArrowLeft")))]


# ---------------------------------------------------------------------------
# Pendulum: swing a weak motorised pendulum up and keep it standing
# ---------------------------------------------------------------------------


class PendulumEnv(Env):
    KIND = "pendulum"
    TITLE = "pendulum"
    DT = 0.05
    DEFAULT_TIME = 10.0          # 200 steps
    DEFAULT_GRAVITY = 10.0
    REWARD_KINDS = {
        "swing": "the classic swing-up score: −(angle² + 0.1·spin² + 0.001·torque²), every step",
        "upright": "1 when the pendulum points straight up, −1 hanging down, every step",
        "energy": "how hard the motor works (throttle²), every step",
        "speed": "how fast the tip moves (m/s), every step",
        "time": "1 every step",
    }
    EVENT_KINDS = ()
    DEFAULT_REWARDS = (("swing", 1.0, False),)
    KEYS = {"ArrowLeft": "turn anticlockwise", "ArrowRight": "turn clockwise"}
    HINTS = {"ppo_rollout": 2048, "ppo_epochs": 10, "ppo_minibatch": 256}
    MAX_SPEED, MAX_TORQUE, LENGTH, MASS = 8.0, 2.0, 1.0, 1.0

    def _setup(self):
        self.obs_names = ["angle cos", "angle sin", "spin"]
        self.action_spec = ActionSpec("continuous", 1, ["torque"], [[-1.0, 0.0, 1.0]])
        self.th, self.thdot = math.pi, 0.0
        self.u = 0.0

    def _reset(self, seed):
        r = _rng(seed)
        self.th = float(r.uniform(-math.pi, math.pi))
        self.thdot = float(r.uniform(-1.0, 1.0))
        self.u = 0.0

    def _step(self, a):
        u = a[0] * self.MAX_TORQUE
        self.u = u
        th, thdot = self.th, self.thdot
        norm = ((th + math.pi) % (2 * math.pi)) - math.pi
        cost = norm * norm + 0.1 * thdot * thdot + 0.001 * u * u
        g, l, m = self.gravity, self.LENGTH, self.MASS
        thdot = thdot + (3 * g / (2 * l) * math.sin(th) + 3.0 / (m * l * l) * u) * self.DT
        thdot = min(max(thdot, -self.MAX_SPEED), self.MAX_SPEED)
        self.th = th + thdot * self.DT
        self.thdot = thdot
        return {"swing": -cost, "upright": math.cos(self.th), "energy": a[0] * a[0],
                "speed": abs(thdot) * l, "time": 1.0}, False

    def observe(self):
        return np.asarray([math.cos(self.th), math.sin(self.th), self.thdot / self.MAX_SPEED], dtype=np.float32)

    def state(self):
        norm = ((self.th + math.pi) % (2 * math.pi)) - math.pi
        st = self.base_state()
        l = self.LENGTH
        st.update({"distance": 0.0, "speed": abs(self.thdot) * l, "tilt": math.degrees(norm),
                   "spin": math.degrees(self.thdot), "height": l * math.cos(self.th), "x": -l * math.sin(self.th),
                   "y": l * math.cos(self.th), "touching_ground": 0, "angle": math.degrees(norm),
                   "torque": self.u})
        return st

    def progress(self):
        return float(math.cos(self.th))

    def episode_info(self):
        info = super().episode_info()
        info["distance"] = 0.0
        return info

    def scene(self):
        l = self.LENGTH
        return {
            "bounds": [-1.5, -1.5, 1.5, 1.5],
            "camera": {"mode": "fixed"},
            "sky": "#f4f6fb",
            "static": [
                {"type": "line", "points": [[-0.4, 0.0], [0.4, 0.0]], "color": "#bbbbbb", "width": 0.02},
                {"type": "circle", "x": 0.0, "y": 0.0, "r": 0.07, "color": "#333333"},
            ],
            "bodies": [
                {"name": "rod", "shape": "rect", "w": 0.1, "h": l, "color": "#e67e22"},
                {"name": "weight", "shape": "circle", "r": 0.13, "color": "#c0392b"},
            ],
        }

    def frame(self):
        l = self.LENGTH
        tx, ty = -l * math.sin(self.th), l * math.cos(self.th)
        return [tx / 2, ty / 2, self.th, tx, ty, self.th]

    def hud(self):
        norm = ((self.th + math.pi) % (2 * math.pi)) - math.pi
        return f"{self.t:.1f} s · angle {math.degrees(norm):+.0f}° · torque {self.u:+.1f}"

    def human_action(self, keys):
        k = keys or {}
        return [float(bool(k.get("ArrowLeft"))) - float(bool(k.get("ArrowRight")))]
