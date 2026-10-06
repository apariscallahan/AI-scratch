"""Rocket lander: fire the main engine and side thrusters to land gently on the pad.

Simple rigid-body physics (60 Hz, 2 sub-steps per decision). Touching the ground ends the episode:
slow and upright = landed (a success if both feet are on the pad); too fast, tilted or body first =
crashed. The default score is shaped like the classic lunar-lander game: points for getting closer to
the pad, slower and more upright (judged at the moment of touchdown), +100 for landing on the pad,
−100 for crashing and a small fuel cost.
"""
from __future__ import annotations

import math

import numpy as np

from .env_base import ActionSpec, Env
from .parts import Fuel, Wind

MAIN_ACC = 13.0      # m/s² at full throttle
SIDE_ACC = 2.0       # sideways m/s² from a side thruster
SIDE_SPIN = 3.0      # rad/s² from a side thruster
SPIN_DAMPING = 1.5   # the rocket's gyroscopes slow down spinning (1/s)
FEET = ((-0.75, -1.15), (0.75, -1.15))
CORNERS = ((-0.35, -0.8), (0.35, -0.8), (-0.35, 0.4), (0.35, 0.4), (0.0, 0.85))
GROUND_DX = 0.5
HALF_WIDTH = 16.0
PAD_HALF = 3.0
SHAPE_DIST = 2.0     # how strongly the landing score pulls towards the pad


class LanderEnv(Env):
    KIND = "lander"
    TITLE = "rocket lander"
    DT = 1 / 30
    SUBSTEPS = 2
    DEFAULT_TIME = 20.0
    DEFAULT_GRAVITY = 3.7
    HANDLES = ("Wind", "Fuel")
    REWARD_KINDS = {
        "approach": "the classic landing score: points for getting closer to the pad, slower and more upright",
        "finish": "once, when the rocket lands gently on the pad",
        "flip": "once, when the rocket crashes (or flies away)",
        "distance": "metres closer to the landing pad, every step",
        "speed": "how fast the rocket moves (m/s), every step",
        "upright": "1 when the rocket points straight up, every step",
        "energy": "engine power used (main + side thrusters), every step",
        "time": "1 every step",
    }
    DEFAULT_REWARDS = (("approach", 1.0, False), ("finish", 100.0, True), ("flip", -100.0, True),
                       ("energy", -0.05, False))
    KEYS = {"ArrowUp": "main engine", "ArrowLeft": "side thruster: push left", "ArrowRight": "side thruster: push right"}
    HINTS = {"ppo_rollout": 2048, "ppo_epochs": 10, "ppo_minibatch": 256}

    @classmethod
    def configure(cls, cfg, parts):
        cfg.wind = 0.0
        cfg.fuel = 10.0
        unused = []
        for p in parts:
            if isinstance(p, Wind):
                cfg.wind = p.strength
            elif isinstance(p, Fuel):
                cfg.fuel = p.amount
            else:
                unused.append(p)
        return unused

    def _setup(self):
        self.obs_names = ["pad dx", "pad dy", "speed x", "speed y", "tilt sin", "tilt cos", "spin", "fuel",
                          "altitude"]
        self.action_spec = ActionSpec("continuous", 2, ["main engine", "side thrusters"],
                                      [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        self.wind = float(self.cfg.wind)
        self.fuel_max = float(self.cfg.fuel)
        self._layout = None
        self.xs = None

    # -- terrain ---------------------------------------------------------------
    def _make_layout(self, seed):
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 23])
        xs = np.arange(-HALF_WIDTH - 2, HALF_WIDTH + 2 + 1e-9, GROUND_DX)
        n = len(xs)
        ctrl = rng.uniform(-1.0, 1.0, n // 6 + 4)
        ys = np.interp(xs, np.linspace(xs[0], xs[-1], len(ctrl)), ctrl) * 1.4
        ys += 0.3 * np.sin(xs * rng.uniform(0.6, 1.2) + rng.uniform(0, 6.28))
        pad_x = float(rng.uniform(-8.0, 8.0))
        pad_y = float(np.interp(pad_x, xs, ys))
        flat = np.abs(xs - pad_x) <= PAD_HALF + 0.5
        ys[flat] = pad_y
        self.xs, self.ys = xs, ys
        self.ylist = ys.tolist()
        self.pad_x, self.pad_y = pad_x, pad_y
        self._layout = seed

    def ground(self, x):
        u = (x - self.xs[0]) / GROUND_DX
        i = int(u)
        if i < 0:
            return self.ylist[0]
        if i >= len(self.ylist) - 1:
            return self.ylist[-1]
        return self.ylist[i] + (self.ylist[i + 1] - self.ylist[i]) * (u - i)

    # -- episode ----------------------------------------------------------------
    def _reset(self, seed):
        lseed = self.cfg.seed if self.cfg.same_world else seed
        if self._layout != lseed or self.xs is None:
            self._make_layout(lseed)
        r = np.random.default_rng([int(seed) & 0xFFFFFFFF, 29])
        self.x = float(np.clip(self.pad_x + r.uniform(-6.0, 6.0), -12.0, 12.0))
        self.y = 14.0 + max(self.ylist)
        self.vx = float(r.uniform(-1.0, 1.0))
        self.vy = float(r.uniform(-1.0, 0.0))
        self.angle = float(r.uniform(-0.1, 0.1))
        self.omega = 0.0
        self.fuel = self.fuel_max
        self.wind_phase = r.uniform(0, 6.28, 2)
        self.main = self.left = self.right = 0.0
        self.landed = self.crashed = False
        self.impact_speed = None
        self.prev_dist = self._dist()
        self.prev_shape = self._shaping()

    def _dist(self):
        return math.hypot(self.x - self.pad_x, self.y - 1.15 - self.pad_y)

    def _shaping(self, speed=None):
        if speed is None:
            speed = math.hypot(self.vx, self.vy)
        return -(SHAPE_DIST * self._dist() + 2.0 * speed + 3.0 * abs(self.angle))

    def _to_world(self, lx, ly):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return self.x + lx * c - ly * s, self.y + lx * s + ly * c

    def _step(self, a):
        main = max(0.0, a[0])
        side = a[1] if abs(a[1]) > 0.25 else 0.0
        if self.fuel <= 0:
            main = side = 0.0
        burn = (main + 0.2 * abs(side)) * self.DT
        if burn > 0:
            self.fuel = max(0.0, self.fuel - burn)
        self.main, self.left, self.right = main, max(0.0, side), max(0.0, -side)
        h = self.DT / self.SUBSTEPS
        over = False
        terms = {"time": 1.0, "energy": main + abs(side)}
        for _ in range(self.SUBSTEPS):
            c, s = math.cos(self.angle), math.sin(self.angle)
            main_acc = MAIN_ACC
            ax = -s * main_acc * main + c * SIDE_ACC * side
            ay = c * main_acc * main + s * SIDE_ACC * side - self.gravity
            if self.wind:
                t = self.t
                ax += self.wind * 1.2 * (0.8 * math.sin(0.5 * t + self.wind_phase[0])
                                         + 0.5 * math.sin(1.3 * t + self.wind_phase[1]))
            alpha = -SIDE_SPIN * side - SPIN_DAMPING * self.omega
            self.vx += ax * h
            self.vy += ay * h
            self.omega += alpha * h
            self.x += self.vx * h
            self.y += self.vy * h
            self.angle += self.omega * h
            # touchdown?
            feet = [self._to_world(*f) for f in FEET]
            corners = [self._to_world(*p) for p in CORNERS]
            foot_hit = any(fy <= self.ground(fx) for fx, fy in feet)
            body_hit = any(py <= self.ground(px) for px, py in corners)
            if foot_hit or body_hit:
                gentle = (self.vy > -2.5 and abs(self.vx) < 1.5 and abs(self.angle) < 0.35 and not body_hit)
                on_pad = all(abs(fx - self.pad_x) <= PAD_HALF for fx, _ in feet)
                self.impact_speed = math.hypot(self.vx, self.vy)   # scores use the speed it hit the ground with
                # settle on the ground
                lift = max(self.ground(fx) - fy for fx, fy in feet + corners)
                if lift > 0:
                    self.y += lift
                self.vx = self.vy = self.omega = 0.0
                if gentle and on_pad:
                    self.landed = self.finished = True
                    terms["finish"] = 1.0
                    self.end_reason = "landed on the pad"
                elif gentle:
                    self.landed = True
                    self.end_reason = "landed next to the pad"
                else:
                    self.crashed = True
                    terms["flip"] = 1.0
                    self.end_reason = "crashed"
                over = True
                break
        if not over and (abs(self.x) > HALF_WIDTH + 4 or self.y > 45):
            self.crashed = True
            terms["flip"] = 1.0
            self.end_reason = "flew away"
            over = True
        d = self._dist()
        terms["distance"] = self.prev_dist - d
        self.prev_dist = d
        speed = self.impact_speed if (self.landed or self.crashed) and self.impact_speed is not None \
            else math.hypot(self.vx, self.vy)
        shp = self._shaping(speed)
        terms["approach"] = shp - self.prev_shape
        self.prev_shape = shp
        terms["speed"] = speed
        terms["upright"] = math.cos(self.angle)
        return terms, over

    def _altitude(self):
        return min(fy - self.ground(fx) for fx, fy in (self._to_world(*f) for f in FEET))

    def observe(self):
        return np.asarray([(self.x - self.pad_x) / 10.0, (self.y - self.pad_y) / 10.0, self.vx / 5.0, self.vy / 5.0,
                           math.sin(self.angle), math.cos(self.angle), self.omega / 2.0,
                           self.fuel / self.fuel_max if self.fuel_max > 0 else 0.0, self._altitude() / 10.0],
                          dtype=np.float32)

    def state(self):
        st = self.base_state()
        st.update({"distance": self._dist(), "speed": math.hypot(self.vx, self.vy), "tilt": math.degrees(self.angle),
                   "spin": math.degrees(self.omega), "height": self._altitude(), "x": self.x, "y": self.y,
                   "fuel": self.fuel, "touching_ground": 1 if (self.landed or self.crashed) else 0,
                   "landed": 1 if self.landed else 0, "crashed": 1 if self.crashed else 0,
                   "vertical_speed": self.vy, "pad_x": self.pad_x})
        return st

    def progress(self):
        return float(self._dist())

    def episode_info(self):
        info = super().episode_info()
        info["distance"] = round(self._dist(), 3)
        return info

    def scene(self):
        pts = np.round(np.stack([self.xs, self.ys], axis=1), 3).tolist()
        px, py = self.pad_x, self.pad_y
        return {
            "bounds": [-HALF_WIDTH, round(float(self.ys.min()) - 2.0, 2), HALF_WIDTH,
                       round(float(self.ys.max()) + 19.0, 2)],
            "camera": {"mode": "fixed"},
            "sky": "#0b1d3a", "ground_fill": "#6d6875",
            "static": [
                {"type": "ground", "points": pts, "color": "#b5a8c0"},
                {"type": "rect", "x": round(px - PAD_HALF, 3), "y": round(py - 0.15, 3), "w": 2 * PAD_HALF, "h": 0.15,
                 "color": "#f1c40f"},
                {"type": "flag", "x": round(px - PAD_HALF, 3), "y": round(py, 3), "label": ""},
                {"type": "flag", "x": round(px + PAD_HALF, 3), "y": round(py, 3), "label": ""},
            ],
            "bodies": [
                {"name": "rocket", "shape": "poly",
                 "points": [[-0.35, -0.8], [0.35, -0.8], [0.35, 0.4], [0.0, 0.85], [-0.35, 0.4]], "color": "#ecf0f1"},
                {"name": "left leg", "shape": "rect", "w": 0.08, "h": 0.7, "color": "#95a5a6"},
                {"name": "right leg", "shape": "rect", "w": 0.08, "h": 0.7, "color": "#95a5a6"},
            ],
            "effects": [
                {"body": 0, "x": 0.0, "y": -0.8, "angle": -90, "size": 1.6, "kind": "flame"},
                {"body": 0, "x": -0.35, "y": 0.3, "angle": 180, "size": 0.6, "kind": "flame"},
                {"body": 0, "x": 0.35, "y": 0.3, "angle": 0, "size": 0.6, "kind": "flame"},
            ],
        }

    def frame(self):
        out = [self.x, self.y, self.angle]
        for sx in (-1.0, 1.0):
            # leg from the body side (±0.32, -0.6) down to the foot (±0.75, -1.15)
            lx, ly = sx * 0.535, -0.875
            wx, wy = self._to_world(lx, ly)
            out += [wx, wy, self.angle + sx * 0.66]
        return out

    def fx(self):
        return [self.main, self.left, self.right]

    def hud(self):
        s = (f"altitude {self._altitude():.1f} m · speed {math.hypot(self.vx, self.vy):.1f} m/s · "
             f"fuel {100 * self.fuel / max(self.fuel_max, 1e-9):.0f}%")
        if self.finished:
            s += " · landed!"
        elif self.crashed:
            s += " · crashed"
        return s

    def human_action(self, keys):
        k = keys or {}
        main = 1.0 if (k.get("ArrowUp") or k.get(" ")) else -1.0
        side = float(bool(k.get("ArrowRight"))) - float(bool(k.get("ArrowLeft")))
        return [main, side]
