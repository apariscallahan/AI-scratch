"""The car world: a little car on 2-D terrain, simulated with pymunk.

* terrain: static segments on a regular 0.25 m grid (smoothed noise, flat start, finish flag, walls at both ends)
* car: chassis (base box + cabin), two wheels on sliding suspension (GrooveJoint + DampedSpring)
* abilities (actions, each -1..1): drive (SimpleMotor on the wheels), brake, lean (torque on the chassis),
  jump (velocity kick when a wheel touches the ground), boost (rocket thrust, limited fuel)
* sensors (observations): speed, tilt, spin, ground rays, wheel contact, height, progress, fuel
* the roof (or the side of the car) touching the ground = flipped (crash)

Physics runs at 60 Hz with 2 sub-steps per decision (30 decisions per second). The static terrain is
built once per layout; each episode only re-creates the car (with the same internal shape ids, so
an episode with the same seed and actions always plays out identically).
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from .env_base import ActionSpec, Env
from .parts import Boost, Brake, CarBody, Drive, Jump, Lean, Sense, Terrain

DX = 0.25                    # terrain grid spacing (m)
START_X = -4.0               # wall behind the start
RUNOFF = 20.0                # flat ground after the finish
PAD = 80.0                   # flat padding of the height lookup table (so ray look-ups never fall off)
# terrain shape: (hill height × 'hills', bump height × 'bumpiness', steepest slope) for each kind
ROUGH = (1.8, 1.2, 1.2)
HILLY = (3.0, 0.08, 0.8)
CAT_GROUND = 0b01
CAT_CAR = 0b10
SENSOR_ORDER = ("speed", "tilt", "spin", "ground", "wheels", "height", "progress", "fuel")
ABILITY_ORDER = ("drive", "brake", "lean", "jump", "boost")
CT_GROUND = 1
CT_PARTS = (2, 3, 4, 5)      # contact slots: rear wheel, front wheel, chassis base, cabin (roof)


def _pymunk():
    try:
        import pymunk
    except ImportError as e:  # pragma: no cover
        from ..errors import missing_package
        raise missing_package("pymunk", "The car world") from e
    return pymunk


# ---------------------------------------------------------------------------
# Terrain generation
# ---------------------------------------------------------------------------


def _catmull(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Smooth curve through random control values (Catmull-Rom); ``u`` is in control-point units."""
    i = np.floor(u).astype(int) + 1
    t = u - np.floor(u)
    p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)


def _noise(rng, xs, wavelength, amp):
    n = int((xs[-1] - xs[0]) / wavelength) + 6
    ctrl = rng.uniform(-1.0, 1.0, n)
    return amp * _catmull(ctrl, (xs - xs[0]) / wavelength)


def _hills(rng, xs, amp):
    y = np.zeros_like(xs)
    if amp <= 0:
        return y
    for wl, a in ((rng.uniform(34, 46), 1.0), (rng.uniform(17, 24), 0.55), (rng.uniform(9, 13), 0.25)):
        y += a * np.sin(2 * np.pi * xs / wl + rng.uniform(0, 2 * np.pi))
    return y * (amp / 1.8)


def _bumps(rng, xs, amp):
    if amp <= 0:
        return np.zeros_like(xs)
    return _noise(rng, xs, rng.uniform(2.0, 2.6), amp) + _noise(rng, xs, 1.1, amp * 0.3)


def _steps(rng, xs, height):
    y = np.zeros_like(xs)
    if height <= 0:
        return y
    x = xs[0]
    level = 0.0
    while x < xs[-1]:
        x += rng.uniform(2.5, 6.0)
        h = rng.uniform(0.35, 1.0) * height
        up = rng.random() < (0.75 if level <= 0 else 0.25)   # mostly alternate so the ground doesn't drift
        level += h if up else -h
        y[xs >= x] = level
    return y


def _limit_slope(y: np.ndarray, max_slope: float) -> np.ndarray:
    out = y.tolist()
    d = max_slope * DX
    for i in range(1, len(out)):
        diff = out[i] - out[i - 1]
        if diff > d:
            out[i] = out[i - 1] + d
        elif diff < -d:
            out[i] = out[i - 1] - d
    return np.asarray(out)


class TerrainData:
    """Heights on a regular grid plus quick look-ups."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, finish: float):
        self.xs = xs
        self.ys = ys
        self.finish = finish
        self.ymin = float(ys.min())
        self.ymax = float(ys.max())
        npad = int(PAD / DX)
        padded = [float(ys[0])] * npad + ys.tolist() + [float(ys[-1])] * (npad + 2)
        self.lut = padded
        self.lut_x0 = float(xs[0]) - npad * DX
        self.inv_dx = 1.0 / DX
        self.lut_n = len(padded)

    def height(self, x: float) -> float:
        u = (x - self.lut_x0) * self.inv_dx
        i = int(u)
        if i < 0:
            return self.lut[0]
        if i >= self.lut_n - 1:
            return self.lut[-1]
        y0 = self.lut[i]
        return y0 + (self.lut[i + 1] - y0) * (u - i)


_TERRAIN_CACHE: "OrderedDict[tuple, TerrainData]" = OrderedDict()


def make_terrain(kind: str, bumpiness: float, hills: float, length: float, seed: int) -> TerrainData:
    key = (kind, round(bumpiness, 6), round(hills, 6), round(length, 6), int(seed))
    td = _TERRAIN_CACHE.get(key)
    if td is not None:
        _TERRAIN_CACHE.move_to_end(key)
        return td
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 7])
    end = length + RUNOFF
    xs = np.arange(START_X - 2.0, end + 2.0 + 1e-9, DX)

    def rough():
        return _limit_slope(_hills(rng, xs, ROUGH[0] * hills) + _bumps(rng, xs, ROUGH[1] * bumpiness), ROUGH[2])

    def hilly():
        return _limit_slope(_hills(rng, xs, HILLY[0] * hills) + _bumps(rng, xs, HILLY[1] * bumpiness), HILLY[2])

    def stepped():
        return _limit_slope(_hills(rng, xs, 0.8 * hills), 0.5) + _steps(rng, xs, 0.12 + 0.36 * bumpiness)

    if kind == "flat":
        y = np.zeros_like(xs)
    elif kind == "hills":
        y = hilly()
    elif kind == "rough":
        y = rough()
    elif kind == "steps":
        y = stepped()
    else:  # mixed: sections of different kinds, blended into each other
        parts = {"hills": hilly(), "rough": rough(), "steps": stepped(), "flat": np.zeros_like(xs)}
        order = ["rough", "hills", "steps", "rough", "flat", "hills", "steps"]
        rng.shuffle(order)
        sec = 30.0
        weights = []
        for j, name in enumerate(order * (int(end / (sec * len(order))) + 2)):
            c = j * sec + sec / 2
            weights.append((name, np.clip(1.0 - np.abs(xs - c) / (sec * 0.6), 0.0, 1.0)))
        tot = sum(w for _, w in weights) + 1e-9
        y = sum(parts[name] * w for name, w in weights) / tot
    # flat start, difficulty ramps up over the first ~25 m, flat run-off after the finish
    ramp = np.clip((xs - 4.0) / 25.0, 0.0, 1.0)
    ramp = ramp * ramp * (3 - 2 * ramp)
    y = (y - y[np.searchsorted(xs, 4.0)]) * ramp
    fin = int(np.searchsorted(xs, length))
    y[fin:] = y[fin]
    td = TerrainData(xs, y, float(length))
    _TERRAIN_CACHE[key] = td
    while len(_TERRAIN_CACHE) > 48:
        _TERRAIN_CACHE.popitem(last=False)
    return td


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


class CarEnv(Env):
    KIND = "car"
    TITLE = "car on terrain"
    DT = 1 / 30
    SUBSTEPS = 2
    DEFAULT_TIME = 20.0
    SUPPORTS_STUCK = True
    DEFAULT_STUCK = 3.0
    HANDLES = ("Terrain", "CarBody", "Drive", "Brake", "Lean", "Jump", "Boost", "Sense")
    REWARD_KINDS = {
        "distance": "metres driven forwards (backwards counts negative)",
        "speed": "forward speed (m/s), every step",
        "upright": "1 when the car is level, −1 upside down, every step",
        "finish": "once, when the car crosses the finish line",
        "flip": "once, when the car crashes onto its roof or side",
        "energy": "how hard the abilities are used, every step",
        "time": "1 every step",
    }
    EVENT_KINDS = ("finish", "flip")
    DEFAULT_REWARDS = (("distance", 1.0, False), ("flip", -10.0, True))
    PROGRESS_UNIT = " m"
    KEYS = {"ArrowRight": "drive forwards", "ArrowLeft": "drive backwards", "ArrowUp": "lean back (nose up)",
            "ArrowDown": "lean forwards (nose down)", " ": "jump", "Shift": "rocket boost", "b": "brake"}
    HINTS = {"ppo_rollout": 2048, "ppo_epochs": 10, "ppo_minibatch": 256}

    # ---- configuration ------------------------------------------------------
    @classmethod
    def configure(cls, cfg, parts):
        from .world import warn_part
        cfg.terrain = Terrain()
        cfg.car = CarBody()
        cfg.abilities = {}
        sensors = {}
        unused = []
        seen_terrain = seen_body = False
        names = {Drive: "drive", Brake: "brake", Lean: "lean", Jump: "jump", Boost: "boost"}
        for p in parts:
            if isinstance(p, Terrain):
                if seen_terrain:
                    warn_part(p, "There are two terrain blocks; the last one is used.")
                cfg.terrain, seen_terrain = p, True
            elif isinstance(p, CarBody):
                if seen_body:
                    warn_part(p, "There are two car body blocks; the last one is used.")
                cfg.car, seen_body = p, True
            elif type(p) in names:
                name = names[type(p)]
                if name in cfg.abilities:
                    warn_part(p, f"The car already has '{name}'; the last block is used.")
                cfg.abilities[name] = p
            elif isinstance(p, Sense):
                if p.kind in sensors:
                    warn_part(p, f"The car already senses '{p.kind}'; the last block is used.")
                sensors[p.kind] = p
            else:
                unused.append(p)
        if not cfg.abilities:
            cfg.abilities["drive"] = Drive("rear")
            cfg.default_notes.append("The car has no abilities, so it gets 'drive (rear wheels)'.")
        if not sensors:
            sensors = {k: Sense(k) for k in ("speed", "tilt", "spin", "ground", "wheels")}
            cfg.default_notes.append("The car has no sensors, so it senses its speed, tilt and spin, the ground "
                                     "ahead (5 rays) and whether its wheels touch the ground.")
        if "fuel" in sensors and "boost" not in cfg.abilities:
            warn_part(sensors["fuel"], "The car has no rocket boost, so its fuel sensor always reads 0.")
        cfg.sensors = [(k, sensors[k]) for k in SENSOR_ORDER if k in sensors]
        return unused

    # ---- setup ----------------------------------------------------------------
    def _setup(self):
        cfg = self.cfg
        car = cfg.car
        self.length = float(cfg.terrain.length)
        self.L = float(car.length)
        self.r = float(car.wheel_size)
        self.weight = float(car.weight)
        ab = cfg.abilities
        self.abil = [k for k in ABILITY_ORDER if k in ab]
        dims = [[-1.0, 0.0, 1.0] if k in ("drive", "lean") else [0.0, 1.0] for k in self.abil]
        self.action_spec = ActionSpec("continuous", len(self.abil), self.abil, dims)
        ai = {k: i for i, k in enumerate(self.abil)}
        self.i_drive, self.i_brake, self.i_lean = ai.get("drive"), ai.get("brake"), ai.get("lean")
        self.i_jump, self.i_boost = ai.get("jump"), ai.get("boost")
        drive = ab.get("drive")
        self.drive_wheels = drive.wheels if drive else None
        self.drive_power = drive.power if drive else 0.0
        self.brake_strength = ab["brake"].strength if "brake" in ab else 0.0
        self.lean_strength = ab["lean"].strength if "lean" in ab else 0.0
        self.jump_strength = ab["jump"].strength if "jump" in ab else 0.0
        self.boost_power = ab["boost"].power if "boost" in ab else 0.0
        self.fuel_max = float(ab["boost"].fuel) if "boost" in ab else 0.0
        self.has_boost = "boost" in ab
        # keyboard help for 'play it yourself': only the abilities this car has
        self.KEYS = {k: v for k, v in CarEnv.KEYS.items()
                     if {"ArrowRight": "drive", "ArrowLeft": "drive", "ArrowUp": "lean", "ArrowDown": "lean",
                         " ": "jump", "Shift": "boost", "b": "brake"}[k] in ab}
        # sensors
        obs_names = []
        angles = []
        self.ray_reach = 4.0
        for kind, s in cfg.sensors:
            if kind == "speed":
                obs_names += ["speed x", "speed y"]
            elif kind == "tilt":
                obs_names += ["tilt sin", "tilt cos"]
            elif kind == "spin":
                obs_names += ["spin"]
            elif kind == "ground":
                n = int(s.rays)
                self.ray_reach = float(s.reach)
                angles = [-60.0] if n == 1 else [-90.0 + 75.0 * i / (n - 1) for i in range(n)]
                obs_names += [f"ground ray {i + 1}" for i in range(n)]
            elif kind == "wheels":
                obs_names += ["rear wheel touching", "front wheel touching"]
            elif kind == "height":
                obs_names += ["height"]
            elif kind == "progress":
                obs_names += ["progress"]
            elif kind == "fuel":
                obs_names += ["fuel"]
        self.obs_names = obs_names
        self.ray_angles = angles
        self._ray_dirs = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angles]
        self._ray_step = self.ray_reach / 12.0
        self.sensor_kinds = [k for k, _ in cfg.sensors]
        # geometry & masses
        self.hb = 0.18                                   # half height of the base box
        self.cabin_h = 0.32
        self.wx = max(0.3, self.L / 2 - 0.25)            # wheel x offset from the centre
        self.rest_y = -0.43                              # wheel centre at rest (local)
        self.m_chassis = 2.5 * self.weight
        self.m_wheel = 0.35 * self.weight
        self.total_mass = self.m_chassis + 2 * self.m_wheel
        n_driven = {"rear": 1, "front": 1, "all": 2}.get(self.drive_wheels, 0)
        self.drive_torque = (7.0 if n_driven == 1 else 4.5) * self.drive_power * self.weight
        self.max_wheel_speed = 15.0 * math.sqrt(max(self.drive_power, 0.05)) / self.r
        self.stuck_limit = self.cfg.stuck
        self.space = None
        self.terrain = None
        self._layout_key = None
        self.chassis = None
        self.wheels = []
        self._contacts = [0, 0, 0, 0]
        self.fuel = self.fuel_max
        self.boost_level = 0.0

    # ---- world building ---------------------------------------------------------
    def _build_space(self, td: TerrainData):
        pm = _pymunk()
        space = pm.Space()
        space.gravity = (0.0, -self.gravity)
        space.iterations = 10
        space.damping = 0.97
        sb = space.static_body
        pts = list(zip(td.xs.tolist(), td.ys.tolist()))
        segs = [pm.Segment(sb, pts[i], pts[i + 1], 0.04) for i in range(len(pts) - 1)]
        y0 = td.height(START_X)
        segs.append(pm.Segment(sb, (START_X, y0 - 1.0), (START_X, y0 + 8.0), 0.1))
        xe = self.length + RUNOFF
        ye = td.height(xe)
        segs.append(pm.Segment(sb, (xe, ye - 1.0), (xe, ye + 8.0), 0.1))
        gfilter = pm.ShapeFilter(categories=CAT_GROUND)
        for s in segs:
            s.friction = 1.0
            s.collision_type = CT_GROUND
            s.filter = gfilter
        space.add(*segs)
        for slot, ct in enumerate(CT_PARTS):
            _on_collision(space, ct, CT_GROUND, *self._handlers(slot))
        self.space = space
        self._id_counter = _get_shape_counter(space)

    def _handlers(self, slot):
        env = self

        def begin(arbiter, space_, data):
            env._contacts[slot] += 1
            return True

        def separate(arbiter, space_, data):
            c = env._contacts
            if c[slot] > 0:
                c[slot] -= 1

        return begin, separate

    def _remove_car(self):
        if self.chassis is not None:
            self.space.remove(*self._car_objects)
            self.chassis = None

    def _add_car(self, x, y):
        pm = _pymunk()
        L, hb, r = self.L, self.hb, self.r
        filt = pm.ShapeFilter(group=1, categories=CAT_CAR, mask=CAT_GROUND)
        chassis = pm.Body(self.m_chassis, pm.moment_for_box(self.m_chassis, (L, 2 * hb + 0.15)))
        chassis.position = (x, y)
        base = pm.Poly(chassis, [(-L / 2, -hb), (L / 2, -hb), (L / 2, hb), (-L / 2, hb)], radius=0.02)
        cw0, cw1 = -0.32 * L, 0.2 * L
        cabin = pm.Poly(chassis, [(cw0 - 0.06 * L, hb), (cw1 + 0.08 * L, hb), (cw1, hb + self.cabin_h),
                                  (cw0, hb + self.cabin_h)], radius=0.02)
        for s, ct in ((base, CT_PARTS[2]), (cabin, CT_PARTS[3])):
            s.friction = 0.6
            s.filter = filt
            s.collision_type = ct
        objs = [chassis, base, cabin]
        wheels, motors, brakes = [], [], []
        k = 100.0 * self.weight
        c = 12.0 * self.weight
        for idx, wx in enumerate((-self.wx, self.wx)):  # 0 = rear, 1 = front
            w = pm.Body(self.m_wheel, pm.moment_for_circle(self.m_wheel, 0, r))
            w.position = (x + wx, y + self.rest_y)
            ws = pm.Circle(w, r)
            ws.friction = 1.4
            ws.filter = filt
            ws.collision_type = CT_PARTS[idx]
            groove = pm.GrooveJoint(chassis, w, (wx, -0.12), (wx, -0.78), (0, 0))
            spring = pm.DampedSpring(chassis, w, (wx, 0.0), (0, 0), 0.55, k, c)
            objs += [w, ws, groove, spring]
            if self.i_drive is not None:
                motor = pm.SimpleMotor(chassis, w, 0.0)
                motor.max_force = 0.0
                objs.append(motor)
                motors.append(motor)
            if self.i_brake is not None:
                brake = pm.SimpleMotor(chassis, w, 0.0)
                brake.max_force = 0.0
                objs.append(brake)
                brakes.append(brake)
            wheels.append(w)
        self.space.add(*objs)
        self._car_objects = objs
        self.chassis = chassis
        self.wheels = wheels
        drive_idx = {"rear": (0,), "front": (1,), "all": (0, 1)}.get(self.drive_wheels, ())
        self.driven = [motors[i] for i in drive_idx] if motors else []
        self.brakes = brakes
        self._drive_set = None
        self._brake_set = None

    def _reset(self, seed):
        t = self.cfg.terrain
        lseed = self.cfg.seed if self.cfg.same_world else seed
        key = (t.kind, t.bumpiness, t.hills, t.length, lseed)
        if self.space is not None:
            self._remove_car()
        if self.space is None or key != self._layout_key:
            self.terrain = make_terrain(t.kind, t.bumpiness, t.hills, t.length, lseed)
            self._build_space(self.terrain)
            self._layout_key = key
        else:
            _set_shape_counter(self.space, self._id_counter)
        self._contacts = [0, 0, 0, 0]
        td = self.terrain
        y = max(td.height(-self.wx - 0.5), td.height(0.0), td.height(self.wx + 0.5)) - self.rest_y + self.r + 0.02
        self._add_car(0.0, y)
        self.fuel = self.fuel_max
        self.boost_level = 0.0
        self.jump_cool = 0.0
        self.prev_x = 0.0
        self.best_x = 0.0
        self.stuck_x = 0.0
        self.stuck_t = 0.0
        self.flipped = False
        self.flips = 0
        self.finished = False
        self._read()

    # ---- stepping -----------------------------------------------------------------
    def _read(self):
        c = self.chassis
        p = c.position
        v = c.velocity
        self.x, self.y = p.x, p.y
        self.vx, self.vy = v.x, v.y
        self.angle = c.angle
        self.omega = c.angular_velocity

    def _step(self, a):
        energy = 0.0
        if self.i_drive is not None:
            thr = a[self.i_drive]
            if thr < 0.05 and thr > -0.05:
                setting = (0.0, 0.0)
            else:
                energy += abs(thr)
                setting = (self.max_wheel_speed if thr > 0 else -0.6 * self.max_wheel_speed,
                           self.drive_torque * abs(thr))
            if setting != self._drive_set:
                for m in self.driven:
                    m.rate, m.max_force = setting
                self._drive_set = setting
        if self.i_brake is not None:
            b = a[self.i_brake]
            f = 12.0 * self.brake_strength * self.weight * b if b > 0 else 0.0
            energy += max(b, 0.0)
            if f != self._brake_set:
                for m in self.brakes:
                    m.max_force = f
                self._brake_set = f
        lean_torque = 0.0
        if self.i_lean is not None:
            ln = a[self.i_lean]
            energy += abs(ln)
            lean_torque = 6.0 * self.lean_strength * self.weight * ln
        if self.i_jump is not None:
            self.jump_cool = max(0.0, self.jump_cool - self.DT)
            if a[self.i_jump] > 0.5 and self.jump_cool <= 0.0 and (self._contacts[0] or self._contacts[1]):
                dv = 5.0 * self.jump_strength
                upx, upy = -math.sin(self.angle) * dv, math.cos(self.angle) * dv
                for body in [self.chassis] + self.wheels:
                    v = body.velocity
                    body.velocity = (v.x + upx, v.y + upy)
                self.jump_cool = 0.8
                energy += 1.0
        thrust = 0.0
        self.boost_level = 0.0
        if self.has_boost:
            bl = a[self.i_boost]
            if bl > 0.0 and self.fuel > 0.0:
                bl = min(bl, self.fuel / self.DT, 1.0)
                self.fuel = max(0.0, self.fuel - bl * self.DT)
                self.boost_level = bl
                thrust = 11.0 * self.boost_power * self.total_mass * bl
                energy += bl
        chassis = self.chassis
        h = self.DT / self.SUBSTEPS
        space = self.space
        for _ in range(self.SUBSTEPS):
            if lean_torque:
                chassis.torque = lean_torque
            if thrust:
                chassis.apply_force_at_local_point((thrust, 0.0), (0.0, 0.0))
            space.step(h)
        self._read()
        x = self.x
        terms = {"time": 1.0, "energy": energy, "distance": x - self.prev_x, "speed": self.vx,
                 "upright": math.cos(self.angle)}
        self.prev_x = x
        if x > self.best_x:
            self.best_x = x
        over = False
        ct = self._contacts
        flipped = bool(ct[3]) or (bool(ct[2]) and abs(math.remainder(self.angle, 2 * math.pi)) > 1.13)
        if flipped and not self.flipped:
            terms["flip"] = 1.0
            self.flips += 1
        self.flipped = flipped
        if x >= self.length:
            terms["finish"] = 1.0
            self.finished = True
            self.end_reason = "finished"
            over = True
        if self.stuck_limit:
            if x > self.stuck_x + 0.5:
                self.stuck_x = x
                self.stuck_t = 0.0
            else:
                self.stuck_t += self.DT
                if self.stuck_t >= self.stuck_limit and not over:
                    over = True
                    self.end_reason = "crashed" if flipped else "stuck"
        if self.y < self.terrain.ymin - 30.0:
            over = True
            self.end_reason = "fell out of the world"
        return terms, over

    def observe(self):
        out = []
        for kind in self.sensor_kinds:
            if kind == "speed":
                out += (self.vx * 0.1, self.vy * 0.1)
            elif kind == "tilt":
                out += (math.sin(self.angle), math.cos(self.angle))
            elif kind == "spin":
                out.append(self.omega * 0.2)
            elif kind == "ground":
                out += self._rays()
            elif kind == "wheels":
                out += (1.0 if self._contacts[0] else 0.0, 1.0 if self._contacts[1] else 0.0)
            elif kind == "height":
                out.append((self.y - self.terrain.height(self.x)) * 0.2)
            elif kind == "progress":
                out.append(self.x / self.length)
            elif kind == "fuel":
                out.append(self.fuel / self.fuel_max if self.fuel_max > 0 else 0.0)
        return np.asarray(out, dtype=np.float32)

    def _rays(self):
        """Distance to the ground along rays fanned out below/ahead of the car (÷ reach; 1 = nothing in reach)."""
        td = self.terrain
        lut, x0, inv = td.lut, td.lut_x0, td.inv_dx
        step = self._ray_step
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        px, py = self.x, self.y
        u = (px - x0) * inv
        i = int(u)
        try:
            g0 = py - (lut[i] + (lut[i + 1] - lut[i]) * (u - i))
        except IndexError:
            return [1.0] * len(self._ray_dirs)
        if g0 <= 0.0:
            return [0.0] * len(self._ray_dirs)
        res = []
        try:
            for lx, ly in self._ray_dirs:
                dx = (lx * ca - ly * sa) * step
                dy = (lx * sa + ly * ca) * step
                prev_g = g0
                hit = 1.0
                x, y = px, py
                for k in range(1, 13):
                    x += dx
                    y += dy
                    u = (x - x0) * inv
                    i = int(u)
                    yl = lut[i]
                    g = y - (yl + (lut[i + 1] - yl) * (u - i))
                    if g <= 0.0:
                        hit = (k - 1 + prev_g / (prev_g - g)) / 12.0
                        break
                    prev_g = g
                res.append(hit)
        except IndexError:
            res += [1.0] * (len(self._ray_dirs) - len(res))
        return res

    # ---- state & drawing ------------------------------------------------------------
    def state(self):
        st = self.base_state()
        ct = self._contacts
        st.update({
            "distance": self.x, "speed": self.vx, "tilt": math.degrees(math.remainder(self.angle, 2 * math.pi)),
            "spin": math.degrees(self.omega), "height": self.y - self.terrain.height(self.x), "x": self.x,
            "y": self.y, "fuel": self.fuel, "touching_ground": 1 if (ct[0] or ct[1]) else 0,
            "wheels_touching": int(bool(ct[0])) + int(bool(ct[1])), "best_distance": self.best_x,
            "flipped": 1 if self.flipped else 0, "finished": 1 if self.finished else 0, "vertical_speed": self.vy,
        })
        return st

    def progress(self):
        return float(self.best_x)

    def scene(self):
        td = self.terrain
        L, hb, ch = self.L, self.hb, self.cabin_h
        mask = (td.xs >= START_X - 1.0) & (td.xs <= self.length + RUNOFF + 1.0)
        pts = np.round(np.stack([td.xs[mask], td.ys[mask]], axis=1), 3).tolist()
        cw0, cw1 = -0.32 * L, 0.2 * L
        silhouette = [[-L / 2, -hb], [L / 2, -hb], [L / 2, hb * 0.55], [cw1 + 0.08 * L, hb], [cw1, hb + ch],
                      [cw0, hb + ch], [cw0 - 0.06 * L, hb], [-L / 2, hb]]
        y0, ye = td.height(START_X), td.height(self.length + RUNOFF)
        scene = {
            "bounds": [round(START_X - 2.0, 2), round(td.ymin - 3.0, 2), round(self.length + RUNOFF + 2.0, 2),
                       round(td.ymax + 8.0, 2)],
            "camera": {"mode": "follow", "width": 22},
            "sky": "#bfe6ff", "ground_fill": "#8d6e63",
            "finish_x": round(self.length, 3),
            "static": [
                {"type": "ground", "points": pts, "color": "#5aa83c"},
                {"type": "rect", "x": round(START_X - 0.4, 3), "y": round(y0 - 1.0, 3), "w": 0.4, "h": 6.0,
                 "color": "#6d4c41"},
                {"type": "rect", "x": round(self.length + RUNOFF, 3), "y": round(ye - 1.0, 3), "w": 0.4, "h": 6.0,
                 "color": "#6d4c41"},
                {"type": "flag", "x": round(self.length, 3), "y": round(td.height(self.length), 3), "label": "finish"},
            ],
            "bodies": [
                {"name": "chassis", "shape": "poly", "points": [[round(x, 3), round(y, 3)] for x, y in silhouette],
                 "color": "#e8463a"},
                {"name": "rear wheel", "shape": "circle", "r": round(self.r, 3), "color": "#333333", "spokes": True},
                {"name": "front wheel", "shape": "circle", "r": round(self.r, 3), "color": "#333333", "spokes": True},
            ],
        }
        if self.has_boost:
            scene["effects"] = [{"body": 0, "x": round(-L / 2, 3), "y": 0.0, "angle": 180, "size": 1.2,
                                 "kind": "flame"}]
        return scene

    def frame(self):
        w0, w1 = self.wheels
        p0, p1 = w0.position, w1.position
        return [self.x, self.y, self.angle, p0.x, p0.y, w0.angle, p1.x, p1.y, w1.angle]

    def fx(self):
        return [self.boost_level] if self.has_boost else None

    def hud(self):
        s = f"{self.x:.1f} m · {self.vx:.1f} m/s"
        if self.has_boost:
            s += f" · fuel {self.fuel:.1f} s"
        if self.finished:
            s += " · finished!"
        elif self.flipped:
            s += " · crashed"
        return s

    def human_action(self, keys):
        k = keys or {}

        def down(*names):
            return any(k.get(n) for n in names)

        a = [0.0] * self.action_spec.size
        if self.i_drive is not None:
            a[self.i_drive] = float(down("ArrowRight", "d", "D")) - float(down("ArrowLeft", "a", "A"))
        if self.i_lean is not None:
            a[self.i_lean] = float(down("ArrowUp", "w", "W")) - float(down("ArrowDown", "s", "S"))
        if self.i_jump is not None:
            a[self.i_jump] = float(down(" ", "Space", "Spacebar"))
        if self.i_boost is not None:
            a[self.i_boost] = float(down("Shift", "ShiftLeft", "ShiftRight", "x", "X"))
        if self.i_brake is not None:
            a[self.i_brake] = float(down("b", "B"))
        return a


# ---------------------------------------------------------------------------
# pymunk version differences
# ---------------------------------------------------------------------------


def _on_collision(space, a, b, begin, separate):
    if hasattr(space, "on_collision"):  # pymunk >= 7: callbacks return nothing
        def _begin(arbiter, space_, data):
            begin(arbiter, space_, data)

        space.on_collision(a, b, begin=_begin, separate=separate)
    else:  # pymunk 6: begin must return True to keep the collision
        h = space.add_collision_handler(a, b)
        h.begin = begin
        h.separate = separate


def _lib():
    try:
        from pymunk._chipmunk_cffi import lib
        return lib
    except Exception:  # noqa: BLE001
        return None


def _get_shape_counter(space):
    lib = _lib()
    if lib is None or not hasattr(lib, "cpSpaceGetShapeIDCounter"):
        return None
    try:
        return int(lib.cpSpaceGetShapeIDCounter(space._space))
    except Exception:  # noqa: BLE001
        return None


def _set_shape_counter(space, value):
    """Give the new car the same internal shape ids every episode, so episodes replay identically."""
    if value is None:
        return
    try:
        _lib().cpSpaceSetShapeIDCounter(space._space, value)
    except Exception:  # noqa: BLE001
        pass
