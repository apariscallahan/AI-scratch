"""World parts: the small setting blocks that snap inside a 'create world' block.

Every part is a tiny object that only stores its settings (and the id of the block
it came from, so errors can point at it). The world decides what each part means.
"""
from __future__ import annotations

import math

from ..errors import NBError

__all__ = [
    "Part", "Terrain", "SameWorld", "CarBody", "Drive", "Brake", "Lean", "Jump", "Boost", "Sense", "Reward",
    "CustomReward", "EndAfter", "EndIfStuck", "EndWhen", "Gravity", "MazeLayout", "MazeSize", "MazeWalls",
    "MazeTraps", "Wind", "Fuel", "Gap", "PipeSpeed", "PoleLength",
    "TERRAINS", "SENSORS", "REWARD_KINDS",
]

TERRAINS = ("flat", "hills", "rough", "steps", "mixed")
SENSORS = ("speed", "tilt", "spin", "ground", "wheels", "height", "progress", "fuel")
# Reward kinds offered by the blocks (worlds may add a few of their own).
REWARD_KINDS = ("distance", "speed", "upright", "finish", "flip", "energy", "time")


def num(value, what: str, bid=None, *, lo=None, hi=None, integer=False):
    """Read a number from a block slot (numbers, numeric text, booleans)."""
    if isinstance(value, bool):
        value = int(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise NBError(f"{what} must be a number (got {value!r}).", block_id=bid) from None
    if not math.isfinite(v):
        raise NBError(f"{what} must be a normal number (got {value!r}).", block_id=bid)
    if lo is not None and v < lo:
        raise NBError(f"{what} must be at least {lo:g} (got {v:g}).", block_id=bid)
    if hi is not None and v > hi:
        raise NBError(f"{what} must be at most {hi:g} (got {v:g}).", block_id=bid)
    return int(round(v)) if integer else v


def flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "same")
    return bool(value)


def word(value) -> str:
    return str(value if value is not None else "").strip().lower().replace("-", "_").replace(" ", "_")


class Part:
    """Base class of every world part."""

    label = "part"

    def __init__(self, _bid=None):
        self.bid = _bid

    def describe(self) -> str:
        return self.label

    def __repr__(self):
        return f"<{type(self).__name__} {self.describe()}>"


# ---------------------------------------------------------------------------
# Car world
# ---------------------------------------------------------------------------


class Terrain(Part):
    """The ground the car drives on: flat, hills, rough, steps or mixed."""

    label = "terrain"

    def __init__(self, kind="rough", bumpiness=0.5, hills=1.0, length=150, _bid=None):
        super().__init__(_bid)
        self.kind = word(kind) or "rough"
        if self.kind not in TERRAINS:
            raise NBError(f"Unknown terrain '{kind}'.", hint=f"Choose one of: {', '.join(TERRAINS)}.", block_id=_bid)
        self.bumpiness = num(bumpiness, "Bumpiness", _bid, lo=0, hi=3)
        self.hills = num(hills, "Hill size", _bid, lo=0, hi=4)
        self.length = num(length, "Terrain length", _bid, lo=10, hi=5000)

    def describe(self):
        return f"{self.kind} terrain, {self.length:g} m"


class SameWorld(Part):
    """True: the same terrain/maze/pipes every episode. False: a new random one each episode."""

    label = "same world every time"

    def __init__(self, same=True, _bid=None):
        super().__init__(_bid)
        self.same = flag(same)

    def describe(self):
        return "same world every episode" if self.same else "new random world every episode"


class CarBody(Part):
    label = "car body"

    def __init__(self, length=2.0, wheel_size=0.45, weight=1.0, _bid=None):
        super().__init__(_bid)
        self.length = num(length, "Car length", _bid, lo=0.8, hi=6)
        self.wheel_size = num(wheel_size, "Wheel size", _bid, lo=0.15, hi=1.5)
        self.weight = num(weight, "Car weight", _bid, lo=0.1, hi=10)

    def describe(self):
        return f"car {self.length:g} m long, wheels {self.wheel_size:g} m, weight {self.weight:g}"


class Drive(Part):
    """Motor on the wheels: the agent controls the throttle (forwards/backwards)."""

    label = "drive"

    def __init__(self, wheels="rear", power=1.0, _bid=None):
        super().__init__(_bid)
        self.wheels = word(wheels) or "rear"
        if self.wheels in ("back",):
            self.wheels = "rear"
        if self.wheels in ("both", "4wd", "awd"):
            self.wheels = "all"
        if self.wheels not in ("rear", "front", "all"):
            raise NBError(f"Unknown drive wheels '{wheels}'.", hint="Use rear, front or all.", block_id=_bid)
        self.power = num(power, "Drive power", _bid, lo=0.05, hi=10)

    def describe(self):
        return f"{self.wheels}-wheel drive, power {self.power:g}"


class Brake(Part):
    label = "brake"

    def __init__(self, strength=1.0, _bid=None):
        super().__init__(_bid)
        self.strength = num(strength, "Brake strength", _bid, lo=0, hi=10)


class Lean(Part):
    """Tilt the car forwards/backwards in the air or on the ground."""

    label = "lean"

    def __init__(self, strength=1.0, _bid=None):
        super().__init__(_bid)
        self.strength = num(strength, "Lean strength", _bid, lo=0, hi=10)


class Jump(Part):
    label = "jump"

    def __init__(self, strength=1.0, _bid=None):
        super().__init__(_bid)
        self.strength = num(strength, "Jump strength", _bid, lo=0, hi=10)


class Boost(Part):
    """A rocket on the back of the car. ``fuel`` is in seconds of thrust."""

    label = "rocket boost"

    def __init__(self, power=1.0, fuel=3.0, _bid=None):
        super().__init__(_bid)
        self.power = num(power, "Boost power", _bid, lo=0, hi=10)
        self.fuel = num(fuel, "Boost fuel", _bid, lo=0, hi=1000)


class Sense(Part):
    """A sensor: what the car can feel (becomes part of the network's input)."""

    label = "sense"

    def __init__(self, kind="speed", rays=5, reach=4.0, _bid=None):
        super().__init__(_bid)
        self.kind = word(kind)
        aliases = {"velocity": "speed", "angle": "tilt", "rotation": "spin", "radar": "ground", "rays": "ground",
                   "ground_ahead": "ground", "wheel": "wheels", "wheel_contact": "wheels", "altitude": "height"}
        self.kind = aliases.get(self.kind, self.kind)
        if self.kind not in SENSORS:
            raise NBError(f"Unknown sensor '{kind}'.", hint=f"Choose one of: {', '.join(SENSORS)}.", block_id=_bid)
        self.rays = num(rays, "Number of rays", _bid, lo=1, hi=32, integer=True)
        self.reach = num(reach, "Ray reach", _bid, lo=0.5, hi=50)

    def describe(self):
        if self.kind == "ground":
            return f"sense ground ({self.rays} rays, {self.reach:g} m)"
        return f"sense {self.kind}"


# ---------------------------------------------------------------------------
# Rewards & episode endings (every world)
# ---------------------------------------------------------------------------


class Reward(Part):
    """A reward (positive weight) or penalty (negative weight)."""

    label = "reward"

    def __init__(self, kind="distance", weight=1.0, end=False, _bid=None):
        super().__init__(_bid)
        self.kind = word(kind)
        aliases = {"progress": "distance", "forward": "distance", "goal": "finish", "land": "finish",
                   "landing": "finish", "crash": "flip", "fall": "flip", "trap": "flip", "alive": "time",
                   "survive": "time", "step": "time", "fuel": "energy", "balance": "upright", "pipes": "pipe"}
        self.kind = aliases.get(self.kind, self.kind)
        if not self.kind:
            raise NBError("A reward needs a kind (e.g. distance).", block_id=_bid)
        self.weight = num(weight, "Reward amount", _bid)
        self.end = flag(end)

    def describe(self):
        s = f"{'reward' if self.weight >= 0 else 'penalty'} {self.kind} × {self.weight:g}"
        return s + (" (ends the episode)" if self.end else "")


class CustomReward(Part):
    """``CustomReward(lambda s: ...)`` — ``s`` is the world's state dictionary."""

    label = "custom reward"

    def __init__(self, fn=None, _bid=None):
        super().__init__(_bid)
        if not callable(fn):
            raise NBError("A custom reward needs an expression to compute.", block_id=_bid)
        self.fn = fn


class EndAfter(Part):
    label = "end after"

    def __init__(self, seconds=20, _bid=None):
        super().__init__(_bid)
        self.seconds = num(seconds, "Episode length (seconds)", _bid, lo=0.1, hi=36000)

    def describe(self):
        return f"end after {self.seconds:g} s"


class EndIfStuck(Part):
    label = "end if stuck"

    def __init__(self, seconds=3, _bid=None):
        super().__init__(_bid)
        self.seconds = num(seconds, "Stuck time (seconds)", _bid, lo=0.2, hi=3600)

    def describe(self):
        return f"end if stuck for {self.seconds:g} s"


class EndWhen(Part):
    """``EndWhen(lambda s: ...)``: the episode ends as soon as this is true."""

    label = "end when"

    def __init__(self, fn=None, _bid=None):
        super().__init__(_bid)
        if not callable(fn):
            raise NBError("'end when' needs a condition.", block_id=_bid)
        self.fn = fn


class Gravity(Part):
    label = "gravity"

    def __init__(self, g=9.8, _bid=None):
        super().__init__(_bid)
        self.g = num(g, "Gravity", _bid, lo=0, hi=100)

    def describe(self):
        return f"gravity {self.g:g} m/s²"


# ---------------------------------------------------------------------------
# Maze
# ---------------------------------------------------------------------------


class MazeLayout(Part):
    """ASCII maze: ``#`` wall, ``S`` start, ``G`` goal, ``T`` trap, ``.`` floor; rows split by newlines or ``/``."""

    label = "maze layout"

    def __init__(self, text="", _bid=None):
        super().__init__(_bid)
        self.text = str(text or "")
        if not self.text.strip():
            raise NBError("The maze layout is empty.", hint="Draw it with # for walls, S start, G goal, . floor.",
                          block_id=_bid)


class MazeSize(Part):
    label = "maze size"

    def __init__(self, width=8, height=8, _bid=None):
        super().__init__(_bid)
        self.width = num(width, "Maze width", _bid, lo=2, hi=60, integer=True)
        self.height = num(height, "Maze height", _bid, lo=2, hi=60, integer=True)


class MazeWalls(Part):
    label = "maze walls"

    def __init__(self, density=0.25, _bid=None):
        super().__init__(_bid)
        self.density = num(density, "Wall density", _bid, lo=0, hi=0.9)


class MazeTraps(Part):
    label = "maze traps"

    def __init__(self, count=2, _bid=None):
        super().__init__(_bid)
        self.count = num(count, "Number of traps", _bid, lo=0, hi=500, integer=True)


# ---------------------------------------------------------------------------
# Lander, flappy, cart-pole
# ---------------------------------------------------------------------------


class Wind(Part):
    label = "wind"

    def __init__(self, strength=1.0, _bid=None):
        super().__init__(_bid)
        self.strength = num(strength, "Wind strength", _bid, lo=0, hi=20)


class Fuel(Part):
    label = "fuel"

    def __init__(self, amount=10.0, _bid=None):
        super().__init__(_bid)
        self.amount = num(amount, "Fuel", _bid, lo=0, hi=10000)


class Gap(Part):
    label = "gap size"

    def __init__(self, size=3.0, _bid=None):
        super().__init__(_bid)
        self.size = num(size, "Gap size", _bid, lo=1.0, hi=9)


class PipeSpeed(Part):
    label = "pipe speed"

    def __init__(self, speed=3.5, _bid=None):
        super().__init__(_bid)
        self.speed = num(speed, "Pipe speed", _bid, lo=0.2, hi=30)


class PoleLength(Part):
    label = "pole length"

    def __init__(self, length=1.0, _bid=None):
        super().__init__(_bid)
        self.length = num(length, "Pole length", _bid, lo=0.1, hi=10)
