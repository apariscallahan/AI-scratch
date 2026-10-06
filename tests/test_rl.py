"""Tests for the reinforcement-learning runtime (neuroblocks.runtime.rl). Fast: quick mode everywhere."""
from __future__ import annotations

import json
import math
import os
import threading
import time

os.environ.setdefault("NEUROBLOCKS_EVENTS", "silent")
os.environ.setdefault("NEUROBLOCKS_NO_RUN_DIR", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import neuroblocks as nb  # noqa: E402
from neuroblocks.runtime.core import STATE  # noqa: E402
from neuroblocks.runtime.errors import NBError  # noqa: E402
from neuroblocks.runtime.events import emitter  # noqa: E402

rl = nb.rl
KINDS = ["car", "cartpole", "mountaincar", "pendulum", "lander", "maze", "flappy"]
STATIC_TYPES = {"ground": ("points",), "line": ("points",), "rect": ("x", "y", "w", "h"), "circle": ("x", "y", "r"),
                "flag": ("x", "y"), "text": ("x", "y", "text")}


@pytest.fixture(autouse=True)
def quick_and_quiet(tmp_path, monkeypatch):
    """Quick mode, a private NeuroBlocks home, and every event captured instead of printed."""
    monkeypatch.setenv("NEUROBLOCKS_HOME", str(tmp_path / "home"))
    old_quick = STATE.quick
    STATE.quick = True
    STATE.skip.clear()
    em = emitter()
    events = []

    def capture(type, **payload):
        ev = {"type": type, **payload}
        events.append(ev)
        return ev

    monkeypatch.setattr(em, "emit", capture)
    yield events
    STATE.quick = old_quick


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_replay(rep: dict):
    """Validate a sim_replay payload against the schema shared with the browser renderer."""
    for key in ("id", "world", "kind", "title", "dt", "scene", "agents", "frames", "stats"):
        assert key in rep, f"replay misses {key}"
    assert "fx_bodies" not in rep
    assert isinstance(rep["id"], str) and isinstance(rep["world"], str) and rep["kind"] in KINDS
    assert _num(rep["dt"]) and rep["dt"] > 0
    sc = rep["scene"]
    xmin, ymin, xmax, ymax = sc["bounds"]
    assert xmin < xmax and ymin < ymax
    assert sc["camera"]["mode"] in ("follow", "fixed")
    if sc["camera"]["mode"] == "follow":
        assert _num(sc["camera"]["width"])
    for st in sc["static"]:
        assert st["type"] in STATIC_TYPES, st["type"]
        for f in STATIC_TYPES[st["type"]]:
            assert f in st, (st["type"], f)
        if "points" in st:
            assert all(len(p) == 2 and _num(p[0]) and _num(p[1]) for p in st["points"])
    bodies = sc["bodies"]
    assert bodies
    for b in bodies:
        assert b["shape"] in ("poly", "circle", "rect")
        if b["shape"] == "poly":
            assert len(b["points"]) >= 3
        elif b["shape"] == "circle":
            assert _num(b["r"])
        else:
            assert _num(b["w"]) and _num(b["h"])
    effects = sc.get("effects", [])
    for e in effects:
        assert 0 <= e["body"] < len(bodies)
        for f in ("x", "y", "angle", "size"):
            assert _num(e[f])
        assert e["kind"] == "flame"
    agents = rep["agents"]
    assert agents and all("label" in a and _num(a["alpha"]) for a in agents)
    frames = rep["frames"]
    assert len(frames) >= 1
    for fr in frames:
        assert len(fr) == len(agents)
        for af in fr:
            assert len(af) == 3 * len(bodies)
            assert all(_num(v) for v in af)
    if "fx" in rep:
        assert len(rep["fx"]) == len(frames)
        for fr in rep["fx"]:
            assert len(fr) == len(agents)
            assert all(len(v) == len(effects) and all(0 <= x <= 1 for x in v) for v in fr)
    if "hud" in rep:
        assert len(rep["hud"]) == len(frames) and all(isinstance(h, str) for h in rep["hud"])
    if rep.get("overlay"):
        for arrow in rep["overlay"]["arrows"]:
            assert len(arrow) == 5 and all(_num(v) for v in arrow)
    st = rep["stats"]
    assert _num(st["reward"]) and _num(st["distance"]) and isinstance(st["steps"], int)
    assert isinstance(st["finished"], bool)
    if rep["kind"] == "car":
        assert _num(sc["finish_x"])
    text = json.dumps(rep)
    assert len(text) < 1_500_000


def make_world(kind, **kw):
    parts = {
        "car": [rl.Terrain("rough"), rl.Drive("rear"), rl.Lean()],
        "cartpole": [rl.PoleLength(1.0)],
        "mountaincar": [],
        "pendulum": [rl.Gravity(9.8)],
        "lander": [rl.Wind(0.5), rl.Fuel(8)],
        "maze": [rl.MazeSize(6, 5), rl.MazeWalls(0.25), rl.MazeTraps(2)],
        "flappy": [rl.Gap(3.0), rl.PipeSpeed(3.5)],
    }[kind]
    return rl.World(kind, name=f"{kind}-world", parts=parts + list(kw.get("extra", [])), _bid=f"{kind}-block")


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_world_steps_and_previews(kind, quick_and_quiet):
    world = make_world(kind)
    env = world.make_env()
    obs = env.reset(1)
    assert obs.shape == (world.obs_size,) and np.isfinite(obs).all()
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, r, term, trunc = env.step(world.action_spec.sample(rng))
        assert obs.shape == (world.obs_size,) and np.isfinite(obs).all() and math.isfinite(r)
        if term or trunc:
            obs = env.reset(2)
    st = env.state()
    for key in ("distance", "speed", "tilt", "spin", "height", "x", "y", "time", "fuel", "touching_ground"):
        assert key in st and _num(st[key]), key
    assert len(env.frame()) == 3 * len(env.scene()["bodies"])
    rl.show_world(world, _bid="show")
    reps = of_type(quick_and_quiet, "sim_replay")
    assert len(reps) == 1 and reps[0]["block"] == "show" and reps[0]["kind"] == kind
    check_replay(reps[0])


def test_car_with_every_part():
    parts = [rl.Terrain("mixed", bumpiness=0.8, hills=1.5, length=60), rl.SameWorld(False),
             rl.CarBody(length=2.4, wheel_size=0.5, weight=1.5), rl.Drive("all", power=1.5), rl.Brake(), rl.Lean(),
             rl.Jump(), rl.Boost(power=1.0, fuel=1.0)]
    parts += [rl.Sense(k) for k in ("speed", "tilt", "spin", "wheels", "height", "progress", "fuel")]
    parts += [rl.Sense("ground", rays=7, reach=5.0), rl.Reward("distance", 1.0), rl.Reward("speed", 0.1),
              rl.Reward("upright", 0.1), rl.Reward("finish", 100.0), rl.Reward("flip", -10.0, end=True),
              rl.Reward("energy", -0.01), rl.Reward("time", -0.01), rl.EndAfter(10), rl.EndIfStuck(2),
              rl.Gravity(9.8)]
    world = rl.World("car", name="every", parts=parts)
    assert world.action_spec.names == ["drive", "brake", "lean", "jump", "boost"]
    assert world.obs_size == 2 + 2 + 1 + 7 + 2 + 1 + 1 + 1
    env = world.make_env()
    env.reset(3)
    for _ in range(40):
        obs, r, term, trunc = env.step([1.0, 0.0, 0.2, 0.0, 1.0])
        if term or trunc:
            break
    assert env.fuel < 1.0                       # the rocket burnt fuel
    assert env.scene()["effects"][0]["angle"] == 180
    assert env.fx() is not None


@pytest.mark.parametrize("kind", ["car", "lander", "flappy", "cartpole"])
def test_worlds_are_deterministic(kind):
    world = make_world(kind)
    acts = [world.action_spec.sample(np.random.default_rng(i)) for i in range(120)]

    def run(env):
        env.reset(5)
        out = []
        for a in acts:
            obs, r, term, trunc = env.step(a)
            out.append(env.frame() + [r])
            if term or trunc:
                break
        return out

    fresh = run(world.make_env())
    used = world.make_env()
    used.reset(9)
    for a in acts[:30]:
        used.step(a)
    assert run(used) == fresh


def test_car_terrain_kinds_and_defaults(quick_and_quiet):
    for kind in ("flat", "hills", "rough", "steps", "mixed"):
        world = rl.World("car", name=f"t-{kind}", parts=[rl.Terrain(kind, length=40)])
        env = world.make_env()
        env.reset(0)
        for _ in range(30):
            env.step([1.0])
        assert env.x > 1.0, f"the car should drive forwards on {kind} terrain"
    # defaults: drive (rear), 5 sensors, distance + flip rewards, 20 s, stuck after 3 s
    w = rl.World("car", name="defaults", parts=[])
    assert w.action_spec.names == ["drive"]
    assert w.obs_size == 12
    assert [r[0] for r in w.cfg.rewards] == ["distance", "flip"]
    assert w.cfg.time_limit == 20 and w.cfg.stuck == 3
    assert any("no abilities" in e.get("text", "") for e in of_type(quick_and_quiet, "log"))


def test_parts_that_do_not_apply_are_ignored_with_a_warning(quick_and_quiet):
    world = rl.World("cartpole", name="cp", parts=[rl.Terrain("hills", _bid="t1"), rl.EndIfStuck(3, _bid="s1"),
                                                   rl.Reward("speed", 0.1), rl.Reward("distance", 1.0)])
    warns = [e for e in of_type(quick_and_quiet, "log") if e.get("level") == "warn"]
    assert {e.get("block") for e in warns} >= {"t1", "s1"}
    assert world.obs_size == 4
    with pytest.raises(NBError):
        rl.World("submarine")
    with pytest.raises(NBError):
        rl.World("car", parts=[rl.Reward("happiness", 1.0)])
    with pytest.raises(NBError):
        rl.Terrain("lava")


def test_custom_reward_and_end_when():
    world = rl.World("cartpole", name="c", parts=[rl.CustomReward(lambda s: 2.0 + 0 * s["tilt"]),
                                                  rl.EndWhen(lambda s: s["time"] > 0.1)])
    env = world.make_env()
    env.reset(0)
    rewards = []
    while True:
        _, r, term, trunc = env.step(1)
        rewards.append(r)
        if term or trunc:
            break
    assert rewards == [2.0] * 6 and term and not trunc


def test_custom_reward_errors_point_at_their_block():
    world = rl.World("car", name="c", parts=[rl.CustomReward(lambda s: s["does_not_exist"], _bid="cr")])
    env = world.make_env()
    env.reset(0)
    with pytest.raises(NBError) as err:
        env.step([0.0])
    assert err.value.nb_block == "cr"
    with pytest.raises(NBError):
        rl.CustomReward(5)


def test_maze_layouts():
    w = rl.World("maze", name="m", parts=[rl.MazeLayout("#####/#S.G#/#####")])
    env = w.make_env()
    env.reset(0)
    _, r, term, _ = env.step(1)
    _, r, term, _ = env.step(1)
    assert term and env.finished and r > 0
    with pytest.raises(NBError):
        rl.World("maze", name="m2", parts=[rl.MazeLayout("#####/#S#G#/#####")])   # goal unreachable
    with pytest.raises(NBError):
        rl.World("maze", name="m3", parts=[rl.MazeLayout("#####/#S..#/#####")])   # no goal
    from neuroblocks.runtime.rl.env_maze import _bfs, random_maze
    for seed in range(15):
        walls, start, goals, traps = random_maze(9, 7, 0.45, 4, seed)
        assert _bfs(walls, start, goals, traps) is not None


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind,algo", [("cartpole", "ppo"), ("car", "ppo"), ("car", "evolution"),
                                       ("cartpole", "dqn"), ("lander", "dqn"), ("maze", "qtable"),
                                       ("pendulum", "evolution"), ("flappy", "ppo")])
def test_algorithms_run(kind, algo, quick_and_quiet):
    world = make_world(kind)
    model = nb.Model("m", [nb.layers.Dense(16, "tanh")])
    calls = []
    STATE.current_block = "train-block"

    def every():
        calls.append(1)
        STATE.current_block = "inner"

    stats = rl.train(model, world, algorithm=algo, steps=2000, generations=2, population=6, every=[(1, "steps", every)],
                     _bid="train-block")
    assert model.agent is not None and model.agent.algorithm == algo
    assert model.meta["task"] == "control" and model.meta["world"] == kind
    assert model.train_seconds > 0
    assert calls and STATE.current_block == "train-block"
    assert stats["steps"] > 0 and stats["score"] is not None
    reps = of_type(quick_and_quiet, "sim_replay")
    assert reps
    for rep in reps:
        check_replay(rep)
    if algo == "evolution":
        assert any(len(r["agents"]) > 1 for r in reps)
        if kind == "car":  # the car always starts the same way: the champion replays exactly as it scored
            assert rl.average_score(model, world, episodes=1) == pytest.approx(stats["score"], abs=1e-3)
    if kind == "maze":
        assert reps[-1]["overlay"]["arrows"]
    metrics = of_type(quick_and_quiet, "metric")
    assert any(m["chart"] == "reward" for m in metrics)
    progress = of_type(quick_and_quiet, "progress")
    assert progress and progress[-1]["id"].startswith("train-")
    assert any(e.get("level") == "success" for e in of_type(quick_and_quiet, "log"))
    score = rl.average_score(model, world, episodes=2)
    assert isinstance(score, float) and math.isfinite(score)


def test_qtable_needs_a_maze():
    with pytest.raises(NBError) as err:
        rl.train(nb.Model("m", []), make_world("car"), algorithm="qtable", _bid="b")
    assert err.value.hint and err.value.nb_block == "b"
    with pytest.raises(NBError):
        rl.train(nb.Model("m", []), make_world("car"), algorithm="magic")
    with pytest.raises(NBError):
        rl.train(make_world("car"), nb.Model("m", []))


def test_qtable_solves_a_small_maze():
    STATE.quick = False
    world = rl.World("maze", name="small", parts=[rl.MazeLayout("#######/#S..#.#/#.#...#/#...#G#/#######")])
    model = nb.Model("m", [])
    rl.train(model, world, algorithm="qtable", steps=4000, explore=0.1)
    env = world.make_env()
    obs = env.reset(world.seed)
    for _ in range(30):
        obs, r, term, trunc = env.step(model.agent.act(obs))
        if term or trunc:
            break
    assert env.finished and env.steps <= 8


def test_training_continues_and_resets_on_a_new_world(quick_and_quiet):
    world = make_world("cartpole")
    model = nb.Model("m", [])
    rl.train(model, world, algorithm="ppo", steps=500)
    agent = model.agent
    first = agent.steps
    rl.train(model, world, algorithm="ppo", steps=500)
    assert model.agent is agent and agent.steps > first
    rl.train(model, make_world("lander"), algorithm="ppo", steps=500)
    assert model.agent is not agent
    assert any("from scratch" in e.get("text", "") for e in of_type(quick_and_quiet, "log"))


def test_skip_finishes_early_but_keeps_the_agent(quick_and_quiet):
    world = make_world("cartpole")
    model = nb.Model("m", [])
    STATE.skip.set()
    stats = rl.train(model, world, algorithm="dqn", steps=2000)
    assert model.agent is not None and stats["rounds"] == 1
    assert any("Skip" in e.get("text", "") for e in of_type(quick_and_quiet, "log"))


def test_save_load_and_watch(tmp_path, quick_and_quiet):
    world = make_world("car")
    model = nb.Model("driver", [nb.layers.Dense(24, "tanh"), nb.layers.Dense(24, "tanh")])
    rl.train(model, world, algorithm="evolution", generations=2, population=6)
    before = rl.average_score(model, world, episodes=1)
    path = nb.save(model, str(tmp_path / "driver.pt"))
    loaded = nb.load_model(path, name="driver2")
    assert loaded.agent is not None and loaded.agent.kind == "policy"
    assert rl.average_score(loaded, world, episodes=1) == pytest.approx(before)
    quick_and_quiet.clear()
    rl.watch(loaded, world, episodes=2, _bid="w")
    reps = of_type(quick_and_quiet, "sim_replay")
    assert len(reps) == 1 and reps[0]["block"] == "w"     # quick mode: 1 episode
    check_replay(reps[0])
    # a model trained in one world can't play in a different one
    with pytest.raises(NBError):
        rl.watch(loaded, make_world("cartpole"))


@pytest.mark.parametrize("algo,kind", [("ppo", "cartpole"), ("dqn", "car"), ("qtable", "maze")])
def test_save_load_other_agent_kinds(tmp_path, algo, kind):
    world = make_world(kind)
    model = nb.Model("m", [])
    rl.train(model, world, algorithm=algo, steps=600)
    before = rl.average_score(model, world, episodes=1)
    loaded = nb.load_model(nb.save(model, str(tmp_path / f"{algo}.pt")))
    assert rl.average_score(loaded, world, episodes=1) == pytest.approx(before)


def test_watch_untrained_model_moves_randomly(quick_and_quiet):
    world = make_world("pendulum")
    rl.watch(nb.Model("fresh", []), world, episodes=1)
    assert of_type(quick_and_quiet, "sim_replay")
    assert any(e.get("level") == "warn" for e in of_type(quick_and_quiet, "log"))


def test_play_yourself(monkeypatch, quick_and_quiet):
    world = make_world("car")
    assert rl.play_yourself(world) == 0.0          # outside the editor it only logs
    from neuroblocks.runtime.rl import play
    monkeypatch.setattr(play, "gui_mode", lambda: True)
    STATE.keys = {"ArrowRight": True}

    def finish():
        time.sleep(0.4)
        STATE.interact_q.put({"cmd": "interact_end"})

    threading.Thread(target=finish, daemon=True).start()
    rl.play_yourself(world, _bid="p")
    STATE.keys = {}
    evs = quick_and_quiet
    start = of_type(evs, "interactive")
    assert start and start[0]["kind"] == "game" and "ArrowRight" in start[0]["options"]["keys"]
    live = of_type(evs, "sim_live")
    assert live[0]["reset"] and live[0]["scene"]["bodies"]
    frames = [e for e in live if "frame" in e and not e.get("reset")]
    assert frames and len(frames[-1]["frame"]) == 9
    assert frames[-1]["frame"][0] > live[0]["frame"][0]   # the right arrow drove the car forwards
    assert of_type(evs, "interactive_end")
