"""Simulation blocks: worlds, their parts (abilities, senses, rewards …) and reinforcement learning."""
from __future__ import annotations

from .core import ORDER_FUNCTION_CALL, ORDER_MEMBER, ORDER_UNARY_SIGN, BlockCtx
from .helpers import assign, fmt_call, fmt_list, int_code, setting, settings_kwargs
from .spec import Drop, Name, Num, Ref, Stack, Text, Toggle, Val, block

WP = "WorldPart"   # fits every world
CP = "CarPart"
MP = "MazePart"
LP = "LanderPart"
FP = "FlappyPart"
PP = "PolePart"
RS = "RLSetting"


def _world(b: BlockCtx, kind: str):
    parts = []
    for child in b.child_blocks("PARTS"):
        code = b.c.setting(child.data)
        if code is None:
            continue
        for s in code if isinstance(code, list) else [code]:
            if s.code:
                parts.append(s.code)
    expr = fmt_call("nb.rl.World", [repr(kind), f"name={b.name()!r}", "parts=" + fmt_list(parts)] + b.bid_kw()) \
        if parts else fmt_call("nb.rl.World", [repr(kind), f"name={b.name()!r}"] + b.bid_kw())
    return assign(b.name_ident(), expr)


def _part(code: str):
    return setting("_part", code)


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


@block("nb_world_car", "sims", "create car world %NAME %PARTS", NAME=Name("world", "world"), PARTS=Stack([WP, CP]),
       defines=("world", "NAME"), section="Worlds",
       prefill={"PARTS": [("nb_w_terrain", {"KIND": "rough"}), ("nb_w_drive", {}), ("nb_w_lean", {}),
                          ("nb_w_sense", {"WHAT": "speed"}), ("nb_w_sense", {"WHAT": "tilt"}),
                          ("nb_w_sense_ground", {}), ("nb_w_reward", {"WHAT": "distance", "W": 1}),
                          ("nb_w_flip", {}), ("nb_w_end_after", {"SECS": 20})]},
       tooltip="A 2-D physics world with a little car on bumpy ground. Give it abilities, senses and rewards, then "
               "let a model learn to drive.",
       help="Build the car's world from parts (World Parts category): the terrain, what the car CAN DO "
            "(abilities become the model's outputs), what it CAN SENSE (senses become the model's inputs), and "
            "what counts as doing well (rewards). Reinforcement learning then tries things, keeps what earns "
            "reward, and slowly gets better.")
def _car(b):
    return _world(b, "car")


@block("nb_world_cartpole", "sims", "create balancing-pole world %NAME %PARTS", NAME=Name("world", "world"),
       PARTS=Stack([WP, PP]), defines=("world", "NAME"),
       tooltip="Classic CartPole: push a cart left or right to keep a pole standing up.")
def _cartpole(b):
    return _world(b, "cartpole")


@block("nb_world_mountaincar", "sims", "create mountain car world %NAME %PARTS", NAME=Name("world", "world"),
       PARTS=Stack([WP]), defines=("world", "NAME"),
       tooltip="A weak car must rock back and forth to build up speed and reach the flag on the hill.")
def _mountaincar(b):
    return _world(b, "mountaincar")


@block("nb_world_pendulum", "sims", "create pendulum world %NAME %PARTS", NAME=Name("world", "world"),
       PARTS=Stack([WP]), defines=("world", "NAME"),
       tooltip="Swing a pendulum up and balance it upside-down using a weak motor.")
def _pendulum(b):
    return _world(b, "pendulum")


@block("nb_world_lander", "sims", "create rocket lander world %NAME %PARTS", NAME=Name("world", "world"),
       PARTS=Stack([WP, LP]), defines=("world", "NAME"),
       tooltip="Fire the main and side thrusters to land a rocket gently on the pad.")
def _lander(b):
    return _world(b, "lander")


@block("nb_world_maze", "sims", "create maze world %NAME %PARTS", NAME=Name("world", "world"), PARTS=Stack([WP, MP]),
       defines=("world", "NAME"), prefill={"PARTS": [("nb_w_maze_size", {"W": 8, "H": 8}), ("nb_w_maze_walls", {"PCT": 25})]},
       tooltip="A grid maze: find the way from the start to the goal (avoid traps).")
def _maze(b):
    return _world(b, "maze")


@block("nb_world_flappy", "sims", "create flappy bird world %NAME %PARTS", NAME=Name("world", "world"),
       PARTS=Stack([WP, FP]), defines=("world", "NAME"),
       tooltip="Flap to fly through the gaps between pipes.")
def _flappy(b):
    return _world(b, "flappy")


# ---------------------------------------------------------------------------
# Learning & testing in worlds
# ---------------------------------------------------------------------------


@block("nb_rl_train", "sims", "train %MODEL to play in %WORLD %SETTINGS", MODEL=Ref("model", "model"),
       WORLD=Ref("world", "world"), SETTINGS=Stack(RS), section="Learn by trying (reinforcement learning)",
       prefill={"SETTINGS": [("nb_rl_algo", {"ALGO": "ppo"}), ("nb_rl_steps", {"N": 100000})]},
       tooltip="Reinforcement learning: the model controls the world, gets rewards, and learns what works.",
       help="The model's inputs are the world's senses and its outputs are the abilities. PPO is a good "
            "all-rounder. Evolution runs a whole population at once and keeps the best — fun to watch. DQN "
            "picks between a few actions. Q-table is for small mazes. Replays appear in the Sim tab.")
def _rl_train(b):
    kws, pre = settings_kwargs(b, "SETTINGS", context="learning in a world")
    return pre + fmt_call("nb.rl.train", [b.ref("MODEL"), b.ref("WORLD")] + kws + b.bid_kw()).split("\n")


def _rs(type_, message, tooltip, **kw):
    return block(type_, "sims", message, shape="setting", check=[RS], tooltip=tooltip, **kw)


@_rs("nb_rl_algo", "learn with %ALGO", "The learning method.",
     ALGO=Drop([("PPO (recommended)", "ppo"), ("evolution (population)", "evolution"), ("DQN", "dqn"),
                ("Q-table (mazes)", "qtable")], "ppo"), section="Learning settings")
def _algo(b):
    return setting("algorithm", b.q("ALGO"))


@_rs("nb_rl_steps", "for %N steps", "How many moments of practice in total (more = better, slower).",
     N=Num(100000, integer=True))
def _rsteps(b):
    return setting("steps", int_code(b, "N"))


@_rs("nb_rl_generations", "for %N generations", "Evolution: how many rounds of survival of the fittest.",
     N=Num(30, integer=True))
def _gens(b):
    return setting("generations", int_code(b, "N"))


@_rs("nb_rl_population", "population %N", "Evolution: how many brains try at the same time.", N=Num(40, integer=True))
def _pop(b):
    return setting("population", int_code(b, "N"))


@_rs("nb_rl_mutation", "mutation strength %S", "Evolution: how much children differ from their parents.",
     S=Num(0.1))
def _mut(b):
    return setting("mutation", b.val("S"))


@_rs("nb_rl_lr", "learning rate %LR", "How big each learning step is.", LR=Num(0.0003))
def _rlr(b):
    return setting("lr", b.val("LR"))


@_rs("nb_rl_gamma", "care about the future %G", "Discount factor: 0.99 = plan ahead, 0.9 = short-sighted.",
     G=Num(0.99))
def _gamma(b):
    return setting("gamma", b.val("G"))


@_rs("nb_rl_parallel", "run %N worlds at once", "More copies of the world = more experience per update.",
     N=Num(8, integer=True))
def _par(b):
    return setting("parallel", int_code(b, "N"))


@_rs("nb_rl_explore", "exploration %E", "How much to try random things (0 = never, 1 = always).", E=Num(0.1))
def _explore(b):
    return setting("explore", b.val("E"))


@_rs("nb_rl_replay", "show a replay every %N updates", "How often to send an animation to the Sim tab.",
     N=Num(5, integer=True))
def _replay(b):
    return setting("replay_every", int_code(b, "N"))


@_rs("nb_rl_label", "name this run %LABEL", "The name used for this run's lines in the charts.", LABEL=Text("run 1"))
def _rlabel(b):
    return setting("label", b.q("LABEL"))


@block("nb_rl_watch", "sims", "watch %MODEL play in %WORLD %N times", MODEL=Ref("model", "model"),
       WORLD=Ref("world", "world"), N=Num(3, integer=True), section="Watch & test",
       tooltip="Replay the trained model in the world (Sim tab).")
def _watch(b):
    return [f"nb.rl.watch({b.ref('MODEL')}, {b.ref('WORLD')}, episodes={int_code(b, 'N')}{b.bid()})"]


@block("nb_rl_show", "sims", "show %WORLD", WORLD=Ref("world", "world"),
       tooltip="Preview the world in the Sim tab before training.")
def _show(b):
    return [f"nb.rl.show_world({b.ref('WORLD')}{b.bid()})"]


@block("nb_rl_play", "sims", "let me play %WORLD", WORLD=Ref("world", "world"),
       tooltip="Control the world yourself with the arrow keys (Play tab). Can you beat the AI?")
def _play(b):
    return [f"nb.rl.play_yourself({b.ref('WORLD')}{b.bid()})"]


@block("nb_rl_score", "sims", "average score of %MODEL in %WORLD over %N tries", shape="value", output="Number",
       MODEL=Ref("model", "model"), WORLD=Ref("world", "world"), N=Num(5, integer=True),
       tooltip="Total reward per try, averaged (no randomness in the model's choices).")
def _score(b):
    return f"nb.rl.average_score({b.ref('MODEL')}, {b.ref('WORLD')}, episodes={int_code(b, 'N')})", ORDER_FUNCTION_CALL


# ---------------------------------------------------------------------------
# World parts: car
# ---------------------------------------------------------------------------


def _cp(type_, message, tooltip, check=CP, **kw):
    return block(type_, "world", message, shape="setting", check=[check], tooltip=tooltip, **kw)


@_cp("nb_w_terrain", "terrain %KIND bumpiness %BUMP hills %HILLS length %LEN m", "The ground the car drives on.",
     KIND=Drop([("rough", "rough"), ("hills", "hills"), ("flat", "flat"), ("steps", "steps"), ("mixed", "mixed")],
               "rough"),
     BUMP=Num(0.5), HILLS=Num(1.0), LEN=Num(150), section="Car world: ground & body")
def _terrain(b):
    return _part(f"nb.rl.Terrain({b.q('KIND')}, bumpiness={b.val('BUMP')}, hills={b.val('HILLS')}, "
                 f"length={b.val('LEN')}{b.bid()})")


@_cp("nb_w_carbody", "car body length %LEN wheel size %WHEEL weight %W", "The car's shape and weight.",
     LEN=Num(2.0), WHEEL=Num(0.45), W=Num(1.0))
def _carbody(b):
    return _part(f"nb.rl.CarBody(length={b.val('LEN')}, wheel_size={b.val('WHEEL')}, weight={b.val('W')}{b.bid()})")


@_cp("nb_w_drive", "can drive with %WHEELS wheels power %POWER", "An ability: spin the wheels forward or back.",
     WHEELS=Drop([("rear", "rear"), ("front", "front"), ("all", "all")], "rear"), POWER=Num(1.0),
     section="Car abilities (the model's outputs)")
def _drive(b):
    return _part(f"nb.rl.Drive({b.q('WHEELS')}, power={b.val('POWER')}{b.bid()})")


@_cp("nb_w_brake", "can brake", "An ability: lock the wheels.")
def _brake(b):
    return _part(f"nb.rl.Brake({b.bid()[2:]})")


@_cp("nb_w_lean", "can lean forwards / backwards strength %S", "An ability: tilt the car's body (like a rider "
                                                                 "shifting their weight).", S=Num(1.0))
def _lean(b):
    return _part(f"nb.rl.Lean(strength={b.val('S')}{b.bid()})")


@_cp("nb_w_jump", "can jump strength %S", "An ability: hop into the air (needs wheels on the ground).", S=Num(1.0))
def _jump(b):
    return _part(f"nb.rl.Jump(strength={b.val('S')}{b.bid()})")


@_cp("nb_w_boost", "can rocket boost power %P fuel %F seconds", "An ability: a rocket that pushes the car forward.",
     P=Num(1.0), F=Num(3.0))
def _boost(b):
    return _part(f"nb.rl.Boost(power={b.val('P')}, fuel={b.val('F')}{b.bid()})")


@_cp("nb_w_sense", "can sense %WHAT", "A sense: becomes one of the model's inputs.",
     WHAT=Drop([("its speed", "speed"), ("its tilt", "tilt"), ("how fast it spins", "spin"),
                ("wheels touching ground", "wheels"), ("height above ground", "height"),
                ("progress to the finish", "progress"), ("fuel left", "fuel")], "speed"),
     section="Car senses (the model's inputs)")
def _sense(b):
    return _part(f"nb.rl.Sense({b.q('WHAT')}{b.bid()})")


@_cp("nb_w_sense_ground", "can sense the ground ahead with %RAYS rays reaching %REACH m",
     "Like a little radar: measures the distance to the ground in several directions ahead.",
     RAYS=Num(5, integer=True), REACH=Num(4.0))
def _sense_ground(b):
    return _part(f"nb.rl.Sense('ground', rays={int_code(b, 'RAYS')}, reach={b.val('REACH')}{b.bid()})")


@_cp("nb_w_reward", "reward %WHAT × %W", "Points for good behaviour (each moment).",
     WHAT=Drop([("distance moved forward", "distance"), ("speed", "speed"), ("staying upright", "upright")],
               "distance"), W=Num(1.0), section="Car rewards")
def _reward(b):
    return _part(f"nb.rl.Reward({b.q('WHAT')}, {b.val('W')}{b.bid()})")


@_cp("nb_w_finish", "reward reaching the finish +%W", "Bonus points for reaching the flag.", W=Num(100))
def _finish(b):
    return _part(f"nb.rl.Reward('finish', {b.val('W')}{b.bid()})")


@_cp("nb_w_flip", "if it flips over: lose %W points %END", "Penalty (and optionally end the try) when the car's "
                                                         "body hits the ground.", W=Num(10), END=Toggle(True))
def _flip(b):
    return _part(f"nb.rl.Reward('flip', -{b.val('W', ORDER_UNARY_SIGN)}, end={b.flag('END')}{b.bid()})")


@_cp("nb_w_penalty", "lose points for %WHAT × %W", "Small penalties to encourage efficiency.",
     WHAT=Drop([("energy used", "energy"), ("every second", "time")], "energy"), W=Num(0.01))
def _penalty(b):
    return _part(f"nb.rl.Reward({b.q('WHAT')}, -{b.val('W', ORDER_UNARY_SIGN)}{b.bid()})")


@_cp("nb_w_stuck", "end the try if stuck for %SECS seconds", "Stop wasting time when the car isn't moving.",
     SECS=Num(3))
def _stuck(b):
    return _part(f"nb.rl.EndIfStuck({b.val('SECS')}{b.bid()})")


# ---------------------------------------------------------------------------
# World parts: every world
# ---------------------------------------------------------------------------


@_cp("nb_w_end_after", "end each try after %SECS seconds", "Time limit for one try (episode).", check=WP,
     SECS=Num(20), section="Any world")
def _end_after(b):
    return _part(f"nb.rl.EndAfter({b.val('SECS')}{b.bid()})")


@_cp("nb_w_same", "same world every try %ON", "On: every try uses the same layout. Off: a new random layout each "
                                               "time (harder, but the model learns to handle anything).",
     check=WP, ON=Toggle(True))
def _same(b):
    return _part(f"nb.rl.SameWorld({b.flag('ON')}{b.bid()})")


@_cp("nb_w_gravity", "gravity %G", "How strongly things fall (Earth = 9.8, Moon = 1.6).", check=WP, G=Num(9.8))
def _gravity(b):
    return _part(f"nb.rl.Gravity({b.val('G')}{b.bid()})")


@_cp("nb_w_custom_reward", "reward each moment %EXPR", "Your own reward formula, built from the world's "
                                                        "measurements (the 'world's …' reporter).",
     check=WP, EXPR=Val("Number"))
def _custom_reward(b: BlockCtx):
    b.c.world_expr += 1
    try:
        expr = b.val("EXPR", default="0")
    finally:
        b.c.world_expr -= 1
    return _part(f"nb.rl.CustomReward(lambda _s: {expr}{b.bid()})")


@_cp("nb_w_end_when", "end the try when %COND", "Your own rule for ending a try early.", check=WP,
     COND=Val("Boolean"))
def _end_when(b: BlockCtx):
    b.c.world_expr += 1
    try:
        cond = b.val("COND", default="False")
    finally:
        b.c.world_expr -= 1
    return _part(f"nb.rl.EndWhen(lambda _s: {cond}{b.bid()})")


@block("nb_w_state", "world", "world's %WHAT", shape="value", output="Number",
       WHAT=Drop([("distance", "distance"), ("speed", "speed"), ("tilt", "tilt"), ("spin", "spin"),
                  ("height", "height"), ("x position", "x"), ("y position", "y"), ("time", "time"),
                  ("fuel", "fuel"), ("touching ground", "touching_ground")], "distance"),
       tooltip="A measurement of the world right now — use it inside 'reward each moment' or 'end the try when'.")
def _state(b: BlockCtx):
    if not b.c.world_expr:
        b.error("'world's …' only works inside 'reward each moment' or 'end the try when'.")
    return f"_s[{b.q('WHAT')}]", ORDER_MEMBER


# ---------------------------------------------------------------------------
# World parts: other worlds
# ---------------------------------------------------------------------------


@_cp("nb_w_pole_length", "pole length %L m", "Longer poles are easier to balance (slower to fall).", check=PP,
     L=Num(1.0), section="Other worlds")
def _pole(b):
    return _part(f"nb.rl.PoleLength({b.val('L')}{b.bid()})")


@_cp("nb_w_wind", "wind strength %W", "Random gusts push the rocket sideways.", check=LP, W=Num(0.0))
def _wind(b):
    return _part(f"nb.rl.Wind({b.val('W')}{b.bid()})")


@_cp("nb_w_fuel", "fuel %F", "How much the rocket can fire its engines.", check=LP, F=Num(100))
def _fuel(b):
    return _part(f"nb.rl.Fuel({b.val('F')}{b.bid()})")


@_cp("nb_w_maze_size", "maze size %W × %H", "Width and height in squares.", check=MP, W=Num(8, integer=True),
     H=Num(8, integer=True))
def _msize(b):
    return _part(f"nb.rl.MazeSize({int_code(b, 'W')}, {int_code(b, 'H')}{b.bid()})")


@_cp("nb_w_maze_walls", "random walls %PCT %", "How much of the maze is wall (always solvable).", check=MP,
     PCT=Num(25))
def _mwalls(b):
    lit = b.literal("PCT")
    dens = repr(round(lit / 100, 4)) if isinstance(lit, float) else f"{b.val('PCT')} / 100"
    return _part(f"nb.rl.MazeWalls({dens}{b.bid()})")


@_cp("nb_w_maze_traps", "%N traps", "Traps end the try with a penalty.", check=MP, N=Num(3, integer=True))
def _mtraps(b):
    return _part(f"nb.rl.MazeTraps({int_code(b, 'N')}{b.bid()})")


@_cp("nb_w_maze_layout", "maze layout %TEXT", "Draw your own maze: # wall, . floor, S start, G goal, T trap; "
                                              "separate rows with /", check=MP,
     TEXT=Text("S..#..../.#.#.##./.#...#../.##.#.#./....#..G"))
def _mlayout(b):
    return _part(f"nb.rl.MazeLayout({b.q('TEXT')}{b.bid()})")


@_cp("nb_w_gap", "gap size %G", "How big the opening between pipes is.", check=FP, G=Num(3.5))
def _gap(b):
    return _part(f"nb.rl.Gap({b.val('G')}{b.bid()})")


@_cp("nb_w_pipe_speed", "pipe speed %S", "How fast the pipes come at you.", check=FP, S=Num(3.0))
def _pspeed(b):
    return _part(f"nb.rl.PipeSpeed({b.val('S')}{b.bid()})")
