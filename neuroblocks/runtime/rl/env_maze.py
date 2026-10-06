"""Maze (grid world): walk from the start to the goal without stepping on traps.

Layouts come from ``MazeLayout`` text (``#`` wall, ``S`` start, ``G`` goal, ``T`` trap, ``.`` floor;
rows split by newlines or ``/``) or are generated (``MazeSize`` + ``MazeWalls`` + ``MazeTraps``) and
are then always solvable. Observations: the agent's cell (one-hot), walls and traps next to it and the
direction of the goal. The Q-table algorithm uses the cell number as its state.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..errors import NBError
from .env_base import ActionSpec, Env
from .parts import MazeLayout, MazeSize, MazeTraps, MazeWalls

DEFAULT_LAYOUT = [
    "##########",
    "#S...#...#",
    "#.##.#.#.#",
    "#.#..T.#.#",
    "#.#.####.#",
    "#...#..#.#",
    "###.#.##.#",
    "#T....#..#",
    "#.###...G#",
    "##########",
]
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))   # up, right, down, left (row, col)
SUBFRAMES = 3


def parse_layout(text: str, bid=None):
    rows = [r.rstrip("\r") for r in str(text).replace("/", "\n").split("\n")]
    rows = [r for r in rows if r.strip()]
    if not rows:
        raise NBError("The maze layout is empty.", block_id=bid)
    width = max(len(r) for r in rows)
    grid = [list(r.ljust(width, "#")) for r in rows]
    start, goals, traps = None, set(), set()
    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch in ("S", "s"):
                if start is not None:
                    raise NBError("The maze has more than one start (S).", block_id=bid)
                start = (i, j)
                row[j] = "."
            elif ch in ("G", "g"):
                goals.add((i, j))
                row[j] = "."
            elif ch in ("T", "t", "X", "x"):
                traps.add((i, j))
                row[j] = "."
            elif ch in ("#", "█", "W", "w"):
                row[j] = "#"
            elif ch in (".", " ", "_", "-", "0"):
                row[j] = "."
            else:
                raise NBError(f"Unknown maze symbol '{ch}'.",
                              hint="Use # for walls, S start, G goal, T trap and . for floor.", block_id=bid)
    if not goals:
        raise NBError("The maze has no goal (G).", block_id=bid)
    if start is None:
        free = [(i, j) for i, row in enumerate(grid) for j, ch in enumerate(row)
                if ch == "." and (i, j) not in goals and (i, j) not in traps]
        if not free:
            raise NBError("The maze has no free cell to start in.", block_id=bid)
        start = free[0]
    walls = [[ch == "#" for ch in row] for row in grid]
    if _bfs(walls, start, goals, traps) is None:
        raise NBError("The goal can't be reached from the start in this maze.",
                      hint="Open a path (with . cells) from S to G that avoids traps.", block_id=bid)
    return walls, start, goals, traps


def _bfs(walls, start, goals, traps=(), from_goal=False):
    """Shortest path length from start to any goal (or None). With from_goal: distance map from the goals."""
    h, w = len(walls), len(walls[0])
    dist = {}
    q = deque()
    sources = list(goals) if from_goal else [start]
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        r, c = q.popleft()
        if not from_goal and (r, c) in goals:
            return dist[(r, c)]
        for dr, dc in MOVES:
            n = (r + dr, c + dc)
            if 0 <= n[0] < h and 0 <= n[1] < w and not walls[n[0]][n[1]] and n not in traps and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist if from_goal else None


def random_maze(width, height, density, n_traps, seed):
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 41])
    H, W = height + 2, width + 2
    walls = [[r in (0, H - 1) or c in (0, W - 1) for c in range(W)] for r in range(H)]
    start, goal = (1, 1), (H - 2, W - 2)
    inner = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if (r, c) not in (start, goal)]
    order = rng.permutation(len(inner))
    for k in order[: int(round(density * len(inner)))]:
        r, c = inner[k]
        walls[r][c] = True
    if _bfs(walls, start, {goal}) is None:
        # carve the cheapest path (fewest walls to remove) with a 0-1 BFS
        cost = {start: 0}
        prev = {}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            if cur == goal:
                break
            for dr, dc in MOVES:
                n = (cur[0] + dr, cur[1] + dc)
                if not (1 <= n[0] < H - 1 and 1 <= n[1] < W - 1):
                    continue
                step = 1 if walls[n[0]][n[1]] else 0
                if n not in cost or cost[cur] + step < cost[n]:
                    cost[n] = cost[cur] + step
                    prev[n] = cur
                    (dq.append if step else dq.appendleft)(n)
        node = goal
        while node != start:
            walls[node[0]][node[1]] = False
            node = prev[node]
    traps = set()
    floor = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)
             if not walls[r][c] and (r, c) not in (start, goal)]
    for k in rng.permutation(len(floor)):
        if len(traps) >= n_traps:
            break
        cand = floor[k]
        if abs(cand[0] - start[0]) + abs(cand[1] - start[1]) <= 1:
            continue
        if _bfs(walls, start, {goal}, traps | {cand}) is not None:
            traps.add(cand)
    return walls, start, {goal}, traps


class MazeEnv(Env):
    KIND = "maze"
    TITLE = "maze"
    DT = 0.25
    USES_GRAVITY = False
    HANDLES = ("MazeLayout", "MazeSize", "MazeWalls", "MazeTraps")
    REWARD_KINDS = {
        "finish": "once, when the agent reaches the goal",
        "flip": "once, when the agent steps on a trap",
        "distance": "+1 for every step closer to the goal, −1 for every step further away",
        "time": "1 every step",
    }
    # goal +10, trap −10, −0.1 per step, and "warmer/colder" hints (+0.5 closer, −0.5 further away)
    DEFAULT_REWARDS = (("finish", 10.0, True), ("flip", -10.0, True), ("time", -0.1, False), ("distance", 0.5, False))
    KEYS = {"ArrowUp": "move up", "ArrowRight": "move right", "ArrowDown": "move down", "ArrowLeft": "move left"}
    HINTS = {"ppo_rollout": 1024, "ppo_epochs": 10, "ppo_minibatch": 128}
    TURN_BASED = True
    FRAME_SUBSTEPS = SUBFRAMES

    @classmethod
    def configure(cls, cfg, parts):
        from .world import warn_part
        layout = size = walls = traps = None
        unused = []
        for p in parts:
            if isinstance(p, MazeLayout):
                layout = p
            elif isinstance(p, MazeSize):
                size = p
            elif isinstance(p, MazeWalls):
                walls = p
            elif isinstance(p, MazeTraps):
                traps = p
            else:
                unused.append(p)
        if layout is not None:
            for p in (size, walls, traps):
                if p is not None:
                    warn_part(p, f"The maze is drawn by 'maze layout', so '{p.label}' was ignored.")
            cfg.maze_fixed = parse_layout(layout.text, layout.bid)
            cfg.maze_random = None
        elif size is None and walls is None and traps is None:
            cfg.maze_fixed = parse_layout("\n".join(DEFAULT_LAYOUT))
            cfg.maze_random = None
        else:
            w = size.width if size else 8
            h = size.height if size else 8
            cfg.maze_fixed = None
            cfg.maze_random = (w, h, walls.density if walls else 0.25, traps.count if traps else 0)
        return unused

    @classmethod
    def default_time(cls, cfg):
        if cfg.maze_fixed is not None:
            walls = cfg.maze_fixed[0]
            cells = sum(not v for row in walls for v in row)
        else:
            w, h, d, _ = cfg.maze_random
            cells = w * h * (1 - d)
        return max(15, int(3 * cells)) * cls.DT

    def _setup(self):
        cfg = self.cfg
        if cfg.maze_fixed is not None:
            walls = cfg.maze_fixed[0]
            self.H, self.W = len(walls), len(walls[0])
        else:
            w, h, _, _ = cfg.maze_random
            self.H, self.W = h + 2, w + 2
        self.n_cells = self.H * self.W
        self.obs_names = [f"at cell {r},{c}" for r in range(self.H) for c in range(self.W)] + \
            ["wall up", "wall right", "wall down", "wall left", "trap up", "trap right", "trap down", "trap left",
             "goal across", "goal down"]
        self.action_spec = ActionSpec("discrete", 4, ["up", "right", "down", "left"], noop=0)
        self._layout = None

    def _load(self, seed):
        cfg = self.cfg
        if cfg.maze_fixed is not None:
            layout = cfg.maze_fixed
        else:
            w, h, d, n = cfg.maze_random
            layout = random_maze(w, h, d, n, seed)
        self.walls, self.start, self.goals, self.traps = layout
        self.goal_dist = _bfs(self.walls, None, self.goals, self.traps, from_goal=True)
        g = sorted(self.goals)[0]
        self.goal_rc = g
        self._layout = seed

    def _reset(self, seed):
        lseed = self.cfg.seed if self.cfg.same_world else seed
        if self._layout is None or (self.cfg.maze_random is not None and self._layout != lseed):
            self._load(lseed)
        self.pos = self.start
        self.prev_pos = self.start
        self.bumped = False
        self.trapped = False

    def _free(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W and not self.walls[r][c]

    def _step(self, a):
        dr, dc = MOVES[a]
        r, c = self.pos
        self.prev_pos = self.pos
        terms = {"time": 1.0}
        if self._free(r + dr, c + dc):
            self.pos = (r + dr, c + dc)
            self.bumped = False
        else:
            self.bumped = True
        d0 = self.goal_dist.get(self.prev_pos)
        d1 = self.goal_dist.get(self.pos)
        if d0 is not None and d1 is not None and d1 != d0:
            terms["distance"] = float(d0 - d1)
        over = False
        if self.pos in self.goals:
            terms["finish"] = 1.0
            self.finished = True
            self.end_reason = "reached the goal"
            over = True
        elif self.pos in self.traps:
            terms["flip"] = 1.0
            self.trapped = True
            self.end_reason = "stepped on a trap"
            over = True
        return terms, over

    def cell_index(self, rc=None):
        r, c = rc or self.pos
        return r * self.W + c

    def obs_at(self, rc):
        r, c = rc
        o = np.zeros(self.obs_size, dtype=np.float32)
        o[r * self.W + c] = 1.0
        base = self.n_cells
        for k, (dr, dc) in enumerate(MOVES):
            n = (r + dr, c + dc)
            o[base + k] = 0.0 if self._free(*n) else 1.0
            o[base + 4 + k] = 1.0 if n in self.traps else 0.0
        gr, gc = self.goal_rc
        o[base + 8] = (gc - c) / max(1, self.W - 1)
        o[base + 9] = (gr - r) / max(1, self.H - 1)
        return o

    def observe(self):
        return self.obs_at(self.pos)

    def state(self):
        r, c = self.pos
        st = self.base_state()
        d = self.goal_dist.get(self.pos)
        st.update({"distance": float(self.goal_dist.get(self.start, 0) - (d if d is not None else 0)),
                   "x": c + 0.5, "y": self.H - 0.5 - r, "row": r, "column": c, "steps_to_goal": d if d is not None
                   else -1, "touching_ground": 1, "bumped": 1 if self.bumped else 0})
        return st

    def progress(self):
        return float(self.state()["distance"])

    def free_cells(self):
        return [(r, c) for r in range(self.H) for c in range(self.W) if not self.walls[r][c]]

    def _xy(self, rc):
        r, c = rc
        return c + 0.5, self.H - 0.5 - r

    def scene(self):
        static = [{"type": "rect", "x": 0.0, "y": 0.0, "w": float(self.W), "h": float(self.H), "color": "#f5f0e6"}]
        for r in range(self.H):
            for c in range(self.W):
                if self.walls[r][c]:
                    static.append({"type": "rect", "x": float(c), "y": float(self.H - 1 - r), "w": 1.0, "h": 1.0,
                                   "color": "#455a64"})
        for r, c in sorted(self.traps):
            static.append({"type": "rect", "x": c + 0.1, "y": self.H - 1 - r + 0.1, "w": 0.8, "h": 0.8,
                           "color": "#e57373"})
        for r, c in sorted(self.goals):
            static.append({"type": "rect", "x": float(c), "y": float(self.H - 1 - r), "w": 1.0, "h": 1.0,
                           "color": "#ffd54f"})
            static.append({"type": "flag", "x": c + 0.4, "y": self.H - 1 - r + 0.1, "label": "goal"})
        sx, sy = self._xy(self.start)
        static.append({"type": "circle", "x": sx, "y": sy, "r": 0.12, "color": "#90caf9"})
        return {"bounds": [0.0, 0.0, float(self.W), float(self.H)], "camera": {"mode": "fixed"}, "sky": "#cfd8dc",
                "static": static, "bodies": [{"name": "agent", "shape": "circle", "r": 0.32, "color": "#1e88e5"}]}

    def frame(self):
        x, y = self._xy(self.pos)
        return [x, y, 0.0]

    def frames(self):
        """Frames for one move, sliding smoothly from the previous cell."""
        x0, y0 = self._xy(self.prev_pos)
        x1, y1 = self._xy(self.pos)
        if self.bumped:
            return [[x1, y1, 0.0]] * SUBFRAMES
        return [[x0 + (x1 - x0) * k / SUBFRAMES, y0 + (y1 - y0) * k / SUBFRAMES, 0.0] for k in range(1, SUBFRAMES + 1)]

    def hud(self):
        s = f"step {self.steps} · reward {self.ep_return:.1f}"
        if self.finished:
            s += " · reached the goal!"
        elif self.trapped:
            s += " · trapped"
        return s

    def human_action(self, keys):
        k = keys or {}
        for i, name in enumerate(("ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft")):
            if k.get(name):
                return i
        return None

    def overlay(self, agent=None, qtable=None):
        """Arrows showing the best move in every free cell (from a Q-table or the agent's greedy policy)."""
        cells = [rc for rc in self.free_cells() if rc not in self.goals]
        if not cells:
            return None
        if qtable is not None:
            vals = np.asarray(qtable)[[self.cell_index(rc) for rc in cells]]
            best = vals.argmax(1)
            score = vals.max(1)
        elif agent is not None:
            obs = np.stack([self.obs_at(rc) for rc in cells])
            best, score = agent.greedy_with_value(obs)
        else:
            return None
        arrows = []
        for (r, c), b, v in zip(cells, best, score):
            dr, dc = MOVES[int(b)]
            x, y = self._xy((r, c))
            arrows.append([round(x, 3), round(y, 3), round(0.32 * dc, 3), round(-0.32 * dr, 3), round(float(v), 3)])
        return {"arrows": arrows}
