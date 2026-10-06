"""Core runtime: setup, device choice, the 'running block' marker, say, hyperparameters,
and small helpers used by code generated from Blockly's built-in blocks."""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import queue
import random as _random
import re
import sys
import threading
import time
import warnings

from .errors import NBError, StopProgram
from .events import LineWriter, emit, emitter

__all__ = [
    "setup", "config", "at", "say", "param", "wait", "stop", "finish", "timer", "number", "text",
    "join", "append", "as_list", "list_get", "list_pop", "list_set", "list_insert", "sublist",
    "sort_list", "index_of", "list_stat", "random_int", "random_float", "trig", "is_prime", "count",
    "plot_point", "plot_list", "sound", "check", "math", "device", "is_quick",
]


class State:
    def __init__(self):
        self.project = "Untitled"
        self.device_override = os.environ.get("NEUROBLOCKS_DEVICE") or None
        self.device_pref = "auto"
        self._device = None
        self.seed = None
        self.quick = os.environ.get("NEUROBLOCKS_QUICK", "") not in ("", "0", "false")
        try:
            self.params = json.loads(os.environ.get("NEUROBLOCKS_PARAMS") or "{}")
        except json.JSONDecodeError:
            self.params = {}
        self.current_block = None
        self.sent_block = None
        self.skip = threading.Event()
        self.interact_q: "queue.Queue[dict]" = queue.Queue()
        self.keys: dict = {}
        self.checks: list[tuple[bool, str]] = []
        self.setup_done = False
        self.start = time.time()
        self.exit_code = 0
        self.stdin_open = False
        self.objects: dict = {}


STATE = State()


def is_quick() -> bool:
    return STATE.quick


def quick_cap(n, cap):
    """In quick-test mode, shrink a loop count."""
    if STATE.quick and n is not None:
        return min(n, cap)
    return n


def gui_mode() -> bool:
    return emitter().mode == "gui"


# ---------------------------------------------------------------------------
# Setup & device
# ---------------------------------------------------------------------------


def _block_flusher():
    while True:
        time.sleep(0.04)
        cur = STATE.current_block
        if cur != STATE.sent_block:
            STATE.sent_block = cur
            emit("block", id=cur)


def _handle_control(line: bytes):
    line = line.strip()
    if not line:
        return
    try:
        msg = json.loads(line.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return
    cmd = msg.get("cmd")
    if cmd == "skip":
        STATE.skip.set()
    elif cmd == "keys":
        STATE.keys = msg.get("keys") or {}
    elif cmd in ("interact", "interact_end"):
        STATE.interact_q.put(msg)


def _stdin_poller():
    """Return a function giving the number of bytes waiting on stdin, without ever blocking.

    A thread blocked in a read on the stdin pipe deadlocks other threads that merely
    query the handle on Windows (many libraries call isatty() while importing), so we
    poll instead of reading blindly.
    """
    fd = sys.stdin.fileno()
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                           ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        kernel32.PeekNamedPipe.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(fd)
        avail = wintypes.DWORD()

        def ready() -> int:
            if not kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None):
                raise EOFError
            return int(avail.value)
    else:
        import select

        def ready() -> int:
            r, _, _ = select.select([fd], [], [], 0)
            return 65536 if r else 0
    return fd, ready


def _stdin_reader():
    STATE.stdin_open = True
    buf = b""
    try:
        fd, ready = _stdin_poller()
        while True:
            n = ready()
            if not n:
                time.sleep(0.02)
                continue
            chunk = os.read(fd, n)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                _handle_control(line)
    except (EOFError, OSError, ValueError, AttributeError):
        pass
    finally:
        STATE.stdin_open = False
        STATE.interact_q.put({"cmd": "interact_end", "reason": "stdin closed"})


def setup(project: str | None = None, **_ignored):
    """Called at the top of every generated program."""
    if project:
        STATE.project = project
    if STATE.setup_done:
        return
    STATE.setup_done = True
    em = emitter()
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    if em.mode == "gui":
        sys.stdout = LineWriter("info")
        threading.Thread(target=_block_flusher, daemon=True, name="nb-blocks").start()
        threading.Thread(target=_stdin_reader, daemon=True, name="nb-stdin").start()
    elif em.mode == "cli":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(errors="replace")
            except (AttributeError, ValueError):
                pass
        if em.run_dir is None and not os.environ.get("NEUROBLOCKS_NO_RUN_DIR"):
            from .. import paths
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            slug = re.sub(r"[^A-Za-z0-9]+", "-", STATE.project).strip("-")[:40] or "run"
            em.set_run_dir(paths.runs_dir() / f"{stamp}-{slug}")
    emit("start", project=STATE.project, quick=STATE.quick, params=STATE.params,
         run_dir=str(em.run_dir) if em.run_dir else None)
    if STATE.quick:
        emit("log", level="info", text="Quick test mode: every training loop is shortened to a few steps.")


def config(device: str = "auto", seed=None):
    """The 'use device … random seed …' block."""
    STATE.device_pref = device or "auto"
    STATE._device = None
    if seed is not None:
        seed_everything(int(seed))


def seed_everything(seed: int):
    STATE.seed = seed
    _random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def device():
    """The torch device programs should compute on (resolved once, lazily)."""
    if STATE._device is not None:
        return STATE._device
    import torch
    pref = (STATE.device_override or STATE.device_pref or "auto").lower()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if pref == "auto":
        pref = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        emit("log", level="warn", text="No CUDA GPU found — using the CPU instead.")
        pref = "cpu"
    if pref == "mps" and not has_mps:
        emit("log", level="warn", text="No Apple GPU (MPS) found — using the CPU instead.")
        pref = "cpu"
    dev = torch.device(pref)
    STATE._device = dev
    if dev.type == "cuda":
        idx = dev.index or 0
        props = torch.cuda.get_device_properties(idx)
        label = f"GPU: {props.name} ({props.total_memory / 2**30:.0f} GB)"
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    elif dev.type == "mps":
        label = "Apple GPU (MPS)"
    else:
        label = f"CPU ({torch.get_num_threads()} threads)"
    emit("device", device=dev.type, label=label)
    return dev


# ---------------------------------------------------------------------------
# Program flow
# ---------------------------------------------------------------------------


def at(block_id: str):
    """Marks which block is running (the editor makes it glow)."""
    STATE.current_block = block_id


def current_block():
    return STATE.current_block


def say(*values):
    msg = " ".join(text(v) for v in values)
    emit("say", text=msg)


def param(name: str, default):
    """A hyperparameter that can be overridden with ``--set name=value``."""
    if name in STATE.params:
        raw = STATE.params[name]
        try:
            if isinstance(default, bool):
                value = str(raw).lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int) and not isinstance(default, bool):
                value = int(float(raw))
            elif isinstance(default, float):
                value = float(raw)
            else:
                value = raw
        except (TypeError, ValueError):
            value = raw
        emit("log", level="info", text=f"hyperparameter {name} = {text(value)} (set from outside)")
        return value
    return default


def wait(seconds):
    seconds = float(number(seconds))
    if STATE.quick:
        seconds = min(seconds, 0.05)
    end = time.time() + max(0.0, seconds)
    while time.time() < end:
        time.sleep(min(0.05, end - time.time()))


def stop():
    raise StopProgram()


def finish():
    """Called at the end of every generated program."""
    if STATE.checks:
        passed = sum(1 for ok, _ in STATE.checks if ok)
        total = len(STATE.checks)
        emit("check_summary", passed=passed, total=total)
        if passed < total and STATE.quick:
            emit("log", level="info", text=f"{total - passed} of {total} checks failed — expected in quick test mode, "
                                           f"where training is cut short (they don't count).")
        elif passed < total:
            STATE.exit_code = 1
            emit("log", level="warn", text=f"{total - passed} of {total} checks failed.")
        else:
            emit("log", level="info", text=f"All {total} checks passed.")


def timer() -> float:
    return round(time.time() - STATE.start, 3)


def register(obj):
    name = getattr(obj, "name", None)
    if name:
        STATE.objects[name] = obj
    return obj


# ---------------------------------------------------------------------------
# Value helpers (used by generated code)
# ---------------------------------------------------------------------------


def number(x) -> float:
    """Coerce to a number like Scratch does (empty/None -> 0)."""
    if x is None or x == "":
        return 0
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return x
    try:
        return float(x)
    except (TypeError, ValueError):
        item = getattr(x, "item", None)
        if item is not None:
            try:
                return item()
            except Exception:
                pass
        return 0


def text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return f"{x:.6g}" if (abs(x) >= 1e-4 or x == 0) else f"{x:.3e}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(text(v) for v in x) + "]"
    if hasattr(x, "describe") and callable(x.describe):
        try:
            return x.describe()
        except Exception:
            pass
    item = getattr(x, "item", None)
    if item is not None and getattr(x, "ndim", 1) == 0:
        try:
            return text(item())
        except Exception:
            pass
    return str(x)


def join(*parts) -> str:
    return "".join(text(p) for p in parts)


def as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        return list(x)
    tolist = getattr(x, "tolist", None)
    if tolist:
        v = tolist()
        return v if isinstance(v, list) else [v]
    return [x]


def append(lst, item) -> list:
    lst = [] if lst is None or lst == "" else (lst if isinstance(lst, list) else as_list(lst))
    lst.append(item)
    return lst


def _index(lst, where, at):
    n = len(lst)
    if where == "first":
        return 0
    if where == "last":
        return n - 1
    if where == "random":
        return _random.randrange(n) if n else 0
    at = int(number(at))
    if where == "from_end":
        return n - at
    return at - 1


def list_get(lst, where="from_start", at=1):
    lst = as_list(lst)
    if not lst:
        raise NBError("Tried to get an item from an empty list.")
    i = _index(lst, where, at)
    if not 0 <= i < len(lst):
        raise NBError(f"The list has {len(lst)} items, so there is no item number {at}.")
    return lst[i]


def list_pop(lst, where="from_start", at=1):
    if not isinstance(lst, list) or not lst:
        raise NBError("Tried to remove an item from an empty list.")
    i = _index(lst, where, at)
    if not 0 <= i < len(lst):
        raise NBError(f"The list has {len(lst)} items, so there is no item number {at}.")
    return lst.pop(i)


def list_set(lst, where, at, value):
    if not isinstance(lst, list):
        raise NBError("That variable doesn't hold a list.")
    i = _index(lst, where, at)
    if not 0 <= i < len(lst):
        raise NBError(f"The list has {len(lst)} items, so there is no item number {at}.")
    lst[i] = value


def list_insert(lst, where, at, value):
    if not isinstance(lst, list):
        raise NBError("That variable doesn't hold a list.")
    i = len(lst) if where == "last" else max(0, _index(lst, where, at) + (1 if where == "from_end" else 0))
    lst.insert(i, value)


def sublist(lst, w1, a1, w2, a2):
    lst = as_list(lst)
    i = _index(lst, w1, a1)
    j = _index(lst, w2, a2)
    return lst[max(0, i): j + 1]


def sort_list(lst, kind="numeric", reverse=False):
    lst = as_list(lst)
    if kind == "numeric":
        key = number
    elif kind == "ignore_case":
        key = lambda v: text(v).lower()  # noqa: E731
    else:
        key = text
    return sorted(lst, key=key, reverse=reverse)


def index_of(lst, item, last=False) -> int:
    lst = as_list(lst)
    try:
        if last:
            return len(lst) - lst[::-1].index(item)
        return lst.index(item) + 1
    except ValueError:
        return 0


def list_stat(op: str, lst):
    vals = [number(v) for v in as_list(lst)]
    if op == "random":
        return _random.choice(vals) if vals else None
    if not vals:
        return 0
    if op == "sum":
        return sum(vals)
    if op == "min":
        return min(vals)
    if op == "max":
        return max(vals)
    if op == "average":
        return sum(vals) / len(vals)
    if op == "median":
        s = sorted(vals)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
    if op == "mode":
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values())
        return [v for v, c in counts.items() if c == best]
    if op == "std_dev":
        mean = sum(vals) / len(vals)
        return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    raise NBError(f"Unknown list operation {op}")


def random_int(a, b) -> int:
    a, b = int(number(a)), int(number(b))
    if a > b:
        a, b = b, a
    return _random.randint(a, b)


def random_float() -> float:
    return _random.random()


def trig(op: str, degrees):
    x = number(degrees)
    if op in ("sin", "cos", "tan"):
        return getattr(math, op)(math.radians(x))
    return math.degrees(getattr(math, op)(x))


def is_prime(n) -> bool:
    n = number(n)
    if n != int(n) or n < 2:
        return False
    n = int(n)
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def count(a, b, step=1):
    a, b, step = number(a), number(b), abs(number(step)) or 1
    if a <= b:
        while a <= b + 1e-12:
            yield a
            a += step
    else:
        while a >= b - 1e-12:
            yield a
            a -= step


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def plot_point(chart, series, x, y):
    emit("metric", chart=text(chart), series=text(series), x=number(x), y=number(y), group="custom")


def plot_list(values, kind="line", title="My chart"):
    vals = [number(v) for v in as_list(values)]
    emit("plot", kind=kind, title=text(title), values=vals)


def sound(name="ding"):
    emit("sound", name=str(name))


_CHECK_OPS = {
    ">=": lambda a, b: a >= b, ">": lambda a, b: a > b, "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}
_OP_TEXT = {">=": "≥", "<=": "≤", "==": "=", "!=": "≠", ">": ">", "<": "<"}


def check(value, op: str, target, label: str | None = None) -> bool:
    """The 'check that … ' test block."""
    v = value
    try:
        a, b = number(v), number(target)
        ok = bool(_CHECK_OPS[op](a, b))
    except Exception:
        a, b = v, target
        ok = bool(_CHECK_OPS.get(op, lambda x, y: x == y)(v, target))
    desc = f"{label + ': ' if label else ''}{text(a)} {_OP_TEXT.get(op, op)} {text(b)}"
    STATE.checks.append((ok, desc))
    emit("check", passed=ok, text=desc, index=len(STATE.checks), block=STATE.current_block)
    return ok
