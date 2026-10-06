"""Runs a generated program and reports how it ended.

Used by the editor (``python -m neuroblocks.runtime.harness script.py`` in a
subprocess, events on stdout) and by ``neuroblocks run`` (in-process, pretty
console output).
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path


def run_script(script: str | Path, names: dict | None = None) -> int:
    from . import core
    from .errors import StopProgram, explain, short_traceback
    from .events import emit

    script = str(script)
    if names is None:
        side = Path(script).with_suffix(".names.json")
        if side.exists():
            try:
                names = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                names = {}
    names = names or {}
    status = "finished"
    t0 = time.time()
    old_argv = sys.argv
    sys.argv = [script]
    sys.path.insert(0, str(Path(script).resolve().parent))
    try:
        runpy.run_path(script, run_name="__main__")
    except StopProgram:
        status = "stopped"
        emit("log", level="info", text="Program stopped.")
    except KeyboardInterrupt:
        status = "stopped"
        emit("log", level="warn", text="Interrupted.")
    except SystemExit as e:
        if e.code not in (None, 0):
            status = "error"
    except BaseException as e:  # noqa: BLE001 - report everything to the user
        status = "error"
        msg, hint = explain(e, names)
        block = getattr(e, "nb_block", None) or core.STATE.current_block
        emit("error", message=msg, hint=hint, block=block, traceback=short_traceback(e))
    finally:
        sys.argv = old_argv
        out = sys.modules.get("neuroblocks.runtime.output")
        if out is not None:
            out.flush()  # show_text updates still held back
    elapsed = round(time.time() - t0, 2)
    code = 0 if status in ("finished", "stopped") else 1
    if status == "finished" and core.STATE.exit_code:
        code = core.STATE.exit_code
    core.STATE.current_block = None
    emit("block", id=None)
    emit("done", status=status, seconds=elapsed, exit_code=code)
    return code


def warm_main():
    """A pre-warmed worker: import the heavy libraries now, then wait for a job on stdin.

    The editor keeps one of these ready so pressing Run starts instantly instead of
    spending ~10 s importing PyTorch (and the compiler stack its optimizers pull in).
    """
    try:
        import numpy  # noqa: F401
        import torch
        try:
            import torch._dynamo  # noqa: F401  (imported lazily by the first optimizer — slow)
        except Exception:  # noqa: BLE001
            pass
        torch.optim.Adam([torch.zeros(1, requires_grad=True)])
        import sklearn.tree  # noqa: F401
        from . import data, evaluate, generate, layers, model, train  # noqa: F401
        try:
            import pymunk  # noqa: F401
        except ImportError:
            pass
    except Exception:  # noqa: BLE001 - a cold start will still work
        pass
    # Read the job line byte by byte (unbuffered), so that control messages sent right
    # after it stay in the pipe for the runtime's stdin poller.
    fd = sys.stdin.fileno()
    raw = bytearray()
    while True:
        ch = os.read(fd, 1)
        if not ch:
            os._exit(0)
        if ch == b"\n":
            break
        raw += ch
    if not raw.strip():
        os._exit(0)
    job = json.loads(raw.decode("utf-8"))
    os.environ.update({k: str(v) for k, v in (job.get("env") or {}).items()})
    if job.get("cwd"):
        os.chdir(job["cwd"])
    from . import core, events
    core.STATE.__init__()  # re-read NEUROBLOCKS_QUICK / PARAMS / DEVICE for this job
    events.reset_emitter()
    code = run_script(job["script"])
    try:
        sys.__stdout__.flush()
    except Exception:
        pass
    os._exit(code)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--warm":
        warm_main()
        return 0
    if not argv:
        print("usage: python -m neuroblocks.runtime.harness SCRIPT.py", file=sys.stderr)
        return 2
    os.environ.setdefault("NEUROBLOCKS_EVENTS", "gui")
    code = run_script(argv[0])
    try:
        sys.__stdout__.flush()
    except Exception:
        pass
    # Daemon threads (stdin reader, block flusher) must not keep us alive.
    os._exit(code)


if __name__ == "__main__":
    main()
