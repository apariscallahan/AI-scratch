"""Runs block programs in a subprocess and relays their events."""
from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from .. import paths
from ..project import slugify

PREFIX = "\x1e"
MAX_EVENTS = 60000
KEEP_REPLAYS = 12


class Run:
    def __init__(self, run_id: str, project: str, run_dir: Path):
        self.id = run_id
        self.project = project
        self.dir = run_dir
        self.proc: subprocess.Popen | None = None
        self.status = "starting"
        self.started = time.time()
        self.ended: float | None = None
        self.exit_code: int | None = None
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.done_sent = False
        self.remote = None  # set for runs proxied to a remote server

    def summary(self) -> dict:
        return {"id": self.id, "project": self.project, "status": self.status, "started": self.started,
                "ended": self.ended, "exit_code": self.exit_code, "dir": str(self.dir),
                "remote": self.remote}


class RunManager:
    def __init__(self, publish: Callable[[dict], None], warm: bool = True):
        self.publish_cb = publish
        self.runs: dict[str, Run] = {}
        self.current: Run | None = None
        self.lock = threading.Lock()
        self.use_warm = warm and not os.environ.get("NEUROBLOCKS_NO_WARM")
        self.warm: subprocess.Popen | None = None
        self.warm_lock = threading.Lock()

    # -- pre-warmed worker ------------------------------------------------------
    @staticmethod
    def _popen_kwargs():
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        return {"start_new_session": True}

    def _base_env(self) -> dict:
        env = dict(os.environ)
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "NEUROBLOCKS_HOME": str(paths.home())})
        return env

    def spawn_warm(self):
        """Start a worker that imports PyTorch now and waits for the next program."""
        if not self.use_warm:
            return
        with self.warm_lock:
            if self.warm is not None and self.warm.poll() is None:
                return
            try:
                self.warm = subprocess.Popen(
                    [sys.executable, "-u", "-m", "neuroblocks.runtime.harness", "--warm"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(paths.home()),
                    env=self._base_env(), **self._popen_kwargs())
            except OSError:
                self.warm = None

    def _take_warm(self) -> subprocess.Popen | None:
        with self.warm_lock:
            w, self.warm = self.warm, None
        if w is not None and w.poll() is None:
            return w
        return None

    def shutdown(self):
        self.stop_current()
        w = self._take_warm()
        if w is not None:
            kill_tree(w)

    # -- events -------------------------------------------------------------
    def publish(self, run: Run, event: dict):
        with run.lock:
            if event.get("type") == "sim_replay":
                replays = [i for i, e in enumerate(run.events) if e.get("type") == "sim_replay"]
                if len(replays) >= KEEP_REPLAYS:
                    run.events.pop(replays[0])
            run.events.append(event)
            if len(run.events) > MAX_EVENTS:
                # Drop the oldest metric points first (they are the bulk).
                for i, e in enumerate(run.events):
                    if e.get("type") in ("metric", "progress", "block", "text_stream"):
                        run.events.pop(i)
                        break
                else:
                    run.events.pop(0)
            if event.get("type") == "done":
                run.done_sent = True
                run.status = event.get("status", "finished")
                run.exit_code = event.get("exit_code")
                run.ended = time.time()
        self.publish_cb({"run": run.id, "event": event})

    # -- lifecycle --------------------------------------------------------------
    def start(self, code: str, names: dict, project: str, workspace: dict | None = None, quick: bool = False,
              params: dict | None = None, device: str | None = None) -> Run:
        self.stop_current()
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = paths.runs_dir() / f"{stamp}-{slugify(project, 'run')}"
        i = 2
        while run_dir.exists():
            run_dir = paths.runs_dir() / f"{stamp}-{slugify(project, 'run')}-{i}"
            i += 1
        run_dir.mkdir(parents=True)
        script = run_dir / "script.py"
        script.write_text(code, encoding="utf-8")
        (run_dir / "script.names.json").write_text(json.dumps(names), encoding="utf-8")
        if workspace is not None:
            (run_dir / "project.nblk").write_text(json.dumps({"format": "neuroblocks-project", "name": project,
                                                              "workspace": workspace}), encoding="utf-8")
        run = Run(run_dir.name, project, run_dir)
        job_env = {
            "NEUROBLOCKS_EVENTS": "gui",
            "NEUROBLOCKS_RUN_DIR": str(run_dir),
            "NEUROBLOCKS_HOME": str(paths.home()),
            "NEUROBLOCKS_QUICK": "1" if quick else "",
            "NEUROBLOCKS_PARAMS": json.dumps(params or {}),
            "NEUROBLOCKS_DEVICE": device or "",
        }
        proc = self._take_warm()
        if proc is not None:
            try:
                job = {"script": str(script), "env": job_env, "cwd": str(paths.home())}
                proc.stdin.write((json.dumps(job) + "\n").encode("utf-8"))
                proc.stdin.flush()
            except OSError:
                kill_tree(proc)
                proc = None
        if proc is None:
            env = self._base_env()
            env.update(job_env)
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "neuroblocks.runtime.harness", str(script)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(paths.home()), env=env,
                **self._popen_kwargs())
        run.proc = proc
        run.status = "running"
        with self.lock:
            self.runs[run.id] = run
            self.current = run
        self.publish(run, {"type": "run_start", "run": run.summary(), "quick": quick})
        threading.Thread(target=self._read_stdout, args=(run,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(run,), daemon=True).start()
        threading.Thread(target=self._wait, args=(run,), daemon=True).start()
        return run

    def _read_stdout(self, run: Run):
        assert run.proc and run.proc.stdout
        for raw in iter(run.proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            if line.startswith(PREFIX):
                try:
                    ev = json.loads(line[1:])
                except json.JSONDecodeError:
                    continue
                self.publish(run, ev)
            else:
                self.publish(run, {"type": "log", "level": "info", "text": line})

    def _read_stderr(self, run: Run):
        assert run.proc and run.proc.stderr
        buf: list[str] = []
        last = time.time()
        for raw in iter(run.proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.strip():
                continue
            # Progress bars from libraries (e.g. downloads) use \r — keep the last part.
            line = line.split("\r")[-1]
            buf.append(line)
            if len(buf) > 40 or time.time() - last > 0.2:
                self.publish(run, {"type": "log", "level": "stderr", "text": "\n".join(buf)})
                buf, last = [], time.time()
        if buf:
            self.publish(run, {"type": "log", "level": "stderr", "text": "\n".join(buf)})

    def _wait(self, run: Run):
        assert run.proc
        code = run.proc.wait()
        time.sleep(0.3)  # let the readers drain
        threading.Thread(target=self.spawn_warm, daemon=True).start()  # get ready for the next Run
        if not run.done_sent:
            status = "stopped" if run.status == "stopping" else ("finished" if code == 0 else "error")
            if status == "error":
                self.publish(run, {"type": "error", "message": f"The program crashed (exit code {code}).",
                                   "hint": "See the red text in the Console for details."})
            self.publish(run, {"type": "block", "id": None})
            self.publish(run, {"type": "done", "status": status, "exit_code": code,
                               "seconds": round(time.time() - run.started, 2)})

    def stop(self, run_id: str | None = None) -> bool:
        run = self.runs.get(run_id) if run_id else self.current
        if run is None or run.proc is None or run.proc.poll() is not None:
            return False
        run.status = "stopping"
        kill_tree(run.proc)
        return True

    def stop_current(self):
        if self.current and self.current.proc and self.current.proc.poll() is None:
            self.stop(self.current.id)
            try:
                self.current.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def control(self, msg: dict, run_id: str | None = None) -> bool:
        run = self.runs.get(run_id) if run_id else self.current
        if run is None or run.proc is None or run.proc.poll() is not None or run.proc.stdin is None:
            return False
        try:
            run.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            run.proc.stdin.flush()
            return True
        except (OSError, ValueError):
            return False


def kill_tree(proc: subprocess.Popen):
    """Kill a process and its children (scikit-learn/joblib can start worker processes)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass
