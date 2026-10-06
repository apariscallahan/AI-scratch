"""Run block programs on a remote NeuroBlocks server (e.g. a cloud GPU) with live events.

The local editor server acts as a proxy: it uploads the project's data files, starts
the run remotely, relays the remote event stream into the local editor, and copies
saved models and the run log back when the run finishes.
"""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

from .. import paths
from ..project import referenced_files, slugify
from .runner import Run, RunManager


class RemoteError(Exception):
    pass


class RemoteManager:
    def __init__(self, runs: RunManager):
        self.runs = runs

    # -- config ---------------------------------------------------------------
    @property
    def cfg_path(self) -> Path:
        return paths.home() / "remotes.json"

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save(self, targets: list[dict]):
        self.cfg_path.write_text(json.dumps(targets, indent=1), encoding="utf-8")

    def list_targets(self) -> list[dict]:
        return [{k: v for k, v in t.items() if k != "token"} | {"has_token": bool(t.get("token"))}
                for t in self._load()]

    def get(self, target_id: str) -> dict:
        for t in self._load():
            if t["id"] == target_id:
                return t
        raise RemoteError(f"Unknown remote server '{target_id}'.")

    def add_target(self, name: str, url: str, token: str) -> dict:
        url = url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        t = {"id": uuid.uuid4().hex[:8], "name": name.strip() or urllib.parse.urlparse(url).netloc, "url": url,
             "token": token.strip()}
        ping = self._json(t, "GET", "/api/ping")
        if ping.get("app") != "neuroblocks":
            raise RemoteError("That address doesn't look like a NeuroBlocks server.")
        info = self._json(t, "GET", "/api/info")
        t["device"] = info.get("device")
        targets = [x for x in self._load() if x["url"] != url] + [t]
        self._save(targets)
        return {k: v for k, v in t.items() if k != "token"}

    def remove_target(self, target_id: str):
        self._save([t for t in self._load() if t["id"] != target_id])

    # -- HTTP ----------------------------------------------------------------------
    def _request(self, t: dict, method: str, path: str, body=None, raw: bytes | None = None,
                 timeout: float = 60) -> bytes:
        url = t["url"] + path
        headers = {"User-Agent": "NeuroBlocks"}
        if t.get("token"):
            headers["Authorization"] = f"Bearer {t['token']}"
        data = None
        if raw is not None:
            data = raw
            headers["Content-Type"] = "application/octet-stream"
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code == 401:
                raise RemoteError("The remote server rejected the token.") from None
            raise RemoteError(f"HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise RemoteError(f"Can't reach {t['url']}: {e.reason}") from None

    def _json(self, t, method, path, body=None, timeout=60):
        return json.loads(self._request(t, method, path, body=body, timeout=timeout).decode("utf-8") or "{}")

    # -- runs ------------------------------------------------------------------------
    def start(self, target_id: str, workspace: dict, project: str, quick: bool, params: dict) -> Run:
        t = self.get(target_id)
        self.runs.stop_current()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = paths.runs_dir() / f"{stamp}-{slugify(project, 'run')}-remote"
        run_dir.mkdir(parents=True, exist_ok=True)
        run = Run(run_dir.name, project, run_dir)
        run.remote = {"target": t["id"], "name": t["name"], "url": t["url"], "run": None}
        run.status = "running"
        with self.runs.lock:
            self.runs.runs[run.id] = run
            self.runs.current = run
        self.runs.publish(run, {"type": "run_start", "run": run.summary(), "quick": quick})
        self.runs.publish(run, {"type": "log", "level": "info", "text": f"Running on remote server '{t['name']}' "
                                                                       f"({t['url']})…"})
        # Upload the data files the project uses.
        for src, rel in referenced_files(workspace):
            self.runs.publish(run, {"type": "log", "level": "info", "text": f"Uploading {rel}…"})
            q = urllib.parse.urlencode({"path": rel})
            self._request(t, "POST", f"/api/upload?{q}", raw=src.read_bytes(), timeout=600)
        ready = threading.Event()
        threading.Thread(target=self._relay, args=(t, run, ready), daemon=True).start()
        ready.wait(10)
        resp = self._json(t, "POST", "/api/run", {"workspace": workspace, "project": project, "quick": quick,
                                                  "params": params})
        if not resp.get("ok"):
            for d in resp.get("diagnostics") or []:
                if d.get("level") == "error":
                    self.runs.publish(run, {"type": "error", "message": d.get("message"), "block": d.get("block_id")})
            self.runs.publish(run, {"type": "done", "status": "error", "exit_code": 1})
            return run
        run.remote["run"] = resp["run"]["id"]
        return run

    def _relay(self, t: dict, run: Run, ready: threading.Event):
        try:
            from websockets.sync.client import connect
        except ImportError:
            ready.set()
            self.runs.publish(run, {"type": "error", "message": "The 'websockets' package is needed for remote runs."})
            return
        ws_url = t["url"].replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        if t.get("token"):
            ws_url += "?" + urllib.parse.urlencode({"token": t["token"]})
        pending: list[dict] = []
        try:
            with connect(ws_url, max_size=None, open_timeout=15) as ws:
                ready.set()
                ws.recv()  # hello (backlog belongs to an older run)
                while True:
                    msg = json.loads(ws.recv())
                    rid = run.remote.get("run")
                    if rid is None:
                        pending.append(msg)
                        continue
                    batch = pending + [msg]
                    pending = []
                    finished = False
                    for m in batch:
                        if m.get("run") != rid:
                            continue
                        ev = m.get("event") or {}
                        if ev.get("type") == "run_start":
                            continue
                        if ev.get("type") == "file" and ev.get("kind") == "model":
                            self._fetch_model(t, run, ev)
                        if ev.get("type") == "done":
                            finished = True
                            self._fetch_run(t, run, rid)
                        self.runs.publish(run, ev)
                    if finished:
                        break
        except Exception as e:  # noqa: BLE001
            ready.set()
            if not run.done_sent:
                self.runs.publish(run, {"type": "error", "message": f"Lost connection to the remote server: {e}"})
                self.runs.publish(run, {"type": "done", "status": "error", "exit_code": 1})

    def _fetch_model(self, t, run, ev):
        """Copy a model the remote run saved into the local models folder."""
        try:
            remote_path = Path(ev["path"])
            rel = "models/" + remote_path.name
            data = self._request(t, "GET", "/api/files/download?" + urllib.parse.urlencode({"path": rel}), timeout=600)
            dest = paths.models_dir() / remote_path.name
            dest.write_bytes(data)
            self.runs.publish(run, {"type": "log", "level": "success",
                                    "text": f"Copied the trained model to this computer: {dest}"})
        except Exception as e:  # noqa: BLE001
            self.runs.publish(run, {"type": "log", "level": "warn", "text": f"Couldn't copy the model back: {e}"})

    def _fetch_run(self, t, run, rid):
        try:
            data = self._request(t, "GET", f"/api/runs/{urllib.parse.quote(rid)}/download", timeout=600)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m in z.infolist():
                    mp = Path(m.filename)
                    if mp.is_absolute() or ".." in mp.parts:
                        continue
                    z.extract(m, run.dir)
        except Exception:  # noqa: BLE001
            pass

    def stop(self, run: Run):
        rid = (run.remote or {}).get("run")
        if rid:
            try:
                self._json(self.get(run.remote["target"]), "POST", "/api/stop", {"run": rid})
            except Exception:  # noqa: BLE001
                pass

    def control(self, run: Run, msg: dict) -> bool:
        try:
            return bool(self._json(self.get(run.remote["target"]), "POST", "/api/control", msg, timeout=10).get("ok"))
        except Exception:  # noqa: BLE001
            return False
