"""The NeuroBlocks editor server (FastAPI)."""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__, paths
from ..compiler import compile_workspace
from ..compiler.toolbox import editor_bundle
from ..project import load_project, make_bundle, new_project, save_project, slugify
from .runner import RunManager

OPTIONAL = ["transformers", "datasets", "tiktoken", "gymnasium", "accelerate"]


def _device_probe() -> dict:
    code = (
        "import json, torch, os\n"
        "d = {'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'threads': torch.get_num_threads()}\n"
        "d['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if d['cuda'] else []\n"
        "d['gpu_mem'] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) "
        "for i in range(torch.cuda.device_count())] if d['cuda'] else []\n"
        "m = getattr(torch.backends, 'mps', None)\n"
        "d['mps'] = bool(m and m.is_available())\n"
        "print(json.dumps(d))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _safe_rel(path: str) -> Path:
    p = Path(str(path).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(400, "Bad path")
    return p


def create_app(token: str | None = None, open_browser_hint: bool = False) -> FastAPI:
    clients: set[asyncio.Queue] = set()
    state: dict = {"loop": None, "device": None, "token": token}

    @asynccontextmanager
    async def lifespan(_app):
        state["loop"] = asyncio.get_running_loop()

        def probe():
            state["device"] = _device_probe()
        threading.Thread(target=probe, daemon=True).start()
        threading.Thread(target=runs.spawn_warm, daemon=True).start()
        yield
        runs.shutdown()

    app = FastAPI(title="NeuroBlocks", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)

    def publish(msg: dict):
        loop = state["loop"]
        if loop is None:
            return
        for q in list(clients):
            loop.call_soon_threadsafe(_put, q, msg)

    def _put(q: asyncio.Queue, msg):
        if q.qsize() < 20000:
            q.put_nowait(msg)

    runs = RunManager(publish)
    from .remote import RemoteManager
    remotes = RemoteManager(runs)
    app.state.runs = runs
    app.state.remotes = remotes

    # -- auth ------------------------------------------------------------------
    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if state["token"] and request.url.path.startswith("/api/") and request.url.path != "/api/ping":
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or \
                       request.query_params.get("token", "")
            if supplied != state["token"]:
                return JSONResponse({"detail": "Missing or wrong token"}, status_code=401)
        return await call_next(request)

    # -- pages -------------------------------------------------------------------
    @app.get("/")
    async def index():
        return FileResponse(paths.STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=str(paths.STATIC_DIR)), name="static")

    @app.get("/api/ping")
    async def ping():
        return {"ok": True, "app": "neuroblocks", "version": __version__, "auth": bool(state["token"])}

    @app.get("/api/info")
    async def info():
        return {
            "version": __version__, "python": sys.version.split()[0], "home": str(paths.home()),
            "device": state["device"], "optional": {m: importlib.util.find_spec(m) is not None for m in OPTIONAL},
            "platform": sys.platform, "auth": bool(state["token"]),
        }

    _bundle_cache: dict = {}

    @app.get("/api/blocks")
    async def blocks():
        if "b" not in _bundle_cache:
            _bundle_cache["b"] = editor_bundle()
        return _bundle_cache["b"]

    # -- compile & run ------------------------------------------------------------
    @app.post("/api/compile")
    async def compile_(body: dict):
        res = compile_workspace(body.get("workspace") or {}, markers=bool(body.get("markers", False)),
                                project=body.get("project") or "Untitled")
        return res.to_json()

    @app.post("/api/run")
    async def run(body: dict):
        ws = body.get("workspace") or {}
        project = body.get("project") or "Untitled"
        target = body.get("target") or "local"
        if target != "local":
            try:
                r = await asyncio.to_thread(remotes.start, target, ws, project, bool(body.get("quick")),
                                            body.get("params") or {})
            except Exception as e:  # noqa: BLE001
                raise HTTPException(502, f"Remote server: {e}")
            return {"ok": True, "run": r.summary()}
        res = compile_workspace(ws, markers=True, project=project)
        if not res.ok:
            return {"ok": False, "diagnostics": [d.to_json() for d in res.diagnostics]}
        r = await asyncio.to_thread(runs.start, res.code, res.names, project, ws, bool(body.get("quick")),
                                    body.get("params") or {}, body.get("device"))
        return {"ok": True, "run": r.summary(), "diagnostics": [d.to_json() for d in res.diagnostics]}

    @app.post("/api/stop")
    async def stop(body: dict | None = None):
        run_id = (body or {}).get("run")
        r = runs.runs.get(run_id) if run_id else runs.current
        if r is not None and r.remote:
            await asyncio.to_thread(remotes.stop, r)
            return {"ok": True}
        return {"ok": await asyncio.to_thread(runs.stop, run_id)}

    @app.post("/api/control")
    async def control(body: dict):
        return {"ok": _control(body)}

    def _control(msg: dict) -> bool:
        r = runs.current
        if r is not None and r.remote:
            return remotes.control(r, msg)
        return runs.control(msg)

    @app.get("/api/runs")
    async def list_runs():
        out = []
        dirs = sorted((d for d in paths.runs_dir().iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
        for d in dirs[:60]:
            r = runs.runs.get(d.name)
            item = {"id": d.name, "status": r.status if r else None, "has_events": (d / "events.jsonl").exists()}
            pj = d / "project.nblk"
            if pj.exists():
                try:
                    item["project"] = json.loads(pj.read_text(encoding="utf-8")).get("name")
                except (OSError, json.JSONDecodeError):
                    pass
            out.append(item)
        return out

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        p = paths.runs_dir() / _safe_rel(run_id) / "events.jsonl"
        if not p.exists():
            raise HTTPException(404, "No events for that run")
        return FileResponse(p, media_type="text/plain; charset=utf-8")

    @app.get("/api/runs/{run_id}/project")
    async def run_project(run_id: str):
        p = paths.runs_dir() / _safe_rel(run_id) / "project.nblk"
        if not p.exists():
            raise HTTPException(404, "No project saved with that run")
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    @app.get("/api/runs/{run_id}/download")
    async def run_download(run_id: str):
        d = paths.runs_dir() / _safe_rel(run_id)
        if not d.is_dir():
            raise HTTPException(404, "No such run")
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
            for f in d.rglob("*"):
                if f.is_file():
                    z.write(f, str(f.relative_to(d)))
        return Response(bio.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})

    # -- websocket ----------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        if state["token"] and ws.query_params.get("token") != state["token"]:
            await ws.close(code=4401)
            return
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        clients.add(q)
        cur = runs.current
        hello = {"type": "hello", "version": __version__, "current": cur.summary() if cur else None}
        if cur is not None:
            with cur.lock:
                hello["backlog"] = list(cur.events)
        await ws.send_text(json.dumps(hello))

        async def sender():
            while True:
                msg = await q.get()
                await ws.send_text(json.dumps(msg))

        task = asyncio.create_task(sender())
        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if msg.get("cmd"):
                    await asyncio.to_thread(_control, msg)
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
            clients.discard(q)

    # -- projects -------------------------------------------------------------------
    @app.get("/api/projects")
    async def list_projects():
        out = []
        for p in sorted(paths.projects_dir().glob("*.nblk"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({"file": p.name, "name": d.get("name", p.stem), "description": d.get("description", ""),
                        "modified": p.stat().st_mtime})
        return out

    @app.get("/api/projects/{file}")
    async def get_project(file: str):
        p = paths.projects_dir() / _safe_rel(file).name
        if not p.exists():
            raise HTTPException(404, "No such project")
        return JSONResponse(load_project(p))

    @app.put("/api/projects/{file}")
    async def put_project(file: str, body: dict):
        name = _safe_rel(file).name
        if not name.endswith(".nblk"):
            name += ".nblk"
        if "workspace" not in body:
            raise HTTPException(400, "Missing workspace")
        p = save_project(paths.projects_dir() / name, body)
        return {"ok": True, "file": p.name}

    @app.delete("/api/projects/{file}")
    async def delete_project(file: str):
        p = paths.projects_dir() / _safe_rel(file).name
        if p.exists():
            trash = paths.projects_dir() / ".trash"
            trash.mkdir(exist_ok=True)
            shutil.move(str(p), str(trash / f"{int(time.time())}-{p.name}"))
        return {"ok": True}

    # -- examples ---------------------------------------------------------------------
    @app.get("/api/examples")
    async def list_examples():
        out = []
        for p in sorted(paths.EXAMPLES_DIR.glob("*.nblk")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({"id": p.stem, "name": d.get("name", p.stem), "description": d.get("description", ""),
                        "category": d.get("category", "Other"), "level": d.get("level", ""),
                        "order": d.get("order", 99), "minutes": d.get("minutes")})
        out.sort(key=lambda e: (e["order"], e["name"]))
        return out

    @app.get("/api/examples/{ex_id}")
    async def get_example(ex_id: str):
        p = paths.EXAMPLES_DIR / f"{_safe_rel(ex_id).name}.nblk"
        if not p.exists():
            raise HTTPException(404, "No such example")
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    # -- export -------------------------------------------------------------------------
    @app.post("/api/export/python")
    async def export_python(body: dict):
        project = body.get("project") or "Untitled"
        res = compile_workspace(body.get("workspace") or {}, markers=False, project=project)
        return {"code": res.code, "filename": f"{slugify(project)}.py", "ok": res.ok,
                "diagnostics": [d.to_json() for d in res.diagnostics]}

    @app.post("/api/export/bundle")
    async def export_bundle(body: dict):
        proj = new_project(body.get("project") or "Untitled", body.get("workspace") or {},
                           body.get("description") or "")
        bio = io.BytesIO()
        try:
            summary = await asyncio.to_thread(make_bundle, proj, bio)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return Response(bio.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{summary["name"]}"',
                                 "X-Bundle-Files": json.dumps(summary["files"])})

    # -- files ----------------------------------------------------------------------------
    @app.get("/api/files")
    async def list_files():
        def ls(root: Path, kind: str):
            out = []
            for f in sorted(root.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    out.append({"path": str(f.relative_to(paths.home())).replace("\\", "/"),
                                "name": str(f.relative_to(root)).replace("\\", "/"), "size": f.stat().st_size,
                                "kind": kind, "modified": f.stat().st_mtime})
            return out[:500]
        return {"home": str(paths.home()), "data": ls(paths.data_dir(), "data"),
                "models": ls(paths.models_dir(), "model")}

    @app.post("/api/upload")
    async def upload(request: Request, path: str):
        rel = _safe_rel(path)
        if rel.parts[0] not in ("data", "models"):
            rel = Path("data") / rel
        dest = paths.home() / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = await request.body()
        if rel.suffix.lower() == ".zip" and request.query_params.get("unzip") == "1":
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                target = dest.with_suffix("")
                for m in z.infolist():
                    mp = Path(m.filename)
                    if mp.is_absolute() or ".." in mp.parts:
                        continue
                    z.extract(m, target)
            return {"ok": True, "path": str(target.relative_to(paths.home())).replace("\\", "/")}
        dest.write_bytes(body)
        return {"ok": True, "path": str(rel).replace("\\", "/"), "size": len(body)}

    @app.get("/api/files/download")
    async def download_file(path: str):
        rel = _safe_rel(path)
        p = paths.home() / rel
        if not p.is_file() or rel.parts[0] not in ("data", "models", "runs"):
            raise HTTPException(404, "No such file")
        return FileResponse(p, filename=p.name)

    # -- remote targets ---------------------------------------------------------------------
    @app.get("/api/targets")
    async def list_targets():
        return remotes.list_targets()

    @app.post("/api/targets")
    async def add_target(body: dict):
        try:
            t = await asyncio.to_thread(remotes.add_target, body.get("name") or "", body.get("url") or "",
                                        body.get("token") or "")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e))
        return t

    @app.delete("/api/targets/{target_id}")
    async def delete_target(target_id: str):
        remotes.remove_target(target_id)
        return {"ok": True}

    return app
