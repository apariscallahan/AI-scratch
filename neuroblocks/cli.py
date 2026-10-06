"""The ``neuroblocks`` command.

  neuroblocks gui                       open the block editor in your browser
  neuroblocks run project.nblk          run a project headless (e.g. on a cloud GPU)
  neuroblocks export project.nblk       turn a project into a plain Python script
  neuroblocks bundle project.nblk       make a zip with everything needed on another machine
  neuroblocks serve --host 0.0.0.0      run the editor/run server for remote use (token protected)
  neuroblocks view runs/<run>           open a finished run's charts and replays in the editor
  neuroblocks blocks --markdown         list every block (reference docs)
  neuroblocks examples                  list / copy the example projects
  neuroblocks doctor                    check your setup (PyTorch, GPU, optional packages)
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _print(msg: str = ""):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


# ---------------------------------------------------------------------------
# gui / serve / view
# ---------------------------------------------------------------------------


def _serve(host: str, port: int, token: str | None, open_path: str | None):
    import uvicorn
    from .server.app import create_app

    app = create_app(token=token)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0', '::') else host}:{port}/"
    if open_path is not None:
        target = url + open_path

        def opener():
            time.sleep(1.2)
            webbrowser.open(target)
        threading.Thread(target=opener, daemon=True).start()
    _print(f"NeuroBlocks editor running at {url}" + (f"?token={token}" if token else ""))
    _print("Press Ctrl+C to quit.")
    uvicorn.run(app, host=host, port=port, log_level="warning", ws_max_size=64 * 2**20)


def cmd_gui(a):
    if a.home:
        os.environ["NEUROBLOCKS_HOME"] = a.home
    _serve(a.host, a.port, None, None if a.no_browser else "")


def cmd_serve(a):
    if a.home:
        os.environ["NEUROBLOCKS_HOME"] = a.home
    token = a.token
    if a.host not in ("127.0.0.1", "localhost") and not token and not a.no_token:
        token = secrets.token_urlsafe(12)
        _print(f"Generated access token: {token}")
    if token:
        _print("In your local editor: Run on ▸ Add remote server, then enter this machine's address and the token.")
        _print(f"Tip: tunnel over SSH instead of opening a port:  ssh -L {a.port}:localhost:{a.port} user@this-machine"
               f"   → then use http://localhost:{a.port}")
    _serve(a.host, a.port, token, None)


def cmd_view(a):
    from . import paths
    src = Path(a.run)
    if src.is_file():
        src = src.parent
    if not (src / "events.jsonl").exists():
        sys.exit(f"No events.jsonl in {src}")
    runs = paths.runs_dir()
    try:
        src.resolve().relative_to(runs.resolve())
        run_id = src.name
    except ValueError:
        run_id = src.name
        dest = runs / run_id
        if not dest.exists():
            shutil.copytree(src, dest)
        _print(f"Copied the run into {dest}")
    _serve("127.0.0.1", a.port, None, f"?view={run_id}")


# ---------------------------------------------------------------------------
# run / export / bundle
# ---------------------------------------------------------------------------


def _parse_sets(items):
    params = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"--set needs name=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        params[k.strip()] = v
    return params


def cmd_run(a):
    from . import paths
    from .project import compile_project, load_project, slugify

    src = Path(a.file)
    if not src.exists():
        sys.exit(f"File not found: {src}")
    if a.home:
        os.environ["NEUROBLOCKS_HOME"] = a.home
    os.environ["NEUROBLOCKS_EVENTS"] = "silent" if a.quiet else "cli"
    if a.device:
        os.environ["NEUROBLOCKS_DEVICE"] = a.device
    if a.quick:
        os.environ["NEUROBLOCKS_QUICK"] = "1"
    params = _parse_sets(a.set)
    os.environ["NEUROBLOCKS_PARAMS"] = json.dumps(params)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    names = {}
    if src.suffix.lower() == ".py":
        script = src.resolve()
        run_dir = Path(a.out) if a.out else paths.runs_dir() / f"{stamp}-{slugify(src.stem)}"
    else:
        project = load_project(src)
        res = compile_project(project, markers=True, header=True)
        for d in res.diagnostics:
            if d.level in ("error", "warning"):
                _print(f"{'ERROR' if d.level == 'error' else 'warning'}: {d.message}")
        if not res.ok:
            sys.exit(1)
        run_dir = Path(a.out) if a.out else paths.runs_dir() / f"{stamp}-{slugify(project.get('name', src.stem))}"
        run_dir.mkdir(parents=True, exist_ok=True)
        script = run_dir / "script.py"
        script.write_text(res.code, encoding="utf-8")
        (run_dir / "project.nblk").write_text(json.dumps(project), encoding="utf-8")
        names = res.names
        (run_dir / "script.names.json").write_text(json.dumps(names), encoding="utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NEUROBLOCKS_RUN_DIR"] = str(run_dir)
    _print(f"NeuroBlocks: running {src.name}  (log & pictures → {run_dir})")

    from .runtime import core
    from .runtime.harness import run_script
    import signal

    presses = {"n": 0, "t": 0.0}

    def on_sigint(signum, frame):
        now = time.time()
        if presses["n"] and now - presses["t"] < 3:
            raise KeyboardInterrupt
        presses["n"] += 1
        presses["t"] = now
        core.STATE.skip.set()
        _print("\nFinishing the current training early… (press Ctrl+C again within 3 s to stop everything)")

    try:
        signal.signal(signal.SIGINT, on_sigint)
    except ValueError:
        pass
    code = run_script(script, names)
    sys.exit(code)


def cmd_export(a):
    from .project import compile_project, load_project, slugify
    project = load_project(a.file)
    res = compile_project(project, markers=False)
    for d in res.diagnostics:
        _print(f"{d.level}: {d.message}")
    if not res.ok:
        sys.exit(1)
    out = Path(a.output or f"{slugify(project.get('name', Path(a.file).stem))}.py")
    out.write_text(res.code, encoding="utf-8")
    _print(f"Wrote {out}  (run it with: python {out}  — needs the neuroblocks package)")


def cmd_bundle(a):
    from .project import load_project, make_bundle, slugify
    project = load_project(a.file)
    out = Path(a.output or f"{slugify(project.get('name', Path(a.file).stem))}_bundle.zip")
    try:
        summary = make_bundle(project, out)
    except ValueError as e:
        sys.exit(str(e))
    _print(f"Wrote {out}")
    if summary["files"]:
        _print("Included data/model files: " + ", ".join(summary["files"]))
    _print("On the GPU machine:  unzip it, cd into the folder, then: bash run.sh")


# ---------------------------------------------------------------------------
# info commands
# ---------------------------------------------------------------------------


def _arg_text(spec, name: str) -> str:
    from .compiler.spec import Port
    arg = spec.args.get(name)
    if isinstance(arg, Port):
        return "●" if arg.direction == "out" else "○"
    return f"[{name.lower()}]"


def cmd_blocks(a):
    from .compiler.spec import CATEGORIES, REGISTRY
    from .compiler import blocks  # noqa: F401
    import re
    cats = sorted(CATEGORIES.values(), key=lambda c: c.order)
    lines = ["# NeuroBlocks block reference", ""] if a.markdown else []
    for cat in cats:
        specs = [s for s in REGISTRY.values() if s.category == cat.key and s.toolbox]  # hidden = old blocks
        if not specs:
            continue
        lines.append(f"## {cat.name}" if a.markdown else f"\n[{cat.name}]")
        if a.markdown:
            lines.append("")
            lines.append("| block | what it does |")
            lines.append("|---|---|")
        for s in sorted(specs, key=lambda s: s.order):
            msg = re.sub(r"%([A-Z_0-9]+)", lambda m: _arg_text(s, m.group(1)), s.message).replace("\n", " ")
            desc = (s.help or s.tooltip).replace("|", "\\|")
            lines.append(f"| `{msg}` | {desc} |" if a.markdown else f"  {msg}  —  {s.tooltip}")
        if a.markdown:
            lines.append("")
    _print("\n".join(lines))


def cmd_examples(a):
    from . import paths
    exs = sorted(paths.EXAMPLES_DIR.glob("*.nblk"))
    if a.copy:
        dest = Path(a.copy)
        dest.mkdir(parents=True, exist_ok=True)
        for e in exs:
            shutil.copy(e, dest / e.name)
        _print(f"Copied {len(exs)} examples to {dest}")
        return
    for e in exs:
        d = json.loads(e.read_text(encoding="utf-8"))
        _print(f"{e.name:38s} {d.get('name', '')} — {d.get('description', '')[:70]}")


def cmd_doctor(a):
    import importlib.util
    from . import __version__, paths
    _print(f"NeuroBlocks {__version__} · Python {sys.version.split()[0]} · home {paths.home()}")
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        _print(f"PyTorch {torch.__version__} · CUDA: {'yes — ' + ', '.join(gpus) if cuda else 'no'}"
               f"{' · Apple MPS: yes' if mps else ''} · CPU threads: {torch.get_num_threads()}")
    except ImportError:
        _print("PyTorch: NOT INSTALLED  (pip install torch)")
    if a.short:
        return
    for mod, why in [("sklearn", "classic ML"), ("pymunk", "car physics"), ("pandas", "CSV files"),
                     ("fastapi", "editor server"), ("uvicorn", "editor server"), ("websockets", "live events"),
                     ("tiktoken", "GPT-2 tokenizer (optional)"), ("transformers", "pretrained models (optional)"),
                     ("datasets", "Hugging Face datasets (optional)")]:
        ok = importlib.util.find_spec(mod) is not None
        _print(f"  {'ok ' if ok else '-- '} {mod:13s} {why}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="neuroblocks", description="Scratch-style blocks for AI and machine learning.")
    p.add_argument("--version", action="version", version=f"neuroblocks {__import__('neuroblocks').__version__}")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("gui", help="open the block editor in your browser")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--no-browser", action="store_true")
    g.add_argument("--home", help="folder for projects/data/models/runs (default ~/NeuroBlocks)")
    g.set_defaults(fn=cmd_gui)

    s = sub.add_parser("serve", help="run the server for remote use (e.g. on a cloud GPU)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", help="access token (generated automatically when listening on a public address)")
    s.add_argument("--no-token", action="store_true", help="allow access without a token (only on private networks!)")
    s.add_argument("--home")
    s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("run", help="run a project (.nblk) or exported script (.py) without the editor")
    r.add_argument("file")
    r.add_argument("--device", help="auto | cpu | cuda | cuda:1 | mps")
    r.add_argument("--quick", action="store_true", help="smoke test: shorten every training loop")
    r.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a hyperparameter block")
    r.add_argument("--out", help="folder for this run's log and pictures")
    r.add_argument("--home")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("export", help="convert a project to a Python script")
    e.add_argument("file")
    e.add_argument("-o", "--output")
    e.set_defaults(fn=cmd_export)

    bnd = sub.add_parser("bundle", help="make a zip to run the project on another machine / cloud GPU")
    bnd.add_argument("file")
    bnd.add_argument("-o", "--output")
    bnd.set_defaults(fn=cmd_bundle)

    v = sub.add_parser("view", help="open a run's results (events.jsonl folder) in the editor")
    v.add_argument("run")
    v.add_argument("--port", type=int, default=8765)
    v.set_defaults(fn=cmd_view)

    bl = sub.add_parser("blocks", help="list every block")
    bl.add_argument("--markdown", action="store_true")
    bl.set_defaults(fn=cmd_blocks)

    ex = sub.add_parser("examples", help="list or copy the example projects")
    ex.add_argument("--copy", metavar="FOLDER")
    ex.set_defaults(fn=cmd_examples)

    d = sub.add_parser("doctor", help="check the setup")
    d.add_argument("--short", action="store_true")
    d.set_defaults(fn=cmd_doctor)

    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        a = p.parse_args(["gui"] + (argv or []))
    a.fn(a)


if __name__ == "__main__":
    main()
