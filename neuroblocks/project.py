"""Project files (.nblk), compilation helpers and cloud bundles."""
from __future__ import annotations

import datetime as _dt
import io
import json
import re
import zipfile
from pathlib import Path

from . import __version__, paths

FORMAT = "neuroblocks-project"


def slugify(name: str, default: str = "project") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name or "")).strip("_").lower()
    return s[:60] or default


def new_project(name: str, workspace: dict, description: str = "") -> dict:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    return {"format": FORMAT, "version": 1, "name": name, "description": description, "created": now,
            "modified": now, "neuroblocks": __version__, "workspace": workspace}


def load_project(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "workspace" not in data and "blocks" in data:  # a bare Blockly workspace
        data = new_project(Path(path).stem, data)
    data.setdefault("name", Path(path).stem)
    return data


def save_project(path: str | Path, project: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    project = dict(project)
    project["format"] = FORMAT
    project["modified"] = _dt.datetime.now().isoformat(timespec="seconds")
    p.write_text(json.dumps(project, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def compile_project(project: dict, markers: bool = False, header: bool = True):
    from .compiler import compile_workspace
    return compile_workspace(project.get("workspace") or {}, markers=markers, project=project.get("name", "Untitled"),
                             header=header)


# ---------------------------------------------------------------------------
# Walking the workspace
# ---------------------------------------------------------------------------


def iter_blocks(workspace: dict):
    def walk(b):
        yield b
        for inp in (b.get("inputs") or {}).values():
            for k in ("block", "shadow"):
                if inp.get(k):
                    yield from walk(inp[k])
        nxt = (b.get("next") or {}).get("block")
        if nxt:
            yield from walk(nxt)

    for top in ((workspace or {}).get("blocks") or {}).get("blocks") or []:
        yield from walk(top)


def _literal_input(block: dict, name: str):
    inp = (block.get("inputs") or {}).get(name) or {}
    b = inp.get("block") or inp.get("shadow")
    if b and b.get("type") == "text":
        return (b.get("fields") or {}).get("TEXT")
    return None


FILE_INPUTS = {"nb_data_csv": ("FILE", "data"), "nb_data_textfile": ("FILE", "data"),
               "nb_data_folder": ("FOLDER", "data"), "nb_load": ("FILE", "model")}


def referenced_files(workspace: dict) -> list[tuple[Path, str]]:
    """Local files the project reads: [(absolute path, path inside a bundle)]."""
    out = []
    home = paths.home()
    for b in iter_blocks(workspace):
        spec = FILE_INPUTS.get(b.get("type"))
        if not spec:
            continue
        val = _literal_input(b, spec[0])
        if not val:
            continue
        p = paths.resolve_data_file(val) if spec[1] == "data" else paths.resolve_model_file(val)
        if not p.exists():
            continue
        try:
            rel = p.resolve().relative_to(home)
        except ValueError:
            rel = Path("data" if spec[1] == "data" else "models") / p.name
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    out.append((f, str(rel / f.relative_to(p)).replace("\\", "/")))
        else:
            out.append((p, str(rel).replace("\\", "/")))
    return out


def requirements_for(workspace: dict) -> list[str]:
    reqs = ["numpy>=1.24", "scikit-learn>=1.3", "pymunk>=6.6", "pandas>=2.0"]
    extra = set()
    for b in iter_blocks(workspace):
        t = b.get("type")
        fields = b.get("fields") or {}
        if t == "nb_pretrained":
            extra |= {"transformers>=4.40", "accelerate>=0.30"}
        if t == "nb_data_hf" or (t == "nb_data_text_classes" and fields.get("KIND") in ("imdb", "ag_news", "sst2")):
            extra.add("datasets>=2.18")
        if t == "nb_ds_tokenizer" and fields.get("TOK") == "gpt2":
            extra.add("tiktoken>=0.6")
    return reqs + sorted(extra)


# ---------------------------------------------------------------------------
# Cloud bundle
# ---------------------------------------------------------------------------

RUN_SH = """#!/usr/bin/env bash
# Run this NeuroBlocks project on a Linux machine — ideally one with an NVIDIA GPU.
#   bash run.sh                       # full run
#   bash run.sh --quick               # 1-minute smoke test first
#   bash run.sh --set steps=50000     # override a hyperparameter block
set -euo pipefail
cd "$(dirname "$0")"
export NEUROBLOCKS_HOME="$PWD"
PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "Creating a virtual environment (re-uses PyTorch if it is already installed)…"
  "$PY" -m venv .venv --system-site-packages
fi
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -c "import torch" 2>/dev/null || python -m pip install torch
python -m pip install --quiet -r requirements.txt
python -m neuroblocks doctor --short || true
python -m neuroblocks run project.nblk "$@"
echo
echo "Done. Results are in ./runs (charts: events.jsonl, pictures) and trained models in ./models"
"""

RUN_BAT = """@echo off
rem Run this NeuroBlocks project on Windows (uses the current Python).
cd /d "%~dp0"
set NEUROBLOCKS_HOME=%CD%
python -m pip install -r requirements.txt
python -m neuroblocks run project.nblk %*
pause
"""

DOCKERFILE = """# GPU container for this NeuroBlocks project.
#   docker build -t my-neuroblocks-job .
#   docker run --gpus all -v "$PWD/runs:/job/runs" -v "$PWD/models:/job/models" my-neuroblocks-job
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /job
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV NEUROBLOCKS_HOME=/job
CMD ["python", "-m", "neuroblocks", "run", "project.nblk"]
"""

README = """# {name} — NeuroBlocks cloud bundle

Everything needed to run this block program on another computer (for example a rented
Linux machine with a big NVIDIA GPU):

| file | what it is |
|---|---|
| `project.nblk` | the block program (open it in the NeuroBlocks editor) |
| `train.py` | the same program as plain Python |
| `neuroblocks/` | the NeuroBlocks runtime library |
| `requirements.txt` | Python packages needed |
| `run.sh` / `run.bat` | one-command runners for Linux / Windows |
| `Dockerfile` | container image definition (GPU) |
| `data/`, `models/` | your data files and models used by the project |

## Run it on a cloud GPU (Lambda, RunPod, Vast.ai, Paperspace, AWS, GCP …)

```bash
scp {zipname} user@gpu-machine:~        # or upload it in the provider's web UI
ssh user@gpu-machine
unzip {zipname} -d {slug} && cd {slug}
bash run.sh --quick                     # optional: 1-minute smoke test
bash run.sh                             # the real run (uses the GPU automatically)
```

Long run? Start it inside `tmux` or `screen`, or `nohup bash run.sh > log.txt &`, so it survives
disconnecting.

Hyperparameter blocks can be overridden without editing anything:
`bash run.sh --set steps=200000 --set layers=8`.

## Google Colab

Upload the zip, then in a cell: `!unzip -q {zipname} -d job && cd job && bash run.sh`.

## Getting the results back

* `models/` — trained models; copy them into your NeuroBlocks `models` folder and use the
  *load model from file* block to test them on your own computer.
* `runs/<date>-<name>/` — the run's log (`events.jsonl`) and pictures. Open it in the editor with
  `neuroblocks view runs/<folder>` (or Runs ▸ Open run log) to see the charts and replays.

## Watch it live instead

Run `python -m neuroblocks serve --host 0.0.0.0 --token SOME-SECRET` on the GPU machine, then in the
editor choose *Run on ▸ Add remote server* (tip: use an SSH tunnel: `ssh -L 8765:localhost:8765 user@gpu-machine`
and the URL `http://localhost:8765`).

Made with NeuroBlocks {version}.
"""


def make_bundle(project: dict, out: str | Path | io.BytesIO) -> dict:
    """Write a cloud bundle zip. Returns a summary dict."""
    ws = project.get("workspace") or {}
    res = compile_project(project, markers=False)
    if not res.ok:
        msgs = "; ".join(d.message for d in res.errors)
        raise ValueError(f"The project has errors: {msgs}")
    name = project.get("name") or "project"
    slug = slugify(name)
    zipname = f"{slug}_bundle.zip"
    files = referenced_files(ws)
    pkg = paths.PACKAGE_DIR
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.nblk", json.dumps(project, indent=1, ensure_ascii=False))
        z.writestr("train.py", res.code)
        z.writestr("requirements.txt", "\n".join(requirements_for(ws)) + "\n")
        info = zipfile.ZipInfo("run.sh")
        info.external_attr = 0o755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, RUN_SH)
        z.writestr("run.bat", RUN_BAT)
        z.writestr("Dockerfile", DOCKERFILE)
        z.writestr("README.md", README.format(name=name, zipname=zipname, slug=slug, version=__version__))
        for f in sorted(pkg.rglob("*")):
            if f.is_dir() or "__pycache__" in f.parts or f.suffix in (".pyc", ".map"):
                continue
            z.write(f, "neuroblocks/" + str(f.relative_to(pkg)).replace("\\", "/"))
        for src, rel in files:
            z.write(src, rel)
        for d in ("data", "models", "runs"):
            z.writestr(f"{d}/.keep", "")
    return {"name": zipname, "files": [rel for _, rel in files], "requirements": requirements_for(ws)}
