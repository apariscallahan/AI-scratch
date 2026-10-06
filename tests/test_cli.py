"""The command line: export, bundle, run (quick mode, hyperparameter overrides, test exit codes)."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from neuroblocks.compiler import B
from neuroblocks.compiler.spec import _stack_json
from neuroblocks.project import new_project, save_project

ROOT = Path(__file__).resolve().parents[1]


def make_project(path: Path, *blocks):
    head = _stack_json([B("nb_when_run")] + list(blocks))
    head["x"], head["y"] = 0, 0
    save_project(path, new_project("CLI test", {"blocks": {"languageVersion": 0, "blocks": [head]}}))
    return path


def run_cli(*args, home):
    env = dict(os.environ, NEUROBLOCKS_HOME=str(home), PYTHONIOENCODING="utf-8")
    env.pop("NEUROBLOCKS_EVENTS", None)
    env.pop("NEUROBLOCKS_NO_RUN_DIR", None)
    return subprocess.run([sys.executable, "-m", "neuroblocks", *args], capture_output=True, text=True,
                          encoding="utf-8", env=env, cwd=str(ROOT), stdin=subprocess.DEVNULL, timeout=600)


def test_importing_project_does_not_import_runtime():
    code = "import sys, neuroblocks.project, neuroblocks.server.app; print('neuroblocks.runtime' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert out.stdout.strip() == "False", out.stderr


def test_export_and_bundle(tmp_path):
    proj = make_project(tmp_path / "p.nblk", B("nb_say", TEXT=B("text", TEXT="hello cli")))
    r = run_cli("export", str(proj), "-o", str(tmp_path / "p.py"), home=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "nb.say('hello cli')" in (tmp_path / "p.py").read_text(encoding="utf-8")
    r = run_cli("bundle", str(proj), "-o", str(tmp_path / "b.zip"), home=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "run.sh" in zipfile.ZipFile(tmp_path / "b.zip").namelist()


def test_run_quick_params_and_checks(tmp_path):
    proj = make_project(
        tmp_path / "p.nblk",
        B("nb_param", VAR={"name": "n"}, VALUE=1),
        B("nb_say", TEXT=B("text_join", _extra={"extraState": {"itemCount": 2}}, ADD0=B("text", TEXT="n is "),
                           ADD1=B("variables_get", VAR={"name": "n"}))),
        B("nb_data_toy", NAME="data", KIND="moons", SETTINGS=[B("nb_ds_samples", N=200)]),
        B("nb_model", NAME="model", LAYERS=[B("nb_l_dense", UNITS=8, ACT="relu"), B("nb_l_output")]),
        B("nb_train", MODEL="model", DATA="data", SETTINGS=[B("nb_t_epochs", N=500)]),
        B("nb_check", VALUE=B("nb_score", METRIC="accuracy", MODEL="model", DATA="data"), OP=">=", TARGET=1.5),
    )
    r = run_cli("run", str(proj), "--quick", "--set", "n=7", home=tmp_path)
    assert "n is 7" in r.stdout, r.stdout + r.stderr
    assert "Quick test mode" in r.stdout
    assert r.returncode == 0  # failed checks don't count in quick mode
    events = [json.loads(l) for d in (tmp_path / "runs").iterdir() for l in (d / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["type"] == "start" and e["quick"] for e in events)
    r = run_cli("run", str(proj), "--set", "n=1", home=tmp_path)
    assert r.returncode == 1  # the impossible check fails a real run
