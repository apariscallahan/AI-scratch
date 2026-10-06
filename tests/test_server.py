"""The editor server: API endpoints, running a program, exports."""
import io
import json
import os
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from neuroblocks.compiler import B
from neuroblocks.compiler.spec import _stack_json


def program(*blocks):
    head = _stack_json([B("nb_when_run")] + list(blocks))
    head["x"], head["y"] = 0, 0
    return {"blocks": {"languageVersion": 0, "blocks": [head]}}


@pytest.fixture(scope="module")
def client():
    os.environ["NEUROBLOCKS_NO_WARM"] = "1"
    from neuroblocks.server.app import create_app
    with TestClient(create_app()) as c:
        yield c


def test_blocks_and_info(client):
    b = client.get("/api/blocks").json()
    assert len(b["blocks"]) > 150 and b["toolbox"]["contents"]
    assert client.get("/api/ping").json()["app"] == "neuroblocks"
    assert "version" in client.get("/api/info").json()


def test_examples(client):
    exs = client.get("/api/examples").json()
    assert len(exs) >= 15
    one = client.get(f"/api/examples/{exs[0]['id']}").json()
    assert "workspace" in one


def test_compile_reports_errors(client):
    res = client.post("/api/compile", json={"workspace": program(B("nb_train", MODEL="ghost", DATA="data"))}).json()
    assert not res["ok"]
    res = client.post("/api/compile", json={"workspace": program(B("nb_say"))}).json()
    assert res["ok"] and "nb.say" in res["code"]


def test_run_streams_events(client):
    ws = program(B("nb_say", TEXT=B("text", TEXT="hi from a test")),
                 B("nb_data_toy", NAME="data", KIND="moons", SETTINGS=[B("nb_ds_samples", N=100)]),
                 B("nb_model", NAME="model", LAYERS=[B("nb_l_dense", UNITS=8, ACT="relu"), B("nb_l_output")]),
                 B("nb_train", MODEL="model", DATA="data", SETTINGS=[B("nb_t_epochs", N=2)]))
    with client.websocket_connect("/ws") as sock:
        hello = json.loads(sock.receive_text())
        assert hello["type"] == "hello"
        res = client.post("/api/run", json={"workspace": ws, "project": "test run"}).json()
        assert res["ok"], res
        types = []
        said = None
        deadline = time.time() + 120
        while time.time() < deadline:
            msg = json.loads(sock.receive_text())
            ev = msg["event"]
            types.append(ev["type"])
            if ev["type"] == "say":
                said = ev["text"]
            if ev["type"] == "done":
                assert ev["status"] == "finished", ev
                break
    assert said == "hi from a test"
    for t in ("run_start", "dataset", "model", "metric", "progress", "done"):
        assert t in types, t
    runs = client.get("/api/runs").json()
    assert runs and runs[0]["has_events"]


def test_bundle_export(client):
    ws = program(B("nb_say", TEXT=B("text", TEXT="cloud!")))
    r = client.post("/api/export/bundle", json={"workspace": ws, "project": "Cloud Test"})
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    for f in ("project.nblk", "train.py", "run.sh", "requirements.txt", "Dockerfile", "README.md",
              "neuroblocks/__init__.py", "neuroblocks/runtime/core.py", "neuroblocks/cli.py"):
        assert f in names, f
    assert "nb.say('cloud!')" in z.read("train.py").decode()


def test_projects_and_upload(client):
    proj = {"name": "Saved one", "workspace": program(B("nb_say"))}
    r = client.put("/api/projects/saved_one.nblk", json=proj).json()
    assert r["ok"]
    assert any(p["file"] == "saved_one.nblk" for p in client.get("/api/projects").json())
    assert client.get("/api/projects/saved_one.nblk").json()["name"] == "Saved one"
    up = client.post("/api/upload?path=data/tiny.csv", content=b"a,b,label\n1,2,x\n3,4,y\n").json()
    assert up["ok"]
    files = client.get("/api/files").json()
    assert any(f["name"] == "tiny.csv" for f in files["data"])
    assert client.post("/api/upload?path=../evil.txt", content=b"x").status_code == 400
