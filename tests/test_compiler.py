"""The block compiler: every block, every example, and the tricky built-in blocks."""
import json
import re

import pytest

from neuroblocks import paths
from neuroblocks.compiler import B, compile_workspace
from neuroblocks.compiler.spec import REGISTRY, Stack, _stack_json, make_block
from neuroblocks.compiler.toolbox import build_toolbox, editor_bundle


def ws(*blocks):
    head = _stack_json([B("nb_when_run")] + list(blocks))
    head["x"], head["y"] = 0, 0
    return {"blocks": {"languageVersion": 0, "blocks": [head]}}


def compiled(*blocks):
    res = compile_workspace(ws(*blocks), markers=False)
    compile(res.code, "<generated>", "exec")  # must be valid Python
    return res


def test_registry_is_big_and_consistent():
    assert len(REGISTRY) > 150
    for t, spec in REGISTRY.items():
        j = spec.to_blockly()
        n = len(re.findall(r"%\d+", j["message0"]))
        assert n == len(j["args0"]), t
        assert spec.gen is not None, t
        assert spec.tooltip, f"{t} needs a tooltip"


def test_every_block_compiles_with_defaults():
    """Drop each block (with its defaults) into a program and compile it."""
    for t, spec in REGISTRY.items():
        if spec.shape in ("hat",):
            continue
        if spec.shape == "value":
            prog = ws(B("nb_say", TEXT=B(t)))
        elif spec.shape == "layer":
            prog = ws(B("nb_model", NAME="model", LAYERS=[B(t)]))
        elif spec.shape == "setting":
            check = spec.check if isinstance(spec.check, list) else [spec.check]
            container = next(s for s in REGISTRY.values() for a in s.args.values()
                             if isinstance(a, Stack) and a.check and set(check) & set(a.check if isinstance(a.check, list) else [a.check]))
            stack_name = next(n for n, a in container.args.items() if isinstance(a, Stack))
            prog = ws(B(container.type, **{stack_name: [B(t)]}))
        else:
            prog = ws(B(t))
        # Definitions for refs used by default.
        head = prog["blocks"]["blocks"][0]
        res = compile_workspace(prog, markers=True)
        compile(res.code, f"<{t}>", "exec")


def test_toolbox_and_bundle_are_json():
    tb = build_toolbox()
    assert tb["kind"] == "categoryToolbox"
    names = [c["name"] for c in tb["contents"]]
    for cat in ("Data", "Neural Nets", "Layers", "Training", "Simulations", "World Parts", "Classic ML"):
        assert cat in names
    json.dumps(editor_bundle())


@pytest.mark.parametrize("path", sorted(paths.EXAMPLES_DIR.glob("*.nblk")), ids=lambda p: p.stem)
def test_examples_compile(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    res = compile_workspace(data["workspace"], markers=True, project=data["name"])
    assert res.ok, [d.message for d in res.diagnostics]
    compile(res.code, path.name, "exec")


@pytest.mark.parametrize("path", sorted(paths.EXAMPLES_DIR.glob("*.nblk")), ids=lambda p: p.stem)
def test_examples_are_clean_and_wired(path):
    """No warnings in any example, and every output stack is wired to weights at the end of training."""
    data = json.loads(path.read_text(encoding="utf-8"))
    res = compile_workspace(data["workspace"], markers=False, project=data["name"])
    assert [d.message for d in res.diagnostics if d.level == "warning"] == []
    tops = data["workspace"]["blocks"]["blocks"]
    for hat in (b for b in tops if b["type"] == "nb_when_output"):
        assert hat["fields"]["PORT"], "output stack without a wire"
        assert "nb.start_output(" in res.code and "nb.weights(" in res.code
    used = set(re.findall(r'"type": "(nb_[a-z_]+)"', json.dumps(data)))
    assert not used & {"nb_predict", "nb_generated", "nb_play_chat", "nb_play_draw", "nb_play_form"}, \
        "examples should build output stacks instead of the old shortcut blocks"


def test_old_shortcut_blocks_are_hidden_and_ports_are_fields():
    bundle = editor_bundle()
    toolbox_types = {i["type"] for c in bundle["toolbox"]["contents"] for i in c.get("contents", []) if i.get("type")}
    assert "nb_weights" in toolbox_types and "nb_when_output" in toolbox_types
    assert not toolbox_types & {"nb_predict", "nb_play_chat", "nb_play_draw", "nb_play_form", "nb_generated"}
    hat = next(d for d in bundle["blocks"] if d["type"] == "nb_when_output")
    assert hat["args0"][0] == {"type": "field_nbport", "name": "PORT", "direction": "in"}
    weights = next(d for d in bundle["blocks"] if d["type"] == "nb_weights")
    assert "nextStatement" not in weights  # nothing goes below it: it ends the training stack


@pytest.mark.parametrize("path", sorted(paths.EXAMPLES_DIR.glob("*.nblk")), ids=lambda p: p.stem)
def test_examples_satisfy_blockly_connection_rules(path):
    """Blockly refuses to load stacks whose neighbours have incompatible connection checks."""
    from neuroblocks.compiler.validate import check_connections
    data = json.loads(path.read_text(encoding="utf-8"))
    assert check_connections(data["workspace"]) == []


def test_toolbox_blocks_satisfy_connection_rules_and_validator_catches_mistakes():
    from neuroblocks.compiler.validate import check_connections
    for cat in build_toolbox()["contents"]:
        for item in cat.get("contents", []):
            if item.get("kind") == "block":
                assert check_connections({"blocks": {"blocks": [item]}}) == [], item["type"]
    bad = {"blocks": {"blocks": [make_block("nb_say") | {"next": {"block": make_block("nb_l_dense")}}]}}
    assert check_connections(bad)
    # an 'any world' part may follow a car-only part inside a car world
    car = make_block("nb_world_car", PARTS=[B("nb_w_stuck"), B("nb_w_end_after"), B("nb_w_terrain")])
    assert check_connections({"blocks": {"blocks": [car]}}) == []


def test_unknown_reference_is_an_error():
    res = compile_workspace(ws(B("nb_data_toy", NAME="data"), B("nb_train", MODEL="nope", DATA="data")))
    assert not res.ok
    assert any("no model called 'nope'" in d.message for d in res.diagnostics)


def test_missing_hat_and_stray_blocks():
    res = compile_workspace({"blocks": {"blocks": [make_block("nb_say")]}})
    assert not res.ok
    w = ws(B("nb_say"))
    w["blocks"]["blocks"].append(dict(make_block("nb_say"), x=500, y=500))
    res = compile_workspace(w)
    assert res.ok and any("isn't attached" in d.message for d in res.diagnostics)


def test_setting_outside_container_is_error():
    res = compile_workspace(ws(B("nb_t_epochs")))
    assert not res.ok


def test_builtin_control_blocks():
    res = compiled(
        B("variables_set", VAR={"name": "x"}, VALUE=B("math_number", NUM=3)),
        B("controls_if", _extra={"extraState": {"elseIfCount": 1, "hasElse": True}},
          IF0=B("logic_compare", OP="GT", A=B("variables_get", VAR={"name": "x"}), B=B("math_number", NUM=2)),
          DO0=[B("nb_say", TEXT=B("text", TEXT="big"))],
          IF1=B("logic_boolean", BOOL="FALSE"), DO1=[B("nb_say", TEXT=B("text", TEXT="never"))],
          ELSE=[B("nb_say", TEXT=B("text", TEXT="small"))]),
        B("controls_for", VAR={"name": "i"}, FROM=B("math_number", NUM=1), TO=B("math_number", NUM=3),
          BY=B("math_number", NUM=1), DO=[B("math_change", VAR={"name": "x"}, DELTA=B("math_number", NUM=1))]),
        B("controls_repeat_ext", TIMES=B("math_number", NUM=2), DO=[]),
    )
    code = res.code
    assert "if x > 2:" in code and "elif False:" in code and "else:" in code
    assert "for i in range(1, 4):" in code
    ns = {}
    exec(code.replace("nb.finish()", ""), ns)  # runs with the real runtime (silent events)
    assert ns["x"] == 6


def test_arithmetic_precedence():
    expr = B("math_arithmetic", OP="MULTIPLY",
             A=B("math_arithmetic", OP="ADD", A=B("math_number", NUM=1), B=B("math_number", NUM=2)),
             B=B("math_number", NUM=3))
    res = compiled(B("variables_set", VAR={"name": "y"}, VALUE=expr))
    assert "y = (1 + 2) * 3" in res.code


def test_procedures():
    proc = {"type": "procedures_defreturn", "x": 400, "y": 0, "fields": {"NAME": "double it"},
            "extraState": {"params": [{"name": "n", "id": "p1"}]},
            "inputs": {"RETURN": {"block": {"type": "math_arithmetic", "fields": {"OP": "MULTIPLY"},
                                            "inputs": {"A": {"block": {"type": "variables_get", "fields": {"VAR": {"name": "n"}}}},
                                                       "B": {"block": {"type": "math_number", "fields": {"NUM": 2}}}}}}}}
    call = {"type": "procedures_callreturn", "extraState": {"name": "double it", "params": ["n"]},
            "inputs": {"ARG0": {"block": {"type": "math_number", "fields": {"NUM": 21}}}}}
    w = ws(B("variables_set", VAR={"name": "answer"}, VALUE=call))
    w["blocks"]["blocks"].append(proc)
    res = compile_workspace(w, markers=False)
    assert res.ok, res.diagnostics
    ns = {}
    exec(res.code.replace("nb.finish()", ""), ns)
    assert ns["answer"] == 42


def test_markers_and_bids():
    w = ws(B("nb_say", _id="abc"), B("nb_model", _id="m1", NAME="model", LAYERS=[B("nb_l_dense", _id="L1")]))
    res = compile_workspace(w, markers=True)
    assert "nb.at('abc')" in res.code
    assert "_bid='L1'" in res.code
    clean = compile_workspace(w, markers=False)
    assert "nb.at(" not in clean.code and "_bid" not in clean.code


def test_world_state_only_inside_reward():
    res = compile_workspace(ws(B("nb_say", TEXT=B("nb_w_state", WHAT="speed"))))
    assert not res.ok
    good = compiled(B("nb_world_car", NAME="world", PARTS=[
        B("nb_w_custom_reward", EXPR=B("nb_w_state", WHAT="speed"))]))
    assert "CustomReward(lambda _s: _s['speed'])" in good.code


def test_hyperparameter_block():
    res = compiled(B("nb_param", VAR={"name": "steps"}, VALUE=500),
                   B("nb_say", TEXT=B("variables_get", VAR={"name": "steps"})))
    assert "steps = nb.param('steps', 500)" in res.code
