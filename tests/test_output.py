"""Output stacks: the 📦 weights block, wiring it to '○ start output', and using models step by step."""
import math

import numpy as np
import pytest
import torch

import neuroblocks as nb
from neuroblocks.compiler import B, compile_workspace
from neuroblocks.compiler.spec import _stack_json
from neuroblocks.runtime import output as out_rt
from neuroblocks.runtime.core import STATE
from neuroblocks.runtime.errors import NBError
from neuroblocks.runtime.events import emitter

L = nb.layers


@pytest.fixture(autouse=True)
def events(monkeypatch):
    """Seeded runtime, and every event captured instead of printed."""
    nb.setup(project="tests")
    nb.config(seed=0)
    captured = []

    def capture(type, **payload):
        ev = {"type": type, **payload}
        captured.append(ev)
        return ev

    monkeypatch.setattr(emitter(), "emit", capture)
    out_rt._OUT.__init__()
    yield captured
    out_rt._OUT.__init__()


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


def stack(blocks, x=0, y=0):
    head = _stack_json(blocks)
    head["x"], head["y"] = x, y
    return head


def program(training, output=None, *more):
    tops = [stack([B("nb_when_run")] + training)]
    if output is not None:
        tops.append(stack(output, 700, 0))
    tops.extend(more)
    return {"blocks": {"languageVersion": 0, "blocks": tops}}


V = lambda name: B("variables_get", VAR={"name": name})  # noqa: E731
TOY = [B("nb_data_toy", NAME="data", KIND="moons", SETTINGS=[B("nb_ds_samples", N=200)]),
       B("nb_model", NAME="model", LAYERS=[B("nb_l_dense", UNITS=16, ACT="relu"), B("nb_l_output")]),
       B("nb_train", MODEL="model", DATA="data", SETTINGS=[B("nb_t_epochs", N=3)])]


# ===========================================================================
# Compiler: wiring rules
# ===========================================================================


def test_wired_output_stack_compiles_after_training():
    res = compile_workspace(program(TOY + [B("nb_weights", _id="W", MODEL="model")],
                                    [B("nb_when_output", _id="H", PORT="W"),
                                     B("nb_say", TEXT=B("nb_o_name", CHOICE=B("nb_o_pick", HOW="best", PROBS=B(
                                         "nb_o_probs", SCORES=B("nb_o_run", INPUT=B("lists_create_with", _extra={
                                             "extraState": {"itemCount": 2}}, ADD0=B("math_number", NUM=0),
                                             ADD1=B("math_number", NUM=0))), TEMP=1))))]),
                            markers=False)
    assert res.ok, [d.message for d in res.diagnostics]
    code = res.code
    assert "model_weights = None" in code
    assert code.index("nb.train(") < code.index("model_weights = nb.weights(model)") < code.index(
        "nb.start_output(model_weights)") < code.index("model_weights.run(") < code.index("nb.end_output(model_weights)")
    compile(code, "<generated>", "exec")


def test_unwired_and_broken_wires_are_errors():
    res = compile_workspace(program(TOY + [B("nb_weights", _id="W", MODEL="model")], [B("nb_when_output", _id="H")]))
    assert not res.ok
    assert any(d.block_id == "H" and "isn't wired" in d.message for d in res.diagnostics)
    assert any(d.block_id == "W" and "Nothing uses these weights" in d.message for d in res.diagnostics)
    res = compile_workspace(program(TOY, [B("nb_when_output", _id="H", PORT="gone")]))
    assert any(d.block_id == "H" and "is gone" in d.message for d in res.diagnostics)


def test_weights_block_must_end_a_training_stack():
    loose = stack([B("nb_weights", _id="W", MODEL="model")], 300, 300)
    res = compile_workspace(program(TOY, [B("nb_when_output", _id="H", PORT="W")], loose))
    assert not res.ok
    assert any(d.block_id == "W" and "very bottom" in d.message for d in res.diagnostics)
    # …and not inside an 'every N steps' setting (that becomes a function)
    every = B("nb_train", MODEL="model", DATA="data", SETTINGS=[B("nb_t_every", N=1, UNIT="epochs", DO=[
        B("nb_weights", _id="W2", MODEL="model")])])
    res = compile_workspace(program(TOY[:2] + [every]))
    assert any(d.block_id == "W2" and "very bottom" in d.message for d in res.diagnostics)


def test_output_blocks_only_work_in_output_stacks_or_my_blocks():
    res = compile_workspace(program(TOY + [B("nb_say", TEXT=B("nb_o_tokens", _id="T"))]))
    assert any(d.block_id == "T" and "belongs in an output stack" in d.message for d in res.diagnostics)
    proc = {"type": "procedures_defreturn", "x": 900, "y": 400, "fields": {"NAME": "first token"},
            "extraState": {"params": []},
            "inputs": {"RETURN": {"block": B("nb_o_tokens", TEXT=B("text", TEXT="hi")).json()}}}
    res = compile_workspace(program(TOY, None, proc), markers=False)
    assert res.ok, [d.message for d in res.diagnostics]
    assert "nb.wired().tokens_of('hi')" in res.code
    # pure helpers (softmax, pick) work anywhere
    res = compile_workspace(program([B("nb_say", TEXT=B("nb_o_pick", HOW="best", PROBS=B(
        "nb_o_probs", SCORES=B("lists_create_with", _extra={"extraState": {"itemCount": 1}},
                               ADD0=B("math_number", NUM=1)), TEMP=1)))]))
    assert res.ok


def test_old_shortcut_blocks_still_compile_with_a_hint():
    res = compile_workspace(program(TOY + [B("nb_play_form", _id="P", MODEL="model")]))
    assert res.ok
    assert any(d.block_id == "P" and "old shortcut" in d.message for d in res.diagnostics)


def test_compiled_program_runs_training_then_output(events):
    ws = program(TOY + [B("nb_weights", _id="W", MODEL="model")],
                 [B("nb_when_output", _id="H", PORT="W"),
                  B("variables_set", VAR={"name": "probs"}, VALUE=B("nb_o_probs", SCORES=B(
                      "nb_o_run", INPUT=B("lists_create_with", _extra={"extraState": {"itemCount": 2}},
                                          ADD0=B("math_number", NUM=1), ADD1=B("math_number", NUM=0))), TEMP=1)),
                  B("nb_o_show_top", N=2, PROBS=V("probs")),
                  B("nb_say", TEXT=B("nb_o_name", CHOICE=B("nb_o_pick", HOW="best", PROBS=V("probs"))))])
    res = compile_workspace(ws, markers=True)
    assert res.ok, [d.message for d in res.diagnostics]
    ns = {}
    exec(res.code.replace("nb.finish()", ""), ns)
    kinds = [e["type"] for e in events]
    assert kinds.index("weights") < kinds.index("output_start") < kinds.index("bars") < kinds.index("output_end")
    start = of_type(events, "output_start")[0]
    assert start["block"] == "H" and start["source"] == "W"
    assert of_type(events, "say")[-1]["text"] in ns["data"].class_names


# ===========================================================================
# Runtime: probabilities and choices
# ===========================================================================


def test_softmax_temperature_and_picking():
    p = nb.softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(p), 1.0) and p[2] > p[1] > p[0]
    assert nb.softmax([1, 5, 2], temperature=0) == [0.0, 1.0, 0.0]
    hot, cold = nb.softmax([1, 2], temperature=5), nb.softmax([1, 2], temperature=0.2)
    assert hot[1] < nb.softmax([1, 2])[1] < cold[1]
    assert nb.pick([0.1, 0.7, 0.2], "best") == 1
    counts = np.bincount([nb.pick([0.2, 0.8]) for _ in range(2000)], minlength=2)
    assert 0.74 < counts[1] / 2000 < 0.86
    assert nb.chance_of([0.25, 0.75], 1) == 0.75
    with pytest.raises(NBError, match="look like raw scores"):
        nb.pick([2.0, -1.0])
    with pytest.raises(NBError, match="no choice 5"):
        nb.chance_of([0.5, 0.5], 5)
    with pytest.raises(NBError, match="empty"):
        nb.softmax([])


# ===========================================================================
# Runtime: every kind of model
# ===========================================================================


def test_language_model_output_step_by_step(events):
    data = nb.data.text("toy_stories", name="stories", context=16, samples=20000)
    gpt = nb.GPT("gpt", layers=1, heads=2, embed=32, dropout=0.0)
    nb.train(gpt, data, steps=40, batch_size=16)
    w = nb.weights(gpt)
    assert of_type(events, "weights")[0]["summary"].endswith(f"{data.tokenizer.vocab_size} tokens")
    nb.start_output(w)
    tokens = w.tokens_of("Once upon")
    assert w.text_of(tokens) == "Once upon"
    scores = w.next_token_scores(tokens)
    assert len(scores) == data.tokenizer.vocab_size == w.fact("vocab")
    assert w.run(tokens) == scores and w.run("Once upon") == scores
    for _ in range(20):  # longer than the context: the oldest tokens are left out
        tokens = nb.append(tokens, nb.pick(nb.softmax(w.next_token_scores(tokens), 0.8)))
        nb.show_text(w.text_of(tokens))
    assert len(tokens) == 29 and w.fact("context") == 16
    w.show_top(nb.softmax(scores), 3)
    bars = of_type(events, "bars")[-1]
    assert len(bars["items"]) == 3 and not bars["scores"]
    assert w.name_of(tokens[0]) == "O"
    nb.end_output(w)
    texts = of_type(events, "output_text")
    assert texts and texts[-1]["text"] == w.text_of(tokens)  # the held-back last update was sent
    with pytest.raises(NBError, match="needs tokens"):
        w.next_token_scores("Once")
    with pytest.raises(NBError, match="no token 999"):
        w.text_of([999])


def test_classifier_regression_and_classic_outputs():
    moons = nb.data.toy("moons", name="moons", samples=300, noise=0.1)
    clf = nb.Model("clf", [L.Dense(16), L.Output()])
    nb.train(clf, moons, epochs=20, lr=0.01)
    w = nb.weights(clf)
    scores = w.run([1.0, 0.0])
    assert len(scores) == 2 and w.fact("classes") == 2 and w.fact("inputs") == 2
    best = nb.pick(nb.softmax(scores), "best")
    assert w.name_of(best) == nb.predict(clf, [1.0, 0.0])
    with pytest.raises(NBError, match="needs a model that writes text"):
        w.next_token_scores([1, 2])

    line = nb.data.toy("linear", name="line", samples=200, noise=0.05)
    reg = nb.Model("reg", [L.Dense(8, activation="tanh")])
    nb.train(reg, line, epochs=10, lr=0.01)
    y = nb.weights(reg).run([1.0])
    assert isinstance(y, float) and math.isclose(y, nb.predict(reg, [1.0]), rel_tol=1e-4)

    iris = nb.data.table("iris", name="iris")
    tree = nb.classic.decision_tree("tree", depth=3)
    nb.train(tree, iris)
    wt = nb.weights(tree)
    flower = [5.1, 3.5, 1.4, 0.2]
    probs = nb.softmax(wt.run(flower))  # log-probabilities → the tree's own probabilities
    assert wt.name_of(nb.pick(probs, "best")) == nb.predict(tree, flower)
    assert "decision tree" in wt.summary()


def test_text_classifier_autoencoder_and_drawings():
    reviews = nb.data.text_classes("toy_sentiment", name="r", samples=400)
    reader = nb.Model("reader", [L.TokenEmbedding(16), L.Recurrent("gru", 16, keep="last")])
    nb.train(reader, reviews, epochs=2, lr=0.01)
    w = nb.weights(reader)
    assert len(w.run("what a great movie")) == 2
    assert len(w.run(w.tokens_of("what a great movie"))) == 2

    digits = nb.data.images("digits", name="d", samples=300)
    cnn = nb.Model("cnn", [L.Flatten(), L.Dense(16), L.Output()])
    nb.train(cnn, digits, epochs=1)
    drawing = nb.ask_drawing("draw!")  # no editor here: a blank drawing
    assert isinstance(drawing, out_rt.Picture) and nb.answer() is drawing
    assert len(nb.weights(cnn).run(drawing)) == 10

    ae = nb.Model("ae", [L.Flatten(), L.Dense(16), L.Dense(4), L.Dense(16)])
    nb.train(ae, digits, epochs=1, objective="reconstruct")
    wa = nb.weights(ae)
    rebuilt = wa.run(out_rt.Picture(torch.rand(1, 8, 8)))
    assert isinstance(rebuilt, out_rt.Picture) and tuple(rebuilt.pixels.shape) == (1, 8, 8)
    assert "rebuilds its input" in wa.summary()
    nb.show_picture(rebuilt)


def test_picture_generator_from_noise():
    digits = nb.data.images("digits", name="d", samples=300, size=16)
    g = nb.generator("artist", kind="vae")
    nb.train(g, digits, epochs=1, batch_size=64)
    w = nb.weights(g)
    noise = w.noise()
    assert len(noise) == w.fact("noise") == g.latent
    pic = w.picture_from(noise, "3")
    assert tuple(pic.pixels.shape) == (1, 16, 16)
    assert torch.equal(w.picture_from(noise, 3).pixels, pic.pixels)  # same noise + class = same picture
    with pytest.raises(NBError, match="doesn't take an input"):
        w.run([1, 2, 3])
    with pytest.raises(NBError, match="noise numbers"):
        w.picture_from([0.0], "3")


@pytest.mark.parametrize("kind,algo", [("cartpole", "ppo"), ("maze", "qtable")])
def test_simulation_policy_sense_decide_act(events, kind, algo):
    old = STATE.quick
    STATE.quick = True
    try:
        world = nb.rl.World(kind, name="world")
        model = nb.Model("agent", [L.Dense(8)] if algo != "qtable" else [])
        nb.rl.train(model, world, algorithm=algo, steps=600)
        w = nb.weights(model)
        w.start_try(world)
        n = 0
        while not w.try_over():
            senses = w.senses()
            assert len(senses) == w.fact("inputs")
            w.do(w.action_from(w.run(senses)))
            n += 1
        assert n == w.try_result("steps") and n <= 150
        w.show_try()
        assert of_type(events, "sim_replay")[-1]["title"].startswith(f"{w.label} · output try 1")
        assert isinstance(w.name_of(0), str)
    finally:
        STATE.quick = old


def test_weights_need_a_trained_model_and_ask_without_keyboard(events):
    with pytest.raises(NBError, match="hasn't been trained"):
        nb.weights(nb.Model("fresh", [L.Dense(4)]))
    with pytest.raises(NBError, match="never ran"):
        nb.start_output(None)
    assert nb.ask("Your name?") == "" and nb.answer() == ""
    assert any("No keyboard" in e.get("text", "") for e in of_type(events, "log"))
