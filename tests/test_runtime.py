"""The runtime library: data, models, training, testing, generation, saving."""
import math

import pytest
import torch

import neuroblocks as nb
from neuroblocks.runtime.errors import NBError

L = nb.layers


@pytest.fixture(autouse=True)
def _setup():
    nb.setup(project="tests")
    nb.config(seed=0)


def test_toy_classification_learns():
    data = nb.data.toy("moons", name="data", samples=400, noise=0.1)
    model = nb.Model("model", [L.Dense(32), L.Dense(32), L.Output()])
    res = nb.train(model, data, epochs=40, lr=0.01)
    assert res["acc"] > 0.9
    assert nb.score(model, data, "accuracy") > 0.9
    assert nb.predict(model, [0.0, 0.0]) in data.class_names
    nb.show_decision_map(model, data)


def test_regression_and_curve():
    data = nb.data.toy("linear", name="line", samples=300, noise=0.05)
    model = nb.Model("m", [L.Dense(16, activation="tanh")])
    nb.train(model, data, epochs=30, lr=0.01)
    assert nb.score(model, data, "r2") > 0.9
    assert isinstance(nb.predict(model, [1.0]), float)


def test_auto_fixes_flatten_and_output():
    data = nb.data.images("digits", name="digits", samples=300)
    model = nb.Model("cnn", [L.Conv2D(8, 3), L.Dense(16)])  # no flatten, no output layer
    model.build_for(data)
    names = [r["name"] for r in model.rows]
    assert "flatten" in names and names[-1] == "output"
    nb.train(model, data, epochs=1)


def test_friendly_layer_errors():
    data = nb.data.table("iris", name="iris")
    with pytest.raises(NBError, match="Convolution layers need images"):
        nb.Model("bad", [L.Conv2D(8, 3)]).build_for(data)
    with pytest.raises(NBError, match="divide evenly"):
        text = nb.data.text("toy_stories", name="t", context=16, samples=5000)
        nb.Model("bad2", [L.TokenEmbedding(30), L.TransformerBlock(4)]).build_for(text)


def test_language_model_trains_and_writes():
    data = nb.data.text("toy_stories", name="stories", context=32, samples=20000)
    gpt = nb.GPT("gpt", layers=1, heads=2, embed=32, dropout=0.0)
    res = nb.train(gpt, data, steps=60, batch_size=16)
    assert res["loss"] < math.log(data.tokenizer.vocab_size)  # better than guessing
    text = nb.generated_text(gpt, "Once", 40)
    assert isinstance(text, str) and len(text) > 0
    out = nb.evaluate(gpt, data)
    assert out["perplexity"] > 1


def test_text_classifier_and_left_padding():
    data = nb.data.text_classes("toy_sentiment", name="s", samples=600)
    x = data.x_train[0]
    assert x[-1] != 0  # real text at the end (left padded)
    lstm = nb.Model("lstm", [L.TokenEmbedding(16), L.Recurrent("gru", 16, keep="last")])
    nb.train(lstm, data, epochs=4, lr=0.01)
    assert nb.score(lstm, data, "accuracy") > 0.7
    assert nb.predict(lstm, "I loved the movie") in ("positive", "negative")


def test_classic_models():
    data = nb.data.table("wine", name="wine")
    for maker in (nb.classic.decision_tree, nb.classic.random_forest, nb.classic.knn, nb.classic.svm,
                  nb.classic.logistic_regression, nb.classic.naive_bayes, nb.classic.gradient_boosting):
        m = maker("m")
        nb.train(m, data)
        assert nb.score(m, data, "accuracy") > 0.8, maker.__name__
    km = nb.classic.kmeans("km", k=3)
    nb.train(km, data)
    assert 0 <= nb.score(km, data, "accuracy") <= 1


def test_save_and_load_roundtrip(tmp_path):
    data = nb.data.toy("blobs", name="blobs", samples=300, classes=3)
    model = nb.Model("m", [L.Dense(16), L.Output()])
    nb.train(model, data, epochs=10, lr=0.01)
    before = nb.predict(model, [0.1, 0.2])
    path = nb.save(model, str(tmp_path / "m.pt"))
    loaded = nb.load_model(path, name="m2")
    assert nb.predict(loaded, [0.1, 0.2]) == before
    tree = nb.classic.decision_tree("t")
    nb.train(tree, data)
    p2 = nb.save(tree, str(tmp_path / "t.pt"))
    assert nb.predict(nb.load_model(p2, name="t2"), [0.1, 0.2]) == nb.predict(tree, [0.1, 0.2])


def test_autoencoder_and_series():
    digits = nb.data.images("digits", name="d", samples=200)
    ae = nb.Model("ae", [L.Flatten(), L.Dense(16), L.Dense(4), L.Dense(16)])
    nb.train(ae, digits, epochs=2, objective="reconstruct")
    assert "loss" in nb.evaluate(ae, digits)
    ts = nb.data.series("sine", name="ts", window=16, samples=400)
    rnn = nb.Model("rnn", [L.Recurrent("lstm", 16, keep="last")])
    nb.train(rnn, ts, epochs=3, lr=0.01)
    assert nb.score(rnn, ts, "r2") > 0.5


def test_callbacks_and_metric_of():
    data = nb.data.toy("xor", name="xor", samples=200)
    model = nb.Model("m", [L.Dense(8), L.Output()])
    calls = []
    nb.train(model, data, epochs=4, every=[(2, "epochs", lambda: calls.append(1))])
    assert len(calls) == 2
    assert nb.metric_of(model, "steps") > 0
    assert 0 <= nb.metric_of(model, "val_acc") <= 1


def test_picture_generators(tmp_path):
    digits = nb.data.images("digits", name="d", samples=400, size=16)
    for kind in ("vae", "gan"):
        g = nb.generator(f"g_{kind}", kind=kind)
        nb.train(g, digits, epochs=2, batch_size=64)
        imgs = g.sample(6, "3")
        assert tuple(imgs.shape) == (6, 1, 16, 16) and 0 <= float(imgs.min()) and float(imgs.max()) <= 1
        nb.show_generated(g, 4, "any")
        loaded = nb.load_model(nb.save(g, str(tmp_path / f"{kind}.pt")), name="again")
        assert tuple(loaded.sample(2).shape) == (2, 1, 16, 16)
    with pytest.raises(NBError, match="isn't one of the classes"):
        g.sample(2, "banana")


def test_tokenizers_roundtrip_and_vectorised_encoding():
    from neuroblocks.runtime.tokenizers import CharTokenizer, WordTokenizer
    text = "Hello, world! Hello again.\nNew line here."
    ch = CharTokenizer.fit(text)
    assert ch.decode(ch.encode(text)) == text
    assert list(ch.encode_array(text + "€")) == ch.encode(text + "€")  # unknown chars dropped the same way
    wt = WordTokenizer.fit(text, max_vocab=50)
    assert wt.decode(wt.encode("Hello, world!")) == "Hello, world!"


def test_core_helpers():
    assert nb.text(0.30000000000000004) == "0.3"
    assert nb.join("a", 1.0, True) == "a1true"
    assert nb.list_get([1, 2, 3], "from_end", 1) == 3
    assert nb.append(None, 5) == [5]
    assert list(nb.count(1, 2, 0.5)) == [1, 1.5, 2]
    assert nb.check(0.95, ">=", 0.9)
    assert nb.param("not_set", 7) == 7
