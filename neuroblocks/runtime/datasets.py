"""Raw dataset sources: generators for built-in toy data and cached downloaders.

Everything here returns plain numpy arrays / strings; :mod:`neuroblocks.runtime.data`
turns them into :class:`Dataset` objects.
"""
from __future__ import annotations

import gzip
import io
import os
import pickle
import random
import struct
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from .. import paths
from .errors import NBError
from .events import emit

# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def cache_path(*parts) -> Path:
    p = paths.cache_dir().joinpath("datasets", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def download(urls, dest: Path, label: str) -> Path:
    """Download the first URL that works into ``dest`` (cached)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if isinstance(urls, str):
        urls = [urls]
    errors = []
    for url in urls:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NeuroBlocks/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                last = 0.0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last > 0.3:
                            last = now
                            emit("progress", id=f"dl-{label}", label=f"Downloading {label}",
                                 current=done, total=total or None,
                                 info=f"{done / 1e6:.1f} MB" + (f" / {total / 1e6:.1f} MB" if total else ""))
            os.replace(tmp, dest)
            emit("progress", id=f"dl-{label}", label=f"Downloading {label}", current=1, total=1,
                 info="done")
            return dest
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
            try:
                tmp.unlink()
            except OSError:
                pass
    raise NBError(f"Couldn't download {label}.",
                  hint="Check your internet connection. Details: " + "; ".join(errors)[:300])


# ---------------------------------------------------------------------------
# Toy 2-D data
# ---------------------------------------------------------------------------

CLASS_COLOURS = ["orange", "blue", "green", "purple", "pink", "brown", "grey", "teal"]


def toy_points(kind: str, n: int, noise: float | None, classes: int | None, rng: np.random.Generator):
    """Returns (X float32 [n,2] or [n,1], y, task, class_names)."""
    kind = kind.lower()
    if kind == "spirals":
        k = int(classes or 2)
        noise = 0.2 if noise is None else noise
        per = max(1, n // k)
        xs, ys = [], []
        for c in range(k):
            r = np.linspace(0.05, 1.0, per)
            t = np.linspace(c * 2 * np.pi / k, c * 2 * np.pi / k + 3.5 * np.pi / 1.0, per)
            t = t + rng.normal(0, noise, per)
            xs.append(np.stack([r * np.sin(t), r * np.cos(t)], 1))
            ys.append(np.full(per, c))
        X, y = np.concatenate(xs), np.concatenate(ys)
        return X.astype(np.float32), y.astype(np.int64), "classification", CLASS_COLOURS[:k]
    if kind == "moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n, noise=0.1 if noise is None else noise, random_state=int(rng.integers(1 << 30)))
        X = (X - [0.5, 0.25]) / 1.2
        return X.astype(np.float32), y.astype(np.int64), "classification", CLASS_COLOURS[:2]
    if kind == "circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n, noise=0.06 if noise is None else noise, factor=0.45,
                            random_state=int(rng.integers(1 << 30)))
        return X.astype(np.float32), y.astype(np.int64), "classification", CLASS_COLOURS[:2]
    if kind == "blobs":
        from sklearn.datasets import make_blobs
        k = int(classes or 3)
        X, y = make_blobs(n, centers=k, cluster_std=1.0 + 4 * (0.1 if noise is None else noise),
                          random_state=int(rng.integers(1 << 30)))
        X = (X - X.mean(0)) / (X.std(0) + 1e-9) * 0.5
        return X.astype(np.float32), y.astype(np.int64), "classification", CLASS_COLOURS[:k]
    if kind == "xor":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
        X = X + rng.normal(0, 0.05 if noise is None else noise, X.shape)
        return X.astype(np.float32), y, "classification", CLASS_COLOURS[:2]
    if kind == "checkerboard":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((np.floor(X[:, 0] * 2) + np.floor(X[:, 1] * 2)) % 2).astype(np.int64)
        X = X + rng.normal(0, 0.02 if noise is None else noise, X.shape)
        return X.astype(np.float32), y, "classification", CLASS_COLOURS[:2]
    if kind in ("sine", "polynomial", "linear", "steps"):
        x = rng.uniform(-3, 3, n)
        noise = 0.1 if noise is None else noise
        if kind == "sine":
            f = np.sin(x * 1.5)
        elif kind == "polynomial":
            f = 0.15 * x ** 3 - 0.5 * x
        elif kind == "linear":
            f = 0.6 * x + 0.3
        else:
            f = np.floor(x) / 2
        y = f + rng.normal(0, noise, n)
        return x[:, None].astype(np.float32), y[:, None].astype(np.float32), "regression", None
    raise NBError(f"Unknown toy dataset '{kind}'.")


def toy_series(kind: str, length: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    if kind == "sine":
        s = np.sin(t * 2 * np.pi / 50)
    elif kind == "seasonal":
        s = 0.5 * np.sin(t * 2 * np.pi / 30) + 0.3 * np.sin(t * 2 * np.pi / 7) + t / length
    elif kind == "random_walk":
        s = np.cumsum(rng.normal(0, 0.1, length))
    elif kind == "square":
        s = np.sign(np.sin(t * 2 * np.pi / 40))
    else:
        raise NBError(f"Unknown series '{kind}'.")
    return (s + rng.normal(0, noise, length)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tables (scikit-learn)
# ---------------------------------------------------------------------------


def sklearn_table(kind: str):
    """Returns (X, y, task, feature_names, class_names, description)."""
    from sklearn import datasets as skd
    kind = kind.lower()
    if kind == "iris":
        d = skd.load_iris()
        return (d.data, d.target, "classification", list(d.feature_names), list(d.target_names),
                "150 iris flowers measured in cm; predict the species.")
    if kind == "wine":
        d = skd.load_wine()
        return (d.data, d.target, "classification", list(d.feature_names), [str(n) for n in d.target_names],
                "178 wines described by chemistry; predict which vineyard they came from.")
    if kind == "breast_cancer":
        d = skd.load_breast_cancer()
        return (d.data, d.target, "classification", list(d.feature_names), list(d.target_names),
                "569 tumour scans; predict malignant vs benign.")
    if kind == "diabetes":
        d = skd.load_diabetes()
        return (d.data, d.target[:, None], "regression", list(d.feature_names), None,
                "442 patients; predict disease progression after one year (a number).")
    if kind == "california":
        d = skd.fetch_california_housing(data_home=str(paths.cache_dir() / "sklearn"))
        return (d.data, d.target[:, None], "regression", list(d.feature_names), None,
                "20,640 California districts; predict the median house price (in $100,000s).")
    raise NBError(f"Unknown table dataset '{kind}'.")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

MNIST_MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
                 "https://storage.googleapis.com/cvdf-datasets/mnist/"]
FASHION_MIRRORS = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
                   "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"]
IDX_FILES = {
    "train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz",
}
FASHION_CLASSES = ["T-shirt", "trousers", "pullover", "dress", "coat", "sandal", "shirt", "sneaker",
                   "bag", "ankle boot"]
CIFAR_CLASSES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    zero, dtype, ndim = struct.unpack(">HBB", data[:4])
    shape = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def idx_dataset(name: str, mirrors: list[str], label: str):
    arrays = {}
    for key, fname in IDX_FILES.items():
        dest = cache_path(name, fname)
        download([m + fname for m in mirrors], dest, f"{label} ({fname.split('-')[0]} {key.split('_')[1]})")
        arrays[key] = _read_idx(dest)
    return (arrays["train_x"][:, None], arrays["train_y"].astype(np.int64),
            arrays["test_x"][:, None], arrays["test_y"].astype(np.int64))


def cifar10():
    dest = cache_path("cifar10", "cifar-10-python.tar.gz")
    download(["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"], dest, "CIFAR-10 (163 MB)")
    xs, ys, tx, ty = [], [], None, None
    with tarfile.open(dest, "r:gz") as tar:
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if base.startswith("data_batch") or base == "test_batch":
                d = pickle.load(tar.extractfile(m), encoding="bytes")
                x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
                y = np.asarray(d[b"labels"], dtype=np.int64)
                if base == "test_batch":
                    tx, ty = x, y
                else:
                    xs.append(x)
                    ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), tx, ty


def digits8():
    from sklearn.datasets import load_digits
    d = load_digits()
    x = np.clip(d.images * 16, 0, 255).astype(np.uint8)[:, None]
    return x, d.target.astype(np.int64)


# ---------------------------------------------------------------------------
# Text corpora
# ---------------------------------------------------------------------------

SHAKESPEARE_URLS = ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"]
NAMES_URLS = ["https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"]

_FALLBACK_NAMES = (
    "emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily elizabeth sofia "
    "ella madison scarlett victoria aria grace chloe camila penelope riley layla lillian nora zoey "
    "mila aubrey hannah lily addison eleanor natalie luna savannah brooklyn leah zoe stella hazel "
    "ellie paisley audrey skylar violet claire bella aurora lucy anna samantha caroline genesis "
    "liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob michael "
    "daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt john jack "
    "luke jayden dylan grayson levi isaac gabriel julian mateo anthony jaxon lincoln joshua "
    "christopher andrew theodore caleb ryan asher nathan thomas leo isaiah charles josiah hudson"
).split()


def shakespeare() -> str:
    try:
        return download(SHAKESPEARE_URLS, cache_path("text", "tinyshakespeare.txt"),
                        "Tiny Shakespeare").read_text(encoding="utf-8")
    except NBError:
        emit("log", level="warn", text="Couldn't download Tiny Shakespeare — using the built-in toy "
                                       "stories instead.")
        return toy_stories(1500)


def names() -> str:
    try:
        return download(NAMES_URLS, cache_path("text", "names.txt"), "baby names").read_text(encoding="utf-8")
    except NBError:
        emit("log", level="warn", text="Couldn't download the names list — using a small built-in one.")
        return "\n".join(_FALLBACK_NAMES * 4) + "\n"


def python_code(max_chars: int = 600_000) -> str:
    """Python source code from the standard library on this computer (offline)."""
    base = Path(os.__file__).parent
    files = ["textwrap.py", "string.py", "random.py", "json/encoder.py", "json/decoder.py", "json/__init__.py",
             "csv.py", "fractions.py", "statistics.py", "heapq.py", "bisect.py", "calendar.py",
             "colorsys.py", "glob.py", "shlex.py", "queue.py", "fnmatch.py", "difflib.py",
             "dataclasses.py", "enum.py", "functools.py", "pprint.py", "argparse.py", "abc.py"]
    out = []
    total = 0
    for f in files:
        p = base / f
        if p.exists():
            t = p.read_text(encoding="utf-8", errors="ignore")
            out.append(t)
            total += len(t)
            if total > max_chars:
                break
    text = "\n\n".join(out)
    return text[:max_chars] if text else toy_stories(800)


_ANIMALS = ["cat", "dog", "rabbit", "fox", "bear", "owl", "mouse", "duck", "frog", "turtle", "lion",
            "pig", "bird", "horse", "squirrel", "puppy", "kitten", "dragon"]
_NAMES = ["Tom", "Lily", "Max", "Mia", "Sam", "Zoe", "Ben", "Ella", "Leo", "Ruby", "Finn", "Nora",
          "Tim", "Sue", "Anna", "Jack", "Lucy", "Pip"]
_PLACES = ["a big forest", "a small town", "a busy city", "a sunny garden", "a farm", "a village by the sea",
           "an old castle", "a quiet park", "a tall tree", "a cozy house"]
_ADJ = ["little", "happy", "brave", "sleepy", "curious", "tiny", "kind", "clever", "shy", "funny", "big",
        "friendly", "silly"]
_LIKES = ["play with a red ball", "eat sweet apples", "sing songs", "read books", "run in the grass",
          "look at the stars", "build sand castles", "paint pictures", "jump in the leaves",
          "dance in the rain", "fly a kite", "bake cookies"]
_PROBLEMS = ["lost a favorite toy", "could not find the way home", "saw a dark cloud", "heard a strange noise",
             "fell into a puddle", "broke a blue cup", "got stuck in a tree", "lost a shiny key"]
_FIXES = {"lost a favorite toy": "looked under every rock and found the toy",
          "could not find the way home": "followed the river all the way home",
          "saw a dark cloud": "waited under a big leaf until the sun came out",
          "heard a strange noise": "found that the noise was only the wind",
          "fell into a puddle": "dried off in the warm sun",
          "broke a blue cup": "glued the cup back together",
          "got stuck in a tree": "climbed down one branch at a time",
          "lost a shiny key": "found the key in the soft grass"}
_FEELINGS = ["sad", "scared", "worried", "upset", "surprised"]
_ENDINGS = ["From that day on, they were best friends.", "{name} smiled and went home.",
            "And they all lived happily ever after.", "It was the best day ever.",
            "{name} said thank you and gave {friend} a big hug.", "That night, {name} slept very well."]


def toy_stories(n: int = 2000, seed: int = 1) -> str:
    """Simple made-up children's stories — a tiny, fully offline corpus that small models learn fast."""
    rng = random.Random(seed)
    stories = []
    for _ in range(n):
        name = rng.choice(_NAMES)
        animal = rng.choice(_ANIMALS)
        friend_animal = rng.choice([a for a in _ANIMALS if a != animal])
        friend = rng.choice([nm for nm in _NAMES if nm != name])
        problem = rng.choice(_PROBLEMS)
        s = (f"Once upon a time, there was a {rng.choice(_ADJ)} {animal} named {name}. "
             f"{name} lived in {rng.choice(_PLACES)}. Every day, {name} liked to {rng.choice(_LIKES)}.\n"
             f"One day, {name} {problem}. {name} felt {rng.choice(_FEELINGS)}. "
             f"Then a {rng.choice(_ADJ)} {friend_animal} named {friend} came by and said, "
             f"\"Do not worry, I will help you!\"\n"
             f"Together they {_FIXES[problem]}. " + rng.choice(_ENDINGS).format(name=name, friend=friend))
        stories.append(s)
    return "\n\n".join(stories) + "\n"


_POS_ADJ = ["great", "wonderful", "amazing", "fantastic", "lovely", "brilliant", "fun", "beautiful",
            "excellent", "delightful", "charming", "exciting"]
_NEG_ADJ = ["terrible", "boring", "awful", "bad", "dull", "horrible", "disappointing", "annoying", "weak",
            "messy", "slow", "confusing"]
_SUBJECTS = ["the movie", "this film", "the story", "the acting", "the music", "the ending", "the characters",
             "the plot", "the book", "this show", "the game", "the food"]
_POS_VERB = ["loved", "enjoyed", "adored", "liked"]
_NEG_VERB = ["hated", "disliked", "regretted"]


def toy_sentiment(n: int = 3000, seed: int = 2):
    """Short made-up reviews with positive/negative labels (includes tricky negations)."""
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n):
        pos = rng.random() < 0.5
        subj = rng.choice(_SUBJECTS)
        adj = rng.choice(_POS_ADJ if pos else _NEG_ADJ)
        opp = rng.choice(_NEG_ADJ if pos else _POS_ADJ)
        verb = rng.choice(_POS_VERB if pos else _NEG_VERB)
        opp_verb = rng.choice(["liked", "enjoyed", "loved"]) if not pos else rng.choice(["hated", "disliked"])
        t = rng.choice([
            f"{subj.capitalize()} was {adj}.",
            f"I thought {subj} was really {adj}.",
            f"{subj.capitalize()} was not {opp} at all.",
            f"What a {adj} experience!",
            f"I {verb} {subj}.",
            f"I did not like {subj}." if not pos else f"I did not hate {subj}.",
            f"Honestly, {subj} was {adj} and I {verb} it.",
            f"I did not {opp_verb} it.",
            f"{subj.capitalize()} was {adj}, {rng.choice(['really', 'very', 'so'])} {adj}.",
            f"Not {opp}. {subj.capitalize()} was {adj}.",
        ])
        texts.append(t)
        labels.append(1 if pos else 0)
    return texts, np.asarray(labels, dtype=np.int64), ["negative", "positive"]
