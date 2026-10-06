"""Datasets: loading, splitting, normalising, batching and previews.

Every loader returns a :class:`Dataset` with a *task* the trainer understands:

* ``classification``  — predict one of several classes (``y`` = int64 labels)
* ``regression``      — predict one or more numbers (``y`` = float32)
* ``language_model``  — predict the next token of a text (:class:`TextDataset`)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch

from .. import paths
from . import datasets as src
from .core import STATE, register
from .errors import NBError, missing_package
from .events import emit, png_base64
from .tokenizers import Tokenizer, make_tokenizer

__all__ = ["Dataset", "TextDataset", "toy", "table", "images", "text", "text_file", "text_url",
           "text_classes", "csv", "image_folder", "huggingface", "series", "size", "class_count"]


def _warn_unused(name: str, unused: dict):
    for k in unused:
        if not k.startswith("_"):
            emit("log", level="warn", text=f"The '{k.replace('_', ' ')}' setting doesn't apply to dataset "
                                             f"'{name}', so it was ignored.")


def _rng(seed):
    if seed is None:
        seed = STATE.seed if STATE.seed is not None else 0
    return np.random.default_rng(int(seed))


def _split(n: int, test: float, rng) -> tuple[np.ndarray, np.ndarray]:
    test = float(test or 0)
    if test > 1:  # given as a percentage
        test = test / 100.0
    test = min(max(test, 0.0), 0.9)
    perm = rng.permutation(n)
    n_test = int(round(n * test))
    if test > 0 and n_test == 0 and n > 1:
        n_test = 1
    return perm[n_test:], perm[:n_test]


def _limit(x, y, n):
    if n and n > 0 and len(x) > n:
        return x[:n], y[:n]
    return x, y


class Dataset:
    kind = "dataset"

    def __init__(self, name: str, *, task: str, modality: str, x_train, y_train, x_test, y_test,
                 class_names=None, feature_names=None, target_names=None, tokenizer: Tokenizer | None = None,
                 context: int | None = None, description: str = "", source: str = "",
                 image_stats=None, augment=None, x_mean=None, x_std=None, y_mean=None, y_std=None,
                 texts_train=None, texts_test=None, raw_series=None):
        self.name = str(name)
        self.task = task
        self.modality = modality
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.class_names = list(class_names) if class_names is not None else None
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.target_names = list(target_names) if target_names is not None else None
        self.tokenizer = tokenizer
        self.context = context
        self.description = description
        self.source = source
        self.image_stats = image_stats  # (mean[C], std[C]) for uint8 images scaled to 0..1
        self.augment = augment
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std
        self.texts_train, self.texts_test = texts_train, texts_test
        self.raw_series = raw_series
        self._dev_cache: dict = {}

    # -- basic facts ----------------------------------------------------------
    @property
    def n_train(self) -> int:
        return int(len(self.x_train))

    @property
    def n_test(self) -> int:
        return int(len(self.x_test)) if self.x_test is not None else 0

    @property
    def input_shape(self) -> tuple:
        return tuple(self.x_train.shape[1:])

    @property
    def input_is_tokens(self) -> bool:
        return self.modality in ("tokens", "text")

    @property
    def num_classes(self) -> int | None:
        return len(self.class_names) if self.class_names is not None else None

    @property
    def output_size(self) -> int:
        if self.task == "classification":
            return self.num_classes
        if self.task == "language_model":
            return self.tokenizer.vocab_size
        if self.y_train is not None and self.y_train.ndim > 1:
            return int(self.y_train.shape[1])
        return 1

    def describe(self) -> str:
        return f"dataset '{self.name}'"

    def __repr__(self):
        return f"<Dataset {self.name}: {self.summary()}>"

    def summary(self) -> str:
        shape = "×".join(str(s) for s in self.input_shape)
        parts = [f"{self.n_train:,} training", f"{self.n_test:,} test examples"]
        if self.modality == "image":
            what = f"images {shape}"
        elif self.modality == "tabular":
            what = f"{self.input_shape[0]} features"
        elif self.modality == "text":
            what = f"texts up to {self.context} tokens"
        elif self.modality == "sequence":
            what = f"sequences of {self.input_shape[0]} steps"
        else:
            what = shape
        task = {"classification": f"{self.num_classes} classes", "regression": "predict a number"}.get(self.task, self.task)
        return f"{self.name}: {', '.join(parts)} — {what}, {task}"

    # -- tensors ----------------------------------------------------------------
    def prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        """Turn stored inputs into what the model sees (e.g. uint8 images -> normalised floats)."""
        if self.modality == "image":
            x = x.float() / 255.0
            if self.image_stats is not None:
                mean, std = self.image_stats
                mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
                std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
                x = (x - mean) / std
        return x

    def unprepare_image(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of prepare_x for images -> floats in 0..1 (for showing)."""
        if self.image_stats is not None:
            mean, std = self.image_stats
            mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            x = x * std + mean
        return x.clamp(0, 1)

    def _tensors(self, split: str, device):
        key = (split, str(device))
        if key not in self._dev_cache:
            x, y = (self.x_train, self.y_train) if split == "train" else (self.x_test, self.y_test)
            nbytes = x.element_size() * x.nelement()
            if nbytes < 3 * 2**30 or str(device) == "cpu":
                self._dev_cache[key] = (x.to(device), y.to(device))
            else:
                self._dev_cache[key] = (x, y)
        return self._dev_cache[key]

    def batches(self, split="train", batch_size=32, shuffle=True, device="cpu", augment=False, limit=None):
        x, y = self._tensors(split, device)
        n = len(x)
        if limit:
            n = min(n, int(limit))
        idx = torch.randperm(len(x), device=x.device)[:n] if shuffle else torch.arange(n, device=x.device)
        bs = max(1, int(batch_size))
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            xb = x[j].to(device, non_blocking=True)
            yb = y[j].to(device, non_blocking=True)
            xb = self.prepare_x(xb)
            if augment and self.augment and self.modality == "image":
                xb = augment_images(xb, self.augment)
            yield xb, yb

    def example_inputs(self, n=1, split="train", device="cpu"):
        x, _ = self._tensors(split, device)
        return self.prepare_x(x[:n].to(device))

    def num_batches(self, batch_size, split="train"):
        n = self.n_train if split == "train" else self.n_test
        return max(1, math.ceil(n / max(1, int(batch_size))))

    # -- regression targets ------------------------------------------------------
    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.y_mean is not None:
            return y * torch.as_tensor(self.y_std, dtype=y.dtype, device=y.device) + \
                torch.as_tensor(self.y_mean, dtype=y.dtype, device=y.device)
        return y

    # -- raw inputs (for 'prediction of model for …') ------------------------------
    def encode_raw(self, raw) -> torch.Tensor:
        """Convert a user-supplied example (list of numbers / text) into a model input batch of 1."""
        if self.modality in ("text",):
            ids = self.tokenizer.encode(str(raw))[: self.context]
            ids = [self.tokenizer.pad_id or 0] * (self.context - len(ids)) + ids
            return torch.tensor([ids], dtype=torch.long)
        if self.modality == "tokens":
            ids = self.tokenizer.encode(str(raw))[-self.context:]
            return torch.tensor([ids], dtype=torch.long)
        vals = raw if isinstance(raw, (list, tuple, np.ndarray)) else [raw]
        arr = np.asarray([float(v) for v in np.ravel(vals)], dtype=np.float32)
        expected = int(np.prod(self.input_shape))
        if arr.size != expected:
            raise NBError(f"This model expects {expected} input numbers but got {arr.size}.",
                          hint=f"Inputs: {', '.join(self.feature_names or [])}" if self.feature_names else None)
        if self.modality == "tabular" and self.x_mean is not None:
            arr = (arr - self.x_mean) / self.x_std
        x = torch.tensor(arr).view(1, *self.input_shape)
        if self.modality == "image":
            x = x.clamp(0, 255).to(torch.uint8)
            x = self.prepare_x(x)
        return x

    # -- metadata saved with models -----------------------------------------------
    def meta(self) -> dict:
        return {
            "name": self.name, "task": self.task, "modality": self.modality,
            "input_shape": list(self.input_shape), "class_names": self.class_names,
            "feature_names": self.feature_names, "target_names": self.target_names,
            "tokenizer": self.tokenizer.to_dict() if self.tokenizer is not None else None,
            "context": self.context, "image_stats": [list(map(float, s)) for s in self.image_stats]
            if self.image_stats is not None else None,
            "x_mean": None if self.x_mean is None else np.asarray(self.x_mean).tolist(),
            "x_std": None if self.x_std is None else np.asarray(self.x_std).tolist(),
            "y_mean": None if self.y_mean is None else np.asarray(self.y_mean).tolist(),
            "y_std": None if self.y_std is None else np.asarray(self.y_std).tolist(),
        }

    # -- previews ---------------------------------------------------------------
    def preview(self) -> dict:
        try:
            if self.modality == "image":
                n = min(32, self.n_train)
                imgs = self.x_train[:n]
                return {"kind": "images", "png": image_grid_png(imgs, cols=8),
                        "labels": [self.label_name(int(v)) for v in self.y_train[:n]]}
            if self.modality == "tabular":
                xs = self.x_train[:200].numpy()
                if self.x_mean is not None:
                    xs = xs * self.x_std + self.x_mean
                ys = self.y_train[:200]
                if self.task == "regression":
                    ys = self.denormalize_y(ys)
                if self.input_shape[0] <= 2:
                    pts = [[float(r[0]), float(r[1]) if len(r) > 1 else 0.0, self._yval(y)] for r, y in zip(xs, ys)]
                    return {"kind": "points", "points": pts, "classes": self.class_names,
                            "dims": int(self.input_shape[0]), "task": self.task,
                            "features": self.feature_names}
                cols = (self.feature_names or [f"x{i}" for i in range(self.input_shape[0])]) + ["→ " + (
                    "class" if self.task == "classification" else (self.target_names or ["target"])[0])]
                rows = [[round(float(v), 4) for v in r] + [self._yval(y)] for r, y in zip(xs[:10], ys[:10])]
                return {"kind": "table", "columns": cols, "rows": rows}
            if self.modality == "tokens":
                return {"kind": "text", "text": getattr(self, "raw_text", "")[:1500]}
            if self.modality == "text":
                rows = [[t, self.label_name(int(y))] for t, y in zip(self.texts_train[:12], self.y_train[:12])]
                return {"kind": "table", "columns": ["text", "label"], "rows": rows}
            if self.modality == "sequence" and self.raw_series is not None:
                return {"kind": "series", "values": [round(float(v), 4) for v in self.raw_series[:400]]}
        except Exception as e:  # previews must never break a run
            return {"kind": "none", "error": str(e)}
        return {"kind": "none"}

    def _yval(self, y):
        if self.task == "classification":
            return self.label_name(int(y))
        y = y.reshape(-1)
        return round(float(y[0]), 4)

    def label_name(self, i: int) -> str:
        if self.class_names and 0 <= i < len(self.class_names):
            return str(self.class_names[i])
        return str(i)

    def class_counts(self):
        if self.task != "classification":
            return None
        counts = torch.bincount(self.y_train.view(-1), minlength=self.num_classes)
        return [int(c) for c in counts]

    def announce(self, block_id=None):
        info = {
            "summary": self.summary(), "task": self.task, "modality": self.modality,
            "n_train": self.n_train, "n_test": self.n_test, "input_shape": list(self.input_shape),
            "classes": self.class_names, "class_counts": self.class_counts(),
            "features": self.feature_names, "description": self.description, "source": self.source,
            "tokenizer": self.tokenizer.describe() if self.tokenizer is not None else None,
            "preview": self.preview(),
        }
        emit("dataset", name=self.name, info=info, block=block_id)


class TextDataset(Dataset):
    """A long text for training a language model to predict the next token."""

    def __init__(self, name: str, text: str, tokenizer: Tokenizer, context: int, val: float = 0.1,
                 description: str = "", source: str = ""):
        ids = tokenizer.encode_array(text)
        if len(ids) < context + 2:
            raise NBError(f"The text is too short ({len(ids)} tokens) for a context length of {context}.",
                          hint="Use a longer text or a smaller context length.")
        ids = torch.from_numpy(ids).long()
        val = float(val or 0)
        if val > 1:
            val /= 100
        n_val = int(len(ids) * min(max(val, 0.0), 0.5))
        if 0 < n_val < context + 2:
            n_val = min(len(ids) // 2, context + 2)
        train_ids = ids[: len(ids) - n_val] if n_val else ids
        val_ids = ids[len(ids) - n_val:] if n_val else ids[-(context + 2) * 4:]
        super().__init__(name, task="language_model", modality="tokens",
                         x_train=train_ids, y_train=None, x_test=val_ids, y_test=None,
                         tokenizer=tokenizer, context=int(context), description=description, source=source)
        self.raw_text = text

    @property
    def n_train(self):
        return int(len(self.x_train))

    @property
    def n_test(self):
        return int(len(self.x_test))

    @property
    def input_shape(self):
        return (self.context,)

    def summary(self):
        return (f"{self.name}: {len(self.raw_text):,} characters → {self.n_train:,} training + "
                f"{self.n_test:,} validation tokens, {self.tokenizer.describe()}, context {self.context}")

    def lm_batch(self, split="train", batch_size=32, device="cpu", context=None):
        data, _ = self._tensors(split, device)
        T = int(context or self.context)
        T = min(T, len(data) - 2)
        ix = torch.randint(len(data) - T - 1, (int(batch_size),), device=data.device)
        offs = torch.arange(T, device=data.device)
        x = data[ix[:, None] + offs]
        y = data[ix[:, None] + offs + 1]
        return x.to(device), y.to(device)

    def _tensors(self, split, device):
        key = (split, str(device))
        if key not in self._dev_cache:
            data = self.x_train if split == "train" else self.x_test
            self._dev_cache[key] = (data.to(device), None)
        return self._dev_cache[key]

    def batches(self, *a, **k):
        raise NBError("Language-model data is used with 'for N steps' training.")

    def num_batches(self, batch_size, split="train"):
        n = self.n_train if split == "train" else self.n_test
        return max(1, n // (max(1, int(batch_size)) * self.context))

    def retokenized(self, tokenizer: Tokenizer, context: int | None = None) -> "TextDataset":
        """The same text, tokenized differently (e.g. for a pretrained model)."""
        n_val = len(self.x_test)
        frac = n_val / max(1, n_val + len(self.x_train))
        return TextDataset(self.name, self.raw_text, tokenizer, int(context or self.context), val=frac,
                           description=self.description, source=self.source)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def image_grid_png(imgs, cols=8, scale=None) -> str:
    """uint8 or float (N,C,H,W) -> base64 PNG grid."""
    x = imgs.detach().cpu() if isinstance(imgs, torch.Tensor) else torch.as_tensor(imgs)
    if x.dtype != torch.uint8:
        x = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    n, c, h, w = x.shape
    if scale is None:
        scale = max(1, int(round(56 / max(h, w))))
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    pad = 2
    grid = torch.full((3, rows * (h * scale + pad) + pad, cols * (w * scale + pad) + pad), 255, dtype=torch.uint8)
    for i in range(n):
        img = x[i]
        if c == 1:
            img = img.repeat(3, 1, 1)
        img = img.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        r, cc = divmod(i, cols)
        y0 = pad + r * (h * scale + pad)
        x0 = pad + cc * (w * scale + pad)
        grid[:, y0:y0 + h * scale, x0:x0 + w * scale] = img[:3]
    return png_base64(grid.permute(1, 2, 0).numpy())


def augment_images(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Random flips / shifts / rotations, applied per image (works on CPU and GPU)."""
    import torch.nn.functional as F
    n = x.shape[0]
    mode = (mode or "").lower()
    dev, dt = x.device, x.dtype
    angle = torch.zeros(n, device=dev, dtype=dt)
    sx = torch.ones(n, device=dev, dtype=dt)
    tx = torch.zeros(n, device=dev, dtype=dt)
    ty = torch.zeros(n, device=dev, dtype=dt)
    if "flip" in mode:
        sx = torch.where(torch.rand(n, device=dev) < 0.5, -sx, sx)
    if "shift" in mode or "rotate" in mode:
        tx = (torch.rand(n, device=dev, dtype=dt) - 0.5) * 0.2
        ty = (torch.rand(n, device=dev, dtype=dt) - 0.5) * 0.2
    if "rotate" in mode:
        angle = (torch.rand(n, device=dev, dtype=dt) - 0.5) * (math.pi / 6)
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.stack([torch.stack([cos * sx, -sin, tx], 1), torch.stack([sin * sx, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def _resize_uint8(x: np.ndarray, size: int) -> np.ndarray:
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return t.round().clamp(0, 255).to(torch.uint8).numpy()


def _image_dataset(name, xtr, ytr, xte, yte, class_names, *, size=None, grayscale=False, augment=None,
                   description="", source=""):
    if grayscale and xtr.shape[1] == 3:
        w = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 3, 1, 1)
        xtr = (xtr.astype(np.float32) * w).sum(1, keepdims=True).round().astype(np.uint8)
        xte = (xte.astype(np.float32) * w).sum(1, keepdims=True).round().astype(np.uint8)
    if size:
        size = int(size)
        if size != xtr.shape[-1]:
            xtr = _resize_uint8(xtr, size)
            xte = _resize_uint8(xte, size)
    xt = torch.from_numpy(np.ascontiguousarray(xtr))
    f = xt.float() / 255.0
    mean = f.mean(dim=(0, 2, 3)).tolist()
    std = (f.std(dim=(0, 2, 3)) + 1e-6).tolist()
    return Dataset(name, task="classification", modality="image",
                   x_train=xt, y_train=torch.from_numpy(ytr).long(),
                   x_test=torch.from_numpy(np.ascontiguousarray(xte)), y_test=torch.from_numpy(yte).long(),
                   class_names=class_names, image_stats=(mean, std), augment=augment,
                   description=description, source=source)


# ---------------------------------------------------------------------------
# Public loaders (one per data block)
# ---------------------------------------------------------------------------


def _finish(ds: Dataset, bid=None) -> Dataset:
    register(ds)
    ds.announce(bid)
    return ds


def toy(kind="spirals", name="data", samples=None, noise=None, classes=None, test=0.2, seed=None,
        standardize=False, _bid=None, **unused):
    """Generated 2-D toy data (spirals, moons, circles, …) or 1-D curves for regression."""
    _warn_unused(name, unused)
    rng = _rng(seed)
    n = int(samples) if samples else 500
    X, y, task, classes_ = src.toy_points(kind, n, None if noise is None else float(noise),
                                          None if classes is None else int(classes), rng)
    tr, te = _split(len(X), test, rng)
    x_mean = x_std = None
    if standardize:
        x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
        X = (X - x_mean) / x_std
    y_mean = y_std = None
    if task == "regression":
        y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-8
        y = (y - y_mean) / y_std
    feats = ["x", "y"] if X.shape[1] == 2 else ["x"]
    nice = {"spirals": "Two spirals", "moons": "Two moons", "circles": "Circles", "blobs": "Blobs",
            "xor": "XOR", "checkerboard": "Checkerboard", "sine": "Sine wave", "polynomial": "Polynomial curve",
            "linear": "Straight line", "steps": "Staircase"}.get(kind, kind)
    ds = Dataset(name, task=task, modality="tabular",
                 x_train=torch.from_numpy(X[tr]), y_train=torch.from_numpy(y[tr]),
                 x_test=torch.from_numpy(X[te]), y_test=torch.from_numpy(y[te]),
                 class_names=classes_, feature_names=feats, target_names=["y"] if task == "regression" else None,
                 x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                 description=f"{nice}: {n} generated points.", source="built-in generator")
    return _finish(ds, _bid)


def table(kind="iris", name="data", test=0.2, standardize=True, samples=None, seed=None, _bid=None, **unused):
    """Classic small tables (iris, wine, breast cancer, diabetes, California housing)."""
    _warn_unused(name, unused)
    X, y, task, feats, classes_, desc = src.sklearn_table(kind)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32 if task == "regression" else np.int64)
    rng = _rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    X, y = _limit(X, y, samples)
    tr, te = _split(len(X), test, rng)
    x_mean = x_std = None
    if standardize:
        x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
        X = (X - x_mean) / x_std
    y_mean = y_std = None
    if task == "regression":
        y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-8
        y = (y - y_mean) / y_std
    ds = Dataset(name, task=task, modality="tabular",
                 x_train=torch.from_numpy(X[tr]), y_train=torch.from_numpy(y[tr]),
                 x_test=torch.from_numpy(X[te]), y_test=torch.from_numpy(y[te]),
                 class_names=classes_, feature_names=feats, target_names=["target"] if task == "regression" else None,
                 x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, description=desc,
                 source="scikit-learn")
    return _finish(ds, _bid)


def images(kind="digits", name="data", samples=None, test=None, size=None, grayscale=False, augment=None,
           seed=None, _bid=None, **unused):
    """Image classification datasets: digits (8×8, offline), MNIST, Fashion-MNIST, CIFAR-10."""
    _warn_unused(name, unused)
    kind = kind.lower()
    rng = _rng(seed)
    if kind == "digits":
        x, y = src.digits8()
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]
        tr, te = _split(len(x), 0.2 if test is None else test, rng)
        xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
        classes_ = [str(i) for i in range(10)]
        desc = "1,797 tiny 8×8 handwritten digits (built in, no download)."
        source = "scikit-learn"
    elif kind in ("mnist", "fashion"):
        if kind == "mnist":
            xtr, ytr, xte, yte = src.idx_dataset("mnist", src.MNIST_MIRRORS, "MNIST")
            classes_ = [str(i) for i in range(10)]
            desc = "70,000 handwritten digits, 28×28 pixels."
        else:
            xtr, ytr, xte, yte = src.idx_dataset("fashion", src.FASHION_MIRRORS, "Fashion-MNIST")
            classes_ = src.FASHION_CLASSES
            desc = "70,000 photos of clothes, 28×28 pixels, 10 kinds."
        source = "downloaded"
    elif kind == "cifar10":
        xtr, ytr, xte, yte = src.cifar10()
        classes_ = src.CIFAR_CLASSES
        desc = "60,000 small colour photos (32×32) of 10 kinds of things."
        source = "downloaded"
    else:
        raise NBError(f"Unknown image dataset '{kind}'.")
    if samples:
        n = int(samples)
        xtr, ytr = xtr[:n], ytr[:n]
        xte, yte = xte[: max(1, n // 5)], yte[: max(1, n // 5)]
    return _finish(_image_dataset(name, xtr, ytr, xte, yte, classes_, size=size, grayscale=grayscale,
                                  augment=augment, description=desc, source=source), _bid)


def image_folder(path, name="data", size=64, test=0.2, augment=None, grayscale=False, samples=None, seed=None,
                 _bid=None, **unused):
    """Your own pictures: one sub-folder per class (folder/cats/*.jpg, folder/dogs/*.jpg …)."""
    from PIL import Image
    _warn_unused(name, unused)
    root = paths.resolve_data_file(str(path))
    if not root.is_dir():
        raise NBError(f"Folder not found: {root}", hint="Make a folder with one sub-folder per class.")
    classes_ = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(classes_) < 2:
        raise NBError("The image folder needs at least two sub-folders (one per class).")
    size = int(size or 64)
    xs, ys = [], []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    for ci, c in enumerate(classes_):
        files = sorted(p for p in (root / c).rglob("*") if p.suffix.lower() in exts)
        if samples:
            files = files[: int(samples)]
        for p in files:
            try:
                img = Image.open(p).convert("RGB").resize((size, size))
            except Exception:
                continue
            xs.append(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1))
            ys.append(ci)
    if not xs:
        raise NBError("No images found in the folder.")
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    rng = _rng(seed)
    tr, te = _split(len(x), test, rng)
    return _finish(_image_dataset(name, x[tr], y[tr], x[te], y[te], classes_, grayscale=grayscale,
                                  augment=augment, description=f"{len(x)} images from {root.name}",
                                  source=str(root)), _bid)


_TEXT_KINDS = {
    "shakespeare": ("Tiny Shakespeare", "40,000 lines from Shakespeare's plays (1 MB)."),
    "toy_stories": ("Toy stories", "Simple made-up children's stories (built in, no download)."),
    "names": ("Baby names", "32,000 first names, one per line — teach a model to invent names."),
    "python": ("Python code", "Python source code from this computer's standard library."),
}


def _text_source(kind: str) -> str:
    if kind == "shakespeare":
        return src.shakespeare()
    if kind == "toy_stories":
        return src.toy_stories()
    if kind == "names":
        return src.names()
    if kind == "python":
        return src.python_code()
    raise NBError(f"Unknown text dataset '{kind}'.")


def _text_dataset(name, raw: str, *, tokenizer="characters", context=128, val=0.1, samples=None,
                  lowercase=False, vocab=5000, description="", source="", bid=None):
    if samples:
        raw = raw[: int(samples)]
    if lowercase:
        raw = raw.lower()
    tok = make_tokenizer(tokenizer, raw, max_vocab=int(vocab or 5000), lowercase=False)
    ds = TextDataset(name, raw, tok, int(context or 128), val=val, description=description, source=source)
    return _finish(ds, bid)


def text(kind="shakespeare", name="data", tokenizer="characters", context=128, val=0.1, samples=None,
         lowercase=False, vocab=5000, test=None, _bid=None, **unused):
    """A built-in text for training a language model."""
    _warn_unused(name, unused)
    title, desc = _TEXT_KINDS.get(kind, (kind, ""))
    raw = _text_source(kind)
    return _text_dataset(name, raw, tokenizer=tokenizer, context=context, val=val if test is None else test,
                         samples=samples, lowercase=lowercase, vocab=vocab, description=desc, source=title,
                         bid=_bid)


def text_file(path, name="data", tokenizer="characters", context=128, val=0.1, samples=None, lowercase=False,
              vocab=5000, test=None, _bid=None, **unused):
    """Your own .txt file as a language-model dataset."""
    _warn_unused(name, unused)
    p = paths.resolve_data_file(str(path))
    if not p.exists():
        raise NBError(f"Text file not found: {p}", hint="Upload it from the Data tab or put it in the data folder.")
    raw = p.read_text(encoding="utf-8", errors="replace")
    return _text_dataset(name, raw, tokenizer=tokenizer, context=context, val=val if test is None else test,
                         samples=samples, lowercase=lowercase, vocab=vocab,
                         description=f"Text from {p.name}", source=str(p), bid=_bid)


def text_url(url, name="data", tokenizer="characters", context=128, val=0.1, samples=None, lowercase=False,
             vocab=5000, test=None, _bid=None, **unused):
    """Text downloaded from a web address."""
    import hashlib
    _warn_unused(name, unused)
    dest = src.cache_path("web", hashlib.sha1(str(url).encode()).hexdigest()[:16] + ".txt")
    src.download([str(url)], dest, str(url)[:60])
    raw = dest.read_text(encoding="utf-8", errors="replace")
    if "<html" in raw[:2000].lower():
        raw = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
    return _text_dataset(name, raw, tokenizer=tokenizer, context=context, val=val if test is None else test,
                         samples=samples, lowercase=lowercase, vocab=vocab, description=f"Text from {url}",
                         source=str(url), bid=_bid)


def _classification_text_dataset(name, texts, labels, class_names, *, tokenizer="words", context=48, test=0.2,
                                 vocab=5000, lowercase=True, seed=None, description="", source="", bid=None):
    rng = _rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    tr, te = _split(len(texts), test, rng)
    texts_tr = [texts[i] for i in tr]
    texts_te = [texts[i] for i in te]
    tok = make_tokenizer(tokenizer, texts_tr, max_vocab=int(vocab or 5000), lowercase=bool(lowercase), pad=True)
    T = int(context or 48)
    lengths = sorted(len(tok.encode(t)) for t in texts_tr[:2000])
    longest = lengths[min(len(lengths) - 1, int(len(lengths) * 0.98))] if lengths else T
    if longest < T:
        emit("log", level="info", text=f"Texts in '{name}' are at most ~{longest} tokens long, so the context "
                                        f"length was shortened from {T} to {max(longest, 4)}.")
        T = max(longest, 4)
    pad = tok.pad_id or 0

    def enc(ts):
        # Pad on the left so the *last* step is always real text (matters for LSTMs / 'last step').
        out = np.full((len(ts), T), pad, dtype=np.int64)
        for i, t in enumerate(ts):
            ids = tok.encode(t)[:T]
            if ids:
                out[i, T - len(ids):] = ids
        return torch.from_numpy(out)

    ds = Dataset(name, task="classification", modality="text", x_train=enc(texts_tr),
                 y_train=torch.from_numpy(labels[tr]), x_test=enc(texts_te), y_test=torch.from_numpy(labels[te]),
                 class_names=class_names, tokenizer=tok, context=T, texts_train=texts_tr, texts_test=texts_te,
                 description=description, source=source)
    return _finish(ds, bid)


def text_classes(kind="toy_sentiment", name="data", tokenizer="words", context=48, test=0.2, vocab=5000,
                 lowercase=True, samples=None, seed=None, _bid=None, **unused):
    """Texts with labels — e.g. reviews labelled positive / negative."""
    _warn_unused(name, unused)
    if kind == "toy_sentiment":
        texts, labels, classes_ = src.toy_sentiment(int(samples) if samples else 3000)
        desc = "Made-up one-line reviews labelled positive or negative (watch out for 'not')."
        source = "built-in generator"
    elif kind in ("imdb", "ag_news", "sst2"):
        hf_id, tcol, lcol = {"imdb": ("imdb", "text", "label"), "ag_news": ("ag_news", "text", "label"),
                             "sst2": ("glue/sst2", "sentence", "label")}[kind]
        return huggingface(hf_id, name=name, text_column=tcol, label_column=lcol, tokenizer=tokenizer,
                           context=context, samples=samples or 20000, test=test, vocab=vocab, _bid=_bid)
    else:
        raise NBError(f"Unknown text classification dataset '{kind}'.")
    return _classification_text_dataset(name, texts, labels, classes_, tokenizer=tokenizer, context=context,
                                        test=test, vocab=vocab, lowercase=lowercase, seed=seed,
                                        description=desc, source=source, bid=_bid)


def csv(path, name="data", target="", task=None, test=0.2, standardize=True, samples=None, seed=None,
        tokenizer="words", context=64, vocab=5000, _bid=None, **unused):
    """A CSV spreadsheet: predict one column from the others."""
    try:
        import pandas as pd
    except ImportError:
        raise missing_package("pandas", "Loading CSV files") from None
    _warn_unused(name, unused)
    p = paths.resolve_data_file(str(path))
    if not p.exists():
        raise NBError(f"CSV file not found: {p}", hint="Upload it from the Data tab or put it in the data folder.")
    df = pd.read_csv(p)
    if samples:
        df = df.iloc[: int(samples)]
    target = str(target or "").strip() or df.columns[-1]
    if target not in df.columns:
        lower = {c.lower(): c for c in df.columns}
        if target.lower() in lower:
            target = lower[target.lower()]
        else:
            raise NBError(f"The CSV has no column called '{target}'.",
                          hint="Columns: " + ", ".join(map(str, df.columns)))
    df = df.dropna(subset=[target])
    yser = df[target]
    feats = df.drop(columns=[target])
    ptypes = pd.api.types

    def is_category(s) -> bool:
        # Text (object or pandas' string dtype), booleans and categoricals are categories.
        return (ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s) or ptypes.is_bool_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype))

    if task in (None, "", "auto"):
        if is_category(yser) or (ptypes.is_integer_dtype(yser) and yser.nunique() <= 20):
            task = "classification"
        else:
            task = "regression"
    # One long-text column -> text classification.
    obj_cols = [c for c in feats.columns if ptypes.is_object_dtype(feats[c]) or ptypes.is_string_dtype(feats[c])]
    text_cols = [c for c in obj_cols if feats[c].astype(str).str.len().mean() > 30]
    if task == "classification" and len(text_cols) == 1 and feats.shape[1] == 1:
        classes_ = sorted(map(str, yser.unique()))
        idx = {c: i for i, c in enumerate(classes_)}
        labels = np.asarray([idx[str(v)] for v in yser], dtype=np.int64)
        return _classification_text_dataset(name, feats[text_cols[0]].astype(str).tolist(), labels, classes_,
                                            tokenizer=tokenizer, context=context, test=test, vocab=vocab,
                                            seed=seed, description=f"Texts from {p.name}", source=str(p), bid=_bid)
    for c in text_cols:
        emit("log", level="warn", text=f"Column '{c}' looks like free text, so it was ignored.")
    feats = feats.drop(columns=text_cols)
    # Categorical columns -> one-hot; numbers -> fill gaps with the median.
    cat_cols = [c for c in feats.columns if is_category(feats[c])]
    if cat_cols:
        feats = pd.get_dummies(feats, columns=cat_cols, dummy_na=False, dtype=np.float32)
    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0)
    X = feats.to_numpy(dtype=np.float32)
    feature_names = [str(c) for c in feats.columns]
    if task == "classification":
        classes_ = sorted(map(str, yser.unique()), key=lambda s: (len(s), s))
        try:
            classes_ = sorted(classes_, key=lambda s: float(s))
        except ValueError:
            pass
        idx = {c: i for i, c in enumerate(classes_)}
        y = np.asarray([idx[str(v)] for v in yser], dtype=np.int64)
    else:
        classes_ = None
        y = pd.to_numeric(yser, errors="coerce").fillna(0).to_numpy(dtype=np.float32)[:, None]
    rng = _rng(seed)
    tr, te = _split(len(X), test, rng)
    x_mean = x_std = None
    if standardize:
        x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
        X = (X - x_mean) / x_std
    y_mean = y_std = None
    if task == "regression":
        y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-8
        y = (y - y_mean) / y_std
    ds = Dataset(name, task=task, modality="tabular", x_train=torch.from_numpy(X[tr]),
                 y_train=torch.from_numpy(y[tr]), x_test=torch.from_numpy(X[te]), y_test=torch.from_numpy(y[te]),
                 class_names=classes_, feature_names=feature_names,
                 target_names=[str(target)] if task == "regression" else None,
                 x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                 description=f"{len(df):,} rows from {p.name}; predicting '{target}'.", source=str(p))
    return _finish(ds, _bid)


def huggingface(dataset_id, name="data", split="train", text_column="text", label_column="", config="",
                tokenizer="characters", context=128, samples=None, test=0.1, vocab=5000, lowercase=False,
                _bid=None, **unused):
    """Any dataset from the Hugging Face hub (needs ``pip install datasets``)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise missing_package("datasets", "Hugging Face datasets") from None
    _warn_unused(name, unused)
    parts = str(dataset_id).split("/")
    if len(parts) == 2 and not config and parts[0] == "glue":
        dataset_id, config = parts
    emit("log", level="info", text=f"Loading '{dataset_id}' from Hugging Face…")
    ds = load_dataset(dataset_id, config or None, split=split)
    if samples:
        ds = ds.shuffle(seed=0).select(range(min(int(samples), len(ds))))
    cols = ds.column_names
    tcol = text_column if text_column in cols else next((c for c in cols if c in ("text", "sentence", "content")), cols[0])
    if label_column and label_column in cols:
        texts = [str(t) for t in ds[tcol]]
        raw_labels = ds[label_column]
        feat = ds.features.get(label_column)
        names_ = getattr(feat, "names", None)
        if names_:
            labels = np.asarray(raw_labels, dtype=np.int64)
            classes_ = list(names_)
        else:
            classes_ = sorted(set(map(str, raw_labels)))
            idx = {c: i for i, c in enumerate(classes_)}
            labels = np.asarray([idx[str(v)] for v in raw_labels], dtype=np.int64)
        return _classification_text_dataset(name, texts, labels, classes_, tokenizer=tokenizer
                                            if tokenizer != "characters" else "words", context=context, test=test,
                                            vocab=vocab, lowercase=lowercase,
                                            description=f"Hugging Face: {dataset_id}", source=str(dataset_id),
                                            bid=_bid)
    raw = "\n\n".join(str(t) for t in ds[tcol])
    return _text_dataset(name, raw, tokenizer=tokenizer, context=context, val=test, lowercase=lowercase,
                         vocab=vocab, description=f"Hugging Face: {dataset_id}", source=str(dataset_id), bid=_bid)


def series(kind="sine", name="data", window=32, horizon=1, samples=2000, noise=0.05, test=0.2, seed=None,
           _bid=None, **unused):
    """A time series cut into windows: look at the last N values, predict the next ones."""
    _warn_unused(name, unused)
    rng = _rng(seed)
    window, horizon = int(window or 32), int(horizon or 1)
    length = int(samples or 2000) + window + horizon
    s = src.toy_series(kind, length, float(noise if noise is not None else 0.05), rng)
    mean, std = float(s.mean()), float(s.std() + 1e-8)
    z = (s - mean) / std
    n = len(z) - window - horizon + 1
    X = np.stack([z[i:i + window] for i in range(n)])[:, :, None].astype(np.float32)
    Y = np.stack([z[i + window:i + window + horizon] for i in range(n)]).astype(np.float32)
    n_test = int(n * min(max(float(test or 0.2) if float(test or 0.2) <= 1 else float(test) / 100, 0.05), 0.5))
    ds = Dataset(name, task="regression", modality="sequence",
                 x_train=torch.from_numpy(X[: n - n_test]), y_train=torch.from_numpy(Y[: n - n_test]),
                 x_test=torch.from_numpy(X[n - n_test:]), y_test=torch.from_numpy(Y[n - n_test:]),
                 target_names=[f"t+{i + 1}" for i in range(horizon)], y_mean=np.zeros(horizon, np.float32),
                 y_std=np.ones(horizon, np.float32), raw_series=z,
                 description=f"{kind.replace('_', ' ')} series: use {window} past values to predict the next "
                             f"{horizon}.", source="built-in generator")
    ds.series_scale = (mean, std)
    return _finish(ds, _bid)


def size(data) -> int:
    if not isinstance(data, Dataset):
        raise NBError("That isn't a dataset.")
    return data.n_train + data.n_test


def class_count(data) -> int:
    if not isinstance(data, Dataset):
        raise NBError("That isn't a dataset.")
    return data.num_classes or 0
