"""Testing models: scores, confusion matrices, prediction galleries, decision maps, data plots."""
from __future__ import annotations

import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from .core import STATE, text as fmt_text
from .errors import NBError
from .events import emit, emitter, png_base64

__all__ = ["evaluate", "score", "show_predictions", "show_confusion", "show_decision_map", "predict",
           "show_examples", "plot_data", "show_map", "emit_image", "PALETTE", "meta_dataset"]

PALETTE = ["#F28E2B", "#4E79A7", "#59A14F", "#B07AA1", "#FF9DA7", "#9C755F", "#76B7B2", "#EDC948",
           "#E15759", "#BAB0AC"]


def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def emit_image(title: str, png: str, caption: str = "", block=None, key: str | None = None):
    ev = {"title": title, "png": png, "caption": caption, "block": block, "key": key or title}
    em = emitter()
    if em.mode == "cli" and em.run_dir is not None:
        import base64
        slug = re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-").lower()[:60] or "image"
        d = em.run_dir / "images"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{slug}.png"
        path.write_bytes(base64.b64decode(png))
        ev["saved"] = str(path)
    emit("image", **ev)


def meta_dataset(meta: dict):
    """A data-less Dataset rebuilt from a saved model's metadata (for encoding inputs)."""
    from .data import Dataset
    from .tokenizers import tokenizer_from_dict
    shape = tuple(meta.get("input_shape") or [1])
    tok = tokenizer_from_dict(meta["tokenizer"]) if meta.get("tokenizer") else None
    dtype = torch.long if meta.get("modality") in ("tokens", "text") else torch.float32
    empty = torch.zeros((0, *shape), dtype=dtype)
    stats = meta.get("image_stats")
    ds = Dataset(meta.get("name", "data"), task=meta["task"], modality=meta["modality"], x_train=empty,
                 y_train=torch.zeros(0, dtype=torch.long), x_test=empty, y_test=torch.zeros(0, dtype=torch.long),
                 class_names=meta.get("class_names"), feature_names=meta.get("feature_names"),
                 target_names=meta.get("target_names"), tokenizer=tok, context=meta.get("context"),
                 image_stats=tuple(stats) if stats else None,
                 x_mean=None if meta.get("x_mean") is None else np.asarray(meta["x_mean"], dtype=np.float32),
                 x_std=None if meta.get("x_std") is None else np.asarray(meta["x_std"], dtype=np.float32),
                 y_mean=None if meta.get("y_mean") is None else np.asarray(meta["y_mean"], dtype=np.float32),
                 y_std=None if meta.get("y_std") is None else np.asarray(meta["y_std"], dtype=np.float32))
    return ds


# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------


def _require_ready(model, data):
    fam = getattr(model, "family", None)
    if fam is None:
        raise NBError("That isn't a model.")
    if fam == "classic":
        if not model.fitted:
            raise NBError(f"'{model.name}' hasn't been trained yet.", hint="Use a 'train' block first.")
        return
    if fam == "agent":
        raise NBError(f"'{model.name}' learned in a simulation — use 'watch … play' to test it.")
    if fam == "generator":
        raise NBError(f"'{model.name}' makes new pictures rather than answering questions — use "
                      f"'show new pictures from …' to see what it learned.")
    if fam == "pretrained_lm":
        return
    if model.net is None:
        if data is None:
            raise NBError(f"'{model.name}' hasn't been trained yet.")
        emit("log", level="warn", text=f"'{model.name}' hasn't been trained, so its predictions are random guesses.")
        model.build_for(data)
    meta = model.meta or {}
    if data is not None and getattr(model, "family", None) == "neural":
        if meta.get("task") != data.task:
            raise NBError(f"'{model.name}' was trained for {meta.get('task')} but '{data.name}' is "
                          f"{data.task} data.")
        if tuple(meta.get("input_shape") or ()) != tuple(data.input_shape):
            raise NBError(f"'{model.name}' expects inputs shaped {tuple(meta.get('input_shape'))} but "
                          f"'{data.name}' has {tuple(data.input_shape)}.")


def _split_tensors(data, split):
    if split == "test" and data.n_test == 0:
        emit("log", level="warn", text=f"'{data.name}' has no test examples, so the training examples were used.")
        split = "train"
    if split == "train":
        return data.x_train, data.y_train, split
    return data.x_test, data.y_test, split


@torch.no_grad()
def _raw_outputs(model, data, x: torch.Tensor) -> torch.Tensor:
    net = model.net
    dev = model.device
    was = net.training
    net.eval()
    outs = []
    try:
        for i in range(0, len(x), 512):
            xb = data.prepare_x(x[i:i + 512].to(dev))
            outs.append(net(xb).float().cpu())
    finally:
        net.train(was)
    return torch.cat(outs) if outs else torch.zeros(0)


def collect(model, data, split="test", limit=None) -> dict:
    """Predictions on a split: {'preds', 'probs', 'y', 'x'} (numpy, original units)."""
    _require_ready(model, data)
    x, y, split = _split_tensors(data, split)
    if limit:
        x, y = x[:limit], y[:limit]
    if model.family == "classic":
        return model.collect(data, x, y)
    out = _raw_outputs(model, data, x)
    if (model.meta or {}).get("objective") == "reconstruct":
        return {"recon": out, "x": x, "y": None}
    if data.task == "classification":
        probs = torch.softmax(out, -1).numpy()
        return {"preds": probs.argmax(-1), "probs": probs, "y": y.numpy(), "x": x}
    preds = data.denormalize_y(out).numpy()
    return {"preds": preds, "probs": None, "y": data.denormalize_y(y.float()).numpy(), "x": x}


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def _class_report(y, preds, classes):
    k = len(classes)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y, preds):
        cm[int(t), int(p)] += 1
    rows = []
    f1s = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        rows.append([str(c), round(float(prec), 3), round(float(rec), 3), round(float(f1), 3), int(cm[i, :].sum())])
    return cm, rows, float(np.mean(f1s)) if f1s else 0.0


def _lm_perplexity(model, data, batches=40):
    from .train import _lm_eval
    import contextlib
    if model.family == "pretrained_lm":
        data = model.adapt_dataset(data)
        forward = model.lm_logits
        net = model.net
    else:
        net = model.net
        forward = net
    n = 4 if STATE.quick else batches
    loss = _lm_eval(forward, net, data, 16, model.device, n, contextlib.nullcontext)
    return loss, math.exp(min(loss, 50))


def evaluate(model, data, split="test", _bid=None) -> dict:
    """The 'test … on …' block: scores + tables + charts."""
    _require_ready(model, data)
    label = getattr(model, "label", model.name)
    if data.task == "language_model":
        loss, ppl = _lm_perplexity(model, data)
        emit("table", title=f"Test results · {label} on {data.name}", columns=["measure", "value"],
             rows=[["validation loss", round(loss, 4)], ["perplexity", round(ppl, 2)],
                   ["meaning", f"the model is about as unsure as picking from {ppl:.1f} tokens"]], block=_bid)
        model.record("test_loss", model.steps_done, loss) if hasattr(model, "record") else None
        return {"loss": loss, "perplexity": ppl}
    res = collect(model, data, split)
    if "recon" in res:
        x = data.prepare_x(res["x"])
        mse = float(F.mse_loss(res["recon"], x))
        emit("table", title=f"Test results · {label} on {data.name}", columns=["measure", "value"],
             rows=[["reconstruction error (MSE)", round(mse, 5)]], block=_bid)
        show_predictions(model, data, 8, _bid=_bid)
        return {"loss": mse}
    if data.task == "classification":
        classes = data.class_names or [str(i) for i in range(int(res["probs"].shape[1] if res["probs"] is not None
                                                                   else res["preds"].max() + 1))]
        y, preds = res["y"], res["preds"]
        acc = float((y == preds).mean()) if len(y) else 0.0
        cm, rows, f1 = _class_report(y, preds, classes)
        emit("table", title=f"Test results · {label} on {data.name} ({len(y):,} examples)",
             columns=["class", "precision", "recall", "F1", "examples"], rows=rows,
             footer=f"accuracy {acc * 100:.2f}% · macro F1 {f1:.3f}", block=_bid)
        emit("confusion", title=f"Confusion matrix · {label}", labels=[str(c) for c in classes],
             matrix=cm.tolist(), block=_bid)
        emit("log", level="success", text=f"{label}: test accuracy {acc * 100:.2f}% on {len(y):,} examples.")
        if hasattr(model, "record"):
            model.record("test_acc", getattr(model, "steps_done", 0), acc)
        return {"accuracy": acc, "f1": f1}
    y, preds = res["y"].reshape(len(res["y"]), -1), res["preds"].reshape(len(res["preds"]), -1)
    err = preds - y
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    var = float(((y - y.mean(0)) ** 2).mean())
    r2 = 1 - float((err ** 2).mean()) / var if var > 1e-12 else 0.0
    emit("table", title=f"Test results · {label} on {data.name} ({len(y):,} examples)",
         columns=["measure", "value"],
         rows=[["mean absolute error", round(mae, 5)], ["root mean squared error", round(rmse, 5)],
               ["R² (1 = perfect, 0 = guessing the average)", round(r2, 4)]], block=_bid)
    k = min(len(y), 600)
    lo, hi = float(min(y[:k, 0].min(), preds[:k, 0].min())), float(max(y[:k, 0].max(), preds[:k, 0].max()))
    emit("scatter", title=f"Predicted vs true · {label}", xlabel="true value", ylabel="predicted",
         points=[[float(a), float(b), 0] for a, b in zip(y[:k, 0], preds[:k, 0])], classes=["examples"],
         line=[[lo, lo], [hi, hi]], block=_bid)
    emit("log", level="success", text=f"{label}: test error ±{mae:.4g} (R² {r2:.3f}).")
    if hasattr(model, "record"):
        model.record("test_mae", getattr(model, "steps_done", 0), mae)
        model.record("test_r2", getattr(model, "steps_done", 0), r2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def score(model, data, metric="accuracy") -> float:
    """A single number for the reporter block ('accuracy of model on data')."""
    metric = metric.lower()
    _require_ready(model, data)
    if data.task == "language_model":
        loss, ppl = _lm_perplexity(model, data, batches=20)
        return round(ppl if metric == "perplexity" else loss, 6)
    res = collect(model, data, "test")
    if "recon" in res:
        return float(F.mse_loss(res["recon"], data.prepare_x(res["x"])))
    if data.task == "classification":
        y, p = res["y"], res["preds"]
        if metric in ("accuracy", "acc"):
            return float((y == p).mean()) if len(y) else 0.0
        if metric == "f1":
            return _class_report(y, p, data.class_names)[2]
        if metric == "loss":
            probs = np.clip(res["probs"], 1e-9, 1)
            return float(-np.log(probs[np.arange(len(y)), y]).mean())
        raise NBError(f"'{metric}' isn't available for classification. Use accuracy, f1 or loss.")
    y, preds = res["y"].reshape(len(res["y"]), -1), res["preds"].reshape(len(res["preds"]), -1)
    err = preds - y
    if metric in ("mae", "error"):
        return float(np.abs(err).mean())
    if metric == "rmse":
        return float(np.sqrt((err ** 2).mean()))
    if metric in ("r2", "accuracy"):
        var = float(((y - y.mean(0)) ** 2).mean())
        return 1 - float((err ** 2).mean()) / var if var > 1e-12 else 0.0
    if metric == "loss":
        return float((err ** 2).mean())
    raise NBError(f"'{metric}' isn't available for number predictions. Use mae, rmse, r2 or loss.")


def show_confusion(model, data, _bid=None):
    res = collect(model, data, "test")
    if data.task != "classification":
        raise NBError("A confusion matrix is for classification (choosing between classes).")
    cm, _, _ = _class_report(res["y"], res["preds"], data.class_names)
    emit("confusion", title=f"Confusion matrix · {getattr(model, 'label', model.name)}",
         labels=[str(c) for c in data.class_names], matrix=cm.tolist(), block=_bid)


# ---------------------------------------------------------------------------
# Galleries
# ---------------------------------------------------------------------------


def _small_png(img: torch.Tensor, data=None) -> str:
    """uint8 (C,H,W) -> base64 PNG, upscaled to ~64 px."""
    if img.dtype != torch.uint8:
        img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
    c, h, w = img.shape
    s = max(1, int(round(64 / max(h, w))))
    img = img.repeat_interleave(s, 1).repeat_interleave(s, 2)
    arr = img.permute(1, 2, 0).numpy()
    if c == 1:
        arr = arr[:, :, 0]
    return png_base64(arr)


def show_predictions(model, data, count=16, _bid=None):
    """Show what the model predicts for some test examples (green = right, red = wrong)."""
    count = max(1, min(int(count), 64))
    label = getattr(model, "label", model.name)
    if data.task == "language_model":
        from .generate import generate
        for _ in range(min(count, 3)):
            generate(model, "", 200, 0.8, _bid=_bid)
        return
    res = collect(model, data, "test", limit=count)
    title = f"Predictions · {label}"
    if "recon" in res:
        recon = data.unprepare_image(res["recon"]) if data.modality == "image" else res["recon"]
        items = []
        for i in range(len(res["x"])):
            if data.modality == "image":
                items.append({"png": _small_png(res["x"][i]), "png2": _small_png(recon[i]),
                              "label": "original → rebuilt"})
        if items:
            emit("predictions", title=f"Reconstructions · {label}", kind="pairs", items=items, block=_bid)
        return
    if data.modality == "image":
        items = []
        for i in range(len(res["x"])):
            p, t = int(res["preds"][i]), int(res["y"][i])
            conf = float(res["probs"][i][p]) if res["probs"] is not None else None
            items.append({"png": _small_png(res["x"][i]), "pred": data.label_name(p), "true": data.label_name(t),
                          "ok": p == t, "conf": conf})
        emit("predictions", title=title, kind="images", items=items, block=_bid)
        return
    if data.modality == "text":
        texts = data.texts_test if data.n_test else data.texts_train
        rows = []
        for i in range(len(res["preds"])):
            p, t = int(res["preds"][i]), int(res["y"][i])
            conf = float(res["probs"][i][p]) if res["probs"] is not None else None
            rows.append([texts[i], data.label_name(t), data.label_name(p), "✓" if p == t else "✗",
                         round(conf, 3) if conf is not None else ""])
        emit("table", title=title, columns=["text", "true", "predicted", "", "confidence"], rows=rows, block=_bid)
        return
    xs = res["x"].numpy().reshape(len(res["x"]), -1)
    if data.x_mean is not None:
        xs = xs * data.x_std + data.x_mean
    feats = data.feature_names or [f"x{i}" for i in range(xs.shape[1])]
    show_f = feats[:6]
    rows = []
    for i in range(len(xs)):
        r = [round(float(v), 3) for v in xs[i][:6]]
        if data.task == "classification":
            p, t = int(res["preds"][i]), int(res["y"][i])
            r += [data.label_name(t), data.label_name(p), "✓" if p == t else "✗"]
        else:
            t = np.ravel(res["y"][i])
            p = np.ravel(res["preds"][i])
            r += [round(float(t[0]), 4), round(float(p[0]), 4), f"±{abs(float(p[0] - t[0])):.3g}"]
        rows.append(r)
    emit("table", title=title, columns=show_f + ["true", "predicted", ""], rows=rows, block=_bid)


def show_examples(data, count=16, _bid=None):
    """Show some examples from a dataset (Data / Results tab)."""
    count = max(1, min(int(count), 64))
    if data.task == "language_model":
        txt = getattr(data, "raw_text", "")
        start = np.random.randint(0, max(1, len(txt) - 600))
        emit("text", title=f"A piece of {data.name}", text=txt[start:start + 600], block=_bid)
        return
    if data.modality == "image":
        items = [{"png": _small_png(data.x_train[i]), "pred": data.label_name(int(data.y_train[i]))}
                 for i in range(min(count, data.n_train))]
        emit("predictions", title=f"Examples from {data.name}", kind="images", items=items, block=_bid)
        return
    p = data.preview()
    if p.get("kind") == "table":
        emit("table", title=f"Examples from {data.name}", columns=p["columns"], rows=p["rows"][:count], block=_bid)
    else:
        plot_data(data, _bid=_bid)


# ---------------------------------------------------------------------------
# Maps & plots
# ---------------------------------------------------------------------------


def _features_2d(data, method="auto", limit=1500):
    """Return (points (n,2), labels or values, axis names, projector or None)."""
    x = data.x_train[:limit]
    y = data.y_train[:limit]
    if data.modality == "image":
        X = data.prepare_x(x).reshape(len(x), -1).numpy()
    elif data.modality in ("tabular", "sequence"):
        X = x.reshape(len(x), -1).numpy().astype(np.float32)
    else:
        raise NBError("This kind of data (text) can't be drawn as a map.")
    if X.shape[1] == 2 and method in ("auto", "pca"):
        return X, y, (data.feature_names or ["x", "y"])[:2], None
    if method == "tsne":
        from sklearn.manifold import TSNE
        P = TSNE(2, init="pca", perplexity=min(30, max(5, len(X) // 10)), random_state=0).fit_transform(X)
        return P, y, ["t-SNE 1", "t-SNE 2"], None
    from sklearn.decomposition import PCA
    pca = PCA(2, random_state=0).fit(X)
    return pca.transform(X), y, ["PCA 1", "PCA 2"], pca


def _label_values(data, y):
    if data.task == "classification":
        return [int(v) for v in y]
    yv = data.denormalize_y(y.float()).reshape(len(y), -1)[:, 0]
    return [round(float(v), 4) for v in yv]


def plot_data(data, _bid=None):
    """Scatter plot of a dataset (uses PCA when there are more than 2 features)."""
    if data.modality == "tabular" and data.input_shape[0] == 1:
        x = data.x_train[:800, 0].numpy()
        if data.x_mean is not None:
            x = x * data.x_std[0] + data.x_mean[0]
        y = data.denormalize_y(data.y_train[:800].float()).reshape(-1).numpy() if data.task == "regression" \
            else data.y_train[:800].numpy()
        emit("scatter", title=f"{data.name}", xlabel=(data.feature_names or ["x"])[0],
             ylabel=(data.target_names or ["y"])[0], points=[[float(a), float(b), 0] for a, b in zip(x, y)],
             classes=["examples"], block=_bid)
        return
    P, y, axes, _ = _features_2d(data)
    pts = [[round(float(a), 4), round(float(b), 4), c] for (a, b), c in zip(P, _label_values(data, y))]
    emit("scatter", title=f"{data.name}" + (" (PCA)" if axes[0].startswith("PCA") else ""), xlabel=axes[0],
         ylabel=axes[1], points=pts, classes=data.class_names, continuous=data.task != "classification", block=_bid)


def show_map(data, method="pca", _bid=None):
    """A 2-D map of the data (PCA or t-SNE), coloured by class."""
    P, y, axes, _ = _features_2d(data, method=method, limit=1500 if method == "tsne" else 3000)
    pts = [[round(float(a), 4), round(float(b), 4), c] for (a, b), c in zip(P, _label_values(data, y))]
    emit("scatter", title=f"Map of {data.name} ({'t-SNE' if method == 'tsne' else 'PCA'})", xlabel=axes[0],
         ylabel=axes[1], points=pts, classes=data.class_names, continuous=data.task != "classification", block=_bid)


def _predict_grid(model, data, X: np.ndarray):
    """Model outputs for raw feature rows X (in the model's input space)."""
    if model.family == "classic":
        return model.grid_outputs(X, data)
    dev = model.device
    with torch.no_grad():
        was = model.net.training
        model.net.eval()
        out = model.net(torch.tensor(X, dtype=torch.float32, device=dev)).float().cpu()
        model.net.train(was)
    if data.task == "classification":
        return torch.softmax(out, -1).numpy()
    return data.denormalize_y(out).numpy().reshape(len(X), -1)[:, 0]


def show_decision_map(model, data, resolution=110, _bid=None):
    """Colour every point of the 2-D plane by what the model predicts there."""
    _require_ready(model, data)
    if data.modality != "tabular":
        raise NBError("Decision maps work with tables of numbers (2-D toy data looks best).")
    label = getattr(model, "label", model.name)
    nfeat = data.input_shape[0]
    if nfeat == 1:
        x = data.x_train[:, 0].numpy()
        grid = np.linspace(x.min() - 0.3, x.max() + 0.3, 200, dtype=np.float32)
        pred = _predict_grid(model, data, grid[:, None])
        xs = grid * (data.x_std[0] if data.x_std is not None else 1) + (data.x_mean[0] if data.x_mean is not None else 0)
        px = x[:500] * (data.x_std[0] if data.x_std is not None else 1) + (data.x_mean[0] if data.x_mean is not None else 0)
        py = data.denormalize_y(data.y_train[:500].float()).reshape(-1).numpy() if data.task == "regression" \
            else data.y_train[:500].numpy()
        emit("scatter", title=f"Model curve · {label}", key=f"Model curve · {label}", xlabel="x", ylabel="y",
             points=[[float(a), float(b), 0] for a, b in zip(px, py)], classes=["examples"],
             line=[[float(a), float(b)] for a, b in zip(xs, np.ravel(pred))], block=_bid)
        return
    X = data.x_train.numpy()
    if nfeat == 2:
        P = X
        pca = None
    else:
        from sklearn.decomposition import PCA
        pca = PCA(2, random_state=0).fit(X)
        P = pca.transform(X)
    lo, hi = P.min(0), P.max(0)
    pad = (hi - lo) * 0.08 + 1e-6
    lo, hi = lo - pad, hi + pad
    r = int(resolution)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], r), np.linspace(hi[1], lo[1], r))
    G = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    Xg = pca.inverse_transform(G).astype(np.float32) if pca is not None else G
    out = _predict_grid(model, data, Xg)
    if data.task == "classification":
        probs = np.asarray(out).reshape(r, r, -1)
        cols = np.stack([_hex(PALETTE[i % len(PALETTE)]) for i in range(probs.shape[-1])])
        img = probs @ cols
        conf = probs.max(-1, keepdims=True)
        img = img * (0.35 + 0.4 * conf) + 255 * (0.65 - 0.4 * conf)
    else:
        v = np.asarray(out).reshape(r, r, 1)
        vmin, vmax = float(v.min()), float(v.max())
        t = (v - vmin) / (vmax - vmin + 1e-9)
        img = (1 - t) * _hex("#4E79A7") + t * _hex("#F28E2B")
        img = img * 0.55 + 255 * 0.45
    scale = 3
    img = np.clip(img, 0, 255).astype(np.uint8).repeat(scale, 0).repeat(scale, 1)
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    W = r * scale
    sel = np.random.default_rng(0).permutation(len(P))[:400]
    ys = data.y_train.numpy()
    for i in sel:
        px = (P[i, 0] - lo[0]) / (hi[0] - lo[0]) * W
        py = (hi[1] - P[i, 1]) / (hi[1] - lo[1]) * W
        if data.task == "classification":
            col = tuple(int(c) for c in _hex(PALETTE[int(ys[i]) % len(PALETTE)]))
        else:
            yv = float(np.ravel(ys[i])[0])
            col = (230, 120, 40) if yv > 0 else (60, 110, 170)
        draw.ellipse([px - 3.5, py - 3.5, px + 3.5, py + 3.5], fill=col, outline=(255, 255, 255))
    cap = "Background colour = what the model predicts there; dots = training examples."
    if pca is not None:
        cap += f" ({nfeat} features squashed onto a 2-D PCA plane.)"
    emit_image(f"Decision map · {label}", png_base64(pil), cap, block=_bid, key=f"Decision map · {label}")


# ---------------------------------------------------------------------------
# Single predictions
# ---------------------------------------------------------------------------


def predict(model, inputs):
    """'prediction of model for …' — returns a class name, a number, or generated text."""
    fam = getattr(model, "family", None)
    if fam == "pretrained_lm" or (fam == "neural" and (model.meta or {}).get("task") == "language_model"):
        from .generate import generated_text
        return generated_text(model, fmt_text(inputs), 60, 0.8)
    if fam == "classic":
        return model.predict_one(inputs)
    if fam != "neural" or model.net is None or not model.meta:
        raise NBError(f"'{getattr(model, 'name', model)}' hasn't been trained yet.")
    ds = meta_dataset(model.meta)
    x = ds.encode_raw(inputs).to(model.device)
    with torch.no_grad():
        was = model.net.training
        model.net.eval()
        out = model.net(x).float().cpu()
        model.net.train(was)
    if ds.task == "classification":
        return ds.label_name(int(out.argmax(-1)[0]))
    vals = ds.denormalize_y(out).reshape(-1).tolist()
    return round(vals[0], 6) if len(vals) == 1 else [round(v, 6) for v in vals]
