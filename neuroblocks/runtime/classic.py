"""Classic machine learning with scikit-learn (trees, forests, k-NN, SVMs, k-means …)."""
from __future__ import annotations

import pickle
import time

import numpy as np
import torch

from .core import STATE, register
from .errors import NBError
from .events import emit

__all__ = ["ClassicModel", "linear_regression", "logistic_regression", "knn", "decision_tree", "random_forest",
           "gradient_boosting", "svm", "naive_bayes", "kmeans", "show_importance", "show_tree"]

_NICE = {"linear": "linear model", "logistic": "logistic regression", "knn": "k-nearest neighbours",
         "tree": "decision tree", "forest": "random forest", "boosting": "gradient boosting",
         "svm": "support vector machine", "bayes": "naive Bayes", "kmeans": "k-means clustering"}


class ClassicModel:
    kind = "model"
    family = "classic"

    def __init__(self, name="model", algo="tree", params=None, _bid=None):
        from .model import _run_label
        self.name = str(name)
        self.label = _run_label(self.name)
        self.algo = algo
        self.params = dict(params or {})
        self.bid = _bid
        self.est = None
        self.fitted = False
        self.meta = None
        self.history: dict = {}
        self.steps_done = 0
        self.train_seconds = 0.0
        self.cluster_map = None
        register(self)

    def describe(self):
        return f"{_NICE.get(self.algo, self.algo)} '{self.name}'"

    def record(self, metric, step, value):
        self.history.setdefault(metric, []).append((step, float(value)))

    def num_params(self) -> int:
        return 0

    # -- estimator --------------------------------------------------------------
    def _make(self, task: str, n_outputs: int):
        p = self.params
        depth = int(p.get("depth") or 0) or None
        cls_task = task == "classification"
        a = self.algo
        if a in ("linear", "logistic"):
            from sklearn.linear_model import LinearRegression, LogisticRegression
            if cls_task:
                return LogisticRegression(C=float(p.get("c", 1.0)), max_iter=2000)
            if a == "logistic":
                emit("log", level="info", text="Logistic regression is for classes; using linear regression to "
                                                "predict numbers.")
            return LinearRegression()
        if a == "knn":
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            k = int(p.get("k", 5))
            return KNeighborsClassifier(k) if cls_task else KNeighborsRegressor(k)
        if a == "tree":
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            return (DecisionTreeClassifier if cls_task else DecisionTreeRegressor)(max_depth=depth, random_state=0)
        if a == "forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return (RandomForestClassifier if cls_task else RandomForestRegressor)(
                n_estimators=int(p.get("trees", 100)), max_depth=depth, random_state=0, n_jobs=-1)
        if a == "boosting":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            if cls_task:
                return GradientBoostingClassifier(n_estimators=int(p.get("trees", 100)),
                                                  learning_rate=float(p.get("lr", 0.1)),
                                                  max_depth=int(p.get("depth") or 3), random_state=0)
            est = GradientBoostingRegressor(n_estimators=int(p.get("trees", 100)), learning_rate=float(p.get("lr", 0.1)),
                                            max_depth=int(p.get("depth") or 3), random_state=0)
            return _multi(est, n_outputs)
        if a == "svm":
            from sklearn.svm import SVC, SVR
            if cls_task:
                return SVC(kernel=p.get("kernel", "rbf"), C=float(p.get("c", 1.0)), random_state=0)
            return _multi(SVR(kernel=p.get("kernel", "rbf"), C=float(p.get("c", 1.0))), n_outputs)
        if a == "bayes":
            if not cls_task:
                raise NBError("Naive Bayes chooses between classes; this data predicts numbers.", block_id=self.bid)
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        if a == "kmeans":
            from sklearn.cluster import KMeans
            return KMeans(int(p.get("k", 3)), n_init=10, random_state=0)
        raise NBError(f"Unknown algorithm '{a}'.")

    # -- features -------------------------------------------------------------------
    def features(self, data, x: torch.Tensor):
        if data.modality == "tokens":
            raise NBError("Classic models can't learn to write text. Use a neural network (e.g. GPT).",
                          block_id=self.bid)
        if data.modality == "text":
            from scipy import sparse
            ids = x.numpy()
            V = data.tokenizer.vocab_size
            rows = np.repeat(np.arange(len(ids)), ids.shape[1])
            cols = ids.ravel()
            m = sparse.csr_matrix((np.ones_like(cols, dtype=np.float32), (rows, cols)), shape=(len(ids), V))
            m[:, data.tokenizer.pad_id or 0] = 0
            m.data = np.log1p(m.data)
            return m
        xp = data.prepare_x(x)
        return xp.reshape(len(xp), -1).numpy().astype(np.float32)

    def fit(self, data, label=None):
        task = data.task
        label = label or self.label
        n = data.n_train
        limit = 3000 if STATE.quick else None
        x, y = data.x_train[:limit] if limit else data.x_train, data.y_train[:limit] if limit else data.y_train
        X = self.features(data, x)
        yv = y.numpy()
        if task == "regression" and yv.ndim > 1 and yv.shape[1] == 1:
            yv = yv[:, 0]
        self.est = self._make(task, data.output_size)
        emit("log", level="info", text=f"Fitting {self.describe()} on {data.name} ({len(yv):,} examples)…")
        emit("progress", id=f"train-{label}", label=f"Training {label}", current=0, total=1, info="fitting…")
        t0 = time.time()
        if self.algo == "kmeans":
            self.est.fit(X)
            if task == "classification":
                clusters = self.est.predict(X)
                self.cluster_map = {}
                for c in np.unique(clusters):
                    labels = yv[clusters == c]
                    self.cluster_map[int(c)] = int(np.bincount(labels).argmax()) if len(labels) else 0
        else:
            self.est.fit(X, yv)
        dt = time.time() - t0
        self.train_seconds += dt
        self.fitted = True
        self.meta = data.meta()
        self.meta["objective"] = "predict"
        self.steps_done += 1
        emit("progress", id=f"train-{label}", label=f"Training {label}", current=1, total=1, info="done")
        if self.algo == "kmeans" and task != "classification":
            sizes = np.bincount(self.est.labels_)
            emit("log", level="success", text=f"Trained {label} in {dt:.2f}s — found {len(sizes)} clusters of "
                                              f"sizes {', '.join(map(str, sizes))}.")
            emit("train_done", model=self.name, label=label, metrics={}, seconds=round(dt, 3))
            return {}
        from .evaluate import collect
        res = {}
        tr = collect(self, data, "train", limit=2000)
        if data.n_test:
            te = collect(self, data, "test")
        else:
            te = tr
        if task == "classification":
            res["train_acc"] = float((tr["preds"] == tr["y"]).mean())
            res["acc"] = float((te["preds"] == te["y"]).mean())
            self.record("train_acc", 1, res["train_acc"])
            self.record("val_acc", 1, res["acc"])
            msg = f"train accuracy {res['train_acc'] * 100:.1f}%, test accuracy {res['acc'] * 100:.1f}%"
            emit("metric", chart="accuracy", series=f"{label} · test", x=1, y=res["acc"], group="train", run=label)
        else:
            err = np.abs(te["preds"].reshape(len(te["preds"]), -1) - te["y"].reshape(len(te["y"]), -1)).mean()
            res["mae"] = float(err)
            self.record("val_mae", 1, res["mae"])
            msg = f"test error ±{res['mae']:.4g}"
        if self.algo == "kmeans":
            sizes = np.bincount(self.est.labels_)
            msg = f"found {len(sizes)} clusters of sizes {', '.join(map(str, sizes))}" + \
                  (f"; matches the real classes {res['acc'] * 100:.1f}% of the time" if task == "classification" else "")
        emit("log", level="success", text=f"Trained {label} in {dt:.2f}s — {msg}.")
        emit("train_done", model=self.name, label=label, metrics=res, seconds=round(dt, 3))
        return res

    # -- predictions -----------------------------------------------------------------
    def _class_probs(self, X):
        if self.algo == "kmeans":
            clusters = self.est.predict(X)
            mapped = np.array([self.cluster_map.get(int(c), 0) if self.cluster_map else int(c) for c in clusters])
            k = len(self.meta.get("class_names") or []) or int(mapped.max() + 1)
            probs = np.zeros((X.shape[0], k))
            probs[np.arange(X.shape[0]), mapped] = 1.0
            return probs
        if not hasattr(self.est, "predict_proba") and hasattr(self.est, "decision_function"):
            # e.g. SVMs: turn decision scores into soft confidences.
            d = np.asarray(self.est.decision_function(X), dtype=np.float64)
            if d.ndim == 1:
                p1 = 1 / (1 + np.exp(-d))
                p = np.stack([1 - p1, p1], 1)
            else:
                e = np.exp(d - d.max(1, keepdims=True))
                p = e / e.sum(1, keepdims=True)
            k = len(self.meta.get("class_names") or []) or p.shape[1]
            if p.shape[1] != k:
                full = np.zeros((p.shape[0], k))
                full[:, self.est.classes_.astype(int)] = p
                p = full
            return p
        if hasattr(self.est, "predict_proba"):
            p = self.est.predict_proba(X)
            k = len(self.meta.get("class_names") or []) or p.shape[1]
            if p.shape[1] != k:  # some classes missing from training data
                full = np.zeros((p.shape[0], k))
                full[:, self.est.classes_.astype(int)] = p
                p = full
            return p
        pred = self.est.predict(X).astype(int)
        k = len(self.meta.get("class_names") or []) or int(pred.max() + 1)
        probs = np.zeros((len(pred), k))
        probs[np.arange(len(pred)), pred] = 1.0
        return probs

    def collect(self, data, x, y):
        X = self.features(data, x)
        if data.task == "classification":
            probs = self._class_probs(X)
            return {"preds": probs.argmax(-1), "probs": probs, "y": y.numpy(), "x": x}
        pred = np.asarray(self.est.predict(X), dtype=np.float32).reshape(len(y), -1)
        preds = data.denormalize_y(torch.from_numpy(pred)).numpy()
        return {"preds": preds, "probs": None, "y": data.denormalize_y(y.float()).numpy(), "x": x}

    def grid_outputs(self, X, data):
        if data.task == "classification":
            return self._class_probs(X)
        pred = np.asarray(self.est.predict(X), dtype=np.float32).reshape(len(X), -1)
        return data.denormalize_y(torch.from_numpy(pred)).numpy()[:, 0]

    def predict_one(self, inputs):
        if not self.fitted:
            raise NBError(f"'{self.name}' hasn't been trained yet.")
        from .evaluate import meta_dataset
        ds = meta_dataset(self.meta)
        x = ds.encode_raw(inputs)
        if ds.modality == "image":
            X = x.reshape(1, -1).numpy()
        elif ds.modality == "text":
            X = self.features(ds, x)
        else:
            X = x.reshape(1, -1).numpy()
        if ds.task == "classification":
            return ds.label_name(int(self._class_probs(X).argmax(-1)[0]))
        v = np.asarray(self.est.predict(X), dtype=np.float32).reshape(1, -1)
        vals = ds.denormalize_y(torch.from_numpy(v)).reshape(-1).tolist()
        return round(vals[0], 6) if len(vals) == 1 else vals

    # -- save / load -----------------------------------------------------------------
    def to_payload(self):
        return {"format": "neuroblocks-model", "family": "classic", "name": self.name, "algo": self.algo,
                "params": self.params, "meta": self.meta, "estimator": pickle.dumps(self.est) if self.est else None,
                "cluster_map": self.cluster_map, "history": self.history}

    @classmethod
    def from_payload(cls, payload, name=None):
        m = cls(name or payload.get("name", "model"), payload["algo"], payload.get("params"))
        m.meta = payload.get("meta")
        m.est = pickle.loads(payload["estimator"]) if payload.get("estimator") else None
        m.fitted = m.est is not None
        m.cluster_map = payload.get("cluster_map")
        m.history = payload.get("history") or {}
        return m


def _multi(est, n_outputs):
    if n_outputs > 1:
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(est)
    return est


def linear_regression(name="model", _bid=None):
    return ClassicModel(name, "linear", {}, _bid)


def logistic_regression(name="model", c=1.0, _bid=None):
    return ClassicModel(name, "logistic", {"c": c}, _bid)


def knn(name="model", k=5, _bid=None):
    return ClassicModel(name, "knn", {"k": k}, _bid)


def decision_tree(name="model", depth=0, _bid=None):
    return ClassicModel(name, "tree", {"depth": depth}, _bid)


def random_forest(name="model", trees=100, depth=0, _bid=None):
    return ClassicModel(name, "forest", {"trees": trees, "depth": depth}, _bid)


def gradient_boosting(name="model", trees=100, lr=0.1, depth=3, _bid=None):
    return ClassicModel(name, "boosting", {"trees": trees, "lr": lr, "depth": depth}, _bid)


def svm(name="model", kernel="rbf", c=1.0, _bid=None):
    return ClassicModel(name, "svm", {"kernel": kernel, "c": c}, _bid)


def naive_bayes(name="model", _bid=None):
    return ClassicModel(name, "bayes", {}, _bid)


def kmeans(name="model", k=3, _bid=None):
    return ClassicModel(name, "kmeans", {"k": k}, _bid)


def show_importance(model, _bid=None):
    """Bar chart of which input features mattered most."""
    if getattr(model, "family", None) != "classic" or not model.fitted:
        raise NBError("Feature importance needs a trained classic model (tree, forest, boosting or linear).",
                      block_id=_bid)
    est = model.est
    imp = getattr(est, "feature_importances_", None)
    if imp is None and hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        imp = np.abs(coef).mean(0) if coef.ndim > 1 else np.abs(coef)
    if imp is None:
        raise NBError(f"A {_NICE.get(model.algo)} doesn't report feature importance.", block_id=_bid)
    names = (model.meta or {}).get("feature_names") or [f"x{i}" for i in range(len(imp))]
    order = np.argsort(imp)[::-1][:20]
    emit("plot", kind="bar", title=f"Feature importance · {model.label}", values=[float(imp[i]) for i in order],
         labels=[str(names[i]) if i < len(names) else f"x{i}" for i in order], block=_bid)


def show_tree(model, _bid=None):
    """Print a decision tree's questions as text."""
    from sklearn.tree import export_text
    if getattr(model, "algo", None) != "tree" or not model.fitted:
        raise NBError("This block shows a trained decision tree.", block_id=_bid)
    import re
    meta = model.meta or {}
    names = meta.get("feature_names")
    names = list(names) if names else [f"x{i}" for i in range(model.est.n_features_in_)]
    txt = export_text(model.est, feature_names=names, max_depth=6, decimals=4)
    # Thresholds are in standardised units; show them in the data's real units instead.
    if meta.get("x_mean") is not None and meta.get("x_std") is not None and meta.get("modality") == "tabular":
        mean, std = meta["x_mean"], meta["x_std"]
        idx = {n: i for i, n in enumerate(names)}
        pat = re.compile(r"\|--- (.+?) (<=|>)\s+(-?[0-9.]+)")

        def fix(m):
            i = idx.get(m.group(1))
            if i is None:
                return m.group(0)
            v = float(m.group(3)) * float(std[i]) + float(mean[i])
            return f"|--- {m.group(1)} {m.group(2)} {v:.3g}"
        txt = pat.sub(fix, txt)
    classes = meta.get("class_names")
    if classes:
        for i, c in enumerate(classes):
            txt = txt.replace(f"class: {i}\n", f"class: {c}\n")
    emit("text", title=f"Decision tree · {model.label}", text=txt, block=_bid)
