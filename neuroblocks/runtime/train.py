"""Training: the 'train … on …' block."""
from __future__ import annotations

import contextlib
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import STATE, device, quick_cap
from .errors import NBError
from .events import emit

__all__ = ["train", "metric_of", "MetricStream", "fmt_time", "make_optimizer", "make_scheduler"]


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class MetricStream:
    """Throttled metric/progress emission so the editor isn't flooded."""

    def __init__(self, label: str, interval: float = 0.25):
        self.label = label
        self.interval = interval
        self.acc: dict[tuple, list] = {}
        self.last_emit: dict[tuple, float] = {}
        self.last_progress = 0.0

    def add(self, chart: str, series: str, x, y, *, force=False, group="train"):
        key = (chart, series)
        self.acc.setdefault(key, []).append((x, float(y)))
        now = time.time()
        if force or now - self.last_emit.get(key, 0) >= self.interval:
            self.flush(key, group)

    def flush(self, key=None, group="train"):
        keys = [key] if key is not None else list(self.acc)
        for k in keys:
            vals = self.acc.get(k)
            if not vals:
                continue
            x = vals[-1][0]
            ys = [v for _, v in vals if math.isfinite(v)]
            y = sum(ys) / len(ys) if ys else float("nan")
            emit("metric", chart=k[0], series=f"{self.label} · {k[1]}" if k[1] else self.label, x=x, y=y,
                 group=group, run=self.label)
            self.acc[k] = []
            self.last_emit[k] = time.time()

    def progress(self, current, total, info="", force=False, label=None):
        now = time.time()
        if force or now - self.last_progress >= 0.3:
            self.last_progress = now
            emit("progress", id=f"train-{self.label}", label=label or f"Training {self.label}",
                 current=current, total=total, info=info)


# ---------------------------------------------------------------------------
# Optimisers & schedules
# ---------------------------------------------------------------------------

def make_optimizer(params, name: str, lr: float | None, weight_decay: float | None, transformer=False):
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr or (3e-4 if transformer else 1e-3), weight_decay=weight_decay or 0.0)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr or (3e-4 if transformer else 1e-3),
                                 weight_decay=0.01 if weight_decay is None else weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr or 0.05, momentum=0.9, weight_decay=weight_decay or 0.0)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr or 1e-3, weight_decay=weight_decay or 0.0)
    raise NBError(f"Unknown optimizer '{name}'.", hint="Use adam, adamw, sgd or rmsprop.")


def make_scheduler(opt, schedule: str | None, warmup: int, total_steps: int):
    schedule = (schedule or "constant").lower()
    warmup = int(warmup or 0)
    total = max(1, int(total_steps))

    def factor(step):
        w = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
        p = min(1.0, step / total)
        if schedule == "cosine":
            d = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
        elif schedule == "linear":
            d = max(0.02, 1.0 - p)
        elif schedule == "step":
            d = 0.5 ** math.floor(p * 3.999)
        else:
            d = 1.0
        return w * d

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def _amp(dev, precision: str):
    """Mixed precision on GPUs (faster, less memory). Returns (context factory, scaler)."""
    precision = (precision or "auto").lower()
    if dev.type == "cuda" and precision in ("auto", "on", "bf16", "fp16"):
        if precision != "fp16" and torch.cuda.is_bf16_supported():
            return (lambda: torch.autocast("cuda", dtype=torch.bfloat16)), None
        scaler = torch.amp.GradScaler("cuda")
        return (lambda: torch.autocast("cuda", dtype=torch.float16)), scaler
    return contextlib.nullcontext, None


def _has_transformer(net) -> bool:
    from .layers import _TransformerBlock
    return any(isinstance(m, _TransformerBlock) for m in net.modules())


class _Callbacks:
    def __init__(self, every):
        self.items = []
        for item in every or []:
            n, unit, fn = item
            self.items.append((max(1, int(n)), str(unit), fn))

    def fire(self, unit: str, count: int, net=None):
        for n, u, fn in self.items:
            if u == unit and count % n == 0:
                if net is not None:
                    net.eval()
                marker = STATE.current_block
                try:
                    with torch.enable_grad():
                        fn()
                finally:
                    STATE.current_block = marker  # the 'train' block glows again
                    if net is not None:
                        net.train()


def _check_skip() -> bool:
    if STATE.skip.is_set():
        STATE.skip.clear()
        emit("log", level="info", text="Training finished early (you pressed Skip).")
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(model, data, *, epochs=None, steps=None, batch_size=None, optimizer="adam", lr=None, schedule=None,
          warmup=0, weight_decay=None, loss="auto", clip=None, patience=None, eval_every=None, precision="auto",
          compile=False, label=None, objective="auto", save_best=None, every=None, _bid=None, **unused):
    """Train ``model`` on ``data``. Works for neural networks, language models and classic ML models."""
    from .data import Dataset
    for k in unused:
        if not k.startswith("_"):
            emit("log", level="warn", text=f"The '{k}' setting doesn't apply to training, so it was ignored.")
    if model is None or not hasattr(model, "family"):
        raise NBError("The first slot of 'train' must be a model.", block_id=_bid)
    if not isinstance(data, Dataset):
        if getattr(data, "kind", None) == "world":
            raise NBError("To learn inside a simulation, use the 'train … to play in …' block (Simulations).",
                          block_id=_bid)
        raise NBError("The second slot of 'train' must be a dataset.", block_id=_bid)
    if model.family == "classic":
        return model.fit(data, label=label)
    if model.family == "generator":
        return model.fit(data, epochs=epochs, steps=steps, batch_size=batch_size, lr=lr, label=label, every=every)
    if model.family == "agent":
        raise NBError("This model learned in a simulation; train it with 'train … to play in …'.", block_id=_bid)
    cfg = dict(epochs=epochs, steps=steps, batch_size=batch_size, optimizer=optimizer, lr=lr, schedule=schedule,
               warmup=warmup, weight_decay=weight_decay, loss=loss, clip=clip, patience=patience,
               eval_every=eval_every, precision=precision, compile=compile, label=label, objective=objective,
               save_best=save_best, callbacks=_Callbacks(every), bid=_bid)
    if data.task == "language_model":
        return _train_lm(model, data, cfg)
    if model.family == "pretrained_lm":
        raise NBError("Pretrained language models train on text datasets (load text first).", block_id=_bid)
    return _train_supervised(model, data, cfg)


def metric_of(model, which: str = "val_loss"):
    """Read a training result: last/best loss or accuracy, training time …"""
    h = getattr(model, "history", {}) or {}
    which = which.lower()
    if which == "train_seconds":
        return round(getattr(model, "train_seconds", 0.0), 2)
    if which == "steps":
        return getattr(model, "steps_done", 0)
    if which == "params":
        n = getattr(model, "num_params", None)
        return n() if callable(n) else 0
    best = which.startswith("best_")
    key = which[5:] if best else which
    vals = [v for _, v in h.get(key, [])]
    if not vals:
        return 0.0
    if best:
        return max(vals) if ("acc" in key or "r2" in key) else min(vals)
    return vals[-1]


# ---------------------------------------------------------------------------
# Supervised learning (classification / regression / autoencoders)
# ---------------------------------------------------------------------------


def _loss_fn(name: str, task: str, objective: str, n_out: int):
    name = (name or "auto").lower()
    if objective in ("reconstruct", "denoise"):
        name = "mse" if name == "auto" else name
    if name == "auto":
        name = "cross_entropy" if task == "classification" else "mse"
    if name == "cross_entropy":
        if task != "classification":
            raise NBError("Cross-entropy loss is for choosing classes; this data predicts numbers.",
                          hint="Use mean squared error (or 'auto').")
        return lambda out, y: F.cross_entropy(out, y), name
    base = {"mse": F.mse_loss, "mae": F.l1_loss, "huber": F.smooth_l1_loss}.get(name)
    if name == "bce":
        if task == "classification":
            return lambda out, y: F.binary_cross_entropy_with_logits(out, F.one_hot(y, n_out).float()), name
        return lambda out, y: F.binary_cross_entropy_with_logits(out, y.clamp(0, 1)), name
    if base is None:
        raise NBError(f"Unknown loss '{name}'.")
    if task == "classification" and objective not in ("reconstruct", "denoise"):
        return lambda out, y: base(out.softmax(-1), F.one_hot(y, n_out).float()), name
    return lambda out, y: base(out, y), name


@torch.no_grad()
def evaluate_split(model, data, loss_fn=None, dev=None, objective="predict", split="test", limit=None):
    """Loss + accuracy / MAE / R² on a split. Returns a dict (empty if the split is empty)."""
    net = model.net
    dev = dev or model.device
    if split == "test" and data.n_test == 0:
        return {}
    was = net.training
    net.eval()
    total, n, correct = 0.0, 0, 0
    abs_err = 0.0
    sq_err = 0.0
    ys_sum = 0.0
    ys_sq = 0.0
    try:
        for xb, yb in data.batches(split, 512, shuffle=False, device=dev, limit=limit):
            out = net(xb)
            if objective in ("reconstruct", "denoise"):
                target = xb
            else:
                target = yb
            if loss_fn is not None:
                total += float(loss_fn(out, target)) * len(xb)
            n += len(xb)
            if objective in ("reconstruct", "denoise"):
                continue
            if data.task == "classification":
                correct += int((out.argmax(-1) == yb).sum())
            else:
                pred = data.denormalize_y(out.float())
                true = data.denormalize_y(yb.float())
                abs_err += float((pred - true).abs().mean(dim=tuple(range(1, pred.ndim))).sum()) if pred.ndim > 1 \
                    else float((pred - true).abs().sum())
                sq_err += float(((pred - true) ** 2).sum())
                ys_sum += float(true.sum())
                ys_sq += float((true ** 2).sum())
    finally:
        net.train(was)
    if n == 0:
        return {}
    res = {}
    if loss_fn is not None:
        res["loss"] = total / n
    if objective in ("reconstruct", "denoise"):
        return res
    if data.task == "classification":
        res["acc"] = correct / n
    else:
        k = data.output_size
        res["mae"] = abs_err / n
        mean = ys_sum / (n * k)
        var = ys_sq / (n * k) - mean ** 2
        res["r2"] = 1 - (sq_err / (n * k)) / var if var > 1e-12 else 0.0
    return res


def _train_supervised(model, data, cfg):
    dev = device()
    objective = cfg["objective"] if cfg["objective"] in ("reconstruct", "denoise") else "predict"
    net = model.build_for(data, objective)
    net.to(dev)
    net.train()
    label = cfg["label"] or model.label
    bs = int(cfg["batch_size"] or 32)
    n_batches = data.num_batches(bs)
    if cfg["steps"]:
        total_steps = int(cfg["steps"])
        epochs = math.ceil(total_steps / n_batches)
    else:
        epochs = int(cfg["epochs"] or 10)
        total_steps = epochs * n_batches
    batch_limit = None
    if STATE.quick:
        epochs = min(epochs, 2)
        batch_limit = min(n_batches, 5) * bs
        total_steps = min(total_steps, epochs * min(n_batches, 5))
    loss_fn, loss_name = _loss_fn(cfg["loss"], data.task, objective, data.output_size)
    opt = make_optimizer(net.parameters(), cfg["optimizer"], cfg["lr"], cfg["weight_decay"], _has_transformer(net))
    sched = make_scheduler(opt, cfg["schedule"], cfg["warmup"], total_steps)
    amp, scaler = _amp(dev, cfg["precision"])
    fwd = net
    if cfg["compile"] and dev.type == "cuda":
        try:
            fwd = torch.compile(net)
        except Exception as e:  # noqa: BLE001
            emit("log", level="warn", text=f"torch.compile isn't available here ({e}); training normally.")
    ms = MetricStream(label)
    cb = cfg["callbacks"]
    patience = int(cfg["patience"]) if cfg["patience"] else None
    best, bad_epochs = None, 0
    step = 0
    t0 = time.time()
    is_cls = data.task == "classification" and objective == "predict"
    what = "epochs" if not cfg["steps"] else "steps"
    emit("log", level="info", text=f"Training {label} on {data.name} for "
                                   f"{epochs if what == 'epochs' else total_steps} {what} "
                                   f"({data.n_train:,} examples, batch size {bs}, {loss_name.replace('_', ' ')} loss)…")
    stopped = False
    for epoch in range(epochs):
        correct = seen = 0
        for xb, yb in data.batches("train", bs, shuffle=True, device=dev, augment=True, limit=batch_limit):
            inp = xb
            target = yb
            if objective in ("reconstruct", "denoise"):
                target = xb
                if objective == "denoise":
                    inp = xb + 0.5 * torch.randn_like(xb)
            with amp():
                out = fwd(inp)
                loss = loss_fn(out.float(), target)
            if not torch.isfinite(loss):
                raise NBError("The loss became infinite/NaN — training blew up.",
                              hint="Lower the learning rate, or add 'clip gradients'.", block_id=cfg["bid"])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg["clip"]:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(net.parameters(), float(cfg["clip"]))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg["clip"]:
                    nn.utils.clip_grad_norm_(net.parameters(), float(cfg["clip"]))
                opt.step()
            sched.step()
            step += 1
            model.steps_done += 1
            lv = loss.item()
            model.record("train_loss", model.steps_done, lv)
            ms.add("loss", "train", model.steps_done, lv)
            if is_cls:
                correct += int((out.argmax(-1) == yb).sum())
                seen += len(yb)
            elapsed = time.time() - t0
            eta = elapsed / step * (total_steps - step) if step else 0
            ms.progress(step, total_steps, f"epoch {epoch + 1}/{epochs} · loss {lv:.4f} · ETA {fmt_time(eta)}")
            cb.fire("steps", step, net)
            if step >= total_steps or _check_skip():
                stopped = step < total_steps
                break
        ms.flush()
        model.epochs_done += 1
        # End-of-epoch evaluation.
        if is_cls and seen:
            model.record("train_acc", model.steps_done, correct / seen)
            ms.add("accuracy", "train", model.steps_done, correct / seen, force=True)
        ev = evaluate_split(model, data, loss_fn, dev, objective, limit=None if not STATE.quick else 256)
        msg = f"epoch {epoch + 1}/{epochs}: train loss {lv:.4f}"
        if ev:
            model.record("val_loss", model.steps_done, ev["loss"])
            ms.add("loss", "test", model.steps_done, ev["loss"], force=True)
            msg += f", test loss {ev['loss']:.4f}"
            if "acc" in ev:
                model.record("val_acc", model.steps_done, ev["acc"])
                ms.add("accuracy", "test", model.steps_done, ev["acc"], force=True)
                msg += f", test accuracy {ev['acc'] * 100:.1f}%"
            if "mae" in ev:
                model.record("val_mae", model.steps_done, ev["mae"])
                model.record("val_r2", model.steps_done, ev["r2"])
                ms.add("error", "test MAE", model.steps_done, ev["mae"], force=True)
                msg += f", test error ±{ev['mae']:.4g}"
        emit("log", level="debug", text=msg)
        score = ev.get("loss") if ev else lv
        improved = best is None or score < best - 1e-6
        if improved:
            best, bad_epochs = score, 0
            if cfg["save_best"]:
                from .io import save
                save(model, cfg["save_best"], quiet=True)
        else:
            bad_epochs += 1
        cb.fire("epochs", epoch + 1, net)
        if stopped or step >= total_steps:
            break
        if patience and bad_epochs >= patience:
            emit("log", level="info", text=f"Stopping early: no improvement for {patience} epochs.")
            break
        if _check_skip():
            break
    net.eval()
    dt = time.time() - t0
    model.train_seconds += dt
    ms.progress(total_steps, total_steps, "done", force=True)
    final = evaluate_split(model, data, loss_fn, dev, objective)
    summary = f"Finished training {label} in {fmt_time(dt)}"
    if "acc" in final:
        summary += f" — test accuracy {final['acc'] * 100:.1f}%"
    elif "mae" in final:
        summary += f" — test error ±{final['mae']:.4g} (R² {final['r2']:.3f})"
    elif "loss" in final:
        summary += f" — test loss {final['loss']:.4f}"
    emit("log", level="success", text=summary)
    emit("train_done", model=model.name, label=label, metrics=final, seconds=round(dt, 2))
    return final


# ---------------------------------------------------------------------------
# Language models
# ---------------------------------------------------------------------------


def _train_lm(model, data, cfg):
    dev = device()
    pretrained = model.family == "pretrained_lm"
    if pretrained:
        data = model.adapt_dataset(data)
        net = model.prepare(dev)
    else:
        net = model.build_for(data)
    net.to(dev)
    net.train()
    label = cfg["label"] or model.label
    steps = int(cfg["steps"] or 0)
    if not steps:
        if cfg["epochs"]:
            per_epoch = max(1, data.n_train // (int(cfg["batch_size"] or 32) * data.context))
            steps = int(cfg["epochs"]) * per_epoch
            emit("log", level="info", text=f"{cfg['epochs']} epoch(s) of this text ≈ {steps:,} steps.")
        else:
            steps = 1000
    steps = quick_cap(steps, 20)
    bs = int(cfg["batch_size"] or (8 if pretrained else 32))
    eval_every = int(cfg["eval_every"] or max(25, steps // 20))
    eval_iters = 2 if STATE.quick else 20
    lr = cfg["lr"] or (5e-5 if pretrained else 1e-3)
    opt = make_optimizer(net.parameters(), cfg["optimizer"] if cfg["optimizer"] != "adam" else "adamw", lr,
                         cfg["weight_decay"] if cfg["weight_decay"] is not None else (0.0 if pretrained else 0.1))
    schedule = cfg["schedule"] or "cosine"
    warmup = int(cfg["warmup"] or min(100, steps // 10))
    sched = make_scheduler(opt, schedule, warmup, steps)
    amp, scaler = _amp(dev, cfg["precision"])
    fwd = net
    if cfg["compile"] and dev.type == "cuda":
        try:
            fwd = torch.compile(net)
        except Exception as e:  # noqa: BLE001
            emit("log", level="warn", text=f"torch.compile isn't available here ({e}).")
    forward = model.lm_logits if pretrained else (lambda x: fwd(x))
    clip = float(cfg["clip"]) if cfg["clip"] else 1.0
    ms = MetricStream(label)
    cb = cfg["callbacks"]
    V = data.tokenizer.vocab_size
    patience = int(cfg["patience"]) if cfg["patience"] else None
    best, bad = None, 0
    emit("log", level="info", text=f"Training language model {label} on {data.name} for {steps:,} steps "
                                   f"(batch {bs} × context {data.context} tokens, vocabulary {V:,})…")
    t0 = time.time()
    tokens_seen = 0
    lv = float("nan")
    for step in range(1, steps + 1):
        x, y = data.lm_batch("train", bs, dev)
        with amp():
            logits = forward(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        if not torch.isfinite(loss):
            raise NBError("The loss became infinite/NaN — training blew up.",
                          hint="Lower the learning rate.", block_id=cfg["bid"])
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
        sched.step()
        model.steps_done += 1
        tokens_seen += x.numel()
        lv = loss.item()
        model.record("train_loss", model.steps_done, lv)
        ms.add("loss", "train", model.steps_done, lv)
        elapsed = time.time() - t0
        eta = elapsed / step * (steps - step)
        ms.progress(step, steps, f"loss {lv:.3f} · {tokens_seen / max(elapsed, 1e-6):,.0f} tokens/s · "
                                 f"ETA {fmt_time(eta)}")
        if step % eval_every == 0 or step == steps:
            vl = _lm_eval(forward, net, data, bs, dev, eval_iters, amp)
            model.record("val_loss", model.steps_done, vl)
            ms.add("loss", "validation", model.steps_done, vl, force=True)
            ms.flush()
            emit("log", level="debug", text=f"step {step:,}/{steps:,}: train loss {lv:.4f}, validation loss {vl:.4f}")
            if best is None or vl < best - 1e-4:
                best, bad = vl, 0
                if cfg["save_best"]:
                    from .io import save
                    save(model, cfg["save_best"], quiet=True)
            else:
                bad += 1
                if patience and bad >= patience:
                    emit("log", level="info", text=f"Stopping early: validation loss stopped improving.")
                    break
        cb.fire("steps", step, net)
        if _check_skip():
            break
    ms.flush()
    net.eval()
    dt = time.time() - t0
    model.train_seconds += dt
    ms.progress(steps, steps, "done", force=True)
    vals = [v for _, v in model.history.get("val_loss", [])]
    final = {"loss": vals[-1] if vals else lv}
    final["perplexity"] = math.exp(min(final["loss"], 50))
    emit("log", level="success", text=f"Finished training {label} in {fmt_time(dt)} — validation loss "
                                      f"{final['loss']:.3f} (perplexity {final['perplexity']:.1f})")
    emit("train_done", model=model.name, label=label, metrics=final, seconds=round(dt, 2))
    return final


@torch.no_grad()
def _lm_eval(forward, net, data, bs, dev, iters, amp):
    net.eval()
    losses = []
    try:
        for _ in range(iters):
            x, y = data.lm_batch("test", bs, dev)
            with amp():
                logits = forward(x)
            losses.append(float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))))
    finally:
        net.train()
    return sum(losses) / max(1, len(losses))
