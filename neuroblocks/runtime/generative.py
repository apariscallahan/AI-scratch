"""Picture generators: models that learn to draw new images like the ones in a dataset.

Two kinds (both can be asked for a particular class, e.g. "draw a 7"):

* ``vae`` — a variational autoencoder: stable and quick to train, pictures a bit blurry.
* ``gan`` — a generative adversarial network: a generator and a critic play a game;
  sharper pictures, but training is less predictable.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import STATE, device, register
from .errors import NBError
from .events import emit

__all__ = ["ImageGenerator", "generator", "show_generated"]


def _mlp(sizes, act=nn.LeakyReLU, final=None, dropout=0.0):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act(0.2) if act is nn.LeakyReLU else act())
            if dropout:
                layers.append(nn.Dropout(dropout))
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)


class _VAE(nn.Module):
    def __init__(self, n_pixels, n_classes, latent, hidden):
        super().__init__()
        self.n_classes = n_classes
        self.enc = _mlp([n_pixels + n_classes, hidden, hidden], final=nn.LeakyReLU(0.2))
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = _mlp([latent + n_classes, hidden, hidden, n_pixels])

    def forward(self, x, y1h):
        h = self.enc(torch.cat([x, y1h], 1))
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(torch.cat([z, y1h], 1)), mu, logvar

    def decode(self, z, y1h):
        return torch.sigmoid(self.dec(torch.cat([z, y1h], 1)))


def _conv_plan(h, w):
    """Number of 2× up/down-sampling stages for a DCGAN on h×w images (None = use MLPs)."""
    if h != w or h < 16:
        return None
    for stages in (2, 3, 4):
        start = h // (2 ** stages)
        if start * 2 ** stages == h and 4 <= start <= 8:
            return stages, start
    return None


class _ConvGen(nn.Module):
    """latent (+ class one-hot) → image in [-1, 1] with transposed convolutions."""

    def __init__(self, zdim, c, start, stages, width=64):
        super().__init__()
        ch = width * 2 ** (stages - 1)
        self.start, self.ch = start, ch
        self.fc = nn.Sequential(nn.Linear(zdim, ch * start * start), nn.BatchNorm1d(ch * start * start), nn.ReLU())
        layers = []
        for i in range(stages):
            out = c if i == stages - 1 else ch // 2
            layers.append(nn.ConvTranspose2d(ch, out, 4, 2, 1))
            if i < stages - 1:
                layers += [nn.BatchNorm2d(out), nn.ReLU()]
            ch = out
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(self.fc(z).view(len(z), self.ch, self.start, self.start)).flatten(1)


class _ConvCritic(nn.Module):
    """image (+ class as extra planes) → real/fake score."""

    def __init__(self, c, k, h, stages, width=64):
        super().__init__()
        self.c, self.k, self.h = c, k, h
        layers, ch = [], c + k
        for i in range(stages):
            out = width * 2 ** i
            layers += [nn.Conv2d(ch, out, 4, 2, 1)] + ([nn.BatchNorm2d(out)] if i else []) + [nn.LeakyReLU(0.2)]
            ch = out
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Dropout(0.2), nn.LazyLinear(1))

    def forward(self, xy):
        x, y = xy[:, : self.c * self.h * self.h], xy[:, self.c * self.h * self.h:]
        x = x.view(len(x), self.c, self.h, self.h)
        if self.k:
            x = torch.cat([x, y[:, :, None, None].expand(-1, -1, self.h, self.h)], 1)
        return self.net(x)


class ImageGenerator:
    """A model that learns to make new pictures."""

    kind = "model"
    family = "generator"

    def __init__(self, name="generator", kind="vae", latent=16, _bid=None):
        from .model import _run_label
        self.name = str(name)
        self.label = _run_label(self.name)
        self.gen_kind = kind if kind in ("vae", "gan") else "vae"
        self.latent = int(latent)
        self.bid = _bid
        self.meta = None
        self.nets: dict[str, nn.Module] = {}
        self.history: dict = {}
        self.steps_done = 0
        self.train_seconds = 0.0
        register(self)

    def describe(self):
        return f"picture generator '{self.name}' ({self.gen_kind.upper()})"

    def record(self, metric, step, value):
        self.history.setdefault(metric, []).append((step, float(value)))

    def num_params(self):
        return sum(p.numel() for n in self.nets.values() for p in n.parameters())

    @property
    def device(self):
        for n in self.nets.values():
            for p in n.parameters():
                return p.device
        return torch.device("cpu")

    # -- building ----------------------------------------------------------------
    def _build(self, shape, n_classes):
        c, h, w = shape
        n_pixels = c * h * w
        hidden = 256 if n_pixels <= 256 else 512
        k = n_classes or 0
        plan = _conv_plan(h, w)
        if self.gen_kind == "vae":
            self.nets = {"vae": _VAE(n_pixels, k, self.latent, hidden)}
        elif plan is not None:
            stages, start = plan
            critic = _ConvCritic(c, k, h, stages)
            with torch.no_grad():  # size the lazy layer
                critic.eval()
                critic(torch.zeros(2, n_pixels + k))
                critic.train()
            self.nets = {"gen": _ConvGen(self.latent + k, c, start, stages), "critic": critic}
        else:
            self.nets = {
                "gen": _mlp([self.latent + k, hidden, hidden * 2 if n_pixels > 256 else hidden, n_pixels],
                            act=nn.ReLU, final=nn.Tanh()),
                "critic": _mlp([n_pixels + k, hidden * 2 if n_pixels > 256 else hidden, hidden, 1], dropout=0.3),
            }
        rows = [{"name": name, "detail": type(net).__name__.strip("_"), "shape": None,
                 "params": sum(p.numel() for p in net.parameters()), "depth": 0} for name, net in self.nets.items()]
        emit("model", name=self.name, info={"summary": f"{self.label}: {self.gen_kind.upper()} picture generator, "
                                                       f"{self.num_params():,} parameters",
                                            "rows": rows, "total_params": self.num_params(), "label": self.label,
                                            "family": self.family, "task": "generation", "input_shape": list(shape)})

    def _onehot(self, y, n):
        if not n:
            return torch.zeros((len(y), 0), device=y.device)
        return F.one_hot(y.long(), n).float()

    # -- training ----------------------------------------------------------------
    def fit(self, data, epochs=None, steps=None, batch_size=None, lr=None, label=None, every=None, **_):
        if data.modality != "image":
            raise NBError("Picture generators learn from image datasets (digits, MNIST, your own folder…).",
                          block_id=self.bid)
        from .train import MetricStream, _Callbacks, _check_skip, fmt_time
        dev = device()
        n_classes = data.num_classes if data.task == "classification" else 0
        shape = tuple(data.input_shape)
        sig = (shape, n_classes, self.gen_kind)
        if not self.nets or (self.meta or {}).get("sig") != list(map(str, sig)):
            self._build(shape, n_classes)
            self.meta = {"task": "generation", "modality": "image", "input_shape": list(shape),
                         "class_names": data.class_names, "sig": list(map(str, sig)), "kind": self.gen_kind}
        for n in self.nets.values():
            n.to(dev).train()
        bs = int(batch_size or 64)
        epochs = int(epochs or (30 if self.gen_kind == "vae" else 60))
        if steps:
            epochs = max(1, math.ceil(int(steps) / data.num_batches(bs)))
        limit = None
        if STATE.quick:
            epochs, limit = min(epochs, 2), 5 * bs
        label = label or self.label
        ms = MetricStream(label)
        cb = _Callbacks(every)
        total = epochs * min(data.num_batches(bs), (limit // bs) if limit else 10 ** 9)
        emit("log", level="info", text=f"Training picture generator {label} ({self.gen_kind.upper()}) on {data.name} "
                                       f"for {epochs} epochs…")
        t0 = time.time()
        step = 0
        if self.gen_kind == "vae":
            opt = torch.optim.Adam(self.nets["vae"].parameters(), lr=lr or 1e-3)
        else:
            opt_g = torch.optim.Adam(self.nets["gen"].parameters(), lr=lr or 2e-4, betas=(0.5, 0.999))
            opt_d = torch.optim.Adam(self.nets["critic"].parameters(), lr=lr or 2e-4, betas=(0.5, 0.999))
        stop = False
        for epoch in range(epochs):
            for xb, yb in data.batches("train", bs, shuffle=True, device=dev, limit=limit):
                x01 = data.unprepare_image(xb).reshape(len(xb), -1)  # pixels in 0..1
                y1h = self._onehot(yb, n_classes)
                if self.gen_kind == "vae":
                    logits, mu, logvar = self.nets["vae"](x01, y1h)
                    rec = F.binary_cross_entropy_with_logits(logits, x01, reduction="sum") / len(xb)
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(xb)
                    loss = rec + kl
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    ms.add("loss", "rebuild error", step, rec.item())
                    ms.add("loss", "KL (tidiness)", step, kl.item())
                    info = f"loss {loss.item():.1f}"
                else:
                    real = x01 * 2 - 1
                    z = torch.randn(len(xb), self.latent, device=dev)
                    fake = self.nets["gen"](torch.cat([z, y1h], 1))
                    d_real = self.nets["critic"](torch.cat([real, y1h], 1))
                    d_fake = self.nets["critic"](torch.cat([fake.detach(), y1h], 1))
                    d_loss = F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, 0.9)) + \
                        F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    opt_d.step()
                    g_out = self.nets["critic"](torch.cat([fake, y1h], 1))
                    g_loss = F.binary_cross_entropy_with_logits(g_out, torch.ones_like(g_out))
                    opt_g.zero_grad(set_to_none=True)
                    g_loss.backward()
                    opt_g.step()
                    ms.add("loss", "critic", step, d_loss.item())
                    ms.add("loss", "generator", step, g_loss.item())
                    info = f"critic {d_loss.item():.2f} · generator {g_loss.item():.2f}"
                step += 1
                self.steps_done += 1
                eta = (time.time() - t0) / step * max(0, total - step)
                ms.progress(step, total, f"epoch {epoch + 1}/{epochs} · {info} · ETA {fmt_time(eta)}")
                cb.fire("steps", step)
                if _check_skip():
                    stop = True
                    break
            ms.flush()
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1 or stop:
                show_generated(self, 16, "any", _key=f"Pictures made by {self.label}",
                               _title=f"Pictures made by {self.label} (epoch {epoch + 1})")
                for n in self.nets.values():
                    n.train()  # sampling switched them to eval mode
            cb.fire("epochs", epoch + 1)
            if stop:
                break
        for n in self.nets.values():
            n.eval()
        dt = time.time() - t0
        self.train_seconds += dt
        ms.progress(total, total, "done", force=True)
        emit("log", level="success", text=f"Finished training {label} in {fmt_time(dt)}.")
        emit("train_done", model=self.name, label=label, metrics={}, seconds=round(dt, 2))
        return {}

    # -- sampling ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, n=16, cls="any") -> torch.Tensor:
        if not self.nets or not self.meta:
            raise NBError(f"'{self.name}' hasn't learned anything yet — train it on pictures first.",
                          block_id=self.bid)
        n = max(1, min(int(n), 100))
        y1h = self.class_onehot(cls, n, spread=True)
        z = torch.randn(n, self.latent, device=self.device)
        return self.decode(z, y1h)

    def class_onehot(self, cls, n=1, spread=False) -> torch.Tensor:
        """The class part of the generator's input. 'any' = one class per picture, round-robin
        (``spread``) or at random."""
        classes = (self.meta or {}).get("class_names") or []
        k = len(classes) if (self.meta or {}).get("sig") and classes else 0
        dev = self.device
        if not k:
            return torch.zeros((n, 0), device=dev)
        if cls in (None, "", "any", "all"):
            y = torch.arange(n, device=dev) % k if spread else torch.randint(0, k, (n,), device=dev)
        else:
            names = [str(c).lower() for c in classes]
            key = str(cls).lower()
            if key.endswith(".0") and key[:-2] in names:  # a number block gives 7.0
                key = key[:-2]
            if key not in names:
                raise NBError(f"'{cls}' isn't one of the classes: {', '.join(map(str, classes))}.",
                              block_id=self.bid)
            y = torch.full((n,), names.index(key), device=dev)
        return F.one_hot(y, k).float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, y1h: torch.Tensor) -> torch.Tensor:
        """Noise (plus the one-hot class) → pictures (N, C, H, W) with values 0…1."""
        c, h, w = self.meta["input_shape"]
        z = z.to(self.device)
        if self.gen_kind == "vae":
            x = self.nets["vae"].decode(z, y1h)
        else:
            self.nets["gen"].eval()
            x = (self.nets["gen"](torch.cat([z, y1h], 1)) + 1) / 2
        return x.reshape(len(z), c, h, w).clamp(0, 1).cpu()

    # -- save / load -----------------------------------------------------------------
    def to_payload(self):
        return {"format": "neuroblocks-model", "family": "generator", "name": self.name, "kind": self.gen_kind,
                "latent": self.latent, "meta": self.meta,
                "state": {k: v.state_dict() for k, v in self.nets.items()}, "history": self.history}

    @classmethod
    def from_payload(cls, payload, name=None):
        g = cls(name or payload["name"], payload["kind"], payload["latent"])
        g.meta = payload["meta"]
        shape = tuple(g.meta["input_shape"])
        k = len(g.meta.get("class_names") or [])
        g._build(shape, k)
        for key, sd in payload["state"].items():
            g.nets[key].load_state_dict(sd)
            g.nets[key].eval()
        g.history = payload.get("history") or {}
        return g


def generator(name="generator", kind="vae", latent=16, _bid=None):
    return ImageGenerator(name, kind, latent, _bid)


def show_generated(model, count=16, cls="any", _bid=None, _key=None, _title=None):
    """Show new pictures made by a generator (Results tab)."""
    from .data import image_grid_png
    from .evaluate import emit_image
    if getattr(model, "family", None) != "generator":
        raise NBError("This block needs a picture generator (Neural Nets ▸ create picture generator).", block_id=_bid)
    imgs = model.sample(count, cls)
    title = _title or f"New pictures from {model.label}" + ("" if cls in (None, "", "any") else f" · {cls}")
    emit_image(title, image_grid_png(imgs, cols=min(8, len(imgs))),
               "These pictures are invented by the model — they are not in the dataset.", block=_bid,
               key=_key or title)
