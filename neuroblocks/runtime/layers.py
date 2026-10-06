"""Layer blocks.

Each layer block becomes a lightweight *spec* (e.g. ``Dense(64, activation="relu")``).
Specs are turned into real PyTorch modules only when the model meets its data,
so input sizes are inferred automatically: you never type "784 inputs".

Shape conventions (without the batch dimension):
  tabular ``(F,)`` · images ``(C, H, W)`` · sequences ``(T, C)`` · token ids ``(T,)``
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .errors import NBError

__all__ = ["Dense", "Output", "Activation", "Dropout", "BatchNorm", "LayerNorm", "Flatten", "Conv2D", "Pool",
           "GlobalPool", "SeqPool", "Upsample", "Reshape", "TokenEmbedding", "PositionalEmbedding",
           "TransformerBlock", "Recurrent", "Repeat", "Residual", "ACTIVATIONS", "make_activation"]

ACTIVATIONS = {
    "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU, "elu": nn.ELU, "softplus": nn.Softplus,
}


def make_activation(kind: str | None):
    kind = (kind or "none").lower()
    if kind in ("none", "linear", ""):
        return None
    if kind == "softmax":
        return nn.Softmax(dim=-1)
    if kind not in ACTIVATIONS:
        raise NBError(f"Unknown activation '{kind}'.")
    return ACTIVATIONS[kind]()


class BuildCtx:
    """What layers need to know about the data while building."""

    def __init__(self, *, task: str, output_size: int, input_is_tokens: bool, vocab_size: int | None = None,
                 context: int | None = None, output_shape: tuple | None = None):
        self.task = task
        self.output_size = output_size
        self.input_is_tokens = input_is_tokens
        self.vocab_size = vocab_size
        self.context = context
        self.output_shape = output_shape  # e.g. image shape for autoencoders
        self.notes: list[tuple[str, str | None]] = []  # (message, block id)
        self.n_transformer = 0

    @property
    def causal(self) -> bool:
        return self.task == "language_model"

    def note(self, msg: str, bid: str | None = None):
        self.notes.append((msg, bid))


def kind_of(shape: tuple, is_tokens: bool = False) -> str:
    if is_tokens and len(shape) == 1:
        return "tokens"
    return {1: "flat", 2: "sequence", 3: "image"}.get(len(shape), "other")


class LayerSpec:
    label = "layer"

    def __init__(self, _bid: str | None = None):
        self.bid = _bid

    def detail(self) -> str:
        return ""

    def build(self, ctx: BuildCtx, shape: tuple, tokens: bool) -> list[tuple["LayerSpec", nn.Module]]:
        """Return a list of (spec, module) pairs (auto-fixes may add extra modules)."""
        raise NotImplementedError

    def fail(self, msg: str, hint: str | None = None):
        raise NBError(msg, hint=hint, block_id=self.bid)

    def __repr__(self):
        d = self.detail()
        return f"{self.label}({d})" if d else self.label


class _Lambda(nn.Module):
    def __init__(self, fn, name="op"):
        super().__init__()
        self.fn = fn
        self.name = name

    def forward(self, x):
        return self.fn(x)

    def extra_repr(self):
        return self.name


class _Seq(nn.Module):
    """Sequential container whose children are listed in the summary."""

    def __init__(self, mods):
        super().__init__()
        self.layers = nn.ModuleList(mods)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class _Auto(LayerSpec):
    """A layer inserted automatically to make things fit."""

    def __init__(self, label, detail="", bid=None):
        super().__init__(bid)
        self.label = label
        self._detail = detail

    def detail(self):
        return self._detail


def _flatten_fix(ctx, spec, shape):
    ctx.note(f"Added a 'flatten' before '{spec.label}' because it received a {len(shape)}-D input "
             f"{tuple(shape)}.", spec.bid)
    return (_Auto("flatten", "added automatically", spec.bid), nn.Flatten())


def _pool_fix(ctx, spec, shape):
    ctx.note(f"Added a 'global average pool' before '{spec.label}' to combine the {shape[0]} sequence "
             f"steps into one.", spec.bid)
    return (_Auto("average over steps", "added automatically", spec.bid), _Lambda(lambda x: x.mean(1), "mean over T"))


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------


class Dense(LayerSpec):
    label = "dense"

    def __init__(self, units=64, activation="relu", _bid=None):
        super().__init__(_bid)
        self.units = int(units)
        self.activation = activation or "none"
        if self.units < 1:
            self.fail("A dense layer needs at least 1 unit.")

    def detail(self):
        return f"{self.units} units" + (f", {self.activation}" if self.activation not in ("none", None) else "")

    def build(self, ctx, shape, tokens):
        out = []
        if tokens and len(shape) == 1:
            self.fail("A dense layer can't read token ids directly.",
                      "Put a 'token embedding' layer first to turn tokens into vectors.")
        if len(shape) == 3:
            out.append(_flatten_fix(ctx, self, shape))
            shape = (int(math.prod(shape)),)
        mods = [nn.Linear(shape[-1], self.units)]
        act = make_activation(self.activation)
        if act is not None:
            mods.append(act)
        out.append((self, nn.Sequential(*mods)))
        return out


class Output(LayerSpec):
    """The last layer: automatically sized to what the data needs (classes / numbers / tokens)."""
    label = "output"

    def __init__(self, activation="none", _bid=None):
        super().__init__(_bid)
        self.activation = activation or "none"

    def detail(self):
        return "auto-sized"

    def build(self, ctx, shape, tokens):
        out = []
        if tokens and len(shape) == 1:
            self.fail("The output layer can't read token ids directly.",
                      "Put a 'token embedding' layer first.")
        if ctx.output_shape is not None:  # autoencoder: reproduce the input shape
            if len(shape) > 1:
                out.append(_flatten_fix(ctx, self, shape))
                shape = (int(math.prod(shape)),)
            target = tuple(ctx.output_shape)
            lin = nn.Linear(shape[-1], int(math.prod(target)))
            mods = [lin, nn.Unflatten(1, target)]
            act = make_activation(self.activation)
            if act is not None:
                mods.append(act)
            out.append((self, nn.Sequential(*mods)))
            return out
        if len(shape) == 3:
            out.append(_flatten_fix(ctx, self, shape))
            shape = (int(math.prod(shape)),)
        if len(shape) == 2 and ctx.task != "language_model":
            out.append(_pool_fix(ctx, self, shape))
            shape = (shape[-1],)
        mods = [nn.Linear(shape[-1], int(ctx.output_size))]
        act = make_activation(self.activation)
        if act is not None:
            mods.append(act)
        out.append((self, nn.Sequential(*mods)))
        return out


class Activation(LayerSpec):
    label = "activation"

    def __init__(self, kind="relu", _bid=None):
        super().__init__(_bid)
        self.kind = kind

    def detail(self):
        return self.kind

    def build(self, ctx, shape, tokens):
        act = make_activation(self.kind)
        return [(self, act if act is not None else nn.Identity())]


class Dropout(LayerSpec):
    label = "dropout"

    def __init__(self, rate=0.2, _bid=None):
        super().__init__(_bid)
        self.rate = float(rate)
        if not 0 <= self.rate < 1:
            self.fail("Dropout must be between 0 and 1 (e.g. 0.2 drops 20% of signals while training).")

    def detail(self):
        return f"{self.rate:g}"

    def build(self, ctx, shape, tokens):
        if len(shape) == 3:
            return [(self, nn.Dropout2d(self.rate))]
        return [(self, nn.Dropout(self.rate))]


class _SeqBatchNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class BatchNorm(LayerSpec):
    label = "batch norm"

    def build(self, ctx, shape, tokens):
        if tokens and len(shape) == 1:
            self.fail("Batch norm can't read token ids.", "Put a 'token embedding' first.")
        if len(shape) == 3:
            return [(self, nn.BatchNorm2d(shape[0]))]
        if len(shape) == 2:
            return [(self, _SeqBatchNorm(shape[1]))]
        return [(self, nn.BatchNorm1d(shape[0]))]


class LayerNorm(LayerSpec):
    label = "layer norm"

    def build(self, ctx, shape, tokens):
        if tokens and len(shape) == 1:
            self.fail("Layer norm can't read token ids.", "Put a 'token embedding' first.")
        if len(shape) == 3:  # normalise each pixel's channels
            ln = nn.GroupNorm(1, shape[0])
            return [(self, ln)]
        return [(self, nn.LayerNorm(shape[-1]))]


class Flatten(LayerSpec):
    label = "flatten"

    def build(self, ctx, shape, tokens):
        return [(self, nn.Flatten())]


class Reshape(LayerSpec):
    label = "reshape"

    def __init__(self, shape="1,28,28", _bid=None):
        super().__init__(_bid)
        try:
            if isinstance(shape, (list, tuple)):
                self.shape = tuple(int(s) for s in shape)
            else:
                cleaned = str(shape).replace("x", ",").replace("×", ",").strip("()[] ")
                self.shape = tuple(int(s) for s in cleaned.split(",") if s.strip())
        except ValueError:
            self.fail(f"Couldn't read the shape '{shape}'. Use numbers separated by commas, like 16,7,7.")

    def detail(self):
        return "×".join(map(str, self.shape))

    def build(self, ctx, shape, tokens):
        if math.prod(shape) != math.prod(self.shape):
            self.fail(f"Can't reshape {math.prod(shape)} numbers {tuple(shape)} into {self.shape} "
                      f"({math.prod(self.shape)} numbers).",
                      f"Make the layer before produce {math.prod(self.shape)} numbers.")
        target = self.shape
        return [(self, _Lambda(lambda x: x.reshape(x.shape[0], *target), f"reshape{target}"))]


# ---------------------------------------------------------------------------
# Image layers
# ---------------------------------------------------------------------------


class Conv2D(LayerSpec):
    label = "conv 2D"

    def __init__(self, filters=32, size=3, stride=1, activation="relu", _bid=None):
        super().__init__(_bid)
        self.filters, self.size, self.stride = int(filters), int(size), int(stride)
        self.activation = activation or "none"

    def detail(self):
        s = f"{self.filters} filters {self.size}×{self.size}"
        if self.stride != 1:
            s += f" stride {self.stride}"
        if self.activation not in ("none", None):
            s += f", {self.activation}"
        return s

    def build(self, ctx, shape, tokens):
        if len(shape) != 3:
            self.fail("Convolution layers need images (channels × height × width).",
                      "Use dense layers for tables of numbers, or load an image dataset.")
        if self.size > shape[1] + 2 * (self.size // 2):
            self.fail(f"The filter ({self.size}×{self.size}) is bigger than the image ({shape[1]}×{shape[2]}).")
        mods = [nn.Conv2d(shape[0], self.filters, self.size, stride=self.stride, padding=self.size // 2)]
        act = make_activation(self.activation)
        if act is not None:
            mods.append(act)
        return [(self, nn.Sequential(*mods))]


class Pool(LayerSpec):
    label = "pool"

    def __init__(self, kind="max", size=2, _bid=None):
        super().__init__(_bid)
        self.kind = kind
        self.size = int(size)

    def detail(self):
        return f"{self.kind} {self.size}×{self.size}"

    def build(self, ctx, shape, tokens):
        if len(shape) == 3:
            if shape[1] < self.size or shape[2] < self.size:
                self.fail(f"The image is already only {shape[1]}×{shape[2]} — too small to pool by {self.size}.",
                          "Remove a pooling layer or use bigger images.")
            m = nn.MaxPool2d(self.size) if self.kind == "max" else nn.AvgPool2d(self.size)
            return [(self, m)]
        if len(shape) == 2:
            k = self.size
            pool = F.max_pool1d if self.kind == "max" else F.avg_pool1d
            return [(self, _Lambda(lambda x: pool(x.transpose(1, 2), k).transpose(1, 2), f"{self.kind} pool over steps"))]
        self.fail("Pooling needs images or sequences.")


class GlobalPool(LayerSpec):
    label = "global pool"

    def __init__(self, kind="average", _bid=None):
        super().__init__(_bid)
        self.kind = kind

    def detail(self):
        return self.kind

    def build(self, ctx, shape, tokens):
        mx = self.kind == "max"
        if len(shape) == 3:
            fn = (lambda x: x.amax(dim=(2, 3))) if mx else (lambda x: x.mean(dim=(2, 3)))
            return [(self, _Lambda(fn, "global pool"))]
        if len(shape) == 2:
            fn = (lambda x: x.amax(dim=1)) if mx else (lambda x: x.mean(dim=1))
            return [(self, _Lambda(fn, "pool over steps"))]
        return [(self, nn.Identity())]


class SeqPool(LayerSpec):
    label = "combine steps"

    def __init__(self, mode="mean", _bid=None):
        super().__init__(_bid)
        self.mode = mode

    def detail(self):
        return self.mode

    def build(self, ctx, shape, tokens):
        if len(shape) != 2:
            self.fail("'combine steps' needs a sequence (e.g. after an LSTM or transformer).")
        fns = {"mean": lambda x: x.mean(1), "max": lambda x: x.amax(1), "last": lambda x: x[:, -1],
               "first": lambda x: x[:, 0]}
        return [(self, _Lambda(fns[self.mode], f"{self.mode} of steps"))]


class Upsample(LayerSpec):
    label = "upsample"

    def __init__(self, scale=2, _bid=None):
        super().__init__(_bid)
        self.scale = int(scale)

    def detail(self):
        return f"×{self.scale}"

    def build(self, ctx, shape, tokens):
        if len(shape) != 3:
            self.fail("Upsample needs images.", "Use 'reshape' to turn numbers into an image first.")
        return [(self, nn.Upsample(scale_factor=self.scale, mode="nearest"))]


# ---------------------------------------------------------------------------
# Text & sequence layers
# ---------------------------------------------------------------------------


class TokenEmbedding(LayerSpec):
    label = "token embedding"

    def __init__(self, dim=128, _bid=None):
        super().__init__(_bid)
        self.dim = int(dim)

    def detail(self):
        return f"{self.dim} numbers per token"

    def build(self, ctx, shape, tokens):
        if not (tokens and len(shape) == 1):
            self.fail("A token embedding needs text data (token ids).",
                      "Load a text dataset, or remove this layer.")
        if not ctx.vocab_size:
            self.fail("This dataset has no tokenizer.")
        return [(self, nn.Embedding(ctx.vocab_size, self.dim))]


class _PosEmb(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        T = x.shape[1]
        if T > self.pos.shape[1]:
            raise NBError(f"This input has {T} steps but the positional embedding only knows "
                          f"{self.pos.shape[1]}.", hint="Use a shorter prompt or a longer context length.")
        return x + self.pos[:, :T]


class PositionalEmbedding(LayerSpec):
    label = "positional embedding"

    def detail(self):
        return "learned positions"

    def build(self, ctx, shape, tokens):
        if len(shape) != 2:
            self.fail("A positional embedding goes after a 'token embedding' (it needs a sequence).")
        length = max(int(ctx.context or shape[0]), shape[0])
        return [(self, _PosEmb(length, shape[1]))]


class _TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout, causal, ff_mult=4):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout
        self.proj.nb_residual = True
        self.ff[2].nb_residual = True

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TransformerBlock(LayerSpec):
    label = "transformer block"

    def __init__(self, heads=4, dropout=0.1, mask="auto", _bid=None):
        super().__init__(_bid)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.mask = mask

    def detail(self):
        m = {"causal": ", can't see the future", "full": ", sees everything"}.get(self.mask, "")
        return f"{self.heads} heads{m}"

    def build(self, ctx, shape, tokens):
        if tokens and len(shape) == 1:
            self.fail("A transformer block needs vectors, not token ids.", "Put a 'token embedding' first.")
        if len(shape) == 1:
            self.fail("A transformer block needs a sequence.", "Use it on text, or after a 'token embedding'.")
        if len(shape) == 3:
            self.fail("A transformer block needs a sequence, not an image.")
        dim = shape[-1]
        if self.heads < 1 or dim % self.heads != 0:
            ok = [h for h in range(1, 17) if dim % h == 0]
            self.fail(f"The embedding size ({dim}) must divide evenly by the number of heads ({self.heads}).",
                      f"Try {', '.join(map(str, ok[-4:]))} heads, or change the embedding size.")
        causal = ctx.causal if self.mask == "auto" else (self.mask == "causal")
        ctx.n_transformer += 1
        return [(self, _TransformerBlock(dim, self.heads, self.dropout, causal))]


class _RNN(nn.Module):
    def __init__(self, kind, inp, units, keep_all):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[kind]
        self.rnn = cls(inp, units, batch_first=True)
        self.keep_all = keep_all

    def forward(self, x):
        out, _ = self.rnn(x)
        return out if self.keep_all else out[:, -1]


class Recurrent(LayerSpec):
    label = "recurrent"

    def __init__(self, kind="lstm", units=128, keep="all", _bid=None):
        super().__init__(_bid)
        self.kind = kind.lower()
        self.units = int(units)
        self.keep = keep

    def detail(self):
        return f"{self.kind.upper()} {self.units} units, " + ("all steps" if self.keep == "all" else "last step")

    def build(self, ctx, shape, tokens):
        if tokens and len(shape) == 1:
            self.fail(f"An {self.kind.upper()} layer needs vectors, not token ids.",
                      "Put a 'token embedding' first.")
        out = []
        if len(shape) == 1:
            ctx.note(f"Treating the {shape[0]} input numbers as a sequence of {shape[0]} steps for the "
                     f"{self.kind.upper()}.", self.bid)
            out.append((_Auto("as sequence", "added automatically", self.bid), _Lambda(lambda x: x.unsqueeze(-1), "unsqueeze")))
            shape = (shape[0], 1)
        if len(shape) == 3:
            self.fail(f"An {self.kind.upper()} layer needs a sequence, not an image.")
        keep_all = self.keep == "all"
        if ctx.task == "language_model":
            keep_all = True
        out.append((self, _RNN(self.kind, shape[-1], self.units, keep_all)))
        return out


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class Repeat(LayerSpec):
    label = "repeat"

    def __init__(self, times=2, layers=(), _bid=None):
        super().__init__(_bid)
        self.times = int(times)
        self.layers = list(layers)

    def detail(self):
        return f"{self.times}×"

    def build(self, ctx, shape, tokens):
        raise AssertionError("Repeat is expanded by the builder")


class _Residual(nn.Module):
    def __init__(self, body, proj):
        super().__init__()
        self.body = body
        self.proj = proj

    def forward(self, x):
        return self.body(x) + (self.proj(x) if self.proj is not None else x)


class Residual(LayerSpec):
    label = "residual"

    def __init__(self, layers=(), _bid=None):
        super().__init__(_bid)
        self.layers = list(layers)

    def detail(self):
        return "adds a shortcut around its layers"

    def build(self, ctx, shape, tokens):
        raise AssertionError("Residual is built by the builder")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _dummy(shape, tokens, device="cpu"):
    if tokens and len(shape) == 1:
        return torch.zeros((2, *shape), dtype=torch.long, device=device)
    return torch.zeros((2, *shape), device=device)


def _probe(mod: nn.Module, shape, tokens):
    was = mod.training
    mod.eval()
    try:
        with torch.no_grad():
            y = mod(_dummy(shape, tokens))
    finally:
        mod.train(was)
    return tuple(y.shape[1:])


def build_layers(specs, ctx: BuildCtx, shape: tuple, tokens: bool, depth: int = 0):
    """Build specs into modules. Returns (modules, rows, out_shape, out_is_tokens)."""
    modules: list[nn.Module] = []
    rows: list[dict] = []
    for spec in specs:
        if isinstance(spec, Repeat):
            if spec.times < 1:
                spec.fail("Repeat needs at least 1.")
            for i in range(spec.times):
                mods, sub_rows, shape, tokens = build_layers(spec.layers, ctx, shape, tokens, depth + 1)
                modules.extend(mods)
                for r in sub_rows:
                    r["repeat"] = f"{i + 1}/{spec.times}"
                rows.extend(sub_rows)
            continue
        if isinstance(spec, Residual):
            mods, sub_rows, new_shape, new_tokens = build_layers(spec.layers, ctx, shape, tokens, depth + 1)
            body = _Seq(mods)
            proj = None
            if tuple(new_shape) != tuple(shape):
                if len(shape) == len(new_shape) == 3 and shape[1:] == new_shape[1:]:
                    proj = nn.Conv2d(shape[0], new_shape[0], 1)
                elif len(shape) == len(new_shape) and shape[:-1] == new_shape[:-1]:
                    proj = nn.Linear(shape[-1], new_shape[-1])
                else:
                    spec.fail(f"The layers inside 'residual' change the shape from {shape} to {new_shape}, "
                              f"so the shortcut can't be added.",
                              "Keep the size the same inside a residual block (no pooling/flatten inside).")
                ctx.note("The residual shortcut got a small projection because the size changed inside it.",
                         spec.bid)
            mod = _Residual(body, proj)
            params = sum(p.numel() for p in mod.parameters())
            rows.append({"name": "residual", "detail": f"shortcut around {len(sub_rows)} layers",
                         "shape": list(new_shape), "params": params - sum(r["params"] for r in sub_rows),
                         "block": spec.bid, "depth": depth})
            for r in sub_rows:
                r["depth"] = r.get("depth", depth) + 1
            rows.extend(sub_rows)
            modules.append(mod)
            shape, tokens = new_shape, new_tokens
            continue
        if not isinstance(spec, LayerSpec):
            raise NBError(f"'{spec}' is not a layer.")
        for s, mod in spec.build(ctx, shape, tokens):
            try:
                new_shape = _probe(mod, shape, tokens)
            except NBError:
                raise
            except Exception as e:  # noqa: BLE001
                raise NBError(f"The '{s.label}' layer doesn't fit after the previous layer "
                              f"(input {tuple(shape)}): {e}", block_id=s.bid) from e
            tokens = False
            params = sum(p.numel() for p in mod.parameters())
            rows.append({"name": s.label, "detail": s.detail(), "shape": list(new_shape), "params": params,
                         "block": s.bid, "depth": depth, "auto": isinstance(s, _Auto)})
            modules.append(mod)
            shape = new_shape
    return modules, rows, shape, tokens


def spec_to_dict(spec: LayerSpec) -> dict:
    args = {}
    for k, v in vars(spec).items():
        if k in ("bid",) or k.startswith("_"):
            continue
        if k == "layers":
            v = [spec_to_dict(s) for s in v]
        elif isinstance(v, tuple):
            v = list(v)
        args[k] = v
    return {"type": type(spec).__name__, "args": args}


def spec_from_dict(d: dict) -> LayerSpec:
    cls = globals().get(d["type"])
    if cls is None or not (isinstance(cls, type) and issubclass(cls, LayerSpec)):
        raise NBError(f"Unknown layer type in saved model: {d['type']}")
    args = dict(d.get("args") or {})
    if "layers" in args:
        args["layers"] = [spec_from_dict(s) for s in args["layers"]]
    return cls(**args)


def gpt_init(net: nn.Module, n_layers: int):
    """GPT-2 style initialisation (helps transformers train)."""
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "nb_residual", False):
                std = 0.02 / math.sqrt(2 * max(1, n_layers))
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
