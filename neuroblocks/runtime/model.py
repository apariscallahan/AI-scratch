"""Neural network models built from layer blocks."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .core import register
from .errors import NBError
from .events import emit
from .layers import (BuildCtx, Dropout, LayerNorm, LayerSpec, Output, PositionalEmbedding, Repeat,
                     TokenEmbedding, TransformerBlock, build_layers, gpt_init, spec_from_dict, spec_to_dict, _Seq)

__all__ = ["Model", "GPT", "show_model", "param_count", "reset"]

_NAME_COUNTS: dict[str, int] = {}


def _run_label(name: str) -> str:
    n = _NAME_COUNTS.get(name, 0) + 1
    _NAME_COUNTS[name] = n
    return name if n == 1 else f"{name} #{n}"


class Model:
    """A neural network made of layer specs. It is built (sized) when it first meets data."""

    kind = "model"
    family = "neural"

    def __init__(self, name: str = "model", layers=(), _bid: str | None = None):
        self.name = str(name)
        self.label = _run_label(self.name)
        self.specs: list[LayerSpec] = list(layers)
        self.bid = _bid
        self.net: nn.Module | None = None
        self.meta: dict | None = None
        self.signature = None
        self.rows: list[dict] = []
        self.history: dict[str, list] = {}
        self.steps_done = 0
        self.epochs_done = 0
        self.train_seconds = 0.0
        self.agent = None  # set when trained with reinforcement learning
        if not self.specs:
            emit("log", level="info", text=f"Model '{self.name}' has no layers yet — an output layer will be "
                                            f"added automatically.")
        register(self)

    # -- facts -----------------------------------------------------------------
    def describe(self) -> str:
        return f"model '{self.name}'"

    def __repr__(self):
        return f"<Model {self.name}: {len(self.specs)} layer blocks>"

    @property
    def task(self) -> str | None:
        return (self.meta or {}).get("task")

    @property
    def device(self):
        if self.net is None:
            return torch.device("cpu")
        try:
            return next(self.net.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def num_params(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters())

    def record(self, metric: str, step, value):
        self.history.setdefault(metric, []).append((step, float(value)))

    # -- building ----------------------------------------------------------------
    def build_for(self, data, objective: str = "auto"):
        """Build (or reuse) the network so that it fits ``data``."""
        reconstruct = objective in ("reconstruct", "denoise")
        if reconstruct and data.input_is_tokens:
            raise NBError("Autoencoder goals (reconstruct/denoise) work with numbers and images, not text.")
        out_shape = tuple(data.input_shape) if reconstruct else None
        vocab = data.tokenizer.vocab_size if data.tokenizer is not None else None
        sig = (data.task, tuple(data.input_shape), data.output_size, data.modality, data.context, out_shape, vocab)
        if self.net is not None and self.signature == sig:
            return self.net
        if self.net is not None:
            emit("log", level="warn", text=f"Model '{self.name}' was built for different data, so it was "
                                            f"rebuilt with fresh weights.")
        ctx = BuildCtx(task=data.task, output_size=data.output_size, input_is_tokens=data.input_is_tokens,
                       vocab_size=vocab, context=data.context, output_shape=out_shape)
        self._build(ctx, tuple(data.input_shape), data.input_is_tokens)
        self.signature = sig
        self.meta = data.meta()
        self.meta["objective"] = "reconstruct" if reconstruct else "predict"
        self.announce()
        return self.net

    def _expected(self, ctx: BuildCtx, in_shape: tuple):
        if ctx.output_shape is not None:
            return tuple(ctx.output_shape)
        if ctx.task == "language_model":
            return (in_shape[0], ctx.output_size)
        return (int(ctx.output_size),)

    def _build(self, ctx: BuildCtx, in_shape: tuple, tokens: bool):
        mods, rows, out_shape, out_tokens = build_layers(self.specs, ctx, in_shape, tokens)
        need = self._expected(ctx, in_shape)
        if tuple(out_shape) != need:
            last = self.specs[-1] if self.specs else None
            if isinstance(last, Output):
                raise NBError(f"The network ends with shape {tuple(out_shape)} but the data needs {need}.",
                              block_id=last.bid)
            if ctx.task == "language_model" and len(out_shape) == 1 and not tokens:
                raise NBError("A language model needs one prediction per token, but this network combines the "
                              "whole sequence into one.",
                              hint="Remove pooling / 'keep last step only' layers.", block_id=self.bid)
            if out_tokens:
                raise NBError("This network reads text, so it needs a 'token embedding' layer first.",
                              hint="Add 'token embedding' at the top of the network.", block_id=self.bid)
            extra_mods, extra_rows, out_shape, _ = build_layers([Output()], ctx, tuple(out_shape), False)
            mods += extra_mods
            for r in extra_rows:
                r["auto"] = True
            rows += extra_rows
            what = {"classification": f"{ctx.output_size} units, one per class",
                    "language_model": f"{ctx.output_size} units, one per token in the vocabulary",
                    "regression": f"{ctx.output_size} unit(s), one per number to predict"}.get(ctx.task, "")
            if ctx.output_shape is not None:
                what = "the same size as the input"
            ctx.note(f"Added an output layer at the end ({what}).", self.bid)
            if tuple(out_shape) != need:
                raise NBError(f"The network ends with shape {tuple(out_shape)} but the data needs {need}.",
                              block_id=self.bid)
        net = _Seq(mods)
        if ctx.n_transformer:
            gpt_init(net, ctx.n_transformer)
        self.net = net
        self.rows = rows
        for msg, bid in ctx.notes:
            emit("log", level="info", text=msg, block=bid)

    def rebuild_from_meta(self):
        """Rebuild the network for a model loaded from a file (no dataset needed)."""
        m = self.meta
        task = m["task"]
        if task == "classification":
            out = len(m["class_names"] or [])
        elif task == "language_model":
            from .tokenizers import tokenizer_from_dict
            out = tokenizer_from_dict(m["tokenizer"]).vocab_size
        else:
            out = len(m.get("target_names") or [1])
        tokens = m["modality"] in ("tokens", "text")
        vocab = None
        if m.get("tokenizer"):
            from .tokenizers import tokenizer_from_dict
            vocab = tokenizer_from_dict(m["tokenizer"]).vocab_size
        out_shape = tuple(m["input_shape"]) if m.get("objective") == "reconstruct" else None
        ctx = BuildCtx(task=task, output_size=out, input_is_tokens=tokens, vocab_size=vocab,
                       context=m.get("context"), output_shape=out_shape)
        self._build(ctx, tuple(m["input_shape"]), tokens)

    def announce(self):
        total = self.num_params()
        info = {
            "summary": f"{self.label}: {len(self.rows)} layers, {total:,} parameters",
            "rows": self.rows, "total_params": total,
            "input_shape": (self.meta or {}).get("input_shape"), "task": self.task, "label": self.label,
            "family": self.family,
        }
        emit("model", name=self.name, info=info, block=self.bid)

    # -- save / load helpers ------------------------------------------------------
    def to_payload(self) -> dict:
        return {"format": "neuroblocks-model", "family": self.family, "name": self.name,
                "specs": [spec_to_dict(s) for s in self.specs], "meta": self.meta,
                "state_dict": self.net.state_dict() if self.net is not None else None,
                "history": self.history, "steps_done": self.steps_done}

    @classmethod
    def from_payload(cls, payload: dict, name: str | None = None):
        m = cls(name or payload.get("name", "model"), [spec_from_dict(d) for d in payload["specs"]])
        m.meta = payload.get("meta")
        m.history = payload.get("history") or {}
        m.steps_done = payload.get("steps_done", 0)
        if payload.get("state_dict") is not None and m.meta:
            m.rebuild_from_meta()
            m.net.load_state_dict(payload["state_dict"])
            m.announce()
        return m


def GPT(name: str = "model", layers=4, heads=4, embed=128, dropout=0.1, _bid=None) -> Model:
    """A GPT-style transformer language model (prebuilt)."""
    layers, heads, embed = int(layers), int(heads), int(embed)
    if embed % heads:
        raise NBError(f"The embedding size ({embed}) must divide evenly by the number of heads ({heads}).",
                      block_id=_bid)
    specs = [TokenEmbedding(embed, _bid=_bid), PositionalEmbedding(_bid=_bid), Dropout(float(dropout), _bid=_bid),
             Repeat(layers, [TransformerBlock(heads, float(dropout), _bid=_bid)], _bid=_bid),
             LayerNorm(_bid=_bid), Output(_bid=_bid)]
    m = Model(name, specs, _bid=_bid)
    m.family = "neural"
    return m


def show_model(model, data=None):
    if getattr(model, "family", None) != "neural":
        emit("log", level="info", text=f"{getattr(model, 'name', model)}: {getattr(model, 'describe', lambda: '')()}")
        return
    if data is not None:
        model.build_for(data)
    if model.net is not None:
        model.announce()
        return
    rows = [{"name": s.label, "detail": s.detail(), "shape": None, "params": None, "block": s.bid, "depth": 0}
            for s in model.specs]
    emit("model", name=model.name, info={"summary": f"{model.label}: not built yet (sizes appear after it meets "
                                                    f"data)", "rows": rows, "total_params": None,
                                         "label": model.label, "family": model.family})


def param_count(model) -> int:
    n = getattr(model, "num_params", None)
    return int(n()) if callable(n) else 0


def reset(model):
    """Forget everything the model learned (fresh random weights next time it trains)."""
    if hasattr(model, "reset_weights"):
        model.reset_weights()
        return
    model.net = None
    model.signature = None
    model.history = {}
    model.steps_done = 0
    model.epochs_done = 0
    model.label = _run_label(model.name)
    emit("log", level="info", text=f"Model '{model.name}' was reset.")
