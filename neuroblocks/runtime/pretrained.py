"""Pretrained language models from Hugging Face (fine-tune them on your own text)."""
from __future__ import annotations

from pathlib import Path

import torch

from .core import register
from .errors import NBError, missing_package
from .events import emit

__all__ = ["PretrainedLM", "pretrained_lm"]


class PretrainedLM:
    kind = "model"
    family = "pretrained_lm"

    def __init__(self, name="model", model_id="distilgpt2", _bid=None, _local_dir=None):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise missing_package("transformers", "Pretrained language models",
                                  "transformers accelerate") from None
        from .model import _run_label
        from .tokenizers import HFTokenizer
        self.name = str(name)
        self.label = _run_label(self.name)
        self.model_id = model_id
        self.bid = _bid
        src = str(_local_dir or model_id)
        emit("log", level="info", text=f"Loading pretrained model '{model_id}' (the first time downloads it)…")
        tok = AutoTokenizer.from_pretrained(src)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = HFTokenizer(model_id, tok)
        self.net = AutoModelForCausalLM.from_pretrained(src)
        cfg = self.net.config
        ctx = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None) or 1024
        self.meta = {"task": "language_model", "modality": "tokens", "context": int(min(ctx, 1024)),
                     "tokenizer": {"kind": "huggingface", "model_id": model_id}, "input_shape": [int(min(ctx, 1024))]}
        self.history: dict = {}
        self.steps_done = 0
        self.train_seconds = 0.0
        self._adapted: dict = {}
        register(self)
        self.announce()

    def describe(self):
        return f"pretrained model '{self.name}' ({self.model_id})"

    def record(self, metric, step, value):
        self.history.setdefault(metric, []).append((step, float(value)))

    @property
    def device(self):
        return next(self.net.parameters()).device

    def num_params(self):
        return sum(p.numel() for p in self.net.parameters())

    def announce(self):
        cfg = self.net.config
        rows = []
        for key, label in (("n_layer", "transformer layers"), ("num_hidden_layers", "transformer layers"),
                           ("n_head", "attention heads"), ("num_attention_heads", "attention heads"),
                           ("n_embd", "embedding size"), ("hidden_size", "embedding size"),
                           ("vocab_size", "vocabulary")):
            v = getattr(cfg, key, None)
            if v is not None and not any(r["name"] == label for r in rows):
                rows.append({"name": label, "detail": str(v), "shape": None, "params": None, "depth": 0})
        emit("model", name=self.name, info={"summary": f"{self.label}: pretrained {self.model_id}, "
                                                       f"{self.num_params():,} parameters",
                                            "rows": rows, "total_params": self.num_params(), "label": self.label,
                                            "family": self.family, "task": "language_model"})

    # -- used by training / generation -----------------------------------------------
    def prepare(self, dev):
        self.net.to(dev)
        return self.net

    def adapt_dataset(self, data):
        key = id(data)
        if key not in self._adapted:
            ctx = min(int(data.context or 128), self.meta["context"])
            emit("log", level="info", text=f"Re-tokenizing '{data.name}' with {self.model_id}'s tokenizer…")
            self._adapted[key] = data.retokenized(self.tokenizer, ctx)
        return self._adapted[key]

    def lm_logits(self, x):
        return self.net(input_ids=x).logits

    @torch.no_grad()
    def generate_text(self, prompt, length, temperature=0.8, top_k=None):
        self.net.eval()
        tok = self.tokenizer.tok
        ids = tok(prompt or tok.eos_token, return_tensors="pt").input_ids.to(self.device)
        out = self.net.generate(ids, max_new_tokens=int(length), do_sample=temperature > 0,
                                temperature=max(float(temperature), 1e-5), top_k=int(top_k or 50),
                                pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    # -- save / load ------------------------------------------------------------------
    def to_payload(self, path: Path):
        folder = Path(path).with_suffix("")
        folder = folder.parent / (folder.name + "_hf")
        self.net.save_pretrained(folder)
        self.tokenizer.tok.save_pretrained(folder)
        return {"format": "neuroblocks-model", "family": "pretrained_lm", "name": self.name,
                "model_id": self.model_id, "dir": folder.name, "history": self.history}

    @classmethod
    def from_payload(cls, payload, name=None, path: Path | None = None):
        from .. import paths
        folder = paths.models_dir() / payload["dir"]
        m = cls(name or payload.get("name", "model"), payload["model_id"], _local_dir=folder if folder.exists() else None)
        m.history = payload.get("history") or {}
        return m


def pretrained_lm(model_id="distilgpt2", name="model", _bid=None):
    if not str(model_id).strip():
        raise NBError("Type the name of a Hugging Face model, e.g. distilgpt2 or gpt2.", block_id=_bid)
    return PretrainedLM(name, str(model_id).strip(), _bid=_bid)
