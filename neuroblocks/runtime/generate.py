"""Text generation with language models."""
from __future__ import annotations

import time

import torch

from .core import device
from .errors import NBError
from .events import emit

__all__ = ["generate", "generated_text", "tokenizer_of", "sample_tokens"]


def tokenizer_of(model):
    tok = getattr(model, "_tokenizer", None)
    if tok is None:
        if getattr(model, "family", None) == "pretrained_lm":
            return model.tokenizer
        meta = getattr(model, "meta", None) or {}
        if not meta.get("tokenizer"):
            raise NBError(f"'{getattr(model, 'name', model)}' hasn't learned any text yet.",
                          hint="Train it on a text dataset first.")
        from .tokenizers import tokenizer_from_dict
        tok = tokenizer_from_dict(meta["tokenizer"])
        model._tokenizer = tok
    return tok


def _check_lm(model):
    fam = getattr(model, "family", None)
    if fam == "pretrained_lm":
        return
    if fam != "neural" or model.net is None:
        raise NBError(f"'{getattr(model, 'name', model)}' can't write text yet.",
                      hint="Train a language model on a text dataset first.")
    if (model.meta or {}).get("task") != "language_model":
        raise NBError(f"'{model.name}' isn't a language model — it was trained to "
                      f"{'classify' if model.meta.get('task') == 'classification' else 'predict numbers'}.",
                      hint="Use 'prediction of' for this model.")


@torch.no_grad()
def sample_tokens(model, prompt_ids, length, temperature=0.8, top_k=None, on_tokens=None):
    net = model.net
    was = net.training
    net.eval()
    dev = model.device
    ctx = int((model.meta or {}).get("context") or 128)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
    out = []
    last = time.time()
    try:
        for i in range(int(length)):
            cond = idx[:, -ctx:]
            logits = net(cond)[:, -1, :].float()
            if temperature is None or temperature <= 1e-6:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / float(temperature)
                if top_k:
                    k = min(int(top_k), logits.shape[-1])
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], 1)
            out.append(int(nxt))
            if on_tokens is not None and time.time() - last > 0.15:
                last = time.time()
                on_tokens(out)
    finally:
        net.train(was)
    return out


def generated_text(model, prompt="", length=200, temperature=0.8, top_k=None, stream_id=None) -> str:
    """Generate a continuation of ``prompt`` (returns only the new text)."""
    _check_lm(model)
    prompt = "" if prompt is None else str(prompt)
    length = max(1, int(length))
    if getattr(model, "family", None) == "pretrained_lm":
        return model.generate_text(prompt, length, temperature, top_k)
    tok = tokenizer_of(model)
    ids = tok.encode(prompt) if prompt else []
    if prompt and not ids:
        emit("log", level="warn", text="None of the prompt's characters were in the model's vocabulary.")
    if not ids:
        nl = tok.encode("\n")
        ids = nl[:1] if nl else [0]
    if top_k is None and tok.vocab_size > 1000:
        top_k = 50

    def stream(out):
        if stream_id:
            emit("text_stream", id=stream_id, prompt=prompt, text=tok.decode(out))

    out = sample_tokens(model, ids, length, temperature, top_k, on_tokens=stream if stream_id else None)
    return tok.decode(out)


def generate(model, prompt="", length=200, temperature=0.8, top_k=None, _bid=None) -> str:
    """The 'write … with model' block: generate and show text."""
    sid = f"gen-{time.time_ns()}"
    t0 = time.time()
    text = generated_text(model, prompt, length, temperature, top_k, stream_id=sid)
    emit("text", id=sid, title=f"{getattr(model, 'label', model.name)} writes", prompt=str(prompt or ""),
         text=text, block=_bid, seconds=round(time.time() - t0, 2))
    return str(prompt or "") + text
