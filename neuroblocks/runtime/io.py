"""Saving and loading models (weights + everything needed to use them again)."""
from __future__ import annotations

import torch

from .. import paths
from .errors import NBError
from .events import emit

__all__ = ["save", "load_model"]


def save(obj, path="model.pt", quiet=False, _bid=None):
    """Save a trained model (neural, classic, pretrained or simulation agent) to a file."""
    fam = getattr(obj, "family", None)
    if fam is None or not hasattr(obj, "to_payload"):
        raise NBError("Only models can be saved with this block.", block_id=_bid)
    p = paths.resolve_model_file(str(path), for_writing=True)
    if not p.suffix:
        p = p.with_suffix(".pt")
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = obj.to_payload(p)  if fam == "pretrained_lm" else obj.to_payload()
    if fam == "neural" and getattr(obj, "agent", None) is not None:
        payload["agent"] = obj.agent.to_payload()
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
    if not quiet:
        size = p.stat().st_size
        emit("log", level="success", text=f"Saved {obj.describe()} to {p} ({size / 1e6:.1f} MB).")
        emit("file", path=str(p), kind="model", name=getattr(obj, "name", ""))
    return str(p)


def load_model(path="model.pt", name="model", _bid=None):
    """Load a model saved with 'save … to file'."""
    p = paths.resolve_model_file(str(path))
    if not p.exists() and not p.suffix:
        p = p.with_suffix(".pt")
    if not p.exists():
        raise NBError(f"Model file not found: {p}",
                      hint="Saved models live in the 'models' folder of your NeuroBlocks home.", block_id=_bid)
    # Model files contain Python objects (tokenizers, scikit-learn models) — only load files you trust.
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "neuroblocks-model":
        raise NBError(f"{p.name} isn't a NeuroBlocks model file.", block_id=_bid)
    fam = payload.get("family")
    if fam == "neural":
        from .model import Model
        m = Model.from_payload(payload, name)
        if payload.get("agent"):
            from .rl.agent import Agent
            m.agent = Agent.from_payload(payload["agent"], m)
    elif fam == "classic":
        from .classic import ClassicModel
        m = ClassicModel.from_payload(payload, name)
    elif fam == "pretrained_lm":
        from .pretrained import PretrainedLM
        m = PretrainedLM.from_payload(payload, name)
    elif fam == "generator":
        from .generative import ImageGenerator
        m = ImageGenerator.from_payload(payload, name)
    else:
        raise NBError(f"Unknown model type in {p.name}: {fam}", block_id=_bid)
    m.bid = _bid
    emit("log", level="success", text=f"Loaded {m.describe()} from {p.name}.")
    return m
