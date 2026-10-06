"""The ``nb`` runtime library used by programs generated from blocks.

Heavy modules (PyTorch, scikit-learn, physics) load lazily on first use.
"""
from __future__ import annotations

import importlib

from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .errors import NBError, StopProgram  # noqa: F401

_LAZY = {
    # models
    "Model": "model", "GPT": "model", "show_model": "model", "param_count": "model", "reset": "model",
    # training
    "train": "train", "metric_of": "train",
    # testing & evaluation
    "evaluate": "evaluate", "score": "evaluate", "show_predictions": "evaluate",
    "show_confusion": "evaluate", "show_decision_map": "evaluate", "predict": "evaluate",
    "show_examples": "evaluate", "plot_data": "evaluate", "show_map": "evaluate",
    # text generation
    "generate": "generate", "generated_text": "generate",
    # files
    "save": "io", "load_model": "io",
    # interactive testing
    "play_chat": "interactive", "play_draw": "interactive", "play_form": "interactive",
    # pretrained models
    "pretrained_lm": "pretrained",
    # picture generators
    "generator": "generative", "show_generated": "generative",
}

__all__ = list(_core_all) + list(_LAZY) + ["NBError", "StopProgram"]


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'neuroblocks.runtime' has no attribute {name!r}")
    return getattr(importlib.import_module(f"neuroblocks.runtime.{mod}"), name)
