"""NeuroBlocks — a Scratch-style visual block language for AI and machine learning.

Programs built from blocks compile to ordinary Python that uses this package as
its runtime library::

    import neuroblocks as nb

    data = nb.data.toy("spirals", name="data")
    model = nb.Model("model", [nb.layers.Dense(32), nb.layers.Output()])
    nb.train(model, data, epochs=50)

The runtime (and therefore PyTorch) is only imported when one of its names is
first used, so the editor server and the compiler start quickly.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

# Submodules that generated code accesses as attributes (nb.layers.Dense, ...).
_SUBMODULES = {
    "layers": "neuroblocks.runtime.layers",
    "data": "neuroblocks.runtime.data",
    "classic": "neuroblocks.runtime.classic",
    "rl": "neuroblocks.runtime.rl",
}


_OWN_MODULES = {"paths", "project", "cli", "compiler", "server", "runtime", "__main__"}


def __getattr__(name: str):
    # `from . import paths` asks the package for the attribute first; don't import the
    # runtime as a side effect (it reads settings from the environment when imported).
    if name.startswith("__") or name in _OWN_MODULES:
        raise AttributeError(name)
    if name in _SUBMODULES:
        mod = importlib.import_module(_SUBMODULES[name])
        globals()[name] = mod
        return mod
    runtime = importlib.import_module("neuroblocks.runtime")
    # Resolve through the explicit name map: a submodule with the same name as a function
    # (e.g. runtime/evaluate.py vs nb.evaluate) would otherwise shadow it once imported.
    lazy = getattr(runtime, "_LAZY", {})
    core = importlib.import_module("neuroblocks.runtime.core")
    try:
        if name in lazy:
            value = getattr(importlib.import_module(f"neuroblocks.runtime.{lazy[name]}"), name)
        elif name in getattr(core, "__all__", ()):
            value = getattr(core, name)
        else:
            value = getattr(runtime, name)
    except AttributeError:
        raise AttributeError(f"module 'neuroblocks' has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    runtime = importlib.import_module("neuroblocks.runtime")
    return sorted(set(globals()) | set(_SUBMODULES) | set(getattr(runtime, "__all__", [])))
