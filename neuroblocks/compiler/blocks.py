"""Import every block module so the registry is complete."""
from . import blocks_core, blocks_data, blocks_model, blocks_train, blocks_test  # noqa: F401

try:  # simulations depend on nothing heavy at compile time, but keep the import guarded
    from . import blocks_rl  # noqa: F401
except ImportError:  # pragma: no cover
    pass
