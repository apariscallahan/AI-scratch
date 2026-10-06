"""Block definitions and the block-program → Python compiler."""
from .core import CompileResult, Diagnostic, compile_workspace  # noqa: F401
from .spec import B, make_block  # noqa: F401
from . import blocks  # noqa: F401,E402  (populate the block registry)
