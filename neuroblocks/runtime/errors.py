"""Friendly errors: messages written for people who are learning."""
from __future__ import annotations

import re
import traceback


class NBError(Exception):
    """A mistake in the block program, explained in plain words."""

    def __init__(self, message: str, hint: str | None = None, block_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.nb_block = block_id


class StopProgram(Exception):
    """Raised by the 'stop the program' block."""


class SkipTraining(Exception):
    """Internal: the user asked to finish the current training loop early."""


def missing_package(package: str, feature: str, pip_name: str | None = None) -> NBError:
    return NBError(
        f"{feature} needs the optional Python package '{package}', which isn't installed.",
        hint=f"Install it with:  pip install {pip_name or package}",
    )


def explain(exc: BaseException, names: dict | None = None) -> tuple[str, str | None]:
    """Turn any exception into (message, hint)."""
    names = names or {}
    if isinstance(exc, NBError):
        return exc.message, exc.hint
    msg = str(exc)
    if isinstance(exc, NameError):
        m = re.search(r"name '([^']+)' is not defined", msg)
        if m and m.group(1) in names:
            info = names[m.group(1)]
            kind = info.get("kind")
            if kind == "variable":
                return (f"The variable '{info['name']}' is used before it has a value.",
                        "Use a 'set variable' block first.")
            return (f"The {kind} '{info['name']}' is used before it is created.",
                    f"Move the block that creates '{info['name']}' above the blocks that use it.")
    if isinstance(exc, ModuleNotFoundError):
        mod = getattr(exc, "name", None) or msg
        return (f"A Python package is missing: {mod}", f"Install it with:  pip install {mod}")
    if "out of memory" in msg.lower() and ("cuda" in msg.lower() or "mps" in msg.lower()):
        return ("The GPU ran out of memory.",
                "Try a smaller batch size, a shorter context length, or a smaller model.")
    if "mat1 and mat2 shapes cannot be multiplied" in msg:
        return ("Two layers don't fit together (their sizes don't match).",
                "Check the order of your layers — for images, put a 'flatten' before dense layers.")
    if isinstance(exc, ZeroDivisionError):
        return ("Something was divided by zero.", None)
    if isinstance(exc, KeyboardInterrupt):
        return ("Stopped.", None)
    if isinstance(exc, MemoryError):
        return ("The computer ran out of memory.", "Use fewer examples, a smaller model or a smaller batch size.")
    if isinstance(exc, FileNotFoundError):
        return (f"File not found: {getattr(exc, 'filename', None) or msg}",
                "Put data files in the 'data' folder (the editor's Data > Upload button does this).")
    return (f"{type(exc).__name__}: {msg}", None)


def short_traceback(exc: BaseException, limit: int = 12) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit))
