"""Helpers shared by the block generators."""
from __future__ import annotations

from .core import INDENT, BlockCtx, Setting

NICE_KEYS = {
    "test": "test split", "samples": "number of examples", "batch_size": "batch size", "lr": "learning rate",
    "weight_decay": "weight decay", "save_best": "save best", "eval_every": "check progress",
}


def settings_kwargs(b: BlockCtx, input_name: str, *, allowed: set | None = None,
                    context: str = "this block") -> tuple[list[str], list[str]]:
    """Collect the setting blocks inside a container into keyword arguments.

    Returns (["key=value", ...], pre_lines).
    """
    seen: dict[str, tuple[str, str | None]] = {}
    pre: list[str] = []
    every: list[str] = []
    for s in b.settings(input_name):
        pre.extend(s.pre)
        if s.key == "every":
            every.append(s.code)
            continue
        if s.key.startswith("_"):
            continue
        if allowed is not None and s.key not in allowed:
            b.c.warn(f"The '{NICE_KEYS.get(s.key, s.key.replace('_', ' '))}' setting doesn't apply to "
                     f"{context}, so it will be ignored.", s.block_id)
            continue
        if s.key in seen:
            b.c.warn(f"There are two '{NICE_KEYS.get(s.key, s.key.replace('_', ' '))}' settings here; the "
                     f"lower one wins.", s.block_id)
        seen[s.key] = (s.code, s.block_id)
    kws = [f"{k}={v}" for k, (v, _) in seen.items()]
    if every:
        kws.append(f"every=[{', '.join(every)}]")
    return kws, pre


def fmt_call(fn: str, args: list[str], width: int = 92) -> str:
    """``fn(a, b, c)`` on one line if short, otherwise one argument per line."""
    args = [a for a in args if a]
    one = f"{fn}({', '.join(args)})"
    if len(one) <= width and "\n" not in one:
        return one
    # Hang a trailing multi-line list:  fn(a, b, [\n    ...\n])
    if args and args[-1].startswith("[") and "\n" in args[-1] and all("\n" not in a for a in args[:-1]):
        head = f"{fn}({', '.join(args[:-1] + ['['])}"
        rest = args[-1].split("\n")[1:]
        if len(head) <= width:
            return "\n".join([head] + rest[:-1] + [rest[-1] + ")"])
    lines = [f"{fn}("]
    for a in args:
        sub = a.split("\n")
        lines.append(INDENT + sub[0])
        lines.extend(INDENT + s for s in sub[1:])
        lines[-1] += ","
    lines.append(")")
    return "\n".join(lines)


def fmt_list(items: list[str]) -> str:
    """A Python list literal with one item per line (items may be multi-line)."""
    if not items:
        return "[]"
    lines = ["["]
    for it in items:
        sub = it.split("\n")
        lines.append(INDENT + sub[0])
        lines.extend(INDENT + s for s in sub[1:])
        lines[-1] += ","
    lines.append("]")
    return "\n".join(lines)


def assign(target: str | None, expr: str) -> list[str]:
    return (f"{target} = {expr}" if target else expr).split("\n")


def setting(key: str, code: str, **extra) -> Setting:
    return Setting(key=key, code=code, extra=extra)


def percent(b: BlockCtx, name: str) -> str:
    """A 0–100 percentage slot as a 0–1 fraction in code."""
    lit = b.literal(name)
    if isinstance(lit, float):
        return repr(round(lit / 100.0, 6))
    from .core import ORDER_MULTIPLICATIVE
    return f"{b.val(name, ORDER_MULTIPLICATIVE)} / 100"


def int_code(b: BlockCtx, name: str) -> str:
    lit = b.literal(name)
    if isinstance(lit, float):
        return str(int(round(lit)))
    return f"int({b.val(name)})"
