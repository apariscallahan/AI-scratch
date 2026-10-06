"""Check a workspace against Blockly's connection rules (without a browser).

Blockly refuses to load a stack where two neighbouring blocks (or a block and the slot it
sits in) have incompatible connection checks. This mirrors that rule so tests can verify
every example project — and saved projects can be diagnosed — from Python.
"""
from __future__ import annotations

from .spec import REGISTRY, Num, Stack, Str, Val

_NUMBER = ["Number"]
_STRING = ["String"]
_BOOL = ["Boolean"]
_ARRAY = ["Array"]

BUILTIN_OUTPUT = {
    "math_number": _NUMBER, "math_arithmetic": _NUMBER, "math_single": _NUMBER, "math_round": _NUMBER,
    "math_modulo": _NUMBER, "math_random_int": _NUMBER, "math_random_float": _NUMBER, "math_on_list": _NUMBER,
    "math_constant": _NUMBER, "math_trig": _NUMBER, "math_constrain": _NUMBER, "math_atan2": _NUMBER,
    "text": _STRING, "text_join": _STRING, "text_length": _NUMBER, "text_changeCase": _STRING, "text_trim": _STRING,
    "logic_compare": _BOOL, "logic_operation": _BOOL, "logic_negate": _BOOL, "logic_boolean": _BOOL,
    "math_number_property": _BOOL, "lists_isEmpty": _BOOL, "text_isEmpty": _BOOL,
    "lists_create_with": _ARRAY, "lists_create_empty": _ARRAY, "lists_repeat": _ARRAY, "lists_sort": _ARRAY,
    "lists_reverse": _ARRAY, "lists_length": _NUMBER, "lists_indexOf": _NUMBER,
}
BUILTIN_INPUTS = {
    "math_arithmetic": {"A": _NUMBER, "B": _NUMBER}, "math_single": {"NUM": _NUMBER}, "math_round": {"NUM": _NUMBER},
    "math_modulo": {"DIVIDEND": _NUMBER, "DIVISOR": _NUMBER}, "math_random_int": {"FROM": _NUMBER, "TO": _NUMBER},
    "logic_operation": {"A": _BOOL, "B": _BOOL}, "logic_negate": {"BOOL": _BOOL},
    "controls_repeat_ext": {"TIMES": _NUMBER}, "controls_whileUntil": {"BOOL": _BOOL},
    "controls_for": {"FROM": _NUMBER, "TO": _NUMBER, "BY": _NUMBER}, "math_on_list": {"LIST": _ARRAY},
    "controls_forEach": {"LIST": _ARRAY},
}


def _norm(check):
    if check is None:
        return None
    return list(check) if isinstance(check, (list, tuple)) else [check]


def _compatible(a, b) -> bool:
    a, b = _norm(a), _norm(b)
    if a is None or b is None:
        return True
    return any(x in a for x in b)


def _prev_check(t):
    spec = REGISTRY.get(t)
    if spec is None:
        return None
    return _norm(spec.to_blockly().get("previousStatement"))


def _next_check(t):
    spec = REGISTRY.get(t)
    if spec is None:
        return None
    return _norm(spec.to_blockly().get("nextStatement"))


def _output_check(t):
    spec = REGISTRY.get(t)
    if spec is None:
        return BUILTIN_OUTPUT.get(t)
    return _norm(spec.output)


def _input_check(t, name):
    spec = REGISTRY.get(t)
    if spec is None:
        if t == "controls_if" and name.startswith("IF"):
            return _BOOL
        return (BUILTIN_INPUTS.get(t) or {}).get(name)
    arg = spec.args.get(name)
    if isinstance(arg, Num):
        return _NUMBER
    if isinstance(arg, Str):
        return ["String", "Number"]
    if isinstance(arg, (Val, Stack)):
        return _norm(arg.check)
    return None


def _is_statement_input(t, name):
    spec = REGISTRY.get(t)
    if spec is not None:
        return isinstance(spec.args.get(name), Stack)
    return name.startswith(("DO", "ELSE", "STACK"))


def check_connections(workspace: dict) -> list[str]:
    """Return a list of problems (empty if Blockly would load the workspace)."""
    problems: list[str] = []

    def walk(b: dict):
        t = b.get("type")
        for name, inp in (b.get("inputs") or {}).items():
            for key in ("shadow", "block"):
                child = inp.get(key)
                if not child:
                    continue
                ct = child.get("type")
                if _is_statement_input(t, name):
                    if not _compatible(_input_check(t, name), _prev_check(ct)):
                        problems.append(f"{ct} can't go inside {t}.{name}")
                else:
                    if not _compatible(_input_check(t, name), _output_check(ct)):
                        problems.append(f"{ct} can't plug into {t}.{name}")
                walk(child)
        nxt = (b.get("next") or {}).get("block")
        if nxt:
            if not _compatible(_next_check(t), _prev_check(nxt.get("type"))):
                problems.append(f"{nxt.get('type')} can't stack under {t}")
            walk(nxt)

    for top in ((workspace or {}).get("blocks") or {}).get("blocks") or []:
        walk(top)
    return problems
