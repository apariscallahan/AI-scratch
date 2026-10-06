"""Build the Blockly toolbox and the block definitions the editor downloads."""
from __future__ import annotations

from . import blocks  # noqa: F401  (populates the registry)
from .spec import BUILTIN_STYLE, CATEGORIES, REGISTRY, B, Name, Ref, make_block


def _num(v):
    return {"shadow": {"type": "math_number", "fields": {"NUM": v}}}


def _txt(v):
    return {"shadow": {"type": "text", "fields": {"TEXT": v}}}


# Built-in Blockly blocks shown in our categories (Blockly defines how they look).
BUILTIN_TOOLBOX = {
    "control": [
        {"type": "controls_repeat_ext", "inputs": {"TIMES": _num(10)}},
        {"type": "controls_for", "fields": {"VAR": {"name": "i"}},
         "inputs": {"FROM": _num(1), "TO": _num(10), "BY": _num(1)}},
        {"type": "controls_forEach", "fields": {"VAR": {"name": "item"}}},
        {"type": "controls_whileUntil"},
        {"type": "controls_if"},
        {"type": "controls_if", "extraState": {"hasElse": True}},
        {"type": "controls_flow_statements"},
    ],
    "operators": [
        {"type": "math_arithmetic", "fields": {"OP": "ADD"}, "inputs": {"A": _num(1), "B": _num(1)}},
        {"type": "math_arithmetic", "fields": {"OP": "MINUS"}, "inputs": {"A": _num(1), "B": _num(1)}},
        {"type": "math_arithmetic", "fields": {"OP": "MULTIPLY"}, "inputs": {"A": _num(1), "B": _num(1)}},
        {"type": "math_arithmetic", "fields": {"OP": "DIVIDE"}, "inputs": {"A": _num(1), "B": _num(1)}},
        {"type": "math_arithmetic", "fields": {"OP": "POWER"}, "inputs": {"A": _num(10), "B": _num(-3)}},
        {"type": "math_random_int", "inputs": {"FROM": _num(1), "TO": _num(10)}},
        {"type": "math_random_float"},
        {"type": "logic_compare", "fields": {"OP": "GT"}, "inputs": {"A": _num(0), "B": _num(50)}},
        {"type": "logic_compare", "fields": {"OP": "LT"}, "inputs": {"A": _num(0), "B": _num(50)}},
        {"type": "logic_compare", "fields": {"OP": "EQ"}, "inputs": {"A": _num(0), "B": _num(50)}},
        {"type": "logic_operation", "fields": {"OP": "AND"}},
        {"type": "logic_operation", "fields": {"OP": "OR"}},
        {"type": "logic_negate"},
        {"type": "logic_boolean"},
        {"type": "text_join", "extraState": {"itemCount": 2},
         "inputs": {"ADD0": _txt("accuracy: "), "ADD1": _txt("")}},
        {"type": "text_length", "inputs": {"VALUE": _txt("hello")}},
        {"type": "math_modulo", "inputs": {"DIVIDEND": _num(10), "DIVISOR": _num(3)}},
        {"type": "math_round", "inputs": {"NUM": _num(3.7)}},
        {"type": "math_single", "fields": {"OP": "ROOT"}, "inputs": {"NUM": _num(9)}},
        {"type": "math_single", "fields": {"OP": "ABS"}, "inputs": {"NUM": _num(-5)}},
        {"type": "math_number", "fields": {"NUM": 0}},
        {"type": "text", "fields": {"TEXT": ""}},
    ],
    "lists": [
        {"type": "lists_create_with", "extraState": {"itemCount": 3},
         "inputs": {"ADD0": _num(0.1), "ADD1": _num(0.01), "ADD2": _num(0.001)}},
        {"type": "lists_create_empty"},
        {"type": "lists_getIndex", "fields": {"MODE": "GET", "WHERE": "FROM_START"}, "inputs": {"AT": _num(1)}},
        {"type": "lists_length"},
        {"type": "math_on_list", "fields": {"OP": "AVERAGE"}},
        {"type": "math_on_list", "fields": {"OP": "MAX"}},
        {"type": "lists_sort"},
    ],
    "output": [
        {"type": "text_print", "inputs": {"TEXT": _txt("Hello!")}},
    ],
}

# Where in the category the built-in entries go: "before" or "after" our own blocks.
BUILTIN_POSITION = {"control": "before", "operators": "before", "lists": "before", "output": "after"}


def _entry(spec) -> dict:
    j = make_block(spec.type)
    j["kind"] = "block"
    return j


def build_toolbox() -> dict:
    contents = []
    for cat in sorted(CATEGORIES.values(), key=lambda c: c.order):
        entry = {"kind": "category", "name": cat.name, "colour": cat.colour, "toolboxitemid": cat.key,
                 "cssconfig": {"container": f"nb-cat nb-cat-{cat.key}"}}
        if cat.key == "variables":
            entry["custom"] = "NB_VARIABLES"
            contents.append(entry)
            continue
        if cat.custom:
            entry["custom"] = cat.custom
            contents.append(entry)
            continue
        items: list[dict] = []
        builtin = [dict(kind="block", **b) for b in BUILTIN_TOOLBOX.get(cat.key, [])]
        if BUILTIN_POSITION.get(cat.key) == "before":
            items.extend(builtin)
        section = None
        for spec in sorted((s for s in REGISTRY.values() if s.category == cat.key and s.toolbox),
                           key=lambda s: s.order):
            if spec.section and spec.section != section:
                section = spec.section
                items.append({"kind": "label", "text": section, "web-class": "nb-flyout-label"})
            items.append(_entry(spec))
        if BUILTIN_POSITION.get(cat.key) == "after":
            items.extend(builtin)
        if not items:
            continue
        entry["contents"] = items
        contents.append(entry)
    return {"kind": "categoryToolbox", "contents": contents}


def variables_extra() -> list[dict]:
    """Our blocks added to the dynamic Variables flyout."""
    out = []
    for spec in REGISTRY.values():
        if spec.category == "variables" and spec.toolbox:
            out.append(_entry(spec))
    return out


def editor_bundle() -> dict:
    """Everything the browser needs: block JSON, toolbox, categories, names metadata, help."""
    defs = [spec.to_blockly() for spec in sorted(REGISTRY.values(), key=lambda s: s.order)]
    definers = {}
    refs = {}
    for spec in REGISTRY.values():
        if spec.defines:
            definers[spec.type] = {"kind": spec.defines[0], "field": spec.defines[1]}
        for name, arg in spec.args.items():
            if isinstance(arg, Ref):
                refs.setdefault(spec.type, {})[name] = arg.kind
    cats = [{"key": c.key, "name": c.name, "colour": c.colour, "secondary": c.secondary, "tertiary": c.tertiary,
             "icon": c.icon} for c in sorted(CATEGORIES.values(), key=lambda c: c.order)]
    help_ = {s.type: {"tooltip": s.tooltip, "help": s.help, "category": s.category} for s in REGISTRY.values()}
    builtin_styles = {style: CATEGORIES[key].colour for style, key in BUILTIN_STYLE.items()}
    return {"blocks": defs, "toolbox": build_toolbox(), "categories": cats, "definers": definers, "refs": refs,
            "help": help_, "builtin_styles": builtin_styles, "variables_extra": variables_extra()}
