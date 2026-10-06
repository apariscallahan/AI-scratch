"""Block definition DSL.

Every NeuroBlocks block is declared exactly once, in Python, with:

* how it looks (a message like ``"dense layer %UNITS units %ACT"`` plus argument
  descriptors) — turned into Blockly JSON for the browser,
* where it fits (statement, reporter, layer, or a *setting* that only snaps into a
  particular container),
* its toolbox entry (including pre-filled child blocks),
* help text, and
* a code generator that turns the block into Python.

The browser downloads the generated Blockly definitions from the server, so the
editor, compiler, CLI and docs always agree.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


@dataclass
class Category:
    key: str
    name: str
    colour: str
    secondary: str
    tertiary: str
    icon: str = ""
    custom: str | None = None  # Blockly dynamic category ("VARIABLE", "PROCEDURE")
    order: int = 0


CATEGORIES: dict[str, Category] = {}


def category(key, name, colour, secondary, tertiary, icon="", custom=None):
    CATEGORIES[key] = Category(key, name, colour, secondary, tertiary, icon, custom, len(CATEGORIES))
    return CATEGORIES[key]


category("events", "Start", "#FFBF00", "#E6AC00", "#CC9900", "flag")
category("data", "Data", "#0FBD8C", "#0DA57A", "#0B8E69", "table")
category("neural", "Neural Nets", "#4C97FF", "#4280D7", "#3373CC", "brain")
category("layers", "Layers", "#5CB1D6", "#47A8D1", "#2E8EB8", "layers")
category("training", "Training", "#F2545B", "#E0444B", "#C9353C", "dumbbell")
category("testing", "Test & Play", "#9966FF", "#855CD6", "#774DCB", "flask")
category("sims", "Simulations", "#2EAD6B", "#27975D", "#1F7F4E", "car")
category("world", "World Parts", "#1F9E89", "#1A8875", "#147062", "gear")
category("classic", "Classic ML", "#CF63CF", "#C94FC9", "#BD42BD", "tree")
category("output", "Show & Say", "#6C7BEF", "#5B69D9", "#4C59C2", "chat")
category("control", "Control", "#FFAB19", "#EC9C13", "#CF8B17", "loop")
category("operators", "Operators", "#40BF4A", "#389438", "#2E7D32", "plus")
category("variables", "Variables", "#FF8C1A", "#FF8000", "#DB6E00", "var", custom="VARIABLE")
category("lists", "Lists", "#FF661A", "#FF5500", "#E64D00", "list")
category("functions", "My Blocks", "#FF6680", "#FF4D6A", "#FF3355", "puzzle", custom="PROCEDURE")
category("files", "Files", "#8D6E63", "#7B5E54", "#6A4F46", "folder")

# Built-in Blockly block types that belong to our categories (for styling).
BUILTIN_STYLE = {
    "logic_blocks": "operators",
    "loop_blocks": "control",
    "math_blocks": "operators",
    "text_blocks": "operators",
    "list_blocks": "lists",
    "variable_blocks": "variables",
    "variable_dynamic_blocks": "variables",
    "procedure_blocks": "functions",
}

# ---------------------------------------------------------------------------
# Argument descriptors
# ---------------------------------------------------------------------------


class Arg:
    """Something that appears in a block message as ``%NAME``."""

    is_input = False  # value/statement input (vs. field)

    def to_blockly(self, name: str) -> dict:
        raise NotImplementedError

    def toolbox_input(self) -> dict | None:
        """Shadow block for value inputs in the toolbox / examples."""
        return None


class Num(Arg):
    """A round number slot (value input with a number shadow) — accepts variables & math."""

    is_input = True

    def __init__(self, default: float = 0, *, min: float | None = None, max: float | None = None,
                 integer: bool = False):
        self.default = default
        self.min = min
        self.max = max
        self.integer = integer

    def to_blockly(self, name):
        return {"type": "input_value", "name": name, "check": "Number"}

    def toolbox_input(self, value=None):
        v = self.default if value is None else value
        return {"shadow": {"type": "math_number", "fields": {"NUM": v}}}


class Str(Arg):
    """A text slot (value input with a text shadow)."""

    is_input = True

    def __init__(self, default: str = ""):
        self.default = default

    def to_blockly(self, name):
        return {"type": "input_value", "name": name, "check": ["String", "Number"]}

    def toolbox_input(self, value=None):
        v = self.default if value is None else value
        return {"shadow": {"type": "text", "fields": {"TEXT": v}}}


class Val(Arg):
    """A value slot without a default (e.g. a condition or an expression)."""

    is_input = True

    def __init__(self, check: str | list | None = None):
        self.check = check

    def to_blockly(self, name):
        d = {"type": "input_value", "name": name}
        if self.check is not None:
            d["check"] = self.check
        return d


class Drop(Arg):
    """A dropdown menu. ``options`` is a list of (label, value) or plain strings."""

    def __init__(self, options: Iterable, default: str | None = None):
        opts = []
        for o in options:
            if isinstance(o, (tuple, list)):
                opts.append([str(o[0]), str(o[1])])
            else:
                opts.append([str(o), str(o)])
        self.options = opts
        self.default = default if default is not None else opts[0][1]
        if self.default not in [v for _, v in opts]:
            raise ValueError(f"default {self.default!r} not in options")
        # Blockly uses the first option as the initial value.
        opts.sort(key=lambda o: 0 if o[1] == self.default else 1)

    def to_blockly(self, name):
        return {"type": "field_dropdown", "name": name, "options": self.options}

    def values(self):
        return [v for _, v in self.options]


class Toggle(Arg):
    """A checkbox."""

    def __init__(self, default: bool = True):
        self.default = default

    def to_blockly(self, name):
        return {"type": "field_checkbox", "name": name, "checked": bool(self.default)}


class Text(Arg):
    """A plain text field (not pluggable)."""

    def __init__(self, default: str = "", spellcheck: bool = False):
        self.default = default
        self.spellcheck = spellcheck

    def to_blockly(self, name):
        return {"type": "field_input", "name": name, "text": self.default, "spellcheck": self.spellcheck}


class Name(Arg):
    """The name of a thing this block creates (dataset/model/world)."""

    def __init__(self, kind: str, default: str):
        self.kind = kind
        self.default = default

    def to_blockly(self, name):
        return {"type": "field_nbname", "name": name, "text": self.default, "kind": self.kind}


class Ref(Arg):
    """A dropdown that lists the names of things of ``kind`` defined elsewhere in the project."""

    def __init__(self, kind: str, default: str):
        self.kind = kind
        self.default = default

    def to_blockly(self, name):
        return {"type": "field_nbref", "name": name, "kind": self.kind, "value": self.default}


class Var(Arg):
    """A Blockly variable picker."""

    def __init__(self, default: str = "x"):
        self.default = default

    def to_blockly(self, name):
        return {"type": "field_variable", "name": name, "variable": self.default}


class Icon(Arg):
    """A small inline image (e.g. the green flag)."""

    def __init__(self, src: str, width: int = 24, height: int = 24, alt: str = ""):
        self.src = src
        self.width = width
        self.height = height
        self.alt = alt

    def to_blockly(self, name):
        return {"type": "field_image", "src": self.src, "width": self.width, "height": self.height,
                "alt": self.alt}


class Stack(Arg):
    """A C-shaped mouth that holds a stack of blocks with the given connection check."""

    is_input = True

    def __init__(self, check: str | list | None = None):
        self.check = check

    def to_blockly(self, name):
        d = {"type": "input_statement", "name": name}
        if self.check is not None:
            d["check"] = self.check
        return d


# Connection checks for the different kinds of stacks.
STMT = "Stmt"
LAYER = "Layer"

# ---------------------------------------------------------------------------
# Block specs
# ---------------------------------------------------------------------------

SHAPES = {"statement", "hat", "terminal", "value", "layer", "setting"}


@dataclass
class BlockSpec:
    type: str
    category: str
    message: str
    args: dict[str, Arg]
    shape: str = "statement"
    gen: Callable | None = None
    tooltip: str = ""
    help: str = ""
    output: Any = None  # value blocks: output check
    check: Any = None  # setting blocks: connection check (str or list)
    inline: bool = True
    prefill: dict = field(default_factory=dict)  # input -> list of child block descriptions
    toolbox: bool = True
    toolbox_variants: list = field(default_factory=list)  # extra toolbox entries: dict(field/input overrides)
    defines: tuple | None = None  # (kind, field_name)
    section: str | None = None  # toolbox label inside the category
    extra: dict = field(default_factory=dict)  # raw Blockly JSON additions
    order: int = 0

    # -- Blockly JSON --------------------------------------------------
    def to_blockly(self) -> dict:
        msg, args = _parse_message(self.message, self.args, self.type)
        d: dict[str, Any] = {
            "type": self.type,
            "message0": msg,
            "args0": args,
            "style": f"nb_{self.category}",
            "tooltip": self.tooltip,
            "inputsInline": self.inline,
        }
        if self.shape == "hat":
            d["nextStatement"] = STMT
            d["style"] = f"nb_{self.category}_hat"
        elif self.shape == "statement":
            d["previousStatement"] = STMT
            d["nextStatement"] = STMT
        elif self.shape == "terminal":
            d["previousStatement"] = STMT
        elif self.shape == "layer":
            d["previousStatement"] = LAYER
            d["nextStatement"] = LAYER
        elif self.shape == "setting":
            d["previousStatement"] = self.check
            d["nextStatement"] = self.next_check()
        elif self.shape == "value":
            d["output"] = self.output
        if self.help:
            d["helpUrl"] = ""
        d.update(self.extra)
        return d

    def next_check(self):
        """What may stack *below* a setting block: anything the containers it fits into accept.

        (Blockly checks every pair of neighbours in a stack, so e.g. an 'any world' part must be
        allowed to follow a car-only part inside a car world.)"""
        mine = set(self.check if isinstance(self.check, list) else [self.check])
        union = set(mine)
        for spec in REGISTRY.values():
            for arg in spec.args.values():
                if isinstance(arg, Stack) and arg.check:
                    accepted = set(arg.check if isinstance(arg.check, list) else [arg.check])
                    if accepted & mine:
                        union |= accepted
        return sorted(union)

    # -- toolbox / example JSON ------------------------------------------
    def instance(self, values: dict | None = None, *, shadow_only=False) -> dict:
        """Serialized-block JSON (Blockly's format) with defaults & shadows filled in."""
        return make_block(self.type, **(values or {}))


REGISTRY: dict[str, BlockSpec] = {}
_counter = [0]


def block(type: str, category: str, message: str, *, shape: str = "statement", **kw):
    """Decorator registering a block whose code generator is the decorated function.

    Keyword arguments that are :class:`Arg` instances become the block's arguments;
    the rest configure the :class:`BlockSpec`.
    """
    args = {k: v for k, v in kw.items() if isinstance(v, Arg)}
    opts = {k: v for k, v in kw.items() if not isinstance(v, Arg)}
    if shape not in SHAPES:
        raise ValueError(shape)

    def deco(fn):
        _counter[0] += 1
        spec = BlockSpec(type=type, category=category, message=message, args=args, shape=shape,
                         gen=fn, order=_counter[0], **opts)
        if type in REGISTRY:
            raise ValueError(f"duplicate block {type}")
        # Validate message placeholders.
        names = re.findall(r"%([A-Z_][A-Z0-9_]*)", message)
        missing = set(names) - set(args)
        unused = set(args) - set(names)
        if missing or unused:
            raise ValueError(f"{type}: placeholders {missing or ''} missing args / unused args {unused or ''}")
        REGISTRY[type] = spec
        return fn

    return deco


def _parse_message(message: str, args: dict[str, Arg], btype: str):
    """Turn ``"train %MODEL on %DATA\\n..."`` into Blockly's ``message0``/``args0``."""
    out_args: list[dict] = []
    pieces: list[str] = []
    lines = message.split("\n")
    for li, line in enumerate(lines):
        line = line.strip()

        def repl(m):
            name = m.group(1)
            prefix = ""
            if isinstance(args[name], Stack) and line[: m.start()].strip():
                # Put the label on its own row so the C-shaped mouth opens underneath it
                # (otherwise Blockly indents the mouth by the label's width).
                out_args.append({"type": "input_end_row", "name": f"_LBL{len(out_args)}"})
                prefix = f"%{len(out_args)} "
            out_args.append(args[name].to_blockly(name))
            return f"{prefix}%{len(out_args)}"

        text = re.sub(r"%([A-Z_][A-Z0-9_]*)", repl, line)
        pieces.append(text)
        if li < len(lines) - 1:
            # Explicit line break: end the current row.
            out_args.append({"type": "input_end_row", "name": f"_ROW{li}"})
            pieces.append(f"%{len(out_args)}")
    return " ".join(p for p in pieces if p), out_args


# ---------------------------------------------------------------------------
# Building serialized block JSON (for the toolbox and for example projects)
# ---------------------------------------------------------------------------


class B:
    """A lightweight description of a block instance, used for toolbox pre-fills and examples.

    ``B("nb_dense", UNITS=128, ACT="relu")`` — keyword values go to fields or (as
    shadow values) to inputs. Values may be nested ``B`` objects (plugged into value
    inputs) or lists of ``B`` (stacks for statement inputs).
    """

    def __init__(self, type: str, _id: str | None = None, _extra: dict | None = None, **values):
        self.type = type
        self.values = values
        self.id = _id
        self.extra = _extra or {}

    def json(self) -> dict:
        return make_block(self.type, _id=self.id, _extra=self.extra, **self.values)


# Built-in block shapes we need when building JSON (value input names etc.).
_BUILTIN_VALUE = {
    "math_number": {"fields": {"NUM": 0}},
    "text": {"fields": {"TEXT": ""}},
    "logic_boolean": {"fields": {"BOOL": "TRUE"}},
}


def _stack_json(items):
    items = [i for i in items if i is not None]
    if not items:
        return None
    head = None
    prev = None
    for it in items:
        j = it.json() if isinstance(it, B) else it
        if head is None:
            head = j
        else:
            prev["next"] = {"block": j}
        prev = j
        # walk to the end of an already-chained block
        while "next" in prev:
            prev = prev["next"]["block"]
    return head


def _as_b(p) -> "B":
    """Prefill entries: "type", ("type", {values}) or a B instance."""
    if isinstance(p, B):
        return p
    if isinstance(p, tuple):
        t, vals = p[0], (p[1] if len(p) > 1 else {})
        return B(t, **vals)
    return B(p)


def literal_block(value):
    """A reporter block holding a Python literal."""
    if isinstance(value, bool):
        return {"type": "logic_boolean", "fields": {"BOOL": "TRUE" if value else "FALSE"}}
    if isinstance(value, (int, float)):
        return {"type": "math_number", "fields": {"NUM": value}}
    return {"type": "text", "fields": {"TEXT": str(value)}}


def make_block(type: str, _id: str | None = None, _extra: dict | None = None, **values) -> dict:
    """Create Blockly serialization JSON for a block with defaults filled in."""
    j: dict[str, Any] = {"type": type}
    if _id:
        j["id"] = _id
    spec = REGISTRY.get(type)
    fields: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    if spec is not None:
        for name, arg in spec.args.items():
            given = values.pop(name, None) if name in values else None
            if isinstance(arg, (Num, Str)):
                inp = arg.toolbox_input(None)
                if given is not None:
                    if isinstance(given, B):
                        inp["block"] = given.json()
                    elif isinstance(given, dict):
                        inp["block"] = given
                    else:
                        inp = arg.toolbox_input(given)
                inputs[name] = inp
            elif isinstance(arg, Val):
                if given is not None:
                    inputs[name] = {"block": given.json() if isinstance(given, B) else
                                    (given if isinstance(given, dict) else literal_block(given))}
            elif isinstance(arg, Stack):
                items = given if given is not None else [_as_b(p) for p in spec.prefill.get(name, [])]
                stack = _stack_json(items if isinstance(items, list) else [items])
                if stack:
                    inputs[name] = {"block": stack}
            elif isinstance(arg, Drop):
                v = arg.default if given is None else str(given)
                if v not in arg.values():
                    raise ValueError(f"{type}.{name}: {v!r} not in {arg.values()}")
                fields[name] = v
            elif isinstance(arg, Toggle):
                fields[name] = arg.default if given is None else bool(given)
            elif isinstance(arg, (Text, Name, Ref)):
                fields[name] = arg.default if given is None else str(given)
            elif isinstance(arg, Var):
                # Variables are referenced by name; Blockly creates them on load.
                if isinstance(given, dict):
                    fields[name] = given
                else:
                    fields[name] = {"name": arg.default if given is None else str(given)}
    else:
        # Built-in Blockly block: values go straight through.
        base = _BUILTIN_VALUE.get(type, {})
        fields.update(base.get("fields", {}))
    # Remaining values for built-in blocks or explicit raw values.
    for name, given in values.items():
        if isinstance(given, list):
            stack = _stack_json(given)
            if stack:
                inputs[name] = {"block": stack}
        elif isinstance(given, B):
            inputs[name] = {"block": given.json()}
        elif isinstance(given, dict) and "type" in given:
            inputs[name] = {"block": given}
        elif isinstance(given, dict) and ("shadow" in given or "block" in given):
            inputs[name] = given
        else:
            fields[name] = given
    if fields:
        j["fields"] = fields
    if inputs:
        j["inputs"] = inputs
    if _extra:
        j.update(_extra)
    return j
