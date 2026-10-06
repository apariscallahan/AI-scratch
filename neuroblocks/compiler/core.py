"""Compile a Blockly workspace (JSON) into a readable Python program."""
from __future__ import annotations

import datetime as _dt
import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .spec import REGISTRY, BlockSpec, Drop, Name, Num, Ref, Stack, Str, Toggle, Val, Var, Text

# Operator precedence (same scheme as Blockly's Python generator).
ORDER_ATOMIC = 0
ORDER_COLLECTION = 1
ORDER_MEMBER = 2.1
ORDER_FUNCTION_CALL = 2.2
ORDER_EXPONENTIATION = 3
ORDER_UNARY_SIGN = 4
ORDER_MULTIPLICATIVE = 5
ORDER_ADDITIVE = 6
ORDER_RELATIONAL = 11
ORDER_LOGICAL_NOT = 12
ORDER_LOGICAL_AND = 13
ORDER_LOGICAL_OR = 14
ORDER_CONDITIONAL = 15
ORDER_LAMBDA = 16
ORDER_NONE = 99

INDENT = "    "

RESERVED = set(keyword.kwlist) | {
    "nb", "math", "random", "time", "print", "len", "list", "dict", "set", "str", "int", "float",
    "bool", "range", "sum", "min", "max", "abs", "round", "type", "id", "input", "open", "map",
    "filter", "zip", "any", "all", "sorted", "reversed", "iter", "next", "object", "super", "format",
    "self", "None", "True", "False", "main", "torch", "np", "numpy",
}

# Generators for built-in Blockly blocks are registered here by blocks_core.
BUILTIN_STATEMENTS: dict[str, Callable] = {}
BUILTIN_VALUES: dict[str, Callable] = {}


def builtin_statement(*types):
    def deco(fn):
        for t in types:
            BUILTIN_STATEMENTS[t] = fn
        return fn
    return deco


def builtin_value(*types):
    def deco(fn):
        for t in types:
            BUILTIN_VALUES[t] = fn
        return fn
    return deco


class CompileError(Exception):
    def __init__(self, message: str, block_id: str | None = None):
        super().__init__(message)
        self.block_id = block_id


@dataclass
class Diagnostic:
    level: str  # "error" | "warning" | "info"
    message: str
    block_id: str | None = None

    def to_json(self):
        return {"level": self.level, "message": self.message, "block_id": self.block_id}


@dataclass
class CompileResult:
    code: str
    diagnostics: list[Diagnostic]
    names: dict[str, dict]  # python identifier -> {"kind":..., "name":...}

    @property
    def ok(self) -> bool:
        return not any(d.level == "error" for d in self.diagnostics)

    @property
    def errors(self):
        return [d for d in self.diagnostics if d.level == "error"]

    def to_json(self):
        return {"code": self.code, "ok": self.ok,
                "diagnostics": [d.to_json() for d in self.diagnostics], "names": self.names}


@dataclass
class Setting:
    """What a *setting* block contributes to its container."""
    key: str
    code: str | None = None
    pre: list[str] = field(default_factory=list)  # lines emitted before the container statement
    block_id: str | None = None
    extra: dict = field(default_factory=dict)


def pyrepr(value: Any) -> str:
    """Python literal for a JSON value."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def number_literal(raw: Any) -> str:
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return "0"
    if f != f:  # NaN
        return "float('nan')"
    if f in (float("inf"), float("-inf")):
        return "float('inf')" if f > 0 else "-float('inf')"
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def sanitize(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unnamed"
    if s[0].isdigit():
        s = "_" + s
    return s


class BlockCtx:
    """What a code generator sees: convenient access to one block's fields and inputs."""

    def __init__(self, compiler: "Compiler", data: dict):
        self.c = compiler
        self.data = data
        self.id: str | None = data.get("id")
        self.type: str = data["type"]
        self.spec: BlockSpec | None = REGISTRY.get(self.type)
        self.extra = data.get("extraState") or {}

    # -- fields ---------------------------------------------------------
    def field(self, name: str, default: Any = None) -> Any:
        fields = self.data.get("fields") or {}
        if name in fields:
            v = fields[name]
            return v
        if self.spec and name in self.spec.args:
            arg = self.spec.args[name]
            return getattr(arg, "default", default)
        return default

    def q(self, name: str) -> str:
        """Field value as a Python string literal."""
        return repr(str(self.field(name, "")))

    def flag(self, name: str) -> bool:
        v = self.field(name, False)
        if isinstance(v, str):
            return v.upper() == "TRUE"
        return bool(v)

    def var(self, name: str = "VAR") -> str:
        """Python identifier of a Blockly variable field."""
        return self.c.var_ident(self.field(name))

    # -- inputs ---------------------------------------------------------
    def input_block(self, name: str) -> dict | None:
        inputs = self.data.get("inputs") or {}
        inp = inputs.get(name)
        if not inp:
            return None
        return inp.get("block") or inp.get("shadow")

    def has_input(self, name: str) -> bool:
        return self.input_block(name) is not None

    def val(self, name: str, order: float = ORDER_NONE, default: str | None = None) -> str:
        blk = self.input_block(name)
        if blk is None:
            if default is not None:
                return default
            if self.spec and name in self.spec.args:
                arg = self.spec.args[name]
                if isinstance(arg, Num):
                    return number_literal(arg.default)
                if isinstance(arg, Str):
                    return repr(arg.default)
            self.c.error(f"The '{name.lower()}' slot of this block is empty.", self.id)
            return "None"
        return self.c.value_to_code(blk, order)

    def num(self, name: str, order: float = ORDER_NONE) -> str:
        return self.val(name, order)

    def is_literal(self, name: str) -> bool:
        blk = self.input_block(name)
        return blk is not None and blk.get("type") in ("math_number", "text", "logic_boolean")

    def literal(self, name: str):
        blk = self.input_block(name)
        if blk is None:
            return None
        if blk.get("type") == "math_number":
            try:
                return float((blk.get("fields") or {}).get("NUM", 0))
            except (TypeError, ValueError):
                return 0.0
        if blk.get("type") == "text":
            return (blk.get("fields") or {}).get("TEXT", "")
        return None

    def stmts(self, name: str) -> list[str]:
        blk = self.input_block(name)
        if blk is None:
            return []
        return self.c.statement_chain(blk)

    def body(self, name: str) -> list[str]:
        """Indented statement lines for a C-shaped input (``pass`` if empty)."""
        lines = self.stmts(name)
        if not any(l.strip() and not l.strip().startswith("#") for l in lines):
            lines = lines + ["pass"]
        return [INDENT + l for l in lines]

    def layers(self, name: str) -> list[str]:
        blk = self.input_block(name)
        out: list[str] = []
        while blk is not None:
            if self.c.enabled(blk):
                out.extend(self.c.layer_code(blk))
            blk = (blk.get("next") or {}).get("block")
        return out

    def settings(self, name: str) -> list[Setting]:
        blk = self.input_block(name)
        out: list[Setting] = []
        while blk is not None:
            if self.c.enabled(blk):
                s = self.c.setting(blk)
                if s is not None:
                    out.extend(s if isinstance(s, list) else [s])
            blk = (blk.get("next") or {}).get("block")
        return out

    def child_blocks(self, name: str) -> list["BlockCtx"]:
        blk = self.input_block(name)
        out = []
        while blk is not None:
            if self.c.enabled(blk):
                out.append(BlockCtx(self.c, blk))
            blk = (blk.get("next") or {}).get("block")
        return out

    # -- names ----------------------------------------------------------
    def name_ident(self, field_name: str = "NAME") -> str:
        arg = self.spec.args[field_name]
        assert isinstance(arg, Name)
        return self.c.object_ident(arg.kind, self.field(field_name))

    def ref(self, field_name: str) -> str:
        arg = self.spec.args[field_name]
        assert isinstance(arg, Ref)
        name = self.field(field_name)
        return self.c.object_ident(arg.kind, name, ref_block=self.id, check=True)

    def name(self, field_name: str = "NAME") -> str:
        return str(self.field(field_name))

    def wired(self) -> str:
        """Code for the trained weights this (output) block works with."""
        return self.c.wired_ident(self.id)

    # -- diagnostics -----------------------------------------------------
    def warn(self, msg: str):
        self.c.warn(msg, self.id)

    def error(self, msg: str):
        self.c.error(msg, self.id)

    # -- helpers ----------------------------------------------------------
    def bid(self) -> str:
        """``, _bid='…'`` keyword for runtime objects so errors can point at this block."""
        return f", _bid={self.id!r}" if (self.c.markers and self.id) else ""

    def bid_kw(self) -> list[str]:
        return [f"_bid={self.id!r}"] if (self.c.markers and self.id) else []


def section(title: str) -> str:
    """A comment line that heads a stack in the generated program."""
    return f"# ── {title} " + "─" * max(4, 74 - len(title))


def call(fn: str, *args: str, **kwargs: str | None) -> str:
    """Format a call; kwargs whose value is None are omitted."""
    parts = [a for a in args if a is not None]
    parts += [f"{k}={v}" for k, v in kwargs.items() if v is not None]
    return f"{fn}({', '.join(parts)})"


def call_multiline(fn: str, args: list[str], width: int = 88) -> str:
    one = f"{fn}({', '.join(args)})"
    if len(one) <= width and "\n" not in one:
        return one
    inner = ",\n".join(INDENT + a.replace("\n", "\n" + INDENT) for a in args)
    return f"{fn}(\n{inner},\n)"


class Compiler:
    def __init__(self, workspace: dict, *, markers: bool = True, project: str = "Untitled",
                 header: bool = True, source_file: str | None = None):
        self.ws = workspace or {}
        self.markers = markers
        self.project = project or "Untitled"
        self.header = header
        self.source_file = source_file
        self.diagnostics: list[Diagnostic] = []
        self.used: set[str] = set(RESERVED)
        self.var_names: dict[str, str] = {}  # var id -> name
        self.var_idents: dict[str, str] = {}  # var name -> ident
        self.obj_idents: dict[tuple[str, str], str] = {}  # (kind, name) -> ident
        self.defined: dict[tuple[str, str], list[str]] = {}  # (kind,name) -> defining block ids
        self.helpers: dict[str, list[str]] = {}
        self._fresh = 0
        self.world_expr = 0  # >0 while generating a world reward/condition formula
        # Training → output wiring.
        self.weights_idents: dict[str, str] = {}  # 📦 weights block id -> Python identifier
        self.weights_models: dict[str, str] = {}  # 📦 weights block id -> name of the model it packs
        self.stack_kind: str | None = None  # "training" / "output" while generating that kind of stack
        self.wired: str | None = None  # the weights identifier the current output stack is wired to
        self.in_proc = 0  # >0 while generating a My Blocks function

    # -- diagnostics -----------------------------------------------------
    def warn(self, msg, block_id=None):
        self.diagnostics.append(Diagnostic("warning", msg, block_id))

    def error(self, msg, block_id=None):
        self.diagnostics.append(Diagnostic("error", msg, block_id))

    def info(self, msg, block_id=None):
        self.diagnostics.append(Diagnostic("info", msg, block_id))

    # -- identifiers -------------------------------------------------------
    def fresh(self, prefix: str) -> str:
        while True:
            self._fresh += 1
            name = f"{prefix}_{self._fresh}"
            if name not in self.used:
                self.used.add(name)
                return name

    def _claim(self, base: str, alt_suffix: str) -> str:
        ident = sanitize(base)
        if ident in self.used or ident.lower() in ("nb",):
            cand = f"{ident}_{alt_suffix}"
            i = 2
            while cand in self.used:
                cand = f"{ident}_{alt_suffix}{i}"
                i += 1
            ident = cand
        self.used.add(ident)
        return ident

    def var_ident(self, field_value: Any) -> str:
        if isinstance(field_value, dict):
            name = field_value.get("name") or self.var_names.get(field_value.get("id", ""), None)
            if name is None:
                name = field_value.get("id", "var")
        else:
            name = self.var_names.get(str(field_value), str(field_value))
        if name not in self.var_idents:
            self.var_idents[name] = self._claim(name, "var")
        return self.var_idents[name]

    def object_ident(self, kind: str, name: str, ref_block: str | None = None, check: bool = False) -> str:
        name = str(name)
        key = (kind, name)
        if check and key not in self.defined:
            pretty = {"dataset": "dataset", "model": "model", "world": "world"}.get(kind, kind)
            self.error(f"There is no {pretty} called '{name}'. Create it first (or pick another name "
                       f"from the dropdown).", ref_block)
        if key not in self.obj_idents:
            self.obj_idents[key] = self._claim(name, kind)
        return self.obj_idents[key]

    def wired_ident(self, block_id: str | None) -> str:
        """The weights an output block uses: the ones wired into the output stack it is in."""
        if self.stack_kind == "output":
            return self.wired or "None"  # an unwired output stack is reported once, on its start block
        if self.in_proc:
            return "nb.wired()"  # a My Blocks function: whichever output stack is calling it
        self.error("This block uses a trained model, so it belongs in an output stack: put it under a "
                   "'○ start output' block that is wired to a 📦 weights block.", block_id)
        return "None"

    # -- tree helpers -----------------------------------------------------
    @staticmethod
    def enabled(blk: dict) -> bool:
        if blk.get("enabled") is False:
            return False
        if blk.get("disabledReasons"):
            return False
        return True

    def _walk(self, blk: dict, fn):
        fn(blk)
        for inp in (blk.get("inputs") or {}).values():
            for k in ("block", "shadow"):
                if inp.get(k):
                    self._walk(inp[k], fn)
        if blk.get("next") and blk["next"].get("block"):
            self._walk(blk["next"]["block"], fn)

    def top_blocks(self) -> list[dict]:
        blocks = ((self.ws.get("blocks") or {}).get("blocks")) or []
        return sorted(blocks, key=lambda b: (round(b.get("y", 0) / 40), b.get("x", 0)))

    # -- code generation ----------------------------------------------------
    def value_to_code(self, blk: dict, outer_order: float = ORDER_NONE) -> str:
        code, inner_order = self.value(blk)
        o, i = int(outer_order), int(inner_order)
        # Same rule as Blockly: parenthesise when the outer operator binds at least as tightly.
        if o <= i and not (o == i and o in (ORDER_ATOMIC, ORDER_NONE)):
            if not (outer_order == ORDER_MEMBER and inner_order == ORDER_FUNCTION_CALL):
                return f"({code})"
        return code

    def value(self, blk: dict) -> tuple[str, float]:
        t = blk.get("type")
        ctx = BlockCtx(self, blk)
        if not self.enabled(blk):
            return "None", ORDER_ATOMIC
        if t in BUILTIN_VALUES:
            res = BUILTIN_VALUES[t](ctx)
        else:
            spec = REGISTRY.get(t)
            if spec is None:
                self.error(f"Unknown block type '{t}'.", blk.get("id"))
                return "None", ORDER_ATOMIC
            if spec.shape != "value":
                self.error("This block can't be used as a value.", blk.get("id"))
                return "None", ORDER_ATOMIC
            res = spec.gen(ctx)
        if isinstance(res, tuple):
            return res
        return res, ORDER_FUNCTION_CALL

    def statement_chain(self, blk: dict | None) -> list[str]:
        lines: list[str] = []
        while blk is not None:
            if self.enabled(blk):
                lines.extend(self.statement(blk))
            blk = (blk.get("next") or {}).get("block")
        return lines

    def statement(self, blk: dict) -> list[str]:
        t = blk.get("type")
        ctx = BlockCtx(self, blk)
        bid = blk.get("id")
        if t in BUILTIN_STATEMENTS:
            out = BUILTIN_STATEMENTS[t](ctx)
            marker = True
        else:
            spec = REGISTRY.get(t)
            if spec is None:
                if t in BUILTIN_VALUES:
                    self.warn("A reporter block on its own does nothing — plug it into a slot.", bid)
                    return []
                self.error(f"Unknown block type '{t}'.", bid)
                return []
            if spec.shape == "layer":
                self.error("Layer blocks belong inside a 'create neural network' block.", bid)
                return []
            if spec.shape == "setting":
                self.error("This setting block must go inside the block it configures "
                           "(look at its shape and colour).", bid)
                return []
            if spec.shape == "value":
                self.warn("A reporter block on its own does nothing — plug it into a slot.", bid)
                return []
            out = spec.gen(ctx)
            marker = spec.shape != "hat"
        if out is None:
            out = []
        if isinstance(out, str):
            out = out.split("\n")
        if self.markers and marker and bid and out:
            out = [f"nb.at({bid!r})"] + list(out)
        return list(out)

    def layer_code(self, blk: dict) -> list[str]:
        t = blk.get("type")
        spec = REGISTRY.get(t)
        if spec is None or spec.shape != "layer":
            self.error("Only layer blocks can go inside a model.", blk.get("id"))
            return []
        res = spec.gen(BlockCtx(self, blk))
        if res is None:
            return []
        return res if isinstance(res, list) else [res]

    def setting(self, blk: dict):
        t = blk.get("type")
        spec = REGISTRY.get(t)
        if spec is None or spec.shape != "setting":
            self.error("Only setting blocks can go here.", blk.get("id"))
            return None
        res = spec.gen(BlockCtx(self, blk))
        if isinstance(res, Setting):
            res.block_id = res.block_id or blk.get("id")
        elif isinstance(res, list):
            for r in res:
                r.block_id = r.block_id or blk.get("id")
        return res

    # -- whole program -----------------------------------------------------
    def compile(self) -> CompileResult:
        for v in self.ws.get("variables") or []:
            self.var_names[v.get("id")] = v.get("name")
        # Variables first, so they get the plain names.
        for v in self.ws.get("variables") or []:
            self.var_ident(v.get("name"))

        tops = self.top_blocks()
        # Pre-scan: which names are defined where.
        def scan(b):
            spec = REGISTRY.get(b.get("type"))
            if spec and spec.defines and self.enabled(b):
                kind, fname = spec.defines
                name = str((b.get("fields") or {}).get(fname, spec.args[fname].default))
                self.defined.setdefault((kind, name), []).append(b.get("id"))
        for tb in tops:
            self._walk(tb, scan)
        for (kind, name) in self.defined:
            self.object_ident(kind, name)

        hats = [b for b in tops if b.get("type") == "nb_when_run" and self.enabled(b)]
        outs = [b for b in tops if b.get("type") == "nb_when_output" and self.enabled(b)]
        procs = [b for b in tops if b.get("type") in ("procedures_defnoreturn", "procedures_defreturn")]
        others = [b for b in tops if b not in hats and b not in outs and b not in procs]
        self.find_weights(hats)

        stray = sum(1 for b in others if b.get("type") not in ("nb_comment",))
        if not hats and not outs:
            self.error("Add a 'start training when ▶ clicked' block (from Start) and attach your blocks under it.")
        elif stray:
            for b in others:
                if b.get("type") == "nb_weights":
                    self.warn("The 📦 weights block goes at the very bottom of a 'start training when ▶ clicked' "
                              "stack.", b.get("id"))
                else:
                    self.warn("This block isn't attached to a 'start training' or '○ start output' block, so it "
                              "won't run.", b.get("id"))
        wired = {str((h.get("fields") or {}).get("PORT") or "") for h in outs}
        for wid in self.weights_idents:
            if wid not in wired:
                self.warn("Nothing uses these weights yet. Drag a wire from this block's ● to the ○ of a "
                          "'○ start output' block (from Output), then build the output stack under it.", wid)

        proc_lines: list[str] = []
        self.in_proc += 1
        for p in procs:
            proc_lines.extend(self.statement(p))
            proc_lines.append("")
        self.in_proc -= 1

        main_lines: list[str] = []
        for h in hats:
            self.stack_kind = "training"
            main_lines.append(section("start training when ▶ clicked"))
            main_lines.extend(self.statement(h))
            main_lines.extend(self.statement_chain((h.get("next") or {}).get("block")))
            main_lines.append("")
        for h in outs:
            main_lines.extend(self.output_stack(h))
            main_lines.append("")
        self.stack_kind = None

        code = self.assemble(proc_lines, main_lines)
        names = {ident: {"kind": k, "name": n} for (k, n), ident in self.obj_idents.items()}
        for name, ident in self.var_idents.items():
            names[ident] = {"kind": "variable", "name": name}
        return CompileResult(code=code, diagnostics=self.diagnostics, names=names)

    def find_weights(self, hats: list[dict]):
        """Give every 📦 weights block that ends up in a training stack a Python name.

        Blocks inside settings (e.g. 'every N steps do') are skipped: they become functions, and the
        weights belong to the main flow of the training program.
        """
        def walk(blk):
            while blk is not None:
                if not self.enabled(blk):
                    pass
                elif blk.get("type") == "nb_weights":
                    model = str((blk.get("fields") or {}).get("MODEL") or "model")
                    self.weights_idents[blk.get("id")] = self._claim(f"{model}_weights", "w")
                    self.weights_models[blk.get("id")] = model
                else:
                    spec = REGISTRY.get(blk.get("type"))
                    if spec is None or spec.shape != "setting":
                        for name, inp in (blk.get("inputs") or {}).items():
                            if spec is None or isinstance(spec.args.get(name), Stack):
                                walk(inp.get("block"))
                blk = (blk.get("next") or {}).get("block")

        for h in hats:
            walk((h.get("next") or {}).get("block"))

    def output_stack(self, hat: dict) -> list[str]:
        """An output stack: runs after training, using the weights wired into its start block."""
        hid = hat.get("id")
        src = str((hat.get("fields") or {}).get("PORT") or "")
        ident = self.weights_idents.get(src)
        if not src:
            self.error("This output stack isn't wired to any weights yet. Drag a wire from the ● on a "
                       "'📦 weights of trained …' block to the ○ on this block.", hid)
        elif ident is None:
            self.error("The 📦 weights block this was wired to is gone (or isn't at the bottom of a training stack "
                       "any more). Draw a new wire from a weights block to this ○.", hid)
        model = self.weights_models.get(src)
        title = f"○ start output ← 📦 weights of trained {model!r}" if model else "○ start output (not wired)"
        lines = [section(title)]
        self.stack_kind, self.wired = "output", ident
        try:
            body = self.statement_chain((hat.get("next") or {}).get("block"))
        finally:
            self.stack_kind, self.wired = None, None
        w = ident or "None"
        if self.markers and hid:
            lines.append(f"nb.at({hid!r})")
        lines.append(f"nb.start_output({w}" + (f", _bid={hid!r})" if self.markers and hid else ")"))
        lines.extend(body)
        lines.append(f"nb.end_output({w})")
        return lines

    def global_names(self, exclude: set[str] = frozenset()) -> list[str]:
        names = list(self.var_idents.values()) + list(self.obj_idents.values())
        return [n for n in names if n not in exclude]

    def assemble(self, proc_lines: list[str], main_lines: list[str]) -> str:
        out: list[str] = []
        if self.header:
            stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            out += [
                f'"""{self.project} — generated by NeuroBlocks on {stamp}.',
                "",
                "Run it with:   python this_file.py            (add --device cuda on a GPU machine)",
                "         or:   neuroblocks run this_file.py",
                '"""',
            ]
        out += ["import neuroblocks as nb", ""]
        setup_kw = [f"project={self.project!r}"]
        out.append(f"nb.setup({', '.join(setup_kw)})")
        out.append("")
        var_idents = sorted(set(self.var_idents.values()))
        params_assigned = set()
        if var_idents:
            out.append("# Variables")
            for v in var_idents:
                out.append(f"{v} = None")
            out.append("")
        if self.weights_idents:
            out.append("# 📦 What training produces (filled in by the weights blocks, used by output stacks)")
            for w in self.weights_idents.values():
                out.append(f"{w} = None")
            out.append("")
        for name, lines in self.helpers.items():
            out.extend(lines)
            out.append("")
        if proc_lines:
            out.append("# My Blocks (functions)")
            out.extend(proc_lines)
        if main_lines:
            out.extend(main_lines)
        out.append("nb.finish()")
        code = "\n".join(out).rstrip() + "\n"
        code = re.sub(r"\n{3,}", "\n\n", code)
        return code


def compile_workspace(workspace: dict, *, markers: bool = True, project: str = "Untitled",
                      header: bool = True) -> CompileResult:
    # Import block modules so the registry is populated.
    from . import blocks  # noqa: F401
    comp = Compiler(workspace, markers=markers, project=project, header=header)
    try:
        return comp.compile()
    except CompileError as e:
        comp.error(str(e), e.block_id)
        return CompileResult(code="", diagnostics=comp.diagnostics, names={})
