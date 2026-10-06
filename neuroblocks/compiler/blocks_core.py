"""Core blocks: Start, Control, Operators, Variables, Lists, My Blocks, Show & Say.

Includes Python generators for the Blockly built-in blocks we put in the toolbox.
"""
from __future__ import annotations

from .core import (ORDER_ADDITIVE, ORDER_ATOMIC, ORDER_COLLECTION, ORDER_CONDITIONAL,
                   ORDER_EXPONENTIATION, ORDER_FUNCTION_CALL, ORDER_LOGICAL_AND, ORDER_LOGICAL_NOT,
                   ORDER_LOGICAL_OR, ORDER_MEMBER, ORDER_MULTIPLICATIVE, ORDER_NONE, ORDER_RELATIONAL,
                   ORDER_UNARY_SIGN, INDENT, BlockCtx, builtin_statement, builtin_value, number_literal,
                   sanitize)
from .spec import Drop, Icon, Num, Stack, Str, Val, Var, block

FLAG_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<path d='M5 2.5v19' stroke='%23455A64' stroke-width='2' stroke-linecap='round'/>"
    "<path d='M6 3.5c3-1.6 5.5 1.6 8.5 0s4-.5 4.5 0v9c-.5-.5-1.5-1.6-4.5 0s-5.5-1.6-8.5 0z' "
    "fill='%234CBF56' stroke='%2345993d' stroke-width='1.2' stroke-linejoin='round'/></svg>"
)

# ===========================================================================
# Start
# ===========================================================================


@block("nb_when_run", "events", "when %FLAG clicked", shape="hat",
       FLAG=Icon(FLAG_SVG, 26, 26, "green flag"),
       tooltip="Your program starts here. Attach blocks underneath, then press the green ▶ Run button.",
       help="Like Scratch's green flag: everything stacked under this block runs, top to bottom, "
            "when you press Run. You can have several of these; they run one after another from "
            "the top of the workspace down.")
def _when_run(b: BlockCtx):
    return []


@block("nb_setup", "events", "use device %DEVICE random seed %SEED",
       DEVICE=Drop([("auto (best available)", "auto"), ("CPU", "cpu"), ("GPU (CUDA)", "cuda"),
                    ("Apple GPU (MPS)", "mps")], "auto"),
       SEED=Num(42, integer=True),
       tooltip="Choose where to compute (CPU or GPU) and fix the random seed so runs are repeatable.",
       help="'auto' picks an NVIDIA GPU if there is one, then an Apple GPU, then the CPU. A seed makes "
            "random things (initial weights, shuffling, terrain) the same every run. On the command "
            "line, --device overrides this block.")
def _setup(b: BlockCtx):
    return [f"nb.config(device={b.q('DEVICE')}, seed={b.val('SEED')})"]


@block("nb_param", "variables", "hyperparameter %VAR = %VALUE",
       VAR=Var("steps"), VALUE=Num(1000),
       tooltip="Sets a variable that can be changed from the command line without editing blocks.",
       help="Works like 'set variable', but when you run the exported program on a cloud machine you "
            "can override it: neuroblocks run project.nblk --set steps=50000. Great for scaling a "
            "small local experiment up on a GPU.")
def _param(b: BlockCtx):
    v = b.var("VAR")
    return [f"{v} = nb.param({v!r}, {b.val('VALUE')})"]


# ===========================================================================
# Control
# ===========================================================================


@block("nb_wait", "control", "wait %SECS seconds", SECS=Num(1),
       tooltip="Pause the program for a number of seconds.")
def _wait(b):
    return [f"nb.wait({b.val('SECS')})"]


@block("nb_stop", "control", "stop the program", shape="terminal",
       tooltip="Stop running right here.")
def _stop(b):
    return ["nb.stop()"]


@builtin_statement("controls_if")
def _if(b: BlockCtx):
    n_elif = int(b.extra.get("elseIfCount", 0) or 0)
    has_else = bool(b.extra.get("hasElse", False))
    # Fallback for old saves without extraState: discover inputs.
    inputs = b.data.get("inputs") or {}
    i = 0
    lines = []
    while True:
        key = f"IF{i}"
        if i > n_elif and key not in inputs:
            break
        cond = b.val(key, ORDER_NONE, default="False")
        lines.append(("if " if i == 0 else "elif ") + cond + ":")
        lines.extend(b.body(f"DO{i}"))
        i += 1
    if has_else or "ELSE" in inputs:
        lines.append("else:")
        lines.extend(b.body("ELSE"))
    return lines


@builtin_statement("controls_repeat_ext", "controls_repeat")
def _repeat(b: BlockCtx):
    if b.type == "controls_repeat":
        times = number_literal(b.field("TIMES", 10))
    else:
        times = b.val("TIMES", ORDER_NONE, default="0")
    loopvar = b.c.fresh("_count")
    times_code = times if times.isdigit() else f"int({times})"
    return [f"for {loopvar} in range({times_code}):"] + b.body("DO")


@builtin_statement("controls_whileUntil")
def _while(b: BlockCtx):
    until = b.field("MODE", "WHILE") == "UNTIL"
    cond = b.val("BOOL", ORDER_LOGICAL_NOT if until else ORDER_NONE, default="False")
    if until:
        cond = f"not {cond}"
    return [f"while {cond}:"] + b.body("DO")


@builtin_statement("controls_for")
def _for(b: BlockCtx):
    v = b.var("VAR")
    lits = [b.literal(k) for k in ("FROM", "TO", "BY")]
    if all(isinstance(x, float) and x.is_integer() for x in lits):
        a, z, step = (int(x) for x in lits)
        step = abs(step) or 1
        if a <= z:
            rng = f"range({a}, {z + 1})" if step == 1 else f"range({a}, {z + 1}, {step})"
        else:
            rng = f"range({a}, {z - 1}, -{step})"
    else:
        rng = f"nb.count({b.val('FROM', default='1')}, {b.val('TO', default='10')}, {b.val('BY', default='1')})"
    return [f"for {v} in {rng}:"] + b.body("DO")


@builtin_statement("controls_forEach")
def _foreach(b: BlockCtx):
    v = b.var("VAR")
    lst = b.val("LIST", ORDER_NONE, default="[]")
    return [f"for {v} in {lst}:"] + b.body("DO")


@builtin_statement("controls_flow_statements")
def _flow(b: BlockCtx):
    return ["break" if b.field("FLOW", "BREAK") == "BREAK" else "continue"]


# ===========================================================================
# Operators (math / logic / text)
# ===========================================================================


@builtin_value("math_number")
def _num(b: BlockCtx):
    code = number_literal(b.field("NUM", 0))
    return code, (ORDER_UNARY_SIGN if code.startswith("-") else ORDER_ATOMIC)


@builtin_value("text")
def _text(b: BlockCtx):
    return repr(str(b.field("TEXT", ""))), ORDER_ATOMIC


_ARITH = {
    "ADD": (" + ", ORDER_ADDITIVE),
    "MINUS": (" - ", ORDER_ADDITIVE),
    "MULTIPLY": (" * ", ORDER_MULTIPLICATIVE),
    "DIVIDE": (" / ", ORDER_MULTIPLICATIVE),
    "POWER": (" ** ", ORDER_EXPONENTIATION),
}


@builtin_value("math_arithmetic")
def _arith(b: BlockCtx):
    op, order = _ARITH[b.field("OP", "ADD")]
    a = b.val("A", order, default="0")
    c = b.val("B", order, default="0")
    return f"{a}{op}{c}", order


@builtin_value("math_single")
def _single(b: BlockCtx):
    op = b.field("OP", "ROOT")
    if op == "NEG":
        return f"-{b.val('NUM', ORDER_UNARY_SIGN, default='0')}", ORDER_UNARY_SIGN
    if op == "ABS":
        return f"abs({b.val('NUM', default='0')})", ORDER_FUNCTION_CALL
    if op == "POW10":
        return f"10 ** {b.val('NUM', ORDER_EXPONENTIATION, default='0')}", ORDER_EXPONENTIATION
    fn = {"ROOT": "nb.math.sqrt", "LN": "nb.math.log", "LOG10": "nb.math.log10", "EXP": "nb.math.exp"}[op]
    return f"{fn}({b.val('NUM', default='0')})", ORDER_FUNCTION_CALL


@builtin_value("math_trig")
def _trig(b: BlockCtx):
    op = b.field("OP", "SIN").lower()
    return f"nb.trig({op!r}, {b.val('NUM', default='0')})", ORDER_FUNCTION_CALL


@builtin_value("math_constant")
def _const(b: BlockCtx):
    c = b.field("CONSTANT", "PI")
    return {
        "PI": ("nb.math.pi", ORDER_MEMBER), "E": ("nb.math.e", ORDER_MEMBER),
        "GOLDEN_RATIO": ("(1 + nb.math.sqrt(5)) / 2", ORDER_MULTIPLICATIVE),
        "SQRT2": ("nb.math.sqrt(2)", ORDER_FUNCTION_CALL), "SQRT1_2": ("nb.math.sqrt(0.5)", ORDER_FUNCTION_CALL),
        "INFINITY": ("float('inf')", ORDER_FUNCTION_CALL),
    }[c]


@builtin_value("math_round")
def _round(b: BlockCtx):
    op = b.field("OP", "ROUND")
    fn = {"ROUND": "round", "ROUNDUP": "nb.math.ceil", "ROUNDDOWN": "nb.math.floor"}[op]
    return f"{fn}({b.val('NUM', default='0')})", ORDER_FUNCTION_CALL


@builtin_value("math_modulo")
def _mod(b: BlockCtx):
    return (f"{b.val('DIVIDEND', ORDER_MULTIPLICATIVE, '0')} % {b.val('DIVISOR', ORDER_MULTIPLICATIVE, '1')}",
            ORDER_MULTIPLICATIVE)


@builtin_value("math_constrain")
def _constrain(b: BlockCtx):
    return (f"min(max({b.val('VALUE', default='0')}, {b.val('LOW', default='0')}), "
            f"{b.val('HIGH', default='0')})", ORDER_FUNCTION_CALL)


@builtin_value("math_random_int")
def _randint(b: BlockCtx):
    return f"nb.random_int({b.val('FROM', default='1')}, {b.val('TO', default='10')})", ORDER_FUNCTION_CALL


@builtin_value("math_random_float")
def _randfloat(b: BlockCtx):
    return "nb.random_float()", ORDER_FUNCTION_CALL


@builtin_value("math_on_list")
def _onlist(b: BlockCtx):
    op = b.field("OP", "SUM").lower()
    return f"nb.list_stat({op!r}, {b.val('LIST', default='[]')})", ORDER_FUNCTION_CALL


@builtin_value("math_number_property")
def _numprop(b: BlockCtx):
    prop = b.field("PROPERTY", "EVEN")
    n = b.val("NUMBER_TO_CHECK", ORDER_MULTIPLICATIVE, "0")
    if prop == "EVEN":
        return f"{n} % 2 == 0", ORDER_RELATIONAL
    if prop == "ODD":
        return f"{n} % 2 == 1", ORDER_RELATIONAL
    if prop == "WHOLE":
        return f"{n} % 1 == 0", ORDER_RELATIONAL
    if prop == "POSITIVE":
        return f"{n} > 0", ORDER_RELATIONAL
    if prop == "NEGATIVE":
        return f"{n} < 0", ORDER_RELATIONAL
    if prop == "DIVISIBLE_BY":
        return f"{n} % {b.val('DIVISOR', ORDER_MULTIPLICATIVE, '1')} == 0", ORDER_RELATIONAL
    if prop == "PRIME":
        return f"nb.is_prime({b.val('NUMBER_TO_CHECK', default='0')})", ORDER_FUNCTION_CALL
    return "False", ORDER_ATOMIC


@builtin_value("logic_boolean")
def _bool(b: BlockCtx):
    return ("True" if b.field("BOOL", "TRUE") == "TRUE" else "False"), ORDER_ATOMIC


@builtin_value("logic_null")
def _null(b: BlockCtx):
    return "None", ORDER_ATOMIC


_CMP = {"EQ": "==", "NEQ": "!=", "LT": "<", "LTE": "<=", "GT": ">", "GTE": ">="}


@builtin_value("logic_compare")
def _cmp(b: BlockCtx):
    op = _CMP[b.field("OP", "EQ")]
    return (f"{b.val('A', ORDER_RELATIONAL, '0')} {op} {b.val('B', ORDER_RELATIONAL, '0')}",
            ORDER_RELATIONAL)


@builtin_value("logic_operation")
def _logic(b: BlockCtx):
    if b.field("OP", "AND") == "AND":
        return f"{b.val('A', ORDER_LOGICAL_AND, 'False')} and {b.val('B', ORDER_LOGICAL_AND, 'False')}", ORDER_LOGICAL_AND
    return f"{b.val('A', ORDER_LOGICAL_OR, 'False')} or {b.val('B', ORDER_LOGICAL_OR, 'False')}", ORDER_LOGICAL_OR


@builtin_value("logic_negate")
def _not(b: BlockCtx):
    return f"not {b.val('BOOL', ORDER_LOGICAL_NOT, 'True')}", ORDER_LOGICAL_NOT


@builtin_value("logic_ternary")
def _ternary(b: BlockCtx):
    return (f"{b.val('THEN', ORDER_CONDITIONAL, 'None')} if {b.val('IF', ORDER_CONDITIONAL, 'False')} "
            f"else {b.val('ELSE', ORDER_CONDITIONAL, 'None')}", ORDER_CONDITIONAL)


@builtin_value("text_join")
def _join(b: BlockCtx):
    n = int(b.extra.get("itemCount", 0) or 0)
    inputs = b.data.get("inputs") or {}
    n = max(n, max([int(k[3:]) + 1 for k in inputs if k.startswith("ADD") and k[3:].isdigit()] or [0]))
    items = [b.val(f"ADD{i}", ORDER_NONE, "''") for i in range(n)]
    if not items:
        return "''", ORDER_ATOMIC
    return f"nb.join({', '.join(items)})", ORDER_FUNCTION_CALL


@builtin_value("text_length")
def _textlen(b: BlockCtx):
    return f"len(nb.text({b.val('VALUE', default=chr(39) * 2)}))", ORDER_FUNCTION_CALL


@builtin_value("text_isEmpty")
def _textempty(b: BlockCtx):
    return f"not nb.text({b.val('VALUE', default=chr(39) * 2)})", ORDER_LOGICAL_NOT


@builtin_value("text_changeCase")
def _case(b: BlockCtx):
    m = {"UPPERCASE": "upper", "LOWERCASE": "lower", "TITLECASE": "title"}[b.field("CASE", "UPPERCASE")]
    return f"nb.text({b.val('TEXT', default=chr(39) * 2)}).{m}()", ORDER_FUNCTION_CALL


@builtin_value("text_trim")
def _trim(b: BlockCtx):
    m = {"BOTH": "strip", "LEFT": "lstrip", "RIGHT": "rstrip"}[b.field("MODE", "BOTH")]
    return f"nb.text({b.val('TEXT', default=chr(39) * 2)}).{m}()", ORDER_FUNCTION_CALL


@builtin_statement("text_print")
def _print(b: BlockCtx):
    return [f"nb.say({b.val('TEXT', default=chr(39) * 2)})"]


@builtin_statement("text_append")
def _append_text(b: BlockCtx):
    v = b.var("VAR")
    return [f"{v} = nb.join({v}, {b.val('TEXT', default=chr(39) * 2)})"]


# ===========================================================================
# Variables
# ===========================================================================


@builtin_value("variables_get", "variables_get_dynamic")
def _varget(b: BlockCtx):
    return b.var("VAR"), ORDER_ATOMIC


@builtin_statement("variables_set", "variables_set_dynamic")
def _varset(b: BlockCtx):
    return [f"{b.var('VAR')} = {b.val('VALUE', default='0')}"]


@builtin_statement("math_change")
def _change(b: BlockCtx):
    v = b.var("VAR")
    return [f"{v} = nb.number({v}) + {b.val('DELTA', ORDER_ADDITIVE, '1')}"]


# ===========================================================================
# Lists
# ===========================================================================


@builtin_value("lists_create_empty")
def _lempty(b: BlockCtx):
    return "[]", ORDER_ATOMIC


@builtin_value("lists_create_with")
def _lwith(b: BlockCtx):
    n = int(b.extra.get("itemCount", 0) or 0)
    inputs = b.data.get("inputs") or {}
    n = max(n, max([int(k[3:]) + 1 for k in inputs if k.startswith("ADD") and k[3:].isdigit()] or [0]))
    items = [b.val(f"ADD{i}", ORDER_NONE, "None") for i in range(n)]
    return f"[{', '.join(items)}]", ORDER_ATOMIC


@builtin_value("lists_repeat")
def _lrepeat(b: BlockCtx):
    return (f"[{b.val('ITEM', default='None')}] * int({b.val('NUM', default='0')})",
            ORDER_MULTIPLICATIVE)


@builtin_value("lists_length")
def _llen(b: BlockCtx):
    return f"len(nb.as_list({b.val('VALUE', default='[]')}))", ORDER_FUNCTION_CALL


@builtin_value("lists_isEmpty")
def _lisempty(b: BlockCtx):
    return f"not nb.as_list({b.val('VALUE', default='[]')})", ORDER_LOGICAL_NOT


@builtin_value("lists_indexOf")
def _lindex(b: BlockCtx):
    last = b.field("END", "FIRST") == "LAST"
    return (f"nb.index_of({b.val('VALUE', default='[]')}, {b.val('FIND', default='None')}, last={last})",
            ORDER_FUNCTION_CALL)


def _where_at(b: BlockCtx):
    where = b.field("WHERE", "FROM_START")
    at = b.val("AT", default="1") if where in ("FROM_START", "FROM_END") else "1"
    return repr(where.lower()), at


@builtin_value("lists_getIndex")
def _lget(b: BlockCtx):
    where, at = _where_at(b)
    mode = b.field("MODE", "GET")
    fn = "nb.list_get" if mode == "GET" else "nb.list_pop"
    return f"{fn}({b.val('VALUE', default='[]')}, {where}, {at})", ORDER_FUNCTION_CALL


@builtin_statement("lists_getIndex")
def _lget_stmt(b: BlockCtx):
    where, at = _where_at(b)
    return [f"nb.list_pop({b.val('VALUE', default='[]')}, {where}, {at})"]


@builtin_statement("lists_setIndex")
def _lset(b: BlockCtx):
    where, at = _where_at(b)
    fn = "nb.list_set" if b.field("MODE", "SET") == "SET" else "nb.list_insert"
    return [f"{fn}({b.val('LIST', default='[]')}, {where}, {at}, {b.val('TO', default='None')})"]


@builtin_value("lists_getSublist")
def _lsub(b: BlockCtx):
    w1 = b.field("WHERE1", "FROM_START")
    w2 = b.field("WHERE2", "FROM_START")
    a1 = b.val("AT1", default="1") if w1 in ("FROM_START", "FROM_END") else "1"
    a2 = b.val("AT2", default="1") if w2 in ("FROM_START", "FROM_END") else "1"
    return (f"nb.sublist({b.val('LIST', default='[]')}, {w1.lower()!r}, {a1}, {w2.lower()!r}, {a2})",
            ORDER_FUNCTION_CALL)


@builtin_value("lists_sort")
def _lsort(b: BlockCtx):
    kind = b.field("TYPE", "NUMERIC").lower()
    rev = str(b.field("DIRECTION", "1")) == "-1"
    return f"nb.sort_list({b.val('LIST', default='[]')}, {kind!r}, reverse={rev})", ORDER_FUNCTION_CALL


@builtin_value("lists_reverse")
def _lrev(b: BlockCtx):
    return f"list(reversed(nb.as_list({b.val('LIST', default='[]')})))", ORDER_FUNCTION_CALL


@builtin_value("lists_split")
def _lsplit(b: BlockCtx):
    if b.field("MODE", "SPLIT") == "SPLIT":
        return f"nb.text({b.val('INPUT', default=chr(39) * 2)}).split({b.val('DELIM', default=chr(39) + ',' + chr(39))})", ORDER_FUNCTION_CALL
    return f"{b.val('DELIM', ORDER_MEMBER, chr(39) * 2)}.join(nb.text(x) for x in {b.val('INPUT', default='[]')})", ORDER_FUNCTION_CALL


@block("nb_list_add", "lists", "add %ITEM to %VAR", ITEM=Str("thing"), VAR=Var("results"),
       tooltip="Add an item to the end of a list (starts a new list if the variable is empty).")
def _list_add(b: BlockCtx):
    v = b.var("VAR")
    return [f"{v} = nb.append({v}, {b.val('ITEM')})"]


# ===========================================================================
# My Blocks (functions)
# ===========================================================================


def _proc_params(b: BlockCtx) -> list[str]:
    params = b.extra.get("params") or []
    names = []
    for p in params:
        if isinstance(p, dict):
            names.append(p.get("name"))
        else:
            names.append(str(p))
    return names


@builtin_statement("procedures_defnoreturn", "procedures_defreturn")
def _procdef(b: BlockCtx):
    fname = b.c.object_ident("function", b.field("NAME", "do_something"))
    params = [b.c.var_ident(p) for p in _proc_params(b)]
    body = b.stmts("STACK")
    lines = [f"def {fname}({', '.join(params)}):"]
    globs = b.c.global_names(exclude=set(params))
    inner = []
    if globs:
        inner.append(f"global {', '.join(sorted(set(globs)))}")
    inner.extend(body)
    if b.type == "procedures_defreturn" and b.has_input("RETURN"):
        inner.append(f"return {b.val('RETURN')}")
    if not any(l.strip() and not l.startswith("global") for l in inner):
        inner.append("pass")
    lines.extend(INDENT + l for l in inner)
    return lines


def _proc_call_args(b: BlockCtx):
    params = b.extra.get("params") or []
    return [b.val(f"ARG{i}", default="None") for i in range(len(params))]


@builtin_statement("procedures_callnoreturn", "procedures_callreturn")
def _proccall(b: BlockCtx):
    fname = b.c.object_ident("function", b.extra.get("name") or b.field("NAME", "do_something"))
    return [f"{fname}({', '.join(_proc_call_args(b))})"]


@builtin_value("procedures_callreturn")
def _proccall_val(b: BlockCtx):
    fname = b.c.object_ident("function", b.extra.get("name") or b.field("NAME", "do_something"))
    return f"{fname}({', '.join(_proc_call_args(b))})", ORDER_FUNCTION_CALL


@builtin_statement("procedures_ifreturn")
def _ifreturn(b: BlockCtx):
    cond = b.val("CONDITION", default="False")
    has_value = b.extra.get("hasReturnValue", True) if isinstance(b.extra, dict) else True
    if b.has_input("VALUE"):
        return [f"if {cond}:", INDENT + f"return {b.val('VALUE')}"]
    return [f"if {cond}:", INDENT + "return"]


# ===========================================================================
# Show & Say
# ===========================================================================


@block("nb_say", "output", "say %TEXT", TEXT=Str("Hello!"),
       tooltip="Show a message in the Console (like a Scratch speech bubble).")
def _say(b):
    return [f"nb.say({b.val('TEXT')})"]


@block("nb_chart_point", "output", "add point x %X y %Y to chart %CHART line %SERIES",
       X=Num(1), Y=Num(1), CHART=Str("My chart"), SERIES=Str("results"),
       tooltip="Draw your own live chart: each block adds one point to a line on the Charts tab.")
def _chart_point(b):
    return [f"nb.plot_point({b.val('CHART')}, {b.val('SERIES')}, {b.val('X')}, {b.val('Y')})"]


@block("nb_chart_list", "output", "show %KIND chart of %LIST titled %TITLE",
       KIND=Drop([("line", "line"), ("bar", "bar"), ("dots", "scatter")], "line"),
       LIST=Val(None), TITLE=Str("My chart"),
       tooltip="Plot a list of numbers as a chart (Results tab).")
def _chart_list(b):
    return [f"nb.plot_list({b.val('LIST', default='[]')}, kind={b.q('KIND')}, title={b.val('TITLE')})"]


@block("nb_sound", "output", "play sound %SOUND",
       SOUND=Drop([("ding", "ding"), ("success", "success"), ("oops", "oops"), ("pop", "pop")], "ding"),
       tooltip="Play a sound in the editor — handy at the end of a long training run.")
def _sound(b):
    return [f"nb.sound({b.q('SOUND')})"]


@block("nb_round_to", "operators", "round %NUM to %DIGITS decimals", shape="value", output="Number",
       NUM=Num(3.14159), DIGITS=Num(2, integer=True),
       tooltip="Round a number to a number of decimal places.")
def _round_to(b):
    from .helpers import int_code
    return f"round({b.val('NUM')}, {int_code(b, 'DIGITS')})", ORDER_FUNCTION_CALL


@block("nb_timer", "control", "seconds since start", shape="value", output="Number",
       tooltip="How many seconds the program has been running.")
def _timer(b):
    return "nb.timer()", ORDER_FUNCTION_CALL
