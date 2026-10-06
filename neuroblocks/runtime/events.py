"""Structured events from a running program.

Modes (``NEUROBLOCKS_EVENTS``):

* ``gui``    — one JSON object per line on stdout, prefixed with ``\\x1e``; the editor
               server relays them to the browser.
* ``cli``    — human-friendly console output (progress lines, tables, ✅/❌).
* ``silent`` — nothing printed (used by tests).

In every mode, events are also appended to ``<run_dir>/events.jsonl`` when a run
directory is known, so a cloud run can be opened in the GUI afterwards.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import threading
import time

PREFIX = "\x1e"
START_TIME = time.time()


def _clean(obj):
    """Make an object JSON-safe (NaN/inf -> None, numpy/torch -> python)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    # numpy / torch scalars & arrays
    tolist = getattr(obj, "tolist", None)
    if tolist is not None:
        try:
            return _clean(tolist())
        except Exception:
            pass
    item = getattr(obj, "item", None)
    if item is not None:
        try:
            return _clean(item())
        except Exception:
            pass
    return str(obj)


def _supports_unicode(stream) -> bool:
    enc = (getattr(stream, "encoding", None) or "").lower()
    return "utf" in enc


class ConsoleReporter:
    """Pretty terminal output for CLI runs."""

    def __init__(self, stream):
        self.out = stream
        self.tty = bool(getattr(stream, "isatty", lambda: False)())
        self.uni = _supports_unicode(stream)
        self.progress_line = False
        self.metrics: dict[str, dict[str, float]] = {}
        self.last_progress_print = 0.0
        self.last_progress_pct: dict[str, int] = {}
        self.last_debug = 0.0
        self.stream_key = None  # 'show … as output' text being written token by token
        self.stream_text = ""
        self.bars = None  # a live bar chart: printed once its loop is done, not after every update

    def sym(self, uni: str, ascii_: str) -> str:
        return uni if self.uni else ascii_

    def _end_progress(self):
        if self.progress_line:
            self.out.write("\n")
            self.progress_line = False
        if self.stream_key is not None:
            self.out.write("\n")
            self.stream_key = None

    def _write(self, text: str):
        try:
            self.out.write(text)
        except UnicodeEncodeError:
            self.out.write(text.encode("ascii", "replace").decode())
        self.out.flush()

    def _stream(self, ev):
        """Output text that grows (a model writing) is printed as it grows, not over and over."""
        key, text = ev.get("key"), ev.get("text", "")
        if key == self.stream_key and text.startswith(self.stream_text):
            self._write(text[len(self.stream_text):])
        else:
            self._end_progress()
            self._write(f"{self.sym('📝', '>')} {text}")
            self.stream_key = key
        self.stream_text = text

    def println(self, text: str = ""):
        self._end_progress()
        try:
            self.out.write(text + "\n")
        except UnicodeEncodeError:
            self.out.write(text.encode("ascii", "replace").decode() + "\n")
        self.out.flush()

    def handle(self, ev: dict):
        t = ev.get("type")
        if t == "say":
            self.println(f"{self.sym('💬', '>')} {ev.get('text', '')}")
        elif t == "log":
            level = ev.get("level", "info")
            if level == "debug":
                # Per-epoch / per-eval lines: at most one every few seconds.
                now = time.time()
                if now - self.last_debug < 4.0:
                    return
                self.last_debug = now
            pre = {"warn": self.sym("⚠️ ", "! "), "error": self.sym("❌ ", "x "), "debug": "  ",
                   "success": self.sym("✔ ", "+ ")}.get(level, "")
            self.println(f"{pre}{ev.get('text', '')}")
        elif t == "plot":
            self._plot(ev)
        elif t == "metric":
            series = ev.get("series", "")
            self.metrics.setdefault(ev.get("group", ""), {})[f"{ev.get('chart', '')}:{series}"] = ev.get("y")
        elif t == "progress":
            self._progress(ev)
        elif t == "text":
            title = ev.get("title", "")
            self.println(f"--- {title} ---" if title else "---")
            self.println(ev.get("text", ""))
            self.println("-" * max(3, len(title) + 8))
        elif t == "table":
            self._table(ev)
        elif t == "check":
            mark = self.sym("✅", "[PASS]") if ev.get("passed") else self.sym("❌", "[FAIL]")
            self.println(f"{mark} {ev.get('text', '')}")
        elif t == "error":
            self.println(f"{self.sym('❌', 'ERROR:')} {ev.get('message', '')}")
            if ev.get("hint"):
                self.println(f"   {self.sym('💡', 'hint:')} {ev['hint']}")
        elif t == "image":
            if ev.get("saved"):
                self.println(f"{self.sym('🖼 ', '[image] ')}{ev.get('title', 'image')} -> {ev['saved']}")
        elif t == "sim_replay":
            if ev.get("saved"):
                self.println(f"{self.sym('🎬', '[replay]')} {ev.get('title', 'replay')} -> {ev['saved']}")
        elif t == "dataset":
            info = ev.get("info", {})
            self.println(f"{self.sym('📦', '[data]')} {info.get('summary', ev.get('name', ''))}")
        elif t == "model":
            info = ev.get("info", {})
            self.println(f"{self.sym('🧠', '[model]')} {info.get('summary', ev.get('name', ''))}")
        elif t == "confusion":
            pass  # summarised by the evaluation table
        elif t == "weights":
            self.println(f"{self.sym('📦', '[weights]')} {ev.get('label', '')}: {ev.get('summary', '')}")
        elif t == "output_start":
            self.println(f"{self.sym('○', 'o')} output, using the trained {ev.get('label', '')} " + "-" * 20)
        elif t == "output_text":
            self._stream(ev)
        elif t == "bars":
            if self.bars is not None and self.bars.get("key") != ev.get("key"):
                self._flush_bars()
            self.bars = ev
        elif t == "output_picture":
            if ev.get("saved"):
                self.println(f"{self.sym('🖼 ', '[picture] ')}{ev.get('caption', 'picture')} -> {ev['saved']}")
        elif t in ("interactive", "ask", "ask_done", "output_end", "done"):
            self._flush_bars()
            self._end_progress()

    def _flush_bars(self):
        ev, self.bars = self.bars, None
        if ev is None:
            return
        self.println(ev.get("title", "top choices"))
        items = ev.get("items") or []
        width = max([len(str(i.get("label", ""))) for i in items] + [1])
        top = max([abs(float(i.get("p") or 0)) for i in items] + [1e-9]) if ev.get("scores") else 1.0
        for i in items:
            p = float(i.get("p") or 0)
            bar = ("█" if self.uni else "#") * int(round(28 * max(0.0, min(1.0, abs(p) / top))))
            val = f"{p:.3g}" if ev.get("scores") else f"{p * 100:.1f}%"
            self.println(f"  {str(i.get('label', '')).ljust(width)} {bar} {val}")

    def _progress(self, ev):
        cur, total = ev.get("current", 0), ev.get("total") or 0
        label = ev.get("label", "")
        info = ev.get("info") or ""
        pct = int(100 * cur / total) if total else 0
        finished = bool(total) and cur >= total
        if self.tty:
            width = 24
            filled = int(width * cur / total) if total else 0
            bar = ("█" if self.uni else "#") * filled + ("░" if self.uni else "-") * (width - filled)
            line = f"\r{label} [{bar}] {pct:3d}% {info}"
            try:
                self.out.write(line[:200].ljust(100))
            except UnicodeEncodeError:
                self.out.write(line.encode("ascii", "replace").decode()[:200])
            self.out.flush()
            self.progress_line = True
            if finished:
                self._end_progress()
        else:
            key = ev.get("id", label)
            last = self.last_progress_pct.get(key, -100)
            now = time.time()
            if finished or pct >= last + 10 or now - self.last_progress_print > 30:
                self.last_progress_pct[key] = pct
                self.last_progress_print = now
                self.println(f"{label} {pct:3d}% {info}")

    def _plot(self, ev):
        vals = ev.get("values") or []
        labels = ev.get("labels") or [str(i + 1) for i in range(len(vals))]
        self.println(ev.get("title", "chart"))
        if not vals:
            return
        top = max(abs(v) for v in vals if v is not None) or 1
        width = max(len(str(l)) for l in labels[:30])
        for lab, v in list(zip(labels, vals))[:30]:
            n = int(round(28 * abs(v or 0) / top))
            self.println(f"  {str(lab).ljust(width)} {('█' if self.uni else '#') * n} {_fmt_cell(v)}")

    def _table(self, ev):
        cols = ev.get("columns") or []
        rows = ev.get("rows") or []
        title = ev.get("title")
        if title:
            self.println(title)
        cells = [[str(c) for c in cols]] + [[_fmt_cell(v) for v in r] for r in rows]
        widths = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(len(cols))]
        for ri, r in enumerate(cells):
            self.println("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(r))))
            if ri == 0:
                self.println("  " + "  ".join("-" * w for w in widths))


def _fmt_cell(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


class Emitter:
    def __init__(self):
        self.mode = os.environ.get("NEUROBLOCKS_EVENTS", "cli")
        self.lock = threading.Lock()
        self.out = sys.stdout
        self.console = ConsoleReporter(self.out) if self.mode == "cli" else None
        self.log_file = None
        self.run_dir = None
        rd = os.environ.get("NEUROBLOCKS_RUN_DIR")
        if rd:
            self.set_run_dir(rd)

    def set_run_dir(self, path):
        from pathlib import Path
        self.run_dir = Path(path)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.log_file = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
        except OSError:
            self.log_file = None

    def emit(self, type: str, **payload):
        ev = {"type": type, "t": round(time.time() - START_TIME, 3)}
        ev.update(payload)
        try:
            line = json.dumps(ev, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
        except (ValueError, TypeError):
            ev = _clean(ev)
            line = json.dumps(ev, separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            if self.log_file is not None:
                try:
                    self.log_file.write(line + "\n")
                    self.log_file.flush()
                except OSError:
                    pass
            if self.mode == "gui":
                try:
                    self.out.write(PREFIX + line + "\n")
                    self.out.flush()
                except (OSError, ValueError):
                    pass
            elif self.mode == "cli" and self.console is not None:
                try:
                    self.console.handle(ev)
                except Exception:
                    pass
        return ev


_emitter: Emitter | None = None


def emitter() -> Emitter:
    global _emitter
    if _emitter is None:
        _emitter = Emitter()
    return _emitter


def emit(type: str, **payload):
    return emitter().emit(type, **payload)


def reset_emitter():
    global _emitter
    _emitter = None


class LineWriter(io.TextIOBase):
    """Replacement for sys.stdout in GUI mode: turns printed lines into log events."""

    def __init__(self, level="info"):
        super().__init__()
        self.level = level
        self.buf = ""

    def writable(self):
        return True

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.rstrip("\r")
            if "\r" in line:  # progress-bar style output: keep the last segment
                line = line.split("\r")[-1]
            if line.strip():
                emit("log", text=line, level=self.level)
        return len(s)

    def flush(self):
        pass

    @property
    def encoding(self):
        return "utf-8"


def png_base64(fig_or_array) -> str:
    """Encode a PIL image or HxWx3 uint8 array as base64 PNG."""
    from PIL import Image
    import numpy as np
    if isinstance(fig_or_array, Image.Image):
        img = fig_or_array
    else:
        arr = np.asarray(fig_or_array)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return base64.b64encode(bio.getvalue()).decode("ascii")
