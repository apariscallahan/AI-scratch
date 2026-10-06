"""Interactive testing in the editor's Output tab (with terminal fallbacks)."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from .core import STATE, gui_mode, text as fmt_text
from .errors import NBError
from .events import emit

__all__ = ["play_chat", "play_draw", "play_form"]


def _session(kind: str, model, options: dict, handler, _bid=None):
    """Show a widget in the Output tab and answer its requests until the user presses Done."""
    sid = f"{kind}-{time.time_ns()}"
    emit("interactive", id=sid, kind=kind, model=getattr(model, "name", ""),
         label=getattr(model, "label", getattr(model, "name", "")), options=options, block=_bid)
    emit("log", level="info", text=f"Try out '{getattr(model, 'label', model.name)}' in the Output tab — press Done "
                                   f"there to continue the program.")
    # Drain stale messages from an earlier session.
    while not STATE.interact_q.empty():
        try:
            STATE.interact_q.get_nowait()
        except Exception:
            break
    while True:
        msg = STATE.interact_q.get()
        if msg.get("cmd") == "interact_end":
            break
        data = msg.get("data") or {}
        rid = msg.get("rid")
        try:
            result = handler(data, rid)
            emit("interactive_result", id=sid, rid=rid, ok=True, result=result)
        except NBError as e:
            emit("interactive_result", id=sid, rid=rid, ok=False, error=e.message)
        except Exception as e:  # noqa: BLE001
            emit("interactive_result", id=sid, rid=rid, ok=False, error=f"{type(e).__name__}: {e}")
    emit("interactive_end", id=sid)


def _tty() -> bool:
    """True only for a real interactive terminal (on Windows, NUL also claims to be a tty)."""
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return False
        if sys.platform == "win32":
            import ctypes
            import msvcrt
            mode = ctypes.c_uint32()
            handle = msvcrt.get_osfhandle(sys.stdin.fileno())
            return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        return True
    except (AttributeError, ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Chat / prompt a language model
# ---------------------------------------------------------------------------


def play_chat(model, _bid=None):
    from .generate import _check_lm, generated_text
    _check_lm(model)
    if not gui_mode():
        if not _tty():
            emit("log", level="info", text="(Skipping the interactive prompt box: no keyboard attached.)")
            return
        print(f"Type a prompt for {model.name} (empty line to finish):")
        while True:
            try:
                prompt = input("you> ")
            except EOFError:
                break
            if not prompt.strip():
                break
            print(prompt + generated_text(model, prompt, 200, 0.8))
        return

    def handle(data, rid):
        prompt = str(data.get("prompt", ""))
        length = int(data.get("length", 200))
        temp = float(data.get("temperature", 0.8))
        out = generated_text(model, prompt, max(1, min(length, 2000)), temp, stream_id=f"chat-{rid}")
        return {"prompt": prompt, "text": out}

    _session("chat", model, {"length": 200, "temperature": 0.8}, handle, _bid)


# ---------------------------------------------------------------------------
# Drawing pad for image classifiers
# ---------------------------------------------------------------------------


def _center_like_mnist(img: np.ndarray, out: int, box: int) -> np.ndarray:
    """Crop the drawing to its ink, scale it into a box×box square and centre it by mass (like MNIST)."""
    from PIL import Image
    ys, xs = np.nonzero(img > 0.05)
    if len(xs) == 0:
        return np.zeros((out, out), dtype=np.float32)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    if out <= 12:
        # Tiny targets (8×8 digits): thicken the pen stroke first, or it shrinks to a hairline.
        k = max(1, int(round(0.08 * max(h, w))))
        padded = np.pad(crop, k)
        thick = np.zeros_like(padded)
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                if dy * dy + dx * dx <= k * k:
                    thick = np.maximum(thick, np.roll(np.roll(padded, dy, 0), dx, 1))
        crop = thick
        h, w = crop.shape
    s = box / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    small = np.asarray(pil, dtype=np.float32) / 255.0
    canvas = np.zeros((out, out), dtype=np.float32)
    y0, x0 = (out - nh) // 2, (out - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    if out > 12:
        # MNIST-style: move the centre of mass to the middle (without wrapping around the edges).
        tot = canvas.sum()
        if tot > 0:
            cy = (canvas.sum(1) * np.arange(out)).sum() / tot
            cx = (canvas.sum(0) * np.arange(out)).sum() / tot
            dy = int(np.clip(round(out / 2 - 0.5 - cy), -y0, out - (y0 + nh)))
            dx = int(np.clip(round(out / 2 - 0.5 - cx), -x0, out - (x0 + nw)))
            shifted = np.zeros_like(canvas)
            shifted[y0 + dy:y0 + dy + nh, x0 + dx:x0 + dx + nw] = canvas[y0:y0 + nh, x0:x0 + nw]
            canvas = shifted
    return canvas


def _drawing_to_input(model, pixels, size):
    meta = model.meta or {}
    shape = tuple(meta.get("input_shape") or ())
    if len(shape) != 3:
        raise NBError("The drawing pad works with image models.")
    c, h, w = shape
    img = np.asarray(pixels, dtype=np.float32).reshape(size, size)
    box = max(2, int(round(h * (20 / 28)))) if h >= 16 else h
    centred = _center_like_mnist(img, h, box)
    if w != h:
        from PIL import Image
        centred = np.asarray(Image.fromarray((centred * 255).astype(np.uint8)).resize((w, h)), dtype=np.float32) / 255
    # Shrinking a pen stroke makes it faint; brighten it to look like the training pictures:
    # full-white strokes, and about the same average brightness as the training data.
    if centred.max() > 0:
        centred = centred / centred.max()
        stats = meta.get("image_stats")
        target = float(np.mean(stats[0])) if stats else None
        if target:
            best = min((1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.22),
                       key=lambda g: abs(float(np.mean(centred ** g)) - target))
            centred = centred ** best
    arr = np.repeat(centred[None], c, 0)
    x = torch.from_numpy((arr * 255).round().clip(0, 255).astype(np.uint8))[None]
    return x, centred


def play_draw(model, _bid=None):
    from .evaluate import _small_png, meta_dataset
    fam = getattr(model, "family", None)
    if fam not in ("neural", "classic") or not getattr(model, "meta", None):
        raise NBError("Train an image classifier first, then draw for it.", block_id=_bid)
    if model.meta.get("modality") != "image":
        raise NBError("The drawing pad is for image models (like digits or MNIST).", block_id=_bid)
    if not gui_mode():
        emit("log", level="info", text="(The drawing pad only works in the NeuroBlocks editor.)")
        return
    ds = meta_dataset(model.meta)
    classes = ds.class_names or []

    def handle(data, rid):
        size = int(data.get("size", 28))
        x, centred = _drawing_to_input(model, data.get("pixels") or [], size)
        if fam == "classic":
            X = ds.prepare_x(x).reshape(1, -1).numpy()
            probs = model._class_probs(X)[0]
        else:
            with torch.no_grad():
                model.net.eval()
                out = model.net(ds.prepare_x(x.to(model.device))).float().cpu()
            probs = torch.softmax(out, -1)[0].numpy()
        order = np.argsort(probs)[::-1]
        return {"probs": [{"label": classes[i] if i < len(classes) else str(i), "p": float(probs[i])}
                          for i in order[:10]],
                "seen": _small_png(torch.from_numpy((centred * 255).astype(np.uint8))[None])}

    _session("draw", model, {"classes": classes, "shape": model.meta.get("input_shape")}, handle, _bid)


# ---------------------------------------------------------------------------
# Form: type feature values, get a prediction
# ---------------------------------------------------------------------------


def play_form(model, _bid=None):
    from .evaluate import predict
    meta = getattr(model, "meta", None) or {}
    fam = getattr(model, "family", None)
    if fam not in ("neural", "classic") or not meta:
        raise NBError("Train a model on a table of numbers first.", block_id=_bid)
    if meta.get("task") == "language_model":
        return play_chat(model, _bid)
    if meta.get("modality") == "image":
        return play_draw(model, _bid)
    if meta.get("modality") == "text":
        fields = [{"name": "text", "kind": "text", "value": ""}]
    else:
        names = meta.get("feature_names") or [f"x{i + 1}" for i in range(int(np.prod(meta.get("input_shape") or [1])))]
        means = meta.get("x_mean") or [0.0] * len(names)
        stds = meta.get("x_std") or [1.0] * len(names)
        fields = [{"name": n, "kind": "number", "value": round(float(m), 3), "step": round(float(s) / 10, 4) or 0.1,
                   "min": round(float(m - 3 * s), 3), "max": round(float(m + 3 * s), 3)}
                  for n, m, s in zip(names, means, stds)]
    if not gui_mode():
        if not _tty():
            emit("log", level="info", text="(Skipping the interactive form: no keyboard attached.)")
            return
        vals = []
        try:
            for f in fields:
                raw = input(f"{f['name']} [{f['value']}]: ").strip()
                vals.append(raw if f["kind"] == "text" else float(raw or f["value"]))
        except (EOFError, OSError):
            emit("log", level="info", text="(Skipping the interactive form: no keyboard input.)")
            return
        except ValueError:
            emit("log", level="warn", text="That wasn't a number — skipping the form.")
            return
        print("prediction:", fmt_text(predict(model, vals[0] if fields[0]["kind"] == "text" else vals)))
        return

    def handle(data, rid):
        vals = data.get("values") or []
        inp = vals[0] if fields and fields[0]["kind"] == "text" else [float(v) for v in vals]
        return {"prediction": fmt_text(predict(model, inp))}

    _session("form", model, {"fields": fields, "task": meta.get("task"),
                             "target": (meta.get("target_names") or ["prediction"])[0]}, handle, _bid)
