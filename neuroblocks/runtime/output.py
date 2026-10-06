"""Output: using a trained model step by step — the blocks under '○ start output'.

The 📦 weights block at the bottom of a training stack calls :func:`weights`, which packs the trained
model together with what is needed to use it (tokenizer, class names, how inputs were scaled …)
into a :class:`Weights` object. An output stack starts with :func:`start_output` and then works
with that object one step at a time, e.g. for a language model::

    tokens = gpt_weights.tokens_of("Once upon a time")
    scores = gpt_weights.next_token_scores(tokens)
    probs = nb.softmax(scores, temperature=0.8)
    tokens = nb.append(tokens, nb.pick(probs, "random"))
    nb.show_text(gpt_weights.text_of(tokens))
"""
from __future__ import annotations

import base64
import random as _random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .core import STATE, as_list, gui_mode, number, text as fmt_text
from .errors import NBError
from .events import emit, emitter

__all__ = ["Weights", "Picture", "weights", "start_output", "end_output", "wired", "softmax", "pick",
           "chance_of", "show_text", "show_picture", "ask", "ask_drawing", "answer", "flush"]

_DOES = {"language_model": "writes text", "classification": "sorts inputs into classes",
         "regression": "predicts numbers", "control": "plays in a world", "generation": "draws pictures"}


class _State:
    def __init__(self):
        self.current: Weights | None = None  # the weights of the output stack that is running
        self.block = None  # its start block
        self.runs = 0  # output stacks started so far
        self.asks = 0  # 'ask' blocks so far: text shown after an ask goes into a new box
        self.answer = ""
        self.pictures = 0  # pictures saved by CLI runs
        self.pending: dict[str, dict] = {}  # show_text updates held back (at most ~20 per second)
        self.sent: dict[str, float] = {}


_OUT = _State()


def _short(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 10_000:
        return f"{n / 1e3:.0f}K"
    return f"{n:,}"


def _numbers(values, what="scores") -> np.ndarray:
    """A list of numbers from a block value (list, tensor or a single number)."""
    if isinstance(values, str):
        raise NBError(f"Expected a list of {what}, but got the text {values[:40]!r}.")
    if isinstance(values, (Picture, Weights)):
        raise NBError(f"Expected a list of {what}, but got {values.describe()}.")
    out = []
    for v in as_list(values):
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            raise NBError(f"Expected a list of {what}, but it contains {fmt_text(v)!r}.") from None
    if not out:
        raise NBError(f"The list of {what} is empty.")
    return np.asarray(out, dtype=np.float64)


def _tty() -> bool:
    from .interactive import _tty as tty
    return tty()


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------


class Picture:
    """A picture value: channels × height × width numbers from 0 (dark) to 1 (bright)."""

    def __init__(self, pixels, drawing: tuple | None = None, label: str | None = None):
        t = torch.as_tensor(pixels, dtype=torch.float32).detach().cpu()
        if t.ndim == 2:
            t = t[None]
        if t.ndim != 3:
            raise NBError("A picture needs height × width numbers.")
        self.pixels = t.clamp(0, 1)
        self.drawing = drawing  # (pad pixels, pad size) when it came from 'ask for a drawing'
        self.label = label  # e.g. the class a generator drew

    def describe(self) -> str:
        _, h, w = self.pixels.shape
        return f"a {h}×{w} picture"

    __repr__ = describe

    def png(self) -> str:
        from .data import image_grid_png
        h, w = self.pixels.shape[1:]
        return image_grid_png(self.pixels[None], cols=1, scale=max(1, int(round(140 / max(h, w)))))

    def fit(self, shape) -> torch.Tensor:
        """As a uint8 batch of one, resized to ``shape`` (C, H, W) — like the pictures in a dataset."""
        c, h, w = (int(v) for v in shape)
        t = self.pixels
        if tuple(t.shape[1:]) != (h, w):
            t = F.interpolate(t[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        if t.shape[0] != c:
            t = t.mean(0, keepdim=True).repeat(c, 1, 1)
        return (t.clamp(0, 1) * 255).round().to(torch.uint8)[None]


# ---------------------------------------------------------------------------
# What training hands to output
# ---------------------------------------------------------------------------


class Weights:
    """A trained model plus everything needed to use it — what flows along the wire."""

    def __init__(self, model, bid=None):
        self.model = model
        self.name = str(getattr(model, "name", "model"))
        self.label = str(getattr(model, "label", self.name))
        self.family = getattr(model, "family", None)
        self.bid = bid
        self._ds = None
        self._try: dict | None = None
        self._tries = 0
        self._told: set[str] = set()

    # -- what is it? ---------------------------------------------------------------
    @property
    def meta(self) -> dict:
        return getattr(self.model, "meta", None) or {}

    @property
    def task(self) -> str | None:
        if self.family == "generator":
            return "generation"
        return self.meta.get("task")

    def describe(self) -> str:
        return f"the trained weights of {self.label}"

    __repr__ = describe

    def _tell(self, key: str, text: str):
        if key not in self._told:
            self._told.add(key)
            emit("log", level="info", text=text, block=STATE.current_block)

    def _need(self, task: str, what: str):
        if self.task != task:
            does = _DOES.get(self.task or "", "isn't trained yet")
            raise NBError(f"{what} needs a model that {_DOES[task]}, but {self.label} {does}.")

    def num_weights(self) -> int:
        agent = getattr(self.model, "agent", None)
        if self.task == "control" and agent is not None:
            return int(agent.num_params())
        n = getattr(self.model, "num_params", None)
        return int(n()) if callable(n) else 0

    def context(self) -> int:
        return int(self.meta.get("context") or 128)

    def _tokenizer(self):
        if self.family != "pretrained_lm" and not self.meta.get("tokenizer"):
            raise NBError(f"{self.label} didn't learn from text, so it has no tokens.",
                          hint="Tokens are for language models and text classifiers.")
        from .generate import tokenizer_of
        return tokenizer_of(self.model)

    def _dataset(self):
        """A data-less dataset that knows how the training inputs were encoded and scaled."""
        if self._ds is None:
            from .evaluate import meta_dataset
            self._ds = meta_dataset(self.meta)
        return self._ds

    def _input_text(self) -> str:
        m = self.meta
        shape = m.get("input_shape") or []
        mod = m.get("modality")
        if mod == "image":
            return f"a picture ({' × '.join(map(str, shape))})"
        if mod == "text":
            return "a text"
        if mod == "sequence":
            return f"the last {shape[0] if shape else '?'} values of a series"
        names = m.get("feature_names")
        n = int(np.prod(shape)) if shape else 0
        return f"{n} number{'s' if n != 1 else ''}" + (f" ({', '.join(map(str, names[:8]))}"
                                                        f"{', …' if len(names) > 8 else ''})" if names else "")

    def summary(self) -> str:
        """A few words for the wire's label."""
        n = self.num_weights()
        task = self.task
        if self.family == "classic":
            return self.model.describe().split(" '")[0]
        head = f"{_short(n)} weights" if n else "weights"
        if self.meta.get("objective") == "reconstruct":
            return f"{head} · rebuilds its input"
        if task == "language_model":
            try:
                return f"{head} · {self._tokenizer().vocab_size:,} tokens"
            except NBError:
                return head
        if task == "classification":
            return f"{head} · {len(self.meta.get('class_names') or [])} classes"
        if task == "control":
            agent = self.model.agent
            return f"{head} · {agent.obs_size} senses → {agent.action.size} actions"
        return head

    def details(self) -> list[list[str]]:
        """What's packed inside, as (label, explanation) rows for the Output tab."""
        rows = []
        n = self.num_weights()
        task = self.task
        m = self.meta
        if self.family == "classic":
            rows.append(["what it learned", f"a {self.model.describe().split(' ' + chr(39))[0]} (no weights — "
                                            f"it stores rules or examples instead)"])
        elif n:
            rows.append(["weights", f"{n:,} numbers that were adjusted during training"])
        if task == "language_model":
            tok = self._tokenizer()
            rows.append(["tokenizer", f"turns text into tokens: {tok.describe()}"])
            rows.append(["context", f"it can look back at most {self.context():,} tokens"])
            rows.append(["output", f"one score for each of its {tok.vocab_size:,} tokens: how likely it comes next"])
        elif task == "classification":
            names = [str(c) for c in (m.get("class_names") or [])]
            rows.append(["input", self._input_text()])
            if m.get("tokenizer"):
                rows.append(["tokenizer", self._tokenizer().describe()])
            rows.append(["output", f"one score for each of {len(names)} classes: "
                                   f"{', '.join(names[:12])}{', …' if len(names) > 12 else ''}"])
        elif task == "regression":
            rows.append(["input", self._input_text()])
            if m.get("objective") == "reconstruct":
                rows.append(["output", "the input again, rebuilt (it's an autoencoder)"])
            else:
                rows.append(["output", ", ".join(map(str, m.get("target_names") or ["a number"]))])
        elif task == "generation":
            rows.append(["input", f"{self.model.latent} random numbers (noise)"
                                  + (" and a class" if m.get("class_names") else "")])
            rows.append(["output", f"a picture ({' × '.join(map(str, m.get('input_shape') or []))})"])
        elif task == "control":
            agent = self.model.agent
            rows.append(["input", f"{agent.obs_size} senses: {', '.join(agent.obs_names[:8])}"
                                  f"{', …' if len(agent.obs_names) > 8 else ''}"])
            rows.append(["output", agent.action.describe()])
        if m.get("objective") == "reconstruct" and task == "classification":
            rows[-1] = ["output", "the input picture, rebuilt (it's an autoencoder)"]
        return rows

    # -- language models -----------------------------------------------------------
    def _ids(self, tokens, vocab: int) -> list[int]:
        vals = [tokens] if isinstance(tokens, (int, float, np.integer, np.floating)) else as_list(tokens)
        ids = []
        for v in vals:
            try:
                i = int(round(float(v)))
            except (TypeError, ValueError):
                raise NBError(f"Tokens are whole numbers, but this contains {fmt_text(v)!r}.",
                              hint="Use 'tokens of (text)' to turn text into tokens.") from None
            if not 0 <= i < vocab:
                raise NBError(f"There is no token {i}: {self.label} knows tokens 0 to {vocab - 1}.")
            ids.append(i)
        return ids

    def tokens_of(self, text) -> list[int]:
        tok = self._tokenizer()
        if isinstance(text, (list, tuple)):
            raise NBError("'tokens of' needs a text, but got a list (maybe it's tokens already?).")
        s = fmt_text(text)
        ids = [int(i) for i in tok.encode(s)]
        if s and not ids:
            self._tell("no-tokens", f"None of the characters in {s[:30]!r} were in {self.label}'s training text, "
                                    f"so it gave no tokens.")
        elif getattr(tok, "kind", "") == "characters" and len(ids) < len(s):
            self._tell("unknown-chars", f"Some characters weren't in {self.label}'s training text, so they were "
                                        f"left out of the tokens.")
        return ids

    def text_of(self, tokens) -> str:
        if isinstance(tokens, str):
            return tokens
        tok = self._tokenizer()
        return tok.decode(self._ids(tokens, tok.vocab_size))

    @torch.no_grad()
    def next_token_scores(self, tokens) -> list[float]:
        self._need("language_model", "'scores for the next token'")
        if isinstance(tokens, str):
            raise NBError("This block needs tokens (numbers), not text.", hint="Use 'tokens of (your text)' first.")
        tok = self._tokenizer()
        ids = self._ids(tokens, tok.vocab_size)
        if not ids:
            nl = tok.encode("\n")
            ids = nl[:1] or [0]
            self._tell("empty", "There were no tokens yet, so the model started from a new line.")
        ctx = self.context()
        if len(ids) > ctx:
            ids = ids[-ctx:]
            self._tell("context", f"{self.label} can only look back {ctx} tokens (its context length), so it saw "
                                  f"just the newest {ctx}.")
        net = self.model.net
        x = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        was = net.training
        net.eval()
        try:
            out = net(input_ids=x).logits if self.family == "pretrained_lm" else net(x)
        finally:
            net.train(was)
        return out[0, -1].float().cpu().tolist()

    # -- any model -----------------------------------------------------------------
    def run(self, x):
        """'run the model on …': the raw output for one input."""
        task = self.task
        if task == "generation":
            raise NBError(f"{self.label} is a picture generator, so it doesn't take an input like this.",
                          hint="Give it noise instead: 'picture drawn from noise (random noise …)'.")
        if task == "control":
            return self._policy_scores(x)
        if task == "language_model":
            if isinstance(x, str):
                self._tell("run-text", "A language model reads tokens, so the text was turned into tokens first.")
                x = self.tokens_of(x)
            return self.next_token_scores(x)
        if task is None:
            raise NBError(f"{self.label} hasn't been trained yet.")
        if self.family == "classic":
            return self._run_classic(x)
        return self._run_neural(x)

    def _encode(self, x) -> torch.Tensor:
        ds = self._dataset()
        if isinstance(x, Picture):
            if ds.modality != "image":
                raise NBError(f"{self.label} learned from {ds.modality} data, not pictures.")
            if x.drawing is not None:
                from .interactive import _drawing_to_input
                xb, _ = _drawing_to_input(self.model, *x.drawing)
                return ds.prepare_x(xb)
            return ds.prepare_x(x.fit(ds.input_shape))
        if ds.modality == "image" and not isinstance(x, (list, tuple)):
            raise NBError(f"{self.label} looks at pictures.", hint="Use 'ask for a drawing' and run it on 'answer'.")
        if ds.modality == "text" and isinstance(x, (list, tuple)):
            ids = self._ids(x, ds.tokenizer.vocab_size)[: ds.context]
            return torch.tensor([[ds.tokenizer.pad_id or 0] * (ds.context - len(ids)) + ids], dtype=torch.long)
        try:
            return ds.encode_raw(x)
        except ValueError:
            raise NBError(f"{self.label} needs {self._input_text()}, but got {fmt_text(x)[:60]!r}.") from None

    @torch.no_grad()
    def _run_neural(self, x):
        model = self.model
        if model.net is None:
            raise NBError(f"{self.label} hasn't been trained yet.")
        ds = self._dataset()
        xb = self._encode(x).to(model.device)
        net = model.net
        was = net.training
        net.eval()
        try:
            out = net(xb).float().cpu()
        finally:
            net.train(was)
        if self.meta.get("objective") == "reconstruct":
            if ds.modality == "image":
                return Picture(ds.unprepare_image(out.view(1, *ds.input_shape))[0])
            return out.reshape(-1).tolist()
        if ds.task == "classification":
            return out[0].tolist()
        vals = ds.denormalize_y(out).reshape(-1).tolist()
        return vals[0] if len(vals) == 1 else vals

    def _run_classic(self, x):
        m = self.model
        if not m.fitted:
            raise NBError(f"{self.label} hasn't been trained yet.")
        ds = self._dataset()
        xt = self._encode(x)
        X = m.features(ds, xt) if ds.modality == "text" else xt.reshape(1, -1).numpy()
        if ds.task == "classification":
            # Classic models give probabilities; their logarithms are the matching scores, so
            # 'probabilities from scores' (temperature 1) gives back exactly what the model believes.
            return np.log(np.clip(m._class_probs(X)[0], 1e-9, 1.0)).tolist()
        if m.algo == "kmeans":
            return (-m.est.transform(X)[0]).tolist()  # closer group = higher score
        v = np.asarray(m.est.predict(X), dtype=np.float32).reshape(1, -1)
        vals = ds.denormalize_y(torch.from_numpy(v)).reshape(-1).tolist()
        return vals[0] if len(vals) == 1 else vals

    def name_of(self, choice) -> str:
        i = int(round(float(number(choice))))
        task = self.task
        if task == "language_model":
            tok = self._tokenizer()
            return tok.decode(self._ids(i, tok.vocab_size))
        if task in ("classification", "generation") and self.meta.get("class_names"):
            names = self.meta["class_names"]
            if not 0 <= i < len(names):
                raise NBError(f"There is no choice {i}: {self.label} has classes 0 to {len(names) - 1}.")
            return str(names[i])
        if task == "control":
            agent = self.model.agent
            spec = agent.action
            if spec.discrete and 0 <= i < len(spec.names):
                return str(spec.names[i])
            if agent.kind == "q" and agent.choices and 0 <= i < len(agent.choices):
                return ", ".join(f"{n} {v:+g}" for n, v in zip(spec.names, agent.choices[i]))
            if 0 <= i < len(spec.names):
                return str(spec.names[i])
        return str(i)

    def _pretty(self, i: int) -> str:
        s = self.name_of(i)
        if self.task == "language_model":
            s = s.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥") or "∅"
        return s

    def show_top(self, probs, n=5, _bid=None):
        """A bar chart of the most likely choices (updates in place when run in a loop)."""
        flush()
        p = _numbers(probs, "probabilities")
        n = max(1, min(int(number(n)), 30, len(p)))
        order = np.argsort(-p, kind="stable")[:n]
        items = [{"choice": int(i), "label": self._pretty(int(i)), "p": float(p[i])} for i in order]
        scores = bool((p < 0).any() or abs(p.sum() - 1) > 0.02)
        emit("bars", key=_key(_bid), title=f"Top {n} choices · {self.label}", items=items, scores=scores,
             block=_bid)

    def fact(self, what: str):
        m = self.meta
        if what == "weights":
            return self.num_weights()
        if what == "context":
            if self.family != "pretrained_lm" and not m.get("context"):
                raise NBError(f"{self.label} doesn't read tokens, so it has no context length.")
            return self.context()
        if what == "vocab":
            return self._tokenizer().vocab_size
        if what == "classes":
            if self.task == "control":
                return self.model.agent.action.size
            return len(m.get("class_names") or [])
        if what == "inputs":
            if self.task == "control":
                return self.model.agent.obs_size
            return int(np.prod(m.get("input_shape") or [0]))
        if what == "noise":
            self._need("generation", "'noise size'")
            return int(self.model.latent)
        raise NBError(f"Unknown fact '{what}'.")

    # -- picture generators --------------------------------------------------------
    def noise(self) -> list[float]:
        self._need("generation", "'random noise'")
        return torch.randn(int(self.model.latent)).tolist()

    def picture_from(self, noise, cls="any") -> Picture:
        self._need("generation", "'picture drawn from noise'")
        g = self.model
        if not g.nets or not g.meta:
            raise NBError(f"{self.label} hasn't learned anything yet — train it on pictures first.")
        z = _numbers(noise, "noise numbers")
        if len(z) != g.latent:
            raise NBError(f"{self.label} turns {g.latent} noise numbers into a picture, but got {len(z)}.",
                          hint="Use 'random noise for the picture generator'.")
        y1h = g.class_onehot(fmt_text(cls) if not isinstance(cls, str) else cls, 1)
        names = self.meta.get("class_names") or []
        label = str(names[int(y1h[0].argmax())]) if y1h.shape[1] and names else None
        return Picture(g.decode(torch.tensor(z, dtype=torch.float32)[None], y1h)[0], label=label)

    # -- simulations -----------------------------------------------------------------
    def _agent(self):
        agent = getattr(self.model, "agent", None)
        if self.task != "control" or agent is None:
            raise NBError(f"{self.label} hasn't learned to play in a world.",
                          hint="Train it with 'train … to play in …' first.")
        return agent

    def _need_try(self) -> dict:
        if self._try is None:
            raise NBError("Start a new try first ('start a new try in …').")
        return self._try

    def start_try(self, world, _bid=None):
        from .rl import replay as R
        from .rl.world import World
        agent = self._agent()
        if not isinstance(world, World):
            raise NBError("'start a new try in' needs a world.", block_id=_bid)
        reason = agent.mismatch(world)
        if reason is not None:
            raise NBError(f"{self.label} can't play in '{world.name}': {reason}.", block_id=_bid)
        env = world.make_env()
        seed = world.seed if self._tries == 0 else (world.seed + 1000 * self._tries + 7) & 0x7FFFFFFF
        self._tries += 1
        obs = env.reset(seed)
        track = R.Track(self.label, 1.0)
        track.add(env, first=True)
        self._try = {"world": world, "env": env, "obs": obs, "track": track, "over": False, "n": 0,
                     "number": self._tries}

    def senses(self) -> list[float]:
        return np.asarray(self._need_try()["obs"], dtype=np.float64).reshape(-1).tolist()

    def _policy_scores(self, senses) -> list[float]:
        agent = self._agent()
        obs = _numbers(senses, "senses").astype(np.float32)
        if obs.size != agent.obs_size:
            raise NBError(f"{self.label} senses {agent.obs_size} numbers, but got {obs.size}.",
                          hint="Use 'what the model senses'.")
        obs = obs[None]
        if agent.kind == "table":
            return agent.qtable[int(obs[0, : agent.n_states].argmax())].tolist()
        net = agent.qnet if agent.kind == "q" else agent.actor
        return np.asarray(agent._forward(net, agent.norm(obs))[0], dtype=np.float64).tolist()

    def action_from(self, scores):
        agent = self._agent()
        s = _numbers(scores, "scores")
        if agent.kind in ("table", "q") or agent.action.discrete:
            i = int(s.argmax())
            if agent.kind == "q" and not agent.action.discrete:
                return list(agent.choices[i])
            return i
        return np.tanh(s).tolist()  # steering-like controls: each output squashed into -1 … 1

    def do(self, action):
        t = self._need_try()
        if t["over"]:
            self._tell("over", "This try is already over — start a new try to keep playing.")
            return
        env = t["env"]
        t["obs"], _, term, trunc = env.step(action)
        t["n"] += 1
        done = term or trunc or (STATE.quick and t["n"] >= 150)
        if t["n"] % env.RECORD_EVERY == 0 or done:
            t["track"].add(env)
        if done:
            t["over"] = True

    def try_over(self) -> bool:
        return bool(self._need_try()["over"])

    def try_result(self, what="score"):
        env = self._need_try()["env"]
        if what == "distance":
            return float(env.progress())
        if what == "steps":
            return int(env.steps)
        return float(env.ep_return)

    def show_try(self, _bid=None):
        from .rl import replay as R
        from .rl.api import _result_text
        t = self._need_try()
        env, world = t["env"], t["world"]
        info = env.episode_info()
        t["track"].stats = info
        res = _result_text(world, info)
        overlay = env.overlay(self.model.agent) if world.kind == "maze" else None
        rep = R.build(world, env.scene(), [t["track"]], title=f"{self.label} · output try {t['number']} · {res}",
                      block=_bid, dt=R.frame_dt(env), stats=info, overlay=overlay)
        R.emit_replay(rep)
        extra = "" if res.startswith("reward") else f" (reward {info['reward']:.1f})"
        emit("log", level="info", text=f"{self.label}'s try {t['number']} in '{world.name}': {res}{extra}. "
                                        f"Watch it in the Sim tab.", block=_bid)


# ---------------------------------------------------------------------------
# Blocks: the 📦 weights block, start / end of an output stack
# ---------------------------------------------------------------------------


def _trained(model) -> bool:
    fam = getattr(model, "family", None)
    if fam == "classic":
        return bool(getattr(model, "fitted", False))
    if fam == "generator":
        return bool(model.nets and model.meta)
    if fam == "pretrained_lm":
        return True
    return getattr(model, "net", None) is not None or getattr(model, "agent", None) is not None


def weights(model, _bid=None) -> Weights:
    """The '📦 weights of trained …' block: pack up what training produced."""
    if getattr(model, "family", None) is None:
        raise NBError("The 📦 weights block needs a model.", block_id=_bid)
    if not _trained(model):
        raise NBError(f"'{model.name}' hasn't been trained yet, so it has no weights to hand over.",
                      hint="Put a 'train' block above the 📦 weights block (or load a trained model from a file).",
                      block_id=_bid)
    if model.family == "neural" and not getattr(model, "steps_done", 0) and getattr(model, "agent", None) is None:
        emit("log", level="warn", text=f"'{model.name}' hasn't been trained, so its weights are still random.",
             block=_bid)
    w = Weights(model, _bid)
    emit("weights", block=_bid, name=w.name, label=w.label, family=w.family, task=w.task, summary=w.summary(),
         details=w.details())
    return w


def start_output(w, _bid=None):
    flush()
    if w is None:
        raise NBError("The 📦 weights block wired to this output stack never ran, so there is nothing to use yet.",
                      hint="Keep the weights block at the very bottom of the training stack (not inside an 'if').",
                      block_id=_bid)
    if not isinstance(w, Weights):
        raise NBError("An output stack needs the trained weights from a 📦 weights block.", block_id=_bid)
    _OUT.current, _OUT.block = w, _bid
    _OUT.runs += 1
    emit("output_start", block=_bid, source=w.bid, name=w.name, label=w.label, task=w.task, summary=w.summary(),
         details=w.details())


def end_output(w=None):
    flush()
    if _OUT.current is not None:
        emit("output_end", block=_OUT.block, source=_OUT.current.bid)
    _OUT.current, _OUT.block = None, None


def wired() -> Weights:
    """The weights of the output stack that is running (used inside My Blocks)."""
    if _OUT.current is None:
        raise NBError("This block uses a trained model, so it only works while an output stack is running.",
                      hint="Use this My Block under a '○ start output' block.")
    return _OUT.current


# ---------------------------------------------------------------------------
# Scores → probabilities → a choice
# ---------------------------------------------------------------------------


def softmax(scores, temperature=1.0) -> list[float]:
    """Raw scores → probabilities that add up to 1 (``temperature`` 0 = all on the top score)."""
    s = _numbers(scores, "scores")
    t = float(number(temperature))
    if t <= 1e-6:
        p = np.zeros_like(s)
        p[int(s.argmax())] = 1.0
        return p.tolist()
    z = s / t
    z -= z.max()
    e = np.exp(z)
    return (e / e.sum()).tolist()


def pick(probs, how="random") -> int:
    """Choose one option: at random by probability (sampling), or the most likely one."""
    p = _numbers(probs, "probabilities")
    if (p < 0).any():
        raise NBError("Probabilities can't be negative — these look like raw scores.",
                      hint="Turn them into probabilities first with 'probabilities from scores'.")
    if how == "best":
        return int(p.argmax())
    total = float(p.sum())
    if total <= 0:
        raise NBError("All the probabilities are 0, so nothing can be picked.")
    cum = np.cumsum(p)
    return min(int(np.searchsorted(cum, _random.random() * total, side="right")), len(p) - 1)


def chance_of(probs, choice) -> float:
    p = _numbers(probs, "probabilities")
    i = int(round(float(number(choice))))
    if not 0 <= i < len(p):
        raise NBError(f"There is no choice {i}: the choices go from 0 to {len(p) - 1}.")
    return float(p[i])


# ---------------------------------------------------------------------------
# Showing results in the Output tab
# ---------------------------------------------------------------------------


def _key(bid) -> str:
    """Results of the same block replace each other — until the next 'ask' or output stack."""
    return f"{bid or 'out'}:{_OUT.runs}:{_OUT.asks}"


def flush():
    """Send show_text updates that were held back."""
    for ev in list(_OUT.pending.values()):
        emit("output_text", **ev)
    _OUT.pending.clear()


def show_text(value, _bid=None):
    key = _key(_bid)
    ev = {"key": key, "text": fmt_text(value), "block": _bid}
    now = time.time()
    if now - _OUT.sent.get(key, 0.0) >= 0.05:
        _OUT.pending.pop(key, None)
        _OUT.sent[key] = now
        emit("output_text", **ev)
    else:
        _OUT.pending[key] = ev


def show_picture(pic, _bid=None):
    flush()
    if not isinstance(pic, Picture):
        raise NBError(f"'show picture' needs a picture, but got {fmt_text(pic)[:40]!r}.",
                      hint="Pictures come from 'ask for a drawing', 'picture drawn from noise' or an autoencoder.")
    png = pic.png()
    ev = {"key": _key(_bid), "png": png, "caption": pic.describe(), "label": pic.label, "block": _bid}
    em = emitter()
    if em.mode == "cli" and em.run_dir is not None:
        d = em.run_dir / "images"
        d.mkdir(parents=True, exist_ok=True)
        _OUT.pictures += 1
        path = d / f"output-{_OUT.runs}-picture-{_OUT.pictures}.png"
        path.write_bytes(base64.b64decode(png))
        ev["saved"] = str(path)
    emit("output_picture", **ev)


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def _wait_for_answer(sid: str, kind: str, question: str, bid):
    emit("ask", id=sid, question=question, kind=kind, block=bid)
    while not STATE.interact_q.empty():  # drop stale messages
        try:
            STATE.interact_q.get_nowait()
        except Exception:
            break
    while True:
        msg = STATE.interact_q.get()
        if msg.get("cmd") == "interact_end":
            return None
        data = msg.get("data") or {}
        if data.get("ask") == sid:
            return data


def ask(question="", _bid=None):
    """'ask … and wait': a question with a text box (Output tab); the reply goes into 'answer'."""
    flush()
    q = fmt_text(question)
    _OUT.asks += 1
    if gui_mode() and STATE.stdin_open:
        sid = f"ask-{time.time_ns()}"
        data = _wait_for_answer(sid, "text", q, _bid)
        _OUT.answer = str((data or {}).get("answer", ""))
        emit("ask_done", id=sid, answer=_OUT.answer)
        return _OUT.answer
    emit("ask", id=None, question=q, kind="text", block=_bid, headless=True)  # (the terminal prints charts first)
    if _tty():
        try:
            _OUT.answer = input(f"{q} ")
        except EOFError:
            _OUT.answer = ""
    else:
        _OUT.answer = ""
        emit("log", level="info", text=f"(No keyboard here, so the answer to {q!r} is empty.)", block=_bid)
    return _OUT.answer


def ask_drawing(question="", _bid=None):
    """'ask for a drawing … and wait': a drawing pad; the drawing goes into 'answer' as a picture."""
    flush()
    q = fmt_text(question)
    _OUT.asks += 1
    blank = Picture(torch.zeros(1, 28, 28))
    if gui_mode() and STATE.stdin_open:
        sid = f"ask-{time.time_ns()}"
        data = _wait_for_answer(sid, "draw", q, _bid) or {}
        size = int(data.get("size") or 28)
        pixels = data.get("pixels") or []
        if len(pixels) == size * size:
            img = np.asarray(pixels, dtype=np.float32).reshape(size, size)
            _OUT.answer = Picture(img, drawing=(pixels, size))
        else:
            _OUT.answer = blank
        emit("ask_done", id=sid, answer=fmt_text(_OUT.answer), png=_OUT.answer.png())
    else:
        emit("ask", id=None, question=q, kind="draw", block=_bid, headless=True)
        _OUT.answer = blank
        emit("log", level="info", text="(Drawings only work in the NeuroBlocks editor, so the drawing is blank.)",
             block=_bid)
    return _OUT.answer


def answer():
    return _OUT.answer
