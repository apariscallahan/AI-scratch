"""Tokenizers: how text is split into the pieces ("tokens") a model reads and writes."""
from __future__ import annotations

import re
from collections import Counter

from .errors import NBError, missing_package

_WORD_RE = re.compile(r"\n|[A-Za-z0-9_']+|[^\sA-Za-z0-9_']")
_NO_SPACE_BEFORE = set(".,!?;:)]}%'\"")
_NO_SPACE_AFTER = set("([{\"$#")


def _chunks(text: str, size: int = 1 << 20):
    """Split a long text into ~1 MB pieces at whitespace (so no token is cut in half)."""
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + size)
        if j < n:
            k = text.rfind(" ", i, j)
            if k > i:
                j = k
        yield text[i:j]
        i = j


class Tokenizer:
    kind = "base"
    pad_id: int | None = None

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def encode_array(self, text: str):
        """Encode a (possibly huge) text into a compact numpy array of token ids."""
        import numpy as np
        if len(text) < (1 << 20):
            return np.asarray(self.encode(text), dtype=np.int64)
        parts = [np.asarray(self.encode(t), dtype=np.int32) for t in _chunks(text)]
        return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)

    def decode(self, ids) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.kind} tokenizer ({self.vocab_size:,} tokens)"


class CharTokenizer(Tokenizer):
    """Every character is a token. Small vocabulary, long sequences — great for learning."""

    kind = "characters"

    def __init__(self, chars: list[str], pad: bool = False, lowercase: bool = False):
        self.pad = pad
        self.lowercase = lowercase
        self.chars = (["\x00"] if pad and (not chars or chars[0] != "\x00") else []) + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.pad_id = 0 if pad else None
        self.unknown_seen = 0

    @classmethod
    def fit(cls, text: str, pad: bool = False, lowercase: bool = False):
        if lowercase:
            text = text.lower()
        return cls(sorted(set(text)), pad=pad, lowercase=lowercase)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        if self.lowercase:
            text = text.lower()
        out = []
        for c in text:
            i = self.stoi.get(c)
            if i is None:
                self.unknown_seen += 1
                continue
            out.append(i)
        return out

    def encode_array(self, text: str):
        """Vectorised character lookup — fast even for hundreds of megabytes of text."""
        import numpy as np
        if self.lowercase:
            text = text.lower()
        cps = np.frombuffer(text.encode("utf-32-le", errors="replace"), dtype=np.uint32)
        keys = np.array([ord(c) for c in self.chars], dtype=np.uint32)
        order = np.argsort(keys)
        skeys = keys[order]
        pos = np.clip(np.searchsorted(skeys, cps), 0, len(skeys) - 1)
        found = skeys[pos] == cps
        self.unknown_seen += int((~found).sum())
        return order[pos[found]].astype(np.int64)

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids if 0 <= int(i) < len(self.chars) and
                       not (self.pad and int(i) == 0))

    def to_dict(self):
        return {"kind": "characters", "chars": self.chars[1:] if self.pad else self.chars, "pad": self.pad,
                "lowercase": self.lowercase}


class WordTokenizer(Tokenizer):
    """Every word (or punctuation mark) is a token; rare words become <unk>."""

    kind = "words"

    def __init__(self, vocab: list[str], lowercase: bool = False):
        self.vocab = list(vocab)
        if not self.vocab or self.vocab[0] != "<pad>":
            self.vocab = ["<pad>", "<unk>"] + [w for w in self.vocab if w not in ("<pad>", "<unk>")]
        self.stoi = {w: i for i, w in enumerate(self.vocab)}
        self.lowercase = lowercase
        self.pad_id = 0
        self.unk_id = 1

    @staticmethod
    def split(text: str, lowercase: bool = False) -> list[str]:
        if lowercase:
            text = text.lower()
        return _WORD_RE.findall(text)

    @classmethod
    def fit(cls, text_or_texts, max_vocab: int = 5000, lowercase: bool = False):
        counts: Counter = Counter()
        if isinstance(text_or_texts, str):
            counts.update(cls.split(text_or_texts, lowercase))
        else:
            for t in text_or_texts:
                counts.update(cls.split(t, lowercase))
        vocab = [w for w, _ in counts.most_common(max(10, int(max_vocab) - 2))]
        return cls(vocab, lowercase=lowercase)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        return [self.stoi.get(w, self.unk_id) for w in self.split(text, self.lowercase)]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == self.pad_id or not 0 <= i < len(self.vocab):
                continue
            w = self.vocab[i]
            if w == "\n":
                out.append("\n")
                continue
            if out and out[-1] != "\n" and w not in _NO_SPACE_BEFORE and not (out[-1] and out[-1][-1] in _NO_SPACE_AFTER):
                out.append(" ")
            out.append(w)
        return "".join(out)

    def to_dict(self):
        return {"kind": "words", "vocab": self.vocab, "lowercase": self.lowercase}


class GPT2Tokenizer(Tokenizer):
    """The byte-pair-encoding tokenizer used by GPT-2 (50,257 sub-word tokens)."""

    kind = "gpt2"

    def __init__(self):
        try:
            import tiktoken
        except ImportError:
            raise missing_package("tiktoken", "The GPT-2 sub-word tokenizer") from None
        self.enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])

    def to_dict(self):
        return {"kind": "gpt2"}


class HFTokenizer(Tokenizer):
    """Wraps a Hugging Face tokenizer (used with pretrained models)."""

    kind = "huggingface"

    def __init__(self, model_id: str, tok=None):
        self.model_id = model_id
        if tok is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise missing_package("transformers", "Pretrained Hugging Face models") from None
            tok = AutoTokenizer.from_pretrained(model_id)
        self.tok = tok
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @property
    def vocab_size(self):
        return len(self.tok)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=True)

    def to_dict(self):
        return {"kind": "huggingface", "model_id": self.model_id}


def make_tokenizer(kind: str, text, *, max_vocab: int = 5000, lowercase: bool = False,
                   pad: bool = False) -> Tokenizer:
    kind = (kind or "characters").lower()
    if kind in ("char", "chars", "characters"):
        if isinstance(text, str):
            return CharTokenizer.fit(text, pad=pad, lowercase=lowercase)
        chars = sorted(set("".join((t.lower() if lowercase else t) for t in text)))
        return CharTokenizer(chars, pad=pad, lowercase=lowercase)
    if kind in ("word", "words"):
        return WordTokenizer.fit(text, max_vocab=max_vocab, lowercase=lowercase)
    if kind in ("gpt2", "bpe", "subwords"):
        return GPT2Tokenizer()
    raise NBError(f"Unknown tokenizer '{kind}'.", hint="Use characters, words or gpt2.")


def tokenizer_from_dict(d: dict) -> Tokenizer:
    kind = d.get("kind")
    if kind == "characters":
        return CharTokenizer(d["chars"], pad=d.get("pad", False), lowercase=d.get("lowercase", False))
    if kind == "words":
        return WordTokenizer(d["vocab"], lowercase=d.get("lowercase", False))
    if kind == "gpt2":
        return GPT2Tokenizer()
    if kind == "huggingface":
        return HFTokenizer(d["model_id"])
    raise NBError(f"Can't rebuild tokenizer of kind {kind!r}.")
