"""Byte-level Byte-Pair Encoding (BPE) tokenizer, written from scratch.

How it works
------------
1. Text is first cut into "chunks" with a regex (words with their leading space,
   numbers, punctuation runs, whitespace). Merges never cross a chunk boundary, so
   " cat" and "cat." do not fuse into odd tokens.
2. Every chunk is turned into its UTF-8 bytes. The 256 possible byte values are the
   base vocabulary, so ANY string can be encoded: there is no unknown token.
3. Training repeatedly finds the most frequent adjacent pair of token ids across the
   corpus and replaces it with a new id (256, 257, ...). Each replacement is a "merge".
4. Encoding replays the learned merges in the order they were learned (lowest rank
   first). Decoding concatenates the bytes of each id and decodes UTF-8.

Special tokens (e.g. ``<|endoftext|>``) get ids after the merges and are never split.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# GPT-2 style pre-tokenisation, restricted to what the stdlib `re` module supports.
SPLIT_PATTERN = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)
EOT = "<|endoftext|>"

Pair = Tuple[int, int]


def _count_pairs(ids: Sequence[int], weight: int, counts: Counter) -> None:
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight


def _merge(ids: Sequence[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of ``pair`` (left to right)."""
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: Dict[Pair, int] = {}          # (a, b) -> new id, insertion order = rank
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: Dict[str, int] = {}
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int, special_tokens: Iterable[str] = (EOT,),
              verbose: bool = False) -> "BPETokenizer":
        specials = list(dict.fromkeys(special_tokens))
        num_merges = vocab_size - 256 - len(specials)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(specials)}")

        # Special tokens are not part of the statistics.
        for tok in specials:
            text = text.replace(tok, "\n")

        # Identical chunks are merged identically, so count each distinct chunk once
        # and weight it by its frequency: much faster than working on the raw stream.
        chunk_freq = Counter(SPLIT_PATTERN.findall(text))
        words: List[List[int]] = [list(c.encode("utf-8")) for c in chunk_freq]
        freqs: List[int] = list(chunk_freq.values())

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for step in range(num_merges):
            counts: Counter = Counter()
            for ids, f in zip(words, freqs):
                if len(ids) > 1:
                    _count_pairs(ids, f, counts)
            if not counts:
                break  # every chunk is a single token already
            # Ties are broken deterministically (smallest pair) so training is reproducible.
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[pair] < 2:
                break  # merging a pair seen once only memorises, it does not compress
            new_id = 256 + len(self.merges)
            words = [_merge(ids, pair, new_id) if len(ids) > 1 else ids for ids in words]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (step < 10 or step % 100 == 0):
                print(f"merge {step:4d}: {pair} -> {new_id} {self.vocab[new_id]!r} ({counts[pair]}x)")

        self.special_tokens = {}
        for tok in specials:
            sid = len(self.vocab)
            self.special_tokens[tok] = sid
            self.vocab[sid] = tok.encode("utf-8")
        self._cache = {}
        return self

    # ------------------------------------------------------------------ encode/decode
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge that is present (that is the training order).
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 50_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        out: List[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        if not allow_special or not self.special_tokens:
            return self.encode_ordinary(text)
        pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        out: List[int] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        try:
            return b"".join(self.vocab[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"unknown token id {exc.args[0]}") from None

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """Readable form of a single token (may be a partial UTF-8 character)."""
        return self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "type": "byte-bpe",
            "merges": [[a, b] for (a, b) in self.merges],  # order = rank
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        tok = cls()
        for a, b in data["merges"]:
            new_id = 256 + len(tok.merges)
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        for name, sid in sorted(data["special_tokens"].items(), key=lambda kv: kv[1]):
            if sid != len(tok.vocab):
                raise ValueError("special token ids must follow the merges")
            tok.special_tokens[name] = sid
            tok.vocab[sid] = name.encode("utf-8")
        return tok

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
