"""Lesson 3 and 4: the label vocabulary.

Classification becomes generation. Every LEDGAR clause type gets its own new token, e.g.
``<Governing Laws>``, and one separator token ``<Label>`` marks where the answer starts:

    {clause text}<Label><Governing Laws><|endoftext|>

Because each label is a single token, the model answers in exactly one generation step,
and "which type is most likely?" is just an argmax over 100 rows of the output layer.
"""

from __future__ import annotations

from typing import List

SEP = "<Label>"
EOS = "<|endoftext|>"


def label_token(name: str) -> str:
    """'Governing Laws' -> '<Governing Laws>'."""
    return f"<{name}>"


def label_name(token: str) -> str:
    """'<Governing Laws>' -> 'Governing Laws'."""
    if not (token.startswith("<") and token.endswith(">")):
        raise ValueError(f"not a label token: {token!r}")
    return token[1:-1]


def new_tokens(label_names: List[str]) -> List[str]:
    """The tokens added to GPT-2's vocabulary: the separator, then one per label."""
    return [SEP] + [label_token(n) for n in label_names]


def prompt(clause: str) -> str:
    """What the model sees at inference time; it must continue with a label token."""
    return clause.strip() + SEP


def training_text(clause: str, name: str) -> str:
    return prompt(clause) + label_token(name) + EOS
