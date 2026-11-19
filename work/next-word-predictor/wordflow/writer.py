"""Interactive story-writing session on top of the n-gram predictor.

A :class:`WritingSession` holds the story so far. At every step it offers a handful of
next-word suggestions (the most probable ones, or -- in creative mode -- words drawn at
random from the model's distribution), and the writer accepts one, types their own
words, asks the model to continue on its own, or undoes the last step.

The session has no I/O, so it is easy to test; :func:`run_writer` is the thin
terminal loop around it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from .generation import TextGenerator
from .ngram_model import NGramModel
from .predictor import NextWordPredictor

POOL_SIZE = 60   # candidates considered when drawing creative suggestions

HELP = """\
  1-9      accept that suggestion
  <text>   type your own words (they are appended to the story)
  r        new suggestions (re-draws them in creative mode)
  m        switch between top-probability and creative suggestions
  a [n]    let the model write the next n words (default 15)
  u        undo the last step
  s FILE   save the story to FILE
  ?        show this help
  q        quit"""


class WritingSession:
    """The story so far plus the model settings used to suggest what comes next."""

    def __init__(
        self,
        model: NGramModel,
        opening: str = "",
        *,
        k: float = 0.01,
        order: int | None = None,
        top_k: int = 5,
        temperature: float = 1.0,
        creative: bool = False,
        seed: int | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.predictor = NextWordPredictor(model)
        self.generator = TextGenerator(model)
        self.k = k
        self.order = min(order or model.order, model.order)
        self.top_k = top_k
        self.temperature = temperature
        self.creative = creative
        self.rng = random.Random(seed)
        self.text = opening.strip()
        self._undo: list[str] = []

    # -- suggestions ---------------------------------------------------------------

    def suggestions(self) -> list[tuple[str, float]]:
        """``(word, probability)`` pairs to offer for the current story."""
        pool = POOL_SIZE if self.creative else self.top_k
        result = self.predictor.suggest_next(
            self.text, top_k=pool, k=self.k, order=self.order
        )
        ranked = [(s.word, s.probability) for s in result.suggestions]
        if not self.creative or len(ranked) <= self.top_k:
            return ranked[: self.top_k]
        return self._draw(ranked)

    def _draw(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Weighted draw without replacement; ``temperature`` flattens the weights."""
        pool = list(ranked)
        chosen: list[tuple[str, float]] = []
        while pool and len(chosen) < self.top_k:
            weights = [p ** (1.0 / self.temperature) for _, p in pool]
            index = self.rng.choices(range(len(pool)), weights=weights)[0]
            chosen.append(pool.pop(index))
        return chosen

    # -- edits ---------------------------------------------------------------------

    def _push(self) -> None:
        self._undo.append(self.text)

    def append(self, words: str) -> None:
        """Append the writer's own words to the story."""
        words = words.strip()
        if not words:
            return
        self._push()
        self.text = f"{self.text} {words}".strip()

    def accept(self, word: str) -> None:
        self.append(word)

    def auto(self, count: int = 15) -> str:
        """Let the model sample ``count`` more words; returns just the new text."""
        if count < 1:
            raise ValueError("count must be at least 1")
        result = self.generator.generate_text(
            self.text, max_words=count, temperature=self.temperature,
            # k=0 samples only continuations seen in the corpus; add-k would hand
            # probability to the whole vocabulary after a sparse context.
            order=self.order, seed=self.rng.randrange(2**31), k=0.0, stop_at_eos=False,
        )
        self._push()
        before = self.text
        self.text = result.text
        return self.text[len(before):].strip() if self.text.startswith(before) else self.text

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.text = self._undo.pop()
        return True

    def toggle_creative(self) -> bool:
        self.creative = not self.creative
        return self.creative

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.text + "\n", encoding="utf-8")
        return path


def _show(session: WritingSession, out: Callable[[str], None]) -> list[str]:
    mode = "creative" if session.creative else "most likely"
    out("")
    out(f"  {session.text or '(empty story)'}")
    out("")
    options = session.suggestions()
    for number, (word, probability) in enumerate(options, 1):
        out(f"  {number}. {word:<16s} {probability:7.2%}")
    out(f"  [{mode} suggestions - ? for help]")
    return [word for word, _ in options]


def run_writer(
    session: WritingSession,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run the interactive loop until the writer quits or input ends."""
    output_fn("Creative Writing Prompter - build a story one word at a time. (? = help)")
    try:
        if not session.text:
            session.append(input_fn("Start your story: "))
        while True:
            options = _show(session, output_fn)
            try:
                line = input_fn("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            command, _, rest = line.partition(" ")
            lowered = command.lower()
            if lowered in {"q", "quit", "exit"}:
                break
            if lowered == "?":
                output_fn(HELP)
            elif lowered == "r":
                continue
            elif lowered == "m":
                session.toggle_creative()
            elif lowered == "u":
                if not session.undo():
                    output_fn("  nothing to undo")
            elif lowered == "a":
                try:
                    count = int(rest) if rest.strip() else 15
                    session.auto(count)
                except ValueError:
                    output_fn("  usage: a [number of words]")
            elif lowered == "s":
                if not rest.strip():
                    output_fn("  usage: s FILE")
                else:
                    try:
                        output_fn(f"  saved to {session.save(rest.strip())}")
                    except OSError as error:
                        output_fn(f"  could not save: {error.strerror or error}")
            elif line.isdigit():
                number = int(line)
                if 1 <= number <= len(options):
                    session.accept(options[number - 1])
                else:
                    output_fn(f"  pick a number from 1 to {len(options)}")
            else:
                session.append(line)
    except (EOFError, KeyboardInterrupt):
        output_fn("")
    output_fn("")
    output_fn("Your story:")
    output_fn(session.text)
