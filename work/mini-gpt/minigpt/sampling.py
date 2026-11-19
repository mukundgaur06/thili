"""Autoregressive decoding: feed the model its own output, one token at a time.

Strategies (all operate on the logits of the LAST position):
  greedy       argmax, deterministic, tends to loop
  temperature  logits / T, then sample  (T < 1 sharper, T > 1 flatter)
  top-k        keep the k most likely tokens, renormalise, sample
  top-p        keep the smallest set whose cumulative probability >= p (nucleus), sample
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import torch

from .model import GPT


@dataclass
class SamplingConfig:
    strategy: str = "top_p"          # greedy | temperature | top_k | top_p
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.0  # > 1 discourages tokens already in the context
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.strategy not in ("greedy", "temperature", "top_k", "top_p"):
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0 (use strategy='greedy' for argmax)")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")


def apply_repetition_penalty(logits: torch.Tensor, context: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or context.numel() == 0:
        return logits
    logits = logits.clone()
    seen = torch.unique(context)
    vals = logits[seen]
    logits[seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, order = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    # Drop a token if the tokens BEFORE it already reach p (the top token always stays).
    drop_sorted = (cum - probs) >= p
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def next_token_distribution(logits: torch.Tensor, cfg: SamplingConfig,
                            context: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Turn raw last-position logits (vocab,) into the distribution actually sampled from."""
    logits = logits.float()
    if context is not None:
        logits = apply_repetition_penalty(logits, context, cfg.repetition_penalty)
    if cfg.strategy == "greedy":
        out = torch.zeros_like(logits)
        out[torch.argmax(logits)] = 1.0
        return out
    logits = logits / cfg.temperature
    if cfg.strategy == "top_k":
        logits = top_k_filter(logits, cfg.top_k)
    elif cfg.strategy == "top_p":
        logits = top_p_filter(logits, cfg.top_p)
    return torch.softmax(logits, dim=-1)


@dataclass
class Step:
    token: int
    prob: float                              # probability of the chosen token (after filtering)
    model_prob: float                        # raw model probability (T = 1, no filtering)
    candidates: List[tuple] = field(default_factory=list)   # [(token, prob), ...]


@torch.no_grad()
def stream(model: GPT, prompt_ids: List[int], max_new_tokens: int, cfg: SamplingConfig,
           stop_ids: tuple = (), n_candidates: int = 5) -> Iterator[Step]:
    """Yield one Step per generated token. The context is cropped to block_size."""
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu")
    if cfg.seed is not None:
        gen.manual_seed(cfg.seed)
    else:
        gen.seed()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if ids.size(1) == 0:
        raise ValueError("prompt must contain at least one token")
    for _ in range(max_new_tokens):
        ctx = ids[:, -model.cfg.block_size:]
        logits, _ = model(ctx)
        last = logits[0, -1]
        probs = next_token_distribution(last, cfg, ctx[0])
        if cfg.strategy == "greedy":
            nxt = int(torch.argmax(probs))
        else:
            nxt = int(torch.multinomial(probs.cpu(), 1, generator=gen))
        raw = torch.softmax(last.float(), dim=-1)
        top = torch.topk(probs, min(n_candidates, probs.numel()))
        yield Step(nxt, float(probs[nxt]), float(raw[nxt]),
                   [(int(i), float(p)) for p, i in zip(top.values, top.indices) if p > 0])
        if nxt in stop_ids:
            return
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)


def generate(model: GPT, prompt_ids: List[int], max_new_tokens: int, cfg: SamplingConfig,
             stop_ids: tuple = ()) -> List[int]:
    """Return only the newly generated ids (a stop token, if hit, is not included)."""
    out = []
    for step in stream(model, prompt_ids, max_new_tokens, cfg, stop_ids):
        if step.token in stop_ids:
            break
        out.append(step.token)
    return out
