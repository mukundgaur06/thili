"""Injecting, saving, merging and swapping LoRA adapters."""

from __future__ import annotations

import io
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .layers import Conv1D, LoRALinear


def _parent(model: nn.Module, name: str):
    parts = name.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]


def inject_lora(model: nn.Module, targets: Iterable[str] = ("c_attn",), r: int = 8,
                alpha: float = 16.0, dropout: float = 0.0) -> List[str]:
    """Replace every Linear/Conv1D whose name ends with a target by a LoRALinear.

    Returns the qualified names that were wrapped. Freezes *all* other parameters
    except ones the caller re-enables (e.g. a classification head).
    """
    targets = tuple(targets)
    wrapped = []
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Linear, Conv1D)) and name.split(".")[-1] in targets:
            parent, attr = _parent(model, name)
            if isinstance(parent, LoRALinear):
                continue
            setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            wrapped.append(name)
    if not wrapped:
        raise ValueError(f"no layer matched targets {targets}")
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    return wrapped


def lora_layers(model: nn.Module) -> Dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "percent": 100.0 * trainable / total}


# --------------------------------------------------------------------------- adapters
def adapter_state_dict(model: nn.Module, extra: Iterable[str] = ()) -> Dict[str, torch.Tensor]:
    """Only the LoRA matrices (plus any extra trainable modules, e.g. 'score')."""
    extra = tuple(extra)
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if "lora_" in k or (extra and k.split(".")[0] in extra)}


def load_adapter(model: nn.Module, adapter: Dict[str, torch.Tensor]) -> None:
    layers = lora_layers(model)
    if any(m.merged for m in layers.values()):
        raise RuntimeError("unmerge before loading a different adapter")
    own = model.state_dict()
    unknown = [k for k in adapter if k not in own]
    if unknown:
        raise KeyError(f"adapter keys not in model: {unknown[:3]}")
    for k, v in adapter.items():
        if own[k].shape != v.shape:
            raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(own[k].shape)}")
    model.load_state_dict(adapter, strict=False)


def save_adapter(path: str, model: nn.Module, meta: dict, extra: Iterable[str] = (),
                 dtype: torch.dtype = torch.float16) -> int:
    """Save to .safetensors (fp16 by default, like PEFT checkpoints); returns the size in bytes.

    Training happens in fp32; storing the small A/B matrices in 16 bits halves the file and
    is cast back to the model's dtype by ``load_adapter``.
    """
    from safetensors.torch import save_file
    sd = {k: v.to(dtype).contiguous() for k, v in adapter_state_dict(model, extra).items()}
    save_file(sd, path, metadata={k: str(v) for k, v in meta.items()})
    import os
    return os.path.getsize(path)


def read_adapter(path: str):
    from safetensors import safe_open
    from safetensors.torch import load_file
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
    return load_file(path), meta


def adapter_bytes(adapter: Dict[str, torch.Tensor]) -> int:
    buf = io.BytesIO()
    torch.save(adapter, buf)
    return buf.getbuffer().nbytes


# --------------------------------------------------------------------------- merge / swap
def merge_lora(model: nn.Module) -> int:
    for m in lora_layers(model).values():
        m.merge()
    return len(lora_layers(model))


def unmerge_lora(model: nn.Module) -> None:
    for m in lora_layers(model).values():
        m.unmerge()


def strip_lora(model: nn.Module) -> nn.Module:
    """Merge every adapter and replace each LoRALinear by its base layer, in place."""
    for name, m in list(lora_layers(model).items()):
        m.merge()
        parent, attr = _parent(model, name)
        setattr(parent, attr, m.base)
    return model


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for m in lora_layers(model).values():
        m.enabled = enabled


class AdapterBank:
    """One frozen base model, many adapters; switch between them in O(adapter size).

    This is how multi-tenant LoRA serving works: the 500 MB base stays in memory, and each
    task/customer is a few hundred KB of A/B matrices that are loaded on request.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.active: str | None = None

    def add(self, name: str, adapter: Dict[str, torch.Tensor]) -> None:
        self.adapters[name] = adapter

    def activate(self, name: str) -> None:
        if name not in self.adapters:
            raise KeyError(name)
        set_lora_enabled(self.model, True)
        load_adapter(self.model, self.adapters[name])
        self.active = name

    def deactivate(self) -> None:
        """Back to the pure base model (LoRA path switched off)."""
        set_lora_enabled(self.model, False)
        self.active = None
