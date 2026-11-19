"""Generate text with a trained mini-GPT.

    python generate.py "Once upon a time"
    python generate.py "Tom had a red ball." --strategy greedy
    python generate.py "One day" --strategy top_k --top-k 10 --temperature 1.2 --seed 7
    python generate.py "Lily" --compare        # same prompt with every decoding strategy
"""

import argparse
from pathlib import Path

from minigpt.sampling import SamplingConfig, generate
from minigpt.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", default="Once upon a time")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--strategy", choices=["greedy", "temperature", "top_k", "top_p"], default="top_p")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--compare", action="store_true", help="run every strategy on the prompt")
    a = p.parse_args()

    if not Path(a.checkpoint).exists():
        raise SystemExit(f"no checkpoint at {a.checkpoint}; run `python train.py` first")
    model, tok, meta = load_checkpoint(a.checkpoint)
    ids = tok.encode(a.prompt)
    strategies = ["greedy", "temperature", "top_k", "top_p"] if a.compare else [a.strategy]
    for s in strategies:
        cfg = SamplingConfig(strategy=s, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                             repetition_penalty=a.repetition_penalty, seed=a.seed)
        out = generate(model, ids, a.max_new_tokens, cfg, stop_ids=(tok.eot_id,))
        if a.compare:
            print(f"--- {s} ---")
        print(a.prompt + tok.decode(out))


if __name__ == "__main__":
    main()
