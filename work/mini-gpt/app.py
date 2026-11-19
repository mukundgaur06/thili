"""mini-GPT Playground: an interactive window into the trained model.

    python app.py            # http://127.0.0.1:8600
    python app.py --open --checkpoint checkpoints/best.pt

Standard-library HTTP server + one static page. Endpoints (JSON):
  GET  /api/info        model config, parameter counts, training history, baselines
  POST /api/tokenize    {text}                      -> BPE pieces with ids
  POST /api/next        {text, strategy, ...}       -> top next-token probabilities (raw + filtered)
  POST /api/generate    {prompt, strategy, ...}     -> tokens with per-step probability & candidates
  POST /api/attention   {text}                      -> attention weights for every layer / head
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from minigpt.sampling import SamplingConfig, next_token_distribution, stream
from minigpt.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
LOCK = threading.Lock()   # torch inference is not re-entrant-friendly on tiny CPUs; serialise it
STATE: dict = {}


class BadRequest(ValueError):
    pass


def _num(p: dict, key: str, default, lo, hi, cast=float):
    v = p.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a number")
    if not (lo <= v <= hi) or (isinstance(v, float) and not math.isfinite(v)):
        raise BadRequest(f"{key} must be between {lo} and {hi}")
    return v


def _text(p: dict, key: str, required=True, limit=2000) -> str:
    v = p.get(key, "")
    if not isinstance(v, str):
        raise BadRequest(f"{key} must be a string")
    if len(v) > limit:
        raise BadRequest(f"{key} is longer than {limit} characters")
    if required and not v.strip():
        raise BadRequest(f"{key} must not be empty")
    return v


def _sampling(p: dict) -> SamplingConfig:
    strategy = p.get("strategy", "top_p")
    seed = p.get("seed")
    try:
        return SamplingConfig(
            strategy=strategy,
            temperature=_num(p, "temperature", 0.8, 0.05, 5.0),
            top_k=_num(p, "top_k", 40, 1, 10_000, int),
            top_p=_num(p, "top_p", 0.9, 0.01, 1.0),
            repetition_penalty=_num(p, "repetition_penalty", 1.0, 1.0, 3.0),
            seed=None if seed in (None, "") else _num(p, "seed", 0, 0, 2**31 - 1, int),
        )
    except ValueError as e:
        raise BadRequest(str(e))


def _piece(tok, i: int) -> str:
    return "⏎END" if i == tok.eot_id else tok.token_str(i)


def _context_ids(tok, model, text: str):
    ids = tok.encode(text)
    return ids[-model.cfg.block_size:]


def api_info(_p):
    model, meta = STATE["model"], STATE["meta"]
    groups = {
        "token embedding (tied with output head)": model.tok_emb.weight.numel(),
        "positional encoding": model.pos_emb.weight.numel() if model.pos_emb is not None else 0,
        "attention (Q,K,V,O)": sum(p.numel() for b in model.blocks for p in b.attn.parameters()),
        "feed-forward": sum(p.numel() for b in model.blocks for p in b.mlp.parameters()),
        "layer norms": sum(p.numel() for n, p in model.named_parameters() if "ln" in n),
    }
    return {
        "checkpoint": STATE["path"].name,
        "config": model.cfg.to_dict(),
        "num_params": model.num_params(),
        "param_groups": groups,
        "meta": {k: v for k, v in meta.items() if k != "history"},
        "history": meta.get("history", {}),
        "vocab_size": STATE["tok"].vocab_size,
        "merges": len(STATE["tok"].merges),
        "sample_merges": [
            [_piece(STATE["tok"], a), _piece(STATE["tok"], b), STATE["tok"].token_str(new)]
            for (a, b), new in list(STATE["tok"].merges.items())[:24]
        ],
    }


def api_tokenize(p):
    tok = STATE["tok"]
    text = _text(p, "text", required=False, limit=5000)
    ids = tok.encode(text)
    return {"ids": ids, "pieces": [_piece(tok, i) for i in ids], "chars": len(text),
            "bytes": len(text.encode("utf-8"))}


def api_next(p):
    model, tok = STATE["model"], STATE["tok"]
    text = _text(p, "text")
    cfg = _sampling(p)
    k = _num(p, "k", 15, 1, 50, int)
    ids = _context_ids(tok, model, text)
    with LOCK, torch.no_grad():
        ctx = torch.tensor([ids])
        logits, _ = model(ctx)
        last = logits[0, -1]
    raw = torch.softmax(last, -1)
    filt = next_token_distribution(last, cfg, ctx[0])
    order = torch.argsort(raw, descending=True)[:k].tolist()
    entropy = float(-(raw * torch.log(raw.clamp_min(1e-12))).sum())
    return {
        "context_tokens": len(ids),
        "entropy_bits": entropy / math.log(2),
        "kept": int((filt > 0).sum()),
        "candidates": [{"id": i, "piece": _piece(tok, i), "logit": float(last[i]),
                        "p": float(raw[i]), "p_sampled": float(filt[i])} for i in order],
    }


def api_generate(p):
    model, tok = STATE["model"], STATE["tok"]
    prompt = _text(p, "prompt")
    cfg = _sampling(p)
    max_new = _num(p, "max_new_tokens", 80, 1, 400, int)
    ids = tok.encode(prompt)
    steps = []
    with LOCK:
        for s in stream(model, ids, max_new, cfg, stop_ids=(tok.eot_id,), n_candidates=5):
            steps.append({"id": s.token, "piece": _piece(tok, s.token), "p": s.prob,
                          "p_model": s.model_prob,
                          "candidates": [[_piece(tok, i), q] for i, q in s.candidates]})
    nll = [-math.log(max(s["p_model"], 1e-12)) for s in steps]
    return {"prompt_tokens": len(ids), "steps": steps,
            "text": tok.decode([s["id"] for s in steps if s["id"] != tok.eot_id]),
            "avg_nll": sum(nll) / len(nll) if nll else None}


def api_attention(p):
    model, tok = STATE["model"], STATE["tok"]
    text = _text(p, "text", limit=1000)
    ids = tok.encode(text)[: model.cfg.block_size]
    with LOCK:
        maps = model.attention_maps(torch.tensor([ids]))
    return {"pieces": [_piece(tok, i) for i in ids],
            "layers": [[[[round(float(x), 4) for x in row] for row in head] for head in layer[0]]
                       for layer in maps]}


ROUTES = {"/api/info": api_info, "/api/tokenize": api_tokenize, "/api/next": api_next,
          "/api/generate": api_generate, "/api/attention": api_attention}
STATIC = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
          "/app.js": ("app.js", "application/javascript"), "/styles.css": ("styles.css", "text/css")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, allow_nan=False).encode(), "application/json")

    def _dispatch(self, params: dict) -> None:
        path = self.path.split("?")[0]
        fn = ROUTES.get(path)
        if fn is None:
            return self._json(404, {"error": "not found"})
        try:
            self._json(200, fn(params))
        except BadRequest as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # pragma: no cover - surfaced to the UI
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in STATIC:
            name, ctype = STATIC[path]
            return self._send(200, (WEB / name).read_bytes(), ctype)
        self._dispatch({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 100_000:
            return self._json(413, {"error": "request too large"})
        try:
            params = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(params, dict):
                raise ValueError
        except ValueError:
            return self._json(400, {"error": "body must be a JSON object"})
        self._dispatch(params)


def load(path: Path) -> None:
    model, tok, meta = load_checkpoint(path)
    STATE.update(model=model, tok=tok, meta=meta, path=path)


def main() -> None:
    ap = argparse.ArgumentParser(description="mini-GPT playground")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    path = Path(a.checkpoint)
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}; run `python train.py` first")
    torch.set_num_threads(a.threads)
    load(path)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"mini-GPT playground on {url}  ({STATE['model'].num_params():,} params, {path.name})")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
