"""Lesson 8: ship it. A small local web page in front of the fine-tuned tagger.

    python app.py            # then open http://127.0.0.1:8610

Standard library only; POST /api/tag with {"clauses": [...]} returns the tags as JSON.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

import clausegpt  # noqa: F401  (must come before transformers)
from clausegpt.tagger import ClauseTagger

ROOT = Path(__file__).resolve().parent
PAGE = (ROOT / "web" / "index.html").read_bytes()
MAX_CLAUSES = 20
MAX_CHARS = 5000


def make_handler(tagger: ClauseTagger):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/labels":
                self._send(200, json.dumps(tagger.names).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/tag":
                return self._send(404, b"not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                clauses = json.loads(self.rfile.read(min(n, 200_000)))["clauses"]
                clauses = [str(c)[:MAX_CHARS] for c in clauses if str(c).strip()][:MAX_CLAUSES]
                if not clauses:
                    raise ValueError("no clauses")
            except Exception as e:  # noqa: BLE001
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            preds = tagger.predict(clauses, k=5)
            body = [{"clause": c, "tag": f"<{p.label}>", "top": p.top, "generated": p.generated, "valid": p.valid}
                    for c, p in zip(clauses, preds)]
            self._send(200, json.dumps(body).encode(), "application/json")

        def log_message(self, *a):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "outputs" / "clause-gpt2"))
    ap.add_argument("--port", type=int, default=8610)
    args = ap.parse_args()
    torch.set_num_threads(4)
    tagger = ClauseTagger.load(args.model)
    print(f"clause tagger on http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tagger)).serve_forever()


if __name__ == "__main__":
    main()
