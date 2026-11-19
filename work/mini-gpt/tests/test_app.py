import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app  # noqa: E402
from minigpt.corpus import make_stories  # noqa: E402
from minigpt.model import GPT, GPTConfig  # noqa: E402
from minigpt.tokenizer import BPETokenizer  # noqa: E402
from pathlib import Path  # noqa: E402


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = BPETokenizer().train("\n\n".join(make_stories(50, 0)), 300)
        torch.manual_seed(0)
        model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2,
                              n_embd=16, dropout=0.0)).eval()
        meta = {"history": {"step": [0], "train_loss": [5.0], "val_loss": [5.1], "lr": [0.0],
                            "batch_loss": []}, "baselines": {"uniform": 5.7, "unigram": 4.9},
                "best_val_loss": 5.1, "val_loss": 5.1, "step": 0}
        app.STATE.update(model=model, tok=tok, meta=meta, path=Path("test.pt"))
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_static_page(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"mini-GPT", r.read())

    def test_info(self):
        code, r = self.call("/api/info")
        self.assertEqual(code, 200)
        self.assertEqual(r["num_params"], app.STATE["model"].num_params())
        self.assertEqual(r["config"]["n_layer"], 2)

    def test_tokenize_roundtrip(self):
        code, r = self.call("/api/tokenize", {"text": "Once upon a time 🙂"})
        self.assertEqual(code, 200)
        self.assertEqual(app.STATE["tok"].decode(r["ids"]), "Once upon a time 🙂")

    def test_next_distribution(self):
        code, r = self.call("/api/next", {"text": "Tom had", "strategy": "top_k", "top_k": 3, "k": 10})
        self.assertEqual(code, 200)
        self.assertEqual(len(r["candidates"]), 10)
        self.assertEqual(r["kept"], 3)
        ps = [c["p"] for c in r["candidates"]]
        self.assertEqual(ps, sorted(ps, reverse=True))

    def test_generate_seeded(self):
        body = {"prompt": "Once", "max_new_tokens": 12, "strategy": "top_p", "seed": 5}
        a = self.call("/api/generate", body)[1]
        b = self.call("/api/generate", body)[1]
        self.assertEqual(a["text"], b["text"])
        self.assertLessEqual(len(a["steps"]), 12)

    def test_attention_is_causal(self):
        code, r = self.call("/api/attention", {"text": "Lily had a key."})
        self.assertEqual(code, 200)
        m = r["layers"][0][0]
        self.assertEqual(len(m), len(r["pieces"]))
        for i, row in enumerate(m):
            self.assertAlmostEqual(sum(row), 1.0, places=3)
            self.assertTrue(all(v == 0 for v in row[i + 1:]))

    def test_validation_errors(self):
        self.assertEqual(self.call("/api/generate", {"prompt": ""})[0], 400)
        self.assertEqual(self.call("/api/generate", {"prompt": "a", "temperature": 0})[0], 400)
        self.assertEqual(self.call("/api/next", {"text": "a", "strategy": "beam"})[0], 400)
        self.assertEqual(self.call("/api/nope", {})[0], 404)


if __name__ == "__main__":
    unittest.main()
