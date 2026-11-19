import math
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.corpus import make_stories, write_corpus  # noqa: E402
from minigpt.data import (Batcher, encode_documents, read_documents,  # noqa: E402
                          split_documents, unigram_cross_entropy)
from minigpt.model import GPT, GPTConfig  # noqa: E402
from minigpt.sampling import (SamplingConfig, generate, next_token_distribution,  # noqa: E402
                              top_k_filter, top_p_filter)
from minigpt.tokenizer import BPETokenizer  # noqa: E402
from minigpt.trainer import TrainConfig, load_checkpoint, lr_at, train  # noqa: E402


class TestData(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        docs = [f"doc {i}" for i in range(50)] + ["doc 3", "doc 3"]
        tr, va = split_documents(docs, 0.2, seed=1)
        self.assertEqual((tr, va), split_documents(docs, 0.2, seed=1))
        self.assertFalse(set(tr) & set(va))           # duplicates never straddle the split
        self.assertEqual(len(tr) + len(va), len(docs))

    def test_split_errors(self):
        with self.assertRaises(ValueError):
            split_documents(["same", "same"])
        with self.assertRaises(ValueError):
            split_documents(["a", "b"], 1.5)

    def test_batches_are_shifted_by_one(self):
        data = torch.arange(100)
        x, y = Batcher(data, 8, 4, seed=0).get()
        self.assertEqual(tuple(x.shape), (4, 8))
        self.assertTrue(torch.equal(y, x + 1))

    def test_sequential_covers_data(self):
        data = torch.arange(33)
        batches = list(Batcher(data, 8, 3).sequential())
        xs = torch.cat([b[0] for b in batches]).flatten()
        self.assertTrue(torch.equal(xs, torch.arange(32)))

    def test_encode_documents_appends_eot(self):
        tok = BPETokenizer().train("hello world hello world", 260)
        ids = encode_documents(tok, ["hello", "world"])
        self.assertEqual(ids.tolist().count(tok.eot_id), 2)
        self.assertEqual(int(ids[-1]), tok.eot_id)

    def test_unigram_baseline_below_uniform(self):
        train_ids = torch.tensor([0] * 90 + [1] * 10)
        val_ids = torch.tensor([0] * 10)
        self.assertLess(unigram_cross_entropy(train_ids, val_ids, 4), math.log(4))

    def test_corpus_reproducible(self):
        self.assertEqual(make_stories(5, 3), make_stories(5, 3))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.txt")
            write_corpus(p, 7, 3)
            self.assertEqual(len(read_documents(p)), 7)


class TestSampling(unittest.TestCase):
    def test_top_k(self):
        out = top_k_filter(torch.tensor([1.0, 4.0, 3.0, 2.0]), 2)
        self.assertEqual(torch.isfinite(out).tolist(), [False, True, True, False])

    def test_top_p_keeps_smallest_nucleus(self):
        logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        self.assertEqual(torch.isfinite(top_p_filter(logits, 0.7)).tolist(), [True, True, False, False])
        self.assertEqual(torch.isfinite(top_p_filter(logits, 0.1)).tolist(), [True, False, False, False])
        self.assertTrue(torch.isfinite(top_p_filter(logits, 1.0)).all())

    def test_temperature(self):
        logits = torch.log(torch.tensor([0.6, 0.4]))
        p = next_token_distribution(logits, SamplingConfig("temperature", temperature=0.5))
        self.assertAlmostEqual(float(p[0]), 0.36 / 0.52, places=5)
        flat = next_token_distribution(logits, SamplingConfig("temperature", temperature=100))
        self.assertAlmostEqual(float(flat[0]), 0.5, places=2)

    def test_repetition_penalty(self):
        logits = torch.tensor([2.0, 2.0, -1.0])
        p = next_token_distribution(logits, SamplingConfig("temperature", temperature=1,
                                                           repetition_penalty=2.0), torch.tensor([0, 2]))
        self.assertLess(float(p[0]), float(p[1]))

    def test_invalid_configs(self):
        for kw in [dict(strategy="beam"), dict(temperature=0), dict(top_k=0), dict(top_p=0),
                   dict(repetition_penalty=0.5)]:
            with self.assertRaises(ValueError):
                SamplingConfig(**kw)

    def test_generation_seeded_and_greedy_deterministic(self):
        torch.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=30, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0))
        a = generate(m, [1, 2], 20, SamplingConfig("top_k", top_k=5, seed=3))
        b = generate(m, [1, 2], 20, SamplingConfig("top_k", top_k=5, seed=3))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)  # longer than block_size: context is cropped
        g1 = generate(m, [1], 10, SamplingConfig("greedy"))
        self.assertEqual(g1, generate(m, [1], 10, SamplingConfig("greedy")))

    def test_stop_token(self):
        torch.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=5, block_size=8, n_layer=1, n_head=1, n_embd=8, dropout=0))
        first = generate(m, [0], 1, SamplingConfig("greedy"))[0]
        self.assertEqual(generate(m, [0], 10, SamplingConfig("greedy"), stop_ids=(first,)), [])


class TestTrainer(unittest.TestCase):
    def test_lr_schedule(self):
        c = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
        self.assertAlmostEqual(lr_at(0, c), 0.1)
        self.assertAlmostEqual(lr_at(9, c), 1.0)
        self.assertAlmostEqual(lr_at(60, c), 0.55)
        self.assertAlmostEqual(lr_at(110, c), 0.1)

    def test_end_to_end_loss_falls_and_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as d:
            corpus = os.path.join(d, "c.txt")
            write_corpus(corpus, 200, 0)
            tcfg = TrainConfig(corpus=corpus, out_dir=d, vocab_size=300, batch_size=8, max_steps=150,
                               eval_every=75, eval_batches=2, warmup_steps=5, threads=2)
            mcfg = GPTConfig(block_size=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
            res = train(tcfg, mcfg, log=lambda s: None)
            h = res["history"]
            self.assertLess(h["val_loss"][-1], h["val_loss"][0] - 1.0)
            self.assertLess(res["best_val_loss"], res["baselines"]["unigram"])
            model, tok, meta = load_checkpoint(os.path.join(d, "best.pt"))
            self.assertEqual(model.cfg.vocab_size, tok.vocab_size)
            self.assertIn("history", meta)
            out = generate(model, tok.encode("Once"), 5, SamplingConfig("greedy"))
            self.assertEqual(len(out), 5)


if __name__ == "__main__":
    unittest.main()
