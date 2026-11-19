import math
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.model import (GPT, GPTConfig, LayerNorm, causal_mask,  # noqa: E402
                           scaled_dot_product_attention, sinusoidal_table)


def tiny(**kw):
    cfg = dict(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    cfg.update(kw)
    torch.manual_seed(0)
    return GPT(GPTConfig(**cfg))


class TestAttention(unittest.TestCase):
    def test_matches_pytorch_reference(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 3, 7, 8) for _ in range(3))
        ours, w = scaled_dot_product_attention(q, k, v, causal_mask(7))
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertTrue(torch.allclose(ours, ref, atol=1e-5))
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(2, 3, 7)))

    def test_mask_zeroes_the_future(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, 5, 4) for _ in range(3))
        _, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
        self.assertTrue(torch.all(w.triu(1) == 0))
        self.assertAlmostEqual(float(w[0, 0, 0, 0]), 1.0)  # the first token can only see itself

    def test_scaling_by_sqrt_dk(self):
        q = torch.ones(1, 1, 2, 4)
        k = torch.tensor([[[[1., 1, 1, 1], [0, 0, 0, 0]]]])
        _, w = scaled_dot_product_attention(q, k, torch.zeros(1, 1, 2, 4))
        s = 4 / math.sqrt(4)  # q.k = 4, scaled by 1/sqrt(4)
        self.assertAlmostEqual(float(w[0, 0, 0, 0]), math.exp(s) / (math.exp(s) + 1), places=6)

    def test_model_is_causal(self):
        """Changing a future token must not change the logits of earlier positions."""
        m = tiny().eval()
        x = torch.randint(0, 50, (1, 10))
        y = x.clone()
        y[0, 6:] = (y[0, 6:] + 1) % 50
        with torch.no_grad():
            a, _ = m(x)
            b, _ = m(y)
        self.assertTrue(torch.allclose(a[0, :6], b[0, :6], atol=1e-6))
        self.assertFalse(torch.allclose(a[0, 6:], b[0, 6:]))

    def test_attention_maps_shapes(self):
        m = tiny().eval()
        maps = m.attention_maps(torch.randint(0, 50, (1, 9)))
        self.assertEqual(len(maps), 2)
        self.assertEqual(tuple(maps[0].shape), (1, 2, 9, 9))
        self.assertTrue(torch.all(maps[1].triu(1) == 0))


class TestLayers(unittest.TestCase):
    def test_layernorm_matches_torch(self):
        torch.manual_seed(0)
        x = torch.randn(3, 5, 16) * 4 + 2
        ours, ref = LayerNorm(16), nn.LayerNorm(16)
        with torch.no_grad():
            ours.weight.uniform_(0.5, 1.5)
            ours.bias.uniform_(-1, 1)
            ref.weight.copy_(ours.weight)
            ref.bias.copy_(ours.bias)
        self.assertTrue(torch.allclose(ours(x), ref(x), atol=1e-5))

    def test_sinusoidal_table(self):
        pe = sinusoidal_table(10, 8)
        self.assertEqual(tuple(pe.shape), (10, 8))
        self.assertTrue(torch.allclose(pe[0, 0::2], torch.zeros(4)))  # sin(0)
        self.assertTrue(torch.allclose(pe[0, 1::2], torch.ones(4)))   # cos(0)
        self.assertAlmostEqual(float(pe[3, 0]), math.sin(3), places=5)

    def test_bad_config(self):
        with self.assertRaises(ValueError):
            GPTConfig(n_embd=30, n_head=4)
        with self.assertRaises(ValueError):
            GPTConfig(pos_encoding="rotary")


class TestGPT(unittest.TestCase):
    def test_shapes_and_initial_loss(self):
        m = tiny()
        x = torch.randint(0, 50, (4, 16))
        logits, loss = m(x, torch.randint(0, 50, (4, 16)))
        self.assertEqual(tuple(logits.shape), (4, 16, 50))
        # Small init -> near-uniform predictions -> loss close to ln(V).
        self.assertLess(abs(float(loss) - math.log(50)), 0.3)

    def test_weight_tying(self):
        m = tiny()
        self.assertIs(m.lm_head.weight, m.tok_emb.weight)

    def test_too_long_sequence(self):
        with self.assertRaises(ValueError):
            tiny()(torch.zeros(1, 17, dtype=torch.long))

    def test_sinusoidal_model_runs(self):
        m = tiny(pos_encoding="sinusoidal")
        self.assertIsNone(m.pos_emb)
        logits, _ = m(torch.randint(0, 50, (2, 8)))
        self.assertEqual(tuple(logits.shape), (2, 8, 50))

    def test_position_matters(self):
        m = tiny().eval()
        x = torch.tensor([[3, 3, 3, 3]])
        with torch.no_grad():
            logits, _ = m(x)
        # Same token, different positions -> different outputs (positional encoding works).
        self.assertFalse(torch.allclose(logits[0, 1], logits[0, 2]))

    def test_can_overfit_one_batch(self):
        """The whole stack can learn: loss on a fixed batch falls far below ln(V)."""
        m = tiny()
        opt = m.configure_optimizer(lr=3e-3, weight_decay=0.0)
        g = torch.Generator().manual_seed(0)
        data = torch.randint(0, 50, (4, 17), generator=g)
        x, y = data[:, :-1], data[:, 1:]  # non-contiguous targets on purpose
        for _ in range(150):
            _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(loss), 0.2)

    def test_optimizer_groups(self):
        m = tiny()
        opt = m.configure_optimizer(1e-3, 0.1)
        decay, no_decay = opt.param_groups
        self.assertTrue(all(p.dim() >= 2 for p in decay["params"]))
        self.assertTrue(all(p.dim() < 2 for p in no_decay["params"]))
        self.assertEqual(sum(p.numel() for g in opt.param_groups for p in g["params"]), m.num_params())


if __name__ == "__main__":
    unittest.main()
