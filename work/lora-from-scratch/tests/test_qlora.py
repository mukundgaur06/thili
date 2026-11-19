import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora.gpt2 import GPT2, GPT2Config  # noqa: E402
from lora.inject import inject_lora  # noqa: E402
from lora.layers import Conv1D, LoRALinear  # noqa: E402
from lora.qlora import (INT4_LEVELS, NF4_REFERENCE, Linear4bit, dequantize_absmax_8bit,  # noqa: E402
                        dequantize_blockwise, nf4_levels, pack_4bit, quantize_absmax_8bit,
                        quantize_blockwise, quantize_model, unpack_4bit)


class TestNF4(unittest.TestCase):
    def test_levels_match_published_table(self):
        lv = nf4_levels()
        self.assertEqual(lv.numel(), 16)
        self.assertTrue(torch.allclose(lv, torch.tensor(NF4_REFERENCE), atol=1e-6))
        self.assertIn(0.0, lv.tolist())          # zero is exactly representable

    def test_nf4_levels_used_evenly_on_normal_data(self):
        torch.manual_seed(0)
        w = torch.randn(64 * 2000)
        codes, _, _ = quantize_blockwise(w, nf4_levels(), 64)
        freq = torch.bincount(codes.flatten().long(), minlength=16).float() / codes.numel()
        codes_int, _, _ = quantize_blockwise(w, INT4_LEVELS, 64)
        freq_int = torch.bincount(codes_int.flatten().long(), minlength=16).float() / codes.numel()
        entropy = lambda f: float(-(f[f > 0] * f[f > 0].log2()).sum())  # noqa: E731
        self.assertGreater(entropy(freq), entropy(freq_int))   # NF4 uses its 4 bits better
        self.assertGreater(entropy(freq), 3.8)

    def test_nf4_beats_int4_on_normal_weights(self):
        torch.manual_seed(0)
        w = torch.randn(256, 256) * 0.02
        errs = {}
        for name, lv in [("nf4", nf4_levels()), ("int4", INT4_LEVELS)]:
            c, a, n = quantize_blockwise(w, lv, 64)
            errs[name] = (dequantize_blockwise(c, a, lv, n, w.shape) - w).pow(2).mean().item()
        self.assertLess(errs["nf4"], errs["int4"])

    def test_blockwise_isolates_outliers(self):
        w = torch.randn(64 * 4) * 0.01
        w[0] = 10.0                                   # one outlier in block 0
        c, a, n = quantize_blockwise(w, nf4_levels(), 64)
        deq = dequantize_blockwise(c, a, nf4_levels(), n, w.shape)
        err_other = (deq[64:] - w[64:]).abs().max().item()
        self.assertLess(err_other, 0.01)              # other blocks unaffected
        self.assertAlmostEqual(deq[0].item(), 10.0, places=4)   # absmax is exact

    def test_pack_unpack(self):
        codes = torch.randint(0, 16, (101,), dtype=torch.uint8)
        packed = pack_4bit(codes)
        self.assertEqual(packed.numel(), 51)
        self.assertTrue(torch.equal(unpack_4bit(packed, 101), codes))

    def test_double_quant(self):
        torch.manual_seed(0)
        s = torch.rand(1000) * 0.1 + 0.01
        q, sc, mean, n = quantize_absmax_8bit(s)
        self.assertEqual(q.dtype, torch.int8)
        self.assertLess((dequantize_absmax_8bit(q, sc, mean, n) - s).abs().max().item(), 1e-3)


class TestLinear4bit(unittest.TestCase):
    def test_close_to_fp32_and_small(self):
        torch.manual_seed(0)
        lin = nn.Linear(256, 128)
        with torch.no_grad():
            lin.weight.normal_(std=0.02)
        q = Linear4bit(lin)
        x = torch.randn(4, 256)
        rel = (q(x) - lin(x)).norm() / lin(x).norm()
        self.assertLess(rel.item(), 0.15)
        self.assertLess(q.storage_bytes(), 0.15 * lin.weight.numel() * 4 + 4 * 128)

    def test_conv1d_orientation(self):
        torch.manual_seed(0)
        c = Conv1D(64, 32)
        q = Linear4bit(c, double_quant=False)
        x = torch.randn(3, 64)
        self.assertLess(((q(x) - c(x)).norm() / c(x).norm()).item(), 0.15)

    def test_qlora_gradients_and_frozen_codes(self):
        torch.manual_seed(0)
        gpt = GPT2(GPT2Config(vocab_size=50, n_positions=16, n_embd=32, n_layer=1, n_head=2))
        inject_lora(gpt, ("c_attn",), r=4)
        done = quantize_model(gpt)
        self.assertEqual(len(done), 4)                # attn.c_attn (inside LoRA), attn.c_proj, mlp.c_fc, mlp.c_proj
        self.assertTrue(isinstance(gpt.h[0].attn.c_attn, LoRALinear))
        self.assertTrue(isinstance(gpt.h[0].attn.c_attn.base, Linear4bit))
        packed = gpt.h[0].attn.c_attn.base.packed.clone()
        gpt(torch.randint(0, 50, (2, 8))).sum().backward()
        self.assertIsNotNone(gpt.h[0].attn.c_attn.lora_B.grad)
        self.assertTrue(torch.equal(gpt.h[0].attn.c_attn.base.packed, packed))
        with self.assertRaises(TypeError):
            gpt.h[0].attn.c_attn.merge()               # cannot merge into 4-bit codes


if __name__ == "__main__":
    unittest.main()
