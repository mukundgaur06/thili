import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora.gpt2 import GPT2, GPT2Classifier, GPT2Config  # noqa: E402
from lora.inject import (AdapterBank, adapter_state_dict, count_params, inject_lora,  # noqa: E402
                         load_adapter, merge_lora, read_adapter, save_adapter, set_lora_enabled,
                         strip_lora, unmerge_lora)
from lora.layers import Conv1D, LoRALinear, lora_param_count  # noqa: E402


def tiny_gpt2(seed=0):
    torch.manual_seed(seed)
    return GPT2(GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=4))


def randomize_lora(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)


class TestLoRALinear(unittest.TestCase):
    def test_zero_init_keeps_base_function(self):
        base = nn.Linear(20, 30)
        x = torch.randn(4, 20)
        ref = base(x)
        lora = LoRALinear(base, r=4)
        self.assertTrue(torch.equal(lora(x), ref))       # B = 0 -> exactly the base output

    def test_only_A_B_trainable_and_count(self):
        lora = LoRALinear(nn.Linear(20, 30), r=4)
        trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"lora_A", "lora_B"})
        self.assertEqual(sum(p.numel() for p in lora.parameters() if p.requires_grad),
                         lora_param_count(20, 30, 4))
        self.assertEqual(lora_param_count(768, 2304, 8), 8 * (768 + 2304))

    def test_forward_equals_explicit_formula(self):
        base = nn.Linear(10, 6)
        lora = LoRALinear(base, r=3, alpha=6)
        with torch.no_grad():
            lora.lora_B.normal_()
        x = torch.randn(5, 10)
        W = base.weight + (6 / 3) * lora.lora_B @ lora.lora_A
        self.assertTrue(torch.allclose(lora(x), x @ W.t() + base.bias, atol=1e-5))
        self.assertEqual(torch.linalg.matrix_rank(lora.delta_weight()).item(), 3)

    def test_conv1d_is_transposed_linear(self):
        c = Conv1D(10, 6)
        x = torch.randn(3, 7, 10)
        self.assertTrue(torch.allclose(c(x), x @ c.weight + c.bias, atol=1e-6))
        lora = LoRALinear(c, r=2)
        with torch.no_grad():
            lora.lora_B.normal_()
        before = lora(x)
        lora.merge()
        self.assertTrue(torch.allclose(lora(x), before, atol=1e-5))

    def test_merge_unmerge_roundtrip(self):
        base = nn.Linear(16, 16)
        w0 = base.weight.detach().clone()
        lora = LoRALinear(base, r=4)
        with torch.no_grad():
            lora.lora_B.normal_()
        x = torch.randn(8, 16)
        y = lora(x)
        lora.merge()
        self.assertFalse(torch.allclose(base.weight, w0))
        self.assertTrue(torch.allclose(lora(x), y, atol=1e-5))
        lora.merge()                                   # idempotent
        lora.unmerge()
        self.assertTrue(torch.allclose(base.weight, w0, atol=1e-6))
        self.assertTrue(torch.allclose(lora(x), y, atol=1e-5))

    def test_gradients_reach_only_adapter(self):
        base = nn.Linear(8, 8)
        lora = LoRALinear(base, r=2)
        lora(torch.randn(4, 8)).sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNotNone(lora.lora_B.grad)
        # B = 0 at init, so dL/dA = 0 on the first step, while dL/dB != 0
        self.assertTrue(torch.all(lora.lora_A.grad == 0))
        self.assertGreater(lora.lora_B.grad.abs().sum().item(), 0)

    def test_disabled_is_base(self):
        lora = LoRALinear(nn.Linear(8, 8), r=2)
        with torch.no_grad():
            lora.lora_B.normal_()
        x = torch.randn(2, 8)
        lora.enabled = False
        self.assertTrue(torch.equal(lora(x), lora.base(x)))

    def test_bad_rank(self):
        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(4, 4), r=0)
        with self.assertRaises(ValueError):
            LoRALinear(nn.Linear(4, 4), r=5)


class TestInjection(unittest.TestCase):
    def test_inject_targets_and_freeze(self):
        model = GPT2Classifier(tiny_gpt2())
        wrapped = inject_lora(model, ("c_attn",), r=4)
        self.assertEqual(wrapped, ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"])
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all("lora_" in n for n in trainable))
        c = count_params(model)
        self.assertEqual(c["trainable"], 2 * lora_param_count(32, 96, 4))
        with self.assertRaises(ValueError):
            inject_lora(model, ("nonexistent",))

    def test_injected_model_starts_identical(self):
        gpt = tiny_gpt2()
        idx = torch.randint(0, 100, (2, 10))
        ref = gpt(idx)
        inject_lora(gpt, ("c_attn", "c_proj", "c_fc"), r=2)
        self.assertTrue(torch.allclose(gpt(idx), ref, atol=1e-6))

    def test_injection_does_not_double_wrap(self):
        gpt = tiny_gpt2()
        inject_lora(gpt, ("c_attn",), r=2)
        with self.assertRaises(ValueError):           # the only c_attn layers are already wrapped
            inject_lora(gpt, ("c_attn",), r=2)
        self.assertEqual(sum(isinstance(m, LoRALinear) for m in gpt.modules()), 2)

    def test_causal_model(self):
        gpt = tiny_gpt2().eval()
        a = torch.randint(0, 100, (1, 12))
        b = a.clone()
        b[0, 8:] = (b[0, 8:] + 1) % 100
        with torch.no_grad():
            self.assertTrue(torch.allclose(gpt(a)[0, :8], gpt(b)[0, :8], atol=1e-6))

    def test_classifier_ignores_right_padding(self):
        model = GPT2Classifier(tiny_gpt2()).eval()
        x = torch.randint(0, 100, (1, 6))
        padded = torch.cat([x, torch.full((1, 4), 99)], dim=1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x, torch.tensor([6])), model(padded, torch.tensor([6])), atol=1e-6))

    def test_training_moves_only_adapter(self):
        model = GPT2Classifier(tiny_gpt2())
        inject_lora(model, ("c_attn",), r=4)
        model.score.weight.requires_grad_(True)
        frozen = {k: v.clone() for k, v in model.state_dict().items() if "lora_" not in k and "score" not in k}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        x, lengths, y = torch.randint(0, 100, (8, 10)), torch.full((8,), 10), torch.randint(0, 2, (8,))
        losses = []
        for _ in range(30):
            loss = nn.functional.cross_entropy(model(x, lengths), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        for k, v in model.state_dict().items():
            if k in frozen:
                self.assertTrue(torch.equal(v, frozen[k]), k)


class TestAdapters(unittest.TestCase):
    def test_save_load_roundtrip(self):
        a = GPT2Classifier(tiny_gpt2())
        inject_lora(a, ("c_attn",), r=4)
        randomize_lora(a)
        idx, lengths = torch.randint(0, 100, (3, 9)), torch.full((3,), 9)
        with tempfile.TemporaryDirectory() as d:
            p16, p32 = os.path.join(d, "a16.safetensors"), os.path.join(d, "a32.safetensors")
            size16 = save_adapter(p16, a, {"rank": 4}, extra=("score",))            # default fp16
            size32 = save_adapter(p32, a, {"rank": 4}, extra=("score",), dtype=torch.float32)
            sd16, meta = read_adapter(p16)
            sd32, _ = read_adapter(p32)
        self.assertEqual(meta["rank"], "4")
        self.assertLess(size16, 0.6 * size32)           # fp16 roughly halves the file
        self.assertTrue(all(v.dtype == torch.float16 for v in sd16.values()))
        self.assertTrue(all("lora_" in k or k.startswith("score") for k in sd16))
        with torch.no_grad():
            ref = a.eval()(idx, lengths)
            for sd, tol in [(sd32, 1e-6), (sd16, 1e-3)]:   # fp32 is exact, fp16 is close
                b = GPT2Classifier(tiny_gpt2())
                inject_lora(b, ("c_attn",), r=4)
                load_adapter(b, sd)
                self.assertEqual(b.transformer.h[0].attn.c_attn.lora_A.dtype, torch.float32)
                self.assertTrue(torch.allclose(ref, b.eval()(idx, lengths), atol=tol))

    def test_load_rejects_bad_adapters(self):
        m = tiny_gpt2()
        inject_lora(m, ("c_attn",), r=4)
        with self.assertRaises(KeyError):
            load_adapter(m, {"nope.lora_A": torch.zeros(1)})
        with self.assertRaises(ValueError):
            load_adapter(m, {"h.0.attn.c_attn.lora_A": torch.zeros(2, 32)})
        merge_lora(m)
        with self.assertRaises(RuntimeError):
            load_adapter(m, adapter_state_dict(m))

    def test_merge_whole_model_and_strip(self):
        m = tiny_gpt2().eval()
        inject_lora(m, ("c_attn", "c_fc"), r=4)
        randomize_lora(m)
        idx = torch.randint(0, 100, (2, 11))
        with torch.no_grad():
            ref = m(idx)
            self.assertEqual(merge_lora(m), 4)
            self.assertTrue(torch.allclose(m(idx), ref, atol=1e-5))
            unmerge_lora(m)
            self.assertTrue(torch.allclose(m(idx), ref, atol=1e-5))
            strip_lora(m)
            self.assertFalse(any(isinstance(x, LoRALinear) for x in m.modules()))
            self.assertTrue(torch.allclose(m(idx), ref, atol=1e-5))

    def test_adapter_bank_swap(self):
        m = tiny_gpt2().eval()
        inject_lora(m, ("c_attn",), r=4)
        idx = torch.randint(0, 100, (2, 8))
        with torch.no_grad():
            base_out = m(idx)
        bank = AdapterBank(m)
        outs = {}
        for name, seed in [("a", 1), ("b", 2)]:
            randomize_lora(m, seed)
            bank.add(name, adapter_state_dict(m))
            with torch.no_grad():
                outs[name] = m(idx)
        self.assertFalse(torch.allclose(outs["a"], outs["b"]))
        with torch.no_grad():
            for name in ["a", "b", "a"]:
                bank.activate(name)
                self.assertTrue(torch.allclose(m(idx), outs[name], atol=1e-6))
            bank.deactivate()
            self.assertTrue(torch.allclose(m(idx), base_out, atol=1e-6))
        set_lora_enabled(m, True)
        with self.assertRaises(KeyError):
            bank.activate("missing")


class TestAgainstTransformers(unittest.TestCase):
    """Runs in the lfllm workbench (which has transformers); skipped elsewhere."""

    def test_matches_hf_gpt2(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            pass
        except OSError:
            # transformers imports torchaudio whenever it is installed; a torchaudio built for a
            # different torch crashes that import. We never use audio, so hide it.
            import transformers.utils as tu
            import transformers.utils.import_utils as iu
            iu.is_torchaudio_available = lambda: False
            tu.is_torchaudio_available = iu.is_torchaudio_available
        try:
            from transformers import GPT2Config as HFConfig, GPT2Model
        except ImportError:
            self.skipTest("transformers not installed")
        torch.manual_seed(0)
        hf = GPT2Model(HFConfig(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=4,
                                attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)).eval()
        ours = GPT2(GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=4))
        sd = {k: v for k, v in hf.state_dict().items() if not k.endswith((".attn.bias", ".attn.masked_bias"))}
        ours.load_state_dict(sd)
        idx = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            self.assertTrue(torch.allclose(ours(idx), hf(idx).last_hidden_state, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
