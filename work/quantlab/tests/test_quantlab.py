import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quantlab  # noqa: E402,F401  (silences torch.ao deprecation warnings)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.ao.quantization.observer import MinMaxObserver  # noqa: E402

from quantlab import data as data_mod  # noqa: E402
from quantlab.benchmarks import compare, interleaved_latency, latency, state_dict_bytes  # noqa: E402
from quantlab.evaluate import confusion_matrix, evaluate, per_class  # noqa: E402
from quantlab.export import export_onnx, export_torchscript  # noqa: E402
from quantlab.model import ModelConfig, QuantCNN, count_parameters  # noqa: E402
from quantlab.quantization.dynamic import quantize_dynamic  # noqa: E402
from quantlab.quantization.fp16 import check_support  # noqa: E402
from quantlab.quantization.qat import QATConfig, prepare_for_qat, quantize_qat  # noqa: E402
from quantlab.quantization.static import (StaticConfig, calibrate, convert_static, observer_stats,  # noqa: E402
                                          prepare_static, quantize_static)
from quantlab.quantization.utilities import (calc_qparams, dequantize, error_report, fake_quantize,  # noqa: E402
                                             pick_engine, qrange, quantize)
from quantlab.train import load_checkpoint, save_checkpoint, train_epochs, validate, TrainConfig  # noqa: E402

FASHION = data_mod.DATA_DIR / "FashionMNIST" / "raw"
torch.set_num_threads(2)


def trained_synthetic(epochs: int = 3):
    """A small model trained for real on the synthetic dataset (a bright block marks the class)."""
    d = data_mod.synthetic()
    torch.manual_seed(0)
    m = QuantCNN(ModelConfig(channels=[8, 16], hidden=32))
    train_epochs(m, d, epochs, lr=3e-3, batch_size=64)
    return m.eval(), d


class TestModel(unittest.TestCase):
    def test_forward_shape_and_parameters(self):
        m = QuantCNN().eval()
        out = m(torch.randn(5, 1, 28, 28))
        self.assertEqual(tuple(out.shape), (5, 10))
        # conv1 144 + bn1 32 + conv2 4608 + bn2 64 + fc1 1568*128+128 + fc2 128*10+10
        self.assertEqual(count_parameters(m), 144 + 32 + 4608 + 64 + 1568 * 128 + 128 + 1290)

    def test_fusion_preserves_outputs(self):
        m = QuantCNN().eval()
        with torch.no_grad():
            m.bn1.running_mean.uniform_(-0.5, 0.5)
            m.bn1.running_var.uniform_(0.5, 2.0)
        x = torch.randn(4, 1, 28, 28)
        ref = m(x)
        import copy
        fused = copy.deepcopy(m).fuse()
        self.assertTrue(torch.allclose(ref, fused(x), atol=1e-5))
        self.assertFalse(any(isinstance(mod, nn.BatchNorm2d) for mod in fused.modules()))


class TestData(unittest.TestCase):
    def test_synthetic_shapes(self):
        d = data_mod.synthetic()
        x, y = next(d.batches(d.train, 32))
        self.assertEqual(tuple(x.shape), (32, 1, 28, 28))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.int64)

    @unittest.skipUnless(FASHION.exists(), "Fashion-MNIST not downloaded (python -m quantlab.pipeline downloads it)")
    def test_fashion_mnist_split_is_leak_free(self):
        d = data_mod.load("fashion_mnist")
        self.assertEqual((len(d.train), len(d.val), len(d.test)), (55000, 5000, 10000))
        self.assertEqual(len(d.classes), 10)
        again = data_mod.load("fashion_mnist")
        self.assertTrue(torch.equal(d.val.y, again.val.y))         # the split is seeded, so reproducible
        self.assertEqual(len(d.train) + len(d.val), 60000)          # carved from the official train split only
        mean, std = data_mod.pixel_stats(d.train.x)
        self.assertAlmostEqual(mean, d.mean, places=2)
        self.assertAlmostEqual(std, d.std, places=2)
        cal = d.calibration_set(100)
        self.assertEqual(tuple(cal.shape), (100, 1, 28, 28))


class TestEvaluation(unittest.TestCase):
    def test_confusion_and_per_class_by_hand(self):
        y = np.array([0, 0, 1, 1, 2])
        p = np.array([0, 1, 1, 1, 0])
        cm = confusion_matrix(y, p, 3)
        self.assertEqual(cm.tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]])
        rows = per_class(cm, ["a", "b", "c"])
        self.assertAlmostEqual(rows[1]["precision"], 2 / 3)
        self.assertAlmostEqual(rows[1]["recall"], 1.0)
        self.assertEqual(rows[2]["f1"], 0.0)

    def test_trained_model_learns_and_evaluate_agrees_with_validate(self):
        m, d = trained_synthetic()
        ev = evaluate(m, d, d.test)
        self.assertGreater(ev["accuracy"], 0.9)
        self.assertEqual(sum(map(sum, ev["confusion_matrix"])), len(d.test))
        self.assertAlmostEqual(ev["accuracy"], validate(m, d, d.test)["accuracy"])


class TestQuantMath(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(qrange(8, False), (0, 255))
        self.assertEqual(qrange(8, True), (-128, 127))
        self.assertEqual(qrange(4, True), (-8, 7))

    def test_qparams_match_pytorch_observer(self):
        torch.manual_seed(0)
        for x in (torch.randn(1000) * 3 + 1, torch.rand(500) * 6, -torch.rand(300)):
            for symmetric, dtype, scheme in ((False, torch.quint8, torch.per_tensor_affine),
                                             (True, torch.qint8, torch.per_tensor_symmetric)):
                obs = MinMaxObserver(dtype=dtype, qscheme=scheme)
                obs(x)
                ref_scale, ref_zp = obs.calculate_qparams()
                p = calc_qparams(float(x.min()), float(x.max()), symmetric=symmetric)
                self.assertAlmostEqual(p.scale, float(ref_scale), places=7)
                self.assertEqual(p.zero_point, int(ref_zp))

    def test_quantize_matches_pytorch_kernel(self):
        torch.manual_seed(1)
        x = torch.randn(2000) * 2
        p = calc_qparams(float(x.min()), float(x.max()))
        ref = torch.quantize_per_tensor(x, p.scale, p.zero_point, torch.quint8)
        self.assertTrue(torch.equal(quantize(x, p), ref.int_repr().to(torch.int32)))
        self.assertTrue(torch.allclose(dequantize(quantize(x, p), p), ref.dequantize(), atol=1e-6))

    def test_error_is_at_most_half_a_step_and_zero_is_exact(self):
        x = torch.linspace(-1.3, 2.7, 10001)
        p = calc_qparams(-1.3, 2.7)
        err = (fake_quantize(x, p) - x).abs().max().item()
        self.assertLessEqual(err, p.scale / 2 + 1e-6)
        self.assertEqual(fake_quantize(torch.tensor([0.0]), p).item(), 0.0)
        r = error_report(x, p)
        self.assertEqual(r["clipped_share"], 0.0)
        self.assertGreater(r["sqnr_db"], 40)

    def test_symmetric_vs_asymmetric(self):
        p = calc_qparams(0.0, 6.0, symmetric=True)           # ReLU-like data
        self.assertEqual(p.zero_point, 0)
        a = calc_qparams(0.0, 6.0, symmetric=False)
        self.assertLess(a.scale, p.scale)                     # asymmetric spends no codes on negatives
        self.assertAlmostEqual(a.scale, 6.0 / 255)

    def test_clipping_outside_range(self):
        p = calc_qparams(-1.0, 1.0)
        r = error_report(torch.tensor([-5.0, 0.0, 0.5, 5.0]), p)
        self.assertAlmostEqual(r["clipped_share"], 0.5)


class TestPTQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pick_engine()
        cls.model, cls.data = trained_synthetic()
        cls.calib = cls.data.calibration_set(256)

    def test_calibration_records_real_activation_ranges(self):
        """MinMax observers must hold exactly the min/max of what flowed through (computed independently)."""
        cfg = StaticConfig(observer="minmax", calibration_images=256)
        prepared = prepare_static(self.model, cfg)
        seen = {}

        def hook(name):
            def f(_, __, out):
                lo, hi = float(out.min()), float(out.max())
                old = seen.get(name, (np.inf, -np.inf))
                seen[name] = (min(old[0], lo), max(old[1], hi))
            return f
        handles = [getattr(prepared, n)[0].register_forward_hook(hook(n)) if isinstance(getattr(prepared, n), nn.Sequential)
                   else getattr(prepared, n).register_forward_hook(hook(n)) for n in ("conv1", "fc2")]
        calibrate(prepared, self.calib, 64)
        for h in handles:
            h.remove()
        stats = {r["layer"]: r for r in observer_stats(prepared) if r["tensor"] == "output activation"}
        self.assertAlmostEqual(stats["fc2"]["min"], seen["fc2"][0], places=4)
        self.assertAlmostEqual(stats["fc2"]["max"], seen["fc2"][1], places=4)
        self.assertGreaterEqual(stats["conv1"]["min"], 0.0)       # after the fused ReLU
        p = calc_qparams(stats["fc2"]["min"], stats["fc2"]["max"])
        self.assertAlmostEqual(stats["fc2"]["scale"], p.scale, places=6)
        self.assertEqual(stats["fc2"]["zero_point"], p.zero_point)

    def test_uncalibrated_observers_have_no_range(self):
        prepared = prepare_static(self.model, StaticConfig(observer="minmax"))
        obs = prepared.fc2.activation_post_process
        self.assertEqual(float(obs.min_val), float("inf"))         # nothing observed, nothing invented

    def test_static_conversion(self):
        q, stats = quantize_static(self.model, self.calib, StaticConfig())
        self.assertIn("quantized", type(q.conv1).__module__)
        self.assertIn("quantized", type(q.fc1).__module__)
        self.assertTrue(any(r["tensor"].startswith("weight") for r in stats))
        acc_q = evaluate(q, self.data, self.data.test)["accuracy"]
        acc_f = evaluate(self.model, self.data, self.data.test)["accuracy"]
        self.assertGreater(acc_q, acc_f - 0.05)
        self.assertLess(state_dict_bytes(q), 0.4 * state_dict_bytes(self.model))

    def test_dynamic_quantizes_linear_only(self):
        q = quantize_dynamic(self.model)
        self.assertIn("dynamic", type(q.fc1).__module__)
        self.assertIsInstance(q.conv1, nn.Conv2d)                   # no dynamic conv in PyTorch
        x = self.calib[:16]
        self.assertGreater((q(x).argmax(1) == self.model(x).argmax(1)).float().mean().item(), 0.9)


class TestQAT(unittest.TestCase):
    def test_fake_quant_matches_formula(self):
        x = torch.randn(1000) * 2
        p = calc_qparams(float(x.min()), float(x.max()))
        ref = torch.fake_quantize_per_tensor_affine(x, p.scale, p.zero_point, 0, 255)
        self.assertTrue(torch.allclose(ref, fake_quantize(x, p), atol=1e-6))

    def test_qat_prepares_trains_and_converts(self):
        m, d = trained_synthetic(epochs=2)
        prepared = prepare_for_qat(m)
        fq = [n for n, mod in prepared.named_modules() if "FakeQuantize" in type(mod).__name__]
        self.assertGreater(len(fq), 4)
        x = d.calibration_set(8)
        prepared.train()
        prepared(x).sum().backward()                                # gradients flow through fake quant
        self.assertIsNotNone(prepared.fc1.weight.grad)
        q, hist = quantize_qat(m, d, QATConfig(epochs=2, freeze_bn_epoch=2, freeze_observer_epoch=2))
        self.assertEqual(len(hist), 2)
        self.assertIn("quantized", type(q.conv1).__module__)
        self.assertGreater(evaluate(q, d, d.test)["accuracy"], 0.85)


    def test_one_epoch_qat_converts(self):
        """Freezing observers in the only epoch must not leave them without a range."""
        m, d = trained_synthetic(epochs=1)
        q, hist = quantize_qat(m, d, QATConfig(epochs=1, freeze_bn_epoch=1, freeze_observer_epoch=1))
        self.assertEqual(len(hist), 1)
        self.assertIn("quantized", type(q.fc1).__module__)


class TestFP16(unittest.TestCase):
    def test_support_is_detected_not_assumed(self):
        ok, reason = check_support(QuantCNN().eval())
        self.assertIsInstance(ok, bool)
        self.assertTrue(reason)


class TestSerialization(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        m = QuantCNN(ModelConfig(channels=[8, 16], hidden=32)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            save_checkpoint(path, m, [{"epoch": 1}], TrainConfig(epochs=1), "synthetic")
            again, ck = load_checkpoint(path)
        x = torch.randn(3, 1, 28, 28)
        self.assertTrue(torch.equal(m(x), again(x)))
        self.assertEqual(ck["model_config"]["channels"], [8, 16])

    def test_torchscript_round_trip_for_int8(self):
        m, d = trained_synthetic(epochs=1)
        q, _ = quantize_static(m, d.calibration_set(64), StaticConfig(calibration_images=64))
        with tempfile.TemporaryDirectory() as tmp:
            r = export_torchscript(q, Path(tmp) / "q.ts", d.calibration_set(16, seed=3))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["max_abs_diff"], 0.0)

    def test_onnx_fp32_parity(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        m = QuantCNN(ModelConfig(channels=[8, 16], hidden=32)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            r = export_onnx(m, Path(tmp) / "m.onnx", torch.randn(4, 1, 28, 28), torch.randn(50, 1, 28, 28))
        self.assertTrue(r["ok"], r)


class TestBenchmarks(unittest.TestCase):
    def test_latency_structure_and_reproducible_accuracy(self):
        m, d = trained_synthetic(epochs=1)
        r = latency(m, 8, warmup=2, iters=7)
        self.assertEqual(r["iters"], 7)
        self.assertLessEqual(r["min_ms"], r["median_ms"])
        self.assertLessEqual(r["median_ms"], r["p90_ms"])
        self.assertAlmostEqual(r["throughput_ips"], 8 / (r["median_ms"] / 1000))
        a, b = evaluate(m, d, d.test), evaluate(m, d, d.test)
        self.assertEqual(a["pred"], b["pred"])                     # evaluation is deterministic

    def test_interleaved_latency_times_every_model_every_round(self):
        a = QuantCNN(ModelConfig(channels=[4, 8], hidden=16)).eval()
        b = QuantCNN(ModelConfig(channels=[8, 16], hidden=32)).eval()
        out = interleaved_latency({"a": a, "b": b}, {}, batch_sizes=(1, 4), rounds=3, iters={1: 5, 4: 2}, warmup=1)
        self.assertEqual(set(out), {"a", "b"})
        self.assertEqual(out["a"]["bs1"]["iters"], 15)
        self.assertEqual(out["b"]["bs4"]["iters"], 6)
        lo, hi = out["a"]["bs1"]["round_median_range_ms"]
        self.assertLessEqual(lo, hi)

    def test_compare_math(self):
        rows = compare([{"name": "fp32", "size_bytes": 800, "accuracy": 0.9, "latency": {"bs1": {"median_ms": 2.0}}},
                        {"name": "int8", "size_bytes": 200, "accuracy": 0.88, "latency": {"bs1": {"median_ms": 1.0}}}])
        self.assertEqual(rows[1]["compression"], 4.0)
        self.assertAlmostEqual(rows[1]["accuracy_drop_pts"], 2.0)
        self.assertEqual(rows[1]["speedup_bs1"], 2.0)


if __name__ == "__main__":
    unittest.main()
