# QuantLab: Neural Network Quantization & Optimization Studio

QuantLab trains a small CNN on **Fashion-MNIST** and compresses it five ways: FP32, FP16,
dynamic INT8, static INT8 (post-training quantization) and quantization-aware training. It
measures what each one costs and saves: accuracy, size, latency, throughput and memory. A
Streamlit dashboard lets you train, calibrate, convert and benchmark live, and work through
the arithmetic in a Mathematics Lab.

Everything runs on a CPU. Every number in this repository was measured; see
[`BENCHMARKS.md`](BENCHMARKS.md).

## Results at a glance

PyTorch 2.10 CPU, oneDNN engine, 4 threads, 10,000 test images:

| Model | Size | Test accuracy | Batch 64 latency | Speed-up |
|---|---:|---:|---:|---:|
| FP32 baseline | 814 KB | 91.90% | 9.46 ms | 1.00× |
| FP16 (CPU) | 410 KB | 91.89% | 88.39 ms | 0.11× |
| Dynamic INT8 | 224 KB | 91.93% | 9.72 ms | 0.97× |
| **Static INT8 (PTQ)** | **214 KB** | 91.78% | **3.43 ms** | **2.76×** |
| **QAT INT8** | **214 KB** | 91.95% | **3.24 ms** | **2.92×** |

Static INT8 is 3.8× smaller and 2.8× faster for −0.12 points of accuracy, which is within
noise: one standard error is about ±0.27 points.

**Caveats:**

* At batch 1, int8 is *slower* than fp32 (1.1 ms vs 0.6 ms), because fixed overhead dominates.
* Dynamic INT8 only shrinks this model; it doesn't speed it up, because the convolutions stay fp32.
* FP16 runs on this x86 CPU but is 9× slower, because the CPU has no native fp16 arithmetic.

`BENCHMARKS.md` explains each of these.

## What is implemented

| Requirement | Where |
|---|---|
| FP32 baseline: architecture, data, training and validation loops, evaluation, serialization | `quantlab/model.py`, `data.py`, `train.py`, `evaluate.py` |
| Quantization arithmetic: scale, zero point, ranges, integer codes, dequantization error, symmetric and asymmetric | `quantlab/quantization/utilities.py`, explained in [`MATHEMATICS.md`](MATHEMATICS.md) |
| Dynamic INT8 | `quantlab/quantization/dynamic.py` |
| Static INT8: fuse, prepare, calibrate, observer statistics, scales and zero points, convert | `quantlab/quantization/static.py` |
| QAT with fake quantization, BN and observer freezing, conversion | `quantlab/quantization/qat.py` |
| FP16 with a real support check | `quantlab/quantization/fp16.py` |
| Benchmarks: size, compression, accuracy drop, median latency, throughput, peak memory | `quantlab/benchmarks.py`, `quantlab/memprobe.py` |
| Export with verification: TorchScript (all variants), ONNX (FP32, static and QAT INT8) | `quantlab/export.py` |
| End-to-end run | `quantlab/pipeline.py` → `artifacts/results.json` |
| Dashboard | `app/` (see below) |
| Tests | `tests/test_quantlab.py`: 26 tests |

## Run it

From the repository root, in the LearnFast **lfvision** workbench (it provides torch and
torchvision):

```bash
docker compose -f lfvision/docker-compose.yml up
# in the workbench terminal:
cd work/quantlab
pip install -r requirements.txt            # streamlit, plotly, onnx, onnxruntime, psutil
```

Or on any machine with Python 3.10+ and CPU PyTorch:

```bash
cd work/quantlab
pip install torch torchvision -r requirements.txt

python -m quantlab.pipeline                # download data, train, quantize, evaluate, benchmark, export (~5 min)
python -m unittest discover -s tests -t .  # 26 tests, ~10 s
streamlit run app/app.py                   # the dashboard, http://localhost:8501
```

Pipeline options: `--epochs 8`, `--qat-epochs 2`, `--calibration-images 1024`, `--threads 4`,
`--dataset mnist`, `--skip-train` (reuse `artifacts/models/fp32.pt`), `--skip-ablations`.

The lfvision container publishes only Jupyter's port (8889), and the workbench files are
deliberately left unchanged. To view the dashboard from inside the container, add `"8501:8501"`
to that compose file's `ports`, or run the dashboard on the host as above.

## The dashboard

| Page | What it does |
|---|---|
| **Overview** | headline metrics, a diagram of where the quantization boundaries sit, size by precision, and a table comparing the techniques |
| **Model Training** | the baseline's loss and accuracy curves and layer table; train a new model live (dataset, channels, hidden units, batch size, epochs, learning rate) with a progress bar and live curves |
| **Quantization Lab** | pick a checkpoint and a method, set the observer, per-channel weights, calibration size or QAT epochs, run it, and see each step, the measured observer ranges, scales and zero points, and the result |
| **Benchmark Studio** | the full measured table; re-run an interleaved latency benchmark live at any batch size and thread count |
| **Visual Analytics** | accuracy, size and latency charts, compression versus accuracy, per-class metrics, confusion matrices and their difference from FP32, the calibration ablation, and per-layer weight SQNR |
| **Mathematics Lab** | quantize a number by hand; the error, clipping and SQNR of whole tensors from 2 to 8 bits; MinMax vs histogram calibration with outliers; fake quantization and the straight-through gradient |
| **Model Export** | every export with its parity check; download buttons only for files that passed |

All seven pages were run headlessly with Streamlit's `AppTest` with no exceptions. The
interactive paths were also run: training, a conversion with every method in the lab, and a
live benchmark. A real server start answered its health check.

## How the five methods differ

| | FP16 | Dynamic INT8 | Static INT8 | QAT INT8 |
|---|---|---|---|---|
| Weights | fp16 | int8 (Linear only) | int8, per-channel | int8, per-channel |
| Activations | fp16 | quantized per call at run time | uint8, ranges fixed by calibration | uint8, ranges learned during QAT |
| Extra data | none | none | ~1k calibration images | the training set |
| Extra compute | none | none | seconds | a short fine-tune (2 epochs here) |
| Limitation | needs fp16 hardware | no Conv2d support; run-time overhead | outliers and distribution shift hurt fixed ranges | needs the training pipeline and time |

[`MATHEMATICS.md`](MATHEMATICS.md) has the equations with a worked example.
[`ARCHITECTURE.md`](ARCHITECTURE.md) shows how the modules connect.

## Tests

`python -m unittest discover -s tests -t .` runs 26 tests. Quantization formulas are checked
against PyTorch's own kernels, a separate implementation, and never against values produced
by the code under test:

* scale and zero point equal `MinMaxObserver`'s, affine and symmetric, on three
  distributions;
* integer codes equal `torch.quantize_per_tensor(...).int_repr()`;
* fake quantization equals `torch.fake_quantize_per_tensor_affine`;
* rounding error ≤ s/2 inside the range, 0.0 is exact, and clipping is counted;
* calibration: MinMax observers hold exactly the min and max that forward hooks recorded
  independently, and uncalibrated observers hold no range;
* static conversion yields quantized modules, stays within 5 points of FP32 on a trained
  model, and shrinks the state_dict below 40% of FP32;
* dynamic quantization swaps the Linear layers and leaves the convolutions fp32;
* QAT inserts fake-quant modules, gradients flow, and 1- and 2-epoch runs convert;
* fusion preserves outputs; checkpoint, TorchScript (int8) and ONNX round trips;
* latency statistics are consistent, evaluation is deterministic, and the interleaved timer
  covers every model in every round;
* the confusion matrix and per-class metrics are checked against hand-computed values;
* Fashion-MNIST split sizes, seeding and pixel statistics. This test is skipped if the
  dataset hasn't been downloaded.

## Limitations

* **One machine, one engine.** This Windows PyTorch wheel ships only the oneDNN quantized
  engine; x86/fbgemm (Linux), qnnpack (ARM) and GPUs were not measured. Speed-ups are
  hardware-specific.
* **A small, easy model.** At 207k parameters, int8 costs almost nothing in accuracy and
  calibration settings barely matter. The ablation shows it. Larger models with outlier
  activations, such as transformers, are where observers and QAT earn their keep.
* **Eager-mode `torch.ao.quantization` is deprecated** in favour of `torchao` and PT2E. It is
  used here because it makes each step explicit, and it is fully functional in 2.10.
* **Memory** is resident-memory growth of a fresh process. It is honest, but coarse at this
  model size, where activations dominate.
* **No int4 model.** Sub-8-bit arithmetic is only shown in the Mathematics Lab.

## Layout

```
quantlab/                 the library
  data.py model.py train.py evaluate.py benchmarks.py memprobe.py export.py pipeline.py
  quantization/           utilities.py dynamic.py static.py qat.py fp16.py
app/                      Streamlit dashboard: app.py (Overview), pages/1-6, common.py
tests/test_quantlab.py
artifacts/                results.json, export_manifest.json, pipeline_log.txt (models/ is regenerated)
README.md MATHEMATICS.md ARCHITECTURE.md BENCHMARKS.md requirements.txt
```

## Reference

Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference*, arXiv:1712.05877.
