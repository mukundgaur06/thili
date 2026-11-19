# Benchmarks

All numbers below were measured by `python -m quantlab.pipeline` on one machine and copied from
`artifacts/results.json`. Run the pipeline yourself and they will differ somewhat, especially
the latencies. Nothing here was measured on a GPU, ARM or Apple silicon, so no claims are made
about those.

**Setup**

| | |
|---|---|
| CPU | Intel64 Family 6 Model 170 (a 14-thread laptop CPU), Windows 11 |
| Software | PyTorch 2.10.0+cpu, Python 3.10, quantized engine **oneDNN** (the only engine this Windows wheel ships) |
| Threads | 4 for every measurement |
| Data | Fashion-MNIST: 55,000 train, 5,000 validation and 10,000 test images (the official test split, never used for training or selection) |
| Model | `QuantCNN`, 206,970 parameters, trained 8 epochs with AdamW and a cosine schedule (141 s) |
| Whole pipeline | 267 s |

## Main results

| Model | Size (KB) | × smaller | Test accuracy | Δ vs FP32 | Same prediction as FP32 | Batch 1 median (ms) | Batch 64 median (ms) | Batch 64 throughput (img/s) | Speed-up at batch 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 814.0 | 1.00 | 91.90% | – | 100% | 0.594 | 9.46 | 6,768 | 1.00× |
| FP16 (CPU) | 409.8 | 1.99 | 91.89% | −0.01 | 99.99% | 5.593 | 88.39 | 724 | **0.11×** |
| Dynamic INT8 | 223.6 | 3.64 | 91.93% | +0.03 | 99.75% | 1.058 | 9.72 | 6,581 | 0.97× |
| Static INT8 (PTQ) | 213.6 | 3.81 | 91.78% | −0.12 | 99.23% | 1.127 | 3.43 | 18,677 | **2.76×** |
| QAT INT8 | 213.6 | 3.81 | 91.95% | +0.05 | 99.10% | 1.072 | 3.24 | 19,770 | **2.92×** |

* **Size** is the serialized `state_dict`. INT8 is 3.8× smaller, not 4×, because biases,
  per-channel scales and zero points stay in fp32 or int64.
* **Latency** comes from 10 untimed warm-up runs, then 5 interleaved rounds with every model
  timed in turn each round. That is 500 timed runs at batch 1 and 60 at batch 64, pooled
  into one median.
* **Round-to-round spread** of the per-round medians at batch 64: FP32 9.14–9.93 ms, static
  INT8 3.21–4.02 ms, QAT INT8 3.02–3.62 ms. Static and QAT run the same int8 kernels, and
  their ranges overlap: they are equally fast.

### What the table says

1. **Static INT8 is the practical winner on this CPU.** It is 3.8× smaller and 2.8× faster at
   batch 64, for −0.12 accuracy points. It gives a different prediction from FP32 on 77 of
   the 10,000 test images: it fixes 29 that FP32 got wrong, breaks 41 that FP32 got right, and
   7 are wrong either way. QAT changes 90 predictions (44 fixed, 39 broken, 7 wrong either
   way); dynamic INT8 changes 25 (15 fixed, 9 broken, 1 wrong either way).
2. **Accuracy differences between the int8 variants are within noise.** One standard error of
   an accuracy near 92% on 10,000 images is about 0.27 points. QAT's +0.17 points over static
   PTQ is real for this run but not statistically meaningful. This small CNN is easy to
   quantize; QAT's advantage grows on models that lose more under PTQ (depthwise convolutions,
   very low bit widths).
3. **Batch 1 is slower in int8 (about 1.1 ms vs 0.59 ms).** For a single 28×28 image, the
   fixed costs dominate: quantizing the input, the oneDNN kernel dispatch and dequantizing the
   output. The arithmetic saved is tiny. Int8 pays off once there is real work per call.
4. **Dynamic INT8 shrinks the model but doesn't speed it up.** Only the two Linear layers are
   int8 (PyTorch has no dynamic Conv2d). The convolutions, which dominate the compute, stay
   fp32, and every Linear call adds a range computation and quantize/dequantize. It is a size
   optimisation here; it is a speed optimisation for Linear-heavy models such as LSTMs and
   transformer MLPs.
5. **FP16 on this x86 CPU is 9× slower.** It runs (the support check passed), and storage
   halves, but there is no native fp16 arithmetic, so PyTorch converts on the fly. FP16 is a
   GPU and Apple-silicon optimisation; on this machine it is the wrong choice. That was
   measured here, not assumed.

## Peak memory

Measured in a fresh Python process per model (`quantlab.memprobe`), loading the TorchScript
export and running 20 batches of 64:

| Model | RSS growth on load (MB) | Peak RSS growth over idle torch (MB) |
|---|---:|---:|
| FP32 | 4.04 | 20.85 |
| FP16 | 3.91 | 15.88 |
| Dynamic INT8 | 5.86 | 22.24 |
| Static INT8 | 8.84 | 17.89 |
| QAT INT8 | 7.86 | 16.93 |

With under 1 MB of weights, memory is dominated by activations, kernel workspaces and code
the quantized kernels load. Int8 loading costs a few MB *more* (the quantized operator
libraries), and its peak is a few MB lower (uint8 activations). For a model this small the
differences are real but irrelevant; they matter for models whose weights are hundreds of MB.

## How well the int8 weights match

SQNR of each int8 weight tensor against the fp32 weights after BN folding (higher is
closer; about 6 dB per bit):

| Layer | Static INT8 | QAT INT8 |
|---|---:|---:|
| conv1 | 47.6 dB | 37.8 dB |
| conv2 | 43.3 dB | 30.7 dB |
| fc1 | 39.8 dB | 27.9 dB |
| fc2 | 42.6 dB | 25.2 dB |

Static PTQ rounds the trained weights, and the 40–48 dB matches what 8 bits should give. QAT's
weights are lower *against the original FP32 weights* because fine-tuning moved them. They
are the weights the int8 model was trained to use.

## Static PTQ ablation

12 conversions of the same FP32 model: two observers, per-tensor or per-channel weights, and
32, 256 or 2,048 calibration images drawn from the training split.

| Observer | Weights | 32 images | 256 images | 2,048 images |
|---|---|---:|---:|---:|
| MinMax | per-tensor | 91.88% | 91.82% | 91.88% |
| MinMax | per-channel | 91.81% | 91.81% | 91.83% |
| Histogram | per-tensor | 91.91% | 91.82% | 91.93% |
| Histogram | per-channel | 91.91% | 91.86% | 91.85% |

Every setting lands within 0.12 points of FP32 (91.90%), inside the noise. On this model,
even 32 calibration images are enough and the observer choice doesn't matter. The
Mathematics Lab shows a tensor with outliers where it *does* matter: MinMax stretches its
range to the outlier, and the histogram observer clips it. Expect these settings to matter
on larger models with outlier activations, such as transformers.

## Per-class F1 (FP32 vs static INT8)

| Class | FP32 | Static INT8 |
|---|---:|---:|
| T-shirt/top | 0.872 | 0.867 |
| Trouser | 0.987 | 0.987 |
| Pullover | 0.871 | 0.875 |
| Dress | 0.920 | 0.918 |
| Coat | 0.870 | 0.869 |
| Sandal | 0.986 | 0.986 |
| Shirt | 0.763 | 0.754 |
| Sneaker | 0.965 | 0.964 |
| Bag | 0.983 | 0.983 |
| Ankle boot | 0.972 | 0.971 |

The images that change answer are borderline cases among the classes FP32 already confuses
(Shirt, T-shirt/top, Pullover, Coat). The Visual Analytics page shows the difference of the
two confusion matrices.

## Exports, each verified after writing

| Model | Format | Size (KB) | Check | Result |
|---|---|---:|---|---|
| FP32 | TorchScript | 833.6 | reload, compare logits | identical |
| FP32 | ONNX | 809.8 | ONNX Runtime 1.23.2 on all 10,000 test images | 100% same predictions, max Δ 1.9e-6 |
| FP16 | TorchScript | 430.8 | reload, compare logits | identical |
| Dynamic INT8 | TorchScript | 250.2 | reload, compare logits | identical |
| Static INT8 | TorchScript | 238.5 | reload, compare logits | identical |
| Static INT8 | ONNX (QDQ int8) | 214.9 | ONNX Runtime on all 10,000 test images | 99.95% same predictions; max Δ = 1.0 output step |
| QAT INT8 | TorchScript | 240.1 | reload, compare logits | identical |
| QAT INT8 | ONNX (QDQ int8) | 214.9 | ONNX Runtime on all 10,000 test images | 99.92% same predictions; max Δ = 1.0 output step |

The int8 ONNX graphs run on ONNX Runtime's own integer kernels. They round some values one
step differently from PyTorch's oneDNN kernels, which flips 5–8 borderline predictions out of
10,000. Not exported: dynamic INT8 to ONNX (no standard path in this PyTorch version) and FP16
to ONNX (not tested, so not offered).

## Not measured

* GPU, ARM (qnnpack), Apple silicon and the x86/fbgemm engines are not available in this build.
* Energy use.
* Int4 or mixed precision. The Mathematics Lab demonstrates 2–8-bit arithmetic, but no model
  was converted below 8 bits.
