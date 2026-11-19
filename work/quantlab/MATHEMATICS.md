# The mathematics of quantization

Everything here is implemented in `quantlab/quantization/utilities.py`. The tests check it
against PyTorch's own kernels, a separate implementation: `MinMaxObserver` for the
parameters, `torch.quantize_per_tensor` for the codes and `torch.fake_quantize_per_tensor_affine`
for fake quantization.

## 1. The two equations

A real number `x` is stored as an integer code `q` with two parameters: a **scale** `s`, the
size of one integer step in real units, and a **zero point** `z`, the integer that stands for
0.0.

```
quantize:    q = clamp( round(x / s) + z ,  q_min, q_max )
dequantize:  x̂ = s · (q − z)
```

`round` is round-half-to-even (`torch.round`), as in PyTorch's kernels. For a value inside the
representable range, the error is at most half a step:

```
|x − x̂| ≤ s / 2
```

Outside the range the value is **clipped** to the nearest end, and the error can be
arbitrarily large. Choosing the range is a trade-off:

* a wide range means a large step, so every value gets rounded more;
* a narrow range means a small step, but more values are clipped.

## 2. Integer ranges

| type | bits | q_min | q_max |
|---|---|---|---|
| uint8 (`quint8`), used for activations | 8 | 0 | 255 |
| int8 (`qint8`), used for weights | 8 | −128 | 127 |
| int4 | 4 | −8 | 7 |

## 3. Affine (asymmetric) quantization

The observed range `[min, max]` is first widened to include zero: `min ← min(min, 0)`,
`max ← max(max, 0)`. Then

```
s = (max − min) / (q_max − q_min)
z = q_min − round(min / s)        (clamped to [q_min, q_max])
```

Zero is then exactly representable: `x = 0` maps to `q = z` with no error. This matters
because zero padding and ReLU outputs are full of exact zeros.

**Worked example.** Activations observed in `[−1.5, 4.0]`, stored as uint8:

```
s = (4.0 − (−1.5)) / 255         = 0.021569
z = 0 − round(−1.5 / 0.021569)   = 0 − round(−69.55) = 70
x = 1.2345:   q = round(1.2345 / 0.021569) + 70 = round(57.24) + 70 = 127
              x̂ = 0.021569 · (127 − 70)          = 1.2294     error −0.0051  (≤ s/2 = 0.0108)
x = 0.0:      q = 70,  x̂ = 0.0 exactly
x = 9.0:      q = round(417.3) + 70 = 487 → clamped to 255,  x̂ = 3.9902   (clipped)
```

## 4. Symmetric quantization

The range is `[−a, a]` with `a = max(|min|, |max|)`:

```
s = a / ((q_max − q_min) / 2)       for int8: s = a / 127.5
z = 0
```

Symmetric quantization is cheaper to multiply. With `z = 0`, the cross terms in an integer
matrix product vanish, so it is the standard choice for **weights**, which are roughly
centred on 0. It is a poor choice for ReLU activations: they are never negative, so half the
codes go unused and the step is twice as coarse as affine would give. The Mathematics Lab
page shows this live.

## 5. Per-tensor versus per-channel

A convolution with 32 output channels can use one scale for the whole weight tensor, or one
scale per output channel. Per-channel scales follow each filter's own range, so a filter
with small weights isn't rounded with the coarse step of the largest one. The cost is 32
scales instead of 1 (128 bytes). This project uses per-channel symmetric int8 weights by
default; the ablation in `BENCHMARKS.md` compares the two.

## 6. Measuring the error: SQNR

```
SQNR (dB) = 10 · log10( mean(x²) / mean((x − x̂)²) )
```

For a uniform quantizer over a well-used range, each extra bit adds about 6.02 dB, so 8 bits
give roughly 40–50 dB on real weights. `results.json` reports the SQNR of every int8 weight
tensor.

## 7. How an int8 layer computes

For `y = W x` with `W ≈ s_w (q_w − z_w)` and `x ≈ s_x (q_x − z_x)`:

```
y ≈ s_w s_x · Σ (q_w − z_w)(q_x − z_x)
```

The sum is computed in int32 accumulators from int8 inputs. It is then rescaled by
`s_w s_x / s_y` into the output's own uint8 grid (Jacob et al., 2018). Between layers only
uint8 tensors move through memory, a quarter of the bytes of fp32. That, plus integer SIMD
instructions, is where the speed-up comes from.

## 8. Dynamic versus static activations

* **Dynamic:** for every input, the layer computes `s` and `z` from that input's actual min
  and max at run time. It is always well fitted, but costs a min/max pass and a
  quantize/dequantize around every layer.
* **Static:** `s` and `z` are fixed ahead of time from **calibration**: the observers record
  the range of each activation over about 1,000 real training images. There is no run-time
  overhead, but an input far outside the calibration range is clipped.

Observers implemented in the lab:

| observer | range estimate |
|---|---|
| `MinMaxObserver` | the extreme min and max seen; one outlier stretches the range |
| `MovingAverageMinMaxObserver` | exponential moving average of per-batch min and max |
| `HistogramObserver` | builds a histogram, then searches for the clipping range that minimises the L2 quantization error |

## 9. Fake quantization and QAT

During quantization-aware training every weight and activation passes through

```
fake_quant(x) = s · ( clamp(round(x / s) + z, q_min, q_max) − z )
```

The value stays a float but carries exactly the rounding error the int8 model will have.
`round` has a gradient of zero almost everywhere, so training uses the **straight-through
estimator**:

```
∂ fake_quant(x) / ∂x  =  1   if x is inside the representable range
                          0   otherwise (clipped values get no gradient)
```

The loss now "sees" the quantization error, and the optimiser moves the weights toward values
that survive rounding. Late in QAT, the BatchNorm statistics and observer ranges are frozen,
so the model that is converted is exactly the model that was last trained.

## 10. FP16

IEEE half precision has a 1 sign bit, a 5-bit exponent and a 10-bit mantissa: about 3
significant decimal digits, a range up to ±65,504, and machine epsilon 9.8e-4. Unlike int8,
it needs no scale or zero point. Whether it is faster depends entirely on the hardware having
fp16 arithmetic. On this project's x86 CPU it runs, but slower than fp32 (see
`BENCHMARKS.md`).

## Reference

Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference.* CVPR. arXiv:1712.05877.
