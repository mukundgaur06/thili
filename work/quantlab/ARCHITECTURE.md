# Architecture

```
                          ┌──────────────────────────── quantlab/ ────────────────────────────┐
 Fashion-MNIST ──► data.py ──► train.py ──► model.py (QuantCNN, fp32) ──► artifacts/models/fp32.pt
 (torchvision)    uint8 tensors,   AdamW + cosine,          │
                  seeded val split  val each epoch          ├──► quantization/fp16.py     ─┐
                                                            ├──► quantization/dynamic.py  ─┤
                  calibration images (train split only) ───►├──► quantization/static.py   ─┤──► evaluate.py   (accuracy, per-class, confusion)
                                                            └──► quantization/qat.py      ─┘    benchmarks.py (size, latency, memprobe)
                                                                                                export.py     (TorchScript, ONNX + parity)
                                                                         pipeline.py ──► artifacts/results.json, export_manifest.json
                                                                                                      │
                                                                                     app/ (Streamlit) ┘ reads results, runs live jobs
```

## Modules

| file | responsibility |
|---|---|
| `quantlab/data.py` | loads Fashion-MNIST or MNIST through torchvision and keeps the images as uint8 (47 MB). Splits the official training set into 55,000 train and 5,000 validation images with a fixed seed. Normalises per batch. `calibration_set` draws only from the training split. `synthetic()` gives a tiny dataset for tests. |
| `quantlab/model.py` | `QuantCNN`: QuantStub → 2 × (Conv+BN+ReLU, MaxPool) → Linear+ReLU → Dropout → Linear → DeQuantStub. `fuse()` folds BN and merges ReLUs for post-training quantization (eval mode) or for QAT (train mode). |
| `quantlab/train.py` | `train_epochs` (AdamW, cosine schedule, per-epoch validation, hooks used by QAT and the dashboard), checkpoint save and load. |
| `quantlab/evaluate.py` | predictions, accuracy, loss, confusion matrix, per-class precision, recall and F1, and agreement between two models. |
| `quantlab/quantization/utilities.py` | the arithmetic: integer ranges, affine and symmetric parameters, quantize, dequantize, fake quantization, error report and SQNR, and backend selection (`pick_engine`). |
| `quantlab/quantization/dynamic.py` | `torch.ao.quantization.quantize_dynamic` on Linear layers, plus a report of which layers ended up int8. |
| `quantlab/quantization/static.py` | prepare (fuse, QConfig, observers), calibrate, read the observer statistics, convert. Observer type, per-channel and calibration size are configurable. |
| `quantlab/quantization/qat.py` | fuse in QAT mode, fake-quant QConfig, fine-tune with BN and observer freezing, convert. |
| `quantlab/quantization/fp16.py` | casts to half precision and *tests* that it runs on this CPU, returning the reason when it does not. |
| `quantlab/benchmarks.py` | serialized size, a latency suite (warm-up, median, p90, throughput), peak memory through `memprobe` in a fresh process, environment capture, comparison against FP32. |
| `quantlab/memprobe.py` | a separate process that loads one TorchScript model and reports its resident-memory growth. |
| `quantlab/export.py` | TorchScript export for every variant, ONNX export for FP32 and the int8 variants, each reloaded and checked for parity before it is listed. |
| `quantlab/pipeline.py` | runs everything in order and writes `artifacts/results.json` and `artifacts/export_manifest.json`. |
| `app/` | the Streamlit dashboard: an Overview page (`app.py`) and six pages in `pages/`, with shared helpers in `common.py`. |
| `tests/test_quantlab.py` | 26 tests (see the README). |

## Design decisions

* **Eager-mode quantization** (`torch.ao.quantization`). Its prepare, calibrate and convert
  steps are explicit, which suits a lab whose point is to show each step. PyTorch now
  recommends `torchao` / PT2E for new production work; the eager API is deprecated but fully
  functional in 2.10.
* **The quantized engine is detected, not hard-coded.** `pick_engine` prefers `x86`, then
  `fbgemm`, `onednn` and `qnnpack`. The Windows PyTorch 2.10 wheel used here ships only
  `onednn`, and every result records which engine produced it.
* **One pipeline, one set of conditions.** Every variant is evaluated on the same 10,000 test
  images and timed with the same random input, thread count, warm-up and iteration counts.
  The dashboard reads the stored results instead of re-measuring on every page load. Live
  re-runs are explicit and labelled.
* **Memory is measured out of process.** Peak resident memory in the dashboard's own process
  would include Streamlit, pandas and every other model. `memprobe` starts a clean
  interpreter per model.
* **Exports are only listed if they pass.** A file that exports but disagrees with PyTorch
  appears in the manifest with its error and gets no download button.
* **Infrastructure is preserved.** `lfvision/` is unchanged. This project's extra
  dependencies are in `work/quantlab/requirements.txt`.
