# Projects

Six projects, each in its own folder with its own README, tests and a demo you can open in a
browser. Every number in their READMEs was measured; none is illustrative.

| # | Project | What it is | Tests | Demo |
|---|---|---|---:|---|
| 1 | [`next-word-predictor`](next-word-predictor/) | **WordFlow**: unigram, bigram and trigram language models with smoothing, generation and perplexity | 93 | `python app.py` → http://127.0.0.1:8501 |
| 2 | [`mini-gpt`](mini-gpt/) | **mini-GPT**: a decoder-only transformer from scratch (BPE tokenizer, causal self-attention, training, sampling) | 48 | `python app.py` → http://127.0.0.1:8600 |
| 3 | [`lora-from-scratch`](lora-from-scratch/) | **LoRA from Scratch**: LoRA and QLoRA (NF4) written by hand, fine-tuning real GPT-2 on SST-2; merge and adapter hot-swap | 28 | `streamlit run dashboard.py --server.port 8620` |
| 4 | [`legal-clause-gpt`](legal-clause-gpt/) | **Legal-Clause GPT**: GPT-2 fine-tuned with the Hugging Face Trainer to tag LEDGAR contract clauses with one of 100 types | 12 | `python app.py` → http://127.0.0.1:8610 |
| 5 | [`quantlab`](quantlab/) | **QuantLab**: FP32, FP16, dynamic INT8, static INT8 and QAT on a Fashion-MNIST CNN, benchmarked | 26 | `streamlit run app/app.py --server.port 8630` |
| 6 | [`basketmind`](basketmind/) | **BasketMind**: exact Apriori vs a continued-pretrained GPT-2 for market-basket recommendation, with bootstrapped evaluation | 29 | `streamlit run app/app.py --server.port 8640` |

Run each demo from inside its project folder. The ports are different, so all six can run at
once.

## Running the tests

```bash
cd next-word-predictor && python -m unittest discover -s tests
cd mini-gpt            && python -m unittest discover -s tests -t .
cd lora-from-scratch   && python -m unittest discover -s tests -t .
cd legal-clause-gpt    && python -m unittest discover -s tests -t .
cd quantlab            && python -m unittest discover -s tests -t .
cd basketmind          && python -m unittest discover -s tests -t .
```

## What each demo needs

Everything runs on a CPU.

* **WordFlow, mini-GPT:** nothing beyond the repository. mini-GPT's trained checkpoint is
  committed.
* **LoRA:** GPT-2 in the Hugging Face cache. The trained adapters (KB-sized) are committed.
* **Legal-Clause GPT:** the fine-tuned model (about 500 MB) is not committed. Rebuild it with
  `python train.py`, which takes about 40 minutes and can be paused and resumed.
* **QuantLab:** `python -m quantlab.pipeline` first (about 5 minutes). It downloads
  Fashion-MNIST and writes the models the dashboard loads; the measured results are committed.
* **BasketMind:** `python -m basketmind.experiments.run generate --dataset full` first, to
  rebuild the synthetic shop deterministically. The trained adapter and all results are
  committed.

Workbench images: `lfllm` (projects 1–4), `lfvision` (QuantLab) and `lfretail` (BasketMind).
Each project's README lists the few extra packages it needs, such as Streamlit.
