# mini-GPT: a GPT built from scratch

A decoder-only transformer language model written from the ground up in PyTorch: a
byte-level BPE tokenizer, embeddings, positional encoding, hand-written causal
self-attention, transformer blocks, a training loop and autoregressive decoding.
**mini-GPT Playground** is an interactive web app that lets you look inside the trained
model.

Drill: *Build GPT from Scratch* (LearnFast, Language Models track). It runs on CPU in the
`lfllm` workbench. The only dependency is `torch`.

```
text ─► BPE tokenizer ─► token + positional embeddings ─► N × [LayerNorm → causal multi-head attention → +residual
                                                              LayerNorm → MLP (GELU)                → +residual]
     ─► LayerNorm ─► linear head (tied) ─► softmax ─► p(next token) ─► greedy / temperature / top-k / top-p
```

## Where each lesson lives

| Lesson | Implementation |
|---|---|
| 1. What a language model is | `GPT.forward` returns a distribution over the next token at every position |
| 2. Text → tokens | `minigpt/tokenizer.py`: byte-level BPE, trained from scratch (`train`, `encode`, `decode`, `<\|endoftext\|>`) |
| 3. Tokens → vectors | `nn.Embedding` token table plus **learned** (GPT-2) or **sinusoidal** (Vaswani) positions (`--pos-encoding`) |
| 4. Self-attention | `scaled_dot_product_attention`: `softmax(QKᵀ/√d_k) V`, multi-head split and merge in `CausalSelfAttention` |
| 5. Look back, never forward | `causal_mask`: lower-triangular mask, future scores set to −∞ (so softmax gives exactly 0) |
| 6. The transformer block | `Block`: pre-norm, hand-written `LayerNorm`, two residual connections, 4× GELU MLP |
| 7. Assemble & train | `GPT` (weight tying, GPT-2 init), `trainer.py`: next-token cross-entropy, AdamW, warm-up + cosine LR, gradient clipping, validation, checkpoints |
| 8. Make it talk | `sampling.py`: greedy, temperature, top-k, top-p (nucleus), repetition penalty, seeded and streaming |
| 9. Ship it | `train.py` / `generate.py` CLIs, `app.py` playground, tests, this README |

None of `nn.MultiheadAttention`, `nn.Transformer*`, `F.scaled_dot_product_attention`,
`nn.LayerNorm` or Hugging Face is used in the model. The tests use PyTorch's versions only
as a reference, to check that the hand-written ones match.

## Quick start

```bash
cd work/mini-gpt
python train.py                     # ~8 min on a laptop CPU; writes checkpoints/best.pt
python generate.py "Once upon a time" --compare
python app.py --open                # playground on http://127.0.0.1:8600
python -m unittest discover -s tests -t .
```

A trained `checkpoints/best.pt` (3.5 MB) is committed, so `generate.py` and `app.py` work
without training. In the workbench container:
`docker compose -f lfllm/docker-compose.yml up`, open a terminal in Jupyter, then run the
same commands from `~/work/mini-gpt`.

### Training options

```bash
python train.py --max-steps 300                          # quick run
python train.py --n-layer 6 --n-embd 192 --block-size 128 # bigger model
python train.py --pos-encoding sinusoidal                 # Vaswani et al. positions
python train.py --corpus path/to/any.txt --vocab-size 1024 # your own text (documents separated by blank lines)
```

## The playground (`python app.py`)

| Page | What you can do |
|---|---|
| **Generate** | Text streams out token by token. Each token is coloured by the model's probability. Hover a token to see the 5 candidates it chose between. Compare all 4 decoding strategies side by side. |
| **Next token** | Type, and the full next-token distribution updates live. It shows raw model probabilities next to what the chosen decoding strategy actually samples from. Click a bar or press <kbd>Tab</kbd> to write the story yourself, one token at a time. |
| **Attention** | Thumbnails of all layer × head attention maps. Click a word to see what it attends to. The full matrix is shown with the masked future hatched out. |
| **Tokenizer** | Live BPE segmentation with ids, compression stats and the first merges learned. |
| **Training** | Train and validation loss curves against the uniform and unigram baselines, the LR schedule and the checkpoint table. |
| **Architecture** | Model diagram and a parameter count per component. |

Shortcuts: <kbd>Alt</kbd>+<kbd>1</kbd>…<kbd>6</kbd> switch pages, <kbd>Ctrl</kbd>+<kbd>Enter</kbd> generates.

## Data

By default `train.py` generates `data/stories.txt`: 3,000 short children's stories from an
original, seeded story grammar (`minigpt/corpus.py`). It has 14 characters with consistent
pronouns, 4 plot types, dialogue and a moral at the end. This is the TinyStories idea
(Eldan & Li, 2023): a model this small cannot learn real English in minutes on a CPU, but it
can learn a controlled language completely. Any text file works through `--corpus`.

How leakage is avoided:

* Stories are split 90/10 into train and validation **before** anything is fitted.
* The BPE merges are learned from training text only.
* Duplicate stories are always kept on the same side of the split.

## Results (default run, CPU)

Default settings: 867,072 parameters, 4 layers × 4 heads, d = 128, 64-token context and a
512-token BPE vocabulary (255 merges, 2.73 characters per token). Training ran for 1,500
steps with batch 32. It took 493 s on CPU over 445k training tokens, scored against 49.8k
validation tokens.

| | val loss (nats/token) | val perplexity |
|---|---:|---:|
| uniform guess, ln V | 6.238 | 512 |
| add-1 unigram baseline | 5.024 | 152 |
| mini-GPT, step 100 | 0.969 | 2.64 |
| mini-GPT, step 500 | 0.281 | 1.32 |
| **mini-GPT, step 1500 (best)** | **0.216** | **1.24** |

Train and validation loss stay together the whole time (0.221 and 0.216 at the end), so the
model is not overfitting. The full log is in `checkpoints/train_log.txt` and the curve is on
the playground's *Training* page.

Sample (`python generate.py "Luna the curious cat" --seed 3`):

> Luna the curious cat wanted to make a gift for her mom. She found a golden star and painted
> it with bright colors. But she dropped it, and it broke. Luna began to cry. A playful puppy
> heard her and came to help. Together they fixed **the crayons**. …

The pronouns stay consistent (`her`/`She`), the plot type holds, and the dialogue punctuation
is correct. The object changes from *a golden star* to *the crayons*. The star was mentioned 58 tokens
earlier, close to the limit of the 64-token window, and keeping track of an entity over that
distance is the first thing a model this small and short-sighted gets wrong.

**How to read these numbers honestly.**

* A validation perplexity close to 1 is **not** "near-perfect English". The corpus comes
  from a small grammar, so once the model knows the template, most tokens are forced:
  pronouns follow names, and phrases repeat inside each plot type.
* What the model really has to predict is the free choices (which character, object or
  feeling comes next). The interesting evidence is therefore:
  * the generated stories keep pronoun agreement with a name chosen tokens earlier;
  * they keep a plot type consistent across sentences;
  * they end with `<|endoftext|>` on their own;
  * the loss falls far below the unigram baseline.
* On real text (`--corpus`), expect a much higher perplexity.

## Tests

```bash
python -m unittest discover -s tests -t .
```

There are 48 tests. They check that:

* Hand-written attention matches `F.scaled_dot_product_attention`, and hand-written LayerNorm
  matches `nn.LayerNorm`.
* The model is causal: changing a future token never changes an earlier logit.
* Attention rows sum to 1 and the masked future is exactly 0.
* Initial loss is about ln V, and the model can overfit a batch.
* BPE round-trips any Unicode string, merges are deterministic, and save/load works.
* Top-k and top-p keep exactly the right tokens, temperature maths is correct, and seeded
  generation is reproducible.
* The LR schedule values are right, and the split is deterministic and leak-free.
* End-to-end training beats the unigram baseline and its checkpoint reloads.
* The playground API works, including input validation.

## Layout

```
minigpt/tokenizer.py   byte-level BPE
minigpt/model.py       LayerNorm, attention, block, GPT
minigpt/data.py        split, encode, batch, unigram baseline
minigpt/corpus.py      story generator
minigpt/trainer.py     training loop, LR schedule, checkpoints
minigpt/sampling.py    decoding strategies, streaming generation
train.py generate.py   CLIs
app.py web/            playground (stdlib HTTP server + vanilla JS, no CDN)
tests/                 unittest suite
```

## Limitations

* Tiny by design: about 0.9M parameters, a 64-token context, and a 512-token vocabulary.
  GPT-2 small is about 140× larger.
* There is no KV cache. Each generated token re-runs the cropped context, which is fine at
  this size.
* The synthetic corpus makes the model fluent in one narrow genre only.
