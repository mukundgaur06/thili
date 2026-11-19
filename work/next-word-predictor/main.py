"""WordFlow command line: split, train, predict, generate and evaluate.

    python main.py split                       # write corpus/train.txt and corpus/test.txt
    python main.py train                       # fit on train.txt, save models/wordflow_model.json
    python main.py predict "i love machine"    # inference from the saved model
    python main.py generate "the model" --seed 7
    python main.py evaluate                    # held-out metrics on test.txt
    python main.py demo                        # all of the above in one go

Training and inference are separate steps: ``predict`` and ``generate`` only ever load
the saved model file, they never look at the corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wordflow.corpus import CorpusError, corpus_from_file, split_train_test  # noqa: E402
from wordflow.evaluation import compare_models  # noqa: E402
from wordflow.generation import TextGenerator  # noqa: E402
from wordflow.ngram_model import NGramModel  # noqa: E402
from wordflow.predictor import NextWordPredictor, order_name  # noqa: E402
from wordflow.preprocessing import tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
DEFAULT_MODEL = ROOT / "models" / "wordflow_model.json"


def read_sentences(path: Path) -> list[list[str]]:
    """Load a one-sentence-per-line file (as written by ``split``) into token lists."""
    corpus = corpus_from_file(path, line_mode="line")
    return [tokenize(sentence) for sentence in corpus.sentences]


def cmd_split(args: argparse.Namespace) -> None:
    corpus = corpus_from_file(args.corpus)
    train, test = split_train_test(corpus.sentences, args.test_fraction, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.out_dir / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    print(f"{corpus.name}: {len(corpus.sentences)} sentences -> "
          f"{len(train)} train / {len(test)} test (seed={args.seed}, "
          f"test_fraction={args.test_fraction}) written to {args.out_dir}")


def cmd_train(args: argparse.Namespace) -> None:
    sentences = read_sentences(args.train)
    model = NGramModel(args.order, args.min_count).fit(sentences)
    path = model.save(args.model)
    print(f"Trained {model}")
    for order in range(1, model.order + 1):
        print(f"  {order_name(order):8s} {model.num_ngrams(order):6d} distinct / "
              f"{model.total_ngrams(order):6d} total")
    print(f"Saved to {path}")


def cmd_predict(args: argparse.Namespace) -> None:
    model = NGramModel.load(args.model)
    result = NextWordPredictor(model).suggest_next(
        args.text, top_k=args.top_k, smoothing=not args.no_smoothing, k=args.k,
        order=min(args.order, model.order),
    )
    print(f'Input: "{args.text}"')
    for rank, s in enumerate(result.suggestions, 1):
        print(f"  {rank}. {s.word:<16s} P = {s.probability:.4f}   (seen {s.count}x)")
    print(result.explanation)


def cmd_generate(args: argparse.Namespace) -> None:
    model = NGramModel.load(args.model)
    result = TextGenerator(model).generate_text(
        args.seed_text, max_words=args.max_words, temperature=args.temperature,
        order=min(args.order, model.order), seed=args.seed, stop_at_eos=not args.keep_going,
    )
    print(result.text)
    print(f"[{len(result.steps)} steps, stopped by {result.stop_reason}, seed={args.seed}]")


def cmd_evaluate(args: argparse.Namespace) -> None:
    model = NGramModel.load(args.model)
    test = read_sentences(args.test)
    print(f"Held-out evaluation on {args.test.name}: {len(test)} sentences, add-k with k={args.k:g}")
    print(f"{'model':28s} {'tokens':>7s} {'H (nats)':>9s} {'perplexity':>11s} {'zero-P':>7s} {'OOV':>7s}")
    for r in compare_models(model, test, args.k):
        h = f"{r.cross_entropy:9.4f}" if r.is_finite else "      inf"
        pp = f"{r.perplexity:11.2f}" if r.is_finite else "        inf"
        print(f"{r.label:28s} {r.num_tokens:7d} {h} {pp} {r.zero_prob_tokens:7d} {r.oov_rate:7.2%}")


def cmd_demo(args: argparse.Namespace) -> None:
    cmd_split(args)
    cmd_train(args)
    print()
    cmd_predict(args)
    print()
    cmd_generate(args)
    print()
    cmd_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", type=Path, default=CORPUS_DIR / "corpus.txt")
    common.add_argument("--out-dir", type=Path, default=CORPUS_DIR)
    common.add_argument("--train", type=Path, default=CORPUS_DIR / "train.txt")
    common.add_argument("--test", type=Path, default=CORPUS_DIR / "test.txt")
    common.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    common.add_argument("--order", type=int, default=3, help="n-gram order (default 3)")
    common.add_argument("--min-count", type=int, default=1, help="vocabulary frequency cut-off")
    common.add_argument("--test-fraction", type=float, default=0.2)
    common.add_argument("--seed", type=int, default=42, help="split / sampling seed")
    common.add_argument("--k", type=float, default=1.0, help="add-k smoothing parameter")
    common.add_argument("--top-k", type=int, default=5)
    common.add_argument("--no-smoothing", action="store_true", help="predict with plain MLE")
    common.add_argument("--max-words", type=int, default=30)
    common.add_argument("--temperature", type=float, default=1.0)
    common.add_argument("--keep-going", action="store_true", help="do not stop at end of sentence")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("split", parents=[common], help="deterministic train/test split").set_defaults(func=cmd_split)
    sub.add_parser("train", parents=[common], help="fit and save the model").set_defaults(func=cmd_train)
    predict = sub.add_parser("predict", parents=[common], help="suggest next words")
    predict.add_argument("text")
    predict.set_defaults(func=cmd_predict)
    generate = sub.add_parser("generate", parents=[common], help="sample text")
    generate.add_argument("seed_text", nargs="?", default="")
    generate.set_defaults(func=cmd_generate)
    sub.add_parser("evaluate", parents=[common], help="held-out metrics").set_defaults(func=cmd_evaluate)
    demo = sub.add_parser("demo", parents=[common], help="run the full pipeline")
    demo.set_defaults(func=cmd_demo, text="i love machine", seed_text="the model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (CorpusError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
