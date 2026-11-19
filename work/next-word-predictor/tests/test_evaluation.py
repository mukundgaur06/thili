import math
import unittest

from helpers import toy_model, toy_sentences
from wordflow import evaluate, split_train_test
from wordflow.corpus import CorpusError, build_corpus, corpus_from_upload
from wordflow.evaluation import compare_models, score_sentence
from wordflow.preprocessing import tokenize
from wordflow.service import TrainingConfig, Workspace


class CrossEntropyTests(unittest.TestCase):
    def setUp(self):
        self.model = toy_model(3)

    def test_cross_entropy_matches_hand_calculation(self):
        # Bigram MLE on "i love ai": P(i|<s>)=1, P(love|i)=1, P(ai|love)=1/3, P(</s>|ai)=1
        result = evaluate(self.model, [["i", "love", "ai"]], order=2, k=0)
        self.assertEqual(result.num_tokens, 4)
        self.assertAlmostEqual(result.total_nll, math.log(3))
        self.assertAlmostEqual(result.cross_entropy, math.log(3) / 4)
        self.assertAlmostEqual(result.cross_entropy_bits, math.log2(3) / 4)

    def test_perplexity_is_exp_of_cross_entropy(self):
        result = evaluate(self.model, [["i", "love", "ai"]], order=2, k=0)
        self.assertAlmostEqual(result.perplexity, 3 ** 0.25)
        self.assertAlmostEqual(result.perplexity, math.exp(result.cross_entropy))

    def test_smoothed_hand_calculation(self):
        # add-1 bigram, V = 8: (3+1)/(3+8), (3+1)/(3+8), (1+1)/(3+8), (1+1)/(1+8)
        probs = [4 / 11, 4 / 11, 2 / 11, 2 / 9]
        result = evaluate(self.model, [["i", "love", "ai"]], order=2, k=1)
        self.assertAlmostEqual(result.cross_entropy, -sum(math.log(p) for p in probs) / 4)

    def test_zero_probability_gives_infinite_perplexity(self):
        result = evaluate(self.model, [["i", "python"]], order=2, k=0)
        self.assertEqual(result.zero_prob_tokens, 1)          # i -> python never seen
        self.assertTrue(math.isinf(result.perplexity))
        self.assertIsNone(result.to_dict()["perplexity"])

    def test_smoothing_keeps_perplexity_finite(self):
        result = evaluate(self.model, [["i", "python", "zebra"]], order=3, k=1)
        self.assertTrue(math.isfinite(result.perplexity))

    def test_uniform_model_has_perplexity_v(self):
        # with a huge k every distribution is ~uniform 1/V, so PP -> V
        result = evaluate(self.model, [["zebra", "i"]], order=2, k=1e9)
        self.assertAlmostEqual(result.perplexity, self.model.vocab_size, places=3)

    def test_oov_rate(self):
        result = evaluate(self.model, [["i", "zebra", "love", "yak"]], order=2, k=1)
        self.assertEqual(result.oov_tokens, 2)
        self.assertAlmostEqual(result.oov_rate, 0.5)

    def test_score_sentence_ends_with_eos(self):
        rows = score_sentence(self.model, ["i", "love"], order=2)
        self.assertEqual([w for _, w, _ in rows], ["i", "love", "</s>"])

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.model, [])

    def test_compare_models_line_up(self):
        labels = [r.label for r in compare_models(self.model, toy_sentences(), k=1)]
        self.assertIn("Trigram (MLE)", labels)
        self.assertIn("Bigram + add-1", labels)
        with self.assertRaises(ValueError):
            compare_models(self.model, toy_sentences(), k=0)


class SplitTests(unittest.TestCase):
    SENTENCES = [f"Sentence number {i} is about topic{i}." for i in range(50)] + ["I love AI.", "i love ai"]

    def test_train_and_test_are_disjoint(self):
        train, test = split_train_test(self.SENTENCES, 0.2, seed=1)
        self.assertFalse({tuple(tokenize(s)) for s in train} & {tuple(tokenize(s)) for s in test})
        self.assertEqual(len(train) + len(test), len(self.SENTENCES))

    def test_duplicates_stay_on_one_side(self):
        dupes = {"I love AI.", "i love ai"}
        for seed in range(10):
            train, test = split_train_test(self.SENTENCES, 0.3, seed=seed)
            self.assertTrue(dupes <= set(train) or dupes <= set(test))

    def test_split_is_deterministic(self):
        self.assertEqual(split_train_test(self.SENTENCES, 0.2, 7), split_train_test(self.SENTENCES, 0.2, 7))
        self.assertNotEqual(split_train_test(self.SENTENCES, 0.2, 7), split_train_test(self.SENTENCES, 0.2, 8))

    def test_tiny_corpus_keeps_training_data(self):
        train, test = split_train_test(["a b.", "c d."], 0.9, seed=0)
        self.assertEqual((len(train), len(test)), (1, 1))

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            split_train_test(self.SENTENCES, 1.5)

    def test_test_sentences_are_never_counted(self):
        ws = Workspace(model_path=None, meta_path=None)
        corpus = build_corpus("t", "\n".join(self.SENTENCES), "text")
        state = ws.train(corpus, TrainingConfig(test_fraction=0.3))
        train_words = {w for s in state.train_tokens for w in s}
        test_only = {w for s in state.test_tokens for w in s} - train_words
        self.assertTrue(test_only, "expected some test-only words in this corpus")
        for word in test_only:
            self.assertNotIn(word, state.model.vocab)
        trained_bigrams = set(state.model.ngram_counts(2))
        for sentence in state.test_tokens:
            for pair in zip(sentence, sentence[1:]):
                if set(pair) & test_only:
                    self.assertNotIn(pair, trained_bigrams)


class CorpusValidationTests(unittest.TestCase):
    def test_rejects_non_txt(self):
        with self.assertRaises(CorpusError):
            corpus_from_upload("notes.pdf", b"hello there. general.")

    def test_rejects_empty_and_binary(self):
        with self.assertRaises(CorpusError):
            corpus_from_upload("a.txt", b"   \n ")
        with self.assertRaises(CorpusError):
            corpus_from_upload("a.txt", b"abc\x00def")

    def test_rejects_too_few_sentences(self):
        with self.assertRaises(CorpusError):
            corpus_from_upload("a.txt", b"Only one sentence")

    def test_accepts_and_cleans_valid_upload(self):
        corpus = corpus_from_upload("a.txt", "﻿The cat sat.\r\nThe dog\x07 ran.".encode("utf-8"))
        self.assertEqual(len(corpus.sentences), 2)
        self.assertEqual(tokenize(corpus.sentences[1]), ["the", "dog", "ran"])


if __name__ == "__main__":
    unittest.main()
