import tempfile
import unittest
from pathlib import Path

from helpers import toy_model
from wordflow import NGramModel
from wordflow.preprocessing import BOS, EOS


class NGramCountTests(unittest.TestCase):
    def setUp(self):
        self.model = toy_model(3)

    def test_unigram_counts(self):
        counts = self.model.ngram_counts(1)
        self.assertEqual(counts[("i",)], 3)
        self.assertEqual(counts[("love",)], 3)
        self.assertEqual(counts[("python",)], 1)
        self.assertEqual(counts[(EOS,)], 3)                 # one end token per sentence
        self.assertEqual(self.model.context_count(()), 13)  # 10 words + 3 </s>

    def test_bigram_counts_from_the_specification(self):
        counts = self.model.ngram_counts(2)
        self.assertEqual(counts[("i", "love")], 3)
        self.assertEqual(counts[("love", "python")], 1)
        self.assertEqual(counts[("love", "machine")], 1)
        self.assertEqual(counts[("love", "ai")], 1)
        self.assertEqual(counts[(BOS, "i")], 3)

    def test_trigram_counts(self):
        counts = self.model.ngram_counts(3)
        self.assertEqual(counts[(BOS, BOS, "i")], 3)
        self.assertEqual(counts[(BOS, "i", "love")], 3)
        self.assertEqual(counts[("i", "love", "machine")], 1)
        self.assertEqual(counts[("machine", "learning", EOS)], 1)

    def test_no_ngram_spans_two_sentences(self):
        for left, right in self.model.ngram_counts(2):
            self.assertNotEqual(left, EOS, "</s> must never be a context")
            self.assertNotIn((left, right), {("python", "i"), ("learning", "i"), ("ai", "i")})

    def test_generalised_order(self):
        self.assertEqual(toy_model(4).ngram_counts(4)[(BOS, "i", "love", "ai")], 1)

    def test_lower_order_tables_match_standalone_models(self):
        self.assertEqual(toy_model(3).ngram_counts(2), toy_model(2).ngram_counts(2))

    def test_totals_equal_sum_of_followers(self):
        for order in (1, 2, 3):
            for context, total in self.model.context_counts(order).items():
                self.assertEqual(total, sum(self.model.followers(context).values()))

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = NGramModel.load(self.model.save(Path(tmp) / "m.json"))
        for order in (1, 2, 3):
            self.assertEqual(loaded.ngram_counts(order), self.model.ngram_counts(order))
        self.assertEqual(loaded.vocab_size, self.model.vocab_size)

    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            NGramModel(2).fit([])
        with self.assertRaises(ValueError):
            NGramModel(2).fit([[], []])

    def test_invalid_order(self):
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                NGramModel(bad)


if __name__ == "__main__":
    unittest.main()
