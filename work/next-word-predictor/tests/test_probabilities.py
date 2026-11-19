import math
import unittest

from helpers import toy_model
from wordflow import NextWordPredictor
from wordflow.preprocessing import BOS, UNK


class MLETests(unittest.TestCase):
    def setUp(self):
        self.model = toy_model(3)

    def test_bigram_mle(self):
        self.assertAlmostEqual(self.model.probability("machine", ("love",)), 1 / 3)
        self.assertAlmostEqual(self.model.probability("love", ("i",)), 1.0)
        self.assertEqual(self.model.probability("python", ("i",)), 0.0)

    def test_trigram_mle(self):
        self.assertAlmostEqual(self.model.probability("learning", ("love", "machine")), 1.0)
        self.assertAlmostEqual(self.model.probability("ai", ("i", "love")), 1 / 3)

    def test_unigram_mle(self):
        self.assertAlmostEqual(self.model.probability("love"), 3 / 13)

    def test_bos_is_never_predicted(self):
        self.assertEqual(self.model.probability(BOS, ()), 0.0)

    def test_mle_distributions_normalise_for_every_seen_context(self):
        for order in (1, 2, 3):
            for context in self.model.context_counts(order):
                total = sum(self.model.distribution(context, 0).values())
                self.assertTrue(math.isclose(total, 1.0), (context, total))

    def test_unseen_context_has_empty_mle_distribution(self):
        self.assertEqual(self.model.distribution(("python", "python"), 0), {})


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.predictor = NextWordPredictor(toy_model(3))

    def test_suggest_next_from_specification(self):
        result = self.predictor.suggest_next("I love", top_k=5, smoothing=False)
        words = {s.word: s.probability for s in result.suggestions}
        self.assertEqual(set(words), {"python", "machine", "ai"})
        for p in words.values():
            self.assertAlmostEqual(p, 1 / 3)
        self.assertFalse(result.backed_off)
        self.assertEqual(result.used_context, ["i", "love"])

    def test_ranking_is_by_probability(self):
        result = self.predictor.suggest_next("love machine", smoothing=False, order=2)
        self.assertEqual(result.suggestions[0].word, "learning")
        probs = [s.probability for s in self.predictor.suggest_next("", smoothing=True).suggestions]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_top_k_limits_results(self):
        self.assertEqual(len(self.predictor.suggest_next("I", top_k=2).suggestions), 2)

    def test_empty_input_predicts_sentence_start(self):
        result = self.predictor.suggest_next("", smoothing=False)
        self.assertEqual(result.used_context, [BOS, BOS])
        self.assertEqual(result.suggestions[0].word, "i")

    def test_one_word_input(self):
        result = self.predictor.suggest_next("love", smoothing=False)
        self.assertEqual(result.requested_context, [BOS, "love"])
        self.assertTrue(result.backed_off)       # "<s> love" never occurs in training
        self.assertEqual(result.used_context, ["love"])

    def test_unknown_words_back_off_to_unigram(self):
        result = self.predictor.suggest_next("zebra quantum", smoothing=False)
        self.assertEqual(result.unknown_words, ["quantum", "zebra"])
        self.assertEqual(result.requested_context, [UNK, UNK])
        self.assertEqual(result.used_order, 1)
        self.assertTrue(result.suggestions)
        self.assertIn("<unk>", result.explanation)

    def test_unseen_context_falls_back(self):
        result = self.predictor.suggest_next("python love", smoothing=False)
        self.assertTrue(result.backed_off)
        self.assertEqual(result.used_context, ["love"])

    def test_special_tokens_are_never_suggested(self):
        words = [s.word for s in self.predictor.suggest_next("I love AI", top_k=25).suggestions]
        self.assertNotIn("</s>", words)
        self.assertNotIn("<unk>", words)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.predictor.suggest_next("I", top_k=0)
        with self.assertRaises(ValueError):
            self.predictor.suggest_next("I", smoothing=True, k=0)
        with self.assertRaises(ValueError):
            self.predictor.suggest_next("I", order=4)


if __name__ == "__main__":
    unittest.main()
