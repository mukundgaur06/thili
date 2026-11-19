import math
import unittest

from helpers import toy_model
from wordflow import NextWordPredictor
from wordflow.smoothing import add_k_probability, mle_probability, validate_k


class SmoothingFormulaTests(unittest.TestCase):
    def test_laplace_formula(self):
        # (C(ctx, w) + k) / (C(ctx) + k V) with C(ctx, w) = 1, C(ctx) = 3, k = 1, V = 8
        self.assertAlmostEqual(add_k_probability(1, 3, 1, 8), 2 / 11)

    def test_add_k_formula(self):
        self.assertAlmostEqual(add_k_probability(0, 3, 0.5, 8), 0.5 / 7)

    def test_unseen_context_is_uniform(self):
        self.assertAlmostEqual(add_k_probability(0, 0, 1, 8), 1 / 8)

    def test_mle_zero_context(self):
        self.assertEqual(mle_probability(0, 0), 0.0)

    def test_invalid_k(self):
        for bad in (-1, float("nan"), float("inf"), "1", None, True):
            with self.assertRaises(ValueError):
                validate_k(bad)
        with self.assertRaises(ValueError):
            add_k_probability(1, 3, 0, 8)


class SmoothedModelTests(unittest.TestCase):
    def setUp(self):
        self.model = toy_model(3)
        self.V = self.model.vocab_size

    def test_vocab_size(self):
        self.assertEqual(self.V, 8)

    def test_smoothed_probability(self):
        self.assertAlmostEqual(self.model.probability("machine", ("love",), k=1), 2 / 11)
        self.assertAlmostEqual(self.model.probability("python", ("i",), k=1), 1 / 11)

    def test_smoothed_distributions_sum_to_one(self):
        for k in (0.01, 0.5, 1, 3):
            for order in (1, 2, 3):
                contexts = list(self.model.context_counts(order)) + [("zzz",) * (order - 1)]
                for context in contexts:
                    total = sum(self.model.distribution(context, k).values())
                    self.assertTrue(math.isclose(total, 1.0), (k, context, total))

    def test_smoothing_gives_unseen_pairs_mass(self):
        self.assertEqual(self.model.probability("python", ("i",), 0), 0)
        self.assertGreater(self.model.probability("python", ("i",), 1), 0)

    def test_smoothing_discounts_seen_pairs(self):
        self.assertLess(self.model.probability("love", ("i",), 1), self.model.probability("love", ("i",), 0))

    def test_larger_k_is_flatter(self):
        small = self.model.probability("love", ("i",), 0.1)
        large = self.model.probability("love", ("i",), 10)
        self.assertGreater(small, large)
        self.assertGreater(large, 1 / self.V)

    def test_predictor_with_smoothing_reports_unseen_counts(self):
        result = NextWordPredictor(self.model).suggest_next("I love", top_k=6, smoothing=True, k=1)
        counts = {s.word: s.count for s in result.suggestions}
        self.assertEqual(counts["machine"], 1)
        self.assertIn(0, counts.values())


if __name__ == "__main__":
    unittest.main()
