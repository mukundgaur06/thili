import math
import random
import unittest

from helpers import toy_model
from wordflow import TextGenerator, apply_temperature
from wordflow.generation import sample
from wordflow.preprocessing import EOS, UNK


class TemperatureTests(unittest.TestCase):
    def test_temperature_one_is_identity(self):
        dist = {"a": 0.5, "b": 0.3, "c": 0.2}
        for word, p in apply_temperature(dist, 1.0).items():
            self.assertAlmostEqual(p, dist[word])

    def test_low_temperature_sharpens_high_flattens(self):
        dist = {"a": 0.6, "b": 0.4}
        self.assertGreater(apply_temperature(dist, 0.5)["a"], 0.6)
        self.assertLess(apply_temperature(dist, 2.0)["a"], 0.6)
        self.assertTrue(math.isclose(sum(apply_temperature(dist, 0.05).values()), 1.0))

    def test_invalid_temperature(self):
        for bad in (0, -1, float("nan"), "1"):
            with self.assertRaises(ValueError):
                apply_temperature({"a": 1.0}, bad)

    def test_sampling_follows_the_distribution(self):
        rng = random.Random(0)
        draws = [sample([("a", 0.75), ("b", 0.25)], rng)[0] for _ in range(4000)]
        self.assertAlmostEqual(draws.count("a") / 4000, 0.75, delta=0.03)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.gen = TextGenerator(toy_model(3))

    def test_same_seed_same_text(self):
        a = self.gen.generate_text("I love", seed=11)
        b = self.gen.generate_text("I love", seed=11)
        self.assertEqual(a.generated_tokens, b.generated_tokens)

    def test_mle_trigram_only_produces_observed_continuations(self):
        result = self.gen.generate_text("I love", seed=3, order=3)
        self.assertIn(result.generated_tokens[0], {"python", "machine", "ai"})
        self.assertEqual(result.stop_reason, "end_of_sentence")
        self.assertEqual(result.generated_tokens[-1], EOS)

    def test_bigram_generation(self):
        result = self.gen.generate_text("", seed=5, order=2)
        self.assertEqual(result.generated_tokens[0], "i")
        self.assertTrue(all(step.order_used <= 2 for step in result.steps))

    def test_max_words_bounds_the_loop(self):
        result = self.gen.generate_text("", max_words=5, seed=1, stop_at_eos=False)
        self.assertEqual(len([t for t in result.generated_tokens if t != EOS]), 5)
        self.assertEqual(result.stop_reason, "max_words")

    def test_unknown_seed_uses_fallback(self):
        result = self.gen.generate_text("zebra", seed=2, order=3)
        self.assertTrue(result.steps[0].backed_off)
        self.assertNotIn(UNK, result.generated_tokens)

    def test_smoothed_sampling_never_emits_unk(self):
        for seed in range(20):
            self.assertNotIn(UNK, self.gen.generate_text("I", seed=seed, k=1.0, max_words=10).generated_tokens)

    def test_invalid_max_words(self):
        with self.assertRaises(ValueError):
            self.gen.generate_text("", max_words=0)


if __name__ == "__main__":
    unittest.main()
