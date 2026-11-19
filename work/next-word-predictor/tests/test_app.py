"""Service and API layer: one trained model as the single source of truth."""

import base64
import math
import unittest

from helpers import TOY_TEXT
from wordflow.server import BadRequest, handle_api
from wordflow.service import TrainingConfig, Workspace


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(model_path=None, meta_path=None)
        self.ws.train_sample(TrainingConfig())

    def call(self, method, route, **params):
        return handle_api(self.ws, method, route, params)

    def test_state(self):
        state = self.call("GET", "state")
        self.assertEqual(state["model"]["max_order"], 3)
        self.assertGreater(state["corpus"]["test_sentences"], 0)

    def test_predict_probabilities_never_exceed_one(self):
        r = self.call("POST", "predict", text="machine", order=2, smoothing=False)
        self.assertEqual(r["suggestions"][0]["word"], "learning")
        total = sum(s["probability"] for s in r["suggestions"]) + sum(r["hidden_mass"].values())
        self.assertLessEqual(total, 1 + 1e-9)

    def test_distribution_sums(self):
        r = self.call("POST", "distribution", context="the model", order=2, k=0.5)
        self.assertTrue(math.isclose(r["checks"]["mle_sum"], 1.0))
        self.assertTrue(math.isclose(r["checks"]["smoothed_sum"], 1.0))

    def test_score_endpoint(self):
        r = self.call("POST", "score", text="I love machine learning", order=2, smoothing=True, k=1)
        self.assertEqual([t["word"] for t in r["tokens"]], ["i", "love", "machine", "learning"])
        expected = math.exp(-sum(math.log(t["probability"]) for t in r["tokens"]) / 4)
        self.assertAlmostEqual(r["perplexity"], expected)

    def test_evaluation_separates_held_out_and_training(self):
        r = self.call("GET", "evaluate", k="1")
        self.assertTrue(r["has_test"])
        held = {x["label"]: x for x in r["held_out"]}
        self.assertFalse(held["Bigram (MLE)"]["finite"])
        self.assertTrue(held["Bigram + add-1"]["finite"])
        self.assertTrue(all(x["finite"] for x in r["training"] if x["k"] == 0))

    def test_generate_is_reproducible(self):
        a = self.call("POST", "generate", seed_text="the", seed=9)
        b = self.call("POST", "generate", seed_text="the", seed=9)
        self.assertEqual(a["text"], b["text"])

    def test_learn_uses_toy_corpus(self):
        r = self.call("POST", "learn", corpus_text=TOY_TEXT, context_word="love", target_word="machine")
        self.assertEqual(r["focus"]["pair_count"], 1)
        self.assertEqual(r["focus"]["context_count"], 3)

    def test_upload_retrains_and_bumps_version(self):
        before = self.ws.state.version
        data = base64.b64encode(b"The cat sat.\nThe dog ran.\nThe cat ran home.").decode()
        r = self.call("POST", "train", source="upload", filename="pets.txt", content_base64=data, max_order=2)
        self.assertEqual(r["corpus"]["name"], "pets.txt")
        self.assertEqual(r["version"], before + 1)
        self.assertEqual(self.call("POST", "predict", text="the", order=2)["version"], before + 1)

    def test_bad_requests(self):
        with self.assertRaises(BadRequest):
            self.call("POST", "predict", text="x", top_k=0)
        with self.assertRaises(BadRequest):
            self.call("POST", "predict", text="x", order=9)
        with self.assertRaises(BadRequest):
            self.call("POST", "predict", text="x", k=-1)
        with self.assertRaises(LookupError):
            self.call("GET", "nope")


if __name__ == "__main__":
    unittest.main()
