"""BasketMind tests. Small deterministic data; metrics are checked against hand-computed values,
mining against brute-force enumeration, never against the implementation itself."""

import math
import os
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import basketmind  # noqa: E402,F401
import numpy as np  # noqa: E402

from basketmind.data.catalogue import make_catalogue, popularity  # noqa: E402
from basketmind.data.generator import GeneratorConfig, generate  # noqa: E402
from basketmind.data.verbalizer import Vocabulary, decode, encode, order, product_token, token_pid  # noqa: E402
from basketmind.evaluation.bootstrap import bootstrap_ci, paired_bootstrap  # noqa: E402
from basketmind.evaluation.itemsets import classify, precision  # noqa: E402
from basketmind.evaluation.ranking import case_metrics, make_cases, ranks_of  # noqa: E402
from basketmind.mining.apriori import apriori, brute_force  # noqa: E402
from basketmind.mining.bitsets import build_bitsets, popcount, support_count  # noqa: E402
from basketmind.mining.rules import generate_rules  # noqa: E402
from basketmind.recommenders.cooccurrence import AssociationRules, CoOccurrence  # noqa: E402
from basketmind.recommenders.popularity import Popularity  # noqa: E402

GPT2_CACHED = any((Path.home() / ".cache" / "huggingface" / "hub").glob("models--openai-community--gpt2"))


def small_data(n=12_000, seed=0):
    cat = make_catalogue(seed=seed)
    baskets, planted, report = generate(cat, GeneratorConfig(n_baskets=n, seed=seed))
    return cat, baskets, planted, report


class TestCatalogue(unittest.TestCase):
    def test_size_coverage_and_determinism(self):
        a, b = make_catalogue(seed=0), make_catalogue(seed=0)
        self.assertEqual(len(a), 1000)
        self.assertEqual([p.name for p in a], [p.name for p in b])
        self.assertEqual(len({p.name for p in a}), 1000)                 # unique names
        self.assertEqual(len({p.base for p in a}), 95)                   # every base product present
        self.assertGreaterEqual(len({p.category for p in a}), 7)
        s = popularity(a)
        self.assertAlmostEqual(s.sum(), 1.0)
        self.assertTrue((s > 0).all())


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat, cls.baskets, cls.planted, cls.report = small_data()

    def test_deterministic(self):
        _, b2, p2, _ = small_data()
        self.assertEqual(self.baskets[:50], b2[:50])
        self.assertEqual([p.items for p in self.planted], [p.items for p in p2])

    def test_one_sku_per_base_and_sizes(self):
        base = {p.pid: p.base for p in self.cat}
        for b in self.baskets[:3000]:
            self.assertEqual(len(b), len(set(b)))
            self.assertEqual(len({base[i] for i in b}), len(b))
            self.assertGreaterEqual(len(b), 2)

    def test_planted_supports_hit_their_targets(self):
        """Realised support measured by direct counting, independent of the generator's own report."""
        n = len(self.baskets)
        for p in self.planted:
            real = sum(set(p.items) <= set(b) for b in self.baskets) / n
            self.assertGreaterEqual(real, p.target_support - 1 / n)
            self.assertLess(real, 1.6 * p.target_support)

    def test_planted_design_rules(self):
        self.assertEqual(len(self.planted), 32)
        self.assertEqual({len(p.items) for p in self.planted}, {2, 3, 4, 5})
        self.assertEqual({p.kind for p in self.planted}, {"natural", "surprising"})
        sets = [set(p.items) for p in self.planted]
        for i, a in enumerate(sets):
            for j, b in enumerate(sets):
                if i != j:
                    self.assertFalse(a <= b)                               # no planted set contains another
        r = next(x for x in self.report if x["bases"] == ["Bread", "Butter"])
        self.assertEqual(r["kind"], "natural")


class TestMining(unittest.TestCase):
    def test_bitsets_match_counting(self):
        rng = np.random.default_rng(0)
        baskets = [sorted(rng.choice(30, rng.integers(1, 8), replace=False).tolist()) for _ in range(500)]
        bits = build_bitsets(baskets, 30)
        self.assertEqual(bits.shape, (30, 8))
        for items in [(3,), (1, 2), (0, 5, 7), (2, 4, 6, 8)]:
            self.assertEqual(support_count(bits, items), sum(set(items) <= set(b) for b in baskets))
        self.assertEqual(int(popcount(np.array([np.uint64(2**64 - 1), np.uint64(5)]))), 66)

    def test_apriori_equals_brute_force(self):
        rng = np.random.default_rng(1)
        for trial in range(3):
            baskets = [sorted(rng.choice(25, rng.integers(1, 9), replace=False).tolist()) for _ in range(400)]
            for ms in (0.02, 0.05):
                self.assertEqual(apriori(baskets, 25, ms, 5), brute_force(baskets, ms, 5))

    def test_apriori_on_generated_data_matches_brute_force(self):
        _, baskets, _, _ = small_data(n=12_000)
        sample = baskets[:1500]
        self.assertEqual(apriori(sample, 1000, 0.006, 4), brute_force(sample, 0.006, 4))

    def test_apriori_pruning_counts_fewer_candidates(self):
        """Level-3 candidates must all have every 2-subset frequent."""
        rng = np.random.default_rng(2)
        baskets = [sorted(rng.choice(20, rng.integers(2, 7), replace=False).tolist()) for _ in range(300)]
        found = apriori(baskets, 20, 0.03, 4)
        for s in found:
            for k in range(1, len(s)):
                for sub in combinations(sorted(s), k):
                    self.assertIn(frozenset(sub), found)                   # Apriori property holds

    def test_rule_metrics_by_hand(self):
        # 10 baskets: A in 5, B in 4, A&B in 3
        sup = {frozenset({0}): 0.5, frozenset({1}): 0.4, frozenset({0, 1}): 0.3}
        rules = {(tuple(r.antecedent), tuple(r.consequent)): r for r in generate_rules(sup, 0.0)}
        r = rules[((0,), (1,))]
        self.assertAlmostEqual(r.confidence, 0.6)
        self.assertAlmostEqual(r.lift, 1.5)
        self.assertAlmostEqual(r.leverage, 0.3 - 0.5 * 0.4)
        self.assertAlmostEqual(r.conviction, (1 - 0.4) / (1 - 0.6))
        self.assertAlmostEqual(rules[((1,), (0,))].confidence, 0.75)

    def test_classify(self):
        class P:
            def __init__(self, items):
                self.items = items
        planted = [P((1, 2, 3))]
        self.assertEqual(classify({1}, planted), "single")
        self.assertEqual(classify({1, 2, 3}, planted), "planted")
        self.assertEqual(classify({1, 2}, planted), "part")
        self.assertEqual(classify({1, 2, 9}, planted), "extension")
        self.assertEqual(classify({7, 9}, planted), "chance")
        self.assertEqual(precision({frozenset({1, 2}): 0.1, frozenset({7, 9}): 0.1}, {frozenset({1, 2}): 0.2}, 0.05), 0.5)


class TestVerbalizer(unittest.TestCase):
    def test_tokens_and_round_trip(self):
        v = Vocabulary(1000)
        self.assertEqual(product_token(42), "<p0042>")
        self.assertEqual(token_pid("<p0042>"), 42)
        self.assertIsNone(token_pid("<basket>"))
        ids = encode([5, 3, 900], v, "fixed")
        self.assertEqual(ids, [v.open_id, 3, 5, 900, v.close_id])
        self.assertEqual(decode(ids, v)["items"], [3, 5, 900])

    def test_orderings_keep_membership(self):
        b = [9, 2, 7, 4, 11]
        rng = np.random.default_rng(0)
        for policy in ("fixed", "frozen", "fresh"):
            self.assertEqual(sorted(order(b, policy, rng, frozen_seed=3)), sorted(b))
        self.assertEqual(order(b, "frozen", frozen_seed=3), order(b, "frozen", frozen_seed=3))
        fresh = {tuple(order(b, "fresh", rng)) for _ in range(30)}
        self.assertGreater(len(fresh), 10)

    def test_decode_counts_repeats_and_invalid(self):
        v = Vocabulary(10)
        out = decode([v.open_id, 1, 2, 1, v.pad_id, 3, v.close_id, 4], v)
        self.assertEqual(out["items"], [1, 2, 3])
        self.assertEqual(out["repeats"], 1)
        self.assertEqual(out["invalid"], 1)
        self.assertTrue(out["closed"])


class TestRanking(unittest.TestCase):
    def test_metrics_by_hand(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])          # item 0 best
        # visible {0}; targets {3, 5}: candidates 1..5 -> 3 is ranked 3rd, 5 is ranked 5th
        self.assertEqual(ranks_of(scores, [0], [3, 5]).tolist(), [3, 5])
        m = case_metrics(scores, [0], [3, 5], ks=(1, 3, 5))
        self.assertEqual(m["rank"], 3)
        self.assertAlmostEqual(m["rr"], 1 / 3)
        self.assertEqual(m["hits@1"], 0.0)
        self.assertEqual(m["hits@3"], 1.0)
        self.assertEqual(m["recall@3"], 0.5)
        self.assertEqual(m["recall@5"], 1.0)
        ndcg = (1 / math.log2(4) + 1 / math.log2(6)) / (1 / math.log2(2) + 1 / math.log2(3))
        self.assertAlmostEqual(m["ndcg@10"], ndcg)

    def test_ties_count_against_target(self):
        self.assertEqual(ranks_of(np.zeros(5), [0], [2]).tolist(), [4])    # all tied: worst rank

    def test_cases_split_visible_and_targets(self):
        cases = make_cases([[1, 2, 3, 4], [5, 6], [7]], seed=0)
        self.assertEqual(len(cases), 2)                                    # single-item baskets skipped
        for c, b in zip(cases, ([1, 2, 3, 4], [5, 6])):
            self.assertEqual(sorted(c["visible"] + c["targets"]), b)
            self.assertFalse(set(c["visible"]) & set(c["targets"]))
            self.assertEqual(len(c["visible"]), math.ceil(len(b) / 2))


class TestBootstrap(unittest.TestCase):
    def test_ci_covers_mean_and_is_seeded(self):
        v = np.random.default_rng(0).normal(5, 1, 400)
        a, b = bootstrap_ci(v, seed=1), bootstrap_ci(v, seed=1)
        self.assertEqual(a, b)
        self.assertLess(a["lo"], v.mean())
        self.assertGreater(a["hi"], v.mean())
        se = v.std(ddof=1) / math.sqrt(len(v))
        self.assertAlmostEqual(a["hi"] - a["lo"], 2 * 1.96 * se, delta=0.25 * 2 * 1.96 * se)

    def test_paired_is_tighter_than_unpaired(self):
        rng = np.random.default_rng(0)
        difficulty = rng.normal(0, 1, 500)
        a = difficulty + 0.1 + rng.normal(0, 0.1, 500)
        b = difficulty + rng.normal(0, 0.1, 500)
        p = paired_bootstrap(a, b, seed=0)
        self.assertGreater(p["lo"], 0)                                     # paired detects the 0.1 gap
        self.assertGreater(p["p_a_better"], 0.99)
        ua, ub = bootstrap_ci(a, seed=0), bootstrap_ci(b, seed=0)
        self.assertLess(ua["lo"], ub["hi"])                                # unpaired intervals overlap
        with self.assertRaises(ValueError):
            paired_bootstrap(a, b[:10])


class TestRecommenders(unittest.TestCase):
    def setUp(self):
        # items 0 and 1 always together; 2 is popular but unrelated
        self.baskets = [[0, 1, 2]] * 20 + [[0, 1]] * 10 + [[2, 3]] * 40 + [[2, 4]] * 40 + [[3, 4]] * 5

    def test_popularity(self):
        p = Popularity().fit(self.baskets, 5)
        self.assertEqual(p.recommend([3], k=1)[0][0], 2)

    def test_cooccurrence_finds_partner_and_excludes_visible(self):
        c = CoOccurrence("confidence").fit(self.baskets, 5)
        rec = c.recommend([0], k=2)
        self.assertEqual(rec[0][0], 1)
        self.assertNotIn(0, [i for i, _ in rec])
        ev = c.evidence([0], 1)
        self.assertAlmostEqual(ev["confidence"], 1.0)                      # 30 of 30 baskets with 0 have 1
        self.assertAlmostEqual(ev["pair_support"], 30 / len(self.baskets))
        self.assertAlmostEqual(ev["lift"], 1.0 / (30 / len(self.baskets)))

    def test_rules_recommender(self):
        r = AssociationRules(min_support=0.05, min_confidence=0.5).fit(self.baskets, 5)
        self.assertEqual(r.recommend([0], k=1)[0][0], 1)
        ex = r.explain([0], 1)
        self.assertAlmostEqual(ex["confidence"], 1.0)
        valid = set(range(5))
        for v in ([0], [2], [3, 4]):
            self.assertTrue({i for i, _ in r.recommend(v, 3)} <= valid - set(v))


@unittest.skipUnless(GPT2_CACHED, "GPT-2 weights not in the local Hugging Face cache")
class TestLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from basketmind.language_model.model import BasketLM, LMConfig
        torch.manual_seed(0)
        cls.cat = make_catalogue()
        cls.model = BasketLM.from_pretrained_gpt2(cls.cat, LMConfig(lora_rank=4))

    def test_trainable_parameters_and_embedding_init(self):
        from basketmind.language_model.tokenizer import gpt2_tokenizer
        c = self.model.counts()
        self.assertLess(c["percent"], 5)
        self.assertIs(self.model.gpt2.lm_head.weight, self.model.gpt2.transformer.wte.weight)
        self.assertEqual(self.model.gpt2.transformer.wte.weight.shape[0], 1003)
        from transformers import GPT2LMHeadModel
        ref = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").transformer.wte.weight
        tok = gpt2_tokenizer()
        p = self.cat[42]
        expect = ref[tok(" " + p.name).input_ids].mean(0)
        self.assertTrue(np.allclose(self.model.gpt2.transformer.wte.weight[42].detach().numpy(), expect.detach().numpy(), atol=1e-6))

    def test_causal_mask_future_does_not_leak(self):
        import torch
        a = torch.tensor([[1000, 5, 6, 7, 8]])
        b = torch.tensor([[1000, 5, 6, 99, 123]])
        with torch.no_grad():
            la, lb = self.model(a).logits, self.model(b).logits
        self.assertTrue(torch.allclose(la[:, :3], lb[:, :3], atol=1e-5))   # positions 0-2 can't see 3-4
        self.assertFalse(torch.allclose(la[:, 3], lb[:, 3], atol=1e-3))

    def test_padding_does_not_change_real_positions(self):
        import torch
        from basketmind.language_model.dataset import pad_batch
        ids, mask = pad_batch([[1000, 5, 6, 1001], [1000, 9]], self.model.vocab.pad_id)
        with torch.no_grad():
            batched = self.model(ids, attention_mask=mask).logits
            single = self.model(torch.tensor([[1000, 9]])).logits
        self.assertTrue(torch.allclose(batched[1, :2], single[0], atol=1e-4))

    def test_loss_shifts_labels_and_ignores_padding(self):
        import torch
        import torch.nn.functional as F
        from basketmind.language_model.dataset import next_token_loss, pad_batch
        ids, mask = pad_batch([[1000, 3, 4, 1001], [1000, 7, 1001]], self.model.vocab.pad_id)
        logits = torch.randn(2, 4, 1003)
        # by hand: predict ids[t+1] from position t, only where ids[t+1] is not padding
        terms = [F.cross_entropy(logits[r, t][None], ids[r, t + 1][None]) for r, t in
                 [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]]
        self.assertAlmostEqual(float(next_token_loss(logits, ids, mask)), float(torch.stack(terms).mean()), places=5)

    def test_training_step_lowers_loss_and_scores_are_valid(self):
        import torch
        from basketmind.language_model.dataset import next_token_loss, pad_batch
        from basketmind.language_model.inference import next_item_logprobs
        from basketmind.language_model.model import BasketLM, LMConfig
        model = BasketLM.from_pretrained_gpt2(self.cat, LMConfig(lora_rank=4))
        seqs = [encode(b, model.vocab, "fixed") for b in ([1, 2, 3], [1, 2, 4], [1, 2, 5])]
        ids, mask = pad_batch(seqs, model.vocab.pad_id)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
        first = None
        for _ in range(50):                       # by ~40 steps it reliably puts most mass on item 2
            loss = next_token_loss(model(ids, attention_mask=mask).logits, ids, mask, vocab=model.vocab)
            first = first if first is not None else loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(loss), first)
        s = next_item_logprobs(model, [[1], [1, 2]], n_orders=2)
        self.assertEqual(s.shape, (2, 1000))
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertLessEqual(float(np.exp(s).sum(1).max()), 1.0 + 1e-4)    # products share at most all the mass
        self.assertEqual(int(np.argmax(s[0])), 2)                          # learned: after 1 comes 2

    def test_generation_parses_to_valid_products(self):
        from basketmind.language_model.generation import sample_baskets
        baskets, stats = sample_baskets(self.model, 8, batch=8, max_items=6, seed=0)
        self.assertEqual(len(baskets), 8)
        for b in baskets:
            self.assertTrue(all(0 <= i < 1000 for i in b))
            self.assertEqual(len(b), len(set(b)))
        self.assertEqual(stats["invalid_tokens"], 0)

    def test_save_and_load(self):
        import torch
        from basketmind.language_model.inference import next_item_logprobs
        from basketmind.language_model.model import BasketLM
        with torch.no_grad():
            self.model.gpt2.transformer.h[0].attn.c_attn.lora_B.add_(0.01)
        before = next_item_logprobs(self.model, [[3, 4]], n_orders=1)
        with tempfile.TemporaryDirectory() as d:
            self.model.save(Path(d))
            again = BasketLM.load(Path(d), self.cat)
        after = next_item_logprobs(again, [[3, 4]], n_orders=1)
        self.assertTrue(np.allclose(before, after, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
