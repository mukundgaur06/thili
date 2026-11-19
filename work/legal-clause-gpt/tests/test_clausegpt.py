import os
import sys
import tempfile
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # tokenizer comes from the local cache
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clausegpt  # noqa: E402,F401  (must come before transformers)
import torch  # noqa: E402
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel  # noqa: E402

from clausegpt.data import Collator, encode, prompt_ids  # noqa: E402
from clausegpt.labels import SEP, label_name, label_token, new_tokens, prompt, training_text  # noqa: E402
from clausegpt.loss import answer_loss  # noqa: E402
from clausegpt.model import BASE, extend_tokenizer, resize_and_init  # noqa: E402
from clausegpt.tagger import ClauseTagger  # noqa: E402

NAMES = ["Governing Laws", "Counterparts", "Notices", "Severability"]
CLAUSES = {
    "Governing Laws": "This Agreement shall be governed by the laws of the State of New York.",
    "Counterparts": "This Agreement may be executed in counterparts, each of which is an original.",
    "Notices": "All notices shall be in writing and delivered by hand or certified mail.",
    "Severability": "If any provision is held invalid, the remaining provisions remain in effect.",
}


def tiny(tok):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_positions=128, n_embd=32, n_layer=2, n_head=2,
                                       resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)).eval()
    resize_and_init(model, tok, NAMES)
    model.config.pad_token_id = tok.pad_token_id
    return model


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(BASE)
        cls.added = extend_tokenizer(cls.tok, NAMES)


class TestLabels(Base):
    def test_round_trip(self):
        self.assertEqual(label_token("Governing Laws"), "<Governing Laws>")
        self.assertEqual(label_name("<Governing Laws>"), "Governing Laws")
        with self.assertRaises(ValueError):
            label_name("Governing Laws")
        self.assertEqual(new_tokens(NAMES)[0], SEP)
        self.assertEqual(training_text(" x ", "Notices"), "x<Label><Notices><|endoftext|>")
        self.assertTrue(prompt("x").endswith(SEP))

    def test_tokenizer_learns_one_id_per_label(self):
        plain = AutoTokenizer.from_pretrained(BASE)
        self.assertGreater(len(plain("<Governing Laws>").input_ids), 1)    # split by BPE before
        self.assertEqual(self.added, len(NAMES) + 1)
        for n in NAMES:
            ids = self.tok(label_token(n)).input_ids
            self.assertEqual(len(ids), 1)
            self.assertGreaterEqual(ids[0], 50257)                       # a brand-new id
        self.assertEqual(len(self.tok(SEP).input_ids), 1)
        self.assertEqual(self.tok.pad_token, self.tok.eos_token)

    def test_full_example_tokenizes_to_expected_tail(self):
        ids = self.tok(training_text("Some clause.", "Notices")).input_ids
        self.assertEqual(self.tok.convert_ids_to_tokens(ids[-3:]), [SEP, "<Notices>", "<|endoftext|>"])


class TestResize(Base):
    def test_rows_initialised_from_label_words_and_tied(self):
        model = tiny(self.tok)
        emb = model.get_input_embeddings().weight
        self.assertEqual(emb.shape[0], len(self.tok))
        plain = AutoTokenizer.from_pretrained(BASE)
        tid = self.tok.convert_tokens_to_ids("<Governing Laws>")
        expect = emb[plain(" Governing Laws").input_ids].mean(0)
        self.assertTrue(torch.allclose(emb[tid], expect))
        self.assertIs(model.lm_head.weight, model.transformer.wte.weight)     # output tied to input


class TestEncoding(Base):
    def test_only_answer_is_graded_and_truncation_keeps_it(self):
        long = "word " * 500
        ex = encode(self.tok, long, "Notices", max_clause_tokens=16)
        self.assertEqual(len(ex["input_ids"]), 16 + 3)
        graded = [i for i, l in enumerate(ex["labels"]) if l != -100]
        self.assertEqual(graded, [17, 18])
        self.assertEqual(self.tok.convert_ids_to_tokens([ex["labels"][17], ex["labels"][18]]),
                         ["<Notices>", "<|endoftext|>"])
        self.assertEqual(ex["input_ids"][:17], prompt_ids(self.tok, long, 16))

    def test_loss_on_prompt(self):
        ex = encode(self.tok, "A clause.", "Notices", loss_on_prompt=True)
        self.assertEqual(ex["labels"], ex["input_ids"])

    def test_collator_right_pads(self):
        a = encode(self.tok, "Short.", "Notices")
        b = encode(self.tok, "A somewhat longer clause about notices.", "Notices")
        batch = Collator(self.tok.pad_token_id)([a, b])
        self.assertEqual(batch["input_ids"].shape[1], len(b["input_ids"]))
        k = len(a["input_ids"])
        self.assertTrue((batch["labels"][0, k:] == -100).all())
        self.assertTrue((batch["attention_mask"][0, k:] == 0).all())
        self.assertEqual(int(batch["attention_mask"][1].sum()), len(b["input_ids"]))


class TestLoss(Base):
    def test_answer_loss_equals_stock_loss_and_gradients(self):
        model = tiny(self.tok)
        batch = Collator(self.tok.pad_token_id)([encode(self.tok, c, n) for n, c in CLAUSES.items()])
        ref = model(**batch).loss
        ours = answer_loss(model, **batch)
        self.assertAlmostEqual(ref.item(), ours.item(), places=5)
        g_ref = torch.autograd.grad(ref, model.transformer.h[0].attn.c_attn.weight)[0]
        g_ours = torch.autograd.grad(answer_loss(model, **batch), model.transformer.h[0].attn.c_attn.weight)[0]
        self.assertTrue(torch.allclose(g_ref, g_ours, atol=1e-6))


class TestTagger(Base):
    def test_free_generation_matches_model_generate_and_constrained_is_valid(self):
        tagger = ClauseTagger(tiny(self.tok), self.tok, NAMES, max_clause_tokens=32)
        preds = tagger.predict(list(CLAUSES.values()), k=3)
        for c, p in zip(CLAUSES.values(), preds):
            self.assertEqual(tagger.generate(c, max_new_tokens=1), p.generated)
            self.assertIn(p.label, NAMES)
            self.assertAlmostEqual(sum(pr for _, pr in tagger.predict([c], k=len(NAMES))[0].top), 1.0, places=5)

    def test_batched_equals_one_by_one(self):
        tagger = ClauseTagger(tiny(self.tok), self.tok, NAMES)
        texts = list(CLAUSES.values())
        batch = tagger.next_token_logits(texts)
        single = torch.cat([tagger.next_token_logits([t]) for t in texts])
        self.assertTrue(torch.allclose(batch, single, atol=1e-4))

    def test_save_and_reload(self):
        tagger = ClauseTagger(tiny(self.tok), self.tok, NAMES, max_clause_tokens=40)
        texts = list(CLAUSES.values())
        before = tagger.next_token_logits(texts)
        with tempfile.TemporaryDirectory() as d:
            tagger.save(d)
            again = ClauseTagger.load(d)
        self.assertEqual(again.names, NAMES)
        self.assertEqual(again.max_clause_tokens, 40)
        self.assertTrue(torch.allclose(before, again.next_token_logits(texts), atol=1e-5))
        self.assertIs(again.model.lm_head.weight, again.model.transformer.wte.weight)

    def test_learns_to_tag(self):
        """A tiny model fine-tuned for 150 steps tags its training clauses."""
        model = tiny(self.tok)
        batch = Collator(self.tok.pad_token_id)([encode(self.tok, c, n) for n, c in CLAUSES.items()])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        model.train()
        for _ in range(150):
            loss = answer_loss(model, **batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
        tagger = ClauseTagger(model, self.tok, NAMES)
        preds = tagger.predict(list(CLAUSES.values()))
        self.assertEqual([p.label for p in preds], list(CLAUSES))
        self.assertEqual([p.generated for p in preds], [label_token(n) for n in CLAUSES])
        self.assertEqual(tagger.tag(CLAUSES["Notices"]), "<Notices>")


if __name__ == "__main__":
    unittest.main()
