import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigpt.tokenizer import EOT, BPETokenizer, _merge  # noqa: E402

TEXT = ("the cat sat on the mat. the cat ate the rat. " * 20) + "Ünïcödé — emoji 🙂 tabs\tand\nlines"


class TestBPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = BPETokenizer().train(TEXT, vocab_size=300)

    def test_merge_replaces_non_overlapping_left_to_right(self):
        self.assertEqual(_merge([1, 1, 1, 2, 1, 1], (1, 1), 9), [9, 1, 2, 9])

    def test_roundtrip_any_text(self):
        for s in [TEXT, "", "unseen words zebra QUIZ 123456", "🙂🙂 ü", "  leading spaces\n\n"]:
            self.assertEqual(self.tok.decode(self.tok.encode(s)), s)

    def test_compresses_frequent_text(self):
        s = "the cat sat on the mat."
        self.assertLess(len(self.tok.encode(s)), len(s.encode("utf-8")) / 2)

    def test_vocab_size_and_special_token(self):
        # Training stops early once no pair occurs twice, so a small text gives < 300.
        self.assertLessEqual(self.tok.vocab_size, 300)
        self.assertEqual(self.tok.eot_id, self.tok.vocab_size - 1)
        big = BPETokenizer().train(TEXT * 3 + " ".join(str(i) for i in range(2000)), 300)
        self.assertEqual(big.vocab_size, 300)
        ids = self.tok.encode(f"the cat{EOT}the rat")
        self.assertEqual(ids.count(self.tok.eot_id), 1)
        self.assertEqual(self.tok.decode(ids), f"the cat{EOT}the rat")
        # encode_ordinary never produces the special id
        self.assertNotIn(self.tok.eot_id, self.tok.encode_ordinary(f"the cat{EOT}"))

    def test_merges_build_on_each_other(self):
        for (a, b), new in self.tok.merges.items():
            self.assertLess(a, new)
            self.assertLess(b, new)
            self.assertEqual(self.tok.vocab[new], self.tok.vocab[a] + self.tok.vocab[b])

    def test_first_merge_is_most_frequent_pair(self):
        # per repeat: "at" occurs 6x (cat cat sat mat ate rat), " t" only 4x (the x4)
        first_pair = next(iter(self.tok.merges))
        self.assertEqual(self.tok.vocab[first_pair[0]] + self.tok.vocab[first_pair[1]], b"at")

    def test_deterministic_training(self):
        other = BPETokenizer().train(TEXT, vocab_size=300)
        self.assertEqual(other.merges, self.tok.merges)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tok.json")
            self.tok.save(p)
            loaded = BPETokenizer.load(p)
        self.assertEqual(loaded.vocab, self.tok.vocab)
        self.assertEqual(loaded.encode(TEXT), self.tok.encode(TEXT))

    def test_invalid_vocab_size(self):
        with self.assertRaises(ValueError):
            BPETokenizer().train("abc", vocab_size=100)

    def test_unknown_id(self):
        with self.assertRaises(ValueError):
            self.tok.decode([10_000])


if __name__ == "__main__":
    unittest.main()
