import unittest

from helpers import toy_sentences
from wordflow.preprocessing import (
    BOS, EOS, UNK, Vocabulary, context_tokens, pad_sentence, split_sentences, tokenize,
)


class TokenizationTests(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(tokenize("Hello, World!"), ["hello", "world"])

    def test_keeps_contractions_and_numbers(self):
        self.assertEqual(tokenize("Don't panic, it's 3.14!"), ["don't", "panic", "it's", "3.14"])

    def test_normalises_typographic_apostrophe(self):
        self.assertEqual(tokenize("It’s fine"), ["it's", "fine"])

    def test_empty_input(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("  ...  "), [])

    def test_special_tokens_cannot_be_produced_from_text(self):
        self.assertEqual(tokenize("<s> </s> <unk>"), ["s", "s", "unk"])


class SentenceBoundaryTests(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self):
        self.assertEqual(split_sentences("I run. You walk! Do we?"), ["I run.", "You walk!", "Do we?"])

    def test_every_line_is_a_sentence_in_line_mode(self):
        self.assertEqual(len(split_sentences("one\ntwo\n\nthree")), 3)

    def test_paragraph_mode_joins_wrapped_lines(self):
        text = "This sentence is\nwrapped over lines.\n\nNew paragraph."
        self.assertEqual(split_sentences(text, "paragraph"),
                         ["This sentence is wrapped over lines.", "New paragraph."])

    def test_abbreviation_does_not_end_sentence(self):
        self.assertEqual(split_sentences("Dr. Smith is here. Hi."), ["Dr. Smith is here.", "Hi."])

    def test_invalid_line_mode(self):
        with self.assertRaises(ValueError):
            split_sentences("x", "weird")

    def test_padding(self):
        self.assertEqual(pad_sentence(["a", "b"], 3), [BOS, BOS, "a", "b", EOS])
        self.assertEqual(pad_sentence(["a"], 1), ["a", EOS])

    def test_context_is_only_the_unfinished_sentence(self):
        self.assertEqual(context_tokens("I ran. The cat"), ["the", "cat"])
        self.assertEqual(context_tokens("I ran."), [])
        self.assertEqual(context_tokens(""), [])


class VocabularyTests(unittest.TestCase):
    def test_size_counts_words_plus_unk_and_eos(self):
        vocab = Vocabulary.build(toy_sentences())
        self.assertEqual(len(vocab), 6)            # i love python machine learning ai
        self.assertEqual(vocab.size, 8)            # + <unk> + </s>
        self.assertNotIn(BOS, vocab.prediction_tokens)

    def test_unknown_words_map_to_unk(self):
        vocab = Vocabulary.build(toy_sentences())
        self.assertEqual(vocab.map_tokens(["i", "zebra"]), ["i", UNK])

    def test_min_count_drops_rare_words(self):
        vocab = Vocabulary.build(toy_sentences(), min_count=2)
        self.assertEqual(set(vocab.words), {"i", "love"})

    def test_invalid_min_count(self):
        with self.assertRaises(ValueError):
            Vocabulary.build(toy_sentences(), min_count=0)


if __name__ == "__main__":
    unittest.main()
