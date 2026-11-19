import unittest

from helpers import toy_model
from wordflow.writer import WritingSession, run_writer


def scripted(lines):
    feed = iter(lines)

    def input_fn(_prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError

    return input_fn


class WritingSessionTests(unittest.TestCase):
    def make(self, opening="i love", **kw):
        return WritingSession(toy_model(), opening, k=0.01, seed=1, **kw)

    def test_top_suggestions_are_ranked_by_probability(self):
        options = self.make().suggestions()
        probabilities = [p for _, p in options]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        self.assertIn(options[0][0], {"python", "machine", "ai"})

    def test_accept_and_undo(self):
        session = self.make()
        session.accept("python")
        self.assertEqual(session.text, "i love python")
        self.assertTrue(session.undo())
        self.assertEqual(session.text, "i love")
        self.assertFalse(session.undo())

    def test_append_ignores_blank_input(self):
        session = self.make()
        session.append("   ")
        self.assertEqual(session.text, "i love")

    def test_auto_extends_the_story_and_is_undoable(self):
        session = self.make()
        added = session.auto(3)
        self.assertTrue(session.text.startswith("I love") or session.text.startswith("i love"))
        self.assertTrue(added)
        session.undo()
        self.assertEqual(session.text, "i love")

    def test_creative_suggestions_are_distinct_and_seeded(self):
        a = self.make(creative=True, top_k=3).suggestions()
        b = self.make(creative=True, top_k=3).suggestions()
        self.assertEqual(a, b)
        words = [w for w, _ in a]
        self.assertEqual(len(words), len(set(words)))

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            self.make(top_k=0)
        with self.assertRaises(ValueError):
            self.make(temperature=0)

    def test_save(self):
        import tempfile
        from pathlib import Path

        session = self.make()
        with tempfile.TemporaryDirectory() as folder:
            path = session.save(Path(folder) / "story.txt")
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "i love")


class RunWriterTests(unittest.TestCase):
    def run_script(self, lines, opening="i love"):
        out = []
        session = WritingSession(toy_model(), opening, k=0.01, seed=1)
        run_writer(session, scripted(lines), out.append)
        return session, "\n".join(out)

    def test_pick_type_and_quit(self):
        session, output = self.run_script(["1", "dragons", "q"])
        self.assertTrue(session.text.startswith("i love "))
        self.assertTrue(session.text.endswith("dragons"))
        self.assertIn("Your story:", output)

    def test_bad_number_and_undo_messages(self):
        _, output = self.run_script(["9", "u", "q"])
        self.assertIn("pick a number from 1 to", output)
        self.assertIn("nothing to undo", output)

    def test_failed_save_does_not_end_the_session(self):
        session, output = self.run_script(["s /no/such/folder/story.txt", "dragons", "q"])
        self.assertIn("could not save", output)
        self.assertTrue(session.text.endswith("dragons"))

    def test_end_of_input_ends_cleanly(self):
        session, output = self.run_script([])
        self.assertEqual(session.text, "i love")
        self.assertIn("Your story:", output)

    def test_asks_for_an_opening_when_none_given(self):
        session, _ = self.run_script(["i love", "q"], opening="")
        self.assertEqual(session.text, "i love")

    def test_toggle_mode_and_auto(self):
        session, output = self.run_script(["m", "a 2", "q"])
        self.assertTrue(session.creative)
        self.assertIn("creative suggestions", output)
        self.assertGreater(len(session.text.split()), 2)


if __name__ == "__main__":
    unittest.main()
