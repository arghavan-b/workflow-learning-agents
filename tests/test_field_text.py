"""Option 1: placeholder vs typed value classified by appearance time.

Synthetic state sequences track one field across frames (same bbox/role so
`match_states` links them). The pass should move vanishing pre-interaction text
to `placeholder_text` and keep growing text as `value`.
"""

from __future__ import annotations

import unittest

from demo2skill.video.statediff import classify_field_text, vote_value
from demo2skill.video.statediff.state import ScreenState, UIElement

BOX = (40, 200, 900, 240)


def _field(value=None, text=""):
    return UIElement(id="title", role="textbox", text=text, bbox=BOX, value=value)


class PlaceholderVanishTest(unittest.TestCase):
    def test_placeholder_then_typed(self):
        states = [
            ScreenState(0, 0, elements=[_field(value="Title")]),        # placeholder at rest
            ScreenState(1, 500, elements=[_field(value=None)]),         # focus clears it
            ScreenState(2, 1000, elements=[_field(value="Bug in login flow")]),  # typed
        ]
        classify_field_text(states)

        f0 = states[0].elements[0]
        f2 = states[2].elements[0]
        self.assertEqual(f0.placeholder_text, "Title")   # tagged as placeholder
        self.assertIsNone(f0.value)                        # and removed from value
        self.assertEqual(f2.value, "Bug in login flow")   # typed value kept

    def test_replaced_without_empty_frame(self):
        # Placeholder replaced directly by typed text (no blank frame between).
        states = [
            ScreenState(0, 0, elements=[_field(value="Search")]),
            ScreenState(1, 500, elements=[_field(value="invoices 2026")]),
        ]
        classify_field_text(states)
        self.assertEqual(states[0].elements[0].placeholder_text, "Search")
        self.assertIsNone(states[0].elements[0].value)
        self.assertEqual(states[1].elements[0].value, "invoices 2026")


class PreExistingValueTest(unittest.TestCase):
    def test_persistent_value_is_not_touched(self):
        # A real value present throughout must not be misread as placeholder.
        states = [
            ScreenState(0, 0, elements=[_field(value="Acme Corp")]),
            ScreenState(1, 500, elements=[_field(value="Acme Corp")]),
        ]
        classify_field_text(states)
        for s in states:
            self.assertEqual(s.elements[0].value, "Acme Corp")
            self.assertIsNone(s.elements[0].placeholder_text)

    def test_growing_text_extends_not_placeholder(self):
        # "Bug" -> "Bug in" -> "Bug in login": each extends the last, so the first
        # reading is NOT a placeholder.
        states = [
            ScreenState(0, 0, elements=[_field(value="Bug")]),
            ScreenState(1, 300, elements=[_field(value="Bug in")]),
            ScreenState(2, 600, elements=[_field(value="Bug in login")]),
        ]
        classify_field_text(states)
        self.assertIsNone(states[0].elements[0].placeholder_text)
        self.assertEqual(states[2].elements[0].value, "Bug in login")


class TemporalVoteTest(unittest.TestCase):
    def test_positional_vote_recovers_majority(self):
        # Three noisy reads of the same complete string; vote cleans it.
        self.assertEqual(
            vote_value(["hello world", "hallo world", "hello world"]), "hello world")

    def test_vote_ignores_growth_prefixes(self):
        # Early prefixes are below 80% length, so they don't corrupt the vote.
        self.assertEqual(
            vote_value(["Bug", "Bug in", "Bug in login flow"]), "Bug in login flow")

    def test_vote_empty(self):
        self.assertIsNone(vote_value(["", "  "]))

    def test_voting_cleans_value_across_run(self):
        # Same field, complete but noisy across trailing frames -> voted clean.
        def f(v):
            return UIElement(id="title", role="textbox", bbox=BOX, value=v)
        states = [
            ScreenState(0, 0, elements=[f(None)]),                  # empty at rest
            ScreenState(1, 300, elements=[f("Bug in login flow")]),
            ScreenState(2, 600, elements=[f("Bug 1n login flow")]),  # OCR slip
            ScreenState(3, 900, elements=[f("Bug in login flow")]),
        ]
        classify_field_text(states)
        typed = [s.elements[0].value for s in states if s.elements[0].value]
        self.assertTrue(typed)
        self.assertTrue(all(v == "Bug in login flow" for v in typed))  # all voted clean


class MidTaskAbstainTest(unittest.TestCase):
    def test_single_frame_field_left_untouched(self):
        # No "before" frame -> can't classify -> abstain (don't guess).
        states = [ScreenState(0, 0, elements=[_field(value="Already Filled")])]
        classify_field_text(states)
        self.assertEqual(states[0].elements[0].value, "Already Filled")
        self.assertIsNone(states[0].elements[0].placeholder_text)


class PlaceholderNoLeakTest(unittest.TestCase):
    def test_placeholder_does_not_leak_into_type_event(self):
        # A field that only ever shows its placeholder must not generate a type
        # action (the placeholder must not survive as _value via el.text). Bug 1.
        from demo2skill.video.statediff import StateDiffIDM
        from demo2skill.video.statediff.state import ScreenState as S, UIElement as U

        def ph():
            return U(id="t", role="textbox", text="Title", bbox=BOX,
                     value=None, placeholder_text="Title")
        action = StateDiffIDM().infer(S(0, 0, elements=[ph()]), S(1, 400, elements=[ph()]))
        self.assertNotEqual(action.action_type, "type")

    def test_noisy_placeholder_variant_is_cleared(self):
        # An OCR variant of the placeholder ("Titlе" with a cyrillic e) must still
        # be recognized as the placeholder and cleared (fuzzy match, Bug 2).
        def f(v):
            return UIElement(id="title", role="textbox", bbox=BOX, value=v)
        states = [
            ScreenState(0, 0, elements=[f("Title")]),
            ScreenState(1, 300, elements=[f("TitIe")]),        # noisy placeholder read
            ScreenState(2, 600, elements=[f("Bug in login flow")]),
        ]
        classify_field_text(states)
        self.assertIsNone(states[1].elements[0].value)         # variant cleared
        self.assertEqual(states[2].elements[0].value, "Bug in login flow")


if __name__ == "__main__":
    unittest.main()
