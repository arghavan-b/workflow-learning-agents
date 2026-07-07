"""Parse stabilization: drop one-frame flicker, but never a coarse parse."""

from __future__ import annotations

import unittest

from demo2skill.video.statediff.stability import stabilize_states
from demo2skill.video.statediff.state import ScreenState, UIElement


def _el(eid, x, role="button"):
    return UIElement(id=eid, role=role, text=eid, bbox=(x, 0, x + 30, 20))


class StabilityTest(unittest.TestCase):
    def test_drops_single_frame_flicker(self):
        states = [
            ScreenState(0, 0, elements=[_el("a", 0), _el("b", 100)]),
            ScreenState(1, 100, elements=[_el("a", 0), _el("b", 100), _el("flick", 300)]),
            ScreenState(2, 200, elements=[_el("a", 0), _el("b", 100)]),
            ScreenState(3, 300, elements=[_el("a", 0), _el("b", 100)]),
        ]
        stabilize_states(states)
        ids = [{e.id for e in s.elements} for s in states]
        self.assertNotIn("flick", ids[1])          # matched to no neighbor -> dropped
        self.assertEqual(ids[0], {"a", "b"})        # persistent elements survive

    def test_coarse_parse_left_untouched(self):
        # Every element unique per frame (like a hand-authored one-state-per-page
        # fixture); filtering would drop everything, so it must abstain.
        states = [
            ScreenState(0, 0, elements=[UIElement(id="a", role="link", text="Issues", bbox=(0, 0, 50, 20))]),
            ScreenState(1, 100, elements=[UIElement(id="b", role="button", text="New", bbox=(500, 300, 600, 340))]),
            ScreenState(2, 200, elements=[UIElement(id="c", role="textbox", text="", bbox=(100, 400, 700, 440))]),
        ]
        stabilize_states(states)
        self.assertEqual([len(s.elements) for s in states], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
