"""State-diff inverse dynamics: parsed screen states -> actions -> skill + graph.

Demonstrates the recommended GUI formulation: visual change proposes candidate
moments, and element-level before/after state plus cursor evidence determine the
action. Inputs are *parsed* screen states (what a screen parser emits); the
pixels->state step is out of scope here and pluggable.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import List

from demo2skill.induction.workflow_generator import induce_workflow
from demo2skill.trace.normalize import normalize_trace
from demo2skill.video.statediff import (
    ScreenState,
    StateDiffIDM,
    StateTrajectoryBuilder,
    UIElement,
    UIStateGraph,
)
from demo2skill.video.statediff.cursor import CursorTrack
from demo2skill.workflow.validator import validate_skill

EXAMPLE = Path(__file__).resolve().parents[1] / "demo2skill" / "examples" / "github_issue"


def _load() -> tuple[List[ScreenState], CursorTrack]:
    data = json.loads((EXAMPLE / "screen_states.json").read_text())
    states = [
        ScreenState(
            index=s["index"], ms=s["ms"], url=s.get("url"), title=s.get("title"),
            elements=[
                UIElement(
                    id=e["id"], role=e["role"], text=e.get("text", ""),
                    bbox=tuple(e.get("bbox", (0, 0, 0, 0))), value=e.get("value"),
                    label=e.get("label"), focused=e.get("focused", False),
                    checked=e.get("checked"), selected=e.get("selected"),
                )
                for e in s["elements"]
            ],
        )
        for s in data["states"]
    ]
    return states, CursorTrack.from_records(data["cursor"])


class StateDiffActionTest(unittest.TestCase):
    def test_actions_recovered_from_state_diff(self):
        states, cursor = _load()
        traj = StateTrajectoryBuilder(cursor).build(states, video_id="issue_states")

        kinds = [a.action_type for a in traj.actions]
        # click Issues -> click New issue -> type Title -> type Body
        self.assertEqual(kinds, ["click", "click", "type", "type"])

        types = [a for a in traj.actions if a.action_type == "type"]
        self.assertEqual(types[0].target_label, "Title")
        self.assertEqual(types[0].text, "Bug in login flow")
        # the page-transition clicks were attributed to the controls under the cursor
        clicks = [a for a in traj.actions if a.action_type == "click"]
        self.assertEqual(clicks[0].target_text, "Issues")
        self.assertEqual(clicks[1].target_text, "New issue")

    def test_typing_beats_navigation_when_value_changes(self):
        # A pure value change with the same URL must be read as type, not navigate.
        states, cursor = _load()
        idm = StateDiffIDM(cursor)
        action = idm.infer(states[2], states[3])
        self.assertEqual(action.action_type, "type")
        self.assertEqual(action.args["text"], "Bug in login flow")
        self.assertEqual(action.effect["field_value"], "Bug in login flow")


class StateToSkillTest(unittest.TestCase):
    def test_full_chain_states_to_valid_skill(self):
        states, cursor = _load()
        traj = StateTrajectoryBuilder(cursor).build(states, video_id="issue_states")

        skill = induce_workflow(normalize_trace(traj.to_raw_trace()))
        errors = [i for i in validate_skill(skill) if i.severity == "error"]
        self.assertEqual(errors, [])

        self.assertGreaterEqual(len(skill.inputs), 2)
        fills = [s for s in skill.steps if s.action == "fill_field"]
        self.assertGreaterEqual(len(fills), 2)
        for step in fills:
            self.assertTrue(step.value and step.value.startswith("${"))
        actions = [s.action for s in skill.steps]
        self.assertTrue("request_user_confirmation" in actions or "stop" in actions)


class StateGraphTest(unittest.TestCase):
    def test_graph_nodes_and_edges(self):
        states, cursor = _load()
        graph = UIStateGraph.build(states, cursor)
        # five distinct states -> five nodes, four action-labeled edges
        self.assertEqual(len(graph.nodes), 5)
        self.assertEqual(len(graph.edges), 4)
        self.assertEqual([e.action_type for e in graph.edges],
                         ["click", "click", "type", "type"])


class ToggleTest(unittest.TestCase):
    def test_checkbox_toggle_is_a_click(self):
        before = ScreenState(0, 0, elements=[
            UIElement(id="cb", role="checkbox", label="Remember me",
                      checked=False, bbox=(10, 10, 30, 30))])
        after = ScreenState(1, 500, elements=[
            UIElement(id="cb", role="checkbox", label="Remember me",
                      checked=True, bbox=(10, 10, 30, 30))])
        action = StateDiffIDM().infer(before, after)
        self.assertEqual(action.action_type, "click")
        self.assertEqual(action.effect.get("checked"), True)
        self.assertEqual(action.target.display(), "Remember me")


if __name__ == "__main__":
    unittest.main()
