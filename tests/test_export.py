"""Round-trip: induced GitHub-issue skill -> run -> verified trajectory.

The exporter is the bridge that turns a Demo2Skill run into training data for a
policy model. These tests pin the two properties that make that data worth
training on:

* concrete supervision - ``${title}`` is bound to the real typed string;
* the verification filter - only confirmed steps leak through, and they carry
  the grounded target / method the run actually used.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from demo2skill.executor import run_workflow
from demo2skill.executor.page import Page
from demo2skill.export import export_trajectory
from demo2skill.export.trajectory import StepObservation
from demo2skill.workflow.schema import WorkflowSkill

EXAMPLE = Path(__file__).resolve().parents[1] / "demo2skill" / "examples" / "github_issue"


def _load():
    workflow = WorkflowSkill.from_yaml((EXAMPLE / "induced_workflow.yaml").read_text())
    inputs = json.loads((EXAMPLE / "test_inputs.json").read_text())
    return workflow, inputs


class MatchedPageExportTest(unittest.TestCase):
    def setUp(self):
        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_match.html")
        self.result, self.final = run_workflow(workflow, page, inputs)
        self.inputs = inputs
        self.traj = export_trajectory(self.final, self.result, inputs)

    def test_episode_is_verified_and_good(self):
        # Halting at the confirmation gate is a good, exportable episode.
        self.assertEqual(self.traj.episode_status, "halted_for_confirmation")
        self.assertTrue(self.traj.verified)

    def test_emits_navigate_and_two_fills(self):
        actions = [s.action["type"] for s in self.traj.steps]
        self.assertEqual(actions, ["navigate", "type", "type"])

    def test_no_control_steps_leak(self):
        ids = [s.provenance["step_id"] for s in self.traj.steps]
        self.assertNotIn("verify_form", ids)
        self.assertNotIn("confirm_before_submit", ids)

    def test_values_are_concrete_not_templated(self):
        fills = [s for s in self.traj.steps if s.action["type"] == "type"]
        texts = [s.action["text"] for s in fills]
        self.assertIn(self.inputs["title"], texts)
        self.assertIn(self.inputs["body"], texts)
        for t in texts:
            self.assertNotIn("${", t)

    def test_steps_carry_grounding_method(self):
        fills = [s for s in self.traj.steps if s.action["type"] == "type"]
        # On the matched page the grounder pins by selector.
        self.assertTrue(all(s.action["grounding"]["method"] == "selector" for s in fills))
        self.assertTrue(all(s.verified for s in self.traj.steps))

    def test_jsonl_is_one_object_per_step(self):
        lines = [ln for ln in self.traj.to_jsonl().splitlines() if ln.strip()]
        self.assertEqual(len(lines), len(self.traj.steps))
        for ln in lines:
            obj = json.loads(ln)  # each line parses on its own
            self.assertEqual(obj["episode_id"], self.traj.episode_id)


class ShiftedPageExportTest(unittest.TestCase):
    def test_repaired_targets_are_what_gets_exported(self):
        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_shifted.html")
        result, final = run_workflow(workflow, page, inputs)

        traj = export_trajectory(final, result, inputs, source_modality="recorder")

        title = next(s for s in traj.steps if s.provenance["step_id"] == "fill_title")
        # The exported target is the *repaired* label, not the brittle demo one.
        self.assertEqual(title.action["target"]["label"], "Issue title")
        self.assertTrue(title.provenance["repaired"])
        self.assertTrue(title.verified)


class FailedRunExportTest(unittest.TestCase):
    def test_failed_episode_is_not_marked_verified(self):
        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_match.html")
        result, final = run_workflow(workflow, page, inputs)
        # Simulate a failed episode: no confirmed steps should be trusted.
        result.status = "failed"
        traj = export_trajectory(final, result, inputs)
        self.assertFalse(traj.verified)
        self.assertTrue(all(not s.verified for s in traj.steps))


class ObservationEnrichmentTest(unittest.TestCase):
    def test_substrate_supplied_screenshot_and_coordinate_pass_through(self):
        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_match.html")
        result, final = run_workflow(workflow, page, inputs)

        obs = {"fill_title": StepObservation(
            screenshot="screens/fill_title.png", coordinate=[120, 240])}
        traj = export_trajectory(final, result, inputs, observations=obs)

        title = next(s for s in traj.steps if s.provenance["step_id"] == "fill_title")
        self.assertEqual(title.observation["screenshot"], "screens/fill_title.png")
        self.assertEqual(title.action["coordinate"], [120, 240])


if __name__ == "__main__":
    unittest.main()
