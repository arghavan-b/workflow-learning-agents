"""End-to-end tests for the executor + self-healing repair loop.

The induced GitHub-issue skill is run against two page snapshots:

* ``page_match.html``   - matches the demo: grounds by selector, zero repairs.
* ``page_shifted.html`` - volatile ids changed and labels reworded: strict
  grounding fails, the repair loop re-grounds semantically and patches the
  targets, and the run still completes (halting at the confirmation gate).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from demo2skill.executor import run_workflow
from demo2skill.executor.page import Page
from demo2skill.workflow.schema import WorkflowSkill

EXAMPLE = Path(__file__).resolve().parents[1] / "demo2skill" / "examples" / "github_issue"


def _load():
    workflow = WorkflowSkill.from_yaml((EXAMPLE / "induced_workflow.yaml").read_text())
    inputs = json.loads((EXAMPLE / "test_inputs.json").read_text())
    return workflow, inputs


class HappyPathTest(unittest.TestCase):
    def test_matched_page_needs_no_repair(self):
        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_match.html")

        result, repaired = run_workflow(workflow, page, inputs)

        # Halts at the confirmation gate (safety), having needed no repairs.
        self.assertEqual(result.status, "halted_for_confirmation")
        self.assertTrue(result.converged)
        self.assertEqual(result.patches, [])

        fills = {s.step_id: s for s in result.steps if s.action == "fill_field"}
        self.assertEqual(fills["fill_title"].status, "ok")
        self.assertEqual(fills["fill_title"].grounding.method, "selector")

    def test_matched_page_completes_when_confirmed(self):
        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_match.html")
        result, _ = run_workflow(workflow, page, inputs, auto_confirm=True)
        self.assertEqual(result.status, "completed")


class SelfHealTest(unittest.TestCase):
    def test_shifted_page_self_heals(self):
        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_shifted.html")

        result, repaired = run_workflow(workflow, page, inputs)

        # Despite the layout shift the run reaches the confirmation gate...
        self.assertEqual(result.status, "halted_for_confirmation")
        # ...but only by repairing itself, so it is NOT converged.
        self.assertFalse(result.converged)
        self.assertTrue(result.repaired)

        repaired_steps = {s.step_id for s in result.steps if s.status == "repaired"}
        self.assertIn("fill_title", repaired_steps)
        self.assertIn("fill_body", repaired_steps)

        # The failures that drove the repairs were grounding failures.
        self.assertTrue(all(f.failure_type == "target_not_found" for f in result.failures))

        # The repaired skill adopted the page's current labels - the more robust
        # version a store would persist for next time.
        title_step = next(s for s in repaired.steps if s.step_id == "fill_title")
        self.assertEqual(title_step.target.label, "Issue title")

    def test_repaired_skill_still_passes_safety_validation(self):
        from demo2skill.workflow.validator import validate_skill

        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_shifted.html")
        _, repaired = run_workflow(workflow, page, inputs)

        errors = [i for i in validate_skill(repaired) if i.severity == "error"]
        self.assertEqual(errors, [])
        # Confirmation gate survived the repairs.
        actions = [s.action for s in repaired.steps]
        self.assertIn("request_user_confirmation", actions)


class IdempotentReplayTest(unittest.TestCase):
    def test_repaired_skill_runs_clean_on_same_page(self):
        """Re-running the *repaired* skill on the shifted page should converge
        with no further repairs - the loop has actually learned."""

        workflow, inputs = _load()
        page = Page.from_file(EXAMPLE / "page_shifted.html")
        _, repaired = run_workflow(workflow, page, inputs)

        page2 = Page.from_file(EXAMPLE / "page_shifted.html")
        result2, _ = run_workflow(repaired, page2, inputs)
        self.assertTrue(result2.converged)
        self.assertEqual(result2.patches, [])


if __name__ == "__main__":
    unittest.main()
