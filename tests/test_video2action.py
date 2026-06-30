"""Full-chain test: raw tutorial video -> trajectory -> reusable skill.

Drives the VIDEO2ACTION inverse-dynamics module with the deterministic
``ScriptedBackend`` (standing in for a video that carries click/keystroke
overlays), then runs the trajectory through the *existing* normalize + induction
pipeline and asserts a valid, parameterized workflow skill comes out the other
end - i.e. the video modality plugs into the rest of Demo2Skill unchanged.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from demo2skill.induction.workflow_generator import induce_workflow
from demo2skill.trace.normalize import normalize_trace
from demo2skill.video.video2action import Frames, ScriptedBackend, Video2Action
from demo2skill.workflow.validator import validate_skill

EXAMPLE = Path(__file__).resolve().parents[1] / "demo2skill" / "examples" / "github_issue"


def _trajectory():
    records = json.loads((EXAMPLE / "video_events.json").read_text())
    backend = ScriptedBackend(records)
    return Video2Action(backend, backend).run(
        Frames.empty(), video_id="github_issue_tutorial", source="video_events.json"
    )


class TrajectoryTest(unittest.TestCase):
    def test_idm_recovers_actions(self):
        traj = _trajectory()
        types = [a.action_type for a in traj.actions]
        self.assertEqual(types, ["navigate", "click", "click", "navigate", "type", "type"])

        typed = [a for a in traj.actions if a.action_type == "type"]
        self.assertEqual(typed[0].text, "Bug in login flow")
        self.assertEqual(typed[0].target_label, "Title")
        # temporal grounding preserved
        self.assertEqual(traj.actions[0].start_ms, 0)

    def test_raw_trace_is_normalizer_ready(self):
        raw = _trajectory().to_raw_trace()
        self.assertEqual(raw["metadata"]["source_modality"], "video")
        # normalizer consumes it without selectors / DOM
        semantic = normalize_trace(raw)
        actions = [e["semantic_action"] for e in semantic["events"]]
        self.assertIn("fill_field", actions)
        self.assertIn("navigate", actions)


class VideoToSkillTest(unittest.TestCase):
    def test_full_chain_induces_valid_parameterized_skill(self):
        semantic = normalize_trace(_trajectory().to_raw_trace())
        skill = induce_workflow(semantic)

        # No structural/safety errors.
        errors = [i for i in validate_skill(skill) if i.severity == "error"]
        self.assertEqual(errors, [])

        # The two typed constants were lifted into named inputs (variables),
        # not baked in as literals - the whole point of inducing a *skill*.
        self.assertGreaterEqual(len(skill.inputs), 2)
        fill_steps = [s for s in skill.steps if s.action == "fill_field"]
        self.assertGreaterEqual(len(fill_steps), 2)
        for step in fill_steps:
            self.assertTrue(step.value and step.value.startswith("${"),
                            f"{step.step_id} value not parameterized: {step.value}")

        # Irreversible submit stays gated (induction adds the confirmation).
        actions = [s.action for s in skill.steps]
        self.assertTrue(
            "request_user_confirmation" in actions or "stop" in actions
        )


if __name__ == "__main__":
    unittest.main()
