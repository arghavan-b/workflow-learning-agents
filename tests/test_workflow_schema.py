import unittest

from pydantic import ValidationError

from demo2skill.workflow.schema import WorkflowSkill
from demo2skill.workflow.validator import validate_skill

VALID = {
    "workflow_id": "create_github_issue_v1",
    "goal": "Create a GitHub issue",
    "inputs": [
        {"name": "title", "type": "string"},
        {"name": "body", "type": "string"},
    ],
    "preconditions": ["user_logged_in"],
    "steps": [
        {"step_id": "open_form", "action": "navigate",
         "url": "https://github.com/u/r/issues/new"},
        {"step_id": "fill_title", "action": "fill_field",
         "target": {"label": "Add a title"}, "value": "${title}"},
        {"step_id": "fill_body", "action": "fill_field",
         "target": {"label": "Markdown value"}, "value": "${body}"},
        {"step_id": "verify_form", "action": "verify",
         "checks": [{"field_equals": {"label": "Add a title", "value": "${title}"}}]},
        {"step_id": "confirm", "action": "request_user_confirmation",
         "reason": "Submitting is irreversible"},
    ],
}


class SchemaStructureTests(unittest.TestCase):
    def test_valid_skill_parses_and_roundtrips(self):
        skill = WorkflowSkill.from_dict(VALID)
        again = WorkflowSkill.from_yaml(skill.to_yaml())
        self.assertEqual(again.workflow_id, "create_github_issue_v1")
        self.assertEqual(again.input_names, ["title", "body"])

    def test_unknown_action_is_rejected(self):
        bad = {**VALID, "steps": [{"step_id": "s1", "action": "teleport"}]}
        with self.assertRaises(ValidationError):
            WorkflowSkill.from_dict(bad)

    def test_missing_step_id_is_rejected(self):
        bad = {**VALID, "steps": [{"step_id": "", "action": "stop"}]}
        with self.assertRaises(ValidationError):
            WorkflowSkill.from_dict(bad)

    def test_duplicate_step_ids_are_rejected(self):
        bad = {**VALID, "steps": [
            {"step_id": "dup", "action": "stop"},
            {"step_id": "dup", "action": "stop"},
        ]}
        with self.assertRaises(ValidationError):
            WorkflowSkill.from_dict(bad)

    def test_fill_field_requires_target_and_value(self):
        bad = {**VALID, "steps": [
            {"step_id": "f", "action": "fill_field", "target": {"label": "X"}}
        ]}
        with self.assertRaises(ValidationError):
            WorkflowSkill.from_dict(bad)

    def test_target_without_identifier_is_rejected(self):
        bad = {**VALID, "steps": [
            {"step_id": "c", "action": "click", "target": {"role": ""}}
        ]}
        with self.assertRaises(ValidationError):
            WorkflowSkill.from_dict(bad)


class SemanticValidationTests(unittest.TestCase):
    def test_valid_skill_has_no_issues(self):
        self.assertEqual(validate_skill(WorkflowSkill.from_dict(VALID)), [])

    def test_unbound_variable_is_an_error(self):
        bad = {**VALID, "inputs": [{"name": "title", "type": "string"}]}
        # body is referenced but no longer declared
        issues = validate_skill(WorkflowSkill.from_dict(bad))
        codes = {i.code for i in issues if i.severity == "error"}
        self.assertIn("unbound_variable", codes)

    def test_unused_input_is_a_warning(self):
        extra = {**VALID, "inputs": VALID["inputs"] + [{"name": "spare", "type": "string"}]}
        issues = validate_skill(WorkflowSkill.from_dict(extra))
        self.assertTrue(any(i.code == "unused_input" and i.severity == "warning"
                            for i in issues))

    def test_ungated_submit_click_is_unsafe(self):
        steps = [
            {"step_id": "fill_title", "action": "fill_field",
             "target": {"label": "Add a title"}, "value": "${title}"},
            {"step_id": "fill_body", "action": "fill_field",
             "target": {"label": "Markdown value"}, "value": "${body}"},
            {"step_id": "submit", "action": "click",
             "target": {"text": "Submit new issue", "role": "button"}},
        ]
        issues = validate_skill(WorkflowSkill.from_dict({**VALID, "steps": steps}))
        self.assertTrue(any(i.code == "unsafe_submit" for i in issues))

    def test_gated_submit_click_is_safe(self):
        steps = [
            {"step_id": "fill_title", "action": "fill_field",
             "target": {"label": "Add a title"}, "value": "${title}"},
            {"step_id": "fill_body", "action": "fill_field",
             "target": {"label": "Markdown value"}, "value": "${body}"},
            {"step_id": "confirm", "action": "request_user_confirmation",
             "reason": "irreversible"},
            {"step_id": "submit", "action": "click",
             "target": {"text": "Submit new issue", "role": "button"}},
        ]
        issues = validate_skill(WorkflowSkill.from_dict({**VALID, "steps": steps}))
        self.assertFalse(any(i.code == "unsafe_submit" for i in issues))


if __name__ == "__main__":
    unittest.main()
