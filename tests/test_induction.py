import unittest

from demo2skill.induction.workflow_generator import (
    generate_workflow_baseline,
    induce_workflow,
)
from demo2skill.workflow.validator import validate_skill


def _semantic_event(event_id, action, target=None, value=None, url=None, title=None):
    return {
        "event_id": event_id,
        "semantic_action": action,
        "target": target or {},
        "value": value,
        "url": url,
        "page_title": title,
    }


# A trace that mirrors the noisy patterns of a real demonstration:
# exploratory navigation, a login flow, a hidden env write, per-keystroke
# typing, a focus click, and a typed correction.
NOISY_TRACE = {
    "schema_version": "demo2skill.semantic_trace.v0",
    "events": [
        _semantic_event("e1", "navigate", {"url": "https://app.test/"},
                        url="https://app.test/"),
        _semantic_event("e2", "navigate", {"url": "https://app.test/login"},
                        url="https://app.test/login"),
        _semantic_event("e3", "set_value",
                        {"name": "webauthn-support", "label": "webauthn-support"},
                        value="supported", url="https://app.test/login"),
        _semantic_event("e4", "fill_field",
                        {"label": "Username or email address", "name": "login"},
                        value="alice", url="https://app.test/login"),
        _semantic_event("e5", "fill_field", {"label": "Password", "name": "password"},
                        value="secret", url="https://app.test/login"),
        _semantic_event("e6", "navigate", {"url": "https://app.test/reports"},
                        url="https://app.test/reports"),
        _semantic_event("e7", "click", {"text": "New Report", "role": "button"},
                        url="https://app.test/reports"),
        _semantic_event("e8", "navigate", {"url": "https://app.test/reports/new"},
                        url="https://app.test/reports/new"),
        # focus click then per-keystroke typing into the amount field
        _semantic_event("e9", "click",
                        {"label": "Amount", "selector": "input[name=amount]"},
                        url="https://app.test/reports/new"),
        _semantic_event("e10", "fill_field",
                        {"label": "Amount", "selector": "input[name=amount]",
                         "role": "textbox"}, value="4",
                        url="https://app.test/reports/new"),
        _semantic_event("e11", "fill_field",
                        {"label": "Amount", "selector": "input[name=amount]",
                         "role": "textbox"}, value="42",
                        url="https://app.test/reports/new"),
        _semantic_event("e12", "set_value",
                        {"label": "Amount", "selector": "input[name=amount]",
                         "role": "textbox"}, value="42.50",
                        url="https://app.test/reports/new"),
        _semantic_event("e13", "fill_field",
                        {"label": "Merchant", "selector": "input[name=merchant]",
                         "role": "textbox"}, value="Amazon",
                        url="https://app.test/reports/new"),
    ],
}


class BaselineInductionTests(unittest.TestCase):
    def setUp(self):
        self.skill = generate_workflow_baseline(NOISY_TRACE)

    def test_login_becomes_precondition_not_steps(self):
        self.assertIn("user_logged_in", self.skill.preconditions)
        blob = self.skill.to_yaml().lower()
        self.assertNotIn("password", blob)
        self.assertNotIn("username", blob)

    def test_environment_writes_are_dropped(self):
        self.assertNotIn("webauthn", self.skill.to_yaml().lower())

    def test_keystrokes_collapse_to_final_value_and_variables(self):
        self.assertEqual(self.skill.input_names, ["amount", "merchant"])
        amount = next(i for i in self.skill.inputs if i.name == "amount")
        self.assertEqual(amount.type, "number")  # "42.50" inferred numeric
        fills = {s.step_id: s.value for s in self.skill.steps if s.action == "fill_field"}
        self.assertEqual(fills["fill_amount"], "${amount}")
        self.assertEqual(fills["fill_merchant"], "${merchant}")

    def test_navigation_chain_collapses_to_single_navigate(self):
        navigates = [s for s in self.skill.steps if s.action == "navigate"]
        self.assertEqual(len(navigates), 1)
        self.assertEqual(navigates[0].url, "https://app.test/reports/new")
        # the exploratory "New Report" click is folded into the navigation
        self.assertFalse(any(s.action == "click" for s in self.skill.steps))

    def test_safety_gate_and_verification_appended(self):
        actions = [s.action for s in self.skill.steps]
        self.assertIn("verify", actions)
        self.assertEqual(actions[-1], "request_user_confirmation")

    def test_baseline_output_is_valid(self):
        self.assertFalse(any(i.severity == "error" for i in validate_skill(self.skill)))


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, *, system, prompt):
        self.calls += 1
        return self.reply


VALID_LLM_YAML = """
workflow_id: llm_skill_v1
goal: Do the thing
inputs:
  - name: amount
    type: number
  - name: merchant
    type: string
steps:
  - step_id: open
    action: navigate
    url: https://app.test/reports/new
  - step_id: fill_amount
    action: fill_field
    target:
      label: Amount
    value: ${amount}
  - step_id: confirm
    action: request_user_confirmation
    reason: irreversible
"""


class LLMPathTests(unittest.TestCase):
    def test_llm_output_is_used_when_valid(self):
        llm = FakeLLM(VALID_LLM_YAML)
        skill = induce_workflow(NOISY_TRACE, llm=llm)
        self.assertEqual(skill.workflow_id, "llm_skill_v1")
        self.assertGreaterEqual(llm.calls, 1)

    def test_falls_back_to_baseline_on_garbage(self):
        skill = induce_workflow(NOISY_TRACE, llm=FakeLLM("not: [valid"))
        # baseline id, not the llm id
        self.assertEqual(skill.workflow_id, generate_workflow_baseline(NOISY_TRACE).workflow_id)


if __name__ == "__main__":
    unittest.main()
