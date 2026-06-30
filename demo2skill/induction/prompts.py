"""Prompt templates for the LLM induction path (Demo2Skill Modules 3-4).

These mirror the prompts in the design doc. They are only used when an
:class:`~demo2skill.induction.llm.LLMClient` is supplied; the deterministic
baseline ignores them.
"""

from __future__ import annotations

from demo2skill.workflow.schema import ACTION_TYPES, INPUT_TYPES

SEGMENT_SYSTEM = """\
You are given a browser demonstration trace as a list of semantic events.
Group the events into meaningful workflow segments.
Remove accidental, duplicate, or exploratory actions.
Return JSON: a list of objects with keys
  segment_id, name, intent, events (list of event_id), essential (bool).
Return only JSON.
"""

INDUCE_SYSTEM = f"""\
Given a segmented human demonstration, infer a reusable workflow skill.

Rules:
- Do not copy mouse coordinates.
- Generalize typed values into ${{variables}}.
- Remove accidental actions.
- Represent targets semantically using text, label, role, or intent.
- Add verification checks after important form-filling steps.
- Add a request_user_confirmation step before any irreversible action.
- Use only these actions: {", ".join(ACTION_TYPES)}.
- Declare every variable as an input with a type from: {", ".join(INPUT_TYPES)}.
- Return valid YAML matching the WorkflowSkill schema, nothing else.
"""

REPAIR_SYSTEM = """\
The workflow failed at the given step.
Given the current DOM/screenshot and the original step, propose a repaired
target or step. Do not change the goal. Return a YAML patch only.
"""


def segment_prompt(semantic_events_json: str) -> str:
    return f"Demonstration trace:\n{semantic_events_json}\n\nReturn the segments as JSON."


def induce_prompt(segments_json: str) -> str:
    return (
        f"Segmented demonstration:\n{segments_json}\n\n"
        "Return the WorkflowSkill as YAML."
    )
