"""Semantic validation for induced workflow skills (Demo2Skill Module 5).

The Pydantic schema in :mod:`demo2skill.workflow.schema` guarantees structure
(known actions, step ids, valid targets). This module adds the cross-field
*safety* and *consistency* rules the design doc calls out:

* unbound variables   - a ``${var}`` step references an undeclared input
* unsafe submit       - an irreversible submit/save click is not gated behind
                        a ``request_user_confirmation`` (or ``stop``) step
* unused input        - a declared input is never referenced (warning only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from demo2skill.workflow.schema import WorkflowSkill, WorkflowStep

# Click targets that look like an irreversible commit. The v0 GitHub milestone
# explicitly stops before submit, so an ungated submit is a hard error.
SUBMIT_PATTERN = re.compile(
    r"\b(submit|create issue|create pull|save changes|send|publish|post comment|"
    r"confirm|place order|pay now|delete)\b",
    re.IGNORECASE,
)

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    step_id: str = ""

    def __str__(self) -> str:
        where = f" [{self.step_id}]" if self.step_id else ""
        return f"{self.severity.upper()} {self.code}{where}: {self.message}"


def _submit_text(step: WorkflowStep) -> str:
    if not step.target:
        return ""
    parts = [step.target.text, step.target.label, step.target.aria_label,
             step.target.selector, step.target.semantic]
    return " ".join(p for p in parts if p)


def validate_skill(skill: WorkflowSkill) -> List[ValidationIssue]:
    """Return all semantic issues; an empty list means the skill is safe to store."""

    issues: List[ValidationIssue] = []
    declared = set(skill.input_names)

    # Unbound variables: every ${var} must resolve to a declared input.
    referenced = set()
    for step_id, var in skill.iter_variable_refs():
        referenced.add(var)
        if var not in declared:
            issues.append(
                ValidationIssue(
                    ERROR,
                    "unbound_variable",
                    f"'${{{var}}}' is not a declared input",
                    step_id,
                )
            )

    # Unused inputs are suspicious but not fatal.
    for name in declared - referenced:
        issues.append(
            ValidationIssue(WARNING, "unused_input", f"input '{name}' is never used")
        )

    # Unsafe submit: a submit-like click must be preceded by a confirmation gate.
    gated = False
    for step in skill.steps:
        if step.action in ("request_user_confirmation", "stop"):
            gated = True
        if step.action == "click" and SUBMIT_PATTERN.search(_submit_text(step)):
            if not gated:
                issues.append(
                    ValidationIssue(
                        ERROR,
                        "unsafe_submit",
                        "irreversible submit click is not gated behind a "
                        "request_user_confirmation or stop step",
                        step.step_id,
                    )
                )

    return issues


def assert_valid(skill: WorkflowSkill) -> None:
    """Raise ``ValueError`` if the skill has any error-severity issues."""

    errors = [i for i in validate_skill(skill) if i.severity == ERROR]
    if errors:
        joined = "\n".join(f"  - {i}" for i in errors)
        raise ValueError(f"workflow '{skill.workflow_id}' failed validation:\n{joined}")
