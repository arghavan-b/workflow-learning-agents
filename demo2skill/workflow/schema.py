"""Pydantic schema for learned workflow skills (Demo2Skill Module 5).

LLM-generated YAML is messy, so this schema is the structural contract every
induced workflow must satisfy before it can be stored or executed. Cross-field
*semantic* rules (unbound variables, unsafe submits) live in
:mod:`demo2skill.workflow.validator`; this module enforces structure only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Action vocabulary supported by the v0 executor. The schema rejects anything
# outside this set so a hallucinated action never reaches the runtime loop.
ACTION_TYPES = (
    "navigate",
    "click",
    "fill_field",
    "upload_file",
    "extract_text",
    "verify",
    "request_user_confirmation",
    "stop",
)

INPUT_TYPES = ("string", "number", "file", "boolean")

VARIABLE_PATTERN = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _prune_empty(value: Any) -> Any:
    """Recursively drop empty lists/dicts so serialized YAML stays terse."""

    if isinstance(value, dict):
        return {k: _prune_empty(v) for k, v in value.items() if v not in ([], {})}
    if isinstance(value, list):
        return [_prune_empty(v) for v in value]
    return value

# Identifying fields a grounder can use to locate an element. A target must
# carry at least one of these (navigate targets carry ``url`` instead).
TARGET_IDENTIFIERS = (
    "text",
    "label",
    "role",
    "selector",
    "aria_label",
    "placeholder",
    "nearby_text",
    "semantic",
)


class StrictModel(BaseModel):
    """Base model that forbids unknown keys so messy YAML fails loudly."""

    model_config = ConfigDict(extra="forbid")


class WorkflowInput(StrictModel):
    name: str = Field(min_length=1)
    type: str = "string"
    required: bool = True
    description: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "WorkflowInput":
        if self.type not in INPUT_TYPES:
            raise ValueError(f"input '{self.name}' has unknown type '{self.type}'")
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", self.name):
            raise ValueError(f"input name '{self.name}' is not a valid identifier")
        return self


class Target(StrictModel):
    text: Optional[str] = None
    label: Optional[str] = None
    role: Optional[str] = None
    selector: Optional[str] = None
    aria_label: Optional[str] = None
    placeholder: Optional[str] = None
    nearby_text: Optional[str] = None
    semantic: Optional[str] = None
    url: Optional[str] = None

    @model_validator(mode="after")
    def _needs_identifier(self) -> "Target":
        if self.url:
            return self
        if not any(getattr(self, key) for key in TARGET_IDENTIFIERS):
            raise ValueError(
                "target must provide at least one of "
                f"{TARGET_IDENTIFIERS} or a url"
            )
        return self


class FieldEquals(StrictModel):
    label: Optional[str] = None
    selector: Optional[str] = None
    value: str

    @model_validator(mode="after")
    def _needs_locator(self) -> "FieldEquals":
        if not (self.label or self.selector):
            raise ValueError("field_equals requires a label or selector")
        return self


class Check(StrictModel):
    field_equals: Optional[FieldEquals] = None
    field_filled: Optional[str] = None
    page_contains: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "Check":
        present = [k for k in ("field_equals", "field_filled", "page_contains")
                   if getattr(self, k) is not None]
        if len(present) != 1:
            raise ValueError("each check must set exactly one of "
                             "field_equals, field_filled, page_contains")
        return self


class Postcondition(StrictModel):
    page_contains: Optional[str] = None
    url_contains: Optional[str] = None

    @model_validator(mode="after")
    def _needs_one(self) -> "Postcondition":
        if not (self.page_contains or self.url_contains):
            raise ValueError("postcondition requires page_contains or url_contains")
        return self


class WorkflowStep(StrictModel):
    step_id: str = Field(min_length=1)
    action: str
    target: Optional[Target] = None
    value: Optional[str] = None
    url: Optional[str] = None
    checks: List[Check] = Field(default_factory=list)
    postcondition: Optional[Postcondition] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _check_action_shape(self) -> "WorkflowStep":
        if self.action not in ACTION_TYPES:
            raise ValueError(
                f"step '{self.step_id}' has unknown action '{self.action}'; "
                f"allowed: {ACTION_TYPES}"
            )
        if self.action == "navigate":
            if not (self.url or (self.target and self.target.url)):
                raise ValueError(f"navigate step '{self.step_id}' requires a url")
        if self.action == "click" and not self.target:
            raise ValueError(f"click step '{self.step_id}' requires a target")
        if self.action in ("fill_field", "upload_file"):
            if not self.target:
                raise ValueError(f"{self.action} step '{self.step_id}' requires a target")
            if self.value in (None, ""):
                raise ValueError(f"{self.action} step '{self.step_id}' requires a value")
        if self.action == "verify" and not self.checks:
            raise ValueError(f"verify step '{self.step_id}' requires at least one check")
        if self.action == "request_user_confirmation" and not self.reason:
            raise ValueError(
                f"request_user_confirmation step '{self.step_id}' requires a reason"
            )
        return self


class RecoveryRule(StrictModel):
    condition: str = Field(min_length=1)
    strategy: str = Field(min_length=1)


class WorkflowSkill(StrictModel):
    workflow_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    inputs: List[WorkflowInput] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)
    steps: List[WorkflowStep] = Field(min_length=1)
    recovery: List[RecoveryRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "WorkflowSkill":
        seen = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"duplicate step_id '{step.step_id}'")
            seen.add(step.step_id)
        return self

    # -- convenience ---------------------------------------------------------

    @property
    def input_names(self) -> List[str]:
        return [inp.name for inp in self.inputs]

    def iter_variable_refs(self) -> Iterator[Tuple[str, str]]:
        """Yield ``(step_id, variable_name)`` for every ``${var}`` reference."""

        for step in self.steps:
            texts = [step.value]
            for check in step.checks:
                if check.field_equals:
                    texts.append(check.field_equals.value)
            for text in texts:
                if not text:
                    continue
                for match in VARIABLE_PATTERN.finditer(text):
                    yield step.step_id, match.group(1)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return _prune_empty(self.model_dump(exclude_none=True))

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            self.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False
        )

    @classmethod
    def from_dict(cls, data: Any) -> "WorkflowSkill":
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, text: str) -> "WorkflowSkill":
        return cls.model_validate(yaml.safe_load(text))
