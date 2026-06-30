"""The runtime loop: execute a skill, verify each step, self-heal on failure.

    for step in workflow.steps:
        bind variables -> ground target -> act -> verify
        on failure: record -> propose patch -> re-validate -> bounded retry

Irreversible actions stay gated: ``request_user_confirmation`` halts the run by
default, and the safety validator runs on every repaired skill, so a patch can
never remove a confirmation gate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

from demo2skill.executor.grounding import Grounder
from demo2skill.executor.models import (
    GROUNDING_THRESHOLD,
    FailureRecord,
    GroundingResult,
    RepairPatch,
    RunResult,
    StepResult,
)
from demo2skill.executor.page import Page
from demo2skill.executor.repair import apply_patch, propose_repair
from demo2skill.executor.verify import Verifier
from demo2skill.workflow.schema import VARIABLE_PATTERN, WorkflowSkill, WorkflowStep

MAX_ATTEMPTS_PER_STEP = 3


def _bind_text(text: Optional[str], inputs: Dict[str, Any], missing: Set[str]) -> Optional[str]:
    if not text:
        return text

    def repl(match):
        name = match.group(1)
        if name in inputs and inputs[name] is not None:
            return str(inputs[name])
        missing.add(name)
        return match.group(0)

    return VARIABLE_PATTERN.sub(repl, text)


def bind_step(step: WorkflowStep, inputs: Dict[str, Any]) -> Tuple[WorkflowStep, Set[str]]:
    """Substitute ``${var}`` in the step's value and checks with ``inputs``."""

    missing: Set[str] = set()
    data = step.model_dump()
    data["value"] = _bind_text(data.get("value"), inputs, missing)
    for check in data.get("checks", []):
        fe = check.get("field_equals")
        if fe and fe.get("value"):
            fe["value"] = _bind_text(fe["value"], inputs, missing)
    return WorkflowStep.model_validate(data), missing


class WorkflowExecutor:
    def __init__(self, page: Page, *, auto_confirm: bool = False) -> None:
        self.page = page
        self.auto_confirm = auto_confirm
        self.grounder = Grounder()
        self.verifier = Verifier(self.grounder)

    def run(self, workflow: WorkflowSkill, inputs: Optional[Dict[str, Any]] = None) -> Tuple[RunResult, WorkflowSkill]:
        """Execute ``workflow``; return the run result and the (possibly
        repaired) workflow so the caller can persist a more robust version."""

        inputs = inputs or {}
        result = RunResult(workflow_id=workflow.workflow_id, status="completed")
        current = workflow
        tried_patches: Set[str] = set()

        for original in workflow.steps:
            step_id = original.step_id
            attempt = 0
            while True:
                attempt += 1
                step = _find_step(current, step_id)
                bound, missing = bind_step(step, inputs)

                if missing:
                    result.failures.append(
                        FailureRecord(step_id, "missing_input", attempt,
                                      detail=f"unbound inputs: {sorted(missing)}")
                    )
                    result.steps.append(StepResult(step_id, step.action, "failed", attempt,
                                                    detail="missing input"))
                    result.status = "failed"
                    return result, current

                failure_type, grounding, detail = self._exec(bound)

                if failure_type is None:
                    status = "repaired" if attempt > 1 else "ok"
                    result.steps.append(StepResult(step_id, step.action, status, attempt,
                                                    grounding, detail))
                    break

                if failure_type == "__halt__":
                    result.steps.append(StepResult(step_id, step.action, "halted", attempt,
                                                    detail=detail))
                    result.status = "halted_for_confirmation"
                    return result, current

                # -- failure: record, propose a repair, retry within budget --
                record = FailureRecord(
                    step_id, failure_type, attempt, detail=detail,
                    grounding_confidence=grounding.confidence if grounding else 0.0,
                    previous_target=(step.target.model_dump(exclude_none=True) if step.target else {}),
                    page_text=self.page.text()[:500],
                )
                result.failures.append(record)

                patch = None
                if attempt < MAX_ATTEMPTS_PER_STEP:
                    patch = propose_repair(record, current, self.page)

                if patch is None or patch.signature() in tried_patches:
                    result.steps.append(StepResult(step_id, step.action, "failed", attempt,
                                                    grounding, detail))
                    result.status = "failed"
                    return result, current

                try:
                    current = apply_patch(current, patch)
                except ValueError as exc:  # rejected by schema/safety validator
                    result.steps.append(StepResult(step_id, step.action, "failed", attempt,
                                                    grounding, f"patch rejected: {exc}"))
                    result.status = "failed"
                    return result, current

                tried_patches.add(patch.signature())
                result.patches.append(patch)
                # loop: retry the step with the patched target

        return result, current

    # -- single-step execution ----------------------------------------------

    def _exec(self, step: WorkflowStep) -> Tuple[Optional[str], Optional[GroundingResult], str]:
        """Return ``(failure_type, grounding, detail)``; ``failure_type`` is
        ``None`` on success and ``"__halt__"`` for a confirmation gate."""

        action = step.action

        if action == "navigate":
            self.page.url = step.url or (step.target.url if step.target else None) or self.page.url
            return None, None, "navigated"

        if action == "request_user_confirmation":
            if self.auto_confirm:
                return None, None, "confirmed"
            return "__halt__", None, step.reason or "awaiting user confirmation"

        if action == "stop":
            return "__halt__", None, "stop"

        if action == "verify":
            check = self.verifier.verify_checks(step.checks, self.page)
            return (None if check.ok else "verification_failed"), None, check.detail

        if action in ("click", "fill_field", "upload_file", "extract_text"):
            grounding = self.grounder.ground(step.target, self.page, strict=True)
            if not grounding.found:
                return "target_not_found", grounding, "no matching element"
            if grounding.confidence < GROUNDING_THRESHOLD:
                return "low_grounding_confidence", grounding, f"confidence {grounding.confidence}"

            if action == "fill_field":
                self.page.fill(grounding.element_id, step.value or "")
            elif action == "upload_file":
                self.page.fill(grounding.element_id, step.value or "")
            elif action == "click":
                self.page.click(grounding.element_id)
            # extract_text: grounding alone is the effect for v0

            post = self.verifier.verify_postcondition(step.postcondition, self.page)
            if not post.ok:
                return "postcondition_failed", grounding, post.detail
            return None, grounding, f"{grounding.method}@{grounding.confidence}"

        return "unsupported_action", None, f"action '{action}' not supported"


def _find_step(workflow: WorkflowSkill, step_id: str) -> WorkflowStep:
    for step in workflow.steps:
        if step.step_id == step_id:
            return step
    raise KeyError(step_id)


def run_workflow(
    workflow: WorkflowSkill,
    page: Page,
    inputs: Optional[Dict[str, Any]] = None,
    *,
    auto_confirm: bool = False,
) -> Tuple[RunResult, WorkflowSkill]:
    """Convenience wrapper: build an executor and run once."""

    return WorkflowExecutor(page, auto_confirm=auto_confirm).run(workflow, inputs)
