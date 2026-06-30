"""Turn a failed step into a schema-validated patch on the workflow.

This is the heart of the reliability story: a learned skill is a hypothesis, and
when it meets a shifted UI the system repairs the *skill* rather than giving up.
Repairs are keyed by ``failure_type`` - the same keys emitted into every
workflow's ``recovery`` block - and every patch is re-validated against the
Pydantic schema + safety validator before it can be retried, so a bad repair can
never reach the runtime loop.
"""

from __future__ import annotations

from typing import Optional

from demo2skill.executor.grounding import best_relabel
from demo2skill.executor.models import FailureRecord, RepairPatch
from demo2skill.executor.page import Page
from demo2skill.workflow.schema import Target, WorkflowSkill
from demo2skill.workflow.validator import assert_valid

# failure_type -> recovery strategy (mirrors the recovery rules in the schema).
STRATEGY = {
    "target_not_found": "semantic_search_then_retry",
    "low_grounding_confidence": "semantic_search_then_retry",
    "verification_failed": "stop_and_report",
    "missing_input": "ask_user",
}


def propose_repair(
    record: FailureRecord, workflow: WorkflowSkill, page: Page
) -> Optional[RepairPatch]:
    """Return a patch that might fix ``record``, or ``None`` to escalate."""

    strategy = STRATEGY.get(record.failure_type)

    if strategy == "semantic_search_then_retry":
        step = _find_step(workflow, record.step_id)
        if step is None or step.target is None:
            return None
        # Re-ground the stale target against the live page, selector stripped,
        # and adopt the current label/role of the element it most likely meant.
        el = best_relabel(step.target, page)
        if el is None:
            return None
        new_target = {"role": el.role} if el.role else {}
        if el.label:
            new_target["label"] = el.label
        elif el.placeholder:
            new_target["placeholder"] = el.placeholder
        elif el.text:
            new_target["text"] = el.text
        if not new_target:
            return None
        return RepairPatch(
            op="replace",
            step_id=record.step_id,
            target=new_target,
            reason=f"re-grounded after {record.failure_type}: "
            f"{record.previous_target} -> {new_target}",
        )

    # verification_failed / missing_input / unknown -> no silent auto-fix.
    return None


def apply_patch(workflow: WorkflowSkill, patch: RepairPatch) -> WorkflowSkill:
    """Return a new, re-validated workflow with ``patch`` applied.

    Raises ``ValueError`` (via :func:`assert_valid`) if the patch would produce
    an unsafe or malformed skill - the repair is rejected before retry.
    """

    data = workflow.to_dict()
    steps = data.get("steps", [])

    if patch.op == "replace":
        for step in steps:
            if step.get("step_id") == patch.step_id:
                if patch.target is not None:
                    step["target"] = patch.target
                break
    elif patch.op == "delete":
        data["steps"] = [s for s in steps if s.get("step_id") != patch.step_id]
    else:
        raise ValueError(f"unsupported patch op '{patch.op}'")

    repaired = WorkflowSkill.from_dict(data)
    assert_valid(repaired)  # schema + safety gate before the patch is trusted
    return repaired


def _find_step(workflow: WorkflowSkill, step_id: str):
    for step in workflow.steps:
        if step.step_id == step_id:
            return step
    return None
