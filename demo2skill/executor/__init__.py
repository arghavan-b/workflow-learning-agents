"""Demo2Skill executor: run an induced workflow skill against a page, verify
each step, and self-heal via a bounded repair loop.

This is the reliability core (Modules 6-9 of the design): a learned skill is a
hypothesis, and execution + verification + repair is what turns it into robust,
reusable procedural memory. The page model here is a dependency-free DOM stand-in
so the loop is fully testable without a browser; the same ``Grounder`` /
``Verifier`` interfaces back onto a real Playwright page in production.
"""

from demo2skill.executor.executor import WorkflowExecutor, run_workflow
from demo2skill.executor.models import (
    FailureRecord,
    GroundingResult,
    RepairPatch,
    RunResult,
    StepResult,
)
from demo2skill.executor.page import Element, Page

__all__ = [
    "WorkflowExecutor",
    "run_workflow",
    "Page",
    "Element",
    "GroundingResult",
    "StepResult",
    "FailureRecord",
    "RepairPatch",
    "RunResult",
]
