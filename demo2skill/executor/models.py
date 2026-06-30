"""Runtime data objects for the executor and repair loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GroundingResult:
    """Outcome of resolving a workflow ``target`` to a concrete element."""

    found: bool
    element_id: Optional[str] = None  # Page-local element handle
    confidence: float = 0.0
    method: str = "none"  # selector | label | aria_label | placeholder | text | role_fuzzy

    @property
    def ok(self) -> bool:
        return self.found and self.confidence >= GROUNDING_THRESHOLD


# Below this confidence we refuse to act blindly and route to the repair loop.
GROUNDING_THRESHOLD = 0.5


@dataclass
class FailureRecord:
    """Why a step failed - the unit the repair loop reasons over."""

    step_id: str
    failure_type: str  # target_not_found | low_grounding_confidence | verification_failed | ...
    attempt: int
    detail: str = ""
    grounding_confidence: float = 0.0
    previous_target: Dict[str, Any] = field(default_factory=dict)
    page_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "failure_type": self.failure_type,
            "attempt": self.attempt,
            "detail": self.detail,
            "grounding_confidence": round(self.grounding_confidence, 3),
            "previous_target": self.previous_target,
        }


@dataclass
class RepairPatch:
    """A minimal, schema-validated edit proposed in response to a failure."""

    op: str  # replace | insert | delete
    step_id: str
    target: Optional[Dict[str, Any]] = None
    reason: str = ""

    def signature(self) -> str:
        """Stable identity used by the oscillation guard."""

        return f"{self.op}:{self.step_id}:{sorted((self.target or {}).items())}"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"op": self.op, "step_id": self.step_id, "reason": self.reason}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass
class StepResult:
    step_id: str
    action: str
    status: str  # ok | repaired | failed | halted
    attempts: int = 1
    grounding: Optional[GroundingResult] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "status": self.status,
            "attempts": self.attempts,
            "grounding_method": self.grounding.method if self.grounding else None,
            "detail": self.detail,
        }


@dataclass
class RunResult:
    workflow_id: str
    status: str  # completed | halted_for_confirmation | failed
    steps: List[StepResult] = field(default_factory=list)
    failures: List[FailureRecord] = field(default_factory=list)
    patches: List[RepairPatch] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return bool(self.patches)

    @property
    def converged(self) -> bool:
        """A run is converged when it finished with no repairs needed."""

        return self.status != "failed" and not self.patches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "converged": self.converged,
            "steps": [s.to_dict() for s in self.steps],
            "failures": [f.to_dict() for f in self.failures],
            "patches": [p.to_dict() for p in self.patches],
        }
