"""Turn a verified workflow run into model-training trajectories.

The contract is deliberately narrow and additive: nothing in the executor or
schema changes. The exporter reconstructs each step's supervision from three
things the run already produces -

* the **final (repaired) workflow** returned by the executor - so the target a
  step trains on is the *corrected* one, not the brittle demo locator;
* the **inputs** - so ``${title}`` becomes the concrete typed string a policy
  must learn to emit;
* the **RunResult** - whose per-step ``status`` (``ok`` / ``repaired``) is the
  verification signal: a step only reaches that status if grounding succeeded
  and any postcondition/verify passed. Failed or halted steps are dropped.

Optional :class:`StepObservation` enrichment (screenshot path, click
coordinate, page text) is supplied by the execution substrate - video frames or
recorder screenshots - and is absent for the dependency-free HTML page model.

Output is one JSON object per step (JSONL), each self-contained with an
``episode_id`` and provenance, which is the standard shape for an SFT shard.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from demo2skill.executor.executor import bind_step
from demo2skill.executor.models import RunResult
from demo2skill.workflow.schema import WorkflowSkill, WorkflowStep

SCHEMA_VERSION = "demo2skill.trajectory.v0"

# A step's RunResult status that means "the run confirmed this action".
VERIFIED_STATUSES = {"ok", "repaired"}

# Steps that correspond to an observable GUI action worth training on. Control
# steps (verify / request_user_confirmation / stop) are not emitted as training
# steps; a passing ``verify`` instead marks the whole episode as verified.
OBSERVABLE_ACTIONS = {"navigate", "click", "fill_field", "upload_file", "extract_text"}

# Map Demo2Skill's action vocabulary onto the GUI action space the video-mining
# papers train on (pyautogui-style: click / type / navigate ...).
_ACTION_ALIAS = {
    "navigate": "navigate",
    "click": "click",
    "fill_field": "type",
    "upload_file": "upload_file",
    "extract_text": "extract_text",
}

# Episode statuses that count as good supervision. Halting at the confirmation
# gate is the *intended* terminal state for an irreversible task, so it is a
# success for training purposes just like a fully completed run.
_GOOD_EPISODE = {"completed", "halted_for_confirmation"}


@dataclass
class StepObservation:
    """Optional per-step observation supplied by the execution substrate."""

    screenshot: Optional[str] = None      # path to the pre-action frame
    url: Optional[str] = None
    page_text: Optional[str] = None
    coordinate: Optional[List[int]] = None  # [x, y] click point, if known
    viewport: Optional[List[int]] = None    # [width, height]

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "screenshot": self.screenshot,
            "url": self.url,
            "page_text": self.page_text,
            "coordinate": self.coordinate,
            "viewport": self.viewport,
        }
        return {k: v for k, v in out.items() if v is not None}


@dataclass
class TrainingStep:
    """One ``(observation, instruction, action)`` example for policy training."""

    episode_id: str
    step_index: int
    instruction: str          # the overall task goal
    subgoal: str              # natural-language description of this step
    observation: Dict[str, Any]
    action: Dict[str, Any]
    verified: bool
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "instruction": self.instruction,
            "subgoal": self.subgoal,
            "observation": self.observation,
            "action": self.action,
            "verified": self.verified,
            "provenance": self.provenance,
        }


@dataclass
class ExportedTrajectory:
    """A verified episode plus its emitted training steps."""

    episode_id: str
    workflow_id: str
    goal: str
    episode_status: str
    verified: bool
    source_modality: str
    steps: List[TrainingStep] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return len(self.steps)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "workflow_id": self.workflow_id,
            "goal": self.goal,
            "episode_status": self.episode_status,
            "verified": self.verified,
            "source_modality": self.source_modality,
            "num_steps": self.kept,
        }

    def to_jsonl(self) -> str:
        """One JSON object per line - the SFT shard format."""

        return "\n".join(json.dumps(s.to_dict(), ensure_ascii=False) for s in self.steps)


def _subgoal(step: WorkflowStep, value: Optional[str]) -> str:
    """A short natural-language instruction for a single step."""

    t = step.target
    name = None
    if t is not None:
        name = t.label or t.text or t.aria_label or t.placeholder or t.role
    if step.action == "navigate":
        return f"Navigate to {step.url or (t.url if t else '')}".strip()
    if step.action == "click":
        return f"Click the {name or 'element'}"
    if step.action == "fill_field":
        where = f"the '{name}' field" if name else "the field"
        return f"Type \"{value}\" into {where}" if value else f"Fill {where}"
    if step.action == "upload_file":
        return f"Upload a file to the {name or 'field'}"
    if step.action == "extract_text":
        return f"Read the text of the {name or 'element'}"
    return step.action


def _action_payload(
    step: WorkflowStep,
    value: Optional[str],
    grounding_method: Optional[str],
    grounding_confidence: float,
    observation: StepObservation,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": _ACTION_ALIAS.get(step.action, step.action)}

    if step.action == "navigate":
        payload["url"] = step.url or (step.target.url if step.target else None)
        return _drop_none(payload)

    if step.target is not None:
        payload["target"] = step.target.model_dump(exclude_none=True)
    if step.action in ("fill_field", "upload_file") and value is not None:
        payload["text"] = value
    if observation.coordinate is not None:
        payload["coordinate"] = observation.coordinate
    # Which identifier actually located the element at run time, and how sure.
    payload["grounding"] = _drop_none({
        "method": grounding_method,
        "confidence": round(grounding_confidence, 3) if grounding_method else None,
    }) or None
    return _drop_none(payload)


def export_trajectory(
    workflow: WorkflowSkill,
    result: RunResult,
    inputs: Optional[Dict[str, Any]] = None,
    *,
    episode_id: Optional[str] = None,
    source_modality: str = "recorder",
    task_instruction: Optional[str] = None,
    observations: Optional[Dict[str, StepObservation]] = None,
) -> ExportedTrajectory:
    """Build a verified training trajectory from one workflow run.

    ``workflow`` must be the **final** workflow the executor returned (the
    repaired one when repairs happened), so corrected targets are what gets
    exported. ``inputs`` are bound into each step to recover concrete values.
    ``observations`` is an optional ``step_id -> StepObservation`` map the
    substrate fills in with screenshots / click coordinates.
    """

    inputs = inputs or {}
    observations = observations or {}
    episode_id = episode_id or workflow.workflow_id
    instruction = task_instruction or workflow.goal

    episode_verified = result.status in _GOOD_EPISODE
    by_id = {s.step_id: s for s in workflow.steps}

    steps: List[TrainingStep] = []
    current_url: Optional[str] = None
    step_index = 0

    for sr in result.steps:
        step = by_id.get(sr.step_id)
        if step is None:
            continue

        bound, _missing = bind_step(step, inputs)
        value = bound.value

        # Track the page URL forward so non-navigate steps carry an observation
        # even when the substrate gives no explicit per-step url.
        if step.action == "navigate":
            current_url = bound.url or (bound.target.url if bound.target else current_url)

        # A verify step is not itself a training step. Its success is already
        # reflected in the run's terminal status (a failed verify propagates to
        # ``failed``), so episode-level verification stays status-derived.
        if step.action not in OBSERVABLE_ACTIONS:
            continue
        if sr.status not in VERIFIED_STATUSES:
            # Only confirmed actions become supervision. This is the whole point.
            continue

        obs = observations.get(sr.step_id) or StepObservation()
        if obs.url is None:
            obs.url = current_url

        grounding_method = sr.grounding.method if sr.grounding else None
        grounding_conf = sr.grounding.confidence if sr.grounding else 0.0

        steps.append(
            TrainingStep(
                episode_id=episode_id,
                step_index=step_index,
                instruction=instruction,
                subgoal=_subgoal(bound, value),
                observation=obs.to_dict(),
                action=_action_payload(bound, value, grounding_method, grounding_conf, obs),
                verified=sr.status in VERIFIED_STATUSES and episode_verified,
                provenance={
                    "workflow_id": workflow.workflow_id,
                    "step_id": sr.step_id,
                    "source_modality": source_modality,
                    "repaired": sr.status == "repaired",
                    "attempts": sr.attempts,
                    "episode_status": result.status,
                },
            )
        )
        step_index += 1

    for ts in steps:
        ts.provenance["episode_verified"] = episode_verified

    return ExportedTrajectory(
        episode_id=episode_id,
        workflow_id=workflow.workflow_id,
        goal=workflow.goal,
        episode_status=result.status,
        verified=episode_verified,
        source_modality=source_modality,
        steps=steps,
    )


def _drop_none(mapping: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in mapping.items() if v not in (None, "", [], {})}
