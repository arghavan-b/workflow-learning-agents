"""Induce a WorkflowSkill from a normalized trace (Modules 3-4 orchestrator).

Default path is fully deterministic (no API key). Pass an
:class:`~demo2skill.induction.llm.LLMClient` to use the prompt-based path, which
falls back to the baseline if the model returns invalid YAML.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from demo2skill.induction.llm import LLMClient, default_client
from demo2skill.induction.prompts import (
    INDUCE_SYSTEM,
    SEGMENT_SYSTEM,
    induce_prompt,
    segment_prompt,
)
from demo2skill.induction.segmenter import (
    CleanEvent,
    clean_events,
    segment_events,
)
from demo2skill.induction.variable_abstraction import AbstractedField, abstract_variables
from demo2skill.workflow.schema import WorkflowInput, WorkflowSkill, WorkflowStep
from demo2skill.workflow.validator import validate_skill

TARGET_FIELDS = ("text", "label", "role", "selector", "aria_label",
                 "placeholder", "nearby_text", "semantic")

DEFAULT_RECOVERY = [
    {"condition": "target_not_found", "strategy": "semantic_search_then_retry"},
    {"condition": "missing_input", "strategy": "ask_user"},
    {"condition": "verification_failed", "strategy": "stop_and_report"},
]


def _slug(text: str, max_words: int = 4) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return "_".join(words[:max_words]) or "step"


def _build_target(target: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    built = {k: target[k] for k in TARGET_FIELDS if target.get(k)}
    return built or None


def _unique(step_id: str, used: set) -> str:
    candidate, n = step_id, 1
    while candidate in used:
        n += 1
        candidate = f"{step_id}_{n}"
    used.add(candidate)
    return candidate


def _infer_goal_and_id(events: List[CleanEvent]) -> Dict[str, str]:
    urls = " ".join(e.url or "" for e in events)
    titles = " ".join(e.page_title or "" for e in events)
    if "issues/new" in urls or "new issue" in titles.lower():
        return {"workflow_id": "create_github_issue_v1", "goal": "Create a GitHub issue"}
    last_title = next((e.page_title for e in reversed(events) if e.page_title), None)
    domain = ""
    match = re.search(r"https?://([^/]+)", urls)
    if match:
        domain = match.group(1)
    label = last_title or domain or "browser task"
    return {"workflow_id": f"{_slug(label)}_v1", "goal": f"Complete the {label} workflow"}


def generate_workflow_baseline(semantic_trace: Mapping[str, Any]) -> WorkflowSkill:
    """Rule-based induction that needs no LLM."""

    raw_events = list(semantic_trace.get("events", []))
    result = clean_events(raw_events)
    cleaned = result.events
    fields = abstract_variables(cleaned)
    field_by_event = {id(f.event): f for f in fields}

    meta = _infer_goal_and_id(cleaned)
    used_ids: set = set()
    steps: List[Dict[str, Any]] = []

    for event in cleaned:
        if event.action == "navigate":
            steps.append({
                "step_id": _unique(_slug(_nav_name(event.url)), used_ids),
                "action": "navigate",
                "url": event.url,
            })
        elif event.action == "click":
            target = _build_target(event.target)
            if not target:
                continue
            label = target.get("text") or target.get("label") or "element"
            steps.append({
                "step_id": _unique(f"click_{_slug(label)}", used_ids),
                "action": "click",
                "target": target,
            })
        elif event.action in ("fill_field", "upload_file"):
            field = field_by_event.get(id(event))
            target = _build_target(event.target)
            if field is None or not target:
                continue
            steps.append({
                "step_id": _unique(f"fill_{field.name}", used_ids),
                "action": event.action,
                "target": target,
                "value": f"${{{field.name}}}",
            })

    steps.extend(_safety_steps(fields, used_ids))

    skill_dict = {
        "workflow_id": meta["workflow_id"],
        "goal": meta["goal"],
        "inputs": _inputs_from_fields(fields),
        "preconditions": result.preconditions,
        "steps": steps,
        "recovery": DEFAULT_RECOVERY,
    }
    return WorkflowSkill.from_dict(skill_dict)


def _nav_name(url: Optional[str]) -> str:
    path = re.sub(r"^https?://[^/]+", "", url or "").strip("/")
    return f"navigate_{path}" if path else "navigate"


def _inputs_from_fields(fields: List[AbstractedField]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        seen.setdefault(f.name, {"name": f.name, "type": f.type})
    return list(seen.values())


def _safety_steps(fields: List[AbstractedField], used_ids: set) -> List[Dict[str, Any]]:
    """Append a verify step over filled fields and a confirmation gate."""

    steps: List[Dict[str, Any]] = []
    checks = []
    for f in fields:
        if f.type == "file":
            continue
        if f.label:
            checks.append({"field_equals": {"label": f.label, "value": f"${{{f.name}}}"}})
    if checks:
        steps.append({
            "step_id": _unique("verify_form", used_ids),
            "action": "verify",
            "checks": checks,
        })
    steps.append({
        "step_id": _unique("confirm_before_submit", used_ids),
        "action": "request_user_confirmation",
        "reason": "Submitting is irreversible; confirm the filled values before final submit.",
    })
    return steps


# -- LLM path --------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def generate_workflow_llm(
    semantic_trace: Mapping[str, Any], llm: LLMClient
) -> WorkflowSkill:
    """LLM induction: segment then induce YAML. Raises if output is unusable."""

    result = clean_events(list(semantic_trace.get("events", [])))
    segments = segment_events(result.events)
    seg_payload = json.dumps([s.to_dict() for s in segments], indent=2)
    # Segmentation prompt is available for richer pipelines; the induce prompt
    # alone is enough to produce the skill here.
    _ = llm.complete(system=SEGMENT_SYSTEM, prompt=segment_prompt(seg_payload))
    yaml_text = llm.complete(system=INDUCE_SYSTEM, prompt=induce_prompt(seg_payload))
    return WorkflowSkill.from_yaml(_strip_code_fences(yaml_text))


def induce_workflow(
    semantic_trace: Mapping[str, Any], *, llm: Optional[LLMClient] = None
) -> WorkflowSkill:
    """Induce a workflow skill, using the LLM when available and valid."""

    if llm is not None:
        try:
            skill = generate_workflow_llm(semantic_trace, llm)
            if not [i for i in validate_skill(skill) if i.severity == "error"]:
                return skill
        except Exception as exc:  # noqa: BLE001 - any LLM/parse failure -> baseline
            print(f"LLM induction failed ({exc}); using deterministic baseline.",
                  file=sys.stderr)
    return generate_workflow_baseline(semantic_trace)


# -- CLI -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Induce a workflow skill from a semantic trace.")
    parser.add_argument("trace", help="Path to semantic_trace.json.")
    parser.add_argument("--output", "-o", help="Output YAML path. Defaults beside the trace.")
    parser.add_argument("--store", help="Optional workflow store directory to save into.")
    parser.add_argument("--llm", action="store_true",
                        help="Use the Anthropic LLM path (requires ANTHROPIC_API_KEY).")
    return parser


def _default_output(trace_path: Path) -> Path:
    return trace_path.with_name("induced_workflow.yaml")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    trace_path = Path(args.trace).expanduser().resolve()
    semantic_trace = json.loads(trace_path.read_text(encoding="utf-8"))

    llm = default_client() if args.llm else None
    skill = induce_workflow(semantic_trace, llm=llm)

    issues = validate_skill(skill)
    for issue in issues:
        print(str(issue), file=sys.stderr)
    if any(i.severity == "error" for i in issues):
        print("Induced workflow has validation errors (see above).", file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output(trace_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(skill.to_yaml(), encoding="utf-8")
    print(f"Saved workflow skill to {output_path}")

    if args.store:
        from demo2skill.workflow.store import WorkflowStore

        stored = WorkflowStore(Path(args.store)).save(skill)
        print(f"Stored workflow in {stored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
