"""``demo2skill-run``: execute an induced skill against an HTML page snapshot.

    demo2skill-run workflow.yaml --page page.html --inputs inputs.json

Prints a run report (steps, grounding methods, any self-healing repairs) and,
when repairs were needed, writes the repaired skill next to the original so the
more robust version is the one that gets reused next time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from demo2skill.executor.executor import run_workflow
from demo2skill.executor.page import Page
from demo2skill.workflow.schema import WorkflowSkill


def _report(result, repaired_path: Optional[Path]) -> str:
    lines = [f"workflow: {result.workflow_id}", f"status:   {result.status}",
             f"converged: {result.converged}", "", "steps:"]
    for step in result.steps:
        d = step.to_dict()
        method = f" via {d['grounding_method']}" if d["grounding_method"] else ""
        lines.append(f"  [{d['status']:>8}] {d['step_id']} ({d['action']}){method}")
    if result.patches:
        lines.append("")
        lines.append("self-healing repairs:")
        for patch in result.patches:
            lines.append(f"  - {patch.step_id}: {patch.reason}")
    if repaired_path:
        lines.append("")
        lines.append(f"repaired skill written to {repaired_path}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute an induced workflow skill.")
    parser.add_argument("workflow", help="Path to the induced workflow YAML.")
    parser.add_argument("--page", required=True, help="HTML snapshot of the page to run against.")
    parser.add_argument("--inputs", help="JSON file of input values.")
    parser.add_argument("--url", default="", help="URL to associate with the page.")
    parser.add_argument("--yes", action="store_true", help="Auto-approve confirmation gates.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    workflow = WorkflowSkill.from_yaml(Path(args.workflow).read_text(encoding="utf-8"))
    page = Page.from_file(args.page, url=args.url)
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8")) if args.inputs else {}

    result, repaired = run_workflow(workflow, page, inputs, auto_confirm=args.yes)

    repaired_path = None
    if result.repaired and result.status != "failed":
        repaired_path = Path(args.workflow).with_name(f"{workflow.workflow_id}.repaired.yaml")
        repaired_path.write_text(repaired.to_yaml(), encoding="utf-8")

    print(_report(result, repaired_path))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
