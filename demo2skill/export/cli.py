"""``demo2skill-export``: run an induced skill and emit training trajectories.

    demo2skill-export workflow.yaml --page page.html --inputs inputs.json \
        -o runs/issue/trajectory.jsonl

Executes the skill against a page snapshot (reusing the executor + repair loop),
then writes the *verified* steps as ``trajectory.jsonl`` plus a ``.meta.json``
sidecar. By default the run halts at the confirmation gate - which is a good,
exportable episode - so no irreversible action is taken to produce training
data. Pass ``--yes`` to drive the run to completion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from demo2skill.executor.executor import run_workflow
from demo2skill.executor.page import Page
from demo2skill.export.trajectory import export_trajectory
from demo2skill.workflow.schema import WorkflowSkill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export verified training trajectories from a workflow run."
    )
    parser.add_argument("workflow", help="Path to the induced workflow YAML.")
    parser.add_argument("--page", required=True, help="HTML snapshot to run against.")
    parser.add_argument("--inputs", help="JSON file of input values.")
    parser.add_argument("--url", default="", help="URL to associate with the page.")
    parser.add_argument("--yes", action="store_true",
                        help="Auto-approve confirmation gates (drive run to completion).")
    parser.add_argument("--source-modality", default="recorder",
                        help="Provenance tag for the demo source (recorder | video).")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to write trajectory.jsonl (a .meta.json is written alongside).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    workflow = WorkflowSkill.from_yaml(Path(args.workflow).read_text(encoding="utf-8"))
    page = Page.from_file(args.page, url=args.url)
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8")) if args.inputs else {}

    result, final_workflow = run_workflow(workflow, page, inputs, auto_confirm=args.yes)

    trajectory = export_trajectory(
        final_workflow, result, inputs, source_modality=args.source_modality
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(trajectory.to_jsonl() + "\n" if trajectory.steps else "", encoding="utf-8")
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(trajectory.to_meta(), indent=2) + "\n", encoding="utf-8")

    print(
        f"episode {trajectory.episode_id}: status={trajectory.episode_status} "
        f"verified={trajectory.verified} -> {trajectory.kept} training step(s)\n"
        f"  {out}\n  {meta_path}"
    )
    # Exit non-zero only if the run failed outright (no usable supervision).
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
