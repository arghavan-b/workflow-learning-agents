"""Workflow memory store: persist and recall induced skills as YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import List

from demo2skill.workflow.schema import WorkflowSkill
from demo2skill.workflow.validator import assert_valid


class WorkflowStore:
    """A directory of validated ``{workflow_id}.yaml`` skill files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, workflow_id: str) -> Path:
        return self.root / f"{workflow_id}.yaml"

    def save(self, skill: WorkflowSkill, *, validate: bool = True) -> Path:
        if validate:
            assert_valid(skill)
        path = self.path_for(skill.workflow_id)
        path.write_text(skill.to_yaml(), encoding="utf-8")
        return path

    def load(self, workflow_id: str) -> WorkflowSkill:
        return WorkflowSkill.from_yaml(self.path_for(workflow_id).read_text(encoding="utf-8"))

    def list_ids(self) -> List[str]:
        return sorted(p.stem for p in self.root.glob("*.yaml"))
