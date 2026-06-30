"""Bridge from executed skills to model-training data.

Demo2Skill's executor + verifier + repair loop produces something the
video-mining pipelines of VideoAgentTrek / Video2GUI lack: a *verified* record
of which action actually worked on which element. This package turns a
:class:`~demo2skill.executor.models.RunResult` into a stream of training-ready
``(observation, instruction, action)`` steps -- the
``trajectory.jsonl`` format a Paper-1/2-style continued-pretraining / SFT run
consumes -- keeping only steps the run confirmed.

This is the keystone of the "Demo2Skill as a clean trajectory generator" bridge:
the same induction that yields an editable skill also yields filtered
supervision for a policy model.
"""

from __future__ import annotations

from demo2skill.export.trajectory import (
    ExportedTrajectory,
    StepObservation,
    TrainingStep,
    export_trajectory,
)

__all__ = [
    "ExportedTrajectory",
    "StepObservation",
    "TrainingStep",
    "export_trajectory",
]
