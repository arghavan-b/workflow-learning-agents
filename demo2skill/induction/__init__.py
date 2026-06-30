"""Workflow induction: segment traces and induce reusable skills (Modules 3-4).

The pipeline is *pluggable*: every stage has a deterministic, dependency-free
baseline that runs with no API key, and an optional LLM path behind the
:class:`~demo2skill.induction.llm.LLMClient` interface.
"""

from demo2skill.induction.segmenter import Segment, clean_events, segment_events
from demo2skill.induction.variable_abstraction import (
    AbstractedField,
    abstract_variables,
)
from demo2skill.induction.workflow_generator import induce_workflow

__all__ = [
    "Segment",
    "clean_events",
    "segment_events",
    "AbstractedField",
    "abstract_variables",
    "induce_workflow",
]
