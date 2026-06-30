"""Abstract typed constants into reusable variables (Module 4).

Each ``fill_field`` / ``upload_file`` value a human typed is generalized into a
named workflow input (``"Implement Video Understanding"`` -> ``${title}``).
The deterministic baseline derives names from the field label; an LLM path can
override this with better intent-aware names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from demo2skill.induction.segmenter import CleanEvent

# Common field labels mapped to conventional variable names. Small and
# overridable; the point is readable names, not exhaustive coverage.
LABEL_ALIASES = {
    "markdown value": "body",
    "leave a comment": "body",
    "add a title": "title",
    "description": "description",
    "merchant": "merchant",
    "amount": "amount",
    "email": "email",
}

# Filler words stripped before slugifying an unmatched label.
FILLER_WORDS = {"add", "a", "an", "the", "your", "enter", "value", "markdown",
                "please", "field", "input", "type", "here"}

NUMERIC_RE = re.compile(r"^\$?-?\d[\d,]*(\.\d+)?$")
BOOLEAN_VALUES = {"true", "false", "yes", "no", "on", "off"}


@dataclass
class AbstractedField:
    name: str
    type: str
    value: str
    label: Optional[str]
    event: CleanEvent


def _infer_type(action: str, value: Optional[str]) -> str:
    if action == "upload_file":
        return "file"
    text = (value or "").strip()
    if text.lower() in BOOLEAN_VALUES:
        return "boolean"
    if NUMERIC_RE.match(text):
        return "number"
    return "string"


def _base_name(event: CleanEvent) -> str:
    label = (event.target.get("label") or event.target.get("text")
             or event.target.get("placeholder") or event.target.get("name") or "")
    lowered = label.strip().lower()
    for needle, alias in LABEL_ALIASES.items():
        if needle in lowered:
            return alias
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", lowered) if w not in FILLER_WORDS]
    return "_".join(words[:3]) or "field"


def abstract_variables(events: List[CleanEvent]) -> List[AbstractedField]:
    """Return one :class:`AbstractedField` per value-bearing fill/upload event."""

    fields: List[AbstractedField] = []
    used: dict = {}
    for event in events:
        if event.action not in ("fill_field", "upload_file"):
            continue
        if event.value in (None, ""):
            continue
        base = _base_name(event)
        count = used.get(base, 0) + 1
        used[base] = count
        name = base if count == 1 else f"{base}_{count}"
        fields.append(
            AbstractedField(
                name=name,
                type=_infer_type(event.action, str(event.value)),
                value=str(event.value),
                label=event.target.get("label") or event.target.get("text"),
                event=event,
            )
        )
    return fields
