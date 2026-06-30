"""Schema helpers for raw-to-semantic Demo2Skill traces."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


SEMANTIC_TRACE_SCHEMA_VERSION = "demo2skill.semantic_trace.v0"

TARGET_KEYS = (
    "selector",
    "label",
    "text",
    "role",
    "aria_label",
    "placeholder",
    "nearby_text",
    "name",
    "id",
    "semantic",
)

ARTIFACT_KEYS = (
    "screenshot_path",
    "dom_snapshot_path",
    "accessibility_tree_path",
)


def drop_empty(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in mapping.items() if value not in (None, "", [], {})}


def event_page_context(raw_event: Mapping[str, Any]) -> Optional[str]:
    title = raw_event.get("page_title")
    url = raw_event.get("url")
    if title and url:
        return f"{title} ({url})"
    if title:
        return str(title)
    if url:
        return str(url)
    return None


def event_artifacts(raw_event: Mapping[str, Any]) -> Dict[str, Any]:
    return drop_empty({key: raw_event.get(key) for key in ARTIFACT_KEYS})
