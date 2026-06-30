"""Normalize raw browser recorder traces into semantic Demo2Skill events."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from demo2skill.trace.schema import (
    SEMANTIC_TRACE_SCHEMA_VERSION,
    TARGET_KEYS,
    drop_empty,
    event_artifacts,
    event_page_context,
)


VARIABLE_LIKE_INPUT_TYPES = {"text", "email", "number", "password", "search", "tel", "url"}
CHECKABLE_INPUT_TYPES = {"checkbox", "radio"}


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def raw_element(raw_event: Mapping[str, Any]) -> Mapping[str, Any]:
    element = raw_event.get("element")
    return element if isinstance(element, Mapping) else {}


def infer_role(raw_event: Mapping[str, Any], element: Mapping[str, Any]) -> Optional[str]:
    role = normalize_text(element.get("role"))
    if role:
        return role

    tag = normalize_text(element.get("tag"))
    input_type = normalize_text(element.get("type"))
    if tag == "textarea":
        return "textbox"
    if tag == "select":
        return "combobox"
    if tag == "a":
        return "link"
    if tag == "button" or input_type in {"button", "submit"}:
        return "button"
    if tag == "input":
        if input_type in CHECKABLE_INPUT_TYPES:
            return input_type
        return "textbox"
    if raw_event.get("target_text"):
        return "text"
    return None


def infer_label(raw_event: Mapping[str, Any], element: Mapping[str, Any]) -> Optional[str]:
    for key in ("target_label", "label", "aria_label", "placeholder", "name", "id"):
        value = normalize_text(raw_event.get(key))
        if value:
            return value
    for key in ("label", "aria_label", "placeholder", "name", "id"):
        value = normalize_text(element.get(key))
        if value:
            return value
    return None


def infer_text(raw_event: Mapping[str, Any], element: Mapping[str, Any]) -> Optional[str]:
    return normalize_text(raw_event.get("target_text")) or normalize_text(element.get("text"))


def infer_value(raw_event: Mapping[str, Any], element: Mapping[str, Any]) -> Any:
    for key in ("typed_text", "keyboard_text", "value"):
        if raw_event.get(key) not in (None, ""):
            return raw_event.get(key)
    if element.get("value") not in (None, ""):
        return element.get("value")
    return None


def build_target(raw_event: Mapping[str, Any]) -> Dict[str, Any]:
    element = raw_element(raw_event)
    target = {
        "selector": normalize_text(raw_event.get("selector")) or normalize_text(element.get("selector")),
        "label": infer_label(raw_event, element),
        "text": infer_text(raw_event, element),
        "role": infer_role(raw_event, element),
        "aria_label": normalize_text(element.get("aria_label")),
        "placeholder": normalize_text(element.get("placeholder")),
        "nearby_text": normalize_text(raw_event.get("nearby_text"))
        or normalize_text(element.get("nearby_text")),
        "name": normalize_text(element.get("name")),
        "id": normalize_text(element.get("id")),
    }
    return drop_empty({key: target.get(key) for key in TARGET_KEYS})


def infer_semantic_action(raw_event: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    action = normalize_text(raw_event.get("action_type"))
    action = action.lower() if action else "unknown"
    element = raw_element(raw_event)
    tag = normalize_text(element.get("tag"))
    input_type = normalize_text(element.get("type"))
    role = normalize_text(target.get("role"))

    if action in {"navigation", "navigate"}:
        return "navigate"
    if input_type == "file":
        return "upload_file"
    if action in {"type", "input"}:
        return "fill_field"
    if action == "change":
        if role == "combobox" or tag == "select":
            return "select_option"
        if input_type in CHECKABLE_INPUT_TYPES:
            return "set_checked"
        return "set_value"
    if action == "click":
        return "click"
    return action


def normalize_event(raw_event: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert one raw recorder event into a semantic event."""

    target = build_target(raw_event)
    semantic_action = infer_semantic_action(raw_event, target)
    value = infer_value(raw_event, raw_element(raw_event))

    normalized: Dict[str, Any] = {
        "event_id": raw_event.get("event_id"),
        "source_event_id": raw_event.get("event_id"),
        "timestamp": raw_event.get("timestamp"),
        "semantic_action": semantic_action,
        "target": target,
        "value": value,
        "page_context": event_page_context(raw_event),
        "url": raw_event.get("url"),
        "page_title": raw_event.get("page_title"),
        "artifacts": event_artifacts(raw_event),
    }

    selected_text = normalize_text(raw_event.get("selected_text"))
    if selected_text:
        normalized["selected_text"] = selected_text

    if semantic_action == "navigate":
        normalized["target"] = drop_empty({"url": raw_event.get("url")})
    if semantic_action == "upload_file" and not target.get("semantic"):
        normalized["target"] = drop_empty({**target, "semantic": "file_upload"})

    return drop_empty(normalized)


def is_duplicate_event(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Remove noisy exact duplicates while preserving intentional repeated actions."""

    duplicate_keys = ("semantic_action", "target", "value", "url")
    if any(previous.get(key) != current.get(key) for key in duplicate_keys):
        return False
    if current.get("semantic_action") in {"click", "navigate", "fill_field", "set_value"}:
        return True
    return False


def normalize_events(raw_events: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for raw_event in raw_events:
        normalized = normalize_event(raw_event)
        if events and is_duplicate_event(events[-1], normalized):
            events[-1] = normalized
            continue
        events.append(normalized)
    return events


def normalize_trace(raw_trace: Mapping[str, Any]) -> Dict[str, Any]:
    raw_events = raw_trace.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError("Raw trace must contain an events list.")

    return {
        "schema_version": SEMANTIC_TRACE_SCHEMA_VERSION,
        "source_schema_version": raw_trace.get("schema_version"),
        "created_at": raw_trace.get("created_at"),
        "metadata": dict(raw_trace.get("metadata") or {}),
        "events": normalize_events(raw_events),
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_output_path(input_path: Path) -> Path:
    if input_path.name == "trace.json":
        return input_path.with_name("semantic_trace.json")
    return input_path.with_suffix(".semantic.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a raw Demo2Skill trace.")
    parser.add_argument("trace", help="Path to raw recorder trace.json.")
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for semantic trace JSON. Defaults beside the raw trace.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.trace).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(input_path)

    semantic_trace = normalize_trace(load_json(input_path))
    write_json(output_path, semantic_trace)
    print(f"Saved semantic trace to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
