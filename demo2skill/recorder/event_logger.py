"""Trace event persistence for Demo2Skill recorder runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second-level readability."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLogger:
    """Collect raw browser events and write a stable `trace.json` artifact."""

    schema_version = "demo2skill.raw_trace.v0"

    def __init__(self, output_dir: Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "trace.json"
        self.created_at = utc_now_iso()
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.events: List[Dict[str, Any]] = []
        self._counter = 0

    def next_event_id(self) -> str:
        self._counter += 1
        return f"evt_{self._counter:06d}"

    def append(
        self,
        raw_event: Mapping[str, Any],
        *,
        event_id: Optional[str] = None,
        artifacts: Optional[Mapping[str, Optional[str]]] = None,
        page_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append one event, enriching it with IDs, timestamps, and artifacts."""

        event: Dict[str, Any] = dict(raw_event)
        event["event_id"] = event_id or event.get("event_id") or self.next_event_id()
        event.setdefault("timestamp", utc_now_iso())

        if page_context:
            for key, value in page_context.items():
                event.setdefault(key, value)

        if artifacts:
            for key, value in artifacts.items():
                if value is not None:
                    event[key] = value

        self.events.append(event)
        return event

    def extend(self, raw_events: Iterable[Mapping[str, Any]]) -> None:
        for raw_event in raw_events:
            self.append(raw_event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "events": self.events,
        }

    def save(self) -> Path:
        self.trace_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self.trace_path


def compact_element_info(element: Optional[MutableMapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop empty element fields while preserving recorder-friendly keys."""

    if not element:
        return None
    return {key: value for key, value in element.items() if value not in (None, "", [], {})}

